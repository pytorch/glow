/**
 * Copyright (c) Glow Contributors. See CONTRIBUTORS file.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "glow/Partitioner/PartitionerBase.h"
#include "glow/Optimizer/GraphOptimizer/GraphOptimizer.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include <fstream>

using namespace glow;
using llvm::isa;

/// Creates and \returns a new DAGNode from \p F given \p mapping.
static std::unique_ptr<DAGNode>
createDAGNodeFromFun(Function *F, NodeToFunctionMap &mapping) {
  std::unique_ptr<DAGNode> DN = glow::make_unique<DAGNode>();
  DN->name = F->getName().str();
  DN->logicalDevices = mapping.getLogicalDeviceIDList(F);
  DN->backendName = mapping.getPartitionBackendName(F);
  DN->size = mapping.getGraphMemInfo(F).getTotalMemSize();
  DN->backendHints = mapping.getBackendHints(F);
  DN->backendSpecificOpts = mapping.getBackendSpecificOpts(F);
  DN->replicationCount = mapping.getReplicationCount(F);
  return DN;
}

// Current only partition the representative function.
DAGListTy PartitionerBase::doPartitioning(
    llvm::StringRef funcName, std::vector<Function *> funcs, Module *module,
    NodeToFunctionMap &mapping, bool saveDAG, BackendSpecificNodeInfo &nodeInfo,
    bool skipCloning) {
  DAGListTy partitions;
  // Add a dummy node to make sure that a DAG has a single entrance.
  DAGNodePtr DAGRoot = glow::make_unique<DAGNode>();
  DAGNodePtrVec nodes;
  DAGRoot->logicalDevices = {0};
  DAGRoot->name = funcName.str();
  DAGRoot->module = module;
  DAGNode *root = DAGRoot.get();

  llvm::DenseMap<Node *, Node *> currToNew;

  if (!skipCloning) {
    // Clone nodes into target partition. Update nodeInfo as necessary.
    for (size_t i = 0, e = funcs.size(); i < e; i++) {
      for (auto &N : funcs[i]->getNodes()) {
        auto *clone = N.clone();
        currToNew[&N] = clone;
        mapping[&N]->addNode(clone);

        // If needed, update NodeInfo to point old Node's info to clone.
        auto itF = nodeInfo.find(funcs[i]);
        if (itF == nodeInfo.end()) {
          continue;
        }
        auto &currNodeInfo = itF->second;
        auto itN = currNodeInfo.find(&N);
        if (itN != currNodeInfo.end()) {
          currNodeInfo[clone] = std::move(itN->second);
          // Erase old NodeInfo; current Nodes will be eliminated later when
          // input funcs will be erased.
          currNodeInfo.erase(itN);
        }
      }
    }
  }

  // For any dependency that crosses a partition, add a placeholder and save
  // node. Record the dependence in the function graph.
  std::unordered_map<NodeValue, Placeholder *> placeholders;
  llvm::DenseMap<Function *, DAGNode *> funcDAG;
  for (auto *subF : mapping.getPartitions()) {
    if (funcDAG.find(subF) == funcDAG.end()) {
      std::unique_ptr<DAGNode> subDAG = createDAGNodeFromFun(subF, mapping);
      funcDAG[subF] = subDAG.get();
      nodes.push_back(std::move(subDAG));
    }

    // Link subF to its parents.
    std::set<Function *> parents;
    for (auto &N : subF->getNodes()) {
      for (int inp = 0, e = N.getNumInputs(); inp < e; inp++) {
        auto input = N.getNthInput(inp);
        // No need to check Constant since it won't be the result of another
        // function.
        if (isa<Constant>(input.getNode())) {
          continue;
        }

        Function *inputF = nullptr;
        // It is possible that one input is the output of anther function.
        if (Placeholder *ph = llvm::dyn_cast<Placeholder>(input.getNode())) {
          for (auto &user : ph->getUsers()) {
            if (auto *save = llvm::dyn_cast<SaveNode>(user.getUser())) {
              placeholders[input] = save->getPlaceholder();
              inputF = mapping[user.getUser()];
              break;
            }
          }
          if (!inputF) {
            continue;
          }
        }

        if (!inputF) {
          inputF = mapping[input.getNode()];
        }
        if (subF == inputF) {
          continue;
        }
        // Check if a DAGNode for subF's parent is created or not. If not,
        // create one.
        if (funcDAG.find(inputF) == funcDAG.end()) {
          std::unique_ptr<DAGNode> subDAG =
              createDAGNodeFromFun(inputF, mapping);
          funcDAG[inputF] = subDAG.get();
          nodes.push_back(std::move(subDAG));
        }

        // subF is a child of inputF, inputF is a parent of subF.
        if (parents.find(inputF) == parents.end()) {
          funcDAG[inputF]->children.push_back(funcDAG[subF]);
          funcDAG[subF]->parents.push_back(funcDAG[inputF]);
          parents.insert(inputF);
        }
        // If we've already created a placeholder for this dependence, use it.
        auto it = placeholders.find(input);
        if (it != placeholders.end()) {
          N.setNthInput(inp, it->second);
          continue;
        }

        // Create a new placeholder to represent this dependence.
        auto *save = inputF->createSave("tmp", input);
        auto *tmp = save->getPlaceholder();
        placeholders[input] = tmp;
        N.setNthInput(inp, tmp);
      }
    }
  }

  if (saveDAG) {
    DAG dag;
    dag.root = std::move(DAGRoot);
    dag.nodes = std::move(nodes);
    partitions.push_back(std::move(dag));
  }

  if (!skipCloning) {
    // Update links between nodes in the cloned functions. Add placeholders (and
    // save nodes) where a link crosses a partition boundary.
    for (auto *subF : mapping.getPartitions()) {
      for (auto &N : subF->getNodes()) {
        for (int inp = 0, e = N.getNumInputs(); inp < e; inp++) {
          auto input = N.getNthInput(inp);
          if (isa<Storage>(input.getNode())) {
            continue;
          }
          // Link this node to the clone of its input.
          auto *clone = currToNew[input.getNode()];
          N.setNthInput(inp, NodeValue(clone, input.getResNo()));
        }
      }
    }
  }

  // For all DAGNode without parents, link them to the root DAG.
  for (auto *subF : mapping.getPartitions()) {
    if (funcDAG[subF]->parents.size() == 0) {
      funcDAG[subF]->parents.push_back(root);
      root->children.push_back(funcDAG[subF]);
    }
  }
  return partitions;
}

void PartitionerBase::dumpDAG(llvm::StringRef dotFilename,
                              const DAGListTy &partitions) const {
  if (partitions.size() == 0) {
    return;
  }
  auto *root = partitions[0].root.get();
  LOG(INFO) << "Writing dotty graph for DAG after graph partitioning: "
            << dotFilename.str();
  std::ofstream myfile;
  myfile.open(dotFilename.str());
  myfile << "digraph DAG {\n\trankdir=TB;\n";
  // Dump DAGNodes
  std::vector<DAGNode *> nodes;
  llvm::SmallSet<DAGNode *, 10> used;
  nodes.push_back(root);
  int cur = 0;
  int num = 1;
  while (cur < num) {
    auto *node = nodes[cur];
    for (size_t i = 0; i < node->children.size(); i++) {
      auto child = node->children[i];
      DescriptionBuilder db(child->name.c_str());
      const std::string &backendName = child->backendName;
      db.addParam("BackendName", backendName);
      myfile << "\"" << escapeDottyString(child->name) << "\""
             << " [ label = \"" << escapeDottyString(db) << "\"";
      myfile << "\tshape = \"record\"\n";
      myfile << "\tstyle=\"filled,rounded\"\n";
      auto colorIdx = llvm::hash_value(backendName);
      myfile << "\tfillcolor=" << getDotFileNodeColor(colorIdx) << "\n";
      myfile << "penwidth = 2];\n";
      if (used.count(child) == 0) {
        nodes.push_back(child);
        used.insert(child);
        num++;
      }
    }
    cur++;
  }

  // Dump edges.
  for (size_t i = 0; i < nodes.size(); i++) {
    auto *node = nodes[i];
    for (size_t j = 0; j < node->children.size(); j++) {
      auto child = node->children[j];
      if (node->name.compare(child->name) == 0) {
        // If a network is too small to be partitioned, the dummy node's name
        // and its child (i.e. the original network) share the same name. The
        // edge will create loop. So in this case, this edge just be ignored.
        continue;
      }
      myfile << "\"" << escapeDottyString(node->name) << "\""
             << " -> "
             << "\"" << escapeDottyString(child->name) << "\""
             << ";";
    }
  }
  myfile << "}";

  myfile.close();
  return;
}
