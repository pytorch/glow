/**
 * Copyright (c) 2017-present, Facebook, Inc.
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

#include "glow/Partitioner/Partitioner.h"
#include "glow/Graph/Context.h"
#include "glow/Graph/Utils.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

using namespace glow;
using llvm::isa;

/// A graph with BFS oder.
struct BFSLevel {
  /// A list of <level, nodelist> with BFS order.
  std::vector<std::pair<int, std::vector<Node *>>> levels;
  /// A set of visited nodes.
  std::unordered_set<const Node *> visited;
};

Partitioner::Partitioner(Module *parent, const std::vector<DeviceInfo> &devices)
    : module_(parent), deviceInfo_(devices) {
  memSize_ = module_->getConstantsSize();
}

Function *Partitioner::selectRepFunc(Module *parent, size_t &memSize) {
  auto funcList = parent->getFunctions();
  Function *ret = nullptr;
  for (Function *F : funcList) {
    size_t size = memSize;

    // The set to keep the placeholders (only for Inputs) whose size is
    // already calculated.
    std::set<llvm::StringRef> pSet;

    for (auto &node : F->getNodes()) {
      int n = node.getNumInputs();
      if (node.getKind() == Kinded::Kind::SaveNodeKind) {
        // Special node, the placeholder should be ignored?
        continue;
      }
      for (int i = 0; i < n; i++) {
        Placeholder *in =
            llvm::dyn_cast<Placeholder>(node.getNthInput(i).getNode());
        if (in && pSet.count(in->getName()) == 0) {
          auto ty = in->getType();
          size += ty->getSizeInBytes();
          pSet.insert(in->getName());
        }
      }
    }
    // Find the function with largest required memory as the representive
    // function.
    if (size > memSize) {
      ret = F;
      memSize = size;
    }
  }
  return ret;
}

/// Get the minimal memory requirement (constant) for each op in the function.
void Partitioner::initOpMemUsage() {
  memUsage_.clear();
  for (auto &node : F_->getNodes()) {
    int n = node.getNumInputs();
    unsigned size = 0;
    if (node.getKind() == Kinded::Kind::SaveNodeKind) {
      memUsage_[&node] = size;
      continue;
    }
    for (int i = 0; i < n; i++) {
      Storage *in = llvm::dyn_cast<Storage>(node.getNthInput(i).getNode());
      if (in) {
        auto ty = in->getType();
        size += ty->getSizeInBytes();
      }
    }
    memUsage_[&node] = size;
  }
}

static BFSLevel getBFSLevel(Function *F) {
  // Visit graph nodes in BFS order. For each non-storage node, get its level.
  // Use the preorder to get the roots. Now assume there is only one output op
  // (i.e. root) now.
  GraphPreOrderVisitor visitor(*F);
  Node *node = nullptr;
  for (auto &N : visitor.getPreOrder()) {
    if (isa<Storage>(N)) {
      continue;
    }
    node = N;
    break;
  }

  BFSLevel bfs;
  int level = 0;
  int current = 0;
  bfs.levels.push_back({level, {node}});
  bfs.visited.insert(node);
  level++;
  while (current < level) {
    std::vector<Node *> nodes;
    for (int i = 0, e = bfs.levels[current].second.size(); i < e; i++) {
      Node *N = bfs.levels[current].second[i];

      for (int j = 0, e = N->getNumInputs(); j < e; ++j) {
        Node *in = N->getNthInput(j).getNode();
        if (isa<Storage>(in) || bfs.visited.count(in)) {
          continue;
        }
        nodes.push_back(in);
        bfs.visited.insert(in);
      }
    }
    if (nodes.size() > 0) {
      auto newPair = std::make_pair(level, nodes);
      bfs.levels.push_back(newPair);
      level++;
    }
    current++;
  }

  return bfs;
}

/// Assign nodes to partitions and return the mapping.
NodeToFunctionMap Partitioner::selectPartitions(Function *F,
                                                unsigned availableMemory) {
  NodeToFunctionMap mapping;
  BFSLevel bfs = getBFSLevel(F);
  unsigned level = bfs.levels.size();
  // A list of cut. The graph can be partitioned by levels (cut[0], level - 1],
  // (cut[1], cut[0] - 1], ..., (-1, cut[n] - 1].
  std::vector<int> cut;

  // Step 1 : get the initial cut based on BFS levels and avaiableMemory.
  // TODO .. need to remove the duplicated memory usage.
  unsigned mem = 0;
  for (int i = level - 1; i >= 0; i--) {
    unsigned tmp = 0;
    for (int j = 0, e = bfs.levels[i].second.size(); j < e; j++) {
      Node *N = bfs.levels[i].second[j];
      tmp += memUsage_[N];
    }
    if (mem + tmp > availableMemory) {
      // mem == 0 means the mem usage for one level exceeds the availableMem,
      // accept it now and will do adjustment later. Otherwise, leave tmp to
      // next stage by assigning it to mem.
      if (mem == 0) {
        cut.push_back(i - 1);
      } else {
        cut.push_back(i);
        mem = tmp;
      }
    } else {
      mem += tmp;
    }
  }

  // The last border.
  cut.push_back(-1);

  // Step 2 : Create the initial mapping between node and functions.
  int color = 0;
  Function *newF;
  for (int k = 0, e = cut.size(); k < e; k++) {
    newF = F->getParent()->createFunction(std::string(F->getName()) + "_part" +
                                          std::to_string(++color));
    mapping.createPartition(newF);
    unsigned mem = 0;
    for (int i = k > 0 ? cut[k - 1] : level - 1; i > cut[k]; i--) {
      for (int j = 0, e1 = bfs.levels[i].second.size(); j < e1; j++) {
        Node *N = bfs.levels[i].second[j];
        if (mem + memUsage_[N] > availableMemory) {
          newF = F->getParent()->createFunction(
              std::string(F->getName()) + "_part" + std::to_string(++color));
          mapping.createPartition(newF);
          mem = memUsage_[N];
        } else {
          mem += memUsage_[N];
        }
        mapping.add(N, newF);
      }
    }
  }
  // Step 3 : adjust the partition based on performance (Advanced Graph
  // Paritioning algrithm will be applied here).
  // --- TODO

  return mapping;
}

/// Adjust the logicalDevice ID for each DAGNode. This happens when \p num (i.e.
/// the number of DAGNodes) is larger than the number of devices. E.g:
/// node1(6GB) -> node2(14GB) -> node3(6GB). The memory limitation is 16GB, and
/// there is only 2 devices.
void Partitioner::adjustLogicalDeviceID(DAGNode *DAG, int num) {}

/// Current only partition the representive function.
void Partitioner::doPartitioning(Function *F, NodeToFunctionMap &mapping) {
  // The dummy node.
  std::unique_ptr<DAGNode> DAG = llvm::make_unique<DAGNode>();
  DAG->logicalDevice = 0;
  DAG->name = F->getName();
  DAG->deviceID = 0;
  DAG->logicalDevice = 0;
  DAGNode *root = DAG.get();
  partitions_.roots.push_back(std::move(DAG));
  llvm::DenseMap<Node *, Node *> currToNew;

  // Clone nodes into target partition.
  for (auto &N : F->getNodes()) {
    auto *clone = N.clone();
    currToNew[&N] = clone;
    mapping[&N]->addNode(clone);
  }

  // For any dependency that crosses a partition, add a placeholder and save
  // node. Record the dependence in the function graph.
  int logicalID = 0;
  llvm::DenseMap<Node *, Placeholder *> placeholders;
  llvm::DenseMap<Function *, DAGNode *> funcDAG;
  for (auto *subF : mapping.getPartitions()) {
    if (funcDAG.find(subF) == funcDAG.end()) {
      std::unique_ptr<DAGNode> subDAG = llvm::make_unique<DAGNode>();
      subDAG->name = subF->getName();
      subDAG->logicalDevice = logicalID++;
      funcDAG[subF] = subDAG.get();
      partitions_.nodes.push_back(std::move(subDAG));
    }

    // Link subF to its parents.
    for (auto &N : subF->getNodes()) {
      for (int inp = 0, e = N.getNumInputs(); inp < e; inp++) {
        auto input = N.getNthInput(inp);
        if (isa<Storage>(input.getNode()))
          continue;

        auto *inputF = mapping[input.getNode()];
        if (subF == inputF)
          continue;

        // Check if a DAGNode for subF's parent is created or not. If not,
        // create one.
        if (funcDAG.find(inputF) == funcDAG.end()) {
          std::unique_ptr<DAGNode> subDAG = llvm::make_unique<DAGNode>();
          subDAG->name = inputF->getName();
          subDAG->logicalDevice = logicalID++;
          funcDAG[inputF] = subDAG.get();
          partitions_.nodes.push_back(std::move(subDAG));
        }

        // subF is a child of inputF, inputF is a parent of subF.
        funcDAG[inputF]->children.push_back(funcDAG[subF]);
        funcDAG[subF]->parents.push_back(funcDAG[inputF]);

        // If we've already created a placeholder for this dependence, use it.
        auto it = placeholders.find(input.getNode());
        if (it != placeholders.end()) {
          N.setNthInput(inp, it->second);
          continue;
        }

        // Create a new placeholder to represent this dependence.
        auto *save = inputF->createSave("tmp", input);
        auto *tmp = save->getPlaceholder();
        placeholders[input.getNode()] = tmp;
        N.setNthInput(inp, tmp);
      }
    }
  }

  // Update links between nodes in the cloned functions. Add placeholders (and
  // save nodes) where a link crosses a partition boundary.
  for (auto *subF : mapping.getPartitions()) {
    for (auto &N : subF->getNodes()) {
      for (int inp = 0, e = N.getNumInputs(); inp < e; inp++) {
        auto input = N.getNthInput(inp);
        if (isa<Storage>(input.getNode()))
          continue;
        // Link this node to the clone of its input.
        auto *clone = currToNew[input.getNode()];
        N.setNthInput(inp, NodeValue(clone, input.getResNo()));
      }
    }
  }

  // For all DAGNode without parents, link them to the root DAG.
  for (auto *subF : mapping.getPartitions()) {
    if (funcDAG[subF]->parents.size() == 0) {
      funcDAG[subF]->parents.push_back(DAG.get());
      root->children.push_back(funcDAG[subF]);
    }
  }

  // Adjust the logicalDevice for each DAGNode.
  if (mapping.getPartitions().size() > deviceInfo_.size()) {
    adjustLogicalDeviceID(DAG.get(), mapping.getPartitions().size());
  }
}

DAGNodeList &Partitioner::Partition() {

  // Find the representive function for running partitioning algrithm.
  F_ = selectRepFunc(module_, memSize_);

  unsigned availMem = deviceInfo_[0].availableMemory;

  if (memSize_ < availMem) {
    // No partition is needed. Create DAGNode and return. This root is alway a
    // dummy function.
    for (auto F : module_->getFunctions()) {
      std::unique_ptr<DAGNode> DAG = llvm::make_unique<DAGNode>();
      DAG->logicalDevice = 0;
      DAG->name = F->getName();
      std::unique_ptr<DAGNode> DAG1 = llvm::make_unique<DAGNode>();
      DAG1->logicalDevice = 0;
      DAG1->name = F->getName();
      DAG1->parents.push_back(DAG.get());
      DAG->children.push_back(DAG1.get());
      partitions_.roots.push_back(std::move(DAG));
      partitions_.nodes.push_back(std::move(DAG1));
    }
    return partitions_;
  }

  // Prepare 1: Get the min memory usage for each op.
  initOpMemUsage();

  // Prepare 2: TODO: get the minimal comunication cost for any 2 ops (i.e. the
  // output data size) Will calculate it on the fly. -- Will double check which
  // way is better.

  // Partition
  // Use BFS to do the initial partitioning. Starting from the final node, BFS
  // until the memory limitation reached one by one.
  NodeToFunctionMap partitionMap = selectPartitions(F_, availMem);

  doPartitioning(F_, partitionMap);

  // Remove the original function after partitioning.
  module_->eraseFunction(F_);

  auto funcList = module_->getFunctions();
  for (Function *F : funcList) {
    (void)F;
    assert(F->verify() && "Conversion led to invalid function");
  }

  // TODO: Optional: if (k < number of devices)
  // Check the computation time of each sub-module, and find out the "key"
  // sub-module to decide if duplicating the sub-module is necessary.

  return partitions_;
}
