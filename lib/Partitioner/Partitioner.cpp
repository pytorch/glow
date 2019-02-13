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

using namespace glow;
using llvm::isa;

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
    size_t size = 0;
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

// Combine the partitions if necessary : if all outside uses of the nodes in
// partition1 is in partition2, and the sum of memory consumption of partition1
// and partition2 is less than availableMemory, combine partition1 and
// partition2.
void Partitioner::partitionsCombine(NodeToFunctionMap &partitions,
                                    FunctionToNodesMapTy &nodesSet,
                                    size_t availableMemory) {

  for (FunctionToNodesMapTy::iterator it = nodesSet.begin();
       it != nodesSet.end(); ++it) {
    std::vector<Node *> outUsers = getOutUsers((*it).second);
    if (outUsers.empty()) {
      continue;
    }

    bool flag = true;
    for (int i = 1, e = outUsers.size(); i < e; i++) {
      if (partitions[outUsers[i]] != partitions[outUsers[i - 1]]) {
        flag = false;
        break;
      }
    }
    if (flag) {
      // This partition only has one successor.
      Function *cur = (*it).first;
      Function *suc = partitions[outUsers[0]];
      NodesSetTy tmp = nodesSet.lookup(suc);
      GraphMemInfo cost1 = partitions.getGraphMemInfo(cur);
      GraphMemInfo cost2 = partitions.getGraphMemInfo(suc);
      if (cost1.constMemSize + cost1.inMemSize + cost2.constMemSize +
              cost2.inMemSize - cost1.outMemSize <
          availableMemory) {
        // We can combine the two partitions to fit one device.
        for (NodesSetTy::iterator it2 = tmp.begin(); it2 != tmp.end(); ++it2) {
          partitions.add(*it2, cur);
        }
        (*it).second.insert(tmp.begin(), tmp.end());
        partitions.deletePartition(suc);
        nodesSet.erase(suc);
        module_->eraseFunction(suc);
      }
    }
  }
}

void Partitioner::partitionsAdjust(NodeToFunctionMap &partitions,
                                   size_t availableMemory) {
  // For each partitioin, create a node set.
  FunctionToNodesMapTy nodesSet;
  for (NodeToFunctionMapTy::iterator it = partitions.begin();
       it != partitions.end(); ++it) {
    nodesSet[(*it).second].insert((*it).first);
  }

  // Initial the memory cost for each partition. Now we use the output size to
  // represent the communication cost.
  for (FunctionToNodesMapTy::iterator it = nodesSet.begin();
       it != nodesSet.end(); ++it) {
    GraphMemInfo cost = getGraphMemInfo((*it).second);
    partitions.setGraphMemInfo((*it).first, cost);
  }

  // Move/Exchange nodes between any two connected partitions, until no gain is
  // get.
  // Step1 Move: Assume Partition1 -> Partition2, try to move nodes from
  // Partition2 to Partition1 if those nodes only use the nodes in
  // Partition1(recursively) and the move won't make Partition1's memory exceeds
  // the memory constraint, and the communication cost is minimized.
  bool gain = true;
  while (gain) {
    // gain is initialized as false, it will be set to be true if there is at
    // least one node can be moved from one set to antoher set.
    gain = false;
    for (FunctionToNodesMapTy::iterator it = nodesSet.begin();
         it != nodesSet.end(); ++it) {
      NodesSetTy nSet = (*it).second;
      std::vector<Node *> outUsers = getOutUsersWithOnePredecessor(nSet);
      if (outUsers.empty()) {
        continue;
      }
      Function *cur = (*it).first;
      size_t memSize = partitions.getGraphMemInfo(cur).constMemSize +
                       partitions.getGraphMemInfo(cur).inMemSize;
      size_t communicationCost = partitions.getGraphMemInfo(cur).outMemSize;
      // Check if a node can be moved to current node set (i.e nSet).
      for (int i = 0, e = outUsers.size(); i < e; i++) {
        // Rule 1: this move won't break memory constraint.
        if (memUsage_[outUsers[i]] + memSize > availableMemory) {
          continue;
        }
        // Rule 2: this move won't cause constant duplication.
        bool cont = false;
        for (int j = 0, e1 = outUsers[i]->getNumInputs(); j < e1; j++) {
          auto in = outUsers[i]->getNthInput(j);
          if (isa<Storage>(in.getNode()) && !in.hasOneUse()) {
            cont = true;
            break;
          }
        }
        if (cont) {
          continue;
        }
        // Rule 3: this move won't increase communication cost. Even if this
        // move won't change communication cost, according to rule 1 and rule 2,
        // the memory consumption of the partition where this node (i.e
        // outUsers[i]) belongs can be reduced. Therefore, it may trigger later
        // node movement or paritionCombine.
        nSet.insert(outUsers[i]);
        GraphMemInfo cost = getGraphMemInfo(nSet);
        if (cost.outMemSize <= communicationCost) {
          // Move this node to current node set.
          nSet.insert(outUsers[i]);
          nodesSet[cur].insert(outUsers[i]);
          Function *suc = partitions[outUsers[i]];
          nodesSet[suc].erase(outUsers[i]);
          // Update the partitions.
          partitions.add(outUsers[i], cur);
          partitions.setGraphMemInfo(cur, cost);
          if (nodesSet[suc].empty()) {
            // It is possible that after moving a node from Partition2 to
            // Partition1, Partition2 become empty. Remove the empty partition.
            partitions.deletePartition(suc);
            module_->eraseFunction(suc);
          } else {
            cost = getGraphMemInfo(nodesSet[suc]);
            partitions.setGraphMemInfo(suc, cost);
          }
          gain = true;
          communicationCost = cost.outMemSize;
          memSize += memUsage_[outUsers[i]];
        }
      }
    }
  }

  // TODO... :Step 2: exchange two nodes from two partitions to minimize
  // communication cost.

  // Combine the current partitions if necessary.
  partitionsCombine(partitions, nodesSet, availableMemory);
}

/// Assign nodes to partitions and return the mapping.
NodeToFunctionMap Partitioner::selectPartitions(Function *F,
                                                size_t availableMemory) {
  NodeToFunctionMap mapping;
  BFSLevel bfs = getBFSLevel(F);
  size_t level = bfs.size();
  // A list of cut. The graph can be partitioned by levels (cut[0], level - 1],
  // (cut[1], cut[0] - 1], ..., (-1, cut[n] - 1].
  std::vector<int> cut;

  // Step 1 : get the initial cut based on BFS levels and availableMemory.
  // TODO .. need to remove the duplicated memory usage.
  size_t mem = 0;
  for (int i = level - 1; i >= 0; i--) {
    size_t tmp = 0;
    for (size_t j = 0, e = bfs[i].second.size(); j < e; j++) {
      Node *N = bfs[i].second[j];
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
  for (size_t k = 0, e = cut.size(); k < e; k++) {
    newF = F->getParent()->createFunction(std::string(F->getName()) + "_part" +
                                          std::to_string(++color));
    mapping.createPartition(newF);
    size_t mem = 0;
    for (int i = k > 0 ? cut[k - 1] : level - 1; i > cut[k]; i--) {
      for (size_t j = 0, e1 = bfs[i].second.size(); j < e1; j++) {
        Node *N = bfs[i].second[j];
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

  // Step 3 : adjust the partition based on performance.
  partitionsAdjust(mapping, availableMemory);

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

  size_t availMem = deviceInfo_[0].availableMemory;

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
