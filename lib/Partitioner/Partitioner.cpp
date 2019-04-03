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

Function *Partitioner::selectRepFunc(Module *parent, uint64_t &memSize) {
  auto funcList = parent->getFunctions();
  Function *ret = nullptr;
  for (Function *F : funcList) {
    uint64_t size = memSize;

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
    uint64_t size = 0;
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

/// Get the minimal compute time for each op in the function.
void Partitioner::initOpComputeTime() {
  computeTime_.clear();

  // This code assumes all ops are BW limited from SRAM; except
  // if the input does not fit in SRAM -- then it is DRAM BW limited
  float peakDramBw = deviceInfo_[0].peakDramBw;
  float peakSramBw = deviceInfo_[0].peakSramBw;
  uint64_t sramCapacity = deviceInfo_[0].sramCapacity;
  float peakCompute = deviceInfo_[0].peakCompute;

  for (auto &node : F_->getNodes()) {
    /// compute memory side bytes for inputs from DRAM, SRAM.
    /// TODO: think about whether this is better off computed inside a Node.

    int n = node.getNumInputs();
    uint64_t sizeDram = 0;
    uint64_t sizeSram = 0;
    if (node.getKind() == Kinded::Kind::SaveNodeKind) {
      computeTime_[&node] = 0.0f;
      continue;
    }

    /// The memory bytes for embedding table lookups is data dependent,
    /// so it needs to be calculated as per the number of indices accessed.
    if (node.getKind() == Kinded::Kind::SparseLengthsWeightedSumNodeKind) {
      auto *SLWSN = llvm::dyn_cast<SparseLengthsWeightedSumNode>(&node);
      /// compute how many entries of the embedding table we look up
      auto numLookups = SLWSN->getIndices().getNode()->dims(0).front();
      /// compute how many bytes we read per lookup
      auto tableSize = SLWSN->getData().getNode()->getType(0)->getSizeInBytes();
      auto numRows = SLWSN->getData().getNode()->dims(0).front();
      auto sizePerLookup = tableSize / numRows;
      /// compute total bytes read
      uint64_t sizeInput = numLookups * sizePerLookup;

      /// does the table fit in SRAM or DRAM
      if (tableSize > sramCapacity) {
        sizeDram += sizeInput;
      } else {
        sizeSram += sizeInput;
      }

      /// we also read the indices, weights and lengths arrays
      sizeSram += SLWSN->getIndices().getNode()->getType(0)->getSizeInBytes();
      sizeSram += SLWSN->getWeights().getNode()->getType(0)->getSizeInBytes();
      sizeSram += SLWSN->getLengths().getNode()->getType(0)->getSizeInBytes();
    } else {
      /// for all other ops, iterate through all inputs and get size in bytes
      for (int i = 0; i < n; i++) {
        auto ty = node.getNthInput(i).getNode()->getType(0);
        uint64_t sizeInput = ty->getSizeInBytes();
        if (sizeInput > sramCapacity) {
          sizeDram += sizeInput;
        } else {
          sizeSram += sizeInput;
        }
      }
    }

    // Repeat for outputs
    if (node.getNumResults() > 0) {
      auto myty = node.getType(0);
      uint64_t sizeOutput = myty->getSizeInBytes();
      if (sizeOutput > sramCapacity) {
        sizeDram += sizeOutput;
      } else {
        sizeSram += sizeOutput;
      }
    }

    /// Calculate compute ops. Currently only computed for Matmul, Conv, FC
    /// TODO: think about whether this is better off computed inside a Node.
    uint64_t totalOps = 0;
    switch (node.getKind()) {
    case Kinded::Kind::MatMulNodeKind: {
      auto *MMN = llvm::dyn_cast<MatMulNode>(&node);
      auto lhsDims = MMN->getLHS().dims();
      auto rhsDims = MMN->getRHS().dims();
      totalOps = 2 * lhsDims[0] * lhsDims[1] * rhsDims[1];
      break;
    }
    case Kinded::Kind::FullyConnectedNodeKind: {
      auto *FCN = llvm::dyn_cast<FullyConnectedNode>(&node);
      auto inputDims = FCN->getInput().dims();
      auto wtDims = FCN->getWeights().dims();
      totalOps = 2 * inputDims[0] * inputDims[1] * wtDims[1];
      break;
    }
    case Kinded::Kind::ConvolutionNodeKind: {
      auto *CN = llvm::dyn_cast<ConvolutionNode>(&node);
      auto resultDims = CN->getResult().dims();
      // Get the product of batch, output height, output dims, output channels
      totalOps = resultDims[0];
      for (size_t i = 1, e = resultDims.size(); i < e; i++) {
        totalOps *= resultDims[i];
      }
      // Multiply in kernel height, kernel width
      auto kernelDims = CN->getKernels();
      totalOps *= kernelDims[0] * kernelDims[1];
      // Multiply in input channels/groups
      auto inputChannels = CN->getInput().dims()[1];
      auto nGroups = CN->getGroup();
      totalOps *= (inputChannels * 1.0 / nGroups);
      break;
    }
    default:
      break;
    }

    /// Compute compute roofline as max of flops, DRAM, SRAM BW
    /// See https://bit.ly/2UdJ3mz
    /// Add epsilons to prevent seg faults on unitialized peak values
    computeTime_[&node] =
        std::max(totalOps * 1.0f / std::max(peakCompute, 1e-6f),
                 std::max(sizeDram * 1.0f / std::max(peakDramBw, 1e-6f),
                          sizeSram * 1.0f / std::max(peakSramBw, 1e-6f)));
  }
}

// Combine the partitions according to the following rules:
// Rule 1 :if all outside uses of the nodes in partition1 is in partition2, and
// the sum of memory consumption of partition1 and partition2 is less than
// availableMemory, combine partition1 and partition2.
void Partitioner::partitionsCombine(NodeToFunctionMap &partitions,
                                    FunctionToNodesMapTy &nodesSet,
                                    uint64_t availableMemory) {

  size_t origPartitions = 0;

  // Do the combination until the size of partitions is stable.
  while (partitions.getPartitions().size() != origPartitions) {
    origPartitions = partitions.getPartitions().size();
    // Rule 1:
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
        NodesSetTy tmp = (nodesSet.find(suc))->second;
        GraphMemInfo cost1 = partitions.getGraphMemInfo(cur);
        GraphMemInfo cost2 = partitions.getGraphMemInfo(suc);
        if (cost1.getTotalMemSize() + cost2.getTotalMemSize() -
                cost1.outMemSize <
            availableMemory) {
          // We can combine the two partitions to fit one device.
          for (NodesSetTy::iterator it2 = tmp.begin(); it2 != tmp.end();
               ++it2) {
            partitions.add(*it2, cur);
          }
          GraphMemInfo newCost;
          newCost.constMemSize = cost1.constMemSize + cost2.constMemSize;
          newCost.inMemSize =
              cost1.inMemSize + cost2.inMemSize - cost1.outMemSize;
          newCost.outMemSize = cost2.outMemSize;
          partitions.setGraphMemInfo((*it).first, newCost);
          (*it).second.insert(tmp.begin(), tmp.end());
          partitions.deletePartition(suc);
          nodesSet.erase(suc);
          module_->eraseFunction(suc);
        }
      }
    }
  }
}

void Partitioner::partitionsAdjust(NodeToFunctionMap &partitions,
                                   uint64_t availableMemory) {
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
      uint64_t memSize = partitions.getGraphMemInfo(cur).constMemSize +
                         partitions.getGraphMemInfo(cur).inMemSize;
      uint64_t communicationCost = partitions.getGraphMemInfo(cur).outMemSize;
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
        Function *suc = partitions[outUsers[i]];
        uint64_t outMem = getOutMemPerNode(nodesSet[suc], outUsers[i]);
        if (cost.outMemSize - outMem <= communicationCost) {
          // Move this node to current node set.
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
            nodesSet.erase(suc);
          } else {
            GraphMemInfo newCost = getGraphMemInfo(nodesSet[suc]);
            partitions.setGraphMemInfo(suc, newCost);
          }
          gain = true;
          communicationCost = cost.outMemSize - outMem;
          memSize += memUsage_[outUsers[i]];
        } else {
          nSet.erase(outUsers[i]);
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
                                                uint64_t availableMemory) {
  NodeToFunctionMap mapping;
  BFSLevel bfs = getBFSLevel(F);
  size_t level = bfs.size();
  // A list of cut. The graph can be partitioned by levels (cut[0], level - 1],
  // (cut[1], cut[0] - 1], ..., (-1, cut[n] - 1].
  std::vector<int> cut;

  // Step 1 : get the initial cut based on BFS levels and availableMemory.
  // TODO .. need to remove the duplicated memory usage.
  uint64_t mem = 0;
  for (int i = level - 1; i >= 0; i--) {
    uint64_t tmp = 0;
    for (size_t j = 0, e = bfs[i].size(); j < e; j++) {
      Node *N = bfs[i][j];
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
    uint64_t mem = 0;
    for (int i = k > 0 ? cut[k - 1] : level - 1; i > cut[k]; i--) {
      for (size_t j = 0, e1 = bfs[i].size(); j < e1; j++) {
        Node *N = bfs[i][j];
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
  rootDAGNodeTy DAGRoot = llvm::make_unique<DAGNode>();
  nodesDAGNodeTy nodes;
  DAGRoot->logicalDevice = 0;
  DAGRoot->name = F->getName();
  DAGRoot->deviceID = 0;
  DAGRoot->logicalDevice = 0;
  DAGNode *root = DAGRoot.get();
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
      nodes.push_back(std::move(subDAG));
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
          nodes.push_back(std::move(subDAG));
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

  DAG dag;
  dag.root = std::move(DAGRoot);
  dag.nodes = std::move(nodes);
  partitions_.push_back(std::move(dag));

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
      funcDAG[subF]->parents.push_back(root);
      root->children.push_back(funcDAG[subF]);
    }
  }

  // Adjust the logicalDevice for each DAGNode.
  if (mapping.getPartitions().size() > deviceInfo_.size()) {
    adjustLogicalDeviceID(root, mapping.getPartitions().size());
  }
}

DAGListTy &Partitioner::Partition() {

  // Find the representive function for running partitioning algrithm.
  F_ = selectRepFunc(module_, memSize_);
  uint64_t availMem = deviceInfo_[0].availableMemory;

  if (memSize_ < availMem) {
    // No partition is needed. Create DAGNode and return. This root is alway a
    // dummy function.
    for (auto F : module_->getFunctions()) {
      std::unique_ptr<DAGNode> DAG0 = llvm::make_unique<DAGNode>();
      DAG0->logicalDevice = 0;
      DAG0->name = F->getName();
      std::unique_ptr<DAGNode> DAG1 = llvm::make_unique<DAGNode>();
      DAG1->logicalDevice = 0;
      DAG1->name = F->getName();
      DAG1->parents.push_back(DAG0.get());
      DAG0->children.push_back(DAG1.get());
      nodesDAGNodeTy nodes;
      nodes.push_back(std::move(DAG1));
      partitions_.push_back({std::move(DAG0), std::move(nodes)});
    }
    return partitions_;
  }

  // Prepare 1: Get the min memory usage for each op.
  initOpMemUsage();

  // Prepare 2: Get the roofline memory bandwidth estimate for each op.
  initOpComputeTime();

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
