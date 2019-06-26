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
#include "glow/Optimizer/GraphOptimizer/GraphOptimizer.h"
#include "glow/Support/Support.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

#include <fstream>

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

/// -log-partition - Command line option to dump Partitioner logs.
static llvm::cl::OptionCategory PartitionerCat("Glow Partitioner Options");
static llvm::cl::opt<bool>
    logPartition("log-partition",
                 llvm::cl::desc("Enable logging partition info"),
                 llvm::cl::init(true), llvm::cl::cat(PartitionerCat));

/// -dump-partition - Command line option to dump the graph of each partitions
/// by calling F->dumpDAG().
static llvm::cl::opt<bool>
    dumpPartition("dump-partition",
                  llvm::cl::desc("Enable dumping the graph of each partitions"),
                  llvm::cl::init(false), llvm::cl::cat(PartitionerCat));

using namespace glow;
using llvm::isa;

// Sorted the std::pair<DAGNode *, uint64_t> based on the second from min to
// max.
bool sortMinMemory(const std::pair<Function *, uint64_t> &a,
                   const std::pair<Function *, uint64_t> &b) {
  return a.second < b.second;
}

void Partitioner::dumpDAG(llvm::StringRef dotFilename) const {
  if (partitions_.size() == 0)
    return;
  auto *root = partitions_[0].root.get();
  LOG(INFO) << "Writing dotty graph for DAG after graph partitioning: "
            << dotFilename.str();
  std::ofstream myfile;
  myfile.open(dotFilename);
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
    auto *root = nodes[i];
    for (size_t j = 0; j < root->children.size(); j++) {
      auto child = root->children[j];
      myfile << "\"" << escapeDottyString(root->name) << "\""
             << " -> "
             << "\"" << escapeDottyString(child->name) << "\""
             << ";";
    }
  }
  myfile << "}";

  myfile.close();
  return;
}

llvm::Error Partitioner::logicalDevicesValidation(
    const NodeToFunctionMap &partitions) const {
  std::map<std::string, std::set<DeviceIDTy>> partitionsNum;
  for (auto &func : partitions.getPartitions()) {
    auto backendName = partitions.getPartitionBackendName(func);
    if (partitionsNum.find(backendName) == partitionsNum.end()) {
      partitionsNum.emplace(backendName, std::set<DeviceIDTy>{});
    }
    auto logicalIDList = partitions.getLogicalDeviceIDList(func);
    for (size_t i = 0, e = logicalIDList.size(); i < e; i++) {
      partitionsNum[backendName].insert(logicalIDList[i]);
    }
    auto backendNum = backendMap_.at(backendName).num;
    RETURN_ERR_IF_NOT(
        partitionsNum[backendName].size() <= backendNum,
        llvm::formatv("Partition failed: the number of given({0}) devices({1}) "
                      "is fewer than the required minimal partitions({2}).",
                      backendName, backendNum,
                      partitionsNum[backendName].size())
            .str());
  }
  return llvm::Error::success();
}

llvm::Error
Partitioner::memoryUsageValidation(const NodeToFunctionMap &partitions) const {
  for (auto &func : partitions.getPartitions()) {
    auto backendName = partitions.getPartitionBackendName(func);
    auto usedMemSize = partitions.getGraphMemInfo(func).getTotalMemSize();
    auto availableMemSize = backendMap_.at(backendName).memSize;
    RETURN_ERR_IF_NOT(
        usedMemSize <= availableMemSize,
        llvm::formatv(
            "Partition failed: the memory usage({0}) of one partition exceeds "
            "the available memory({1}) of given devices({2}).",
            usedMemSize, availableMemSize, backendName)
            .str());
  }
  return llvm::Error::success();
}

Partitioner::Partitioner(Module *parent, const std::vector<DeviceInfo> &devices,
                         const std::vector<Backend *> &backends,
                         bool saturateHost, bool optimized)
    : module_(parent), deviceInfo_(devices), backends_(backends),
      saturateHost_(saturateHost), optimized_(optimized) {
  memSize_ = module_->getConstantsSize();
  logicalDeviceID_ = 0;
}

Partitioner::Partitioner(Module *parent, const std::vector<DeviceInfo> &devices,
                         bool saturateHost, bool optimized)
    : module_(parent), deviceInfo_(devices), saturateHost_(saturateHost),
      optimized_(optimized) {
  memSize_ = module_->getConstantsSize();
  logicalDeviceID_ = 0;
}

Function *Partitioner::selectRepFunc(Module *parent, uint64_t &memSize) {
  auto funcList = parent->getFunctions();
  Function *ret = nullptr;
  uint64_t maxMemSize = 0;
  for (Function *F : funcList) {
    uint64_t curSize = memSize;

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
        if (in && pSet.find(in->getName()) == pSet.end()) {
          auto ty = in->getType();
          curSize += ty->getSizeInBytes();
          pSet.insert(in->getName());
        }
      }
    }
    // Find the function with largest required memory as the representative
    // function.
    if (!ret || curSize > maxMemSize) {
      ret = F;
      maxMemSize = curSize;
    }
  }
  memSize = maxMemSize;
  return ret;
}

/// Get the minimal memory requirement (constant) for each op in the function.
void Partitioner::initOpMemUsage(Function *F) {
  memUsage_.clear();
  for (auto &node : F->getNodes()) {
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
void Partitioner::initOpComputeTime(Function *F) {
  computeTime_.clear();

  // This code assumes all ops are BW limited from SRAM; except
  // if the input does not fit in SRAM -- then it is DRAM BW limited
  float peakDramBw = deviceInfo_[0].peakDramBw;
  float peakSramBw = deviceInfo_[0].peakSramBw;
  uint64_t sramCapacity = deviceInfo_[0].sramCapacity;
  float peakCompute = deviceInfo_[0].peakCompute;

  for (auto &node : F->getNodes()) {
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
      auto numLookups = SLWSN->getIndices().dims().front();
      /// compute how many bytes we read per lookup
      auto tableSize = SLWSN->getData().getType()->getSizeInBytes();
      auto numRows = SLWSN->getData().dims().front();
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
      sizeSram += SLWSN->getIndices().getType()->getSizeInBytes();
      sizeSram += SLWSN->getWeights().getType()->getSizeInBytes();
      sizeSram += SLWSN->getLengths().getType()->getSizeInBytes();
    } else {
      /// for all other ops, iterate through all inputs and get size in bytes
      for (int i = 0; i < n; i++) {
        auto ty = node.getNthInput(i).getType();
        uint64_t sizeInput = ty->getSizeInBytes();
        if (sizeInput > sramCapacity) {
          sizeDram += sizeInput;
        } else {
          sizeSram += sizeInput;
        }
      }
    }

    // Repeat for outputs
    for (size_t i = 0, e = node.getNumResults(); i < e; i++) {
      auto myty = node.getType(i);
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
    /// Add epsilons to prevent seg faults on uninitialized peak values.
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
  // For each partition, create a node set.
  FunctionToNodesMapTy nodesSet;
  for (NodeToFunctionMapTy::iterator it = partitions.begin();
       it != partitions.end(); ++it) {
    nodesSet[(*it).second].insert((*it).first);
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
    // least one node can be moved from one set to another set.
    gain = false;
    for (FunctionToNodesMapTy::iterator it = nodesSet.begin();
         it != nodesSet.end(); ++it) {
      NodesSetTy &curSet = (*it).second;
      std::vector<Node *> outUsers = getOutUsersWithOnePredecessor(curSet);
      if (outUsers.empty()) {
        continue;
      }
      Function *cur = (*it).first;
      GraphMemInfo curCost = partitions.getGraphMemInfo(cur);
      // Check if a node can be moved to current node set (i.e curSet).
      for (int i = 0, e = outUsers.size(); i < e; i++) {
        // Get the new cost if outUsers[i] is added.
        GraphMemInfo newCurCost =
            updateGraphMemInfoByAddingNode(curSet, curCost, outUsers[i]);

        // Rule 1: this move won't break memory constraint.
        if (newCurCost.getTotalMemSize() > availableMemory) {
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
        // node movement or partitionsCombine.
        Function *suc = partitions[outUsers[i]];
        uint64_t outMem = getOutMemPerNode(nodesSet[suc], outUsers[i]);
        if (newCurCost.outMemSize - outMem <= curCost.outMemSize) {
          // Move this node to current node set.
          curSet.insert(outUsers[i]);
          Function *suc = partitions[outUsers[i]];
          nodesSet[suc].erase(outUsers[i]);
          curCost = newCurCost;
          // Update the partitions.
          partitions.add(outUsers[i], cur);
          partitions.setGraphMemInfo(cur, newCurCost);
          if (nodesSet[suc].empty()) {
            // It is possible that after moving a node from Partition2 to
            // Partition1, Partition2 become empty. Remove the empty partition.
            partitions.deletePartition(suc);
            nodesSet.erase(suc);
            module_->eraseFunction(suc);
          } else {
            GraphMemInfo newCost = getGraphMemInfo(nodesSet[suc]);
            partitions.setGraphMemInfo(suc, newCost);
          }
          gain = true;
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
                                                uint64_t availableMemory,
                                                llvm::StringRef backendName) {
  NodeToFunctionMap mapping;
  BFSLevel bfs = getBFSLevel(F);
  size_t level = bfs.size();

  // Step 1 : get the initial cut based on BFS levels and availableMemory.
  int color = 0;
  Function *newF;
  newF = F->getParent()->createFunction(std::string(F->getName()) + "_part" +
                                        std::to_string(++color));
  mapping.createPartition(newF, backendName);
  NodesSetTy currentPartition;
  GraphMemInfo graphMem;

  for (int i = level - 1; i >= 0; i--) {
    for (size_t j = 0, e = bfs[i].size(); j < e; j++) {
      Node *N = bfs[i][j];
      graphMem = updateGraphMemInfoByAddingNode(currentPartition, graphMem, N);
      // If after adding node N, the memory usage of this partition exceeds the
      // device memory limitations, N can't be added into the current partition
      // and a new partition is created.
      if (graphMem.getTotalMemSize() > availableMemory) {
        newF = F->getParent()->createFunction(
            std::string(F->getName()) + "_part" + std::to_string(++color));
        mapping.createPartition(newF, backendName);
        currentPartition.clear();
        graphMem =
            updateGraphMemInfoByAddingNode(currentPartition, GraphMemInfo{}, N);
      }
      currentPartition.insert(N);
      mapping.add(N, newF);
      mapping.setGraphMemInfo(newF, graphMem);
    }
  }

  // Step 2 : adjust the partition based on performance.
  partitionsAdjust(mapping, availableMemory);

  return mapping;
}

/// Assign the logicalDevice ID to each partition. The partitions with the same
/// logicalDevice ID will be assigned on the same physical devices. E.g: there
/// are 3 partitions node1(6GB) -> node2(14GB) -> node3(6GB). But we only have 2
/// devices with 16GB memory. The logicalDevice ID assigning rules are:
/// 1. For each type of backend, if the number of available physical devices is
/// equal or larger than the number of partitions, different partitions are
/// assigned with a different logicalDevice ID(i.e. each partition will be put
/// on a different physical device for execution). E.g. we have 3 partitions
/// node1->node2->node3, and 3 devices, the logicalDevice ID for each partition
/// with be (node1, 0), (node2, 1), and (node3, 2).
/// 2. For each type of backend, if the number of available physical devices is
/// smaller than the number of partitions, and we can find a way to put all
/// partitions on those pysical devices, this assignment will be applied and the
/// partitions on the same physical devices will be assigned the same
/// logicalDevice ID.  E.g: there are 3 partitions node1(6GB) -> node2(14GB) ->
/// node3(6GB). But we only have 2 devices with 16GB memory. The assignment will
/// be : (node1, 0), (node2, 1), (node3, 0).
/// 3. For each type of backend, if the number of available physical devices is
/// smaller than the number of partitions, and we can not find a way to put all
/// partitions on those pysical devices, we assign defferent partitions with
/// different logicalDevice ID.  E.g: there are 3 partitions node1(6GB) ->
/// node2(14GB) -> node3(6GB). But we only have 1 device with 16GB memory. The
/// assignment will be : (node1, 0), (node2, 1), (node3, 2). That is, even we
/// can put node1 and node3 on the same device, we won't do it.
DeviceIDTy Partitioner::assignLogicalDeviceID(NodeToFunctionMap &mapping) {
  DeviceIDTy logicalDeviceID = 0;

  std::map<std::string, std::vector<Function *>> backendFuncMap;
  for (auto &func : mapping.getPartitions()) {
    // Traverse the partitions, and get list of partitions with each
    // backendName.
    auto backendName = mapping.getPartitionBackendName(func);
    if (backendFuncMap.find(backendName) == backendFuncMap.end()) {
      backendFuncMap.emplace(backendName, std::vector<Function *>{func});
    } else {
      backendFuncMap[backendName].push_back(func);
    }
  }

  // For each type of the backend, assign the logicalDevice ID.
  for (const auto &p : backendFuncMap) {
    if (mapping.getPartitions().size() <= backendMap_[p.first].num) {
      // There is enough device with this backendName, no need to adjust the
      // logical ID.
      for (auto &func : p.second) {
        mapping.appendLogicalDeviceID(func, logicalDeviceID++);
      }
      continue;
    }
    // Get the list of functions with current BackendName, and sort it based on
    // used memory from min to max.
    std::vector<std::pair<Function *, uint64_t>> nodeSize;
    for (size_t i = 0, e = p.second.size(); i < e; i++) {
      Function *function = p.second[i];
      uint64_t totalMem = mapping.getGraphMemInfo(function).getTotalMemSize();
      nodeSize.push_back(std::make_pair(p.second[i], totalMem));
    }
    std::sort(nodeSize.begin(), nodeSize.end(), sortMinMemory);

    // Assume we have n devices(NOTE: here the n devices have the same avaiable
    // memory size, and the following algorithm can find the accurate result. If
    // the memory size are differnt, this assignment issue will be a NP problem
    // -- multiple knapsack problem, and the following algorithm becomes greedy
    // and the result may not be optimal), and m partitions, where m > n. If
    // these m partitions can be assigned to n devices, there must be 1 device
    // have at least (m - 1)/n + 1 partitions(Pigeonhole principle). Based on
    // this theory, the algorithm is: Given N devices, and M partitions:
    // Step 1 : sort the partitions from min to max based on their memory usage.
    std::sort(nodeSize.begin(), nodeSize.end(), sortMinMemory);
    // Step 2 : let n = N, m = M.
    size_t m = p.second.size();
    size_t n = backendMap_[p.first].num;
    while (m > 0) {
      // Step 3 : find the first k partitions whose total memory usage still
      // under the memory limitation (k should be max).
      uint64_t usedMem = 0;
      size_t numOfPartitionsWithSameID = (m - 1) / n + 1;
      size_t start = p.second.size() - m;
      size_t i;
      for (i = start; i < p.second.size(); i++) {
        if (usedMem + nodeSize[i].second > backendMap_[p.first].memSize) {
          break;
        }
        usedMem += nodeSize[i].second;
      }
      // Step 4 : if k = start - i found in step 3 is smaller than (m - 1) / n +
      // 1, this means we can't find a proper assignment to fit the number of
      // devices. Assign each partition with a unique logicalDevice ID and
      // return.
      if (i - start < numOfPartitionsWithSameID) {
        // Can't find a proper assignment. Assign each partition a unique
        // logicalDevice ID and return;
        logicalDeviceID = 0;
        for (auto &func : mapping.getPartitions()) {
          mapping.appendLogicalDeviceID(func, logicalDeviceID++);
        }
        return logicalDeviceID;
      }

      // Step 5 : Assign these partitions which are assigned to one device with
      // the same logical ID.
      for (size_t j = start; j < i; j++) {
        mapping.appendLogicalDeviceID(nodeSize[j].first, logicalDeviceID);
      }
      logicalDeviceID++;
      // Step 6 : Update the left number of devices and partitions.
      n--;
      m = m - (i - start);
    }
  }
  return logicalDeviceID;
}

/// Duplicate the network to saturate the number of devices. For example: If a
/// network is partitioned into two parts (\p logicalDeviceCount) and there are
/// six devices this would duplicate the network three times.
void Partitioner::saturateHost(unsigned logicalDeviceCount) {
  unsigned duplications = deviceInfo_.size() / logicalDeviceCount;
  if (duplications < 2) {
    return;
  }
  // Add additional logical devices to each node.
  for (auto &network : partitions_) {
    for (auto &node : network.nodes) {
      // Build list of new logical devices to add to node.
      std::vector<unsigned> newDevices;
      for (auto logical : node->logicalDevices) {
        // To ensure we do not have a logicalID collision we use the following
        // scheme. We have an iterator starting at 1 for each duplication pass.
        // The new ID we add is calculated as follows:
        // (iterator * logicalDeviceCount) + initialLogicalID
        for (unsigned i = 1; i < duplications; i++) {
          newDevices.push_back(logical + (i * logicalDeviceCount));
        }
      }
      // Append the new logical devices to the node's logical device vector.
      node->logicalDevices.insert(node->logicalDevices.end(),
                                  newDevices.begin(), newDevices.end());
    }
  }
}

/// Current only partition the representative function.
void Partitioner::doPartitioning(llvm::StringRef funcName,
                                 std::vector<Function *> funcs,
                                 NodeToFunctionMap &mapping, bool saveDAG) {
  // Add a dummy node to make sure that a DAG has a single entrance.
  DAGNodePtr DAGRoot = llvm::make_unique<DAGNode>();
  DAGNodePtrVec nodes;
  DAGRoot->logicalDevices = {0};
  DAGRoot->name = funcName;
  DAGRoot->module = module_;
  DAGRoot->deviceIDs = {0};
  DAGNode *root = DAGRoot.get();

  llvm::DenseMap<Node *, Node *> currToNew;

  // Clone nodes into target partition.
  for (size_t i = 0, e = funcs.size(); i < e; i++) {
    for (auto &N : funcs[i]->getNodes()) {
      auto *clone = N.clone();
      currToNew[&N] = clone;
      mapping[&N]->addNode(clone);
    }
  }

  // For any dependency that crosses a partition, add a placeholder and save
  // node. Record the dependence in the function graph.
  std::unordered_map<NodeValue, Placeholder *> placeholders;
  llvm::DenseMap<Function *, DAGNode *> funcDAG;
  for (auto *subF : mapping.getPartitions()) {
    if (funcDAG.find(subF) == funcDAG.end()) {
      std::unique_ptr<DAGNode> subDAG = llvm::make_unique<DAGNode>();
      subDAG->name = subF->getName();
      subDAG->logicalDevices = mapping.getLogicalDeviceIDList(subF);
      subDAG->backendName = mapping.getPartitionBackendName(subF);
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
        if (subF == inputF)
          continue;

        // Check if a DAGNode for subF's parent is created or not. If not,
        // create one.
        if (funcDAG.find(inputF) == funcDAG.end()) {
          std::unique_ptr<DAGNode> subDAG = llvm::make_unique<DAGNode>();
          subDAG->name = inputF->getName();
          subDAG->logicalDevices = mapping.getLogicalDeviceIDList(inputF);
          subDAG->backendName = mapping.getPartitionBackendName(inputF);
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
    partitions_.push_back(std::move(dag));
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
      funcDAG[subF]->parents.push_back(root);
      root->children.push_back(funcDAG[subF]);
    }
  }
}

FunctionToBackendNameMapTy
Partitioner::backendBasedPartition(Function *F,
                                   std::vector<Backend *> &backends) {
  FunctionToBackendNameMapTy ret;
  NodeToFunctionMap mapping;
  llvm::DenseMap<Node *, std::string> nodeToBackendName;

  // For each node find a backend that supports it.
  for (auto &N : F->getNodes()) {
    for (auto &backend : backends) {
      // Find the first backend that supports this node. The order of backends
      // is important.
      // TODO: the logic here need to be improved.
      if (backend->shouldLower(&N) || backend->isOpSupported(N)) {
        // Put this node into a partition for this backend.
        nodeToBackendName[&N] = backend->getBackendName();
        break;
      }
    }
    assert(nodeToBackendName.find(&N) != nodeToBackendName.end() &&
           "Node is not supported by any of the provided backends");
  }

  BFSLevel bfs = getBFSLevel(F);
  size_t level = bfs.size();
  int color = 0;
  Function *newF;
  newF = F->getParent()->createFunction(std::string(F->getName()) + "_part" +
                                        std::to_string(++color));
  auto backendName = nodeToBackendName[bfs[level - 1][0]];
  mapping.createPartition(newF, backendName);
  ret[newF] = backendName;
  for (int i = level - 1; i >= 0; i--) {
    for (size_t j = 0, e = bfs[i].size(); j < e; j++) {
      Node *N = bfs[i][j];
      auto bk = nodeToBackendName[N];
      if (bk != backendName) {
        backendName = bk;
        newF = F->getParent()->createFunction(
            std::string(F->getName()) + "_part" + std::to_string(++color));
        mapping.createPartition(newF, backendName);
        ret[newF] = backendName;
      }
      mapping.add(N, newF);
    }
  }

  // Here we just need to split the function without generating DAG.
  std::vector<Function *> funcs;
  funcs.push_back(F);
  doPartitioning(F->getName(), funcs, mapping, false);

  return ret;
}

void Partitioner::getBackendMap(
    std::map<std::string, BackendInfo> &backendMap,
    std::vector<std::unique_ptr<Backend>> &backendsHolder,
    std::vector<Backend *> &backends) {
  // If the backends are created already, we use them directly.
  bool hasBackends = backends_.size() != 0;
  if (hasBackends) {
    assert(backends_.size() == deviceInfo_.size() &&
           "number of backends and devices is not match.");
  }

  int n = 0;
  for (size_t i = 0, e = deviceInfo_.size(); i < e; i++) {
    std::string backendName = deviceInfo_[i].backendName;
    if (hasBackends) {
      assert(backends_[i]->getBackendName() == backendName &&
             "Backend Type mismatch.");
    }
    if (backendMap.find(backendName) == backendMap.end()) {
      BackendInfo backendInfo;
      backendInfo.num = 1;
      // We assume that for the same type of devices, the available memory size
      // is the same.
      // TODO : will improve the algorithm for different memory size.
      backendInfo.memSize = deviceInfo_[i].availableMemory;
      if (hasBackends) {
        backendInfo.backend = backends_[i];
      } else {
        backendsHolder.emplace_back(createBackend(backendName));
        backendInfo.backend = backendsHolder[n++].get();
      }
      backendMap[backendName] = backendInfo;
      backends.push_back(backendMap[backendName].backend);
    } else {
      backendMap[backendName].num += 1;
    }
  }
}

llvm::Error Partitioner::createDAGWithoutPartition(
    llvm::StringRef backendName, std::map<std::string, BackendInfo> &backendMap,
    CompilationContext &cctx) {
  for (auto F : module_->getFunctions()) {
    if (!optimized_) {
      auto backend = backendMap[backendName].backend;
      RETURN_IF_ERR(::glow::optimizeFunction(F, *backend, cctx));
    }
    std::unique_ptr<DAGNode> DAG0 = llvm::make_unique<DAGNode>();
    DAG0->logicalDevices = {0};
    DAG0->name = F->getName();
    DAG0->module = module_;
    std::unique_ptr<DAGNode> DAG1 = llvm::make_unique<DAGNode>();
    DAG1->logicalDevices = {0};
    DAG1->name = F->getName();
    DAG1->backendName = backendName;
    DAG1->parents.push_back(DAG0.get());
    DAG0->children.push_back(DAG1.get());
    DAGNodePtrVec nodes;
    nodes.push_back(std::move(DAG1));
    partitions_.push_back({std::move(DAG0), std::move(nodes)});
  }
  if (saturateHost_) {
    // Saturate the Host.
    saturateHost(1);
  }
  return llvm::Error::success();
}

llvm::Error Partitioner::Partition(CompilationContext &cctx) {
  // Prepare the mapping between BackendName and BackendInfo.
  std::vector<Backend *> backends;
  std::vector<std::unique_ptr<Backend>> backendHolder;
  getBackendMap(backendMap_, backendHolder, backends);

  // Step 0: Find the representative function for running partitioning
  // algorithm.
  F_ = selectRepFunc(module_, memSize_);

  // Step 1 : do the partition based on backends type.
  FunctionToBackendNameMapTy funcToBackend;
  std::string origName(F_->getName().data());
  if (backends.size() == 1) {
    // Only one type of backends, no need to backendName based partition.
    auto backendName = backends[0]->getBackendName();
    funcToBackend[F_] = backendName;

    if (memSize_ < backendMap_[backendName].memSize) {
      // No partition is needed. Create DAGNode and return. This root is alway a
      // dummy function.
      if (logPartition) {
        LOG(INFO) << "The model is too small for applying partition.\n"
                  << "Model size : " << memSize_ << "\n"
                  << "Device memory: " << backendMap_[backendName].memSize
                  << "\n";
      }
      return createDAGWithoutPartition(backendName, backendMap_, cctx);
    }
  } else {
    funcToBackend = backendBasedPartition(F_, backends);
    module_->eraseFunction(F_);
  }

  // Step 2 : optimize each functions based on its backend type and apply the
  // partition algorithm.
  NodeToFunctionMap mapping;
  std::vector<Function *> funcs;
  for (auto i = funcToBackend.begin(); i != funcToBackend.end(); ++i) {
    auto *func = i->first;
    auto *backend = backendMap_[i->second].backend;
    auto availMem = backendMap_[i->second].memSize;
    funcs.push_back(func);
    assert(func->verify() && "Conversion led to invalid function");
    // Step 2.1 : optimize a function if it has not been optimized yet.
    if (!optimized_) {
      RETURN_IF_ERR(::glow::optimizeFunction(func, *backend, cctx));
    }

    // Step 2.2 : get the min memory usage and the roofline memory bandwidth
    // estimate for each op.
    initOpMemUsage(func);
    initOpComputeTime(func);

    // Step 2.3 : apply graph partitioning algrithm to find out the partition.
    NodeToFunctionMap partitionMap =
        selectPartitions(func, availMem, i->second);
    mapping.insert(partitionMap);
  }

  // Check if the memory usage meets the device memory limitation.
  RETURN_IF_ERR(memoryUsageValidation(mapping));

  // Step 3 : assign each partition with a logical device id. The partitions
  // with the same logical device id will be assigned into the same physical
  // device.
  logicalDeviceID_ = assignLogicalDeviceID(mapping);

  // Check if the number of logical devices is less than the given physical
  // devices.
  RETURN_IF_ERR(logicalDevicesValidation(mapping));

  // Step 4 : do the real partitioning for the function list.
  doPartitioning(origName, funcs, mapping, true);

  // Step 5 : post-partition optimization - Adjust the logicalDevice for each
  // DAGNode.
  if (saturateHost_ && backends.size() == 1 &&
      mapping.getPartitions().size() < deviceInfo_.size()) {
    // Attempt to saturate the host when there is only one type of backend.
    // Passing in the count of logical devices. Since logicalId starts at 0 we
    // add one.
    saturateHost(logicalDeviceID_);
  }

  // Step 6 : clean up and verify the generate new functions.
  for (auto i = funcToBackend.begin(); i != funcToBackend.end(); ++i) {
    module_->eraseFunction(i->first);
  }

  auto funcList = module_->getFunctions();
  if (logPartition) {
    LOG(INFO) << "The number of partitions is : " << funcList.size()
              << ", and the DAG is dumped into DAG.dot file.\n";
    dumpDAG("DAG.dot");
  }

  int i = 0;
  for (Function *subF : funcList) {
    (void)subF;
    if (logPartition) {
      LOG(INFO) << "\t Partition " << i << ":\n"
                << "\t\t Name :\t" << subF->getName().str() << "\n"
                << "\t\t BackendKind :\t"
                << mapping.getPartitionBackendName(subF) << "\n"
                << "\t\t Memory :\t"
                << mapping.getGraphMemInfo(subF).getTotalMemSize() << "\n"
                << "\t\t LogicalDeviceIDs :\t"
                << mapping.getLogicalDeviceIDList(subF)[0] << "\n";
    }
    if (dumpPartition) {
      subF->dumpDAG("partitionLogicalID" +
                    std::to_string(mapping.getLogicalDeviceIDList(subF)[0]) +
                    "__" + subF->getName().str() + "__" +
                    mapping.getPartitionBackendName(subF) + ".dot");
    }
    i++;
    assert(subF->verify() && "Conversion led to invalid function");
  }
  return llvm::Error::success();
}
