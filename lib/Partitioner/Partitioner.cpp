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

#include "glow/Partitioner/Partitioner.h"

#include "folly/String.h"
#include "glow/Flags/Flags.h"
#include "glow/Optimizer/GraphOptimizer/GraphOptimizer.h"
#include "glow/Partitioner/PartitionerOptimizer.h"
#include "glow/Partitioner/PartitionerUtils.h"
#include "glow/Partitioner/PartitionerValidation.h"
#include "glow/Support/Support.h"

#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include <unordered_map>

#include <fstream>

namespace glow {
static llvm::cl::opt<bool, /* ExternalStorage */ true>
    GlowEnableLoadBalancedPartitioningOpt(
        "partitioner_enable_load_balance",
        llvm::cl::desc(
            "Enable a partitioner pass to optimize for "
            "load balance in addition to memory capacity constraints"),
        llvm::cl::location(glow::flags::EnableLoadBalancedPartitioning));
} // namespace glow

/// -log-partition - Command line option to dump Partitioner logs.
static llvm::cl::OptionCategory PartitionerCat("Glow Partitioner Options");
static llvm::cl::opt<bool, /* ExternalStorage */ true>
    logPartition("log-partition",
                 llvm::cl::desc("Enable logging partition info"),
                 llvm::cl::location(glow::flags::LogPartition),
                 llvm::cl::cat(PartitionerCat));

/// -dump-partition - Command line option to dump the graph of each partitions
/// by calling F->dumpDAG().
static llvm::cl::opt<bool, /* ExternalStorage */ true>
    dumpPartition("dump-partition",
                  llvm::cl::desc("Enable dumping the graph of each partitions"),
                  llvm::cl::location(glow::flags::DumpPartition),
                  llvm::cl::cat(PartitionerCat));

using namespace glow;
using llvm::isa;

// Sorted the std::pair<DAGNode *, uint64_t> based on the second from min to
// max.
bool sortMinMemory(const std::pair<Function *, uint64_t> &a,
                   const std::pair<Function *, uint64_t> &b) {
  return a.second < b.second;
}

void Partitioner::init() {
  memSize_ = module_->getConstantsSize();
  logicalDeviceID_ = 0;
  multiBackendNames_ = false;
  for (size_t i = 1, e = deviceInfo_.size(); i < e; i++) {
    if (deviceInfo_[i].backendName != deviceInfo_[0].backendName) {
      multiBackendNames_ = true;
      break;
    }
  }
}

Error Partitioner::finalize(const DAGListTy &partitions,
                            const NodeToFunctionMap &mapping) {

  // NOTE: Cannot validate the functions after partitioning here. The validation
  // needs the backend specific verifier. Tensor layouts, for example, might
  // have gone from canonical form to backend specific form.

  if (logPartition) {
    LOG(INFO) << "The number of partitions is : "
              << mapping.getPartitions().size();
    logPartitionInfo(mapping);
  }

  // Dump the graph of each function after partitioning.
  if (dumpPartition) {
    LOG(INFO) << "Dumping partitioning DAG to DAG.dot file.";
    dumpDAG("DAG.dot", partitions);
    for (const auto &node : partitions[0].nodes) {
      Function *subF = module_->getFunction(node->name);
      if (!subF) {
        // If we fail dump partition info for debugging.
        logPartitionInfo(mapping);
        return MAKE_ERR(ErrorValue::ErrorCode::PARTITIONER_ERROR,
                        "Invalid function name " + node->name);
      }
      subF->dumpDAG("partitionLogicalID" +
                    std::to_string(node->logicalDevices[0]) + "__" +
                    subF->getName().str() + "__" + node->backendName + ".dot");
    }
  }
  return Error::success();
}

Partitioner::Partitioner(Module *parent, const std::vector<DeviceInfo> &devices,
                         const std::vector<Backend *> &backends, bool optimized)
    : module_(parent), deviceInfo_(devices), backends_(backends),
      optimized_(optimized) {
  init();
}

Partitioner::Partitioner(Module *parent, const std::vector<DeviceInfo> &devices,
                         bool optimized, PartitionConfig partitionConfig)
    : module_(parent), deviceInfo_(devices), optimized_(optimized),
      partitionConfig_(partitionConfig) {
  init();
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

void Partitioner::partitionsAdjust(NodeToFunctionMap &partitions,
                                   uint64_t availableMemory) {
  // For each partition, create a node set.
  FunctionToNodesMap nodesSet;
  for (auto it = partitions.begin(); it != partitions.end(); ++it) {
    nodesSet[(*it).second].insert((*it).first);
  }

  // Optimize the communication cost.
  optimizeCommunicationCost(partitions, nodesSet, module_, availableMemory);

  // Combine the current partitions if necessary.
  partitionsCombine(partitions, nodesSet, module_, availableMemory);
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
  NodesSet currentPartition;
  GraphMemInfo graphMem;
  graphMem.contextCount = contextCount_;

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
      graphMem.contextCount = contextCount_;
      mapping.setGraphMemInfo(newF, graphMem);
    }
  }

  // Step 2 : adjust the partition based on performance.
  partitionsAdjust(mapping, availableMemory);

  return mapping;
}

void Partitioner::saturateHost(unsigned logicalDeviceCount,
                               const DAGListTy &partitions,
                               size_t availableLogicalDevices) {
  DCHECK(availableLogicalDevices <= deviceInfo_.size())
      << "Requested number of logical devices must be less than or euqal "
         "the number of found devices.";
  // If not specified, use number of available physical devices.
  if (availableLogicalDevices == 0 ||
      availableLogicalDevices > deviceInfo_.size()) {
    availableLogicalDevices = deviceInfo_.size();
  }
  unsigned duplications = availableLogicalDevices / logicalDeviceCount;
  if (duplications < 2) {
    return;
  }
  // Add additional logical devices to each node.
  for (auto &network : partitions) {
    for (auto &node : network.nodes) {
      // Set instanceCount.
      node->instanceCount = duplications;
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

Expected<DAGListTy> Partitioner::backendBasedPartition(
    FunctionToBackendNameMap &funcToBackend, Function *F,
    std::vector<Backend *> &backends, CompilationContext &cctx) {
  NodeToFunctionMap mapping;
  llvm::DenseMap<Node *, std::string> nodeToBackendName;

  // For each node find a backend that supports it.
  for (auto &N : F->getNodes()) {
    for (auto &backend : backends) {
      // Find the first backend that supports this node. The order of backends
      // is important. The check flow is :

      // Step 1: If a node is in pre-defined non-supported nodes set, it can not
      // be assigned to this backend. Continue.
      const auto &nonSupportedNodesKinds =
          backendMap_[backend->getBackendName()].nonSupportedNodesKinds;
      if (nonSupportedNodesKinds.count(N.getKind())) {
        // This op is on the pre-defined non-supported op list:
        continue;
      }
      // Step 2: If the pre-defined supported nodes set is empty, it means all
      // nodes could be assigned to this backend. If the pre-defined supported
      // nodes set is not empty, we check that if the node from Step 1 is in
      // this set or not. If not, continue.
      const auto &supportedNodesKinds =
          backendMap_[backend->getBackendName()].supportedNodesKinds;
      if (!supportedNodesKinds.empty() &&
          !supportedNodesKinds.count(N.getKind())) {
        // This op is not on the pre-definded supported op list:
        continue;
      }
      // Step 3: Check if the node is actually supported in this backend, if so,
      // assign it to this backend and break. Otherwise continue.
      // TODO: the logic here need to be improved.
      if (backend->shouldLower(&N) || backend->isOpSupported(N)) {
        // Put this node into a partition for this backend.
        nodeToBackendName[&N] = backend->getBackendName();
        break;
      }
    }
    if (nodeToBackendName.find(&N) == nodeToBackendName.end()) {
      logPartitionInfo(mapping);
      return MAKE_ERR(ErrorValue::ErrorCode::PARTITIONER_ERROR,
                      "Node is not supported by any of the provided backends");
    }
  }

  BFSLevel bfs = getBFSLevel(F);
  size_t level = bfs.size();
  int color = 0;
  Function *newF;
  newF = F->getParent()->createFunction(std::string(F->getName()) + "_part" +
                                        std::to_string(++color));
  auto backendName = nodeToBackendName[bfs[level - 1][0]];
  if (cctx.precisionConfig.quantMode == QuantizationMode::Profile) {
    // When profiling, all the partition backend is assigned to
    // profilingBackend.
    mapping.createPartition(newF, profilingBackend);
    funcToBackend[newF] = profilingBackend;
  } else {
    mapping.createPartition(newF, backendName);
    funcToBackend[newF] = backendName;
  }
  for (int i = level - 1; i >= 0; i--) {
    for (size_t j = 0, e = bfs[i].size(); j < e; j++) {
      Node *N = bfs[i][j];
      auto bk = nodeToBackendName[N];
      if (bk != backendName) {
        backendName = bk;
        newF = F->getParent()->createFunction(
            std::string(F->getName()) + "_part" + std::to_string(++color));
        if (cctx.precisionConfig.quantMode == QuantizationMode::Profile) {
          // When profiling, all the partition backend is assigned to be
          // profilingBackend.
          mapping.createPartition(newF, profilingBackend);
          funcToBackend[newF] = profilingBackend;
        } else {
          mapping.createPartition(newF, backendName);
          funcToBackend[newF] = backendName;
        }
      }
      mapping.add(N, newF);
    }
  }

  std::vector<Function *> funcs;
  funcs.push_back(F);
  // When profiling, the partition flow will be stopped after
  // backendBasedPartition. Therefore, the DAG needs to be generated. Otherwise,
  // no need to generate DAG.
  bool genDAG = cctx.precisionConfig.quantMode == QuantizationMode::Profile
                    ? true
                    : false;
  if (genDAG) {
    DeviceIDTy logicalDeviceID = 0;
    for (auto &func : mapping.getPartitions()) {
      mapping.appendLogicalDeviceID(func, logicalDeviceID++);
    }
  }
  return doPartitioning(F->getName(), funcs, module_, mapping, genDAG,
                        cctx.backendOpts.backendSpecificNodeInfo);
}

void Partitioner::genBackendMap(
    std::map<std::string, BackendInfo> &backendMap,
    std::vector<std::unique_ptr<Backend>> &backendsHolder,
    std::vector<Backend *> &backends) {
  // If the backends are created already, we use them directly.
  bool hasBackends = backends_.size() != 0;
  if (hasBackends) {
    DCHECK(backends_.size() == deviceInfo_.size())
        << "number of backends and devices is not match.";
  }

  int n = 0;
  for (size_t i = 0, e = deviceInfo_.size(); i < e; i++) {
    std::string backendName = deviceInfo_[i].backendName;
    if (hasBackends) {
      DCHECK(backends_[i]->getBackendName() == backendName)
          << "Backend Type mismatch.";
    }
    if (backendMap.find(backendName) == backendMap.end()) {
      BackendInfo backendInfo;
      backendInfo.num = 1;
      // We assume that for the same type of devices, the available memory size
      // is the same.
      // TODO : will improve the algorithm for different memory size.
      backendInfo.memSize = deviceInfo_[i].availableMemory;
      backendInfo.inputCountMax = deviceInfo_[i].inputCountMax;
      backendInfo.peakDramBw = deviceInfo_[i].peakDramBw;
      backendInfo.peakSramBw = deviceInfo_[i].peakSramBw;
      backendInfo.sramCapacity = deviceInfo_[i].sramCapacity;
      backendInfo.peakCompute = deviceInfo_[i].peakCompute;
      backendInfo.nonSupportedNodesKinds =
          generateNodeKindsSet(deviceInfo_[i].nonSupportedNodes);
      backendInfo.supportedNodesKinds =
          generateNodeKindsSet(deviceInfo_[i].supportedNodes);
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
      // Since we are currently assuming one value it should be the max.
      backendMap[backendName].memSize = std::max(
          backendMap[backendName].memSize, deviceInfo_[i].availableMemory);
    }
  }
}

const DeviceInfo &
Partitioner::getDeviceInfoForBackend(llvm::StringRef backendName) {
  for (DeviceInfo &devInfo : deviceInfo_) {
    if (devInfo.backendName == backendName)
      return devInfo;
  }
  llvm_unreachable("Each backend should have at least one device");
}

Expected<DAGListTy> Partitioner::createDAGWithoutPartition(
    llvm::StringRef backendName, std::map<std::string, BackendInfo> &backendMap,
    CompilationContext &cctx) {
  DAGListTy partitions;
  const DeviceIDTy logDevice = 0;
  for (auto F : module_->getFunctions()) {
    if (!optimized_) {
      auto backend = backendMap[backendName.str()].backend;
      RETURN_IF_ERR(::glow::optimizeFunction(
          F, *backend, cctx, &getDeviceInfoForBackend(backendName)));
    }
    std::unique_ptr<DAGNode> DAG0 = glow::make_unique<DAGNode>();
    DAG0->logicalDevices = {logDevice};
    DAG0->name = F->getName().str();
    DAG0->module = module_;
    std::unique_ptr<DAGNode> DAG1 = glow::make_unique<DAGNode>();
    DAG1->logicalDevices = {logDevice};
    DAG1->name = F->getName().str();
    DAG1->backendName = backendName.str();
    DAG1->parents.push_back(DAG0.get());
    DAG0->children.push_back(DAG1.get());
    DAG1->replicationCount = cctx.replicationCount;
    DAGNodePtrVec nodes;
    nodes.push_back(std::move(DAG1));
    partitions.push_back({std::move(DAG0), std::move(nodes)});
  }
  if (cctx.saturateHost) {
    // Saturate the Host.
    saturateHost(1, partitions, cctx.saturateKDevices);
  }

  NodeToFunctionMap mapping;
  for (auto func : module_->getFunctions()) {
    mapping.createPartition(func, backendName);
    mapping.setGraphMemInfo(func, getFunctionMemory(func));

    // Use the same hard-coded logical device ID as used for the DAG itself.
    mapping.appendLogicalDeviceID(func, logDevice);
  }

  RETURN_IF_ERR(finalize(partitions, mapping));

  return std::move(partitions);
}

Expected<DAGListTy> Partitioner::loadBalancedPartition(CompilationContext &cctx,
                                                       size_t numDevices) {

  if (multiBackendNames_) {
    VLOG(1) << "For multi backend types, load-balanced partition can't be "
               "applied. Call heterogeneous partition instead.";
    return heterogeneousPartition(cctx);
  }
  F_ = selectRepFunc(module_, memSize_);
  std::string origName(F_->getName().data());
  DAGListTy partitions;
  std::vector<Backend *> backends;
  genBackendMap(backendMap_, backendHolder_, backends);
  auto backendName = backends[0]->getBackendName();

  if (memSize_ < backendMap_[backendName].memSize) {
    // No partition is needed. Create DAGNode and return. This root is always a
    // dummy function.
    if (logPartition) {
      LOG(INFO) << "The model is too small for applying partition.\n"
                << "Model size : " << memSize_ << "\n"
                << "Backend Name : " << backendName << "\n"
                << "Device memory: " << backendMap_[backendName].memSize
                << "\n";
    }
    return createDAGWithoutPartition(backendName, backendMap_, cctx);
  }

  // Step 1: Get the minial number of partitions from auto-partition.
  uint64_t availableMemory = backendMap_[backendName].memSize;
  if (!optimized_) {
    RETURN_IF_ERR(::glow::optimizeFunction(F_, *(backends[0]), cctx));
  }
  NodeToFunctionMap mapping =
      selectPartitions(F_, availableMemory, backendName);
  logicalDeviceID_ = assignLogicalDeviceID(mapping, backendMap_);

  if (logicalDeviceID_ > numDevices) {
    numDevices = logicalDeviceID_;
  }
  // Step 2:
  // Currently, the load balanced partitioner disregards the input mapping
  // and only uses the numPartitions input from previous partitioning passes
  // But we take this in to leave open the option of using the previous mapping
  // at a later point.
  // The main idea here is to use the roofline estimates to load balance
  // partitions. At this point, we stick to one partition per device, so
  // we ensure that we only have edges from nodes in smaller partition ids to
  // nodes in larger partition ids to ensure an acyclic DAGNode graph.
  //
  // The overall algorithm is as follows:
  // Iterate through all operators in breadth-first fashion.
  // For each operator do:
  // (a) Find the maximum partition id of each input node.
  // (b) Assign the operator to this partition if memory
  //     constraints are satisfied and the total sum of operator runtimes
  //     assigned to the partition exceeds 1/numPartitions fraction of
  //     overall roofline runtime
  // (c) In case memory constraint isnt satisfied, then try to put operator
  //     in successively higher partitions until the conditions get satisfied.
  //     If we cannot find such a partition where this operator can be assigned,
  //     throw an error.

  // Initialize runtimes and memory availability per device
  std::vector<float> deviceTime(numDevices, 0);
  std::vector<size_t> memoryAvailable(numDevices, availableMemory);
  std::vector<NodesSet> nodesInPartitions(numDevices);
  std::vector<GraphMemInfo> graphMem(numDevices, GraphMemInfo{});
  std::vector<Function *> partitionFuncs(numDevices);

  // Compute total roofline time
  NodeToFunctionMap partitionMap;
  float totalRooflineTime = 0;
  for (auto &n : F_->getNodes()) {
    totalRooflineTime +=
        getNodeComputeTime(&n, backendMap_[deviceInfo_[0].backendName]);
  }

  float timePerPartition = totalRooflineTime / numDevices;

  // Get the BFS levels
  Function *newF;
  BFSLevel bfs = getBFSLevel(F_);
  size_t level = bfs.size();

  // Create the functions and push them into the mapping
  for (DeviceIDTy curPartition = 0; curPartition < numDevices; curPartition++) {
    std::string funcName =
        std::string(F_->getName()) + "_part" + std::to_string(curPartition + 1);
    if (F_->getParent()->hasFunction(funcName)) {
      newF = F_->getParent()->getFunction(funcName);
      F_->getParent()->eraseFunction(newF);
    }
    newF = F_->getParent()->createFunction(funcName);
    partitionMap.createPartition(newF, backendName);
    partitionMap.appendLogicalDeviceID(newF, curPartition);
    partitionFuncs[curPartition] = newF;
  }

  // Go through operators level by level
  for (int i = level - 1; i >= 0; i--) {
    for (size_t j = 0, e = bfs[i].size(); j < e; j++) {
      Node *N = bfs[i][j];

      // Find the maximum partition id of the inputs to the node
      DeviceIDTy maxLogicalDeviceId = 0;
      for (auto &I : getInputs(N)) {
        Function *inpF = partitionMap[I];
        auto logicalDeviceIds = partitionMap.getLogicalDeviceIDList(inpF);
        DCHECK(logicalDeviceIds.size() == 1);
        auto logicalDeviceId = logicalDeviceIds[0];
        if (logicalDeviceId > maxLogicalDeviceId) {
          maxLogicalDeviceId = logicalDeviceId;
        }
      }

      auto curOpTime =
          getNodeComputeTime(N, backendMap_[deviceInfo_[0].backendName]);
      auto curOpMemory = getNodeMemUsage(N);

      // Find a partition to put this node into
      DeviceIDTy curPartition = maxLogicalDeviceId;
      const float allowedLoadImbalanceFraction = 0.5f;
      for (; curPartition < numDevices; curPartition++) {
        // Put the op in current partition if
        // (a) memory constaints and load balance constraints are not violated,
        // or (b) this is the last partition and memory capacity isnt exceeded
        // The allowedLoadImbalanceFraction in the load balance case is to avoid
        // edge cases where load balance is only violated by a small amount and
        // moving to the next partition would result in significant imbalance in
        // runtime. Hence if the violation is by less than
        // allowedLoadImbalanceFraction of the operator cost, then we prefer to
        // keep it in the current partition.
        bool loadBalanceValid = deviceTime[curPartition] +
                                    curOpTime * allowedLoadImbalanceFraction <
                                timePerPartition;
        bool memValid = memoryAvailable[curPartition] >= curOpMemory;

        if (memValid && (loadBalanceValid || curPartition == numDevices - 1)) {
          // valid, put the node in the current partition
          Function *curF = partitionFuncs[curPartition];
          partitionMap.add(N, curF);
          deviceTime[curPartition] += curOpTime;
          memoryAvailable[curPartition] -= curOpMemory;
          graphMem[curPartition] = updateGraphMemInfoByAddingNode(
              nodesInPartitions[curPartition], graphMem[curPartition], N);
          nodesInPartitions[curPartition].insert(N);
          partitionMap.setGraphMemInfo(curF, graphMem[curPartition]);
          break;
        }
      }

      // Throw error if we were not able to put this node into any partition
      if (curPartition >= numDevices) {
        logPartitionInfo(partitionMap);
        return MAKE_ERR(ErrorValue::ErrorCode::PARTITIONER_ERROR,
                        "Load balance partition error");
      }
    }
  }
  for (size_t i = 0; i < numDevices; i++) {
    VLOG(1) << "Partition #" << i << " has estimated runtime " << deviceTime[i];
  }
  // Check if the memory usage meets the device memory limitation.
  RETURN_IF_ERR(memoryUsageValidation(partitionMap, backendMap_));

  // assignLogicalDeviceID adds all partitions to their logical device, clear
  // the existing first to prevent duplication.
  partitionMap.clearLogicalDeviceID();
  logicalDeviceID_ = assignLogicalDeviceID(partitionMap, backendMap_);
  RETURN_IF_ERR(logicalDevicesValidation(partitionMap, backendMap_));
  RETURN_IF_ERR(resourceCountValidation(partitionMap, backendMap_));

  partitions =
      doPartitioning(origName, {F_}, module_, partitionMap, /* saveDAG */ true,
                     cctx.backendOpts.backendSpecificNodeInfo);
  module_->eraseFunction(F_);

  if (cctx.saturateHost &&
      partitionMap.getPartitions().size() < deviceInfo_.size()) {
    saturateHost(logicalDeviceID_, partitions, cctx.saturateKDevices);
  }

  RETURN_IF_ERR(finalize(partitions, partitionMap));

  return std::move(partitions);
}

Expected<DAGListTy>
Partitioner::quantizationProfilingPartition(CompilationContext &cctx) {
  // For quantization profiling flow, currently we assume there is only 1
  // function in a module.
  if (module_->getFunctions().size() != 1) {
    return MAKE_ERR(
        ErrorValue::ErrorCode::PARTITIONER_ERROR,
        strFormat(
            "Invalid : %lu functions in a module. In quantization profiling "
            "partition flow, the module can only contain 1 function",
            module_->getFunctions().size()));
  }

  // Quantization profiling flow is run under CPU backend, so we don't really
  // need the concrete partition. The backendBasedPartition is necessary since
  // we need the mapping between quantized tensor and original tensor.
  DAGListTy partitions;
  std::vector<Backend *> backends;
  genBackendMap(backendMap_, backendHolder_, backends);
  F_ = selectRepFunc(module_, memSize_);

  FunctionToBackendNameMap funcToBackend;
  ASSIGN_VALUE_OR_RETURN_ERR(
      partitions, backendBasedPartition(funcToBackend, F_, backends, cctx));
  module_->eraseFunction(F_);
  std::unique_ptr<Backend> backend(createBackend(profilingBackend));
  for (Function *subF : module_->getFunctions()) {
    DCHECK(subF->verify()) << "Conversion led to invalid function";
    if (!optimized_) {
      RETURN_IF_ERR(::glow::optimizeFunction(subF, *backend, cctx));
    }
  }
  if (logPartition) {
    LOG(INFO)
        << "Profiling a model to be partitioned cross different backends. Each "
           "sub-network will be optimized and run on cpu backend.\n";
  }
  return std::move(partitions);
}

Expected<DAGListTy>
Partitioner::heterogeneousPartition(CompilationContext &cctx) {
  DAGListTy partitions;
  // Prepare the mapping between BackendName and BackendInfo.
  std::vector<Backend *> backends;
  genBackendMap(backendMap_, backendHolder_, backends);

  // Step 0: Find the representative function for running partitioning
  // algorithm.
  F_ = selectRepFunc(module_, memSize_);

  // Step 1 : do the partition based on backends type.
  FunctionToBackendNameMap funcToBackend;
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
                  << "Backend Name : " << backendName << "\n"
                  << "Device memory: " << backendMap_[backendName].memSize
                  << "\n";
      }
      return createDAGWithoutPartition(backendName, backendMap_, cctx);
    }
    // NOTE: the following error detection will be removed once multi-functions
    // in a module is supported.
    if (module_->getFunctions().size() != 1) {
      return MAKE_ERR(
          ErrorValue::ErrorCode::PARTITIONER_ERROR,
          strFormat("Invalid : %lu functions in a module. Now in heterogeneous "
                    "partition flow, the module can only contain 1 function",
                    module_->getFunctions().size()));
    }
  } else {
    // NOTE: the following error detection will be removed once multi-functions
    // in a module is supported.
    if (module_->getFunctions().size() != 1) {
      return MAKE_ERR(
          ErrorValue::ErrorCode::PARTITIONER_ERROR,
          strFormat(
              "Invalid : %lu functions in a module. Now in heterogeneous partition\
 flow, the module can only contain 1 function",
              module_->getFunctions().size()));
    }
    ASSIGN_VALUE_OR_RETURN_ERR(
        partitions, backendBasedPartition(funcToBackend, F_, backends, cctx));
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
    DCHECK(func->verify()) << "Conversion led to invalid function";
    // Step 2.1 : optimize a function if it has not been optimized yet.
    if (!optimized_) {
      RETURN_IF_ERR(::glow::optimizeFunction(
          func, *backend, cctx,
          &getDeviceInfoForBackend(backend->getBackendName())));
    }

    // Step 2.2 : apply graph partitioning algrithm to find out the partition.
    NodeToFunctionMap partitionMap =
        selectPartitions(func, availMem, i->second);
    mapping.insert(partitionMap);
  }

  // Check if the memory usage meets the device memory limitation.
  RETURN_IF_ERR(memoryUsageValidation(mapping, backendMap_));

  // Step 3 : assign each partition with a logical device id. The partitions
  // with the same logical device id will be assigned into the same physical
  // device.
  logicalDeviceID_ = assignLogicalDeviceID(mapping, backendMap_);

  // Check if the number of logical devices is less than the given physical
  // devices.
  RETURN_IF_ERR(logicalDevicesValidation(mapping, backendMap_));

  // Step 4 : do the real partitioning for the function list.
  partitions =
      doPartitioning(origName, funcs, module_, mapping, /* saveDAG */ true,
                     cctx.backendOpts.backendSpecificNodeInfo);

  // Step 5 : Post-partition optimization - Adjust the logicalDevice for each
  // DAGNode.
  if (cctx.saturateHost && backends.size() == 1 &&
      mapping.getPartitions().size() < deviceInfo_.size()) {
    // Attempt to saturate the host when there is only one type of backend.
    // Passing in the count of logical devices. Since logicalId starts at 0 we
    // add one.
    saturateHost(logicalDeviceID_, partitions, cctx.saturateKDevices);
  }

  // Step 6 : clean up and verify the generated new functions.
  for (auto i = funcToBackend.begin(); i != funcToBackend.end(); ++i) {
    module_->eraseFunction(i->first);
  }

  RETURN_IF_ERR(finalize(partitions, mapping));

  return std::move(partitions);
}

Expected<DAGListTy>
Partitioner::partitionFromConfig(const PartitionConfig &partitionConfig,
                                 CompilationContext &cctx) {
  DAGListTy partitions;
  // Prepare the mapping between BackendName and BackendInfo.
  std::vector<Backend *> backends;
  genBackendMap(backendMap_, backendHolder_, backends);
  Function *F = module_->getFunction(partitionConfig.funcName);
  if (!F) {
    return MAKE_ERR(ErrorValue::ErrorCode::PARTITIONER_ERROR,
                    strFormat("Can't find function %s in current module.",
                              F->getName().str().data()));
  }

  DCHECK(
      partitionConfig.numOfPartitions == partitionConfig.backendNames.size() &&
      partitionConfig.numOfPartitions == partitionConfig.partitionNames.size())
      << "Invalid user-defined partition config.";

  if (partitionConfig.backendHints.size()) {
    DCHECK(partitionConfig.numOfPartitions ==
           partitionConfig.backendHints.size())
        << "Invalid user-defined partition config (backendHints).";
  }

  NodeToFunctionMap partitionMap;
  std::vector<Function *> funcList;
  std::unordered_set<size_t> unused;
  std::vector<NodesSet> nodesSets(partitionConfig.numOfPartitions);
  // Create partitions based on the given number and names.
  for (size_t i = 0; i < partitionConfig.numOfPartitions; i++) {
    Function *newF = module_->createFunction(partitionConfig.partitionNames[i]);
    funcList.push_back(newF);
    partitionMap.createPartition(newF, partitionConfig.backendNames[i]);
    unused.insert(i);
  }

  // Map the nodes the the partitions.
  std::vector<Node *> unMapped;
  for (auto &node : F->getNodes()) {
    auto iter = partitionConfig.nodeToPartition.find(node.getName());
    if (iter == partitionConfig.nodeToPartition.end()) {
      // If a node in F is not in the node to partition mapping, put it into
      // unMaped list.
      unMapped.push_back(&node);
    } else {
      size_t partitionID = iter->second;
      DCHECK(partitionID < partitionConfig.numOfPartitions)
          << "Invalid partition id :" << partitionID;
      partitionMap.add(&node, funcList[partitionID]);
      unused.erase(partitionID);
      nodesSets[partitionID].insert(&node);
    }
  }

  // If there is unused partition and unmapped nodes, map those nodes to the
  // unused partition.
  if (unMapped.size()) {
    DCHECK_EQ(unused.size(), 1) << "There must be exactly 1 unused partition.";
    auto partitionID = *(unused.begin());
    for (auto &node : unMapped) {
      partitionMap.add(node, funcList[partitionID]);
      nodesSets[partitionID].insert(node);
    }
  }

  // Set backend hints if they exist
  if (partitionConfig.backendHints.size()) {
    for (size_t i = 0; i < partitionConfig.numOfPartitions; i++) {
      auto func = funcList[i];
      partitionMap.setBackendHints(func, partitionConfig.backendHints[i]);
    }
  }

  // Validate memory usage.
  for (size_t i = 0; i < partitionConfig.numOfPartitions; i++) {
    GraphMemInfo cost = getGraphMemInfo(nodesSets[i], contextCount_);
    partitionMap.setGraphMemInfo(funcList[i], cost);
  }
  RETURN_IF_ERR(memoryUsageValidation(partitionMap, backendMap_));

  // If logical device assignments are provided use them otherwise assign them.
  if (partitionConfig.logicalIDs.size()) {
    DCHECK(partitionConfig.numOfPartitions ==
           partitionConfig.logicalIDs.size());
    for (size_t i = 0; i < partitionConfig.numOfPartitions; i++) {
      auto func = funcList[i];
      for (auto logicalDevice : partitionConfig.logicalIDs[i]) {
        partitionMap.appendLogicalDeviceID(func, logicalDevice);
      }
    }

  } else {
    // Logical device ID validation.
    logicalDeviceID_ = assignLogicalDeviceID(partitionMap, backendMap_);
  }
  // Add replication count to config if provided.
  for (auto &replicationAssignment : partitionConfig.replicationCount) {
    auto func = funcList.at(replicationAssignment.first);
    partitionMap.addReplicationCount(func, replicationAssignment.second);
  }

  RETURN_IF_ERR(logicalDevicesValidation(partitionMap, backendMap_));
  RETURN_IF_ERR(resourceCountValidation(partitionMap, backendMap_));

  // Do partition.
  partitions = doPartitioning(F->getName(), {F}, module_, partitionMap,
                              /* saveDAG */ true,
                              cctx.backendOpts.backendSpecificNodeInfo);
  module_->eraseFunction(F);

  // DAG validation.
  RETURN_IF_ERR(dagValidation(partitions[0]));

  // Verify the function.
  for (size_t i = 0; i < partitionConfig.numOfPartitions; i++) {
    auto func = funcList[i];
    DCHECK(func->verify()) << "Conversion led to invalid function";
  }

  RETURN_IF_ERR(finalize(partitions, partitionMap));

  return std::move(partitions);
}

Expected<DAGListTy>
Partitioner::setupPrepartitionedModule(CompilationContext &cctx) {
  const PrePartitionedConfig &config = *cctx.prepartitionedConfig;

  RETURN_ERR_IF_NOT(
      !multiBackendNames_,
      "Do not support multiple backend kinds in prepartitioned flow.");

  // Prepare the mapping between BackendName and BackendInfo.
  std::vector<Backend *> backends;
  genBackendMap(backendMap_, backendHolder_, backends);

  const std::vector<Function *> &funcs = config.funcs;

  Backend *B = backends[0];
  auto backendName = B->getBackendName();

  // Optimize all Functions if necessary.
  if (!optimized_) {
    for (Function *F : funcs) {
      RETURN_IF_ERR(::glow::optimizeFunction(
          F, *B, cctx, &getDeviceInfoForBackend(backendName)));
    }
  }

  NodeToFunctionMap partitionMap;
  // Create partitions based on the given number and names.
  for (size_t i = 0, e = funcs.size(); i < e; i++) {
    partitionMap.createPartition(funcs[i], deviceInfo_[0].backendName);
  }

  // Map the nodes the the partitions.
  for (Function *F : funcs) {
    for (auto &node : F->getNodes()) {
      partitionMap.add(&node, F);
    }
  }

  // Validate memory usage.
  for (Function *F : funcs) {
    partitionMap.setGraphMemInfo(F, getFunctionMemory(F));
  }
  RETURN_IF_ERR(memoryUsageValidation(partitionMap, backendMap_));

  // If logical device assignments are provided use them otherwise assign them.
  DCHECK(funcs.size() == config.logicalIDs.size());
  for (size_t i = 0; i < funcs.size(); i++) {
    Function *F = funcs[i];
    for (auto logicalDevice : config.logicalIDs[i]) {
      partitionMap.appendLogicalDeviceID(F, logicalDevice);
    }
  }
  RETURN_IF_ERR(logicalDevicesValidation(partitionMap, backendMap_));
  RETURN_IF_ERR(resourceCountValidation(partitionMap, backendMap_));

  // Copy in or validate all members of the PPC.
  RETURN_ERR_IF_NOT(
      funcs.size() == config.backendSpecificOpts.size(),
      "Number of Functions must equal number of backendSpecificOpts");
  RETURN_ERR_IF_NOT(funcs.size() == config.backendHints.size(),
                    "Number of Functions must equal number of backendHints");
  RETURN_ERR_IF_NOT(funcs.size() == config.replicationCounts.size(),
                    "Number of Functions must equal");
  RETURN_ERR_IF_NOT(
      funcs.size() == config.backendNames.size() || config.backendNames.empty(),
      "If there are backendNames specified, there must be one per Function");
  for (size_t i = 0, e = funcs.size(); i < e; i++) {
    Function *F = funcs[i];
    partitionMap.setBackendSpecificOpts(F, config.backendSpecificOpts[i]);
    partitionMap.setBackendHints(F, config.backendHints[i]);
    partitionMap.addReplicationCount(F, config.replicationCounts[i]);
    if (!config.backendNames.empty()) {
      RETURN_ERR_IF_NOT(backendName == config.backendNames[i],
                        "Mismatch on backendName for partition");
    }
  }

  // Do partition.
  DAGListTy partitions = doPartitioning(
      config.funcName, funcs, module_, partitionMap,
      /* saveDAG */ true, cctx.backendOpts.backendSpecificNodeInfo,
      /* skipCloning */ true);

  // DAG validation.
  RETURN_IF_ERR(dagValidation(partitions[0]));

  // Verify the function.
  for (Function *F : funcs) {
    DCHECK(F->verify()) << "Conversion led to invalid function";
  }

  RETURN_IF_ERR(finalize(partitions, partitionMap));

  if (cctx.saturateHost) {
    // Use the config's logical IDs to determine how many cards it's using.
    llvm::SmallSet<DeviceIDTy, 6> allLogicalIDs;
    for (const auto &IDs : config.logicalIDs) {
      for (const auto &id : IDs) {
        allLogicalIDs.insert(id);
      }
    }
    saturateHost(allLogicalIDs.size(), partitions, cctx.saturateKDevices);
  }

  return std::move(partitions);
}

// Do a search starting at an SLS node to split any concats/tanh that will
// be included the SLS partition
static void splitConcatTanhFromNode(Function *F, Node *node,
                                    int concatSplitSize,
                                    const KindSet &pairSLSWithNodeKinds,
                                    bool concatTanhSinkApplied) {
  auto users = node->getUsers();
  for (auto &j : users) {
    Node *user = j.getUser();
    auto shouldPairWithSls = pairSLSWithNodeKinds.count(user->getKind());
    if (!shouldPairWithSls) {
      continue;
    }

    if (auto *CN = llvm::dyn_cast<ConcatNode>(user)) {
      if (concatTanhSinkApplied) {
        auto concatUsers = CN->getUsers();
        // Skip splitting concats which don't go into a tanh sink or are small
        if (concatUsers.empty() ||
            concatUsers.begin()->getUser()->getKind() !=
                glow::Kinded::Kind::TanhNodeKind ||
            CN->getNumInputs() <= concatSplitSize) {
          continue;
        }
        auto tanhNode =
            llvm::dyn_cast<TanhNode>(concatUsers.begin()->getUser());
        auto dim = CN->getDim();
        // Split the concat into smaller concats and create a tanh sink for each
        // split
        std::vector<NodeValue> concats;
        for (size_t i = 0, n = CN->getNumInputs(); i < n;
             i += concatSplitSize) {
          auto begin = CN->getInputs().begin() + i;
          auto length = i + concatSplitSize < n ? concatSplitSize : n - i;

          std::vector<NodeValue> concatInputs(begin, begin + length);
          auto *concat = F->createConcat(CN->getName().str() + "_part_" +
                                             std::to_string(i / n),
                                         concatInputs, dim);
          auto *tanh = F->createTanh(CN->getName().str() + "_tanh_part_" +
                                         std::to_string(i / n),
                                     concat->getResult());
          concats.emplace_back(tanh->getResult());
        }
        // Combine split up concats
        auto *newConcat =
            F->createConcat(CN->getName().str() + "_combined", concats, dim);
        tanhNode->getResult().replaceAllUsesOfWith(newConcat->getResult());
        F->eraseNode(CN);
        F->eraseNode(tanhNode);
      } else {
        // Skip splitting concats which don't have all tanh inputs or are small
        if (!checkNodeInputsAllKind(user, glow::Kinded::Kind::TanhNodeKind) ||
            CN->getNumInputs() <= concatSplitSize) {
          continue;
        }
        auto dim = CN->getDim();
        // Split the concat into smaller concats
        std::vector<NodeValue> concats;
        for (size_t i = 0, n = CN->getNumInputs(); i < n;
             i += concatSplitSize) {
          auto begin = CN->getInputs().begin() + i;
          auto length = i + concatSplitSize < n ? concatSplitSize : n - i;

          std::vector<NodeValue> concatInputs(begin, begin + length);
          auto *concat = F->createConcat(CN->getName().str() + "_part_" +
                                             std::to_string(i / n),
                                         concatInputs, dim);
          concats.emplace_back(concat->getResult());
        }
        // Combine split-up concats
        auto *newConcat =
            F->createConcat(CN->getName().str() + "_combined", concats, dim);
        CN->getResult().replaceAllUsesOfWith(newConcat->getResult());
        F->eraseNode(CN);
      }
    } else {
      splitConcatTanhFromNode(F, user, concatSplitSize, pairSLSWithNodeKinds,
                              concatTanhSinkApplied);
    }
  }
}

static void splitConcatTanh(Function *F, int concatSplitSize,
                            std::vector<std::string> pairSLSWith,
                            bool concatTanhSinkApplied) {
  const std::unordered_map<std::string, glow::Kinded::Kind> nameToNodeKind = {
      {"Concat", glow::Kinded::Kind::ConcatNodeKind},
      {"LayerNorm", glow::Kinded::Kind::LayerNormalizationNodeKind},
      {"Tile", glow::Kinded::Kind::TileNodeKind},
      {"Tanh", glow::Kinded::Kind::TanhNodeKind}};
  for (auto &node : F->getNodes()) {
    switch (node.getKind()) {

#define SPLIT_CONCAT_TANH_CASE(NODE_NAME_)                                     \
  case Kinded::Kind::NODE_NAME_##Kind: {                                       \
    auto SLS = llvm::cast<NODE_NAME_>(&node);                                  \
    KindSet pairSLSWithNodeKinds;                                              \
    for (auto &s : pairSLSWith) {                                              \
      if (nameToNodeKind.find(s) == nameToNodeKind.end() ||                    \
          pairSLSWithNodeKinds.count(nameToNodeKind.at(s))) {                  \
        continue;                                                              \
      }                                                                        \
      if (s == "Tile") {                                                       \
        if (SLS->getResult().dims()[0] == 1) {                                 \
          pairSLSWithNodeKinds.insert(nameToNodeKind.at(s));                   \
        }                                                                      \
      } else {                                                                 \
        pairSLSWithNodeKinds.insert(nameToNodeKind.at(s));                     \
      }                                                                        \
    }                                                                          \
    splitConcatTanhFromNode(F, SLS, concatSplitSize, pairSLSWithNodeKinds,     \
                            concatTanhSinkApplied);                            \
  }                                                                            \
    continue;

      SPLIT_CONCAT_TANH_CASE(FusedRowwiseQuantizedSparseLengthsWeightedSumNode);
      SPLIT_CONCAT_TANH_CASE(FusedRowwiseQuantizedSparseLengthsSumNode);
      SPLIT_CONCAT_TANH_CASE(RowwiseQuantizedSparseLengthsWeightedSumNode);
      SPLIT_CONCAT_TANH_CASE(SparseLengthsSumNode);
      SPLIT_CONCAT_TANH_CASE(SparseLengthsWeightedSumNode);
      SPLIT_CONCAT_TANH_CASE(EmbeddingBagNode);
      SPLIT_CONCAT_TANH_CASE(EmbeddingBagByteRowwiseOffsetsNode);
#undef SPLIT_CONCAT_TANH_CASE

    default:
      continue;
    }
  }
}

// Do a search starting at an SLS output to capture any Clip,
// LayerNormalization, Tile, Tanh nodes which are there
static void
expandFrontier(Node *node, const NodeValue &value,
               std::unordered_set<NodeValue> &frontier,
               std::unordered_set<Node *> &traversedNodes,
               const std::map<glow::Kinded::Kind, size_t> &pairSlsWithNodeKinds,
               bool concatTanhSinkApplied) {
  traversedNodes.insert(node);
  bool covered = true;
  auto users = node->getUsers();
  for (auto j = users.begin(), f = users.end(); j != f; ++j) {
    Node *user = (*j).getUser();
    if (ClipNode *CN = llvm::dyn_cast<ClipNode>(user)) {
      expandFrontier(user, CN->getResult(), frontier, traversedNodes,
                     pairSlsWithNodeKinds, concatTanhSinkApplied);
    } else {
      auto it = pairSlsWithNodeKinds.find(user->getKind());
      if (it != pairSlsWithNodeKinds.end()) {

        if (it->first == glow::Kinded::Kind::ConcatNodeKind) {
          auto concatUsers = user->getUsers();
          // If tanh sink was applied, only include concats which go into tanh
          // sink
          if (concatTanhSinkApplied && !concatUsers.empty() &&
              concatUsers.begin()->getUser()->getKind() ==
                  glow::Kinded::Kind::TanhNodeKind) {
            expandFrontier(user, user->getNthResult(it->second), frontier,
                           traversedNodes, pairSlsWithNodeKinds,
                           concatTanhSinkApplied);
          }
          // If tanh sink was not applied, only include concats whose inputs are
          // all tanh
          else if (!concatTanhSinkApplied &&
                   checkNodeInputsAllKind(user,
                                          glow::Kinded::Kind::TanhNodeKind)) {
            expandFrontier(user, user->getNthResult(it->second), frontier,
                           traversedNodes, pairSlsWithNodeKinds,
                           concatTanhSinkApplied);
          } else {
            covered = false;
          }
        } else {
          expandFrontier(user, user->getNthResult(it->second), frontier,
                         traversedNodes, pairSlsWithNodeKinds,
                         concatTanhSinkApplied);
        }
      } else {
        covered = false;
      }
    }
  }
  if (!covered) {
    frontier.insert(value);
  }
}

/// Helper function for SparseNN Partitioning scheme. Checks for each
/// kind of SLS table and appends their metadata to the vector.
template <typename SLSType>
static Error appendSLSTable(SLSType *SLS, std::vector<SLSTableInfo> &slsTables,
                            bool doPerfModelBalance, Backend *backend,
                            const std::vector<std::string> &pairSLSWith,
                            bool concatTanhSinkApplied) {
  uint64_t cost = 1;
  uint64_t numBytesInTable =
      (uint64_t)SLS->getData().getType()->getSizeInBytes();

  // If average length is available, then compute cost using perf model
  if (doPerfModelBalance) {
    double cost_d;
    ASSIGN_VALUE_OR_RETURN_ERR(cost_d, backend->estimateNodeCost(SLS));
    cost = (uint64_t)cost_d;
  }
  auto slsResult = SLS->getResult();
  const std::unordered_map<std::string, std::pair<glow::Kinded::Kind, size_t>>
      nameToNodeKind{
          {"Concat",
           {glow::Kinded::Kind::ConcatNodeKind, ConcatNode::ResultIdx}},
          {"LayerNorm",
           {glow::Kinded::Kind::LayerNormalizationNodeKind,
            LayerNormalizationNode::ResultIdx}},
          {"Tile", {glow::Kinded::Kind::TileNodeKind, TileNode::ResultIdx}},
          {"Tanh", {glow::Kinded::Kind::TanhNodeKind, TanhNode::ResultIdx}},
      };
  std::map<glow::Kinded::Kind, size_t> pairSlsWithNodeKinds;
  for (auto &s : pairSLSWith) {
    if (nameToNodeKind.find(s) == nameToNodeKind.end() ||
        pairSlsWithNodeKinds.find(nameToNodeKind.at(s).first) !=
            pairSlsWithNodeKinds.end()) {
      continue;
    }
    // Only expand SLS w/ tile for user embeddings
    if (s == "Tile") {
      // The first dimension = 1 corresponds to user embeddings, so we expand w/
      // Tile
      if (slsResult.dims()[0] == 1) {
        pairSlsWithNodeKinds.insert(nameToNodeKind.at(s));
      }
    } else {
      pairSlsWithNodeKinds.insert(nameToNodeKind.at(s));
    }
  }
  std::unordered_set<NodeValue> frontier;
  std::unordered_set<Node *> neighbors;
  expandFrontier(SLS, slsResult, frontier, neighbors, pairSlsWithNodeKinds,
                 concatTanhSinkApplied);

  // neighbors contains only successors; add all predecessors too.
  std::unordered_set<Node *> addedSLSNeighbors;
  std::queue<Node *> preds;
  for (auto *N : neighbors) {
    preds.push(N);
  }
  preds.push(SLS);
  auto hasConcat = pairSlsWithNodeKinds.find(Kinded::Kind::ConcatNodeKind) !=
                   pairSlsWithNodeKinds.end();
  while (!preds.empty()) {
    auto *cur = preds.front();
    if (cur != SLS) {
      neighbors.insert(cur);
      // Sum up the total sizes of SLS nodes under the same concat since they'll
      // all be in the same partition
      if (hasConcat && isSLSNode(cur) &&
          addedSLSNeighbors.find(cur) == addedSLSNeighbors.end()) {
        addedSLSNeighbors.insert(cur);
        switch (cur->getKind()) {
#define ADD_SLS_NB_NODE_SIZE_CASE(NODE_NAME_)                                  \
  case Kinded::Kind::NODE_NAME_##Kind: {                                       \
    auto SLS = llvm::cast<NODE_NAME_>(cur);                                    \
    numBytesInTable += (uint64_t)SLS->getData().getType()->getSizeInBytes();   \
  }                                                                            \
    continue;

          ADD_SLS_NB_NODE_SIZE_CASE(
              FusedRowwiseQuantizedSparseLengthsWeightedSumNode);
          ADD_SLS_NB_NODE_SIZE_CASE(FusedRowwiseQuantizedSparseLengthsSumNode);
          ADD_SLS_NB_NODE_SIZE_CASE(
              RowwiseQuantizedSparseLengthsWeightedSumNode);
          ADD_SLS_NB_NODE_SIZE_CASE(SparseLengthsSumNode);
          ADD_SLS_NB_NODE_SIZE_CASE(SparseLengthsWeightedSumNode);
          ADD_SLS_NB_NODE_SIZE_CASE(EmbeddingBagNode);
          ADD_SLS_NB_NODE_SIZE_CASE(EmbeddingBagByteRowwiseOffsetsNode);
#undef ADD_SLS_NB_NODE_SIZE_CASE
        default:
          continue;
        }
      }
    }
    preds.pop();
    for (auto *N : getInputs(cur)) {
      preds.push(N);
    }
  }

  slsTables.push_back(
      {SLS, neighbors, frontier, numBytesInTable, 0, slsResult, cost});
  return Error::success();
}

// Check if the input for \p targetNode is a SplatNode with more than one
// user, and if so clone the splat node into \p F and set it to be the new
// input of \p targetNode.
static void cloneSplatInputIfNecessary(Node *targetNode, Function *F) {
  for (int inp = 0, e = targetNode->getNumInputs(); inp < e; inp++) {
    auto input = targetNode->getNthInput(inp);
    SplatNode *splatInput = llvm::dyn_cast<SplatNode>(input.getNode());
    if (!splatInput || splatInput->getNumUsers() <= 1) {
      continue;
    }
    SplatNode *splatInputClone =
        F->addNode(llvm::cast<SplatNode>(splatInput->clone()));
    targetNode->setNthInput(inp, splatInputClone->getResult());
  }
}

// Insert Split->Concat at barrier between SLS and Non-SLS partitions
static Error
sparseNNInsertSplitConcat(Function *F,
                          std::vector<std::unordered_set<NodeValue>> &frontiers,
                          PartitionConfig &partitionConfig) {

  // Walk through SLS tables and check that all the results are able to concat
  std::vector<std::vector<NodeValue>> concatInputs(frontiers.size());
  // Insert concat and slice nodes and assign them to partitions
  for (size_t p = 0; p < frontiers.size(); p++) {
    auto &frontier = frontiers[p];

    if (frontier.size() == 0) {
      continue;
    }
    auto &templateResult = *frontier.begin();
    auto templateDims = templateResult.dims();
    auto templateConcatDim = templateDims.size() - 1;

    for (auto &tableResult : frontier) {
      auto tableDims = tableResult.dims();
      RETURN_ERR_IF_NOT(tableDims.size() == templateDims.size(),
                        strFormat("SLS concat addition encountered tensors "
                                  "with differing dimensions (%zu vs %zu)",
                                  (size_t)tableDims.size(),
                                  (size_t)templateDims.size()));
      for (dim_t otherDim = 0; otherDim < templateConcatDim; otherDim++) {
        RETURN_ERR_IF_NOT(tableDims[otherDim] == templateDims[otherDim],
                          strFormat("SLS concat addition encountered tensors "
                                    "with differing dimension (%zu vs %zu)",
                                    (size_t)tableDims[otherDim],
                                    (size_t)templateDims[otherDim]));
      }
      RETURN_ERR_IF_NOT(tableResult.getType()->getElementType() ==
                            templateResult.getType()->getElementType(),
                        "SLS concat addition encountered tensors with "
                        "differing ElementType");
      concatInputs[p].push_back(tableResult);
    }

    if (concatInputs[p].size() > 1) {

      // Insert concat
      auto *deviceConcat = F->createConcat("concat_dev_" + std::to_string(p),
                                           concatInputs[p], templateConcatDim);
      partitionConfig.nodeToPartition[deviceConcat->getName()] = p;

      // Insert slices
      std::vector<dim_t> splits(concatInputs[p].size());
      for (dim_t i = 0; i < concatInputs[p].size(); i++) {
        auto inputDim = concatInputs[p][i].dims();
        splits[i] = inputDim[templateConcatDim];
      }
      std::vector<SliceNode *> splitOutputs;
      F->createSplit("split_dev" + std::to_string(p), deviceConcat,
                     splits.size(), templateConcatDim, splits, splitOutputs);
      for (dim_t i = 0; i < concatInputs[p].size(); i++) {
        assert(i < splitOutputs.size());
        concatInputs[p][i].replaceAllUsesOfWith(splitOutputs[i]);
        deviceConcat->setNthInput(i, concatInputs[p][i]);
        partitionConfig.nodeToPartition[splitOutputs[i]->getName()] =
            partitionConfig.numOfPartitions - 1;
      }
    }
  }
  return Error::success();
};

Expected<DAGListTy> Partitioner::partitionSparseNN(CompilationContext &cctx) {
  VLOG(1) << "Doing SparseNN partitioning" << std::endl;
  PartitionConfig partitionConfig;
  partitionConfig.numOfPartitions = 0;

  // Find the first partition with an SLS node
  Function *F = nullptr;
  for (Function *currF : module_->getFunctions()) {
    for (auto &node : currF->getNodes()) {
      if (node.getKind() ==
              glow::Kinded::Kind::
                  FusedRowwiseQuantizedSparseLengthsWeightedSumNodeKind ||
          node.getKind() == glow::Kinded::Kind::
                                FusedRowwiseQuantizedSparseLengthsSumNodeKind ||
          node.getKind() ==
              glow::Kinded::Kind::
                  RowwiseQuantizedSparseLengthsWeightedSumNodeKind ||
          node.getKind() == glow::Kinded::Kind::SparseLengthsSumNodeKind ||
          node.getKind() ==
              glow::Kinded::Kind::SparseLengthsWeightedSumNodeKind ||
          node.getKind() == glow::Kinded::Kind::EmbeddingBagNodeKind ||
          node.getKind() ==
              glow::Kinded::Kind::EmbeddingBagByteRowwiseOffsetsNodeKind) {
        F = currF;
        break;
      }
    }
    if (F) {
      break;
    }
  }

  // If no matching functions then return empty config
  if (!F) {
    return MAKE_ERR(ErrorValue::ErrorCode::PARTITIONER_ERROR,
                    "Did not find a partition with an SLS node");
  }

  if (deviceInfo_.size() <
      cctx.optimizationOpts.sparseNNPartitioningSchemeNumCards) {
    return MAKE_ERR(
        ErrorValue::ErrorCode::PARTITIONER_ERROR,
        strFormat("Not enough devices to partition. Num Devices is %zu and Num "
                  "SparseNN Cards Needed is %u",
                  deviceInfo_.size(),
                  cctx.optimizationOpts.sparseNNPartitioningSchemeNumCards));
  }

  // Otherwise partition this function
  partitionConfig.funcName = F->getName().str();

  // First optimize the function
  std::vector<Backend *> backends;
  genBackendMap(backendMap_, backendHolder_, backends);
  // First optimize it
  if (!optimized_) {
    RETURN_IF_ERR(::glow::optimizeFunction(F, *(backends[0]), cctx));
  }

  // Now we may want to duplicate Splat input nodes in case they have been
  // CSE'd (CSE stands for common subexpression elimination) into a single
  // SplatNode. This is because if two SLWS that share Splat input nodes are
  // separated to two partitions, then partitioning will force a dependence
  // from whichever partition the input node are placed to the other
  // partition. After partitioning when we optimize each partition
  // individually, they may be merged again inside the partition. Besides, the
  // potential partition dependency introduced might lead to a circular
  // dependency in the final graph.
  //
  // We fix this issue by iterating over the Function and finding Splat input
  // nodes with multiple users and just creating new Splats (by cloning) for
  // each user.
  for (auto &node : F->getNodes()) {
    cloneSplatInputIfNecessary(&node, F);
  }

  // Create list of SLS Tables
  std::vector<SLSTableInfo> slsTables;
  partitionConfig.funcName = std::string(F->getName());
  VLOG(1) << "Function: " << std::string(F->getName()) << std::endl;

  std::vector<std::string> pairSLSWith;
  folly::split<char, std::string, std::string>(
      ',', cctx.optimizationOpts.sparseNNPartitioningPairSLSWith, pairSLSWith,
      /*ignoreEmpty*/ true);
  if (cctx.optimizationOpts.sparseNNPartitioningPairTileWithSLS) {
    pairSLSWith.emplace_back("Tile");
  }
  if (cctx.optimizationOpts.sparseNNPartitioningPairLNWithSLS) {
    pairSLSWith.emplace_back("LayerNorm");
  }
  bool concatTanhSinkApplied = cctx.optimizationOpts.sinkTanhBelowConcat;
  if (std::find(pairSLSWith.begin(), pairSLSWith.end(), "Concat") !=
      pairSLSWith.end()) {
    auto splitConcatSize =
        cctx.optimizationOpts.sparseNNPartitioningConcatSplitSize;
    splitConcatTanh(F, splitConcatSize, pairSLSWith, concatTanhSinkApplied);
  }
  const bool doPerfModelBalance =
      cctx.optimizationOpts.sparseNNPartitioningBalancePerfModel;
  size_t totalSLSTableSizes = 0;
  for (auto &node : F->getNodes()) {
    switch (node.getKind()) {

#define APPEND_TABLE_CASE(NODE_NAME_)                                          \
  case Kinded::Kind::NODE_NAME_##Kind:                                         \
    RETURN_IF_ERR(appendSLSTable<NODE_NAME_>(                                  \
        llvm::cast<NODE_NAME_>(&node), slsTables, doPerfModelBalance,          \
        backends[0], pairSLSWith, concatTanhSinkApplied));                     \
    totalSLSTableSizes += slsTables.back().numBytesInTable;                    \
    continue;

      APPEND_TABLE_CASE(FusedRowwiseQuantizedSparseLengthsWeightedSumNode);
      APPEND_TABLE_CASE(FusedRowwiseQuantizedSparseLengthsSumNode);
      APPEND_TABLE_CASE(RowwiseQuantizedSparseLengthsWeightedSumNode);
      APPEND_TABLE_CASE(SparseLengthsSumNode);
      APPEND_TABLE_CASE(SparseLengthsWeightedSumNode);
      APPEND_TABLE_CASE(EmbeddingBagNode);
      APPEND_TABLE_CASE(EmbeddingBagByteRowwiseOffsetsNode);
#undef APPEND_TABLE_CASE

    default:
      continue;
    }
  }
  LOG(INFO) << "Total size of all " << slsTables.size()
            << " SLS embedding tables: " << totalSLSTableSizes;

  // Now determine all nodes that fit in the NonSLS partition, so we know its
  // total size and can better judge how much space is left for SLS
  // partitions.
  std::unordered_set<const Node *> slsPartitionNodes;
  for (auto &slsTable : slsTables) {
    slsPartitionNodes.insert(slsTable.node);
    for (const Node *N : slsTable.neighbors) {
      slsPartitionNodes.insert(N);
    }
  }

  NodesSet nonSLSPartitionNodes;
  for (auto &node : F->getNodes()) {
    if (!slsPartitionNodes.count(&node)) {
      nonSLSPartitionNodes.insert(&node);
    }
  }

  // Calculate how much space the NonSLS partition takes up, and compare that
  // to how much memory the device has to determine the allows SLS partition
  // size.
  const uint64_t nonSLSPartitionSize =
      getGraphMemInfo(nonSLSPartitionNodes, contextCount_).getTotalMemSize();
  const uint64_t totalDeviceMemory = deviceInfo_[0].availableMemory;
  RETURN_ERR_IF_NOT(nonSLSPartitionSize < totalDeviceMemory,
                    strFormat("nonSLSPartitionSize %lu must be less than %s "
                              "totalDeviceMemory %lu",
                              nonSLSPartitionSize,
                              deviceInfo_[0].backendName.c_str(),
                              totalDeviceMemory));
  const uint64_t allowedSLSMemBytes = totalDeviceMemory - nonSLSPartitionSize;

  // Create table of devices
  std::vector<SLSDeviceInfo> slsDevices;
  std::vector<std::unordered_set<NodeValue>> frontierValues;
  unsigned int snnNumCards =
      cctx.optimizationOpts.sparseNNPartitioningSchemeNumCards;

  LOG(INFO) << "totalDeviceMemory=" << totalDeviceMemory
            << ", nonSLSPartitionSize=" << nonSLSPartitionSize
            << ", allowedSLSMemBytes=" << allowedSLSMemBytes
            << ", snnNumCards=" << snnNumCards;

  bool partitionSucceeded = false;
  std::vector<unsigned int> factors;
  factors.push_back(snnNumCards);
  for (unsigned int i = snnNumCards + 1, e = deviceInfo_.size(); i <= e; ++i) {
    if (deviceInfo_.size() % i == 0) {
      factors.push_back(i);
    }
  }
  auto it = std::lower_bound(factors.begin(), factors.end(), snnNumCards);
  for (unsigned i = std::distance(factors.begin(), it); i < factors.size();
       i++) {
    snnNumCards = factors[i];
    LOG(INFO) << "Trying " << snnNumCards << " sparse partitions.";
    // Reset some of the contexts.
    slsDevices.clear();
    for (unsigned int device = 0; device < snnNumCards; device++) {
      slsDevices.push_back({device, allowedSLSMemBytes, 0});
    }
    frontierValues.clear();
    frontierValues.resize(slsDevices.size());

    // Now assign SLS Nodes to devices
    if (ERR_TO_BOOL(assignSlsTablesToDevices(slsTables, slsDevices,
                                             frontierValues, contextCount_))) {
      LOG(INFO) << "Failed to partition SLS tables, fall back to greedy "
                   "algorithm.";
      if (!ERR_TO_BOOL(assignSlsTablesToDevicesGreedy(
              slsTables, slsDevices, frontierValues, contextCount_))) {
        partitionSucceeded = true;
      };
    } else {
      partitionSucceeded = true;
    }

    if (partitionSucceeded) {
      LOG(INFO) << "Successfully got a SparseNN partition solution with "
                << snnNumCards << " sparse partitions.";
      break;
    } else {
      LOG(WARNING) << "Cannot find a valid SparseNN partition solution with "
                   << snnNumCards << " sparse partitions.";
    }
  }

  if (!partitionSucceeded) {
    return MAKE_ERR(ErrorValue::ErrorCode::PARTITIONER_ERROR,
                    "SLS Balancing Partitioning Error: Not enough memory");
  }

  VLOG(1) << "Final table assignments: ";
  printSlsTableInfo(slsTables);

  // Fill up the last partition with NonSLS nodes.
  for (auto *node : nonSLSPartitionNodes) {
    partitionConfig.nodeToPartition[node->getName()] = snnNumCards;
  }

  // Create manual partition
  partitionConfig.numOfPartitions = slsDevices.size() + 1;
  std::vector<unsigned int> allLogicalIDs;

  // Add SLS Partitions
  for (size_t p = 0; p < slsDevices.size(); p++) {
    partitionConfig.partitionNames.push_back(std::string("SLSPartition_") +
                                             std::to_string(p));
    partitionConfig.backendNames.push_back(deviceInfo_[p].backendName);
    partitionConfig.logicalIDs.push_back({(unsigned int)p});
    BackendHints backendHints;
    backendHints.executionUnits =
        cctx.optimizationOpts.sparseNNPartitioningSchemeNumCoresSLS;
    partitionConfig.backendHints.push_back(backendHints);
    allLogicalIDs.push_back(p);
  }

  // Add last partition
  partitionConfig.partitionNames.push_back(std::string("NonSLSPartition_"));
  partitionConfig.backendNames.push_back(deviceInfo_[0].backendName);
  partitionConfig.logicalIDs.push_back(allLogicalIDs);
  BackendHints backendHints;
  backendHints.executionUnits =
      cctx.optimizationOpts.sparseNNPartitioningSchemeNumCoresOther;
  partitionConfig.backendHints.push_back(backendHints);

  // Map SLS nodes to their partitions
  for (auto &table : slsTables) {
    partitionConfig.nodeToPartition[table.node->getName()] = table.deviceId;
    for (Node *N : table.neighbors) {
      partitionConfig.nodeToPartition[N->getName()] = table.deviceId;
    }
  }

  // Insert Split->Concat at barrier between SLS and Non-SLS partitions
  if (cctx.optimizationOpts.sparseNNPartitioningAddSLSConcats) {
    RETURN_IF_ERR(
        sparseNNInsertSplitConcat(F, frontierValues, partitionConfig));
  }

  VLOG(1) << " Finished SparseNN partitioning" << std::endl;
  VLOG(1) << " PartitionConfig ::: funcName = " << partitionConfig.funcName
          << "\n";
  VLOG(1) << " PartitionConfig ::: numOfPartitions = "
          << partitionConfig.numOfPartitions << "\n";
  VLOG(1) << " PartitionConfig ::: partitionNames = ";
  for (unsigned i = 0; i < partitionConfig.numOfPartitions; i++) {
    VLOG(1) << partitionConfig.partitionNames[i] << " ";
  }
  VLOG(1) << "\n";
  VLOG(1) << " PartitionConfig ::: logicalIDs = ";
  for (unsigned i = 0; i < partitionConfig.numOfPartitions; i++) {
    for (auto &id : partitionConfig.logicalIDs[i]) {
      VLOG(1) << id << " ";
    }
    VLOG(1) << "\n";
  }

  DAGListTy partitions;
  ASSIGN_VALUE_OR_RETURN_ERR(partitions,
                             partitionFromConfig(partitionConfig, cctx));
  if (cctx.saturateHost) {
    saturateHost(snnNumCards, partitions, cctx.saturateKDevices);
  }
  return std::move(partitions);
}

Expected<DAGListTy> Partitioner::partition(CompilationContext &cctx) {
  if (cctx.prepartitionedConfig &&
      cctx.prepartitionedConfig->funcs.size() != 0) {
    VLOG(1) << "Using prepartitioned config";
    return setupPrepartitionedModule(cctx);
  }

  if (cctx.partitionConfig) {
    VLOG(1) << "Using partition config";
    partitionConfig_ = *cctx.partitionConfig;
  }

  if (partitionConfig_.enabled()) {
    // Call user-defined partition flow.
    return partitionFromConfig(partitionConfig_, cctx);
  }

  if (!multiBackendNames_ &&
      cctx.optimizationOpts.useSparseNNPartitioningScheme) {
    VLOG(1) << "Using SNN Partition Scheme";
    return partitionSparseNN(cctx);
  }

  if (cctx.precisionConfig.quantMode == QuantizationMode::Profile) {
    // Call quantization profiling partition flow.
    VLOG(1) << "Using QuantProfile Partition";
    return quantizationProfilingPartition(cctx);
  }

  if (!multiBackendNames_ && glow::flags::EnableLoadBalancedPartitioning) {
    // Call load-balance partition flow.
    VLOG(1) << "Using Load balance Partition";
    return loadBalancedPartition(cctx);
  }

  VLOG(1) << "Using Heterogenous Partition";
  // Call heterogeneous partition flow.
  return heterogeneousPartition(cctx);
}
