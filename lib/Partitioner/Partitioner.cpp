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
#include "glow/Optimizer/GraphOptimizer/GraphOptimizer.h"
#include "glow/Partitioner/PartitionerOptimizer.h"
#include "glow/Partitioner/PartitionerUtils.h"
#include "glow/Partitioner/PartitionerValidation.h"
#include "glow/Support/Support.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

#include <fstream>
namespace glow {
bool GlowEnableLoadBalancedPartitioning = false;
static llvm::cl::opt<bool, /* ExternalStorage */ true>
    GlowEnableLoadBalancedPartitioningOpt(
        "glow_partitioner_enable_load_balance",
        llvm::cl::desc(
            "Enable a partitioner pass to optimize for "
            "load balance in addition to memory capacity constraints"),
        llvm::cl::location(GlowEnableLoadBalancedPartitioning));
} // namespace glow

/// -log-partition - Command line option to dump Partitioner logs.
static llvm::cl::OptionCategory PartitionerCat("Glow Partitioner Options");
static llvm::cl::opt<bool>
    logPartition("log-partition",
                 llvm::cl::desc("Enable logging partition info"),
                 llvm::cl::init(false), llvm::cl::cat(PartitionerCat));

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
              << module_->getFunctions().size()
              << ", and the DAG is dumped into DAG.dot file.\n";
    dumpDAG("DAG.dot", partitions);
    logPartitionInfo(mapping);
  }

  // Dump the graph of each function after partitioning.
  if (dumpPartition) {
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
                         const std::vector<Backend *> &backends,
                         bool saturateHost, bool optimized)
    : module_(parent), deviceInfo_(devices), backends_(backends),
      saturateHost_(saturateHost), optimized_(optimized) {
  init();
}

Partitioner::Partitioner(Module *parent, const std::vector<DeviceInfo> &devices,
                         bool saturateHost, bool optimized,
                         PartitionConfig partitionConfig)
    : module_(parent), deviceInfo_(devices), saturateHost_(saturateHost),
      optimized_(optimized), partitionConfig_(partitionConfig) {
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

void Partitioner::saturateHost(unsigned logicalDeviceCount,
                               const DAGListTy &partitions) {
  unsigned duplications = deviceInfo_.size() / logicalDeviceCount;
  if (duplications < 2) {
    return;
  }
  // Add additional logical devices to each node.
  for (auto &network : partitions) {
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
  return doPartitioning(F->getName(), funcs, module_, mapping, genDAG);
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
      auto backend = backendMap[backendName].backend;
      RETURN_IF_ERR(::glow::optimizeFunction(
          F, *backend, cctx, &getDeviceInfoForBackend(backendName)));
    }
    std::unique_ptr<DAGNode> DAG0 = glow::make_unique<DAGNode>();
    DAG0->logicalDevices = {logDevice};
    DAG0->name = F->getName();
    DAG0->module = module_;
    std::unique_ptr<DAGNode> DAG1 = glow::make_unique<DAGNode>();
    DAG1->logicalDevices = {logDevice};
    DAG1->name = F->getName();
    DAG1->backendName = backendName;
    DAG1->parents.push_back(DAG0.get());
    DAG0->children.push_back(DAG1.get());
    DAGNodePtrVec nodes;
    nodes.push_back(std::move(DAG1));
    partitions.push_back({std::move(DAG0), std::move(nodes)});
  }
  if (saturateHost_) {
    // Saturate the Host.
    saturateHost(1, partitions);
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
  if (module_->getFunctions().size() != 1) {
    return MAKE_ERR(
        ErrorValue::ErrorCode::PARTITIONER_ERROR,
        strFormat("Invalid : %lu functions in a module. Now in load-balanced "
                  "partition flow, the module can only contain 1 function",
                  module_->getFunctions().size()));
  }

  if (multiBackendNames_) {
    VLOG(1) << "For multi backend types, load-balanced partition can't be "
               "applied. Call heterogeneous partition instead.";
    return heterogeneousPartition(cctx);
  }
  F_ = selectRepFunc(module_, memSize_);
  std::string origName(F_->getName().data());
  DAGListTy partitions;
  std::vector<Backend *> backends;
  genBackendMap(backendMap_, backendHolder, backends);

  // Step 1: Get the minial number of partitions from auto-partition.
  auto backendName = backends[0]->getBackendName();
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

  logicalDeviceID_ = assignLogicalDeviceID(partitionMap, backendMap_);
  RETURN_IF_ERR(logicalDevicesValidation(partitionMap, backendMap_));

  partitions =
      doPartitioning(origName, {F_}, module_, partitionMap, /* saveDAG */ true);
  module_->eraseFunction(F_);

  if (saturateHost_ &&
      partitionMap.getPartitions().size() < deviceInfo_.size()) {
    saturateHost(logicalDeviceID_, partitions);
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
  genBackendMap(backendMap_, backendHolder, backends);
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
  genBackendMap(backendMap_, backendHolder, backends);

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
      doPartitioning(origName, funcs, module_, mapping, /* saveDAG */ true);

  // Step 5 : Post-partition optimization - Adjust the logicalDevice for each
  // DAGNode.
  if (saturateHost_ && backends.size() == 1 &&
      mapping.getPartitions().size() < deviceInfo_.size()) {
    // Attempt to saturate the host when there is only one type of backend.
    // Passing in the count of logical devices. Since logicalId starts at 0 we
    // add one.
    saturateHost(logicalDeviceID_, partitions);
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
  genBackendMap(backendMap_, backendHolder, backends);
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
    DCHECK(unused.size() == 1) << "There must be exactly 1 unused partition.";
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
    GraphMemInfo cost = getGraphMemInfo(nodesSets[i]);
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
  RETURN_IF_ERR(logicalDevicesValidation(partitionMap, backendMap_));

  // Do partition.
  partitions = doPartitioning(F->getName(), {F}, module_, partitionMap,
                              /* saveDAG */ true);
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

template <typename SLSType>
void Partitioner::appendSLSTable(
    Node &node, std::vector<Partitioner::SLSTableInfo> &slsTables) {
  auto *SLS0 = llvm::dyn_cast<SLSType>(&node);
  if (SLS0) {
    auto SLSDataDims = SLS0->getData().dims();
    auto SLSIndexDims = SLS0->getIndices().dims();
    size_t numElementsPerRowUpperBound =
        std::max((size_t)SLSDataDims[1], (size_t)0);
    size_t numIndices = SLSIndexDims[0];
    uint64_t numBytesInTable =
        (uint64_t)SLSDataDims[0] * (uint64_t)SLSDataDims[1];
    slsTables.push_back(
        {SLS0, numBytesInTable, numElementsPerRowUpperBound, numIndices, 0});
  }
}

// Check if the weights input for \p SLWS is a SplatNode with more than one
// user, and if so clone the splat node into \p F and set it to be the new
// weights of \p SLWS.
template <class T>
static void cloneSplatWeightsIfNecessary(T *SLWS, Function *F) {
  SplatNode *splatWeights = llvm::dyn_cast<SplatNode>(SLWS->getWeights());
  if (!splatWeights || splatWeights->getNumUsers() <= 1) {
    return;
  }
  SplatNode *splatWeightsClone =
      F->addNode(llvm::cast<SplatNode>(splatWeights->clone()));
  SLWS->setNthInput(T::WeightsIdx, splatWeightsClone->getResult());
}

Expected<DAGListTy> Partitioner::partitionSparseNN(CompilationContext &cctx) {

  VLOG(1) << "Doing SparseNN partitioning" << std::endl;
  PartitionConfig partitionConfig;
  partitionConfig.numOfPartitions = 0;

  // Find the first partition with an SLS node
  std::string funcName;
  bool foundFunction = false;
  for (Function *F : module_->getFunctions()) {
    for (auto &node : F->getNodes()) {
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
              glow::Kinded::Kind::SparseLengthsWeightedSumNodeKind) {
        funcName = std::string(F->getName());
        foundFunction = true;
        break;
      }
    }
    if (foundFunction) {
      break;
    }
  }

  // If no matching functions then return empty config
  if (!foundFunction) {
    return MAKE_ERR(ErrorValue::ErrorCode::PARTITIONER_ERROR,
                    "Did not find a partition with an SLS node");
  }

  if (deviceInfo_.size() == 0) {
    return MAKE_ERR(ErrorValue::ErrorCode::PARTITIONER_ERROR,
                    "Not enough devices to partition");
  }

  // Otherwise partition this function
  partitionConfig.funcName = funcName;

  // First optimize the function
  Function *F = module_->getFunction(funcName);
  std::vector<Backend *> backends;
  genBackendMap(backendMap_, backendHolder, backends);
  // First optimize it
  if (!optimized_) {
    RETURN_IF_ERR(::glow::optimizeFunction(F, *(backends[0]), cctx));
  }

  // Now we may want to duplicate Splat weights in case they have been CSE'd
  // into a single SplatNode. This is because if two SLWS that share weights are
  // separated to two partitions, then partitioning will force a dependence from
  // whichever partition the weights are placed to the other partition. After
  // partitioning when we optimize each partition individually, they may be
  // merged again inside the partition.
  for (auto &node : F->getNodes()) {
    if (auto *SLWS = llvm::dyn_cast<SparseLengthsWeightedSumNode>(&node)) {
      cloneSplatWeightsIfNecessary(SLWS, F);
    } else if (auto *SLWS = llvm::dyn_cast<
                   FusedRowwiseQuantizedSparseLengthsWeightedSumNode>(&node)) {
      cloneSplatWeightsIfNecessary(SLWS, F);
    } else if (auto *SLWS =
                   llvm::dyn_cast<RowwiseQuantizedSparseLengthsWeightedSumNode>(
                       &node)) {
      cloneSplatWeightsIfNecessary(SLWS, F);
    }
  }

  // Create list of SLS Tables
  std::vector<SLSTableInfo> slsTables;
  partitionConfig.funcName = std::string(F->getName());
  VLOG(1) << "Function: " << std::string(F->getName()) << std::endl;
  for (auto &node : F->getNodes()) {
    appendSLSTable<FusedRowwiseQuantizedSparseLengthsWeightedSumNode>(
        node, slsTables);
    appendSLSTable<FusedRowwiseQuantizedSparseLengthsSumNode>(node, slsTables);
    appendSLSTable<RowwiseQuantizedSparseLengthsWeightedSumNode>(node,
                                                                 slsTables);
    appendSLSTable<SparseLengthsSumNode>(node, slsTables);
    appendSLSTable<SparseLengthsWeightedSumNode>(node, slsTables);
  }

  // Now sort SLS tables by size decreasing
  VLOG(1) << "SLS tables sorted by size decreasing" << std::endl;
  std::sort(slsTables.begin(), slsTables.end(),
            [](const SLSTableInfo &l, const SLSTableInfo &r) {
              return l.numBytesInTable > r.numBytesInTable;
            });

  // Print SLS tables
  for (auto &table : slsTables) {
    VLOG(1) << "(numBytesInTable, numElementsPerRowUpperBound, numIndices, "
               "deviceId)"
            << "\t" << table.numBytesInTable << "\t"
            << table.numElementsPerRowUpperBound << "\t" << table.numIndices
            << "\t" << table.deviceId << std::endl;
  }

  // Create table of devices
  std::vector<SLSDeviceInfo> slsDevices;
  for (unsigned int device = 0;
       device < cctx.optimizationOpts.sparseNNPartitioningSchemeNumCards;
       device++) {
    slsDevices.push_back(
        {device,
         (uint64_t)(
             (uint64_t)1024 *
             (uint64_t)(cctx.optimizationOpts
                            .sparseNNPartitioningSchemeSLSTableKBytesPerCard)),
         0});
  }

  // Now assign SLS Nodes to devices
  for (auto &table : slsTables) {

    // Sort by cost increasing
    std::sort(slsDevices.begin(), slsDevices.end(),
              [](const SLSDeviceInfo &l, const SLSDeviceInfo &r) {
                return l.currentCost < r.currentCost;
              });

    VLOG(1) << "Devices sorted by cost increasing" << std::endl;
    for (auto &device : slsDevices) {
      VLOG(1) << "(deviceId, memAvailableInBytes, currentCost): "
              << device.deviceId << "\t" << device.memAvailableInBytes << "\t"
              << device.currentCost << std::endl;
    }

    // Pick the first that fits
    bool deviceFound = false;
    for (unsigned int d = 0; d < slsDevices.size(); d++) {
      if (slsDevices[d].memAvailableInBytes >= table.numBytesInTable) {
        deviceFound = true;
        slsDevices[d].memAvailableInBytes -= table.numBytesInTable;
        slsDevices[d].currentCost += 1; // Cost is 1 per table
        table.deviceId = slsDevices[d].deviceId;
        break;
      }
    }
    if (!deviceFound) {
      return MAKE_ERR(ErrorValue::ErrorCode::PARTITIONER_ERROR,
                      "SLS Balancing Partitioning Error: Not enough memory");
    }
  }

  // Print final device info
  VLOG(1) << "Devices sorted by cost increasing" << std::endl;
  for (auto &device : slsDevices) {
    VLOG(1) << "(deviceId, memAvailableInBytes, currentCost): "
            << device.deviceId << "\t" << device.memAvailableInBytes << "\t"
            << device.currentCost << std::endl;
  }

  // Print assignments
  for (auto &table : slsTables) {
    VLOG(1) << "(numBytesInTable, numElementsPerRow, numIndices, deviceId)"
            << "\t" << table.numBytesInTable << "\t"
            << table.numElementsPerRowUpperBound << "\t" << table.numIndices
            << "\t" << table.deviceId << std::endl;
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
  }

  // For each partition, go through all SLS tables assigned and get all their
  // predecessors into the same partition
  for (size_t p = 0; p < slsDevices.size(); p++) {
    std::queue<Node *> preds;
    for (auto &table : slsTables) {
      if (table.deviceId == p) {
        preds.push(table.node);
      }
    }
    while (!preds.empty()) {
      auto cur = preds.front();
      preds.pop();
      for (auto &node : getInputs(cur)) {
        if (partitionConfig.nodeToPartition.find(node->getName()) ==
            partitionConfig.nodeToPartition.end()) {
          partitionConfig.nodeToPartition[node->getName()] = p;
          preds.push(node);
        }
      }
    }
  }

  // Also, go through all SLS tables and assign any Clip's following them to
  // same partition
  for (size_t p = 0; p < slsDevices.size(); p++) {
    for (auto &table : slsTables) {
      if (table.deviceId == p) {
        auto users = table.node->getUsers();
        for (auto j = users.begin(), f = users.end(); j != f; ++j) {
          const Node *user = (*j).getUser();
          if (user->getKind() == Kinded::Kind::ClipNodeKind) {
            partitionConfig.nodeToPartition[user->getName()] = p;
          }
        }
      }
    }
  }

  // All other nodes go in the last partition
  for (auto &node : F->getNodes()) {
    if (partitionConfig.nodeToPartition.find(node.getName()) ==
        partitionConfig.nodeToPartition.end()) {
      partitionConfig.nodeToPartition[node.getName()] = slsDevices.size();
    }
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

  auto partitions = partitionFromConfig(partitionConfig, cctx);
  if (saturateHost_) {
    saturateHost(cctx.optimizationOpts.sparseNNPartitioningSchemeNumCards,
                 std::move(partitions.get()));
  }
  return partitions;
}

Expected<DAGListTy> Partitioner::partition(CompilationContext &cctx) {
  if (cctx.partitionConfig) {
    partitionConfig_ = *cctx.partitionConfig;
  }

  if (partitionConfig_.enabled()) {
    // Call user-defined partition flow.
    return partitionFromConfig(partitionConfig_, cctx);
  }

  if (!multiBackendNames_ &&
      cctx.optimizationOpts.useSparseNNPartitioningScheme) {
    return partitionSparseNN(cctx);
  }

  if (cctx.precisionConfig.quantMode == QuantizationMode::Profile) {
    // Call quantization profiling partition flow.
    return quantizationProfilingPartition(cctx);
  }

  if (!multiBackendNames_ && glow::GlowEnableLoadBalancedPartitioning) {
    // Call load-balance partition flow.
    return loadBalancedPartition(cctx);
  }

  // Call heterogeneous partition flow.
  return heterogeneousPartition(cctx);
}
