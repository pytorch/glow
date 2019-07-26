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
#include "glow/Partitioner/PartitionerOptimizer.h"
#include "glow/Partitioner/PartitionerUtils.h"
#include "glow/Partitioner/PartitionerValidation.h"
#include "glow/Support/Support.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

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

Partitioner::Partitioner(Module *parent, const std::vector<DeviceInfo> &devices,
                         const std::vector<Backend *> &backends,
                         bool saturateHost, bool optimized)
    : module_(parent), deviceInfo_(devices), backends_(backends),
      saturateHost_(saturateHost), optimized_(optimized) {
  memSize_ = module_->getConstantsSize();
  logicalDeviceID_ = 0;
}

Partitioner::Partitioner(Module *parent, const std::vector<DeviceInfo> &devices,
                         bool saturateHost, bool optimized,
                         PartitionConfig partitionConfig)
    : module_(parent), deviceInfo_(devices), saturateHost_(saturateHost),
      optimized_(optimized), partitionConfig_(partitionConfig) {
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

FunctionToBackendNameMap Partitioner::backendBasedPartition(
    Function *F, std::vector<Backend *> &backends, CompilationContext &cctx) {
  FunctionToBackendNameMap ret;
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
  if (cctx.precisionConfig.quantMode == QuantizationMode::Profile) {
    // When profiling, all the partition backend is assigned to
    // profilingBackend.
    mapping.createPartition(newF, profilingBackend);
    ret[newF] = profilingBackend;
  } else {
    mapping.createPartition(newF, backendName);
    ret[newF] = backendName;
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
          ret[newF] = profilingBackend;
        } else {
          mapping.createPartition(newF, backendName);
          ret[newF] = backendName;
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
  doPartitioning(F->getName(), funcs, mapping, genDAG);

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

llvm::Error Partitioner::loadBalancedPartitioning(Function *F,
                                                  DeviceIDTy numDevices,
                                                  uint64_t availableMemory,
                                                  llvm::StringRef backendName,
                                                  NodeToFunctionMap &mapping) {
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
  std::vector<Function *> partitions(numDevices);

  // Compute total roofline time
  float totalRooflineTime = 0;
  for (auto &n : F->getNodes()) {
    totalRooflineTime +=
        getNodeComputeTime(&n, backendMap_[deviceInfo_[0].backendName]);
  }

  float timePerPartition = totalRooflineTime / numDevices;

  // Get the BFS levels
  NodeToFunctionMap partitionMap;
  Function *newF;
  BFSLevel bfs = getBFSLevel(F);
  size_t level = bfs.size();

  // Create the functions and push them into the mapping
  for (DeviceIDTy curPartition = 0; curPartition < numDevices; curPartition++) {
    std::string funcName =
        std::string(F->getName()) + "_part" + std::to_string(curPartition + 1);
    if (F->getParent()->hasFunction(funcName)) {
      newF = F->getParent()->getFunction(funcName);
      F->getParent()->eraseFunction(newF);
    }
    newF = F->getParent()->createFunction(funcName);
    partitionMap.createPartition(newF, backendName);
    partitionMap.appendLogicalDeviceID(newF, curPartition);
    partitions[curPartition] = newF;
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
      int curPartition = maxLogicalDeviceId;
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
          Function *curF = partitions[curPartition];
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
      RETURN_ERR_IF_NOT(curPartition < numDevices,
                        "Load balance partition error");
    }
  }
  for (int i = 0; i < numDevices; i++) {
    VLOG(1) << "Partition #" << i << " has estimated runtime " << deviceTime[i];
  }

  mapping = partitionMap;
  return llvm::Error::success();
}

llvm::Error Partitioner::QuantizationProfilingPartition(
    CompilationContext &cctx, Function *F, std::vector<Backend *> backends) {
  // Quantization profiling flow is run under CPU backend, so we don't really
  // need the concrete partition. The backendBasedPartition is necessary since
  // we need the mapping between quantized tensor and original tensor.
  FunctionToBackendNameMap funcToBackend;
  funcToBackend = backendBasedPartition(F_, backends, cctx);
  module_->eraseFunction(F_);
  std::unique_ptr<Backend> backend(createBackend(profilingBackend));
  for (Function *subF : module_->getFunctions()) {
    assert(subF->verify() && "Conversion led to invalid function");
    if (!optimized_) {
      RETURN_IF_ERR(::glow::optimizeFunction(subF, *backend, cctx));
    }
  }
  if (logPartition) {
    LOG(INFO)
        << "Profiling a model to be partitioned cross different backends. Each "
           "sub-network will be optimized and run on cpu backend.\n";
  }
  return llvm::Error::success();
}

llvm::Error Partitioner::Partition(CompilationContext &cctx) {
  // Prepare the mapping between BackendName and BackendInfo.
  std::vector<Backend *> backends;
  std::vector<std::unique_ptr<Backend>> backendHolder;
  getBackendMap(backendMap_, backendHolder, backends);

  if (partitionConfig_.enabled()) {
    // Jump into user-defined partition, and skip the following auto partition.
    return PartitionFromConfig();
  }

  // Step 0: Find the representative function for running partitioning
  // algorithm.
  F_ = selectRepFunc(module_, memSize_);

  if (cctx.precisionConfig.quantMode == QuantizationMode::Profile) {
    // Jump into profiling flow, and leave without generating partitions for the
    // backends with same type..
    return QuantizationProfilingPartition(cctx, F_, backends);
  }

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
  } else {
    funcToBackend = backendBasedPartition(F_, backends, cctx);
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

  // Step 4 : Optimization pass to modify results of default partitioner.
  // If load balanced partitioner optimization is enabled, then modify
  // the results of the default partitioner to optimize based on roofline
  // performance.
  if (backends.size() == 1 && glow::GlowEnableLoadBalancedPartitioning) {
    auto backendName = backends[0]->getBackendName();
    size_t numDevices = logicalDeviceID_;
    RETURN_IF_ERR(loadBalancedPartitioning(F_, numDevices,
                                           backendMap_[backendName].memSize,
                                           backendName, mapping));
    // Check if the memory usage meets the device memory limitation.
    RETURN_IF_ERR(memoryUsageValidation(mapping, backendMap_));
    // Check if the number of logical devices is less than the given physical
    // devices.
    RETURN_IF_ERR(logicalDevicesValidation(mapping, backendMap_));
    funcs.clear();
    funcs.push_back(F_);
  }

  // Step 5 : do the real partitioning for the function list.
  doPartitioning(origName, funcs, mapping, true);

  // Step 6 : Post-partition optimization - Adjust the logicalDevice for each
  // DAGNode.
  if (saturateHost_ && backends.size() == 1 &&
      mapping.getPartitions().size() < deviceInfo_.size()) {
    // Attempt to saturate the host when there is only one type of backend.
    // Passing in the count of logical devices. Since logicalId starts at 0 we
    // add one.
    saturateHost(logicalDeviceID_);
  }

  // Step 7 : clean up and verify the generated new functions.
  for (auto i = funcToBackend.begin(); i != funcToBackend.end(); ++i) {
    module_->eraseFunction(i->first);
  }

  auto funcList = module_->getFunctions();
  if (logPartition) {
    LOG(INFO) << "The number of partitions is : " << funcList.size()
              << ", and the DAG is dumped into DAG.dot file.\n";
    dumpDAG("DAG.dot");
  }

  for (Function *subF : funcList) {
    if (dumpPartition) {
      subF->dumpDAG("partitionLogicalID" +
                    std::to_string(mapping.getLogicalDeviceIDList(subF)[0]) +
                    "__" + subF->getFilename() + "__" +
                    mapping.getPartitionBackendName(subF) + ".dot");
    }
    assert(subF->verify() && "Conversion led to invalid function");
  }
  if (logPartition) {
    logPartitionInfo(mapping);
  }
  return llvm::Error::success();
}

llvm::Error Partitioner::PartitionFromConfig() {
  Function *F = module_->getFunction(partitionConfig_.funcName);
  RETURN_ERR_IF_NOT(F, strFormat("Can't find function %s in current module.",
                                 F->getName().str().data()));

  DCHECK(partitionConfig_.numOfPartitions ==
             partitionConfig_.backendNames.size() &&
         partitionConfig_.numOfPartitions ==
             partitionConfig_.partitionNames.size())
      << "Invalid user-defined partition config.";

  NodeToFunctionMap partitionMap;
  std::vector<Function *> funcList;
  std::unordered_set<size_t> unused;
  std::vector<NodesSet> nodesSets(partitionConfig_.numOfPartitions);
  // Create partitions based on the given number and names.
  for (size_t i = 0; i < partitionConfig_.numOfPartitions; i++) {
    Function *newF =
        module_->createFunction(partitionConfig_.partitionNames[i]);
    funcList.push_back(newF);
    partitionMap.createPartition(newF, partitionConfig_.backendNames[i]);
    unused.insert(i);
  }

  // Map the nodes the the partitions.
  std::vector<Node *> unMapped;
  for (auto &node : F->getNodes()) {
    auto iter = partitionConfig_.nodeToPartition.find(node.getName());
    if (iter == partitionConfig_.nodeToPartition.end()) {
      // If a node in F is not in the node to partition mapping, put it into
      // unMaped list.
      unMapped.push_back(&node);
    } else {
      size_t partitionID = iter->second;
      DCHECK(partitionID < partitionConfig_.numOfPartitions)
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

  // Validate memory usage.
  for (size_t i = 0; i < partitionConfig_.numOfPartitions; i++) {
    GraphMemInfo cost = getGraphMemInfo(nodesSets[i]);
    partitionMap.setGraphMemInfo(funcList[i], cost);
  }
  RETURN_IF_ERR(memoryUsageValidation(partitionMap, backendMap_));

  // Logical device ID validation.
  logicalDeviceID_ = assignLogicalDeviceID(partitionMap, backendMap_);
  RETURN_IF_ERR(logicalDevicesValidation(partitionMap, backendMap_));

  // TODO : loop-free validation.

  // Do partition.
  doPartitioning(F->getName(), {F}, partitionMap, true);
  module_->eraseFunction(F);

  // Do optimization based on backendName.
  for (size_t i = 0; i < partitionConfig_.numOfPartitions; i++) {
    auto func = funcList[i];
    assert(func->verify() && "Conversion led to invalid function");
    std::unique_ptr<Backend> backend(
        createBackend(partitionConfig_.backendNames[i]));
    if (!optimized_) {
      CompilationContext cctx;
      RETURN_IF_ERR(::glow::optimizeFunction(func, *backend, cctx));
    }
  }
  if (logPartition) {
    logPartitionInfo(partitionMap);
  }
  return llvm::Error::success();
}
