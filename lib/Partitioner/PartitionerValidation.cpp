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
#include "glow/Partitioner/PartitionerValidation.h"
#include "glow/Partitioner/PartitionerUtils.h"

#include "llvm/Support/FormatVariadic.h"

namespace glow {
Error logicalDevicesValidation(
    const NodeToFunctionMap &partitions,
    const std::map<std::string, BackendInfo> &backendMap) {
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
    auto backendNum = backendMap.at(backendName).num;
    if (partitionsNum[backendName].size() > backendNum) {
      logPartitionInfo(partitions);
      return MAKE_ERR(
          llvm::formatv(
              "Partition failed: the number of given({0}) devices({1}) "
              "is fewer than the required minimal partitions({2}).",
              backendName, backendNum, partitionsNum[backendName].size())
              .str());
    }
  }
  return Error::success();
}

Error resourceCountValidation(
    const NodeToFunctionMap &partitions,
    const std::map<std::string, BackendInfo> &backendMap) {
  VLOG(1) << "Entering resource count validation";
  std::map<DeviceIDTy, uint64_t> logicalIDInputResources;
  // We are assuming the same resource limit for all devices.
  // Since we do not support P2P for heterogenous devices we do not run this
  // check for heterogeneous partitioning.
  uint64_t availableInputResources = 0;
  for (auto &func : partitions.getPartitions()) {
    auto logicalDeviceList = partitions.getLogicalDeviceIDList(func);
    auto backendName = partitions.getPartitionBackendName(func);
    auto graphMem = partitions.getGraphMemInfo(func);
    auto inputCount = graphMem.inputCount;
    if (!availableInputResources) {
      availableInputResources = backendMap.at(backendName).inputCountMax;
    }
    // For each logicalDevice add the input resources used.
    for (auto dev : logicalDeviceList) {
      if (!logicalIDInputResources.count(dev)) {
        logicalIDInputResources[dev] = 0;
      }
      logicalIDInputResources[dev] += inputCount;
    }
  }

  // If availableInputResources is not 0 check that we are below the limit.
  if (availableInputResources) {
    for (auto &resourceCount : logicalIDInputResources) {
      VLOG(1) << "Checking logical device " << resourceCount.first
              << " resource count: " << resourceCount.second
              << "is less than: " << availableInputResources << "\n";
      if (resourceCount.second > availableInputResources) {
        return MAKE_ERR(
            llvm::formatv(
                "Partition failed: the resource count usage({0}) of one "
                "partition exceeds "
                "the available resource count ({1}).",
                resourceCount.second, availableInputResources)
                .str());
      }
    }
  }
  return Error::success();
}

Error memoryUsageValidation(
    const NodeToFunctionMap &partitions,
    const std::map<std::string, BackendInfo> &backendMap) {
  VLOG(1) << "Entering mem validation";
  for (auto &func : partitions.getPartitions()) {
    auto backendName = partitions.getPartitionBackendName(func);
    auto usedMemSize = partitions.getGraphMemInfo(func).getTotalMemSize();
    auto availableMemSize = backendMap.at(backendName).memSize;
    VLOG(1) << "Comparing " << usedMemSize << " " << availableMemSize << " for "
            << backendName;
    if (usedMemSize > availableMemSize) {
      logPartitionInfo(partitions);
      return MAKE_ERR(
          llvm::formatv("Partition failed: the memory usage({0}) of one "
                        "partition exceeds "
                        "the available memory({1}) of given devices({2}).",
                        usedMemSize, availableMemSize, backendName)
              .str());
    }
  }
  return Error::success();
}

/// \returns true if \p node contains no cycles. \p path contains the nodes in a
/// path, and \p visited contains the nodes checked before.
static bool isDAG(DAGNode *node, llvm::SmallSet<DAGNode *, 10> &path,
                  llvm::SmallSet<DAGNode *, 10> &visited) {
  if (!visited.count(node)) {
    path.insert(node);
    visited.insert(node);
    for (size_t i = 0; i < node->children.size(); i++) {
      auto child = node->children[i];
      if (path.count(child)) {
        // Cycle found.
        return false;
      }
      if (!isDAG(child, path, visited)) {
        return false;
      }
    }
  }
  if (path.count(node)) {
    path.erase(node);
  }
  return true;
}

Error dagValidation(const DAG &dag) {
  auto *root = dag.root.get();
  llvm::SmallSet<DAGNode *, 10> path;
  llvm::SmallSet<DAGNode *, 10> visited;
  // For the first condition: root->children.size() > 0 -- When a dag is
  // created, its root is a dummy node and other DAGNode without parents will be
  // linked to this root. Therefore, root without any child means that each of
  // the rest of DAGNodes has at least one parent. That is, a cycle exists.

  RETURN_ERR_IF_NOT((root->children.size() > 0 && isDAG(root, path, visited)),
                    "Invalid partition: a cycle is detected.");

  // There should not be isolated nodes in partitions.
  RETURN_ERR_IF_NOT((visited.size() == dag.nodes.size() + 1),
                    "Invalid partition: isolated node is detected.");
  return Error::success();
}
} // namespace glow
