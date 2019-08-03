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
#include "glow/Partitioner/PartitionerValidation.h"

namespace glow {
llvm::Error
logicalDevicesValidation(const NodeToFunctionMap &partitions,
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
memoryUsageValidation(const NodeToFunctionMap &partitions,
                      const std::map<std::string, BackendInfo> &backendMap) {
  for (auto &func : partitions.getPartitions()) {
    auto backendName = partitions.getPartitionBackendName(func);
    auto usedMemSize = partitions.getGraphMemInfo(func).getTotalMemSize();
    auto availableMemSize = backendMap.at(backendName).memSize;
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
} // namespace glow
