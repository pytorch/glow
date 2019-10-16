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
#ifndef GLOW_PARTITIONER_PARTITIONEROPTIMIZER_H
#define GLOW_PARTITIONER_PARTITIONEROPTIMIZER_H

#include "glow/Partitioner/PartitionerTypes.h"

namespace glow {
/// By using heuristic algorithm to move nodes among \p partitions, optimize the
/// total communication cost of running a module and keep the memory usage of
/// each partition within \p availableMemory.
void optimizeCommunicationCost(NodeToFunctionMap &partitions,
                               FunctionToNodesMap &nodesSet, Module *mod,
                               uint64_t availableMemory);

/// Combine partitions according to the following rules: Rule 1 :if all outside
/// uses of the nodes in partition1 is in partition2, and the sum of memory
/// consumption of partition1 and partition2 is less than availableMemory,
/// combine partition1 and partition2.
void partitionsCombine(NodeToFunctionMap &partitions,
                       FunctionToNodesMap &nodesSet, Module *mod,
                       uint64_t availableMemory);

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
DeviceIDTy
assignLogicalDeviceID(NodeToFunctionMap &mapping,
                      const std::map<std::string, BackendInfo> &backendMap);
} // namespace glow
#endif // GLOW_PARTITIONER_PARTITIONEROPTIMIZER_H
