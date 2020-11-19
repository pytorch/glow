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
#ifndef GLOW_PARTITIONER_PARTITIONERVALIDATION_H
#define GLOW_PARTITIONER_PARTITIONERVALIDATION_H

#include "glow/Partitioner/PartitionerTypes.h"

namespace glow {
/// Check if \p partitions satisfies number of physical devices restriction.
/// I.e. check if the number of logical devices is less than the given
/// physical devices.
Error logicalDevicesValidation(
    const NodeToFunctionMap &partitions,
    const std::map<std::string, BackendInfo> &backendMap);

/// Check if the memory usage of each partition meets the physical device
/// memory restriction.
Error memoryUsageValidation(
    const NodeToFunctionMap &partitions,
    const std::map<std::string, BackendInfo> &backendMap);

/// Verify number of input resources meet the backend constraints. Only intended
/// for homogeneous backends.
Error resourceCountValidation(
    const NodeToFunctionMap &partitions,
    const std::map<std::string, BackendInfo> &backendMap);

/// Check if the current partition is a valid DAG. This check can only be
/// called after a real partition is created and the DAG is generated.
Error dagValidation(const DAG &dag);

} // namespace glow
#endif // GLOW_PARTITIONER_PARTITIONERVALIDATION_H
