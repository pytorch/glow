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

#include "glow/Runtime/Provisioner/Provisioner.h"
#include "glow/Partitioner/Partitioner.h"
#include "glow/Runtime/Executor/Executor.h"

using namespace glow;
Provisioner::Provisioner() = default;

ResultCode
Provisioner::provision(dependencyDAG &networks, executionDAG &runDAG,
                       std::unordered_map<int, DeviceManager> &devices){
    // Check that there is available space for provisioning.
    // Assuming there is space, start provisioning.
    // Walk the list of modules and call addNetwork.
    // If a module fails to provision try on a different device if available.

};