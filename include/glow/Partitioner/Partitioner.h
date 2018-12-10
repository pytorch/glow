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
#ifndef GLOW_PARTITIONER_PARTITIONER_H
#define GLOW_PARTITIONER_PARTITIONER_H

#include "glow/Graph/Context.h"
#include "glow/Graph/Graph.h"

#include <map>
#include <set>
#include <string>

namespace glow {

using ModuleList = std::unique_ptr<Module>;

/// Dependency map type.                                                                                                      
using MapTy = llvm::DenseMap<Module *, ModuleList>;

/// This struct keeps the result of partitioning.
struct DependencyDAG {
  /// List of modules after partitioning. Each module will be assigned to run a
  /// device by provioner.
  ModuleList modules_;
  /// The dependencies of each module.
  MapTy modulesDAG_;
  /// The root of this DAG.
  Module *root;
  /// The number of nodes in the DAG.
  unsigned k;
};

/// Here we assume all the devices are identical.
struct DeviceInfo {
  /// Number of the devices.
  unsigned num;
  /// Memory constraints of each device. unit MB.
  unsigned memSize;
  /// TO DO... communication constraints and so on
};

/// Partition
class Partitioner {

  /// The module need to be decomposed.
  Module &module_;

  /// The cost model related to device.
  DeviceInfo *deviceInfo_;

  /// The result of module partitioning.
  DependencyDAG *dependencyDAG_;

public:
  /// \p parent is the module which contains the functions need to be divided.
  /// Here we assume that all the functions in one module belong to a same
  /// "Function Family", that is, without considerting the "dynamic stuff" (i.e.
  /// batch size, input/output shape of each op), all the functions are
  /// identical. The required memory and computation cost for each op can be
  /// found in Module. The \p devices provides the cost model related to
  /// devices.
  Partitioner(const Module &parent, const DeviceInfo &devices);

  /// Divide the module into minimal k sub-modules. The number of k is the
  /// number of , where k is number of modules in the return value. k should be
  /// less or equal to the number of devices. If we can't find a proper k
  /// (though highly impossile I think. What I can think of is the size of some
  /// constant > the momeries of all devices), I think anyway the partitioner
  /// will return the result, and provisioner will check if k is proper?
  DependencyDAG* Partition();
};
} // namespace glow
#endif
