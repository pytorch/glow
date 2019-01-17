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
#ifndef GLOW_RUNTIME_PROVISIONER_H
#define GLOW_RUNTIME_PROVISIONER_H

#include "glow/Backends/Backend.h"
#include "glow/Backends/DeviceManager.h"
#include "glow/Runtime/RuntimeTypes.h"

#include <map>

namespace glow {
namespace runtime {

/// The Provisioner is responsible for assigning networks to an actual device.
/// It is a stateless class, relying on information being passed in by the
/// caller.
class Provisioner final {
public:
  /// Walks \p networks and assigns each function to a DeviceManager in \p
  /// devices. The Provisioner calls the addNetwork method for each
  /// DeviceManager and uses the returned networkID to populate \p runDAG.
  /// Returns a ResultCode indicating if the operation was a success.
  ResultCode
  provision(std::vector<DAGNode> &networks,
            std::map<DeviceIDTy, std::unique_ptr<DeviceManager>> &devices,
            Module &module);

private:
  std::unique_ptr<Backend> backend_;
};
} // namespace runtime
} // namespace glow

#endif // GLOW_RUNTIME_PROVISIONER_H
