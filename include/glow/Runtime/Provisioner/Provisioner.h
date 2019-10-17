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
#ifndef GLOW_RUNTIME_PROVISIONER_H
#define GLOW_RUNTIME_PROVISIONER_H

#include "glow/Backend/Backend.h"
#include "glow/Backends/DeviceManager.h"
#include "glow/Runtime/RuntimeTypes.h"
#include "glow/Support/Error.h"

#include <map>

namespace glow {
namespace runtime {

/// The Provisioner is responsible for assigning networks to an actual device.
/// It also compiles the networks before passing the compiled functions to the
/// device.
class Provisioner final {
public:
  Provisioner(DeviceManagerMapTy &devices);

  /// Traverses the DAG \p networks and:
  ///   1. Retrieves each node's Function from the provided \p module.
  ///   2. Compiles it using the provided CompilationContext \p cctx.
  ///   3. Assigns a device and calls addNetwork on the chosen device(s).
  /// \returns a Error indicating if the operation was a success.
  Error provision(DAGListTy &networks, Module &module,
                  CompilationContext &cctx);

  /// Remove stored compiledFunction.
  Error removeFunction(llvm::StringRef name);

private:
  /// Pointer to backend used for compilation. This currently gets reset per
  /// device to ensure the correct backed per device.
  std::vector<std::unique_ptr<Backend>> backends_;

  /// Map of compiledFunction unique pointers. This maintains ownership of the
  /// functions.
  std::unordered_map<std::string, std::unique_ptr<CompiledFunction>> functions_;

  /// Set of active functions - these are functions that are currently being
  /// compiled/added to devices.
  std::set<std::string> activeFunctions_;

  /// Mutex for functions_ and activeFunctions_ since add/remove can be called
  /// from multiple threads simultaneously.
  std::mutex functionsLock_;

  /// List of available DeviceManagers added during initialization.
  std::vector<DeviceManager *> devices_;

  /// Helper function to cleanup a provision call. On a success free resources
  /// that are no longer needed by the compiledFunctions. On failure free the
  /// compiledFunctions that were created.
  void cleanupProvision(llvm::ArrayRef<std::string> names, bool failure = true);
};
} // namespace runtime
} // namespace glow

#endif // GLOW_RUNTIME_PROVISIONER_H
