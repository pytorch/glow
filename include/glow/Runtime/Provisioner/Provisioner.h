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
#include "glow/Backend/BlockStreamBase.h"
#include "glow/Backends/DeviceManager.h"
#include "glow/Runtime/RuntimeTypes.h"
#include "glow/Support/Error.h"

#include <map>

#if FACEBOOK_INTERNAL
namespace folly {
struct dynamic;

} // namespace folly
namespace glow {
namespace runtime {
using FXFunction = folly::dynamic;
}
} // namespace glow
#endif

namespace glow {
namespace runtime {

enum class NetworkType {
  // FX Network
  FX_NETWORK,
  // Glow Module network
  GLOW_NETWORK,
};

/// Base struct for passing in a network to Provisioner. It contains all common
/// elements: DagListTy, Module, and CCTX and a NetworkType denoting what
/// subclass of network it is.
struct Network {
  /// Backend used for this config. It is used in
  /// checking the type of config before casting to a derived class.
  const NetworkType networkType;

  /// Dag structure for the network to be added.
  DAGListTy &networks;

  /// Module containing PH's for the network and in some cases the network.
  Module &module;

  /// Compilation Context for the network being added.
  CompilationContext &cctx;

  Network(NetworkType netType, DAGListTy &networks, Module &module,
          CompilationContext &cctx)
      : networkType(netType), networks(networks), module(module), cctx(cctx) {}
  virtual ~Network() = default;
};
#if FACEBOOK_INTERNAL
struct FXNetwork : Network {
  const FXFunction &FXIR;
  const llvm::StringMap<const void *> &constants;
  FXNetwork(DAGListTy &networks, Module &module, CompilationContext &cctx,
            const FXFunction &FXIR,
            const llvm::StringMap<const void *> &constants)
      : Network(NetworkType::FX_NETWORK, networks, module, cctx), FXIR(FXIR),
        constants(constants) {}
};
#endif

struct GlowNetwork : Network {
  GlowNetwork(DAGListTy &networks, Module &module, CompilationContext &cctx)
      : Network(NetworkType::GLOW_NETWORK, networks, module, cctx) {}
};

/// The Provisioner is responsible for assigning networks to an actual device.
/// It also compiles the networks before passing the compiled functions to the
/// device.
class Provisioner final {
public:
  Provisioner(DeviceManagerMapTy &devices);

  /// Traverses the DAG \p networks and compiles all the node's Functions from
  /// \p module using \p cctx. Then add compiled functions to assigned devices.
  ///
  /// Pseudocode:
  ///
  /// generate device assignments
  /// create map `optsMap`, `compiledFunctions`, `remainingDeviceCount`
  ///
  /// for each assignment
  ///     create vector functionsToCompile
  ///     create map functionMap
  ///     for each node in logical device
  ///         if Function hasn't been compiled before
  ///             add Function to `functionsToCompile`
  ///             add Function's BackendOptions to `optsMap`
  ///             set `remainingDeviceCount` for Function
  ///         else
  ///             decrease `remainingDeviceCount` for Function by 1
  ///
  ///     call Backend::compiledFunctions with `functionsToCompile` and
  ///     `optsMap`
  ///     move compiled functions to `compiledFunctions`
  ///
  ///     for each node in logical device
  ///         add corresponding compiled functions in `compiledFunctions` to
  ///         `functionMap`
  ///         add replications to `functionMap` using the same compiled function
  ///         with a different name
  ///
  ///     call DeviceManager::addNetwork with `FunctionMap`
  ///
  ///     for each node in logical device
  ///         if `remainingDeviceCount` for Function is 0
  ///             free up compilation resources
  ///             move corresponding compiled function from `compiledFunctions`
  ///             to `Provisioner::functions_`
  Error provision(DAGListTy &networks, Module &module,
                  CompilationContext &cctx);

#if FACEBOOK_INTERNAL
  /// Traverses the DAG \p networks and:
  ///   1. Retrieves each node's Function from the provided \p FXIR.
  ///   2. Compiles it using the provided CompilationContext \p cctx.
  ///   3. Assigns a device and calls addNetwork on the chosen device(s).
  /// \returns a Error indicating if the operation was a success.
  Error provisionFX(DAGListTy &networks, Module &module, const FXFunction &FXIR,
                    const llvm::StringMap<const void *> &constants,
                    CompilationContext &cctx);
#endif
  // Unified provisioning function, tries to re-use most shared logic between
  // provision and provisionFX.
  Error provisionNetwork(std::unique_ptr<Network> network);
  /// Remove stored compiledFunction.
  Error removeFunction(llvm::StringRef name);

  /// Evict function from device.
  Error evictFunction(llvm::StringRef name, DeviceManager *device,
                      unsigned replicaCount);

  /// \returns a reference to the backend with name \p backendName.
  Backend &getBackend(llvm::StringRef backendName) const;

  /// \returns a reference to the Backend if only one Backend is found,
  /// otherwise returns an Error.
  Expected<Backend *> getBackend() const;

  /// Update the list of available devices.
  void updateAvailableDevices(const std::vector<DeviceManager *> &devices,
                              const std::vector<DeviceIDTy> &mappings) {
    devices_ = devices;
    deviceMappings_ = mappings;
  }

  // Extract function streams from functions_ to serializedFunctionMap_,
  // and return a ptr of serializedFunctionMap_.
  // Each time this function called, serializedFunctionMap_ will be regenerated.
  std::unique_ptr<
      std::unordered_map<std::string, std::unique_ptr<BlockStreamBase>>>
  getAllSerializedFunctionsMap();

  // Clean up all stored serializedFunctionMap_.
  void cleanUpSerializedFunctionMap();

private:
  /// Map of backends for all devices, one backend per device type.
  std::unordered_map<std::string, std::unique_ptr<Backend>> backends_;

  /// Map of compiledFunction unique pointers. This maintains
  /// ownership of the functions.
  std::unordered_map<std::string, std::unique_ptr<CompiledFunction>> functions_;

  /// Map of serialized function pointers, storing all serialized functions on
  /// backends.
  /// Only used in serialization.
  std::unordered_map<std::string, std::unique_ptr<BlockStreamBase>>
      serializedFunctionMap_;

  /// Set of active functions - these are functions that are currently being
  /// compiled/added to devices.
  std::set<std::string> activeFunctions_;

  /// Mapping from function name to its number of replications
  std::unordered_map<std::string, unsigned> functionReplicaCount_;

  /// Mutex for functions_ and activeFunctions_ since add/remove can be called
  /// from multiple threads simultaneously.
  std::mutex functionsLock_;

  /// List of available DeviceManagers added during initialization.
  std::vector<DeviceManager *> devices_;

  /// Mapping from available devices to deviceID;
  std::vector<DeviceIDTy> deviceMappings_;

  /// Helper function to cleanup a provision call. On \p failure free the
  /// compiledFunctions that were created, \p names , and remove networks
  /// already added to devices, \p currentNetworkResidency .
  void cleanupProvision(llvm::ArrayRef<std::string> names,
                        std::map<DeviceIDTy, std::vector<std::string>> const
                            &currentNetworkResidency,
                        bool failure = true);

  /// Helper function to parse the DAG and generate logicalDevices.
  std::map<DeviceIDTy, std::vector<DAGNode *>>
  generateLogicalDevices(const DAGListTy &networks);

  /// Helper method to check that new networks don't collide with another
  /// network currently being added. Note: This cannot be called under a lock on
  /// functionsLock_ as it acquires a lock internally.
  Error checkActiveNetworks(const DAGListTy &networks,
                            std::vector<std::string> &localActiveNames);

  /// This function pairs logical devices with phsyical devices, it sorts both
  /// sets of devices by available memory and attempts to find pairings for all
  /// of then. The output is a map between logicalDevice and PhysicalDevice.
  /// Requires a vector of DeviceID:memorySize pairs \p logicalDeviceSize, \p
  /// deviceMemoryMap a mapping from backendName to a list of device:memorySize
  /// pairs for all devices of the specified backend, and \p logicalDevices a
  /// map of logicalIDs to all associated DAGNodes.
  Expected<std::map<DeviceIDTy, DeviceIDTy>> generateDeviceAssignments(
      const std::vector<std::pair<DeviceIDTy, uint64_t>> &logicalDeviceSize,
      std::map<std::string, std::vector<std::pair<DeviceIDTy, uint64_t>>>
          &deviceMemoryMap,
      std::map<DeviceIDTy, std::vector<DAGNode *>> &logicalDevices);
};
} // namespace runtime
} // namespace glow

#endif // GLOW_RUNTIME_PROVISIONER_H
