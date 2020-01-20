/*
 * Copyright (c) Glow Contributors. See CONTRIBUTORS file.
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

#ifndef GLOW_BACKENDS_NNPI_NNPIDEVICEMANAGER_H
#define GLOW_BACKENDS_NNPI_NNPIDEVICEMANAGER_H

#include "InferencePool.h"
#include "NNPITracing.h"
#include "glow/Backends/DeviceManager.h"
#include "glow/Runtime/RuntimeTypes.h"
#include "glow/Support/ThreadPool.h"
#include "nnpi_inference.h"
#include <atomic>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace glow {
class NNPICompiledFunction;

namespace runtime {

/// A class controlling a single "NNPI Device", a thread of execution in
/// the IR-NNPI. Many NNPIFunctions may be added, but only one
/// inference is executed at a time.
class InferenceThreadEnv;

class NNPIStaticPlaceholderContainer {
  struct NamedResourceWithRef : public NamedResource {
    uint64_t refCount = 0;
    NamedResourceWithRef() : refCount(0) {}
    NamedResourceWithRef(const NamedResource &nr)
        : NamedResource(nr), refCount(0){};
  };
  std::unordered_map<const Placeholder *, NamedResourceWithRef>
      staticPlaceholdersDeviceResource_;
  /// NNPI Device Context handle.
  NNPIDeviceContext device_;
  // Given a Placeholder, erase its allocated NamedResourceWithRef and destory
  // its DeviceResource
  // return true on success, false otherwise
  // NOTE: Doesn't check for remaining refs pointing at the DeviceResource
  bool EraseAndDestroyDeviceResource_(const Placeholder *PH);

public:
  NNPIStaticPlaceholderContainer() : device_(NNPI_INVALID_NNPIHANDLE){};
  ~NNPIStaticPlaceholderContainer();

  // Set the device that the container create and destroy device resource for
  // return false if device is invalid
  bool SetDevice(NNPIDeviceContext device, bool inferOnRuntime);

  // Acquire a device resource with a given static Placeholder and
  // NamedResource
  // returned NamedResource will contain the allocated handle
  // caller should check the returned handle before using it.
  NamedResource AcquireDeviceResource(const Placeholder *,
                                      const NamedResource &nr);

  // Release a device resource given a static Placeholder
  // return false if release failed.
  bool ReleaseDeviceResource(const Placeholder *);
};

// thread pool per network
class NNPIDeviceManager : public DeviceManager {
  /// Compiled function list by name.
  FunctionMapTy functions_;
  /// Maximum available memory on the device, for local devices fix to some
  /// constant.
  uint64_t maxMemoryBytes_{14000000000l};
  /// Amount of memory used by all models.
  uint64_t usedMemoryBytes_{0};
  /// Static memory cost of the InterpreterFunction.
  const uint64_t functionCost_{1};
  /// Number of worker threads allocated per loaded function.
  unsigned numWorkersPerFunction_;

  /// Inference id counter.
  static std::atomic<RunIdentifierTy> runIdentifier_;

  /// NNPI Device id.
  unsigned deviceId_;
  /// Inference objects kept per added network.
  InferencePoolMap inferenceEnvs_;

  /// NNPI Adapter handle.
  NNPIAdapter adapter_;
  /// NNPI Device Context handle.
  NNPIDeviceContext device_;
  /// Lock to synchronize function adding/removing to/from the device manager.
  std::mutex functionMapMutex_;
  /// Device Tracing contorl.
  std::shared_ptr<NNPIDeviceTracing> deviceTracing_;
  /// Maps between static placeholders' names to their device resource.
  NNPIStaticPlaceholderContainer staticPlaceholderContainer_;
  /// NNPI Device options (environment variables + DeviceConfig options).
  NNPIDeviceOptions deviceOptions_;

public:
  explicit NNPIDeviceManager(const DeviceConfig &config,
                             unsigned numInferenceWorkers = 0);
  virtual ~NNPIDeviceManager();

  Error init() override;
  void addNetwork(const Module *module, FunctionMapTy functions,
                  ReadyCBTy readyCB) override;
  void evictNetwork(std::string functionName,
                    EvictFunctionCBTy evictCB) override;
  RunIdentifierTy runFunction(std::string functionName,
                              std::unique_ptr<ExecutionContext> ctx,
                              runtime::ResultCBTy resultCB) override;
  Error stop(bool block) override;
  uint64_t getMaximumMemory() const override;
  uint64_t getAvailableMemory() const override;
  bool isMemoryAvailable(uint64_t estimate) const override;

  void transferStaticPlaceholderToDevice(Placeholder *PH, Tensor *T,
                                         std::function<void(Error)> resultCB);
};

DeviceManager *createNNPIDeviceManager(const DeviceConfig &config);
} // namespace runtime
} // namespace glow

#endif // GLOW_BACKENBDS_NNPI_NNPIDEVICEMANAGER_H
