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
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace glow {
class NNPICompiledFunction;
class NNPIAdapterContainer;
namespace runtime {

class NNPIResource;
class InferenceContext;
using StaticPlaceholderMap =
    std::unordered_map<const Placeholder *, std::weak_ptr<NNPIResource>>;

/// A class controlling a single "NNPI Device", a thread of execution in
/// the IR-NNPI. Many NNPIFunctions may be added, but only one
/// inference is executed at a time.
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

  /// Inference id counter.
  static std::atomic<RunIdentifierTy> runIdentifier_;

  /// NNPI Device id.
  unsigned deviceId_;
  /// Inference objects kept per added network.
  InferencePoolMap inferencePools_;

  /// NNPI Adapter container.
  NNPIAdapterContainer *pAdapter_ = nullptr;
  /// NNPI Device Context handle.
  NNPIDeviceContext device_;
  /// Lock to synchronize function adding/removing to/from the device manager.
  std::mutex functionMapMutex_;
  /// Static placeholders known by the device manager (the device manager
  /// doesn't own a ref on static resources, only networks added to the device
  /// manager).
  StaticPlaceholderMap staticPlaceholders_;
  /// NNPI Device options (environment variables + DeviceConfig options).
  std::shared_ptr<NNPIDeviceOptions> deviceOptions_;

public:
  explicit NNPIDeviceManager(const DeviceConfig &config,
                             std::shared_ptr<NNPIDeviceOptions> deviceOptions,
                             NNPIAdapterContainer *adapter);
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

  virtual Error startDeviceTrace(TraceContext *traceContext) override;
  virtual Error stopDeviceTrace(TraceContext *traceContext) override;
  Error bindContext(std::string functionName, ExecutionContext *ctx,
                    PlaceholderUsageMap &phUsage);
  void addPlaceholderUsageCount(std::string functionName,
                                PlaceholderUsageMap &phUsage);

  void *allocateDeviceIOBuffer(dim_t size) override;
  void freeAllocatedDeviceIOBuffer(void *buffer) override;

  std::string getStatusStr() const;
};

class NNPIDeviceBindings : public DeviceBindings {
public:
  NNPIDeviceBindings(std::shared_ptr<InferenceContext> &infCtx)
      : DeviceBindings("NNPI"), infCtx_(infCtx) {}

  virtual ~NNPIDeviceBindings() {}

  std::unique_ptr<DeviceBindings> clone() override {
    return std::make_unique<NNPIDeviceBindings>(infCtx_);
  }

  std::shared_ptr<InferenceContext> getInferenceContext() const {
    return infCtx_;
  }

private:
  std::shared_ptr<InferenceContext> infCtx_;
};

DeviceManager *createNNPIDeviceManager(const DeviceConfig &config,
                                       NNPIAdapterContainer *adapter);
} // namespace runtime
} // namespace glow

#endif // GLOW_BACKENBDS_NNPI_NNPIDEVICEMANAGER_H
