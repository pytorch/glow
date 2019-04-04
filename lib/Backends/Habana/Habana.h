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
#ifndef GLOW_BACKENDS_HABANA_HABANA_H
#define GLOW_BACKENDS_HABANA_HABANA_H

#include "glow/Backends/Backend.h"
#include "glow/Backends/CompiledFunction.h"
#include "glow/IR/Instrs.h"
#include "glow/Quantization/Base/Base.h"

#include "synapse.h"

#include <string>

namespace glow {

class HabanaFunction final : public CompiledFunction {
public:
  /// Constructor.
  HabanaFunction(const runtime::RuntimeBundle &bundle,
                 const std::string &recipeName, PlaceholderList &&inputs,
                 PlaceholderList &&outputs);

  /// @name CompiledFunction interface
  ///@{
  ~HabanaFunction() override;

  const std::string &getRecipeName() const { return recipeName_; }

  void execute(ExecutionContext *context) override;

  /// Allocates on device buffer and copies Constant weights to device.
  void setupRuns() override;

  /// Per run setup, copies Inputs from \p ctx to on device memory.
  void beforeRun(const PlaceholderBindings &ctx) override;

  /// Copies outputs from device to tensors in \p ctx.
  void afterRun(const PlaceholderBindings &ctx) override;

  /// Final cleanup.
  void tearDownRuns() override;
  ///@}

  /// \returns the Kind of Backend used to compile this function.
  BackendKind getCompileBackendKind() const override {
    return BackendKind::Habana;
  }

  const PlaceholderList &getInputs() const { return inputs_; }

  const PlaceholderList &getOutputs() const { return outputs_; }

private:
  Function *F_;
  std::string recipeName_;
  const PlaceholderBindings *ctx_;
  PlaceholderList inputs_;
  PlaceholderList outputs_;
};

/// RAII wrapper for MMU mapping from host to device memory.
class MemoryMap {
public:
  MemoryMap(int32_t deviceId, uint64_t size) : deviceId_(deviceId) {
    auto status =
        synMalloc(deviceId_, size, synMemFlags::synMemHost, &buffer_, 0);
    GLOW_ASSERT(status == synSuccess);
  }

  ~MemoryMap() {
    auto status = synFree(deviceId_, buffer_);
    GLOW_ASSERT(status == synSuccess);
  }

  MemoryMap(const MemoryMap &) = delete;
  MemoryMap &operator=(const MemoryMap &) = delete;

  char *getBuffer() { return (char *)buffer_; }

private:
  int32_t deviceId_;
  void *buffer_;
};

/// Thie struct wraps a topology ID with its corresponding HabanaFunction so
/// that only one map is needed to keep track of both.
struct HabanaFunctionMeta {
  HabanaFunctionMeta() = default;

  HabanaFunctionMeta(int32_t deviceId, uint64_t topo, HabanaFunction *HF);

  char *getPointer(Placeholder *P) { return ioBuffer->getBuffer() + ioMap[P]; }

  /// The topology ID of the function. This is returned by the Synapse API
  /// after loading a recipe.
  uint64_t topologyId;

  /// The HabanaFunction corresponding to topologyId. This is needed in order
  /// to call HabanaFunction::executeOnDevice after loading and activating a
  /// topology.
  HabanaFunction *function;

  std::unique_ptr<MemoryMap> ioBuffer;

  std::unordered_map<Placeholder *, size_t> ioMap;
};

class HabanaBindings : public DeviceBindings {
public:
  HabanaBindings(uint32_t deviceId, HabanaFunctionMeta *meta)
      : DeviceBindings(BackendKind::Habana), deviceId_(deviceId), meta_(meta) {}

  virtual ~HabanaBindings() {}

  std::unique_ptr<DeviceBindings> clone() override {
    return llvm::make_unique<HabanaBindings>(deviceId_, meta_);
  }

  uint32_t getDeviceId() const { return deviceId_; }

  HabanaFunctionMeta *getMeta() { return meta_; }

private:
  uint32_t deviceId_;
  HabanaFunctionMeta *meta_;
};

class HabanaBackend final : public Backend {
public:
  /// Constructor.
  HabanaBackend() = default;

  /// @name Backend methods.
  /// This is the implementation of the Backend interface.
  ///@{
  ~HabanaBackend() override = default;

  BackendKind getBackendKind() const override { return BackendKind::Habana; }

  std::unique_ptr<CompiledFunction>
  compile(Function *F, const CompilationOptions &opts) const override;

  bool isOpSupported(const NodeInfo &NI) const override;

  bool shouldLower(const Node *N) const override;

  bool transformPostLowering(Function *F,
                             const CompilationOptions &opts) const override;

  bool shouldShareBuffers() const override { return false; }
  /// @}
};

} // namespace glow

#endif // GLOW_BACKENDS_HABANA_HABANA_H
