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

#ifndef GLOW_NNPI_BACKEND_H
#define GLOW_NNPI_BACKEND_H

#include "NNPIOptions.h"
#include "glow/Backend/Backend.h"

namespace glow {

/// This is the Intel Neural-Network Processor for Inference (NNPI) backend.
class NNPIBackend final : public Backend {
public:
  /// Ctor.
  explicit NNPIBackend() = default;

  /// @name Backend methods.
  /// This is the implementation of the Backend interface.
  ///@{
  ~NNPIBackend() override = default;

  std::string getBackendName() const override { return getName(); }
  static std::string getName() { return "NNPI"; }
  static unsigned numDevices();

  Expected<std::unique_ptr<CompiledFunction>>
  compile(Function *F, const BackendOptions &opts) const override;

  bool isOpSupported(const NodeInfo &NI) const override;
  bool shouldLower(const Node *N) const override;
  bool shouldShareBuffers() const override { return false; }
  bool supportsPartialTensors() const override { return true; }
  bool supportsStaticPlaceholders() const override { return true; }
  FunctionPassPipeline getOptimizationPipeline() const override;

  runtime::DeviceManager *
  createDeviceManager(const runtime::DeviceConfig &deviceConfig) override;

  bool transformPostLowering(
      Function *F, CompilationContext &cctx,
      const glow::runtime::DeviceInfo *devInfo = nullptr) const override;

  virtual llvm::StringMap<std::string>
  getSupportedCompiledFunctionOptions() const override {
    NNPICompilationOptions options({});
    return options.getSupportedOptions();
  };

  virtual llvm::StringMap<std::string>
  getSupportedDeviceManagerOptions() const override {
    NNPIDeviceOptions options({});
    return options.getSupportedOptions();
  };
  /// @}

private:
#if FACEBOOK_INTERNAL
  /// Performs FB-private transformations on \p F given \p cctx.
  /// \returns whether \p F is modified.
  bool transformPrivate(Function *F, CompilationContext &cctx) const;
#endif /* FACEBOOK_INTERNAL */

  static NNPIBackendOptions backendOptions_;
};

Backend *createNNPIBackend(const runtime::DeviceConfig &deviceConfig);

} // namespace glow
#endif // GLOW_NNPI_BACKEND_H
