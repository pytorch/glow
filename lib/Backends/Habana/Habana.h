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
#ifndef GLOW_BACKENDS_HABANA_HABANA_H
#define GLOW_BACKENDS_HABANA_HABANA_H

#include "HabanaDeviceManager.h"

#include "glow/Backend/Backend.h"
#include "glow/Backend/CompiledFunction.h"
#include "glow/IR/Instrs.h"
#include "glow/Quantization/Base/Base.h"

#include "synapse.h"

#include <string>

namespace glow {

class HabanaBackend final : public Backend {
public:
  /// Constructor.
  HabanaBackend() = default;

  /// @name Backend methods.
  /// This is the implementation of the Backend interface.
  ///@{
  ~HabanaBackend() override = default;

  std::string getBackendName() const override { return getName(); }
  static std::string getName() { return "Habana"; }
  static unsigned numDevices();

  Expected<std::unique_ptr<CompiledFunction>>
  compile(Function *F, const BackendOptions &opts) const override;

  bool isOpSupported(const NodeInfo &NI) const override;

  bool shouldLower(const Node *N) const override;

  bool transformPostLowering(
      Function *F, CompilationContext &cctx,
      const glow::runtime::DeviceInfo *devInfo = nullptr) const override;

  bool shouldShareBuffers() const override { return false; }

  bool supportsPartialTensors() const override { return true; }

  runtime::DeviceManager *
  createDeviceManager(const runtime::DeviceConfig &deviceConfig) override {
    return createHabanaDeviceManager(deviceConfig);
  }
  /// @}

  static bool isVersionBiggerEqualTo(std::string versionToCompare);
};

} // namespace glow

#endif // GLOW_BACKENDS_HABANA_HABANA_H
