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
#ifndef GLOW_BACKENDS_CPU_CPUBACKEND_H
#define GLOW_BACKENDS_CPU_CPUBACKEND_H

#include "CPUDeviceManager.h"

#include "glow/Backend/Backend.h"
#include "glow/Base/Tensor.h"
#include "glow/LLVMIRCodeGen/LLVMBackend.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/IR/IRBuilder.h"

namespace glow {

class NodeInfo;

class CPUBackend : public LLVMBackend {
public:
  CPUBackend() = default;

  /// @name Backend methods.
  /// This is the implementation of the Backend interface.
  ///@{
  virtual ~CPUBackend() override = default;

  std::string getBackendName() const override { return getName(); }
  static std::string getName() { return "CPU"; }
  static unsigned numDevices();

  bool transformPostLowering(
      Function *F, CompilationContext &cctx,
      const glow::runtime::DeviceInfo *devInfo = nullptr) const override;

  bool isOpSupported(const NodeInfo &NI) const override;

  bool shouldLower(const Node *N) const override;

  runtime::DeviceManager *
  createDeviceManager(const runtime::DeviceConfig &deviceConfig) override {
    return createCPUDeviceManager(deviceConfig);
  }
  /// @}

public:
  /// @name LLVMBackend methods.
  /// This is the implementation of the LLVMBackend interface.
  ///@{
  virtual std::unique_ptr<LLVMIRGen>
  createIRGen(const IRFunction *IR,
              AllocationsInfo &allocationsInfo) const override;

protected:
  virtual std::unique_ptr<CompiledFunction>
  createCompiledFunction(std::unique_ptr<llvm::orc::GlowJIT> JIT,
                         runtime::RuntimeBundle &&runtimeBundle) const override;

  virtual llvm::StringRef getLibjitBitcode() const override;
  /// @}
};

} // namespace glow

#endif // GLOW_BACKENDS_CPU_CPUBACKEND_H
