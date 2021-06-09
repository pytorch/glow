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

#include <vector>

namespace glow {

class NodeInfo;

class CPUBackend : public LLVMBackend {
public:
  CPUBackend();

  /// @name Backend methods.
  /// This is the implementation of the Backend interface.
  ///@{
  virtual ~CPUBackend() override = default;

  std::string getBackendName() const override {
    return Named::getName().empty() ? getName() : Named::getName().str();
  }
  static std::string getName() { return "CPU"; }
  static unsigned numDevices();
  static std::vector<unsigned> scanDeviceIDs();

  Expected<bool> transformPostLowering(
      Function *F, CompilationContext &cctx,
      const glow::runtime::DeviceInfo *devInfo = nullptr) const override;

  /// \returns whether the provided \p NI is supported by the backend.
  bool isOpSupported(const NodeInfo &NI) const override;

  bool shouldLower(const Node *N) const override;

  /// \returns whether the backend supports fusing \p activation into \p parent.
  bool supportsFusedActivation(Node *parent, Node *activation) const override;

  runtime::DeviceManager *
  createDeviceManager(const runtime::DeviceConfig &deviceConfig) override {
    return createCPUDeviceManager(deviceConfig);
  }

  /// \returns true if network supports Type Lowering from \p T1 to \p T2.
  /// Populates PrecisionConfiguration with black list of operations that can't
  /// be converted.
  virtual bool
  canDoIndexTypeDemotion(ElemKind fromTy, ElemKind toTy,
                         PrecisionConfiguration &precConfig) const override;

  llvm::ArrayRef<llvm::MemoryBufferRef> getObjectRegistry() const override;
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
  createCompiledFunction(std::unique_ptr<GlowJIT> JIT,
                         runtime::RuntimeBundle &&runtimeBundle) const override;

  virtual llvm::StringRef getLibjitBitcode() const override;
  /// @}
};

} // namespace glow

#endif // GLOW_BACKENDS_CPU_CPUBACKEND_H
