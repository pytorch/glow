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
#ifndef GLOW_LLVMIRCODEGEN_LLVMBACKEND_H
#define GLOW_LLVMIRCODEGEN_LLVMBACKEND_H

#include "glow/Backend/Backend.h"
#include "glow/Backend/CompiledFunction.h"
#include "glow/Base/Tensor.h"
#include "glow/LLVMIRCodeGen/GlowJIT.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/IR/IRBuilder.h"

namespace glow {

struct AllocationsInfo;
class PlaceholderBindings;
class LLVMIRGen;

class LLVMBackend : public BackendUsingGlowIR {
  /// Target used by this backend.
  std::string target_;
  /// Arch used by this backend.
  std::string arch_;
  /// Cpu used by this backend.
  std::string cpu_;

public:
  LLVMBackend();
  /// \returns target used by this backend.
  llvm::StringRef getTarget() const { return target_; }
  /// Sets target used by this backend.
  void setTarget(llvm::StringRef target) { target_ = target; }
  /// \returns arch used by this backend.
  llvm::StringRef getArch() const { return arch_; }
  /// Sets arch used by this backend.
  void setArch(llvm::StringRef arch) { arch_ = arch; }
  /// \returns cpu used by this backend.
  llvm::StringRef getCPU() const { return cpu_; }
  /// Sets cpu used by this backend.
  void setCPU(llvm::StringRef cpu) { cpu_ = cpu; }
  /// @name Backend methods.
  /// This is the implementation of the Backend interface.
  ///@{
  virtual ~LLVMBackend() override = default;

  virtual std::unique_ptr<CompiledFunction>
  compileIR(std::unique_ptr<IRFunction> IR) const override;

  virtual std::unique_ptr<CompiledFunction>
  compileIRWithoutConstants(IRFunction *IR) const;

  virtual llvm::Expected<std::unique_ptr<CompiledFunction>>
  compile(Function *F, const BackendOptions &opts) const override;

  virtual void save(Function *F, llvm::StringRef outputDir,
                    llvm::StringRef bundleName,
                    llvm::StringRef mainEntryName) const override;
  /// @}

  /// \returns the size of metrics collected for a single TraceEvent.
  virtual size_t getTraceEventDataSize() const override {
    return sizeof(uint64_t);
  }

  /// Method that creates the LLVM IR generator. This gives the possibility to
  /// create a backend that inherits from the CPU backend, while providing
  /// a specific version of the LLVM IR generator derived from LLVMIRGen.
  /// \param IR the IRFunction function to be converted into the LLVM IR.
  /// \param allocationsInfo information about allocation of weights and
  /// activations.
  /// \returns backend-specific LLVMIRGen instance.
  virtual std::unique_ptr<LLVMIRGen>
  createIRGen(const IRFunction *IR, AllocationsInfo &allocationsInfo) const = 0;

protected:
  /// Method that creates a CompiledFunction.
  /// \param JIT GlowJIT to be used.
  /// \param runtimeBundle bundle to be used for compiling the function.
  /// \returns created CompiledFunction.
  virtual std::unique_ptr<CompiledFunction>
  createCompiledFunction(std::unique_ptr<llvm::orc::GlowJIT> JIT,
                         const runtime::RuntimeBundle &runtimeBundle) const = 0;

  /// \returns libjit bitcode for the current backend.
  virtual llvm::StringRef getLibjitBitcode() const = 0;

  /// Emit the jitmain function.
  virtual void emitJitMain(LLVMIRGen &irgen) const;
};

} // namespace glow

#endif // GLOW_LLVMIRCODEGEN_LLVMBACKEND_H
