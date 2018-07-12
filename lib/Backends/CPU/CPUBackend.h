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
#ifndef GLOW_BACKENDS_CPU_CPUBACKEND_H
#define GLOW_BACKENDS_CPU_CPUBACKEND_H

#include "glow/Backends/Backend.h"
#include "glow/Base/Tensor.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/IR/IRBuilder.h"

namespace glow {

/// Helper function to create a new CallInst, with the specified \p builder, \p
/// callee, and \p args. Verifies that the function signature is correct,
/// and then creates and \returns the CallInst.
llvm::CallInst *createCall(llvm::IRBuilder<> &builder, llvm::Function *callee,
                           llvm::ArrayRef<llvm::Value *> args);

class CPUBackend final : public Backend {
public:
  /// Ctor.
  CPUBackend() = default;

  /// @name Backend methods.
  /// This is the implementation of the Backend interface.
  ///@{
  ~CPUBackend() override = default;

  std::unique_ptr<CompiledFunction>
  compile(std::unique_ptr<IRFunction> IR) const override;

  void save(std::unique_ptr<IRFunction> IR,
            llvm::StringRef outputDir) const override;

  bool transformPostLowering(Function *F, CompilationMode mode) const override;

  bool isOpSupported(Kinded::Kind opKind, ElemKind elementTy) const override;

  bool shouldLower(const Node *N) const override;
  /// @}
};

} // namespace glow

#endif // GLOW_BACKENDS_CPU_CPUBACKEND_H
