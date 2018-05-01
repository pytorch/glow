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
#ifndef GLOW_BACKENDS_JIT_JIT_H
#define GLOW_BACKENDS_JIT_JIT_H

#include "AllocationsInfo.h"
#include "GlowJIT.h"
#include "LLVMIRGen.h"
#include "glow/Backends/Backend.h"
#include "glow/Base/Tensor.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"

namespace glow {

class Context;
class IRFunction;
class Value;
class Tensor;
class Variable;
class Instruction;
class WeightVar;
struct AllocationsInfo;

class CPUBackend final : public Backend {
  /// The Module that holds the glow IR. This does not own the module.
  IRFunction *F_;
  /// Information about allocations.
  AllocationsInfo allocationsInfo_;
  /// The LLVM JIT engine. The jit must be initialized after the ctor
  /// initializes the LLVM backends.
  std::unique_ptr<llvm::orc::GlowJIT> JIT_{nullptr};

  /// The LLVM IR code generator.
  LLVMIRGen irgen_;
  /// This represents the heap, that stores the activations at runtime.
  void *heap_{nullptr};
  /// Produce the main entry point for JIT execution.
  void emitJitMain();
  /// Perform memory allocation for a JIT execution.
  void performJITMemoryAllocation();
  /// Perform memory allocation for a bundle.
  void performBundleMemoryAllocation();
  /// Save weights for the bundle.
  void saveWeights(llvm::StringRef weightsFileName);
  /// Produce a bundle.
  void produceBundle(llvm::StringRef outputDir);
  /// Emit config for a bundle.
  void emitBundleConfig();
  /// Emit the symbol table for a bundle.
  void emitSymbolTable();
  /// Emit the entry function for the bundle.
  void emitBundleEntryFunction();

public:
  /// Ctor.
  explicit CPUBackend(IRFunction *M);

  /// @name Backend methods.
  /// This is the implementation of the Backend interface.
  ///@{
  ~CPUBackend() override;

  void clear() override;

  void init() override;

  void save(llvm::StringRef outputDir) override;

  void doForwardPass() override;

  bool transformPostLowering(Function *F, CompilationMode mode) override;

  bool isOpSupported(Kinded::Kind opKind, ElemKind elementTy) const override;
  /// @}
};

/// Create a new instance of the JITBackend backend.
inline Backend *createCPUBackend(IRFunction *M) { return new CPUBackend(M); }

} // namespace glow

#endif // GLOW_BACKENDS_JIT_JIT_H
