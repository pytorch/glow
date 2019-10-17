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
#ifndef GLOW_BACKENDS_CPU_CPULLVMIRGEN_H
#define GLOW_BACKENDS_CPU_CPULLVMIRGEN_H

#include "glow/LLVMIRCodeGen/LLVMIRGen.h"

namespace glow {

/// This is a class containing a common logic for the generation of the LLVM IR
/// from an IRFunction. The primary clients of this class are JITs and bundlers.
class CPULLVMIRGen : public LLVMIRGen {

public:
  /// Destructor
  virtual ~CPULLVMIRGen() = default;
  /// Ctor.
  explicit CPULLVMIRGen(const IRFunction *M, AllocationsInfo &allocationsInfo,
                        std::string mainEntryName, llvm::StringRef libjitBC);

  /// Emit LLVM-IR for the instruction \p I, using the builder \p builder.
  virtual void generateLLVMIRForInstr(llvm::IRBuilder<> &builder,
                                      const glow::Instruction *I) override;
  /// Emit IR for the data parallel instruction \p I which is invoked inside the
  /// stacked \p kernel. The current loop count is described by \p loopCount.
  /// The \p bufferToArgNum map can be used to find the required buffers, which
  /// are provided as arguments to the stacked \p kernel.
  virtual void generateLLVMIRForDataParallelInstr(
      llvm::IRBuilder<> &builder, const glow::Instruction *I,
      llvm::Function *kernel, llvm::DenseMap<Value *, int> &bufferToArgNum,
      llvm::Value *loopCount) override;
  /// Emit LLVM-IR for the whole IRFunction.
  virtual void generateLLVMIRForModule(llvm::IRBuilder<> &builder) override;
};

} // namespace glow

#endif // GLOW_BACKENDS_CPU_CPULLVMIRGEN_H
