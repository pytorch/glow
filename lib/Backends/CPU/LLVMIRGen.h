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
#ifndef GLOW_BACKENDS_JIT_LLVMIRGEN_H
#define GLOW_BACKENDS_JIT_LLVMIRGEN_H

#include "AllocationsInfo.h"
#include "glow/Base/Tensor.h"
#include "glow/IR/IR.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Target/TargetMachine.h"

namespace glow {

class Context;
class IRFunction;
class Value;
class Tensor;
class Variable;
class Instruction;
class WeightVar;
struct AllocationsInfo;

/// This is a class containing a common logic for the generation of the LLVM IR
/// from an IRFunction. The primary clients of this class are JITs and bundlers.
class LLVMIRGen {
  /// The Module that holds the glow IR. This does not own the module.
  IRFunction *F_;
  /// The LLVM context.
  llvm::LLVMContext ctx_;
  /// The LLVM IR module.
  std::unique_ptr<llvm::Module> llmodule_{nullptr};
  /// The target machine.
  std::unique_ptr<llvm::TargetMachine> TM_;
  /// Information about allocations.
  AllocationsInfo &allocationsInfo_;
  /// Name of the main entry.
  std::string mainEntryName_;
  /// Instruction number for the module.
  std::unique_ptr<InstructionNumbering> instrNumbering_;
  /// Value holding the base address of the activations memory area.
  llvm::Value *baseActivationsAddr_{nullptr};
  /// Value holding the base address of the constant WeightVars memory area.
  llvm::Value *baseConstantWeightVarsAddr_{nullptr};
  /// Value holding the base address of mutable WeightVars memory area.
  llvm::Value *baseMutableWeightVarsAddr_{nullptr};
  /// Value holding the address of the offsets array.
  llvm::Value *offsetsArray_{nullptr};
  /// Maps constant arrays to the constant expressions representing size_t
  /// pointers to these arrays. This is done to ensure the proper uniqueness
  /// semantics of such pointers just like it is done for llvm::Constants.
  llvm::DenseMap<llvm::Constant *, llvm::Value *> constArrayPtrs_;
  /// The IRBuilder used for the code generation.
  std::unique_ptr<llvm::IRBuilder<>> builder_;
  /// Output directory for bundles, debug info files, etc.
  llvm::StringRef outputDir_;
  /// Debug info emission support.
  struct DebugInfo {
    /// Source file for the main function.
    llvm::DIFile *mainFile_{nullptr};
    /// Debug info for the main function.
    llvm::DISubprogram *mainF_{nullptr};
    /// Line number for the first instruction in the textual representation of
    /// the Glow IR.
    size_t mainFileFirstInstrLineNo_{0};
    /// Debug info for the current compilation unit.
    llvm::DICompileUnit *compilationUnit_{nullptr};
    /// Mapping from LLVM types to DebugInfo types.
    llvm::DenseMap<llvm::Type *, llvm::DIType *> DITypes_;
    /// Global variable holding the base address of the constant WeightVars
    /// memory
    /// area. Used only when producing a debug information.
    llvm::GlobalVariable *constWeightsBaseAddressGV_{nullptr};
    /// Global variable holding the base address of mutable WeightVars memory
    /// area. Used only when producing a debug information.
    llvm::GlobalVariable *mutableWeightsBaseAddressGV_{nullptr};
    /// Global variable holding the base address of the activations memory area.
    /// Used only when producing a debug information.
    llvm::GlobalVariable *activationsBaseAddressGV_{nullptr};
  } dbgInfo_;
  /// Debug info builder.
  std::unique_ptr<llvm::DIBuilder> DIBuilder_;

  /// Generates LLVM IR that computes the address of \p val using \p builder.
  /// The address type is specified by \p ptrTy.
  llvm::Value *emitValueAddress(llvm::IRBuilder<> &builder, glow::Value *val);
  /// Generates LLVM IR that computes the size of the tensor of \p val using
  /// \p builder. The size type is native to the machine (size_t).
  llvm::Value *emitValueSize(llvm::IRBuilder<> &builder, glow::Value *val);
  /// Generates LLVM IR that materializes the constant \p val.
  llvm::Value *emitConstF32(llvm::IRBuilder<> &builder, float val);
  /// Generates LLVM IR that materializes the constant \p val.
  llvm::Value *emitConstI32(llvm::IRBuilder<> &builder, int32_t val);
  /// Generates LLVM IR that materializes the constant \p val.
  llvm::Value *emitConstI8(llvm::IRBuilder<> &builder, int8_t val);
  /// Generates LLVM IR that materializes the constant \p val.
  llvm::Value *emitConstSizeT(llvm::IRBuilder<> &builder, size_t val);
  /// Generates LLVM IR that materializes the constant \p val as a constant of
  /// the type specified by \p kind.
  llvm::Value *emitConst(llvm::IRBuilder<> &builder, float val,
                         glow::ElemKind kind);
  /// Generates LLVM IR that materializes the constant array \p vals.
  llvm::Value *emitConstArray(llvm::IRBuilder<> &builder,
                              llvm::ArrayRef<size_t> vals);

  /// Generates LLVM IR that materializes the constant array \p vals. Elements
  /// of vals have the type \p elemTy.
  llvm::Value *emitConstArray(llvm::IRBuilder<> &builder,
                              llvm::ArrayRef<llvm::Constant *> vals,
                              llvm::Type *elemTy);
  /// Generates LLVM IR that computes the dimensions of \p val using \p builder.
  /// The result type is "size_t*".
  llvm::Value *emitValueDims(llvm::IRBuilder<> &builder, glow::Value *val);
  /// Load base addresses of different memory areas (activations, const
  /// weightvars, mutable weight vars) so that they can be reused inside the
  /// body of the function.
  void loadBaseAddresses(llvm::IRBuilder<> &builder);
  /// Create a function representing a stacked kernel for instructions provided
  /// in \p stackedInstrs.
  void emitDataParallelKernel(llvm::IRBuilder<> &builder,
                              llvm::ArrayRef<Instruction *> stackedInstrs);
  /// Emit IR for the data parallel instruction \p I which is invoked inside the
  /// stacked \p kernel. The current loop count is described by \p loopCount.
  /// The \p bufferToArgNum map can be used to find the required buffers, which
  /// are provided as arguments to the stacked \p kernel.
  void generateLLVMIRForDataParallelInstr(
      llvm::IRBuilder<> &builder, glow::Instruction *I, llvm::Function *kernel,
      llvm::DenseMap<Value *, int> &bufferToArgNum, llvm::Value *loopCount);
  /// \returns the llvm type of the glow vale \p val.
  llvm::Type *getElementType(llvm::IRBuilder<> &builder, const Value *val);
  /// Create a debug information for a given LLVM type \p ty.
  llvm::DIType *getDebugType(llvm::IRBuilder<> &builder, llvm::Type *ty);
  /// Init the generation of debug information.
  void initDebugInfo();
  /// Generate debug information.
  void generateDebugInfo();
  /// Set the debug location for the \p builder, so that it corresponds to the
  /// instruction \p I in the textual representation of the Glow IR.
  void setCurrentDebugLocation(llvm::IRBuilder<> &builder,
                               glow::Instruction *I);
  /// Get or create a debug information for a given LLVM function.
  llvm::DISubprogram *getOrCreateFunctionDebugInfo(llvm::Function *F,
                                                   llvm::DIScope *scope,
                                                   llvm::DIFile *file,
                                                   unsigned lineNo);
  /// Emit a debug info for the logical global variable representing a weight or
  /// an activation described by \p val. This allows for inspecting the values
  /// of weights and activations when using a debugger. Logical global variables
  /// are not materialized and do not require any additional memory to be
  /// reserved or allocated. Instead, they reside at offsets described by
  /// AllocationsInfo inside the memory blocks dynamically allocated by clients
  /// for weights and activations, but behave like regular global variables from
  /// the debugger's perspective.
  void emitDebugGlobalVariableForValue(const Value *val);

public:
  /// Ctor.
  explicit LLVMIRGen(IRFunction *M, AllocationsInfo &allocationsInfo,
                     std::string mainEntryName);

  /// Init the TargetMachine using a given target and code model.
  void initTargetMachine(llvm::StringRef T, llvm::CodeModel::Model CM);

  /// Emit LLVM-IR for the instruction \p I, using the builder \p builder.
  void generateLLVMIRForInstr(llvm::IRBuilder<> &builder, glow::Instruction *I);
  /// Emit LLVM-IR for the whole IRFunction.
  void generateLLVMIRForModule(llvm::IRBuilder<> &builder);
  /// \returns a libjit API function by name.
  llvm::Function *getFunction(const std::string &name);
  /// \returns a libjit API function by name and tensor element type.
  llvm::Function *getFunction(const std::string &name, glow::ElemKind elemTy);
  /// Creates global variables for the base addresses of different memory areas
  /// and invokes a library function to set their values.
  void
  createGlobalVariables(llvm::IRBuilder<> &builder,
                        llvm::ArrayRef<llvm::Value *> initFunctionCallArgs);
  /// Optimize the function \p F and the module that owns it. Use the target
  /// information from the \p TM target machine.
  void optimizeLLVMModule(llvm::Function *F, llvm::TargetMachine &TM);
  /// Performs specialization of operations based on constant parameters.
  void performSpecialization();
  /// Performs specialization of a call based on constant parameters.
  /// In case of a successful specialization, the old call instruction is
  /// replaced by the new one and the old one is erases. \returns the new
  /// specialized call or nullptr if no specialization was possible.
  llvm::CallInst *specializeCallWithConstantArguments(llvm::CallInst *call);
  /// \returns allocations info.
  AllocationsInfo &getAllocationsInfo() { return allocationsInfo_; }
  /// \returns the name of the main entry point.
  /// When JITting, it will be "main". In case of bundling it will be the name
  /// of the bundle.
  std::string getMainEntryName() const;
  /// Set the name of the main entry point.
  void setMainEntryName(std::string name);
  /// Creates an LLVM module, the entry function, etc.
  void initCodeGen();
  /// Emits the code of the entry function, performs optimizations, etc.
  void performCodeGen();
  /// \returns the current builder.
  llvm::IRBuilder<> &getBuilder() { return *builder_; }
  /// \returns the target machine description.
  llvm::TargetMachine &getTargetMachine() { return *TM_; }
  /// \returns the LLVMContext being used.
  llvm::LLVMContext &getLLVMContext() { return ctx_; }
  /// Borrows the LLVM module for further processing, e.g. by a JIT.
  /// The module cannot be used by the LLVMIRGen afterwards.
  std::unique_ptr<llvm::Module> borrowModule() { return std::move(llmodule_); }
  /// \returns current LLVM module.
  llvm::Module &getModule() { return *llmodule_; }
  /// \returns the IR function.
  IRFunction *getIRFunction() { return F_; }
  /// Set output directory for bundles, debug info files, etc.
  void setOutputDir(llvm::StringRef outputDir) { outputDir_ = outputDir; }
  /// Get output directory for bundles, debug info files, etc.
  llvm::StringRef getOutputDir() const { return outputDir_; }
  /// Emit the array of constant offsets as provided by the \p allocationsInfo.
  llvm::Value *emitConstOffsetsArray(llvm::IRBuilder<> &builder,
                                     const AllocationsInfo &allocationsInfo);
  /// Generate debug info for a LLVM function \p F.
  void generateFunctionDebugInfo(llvm::Function *F);
  /// Generates LLVM IR that materializes the string literal \p str.
  llvm::Value *emitStringConst(llvm::IRBuilder<> &builder, llvm::StringRef str);
};

} // namespace glow

#endif // GLOW_BACKENDS_JIT_LLVMIRGEN_H
