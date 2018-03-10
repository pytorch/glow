#ifndef GLOW_BACKENDS_JIT_LLVMIRGEN_H
#define GLOW_BACKENDS_JIT_LLVMIRGEN_H

#include "AllocationsInfo.h"
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
  llvm::Value *emitConstSizeT(llvm::IRBuilder<> &builder, size_t val);
  /// Generates LLVM IR that materializes the string literal \p str.
  llvm::Value *emitStringConst(llvm::IRBuilder<> &builder, llvm::StringRef str);
  /// Generates LLVM IR that materializes the constant array \p vals.
  llvm::Value *emitConstArray(llvm::IRBuilder<> &builder,
                              llvm::ArrayRef<size_t> vals);
  /// Generates LLVM IR that computes the dimensions of \p val using \p builder.
  /// The result type is "size_t*".
  llvm::Value *emitValueDims(llvm::IRBuilder<> &builder, glow::Value *val);
  /// Load base addresses of different memory areas (activations, const
  /// weightvars, mutable weight vars) so that they can be reused inside the
  /// body of the function.
  void loadBaseAddresses(llvm::IRBuilder<> &builder);

public:
  /// Ctor.
  explicit LLVMIRGen(IRFunction *M, AllocationsInfo &allocationsInfo,
                     std::string mainEntryName);

  /// Init the TargetMachine using a given target and code model.
  void initTargetMachine(llvm::StringRef T, llvm::CodeModel::Model CM);

  /// Emit LLVM-IR for the instruction \p I, using the builder \p builder.
  void generateLLVMIRForInstr(llvm::IRBuilder<> &builder, glow::Instruction *I);
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
  /// Emit the array of constant offsets as provided by the \p allocationsInfo.
  llvm::Value *emitConstOffsetsArray(llvm::IRBuilder<> &builder,
                                     const AllocationsInfo &allocationsInfo);
};

} // namespace glow

#endif // GLOW_BACKENDS_JIT_LLVMIRGEN_H
