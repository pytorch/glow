#ifndef GLOW_BACKENDS_JIT_JIT_H
#define GLOW_BACKENDS_JIT_JIT_H

#include "GlowJIT.h"

#include "glow/Backends/Backend.h"
#include "glow/Base/Tensor.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"

namespace glow {

class Context;
class Module;
class Value;
class Tensor;
class Variable;
class Instruction;

class JITBackend final : public Backend {
  /// The Module that holds the glow IR. This does not own the module.
  Module *M_;
  /// The LLVM context.
  llvm::LLVMContext ctx_;
  /// The LLVM IR module.
  std::unique_ptr<llvm::Module> llmodule_{nullptr};
  /// Points to the main function in the jitted code. The function is owned by
  /// the LLVM module.
  llvm::Function *func_{nullptr};
  /// Maps Values in the module to their memory addresses.
  llvm::DenseMap<Value *, void *> allocatedAddressed_;
  /// Maps constant arrays to the constant expressions representing size_t
  /// pointers to these arrays. This is done to ensure the proper uniqueness
  /// semantics of such pointers just like it is done for llvm::Constants.
  llvm::DenseMap<llvm::Constant *, llvm::Value *> constArrayPtrs_;
  /// This represents the heap, that stores the activations at runtime.
  std::vector<uint8_t> heap_{};
  /// The LLVM JIT engine. The jit must be initialized after the ctor
  /// initializes the LLVM backends.
  std::unique_ptr<llvm::orc::GlowJIT> JIT_{nullptr};

  /// Assign memory addresses to activations, allocate the heap and register all
  /// weights and activations into the address-map.
  void allocateActivationsAndWeights();

  /// Generates LLVM IR that computes the address of \p val using \p builder.
  /// The address type is specified by \p ptrTy.
  llvm::Value *emitValueAddress(llvm::IRBuilder<> &builder, glow::Value *val,
                                ElemKind ptrTy);
  /// Generates LLVM IR that computes the size of the tensor of \p val using
  /// \p builder. The size type is native to the machine (size_t).
  llvm::Value *emitValueSize(llvm::IRBuilder<> &builder, glow::Value *val);
  /// Generates LLVM IR that materializes the constant \p val.
  llvm::Value *emitConst(llvm::IRBuilder<> &builder, float val);
  /// Generates LLVM IR that materializes the constant \p val.
  llvm::Value *emitConst(llvm::IRBuilder<> &builder, size_t val);
  /// Generates LLVM IR that materializes the constant array \p vals.
  llvm::Value *emitConstArray(llvm::IRBuilder<> &builder,
                              llvm::ArrayRef<size_t> vals);
  /// Generates LLVM IR that computes the dimensions of \p val using \p builder.
  /// The result type is "size_t*".
  llvm::Value *emitValueDims(llvm::IRBuilder<> &builder, glow::Value *val);

  /// \returns a function from the module that we are building on nullptr if
  /// none was found.
  llvm::Function *getFunction(const std::string &name);

  /// Emit LLVM-IR for the instruction \p I, using the builder \p builder.
  void generateLLVMIRForInstr(llvm::IRBuilder<> &builder, glow::Instruction *I);

  /// Optimize the function \p F and the module that owns it. Use the target
  /// information from the \p TM target machine.
  void optimizeLLVMModule(llvm::Function *F, llvm::TargetMachine &TM);

  /// Performs specialization of operations based on constant parameters.
  void performSpecialization();

public:
  /// Ctor.
  explicit JITBackend(Module *M);

  /// @name Backend methods.
  /// This is the implementation of the Backend interface.
  ///@{
  ~JITBackend() override;

  void clear() override;

  void init() override;

  void doForwardPass(bool isTrain) override;

  bool transform(Graph &G) override;
  /// @}
};

/// Create a new instance of the JITBackend backend.
inline Backend *createJIT(Module *M) { return new JITBackend(M); }

} // namespace glow

#endif // GLOW_BACKENDS_JIT_JIT_H
