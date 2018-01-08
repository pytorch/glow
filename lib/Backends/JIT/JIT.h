#ifndef GLOW_BACKENDS_JIT_JIT_H
#define GLOW_BACKENDS_JIT_JIT_H

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

public:
  /// Ctor.
  explicit JITBackend(Module *M) : M_(M) {}

  /// @name Backend methods.
  /// This is the implementation of the Backend interface.
  ///@{
  ~JITBackend() override;

  void clear() override;

  void init() override;

  void doForwardPass(bool isTrain) override;

  void registerGraphTensor(const Value *v, Tensor *t) override;
  /// @}
};

/// Create a new instance of the JITBackend backend.
inline Backend *createJIT(Module *M) { return new JITBackend(M); }

} // namespace glow

#endif // GLOW_BACKENDS_JIT_JIT_H
