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

class JITBackend final : public Backend {
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
  std::vector<uint8_t> heap_{};
  /// Produce the main entry point for JIT execution.
  void emitJitMain();
  /// Perform memory allocation for a JIT execution.
  void performJITMemoryAllocation();
  
public:
  /// Ctor.
  explicit JITBackend(IRFunction *M);

  /// @name Backend methods.
  /// This is the implementation of the Backend interface.
  ///@{
  ~JITBackend() override;

  void clear() override;

  void init() override;

  void doForwardPass(bool isTrain) override;

  bool transform(Function *F) override;
  /// @}
};

/// Create a new instance of the JITBackend backend.
inline Backend *createJIT(IRFunction *M) { return new JITBackend(M); }

} // namespace glow

#endif // GLOW_BACKENDS_JIT_JIT_H
