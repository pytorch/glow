#ifndef GLOW_BACKENDS_JIT_JIT_H
#define GLOW_BACKENDS_JIT_JIT_H

#include "glow/Backends/Backend.h"
#include "glow/Base/Tensor.h"

#include "llvm/ADT/ArrayRef.h"

namespace glow {

class Context;
class Module;
class Value;
class Tensor;
class Variable;

class JITBackend final : public Backend {
  /// The Module that holds the IR. This does not own the module.
  Module *M_;

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
