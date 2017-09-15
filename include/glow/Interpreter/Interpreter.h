#ifndef GLOW_INTERPRETER_INTERPRETER_H
#define GLOW_INTERPRETER_INTERPRETER_H

#include "glow/IR/IR.h"
#include "glow/IR/IRBuilder.h"
#include "glow/Network/Tensor.h"

#include <unordered_map>

namespace glow {

/// This is the IR-interpreter. It owns the IR, and the heap, and is able to
/// execute the instructions one at a time.
class Interpreter final {
  /// The Module that holds the IR.
  Module M_;
  /// The IR Builder.
  IRBuilder builder_;
  /// Maps values to Tensors, that are owned by this class.
  std::unordered_map<Value *, Tensor *> tensors_;

public:
  /// \returns the internal module.
  Module &getModule() { return M_; }
  /// \returns the internal IR builder.
  IRBuilder &getBuilder() { return builder_; }
  /// Ctor.
  Interpreter();
  /// Dtor.
  ~Interpreter();

  /// Registers the tensor \p t under the IR value \p v.
  void registerTensor(Value *v, Tensor *t);

  /// \returns a pointer to the tensor that is saved under \p v. The tensor
  /// is owned by the Interpreter.
  const Tensor *getTensorForValue(Value *v) const;
};

} // namespace glow

#endif // GLOW_INTERPRETER_INTERPRETER_H
