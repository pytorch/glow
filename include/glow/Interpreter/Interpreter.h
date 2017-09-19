#ifndef GLOW_INTERPRETER_INTERPRETER_H
#define GLOW_INTERPRETER_INTERPRETER_H

#include "glow/IR/IR.h"
#include "glow/IR/IRBuilder.h"
#include "glow/Network/Tensor.h"

#include <unordered_map>

namespace glow {

class Context;

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
  Tensor *getTensorForValue(Value *v) const;

  /// \returns a float-handle to the tensor that is stored at \p v.
  Handle<FloatTy> getWeightHandle(Context *, Value *v) const;

  /// \returns a float-handle to the tensor that is stored at \p v.
  Handle<FloatTy> getGradHandle(Context *, Value *v) const;

  /// Initialize all of the variables in the program.
  void initVars();

  /// Runs the program in a forward pass.
  void infer();

#define DEF_VALUE(CLASS, NAME)
#define DEF_INSTR(CLASS, NAME)                                                 \
  void fwd##CLASS(Context *ctx, bool isTrain, CLASS *I);                       \
  void bwd##CLASS(Context *ctx, CLASS *I);
#include "glow/IR/Instrs.def"
#undef DEF_INSTR
#undef DEF_VALUE
};

} // namespace glow

#endif // GLOW_INTERPRETER_INTERPRETER_H
