#ifndef GLOW_INTERPRETER_INTERPRETER_H
#define GLOW_INTERPRETER_INTERPRETER_H

#include "glow/Backends/Backend.h"
#include "glow/Base/Tensor.h"

#include "llvm/ADT/ArrayRef.h"

#include <unordered_map>

namespace glow {

class Context;
class IRFunction;
class Value;
class Tensor;
class Variable;

// Forward declare all of the classes.
#define DEF_VALUE(CLASS, NAME) class CLASS;
#define DEF_INSTR(CLASS, NAME) class CLASS;
#define DEF_BACKEND_SPECIFIC_INSTR(CLASS, NAME)
#include "AutoGenInstr.def"

/// This is the IR-interpreter. It owns the IR, and the heap, and is able to
/// execute the instructions one at a time.
class Interpreter final : public Backend {
  /// The Module that holds the IR. This does not own the module.
  IRFunction *F_;
  /// Maps values to Tensors, that are owned by this class.
  std::unordered_map<const Value *, Tensor *> tensors_;

  /// Maps values to Tensors, that are *not* owned by this class.
  std::unordered_map<const Value *, Tensor *> externalTensors_;

public:
  /// Ctor.
  explicit Interpreter(IRFunction *F) : F_(F) {}

  /// @name Backend methods.
  /// This is the implementation of the Backend interface.
  ///@{
  ~Interpreter() override;

  void clear() override;

  void init() override;

  void doForwardPass() override;

  bool isOpSupported(Kinded::Kind opKind, ElemKind elementTy) const override;
  /// @}

private:
  /// \returns a pointer to the tensor that is saved under \p v.
  Tensor *getTensor(const Value *v) const;

  /// Allocate a tensor to back the value \p v. Do not allocate anything if a
  /// tensor is already allocated for \p v.
  /// \returns a tensor for \p v.
  Tensor *getOrCreateTensor(const Value *v);

  /// Allocate an unowned tensor to back the value \p v. The source tensor of
  /// the unowned tensor is provided by \p src.
  /// \returns a tensor for \p v.
  Tensor *getOrCreateUnownedTensor(const Value *v, const Value *src);

  /// If a tensor is allocated for \p v then delete it.
  void deleteTensor(const Value *v);

  /// \returns a typed handle to the tensor that is stored at \p v.
  template <class ElemTy = float>
  Handle<ElemTy> getWeightHandle(Value *v) const {
    return getTensor(v)->getHandle<ElemTy>();
  }

  /// @name Interpreter methods. This is a list of method declerations that are
  /// used by the interpreter to dispatch different instructions.
  ///@{

#define DEF_VALUE(CLASS, NAME)
#define DEF_INSTR(CLASS, NAME) void fwd##CLASS(const CLASS *I);
#define DEF_BACKEND_SPECIFIC_INSTR(CLASS, NAME)
#include "AutoGenInstr.def"

  void fwdConvolutionInst_I8Impl(Value *inV, Value *outV, Value *filterV,
                                 Value *biasV, size_t filterSize, size_t stride,
                                 size_t pad);
  void fwdConvolutionInst_FloatImpl(Value *inV, Value *outV, Value *filterV,
                                    Value *biasV, size_t filterSize,
                                    size_t stride, size_t pad);
  ///@}
};

/// Create a new instance of the Interpreter backend.
inline Backend *createInterpreter(IRFunction *M) { return new Interpreter(M); }

} // namespace glow

#endif // GLOW_INTERPRETER_INTERPRETER_H
