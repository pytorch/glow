#ifndef GLOW_INTERPRETER_INTERPRETER_H
#define GLOW_INTERPRETER_INTERPRETER_H

#include "glow/Base/Tensor.h"
#include "glow/Base/Train.h"
#include "glow/Graph/Graph.h"
#include "glow/IR/IR.h"
#include "glow/IR/IRBuilder.h"
#include "glow/Optimizer/Optimizer.h"

#include "llvm/ADT/ArrayRef.h"

#include <unordered_map>

namespace glow {

class Context;

/// This is the IR-interpreter. It owns the IR, and the heap, and is able to
/// execute the instructions one at a time.
class Interpreter final {
  /// The Module that holds the IR.
  Module &M_;
  /// Maps values to Tensors, that are owned by this class.
  std::unordered_map<const Value *, Tensor *> tensors_;

  /// Maps weight tensors to the gradients that update them. The value tensors
  /// are owned by this map.
  std::unordered_map<Tensor *, Tensor *> gradients_;

public:
  /// \returns the internal module.
  Module &getModule() { return M_; }

  /// Ctor.
  Interpreter(Module &M) : M_(M) {}
  /// Dtor.
  ~Interpreter();

  /// \returns a pointer to the tensor that is saved under \p v. The tensor
  /// is owned by the Interpreter.
  Tensor *getTensor(const Value *v) const;

  /// Allocate a tensor to back the value \p v. Do not allocate anything if a
  /// tensor is already allocated for \p v.
  /// \returns a tensor for \p v.
  Tensor *getOrCreateTensor(const Value *v);

  /// \returns True if a tensor was allocated for \p v.
  bool hasTensor(const Value *v);

  /// Copies the content of the tensor \p t into the value \p v.
  void initValue(const Value *v, const Tensor *t);

  /// \returns gets or creates a new tensor to back the value \p v. If the
  /// tensor does not exist then this method creates it. The dimension of the
  /// gradient tensor must match the dimensions of the tensor that backs \p v.
  Tensor *getOrCreateGradTensor(const Value *v);

  /// \returns a float-handle to the tensor that is stored at \p v.
  Handle<FloatTy> getWeightHandle(Value *v) const;

  /// \returns a float-handle to the tensor that is stored at \p v.
  Handle<FloatTy> getGradHandle(Value *v);

  /// Perform a single forward scan of the network, interpreting all of the
  /// instructions.
  void doForwardPass(bool isTrain);

  /// Perform a single backward scan of the network, interpreting all of the
  /// instructions.
  void doBackwardPass();

private:
  /// @name Interpreter methods. This is a list of method declerations that are
  /// used by the interpreter to dispatch different instructions.
  ///@{
#define DEF_VALUE(CLASS, NAME)
#define DEF_NODE(CLASS, NAME)
#define DEF_INSTR(CLASS, NAME)                                                 \
  void fwd##CLASS(Context *ctx, bool isTrain, const CLASS *I);                 \
  void bwd##CLASS(Context *ctx, const CLASS *I);
#include "glow/IR/Instrs.def"

  void fwdPoolMax_impl(Context *ctx, const PoolInst *I);
  void fwdPoolAvg_impl(Context *ctx, const PoolInst *I);
  void bwdPoolMax_impl(Context *ctx, const PoolInst *I);
  void bwdPoolAvg_impl(Context *ctx, const PoolInst *I);

  void fwdBatchNormalizationInst_infer(Context *ctx,
                                       const BatchNormalizationInst *I);
  void fwdBatchNormalizationInst_train(Context *ctx,
                                       const BatchNormalizationInst *I);
  ///@}
};

} // namespace glow

#endif // GLOW_INTERPRETER_INTERPRETER_H
