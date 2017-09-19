#ifndef GLOW_INTERPRETER_INTERPRETER_H
#define GLOW_INTERPRETER_INTERPRETER_H

#include "glow/IR/IR.h"
#include "glow/IR/IRBuilder.h"
#include "glow/Network/Tensor.h"
#include "glow/Network/Train.h"

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

  /// Maps weight tensors to the gradients that update them. The value tensors
  /// are owned by this map.
  std::unordered_map<Tensor *, Tensor *> gradients_;

  /// The network trainer.
  Trainer trainer_;

public:
  /// \returns the internal module.
  Module &getModule() { return M_; }
  /// \returns the internal IR builder.
  IRBuilder &getBuilder() { return builder_; }
  /// Ctor.
  Interpreter();
  /// Dtor.
  ~Interpreter();

  /// Provides access to the training configuration.
  TrainingConfig &getConfig() { return trainer_.config; }

  /// Registers the tensor \p t under the IR value \p v.
  void registerTensor(Value *v, Tensor *t);

  /// \returns a pointer to the tensor that is saved under \p v. The tensor
  /// is owned by the Interpreter.
  Tensor *getTensorForValue(Value *v) const;

  /// \returns gets or creates a new tensor to back the value \p v. If the
  /// tensor does not exist then this method creates it. The dimension of the
  /// gradient tensor must match the dimensions of the tensor that backs \p v.
  Tensor *getOrCreateGradTensor(Value *v);

  /// Update the content of the tensor \p v with data that comes from \p input.
  /// The data starts at slice \p sampleIdx and wraps around until the data in
  /// \p v is filled. All dimensions, except for the first (batch) dimension
  /// must be identical.
  void loadValueFromTensor(Value *v, Tensor *input, size_t sampleIdx);

  /// \returns a float-handle to the tensor that is stored at \p v.
  Handle<FloatTy> getWeightHandle(Context *, Value *v) const;

  /// \returns a float-handle to the tensor that is stored at \p v.
  Handle<FloatTy> getGradHandle(Context *, Value *v);

  /// Initialize all of the variables in the program.
  void initVars();

  /// Runs the program in a forward pass. Update the nodes in \p nodes with the
  /// values \p inputs.
  void infer(ArrayRef<Value *> vars, ArrayRef<Tensor *> inputs);

  /// Train the network. Perform \p iterations in the training loop. Each
  /// iteration does a full forward and backward pass of a whole batch.
  /// The method updates the variables in \p vars with the tensors \p inputs.
  void train(size_t iterations, ArrayRef<Value *> vars,
             ArrayRef<Tensor *> inputs);

private:
  /// Update all of the weight tensors (non-activation) with their gradients.
  void learnGradient(size_t batchSize);

  /// Update the inputs for all variables \p vars with data from the inputs \p
  /// inputs at offset \p sampleIdx. Then perform a forward and backwards scan.
  void updateForwardBackward(ArrayRef<Value *> vars, ArrayRef<Tensor *> inputs,
                             size_t sampleIdx);

  /// Perform a single forward scan of the network, interpreting all of the
  /// instructions.
  void doForwardPass(bool isTrain);

  /// Perform a single backward scan of the network, interpreting all of the
  /// instructions.
  void doBackwardPass();

  /// @name Interpreter methods. This is a list of method declerations that are
  /// used by the interpreter to dispatch different instructions.
  ///@{
#define DEF_VALUE(CLASS, NAME)
#define DEF_INSTR(CLASS, NAME)                                                 \
  void fwd##CLASS(Context *ctx, bool isTrain, CLASS *I);                       \
  void bwd##CLASS(Context *ctx, CLASS *I);
#include "glow/IR/Instrs.def"
#undef DEF_INSTR
#undef DEF_VALUE

  void fwdPoolMax_impl(Context *ctx, PoolInst *I);
  void fwdPoolAvg_impl(Context *ctx, PoolInst *I);
  void bwdPoolMax_impl(Context *ctx, PoolInst *I);
  void bwdPoolAvg_impl(Context *ctx, PoolInst *I);
  ///@}
};

} // namespace glow

#endif // GLOW_INTERPRETER_INTERPRETER_H
