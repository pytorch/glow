#include "glow/Interpreter/Interpreter.h"
#include "glow/IR/Instrs.h"
#include "glow/Optimizer/Optimizer.h"
#include "glow/Support/Casting.h"

using namespace glow;

Interpreter::Interpreter() {}

Interpreter::~Interpreter() {
  // Delete the tensors that are owned by this module.
  for (auto p : tensors_) {
    delete p.second;
  }

  // Delete the attached gradients.
  for (auto &p : gradients_) {
    delete p.second;
  }
}

void Interpreter::optimize() {
  ::glow::optimize(M_);
}

void Interpreter::registerTensor(Value *v, Tensor *t) {
  assert(t->getType().isEqual(v->getType()) &&
         "Tensor must match variable dimensions");

  auto it = tensors_.find(v);
  if (it != tensors_.end()) {
    delete it->second;
    it->second = t;
    return;
  }
  tensors_[v] = t;
}

Tensor *Interpreter::getTensorForValue(const Value *v) const {
  auto it = tensors_.find(v);
  assert(it != tensors_.end() && "Unknown key Value.");
  return it->second;
}

void Interpreter::deleteTensor(const Value *v) {
  auto it = tensors_.find(v);
  assert(it != tensors_.end() && "Unknown key Value.");
  auto *T = it->second;
  delete T;
  tensors_.erase(it);

  auto git = gradients_.find(T);
  if (git != gradients_.end()) {
    delete git->second;
    gradients_.erase(git);
  }
}

void Interpreter::initValue(const Value *v, const Tensor *t) {
  auto it = tensors_.find(v);
  if (it != tensors_.end()) {
    it->second->copyFrom(t);
  }

  // Create a new tensor, register it and return it.
  auto *N = new Tensor();
  N->copyFrom(t);
  tensors_[v] = N;
}

Tensor *Interpreter::getOrCreateGradTensor(const Value *v) {
  auto *T = getTensorForValue(v);
  auto it = gradients_.find(T);
  if (it != gradients_.end()) {
    return it->second;
  }

  // Create a new tensor, register it and return it.
  auto *N = new Tensor(T->getType());
  gradients_[T] = N;
  return N;
}

void Interpreter::loadValueFromTensor(const Value *v, Tensor *input,
                                      size_t sampleIdx) {
  auto *t = getTensorForValue(v);

  auto dim = input->dims();
  assert(t->dims().drop_front() == dim.drop_front() && "Invalid slice size");
  // Extract the n'th slice, that must be a tensor.
  size_t slc = sampleIdx % dim[0];
  t->copyConsecutiveSlices(input, slc);
}

Handle<FloatTy> Interpreter::getWeightHandle(Context *ctx, Value *v) const {
  return getTensorForValue(v)->getHandle<FloatTy>();
}

Handle<FloatTy> Interpreter::getGradHandle(Context *ctx, Value *v) {
  return getOrCreateGradTensor(v)->getHandle<FloatTy>();
}

Tensor *Interpreter::allocateBackingTensor(const Value *v) {
  // Allocate a tensor for the variable.
  Tensor *T = nullptr;
  // Pick the tensor.
  auto it = tensors_.find(v);
  if (it == tensors_.end()) {
    T = new Tensor(v->getType());
    tensors_[v] = T;
    return T;
  }
  return it->second;
}

void Interpreter::initVars() {
  for (auto *W : M_.getWeights()) {
    // Don't initialize tensors that are already initialized.
    if (tensors_.count(W)) {
      continue;
    }

    auto *T = allocateBackingTensor(W);
    // The parameter to the instruction.
    auto val = W->getVal();

    switch (W->getInitKind()) {
    case WeightVar::InitKind::kExtern:
      break;

    case WeightVar::InitKind::kBroadcast: {
      switch (T->getElementType()) {
      case ElemKind::FloatTy: {
        T->getHandle<float>().clear(val);
        break;
      }
      case ElemKind::DoubleTy: {
        T->getHandle<double>().clear(val);
        break;
      }
      case ElemKind::Int8Ty: {
        T->getHandle<int8_t>().clear(val);
        break;
      };
      case ElemKind::Int32Ty: {
        T->getHandle<int32_t>().clear(val);
        break;
      }
      case ElemKind::IndexTy: {
        T->getHandle<size_t>().clear(val);
        break;
      }
      }
      break;
    }

    case WeightVar::InitKind::kXavier: {
      switch (T->getElementType()) {
      case ElemKind::FloatTy: {
        T->getHandle<float>().randomize(val);
        break;
      }
      case ElemKind::DoubleTy: {
        T->getHandle<double>().randomize(val);
        break;
      }
      case ElemKind::Int8Ty: {
        T->getHandle<int8_t>().randomize(val);
        break;
      };
      case ElemKind::Int32Ty: {
        T->getHandle<int32_t>().randomize(val);
        break;
      }
      case ElemKind::IndexTy: {
        T->getHandle<size_t>().randomize(val);
        break;
      }
      }
      break;
    }
    }
  }
}

void Interpreter::infer(ArrayRef<Value *> vars, ArrayRef<Tensor *> inputs) {
  assert(!inputs.empty() && "No inputs");
  assert(inputs.size() == vars.size() &&
         "The number of inputs does not match the number of variables");

  // Update the input variables.
  for (int i = 0, e = vars.size(); i < e; i++) {
    loadValueFromTensor(vars[i], inputs[i], 0);
  }

  doForwardPass(false);
}

void Interpreter::train(size_t iterations, ArrayRef<Value *> vars,
                        ArrayRef<Tensor *> inputs) {
  static size_t trainCounter = 0;

  assert(!inputs.empty() && "No inputs");
  assert(inputs.size() == vars.size() &&
         "The number of inputs does not match the number of variables");

  // This is the size of one batch (the number of samples in the batch).
  size_t batchSize = vars[0]->dims()[0];

  for (size_t i = 0; i < iterations; i++) {
    // Launch threads that update the different chunks in the batch:
    updateForwardBackward(vars, inputs, trainCounter + batchSize);

    trainCounter += batchSize;

    // The algorithm for merging the state from the different threads is
    /// described in the paper: Alex Krizhevsky [2014]
    // "One weird trick for parallelizing convolutional neural networks"
    learnGradient(batchSize);
  }
}

void Interpreter::learnGradient(size_t batchSize) {
  for (auto *V : M_.getWeights()) {
    // Do not try to learn the values of input/output buffers.
    if (V->getInitKind() == WeightVar::InitKind::kExtern) {
      continue;
    }

    auto W = getTensorForValue(V);
    auto G = getOrCreateGradTensor(V);

    // Handle weight update by learning the gradients into the weights.
    trainer_.train(W, G, batchSize);
  }
}

void Interpreter::updateForwardBackward(ArrayRef<Value *> vars,
                                        ArrayRef<Tensor *> inputs,
                                        size_t sampleIdx) {
  // Update the input variables.
  for (int i = 0, e = vars.size(); i < e; i++) {
    loadValueFromTensor(vars[i], inputs[i], sampleIdx);
  }

  doForwardPass(true);

  doBackwardPass();
}

void Interpreter::doForwardPass(bool isTrain) {

  // Do the forward pass.
#define DEF_VALUE(CLASS, NAME)
#define DEF_INSTR(CLASS, NAME)                                                 \
  case Kinded::Kind::CLASS##Kind: {                                            \
    fwd##CLASS(nullptr, isTrain, cast<CLASS>(I));                              \
    break;                                                                     \
  }
  // Dispatch the interpreter on each instruction in the program:
  for (auto *I : M_.getInstrs()) {
    switch (I->getKind()) {
#include "glow/IR/Instrs.def"
    default:
      glow_unreachable();
    }
  }
#undef DEF_INSTR
#undef DEF_VALUE
}

void Interpreter::doBackwardPass() {
  // Do the backward pass.
#define DEF_VALUE(CLASS, NAME)
#define DEF_INSTR(CLASS, NAME)                                                 \
  case Kinded::Kind::CLASS##Kind: {                                            \
    bwd##CLASS(nullptr, cast<CLASS>(*it));                                     \
    break;                                                                     \
  }
  // Dispatch the interpreter on each instruction in the program, in reverse
  // order.
  auto &L = M_.getInstrs();
  for (auto it = L.rbegin(), e = L.rend(); it != e; it++) {
    switch ((*it)->getKind()) {
#include "glow/IR/Instrs.def"
    default:
      glow_unreachable();
    }
  }
#undef DEF_INSTR
#undef DEF_VALUE
}
