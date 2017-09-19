#include "glow/Interpreter/Interpreter.h"
#include "glow/IR/Instrs.h"
#include "glow/Support/Casting.h"

using namespace glow;

Interpreter::Interpreter() : M_(), builder_(M_) {}

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

Tensor *Interpreter::getTensorForValue(Value *v) const {
  auto it = tensors_.find(v);
  assert(it != tensors_.end() && "Unknown key Value.");
  return it->second;
}

Tensor *Interpreter::getOrCreateGradTensor(Value *v) {
  auto *T = getTensorForValue(v);
  auto it = gradients_.find(T);
  if (it != gradients_.end())
    return it->second;

  // Create a new tensor, register it and return it.
  Tensor *N = new Tensor(T->getType());
  gradients_[T] = N;
  return N;
}

void Interpreter::loadValueFromTensor(Value *v, Tensor *input,
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

void Interpreter::initVars() {
  for (auto *V : M_.getVars()) {
    auto SV = dyn_cast<StaticVariable>(V);
    // At the moment we only support static variables.

    if (!SV)
      continue;

    Tensor *T = nullptr;
    // Pick the tensor.
    auto it = tensors_.find(V);
    if (it == tensors_.end()) {
      T = new Tensor(V->getType());
      tensors_[V] = T;
    } else {
      T = it->second;
    }

    // The parameter to the instruction.
    auto val = SV->getVal();

    switch (SV->getInitKind()) {
    case StaticVariable::InitKind::kExtern:
      break;

    case StaticVariable::InitKind::kBroadcast: {
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

    case StaticVariable::InitKind::kXavier: {
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

  // Dispatch the interpreter on each instruction in the program:
#define DEF_VALUE(CLASS, NAME)
#define DEF_INSTR(CLASS, NAME)                                                 \
  case Kinded::Kind::CLASS##Kind: {                                            \
    fwd##CLASS(nullptr, false, cast<CLASS>(I));                                \
    break;                                                                     \
  }
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
  for (auto *V : M_.getVars()) {
    auto SV = dyn_cast<StaticVariable>(V);

    auto W = getTensorForValue(V);
    auto G = getOrCreateGradTensor(V);

    // Handle weight update by learning the gradients into the weights.
    if (SV->getShareKind() == StaticVariable::ShareKind::kWeight) {
      trainer_.train(W, G, batchSize);
      continue;
    }

    // Handle activation gradients by zeroing the grads and activations.
    W->zero();
    G->zero();
  }
}

void Interpreter::updateForwardBackward(ArrayRef<Value *> vars,
                                        ArrayRef<Tensor *> inputs,
                                        size_t sampleIdx) {
  // Update the input variables.
  for (int i = 0, e = vars.size(); i < e; i++) {
    loadValueFromTensor(vars[i], inputs[i], sampleIdx);
  }

  // Do the forward pass.
#define DEF_VALUE(CLASS, NAME)
#define DEF_INSTR(CLASS, NAME)                                                 \
  case Kinded::Kind::CLASS##Kind: {                                            \
    fwd##CLASS(nullptr, true, cast<CLASS>(I));                                 \
    break;                                                                     \
  }
  // Dispatch the interpreter on each instruction in the program:
  for (auto *I : M_.getInstrs()) {
    // Prepare for the next backprop iteration by zeroing the gradient
    // tensors. Notice that this only zeros the temporary grad tensors that
    // match the output tensors but not the gradient tensors that are
    // paired with filters. These are cleared during the learning process
    // at the end of the batch.
    getOrCreateGradTensor(I->getOperand(0).first)->zero();

    switch (I->getKind()) {
#include "glow/IR/Instrs.def"
    default:
      glow_unreachable();
    }
  }
#undef DEF_INSTR
#undef DEF_VALUE

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
