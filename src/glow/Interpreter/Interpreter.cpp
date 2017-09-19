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
  assert(it != tensors_.end() && "Unknown value key");
  return it->second;
}

Handle<FloatTy> Interpreter::getWeightHandle(Context *ctx, Value *v) const {
  return getTensorForValue(v)->getHandle<FloatTy>();
}

Handle<FloatTy> Interpreter::getGradHandle(Context *ctx, Value *v) const {
  glow_unreachable();
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

    switch (SV->getMode()) {
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

void Interpreter::infer() {
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
