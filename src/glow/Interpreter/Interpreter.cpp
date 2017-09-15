#include "glow/Interpreter/Interpreter.h"

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

const Tensor *Interpreter::getTensorForValue(Value *v) const {
  auto it = tensors_.find(v);
  assert(it != tensors_.end() && "Unknown value key");
  return it->second;
}
