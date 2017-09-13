#include "glow/IR/IR.h"

using namespace glow;

TypeRef Module::uniqueType(const Type &T) {
  for (auto &tp : types_) {
    if (T.isEqual(tp))
      return &tp;
  }

  return &*types_.insert(types_.begin(), T);
}

void Value::removeUse(Value::Use U) {
  auto it = std::find(users_.begin(), users_.end(), U);
  assert(it != users_.end() && "User not in list");
  users_.erase(it);
}

void Value::addUse(Use U) { users_.push_back(U); }

bool Value::hasUser(Instruction *I) {
  for (auto &U : users_) {
    if (U.second == I)
      return true;
  }
  return false;
}

void Value::replaceAllUsesOfWith(Value *v) {
  for (auto &U : users_) {
    U.second->setOperand(U.first, v);
  }
}

void Instruction::pushOperand(Value *v) {
  ops_.push_back(nullptr);
  setOperand(ops_.size() - 1, v);
}

void Instruction::setOperand(unsigned idx, Value *v) {
  Value *currVal = ops_[idx];

  if (currVal == v)
    return;

  if (currVal) {
    currVal->removeUse({idx, this});
  }

  if (v) {
    ops_[idx] = v;
    v->addUse({idx, this});
  }
}

Value *Instruction::getOperand(unsigned idx) {
  assert(ops_.size() > idx && "Invalid operand");
  return ops_[idx];
}

void Instruction::verifyUseList() {
  for (int i = 0, e = ops_.size(); i < e; i++) {
    Value *v = ops_[i];
    assert(v && "Instruction operand must be a real value");
    assert(v->hasUser(this) && "Invalid use-list");
    assert(v != this && "Use-list cycle");
  }
}
