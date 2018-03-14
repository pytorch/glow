// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "glow/IR/Instrs.h"
#include "glow/IR/IR.h"
#include "glow/Support/Support.h"

#include "llvm/Support/Casting.h"

#include <cassert>

using namespace glow;
using llvm::cast;
using llvm::isa;

//===----------------------------------------------------------------------===//
//                      Instruction textual printers
//===----------------------------------------------------------------------===//

const char *WeightVar::getMutabilityStr(MutabilityKind kind) {
  const char *names[] = {"const", "mutable", nullptr};
  return names[static_cast<int>(kind)];
}

const char *WeightVar::getMutabilityStr() const {
  return getMutabilityStr(mut_);
}

void WeightVar::dump(llvm::raw_ostream &os) const {
  os << "%" << getName() << " = WeightVar ";
  os << *getType() << " " << getMutabilityStr();
}

//===----------------------------------------------------------------------===//
//                       Instruction verification
//===----------------------------------------------------------------------===//

void CopyInst::verify() const {
  auto *dest = getDest();
  auto *src = getSrc();
  assert(dest->getType() == src->getType() && "Invalid type.");
  // The operands of the copy instruction must be variables.
  assert(isa<AllocActivationInst>(dest) || isa<WeightVar>(dest) ||
         isa<TensorViewInst>(dest));
  assert(isa<AllocActivationInst>(src) || isa<WeightVar>(src) ||
         isa<TensorViewInst>(src));
}

void TensorViewInst::verify() const {
  assert(getSrc()->getType()->size() == getType()->size() &&
         "TensorView view size should be the same as Src size");
  assert(getSrc()->getElementType() == getType()->getElementType() &&
         "TensorView view element type should be the same as Src type");
}

void AllocActivationInst::verify() const {
  unsigned numDealloc = 0;
  for (const Use &U : getUsers()) {
    numDealloc += isa<DeallocActivationInst>(U.get());
  }

  // Make sure that there is exactly one user is a deallocation.
  assert(numDealloc == 1 && "Invalid number of tensor deallocation");
}

void DeallocActivationInst::verify() const {
  // The operand of this instruction needs to be an AllocActivationInst.
  assert(isa<AllocActivationInst>(getSrc()) && "Invalid operand");
}
