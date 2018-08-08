/**
 * Copyright (c) 2017-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "InterpreterFunction.h"

#include "glow/IR/IR.h"
#include "glow/IR/IRUtils.h"
#include "glow/IR/Instrs.h"

#include "llvm/Support/Casting.h"

using namespace glow;

InterpreterFunction::InterpreterFunction(std::unique_ptr<IRFunction> F)
    : F_(std::move(F)) {
  for (auto &v : F_->getGraph()->getParent()->getVars()) {
    auto *w = F_->getWeightForNode(v);
    assert(!externalTensors_.count(w) && "The tensor is already registered");
    externalTensors_[w] = &v->getPayload();
  }

  for (auto *W : F_->getWeights()) {
    getOrCreateTensor(W);
  }
}

InterpreterFunction::~InterpreterFunction() {
  // Delete the tensors that are owned by this backend.
  for (auto p : tensors_) {
    delete p.second;
  }
  tensors_.clear();
  externalTensors_.clear();
}

Tensor *InterpreterFunction::getTensor(const Value *v) const {
  auto it = tensors_.find(v);
  if (it != tensors_.end()) {
    return it->second;
  }

  auto ie = externalTensors_.find(v);
  assert(ie != externalTensors_.end() && "Unknown key Value.");
  return ie->second;
}

Tensor *InterpreterFunction::getOrCreateTensor(const Value *v) {
  auto ie = externalTensors_.find(v);
  if (ie != externalTensors_.end()) {
    return ie->second;
  }

  // Pick the tensor.
  auto it = tensors_.find(v);
  if (it == tensors_.end()) {
    auto *T = new Tensor(v->getType());
    tensors_[v] = T;
    return T;
  }
  return it->second;
}

Tensor *InterpreterFunction::getOrCreateUnownedTensor(
    const Value *v, const Value *src, llvm::ArrayRef<uint64_t> offsets) {
  assert(llvm::isa<TensorViewInst>(v) && "Expected a tensor view");

  // Pick the tensor.
  auto it = tensors_.find(v);

  // Release unowned tensors before re-creating them.
  if (it != tensors_.end()) {
    deleteTensor(v);
  }

  auto *T = new Tensor();
  *T = getTensor(src)->getUnowned(v->dims(), offsets);
  tensors_[v] = T;
  return T;
}

void InterpreterFunction::deleteTensor(const Value *v) {
  auto it = tensors_.find(v);
  if (it == tensors_.end()) {
    return;
  }

  delete it->second;
  tensors_.erase(it);
}

void InterpreterFunction::execute() {
// Do the forward pass.
#define DEF_VALUE(CLASS, NAME)
#define DEF_INSTR(CLASS, NAME)                                                 \
  case Kinded::Kind::CLASS##Kind: {                                            \
    fwd##CLASS(llvm::cast<CLASS>(&I));                                         \
    break;                                                                     \
  }
#define DEF_BACKEND_SPECIFIC_INSTR(CLASS, NAME)
  // Dispatch the interpreter on each instruction in the program:
  for (const auto &I : F_->getInstrs()) {
    switch (I.getKind()) {
#include "glow/AutoGenInstr.def"

    default:
      llvm_unreachable("Invalid instruction.");
    }
  }
}
