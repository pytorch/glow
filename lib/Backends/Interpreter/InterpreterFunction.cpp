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

#include "llvm/Support/raw_ostream.h"
using namespace glow;

InterpreterFunction::InterpreterFunction(std::unique_ptr<IRFunction> F,
                                         const runtime::RuntimeBundle &bundle)
    : CompiledFunction(bundle), F_(std::move(F)) {}

InterpreterFunction::~InterpreterFunction() {
  for (const auto &p : constants_) {
    delete p.second;
  }
  constants_.clear();

  alignedFree(runtimeBundle_.getConstants());
  tearDownRuns();
}

void InterpreterFunction::collectConstants(IRFunction *F) {
  runtimeBundle_.collectConstants(F);
  if (constants_.empty()) {
    if (runtimeBundle_.getConstantWeightSize()) {
      for (const auto &v : F_->getGraph()->getParent()->getConstants()) {
        auto symbolInfo = runtimeBundle_.getSymbolInfo(v);
        auto addr = runtimeBundle_.getConstants() + symbolInfo.offset;
        auto tensor = new Tensor(addr, &symbolInfo.type);
        constants_.emplace(std::string(v->getName()), tensor);
      }
    }
  }
}

void InterpreterFunction::execute(Context *ctx) {
  if (constants_.empty()) {
    collectConstants(F_.get());
  }
  BoundInterpreterFunction boundFunc(constants_);
  boundFunc.execute(F_.get(), ctx);
}

BoundInterpreterFunction::~BoundInterpreterFunction() {
  // Delete the tensors that are owned by this backend.
  for (const auto &p : tensors_) {
    delete p.second;
  }
  tensors_.clear();
  externalTensors_.clear();
}

Tensor *BoundInterpreterFunction::getTensor(const Value *v) const {
  auto it = tensors_.find(v);
  if (it != tensors_.end()) {
    return it->second;
  }
  auto ic = constants_.find(std::string(v->getName()));
  if (ic != constants_.end()) {
    return ic->second;
  }

  auto ie = externalTensors_.find(v);
  assert(ie != externalTensors_.end() && "Unknown key Value.");
  return ie->second;
}

Tensor *BoundInterpreterFunction::getOrCreateTensor(const Value *v) {
  auto ie = externalTensors_.find(v);
  if (ie != externalTensors_.end()) {
    return ie->second;
  }
  auto ic = constants_.find(std::string(v->getName()));
  if (ic != constants_.end()) {
    return ic->second;
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

Tensor *BoundInterpreterFunction::getOrCreateUnownedTensor(
    const Value *v, const Value *src, llvm::ArrayRef<size_t> offsets) {
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

void BoundInterpreterFunction::deleteTensor(const Value *v) {
  auto it = tensors_.find(v);
  if (it == tensors_.end()) {
    return;
  }

  delete it->second;
  tensors_.erase(it);
}

void BoundInterpreterFunction::execute(IRFunction *F, Context *ctx) {
  // Register the concrete tensors that back the placeholder tensors.
  for (auto &ph : ctx->pairs()) {
    auto *w = F->getWeightForNode(ph.first);
    assert(!externalTensors_.count(w) && "The tensor is already registered");
    externalTensors_[w] = ph.second;
  }

// Do the forward pass.
#define DEF_VALUE(CLASS, NAME)
#define DEF_INSTR(CLASS, NAME)                                                 \
  case Kinded::Kind::CLASS##Kind: {                                            \
    fwd##CLASS(llvm::cast<CLASS>(&I));                                         \
    break;                                                                     \
  }
#define DEF_BACKEND_SPECIFIC_INSTR(CLASS, NAME)
  // Dispatch the interpreter on each instruction in the program:
  for (const auto &I : F->getInstrs()) {
    switch (I.getKind()) {
#include "glow/AutoGenInstr.def"

    default:
      llvm_unreachable("Invalid instruction.");
    }
  }

  // Remove the concrete tensors that back the placeholder tensors.
  for (auto &ph : ctx->pairs()) {
    auto *w = F->getWeightForNode(ph.first);
    externalTensors_.erase(w);
  }
}
