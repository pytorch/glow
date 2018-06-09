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

#include "Interpreter.h"

#include "glow/Graph/Graph.h"
#include "glow/Graph/Nodes.h"
#include "glow/IR/IRUtils.h"
#include "glow/IR/Instrs.h"

#include "llvm/Support/Casting.h"

using namespace glow;
using llvm::isa;

Interpreter::~Interpreter() { clear(); }

void Interpreter::clear() {
  // Delete the tensors that are owned by this backend.
  for (auto p : tensors_) {
    delete p.second;
  }

  tensors_.clear();
  externalTensors_.clear();
}

void Interpreter::init() {
  for (auto &v : F_->getGraph()->getParent()->getVars()) {
    auto *w = F_->getWeightForNode(v);
    assert(!externalTensors_.count(w) && "The tensor is already registered");
    externalTensors_[w] = &v->getPayload();
  }

  for (auto *W : F_->getWeights()) {
    getOrCreateTensor(W);
  }
}

Tensor *Interpreter::getTensor(const Value *v) const {
  auto it = tensors_.find(v);
  if (it != tensors_.end()) {
    return it->second;
  }

  auto ie = externalTensors_.find(v);
  assert(ie != externalTensors_.end() && "Unknown key Value.");
  return ie->second;
}

Tensor *Interpreter::getOrCreateTensor(const Value *v) {
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

Tensor *Interpreter::getOrCreateUnownedTensor(const Value *v, const Value *src,
                                              llvm::ArrayRef<size_t> offsets) {
  assert(isa<TensorViewInst>(v) && "Expected a tensor view");

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

void Interpreter::deleteTensor(const Value *v) {
  auto it = tensors_.find(v);
  if (it == tensors_.end()) {
    return;
  }

  delete it->second;
  tensors_.erase(it);
}

bool Interpreter::isOpSupported(Kinded::Kind opKind, ElemKind elementTy) const {
  // Check quantization support.
  if (elementTy == ElemKind::Int8QTy) {
    switch (opKind) {
    case Kinded::Kind::AddNodeKind:
    case Kinded::Kind::BatchedAddNodeKind:
    case Kinded::Kind::BatchedReduceAddNodeKind:
    case Kinded::Kind::CmpLTENodeKind:
    case Kinded::Kind::ConcatNodeKind:
    case Kinded::Kind::ConvolutionNodeKind:
    case Kinded::Kind::DequantizeNodeKind:
    case Kinded::Kind::DivNodeKind:
    case Kinded::Kind::FullyConnectedNodeKind:
    case Kinded::Kind::GatherNodeKind:
    case Kinded::Kind::MatMulNodeKind:
    case Kinded::Kind::MaxNodeKind:
    case Kinded::Kind::MinNodeKind:
    case Kinded::Kind::MulNodeKind:
    case Kinded::Kind::PoolAvgNodeKind:
    case Kinded::Kind::PoolMaxNodeKind:
    case Kinded::Kind::QuantizeNodeKind:
    case Kinded::Kind::ReluNodeKind:
    case Kinded::Kind::RescaleQuantizedNodeKind:
    case Kinded::Kind::ReshapeNodeKind:
    case Kinded::Kind::SelectNodeKind:
    case Kinded::Kind::SigmoidNodeKind:
    case Kinded::Kind::SliceNodeKind:
    case Kinded::Kind::SubNodeKind:
    case Kinded::Kind::TanhNodeKind:
    case Kinded::Kind::TopKNodeKind:
    case Kinded::Kind::TransposeNodeKind:
      return true;
    default:
      return false;
    }
  }

  return true;
}

bool Interpreter::shouldLower(const Node *N) const {
  if (N->getKind() == Kinded::Kind::ConvolutionNodeKind)
    return false;
  return true;
}

void Interpreter::doForwardPass() {
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
#include "AutoGenInstr.def"

    default:
      llvm_unreachable("Invalid instruction.");
    }
  }
}
