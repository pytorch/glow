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

#include "glow/IR/IRUtils.h"
#include "glow/IR/Instrs.h"

#include "llvm/Support/Casting.h"

using namespace glow;
using llvm::dyn_cast;
using llvm::isa;

bool glow::isTensorView(glow::Value *v) { return isa<TensorViewInst>(v); }

Value *glow::getAllocationOrigin(Value *V) {
  while (true) {
    if (auto *AI = dyn_cast<AllocActivationInst>(V))
      return AI;
    if (auto *TVI = dyn_cast<TensorViewInst>(V)) {
      V = TVI->getSrc();
      continue;
    }
    return nullptr;
  }
  return nullptr;
}

Value *glow::getOrigin(Value *V) {
  return const_cast<Value *>(getOrigin(const_cast<const Value *>(V)));
}

const Value *glow::getOrigin(const Value *V) {
  while (true) {
    auto *TVI = dyn_cast<TensorViewInst>(V);
    if (!TVI)
      return V;
    V = TVI->getSrc();
  }
  return V;
}
