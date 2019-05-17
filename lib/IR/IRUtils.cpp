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

size_t glow::getOriginOffset(Value *V) {
  size_t off = 0;

  // Since each TensorView can either maintain or decrease the size of a buffer,
  // the linearized offset of a TensorView into its underlying buffer can be
  // computed by adding together the linearized offsets of intermediate
  // TensorViews relative to their sources.
  while (true) {
    auto *TVI = dyn_cast<TensorViewInst>(V);
    if (!TVI) {
      return off;
    }

    llvm::ArrayRef<size_t> offsets = TVI->getOffsets();
    llvm::ArrayRef<size_t> srcDims = TVI->getSrc()->getType()->dims();

    size_t numSrcDims = srcDims.size();

    // Iterate backwards in order to figure out how big the slices corresponding
    // to each offset element are.
    for (size_t i = 0, j = numSrcDims - 1; i < numSrcDims; ++i, --j) {
      // For each offset, add into the linearized offset the offset value
      // multiplied by the slice size corresponding to that offset.
      off += offsets[j] * TVI->getSrc()->getType()->getSliceSize(j);
    }

    // Move on to the src of the TensorView that was just processed.
    V = TVI->getSrc();
  }

  return off;
}
