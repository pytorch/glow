/**
 * Copyright (c) Glow Contributors. See CONTRIBUTORS file.
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

#include "glow/LLVMIRCodeGen/AllocationsInfo.h"
#include "glow/Backend/BackendUtils.h"
#include "glow/Backend/CompiledFunction.h"
#include "glow/CodeGen/MemoryAllocator.h"
#include "glow/Graph/Graph.h"
#include "glow/Graph/Nodes.h"
#include "glow/Graph/PlaceholderBindings.h"
#include "glow/IR/IRUtils.h"
#include "glow/IR/Instrs.h"
#include "glow/Support/Debug.h"
#include "glow/Support/Memory.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "jit-allocations"

using namespace glow;
using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;

void AllocationsInfo::allocateWeightVars(const IRFunction *F) {
  // Compute the new offsets for all the weights, do not reuse their current
  // addresses. Process all constant WeightVars first.
  for (auto &v : F->findConstants()) {
    assert(isa<WeightVar>(F->getWeightForNode(v)) && "Expected WeightVar");
    auto *w = cast<WeightVar>(F->getWeightForNode(v));
    if (allocatedAddress_.count(v)) {
      allocatedAddress_[w] = allocatedAddress_[v];
      continue;
    }
    auto numBytes = w->getSizeInBytes();
    size_t addr = constantWeightVarsAllocator.allocate(numBytes, v);
    allocatedAddress_[v] = addr;
    allocatedAddress_[w] = addr;
  }

  // Placeholders should be allocated in a order of
  // intput|inputOutput|output|neither.
  auto contiguousPlaceholders =
      getContiguousPlaceHolder(F->findPlaceholders(), *F);

  // Compute the offsets and total memory requirements for Placeholders.
  for (auto it = contiguousPlaceholders.begin();
       it != contiguousPlaceholders.end(); it++) {
    auto &v = it->addr;
    // Get the WeightVar for each Placeholder to calculate offsets.
    assert(isa<WeightVar>(F->getWeightForNode(v)) && "Expected WeightVar");
    auto *w = cast<WeightVar>(F->getWeightForNode(v));
    if (allocatedAddress_.count(w)) {
      continue;
    }
    auto numBytes = w->getSizeInBytes();
    size_t addr = mutableWeightVarsAllocator.allocate(numBytes, w);
    allocatedAddress_[w] = addr;
  }

  // Remember that max required memory size for each kind of weights.
  constantWeightVarsMemSize_ = constantWeightVarsAllocator.getMaxMemoryUsage();
  mutableWeightVarsMemSize_ = mutableWeightVarsAllocator.getMaxMemoryUsage();

  DEBUG_GLOW(for (auto &A
                  : allocatedAddress_) {
    if (isa<AllocActivationInst>(A.first) || isa<TensorViewInst>(A.first))
      continue;
    assert(valueNumbers_.count(A.first) && "Unknown weight");
    if (isa<Constant>(A.first))
      continue;
    auto *weight = dyn_cast<WeightVar>(A.first);
    llvm::StringRef kind =
        valueNumbers_[weight].first == ValueKind::ConstantWeight
            ? "constant weight"
            : "mutable weight";
    llvm::dbgs() << "Allocated " << kind << " " << weight->getName()
                 << " size: " << weight->getSizeInBytes()
                 << "  address range:  [" << allocatedAddress_[weight] << ", "
                 << allocatedAddress_[weight] + weight->getSizeInBytes()
                 << "]\n";
  });
}

void AllocationsInfo::allocateActivations(const IRFunction *F) {
  // Maps activations and views to some offset within the heap.
  llvm::DenseMap<const Value *, uint64_t> activationAddr;

  // Assign device-space addresses to the activations.
  for (const auto &I : F->getInstrs()) {
    if (auto *A = dyn_cast<AllocActivationInst>(&I)) {
      auto numBytes = I.getSizeInBytes();
      size_t addr = activationsAllocator.allocate(numBytes, A);
      assert(!activationAddr.count(A) && "Allocation already made!");
      activationAddr[A] = addr;
      continue;
    }

    if (auto *D = dyn_cast<DeallocActivationInst>(&I)) {
      auto *A = D->getAlloc();
      assert(activationAddr.count(A) && "Invalid deallocation!");
      activationsAllocator.deallocate(A);
      continue;
    }
  }

  activationsMemSize_ = activationsAllocator.getMaxMemoryUsage();

  // Register specific addresses within the heap to activations.
  for (auto &A : activationAddr) {
    allocatedAddress_[A.first] = A.second;
  }
  DEBUG_GLOW(for (auto &A
                  : allocatedAddress_) {
    if (!isa<AllocActivationInst>(A.first)) {
      continue;
    }
    if (isa<Constant>(A.first))
      continue;
    auto *act = dyn_cast<AllocActivationInst>(A.first);
    llvm::dbgs() << "Allocated activation " << act->getName()
                 << " size: " << act->getSizeInBytes() << "  address range:  ["
                 << allocatedAddress_[act] << ", "
                 << allocatedAddress_[act] + act->getSizeInBytes() << "]\n";
  });
}

/// Calculate the offset for \p TVI into the underlying alloc activation.
static size_t calculateTensorViewOffset(const TensorViewInst *TVI) {
  // Pop tensor views off repeatedly until we reach the origin, in case there
  // are multiple stacked together, to calculate the total offset.
  const TensorViewInst *currTVI = TVI;
  size_t totalOffsetLength = 0;
  do {
    // Get the offset into the current base Tensor in bytes. Aggregate all
    // offsets from stacked TVIs into totalOffsetLength.
    totalOffsetLength +=
        getFlattenedOffset(currTVI->getSrc()->getType()->strides(),
                           currTVI->getOffsets()) *
        currTVI->getType()->getElementSize();
  } while ((currTVI = dyn_cast<TensorViewInst>(currTVI->getSrc())));

  return totalOffsetLength;
}

void AllocationsInfo::allocateTensorViews(const IRFunction *F) {
  for (const auto &I : F->getInstrs()) {
    if (const auto *TVI = dyn_cast<TensorViewInst>(&I)) {
      auto *viewOrigin = getOrigin(TVI);
      assert(allocatedAddress_.count(viewOrigin) &&
             "Did not find original WeightVar or AllocActivation for a "
             "TensorView.");
      size_t originAddr = allocatedAddress_[viewOrigin];

      // Calculate the offset into the underlying alloc activation.
      size_t offset = calculateTensorViewOffset(TVI);

      // Calculate the correct address using this offset into the alloc
      // activation and map from the original TVI to it.
      assert(!allocatedAddress_.count(TVI) && "Allocation already made!");
      allocatedAddress_[TVI] = originAddr + offset;
      continue;
    }
  }
}

void AllocationsInfo::numberValues(const IRFunction *F) {
  // Assign numbers to all weights.
  for (auto &v : F->findConstants()) {
    assert(isa<WeightVar>(F->getWeightForNode(v)));
    auto *w = cast<WeightVar>(F->getWeightForNode(v));
    if (valueNumbers_.count(v)) {
      valueNumbers_[w] = valueNumbers_[v];
      continue;
    }
    valueNumbers_[v] = std::make_pair(ValueKind::ConstantWeight, valueIdx_);
    valueNumbers_[w] = std::make_pair(ValueKind::ConstantWeight, valueIdx_++);
  }

  // Assign numbers to all placeholders.
  for (auto &v : F->findPlaceholders()) {
    assert(isa<WeightVar>(F->getWeightForNode(v)));
    auto *w = cast<WeightVar>(F->getWeightForNode(v));
    if (valueNumbers_.count(w)) {
      continue;
    }
    valueNumbers_[w] = std::make_pair(ValueKind::MutableWeight, valueIdx_++);
  }

  // Assign numbers to all activations and tensorviews.
  for (const auto &I : F->getInstrs()) {
    if (auto *A = dyn_cast<AllocActivationInst>(&I)) {
      assert(!valueNumbers_.count(A) &&
             "Activation should be defined only once");
      valueNumbers_[A] = std::make_pair(ValueKind::Activation, valueIdx_++);
      continue;
    }
    if (auto *A = dyn_cast<TensorViewInst>(&I)) {
      auto *viewOrigin = getOrigin(A);
      auto kind = ValueKind::Activation;
      if (auto *w = dyn_cast<WeightVar>(viewOrigin)) {
        kind = w->isConstant() ? ValueKind::ConstantWeight
                               : ValueKind::MutableWeight;
      }
      assert(!valueNumbers_.count(A) &&
             "TensorView should be defined only once");
      valueNumbers_[A] = std::make_pair(kind, valueIdx_++);
      continue;
    }
  }
  DEBUG_GLOW(for (auto &A
                  : valueNumbers_) {
    if (isa<Constant>(A.first))
      continue;
    auto *v = static_cast<const Value *>(A.first);
    llvm::dbgs() << "Value number for " << v->getName() << ": "
                 << A.second.second << "\n";
  });
}
