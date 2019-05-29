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
  // Use two different allocators, because constant weights and mutable weights
  // may use different memory blocks.
  MemoryAllocator constantWeightVarsAllocator("ConstantWeights", 0);
  MemoryAllocator mutableWeightVarsAllocator("MutableWeights", 0);

  // Compute the new offsets for all the weights, do not reuse their current
  // addresses. Process all constant WeightVars first.
  for (auto &v : F->findConstants()) {
    assert(isa<WeightVar>(F->getWeightForNode(v)) && "Expected WeightVar");
    auto *w = cast<WeightVar>(F->getWeightForNode(v));
    auto numBytes = w->getSizeInBytes();
    size_t addr = constantWeightVarsAllocator.allocate(numBytes, w);
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
    llvm::StringRef kind =
        valueNumbers_[A.first].first == ValueKind::ConstantWeight
            ? "constant weight"
            : "mutable weight";
    llvm::dbgs() << "Allocated " << kind << " " << A.first->getName()
                 << " size: " << A.first->getSizeInBytes()
                 << "  address range:  [" << allocatedAddress_[A.first] << ", "
                 << allocatedAddress_[A.first] + A.first->getSizeInBytes()
                 << "]\n";
  });
}

void AllocationsInfo::allocateActivations(const IRFunction *F) {
  // Use a memory allocator with no upper bound on how much memory we can
  // allocate.
  MemoryAllocator activationsAllocator("Activations", 0);

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
    llvm::dbgs() << "Allocated activation " << A.first->getName()
                 << " size: " << A.first->getSizeInBytes()
                 << "  address range:  [" << allocatedAddress_[A.first] << ", "
                 << allocatedAddress_[A.first] + A.first->getSizeInBytes()
                 << "]\n";
  });
}

/// Calculate the offset for \p TVI into the underlying alloc activation.
static size_t calculateTensorViewOffset(const TensorViewInst *TVI) {
  // Pop tensor views off repeatedly until we reach the origin, in case there
  // are multiple stacked together, to calculate the total offset.
  const TensorViewInst *currTVI = TVI;
  size_t totalOffsetLength = 0;
  do {
    // Calculate and store the length of the current tensorview's offset
    // into the source of the tensorview. Note that this source may be
    // another tensorview.
    size_t currOffsetLength =
        currTVI->getOffsets().empty() ? 0 : currTVI->getOffsets()[0];
    auto *tvSource = currTVI->getSrc();
    for (size_t i = 1; i < tvSource->dims().size(); ++i) {
      currOffsetLength *= tvSource->dims()[i];
    }

    // Increment the running total offset length which will be used to store
    // into allocatedAddressed.
    totalOffsetLength +=
        currOffsetLength * currTVI->getType()->getElementSize();
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
  size_t valueIdx = 0;
  // Assign numbers to all weights.
  for (auto &v : F->findConstants()) {
    assert(isa<WeightVar>(F->getWeightForNode(v)));
    auto *w = cast<WeightVar>(F->getWeightForNode(v));
    valueNumbers_[w] = std::make_pair(ValueKind::ConstantWeight, valueIdx++);
  }

  // Assign numbers to all placeholders.
  for (auto &v : F->findPlaceholders()) {
    assert(isa<WeightVar>(F->getWeightForNode(v)));
    auto *w = cast<WeightVar>(F->getWeightForNode(v));
    valueNumbers_[w] = std::make_pair(ValueKind::MutableWeight, valueIdx++);
  }

  // Assign numbers to all activations and tensorviews.
  for (const auto &I : F->getInstrs()) {
    if (auto *A = dyn_cast<AllocActivationInst>(&I)) {
      valueNumbers_[A] = std::make_pair(ValueKind::Activation, valueIdx++);
      continue;
    }
    if (auto *A = dyn_cast<TensorViewInst>(&I)) {
      auto *viewOrigin = getOrigin(A);
      auto kind = ValueKind::Activation;
      if (auto *w = dyn_cast<WeightVar>(viewOrigin)) {
        kind = w->isConstant() ? ValueKind::ConstantWeight
                               : ValueKind::MutableWeight;
      }
      valueNumbers_[A] = std::make_pair(kind, valueIdx++);
      continue;
    }
  }
}
