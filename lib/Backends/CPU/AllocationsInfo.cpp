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
#define DEBUG_TYPE "jit-allocations"

#include "AllocationsInfo.h"
#include "glow/CodeGen/MemoryAllocator.h"
#include "glow/Graph/Graph.h"
#include "glow/Graph/Nodes.h"
#include "glow/IR/IRUtils.h"
#include "glow/IR/Instrs.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace glow;

using namespace glow;
using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;
using llvm::StringRef;

void AllocationsInfo::allocateWeightVars(IRFunction *F, bool reuseAddresses) {
  // Use two different allocators, because constant weights and mutable weights
  // may use different memory blocks.
  MemoryAllocator constantWeightVarsAllocator(0);
  MemoryAllocator mutableWeightVarsAllocator(0);

  // Compute the new offsets for all the weights, do not reuse their current
  // addresses. Process all constant WeightVars first.
  for (auto &v : F->getGraph()->getParent()->getVars()) {
    assert(isa<WeightVar>(F->getWeightForNode(v)));
    auto *w = cast<WeightVar>(F->getWeightForNode(v));
    if (v->getVisibilityKind() == VisibilityKind::Public)
      continue;
    auto numBytes = w->getSizeInBytes();
    size_t addr = constantWeightVarsAllocator.allocate(numBytes);
    if (!reuseAddresses) {
      allocatedAddressed_[w] = addr;
    } else {
      // Reuse the address used by the payload.
      allocatedAddressed_[w] =
          v->getPayload().getUnsafePtr() - static_cast<char *>(nullptr);
    }
  }

  // Process all mutable WeightVars afterwards.
  for (auto &v : F->getGraph()->getParent()->getVars()) {
    assert(isa<WeightVar>(F->getWeightForNode(v)));
    auto *w = cast<WeightVar>(F->getWeightForNode(v));
    if (v->getVisibilityKind() != VisibilityKind::Public)
      continue;
    auto numBytes = w->getSizeInBytes();
    size_t addr = mutableWeightVarsAllocator.allocate(numBytes);
    if (!reuseAddresses) {
      allocatedAddressed_[w] = addr;
    } else {
      // Reuse the address used by the payload.
      allocatedAddressed_[w] =
          v->getPayload().getUnsafePtr() - static_cast<char *>(nullptr);
    }
  }

  // Remember that max required memory size for each kind of weights.
  constantWeightVarsMemSize_ = constantWeightVarsAllocator.getMaxMemoryUsage();
  mutableWeightVarsMemSize_ = mutableWeightVarsAllocator.getMaxMemoryUsage();

  DEBUG(for (auto &A
             : allocatedAddressed_) {
    if (isa<AllocActivationInst>(A.first) || isa<TensorViewInst>(A.first))
      continue;
    assert(valueNumbers_.count(A.first) && "Unknown weight");
    llvm::StringRef kind =
        valueNumbers_[A.first].first == ValueKind::ConstantWeight
            ? "constant weight"
            : "mutable weight";
    llvm::errs() << "Allocated " << kind << " " << A.first->getName()
                 << " size: " << A.first->getSizeInBytes()
                 << "  address range:  [" << allocatedAddressed_[A.first]
                 << ", "
                 << allocatedAddressed_[A.first] + A.first->getSizeInBytes()
                 << "]\n";
  });
}

void AllocationsInfo::allocateActivations(IRFunction *F) {
  // Use a memory allocator with no upper bound on how much memory we can
  // allocate.
  MemoryAllocator activationsAllocator(0);

  // Maps activations and views to some offset within the heap.
  llvm::DenseMap<const Value *, size_t> activationAddr;

  // Assign device-space addresses to the activations.
  for (const auto &I : F->getInstrs()) {
    if (auto *A = dyn_cast<AllocActivationInst>(&I)) {
      auto numBytes = I.getSizeInBytes();
      size_t addr = activationsAllocator.allocate(numBytes);
      assert(!activationAddr.count(A) && "Allocation already made!");
      activationAddr[A] = addr;
      continue;
    }

    if (auto *D = dyn_cast<DeallocActivationInst>(&I)) {
      auto *A = D->getAlloc();
      assert(activationAddr.count(A) && "Invalid deallocation!");
      activationsAllocator.deallocate(activationAddr[A]);
      continue;
    }
  }

  activationsMemSize_ = activationsAllocator.getMaxMemoryUsage();

  // Register specific addresses within the heap to activations.
  for (auto &A : activationAddr) {
    allocatedAddressed_[A.first] = A.second;
  }
  DEBUG(for (auto &A
             : allocatedAddressed_) {
    llvm::errs() << "Allocated activation " << A.first->getName()
                 << " size: " << A.first->getSizeInBytes()
                 << "  address range:  [" << allocatedAddressed_[A.first]
                 << ", "
                 << allocatedAddressed_[A.first] + A.first->getSizeInBytes()
                 << "]\n";
  });
}

void AllocationsInfo::allocateTensorViews(IRFunction *F) {
  for (const auto &I : F->getInstrs()) {
    if (const auto *A = dyn_cast<TensorViewInst>(&I)) {
      auto *viewOrigin = getOrigin(A);
      assert(allocatedAddressed_.count(viewOrigin) &&
             "Did not find original WeightVar or AllocActivation for a "
             "TensorView.");
      size_t originAddr = allocatedAddressed_[viewOrigin];

      // Calculate and store the length of the offset into the base, using the
      // source of the tensorview.
      size_t offsetLength = A->getOffsets()[0];
      auto *tvSource = A->getSrc();
      if (tvSource->dims().size() > 1) {
        for (size_t i = 1; i < tvSource->dims().size(); ++i) {
          offsetLength *= tvSource->dims()[i];
        }
      }
      assert(!allocatedAddressed_.count(A) && "Allocation already made!");
      allocatedAddressed_[A] =
          originAddr + (offsetLength * A->getType()->getElementSize());
      continue;
    }
  }
}

void AllocationsInfo::numberValues(IRFunction *F) {
  size_t valueIdx = 0;
  // Assign numbers to all weights.
  for (auto &v : F->getGraph()->getParent()->getVars()) {
    assert(isa<WeightVar>(F->getWeightForNode(v)));
    auto *w = cast<WeightVar>(F->getWeightForNode(v));
    auto kind = v->getVisibilityKind() != VisibilityKind::Public
                    ? ValueKind::ConstantWeight
                    : ValueKind::MutableWeight;
    valueNumbers_[w] = std::make_pair(kind, valueIdx++);
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
        kind = w->getVisibility() != VisibilityKind::Public
                   ? ValueKind::ConstantWeight
                   : ValueKind::MutableWeight;
      }
      valueNumbers_[A] = std::make_pair(kind, valueIdx++);
      continue;
    }
  }
}

void AllocationsInfo::clear() {
  allocatedAddressed_.clear();
  valueNumbers_.clear();
  baseActivationsAddress_ = nullptr;
  baseConstantWeightVarsAddress_ = nullptr;
  baseMutableWeightVarsAddress_ = nullptr;
  activationsMemSize_ = 0;
  constantWeightVarsMemSize_ = 0;
  mutableWeightVarsMemSize_ = 0;
}
