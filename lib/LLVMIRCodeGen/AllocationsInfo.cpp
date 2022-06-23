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

AllocationsInfo::AllocationsInfo()
    : constantWeightVarsAllocator_("ConstantWeights", 0),
      mutableWeightVarsAllocator_("MutableWeights", 0),
      activationsAllocator_("Activations", 0), runtimeBundle_(0, 0, 0),
      symbolTable_(runtimeBundle_.getSymbolTable()) {}

void AllocationsInfo::initTables(const IRFunction *F) {
  // Bail if F was processed already.
  if (processedFuncs_.count(F)) {
    return;
  }
  processedFuncs_.insert(F);
  if (!memRegionDescriptions_) {
    // Use default memory region descriptions if nothig else was specified.
    memRegionDescriptions_ = getDefaultMemoryRegionDescriptions();
  }
  createMemoryRegionTable(*F, *getMemoryRegionDescriptions(),
                          runtimeBundle_.getMemoryRegionTable(),
                          runtimeBundle_.getSymbolTable());
  // Reserve memory at the beginning of the activations region if needed.
  // TODO: Generalize for all regions?
  if (activationsAllocator_.getMaxMemoryUsage() > 0) {
    auto &defaultActivationsRegion =
        runtimeBundle_.getMemoryRegion(runtime::MemoryRegions::Activation);
    if (!defaultActivationsRegion.hasAttribute("alloc.reserve_at_front")) {
      defaultActivationsRegion.addAttribute(
          "alloc.reserve_at_front",
          std::to_string(activationsAllocator_.getMaxMemoryUsage()));
    }
  }
  allocateMemory(*F, runtimeBundle_.getMemoryRegionTable(),
                 runtimeBundle_.getSymbolTable());
}

void AllocationsInfo::allocate(const IRFunction *F) {
  initTables(F);
  numberValues(F);
  allocateWeightVars(F);
  allocateActivations(F);
  allocateTensorViews(F);
}

void AllocationsInfo::allocateWeightVars(const IRFunction *F) {
  initTables(F);
  // Compute the new offsets for all the weights, do not reuse their current
  // addresses. Process all constant WeightVars first.
  allocateConstants(F->findConstants(), constantWeightVarsAllocator_,
                    runtimeBundle_);
  for (auto &c : F->findConstants()) {
    auto name = std::string(c->getName());
    auto symbolIt = symbolTable_.find(name);
    CHECK(symbolIt != symbolTable_.end())
        << "Expected to find " << name << " in symbol table";
    auto *w = cast<WeightVar>(F->getWeightForNode(c));
    // Bail if a placeholder was assigned an address already.
    if (allocatedAddress_.count(w)) {
      continue;
    }
    if (allocatedAddress_.count(c)) {
      allocatedAddress_[w] = allocatedAddress_[c];
      continue;
    }
    auto &symbol = *symbolIt->second;
    CHECK(valueNumbers_.count(c))
        << "Unexpected uncounted constant: " << c->getName().str();
    symbol.index = valueNumbers_[c].second;
    auto addr = symbol.offset;
    // Update the address of a weight.
    allocatedAddress_[c] = addr;
    allocatedAddress_[w] = addr;
    // Update the memory region number.
    valueNumbers_[c].first = symbol.getMemRegionId();
    CHECK(valueNumbers_.count(w))
        << "Cannot find valueNumber of a weight: " << w->getName().str();
    valueNumbers_[w].first = symbol.getMemRegionId();
  }

  // Placeholders should be allocated in a order of
  // intput|inputOutput|output|neither.
  auto contiguousPlaceholders =
      getContiguousPlaceHolder(F->findPlaceholders(), *F);

  allocatePlaceholders(contiguousPlaceholders, mutableWeightVarsAllocator_,
                       runtimeBundle_);
  // Compute the offsets and total memory requirements for Placeholders.
  for (auto it = contiguousPlaceholders.begin();
       it != contiguousPlaceholders.end(); it++) {
    auto &v = it->addr;
    auto name = std::string(v->getName());
    auto symbolIt = symbolTable_.find(name);
    CHECK(symbolIt != symbolTable_.end())
        << "Expected to find " << name << " in symbol table";
    auto *w = cast<WeightVar>(F->getWeightForNode(v));
    // Bail if a placeholder was assigned an address already.
    if (allocatedAddress_.count(w)) {
      continue;
    }
    auto &symbol = *symbolIt->second;
    CHECK(valueNumbers_.count(w))
        << "Cannot find valueNumber of a weight: " << w->getName().str();
    symbol.index = valueNumbers_[w].second;
    auto addr = symbol.offset;
    // Update the address of a weight.
    allocatedAddress_[w] = addr;
    // Update the memory region number.
    valueNumbers_[w].first = symbol.getMemRegionId();
  }

  // Remember that max required memory size for each kind of weights.
  constantWeightVarsMemSize_ = constantWeightVarsAllocator_.getMaxMemoryUsage();
  mutableWeightVarsMemSize_ = mutableWeightVarsAllocator_.getMaxMemoryUsage();

  if (!constantWeightVarsMemSize_) {
    constantWeightVarsMemSize_ = runtimeBundle_.getMemoryRegionSize(
        runtime::MemoryRegions::ConstantWeight);
  }
  if (!mutableWeightVarsMemSize_) {
    mutableWeightVarsMemSize_ = runtimeBundle_.getMemoryRegionSize(
        runtime::MemoryRegions::MutableWeight);
  }

  DEBUG_GLOW(for (auto &A
                  : allocatedAddress_) {
    if (isa<AllocActivationInst>(A.first) || isa<TensorViewInst>(A.first))
      continue;
    (CHECK(valueNumbers_.count(A.first)) << "Unknown weight");
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
  initTables(F);
  glow::allocateActivations(F->getInstrs(), activationsAllocator_,
                            runtimeBundle_);

  // Maps activations and views to some offset within the heap.
  llvm::DenseMap<const Value *, uint64_t> activationAddr;

  // Assign device-space addresses to the activations.
  for (const auto &I : F->getInstrs()) {
    if (auto *A = dyn_cast<AllocActivationInst>(&I)) {
      auto name = std::string(A->getName());
      auto symbolIt = symbolTable_.find(name);
      CHECK(symbolIt != symbolTable_.end())
          << "Expected to find " << name << " in symbol table";
      auto &symbol = *symbolIt->second;
      CHECK(valueNumbers_.count(A)) << "Unexpected uncounted activation";
      symbol.index = valueNumbers_[A].second;
      CHECK(!activationAddr.count(A)) << "Allocation already made!";
      auto addr = symbol.offset;
      activationAddr[A] = addr;
      // Update the memory region number.
      CHECK(valueNumbers_.count(A))
          << "Cannot find valueNumber of an activation: " << A->getName().str();
      valueNumbers_[A].first = symbol.getMemRegionId();
      continue;
    }
  }

  activationsMemSize_ = activationsAllocator_.getMaxMemoryUsage();
  if (!activationsMemSize_) {
    activationsMemSize_ =
        runtimeBundle_.getMemoryRegionSize(runtime::MemoryRegions::Activation);
  }

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

void AllocationsInfo::allocateTensorViews(const IRFunction *F) {
  initTables(F);
  for (const auto &I : F->getInstrs()) {
    if (const auto *TVI = dyn_cast<TensorViewInst>(&I)) {
      auto *viewOrigin = getOrigin(TVI);
      CHECK(allocatedAddress_.count(viewOrigin))
          << "Did not find original WeightVar or AllocActivation for a "
          << "TensorView.";
      size_t originAddr = allocatedAddress_[viewOrigin];

      // Calculate the offset into the underlying alloc activation.
      size_t offset = calculateTensorViewOffset(TVI);

      // Calculate the correct address using this offset into the alloc
      // activation and map from the original TVI to it.
      CHECK(!allocatedAddress_.count(TVI)) << "Allocation already made!";
      allocatedAddress_[TVI] = originAddr + offset;

      auto name = std::string(TVI->getName());
      CHECK(symbolTable_.count(name)) << "Unexpected tensorview symbol";
      auto &symbol = *symbolTable_[name];
      CHECK(valueNumbers_.count(TVI)) << "Unexpected uncounted tensorview";
      symbol.index = valueNumbers_[TVI].second;
      // Update the memory region number.
      CHECK(valueNumbers_.count(TVI))
          << "Cannot find valueNumber of a tensor_view: "
          << TVI->getName().str();
      valueNumbers_[TVI].first = valueNumbers_[viewOrigin].first;
      continue;
    }
  }
}

/// Assign number to all weights, placeholders, activations and tensor views.
void AllocationsInfo::numberValues(const IRFunction *F) {
  initTables(F);

  // Assign numbers to all weights.
  for (auto &v : F->findConstants()) {
    CHECK(isa<WeightVar>(F->getWeightForNode(v)))
        << "Expected a weight variable";
    auto *w = cast<WeightVar>(F->getWeightForNode(v));
    // Don't number constants which are numbered already.
    if (valueNumbers_.count(v)) {
      if (valueNumbers_.count(w)) {
        continue;
      }
      valueNumbers_[w] = valueNumbers_[v];
      continue;
    }
    CHECK(!valueNumbers_.count(w))
        << "WeightVar for a Constant should not have an assigned value number";
    valueNumbers_[v] = std::make_pair(ValueKind::ConstantWeight, valueIdx_);
    valueNumbers_[w] = std::make_pair(ValueKind::ConstantWeight, valueIdx_);
    valueIdx_++;
  }

  // Assign numbers to all placeholders.
  for (auto &v : F->findPlaceholders()) {
    CHECK(isa<WeightVar>(F->getWeightForNode(v)))
        << "Expected a weight variable";
    auto *w = cast<WeightVar>(F->getWeightForNode(v));
    // Don't number placeholder which are numbered already.
    if (valueNumbers_.count(w)) {
      continue;
    }
    valueNumbers_[w] = std::make_pair(ValueKind::MutableWeight, valueIdx_++);
  }

  // Assign numbers to all activations and tensorviews.
  for (const auto &I : F->getInstrs()) {
    if (auto *A = dyn_cast<AllocActivationInst>(&I)) {
      CHECK(!valueNumbers_.count(A))
          << "Activation should be numbered only once";
      valueNumbers_[A] = std::make_pair(ValueKind::Activation, valueIdx_++);
      continue;
    }
    if (auto *A = dyn_cast<TensorViewInst>(&I)) {
      CHECK(!valueNumbers_.count(A))
          << "TensorView should be numbered only once";
      auto *viewOrigin = getOrigin(A);
      auto kind = ValueKind::Activation;
      if (auto *w = dyn_cast<WeightVar>(viewOrigin)) {
        kind = w->isConstant() ? ValueKind::ConstantWeight
                               : ValueKind::MutableWeight;
      }
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
