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
#ifndef GLOW_LLVMIRCODEGEN_ALLOCATIONSINFO_H
#define GLOW_LLVMIRCODEGEN_ALLOCATIONSINFO_H

#include "glow/Backend/BackendUtils.h"
#include "glow/CodeGen/MemoryAllocator.h"
#include "glow/Graph/Nodes.h"
#include "llvm/IR/Module.h"

#include <functional>

namespace glow {
class Value;
class IRFunction;
class WeightVar;
class Constant;
class PlaceholderBindings;

/// Information about allocations for activations, constant weight variables
/// and mutable weight variables.
class AllocationsInfo {
public:
  using ValueKind = runtime::MemoryRegions;
  using MemRegionAndNumber = std::pair<runtime::MemoryRegionId, size_t>;
  /// Map Values in the module to their numbers.
  llvm::DenseMap<const Kinded *, MemRegionAndNumber> valueNumbers_;
  /// To get the offset of a given value simply use
  /// numberOffsets_[valueNumbers_[v]]

  /// Maps Values in the module to their offsets.
  llvm::DenseMap<const Kinded *, uint64_t> allocatedAddress_;
  /// Amount of memory to be allocated for constant WeightVars.
  size_t constantWeightVarsMemSize_{0};
  /// Amount of memory to be allocated for mutable WeightVars.
  size_t mutableWeightVarsMemSize_{0};
  /// Amount of memory to be allocated for activations.
  size_t activationsMemSize_{0};
  /// Base address of stored constant weights.
  uint8_t *baseConstantWeightVarsStore_{nullptr};

  /// Ctor.
  AllocationsInfo();
  /// Dtor.
  virtual ~AllocationsInfo() = default;
  /// Perform allocation for the function \p F.
  virtual void allocate(const IRFunction *F);
  /// Assign offsets to all of the variables in the module \p M and to the
  /// placeholders.
  virtual void allocateWeightVars(const IRFunction *F);
  /// Assign offsets to all activations.
  /// No actual memory allocation is performed. All the allocations should be
  /// performed by the client based on the information provided by the
  /// AllocationsInfo or RuntimeBundle.
  virtual void allocateActivations(const IRFunction *F);
  /// Assign offsets to all tensorviews.
  /// No memory allocation is performed. Sets up all offsets into already
  /// defined offsets for WeightVars and AllocActivations. Assumes the weight
  /// vars and alloc activations have already been added to allocatedAddress_.
  virtual void allocateTensorViews(const IRFunction *F);
  /// Number all allocations and weight variables by assigning them unique
  /// numbers.
  virtual void numberValues(const IRFunction *F);
  /// Getters for allocators.
  MemoryAllocator &getConstantWeightVarsAllocator() {
    return constantWeightVarsAllocator_;
  }
  MemoryAllocator &getMutableWeightVarsAllocator() {
    return mutableWeightVarsAllocator_;
  }
  MemoryAllocator &getActivationsAllocator() { return activationsAllocator_; }

  const glow::runtime::SymbolTableTy &getSymbolTable() const {
    return symbolTable_;
  }
  const glow::runtime::RuntimeBundle &getRuntimeBundle() const {
    return runtimeBundle_;
  }
  glow::runtime::SymbolTableTy &getSymbolTable() { return symbolTable_; }
  glow::runtime::RuntimeBundle &getRuntimeBundle() { return runtimeBundle_; }
  std::shared_ptr<runtime::MemoryRegionDescriptions>
  getMemoryRegionDescriptions() const {
    return memRegionDescriptions_;
  }
  void
  setMemoryRegionDescriptions(std::shared_ptr<runtime::MemoryRegionDescriptions>
                                  memRegionDescriptions) {
    memRegionDescriptions_ = memRegionDescriptions;
  }

protected:
  /// Initialize internal tables to prepare for processing of the function \p F.
  virtual void initTables(const IRFunction *F);

  /// Index to be used for a new value.
  size_t valueIdx_{0};
  /// Use two different allocators, because constant weights and mutable weights
  /// may use different memory blocks.
  MemoryAllocator constantWeightVarsAllocator_;
  MemoryAllocator mutableWeightVarsAllocator_;
  /// Use a memory allocator with no upper bound on how much memory we can
  /// allocate.
  MemoryAllocator activationsAllocator_;
  /// Runtime bundle for the allocated symbols.
  glow::runtime::RuntimeBundle runtimeBundle_;
  /// Symbol table for the allocated symbols.
  glow::runtime::SymbolTableTy &symbolTable_;
  /// Set of processed functions.
  std::set<const glow::IRFunction *> processedFuncs_;
  /// Descriptions of memory regions if available.
  std::shared_ptr<runtime::MemoryRegionDescriptions> memRegionDescriptions_;
};

} // namespace glow
#endif // GLOW_LLVMIRCODEGEN_ALLOCATIONSINFO_H
