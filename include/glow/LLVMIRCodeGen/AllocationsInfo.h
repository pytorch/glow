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
#ifndef GLOW_LLVMIRCODEGEN_ALLOCATIONSINFO_H
#define GLOW_LLVMIRCODEGEN_ALLOCATIONSINFO_H

#include "glow/Graph/Nodes.h"
#include "llvm/IR/Module.h"

#include <functional>

namespace glow {
class Value;
class IRFunction;
class WeightVar;
class Constant;
class PlaceholderBindings;

namespace runtime {
class RuntimeBundle;
}

/// Information about allocations for activations, constant weight variables
/// and mutable weight variables.
struct AllocationsInfo {
  /// Different kinds of values that need to be allocated.
  enum class ValueKind { ConstantWeight, MutableWeight, Activation };
  using KindAndNumber = std::pair<ValueKind, size_t>;
  /// Map Values in the module to their numbers.
  llvm::DenseMap<const Value *, KindAndNumber> valueNumbers_;
  /// To get the offset of a given value simply use
  /// numberOffsets_[valueNumbers_[v]]

  /// Maps Values in the module to their offsets.
  llvm::DenseMap<const Value *, uint64_t> allocatedAddress_;
  /// Amount of memory to be allocated for constant WeightVars.
  size_t constantWeightVarsMemSize_{0};
  /// Amount of memory to be allocated for mutable WeightVars.
  size_t mutableWeightVarsMemSize_{0};
  /// Amount of memory to be allocated for activations.
  size_t activationsMemSize_{0};
  /// Base address of stored constant weights.
  uint8_t *baseConstantWeightVarsStore_{nullptr};
  /// Base address of constant weights.

  /// Assign offsets to all of the variables in the module \p M and to the
  /// placeholders.
  void allocateWeightVars(const IRFunction *F);
  /// Assign offsets to all activations.
  /// No actual memory allocation is performed. All the allocations should be
  /// performed by the client based on the information provided by the
  /// AllocationsInfo or RuntimeBundle.
  void allocateActivations(const IRFunction *F);
  /// Assign offsets to all tensorviews.
  /// No memory allocation is performed. Sets up all offsets into already
  /// defined offsets for WeightVars and AllocActivations. Assumes the weight
  /// vars and alloc activations have already been added to allocatedAddress_.
  void allocateTensorViews(const IRFunction *F);
  /// Number all allocations and weight variables by assigning them unique
  /// numbers.
  void numberValues(const IRFunction *F);
};

} // namespace glow
#endif // GLOW_LLVMIRCODEGEN_ALLOCATIONSINFO_H
