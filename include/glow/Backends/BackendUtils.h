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
#ifndef GLOW_BACKENDS_BACKENDUTILS_H
#define GLOW_BACKENDS_BACKENDUTILS_H

#include "glow/CodeGen/MemoryAllocator.h"
#include "glow/IR/IR.h"

namespace glow {
namespace runtime {
/// Contains information for initialization and handling of symbol at runtime.
struct RuntimeSymbolInfo {
  /// The size in bytes.
  size_t size;
  /// Offset in bytes from the base address.
  size_t offset;
  /// Type of symbol.
  Type type;
};

/// Contains the information needed to be passed forward from compile time to
/// runtime. In order to allocate and initialize memory.
class RuntimeBundle {
  /// Map from symbol name to a RuntimeSymbolInfo.
  std::unordered_map<std::string, RuntimeSymbolInfo> symbolTable_;
  /// Pointer to memory containing the weights for execution.
  uint8_t *constants_;
  /// Amount of memory needed for weights.
  const size_t constantWeightVarsMemSize_;
  /// Amount of memory needed for mutable vars.
  const size_t mutableWeightVarsMemSize_;
  /// Amount of memory needed for activations.
  const size_t activationsMemSize_;

public:
  /// Get Constant Weights memory size.
  size_t getConstantWeightSize() const { return constantWeightVarsMemSize_; }
  /// Get Mutable Weights memory size.
  size_t getMutableWeightSize() const { return mutableWeightVarsMemSize_; }
  /// Get Activations Weights memory size.
  size_t getActivationsSize() const { return activationsMemSize_; }
  /// Get pointer to memory block of constants.
  uint8_t *getConstants() const { return constants_; }
  /// Helper function, gets offset of \p v.
  size_t getValueOffset(const Named *v) const;
  /// Helper function, gets symbol info for \p v.
  RuntimeSymbolInfo getSymbolInfo(const Named *v) const;
  /// At compile time condense constants to a single block of memory.
  /// This allows the graph to go away after compile time.
  /// Allocates a block of memory of size \p constantMaxSize then walks the
  /// given function \p F and and copies weights to their address as specified
  /// by offsets contained in symbolTable_.
  void collectConstants(const IRFunction *F);
  RuntimeBundle(std::unordered_map<std::string, RuntimeSymbolInfo> &symbolTable,
                size_t constWeight, size_t mutableWeight, size_t activations)
      : symbolTable_(std::move(symbolTable)),
        constantWeightVarsMemSize_(constWeight),
        mutableWeightVarsMemSize_(mutableWeight),
        activationsMemSize_(activations) {}
};
} // namespace runtime

/// Computes offsets and total allocation for Constants, Placeholders, and
/// Activations to build runtime symbol table. Returns RuntimeBundle.
runtime::RuntimeBundle
generateRuntimeBundle(const IRFunction &F, MemoryAllocator &constantAllocator,
                      MemoryAllocator &placeholderAllocator,
                      MemoryAllocator &activationsAllocator,
                      bool collectConstants = true);

} // end namespace glow
#endif // GLOW_BACKENDS_BACKENDUTILS_H
