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
  size_t size{0};
  /// Offset in bytes from the base address.
  size_t offset{0};
  /// Type of symbol.
  Type type;
  /// Is the symbol an input for the function.
  bool input{true};
  /// Is the symbol an output for the function.
  bool output{true};
};

using SymbolTableTy = std::unordered_map<std::string, RuntimeSymbolInfo>;

/// Contains the information needed to be passed forward from compile time to
/// runtime. In order to allocate and initialize memory.
class RuntimeBundle {
  /// Map from symbol name to a RuntimeSymbolInfo.
  SymbolTableTy symbolTable_;
  /// Pointer to memory containing the weights for execution.
  uint8_t *constants_{nullptr};
  /// Amount of memory needed for weights.
  size_t constantWeightVarsMemSize_{0};
  /// Amount of memory needed for mutable vars.
  size_t mutableWeightVarsMemSize_{0};
  /// Amount of memory needed for activations.
  size_t activationsMemSize_{0};

public:
  /// Get Constant Weights memory size.
  size_t getConstantWeightSize() const { return constantWeightVarsMemSize_; }
  /// Get Mutable Weights memory size.
  size_t getMutableWeightSize() const { return mutableWeightVarsMemSize_; }
  /// Get Activations Weights memory size.
  size_t getActivationsSize() const { return activationsMemSize_; }
  /// Get pointer to memory block of constants.
  uint8_t *getConstants() const { return constants_; }
  /// Set pointer to memory block of constants.
  void setConstants(uint8_t *constants) { constants_ = constants; }
  /// Helper function, gets offset of \p v.
  size_t getValueOffset(const Named *v) const;
  /// Helper function, gets symbol info for \p v.
  const RuntimeSymbolInfo &getSymbolInfo(const Named *v) const;
  /// Get a const reference to the symbol table.
  const SymbolTableTy &getSymbolTable() const { return symbolTable_; }
  /// At compile time condense constants to a single block of memory.
  /// This allows the graph to go away after compile time.
  /// Allocates a block of memory of size \p constantMaxSize then walks the
  /// given function \p F and and copies weights to their address as specified
  /// by offsets contained in symbolTable_.
  void collectConstants(const IRFunction *F);
  void collectConstants(const Module *M);
  /// Free constants.
  void freeConstants();

  /// Sets the input and output flags for each symbol in the symbolBundle.
  void setInputsandOutputs();

  /// Computes offsets and total allocation for Constants, Placeholders, and
  /// Activations to build runtime symbol table. Returns RuntimeBundle.
  static runtime::RuntimeBundle create(const IRFunction &F,
                                       MemoryAllocator &constantAllocator,
                                       MemoryAllocator &placeholderAllocator,
                                       MemoryAllocator &activationsAllocator);

  /// Build a runtime symbol table from a Function.  Computes Constant and
  /// Placeholder sizes, but not Activations, since Functions are unserialized.
  /// Only use this method to generate bundles for backends that do not use
  /// Glow's IR.
  static runtime::RuntimeBundle create(const Function &F);

  /// Deleted default constructor.  A properly constructed RuntimeBundle is
  /// necessary for correct execution using the HostManager.
  RuntimeBundle() = delete;

  // Constructor.
  RuntimeBundle(SymbolTableTy &symbolTable, size_t constWeight,
                size_t mutableWeight, size_t activations)
      : symbolTable_(std::move(symbolTable)), constants_(nullptr),
        constantWeightVarsMemSize_(constWeight),
        mutableWeightVarsMemSize_(mutableWeight),
        activationsMemSize_(activations) {}
};
} // namespace runtime

} // end namespace glow
#endif // GLOW_BACKENDS_BACKENDUTILS_H
