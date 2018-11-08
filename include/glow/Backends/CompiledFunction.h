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
#ifndef GLOW_BACKENDS_COMPILEDFUNCTION_H
#define GLOW_BACKENDS_COMPILEDFUNCTION_H

#include "glow/Graph/Nodes.h"
#include <unordered_map>

namespace glow {

class Context;
namespace runtime {
/// RuntimeSymbolInfo
/// Contains information for initialization and handling of symbol at runtime.
struct RuntimeSymbolInfo {
  /// The size in bytes.
  size_t size;
  /// Offset in bytes from the base address.
  size_t offset;
  /// Type of symbol.
  Type type;
};
/// Runtime Bundle
/// Contains the information needed to be passed forward from compile time to
/// runtime. In order to allocate and initialize memory.
struct RuntimeBundle {
  /// Map from symbol name to a RuntimeSymbolInfo.
  std::unordered_map<std::string, RuntimeSymbolInfo> symbolTable;
  /// Pointer to memory containing the weights for execution.
  uint8_t *constants;
  /// Amount of memory needed for weights.
  const size_t constantWeightVarsMemSize;
  /// Amount of memory needed for mutable vars.
  const size_t mutableWeightVarsMemSize;
  /// Amount of memory needed for activations.
  const size_t activationsMemSize;
  RuntimeBundle(size_t constWeight, size_t mutableWeight, size_t activations)
      : constantWeightVarsMemSize(constWeight),
        mutableWeightVarsMemSize(mutableWeight),
        activationsMemSize(activations) {}
};
} // end namespace runtime
/// Interface for executing a compiled function.
class CompiledFunction {
public:
  /// Dtor.
  virtual ~CompiledFunction() = default;
  /// Execute the network and allocate Placeholder memory with given
  /// \p ctx providing mapping between Placeholder and populated tensor.
  virtual void execute(Context &ctx) = 0;
};
} // end namespace glow

#endif // GLOW_BACKENDS_COMPILEDFUNCTION_H
