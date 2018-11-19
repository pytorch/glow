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
/// Interface for executing a compiled function.
class CompiledFunction {
public:
  /// Dtor.
  virtual ~CompiledFunction() = default;
  /// Execute the network and allocate Placeholder memory with given
  /// \p ctx providing mapping between Placeholder and populated tensor.
  virtual void execute() = 0;

  /// Does any needed initialization work for the Backend.
  /// This includes device init constant memory allocation and copying to
  /// device.
  virtual void setupRuns() = 0;
  /// Per run setup. Copy inputs to device.
  virtual void beforeRun(const Context &ctx) = 0;
  /// Per run cleanup. Copy outputs from device.
  virtual void afterRun(const Context &ctx) = 0;
  /// Final cleanup. Release memory, reset device.
  virtual void tearDownRuns() = 0;
};
} // end namespace glow

#endif // GLOW_BACKENDS_COMPILEDFUNCTION_H
