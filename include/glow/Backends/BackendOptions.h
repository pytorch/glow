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
#ifndef GLOW_BACKENDS_BACKENDOPTIONS_H
#define GLOW_BACKENDS_BACKENDOPTIONS_H

#include "llvm/ADT/SmallVector.h"
#include <map>
#include <string>
#include <vector>

namespace glow {
class Storage;

/// Hints provided to the Backend, the backend is not required to honor them.
struct BackendHints {
  /// Number of execution units to reserve, these are the processing elements
  /// like cores, 0 for unspecified.
  unsigned executionUnits{0};

  /// Storage nodes to be pinned to SRAM listed in order of priority.
  std::vector<std::string> SRAMPrioritization;
};

/// Options relevant to Backends during compilation.
struct BackendOptions {
  /// Allocate and collect constant Tensors in the RuntimeBundle.
  bool collectConstants{true};

  /// Insert TraceEvents between all instructions for profiling.
  bool autoInstrument{false};

  /// Hints for the compiler for this compilation.
  BackendHints backendHints;

  /// Options that are specific to a backend. Backend is responsible for
  /// parsing.
  std::map<std::string, std::string> backendSpecificOpts;
};

}; // namespace glow

#endif // GLOW_BACKENDS_BACKENDOPTIONS_H
