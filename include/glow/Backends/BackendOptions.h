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
#include "llvm/ADT/StringMap.h"

#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace glow {
class Function;
class Node;
class Storage;

/// Hints provided to the Backend, the backend is not required to honor them.
struct BackendHints {
  /// Number of execution units to reserve, these are the processing elements
  /// like cores, 0 for unspecified.
  unsigned executionUnits{0};

  /// Storage nodes to be pinned to SRAM listed in order of priority.
  std::vector<std::string> SRAMPrioritization;
};

/// A flexible map used for storing options for a backend. Keys are usually
/// prefixed with a Backend's name, e.g. "Interpreter_OptionA".
using BackendSpecificOptions = std::map<std::string, std::string>;

/// A structure used for storing backend-specific information for Nodes in a
/// Function. The outer map with Functions as a key map to another map with
/// Nodes as a key, where all Nodes in that map are children of the original
/// Function. The StringMap for each Node maps from an option name to a vector
/// of values for that option.
using BackendSpecificNodeInfo = std::unordered_map<
    const Function *,
    std::unordered_map<const Node *,
                       llvm::StringMap<std::vector<std::string>>>>;

/// Options relevant to Backends during compilation.
struct BackendOptions {
  /// Allocate and collect constant Tensors in the RuntimeBundle.
  bool collectConstants{true};

  /// Insert TraceEvents between all instructions for profiling.
  bool autoInstrument{false};

  /// Use a serialized precompiled function instead of compiling.
  bool useDeserialize{false};

  /// Hints for the compiler for this compilation.
  BackendHints backendHints;

  /// Options that are specific to a backend. Backend is responsible for
  /// parsing.
  BackendSpecificOptions backendSpecificOpts;

  /// Options that are specified per-Node. Note that this structure is keyed off
  /// of Functions and then Nodes. The Node keys for this structure are Node
  /// pointers, so any changes of Nodes should be tracked and propagated into
  /// new Nodes once this is set.
  BackendSpecificNodeInfo backendSpecificNodeInfo;
};

}; // namespace glow

#endif // GLOW_BACKENDS_BACKENDOPTIONS_H
