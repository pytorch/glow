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
#ifndef GLOW_BACKENDS_COMPILEOPTIONS_H
#define GLOW_BACKENDS_COMPILEOPTIONS_H

namespace glow {

enum class CompilationMode {
  Train, /// Compile the graph in preperation for training.
  Infer, /// Compile the graph for inference. Notice that this operation
         /// changes the graph in a way that is not reversible.
};

/// Configuration options for the compile() method on Backend.
struct CompilationOptions {
  CompilationMode mode{CompilationMode::Infer};

  /// Allocate and collect constant Tensors in the RuntimeBundle.
  bool collectConstants{true};

  /// Insert TraceEvents between all instructions for profiling.
  bool autoInstrument{false};
};

}; // namespace glow

#endif // GLOW_BACKENDS_COMPILEOPTIONS_H
