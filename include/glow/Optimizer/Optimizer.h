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
#ifndef GLOW_OPTIMIZER_OPTIMIZER_H
#define GLOW_OPTIMIZER_OPTIMIZER_H

namespace glow {

class IRFunction;
class Function;
class Backend;

enum class CompilationMode {
  Train, /// Compile the graph in preperation for training.
  Infer, /// Compile the graph for inference. Notice that this operation
         /// changes the graph in a way that is not reversible.
};

/// Perform optimizations on the IR representation.
void optimize(IRFunction &M, CompilationMode mode);
/// Perform optimizations on the graph representation.
void optimize(Function *F, CompilationMode mode);

/// Lower the high-level neural network operators into low-level lineal algebra
/// operators.
void lower(Function *F, CompilationMode mode, Backend *B = nullptr);

/// Instrument function \p F by inserting quantization profile nodes
/// for capturing stats for quantization.
void profileQuantization(Function *F);

} // namespace glow

#endif // GLOW_OPTIMIZER_OPTIMIZER_H
