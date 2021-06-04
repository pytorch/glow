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
#ifndef GLOW_OPTIMIZER_IROPTIMIZER_IROPTIMIZER_H
#define GLOW_OPTIMIZER_IROPTIMIZER_IROPTIMIZER_H

#include <memory>

namespace glow {

class IRFunction;
class IRContainer;
class Backend;

/// Perform target-independent optimizations on the IR representation of \p M.
/// Apply the buffer sharing optimizations if \p shouldShareBuffers is true.
void optimize(IRFunction &M, bool shouldShareBuffers);

/// Perform backend-specific optimizations on the IR representation of \p M.
/// Apply the buffer sharing optimizations if \p shouldShareBuffers is true.
void optimize(IRFunction &M, const Backend &B, bool shouldShareBuffers);

/// Helper to generate and optimize IR from given Function \p F. \p
/// shouldShareBuffers signifies whether to use the share buffers optimization.
/// Backend /p B is used to allow for custom lowering from Node to
/// Instruction IR.
std::unique_ptr<IRFunction> generateAndOptimizeIR(IRContainer *F,
                                                  const Backend &B,
                                                  bool shouldShareBuffers);
} // namespace glow

#endif // GLOW_OPTIMIZER_IROPTIMIZER_IROPTIMIZER_H
