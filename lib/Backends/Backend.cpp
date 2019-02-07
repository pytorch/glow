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

#include "glow/Backends/Backend.h"
#include "glow/Graph/Graph.h"
#include "glow/Optimizer/Optimizer.h"

using namespace glow;

void Backend::optimizeFunction(CompilationMode mode, Function *F) {
  // Verify the function pre-optimization/lowering.
  assert(F->verify() && "Function must be valid");

  // Optimize the graph.
  ::glow::optimize(F, mode);

  // Lower the graph into a sequence of low-level linear algebra operations.
  ::glow::lower(F, *this);

  // Optimize the graph again.
  ::glow::optimize(F, mode);

  // Allow the backend to transform the graph after lowering.
  if (transformPostLowering(F, mode)) {
    // Optimize the graph again after the backend transformation.
    // In particular, DCE is very likely to be useful.
    ::glow::optimize(F, mode);
  }
}
