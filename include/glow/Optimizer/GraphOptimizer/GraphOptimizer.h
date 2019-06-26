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
#ifndef GLOW_OPTIMIZER_GRAPHOPTIMIZER_GRAPHOPTIMIZER_H
#define GLOW_OPTIMIZER_GRAPHOPTIMIZER_GRAPHOPTIMIZER_H

#include "glow/Optimizer/GraphOptimizer/CompilationContext.h"
#include "glow/Support/Error.h"

#include "llvm/ADT/ArrayRef.h"

namespace glow {

class Function;
class Backend;
class Module;
class PlaceholderBindings;
class Placeholder;

/// Perform optimizations on the graph representation.
void optimize(Function *F, CompilationContext &cctx);
void optimize(Function *F, CompilationMode mode);
/// Fold nodes that were expressed lowered in the input model.
void fold(Function *F, CompilationContext &cctx);
void fold(Function *F, CompilationMode mode);

/// Lower the high-level neural network nodes found in \p F into low-level
/// linear algebra operators. If \p B is not a nullptr then it can prevent
/// lowering of a node via \ref Backend::shouldLower(); otherwise everything
/// will be lowered. \p cctx will contain a mapping of loweredMap from output
/// names of the nodes found and lowered in \p F to the output names of the
/// nodes they were lowered from along with the NodeKind. \p doNotLowerKinds is
/// a set of NodeKinds which represents all Nodes that should not be lowered.
void lower(Function *F, CompilationContext &cctx, const Backend *B = nullptr,
           const KindSet &doNotLowerKinds = {});

/// Convert placeholders in Module \p M to constants based on the values in \p
/// bindings.  Do not convert any placeholders explicitly listed in \p vars.
void convertPlaceholdersToConstants(Function *F,
                                    const PlaceholderBindings &bindings,
                                    llvm::ArrayRef<Placeholder *> vars);

/// Instrument function \p F by inserting quantization profile nodes for
/// capturing stats for quantization. The nodes will refer to tensors allocate
/// in context \p bindings.
void profileQuantization(PlaceholderBindings &bindings, Function *F);

/// Optimize the Function \p F given compilation options \p cctx for Backend \B.
/// \returns success if all nodes in the final resulting optimized Function are
/// supported by \p B; if not, this represents a compiler error.
llvm::Error optimizeFunction(Function *F, const Backend &B,
                             CompilationContext &cctx);

/// Optimize the Function \p F given compilation options \p cctx performing
/// backend-independent optimizations that can be done before lowering.
/// \returns success if there were no compiler errors; if not, this represents a
/// compiler error.
llvm::Error optimizeFunctionBeforeLowering(Function *F,
                                           CompilationContext &cctx);
} // namespace glow

#endif // GLOW_OPTIMIZER_GRAPHOPTIMIZER_GRAPHOPTIMIZER_H
