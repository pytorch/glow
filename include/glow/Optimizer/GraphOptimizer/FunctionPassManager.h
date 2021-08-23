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
#ifndef GLOW_OPTIMIZER_GRAPHOPTIMIZER_FUNCTIONPASSMANAGER_H
#define GLOW_OPTIMIZER_GRAPHOPTIMIZER_FUNCTIONPASSMANAGER_H

#include "glow/Optimizer/GraphOptimizer/FunctionPass.h"
#include "glow/Optimizer/GraphOptimizer/FunctionPassPipeline.h"
#include "glow/PassManager/PassManager.h"

namespace glow {

/// Pass manager for the graph-level IR passes.
using FunctionPassManager = PassManager<FunctionPassPipeline, FunctionPass>;

/// Helper to run a DCE pass on \p F given \p cctx. \returns if \p was modified.
bool runDCEPass(Function *F, const CompilationContext &cctx);

} // namespace glow

#endif // GLOW_OPTIMIZER_GRAPHOPTIMIZER_FUNCTIONPASSMANAGER_H
