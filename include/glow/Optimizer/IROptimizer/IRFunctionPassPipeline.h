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
#ifndef GLOW_OPTIMIZER_IROPTIMIZER_IRFUNCTIONPASSPIPELINE_H
#define GLOW_OPTIMIZER_IROPTIMIZER_IRFUNCTIONPASSPIPELINE_H

#include "glow/Optimizer/IROptimizer/IRFunctionPass.h"
#include "glow/PassManager/Pipeline.h"

namespace glow {

/// Define an enum to identify all FunctionPasses, used to declare inside a
/// FunctionPassConfig for a Pipeline.
enum class IRFunctionPassID {
#define IR_FUN_PASS(PASS_NAME) PASS_NAME,
#include "glow/Optimizer/IROptimizer/IRPasses.def"
};

/// IR passes pipeline.
using IRFunctionPassPipeline = PassPipeline<IRFunctionPass>;

/// \returns the default, target-independent IR optimization pipeline.
IRFunctionPassPipeline createDefaultIRFunctionOptimizationPipeline();

/// \returns the name of a Pass given its \p passID.
llvm::StringRef getNameOfPass(IRFunctionPassID passID);

} // namespace glow

#endif // GLOW_OPTIMIZER_IROPTIMIZER_IRFUNCTIONPASSPIPELINE_H
