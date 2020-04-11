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
#ifndef GLOW_OPTIMIZER_GRAPHOPTIMIZER_FUNCTIONPASSPIPELINE_H
#define GLOW_OPTIMIZER_GRAPHOPTIMIZER_FUNCTIONPASSPIPELINE_H

#include "glow/Optimizer/GraphOptimizer/CompilationContext.h"
#include "glow/PassManager/Pipeline.h"
#include "glow/Support/Support.h"

#include <bitset>

namespace glow {

/// Define an enum to identify all FunctionPasses, used to declare inside a
/// FunctionPassConfig for a Pipeline.
enum class FunctionPassID {
#define FUN_PASS(PASS_NAME) PASS_NAME,
#include "glow/Optimizer/GraphOptimizer/FunctionPasses.def"
};

/// Specifies whether the pass requires DCE.
enum class DCERequiredMode {
  /// Require that DCE is run before the pass.
  BeforePass,
  /// Signify the pass has no requirement/dependence on DCE.
  None,
};

class FunctionPassConfig : public PassConfig<FunctionPassID> {
  /// Represents whether DCE is required for this pass.
  DCERequiredMode dceMode_{DCERequiredMode::BeforePass};

public:
  FunctionPassConfig(FunctionPassID ID,
                     ConvergenceMode convergenceMode = ConvergenceMode::OnePass,
                     const std::set<CompilationMode> &enabledCompModes =
                         {CompilationMode::Infer, CompilationMode::Train},
                     DCERequiredMode dceMode = DCERequiredMode::BeforePass)
      : PassConfig(ID, convergenceMode), dceMode_(dceMode) {}
  /// \returns the DCERequiredMode of this config.
  DCERequiredMode getDCERequiredMode() const { return dceMode_; }
  void dump(llvm::raw_ostream &os) const;
};

/// IR passes pipeline.
using FunctionPassPipeline = PassPipeline<FunctionPassConfig>;

/// \returns the name of a Pass given its \p passID.
llvm::StringRef getNameOfPass(FunctionPassID passID);

/// \returns the default, target-independent graph optimization pipeline
FunctionPassPipeline createDefaultGraphOptimizationPassPipeline();

/// \returns the fp16 specific optimization pipeline
FunctionPassPipeline createFP16GraphOptimizationPassPipeline();

/// \returns the default fold pipeline.
FunctionPassPipeline createDefaultFoldPassPipeline();

/// \returns a FunctionPassConfig for performing DCE.
FunctionPassConfig getDCEPassConfig();

/// \returns the name of a FunctionPass given its \p passID.
llvm::StringRef getNameOfPass(FunctionPassID passID);

} // namespace glow

#endif // GLOW_OPTIMIZER_GRAPHOPTIMIZER_FUNCTIONPASSPIPELINE_H
