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
#ifndef GLOW_OPTIMIZER_GRAPHOPTIMIZER_FUNCTIONPASS_H
#define GLOW_OPTIMIZER_GRAPHOPTIMIZER_FUNCTIONPASS_H

#include "glow/PassManager/Pass.h"
#include "glow/PassManager/PassConfig.h"

namespace glow {

class Function;
struct CompilationContext;
enum class FunctionPassID;

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

/// Class used for all passes over Functions. All passes over Functions should
/// derive from this class, implementing the pass logic and additionally can add
/// logic for running before and after the pass runs.
using FunctionPass = Pass<Function, FunctionPassConfig>;

} // namespace glow

#endif // GLOW_OPTIMIZER_GRAPHOPTIMIZER_FUNCTIONPASS_H
