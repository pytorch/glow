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
#ifndef GLOW_OPTIMIZER_GRAPHOPTIMIZER_PASSMANAGER_H
#define GLOW_OPTIMIZER_GRAPHOPTIMIZER_PASSMANAGER_H

#include "glow/Backend/Backend.h"

#include "glow/Optimizer/GraphOptimizer/CompilationContext.h"
#include "glow/Optimizer/GraphOptimizer/FunctionPass.h"
#include "glow/Optimizer/GraphOptimizer/FunctionPasses.h"
#include "glow/Optimizer/GraphOptimizerPipeline/Pipeline.h"

#include "llvm/ADT/SmallVector.h"

namespace glow {

/// Manager for running a series of FunctionPasses. Given some Function,
/// CompilationContext, and provided Pipeline, it will run all passes on the
/// Function. Enables easier debugging given runPrePass() and runPostPass()
/// calls, which can be modified to run some code before or before every pass.
class FunctionPassManager : public Named {
private:
  /// The pipeline of passes to run.
  FunctionPassPipeline pipeline_;

  /// The Backend we have for backend-specific verification.
  const Backend *backend_;

  /// The index of the current pass being executed in the pipeline.
  size_t passIdx_ = 0;

  /// Creates and \returns a FunctionPass given a provided \p passID.
  std::unique_ptr<FunctionPass> createFunctionPass(FunctionPassID passID);

  /// Logic to execute before pass \p P is run on \p F, given \p cctx. \returns
  /// if \p F was modified.
  bool runPrePass(Function *F, const CompilationContext &cctx,
                  const FunctionPass &P);

  /// Logic to execute after pass \p P is run on \p F, given \p cctx. \returns
  /// if \p F was modified.
  bool runPostPass(Function *F, const CompilationContext &cctx,
                   const FunctionPass &P);

  /// Runs a FunctionPass described by \p passConfig over \p F given \p cctx.
  bool runPass(const FunctionPassConfig &passConfig, Function *F,
               const CompilationContext &cctx);

public:
  FunctionPassManager(llvm::StringRef name, FunctionPassPipeline pipeline,
                      const Backend *backend = nullptr)
      : Named(name), pipeline_(pipeline), backend_(backend), passIdx_(0) {}
  ~FunctionPassManager() = default;

  /// Run the FunctionPassPipeline given the \ref pipeline_ and
  /// \p cctx. \returns whether \p F was modified.
  bool run(Function *F, const CompilationContext &cctx);

  /// Getter for a reference to the Pipeline used by this PassManager.
  const FunctionPassPipeline &getPipeline() const { return pipeline_; };

  /// Dump a textual representation of the FunctionPassManager to \p os.
  void dump(llvm::raw_ostream &os = llvm::outs()) const;
};

/// Helper to run a DCE pass on \p F given \p cctx. \returns if \p was modified.
bool runDCEPass(Function *F, CompilationContext &cctx);

} // namespace glow

#endif // GLOW_OPTIMIZER_GRAPHOPTIMIZER_PASSMANAGER_H
