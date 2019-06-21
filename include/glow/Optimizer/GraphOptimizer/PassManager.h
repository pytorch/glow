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
#ifndef GLOW_OPTIMIZER_PASSMANAGER_H
#define GLOW_OPTIMIZER_PASSMANAGER_H

#include "glow/Optimizer/GraphOptimizer/CompilationContext.h"
#include "glow/Optimizer/GraphOptimizer/FunctionPass.h"
#include "glow/Optimizer/GraphOptimizer/FunctionPasses.h"

#include "llvm/ADT/SmallVector.h"

namespace glow {

/// Specifies a configuration for running a FunctionPass when used in a
/// FunctionPassPipeline.
class FunctionPassConfig {
public:
  /// Specifies convergence mode for a FunctionPass.
  enum class ConvergenceMode {
    OnePass,         // Run a single pass over the Function.
    UntilFixedPoint, // Run the pass over the Function until a fixed point is
                     // reached. Runs DCE between each pass.
  };

private:
  /// ID of the FunctionPass to run.
  FunctionPassID passID_{FunctionPassID::EmptyPass};

  /// Convergence mode to inform the PassManager how to run the FunctionPass.
  ConvergenceMode convergenceMode_{ConvergenceMode::OnePass};

  /// Which CompilationModes the FunctionPass should be run in.
  std::bitset<convertEnumToUnsigned(CompilationMode::NumCompilationModes)>
      enabledCompModes_;

public:
  FunctionPassConfig(FunctionPassID ID,
                     ConvergenceMode convergenceMode = ConvergenceMode::OnePass,
                     const std::set<CompilationMode> &enabledCompModes =
                         {CompilationMode::Infer, CompilationMode::Train})
      : passID_(ID), convergenceMode_(convergenceMode) {
    for (const auto &mode : enabledCompModes) {
      enabledCompModes_.set(convertEnumToUnsigned(mode));
    }
  }

  /// \returns the FunctionPassID of this config.
  FunctionPassID getFunctionPassID() const { return passID_; }

  /// \returns the ConvergenceMode of this config.
  ConvergenceMode getConvergenceMode() const { return convergenceMode_; }

  /// \returns whether \p mode is an enabled mode for this config.
  bool isEnabledForCompilationMode(CompilationMode mode) const {
    return enabledCompModes_.test(convertEnumToUnsigned(mode));
  }
};

/// Implementation of a pipeline for executing a series of FunctionPasses.
class FunctionPassPipeline : private llvm::SmallVector<FunctionPassConfig, 64> {
private:
  using ParentImpl = llvm::SmallVectorImpl<FunctionPassConfig>;

public:
  FunctionPassPipeline() = default;

  /// Constructs a FunctionPassPipeline from an initializer_list \p IL.
  FunctionPassPipeline(std::initializer_list<FunctionPassConfig> IL) {
    this->assign(IL);
  }

  /// Forward iterator creation methods.
  ///@{
  iterator begin() { return ParentImpl::begin(); }
  const_iterator begin() const { return ParentImpl::begin(); }
  iterator end() { return begin() + size(); }
  const_iterator end() const { return begin() + size(); }
  /// @}

  /// Push a new \p FPC to the end of the pipeline.
  void add(FunctionPassConfig FPC) { push_back(FPC); }
};

/// Manager for running a series of FunctionPasses. Given some Function,
/// CompilationContext, and provided Pipeline, it will run all passes on the
/// Function. Enables easier debugging given runPrePass() and runPostPass()
/// calls, which can be modified to run some code before or before every pass.
class FunctionPassManager : public Named {
private:
  /// The pipeline of passes to run.
  FunctionPassPipeline pipeline_;

  /// Logic to execute before pass \p P is run on \p F, given \p cctx. \returns
  /// if \p F was modified.
  bool runPrePass(Function *F, const CompilationContext &cctx,
                  const FunctionPass &P);

  /// Logic to execute after pass \p P is run on \p F, given \p cctx. \returns
  /// if \p F was modified.
  bool runPostPass(Function *F, const CompilationContext &cctx,
                   const FunctionPass &P);

public:
  FunctionPassManager(llvm::StringRef name) : Named(name) {}
  ~FunctionPassManager() = default;

  /// Run the FunctionPassPipeline given the \ref pipeline_ and
  /// \p cctx. \returns whether \p F was modified.
  bool run(Function *F, const CompilationContext &cctx);

  /// Getter for a reference to the Pipeline used by this PassManager..
  const FunctionPassPipeline &getPipeline() { return pipeline_; };

  /// Adds a pass \p config to the end of the pipeline.
  void addPass(const FunctionPassConfig &config) { pipeline_.add(config); }
};

} // namespace glow

#endif // GLOW_OPTIMIZER_PASSMANAGER_H
