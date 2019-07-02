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
#ifndef GLOW_OPTIMIZER_GRAPHOPTIMIZER_GRAPHOPTIMIZER_PIPELINES_PIPELINES_H
#define GLOW_OPTIMIZER_GRAPHOPTIMIZER_GRAPHOPTIMIZER_PIPELINES_PIPELINES_H

#include "glow/Optimizer/GraphOptimizer/CompilationContext.h"
#include "glow/Support/Support.h"

#include <bitset>

namespace glow {

/// Define an enum to identify all FunctionPasses, used to declare inside a
/// FunctionPassConfig for a Pipeline.
enum class FunctionPassID {
#define FUN_PASS(PASS_NAME) PASS_NAME,
#include "glow/Optimizer/GraphOptimizer/FunctionPasses.def"
};

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
  void pushBack(FunctionPassConfig FPC) { push_back(FPC); }
};

/// \returns the default, target-independent graph optimization pipeline
FunctionPassPipeline createDefaultGraphOptimizationPasses();

/// \returns the default fold pipeline.
FunctionPassPipeline createDefaultFoldPasses();

} // namespace glow

#endif // GLOW_OPTIMIZER_GRAPHOPTIMIZER_GRAPHOPTIMIZER_PIPELINES_PIPELINES_H
