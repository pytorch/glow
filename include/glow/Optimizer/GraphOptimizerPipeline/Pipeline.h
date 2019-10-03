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

/// Specifies convergence mode for a FunctionPass.
enum class ConvergenceMode {
  /// Run a single pass over the Function.
  OnePass,
  /// Run the pass over the Function until a fixed point is reached.
  UntilFixedPoint,
};

/// Specifies whether the pass requires DCE.
enum class DCERequiredMode {
  /// Require that DCE is run before the pass.
  BeforePass,
  /// Signify the pass has no requirement/dependence on DCE.
  None,
};

/// Specifies a configuration for running a FunctionPass when used in a
/// FunctionPassPipeline.
class FunctionPassConfig {
private:
  /// ID of the FunctionPass to run.
  FunctionPassID passID_{FunctionPassID::EmptyPass};

  /// Convergence mode to inform the PassManager how to run the FunctionPass.
  ConvergenceMode convergenceMode_{ConvergenceMode::OnePass};

  /// Which CompilationModes the FunctionPass should be run in.
  std::bitset<convertEnumToUnsigned(CompilationMode::NumCompilationModes)>
      enabledCompModes_;

  /// Represents whether DCE is required for this pass.
  DCERequiredMode dceMode_{DCERequiredMode::BeforePass};

public:
  FunctionPassConfig(FunctionPassID ID,
                     ConvergenceMode convergenceMode = ConvergenceMode::OnePass,
                     const std::set<CompilationMode> &enabledCompModes =
                         {CompilationMode::Infer, CompilationMode::Train},
                     DCERequiredMode dceMode = DCERequiredMode::BeforePass)
      : passID_(ID), convergenceMode_(convergenceMode), dceMode_(dceMode) {
    for (const auto &mode : enabledCompModes) {
      enabledCompModes_.set(convertEnumToUnsigned(mode));
    }
  }

  /// \returns the FunctionPassID of this config.
  FunctionPassID getFunctionPassID() const { return passID_; }

  /// \returns the ConvergenceMode of this config.
  ConvergenceMode getConvergenceMode() const { return convergenceMode_; }

  /// \returns the DCERequiredMode of this config.
  DCERequiredMode getDCERequiredMode() const { return dceMode_; }

  /// \returns whether \p mode is an enabled mode for this config.
  bool isEnabledForCompilationMode(CompilationMode mode) const {
    return enabledCompModes_.test(convertEnumToUnsigned(mode));
  }

  /// Dump a textual representation of this config to \p os.
  void dump(llvm::raw_ostream &os = llvm::outs()) const;
};

/// Implementation of a pipeline for executing a series of FunctionPasses.
class FunctionPassPipeline : private llvm::SmallVector<FunctionPassConfig, 64> {
private:
  using ParentImpl = llvm::SmallVectorImpl<FunctionPassConfig>;

  /// Removes the first instance of a pass with ID \p FPID. \returns whether an
  /// instance of the pass was successfully found and removed.
  bool removeFirstInstanceOfPass(FunctionPassID FPID);

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

  /// Forward to parent size() method. \returns size of pipeline.
  size_t size() const { return ParentImpl::size(); }

  /// Helper to get the FunctionPassConfig at index \p i in the pipeline.
  const FunctionPassConfig at(size_t i) const { return begin()[i]; }

  /// Push a new \p FPC to the end of the pipeline.
  void pushBack(FunctionPassConfig FPC) { push_back(FPC); }

  /// Push a new \p FPC to the start of the pipeline.
  void pushFront(FunctionPassConfig FPC) { insert(begin(), FPC); }

  /// Removes all instances of a pass with ID \p FPID.
  void removeAllInstancesOfPass(FunctionPassID FPID);

  /// Dump a textual representation of the pipeline to \p os.
  void dump(llvm::raw_ostream &os = llvm::outs()) const;
};

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

#endif // GLOW_OPTIMIZER_GRAPHOPTIMIZER_GRAPHOPTIMIZER_PIPELINES_PIPELINES_H
