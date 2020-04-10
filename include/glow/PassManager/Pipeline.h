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
#ifndef GLOW_OPTIMIZER_IROPTIMIZERPIPELINE_PIPELINE_H
#define GLOW_OPTIMIZER_IROPTIMIZERPIPELINE_PIPELINE_H

#include "glow/Optimizer/GraphOptimizer/CompilationContext.h"
#include "glow/Support/Support.h"

#include <bitset>
#include <iterator>

namespace glow {

/// Specifies convergence mode for a FunctionPass.
enum class ConvergenceMode {
  /// Run a single pass over the Function.
  OnePass,
  /// Run the pass over the Function until a fixed point is reached.
  UntilFixedPoint,
};

/// Specifies a configuration for running an Pass when used in a
/// PassPipeline.
template <typename PASS_ID> class PassConfig {
public:
  using PassID = PASS_ID;

private:
  /// ID of the FunctionPass to run.
  PassID passID_{PassID::EmptyPass};

  /// Convergence mode to inform the PassManager how to run the FunctionPass.
  ConvergenceMode convergenceMode_{ConvergenceMode::OnePass};

  /// Which CompilationModes the Pass should be run in.
  std::bitset<convertEnumToUnsigned(CompilationMode::NumCompilationModes)>
      enabledCompModes_;

public:
  PassConfig(PassID ID,
             ConvergenceMode convergenceMode = ConvergenceMode::OnePass,
             const std::set<CompilationMode> &enabledCompModes =
                 {CompilationMode::Infer, CompilationMode::Train})
      : passID_(ID), convergenceMode_(convergenceMode) {
    for (const auto &mode : enabledCompModes) {
      enabledCompModes_.set(convertEnumToUnsigned(mode));
    }
  }

  /// \returns the passID of this config.
  PassID getPassID() const { return passID_; }

  /// \returns the ConvergenceMode of this config.
  ConvergenceMode getConvergenceMode() const { return convergenceMode_; }

  /// \returns whether \p mode is an enabled mode for this config.
  bool isEnabledForCompilationMode(CompilationMode mode) const {
    return enabledCompModes_.test(convertEnumToUnsigned(mode));
  }

  /// Dump a textual representation of this config to \p os.
  void dump(llvm::raw_ostream &os = llvm::outs()) const;
};

/// Implementation of a pipeline for executing a series of FunctionPasses.
template <typename PASS_CONFIG>
class PassPipeline : private llvm::SmallVector<PASS_CONFIG, 64> {
public:
  using PassConfig = PASS_CONFIG;
  using PassID = typename PassConfig::PassID;

private:
  using ParentImpl = llvm::SmallVectorImpl<PassConfig>;

  /// Removes the first instance of a pass with ID \p PID. \returns whether an
  /// instance of the pass was successfully found and removed.
  bool removeFirstInstanceOfPass(PassID PID);

public:
  using Base = llvm::SmallVector<PASS_CONFIG, 64>;
  using iterator = typename Base::iterator;
  using const_iterator = typename Base::const_iterator;
  PassPipeline() = default;

  /// Constructs a FunctionPassPipeline from an initializer_list \p IL.
  PassPipeline(std::initializer_list<PassConfig> IL) { this->assign(IL); }

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
  const PassConfig at(size_t i) const { return begin()[i]; }

  /// Push a new \p FPC to the end of the pipeline.
  void pushBack(PassConfig FPC) { ParentImpl::push_back(FPC); }

  /// Push a new \p FPC to the start of the pipeline.
  void pushFront(PassConfig FPC) { ParentImpl::insert(begin(), FPC); }

  /// Removes all instances of a pass with ID \p PID.
  void removeAllInstancesOfPass(PassID PID);

  /// Dump a textual representation of the pipeline to \p os.
  void dump(llvm::raw_ostream &os = llvm::outs()) const;
};

} // namespace glow

#endif // GLOW_OPTIMIZER_IROPTIMIZERPIPELINE_PIPELINE_H
