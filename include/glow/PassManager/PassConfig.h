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
#ifndef GLOW_PASSMANAGER_PASSCONFIG_H
#define GLOW_PASSMANAGER_PASSCONFIG_H

#include "glow/Optimizer/GraphOptimizer/CompilationContext.h"
#include "glow/Support/Support.h"

#include <bitset>

namespace glow {

/// Specifies convergence mode for a pass.
enum class ConvergenceMode {
  /// Run a single pass over the Function.
  OnePass,
  /// Run the pass over the Function until a fixed point is reached.
  UntilFixedPoint,
};

/// The base class for all pass config classes.
class PassConfigBase {
protected:
  /// Convergence mode to inform the PassManager how to run the FunctionPass.
  ConvergenceMode convergenceMode_{ConvergenceMode::OnePass};
  /// Which CompilationModes the Pass should be run in.
  std::bitset<convertEnumToUnsigned(CompilationMode::NumCompilationModes)>
      enabledCompModes_;

public:
  PassConfigBase(ConvergenceMode convergenceMode = ConvergenceMode::OnePass,
                 const std::set<CompilationMode> &enabledCompModes =
                     {CompilationMode::Infer, CompilationMode::Train})
      : convergenceMode_(convergenceMode) {
    for (const auto &mode : enabledCompModes) {
      enabledCompModes_.set(convertEnumToUnsigned(mode));
    }
  }
  /// \returns the ConvergenceMode of this config.
  ConvergenceMode getConvergenceMode() const { return convergenceMode_; }

  /// \returns whether \p mode is an enabled mode for this config.
  bool isEnabledForCompilationMode(CompilationMode mode) const {
    return enabledCompModes_.test(convertEnumToUnsigned(mode));
  }
  /// Dump a textual representation of this config to \p os.
  void dump(llvm::raw_ostream &os, llvm::StringRef passName) const;
};

/// Specifies a configuration for running an Pass when used in a
/// PassPipeline. Pass ids are represented by the type \p PASS_ID.
template <typename PASS_ID> class PassConfig : public PassConfigBase {
public:
  using PassIDTy = PASS_ID;

private:
  /// ID of the FunctionPass to run.
  PassIDTy passID_{PassIDTy::EmptyPass};

public:
  PassConfig(PassIDTy ID,
             ConvergenceMode convergenceMode = ConvergenceMode::OnePass,
             const std::set<CompilationMode> &enabledCompModes =
                 {CompilationMode::Infer, CompilationMode::Train})
      : PassConfigBase(convergenceMode, enabledCompModes), passID_(ID) {}

  /// \returns the passID of this config.
  PassIDTy getPassID() const { return passID_; }

  /// Dump a textual representation of this config to \p os.
  void dump(llvm::raw_ostream &os = llvm::outs()) const;
};

} // namespace glow

#endif // GLOW_PASSMANAGER_PASSCONFIG_H
