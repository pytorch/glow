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
  unsigned enabledCompModes_;
  /// ID of the pass.
  unsigned passID_;

public:
  /// Destructor.
  virtual ~PassConfigBase() = default;
  /// Constructor.
  PassConfigBase(unsigned passID,
                 ConvergenceMode convergenceMode = ConvergenceMode::OnePass,
                 const std::set<CompilationMode> &enabledCompModes =
                     {CompilationMode::Infer, CompilationMode::Train})
      : convergenceMode_(convergenceMode), enabledCompModes_(0),
        passID_(passID) {
    for (const auto &mode : enabledCompModes) {
      enabledCompModes_ |= 1 << (convertEnumToUnsigned(mode));
    }
  }

  /// Constructor.
  PassConfigBase(unsigned passID, ConvergenceMode convergenceMode,
                 unsigned enabledCompModes)
      : convergenceMode_(convergenceMode), enabledCompModes_(enabledCompModes),
        passID_(passID) {
    CHECK(
        (~((1 << convertEnumToUnsigned(CompilationMode::NumCompilationModes)) -
           1) &
         enabledCompModes) == 0)
        << "Unknown compilation modes: " << enabledCompModes;
  }

  /// \returns the ConvergenceMode of this config.
  ConvergenceMode getConvergenceMode() const { return convergenceMode_; }

  /// \returns whether \p mode is an enabled mode for this config.
  bool isEnabledForCompilationMode(CompilationMode mode) const {
    return enabledCompModes_ & (1 << (convertEnumToUnsigned(mode)));
  }

  /// \returns enabled compilation modes.
  unsigned getEnabledCompilationModes() const { return enabledCompModes_; }

  unsigned getID() const { return passID_; }

  /// Dump a textual representation of this config to \p os.
  virtual void dump(llvm::raw_ostream &os, llvm::StringRef passName) const;

  /// \returns the name of the pass for this config.
  virtual llvm::StringRef getNameOfPass() const = 0;

  /// \return true if two configs are equal.
  virtual bool equals(const PassConfigBase &other) const;
};

/// Specifies a configuration for running an Pass when used in a
/// PassPipeline. Pass ids are represented by the type \p PASS_ID.
template <typename PASS_ID> class PassConfig : public PassConfigBase {
public:
  using PassIDTy = PASS_ID;

public:
  // Constructor.
  PassConfig(PassIDTy ID,
             ConvergenceMode convergenceMode = ConvergenceMode::OnePass,
             const std::set<CompilationMode> &enabledCompModes =
                 {CompilationMode::Infer, CompilationMode::Train})
      : PassConfigBase(static_cast<unsigned>(ID), convergenceMode,
                       enabledCompModes) {}
  // Constructor.
  PassConfig(PassIDTy ID, ConvergenceMode convergenceMode,
             unsigned enabledCompModes)
      : PassConfigBase(static_cast<unsigned>(ID), convergenceMode,
                       enabledCompModes) {}
  // Destructor.
  ~PassConfig() = default;

  /// \returns the passID of this config.
  PassIDTy getPassID() const { return static_cast<PassIDTy>(passID_); }

  virtual llvm::StringRef getNameOfPass() const override {
    return "<unknown pass>";
  }

  virtual void dump(llvm::raw_ostream &os,
                    llvm::StringRef passName) const override {
    PassConfigBase::dump(os, passName);
  }

  /// Dump a textual representation of this config to \p os.
  virtual void dump(llvm::raw_ostream &os = llvm::outs()) const {
    dump(os, getNameOfPass());
  }

  /// \return true if two configs are equal.
  virtual bool equals(const PassConfigBase &other) const override {
    return (*this).PassConfigBase::equals(other);
  }
};

} // namespace glow

#endif // GLOW_PASSMANAGER_PASSCONFIG_H
