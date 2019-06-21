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
#ifndef GLOW_OPTIMIZER_GRAPHOPTIMIZER_FUNCTIONPASSES_H
#define GLOW_OPTIMIZER_GRAPHOPTIMIZER_FUNCTIONPASSES_H

#include "glow/Optimizer/GraphOptimizer/FunctionPass.h"

#include <bitset>

namespace glow {

/// Define an enum to identify all FunctionPasses.
enum class FunctionPassID {
#define FUN_PASS(PASS_NAME) PASS_NAME,
#include "FunctionPasses.def"
};

/// Declare all FunctionPass classes.
#define FUN_PASS(PASS_NAME)                                                    \
  class PASS_NAME : public FunctionPass {                                      \
  public:                                                                      \
    bool run(Function *F) override;                                            \
    llvm::StringRef getName() const override { return #PASS_NAME; }            \
    FunctionPassID getID() const override {                                    \
      return FunctionPassID::PASS_NAME;                                        \
    }                                                                          \
  };
#include "FunctionPasses.def"

/// Helper that creates and \returns a FunctionPass given a provided \p passID.
inline std::unique_ptr<FunctionPass> createFunctionPass(FunctionPassID passID) {
  switch (passID) {
#define FUN_PASS(PASS_NAME)                                                    \
  case (FunctionPassID::PASS_NAME):                                            \
    return llvm::make_unique<PASS_NAME>();
#include "FunctionPasses.def"
  }
}

/// Implement a set of FunctionPasses. Implemented internally with a bitset.
class FunctionPassSet
    : private std::bitset<convertEnumToUnsigned(FunctionPassID::EmptyPass)> {

public:
  FunctionPassSet() = default;

  /// Construct a FunctionPassSet via an initializer_list \p IL.
  FunctionPassSet(std::initializer_list<FunctionPassID> IL) {
    for (const FunctionPassID &passID : IL) {
      this->set(convertEnumToUnsigned(passID));
    }
  }

  /// Inserts \p passID into the set.
  void insert(FunctionPassID passID) {
    this->set(convertEnumToUnsigned(passID));
  }
  /// Erases \p passID from the set.
  void erase(FunctionPassID passID) {
    this->reset(convertEnumToUnsigned(passID));
  }
  /// \returns whether \p passID is contained in the set.
  bool contains(FunctionPassID passID) const {
    return this->test(convertEnumToUnsigned(passID));
  }
};

} // namespace glow

#endif // GLOW_OPTIMIZER_GRAPHOPTIMIZER_FUNCTIONPASSES_H
