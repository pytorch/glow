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
#ifndef GLOW_OPTIMIZER_IROPTIMIZER_IRFUNCTIONPASSES_H
#define GLOW_OPTIMIZER_IROPTIMIZER_IRFUNCTIONPASSES_H

#include "glow/Optimizer/IROptimizer/IRFunctionPass.h"
#include "glow/Optimizer/IROptimizerPipeline/IRFunctionPassPipeline.h"

namespace glow {

/// Declare all IR Pass classes.
#define IR_FUN_PASS(PASS_NAME)                                                 \
  namespace ir {                                                               \
  class PASS_NAME : public IRFunctionPass {                                    \
  private:                                                                     \
    bool run(IRFunction *F, const CompilationContext &cctx) override;          \
    llvm::StringRef getName() const override { return #PASS_NAME; }            \
    IRFunctionPassID getID() const override {                                  \
      return IRFunctionPassID::PASS_NAME;                                      \
    }                                                                          \
  };                                                                           \
  }
#include "IRPasses.def"

/// Helper that creates and \returns a FunctionPass given a provided \p passID.
std::unique_ptr<IRFunctionPass> createFunctionPass(IRFunctionPassID passID);

} // namespace glow

#endif // GLOW_OPTIMIZER_IROPTIMIZER_IRFUNCTIONPASSES_H
