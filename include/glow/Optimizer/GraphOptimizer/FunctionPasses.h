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

namespace glow {

#define FUN_PASS(PASS_NAME)                                                    \
  class PASS_NAME : public FunctionPass {                                      \
  public:                                                                      \
    bool run(Function *F) override;                                            \
    llvm::StringRef getName() const override { return #PASS_NAME; }            \
  };

#include "FunctionPasses.def"

} // namespace glow

#endif // GLOW_OPTIMIZER_GRAPHOPTIMIZER_FUNCTIONPASSES_H
