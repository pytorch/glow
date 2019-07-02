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

#include "glow/Optimizer/GraphOptimizer/FunctionPasses.h"

using namespace glow;

/// Helper that creates and \returns a FunctionPass given a provided \p passID.
std::unique_ptr<FunctionPass> glow::createFunctionPass(FunctionPassID passID) {
  switch (passID) {
#define FUN_PASS(PASS_NAME)                                                    \
  case (FunctionPassID::PASS_NAME):                                            \
    return llvm::make_unique<PASS_NAME>();
#include "glow/Optimizer/GraphOptimizer/FunctionPasses.def"
  }
}
