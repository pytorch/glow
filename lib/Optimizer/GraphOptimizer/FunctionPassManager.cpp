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

#include "glow/Optimizer/GraphOptimizer/FunctionPassManager.h"
#include "glow/Optimizer/GraphOptimizer/FunctionPasses.h"

#include <glog/logging.h>

namespace glow {

/// The purpose of ThePassManager alias is to make the code of this pass manager
/// look as similar to other pass managers as possible. Often the changes in one
/// pass manager need to be replicated to other pass managaers. This alias makes
/// it easier to copy changes among pass managers as the code looks almost
/// identical.
using ThePassManager = FunctionPassManager;

/// Options for IRFunctionPassManager.
static PassManagerOptions functionPassManagerOptions("graph");

template <>
PassManagerOptions &ThePassManager::options_ = functionPassManagerOptions;

template <>
void ThePassManager::dumpIR(IRContainer *C, llvm::raw_ostream &os,
                            const std::string &outputFileName) const {
  static_cast<IRContainerTy *>(C)->dumpDAG(outputFileName);
}

std::unique_ptr<ThePassManager::IRPassTy>
createFunctionPass(ThePassManager::PassIDTy passID) {
  switch (passID) {
#define FUN_PASS(PASS_NAME)                                                    \
  case (ThePassManager::PassIDTy::PASS_NAME):                                  \
    return glow::make_unique<PASS_NAME>();
#include "glow/Optimizer/GraphOptimizer/FunctionPasses.def"
  }
  llvm_unreachable("Unexpected pass.");
}

template <>
std::unique_ptr<ThePassManager::IRPassTy>
ThePassManager::createFunctionPass(PassIDTy passID) const {
  return glow::createFunctionPass(passID);
}

template <>
bool ThePassManager::runPassHook(const PassConfigBase &passConfig, PassBase &P,
                                 IRContainer *C,
                                 const CompilationContext &cctx) {
  const IRPassConfigTy *thePassConfig =
      static_cast<const IRPassConfigTy *>(&passConfig);
  assert(
      !(thePassConfig->getPassID() == PassIDTy::DCE &&
        thePassConfig->getDCERequiredMode() == DCERequiredMode::BeforePass) &&
      "Cannot specify DCE requires DCE before it.");
  // Run DCE before this pass if it requires it.
  if (thePassConfig->getDCERequiredMode() == DCERequiredMode::BeforePass) {
    runPass(getDCEPassConfig(), static_cast<IRContainerTy *>(C), cctx);
  }
  return static_cast<IRPassTy *>(&P)->run(static_cast<IRContainerTy *>(C),
                                          cctx);
}

bool runDCEPass(ThePassManager::IRContainerTy *F,
                const CompilationContext &cctx) {
  auto pipeline = glow::make_unique<FunctionPassPipeline>();
  pipeline->pushBack(getDCEPassConfig());
  return FunctionPassManager("DCE_FPM", std::move(pipeline)).run(F, cctx);
}

void test() { FunctionPassManager FPM("name", "pipeline.def"); }

} // namespace glow
