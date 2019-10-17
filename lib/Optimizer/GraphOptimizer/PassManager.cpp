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

#include "glow/Optimizer/GraphOptimizer/PassManager.h"

#include "glow/Graph/Graph.h"
#include "glow/Optimizer/GraphOptimizer/FunctionPasses.h"

#include "llvm/Support/CommandLine.h"

#include <glog/logging.h>

#include <atomic>

using namespace glow;

namespace {
llvm::cl::OptionCategory passManagerCat("PassManager Options");

llvm::cl::opt<bool> verifyBeforeAllPassesOpt(
    "verify-before-all-passes",
    llvm::cl::desc("Verify the Function before all passes."),
    llvm::cl::Optional, llvm::cl::cat(passManagerCat));

llvm::cl::list<std::string> verifyBeforePassesOpt(
    "verify-before-passes",
    llvm::cl::desc("Verify the Function before the listed Passes."),
    llvm::cl::value_desc("FunctionPass names (e.g. DCE,CSE)"),
    llvm::cl::ZeroOrMore, llvm::cl::CommaSeparated,
    llvm::cl::cat(passManagerCat));

llvm::cl::opt<bool> verifyAfterAllPassesOpt(
    "verify-after-all-passes",
    llvm::cl::desc("Verify the Function after all passes."), llvm::cl::Optional,
    llvm::cl::cat(passManagerCat));

llvm::cl::list<std::string> verifyAfterPassesOpt(
    "verify-after-passes",
    llvm::cl::desc("Verify the Function after the listed Passes."),
    llvm::cl::value_desc("FunctionPass names (e.g. DCE,CSE)"),
    llvm::cl::ZeroOrMore, llvm::cl::CommaSeparated,
    llvm::cl::cat(passManagerCat));

llvm::cl::opt<bool> dumpDAGBeforeAllPassesOpt(
    "dump-graph-DAG-before-all-passes",
    llvm::cl::desc(
        "Debug option to export the Graph in DOT format before all passes."),
    llvm::cl::Optional, llvm::cl::cat(passManagerCat));

llvm::cl::list<std::string> dumpDAGBeforePassesOpt(
    "dump-graph-DAG-before-passes",
    llvm::cl::desc("Debug option to export the Graph in DOT format before the "
                   "listed Passes."),
    llvm::cl::value_desc("FunctionPass names (e.g. DCE,CSE)"),
    llvm::cl::ZeroOrMore, llvm::cl::CommaSeparated,
    llvm::cl::cat(passManagerCat));

llvm::cl::opt<bool> dumpDAGAfterAllPassesOpt(
    "dump-graph-DAG-after-all-passes",
    llvm::cl::desc(
        "Debug option to export the Graph in DOT format after all passes."),
    llvm::cl::Optional, llvm::cl::cat(passManagerCat));

llvm::cl::list<std::string> dumpDAGAfterPassesOpt(
    "dump-graph-DAG-after-passes",
    llvm::cl::desc("Debug option to export the Graph in DOT format after the "
                   "listed Passes."),
    llvm::cl::value_desc("FunctionPass names (e.g. DCE,CSE)"),
    llvm::cl::ZeroOrMore, llvm::cl::CommaSeparated,
    llvm::cl::cat(passManagerCat));

llvm::cl::opt<bool> printPassesOpt(
    "print-passes",
    llvm::cl::desc("Print all of the passes run by the pass manager."),
    llvm::cl::Optional, llvm::cl::cat(passManagerCat));

llvm::cl::opt<unsigned> stopAfterPassNumOpt(
    "stop-passes-after-num",
    llvm::cl::desc("Number of passes to run before preventing running any "
                   "passes. Used for debugging."),
    llvm::cl::init(std::numeric_limits<unsigned>::max()),
    llvm::cl::cat(passManagerCat));

/// Helper to check if \p otherStr is in \p strList.
static bool listContainsString(llvm::ArrayRef<std::string> strList,
                               llvm::StringRef otherStr) {
  for (llvm::StringRef str : strList) {
    if (str == otherStr) {
      return true;
    }
  }
  return false;
}

/// Global pass counter used to identify each pass.
static std::atomic<unsigned> globalPassCounter{0};

} // namespace

bool glow::runDCEPass(Function *F, CompilationContext &cctx) {
  return FunctionPassManager("DCE_FPM", {getDCEPassConfig()}).run(F, cctx);
}

void FunctionPassManager::dump(llvm::raw_ostream &os) const {
  os << "FunctionPassManager " << this->getName()
     << ": Current PassIdx: " << passIdx_
     << "; Current globalPassCounter: " << globalPassCounter << "\n";
  getPipeline().dump();
}

bool FunctionPassManager::runPrePass(Function *F,
                                     const CompilationContext &cctx,
                                     const FunctionPass &P) {
  if (printPassesOpt) {
    LOG(INFO) << "Starting Pass #" << globalPassCounter << ": "
              << P.getName().str() << " on Function: \""
              << F->getName().str() + "\"\n";
  }
  if (dumpDAGBeforeAllPassesOpt ||
      listContainsString(dumpDAGBeforePassesOpt, P.getName())) {
    F->dumpDAG(F->getName().str() + "_PrePass_" + P.getName().str() + "_n" +
               std::to_string(globalPassCounter) + ".dot");
  }
  if (verifyBeforeAllPassesOpt ||
      listContainsString(verifyBeforePassesOpt, P.getName())) {
    if (backend_) {
      // Do backend-specific verification.
      CHECK(backend_->verify(*F));
    } else {
      CHECK(F->verify());
    }
  }
  return false;
}

bool FunctionPassManager::runPostPass(Function *F,
                                      const CompilationContext &cctx,
                                      const FunctionPass &P) {
  if (printPassesOpt) {
    LOG(INFO) << "Finished Pass #" << globalPassCounter << ": "
              << P.getName().str() << " on Function: \""
              << F->getName().str() + "\"\n";
  }
  if (dumpDAGAfterAllPassesOpt ||
      listContainsString(dumpDAGAfterPassesOpt, P.getName())) {
    F->dumpDAG(F->getName().str() + "_PostPass_" + P.getName().str() + "_n" +
               std::to_string(globalPassCounter) + ".dot");
  }
  if (verifyAfterAllPassesOpt ||
      listContainsString(verifyAfterPassesOpt, P.getName())) {
    if (backend_) {
      // Do backend-specific verification.
      CHECK(backend_->verify(*F));
    } else {
      CHECK(F->verify());
    }
  }
  return false;
}

std::unique_ptr<FunctionPass>
FunctionPassManager::createFunctionPass(FunctionPassID passID) {
  switch (passID) {
#define FUN_PASS(PASS_NAME)                                                    \
  case (FunctionPassID::PASS_NAME):                                            \
    return llvm::make_unique<PASS_NAME>();
#include "glow/Optimizer/GraphOptimizer/FunctionPasses.def"
  }
  LOG(DFATAL) << "Cannot reach here.";
}

bool FunctionPassManager::runPass(const FunctionPassConfig &passConfig,
                                  Function *F, const CompilationContext &cctx) {
  const FunctionPassID &passID = passConfig.getFunctionPassID();
  assert(!(passID == FunctionPassID::DCE &&
           passConfig.getDCERequiredMode() == DCERequiredMode::BeforePass) &&
         "Cannot specify DCE requires DCE before it.");

  // Run DCE before this pass if it requires it.
  if (passConfig.getDCERequiredMode() == DCERequiredMode::BeforePass) {
    runPass(getDCEPassConfig(), F, cctx);
  }

  auto P = createFunctionPass(passID);
  bool changed = runPrePass(F, cctx, *P);
  changed |= P->run(F, cctx);
  changed |= runPostPass(F, cctx, *P);

  return changed;
}

bool FunctionPassManager::run(Function *F, const CompilationContext &cctx) {
  bool changed = false;
  for (passIdx_ = 0; passIdx_ < getPipeline().size(); passIdx_++) {
    const FunctionPassConfig &passConfig = getPipeline().at(passIdx_);
    // If we've exceeded the number of passes to run then early exit.
    if (++globalPassCounter > stopAfterPassNumOpt) {
      return changed;
    }

    // Skip some passes if specified by the config that they shouldn't be
    // executed in this compilation mode.
    if (!passConfig.isEnabledForCompilationMode(cctx.compMode)) {
      continue;
    }

    switch (passConfig.getConvergenceMode()) {
    case ConvergenceMode::OnePass:
      changed |= runPass(passConfig, F, cctx);
      break;

    case ConvergenceMode::UntilFixedPoint:
      while (runPass(passConfig, F, cctx)) {
        changed = true;
        VLOG_IF_EVERY_N(0, google::COUNTER > 1, 100)
            << "Warning: "
            << getNameOfPass(passConfig.getFunctionPassID()).str()
            << " Pass applied another 100 iterations without reaching fixed "
               "point";
      }
      break;
    }
  }
  return changed;
}
