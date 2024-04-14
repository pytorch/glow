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

#include "glow/PassManager/PassManager.h"

#include <glog/logging.h>

using namespace glow;

/// Helper to check if \p otherStr is in \p strList.
bool PassManagerOptions::listContainsString(
    const llvm::cl::list<std::string> &strList, llvm::StringRef otherStr) {
  for (llvm::StringRef str : strList) {
    if (str == otherStr) {
      return true;
    }
  }
  return false;
}

/// Construct pass manager command-line options. Use a provided identifier \p id
/// to create different option names for different pass managers.
///
/// NOTE: The names of the command-line options are created dynamically using
/// the unique \p id. It is important to make those strings live until the end
/// of the execution due to the way how LLVM's command-line registration
/// machinery works.
PassManagerOptions::PassManagerOptions(const char *id)
    : passManagerID(id),
      passManagerCat{staticStrFormat("%s Pass Manager Options", id)},
      verifyBeforeAllPassesOpt{
          llvm::StringRef(staticStrFormat("verify-before-all-%s-passes", id)),
          llvm::cl::desc("Verify the IR before all passes."),
          llvm::cl::Optional, llvm::cl::cat(passManagerCat)},

      verifyBeforePassesOpt{
          llvm::StringRef(staticStrFormat("verify-before-%s-passes", id)),
          llvm::cl::desc("Verify the IR before the listed passes."),
          llvm::cl::value_desc("Comma separated pass names (e.g. DSE, DCE)"),
          llvm::cl::ZeroOrMore,
          llvm::cl::CommaSeparated,
          llvm::cl::cat(passManagerCat)},

      verifyAfterAllPassesOpt{
          llvm::StringRef(staticStrFormat("verify-after-all-%s-passes", id)),
          llvm::cl::desc("Verify the IR after all passes."), llvm::cl::Optional,
          llvm::cl::cat(passManagerCat)},

      verifyAfterPassesOpt{
          llvm::StringRef(staticStrFormat("verify-after-%s-passes", id)),
          llvm::cl::desc("Verify the IR after the listed passes."),
          llvm::cl::value_desc("Comma separated pass names (e.g. DSE, DCE)"),
          llvm::cl::ZeroOrMore,
          llvm::cl::CommaSeparated,
          llvm::cl::cat(passManagerCat)},

      dumpIRBeforeAllPassesOpt{
          llvm::StringRef(staticStrFormat("dump-%s-before-all-passes", id)),
          llvm::cl::desc("Debug option to dump the IR before all passes."),
          llvm::cl::Optional, llvm::cl::cat(passManagerCat)},

      dumpIRBeforePassesOpt{
          llvm::StringRef(staticStrFormat("dump-%s-before-passes", id)),
          llvm::cl::desc("Debug option to dump the IR before the "
                         "listed passes."),
          llvm::cl::value_desc("Comma separated pass names (e.g. DSE, DCE)"),
          llvm::cl::ZeroOrMore,
          llvm::cl::CommaSeparated,
          llvm::cl::cat(passManagerCat)},

      dumpIRAfterAllPassesOpt{
          llvm::StringRef(staticStrFormat("dump-%s-after-all-passes", id)),
          llvm::cl::desc("Debug option to dump the IR after all passes."),
          llvm::cl::Optional, llvm::cl::cat(passManagerCat)},

      dumpIRAfterPassesOpt{
          llvm::StringRef(staticStrFormat("dump-%s-after-passes", id)),
          llvm::cl::desc("Debug option to dump the IR after listed passes."),
          llvm::cl::value_desc("Comma separated pass names (e.g. DSE, DCE)"),
          llvm::cl::ZeroOrMore,
          llvm::cl::CommaSeparated,
          llvm::cl::cat(passManagerCat)},

      printPassesOpt{
          llvm::StringRef(staticStrFormat("print-%s-passes", id)),
          llvm::cl::desc("Print all of the passes run by the pass manager."),
          llvm::cl::Optional, llvm::cl::cat(passManagerCat)},

      stopAfterPassNumOpt{
          llvm::StringRef(staticStrFormat("stop-%s-passes-after-num", id)),
          llvm::cl::desc(
              "Number of passes to run before preventing running any "
              "passes. Used for debugging."),
          llvm::cl::init(std::numeric_limits<unsigned>::max()),
          llvm::cl::cat(passManagerCat)} {}

void PassManagerBase::dump(llvm::raw_ostream &os) const {
  os << getOptions().passManagerID << "PassManager " << getName()
     << ": Current PassIdx: " << passIdx_
     << "; Current globalPassCounter: " << globalPassCounter() << "\n";
}

bool PassManagerBase::runPrePass(IRContainer *C, const CompilationContext &cctx,
                                 const PassBase &P) {
  if (getOptions().printPassesOpt) {
    LOG(INFO) << "Starting Pass #" << globalPassCounter() << ": "
              << P.getName().str() << " on Function: \""
              << C->getName().str() + "\"\n";
  }
  runPrePassHook(C, cctx, P);
  if (getOptions().verifyBeforeAllPassesOpt ||
      PassManagerOptions::listContainsString(getOptions().verifyBeforePassesOpt,
                                             P.getName())) {
    if (backend_) {
      // Do backend-specific verification.
      CHECK(verify(*backend_, *C));
    } else {
      CHECK(verify(*C));
    }
  }
  return false;
}

bool PassManagerBase::runPostPass(IRContainer *C,
                                  const CompilationContext &cctx,
                                  const PassBase &P) {
  if (getOptions().printPassesOpt) {
    LOG(INFO) << "Finished Pass #" << globalPassCounter() << ": "
              << P.getName().str() << " on Function: \""
              << C->getName().str() + "\"\n";
  }
  runPostPassHook(C, cctx, P);
  if (getOptions().verifyAfterAllPassesOpt ||
      PassManagerOptions::listContainsString(getOptions().verifyAfterPassesOpt,
                                             P.getName())) {
    if (backend_) {
      // Do backend-specific verification.
      CHECK(verify(*backend_, *C));
    } else {
      CHECK(verify(*C));
    }
  }
  return false;
}

void PassManagerBase::runPrePassHook(IRContainer *C,
                                     const CompilationContext &cctx,
                                     const PassBase &P) {
  if (getOptions().dumpIRBeforeAllPassesOpt ||
      PassManagerOptions::listContainsString(getOptions().dumpIRBeforePassesOpt,
                                             P.getName())) {
    glow::outs() << getOptions().passManagerID << " before pass " << P.getName()
                 << "\n";
    dumpIR(C, glow::outs(),
           C->getName().str() + "_PrePass_" + P.getName().str() + "_n" +
               std::to_string(globalPassCounter()) + "_" +
               std::to_string(iterationCount_++) + ".dot");
  }
}

void PassManagerBase::runPostPassHook(IRContainer *C,
                                      const CompilationContext &cctx,
                                      const PassBase &P) {
  if (getOptions().dumpIRAfterAllPassesOpt ||
      PassManagerOptions::listContainsString(getOptions().dumpIRAfterPassesOpt,
                                             P.getName())) {
    glow::outs() << getOptions().passManagerID << " after pass " << P.getName()
                 << "\n";
    dumpIR(C, glow::outs(),
           C->getName().str() + "_PostPass_" + P.getName().str() + "_n" +
               std::to_string(globalPassCounter()) + "_" +
               std::to_string(iterationCount_++) + ".dot");
  }
}

bool PassManagerBase::runPass(const PassConfigBase &passConfig, IRContainer *F,
                              const CompilationContext &cctx) {
  auto pass = createFunctionPass(passConfig);
  auto &P = *pass;
  bool changed = runPrePass(F, cctx, P);
  changed |= runPassHook(passConfig, P, F, cctx);
  changed |= runPostPass(F, cctx, P);
  return changed;
}

bool PassManagerBase::run(IRContainer *C, const CompilationContext &cctx) {
  bool changed = false;
  size_t e = getPipelineSize();
  for (passIdx_ = 0; passIdx_ < e; passIdx_++) {
    const PassConfigBase &passConfig = getPipelineElement(passIdx_);

    // If we've exceeded the number of passes to run then early exit.
    if (++globalPassCounter() > getOptions().stopAfterPassNumOpt) {
      return changed;
    }

    // Skip some passes if specified by the config that they shouldn't be
    // executed in this compilation mode.
    if (!passConfig.isEnabledForCompilationMode(cctx.compMode)) {
      continue;
    }

    switch (passConfig.getConvergenceMode()) {
    case ConvergenceMode::OnePass:
      changed |= runPassWithConfig(passConfig, C, cctx);
      break;

    case ConvergenceMode::UntilFixedPoint:
      while (runPassWithConfig(passConfig, C, cctx)) {
        changed = true;
        VLOG_IF_EVERY_N(0, google::COUNTER > 1, 100)
            << "Warning: " << getNameOfPass(passConfig).str()
            << " Pass applied another 100 iterations without reaching fixed "
               "point";
      }
      break;
    }
  }
  return changed;
}
