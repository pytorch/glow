// Copyright 2017 Facebook Inc.  All Rights Reserved.
#define DEBUG_TYPE "optimizer"

#include "glow/Optimizer/OptimizerUtils.h"
#include "glow/Optimizer/Optimizer.h"

#include "llvm/Support/CommandLine.h"

#include <iostream>
#include <sstream>

using namespace glow;

namespace {

/// The format of this option is:
/// -max-changes-num=pass-name,max-changes-numer-for-this-pass
/// A typical use could look like: -max-changes-num=ir-share-buffers,55
static llvm::cl::list<std::string> stopAfterChanges(
    "max-changes-num",
    llvm::cl::desc(
        "Max number of changes to be performed by an optimization pass"),
    llvm::cl::Hidden, llvm::cl::ZeroOrMore);
} // namespace

namespace glow {
/// Initialize the NumMaxChanges_ from the command-line options.
void ChangeManager::initFromCommandLineOptions() {
  for (auto &opt : stopAfterChanges) {
    std::istringstream opts(opt);
    std::string passName;
    std::string numMaxChangesStr;
    std::getline(opts, passName, ',');
    std::getline(opts, numMaxChangesStr);
    int numChanges = atoi(numMaxChangesStr.c_str());
    if (passName != getOptId())
      continue;
    if (errno) {
      std::cerr << "Unknown arguments in option max-changes-num: " << opt
                << " pass name: '" << passName
                << "' max changes number: " << numMaxChangesStr << "\n";
      abort();
    }
    setNumMaxChanges(numChanges);
    std::cout << "Set max-changes-number for the pass `" << passName << "` to "
              << numChanges << "\n";
    return;
  }
}
} // namespace glow
