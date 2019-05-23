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

#include "glow/Support/CompilationLog.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <fstream>

namespace glow {
auto constexpr defaultCompileLogFilename = "compile.log";
static llvm::cl::opt<bool>
    dumpCompilationLogOpt("dump-compilation-log",
                          llvm::cl::desc("Dump compilation log"));

CompilationScope::CompilationScope(llvm::StringRef funcName,
                                   llvm::StringRef scopeName)
    : funcName_(funcName), scopeName_(scopeName) {
  if (dumpCompilationLogOpt) {
    std::string startScopeMsg =
        "\n---Entering the compilation scope: " + funcName_ + " -> " +
        scopeName_;
    dumpCompilationLog(startScopeMsg.append(scopeName_));
  }
}

CompilationScope::~CompilationScope() {
  if (dumpCompilationLogOpt) {
    std::string endScopeMsg =
        "\n---Leaving the compilation scope: " + funcName_ + " -> " +
        scopeName_;
    dumpCompilationLog(endScopeMsg);
  }
}

void dumpCompilationLog(llvm::StringRef logContent) {
  dumpCompilationLog(defaultCompileLogFilename, logContent);
}
void dumpCompilationLog(llvm::StringRef logFilename,
                        llvm::StringRef logContent) {
  llvm::outs() << "Writing compilation log file for Module to: " << logFilename
               << '\n';
  std::error_code EC;
  llvm::raw_fd_ostream myfile(logFilename, EC, llvm::sys::fs::OF_Append);
  myfile << logContent << "\n";
}

} // namespace glow
