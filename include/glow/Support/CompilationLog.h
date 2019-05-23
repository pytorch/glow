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
#ifndef GLOW_SUPPORT_COMPILATIONLOG_H
#define GLOW_SUPPORT_COMPILATIONLOG_H

#include "llvm/ADT/StringRef.h"

#include <string>

namespace glow {

/// A class that defines the compilation scope.
class CompilationScope {
private:
  /// Name of current scope.
  /// Example: "ExecutionEngine::compile"
  std::string funcName_;
  std::string scopeName_;

public:
  /// When the constructor is called, current stack is pushed onto the scope
  /// stack.
  CompilationScope(llvm::StringRef funcName, llvm::StringRef scopeName);

  /// When the destructor is called, current stack is popped out from the scope
  /// stack.
  ~CompilationScope();

  llvm::StringRef getScopeName() const { return scopeName_; };
};

/// Dump the compilation log.
/// \param logContent the content that will be written into the log file.
void dumpCompilationLog(llvm::StringRef logContent);

/// Dump the compilation log to a certain file.
/// \param logFilename the file where the log will be written into.
/// \param logContent the content that will be written into the log file.
void dumpCompilationLog(llvm::StringRef logFilename,
                        llvm::StringRef logContent);

} // namespace glow

#endif // GLOW_SUPPORT_COMPILATIONLOG_H
