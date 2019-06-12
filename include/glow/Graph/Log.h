/**
 * Copyright (c) 2018-present, Facebook, Inc.
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

#ifndef GLOW_SUPPORT_LOG_H
#define GLOW_SUPPORT_LOG_H

#include "llvm/ADT/StringRef.h"

#include <string>
#include <vector>

namespace glow {

class Node;

/// A class for logging all compilation related activities.
class LogContext {
private:
  /// A vector that keeps track of all log contents.
  std::vector<std::string> logContents_;

  /// A vector that keeps track of current compilation scopes.
  std::vector<std::string> logScopes_;

  /// A string that represents the current full scope name.
  std::string currentFullScope_;

public:
  LogContext() { addLogMetaData(); };

  /// Add content into the contents vector.
  void addLogContent(llvm::StringRef logContent);

  /// Add a scope into the scope vector.
  void pushLogScope(llvm::StringRef scopeName);

  /// Getter that returns the current full scope name.
  llvm::StringRef getFullScopeName() { return currentFullScope_; }

  /// Pops out the most recently added scope.
  void popLogScope();

  /// Dumps the log into a file named after the given \p funcName.
  void dumpLog(llvm::StringRef funcName);

  /// Logs the node creation with a list of input nodes.
  void logNodeCreation(const Node *newNode);

  /// Logs the node replacement.
  void logNodeReplacement(const Node *oldNode, const Node *newNode);

  /// Logs the node deletion.
  void logNodeDeletion(const Node &deletedNode);

private:
  /// Add log metadata which includes version number and latest commit's info
  /// (hash, date).
  void addLogMetaData();
};

/// Logs a new log scope.
#define LOG_SCOPE(ctx, name) ScopedLogBlock __scope__(ctx, name);

/// Helper class which traces the start and end of a compilation log scope.
class ScopedLogBlock {

  /// Reference to the log context.
  LogContext &ctx_;

  /// The name of the log scope.
  std::string name_;

  /// Whether this log scope has already ended, avoiding logging
  /// it twice.
  bool end_{false};

public:
  ScopedLogBlock(LogContext &ctx, llvm::StringRef name);
  ~ScopedLogBlock();

private:
  /// Triggers the endding operation of the current log scope.
  void end();
};

} // namespace glow.

#endif // GLOW_SUPPORT_LOG_H
