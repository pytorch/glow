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

#ifndef GLOW_SUPPORT_LOG_H
#define GLOW_SUPPORT_LOG_H

#include "llvm/ADT/StringRef.h"

#include <string>
#include <vector>

namespace llvm {
class raw_fd_ostream;
}

namespace glow {

class Module;
class Function;
class Node;

struct NodeValue;

struct LogEvent {
  std::string name;
  LogEvent *parent{nullptr};
  std::vector<LogEvent *> children;

  LogEvent(llvm::StringRef n) : name(n) {}

  virtual ~LogEvent() {
    for (auto *c : children) {
      delete c;
    }
  }

  void pushEvent(LogEvent *e) {
    e->parent = this;
    children.push_back(e);
  }

  /// writes this Event and all children to the provided stream.
  /// \returns true if anything was written.
  virtual bool dump(llvm::raw_fd_ostream &ostream);

  virtual bool dumpChildren(llvm::raw_fd_ostream &ostream);

  /// \returns true if this even will not log to the stream.
  virtual bool silent() { return false; }

  virtual LogEvent *clone();
};

// A special LogEvent for Scope enclosing events.
struct LogScope : public LogEvent {
  LogScope(llvm::StringRef name) : LogEvent(name) {}
  virtual ~LogScope(){};
  bool dump(llvm::raw_fd_ostream &ostream) override;
  bool silent() override;
  LogEvent *clone() override;
};

struct LogCreate : public LogEvent {
  std::string kindName;
  std::vector<std::string> inputs;

  LogCreate(const Node *node);
  LogCreate(llvm::StringRef n, llvm::StringRef k, std::vector<std::string> &i);
  virtual ~LogCreate(){};

  bool dump(llvm::raw_fd_ostream &ostream) override;
  LogEvent *clone() override;
};

struct LogDelete : public LogEvent {
  std::string kindName;

  LogDelete(const Node *node);
  LogDelete(llvm::StringRef n, llvm::StringRef k);
  virtual ~LogDelete(){};

  bool dump(llvm::raw_fd_ostream &ostream) override;
  LogEvent *clone() override;
};

struct LogInputChange : public LogEvent {
  std::string kindName;
  std::string beforeName;
  std::string afterName;

  LogInputChange(const Node *user, const NodeValue &before,
                 const NodeValue &after);
  LogInputChange(llvm::StringRef n, llvm::StringRef k, llvm::StringRef b,
                 llvm::StringRef a);

  bool dump(llvm::raw_fd_ostream &ostream) override;
  LogEvent *clone() override;
};

/// A class for logging all compilation related activities.
class LogContext final {
private:
  LogScope topScope_{"Init"};
  LogEvent *currentScope_{nullptr};
  Module *parent_;

public:
  LogContext(Module *parent);

  /// Add a new event to the log.
  void pushEvent(LogEvent *ev);

  /// Add a new scope.
  void pushLogScope(llvm::StringRef scopeName);

  /// Pops out the most recently added scope.
  void popLogScope();

  /// Dumps the log in JSON format into the logfile (configured by command line
  /// option). /p funcName currently unused.
  void dumpLog(llvm::StringRef funcName);

  /// Logs the node creation. Also logs into Module log context if \p
  /// logIntoModule set as true.
  void logNodeCreation(const Node &newNode, bool logIntoModule = false);

  /// Logs the node deletion. Also logs into Module log context if \p
  /// logIntoModule set as true.
  void logNodeDeletion(const Node &deletedNode, bool logIntoModule = false);

  /// Logs node's input changes.
  /// \p user is the user node of the operands
  /// \p prevOpr previous operand
  /// \p newOpr new operand
  void logNodeInputChange(const Node &user, const NodeValue &prevOprVal,
                          const NodeValue &newOprVal);

  /// Get a deep cloned copy of the top level scope of this LogContext.
  LogEvent *getClonedScope();
};

/// Logs a new log scope.
#define LOG_SCOPE(ctx, name) ScopedLogBlock __scope__(ctx, name);

/// Helper class which traces the start and end of a compilation log scope.
class ScopedLogBlock {

  /// Reference to the log context.
  std::shared_ptr<LogContext> ctx_;

  /// The name of the log scope.
  std::string name_;

  /// Whether this log scope has already ended, avoiding logging
  /// it twice.
  bool end_{false};

public:
  ScopedLogBlock(std::shared_ptr<LogContext> ctx, llvm::StringRef name);
  ~ScopedLogBlock();

private:
  /// Triggers the endding operation of the current log scope.
  void end();
};

} // namespace glow.

#endif // GLOW_SUPPORT_LOG_H
