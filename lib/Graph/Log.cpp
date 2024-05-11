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

#include "glow/Graph/Log.h"

#include "glow/Flags/Flags.h"
#include "glow/Graph/Graph.h"
#include "glow/Graph/Node.h"
#include "glow/Graph/NodeValue.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

namespace glow {

/// Log version number.
static constexpr auto logVersionNo_ = "v1.0.0";

static llvm::cl::opt<bool, true>
    enableCompilationLogOpt("compilation-log",
                            llvm::cl::desc("Dump Compilation Log"),
                            llvm::cl::location(flags::DumpCompilationLog));

static llvm::cl::opt<bool> verboseCompilationLogOpt(
    "verbose-compilation", llvm::cl::init(false),
    llvm::cl::desc("Log empty passes to Compilation Log"));

bool LogEvent::dump(llvm::raw_fd_ostream &ostream) {
  ostream << llvm::formatv("{ \"name\":\"{0}\",", name);

  if (!children.empty()) {
    ostream << llvm::format(", \"children\":\n");
    dumpChildren(ostream);
  }

  ostream << std::string("}");
  return true;
}

bool LogEvent::dumpChildren(llvm::raw_fd_ostream &ostream) {
  ostream << std::string("[");
  bool first = true;
  for (auto *c : children) {
    DCHECK(c);
    if (c->silent()) {
      continue;
    }

    if (first) {
      first = false;
    } else {
      ostream << std::string(",\n");
    }

    c->dump(ostream);
  }
  ostream << std::string("]");
  return true;
}

LogEvent *LogEvent::clone() {
  LogEvent *copy = new LogEvent(name);
  copy->parent = parent;
  for (auto *c : children) {
    copy->children.push_back(c->clone());
  }
  return copy;
}

bool LogScope::dump(llvm::raw_fd_ostream &ostream) {
  if (silent()) {
    return false;
  }

  ostream << llvm::formatv("{\"{0}\":", name);
  dumpChildren(ostream);
  ostream << std::string("}");
  return true;
}

bool LogScope::silent() {
  if (children.empty()) {
    return true;
  }

  for (auto &c : children) {
    if (!c->silent()) {
      return false;
    }
  }

  return true;
}

LogEvent *LogScope::clone() {
  LogEvent *copy = new LogScope(name);
  copy->parent = parent;
  for (auto *c : children) {
    copy->children.push_back(c->clone());
  }
  return copy;
}

LogCreate::LogCreate(const Node *node) : LogEvent(node->getName()) {
  kindName = node->getKindName();

  inputs.resize(node->getNumInputs());
  for (size_t idx = 0; idx < node->getNumInputs(); idx++) {
    auto &nv = node->getNthInput(idx);
    inputs[idx] =
        llvm::formatv("\"{0}:{1}\"", nv.getNode()->getName(), nv.getResNo());
  }
}

LogCreate::LogCreate(llvm::StringRef n, llvm::StringRef k,
                     std::vector<std::string> &i)
    : LogEvent(n), kindName(k) {
  std::copy(i.begin(), i.end(), inputs.begin());
}

bool LogCreate::dump(llvm::raw_fd_ostream &ostream) {
  ostream << llvm::formatv(
      "{\"create\":\"{0}\", \"kind\":\"{1}\", \"inputs\": [", name, kindName);

  if (!inputs.empty()) {
    ostream << "\n" + llvm::join(inputs.begin(), inputs.end(), ",\n");
  }

  ostream << std::string("]}");
  return true;
}

LogEvent *LogCreate::clone() {
  LogEvent *copy = new LogCreate(name, kindName, inputs);
  copy->parent = parent;
  for (auto *c : children) {
    copy->children.push_back(c->clone());
  }
  return copy;
}

LogDelete::LogDelete(const Node *node) : LogEvent(node->getName()) {
  kindName = node->getKindName();
}

LogDelete::LogDelete(llvm::StringRef n, llvm::StringRef k)
    : LogEvent(n), kindName(k) {}

bool LogDelete::dump(llvm::raw_fd_ostream &ostream) {
  ostream << llvm::formatv("{\"delete\":\"{0}\", \"kind\":\"{1}\"}", name,
                           kindName);
  return true;
}

LogEvent *LogDelete::clone() {
  LogEvent *copy = new LogDelete(name, kindName);
  copy->parent = parent;
  for (auto *c : children) {
    copy->children.push_back(c->clone());
  }
  return copy;
}

LogInputChange::LogInputChange(const Node *user, const NodeValue &before,
                               const NodeValue &after)
    : LogEvent(user->getName()) {
  kindName = user->getKindName();
  beforeName =
      llvm::formatv("{0}:{1}", before.getNode()->getName(), before.getResNo());
  afterName = "NONE";
  if (after.getNode()) {
    afterName =
        llvm::formatv("{0}:{1}", after.getNode()->getName(), after.getResNo());
  }
}

LogInputChange::LogInputChange(llvm::StringRef n, llvm::StringRef k,
                               llvm::StringRef b, llvm::StringRef a)
    : LogEvent(n), kindName(k), beforeName(b), afterName(a) {}

bool LogInputChange::dump(llvm::raw_fd_ostream &ostream) {
  ostream << llvm::formatv("{\"input_change\":\"{0}\", \"kind\":\"{1}\", "
                           "\"before\":\"{2}\", \"after\":\"{3}\"}",
                           name, kindName, beforeName, afterName);
  return true;
}

LogEvent *LogInputChange::clone() {
  LogEvent *copy = new LogInputChange(name, kindName, beforeName, afterName);
  copy->parent = parent;
  for (auto *c : children) {
    copy->children.push_back(c->clone());
  }
  return copy;
}

LogContext::LogContext(Module *parent)
    : currentScope_(&topScope_), parent_(parent) {}

void LogContext::pushEvent(LogEvent *ev) { currentScope_->pushEvent(ev); }

void LogContext::pushLogScope(llvm::StringRef scopeName) {
  LogScope *scope = new LogScope(scopeName);
  currentScope_->pushEvent(scope);
  currentScope_ = currentScope_->children.back();
}

void LogContext::popLogScope() {
  DCHECK(currentScope_->parent);
  currentScope_ = currentScope_->parent;
}

void LogContext::dumpLog(llvm::StringRef compileLogFilename) {
  if (!flags::DumpCompilationLog) {
    return;
  }

  llvm::outs() << "Writing compilation log file to: " << compileLogFilename
               << '\n';
  std::error_code EC;
  llvm::raw_fd_ostream myfile(compileLogFilename, EC);
  myfile << llvm::formatv("{ \"log\":\"Glow Compilation Log\", "
                          "\"version\":\"{0}\", ",
                          logVersionNo_);
#ifdef GIT_SHA1
  myfile << llvm::formatv("\"commitHash\":\"{0}\", ", GIT_SHA1);
#endif
#ifdef GIT_DATE
  myfile << llvm::formatv("\"commitDate\":\"{0}\", ", GIT_DATE);
#endif
  myfile << std::string("\"passes\":");
  topScope_.dumpChildren(myfile);
  myfile << std::string("}\n");
}

/// Logs the node creation with a list of input nodes.
void LogContext::logNodeCreation(const Node &newNode, bool logIntoModule) {
  if (!flags::DumpCompilationLog) {
    return;
  }

  LogEvent *ev = new LogCreate(&newNode);
  if (logIntoModule) {
    parent_->getModuleLogContext()->pushEvent(ev);
  } else {
    currentScope_->pushEvent(ev);
  }
}

/// Logs the node deletion.
void LogContext::logNodeDeletion(const Node &deletedNode, bool logIntoModule) {
  if (!flags::DumpCompilationLog) {
    return;
  }

  LogEvent *ev = new LogDelete(&deletedNode);
  if (logIntoModule) {
    parent_->getModuleLogContext()->pushEvent(ev);
  } else {
    currentScope_->pushEvent(ev);
  }
}

/// Logs node's input changes.
void LogContext::logNodeInputChange(const Node &user,
                                    const NodeValue &prevOprVal,
                                    const NodeValue &newOprVal) {
  if (!flags::DumpCompilationLog) {
    return;
  }

  LogEvent *ev = new LogInputChange(&user, prevOprVal, newOprVal);
  currentScope_->pushEvent(ev);
}

LogEvent *LogContext::getClonedScope() { return topScope_.clone(); }

ScopedLogBlock::ScopedLogBlock(std::shared_ptr<LogContext> ctx,
                               llvm::StringRef name)
    : ctx_(ctx), name_(name) {
  ctx_->pushLogScope(name_);
};

ScopedLogBlock::~ScopedLogBlock() { end(); };

void ScopedLogBlock::end() {
  if (!end_) {
    ctx_->popLogScope();
  }
  end_ = true;
}

} // namespace glow
