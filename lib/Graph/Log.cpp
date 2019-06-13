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

#include "glow/Graph/Log.h"
#include "glow/Graph/Node.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

namespace glow {

/// Log version number.
static constexpr auto logVersionNo_ = "v1.0.0";

static llvm::cl::opt<bool>
    dumpCompilationLogOpt("dump-compilation-log", llvm::cl::init(false),
                          llvm::cl::desc("Dump compilation log"));

void LogContext::addLogMetaData() {
  addLogContent("<!-- Log Version: ");
  addLogContent(logVersionNo_);
  addLogContent(" -->\n");
#ifdef GIT_SHA1
  addLogContent("<!-- Log commit sha1: ");
  addLogContent(GIT_SHA1);
  addLogContent(" -->\n");
#endif
#ifdef GIT_DATE
  addLogContent("<!-- Log commit date: ");
  addLogContent(GIT_DATE);
  addLogContent(" -->\n");
#endif
  addLogContent("\n");
}

void LogContext::addLogContent(llvm::StringRef logContent) {
  logContents_.push_back(logContent);
}

void LogContext::pushLogScope(llvm::StringRef scopeName) {
  logScopes_.push_back(scopeName);
  currentFullScope_ += "->";
  currentFullScope_ += scopeName;
}

void LogContext::popLogScope() {
  if (logScopes_.size() > 0) {
    currentFullScope_ = currentFullScope_.substr(
        0, currentFullScope_.size() - logScopes_.back().size() - 2);
    logScopes_.pop_back();
  }
}

void LogContext::dumpLog(llvm::StringRef funcName) {
  if (dumpCompilationLogOpt) {
    std::string compileLogFilename = funcName;
    compileLogFilename.append("_compile.log");

    llvm::outs() << "Writing compilation log file for Module to: "
                 << compileLogFilename << '\n';
    std::error_code EC;
    llvm::raw_fd_ostream myfile(compileLogFilename, EC);
    for (const auto &s : logContents_) {
      myfile << s;
    }
  }
}

/// Logs the node creation with a list of input nodes.
void LogContext::logNodeCreation(const Node *newNode) {
  if (dumpCompilationLogOpt) {
    std::vector<Node *> inputs;
    for (size_t idx = 0; idx < newNode->getNumInputs(); idx++) {
      inputs.push_back(newNode->getNthInput(idx).getNode());
    }
    std::string logStr;
    logStr.append("[FULL SCOPE: ");
    logStr.append(getFullScopeName());
    logStr.append("]");
    logStr.append(" --- CREATE { (Kind: ");
    logStr.append(newNode->getKindName());
    logStr.append(", Name: ");
    logStr.append(newNode->getName());
    logStr.append(") <== ");
    for (auto n : inputs) {
      logStr.append(" (Kind: ");
      logStr.append(n->getKindName());
      logStr.append(", Name: ");
      logStr.append(n->getName());
      logStr.append(") ");
    }
    logStr.append(" }\n");
    addLogContent(logStr);
  }
}

/// Logs the node replacement.
void LogContext::logNodeReplacement(const Node *oldNode, const Node *newNode) {
  if (dumpCompilationLogOpt) {
    std::string logStr;
    logStr.append("[FULL SCOPE: ");
    logStr.append(getFullScopeName());
    logStr.append(" ]");
    logStr.append(" --- REPLACE { (Kind: ");
    logStr.append(oldNode->getKindName());
    logStr.append(", Name: ");
    logStr.append(oldNode->getName());
    logStr.append(") <== (Kind: ");
    logStr.append(newNode->getKindName());
    logStr.append(", Name: ");
    logStr.append(newNode->getName());
    logStr.append(") }\n");
    addLogContent(logStr);
  }
}

/// Logs the node deletion.
void LogContext::logNodeDeletion(const Node &deletedNode) {
  if (dumpCompilationLogOpt) {
    std::string logStr;
    logStr.append("[FULL SCOPE: ");
    logStr.append(getFullScopeName());
    logStr.append(" ]");
    logStr.append(" --- DELETE { (Kind: ");
    logStr.append(deletedNode.getKindName());
    logStr.append(", Name: ");
    logStr.append(deletedNode.getName());
    logStr.append(") }\n");
    addLogContent(logStr);
  }
}

ScopedLogBlock::ScopedLogBlock(LogContext &ctx, llvm::StringRef name)
    : ctx_(ctx), name_(name) {
  ctx_.pushLogScope(name_);
  ctx_.addLogContent("============= ENTER SCOPE: ");
  ctx_.addLogContent(ctx_.getFullScopeName());
  ctx_.addLogContent(" ================================\n");
};

ScopedLogBlock::~ScopedLogBlock() {
  ctx_.addLogContent("============= EXIT SCOPE: ");
  ctx_.addLogContent(ctx_.getFullScopeName());
  ctx_.addLogContent(" ================================\n");
  end();
};

void ScopedLogBlock::end() {
  if (!end_) {
    ctx_.popLogScope();
  }
  end_ = true;
}

} // namespace glow
