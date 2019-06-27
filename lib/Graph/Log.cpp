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
#include "glow/Graph/Graph.h"
#include "glow/Graph/Node.h"
#include "glow/Graph/NodeValue.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

namespace glow {

/// Log version number.
static constexpr auto logVersionNo_ = "v1.0.0";

static llvm::cl::opt<bool>
    dumpCompilationLogOpt("dump-compilation-log", llvm::cl::init(false),
                          llvm::cl::desc("Dump compilation log"));

void ModuleLogContext::addModuleLogContent(llvm::StringRef logContent) {
  moduleLogContents_.push_back(logContent);
}

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

LogContext::LogContext() {
  addLogMetaData();

  if (!parent_) {
    return;
  }

  // Add the logs of module constants/placeholders into the log context of this
  // function. Such that the Log Context has the information of any
  // constants/placeholders created in previous functions.
  for (const auto &s :
       parent_->getParent()->getModuleLogContext().getModuleLog()) {
    addLogContent(s);
  }
  addLogContent(llvm::formatv("----------- Enter function: {0} ----------\n",
                              parent_->getName())
                    .str());
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
  if (!dumpCompilationLogOpt) {
    return;
  }
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

/// Logs the node creation with a list of input nodes.
void LogContext::logNodeCreation(const Node &newNode, bool logIntoModule) {
  if (!dumpCompilationLogOpt) {
    return;
  }
  std::vector<NodeValue> inputs;
  for (size_t idx = 0; idx < newNode.getNumInputs(); idx++) {
    inputs.push_back(newNode.getNthInput(idx));
  }

  std::string contentStr =
      llvm::formatv(" --- CREATE { (Kind: {0}, Name: {1}) <== ",
                    newNode.getKindName(), newNode.getName())
          .str();
  for (auto &n : inputs) {
    contentStr += llvm::formatv(" (Kind: {0}, Name: {1}, ResNo: {2}) ",
                                n.getNode()->getKindName(),
                                n.getNode()->getName(), n.getResNo())
                      .str();
  }
  contentStr += " }\n";

  // Use scope as "Module" when writting into Module logs.
  if (logIntoModule) {
    if (parent_) {
      std::string scopeName = "[FULL SCOPE: Module ]";
      parent_->getParent()->getModuleLogContext().addModuleLogContent(
          scopeName + contentStr);
    }

  } else {
    std::string scopeName =
        llvm::formatv("[FULL SCOPE: {0}]", getFullScopeName()).str();
    addLogContent(scopeName + contentStr);
  }
}

/// Logs the node deletion.
void LogContext::logNodeDeletion(const Node &deletedNode, bool logIntoModule) {
  if (!dumpCompilationLogOpt) {
    return;
  }

  std::string contentStr =
      llvm::formatv(" --- DELETE ( (Kind: {0}, "
                    "Name: {1}) }\n",
                    deletedNode.getKindName(), deletedNode.getName())
          .str();

  // Use scope as "Module" when writting into Module logs.
  if (logIntoModule) {
    if (parent_) {
      std::string scopeName = "[FULL SCOPE: Module ]";
      parent_->getParent()->getModuleLogContext().addModuleLogContent(
          scopeName + contentStr);
    }

  } else {
    std::string scopeName =
        llvm::formatv("[FULL SCOPE: {0} ]", getFullScopeName()).str();
    addLogContent(scopeName + contentStr);
  }
}

/// Logs node's input changes.
void LogContext::logNodeInputChange(const Node &user,
                                    const NodeValue &prevOprVal,
                                    const NodeValue &newOprVal) {
  if (!dumpCompilationLogOpt) {
    return;
  }
  addLogContent(
      llvm::formatv(
          "[FULL SCOPE: {0} ] --- NODE_INPUT_CHANGE { User(Kind: {1}, "
          "Name: {2}) :: ",
          getFullScopeName(), user.getKindName(), user.getName())
          .str());

  // prevOpr.getNode()should never be null.
  addLogContent(
      llvm::formatv("PrevOprValue(Kind: {0}, Name: {1}, ResNo: {2}) -> ",
                    prevOprVal.getNode()->getKindName(),
                    prevOprVal.getNode()->getName(), prevOprVal.getResNo())
          .str());

  if (newOprVal.getNode()) {
    addLogContent(
        llvm::formatv("NewOprValue(Kind: {0}, Name: {1}, ResNo: {2}) }\n",
                      newOprVal.getNode()->getKindName(),
                      newOprVal.getNode()->getName(), newOprVal.getResNo())
            .str());
  } else {
    addLogContent("NewOprValue(null) }\n");
  }
}

ScopedLogBlock::ScopedLogBlock(std::shared_ptr<LogContext> ctx,
                               llvm::StringRef name)
    : ctx_(ctx), name_(name) {
  ctx_->pushLogScope(name_);
  ctx_->addLogContent("============= ENTER SCOPE: ");
  ctx_->addLogContent(ctx_->getFullScopeName());
  ctx_->addLogContent(" ================================\n");
};

ScopedLogBlock::~ScopedLogBlock() {
  ctx_->addLogContent("============= EXIT SCOPE: ");
  ctx_->addLogContent(ctx_->getFullScopeName());
  ctx_->addLogContent(" ================================\n");
  end();
};

void ScopedLogBlock::end() {
  if (!end_) {
    ctx_->popLogScope();
  }
  end_ = true;
}

} // namespace glow
