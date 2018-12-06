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

#include "ThreadPoolExecutor.h"
#include "glow/Backends/DeviceManager.h"

namespace glow {
namespace runtime {

/// Forward declaration of getDeviceManager function.
/// TODO: Talk to gcatron about the details of this.
std::shared_ptr<DeviceManager> getDeviceManager(DeviceIDty deviceId);

void ThreadPoolExecutor::run(std::vector<DAGNode *> roots,
                             std::unique_ptr<Context> context,
                             RunIdentifierTy runId, DoneCb cb) {
  // If list of roots is empty, there is nothing to do.
  if (roots.empty()) {
    cb(runId, ResultCode::EXECUTED, nullptr);
    return;
  }

  // If the given run ID corresponds to a run already in progress, there is
  // also nothing to do, but return an error.
  std::unique_lock<std::mutex> lock(executionStateLocksMtx_);
  if (executionStateLocks_.find(runId) != executionStateLocks_.end()) {
    cb(runId, ResultCode::FAILED, nullptr);
    return;
  }

  // Create execution state tracker object and lock for this run ID.
  executionStateLocks_[runId];
  executionStates_.insert(std::make_pair(
      runId, std::make_shared<ExecutionState>(runId, std::move(cb))));
  lock.unlock();

  // Execute all root nodes.
  for (auto const &root : roots) {
    // Propagate placeholders from the given starter Context into the input
    // Context for the current node being processed.
    propagatePlaceholdersForNode(runId, root, context.get());

    // Execute the node.
    executeDAGNode(runId, root);
  }
}

void ThreadPoolExecutor::propagatePlaceholdersForNode(RunIdentifierTy runId,
                                                      DAGNode *node,
                                                      Context *ctx) {
  // Get the execution state for the run.
  std::shared_ptr<ExecutionState> executionState = executionStates_[runId];

  // Try to get a pointer to the input Context for the node. If such a
  /// Context does not exist yet, make one.
  auto inputCtxIt = (executionState->inputCtxs).find(node);

  if (inputCtxIt == (executionState->inputCtxs).end()) {
    (executionState->inputCtxs)
        .insert(std::make_pair(node, std::make_unique<Context>()));
    inputCtxIt = (executionState->inputCtxs).find(node);
  }

  Context *nodeInputCtx = (inputCtxIt->second).get();

  // Get the map of Placeholder -> Tensor mappings within ctx.
  const Context::PlaceholderMap &phMap = ctx->pairs();

  // Get the symbol table for the node.
  const RuntimeBundle &bundle = node->runtimeBundle;
  const SymbolTable &symbolTable = bundle.getSymbolTable();

  for (const auto &symbolPair : symbolTable) {
    const auto &symbolName = symbolPair.first;
    const auto &symbolInfo = symbolPair.second;

    if (symbolInfo.input) {
      // If the symbol is an input, look for a Placeholder in ctx with
      // the same name and copy the corresponding Tensor into the input Context
      // being prepared for the node.
      for (const auto &phTensorPair : phMap) {
        auto *placeholder = phTensorPair.first;
        const auto *tensor = phTensorPair.second;

        if (symbolName == placeholder->getName()) {
          nodeInputCtx->insert(placeholder, tensor->clone());
        }
      }
    }
  }
}

void ThreadPoolExecutor::executeDAGNode(RunIdentifierTy runId, DAGNode *node) {
  // Get the execution state for the run.
  std::shared_ptr<ExecutionState> executionState = executionStates_[runId];

  // Get the DeviceManager that can run the node.
  std::shared_ptr<DeviceManager> deviceManager =
      getDeviceManager(node->deviceID);

  // Get the Context containing all of the inputs for the node.
  std::unique_ptr<Context> nodeCtx =
      std::move((executionState->inputCtxs)[node]);
  // Mark the node as "inflight" (i.e. currently executing).
  (executionState->inflightNodes).insert(node);
  // Run the node using the DeviceManager.
  deviceManager->runFunction(
      node->name, std::move(nodeCtx),
      [this, runId, node](RunIdentifierTy id, ResultCode resultCode,
                          std::unique_ptr<Context> resultCtx) {
        // Immediately move the handling of the result onto threadPool_ to
        // avoid doing work on the DeviceManager thread.
        this->threadPool_.submit([this, runId, node, resultCode, &resultCtx]() {
          this->handleDeviceManagerResult(runId, std::move(resultCode),
                                          std::move(resultCtx), node);
        });
      });
}

void ThreadPoolExecutor::propagateOutputPlaceholders(
    RunIdentifierTy runId, std::unique_ptr<Context> ctx) {
  // Get the execution state for the run.
  std::shared_ptr<ExecutionState> executionState = executionStates_[runId];

  // Copy all of the Placeholders in ctx into the result Context for the run.
  for (const auto &phTensorPair : ctx->pairs()) {
    auto *placeholder = phTensorPair.first;
    const auto *tensor = phTensorPair.second;

    executionState->resultCtx->insert(placeholder, tensor->clone());
  }
}

void ThreadPoolExecutor::handleDeviceManagerResult(RunIdentifierTy runId,
                                                   ResultCode resultCode,
                                                   std::unique_ptr<Context> ctx,
                                                   DAGNode *node) {
  // Check if there is a lock for the run corresponding to runId.
  std::unique_lock<std::mutex> locksMtxLock(executionStateLocksMtx_);

  auto executionStateLockIt = executionStateLocks_.find(runId);
  if (executionStateLockIt == executionStateLocks_.end()) {
    // This means an earlier call to this function must have deleted the lock
    // for this runId after a failed run of one DAG component. Do nothing.
    return;
  }

  // Lock the lock for the runId.
  std::unique_lock<std::mutex> executionStateLock(executionStateLockIt->second);

  // Get the execution state for the run.
  auto executionStateIt = executionStates_.find(runId);
  if (executionStateIt == executionStates_.end()) {
    // This should never happen. TODO: Log, assert, something.
    return;
  }

  std::shared_ptr<ExecutionState> executionState = executionStateIt->second;

  if (resultCode == ResultCode::CANCELLED || resultCode == ResultCode::FAILED) {
    // If the DeviceManager failed to execute the node, call the callback
    // provided when ThreadPoolExecutor::run() was called, clean up the
    // lock and execution state for the run and exit.
    DoneCb cb = executionState->cb;
    cb(runId, resultCode, nullptr);
    executionStates_.erase(runId);
    executionStateLocks_.erase(runId);
  } else {
    // The DeviceManager did not fail to execute the node. This lock is no
    // longer needed since executionStateLocks_ won't be modified.
    locksMtxLock.unlock();

    // Erase the input Context for the node because it is no longer needed.
    (executionState->inputCtxs).erase(node);

    if ((node->children).empty()) {
      // If the node has no children, propagate its outputs to the result
      // Context for the run.
      propagateOutputPlaceholders(runId, std::move(ctx));
    } else {
      // If the node has children, propagate its outputs to the input Contexts
      // for any of its children that need them as inputs.
      for (auto &child : node->children) {
        propagatePlaceholdersForNode(runId, child, ctx.get());

        // Execute any children that has no parent nodes left to execute.
        (executionState->nodeParentsDone)[child] += 1;
        if ((executionState->nodeParentsDone)[child] ==
            (child->parents).size()) {
          executeDAGNode(runId, child);
        }
      }
    }

    // Clean up all the state associated with the node that finished
    // executing.
    (executionState->inflightNodes).erase(node);
    (executionState->nodeParentsDone).erase(node);
    (executionState->inputCtxs).erase(node);

    // Now, check if all nodes in the graph are done. If so, the callback can be
    // called and all state associated with the run can be erased.

    // The code at the top of this function acquires locksMtxLock, and then
    // executionStateLock, so they must be acquired in the same order to avoid
    // deadlock. Unlock executionStateLock and acquire the locksMtxLock.
    executionStateLock.unlock();
    locksMtxLock.lock();

    // Make sure the state for the run was not cleaned up while the
    // executionStateLock was unlocked.
    executionStateLockIt = executionStateLocks_.find(runId);
    if (executionStateLockIt == executionStateLocks_.end()) {
      // This means another thread snuck in while executionStateLock was
      // briefly unlocked and cleaned up the state for the run because
      // another node in the graph failed to execute. Do nothing.
      return;
    }

    // Lock executionStateLock before accessing the shared execution state.
    executionStateLock.lock();

    if ((executionState->inflightNodes).empty()) {
      // If there are no nodes inflight, that means all nodes are done. Call
      // the callback and erase the state information.
      executionState->cb(runId, ResultCode::EXECUTED,
                         std::move(executionState->resultCtx));
      executionStates_.erase(runId);
      executionStateLocks_.erase(runId);
    }
  }
}

} // namespace runtime

} // namespace glow
