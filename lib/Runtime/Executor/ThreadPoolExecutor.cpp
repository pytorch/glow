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
#include "glow/Backends/ExecutionContext.h"

#include <queue>
#include <unordered_set>

namespace glow {
namespace runtime {

void InflightBarrier::decrement(unsigned decr) {
  std::unique_lock<std::mutex> lock(mtx_);
  assert(count_ >= decr && "Barrier decrement cannot be less than count!");
  count_ -= decr;

  // If count_ has hit zero, wake up all threads that are waiting.
  if (count_ == 0) {
    cv_.notify_all();
  }
} // namespace runtime

void InflightBarrier::increment(unsigned incr) {
  std::unique_lock<std::mutex> lock(mtx_);
  count_ += incr;
}

unsigned InflightBarrier::count() {
  std::unique_lock<std::mutex> lock(mtx_);
  return count_;
}

void InflightBarrier::wait() {
  std::unique_lock<std::mutex> lock(mtx_);
  // If count_ is not 0, wait until a signal is received that it is.
  // The second argument below is a predicate that returns true when
  // it is safe to wake up. It preserves correctness in the case of
  // spurious wakeups.
  cv_.wait(lock, [&] { return count_ == 0; });
}

ExecutionState::ExecutionState(RunIdentifierTy id, const DAGNode *root,
                               std::unique_ptr<ExecutionContext> resultContext,
                               ResultCBTy doneCb)
    : runId_(id), cb_(doneCb), resultCtx_(std::move(resultContext)),
      inflightNodes_(0) {
  // Create a queue for the breadth-first traversal through the graph.
  std::queue<const DAGNode *> bfsQueue;

  // Place the root nodes in the queue.
  for (const auto &node : root->children) {
    bfsQueue.push(node);
  }

  // Breadth-first search.
  while (!bfsQueue.empty()) {
    // Get the next node in the BFS queue.
    const DAGNode *node = bfsQueue.front();
    bfsQueue.pop();

    // Make a counter for the number of node parents done.
    nodeParentsDone_[node] = 0;

    // Make an (empty) input context for the node.
    inputCtxs_.insert(
        std::make_pair(node, llvm::make_unique<ExecutionContext>()));

    // Push all unvisited children onto the BFS queue.
    for (const auto &child : node->children) {
      // Use nodeParentsDone_ as a set of nodes that have been visited already
      // to avoid visiting a node more than once.
      if (!nodeParentsDone_.count(child)) {
        bfsQueue.push(child);
      }
    }
  }
}

void ExecutionState::insertIntoNodeCtx(const DAGNode *node, Placeholder *P,
                                       Tensor &&T) {
  // Get a raw pointer to the input ExecutionContext for the node. It should
  // have been created in the constructor.
  auto ctxIt = inputCtxs_.find(node);

  if (ctxIt == inputCtxs_.end()) {
    assert(!"Input bindings not found but should exist!");
  }

  PlaceholderBindings *bindings = (ctxIt->second)->getPlaceholderBindings();
  assert(bindings && "Input bindings for node is null");

  // Insert the placeholder-tensor pair.
  std::lock_guard<std::mutex> lock(bindingsMtx_);
  bindings->insert(P, std::move(T));
}

std::unique_ptr<ExecutionContext>
ExecutionState::getUniqueNodeContextPtr(const DAGNode *node) {
  // The input PlaceholderBindings for the node should have been created in the
  // constructor.
  auto ctxIt = inputCtxs_.find(node);

  if (ctxIt == inputCtxs_.end()) {
    assert(!"Input bindings not found but should exist!");
  }

  return std::move(ctxIt->second);
}

void ExecutionState::incrementInflightNodes(unsigned increment) {
  inflightNodes_ += increment;
}

bool ExecutionState::decrementInflightNodes(unsigned decrement) {
  // fetch_sub must be used here so that the function returns true to only one
  // caller.
  unsigned previousValue = inflightNodes_.fetch_sub(decrement);

  // The decrement should never be more than the value of the counter at the
  // time of decrement.
  if (previousValue < decrement) {
    assert(!"More decrements than increments to inflight nodes!");
  }

  // Return true when the counter hits zero.
  return (previousValue == decrement);
}

bool ExecutionState::incrementNodeParentsDone(const DAGNode *node,
                                              unsigned increment) {
  // Get the parents done counter for the node. It should have
  // been created in the constructor.
  auto it = nodeParentsDone_.find(node);

  if (it == nodeParentsDone_.end()) {
    assert(!"Node parents done counter should exist but not found!");
  }

  // fetch_add must be used here so that the function returns true to only
  // one caller.
  unsigned numParents = (node->parents).size();
  unsigned previousValue = (it->second).fetch_add(increment);
  unsigned newValue = previousValue + increment;

  // The new value of the counter cannot exceed the number of parents that
  // the node has.
  if (newValue > numParents) {
    assert(!"Node parents done counter incremented beyond limit!");
  }

  // Return true only when the counter hits the total numer of parents.
  return (newValue == numParents);
}

void ExecutionState::insertIntoResultCtx(Placeholder *P, Tensor &&T) {
  // The result PlaceholderBindings should have been been created in the
  // constructor and should not yet have been moved out if this function is
  // being called.
  assert(resultCtx_ && resultCtx_->getPlaceholderBindings() &&
         "Execution result bindings should exist!");
  std::lock_guard<std::mutex> lock(bindingsMtx_);
  Tensor *tensor = resultCtx_->getPlaceholderBindings()->get(P);

  if (tensor) {
    *tensor = std::move(T);
  } else {
    resultCtx_->getPlaceholderBindings()->insert(P, std::move(T));
  }
}

std::unique_ptr<ExecutionContext> ExecutionState::getUniqueResultContextPtr() {
  // The result PlaceholderBindings should have been been created in the
  // constructor.
  assert(resultCtx_ && "Execution result bindings should exist!");
  return std::move(resultCtx_);
}

ExecutionContext *ExecutionState::getRawResultContextPtr() const {
  // The result PlaceholderBindings should have been been created in the
  // constructor and should not yet have been moved out if this function is
  // being called.
  assert(resultCtx_ && "Execution result bindings should exist!");
  return resultCtx_.get();
}

void ThreadPoolExecutor::shutdown() {
  // Prevent more requests from being processed.
  shuttingDown_ = true;

  // Wait for all inflight DeviceManager::runFunction() calls to return and be
  // processed before starting to destroy state that is used in
  // handleDeviceManagerResult().
  inflightBarrier_.wait();
}

void ThreadPoolExecutor::run(const DAGNode *root,
                             std::unique_ptr<ExecutionContext> context,
                             RunIdentifierTy runId, ResultCBTy cb) {
  // Don't process new requests if the executor is shutting down.
  if (shuttingDown_) {
    cb(runId,
       MAKE_ERR(GlowErr::ErrorCode::RUNTIME_REQUEST_REFUSED,
                "ThreadPoolExecutor is shutting down"),
       std::move(context));
    return;
  }

  // If list of roots is empty, there is nothing to do. Give back the
  // bindings so the caller can reuse it.
  if (!root) {
    cb(runId, llvm::Error::success(), std::move(context));
    return;
  }

  std::shared_ptr<ExecutionState> executionState = nullptr;
  {
    std::lock_guard<std::mutex> lock(executionStatesMutex_);

    // If the given run ID corresponds to a run already in progress, there is
    // also nothing to do, but return an error. Give back the bindings so the
    // caller can reuse it.
    if (executionStates_.find(runId) != executionStates_.end()) {
      cb(runId,
         MAKE_ERR(
             GlowErr::ErrorCode::RUNTIME_REQUEST_REFUSED,
             "ThreadPoolExecutor found another run with the same request id"),
         std::move(context));
      return;
    }

    // Otherwise, create execution state tracker object for this run ID.
    executionState = std::make_shared<ExecutionState>(
        runId, root, std::move(context), std::move(cb));
    executionStates_.insert(std::make_pair(runId, executionState));
  }

  // Execute all child nodes of root.

  // Mark the child nodes as "inflight" (i.e. currently executing). This must be
  // done here instead of inside executeDAGNode() so that a node can be
  // executed while placeholders are being propagated for the next node without
  // the callback for that node deleting the execution state.
  auto numChildren = (root->children).size();
  executionState->incrementInflightNodes(numChildren);
  inflightBarrier_.increment(numChildren);

  for (auto const &node : root->children) {
    // Propagate placeholders from the given starter PlaceholderBindings into
    // the input PlaceholderBindings for the current node being processed.
    propagatePlaceholdersForNode(executionState, node,
                                 executionState->getRawResultContextPtr());

    // Execute the node.
    executeDAGNode(executionState, node);
  }
}

void ThreadPoolExecutor::propagatePlaceholdersForNode(
    std::shared_ptr<ExecutionState> executionState, const DAGNode *node,
    const ExecutionContext *ctx) {
  // Get the symbol table for the node.
  const SymbolTableTy &symbolTable = node->runtimeBundle->getSymbolTable();

  for (const auto &symbolPair : symbolTable) {
    const auto &symbolName = symbolPair.first;
    const auto &symbolInfo = symbolPair.second;

    if (symbolInfo.input) {
      // If the symbol is an input, look for a Placeholder in ctx with
      // the same name and copy the corresponding Tensor into the input
      // PlaceholderBindings being prepared for the node.
      auto *placeholder =
          ctx->getPlaceholderBindings()->getPlaceholderByName(symbolName);

      if (placeholder) {
        const auto *tensor = ctx->getPlaceholderBindings()->get(placeholder);
        executionState->insertIntoNodeCtx(node, placeholder, tensor->clone());
      }
    }
  }
}

void ThreadPoolExecutor::executeDAGNode(
    std::shared_ptr<ExecutionState> executionState, const DAGNode *node) {
  // If execution has already failed due to another node, don't bother running
  // this one.
  if (executionState->getErrorContainer().containsErr()) {
    // Mark the node as no longer executing.
    executionState->decrementInflightNodes();
    inflightBarrier_.decrement();
    return;
  }

  // Get the DeviceManager that can run the node.
  auto deviceManagerIt = deviceManagers_.find(node->deviceID);

  if (deviceManagerIt == deviceManagers_.end()) {
    // Mark the node as no longer executing.
    executionState->getErrorContainer().set(
        MAKE_ERR(GlowErr::ErrorCode::RUNTIME_DEVICE_NOT_FOUND,
                 "Cannot find the DeviceManager specified."));
    executionState->decrementInflightNodes();
    inflightBarrier_.decrement();
    return;
  }

  auto &deviceManager = deviceManagerIt->second;

  // Get the PlaceholderBindings containing all of the inputs for the node.
  std::unique_ptr<ExecutionContext> nodeCtx =
      executionState->getUniqueNodeContextPtr(node);
  // Run the node using the DeviceManager.
  deviceManager->runFunction(
      node->name, std::move(nodeCtx),
      [this, executionState,
       node](RunIdentifierTy id, llvm::Error err,
             std::unique_ptr<ExecutionContext> resultCtx) {
        // Immediately move the handling of the result onto threadPool_ to
        // avoid doing work on the DeviceManager thread.
        this->threadPool_.submit([this, executionState, node,
                                  err = std::move(err),
                                  ctx = std::move(resultCtx)]() mutable {
          this->handleDeviceManagerResult(executionState, std::move(err),
                                          std::move(ctx), node);
        });
      });
}

void ThreadPoolExecutor::propagateOutputPlaceholders(
    std::shared_ptr<ExecutionState> executionState,
    std::unique_ptr<ExecutionContext> ctx) {
  // Copy all of the Placeholders in ctx into the result PlaceholderBindings for
  // the run.
  for (const auto &phTensorPair : ctx->getPlaceholderBindings()->pairs()) {
    auto *placeholder = phTensorPair.first;
    auto *tensor = phTensorPair.second;

    executionState->insertIntoResultCtx(placeholder, std::move(*tensor));
  }
}

void ThreadPoolExecutor::handleDeviceManagerResult(
    std::shared_ptr<ExecutionState> executionState, llvm::Error err,
    std::unique_ptr<ExecutionContext> ctx, const DAGNode *node) {
  // If executionState is null, that means that the object was deleted
  // while a node was executing. That should never happen.
  assert(executionState && "Execution state should not be null");

  auto runWasSuccess = !err;

  // Set the result code for the run.
  executionState->getErrorContainer().set(std::move(err));

  // If the DeviceManager executed the node, propagate its output Placeholders
  // to its children or the result PlaceholderBindings as appropriate.
  if (runWasSuccess) {
    if ((node->children).empty()) {
      // If the node has no children, propagate its outputs to the result
      // PlaceholderBindings for the run.
      propagateOutputPlaceholders(executionState, std::move(ctx));
    } else {
      // If the node has children, propagate its outputs to the input
      // PlaceholderBindings for any of its children that need them as inputs.
      for (auto &child : node->children) {
        propagatePlaceholdersForNode(executionState, child, ctx.get());

        // Execute any child that has no parent nodes left to execute.
        bool childReadyToExecute =
            executionState->incrementNodeParentsDone(child);
        if (childReadyToExecute) {
          // Mark the node as "inflight" (i.e. currently executing).
          executionState->incrementInflightNodes();
          inflightBarrier_.increment();
          executeDAGNode(executionState, child);
        }
      }
    }
  }

  // Now, check if all nodes in the graph are done. If so, the callback can be
  // called and all state associated with the run can be erased.
  bool noNodesInflight = executionState->decrementInflightNodes();

  if (noNodesInflight) {
    // If there are no nodes inflight, that means all nodes are done. Call
    // the callback and erase the state information.
    ResultCBTy cb = executionState->getCallback();
    cb(executionState->getRunId(), executionState->getErrorContainer().get(),
       executionState->getUniqueResultContextPtr());

    // Clean up the state stored for the run.
    std::lock_guard<std::mutex> lock(executionStatesMutex_);
    executionStates_.erase(executionState->getRunId());
  }

  // Decrement the inflight barrier for the executor keeping track of all
  // outstanding DeviceManager::runFunction() calls. This must be done here
  // instead of right after executionState->decrementInflightNodes() so that
  // ~ThreadPoolExecutor does not delete executor state before this function
  // is done using it (e.g. when erasing the ExecutionState object for a
  // run).
  inflightBarrier_.decrement();
}

} // namespace runtime
} // namespace glow
