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

#include "ExecutionState.h"

#include "glow/Backends/DeviceManager.h"
#include "glow/ExecutionContext/ExecutionContext.h"
#include "glow/Runtime/Executor/ThreadPoolExecutor.h"

#include <queue>
#include <unordered_set>

#include "llvm/Support/FormatVariadic.h"
#include <glog/logging.h>

namespace glow {
namespace runtime {

void InflightBarrier::decrement(unsigned decr) {
  std::unique_lock<std::mutex> lock(mtx_);
  DCHECK_GE(count_, decr) << "Barrier decrement cannot be less than count!";
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

ThreadPoolExecutor::ThreadPoolExecutor(const DeviceManagerMapTy &deviceManagers,
                                       unsigned numWorkers,
                                       const std::string &name)
    : threadPool_(numWorkers,
                  std::make_shared<folly::NamedThreadFactory>(name)),
      deviceManagers_(deviceManagers) {}

void ThreadPoolExecutor::shutdown() {
  // Prevent more requests from being processed.
  shuttingDown_ = true;

  // Wait for all inflight DeviceManager::runFunction() calls to return and be
  // processed before starting to destroy state that is used in
  // handleDeviceManagerResult().
  inflightBarrier_.wait();

  threadPool_.stop();
  threadPool_.join();
}

void ThreadPoolExecutor::run(const DAGNode *root,
                             std::unique_ptr<ExecutionContext> context,
                             RunIdentifierTy runId, ResultCBTy cb) {
  DCHECK(cb != nullptr);

  TRACE_EVENT_SCOPE(context->getTraceContext(), TraceLevel::RUNTIME,
                    "ThreadPoolExecutor::run");

  if (context->getTraceContext()) {
    auto tid = threads::getThreadId();
    if (!context->getTraceContext()->getThreadNames().count(tid)) {
      context->getTraceContext()->setThreadName(tid, "ThreadPoolExecutor");
    }
  }

  // Don't process new requests if the executor is shutting down.
  if (shuttingDown_) {
    cb(runId,
       MAKE_ERR(ErrorValue::ErrorCode::RUNTIME_REQUEST_REFUSED,
                "ThreadPoolExecutor is shutting down"),
       std::move(context));
    return;
  }

  // If list of roots is empty, there is nothing to do. Give back the
  // bindings so the caller can reuse it.
  if (!root) {
    cb(runId, Error::success(), std::move(context));
    return;
  }

  auto numChildren = (root->children).size();
  // Mark the child nodes as "inflight" (i.e. currently executing). This must
  // be done here instead of inside executeDAGNode() so that a node can be
  // executed while placeholders are being propagated for the next node
  // without the callback for that node deleting the execution state.
  inflightBarrier_.increment(numChildren);

  auto *traceContext = context->getTraceContext();

  // Get and bind state.
  auto currentState = states_[root]->getNextNetworkExecutionState();
  TRACE_EVENT_BEGIN(traceContext, TraceLevel::RUNTIME,
                    "bind network execution state");
  currentState->bind(std::move(context), std::move(cb), runId);
  TRACE_EVENT_END(traceContext, TraceLevel::RUNTIME,
                  "bind network execution state");

  currentState->incrementInflightNodes(numChildren);

  // End the trace block before calling executeDAGNode() which can trigger the
  // result cb. Once the result cb is called, it's no longer safe to access the
  // trace context.
  TRACE_EVENT_SCOPE_END();
  for (auto const &node : root->children) {
    // Run with cached state
    executeDAGNode(currentState, node);
  }
}

void ThreadPoolExecutor::executeDAGNode(NetworkExecutionState *executionState,
                                        DAGNode *node) {
  TRACE_EVENT_SCOPE(executionState->getRawResultContextPtr()->getTraceContext(),
                    TraceLevel::RUNTIME, "ThreadPoolExecutor::executeDAGNode");
  if (executionState->getErrorContainer().containsErr()) {
    // Mark the node as no longer executing.
    executionState->decrementInflightNodes();
    inflightBarrier_.decrement();
    return;
  }

  // Get the PlaceholderBindings containing all of the inputs for the node.
  std::unique_ptr<ExecutionContext> nodeCtx =
      executionState->getUniqueNodeContextPtr(node);

  // Get the DeviceManager that can run the node.
  auto currentDevice = node->getNextDevice();
  auto deviceManagerIt = deviceManagers_.find(currentDevice);

  if (deviceManagerIt == deviceManagers_.end()) {
    // Mark the node as no longer executing.
    executionState->getErrorContainer().set(
        MAKE_ERR(ErrorValue::ErrorCode::RUNTIME_DEVICE_NOT_FOUND,
                 "Cannot find the DeviceManager specified."));
    executionState->decrementInflightNodes();
    inflightBarrier_.decrement();
    return;
  }
  DeviceManager *deviceManager = deviceManagerIt->second.get();
  // If the context has a deviceManager bound use that instead.
  if (nodeCtx->getBoundDeviceManager()) {
    deviceManager = nodeCtx->getBoundDeviceManager();
  }

  // End the trace block before calling deviceManager->runFunction which can
  // trigger the result cb in a different thread. Once the result cb is called,
  // it's no longer safe to access the trace context.
  TRACE_EVENT_SCOPE_END();
  // Run the node using the DeviceManager.
  deviceManager->runFunction(
      node->getNextName(currentDevice), std::move(nodeCtx),
      [this, executionState, currentDevice,
       node](RunIdentifierTy id, Error err,
             std::unique_ptr<ExecutionContext> resultCtx) {
        TRACE_EVENT_LOG_ID(resultCtx->getTraceContext(), TraceLevel::REQUEST,
                           "handle result queuing", TraceEvent::AsyncBeginType,
                           TraceEvent::now(), id);

        // Immediately move the handling of the result onto this run's executor
        // to avoid doing work on the DeviceManager thread.
        threadPool_.add([this, executionState, node, err = std::move(err),
                         currentDevice, id,
                         ctx = std::move(resultCtx)]() mutable {
          TRACE_EVENT_LOG_ID(ctx->getTraceContext(), TraceLevel::REQUEST,
                             "handle result queuing", TraceEvent::AsyncEndType,
                             TraceEvent::now(), id);

          node->markFinished(currentDevice);
          this->handleDeviceManagerResult(executionState, std::move(err),
                                          std::move(ctx), node);
        });
      });
}

void ThreadPoolExecutor::handleDeviceManagerResult(
    NetworkExecutionState *executionState, Error err,
    std::unique_ptr<ExecutionContext> ctx, const DAGNode *node) {
  TraceContext *traceContext = ctx->getTraceContext();
  if (traceContext) {
    TRACE_EVENT_BEGIN(traceContext, TraceLevel::RUNTIME,
                      "ThreadPoolExecutor::handleResult");
  }

  auto runWasSuccess = !err;

  // Set the result code for the run.
  executionState->getErrorContainer().set(std::move(err));

  // If the DeviceManager executed the node, propagate its output Placeholders
  // to its children or the result PlaceholderBindings as appropriate.
  if (runWasSuccess) {
    for (auto &child : node->children) {
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
  // Return intermediateContext to executionState.
  executionState->returnUniqueNodeContextPtr(node, std::move(ctx));

  // This needs to happen before decrementInflightNodes(). Otherwise a race
  // condition can happen where two threads call into this function at the same
  // time. Once decrementInflightNodes() is called, only the thread that get
  // noNodesInflight == true can access executionState.
  if (traceContext) {
    TRACE_EVENT_END(traceContext, TraceLevel::RUNTIME,
                    "ThreadPoolExecutor::handleResult");
    executionState->insertIntoTraceContext(traceContext);
  }

  // Now, check if all nodes in the graph are done. If so, the callback can be
  // called and all state associated with the run can be erased.
  bool noNodesInflight = executionState->decrementInflightNodes();

  if (noNodesInflight) {
    // If there are no nodes inflight, that means all nodes are done. Transfer
    // the outpus. Call the callback and erase the state information.
    // Because we are redirecting inputs and outputs to use the provided tensor
    // we do not have to transfer outputs here. Once we have pinned memory we
    // will transfer. //executionState->transferOutputs();
    ResultCBTy cb = executionState->getCallback();
    DCHECK(cb != nullptr);

    // Get what we need from the executionState and return it to the pool.
    auto runId = executionState->getRunId();
    auto err = executionState->getErrorContainer().get();
    auto resultCtx = executionState->getUniqueResultContextPtr();
    states_[executionState->getRoot()]->returnNetworkExecutionState(
        executionState);

    cb(runId, std::move(err), std::move(resultCtx));
  }

  // Decrement the inflight barrier for the executor keeping track of all
  // outstanding DeviceManager::runFunction() calls. This must be done here
  // instead of right after executionState->decrementInflightNodes() so that
  // ~ThreadPoolExecutor does not delete executor state before this function
  // is done using it (e.g. when erasing the ExecutionState object for a
  // run).
  inflightBarrier_.decrement();
}

void ThreadPoolExecutor::createPool(const DAGNode *root, unsigned poolSize,
                                    bool enableP2P, bool enableDRT) {
  std::unordered_map<DAGNode *, DeviceIDTy> assignment;

  // For static assignment we need to track devices each node is assigned to.
  std::unordered_map<DAGNode *, std::vector<DeviceIDTy>> assignments;
  std::unordered_map<DAGNode *, unsigned> currentAssignment;
  if (enableP2P || enableDRT) {
    // Walk the nodes and get assignments.
    std::queue<DAGNode *> remaining;
    for (auto node : root->children) {
      remaining.push(node);
    }
    while (remaining.size()) {
      auto node = remaining.front();
      remaining.pop();
      // Add any new children to the queue.
      for (auto child : node->children) {
        auto it = assignments.find(child);
        if (it == assignments.end()) {
          remaining.push(child);
        }
      }
      std::vector<DeviceIDTy> assignment;
      for (auto dev : node->deviceRuntimeInfos) {
        assignment.push_back(dev.first);
      }
      assignments[node] = assignment;
      currentAssignment[node] = 0;
    }
  }

  std::unique_ptr<NetworkExecutionStatePool> pool =
      glow::make_unique<NetworkExecutionStatePool>();
  for (unsigned i = 0; i < poolSize; i++) {
    auto newState = glow::make_unique<NetworkExecutionState>(root);
    // If assignStatic, calculate the device assignments for this
    // executionState. For now we are assigning a round robin pattern per node.
    if (enableDRT || enableP2P) {
      for (auto it : currentAssignment) {
        auto &nodeAssignments = assignments.at(it.first);
        auto newAssignmentIdx = (it.second + 1) % nodeAssignments.size();
        auto newAssignment = nodeAssignments[newAssignmentIdx];
        assignment[it.first] = newAssignment;
        currentAssignment[it.first] = newAssignmentIdx;
      }
    }
    newState->init(deviceManagers_, assignment);
    pool->addNewState(std::move(newState));
  }
  states_[root] = std::move(pool);
}

void ThreadPoolExecutor::freePool(const DAGNode *root) { states_.erase(root); }

} // namespace runtime
} // namespace glow
