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
#include "glow/Runtime/Executor/NetworkExecutionState.h"
#include "glow/Backends/DeviceManager.h"

using namespace glow;
using namespace glow::runtime;

void NetworkExecutionStatePool::addNewState(
    std::unique_ptr<NetworkExecutionState> state) {

  std::lock_guard<std::mutex> lock(stateLock_);
  availableStates_.push_back(state.get());
  states_.push_back(std::move(state));
}

NetworkExecutionState::NetworkExecutionState(const DAGNode *root)
    : inflightNodes_(0), module_(root->module), root_(root) {}

NetworkExecutionState::~NetworkExecutionState() {
  // Free all allocated buffers.
  for (auto &allocation : deviceAllocations_) {
    allocation.second->freeAllocatedDeviceIOBuffer(allocation.first);
  }
}

void NetworkExecutionState::bind(std::unique_ptr<ExecutionContext> resultCtx,
                                 ResultCBTy cb, RunIdentifierTy runId) {
  resultCtx_ = std::move(resultCtx);
  cb_ = std::move(cb);
  runId_ = runId;
  // Reset execution state, inflight nodes, parents done, etc.
  for (auto &count : nodeParentsDone_) {
    count.second = 0;
  }
  inflightNodes_ = 0;
  // Setup tracing if desired.
  auto resultTraceContext = resultCtx_->getTraceContext();
  if (resultTraceContext) {
    for (auto &context : intermediateContexts_) {
      context.second->setTraceContext(
          glow::make_unique<TraceContext>(resultTraceContext->getTraceLevel()));
    }
  } else {
    // Clear any trace context from a previous run.
    for (auto &context : intermediateContexts_) {
      context.second->setTraceContext(nullptr);
    }
  }
  // Move inputs into tensors backing intermediate contexts.
  // Instead we point the tensors to the provided buffers to avoid copy in and
  // out. Once we have pinned allocations we will need to transfer.
  // For now point input and output tensors to buffers used in resultCtx.
  auto resultPHBindings = resultCtx_->getPlaceholderBindings();
  for (auto &pair : resultPHBindings->pairs()) {
    auto PH = pair.first;
    auto resultTensor = pair.second;
    for (auto binding : externalIntermediates_[PH]) {
      if (binding->get(PH)) {
        binding->update(PH, resultTensor->getUnowned());
      } else {
        binding->insert(PH, resultTensor->getUnowned());
      }
    }
  }
}

void NetworkExecutionState::init(
    const DeviceManagerMapTy &devices,
    std::unordered_map<DAGNode *, DeviceIDTy> &staticAssignment) {
  // Create a queue for the breadth-first traversal through the graph.
  std::queue<DAGNode *> bfsQueue;
  // Marking the default err as checked so we don't get an unchecked error in
  // destructor if we never use this state.
  errContainer_.containsErr();

  // Place the root nodes in the queue.
  for (auto &node : root_->children) {
    bfsQueue.push(node);
    // Make a counter for the number of node parents done. This also is used for
    // tracking if we've added the node already.
    nodeParentsDone_[node] = 0;
  }

  // Breadth-first search.
  while (!bfsQueue.empty()) {
    // Get the next node in the BFS queue.
    DAGNode *node = bfsQueue.front();
    bfsQueue.pop();

    // Push all unvisited children onto the BFS queue.
    for (const auto &child : node->children) {
      // Use nodeParentsDone_ as a set of nodes that have been visited already
      // to avoid visiting a node more than once.
      if (!nodeParentsDone_.count(child)) {
        nodeParentsDone_[child] = 0;
        bfsQueue.push(child);
      }
    }

    // Make an (empty) context for the node.
    auto intermediateContext = glow::make_unique<ExecutionContext>();
    auto it = staticAssignment.find(node);
    // If an assignment is provided for this context set it here.
    if (it != staticAssignment.end()) {
      auto dm = devices.find(it->second)->second.get();
      intermediateContext->setBoundDeviceManager(dm);
    }
    // Get a device to do allocation we can use the first device since the
    // allocation is not device specific.
    auto &device = devices.begin()->second;

    auto intermediatePHBindings = intermediateContext->getPlaceholderBindings();

    // Get the symbol table for the node.
    const SymbolTableTy &symbolTable = node->runtimeBundle->getSymbolTable();

    // Add inputs/outputs to the context. Skip any marked as static.
    for (const auto &symbolPair : symbolTable) {
      const auto &symbolName = symbolPair.first;
      const auto &symbolInfo = symbolPair.second;
      if (symbolInfo.symbolCategory == SymbolCategory::Placeholder) {
        auto PH = module_->getPlaceholderByName(symbolName);

        DCHECK(PH) << "Placeholder: " << symbolName << " is not in the module";
        // If PH is marked static skip it.
        if (PH->isStatic()) {
          continue;
        }
        // If we haven't allocated a buffer for this PH yet do so, otherwise
        // reuse the allocation.
        auto bufferIt = buffers_.find(PH);
        if (bufferIt == buffers_.end()) {

          buffers_[PH] =
              device->allocateDeviceIOBuffer(PH->getType()->getSizeInBytes());
        }
        auto buffer = buffers_[PH];
        Tensor *backingTensor = new Tensor(buffer, PH->getType());
        intermediatePHBindings->insert(PH, backingTensor);
        externalIntermediates_[PH].push_back(intermediatePHBindings);
      }
    }

    // Insert the prepared ExecutionContext into the input contexts map.
    intermediateContexts_.emplace(node, std::move(intermediateContext));
  }
  // If we used a static assignment call backend->bindContexts() on the new
  // contexts.
  if (staticAssignment.size()) {
    std::vector<runtime::ContextBinding> contexts;
    for (auto &intermediate : intermediateContexts_) {
      runtime::ContextBinding intermediateBinding;
      intermediateBinding.context = intermediate.second.get();
      intermediateBinding.networkName = intermediate.first->name;
      intermediateBinding.device = intermediate.second->getBoundDeviceManager();
      contexts.push_back(intermediateBinding);
    }
    const auto &backendName = devices.begin()->second->getBackendName();
    // Create a backend to call bindContexts on, since bindContexts only puts
    // state in the DeviceManager and Context we can safely discard this backend
    // once we are done with it.
    std::unique_ptr<Backend> newBackend(createBackend(backendName));

    EXIT_ON_ERR(newBackend->bindContexts(contexts, root_, /*enableP2P*/ true,
                                         /*enableDRT*/ true));
  }
  initialized_ = true;
}

std::unique_ptr<ExecutionContext>
NetworkExecutionState::getUniqueNodeContextPtr(const DAGNode *node) {
  // The input PlaceholderBindings for the node should have been created in
  // the constructor.
  auto ctxIt = intermediateContexts_.find(node);

  DCHECK(ctxIt != intermediateContexts_.end())
      << "Input bindings not found but should exist!";

  return std::move(ctxIt->second);
}

void NetworkExecutionState::returnUniqueNodeContextPtr(
    const DAGNode *node, std::unique_ptr<ExecutionContext> ctx) {
  intermediateContexts_[node] = std::move(ctx);
}

void NetworkExecutionState::incrementInflightNodes(unsigned increment) {
  inflightNodes_ += increment;
}

bool NetworkExecutionState::decrementInflightNodes(unsigned decrement) {
  // fetch_sub must be used here so that the function returns true to only one
  // caller.
  unsigned previousValue = inflightNodes_.fetch_sub(decrement);

  // The decrement should never be more than the value of the counter at the
  // time of decrement.
  DCHECK_GE(previousValue, decrement)
      << "More decrements than increments to inflight nodes!";

  // Return true when the counter hits zero.
  return (previousValue == decrement);
}

bool NetworkExecutionState::incrementNodeParentsDone(const DAGNode *node,
                                                     unsigned increment) {
  // Get the parents done counter for the node. It should have
  // been created in the constructor.
  auto it = nodeParentsDone_.find(node);

  DCHECK(it != nodeParentsDone_.end())
      << "Node parents done counter should exist but not found!";

  // fetch_add must be used here so that the function returns true to only
  // one caller.
  unsigned numParents = (node->parents).size();
  unsigned previousValue = (it->second).fetch_add(increment);
  unsigned newValue = previousValue + increment;

  // The new value of the counter cannot exceed the number of parents that
  // the node has.
  DCHECK_LE(newValue, numParents)
      << "Node parents done counter incremented beyond limit!";

  // Return true only when the counter hits the total numer of parents.
  return (newValue == numParents);
}

void NetworkExecutionState::insertIntoTraceContext(TraceContext *runCtx) {
  if (!resultCtx_->getTraceContext()) {
    return;
  }

  resultCtx_->getTraceContext()->merge(runCtx);
}

std::unique_ptr<ExecutionContext>
NetworkExecutionState::getUniqueResultContextPtr() {
  // The result PlaceholderBindings should have been been created in the
  // constructor.
  DCHECK_NOTNULL(resultCtx_.get());
  return std::move(resultCtx_);
}

ExecutionContext *NetworkExecutionState::getRawResultContextPtr() const {
  // The result PlaceholderBindings should have been been created in the
  // constructor and should not yet have been moved out if this function is
  // being called.
  DCHECK_NOTNULL(resultCtx_.get());
  return resultCtx_.get();
}
