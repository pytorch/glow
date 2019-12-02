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

using namespace glow;
using namespace glow::runtime;

ExecutionState::ExecutionState(RunIdentifierTy id, const DAGNode *root,
                               ThreadExecutor *executor,
                               std::unique_ptr<ExecutionContext> resultContext,
                               ResultCBTy doneCb)
    : runId_(id), cb_(doneCb), resultCtx_(std::move(resultContext)),
      inflightNodes_(0), module_(root->module), root_(root),
      executor_(executor) {
  DCHECK(cb_ != nullptr);
}

void ExecutionState::init() {
  // Create a queue for the breadth-first traversal through the graph.
  std::queue<const DAGNode *> bfsQueue;

  // Place the root nodes in the queue.
  for (const auto &node : root_->children) {
    bfsQueue.push(node);
  }

  auto *resultTraceContext = resultCtx_->getTraceContext();

  // Breadth-first search.
  while (!bfsQueue.empty()) {
    // Get the next node in the BFS queue.
    const DAGNode *node = bfsQueue.front();
    bfsQueue.pop();

    // Make a counter for the number of node parents done.
    nodeParentsDone_[node] = 0;

    // Make an (empty) input context for the node.
    auto nodeInputCtx = glow::make_unique<ExecutionContext>();

    if (resultTraceContext) {
      nodeInputCtx->setTraceContext(
          glow::make_unique<TraceContext>(resultTraceContext->getTraceLevel()));
    }

    auto nodeInputPhBindings = nodeInputCtx->getPlaceholderBindings();

    // Get the symbol table for the node.
    const SymbolTableTy &symbolTable = node->runtimeBundle->getSymbolTable();

    // Create Placeholders for the symbols of all intermediate nodes. These are
    // not in the ExecutionContext passed to Executor::run, so they must be
    // created by the Executor.
    auto *resultBindings = resultCtx_->getPlaceholderBindings();
    for (const auto &symbolPair : symbolTable) {
      const auto &symbolName = symbolPair.first;
      const auto &symbolInfo = symbolPair.second;

      if (symbolInfo.symbolCategory == SymbolCategory::Placeholder) {
        auto *PH = resultBindings->getPlaceholderByName(symbolName);
        if (!PH) {
          PH = module_->getPlaceholderByName(symbolName);
          DCHECK(PH) << "Placeholder: " << symbolName
                     << " is not in the module";
          // If PH is marked static skip it.
          if (PH->isStatic()) {
            continue;
          }
          // allocate into the resultBindings because they have the longest
          // lifetime.
          resultBindings->insert(PH,
                                 intermediateTensorPool_.get(PH->getType()));
          intermediatePlaceholders_.push_back(PH);
        }
        // Check that provided context does not contain a static PH.
        DCHECK(!PH->isStatic())
            << "Placeholder: " << symbolName
            << " is static and shouldn't be in Result Context.";

        nodeInputPhBindings->insert(
            PH, resultBindings->get(PH)->getUnowned(PH->dims()));
      }
    }

    // Insert the prepared ExecutionContext into the input contexts map.
    inputCtxs_.insert(std::make_pair(node, std::move(nodeInputCtx)));

    // Push all unvisited children onto the BFS queue.
    for (const auto &child : node->children) {
      // Use nodeParentsDone_ as a set of nodes that have been visited already
      // to avoid visiting a node more than once.
      if (!nodeParentsDone_.count(child)) {
        bfsQueue.push(child);
      }
    }
  }
  initialized_ = true;
}

std::unique_ptr<ExecutionContext>
ExecutionState::getUniqueNodeContextPtr(const DAGNode *node) {
  // The input PlaceholderBindings for the node should have been created in the
  // constructor.
  auto ctxIt = inputCtxs_.find(node);

  DCHECK(ctxIt != inputCtxs_.end())
      << "Input bindings not found but should exist!";

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
  DCHECK_GE(previousValue, decrement)
      << "More decrements than increments to inflight nodes!";

  // Return true when the counter hits zero.
  return (previousValue == decrement);
}

bool ExecutionState::incrementNodeParentsDone(const DAGNode *node,
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

void ExecutionState::insertIntoTraceContext(TraceContext *runCtx) {
  if (!resultCtx_->getTraceContext()) {
    return;
  }

  resultCtx_->getTraceContext()->merge(runCtx);
}

void ExecutionState::removeIntermediatePlaceholders() {
  for (auto &p : intermediatePlaceholders_) {
    resultCtx_->getPlaceholderBindings()->erase(p);
  }
  intermediatePlaceholders_.clear();
}

std::unique_ptr<ExecutionContext> ExecutionState::getUniqueResultContextPtr() {
  // The result PlaceholderBindings should have been been created in the
  // constructor.
  DCHECK_NOTNULL(resultCtx_.get());
  return std::move(resultCtx_);
}

ExecutionContext *ExecutionState::getRawResultContextPtr() const {
  // The result PlaceholderBindings should have been been created in the
  // constructor and should not yet have been moved out if this function is
  // being called.
  DCHECK_NOTNULL(resultCtx_.get());
  return resultCtx_.get();
}
