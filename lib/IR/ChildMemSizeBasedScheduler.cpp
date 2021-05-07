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

#include "GraphScheduler.h"

#include "glow/Graph/Utils.h"
#include "glow/Support/Debug.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "graph-scheduler"

using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;

namespace glow {
/// \returns true if a node \p N is scheduled already.
bool ChildMemSizeBasedScheduler::isScheduled(const Node *N) const {
  return std::find(scheduled_.begin(), scheduled_.end(), N) != scheduled_.end();
}

/// Computes the amount of memory required to keep the result
/// of each node.
void ChildMemSizeBasedScheduler::computeNodeResultsMemorySize() {
  for (auto &N : G_.getNodes()) {
    int64_t resultSize = 0;
    for (size_t idx = 0, e = N.getNumResults(); idx < e; ++idx) {
      resultSize += N.getType(idx)->getSizeInBytes();
    }
    resultMemSize_[&N] = resultSize;
    DEBUG_GLOW(llvm::dbgs()
               << "ResultSize of " << N.getName() << ":" << resultSize << "\n");
  }
}

/// Computes the max amount of memory required during the computation
/// of children for each node.
void ChildMemSizeBasedScheduler::computeNodeComputationMaxMemorySize() {
  // Traverse nodes in such a way, that dependnecies are processed
  // before the node using them.
  GraphPostOrderVisitor visitor(G_);
  for (auto *N : visitor.getPostOrder()) {
    int64_t maxSize = (N->getNumInputs() > 0)
                          ? std::max(resultMemSize_[N->getNthInput(0)],
                                     maxMemSize_[N->getNthInput(0)])
                          : 0;
    for (size_t idx = 1, e = N->getNumInputs(); idx < e; ++idx) {
      const auto &input = N->getNthInput(idx);
      // Skip operands that do not require memory allocations for storing
      // their results.
      if (isa<Storage>(input))
        continue;
      assert(resultMemSize_.count(input) > 0);
      assert(maxMemSize_.count(input) > 0);
      maxSize += resultMemSize_[input];
      if (maxSize < maxMemSize_[input])
        maxSize = maxMemSize_[input];
    }
    maxMemSize_[N] = maxSize;
    DEBUG_GLOW(llvm::dbgs()
               << "MaxSize of " << N->getName() << ":" << maxSize << "\n");
  }
}

/// Order children by (maxSize - resultSize). It gives more
/// priority to the nodes that free more memory after
/// their computation.
void ChildMemSizeBasedScheduler::orderChildNodesAndSchedule(Node *node) {
  // Use `worklist` as a stack-like container to hold nodes for processing. Each
  // node is going to appear in the container at least twice:
  // 1) To put its children (and sometimes parents) in the worklist
  // 2) To schedule the node itself (i.e. append it to `scheduled_` list)
  // At the end of stage #1, the node is added in `readyNodes` set. By checking
  // this set the algorithm figures out if it should perform stage #1 or #2 with
  // the node it pulls out of the worklist.
  // Since the algorithm does some special handling of node's parents, a node
  // can get into the worklist more than twice. To ensure that stage #2 happens
  // only after all children of the node are scheduled, `readyNodes` keeps
  // node's worklist index instead of just node's pointer.
  std::vector<Node *> worklist;
  std::unordered_set<size_t> readyNodes;
  worklist.push_back(node);

  while (worklist.size()) {
    auto *N = worklist.back();
    worklist.pop_back();
    size_t idx = worklist.size();

    // Do not explicitly schedule storage nodes.
    if (isa<Storage>(N)) {
      continue;
    }
    // Each child should be scheduled just once.
    if (isScheduled(N)) {
      readyNodes.erase(idx);
      continue;
    }

    // If node is marked as ready, all its children have been scheduled, so it
    // can be scheduled now too.
    if (readyNodes.count(idx)) {
      readyNodes.erase(idx);
      scheduled_.push_back(N);
      continue;
    }

    // If this node has a user which does not have any users and which does not
    // require any additional memory, schedule it here, because we don't want to
    // extend the lifetime of this value for no reason. We want to execute and
    // get rid of this node as soon as possible to reduce the memory pressure.
    for (NodeUse &use : N->getUsers()) {
      Node *user = use.getUser();
      // Users may be scattered across different functions.
      // Only accounts for the ones in that function.
      if (&G_ != user->getParent()) {
        continue;
      }
      // Bail if a nodes has users, because nodes that have users can't be
      // scheduled safely without violating dependencies.
      if (user->getNumUsers()) {
        continue;
      }
      // Schedule a node if it does not require any additional memory.
      if (resultMemSize_[user] == 0) {
        worklist.push_back(user);
      }
    }

    // Push the node again in the stack. By the time it's going to pop up again,
    // all children are going to be scheduled, so mark it as ready.
    readyNodes.insert(worklist.size());
    worklist.push_back(N);

    // Take care about node's children.
    size_t childrenStartIdx = worklist.size();
    for (int i = 0, e = N->getNumInputs(); i < e; ++i) {
      worklist.push_back(N->getNthInput(i));
    }

    if (N->hasPredicate()) {
      worklist.push_back(N->getPredicate());
    }

    // We don't model memory dependencies, but we still need to honor them.
    // Make sure the a node mutating any of its inputs happens after the last
    // non-mutating use of the operand being mutated. Some examples of such
    // nodes would be SaveNode and QuantizationProfileNode.
    for (unsigned i = 0, e = N->getNumInputs(); i < e; ++i) {
      // We don't care about inputs that are not mutated by the node.
      if (!N->isOverwrittenNthInput(i)) {
        continue;
      }
      auto mutatedInput = N->getNthInput(i);
      auto *destination = mutatedInput.getNode();
      for (NodeUse &use : destination->getUsers()) {
        Node *user = use.getUser();
        if (user == N) {
          continue;
        }
        // Nodes may have users scattered across different functions.
        // Only accounts for the ones in that function.
        if (&G_ != user->getParent()) {
          continue;
        }
        worklist.push_back(user);
      }
    }

    // Order children by (maxSize - resultSize). It gives more
    // priority to the nodes that free more memory after
    // their computation.
    for (size_t j = childrenStartIdx, e = worklist.size(); j < e; ++j) {
      for (size_t i = j; i > childrenStartIdx; --i) {
        auto &currentChild = worklist[i];
        auto &prevChild = worklist[i - 1];
        if (maxMemSize_[currentChild] - resultMemSize_[currentChild] <=
            maxMemSize_[prevChild] - resultMemSize_[prevChild]) {
          std::swap(currentChild, prevChild);
        }
      }
    }

    DEBUG_GLOW(llvm::dbgs() << "\nAbout to schedule children of "
                            << N->getName() << "\n";
               llvm::dbgs() << "Children are:\n");
    DEBUG_GLOW(
        for (size_t i = childrenStartIdx, e = worklist.size(); i < e; ++i) {
          auto &child = worklist[i];
          llvm::dbgs() << "Child " << child->getName() << ": "
                       << maxMemSize_[child] - resultMemSize_[child] << "\n";
        });
  }
}

void ChildMemSizeBasedScheduler::scheduleNodes() {
  /// Try to schedule all root nodes.
  for (auto &N : G_.getNodes()) {
    if (N.getNumUsers() == 0)
      orderChildNodesAndSchedule(&N);
  }
}

void ChildMemSizeBasedScheduler::schedule() {
  computeNodeResultsMemorySize();
  computeNodeComputationMaxMemorySize();
  scheduleNodes();
}
} // namespace glow
