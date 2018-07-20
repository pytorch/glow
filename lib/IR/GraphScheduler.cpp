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
#define DEBUG_TYPE "graph-scheduler"
#include "glow/Graph/Graph.h"
#include "glow/Graph/Nodes.h"
#include "glow/Graph/Utils.h"
#include "glow/IR/IR.h"
#include "glow/Support/Debug.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <unordered_map>
#include <unordered_set>

using namespace glow;

//===----------------------------------------------------------------------===//
//                               Graph scheduler
//===----------------------------------------------------------------------===//

namespace glow {

using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;

class Scheduler {
protected:
  /// Graph being processed.
  Function &G_;
  /// Scheduled nodes.
  NodesPtrList &scheduled_;

public:
  Scheduler(Function &G, NodesPtrList &scheduled)
      : G_(G), scheduled_(scheduled) {}

  virtual ~Scheduler() = default;

  // Create a linear execution schedule for a graph.
  virtual void schedule() = 0;

  NodesPtrList &getSchedule() { return scheduled_; }
};

/// This is a scheduler based on the generalized the paper "Generalizations of
/// the Sethi-Ullman algorithm for register allocation" by Andrew W. Appel and
/// Kenneth J. Supowit.
/// http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.54.319&rep=rep1&type=pdf
///
/// The idea is to give more priority and schedule earlier those child nodes
/// that free more memory after their computation.
class ChildMemSizeBasedScheduler : public Scheduler {
  /// Required number of bytes to hold the results of a given node.
  std::unordered_map<const Node *, size_t> resultMemSize_;
  /// Max number of bytes required during the computation of a given node.
  std::unordered_map<const Node *, size_t> maxMemSize_;

  /// \returns true if a node \p N is scheduled already.
  bool isScheduled(const Node *N) const {
    return std::find(scheduled_.begin(), scheduled_.end(), N) !=
           scheduled_.end();
  }

  /// Computes the amount of memory required to keep the result
  /// of each node.
  void computeNodeResultsMemorySize() {
    for (auto &N : G_.getNodes()) {
      size_t resultSize = 0;
      for (size_t idx = 0, e = N.getNumResults(); idx < e; ++idx) {
        resultSize += N.getType(idx)->getSizeInBytes();
      }
      resultMemSize_[&N] = resultSize;
      DEBUG_GLOW(llvm::outs() << "ResultSize of " << N.getName() << ":"
                              << resultSize << "\n");
    }
  }

  /// Computes the max amount of memory required during the computation
  /// of children for each node.
  void computeNodeComputationMaxMemorySize() {
    // Traverse nodes in such a way, that dependnecies are processed
    // before the node using them.
    GraphPostOrderVisitor visitor(G_);
    for (auto *N : visitor.getPostOrder()) {
      size_t maxSize = (N->getNumInputs() > 0)
                           ? std::max(resultMemSize_[N->getNthInput(0)],
                                      maxMemSize_[N->getNthInput(0)])
                           : 0;
      for (size_t idx = 1, e = N->getNumInputs(); idx < e; ++idx) {
        const auto &input = N->getNthInput(idx);
        // Skip operands that do not require memory allocations for storing
        // their results.
        if (isa<Variable>(input))
          continue;
        assert(resultMemSize_.count(input) > 0);
        assert(maxMemSize_.count(input) > 0);
        maxSize += resultMemSize_[input];
        if (maxSize < maxMemSize_[input])
          maxSize = maxMemSize_[input];
      }
      maxMemSize_[N] = maxSize;
      DEBUG_GLOW(llvm::outs()
                 << "MaxSize of " << N->getName() << ":" << maxSize << "\n");
    }
  }

  /// Order children by (maxSize - resultSize). It gives more
  /// priority to the nodes that free more memory after
  /// their computation.
  void orderChildNodesAndSchedule(Node *N) {
    // Each child should be scheduled just once.
    if (isScheduled(N))
      return;
    // Do not explicitly schedule variables.
    if (isa<Variable>(N))
      return;
    // A set of node's sorted children.
    llvm::SmallVector<Node *, 8> orderedChildren;
    for (int idx = 0, e = N->getNumInputs(); idx < e; ++idx) {
      orderedChildren.push_back(N->getNthInput(idx));
    }

    if (N->hasPredicate()) {
      orderedChildren.push_back(N->getPredicate());
    }

    // SaveNode hack:
    // We don't model memory dependencies, but we still need to honor them.
    // Make sure the SaveNode happens after the last use of the output variable.
    if (auto *save = dyn_cast<SaveNode>(N)) {
      Variable *output = save->getVariable();
      for (NodeUse &use : output->getUsers()) {
        Node *user = use.getUser();
        if (user == save) {
          continue;
        }
        // Variables may have users scattered across different functions.
        // Only accounts for the ones in that function.
        if (&G_ != user->getParent()) {
          continue;
        }
        assert(!isa<SaveNode>(user) &&
               "Variables must be saved at most once in each function");
        orderedChildren.push_back(user);
      }
    }

    // Order children by (maxSize - resultSize). It gives more
    // priority to the nodes that free more memory after
    // their computation.
    for (size_t j = 0, e = orderedChildren.size(); j < e; ++j) {
      for (size_t i = j; i > 0; --i) {
        auto &currentChild = orderedChildren[i];
        auto &prevChild = orderedChildren[i - 1];
        if (maxMemSize_[currentChild] - resultMemSize_[currentChild] >
            maxMemSize_[prevChild] - resultMemSize_[prevChild]) {
          std::swap(currentChild, prevChild);
        }
      }
    }

    DEBUG_GLOW(llvm::outs() << "\nAbout to schedule children of "
                            << N->getName() << "\n";
               llvm::outs() << "Children are:\n");
    DEBUG_GLOW(for (auto child
                    : orderedChildren) {
      llvm::outs() << "Child " << child->getName() << ": "
                   << maxMemSize_[child] - resultMemSize_[child] << "\n";
    });

    // Process the children according to the computed ordering.
    for (auto child : orderedChildren) {
      orderChildNodesAndSchedule(child);
    }

    // Schedule the node after all its children are scheduled.
    DEBUG_GLOW(llvm::outs() << "Scheduled node: " << N->getName() << "\n");
    scheduled_.push_back(N);
  }

  void scheduleNodes() {
    /// Try to schedule all root nodes.
    for (auto &N : G_.getNodes()) {
      if (N.getNumUsers() == 0)
        orderChildNodesAndSchedule(&N);
    }
  }

public:
  ChildMemSizeBasedScheduler(Function &G, NodesPtrList &Schedule)
      : Scheduler(G, Schedule) {}

  ~ChildMemSizeBasedScheduler() override = default;

  void schedule() override {
    computeNodeResultsMemorySize();
    computeNodeComputationMaxMemorySize();
    scheduleNodes();
  }
};

void IRFunction::scheduleGraph(NodesPtrList &Schedule) {
  Schedule.clear();
  for (auto &N : G_->getParent()->getVars()) {
    Schedule.push_back(N);
  }
  ChildMemSizeBasedScheduler CMSBScheduler(*G_, Schedule);
  CMSBScheduler.schedule();
  assert(CMSBScheduler.getSchedule().size() ==
             G_->getNodes().size() + G_->getParent()->getVars().size() &&
         "All graph nodes have to be scheduled");
}

} // namespace glow
