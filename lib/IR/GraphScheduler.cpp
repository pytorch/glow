// Copyright 2017 Facebook Inc.  All Rights Reserved.
#define DEBUG_TYPE "graph-scheduler"
#include "glow/Graph/Graph.h"
#include "glow/Graph/Nodes.h"
#include "glow/Graph/Utils.h"
#include "glow/IR/IR.h"

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
  const Function &G_;
  /// Scheduled nodes.
  NodesList &scheduled_;

public:
  Scheduler(const Function &G, NodesList &scheduled)
      : G_(G), scheduled_(scheduled) {}
  // Create a linear execution schedule for a graph.
  virtual void schedule() = 0;

  NodesList &getSchedule() { return scheduled_; }
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
    for (auto *N : G_.getNodes()) {
      size_t resultSize = 0;
      for (size_t idx = 0, e = N->getNumResults(); idx < e; ++idx) {
        resultSize += N->getType(idx)->getSizeInBytes();
      }
      resultMemSize_[N] = resultSize;
      DEBUG(llvm::outs() << "ResultSize of " << N->getName() << ":"
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
      DEBUG(llvm::outs() << "MaxSize of " << N->getName() << ":" << maxSize
                         << "\n");
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

    DEBUG(llvm::outs() << "\nAbout to schedule children of " << N->getName()
                       << "\n";
          llvm::outs() << "Children are:\n");
    DEBUG(for (auto child
               : orderedChildren) {
      llvm::outs() << "Child " << child->getName() << ": "
                   << maxMemSize_[child] - resultMemSize_[child] << "\n";
    });

    // Process the children according to the computed ordering.
    // TODO: This can be generalize to schedule on multiple devices
    // once it is supported.
    for (auto child : orderedChildren) {
      orderChildNodesAndSchedule(child);
    }

    // Schedule the node after all its children are scheduled.
    DEBUG(llvm::outs() << "Scheduled node: " << N->getName() << "\n");
    scheduled_.push_back(N);
  }

  void scheduleNodes() {
    /// Try to schedule all root nodes.
    for (auto *N : G_.getNodes()) {
      if (N->getNumUsers() == 0)
        orderChildNodesAndSchedule(N);
    }
  }

public:
  ChildMemSizeBasedScheduler(const Function &G, NodesList &Schedule)
      : Scheduler(G, Schedule) {}

  void schedule() override {
    computeNodeResultsMemorySize();
    computeNodeComputationMaxMemorySize();
    scheduleNodes();
  }
};

void IRFunction::scheduleGraph(NodesList &Schedule) {
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
