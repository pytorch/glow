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
  const Graph &G_;
  /// Scheduled nodes.
  NodesList &Scheduled_;

public:
  Scheduler(const Graph &G, NodesList &Scheduled)
      : G_(G), Scheduled_(Scheduled) {}
  // Create a linear execution schedule for a graph.
  virtual void schedule() = 0;

  NodesList &getSchedule() { return Scheduled_; }
};

/// This is a scheduler based on the generalized the paper "Generalizations of
/// the Sethi-Ullman algorithm for register allocation" by Andrew W. Appel and
/// Kenneth J. Supowit.
/// http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.54.319&rep=rep1&type=pdf
///
/// The idea is to give more priority and schedule earlier those child nodes
/// that free more memory after their computation.
class ChildMemSizeBasedScheduler : Scheduler {
  /// Required number of bytes to hold the results of a given node.
  std::unordered_map<const Node *, int> resultMemSize_;
  /// Max number of bytes required during the computation of a given node.
  std::unordered_map<const Node *, int> maxMemSize_;

  /// \returns true if a node \p N is scheduled already.
  bool isScheduled(const Node *N) const {
    return std::find(Scheduled_.begin(), Scheduled_.end(), N) !=
           Scheduled_.end();
  }

  /// Computes the amount of memory required to keep the result
  /// of each node.
  void computeNodeResultsMemorySize() {
    for (auto *N : G_.getNodes()) {
      size_t ResultSize = 0;
      for (size_t idx = 0, e = N->getNumResults(); idx < e; ++idx) {
        ResultSize += N->getType(idx)->getSizeInBytes();
      }
      resultMemSize_[N] = ResultSize;
      DEBUG(llvm::outs() << "ResultSize of " << N->getName() << ":"
                         << ResultSize << "\n");
    }
  }

  /// Computes the max amount of memory required during the computation
  /// of children for each node.
  void computeNodeComputationMaxMemorySize() {
    // Traverse nodes in such a way, that dependnecies are processed
    // before the node using them.
    GraphPostOrderVisitor visitor(G_);
    for (auto *N : visitor.getPostOrder()) {
      size_t MaxSize = (N->getNumInputs() > 0)
                           ? std::max(resultMemSize_[N->getNthInput(0)],
                                      maxMemSize_[N->getNthInput(0)])
                           : 0;
      for (size_t idx = 1, e = N->getNumInputs(); idx < e; ++idx) {
        const auto &Input = N->getNthInput(idx);
        // Skip operands that do not require memory allocations for storing
        // their results.
        if (isa<Variable>(Input))
          continue;
        assert(resultMemSize_.count(Input) > 0);
        assert(maxMemSize_.count(Input) > 0);
        MaxSize += resultMemSize_[Input];
        if (MaxSize < maxMemSize_[Input])
          MaxSize = maxMemSize_[Input];
      }
      maxMemSize_[N] = MaxSize;
      DEBUG(llvm::outs() << "MaxSize of " << N->getName() << ":" << MaxSize
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
    llvm::SmallVector<Node *, 8> OrderedChildren;
    for (int idx = 0, e = N->getNumInputs(); idx < e; ++idx) {
      OrderedChildren.push_back(N->getNthInput(idx));
    }

    // Order children by (maxSize - resultSize). It gives more
    // priority to the nodes that free more memory after
    // their computation.
    for (int j = 0, e = OrderedChildren.size(); j < e; ++j) {
      for (int i = j; i > 0; --i) {
        auto &CurrentChild = OrderedChildren[i];
        auto &PrevChild = OrderedChildren[i - 1];
        if (maxMemSize_[CurrentChild] - resultMemSize_[CurrentChild] >
            maxMemSize_[PrevChild] - resultMemSize_[PrevChild]) {
          std::swap(CurrentChild, PrevChild);
        }
      }
    }

    DEBUG(llvm::outs() << "\nAbout to schedule children of " << N->getName()
                       << "\n";
          llvm::outs() << "Children are:\n");
    DEBUG(for (auto Child
               : OrderedChildren) {
      llvm::outs() << "Child " << Child->getName() << ": "
                   << maxMemSize_[Child] - resultMemSize_[Child] << "\n";
    });

    // Process the children according to the computed ordering.
    // TODO: This can be generalize to schedule on multiple devices
    // once it is supported.
    for (auto Child : OrderedChildren) {
      orderChildNodesAndSchedule(Child);
    }

    // Schedule the node after all its children are scheduled.
    DEBUG(llvm::outs() << "Scheduled node: " << N->getName() << "\n");
    Scheduled_.push_back(N);
  }

  void scheduleNodes() {
    /// Try to schedule all root nodes.
    for (auto *N : G_.getNodes()) {
      if (N->getNumUsers() == 0)
        orderChildNodesAndSchedule(N);
    }
  }

public:
  ChildMemSizeBasedScheduler(const Graph &G, NodesList &Schedule)
      : Scheduler(G, Schedule) {}

  void schedule() override {
    computeNodeResultsMemorySize();
    computeNodeComputationMaxMemorySize();
    scheduleNodes();
  }

  NodesList &getSchedule() { return Scheduled_; }
};

void Module::scheduleGraph(NodesList &Schedule) {
  Schedule.clear();
  for (auto &N : G_->getVars()) {
    Schedule.push_back(N);
  }
  ChildMemSizeBasedScheduler CWBScheduler(*G_, Schedule);
  CWBScheduler.schedule();
  assert(CWBScheduler.getSchedule().size() ==
             G_->getNodes().size() + G_->getVars().size() &&
         "All graph nodes have to be scheduled");
}

} // namespace glow
