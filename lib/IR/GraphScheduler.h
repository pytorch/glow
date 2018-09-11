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
#ifndef GLOW_IR_GRAPH_SCHEDULER_H
#define GLOW_IR_GRAPH_SCHEDULER_H

#include "glow/IR/IR.h"

#include <unordered_map>

namespace glow {
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
  bool isScheduled(const Node *N) const;

  /// Computes the amount of memory required to keep the result
  /// of each node.
  void computeNodeResultsMemorySize();

  /// Computes the max amount of memory required during the computation
  /// of children for each node.
  void computeNodeComputationMaxMemorySize();

  /// Order children by (maxSize - resultSize). It gives more
  /// priority to the nodes that free more memory after
  /// their computation.
  void orderChildNodesAndSchedule(Node *N);

  void scheduleNodes();

public:
  ChildMemSizeBasedScheduler(Function &G, NodesPtrList &Schedule)
      : Scheduler(G, Schedule) {}

  ~ChildMemSizeBasedScheduler() override = default;

  void schedule() override;
};

} // namespace glow

#endif // GLOW_IR_GRAPH_SCHEDULER_H
