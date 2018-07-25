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

#include "glow/Optimizer/Partition.h"

#include "glow/Graph/Graph.h"
#include "glow/Graph/Utils.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Casting.h"

using namespace glow;
using llvm::isa;

namespace {

/// Helper structure for building a partition. Records a mapping of nodes in the
/// original function to destination partitions, along with a list of the
/// newly-created functions.
class NodeFunctionMap {
  using Map = llvm::DenseMap<Node *, Function *>;

  /// Newly-created partitions.
  FunctionList functions_;

  /// Map of nodes in the original function to their target partition.
  Map nodeToFunction_;

public:
  /// Create a new partition \p F. Add it to the map with initial node \p N.
  void create(Node *N, Function *F) {
    functions_.emplace_back(F);
    nodeToFunction_[N] = F;
  }

  /// Add a new Node->Function mapping.
  void add(Node *N, Function *F) { nodeToFunction_[N] = F; }

  /// Get list of functions contained in this map.
  const FunctionList &getFunctions() const { return functions_; }

  /// \returns the number of partitions.
  Map::size_type size() const { return functions_.size(); }

  /// Map API.
  Map::iterator find(Node *N) { return nodeToFunction_.find(N); }
  Map::iterator begin() { return nodeToFunction_.begin(); }
  Map::iterator end() { return nodeToFunction_.end(); }
  Function *operator[](Node *n) { return nodeToFunction_[n]; }
};

/// If \p node has a single input that is not a variable, return it.  Otherwise
/// return nullptr.
Node *singleNonVariableInput(Node *node) {
  Node *nonVarInput = nullptr;

  for (unsigned i = 0, e = node->getNumInputs(); i < e; i++) {
    Node *in = node->getNthInput(i).getNode();
    if (isa<Variable>(in))
      continue;
    if (nonVarInput)
      return nullptr;
    nonVarInput = in;
  }
  return nonVarInput;
}

/// Assign nodes to partitions and return the mapping.  This algorithm
/// partitions the graph in a manner that looks like basic blocks in a control
/// flow graphs: regions end after a node with multiple outputs or before a node
/// with multiple inputs.
NodeFunctionMap selectPartitions(Function *F) {
  NodeFunctionMap mapping;

  // Visit graph nodes in reverse post order so that a node's inputs are already
  // assigned to a partition before it is assigned.
  GraphPostOrderVisitor visitor(*F);
  for (auto *node : visitor.getPostOrder()) {
    if (isa<Variable>(node))
      continue;

    // If node has only one input, and that input has only one output, place it
    // in the same partition.
    auto *in = singleNonVariableInput(node);
    if (in && in->getNumUsers() == 1) {
      auto it = mapping.find(in);
      assert(it != mapping.end());
      mapping.add(node, it->second);
      continue;
    }

    // Start a new partition with this node.
    auto *newF = F->getParent()->createFunction(
        std::string(F->getName()) + "_part" + std::to_string(mapping.size()));
    mapping.create(node, newF);
  }

  return mapping;
}

/// Given a function \p F and partitioning \p mapping, \return a FunctionGraph
/// that contains appropriately-partitioned functions and their dependences.
FunctionGraph doPartitioning(Function *F, NodeFunctionMap &mapping) {
  FunctionGraph G(mapping.getFunctions());
  llvm::DenseMap<Node *, Node *> currToNew;
  auto *mod = F->getParent();

  // Clone nodes into target partition.
  for (auto &N : F->getNodes()) {
    auto *clone = N.clone();
    currToNew[&N] = clone;
    mapping[&N]->addNode(clone);
  }

  // For any dependency that crosses a partition, add a variable and save
  // node. Record the dependence in the function graph.
  llvm::DenseMap<Node *, Variable *> variables;
  for (auto *F : mapping.getFunctions()) {
    for (auto &N : F->getNodes()) {
      for (unsigned inp = 0, e = N.getNumInputs(); inp < e; inp++) {
        auto input = N.getNthInput(inp);
        if (isa<Variable>(input.getNode()))
          continue;

        auto *inputF = mapping[input.getNode()];
        if (F == inputF)
          continue;

        // Add this dependence to the FunctionGraph.
        G.add(F, inputF);

        // If we've already created a variable for this dependence, use it.
        auto it = variables.find(input.getNode());
        if (it != variables.end()) {
          N.setNthInput(inp, it->second);
          continue;
        }

        // Create a new variable to represent this dependence.
        auto *tmp = mod->createVariable(
            input.getType(), std::string(input->getName()) + "_tmp",
            VisibilityKind::Private, Variable::TrainKind::None);
        inputF->createSave("tmp", input, tmp);
        variables[input.getNode()] = tmp;
        N.setNthInput(inp, tmp);
      }
    }
  }

  // Update links between nodes in the cloned functions.  Add variables (and
  // save nodes) where a link crosses a partition boundary.
  for (auto *F : mapping.getFunctions()) {
    for (auto &N : F->getNodes()) {
      for (unsigned inp = 0, e = N.getNumInputs(); inp < e; inp++) {
        auto input = N.getNthInput(inp);

        if (isa<Variable>(input.getNode()))
          continue;

        // Link this node to the clone of its input.
        auto *clone = currToNew[input.getNode()];
        N.setNthInput(inp, NodeValue(clone, input.getResNo()));
      }
    }
  }

  return G;
}

} // end namespace

FunctionGraph::FunctionGraph(const FunctionList &functions)
    : functions_(functions) {
  for (auto *F : functions_) {
    inputs_[F] = FunctionList();
  }
}

FunctionGraph glow::partition(Function *F) {
  NodeFunctionMap partitionMap = selectPartitions(F);
  return doPartitioning(F, partitionMap);
}
