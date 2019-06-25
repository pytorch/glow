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

#include "glow/Partitioner/PartitionerUtils.h"
#include <unordered_set>

using llvm::isa;

namespace glow {

namespace {
/// Used to sort 2 Nodes by their name, i.e. n1->name < n2->name order.
auto compFunc = [](const Node *n1, Node *n2) -> bool {
  return n1->compareByName(*n2);
};
} // namespace

/// The nodes in function \p F which be grouped into levels based on how far
/// (the longest distance) they are from the roots.
BFSLevel getBFSLevel(Function *F) {
  // The current set of nodes needs to be visited
  std::unordered_set<Node *> cur;
  // A map between a node and its level.
  llvm::DenseMap<Node *, int> nodeLevel;

  // Get the roots set (i.e. the nodes without users).
  for (auto &node : F->getNodes()) {
    if (node.getNumUsers() == 0) {
      // A root has no users.
      cur.insert(&node);
      nodeLevel[&node] = 0;
    }
  }

  // Create the node to level map by traversing the nodes with BFS order.
  BFSLevel bfs;
  int level = 0;
  int current = 0;
  bfs.push_back(std::vector<Node *>());
  level++;
  while (current < level) {
    std::unordered_set<Node *> nodes;
    for (std::unordered_set<Node *>::iterator it = cur.begin(); it != cur.end();
         ++it) {
      Node *N = *it;
      for (size_t j = 0, e = N->getNumInputs(); j < e; ++j) {
        Node *in = N->getNthInput(j).getNode();
        if (isa<Storage>(in)) {
          continue;
        }
        nodes.insert(in);
        nodeLevel[in] = level;
      }
    }
    if (nodes.size() > 0) {
      bfs.push_back(std::vector<Node *>());
      level++;
      cur = std::move(nodes);
    }
    current++;
  }

  // Based on the node to level map, group these nodes by levels.
  for (llvm::DenseMap<Node *, int>::iterator it = nodeLevel.begin();
       it != nodeLevel.end(); ++it) {
    Node *in = (*it).first;
    int level = (*it).second;
    bfs[level].push_back(in);
  }

  // Sort the nodes of each level by name to make sure the nodes sequence are
  // the same for different run.
  for (int i = 0; i < level; i++) {
    std::sort(bfs[i].begin(), bfs[i].end(), compFunc);
  }
  return bfs;
}

/// Given \p nodes, return a list of nodes who are not in this set but use any
/// node in this set.
std::vector<Node *> getOutUsers(const NodesSetTy &nodes) {
  NodesSetTy used;
  for (NodesSetTy::iterator it = nodes.begin(); it != nodes.end(); ++it) {
    Node *cur = *it;
    for (auto &U : cur->getUsers()) {
      if (nodes.count(U.getUser())) {
        continue;
      }
      used.insert(U.getUser());
    }
  }

  std::vector<Node *> ret(used.begin(), used.end());
  std::sort(ret.begin(), ret.end(), compFunc);
  return ret;
}

/// Given \p nodes, return a list of nodes who are not in this set but use only
/// the nodes in this set or constant.
std::vector<Node *> getOutUsersWithOnePredecessor(const NodesSetTy &nodes) {
  NodesSetTy used;
  for (NodesSetTy::iterator it = nodes.begin(); it != nodes.end(); ++it) {
    Node *cur = *it;
    for (auto &U : cur->getUsers()) {
      Node *user = U.getUser();
      if (nodes.count(user)) {
        continue;
      }
      bool flag = true;
      for (size_t i = 0, e = user->getNumInputs(); i < e; i++) {
        Node *in = user->getNthInput(i).getNode();
        if (llvm::isa<Storage>(in) || nodes.count(in)) {
          continue;
        }
        flag = false;
        break;
      }
      if (flag) {
        used.insert(user);
      }
    }
  }

  std::vector<Node *> ret(used.begin(), used.end());
  std::sort(ret.begin(), ret.end(), compFunc);
  return ret;
}

/// \returns the memory usage of the output caused by \p node who has users not
/// in the set \p nodes.
uint64_t getOutMemPerNode(const NodesSetTy &nodes, const Node *node) {
  uint64_t ret = 0;
  for (size_t i = 0, e = node->getNumResults(); i < e; i++) {
    NodeValue nodeVal = node->getNthResult(i);
    for (auto &U : nodeVal.getUsers()) {
      Node *user = U.getUser();
      if (nodes.find(const_cast<Node *>(user)) == nodes.end()) {
        ret += node->getType(i)->getSizeInBytes();
        break;
      }
    }
  }
  return ret;
}

/// Given nodes set \p currNodes and its memory usage info \p info, \returns the
/// new memory usage if \p newNode is added into \p currNodes.
GraphMemInfo updateGraphMemInfoByAddingNode(const NodesSetTy &currNodes,
                                            const GraphMemInfo &info,
                                            Node *newNode) {
  GraphMemInfo ret = info;

  // Collect the used NodeValues (Storage nodes and outputs from the nodes
  // outside of currNodes).
  std::set<NodeValue> usedNodeValue;
  for (auto N : currNodes) {
    for (size_t i = 0, e = N->getNumInputs(); i < e; i++) {
      NodeValue nodeVal = N->getNthInput(i);
      if (currNodes.count(nodeVal.getNode()) == 0) {
        usedNodeValue.insert(nodeVal);
      }
    }
  }

  // The memory usage changes due to newNode's inputs:
  for (size_t i = 0, e = newNode->getNumInputs(); i < e; i++) {
    if (llvm::isa<SaveNode>(newNode) && i == SaveNode::OutputIdx) {
      continue;
    }
    NodeValue nodeVal = newNode->getNthInput(i);
    Node *N = nodeVal.getNode();

    if (usedNodeValue.count(nodeVal)) {
      // This input has been considered already, nothing to do.
      continue;
    }

    Storage *in = llvm::dyn_cast<Storage>(N);
    if (in) {
      // Node uses placeholders or constants which are not used in this set
      // before, need to add the memory.
      uint64_t size = in->getType()->getSizeInBytes();
      if (in->getKind() == Kinded::Kind::ConstantKind) {
        ret.constMemSize += size;
      } else {
        // PlaceHolder for Input.
        ret.inMemSize += size;
      }
      usedNodeValue.insert(nodeVal);
      continue;
    }

    if (currNodes.count(N)) {
      // This input is inside of currNodes. Let node1 belongs to currNodes and
      // only newNode uses nodes1's output(i.e. node1 -> newNode, and if node1
      // -> node2, node2 belongs to currNodes). Before newNode is added, the
      // size of node1's output is added into outMemSize. But if newNode is
      // added, node1's output size should be removed from outMemSize.
      bool removable = true;
      for (auto &U : nodeVal.getUsers()) {
        if (U.getUser() != newNode && currNodes.find(const_cast<Node *>(
                                          U.getUser())) == currNodes.end()) {
          // This means nodeVal is used by some other node not in currNodes, so
          // even newNode is added, the output size still need to be counted.
          removable = false;
          break;
        }
      }
      if (removable) {
        ret.outMemSize -= nodeVal.getType()->getSizeInBytes();
      }
    } else {
      // In this case, this input is not a storage type node nor belongs
      // to this subgraph. Therefore, when creating paritions, we need to add
      // a PlaceHolder for the data from outside.
      ret.inMemSize += nodeVal.getType()->getSizeInBytes();
      usedNodeValue.insert(nodeVal);
    }
  }

  // The memory usage changes due to newNode's outputs.
  if (auto *SN = llvm::dyn_cast<SaveNode>(newNode)) {
    // For SaveNode, add the output size.
    Storage *out = llvm::dyn_cast<Storage>(SN->getPlaceholder());
    ret.outMemSize += out->getType()->getSizeInBytes();
    return ret;
  }

  for (size_t i = 0, e = newNode->getNumResults(); i < e; i++) {
    auto nodeVal = newNode->getNthResult(i);
    for (auto &U : nodeVal.getUsers()) {
      if (currNodes.count(U.getUser()) == 0) {
        // The nodeVal (i.e. the ith output of newNode) is not used in
        // currNodes:
        continue;
      }
      // Assume newNode -> node1, where node1 belongs to currNodes set. Before
      // newNode is added, node1's input size (from newNode) should be added
      // into inMemSize. But afater newNode is added, the input size should be
      // removed.
      ret.inMemSize -= nodeVal.getType()->getSizeInBytes();
      break;
    }
  }

  // Add the memory usage caused by newNode.
  ret.outMemSize += getOutMemPerNode(currNodes, newNode);
  return ret;
}

GraphMemInfo getGraphMemInfo(const NodesSetTy &nodes) {
  GraphMemInfo ret;
  NodesSetTy nodeSet;
  for (NodesSetTy::iterator it = nodes.begin(); it != nodes.end(); ++it) {
    Node *cur = *it;
    ret = updateGraphMemInfoByAddingNode(nodeSet, ret, cur);
    nodeSet.insert(cur);
  }
  return ret;
}

} // namespace glow
