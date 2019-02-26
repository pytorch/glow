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

/// Used to sort 2 Nodes by their name.
static bool compFunc(Node *n1, Node *n2) {
  return (n1->getName().compare(n2->getName()) > 0);
}

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
std::vector<Node *> getOutUsers(const std::set<Node *> &nodes) {
  std::vector<Node *> ret;
  for (std::set<Node *>::iterator it = nodes.begin(); it != nodes.end(); ++it) {
    Node *cur = *it;
    for (auto &U : cur->getUsers()) {
      if (nodes.count(U.getUser())) {
        continue;
      }
      ret.push_back(U.getUser());
    }
  }
  return ret;
}

/// Given \p nodes, return a list of nodes who are not in this set but use only
/// the nodes in this set or constant.
std::vector<Node *>
getOutUsersWithOnePredecessor(const std::set<Node *> &nodes) {
  std::vector<Node *> ret;
  for (std::set<Node *>::iterator it = nodes.begin(); it != nodes.end(); ++it) {
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
        ret.push_back(user);
      }
    }
  }
  return ret;
}

GraphMemInfo getGraphMemInfo(const std::set<Node *> &nodes) {
  GraphMemInfo ret;
  std::set<Node *> nSet;
  for (std::set<Node *>::iterator it = nodes.begin(); it != nodes.end(); ++it) {
    Node *cur = *it;
    // For Save onde, the only required memory is for output.
    if (auto *SN = llvm::dyn_cast<SaveNode>(cur)) {
      Storage *out = llvm::dyn_cast<Storage>(SN->getOutput().getNode());
      ret.outMemSize += out->getType()->getSizeInBytes();
      continue;
    }
    // Check the inputs of each node in this subgraph and decide if it
    // contributes to the memory usage:
    for (size_t i = 0, e = cur->getNumInputs(); i < e; i++) {
      Node *node = cur->getNthInput(i).getNode();
      if (nodes.count(node) || nSet.count(node)) {
        // This input belongs to this subgraph or it has been considered
        // already, nothing to do.
        continue;
      }
      nSet.insert(node);
      Storage *in = llvm::dyn_cast<Storage>(node);
      if (in) {
        size_t size = in->getType()->getSizeInBytes();
        if (node->getKind() == Kinded::Kind::ConstantKind) {
          // Constant.
          ret.constMemSize += size;
        } else {
          // PlaceHolder for Input.
          ret.inMemSize += size;
        }
      } else {
        // In this case, this input is neither a storage type node nor belongs
        // to this subgraph. Therefore, when creating paritions, we need to add
        // a PlaceHolder for the data from outside.
        for (auto &U : node->getUsers()) {
          if (U.getUser() == cur) {
            ret.inMemSize += node->getType(0)->getSizeInBytes();
            break;
          }
        }
      }
    }
    // Check the outputs of each node in this subgraph and decide if it
    // contributes to the memory usage. Although at the stage, the output may
    // not be a storage node, after real partitioning, a Save node will be added
    // to hold the output:
    for (size_t i = 0, e = cur->getNumResults(); i < e; i++) {
      for (auto &U : cur->getNthResult(i).getNode()->getUsers()) {
        Node *node = U.getUser();
        if (nodes.count(node) || nSet.count(node)) {
          // The output belongs to this subgraph, nothing needs to do.
          continue;
        }
        nSet.insert(node);
        ret.outMemSize += cur->getType(i)->getSizeInBytes();
      }
    }
  }
  return ret;
}
} // namespace glow
