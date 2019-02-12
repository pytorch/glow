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

#include <set>

namespace glow {

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
      for (int i = 0, e = user->getNumInputs(); i < e; i++) {
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
    for (int i = 0, e = cur->getNumInputs(); i < e; i++) {
      Node *node = cur->getNthInput(i).getNode();
      if (nodes.count(node) || nSet.count(node)) {
        // This input belongs to this subgraph or it has been considered
        // already, nothing to do.
        continue;
      }
      nSet.insert(node);
      Storage *in = llvm::dyn_cast<Storage>(node);
      if (in) {
        uint64_t size = in->getType()->getSizeInBytes();
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
    for (int i = 0, e = cur->getNumResults(); i < e; i++) {
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
