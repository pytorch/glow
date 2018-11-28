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

#include "glow/Graph/Hook.h"

#include "glow/Graph/Graph.h"
#include "glow/Graph/Node.h"

#include "llvm/ADT/DenseMap.h"

using namespace glow;

namespace {
/// Map from original Nodes to cloned Nodes.
using NodeMap = llvm::DenseMap<Node *, Node *>;
} // namespace

/// Clone \p node and its sources into \p newF using old-to-new mapping \p
/// currToNew.
static Node *recursiveClone(Function *newF, Node *node, NodeMap &currToNew) {
  Node *copy = node->clone();
  currToNew[node] = copy;
  newF->addNode(copy);
  for (unsigned inp = 0, e = copy->getNumInputs(); inp < e; inp++) {
    auto input = copy->getNthInput(inp);
    auto it = currToNew.find(input.getNode());
    Node *newInput;
    if (it != currToNew.end()) {
      newInput = it->second;
    } else if (llvm::isa<Storage>(input.getNode())) {
      continue;
    } else {
      newInput = recursiveClone(newF, input.getNode(), currToNew);
    }
    copy->setNthInput(inp, NodeValue(newInput, input.getResNo()));
  }
  return copy;
}

HookedFunction glow::hookOutput(Function *F, Node *node) {
  NodeMap currToNew;
  auto *newF = F->getParent()->createFunction("hook");
  Node *hooked = recursiveClone(newF, node, currToNew);
  auto *save = newF->createSave("hook_save", hooked);
  return HookedFunction{newF, save, save->getPlaceholder()};
}

HookedFunction glow::hookOutput(Function *F, llvm::StringRef nodeName) {
  auto *node = F->getNodeByName(nodeName);
  return hookOutput(F, node);
}
