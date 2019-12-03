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

#include "glow/Graph/Hook.h"

#include "glow/Graph/Graph.h"
#include "glow/Graph/Node.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/FormatVariadic.h"

using namespace glow;

HookedFunction glow::hookOutput(Function *F, Node *node) {
  NodeMap currToNew;
  auto *newF = F->getParent()->createFunction("hook");
  Node *hooked = recursiveClone(newF, node, currToNew);

  std::list<SaveNode *> saves;
  std::list<Placeholder *> placeholders;

  for (unsigned i = 0; i < hooked->getNumResults(); ++i) {
    auto *save = newF->createSave(llvm::formatv("hook_save_{}", i).str(),
                                  hooked->getNthResult(i));
    saves.emplace_back(save);
    placeholders.emplace_back(save->getPlaceholder());
  }

  return HookedFunction{newF, std::move(saves), std::move(placeholders)};
}

HookedFunction glow::hookOutput(Function *F, llvm::StringRef nodeName) {
  auto *node = F->getNodeByName(nodeName);
  return hookOutput(F, node);
}
