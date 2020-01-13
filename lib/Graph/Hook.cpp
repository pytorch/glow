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

HookedFunction glow::hookNode(Function *F, Node *node, bool hookInputs) {
  NodeMap currToNew;
  auto *newF = F->getParent()->createFunction("hook");
  Node *hooked = recursiveClone(newF, node, currToNew);

  std::list<SaveNode *> output_saves;
  std::list<Placeholder *> output_placeholders;

  std::list<SaveNode *> input_saves;
  std::list<Placeholder *> input_placeholders;

  for (unsigned i = 0; i < hooked->getNumResults(); ++i) {
    auto *save =
        newF->createSave(hooked->getOutputName(i), hooked->getNthResult(i));
    output_saves.emplace_back(save);
    output_placeholders.emplace_back(save->getPlaceholder());
  }

  if (hookInputs) {
    for (unsigned i = 0; i < hooked->getNumInputs(); ++i) {
      auto *save =
          newF->createSave(hooked->getInputName(i), hooked->getNthInput(i));
      input_saves.emplace_back(save);
      input_placeholders.emplace_back(save->getPlaceholder());
    }
  }
  return HookedFunction{newF, std::move(output_saves),
                        std::move(output_placeholders), std::move(input_saves),
                        std::move(input_placeholders)};
}

HookedFunction glow::hookNode(Function *F, llvm::StringRef nodeName,
                              bool hookInputs) {
  auto *node = F->getNodeByName(nodeName);
  return hookNode(F, node, hookInputs);
}
