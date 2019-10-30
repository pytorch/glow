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
#ifndef GLOW_GRAPH_HOOK_H
#define GLOW_GRAPH_HOOK_H

#include "llvm/ADT/StringRef.h"
#include <list>

namespace glow {

class Function;
class Node;
class Placeholder;
class SaveNode;

struct HookedFunction {
  Function *function;
  std::list<SaveNode *> saves;
  std::list<Placeholder *> outputs;
};

HookedFunction hookOutput(Function *F, Node *node);

HookedFunction hookOutput(Function *F, llvm::StringRef nodeName);

} // namespace glow

#endif // GLOW_GRAPH_HOOK_H
