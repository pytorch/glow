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

/// This struct is used to keep information returned from
/// hooking a function.
struct HookedFunction {
  /// Saves the function hookNode() creates and inserts hooks in.
  Function *function;
  /// List of Save nodes hookNode() inserts at the outputs of the layer.
  std::list<SaveNode *> outputSaves;
  /// List of the Placeholders associated with Save nodes in outputSaves.
  std::list<Placeholder *> outputs;
  /// List of Save nodes hookNode() inserts at the input of the layer.
  std::list<SaveNode *> inputSaves;
  /// List of the Placeholders associated with Save nodes in inputSaves.
  std::list<Placeholder *> inputs;
};

/// Given a function \p F and a node \p node it creates a function
/// in the same module and populates it with a recursive clone of
/// the node \p node then inserts Save nodes at the output of to capture it.
/// If \p hookInputs is set it also inserts Save nodes to capture the inputs.
/// \returns the list of inserted nodes and associated placeholder in
/// \p HookedFunction object.
HookedFunction hookNode(Function *F, Node *node, bool hookInputs = false);

/// Given a function \p F and a layer name \p nodeName it finds the
/// corresponding nodes in the function \p F then calls above override.
HookedFunction hookNode(Function *F, llvm::StringRef nodeName,
                        bool hookInputs = false);

} // namespace glow

#endif // GLOW_GRAPH_HOOK_H
