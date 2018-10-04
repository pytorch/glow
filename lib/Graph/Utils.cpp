/**
 * Copyright (c) 2018-present, Facebook, Inc.
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

#include "glow/Base/Tensor.h"
#include "glow/Base/Type.h"
#include "glow/Graph/Context.h"
#include "glow/Graph/Utils.h"
#include "glow/Graph/Graph.h"
#include "glow/Graph/Nodes.h"

using namespace glow;

void glow::mutateNodesType(Function &F, ElemKind origTy, ElemKind newTy,
                           Context *context) {
  Module &M = *F.getParent();
  // Check that the module only has one function. If it has more than one, we
  // would have to duplicate the placeholders and other module scoped objects
  // to make sure we are not modifying other functions.
  GLOW_ASSERT(M.getFunctions().size() == 1 && "No yet implemented");
  for (Node &node : F.getNodes()) {
    mutateNodeResTypeThatMatch(M, node, origTy, newTy);
  }
  for (Variable *var : M.getVars()) {
    if (var->getElementType() != origTy) {
      continue;
    }
    mutateNodeResTypeThatMatch(M, *var, origTy, newTy);
    // Update the values in the tensor.
    var->getPayload().convertToType(newTy);
  }
  for (Placeholder *placeholder : M.getPlaceholders()) {
    if (placeholder->getElementType() != origTy) {
      continue;
    }
    mutateNodeResTypeThatMatch(M, *placeholder, origTy, newTy);
    if (context && context->get(placeholder)) {
      context->get(placeholder)->convertToType(newTy);
    }
  }
}
