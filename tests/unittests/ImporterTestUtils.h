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
#ifndef GLOW_TESTS_IMPORTERTESTUTILS_H
#define GLOW_TESTS_IMPORTERTESTUTILS_H

#include "glow/Base/Tensor.h"
#include "glow/Graph/Graph.h"

namespace glow {

/// Populate \p result with increasing number following the NCHW format.
/// E.g.:
/// N=1:
///               W=3
///             <---->
///      ^  +---+---+---+
///  C=1/   | 0 | 1 | 2 |  ^
///    /    +---+---+---+  | H=3
///   v     | 3 | 4 | 5 |  v
///         +---+---+---+
///         | 6 | 7 | 8 |
///         +---+---+---+
void getNCHWData(Tensor *result, dim_t n, dim_t c, dim_t h, dim_t w) {
  result->reset(ElemKind::FloatTy, {n, c, h, w});
  auto RH = result->getHandle<>();
  for (size_t i = 0, e = n * c * h * w; i < e; i++)
    RH.raw(i) = i;
}

/// Populate \p result with increasing number following the NCTHW format.
/// E.g.:
/// N=1, T=2:
///               T=0              T=1
///               W=3
///             <---->
///      ^  +---+---+---+    +----+----+----+
///  C=1/   | 0 | 1 | 2 |    |  9 | 10 | 11 |  ^
///    /    +---+---+---+    +----+----+----+  | H=3
///   v     | 3 | 4 | 5 |    | 12 | 13 | 14 |  v
///         +---+---+---+    +----+----+----+
///         | 6 | 7 | 8 |    | 15 | 16 | 17 |
///         +---+---+---+    +----+----+----+
void getNCTHWData(Tensor *result, dim_t n, dim_t c, dim_t t, dim_t h, dim_t w) {
  result->reset(ElemKind::FloatTy, {n, c, t, h, w});
  auto RH = result->getHandle<>();
  for (size_t i = 0, e = n * c * t * h * w; i < e; i++)
    RH.raw(i) = i;
}

/// \returns the number of nodes in \p F of kind \p kind.
unsigned countNodeKind(Function *F, Kinded::Kind kind) {
  unsigned count = 0;
  for (auto &n : F->getNodes()) {
    if (n.getKind() == kind) {
      count++;
    }
  }
  return count;
}

/// Helper function to get the save node from a Variable \p var.
/// \pre (var->getUsers().size() == 1)
SaveNode *getSaveNodeFromDest(Storage *var) {
  auto &varUsers = var->getUsers();
  assert(varUsers.size() == 1);
  auto *saveNode = llvm::dyn_cast<SaveNode>(varUsers.begin()->getUser());
  assert(saveNode != nullptr);
  return saveNode;
}

} // namespace glow

#endif // GLOW_TESTS_IMPORTERTESTUTILS_H
