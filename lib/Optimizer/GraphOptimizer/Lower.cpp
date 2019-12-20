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

#include "glow/Optimizer/Lower/Lower.h"
#include "glow/Backend/Backend.h"
#include "glow/Graph/Graph.h"
#include "glow/Graph/Node.h"
#include "glow/Graph/Nodes.h"
#include "glow/Graph/TensorLayout.h"
#include "glow/Optimizer/GraphOptimizer/FunctionPasses.h"
#include "glow/Optimizer/GraphOptimizer/GraphOptimizer.h"

#include "llvm/Support/Casting.h"

#include <numeric>

using namespace glow;
using llvm::dyn_cast;

void glow::lower(Function *F, CompilationContext &cctx, const Backend *B,
                 const KindSet &doNotLowerKinds) {
  LOG_SCOPE(F->getLogContext(), "glow::lower")
  F->setState(FunctionState::FuncLoaded);

  auto &nodes = F->getNodes();
  for (auto &N : nodes) {
    if (B && !B->shouldLower(&N)) {
      continue;
    }
    if (doNotLowerKinds.count(N.getKind())) {
      continue;
    }
    lowerNode(F, &N, cctx);
  }

  for (auto it = F->getNodes().begin(), e = F->getNodes().end(); it != e;) {
    auto cur = &*(it++);
    if (dyn_cast<SGDNode>(cur)) {
      F->eraseNode(cur);
    }
  }

  // Remove nodes that were lowered.
  runDCEPass(F, cctx);
}
