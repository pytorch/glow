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
#define DEBUG_TYPE "partition"

#include "glow/Graph/FunctionGraph.h"

#include "glow/Graph/Graph.h"
#include "glow/Graph/Utils.h"
#include "glow/Support/Debug.h"

#include "llvm/Support/raw_ostream.h"

using namespace glow;

FunctionGraph::FunctionGraph(FunctionList fs, std::unordered_set<Variable *> vs)
    : functions(fs), channels(vs) {
  // Given a list of functions and a set of variables used as communication
  // channels between them, build a list of input and output dependences for
  // each function.
  for (auto &function : functions) {
    inputs.emplace(function, VariablesList());
    outputs.emplace(function, VariablesList());
    for (auto &N : function->getNodes()) {
      // Output deps.
      if (auto *S = llvm::dyn_cast<SaveNode>(&N)) {
        auto *V = S->getVariable();
        if (channels.count(V))
          outputs[function].emplace_back(V);
        continue;
      }
      // Input deps.
      for (unsigned inp = 0, e = N.getNumInputs(); inp < e; inp++) {
        auto const &in = N.getNthInput(inp);
        if (auto *V = llvm::dyn_cast<Variable>(in.getNode())) {
          if (channels.count(V))
            inputs[function].emplace_back(V);
        }
      }
    }
  }
}
