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
#include "glow/Backends/BackendUtils.h"
#include "glow/IR/Instrs.h"

using namespace glow;
using llvm::cast;
using llvm::isa;

uint8_t *glow::collectConstants(
    const IRFunction *F, uint64_t constantMaxSize,
    const std::unordered_map<std::string, runtime::RuntimeSymbolInfo>
        &symbolTable) {

  // At compile time condense constants to a single block of memory.
  // This allows the graph to go away after compile time.
  uint8_t *baseConstantWeightVarsStore =
      (uint8_t *)alignedAlloc(constantMaxSize, TensorAlignment);
  for (auto &v : F->getGraph()->getParent()->getConstants()) {
    assert(isa<WeightVar>(F->getWeightForNode(v)));
    auto *w = cast<WeightVar>(F->getWeightForNode(v));
    auto payload = v->getPayload().getUnsafePtr();
    auto numBytes = w->getSizeInBytes();
    auto it = symbolTable.find(std::string(w->getName()));
    assert(it != symbolTable.end() && "Symbol not found.");
    auto addr = it->second.offset;
    // Copy weight to offset.
    memcpy(baseConstantWeightVarsStore + addr, payload, numBytes);
  }
  return baseConstantWeightVarsStore;
}

/// Helper function, gets offset of \p v from \p symbolTable.
size_t glow::getValueOffset(
    Value *v, const std::unordered_map<std::string, runtime::RuntimeSymbolInfo>
                  &symbolTable) {
  auto it = symbolTable.find(std::string(v->getName()));
  assert(it != symbolTable.end() && "Symbol not found.");
  return it->second.offset;
}
