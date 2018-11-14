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
  // If there are no constants return nullptr.
  if (constantMaxSize == 0) {
    return nullptr;
  }
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

runtime::RuntimeBundle
glow::generateRuntimeBundle(const IRFunction *F,
                            MemoryAllocator *constantAllocator,
                            MemoryAllocator *placeholderAllocator,
                            MemoryAllocator *activationsAllocator) {
  // Handle Constants, Placeholders, and Activations, in that order.
  /// Symbol table mapping symbol name to offset for runtime.
  std::unordered_map<std::string, runtime::RuntimeSymbolInfo> symbolTable;
  uint64_t constantMaxSize = 0;
  uint64_t placeholderMaxSize = 0;
  uint64_t activationsMaxSize = 0;

  // Compute the offsets for Constants.
  if (constantAllocator) {
    for (auto &v : F->getGraph()->getParent()->getConstants()) {
      assert(isa<WeightVar>(F->getWeightForNode(v)) && "Expected WeightVar");
      auto *w = cast<WeightVar>(F->getWeightForNode(v));
      auto numBytes = w->getSizeInBytes();
      size_t addr = constantAllocator->allocate(numBytes, v);
      runtime::RuntimeSymbolInfo symbol;
      symbol.size = numBytes;
      symbol.offset = addr;
      symbol.type = *w->getType();
      symbolTable.emplace(std::string(v->getName()), symbol);
    }
    constantMaxSize = constantAllocator->getMaxMemoryUsage();
  }

  // Compute the offsets for Placeholders.
  if (placeholderAllocator) {
    for (auto &v : F->getGraph()->getParent()->getPlaceholders()) {
      // Get the WeightVar for each Placeholder to calculate offsets.
      assert(isa<WeightVar>(F->getWeightForNode(v)) && "Expected WeightVar");
      auto *w = cast<WeightVar>(F->getWeightForNode(v));
      auto numBytes = w->getSizeInBytes();
      size_t addr = placeholderAllocator->allocate(numBytes, w);
      runtime::RuntimeSymbolInfo symbol;
      symbol.offset = addr;
      symbol.size = numBytes;
      symbol.type = *w->getType();
      symbolTable.emplace(std::string(v->getName()), symbol);
    }
    placeholderMaxSize = placeholderAllocator->getMaxMemoryUsage();
  }
  // Compute the offsets for Activations.
  if (activationsAllocator) {
    for (const auto &I : F->getInstrs()) {
      if (auto *A = llvm::dyn_cast<AllocActivationInst>(&I)) {
        auto numBytes = I.getSizeInBytes();
        size_t addr = activationsAllocator->allocate(numBytes, A);
        assert(!symbolTable.count(std::string(A->getName())) &&
               "Allocation already made!");
        runtime::RuntimeSymbolInfo symbol;
        symbol.offset = addr;
        symbol.size = numBytes;
        symbol.type = *A->getType();
        symbolTable.emplace(std::string(A->getName()), symbol);
        continue;
      }

      if (auto *TV = llvm::dyn_cast<TensorViewInst>(&I)) {
        // Calculate and store the length of the offset into the base, using the
        // source of the tensorview.
        assert(!symbolTable.count(std::string(TV->getName())) &&
               "Allocation already made!");
        size_t offsetLength =
            TV->getOffsets().empty() ? 0 : TV->getOffsets()[0];
        auto *tvSource = TV->getSrc();
        if (tvSource->dims().size() > 1) {
          for (size_t i = 1; i < tvSource->dims().size(); ++i) {
            offsetLength *= tvSource->dims()[i];
          }
        }
        assert(symbolTable.count(std::string(tvSource->getName())) &&
               "Source allocation not found!");
        runtime::RuntimeSymbolInfo symbol;
        symbol.offset = getValueOffset(tvSource, symbolTable) +
                        (offsetLength * TV->getType()->getElementSize());
        symbol.size = TV->getSizeInBytes();
        symbol.type = *TV->getType();
        symbolTable.emplace(std::string(TV->getName()), symbol);
        continue;
      }

      if (auto *D = llvm::dyn_cast<DeallocActivationInst>(&I)) {
        auto *A = D->getAlloc();
        assert(symbolTable.count(std::string(A->getName())) &&
               "Invalid deallocation!");
        activationsAllocator->deallocate(A);
        continue;
      }
    }
    activationsMaxSize = activationsAllocator->getMaxMemoryUsage();
  }

  runtime::RuntimeBundle info(constantMaxSize, placeholderMaxSize,
                              activationsMaxSize);
  info.symbolTable = std::move(symbolTable);
  info.constants = collectConstants(F, constantMaxSize, info.symbolTable);
  return info;
}