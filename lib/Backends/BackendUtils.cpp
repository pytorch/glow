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
using llvm::dyn_cast;
using llvm::isa;

void glow::runtime::RuntimeBundle::collectConstants(const IRFunction *F) {
  collectConstants(F->getGraph()->getParent());
}

void glow::runtime::RuntimeBundle::freeConstants() {
  if (constants_) {
    glow::alignedFree(constants_);
    constants_ = nullptr;
  }
}
void glow::runtime::RuntimeBundle::collectConstants(const Module *M) {
  // At compile time condense constants to a single block of memory.
  // This allows the graph to go away after compile time.
  // If there are no constants return nullptr.
  if (constantWeightVarsMemSize_ == 0) {
    constants_ = nullptr;
    return;
  }

  assert(constants_ == nullptr && "constants already allocated");
  constants_ =
      (uint8_t *)alignedAlloc(constantWeightVarsMemSize_, TensorAlignment);

  for (const auto &symbol : symbolTable_) {
    llvm::StringRef name = symbol.first;
    const RuntimeSymbolInfo &info = symbol.second;

    Constant *c = M->getConstantByName(name);
    if (!c) {
      continue;
    }
    auto *payload = c->getPayload().getUnsafePtr();
    assert(info.size == c->getPayload().getSizeInBytes() &&
           "Mismatched constant size");

    // Copy weight to offset.
    memcpy(constants_ + info.offset, payload, info.size);
  }
}

size_t glow::runtime::RuntimeBundle::getValueOffset(const Named *v) const {
  auto it = symbolTable_.find(std::string(v->getName()));
  assert(it != symbolTable_.end() && "Symbol not found.");
  return it->second.offset;
}

const runtime::RuntimeSymbolInfo &
runtime::RuntimeBundle::getSymbolInfo(const Named *v) const {
  auto it = symbolTable_.find(std::string(v->getName()));
  assert(it != symbolTable_.end() && "Symbol not found.");
  return it->second;
}

/// If \p PH is an output placeholder, \returns true.
/// This is determined by checking if the PH has a user which uses the PH as an
/// overwritten input.
bool isOutput(const Placeholder *PH) {
  for (const auto &use : PH->getUsers()) {
    // Look through the inputs of the PH's users. If an input is overwritten
    // check if it's the PH, if it is return true.
    auto *user = use.getUser();
    for (unsigned i = 0, numInputs = user->getNumInputs(); i < numInputs; i++) {
      // If the input is not overwritten we can continue.
      if (!user->isOverwrittenNthInput(i)) {
        continue;
      }
      auto input = use.getUser()->getNthInput(i);
      if (input.getNode() == PH) {
        return true;
      }
    }
  }
  return false;
}

/// If \p PH is an input placeholder, \returns true.
bool isInput(const Placeholder *PH) {
  // Check that the PH is the input to a saveNode or is used by a non saveNode.
  for (const auto &use : PH->getUsers()) {
    // Check if PH is an input to a saveNode.
    if (auto *save = dyn_cast<SaveNode>(use.getUser())) {
      auto input = save->getInput();
      // If the PH is not an input to the saveNode we keep looking.
      if (input.getNode() != PH) {
        continue;
      }
    }
    return true;
  }
  return false;
}

runtime::RuntimeBundle runtime::RuntimeBundle::create(const Function &F) {
  std::unordered_map<std::string, runtime::RuntimeSymbolInfo> symbolTable;

  MemoryAllocator constants("constants", 0);
  MemoryAllocator placeholders("placeholders", 0);

  // Allocate constants.
  for (auto const *V : F.findConstants()) {
    auto size = V->getType()->getSizeInBytes();
    auto offset = constants.allocate(size, V);
    runtime::RuntimeSymbolInfo symbol;
    symbol.offset = offset;
    symbol.size = size;
    symbol.type = *V->getType();
    symbol.input = false;
    symbol.output = false;
    symbol.symbolCategory = SymbolCategory::Constant;
    symbolTable.emplace(V->getName(), symbol);
  }

  // Allocate placeholders.
  for (auto const *V : F.findPlaceholders()) {
    auto size = V->getType()->getSizeInBytes();
    auto offset = placeholders.allocate(size, V);
    runtime::RuntimeSymbolInfo symbol;
    symbol.offset = offset;
    symbol.size = size;
    symbol.type = *V->getType();
    symbol.output = isOutput(V);
    symbol.input = isInput(V);
    symbol.symbolCategory = SymbolCategory::Placeholder;
    symbolTable.emplace(V->getName(), symbol);
  }

  return runtime::RuntimeBundle(symbolTable, constants.getMaxMemoryUsage(),
                                placeholders.getMaxMemoryUsage(),
                                /*activationsMaxSize*/ 0);
}

runtime::RuntimeBundle
runtime::RuntimeBundle::create(const IRFunction &F,
                               MemoryAllocator &constantAllocator,
                               MemoryAllocator &placeholderAllocator,
                               MemoryAllocator &activationsAllocator) {
  // Handle Constants, Placeholders, and Activations, in that order.
  // Symbol table mapping symbol name to offset for runtime.
  std::unordered_map<std::string, runtime::RuntimeSymbolInfo> symbolTable;
  // Compute the offsets for Constants.
  for (auto &v : F.findConstants()) {
    assert(isa<WeightVar>(F.getWeightForNode(v)) && "Expected WeightVar");
    auto *w = cast<WeightVar>(F.getWeightForNode(v));
    auto numBytes = w->getSizeInBytes();
    size_t addr = constantAllocator.allocate(numBytes, v);
    runtime::RuntimeSymbolInfo symbol;
    symbol.size = numBytes;
    symbol.offset = addr;
    symbol.type = *w->getType();
    symbol.input = false;
    symbol.output = false;
    symbol.symbolCategory = SymbolCategory::Constant;
    symbolTable.emplace(std::string(v->getName()), symbol);
  }
  auto constantMaxSize = constantAllocator.getMaxMemoryUsage();

  // Compute the offsets for Placeholders.
  for (auto &v : F.findPlaceholders()) {
    // Get the WeightVar for each Placeholder to calculate offsets.
    assert(isa<WeightVar>(F.getWeightForNode(v)) && "Expected WeightVar");
    auto *w = cast<WeightVar>(F.getWeightForNode(v));
    auto numBytes = w->getSizeInBytes();
    size_t addr = placeholderAllocator.allocate(numBytes, w);
    runtime::RuntimeSymbolInfo symbol;
    symbol.offset = addr;
    symbol.size = numBytes;
    symbol.type = *w->getType();
    symbol.output = isOutput(v);
    symbol.input = isInput(v);
    symbol.symbolCategory = SymbolCategory::Placeholder;
    symbolTable.emplace(std::string(v->getName()), symbol);
  }
  auto placeholderMaxSize = placeholderAllocator.getMaxMemoryUsage();

  // Compute the offsets for Activations.

  for (const auto &I : F.getInstrs()) {
    if (auto *A = dyn_cast<AllocActivationInst>(&I)) {
      auto numBytes = I.getSizeInBytes();
      size_t addr = activationsAllocator.allocate(numBytes, A);
      assert(!symbolTable.count(std::string(A->getName())) &&
             "Allocation already made!");
      runtime::RuntimeSymbolInfo symbol;
      symbol.offset = addr;
      symbol.size = numBytes;
      symbol.type = *A->getType();
      symbol.input = false;
      symbol.output = false;
      symbol.symbolCategory = SymbolCategory::Activation;
      symbolTable.emplace(std::string(A->getName()), symbol);
      continue;
    }

    if (auto *TV = dyn_cast<TensorViewInst>(&I)) {
      // Calculate and store the length of the offset into the base, using the
      // source of the tensorview.
      assert(!symbolTable.count(std::string(TV->getName())) &&
             "Allocation already made!");
      size_t offsetLength = TV->getOffsets().empty() ? 0 : TV->getOffsets()[0];
      auto *tvSource = TV->getSrc();
      if (tvSource->dims().size() > 1) {
        for (size_t i = 1; i < tvSource->dims().size(); ++i) {
          offsetLength *= tvSource->dims()[i];
        }
      }
      assert(symbolTable.count(std::string(tvSource->getName())) &&
             "Source allocation not found!");
      runtime::RuntimeSymbolInfo symbol;
      symbol.offset = symbolTable[std::string(tvSource->getName())].offset +
                      (offsetLength * TV->getType()->getElementSize());
      symbol.size = TV->getSizeInBytes();
      symbol.type = *TV->getType();
      symbol.input = false;
      symbol.output = false;
      auto parentCategory =
          symbolTable.find(tvSource->getName())->second.symbolCategory;
      if (parentCategory == SymbolCategory::Placeholder) {
        symbol.symbolCategory = SymbolCategory::PlaceholderTensorView;
      } else {
        symbol.symbolCategory = SymbolCategory::ConstantTensorView;
      }
      symbolTable.emplace(std::string(TV->getName()), symbol);
      continue;
    }

    if (auto *D = dyn_cast<DeallocActivationInst>(&I)) {
      auto *A = D->getAlloc();
      assert(symbolTable.count(std::string(A->getName())) &&
             "Invalid deallocation!");
      activationsAllocator.deallocate(A);
      continue;
    }
  }
  auto activationsMaxSize = activationsAllocator.getMaxMemoryUsage();

  return runtime::RuntimeBundle(symbolTable, constantMaxSize,
                                placeholderMaxSize, activationsMaxSize);
}
