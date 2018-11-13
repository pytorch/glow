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
#include "CPUFunction.h"

#include "glow/Graph/Context.h"
#include "glow/Support/Compiler.h"
#include "glow/Support/Memory.h"

using namespace glow;

CPUFunction::CPUFunction(std::unique_ptr<llvm::orc::GlowJIT> JIT,
                         const runtime::RuntimeBundle &runtimeBundle)
    : JIT_(std::move(JIT)), runtimeBundle_(runtimeBundle) {}

CPUFunction::~CPUFunction() { alignedFree(runtimeBundle_.constants); }

void CPUFunction::setupRuns() {
  if (runtimeBundle_.activationsMemSize != 0) {
    baseActivationsAddress_ = (uint8_t *)alignedAlloc(
        runtimeBundle_.activationsMemSize, TensorAlignment);
  }

  if (runtimeBundle_.mutableWeightVarsMemSize != 0) {
    baseMutableWeightVarsAddress_ = (uint8_t *)alignedAlloc(
        runtimeBundle_.mutableWeightVarsMemSize, TensorAlignment);
  }
}

void CPUFunction::beforeRun(const Context &ctx) {
  // Copy Placeholders into allocated memory.
  for (auto PH : ctx.pairs()) {
    auto payload = PH.second->getUnsafePtr();
    auto symbolInfo =
        runtimeBundle_.symbolTable.find(std::string(PH.first->getName()));
    assert(symbolInfo != runtimeBundle_.symbolTable.end() &&
           "Symbol not found");
    auto addr = symbolInfo->second.offset;
    auto numBytes = symbolInfo->second.size;
    // copy PH to allocated memory.
    memcpy(baseMutableWeightVarsAddress_ + addr, payload, numBytes);
  }
}

void CPUFunction::afterRun(const Context &ctx) {
  // Copy placeholders from device back into context.
  for (auto PH : ctx.pairs()) {
    auto symbolInfo =
        runtimeBundle_.symbolTable.find(std::string(PH.first->getName()));
    auto payload = baseMutableWeightVarsAddress_ + symbolInfo->second.offset;
    auto numBytes = symbolInfo->second.size;
    auto addr = PH.second->getUnsafePtr();
    // copy PH from allocated memory.
    memcpy(addr, payload, numBytes);
  }
}

void CPUFunction::tearDownRuns() {
  if (baseMutableWeightVarsAddress_) {
    alignedFree(baseMutableWeightVarsAddress_);
  }

  if (baseActivationsAddress_) {
    alignedFree(baseActivationsAddress_);
  }
}

void CPUFunction::execute() {
  auto sym = JIT_->findSymbol("jitmain");
  assert(sym && "Unable to JIT the code!");
  using JitFuncType =
      void (*)(uint8_t * constantWeightVars, uint8_t * mutableWeightVars,
               uint8_t * activations);
  auto address = sym.getAddress();
  if (address) {
    JitFuncType funcPtr = reinterpret_cast<JitFuncType>(address.get());
    funcPtr(runtimeBundle_.constants, baseMutableWeightVarsAddress_,
            baseActivationsAddress_);
  } else {
    GLOW_ASSERT(false && "Error getting address.");
  }
}
