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
    : CompiledFunction(runtimeBundle), JIT_(std::move(JIT)) {}

CPUFunction::~CPUFunction() {
  alignedFree(runtimeBundle_.getConstants());
  tearDownRuns();
}

void CPUFunction::setupRuns() {
  if (!runsSetup_) {
    if (runtimeBundle_.getActivationsSize() != 0) {
      baseActivationsAddress_ = (uint8_t *)alignedAlloc(
          runtimeBundle_.getActivationsSize(), TensorAlignment);
    }

    if (runtimeBundle_.getMutableWeightSize() != 0) {
      baseMutableWeightVarsAddress_ = (uint8_t *)alignedAlloc(
          runtimeBundle_.getMutableWeightSize(), TensorAlignment);
    }
    runsSetup_ = true;
  }
}

void CPUFunction::collectConstants(IRFunction *F) {
  runtimeBundle_.collectConstants(F);
}

void CPUFunction::beforeRun(const Context &ctx) {
  // Copy Placeholders into allocated memory.
  for (auto PH : ctx.pairs()) {
    auto payload = PH.second->getUnsafePtr();
    auto symbolInfo = runtimeBundle_.getSymbolInfo(PH.first);
    auto addr = symbolInfo.offset;
    auto numBytes = symbolInfo.size;
    // copy PH to allocated memory.
    memcpy(baseMutableWeightVarsAddress_ + addr, payload, numBytes);
  }
}

void CPUFunction::afterRun(const Context &ctx) {
  // Copy placeholders from device back into context.
  for (auto PH : ctx.pairs()) {
    auto symbolInfo = runtimeBundle_.getSymbolInfo(PH.first);
    auto payload = baseMutableWeightVarsAddress_ + symbolInfo.offset;
    auto numBytes = symbolInfo.size;
    auto addr = PH.second->getUnsafePtr();
    // copy PH from allocated memory.
    memcpy(addr, payload, numBytes);
  }
}

void CPUFunction::tearDownRuns() {
  if (baseMutableWeightVarsAddress_) {
    alignedFree(baseMutableWeightVarsAddress_);
    baseMutableWeightVarsAddress_ = nullptr;
  }

  if (baseActivationsAddress_) {
    alignedFree(baseActivationsAddress_);
    baseActivationsAddress_ = nullptr;
  }
  runsSetup_ = false;
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
    funcPtr(runtimeBundle_.getConstants(), baseMutableWeightVarsAddress_,
            baseActivationsAddress_);
  } else {
    GLOW_ASSERT(false && "Error getting address.");
  }
}
