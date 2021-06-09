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
#include "glow/LLVMIRCodeGen/LLVMCompiledFunction.h"

#include "glow/Graph/PlaceholderBindings.h"
#include "glow/Support/Compiler.h"
#include "glow/Support/Memory.h"
#include "glow/Support/ThreadPool.h"

using namespace glow;

LLVMCompiledFunction::LLVMCompiledFunction(
    std::unique_ptr<GlowJIT> JIT, runtime::RuntimeBundle &&runtimeBundle)
    : CompiledFunction(std::move(runtimeBundle)), JIT_(std::move(JIT)) {}

void LLVMCompiledFunction::collectConstants(const Module *module) {
  runtimeBundle_.collectConstants(module);
}

void LLVMCompiledFunction::loadPlaceholders(
    PlaceholderBindings *bindings, uint8_t *baseMutableWeightVarsAddress) {
  // Make sure our inputs are on the host.
  bindings->ensureOnHost();

  // Copy Placeholders into allocated memory.
  auto &symbolTable = runtimeBundle_.getSymbolTable();
  for (auto &PH : bindings->pairs()) {
    auto it = symbolTable.find(PH.first->getName().str());
    if (it == symbolTable.end()) {
      continue;
    }
    assert(!PH.second.isDeviceResident());
    auto symbolInfo = it->second;
    auto payload = PH.second.getUnsafePtr();
    auto addr = symbolInfo.offset;
    auto numBytes = PH.second.getUnpaddedSizeInBytes();
    // copy PH to allocated memory.
    memcpy(baseMutableWeightVarsAddress + addr, payload, numBytes);
  }
}

void LLVMCompiledFunction::updatePlaceholders(
    PlaceholderBindings *bindings, uint8_t *baseMutableWeightVarsAddress) {
  // Copy placeholders from device back into bindings.
  auto &symbolTable = runtimeBundle_.getSymbolTable();
  for (auto &PH : bindings->pairs()) {
    auto it = symbolTable.find(PH.first->getName().str());
    if (it == symbolTable.end()) {
      continue;
    }
    auto symbolInfo = it->second;
    auto payload = baseMutableWeightVarsAddress + symbolInfo.offset;
    auto numBytes = PH.second.getUnpaddedSizeInBytes();
    auto addr = PH.second.getUnsafePtr();
    // copy PH from allocated memory.
    memcpy(addr, payload, numBytes);
  }
}

Error LLVMCompiledFunction::execute(ExecutionContext *context) {
  uint8_t *baseActivationsAddress{nullptr};

  /// Base address for Mutable weights memory block, Inputs and Outputs.
  uint8_t *baseMutableWeightVarsAddress{nullptr};

  {
    TRACE_EVENT_SCOPE(context, TraceLevel::RUNTIME, "allocBuffers");
    if (runtimeBundle_.getActivationsSize() != 0) {
      baseActivationsAddress = (uint8_t *)alignedAlloc(
          runtimeBundle_.getActivationsSize(), TensorAlignment);
    }

    if (runtimeBundle_.getMutableWeightSize() != 0) {
      baseMutableWeightVarsAddress = (uint8_t *)alignedAlloc(
          runtimeBundle_.getMutableWeightSize(), TensorAlignment);
    }
  }

  {
    TRACE_EVENT_SCOPE(context, TraceLevel::RUNTIME, "loadPlaceholders");
    loadPlaceholders(context->getPlaceholderBindings(),
                     baseMutableWeightVarsAddress);
  }

  auto *traceContext = context->getTraceContext();
  TRACE_EVENT_SCOPE_NAMED(traceContext, TraceLevel::RUNTIME,
                          "findJitmainSymbol", fjEvent);
  Expected<llvm::JITTargetAddress> address = NULL;
  {
    std::lock_guard<std::mutex> lock(JITLock_);
    auto sym = JIT_->findSymbol("jitmain");

    DCHECK(sym) << "Unable to JIT the code!";
    // We know address is success since we just made it. Mark it as checked.
    if (address) {
      auto addrOrLLVMError = sym.getAddress();
      if (addrOrLLVMError) {
        address = addrOrLLVMError.get();
      } else {
        address = MAKE_ERR(
            strFormat("Failed to get address: %s",
                      llvm::toString(addrOrLLVMError.takeError()).data()));
      }
    }
  }
  using JitFuncType =
      void (*)(uint8_t * constantWeightVars, uint8_t * mutableWeightVars,
               uint8_t * activations);
  if (address) {
    JitFuncType funcPtr = reinterpret_cast<JitFuncType>(address.get());
    TRACE_EVENT_SCOPE_END_NAMED(fjEvent);
    TRACE_EVENT_SCOPE(traceContext, TraceLevel::RUNTIME, "execute");
    funcPtr(runtimeBundle_.getConstants(), baseMutableWeightVarsAddress,
            baseActivationsAddress);
  } else {
    return MAKE_ERR("Error getting address");
  }

  {
    TRACE_EVENT_SCOPE(context, TraceLevel::RUNTIME, "updatePlaceholders");
    updatePlaceholders(context->getPlaceholderBindings(),
                       baseMutableWeightVarsAddress);
  }

  {
    TRACE_EVENT_SCOPE(context, TraceLevel::RUNTIME, "freeBuffers");
    alignedFree(baseMutableWeightVarsAddress);
    alignedFree(baseActivationsAddress);
  }

  {
    TRACE_EVENT_SCOPE(context, TraceLevel::RUNTIME, "processInstrumentation");
    translateTraceEvents(context);
  }

  return Error::success();
}

void LLVMCompiledFunction::translateTraceEvents(
    ExecutionContext *context) const {
  auto &traceInfo = getTraceInfo();
  if (!traceInfo.enabled) {
    return;
  }

  TraceContext *traceContext = context->getTraceContext();

  if (!traceContext->shouldLog(TraceLevel::OPERATOR)) {
    return;
  }

  PlaceholderBindings *bindings = context->getPlaceholderBindings();

  int tid = threads::getThreadId();
  for (auto &backing : traceInfo.events) {
    Tensor *backingTensor = bindings->get(backing.first);
    DCHECK(backingTensor) << "Could not get backing tensor for Placeholder: "
                          << backing.first->getName().str();

    auto &traceEvents = traceContext->getTraceEvents();
    for (const TraceInfo::Event &event : backing.second) {
      // If it's a complete event grab both timestamps.
      if (event.type == TraceEvent::CompleteType) {
        uint64_t start{0}, end{0};
        memcpy(&start,
               backingTensor->getUnsafePtr() +
                   (event.startIndex * traceInfo.dataSize),
               traceInfo.dataSize);
        memcpy(&end,
               backingTensor->getUnsafePtr() +
                   (event.endIndex * traceInfo.dataSize),
               traceInfo.dataSize);
        traceEvents.push_back({event.name,
                               TraceLevel::OPERATOR,
                               start,
                               end - start,
                               tid,
                               {{"kind", event.kind}}});
      } else {
        uint64_t ts{0};
        memcpy(&ts,
               backingTensor->getUnsafePtr() +
                   (event.startIndex * traceInfo.dataSize),
               traceInfo.dataSize);
        traceEvents.push_back({event.name,
                               TraceLevel::OPERATOR,
                               ts,
                               event.type,
                               tid,
                               {{"kind", event.kind}}});
      }
    }
  }
}
