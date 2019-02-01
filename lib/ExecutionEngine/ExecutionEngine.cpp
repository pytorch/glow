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

#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Backends/Backend.h"
#include "glow/Graph/Context.h"
#include "glow/Graph/Graph.h"
#include "glow/Optimizer/Optimizer.h"

#include "llvm/ADT/STLExtras.h"

using namespace glow;

ExecutionEngine::ExecutionEngine(BackendKind backendKind) {
  setBackend(backendKind);
}

/// Set the code generator kind to \p backendKind.
void ExecutionEngine::setBackend(BackendKind backendKind) {
  setBackend(createBackend(backendKind));
}

/// Set the code generator to the given \p backend.
void ExecutionEngine::setBackend(Backend *backend, bool ownsBackend) {
  if (ownsBackend_) {
    delete backend_;
  }
  backend_ = backend;
  ownsBackend_ = ownsBackend;
  compiledFunctions_.clear();
}

const Backend *ExecutionEngine::getBackend() const { return backend_; }

ExecutionEngine::~ExecutionEngine() {
  // Call setBackend to make sure that backend_ is deleted if it's owned.
  setBackend(nullptr, /*ownsBackend*/ false);
}

void glow::updateInputPlaceholders(Context &ctx,
                                   llvm::ArrayRef<Placeholder *> ph,
                                   llvm::ArrayRef<Tensor *> inputs) {
  assert(inputs.size() == ph.size() &&
         "The number of inputs does not match the number of Placeholders");

  for (int i = 0, e = ph.size(); i < e; i++) {
    assert(ph[i] && "Invalid value");
    auto *backingTensor = ctx.get(ph[i]);
    assert(backingTensor && "Can't find the placeholder");
    auto dim = inputs[i]->dims();
    (void)dim;
    assert(backingTensor->getType().isEqual(inputs[i]->getType()) &&
           "Mismatch on Placeholder and Tensor types.");
    backingTensor->assign(inputs[i]);
  }
}

void glow::updateInputPlaceholdersByName(Context &ctx, Module *mod,
                                         llvm::ArrayRef<llvm::StringRef> ph,
                                         llvm::ArrayRef<Tensor *> inputs) {
  assert(inputs.size() == ph.size() &&
         "The number of inputs does not match the number of Placeholders");

  for (int i = 0, e = ph.size(); i < e; i++) {
    Placeholder *p = mod->getPlaceholderByName(ph[i]);
    Tensor *t = inputs[i];
    assert(t && "Invalid tensor.");
    assert(p && "Invalid placeholder.");
    updateInputPlaceholders(ctx, {p}, {t});
  }
}

void ExecutionEngine::runInternal(Context &ctx,
                                  CompiledFunction &compiledFunction) {
  // Make sure that the context has backing tensors for all placeholders.
  ctx.allocate(M_.getPlaceholders());
  compiledFunction.setupRuns();
  compiledFunction.beforeRun(ctx);
  compiledFunction.execute(&ctx);
  compiledFunction.afterRun(ctx);
}

void ExecutionEngine::run(Context &ctx) {
  runInternal(ctx, getCompiledFunction());
}

void ExecutionEngine::run(Context &ctx, llvm::StringRef name) {
  runInternal(ctx, getCompiledFunction(name));
}

CompiledFunction &ExecutionEngine::getCompiledFunction() {
  assert(compiledFunctions_.size() == 1 &&
         "Expected exactly one compiled function.");
  return *compiledFunctions_.begin()->second;
}

CompiledFunction &ExecutionEngine::getCompiledFunction(llvm::StringRef name) {
  auto functionIt = compiledFunctions_.find(name);
  assert(functionIt != compiledFunctions_.end() &&
         "Could not find a compiled function with the given name.");
  return *functionIt->second;
}

void glow::runBatch(ExecutionEngine &EE, Context &ctx, size_t iterations,
                    size_t &sampleCounter, llvm::ArrayRef<Placeholder *> ph,
                    llvm::ArrayRef<Tensor *> inputs) {
  // This is the size of one batch (the number of samples in the batch).
  size_t batchSize = ph[0]->getType()->dims()[0];

  assert(!inputs.empty() && "No inputs");
  assert(inputs.size() == ph.size() &&
         "The number of inputs does not match the number of placeholders");

  // For each iteration in the batch:
  for (size_t j = 0; j < iterations; j++) {

    // Update the input placeholders.
    for (int i = 0, e = ph.size(); i < e; i++) {
      assert(ph[i] && "Invalid value");
      auto *backingTensor = ctx.get(ph[i]);
      assert(backingTensor && "Can't find the backing tensor");

      auto dim = inputs[i]->dims();
      assert(backingTensor->dims().drop_front() == dim.drop_front() &&
             "Invalid slice size");
      // Extract the n'th slice, that must be a tensor.
      size_t slc = sampleCounter % dim[0];
      // Pick up one slice from the input tensors, and load it into the
      // corresponding network Placeholder.
      backingTensor->copyConsecutiveSlices(inputs[i], slc);
    }

    // Run the network.
    EE.run(ctx);
    sampleCounter += batchSize;
  }
}

void ExecutionEngine::compile(CompilationMode mode, Function *F) {
  backend_->optimizeFunction(mode, F);
  compiledFunctions_.clear();
  compiledFunctions_[F->getName()] = backend_->compile(F);
}

void ExecutionEngine::compile(CompilationMode mode, Function *F,
                              llvm::StringRef name) {
  assert(!compiledFunctions_.count(name) &&
         "A function with this name has already been compiled.");
  compiledFunctions_[name] = backend_->compile(F);
}

void ExecutionEngine::save(CompilationMode mode, Function *F,
                           llvm::StringRef outputDir,
                           llvm::StringRef networkName) {
  backend_->optimizeFunction(mode, F);
  backend_->save(F, outputDir, networkName);
}
