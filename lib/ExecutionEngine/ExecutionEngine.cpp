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

ExecutionEngine::ExecutionEngine(BackendKind backendKind)
    : backend_(createBackend(backendKind)) {}

/// Set the code generator kind to \p backendKind.
void ExecutionEngine::setBackend(BackendKind backendKind) {
  backend_.reset(createBackend(backendKind));
  function_.reset();
}

/// Set the code generator kind to \p backend.
void ExecutionEngine::setBackend(Backend *backend) {
  backend_.reset(backend);
  function_.reset();
}

ExecutionEngine::~ExecutionEngine() = default;

void glow::updateVariables(llvm::ArrayRef<Variable *> vars,
                           llvm::ArrayRef<Tensor *> inputs) {
  assert(inputs.size() == vars.size() &&
         "The number of inputs does not match the number of variables");

  // Update the input variables.
  for (int i = 0, e = vars.size(); i < e; i++) {
    assert(vars[i] && "Invalid value");
    assert(vars[i]->getVisibilityKind() == VisibilityKind::Public &&
           "Trying to update a private variable");
    auto &t = vars[i]->getPayload();
    auto dim = inputs[i]->dims();
    (void)dim;
    assert(t.dims() == dim &&
           t.getElementType() == inputs[i]->getElementType() &&
           "Mismatch on Variable and Tensor types.");
    t.assign(inputs[i]);
  }
}

void glow::updateVariables(Context &ctx, llvm::ArrayRef<Placeholder *> ph,
                           llvm::ArrayRef<Tensor *> inputs) {
  assert(inputs.size() == ph.size() &&
         "The number of inputs does not match the number of Placeholders");

  // Update the input variables.
  for (int i = 0, e = ph.size(); i < e; i++) {
    assert(ph[i] && "Invalid value");
    auto *backingTensor = ctx.get(ph[i]);
    assert(backingTensor && "Can't find the placeholder");
    auto dim = inputs[i]->dims();
    (void)dim;
    assert(backingTensor->getType().isEqual(inputs[i]->getType()) &&
           "Mismatch on Variable and Tensor types.");
    backingTensor->assign(inputs[i]);
  }
}

void glow::updateInputsByName(Context &ctx, Module *mod,
                              llvm::ArrayRef<llvm::StringRef> ph,
                              llvm::ArrayRef<Tensor *> inputs) {
  assert(inputs.size() == ph.size() &&
         "The number of inputs does not match the number of Placeholders");

  for (int i = 0, e = ph.size(); i < e; i++) {
    Placeholder *p = mod->getPlaceholderByName(ph[i]);
    Tensor *t = inputs[i];
    assert(t && "Invalid tensor.");
    assert(p && "Invalid placeholder.");
    updateVariables(ctx, {p}, {t});
  }
}

void ExecutionEngine::run() {
  assert(function_ && "No function has been compiled");
  function_->execute();
}

/// Update the content of the tensors \p vars with some slices that are from \p
/// inputs. The data starts at slice \p sampleIdx and wraps around until the
/// data in \p v is filled. All dimensions, except for the first (batch)
/// dimension must be identical.
void glow::updateVariablesFromBatch(llvm::ArrayRef<Variable *> vars,
                                    llvm::ArrayRef<Tensor *> inputs,
                                    size_t sampleIdx) {
  assert(!inputs.empty() && "No inputs");
  assert(inputs.size() == vars.size() &&
         "The number of inputs does not match the number of variables");

  // Update the input variables.
  for (int i = 0, e = vars.size(); i < e; i++) {
    assert(vars[i] && "Invalid value");
    auto &t = vars[i]->getPayload();

    auto dim = inputs[i]->dims();
    assert(t.dims().drop_front() == dim.drop_front() && "Invalid slice size");
    // Extract the n'th slice, that must be a tensor.
    size_t slc = sampleIdx % dim[0];
    t.copyConsecutiveSlices(inputs[i], slc);
  }
}

void glow::runBatch(ExecutionEngine &EE, size_t iterations,
                    size_t &sampleCounter, llvm::ArrayRef<Variable *> vars,
                    llvm::ArrayRef<Tensor *> inputs) {
  // This is the size of one batch (the number of samples in the batch).
  size_t batchSize = vars[0]->getType()->dims()[0];

  for (size_t i = 0; i < iterations; i++) {
    // Pick up one slice from the input tensors, and load it into corresponding
    // network Variables. Then, run a single pass over the network.
    glow::updateVariablesFromBatch(vars, inputs, sampleCounter);

    // Run the network.
    EE.run();
    sampleCounter += batchSize;
  }
}

void glow::runBatch(ExecutionEngine &EE, Context &ctx, size_t iterations,
                    size_t &sampleCounter, llvm::ArrayRef<Placeholder *> ph,
                    llvm::ArrayRef<Tensor *> inputs) {
  // This is the size of one batch (the number of samples in the batch).
  size_t batchSize = ph[0]->getType()->dims()[0];

  assert(!inputs.empty() && "No inputs");
  assert(inputs.size() == ph.size() &&
         "The number of inputs does not match the number of variables");

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
    EE.run();
    sampleCounter += batchSize;
  }
}

void ExecutionEngine::optimizeFunction(CompilationMode mode, Function *F) {
  // Verify the function pre-optimization/lowering.
  F->verify();

  // Optimize the graph.
  ::glow::optimize(F, mode);

  // Allow the backend to transform the graph prior to lowering.
  if (backend_->transformPreLowering(F, mode)) {
    // Optimize the graph again after the backend transformation.
    // In particular, DCE is very likely to be useful.
    ::glow::optimize(F, mode);
  }

  // Lower the graph into a sequence of low-level linear algebra operations.
  ::glow::lower(F, *backend_);

  // Optimize the graph again.
  ::glow::optimize(F, mode);

  // Allow the backend to transform the graph after lowering.
  if (backend_->transformPostLowering(F, mode)) {
    // Optimize the graph again after the backend transformation.
    // In particular, DCE is very likely to be useful.
    ::glow::optimize(F, mode);
  }
}

void ExecutionEngine::compile(CompilationMode mode, Function *F, Context &ctx) {
  optimizeFunction(mode, F);
  // Make sure that the context has backing tensors for all placeholders.
  ctx.allocate(M_.getPlaceholders());
  function_ = backend_->compile(F, ctx);
}

void ExecutionEngine::save(CompilationMode mode, Function *F,
                           llvm::StringRef outputDir,
                           llvm::StringRef networkName) {
  optimizeFunction(mode, F);
  backend_->save(F, outputDir, networkName);
}
