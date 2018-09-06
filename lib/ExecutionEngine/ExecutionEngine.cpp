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
#include "glow/Graph/Graph.h"
#include "glow/IR/IR.h"
#include "glow/IR/IRBuilder.h"
#include "glow/IR/Instrs.h"
#include "glow/Optimizer/Optimizer.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"

using namespace glow;

using llvm::cast;

namespace {
static llvm::cl::opt<std::string>
    dumpIRDAG("dump-ir-dag",
              llvm::cl::desc("Specify the file to export the IR in DOT format"),
              llvm::cl::value_desc("file.dot"));

static llvm::cl::opt<bool> dumpIR("dump-ir",
                                  llvm::cl::desc("Prints IR to stdout"));
} // namespace

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

void ExecutionEngine::run(llvm::ArrayRef<Storage *> vars,
                          llvm::ArrayRef<Tensor *> inputs) {
  assert(function_ && "No function has been compiled");
  assert(inputs.size() == vars.size() &&
         "The number of inputs does not match the number of variables");

  llvm::SmallVector<Placeholder *, 8> placeholders;
  llvm::SmallVector<Tensor *, 8> tensors;

  // Update the input variables, or register placeholders.
  for (int i = 0, e = vars.size(); i < e; i++) {
    // This is a variable. In here we override the content of the tensor with
    // the requested tensor.
    // Note: When we finish the migration to Placeholder this code needs to be
    // removed because it's not threadsafe. See issue #1334.
    if (Variable *V = llvm::dyn_cast<Variable>(vars[i])) {
      assert(V->getVisibilityKind() == VisibilityKind::Public &&
             "Trying to update a private variable");
      loadValueFromTensor(V, inputs[i]);
      continue;
    }

    // This is a placeholder that we need to make concrete during the execution
    // of the program.
    placeholders.push_back(cast<Placeholder>(vars[i]));
    tensors.push_back(inputs[i]);
  }

  function_->execute(placeholders, tensors);
}

void ExecutionEngine::runBatch(size_t iterations,
                               llvm::ArrayRef<Storage *> vars,
                               llvm::ArrayRef<Tensor *> inputs) {
  static size_t trainCounter = 0;

  assert(function_ && "No function has been compiled");
  assert(!inputs.empty() && "No inputs");
  assert(inputs.size() == vars.size() &&
         "The number of inputs does not match the number of variables");

  // This is the size of one batch (the number of samples in the batch).
  size_t batchSize = vars[0]->getType()->dims()[0];

  for (size_t i = 0; i < iterations; i++) {
    // Pick up one slice from the input tensors, and load it into corresponding
    // network Variables. Then, run a single pass over the network.
    updateInputsAndRunNetwork(vars, inputs, trainCounter);

    trainCounter += batchSize;
  }
}

void ExecutionEngine::updateInputsAndRunNetwork(llvm::ArrayRef<Storage *> vars,
                                                llvm::ArrayRef<Tensor *> inputs,
                                                size_t sampleIdx) {
  llvm::SmallVector<Placeholder *, 8> placeholders;

  // This container saves the tensor slices that were extracted from the inputs.
  // The tensors are alive during the lifetime of this function. These tensors
  // must not move, and we must pass them as ArrayRef, so we allocate them here
  // manually.
  llvm::SmallVector<Tensor, 8> tempTensorViews;
  // We need to pass the tensors as array ref, so we reference the static
  // storage above, that's frozen during the execution of the program.
  llvm::SmallVector<Tensor *, 8> tensors;

  // Update the input variables.
  for (int i = 0, e = vars.size(); i < e; i++) {
    // Note: When we finish the migration to Placeholder this code needs to be
    // removed because it's not threadsafe. See issue #1334.
    if (Variable *V = llvm::dyn_cast<Variable>(vars[i])) {
      loadValueFromTensorSlice(V, inputs[i], sampleIdx);
      continue;
    }

    // This is a placeholder that we need to make concrete during the execution
    // of the program.
    placeholders.push_back(cast<Placeholder>(vars[i]));

    // Extract a tensor view from the input values. This has the same
    // functionality as the logic in loadValueFromTensorSlice that copies the
    // content of the tensor.
    auto batchSize = inputs[i]->dims()[0];
    auto startIdx = sampleIdx % batchSize;
    auto phDims = placeholders[i]->dims();

    // The start offset is all zeros except for the batch dimension.
    std::vector<size_t> sliceOffset(inputs[i]->dims().size(), 0);
    sliceOffset[0] = startIdx;

    // Create the tensor view.
    tempTensorViews.push_back(inputs[i]->getUnowned(phDims, sliceOffset));
  }

  // Save the pointer to the allocated tensor view, because the interface
  // requires ArrayRef of tensors.
  for (int i = 0, e = tempTensorViews.size(); i < e; i++) {
    tensors.push_back(&tempTensorViews[i]);
  }

  // Run the network.
  function_->execute(placeholders, tensors);
}

void ExecutionEngine::loadValueFromTensorSlice(Variable *v, Tensor *input,
                                               size_t sampleIdx) {
  assert(v && "Invalid value");
  auto &t = v->getPayload();

  auto dim = input->dims();
  assert(t.dims().drop_front() == dim.drop_front() && "Invalid slice size");
  // Extract the n'th slice, that must be a tensor.
  size_t slc = sampleIdx % dim[0];
  t.copyConsecutiveSlices(input, slc);
}

void ExecutionEngine::loadValueFromTensor(Variable *v, Tensor *input) {
  assert(v && "Invalid value");
  auto &t = v->getPayload();
  auto dim = input->dims();
  (void)dim;
  assert(t.dims() == dim && "Invalid slice size");
  t.assign(input);
}

std::unique_ptr<IRFunction> ExecutionEngine::generateIR(CompilationMode mode,
                                                        Function *F) {
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

  /// Prepare the IR container to handle our function.
  auto IR = llvm::make_unique<IRFunction>(F);

  // Generate IR from the graph.
  IR->generateIR();

  // Optimize the generated IR.
  ::glow::optimize(*IR, mode, *backend_);

  // If requested, dump IR to stdout and/or dot file for debugging.
  if (dumpIR) {
    IR->dump();
  }
  if (!dumpIRDAG.empty()) {
    IR->dumpDAG(dumpIRDAG.getValue());
  }

  return IR;
}

void ExecutionEngine::compile(CompilationMode mode, Function *F) {
  function_ = backend_->compile(generateIR(mode, F));
}

void ExecutionEngine::save(CompilationMode mode, Function *F,
                           llvm::StringRef outputDir,
                           llvm::StringRef networkName) {
  backend_->save(generateIR(mode, F), outputDir, networkName);
}
