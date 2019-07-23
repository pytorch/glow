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

#include "glow/ExecutionEngine/ExecutionEngine2.h"
#include "glow/Backend/Backend.h"
#include "glow/Graph/Graph.h"
#include "glow/Graph/PlaceholderBindings.h"
#include "glow/Optimizer/GraphOptimizer/GraphOptimizer.h"

#include "llvm/ADT/STLExtras.h"

#include <future>

using namespace glow;

ExecutionEngine2::ExecutionEngine2(llvm::StringRef backend) {
  setBackendName(backend);
}

/// Set the code generator to the given \p backend.
void ExecutionEngine2::setBackendName(llvm::StringRef backend) {
  module_.reset(new Module);
  rawModule_ = module_.get();
  backendName_ = backend;
  clear();

  if (hostManager_) {
    EXIT_ON_ERR(hostManager_->clearHost());
    hostManager_.reset();
  }

  if (backend != "") {
    std::vector<std::unique_ptr<runtime::DeviceConfig>> configs;
    auto config = llvm::make_unique<runtime::DeviceConfig>(backend);
    if (deviceMemory_) {
      config->setDeviceMemory(deviceMemory_);
    }
    configs.push_back(std::move(config));
    hostManager_ = llvm::make_unique<runtime::HostManager>(std::move(configs));
  }
}

llvm::StringRef ExecutionEngine2::getBackendName() const {
  return backendName_;
}

ExecutionEngine2::~ExecutionEngine2() {
  // Call setBackendName with backend="" to clear the EE.
  setBackendName("");
}

void ExecutionEngine2::clear() {
  if (hostManager_) {
    EXIT_ON_ERR(hostManager_->clearHost());
  }
  compiledFunctions_.clear();
}

void glow::updateInputPlaceholders2(PlaceholderBindings &bindings,
                                    llvm::ArrayRef<Placeholder *> ph,
                                    llvm::ArrayRef<Tensor *> inputs) {
  assert(inputs.size() == ph.size() &&
         "The number of inputs does not match the number of Placeholders");

  for (int i = 0, e = ph.size(); i < e; i++) {
    assert(ph[i] && "Invalid value");
    auto *backingTensor = bindings.get(ph[i]);
    assert(backingTensor && "Can't find the placeholder");
    auto dim = inputs[i]->dims();
    (void)dim;
    assert(backingTensor->getType().isEqual(inputs[i]->getType()) &&
           "Mismatch on Placeholder and Tensor types.");
    backingTensor->assign(inputs[i]);
  }
}

void glow::updateInputPlaceholdersByName2(PlaceholderBindings &bindings,
                                          Module *mod,
                                          llvm::ArrayRef<llvm::StringRef> ph,
                                          llvm::ArrayRef<Tensor *> inputs) {
  assert(inputs.size() == ph.size() &&
         "The number of inputs does not match the number of Placeholders");

  for (int i = 0, e = ph.size(); i < e; i++) {
    Placeholder *p = mod->getPlaceholderByName(legalizeName(ph[i]));
    Tensor *t = inputs[i];
    assert(t && "Invalid tensor.");
    assert(p && "Invalid placeholder.");
    updateInputPlaceholders2(bindings, {p}, {t});
  }
}

void ExecutionEngine2::runInternal(ExecutionContext &context,
                                   llvm::StringRef name) {
  std::unique_ptr<ExecutionContext> contextPtr(&context);
  std::promise<void> runPromise;
  auto fut = runPromise.get_future();
  llvm::Error runErr = llvm::Error::success();
  hostManager_->runNetwork(
      name, std::move(contextPtr),
      [&runPromise, &runErr](runtime::RunIdentifierTy, llvm::Error err,
                             std::unique_ptr<ExecutionContext> contextPtr) {
        // Don't delete context.
        contextPtr.release();
        runErr = std::move(err);
        runPromise.set_value();
      });

  fut.wait();
  EXIT_ON_ERR(std::move(runErr));
}

void ExecutionEngine2::run(ExecutionContext &context) {
  assert(compiledFunctions_.size() == 1 &&
         "Expected exactly one compiled function.");
  runInternal(context, *compiledFunctions_.begin());
}

void ExecutionEngine2::run(ExecutionContext &context, llvm::StringRef name) {
  runInternal(context, name);
}

void ExecutionEngine2::run(PlaceholderBindings &bindings) {
  assert(compiledFunctions_.size() == 1 &&
         "Expected exactly one compiled function.");
  std::unique_ptr<PlaceholderBindings> bindingsPtr(&bindings);
  ExecutionContext context(std::move(bindingsPtr));
  runInternal(context, *compiledFunctions_.begin());
  // Don't delete bindings.
  context.movePlaceholderBindings().release();
}

void ExecutionEngine2::run(PlaceholderBindings &bindings,
                           llvm::StringRef name) {
  std::unique_ptr<PlaceholderBindings> bindingsPtr(&bindings);
  ExecutionContext context(std::move(bindingsPtr));
  runInternal(context, name);
  // Don't delete bindings.
  context.movePlaceholderBindings().release();
}

void glow::runBatch2(ExecutionEngine2 &EE, PlaceholderBindings &bindings,
                     size_t iterations, size_t &sampleCounter,
                     llvm::ArrayRef<Placeholder *> ph,
                     llvm::ArrayRef<Tensor *> inputs, llvm::StringRef name) {
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
      auto *backingTensor = bindings.get(ph[i]);
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
    if (name == "") {
      EE.run(bindings);
    } else {
      EE.run(bindings, name);
    }
    sampleCounter += batchSize;
  }
}

void ExecutionEngine2::compile(CompilationMode mode) {
  CompilationContext cctx;
  cctx.compMode = mode;
  compile(cctx);
}

void ExecutionEngine2::compile(CompilationContext &cctx) {
  assert(module_.get() && "Compile has already been called.");

  for (auto &function : module_->getFunctions()) {
    compiledFunctions_.insert(function->getName());
  }

  EXIT_ON_ERR(hostManager_->addNetwork(std::move(module_), cctx));
}
