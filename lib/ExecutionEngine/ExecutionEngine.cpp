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
#include "glow/Graph/PlaceholderBindings.h"
#include "glow/Optimizer/Optimizer.h"

#include "llvm/ADT/STLExtras.h"

#include <future>

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
  bool differentKinds = (backend_ == nullptr || backend == nullptr) ||
                        backend->getBackendKind() != backend_->getBackendKind();
  if (ownsBackend_) {
    delete backend_;
  }
  backend_ = backend;
  ownsBackend_ = ownsBackend;
  clear();

  if (differentKinds) {
    if (device_) {
      EXIT_ON_ERR(device_->stop());
      device_.reset();
    }

    if (backend) {
      device_ = std::unique_ptr<runtime::DeviceManager>(
          runtime::DeviceManager::createDeviceManager(
              backend->getBackendKind()));
      EXIT_ON_ERR(device_->init());
    }
  }
}

const Backend *ExecutionEngine::getBackend() const { return backend_; }

ExecutionEngine::~ExecutionEngine() {
  // Call setBackend to make sure that backend_ is deleted if it's owned.
  setBackend(nullptr, /*ownsBackend*/ false);
}

void ExecutionEngine::clear() {
  for (auto &func : compiledFunctions_) {
    device_->evictNetwork(func.first(), [](std::string, llvm::Error err) {
      EXIT_ON_ERR(std::move(err));
    });
  }
  compiledFunctions_.clear();
}

void glow::updateInputPlaceholders(PlaceholderBindings &bindings,
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

void glow::updateInputPlaceholdersByName(PlaceholderBindings &bindings,
                                         Module *mod,
                                         llvm::ArrayRef<llvm::StringRef> ph,
                                         llvm::ArrayRef<Tensor *> inputs) {
  assert(inputs.size() == ph.size() &&
         "The number of inputs does not match the number of Placeholders");

  for (int i = 0, e = ph.size(); i < e; i++) {
    Placeholder *p = mod->getPlaceholderByName(ph[i]);
    Tensor *t = inputs[i];
    assert(t && "Invalid tensor.");
    assert(p && "Invalid placeholder.");
    updateInputPlaceholders(bindings, {p}, {t});
  }
}

void ExecutionEngine::runInternal(ExecutionContext &context,
                                  llvm::StringRef name,
                                  CompiledFunction &compiledFunction) {
  // Make sure that the bindings have backing tensors for all placeholders.
  context.getPlaceholderBindings()->allocate(M_.getPlaceholders());

  std::unique_ptr<ExecutionContext> contextPtr(&context);
  std::promise<void> runPromise;
  auto fut = runPromise.get_future();
  llvm::Error runErr = llvm::Error::success();
  device_->runFunction(
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

void ExecutionEngine::run(ExecutionContext &context) {
  assert(compiledFunctions_.size() == 1 &&
         "Expected exactly one compiled function.");
  runInternal(context, *compiledFunctions_.keys().begin(),
              getCompiledFunction());
}

void ExecutionEngine::run(ExecutionContext &context, llvm::StringRef name) {
  runInternal(context, name, getCompiledFunction(name));
}

void ExecutionEngine::run(PlaceholderBindings &bindings) {
  assert(compiledFunctions_.size() == 1 &&
         "Expected exactly one compiled function.");
  std::unique_ptr<PlaceholderBindings> bindingsPtr(&bindings);
  ExecutionContext context(std::move(bindingsPtr));
  runInternal(context, *compiledFunctions_.keys().begin(),
              getCompiledFunction());
  // don't delete bindings
  context.movePlaceholderBindings().release();
}

void ExecutionEngine::run(PlaceholderBindings &bindings, llvm::StringRef name) {
  std::unique_ptr<PlaceholderBindings> bindingsPtr(&bindings);
  ExecutionContext context(std::move(bindingsPtr));
  runInternal(context, name, getCompiledFunction(name));
  // don't delete bindings
  context.movePlaceholderBindings().release();
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

void ExecutionEngine::insertCompiledFunction(
    llvm::StringRef name, std::unique_ptr<CompiledFunction> func) {
  assert(compiledFunctions_.find(name) == compiledFunctions_.end());

  runtime::FunctionMapTy functionMap;
  functionMap[name] = func.get();
  compiledFunctions_[name] = std::move(func);

  std::promise<llvm::Error> addPromise;
  auto fut = addPromise.get_future();
  device_->addNetwork(&M_, std::move(functionMap),
                      [&addPromise](const Module *, llvm::Error err) {
                        addPromise.set_value(std::move(err));
                      });
  EXIT_ON_ERR(fut.get());
}

void glow::runBatch(ExecutionEngine &EE, PlaceholderBindings &bindings,
                    size_t iterations, size_t &sampleCounter,
                    llvm::ArrayRef<Placeholder *> ph,
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
    EE.run(bindings);
    sampleCounter += batchSize;
  }
}

void ExecutionEngine::compile(CompilationMode mode, Function *F,
                              bool clearOtherFunctions) {
  CompilationOptions opts;
  opts.mode = mode;
  compile(F, opts, clearOtherFunctions);
}

void ExecutionEngine::compile(Function *F, const CompilationOptions &opts,
                              bool clearOtherFunctions) {
  llvm::StringRef name = F->getName();

  if (clearOtherFunctions) {
    clear();
  }

  assert(!compiledFunctions_.count(name) &&
         "A function with this name has already been compiled.");

  backend_->optimizeFunction(F, opts);

  for (const Node &N : F->getNodes()) {
    (void)N;
    assert(backend_->isOpSupported(N) &&
           "Backend must support all nodes after high-level optimizations.");
  }

  auto func = backend_->compile(F, opts);
  insertCompiledFunction(name, std::move(func));
}

void ExecutionEngine::save(Function *F, const CompilationOptions &opts,
                           llvm::StringRef outputDir,
                           llvm::StringRef networkName) {
  backend_->optimizeFunction(F, opts);
  backend_->save(F, outputDir, networkName);
}
