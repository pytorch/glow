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

#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Backend/Backend.h"
#include "glow/Graph/Graph.h"
#include "glow/Graph/PlaceholderBindings.h"
#include "glow/Optimizer/GraphOptimizer/GraphOptimizer.h"
#include "glow/Support/Error.h"

#include "llvm/ADT/STLExtras.h"

#include <future>

using namespace glow;

ExecutionEngine::ExecutionEngine(llvm::StringRef backend, uint64_t deviceMemory,
                                 bool ignoreUserDeviceConfig,
                                 unsigned numDevices)
    : deviceMemory_(deviceMemory),
      ignoreUserDeviceConfig_(ignoreUserDeviceConfig) {
  setBackendName(backend, numDevices);
}

/// Set the code generator to the given \p backend.
void ExecutionEngine::setBackendName(llvm::StringRef backend,
                                     size_t numDevices) {
  clear();
  module_.reset(new Module);
  rawModule_ = module_.get();
  backendName_ = backend.str();

  if (hostManager_) {
    EXIT_ON_ERR(hostManager_->clearHost());
    hostManager_.reset();
  }

  std::vector<std::unique_ptr<runtime::DeviceConfig>> configs;
  if (!ignoreUserDeviceConfig_ &&
      loadDeviceConfigsFromFile(configs, deviceMemory_)) {
    // Warning if there is more than a single device configured.
    if (configs.size() != 1) {
      LOG(WARNING) << "Found " << configs.size()
                   << " devices configured for the ExecutionEngine";
    }
    // Verify that all configured devices match the expected backend name.
    CHECK(backendName_ == configs[0]->backendName)
        << "Expected backend name to match the ExecutionEngine";
  } else {
    for (size_t i = 0; i < numDevices; i++) {
      auto config = glow::make_unique<runtime::DeviceConfig>(backendName_);
      if (deviceMemory_) {
        config->setDeviceMemory(deviceMemory_);
      }
      configs.push_back(std::move(config));
    }
  }
  hostManager_ = glow::make_unique<runtime::HostManager>(std::move(configs));
}

llvm::StringRef ExecutionEngine::getBackendName() const { return backendName_; }

Function *ExecutionEngine::getSingleFunctionFromModule() const {
  auto &fList = getModule().getFunctions();
  assert(fList.size() == 1 && "More than one Function in Module.");
  return *fList.begin();
}

ExecutionEngine::~ExecutionEngine() { clear(); }

void ExecutionEngine::clear() {
  if (hostManager_) {
    EXIT_ON_ERR(hostManager_->clearHost());
  }
  compiledFunctions_.clear();
  module_.reset(nullptr);
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
    Placeholder *p = mod->getPlaceholderByNameSlow(legalizeName(ph[i]));
    Tensor *t = inputs[i];
    assert(t && "Invalid tensor.");
    assert(p && "Invalid placeholder.");
    updateInputPlaceholders(bindings, {p}, {t});
  }
}

void ExecutionEngine::runInternal(ExecutionContext &context,
                                  llvm::StringRef name) {
  std::unique_ptr<ExecutionContext> contextPtr(&context);
  std::unique_ptr<ExecutionContext> contextOut;
  std::promise<void> runPromise;
  auto fut = runPromise.get_future();
  Error runErr = Error::empty();
  hostManager_->runNetwork(name, std::move(contextPtr),
                           [&runPromise, &runErr, &contextOut](
                               runtime::RunIdentifierTy, Error err,
                               std::unique_ptr<ExecutionContext> contextPtr) {
                             contextOut = std::move(contextPtr);
                             runErr = std::move(err);
                             runPromise.set_value();
                           });

  fut.wait();
  if (ensureOutputsOnHost_) {
    contextOut->getPlaceholderBindings()->ensureOnHost();
  }
  // Don't delete context.
  contextOut.release();
  EXIT_ON_ERR(std::move(runErr));
}

void ExecutionEngine::run(ExecutionContext &context) {
  assert((compiledFunctions_.size() == 1 || allowMultiFunction_) &&
         "Expected exactly one compiled function.");
  runInternal(context, *compiledFunctions_.begin());
}

void ExecutionEngine::run(ExecutionContext &context, llvm::StringRef name) {
  runInternal(context, name);
}

void ExecutionEngine::run(PlaceholderBindings &bindings) {
  assert((compiledFunctions_.size() == 1 || allowMultiFunction_) &&
         "Expected exactly one compiled function.");
  std::unique_ptr<PlaceholderBindings> bindingsPtr(&bindings);
  ExecutionContext context(std::move(bindingsPtr));
  runInternal(context, *compiledFunctions_.begin());
  // Don't delete bindings.
  context.movePlaceholderBindings().release();
}

void ExecutionEngine::run(PlaceholderBindings &bindings, llvm::StringRef name) {
  std::unique_ptr<PlaceholderBindings> bindingsPtr(&bindings);
  ExecutionContext context(std::move(bindingsPtr));
  runInternal(context, name);
  // Don't delete bindings.
  context.movePlaceholderBindings().release();
}

void glow::runBatch(ExecutionEngine &EE, PlaceholderBindings &bindings,
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

void glow::evalBatch(
    ExecutionEngine &EE, PlaceholderBindings &bindings, size_t numMinibatchRuns,
    size_t &sampleCounter, Placeholder *inputPH, Placeholder *outputPH,
    Tensor &samplesInput, Tensor &labelsInput, llvm::StringRef name,
    std::function<void(const Tensor &sampleIn, const Tensor &sampleOut,
                       const Tensor &label, size_t sampleIndex)> &&cb) {
  // The number of samples in a minibatch (a single function run)
  size_t minibatchSize = inputPH->getType()->dims()[0];

  assert(samplesInput.dims()[0] == labelsInput.dims()[0] &&
         "The number of sample inputs does not match the number of labels");

  auto LIH = labelsInput.getHandle<int64_t>();

  // For each iteration in the batch:
  for (size_t j = 0; j < numMinibatchRuns; j++) {
    assert(inputPH && "Invalid value");
    auto *backingTensor = bindings.get(inputPH);
    assert(backingTensor && "Can't find the backing tensor");
    auto dim = samplesInput.dims();
    assert(backingTensor->dims().drop_front() == dim.drop_front() &&
           "Invalid slice size");
    // Extract the n'th slice, that must be a tensor.
    size_t slc = sampleCounter % dim[0];
    // Pick up one slice from the input tensors, and load it into the
    // corresponding network Placeholder.
    backingTensor->copyConsecutiveSlices(&samplesInput, slc);

    // Run the network.
    if (name == "") {
      EE.run(bindings);
    } else {
      EE.run(bindings, name);
    }

    for (unsigned i = 0; i < minibatchSize; i++) {
      auto sampleInputTensor = backingTensor->getHandle().extractSlice(i);
      auto sampleOutputTensor =
          bindings.get(outputPH)->getHandle().extractSlice(i);
      // If index is out of bounds of samples/labels first dimension, it is
      // wrapped around to be consistent with "copyConsecutiveSlices".
      auto labelTensor = LIH.extractSlice((sampleCounter + i) % dim[0]);

      cb(sampleInputTensor, sampleOutputTensor, labelTensor, sampleCounter + i);
    }
    sampleCounter += minibatchSize;
  }
}

void ExecutionEngine::compile(CompilationMode mode) {
  CompilationContext cctx;
  cctx.compMode = mode;
  if (skipModuleStrip_) {
    cctx.skipModuleStrip = true;
  }
  compile(cctx);
}

void ExecutionEngine::compile(CompilationContext &cctx) {
  assert(module_.get() && "Compile has already been called.");

  if (skipModuleStrip_) {
    cctx.skipModuleStrip = true;
  }

  if (cctx.prepartitionedConfig) {
    if (cctx.prepartitionedConfig->funcs.size() > 1) {
      allowMultiFunction_ = true;
    }

    // If we are compiling a prepartitioned model then we should add the name of
    // the prepartitioned config, which is later used as the root of the DAG,
    // and also used to kick off running.
    compiledFunctions_.insert(cctx.prepartitionedConfig->funcName);
  }

  for (auto *F : module_->getFunctions()) {
    // Check to see if this Function is part of the prepartitioned config if it
    // exists, as we do not kick off execution for the partitions individually.
    bool skipAdding = false;
    if (cctx.prepartitionedConfig) {
      auto &pFuns = cctx.prepartitionedConfig->funcs;
      skipAdding = std::find(pFuns.begin(), pFuns.end(), F) != pFuns.end();
    }
    if (!skipAdding) {
      compiledFunctions_.insert(F->getName().str());
    }
  }

  EXIT_ON_ERR(hostManager_->addNetwork(std::move(module_), cctx));
}

Backend &ExecutionEngine::getBackend(llvm::StringRef backendName) const {
  return hostManager_->getBackend(backendName);
}

Backend &ExecutionEngine::getBackend() const {
  return hostManager_->getBackend(backendName_);
}
