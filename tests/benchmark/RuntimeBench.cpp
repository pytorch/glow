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

#include "benchmark/benchmark.h"

#include "glow/Backends/DeviceManager.h"
#include "glow/Optimizer/GraphOptimizer/GraphOptimizer.h"
#include "glow/Runtime/Executor/ThreadPoolExecutor.h"
#include "glow/Runtime/HostManager/HostManager.h"

#include "CPUBackend.h"

#include <future>

using namespace glow;
using namespace glow::runtime;

//===--------------------------------------------------------------------===//
//              Benchmark Declaration and Instantiation Macros              //
//===--------------------------------------------------------------------===//

/// Declare a subclass of ExecutorBenchmark and override its setUpModule and
/// setUpDAG methods with the given moduleCreator and dagCreator functions.
#define DECLARE_EXECUTOR_BENCHMARK(name, moduleCreator, dagCreator)            \
  template <typename BackendTy>                                                \
  class name##ExecutorBenchmark : public ExecutorBenchmark<BackendTy> {        \
  protected:                                                                   \
    void setUpModule(benchmark::State &state) override {                       \
      this->mod_ = moduleCreator();                                            \
    }                                                                          \
    void setUpDAG(benchmark::State &state) override {                          \
      this->dag_ = dagCreator(this->deviceManagersFunctions_[0]);              \
    }                                                                          \
  };

/// Declare a subclass of an arbitrary Benchmark class and override its
/// setUpModule method with the given moduleCreator function.
#define DECLARE_RUNTIME_COMPONENT_BENCHMARK(name, moduleCreator, component)    \
  template <typename BackendTy>                                                \
  class name##component##Benchmark : public component##Benchmark<BackendTy> {  \
  protected:                                                                   \
    void setUpModule(benchmark::State &state) override {                       \
      this->mod_ = moduleCreator();                                            \
    }                                                                          \
  };

/// Declare subclasses of all RuntimeBenchmark subclasses with appropriate
/// overrides.
#define DECLARE_RUNTIME_BENCHMARK(name, moduleCreator, dagCreator)             \
  DECLARE_RUNTIME_COMPONENT_BENCHMARK(name, moduleCreator, HostManager)        \
  DECLARE_EXECUTOR_BENCHMARK(name, moduleCreator, dagCreator)                  \
  DECLARE_RUNTIME_COMPONENT_BENCHMARK(name, moduleCreator, DeviceManager)

/// Define a RuntimeBenchmark subclass declared using
/// DECLARE_XXX_BENCHMARK for a specific backend and component. This instance
/// calls RuntimeBenchmark::runBenchmark to run the benchmark.
#define INSTANTIATE_RUNTIME_COMPONENT_BENCHMARK(name, backend, component)      \
  BENCHMARK_TEMPLATE_DEFINE_F(name##component##Benchmark, component##backend,  \
                              backend)                                         \
  (benchmark::State & state) { runBenchmark(state); }                          \
  BENCHMARK_REGISTER_F(name##component##Benchmark, component##backend)         \
      ->Unit(benchmark::kMicrosecond);

/// Define RuntimeBenchmark subclasses for all runtime components.
#define INSTANTIATE_RUNTIME_BENCHMARK(name, backend)                           \
  INSTANTIATE_RUNTIME_COMPONENT_BENCHMARK(name, backend, HostManager)          \
  INSTANTIATE_RUNTIME_COMPONENT_BENCHMARK(name, backend, Executor)             \
  INSTANTIATE_RUNTIME_COMPONENT_BENCHMARK(name, backend, DeviceManager)

//===--------------------------------------------------------------------===//
//                       Common Utility Functions                           //
//===--------------------------------------------------------------------===//

/// Set up a DeviceManager instance by instantiating it, compiling all the
/// functions in \p mod using \p backend and adding them to the DeviceManager
/// instance. A reference to the configured DeviceManager instance is returned
/// in \p deviceManager, and all CompiledFunctions creating during compilation
/// are returned in \p deviceManagerFunctions.
void setUpDeviceManagerCommon(
    benchmark::State &state, std::unique_ptr<Backend> &backend,
    std::unique_ptr<Module> &mod, std::unique_ptr<DeviceManager> &deviceManager,
    std::unordered_map<std::string, std::unique_ptr<CompiledFunction>>
        &deviceManagerFunctions) {

  // Check that the backend is valid.
  if (!backend) {
    state.SkipWithError("Unable to set up DeviceManager - backend not set up!");
    return;
  }

  // Check that the module is valid.
  if (!mod) {
    state.SkipWithError("Unable to set up DeviceManager - module not set up!");
    return;
  }

  // Create and initialize the DeviceManager instance.
  deviceManager =
      std::unique_ptr<DeviceManager>(DeviceManager::createDeviceManager(
          DeviceConfig(backend->getBackendName())));
  bool error = errToBool(deviceManager->init());

  if (error) {
    state.SkipWithError("Unable to set up DeviceManager - failed to "
                        "initialize DeviceManager!");
    return;
  }

  FunctionMapTy funcs;
  CompilationContext cctx;

  // Compile all functions in the module.
  for (auto *function : mod->getFunctions()) {
    EXIT_ON_ERR(::glow::optimizeFunction(function, *backend, cctx));
    std::unique_ptr<CompiledFunction> compiledFunction =
        EXIT_ON_ERR(backend->compile(function));
    funcs.insert(std::make_pair(function->getName(), compiledFunction.get()));
    deviceManagerFunctions.insert(
        std::make_pair(function->getName(), std::move(compiledFunction)));
  }

  // Add all compiled functions to the DeviceManager instance.
  std::promise<bool> promise;
  std::future<bool> future = promise.get_future();
  deviceManager->addNetwork(
      mod.get(), funcs, [&promise](const Module * /*mod*/, llvm::Error err) {
        promise.set_value(errToBool(std::move(err)));
      });
  future.wait();
  error = future.get();

  if (error) {
    state.SkipWithError(
        "Unable to set up DeviceManager - failed to add functions!");
  }
}

/// Tear down a DeviceManager instance (\p deviceManager) by evicted all
/// functions added to it and shutting down the device. \p
/// deviceManagerFunctions contains the names of all resident functions.
void tearDownDeviceManagerCommon(
    benchmark::State &state, std::unique_ptr<DeviceManager> &deviceManager,
    std::unordered_map<std::string, std::unique_ptr<CompiledFunction>>
        &deviceManagerFunctions) {
  // Check that the DeviceManager is valid.
  if (!deviceManager) {
    state.SkipWithError(
        "Unable to tear down DeviceManager - DeviceManager not set up!");
  }

  // Evict all functions from the DeviceManager instance.
  for (const auto &func : deviceManagerFunctions) {
    std::promise<bool> promise;
    std::future<bool> future = promise.get_future();
    deviceManager->evictNetwork(
        func.first, [&promise](std::string /*name*/, llvm::Error err) {
          promise.set_value(errToBool(std::move(err)));
        });
    future.wait();
    bool error = future.get();

    if (error) {
      state.SkipWithError("Unable to tear down DeviceManager - could not "
                          "evict all functions!");
    }
  }

  deviceManagerFunctions.clear();

  // Stop the device.
  bool error = errToBool(deviceManager->stop());
  if (error) {
    state.SkipWithError("Unable to tear down DeviceManager - failed to stop "
                        "DeviceManager!");
  }
}

//===--------------------------------------------------------------------===//
//                  Benchmark Template Fixture Classes                      //
//===--------------------------------------------------------------------===//

/// Abstract base class for all runtime benchmarks. Other than definining the
/// benchmark interface, this class contains the Backend, Module and
/// ExecutionContext instances used by all benchmark types.
template <typename BackendTy>
class RuntimeBenchmark : public benchmark::Fixture {
public:
  /// Set up to run the benchmark.
  void SetUp(benchmark::State &state) override {
    setUpBackend(state);
    setUpModule(state);
    setUpExecutionContext(state);
  }

  /// Tear down after the benchmark has run.
  void TearDown(benchmark::State &state) override {
    tearDownExecutionContext(state);
    tearDownModule(state);
    tearDownBackend(state);
  }

protected:
  std::unique_ptr<Backend> &getBackend() { return backend_; }
  void setUpBackend(benchmark::State &state) {
    backend_ = llvm::make_unique<BackendTy>();
  }
  virtual void tearDownBackend(benchmark::State &state) {}

  std::unique_ptr<Module> &getModule() { return mod_; }
  /// Create the module that will be used for the benchmark.
  virtual void setUpModule(benchmark::State &state) = 0;
  virtual void tearDownModule(benchmark::State &state) {}

  std::unique_ptr<ExecutionContext> &getExecutionContext() { return ctx_; }
  virtual void setUpExecutionContext(benchmark::State &state) {
    // Check that the module is valid.
    if (!mod_) {
      state.SkipWithError(
          "Unable to set up execution context - module not set up!");
      return;
    }

    // Allocate all Placeholders in mod_ and move the bindings into an
    // ExecutionContext object.
    auto bindings = llvm::make_unique<PlaceholderBindings>();
    bindings->allocate(mod_->getPlaceholders());
    ctx_ = llvm::make_unique<ExecutionContext>(std::move(bindings));
  }
  virtual void tearDownExecutionContext(benchmark::State &state) {}

  virtual void runBenchmark(benchmark::State &state) = 0;

  /// An instance of the Backend the benchmark is running against.
  std::unique_ptr<Backend> backend_;
  /// The module to use for the benchmark.
  std::unique_ptr<Module> mod_;
  /// The execution context to use for the benchmark.
  std::unique_ptr<ExecutionContext> ctx_;
};

/// RuntimeBenchmark subclass that benchmarks at the HostManager level (i.e.
/// HostManager + Executor + DeviceManager).
template <typename BackendTy>
class HostManagerBenchmark : public RuntimeBenchmark<BackendTy> {
public:
  void SetUp(benchmark::State &state) override {
    RuntimeBenchmark<BackendTy>::SetUp(state);
    setUpHostManager(state);
  }

  void TearDown(benchmark::State &state) override {
    RuntimeBenchmark<BackendTy>::TearDown(state);
    tearDownHostManager(state);
  }

protected:
  virtual void setUpHostManager(benchmark::State &state) {
    // Get references to the backend and module stored in the parent class.
    // this->xxx() must be used since this is a template class (as is its
    // superclass).
    std::unique_ptr<Backend> &backend = this->getBackend();
    std::unique_ptr<Module> &mod = this->getModule();

    // Check that the backend is valid.
    if (!backend) {
      state.SkipWithError(
          "Unable to set up host manager - backend not set up!");
      return;
    }

    // Check that the module is valid.
    if (!mod) {
      state.SkipWithError("Unable to set up host manager - module not set up!");
      return;
    }

    // Create DeviceConfigs with which to initialize the HostManager
    // instance.
    std::vector<std::unique_ptr<DeviceConfig>> configs;
    for (unsigned i = 0; i < numDeviceManagers_; ++i) {
      configs.emplace_back(
          llvm::make_unique<DeviceConfig>(backend->getBackendName()));
    }

    // Create and initialize the HostManager instance.
    hostManager_ = llvm::make_unique<HostManager>(std::move(configs));

    // Remember the names of all functions in the module before passing
    // ownership to the HostManager.
    for (auto *function : mod->getFunctions()) {
      functions_.emplace_back(function->getName());
    }

    // Add the module to the HostManager instance.
    CompilationContext cctx;
    bool error = errToBool(hostManager_->addNetwork(std::move(mod), cctx));
    if (error) {
      state.SkipWithError("Unable to set up host manager - failed to add "
                          "module!");
    }
  }

  virtual void tearDownHostManager(benchmark::State &state) {
    // Check that the HostManager instance is valid.
    if (!hostManager_) {
      state.SkipWithError(
          "Unable to tear down host manager - host manager not set up!");
      return;
    }

    // Clear all networks and stop all devices.
    bool error = errToBool(hostManager_->clearHost());
    if (error) {
      state.SkipWithError(
          "Unable to tear down host manager - failed to clear host!");
    }

    functions_.clear();
  }

  void runBenchmark(benchmark::State &state) override {
    // Get references to the context stored in the parent class.
    // this->xxx() must be used since this is a template class (as is its
    // superclass).
    std::unique_ptr<ExecutionContext> &ctx = this->getExecutionContext();

    for (auto _ : state) {
      // Run all functions in the module synchronously.
      for (const auto &function : functions_) {
        std::promise<void> promise;
        std::future<void> future = promise.get_future();
        hostManager_->runNetwork(
            function, std::move(ctx),
            [&promise, &ctx](runtime::RunIdentifierTy /*runId*/,
                             llvm::Error /*err*/,
                             std::unique_ptr<ExecutionContext> result) {
              ctx = std::move(result);
              promise.set_value();
            });
        future.wait();
      }
    }
  }

  /// The HostManager instance being benchmarked.
  std::unique_ptr<HostManager> hostManager_;
  /// The number of DeviceManagers to use during the benchmark.
  static constexpr unsigned numDeviceManagers_{1};
  /// List of functions in the module.
  std::vector<std::string> functions_;
};

/// RuntimeBenchmark subclass that benchmarks at the Executor level (i.e.
/// Executor + DeviceManager).
template <typename BackendTy>
class ExecutorBenchmark : public RuntimeBenchmark<BackendTy> {
public:
  void SetUp(benchmark::State &state) override {
    RuntimeBenchmark<BackendTy>::SetUp(state);
    setUpExecutor(state);
  }

  void TearDown(benchmark::State &state) override {
    RuntimeBenchmark<BackendTy>::TearDown(state);
    tearDownExecutor(state);
  }

protected:
  virtual void setUpDeviceManagers(benchmark::State &state) {
    // Get references to the backend and module stored in the parent class.
    // this->xxx() must be used since this is a template class (as is its
    // superclass).
    std::unique_ptr<Backend> &backend = this->getBackend();
    std::unique_ptr<Module> &module = this->getModule();

    // Create numDeviceManagers_ DeviceManagers and store references to them as
    // well as the CompiledFunctions loaded onto them.
    for (unsigned i = 0; i < numDeviceManagers_; ++i) {
      std::unique_ptr<DeviceManager> deviceManager;
      std::unordered_map<std::string, std::unique_ptr<CompiledFunction>>
          deviceManagerFunctions;
      setUpDeviceManagerCommon(state, backend, module, deviceManager,
                               deviceManagerFunctions);
      deviceManagers_.insert(std::make_pair(i, std::move(deviceManager)));
      deviceManagersFunctions_.insert(
          std::make_pair(i, std::move(deviceManagerFunctions)));
    }
  }

  virtual void tearDownDeviceManagers(benchmark::State &state) {
    // Tear down the numDeviceManagers_ DeviceManagers used for the benchmark.
    for (unsigned i = 0; i < numDeviceManagers_; ++i) {
      tearDownDeviceManagerCommon(state, deviceManagers_[i],
                                  deviceManagersFunctions_[i]);
      deviceManagers_.erase(i);
      deviceManagersFunctions_.erase(i);
    }
  }

  virtual void setUpExecutor(benchmark::State &state) {
    setUpDeviceManagers(state);
    executor_ =
        std::unique_ptr<Executor>(new ThreadPoolExecutor(deviceManagers_));
    setUpDAG(state);
  }

  virtual void tearDownExecutor(benchmark::State &state) {
    tearDownDeviceManagers(state);

    if (!executor_) {
      state.SkipWithError(
          "Unable to tear down executor -  executor not set up!");
    }

    executor_->shutdown();
    tearDownDAG(state);
  }

  /// Set up the executor DAG that the Executor taken in as input to
  /// Executor::run().
  virtual void setUpDAG(benchmark::State &state) = 0;
  /// Tear down the executor DAG.
  virtual void tearDownDAG(benchmark::State &state) {}

  virtual void runBenchmark(benchmark::State &state) override {
    // Get a reference to the context stored in the parent class.
    // this->xxx() must be used since this is a template class (as is its
    // superclass).
    std::unique_ptr<ExecutionContext> &ctx = this->getExecutionContext();
    for (auto _ : state) {
      // Run the DAG synchronously.
      std::promise<void> promise;
      std::future<void> future = promise.get_future();
      executor_->run(
          (dag_->root).get(), std::move(ctx), /*runId=*/0,
          [&promise, &ctx](runtime::RunIdentifierTy /*runId*/,
                           llvm::Error /*err*/,
                           std::unique_ptr<ExecutionContext> result) {
            ctx = std::move(result);
            promise.set_value();
          });
      future.wait();
    }
  }

  /// The Executor instance being benchmarked.
  std::unique_ptr<Executor> executor_;
  /// The DAG needed by the Executor.
  std::unique_ptr<DAG> dag_;
  /// The number of DeviceManagers to create.
  static constexpr unsigned numDeviceManagers_{1};
  /// The DeviceManagers used by the Executor during the benchmark (map from
  /// DeviceID -> DeviceManager).
  DeviceManagerMapTy deviceManagers_;
  /// The CompiledFunctions loaded on the DeviceManagers used by the Executor
  /// during the benchmark (map from DeviceID -> (map from string ->
  /// CompiledFunction)).
  std::unordered_map<
      DeviceIDTy,
      std::unordered_map<std::string, std::unique_ptr<CompiledFunction>>>
      deviceManagersFunctions_;
};

/// RuntimeBenchmark subclass that benchmarks at the DeviceManager level.
template <typename BackendTy>
class DeviceManagerBenchmark : public RuntimeBenchmark<BackendTy> {
public:
  void SetUp(benchmark::State &state) override {
    RuntimeBenchmark<BackendTy>::SetUp(state);
    setUpDeviceManager(state);
  }

  void TearDown(benchmark::State &state) override {
    tearDownDeviceManager(state);
    RuntimeBenchmark<BackendTy>::TearDown(state);
  }

protected:
  virtual void setUpDeviceManager(benchmark::State &state) {
    setUpDeviceManagerCommon(state, this->getBackend(), this->getModule(),
                             deviceManager_, deviceManagerFunctions_);
  }

  virtual void tearDownDeviceManager(benchmark::State &state) {
    tearDownDeviceManagerCommon(state, deviceManager_, deviceManagerFunctions_);
  }

  virtual void runBenchmark(benchmark::State &state) override {
    // Get a reference to the context stored in the parent class.
    // this->xxx() must be used since this is a template class (as is its
    // superclass).
    std::unique_ptr<ExecutionContext> &ctx = this->getExecutionContext();
    for (auto _ : state) {
      // Run all functions added to the DeviceManager.
      for (const auto &func : deviceManagerFunctions_) {
        std::promise<void> promise;
        std::future<void> future = promise.get_future();
        deviceManager_->runFunction(
            func.first, std::move(ctx),
            [&promise, &ctx](runtime::RunIdentifierTy /*runId*/,
                             llvm::Error /*err*/,
                             std::unique_ptr<ExecutionContext> result) {
              ctx = std::move(result);
              promise.set_value();
            });
        future.wait();
      }
    }
  }

  /// The DeviceManager instance being benchmarked.
  std::unique_ptr<DeviceManager> deviceManager_;
  /// All of the CompiledFunctions added to the DeviceManager (they have to be
  /// kept somewhere since the DeviceManager class does not own them).
  std::unordered_map<std::string, std::unique_ptr<CompiledFunction>>
      deviceManagerFunctions_;
};

//===--------------------------------------------------------------------===//
//              Benchmark Module and DAG Creator Functions                  //
//===--------------------------------------------------------------------===//

//----------------------------- Single Node --------------------------------//
/// Create a module consisting of a single FC operator.
std::unique_ptr<Module> createSingleNodeModule() {
  auto mod = llvm::make_unique<Module>();
  auto fn = mod->createFunction("singleNode");
  PlaceholderBindings bindings;

  auto *input =
      mod->createPlaceholder(ElemKind::FloatTy, {16, 32}, "input", false);
  auto *weights =
      mod->createPlaceholder(ElemKind::FloatTy, {32, 32}, "weights1", false);
  auto *bias = mod->createPlaceholder(ElemKind::FloatTy, {32}, "bias", false);
  auto *output =
      mod->createPlaceholder(ElemKind::FloatTy, {16, 32}, "output", false);

  auto *fc = fn->createFullyConnected("fc", input, weights, bias);
  fn->createSave("save", fc, output);

  bindings.allocate(weights)->getHandle().clear(0);
  bindings.allocate(bias)->getHandle().clear(32);

  glow::convertPlaceholdersToConstants(fn, bindings, {input, output});

  return mod;
}

/// Create an Executor DAG consisting of just one node for the single FC
/// operator.
std::unique_ptr<DAG> createSingleNodeDAG(
    std::unordered_map<std::string, std::unique_ptr<CompiledFunction>>
        &compiledFunctions) {
  // The DAG should have one root node and one actual node corresponding to the
  // CompiledFunction obtained by compiling the singular function in the Module
  // created by createSingleNodeModule.
  auto root = llvm::make_unique<DAGNode>();
  auto singleNode = llvm::make_unique<DAGNode>();

  root->children.emplace_back(singleNode.get());

  singleNode->parents.emplace_back(root.get());
  singleNode->deviceIDs = {0};
  singleNode->name = "singleNode";
  singleNode->runtimeBundle = llvm::make_unique<RuntimeBundle>(
      compiledFunctions["singleNode"]->getRuntimeBundle());

  std::vector<std::unique_ptr<DAGNode>> nodes;
  nodes.emplace_back(std::move(singleNode));

  auto dag = llvm::make_unique<DAG>();
  dag->root = std::move(root);
  dag->nodes = std::move(nodes);

  return dag;
}
//--------------------------------------------------------------------------//

//===--------------------------------------------------------------------===//
//              Benchmark Declarations and Instantiations                   //
//===--------------------------------------------------------------------===//

// Declare a runtime benchmark named SingleNode that uses createSingleNodeModule
// to create the module and createSingleNodeDAG to create the Module and DAG for
// benchmarking. This declares appropriately named subclasses of
// HostManagerBenchmark, ExecutorBenchmark and DeviceManagerBenchmark.
DECLARE_RUNTIME_BENCHMARK(SingleNode, createSingleNodeModule,
                          createSingleNodeDAG);

// Instantiate the SingleNode benchmark for the CPU backend. This creates
// instances of the HostManagerBenchmark, ExecutorBenchmark and
// DeviceManagerBenchmark subclasses declared by the macro above for the CPU
// backend.
INSTANTIATE_RUNTIME_BENCHMARK(SingleNode, CPUBackend);

//===--------------------------------------------------------------------===//
//                           Benchmark Main                                 //
//===--------------------------------------------------------------------===//

// Benchmark main.
BENCHMARK_MAIN();
