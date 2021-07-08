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
#include <array>
#include <cstdlib>
#include <future>
#include <random>

#include "Bench.h"

#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Optimizer/GraphOptimizer/GraphOptimizer.h"

using namespace glow;

/*
 * Benchmark a number of (m x n) * (n x n) matrix multiplications.
 * There are a number of parallel FC nodes which are created, one per core.
 * Each core handles one weight matrix. Then these are
 * chained together in multiple layers. After each layer, output tensor
 * is passed to the next layer.
 */
class Int8GemmParallelBench : public Benchmark {
  /// Matrices.
  std::vector<float> a;
  std::vector<float> b;
  std::vector<float> c;

  /// Dimensions expressed in libjit's format.
  size_t aDims[2];
  size_t cDims[2];
  size_t numLayers_;
  PlaceholderBindings bindings_;
  std::unique_ptr<runtime::HostManager> hostManager_;
  size_t asyncLaunchSize_;
  size_t numCores_;
  const char *backendStr_;
  const char *devId_;

public:
  Int8GemmParallelBench(size_t m, size_t n, size_t numLayers_,
                        size_t asyncLaunchSize_, size_t numCores_,
                        const char *backendStr_, const char *devId_)
      : aDims{m, n}, cDims{m, n}, numLayers_(numLayers_),
        asyncLaunchSize_(asyncLaunchSize_), numCores_(numCores_),
        backendStr_(backendStr_), devId_(devId_) {}

  void setup() override {

    // Setup host manager
    std::vector<std::unique_ptr<runtime::DeviceConfig>> configs;
    auto config = glow::make_unique<runtime::DeviceConfig>(backendStr_);
    if (devId_ != nullptr) {
      config->parameters["DeviceID"] = devId_;
    }
    configs.push_back(std::move(config));
    hostManager_ = glow::make_unique<runtime::HostManager>(std::move(configs));
    printf("set up host manager\n");

    dim_t m = cDims[0];
    dim_t n = cDims[1];
    dim_t k = aDims[1];
    a.resize(m * k);
    b.resize(k * n);
    c.resize(m * n);

    std::unique_ptr<Module> mod(new Module);
    auto fn = mod->createFunction("singleNode");
    printf("set up module \n");

    std::vector<Node *> cur(numCores_);
    std::vector<Placeholder *> weights(numCores_);
    std::vector<Placeholder *> bias(numCores_);
    std::vector<Node *> fc(numCores_);
    std::vector<Placeholder *> input(numCores_);
    std::vector<Placeholder *> output(numCores_);

    printf("set up inputs and outputs");
    for (size_t core = 0; core < numCores_; core++) {
      input[core] =
          mod->createPlaceholder(ElemKind::Int8QTy, {m, k}, 1.0, 0,
                                 "input_" + std::to_string(core), false);
      output[core] =
          mod->createPlaceholder(ElemKind::Int8QTy, {m, n}, 1.0, 0,
                                 "output_" + std::to_string(core), false);
      cur[core] = input[core];
    }

    printf("set up weights and bias");
    for (size_t layer = 0; layer < numLayers_; layer++) {
      for (size_t core = 0; core < numCores_; core++) {
        weights[core] =
            mod->createPlaceholder(ElemKind::Int8QTy, {k, n}, 1.0, 0,
                                   "weights_" + std::to_string(core), false);
        bias[core] =
            mod->createPlaceholder(ElemKind::Int32QTy, {n}, 1.0, 0,
                                   "bias_" + std::to_string(core), false);
        bindings_.allocate(weights[core])
            ->getHandle<int8_t>()
            .randomize(0, 128, mod->getPRNG());
        bindings_.allocate(bias[core])
            ->getHandle<int32_t>()
            .randomize(0, 128, mod->getPRNG());
        fc[core] = fn->createFullyConnected(
            "fc" + std::to_string(core) + "_" + std::to_string(layer),
            cur[core], weights[core], bias[core]);
        cur[core] = fc[core];
      }
    }
    printf("save output");
    for (size_t core = 0; core < numCores_; core++) {
      fn->createSave("save" + std::to_string(core), cur[core], output[core]);
    }

    for (size_t core = 0; core < numCores_; core++) {
      ::glow::convertPlaceholdersToConstants(fn, bindings_,
                                             {
                                                 input[core],
                                                 output[core],
                                             });
    }

    CompilationContext ctx;
    EXIT_ON_ERR(hostManager_->addNetwork(std::move(mod), ctx));
  }

  void run() override {
    printf("Running module");
    std::vector<std::promise<void>> promises(asyncLaunchSize_);
    std::vector<std::future<void>> futures;
    for (auto &runPromise : promises) {
      std::unique_ptr<ExecutionContext> contextPtr(new ExecutionContext);
      futures.push_back(runPromise.get_future());
      hostManager_->runNetwork(
          "singleNode", std::move(contextPtr),
          [&runPromise](runtime::RunIdentifierTy, Error err,
                        std::unique_ptr<ExecutionContext> /* contextPtr */) {
            EXIT_ON_ERR(std::move(err));
            runPromise.set_value();
          });
    }
    for (auto &fut : futures) {
      fut.wait();
    }
  }

  void teardown() override {}

  double gflops() const {
    return 2.0 * cDims[0] * cDims[1] * aDims[1] * numLayers_ * numCores_ / 1e9;
  }
};

int main(int argc, char *argv[]) {
  size_t m = atoi(argv[1]);
  size_t n = atoi(argv[2]);
  size_t numLayers = atoi(argv[3]);
  size_t reps = atoi(argv[4]);
  size_t asyncLaunches = atoi(argv[5]);
  size_t numCores = atoi(argv[6]);
  const char *backendStr = argv[7];
  char *dev_id = nullptr;

  printf("Int8GEMMParallel Microbenchmark\n");
  printf(
      "Usage: Int8GemmParallelBench m(Int) n(Int) numLayers(Int) numReps(Int) "
      "numAsyncLaunches(Int) numCores(Int) backendStr(String) dev_id(Int)\n");
  printf("Standard Glow command-line options may be passed via the GLOW_OPTS "
         "environment variable\n");
  benchParseGlowOpts(argc, argv);
  assert(argc == 8 || argc == 9);
  if (argc > 8) {
    dev_id = argv[8];
    printf("Setting backend device: \"%s\"\n", dev_id);
  }
  printf("Start Int8GemmParallelBench\n");
  Int8GemmParallelBench b(m, n, numLayers, asyncLaunches, numCores, backendStr,
                          dev_id);
  auto times = bench(&b, reps);
  for (auto t : times) {
    printf("BenchResult,GemmParallelBench,SW,%4zu,%4zu,%4zu,%4zu,%4zu,%4zu,%s,%"
           "2.6lf,%5.2lf\n",
           m, n, numLayers, reps, asyncLaunches, numCores, backendStr,
           t / asyncLaunches, b.gflops() * asyncLaunches / t);
  }
  double min = *(std::min_element(times.begin(), times.end()));
  size_t midElt = times.size() / 2;
  std::nth_element(times.begin(), times.begin() + midElt, times.end());
  double median = times[midElt];
  double median_runtime = median / ((double)asyncLaunches);
  double min_runtime = min / ((double)asyncLaunches);
  printf("BenchSummary,GemmParallelBench,SW,%4zu,%4zu,%4zu,%4zu,%4zu,%4zu,%s,%"
         "2.6lf,%2.6lf,%5.2lf, %5.2lf\n",
         m, n, numLayers, reps, asyncLaunches, numCores, backendStr,
         median_runtime, min_runtime, b.gflops() / median_runtime,
         b.gflops() / min_runtime);
}
