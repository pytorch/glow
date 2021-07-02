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
class GemmParallelBench : public Benchmark {
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
  const char *dtypeStr_;

public:
  GemmParallelBench(size_t m, size_t n, size_t numLayers_,
                    size_t asyncLaunchSize_, size_t numCores_,
                    const char *backendStr_, const char *dtypeStr_)
      : aDims{m, n}, cDims{m, n}, numLayers_(numLayers_),
        asyncLaunchSize_(asyncLaunchSize_), numCores_(numCores_),
        backendStr_(backendStr_), dtypeStr_(dtypeStr_) {}

  void setup() override {

    // Setup host manager
    std::vector<std::unique_ptr<runtime::DeviceConfig>> configs;
    auto config = glow::make_unique<runtime::DeviceConfig>(backendStr_);
    configs.push_back(std::move(config));
    hostManager_ = glow::make_unique<runtime::HostManager>(std::move(configs));
    dim_t m = cDims[0];
    dim_t n = cDims[1];
    dim_t k = aDims[1];
    a.resize(m * k);
    b.resize(k * n);
    c.resize(m * n);

    ElemKind dtype = ElemKind::Float16Ty;
    if (std::string(dtypeStr_) == "Float16") {
      dtype = ElemKind::Float16Ty;
    } else if (std::string(dtypeStr_) == "Float32") {
      dtype = ElemKind::FloatTy;
    }

    std::unique_ptr<Module> mod(new Module);
    auto fn = mod->createFunction("singleNode");

    std::vector<Node *> cur(numCores_);
    std::vector<Placeholder *> weights(numCores_);
    std::vector<Placeholder *> bias(numCores_);
    std::vector<Node *> fc(numCores_);
    std::vector<Placeholder *> input(numCores_);
    std::vector<Placeholder *> output(numCores_);

    for (size_t core = 0; core < numCores_; core++) {
      input[core] = mod->createPlaceholder(
          dtype, {m, k}, "input" + std::to_string(core), false);
      output[core] = mod->createPlaceholder(
          dtype, {m, n}, "output" + std::to_string(core), false);
      cur[core] = input[core];
    }

    for (size_t layer = 0; layer < numLayers_; layer++) {
      for (size_t core = 0; core < numCores_; core++) {
        weights[core] = mod->createPlaceholder(
            dtype, {k, n}, "weights" + std::to_string(core), false);
        bias[core] = mod->createPlaceholder(
            dtype, {n}, "bias" + std::to_string(core), false);
        bindings_.allocate(weights[core])
            ->getHandle<float16_t>()
            .randomize(-128.f, 128.f, mod->getPRNG());
        bindings_.allocate(bias[core])
            ->getHandle<float16_t>()
            .randomize(-128.f, 128.f, mod->getPRNG());
        fc[core] = fn->createFullyConnected(
            "fc" + std::to_string(core) + "_" + std::to_string(layer),
            cur[core], weights[core], bias[core]);
        cur[core] = fc[core];
      }
    }
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
  benchParseGlowOpts(argc, argv);
  assert(argc == 9);
  size_t m = atoi(argv[1]);
  size_t n = atoi(argv[2]);
  size_t numLayers = atoi(argv[3]);
  size_t reps = atoi(argv[4]);
  size_t asyncLaunches = atoi(argv[5]);
  size_t numCores = atoi(argv[6]);
  const char *backendStr = argv[7];
  const char *dtypeStr = argv[8];

  GemmParallelBench b(m, n, numLayers, asyncLaunches, numCores, backendStr,
                      dtypeStr);
  auto times = bench(&b, reps);
  for (auto t : times) {
    printf(
        "BenchResult,GemmParallelBench,SW,%4zu,%4zu,%4zu,%4zu,%4zu,%4zu,%s,%s,%"
        "2.6lf,%5.2lf\n",
        m, n, numLayers, reps, asyncLaunches, numCores, backendStr, dtypeStr,
        t / asyncLaunches, b.gflops() * asyncLaunches / t);
  }
  double min = *(std::min_element(times.begin(), times.end()));
  size_t midElt = times.size() / 2;
  std::nth_element(times.begin(), times.begin() + midElt, times.end());
  double median = times[midElt];
  double median_runtime = median / ((double)asyncLaunches);
  double min_runtime = min / ((double)asyncLaunches);
  printf(
      "BenchSummary,GemmParallelBench,SW,%4zu,%4zu,%4zu,%4zu,%4zu,%4zu,%s,%s,%"
      "2.6lf,%2.6lf,%5.2lf, %5.2lf\n",
      m, n, numLayers, reps, asyncLaunches, numCores, backendStr, dtypeStr,
      median_runtime, min_runtime, b.gflops() / median_runtime,
      b.gflops() / min_runtime);
}
