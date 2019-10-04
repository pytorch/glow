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
#include <array>
#include <cstdlib>
#include <future>
#include <random>

#include "Bench.h"
#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Optimizer/GraphOptimizer/GraphOptimizer.h"

using namespace glow;

/*
 * Benchmark an (m x k) * (k x n) = (m x n) matrix multiplication.
 * There are a number of parallel FC nodes which are created, one per core.
 * Each core handles n/num_cores columns of the weight matrix. Then these are
 * chained together in multiple layers. After each layer, output tensor
 * is concatenated into one tensor for consumption in the next layer.
 */
class GemmBench : public Benchmark {
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
  GemmBench(size_t m, size_t n, size_t k, size_t numLayers_,
            size_t asyncLaunchSize_, size_t numCores_, const char *backendStr_,
            const char *dtypeStr_)
      : aDims{m, k}, cDims{m, n}, numLayers_(numLayers_),
        asyncLaunchSize_(asyncLaunchSize_), numCores_(numCores_),
        backendStr_(backendStr_), dtypeStr_(dtypeStr_) {}

  void setup() override {

    // Setup host manager
    std::vector<std::unique_ptr<runtime::DeviceConfig>> configs;
    configs.push_back(llvm::make_unique<runtime::DeviceConfig>(backendStr_));
    hostManager_ = llvm::make_unique<runtime::HostManager>(std::move(configs));

    size_t m = cDims[0];
    size_t n = cDims[1];
    size_t k = aDims[1];
    a.resize(m * k);
    b.resize(k * n);
    c.resize(m * n);

    std::unique_ptr<Module> mod(new Module);
    auto fn = mod->createFunction("singleNode");
    ElemKind dtype = ElemKind::Float16Ty;
    if (std::string(dtypeStr_) == "Float16") {
      dtype = ElemKind::Float16Ty;
    } else if (std::string(dtypeStr_) == "Float32") {
      dtype = ElemKind::FloatTy;
    }

    auto *input = mod->createPlaceholder(dtype, {m, k}, "input", false);
    auto *output = mod->createPlaceholder(dtype, {m, n}, "output", false);
    Node *cur = input;
    std::vector<Placeholder *> weights(numCores_);
    std::vector<Placeholder *> bias(numCores_);
    std::vector<Node *> fc(numCores_);
    Node *concat;
    for (size_t layer = 0; layer < numLayers_; layer++) {
      for (size_t core = 0; core < numCores_; core++) {
        weights[core] = mod->createPlaceholder(
            dtype, {k, n / numCores_}, "weights" + std::to_string(core), false);
        bias[core] = mod->createPlaceholder(
            dtype, {n / numCores_}, "bias" + std::to_string(core), false);
        bindings_.allocate(weights[core])->getHandle<float16_t>().clear(0);
        bindings_.allocate(bias[core])->getHandle<float16_t>().clear(32);
        fc[core] = fn->createFullyConnected("fc" + std::to_string(core) + "_" +
                                                std::to_string(layer),
                                            cur, weights[core], bias[core]);
      }

      std::vector<NodeValue> fcNv;
      for (auto _fc : fc) {
        fcNv.push_back(_fc);
      }
      concat = fn->createConcat("concat_" + std::to_string(layer), fcNv, 1);
      cur = concat;
    }
    fn->createSave("save1", cur, output);

    ::glow::convertPlaceholdersToConstants(fn, bindings_, {input, output});

    CompilationContext ctx;
    hostManager_->addNetwork(std::move(mod), ctx);
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
    return 2.0 * cDims[0] * cDims[1] * aDims[1] * numLayers_ / 1e9;
  }
};

int main(int argc, char *argv[]) {
  assert(argc == 10);
  size_t m = atoi(argv[1]);
  size_t n = atoi(argv[2]);
  size_t k = atoi(argv[3]);
  size_t numLayers = atoi(argv[4]);
  size_t reps = atoi(argv[5]);
  size_t asyncLaunches = atoi(argv[6]);
  size_t numCores = atoi(argv[7]);
  const char *backendStr = argv[8];
  const char *dtypeStr = argv[9];
  GemmBench b(m, n, k, numLayers, asyncLaunches, numCores, backendStr,
              dtypeStr);
  auto times = bench(&b, reps);
  for (auto t : times) {
    printf(
        "BenchResult,GemmBench,SW,%4zu,%4zu,%4zu,%4zu,%4zu,%4zu,%4zu,%s,%s,%2."
        "6lf,%5.2lf\n",
        m, n, k, numLayers, reps, asyncLaunches, numCores, backendStr, dtypeStr,
        t / asyncLaunches, b.gflops() * asyncLaunches / t);
  }
  double min = *(std::min_element(times.begin(), times.end()));
  size_t midElt = times.size() / 2;
  std::nth_element(times.begin(), times.begin() + midElt, times.end());
  double median = times[midElt];
  double median_runtime = median / ((double)asyncLaunches);
  double min_runtime = min / ((double)asyncLaunches);
  printf(
      "BenchSummary,GemmBench,SW,%4zu,%4zu,%4zu,%4zu,%4zu,%4zu,%4zu,%s,%s,%2."
      "6lf,%2.6lf,%5.2lf, %5.2lf\n",
      m, n, k, numLayers, reps, asyncLaunches, numCores, backendStr, dtypeStr,
      median_runtime, min_runtime, b.gflops() / median_runtime,
      b.gflops() / min_runtime);
}
