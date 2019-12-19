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
 * This class implements a GEMM/FC microbenchmark. There are a set of
 * (m x k) * (k x n) = (m x n) matrix multiplications, chained together in
 * multiple layers.
 *
 * Microbenchmarks are generally useful for understanding performance
 * through targeted experiementation and are not representative of
 * end-to-end workloads.
 */
class GemmBench : public Benchmark {
  dim_t m_;
  dim_t n_;
  dim_t k_;
  dim_t numLayers_;
  PlaceholderBindings bindings_;
  std::unique_ptr<runtime::HostManager> hostManager_;
  dim_t asyncLaunchSize_;
  dim_t numSplits_;
  const char *backendStr_;
  const char *dtypeStr_;
  const char *devId_;

public:
  GemmBench(dim_t m_, dim_t n_, dim_t k_, dim_t numLayers_,
            dim_t asyncLaunchSize_, dim_t numSplits_, const char *backendStr_,
            const char *dtypeStr_, const char *devId_ = nullptr)
      : m_(m_), n_(n_), k_(k_), numLayers_(numLayers_),
        asyncLaunchSize_(asyncLaunchSize_), numSplits_(numSplits_),
        backendStr_(backendStr_), dtypeStr_(dtypeStr_), devId_(devId_) {}

  void setup() override {

    // Setup host manager
    std::vector<std::unique_ptr<runtime::DeviceConfig>> configs;
    auto config = glow::make_unique<runtime::DeviceConfig>(backendStr_);
    if (devId_ != nullptr) {
      config->parameters["DeviceID"] = devId_;
    }
    configs.push_back(std::move(config));
    hostManager_ = glow::make_unique<runtime::HostManager>(std::move(configs));

    std::unique_ptr<Module> mod(new Module);
    auto fn = mod->createFunction("singleNode");
    ElemKind dtype = ElemKind::Float16Ty;
    if (std::string(dtypeStr_) == "Float16") {
      dtype = ElemKind::Float16Ty;
    } else if (std::string(dtypeStr_) == "Float32") {
      dtype = ElemKind::FloatTy;
    }

    auto *input = mod->createPlaceholder(dtype, {m_, k_}, "input", false);
    auto *output = mod->createPlaceholder(dtype, {m_, n_}, "output", false);
    Node *cur = input;

    // Create multiple layers of FC nodes
    for (size_t layer = 0; layer < numLayers_; layer++) {
      Placeholder *weights;
      Placeholder *bias;
      Node *fc;
      weights = mod->createPlaceholder(
          dtype, {k_, n_}, "weights" + std::to_string(layer), false);
      bias = mod->createPlaceholder(dtype, {n_}, "bias" + std::to_string(layer),
                                    false);

      if (std::string(dtypeStr_) == "Float16") {
        bindings_.allocate(weights)->getHandle<float16_t>().clear(0);
        bindings_.allocate(bias)->getHandle<float16_t>().clear(32);
      } else if (std::string(dtypeStr_) == "Float32") {
        bindings_.allocate(weights)->getHandle<float>().clear(0);
        bindings_.allocate(bias)->getHandle<float>().clear(32);
      }
      fc = fn->createFullyConnected("fc_" + std::to_string(layer), cur, weights,
                                    bias);
      cur = fc;
    }
    fn->createSave("save1", cur, output);

    ::glow::convertPlaceholdersToConstants(fn, bindings_, {input, output});

    // Split weights
    executeVerticalFCWeightsSplit(fn, numSplits_, n_);

    CompilationContext ctx;
    EXIT_ON_ERR(hostManager_->addNetwork(std::move(mod), ctx));
  }

  void run() override {
    std::vector<std::promise<void>> promises(asyncLaunchSize_);
    std::vector<std::future<void>> futures;

    // Launch a number of independent requests
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

  double gflops() const { return 2.0 * m_ * n_ * k_ * numLayers_ / 1e9; }
};

int main(int argc, char *argv[]) {
  printf("GEMM Microbenchmark\n");
  printf("Usage: GemmBench m(Int) n(Int) k(Int) numLayers(Int) numReps(Int) "
         "numAsyncLaunches(Int) numSplits(Int) backendStr(String) "
         "dtypeStr(\"Float16\"|\"Float32\") dev_id(Int)\n");

  assert(argc == 10 || argc == 11);
  size_t m = atoi(argv[1]);
  size_t n = atoi(argv[2]);
  size_t k = atoi(argv[3]);
  size_t numLayers = atoi(argv[4]);
  size_t reps = atoi(argv[5]);
  size_t asyncLaunches = atoi(argv[6]);
  size_t numSplits = atoi(argv[7]);
  const char *backendStr = argv[8];
  const char *dtypeStr = argv[9];
  char *dev_id = nullptr;

  if (argc > 10) {
    dev_id = argv[10];
    printf("Setting backend device: \"%s\"\n", dev_id);
  }
  GemmBench b(m, n, k, numLayers, asyncLaunches, numSplits, backendStr,
              dtypeStr, dev_id);
  auto times = bench(&b, reps);
  printf("_,benchName,_,m,n,k,numLayers,numReps,numAsyncLaunches,numSplits,"
         "backendStr,dtypeStr,runtime,gflopPerSec\n");
  for (auto t : times) {
    printf(
        "BenchResult,GemmBench,SW,%4zu,%4zu,%4zu,%4zu,%4zu,%4zu,%4zu,%s,%s,%2."
        "6lf,%5.2lf\n",
        m, n, k, numLayers, reps, asyncLaunches, numSplits, backendStr,
        dtypeStr, t / asyncLaunches, b.gflops() * asyncLaunches / t);
  }
  double min = *(std::min_element(times.begin(), times.end()));
  size_t midElt = times.size() / 2;
  std::nth_element(times.begin(), times.begin() + midElt, times.end());
  double median = times[midElt];
  double median_runtime = median / ((double)asyncLaunches);
  double min_runtime = min / ((double)asyncLaunches);
  printf("_,benchName,_,m,n,k,numLayers,numReps,numAsyncLaunches,numSplits,"
         "backendStr,dtypeStr,medianRuntime,minRuntime,medianGflopPerSec,"
         "maxGflopPerSec\n");
  printf(
      "BenchSummary,GemmBench,SW,%4zu,%4zu,%4zu,%4zu,%4zu,%4zu,%4zu,%s,%s,%2."
      "6lf,%2.6lf,%5.2lf, %5.2lf\n",
      m, n, k, numLayers, reps, asyncLaunches, numSplits, backendStr, dtypeStr,
      median_runtime, min_runtime, b.gflops() / median_runtime,
      b.gflops() / min_runtime);
}
