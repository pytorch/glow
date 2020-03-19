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
class Int8GemmBench : public Benchmark {
  dim_t m_;
  dim_t n_;
  dim_t k_;
  dim_t numLayers_;
  PlaceholderBindings bindings_;
  std::unique_ptr<runtime::HostManager> hostManager_;
  dim_t asyncLaunchSize_;
  dim_t numSplits_;
  const char *backendStr_;
  const char *devId_;

public:
  Int8GemmBench(dim_t m_, dim_t n_, dim_t k_, dim_t numLayers_,
                dim_t asyncLaunchSize_, dim_t numSplits_,
                const char *backendStr_, const char *devId_ = nullptr)
      : m_(m_), n_(n_), k_(k_), numLayers_(numLayers_),
        asyncLaunchSize_(asyncLaunchSize_), numSplits_(numSplits_),
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

    std::unique_ptr<Module> mod(new Module);
    auto fn = mod->createFunction("singleNode");

    auto *input =
        mod->createPlaceholder(ElemKind::FloatTy, {m_, k_}, "input", false);
    auto *output =
        mod->createPlaceholder(ElemKind::FloatTy, {m_, n_}, "output", false);
    auto *q_input = fn->createQuantize(
        "int8_quantize", input,
        mod->uniqueType(ElemKind::Int8QTy, {m_, k_}, 1.0, 0));

    Node *cur = q_input;

    // Create multiple layers of FC nodes
    for (size_t layer = 0; layer < numLayers_; layer++) {
      Placeholder *weights;
      Placeholder *bias;
      Node *fc;
      weights =
          mod->createPlaceholder(ElemKind::Int8QTy, {k_, n_}, 1.0, 0,
                                 "weights" + std::to_string(layer), false);
      bias = mod->createPlaceholder(ElemKind::Int32QTy, {n_}, 1.0, 0,
                                    "bias" + std::to_string(layer), false);

      bindings_.allocate(weights)->getHandle<int8_t>().clear(1);
      bindings_.allocate(bias)->getHandle<int32_t>().clear(2);

      fc = fn->createFullyConnected("fc_" + std::to_string(layer), cur, weights,
                                    bias);

      cur = fc;
    }

    auto *dequantized_fc = fn->createDequantize(
        "int8_dequantize", cur, mod->uniqueType(ElemKind::FloatTy, {m_, n_}));
    cur = dequantized_fc;
    fn->createSave("save1", cur, output);
    ::glow::convertPlaceholdersToConstants(fn, bindings_, {input, output});

    // Model parallelize FCs
    llvm::DenseMap<Node *, size_t> numOfChunks;
    llvm::DenseMap<Node *, ParallelTransformKind> parOpts;
    for (auto &N : fn->getNodes()) {
      if (N.getKind() == Kinded::Kind::FullyConnectedNodeKind) {
        numOfChunks[&N] = numSplits_;
        parOpts[&N] = ParallelTransformKind::Model;
      }
    }

    // Parallelize Quantize/Dequantize
    for (auto &N : fn->getNodes()) {
      if (N.getKind() == Kinded::Kind::QuantizeNodeKind ||
          N.getKind() == Kinded::Kind::DequantizeNodeKind) {
        numOfChunks[&N] = numSplits_;
        parOpts[&N] = ParallelTransformKind::Data;
      }
    }
    parallelizeOps(fn, numOfChunks, parOpts, 1);
    optimize(fn, CompilationMode::Infer);
    CompilationContext ctx;
    EXIT_ON_ERR(hostManager_->addNetwork(std::move(mod), ctx));
  }

  void run() override {
    printf("Running module\n");
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
  printf(
      "Usage: GemmBench m(Int) n(Int) k(Int) numLayers(Int) numReps(Int) "
      "numAsyncLaunches(Int) numSplits(Int) backendStr(String) dev_id(Int)\n");

  assert(argc == 9 || argc == 10);
  size_t m = atoi(argv[1]);
  size_t n = atoi(argv[2]);
  size_t k = atoi(argv[3]);
  size_t numLayers = atoi(argv[4]);
  size_t reps = atoi(argv[5]);
  size_t asyncLaunches = atoi(argv[6]);
  size_t numSplits = atoi(argv[7]);
  const char *backendStr = argv[8];
  char *dev_id = nullptr;

  if (argc > 9) {
    dev_id = argv[9];
    printf("Setting backend device: \"%s\"\n", dev_id);
  }
  Int8GemmBench b(m, n, k, numLayers, asyncLaunches, numSplits, backendStr,
                  dev_id);
  printf("Start to bench\n");

  auto times = bench(&b, reps);
  printf("_,benchName,_,m,n,k,numLayers,numReps,numAsyncLaunches,numSplits,"
         "backendStr,runtime,gflopPerSec\n");
  for (auto t : times) {
    printf("BenchResult,GemmBench,SW,%4zu,%4zu,%4zu,%4zu,%4zu,%4zu,%4zu,%s,%2."
           "6lf,%5.2lf\n",
           m, n, k, numLayers, reps, asyncLaunches, numSplits, backendStr,
           t / asyncLaunches, b.gflops() * asyncLaunches / t);
  }
  double min = *(std::min_element(times.begin(), times.end()));
  size_t midElt = times.size() / 2;
  std::nth_element(times.begin(), times.begin() + midElt, times.end());
  double median = times[midElt];
  double median_runtime = median / ((double)asyncLaunches);
  double min_runtime = min / ((double)asyncLaunches);
  printf("_,benchName,_,m,n,k,numLayers,numReps,numAsyncLaunches,numSplits,"
         "backendStr,medianRuntime,minRuntime,medianGflopPerSec,"
         "maxGflopPerSec\n");

  printf("BenchSummary,GemmBench,SW,%4zu,%4zu,%4zu,%4zu,%4zu,%4zu,%4zu,%s,%2."
         "6lf,%2.6lf,%5.2lf, %5.2lf\n",
         m, n, k, numLayers, reps, asyncLaunches, numSplits, backendStr,
         median_runtime, min_runtime, b.gflops() / median_runtime,
         b.gflops() / min_runtime);
}
