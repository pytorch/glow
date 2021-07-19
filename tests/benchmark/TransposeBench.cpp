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
#include <algorithm>
#include <array>
#include <cstdlib>
#include <future>
#include <random>

#include "Bench.h"

#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Optimizer/GraphOptimizer/GraphOptimizer.h"

using namespace glow;

/*
 * This class implements a transpose microbenchmark. There are multiple
 * layers of transpose, followed by an Add with the tensor from the previous
 * layer.
 *
 * Microbenchmarks are generally useful for understanding performance
 * through targeted experiementation and are not representative of
 * end-to-end workloads.
 */
class TransposeBench : public Benchmark {
  dim_t batchSize_;
  dim_t n_;
  dim_t numLayers_;
  std::unique_ptr<runtime::HostManager> hostManager_;
  std::vector<std::unique_ptr<ExecutionContext>> contexts_;
  dim_t asyncLaunchSize_;
  dim_t numCores_;
  const char *backendStr_;
  ElemKind dtype_;
  dim_t elementSize_;
  const char *devId_;

public:
  TransposeBench(dim_t batchSize_, dim_t n_, dim_t numLayers_,
                 dim_t asyncLaunchSize_, dim_t numCores_,
                 const char *backendStr_, const char *dtypeStr_,
                 const char *devId_ = nullptr)
      : batchSize_(batchSize_), n_(n_), numLayers_(numLayers_),
        asyncLaunchSize_(asyncLaunchSize_), numCores_(numCores_),
        backendStr_(backendStr_), devId_(devId_) {

    dtype_ = ElemKind::Float16Ty;
    elementSize_ = 2;
    if (std::string(dtypeStr_) == "Float16") {
      dtype_ = ElemKind::Float16Ty;
      elementSize_ = 2;
    } else if (std::string(dtypeStr_) == "Float32") {
      dtype_ = ElemKind::FloatTy;
      elementSize_ = 4;
    }
  }

  void setup() override {

    // Create execution contexts here
    for (dim_t i = 0; i < asyncLaunchSize_; i++) {
      std::unique_ptr<ExecutionContext> context(new ExecutionContext);
      contexts_.push_back(std::move(context));
    }

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

    std::vector<Placeholder *> input(numCores_);
    std::vector<SaveNode *> S(numCores_);
    auto batchSizePerCore = getBatchSizePerCore(batchSize_, numCores_);

    for (dim_t core = 0; core < numCores_; core++) {
      if (batchSizePerCore[core] == 0)
        continue;
      input[core] =
          mod->createPlaceholder(dtype_, {batchSizePerCore[core], n_, n_},
                                 "A" + std::to_string(core), false);
    }

    // Create multiple chains of Transpose and Add nodes
    for (dim_t core = 0; core < numCores_; core++) {
      if (batchSizePerCore[core] == 0)
        continue;
      // for each context, add input bindings
      for (dim_t i = 0; i < asyncLaunchSize_; i++) {
        if (dtype_ == ElemKind::FloatTy) {
          contexts_[i]
              ->getPlaceholderBindings()
              ->allocate(input[core])
              ->getHandle<float>()
              .randomize(0.0f, 1.0f, mod->getPRNG());
        } else if (dtype_ == ElemKind::Float16Ty) {
          contexts_[i]
              ->getPlaceholderBindings()
              ->allocate(input[core])
              ->getHandle<float16_t>()
              .randomize(0.0f, 1.0f, mod->getPRNG());
        }
      }

      Node *cur = input[core];
      for (dim_t layer = 0; layer < numLayers_; layer++) {
        auto *xp = fn->createTranspose("transpose_" + std::to_string(layer) +
                                           "_" + std::to_string(core),
                                       cur, {0, 2, 1});
        auto *ad = fn->createAdd("add_" + std::to_string(layer) + "_" +
                                     std::to_string(core),
                                 cur, xp);
        cur = ad;
      }

      S[core] = fn->createSave("save", cur);

      // for each context, allocate output
      for (dim_t i = 0; i < asyncLaunchSize_; i++) {
        contexts_[i]->getPlaceholderBindings()->allocate(
            S[core]->getPlaceholder());
      }
    }

    CompilationContext ctx;
    EXIT_ON_ERR(hostManager_->addNetwork(std::move(mod), ctx));
  }

  void run() override {
    std::vector<std::unique_ptr<ExecutionContext>> localContexts(
        asyncLaunchSize_);
    std::vector<std::promise<void>> promises(asyncLaunchSize_);
    std::vector<std::future<void>> futures;

    // Launch a number of independent requests
    int i = 0;
    for (auto &promise : promises) {
      futures.push_back(promise.get_future());
      hostManager_->runNetwork(
          "singleNode", std::move(contexts_[i]),
          [&localContexts, &promise,
           i](runtime::RunIdentifierTy, Error err,
              std::unique_ptr<ExecutionContext> contextPtr) {
            EXIT_ON_ERR(std::move(err));
            localContexts[i] = std::move(contextPtr);
            promise.set_value();
          });
      i++;
    }
    for (auto &fut : futures) {
      fut.wait();
    }
    for (dim_t j = 0; j < asyncLaunchSize_; j++) {
      contexts_[j] = std::move(localContexts[j]);
    }
  }

  void teardown() override {}

  // Each layer reads the tensor thrice, and writes the tensor twice
  double gbytes() const {
    return (5.0 * numLayers_ * batchSize_ * n_ * n_ * elementSize_) / 1e9;
  }
};

int main(int argc, char *argv[]) {
  printf("Transpose Microbenchmark\n");
  printf("Usage: TransposeBench batchSize(Int) n(Int) numLayers(Int) "
         "numReps(Int) numAsyncLaunches(Int) numTransposeChains(Int) "
         "backendStr(String) dtypeStr(\"Float16\"|\"Float32\") dev_id(Int)\n");
  printf("Standard Glow command-line options may be passed via the GLOW_OPTS "
         "environment variable\n");
  benchParseGlowOpts(argc, argv);
  assert(argc == 9 || argc == 10);
  size_t batchSize = atoi(argv[1]);
  size_t n = atoi(argv[2]);
  size_t numLayers = atoi(argv[3]);
  size_t numReps = atoi(argv[4]);
  size_t numAsyncLaunches = atoi(argv[5]);
  size_t numCores = atoi(argv[6]);
  const char *backendStr = argv[7];
  const char *dtypeStr = argv[8];
  char *dev_id = nullptr;

  if (argc > 9) {
    dev_id = argv[9];
    printf("Setting backend device: \"%s\"\n", dev_id);
  }

  assert(numReps > 0);

  TransposeBench b(batchSize, n, numLayers, numAsyncLaunches, numCores,
                   backendStr, dtypeStr, dev_id);

  auto times = bench(&b, numReps);
  printf("_,benchName,_,batchSize,n,numLayers,numReps,numAsyncLaunches,"
         "numTransposeChains,backendStr,dtypeStr,runtime,gbytesPerSec\n");
  for (auto t : times) {
    printf(
        "BenchResult,TransposeBench,SW,%zu,%zu,%zu,%zu,%zu,%zu,%s,%s,%f,%f\n",
        batchSize, n, numLayers, numReps, numAsyncLaunches, numCores,
        backendStr, dtypeStr, t / numAsyncLaunches,
        b.gbytes() * numAsyncLaunches / t);
  }
  double min = *(std::min_element(times.begin(), times.end()));
  size_t midElt = times.size() / 2;
  std::nth_element(times.begin(), times.begin() + midElt, times.end());
  double median = times[midElt];
  double median_runtime = median / ((double)numAsyncLaunches);
  double min_runtime = min / ((double)numAsyncLaunches);
  printf("_,benchName,_,batchSize,n,numLayers,numReps,numAsyncLaunches,"
         "numTransposeChains,backendStr,dtypeStr,medianRuntime,minRuntime,"
         "medianGbytesPerSec,maxGbytesPerSec\n");
  printf(
      "BenchSummary,TransposeBench,SW,%zu,%zu,%zu,%zu,%zu,%zu,%s,%s,%f,%f,%f,%"
      "f\n",
      batchSize, n, numLayers, numReps, numAsyncLaunches, numCores, backendStr,
      dtypeStr, median_runtime, min_runtime, b.gbytes() / median_runtime,
      b.gbytes() / min_runtime);
}
