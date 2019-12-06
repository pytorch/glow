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
 * This class implements an add microbenchmark. There are a number of
 * parallel Add nodes which are created, one per core. Then these are
 * chained together in multiple layers.
 *
 * Microbenchmarks are generally useful for understanding performance
 * through targeted experiementation and are not representative of
 * end-to-end workloads.
 */
class AddBench : public Benchmark {
  dim_t n_;
  dim_t numLayers_;
  PlaceholderBindings bindings_;
  std::unique_ptr<runtime::HostManager> hostManager_;
  size_t asyncLaunchSize_;
  size_t numCores_;
  const char *backendStr_;
  ElemKind dtype_;
  size_t elementSize_;
  const char *devId_;

public:
  AddBench(dim_t n_, dim_t numLayers_, dim_t asyncLaunchSize_, dim_t numCores_,
           const char *backendStr_, const char *dtypeStr_,
           const char *devId_ = nullptr)
      : n_(n_), numLayers_(numLayers_), asyncLaunchSize_(asyncLaunchSize_),
        numCores_(numCores_), backendStr_(backendStr_), devId_(devId_) {

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

    // Create multiple chains of Add nodes
    std::vector<Placeholder *> A(numCores_);
    std::vector<Placeholder *> B(numCores_);
    std::vector<Placeholder *> output(numCores_);
    std::vector<Node *> cur(numCores_);
    for (size_t core = 0; core < numCores_; core++) {
      A[core] = mod->createPlaceholder(dtype_, {n_}, "A" + std::to_string(core),
                                       false);
      B[core] = mod->createPlaceholder(dtype_, {n_}, "B" + std::to_string(core),
                                       false);
      output[core] = mod->createPlaceholder(
          dtype_, {n_}, "output" + std::to_string(core), false);
      cur[core] = A[core];
    }

    std::vector<Node *> eltwise(numCores_);
    for (size_t layer = 0; layer < numLayers_; layer++) {
      for (size_t core = 0; core < numCores_; core++) {
        eltwise[core] = fn->createAdd("eltwise" + std::to_string(core) + "_" +
                                          std::to_string(layer),
                                      cur[core], B[core]);
        cur[core] = eltwise[core];
      }
    }
    for (size_t core = 0; core < numCores_; core++) {
      fn->createSave("save" + std::to_string(core), cur[core], output[core]);
    }

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

  // Two inputs per layer and one output
  double gbytes() const { return elementSize_ * n_ * (3 * numLayers_) / 1e9; }
};

int main(int argc, char *argv[]) {
  printf("Add Microbenchmark\n");
  printf("Usage: AddBench n(Int) numLayers(Int) numReps(Int) "
         "numAsyncLaunches(Int) numAddChains(Int) backendStr(String) "
         "dtypeStr(\"Float16\"|\"Float32\") dev_id(Int)\n");
  assert(argc == 8 || argc == 9);
  size_t n = atoi(argv[1]);
  size_t numLayers = atoi(argv[2]);
  size_t reps = atoi(argv[3]);
  size_t asyncLaunches = atoi(argv[4]);
  size_t numCores = atoi(argv[5]);
  const char *backendStr = argv[6];
  const char *dtypeStr = argv[7];
  char *dev_id = nullptr;

  if (argc > 8) {
    dev_id = argv[8];
    printf("Setting backend device: \"%s\"\n", dev_id);
  }

  assert(reps > 0);

  AddBench b(n, numLayers, asyncLaunches, numCores, backendStr, dtypeStr,
             dev_id);
  auto times = bench(&b, reps);
  printf("_,benchName,_,n,numLayers,numReps,numAsyncLaunches,numAddChains,"
         "backendStr,dtypeStr,runtime,gbytesPerSecPerChain\n");
  for (auto t : times) {
    printf("BenchResult,AddBench,SW,%4zu,%4zu,%4zu,%4zu,%4zu,%s,%s,%2.6lf,"
           "%5.2lf\n",
           n, numLayers, reps, asyncLaunches, numCores, backendStr, dtypeStr,
           t / asyncLaunches, b.gbytes() * asyncLaunches / t);
  }
  double min = *(std::min_element(times.begin(), times.end()));
  size_t midElt = times.size() / 2;
  std::nth_element(times.begin(), times.begin() + midElt, times.end());
  double median = times[midElt];
  double median_runtime = median / ((double)asyncLaunches);
  double min_runtime = min / ((double)asyncLaunches);
  printf("_,benchName,_,n,numLayers,numReps,numAsyncLaunches,numAddChains,"
         "backendStr,dtypeStr,medianRuntime,minRuntime,"
         "medianGbytesPerSecPerChain,maxGbytesPerSecPerChain\n");
  printf(
      "BenchSummary,AddBench,SW,%4zu,%4zu,%4zu,%4zu,%4zu,%s,%s,%2.6lf,%2.6lf,%"
      "5.2lf, %5.2lf\n",
      n, numLayers, reps, asyncLaunches, numCores, backendStr, dtypeStr,
      median_runtime, min_runtime, b.gbytes() / median_runtime,
      b.gbytes() / min_runtime);
}
