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
 * This class implements an Concat microbenchmark. There are a number of
 * parallel Concat nodes which are created, one per core. Then these are
 * chained together in multiple layers.
 *
 * Microbenchmarks are generally useful for understanding performance
 * through targeted experiementation and are not representative of
 * end-to-end workloads.
 */
class ConcatBench : public Benchmark {
  dim_t m_;
  dim_t n_;
  dim_t numTensors_;
  dim_t numLayers_;
  PlaceholderBindings bindings_;
  std::unique_ptr<runtime::HostManager> hostManager_;
  size_t asyncLaunchSize_;
  const char *backendStr_;
  ElemKind dtype_;
  size_t elementSize_;
  const char *devId_;

public:
  ConcatBench(dim_t m_, dim_t n_, dim_t numTensors_, dim_t numLayers_,
              dim_t asyncLaunchSize_, const char *backendStr_,
              const char *dtypeStr_, const char *devId_ = nullptr)
      : m_(m_), n_(n_), numTensors_(numTensors_), numLayers_(numLayers_),
        asyncLaunchSize_(asyncLaunchSize_), backendStr_(backendStr_),
        devId_(devId_) {

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
    // Create multiple chains of Concat nodes
    std::vector<Placeholder *> A(numTensors_);
    std::vector<NodeValue> A_broadcast(numTensors_);
    std::vector<NodeValue> A_concat(numTensors_);
    std::vector<NodeValue> slices(numTensors_);

    Placeholder *output;

    for (size_t tensor = 0; tensor < numTensors_; tensor++) {
      A[tensor] = mod->createPlaceholder(dtype_, {1, n_},
                                         "A" + std::to_string(tensor), false);
      A_broadcast[tensor] = fn->createBroadcast(
          "A_bcast" + std::to_string(tensor), A[tensor], {m_, n_}, 0);
    }
    output =
        mod->createPlaceholder(dtype_, {1, n_ * numTensors_}, "output", false);

    for (size_t tensor = 0; tensor < numTensors_; tensor++) {
      A_concat[tensor / 2 * 2 + ((tensor % 2) ? 0 : 1)] = A_broadcast[tensor];
    }
    auto *concat = fn->createConcat("concat_0", A_concat, 1);

    for (size_t layer = 1; layer < numLayers_; layer++) {
      for (size_t tensor = 0; tensor < numTensors_; tensor++) {
        dim_t start_n =
            tensor / 2 * 2 * n_ + ((tensor % 2) ? (3 * n_ / 2) : (0));
        dim_t end_n = start_n + ((tensor % 2) ? (n_ / 2) : (3 * n_ / 2));
        slices[tensor] = fn->createSlice("slice_" + std::to_string(tensor),
                                         concat, {0, start_n}, {m_, end_n});
      }
      for (size_t tensor = 0; tensor < numTensors_; tensor++) {
        A_concat[tensor / 2 * 2 + ((tensor % 2) ? 0 : 1)] = slices[tensor];
      }
      concat = fn->createConcat("concat_" + std::to_string(layer), A_concat, 1);
    }
    Node *slice =
        fn->createSlice("slice_final", concat, {0, 0}, {1, n_ * numTensors_});
    fn->createSave("save", slice, output);
    CompilationContext ctx;
    ctx.dumpFinalGraph = true;
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
  double gbytes() const {
    return elementSize_ * m_ * n_ * numTensors_ * numLayers_ / 1e9;
  }
};

int main(int argc, char *argv[]) {
  printf("Concat Microbenchmark\n");
  printf("Usage: ConcatBench m(Int) n(Int) numTensors(Int) "
         "numLayers(Int) numReps(Int) "
         "numAsyncLaunches(Int) backendStr(String) "
         "dtypeStr(\"Float16\"|\"Float32\") dev_id(Int)\n");
  printf("Standard Glow command-line options may be passed via the GLOW_OPTS "
         "environment variable\n");
  benchParseGlowOpts(argc, argv);
  assert(argc == 9 || argc == 10);
  size_t m = atoi(argv[1]);
  size_t n = atoi(argv[2]);
  size_t numTensors = atoi(argv[3]);
  size_t numLayers = atoi(argv[4]);
  size_t reps = atoi(argv[5]);
  size_t asyncLaunches = atoi(argv[6]);
  const char *backendStr = argv[7];
  const char *dtypeStr = argv[8];
  char *dev_id = nullptr;

  if (argc > 9) {
    dev_id = argv[9];
    printf("Setting backend device: \"%s\"\n", dev_id);
  }

  assert(reps > 0);

  ConcatBench b(m, n, numTensors, numLayers, asyncLaunches, backendStr,
                dtypeStr, dev_id);
  auto times = bench(&b, reps);
  printf("_,benchName,_,m,n,numTensors,numLayers,numReps,numAsyncLaunches,"
         "backendStr,dtypeStr,runtime,gbytesPerSecPerChain\n");
  for (auto t : times) {
    printf("BenchResult,ConcatBench,SW,%4zu,%4zu,%4zu,%4zu,%4zu,%4zu,%s,%s,"
           "%2.6lf,%5.2lf\n",
           m, n, numTensors, numLayers, reps, asyncLaunches, backendStr,
           dtypeStr, t / asyncLaunches, b.gbytes() * asyncLaunches / t);
  }
  double min = *(std::min_element(times.begin(), times.end()));
  size_t midElt = times.size() / 2;
  std::nth_element(times.begin(), times.begin() + midElt, times.end());
  double median = times[midElt];
  double median_runtime = median / ((double)asyncLaunches);
  double min_runtime = min / ((double)asyncLaunches);
  printf("_,benchName,_,m,n,numTensors,numLayers,numReps,numAsyncLaunches,"
         "backendStr,dtypeStr,medianRuntime,minRuntime,"
         "medianGbytesPerSecPerChain,maxGbytesPerSecPerChain\n");
  printf("BenchSummary,ConcatBench,SW,%4zu,%4zu,%4zu,%4zu,%4zu,%4zu,%s,%s,"
         "%2.6lf,%2.6lf,%"
         "5.2lf, %5.2lf\n",
         m, n, numTensors, numLayers, reps, asyncLaunches, backendStr, dtypeStr,
         median_runtime, min_runtime, b.gbytes() / median_runtime,
         b.gbytes() / min_runtime);
}
