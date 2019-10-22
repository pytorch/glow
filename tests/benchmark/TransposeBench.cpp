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
 * Benchmark m independent nxk transposes along with add layers.
 */
class TransposeBench : public Benchmark {
  size_t m_;
  size_t n_;
  size_t k_;
  size_t numLayers_;
  std::unique_ptr<runtime::HostManager> hostManager_;
  std::vector<std::unique_ptr<ExecutionContext>> contexts_;
  size_t asyncLaunchSize_;
  const char *backendStr_;
  ElemKind dtype_;
  size_t elementSize_;

public:
  TransposeBench(size_t m_, size_t n_, size_t k_, size_t numLayers_,
                 size_t asyncLaunchSize_, const char *backendStr_,
                 const char *dtypeStr_)
      : m_(m_), n_(n_), k_(k_), numLayers_(numLayers_),
        asyncLaunchSize_(asyncLaunchSize_), backendStr_(backendStr_) {

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
    for (int i = 0; i < asyncLaunchSize_; i++) {
      std::unique_ptr<ExecutionContext> context(new ExecutionContext);
      contexts_.push_back(std::move(context));
    }

    // Setup host manager
    std::vector<std::unique_ptr<runtime::DeviceConfig>> configs;
    auto config = llvm::make_unique<runtime::DeviceConfig>(backendStr_);
    configs.push_back(std::move(config));
    hostManager_ = llvm::make_unique<runtime::HostManager>(std::move(configs));

    std::unique_ptr<Module> mod(new Module);
    auto fn = mod->createFunction("singleNode");

    Placeholder *input;
    Node *cur;
    SaveNode *S;

    input = mod->createPlaceholder(dtype_, {m_, n_, k_}, "A", false);

    // for each context, add input bindings
    for (int i = 0; i < asyncLaunchSize_; i++) {
      if (dtype_ == ElemKind::FloatTy) {
        contexts_[i]
            ->getPlaceholderBindings()
            ->allocate(input)
            ->getHandle<float>()
            .randomize(0.0f, 1.0f, mod->getPRNG());
      } else if (dtype_ == ElemKind::Float16Ty) {
        contexts_[i]
            ->getPlaceholderBindings()
            ->allocate(input)
            ->getHandle<float16_t>()
            .randomize(0.0f, 1.0f, mod->getPRNG());
      }
    }

    cur = input;
    for (int layer = 0; layer < numLayers_; layer++) {
      auto *xp = fn->createTranspose("transpose_" + std::to_string(layer), cur,
                                     {0, 2, 1});
      auto *ad = fn->createAdd("add_" + std::to_string(layer), cur, xp);
      cur = ad;
    }

    S = fn->createSave("save", cur);

    // for each context, add output bindings
    for (int i = 0; i < asyncLaunchSize_; i++) {
      contexts_[i]->getPlaceholderBindings()->allocate(S->getPlaceholder());
    }

    CompilationContext ctx;
    hostManager_->addNetwork(std::move(mod), ctx);
  }

  void run() override {
    std::vector<std::unique_ptr<ExecutionContext>> localContexts(
        asyncLaunchSize_);
    std::vector<std::promise<void>> promises(asyncLaunchSize_);
    std::vector<std::future<void>> futures;

    // Launch a number of parallel requests
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
    for (int j = 0; j < asyncLaunchSize_; j++) {
      contexts_[j] = std::move(localContexts[j]);
    }
  }

  void teardown() override {}

  // Each layer reads the tensor thrice, and writes the tensor twice
  double gbytes() const {
    return (5.0 * numLayers_ * m_ * n_ * k_ * elementSize_) / 1e9;
  }
};

int main(int argc, char *argv[]) {
  assert(argc == 9);
  size_t m = atoi(argv[1]);
  size_t n = atoi(argv[2]);
  size_t k = atoi(argv[3]);
  size_t numLayers = atoi(argv[4]);
  size_t numReps = atoi(argv[5]);
  size_t numAsyncLaunches = atoi(argv[6]);
  const char *backendStr = argv[7];
  const char *dtypeStr = argv[8];
  assert(numReps > 0);

  TransposeBench b(m, n, k, numLayers, numAsyncLaunches, backendStr, dtypeStr);

  auto times = bench(&b, numReps);
  for (auto t : times) {
    printf(
        "BenchResult,TransposeBench,SW,%zu,%zu,%zu,%zu,%zu,%zu,%s,%s,%f,%f\n",
        m, n, k, numLayers, numReps, numAsyncLaunches, backendStr, dtypeStr,
        t / numAsyncLaunches, b.gbytes() * numAsyncLaunches / t);
  }
  double min = *(std::min_element(times.begin(), times.end()));
  size_t midElt = times.size() / 2;
  std::nth_element(times.begin(), times.begin() + midElt, times.end());
  double median = times[midElt];
  double median_runtime = median / ((double)numAsyncLaunches);
  double min_runtime = min / ((double)numAsyncLaunches);
  printf(
      "BenchSummary,TransposeBench,SW,%zu,%zu,%zu,%zu,%zu,%zu,%s,%s,%f,%f,%f,%"
      "f\n",
      m, n, k, numLayers, numReps, numAsyncLaunches, backendStr, dtypeStr,
      median_runtime, min_runtime, b.gbytes() / median_runtime,
      b.gbytes() / min_runtime);
}
