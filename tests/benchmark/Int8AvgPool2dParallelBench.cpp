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

#include "ConvUtils.h"
#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Optimizer/GraphOptimizer/GraphOptimizer.h"

using namespace glow;
using namespace std;

vector<avg_pool_param_t<2>> shapes_2d = {
    // OC = 1x1
    // MB, IC, IH, IW (=IH), KH (=IH), KW (=IW),
    // 2D Avg Pool with broadcasts to make multi-layer avg pools work out.
    avg_pool_param_t<>(1, 768, {50, 50}, {50, 50}, {1, 1}, {0, 0, 0, 0}),
    avg_pool_param_t<>(1, 224, {100, 100}, {100, 100}, {1, 1}, {0, 0, 0, 0}),
    avg_pool_param_t<>(1, 192, {100, 100}, {100, 100}, {1, 1}, {0, 0, 0, 0}),
    avg_pool_param_t<>(1, 640, {50, 50}, {50, 50}, {1, 1}, {0, 0, 0, 0}),

    avg_pool_param_t<>(1, 432, {30, 30}, {30, 30}, {1, 1}, {0, 0, 0, 0}),
    avg_pool_param_t<>(1, 128, {60, 60}, {60, 60}, {1, 1}, {0, 0, 0, 0}),
    avg_pool_param_t<>(1, 168, {60, 60}, {60, 60}, {1, 1}, {0, 0, 0, 0}),
    avg_pool_param_t<>(1, 440, {30, 30}, {30, 30}, {1, 1}, {0, 0, 0, 0}),

    avg_pool_param_t<>(1, 7392, {7, 7}, {7, 7}, {1, 1}, {0, 0, 0, 0}),
    avg_pool_param_t<>(1, 528, {56, 56}, {56, 56}, {1, 1}, {0, 0, 0, 0}),
    avg_pool_param_t<>(1, 1056, {28, 28}, {28, 28}, {1, 1}, {0, 0, 0, 0}),
    avg_pool_param_t<>(1, 2904, {14, 14}, {14, 14}, {1, 1}, {0, 0, 0, 0}),

    avg_pool_param_t<>(1, 1536, {3, 3}, {3, 3}, {1, 1}, {0, 0, 0, 0}),
    avg_pool_param_t<>(1, 3072, {3, 3}, {3, 3}, {1, 1}, {0, 0, 0, 0}),
    avg_pool_param_t<>(1, 1920, {3, 3}, {3, 3}, {1, 1}, {0, 0, 0, 0}),
    avg_pool_param_t<>(1, 2304, {3, 3}, {3, 3}, {1, 1}, {0, 0, 0, 0}),
    avg_pool_param_t<>(1, 512, {3, 3}, {3, 3}, {1, 1}, {0, 0, 0, 0}),

    avg_pool_param_t<>(1, 1240, {3, 3}, {3, 3}, {1, 1}, {0, 0, 0, 0}),
    avg_pool_param_t<>(1, 864, {3, 3}, {3, 3}, {1, 1}, {0, 0, 0, 0}),
    avg_pool_param_t<>(1, 1488, {3, 3}, {3, 3}, {1, 1}, {0, 0, 0, 0}),
    avg_pool_param_t<>(1, 272, {3, 3}, {3, 3}, {1, 1}, {0, 0, 0, 0})

};

/*
 * Benchmark a number of AvgPool2d operators with representative input shapes.
 * There are a number of parallel AvgPoool2d nodes which are created, one
 * per core. Then these are chained together in multiple layers.
 * To ensure sizes match up between the output of a layer and
 * input of the next layer, we introduce a broadcast op.
 * After each layer, output tensor is passed to the next layer.
 */
class Int8AvgPool2dParallelBench : public Benchmark {
  /// Matrices.
  avg_pool_param_t<2> input_shape_;
  size_t numLayers_;
  PlaceholderBindings bindings_;
  std::unique_ptr<runtime::HostManager> hostManager_;
  size_t asyncLaunchSize_;
  size_t numCores_;
  const char *backendStr_;
  const char *devId_;

public:
  Int8AvgPool2dParallelBench(avg_pool_param_t<2> &input_shape_,
                             size_t numLayers_, size_t asyncLaunchSize_,
                             size_t numCores_, const char *backendStr_,
                             const char *devId_)
      : input_shape_(input_shape_), numLayers_(numLayers_),
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

    dim_t N, IC, IH, IW, OC;
    N = input_shape_.MB;
    IC = input_shape_.IC;
    IH = input_shape_.IN_DIM[0];
    IW = input_shape_.IN_DIM[1];
    OC = input_shape_.OC;

    std::unique_ptr<Module> mod(new Module);
    auto fn = mod->createFunction("singleNode");

    std::vector<Node *> cur(numCores_);

    std::vector<Placeholder *> input(numCores_);
    std::vector<Placeholder *> output(numCores_);

    for (size_t core = 0; core < numCores_; core++) {
      input[core] =
          mod->createPlaceholder(ElemKind::Int8QTy, {N, IH, IW, IC}, 1.0, 0,
                                 "input_" + std::to_string(core), false);
      output[core] =
          mod->createPlaceholder(ElemKind::Int8QTy, {N, IH, IW, OC}, 1.0, 0,
                                 "output_" + std::to_string(core), false);
      cur[core] = input[core];
    }

    for (size_t layer = 0; layer < numLayers_; layer++) {
      for (size_t core = 0; core < numCores_; core++) {
        auto pool = fn->createAvgPool("pool_" + std::to_string(core) + "_" +
                                          std::to_string(layer),
                                      cur[core],
                                      {(unsigned int)(input_shape_.K[0]),
                                       (unsigned int)(input_shape_.K[1])},
                                      {(unsigned int)(input_shape_.stride[0]),
                                       (unsigned int)(input_shape_.stride[1])},
                                      {(unsigned int)(input_shape_.pad[0]),
                                       (unsigned int)(input_shape_.pad[1]),
                                       (unsigned int)(input_shape_.pad[2]),
                                       (unsigned int)(input_shape_.pad[3])});
        auto tilex = fn->createTile("tile_dim1_" + std::to_string(core) + "_" +
                                        std::to_string(layer),
                                    pool, (unsigned int)(input_shape_.K[0]), 1);
        auto tiley = fn->createTile(
            "tile_dim2_" + std::to_string(core) + "_" + std::to_string(layer),
            tilex, (unsigned int)(input_shape_.K[1]), 2);
        cur[core] = tiley;
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
};

int main(int argc, char *argv[]) {
  size_t numLayers = atoi(argv[1]);
  size_t reps = atoi(argv[2]);
  size_t asyncLaunches = atoi(argv[3]);
  size_t numCores = atoi(argv[4]);
  const char *backendStr = argv[5];
  char *dev_id = nullptr;

  printf("Int8AvgPool2dParallel Microbenchmark\n");
  printf(
      "Usage: Int8AvgPool2dParallelBench numLayers(Int) "
      "numReps(Int) "
      "numAsyncLaunches(Int) numCores(Int) backendStr(String) dev_id(Int)\n");
  printf("Standard Glow command-line options may be passed via the GLOW_OPTS "
         "environment variable\n");
  benchParseGlowOpts(argc, argv);
  assert(argc == 6 || argc == 7);
  if (argc > 6) {
    dev_id = argv[6];
    printf("Setting backend device: \"%s\"\n", dev_id);
  }
  printf("Start Int8AvgPool2dParallelBench\n");
  size_t shape_idx = 0;
  size_t total_input_shapes = shapes_2d.size();
  for (auto shape : shapes_2d) {
    double gflops = 1.0 * (shape.IC) * shape.K[0] * shape.K[1] * (shape.OC) *
                    shape.OUT_DIM[0] * shape.OUT_DIM[1];
    gflops *= numLayers * numCores / 1e9;

    string shape_info = shape.toString();

    printf("\n=====Input shape %zu/%zu: %s\n", shape_idx, total_input_shapes,
           shape_info.c_str());
    Int8AvgPool2dParallelBench b(shape, numLayers, asyncLaunches, numCores,
                                 backendStr, dev_id);
    auto times = bench(&b, reps);
    for (auto t : times) {
      printf("BenchResult,AvgPool2dParallelBench,SW,%4zu,%4zu,%4zu,%4zu,"
             "%s,%"
             "2.6lf,%5.2lf\n",
             numLayers, reps, asyncLaunches, numCores, backendStr,
             t / asyncLaunches, gflops * asyncLaunches / t);
    }
    double min = *(std::min_element(times.begin(), times.end()));
    size_t midElt = times.size() / 2;
    std::nth_element(times.begin(), times.begin() + midElt, times.end());
    double median = times[midElt];
    double median_runtime = median / ((double)asyncLaunches);
    double min_runtime = min / ((double)asyncLaunches);
    printf("BenchSummary,AvgPool2dParallelBench,SW,%4zu,%4zu,%4zu,%4zu,%s,%"
           "2.6lf,%2.6lf,%5.2lf,%5.2lf\n",
           numLayers, reps, asyncLaunches, numCores, backendStr, median_runtime,
           min_runtime, gflops / median_runtime, gflops / min_runtime);
    shape_idx++;
  }
}
