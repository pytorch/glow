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

vector<vector<conv_param_t<2>>> shapes_2d = {
    // MB, IC, OC, IH, IW, G, KH, KW, stride_h, stride_w,
    // pad_h_top, pad_w_left, pad_h_bottom, pad_w_right
    // 2D convolutions
    // regular
    {conv_param_t<>(1, 128, 128, {56, 56}, 1, {3, 3}, {1, 1}, {1, 1, 1, 1})},
    // groupwise
    {conv_param_t<>(1, 128, 128, {56, 56}, 32, {3, 3}, {1, 1}, {1, 1, 1, 1})},
    // DW
    {conv_param_t<>(1, 272, 272, {47, 125}, 272, {3, 3}, {1, 1}, {1, 1, 1, 1})},
    // Pointwise
    {conv_param_t<>(1, 128, 128, {56, 56}, 1, {1, 1}, {1, 1}, {0, 0, 0, 0})},
    // bottleneck blocks
    {conv_param_t<>(1, 256, 128, {56, 56}, 1, {1, 1}, {1, 1}, {0, 0, 0, 0}),
     conv_param_t<>(1, 128, 128, {56, 56}, 32, {3, 3}, {1, 1}, {1, 1, 1, 1}),
     conv_param_t<>(1, 128, 256, {56, 56}, 1, {1, 1}, {1, 1}, {0, 0, 0, 0})},
    {conv_param_t<>(1, 512, 256, {28, 28}, 1, {1, 1}, {1, 1}, {0, 0, 0, 0}),
     conv_param_t<>(1, 256, 256, {28, 28}, 32, {3, 3}, {1, 1}, {1, 1, 1, 1}),
     conv_param_t<>(1, 256, 512, {28, 28}, 1, {1, 1}, {1, 1}, {0, 0, 0, 0})},
    {conv_param_t<>(1, 1024, 512, {14, 14}, 1, {1, 1}, {1, 1}, {0, 0, 0, 0}),
     conv_param_t<>(1, 512, 512, {14, 14}, 32, {3, 3}, {1, 1}, {1, 1, 1, 1}),
     conv_param_t<>(1, 512, 1024, {14, 14}, 1, {1, 1}, {1, 1}, {0, 0, 0, 0})},
    {conv_param_t<>(1, 2048, 1024, {7, 7}, 1, {1, 1}, {1, 1}, {0, 0, 0, 0}),
     conv_param_t<>(1, 1024, 1024, {7, 7}, 32, {3, 3}, {1, 1}, {1, 1, 1, 1}),
     conv_param_t<>(1, 1024, 2048, {7, 7}, 1, {1, 1}, {1, 1}, {0, 0, 0, 0})}

};

/*
 * Benchmark a number of Conv2d operators with representative input shapes.
 * There are a number of parallel Conv2d nodes which are created, one
 * per core. Each core handles one weight matrix. Then these are chained
 * together in multiple layers. After each layer, output tensor is passed to the
 * next layer.
 */
class Int8Conv2dParallelBench : public Benchmark {
  /// Matrices.
  std::vector<conv_param_t<2>> input_shapes_;
  size_t numLayers_;
  PlaceholderBindings bindings_;
  std::unique_ptr<runtime::HostManager> hostManager_;
  size_t asyncLaunchSize_;
  size_t numCores_;
  const char *backendStr_;
  const char *devId_;

public:
  Int8Conv2dParallelBench(vector<conv_param_t<2>> &input_shapes_,
                          size_t numLayers_, size_t asyncLaunchSize_,
                          size_t numCores_, const char *backendStr_,
                          const char *devId_)
      : input_shapes_(input_shapes_), numLayers_(numLayers_),
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

    dim_t N, IC, IH, IW, OC, OH, OW;
    if (input_shapes_.size() == 1) {
      N = input_shapes_[0].MB;
      IC = input_shapes_[0].IC;
      IH = input_shapes_[0].IN_DIM[0];
      IW = input_shapes_[0].IN_DIM[1];
      OC = input_shapes_[0].OC;
      OH = input_shapes_[0].OUT_DIM[0];
      OW = input_shapes_[0].OUT_DIM[1];
    } else {
      N = input_shapes_[0].MB;
      IC = input_shapes_[0].IC;
      IH = input_shapes_[0].IN_DIM[0];
      IW = input_shapes_[0].IN_DIM[1];
      OC = input_shapes_[input_shapes_.size() - 1].OC;
      OH = input_shapes_[input_shapes_.size() - 1].OUT_DIM[0];
      OW = input_shapes_[input_shapes_.size() - 1].OUT_DIM[1];
    }
    std::unique_ptr<Module> mod(new Module);
    auto fn = mod->createFunction("singleNode");

    std::vector<Node *> cur(numCores_);
    std::vector<Placeholder *> filters(numCores_ * input_shapes_.size());
    std::vector<Placeholder *> bias(numCores_ * input_shapes_.size());
    std::vector<Node *> conv(numCores_ * input_shapes_.size());
    std::vector<Placeholder *> input(numCores_);
    std::vector<Placeholder *> output(numCores_);

    for (size_t core = 0; core < numCores_; core++) {
      input[core] =
          mod->createPlaceholder(ElemKind::Int8QTy, {N, IH, IW, IC}, 1.0, 0,
                                 "input_" + std::to_string(core), false);
      output[core] =
          mod->createPlaceholder(ElemKind::Int8QTy, {N, OH, OW, OC}, 1.0, 0,
                                 "output_" + std::to_string(core), false);
      cur[core] = input[core];
    }

    for (size_t layer = 0; layer < numLayers_; layer++) {
      for (size_t core = 0; core < numCores_; core++) {
        size_t conv_ops = 0;
        for (auto conv_param : input_shapes_) {
          filters[core * input_shapes_.size() + conv_ops] =
              mod->createPlaceholder(ElemKind::Int8QTy,
                                     {(dim_t)(conv_param.OC),
                                      (dim_t)(conv_param.K[0]),
                                      (dim_t)(conv_param.K[1]),
                                      (dim_t)(conv_param.IC / conv_param.G)},
                                     1.0, 0,
                                     "filters_" + std::to_string(core) + "_" +
                                         std::to_string(conv_ops),
                                     false);
          bias[core * input_shapes_.size() + conv_ops] = mod->createPlaceholder(
              ElemKind::Int32QTy, {(dim_t)(conv_param.OC)}, 1.0, 0,
              "bias_" + std::to_string(core) + "_" + std::to_string(conv_ops),
              false);
          bindings_.allocate(filters[core * input_shapes_.size() + conv_ops])
              ->getHandle<int8_t>()
              .clear(0);
          bindings_.allocate(bias[core * input_shapes_.size() + conv_ops])
              ->getHandle<int32_t>()
              .clear(0);
          auto outTy = mod->uniqueType(
              ElemKind::Int8QTy,
              {(dim_t)(conv_param.MB), (dim_t)(conv_param.OUT_DIM[0]),
               (dim_t)(conv_param.OUT_DIM[1]), (dim_t)(conv_param.OC)},
              1.0, 0);
          conv[core * input_shapes_.size() + conv_ops] = fn->createConv(
              "conv" + std::to_string(core) + "_" + std::to_string(layer) +
                  "_" + std::to_string(conv_ops),
              cur[core], filters[core * input_shapes_.size() + conv_ops],
              bias[core * input_shapes_.size() + conv_ops], outTy,
              {(unsigned int)(conv_param.K[0]),
               (unsigned int)(conv_param.K[1])},
              {(unsigned int)(conv_param.stride[0]),
               (unsigned int)(conv_param.stride[1])},
              {(unsigned int)(conv_param.pad[0]),
               (unsigned int)(conv_param.pad[1]),
               (unsigned int)(conv_param.pad[2]),
               (unsigned int)(conv_param.pad[3])},
              (unsigned int)(conv_param.G),
              {(unsigned int)(conv_param.dilation[0]),
               (unsigned int)(conv_param.dilation[1])});
          cur[core] = conv[core * input_shapes_.size() + conv_ops];
          conv_ops += 1;
        }
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

  printf("Int8Conv2dParallel Microbenchmark\n");
  printf(
      "Usage: Int8Conv2dParallelBench numLayers(Int) "
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
  printf("Start Int8Conv2dParallelBench\n");
  size_t shape_idx = 0;
  size_t total_input_shapes = shapes_2d.size();
  for (auto shapes : shapes_2d) {
    double gflops = 0;
    string shape_info = "";
    for (auto shape : shapes) {
      gflops += 2.0 * shape.G * (shape.IC / shape.G) * shape.K[0] * shape.K[1] *
                (shape.OC / shape.G) * shape.OUT_DIM[0] * shape.OUT_DIM[1];
      if (shape_info != "") {
        shape_info += ";";
      }
      shape_info += shape.toString();
    }
    gflops *= numLayers * numCores / 1e9;
    printf("\n=====Input shape %zu/%zu: %s\n", shape_idx, total_input_shapes,
           shape_info.c_str());
    Int8Conv2dParallelBench b(shapes, numLayers, asyncLaunches, numCores,
                              backendStr, dev_id);
    auto times = bench(&b, reps);
    for (auto t : times) {
      printf("BenchResult,Conv2dParallelBench,SW,%4zu,%4zu,%4zu,%4zu,"
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
    printf("BenchSummary,Conv2dParallelBench,SW,%4zu,%4zu,%4zu,%4zu,%s,%"
           "2.6lf,%2.6lf,%5.2lf,%5.2lf\n",
           numLayers, reps, asyncLaunches, numCores, backendStr, median_runtime,
           min_runtime, gflops / median_runtime, gflops / min_runtime);
    shape_idx++;
  }
}
