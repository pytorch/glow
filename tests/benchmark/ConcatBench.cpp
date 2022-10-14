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
#include <fstream>
#include <future>
#include <random>

#include "Bench.h"

#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Optimizer/GraphOptimizer/GraphOptimizer.h"

#include "tests/unittests/BackendTestUtils.h"

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
llvm::cl::OptionCategory ConcatBenchCat("ConcatBench Category");
llvm::cl::opt<bool> checkCorrectness(
    "check-results",
    llvm::cl::desc("Check the correctness of the results against the reference "
                   "backend (Interpreter)"),
    llvm::cl::Optional, llvm::cl::init(false), llvm::cl::cat(ConcatBenchCat));
llvm::cl::opt<bool> dumpOnnx("dump_onnx",
                             llvm::cl::desc("dump onnx text format for model"),
                             llvm::cl::Optional, llvm::cl::init(false),
                             llvm::cl::cat(ConcatBenchCat));

struct ConcatParam {
  dim_t m_;
  dim_t n_;
  dim_t numTensors_;
  dim_t numLayers_;
  dim_t numReps_;
  dim_t numAsyncLaunches_;
  std::string backendStr_;
  std::string devId_;
  ElemKind dtype_;
};

class ConcatBench : public Benchmark {
  ConcatParam param_;
  size_t elementSize_;
  ExecutionContext context_;
  PlaceholderBindings &bindings_;
  std::unique_ptr<runtime::HostManager> hostManager_;

  // Refernce bindings and network:
  ExecutionContext refContext_;
  PlaceholderBindings &refBindings_;
  std::unique_ptr<runtime::HostManager> refHostManager_;

public:
  explicit ConcatBench(ConcatParam param_)
      : param_(param_), bindings_(*context_.getPlaceholderBindings()),
        refBindings_(*refContext_.getPlaceholderBindings()) {
    elementSize_ = 2;
    if (param_.dtype_ == ElemKind::Float16Ty) {
      elementSize_ = 2;

    } else {
      elementSize_ = 4;
    }
  }

  void addConcatNode(std::unique_ptr<Module> &mod, Function *fn,
                     ConcatParam param) {
    // Create multiple chains of Concat nodes
    std::vector<Placeholder *> A(param.numTensors_);
    std::vector<NodeValue> A_broadcast(param.numTensors_);
    std::vector<NodeValue> A_concat(param.numTensors_);
    std::vector<NodeValue> slices(param.numTensors_);

    Placeholder *output;

    for (size_t tensor = 0; tensor < param.numTensors_; tensor++) {
      A[tensor] = mod->createPlaceholder(param.dtype_, {1, param.n_},
                                         "A" + std::to_string(tensor), false);
      A_broadcast[tensor] =
          fn->createBroadcast("A_bcast" + std::to_string(tensor), A[tensor],
                              {param.m_, param.n_}, 0);
    }
    output = mod->createPlaceholder(
        param.dtype_, {1, param.n_ * param.numTensors_}, "output", false);

    for (size_t tensor = 0; tensor < param.numTensors_; tensor++) {
      A_concat[tensor / 2 * 2 + ((tensor % 2) ? 0 : 1)] = A_broadcast[tensor];
    }
    auto *concat = fn->createConcat("concat_0", A_concat, 1);

    for (size_t layer = 1; layer < param.numLayers_; layer++) {
      for (size_t tensor = 0; tensor < param.numTensors_; tensor++) {
        dim_t start_n = tensor / 2 * 2 * param.n_ +
                        ((tensor % 2) ? (3 * param.n_ / 2) : (0));
        dim_t end_n =
            start_n + ((tensor % 2) ? (param.n_ / 2) : (3 * param.n_ / 2));
        slices[tensor] =
            fn->createSlice("slice_" + std::to_string(tensor), concat,
                            {0, start_n}, {param.m_, end_n});
      }
      for (size_t tensor = 0; tensor < param.numTensors_; tensor++) {
        A_concat[tensor / 2 * 2 + ((tensor % 2) ? 0 : 1)] = slices[tensor];
      }
      concat = fn->createConcat("concat_" + std::to_string(layer), A_concat, 1);
    }
    Node *slice = fn->createSlice("slice_final", concat, {0, 0},
                                  {1, param.n_ * param.numTensors_});
    fn->createSave("save", slice, output);
  }

  void setupInternal(bool isRef) {
    // Setup host manager
    std::string backendStr = isRef ? "Interpreter" : param_.backendStr_.c_str();
    std::vector<std::unique_ptr<runtime::DeviceConfig>> configs;
    auto config = glow::make_unique<runtime::DeviceConfig>(backendStr.c_str());
    if (param_.devId_ != "") {
      config->parameters["DeviceID"] = param_.devId_.c_str();
    }
    configs.push_back(std::move(config));
    if (isRef) {
      refHostManager_ =
          glow::make_unique<runtime::HostManager>(std::move(configs));
    } else {
      hostManager_ =
          glow::make_unique<runtime::HostManager>(std::move(configs));
    }

    std::unique_ptr<Module> mod(new Module);
    auto fn = mod->createFunction("singleNode");

    addConcatNode(mod, fn, param_);

    CompilationContext ctx;
    ctx.dumpFinalGraph = true;
    ctx.serializeCompiledDAG = dumpOnnx;
    if (isRef) {
      EXIT_ON_ERR(refHostManager_->addNetwork(std::move(mod), ctx));
    } else {
      EXIT_ON_ERR(hostManager_->addNetwork(std::move(mod), ctx));
    }
  }

  void checkOutput() {
    // First run on the reference backend
    dispatchInference("singleNode", refHostManager_.get(), refContext_,
                      param_.numAsyncLaunches_,
                      /*useNewExecutionContext*/ true);
    Tensor *refTensor =
        refBindings_.get(refBindings_.getPlaceholderByNameSlow("output"));
    CHECK(refTensor) << "Reference Tensor not found";

    Tensor *noRefTensor =
        bindings_.get(bindings_.getPlaceholderByNameSlow("output"));
    CHECK(noRefTensor) << "non-reference Tensor not found";

    // Compare the tensors
    if (!noRefTensor->isEqual(*refTensor)) {
      noRefTensor->dump();
      refTensor->dump();
      LOG(FATAL) << "Tensors don't match\n";
    } else {
      LOG(INFO) << "Tensors match\n";
    }
  }

  void setup() override {
    if (checkCorrectness) {
      setupInternal(/* isRef */ true);
    }
    setupInternal(/* isRef */ false);
  }
  void run() override {
    dispatchInference("singleNode", hostManager_.get(), context_,
                      param_.numAsyncLaunches_,
                      /*useNewExecutionContext*/ true);
    if (checkCorrectness) {
      checkOutput();
    }
  }

  void teardown() override {}

  // Two inputs per layer and one output
  double gbytes() const {
    return elementSize_ * param_.m_ * param_.n_ * param_.numTensors_ *
           param_.numLayers_ / 1e9;
  }
};

#define DEVICE_ID 9

ConcatParam parseArgs(int argc, char *argv[]) {
  ConcatParam param;

  param.m_ = atoi(argv[1]);
  param.n_ = atoi(argv[2]);
  param.numTensors_ = atoi(argv[3]);
  param.numLayers_ = atoi(argv[4]);
  param.numReps_ = atoi(argv[5]);
  param.numAsyncLaunches_ = atoi(argv[6]);
  param.backendStr_ = std::string(argv[7]);
  if (std::string(argv[8]) == "Float16") {
    param.dtype_ = ElemKind::Float16Ty;
  } else if (std::string(argv[8]) == "Float32") {
    param.dtype_ = ElemKind::FloatTy;
  } else {
    llvm_unreachable("Invalid dtype");
  }

  printf("m %zu\n", (size_t)param.m_);
  printf("n %zu\n", (size_t)param.n_);
  printf("numTensors %zu\n", (size_t)param.numTensors_);
  printf("numLayers %zu\n", (size_t)param.numLayers_);
  printf("numReps %zu\n", (size_t)param.numReps_);
  printf("numAsyncLaunches %zu\n", (size_t)param.numAsyncLaunches_);
  printf("backendStr %s\n", param.backendStr_.c_str());
  printf("dtypeStr %s\n", argv[8]);

  if (argc > DEVICE_ID) {
    printf("devId %s\n", argv[DEVICE_ID]);
    param.devId_ = std::string(argv[DEVICE_ID]);
  } else {
    param.devId_ = std::string("");
  }
  printf("\n\n");
  return param;
}

int main(int argc, char *argv[]) {
  printf("Concat Microbenchmark\n");
  printf("Usage: ConcatBench m(Int) n(Int) numTensors(Int) "
         "numLayers(Int) numReps(Int) "
         "numAsyncLaunches(Int) backendStr(String) "
         "dtypeStr(\"Float16\"|\"Float32\") dev_id(Int)\n");
  printf("Standard Glow command-line options may be passed via the GLOW_OPTS "
         "environment variable\n");
  benchParseGlowOpts(argc, argv);

  ConcatParam param = parseArgs(argc, argv);
  if (param.numTensors_ % 2 != 0) {
    fprintf(stderr, "Error: numTensors must be a multiple of 2!\n");
    return -1;
  }

  ConcatBench b(param);
  auto times = bench(&b, param.numReps_);
  printf("_,benchName,_,m,n,numTensors,numLayers,numReps,numAsyncLaunches,"
         "backendStr,dtypeStr,runtime,gbytesPerSecPerChain\n");
  for (auto t : times) {
    printf("BenchResult,ConcatBench,SW,%4u,%4u,%4u,%4u,%4u,%4u,%s,%s,"
           "%2.6lf,%5.2lf\n",
           static_cast<unsigned>(param.m_), static_cast<unsigned>(param.n_),
           static_cast<unsigned>(param.numTensors_),
           static_cast<unsigned>(param.numLayers_),
           static_cast<unsigned>(param.numReps_),
           static_cast<unsigned>(param.numAsyncLaunches_),
           param.backendStr_.c_str(), argv[8], t / param.numAsyncLaunches_,
           b.gbytes() * param.numAsyncLaunches_ / t);
  }
  double min = *(std::min_element(times.begin(), times.end()));
  size_t midElt = times.size() / 2;
  std::nth_element(times.begin(), times.begin() + midElt, times.end());
  double median = times[midElt];
  double medianRuntime = median / ((double)param.numAsyncLaunches_);
  double minRuntime = min / ((double)param.numAsyncLaunches_);
  printf("_,benchName,_,m,n,numTensors,numLayers,numReps,numAsyncLaunches,"
         "backendStr,dtypeStr,medianRuntime,minRuntime,"
         "medianGbytesPerSecPerChain,maxGbytesPerSecPerChain\n");
  printf("BenchSummary,ConcatBench,SW,%4u,%4u,%4u,%4u,%4u,%4u,%s,%s,"
         "%2.6lf,%2.6lf,%"
         "5.2lf, %5.2lf\n",
         static_cast<unsigned>(param.m_), static_cast<unsigned>(param.n_),
         static_cast<unsigned>(param.numTensors_),
         static_cast<unsigned>(param.numLayers_),
         static_cast<unsigned>(param.numReps_),
         static_cast<unsigned>(param.numAsyncLaunches_),
         param.backendStr_.c_str(), argv[8], medianRuntime, minRuntime,
         b.gbytes() / medianRuntime, b.gbytes() / minRuntime);
}
