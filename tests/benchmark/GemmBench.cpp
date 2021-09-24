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
 * This class implements a GEMM/FC microbenchmark. There are a set of
 * (m x k) * (k x n) = (m x n) matrix multiplications, chained together in
 * multiple layers.
 *
 * Microbenchmarks are generally useful for understanding performance
 * through targeted experiementation and are not representative of
 * end-to-end workloads.
 */

llvm::cl::OptionCategory GemmBenchCat("GemmBench Category");
llvm::cl::opt<bool> checkCorrectness(
    "check-results",
    llvm::cl::desc("Check the correctness of the results against the reference "
                   "backend (Interpreter)"),
    llvm::cl::Optional, llvm::cl::init(false), llvm::cl::cat(GemmBenchCat));
llvm::cl::opt<bool> dumpOnnx("dump_onnx",
                             llvm::cl::desc("dump onnx text format for model"),
                             llvm::cl::Optional, llvm::cl::init(false),
                             llvm::cl::cat(GemmBenchCat));

struct GemmParam {
  dim_t m_;
  dim_t n_;
  dim_t k_;
  dim_t numLayers_;
  dim_t numReps_;
  dim_t numAsyncLaunches_;
  dim_t numSplits_;
  std::string backendStr_;
  std::string devId_;
  ElemKind dtype_;
};

class GemmBench : public Benchmark {
  GemmParam param_;
  ExecutionContext context_;
  PlaceholderBindings &bindings_;
  std::unique_ptr<runtime::HostManager> hostManager_;

  // Refernce bindings and network:
  ExecutionContext refContext_;
  PlaceholderBindings &refBindings_;
  std::unique_ptr<runtime::HostManager> refHostManager_;

public:
  explicit GemmBench(GemmParam param_)
      : param_(param_), bindings_(*context_.getPlaceholderBindings()),
        refBindings_(*refContext_.getPlaceholderBindings()) {}

  void addGemmNode(std::unique_ptr<Module> &mod, Function *fn, GemmParam param,
                   bool isRef) {
    PlaceholderBindings &bindings = isRef ? refBindings_ : bindings_;
    auto *input = mod->createPlaceholder(param.dtype_, {param.m_, param.k_},
                                         "input", false);
    if (param.dtype_ == ElemKind::Float16Ty) {
      bindings.allocate(input)->getHandle<float16>().randomize(-1.f, 1.f,
                                                               mod->getPRNG());
    } else {
      assert(param.dtype_ == ElemKind::FloatTy);
      bindings.allocate(input)->getHandle<float>().randomize(-1.f, 1.f,
                                                             mod->getPRNG());
    }
    auto *output = mod->createPlaceholder(param.dtype_, {param.m_, param.n_},
                                          "output", false);
    bindings.allocate(output);
    Node *cur = input;

    Placeholder *ones;
    if (param.k_ > param.n_) {
      ones = mod->createPlaceholder(
          param.dtype_, {param.m_ * (param.k_ - param.n_)}, "ones", false);
      if (param.dtype_ == ElemKind::Float16Ty) {
        bindings.allocate(ones)->getHandle<float16_t>().clear(1.0);
      } else if (param.dtype_ == ElemKind::FloatTy) {
        bindings.allocate(ones)->getHandle<float>().clear(1.0);
      }
    }

    Placeholder *weights;
    Placeholder *bias;

    // Create multiple layers of FC nodes
    for (size_t layer = 0; layer < param.numLayers_; layer++) {
      weights =
          mod->createPlaceholder(param.dtype_, {param.k_, param.n_},
                                 "weights" + std::to_string(layer), false);
      bias = mod->createPlaceholder(param.dtype_, {param.n_},
                                    "bias" + std::to_string(layer), false);

      if (param.dtype_ == ElemKind::Float16Ty) {
        bindings.allocate(weights)->getHandle<float16_t>().randomize(
            -1.f, 1.f, mod->getPRNG());
        bindings.allocate(bias)->getHandle<float16_t>().clear(32);
      } else if (param.dtype_ == ElemKind::FloatTy) {
        bindings.allocate(weights)->getHandle<float>().randomize(
            -1.f, 1.f, mod->getPRNG());
        bindings.allocate(bias)->getHandle<float>().clear(32);
      }

      Node *fc;
      fc = fn->createFullyConnected("fc_" + std::to_string(layer), cur, weights,
                                    bias);
      cur = fc;

      // Handle non-square cases
      if (param.k_ > param.n_ && layer < (param.numLayers_ - 1)) {
        Node *reshape1 = fn->createReshape("reshape1_" + std::to_string(layer),
                                           fc, {param.m_ * param.n_});
        Node *concat = fn->createConcat("concat_" + std::to_string(layer),
                                        {reshape1, ones}, 0);
        Node *reshape2 = fn->createReshape("reshape2_" + std::to_string(layer),
                                           concat, {param.m_, param.k_});
        cur = reshape2;
      } else if (param.k_ < param.n_ && layer < (param.numLayers_ - 1)) {
        Node *slice = fn->createSlice("slice_" + std::to_string(layer), fc,
                                      {0, 0}, {param.m_, param.k_});
        cur = slice;
      }
    }
    fn->createSave("save1", cur, output);
    ::glow::convertPlaceholdersToConstants(fn, bindings, {input, output});
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

    addGemmNode(mod, fn, param_, isRef);

    // Split weights
    if (param_.numSplits_ > 1) {
      executeVerticalFCWeightsSplit(fn, param_.numSplits_, param_.n_);
    }

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

  double gflops() const {
    return 2.0 * param_.m_ * param_.n_ * param_.k_ * param_.numLayers_ / 1e9;
  }
};

#define DEVICE_ID 10

GemmParam parseArgs(int argc, char *argv[]) {
  GemmParam param;

  param.m_ = atoi(argv[1]);
  param.n_ = atoi(argv[2]);
  param.k_ = atoi(argv[3]);
  param.numLayers_ = atoi(argv[4]);
  param.numReps_ = atoi(argv[5]);
  param.numAsyncLaunches_ = atoi(argv[6]);
  param.numSplits_ = atoi(argv[7]);
  param.backendStr_ = std::string(argv[8]);
  if (std::string(argv[9]) == "Float16") {
    param.dtype_ = ElemKind::Float16Ty;
  } else if (std::string(argv[9]) == "Float32") {
    param.dtype_ = ElemKind::FloatTy;
  } else {
    llvm_unreachable("Invalid dtype");
  }

  printf("m %zu\n", (size_t)param.m_);
  printf("n %zu\n", (size_t)param.n_);
  printf("k %zu\n", (size_t)param.k_);
  printf("numLayers %zu\n", (size_t)param.numLayers_);
  printf("numReps %zu\n", (size_t)param.numReps_);
  printf("numAsyncLaunches %zu\n", (size_t)param.numAsyncLaunches_);
  printf("numSplits %zu\n", (size_t)param.numSplits_);
  printf("backendStr %s\n", param.backendStr_.c_str());
  printf("dtypeStr %s\n", argv[9]);

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
  printf("GEMM Microbenchmark\n");
  printf("Usage: GemmBench m(Int) n(Int) k(Int) numLayers(Int) numReps(Int) "
         "numAsyncLaunches(Int) numSplits(Int) backendStr(String) "
         "dtypeStr(\"Float16\"|\"Float32\") dev_id(Int)\n");
  printf("Standard Glow command-line options may be passed via the GLOW_OPTS "
         "environment variable\n");
  benchParseGlowOpts(argc, argv);

  std::vector<GemmParam> params;
  std::string runHeader;
  std::string runPrefix;

  // Using a config file
  if (argc == 2) {
    auto fname = std::string(argv[1]);
    std::ifstream fin(fname.c_str());
    if (!fin) {
      std::cout << "Could not open file: " << fname << std::endl;
      exit(0);
    }
    std::string line;
    while (getline(fin, line)) {
      std::array<char, 1024> buf;
      char *saveptr = nullptr;
      std::vector<char *> argVec;
      strcpy(buf.data(), line.c_str());
      char *ptr = strtok_r(buf.data(), " ", &saveptr);
      while (ptr != nullptr) {
        argVec.push_back(ptr);
        ptr = strtok_r(nullptr, " ", &saveptr);
      }
      GemmParam param = parseArgs(argVec.size(), argVec.data());
      params.push_back(param);
      runHeader = std::string("_,benchName,_,filename");
      runPrefix = std::string(strFormat("GemmBench,SW,%s", fname.c_str()));
    }
  } else if (argc == 10 || argc == 11) {
    GemmParam param = parseArgs(argc, argv);
    params.push_back(param);
    runHeader = std::string(
        "_,benchName,_,m,n,k,numLayers,numReps,numAsyncLaunches,numSplits,"
        "backendStr,dtypeStr\n");
    runPrefix = std::string(strFormat(
        "GemmBench,SW,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%s,%s", (size_t)param.m_,
        (size_t)param.n_, (size_t)param.k_, (size_t)param.numLayers_,
        (size_t)param.numReps_, (size_t)param.numAsyncLaunches_,
        (size_t)param.numSplits_, argv[8], argv[9]));
  } else {
    llvm_unreachable("Invalid command line");
  }

  for (auto param : params) {
    GemmBench b(param);
    auto times = bench(&b, param.numReps_);

    printf("%s,runtime,gflopPerSec\n", runHeader.c_str());
    for (auto t : times) {
      printf("BenchResult,%s,%f,%f\n", runPrefix.c_str(),
             t / param.numAsyncLaunches_,
             b.gflops() * param.numAsyncLaunches_ / t);
    }
    double min = *(std::min_element(times.begin(), times.end()));
    dim_t midElt = times.size() / 2;
    std::nth_element(times.begin(), times.begin() + midElt, times.end());
    double median = times[midElt];
    double medianRuntime = median / ((double)param.numAsyncLaunches_);
    double minRuntime = min / ((double)param.numAsyncLaunches_);
    printf("%s,medianRuntime,minRuntime,medianGflopPerSec,maxGflopPerSec\n",
           runHeader.c_str());
    printf("BenchSummary,%s,%f,%f,%f,%f\n", runPrefix.c_str(), medianRuntime,
           minRuntime, b.gflops() / medianRuntime, b.gflops() / minRuntime);
  }
}
