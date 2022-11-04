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

#include "tests/unittests/BackendTestUtils.h"

using namespace glow;

/*
 * This class implements a transpose microbenchmark. There are multiple
 * layers of transpose, followed by an Add with the tensor from the previous
 * layer. This benchmark only supports 021 transpose at this moment, we will
 * add more general coverage and associated tests later.
 *
 * Microbenchmarks are generally useful for understanding performance
 * through targeted experiementation and are not representative of
 * end-to-end workloads.
 */

llvm::cl::OptionCategory TransposeBenchCat("TransposeBench Category");
llvm::cl::opt<bool> checkCorrectness(
    "check-results",
    llvm::cl::desc("Check the correctness of the results against the reference "
                   "backend (Interpreter)"),
    llvm::cl::Optional, llvm::cl::init(false),
    llvm::cl::cat(TransposeBenchCat));
llvm::cl::opt<bool> dumpOnnx("dump_onnx",
                             llvm::cl::desc("dump onnx text format for model"),
                             llvm::cl::Optional, llvm::cl::init(false),
                             llvm::cl::cat(TransposeBenchCat));

struct TransposeParam {
  dim_t batchSize_;
  dim_t m_;
  dim_t n_;
  dim_t numLayers_;
  dim_t numReps_;
  dim_t numAsyncLaunches_;
  dim_t numSplits_;
  std::string backendStr_;
  std::string devId_;
  ElemKind dtype_;
};

class TransposeBench : public Benchmark {
  TransposeParam param_;
  ExecutionContext context_;
  PlaceholderBindings &bindings_;
  std::unique_ptr<runtime::HostManager> hostManager_;

  // Refernce bindings and network:
  ExecutionContext refContext_;
  PlaceholderBindings &refBindings_;
  std::unique_ptr<runtime::HostManager> refHostManager_;

public:
  explicit TransposeBench(TransposeParam param_)
      : param_(param_), bindings_(*context_.getPlaceholderBindings()),
        refBindings_(*refContext_.getPlaceholderBindings()) {}

  void addTransposeNode(std::unique_ptr<Module> &mod, Function *fn,
                        TransposeParam param, bool isRef) {

    PlaceholderBindings &bindings = isRef ? refBindings_ : bindings_;
    auto *inputA = mod->createPlaceholder(
        param.dtype_, {param.batchSize_, param.m_, param.n_}, "inputA", false);
    auto *inputB = mod->createPlaceholder(
        param.dtype_, {param.batchSize_, param.n_, param.m_}, "inputB", false);
    if (param.dtype_ == ElemKind::Float16Ty) {
      bindings.allocate(inputA)->getHandle<float16>().randomize(-1.f, 1.f,
                                                                mod->getPRNG());

      bindings.allocate(inputB)->getHandle<float16>().randomize(-1.f, 1.f,
                                                                mod->getPRNG());

    } else {
      assert(param.dtype_ == ElemKind::FloatTy);
      bindings.allocate(inputA)->getHandle<float>().randomize(-1.f, 1.f,
                                                              mod->getPRNG());
      bindings.allocate(inputB)->getHandle<float>().randomize(-1.f, 1.f,
                                                              mod->getPRNG());
    }
    glow::Placeholder *output;
    if (param.numLayers_ % 2) {
      output = mod->createPlaceholder(param.dtype_,
                                      {param.batchSize_, param.n_, param.m_},
                                      "output", false);

    } else {
      output = mod->createPlaceholder(param.dtype_,
                                      {param.batchSize_, param.m_, param.n_},
                                      "output", false);
    }
    bindings.allocate(output);
    Node *cur = inputA;

    for (dim_t layer = 0; layer < param.numLayers_; layer++) {
      auto *xp = fn->createTranspose("transpose_" + std::to_string(layer), cur,
                                     {0, 2, 1});
      Node *ad;

      if (layer % 2) {
        ad = fn->createAdd("add_" + std::to_string(layer), inputA, xp);
      } else {
        ad = fn->createAdd("add_" + std::to_string(layer), inputB, xp);
      }
      cur = ad;
    }

    fn->createSave("save1", cur, output);
    ::glow::convertPlaceholdersToConstants(fn, bindings,
                                           {inputA, inputB, output});
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

    addTransposeNode(mod, fn, param_, isRef);

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

  // Each layer reads the tensor thrice, and writes the tensor twice
  double gbytes() const {
    return (5.0 * param_.numLayers_ * param_.batchSize_ * param_.n_ *
            param_.m_) /
           1e9;
  }
};

#define DEVICE_ID 11

TransposeParam parseArgs(int argc, char *argv[]) {
  TransposeParam param;
  param.batchSize_ = atoi(argv[1]);
  param.m_ = atoi(argv[2]);
  param.n_ = atoi(argv[3]);
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

  printf("batchsize %zu\n", (size_t)param.batchSize_);
  printf("m %zu\n", (size_t)param.m_);
  printf("n %zu\n", (size_t)param.n_);
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
  printf("Transpose Microbenchmark\n");
  printf("Usage: TransposeBench batchSize(Int) n(Int) numLayers(Int) "
         "numReps(Int) numAsyncLaunches(Int) numTransposeChains(Int) "
         "backendStr(String) dtypeStr(\"Float16\"|\"Float32\") dev_id(Int)\n");
  printf("Standard Glow command-line options may be passed via the GLOW_OPTS "
         "environment variable\n");
  benchParseGlowOpts(argc, argv);
  assert(argc == 10 || argc == 11);

  std::vector<TransposeParam> params;
  std::string runHeader;
  std::string runPrefix;

  TransposeParam param = parseArgs(argc, argv);
  params.push_back(param);

  runHeader = std::string("_,benchName,_,batchsize,m,n,numLayers,numReps,"
                          "numAsyncLaunches,numSplits,"
                          "backendStr,dtypeStr\n");
  runPrefix = std::string(
      strFormat("TransposeBench,SW,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%s,%s",
                (size_t)param.batchSize_, (size_t)param.m_, (size_t)param.n_,
                (size_t)param.numLayers_, (size_t)param.numReps_,
                (size_t)param.numAsyncLaunches_, (size_t)param.numSplits_,
                argv[8], argv[9]));

  TransposeBench b(param);
  auto times = bench(&b, param.numReps_);

  printf("%s,runtime,gBytesPerSec\n", runHeader.c_str());
  for (auto t : times) {
    printf("BenchResult,%s,%f,%f\n", runPrefix.c_str(),
           t / param.numAsyncLaunches_,
           b.gbytes() * param.numAsyncLaunches_ / t);
  }

  double min = *(std::min_element(times.begin(), times.end()));
  dim_t midElt = times.size() / 2;
  std::nth_element(times.begin(), times.begin() + midElt, times.end());
  double median = times[midElt];
  double medianRuntime = median / ((double)param.numAsyncLaunches_);
  double minRuntime = min / ((double)param.numAsyncLaunches_);

  printf("%s,medianRuntime,minRuntime,medianGBPerSec,maxGBPerSec\n",
         runHeader.c_str());
  printf("BenchSummary,%s,%f,%f,%f,%f\n", runPrefix.c_str(), medianRuntime,
         minRuntime, b.gbytes() / medianRuntime, b.gbytes() / minRuntime);
}
