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
#include <cstdlib>
#include <future>
#include <random>

#include "Bench.h"

#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Optimizer/GraphOptimizer/GraphOptimizer.h"

using namespace glow;

/*
 * This class implements a performance proxy for a single layer of
 * the BERT network.
 */
class BERTProxyLayerBench : public Benchmark {
  size_t batchSize_;
  size_t hiddenSize_;
  size_t numHeads_;
  size_t numFCSplits_;
  std::unique_ptr<runtime::HostManager> hostManager_;
  std::vector<std::unique_ptr<ExecutionContext>> contexts_;
  size_t asyncLaunchSize_;
  const char *backendStr_;
  ElemKind dtype_;
  size_t elementSize_;

public:
  BERTProxyLayerBench(size_t batchSize_, size_t hiddenSize_, size_t numHeads_,
                      size_t numFCSplits_, size_t asyncLaunchSize_,
                      const char *backendStr_, const char *dtypeStr_)
      : batchSize_(batchSize_), hiddenSize_(hiddenSize_), numHeads_(numHeads_),
        numFCSplits_(numFCSplits_), asyncLaunchSize_(asyncLaunchSize_),
        backendStr_(backendStr_) {

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

  void randomizeTensor(Tensor *tn, PseudoRNG rng) {
    if (dtype_ == ElemKind::FloatTy) {
      tn->getHandle<float>().randomize(0.0f, 1.0f, rng);
    } else if (dtype_ == ElemKind::Float16Ty) {
      tn->getHandle<float16_t>().randomize(0.0f, 1.0f, rng);
    }
  }

  void setTensor(Tensor *tn, float val) {
    if (dtype_ == ElemKind::FloatTy) {
      tn->getHandle<float>().clear(val);
    } else if (dtype_ == ElemKind::Float16Ty) {
      tn->getHandle<float16_t>().clear(val);
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

    // Input Placeholder (batchSize x hiddenSize)
    Placeholder *input = mod->createPlaceholder(
        dtype_, {batchSize_, hiddenSize_}, "input", false);

    // for each context, add input bindings
    for (int i = 0; i < asyncLaunchSize_; i++) {
      randomizeTensor(contexts_[i]->getPlaceholderBindings()->allocate(input),
                      mod->getPRNG());
    }

    // Weights/bias constants for QKV GEMM
    Tensor W_QKV_Tensor(dtype_, {hiddenSize_, 3 * hiddenSize_});
    randomizeTensor(&W_QKV_Tensor, mod->getPRNG());
    Constant *W_QKV = mod->createConstant("W_FC1", W_QKV_Tensor);
    Tensor b_QKV_Tensor(dtype_, {3 * hiddenSize_});
    setTensor(&b_QKV_Tensor, 0.0f);
    Constant *b_QKV = mod->createConstant("b_FC1", b_QKV_Tensor);

    // Weights/bias constants for ZxWo FC
    Tensor W_ZWO_Tensor(dtype_, {hiddenSize_, hiddenSize_});
    randomizeTensor(&W_ZWO_Tensor, mod->getPRNG());
    Constant *W_ZWO = mod->createConstant("W_ZWO", W_ZWO_Tensor);
    Tensor b_ZWO_Tensor(dtype_, {hiddenSize_});
    randomizeTensor(&b_ZWO_Tensor, mod->getPRNG());
    Constant *b_ZWO = mod->createConstant("W_ZWO", b_ZWO_Tensor);

    // Constant scaling factor
    float sqrt_dk_flt =
        (float)(1.0 / std::sqrt(((double)hiddenSize_) / ((double)numHeads_)));

    // Softmax expected output. Not needed for inference
    Tensor expected_Tensor(ElemKind::Int64ITy, {batchSize_, 1});
    Constant *expected = mod->createConstant("expected", expected_Tensor);

    // Weights/bias constants for FC1
    Tensor W_FC1_Tensor(dtype_, {hiddenSize_, 4 * hiddenSize_});
    randomizeTensor(&W_FC1_Tensor, mod->getPRNG());
    Constant *W_FC1 = mod->createConstant("W_FC1", W_FC1_Tensor);
    Tensor b_FC1_Tensor(dtype_, {4 * hiddenSize_});
    randomizeTensor(&b_FC1_Tensor, mod->getPRNG());
    Constant *b_FC1 = mod->createConstant("b_FC1", b_FC1_Tensor);

    // Weights/bias constants for FC2
    Tensor W_FC2_Tensor(dtype_, {4 * hiddenSize_, hiddenSize_});
    randomizeTensor(&W_FC2_Tensor, mod->getPRNG());
    Constant *W_FC2 = mod->createConstant("W_FC2", W_FC2_Tensor);
    Tensor b_FC2_Tensor(dtype_, {hiddenSize_});
    randomizeTensor(&b_FC2_Tensor, mod->getPRNG());
    Constant *b_FC2 = mod->createConstant("b_FC2", b_FC2_Tensor);

    // QKV GEMM
    auto *QKV = fn->createFullyConnected("Gemm_QKV", input, W_QKV, b_QKV);

    // Split into Q, K, V
    std::vector<SliceNode *> outputs(3);
    fn->createSplit("split", QKV, 3, 1, {}, outputs);
    SliceNode *Q = outputs[0];
    SliceNode *K = outputs[1];
    SliceNode *V = outputs[2];

    // Multi-headed attention split
    std::vector<SliceNode *> Qsplits(numHeads_); // batchSize x 64
    std::vector<SliceNode *> Ksplits(numHeads_); // batchSize x 64
    std::vector<SliceNode *> Vsplits(numHeads_); // batchSize x 64
    std::vector<NodeValue> Zsplits(numHeads_);   // batchSize x 64
    fn->createSplit("splitQ", Q, numHeads_, 1, {}, Qsplits);
    fn->createSplit("splitK", K, numHeads_, 1, {}, Ksplits);
    fn->createSplit("splitV", V, numHeads_, 1, {}, Vsplits);

    // BatchMatMul
    for (int i = 0; i < numHeads_; i++) {
      auto *Kt = fn->createTranspose("transpose_" + std::to_string(i),
                                     Ksplits[i], {1, 0});
      // Tmp = Q * K^T
      auto *tmp = fn->createMatMul("matmul_Q_KT_" + std::to_string(i),
                                   Qsplits[i], Kt->getResult());

      // Softmax_output = softmax(Tmp / sqrt(dk))
      auto *sqrt_dk_splat =
          fn->createSplat("sqrt_dk_" + std::to_string(i),
                          tmp->getResult().getType(), sqrt_dk_flt);
      auto *tmp_div =
          fn->createMul("div_" + std::to_string(i), tmp, sqrt_dk_splat);
      auto *softmax_output =
          fn->createSoftMax("softmax_" + std::to_string(i), tmp_div, expected);

      Zsplits[i] = fn->createMatMul("matmul_tmp_v_" + std::to_string(i),
                                    softmax_output, Vsplits[i]);
    }

    auto *Z = fn->createConcat("concat", Zsplits, 1);

    // Z x W_o
    auto *ZWO = fn->createFullyConnected("Gemm_ZWO", Z, W_ZWO, b_ZWO);

    // FC1
    auto *FC1 = fn->createFullyConnected("Gemm_FC1", ZWO, W_FC1, b_FC1);

    // FC2
    auto *FC2 = fn->createFullyConnected("Gemm_FC2", FC1, W_FC2, b_FC2);

    // Save result
    SaveNode *S = fn->createSave("save", FC2);
    for (int i = 0; i < asyncLaunchSize_; i++) {
      contexts_[i]->getPlaceholderBindings()->allocate(S->getPlaceholder());
    }

    // Split FCs
    executeVerticalFCWeightsSplit(fn, numFCSplits_, hiddenSize_);

    CompilationContext ctx;
    hostManager_->addNetwork(std::move(mod), ctx);
    fn->dumpDAG(std::string("BERT.dot"));
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

  // Only counting GEMMs
  double gflops() const {
    double num_flops = 0.0;

    // QKV
    num_flops += 2.0 * hiddenSize_ * 3 * hiddenSize_;

    // BMM
    num_flops += 2.0 * hiddenSize_ * batchSize_;
    num_flops += 2.0 * hiddenSize_ * batchSize_;

    // ZWO
    num_flops += 2.0 * hiddenSize_ * hiddenSize_;

    // FC1
    num_flops += 2.0 * hiddenSize_ * 4 * hiddenSize_;

    // FC2
    num_flops += 2.0 * hiddenSize_ * 4 * hiddenSize_;

    return batchSize_ * num_flops / 1e9;
  }
};

int main(int argc, char *argv[]) {
  printf("Usage: BERTLayerBench batchSize hiddenSize numHeads numFCSplits "
         "numReps numAsyncLaunches backendStr dtypeStr\n");
  assert(argc == 8);
  size_t batchSize = atoi(argv[1]);
  size_t hiddenSize = atoi(argv[2]);
  size_t numHeads = atoi(argv[3]);
  size_t numFCSplits = atoi(argv[4]);
  size_t numReps = atoi(argv[5]);
  size_t numAsyncLaunches = atoi(argv[6]);
  const char *backendStr = argv[7];
  const char *dtypeStr = argv[8];
  assert(numReps > 0);

  BERTProxyLayerBench b(batchSize, hiddenSize, numHeads, numFCSplits,
                        numAsyncLaunches, backendStr, dtypeStr);

  auto times = bench(&b, numReps);
  printf("_,benchName,batchSize,hiddenSize,numHeads,numFCSplits,numReps,"
         "numAsyncLaunches,"
         "backendStr,dtypeStr,averageTime,averageGFLOP\n");
  for (auto t : times) {
    printf("BenchResult,BERTProxyLayerBench,SW,%zu,%zu,%zu,%zu,%zu,%zu,%s,%"
           "s,%f,%f\n",
           batchSize, hiddenSize, numHeads, numFCSplits, numReps,
           numAsyncLaunches, backendStr, dtypeStr, t / numAsyncLaunches,
           b.gflops() * numAsyncLaunches / t);
  }
  double min = *(std::min_element(times.begin(), times.end()));
  size_t midElt = times.size() / 2;
  std::nth_element(times.begin(), times.begin() + midElt, times.end());
  double median = times[midElt];
  double median_runtime = median / ((double)numAsyncLaunches);
  double min_runtime = min / ((double)numAsyncLaunches);
  printf(
      "_,benchName,batchSize,hiddenSize,numHeads,numFCSplits,numReps,"
      "numAsyncLaunches,"
      "backendStr,dtypeStr,medianRuntime,minRuntime,medianGFLOPS,minGFLOPS\n");
  printf("BenchSummary,BERTProxyLayerBench,SW,%zu,%zu,%zu,%zu,%zu,%zu,%s,%s,"
         "%f,%f,%f,%"
         "f\n",
         batchSize, hiddenSize, numHeads, numFCSplits, numReps,
         numAsyncLaunches, backendStr, dtypeStr, median_runtime, min_runtime,
         b.gflops() / median_runtime, b.gflops() / min_runtime);
}
