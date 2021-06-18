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
  dim_t maxSequenceLength_;
  dim_t batchSize_;
  dim_t hiddenSize_;
  dim_t numHeads_;
  dim_t numCores_;
  std::unique_ptr<runtime::HostManager> hostManager_;
  std::vector<std::unique_ptr<ExecutionContext>> contexts_;
  dim_t asyncLaunchSize_;
  const char *backendStr_;
  ElemKind dtype_;
  ElemKind FCWeightType_;
  ElemKind FCBiasType_;
  float FCWeightScale_;
  int32_t FCWeightOffset_;
  bool quantize;

public:
  BERTProxyLayerBench(dim_t maxSequenceLength_, dim_t batchSize_,
                      dim_t hiddenSize_, dim_t numHeads_, dim_t numCores_,
                      dim_t asyncLaunchSize_, const char *backendStr_,
                      const char *dtypeStr_, const char *useInt8FCs)
      : maxSequenceLength_(maxSequenceLength_), batchSize_(batchSize_),
        hiddenSize_(hiddenSize_), numHeads_(numHeads_), numCores_(numCores_),
        asyncLaunchSize_(asyncLaunchSize_), backendStr_(backendStr_) {

    dtype_ = ElemKind::Float16Ty;
    quantize = false;
    if (std::string(dtypeStr_) == "Float16") {
      dtype_ = ElemKind::Float16Ty;
      FCWeightType_ = ElemKind::Float16Ty;
      FCBiasType_ = ElemKind::Float16Ty;
      FCWeightScale_ = 1.0f;
      FCWeightOffset_ = 0;
    } else if (std::string(dtypeStr_) == "Float32") {
      dtype_ = ElemKind::FloatTy;
      FCWeightType_ = ElemKind::FloatTy;
      FCBiasType_ = ElemKind::FloatTy;
      FCWeightScale_ = 1.0f;
      FCWeightOffset_ = 0;
    }
    // If quantization is requested then use Int8/Int32
    if (std::string(useInt8FCs) == "True") {
      FCWeightType_ = ElemKind::Int8QTy;
      FCBiasType_ = ElemKind::Int32QTy;
      quantize = true;
      FCWeightScale_ = 1.0;
      FCWeightOffset_ = 128;
    }
  }

  // Handle different tensor types
  void randomizeTensor(Tensor *tn, PseudoRNG rng) {
    if (tn->getElementType() == ElemKind::FloatTy) {
      tn->getHandle<float_t>().randomize(0.0f, 1.0f, rng);
    } else if (tn->getElementType() == ElemKind::Float16Ty) {
      tn->getHandle<float16_t>().randomize(0.0f, 1.0f, rng);
    } else if (tn->getElementType() == ElemKind::Int8QTy) {
      tn->getHandle<int8_t>().randomize(-127, 127, rng);
    } else if (tn->getElementType() == ElemKind::Int32QTy) {
      tn->getHandle<int32_t>().randomize(-128, 128, rng);
    }
  }

  // Handle different tensor types
  void setTensor(Tensor *tn, float val) {
    if (tn->getElementType() == ElemKind::FloatTy) {
      tn->getHandle<float_t>().clear(val);
    } else if (tn->getElementType() == ElemKind::Float16Ty) {
      tn->getHandle<float16_t>().clear(val);
    } else if (tn->getElementType() == ElemKind::Int8QTy) {
      tn->getHandle<int8_t>().clear(val);
    } else if (tn->getElementType() == ElemKind::Int32QTy) {
      tn->getHandle<int32_t>().clear(val);
    }
  }

  Node *createFC(Function *fn, std::unique_ptr<Module> &mod, std::string name,
                 Node *In, Constant *W, Constant *b) {
    // Optionally add nodes for quantization of FCs
    if (quantize) {
      TypeRef InQTy = mod->uniqueType(FCWeightType_, In->dims(0), 2.0, -128.0);
      auto *InQ = fn->createQuantize(name, In, InQTy);
      auto *FCQ = fn->createFullyConnected(name, InQ, W, b);
      TypeRef FCQTy = mod->uniqueType(dtype_, FCQ->dims(0));
      Node *FCO = fn->createDequantize(name, FCQ, FCQTy);
      return FCO;
    } else {
      return fn->createFullyConnected(name, In, W, b);
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
    configs.push_back(std::move(config));
    hostManager_ = glow::make_unique<runtime::HostManager>(std::move(configs));

    std::unique_ptr<Module> mod(new Module);
    auto fn = mod->createFunction("singleNode");

    // Input Placeholder ((maxSequenceLength*batchSize) x hiddenSize) (split)
    Placeholder *input = mod->createPlaceholder(
        dtype_, {maxSequenceLength_ * batchSize_, hiddenSize_}, "input", false);

    // for each context, add input bindings
    for (dim_t i = 0; i < asyncLaunchSize_; i++) {
      randomizeTensor(contexts_[i]->getPlaceholderBindings()->allocate(input),
                      mod->getPRNG());
    }

    // Weights/bias constants for QKV GEMM
    Tensor W_QKV_Tensor =
        (quantize) ? Tensor(FCWeightType_, {hiddenSize_, 3 * hiddenSize_},
                            FCWeightScale_, FCWeightOffset_)
                   : Tensor(FCWeightType_, {hiddenSize_, 3 * hiddenSize_});
    randomizeTensor(&W_QKV_Tensor, mod->getPRNG());
    Constant *W_QKV = mod->createConstant("W_QKV", W_QKV_Tensor);
    Tensor b_QKV_Tensor = (quantize) ? Tensor(FCBiasType_, {3 * hiddenSize_},
                                              FCWeightScale_, FCWeightOffset_)
                                     : Tensor(FCBiasType_, {3 * hiddenSize_});
    setTensor(&b_QKV_Tensor, 0.0f);
    Constant *b_QKV = mod->createConstant("b_QKV", b_QKV_Tensor);

    // Weights/bias constants for ZxWo FC
    Tensor W_ZWO_Tensor =
        (quantize) ? Tensor(FCWeightType_, {hiddenSize_, hiddenSize_},
                            FCWeightScale_, FCWeightOffset_)
                   : Tensor(FCWeightType_, {hiddenSize_, hiddenSize_});
    randomizeTensor(&W_ZWO_Tensor, mod->getPRNG());
    Constant *W_ZWO = mod->createConstant("W_ZWO", W_ZWO_Tensor);
    Tensor b_ZWO_Tensor = (quantize) ? Tensor(FCBiasType_, {hiddenSize_},
                                              FCWeightScale_, FCWeightOffset_)
                                     : Tensor(FCBiasType_, {hiddenSize_});
    randomizeTensor(&b_ZWO_Tensor, mod->getPRNG());
    Constant *b_ZWO = mod->createConstant("b_ZWO", b_ZWO_Tensor);

    // Constant scaling factor
    float sqrt_dk_flt =
        (float)(1.0 / std::sqrt(((double)hiddenSize_) / ((double)numHeads_)));

    // Softmax expected output. Not needed for inference
    Tensor expected_Tensor(ElemKind::Int64ITy,
                           {maxSequenceLength_ * batchSize_, 1});
    Constant *expected = mod->createConstant("expected", expected_Tensor);

    // Weights/bias constants for FC1
    Tensor W_FC1_Tensor =
        (quantize) ? Tensor(FCWeightType_, {hiddenSize_, 4 * hiddenSize_},
                            FCWeightScale_, FCWeightOffset_)
                   : Tensor(FCWeightType_, {hiddenSize_, 4 * hiddenSize_});
    randomizeTensor(&W_FC1_Tensor, mod->getPRNG());
    Constant *W_FC1 = mod->createConstant("W_FC1", W_FC1_Tensor);
    Tensor b_FC1_Tensor = (quantize) ? Tensor(FCBiasType_, {4 * hiddenSize_},
                                              FCWeightScale_, FCWeightOffset_)
                                     : Tensor(FCBiasType_, {4 * hiddenSize_});
    randomizeTensor(&b_FC1_Tensor, mod->getPRNG());
    Constant *b_FC1 = mod->createConstant("b_FC1", b_FC1_Tensor);

    // Weights/bias constants for FC2
    Tensor W_FC2_Tensor =
        (quantize) ? Tensor(FCWeightType_, {4 * hiddenSize_, hiddenSize_},
                            FCWeightScale_, FCWeightOffset_)
                   : Tensor(FCWeightType_, {4 * hiddenSize_, hiddenSize_});
    randomizeTensor(&W_FC2_Tensor, mod->getPRNG());
    Constant *W_FC2 = mod->createConstant("W_FC2", W_FC2_Tensor);
    Tensor b_FC2_Tensor = (quantize) ? Tensor(FCBiasType_, {hiddenSize_},
                                              FCWeightScale_, FCWeightOffset_)
                                     : Tensor(FCBiasType_, {hiddenSize_});
    randomizeTensor(&b_FC2_Tensor, mod->getPRNG());
    Constant *b_FC2 = mod->createConstant("b_FC2", b_FC2_Tensor);

    // batchSizePerCore is the number of sentences assigned to each
    // core (each data-parallel chunk)
    auto batchSizePerCore = getBatchSizePerCore(batchSize_, numCores_);

    // rowSizePerCore is the number of tokens assigned to each
    // core (each data-parallel chunk)
    dim_t numNonzeroCores = 0;
    std::vector<dim_t> rowSizePerCore;
    for (dim_t i = 0; i < batchSizePerCore.size(); i++) {
      if (batchSizePerCore[i] > 0) {
        rowSizePerCore.push_back(batchSizePerCore[i] * maxSequenceLength_);
        numNonzeroCores++;
      }
    }

    // Split the batch across cores in a data-parallel fashion
    std::vector<SliceNode *> inputs(numNonzeroCores);
    std::vector<SaveNode *> S(numNonzeroCores);

    // Split the input into cores of data-parallel fashion
    fn->createSplit("DPsplit", input, numNonzeroCores, 0, rowSizePerCore,
                    inputs);

    // For each core (sub-batch), create a network which does one layer
    for (int core = 0; core < int(numNonzeroCores); core++) {

      // Layer Norm 1 bias and scale
      Tensor LN1_scale_Tensor(dtype_, {hiddenSize_});
      randomizeTensor(&LN1_scale_Tensor, mod->getPRNG());
      Constant *LN1_scale = mod->createConstant("LN1_scale", LN1_scale_Tensor);
      Tensor LN1_bias_Tensor(dtype_, {hiddenSize_});
      randomizeTensor(&LN1_bias_Tensor, mod->getPRNG());
      Constant *LN1_bias = mod->createConstant("LN1_bias", LN1_bias_Tensor);

      // Layer Norm 2 bias and scale
      Tensor LN2_scale_Tensor(dtype_, {hiddenSize_});
      randomizeTensor(&LN2_scale_Tensor, mod->getPRNG());
      Constant *LN2_scale = mod->createConstant("LN2_scale", LN2_scale_Tensor);
      Tensor LN2_bias_Tensor(dtype_, {hiddenSize_});
      randomizeTensor(&LN2_bias_Tensor, mod->getPRNG());
      Constant *LN2_bias = mod->createConstant("LN2_bias", LN2_bias_Tensor);

      // QKV GEMM
      auto *QKV = createFC(fn, mod, strFormat("Gemm_QKV_core%d", core),
                           inputs[core], W_QKV, b_QKV);

      // Split into Q, K, V
      std::vector<SliceNode *> outputs(3);
      fn->createSplit(strFormat("split_core%d", core), QKV, 3, 1, {}, outputs);
      SliceNode *Q = outputs[0];
      SliceNode *K = outputs[1];
      SliceNode *V = outputs[2];

      // Multi-headed attention split
      std::vector<SliceNode *> Qsplits(numHeads_); // maxSequenceLength x 64
      std::vector<SliceNode *> Ksplits(numHeads_); // maxSequenceLength x 64
      std::vector<SliceNode *> Vsplits(numHeads_); // maxSequenceLength x 64
      std::vector<NodeValue> Zsplits(numHeads_);   // maxSequenceLength x 64
      fn->createSplit(strFormat("splitQ_core%d", core), Q, numHeads_, 1, {},
                      Qsplits);
      fn->createSplit(strFormat("splitK_core%d", core), K, numHeads_, 1, {},
                      Ksplits);
      fn->createSplit(strFormat("splitV_core%d", core), V, numHeads_, 1, {},
                      Vsplits);

      for (int i = 0; i < int(numHeads_); i++) {
        // Split the subbatch into individual sentences for the
        // batch matmul
        std::vector<SliceNode *> QBatchSplits(batchSizePerCore[core]);
        std::vector<SliceNode *> KBatchSplits(batchSizePerCore[core]);
        std::vector<SliceNode *> VBatchSplits(batchSizePerCore[core]);
        std::vector<NodeValue> ZBatchSplits(batchSizePerCore[core]);

        fn->createSplit(strFormat("splitBatchQ_core%d", core), Qsplits[i],
                        batchSizePerCore[core], 0, {}, QBatchSplits);
        fn->createSplit(strFormat("splitBatchK_core%d", core), Ksplits[i],
                        batchSizePerCore[core], 0, {}, KBatchSplits);
        fn->createSplit(strFormat("splitBatchV_core%d", core), Vsplits[i],
                        batchSizePerCore[core], 0, {}, VBatchSplits);

        // BatchMatMul
        for (int b = 0; b < int(batchSizePerCore[core]); b++) {

          auto *Kt =
              fn->createTranspose(strFormat("transpose_core%d_%d", core, i),
                                  KBatchSplits[b], {1, 0});
          // Tmp = Q * K^T
          auto *tmp =
              fn->createMatMul(strFormat("matmul_Q_KT_core%d_%d", core, i),
                               QBatchSplits[b], Kt->getResult());

          // Softmax_output = softmax(Tmp / sqrt(dk))
          auto *sqrt_dk_splat =
              fn->createSplat(strFormat("sqrt_dk_core%d_%d", core, i),
                              tmp->getResult().getType(), sqrt_dk_flt);
          auto *tmp_div = fn->createMul(strFormat("div_core%d_%d", core, i),
                                        tmp, sqrt_dk_splat);
          auto *softmax_output = fn->createSoftMax(
              strFormat("softmax_core%d_%d", core, i), tmp_div, expected);

          ZBatchSplits[b] =
              fn->createMatMul(strFormat("matmul_tmp_v_core%d_%d", core, i),
                               softmax_output, VBatchSplits[b]);
        }

        // Concatenate all the Z matrices for the whole subbatch
        Zsplits[i] =
            fn->createConcat(strFormat("concat_core%d", core), ZBatchSplits, 0);
      }

      // Concatenate all the Z matrices that we previously split on the hidden
      // dimension
      auto *Z = fn->createConcat(strFormat("concat_core%d", core), Zsplits, 1);

      // Z x W_o
      auto *ZWO = createFC(fn, mod, strFormat("Gemm_ZWO_core%d", core), Z,
                           W_ZWO, b_ZWO);

      // Layer norm
      auto *ZWO_norm = fn->createLayerNormalization(
          strFormat("LayerNorm1_core%d", core), ZWO->getNthResult(0).getType(),
          ZWO, LN1_scale, LN1_bias, 1e-5);

      // FC1
      auto *FC1 = createFC(fn, mod, strFormat("Gemm_FC1_core%d", core),
                           ZWO_norm, W_FC1, b_FC1);

      // Create gelu
      auto *FC1_gelu = fn->createGELU(strFormat("GELU_FC1_core%d", core), FC1);

      // FC2
      auto *FC2 = createFC(fn, mod, strFormat("Gemm_FC2_core%d", core),
                           FC1_gelu, W_FC2, b_FC2);

      // Layer norm
      auto *FC2_norm = fn->createLayerNormalization(
          strFormat("LayerNorm2_core%d", core), FC2->getNthResult(0).getType(),
          FC2, LN2_scale, LN2_bias, 1e-5);

      // Save result
      S[core] = fn->createSave(strFormat("save_core%d", core), FC2_norm);
      for (int i = 0; i < int(asyncLaunchSize_); i++) {
        contexts_[i]->getPlaceholderBindings()->allocate(
            S[core]->getPlaceholder());
      }
    } // For each core

    // Special case for batch-1, use model parallelism for FCs
    if ((batchSize_ == 1) && (numCores_ > 1)) {
      executeVerticalFCWeightsSplit(fn, numCores_, hiddenSize_);
    }

    CompilationContext ctx;
    EXIT_ON_ERR(hostManager_->addNetwork(std::move(mod), ctx));
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
    for (dim_t j = 0; j < asyncLaunchSize_; j++) {
      contexts_[j] = std::move(localContexts[j]);
    }
  }

  void teardown() override {}

  // Only counting GEMMs
  double gflops() const {
    double num_flops = 0.0;

    // QKV
    num_flops += 2.0 * maxSequenceLength_ * hiddenSize_ * 3 * hiddenSize_;

    // BMM
    num_flops += 2.0 * hiddenSize_ * maxSequenceLength_ * maxSequenceLength_;
    num_flops += 2.0 * hiddenSize_ * maxSequenceLength_ * maxSequenceLength_;

    // ZWO
    num_flops += 2.0 * maxSequenceLength_ * hiddenSize_ * hiddenSize_;

    // FC1
    num_flops += 2.0 * maxSequenceLength_ * hiddenSize_ * 4 * hiddenSize_;

    // FC2
    num_flops += 2.0 * maxSequenceLength_ * hiddenSize_ * 4 * hiddenSize_;

    return batchSize_ * num_flops / 1e9;
  }
};

int main(int argc, char *argv[]) {
  printf(
      "Usage: BERTLayerBench maxSequenceLength batchSize hiddenSize numHeads "
      "numCores "
      "numReps numAsyncLaunches backendStr dtypeStr useInt8FCs\n");
  printf("Standard Glow command-line options may be passed via the GLOW_OPTS "
         "environment variable\n");
  benchParseGlowOpts(argc, argv);
  assert(argc == 11);
  size_t maxSequenceLength = atoi(argv[1]);
  size_t batchSize = atoi(argv[2]);
  size_t hiddenSize = atoi(argv[3]);
  size_t numHeads = atoi(argv[4]);
  size_t numCores = atoi(argv[5]);
  size_t numReps = atoi(argv[6]);
  size_t numAsyncLaunches = atoi(argv[7]);
  const char *backendStr = argv[8];
  const char *dtypeStr = argv[9];
  const char *useInt8FCs = argv[10];
  assert(numReps > 0);

  BERTProxyLayerBench b(maxSequenceLength, batchSize, hiddenSize, numHeads,
                        numCores, numAsyncLaunches, backendStr, dtypeStr,
                        useInt8FCs);

  auto times = bench(&b, numReps);
  printf("_,benchName,maxSequenceLength,batchSize,hiddenSize,numHeads,numCores,"
         "numReps,"
         "numAsyncLaunches,"
         "backendStr,dtypeStr,useInt8FCs,averageTime,averageGFLOP\n");
  for (auto t : times) {
    printf("BenchResult,BERTProxyLayerBench,SW,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%s,%"
           "s,%s,%f,%f\n",
           maxSequenceLength, batchSize, hiddenSize, numHeads, numCores,
           numReps, numAsyncLaunches, backendStr, dtypeStr, useInt8FCs,
           t / numAsyncLaunches, b.gflops() * numAsyncLaunches / t);
  }
  double min = *(std::min_element(times.begin(), times.end()));
  size_t midElt = times.size() / 2;
  std::nth_element(times.begin(), times.begin() + midElt, times.end());
  double median = times[midElt];
  double median_runtime = median / ((double)numAsyncLaunches);
  double min_runtime = min / ((double)numAsyncLaunches);
  printf("_,benchName,maxSequenceLength,batchSize,hiddenSize,numHeads,numCores,"
         "numReps,"
         "numAsyncLaunches,"
         "backendStr,dtypeStr,useInt8FCs,medianRuntime,minRuntime,medianGFLOPS,"
         "minGFLOPS\n");
  printf("Total gflop: %f\n", b.gflops());
  printf("BenchSummary,BERTProxyLayerBench,SW,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%s,%"
         "s,%s,"
         "%f,%f,%f,%"
         "f\n",
         maxSequenceLength, batchSize, hiddenSize, numHeads, numCores, numReps,
         numAsyncLaunches, backendStr, dtypeStr, useInt8FCs, median_runtime,
         min_runtime, b.gflops() / median_runtime, b.gflops() / min_runtime);
}
