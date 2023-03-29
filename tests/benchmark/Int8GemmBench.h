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
#pragma once

#include "Bench.h"
#include "glow/Optimizer/GraphOptimizer/GraphOptimizer.h"

#include "llvm/Support/CommandLine.h"

#include "tests/unittests/BackendTestUtils.h"

extern llvm::cl::opt<bool> checkCorrectness;

namespace glow {

struct Int8GemmParam {
  dim_t m_;
  dim_t n_;
  dim_t k_;
  dim_t numLayers_;
  dim_t numReps_;
  dim_t numAsyncLaunches_;
  dim_t numSplits_;
  std::string backendStr_;
  std::string devId_;
};

class Int8GemmBench : public Benchmark {
  Int8GemmParam param_;
  ExecutionContext context_;
  PlaceholderBindings &bindings_;
  std::unique_ptr<runtime::HostManager> hostManager_;

  // Refernce bindings and network:
  ExecutionContext refContext_;
  PlaceholderBindings &refBindings_;
  std::unique_ptr<runtime::HostManager> refHostManager_;

public:
  explicit Int8GemmBench(Int8GemmParam param_)
      : param_(param_), bindings_(*context_.getPlaceholderBindings()),
        refBindings_(*refContext_.getPlaceholderBindings()) {}

  virtual void addInt8GemmNode(std::unique_ptr<Module> &mod, Function *fn,
                               Int8GemmParam param, bool isRef) {

    PlaceholderBindings &bindings = isRef ? refBindings_ : bindings_;
    auto *input = mod->createPlaceholder(ElemKind::Float16Ty,
                                         {param.m_, param.k_}, "input", false);
    bindings.allocate(input)->getHandle<float16>().randomize(-5.f, 5.f,
                                                             mod->getPRNG());
    auto *output = mod->createPlaceholder(
        ElemKind::Float16Ty, {param.m_, param.n_}, "output", false);
    auto *q_input = fn->createQuantize(
        "int8_quantize", input,
        mod->uniqueType(ElemKind::Int8QTy, {param.m_, param.k_}, 1.0, 0));
    Node *cur = q_input;

    Placeholder *ones;
    if (param.k_ > param.n_) {
      ones = mod->createPlaceholder(ElemKind::Int8QTy,
                                    {param.m_ * (param.k_ - param.n_)}, 1.0, 0,
                                    "ones", false);
      bindings.allocate(ones)->getHandle<int8_t>().clear(1);
    }

    Placeholder *weights;
    Placeholder *bias;

    // Create multiple layers of FC nodes
    for (size_t layer = 0; layer < param.numLayers_; layer++) {
      weights =
          mod->createPlaceholder(ElemKind::Int8QTy, {param.k_, param.n_}, 1.0,
                                 0, "weights" + std::to_string(layer), false);
      bias = mod->createPlaceholder(ElemKind::Int32QTy, {param.n_}, 1.0, 0,
                                    "bias" + std::to_string(layer), false);

      bindings.allocate(weights)->getHandle<int8_t>().randomize(-128, 127,
                                                                mod->getPRNG());
      bindings.allocate(bias)->getHandle<int32_t>().randomize(-128, 127,
                                                              mod->getPRNG());

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
    auto *dequantized_fc = fn->createDequantize(
        "int8_dequantize", cur,
        mod->uniqueType(ElemKind::Float16Ty, {param.m_, param.n_}));
    cur = dequantized_fc;

    fn->createSave("save1", cur, output);
    bindings.allocate(output);
    ::glow::convertPlaceholdersToConstants(fn, bindings, {input, output});
  }

  void parallelize(Function *fn) {
    // Model parallelize FCs
    llvm::DenseMap<Node *, size_t> numOfChunks;
    llvm::DenseMap<Node *, ParallelTransformKind> parOpts;
    for (auto &N : fn->getNodes()) {
      if (N.getKind() == Kinded::Kind::FullyConnectedNodeKind) {
        numOfChunks[&N] = param_.numSplits_;
        parOpts[&N] = ParallelTransformKind::Model;
      }
    }

    // Parallelize Quantize/Dequantize
    for (auto &N : fn->getNodes()) {
      if (N.getKind() == Kinded::Kind::QuantizeNodeKind ||
          N.getKind() == Kinded::Kind::DequantizeNodeKind) {
        numOfChunks[&N] = param_.numSplits_;
        parOpts[&N] = ParallelTransformKind::Data;
      }
    }
    EXIT_ON_ERR(parallelizeOps(fn, numOfChunks, parOpts, 1));
  }

  void setup_internal(bool isRef) {
    // Setup host manager
    std::string backendStr = isRef ? "Interpreter" : param_.backendStr_.c_str();
    std::vector<std::unique_ptr<runtime::DeviceConfig>> configs;
    auto config = glow::make_unique<runtime::DeviceConfig>(backendStr);
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

    addInt8GemmNode(mod, fn, param_, isRef);
    parallelize(fn);
    optimize(fn, CompilationMode::Infer);

    CompilationContext ctx;
    ctx.dumpFinalGraph = true;
    if (isRef) {
      EXIT_ON_ERR(refHostManager_->addNetwork(std::move(mod), ctx));
    } else {
      EXIT_ON_ERR(hostManager_->addNetwork(std::move(mod), ctx));
    }
  }

  void setup() override {
    if (checkCorrectness) {
      setup_internal(/* isRef */ true);
    }
    setup_internal(/* isRef */ false);
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

  void run() override {
    dispatchInference("singleNode", hostManager_.get(), context_,
                      param_.numAsyncLaunches_,
                      /*useNewExecutionContext*/ true);
    if (checkCorrectness) {
      checkOutput();
    }
  }

  void teardown() override {}

  double gops() const {
    return 2.0 * param_.m_ * param_.n_ * param_.k_ * param_.numLayers_ / 1e9;
  }
};
} // namespace glow
