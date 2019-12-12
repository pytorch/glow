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
 * This class implements an SLS microbenchmark. There are a number of
 * parallel FusedRowwiseQuantizedSparseLengthsWeightedSum or
 * FusedRowwiseQuantizedSparseLengthsSum nodes which are created.
 *
 * Microbenchmarks are generally useful for understanding performance
 * through targeted experiementation and are not representative of
 * end-to-end workloads.
 */
class SLSBench : public Benchmark {
  /// Dimensions expressed in libjit's format.
  dim_t batchSize_;
  dim_t numIndicesPerBatch_;
  dim_t numIndicesPerBatchPad_;
  dim_t numTableEntries_;
  dim_t numElementsPerRow_;
  std::unique_ptr<runtime::HostManager> hostManager_;
  std::vector<std::unique_ptr<ExecutionContext>> contexts_;
  std::vector<std::vector<Tensor>> indicesReal_;
  std::vector<std::vector<Tensor>> weightsReal_;
  size_t asyncLaunchSize_;
  size_t numSLSNodes_;
  const char *weightedStr_;
  const char *sortedStr_;
  const char *backendStr_;
  ElemKind dtype_;
  ElemKind fusedDtype_;
  size_t elementSize_;
  const char *devId_;

public:
  SLSBench(dim_t batchSize_, dim_t numIndicesPerBatch_,
           dim_t numIndicesPerBatchPad_, dim_t numTableEntries_,
           dim_t numElementsPerRow_, size_t asyncLaunchSize_,
           size_t numSLSNodes_, const char *weightedStr_,
           const char *sortedStr_, const char *backendStr_,
           const char *dtypeStr_, const char *devId_ = nullptr)
      : batchSize_(batchSize_), numIndicesPerBatch_(numIndicesPerBatch_),
        numIndicesPerBatchPad_(numIndicesPerBatchPad_),
        numTableEntries_(numTableEntries_),
        numElementsPerRow_(numElementsPerRow_),
        asyncLaunchSize_(asyncLaunchSize_), numSLSNodes_(numSLSNodes_),
        weightedStr_(weightedStr_), sortedStr_(sortedStr_),
        backendStr_(backendStr_), devId_(devId_) {
    elementSize_ = 2;
    if (std::string(dtypeStr_) == "Float16") {
      dtype_ = ElemKind::Float16Ty;
      fusedDtype_ = ElemKind::UInt8FusedFP16QTy;
      elementSize_ = 2;
    } else if (std::string(dtypeStr_) == "Float32") {
      dtype_ = ElemKind::FloatTy;
      fusedDtype_ = ElemKind::UInt8FusedQTy;
      elementSize_ = 4;
    } else {
      llvm_unreachable("Unhandled ElemKind.");
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
    if (devId_ != nullptr) {
      config->parameters["DeviceID"] = devId_;
    }
    configs.push_back(std::move(config));
    hostManager_ = glow::make_unique<runtime::HostManager>(std::move(configs));

    std::unique_ptr<Module> mod(new Module);
    auto fn = mod->createFunction("singleNode");

    std::vector<Placeholder *> weights(numSLSNodes_);
    std::vector<Placeholder *> indices(numSLSNodes_);
    std::vector<Placeholder *> lengths(numSLSNodes_);
    std::vector<SaveNode *> S(numSLSNodes_);
    indicesReal_.resize(asyncLaunchSize_);
    weightsReal_.resize(asyncLaunchSize_);
    for (dim_t i = 0; i < asyncLaunchSize_; i++) {
      for (dim_t slsNodeId = 0; slsNodeId < numSLSNodes_; slsNodeId++) {
        Tensor indicesReal(ElemKind::Int64ITy,
                           {numIndicesPerBatch_ * batchSize_});
        indicesReal.getHandle<int64_t>().randomize(0, numTableEntries_ - 1,
                                                   mod->getPRNG());
        // Sort each segment
        if (std::string(sortedStr_) == "Sorted") {
          int64_t *indicesRealPtr = (int64_t *)indicesReal.getUnsafePtr();
          for (dim_t b = 0; b < batchSize_; b++) {
            std::sort(indicesRealPtr + b * numIndicesPerBatch_,
                      indicesRealPtr + (b + 1) * numIndicesPerBatch_);
          }
        }
        indicesReal_[i].push_back(std::move(indicesReal));

        if (dtype_ == ElemKind::FloatTy) {
          Tensor weightsReal(ElemKind::FloatTy,
                             {numIndicesPerBatch_ * batchSize_});
          weightsReal.getHandle<float>().clear(1.0f);
          weightsReal_[i].push_back(std::move(weightsReal));
        } else if (dtype_ == ElemKind::Float16Ty) {
          Tensor weightsReal(ElemKind::Float16Ty,
                             {numIndicesPerBatch_ * batchSize_});
          weightsReal.getHandle<float16_t>().clear(1.0f);
          weightsReal_[i].push_back(std::move(weightsReal));
        }
      }
    }

    // Create multiple SLS nodes
    for (dim_t slsNodeId = 0; slsNodeId < numSLSNodes_; slsNodeId++) {
      Tensor data(ElemKind::FloatTy, {numTableEntries_, numElementsPerRow_});
      data.getHandle().clear(1.0f);

      weights[slsNodeId] =
          mod->createPlaceholder(dtype_, {numIndicesPerBatchPad_ * batchSize_},
                                 "weights_" + std::to_string(slsNodeId), false);

      indices[slsNodeId] = mod->createPlaceholder(
          ElemKind::Int64ITy, {(dim_t)numIndicesPerBatchPad_ * batchSize_},
          "indices_" + std::to_string(slsNodeId),
          /* isTrainable */ false);

      lengths[slsNodeId] =
          mod->createPlaceholder(ElemKind::Int32ITy, {batchSize_}, "lengths",
                                 /* isTrainable */ false);

      // for each context, add input bindings
      for (dim_t i = 0; i < asyncLaunchSize_; i++) {
        Tensor indicesPartial(indicesReal_[i][slsNodeId].getUnsafePtr(),
                              indices[slsNodeId]->getType(),
                              indicesReal_[i][slsNodeId].getSizeInBytes());

        contexts_[i]->getPlaceholderBindings()->insert(
            indices[slsNodeId], std::move(indicesPartial));

        contexts_[i]
            ->getPlaceholderBindings()
            ->allocate(lengths[slsNodeId])
            ->getHandle<int32_t>()
            .clear(numIndicesPerBatch_);

        Tensor weightsPartial(weightsReal_[i][slsNodeId].getUnsafePtr(),
                              weights[slsNodeId]->getType(),
                              weightsReal_[i][slsNodeId].getSizeInBytes());
        contexts_[i]->getPlaceholderBindings()->insert(
            weights[slsNodeId], std::move(weightsPartial));
      }

      Node *R;
      if (std::string(weightedStr_) == "Unweighted") {
        R = fn->createFusedRowwiseQuantizedSparseLengthsSum(
            "RQSLWS_" + std::to_string(slsNodeId), data, indices[slsNodeId],
            lengths[slsNodeId], fusedDtype_, false);
      } else {
        R = fn->createFusedRowwiseQuantizedSparseLengthsWeightedSum(
            "RQSLWS_" + std::to_string(slsNodeId), data, weights[slsNodeId],
            indices[slsNodeId], lengths[slsNodeId], fusedDtype_, false);
      }

      S[slsNodeId] = fn->createSave("save_" + std::to_string(slsNodeId), R);

      // for each context, add output bindings
      for (size_t i = 0; i < asyncLaunchSize_; i++) {
        contexts_[i]->getPlaceholderBindings()->allocate(
            S[slsNodeId]->getPlaceholder());
      }
    } // For each slsNodeId

    CompilationContext ctx;
    EXIT_ON_ERR(hostManager_->addNetwork(std::move(mod), ctx));
  }

  void run() override {
    std::vector<std::unique_ptr<ExecutionContext>> localContexts(
        asyncLaunchSize_);
    std::vector<std::promise<void>> promises(asyncLaunchSize_);
    std::vector<std::future<void>> futures;

    // Launch a number of independent requests
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
    for (size_t j = 0; j < asyncLaunchSize_; j++) {
      contexts_[j] = std::move(localContexts[j]);
    }
  }

  void teardown() override {}

  // Each row has numElementsPerRow bytes per row, plus scale and offset
  double gbytes() const {

    // Embedding data
    double input_gbytes = (numSLSNodes_ * batchSize_ * numIndicesPerBatch_ *
                           (numElementsPerRow_ + 2 * elementSize_)) /
                          1e9;
    // + indices
    input_gbytes +=
        (numSLSNodes_ * batchSize_ * numIndicesPerBatch_ * sizeof(int32_t)) /
        1e9;

    // + weights
    if (std::string(weightedStr_) != "Unweighted") {
      input_gbytes +=
          (numSLSNodes_ * batchSize_ * numIndicesPerBatch_ * elementSize_) /
          1e9;
    }

    // + lengths
    input_gbytes += (numSLSNodes_ * batchSize_ * sizeof(int32_t)) / 1e9;

    double output_gbytes =
        (numSLSNodes_ * batchSize_ * (numElementsPerRow_ * elementSize_)) / 1e9;

    return input_gbytes + output_gbytes;
  }
};

int main(int argc, char *argv[]) {

  printf("SLS Microbenchmark\n");
  printf("Usage: SLSBench batchSize(Int) numIndicesPerBatch(Int) "
         "numIndicesPerBatchPad(Int) numTableEntries(Int) "
         "numElementsPerRow(int) numReps(Int) "
         "numAsyncLaunches(Int) numSLSNodes(Int) "
         "weightedStr(\"Weighted\"|\"Unweighted\") "
         "sortedStr(\"Sorted\"|\"Unsorted\") backendStr(String) "
         "dtypeStr(\"Float16\"|\"Float32\") dev_id(Int)\n");
  printf("\n");

  assert(argc == 13 || argc == 14);
  size_t batchSize = atoi(argv[1]);
  size_t numIndicesPerBatch = atoi(argv[2]);
  size_t numIndicesPerBatchPad = atoi(argv[3]);
  size_t numTableEntries = atoi(argv[4]);
  size_t numElementsPerRow = atoi(argv[5]);
  size_t numReps = atoi(argv[6]);
  size_t numAsyncLaunches = atoi(argv[7]);
  size_t numSLSNodes = atoi(argv[8]);
  const char *weightedStr = argv[9];
  const char *sortedStr = argv[10];
  const char *backendStr = argv[11];
  const char *dtypeStr = argv[12];
  char *dev_id = nullptr;

  if (argc > 13) {
    dev_id = argv[13];
    printf("Setting backend device: \"%s\"\n", dev_id);
  }
  assert(numReps > 0);

  SLSBench b(batchSize, numIndicesPerBatch, numIndicesPerBatchPad,
             numTableEntries, numElementsPerRow, numAsyncLaunches, numSLSNodes,
             weightedStr, sortedStr, backendStr, dtypeStr, dev_id);
  auto times = bench(&b, numReps);
  printf("_,benchName,_,batchSize,numIndicesPerBatch,numIndicesPerBatchPad,"
         "numTableEntries,"
         "numElementsPerRow,numReps,numAsyncLaunches,numSLSNodes,weightedStr,"
         "backendStr,dtypeStr,runtime,gbytesPerSec\n");
  for (auto t : times) {
    printf(
        "BenchResult,SLSBench,SW,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%s,%s,%s,%s,%"
        "f,%f\n",
        batchSize, numIndicesPerBatch, numIndicesPerBatchPad, numTableEntries,
        numElementsPerRow, numReps, numAsyncLaunches, numSLSNodes, weightedStr,
        sortedStr, backendStr, dtypeStr, t / numAsyncLaunches,
        b.gbytes() * numAsyncLaunches / t);
  }
  double min = *(std::min_element(times.begin(), times.end()));
  size_t midElt = times.size() / 2;
  std::nth_element(times.begin(), times.begin() + midElt, times.end());
  double median = times[midElt];
  double median_runtime = median / ((double)numAsyncLaunches);
  double min_runtime = min / ((double)numAsyncLaunches);
  printf("_,benchName,_,batchSize,numIndicesPerBatch,numIndicesPerBatchPad,"
         "numTableEntries,"
         "numElementsPerRow,numReps,numAsyncLaunches,numSLSNodes,weightedStr,"
         "sortedStr,backendStr,dtypeStr,medianRuntime,minRuntime,"
         "medianGbytesPerSec,"
         "maxGbytesPerSec\n");
  printf(
      "BenchSummary,SLSBench,SW,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%s,%s,%s,%s,%f,"
      "%f,%f,%"
      "f\n",
      batchSize, numIndicesPerBatch, numIndicesPerBatchPad, numTableEntries,
      numElementsPerRow, numReps, numAsyncLaunches, numSLSNodes, weightedStr,
      sortedStr, backendStr, dtypeStr, median_runtime, min_runtime,
      b.gbytes() / median_runtime, b.gbytes() / min_runtime);
}
