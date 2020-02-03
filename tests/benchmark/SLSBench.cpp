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
#include <fstream>
#include <future>
#include <random>
#include <string>

#include "Bench.h"

#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Optimizer/GraphOptimizer/GraphOptimizer.h"

using namespace glow;

/*
 * This class implements an SLS microbenchmark. There are a number of
 * parallel FusedRowwiseQuantizedSparseLengthsWeightedSum,
 * FusedRowwiseQuantizedSparseLengthsSum, SparseLengthsWeightedSum, or
 * SparseLengthsSum nodes which are created.
 *
 * Microbenchmarks are generally useful for understanding performance
 * through targeted experiementation and are not representative of
 * end-to-end workloads.
 */

enum SLSKind {
  NONQUANTIZED_UNWEIGHTED,
  NONQUANTIZED_WEIGHTED,
  QUANTIZED_UNWEIGHTED,
  QUANTIZED_WEIGHTED
};

struct SLSParam {
  dim_t batchSize;
  dim_t numReps;
  dim_t numAsyncLaunches;
  std::string backendStr;
  std::string devId;
  dim_t numIndicesPerBatch;
  dim_t numIndicesPerBatchPad;
  dim_t numTableEntries;
  dim_t numElementsPerRow;
  dim_t numSLSNodes;
  SLSKind slsKind;
  bool isSorted;
  bool addClip;
  bool useFP16Accumulation;
  ElemKind fusedDtype;
  ElemKind dtype;
};

std::string getSLSDescription(SLSParam param) {
  std::string SLSStr = (param.slsKind == NONQUANTIZED_UNWEIGHTED)
                           ? std::string("SLS")
                           : (param.slsKind == NONQUANTIZED_WEIGHTED)
                                 ? std::string("SLWS")
                                 : (param.slsKind == QUANTIZED_UNWEIGHTED)
                                       ? std::string("RWQLSS")
                                       : std::string("RWQLSWS");

  return strFormat(
      "%s_%zu_%zu_%zu_%zu", SLSStr.c_str(), (size_t)param.numIndicesPerBatch,
      (size_t)param.numIndicesPerBatchPad, (size_t)param.numTableEntries,
      (size_t)param.numElementsPerRow);
}

class SLSBench : public Benchmark {
  std::unique_ptr<runtime::HostManager> hostManager_;
  std::vector<std::unique_ptr<ExecutionContext>> contexts_;
  std::vector<std::vector<Tensor>> indicesReal_;
  std::vector<std::vector<Tensor>> weightsReal_;
  dim_t batchSize_;
  dim_t asyncLaunchSize_;
  std::string backendStr_;
  std::vector<SLSParam> params_;
  std::string devId_;
  float tableValue;

public:
  SLSBench(dim_t batchSize_, dim_t asyncLaunchSize_, std::string backendStr_,
           std::vector<SLSParam> params_, std::string devId_ = std::string(""))
      : batchSize_(batchSize_), asyncLaunchSize_(asyncLaunchSize_),
        backendStr_(backendStr_), params_(params_), devId_(devId_),
        tableValue(1.0f) {}

  double countSLSGbytes(SLSParam param) const {

    dim_t elementSize = 2;
    if (param.dtype == ElemKind::FloatTy) {
      elementSize = 4;
    }

    // Embedding data
    double input_gbytes = 0.0;
    if ((param.slsKind == NONQUANTIZED_WEIGHTED) ||
        (param.slsKind == NONQUANTIZED_UNWEIGHTED)) {
      input_gbytes +=
          (param.numSLSNodes * batchSize_ * param.numIndicesPerBatch *
           (param.numElementsPerRow * elementSize)) /
          1e9;
    } else { // Quantized
      if (param.fusedDtype == ElemKind::UInt8FusedFP16QTy) {
        input_gbytes +=
            (param.numSLSNodes * batchSize_ * param.numIndicesPerBatch *
             (param.numElementsPerRow + 2 * elementSize)) /
            1e9;
      } else { // Int4
        input_gbytes +=
            (param.numSLSNodes * batchSize_ * param.numIndicesPerBatch *
             ((param.numElementsPerRow + 1) / 2 + 2 * elementSize)) /
            1e9;
      }
    }

    // + indices
    input_gbytes += (param.numSLSNodes * batchSize_ * param.numIndicesPerBatch *
                     sizeof(int32_t)) /
                    1e9;

    // + weights
    if ((param.slsKind == QUANTIZED_WEIGHTED) ||
        (param.slsKind == NONQUANTIZED_WEIGHTED)) {
      input_gbytes += (param.numSLSNodes * batchSize_ *
                       param.numIndicesPerBatch * elementSize) /
                      1e9;
    }

    // + lengths
    input_gbytes += (param.numSLSNodes * batchSize_ * sizeof(int32_t)) / 1e9;

    double output_gbytes = (param.numSLSNodes * batchSize_ *
                            (param.numElementsPerRow * elementSize)) /
                           1e9;

    return input_gbytes + output_gbytes;
  }

  void addSLSNode(std::unique_ptr<Module> &mod, Function *fn, SLSParam param) {

    // Create and initialize data tensor
    Tensor data(ElemKind::FloatTy,
                {param.numTableEntries, param.numElementsPerRow});
    data.getHandle().clear(tableValue);
    tableValue += 1.0f;

    // Constant needed for Non-quantized case
    Constant *dataConstant = nullptr;
    if ((param.slsKind == NONQUANTIZED_WEIGHTED) ||
        (param.slsKind == NONQUANTIZED_UNWEIGHTED)) {
      Tensor dataConstantTensor(
          param.dtype, {param.numTableEntries, param.numElementsPerRow});
      if (param.dtype == ElemKind::FloatTy) {
        dataConstantTensor.getHandle<float>().clear(1.0f);
      } else {
        dataConstantTensor.getHandle<float16_t>().clear(1.0f);
      }
      dataConstant = mod->createConstant("SLSData", dataConstantTensor);
    }

    // Create placeholders for weights, indices and lengths
    auto *weights = mod->createPlaceholder(
        param.dtype, {param.numIndicesPerBatchPad * batchSize_}, "weights",
        false);

    auto *indices = mod->createPlaceholder(
        ElemKind::Int64ITy, {param.numIndicesPerBatchPad * batchSize_},
        "indices",
        /* isTrainable */ false);

    auto *lengths =
        mod->createPlaceholder(ElemKind::Int32ITy, {batchSize_}, "lengths",
                               /* isTrainable */ false);

    for (dim_t i = 0; i < asyncLaunchSize_; i++) {

      // Create and sort indices
      Tensor indicesReal(ElemKind::Int64ITy,
                         {param.numIndicesPerBatch * batchSize_});
      indicesReal.getHandle<int64_t>().randomize(0, param.numTableEntries - 1,
                                                 mod->getPRNG());
      // Sort each segment
      if (param.isSorted) {
        int64_t *indicesRealPtr = (int64_t *)indicesReal.getUnsafePtr();
        for (dim_t b = 0; b < batchSize_; b++) {
          std::sort(indicesRealPtr + b * param.numIndicesPerBatch,
                    indicesRealPtr + (b + 1) * param.numIndicesPerBatch);
        }
      }
      indicesReal_[i].push_back(std::move(indicesReal));

      // Create weights
      if (param.dtype == ElemKind::FloatTy) {
        Tensor weightsReal(ElemKind::FloatTy,
                           {param.numIndicesPerBatch * batchSize_});
        weightsReal.getHandle<float>().clear(1.0f);
        weightsReal_[i].push_back(std::move(weightsReal));
      } else if (param.dtype == ElemKind::Float16Ty) {
        Tensor weightsReal(ElemKind::Float16Ty,
                           {param.numIndicesPerBatch * batchSize_});
        weightsReal.getHandle<float16_t>().clear(1.0f);
        weightsReal_[i].push_back(std::move(weightsReal));
      }

      Tensor indicesPartial(indicesReal_[i].back().getUnsafePtr(),
                            indices->getType(),
                            indicesReal_[i].back().getSizeInBytes());

      contexts_[i]->getPlaceholderBindings()->insert(indices,
                                                     std::move(indicesPartial));

      contexts_[i]
          ->getPlaceholderBindings()
          ->allocate(lengths)
          ->getHandle<int32_t>()
          .clear(param.numIndicesPerBatch);

      Tensor weightsPartial(weightsReal_[i].back().getUnsafePtr(),
                            weights->getType(),
                            weightsReal_[i].back().getSizeInBytes());
      contexts_[i]->getPlaceholderBindings()->insert(weights,
                                                     std::move(weightsPartial));
    } // i

    // Create SLS node, optional clip node, and save node
    Node *R = nullptr;
    if (param.slsKind == QUANTIZED_UNWEIGHTED) {
      R = fn->createFusedRowwiseQuantizedSparseLengthsSum(
          getSLSDescription(param), data, indices, lengths, param.fusedDtype,
          param.useFP16Accumulation);
    } else if (param.slsKind == QUANTIZED_WEIGHTED) {
      R = fn->createFusedRowwiseQuantizedSparseLengthsWeightedSum(
          getSLSDescription(param), data, weights, indices, lengths,
          param.fusedDtype, param.useFP16Accumulation);
    } else if (param.slsKind == NONQUANTIZED_WEIGHTED) {
      R = fn->createSparseLengthsWeightedSum(
          getSLSDescription(param), dataConstant, weights, indices, lengths);
    } else { // NonquantizedUnweighted
      R = fn->createSparseLengthsSum(getSLSDescription(param), dataConstant,
                                     indices, lengths);
    }
    SaveNode *S = nullptr;
    if (param.addClip) {
      auto *clp = fn->createClip("clip", R, -65504.0f, 65504.0f);
      S = fn->createSave("save", clp);
    } else {
      S = fn->createSave("save", R);
    }

    // for each context, add output bindings
    for (dim_t i = 0; i < asyncLaunchSize_; i++) {
      contexts_[i]->getPlaceholderBindings()->allocate(S->getPlaceholder());
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
    auto config = glow::make_unique<runtime::DeviceConfig>(backendStr_.c_str());
    if (devId_ != "") {
      config->parameters["DeviceID"] = devId_.c_str();
    }
    configs.push_back(std::move(config));
    hostManager_ = glow::make_unique<runtime::HostManager>(std::move(configs));

    // Create a function
    std::unique_ptr<Module> mod(new Module);
    auto fn = mod->createFunction("singleNode");

    // Keep tensors around so they aren't deleted
    indicesReal_.resize(asyncLaunchSize_);
    weightsReal_.resize(asyncLaunchSize_);

    // Add SLS nodes
    for (auto &param : params_) {
      for (int i = 0; i < param.numSLSNodes; i++) {
        addSLSNode(mod, fn, param);
      }
    }

    fn->dumpDAG("slsbench.dot");
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
    for (dim_t j = 0; j < asyncLaunchSize_; j++) {
      contexts_[j] = std::move(localContexts[j]);
    }
  }

  void teardown() override {}

  double gbytes() const {
    double total = 0.0;
    for (auto &param : params_) {
      total += countSLSGbytes(param);
    }
    return total;
  }
};

// Indices of arguments
#define ROWWISE_QUANT 14
#define ACCUM_TYPE 15
#define DEVICE_ID 16

SLSParam parseArgs(int argc, char *argv[]) {
  SLSParam param;
  param.batchSize = atoi(argv[1]);
  param.numIndicesPerBatch = atoi(argv[2]);
  param.numIndicesPerBatchPad = atoi(argv[3]);
  param.numTableEntries = atoi(argv[4]);
  param.numElementsPerRow = atoi(argv[5]);
  param.numReps = atoi(argv[6]);
  param.numAsyncLaunches = atoi(argv[7]);
  param.numSLSNodes = atoi(argv[8]);
  printf("batchSize %zu\n", (size_t)param.batchSize);
  printf("numIndicesPerBatch %zu\n", (size_t)param.numIndicesPerBatch);
  printf("numIndicesPerBatchPad %zu\n", (size_t)param.numIndicesPerBatchPad);
  printf("numTableEntries %zu\n", (size_t)param.numTableEntries);
  printf("numElementsPerRow %zu\n", (size_t)param.numElementsPerRow);
  printf("numReps %zu\n", (size_t)param.numReps);
  printf("numAsyncLaunches %zu\n", (size_t)param.numAsyncLaunches);
  printf("numSLSNodes %zu\n", (size_t)param.numSLSNodes);
  printf("slsKind %s\n", argv[9]);
  if (std::string(argv[9]) == "NonquantizedUnweighted") {
    param.slsKind = NONQUANTIZED_UNWEIGHTED;
  } else if (std::string(argv[9]) == "NonquantizedWeighted") {
    param.slsKind = NONQUANTIZED_WEIGHTED;
  } else if (std::string(argv[9]) == "QuantizedUnweighted") {
    param.slsKind = QUANTIZED_UNWEIGHTED;
  } else if (std::string(argv[9]) == "QuantizedWeighted") {
    param.slsKind = QUANTIZED_WEIGHTED;
  } else {
    llvm_unreachable("Invalid SLS Kind");
  }
  printf("sortedStr %s\n", argv[10]);
  if (std::string(argv[10]) == "Sorted") {
    param.isSorted = true;
  } else if (std::string(argv[10]) == "Unsorted") {
    param.isSorted = false;
  } else {
    llvm_unreachable("Invalid sortedStr");
  }
  printf("backendStr %s\n", argv[11]);
  param.backendStr = std::string(argv[11]);
  printf("dtypeStr %s\n", argv[12]);
  if (std::string(argv[12]) == "Float16") {
    param.dtype = ElemKind::Float16Ty;
  } else if (std::string(argv[12]) == "Float32") {
    param.dtype = ElemKind::FloatTy;
  } else {
    llvm_unreachable("Invalid dtype");
  }
  printf("addClipStr %s\n", argv[13]);
  if (std::string(argv[13]) == "True") {
    param.addClip = true;
  } else if (std::string(argv[13]) == "False") {
    param.addClip = false;
  } else {
    llvm_unreachable("Invalid addClipStr");
  }
  if (argc > ROWWISE_QUANT) {
    printf("fusedDtype%s\n", argv[ROWWISE_QUANT]);
    if (std::string(argv[ROWWISE_QUANT]) == "Int8") {
      param.fusedDtype = ElemKind::UInt8FusedFP16QTy;
    } else if (std::string(argv[ROWWISE_QUANT]) == "Int4") {
      param.fusedDtype = ElemKind::UInt4FusedFP16QTy;
    } else {
      llvm_unreachable("Invalid Quantization datatype");
    }
  } else {
    param.fusedDtype = ElemKind::UInt8FusedFP16QTy;
  }
  if (argc > ACCUM_TYPE) {
    printf("useFP16Accumulation %s\n", argv[ACCUM_TYPE]);
    if (std::string(argv[ACCUM_TYPE]) == "True") {
      param.useFP16Accumulation = true;
    } else if (std::string(argv[ACCUM_TYPE]) == "False") {
      param.useFP16Accumulation = false;
    } else {
      llvm_unreachable("Invalid useFP16Accumulation");
    }
  } else {
    param.useFP16Accumulation = false;
  }
  if (argc > DEVICE_ID) {
    printf("devId %s\n", argv[DEVICE_ID]);
    param.devId = std::string(argv[DEVICE_ID]);
  } else {
    param.devId = std::string("");
  }
  printf("\n\n");
  return param;
}

int main(int argc, char *argv[]) {

  printf("SLS Microbenchmark\n");
  printf(
      "Usage: SLSBench batchSize(Int) numIndicesPerBatch(Int) "
      "numIndicesPerBatchPad(Int) numTableEntries(Int) "
      "numElementsPerRow(int) numReps(Int) "
      "numAsyncLaunches(Int) numSLSNodes(Int) "
      "slsKindStr(\"QuantizedWeighted\"|\"QuantizedUnweighted\"|"
      "\"NonquantizedWeighted\"|"
      "\"NonquantizedUnweighted\") "
      "sortedStr(\"Sorted\"|\"Unsorted\") backendStr(String) "
      "dtypeStr(\"Float16\"|\"Float32\") "
      "addClipStr(\"True\"|\"False\")\nQuantized only options: "
      "quantizationDtypeStr(\"Int8\"|\"Int4\") "
      "useFP16AccumulationStr(\"True\"|\"False\") \nOptional: dev_id(Int)\n");
  printf("\n");

  std::vector<SLSParam> params;
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
      SLSParam param = parseArgs(argVec.size(), argVec.data());
      params.push_back(param);
      runHeader = std::string("_,benchName,_,filename");
      runPrefix = std::string(strFormat("SLSBench,SW,%s", fname.c_str()));
    }
  }
  // Using command line
  else if (argc == 14 || argc == 15 || argc == 16 || argc == 17) {
    SLSParam param = parseArgs(argc, argv);
    params.push_back(param);

    runHeader = std::string(
        "_,benchName,_,batchSize,numIndicesPerBatch,numIndicesPerBatchPad,"
        "numTableEntries,"
        "numElementsPerRow,numReps,numAsyncLaunches,numSLSNodes,slsKindStr,"
        "backendStr,dtypeStr,addClipStr,quantizationDtypeStr,"
        "useFP16AccumulationStr");
    runPrefix = std::string(strFormat(
        "SLSBench,SW,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%s,%s,%s,%"
        "s,%s,%"
        "s,%s",
        (size_t)param.batchSize, (size_t)param.numIndicesPerBatch,
        (size_t)param.numIndicesPerBatchPad, (size_t)param.numTableEntries,
        (size_t)param.numElementsPerRow, (size_t)param.numReps,
        (size_t)param.numAsyncLaunches, (size_t)param.numSLSNodes, argv[9],
        argv[10], argv[11], argv[12], argv[13], argv[14], argv[15]));
  } else {
    llvm_unreachable("Invalid command line");
  }

  SLSParam param = params.front();
  SLSBench b(param.batchSize, param.numAsyncLaunches, param.backendStr, params,
             param.devId);
  auto times = bench(&b, param.numReps);

  printf("%s,runtime,gbytesPerSec\n", runHeader.c_str());
  for (auto t : times) {
    printf("BenchResult,%s,%f,%f\n", runPrefix.c_str(),
           t / param.numAsyncLaunches, b.gbytes() * param.numAsyncLaunches / t);
  }
  double min = *(std::min_element(times.begin(), times.end()));
  dim_t midElt = times.size() / 2;
  std::nth_element(times.begin(), times.begin() + midElt, times.end());
  double median = times[midElt];
  double medianRuntime = median / ((double)param.numAsyncLaunches);
  double minRuntime = min / ((double)param.numAsyncLaunches);
  printf("%s,medianRuntime,minRuntime,medianGbytesPerSec,maxGbytesPerSec\n",
         runHeader.c_str());
  printf("BenchSummary,%s,%f,%f,%f,%f\n", runPrefix.c_str(), medianRuntime,
         minRuntime, b.gbytes() / medianRuntime, b.gbytes() / minRuntime);
}
