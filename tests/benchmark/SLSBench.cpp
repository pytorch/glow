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

llvm::cl::OptionCategory SLSBenchCat("SLSBench Category");
llvm::cl::opt<bool> dumpOnnx("dump_onnx",
                             llvm::cl::desc("dump onnx text format for model"),
                             llvm::cl::Optional, llvm::cl::init(false),
                             llvm::cl::cat(SLSBenchCat));

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
  dim_t numIndicesPerBatchMin;
  dim_t numIndicesPerBatchMax;
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
  bool convertFusedToFP32;
};

std::string getSLSDescription(SLSParam param) {
  std::string SLSStr =
      (param.slsKind == NONQUANTIZED_UNWEIGHTED) ? std::string("SLS")
      : (param.slsKind == NONQUANTIZED_WEIGHTED) ? std::string("SLWS")
      : (param.slsKind == QUANTIZED_UNWEIGHTED)  ? std::string("RWQLSS")
                                                 : std::string("RWQLSWS");

  return strFormat(
      "%s__%zu_%zu__%zu__%zu__%zu", SLSStr.c_str(),
      (size_t)param.numIndicesPerBatchMin, (size_t)param.numIndicesPerBatchMax,
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
  bool convertFusedToFP32_;
  std::string devId_;

public:
  SLSBench(dim_t batchSize_, dim_t asyncLaunchSize_, std::string backendStr_,
           std::vector<SLSParam> params_, bool convertFusedToFP32,
           std::string devId_ = std::string(""))
      : batchSize_(batchSize_), asyncLaunchSize_(asyncLaunchSize_),
        backendStr_(backendStr_), params_(params_),
        convertFusedToFP32_(convertFusedToFP32), devId_(devId_) {}

  double countSLSGbytes(SLSParam param) const {

    dim_t elementSize = 2;
    if (param.dtype == ElemKind::FloatTy) {
      elementSize = 4;
    }

    dim_t scaleSize = 2;
    if (param.convertFusedToFP32) {
      scaleSize = 4;
    }

    // This is approximate when numIndicesPerBatchMin != numIndicesPerBatchMax.
    const double avgIndicesPerBatch =
        (double)(param.numIndicesPerBatchMin + param.numIndicesPerBatchMax) /
        2.0;

    // Embedding data
    double input_gbytes = 0.0;
    if ((param.slsKind == NONQUANTIZED_WEIGHTED) ||
        (param.slsKind == NONQUANTIZED_UNWEIGHTED)) {
      input_gbytes += (param.numSLSNodes * batchSize_ * avgIndicesPerBatch *
                       (param.numElementsPerRow * elementSize)) /
                      1e9;
    } else { // Quantized
      if (param.fusedDtype == ElemKind::UInt8FusedFP16QTy) {
        input_gbytes += (param.numSLSNodes * batchSize_ * avgIndicesPerBatch *
                         (param.numElementsPerRow + 2 * scaleSize)) /
                        1e9;
      } else { // Int4
        input_gbytes += (param.numSLSNodes * batchSize_ * avgIndicesPerBatch *
                         ((param.numElementsPerRow + 1) / 2 + 2 * scaleSize)) /
                        1e9;
      }
    }

    // + indices
    input_gbytes += (param.numSLSNodes * batchSize_ * avgIndicesPerBatch *
                     sizeof(int32_t)) /
                    1e9;

    // + weights
    if ((param.slsKind == QUANTIZED_WEIGHTED) ||
        (param.slsKind == NONQUANTIZED_WEIGHTED)) {
      input_gbytes +=
          (param.numSLSNodes * batchSize_ * avgIndicesPerBatch * elementSize) /
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

    // Constant needed for Non-quantized case
    Tensor dataConstantTensor;
    if ((param.slsKind == NONQUANTIZED_WEIGHTED) ||
        (param.slsKind == NONQUANTIZED_UNWEIGHTED)) {
      dataConstantTensor =
          Tensor(param.dtype, {param.numTableEntries, param.numElementsPerRow});
    } else {
      // If RWQ then we need to account for per-row scale/offset in the shape.
      int64_t numBytePerRow = param.numElementsPerRow;
      if (param.fusedDtype == ElemKind::UInt4FusedFP16QTy) {
        // For 4bit tables the number of bytes should be halved (rounded up).
        numBytePerRow = (numBytePerRow + 1) / 2;
      }
      const dim_t numTotalColumns = numBytePerRow + 2 * sizeof(float16_t);
      dataConstantTensor = Tensor(
          param.fusedDtype, {param.numTableEntries, numTotalColumns}, 1.0, 0);
    }
    Constant *dataConstant = mod->createConstant("SLSData", dataConstantTensor);

    // Create placeholders for weights, indices and lengths
    const dim_t maxNumIndicesWeights = param.numIndicesPerBatchPad * batchSize_;
    auto *weights = mod->createPlaceholder(param.dtype, {maxNumIndicesWeights},
                                           "weights", false);

    auto *indices = mod->createPlaceholder(ElemKind::Int64ITy,
                                           {maxNumIndicesWeights}, "indices",
                                           /* isTrainable */ false);

    auto *lengths =
        mod->createPlaceholder(ElemKind::Int32ITy, {batchSize_}, "lengths",
                               /* isTrainable */ false);

    size_t totalLengthsSum = 0;
    size_t totalNumLengths = 0;
    for (dim_t i = 0; i < asyncLaunchSize_; i++) {
      auto lengthsHandle = contexts_[i]
                               ->getPlaceholderBindings()
                               ->allocate(lengths)
                               ->getHandle<int32_t>();

      // Generate lengths across a uniform distribution.
      lengthsHandle.randomize(param.numIndicesPerBatchMin,
                              param.numIndicesPerBatchMax, mod->getPRNG());
      dim_t lengthsSum = 0;
      for (size_t j = 0, e = lengthsHandle.size(); j < e; j++) {
        auto &nextLength = lengthsHandle.raw(j);
        if (lengthsSum == maxNumIndicesWeights) {
          // If we have maxed out the maximum allowed indices then zero out the
          // rest of the lengths.
          nextLength = 0;
          continue;
        } else if (lengthsSum + nextLength > maxNumIndicesWeights) {
          // If the next length will equal or overflow the maximum allowed
          // indices then fill it up totally.
          nextLength = maxNumIndicesWeights - lengthsSum;
        }
        lengthsSum += nextLength;
        totalNumLengths += 1;
      }
      totalLengthsSum += lengthsSum;

      // Create and sort indices
      Tensor indicesReal(ElemKind::Int64ITy, {lengthsSum});
      indicesReal.getHandle<int64_t>().randomize(0, param.numTableEntries,
                                                 mod->getPRNG());
      // Sort each segment
      if (param.isSorted) {
        int64_t *indicesRealPtr = (int64_t *)indicesReal.getUnsafePtr();
        for (size_t j = 0, e = lengthsHandle.size(); j < e; j++) {
          const size_t curLength = lengthsHandle.raw(j);
          std::sort(indicesRealPtr, indicesRealPtr + curLength);
          indicesRealPtr += curLength;
        }
      }
      indicesReal_[i].push_back(std::move(indicesReal));

      // Create weights
      if (param.dtype == ElemKind::FloatTy) {
        Tensor weightsReal(ElemKind::FloatTy, {lengthsSum});
        weightsReal.getHandle<float>().clear(1.0f);
        weightsReal_[i].push_back(std::move(weightsReal));
      } else if (param.dtype == ElemKind::Float16Ty) {
        Tensor weightsReal(ElemKind::Float16Ty, {lengthsSum});
        weightsReal.getHandle<float16_t>().clear(1.0f);
        weightsReal_[i].push_back(std::move(weightsReal));
      }

      Tensor indicesPartial(indicesReal_[i].back().getUnsafePtr(),
                            indices->getType(),
                            indicesReal_[i].back().getSizeInBytes());

      contexts_[i]->getPlaceholderBindings()->insert(indices,
                                                     std::move(indicesPartial));

      Tensor weightsPartial(weightsReal_[i].back().getUnsafePtr(),
                            weights->getType(),
                            weightsReal_[i].back().getSizeInBytes());
      contexts_[i]->getPlaceholderBindings()->insert(weights,
                                                     std::move(weightsPartial));
    } // i

    // Calculate the average length based on all of the lengths generated.
    const double avgLength = (double)totalLengthsSum / (double)totalNumLengths;

    // Create SLS node, optional clip node, and save node
    const LengthsMode LM =
        avgLength == 1.f ? LengthsMode::AllOne : LengthsMode::Variable;
    Node *R = nullptr;
    if (param.slsKind == QUANTIZED_UNWEIGHTED) {
      R = fn->createFusedRowwiseQuantizedSparseLengthsSum(
          getSLSDescription(param), dataConstant, indices, lengths,
          param.useFP16Accumulation, LM, avgLength);
    } else if (param.slsKind == QUANTIZED_WEIGHTED) {
      R = fn->createFusedRowwiseQuantizedSparseLengthsWeightedSum(
          getSLSDescription(param), dataConstant, weights, indices, lengths,
          param.useFP16Accumulation, LM, avgLength);
    } else if (param.slsKind == NONQUANTIZED_WEIGHTED) {
      R = fn->createSparseLengthsWeightedSum(getSLSDescription(param),
                                             dataConstant, weights, indices,
                                             lengths, LM, avgLength);
    } else { // NonquantizedUnweighted
      R = fn->createSparseLengthsSum(getSLSDescription(param), dataConstant,
                                     indices, lengths, LM, avgLength);
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
      for (dim_t i = 0; i < param.numSLSNodes; i++) {
        addSLSNode(mod, fn, param);
      }
    }

    fn->dumpDAG("slsbench.dot");
    CompilationContext ctx;
    ctx.dumpFinalGraph = true;
    ctx.serializeCompiledDAG = dumpOnnx;

    if (convertFusedToFP32_) {
      ctx.precisionConfig.convert4BitFusedToFP32 = true;
      ctx.precisionConfig.convert8BitFusedToFP32 = true;
    }

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
  llvm::StringRef numIndicesPerBatchStr(argv[2]);
  auto split = numIndicesPerBatchStr.split(':');
  if (split.second == "") {
    ASSIGN_VALUE_OR_FATAL(param.numIndicesPerBatchMin, getIntFromStr(argv[2]));
    param.numIndicesPerBatchMax = param.numIndicesPerBatchMin;
  } else {
    ASSIGN_VALUE_OR_FATAL(param.numIndicesPerBatchMin,
                          getIntFromStr(split.first));
    ASSIGN_VALUE_OR_FATAL(param.numIndicesPerBatchMax,
                          getIntFromStr(split.second));
    CHECK_LE(param.numIndicesPerBatchMin, param.numIndicesPerBatchMax);
  }
  ASSIGN_VALUE_OR_FATAL(param.numIndicesPerBatchPad, getIntFromStr(argv[3]));
  CHECK_LE(param.numIndicesPerBatchMax, param.numIndicesPerBatchPad);
  ASSIGN_VALUE_OR_FATAL(param.numTableEntries, getIntFromStr(argv[4]));
  ASSIGN_VALUE_OR_FATAL(param.numElementsPerRow, getIntFromStr(argv[5]));
  ASSIGN_VALUE_OR_FATAL(param.numReps, getIntFromStr(argv[6]));
  ASSIGN_VALUE_OR_FATAL(param.numAsyncLaunches, getIntFromStr(argv[7]));
  ASSIGN_VALUE_OR_FATAL(param.numSLSNodes, getIntFromStr(argv[8]));
  printf("batchSize %zu\n", (size_t)param.batchSize);
  printf("numIndicesPerBatchMin %zu\n", (size_t)param.numIndicesPerBatchMin);
  printf("numIndicesPerBatchMax %zu\n", (size_t)param.numIndicesPerBatchMax);
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
  param.convertFusedToFP32 = false;
  if (argc > ROWWISE_QUANT) {
    printf("fusedDtype %s\n", argv[ROWWISE_QUANT]);
    if (std::string(argv[ROWWISE_QUANT]) == "Int8") {
      param.fusedDtype = ElemKind::UInt8FusedFP16QTy;
    } else if (std::string(argv[ROWWISE_QUANT]) == "Int8_Fp32") {
      param.fusedDtype = ElemKind::UInt8FusedFP16QTy;
      param.convertFusedToFP32 = true;
    } else if (std::string(argv[ROWWISE_QUANT]) == "Int4") {
      param.fusedDtype = ElemKind::UInt4FusedFP16QTy;
    } else if (std::string(argv[ROWWISE_QUANT]) == "Int4_Fp32") {
      param.fusedDtype = ElemKind::UInt4FusedFP16QTy;
      param.convertFusedToFP32 = true;
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
  printf("Usage: SLSBench batchSize(Int) "
         "[numIndicesPerBatch(Int) | "
         "numIndicesPerBatchMin(Int):numIndicesPerBatchMax(Int)] "
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
         "useFP16AccumulationStr(\"True\"|\"False\") \n"
         "Optional: dev_id(Int)\n");
  printf("\n");
  printf("Standard Glow command-line options may be passed via the GLOW_OPTS "
         "environment variable\n");
  benchParseGlowOpts(argc, argv);

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
  else if (argc == 14 || argc == 15 || argc == 16 || argc == 17 || argc == 18) {
    SLSParam param = parseArgs(argc, argv);
    params.push_back(param);

    runHeader = std::string(
        "_,benchName,_,batchSize,numIndicesPerBatchMin:numIndicesPerBatchMax,"
        "numIndicesPerBatchPad,numTableEntries,numElementsPerRow,numReps,"
        "numAsyncLaunches,numSLSNodes,slsKindStr,backendStr,dtypeStr,"
        "addClipStr,quantizationDtypeStr,useFP16AccumulationStr");
    runPrefix = std::string(strFormat(
        "SLSBench,SW,%zu,%zu:%zu,%zu,%zu,%zu,%zu,%zu,%zu,%s,%s,%s,%s,%s,%s,%s",
        (size_t)param.batchSize, (size_t)param.numIndicesPerBatchMin,
        (size_t)param.numIndicesPerBatchMax,
        (size_t)param.numIndicesPerBatchPad, (size_t)param.numTableEntries,
        (size_t)param.numElementsPerRow, (size_t)param.numReps,
        (size_t)param.numAsyncLaunches, (size_t)param.numSLSNodes, argv[9],
        argv[10], argv[11], argv[12], argv[13], argv[14], argv[15]));
  } else {
    llvm_unreachable("Invalid command line");
  }

  SLSParam param = params.front();
  SLSBench b(param.batchSize, param.numAsyncLaunches, param.backendStr, params,
             param.convertFusedToFP32, param.devId);
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
