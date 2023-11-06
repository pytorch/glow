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
#include "Bench.h"
#include "glow/Base/DimType.h"
#include "glow/Base/Tensor.h"
#include "glow/Base/Type.h"
#include "glow/Graph/Graph.h"
#include "glow/Graph/Nodes.h"
#include <algorithm>
#include <array>
#include <cstdlib>
#include <fstream>
#include <future>
#include <random>
#include <string>

#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Optimizer/GraphOptimizer/GraphOptimizer.h"

namespace glow {
namespace benchmark {
enum SLSKind {
  NONQUANTIZED_UNWEIGHTED,
  NONQUANTIZED_WEIGHTED,
  QUANTIZED_UNWEIGHTED,
  QUANTIZED_WEIGHTED
};

enum BenchmarkType { SLS_BENCH, EB_BENCH };

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
  ElemKind fusedDtype;
  ElemKind dtype;
  bool useFP16Accumulation;
  bool hasEndOffset;
  bool convertFusedToFP32;
};

inline std::string getBenchPrefix(const BenchmarkType benchType,
                                  const bool toLower = false) {
  std::string prefix;
  switch (benchType) {
  case SLS_BENCH:
    prefix = "SLS";
    break;
  case EB_BENCH:
    prefix = "EB";
    break;
  default:
    throw std::runtime_error("Bench type is not implemented!");
  }

  if (toLower)
    std::transform(prefix.begin(), prefix.end(), prefix.begin(),
                   [](unsigned char c) { return std::tolower(c); });

  return prefix;
}

inline std::string getSLSDescription(const SLSParam &param,
                                     const BenchmarkType benchType) {
  std::string SLSStr;
  if (benchType == EB_BENCH) {
    SLSStr = "EB";
  } else if (benchType == SLS_BENCH) {
    switch (param.slsKind) {
    case NONQUANTIZED_UNWEIGHTED:
      SLSStr = "SLS";
      break;
    case NONQUANTIZED_WEIGHTED:
      SLSStr = "SLWS";
      break;
    case QUANTIZED_UNWEIGHTED:
      SLSStr = "RWQLSS";
      break;
    case QUANTIZED_WEIGHTED:
      SLSStr = "RWQLSWS";
      break;
    default:
      throw std::runtime_error("Unsupported slsKind type!");
    }
  } else {
    throw std::runtime_error("Bench type is not implemented!");
  }

  return strFormat(
      "%s__%zu_%zu__%zu__%zu__%zu", SLSStr.c_str(),
      (size_t)param.numIndicesPerBatchMin, (size_t)param.numIndicesPerBatchMax,
      (size_t)param.numIndicesPerBatchPad, (size_t)param.numTableEntries,
      (size_t)param.numElementsPerRow);
}

inline double countSLSGbytes(const SLSParam &param, const dim_t batchSize) {

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
      (double)(param.numIndicesPerBatchMin + param.numIndicesPerBatchMax) / 2.0;

  // Embedding data
  double input_gbytes = 0.0;
  if ((param.slsKind == NONQUANTIZED_WEIGHTED) ||
      (param.slsKind == NONQUANTIZED_UNWEIGHTED)) {
    input_gbytes += (param.numSLSNodes * batchSize * avgIndicesPerBatch *
                     (param.numElementsPerRow * elementSize)) /
                    1e9;
  } else { // Quantized
    if (param.fusedDtype == ElemKind::UInt8FusedFP16QTy) {
      input_gbytes += (param.numSLSNodes * batchSize * avgIndicesPerBatch *
                       (param.numElementsPerRow + 2 * scaleSize)) /
                      1e9;
    } else { // Int4
      input_gbytes += (param.numSLSNodes * batchSize * avgIndicesPerBatch *
                       ((param.numElementsPerRow + 1) / 2 + 2 * scaleSize)) /
                      1e9;
    }
  }

  // + indices
  input_gbytes +=
      (param.numSLSNodes * batchSize * avgIndicesPerBatch * sizeof(int32_t)) /
      1e9;

  // + weights
  if ((param.slsKind == QUANTIZED_WEIGHTED) ||
      (param.slsKind == NONQUANTIZED_WEIGHTED)) {
    input_gbytes +=
        (param.numSLSNodes * batchSize * avgIndicesPerBatch * elementSize) /
        1e9;
  }

  // + offsets
  input_gbytes += (param.numSLSNodes * batchSize * sizeof(int32_t)) / 1e9;

  double output_gbytes = (param.numSLSNodes * batchSize *
                          (param.numElementsPerRow * elementSize)) /
                         1e9;

  return input_gbytes + output_gbytes;
}

inline void
addSLSNode(const std::unique_ptr<Module> &mod, Function *fn,
           const SLSParam &param, const dim_t batchSize,
           const dim_t asyncLaunchSize,
           const std::vector<std::unique_ptr<ExecutionContext>> &contexts,
           std::vector<std::vector<Tensor>> &indicesRealVec,
           std::vector<std::vector<Tensor>> &weightsRealVec,
           const BenchmarkType benchType) {

  const std::string benchPrefix = getBenchPrefix(benchType, false);

  // TODO(T115782297): Support non-quantized and weighted embedding bags.
  if (benchType == EB_BENCH)
    CHECK_EQ(param.slsKind, QUANTIZED_UNWEIGHTED);

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
  Constant *dataConstant =
      mod->createConstant(benchPrefix + "Data", dataConstantTensor);

  // Create placeholders for weights, indices and offsets
  const dim_t maxNumIndicesWeights = param.numIndicesPerBatchPad * batchSize;

  // SLSBench vs EBBench : weights SLS uses Placeholder while EB uses Constant
  Constant *weightsC;
  Placeholder *weightsP;
  switch (benchType) {
  case SLS_BENCH:
    weightsP = mod->createPlaceholder(param.dtype, {maxNumIndicesWeights},
                                      "weights", false);
    break;
  case EB_BENCH:
    weightsC =
        mod->createConstant(param.dtype, {maxNumIndicesWeights}, "weights");
    break;
  default:
    throw std::runtime_error("Bench type is not implemented!");
  }
  auto *indices = mod->createPlaceholder(ElemKind::Int64ITy,
                                         {maxNumIndicesWeights}, "indices",
                                         /* isTrainable */ false);

  // lengths are used to populate offsets values
  auto *lengths =
      mod->createPlaceholder(ElemKind::Int32ITy, {batchSize}, "lengths",
                             /* isTrainable */ false);
  auto *offsets =
      mod->createPlaceholder(ElemKind::Int32ITy, {batchSize + 1}, "offsets",
                             /* isTrainable */ false);

  size_t totalLengthsSum = 0;
  size_t totalNumLengths = 0;
  for (dim_t i = 0; i < asyncLaunchSize; i++) {
    auto lengthsHandle = contexts[i]
                             ->getPlaceholderBindings()
                             ->allocate(lengths)
                             ->getHandle<int32_t>();
    auto offsetsHandle = contexts[i]
                             ->getPlaceholderBindings()
                             ->allocate(offsets)
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
      offsetsHandle.raw(j) = lengthsSum;
      lengthsSum += nextLength;
      totalNumLengths += 1;
    }
    totalLengthsSum += lengthsSum;
    offsetsHandle.raw(lengthsHandle.size()) = lengthsSum;

    // Create and sort indices
    Tensor indicesReal(ElemKind::Int64ITy, {lengthsSum});
    indicesReal.getHandle<int64_t>().randomize(0, param.numTableEntries,
                                               mod->getPRNG());
    // Sort each segment
    if (param.isSorted) {
      auto *indicesRealPtr = (int64_t *)indicesReal.getUnsafePtr();
      for (size_t j = 0, e = lengthsHandle.size(); j < e; j++) {
        const size_t curLength = lengthsHandle.raw(j);
        std::sort(indicesRealPtr, indicesRealPtr + curLength);
        indicesRealPtr += curLength;
      }
    }
    indicesRealVec[i].push_back(std::move(indicesReal));

    // Create and assign weights
    if (param.dtype == ElemKind::FloatTy) {
      Tensor weightsReal(ElemKind::FloatTy, {lengthsSum});
      weightsReal.getHandle<float>().clear(1.0f);
      weightsRealVec[i].push_back(std::move(weightsReal));
    } else if (param.dtype == ElemKind::Float16Ty) {
      Tensor weightsReal(ElemKind::Float16Ty, {lengthsSum});
      weightsReal.getHandle<float16_t>().clear(1.0f);
      weightsRealVec[i].push_back(std::move(weightsReal));
    }

    // SLSBench vs EBBench : weight assignment difference
    switch (benchType) {
    case SLS_BENCH: {
      Tensor weightsPartial(weightsRealVec[i].back().getUnsafePtr(),
                            weightsP->getType(),
                            weightsRealVec[i].back().getSizeInBytes());
      contexts[i]->getPlaceholderBindings()->insert(weightsP,
                                                    std::move(weightsPartial));
      break;
    }
    case EB_BENCH: {
      weightsC->assign(&weightsRealVec[i].back());
      break;
    }
    default: {
      throw std::runtime_error("Bench type is not implemented!");
    }
    }

    Tensor indicesPartial(indicesRealVec[i].back().getUnsafePtr(),
                          indices->getType(),
                          indicesRealVec[i].back().getSizeInBytes());

    contexts[i]->getPlaceholderBindings()->insert(indices,
                                                  std::move(indicesPartial));

  } // i

  // Calculate the average length based on all of the lengths generated.
  const double avgLength = (double)totalLengthsSum / (double)totalNumLengths;

  // Create EB node, optional clip node, and save node
  const LengthsMode LM =
      avgLength == 1.f ? LengthsMode::AllOne : LengthsMode::Variable;
  Node *R = nullptr;

  // SLSBench vs EBBench: result processing differences
  if (benchType == SLS_BENCH) {
    switch (param.slsKind) {
    case QUANTIZED_UNWEIGHTED:
      R = fn->createFusedRowwiseQuantizedSparseLengthsSum(
          getSLSDescription(param, benchType), dataConstant, indices, lengths,
          param.useFP16Accumulation, LM, avgLength);
      break;
    case QUANTIZED_WEIGHTED:
      R = fn->createFusedRowwiseQuantizedSparseLengthsWeightedSum(
          getSLSDescription(param, benchType), dataConstant, weightsP, indices,
          lengths, param.useFP16Accumulation, LM, avgLength);
      break;
    case NONQUANTIZED_WEIGHTED:
      R = fn->createSparseLengthsWeightedSum(
          getSLSDescription(param, benchType), dataConstant, weightsP, indices,
          lengths, LM, avgLength);
      break;
    case NONQUANTIZED_UNWEIGHTED:
      R = fn->createSparseLengthsSum(getSLSDescription(param, benchType),
                                     dataConstant, indices, lengths, LM,
                                     avgLength);
      break;
    default:
      throw std::runtime_error("Unsupported slsKind type!");
    }
  } else if (benchType == EB_BENCH) {
    R = fn->createEmbeddingBagByteRowwiseOffsets(
        getSLSDescription(param, benchType), dataConstant, weightsC, indices,
        offsets, param.useFP16Accumulation, /*hasEndOffset*/ true, LM,
        avgLength);
  }

  SaveNode *S = nullptr;
  if (param.addClip) {
    auto *clp = fn->createClip("clip", R, -65504.0f, 65504.0f);
    S = fn->createSave("save", clp);
  } else {
    S = fn->createSave("save", R);
  }

  // for each context, add output bindings
  for (dim_t i = 0; i < asyncLaunchSize; i++) {
    contexts[i]->getPlaceholderBindings()->allocate(S->getPlaceholder());
  }
}

inline void setupSLS(const dim_t batchSize, const dim_t asyncLaunchSize,
                     const std::string &backendStr, const std::string &devId,
                     const bool convertFusedToFP32,
                     std::unique_ptr<runtime::HostManager> &hostManager,
                     std::vector<std::unique_ptr<ExecutionContext>> &contexts,
                     std::vector<std::vector<Tensor>> &indicesRealVec,
                     std::vector<std::vector<Tensor>> &weightsRealVec,
                     const std::vector<SLSParam> &params, const bool dumpOnnx,
                     const BenchmarkType benchType) {

  const std::string benchPrefix = getBenchPrefix(benchType, true);

  // Create execution contexts here
  for (dim_t i = 0; i < asyncLaunchSize; i++) {
    std::unique_ptr<ExecutionContext> context(new ExecutionContext);
    contexts.push_back(std::move(context));
  }

  // Setup host manager
  std::vector<std::unique_ptr<runtime::DeviceConfig>> configs;
  auto config = glow::make_unique<runtime::DeviceConfig>(backendStr.c_str());
  if (devId != "") {
    config->parameters["DeviceID"] = devId.c_str();
  }
  configs.push_back(std::move(config));
  hostManager = glow::make_unique<runtime::HostManager>(std::move(configs));

  // Create a function
  std::unique_ptr<Module> mod(new Module);
  auto fn = mod->createFunction("singleNode");

  // Keep tensors around so they aren't deleted
  indicesRealVec.resize(asyncLaunchSize);
  weightsRealVec.resize(asyncLaunchSize);

  // Add SLS nodes
  for (auto param : params) {
    for (dim_t i = 0; i < param.numSLSNodes; i++) {
      addSLSNode(mod, fn, param, batchSize, asyncLaunchSize, contexts,
                 indicesRealVec, weightsRealVec, benchType);
    }
  }

  // SLSBench vs EBBench: Save DAG in .dot files differently
  fn->dumpDAG(benchPrefix + "bench.dot");
  CompilationContext ctx;
  ctx.dumpFinalGraph = true;
  ctx.serializeCompiledDAG = dumpOnnx;

  if (convertFusedToFP32) {
    ctx.precisionConfig.convert4BitFusedToFP32 = true;
    ctx.precisionConfig.convert8BitFusedToFP32 = true;
  }

  EXIT_ON_ERR(hostManager->addNetwork(std::move(mod), ctx));
}

inline void runSLS(const dim_t asyncLaunchSize,
                   const std::unique_ptr<runtime::HostManager> &hostManager,
                   std::vector<std::unique_ptr<ExecutionContext>> &contexts) {
  std::vector<std::unique_ptr<ExecutionContext>> localContexts(asyncLaunchSize);
  std::vector<std::promise<void>> promises(asyncLaunchSize);
  std::vector<std::future<void>> futures;

  // Launch a number of independent requests
  int i = 0;
  for (auto &promise : promises) {
    futures.push_back(promise.get_future());
    hostManager->runNetwork("singleNode", std::move(contexts[i]),
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
  for (dim_t j = 0; j < asyncLaunchSize; j++) {
    contexts[j] = std::move(localContexts[j]);
  }
}

// Indices of arguments
#define ROWWISE_QUANT 14
#define ACCUM_TYPE 15
#define DEVICE_ID 16

inline SLSParam parseArgs(int argc, char *argv[], BenchmarkType benchType) {
  std::string benchPrefix = getBenchPrefix(benchType, false);
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
  printf("num%sNodes %zu\n", benchPrefix.c_str(), (size_t)param.numSLSNodes);
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
    llvm_unreachable(strFormat("Invalid %s Kind", benchPrefix.c_str()).c_str());
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

inline std::tuple<std::vector<SLSParam>, std::string, std::string>
preMain(const int argc, char *argv[], const BenchmarkType benchType) {
  std::string benchPrefix = getBenchPrefix(benchType);
  std::string benchName = getBenchPrefix(benchType) + "Bench";

  printf("%s Microbenchmark\n", benchPrefix.c_str());
  printf("Usage: %sBench batchSize(Int) "
         "[numIndicesPerBatch(Int) | "
         "numIndicesPerBatchMin(Int):numIndicesPerBatchMax(Int)] "
         "numIndicesPerBatchPad(Int) numTableEntries(Int) "
         "numElementsPerRow(int) numReps(Int) "
         "numAsyncLaunches(Int) num%sNodes(Int) "
         "slsKindStr(\"QuantizedWeighted\"|\"QuantizedUnweighted\"|"
         "\"NonquantizedWeighted\"|"
         "\"NonquantizedUnweighted\") "
         "sortedStr(\"Sorted\"|\"Unsorted\") backendStr(String) "
         "dtypeStr(\"Float16\"|\"Float32\") "
         "addClipStr(\"True\"|\"False\")\nQuantized only options: "
         "quantizationDtypeStr(\"Int8\"|\"Int4\") "
         "useFP16AccumulationStr(\"True\"|\"False\") \n"
         "Optional: dev_id(Int)\n",
         benchPrefix.c_str(), benchPrefix.c_str());
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
      SLSParam param = parseArgs(argVec.size(), argVec.data(), benchType);
      params.push_back(param);
      runHeader = std::string("_,benchName,_,filename");
      runPrefix =
          std::string(strFormat("%s,SW,%s", benchName.c_str(), fname.c_str()));
    }
  }
  // Using command line
  else if (argc == 14 || argc == 15 || argc == 16 || argc == 17 || argc == 18) {
    SLSParam param = parseArgs(argc, argv, benchType);
    params.push_back(param);

    runHeader = std::string(strFormat(
        "_,benchName,_,batchSize,numIndicesPerBatchMin:numIndicesPerBatchMax,"
        "numIndicesPerBatchPad,numTableEntries,numElementsPerRow,numReps,"
        "numAsyncLaunches,num%sNodes,slsKindStr,backendStr,dtypeStr,"
        "addClipStr,quantizationDtypeStr,useFP16AccumulationStr",
        benchPrefix.c_str()));
    runPrefix = std::string(strFormat(
        "%s,SW,%zu,%zu:%zu,%zu,%zu,%zu,%zu,%zu,%zu,%s,%s,%s,%s,%s,%s,%s",
        benchName.c_str(), (size_t)param.batchSize,
        (size_t)param.numIndicesPerBatchMin,
        (size_t)param.numIndicesPerBatchMax,
        (size_t)param.numIndicesPerBatchPad, (size_t)param.numTableEntries,
        (size_t)param.numElementsPerRow, (size_t)param.numReps,
        (size_t)param.numAsyncLaunches, (size_t)param.numSLSNodes, argv[9],
        argv[10], argv[11], argv[12], argv[13], argv[14], argv[15]));
  } else {
    llvm_unreachable("Invalid command line");
  }

  return std::make_tuple(params, runPrefix, runHeader);
}

inline void printSummary(const std::string &runPrefix,
                         const std::string &runHeader, const SLSParam &param,
                         std::vector<double> &times, const double gb) {

  printf("%s,runtime,gbytesPerSec\n", runHeader.c_str());
  for (auto t : times) {
    printf("BenchResult,%s,%f,%f\n", runPrefix.c_str(),
           t / param.numAsyncLaunches, gb * param.numAsyncLaunches / t);
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
         minRuntime, gb / medianRuntime, gb / minRuntime);
}
}; // namespace benchmark
}; // namespace glow
