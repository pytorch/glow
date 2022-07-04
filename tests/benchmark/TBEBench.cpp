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

#include "tests/unittests/BackendTestUtils.h"

using namespace glow;

#define DEVICE_ID 16

/*
 * This class implements a TableBatchedEmbedding microbenchmark. In this
 * microbenchmark, there are a set of TBE nodes chained together. Each TBE
 * node specifies the number of tables (numTables_), the batch dimension
 * (batchSize_), embedding dimension (numElementsPerRow_), pooling factor
 * (numIndicesPerBatchPad_).
 *
 * Microbenchmarks are generally useful for understanding performance
 * through targeted experiementation and are not representative of
 * end-to-end workloads.
 */

llvm::cl::OptionCategory TBEBenchCat("TBEBench Category");
llvm::cl::opt<bool> checkCorrectness(
    "check-results",
    llvm::cl::desc("Check the correctness of the results against the reference "
                   "backend (Interpreter)"),
    llvm::cl::Optional, llvm::cl::init(false), llvm::cl::cat(TBEBenchCat));
llvm::cl::opt<bool> dumpOnnx("dump_onnx",
                             llvm::cl::desc("dump onnx text format for model"),
                             llvm::cl::Optional, llvm::cl::init(false),
                             llvm::cl::cat(TBEBenchCat));
struct TBEParam {
  dim_t batchSize_;
  dim_t numReps_;
  dim_t numAsyncLaunches_;
  std::string backendStr_;
  std::string devId_;
  dim_t numIndicesPerBatchMin_;
  dim_t numIndicesPerBatchMax_;
  dim_t numIndicesPerBatchPad_;
  dim_t numTableEntries_;
  dim_t numTables_;
  dim_t numElementsPerRow_;
  dim_t numTBENodes_;
  bool weighted_;
  bool isSorted_;
  bool addClip_;
  ElemKind fusedDtype_;
  ElemKind dtype_;
};

class TBEBench : public Benchmark {
  std::unique_ptr<runtime::HostManager> hostManager_;
  ExecutionContext context_;
  TBEParam param;
  PlaceholderBindings &bindings_;

public:
  TBEBench(TBEParam param_)
      : param(param_), bindings_(*context_.getPlaceholderBindings()) {}

  inline void addTBENode(const std::unique_ptr<Module> &mod, Function *fn,
                         const TBEParam &param) {

    Tensor dataConstantTensor;
    int64_t numBytePerRow = param.numElementsPerRow_;
    if (param.fusedDtype_ == ElemKind::UInt4FusedFP16QTy) {
      // For 4bit tables the number of bytes should be halved (rounded up).
      numBytePerRow = (numBytePerRow + 1) / 2;
    } else if (param.fusedDtype_ == ElemKind::UInt8FusedQTy) {
      // For 8bit tables.
      numBytePerRow = numBytePerRow;
    } else { // (param.fusedDtype_ == ElemKind::FP16QTy)
      // For 16bit tables.
      numBytePerRow = numBytePerRow * 2;
    }

    // quantized scale/offsets (at beginning of line for TBE kernel)
    dim_t numTotalColumns = numBytePerRow + 2 * sizeof(float);
    // FP16 type do not need scale/offsets
    if (param.fusedDtype_ == ElemKind::Float16Ty) {
      numTotalColumns = numBytePerRow;
    }
    dataConstantTensor = Tensor(
        /*param.fusedDtype_*/ ElemKind::Int8QTy,
        {param.numTableEntries_, numTotalColumns}, 1.0, 0);
    Constant *dataConstant = mod->createConstant("Data", dataConstantTensor);

    const dim_t maxNumIndicesWeights =
        param.numIndicesPerBatchPad_ * param.batchSize_ * param.numTables_;

    for (size_t layer = 0; layer < param.numTBENodes_; layer++) {

      // size_t totalLengthsSum = 0;
      size_t totalNumLengths = 0;

      // Create placeholders for weights
      auto *weights =
          mod->createPlaceholder(param.dtype_, {maxNumIndicesWeights},
                                 "weights" + std::to_string(layer), false);

      if (param.dtype_ == ElemKind::Float16Ty) {
        bindings_.allocate(weights)->getHandle<float16_t>().randomize(
            -1.f, 1.f, mod->getPRNG());
      } else if (param.dtype_ == ElemKind::FloatTy) {
        bindings_.allocate(weights)->getHandle<float>().randomize(
            -1.f, 1.f, mod->getPRNG());
      }

      // Create dimOffset
      auto *dimOffset =
          mod->createPlaceholder(ElemKind::Int32ITy, {param.numTables_ + 1},
                                 "dimOffset_" + std::to_string(layer), false);
      Tensor dimOffsetVal(ElemKind::Int32ITy, {param.numTables_ + 1});
      for (int i = 0; i < param.numTables_ + 1; i++) {
        dimOffsetVal.getHandle<int32_t>().raw(i) = i * param.numElementsPerRow_;
      }

      bindings_.insert(dimOffset, std::move(dimOffsetVal));

      // Create weightOffsets
      Tensor weightsOffsetsReal(ElemKind::Int32ITy, {param.numTables_ + 1});

      auto weightsOffsets = mod->createPlaceholder(
          ElemKind::Int32ITy, {param.numTables_ + 1}, "weightsOffsets", false);
      for (int i = 0; i < param.numTables_ + 1; i++) {
        weightsOffsetsReal.getHandle<int32_t>().raw(i) =
            i * param.numElementsPerRow_;
      }
      bindings_.insert(weightsOffsets, std::move(weightsOffsetsReal));

      // Create weightsTysTensorReal
      Tensor weightsTysTensorReal(ElemKind::UInt8ITy, {param.numTables_});
      auto *weightsTysTensor =
          mod->createPlaceholder(ElemKind::UInt8ITy, {param.numTables_},
                                 "weightsTys_" + std::to_string(layer), false);
      if (param.fusedDtype_ == ElemKind::UInt4FusedFP16QTy) {
        for (int i = 0; i < param.numTables_; i++) {
          weightsTysTensorReal.getHandle<uint8_t>().raw(i) = 3; // EB_INT4 = 3
        }
      } else if (param.fusedDtype_ == ElemKind::UInt8FusedQTy) {
        for (int i = 0; i < param.numTables_; i++) {
          weightsTysTensorReal.getHandle<uint8_t>().raw(i) = 2; // EB_INT8 = 2
        }
      } else { // Float16Ty
        for (int i = 0; i < param.numTables_; i++) {
          weightsTysTensorReal.getHandle<uint8_t>().raw(i) =
              1; // EB_FLOAT16 = 1
        }
      }
      bindings_.insert(weightsTysTensor, std::move(weightsTysTensorReal));

      // Create weightsPlacement: only a placeholder
      Tensor weightsPlacementReal(ElemKind::Int32QTy, {param.numTables_});
      auto weightsPlacement = mod->createPlaceholder(
          ElemKind::Int32QTy, {param.numTables_}, "weightsPlacement", false);
      bindings_.insert(weightsPlacement, std::move(weightsPlacementReal));

      // Create lengths and offsets
      // lengths are used to populate offsets values
      auto *lengths = mod->createPlaceholder(
          ElemKind::Int32ITy, {param.numTables_ * param.batchSize_},
          "lengths" + std::to_string(layer),
          /* isTrainable */ false);
      auto *offsets = mod->createPlaceholder(
          ElemKind::Int32ITy, {param.numTables_ * param.batchSize_ + 1},
          "offsets", /* isTrainable */ false);

      auto lengthsHandle = bindings_.allocate(lengths)->getHandle<int32_t>();
      auto offsetsHandle = bindings_.allocate(offsets)->getHandle<int32_t>();

      // Generate lengths across a uniform distribution.
      lengthsHandle.randomize(param.numIndicesPerBatchMin_,
                              param.numIndicesPerBatchMax_, mod->getPRNG());
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
      // totalLengthsSum += lengthsSum;
      offsetsHandle.raw(lengthsHandle.size()) = lengthsSum;

      // Create and sort indices
      Tensor indicesReal(ElemKind::Int64ITy, {lengthsSum});
      indicesReal.getHandle<int64_t>().randomize(0, param.numTableEntries_,
                                                 mod->getPRNG());
      // Sort each segment
      if (param.isSorted_) {
        auto *indicesRealPtr = (int64_t *)indicesReal.getUnsafePtr();
        for (size_t j = 0, e = lengthsHandle.size(); j < e; j++) {
          const size_t curLength = lengthsHandle.raw(j);
          std::sort(indicesRealPtr, indicesRealPtr + curLength);
          indicesRealPtr += curLength;
        }
      }
      // Create indices
      auto *indices =
          mod->createPlaceholder(ElemKind::Int64ITy, {maxNumIndicesWeights},
                                 "indices" + std::to_string(layer),
                                 /* isTrainable */ false);

      bindings_.insert(indices, std::move(indicesReal));

      Node *R = nullptr;

      if (!param.weighted_) {
        R = fn->createIntNBitSplitEmbeddingBags(
            "tbe_" + std::to_string(layer),
            /*devWeights*/ dataConstant, /*uvmWeights*/ dataConstant,
            weightsPlacement, weightsOffsets, weightsTysTensor, dimOffset,
            /*totalDims*/ 1, indices, offsets,
            SplitEmbeddingPoolingMode::EP_SUM,
            // output type: should only be EST_FLOAT16
            SplitEmbeddingSparseType::EST_FLOAT16);
      } else {
        R = fn->createIntNBitSplitEmbeddingWeightedBags(
            "tbe_" + std::to_string(layer),
            /*devWeights*/ dataConstant, /*uvmWeights*/ dataConstant,
            weightsPlacement, weightsOffsets, weightsTysTensor, dimOffset,
            /*totalDims*/ 1, indices, offsets,
            SplitEmbeddingPoolingMode::EP_SUM,
            // output type: should only be EST_FLOAT16
            SplitEmbeddingSparseType::EST_FLOAT16, weights);
      }

      SaveNode *S = nullptr;
      if (param.addClip_) {
        auto *clp = fn->createClip("clip", R, -65504.0f, 65504.0f);
        S = fn->createSave("save", clp);
      } else {
        S = fn->createSave("save", R);
      }

      bindings_.allocate(S->getPlaceholder());
    } // layer
  }

  inline void setup() override {
    // Setup host manager
    std::string backendStr = param.backendStr_.c_str();
    std::vector<std::unique_ptr<runtime::DeviceConfig>> configs;
    auto config = glow::make_unique<runtime::DeviceConfig>(backendStr.c_str());
    if (param.devId_ != "") {
      config->parameters["DeviceID"] = param.devId_.c_str();
    }
    configs.push_back(std::move(config));
    hostManager_ = glow::make_unique<runtime::HostManager>(std::move(configs));

    std::unique_ptr<Module> mod(new Module);
    auto fn = mod->createFunction("singleNode");

    addTBENode(mod, fn, param);

    CompilationContext ctx;
    ctx.dumpFinalGraph = true;
    ctx.serializeCompiledDAG = dumpOnnx;
    EXIT_ON_ERR(hostManager_->addNetwork(std::move(mod), ctx));
  }

  inline void run() override {
    dispatchInference("singleNode", hostManager_.get(), context_,
                      param.numAsyncLaunches_,
                      /*useNewExecutionContext*/ true);
  }

  void teardown() override {}

  double gbytes() const {
    return 2.0 * param.numIndicesPerBatchPad_ * param.numElementsPerRow_ *
           param.numTables_ / 1e9;
  }

}; // benchmark

inline TBEParam parseArgs(int argc, char *argv[]) {
  TBEParam param;
  // param.batchSize = getIntFromStr(argv[1]);
  ASSIGN_VALUE_OR_FATAL(param.batchSize_, getIntFromStr(argv[1]));

  llvm::StringRef numIndicesPerBatchStr(argv[2]);
  auto split = numIndicesPerBatchStr.split(':');
  if (split.second == "") {
    ASSIGN_VALUE_OR_FATAL(param.numIndicesPerBatchMin_, getIntFromStr(argv[2]));
    param.numIndicesPerBatchMax_ = param.numIndicesPerBatchMin_;
  } else {
    ASSIGN_VALUE_OR_FATAL(param.numIndicesPerBatchMin_,
                          getIntFromStr(split.first));
    ASSIGN_VALUE_OR_FATAL(param.numIndicesPerBatchMax_,
                          getIntFromStr(split.second));
    CHECK_LE(param.numIndicesPerBatchMin_, param.numIndicesPerBatchMax_);
  }
  ASSIGN_VALUE_OR_FATAL(param.numIndicesPerBatchPad_, getIntFromStr(argv[3]));
  CHECK_LE(param.numIndicesPerBatchMax_, param.numIndicesPerBatchPad_);
  ASSIGN_VALUE_OR_FATAL(param.numTableEntries_, getIntFromStr(argv[4]));
  ASSIGN_VALUE_OR_FATAL(param.numTables_, getIntFromStr(argv[5]));
  ASSIGN_VALUE_OR_FATAL(param.numElementsPerRow_, getIntFromStr(argv[6]));
  ASSIGN_VALUE_OR_FATAL(param.numReps_, getIntFromStr(argv[7]));
  ASSIGN_VALUE_OR_FATAL(param.numAsyncLaunches_, getIntFromStr(argv[8]));
  ASSIGN_VALUE_OR_FATAL(param.numTBENodes_, getIntFromStr(argv[9]));
  printf("batchSize %zu\n", (size_t)param.batchSize_);
  printf("numIndicesPerBatchMin %zu\n", (size_t)param.numIndicesPerBatchMin_);
  printf("numIndicesPerBatchMax %zu\n", (size_t)param.numIndicesPerBatchMax_);
  printf("numIndicesPerBatchPad %zu\n", (size_t)param.numIndicesPerBatchPad_);
  printf("numTableEntries %zu\n", (size_t)param.numTableEntries_);
  printf("numTables %zu\n", (size_t)param.numTables_);
  printf("numElementsPerRow %zu\n", (size_t)param.numElementsPerRow_);
  printf("numReps %zu\n", (size_t)param.numReps_);
  printf("numAsyncLaunches %zu\n", (size_t)param.numAsyncLaunches_);
  printf("numTBENodes %zu\n", (size_t)param.numTBENodes_);
  printf("tbeWeighted %s\n", argv[10]);
  if (std::string(argv[10]) == "Unweighted") {
    param.weighted_ = false;
  } else if (std::string(argv[10]) == "Weighted") {
    param.weighted_ = true;
  } else {
    llvm_unreachable(strFormat("Invalid Weighted").c_str());
  }
  printf("sortedStr %s\n", argv[11]);
  if (std::string(argv[11]) == "Sorted") {
    param.isSorted_ = true;
  } else if (std::string(argv[11]) == "Unsorted") {
    param.isSorted_ = false;
  } else {
    llvm_unreachable("Invalid sortedStr");
  }
  printf("backendStr %s\n", argv[12]);
  param.backendStr_ = std::string(argv[12]);
  printf("dtypeStr %s\n", argv[13]);
  if (std::string(argv[13]) == "Float16") {
    param.dtype_ = ElemKind::Float16Ty;
  } else if (std::string(argv[13]) == "Float32") {
    param.dtype_ = ElemKind::FloatTy;
  } else {
    llvm_unreachable("Invalid dtype");
  }
  printf("fusedDtypeStr %s\n", argv[14]);
  if (std::string(argv[14]) == "Int4") {
    param.fusedDtype_ = ElemKind::UInt4FusedFP16QTy;
  } else if (std::string(argv[14]) == "Int8") {
    param.fusedDtype_ = ElemKind::UInt8FusedQTy;
  } else if (std::string(argv[14]) == "FP16") {
    param.fusedDtype_ = ElemKind::Float16Ty;
  } else {
    llvm_unreachable("Invalid fusedDtype");
  }
  printf("addClipStr %s\n", argv[15]);
  if (std::string(argv[15]) == "True") {
    param.addClip_ = true;
  } else if (std::string(argv[15]) == "False") {
    param.addClip_ = false;
  } else {
    llvm_unreachable("Invalid addClipStr");
  }
  // param.convertFusedToFP32 = false;
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
  printf("TableBatchedEmbedding Microbenchmark\n");
  printf("Usage: TBEBench batchSize(Int) "
         "[numIndicesPerBatch(Int) | "
         "numIndicesPerBatchMin(Int):numIndicesPerBatchMax(Int)] "
         "numIndicesPerBatchPad(Int) numTableEntries(Int) "
         "numTBETables(Int) "
         "numElementsPerRow(int) numReps(Int) "
         "numAsyncLaunches(Int) numTBENodes(Int) "
         "tbeWeightedStr(\"Weighted\"|\"Unweighted\" "
         "sortedStr(\"Sorted\"|\"Unsorted\") backendStr(String) "
         "dtypeStr(\"Float16\"|\"Float32\") "
         "fusedDtypeStr(\"Int4\"|\"Int8\"|\"FP16\") "
         "addClipStr(\"True\"|\"False\")\nQuantized only options: "
         "quantizationDtypeStr(\"Int8\"|\"Int4\") "
         "useFP16AccumulationStr(\"True\"|\"False\") \n"
         "Optional: dev_id(Int)\n");
  printf("\n");
  printf("Standard Glow command-line options may be passed via the GLOW_OPTS "
         "environment variable\n");
  benchParseGlowOpts(argc, argv);

  std::vector<TBEParam> params;
  TBEParam param = parseArgs(argc, argv);
  params.push_back(param);

  std::string runHeader;
  std::string runPrefix;

  for (auto param : params) {
    runHeader = std::string(strFormat(
        "_,benchName,_,batchSize,numIndicesPerBatchMin:numIndicesPerBatchMax,"
        "numIndicesPerBatchPad,numTableEntries,numElementsPerRow,numReps,"
        "numAsyncLaunches,numTBENodes,weighted,backendStr,dtype,fuseDtype"));
    runPrefix = std::string(strFormat(
        "SW,%zu,%zu:%zu,%zu,%zu,%zu,%zu,%zu,%zu,%s,%s,%s,%s",
        (size_t)param.batchSize_, (size_t)param.numIndicesPerBatchMin_,
        (size_t)param.numIndicesPerBatchMax_,
        (size_t)param.numIndicesPerBatchPad_, (size_t)param.numTableEntries_,
        (size_t)param.numElementsPerRow_, (size_t)param.numReps_,
        (size_t)param.numAsyncLaunches_, (size_t)param.numTBENodes_, argv[9],
        argv[10], argv[11], argv[12]));

    TBEBench b(param);

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
    printf("%s,medianRuntime,minRuntime,medianGflopPerSec,maxGflopPerSec\n",
           runHeader.c_str());
    printf("BenchSummary,%s,%f,%f,%f,%f\n", runPrefix.c_str(), medianRuntime,
           minRuntime, b.gbytes() / medianRuntime, b.gbytes() / minRuntime);
  }
}
