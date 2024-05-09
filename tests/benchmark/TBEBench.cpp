#include "Bench.h"
#include "glow/Base/DimType.h"
#include "glow/Base/Tensor.h"
#include "glow/Base/Type.h"
#include "glow/Graph/Graph.h"
#include "glow/Graph/Nodes.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <future>
#include <random>
#include <string>
#include <torch/torch.h>

#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Optimizer/GraphOptimizer/GraphOptimizer.h"

#include "include/glow/Base/Type.h"
#include "tests/unittests/BackendTestUtils.h"

using namespace glow;

#define DATA_FILE 19
#define DEVICE_ID (DATA_FILE + 1)

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
  std::string data_file;
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
  ElemKind ttype_;
  ElemKind otype_;
  ElemKind itype_;
};

class TBEBench : public Benchmark {
  std::unique_ptr<runtime::HostManager> hostManager_;
  ExecutionContext context_;
  TBEParam param;
  PlaceholderBindings &bindings_;

  bool extern_data;
  torch::Tensor idx_tensor;
  torch::Tensor off_tensor;
  dim_t numTables;
  dim_t batchSize;

public:
  explicit TBEBench(TBEParam param_)
      : param(param_), bindings_(*context_.getPlaceholderBindings()) {
    if (param.data_file != "") {
      extern_data = true;
      std::cout << "Loading data from " << param.data_file << std::endl;

      std::ifstream fin(param.data_file);
      std::vector<char> data;

      fin >> std::noskipws;
      data.insert(data.begin(), std::istreambuf_iterator<char>(fin),
                  std::istreambuf_iterator<char>());
      // For the format of the data file, please refer to
      // https://fburl.com/code/y8b9yyj0
      torch::IValue ivalue = torch::pickle_load(data);
      auto tensors = ivalue.toList();
      idx_tensor = tensors.get(0).toTensor();
      off_tensor = tensors.get(1).toTensor();
      torch::Tensor len_tensor = tensors.get(2).toTensor();

      numTables = len_tensor.size(0);
      batchSize = len_tensor.size(1);

      std::cout << "Number of tables = " << numTables
                << ", Batch size = " << batchSize << std::endl;
    } else {
      extern_data = false;
      numTables = param.numTables_;
      batchSize = param.batchSize_;
    }
  }

  inline void addTBENode(const std::unique_ptr<Module> &mod, Function *fn,
                         const TBEParam &param) {
    Tensor dataConstantTensor;
    int64_t numBytePerRow = param.numElementsPerRow_;

    if (param.fusedDtype_ == ElemKind::UInt4FusedFP16QTy) {
      // For 4bit tables the number of bytes should be halved (rounded up).
      numBytePerRow = (numBytePerRow + 1) / 2;
    } else if (param.fusedDtype_ == ElemKind::UInt8FusedQTy) {
      // For 8bit tables numBytePerRow is already correct
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
        {param.numTableEntries_ * numTotalColumns * param.numTables_}, 1.0, 0);
    Constant *dataConstant = mod->createConstant("Data", dataConstantTensor);

    const dim_t maxNumIndicesWeights =
        param.numIndicesPerBatchPad_ * batchSize * numTables;

    for (size_t layer = 0; layer < param.numTBENodes_; layer++) {

      // size_t totalLengthsSum = 0;

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
          mod->createPlaceholder(ElemKind::Int32ITy, {numTables + 1},
                                 "dimOffset_" + std::to_string(layer), false);
      Tensor dimOffsetVal(ElemKind::Int32ITy, {numTables + 1});
      for (int i = 0; i < numTables + 1; i++) {
        dimOffsetVal.getHandle<int32_t>().raw(i) = i * param.numElementsPerRow_;
      }

      bindings_.insert(dimOffset, std::move(dimOffsetVal));

      // Create weightOffsets
      Placeholder *weightsOffsets;
      if (param.ttype_ == ElemKind::Int32ITy) {
        Tensor weightsOffsetsReal(ElemKind::Int32ITy, {numTables + 1});

        weightsOffsets = mod->createPlaceholder(
            ElemKind::Int32ITy, {numTables + 1}, "weightsOffsets", false);
        for (int i = 0; i < numTables + 1; i++) {
          weightsOffsetsReal.getHandle<int32_t>().raw(i) =
              i * param.numTableEntries_ * numTotalColumns;
        }
        bindings_.insert(weightsOffsets, std::move(weightsOffsetsReal));
      } else {
        Tensor weightsOffsetsReal(ElemKind::Int64ITy, {numTables + 1});

        weightsOffsets = mod->createPlaceholder(
            ElemKind::Int64ITy, {numTables + 1}, "weightsOffsets", false);
        for (int i = 0; i < numTables + 1; i++) {
          weightsOffsetsReal.getHandle<int64_t>().raw(i) =
              i * param.numTableEntries_ * numTotalColumns;
        }
        bindings_.insert(weightsOffsets, std::move(weightsOffsetsReal));
      }

      // Create weightsTysTensorReal
      Tensor weightsTysTensorReal(ElemKind::UInt8ITy, {numTables});
      auto *weightsTysTensor =
          mod->createPlaceholder(ElemKind::UInt8ITy, {numTables},
                                 "weightsTys_" + std::to_string(layer), false);
      if (param.fusedDtype_ == ElemKind::UInt4FusedFP16QTy) {
        for (int i = 0; i < numTables; i++) {
          weightsTysTensorReal.getHandle<uint8_t>().raw(i) = 3; // EB_INT4 = 3
        }
      } else if (param.fusedDtype_ == ElemKind::UInt8FusedQTy) {
        for (int i = 0; i < numTables; i++) {
          weightsTysTensorReal.getHandle<uint8_t>().raw(i) = 2; // EB_INT8 = 2
        }
      } else { // Float16Ty
        for (int i = 0; i < numTables; i++) {
          weightsTysTensorReal.getHandle<uint8_t>().raw(i) =
              1; // EB_FLOAT16 = 1
        }
      }
      bindings_.insert(weightsTysTensor, std::move(weightsTysTensorReal));

      // Create weightsPlacement: only a placeholder
      Tensor weightsPlacementReal(ElemKind::Int32QTy, {numTables});
      auto weightsPlacement = mod->createPlaceholder(
          ElemKind::Int32QTy, {numTables}, "weightsPlacement", false);
      bindings_.insert(weightsPlacement, std::move(weightsPlacementReal));

      // Create lengths and offsets
      // lengths are used to populate offsets values
      Placeholder *offsets;
      Placeholder *lengths =
          mod->createPlaceholder(ElemKind::Int64ITy, {numTables * batchSize},
                                 "lengths" + std::to_string(layer), false);
      auto lengthsHandle = bindings_.allocate(lengths)->getHandle<int64_t>();
      dim_t lengthsSum = 0;
      if (param.otype_ == ElemKind::Int32ITy) {

        offsets = mod->createPlaceholder(
            ElemKind::Int32ITy, {numTables * batchSize + 1}, "offsets", false);
        auto offsetsHandle = bindings_.allocate(offsets)->getHandle<int32_t>();

        if (extern_data) {
          int32_t cur, pre, base = 0;

          for (size_t j = 0, e = offsetsHandle.size(); j < e; j++) {
            cur = off_tensor[j].item<int>();
            if (j % batchSize == 0) {
              base = cur;
            }
            offsetsHandle.raw(j) = cur - base;
            if (j > 0) {
              lengthsHandle.raw(j) = cur - pre;
            }
            pre = cur;
          }

          lengthsSum = pre;
        } else {
          // Generate lengths across a uniform distribution.
          lengthsHandle.randomize(param.numIndicesPerBatchMin_,
                                  param.numIndicesPerBatchMax_, mod->getPRNG());
          for (size_t j = 0, e = lengthsHandle.size(); j < e; j++) {
            auto &nextLength = lengthsHandle.raw(j);
            if (lengthsSum == maxNumIndicesWeights) {
              // If we have maxed out the maximum allowed indices then zero out
              // the rest of the lengths.
              nextLength = 0;
              continue;
            } else if (lengthsSum + nextLength > maxNumIndicesWeights) {
              // If the next length will equal or overflow the maximum allowed
              // indices then fill it up totally.
              nextLength = maxNumIndicesWeights - lengthsSum;
            }
            offsetsHandle.raw(j) = lengthsSum;
            lengthsSum += nextLength;
          }
          // totalLengthsSum += lengthsSum;
          offsetsHandle.raw(lengthsHandle.size()) = lengthsSum;
        }
      } else {
        offsets = mod->createPlaceholder(
            ElemKind::Int64ITy, {numTables * batchSize + 1}, "offsets", false);
        auto offsetsHandle = bindings_.allocate(offsets)->getHandle<int64_t>();

        if (extern_data) {
          int32_t cur, pre, base = 0;

          for (size_t j = 0, e = offsetsHandle.size(); j < e; j++) {
            cur = off_tensor[j].item<int>();
            if (j % batchSize == 0) {
              base = cur;
            }
            offsetsHandle.raw(j) = cur - base;
            if (j > 0) {
              lengthsHandle.raw(j) = cur - pre;
            }
            pre = cur;
          }

          lengthsSum = pre;
        } else {
          // Generate lengths across a uniform distribution.
          lengthsHandle.randomize(param.numIndicesPerBatchMin_,
                                  param.numIndicesPerBatchMax_, mod->getPRNG());
          for (size_t j = 0, e = lengthsHandle.size(); j < e; j++) {
            auto &nextLength = lengthsHandle.raw(j);
            if (lengthsSum == maxNumIndicesWeights) {
              // If we have maxed out the maximum allowed indices then zero out
              // the rest of the lengths.
              nextLength = 0;
              continue;
            } else if (lengthsSum + nextLength > maxNumIndicesWeights) {
              // If the next length will equal or overflow the maximum allowed
              // indices then fill it up totally.
              nextLength = maxNumIndicesWeights - lengthsSum;
            }
            offsetsHandle.raw(j) = lengthsSum;
            lengthsSum += nextLength;
          }
          // totalLengthsSum += lengthsSum;
          offsetsHandle.raw(lengthsHandle.size()) = lengthsSum;
        }
      }

      // Create and sort indices
      Placeholder *indices;
      if (param.itype_ == ElemKind::Int64ITy) {
        Tensor indicesReal(ElemKind::Int64ITy, {lengthsSum});

        if (extern_data) {
          for (size_t j = 0; j < lengthsSum; j++) {
            indicesReal.getHandle<int64_t>().raw(j) = idx_tensor[j].item<int>();
          }
        } else {
          indicesReal.getHandle<int64_t>().randomize(0, param.numTableEntries_,
                                                     mod->getPRNG());
        }
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
        indices =
            mod->createPlaceholder(ElemKind::Int64ITy, {maxNumIndicesWeights},
                                   "indices" + std::to_string(layer),
                                   /* isTrainable */ false);
        bindings_.insert(indices, std::move(indicesReal));
      } else {
        Tensor indicesReal(ElemKind::Int32ITy, {lengthsSum});

        if (extern_data) {
          for (size_t j = 0; j < lengthsSum; j++) {
            indicesReal.getHandle<int32_t>().raw(j) = idx_tensor[j].item<int>();
          }
        } else {
          indicesReal.getHandle<int32_t>().randomize(0, param.numTableEntries_,
                                                     mod->getPRNG());
        }
        // Sort each segment
        if (param.isSorted_) {
          auto *indicesRealPtr = (int32_t *)indicesReal.getUnsafePtr();
          for (size_t j = 0, e = lengthsHandle.size(); j < e; j++) {
            const size_t curLength = lengthsHandle.raw(j);
            std::sort(indicesRealPtr, indicesRealPtr + curLength);
            indicesRealPtr += curLength;
          }
        }

        // Create indices
        indices =
            mod->createPlaceholder(ElemKind::Int32ITy, {maxNumIndicesWeights},
                                   "indices" + std::to_string(layer),
                                   /* isTrainable */ false);
        bindings_.insert(indices, std::move(indicesReal));
      }

      Node *R = nullptr;

      if (!param.weighted_) {
        R = fn->createIntNBitSplitEmbeddingBags(
            "tbe_" + std::to_string(layer),
            /*devWeights*/ dataConstant, /*uvmWeights*/ dataConstant,
            weightsPlacement, weightsOffsets, weightsTysTensor, dimOffset,
            /*totalDims*/ numTables * param.numElementsPerRow_, indices,
            offsets, SplitEmbeddingPoolingMode::EP_SUM,
            // output type: should only be EST_FLOAT16
            SplitEmbeddingSparseType::EST_FLOAT16);
      } else {
        R = fn->createIntNBitSplitEmbeddingWeightedBags(
            "tbe_" + std::to_string(layer),
            /*devWeights*/ dataConstant, /*uvmWeights*/ dataConstant,
            weightsPlacement, weightsOffsets, weightsTysTensor, dimOffset,
            /*totalDims*/ numTables * param.numElementsPerRow_, indices,
            offsets, SplitEmbeddingPoolingMode::EP_SUM,
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
           this->numTables / 1e9;
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

  printf("tableOffsetDtypeStr %s\n", argv[15]);
  if (std::string(argv[15]) == "Int32") {
    param.ttype_ = ElemKind::Int32ITy;
  } else if (std::string(argv[15]) == "Int64") {
    param.ttype_ = ElemKind::Int64ITy;
  } else {
    llvm_unreachable("Invalid tableOffsetDtype");
  }

  printf("offsetDtypeStr %s\n", argv[16]);
  if (std::string(argv[16]) == "Int32") {
    param.otype_ = ElemKind::Int32ITy;
  } else if (std::string(argv[16]) == "Int64") {
    param.otype_ = ElemKind::Int64ITy;
  } else {
    llvm_unreachable("Invalid offsetDtype");
  }

  printf("indexDtypeStr %s\n", argv[17]);
  if (std::string(argv[17]) == "Int32") {
    param.itype_ = ElemKind::Int32ITy;
  } else if (std::string(argv[17]) == "Int64") {
    param.itype_ = ElemKind::Int64ITy;
  } else {
    llvm_unreachable("Invalid indexDtype");
  }

  printf("addClipStr %s\n", argv[18]);
  if (std::string(argv[18]) == "True") {
    param.addClip_ = true;
  } else if (std::string(argv[18]) == "False") {
    param.addClip_ = false;
  } else {
    llvm_unreachable("Invalid addClipStr");
  }
  // param.convertFusedToFP32 = false;

  if (argc > DATA_FILE) {
    printf("data_file %s\n", argv[DATA_FILE]);
    param.data_file = std::string(argv[DATA_FILE]);
  } else {
    param.data_file = std::string("");
  }

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
         "tableOffsetDtypeStr(\"Int32\"|\"Int64\") "
         "offsetDtypeStr(\"Int32\"|\"Int64\") "
         "indexDtypeStr(\"Int32\"|\"Int64\") "
         "addClipStr(\"True\"|\"False\")\nQuantized only options: "
         "quantizationDtypeStr(\"Int8\"|\"Int4\") "
         "useFP16AccumulationStr(\"True\"|\"False\") \n"
         "Optional: dev_id(Int)\n"
         "Optional: data_file(Int)\n");
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
        "numIndicesPerBatchPad,numTableEntries,numTables,numElementsPerRow,"
        "numReps,numAsyncLaunches,numTBENodes,weighted,backendStr,dtype,"
        "fuseDtype,tableOffsetDtype,offsetDtype,indexDtype"));
    runPrefix = std::string(strFormat(
        "TBEBench,SW,%zu,%zu:%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%s,%s,%s,%s,%s,%s,"
        "%s",
        (size_t)param.batchSize_, (size_t)param.numIndicesPerBatchMin_,
        (size_t)param.numIndicesPerBatchMax_,
        (size_t)param.numIndicesPerBatchPad_, (size_t)param.numTableEntries_,
        (size_t)param.numTables_, (size_t)param.numElementsPerRow_,
        (size_t)param.numReps_, (size_t)param.numAsyncLaunches_,
        (size_t)param.numTBENodes_, argv[10], argv[12], argv[13], argv[14],
        argv[15], argv[16], argv[17]));

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
