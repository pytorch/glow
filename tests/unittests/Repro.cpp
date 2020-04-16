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

#include "BackendTestUtils.h"
#include "folly/executors/CPUThreadPoolExecutor.h"
#include "glow/Backend/Backend.h"
#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Exporter/ONNXModelWriter.h"
#include "glow/Graph/Graph.h"
#include "glow/Importer/ONNXModelLoader.h"
#include "glow/Runtime/DeferredWeightLoader.h"
#include "glow/Support/ZipUtils.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Signals.h"

#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include <glog/logging.h>

#include "folly/stats/Histogram.h"

#include <fstream>
#include <iostream>
#include <string>

using namespace glow;

namespace {
llvm::cl::OptionCategory reproTestCat("Repro Category");
llvm::cl::opt<std::string> modelPathOpt("model", llvm::cl::desc("Input models"),
                                        llvm::cl::value_desc("modelPath"),
                                        llvm::cl::Required,
                                        llvm::cl::cat(reproTestCat));
llvm::cl::opt<std::string> deferredWeightsPathOpt(
    "deferred_weights", llvm::cl::desc("Path to the deferred weights file"),
    llvm::cl::Optional, llvm::cl::init(""), llvm::cl::cat(reproTestCat));
llvm::cl::list<std::string> inputsOpt("inputs", llvm::cl::desc("Inputs"),
                                      llvm::cl::value_desc("Inputs"),
                                      llvm::cl::Optional, llvm::cl::ZeroOrMore,
                                      llvm::cl::cat(reproTestCat));
llvm::cl::list<std::string> outputsOpt("outputs", llvm::cl::desc("Ouptuts"),
                                       llvm::cl::value_desc("Ouptuts"),
                                       llvm::cl::Optional, llvm::cl::ZeroOrMore,
                                       llvm::cl::cat(reproTestCat));
llvm::cl::opt<std::string>
    inputPatternOpt("input_pattern",
                    llvm::cl::desc("Input file pattern. in_{}.onnx"),
                    llvm::cl::init(""), llvm::cl::cat(reproTestCat));
llvm::cl::opt<std::string>
    outputPatternOpt("output_pattern",
                     llvm::cl::desc("Output file pattern. out_{}.onnx"),
                     llvm::cl::init(""), llvm::cl::cat(reproTestCat));
llvm::cl::opt<unsigned> seqStartOpt(
    "seq_start", llvm::cl::desc("Start index of input/output files"),
    llvm::cl::Optional, llvm::cl::init(0), llvm::cl::cat(reproTestCat));
llvm::cl::opt<unsigned> seqLenOpt(
    "seq_len", llvm::cl::desc("Lengths of the input/output file seqquence."),
    llvm::cl::Optional, llvm::cl::init(0), llvm::cl::cat(reproTestCat));

llvm::cl::opt<std::string> ExecutionBackend(
    "backend", llvm::cl::desc("Backend to use, e.g., Interpreter, CPU, NNPI:"),
    llvm::cl::init("NNPI"), llvm::cl::cat(reproTestCat));

llvm::cl::opt<unsigned> concurrentRequestsOpt(
    "concurrent_count", llvm::cl::desc("Number of concurrent requests."),
    llvm::cl::Optional, llvm::cl::init(1), llvm::cl::cat(reproTestCat));

llvm::cl::opt<unsigned> numDevicesOpt(
    "glow_num_devices", llvm::cl::desc("Number of devices for Glow backend"),
    llvm::cl::Optional, llvm::cl::init(1), llvm::cl::cat(reproTestCat));

llvm::cl::opt<float> deviceMemoryOpt(
    "glow_device_memory",
    llvm::cl::desc("Size of memory for a certain Glow backend device"),
    llvm::cl::Optional, llvm::cl::init(256 * 1024.0 * 1024.0 * 1024.0),
    llvm::cl::cat(reproTestCat));

llvm::cl::opt<float> thresholdOpt(
    "threshold", llvm::cl::desc("theshold for tensor numeric comparison"),
    llvm::cl::Optional, llvm::cl::init(1e-5), llvm::cl::cat(reproTestCat));

llvm::cl::opt<bool> glowDumpGraphOpt(
    "glow_dump_graph",
    llvm::cl::desc("Dump the glow Graph into files before compilation"),
    llvm::cl::Optional, llvm::cl::init(false), llvm::cl::cat(reproTestCat));

llvm::cl::opt<bool> glowDumpGraphAfterLoadOpt(
    "glow_dump_graph_after_load",
    llvm::cl::desc(
        "Dump the glow Graph into files immediately after loading from ONNX"),
    llvm::cl::Optional, llvm::cl::init(false), llvm::cl::cat(reproTestCat));

llvm::cl::opt<bool>
    globalFp16Opt("glow_global_fp16",
                  llvm::cl::desc("Enable fp16 lowering for all ops on the net"),
                  llvm::cl::Optional, llvm::cl::init(true),
                  llvm::cl::cat(reproTestCat));

llvm::cl::opt<bool> globalFp16ConstantsOpt(
    "glow_global_fp16_constants",
    llvm::cl::desc("Enable fp16 conversion for Constants"), llvm::cl::Optional,
    llvm::cl::init(true), llvm::cl::cat(reproTestCat));

llvm::cl::opt<bool> globalFp16PlaceholdersOpt(
    "glow_global_fp16_placeholders",
    llvm::cl::desc("Enable fp16 conversion for Placeholders"),
    llvm::cl::Optional, llvm::cl::init(true), llvm::cl::cat(reproTestCat));

llvm::cl::opt<bool> sliceConcatFp32Opt(
    "glow_slice_concat_fp32",
    llvm::cl::desc("Don't convert slice and concat ops's precision when "
                   "--glow_global_fp16 is used."),
    llvm::cl::Optional, llvm::cl::init(false), llvm::cl::cat(reproTestCat));

llvm::cl::opt<bool> dumpOutputsOpt("dump_outputs",
                                   llvm::cl::desc("Dump output tensors"),
                                   llvm::cl::Optional, llvm::cl::init(true),
                                   llvm::cl::cat(reproTestCat));

llvm::cl::opt<bool> fuseScaleOffsetFp16Opt(
    "glow_global_fused_scale_offset_fp16",
    llvm::cl::desc(
        "Enable fp16 lowering for all op inputs using fused scale/offset"),
    llvm::cl::Optional, llvm::cl::init(true), llvm::cl::cat(reproTestCat));

llvm::cl::opt<bool>
    ClipFp16Opt("glow_clip_fp16",
                llvm::cl::desc("Force glow to clip fp16 values to min/max"),
                llvm::cl::Optional, llvm::cl::init(true),
                llvm::cl::cat(reproTestCat));

llvm::cl::opt<bool> ClipFp16SkipInputsOpt(
    "glow_clip_fp16_skip_inputs",
    llvm::cl::desc("Force glow to skip clipping fp16 Node inputs to min/max"),
    llvm::cl::Optional, llvm::cl::init(true), llvm::cl::cat(reproTestCat));

llvm::cl::opt<bool>
    forceFP16AccumSLSOpt("glow_global_force_sls_fp16_accum",
                         llvm::cl::desc("Force FP16 accumulation for SLS ops"),
                         llvm::cl::Optional, llvm::cl::init(true),
                         llvm::cl::cat(reproTestCat));

llvm::cl::opt<bool> enablePartialTensor("glow_enable_partial_tensor",
                                        llvm::cl::desc("Enable partial tensor"),
                                        llvm::cl::Optional,
                                        llvm::cl::init(true),
                                        llvm::cl::cat(reproTestCat));

llvm::cl::opt<unsigned> itersOpt(
    "iters", llvm::cl::desc("Number of times to loop over provided input."),
    llvm::cl::Optional, llvm::cl::init(1), llvm::cl::cat(reproTestCat));

llvm::cl::opt<bool> useSparseNNPartitioningScheme(
    "glow_use_sparsenn_partitioning_scheme",
    llvm::cl::desc("Enable SparseNN partitioning scheme"), llvm::cl::Optional,
    llvm::cl::init(false), llvm::cl::cat(reproTestCat));

llvm::cl::opt<bool> sparseNNPartitioningAddSLSConcats(
    "glow_sparsenn_partitioning_add_sls_concats",
    llvm::cl::desc("Add extra concats inside of SLS partitions for more "
                   "efficient inter-partitition transfers"),
    llvm::cl::Optional, llvm::cl::init(false), llvm::cl::cat(reproTestCat));

llvm::cl::opt<int32_t> sparseNNPartitioningSchemeNumCards(
    "glow_snn_partitioning_num_cards",
    llvm::cl::desc("Num cards used in SparseNN partitioning scheme"),
    llvm::cl::Optional, llvm::cl::init(1), llvm::cl::cat(reproTestCat));

llvm::cl::opt<int32_t> sparseNNPartitioningSchemeKBytesPerCard(
    "glow_snn_partitioning_kbytes_per_card",
    llvm::cl::desc("Num kbytes per card in SparseNN partitioning scheme"),
    llvm::cl::Optional, llvm::cl::init(5000000), llvm::cl::cat(reproTestCat));

llvm::cl::opt<int32_t> sparseNNPartitioningSchemeNumCoresSLS(
    "glow_snn_partitioning_num_cores_sls",
    llvm::cl::desc("Num cores used for SLS in SparseNN partitioning scheme"),
    llvm::cl::Optional, llvm::cl::init(6), llvm::cl::cat(reproTestCat));

llvm::cl::opt<int32_t> sparseNNPartitioningSchemeNumCoresOther(
    "glow_snn_partitioning_num_cores_other",
    llvm::cl::desc(
        "Num cores used for non-SLS in SparseNN partitioning scheme"),
    llvm::cl::Optional, llvm::cl::init(6), llvm::cl::cat(reproTestCat));

llvm::cl::opt<int32_t> glowNNPINumParallelChunks(
    "glow_nnpi_num_parallel_chunks",
    llvm::cl::desc("Number of parallel splits to apply to certain ops"),
    llvm::cl::Optional, llvm::cl::init(1), llvm::cl::cat(reproTestCat));

llvm::cl::opt<bool> glowDumpTrace("glow_dump_debug_traces",
                                  llvm::cl::desc("Dump glow trace"),
                                  llvm::cl::Optional, llvm::cl::init(false),
                                  llvm::cl::cat(reproTestCat));

llvm::cl::opt<bool> skipCorrectnessCheck(
    "skip_correctness_check", llvm::cl::desc("Skip correctness check"),
    llvm::cl::Optional, llvm::cl::init(false), llvm::cl::cat(reproTestCat));

llvm::cl::opt<std::string>
    glowDumpTraceFile("glow_dump_debug_traces_file",
                      llvm::cl::desc("Dump glow trace file"),
                      llvm::cl::Optional, llvm::cl::init(std::string("")),
                      llvm::cl::cat(reproTestCat));

llvm::cl::opt<unsigned> glowMaxActiveRequests(
    "glow_max_active_requests",
    llvm::cl::desc(
        "Number of active requests before host manager start queuing"),
    llvm::cl::Optional, llvm::cl::init(48), llvm::cl::cat(reproTestCat));

llvm::cl::opt<unsigned> glowMaxQueueSize(
    "glow_max_queue_size",
    llvm::cl::desc(
        "Max number of pending requeusts in glow's host manager queue before "
        "rejecting new request"),
    llvm::cl::Optional, llvm::cl::init(100), llvm::cl::cat(reproTestCat));

llvm::cl::opt<unsigned> glowExecutorThreads(
    "glow_executor_threads",
    llvm::cl::desc("Number of executor threads for host manager"),
    llvm::cl::Optional, llvm::cl::init(10), llvm::cl::cat(reproTestCat));

llvm::cl::opt<bool> glowSaturateHost(
    "glow_saturate_host",
    llvm::cl::desc("Duplicate netowrk on all available devices"),
    llvm::cl::Optional, llvm::cl::init(false), llvm::cl::cat(reproTestCat));

llvm::cl::opt<int32_t>
    topKCompare("topk_compare",
                llvm::cl::desc("Compare the topk results against reference"),
                llvm::cl::Optional, llvm::cl::init(0),
                llvm::cl::cat(reproTestCat));

llvm::cl::opt<float> top1Threshold(
    "top1_threshold",
    llvm::cl::desc(
        "Percentage of top1 matches to reference that must be achieved"),
    llvm::cl::Optional, llvm::cl::init(0.0), llvm::cl::cat(reproTestCat));

llvm::cl::opt<bool> logTopKResultsPerExample(
    "log_topk_results_per_example",
    llvm::cl::desc("Whether to log topk results vs reference for each example"),
    llvm::cl::Optional, llvm::cl::init(false), llvm::cl::cat(reproTestCat));

llvm::cl::opt<bool> cosineSimilarityStats(
    "cosine_similarity_stats",
    llvm::cl::desc("Whether to compute cosine similarity stats"),
    llvm::cl::Optional, llvm::cl::init(false), llvm::cl::cat(reproTestCat));

llvm::cl::opt<float> cosineSimilarityThreshold(
    "p50_cosine_similarity_threshold",
    llvm::cl::desc(
        "Percentage of top1 matches to reference that must be achieved"),
    llvm::cl::Optional, llvm::cl::init(0.0), llvm::cl::cat(reproTestCat));

llvm::cl::opt<bool> onnxLoaderZipMode(
    "zip_mode", llvm::cl::desc("zipMode to use with OnnxModelLoader"),
    llvm::cl::Optional, llvm::cl::init(true), llvm::cl::cat(reproTestCat));

void parseCommandLine(int argc, char **argv) {
  llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);
  llvm::cl::ParseCommandLineOptions(
      argc, argv,
      " The Glow compiler\n\n"
      "Glow is a compiler for neural network accelerators.\n");

  if (top1Threshold > 0.0 && topKCompare == 0) {
    topKCompare = 1;
  }

  if (cosineSimilarityThreshold > 0.0) {
    cosineSimilarityStats = true;
  }
}

struct InferenceResult {
  Error error = Error::empty();
  std::unique_ptr<ExecutionContext> ctx;
  int index = 0;
  std::chrono::time_point<std::chrono::steady_clock> endTime;
};

class ZipFileBackedDeferredBlobLoader
    : public ::glow::runtime::DeferredWeightLoader {
public:
  explicit ZipFileBackedDeferredBlobLoader(const std::string &path) {
    zip_ = ::glow::make_unique<::glow::ZipReader>(path);
    CHECK(zip_);
    auto numWeightsStr = zip_->getRecord("weights");
    weightsToLoad_ = atoi(numWeightsStr.c_str());
    i_ = 0;
  }

  ::glow::Error loadNextWeight() override {
    if (weightsToLoad_ == i_) {
      llvm::outs() << "All deferred weights are loaded\n";
      currentBlobName_ = "";
      currentTensor_.reset();
      zip_.reset(nullptr);
      return ::glow::Error::success();
    }

    std::stringstream ss;
    ss << "weight_" << i_++;
    largeBuffer_ = zip_->getRecord(ss.str());
    ::ONNX_NAMESPACE::TensorProto t;
    t.ParseFromString(largeBuffer_);

    currentBlobName_ = t.name();
    auto tyIdx = typeInfo_.find(currentBlobName_);
    if (tyIdx == typeInfo_.end()) {
      return ::MAKE_ERR(
          ::glow::ErrorValue::ErrorCode::RUNTIME_ERROR,
          ::glow::strFormat(
              "Error: Blob name: %s not found in list of static placeholders.",
              currentBlobName_.c_str()));
    }
    auto ty = typeInfo_[currentBlobName_];

    currentTensor_.reset(new ::glow::Tensor());
    RETURN_IF_ERR(::glow::loadTensor(t, currentTensor_.get()));
    CHECK(currentTensor_->getType().isEqual(ty))
        << "Mismatched tensor type: " << currentTensor_->getType().toString()
        << " vs " << ty.toString();

    return ::glow::Error::success();
  }

  ::glow::Error setSrc(void * /*unused*/) override {
    return ::glow::Error::success();
  }

  std::string getName() override { return currentBlobName_; }

  ::glow::Tensor *getTensor() override { return currentTensor_.get(); }

  void setTypeInfo(std::map<std::string, ::glow::Type> info) override {
    typeInfo_ = std::move(info);
  }

private:
  std::unique_ptr<::glow::ZipReader> zip_;
  std::string largeBuffer_;
  std::map<std::string, ::glow::Type> typeInfo_;
  std::string currentBlobName_;
  std::unique_ptr<::glow::Tensor> currentTensor_;
  size_t weightsToLoad_{0};
  size_t i_{0};
};

/// Given a float Tensor \p t, \returns a vector of pairs of entries in t with
/// the first element in the pair being the value and the second element being
/// the original index of that value in t. The vector is partially sorted such
/// that the first \p k elements are the k elements from t with the greatest
/// values.
static std::vector<std::pair<float, size_t>>
partialSortFloatTensor(const Tensor &t, size_t k) {
  std::vector<std::pair<float, size_t>> vec;
  auto handle = t.getHandle<float>();
  for (size_t i = 0; i < handle.size(); ++i) {
    vec.push_back({handle.raw(i), i});
  }
  std::partial_sort(
      vec.begin(), vec.begin() + k, vec.end(),
      [](const auto &p1, const auto &p2) { return p1.first > p2.first; });
  return vec;
}

static float dotProd(const Tensor &t1, const Tensor &t2) {
  CHECK(t1.getElementType() == ElemKind::FloatTy);
  CHECK(t2.getElementType() == ElemKind::FloatTy);
  auto t1H = t1.getHandle<float>();
  auto t2H = t2.getHandle<float>();
  CHECK_EQ(t1H.size(), t2H.size());
  float res = 0.0f;
  for (dim_t i = 0; i < t1H.size(); i++) {
    res += t1H.raw(i) * t2H.raw(i);
  }
  return res;
}

static float cosineSimilarity(const Tensor &t1, const Tensor &t2) {
  auto fn = [](const Tensor &t1, const Tensor &t2) {
    return dotProd(t1, t2) /
           (std::sqrt(dotProd(t1, t1)) * std::sqrt(dotProd(t2, t2)));
  };
  if (t1.getType().isQuantizedType()) {
    auto t1Float = quantization::dequantizeTensor(t1, ElemKind::FloatTy);
    auto t2Float = quantization::dequantizeTensor(t2, ElemKind::FloatTy);
    return fn(t1Float, t2Float);
  } else {
    return fn(t1, t2);
  }
}

int run() {
  int numFailed = 0;

  int numTop1Matches = 0;
  int numTopKMatches = 0;
  int numTotalTopKCompares = 0;

  folly::Histogram<float> cosineHist(/* bucketSize */ 0.1f, /* min */ 0.0f,
                                     /* max */ 1.0f);

  // Build the execution engine and deserialize the Function.
  auto mod = glow::make_unique<Module>();
  Function *F = mod->createFunction("test");
  Error err = Error::empty();
  bool usingGlowCustomOps = false;
  CompilationContext cctx;
  {
    ONNXModelLoader onnxLD(modelPathOpt, {}, {}, *F, &err, onnxLoaderZipMode,
                           &cctx.backendOpts.backendSpecificNodeInfo);
    usingGlowCustomOps = onnxLD.usingGlowCustomOps();
  }
  CHECK(!ERR_TO_BOOL(std::move(err)))
      << "ONNXModelLoader failed to load model: " << modelPathOpt;
  llvm::outs() << "End onnx model load\n";

  if (glowDumpGraphAfterLoadOpt) {
    F->dumpDAG("dag.dot");
  }

  // Build host manager and compile the module.
  PrecisionConfiguration &precConfig = cctx.precisionConfig;
  if (globalFp16Opt) {
    precConfig.convertToFP16 = globalFp16Opt;
    if (sliceConcatFp32Opt) {
      precConfig.precisionModeKindSet.insert(Kinded::Kind::SliceNodeKind);
      precConfig.precisionModeKindSet.insert(Kinded::Kind::ConcatNodeKind);
    }
    llvm::outs() << "Conversion to fp16 enabled\n";
  }
  if (globalFp16PlaceholdersOpt) {
    precConfig.convertPlaceholdersToFP16 = globalFp16PlaceholdersOpt;
    llvm::outs() << "Conversion of Placeholders to fp16 enabled\n";
  }
  if (globalFp16ConstantsOpt) {
    precConfig.convertConstantsToFP16 = globalFp16ConstantsOpt;
    llvm::outs() << "Conversion of Constants to fp16 enabled\n";
  }
  if (fuseScaleOffsetFp16Opt) {
    precConfig.convertFusedToFP16 = fuseScaleOffsetFp16Opt;
    llvm::outs() << "Conversion of fused scales/offsets to fp16 enabled\n";
  }
  if (ClipFp16Opt) {
    precConfig.clipFP16 = ClipFp16Opt;
    llvm::outs() << "Clipping to fp16 enabled\n";
  }
  if (ClipFp16SkipInputsOpt) {
    precConfig.clipFP16SkipInputs = ClipFp16SkipInputsOpt;
    llvm::outs() << "Skipping clipping for fp16 Node inputs fp16\n";
  }
  if (forceFP16AccumSLSOpt) {
    precConfig.forceFP16AccumSLS = true;
    llvm::outs() << "Forcing fp16 accumulation for SLS ops enabled\n";
  }
  if (glowDumpGraphOpt) {
    cctx.dumpFinalGraph = true;
  }

  if (useSparseNNPartitioningScheme) {
    cctx.optimizationOpts.useSparseNNPartitioningScheme =
        useSparseNNPartitioningScheme;
    cctx.optimizationOpts.sparseNNPartitioningAddSLSConcats =
        sparseNNPartitioningAddSLSConcats;
    cctx.optimizationOpts.sparseNNPartitioningSchemeNumCards =
        sparseNNPartitioningSchemeNumCards;
    cctx.optimizationOpts.sparseNNPartitioningSchemeSLSTableKBytesPerCard =
        sparseNNPartitioningSchemeKBytesPerCard;
    cctx.optimizationOpts.sparseNNPartitioningSchemeNumCoresSLS =
        sparseNNPartitioningSchemeNumCoresSLS;
    cctx.optimizationOpts.sparseNNPartitioningSchemeNumCoresOther =
        sparseNNPartitioningSchemeNumCoresOther;
  }

  if (glowNNPINumParallelChunks > 1) {
    cctx.backendOpts.backendSpecificOpts["NNPINumParallelChunks"] =
        std::to_string(glowNNPINumParallelChunks);
  }

  // Load deferred weights if applicable
  const auto &placeholderList = mod->getPlaceholders();
  glow::PlaceholderList nonStaticPlaceholderList;
  std::copy_if(placeholderList.begin(), placeholderList.end(),
               std::back_inserter(nonStaticPlaceholderList),
               [](const glow::Placeholder *p) { return !p->isStatic(); });
  if (!deferredWeightsPathOpt.empty()) {
    ::glow::runtime::DeferredLoader()->registerLoader(
        new ZipFileBackedDeferredBlobLoader(deferredWeightsPathOpt));
    // Initialize loader and set field in cctx.
    auto *loader = runtime::DeferredLoader()->getLoader();
    CHECK(loader) << "No deferred weights loader registered!";

    // Generate a map of type date for all static placeholders.
    std::map<std::string, Type> staticPlaceholderTypes;
    for (auto *PH : placeholderList) {
      if (PH->isStatic()) {
        staticPlaceholderTypes[std::string(PH->getName())] = *PH->getType();
      }
    }
    loader->setTypeInfo(std::move(staticPlaceholderTypes));
    CHECK(!loader->setSrc(nullptr));
    cctx.deferredWeightLoader = loader;
    // Signal that we want to fold convertTo and Quantize into static
    // Placeholders.
    cctx.optimizationOpts.foldStaticPlaceholderConversions = true;
  }

  auto configs = runtime::generateDeviceConfigs(numDevicesOpt, ExecutionBackend,
                                                deviceMemoryOpt);
  runtime::HostConfig hostConfig;
  hostConfig.maxActiveRequests = glowMaxActiveRequests;
  hostConfig.maxQueueSize = glowMaxQueueSize;
  hostConfig.executorThreads = glowExecutorThreads;

  auto hostManager =
      glow::make_unique<runtime::HostManager>(std::move(configs), hostConfig);
  cctx.saturateHost = glowSaturateHost;
  EXIT_ON_ERR(hostManager->addNetwork(std::move(mod), cctx));

  // Parse all input and output files ahead of inference.
  std::vector<::ONNX_NAMESPACE::GraphProto> parsedInputs;
  std::vector<::ONNX_NAMESPACE::GraphProto> parsedOutputs;
  size_t inputGroupSize = inputsOpt.size();
  if (inputGroupSize) {
    for (int i = 0; i < inputGroupSize; ++i) {
      llvm::outs() << "Loading input file: " << inputsOpt[i] << "\n";
      auto inputGroup = parseOnnxFile(inputsOpt[i]);
      parsedInputs.push_back(std::move(inputGroup));
      llvm::outs() << "Loading output file: " << outputsOpt[i] << "\n";
      auto outputGroup = parseOnnxFile(outputsOpt[i]);
      parsedOutputs.push_back(std::move(outputGroup));
    }
  } else if (!inputPatternOpt.empty() && !outputPatternOpt.empty() &&
             seqLenOpt > 0) {
    inputGroupSize = seqLenOpt;
    size_t input_iter = inputPatternOpt.find("{}");
    CHECK_NE(input_iter, std::string::npos)
        << "Input pattern " << inputPatternOpt << " has to contain {}";
    size_t output_iter = outputPatternOpt.find("{}");
    CHECK_NE(output_iter, std::string::npos)
        << "Output pattern " << outputPatternOpt << " has to contain {}";
    for (unsigned i = 0; i < seqLenOpt; ++i) {
      std::string copy = inputPatternOpt;
      copy.replace(input_iter, 2, std::to_string(seqStartOpt + i));
      llvm::outs() << "Loading input file: " << copy << "\n";
      auto inputGroup = parseOnnxFile(copy);
      parsedInputs.push_back(std::move(inputGroup));
      copy = outputPatternOpt;
      copy.replace(output_iter, 2, std::to_string(seqStartOpt + i));
      llvm::outs() << "Loading output file: " << copy << "\n";
      auto outputGroup = parseOnnxFile(copy);
      parsedOutputs.push_back(std::move(outputGroup));
    }
  }

  if (parsedInputs.empty()) {
    llvm::outs() << "No inputs are provided. Exiting...\n";
    return -1;
  }

  llvm::outs() << "Starting inference\n";
  TraceContext mergedTraceContext(TraceLevel::STANDARD);
  folly::CPUThreadPoolExecutor threadPool(concurrentRequestsOpt);
  std::mutex mutex;
  std::condition_variable cv;
  int numTotalInferences = inputGroupSize * itersOpt;
  int numFinishedInferences = 0;

  std::list<InferenceResult> results;

  auto startTime = std::chrono::steady_clock::now();
  for (int ioIndex = 0, numInferencesIssued = 0;
       numInferencesIssued < numTotalInferences;
       ++numInferencesIssued, ioIndex = numInferencesIssued % inputGroupSize) {

    results.emplace_back(InferenceResult());
    auto &result = results.back();

    threadPool.add([&parsedInputs, &nonStaticPlaceholderList, ioIndex,
                    &mergedTraceContext, &hostManager, &result, &cv, &mutex,
                    numTotalInferences, &numFinishedInferences,
                    usingGlowCustomOps]() {
      // Setup the inputs.
      auto ctx = glow::make_unique<ExecutionContext>();

      TraceContext *traceContext = nullptr;
      if (glowDumpTrace) {
        ctx->setTraceContext(
            glow::make_unique<TraceContext>(TraceLevel::STANDARD));
        traceContext = ctx->getTraceContext();
        traceContext->setThreadName("Caller");
      }
      TRACE_EVENT_SCOPE(traceContext, TraceLevel::RUNTIME,
                        "Dispatch to prep input and dispatch");

      auto &bindings = *ctx->getPlaceholderBindings();
      bindings.clear();
      bindings.allocate(nonStaticPlaceholderList);
      const auto &ps = bindings.pairs();
      for (const auto &kv : ps) {
        VLOG(1) << "Placeholder allocated: " << kv.first->getName().str();
      }

      const auto &inputGroup = parsedInputs[ioIndex];
      // This holds the tensor that actually owns the data for all the partial
      // inputs.
      std::vector<Tensor> partialTensorPayloads;
      fillPlaceholders(inputGroup, &bindings,
                       enablePartialTensor ? &partialTensorPayloads : nullptr,
                       usingGlowCustomOps);

      std::promise<void> promise;
      auto future = promise.get_future();

      hostManager->runNetwork(
          "test", std::move(ctx),
          [&promise, index = ioIndex,
           &result](runtime::RunIdentifierTy, Error err,
                    std::unique_ptr<ExecutionContext> contextPtr) mutable {
            result.error = std::move(err);
            result.ctx = std::move(contextPtr);
            result.index = index;
            result.endTime = std::chrono::steady_clock::now();
            promise.set_value();
          });

      // wait for glow to finish.
      future.wait();
      traceContext = result.ctx->getTraceContext();
      if (traceContext) {
        // merge() has internal lock and is thread safe.
        mergedTraceContext.merge(traceContext);
      }

      if (skipCorrectnessCheck) {
        // if skipping correctness check, throw away the context to keep
        // memory usage low.
        result.ctx.reset();
      }

      std::unique_lock<std::mutex> lock(mutex);
      if (++numFinishedInferences >= numTotalInferences) {
        lock.unlock();
        cv.notify_all();
      }
    });
  }

  // wait for all inferneces to finish
  std::unique_lock<std::mutex> lock(mutex);
  cv.wait(lock, [&]() { return numFinishedInferences >= numTotalInferences; });

  auto endTime = startTime;
  llvm::outs() << "All inferences done. Checking results\n";
  for (auto &result : results) {
    if (result.endTime > endTime) {
      endTime = result.endTime;
    }

    if (result.error) {
      llvm::outs() << "Inference failed!\n";
      ++numFailed;
    } else {
      const auto &outputGroup = parsedOutputs[result.index];
      ONNX_NAMESPACE::GraphProto outputG;
      std::ofstream of;
      if (dumpOutputsOpt) {
        std::stringstream ss;
        ss << "output_dump_" << result.index << ".onnx";
        of.open(ss.str(), std::ios::binary);
        CHECK(of) << "Cannot create output dump file: " << ss.str();
      }

      if (!skipCorrectnessCheck || topKCompare > 0 || cosineSimilarityStats) {
        CHECK(result.ctx);
        const auto &bindings = *result.ctx->getPlaceholderBindings();
        for (const auto &tp : outputGroup.initializer()) {
          Tensor tensorRef;
          auto error = loadTensor(tp, &tensorRef, usingGlowCustomOps);
          CHECK(!ERR_TO_BOOL(std::move(error)))
              << "Cannot load output ref tensor";
          const auto *tensor =
              bindings.get(bindings.getPlaceholderByName(tp.name()));
          CHECK(tensor) << "Missing " << tp.name() << " in output placeholder";

          if (cosineSimilarityStats) {
            cosineHist.addValue(cosineSimilarity(*tensor, tensorRef));
          }

          if (topKCompare > 0) {
            numTotalTopKCompares++;
            assert(tensor->size() == tensorRef.size());
            auto sortedResults = partialSortFloatTensor(*tensor, topKCompare);
            auto sortedRefs = partialSortFloatTensor(tensorRef, topKCompare);
            assert(sortedResults.size() == topKCompare &&
                   sortedResults.size() == topKCompare);

            bool allKMatch = true;
            std::stringstream ss;
            for (auto i = 0; i < topKCompare; i++) {
              if (sortedResults[i].second == sortedRefs[i].second) {
                if (i == 0) {
                  numTop1Matches++;
                }
              } else {
                allKMatch = false;
              }
              if (logTopKResultsPerExample) {
                ss << i << ": Test result: " << sortedResults[i].second
                   << " (p=" << sortedResults[i].first
                   << ") Reference result: " << sortedRefs[i].second
                   << " (p=" << sortedRefs[i].first << ")\n";
              }
            }
            if (logTopKResultsPerExample) {
              llvm::outs() << ss.str() << "\n";
            }
            if (allKMatch) {
              numTopKMatches++;
            }
          }

          if (dumpOutputsOpt) {
            auto *t = outputG.add_initializer();
            ONNXModelWriter::writeTensor(*tensor, t, usingGlowCustomOps);
            t->set_name(tp.name());
          }

          if (!skipCorrectnessCheck) {
            bool equal = tensorRef.isEqual(*tensor, thresholdOpt, true);
            if (!equal) {
              llvm::outs() << "Verification failed at input/output pair "
                           << result.index << " for output tensor " << tp.name()
                           << "\n";
              ++numFailed;
              break;
            }
          }
        }
      }

      if (dumpOutputsOpt) {
        std::string buffer;
        outputG.SerializeToString(&buffer);
        of << buffer;
      }
    }
  }

  if (glowDumpTrace) {
    llvm::SmallString<64> path;
    if (glowDumpTraceFile.empty()) {
      auto tempFileRes =
          llvm::sys::fs::createTemporaryFile("glow-trace", "json", path);
      if (tempFileRes.value() != 0) {
        LOG(ERROR) << "Failed to create temp file for Glow trace events: "
                   << tempFileRes;
      } else {
        LOG(INFO) << "Trace path=" << path.c_str();
        mergedTraceContext.dump(path);
      }
    } else {
      LOG(INFO) << "Trace path=" << path.c_str();
      mergedTraceContext.dump(glowDumpTraceFile);
    }
  }

  if (!skipCorrectnessCheck) {
    if (numFailed == 0) {
      llvm::outs() << "All passed!\n";
    } else {
      llvm::outs() << numFailed << " inferences failed to match reference.\n";
    }
  }

  if (topKCompare > 0) {
    llvm::outs() << "Num top1 exact matches: " << numTop1Matches << "/"
                 << numTotalTopKCompares << "\n";
    llvm::outs() << "Num topK exact matches (k=" << topKCompare
                 << "): " << numTopKMatches << "/" << numTotalTopKCompares
                 << "\n";

    if (top1Threshold > 0.0) {
      float top1MatchRate = float(numTop1Matches) / numTotalTopKCompares;
      if (top1MatchRate < top1Threshold) {
        llvm::outs() << "Expected top1 match rate of at least " << top1Threshold
                     << " but only achieved " << top1MatchRate << "\n";
        return numTotalTopKCompares - numTop1Matches;
      }
    }
  }

  if (cosineSimilarityStats) {
    float p50Similarity = cosineHist.getPercentileEstimate(0.5);
    llvm::outs() << "cosine similarity stats:\n"
                 << "p01: " << cosineHist.getPercentileEstimate(0.01) << "\n"
                 << "p02: " << cosineHist.getPercentileEstimate(0.02) << "\n"
                 << "p05: " << cosineHist.getPercentileEstimate(0.05) << "\n"
                 << "p10: " << cosineHist.getPercentileEstimate(0.1) << "\n"
                 << "p25: " << cosineHist.getPercentileEstimate(0.25) << "\n"
                 << "p50: " << p50Similarity << "\n"
                 << "p75: " << cosineHist.getPercentileEstimate(0.75) << "\n"
                 << "p90: " << cosineHist.getPercentileEstimate(0.90) << "\n"
                 << "p95: " << cosineHist.getPercentileEstimate(0.95) << "\n"
                 << "p98: " << cosineHist.getPercentileEstimate(0.98) << "\n"
                 << "p99: " << cosineHist.getPercentileEstimate(0.99) << "\n";
    if (cosineSimilarityThreshold > 0.0) {
      if (p50Similarity < cosineSimilarityThreshold) {
        llvm::outs() << "Expected p50 cosine similarity of at least "
                     << cosineSimilarityThreshold << " but only achieved "
                     << p50Similarity << "\n";
        return 1;
      }
    }
  }

  std::chrono::duration<double, std::milli> duration = endTime - startTime;
  std::cout << "Total inference duration (ms): " << duration.count() << "\n";
  std::cout << "Avg inference duration (ms): "
            << duration.count() / numTotalInferences << "\n";
  std::cout << "Avg inference per second: "
            << numTotalInferences * 1000 / duration.count() << "\n";

  return numFailed;
}

} // namespace

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  parseCommandLine(argc, argv);
  return run();
}
