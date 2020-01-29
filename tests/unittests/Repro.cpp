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

#include <fstream>
#include <string>

using namespace glow;

namespace glow {
namespace onnxifi {
#ifdef GLOW_WITH_NNPI
extern bool GlowDumpGraph;
#endif
} // namespace onnxifi
} // namespace glow

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

llvm::cl::opt<bool>
    globalFp16Opt("glow_global_fp16",
                  llvm::cl::desc("Enable fp16 lowering for all ops on the net"),
                  llvm::cl::Optional, llvm::cl::init(true),
                  llvm::cl::cat(reproTestCat));

llvm::cl::opt<bool> globalFp16ConstantsOpt(
    "glow_global_fp16_constants",
    llvm::cl::desc("Enable fp16 conversion for Constants"), llvm::cl::Optional,
    llvm::cl::init(false), llvm::cl::cat(reproTestCat));

llvm::cl::opt<bool> globalFp16PlaceholdersOpt(
    "glow_global_fp16_placeholders",
    llvm::cl::desc("Enable fp16 conversion for Placeholders"),
    llvm::cl::Optional, llvm::cl::init(false), llvm::cl::cat(reproTestCat));

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
    llvm::cl::Optional, llvm::cl::init(false), llvm::cl::cat(reproTestCat));

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

llvm::cl::opt<bool> glowDumpTrace("glow_dump_debug_traces",
                                  llvm::cl::desc("Dump glow trace"),
                                  llvm::cl::Optional, llvm::cl::init(false),
                                  llvm::cl::cat(reproTestCat));

void parseCommandLine(int argc, char **argv) {
  llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);
  llvm::cl::ParseCommandLineOptions(
      argc, argv,
      " The Glow compiler\n\n"
      "Glow is a compiler for neural network accelerators.\n");
}

::ONNX_NAMESPACE::GraphProto parseIO(const std::string &filename) {
  ::ONNX_NAMESPACE::GraphProto g;
  std::ifstream ff(filename, std::ios::in | std::ios::binary);
  CHECK(ff) << "Can't find the input file for " << filename.c_str();
  google::protobuf::io::IstreamInputStream fileStream(&ff);
  google::protobuf::io::CodedInputStream codedStream(&fileStream);
  codedStream.SetTotalBytesLimit(MAX_PROTO_SIZE, MAX_PROTO_SIZE);
  bool yes = g.ParseFromCodedStream(&codedStream);
  CHECK(yes) << "Failed to parse GraphProto";
  return g;
}

struct InferenceResult {
  Error error = Error::empty();
  std::unique_ptr<ExecutionContext> ctx;
  int index = 0;
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
    typeInfo_ = info;
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

int run() {
  int numFailed = 0;

  // Build the execution engine and deserialize the Function.
  auto mod = glow::make_unique<Module>();
  Function *F = mod->createFunction("test");
  Error err = Error::empty();
  { ONNXModelLoader onnxLD(modelPathOpt, {}, {}, *F, &err, /*zipMode*/ true); }
  CHECK(!ERR_TO_BOOL(std::move(err)))
      << "ONNXModelLoader failed to load model: " << modelPathOpt;

  // Build host manager and compile the module.
  PrecisionConfiguration precConfig;
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
    llvm::outs() << "Conversion of Placeholders to fp16 enabled";
  }
  if (globalFp16ConstantsOpt) {
    precConfig.convertConstantsToFP16 = globalFp16ConstantsOpt;
    llvm::outs() << "Conversion of Constants to fp16 enabled";
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
    llvm::outs() << "Skipping clipping for fp16 Node inputs fp16";
  }
  if (forceFP16AccumSLSOpt) {
    precConfig.forceFP16AccumSLS = true;
    llvm::outs() << "Forcing fp16 accumulation for SLS ops enabled\n";
  }
  CompilationContext cctx;
  cctx.precisionConfig = precConfig;
  if (useSparseNNPartitioningScheme) {
    cctx.optimizationOpts.useSparseNNPartitioningScheme =
        useSparseNNPartitioningScheme;
    cctx.optimizationOpts.sparseNNPartitioningSchemeNumCards =
        sparseNNPartitioningSchemeNumCards;
    cctx.optimizationOpts.sparseNNPartitioningSchemeSLSTableKBytesPerCard =
        sparseNNPartitioningSchemeKBytesPerCard;
    cctx.optimizationOpts.sparseNNPartitioningSchemeNumCoresSLS =
        sparseNNPartitioningSchemeNumCoresSLS;
    cctx.optimizationOpts.sparseNNPartitioningSchemeNumCoresOther =
        sparseNNPartitioningSchemeNumCoresOther;
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
  auto hostManager =
      glow::make_unique<runtime::HostManager>(std::move(configs));
  EXIT_ON_ERR(hostManager->addNetwork(std::move(mod), cctx));

  // Parse all input and output files ahead of inference.
  std::vector<::ONNX_NAMESPACE::GraphProto> parsedInputs;
  std::vector<::ONNX_NAMESPACE::GraphProto> parsedOutputs;
  size_t inputGroupSize = inputsOpt.size();
  if (inputGroupSize) {
    for (int i = 0; i < inputGroupSize; ++i) {
      llvm::outs() << "Loading input file: " << inputsOpt[i] << "\n";
      auto inputGroup = parseIO(inputsOpt[i]);
      parsedInputs.push_back(std::move(inputGroup));
      llvm::outs() << "Loading output file: " << outputsOpt[i] << "\n";
      auto outputGroup = parseIO(outputsOpt[i]);
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
      auto inputGroup = parseIO(copy);
      parsedInputs.push_back(std::move(inputGroup));
      copy = outputPatternOpt;
      copy.replace(output_iter, 2, std::to_string(seqStartOpt + i));
      llvm::outs() << "Loading output file: " << copy << "\n";
      auto outputGroup = parseIO(copy);
      parsedOutputs.push_back(std::move(outputGroup));
    }
  }

  if (parsedInputs.empty()) {
    llvm::outs() << "No inputs are provided. Exiting...\n";
    return -1;
  }

  llvm::outs() << "Starting inference\n";

  std::list<std::promise<InferenceResult>> promises;
  std::list<std::future<InferenceResult>> futures;
  // This holds the tensor that actually owns the data for all the partial
  // inputs.
  std::vector<Tensor> partialTensorPayloads;

  auto timer = std::chrono::steady_clock::now();
  int numTotalInferences = inputGroupSize * itersOpt;
  for (int ioIndex = 0, numInferencesIssued = 0;
       numInferencesIssued < numTotalInferences;
       ++numInferencesIssued, ioIndex = numInferencesIssued % inputGroupSize) {
    // Setup the inputs.
    auto ctx = glow::make_unique<ExecutionContext>();

    TraceContext *traceContext = nullptr;
    if (glowDumpTrace) {
      ctx->setTraceContext(
          glow::make_unique<TraceContext>(TraceLevel::STANDARD));
      traceContext = ctx->getTraceContext();
      traceContext->setThreadName("main");
    }
    {
      TRACE_EVENT_SCOPE(traceContext, TraceLevel::RUNTIME,
                        "Prepping input to Glow");
      auto &bindings = *ctx->getPlaceholderBindings();
      bindings.clear();
      bindings.allocate(nonStaticPlaceholderList);
      const auto &ps = bindings.pairs();
      for (const auto &kv : ps) {
        VLOG(1) << "Placeholder allocated: " << kv.first->getName().str();
      }

      const auto &inputGroup = parsedInputs[ioIndex];
      for (const auto &tp : inputGroup.initializer()) {
        auto *tensor = bindings.get(bindings.getPlaceholderByName(tp.name()));
        CHECK(tensor) << "Unable to get tensor for " << tp.name();
        size_t fullSize = tensor->getSizeInBytes();
        const auto fullType = tensor->getType();

        auto error = loadTensor(tp, tensor);
        bool hasError = ERR_TO_BOOL(std::move(error));
        CHECK(!hasError) << "Cannot load input tensor";
        size_t loadedSize = tensor->getSizeInBytes();
        if (loadedSize != fullSize) {
          if (enablePartialTensor) {
            VLOG(1) << "Loading " << tp.name()
                    << " as a partial tensor: partial size="
                    << tensor->getType().toString()
                    << " full size=" << fullType.toString();
            Tensor fullTensor(tensor->getUnsafePtr(), &fullType,
                              tensor->getSizeInBytes());
            // 'fullTensor' doesn't own the underlying data. 'tensor' does. So
            // we want to keep the original tensor object around until inference
            // is finished.
            partialTensorPayloads.emplace_back(std::move(*tensor));
            *tensor = std::move(fullTensor);
          } else {
            // pad with 0
            VLOG(1) << "Loading and padding " << tp.name()
                    << " as a partial tensor: partial size="
                    << tensor->getType().toString()
                    << " full size=" << fullType.toString();
            Tensor fullTensor(&fullType);
            std::memcpy(fullTensor.getUnsafePtr(), tensor->getUnsafePtr(),
                        tensor->getSizeInBytes());
            std::memset(fullTensor.getUnsafePtr() + tensor->getSizeInBytes(), 0,
                        fullTensor.getSizeInBytes() - tensor->getSizeInBytes());

            *tensor = std::move(fullTensor);
          }
        }
      }
    }

    {
      TRACE_EVENT_SCOPE(traceContext, TraceLevel::RUNTIME,
                        "Dispatch to host manager");

      // dispatch inference
      promises.emplace_back(std::promise<InferenceResult>());
      std::promise<InferenceResult> &promise = promises.back();
      futures.emplace_back(promise.get_future());

      hostManager->runNetwork(
          "test", std::move(ctx),
          [&promise, index = ioIndex](
              runtime::RunIdentifierTy, Error err,
              std::unique_ptr<ExecutionContext> contextPtr) mutable {
            InferenceResult result;
            result.error = std::move(err);
            result.ctx = std::move(contextPtr);
            result.index = index;
            promise.set_value(std::move(result));
          });
    }
    // stop and wait for results when we reach the max concurrent request we
    // want to send or we have no more requests to send.
    if (futures.size() >= concurrentRequestsOpt ||
        numInferencesIssued == numTotalInferences - 1) {
      for (auto &future : futures) {
        future.wait();
      }
      std::chrono::duration<double, std::milli> duration =
          std::chrono::steady_clock::now() - timer;
      LOG(INFO) << "Executed " << futures.size() << " inferences in "
                << duration.count() << "ms";

      for (auto &future : futures) {
        auto result = future.get();

        if (result.error) {
          llvm::outs() << "Inference failed!";
          ++numFailed;
        } else {
          const auto &bindings = *result.ctx->getPlaceholderBindings();

          const auto &outputGroup = parsedOutputs[result.index];
          ONNX_NAMESPACE::GraphProto outputG;
          std::ofstream of;
          if (dumpOutputsOpt) {
            std::stringstream ss;
            ss << "output_dump_" << result.index << ".onnx";
            of.open(ss.str(), std::ios::binary);
            CHECK(of) << "Cannot create output dump file: " << ss.str();
          }
          for (const auto &tp : outputGroup.initializer()) {
            Tensor tensorRef;
            auto error = loadTensor(tp, &tensorRef);
            CHECK(!ERR_TO_BOOL(std::move(error)))
                << "Cannot load output ref tensor";
            const auto *tensor =
                bindings.get(bindings.getPlaceholderByName(tp.name()));
            CHECK(tensor) << "Missing " << tp.name()
                          << " in output placeholder";
            if (dumpOutputsOpt) {
              auto *t = outputG.add_initializer();
              ONNXModelWriter::writeTensor(*tensor, t);
              t->set_name(tp.name());
            }
            bool equal = tensorRef.isEqual(*tensor, thresholdOpt, true);
            if (!equal) {
              llvm::outs() << "Verification failed at input/output pair "
                           << result.index << " for output tensor " << tp.name()
                           << "\n";
              ++numFailed;
              break;
            }
          }
          if (dumpOutputsOpt) {
            std::string buffer;
            outputG.SerializeToString(&buffer);
            of << buffer;
          }

          if (glowDumpTrace) {
            llvm::SmallString<64> path;
            auto tempFileRes =
                llvm::sys::fs::createTemporaryFile("glow-trace", "json", path);
            if (tempFileRes.value() != 0) {
              LOG(ERROR) << "Failed to create temp file for Glow trace events: "
                         << tempFileRes;
            } else {
              traceContext->dump(path);
            }
          }
        }
      }
      promises.clear();
      futures.clear();
      partialTensorPayloads.clear();
      timer = std::chrono::steady_clock::now();
    }
  }

  CHECK(promises.empty());
  CHECK(futures.empty());

  if (numFailed == 0) {
    llvm::outs() << "All passed!\n";
  } else {
    llvm::outs() << numFailed << " inferences failed to match reference.\n";
  }
  return numFailed;
}

} // namespace

int main(int argc, char **argv) {
  parseCommandLine(argc, argv);
  if (glowDumpGraphOpt) {
#ifdef GLOW_WITH_NNPI
    glow::onnxifi::GlowDumpGraph = true;
#endif
  }
  return run();
}
