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

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Signals.h"

#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"

#include <fstream>
#include <string>

constexpr size_t MAX_MEMORY = 64e+9;

using namespace glow;

namespace {
llvm::cl::OptionCategory reproTestCat("Repro Category");
llvm::cl::opt<std::string> modelPathOpt("model", llvm::cl::desc("Input models"),
                                        llvm::cl::value_desc("modelPath"),
                                        llvm::cl::Required,
                                        llvm::cl::cat(reproTestCat));
llvm::cl::list<std::string> inputsOpt("inputs", llvm::cl::desc("Inputs"),
                                      llvm::cl::value_desc("Inputs"),
                                      llvm::cl::Required, llvm::cl::OneOrMore,
                                      llvm::cl::cat(reproTestCat));
llvm::cl::list<std::string> outputsOpt("outputs", llvm::cl::desc("Ouptuts"),
                                       llvm::cl::value_desc("Ouptuts"),
                                       llvm::cl::Required, llvm::cl::OneOrMore,
                                       llvm::cl::cat(reproTestCat));

llvm::cl::opt<std::string> ExecutionBackend(
    "backend", llvm::cl::desc("Backend to use, e.g., Interpreter, CPU, NNPI:"),
    llvm::cl::init("Interpreter"), llvm::cl::cat(reproTestCat));

llvm::cl::opt<unsigned> concurrentRequestsOpt(
    "concurrent-count", llvm::cl::desc("Number of concurrent requests."),
    llvm::cl::Optional, llvm::cl::init(1), llvm::cl::cat(reproTestCat));

llvm::cl::opt<float> thresholdOpt(
    "threshold", llvm::cl::desc("theshold for tensor numeric comparison"),
    llvm::cl::Optional, llvm::cl::init(1e-5), llvm::cl::cat(reproTestCat));

llvm::cl::opt<bool>
    globalFp16Opt("glow_global_fp16",
                  llvm::cl::desc("Enable fp16 lowering for all ops on the net"),
                  llvm::cl::Optional, llvm::cl::cat(reproTestCat));
llvm::cl::opt<bool> dumpOutputsOpt("dump_outputs",
                                   llvm::cl::desc("Dump output tensors"),
                                   llvm::cl::Optional,
                                   llvm::cl::cat(reproTestCat));

llvm::cl::opt<bool> fuseScaleOffsetFp16Opt(
    "glow_global_fused_scale_offset_fp16",
    llvm::cl::desc(
        "Enable fp16 lowering for all op inputs using fused scale/offset"),
    llvm::cl::Optional, llvm::cl::cat(reproTestCat));

llvm::cl::opt<bool>
    ClipFp16Opt("glow_clip_fp16",
                llvm::cl::desc("Force glow to clip fp16 values to min/max"),
                llvm::cl::Optional, llvm::cl::cat(reproTestCat));

llvm::cl::opt<bool> enablePartialTensor("glow_enable_partial_tensor",
                                        llvm::cl::desc("Enable partial tensor"),
                                        llvm::cl::Optional,
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

void run() {
  // Build the execution engine and deserialize the Function.
  auto mod = glow::make_unique<Module>();
  Function *F = mod->createFunction("test");
  Error err = Error::empty();
  { ONNXModelLoader onnxLD(modelPathOpt, {}, {}, *F, &err, /*zipMode*/ true); }
  CHECK(!ERR_TO_BOOL(std::move(err)))
      << "ONNXModelLoader failed to load model: " << modelPathOpt;

  // Setup the inputs.
  auto ctx = glow::make_unique<ExecutionContext>();
  auto &bindings = *ctx->getPlaceholderBindings();
  bindings.clear();
  bindings.allocate(mod->getPlaceholders());
  const auto &ps = bindings.pairs();
  for (const auto &kv : ps) {
    llvm::outs() << "Placeholder allocated: " << kv.first->getName() << "\n";
  }

  // Build host manager and compile the module.
  // TODO: fix MAX_MEMORY
  PrecisionConfiguration precConfig;
  if (globalFp16Opt) {
    precConfig.convertToFP16 = globalFp16Opt;
    llvm::outs() << "Conversion to fp16 enabled\n";
  }
  if (fuseScaleOffsetFp16Opt) {
    precConfig.convertFusedToFP16 = fuseScaleOffsetFp16Opt;
    llvm::outs() << "Conversion of fused scales/offsets to fp16 enabled\n";
  }
  if (ClipFp16Opt) {
    precConfig.clipFP16 = ClipFp16Opt;
    llvm::outs() << "Clipping to fp16 enabled\n";
  }
  auto configs =
      runtime::generateDeviceConfigs(1, ExecutionBackend, MAX_MEMORY);
  auto hostManager =
      glow::make_unique<runtime::HostManager>(std::move(configs));
  CompilationContext cctx;
  cctx.precisionConfig = precConfig;
  EXIT_ON_ERR(hostManager->addNetwork(std::move(mod), cctx));

  // Run inference.
  size_t inputGroupSize = inputsOpt.size();
  for (int i = 0; i < inputGroupSize; ++i) {
    // This holds the tensor that actually owns the data for all the partial
    // inputs.
    std::vector<Tensor> partialTensorPayloads;

    llvm::outs() << "input file: " << inputsOpt[i] << "\n";
    auto inputGroup = parseIO(inputsOpt[i]);
    for (const auto &tp : inputGroup.initializer()) {
      auto *tensor = bindings.get(bindings.getPlaceholderByName(tp.name()));
      CHECK(tensor);
      size_t fullSize = tensor->getSizeInBytes();
      const auto fullType = tensor->getType();

      auto error = loadTensor(tp, tensor);
      bool hasError = ERR_TO_BOOL(std::move(error));
      CHECK(!hasError) << "Cannot load input tensor";
      size_t loadedSize = tensor->getSizeInBytes();
      if (loadedSize != fullSize) {
        if (enablePartialTensor) {
          LOG(INFO) << "Loading " << tp.name()
                    << " as a partial tensor: partial size="
                    << tensor->getType().toString()
                    << " full size=" << fullType.toString();
          Tensor fullTensor(tensor->getUnsafePtr(), &fullType,
                            tensor->getSizeInBytes());
          // 'fullTensor' doesn't own the underlying data. 'tensor' does. So we
          // want to keep the original tensor object around until inference is
          // finished.
          partialTensorPayloads.emplace_back(std::move(*tensor));
          *tensor = std::move(fullTensor);
        } else {
          // pad with 0
          LOG(INFO) << "Loading and padding " << tp.name()
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

    dispatchInference("test", hostManager.get(), *ctx, concurrentRequestsOpt);

    llvm::outs() << "output file: " << outputsOpt[i] << "\n";
    auto outputGroup = parseIO(outputsOpt[i]);
    ONNX_NAMESPACE::GraphProto outputG;
    std::ofstream of;
    if (dumpOutputsOpt) {
      std::stringstream ss;
      ss << "output_dump_" << i << ".onnx";
      of.open(ss.str(), std::ios::binary);
      CHECK(of) << "Cannot create output dump file: " << ss.str();
    }
    for (const auto &tp : outputGroup.initializer()) {
      Tensor tensorRef;
      auto error = loadTensor(tp, &tensorRef);
      CHECK(!ERR_TO_BOOL(std::move(error))) << "Cannot load output ref tensor";
      const auto *tensor =
          bindings.get(bindings.getPlaceholderByName(tp.name()));
      CHECK(tensor);
      if (dumpOutputsOpt) {
        auto *t = outputG.add_initializer();
        ONNXModelWriter::writeTensor(*tensor, t);
        t->set_name(tp.name());
      }
      bool equal = tensorRef.isEqual(*tensor, thresholdOpt, true);
      if (!equal) {
        llvm::outs() << "Verification failed at input/output pair " << i
                     << " for output tensor " << tp.name() << "\n";
        return;
      }
    }
    if (dumpOutputsOpt) {
      std::string buffer;
      outputG.SerializeToString(&buffer);
      of << buffer;
    }
  }

  llvm::outs() << "All passed!\n";
}

} // namespace

int main(int argc, char **argv) {
  parseCommandLine(argc, argv);
  run();
  return 0;
}
