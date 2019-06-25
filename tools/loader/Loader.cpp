/**
 * Copyright (c) 2017-present, Facebook, Inc.
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

#include "Loader.h"

#include "glow/Base/Tensor.h"
#include "glow/Converter/TypeAToTypeBFunctionConverter.h"
#include "glow/IR/IR.h"
#include "glow/Optimizer/GraphOptimizer/GraphOptimizer.h"
#include "glow/Quantization/Quantization.h"
#include "glow/Quantization/Serialization.h"
#include "glow/Runtime/RuntimeTypes.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/raw_ostream.h"

#include <sstream>

using namespace glow;

/// -enable-rowwise : Command line option to enable rowwise quantized
/// fullyconnected in quantization producure.
bool enableRowwiseOpt;
static llvm::cl::opt<bool, true>
    enableRowwiseF("enable-rowwise",
                   llvm::cl::desc("Enable rowwise quantized fully connected."),
                   llvm::cl::location(enableRowwiseOpt), llvm::cl::init(false));

namespace {
llvm::cl::OptionCategory loaderCat("Loader Options");

llvm::cl::list<std::string> modelPathOpt(
    "model",
    llvm::cl::desc(
        "Specify one of three:\n"
        "1. Path to ONNX model file.\n"
        "2. Two paths to Caffe2 model files: network structure and weight.\n"
        "3. Path to directory with the Caffe2 network structure "
        "<predict_net.pb> and weight <init_net.pb> files."),
    llvm::cl::value_desc("modelPath"), llvm::cl::Required, llvm::cl::OneOrMore,
    llvm::cl::cat(loaderCat));
llvm::cl::alias modelPathAOpt("m", llvm::cl::desc("Alias for -model"),
                              llvm::cl::aliasopt(modelPathOpt),
                              llvm::cl::cat(loaderCat));

llvm::cl::opt<bool>
    verbose("verbose",
            llvm::cl::desc("Specify whether to run with verbose output"),
            llvm::cl::Optional, llvm::cl::cat(loaderCat));

llvm::cl::opt<std::string> dumpProfileFileOpt(
    "dump-profile",
    llvm::cl::desc("Perform quantization profiling for a given graph "
                   "and dump result to the file."),
    llvm::cl::value_desc("profile.yaml"), llvm::cl::Optional,
    llvm::cl::cat(loaderCat));

llvm::cl::opt<quantization::Schema> quantizationSchema(
    "quantization-schema",
    llvm::cl::desc("Specify which quantization schema to use"),
    llvm::cl::Optional,
    llvm::cl::values(
        clEnumValN(quantization::Schema::Asymmetric, "asymmetric",
                   "Use asymmetric ranges"),
        clEnumValN(quantization::Schema::Symmetric, "symmetric",
                   "Use symmetric ranges"),
        clEnumValN(quantization::Schema::SymmetricWithUnsigned,
                   "symmetric_with_uint8",
                   "Use symmetric ranges with potentially uint8 ranges")),
    llvm::cl::init(quantization::Schema::Asymmetric), llvm::cl::cat(loaderCat));

llvm::cl::opt<ElemKind> quantizationPrecision(
    "quantization-precision",
    llvm::cl::desc("Specify which quantization precision to use, e.g., Int8"),
    llvm::cl::Optional,
    llvm::cl::values(
        clEnumValN(ElemKind::Int8QTy, "Int8", "Use Int8 quantization"),
        clEnumValN(ElemKind::Int16QTy, "Int16", "Use Int16 quantization")),
    llvm::cl::init(ElemKind::Int8QTy), llvm::cl::cat(loaderCat));

llvm::cl::opt<std::string> loadProfileFileOpt(
    "load-profile",
    llvm::cl::desc("Load quantization profile file and quantize the graph"),
    llvm::cl::value_desc("profile.yaml"), llvm::cl::Optional,
    llvm::cl::cat(loaderCat));

llvm::cl::list<std::string> keepOriginalPrecisionForNodesOpt(
    "keep-original-precision-for-nodes",
    llvm::cl::desc(
        "Use to specify the name of nodes (e.g. Add, Div, etc.) that should "
        "be kept as is when conversion/quantization is requested. "
        "All nodes of the listed kinds will be kept as is;"
        "e.g. if Add is specified and there are multiple Add nodes "
        "in the input loaded model, none would be quantized/converted."),
    llvm::cl::value_desc("NodeNames (e.g. Add,Div)"), llvm::cl::ZeroOrMore,
    llvm::cl::CommaSeparated, llvm::cl::cat(loaderCat));

llvm::cl::list<std::string> doNotLowerNodesForProfilingOpt(
    "do-not-lower-nodes-for-profiling",
    llvm::cl::desc(
        "Use to specify the name of nodes (e.g. Convolution, FullyConnected, "
        "etc.) that should not be lowered during profiling. All nodes of the "
        "listed kinds will be kept as is; e.g. if Conv is specified and the "
        "model has group convolutions then the convolution will not be lowered "
        "for profiling. This means when using the profile for quantization, "
        "the node should not be lowered then either."),
    llvm::cl::value_desc("NodeNames (e.g. Convolution,FullyConnected)"),
    llvm::cl::ZeroOrMore, llvm::cl::CommaSeparated, llvm::cl::cat(loaderCat));

llvm::cl::opt<std::string> ExecutionBackend(
    "backend",
    llvm::cl::desc("Backend to use, e.g., Interpreter, CPU, OpenCL:"),
    llvm::cl::init("Interpreter"), llvm::cl::cat(loaderCat));

/// Debugging options.
llvm::cl::OptionCategory
    modelExportCat("How to export the Glow Intermediate Representation/Graphs",
                   "These options are for debugging the "
                   "graphs by writing the IR/Graphs to "
                   "given files/stdout");

llvm::cl::opt<std::string> dumpGraphDAGFileBeforeCompilationOpt(
    "dump-graph-DAG-before-compile",
    llvm::cl::desc("Specify the file to export the Graph in DOT format"),
    llvm::cl::value_desc("file.dot"), llvm::cl::cat(modelExportCat));

llvm::cl::opt<std::string> dumpGraphDAGFileOpt(
    "dump-graph-DAG",
    llvm::cl::desc("Specify the file to export the Graph in DOT format"),
    llvm::cl::value_desc("file.dot"), llvm::cl::cat(modelExportCat));

llvm::cl::opt<bool> dumpGraphOpt("dump-graph",
                                 llvm::cl::desc("Prints Graph to stdout"),
                                 llvm::cl::cat(modelExportCat));

llvm::cl::opt<bool>
    convertToFP16("convert-to-fp16",
                  llvm::cl::desc("Run all floating-point computation in fp16."),
                  llvm::cl::init(false), llvm::cl::cat(loaderCat));

/// Emit a bundle into the specified output directory.
llvm::cl::opt<std::string>
    emitBundle("emit-bundle",
               llvm::cl::desc("Output directory for the bundle serialization"),
               llvm::cl::cat(loaderCat));

llvm::cl::opt<bool> assertAllNodesQuantizedOpt(
    "assert-all-nodes-quantized",
    llvm::cl::desc(
        "Debugging tool, used to assert the quantizer quantizes all nodes in "
        "the model, or abort otherwise. When false, nodes that are unsupported "
        "as quantized by the backend will be left unquantized, and may have "
        "their inputs dequantized/outputs quantized as necessary. Can be used "
        "in conjunction with -keep-original-precision-for-nodes to explicitly "
        "whitelist node kinds that are allowed to be left unquantized."),
    llvm::cl::init(false), llvm::cl::cat(loaderCat));

/// The device configs file used for Runtime.
llvm::cl::opt<std::string> loadDeviceConfigsFileOpt(
    "load-device-configs",
    llvm::cl::desc("Load device configs used in Runtime"),
    llvm::cl::value_desc("profile.yaml"), llvm::cl::Optional,
    llvm::cl::cat(loaderCat));

/// Name of the network being bundled.
llvm::cl::opt<std::string> networkName(
    "network-name",
    llvm::cl::desc("Name of the network being bundled. This name is used as a "
                   "prefix for all the files that are generated."),
    llvm::cl::cat(loaderCat));

/// Name of the main entry of the bundle.
llvm::cl::opt<std::string>
    mainEntryName("main-entry-name",
                  llvm::cl::desc("Name of the main entry in the bundle. "
                                 "This name is used as the function name "
                                 "of the entry point to the network."),
                  llvm::cl::cat(loaderCat));

llvm::cl::opt<unsigned> numDevices("num-devices",
                                   llvm::cl::desc("Number of Devices to use"),
                                   llvm::cl::init(1), llvm::cl::value_desc("N"),
                                   llvm::cl::cat(loaderCat));
} // namespace

// timeOpt and iterationsOpt are outside the namespace so they can be used by
// the image-classifier.
llvm::cl::opt<bool>
    timeOpt("time",
            llvm::cl::desc("Print timer output to stderr detailing how long it "
                           "takes for the program to execute"),
            llvm::cl::Optional, llvm::cl::cat(loaderCat));

llvm::cl::opt<unsigned> iterationsOpt(
    "iterations", llvm::cl::desc("Number of iterations to perform"),
    llvm::cl::Optional, llvm::cl::init(0), llvm::cl::cat(loaderCat));

llvm::StringRef Loader::getModelOptPath() {
  assert(modelPathOpt.size() == 1 && "Model path must be a single path.");
  return modelPathOpt[0];
}

llvm::StringRef Loader::getModelOptDir() {
  assert(modelPathOpt.size() == 1 &&
         llvm::sys::fs::is_directory(*modelPathOpt.begin()) &&
         "Model path must be a single directory.");
  return modelPathOpt[0];
}

bool glow::emittingBundle() { return !emitBundle.empty(); }

bool glow::profilingGraph() { return !dumpProfileFileOpt.empty(); }

static bool commandLineIsInvalid() {
  if (!dumpProfileFileOpt.empty() &&
      (!loadProfileFileOpt.empty() || convertToFP16)) {
    llvm::errs() << "Loader: the -" << dumpProfileFileOpt.ArgStr
                 << " option cannot be specified at the same time as either -"
                 << loadProfileFileOpt.ArgStr << " or -" << convertToFP16.ArgStr
                 << ".\n";
    return true;
  }

  if (emitBundle.getNumOccurrences()) {
    if (networkName.getNumOccurrences()) {
      if (networkName.empty()) {
        llvm::errs() << "Loader: -" << networkName.ArgStr
                     << " must not be empty.\n";
        return true;
      } // FIXME: else make sure networkName does not have any sequence of
        // characters that could turn into evil stuff in the assembler.
    } else {
      // By default, use the last directory in the model path
      // as the name of the network.
      // Only do that when there is just one path specified.
      if (modelPathOpt.size() == 1) {
        for (auto it = llvm::sys::path::rbegin(modelPathOpt[0]),
                  end = llvm::sys::path::rend(modelPathOpt[0]);
             it != end; ++it) {
          networkName = *it;
          // Empty names are replaced by '.' (see Path.h in LLVM).
          if (!networkName.empty() && networkName != ".") {
            break;
          }
        }
      }
      if (networkName.empty()) {
        llvm::errs() << "Loader: Use -" << networkName.ArgStr
                     << " to specify a non-empty network name.\n";
        return true;
      }
    }
  } else if (networkName.getNumOccurrences()) {
    llvm::errs() << "Loader: -" << networkName.ArgStr
                 << " only makes sense when -" << emitBundle.ArgStr
                 << " is used.\n";
    return true;
  }
  return false;
}

void glow::parseCommandLine(int argc, char **argv) {
  llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);
  llvm::cl::ParseCommandLineOptions(
      argc, argv,
      " The Glow compiler\n\n"
      "Glow is a compiler for neural network accelerators.\n");

  if (commandLineIsInvalid()) {
    std::exit(1);
  }

  if (modelPathOpt.size() > 2) {
    llvm::errs() << "-model flag should have either 1 or 2 paths assigned. "
                    "Please see flag's description.\n";
    std::exit(1);
  }
}

/// Helper to get the Kind of a Node (e.g. Kinded::Kind::AddNodeKind) given its
/// \p nodeName (e.g. Add).
static Kinded::Kind getKindFromNodeName(llvm::StringRef nodeName) {
#define DEF_NODE(CLASS, NAME)                                                  \
  if (nodeName == #NAME) {                                                     \
    return Kinded::Kind::CLASS##Kind;                                          \
  }
#include "glow/AutoGenNodes.def"
  LOG(FATAL) << "Unknown node name: " << nodeName.str();
}

/// Helper to get the parameters in DeviceConfig from \p str. The \p str has
/// multiple lines, and each line with this format : "str1" : "str2".
static llvm::StringMap<std::string> getBackendParams(std::string &str) {
  llvm::StringMap<std::string> ret{};
  std::string s;
  std::istringstream f(str.c_str());
  while (getline(f, s, '\n')) {
    // Abstract the mapping from each line's string:
    // ""str1" : "str2"" => ret["str1"] = "str2";
    size_t pos1, pos2, pos3, pos4;
    pos1 = s.find('"');
    assert(pos1 != std::string::npos && "invalid string format");
    pos2 = s.find('"', pos1 + 1);
    assert(pos2 != std::string::npos && "invalid string format");
    pos3 = s.find('"', pos2 + 1);
    assert(pos3 != std::string::npos && "invalid string format");
    pos4 = s.find('"', pos3 + 1);
    assert(pos4 != std::string::npos && "invalid string format");
    ret[s.substr(pos1 + 1, pos2 - pos1 - 1)] =
        s.substr(pos3 + 1, pos4 - pos3 - 1);
  }
  return ret;
}

/// If the device config file \p loadDeviceDoncfigsFile available, load \p
/// configs from the file. Otherwise, create \p numDevices number of devices
/// based on \p backendName.
static std::vector<std::unique_ptr<runtime::DeviceConfig>>
generateDeviceConfigs(std::string &loadDeviceConfigsFile,
                      unsigned int numDevices, llvm::StringRef backendName) {
  std::vector<std::unique_ptr<runtime::DeviceConfig>> configs;
  if (loadDeviceConfigsFile.empty()) {
    // If there is no device config file, use numDevices to generate the
    // configs.
    for (unsigned int i = 0; i < numDevices; ++i) {
      auto config = llvm::make_unique<runtime::DeviceConfig>(backendName);
      configs.push_back(std::move(config));
    }
  } else {
    // Get the configs from the config file.
    std::vector<DeviceConfigHelper> lists;
    lists = deserializeDeviceConfigFromYaml(loadDeviceConfigsFile);
    for (unsigned int i = 0; i < lists.size(); ++i) {
      auto backendName = lists[i].backendName_;
      auto name = lists[i].name_;
      auto parameters = getBackendParams(lists[i].parameters_.str);
      auto config = llvm::make_unique<runtime::DeviceConfig>(backendName, name,
                                                             parameters);
      configs.push_back(std::move(config));
    }
  }
  return configs;
}

void Loader::compile(PlaceholderBindings &bindings) {
  // Dump the DAG before compilation if needed.
  if (!dumpGraphDAGFileBeforeCompilationOpt.empty()) {
    F_->dumpDAG(dumpGraphDAGFileBeforeCompilationOpt.c_str());
  }

  CompilationContext cctx{&bindings, &loweredMap_};
  PrecisionConfiguration &precConfig = cctx.precisionConfig;

  // Handle the request to profile the graph in preparation for quantization.
  if (!dumpProfileFileOpt.empty()) {
    precConfig.quantMode = QuantizationMode::Profile;

    // By default everything will be lowered for profiling. However this may
    // cause performance issues for some models, e.g. if a model has group
    // Convolutions which explode the size of the graph when lowered. Thus allow
    // for disabling certain NodeKinds for profiling. This means that during
    // quantization, these nodes should also not be lowered by the backend.
    for (llvm::StringRef kindName : doNotLowerNodesForProfilingOpt) {
      precConfig.precisionModeKindSet.insert(getKindFromNodeName(kindName));
    }
  } else {
    // By default, when converting models, all nodes that can be converted are
    // converted. However, some models may need to keep higher precision for
    // some nodes to prevent high accuracy loss. Those nodes are gathered via
    // the keepOriginalPrecisionForNodesOpt option and passed to the related
    // conversion function.
    for (llvm::StringRef kindName : keepOriginalPrecisionForNodesOpt) {
      precConfig.precisionModeKindSet.insert(getKindFromNodeName(kindName));
    }
  }

  if (!loadProfileFileOpt.empty()) {
    precConfig.quantMode = QuantizationMode::Quantize;
    precConfig.quantConfig.precision = quantizationPrecision;
    precConfig.quantConfig.infos = deserializeFromYaml(loadProfileFileOpt);
    precConfig.quantConfig.schema = quantizationSchema;
    precConfig.quantConfig.enableRowwise = enableRowwiseOpt;
    precConfig.quantConfig.assertAllNodesQuantized = assertAllNodesQuantizedOpt;
  }

  precConfig.convertToFP16 = convertToFP16;

  // Store a raw pointer to the Module, we pass the unique_ptr to HostManager
  // but the Module is stored by Hostmanager so the pointer will remain valid.
  auto module = M_.get();

  if (emittingBundle()) {
    // Emit IR for the graph, compile it and save as a bundle.
    auto error = ::glow::optimizeFunction(F_, *backend_, cctx);
    EXIT_ON_ERR(std::move(error));
    backend_->save(F_, emitBundle, networkName,
                   mainEntryName.empty() ? networkName : mainEntryName);
  } else {
    // Emit IR for the graph and compile it.
    auto error = hostManager_->addNetwork(std::move(M_), cctx);
    EXIT_ON_ERR(std::move(error));
  }
  if (dumpGraphOpt) {
    for (auto function : module->getFunctions()) {
      function->dump();
    }
  }
  if (!dumpGraphDAGFileOpt.empty()) {
    for (auto function : module->getFunctions()) {
      std::string filename =
          function->getName().str() + "_" + dumpGraphDAGFileOpt;
      function->dumpDAG(filename.c_str());
    }
  }
}

void Loader::runInference(PlaceholderBindings &bindings, size_t batchSize) {
  assert(!emittingBundle() &&
         "No inference is performed in the bundle generation mode.");
  unsigned iterations = iterationsOpt == 0 ? 1 : iterationsOpt;
  llvm::Timer timer("Infer", "Infer");
  if (timeOpt) {
    timer.startTimer();
  }
  for (unsigned i = 0; i < iterations; i++) {
    auto runErr = hostManager_->runNetworkBlocking(modelPathOpt[0], bindings);
    EXIT_ON_ERR(std::move(runErr));
  }
  if (timeOpt) {
    timer.stopTimer();
    llvm::outs() << llvm::formatv("Wall time per item (s): {0:f4}\n",
                                  timer.getTotalTime().getWallTime() /
                                      iterations / batchSize);
  }
}

void Loader::generateAndSerializeQuantizationInfos(
    PlaceholderBindings &bindings) {
  assert(!dumpProfileFileOpt.empty() &&
         "Filename to dump serialized profile to must not be empty.");
  std::vector<NodeQuantizationInfo> QI =
      quantization::generateNodeQuantizationInfos(
          bindings, F_, loweredMap_, quantizationSchema, quantizationPrecision);
  serializeToYaml(dumpProfileFileOpt, QI);
}

Loader::Loader() {
  if (modelPathOpt.size() == 1) {
    if (llvm::sys::fs::is_directory(*modelPathOpt.begin())) {
      caffe2NetDescFilename_ = modelPathOpt[0] + "/predict_net.pb";
      caffe2NetWeightFilename_ = modelPathOpt[0] + "/init_net.pb";
    } else {
      onnxModelFilename_ = modelPathOpt[0];
    }
  } else {
    caffe2NetDescFilename_ = modelPathOpt[0];
    caffe2NetWeightFilename_ = modelPathOpt[1];
  }
  M_.reset(new Module);

  std::vector<std::unique_ptr<runtime::DeviceConfig>> configs =
      generateDeviceConfigs(loadDeviceConfigsFileOpt, numDevices,
                            ExecutionBackend);

  hostManager_ = llvm::make_unique<runtime::HostManager>(std::move(configs));
  backend_ = createBackend(ExecutionBackend);
  F_ = M_->createFunction(modelPathOpt[0]);
  functionName_ = modelPathOpt[0];
}
