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
#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/IR/IR.h"
#include "glow/Quantization/Quantization.h"
#include "glow/Quantization/Serialization.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/raw_ostream.h"

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

llvm::cl::opt<bool>
    timeOpt("time",
            llvm::cl::desc("Print timer output to stderr detailing how long it "
                           "takes for the program to execute"),
            llvm::cl::Optional, llvm::cl::cat(loaderCat));

llvm::cl::opt<unsigned> iterationsOpt(
    "iterations", llvm::cl::desc("Number of iterations to perform"),
    llvm::cl::Optional, llvm::cl::init(1), llvm::cl::cat(loaderCat));

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

llvm::cl::opt<BackendKind> ExecutionBackend(
    llvm::cl::desc("Backend to use:"),
    llvm::cl::values(clEnumValN(BackendKind::Interpreter, "interpreter",
                                "Use interpreter"),
                     clEnumValN(BackendKind::CPU, "cpu", "Use CPU"),
                     clEnumValN(BackendKind::OpenCL, "opencl", "Use OpenCL"),
                     clEnumValN(BackendKind::Habana, "habana", "Use Habana")),
    llvm::cl::init(BackendKind::Interpreter), llvm::cl::cat(loaderCat));

/// Debugging options.
llvm::cl::OptionCategory
    modelExportCat("How to export the Glow Intermediate Representation/Graphs",
                   "These options are for debugging the "
                   "graphs by writing the IR/Graphs to "
                   "given files/stdout");

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

/// Name of the network being bundled.
llvm::cl::opt<std::string> networkName(
    "network-name",
    llvm::cl::desc("Name of the network being bundled. "
                   "This name is used as both the function name "
                   "of the entry point to the network "
                   "and as a prefix for all the files that are generated."),
    llvm::cl::cat(loaderCat));
} // namespace

llvm::StringRef Loader::getModelOptPath() {
  assert(modelPathOpt.size() == 1 &&
         llvm::sys::fs::is_directory(*modelPathOpt.begin()) &&
         "Model path must be a single directory.");
  return modelPathOpt[0];
}

bool glow::emittingBundle() { return !emitBundle.empty(); }

bool glow::profilingGraph() { return !dumpProfileFileOpt.empty(); }

static bool commandLineIsInvalid() {
  if (!dumpProfileFileOpt.empty() && !loadProfileFileOpt.empty()) {
    llvm::errs() << "Loader: the -" << dumpProfileFileOpt.ArgStr << " and -"
                 << loadProfileFileOpt.ArgStr
                 << " options may not be specified together.\n";
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

/// Helper to get the Kind of a Node (e.g. Kinded::Kind::AddNodeKind) given its
/// \p nodeName (e.g. Add).
static Kinded::Kind getKindFromNodeName(llvm::StringRef nodeName) {
#define DEF_NODE(CLASS, NAME)                                                  \
  if (nodeName == #NAME) {                                                     \
    return Kinded::Kind::CLASS##Kind;                                          \
  }
#include "glow/AutoGenNodes.def"
  GLOW_UNREACHABLE("Unknown node name.");
}

void Loader::compile(PlaceholderBindings &bindings) {
  // Fold low-level operators into higher-level operators.
  // This is useful when compiling an input model where some high-level
  // operators have been lowered (this can be for instance a side effect of
  // model converters, like converters from Tensorflow to ONNX). In this
  // situation, such folding can then enable more optimizations and also improve
  // the performance backends that support natively such high-level operators.
  ::fold(F_, glow::CompilationMode::Infer);

  // Handle the request to profile the graph in preparation for quantization.
  if (!dumpProfileFileOpt.empty()) {
    // Perform the high-level optimizations before instrumenting the graph. This
    // optimization phase will remove stuff like repetitive transpose operations
    // perform CSE, etc.
    ::optimize(F_, glow::CompilationMode::Infer);

    // By default everything will be lowered for profiling. However this may
    // cause performance issues for some models, e.g. if a model has group
    // Convolutions which explode the size of the graph when lowered. Thus allow
    // for disabling certain NodeKinds for profiling. This means that during
    // quantization, these nodes should also not be lowered by the backend.
    KindSet doNotLowerNodesForProfiling;
    for (llvm::StringRef kindName : doNotLowerNodesForProfilingOpt) {
      doNotLowerNodesForProfiling.insert(getKindFromNodeName(kindName));
    }

    // Lower everything, keeping track of what NodeValues were lowered to other
    // NodeValues via the loweredMap_. This loweredMap_ is passed to
    // generateNodeQuantizationInfos() when writing out the profile, allowing
    // for both lowered and unlowered NodeValues to find their quantization
    // parameters.
    ::lower(F_, &loweredMap_, /* backend */ nullptr,
            doNotLowerNodesForProfiling);

    // Instrument the graph to capture profiles for nodes' outputs.
    F_ = ::profileQuantization(bindings, F_);
  }

  // By default, when converting models, all nodes that can be
  // converted are converted. However, some models may need to
  // keep higher precision for some nodes to prevent high accuracy loss.
  // Those nodes are gathered via the keepOriginalPrecisionForNodesOpt
  // option and passed to the related conversion function.
  KindSet keepOriginalPrecisionForNodes;
  for (llvm::StringRef kindName : keepOriginalPrecisionForNodesOpt) {
    keepOriginalPrecisionForNodes.insert(getKindFromNodeName(kindName));
  }

  // Load the quantization profile and transform the graph.
  if (!loadProfileFileOpt.empty()) {
    // The profiled graph was optimized before it was instrumentated. In this
    // part of the code we repeat the same transformation in order to create
    // the same graph structure.
    ::optimize(F_, glow::CompilationMode::Infer);

    // Lower as the backend prefers. When generating the profile everything was
    // lowered, however all lowered and unlowered components have a profile, and
    // so the backend can lower however it prefers and always find all of its
    // NodeValue's quantization parameters.
    ::lower(F_, &loweredMap_, EE_.getBackend());

    quantization::QuantizationConfiguration quantConfig{
        deserializeFromYaml(loadProfileFileOpt)};

    // In AOT compilation mode the name of the symbol depends on the name of the
    // function. Our tutorial expects the quantized name to be identical to the
    // original name, so we rename the floating-point network and give the
    // quantized network a new name.
    quantConfig.newFuncName = F_->getName();
    F_->setName("old");

    // Quantize the graph based on the captured profile.
    quantConfig.precision = quantizationPrecision;
    quantConfig.schema = quantizationSchema;
    quantConfig.enableRowwise = enableRowwiseOpt;
    quantConfig.assertAllNodesQuantized = assertAllNodesQuantizedOpt;

    auto *Q = quantization::quantizeFunction(F_, quantConfig, *EE_.getBackend(),
                                             loweredMap_,
                                             keepOriginalPrecisionForNodes);

    // Erase the original function so that the redundant variables that are only
    // referenced by the original function will be removed.
    Q->getParent()->eraseFunction(F_);
    F_ = Q;
  }

  if (convertToFP16) {
    TypeAToTypeBFunctionConverter converter(*F_, ElemKind::FloatTy,
                                            ElemKind::Float16Ty,
                                            &keepOriginalPrecisionForNodes);
    converter.convert();
    ::optimize(F_, glow::CompilationMode::Infer);
  }

  CompilationContext cctx;
  cctx.mode = CompilationMode::Infer;
  if (emittingBundle()) {
    // Emit IR for the graph, compile it and save as a bundle.
    EE_.save(F_, cctx, emitBundle, networkName);
  } else {
    // Emit IR for the graph and compile it.
    EE_.compile(F_, cctx);
  }

  if (dumpGraphOpt) {
    F_->dump();
  }
  if (!dumpGraphDAGFileOpt.empty()) {
    F_->dumpDAG(dumpGraphDAGFileOpt.c_str());
  }
}

void Loader::runInference(PlaceholderBindings &bindings, size_t batchSize) {
  assert(!emittingBundle() &&
         "No inference is performed in the bundle generation mode.");

  llvm::Timer timer("Infer", "Infer");
  if (timeOpt) {
    timer.startTimer();
  }
  for (unsigned i = 0; i < iterationsOpt; i++) {
    EE_.run(bindings);
  }
  if (timeOpt) {
    timer.stopTimer();
    llvm::outs() << llvm::formatv("Wall time per item (s): {0:f4}\n",
                                  timer.getTotalTime().getWallTime() /
                                      iterationsOpt / batchSize);
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

Loader::Loader(int argc, char **argv) {
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

  EE_.setBackend(ExecutionBackend);
  F_ = EE_.getModule().createFunction(modelPathOpt[0]);
}
