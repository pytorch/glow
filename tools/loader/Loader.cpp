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

#include "Loader.h"

#include "glow/Base/Image.h"
#include "glow/Base/Tensor.h"
#include "glow/Converter/TypeAToTypeBFunctionConverter.h"
#include "glow/IR/IR.h"
#include "glow/Importer/Caffe2ModelLoader.h"
#include "glow/Importer/ONNXModelLoader.h"
#include "glow/Importer/TFLiteModelLoader.h"
#include "glow/Optimizer/GraphOptimizer/CompilationContext.h"
#include "glow/Optimizer/GraphOptimizer/GraphOptimizer.h"
#include "glow/Quantization/Serialization.h"
#include "glow/Runtime/RuntimeTypes.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <future>
#include <sstream>

using namespace glow;

llvm::cl::OptionCategory loaderCat("Loader Options");

std::vector<std::string> modelPathOpt;
static llvm::cl::list<std::string, std::vector<std::string>> modelPathOptF(
    "model",
    llvm::cl::desc(
        "Specify one of three:\n"
        "1. Path to ONNX model file.\n"
        "2. Two paths to Caffe2 model files: network structure and weight.\n"
        "3. Path to directory with the Caffe2 network structure "
        "<predict_net.pb> and weight <init_net.pb> files."),
    llvm::cl::value_desc("modelPath"), llvm::cl::Required, llvm::cl::OneOrMore,
    llvm::cl::cat(loaderCat), llvm::cl::location(modelPathOpt));
static llvm::cl::alias modelPathAOpt("m", llvm::cl::desc("Alias for -model"),
                                     llvm::cl::aliasopt(modelPathOptF),
                                     llvm::cl::cat(loaderCat));

namespace {

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
                   "Use symmetric ranges with potentially uint8 ranges"),
        clEnumValN(quantization::Schema::SymmetricWithPower2Scale,
                   "symmetric_with_power2_scale",
                   "Use symmetric ranges with power of 2 scaling factor")),
    llvm::cl::init(quantization::Schema::Asymmetric), llvm::cl::cat(loaderCat));

llvm::cl::opt<quantization::Calibration> quantizationCalibrationOpt(
    "quantization-calibration",
    llvm::cl::desc("Specify which quantization calibration method to use"),
    llvm::cl::Optional,
    llvm::cl::values(
        clEnumValN(quantization::Calibration::None, "none", "No calibration"),
        clEnumValN(quantization::Calibration::KLMinimization, "KL",
                   "Quantization calibration method based on minimizing the "
                   "Kullback-Leibler divergence metric (relative entropy)")),
    llvm::cl::init(quantization::Calibration::None), llvm::cl::cat(loaderCat));

llvm::cl::opt<bool> calibrateConstantsOpt(
    "calibrate-constants",
    llvm::cl::desc("Option to enable the quantization calibration for constant "
                   "weights which is disabled by default."),
    llvm::cl::init(false), llvm::cl::Optional, llvm::cl::cat(loaderCat));

llvm::cl::opt<ElemKind> quantizationPrecision(
    "quantization-precision",
    llvm::cl::desc("Specify which quantization precision to use, e.g., Int8"),
    llvm::cl::Optional,
    llvm::cl::values(
        clEnumValN(ElemKind::Int8QTy, "Int8", "Use Int8 quantization"),
        clEnumValN(ElemKind::Int16QTy, "Int16", "Use Int16 quantization")),
    llvm::cl::init(ElemKind::Int8QTy), llvm::cl::cat(loaderCat));

llvm::cl::opt<ElemKind> quantizationPrecisionBias(
    "quantization-precision-bias",
    llvm::cl::desc("Specify which quantization precision to use for bias "
                   "of Convolution and Fully Connected nodes."),
    llvm::cl::Optional,
    llvm::cl::values(
        clEnumValN(ElemKind::Int8QTy, "Int8", "Use Int8 bias quantization"),
        clEnumValN(ElemKind::Int16QTy, "Int16", "Use Int16 bias quantization"),
        clEnumValN(ElemKind::Int32QTy, "Int32", "Use Int32 bias quantization")),
    llvm::cl::init(ElemKind::Int32QTy), llvm::cl::cat(loaderCat));

llvm::cl::opt<bool>
    enableRowwiseOpt("enable-rowwise",
                     llvm::cl::desc("Enable rowwise quantized FullyConnected."),
                     llvm::cl::Optional, llvm::cl::init(false),
                     llvm::cl::cat(loaderCat));

llvm::cl::opt<bool> enableChannelwiseOpt(
    "enable-channelwise",
    llvm::cl::desc("Enable channelwise quantized Convolution."),
    llvm::cl::Optional, llvm::cl::init(false), llvm::cl::cat(loaderCat));

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

llvm::cl::opt<PrecisionConfiguration::Float16Format> fp16Format(
    "fp16-format", llvm::cl::desc("fp16 format to use."),
    llvm::cl::values(clEnumValN(PrecisionConfiguration::Float16Format::FP16,
                                "fp16", "Use fp16"),
                     clEnumValN(PrecisionConfiguration::Float16Format::BFloat16,
                                "bfloat16", "Use bfloat16")),
    llvm::cl::init(PrecisionConfiguration::Float16Format::FP16),
    llvm::cl::cat(loaderCat));

llvm::cl::opt<bool> convertPlaceholdersOpt(
    "convert-placeholders",
    llvm::cl::desc("Convert model placeholders by merging ConvertTo, Quantize "
                   "and Dequantize nodes into the model inputs and outputs."),
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

llvm::cl::opt<unsigned> numHistogramBinsOpt(
    "num-histogram-bins",
    llvm::cl::desc("Number of bins used for histogram during profiling. If "
                   "histogram based calibration is used then the number of "
                   "histogram bins must be greater than 255 in order for any "
                   "calibration to take place (in the order of 1000's)."),
    llvm::cl::init(10), llvm::cl::value_desc("N"), llvm::cl::cat(loaderCat));

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

} // namespace

// These are outside the namespace so they can be used by the image-classifier.
std::vector<std::string> modelInputsOpt;
static llvm::cl::list<std::string, std::vector<std::string>> modelInputsOptF(
    "model-input", llvm::cl::ZeroOrMore, llvm::cl::location(modelInputsOpt),
    llvm::cl::desc(
        " For ONNX models the inputs of the graph can be inferred   \n"
        " automatically and hence this option is not mandatory.     \n"
        " For Caffe2 models the graph definition does not contain   \n"
        " the description of the inputs and hence must be provided  \n"
        " explicitly using this option. One or more model inputs    \n"
        " are provided using the following format:                  \n"
        "    -model-input=<inputName1>,<inputType1>,<inputShape1>   \n"
        "    -model-input=<inputName2>,<inputType2>,<inputShape2>   \n"
        "    ....................................................   \n"
        " For quantized types the format is slightly different since\n"
        " the scale and offset parameters should also be provided:  \n"
        "    -model-input=<name>,<type>,<scale>,<offset>,<shape>    \n"
        " For example we can can provide one or more inputs:        \n"
        "    -model-input=input_03_data,float,[1]                   \n"
        "    -model-input=data_bias,int32,[1,32,32]                 \n"
        "    -model-input=data,int8q,0.123,-13,[1,10]               \n"
        " If only the name is provided, the default type is 'float' \n"
        " and the default shape is '[1]':                           \n"
        "    -model-input=<inputName1>                              \n"
        " The supported types are:                                  \n"
        "    - float, float16 (floating point types)                \n"
        "    - int32, int64 (integer types)                         \n"
        "    - int8q, int16q, int32q (integer quantized types)      \n"
        "    - bool (logic type)\n"),
    llvm::cl::value_desc("name,[type,[scale,offset],shape]"),
    llvm::cl::cat(loaderCat));

llvm::cl::alias modelInputName("model-input-name",
                               llvm::cl::desc("Alias for -model-input"),
                               llvm::cl::aliasopt(modelInputsOptF),
                               llvm::cl::cat(loaderCat));

llvm::cl::opt<unsigned> numDevices("num-devices",
                                   llvm::cl::desc("Number of Devices to use"),
                                   llvm::cl::init(1), llvm::cl::value_desc("N"),
                                   llvm::cl::cat(loaderCat));

llvm::cl::opt<bool> runAllInputsOnAllDevices(
    "run-all-inputs-on-all-devices",
    llvm::cl::desc("Run all inputs on all devices. Used for testing purposes."),
    llvm::cl::init(false), llvm::cl::cat(loaderCat));

llvm::cl::opt<bool>
    timeOpt("time",
            llvm::cl::desc("Print timer output to stderr detailing how long it "
                           "takes for the program to execute"),
            llvm::cl::Optional, llvm::cl::cat(loaderCat));

llvm::cl::opt<unsigned> iterationsOpt(
    "iterations", llvm::cl::desc("Number of iterations to perform"),
    llvm::cl::Optional, llvm::cl::init(0), llvm::cl::cat(loaderCat));

std::string Loader::getModelOptPath() {
  // If given a single path, return it.
  if (modelPathOpt.size() == 1 &&
      llvm::sys::fs::is_directory(*modelPathOpt.begin())) {
    return *modelPathOpt.begin();
  }

  // Model path must be to one or more files. Use the path of the first file.
  size_t found = modelPathOpt[0].find_last_of("/");
  return found == std::string::npos ? "." : modelPathOpt[0].substr(0, found);
}

llvm::StringRef Loader::getModelOptDir() {
  assert(modelPathOpt.size() == 1 &&
         llvm::sys::fs::is_directory(*modelPathOpt.begin()) &&
         "Model path must be a single directory.");
  return modelPathOpt[0];
}

bool glow::emittingBundle() { return !emitBundle.empty(); }

bool glow::profilingGraph() { return !dumpProfileFileOpt.empty(); }

/// Parse the 'modelInputsOpt' option and get the model input names and types.
/// The expected format is one of the following:
/// - <name> (default type is 'float', default shape is '[1]')
/// - <name>,<type>,<shape> for non-quantized types.
/// - <name>,<type>,<scale>,<offset>,<shape> for quantized types.
static void getModelInputs(std::vector<std::string> &inputNames,
                           std::vector<Type> *inputTypes) {
  for (const auto &str : modelInputsOpt) {
    // Parse name.
    auto strPair = llvm::StringRef(str).split(',');
    llvm::StringRef name = strPair.first;
    CHECK(name.size()) << "Model input name empty";

    // Verify name is unique and add to vector.
    for (const auto &nameIter : inputNames) {
      if (name.equals(nameIter)) {
        LOG(FATAL) << strFormat("Model input name \"%s\" is not unique. Check "
                                "the graph definition for the input names.",
                                std::string(name).c_str());
      }
    }
    inputNames.push_back(name.str());

    if (!inputTypes) {
      continue;
    }

    // If only the name is provided, use the default type and shape.
    if (strPair.second.size() == 0) {
      inputTypes->push_back(Type(ElemKind::FloatTy, {1}));
      continue;
    }

    // Parse type.
    strPair = strPair.second.split(',');
    llvm::StringRef type = strPair.first;
    CHECK(type.size()) << "Model input type empty";
    ElemKind kind;
    if (type.equals("float")) {
      kind = ElemKind::FloatTy;
    } else if (type.equals("float16")) {
      kind = ElemKind::Float16Ty;
    } else if (type.equals("bfloat16")) {
      kind = ElemKind::BFloat16Ty;
    } else if (type.equals("int8q")) {
      kind = ElemKind::Int8QTy;
    } else if (type.equals("int16q")) {
      kind = ElemKind::Int16QTy;
    } else if (type.equals("int32q")) {
      kind = ElemKind::Int32QTy;
    } else if (type.equals("int32")) {
      kind = ElemKind::Int32ITy;
    } else if (type.equals("int64")) {
      kind = ElemKind::Int64ITy;
    } else if (type.equals("bool")) {
      kind = ElemKind::BoolTy;
    } else {
      LOG(FATAL) << strFormat("Model input type \"%s\" not supported",
                              std::string(type).c_str());
    }

    // For quantized type get scale and offset.
    double scale;
    int32_t offset;
    if (isQuantizedElemKind(kind)) {
      strPair = strPair.second.split(',');
      CHECK(strPair.first.size()) << "Model input scale empty";
      CHECK(!strPair.first.getAsDouble(scale))
          << "Model input scale parameter invalid";
      strPair = strPair.second.split(',');
      CHECK(strPair.first.size()) << "Model input offset empty";
      CHECK(!strPair.first.getAsInteger(0, offset))
          << "Model input offset parameter invalid";
    }

    // Parse shape string.
    llvm::StringRef shape = strPair.second;
    CHECK(shape.size()) << "Model input shape empty";
    ShapeVector dims;
    CHECK_EQ(shape.front(), '[') << "First shape char should be [";
    shape = shape.drop_front();
    CHECK_EQ(shape.back(), ']') << "First shape char should be ]";
    shape = shape.drop_back();
    CHECK(shape.size()) << "Model input shape empty";
    size_t val;
    while (shape.contains(',')) {
      auto splitRes = shape.split(',');
      CHECK(!splitRes.first.getAsInteger(0, val))
          << "Model input shape integer invalid";
      dims.push_back(val);
      shape = splitRes.second;
    }
    CHECK(!shape.getAsInteger(0, val)) << "Model input shape integer invalid";
    dims.push_back(val);

    // Build type and add to vector.
    if (isQuantizedElemKind(kind)) {
      inputTypes->push_back(Type(kind, dims, (float)scale, offset));
    } else {
      inputTypes->push_back(Type(kind, dims));
    }
  }
}

void Loader::loadModel(PlaceholderBindings *bindings,
                       llvm::ArrayRef<TypeRef> inputType) {

  // Get model input names and types.
  std::vector<std::string> inputNames;
  std::vector<Type> inputTypes;
  getModelInputs(inputNames, &inputTypes);
  std::vector<const char *> inputNameRefs;
  std::vector<TypeRef> inputTypeRefs;
  for (size_t idx = 0, e = inputNames.size(); idx < e; idx++) {
    inputNameRefs.push_back(inputNames[idx].c_str());
    inputTypeRefs.push_back(&inputTypes[idx]);
  }

  // Use explicit input type if given.
  if (inputType.size()) {
    inputTypeRefs = inputType;
  }

  // Load the model based on the model format.
  if (!getCaffe2NetDescFilename().empty()) {
    // For Caffe2 format the input placeholder names/types must be provided
    // explicitly (mandatory).
    std::unique_ptr<ProtobufLoader> protoLoader;
    protoLoader.reset(new Caffe2ModelLoader(
        getCaffe2NetDescFilename().str(), getCaffe2NetWeightFilename().str(),
        inputNameRefs, inputTypeRefs, *getFunction()));
    // Load the maps between original model names and the placeholders.
    inputPlaceholderByName_ = protoLoader->getInputVarsMapping();
    outputPlaceholderByName_ = protoLoader->getOutputVarsMapping();
    if (bindings) {
      postModelLoad(*bindings, *protoLoader.get(), outputPlaceholderByName_,
                    inputType);
    }
  } else if (!getTFLiteModelFilename().empty()) {
    // For TensorFlowLite format the input placeholder names/types are not
    // provided since are used directly from the model.
    auto tfliteLoader = glow::make_unique<TFLiteModelLoader>(
        getTFLiteModelFilename().str(), getFunction());
    // Load the maps between original model names and the placeholders.
    inputPlaceholderByName_ = tfliteLoader->getInputPlaceholderMap();
    outputPlaceholderByName_ = tfliteLoader->getOutputPlaceholderMap();
    // Since TensorFlowLite loader currently does not have the capability to
    // enforce the input type (for batching) we must validate that when the
    // input type is explicitly given it actually matches the model input type.
    if (bindings) {
      postModelLoad(*bindings, *tfliteLoader, outputPlaceholderByName_,
                    inputType);
    }
    if (inputType.size()) {
      CHECK(inputPlaceholderByName_.size() == 1)
          << "Model is expected to have only 1 input!";
      Placeholder *inpPH = inputPlaceholderByName_.begin()->second;
      auto modelBatchSize = inpPH->getType()->dims()[0];
      auto inputBatchSize = inputType[0]->dims()[0];
      CHECK(inputBatchSize == modelBatchSize)
          << "Mismatch between the model batch size (" << modelBatchSize
          << ") and the dataset batch size (" << inputBatchSize << ")! "
          << "If you are using the 'image-classifier' tool set the "
          << "dataset batch size with the option '-minibatch=" << modelBatchSize
          << "'!";
    }
  } else {
    // For ONNX format the input placeholders names/types can be optionally
    // provided but is not mandatory. If not provided (the arrays are empty)
    // they are derived automatically. One might want to provide explicitly
    // the input placeholder types in order to override the placeholder sizes
    // (one such example is the batch size).
    std::unique_ptr<ProtobufLoader> protoLoader;
    protoLoader.reset(new ONNXModelLoader(getOnnxModelFilename().str(),
                                          inputNameRefs, inputTypeRefs,
                                          *getFunction()));
    // Load the maps between original model names and the placeholders.
    inputPlaceholderByName_ = protoLoader->getInputVarsMapping();
    outputPlaceholderByName_ = protoLoader->getOutputVarsMapping();
    if (bindings) {
      postModelLoad(*bindings, *protoLoader.get(), outputPlaceholderByName_,
                    inputType);
    }
  }
}

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
          networkName = std::string(*it);
          // Strip extension (if any).
          size_t lastDotPos = networkName.find_last_of(".");
          if (lastDotPos != std::string::npos) {
            networkName = networkName.substr(0, lastDotPos);
          }
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

/// Clear external storage for cmd args defined in Loader.
static void initCmdArgVars() {
  llvm::cl::ResetAllOptionOccurrences();
  modelInputsOpt.clear();
  modelPathOpt.clear();
}

void glow::parseCommandLine(int argc, char **argv) {

  initCmdArgVars();

  llvm::cl::SetVersionPrinter([](llvm::raw_ostream &os) {
#ifdef GLOW_VERSION
    os << "Glow Tools version: " << GLOW_VERSION << "\n";
#endif
  });
  // TODO - registered once to avoid error:
  // "LLVM ERROR: too many signal callbacks already registered."
  static bool stackTraceRegistered = false;
  if (!stackTraceRegistered) {
    stackTraceRegistered = true;
    llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);
  }
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

quantization::QuantizationConfiguration Loader::getQuantizationConfiguration() {
  quantization::QuantizationConfiguration quantConfig;
  quantConfig.precision = quantizationPrecision;
  quantConfig.precisionBias = quantizationPrecisionBias;
  quantConfig.schema = quantizationSchema;
  quantConfig.calibration = quantizationCalibrationOpt;
  quantConfig.calibrateConstants = calibrateConstantsOpt;
  quantConfig.enableRowwise = enableRowwiseOpt;
  quantConfig.enableChannelwise = enableChannelwiseOpt;
  quantConfig.assertAllNodesQuantized = assertAllNodesQuantizedOpt;
  if (!loadProfileFileOpt.empty()) {
    auto fileExists = deserializeProfilingInfosFromYaml(
        loadProfileFileOpt, quantConfig.graphPreLowerHash, quantConfig.infos);
    CHECK(fileExists) << strFormat("Profile file \"%s\" does not exist!",
                                   loadProfileFileOpt.c_str());
  }
  quantConfig.checkGraphPreLowerHash = true;
  return quantConfig;
}

CompilationContext Loader::getCompilationContext(QuantizationMode mode) {

  // Common configurations.
  CompilationContext cctx;
  cctx.loweredInfoMap = &loweredMap_;
  PrecisionConfiguration &precConfig = cctx.precisionConfig;
  precConfig.convertToFP16 = convertToFP16;
  precConfig.float16Format = fp16Format;

  // Specific configurations.
  precConfig.quantMode = mode;
  if (mode == QuantizationMode::None) {

    // By default, when converting models, all nodes that can be converted are
    // converted. However, some models may need to keep higher precision for
    // some nodes to prevent high accuracy loss. Those nodes are gathered via
    // the keepOriginalPrecisionForNodesOpt option and passed to the related
    // conversion function.
    for (llvm::StringRef kindName : keepOriginalPrecisionForNodesOpt) {
      precConfig.precisionModeKindSet.insert(getKindFromNodeName(kindName));
    }

  } else if (mode == QuantizationMode::Quantize) {

    // By default, when converting models, all nodes that can be converted are
    // converted. However, some models may need to keep higher precision for
    // some nodes to prevent high accuracy loss. Those nodes are gathered via
    // the keepOriginalPrecisionForNodesOpt option and passed to the related
    // conversion function.
    for (llvm::StringRef kindName : keepOriginalPrecisionForNodesOpt) {
      precConfig.precisionModeKindSet.insert(getKindFromNodeName(kindName));
    }
    precConfig.quantConfig = getQuantizationConfiguration();

  } else if (mode == QuantizationMode::Profile) {

    // Profiling parameters.
    precConfig.profConfig.numHistogramBins = numHistogramBinsOpt;

    // By default everything will be lowered for profiling. However this may
    // cause performance issues for some models, e.g. if a model has group
    // Convolutions which explode the size of the graph when lowered. Thus allow
    // for disabling certain NodeKinds for profiling. This means that during
    // quantization, these nodes should also not be lowered by the backend.
    for (llvm::StringRef kindName : doNotLowerNodesForProfilingOpt) {
      precConfig.precisionModeKindSet.insert(getKindFromNodeName(kindName));
    }

  } else {
    LOG(FATAL) << "Quantization mode not supported";
  }

  // When converting the model placeholders, if the placeholders are already
  // allocated, we should also convert the backing tensors. Since this procedure
  // is not yet in place, we only convert when emitting a bundle.
  if (convertPlaceholdersOpt && !emittingBundle()) {
    llvm::errs() << "The flag 'convert-placeholders' can only be used when "
                    "emitting a bundle!\n";
    std::exit(1);
  }
  cctx.optimizationOpts.foldElemKindConversionIntoIO = convertPlaceholdersOpt;

  return cctx;
}

CompilationContext Loader::getCompilationContext() {
  if (!dumpProfileFileOpt.empty()) {
    return Loader::getCompilationContext(QuantizationMode::Profile);
  } else if (!loadProfileFileOpt.empty()) {
    return Loader::getCompilationContext(QuantizationMode::Quantize);
  } else {
    return Loader::getCompilationContext(QuantizationMode::None);
  }
}

void Loader::compile(PlaceholderBindings &bindings) {
  CompilationContext cctx = getCompilationContext();
  cctx.bindings = &bindings;
  compile(cctx);
}

void Loader::compile(CompilationContext &cctx) {

  // Dump the DAG before compilation if needed.
  if (!dumpGraphDAGFileBeforeCompilationOpt.empty()) {
    F_->dumpDAG(dumpGraphDAGFileBeforeCompilationOpt.c_str());
  }

  // Store a raw pointer to the Module, we pass the unique_ptr to HostManager
  // but the Module is stored by Hostmanager so the pointer will remain valid.
  auto module = M_.get();

  if (emittingBundle()) {
    // Create bundle directory if not exists.
    if (!llvm::sys::fs::is_directory(emitBundle)) {
      llvm::sys::fs::create_directory(emitBundle);
    }
    // Emit IR for the graph, compile it and save as a bundle. Replicate the
    // same optimizations seen during normal execution inside addNetwork().
    EXIT_ON_ERR(::glow::optimizeFunctionBeforeLowering(F_, cctx));
    EXIT_ON_ERR(::glow::optimizeFunction(F_, *backend_, cctx));
    backend_->save(F_, emitBundle, networkName,
                   mainEntryName.empty() ? networkName : mainEntryName);
  } else {
    // Emit IR for the graph and compile it.
    cctx.saturateHost = !runAllInputsOnAllDevices;
    auto error = hostManager_->addNetwork(std::move(M_), cctx);
    EXIT_ON_ERR(std::move(error));
    // After partitioning, the original function may be removed. Need to update
    // F_.
    F_ = module->getFunctions().front();
  }
  if (dumpGraphOpt) {
    for (auto function : module->getFunctions()) {
      function->dump();
    }
  }
  if (!dumpGraphDAGFileOpt.empty()) {
    for (auto function : module->getFunctions()) {
      std::string filename =
          function->getFilename() + "_" + dumpGraphDAGFileOpt;
      if (module->getFunctions().size() == 1) {
        filename = dumpGraphDAGFileOpt;
      }
      function->dumpDAG(filename.c_str());
    }
  }
  // Store compilation info in the Loader.
  compilationInfo_ = cctx.info;
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
    auto runErr = hostManager_->runNetworkBlocking(functionName_, bindings);
    EXIT_ON_ERR(std::move(runErr));
  }
  if (timeOpt) {
    timer.stopTimer();
    llvm::outs() << llvm::formatv("Wall time per item (s): {0:f4}\n",

                                  timer.getTotalTime().getWallTime() /
                                      iterations / batchSize);
  }
}

void Loader::runInference(ExecutionContext *context, size_t batchSize) {
  std::unique_ptr<ExecutionContext> contextP(context);

  unsigned iterations = iterationsOpt == 0 ? 1 : iterationsOpt;
  llvm::Timer timer("Infer", "Infer");
  if (timeOpt) {
    timer.startTimer();
  }

  for (unsigned i = 0; i < iterations; i++) {
    std::promise<void> runPromise;
    auto fut = runPromise.get_future();
    std::unique_ptr<Error> runErr;
    hostManager_->runNetwork(
        functionName_, std::move(contextP),
        [&runPromise, &runErr](runtime::RunIdentifierTy, Error err,
                               std::unique_ptr<ExecutionContext> contextPtr) {
          // Don't really delete context since we don't own it.
          contextPtr.release();

          runErr = glow::make_unique<Error>(std::move(err));
          runPromise.set_value();
        });
    fut.wait();
    EXIT_ON_ERR(std::move(*DCHECK_NOTNULL(runErr.get())));
  }
  if (timeOpt) {
    timer.stopTimer();
    llvm::outs() << llvm::formatv("Wall time per item (s): {0:f4}\n",
                                  timer.getTotalTime().getWallTime() /
                                      iterations / batchSize);
  }
}

static bool comparePI(const NodeProfilingInfo &a, const NodeProfilingInfo &b) {
  return (a.nodeOutputName_.compare(b.nodeOutputName_) < 0);
}

void Loader::generateAndSerializeProfilingInfos(PlaceholderBindings &bindings) {
  assert(!dumpProfileFileOpt.empty() &&
         "Filename to dump serialized profile to must not be empty.");
  std::vector<NodeProfilingInfo> PI;
  for (auto F : getModule()->getFunctions()) {
    std::vector<NodeProfilingInfo> tmp =
        quantization::generateNodeProfilingInfos(bindings, F, loweredMap_);
    PI.insert(PI.end(), tmp.begin(), tmp.end());
  }
  std::sort(PI.begin(), PI.end(), comparePI);
  serializeProfilingInfosToYaml(dumpProfileFileOpt,
                                compilationInfo_.graphPreLowerHash, PI);
}

Loader &Loader::registerExtension(std::unique_ptr<LoaderExtension> extension) {
  loaderExtensionList_.push_back(std::move(extension));
  return *this;
}

void Loader::postModelLoad(PlaceholderBindings &bindings,
                           ProtobufLoader &protoLoader,
                           llvm::StringMap<Placeholder *> &placeholderMap,
                           llvm::ArrayRef<TypeRef> inputImageType) {
  for (auto &&ext : loaderExtensionList_) {
    ext->postModelLoad(*this, bindings, protoLoader, placeholderMap,
                       inputImageType);
  }
}

void Loader::postModelLoad(PlaceholderBindings &bindings,
                           TFLiteModelLoader &tfloader,
                           llvm::StringMap<Placeholder *> &placeholderMap,
                           llvm::ArrayRef<TypeRef> inputImageType) {
  for (auto &&ext : loaderExtensionList_) {
    ext->postModelLoad(*this, bindings, tfloader, placeholderMap,
                       inputImageType);
  }
}

void Loader::inferInitMiniBatch(PlaceholderBindings &bindings,
                                size_t minibatchIndex, size_t minibatchSize) {
  for (auto &&ext : loaderExtensionList_) {
    ext->inferInitMiniBatch(*this, bindings, minibatchIndex, minibatchSize);
  }
}

void Loader::inferEndMiniBatch(PlaceholderBindings &bindings,
                               size_t minibatchIndex, size_t minibatchSize) {
  for (auto &&ext : loaderExtensionList_) {
    ext->inferEndMiniBatch(*this, bindings, minibatchIndex, minibatchSize);
  }
}

Loader::Loader(llvm::ArrayRef<size_t> configDeviceIDs) {
  if (modelPathOpt.size() == 1) {
    if (llvm::sys::fs::is_directory(*modelPathOpt.begin())) {
      caffe2NetDescFilename_ = modelPathOpt[0] + "/predict_net.pb";
      caffe2NetWeightFilename_ = modelPathOpt[0] + "/init_net.pb";
    } else {
      llvm::StringRef modelPath = modelPathOpt[0];
      if (modelPath.endswith("tflite")) {
        tfliteModelFilename_ = modelPath.str();
      } else {
        onnxModelFilename_ = modelPath.str();
      }
    }
  } else {
    caffe2NetDescFilename_ = modelPathOpt[0];
    caffe2NetWeightFilename_ = modelPathOpt[1];
  }
  M_.reset(new Module);

  std::vector<std::unique_ptr<runtime::DeviceConfig>> configs;

  if (configDeviceIDs.empty()) {
    configs = runtime::generateDeviceConfigs(numDevices, ExecutionBackend);
  } else {
    for (size_t ID : configDeviceIDs) {
      CHECK(ID < numDevices) << "IDs must be less than the number of devices";
      auto config = glow::make_unique<runtime::DeviceConfig>(ExecutionBackend);
      config->deviceID = ID;
      configs.push_back(std::move(config));
    }
  }

  hostManager_ = glow::make_unique<runtime::HostManager>(std::move(configs));
  backend_ = std::unique_ptr<Backend>(createBackend(ExecutionBackend));
  F_ = M_->createFunction(modelPathOpt[0]);
  functionName_ = modelPathOpt[0];
}
