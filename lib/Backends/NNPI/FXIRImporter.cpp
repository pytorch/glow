// Copyright 2004-present Facebook. All Rights Reserved.

#include "glow/lib/Backends/NNPI/FXIRImporter.h"
#include "glow/lib/Backends/NNPI/DebugMacros.h"
#include "nnpi_transformer_types.h"

using namespace utils;

namespace {

/// \returns true if \p opCode is not either "placeholder" or "output" (which
/// indicate the node is an operator).
static bool isOps(const std::string &opCode) {
  return opCode != "placeholder" && opCode != "output";
}

/// \returns the final char pointer to pass into NNPI importer APIs for \p str.
static const char *finalize(const std::string &str) {
  return str.empty() ? nullptr : str.c_str();
}

/// Node Importers
template <NNPI_ELTWISE_TYPE eltwiseType>
class BinaryEltwiseNodeImporter : public INNPIFXNodeImporter {
public:
  NNPIErrorCode
  importNode(const folly::dynamic &node,
             const std::function<string(string)> & /* getQualName */,
             FXNNPIImporter &importer) override {
    const auto &inputs = node["args"];
    const auto &name = node["name"].getString();

    // TODO: broadcast inputs if input is not a node.
    std::array<NNPIObjectName, 2> inputNames;
    snprintf(inputNames[0], NNPI_MAX_STRING_LEN, "%s",
             importer.getInputNodeName(inputs[0]).c_str());
    snprintf(inputNames[1], NNPI_MAX_STRING_LEN, "%s",
             importer.getInputNodeName(inputs[1]).c_str());

    importer.setUsedTensors({inputNames[0], inputNames[1]}, {name});
    return nnpiNetworkAddElementwiseOp(importer.getNetwork(), finalize(name),
                                       inputNames.data(), 2, finalize(name),
                                       eltwiseType);
  }
};

class LinearNodeImporter : public INNPIFXNodeImporter {
public:
  NNPIErrorCode
  importNode(const folly::dynamic &node,
             const std::function<string(string)> & /* getQualName */,
             FXNNPIImporter &importer) override {
    const auto &name = node["name"].getString();
    const auto &kwargs = node["kwargs"];
    const auto &inputName = importer.getInputNodeName(kwargs["input"]);
    const auto &weightName = importer.getInputNodeName(kwargs["weight"]);
    const auto &biasName =
        importer.getInputNodeName(kwargs["bias"], /* optional */ true);

    LOG_AND_RETURN_IF_NOT(ERROR, importer.isConstant(weightName),
                          "linear (" + name + ") weight (" + weightName +
                              ") must be constant",
                          NNPI_INVALID_PARAM);

    LOG_AND_RETURN_IF_NOT(
        ERROR, biasName.empty() || importer.isConstant(biasName),
        "linear (" + name + ") bias (" + biasName + ") must be constant",
        NNPI_INVALID_PARAM);

    importer.setUsedTensors({inputName, weightName, biasName}, {name});
    return nnpiNetworkAddFullyConnectedOp(
        importer.getNetwork(), finalize(name), finalize(inputName),
        finalize(name), finalize(weightName), finalize(biasName));
  }
};

template <size_t convDims = 2>
class ConvolutionNodeImporter : public INNPIFXNodeImporter {
public:
  NNPIErrorCode
  importNode(const folly::dynamic &node,
             const std::function<string(string)> & /* getQualName */,
             FXNNPIImporter &importer) override {
    const auto &inputs = node["kwargs"];
    const auto &name = node["name"].getString();
    const auto &inputName = importer.getInputNodeName(inputs["input"]);
    const auto &filterName = importer.getInputNodeName(inputs["weight"]);
    const auto &biasName =
        importer.getInputNodeName(inputs["bias"], /* optional */ true);

    // Kernel size is implicit in the shape of the filter.
    const auto &filterDesc = importer.getTensorDesc(filterName);
    LOG_AND_RETURN_IF_NOT(ERROR, filterDesc.numDims == convDims + 2,
                          "Unexpected filter dims", NNPI_INVALID_PARAM);
    std::vector<uint32_t> kernelSize;
    for (size_t i = 0; i < convDims; i++) {
      kernelSize.push_back(filterDesc.dims[i + 2]);
    }

    auto stride =
        toIntegerArray<uint32_t>(inputs["stride"], /* length */ convDims);
    auto padding =
        toIntegerArray<uint32_t>(inputs["padding"], /* length */ convDims);
    auto dilation =
        toIntegerArray<uint32_t>(inputs["dilation"], /* length */ convDims);
    const auto &groups = inputs["groups"].asInt();

    LOG_AND_RETURN_IF_NOT(ERROR,
                          std::all_of(dilation.cbegin(), dilation.cend(),
                                      [](int i) { return i == 1; }),
                          "Dilation is not supported", NNPI_INVALID_PARAM);
    LOG_AND_RETURN_IF_NOT(ERROR, groups == 1, "Group is not supported",
                          NNPI_INVALID_PARAM);

    LOG_AND_RETURN_IF_NOT(ERROR, importer.isConstant(filterName),
                          "convolution (" + name + ") filter (" + filterName +
                              ") must be constant",
                          NNPI_INVALID_PARAM);

    LOG_AND_RETURN_IF_NOT(
        ERROR, biasName.empty() || importer.isConstant(biasName),
        "convolution (" + name + ") bias (" + biasName + ") must be constant",
        NNPI_INVALID_PARAM);

    importer.setUsedTensors({inputName, filterName, biasName}, {name});
    return nnpiNetworkAddConvolutionOp(
        importer.getNetwork(), finalize(name), finalize(inputName),
        finalize(name), finalize(filterName), finalize(biasName),
        kernelSize.data(), padding.data(), padding.data(), stride.data(),
        dilation.data(), convDims, groups);
  }
};

class BatchNormalizationNodeImporter : public INNPIFXNodeImporter {
public:
  NNPIErrorCode importNode(const folly::dynamic &node,
                           const std::function<string(string)> &getQualName,
                           FXNNPIImporter &importer) override {
    const auto &name = node["name"].getString();
    const auto &kwargs = node["kwargs"];
    const auto &inputName = importer.getInputNodeName(kwargs["input"]);
    const auto &weightName = importer.getInputNodeName(kwargs["weight"]);
    const auto &biasName = importer.getInputNodeName(kwargs["bias"]);
    const auto &meanName = importer.getInputNodeName(kwargs["running_mean"]);
    const auto &varName = importer.getInputNodeName(kwargs["running_var"]);

    // Get parameters.
    const auto &eps = kwargs["eps"].asDouble();

    importer.setUsedTensors(
        {inputName, weightName, biasName, meanName, varName}, {name});
    return nnpiNetworkAddBatchNormOp(
        importer.getNetwork(), finalize(name), finalize(inputName),
        finalize(name), finalize(meanName), finalize(varName),
        finalize(weightName), finalize(biasName), eps);
  }
};

class ReluNodeImporter : public INNPIFXNodeImporter {
public:
  NNPIErrorCode
  importNode(const folly::dynamic &node,
             const std::function<string(string)> & /* getQualName */,
             FXNNPIImporter &importer) override {
    const auto &name = node["name"].getString();
    const auto &kwargs = node["kwargs"];
    const auto &inputName = importer.getInputNodeName(kwargs["input"]);

    // Get parameters.
    const auto &inplace = kwargs["inplace"].asBool();
    // TODO: replace users of ReLU input after ReLU with ReLU output.
    LOG_IF_NOT(WARNING, !inplace) << "Inplace ReLU is not supported";

    importer.setUsedTensors({inputName}, {name});
    return nnpiNetworkAddReluOp(importer.getNetwork(), finalize(name),
                                finalize(inputName), finalize(name));
  }
};

template <NNPI_POOLING_TYPE poolType>
class AdaptivePoolNodeImporter : public INNPIFXNodeImporter {
public:
  NNPIErrorCode
  importNode(const folly::dynamic &node,
             const std::function<string(string)> & /* getQualName */,
             FXNNPIImporter &importer) override {
    const auto &name = node["name"].getString();
    const auto &kwargs = node["kwargs"];
    const auto &inputName = importer.getInputNodeName(kwargs["input"]);

    importer.setUsedTensors({inputName}, {name});
    return nnpiNetworkAddAdaptivePoolingOp(importer.getNetwork(),
                                           finalize(name), finalize(inputName),
                                           finalize(name), poolType);
  }
};

template <NNPI_POOLING_TYPE poolType, size_t poolDims = 2>
class PoolNodeImporter : public INNPIFXNodeImporter {
public:
  NNPIErrorCode
  importNode(const folly::dynamic &node,
             const std::function<string(string)> & /* getQualName */,
             FXNNPIImporter &importer) override {
    const auto &name = node["name"].getString();
    const auto &kwargs = node["kwargs"];
    const auto &inputName = importer.getInputNodeName(kwargs["input"]);

    // Get parameters
    auto kernelSize = toIntegerArray<uint32_t>(kwargs["kernel_size"],
                                               /* length */ poolDims);
    auto stride = toIntegerArray<uint32_t>(kwargs["stride"],
                                           /* length */ poolDims);
    auto padding = toIntegerArray<uint32_t>(kwargs["padding"],
                                            /* length */ poolDims);
    const auto &ceilMode = kwargs["ceil_mode"].asBool();
    std::vector<uint32_t> dilation(poolDims, 1);
    auto returnIndices = false;
    bool countIncludePads = true;
    if (poolType == NNPI_POOL_AVG) {
      countIncludePads = kwargs["count_include_pad"].asBool();
    }
    if (poolType == NNPI_POOL_MAX) {
      returnIndices = kwargs["return_indices"].asBool();
      dilation = toIntegerArray<uint32_t>(kwargs["dilation"]);
    }

    LOG_AND_RETURN_IF_NOT(ERROR,
                          std::all_of(dilation.cbegin(), dilation.cend(),
                                      [](int i) { return i == 1; }),
                          "Dilation is not supported", NNPI_INVALID_PARAM);
    LOG_AND_RETURN_IF_NOT(ERROR, !returnIndices,
                          "Return_indices is not supported",
                          NNPI_INVALID_PARAM);

    importer.setUsedTensors({inputName}, {name});
    return nnpiNetworkAddPoolingOp(importer.getNetwork(), finalize(name),
                                   finalize(inputName), finalize(name),
                                   /* return indices tensor */ nullptr,
                                   kernelSize.data(), padding.data(),
                                   padding.data(), stride.data(), poolDims,
                                   poolType, !countIncludePads, ceilMode);
  }
};

class ReshapeNodeImporter : public INNPIFXNodeImporter {
public:
  NNPIErrorCode
  importNode(const folly::dynamic &node,
             const std::function<string(string)> & /* getQualName */,
             FXNNPIImporter &importer) override {
    const auto &inputs = node["kwargs"];
    const auto &name = node["name"].getString();
    const auto &inputName = importer.getInputNodeName(inputs["input"]);

    NNPITensorDesc desc;
    importer.updateDescDimsFromFX(
        toIntegerArray<glow::dim_t>(node["shape"].getString()), desc);
    importer.setUsedTensors({inputName}, {name});
    return nnpiNetworkAddReshapeOp(importer.getNetwork(), finalize(name),
                                   finalize(inputName), finalize(name), &desc);
  }
};

static std::unordered_map<std::string,
                          std::unique_ptr<INNPIFXNodeImporter>>::value_type
    FXImporterInit[] = {
        // _operator
        {"_operator.add",
         std::make_unique<BinaryEltwiseNodeImporter<NNPI_ELTWISE_ADD>>()},

        // torch
        {"torch.flatten", std::make_unique<ReshapeNodeImporter>()},

        // torch.nn.modules
        {"torch.nn.functional.linear", std::make_unique<LinearNodeImporter>()},
        {"torch.conv2d", std::make_unique<ConvolutionNodeImporter<2>>()},
        {"torch.nn.functional.batch_norm",
         std::make_unique<BatchNormalizationNodeImporter>()},
        {"torch.nn.functional.relu", std::make_unique<ReluNodeImporter>()},
        {"torch.nn.functional.adaptive_avg_pool2d",
         std::make_unique<AdaptivePoolNodeImporter<NNPI_POOL_AVG>>()},
        {"torch.nn.functional.max_pool2d",
         std::make_unique<PoolNodeImporter<NNPI_POOL_MAX, 2>>()},
};

static const std::unordered_map<std::string,
                                std::unique_ptr<INNPIFXNodeImporter>>
    FXNodeImporters = {std::make_move_iterator(std::begin(FXImporterInit)),
                       std::make_move_iterator(std::end(FXImporterInit))};

} // namespace

FXNNPIImporter::FXNNPIImporter(
    const glow::NNPICompilationOptions &compileOptions,
    const llvm::StringMap<const void *> &constants)
    : network_(NNPI_INVALID_NNPIHANDLE), compileOptions_(compileOptions),
      constants_(constants) {
  ASSERT_LOG_NNPI_ERROR(nnpiNetworkCreate(&network_),
                        "Failed to create NNPI network");
}

/// Destructor.
FXNNPIImporter::~FXNNPIImporter() {
  if (network_ != NNPI_INVALID_NNPIHANDLE) {
    LOG_NNPI_IF_ERROR(nnpiNetworkDestroy(network_),
                      "Failed to destroy NNPI network");
  }
}

const char *FXNNPIImporter::getConstantName(llvm::StringRef name) const {
  if (name.empty()) {
    return nullptr;
  }
  // If this is a Constant already, return the name back untouched.
  if (constants_.count(name)) {
    return name.data();
  }
  // Else return the name the getattr maps to if it exists.
  auto it = getattrs_.find(name);
  return it != getattrs_.end() ? it->second.c_str() : nullptr;
}

const void *FXNNPIImporter::getConstant(llvm::StringRef name) const {
  const char *baseName = getConstantName(name);
  if (!baseName) {
    return nullptr;
  }
  // There must be a constant with name baseName, so return it.
  auto it = constants_.find(baseName);
  CHECK(it != constants_.end())
      << "Should have found constant with name " << baseName;
  return it->second;
}

const NNPITensorDesc &
FXNNPIImporter::getTensorDesc(llvm::StringRef name) const {
  auto it = tensorDescs_.find(name);
  CHECK(it != tensorDescs_.end())
      << "Did not find NNPITensorDesc for " << name.str();
  return it->second;
}

const std::string &FXNNPIImporter::getInputNodeName(const folly::dynamic &node,
                                                    bool optional) const {
  if (node.isNull()) {
    CHECK(optional) << "Non-optional node must be non-null";
    static const std::string empty;
    return empty;
  }

  CHECK(node["is_node"].asBool()) << "Expected is_node";

  const auto &name = node["name"].getString();

  // Check if this name was for a getattr. If so, return the underlying Constant
  // name. Otherwise return the name unchanged.
  auto it = getattrs_.find(name);
  return it != getattrs_.end() ? it->second : name;
}

void FXNNPIImporter::updateDescQuantFromFX(const DTYPE &dtype,
                                           NNPITensorDesc &desc,
                                           const std::string &scaleTensor,
                                           const std::string &offsetTensor,
                                           bool forceSymlowp) {
  desc.quantParams.params.gemlowp.scale = 1.f;
  desc.quantParams.params.gemlowp.offset = 0;
  switch (dtype) {
  case DTYPE::FLOAT32:
    LOG_ERROR_IF_NOT((scaleTensor.empty() && offsetTensor.empty()))
        << "Scales and offsets provided for Float";
    desc.quantParams.precision = NNPI_PRECISION_FLOAT32;
    desc.quantParams.type = NNPI_QUANTIZATION_NONE;
    break;
  case DTYPE::FLOAT16:
    LOG_ERROR_IF_NOT((scaleTensor.empty() && offsetTensor.empty()))
        << "Scales and offsets provided for Float16";
    desc.quantParams.precision = NNPI_PRECISION_FLOAT16;
    desc.quantParams.type = NNPI_QUANTIZATION_NONE;
    break;
  case DTYPE::INT64:
    LOG_ERROR_IF_NOT((scaleTensor.empty() && offsetTensor.empty()))
        << "Scales and offsets provided for Int64";
    desc.quantParams.precision = NNPI_PRECISION_INT32;
    desc.quantParams.type = NNPI_QUANTIZATION_NONE;
    break;
  default:
    LOG(FATAL) << "Unhandled tensor data type";
  }
}

void FXNNPIImporter::updateDescDimsFromFX(
    const llvm::ArrayRef<glow::dim_t> &dims, NNPITensorDesc &desc) {
  desc.numDims = dims.size();
  for (size_t d = 0; d < desc.numDims; d++) {
    desc.dims[d] = dims[d];
  }
  switch (desc.numDims) {
  case 6:
  case 5:
  case 4:
    desc.layout = NNPI_LAYOUT_ANY;
    break;
  case 3:
    desc.layout = NNPI_LAYOUT_CHW;
    break;
  case 2:
    desc.layout = NNPI_LAYOUT_NC;
    break;
  case 1:
    desc.layout = NNPI_LAYOUT_C;
    break;
  case 0:
  default:
    LOG(FATAL) << "Invalid number of dims: " << desc.numDims;
  }
}

NNPIErrorCode
FXNNPIImporter::addTensor(const std::string &name, const string &dtypeStr,
                          const llvm::ArrayRef<glow::dim_t> dims, bool input,
                          bool output, const std::string &scaleTensor,
                          const std::string &offsetTensor, bool forceSymlowp) {
  const auto &dtypeElt = stringToDTYPE.find(dtypeStr);
  LOG_ERROR_IF_NOT(dtypeElt != stringToDTYPE.end())
      << dtypeStr << " is not supported!";
  const auto &dtype = dtypeElt->second;

  if (definedTensors_.count(name) && !forceSymlowp && !input && !output) {
    // The value was already defined and unless we're forcing a change in
    // layout/input/output/etc. Don't redefine it.
    return NNPI_NO_ERROR;
  } else {
    definedTensors_.insert(name);
  }

  NNPITensorDesc desc;
  desc.attributes.value = 0;
  desc.attributes.input = input;
  desc.attributes.output = output;
  updateDescQuantFromFX(dtype, desc, scaleTensor, offsetTensor,
                        forceSymlowp || compileOptions_.useSymlowp);
  updateDescDimsFromFX(dims, desc);

  const void *pRawData = getConstant(name);
  std::unique_ptr<int[]> pDataInt32;
  if (pRawData) {
    desc.attributes.constant = 1;
    switch (dtype) {
    case DTYPE::INT64: {
      auto it = constants_.find(name);
      CHECK(it != constants_.end()) << "Did not find constant for " << name;
      const auto *pDataInt64 = static_cast<const int64_t *>(it->second);
      const auto &size = std::accumulate(dims.begin(), dims.end(), 1,
                                         [](int x, int y) { return x * y; });
      pDataInt32 = std::make_unique<int[]>(size);
      for (size_t i = 0; i < size; i++) {
        pDataInt32[i] = static_cast<int32_t>(pDataInt64[i]);
      }
      pRawData = static_cast<void *>(pDataInt32.get());
      break;
    }
    default:
      break;
    }
  }

  // Outputs have already had their descs added by the named node itself.
  if (!output) {
    bool inserted = tensorDescs_.try_emplace(name, desc).second;
    CHECK(inserted) << "Already found tensor desc with name " << name;
  }

  return nnpiNetworkAddTensor(network_, finalize(name), &desc, pRawData);
}

NNPINetwork FXNNPIImporter::importFunction(const folly::dynamic &FXIR,
                                           const std::string &submodule) {
  const auto &mod = submodule.empty() ? FXIR : FXIR["modules"][submodule];
  const auto &prefix = submodule.empty() ? submodule : submodule + ".";

  // Clear internals.
  readTensors_.clear();
  writeTensors_.clear();
  definedTensors_.clear();
  DBG_MEM_USAGE("ImportFunction <<");

  // Add constants.
  const auto &weights = mod["weights"];
  for (const auto &key : weights.keys()) {
    const auto &name = key.getString();
    DBG("Importing Constant: " << name);
    CHECK(constants_.count(name)) << "Constant not found for weight " << name;
    LOG_NNPI_IF_ERROR_RETURN_INVALID_HANDLE(
        addTensor(
            name, weights[name]["dtype"].getString(),
            toIntegerArray<glow::dim_t>(weights[name]["shape"].getString())),
        "Failed to add constant");
  }

  // Add ops node.
  // TODO, currently we assume that the target of get_attr nodes matches the
  // name of the node, if they don't match we fail to load the graph.
  for (const auto &node : mod["nodes"]) {
    const auto &opCode = node["op_code"].getString();
    const auto &nodeName = node["name"].getString();
    if (!isOps(opCode)) {
      continue;
    }
    DBG("Importing Node: " << nodeName);
    // Add node outputs.
    LOG_NNPI_IF_ERROR_RETURN_INVALID_HANDLE(
        addTensor(nodeName, node["dtype"].getString(),
                  toIntegerArray<glow::dim_t>(node["shape"].getString())),
        "Failed to add intermediate");

    // Track what Constant each get_attr points to.
    if (opCode == "get_attr") {
      bool inserted =
          getattrs_.try_emplace(nodeName, node["target"].getString()).second;
      CHECK(inserted) << "Already mapped a getattr by name " << nodeName
                      << " to its underlying Constant";
      continue;
    }
    const auto &targetName = node["target"].getString();
    const auto &functionName = opCode != "call_module"
                                   ? targetName
                                   : node["parameters"]["name"].getString();
    // Import node.
    LOG_NNPI_IF_ERROR_RETURN_INVALID_HANDLE(
        FXNodeImporters.at(functionName)
            ->importNode(
                node,
                [&prefix, &targetName](const std::string &name) {
                  return folly::to<std::string>(prefix, targetName, ".", name);
                },
                *this),
        "Failed to import node");
  }

  // Add placeholder.
  for (const auto &node : mod["nodes"]) {
    const auto &opCode = node["op_code"].getString();

    if (opCode == "placeholder") {
      const auto &name = node["name"].getString();

      DBG("Add placeholder: " << name);
      CHECK(!writeTensors_.count(name)) << "Placeholder can't be written";

      if (readTensors_.count(name)) {
        LOG_NNPI_IF_ERROR_RETURN_INVALID_HANDLE(
            addTensor(name, node["dtype"].getString(),
                      toIntegerArray<glow::dim_t>(node["shape"].getString()),
                      /* input */ true, /* output */ false),
            "Failed to add placeholder");
      } else {
        DBG("[--IO--] Unused Placeholder: " << name);
      }
    } else if (opCode == "output") {
      const auto &args = node["args"];

      for (const auto &arg : args) {
        const auto &outputName = getInputNodeName(arg);

        DBG("Add output" << outputName);
        CHECK(writeTensors_.count(outputName))
            << "output must be in writeTensors_";

        LOG_NNPI_IF_ERROR_RETURN_INVALID_HANDLE(
            addTensor(outputName, arg["dtype"].getString(),
                      toIntegerArray<glow::dim_t>(arg["shape"].getString()),
                      /* input */ false, /* output */ true),
            "Failed to add output");
      }
    }
  }

  DBG_MEM_USAGE("ImportFunction call nnpiNetworkBuild");

  // Build network.
  NNPINetwork net;
  NNPIErrorCode res = nnpiNetworkBuild(network_);
  if (res != NNPI_NO_ERROR) {
    LOG(ERROR) << "Failed to build network";
    LOG_NNPI_IF_ERROR(nnpiNetworkDestroy(network_),
                      "Failed to destroy NNPI network");
    net = NNPI_INVALID_NNPIHANDLE;
  } else {
    net = network_;
  }

  // Detach network from importer (if failed to build then it's already
  // destroyed, otherwise relinquish ownership to the backend).
  network_ = NNPI_INVALID_NNPIHANDLE;
  DBG_MEM_USAGE("ImportFunction done >>");
  return net;
}
