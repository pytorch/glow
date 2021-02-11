// Copyright 2004-present Facebook. All Rights Reserved.

#include "glow/lib/Backends/NNPI/FXIRImporter.h"
#include "glow/lib/Backends/NNPI/DebugMacros.h"
#include "nnpi_transformer_types.h"

using namespace utils;

namespace {

/// \returns true if \p opCode is not either "placeholder" or "output" (which
/// indicate the node is an operater).
bool isOps(const std::string &opCode) {
  return opCode != "placeholder" && opCode != "output";
}

// Node Importers
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
             inputs[0]["name"].getString().c_str());
    snprintf(inputNames[1], NNPI_MAX_STRING_LEN, "%s",
             inputs[1]["name"].getString().c_str());

    importer.setUsedTensors({inputNames[0], inputNames[1]}, {name});
    return nnpiNetworkAddElementwiseOp(importer.getNetwork(), name.c_str(),
                                       inputNames.data(), 2, name.c_str(),
                                       eltwiseType);
  }
};

class LinearNodeImporter : public INNPIFXNodeImporter {
public:
  NNPIErrorCode importNode(const folly::dynamic &node,
                           const std::function<string(string)> &getQualName,
                           FXNNPIImporter &importer) override {
    const auto &inputs = node["args"];
    const auto &name = node["name"].getString();
    const auto &weightName = getQualName("weight");
    const auto &biasName = getQualName("bias");

    importer.setUsedTensors(
        {inputs[0]["name"].getString(), weightName, biasName}, {name});
    return nnpiNetworkAddFullyConnectedOp(
        importer.getNetwork(), name.c_str(),
        inputs[0]["name"].getString().c_str(), name.c_str(), weightName.c_str(),
        importer.getConstant(biasName) ? biasName.c_str() : nullptr);
  }
};

template <size_t convDims = 2>
class ConvolutionNodeImporter : public INNPIFXNodeImporter {
public:
  NNPIErrorCode importNode(const folly::dynamic &node,
                           const std::function<string(string)> &getQualName,
                           FXNNPIImporter &importer) override {
    const auto &inputs = node["args"];
    const auto &name = node["name"].getString();
    const auto &filterName = getQualName("weight");
    const auto &biasName = getQualName("bias");

    // Get parameters.
    auto kernelSize =
        toIntegerArray<uint32_t>(node["parameters"]["kernel_size"].getString(),
                                 /* length */ convDims);
    auto stride = toIntegerArray<uint32_t>(
        node["parameters"]["stride"].getString(), /* length */ convDims);
    auto padding =
        toIntegerArray<uint32_t>(node["parameters"]["padding"].getString(),
                                 /* length */ convDims);
    auto dilation =
        toIntegerArray<uint32_t>(node["parameters"]["dilation"].getString(),
                                 /* length */ convDims);
    const auto &groups = node["parameters"]["groups"].asInt();
    const auto &paddingMode = node["parameters"]["padding_mode"].getString();

    LOG_AND_RETURN_IF_NOT(ERROR,
                          std::all_of(dilation.cbegin(), dilation.cend(),
                                      [](int i) { return i == 1; }),
                          "Dilation is not supported", NNPI_INVALID_PARAM);
    LOG_AND_RETURN_IF_NOT(ERROR, groups == 1, "Group is not supported",
                          NNPI_INVALID_PARAM);
    LOG_AND_RETURN_IF_NOT(ERROR, paddingMode == "zeros",
                          "Only support zeros padding mode",
                          NNPI_INVALID_PARAM);

    importer.setUsedTensors(
        {inputs[0]["name"].getString(), filterName, biasName}, {name});
    return nnpiNetworkAddConvolutionOp(
        importer.getNetwork(), name.c_str(),
        inputs[0]["name"].getString().c_str(), name.c_str(), filterName.c_str(),
        importer.getConstant(biasName) ? biasName.c_str() : nullptr,
        kernelSize.data(), padding.data(), padding.data(), stride.data(),
        dilation.data(), convDims, groups);
  }
};

class BatchNormalizationNodeImporter : public INNPIFXNodeImporter {
public:
  NNPIErrorCode importNode(const folly::dynamic &node,
                           const std::function<string(string)> &getQualName,
                           FXNNPIImporter &importer) override {
    const auto &inputs = node["args"];
    const auto &name = node["name"].getString();
    const auto &weightName = getQualName("weight");
    const auto &biasName = getQualName("bias");
    const auto &meanName = getQualName("running_mean");
    const auto &varName = getQualName("running_var");

    // Get parameters.
    const auto &eps = node["parameters"]["eps"].asDouble();

    importer.setUsedTensors({inputs[0]["name"].getString(), weightName,
                             biasName, meanName, varName},
                            {name});
    return nnpiNetworkAddBatchNormOp(
        importer.getNetwork(), name.c_str(),
        inputs[0]["name"].getString().c_str(), name.c_str(), meanName.c_str(),
        varName.c_str(), weightName.c_str(), biasName.c_str(), eps);
  }
};

class ReluNodeImporter : public INNPIFXNodeImporter {
public:
  NNPIErrorCode
  importNode(const folly::dynamic &node,
             const std::function<string(string)> & /* getQualName */,
             FXNNPIImporter &importer) override {
    const auto &inputs = node["args"];
    const auto &name = node["name"].getString();

    // Get parameters.
    const auto &inplace_ = node["parameters"]["inplace"].asBool();
    // TODO: replace users of ReLU input after ReLU with ReLU output.
    LOG_IF_NOT(WARNING, !inplace_) << "Inplace ReLU is not supported";

    importer.setUsedTensors({inputs[0]["name"].getString()}, {name});
    return nnpiNetworkAddReluOp(importer.getNetwork(), name.c_str(),
                                inputs[0]["name"].getString().c_str(),
                                name.c_str());
  }
};

template <NNPI_POOLING_TYPE poolType>
class AdaptivePoolNodeImporter : public INNPIFXNodeImporter {
public:
  NNPIErrorCode
  importNode(const folly::dynamic &node,
             const std::function<string(string)> & /* getQualName */,
             FXNNPIImporter &importer) override {
    const auto &inputs = node["args"];
    const auto &name = node["name"].getString();

    importer.setUsedTensors({inputs[0]["name"].getString()}, {name});
    return nnpiNetworkAddAdaptivePoolingOp(
        importer.getNetwork(), name.c_str(),
        inputs[0]["name"].getString().c_str(), name.c_str(), poolType);
  }
};

template <NNPI_POOLING_TYPE poolType, size_t poolDims = 2>
class PoolNodeImporter : public INNPIFXNodeImporter {
public:
  NNPIErrorCode
  importNode(const folly::dynamic &node,
             const std::function<string(string)> & /* getQualName */,
             FXNNPIImporter &importer) override {
    const auto &inputs = node["args"];
    const auto &name = node["name"].getString();

    // Get parameters
    auto kernelSize =
        toIntegerArray<uint32_t>(node["parameters"]["kernel_size"].getString(),
                                 /* length */ poolDims);
    auto stride = toIntegerArray<uint32_t>(
        node["parameters"]["stride"].getString(), /* length */ poolDims);
    auto padding =
        toIntegerArray<uint32_t>(node["parameters"]["padding"].getString(),
                                 /* length */ poolDims);
    const auto &ceilMode = node["parameters"]["ceil_mode"].asBool();
    std::vector<uint32_t> dilation(poolDims, 1);
    auto returnIndices = false;
    bool countIncludePads = true;
    if (poolType == NNPI_POOL_AVG) {
      countIncludePads = node["parameters"]["count_include_pad"].asBool();
    }
    if (poolType == NNPI_POOL_MAX) {
      returnIndices = node["parameters"]["return_indices"].asBool();
      dilation =
          toIntegerArray<uint32_t>(node["parameters"]["dilation"].getString());
    }

    LOG_AND_RETURN_IF_NOT(ERROR,
                          std::all_of(dilation.cbegin(), dilation.cend(),
                                      [](int i) { return i == 1; }),
                          "Dilation is not supported", NNPI_INVALID_PARAM);
    LOG_AND_RETURN_IF_NOT(ERROR, !returnIndices,
                          "Return_indices is not supported",
                          NNPI_INVALID_PARAM);

    importer.setUsedTensors({inputs[0]["name"].getString()}, {name});
    return nnpiNetworkAddPoolingOp(
        importer.getNetwork(), name.c_str(),
        inputs[0]["name"].getString().c_str(), name.c_str(),
        /* return indices tensor */ nullptr, kernelSize.data(), padding.data(),
        padding.data(), stride.data(), poolDims, poolType, !countIncludePads,
        ceilMode);
  }
};

class ReshapeNodeImporter : public INNPIFXNodeImporter {
public:
  NNPIErrorCode
  importNode(const folly::dynamic &node,
             const std::function<string(string)> & /* getQualName */,
             FXNNPIImporter &importer) override {
    const auto &inputs = node["args"];
    const auto &name = node["name"].getString();

    NNPITensorDesc desc;
    importer.updateDescDimsFromFX(
        toIntegerArray<glow::dim_t>(node["shape"].getString()), desc);
    importer.setUsedTensors({inputs[0]["name"].getString()}, {name});
    return nnpiNetworkAddReshapeOp(importer.getNetwork(), name.c_str(),
                                   inputs[0]["name"].getString().c_str(),
                                   name.c_str(), &desc);
  }
};

std::unordered_map<
    std::string,
    std::unique_ptr<INNPIFXNodeImporter>>::value_type FXImporterInit[] = {
    // _operator
    {"_operator.add",
     std::make_unique<BinaryEltwiseNodeImporter<NNPI_ELTWISE_ADD>>()},

    // torch
    {"torch.flatten", std::make_unique<ReshapeNodeImporter>()},

    // torch.nn.modules
    {"torch.nn.modules.linear.Linear", std::make_unique<LinearNodeImporter>()},
    {"torch.nn.modules.conv.Conv2d",
     std::make_unique<ConvolutionNodeImporter<2>>()},
    {"torch.nn.modules.batchnorm.BatchNorm2d",
     std::make_unique<BatchNormalizationNodeImporter>()},
    {"torch.nn.modules.activation.ReLU", std::make_unique<ReluNodeImporter>()},
    {"torch.nn.modules.pooling.AdaptiveAvgPool2d",
     std::make_unique<AdaptivePoolNodeImporter<NNPI_POOL_AVG>>()},
    {"torch.nn.modules.pooling.MaxPool2d",
     std::make_unique<PoolNodeImporter<NNPI_POOL_MAX, 2>>()},
};

const std::unordered_map<std::string, std::unique_ptr<INNPIFXNodeImporter>>
    FXNodeImporters = {std::make_move_iterator(std::begin(FXImporterInit)),
                       std::make_move_iterator(std::end(FXImporterInit))};

} // namespace

FXNNPIImporter::FXNNPIImporter(
    const glow::NNPICompilationOptions &compileOptions)
    : network_(NNPI_INVALID_NNPIHANDLE), compileOptions_(compileOptions) {
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

const void *FXNNPIImporter::getConstant(const std::string &name) const {
  return constants_->count(name) ? constants_->find(name)->second : nullptr;
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
      const auto *pDataInt64 =
          static_cast<const int64_t *>(constants_->find(name)->second);
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

  return nnpiNetworkAddTensor(network_, name.c_str(), &desc, pRawData);
}

NNPINetwork
FXNNPIImporter::importFunction(const folly::dynamic &FXIR,
                               const std::string &submodule,
                               const llvm::StringMap<const void *> &constants) {
  const auto &mod = submodule.empty() ? FXIR : FXIR["modules"][submodule];
  const auto &prefix = submodule.empty() ? submodule : submodule + ".";

  // Clear internals.
  readTensors_.clear();
  writeTensors_.clear();
  definedTensors_.clear();
  DBG_MEM_USAGE("ImportFunction <<");

  // Add constants.
  constants_ = &constants;
  const auto &weights = mod["weights"];
  for (const auto &key : weights.keys()) {
    const auto &name = key.getString();
    DBG("Importing Constant: " << name);
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
    if (isOps(opCode)) {
      DBG("Importing Node: " << node["name"].getString());
      // Add node outputs.
      LOG_NNPI_IF_ERROR_RETURN_INVALID_HANDLE(
          addTensor(node["name"].getString(), node["dtype"].getString(),
                    toIntegerArray<glow::dim_t>(node["shape"].getString())),
          "Failed to add intermediate");

      if (opCode != "get_attr") {
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
                      return folly::to<std::string>(prefix, targetName, ".",
                                                    name);
                    },
                    *this),
            "Failed to import node");
      }
    }
  }

  // Add placeholder.
  for (const auto &node : mod["nodes"]) {
    const auto &opCode = node["op_code"].getString();

    if (!isOps(opCode)) {
      const auto &name = opCode != "output"
                             ? node["name"].getString()
                             : node["args"][0]["name"].getString();
      bool inputVar(readTensors_.count(name) && !writeTensors_.count(name));
      bool outputVar(!readTensors_.count(name) && writeTensors_.count(name));
      DBG("Add placeholder: " << name);

      if (inputVar || outputVar) {
        LOG_NNPI_IF_ERROR_RETURN_INVALID_HANDLE(
            addTensor(name, node["dtype"].getString(),
                      toIntegerArray<glow::dim_t>(node["shape"].getString()),
                      inputVar, outputVar),
            "Failed to add placeholder");
        DBG("[--IO--] Setting IO variable: " << name << ", R:" << inputVar
                                             << ", W:" << outputVar);
      } else {
        DBG("[--IO--] Unused Placeholder: " << name);
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
