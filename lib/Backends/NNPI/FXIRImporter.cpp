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

#include "glow/lib/Backends/NNPI/FXIRImporter.h"
#include "glow/Flags/Flags.h"
#include "glow/Support/Support.h"
#include "glow/lib/Backends/NNPI/DebugMacros.h"
#include "nnpi_transformer_types.h"
#include <vector>

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
    const auto &name = node["name"].getString();
    const auto &kwargs = node["kwargs"];

    // TODO: broadcast inputs if input is not a node.
    std::array<NNPIObjectName, 2> inputNames;
    snprintf(inputNames[0], NNPI_MAX_STRING_LEN, "%s",
             importer.getInputNodeName(kwargs["input"]).c_str());
    snprintf(inputNames[1], NNPI_MAX_STRING_LEN, "%s",
             importer.getInputNodeName(kwargs["other"]).c_str());

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
  NNPIErrorCode
  importNode(const folly::dynamic &node,
             const std::function<string(string)> & /* getQualName */,
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

class EmbeddingBagNodeImporter : public INNPIFXNodeImporter {
public:
  NNPIErrorCode importNode(const folly::dynamic &node,
                           const std::function<string(string)> &getQualName,
                           FXNNPIImporter &importer) override {
    const auto &name = node["name"].getString();
    const auto &kwargs = node["kwargs"];
    const auto &inputName = importer.getInputNodeName(kwargs["input"]);
    const auto &weightName = importer.getInputNodeName(kwargs["weight"]);
    const auto &per_sample_weights = importer.getInputNodeName(
        kwargs["per_sample_weights"], /* optional */ true);
    const auto &offsetsName = importer.getInputNodeName(kwargs["offsets"]);

    const auto &hasEndOffset = kwargs["include_last_offset"].asBool();
    LOG_AND_RETURN_IF_NOT(ERROR, hasEndOffset,
                          "[EmbeddingBag] hasEndOffset must be true",
                          NNPI_INVALID_PARAM);

    // We will assume that all embedding bags have been legalized to include per
    // index weights.
    importer.setUsedTensors(
        {
            weightName,
            per_sample_weights,
            inputName,
            offsetsName,
        },
        {name});

    return nnpiNetworkAddSparseLengthsWeightedSumOp(
        importer.getNetwork(), finalize(name), finalize(weightName),
        finalize(name), finalize(per_sample_weights), finalize(inputName),
        finalize(offsetsName),
        /* useFP32Accumulation */ 0, /* useLengthsAsOffsets */ 1,
        /*avg length*/ NAN, NNPI_LENGTH_VARIABLE);
  }
};

class EmbeddingBagByteRowwiseOffsetsNodeImporter : public INNPIFXNodeImporter {
public:
  NNPIErrorCode importNode(const folly::dynamic &node,
                           const std::function<string(string)> &getQualName,
                           FXNNPIImporter &importer) override {

    const auto &name = node["name"].getString();
    const auto &kwargs = node["kwargs"];
    const auto &inputName = importer.getInputNodeName(kwargs["indices"]);
    const auto &weightName = importer.getInputNodeName(kwargs["weight"]);
    const auto &per_sample_weights = importer.getInputNodeName(
        kwargs["per_sample_weights"], /* optional */ true);
    const auto &offsetsName = importer.getInputNodeName(kwargs["offsets"]);

    const auto &hasEndOffset = kwargs["include_last_offset"].asBool();
    LOG_AND_RETURN_IF_NOT(ERROR, hasEndOffset,
                          "[EmbeddingBag] hasEndOffset must be true",
                          NNPI_INVALID_PARAM);

    importer.setUsedTensors(
        {weightName, per_sample_weights, inputName, offsetsName}, {name});

    return nnpiNetworkAddSparseLengthsWeightedSumOp(
        importer.getNetwork(), finalize(name), finalize(weightName),
        finalize(name), finalize(per_sample_weights), finalize(inputName),
        finalize(offsetsName),
        /* useFP32Accumulation */ true, /* useLengthsAsOffsets */ 1,
        /*avg length*/ NAN, NNPI_LENGTH_VARIABLE);
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

class SigmoidNodeImporter : public INNPIFXNodeImporter {
public:
  NNPIErrorCode
  importNode(const folly::dynamic &node,
             const std::function<string(string)> & /* getQualName */,
             FXNNPIImporter &importer) override {
    const auto &inputs = node["kwargs"];
    const auto &name = node["name"].getString();
    const auto &inputName = importer.getInputNodeName(inputs["input"]);

    importer.setUsedTensors({inputName}, {name});

    return nnpiNetworkAddSigmoidOp(importer.getNetwork(), finalize(name),
                                   finalize(inputName), finalize(name));
  }
};

class SumNodeImporter : public INNPIFXNodeImporter {
public:
  NNPIErrorCode
  importNode(const folly::dynamic &node,
             const std::function<string(string)> & /* getQualName */,
             FXNNPIImporter &importer) override {

    const auto &inputs = node["kwargs"];
    const auto &name = node["name"].getString();
    const auto &inputName = importer.getInputNodeName(inputs["input"]);

    importer.setUsedTensors({inputName}, {name});

    auto dim = inputs.find("dim");
    std::vector<uint32_t> axis;
    if (dim != inputs.items().end()) {
      auto dims = toIntegerArray<uint32_t>(inputs["dim"]);
      for (auto &d : dims) {
        axis.push_back(d);
      }
    }

    return nnpiNetworkAddReduceOp(importer.getNetwork(), finalize(name),
                                  finalize(inputName), finalize(name),
                                  NNPI_REDUCE_SUM, axis.data(), axis.size(), 0);
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

class SliceTensorNodeImporter : public INNPIFXNodeImporter {
public:
  NNPIErrorCode
  importNode(const folly::dynamic &node,
             const std::function<string(string)> & /* getQualName */,
             FXNNPIImporter &importer) override {

    const auto &inputs = node["kwargs"];
    const auto &name = node["name"].getString();
    const auto &inputName = importer.getInputNodeName(inputs["input"]);
    auto dims = toIntegerArray<uint32_t>(inputs["dims"]);
    auto starts = toIntegerArray<uint32_t>(inputs["starts"]);
    auto stops = toIntegerArray<uint32_t>(inputs["stops"]);
    auto steps = toIntegerArray<uint32_t>(inputs["steps"]);
    CHECK_EQ(dims.size(), 1) << "Only supporting single dim slice";
    CHECK_EQ(starts.size(), 1) << "Only supporting single start slice";
    CHECK_EQ(stops.size(), 1) << "Only supporting single stop slice";
    CHECK_EQ(steps.size(), 1) << "Only supporting single step slice";
    CHECK_EQ(steps[0], 1) << "Only supporting step == 1";

    auto shape = toIntegerArray<glow::dim_t>(node["shape"].getString());

    int32_t startOffset[NNPI_MAX_DIMS] = {0};
    int32_t endOffset[NNPI_MAX_DIMS] = {0};

    for (size_t i = 0, e = shape.size(); i < e; i++) {
      if (i != dims[0]) {
        startOffset[i] = 0;
        endOffset[i] = shape[i];
        continue;
      }
      startOffset[i] = starts[0];
      endOffset[i] = stops[0];
    }

    importer.setUsedTensors({inputName}, {name});

    return nnpiNetworkAddSliceOp(importer.getNetwork(), finalize(name),
                                 finalize(inputName), finalize(name),
                                 startOffset, endOffset, nullptr,
                                 uint32_t(shape.size()));
  }
};

class TransposeNodeImporter : public INNPIFXNodeImporter {
public:
  NNPIErrorCode
  importNode(const folly::dynamic &node,
             const std::function<string(string)> & /* getQualName */,
             FXNNPIImporter &importer) override {

    auto dimSize = toIntegerArray<glow::dim_t>(node["shape"].getString());
    const auto &kwargs = node["kwargs"];
    const auto &name = node["name"].getString();
    const auto &inputName = importer.getInputNodeName(kwargs["input"]);
    uint32_t nnpiOrder[NNPI_MAX_DIMS];
    for (size_t i = 0, e = dimSize.size(); i < e; i++) {
      nnpiOrder[i] = i;
    }
    auto dim0 = kwargs["dim0"].getInt();
    auto dim1 = kwargs["dim1"].getInt();
    nnpiOrder[dim0] = dim1;
    nnpiOrder[dim1] = dim0;

    importer.setUsedTensors({inputName}, {name});

    return nnpiNetworkAddTransposeOp(importer.getNetwork(), finalize(name),
                                     finalize(inputName), finalize(name),
                                     nnpiOrder, dimSize.size());
  }
};

class PermuteNodeImporter : public INNPIFXNodeImporter {
public:
  NNPIErrorCode
  importNode(const folly::dynamic &node,
             const std::function<string(string)> & /* getQualName */,
             FXNNPIImporter &importer) override {

    auto dimSize = toIntegerArray<glow::dim_t>(node["shape"].getString());
    const auto &kwargs = node["kwargs"];
    const auto &name = node["name"].getString();
    const auto &inputName = importer.getInputNodeName(kwargs["input"]);
    auto dims = toIntegerArray<uint32_t>(kwargs["permutation"]);
    uint32_t nnpiOrder[NNPI_MAX_DIMS];
    for (size_t i = 0, e = dimSize.size(); i < e; i++) {
      nnpiOrder[i] = dims[i];
    }

    importer.setUsedTensors({inputName}, {name});

    return nnpiNetworkAddTransposeOp(importer.getNetwork(), finalize(name),
                                     finalize(inputName), finalize(name),
                                     nnpiOrder, dimSize.size());
  }
};

class MatMulNodeImporter : public INNPIFXNodeImporter {
public:
  NNPIErrorCode
  importNode(const folly::dynamic &node,
             const std::function<string(string)> & /* getQualName */,
             FXNNPIImporter &importer) override {

    const auto &kwargs = node["kwargs"];
    const auto &name = node["name"].getString();
    const auto &inputName = importer.getInputNodeName(kwargs["input"]);
    const auto &otherName = importer.getInputNodeName(kwargs["other"]);

    importer.setUsedTensors({inputName, otherName}, {name});

    return nnpiNetworkAddMatMulOp(importer.getNetwork(), finalize(name),
                                  finalize(inputName), finalize(otherName),
                                  finalize(name));
  }
};

class ConcatNodeImporter : public INNPIFXNodeImporter {
public:
  NNPIErrorCode
  importNode(const folly::dynamic &node,
             const std::function<string(string)> & /* getQualName */,
             FXNNPIImporter &importer) override {
    const auto &kwargs = node["kwargs"];
    const auto &tensors = kwargs["tensors"];
    const auto &name = node["name"].getString();
    const size_t numInputs = tensors.size();

    NNPIObjectName inputs[numInputs];
    std::unordered_set<std::string> inputTensors;
    for (size_t i = 0; i < numInputs; i++) {
      auto nvName = tensors[i]["name"].getString();
      strncpy(inputs[i], nvName.c_str(), sizeof(NNPIObjectName));
      inputTensors.insert(nvName);
    }

    importer.setUsedTensors(inputTensors, {name});
    return nnpiNetworkAddConcatOp(importer.getNetwork(), finalize(name), inputs,
                                  numInputs, finalize(name),
                                  kwargs["dim"].getInt());
  }
};

class LayerNormalizationNodeImporter : public INNPIFXNodeImporter {
public:
  NNPIErrorCode
  importNode(const folly::dynamic &node,
             const std::function<string(string)> & /* getQualName */,
             FXNNPIImporter &importer) override {

    const auto &kwargs = node["kwargs"];
    const auto &name = node["name"].getString();
    const auto &inputName = importer.getInputNodeName(kwargs["input"]);
    const auto &weightName = importer.getInputNodeName(kwargs["weight"]);
    const auto &biasName = importer.getInputNodeName(kwargs["bias"]);
    auto eps = kwargs["eps"].getDouble();
    auto shape = toIntegerArray<uint32_t>(kwargs["normalized_shape"]);

    importer.setUsedTensors({inputName, weightName}, {biasName, name});

    return nnpiNetworkAddLayerNormOp(importer.getNetwork(), finalize(name),
                                     finalize(inputName), finalize(name),
                                     finalize(weightName), finalize(biasName),
                                     shape.data(), shape.size(), eps);
  }
};

class TanhNodeImporter : public INNPIFXNodeImporter {
public:
  NNPIErrorCode
  importNode(const folly::dynamic &node,
             const std::function<string(string)> & /* getQualName */,
             FXNNPIImporter &importer) override {
    const auto &kwargs = node["kwargs"];
    const auto &name = node["name"].getString();
    const auto &inputName = importer.getInputNodeName(kwargs["input"]);

    importer.setUsedTensors({inputName}, {name});

    return nnpiNetworkAddTanhOp(importer.getNetwork(), finalize(name),
                                finalize(inputName), finalize(name));
  }
};

class ConvertNodeImporter : public INNPIFXNodeImporter {
public:
  NNPIErrorCode
  importNode(const folly::dynamic &node,
             const std::function<string(string)> & /* getQualName */,
             FXNNPIImporter &importer) override {
    const auto &args = node["args"];
    const auto &kwargs = node["kwargs"];
    const auto &name = node["name"].getString();

    const auto &inputName = kwargs.count("input")
                                ? importer.getInputNodeName(kwargs["input"])
                                : importer.getInputNodeName(args[0]);

    importer.setUsedTensors({inputName}, {name});

    return nnpiNetworkAddConvertOp(importer.getNetwork(), finalize(name),
                                   finalize(inputName), finalize(name));
  }
};

static std::unordered_map<
    std::string,
    std::unique_ptr<INNPIFXNodeImporter>>::value_type FXImporterInit[] = {
    {"acc_ops.add",
     std::make_unique<BinaryEltwiseNodeImporter<NNPI_ELTWISE_ADD>>()},
    {"acc_ops.quantized_add",
     std::make_unique<BinaryEltwiseNodeImporter<NNPI_ELTWISE_ADD>>()},
    {"acc_ops.sub",
     std::make_unique<BinaryEltwiseNodeImporter<NNPI_ELTWISE_SUB>>()},
    {"acc_ops.mul",
     std::make_unique<BinaryEltwiseNodeImporter<NNPI_ELTWISE_MUL>>()},
    {"acc_ops.quantized_mul",
     std::make_unique<BinaryEltwiseNodeImporter<NNPI_ELTWISE_MUL>>()},
    {"acc_ops.div",
     std::make_unique<BinaryEltwiseNodeImporter<NNPI_ELTWISE_DIV>>()},
    {"acc_ops.reshape", std::make_unique<ReshapeNodeImporter>()},
    {"acc_ops.tanh", std::make_unique<TanhNodeImporter>()},
    {"acc_ops.slice_tensor", std::make_unique<SliceTensorNodeImporter>()},
    {"acc_ops.linear", std::make_unique<LinearNodeImporter>()},
    {"acc_ops.quantized_linear", std::make_unique<LinearNodeImporter>()},
    {"acc_ops.conv2d", std::make_unique<ConvolutionNodeImporter<2>>()},
    {"acc_ops.quantized_conv2d",
     std::make_unique<ConvolutionNodeImporter<2>>()},
    {"acc_ops.batch_norm", std::make_unique<BatchNormalizationNodeImporter>()},
    {"acc_ops.quantized_batch_norm2d",
     std::make_unique<BatchNormalizationNodeImporter>()},
    {"acc_ops.layer_norm", std::make_unique<LayerNormalizationNodeImporter>()},
    {"acc_ops.relu", std::make_unique<ReluNodeImporter>()},
    {"acc_ops.sigmoid", std::make_unique<SigmoidNodeImporter>()},
    {"acc_ops.adaptive_avg_pool2d",
     std::make_unique<AdaptivePoolNodeImporter<NNPI_POOL_AVG>>()},
    {"acc_ops.embedding_bag", std::make_unique<EmbeddingBagNodeImporter>()},
    {"acc_ops.embedding_bag_byte_rowwise_offsets",
     std::make_unique<EmbeddingBagByteRowwiseOffsetsNodeImporter>()},
    {"acc_ops.cat", glow::make_unique<ConcatNodeImporter>()},
    {"acc_ops.sum", glow::make_unique<SumNodeImporter>()},
    {"acc_ops.transpose", glow::make_unique<TransposeNodeImporter>()},
    {"acc_ops.permute", glow::make_unique<PermuteNodeImporter>()},
    {"acc_ops.matmul", glow::make_unique<MatMulNodeImporter>()},
    {"acc_ops.max_pool2d",
     std::make_unique<PoolNodeImporter<NNPI_POOL_MAX, 2>>()},
    {"acc_ops.quantize_per_tensor", std::make_unique<ConvertNodeImporter>()},
    {"acc_ops.dequantize", std::make_unique<ConvertNodeImporter>()},
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

  CHECK(node.isObject()) << ": Expected Node object, but found "
                         << node.typeName() << ": " << node;
  CHECK(node.find("is_node") != node.items().end() && node["is_node"].asBool())
      << "Expected is_node";

  const auto &name = node["name"].getString();

  // Check if this name was for a getattr. If so, return the underlying Constant
  // name. Otherwise return the name unchanged.
  auto it = getattrs_.find(name);
  return it != getattrs_.end() ? it->second : name;
}

void FXNNPIImporter::updateDescQuantFromFX(
    const DTYPE &dtype, NNPITensorDesc &desc, const float &scale,
    const int32_t &offset, const std::string &scaleTensor,
    const std::string &offsetTensor, bool forceSymlowp, bool zeroOffset) {
  desc.quantParams.params.gemlowp.scale = scale;
  desc.quantParams.params.gemlowp.offset = offset;

  switch (dtype) {
  case DTYPE::FLOAT32:
    LOG_ERROR_IF_NOT((scaleTensor.empty() && offsetTensor.empty()))
        << "Scales and offsets provided for Float32";
    desc.quantParams.precision = NNPI_PRECISION_FLOAT32;
    desc.quantParams.type = NNPI_QUANTIZATION_NONE;
    break;
  case DTYPE::UINT8FUSED:
    desc.quantParams.precision = NNPI_PRECISION_UINT8;
    desc.quantParams.type = NNPI_QUANTIZATION_GEMMLOWP_PCQ_FUSED;
    break;
  case DTYPE::FLOAT16:
    LOG_ERROR_IF_NOT((scaleTensor.empty() && offsetTensor.empty()))
        << "Scales and offsets provided for Float16";
    desc.quantParams.precision = NNPI_PRECISION_FLOAT16;
    desc.quantParams.type = NNPI_QUANTIZATION_NONE;
    break;
  case DTYPE::INT32:
  case DTYPE::INT64:
    LOG_ERROR_IF_NOT((scaleTensor.empty() && offsetTensor.empty()))
        << "Scales and offsets provided for Int64 or Int32";
    desc.quantParams.precision = NNPI_PRECISION_INT32;
    desc.quantParams.type = NNPI_QUANTIZATION_NONE;
    break;
  case DTYPE::QINT8:
    desc.quantParams.precision = NNPI_PRECISION_INT8;

    // If we have scales tensor, this is PCQ case.
    if (!scaleTensor.empty()) {
      LOG_ERROR_IF_NOT(!forceSymlowp || zeroOffset)
          << "Offset is not 0 when forcing symlowp";
      // If there is no offsets, or Symlowp workaround is used and all offsets
      // are zero, the quantization type is SYMLOWP_PCQ.
      if (offsetTensor.empty() || (forceSymlowp && zeroOffset)) {
        desc.quantParams.type = NNPI_QUANTIZATION_SYMLOWP_PCQ;
        std::strncpy(desc.quantParams.params.symlowpPCQ.scalesTensor,
                     scaleTensor.c_str(), NNPI_MAX_STRING_LEN - 1);
      } else { // Both scales and offsets are present.
        desc.quantParams.type = NNPI_QUANTIZATION_GEMMLOWP_PCQ;
        std::strncpy(desc.quantParams.params.gemmlowpPCQ.scalesTensor,
                     scaleTensor.c_str(), NNPI_MAX_STRING_LEN - 1);
        std::strncpy(desc.quantParams.params.gemmlowpPCQ.offsetsTensor,
                     offsetTensor.c_str(), NNPI_MAX_STRING_LEN - 1);
      }
    } else {
      desc.quantParams.type = NNPI_QUANTIZATION_GEMMLOWP;
      if (forceSymlowp && zeroOffset) {
        desc.quantParams.type = NNPI_QUANTIZATION_SYMLOWP;
        desc.quantParams.params.symlowp.scale = scale;
      }
    }
    break;
  case DTYPE::QUINT8:
    desc.quantParams.precision = NNPI_PRECISION_UINT8;
    if (!scaleTensor.empty()) {
      desc.quantParams.type = NNPI_QUANTIZATION_GEMMLOWP_PCQ;
      std::strncpy(
          desc.quantParams.params.gemmlowpPCQ.scalesTensor, scaleTensor.c_str(),
          sizeof(desc.quantParams.params.gemmlowpPCQ.scalesTensor) - 1);
      std::strncpy(desc.quantParams.params.gemmlowpPCQ.offsetsTensor,
                   offsetTensor.c_str(), NNPI_MAX_STRING_LEN - 1);
    } else {
      desc.quantParams.type = NNPI_QUANTIZATION_GEMMLOWP;
      desc.quantParams.params.gemlowp.scale = scale;
      desc.quantParams.params.gemlowp.offset = offset;
    }
    break;
  default:
    LOG(FATAL) << "Unhandled tensor data type";
  }
}

void FXNNPIImporter::updateDescDimsFromFX(llvm::ArrayRef<glow::dim_t> dims,
                                          NNPITensorDesc &desc) {
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

NNPIErrorCode FXNNPIImporter::addTensor(
    const std::string &name, const string &dtypeStr,
    llvm::ArrayRef<glow::dim_t> dims, bool input, bool output,
    const float &scale, const int32_t &offset, const std::string &scaleTensor,
    const std::string &offsetTensor, bool forceSymlowp, bool zeroOffset) {
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
  updateDescQuantFromFX(dtype, desc, scale, offset, scaleTensor, offsetTensor,
                        forceSymlowp || compileOptions_.useSymlowp, zeroOffset);
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

bool FXNNPIImporter::isZeroes(const std::string &name, const DTYPE &dtype,
                              const size_t &size) const {
  const auto *t = getConstant(name);
  CHECK(t) << "Can't find constant with name " << name;

  switch (dtype) {
  case DTYPE::INT32: {
    const auto *pDataInt32 = static_cast<const int32_t *>(t);
    return std::all_of(pDataInt32, pDataInt32 + size,
                       [](int32_t x) { return x == 0; });
  }
  default:
    return false;
  }
}

NNPIErrorCode FXNNPIImporter::addTensor(const std::string &name,
                                        const folly::dynamic &node, bool input,
                                        bool output) {
  const auto &dims = toIntegerArray<glow::dim_t>(node["shape"].getString());
  bool zeroOffset = false;
  bool forceSymlowp = false;
  float scale = 1.0f;
  int32_t zero_point = 0;
  std::string scaleTensor;
  std::string offsetTensor;

  if (node["is_quantized"].getBool()) {
    forceSymlowp = node["dtype"].getString() == "torch.qint8";

    if (node["qscheme"].getString().find("per_tensor") != std::string::npos) {
      scale = node["q_scale"].getDouble();
      zero_point = node["q_zero_point"].getInt();
      zeroOffset = zero_point == 0;
    } else {
      scaleTensor = node["q_per_channel_scales"].getString();
      offsetTensor = node["q_per_channel_zero_points"].getString();
      zeroOffset =
          isZeroes(offsetTensor, /* dtype */ DTYPE::INT32,
                   /* size */ dims[node["q_per_channel_axis"].getInt()]);
    }
  }

  return addTensor(name, node["dtype"].getString(), /* dims */ dims,
                   /* input */ input, /* output */ output,
                   /* scale */ scale,
                   /* offset */ zero_point, /* scaleTensor */ scaleTensor,
                   /* offsetTensor */ offsetTensor,
                   /* forceSymlowp */ forceSymlowp,
                   /* zeroOffset */ zeroOffset);
}

void FXNNPIImporter::logUnsupportedNodes(const folly::dynamic &mod) {
  for (const auto &node : mod["nodes"]) {
    const auto &opCode = node["op_code"].getString();
    if (!isOps(opCode)) {
      continue;
    }

    if (opCode == "get_attr") {
      continue;
    }
    const auto &targetName = node["target"].getString();
    const auto &functionName = opCode != "call_module"
                                   ? targetName
                                   : node["parameters"]["name"].getString();
    // Log unsupported node.
    if (FXNodeImporters.find(functionName) == FXNodeImporters.end()) {
      LOG(INFO) << "No support for node: " << functionName;
    }
  }
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
    const auto &weight = weights[name];
    DBG("Importing Constant: " << name);
    CHECK(constants_.count(name)) << "Constant not found for weight " << name;
    LOG_NNPI_IF_ERROR_RETURN_INVALID_HANDLE(addTensor(name, weight),
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

    // Track what Constant each get_attr points to.
    if (opCode == "get_attr") {
      bool inserted =
          getattrs_.try_emplace(nodeName, node["target"].getString()).second;
      CHECK(inserted) << "Already mapped a getattr by name " << nodeName
                      << " to its underlying Constant";
      continue;
    }

    // Add node outputs. We don't add get_attr node output because they have
    // been added when adding constants.
    LOG_NNPI_IF_ERROR_RETURN_INVALID_HANDLE(addTensor(nodeName, node),
                                            "Failed to add intermediate");

    const auto &targetName = node["target"].getString();
    const auto &functionName = opCode != "call_module"
                                   ? targetName
                                   : node["parameters"]["name"].getString();
    // Import node.
    if (FXNodeImporters.find(functionName) == FXNodeImporters.end()) {
      // Before returning walk the graph and log all unsupported nodes.
      logUnsupportedNodes(mod);
      LOG_NNPI_IF_ERROR_RETURN_INVALID_HANDLE(
          NNPIErrorCode::NNPI_INVALID_NETWORK,
          glow::strFormat(
              "Could not import node with opCode '%s', target '%s'.",
              opCode.c_str(), targetName.c_str()))
    }

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

  // Add placeholder and output node
  for (const auto &node : mod["nodes"]) {
    const auto &opCode = node["op_code"].getString();

    if (opCode == "placeholder") {
      const auto &name = node["name"].getString();

      DBG("Add placeholder: " << name);
      CHECK(!writeTensors_.count(name)) << "Placeholder can't be written";

      if (readTensors_.count(name)) {
        LOG_NNPI_IF_ERROR_RETURN_INVALID_HANDLE(addTensor(name, node,
                                                          /* input */ true,
                                                          /* output */ false),
                                                "Failed to add placeholder");
      } else {
        DBG("[--IO--] Unused Placeholder: " << name);
      }

      // Gather Placeholders that allow partial input and require padding.
      if (getValOrDefault(node, "allow_partial", false)) {
        allowPartialPlaceholderNames_.insert(name);
      }
      if (getValOrDefault(node, "requires_padding", false)) {
        requiresPaddingPlaceholderNames_.insert(name);
      }
    } else if (opCode == "output") {
      const auto &args = node["args"];

      for (const auto &arg : args) {
        const auto &outputName = getInputNodeName(arg);

        DBG("Add output" << outputName);
        CHECK(writeTensors_.count(outputName))
            << "output must be in writeTensors_";

        LOG_NNPI_IF_ERROR_RETURN_INVALID_HANDLE(addTensor(outputName, arg,
                                                          /* input */ false,
                                                          /* output */ true),
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
