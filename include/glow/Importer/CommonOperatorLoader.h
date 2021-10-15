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

#ifndef GLOW_IMPORTER_COMMONOPERATORLOADER_H
#define GLOW_IMPORTER_COMMONOPERATORLOADER_H

#include "foxi/onnxifi.h"

#include "glow/Importer/ProtobufLoader.h"

#include "glow/Base/Tensor.h"
#include "glow/Graph/Graph.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <functional>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

namespace glow {

/// Result of loading a weight, potentially with additional offsets and
/// scales tensors containing quantization parameters only if the loaded weight
/// was found to have multiple quantization parameters.
struct LoadWeightResult {
  /// Main Glow tensor, this is always non-null.
  std::unique_ptr<Tensor> t;
  /// Glow tensor containing quantization offsets. This should only be non-null
  /// if there is more than 1 quantization parameter found.
  std::unique_ptr<Tensor> offsets;
  /// Glow tensor containing quantization scales. This should only be non-null
  /// if there is more than 1 quantization parameter found.
  std::unique_ptr<Tensor> scales;
  /// Type info of the weight, this is used for offline weights.
  Type type;
};

#define dispatchQuantizedImpl(functionName, elemTy, ...)                       \
  switch (elemTy) {                                                            \
  case ElemKind::Int8QTy:                                                      \
    functionName<int8_t>(__VA_ARGS__);                                         \
    break;                                                                     \
  case ElemKind::Int16QTy:                                                     \
    functionName<int16_t>(__VA_ARGS__);                                        \
    break;                                                                     \
  case ElemKind::Int32QTy:                                                     \
    functionName<int32_t>(__VA_ARGS__);                                        \
    break;                                                                     \
  default:                                                                     \
    llvm_unreachable("Type is not supported");                                 \
  }

template <typename eTy>
void rescaleQTensor(const Tensor &oldT, Tensor &rescaledT, float newMin,
                    float newMax) {
  const Type &oldTy = oldT.getType();
  const TensorQuantizationParams oldQParams = {oldTy.getScale(),
                                               oldTy.getOffset()};
  const TensorQuantizationParams newQParams = chooseQuantizationParams(
      {newMin, newMax}, quantization::Asymmetric, oldTy.getElementType());

  // Setup Tensor to copy rescaled Tensor into.
  Type rescaledTy(oldTy.getElementType(), oldTy.dims(), newQParams.scale,
                  newQParams.offset);
  rescaledT.reset(rescaledTy);

  auto srcH = oldT.getHandle<eTy>();
  auto destH = rescaledT.getHandle<eTy>();
  for (size_t i = 0, e = destH.size(); i < e; ++i) {
    float val = quantization::dequantize(srcH.raw(i), oldQParams);
    destH.raw(i) = quantization::quantize(val, newQParams);
  }
}

/// Given \p result, rescale it given \p newMin and \p newMax.
template <typename eTy>
void rescaleQTensorResult(LoadWeightResult &result, float newMin,
                          float newMax) {
  // Get new type based on newMin/newMax and old elem kind.
  auto rescaledT = glow::make_unique<Tensor>();
  rescaleQTensor<eTy>(*result.t, *rescaledT, newMin, newMax);
  result.t = std::move(rescaledT);
  result.type = result.t->getType();
}

/// Contains loaders for operators, which are common to ONNX and Caffe2
/// formats. Every loader method adds necessary nodes to property G_, which
/// is inherited from ProtobufLoader class, therefore modifying the class
/// instance itself.
template <typename OpType, typename AttrType>
class CommonOperatorLoader : public ProtobufLoader {
  /// Loads the onnxTensorDescriptorV1 \p in and \returns a LoadWeightResult
  /// where result.t is the main contents of the the onnxTensorDescriptorV1 and
  /// result.offsets and result.scales are the quantization scales and offsets
  /// of the onnxTensorDescriptorV1 if there were more than 1. If there is
  /// exactly 1 scale and offset then result.t will be a quantized glow tensor.
  inline Expected<LoadWeightResult>
  loadWeight(const onnxTensorDescriptorV1 &in) {
    // Only support CPU memory tensors.
    if (in.memoryType != ONNXIFI_MEMORY_TYPE_CPU) {
      return MAKE_ERR("Only support CPU memory tensors.");
    }

    // Number of qparams in the onnxTensorDescriptor.
    const dim_t qparams = static_cast<dim_t>(in.quantizationParams);

    // Only support quantizationAxis=1 for now.
    if (qparams > 0 && in.quantizationAxis != 1) {
      return MAKE_ERR(strFormat(
          "Glow can only import quantized tensors with quantizationAxis=1 but "
          "the tensor %s has quantizationAxis=%u",
          in.name, in.quantizationAxis));
    }

    LoadWeightResult result;
    result.t = glow::make_unique<Tensor>();

    std::vector<dim_t> dims;
    for (unsigned i = 0; i < in.dimensions; ++i) {
      dims.push_back(in.shape[i]);
    }

    // Load unquantized tensor.
    if (in.quantizationParams == 0) {
      if (in.dataType == ONNXIFI_DATATYPE_FLOAT32) {
        result.type = Type(ElemKind::FloatTy, dims);
      } else if (in.dataType == ONNXIFI_DATATYPE_FLOAT16) {
        result.type = Type(ElemKind::Float16Ty, dims);
      } else if (in.dataType == ONNXIFI_DATATYPE_BFLOAT16) {
        result.type = Type(ElemKind::BFloat16Ty, dims);
      } else if (in.dataType == ONNXIFI_DATATYPE_INT32) {
        result.type = Type(ElemKind::Int32ITy, dims);
      } else if (in.dataType == ONNXIFI_DATATYPE_INT64) {
        result.type = Type(ElemKind::Int64ITy, dims);
      } else if (in.dataType == ONNXIFI_DATATYPE_UINT8) {
        // UInt8 type is used for variety of rowwise quantized SLSs.
        // Make dummy scale and offset for these cases.
        result.type = Type(ElemKind::UInt8QTy, dims, 1.0, 0);
      } else if (in.dataType == ONNXIFI_DATATYPE_UINT64) {
        result.type = Type(ElemKind::Int64ITy, dims);
        for (size_t i = 0; i < result.t->size(); ++i) {
          RETURN_ERR_IF_NOT(
              ((int64_t *)in.buffer)[i] >= 0,
              "Disallow overflow of loaded UINT64 data into Int64ITy.");
        }
      } else {
        return MAKE_ERR(strFormat(
            "Only float, index, and uint8 unquantized tensors are supported, "
            "got input with ONNXIFI_DATATYPE: %zu",
            static_cast<size_t>(in.dataType)));
      }
      if (!in.isOffline) {
        *result.t = Tensor((void *)in.buffer, &result.type);
      }
      return Expected<LoadWeightResult>(std::move(result));
    }

    // Load quantized tensor with either a single or multiple qparams.
    float scale = 1.0;
    int32_t offset = 0;

    // If multiple qparams are present then load them as tensors and use the
    // the default qparams for the result.t otherwise use the first (only)
    // qparams.
    if (in.quantizationParams == 1) {
      scale = in.scales[0];
      offset = in.biases[0];
    } else {
      RETURN_ERR_IF_NOT(!loadUniquedDummyQParams_,
                        strFormat("Unsupported loading of uniqued qparams for "
                                  "vector of scales/biases for %s",
                                  in.name));
      Type scalesTy(ElemKind::FloatTy, llvm::makeArrayRef({qparams}));
      Type offsetsTy(ElemKind::Int32ITy, llvm::makeArrayRef({qparams}));
      result.scales = glow::make_unique<Tensor>((void *)in.scales, &scalesTy);
      result.offsets = glow::make_unique<Tensor>((void *)in.biases, &offsetsTy);
    }

    // If we have a scale of dummyScale, then this must be a dummy pair of
    // scale/offset. Look up the actual scale/offset to use as previously
    // loaded, using the offset as the key to updatedTQPs_.
    if (replaceDummyTQPs_ && scale == dummyScale) {
      TensorQuantizationParams TQP;
      ASSIGN_VALUE_OR_RETURN_ERR(TQP, getUpdatedTQP(offset));
      scale = TQP.scale;
      offset = TQP.offset;
    }

    if (in.dataType == ONNXIFI_DATATYPE_UINT8) {
      TypeRef outTy;
      ASSIGN_VALUE_OR_RETURN_ERR(
          outTy, ProtobufLoader::loadQuantTy(
                     in.name, ElemKind::Int8QTy, dims, scale, offset,
                     /* shiftUInt8ToInt8 */ true,
                     /* skipClipQuantRangeToFP16 */ true));
      // Must copy the weights here because we will need to modify them by
      // adjusting for UINT8_TO_INT8_SHIFT.
      result.type = *outTy;
      if (!in.isOffline) {
        result.t->reset(result.type);

        auto TH = result.t->getHandle<int8_t>();
        uint8_t *data = (uint8_t *)in.buffer;
        for (size_t i = 0; i < TH.size(); ++i) {
          TH.raw(i) = (int8_t)(data[i] - UINT8_TO_INT8_SHIFT);
        }
      }
    } else if (in.dataType == ONNXIFI_DATATYPE_INT32) {
      TypeRef outTy;
      ASSIGN_VALUE_OR_RETURN_ERR(
          outTy, ProtobufLoader::loadQuantTy(
                     in.name, ElemKind::Int32QTy, dims, scale, offset,
                     /* shiftUInt8ToInt8 */ true,
                     /* skipClipQuantRangeToFP16 */ true));
      result.type = *outTy;
      if (!in.isOffline) {
        *result.t = Tensor((void *)in.buffer, &result.type);
      }
    } else if (in.dataType == ONNXIFI_DATATYPE_INT8) {
      TypeRef outTy;
      ASSIGN_VALUE_OR_RETURN_ERR(
          outTy, ProtobufLoader::loadQuantTy(
                     in.name, ElemKind::Int8QTy, dims, scale, offset,
                     /* shiftUInt8ToInt8 */ false,
                     /* skipClipQuantRangeToFP16 */ true));
      result.type = *outTy;
      if (!in.isOffline) {
        *result.t = Tensor((void *)in.buffer, &result.type);
      }
    } else {
      return MAKE_ERR(
          strFormat("Only uint8, int32, and int8, quantized tensors are "
                    "supported, got input with ONNXIFI_DATATYPE: %zu",
                    static_cast<size_t>(in.dataType)));
    }

    // If we're clipping quantized ranges tp FP16, then we need to rescale the
    // Tensor and update its type, plus the type in result.
    if (clipQuantRangeToFP16_) {
      const ElemKind k = result.type.getElementType();
      const auto qMinMax = getQuantizedValueRange(scale, offset, k);
      const float newMin = std::max(qMinMax.first, kMinFP16);
      const float newMax = std::min(qMinMax.second, kMaxFP16);

      // If min or max are clipped then create a new Tensor with the adjusted
      // type, and rescale its payload.
      if (newMin != qMinMax.first || newMax != qMinMax.second) {
        RETURN_ERR_IF_NOT(
            !in.isOffline,
            strFormat("For clipQuantRangeToFP16, currently do "
                      "not support offline quantizated weights: %s",
                      in.name));
        RETURN_ERR_IF_NOT(!result.offsets && !result.scales,
                          strFormat("For clipQuantRangeToFP16, currently do "
                                    "not support multiple qparams: %s",
                                    in.name));

        dispatchQuantizedImpl(rescaleQTensorResult, k, result, newMin, newMax);
      }
    }

    return Expected<LoadWeightResult>(std::move(result));
  }

  /// Merge shape \p shape into \p mergeShape, following multidirectional
  /// broadcasting rules.
  Error mergeMultidirectionalBroadcast(std::vector<dim_t> &mergeShape,
                                       llvm::ArrayRef<dim_t> shape) {
    size_t shift = mergeShape.size() - shape.size();
    for (size_t i = 0; i < shape.size(); i++) {
      if (shape[i] != 1) {
        RETURN_ERR_IF_NOT((shape[i] == mergeShape[shift + i]) ||
                              (mergeShape[shift + i] == 1),
                          "Incompatible dimension for the broadcast");
        mergeShape[shift + i] = shape[i];
      }
      // Otherwise, just leave mergeShape[i] as it is.
    }
    return Error::success();
  }

protected:
  CommonOperatorLoader(llvm::ArrayRef<const char *> names,
                       llvm::ArrayRef<TypeRef> types, Function *F,
                       Error *errPtr = nullptr,
                       bool loadIntoExistingModule = false,
                       OriginNameToTQPMap *originNameToTQPMap = nullptr,
                       bool loadUniquedDummyQParams = false,
                       bool zeroScaleFP16Clip = false,
                       bool clipQuantRangeToFP16 = false)
      : ProtobufLoader(names, types, F, errPtr, loadIntoExistingModule,
                       originNameToTQPMap, loadUniquedDummyQParams,
                       /* replaceDummyTQPs */ false, zeroScaleFP16Clip,
                       clipQuantRangeToFP16) {}

  CommonOperatorLoader(
      llvm::ArrayRef<const char *> names, llvm::ArrayRef<TypeRef> types,
      Module &mod, Error *errPtr = nullptr, bool loadIntoExistingModule = false,
      OriginNameToTQPMap *originNameToTQPMap = nullptr,
      bool loadUniquedDummyQParams = false, bool replaceDummyTQPs = false,
      bool zeroScaleFP16Clip = false, bool clipQuantRangeToFP16 = false)
      : ProtobufLoader(names, types, mod, errPtr, loadIntoExistingModule,
                       originNameToTQPMap, loadUniquedDummyQParams,
                       replaceDummyTQPs, zeroScaleFP16Clip,
                       clipQuantRangeToFP16) {}

  using ArgumentDictionaryTy =
      std::unordered_map<std::string, const AttrType *>;

  /// If we were replacing or loading dummy TQPs, \returns success if there
  /// aren't any dummies left, or there are only dummies left.
  Error verifyDummyQParams() {
    RETURN_ERR_IF_NOT(!(replaceDummyTQPs_ && loadUniquedDummyQParams_),
                      "Cannot replace dummy TQPs when loading uniqued TQPs.");
    if (replaceDummyTQPs_ || loadUniquedDummyQParams_) {
      RETURN_IF_ERR(mod_.verifyDummyQParams(loadUniquedDummyQParams_));
    }
    return Error::success();
  }

  /// Helper to load quantization parameters from \p dict for op named \p name.
  /// \returns a new TypeRef given \p k and \p dims.
  Expected<TypeRef> loadQuantTy(const std::string &name, ElemKind k,
                                llvm::ArrayRef<dim_t> dims,
                                ArgumentDictionaryTy &dict,
                                bool skipClipQuantRangeToFP16 = false) {
    RETURN_ERR_IF_NOT(dict.count("Y_scale"),
                      "missing Y_scale for quantized output type for " + name);
    RETURN_ERR_IF_NOT(dict.count("Y_zero_point"),
                      "missing zero point for quantized output type for " +
                          name);

    float scale;
    ASSIGN_VALUE_OR_RETURN_ERR(scale, loadFloat(dict["Y_scale"]));
    int32_t offset;
    ASSIGN_VALUE_OR_RETURN_ERR(offset, loadInt(dict["Y_zero_point"]));

    return ProtobufLoader::loadQuantTy(name, k, dims, scale, offset,
                                       /* shiftUInt8ToInt8 */ true,
                                       skipClipQuantRangeToFP16);
  }

  /// \returns True if the operator has broadcasting activated.
  virtual Expected<bool> getBroadcast(ArgumentDictionaryTy &dict) = 0;

  /// \returns True if the operator with the name \p typeName has support
  /// for multidirectional broadcasting.
  virtual bool hasMultidirectionalBroadcast(const llvm::StringRef typeName) = 0;

  inline Expected<LengthsMode> getLengthsMode(ArgumentDictionaryTy &dict) {
    bool length1 = false;
    if (dict.count("length1")) {
      ASSIGN_VALUE_OR_RETURN_ERR(length1, loadInt(dict["length1"]));
    }
    if (length1) {
      return LengthsMode::AllOne;
    }
    return LengthsMode::Variable;
  }

  inline Expected<float> getAvgLength(ArgumentDictionaryTy &dict) {
    float avgLength = NAN;
    if (dict.count("average_lookup_length")) {
      ASSIGN_VALUE_OR_RETURN_ERR(avgLength,
                                 loadFloat(dict["average_lookup_length"]));
    }
    return avgLength;
  }

  const std::string opErrMsg(const OpType &op, const std::string &errMsg) {
    const std::string &opName = loadOperatorName(op);
    return strFormat(" [Operator-'%s'] : %s ", opName.c_str(), errMsg.c_str());
  }

  /// Associate the name of operation outputs to a NodeValues corresponding to
  /// node \p node. If \p numOutputs is lower than 0, then all outputs are
  /// associated. Otherwise, the first \p numOutputs outputs are associated.
  Error addNodeAsOutput(const OpType &op, Node *node, int numOutputs = -1) {
    RETURN_ERR_IF_NOT(numOutputs <= op.output_size(),
                      "Can't register more than outputs in the operation.");
    numOutputs = (numOutputs < 0) ? op.output_size() : numOutputs;
    for (int i = 0; i < numOutputs; i++) {
      nodeValueByName_[op.output(i)] = NodeValue(node, i);
    }
    return Error::success();
  }

  /// Loads RELU operator, given its protobuf representation and parsed args.
  Error loadRelu(const OpType &op, ArgumentDictionaryTy &dict) {
    const std::string &opName = loadOperatorName(op);
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));
    auto *R = G_->createRELU(opName, in);
    RETURN_IF_ERR(addNodeAsOutput(op, R));
    return Error::success();
  }

#define LOAD_UNARY_OP(OPNAME)                                                  \
  Error load##OPNAME(const OpType &op, ArgumentDictionaryTy &dict) {           \
    const std::string &opName = loadOperatorName(op);                          \
    NodeValue in;                                                              \
    ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));           \
    auto *T = G_->create##OPNAME(opName, in);                                  \
    RETURN_IF_ERR(addNodeAsOutput(op, T));                                     \
    return Error::success();                                                   \
  }

  LOAD_UNARY_OP(Sigmoid)
  LOAD_UNARY_OP(Tanh)
  LOAD_UNARY_OP(Exp)
  LOAD_UNARY_OP(Neg)
  LOAD_UNARY_OP(Floor)
  LOAD_UNARY_OP(Ceil)
  LOAD_UNARY_OP(Truncate)
  LOAD_UNARY_OP(Log)

  Error loadShape(const OpType &op, ArgumentDictionaryTy &dict) {
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));

    // This is statically known data, and so we create a Tensor for it.
    Tensor T(ElemKind::Int64ITy, {(dim_t)in.dims().size()});
    T.getHandle<int64_t>() =
        std::vector<int64_t>(in.dims().begin(), in.dims().end());

    RETURN_IF_ERR(createAndRegisterConstant(op.output(0), std::move(T)));

    return Error::success();
  }

  /// Loads Pow operator, given its protobuf representation and parsed args.
  Error loadPow(const OpType &op, ArgumentDictionaryTy &dict) {
    const std::string &opName = loadOperatorName(op);
    NodeValue base;
    ASSIGN_VALUE_OR_RETURN_ERR(base, getNodeValueByName(op.input(0)));
    NodeValue exp;
    ASSIGN_VALUE_OR_RETURN_ERR(exp, getNodeValueByName(op.input(1)));
    auto R = G_->createNodeWithBroadcast<PowNode>(opName, -1, base, exp);
    RETURN_IF_ERR(addNodeAsOutput(op, R));
    return Error::success();
  }

  /// Loads Sqrt operator, given its protobuf representation and parsed args.
  Error loadSqrt(const OpType &op, ArgumentDictionaryTy &dict) {
    const std::string &opName = loadOperatorName(op);
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));
    auto *R = G_->createPow(opName, in, 0.5f);
    RETURN_IF_ERR(addNodeAsOutput(op, R));
    return Error::success();
  }

  /// Loads Sqr operator, given its protobuf representation and parsed args.
  Error loadSqr(const OpType &op, ArgumentDictionaryTy &dict) {
    const std::string &opName = loadOperatorName(op);
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));
    auto *R = G_->createPow(opName, in, 2.0f);
    RETURN_IF_ERR(addNodeAsOutput(op, R));
    return Error::success();
  }

  /// Loads Reciprocal operator, given its protobuf representation and parsed
  /// args.
  Error loadReciprocal(const OpType &op, ArgumentDictionaryTy &dict) {
    const std::string &opName = loadOperatorName(op);
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));
    auto *R = G_->createPow(opName, in, -1.0f);
    RETURN_IF_ERR(addNodeAsOutput(op, R));
    return Error::success();
  }

  Error loadSum(const OpType &op, ArgumentDictionaryTy &dict) {
    if (op.input_size() == 1) {
      NodeValue in;
      ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));
      RETURN_IF_ERR(addNodeAsOutput(op, in));
    } else if (op.input_size() == 2) {
      const std::string &opName = loadOperatorName(op);
      NodeValue in0;
      ASSIGN_VALUE_OR_RETURN_ERR(in0, getNodeValueByName(op.input(0)));
      NodeValue in1;
      ASSIGN_VALUE_OR_RETURN_ERR(in1, getNodeValueByName(op.input(1)));
      auto *node = G_->createAdd(opName, in0, in1);
      RETURN_IF_ERR(addNodeAsOutput(op, node));
    } else {
      const std::string &opName = loadOperatorName(op);
      const unsigned numInputs = op.input_size();
      llvm::SmallVector<NodeValue, 4> inputs;
      inputs.reserve(numInputs);
      for (unsigned i = 0; i < numInputs; i++) {
        NodeValue in;
        ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(i)));
        inputs.push_back(G_->createExpandDims(opName, in, {0}));
      }
      ConcatNode *concat = G_->createConcat(opName, inputs, /* axis */ 0);
      Node *node = G_->createBatchedReduceAdd(opName, concat, /* axis */ {0});
      RETURN_IF_ERR(addNodeAsOutput(op, node));
    }
    return Error::success();
  }

  Error loadLRN(const OpType &op, ArgumentDictionaryTy &dict) {
    const std::string &opName = loadOperatorName(op);
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));

    size_t size;
    ASSIGN_VALUE_OR_RETURN_ERR(size, loadInt(dict["size"]));
    float alpha;
    ASSIGN_VALUE_OR_RETURN_ERR(alpha, loadFloat(dict["alpha"]));
    float beta;
    ASSIGN_VALUE_OR_RETURN_ERR(beta, loadFloat(dict["beta"]));
    float k;
    ASSIGN_VALUE_OR_RETURN_ERR(k, loadFloat(dict["bias"]));

    auto *tr = G_->createTranspose(opName, in, NCHW2NHWC);

    auto *node = G_->createLocalResponseNormalization(opName, tr, size / 2,
                                                      alpha, beta, k);

    auto *N = G_->createTranspose(opName, node, NHWC2NCHW);

    // LRN in Caffe2 has a scale_ output, but I believe it's unused for
    // inference. So explicitly only set output 0.
    nodeValueByName_[op.output(0)] = N->getResult();
    return Error::success();
  }

  Error loadMinMax(llvm::StringRef typeName, const OpType &op,
                   ArgumentDictionaryTy &dict) {
    const std::string &opName = loadOperatorName(op);
    NodeValue in0;
    ASSIGN_VALUE_OR_RETURN_ERR(in0, getNodeValueByName(op.input(0)));
    NodeValue in1;
    ASSIGN_VALUE_OR_RETURN_ERR(in1, getNodeValueByName(op.input(1)));

    Node *node = nullptr;
    if (typeName == "Min") {
      node = G_->createNodeWithBroadcast<MinNode>(opName, -1, in0, in1);
    } else if (typeName == "Max") {
      node = G_->createNodeWithBroadcast<MaxNode>(opName, -1, in0, in1);
    } else {
      return MAKE_ERR(opErrMsg(op, "Invalid min or max operator"));
    }

    RETURN_IF_ERR(addNodeAsOutput(op, node));
    return Error::success();
  }

  static Expected<NodeValue> handleMatMulTranspose(Function *F,
                                                   ArgumentDictionaryTy &dict,
                                                   llvm::StringRef key,
                                                   NodeValue input) {
    if (!dict.count(key.str())) {
      return input;
    }

    int isTransposed;
    ASSIGN_VALUE_OR_RETURN_ERR(isTransposed, loadInt(dict[key.str()]));
    if (isTransposed == 1) {
      auto dimsSize = input.dims().size();
      RETURN_ERR_IF_NOT(dimsSize >= 2,
                        "C2 specs say rank of inputs must be >= 2");

      std::vector<unsigned_t> shuffle;
      unsigned_t i;
      for (i = 0; i < dimsSize - 2; ++i) {
        shuffle.push_back(i);
      }
      shuffle.push_back(i + 1);
      shuffle.push_back(i);

      return F->createTranspose(input.getNode()->getName().str() + ".transpose",
                                input, shuffle);
    }

    return input;
  }

  Error loadMatMul(const OpType &op, ArgumentDictionaryTy &dict) {
    const std::string &opName = loadOperatorName(op);
    NodeValue LHS;
    ASSIGN_VALUE_OR_RETURN_ERR(LHS, getNodeValueByName(op.input(0)));
    NodeValue RHS;
    ASSIGN_VALUE_OR_RETURN_ERR(RHS, getNodeValueByName(op.input(1)));

    ASSIGN_VALUE_OR_RETURN_ERR(LHS,
                               handleMatMulTranspose(G_, dict, "trans_a", LHS));
    ASSIGN_VALUE_OR_RETURN_ERR(RHS,
                               handleMatMulTranspose(G_, dict, "trans_b", RHS));

    Node *node = G_->createMatMul(opName, LHS, RHS);

    RETURN_IF_ERR(addNodeAsOutput(op, node));
    return Error::success();
  }

  Error loadBatchMatMul(const OpType &op, ArgumentDictionaryTy &dict) {
    const std::string &opName = loadOperatorName(op);
    NodeValue LHS;
    ASSIGN_VALUE_OR_RETURN_ERR(LHS, getNodeValueByName(op.input(0)));
    NodeValue RHS;
    ASSIGN_VALUE_OR_RETURN_ERR(RHS, getNodeValueByName(op.input(1)));

    ASSIGN_VALUE_OR_RETURN_ERR(LHS,
                               handleMatMulTranspose(G_, dict, "trans_a", LHS));
    ASSIGN_VALUE_OR_RETURN_ERR(RHS,
                               handleMatMulTranspose(G_, dict, "trans_b", RHS));

    const size_t numDimsLHS = LHS.dims().size();
    const size_t numDimsRHS = RHS.dims().size();
    RETURN_ERR_IF_NOT(
        numDimsLHS >= 2,
        opErrMsg(op, "BatchMatMul 1D operands are not yet supported."));
    RETURN_ERR_IF_NOT(
        numDimsRHS >= 2,
        opErrMsg(op, "BatchMatMul 1D operands are not yet supported."));

    // This is a very simple case when we don't need any broadcasting
    if (numDimsLHS == 2 && numDimsRHS == 2) {
      Node *node = G_->createMatMul(opName, LHS, RHS);
      RETURN_IF_ERR(addNodeAsOutput(op, node));
      return Error::success();
    }

    // In the rest of the function body we:
    // 1. normalize operands using broadcasting rules,
    // 2. convert normalized operands to 3D matrices, so they look like these:
    //    LHS = {numBatches, N, M}
    //    RHS = {numBatches, M, P}
    //    Result = {numBatches, N, P},
    // 3. multiply 3D matrices using createBatchMatMul(), result will be 3D,
    // 4. convert the result to the normalized broadcast shape.

    const dim_t N = LHS.dims()[numDimsLHS - 2];
    const dim_t M = LHS.dims()[numDimsLHS - 1];
    const dim_t P = RHS.dims()[numDimsRHS - 1];

    RETURN_ERR_IF_NOT(
        RHS.dims()[numDimsRHS - 2] == M,
        opErrMsg(op, "BatchMatMul operands dimensions are invalid."));

    // Calculate broadcast shape and convert both operands to that shape
    const std::vector<dim_t> originalDimsLHS{LHS.dims().begin(),
                                             LHS.dims().end()};
    const std::vector<dim_t> originalDimsRHS{RHS.dims().begin(),
                                             RHS.dims().end()};
    std::vector<dim_t> resultShape{P, N};
    resultShape.reserve(std::max(numDimsLHS, numDimsRHS));
    dim_t numBatches = 1;
    int indLHS = numDimsLHS - 3; // skip last two dims
    int indRHS = numDimsRHS - 3; // skip last two dims
    for (; indLHS >= 0 && indRHS >= 0; --indLHS, --indRHS) {
      const dim_t dimLHS = originalDimsLHS[indLHS];
      const dim_t dimRHS = originalDimsRHS[indRHS];

      RETURN_ERR_IF_NOT(
          (dimLHS == dimRHS || (dimLHS == 1) || dimRHS == 1),
          opErrMsg(op, "BatchMatMul dimensions cannot be broadcast."));
      dim_t dim = 1;
      if (dimLHS == dimRHS) {
        dim = dimLHS;
      } else if (dimLHS == 1) {
        dim = dimRHS;
        LHS = G_->createTile(opName + ".tileDim", LHS, dim, indLHS);
      } else {
        dim = dimLHS;
        RHS = G_->createTile(opName + ".tileDim", RHS, dim, indRHS);
      }
      resultShape.push_back(dim);
      numBatches *= dim;
    }
    for (; indLHS >= 0; --indLHS) {
      const dim_t dim = originalDimsLHS[indLHS];
      resultShape.push_back(dim);
      numBatches *= dim;
      RHS = G_->createExpandDims(opName + ".addDim", RHS, {0});
      RHS = G_->createTile(opName + ".tileDim", RHS, dim, 0);
    }
    for (; indRHS >= 0; --indRHS) {
      const dim_t dim = originalDimsRHS[indRHS];
      resultShape.push_back(dim);
      numBatches *= dim;
      LHS = G_->createExpandDims(opName + ".addDim", LHS, {0});
      LHS = G_->createTile(opName + ".tileDim", LHS, dim, 0);
    }
    std::reverse(resultShape.begin(), resultShape.end());

    // Broadcast shape might have more than 3 dims,
    // therefore, optionally, reshape the operands
    if (resultShape.size() > 3) {
      LHS =
          G_->createReshape(opName + ".reshapeLHS3D", LHS, {numBatches, N, M});
      RHS =
          G_->createReshape(opName + ".reshapeRHS3D", RHS, {numBatches, M, P});
    }
    Node *node = G_->createBatchMatMul(opName, LHS, RHS);

    // Optionally, reshape result to broadcast shape
    if (resultShape.size() != 3) {
      node = G_->createReshape(opName + ".reshapeResult", node, resultShape);
    }

    RETURN_IF_ERR(addNodeAsOutput(op, node));
    return Error::success();
  }

  Error loadArithmetic(llvm::StringRef typeName, const OpType &op,
                       ArgumentDictionaryTy &dict) {
    const std::string &opName = loadOperatorName(op);
    NodeValue in0;
    ASSIGN_VALUE_OR_RETURN_ERR(in0, getNodeValueByName(op.input(0)));
    NodeValue in1;
    ASSIGN_VALUE_OR_RETURN_ERR(in1, getNodeValueByName(op.input(1)));

    bool broadcast;
    ASSIGN_VALUE_OR_RETURN_ERR(broadcast, getBroadcast(dict));
    // Check implicit broadcast
    if (!broadcast && in0.dims().size() != in1.dims().size()) {
      bool validBroadcast = true;
      auto dimsA = in0.dims();
      auto dimsB = in1.dims();
      for (int i = dimsA.size() - 1, j = dimsB.size() - 1; i >= 0 && j >= 0;) {
        auto a = dimsA[i];
        auto b = dimsB[j];
        if (!(a == b || a == 1 || b == 1)) {
          validBroadcast = false;
          break;
        }
        --i;
        --j;
      }
      if (!validBroadcast) {
        LOG(WARNING) << "Invalid broadcast rule for inputs of " << opName;
      }
      broadcast = validBroadcast;
    }

    int axis = -1;

    // Broadcasting can be:
    // - multidirectional (ONNX opset 7+), or
    // - unidirectional (ONNX opset 1->6,  Caffe2).

    // Unidirectional broadcasting consists of broadcasting the right operand
    // (in1) so that it matches the shape of the left operand (in0).
    if (broadcast && !hasMultidirectionalBroadcast(typeName)) {
      // With unidirectional broadcasting, the 'axis' attribute specifies
      // from how much the right operand shape must be 'shifted' right.
      // - In Caffe2, the 'axis' attribute is optional. If not specified, axis
      // must be automatically computed so that the trailing-most dimensions
      // of in1 is aligned to the trailing-most dimension of in0.
      // - In ONNX, the 'axis' attribute is mandatory. axis == -1 is
      // equivalent to no axis specified in Caffe2.

      if (dict.count("axis")) {
        ASSIGN_VALUE_OR_RETURN_ERR(axis, loadInt(dict["axis"]));
      }
      if (axis == -1) {
        // Align trailing most dimensions.
        axis = in0.dims().size() - in1.dims().size();
      }
    }

    Node *node = nullptr;
    if (broadcast) {
      if (typeName == "Mul") {
        node = G_->createNodeWithBroadcast<MulNode>(opName, axis, in0, in1);
      } else if (typeName == "Add") {
        node = G_->createNodeWithBroadcast<AddNode>(opName, axis, in0, in1);
      } else if (typeName == "Sub") {
        node = G_->createNodeWithBroadcast<SubNode>(opName, axis, in0, in1);
      } else if (typeName == "Div") {
        node = G_->createNodeWithBroadcast<DivNode>(opName, axis, in0, in1);
      } else if (typeName == "Pow") {
        node = G_->createNodeWithBroadcast<PowNode>(opName, axis, in0, in1);
      } else {
        return MAKE_ERR("Unsupported arithmetic typeName");
      }
    } else {
      if (typeName == "Mul") {
        node = G_->createMul(opName, in0, in1);
      } else if (typeName == "Add") {
        node = G_->createAdd(opName, in0, in1);
      } else if (typeName == "Sub") {
        node = G_->createSub(opName, in0, in1);
      } else if (typeName == "Div") {
        node = G_->createDiv(opName, in0, in1);
      } else if (typeName == "Pow") {
        node = G_->createPow(opName, in0, in1);
      } else {
        return MAKE_ERR("Unsupported arithmetic typeName");
      }
    }

    RETURN_IF_ERR(addNodeAsOutput(op, node));
    return Error::success();
  }

  Error loadSplit(const OpType &op, ArgumentDictionaryTy &dict) {
    const std::string &opName = loadOperatorName(op);
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));
    size_t axis = 0;
    if (dict.count("axis")) {
      ASSIGN_VALUE_OR_RETURN_ERR(
          axis, loadAxis<size_t>(dict["axis"], in.dims().size()));
    }

    std::vector<dim_t> split;
    if (dict.count("split")) {
      ASSIGN_VALUE_OR_RETURN_ERR(split, getShape<dim_t>(dict["split"]));
    }

    std::vector<SliceNode *> outputs;
    G_->createSplit(opName, in, op.output_size(), axis, split, outputs);

    for (int i = 0, e = op.output_size(); i < e; i++) {
      // Each output from Split is a SliceNode which only has a single output,
      // so only use 0 here as the node value result.
      nodeValueByName_[op.output(i)] = outputs[i]->getResult();
    }
    return Error::success();
  }

  Error loadReshape(const OpType &op, ArgumentDictionaryTy &dict) {
    const std::string &opName = loadOperatorName(op);
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));

    // Get the requested shape from the model.
    // First look at input tensors, then at the "shape" attribute.
    std::vector<dim_t> requestedDims;
    if (op.input_size() > 1) {
      if (!getConstantByNameOrNull(op.input(1))) {
        return MAKE_ERR(opErrMsg(
            op,
            "Reshape: Non-constant shape tensors are unsupported by Glow."));
      }
      const Constant *constShapeConst;
      ASSIGN_VALUE_OR_RETURN_ERR(constShapeConst,
                                 getConstantByName(op.input(1)));
      auto TH = constShapeConst->getPayload().getHandle<int64_t>();
      for (auto dim : TH) {
        requestedDims.push_back(dim);
      }
    } else if (dict.count("shape")) {
      RETURN_ERR_IF_NOT(
          op.input_size() == 1,
          opErrMsg(
              op,
              "Reshape: Cannot specify new shape by both argument and input."));
      std::vector<int64_t> protoDims;
      ASSIGN_VALUE_OR_RETURN_ERR(protoDims, getShape<int64_t>(dict["shape"]));

      for (auto dim : protoDims) {
        requestedDims.push_back(dim);
      }
    } else {
      return MAKE_ERR(opErrMsg(op,
                               "Reshape: Missing output shape information for "
                               "the Reshape operator."));
    }

    // Compute the actual new shape
    ssize_t negOneIndex = -1;
    llvm::ArrayRef<dim_t> inputDims = in.dims();
    std::vector<dim_t> outputDims;
    int64_t dimProduct = 1;
    for (size_t i = 0, e = requestedDims.size(); i != e; i++) {
      dim_t newDim = requestedDims[i];
      if (newDim == 0) {
        // 0 means that corresponding input dimension should be propagated to
        // the output.
        newDim = inputDims[i];
      }
      if (newDim != (dim_t)-1) {
        dimProduct *= newDim;
        outputDims.push_back(newDim);
      } else {
        // -1 means that the corresponding dimension should be inferred
        // from all other dimensions, so that tensor size remains the same.
        RETURN_ERR_IF_NOT(
            negOneIndex < 0,
            opErrMsg(
                op,
                "Reshape: At most one dimension of the new shape can be -1."));
        negOneIndex = (ssize_t)i;
        // The -1 case value is handled later.
        outputDims.push_back(0);
      }
    }
    if (negOneIndex >= 0) {
      outputDims[negOneIndex] = in.getType()->size() / dimProduct;
    }

    auto *node = G_->createReshape(opName, in, outputDims);

    // Caffe2 sometimes outputs old_shape which goes unused. We do not currently
    // support it, so explicitly only set the first output.
    nodeValueByName_[op.output(0)] = node->getResult();
    return Error::success();
  }

  Error loadTranspose(const OpType &op, ArgumentDictionaryTy &dict,
                      llvm::StringRef permArgName) {
    const std::string &opName = loadOperatorName(op);
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));

    // There is a difference between ONNX and Caffe2 specs for Transpose:
    // one contains permutation under name "perm", the other contains it under
    // argument name "axes". That's why the name is passed as a parameter.
    std::vector<unsigned_t> perm;
    if (dict.count(permArgName.str()))
      ASSIGN_VALUE_OR_RETURN_ERR(perm,
                                 getShape<unsigned_t>(dict[permArgName.str()]));

    if (perm.empty()) {
      // Empty permutation argument means reversing axes order.
      size_t N = in.dims().size();
      for (int64_t i = N - 1; i >= 0; i--)
        perm.push_back(i);
    }

    auto *T = G_->createTranspose(opName, in, perm);

    RETURN_IF_ERR(addNodeAsOutput(op, T));
    return Error::success();
  }

  Error loadFlatten(const OpType &op, ArgumentDictionaryTy &dict) {
    const std::string &opName = loadOperatorName(op);
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));
    int axis = 1;
    if (dict.count("axis")) {
      ASSIGN_VALUE_OR_RETURN_ERR(axis,
                                 loadAxis<int>(dict["axis"], in.dims().size()));
    }
    auto *node = G_->createFlatten(opName, in, axis);
    RETURN_IF_ERR(addNodeAsOutput(op, node));
    return Error::success();
  }

  Error loadIdentity(const OpType &op, ArgumentDictionaryTy &dict) {
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));

    // If loading partitioned DAG then check if this identity is used for an
    // intermediate, and if so create the Save+PH with the correct name.
    if (partNameToFun_.size()) {
      int intermediate = 0;
      if (dict.count("isIntermediateOutputForDAG")) {
        ASSIGN_VALUE_OR_RETURN_ERR(
            intermediate, loadInt(dict.at("isIntermediateOutputForDAG")));
      }

      if (intermediate) {
        const std::string &opName = loadOperatorName(op);
        Placeholder *PH = mod_.getPlaceholderByNameSlow(op.output(0));
        if (!PH) {
          PH = mod_.createPlaceholder(in.getType(), op.output(0),
                                      /* isTrainable */ false);
        } else {
          RETURN_ERR_IF_NOT(
              loadIntoExistingModule_,
              opErrMsg(op, "Found pre-existing PH by name " + op.output(0)));
          RETURN_ERR_IF_NOT(
              PH->getType()->isEqual(in.getType()),
              opErrMsg(op, "Mismatch on pre-existing intermediate PH type"));
        }
        G_->createSave(opName, in, PH, /* skipSuffix */ true);
        intermediatePHsByName_[op.output(0)] = PH;
        in = PH->getOutput();
      }
    }

    nodeValueByName_[op.output(0)] = in;
    return Error::success();
  }

  Error loadReduceOp(llvm::StringRef typeName, const OpType &op,
                     ArgumentDictionaryTy &dict) {
    const std::string &opName = loadOperatorName(op);
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));

    std::vector<unsigned_t> shapeAxes = {};
    if (dict.count("axes")) {
      ASSIGN_VALUE_OR_RETURN_ERR(
          shapeAxes, loadAxes<unsigned_t>(dict["axes"], in.dims().size()));
    } else {
      shapeAxes.resize(in.dims().size());
      std::iota(shapeAxes.begin(), shapeAxes.end(), 0);
    }

    std::sort(shapeAxes.begin(), shapeAxes.end());

    llvm::ArrayRef<unsigned_t> axes(shapeAxes);

    // Check if axes elements are unique.
    if (axes.size() > 1) {
      auto it = std::unique(shapeAxes.begin(), shapeAxes.end());
      if (it != shapeAxes.end()) {
        return MAKE_ERR(opErrMsg(op, "ReduceOp Axes values are not unique."),
                        ErrorValue::ErrorCode::MODEL_LOADER_UNSUPPORTED_SHAPE);
      }
    }

    bool keepDims = true;
    if (dict.count("keepdims")) {
      int keepdims;
      ASSIGN_VALUE_OR_RETURN_ERR(keepdims, loadInt(dict["keepdims"]));
      keepDims = (bool)keepdims;
    }

    NodeValue node;
    if (typeName == "ReduceMean") {
      node = G_->createBatchedReduceMean(opName, in, axes);
    } else if (typeName == "ReduceSum") {
      node = G_->createBatchedReduceAdd(opName, in, axes);
    } else if (typeName == "ReduceMin") {
      node = G_->createBatchedReduceMin(opName, in, axes);
    } else if (typeName == "ReduceMax") {
      node = G_->createBatchedReduceMax(opName, in, axes);
    } else if (typeName == "ReduceProd") {
      node = G_->createBatchedReduceProd(opName, in, axes);
    } else if (typeName == "ReduceSumSquare") {
      node = G_->createBatchedReduceSumSquare(opName, in, axes);
    } else {
      return MAKE_ERR("Unsupported Reduce Op " + typeName.str());
    }

    // Our batched reduce add/mean does not keep the dim; reshape if necessary.
    if (keepDims) {

      std::vector<dim_t> shape = node.dims();

      // Add removed axes. Requires increasing order sort - done above.
      for (const auto &axis : shapeAxes) {
        shape.insert(shape.begin() + axis, 1);
      }
      node = G_->createReshape(opName, node, shape);
    }

    RETURN_IF_ERR(addNodeAsOutput(op, node));
    return Error::success();
  }

  Error loadBatchOneHot(const OpType &op) {
    const std::string &opName = loadOperatorName(op);
    NodeValue data;
    ASSIGN_VALUE_OR_RETURN_ERR(data, getNodeValueByName(op.input(0)));
    NodeValue lengths;
    ASSIGN_VALUE_OR_RETURN_ERR(lengths, getNodeValueByName(op.input(1)));
    NodeValue values;
    ASSIGN_VALUE_OR_RETURN_ERR(values, getNodeValueByName(op.input(2)));

    auto *node = G_->createBatchOneHot(opName, data, lengths, values);
    RETURN_IF_ERR(addNodeAsOutput(op, node));
    return Error::success();
  }

  Error loadSparseLengthsSum(const OpType &op, ArgumentDictionaryTy &dict) {
    NodeValue in0;
    ASSIGN_VALUE_OR_RETURN_ERR(in0, getNodeValueByName(op.input(0)));
    NodeValue in1;
    ASSIGN_VALUE_OR_RETURN_ERR(in1, getNodeValueByName(op.input(1)));
    NodeValue in2;
    ASSIGN_VALUE_OR_RETURN_ERR(in2, getNodeValueByName(op.input(2)));
    LengthsMode lengthsMode;
    ASSIGN_VALUE_OR_RETURN_ERR(lengthsMode, getLengthsMode(dict));
    float avgLength;
    ASSIGN_VALUE_OR_RETURN_ERR(avgLength, getAvgLength(dict));
    auto *node = G_->createSparseLengthsSum(loadOperatorName(op), in0, in1, in2,
                                            lengthsMode, avgLength);
    RETURN_IF_ERR(addNodeAsOutput(op, node));
    return Error::success();
  }

  Error loadSparseLengthsWeightedSum(const OpType &op,
                                     ArgumentDictionaryTy &dict) {
    NodeValue in0;
    ASSIGN_VALUE_OR_RETURN_ERR(in0, getNodeValueByName(op.input(0)));
    NodeValue in1;
    ASSIGN_VALUE_OR_RETURN_ERR(in1, getNodeValueByName(op.input(1)));
    NodeValue in2;
    ASSIGN_VALUE_OR_RETURN_ERR(in2, getNodeValueByName(op.input(2)));
    NodeValue in3;
    ASSIGN_VALUE_OR_RETURN_ERR(in3, getNodeValueByName(op.input(3)));
    LengthsMode lengthsMode;
    ASSIGN_VALUE_OR_RETURN_ERR(lengthsMode, getLengthsMode(dict));
    float avgLength;
    ASSIGN_VALUE_OR_RETURN_ERR(avgLength, getAvgLength(dict));
    auto *node = G_->createSparseLengthsWeightedSum(
        loadOperatorName(op), in0, in1, in2, in3, lengthsMode, avgLength);
    RETURN_IF_ERR(addNodeAsOutput(op, node));
    return Error::success();
  }

  Error loadEmbedding(const OpType &op, ArgumentDictionaryTy &dict) {
    NodeValue weights;
    ASSIGN_VALUE_OR_RETURN_ERR(weights, getNodeValueByName(op.input(0)));

    NodeValue indices;
    ASSIGN_VALUE_OR_RETURN_ERR(indices, getNodeValueByName(op.input(1)));

    int64_t padIdx = -1;
    if (dict.count("padIdx")) {
      ASSIGN_VALUE_OR_RETURN_ERR(padIdx, loadInt(dict["padIdx"]));
    }
    bool scale = false;
    if (dict.count("scale")) {
      ASSIGN_VALUE_OR_RETURN_ERR(scale, loadInt(dict["scale"]));
      scale = (bool)scale;
      RETURN_ERR_IF_NOT(scale == false,
                        "Currently only support scale_grad_by_freq == 'false'");
    }
    bool sparse = false;
    if (dict.count("sparse")) {
      ASSIGN_VALUE_OR_RETURN_ERR(sparse, loadInt(dict["sparse"]));
      sparse = (bool)sparse;
      RETURN_ERR_IF_NOT(sparse == false,
                        "Currently only support sparse == 'false'");
    }
    auto *node = G_->createEmbedding(loadOperatorName(op), weights, indices,
                                     padIdx, scale, sparse);
    RETURN_IF_ERR(addNodeAsOutput(op, node));
    return Error::success();
  }

  Error loadEmbeddingBag(const OpType &op, ArgumentDictionaryTy &dict) {
    NodeValue in0;
    ASSIGN_VALUE_OR_RETURN_ERR(in0, getNodeValueByName(op.input(0)));
    NodeValue in1;
    ASSIGN_VALUE_OR_RETURN_ERR(in1, getNodeValueByName(op.input(1)));
    NodeValue in2;
    ASSIGN_VALUE_OR_RETURN_ERR(in2, getNodeValueByName(op.input(2)));
    NodeValue in3;
    ASSIGN_VALUE_OR_RETURN_ERR(in3, getNodeValueByName(op.input(3)));
    LengthsMode lengthsMode;
    ASSIGN_VALUE_OR_RETURN_ERR(lengthsMode, getLengthsMode(dict));
    float avgLength;
    ASSIGN_VALUE_OR_RETURN_ERR(avgLength, getAvgLength(dict));
    auto *node = G_->createEmbeddingBag(
        loadOperatorName(op), in0, in1, in2, in3,
        /* hasEndOffset */ false, lengthsMode, avgLength);
    RETURN_IF_ERR(addNodeAsOutput(op, node));
    return Error::success();
  }

  Error loadLengthsToRanges(const OpType &op) {
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));
    auto *node = G_->createLengthsToRanges(loadOperatorName(op), in);
    RETURN_IF_ERR(addNodeAsOutput(op, node));
    return Error::success();
  }

  Error loadBatchBoxCox(const OpType &op) {
    NodeValue data;
    ASSIGN_VALUE_OR_RETURN_ERR(data, getNodeValueByName(op.input(0)));
    NodeValue lambda1;
    ASSIGN_VALUE_OR_RETURN_ERR(lambda1, getNodeValueByName(op.input(1)));
    NodeValue lambda2;
    ASSIGN_VALUE_OR_RETURN_ERR(lambda2, getNodeValueByName(op.input(2)));
    auto *node =
        G_->createBatchBoxCox(loadOperatorName(op), data, lambda1, lambda2);
    RETURN_IF_ERR(addNodeAsOutput(op, node));
    return Error::success();
  }

  Error loadDotProduct(const OpType &op) {
    NodeValue X;
    ASSIGN_VALUE_OR_RETURN_ERR(X, getNodeValueByName(op.input(0)));
    NodeValue Y;
    ASSIGN_VALUE_OR_RETURN_ERR(Y, getNodeValueByName(op.input(1)));
    RETURN_ERR_IF_NOT(X.dims() == Y.dims(),
                      opErrMsg(op, "X and Y must have the same dimensions"));
    auto *node = G_->createDotProduct(loadOperatorName(op), X, Y);
    RETURN_IF_ERR(addNodeAsOutput(op, node));
    return Error::success();
  }

  Error loadReplaceNaN(const OpType &op, ArgumentDictionaryTy &dict) {
    // Load the input and NaN replacement value:
    NodeValue input;
    ASSIGN_VALUE_OR_RETURN_ERR(input, getNodeValueByName(op.input(0)));
    auto valueIt = dict.find("value");
    float value = 0.0f;
    if (valueIt != dict.end()) {
      ASSIGN_VALUE_OR_RETURN_ERR(value, loadFloat(valueIt->second));
    }
    auto *node = G_->createReplaceNaN(loadOperatorName(op), input, value);
    RETURN_IF_ERR(addNodeAsOutput(op, node));
    return Error::success();
  }

  Error loadLengthsSum(const OpType &op) {
    const std::string &opName = loadOperatorName(op);
    NodeValue data;
    ASSIGN_VALUE_OR_RETURN_ERR(data, getNodeValueByName(op.input(0)));
    NodeValue lengths;
    ASSIGN_VALUE_OR_RETURN_ERR(lengths, getNodeValueByName(op.input(1)));

    RETURN_ERR_IF_NOT(
        lengths.dims().size() == 1,
        opErrMsg(
            op,
            strFormat("LengthsSum: Lengths must be a 1D vector, but found %zu ",
                      lengths.dims().size())));

    auto *node = G_->createLengthsSum(opName, data, lengths);
    RETURN_IF_ERR(addNodeAsOutput(op, node));
    return Error::success();
  }

  Error loadExpandDims(const OpType &op, ArgumentDictionaryTy &dict) {
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));
    std::vector<dim_t> shape;
    ASSIGN_VALUE_OR_RETURN_ERR(shape, getShape<dim_t>(dict["dims"]));

    Node *node = G_->createExpandDims(loadOperatorName(op), in, shape);
    RETURN_IF_ERR(addNodeAsOutput(op, node));

    return Error::success();
  }

  Error loadSparseToDense(const OpType &op, ArgumentDictionaryTy &dict) {
    if (op.input_size() != 3) {
      return MAKE_ERR(opErrMsg(
          op,
          strFormat(
              "SparseToDense operator must have three inputs, but found %d ",
              op.input_size())));
    }

    NodeValue indices;
    ASSIGN_VALUE_OR_RETURN_ERR(indices, getNodeValueByName(op.input(0)));
    NodeValue values;
    ASSIGN_VALUE_OR_RETURN_ERR(values, getNodeValueByName(op.input(1)));
    NodeValue dataToInferDim;
    ASSIGN_VALUE_OR_RETURN_ERR(dataToInferDim, getNodeValueByName(op.input(2)));

    RETURN_ERR_IF_NOT(indices.dims().size() == 1 || indices.dims().size() == 2,
                      opErrMsg(op, "Indices must be 1D or 2D tensor."));
    RETURN_ERR_IF_NOT(indices.getElementType() == ElemKind::Int32ITy ||
                          indices.getElementType() == ElemKind::Int64ITy,
                      opErrMsg(op, "Indices must be of int32 or int64 type."));

    const std::string &opName = loadOperatorName(op);

    if (indices.dims().size() == 1) {
      indices = G_->createReshape(opName + ".indices2D", indices,
                                  {indices.dims()[0], 1});
    } else {
      RETURN_ERR_IF_NOT(
          indices.dims()[1] == 1,
          opErrMsg(op, "Second dimension should be 1 in indices."));
    }

    ShapeVector outDims{values.dims().begin(), values.dims().end()};
    outDims[0] = dataToInferDim.dims()[0];
    auto outTy =
        G_->getParent()->uniqueTypeWithNewShape(values.getType(), outDims);
    Node *data = G_->createSplat(opName + ".data", outTy, 0.f);

    // SparseToDense has very similar behavior to ScatterND from ONNX
    // https://github.com/onnx/onnx/blob/master/docs/Operators.md#ScatterND,
    // therefore we can use ScatterND to implement SparseToDense.
    Node *result = G_->createScatterData(opName + ".scatterData", data, indices,
                                         values, true);

    RETURN_IF_ERR(addNodeAsOutput(op, result));
    return Error::success();
  }

  Error loadSparseToDenseMask(const OpType &op, ArgumentDictionaryTy &dict) {
    size_t inputSize = op.input_size();
    if (inputSize != 3 && inputSize != 4) {
      return MAKE_ERR(
          opErrMsg(op, strFormat("SparseToDenseMask operator must have "
                                 "3 or 4 inputs, but found %zu ",
                                 inputSize)));
    }

    NodeValue indices;
    ASSIGN_VALUE_OR_RETURN_ERR(indices, getNodeValueByName(op.input(0)));
    NodeValue values;
    ASSIGN_VALUE_OR_RETURN_ERR(values, getNodeValueByName(op.input(1)));
    NodeValue defaultValue;
    ASSIGN_VALUE_OR_RETURN_ERR(defaultValue, getNodeValueByName(op.input(2)));

    NodeValue lengths;
    if (inputSize == 4) {
      ASSIGN_VALUE_OR_RETURN_ERR(lengths, getNodeValueByName(op.input(3)));
    } else {
      // If Lengths input is not present, create scalar containing number of
      // index-value pairs.
      auto *lengthsConstant =
          mod_.createConstant(ElemKind::Int32ITy, {}, "lengthsConstant");
      lengthsConstant->getPayloadMutable().template getHandle<int32_t>().raw(
          0) = indices.dims()[0];
      lengths = lengthsConstant->getOutput();
    }

    std::vector<dim_t> mask;
    ASSIGN_VALUE_OR_RETURN_ERR(mask, getShape<dim_t>(dict["mask"]));

    auto *node = G_->createSparseToDenseMask(
        loadOperatorName(op), indices, values, defaultValue, lengths, mask);
    RETURN_IF_ERR(addNodeAsOutput(op, node));
    return Error::success();
  }

  Error loadGatherOps(const std::string &typeName, const OpType &op,
                      const ArgumentDictionaryTy &dict) {

    NodeValue data;
    ASSIGN_VALUE_OR_RETURN_ERR(data, getNodeValueByName(op.input(0)));
    NodeValue indices;
    ASSIGN_VALUE_OR_RETURN_ERR(indices, getNodeValueByName(op.input(1)));
    size_t axis = typeName == "Gather" ? 0 : 1;

    if (dict.count("axis")) {
      ASSIGN_VALUE_OR_RETURN_ERR(
          axis,
          loadAxis<size_t>(dict.find("axis")->second, data.dims().size()));
    }

    if (indices.getElementType() != ElemKind::Int64ITy &&
        indices.getElementType() != ElemKind::Int32ITy) {
      // If the index type is not Int32 or Int64 insert a conversion layer to
      // introduce robustness against model problems. Constant Float indices
      // will get converted to integer indices via constant folding pass.
      indices = G_->createConvertTo(
          loadOperatorName(op) + "_idx_convertToi32", indices,
          G_->getParent()->uniqueType(ElemKind::Int32ITy, indices.dims()));
    }

    auto *GN = G_->createGather(loadOperatorName(op), data, indices, axis);
    RETURN_IF_ERR(addNodeAsOutput(op, GN));
    return Error::success();
  }

  Error loadGatherND(const std::string &typeName, const OpType &op,
                     const ArgumentDictionaryTy &dict) {

    NodeValue data;
    ASSIGN_VALUE_OR_RETURN_ERR(data, getNodeValueByName(op.input(0)));
    NodeValue indices;
    ASSIGN_VALUE_OR_RETURN_ERR(indices, getNodeValueByName(op.input(1)));

    if (indices.getElementType() != ElemKind::Int64ITy &&
        indices.getElementType() != ElemKind::Int32ITy) {
      // If the index type is not Int32 or Int64 insert a conversion layer to
      // introduce robustness against model problems. Constant Float indices
      // will get converted to integer indices via constant folding pass.
      indices = G_->createConvertTo(
          loadOperatorName(op) + "_idx_convertToi32", indices,
          G_->getParent()->uniqueType(ElemKind::Int32ITy, indices.dims()));
    }

    auto *GN = G_->createGatherND(loadOperatorName(op), data, indices);
    RETURN_IF_ERR(addNodeAsOutput(op, GN));
    return Error::success();
  }

  Error loadGatherRanges(const std::string &typeName, const OpType &op,
                         ArgumentDictionaryTy &dict) {
    NodeValue data;
    ASSIGN_VALUE_OR_RETURN_ERR(data, getNodeValueByName(op.input(0)));
    RETURN_ERR_IF_NOT(
        data.dims().size() == 1,
        opErrMsg(op, strFormat("GatherRanges: Data must be a 1D vector, but "
                               "found vector size %zu ",
                               data.dims().size())));

    NodeValue ranges;
    ASSIGN_VALUE_OR_RETURN_ERR(ranges, getNodeValueByName(op.input(1)));
    RETURN_ERR_IF_NOT(
        ranges.dims().size() == 3,
        opErrMsg(op, strFormat("GatherRanges: Ranges must be a 3D vector, but "
                               "found vector size %zu ",
                               ranges.dims().size())));
    RETURN_ERR_IF_NOT(
        ranges.dims()[2] == 2,
        opErrMsg(op, strFormat("GatherRanges: Last dimension of "
                               "ranges must be 2, but found %s",
                               std::to_string(ranges.dims()[2]).c_str())));

    auto maxOutputSizeIt = dict.find("maxOutputSize");
    RETURN_ERR_IF_NOT(maxOutputSizeIt != dict.end(),
                      opErrMsg(op, "GatherRanges: Require maxOutputSize when "
                                   "loading LengthsRangeFill."));
    unsigned_t maxOutputSize;
    ASSIGN_VALUE_OR_RETURN_ERR(maxOutputSize, loadInt(maxOutputSizeIt->second));

    Node *GR = G_->createGatherRanges(loadOperatorName(op), data, ranges,
                                      maxOutputSize);
    RETURN_IF_ERR(addNodeAsOutput(op, GR));
    return Error::success();
  }

  // Loads Less operator. Internally it's a cmpLT Node.
  Error loadLess(const OpType &op, ArgumentDictionaryTy &dict) {
    // Input Type.
    NodeValue xNV;
    ASSIGN_VALUE_OR_RETURN_ERR(xNV, getNodeValueByName(op.input(0)));
    NodeValue yNV;
    ASSIGN_VALUE_OR_RETURN_ERR(yNV, getNodeValueByName(op.input(1)));

    std::string opName = loadOperatorName(op);

    auto *xNode = xNV.getNode();
    auto *yNode = yNV.getNode();

    Node *N = G_->createNodeWithBroadcast<CmpLTNode>(opName, /* axis */ -1,
                                                     xNode, yNode);

    RETURN_IF_ERR(addNodeAsOutput(op, N));
    return Error::success();
  }

  Error loadLogicalOps(llvm::StringRef typeName, const OpType &op) {
    std::string opName = loadOperatorName(op);
    NodeValue xNV;
    ASSIGN_VALUE_OR_RETURN_ERR(xNV, getNodeValueByName(op.input(0)));
    NodeValue yNV;
    ASSIGN_VALUE_OR_RETURN_ERR(yNV, getNodeValueByName(op.input(1)));
    constexpr int axis = -1;
    Node *N = nullptr;
    if (typeName == "And") {
      N = G_->createNodeWithBroadcast<AndNode>(opName, axis, xNV, yNV);
    } else if (typeName == "Or") {
      N = G_->createNodeWithBroadcast<OrNode>(opName, axis, xNV, yNV);
    } else if (typeName == "Xor") {
      N = G_->createNodeWithBroadcast<XorNode>(opName, axis, xNV, yNV);
    } else {
      return MAKE_ERR("Unsupported Logical Operator");
    }
    RETURN_IF_ERR(addNodeAsOutput(op, N));
    return Error::success();
  }

  Error loadNotOp(llvm::StringRef typeName, const OpType &op) {
    std::string opName = loadOperatorName(op);
    NodeValue xNV;
    ASSIGN_VALUE_OR_RETURN_ERR(xNV, getNodeValueByName(op.input(0)));
    Node *N = G_->createNot(opName, xNV);
    RETURN_IF_ERR(addNodeAsOutput(op, N));
    return Error::success();
  }

  // Loads Abs operator
  Error loadAbs(const OpType &op, ArgumentDictionaryTy &dict) {
    std::string opName = loadOperatorName(op);
    NodeValue xNV;
    ASSIGN_VALUE_OR_RETURN_ERR(xNV, getNodeValueByName(op.input(0)));
    auto *input = xNV.getNode();

    auto *N = G_->createAbs(opName, input);
    RETURN_IF_ERR(addNodeAsOutput(op, N));
    return Error::success();
  }

  /// If operator type is supported, returns Expected<true> and creates new
  /// operator. Returns Operator<false> if operator type is not supported.
  /// Returns Error if an error occurred
  Expected<bool> tryLoadCommonOperator(llvm::StringRef typeName,
                                       const OpType &op,
                                       ArgumentDictionaryTy &dict) {
    if (typeName == "Relu") {
      RETURN_IF_ERR(loadRelu(op, dict));
      return true;
    }
    if (typeName == "Sigmoid") {
      RETURN_IF_ERR(loadSigmoid(op, dict));
      return true;
    }
    if (typeName == "Tanh") {
      RETURN_IF_ERR(loadTanh(op, dict));
      return true;
    }
    if (typeName == "Exp") {
      RETURN_IF_ERR(loadExp(op, dict));
      return true;
    }
    if (typeName == "Log") {
      RETURN_IF_ERR(loadLog(op, dict));
      return true;
    }
    if (typeName == "Neg") {
      RETURN_IF_ERR(loadNeg(op, dict));
      return true;
    }
    if (typeName == "Abs") {
      RETURN_IF_ERR(loadAbs(op, dict));
      return true;
    }
    if (typeName == "Ceil") {
      RETURN_IF_ERR(loadCeil(op, dict));
      return true;
    }
    if (typeName == "Floor") {
      RETURN_IF_ERR(loadFloor(op, dict));
      return true;
    }
    if (typeName == "Shape") {
      RETURN_IF_ERR(loadShape(op, dict));
      return true;
    }
    if (typeName == "Sqrt") {
      RETURN_IF_ERR(loadSqrt(op, dict));
      return true;
    }
    if (typeName == "Sqr") {
      RETURN_IF_ERR(loadSqr(op, dict));
      return true;
    }
    if (typeName == "Reciprocal") {
      RETURN_IF_ERR(loadReciprocal(op, dict));
      return true;
    }
    if (typeName == "Sum") {
      RETURN_IF_ERR(loadSum(op, dict));
      return true;
    }
    if (typeName == "LRN") {
      RETURN_IF_ERR(loadLRN(op, dict));
      return true;
    }
    if (typeName == "Min" || typeName == "Max") {
      RETURN_IF_ERR(loadMinMax(typeName, op, dict));
      return true;
    }
    if (typeName == "Mul" || typeName == "Add" || typeName == "Sub" ||
        typeName == "Div" || typeName == "Pow") {
      RETURN_IF_ERR(loadArithmetic(typeName, op, dict));
      return true;
    }
    if (typeName == "Split") {
      RETURN_IF_ERR(loadSplit(op, dict));
      return true;
    }
    if (typeName == "Reshape") {
      RETURN_IF_ERR(loadReshape(op, dict));
      return true;
    }
    if (typeName == "Flatten") {
      RETURN_IF_ERR(loadFlatten(op, dict));
      return true;
    }
    if (typeName == "Dropout") {
      // Save the identity operation. Note the second output (mask) for Dropout
      // in Caffe2 and ONNX is unused during inference, and our Dropout does not
      // currently implement it, thus we do not call addNodeAsOutput() here.
      RETURN_IF_ERR(loadIdentity(op, dict));
      return true;
    }
    if (typeName == "Identity" || typeName == "Alias") {
      RETURN_IF_ERR(loadIdentity(op, dict));
      return true;
    }
    if (typeName == "ReduceMean" || typeName == "ReduceSum" ||
        typeName == "ReduceMin" || typeName == "ReduceMax" ||
        typeName == "ReduceProd") {
      RETURN_IF_ERR(loadReduceOp(typeName, op, dict));
      return true;
    }
    if (typeName == "BatchMatMul") {
      RETURN_IF_ERR(loadBatchMatMul(op, dict));
      return true;
    }
    if (typeName == "BatchOneHot") {
      RETURN_IF_ERR(loadBatchOneHot(op));
      return true;
    }
    if (typeName == "SparseLengthsSum") {
      RETURN_IF_ERR(loadSparseLengthsSum(op, dict));
      return true;
    }
    if (typeName == "SparseLengthsWeightedSum") {
      RETURN_IF_ERR(loadSparseLengthsWeightedSum(op, dict));
      return true;
    }
    if (typeName == "EmbeddingBag") {
      RETURN_IF_ERR(loadEmbeddingBag(op, dict));
      return true;
    }
    if (typeName == "Embedding") {
      RETURN_IF_ERR(loadEmbedding(op, dict));
      return true;
    }
    if (typeName == "LengthsToRanges") {
      RETURN_IF_ERR(loadLengthsToRanges(op));
      return true;
    }
    if (typeName == "BatchBoxCox") {
      RETURN_IF_ERR(loadBatchBoxCox(op));
      return true;
    }
    if (typeName == "DotProduct") {
      RETURN_IF_ERR(loadDotProduct(op));
      return true;
    }
    if (typeName == "ReplaceNaN") {
      RETURN_IF_ERR(loadReplaceNaN(op, dict));
      return true;
    }
    if (typeName == "LengthsSum") {
      RETURN_IF_ERR(loadLengthsSum(op));
      return true;
    }
    if (typeName == "ExpandDims") {
      RETURN_IF_ERR(loadExpandDims(op, dict));
      return true;
    }
    if (typeName == "SparseToDense") {
      RETURN_IF_ERR(loadSparseToDense(op, dict));
      return true;
    }
    if (typeName == "SparseToDenseMask") {
      RETURN_IF_ERR(loadSparseToDenseMask(op, dict));
      return true;
    }
    if (typeName == "Gather" || typeName == "BatchGather") {
      RETURN_IF_ERR(loadGatherOps(typeName.str(), op, dict));
      return true;
    }
    if (typeName == "GatherND") {
      RETURN_IF_ERR(loadGatherND(typeName.str(), op, dict));
      return true;
    }
    if (typeName == "GatherRanges") {
      RETURN_IF_ERR(loadGatherRanges(typeName.str(), op, dict));
      return true;
    }
    if (typeName == "Less") {
      RETURN_IF_ERR(loadLess(op, dict));
      return true;
    }
    if (typeName == "And" || typeName == "Or" || typeName == "Xor") {
      RETURN_IF_ERR(loadLogicalOps(typeName, op));
      return true;
    }
    if (typeName == "Not") {
      RETURN_IF_ERR(loadNotOp(typeName, op));
      return true;
    }
    if (typeName == "Pow") {
      RETURN_IF_ERR(loadPow(op, dict));
      return true;
    }

    return false;
  }

  /// Utility function which computes the resulting shape in case of
  /// multidirectional broadcasting.
  Expected<std::vector<dim_t>>
  computeMultidirectionalBroadcast(llvm::ArrayRef<dim_t> shape0,
                                   llvm::ArrayRef<dim_t> shape1) {
    size_t numDims0 = shape0.size();
    size_t numDims1 = shape1.size();
    size_t newNumDims = numDims0 > numDims1 ? numDims0 : numDims1;
    std::vector<dim_t> reshapeDims(newNumDims);

    for (size_t i = 0; i < newNumDims; i++) {
      reshapeDims[i] = 1;
    }
    RETURN_IF_ERR(mergeMultidirectionalBroadcast(reshapeDims, shape0));
    RETURN_IF_ERR(mergeMultidirectionalBroadcast(reshapeDims, shape1));

    return reshapeDims;
  }

  /// Associate all outputs of \p op with nodes in \p NVs. Number of outputs of
  /// \p op should match the number of elements of \p NVs.
  /// \returns error code in case of error.
  Error assignNodeOutputs(const OpType &op, llvm::ArrayRef<NodeValue> NVs) {
    RETURN_ERR_IF_NOT((dim_t)NVs.size() == (dim_t)op.output_size(),
                      "Output size mismatch.");
    for (size_t i = 0; i < NVs.size(); i++) {
      nodeValueByName_[op.output(i)] = NVs[i];
    }
    return Error::success();
  }

  /// Load pre-trained weights from \p weightDescriptors.
  Error loadWeights(uint32_t weightsCount,
                    const onnxTensorDescriptorV1 *weightDescriptors) {
    for (uint32_t i = 0; i < weightsCount; ++i) {
      const char *name = weightDescriptors[i].name;

      LoadWeightResult loadResult;
      if (auto resOrErr = loadWeight(weightDescriptors[i])) {
        loadResult = std::move(*resOrErr);
      } else {
        RETURN_ERR(resOrErr.takeError());
      }

      // If the weight is offline create a static placeholder, otherwise create
      // a constant.
      if (weightDescriptors[i].isOffline) {
        RETURN_ERR_IF_NOT(
            !clipQuantRangeToFP16_ ||
                !loadResult.t->getType().isQuantizedType() ||
                loadResult.t->getType().isFusedQuantizedType(),
            strFormat("Do not support clipQuantRangeToFP16 with unfused "
                      "quantized input Placeholders: %s",
                      name));
        Placeholder *pl;
        ASSIGN_VALUE_OR_RETURN_ERR(
            pl, createAndRegisterPlaceholder(name, &loadResult.type,
                                             /*isStatic*/ true));
        (void)pl;
      } else {
        RETURN_IF_ERR(
            createAndRegisterConstant(name, std::move(*loadResult.t)));
      }

      if (loadResult.offsets) {
        auto offsetsName = strFormat("%s_loaded_offsets", name);
        RETURN_IF_ERR(createAndRegisterConstant(
            offsetsName, std::move(*loadResult.offsets)));
      }

      if (loadResult.scales) {
        auto scalesName = strFormat("%s_loaded_scales", name);
        RETURN_IF_ERR(createAndRegisterConstant(scalesName,
                                                std::move(*loadResult.scales)));
      }
    }

    return Error::success();
  }

  /// Sets the type of \p S to have \p dstKind, using the same dims as S.
  Error setFusedTy(Storage *S, ElemKind dstKind) {
    // Use dummy 0.0/0 as scale/offset, since the actual scales/offsets
    // are fused inline with the data.
    TypeRef fusedTy = mod_.uniqueType(dstKind, S->dims(), 0.0, 0);
    return setFusedTy(S, fusedTy);
  }

  /// Sets the type of \p S to have \p fusedTy. If \p S already has type \p
  /// fusedTy, then this is a noop. Otherwise, expected that the original S is
  /// UInt8QTy. If \p S is a Constant, then also sets the payload of the
  /// Constant to have the same type.
  /// The motivation here is that there is no fused quantized type in
  /// Caffe2/ONNX, so we will always load them in UInt8QTy. We then change it
  /// from UInt8QTy to one of the fused kinds here. This may not be necessary if
  /// another user has already changed it, or the type may already have been
  /// modified in the case of loading into an existing module.
  Error setFusedTy(Storage *S, TypeRef fusedTy) {
    assert(fusedTy->isFusedQuantizedType() && "Expected fused quantized type.");

    // If S already has the requested type then return early.
    if (S->getOutput().getType()->isEqual(*fusedTy)) {
      return Error::success();
    }

    RETURN_ERR_IF_NOT(S->getElementType() == ElemKind::UInt8QTy,
                      "Data must be UInt8QTy, but was " +
                          Type::getElementName(S->getElementType()).str());
    S->setType(Storage::OutputIdx, fusedTy);
    // If the node is a Constant set the payload type as well.
    if (auto *C = llvm::dyn_cast<Constant>(S)) {
      C->setPayloadType(fusedTy);
    }

    return Error::success();
  }

  static Expected<bool> getCountIncludePads(ArgumentDictionaryTy &dict,
                                            bool defaultValue) {
    if (dict.count("count_include_pad")) {
      int countIncludePads;
      ASSIGN_VALUE_OR_RETURN_ERR(countIncludePads,
                                 loadInt(dict.at("count_include_pad")));
      return (bool)countIncludePads;
    }
    // Return default value if can't find in the dict
    return defaultValue;
  }

  static Expected<std::vector<unsigned_t>>
  getDilations(ArgumentDictionaryTy &dict,
               const std::vector<unsigned_t> &defaultValue) {
    // For Caffe2 Model, `dilation` field can be either one integer or multiple
    // integers (one for each axis). When it's one integer the field in the dict
    // will be `dilation`. Otherwise, the field in the dict will be `dilations`.

    // For Onnx Model, it can only be `dilations` and it must be a list of
    // integers.
    if (dict.count("dilation")) {
      unsigned_t dilation;
      ASSIGN_VALUE_OR_RETURN_ERR(dilation, loadInt(dict.at("dilation")));
      return std::vector<unsigned_t>{dilation, dilation};
    }
    if (dict.count("dilations")) {
      std::vector<unsigned_t> shape;
      ASSIGN_VALUE_OR_RETURN_ERR(shape,
                                 getShape<unsigned_t>(dict["dilations"]));
      return shape;
    }

    return defaultValue;
  }
};

} // namespace glow

#endif // GLOW_IMPORTER_COMMONOPERATORLOADER_H
