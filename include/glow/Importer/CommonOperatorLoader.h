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

#ifndef GLOW_IMPORTER_COMMONOPERATORLOADER_H
#define GLOW_IMPORTER_COMMONOPERATORLOADER_H

#include "foxi/onnxifi.h"

#include "glow/Importer/ProtobufLoader.h"

#include "glow/Base/Tensor.h"
#include "glow/Graph/Graph.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <functional>
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
};

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
  inline llvm::Expected<LoadWeightResult>
  loadWeight(const onnxTensorDescriptorV1 &in) {
    // Only support CPU memory tensors.
    if (in.memoryType != ONNXIFI_MEMORY_TYPE_CPU) {
      RETURN_ERR("Only support CPU memory tensors.");
    }

    // Number of qparams in the onnxTensorDescriptor.
    const size_t qparams = static_cast<size_t>(in.quantizationParams);

    // Only support quantizationAxis=1 for now.
    if (qparams > 0 && in.quantizationAxis != 1) {
      RETURN_ERR(strFormat(
          "Glow can only import quantized tensors with quantizationAxis=1 but "
          "the tensor %s has quantizationAxis=%u",
          in.name, in.quantizationAxis));
    }

    // This is a caffe2 offset shift.
    constexpr int32_t OFFSETSHIFT = 128;

    LoadWeightResult result;
    result.t = llvm::make_unique<Tensor>();

    std::vector<size_t> dims;
    for (unsigned i = 0; i < in.dimensions; ++i) {
      dims.push_back(in.shape[i]);
    }

    // Load unquantized tensor.
    if (in.quantizationParams == 0) {
      if (in.dataType == ONNXIFI_DATATYPE_FLOAT32) {
        Type ty(ElemKind::FloatTy, dims);
        *result.t = Tensor((void *)in.buffer, &ty);
      } else if (in.dataType == ONNXIFI_DATATYPE_INT32) {
        Type ty(ElemKind::Int32ITy, dims);
        *result.t = Tensor((void *)in.buffer, &ty);
      } else if (in.dataType == ONNXIFI_DATATYPE_INT64) {
        Type ty(ElemKind::Int64ITy, dims);
        *result.t = Tensor((void *)in.buffer, &ty);
      } else if (in.dataType == ONNXIFI_DATATYPE_UINT8) {
        // Must copy the weights here because we will need to modify them by
        // adjusting for OFFSETSHIFT.
        result.t->reset(ElemKind::Int8QTy, dims, 1.0, 0);
        auto TH = result.t->template getHandle<int8_t>();
        uint8_t *data = (uint8_t *)in.buffer;
        for (size_t i = 0; i < TH.size(); ++i) {
          TH.raw(i) = static_cast<int8_t>((((uint8_t)data[i]) - OFFSETSHIFT));
        }
      } else if (in.dataType == ONNXIFI_DATATYPE_UINT64) {
        Type ty(ElemKind::Int64ITy, dims);
        *result.t = Tensor((void *)in.buffer, &ty);
        for (size_t i = 0; i < result.t->size(); ++i) {
          RETURN_ERR_IF_NOT(
              ((int64_t *)in.buffer)[i] >= 0,
              "Disallow overflow of loaded UINT64 data into Int64ITy.");
        }
      } else {
        RETURN_ERR(strFormat(
            "Only float, index, and uint8 unquantized tensors are supported, "
            "got input with ONNXIFI_DATATYPE: %zu",
            static_cast<size_t>(in.dataType)));
      }

      return llvm::Expected<LoadWeightResult>(std::move(result));
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
      Type scalesTy(ElemKind::FloatTy, llvm::makeArrayRef({qparams}));
      Type offsetsTy(ElemKind::Int32ITy, llvm::makeArrayRef({qparams}));
      result.scales = llvm::make_unique<Tensor>((void *)in.scales, &scalesTy);
      result.offsets = llvm::make_unique<Tensor>((void *)in.biases, &offsetsTy);
    }

    if (in.dataType == ONNXIFI_DATATYPE_UINT8) {
      // Must copy the weights here because we will need to modify them by
      // adjusting for OFFSETSHIFT.
      result.t->reset(ElemKind::Int8QTy, dims, scale, offset - OFFSETSHIFT);

      auto TH = result.t->getHandle<int8_t>();
      uint8_t *data = (uint8_t *)in.buffer;
      for (size_t i = 0; i < TH.size(); ++i) {
        TH.raw(i) = (int8_t)(data[i] - OFFSETSHIFT);
      }
    } else if (in.dataType == ONNXIFI_DATATYPE_INT32) {
      Type ty(ElemKind::Int32QTy, dims, scale, offset);
      *result.t = Tensor((void *)in.buffer, &ty);
    } else if (in.dataType == ONNXIFI_DATATYPE_INT8) {
      Type ty(ElemKind::Int8QTy, dims, scale, offset);
      *result.t = Tensor((void *)in.buffer, &ty);
    } else {
      RETURN_ERR(strFormat("Only uint8, int32, and int8, quantized tensors are "
                           "supported, got input with ONNXIFI_DATATYPE: %zu",
                           static_cast<size_t>(in.dataType)));
    }

    return llvm::Expected<LoadWeightResult>(std::move(result));
  }

  /// Merge shape \p shape into \p mergeShape, following multidirectional
  /// broadcasting rules.
  llvm::Error mergeMultidirectionalBroadcast(std::vector<size_t> &mergeShape,
                                             llvm::ArrayRef<size_t> shape) {
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
    return llvm::Error::success();
  }

protected:
  CommonOperatorLoader(llvm::ArrayRef<const char *> names,
                       llvm::ArrayRef<TypeRef> types, Function &F)
      : ProtobufLoader(names, types, F) {}

  using ArgumentDictionaryTy =
      std::unordered_map<std::string, const AttrType *>;

  /// \returns True if the operator has broadcasting activated.
  virtual llvm::Expected<bool>
  getBroadcast(const ArgumentDictionaryTy &dict) = 0;

  /// \returns True if the operator with the name \p typeName has support
  /// for multidirectional broadcasting.
  virtual bool hasMultidirectionalBroadcast(const llvm::StringRef typeName) = 0;

  /// Associate the name of operation outputs to a NodeValues corresponding to
  /// node \p node. If \p numOutputs is lower than 0, then all outputs are
  /// associated. Otherwise, the first \p numOutputs outputs are associated.
  llvm::Error addNodeAsOutput(const OpType &op, Node *node,
                              int numOutputs = -1) {
    RETURN_ERR_IF_NOT(numOutputs <= op.output_size(),
                      "Can't register more than outputs in the operation.");
    numOutputs = (numOutputs < 0) ? op.output_size() : numOutputs;
    for (int i = 0; i < numOutputs; i++) {
      nodeValueByName_[op.output(i)] = NodeValue(node, i);
    }
    return llvm::Error::success();
  }

  /// Loads RELU operator, given its protobuf representation and parsed args.
  llvm::Error loadRelu(const OpType &op, ArgumentDictionaryTy &dict) {
    const std::string &opName = loadOperatorName(op);
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));
    auto *R = G_.createRELU(opName, in);
    RETURN_IF_ERR(addNodeAsOutput(op, R));
    return llvm::Error::success();
  }

  /// Loads PRELU operator, given its protobuf representation and parsed args.
  /// Follows undirectional broadcasting described here:
  /// https://github.com/onnx/onnx/blob/fb1a80692c1ab0bd27b1072f2e7bffacba336777/docs/Broadcasting.md
  llvm::Error loadPRelu(const OpType &op, ArgumentDictionaryTy &dict) {
    const std::string &opName = loadOperatorName(op);

    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));

    NodeValue slope;
    ASSIGN_VALUE_OR_RETURN_ERR(slope, getNodeValueByName(op.input(1)));

    // Do broadcasting.
    auto targetDim = in.dims();
    // Sets the axis of each inputs so that the trailing-most dimensions of
    // input tensors and the target shape are aligned.
    int axis = targetDim.size() - slope.dims().size();
    auto *finalSlope = G_.createBroadcast(opName, slope, targetDim, axis);
    auto *R = G_.createPRELU(opName, in, finalSlope);
    RETURN_IF_ERR(addNodeAsOutput(op, R));
    return llvm::Error::success();
  }

  llvm::Error loadSigmoid(const OpType &op, ArgumentDictionaryTy &dict) {
    const std::string &opName = loadOperatorName(op);
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));
    auto *S = G_.createSigmoid(opName, in);
    RETURN_IF_ERR(addNodeAsOutput(op, S));
    return llvm::Error::success();
  }

  llvm::Error loadTanh(const OpType &op, ArgumentDictionaryTy &dict) {
    const std::string &opName = loadOperatorName(op);
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));
    auto *T = G_.createTanh(opName, in);
    RETURN_IF_ERR(addNodeAsOutput(op, T));
    return llvm::Error::success();
  }

  llvm::Error loadExp(const OpType &op, ArgumentDictionaryTy &dict) {
    const std::string &opName = loadOperatorName(op);
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));
    auto *E = G_.createExp(opName, in);
    RETURN_IF_ERR(addNodeAsOutput(op, E));
    return llvm::Error::success();
  }

  llvm::Error loadShape(const OpType &op, ArgumentDictionaryTy &dict) {
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));

    // This is statically known data, and so we create a Tensor for it.
    Tensor T(ElemKind::Int64ITy, {in.dims().size()});
    T.getHandle<int64_t>() =
        std::vector<int64_t>(in.dims().begin(), in.dims().end());

    RETURN_IF_ERR(createAndRegisterConstant(op.output(0), std::move(T)));

    return llvm::Error::success();
  }

  /// Loads Sqrt operator, given its protobuf representation and parsed args.
  llvm::Error loadSqrt(const OpType &op, ArgumentDictionaryTy &dict) {
    const std::string &opName = loadOperatorName(op);
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));
    auto *R = G_.createPow(opName, in, 0.5f);
    RETURN_IF_ERR(addNodeAsOutput(op, R));
    return llvm::Error::success();
  }

  /// Loads Reciprocal operator, given its protobuf representation and parsed
  /// args.
  llvm::Error loadReciprocal(const OpType &op, ArgumentDictionaryTy &dict) {
    const std::string &opName = loadOperatorName(op);
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));
    auto *R = G_.createPow(opName, in, -1.0f);
    RETURN_IF_ERR(addNodeAsOutput(op, R));
    return llvm::Error::success();
  }

  llvm::Error loadSum(const OpType &op, ArgumentDictionaryTy &dict) {
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
      auto *node = G_.createAdd(opName, in0, in1);
      RETURN_IF_ERR(addNodeAsOutput(op, node));
    } else {
      const std::string &opName = loadOperatorName(op);
      const unsigned numInputs = op.input_size();
      llvm::SmallVector<NodeValue, 4> inputs;
      inputs.reserve(numInputs);
      for (unsigned i = 0; i < numInputs; i++) {
        NodeValue in;
        ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(i)));
        inputs.push_back(G_.createExpandDims(opName, in, {0}));
      }
      ConcatNode *concat = G_.createConcat(opName, inputs, /* axis */ 0);
      Node *node = G_.createBatchedReduceAdd(opName, concat, /* axis */ 0);
      RETURN_IF_ERR(addNodeAsOutput(op, node));
    }
    return llvm::Error::success();
  }

  llvm::Error loadSoftmax(const OpType &op, ArgumentDictionaryTy &dict) {
    const std::string &opName = loadOperatorName(op);

    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));

    // We do not do training right now on loaded protos. C2 and ONNX do not even
    // have an option for a selected input anyway. So I am creating this as a
    // placeholder which goes unused during inference.
    auto selected = G_.getParent()->createConstant(
        ElemKind::Int64ITy, {in.dims()[0], 1}, "selected");

    // ONNX allows shapes like <N x 10 x 1 x 1 >. Flatten the inputs to the
    // softmax function. This is similar to a bitcast operation.
    int axis = 1;
    if (dict.count("axis")) {
      ASSIGN_VALUE_OR_RETURN_ERR(axis, loadInt(dict["axis"]));
    }

    auto *FN = G_.createFlatten("reshapeInput", in, axis);

    auto *SM = G_.createSoftMax(opName, FN, selected);

    // The output should have the same shape as the original input.
    auto origInDims = in.getType()->dims();
    auto *RN = G_.createReshape("reshapeOutput", SM, origInDims);
    RETURN_IF_ERR(addNodeAsOutput(op, RN));
    return llvm::Error::success();
  }

  llvm::Error loadLRN(const OpType &op, ArgumentDictionaryTy &dict) {
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

    auto *tr = G_.createTranspose(opName, in, NCHW2NHWC);

    auto *node = G_.createLocalResponseNormalization(opName, tr, size / 2,
                                                     alpha, beta, k);

    auto *N = G_.createTranspose(opName, node, NHWC2NCHW);

    // LRN in Caffe2 has a scale_ output, but I believe it's unused for
    // inference. So explicitly only set output 0.
    nodeValueByName_[op.output(0)] = N->getResult();
    return llvm::Error::success();
  }

  llvm::Error loadMinMax(llvm::StringRef typeName, const OpType &op,
                         ArgumentDictionaryTy &dict) {
    const std::string &opName = loadOperatorName(op);
    NodeValue in0;
    ASSIGN_VALUE_OR_RETURN_ERR(in0, getNodeValueByName(op.input(0)));
    NodeValue in1;
    ASSIGN_VALUE_OR_RETURN_ERR(in1, getNodeValueByName(op.input(1)));

    Node *node = nullptr;
    if (typeName == "Min") {
      node = G_.createMin(opName, in0, in1);
    } else if (typeName == "Max") {
      node = G_.createMax(opName, in0, in1);
    } else {
      RETURN_ERR("Invalid min or max operator");
    }

    RETURN_IF_ERR(addNodeAsOutput(op, node));
    return llvm::Error::success();
  }

  static llvm::Expected<NodeValue>
  handleBatchMatMulTranspose(Function &F, ArgumentDictionaryTy &dict,
                             llvm::StringRef key, NodeValue input) {
    if (!dict.count(key)) {
      return input;
    }

    int isTransposed;
    ASSIGN_VALUE_OR_RETURN_ERR(isTransposed, loadInt(dict[key]));
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

      return F.createTranspose(input.getNode()->getName().str() + ".transpose",
                               input, shuffle);
    }

    return input;
  }

  llvm::Error loadBatchMatMul(const OpType &op, ArgumentDictionaryTy &dict,
                              bool isBatched) {
    const std::string &opName = loadOperatorName(op);
    NodeValue LHS;
    ASSIGN_VALUE_OR_RETURN_ERR(LHS, getNodeValueByName(op.input(0)));
    NodeValue RHS;
    ASSIGN_VALUE_OR_RETURN_ERR(RHS, getNodeValueByName(op.input(1)));

    ASSIGN_VALUE_OR_RETURN_ERR(
        LHS, handleBatchMatMulTranspose(G_, dict, "trans_a", LHS));
    ASSIGN_VALUE_OR_RETURN_ERR(
        RHS, handleBatchMatMulTranspose(G_, dict, "trans_b", RHS));

    Node *node = nullptr;

    // BatchMatMul sometimes is actually just a matmul, depending on dimensions
    // of inputs. Thus, only do batch matmul if LHS is 3-dimensional.
    if (isBatched && LHS.dims().size() == 3) {
      node = G_.createBatchMatMul(opName, LHS, RHS);
    } else {
      node = G_.createMatMul(opName, LHS, RHS);
    }

    RETURN_IF_ERR(addNodeAsOutput(op, node));
    return llvm::Error::success();
  }

  llvm::Error loadArithmetic(llvm::StringRef typeName, const OpType &op,
                             ArgumentDictionaryTy &dict) {
    const std::string &opName = loadOperatorName(op);
    NodeValue in0;
    ASSIGN_VALUE_OR_RETURN_ERR(in0, getNodeValueByName(op.input(0)));
    NodeValue in1;
    ASSIGN_VALUE_OR_RETURN_ERR(in1, getNodeValueByName(op.input(1)));

    bool broadcast;
    ASSIGN_VALUE_OR_RETURN_ERR(broadcast, getBroadcast(dict));

    NodeValue finalIn0 = in0;
    NodeValue finalIn1 = in1;

    if (broadcast) {
      // Broadcasting can be:
      // - multidirectional (ONNX opset 7+), or
      // - unidirectional (ONNX opset 1->6,  Caffe2).
      if (hasMultidirectionalBroadcast(typeName)) {
        // Compute the target shape that is a combination of the operand shapes.
        std::vector<size_t> targetDim;
        ASSIGN_VALUE_OR_RETURN_ERR(targetDim, computeMultidirectionalBroadcast(
                                                  in0.dims(), in1.dims()));
        // Sets the axis of each inputs so that the trailing-most dimensions of
        // input tensors and the target shape are aligned.
        int axis0 = targetDim.size() - in0.dims().size();
        int axis1 = targetDim.size() - in1.dims().size();
        finalIn0 = G_.createBroadcast(opName, in0, targetDim, axis0);
        finalIn1 = G_.createBroadcast(opName, in1, targetDim, axis1);
      }
      // Unidirectional broadcasting consists of broadcasting the right operand
      // (in1) so that it matches the shape of the left operand (in0).
      else {
        // With unidirectional broadcasting, the 'axis' attribute specifies
        // from how much the right operand shape must be 'shifted' right.
        // - In Caffe2, the 'axis' attribute is optional. If not specified, axis
        // must be automatically computed so that the trailing-most dimensions
        // of in1 is aligned to the trailing-most dimension of in0.
        // - In ONNX, the 'axis' attribute is mandatory. axis == -1 is
        // equivalent to no axis specified in Caffe2.
        int axis = -1;
        if (dict.count("axis")) {
          ASSIGN_VALUE_OR_RETURN_ERR(axis, loadInt(dict["axis"]));
        }
        if (axis == -1) {
          // Align trailing most dimensions.
          axis = in0.dims().size() - in1.dims().size();
        }
        finalIn1 = G_.createBroadcast(opName, in1, in0.dims(), axis);
      }
    }

    Node *node = nullptr;
    if (typeName == "Mul") {
      node = G_.createMul(opName, finalIn0, finalIn1);
    } else if (typeName == "Add") {
      node = G_.createAdd(opName, finalIn0, finalIn1);
    } else if (typeName == "Sub") {
      node = G_.createSub(opName, finalIn0, finalIn1);
    } else if (typeName == "Div") {
      node = G_.createDiv(opName, finalIn0, finalIn1);
    } else {
      RETURN_ERR("Unsupported arithmetic typeName");
    }

    RETURN_IF_ERR(addNodeAsOutput(op, node));
    return llvm::Error::success();
  }

  llvm::Error loadSplit(const OpType &op, ArgumentDictionaryTy &dict) {
    const std::string &opName = loadOperatorName(op);
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));
    size_t axis = 0;
    if (dict.count("axis")) {
      ASSIGN_VALUE_OR_RETURN_ERR(axis, loadInt(dict["axis"]));
    }

    std::vector<size_t> split;
    if (dict.count("split"))
      split = getShape(dict["split"]);

    std::vector<SliceNode *> outputs;
    G_.createSplit(opName, in, op.output_size(), axis, split, outputs);

    for (int i = 0, e = op.output_size(); i < e; i++) {
      // Each output from Split is a SliceNode which only has a single output,
      // so only use 0 here as the node value result.
      nodeValueByName_[op.output(i)] = outputs[i]->getResult();
    }
    return llvm::Error::success();
  }

  llvm::Error loadReshape(const OpType &op, ArgumentDictionaryTy &dict) {
    const std::string &opName = loadOperatorName(op);
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));

    // Get the requested shape from the model.
    // First look at input tensors, then at the "shape" attribute.
    std::vector<int64_t> requestedDims;
    if (op.input_size() > 1) {
      if (!getConstantByNameOrNull(op.input(1))) {
        RETURN_ERR("Non-constant shape tensors are unsupported by Glow.");
      }
      const Constant *constShapeConst;
      ASSIGN_VALUE_OR_RETURN_ERR(constShapeConst,
                                 getConstantByName(op.input(1)));
      auto TH = constShapeConst->getPayload().getHandle<int64_t>();
      for (auto dim : TH) {
        requestedDims.push_back(dim);
      }
    } else if (dict.count("shape")) {
      RETURN_ERR_IF_NOT(op.input_size() == 1,
                        "Cannot specify new shape by both argument and input.");
      std::vector<int64_t> protoDims = getShape<int64_t>(dict["shape"]);
      for (auto dim : protoDims) {
        requestedDims.push_back(dim);
      }
    } else {
      RETURN_ERR("Missing output shape information for the Reshape operator.");
    }

    // Compute the actual new shape
    ssize_t negOneIndex = -1;
    llvm::ArrayRef<size_t> inputDims = in.dims();
    std::vector<size_t> outputDims;
    int64_t dimProduct = 1;
    for (size_t i = 0, e = requestedDims.size(); i != e; i++) {
      int64_t newDim = requestedDims[i];
      if (newDim == 0) {
        // 0 means that corresponding input dimension should be propagated to
        // the output.
        newDim = inputDims[i];
      }
      if (newDim != -1) {
        dimProduct *= newDim;
        outputDims.push_back(newDim);
      } else {
        // -1 means that the corresponding dimension should be inferred
        // from all other dimensions, so that tensor size remains the same.
        RETURN_ERR_IF_NOT(negOneIndex < 0,
                          "At most one dimension of the new shape can be -1.");
        negOneIndex = (ssize_t)i;
        // The -1 case value is handled later.
        outputDims.push_back(0);
      }
    }
    if (negOneIndex >= 0) {
      outputDims[negOneIndex] = in.getType()->size() / dimProduct;
    }

    auto *node = G_.createReshape(opName, in, outputDims);

    // Caffe2 sometimes outputs old_shape which goes unused. We do not currently
    // support it, so explicitly only set the first output.
    nodeValueByName_[op.output(0)] = node->getResult();
    return llvm::Error::success();
  }

  llvm::Error loadTranspose(const OpType &op, ArgumentDictionaryTy &dict,
                            llvm::StringRef permArgName) {
    const std::string &opName = loadOperatorName(op);
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));

    // There is a difference between ONNX and Caffe2 specs for Transpose:
    // one contains permutation under name "perm", the other contains it under
    // argument name "axes". That's why the name is passed as a parameter.
    std::vector<unsigned_t> perm = getShape<unsigned_t>(dict[permArgName]);
    if (perm.empty()) {
      // Empty permutation argument means reversing axes order.
      size_t N = in.dims().size();
      for (int64_t i = N - 1; i >= 0; i--)
        perm.push_back(i);
    }

    auto *T = G_.createTranspose(opName, in, perm);

    RETURN_IF_ERR(addNodeAsOutput(op, T));
    return llvm::Error::success();
  }

  llvm::Error loadFlatten(const OpType &op, ArgumentDictionaryTy &dict) {
    const std::string &opName = loadOperatorName(op);
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));
    int axis = 1;
    if (dict.count("axis")) {
      ASSIGN_VALUE_OR_RETURN_ERR(axis, loadInt(dict["axis"]));
    }
    auto *node = G_.createFlatten(opName, in, axis);
    RETURN_IF_ERR(addNodeAsOutput(op, node));
    return llvm::Error::success();
  }

  llvm::Error loadIdentity(const OpType &op, ArgumentDictionaryTy &dict) {
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));
    nodeValueByName_[op.output(0)] = in;
    return llvm::Error::success();
  }

  llvm::Error loadTopK(const OpType &op, ArgumentDictionaryTy &dict) {
    const std::string &opName = loadOperatorName(op);
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));
    unsigned_t k;
    ASSIGN_VALUE_OR_RETURN_ERR(k, loadInt(dict["k"]));

    int axis = -1;
    if (dict.count("axis")) {
      ASSIGN_VALUE_OR_RETURN_ERR(axis, loadInt(dict["axis"]));
    }
    int lastDim = in.dims().size() - 1;
    if (axis == -1) {
      axis = lastDim;
    }

    RETURN_ERR_IF_NOT(axis == lastDim,
                      "Currently only support axis being last dimension.");

    auto *R = G_.createTopK(opName, in, k);
    RETURN_IF_ERR(addNodeAsOutput(op, R));
    return llvm::Error::success();
  }

  llvm::Error loadReduceMeanOrSum(llvm::StringRef typeName, const OpType &op,
                                  ArgumentDictionaryTy &dict) {
    const std::string &opName = loadOperatorName(op);
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));

    auto shapeAxes = getShape<unsigned_t>(dict["axes"]);
    std::sort(shapeAxes.begin(), shapeAxes.end());

    llvm::ArrayRef<unsigned_t> axes(shapeAxes);

    // Check if axes elements are unique.
    if (axes.size() > 1) {
      auto it = std::unique(shapeAxes.begin(), shapeAxes.end());
      if (it != shapeAxes.end()) {
        RETURN_ERR("Axes values are not unique.",
                   GlowErr::ErrorCode::MODEL_LOADER_UNSUPPORTED_SHAPE);
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
      node = G_.createBatchedReduceMean(opName, in, axes);
    } else {
      node = G_.createBatchedReduceAdd(opName, in, axes);
    }

    // Our batched reduce add/mean does not keep the dim; reshape if necessary.
    if (keepDims) {

      std::vector<size_t> shape = node.dims();

      // Add removed axes. Requires increasing order sort - done above.
      for (const auto &axis : shapeAxes) {
        shape.insert(shape.begin() + axis, 1);
      }
      node = G_.createReshape(opName, node, shape);
    }

    RETURN_IF_ERR(addNodeAsOutput(op, node));
    return llvm::Error::success();
  }

  llvm::Error loadBatchOneHot(const OpType &op) {
    const std::string &opName = loadOperatorName(op);
    NodeValue data;
    ASSIGN_VALUE_OR_RETURN_ERR(data, getNodeValueByName(op.input(0)));
    NodeValue lengths;
    ASSIGN_VALUE_OR_RETURN_ERR(lengths, getNodeValueByName(op.input(1)));
    NodeValue values;
    ASSIGN_VALUE_OR_RETURN_ERR(values, getNodeValueByName(op.input(2)));

    auto *node = G_.createBatchOneHot(opName, data, lengths, values);
    RETURN_IF_ERR(addNodeAsOutput(op, node));
    return llvm::Error::success();
  }

  llvm::Error loadSparseLengthsSum(const OpType &op) {
    NodeValue in0;
    ASSIGN_VALUE_OR_RETURN_ERR(in0, getNodeValueByName(op.input(0)));
    NodeValue in1;
    ASSIGN_VALUE_OR_RETURN_ERR(in1, getNodeValueByName(op.input(1)));
    NodeValue in2;
    ASSIGN_VALUE_OR_RETURN_ERR(in2, getNodeValueByName(op.input(2)));
    auto *node = G_.createSparseLengthsSum(loadOperatorName(op), in0, in1, in2);
    RETURN_IF_ERR(addNodeAsOutput(op, node));
    return llvm::Error::success();
  }

  llvm::Error loadSparseLengthsWeightedSum(const OpType &op) {
    NodeValue in0;
    ASSIGN_VALUE_OR_RETURN_ERR(in0, getNodeValueByName(op.input(0)));
    NodeValue in1;
    ASSIGN_VALUE_OR_RETURN_ERR(in1, getNodeValueByName(op.input(1)));
    NodeValue in2;
    ASSIGN_VALUE_OR_RETURN_ERR(in2, getNodeValueByName(op.input(2)));
    NodeValue in3;
    ASSIGN_VALUE_OR_RETURN_ERR(in3, getNodeValueByName(op.input(3)));
    auto *node = G_.createSparseLengthsWeightedSum(loadOperatorName(op), in0,
                                                   in1, in2, in3);
    RETURN_IF_ERR(addNodeAsOutput(op, node));
    return llvm::Error::success();
  }

  llvm::Error loadLengthsToRanges(const OpType &op) {
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));
    auto *node = G_.createLengthsToRanges(loadOperatorName(op), in);
    RETURN_IF_ERR(addNodeAsOutput(op, node));
    return llvm::Error::success();
  }

  llvm::Error loadBatchBoxCox(const OpType &op) {
    NodeValue data;
    ASSIGN_VALUE_OR_RETURN_ERR(data, getNodeValueByName(op.input(0)));
    NodeValue lambda1;
    ASSIGN_VALUE_OR_RETURN_ERR(lambda1, getNodeValueByName(op.input(1)));
    NodeValue lambda2;
    ASSIGN_VALUE_OR_RETURN_ERR(lambda2, getNodeValueByName(op.input(2)));
    auto *node =
        G_.createBatchBoxCox(loadOperatorName(op), data, lambda1, lambda2);
    RETURN_IF_ERR(addNodeAsOutput(op, node));
    return llvm::Error::success();
  }

  llvm::Error loadDotProduct(const OpType &op) {
    NodeValue X;
    ASSIGN_VALUE_OR_RETURN_ERR(X, getNodeValueByName(op.input(0)));
    NodeValue Y;
    ASSIGN_VALUE_OR_RETURN_ERR(Y, getNodeValueByName(op.input(1)));
    auto *node = G_.createDotProduct(loadOperatorName(op), X, Y);
    RETURN_IF_ERR(addNodeAsOutput(op, node));
    return llvm::Error::success();
  }

  llvm::Error loadReplaceNaN(const OpType &op,
                             const ArgumentDictionaryTy &dict) {
    // Load the input and NaN replacement value:
    NodeValue input;
    ASSIGN_VALUE_OR_RETURN_ERR(input, getNodeValueByName(op.input(0)));
    auto valueIt = dict.find("value");
    float value = 0.0f;
    if (valueIt != dict.end()) {
      ASSIGN_VALUE_OR_RETURN_ERR(value, loadFloat(valueIt->second));
    }
    auto *node = G_.createReplaceNaN(loadOperatorName(op), input, value);
    RETURN_IF_ERR(addNodeAsOutput(op, node));
    return llvm::Error::success();
  }

  llvm::Error loadLengthsSum(const OpType &op) {
    const std::string &opName = loadOperatorName(op);
    NodeValue data;
    ASSIGN_VALUE_OR_RETURN_ERR(data, getNodeValueByName(op.input(0)));
    NodeValue lengths;
    ASSIGN_VALUE_OR_RETURN_ERR(lengths, getNodeValueByName(op.input(1)));

    RETURN_ERR_IF_NOT(lengths.dims().size() == 1,
                      "Lengths must be a 1D vector.");

    auto *node = G_.createLengthsSum(opName, data, lengths);
    RETURN_IF_ERR(addNodeAsOutput(op, node));
    return llvm::Error::success();
  }

  llvm::Error loadExpandDims(const OpType &op,
                             const ArgumentDictionaryTy &dict) {
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));
    auto dims = dict.find("dims");
    if (dims == dict.end()) {
      RETURN_ERR("Missing dims argument for ExpandDims operator.");
    }
    Node *node =
        G_.createExpandDims(loadOperatorName(op), in, getShape(dims->second));
    RETURN_IF_ERR(addNodeAsOutput(op, node));

    return llvm::Error::success();
  }

  llvm::Error loadClip(const OpType &op, const ArgumentDictionaryTy &dict) {
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in, getNodeValueByName(op.input(0)));
    float cmin = std::numeric_limits<float>::lowest();
    if (dict.count("min")) {
      ASSIGN_VALUE_OR_RETURN_ERR(cmin, loadFloat(dict.find("min")->second));
    }

    float cmax = std::numeric_limits<float>::max();
    if (dict.count("max")) {
      ASSIGN_VALUE_OR_RETURN_ERR(cmax, loadFloat(dict.find("max")->second));
    }

    auto *node = G_.createClip(loadOperatorName(op), in, cmin, cmax);
    RETURN_IF_ERR(addNodeAsOutput(op, node));
    return llvm::Error::success();
  }

  llvm::Error loadSparseToDense(const OpType &op,
                                const ArgumentDictionaryTy &dict) {
    if (op.input_size() != 3) {
      RETURN_ERR("SparseToDense operator must have three inputs.");
    }

    NodeValue indices;
    ASSIGN_VALUE_OR_RETURN_ERR(indices, getNodeValueByName(op.input(0)));
    NodeValue values;
    ASSIGN_VALUE_OR_RETURN_ERR(values, getNodeValueByName(op.input(1)));
    NodeValue dataToInferDim;
    ASSIGN_VALUE_OR_RETURN_ERR(dataToInferDim, getNodeValueByName(op.input(2)));

    auto *node = G_.createSparseToDense(loadOperatorName(op), indices, values,
                                        dataToInferDim);
    RETURN_IF_ERR(addNodeAsOutput(op, node));
    return llvm::Error::success();
  }

  llvm::Error loadSparseToDenseMask(const OpType &op,
                                    const ArgumentDictionaryTy &dict) {
    size_t inputSize = op.input_size();
    if (inputSize != 3 && inputSize != 4) {
      RETURN_ERR("SparseToDenseMask operator must have 3 or 4 inputs.");
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
      auto *lengthsConstant = G_.getParent()->createConstant(
          ElemKind::Int32ITy, {}, "lengthsConstant");
      lengthsConstant->getPayloadMutable().template getHandle<int32_t>().raw(
          0) = indices.dims()[0];
      lengths = lengthsConstant->getOutput();
    }

    auto maskIt = dict.find("mask");
    RETURN_ERR_IF_NOT(maskIt != dict.end(),
                      "Require mask when loading SparseToDenseMask.");
    auto mask = getShape<int64_t>(maskIt->second);

    auto *node = G_.createSparseToDenseMask(
        loadOperatorName(op), indices, values, defaultValue, lengths, mask);
    RETURN_IF_ERR(addNodeAsOutput(op, node));
    return llvm::Error::success();
  }

  llvm::Error loadGatherOps(const std::string &typeName, const OpType &op,
                            const ArgumentDictionaryTy &dict) {

    NodeValue data;
    ASSIGN_VALUE_OR_RETURN_ERR(data, getNodeValueByName(op.input(0)));
    NodeValue indices;
    ASSIGN_VALUE_OR_RETURN_ERR(indices, getNodeValueByName(op.input(1)));
    size_t batchDims = typeName == "Gather" ? 0 : 1;

    if (dict.count("axis")) {
      int axis;
      ASSIGN_VALUE_OR_RETURN_ERR(axis, loadInt(dict.find("axis")->second));
      if (axis != 0 && axis != 1) {
        RETURN_ERR("Axis must be 0 or 1.");
      }

      batchDims = axis;
    }

    Node *GN = G_.createGather(loadOperatorName(op), data, indices, batchDims);
    RETURN_IF_ERR(addNodeAsOutput(op, GN));
    return llvm::Error::success();
  }

  llvm::Error loadGatherRanges(const std::string &typeName, const OpType &op,
                               const ArgumentDictionaryTy &dict) {
    NodeValue data;
    ASSIGN_VALUE_OR_RETURN_ERR(data, getNodeValueByName(op.input(0)));
    RETURN_ERR_IF_NOT(data.dims().size() == 1, "Data must be a 1D vector.");

    NodeValue ranges;
    ASSIGN_VALUE_OR_RETURN_ERR(ranges, getNodeValueByName(op.input(1)));
    RETURN_ERR_IF_NOT(ranges.dims().size() == 3, "Ranges must be a 3D vector.");
    RETURN_ERR_IF_NOT(ranges.dims()[2] == 2,
                      "Last dimension of ranges must be 2.");

    auto maxOutputSizeIt = dict.find("maxOutputSize");
    RETURN_ERR_IF_NOT(maxOutputSizeIt != dict.end(),
                      "Require maxOutputSize when loading LengthsRangeFill.");
    unsigned_t maxOutputSize;
    ASSIGN_VALUE_OR_RETURN_ERR(maxOutputSize, loadInt(maxOutputSizeIt->second));

    Node *GR = G_.createGatherRanges(loadOperatorName(op), data, ranges,
                                     maxOutputSize);
    RETURN_IF_ERR(addNodeAsOutput(op, GR));
    return llvm::Error::success();
  }

  using ProtobufLoader::ProtobufLoader;

  /// If operator type is supported, returns Expected<true> and creates new
  /// operator. Returns Operator<false> if operator type is not supported.
  /// Returns Error if an error occurred
  llvm::Expected<bool> tryLoadCommonOperator(llvm::StringRef typeName,
                                             const OpType &op,
                                             ArgumentDictionaryTy &dict) {
    if (typeName == "Relu") {
      RETURN_IF_ERR(loadRelu(op, dict));
      return true;
    }
    if (typeName == "PRelu") {
      RETURN_IF_ERR(loadPRelu(op, dict));
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
    if (typeName == "Shape") {
      RETURN_IF_ERR(loadShape(op, dict));
      return true;
    }
    if (typeName == "Sqrt") {
      RETURN_IF_ERR(loadSqrt(op, dict));
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
    if (typeName == "Softmax") {
      RETURN_IF_ERR(loadSoftmax(op, dict));
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
        typeName == "Div") {
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
    if (typeName == "TopK") {
      RETURN_IF_ERR(loadTopK(op, dict));
      return true;
    }
    if (typeName == "ReduceMean" || typeName == "ReduceSum") {
      RETURN_IF_ERR(loadReduceMeanOrSum(typeName, op, dict));
      return true;
    }
    if (typeName == "BatchMatMul") {
      RETURN_IF_ERR(loadBatchMatMul(op, dict, true));
      return true;
    }
    if (typeName == "BatchOneHot") {
      RETURN_IF_ERR(loadBatchOneHot(op));
      return true;
    }
    if (typeName == "SparseLengthsSum") {
      RETURN_IF_ERR(loadSparseLengthsSum(op));
      return true;
    }
    if (typeName == "SparseLengthsWeightedSum") {
      RETURN_IF_ERR(loadSparseLengthsWeightedSum(op));
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
    if (typeName == "Clip") {
      RETURN_IF_ERR(loadClip(op, dict));
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
      RETURN_IF_ERR(loadGatherOps(typeName, op, dict));
      return true;
    }
    if (typeName == "GatherRanges") {
      RETURN_IF_ERR(loadGatherRanges(typeName, op, dict));
      return true;
    }

    return false;
  }

  /// Utility function which computes the resulting shape in case of
  /// multidirectional broadcasting.
  llvm::Expected<std::vector<size_t>>
  computeMultidirectionalBroadcast(llvm::ArrayRef<size_t> shape0,
                                   llvm::ArrayRef<size_t> shape1) {
    size_t numDims0 = shape0.size();
    size_t numDims1 = shape1.size();
    size_t newNumDims = numDims0 > numDims1 ? numDims0 : numDims1;
    std::vector<size_t> reshapeDims(newNumDims);

    for (size_t i = 0; i < newNumDims; i++) {
      reshapeDims[i] = 1;
    }
    RETURN_IF_ERR(mergeMultidirectionalBroadcast(reshapeDims, shape0));
    RETURN_IF_ERR(mergeMultidirectionalBroadcast(reshapeDims, shape1));

    return reshapeDims;
  }

  /// Load pre-trained weights from \p weightDescriptors.
  llvm::Error loadWeights(uint32_t weightsCount,
                          const onnxTensorDescriptorV1 *weightDescriptors) {
    for (uint32_t i = 0; i < weightsCount; ++i) {
      const char *name = weightDescriptors[i].name;

      LoadWeightResult loadResult;
      if (auto resOrErr = loadWeight(weightDescriptors[i])) {
        loadResult = std::move(*resOrErr);
      } else {
        return resOrErr.takeError();
      }

      RETURN_IF_ERR(createAndRegisterConstant(name, std::move(*loadResult.t)));

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

    return llvm::Error::success();
  }
};

} // namespace glow

#endif // GLOW_IMPORTER_COMMONOPERATORLOADER_H
