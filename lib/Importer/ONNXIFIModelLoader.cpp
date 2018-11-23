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

#include "glow/Importer/ONNXIFIModelLoader.h"

#include "onnx/onnx_pb.h"

namespace glow {

/// Creates tensor \p T from the input \p in. Note, there is no data associated
/// with the Tensor. This method makes sure that the tensor is created with the
/// proper shape and element type.
static llvm::Error setTensorType(const ONNX_NAMESPACE::TypeProto &in,
                                 Tensor *T) {
  std::vector<size_t> dim;
  for (auto d : in.tensor_type().shape().dim()) {
    dim.push_back(d.dim_value());
  }

  if (in.tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto::FLOAT) {
    T->reset(ElemKind::FloatTy, dim);
    RETURN_SUCCESS();
  } else if (in.tensor_type().elem_type() ==
             ONNX_NAMESPACE::TensorProto::INT64) {
    T->reset(ElemKind::Int64ITy, dim);
    RETURN_SUCCESS();
  } else if (in.tensor_type().elem_type() ==
             ONNX_NAMESPACE::TensorProto::INT32) {
    T->reset(ElemKind::Int32ITy, dim);
    RETURN_SUCCESS();
  } else {
    RETURN_ERR("Only float and index tensors are supported");
  }
}

llvm::Error ONNXIFIModelLoader::loadInputs(ONNX_NAMESPACE::GraphProto &net) {
  for (const auto &in : net.input()) {
    // Skip static weights.
    if (tensors_.count(in.name())) {
      continue;
    }

    Tensor T;
    RETURN_IF_ERR(setTensorType(in.type(), &T));
    if (auto varOrErr = createAndRegisterPlaceholder(in.name(), &T.getType())) {
      onnxNameToInputVars_.try_emplace(in.name(), UNWRAP(std::move(varOrErr)));
    } else {
      return varOrErr.takeError();
    }
  }
  RETURN_SUCCESS();
}

/// Loads tensor \p T from the input \p in.
static llvm::Error loadWeight(const onnxTensorDescriptorV1 &in, Tensor *T) {
  // Only support CPU memory tensors.
  if (in.memoryType != ONNXIFI_MEMORY_TYPE_CPU) {
    RETURN_ERR("Only support CPU memory tensors.");
  }

  std::vector<size_t> dims;
  for (unsigned i = 0; i < in.dimensions; ++i) {
    dims.push_back(in.shape[i]);
  }

  if (in.dataType == ONNXIFI_DATATYPE_FLOAT32) {
    T->reset(ElemKind::FloatTy, dims);

    auto TH = T->getHandle<>();
    float *data = (float *)in.buffer;
    for (size_t i = 0; i < TH.size(); ++i) {
      TH.raw(i) = data[i];
    }
  } else if (in.dataType == ONNXIFI_DATATYPE_UINT64 ||
             in.dataType == ONNXIFI_DATATYPE_INT64) {
    const bool inDataSigned = in.dataType == ONNXIFI_DATATYPE_INT64;
    (void)inDataSigned;
    T->reset(ElemKind::Int64ITy, dims);

    auto TH = T->getHandle<int64_t>();
    int64_t *data = (int64_t *)in.buffer;
    for (size_t i = 0; i < TH.size(); ++i) {
      RETURN_ERR_IF_NOT(
          (inDataSigned || data[i] >= 0),
          "Disallow overflow of loaded UINT64 data into Int64ITy.");
      TH.raw(i) = data[i];
    }
  } else if (in.dataType == ONNXIFI_DATATYPE_INT32) {
    T->reset(ElemKind::Int32ITy, dims);

    auto TH = T->getHandle<int32_t>();
    int32_t *data = (int32_t *)in.buffer;
    for (size_t i = 0; i < TH.size(); ++i) {
      TH.raw(i) = data[i];
    }
  } else {
    RETURN_ERR("Only float and index tensors are supported.");
  }

  RETURN_SUCCESS();
}

llvm::Error ONNXIFIModelLoader::loadWeights(
    uint32_t weightsCount, const onnxTensorDescriptorV1 *weightDescriptors) {
  for (uint32_t i = 0; i < weightsCount; ++i) {
    Tensor *T = new Tensor();

    if (auto err = loadWeight(weightDescriptors[i], T)) {
      delete T;
      return err;
    }

    tensors_[weightDescriptors[i].name] = T;
  }

  RETURN_SUCCESS();
}

llvm::Expected<ONNXIFIModelLoader> ONNXIFIModelLoader::parse(
    const void *onnxModel, uint32_t onnxModelSize, uint32_t weightsCount,
    const onnxTensorDescriptorV1 *weightDescriptors, Function &F) {
  ONNXIFIModelLoader loader(F);

  ONNX_NAMESPACE::ModelProto modelDef;
  ASSIGN_VALUE_OR_RETURN_ERR(modelDef,
                             loader.loadProto(onnxModel, onnxModelSize));

  RETURN_IF_ERR(loader.setVersion(modelDef));

  RETURN_IF_ERR(loader.loadWeights(weightsCount, weightDescriptors));

  ONNX_NAMESPACE::GraphProto graphDef = modelDef.graph();

  RETURN_IF_ERR(loader.loadInputs(graphDef));

  RETURN_IF_ERR(loader.loadNetwork(graphDef));

  RETURN_IF_ERR(loader.setOutputNodes(graphDef));

  return loader;
}

std::vector<std::pair<Kinded::Kind, ElemKind>>
ONNXIFIModelLoader::parseOperators(const void *onnxModel,
                                   size_t onnxModelSize) {
  std::vector<std::pair<Kinded::Kind, ElemKind>> result;
  ONNX_NAMESPACE::ModelProto modelDef =
      TEMP_UNWRAP(ONNXModelLoader::loadProto(onnxModel, onnxModelSize));

  ONNX_NAMESPACE::GraphProto graph = modelDef.graph();

#define ADD_OP_MAPPING(NODE_KIND_, ELEM_KIND_)                                 \
  result.emplace_back(Kinded::Kind::NODE_KIND_, ElemKind::ELEM_KIND_);
  for (const auto &node : graph.node()) {
    const auto &operation = node.op_type();
    // Single ONNX node can be represented by several Glow nodes,
    // collect corresponding mapping in result vector.
    // Quantized and non-quantized operations are handled by
    // different ONNX operators, for now only handle fp32.
    // TODO: Add more operators.
    if (operation == "BatchNormalization") {
      ADD_OP_MAPPING(BatchNormalizationNodeKind, FloatTy);
    } else if (operation == "Conv") {
      ADD_OP_MAPPING(TransposeNodeKind, FloatTy);
      ADD_OP_MAPPING(ConvolutionNodeKind, FloatTy);
    } else if (operation == "Relu") {
      ADD_OP_MAPPING(ReluNodeKind, FloatTy);
    } else if (operation == "Softmax") {
      ADD_OP_MAPPING(SoftMaxNodeKind, FloatTy);
      ADD_OP_MAPPING(ReshapeNodeKind, FloatTy);
    } else if (operation == "Transpose") {
      ADD_OP_MAPPING(TransposeNodeKind, FloatTy);
    } else if (operation == "MaxPool") {
      ADD_OP_MAPPING(TransposeNodeKind, FloatTy);
      ADD_OP_MAPPING(MaxPoolNodeKind, FloatTy);
    } else if (operation == "AveragePool") {
      ADD_OP_MAPPING(TransposeNodeKind, FloatTy);
      ADD_OP_MAPPING(AvgPoolNodeKind, FloatTy);
    } else if (operation == "Add") {
      ADD_OP_MAPPING(AddNodeKind, FloatTy);
    } else if (operation == "Reshape") {
      ADD_OP_MAPPING(ReshapeNodeKind, FloatTy);
    } else if (operation == "Sum") {
      ADD_OP_MAPPING(AddNodeKind, FloatTy);
    } else if (operation == "Gemm") {
      ADD_OP_MAPPING(ReshapeNodeKind, FloatTy);
      ADD_OP_MAPPING(TransposeNodeKind, FloatTy);
      ADD_OP_MAPPING(MatMulNodeKind, FloatTy);
    } else if (operation == "Sigmoid") {
      ADD_OP_MAPPING(SigmoidNodeKind, FloatTy);
    } else if (operation == "Flatten") {
      ADD_OP_MAPPING(ReshapeNodeKind, FloatTy);
    } else if (operation == "Concat") {
      ADD_OP_MAPPING(ConcatNodeKind, FloatTy);
    } else if (operation == "SparseLengthsSum" ||
               operation == "SparseLengthsWeightedSum") {
      ADD_OP_MAPPING(SparseLengthsWeightedSumNodeKind, FloatTy);
    } else if (operation == "Clip") {
      ADD_OP_MAPPING(MinNodeKind, FloatTy);
      ADD_OP_MAPPING(MaxNodeKind, FloatTy);
      ADD_OP_MAPPING(SplatNodeKind, FloatTy);
    } else if (operation == "BatchMatMul") {
      ADD_OP_MAPPING(TransposeNodeKind, FloatTy);
      ADD_OP_MAPPING(SliceNodeKind, FloatTy);
      ADD_OP_MAPPING(ReshapeNodeKind, FloatTy);
      ADD_OP_MAPPING(ConcatNodeKind, FloatTy);
      ADD_OP_MAPPING(MatMulNodeKind, FloatTy);
    } else if (operation == "BatchBoxCox") {
      ADD_OP_MAPPING(ReshapeNodeKind, FloatTy);
      ADD_OP_MAPPING(TileNodeKind, FloatTy);
      ADD_OP_MAPPING(SplatNodeKind, FloatTy);
      ADD_OP_MAPPING(AddNodeKind, FloatTy);
      ADD_OP_MAPPING(MaxNodeKind, FloatTy);
      ADD_OP_MAPPING(LogNodeKind, FloatTy);
      ADD_OP_MAPPING(PowNodeKind, FloatTy);
      ADD_OP_MAPPING(SubNodeKind, FloatTy);
      ADD_OP_MAPPING(DivNodeKind, FloatTy);
      ADD_OP_MAPPING(CmpEQNodeKind, FloatTy);
    } else if (operation == "ReplaceNaN") {
      ADD_OP_MAPPING(IsNaNNodeKind, FloatTy);
      ADD_OP_MAPPING(SplatNodeKind, FloatTy);
      ADD_OP_MAPPING(SelectNodeKind, FloatTy);
    } else if (operation == "DotProduct") {
      ADD_OP_MAPPING(BatchedReduceAddNodeKind, FloatTy);
      ADD_OP_MAPPING(MulNodeKind, FloatTy);
    } else if (operation == "LengthsToRanges") {
      ADD_OP_MAPPING(LengthsToRangesNodeKind, Int32ITy);
    } else if (operation == "SparseToDense") {
      ADD_OP_MAPPING(SparseToDenseNodeKind, FloatTy);
    } else if (operation == "LengthsSum") {
      ADD_OP_MAPPING(LengthsSumNodeKind, FloatTy);
    } else if (operation == "FCTransposed") {
      ADD_OP_MAPPING(ReshapeNodeKind, FloatTy);
      ADD_OP_MAPPING(BatchedAddNodeKind, FloatTy);
      ADD_OP_MAPPING(MatMulNodeKind, FloatTy);
    } else if (operation == "ExpandDims") {
      ADD_OP_MAPPING(ReshapeNodeKind, FloatTy);
    } else if (operation == "Gather" || operation == "BatchGather") {
      ADD_OP_MAPPING(GatherNodeKind, FloatTy);
    }
  }
#undef ADD_OP_MAPPING

  return result;
}

} // namespace glow
