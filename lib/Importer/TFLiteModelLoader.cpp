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

#include "glow/Importer/TFLiteModelLoader.h"
#include "glow/Base/Tensor.h"
#include "glow/Graph/Graph.h"
#include "glow/Graph/Nodes.h"
#include "glow/Importer/CommonOperatorLoader.h"
#include "glow/Support/Support.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace glow;
using llvm::cast;

namespace {

llvm::cl::OptionCategory
    tfliteModelLoaderCat("TensorFlowLite Model Loader Options");

llvm::cl::opt<bool> tfliteUint8ToInt8Opt(
    "tflite-uint8-to-int8",
    llvm::cl::desc("TensorFlowLite loader option to convert the model from "
                   "UINT8 data type to INT8 data type."),
    llvm::cl::init(true), llvm::cl::Optional,
    llvm::cl::cat(tfliteModelLoaderCat));

llvm::cl::opt<bool> tfliteFloatInputsOpt(
    "tflite-float-inputs",
    llvm::cl::desc("TensorFlowLite loader option to replace the quantized "
                   "inputs with floating point inputs."),
    llvm::cl::init(false), llvm::cl::Optional,
    llvm::cl::cat(tfliteModelLoaderCat));

llvm::cl::opt<bool> tfliteFloatOutputsOpt(
    "tflite-float-outputs",
    llvm::cl::desc("TensorFlowLite loader option to replace the quantized "
                   "outputs with floating point outputs."),
    llvm::cl::init(false), llvm::cl::Optional,
    llvm::cl::cat(tfliteModelLoaderCat));

llvm::cl::opt<bool> tfliteFloatSoftmaxOpt(
    "tflite-float-softmax",
    llvm::cl::desc("TensorFlowLite loader option to replace a quantized Softmax"
                   "with a floating point Softmax."),
    llvm::cl::init(true), llvm::cl::Optional,
    llvm::cl::cat(tfliteModelLoaderCat));

llvm::cl::opt<float> tfliteBiasScaleCheckMaxErrorOpt(
    "tflite-bias-scale-check-max-error",
    llvm::cl::desc(
        "TensorFlowLite mandates that for quantized operators like Conv2D the "
        "bias quantization parameter biasScale = inputScale * weightsScale but "
        "some pre-quantized models do not EXACTLY satisfy this relation but "
        "with very small relative errors (around 1e-8). Hence we allow a "
        "tolerance of 1e-6 which, if satisfied, then we adjust the bias to "
        "conform to the restriction."),
    llvm::cl::init(1e-6), llvm::cl::Optional,
    llvm::cl::cat(tfliteModelLoaderCat));

llvm::cl::opt<bool> tfliteBiasScaleCheckThrowErrorOpt(
    "tflite-bias-scale-check-throw-error",
    llvm::cl::desc(
        "TensorFlowLite mandates that for quantized operators like Conv2D the "
        "bias quantization parameter biasScale = inputScale * weightsScale. If "
        "this contraint is not met within the given tolerance then an error "
        "will be thrown if this option is enabled."),
    llvm::cl::init(false), llvm::cl::Optional,
    llvm::cl::cat(tfliteModelLoaderCat));

/// Function to read a TensorFlowLite model from the file \p modelFilename into
/// the data buffer \p modelData provided by the caller. The \p modelData buffer
/// is allocated and initialized by this function but the caller must ensure its
/// existence through the graph loading process. \returns the TensorFlowLite
/// model object or Error in case something went wrong.
Expected<const tflite::Model *> readModel(std::vector<char> &modelData,
                                          const std::string &modelFilename) {
  // Open file.
  std::ifstream modelFile;
  modelFile.open(modelFilename, std::ios::binary);
  RETURN_ERR_IF_NOT(modelFile.is_open(),
                    strFormat("TensorFlowLite: Error opening model file '%s'!",
                              modelFilename.c_str()));
  // Get model size.
  modelFile.seekg(0, std::ios::end);
  std::streamsize modelSize = modelFile.tellg();
  modelFile.seekg(0, std::ios::beg);
  // Read model data.
  modelData = std::vector<char>(modelSize);
  RETURN_ERR_IF_NOT(modelFile.read(modelData.data(), modelSize),
                    strFormat("TensorFlowLite: Error reading model file '%s'!",
                              modelFilename.c_str()));
  modelFile.close();
  // Return model object.
  return tflite::GetModel(modelData.data());
}

/// Function to convert the UINT8 data from the buffer \p inpPtr to INT8 format
/// into the buffer \p outPtr. The buffer size is given by \p numElem. This
/// function is used to transform the UINT8 weights of a TensorFlowLite model to
/// INT8 format which is the format preferred and supported by Glow.
void convertUint8ToInt8(const uint8_t *inpPtr, int8_t *outPtr, size_t numElem) {
  for (size_t idx = 0, idxEnd = numElem; idx < idxEnd; ++idx) {
    int32_t val = inpPtr[idx];
    val -= UINT8_TO_INT8_SHIFT;
    outPtr[idx] = static_cast<int8_t>(val);
  }
}

/// Function to compute the padding according to the TensorFlowLite "SAME"
/// algorithm along a single dimension for the given input size \p inputSize and
/// output size \p outputSize and for the given filter (kernel) size \p kernel
/// \p stride and \p dilation. \returns a pair with the padding to be used for
/// the input before and after the actual input data.
std::pair<unsigned_t, unsigned_t> getSamePads(dim_t inputSize, dim_t outputSize,
                                              unsigned_t kernel,
                                              unsigned_t stride,
                                              unsigned_t dilation = 1) {
  // Effective dilated filter (kernel) size.
  unsigned_t effKernel = (kernel - 1) * dilation + 1;

  // Compute the total padding size while saturating above 0.
  unsigned_t padTotal = (outputSize - 1) * stride + effKernel;
  padTotal = std::max(padTotal, static_cast<unsigned_t>(inputSize)) - inputSize;

  // We split the total padding evenly before/after. If the padding is odd then
  // the "after" part gets the extra unit.
  unsigned_t padBefore = padTotal / 2;
  unsigned_t padAfter = padTotal - padBefore;
  return std::pair<unsigned_t, unsigned_t>(padBefore, padAfter);
}

/// Function to verify the quantization parameters of the bias operand. The
/// TensorFlowLite format mandates that the bias scale must be equal to the
/// product inputScale * weightsScale and the bias offset must be 0. This
/// function is provided with the module \p mod and the node values \p input,
/// \p weights, \p bias and \returns Error::success() if the bias parameters
/// are valid and Error otherwise.
Error checkBiasQuantizationParams(Module &mod, NodeValue input,
                                  NodeValue weights, NodeValue bias) {
  auto inputTy = input.getType();
  auto weightsTy = weights.getType();
  auto biasTy = bias.getType();
  if (inputTy->isQuantizedType() && weightsTy->isQuantizedType() &&
      biasTy->isQuantizedType()) {
    float inputScale = inputTy->getScale();
    float weightsScale = weightsTy->getScale();
    float matMulScale = inputScale * weightsScale;
    float biasScale = biasTy->getScale();
    // Check bias scale relative error to inputScale * weightsScale.
    if (biasScale != matMulScale) {
      float relErr = std::abs(matMulScale - biasScale) / matMulScale;
      llvm::errs() << strFormat(
          "TensorFlowLite: WARNING: Bias scale value was expected "
          "to be exactly %E (inputScale * weightsScale) but found "
          "%E instead! Relative absolute error is %E!\n",
          matMulScale, biasScale, relErr);
      if (relErr < tfliteBiasScaleCheckMaxErrorOpt) {
        // Set new bias type.
        TypeRef newBiasTy =
            mod.uniqueType(biasTy->getElementType(), biasTy->dims(),
                           matMulScale, biasTy->getOffset());
        bias.setType(newBiasTy);
        // If bias is constant we must also change the payload type.
        if (auto *biasC = llvm::dyn_cast<Constant>(bias.getNode())) {
          biasC->setPayloadType(newBiasTy);
        }
      } else if (tfliteBiasScaleCheckThrowErrorOpt) {
        RETURN_ERR(
            strFormat("TensorFlowLite: ERROR: Bias scale value was expected "
                      "to be exactly %E (inputScale * weightsScale) but found "
                      "%E instead! Relative absolute error is %E!\n",
                      matMulScale, biasScale, relErr));
      }
    }
    int32_t biasOffset = biasTy->getOffset();
    if (biasOffset != 0) {
      RETURN_ERR(strFormat("TensorFlowLite: Bias offset value was expected to "
                           "be 0 but found %d instead!",
                           biasOffset));
    }
  }
  return Error::success();
}

} // namespace

///===---------------------------------------------------------------------===//
///                              Tensor Utilities
///===---------------------------------------------------------------------===//
Expected<const tflite::Tensor *>
TFLiteModelLoader::getTensorByIndex(size_t index) {
  auto *tensors = graph_->tensors();
  RETURN_ERR_IF_NOT(
      index < tensors->size(),
      strFormat("TensorFlowLite: Tensor index %zu out of range!", index));
  return (*tensors)[index];
}

std::string TFLiteModelLoader::getTensorName(const tflite::Tensor *tensor) {
  return tensor->name()->str();
}

Expected<std::vector<dim_t>>
TFLiteModelLoader::getTensorShape(const tflite::Tensor *tensor) {
  std::vector<dim_t> shape;
  for (auto dim : *(tensor->shape())) {
    RETURN_ERR_IF_NOT(dim > 0,
                      strFormat("TensorFlowLite: Tensor '%s' has invalid shape "
                                "element '%d'!",
                                getTensorName(tensor).c_str(), dim));
    shape.push_back(static_cast<dim_t>(dim));
  }
  // If tensor shape is empty (scalar) we use a 1D shape with size 1.
  if (shape.empty()) {
    shape = {1};
  }
  return shape;
}

Expected<ElemKind>
TFLiteModelLoader::getTensorElemKind(const tflite::Tensor *tensor) {
  bool isQuantized = isTensorQuantized(tensor);
  switch (tensor->type()) {
  case tflite::TensorType_FLOAT32: {
    RETURN_ERR_IF_NOT(
        !isQuantized,
        "TensorFlowLite: FLOAT32 type should have no quantization parameters!");
    return ElemKind::FloatTy;
  }
  case tflite::TensorType_FLOAT16: {
    RETURN_ERR_IF_NOT(
        !isQuantized,
        "TensorFlowLite: FLOAT16 type should have no quantization parameters!");
    return ElemKind::Float16Ty;
  }
  case tflite::TensorType_INT8: {
    if (isQuantized) {
      return ElemKind::Int8QTy;
    } else {
      RETURN_ERR("TensorFlowLite: Non-quantized INT8 type not supported!");
    }
  }
  case tflite::TensorType_UINT8: {
    if (isQuantized) {
      // Convert UINT8 element type to INT8 element type.
      if (tfliteUint8ToInt8Opt) {
        return ElemKind::Int8QTy;
      } else {
        return ElemKind::UInt8QTy;
      }
    } else {
      RETURN_ERR("TensorFlowLite: Non-quantized UINT8 type not supported!");
    }
  }
  case tflite::TensorType_INT16: {
    if (isQuantized) {
      return ElemKind::Int16QTy;
    } else {
      RETURN_ERR("TensorFlowLite: Non-quantized INT16 type not supported!");
    }
  }
  case tflite::TensorType_INT32: {
    if (isQuantized) {
      return ElemKind::Int32QTy;
    } else {
      return ElemKind::Int32ITy;
    }
  }
  case tflite::TensorType_INT64: {
    if (isQuantized) {
      RETURN_ERR("TensorFlowLite: Quantized INT64 type not supported!");
    } else {
      return ElemKind::Int64ITy;
    }
  }
  case tflite::TensorType_BOOL: {
    RETURN_ERR_IF_NOT(
        !isQuantized,
        "TensorFlowLite: BOOL type should have no quantization parameters!");
    return ElemKind::BoolTy;
  }
  default:
    RETURN_ERR(strFormat("TensorFlowLite: Tensor '%s' type '%s' not supported!",
                         getTensorName(tensor).c_str(),
                         tflite::EnumNameTensorType(tensor->type())));
  }
}

bool TFLiteModelLoader::isTensorQuantized(const tflite::Tensor *tensor) {
  auto *tensorQParams = tensor->quantization();
  if (!tensorQParams) {
    return false;
  }
  auto *scales = tensorQParams->scale();
  auto *offsets = tensorQParams->zero_point();
  if (!(scales && offsets)) {
    return false;
  }
  if (!(scales->size() && offsets->size())) {
    return false;
  }
  return true;
}

bool TFLiteModelLoader::isTensorPerAxisQuantized(const tflite::Tensor *tensor) {
  if (!isTensorQuantized(tensor)) {
    return false;
  }
  auto *tensorQParams = tensor->quantization();
  auto *scales = tensorQParams->scale();
  auto *offsets = tensorQParams->zero_point();
  return (scales->size() > 1) && (offsets->size() > 1);
}

Expected<float>
TFLiteModelLoader::getTensorScale(const tflite::Tensor *tensor) {
  auto *tensorQParams = tensor->quantization();
  RETURN_ERR_IF_NOT(
      isTensorQuantized(tensor),
      strFormat("TensorFlowLite: Tensor '%s' has no quantization parameters!",
                getTensorName(tensor).c_str()));
  RETURN_ERR_IF_NOT(
      tensorQParams->details_type() == tflite::QuantizationDetails_NONE,
      strFormat("TensorFlowLite: Tensor '%s' has custom quantization which is "
                "not supported!",
                getTensorName(tensor).c_str()));
  auto *scales = tensorQParams->scale();
  RETURN_ERR_IF_NOT(scales->size() == 1,
                    strFormat("TensorFlowLite: Tensor '%s' has %d quantization "
                              "parameters but only one was expected!",
                              getTensorName(tensor).c_str(), scales->size()));
  float scale = (*scales)[0];
  return scale;
}

Expected<int32_t>
TFLiteModelLoader::getTensorOffset(const tflite::Tensor *tensor) {
  auto *tensorQParams = tensor->quantization();
  RETURN_ERR_IF_NOT(
      isTensorQuantized(tensor),
      strFormat("TensorFlowLite: Tensor '%s' has no quantization parameters!",
                getTensorName(tensor).c_str()));
  RETURN_ERR_IF_NOT(
      tensorQParams->details_type() == tflite::QuantizationDetails_NONE,
      strFormat("TensorFlowLite: Tensor '%s' has custom quantization which is "
                "not supported!",
                getTensorName(tensor).c_str()));
  auto *offsets = tensorQParams->zero_point();
  RETURN_ERR_IF_NOT(offsets->size() == 1,
                    strFormat("TensorFlowLite: Tensor '%s' has %d quantization "
                              "parameters but only one was expected!",
                              getTensorName(tensor).c_str(), offsets->size()));
  // TensorFlowLite defines the offset as int64 since it also supports int64
  // quantized type. Since Glow defines the offset as int32 we perform a cast
  // here and also validate that the offset is within the int32 range.
  int64_t offsetInt64 = (*offsets)[0];
  RETURN_ERR_IF_NOT(
      (std::numeric_limits<int32_t>::min() <= offsetInt64) &&
          (offsetInt64 <= std::numeric_limits<int32_t>::max()),
      strFormat(
          "TensorFlowLite: Tensor '%s' has an offset out of the int32 range!",
          getTensorName(tensor).c_str()));
  int32_t offset = static_cast<int32_t>(offsetInt64);
  // Convert UINT8 offset to INT8 offset.
  if (tfliteUint8ToInt8Opt && (tensor->type() == tflite::TensorType_UINT8)) {
    offset -= UINT8_TO_INT8_SHIFT;
  }
  return offset;
}

Expected<std::vector<float>>
TFLiteModelLoader::getTensorScales(const tflite::Tensor *tensor) {
  auto *tensorQParams = tensor->quantization();
  RETURN_ERR_IF_NOT(
      isTensorQuantized(tensor),
      strFormat("TensorFlowLite: Tensor '%s' has no quantization parameters!",
                getTensorName(tensor).c_str()));
  RETURN_ERR_IF_NOT(
      tensorQParams->details_type() == tflite::QuantizationDetails_NONE,
      strFormat("TensorFlowLite: Tensor '%s' has custom quantization which is "
                "not supported!",
                getTensorName(tensor).c_str()));
  auto *scales = tensorQParams->scale();
  RETURN_ERR_IF_NOT(scales->size() > 1,
                    strFormat("TensorFlowLite: Tensor '%s' has %d quantization "
                              "parameters but at least one was expected!",
                              getTensorName(tensor).c_str(), scales->size()));
  std::vector<float> scalesVec =
      std::vector<float>(scales->begin(), scales->end());
  return scalesVec;
}

Expected<std::vector<int32_t>>
TFLiteModelLoader::getTensorOffsets(const tflite::Tensor *tensor) {
  auto *tensorQParams = tensor->quantization();
  RETURN_ERR_IF_NOT(
      isTensorQuantized(tensor),
      strFormat("TensorFlowLite: Tensor '%s' has no quantization parameters!",
                getTensorName(tensor).c_str()));
  RETURN_ERR_IF_NOT(
      tensorQParams->details_type() == tflite::QuantizationDetails_NONE,
      strFormat("TensorFlowLite: Tensor '%s' has custom quantization which is "
                "not supported!",
                getTensorName(tensor).c_str()));
  auto *offsets = tensorQParams->zero_point();
  RETURN_ERR_IF_NOT(offsets->size() > 1,
                    strFormat("TensorFlowLite: Tensor '%s' has %d quantization "
                              "parameters but at least one was expected!",
                              getTensorName(tensor).c_str(), offsets->size()));
  // TensorFlowLite defines the offset as int64 since it also supports int64
  // quantized type. Since Glow defines the offset as int32 we perform a cast
  // here and also validate that the offset is within the int32 range.
  std::vector<int32_t> offsetsVec;
  for (auto offsetInt64 : *offsets) {
    RETURN_ERR_IF_NOT(
        (std::numeric_limits<int32_t>::min() <= offsetInt64) &&
            (offsetInt64 <= std::numeric_limits<int32_t>::max()),
        strFormat(
            "TensorFlowLite: Tensor '%s' has an offset out of the int32 range!",
            getTensorName(tensor).c_str()));
    int32_t offset = static_cast<int32_t>(offsetInt64);
    // Convert UINT8 offset to INT8 offset.
    if (tfliteUint8ToInt8Opt && (tensor->type() == tflite::TensorType_UINT8)) {
      offset -= UINT8_TO_INT8_SHIFT;
    }
    offsetsVec.push_back(offset);
  }
  return offsetsVec;
}

Expected<Type> TFLiteModelLoader::getTensorType(const tflite::Tensor *tensor) {
  ElemKind elemKind;
  ASSIGN_VALUE_OR_RETURN_ERR(elemKind, getTensorElemKind(tensor));
  std::vector<dim_t> shape;
  ASSIGN_VALUE_OR_RETURN_ERR(shape, getTensorShape(tensor));
  if (isQuantizedElemKind(elemKind)) {
    // If tensor is quantized per-axis we use a dummy scale 1.0 and offset 0.
    float scale = 1.0;
    int32_t offset = 0;
    if (!isTensorPerAxisQuantized(tensor)) {
      ASSIGN_VALUE_OR_RETURN_ERR(scale, getTensorScale(tensor));
      ASSIGN_VALUE_OR_RETURN_ERR(offset, getTensorOffset(tensor));
    }
    return Type(elemKind, shape, scale, offset);
  } else {
    return Type(elemKind, shape);
  }
}

Expected<std::pair<const char *, size_t>>
TFLiteModelLoader::getTensorDataAndSize(const tflite::Tensor *tensor) {
  uint32_t tensorBufferIdx = tensor->buffer();
  auto *modelBuffers = model_->buffers();
  RETURN_ERR_IF_NOT(tensorBufferIdx < modelBuffers->size(),
                    strFormat("TensorFlowLite: Tensor '%s' has a buffer index "
                              "out of range!",
                              getTensorName(tensor).c_str()));
  const char *tensorData = nullptr;
  size_t tensorSize = 0;
  if (auto *buffer = (*modelBuffers)[tensorBufferIdx]) {
    if (auto *array = buffer->data()) {
      if (array->size()) {
        tensorData =
            const_cast<char *>(reinterpret_cast<const char *>(array->data()));
        tensorSize = array->size();
      }
    }
  }
  return std::pair<const char *, size_t>(tensorData, tensorSize);
}

///===---------------------------------------------------------------------===//
///                              Operator Utilities
///===---------------------------------------------------------------------===//
Expected<tflite::BuiltinOperator>
TFLiteModelLoader::getOperatorCode(const tflite::Operator *op) {
  const auto *modelOpCodes = model_->operator_codes();
  auto opCodeIdx = op->opcode_index();
  RETURN_ERR_IF_NOT(opCodeIdx < modelOpCodes->size(),
                    strFormat("TensorFlowLite: Missing registration for "
                              "opcode_index %d!",
                              opCodeIdx));
  auto *opCode = (*modelOpCodes)[opCodeIdx];
  auto builtinCode = opCode->builtin_code();
  RETURN_ERR_IF_NOT(
      (tflite::BuiltinOperator_MIN <= builtinCode) &&
          (builtinCode <= tflite::BuiltinOperator_MAX),
      strFormat(
          "TensorFlowLite: Operator builtin_code %d out of the supported "
          "range! You might be using a newer model than currently supported!",
          builtinCode));
  return builtinCode;
}

Expected<int32_t>
TFLiteModelLoader::getOperatorVersion(const tflite::Operator *op) {
  const auto *modelOpCodes = model_->operator_codes();
  auto opCodeIdx = op->opcode_index();
  RETURN_ERR_IF_NOT(opCodeIdx < modelOpCodes->size(),
                    strFormat("TensorFlowLite: Missing registration for "
                              "opcode_index %d!",
                              opCodeIdx));
  auto *opCode = (*modelOpCodes)[opCodeIdx];
  return opCode->version();
}

Expected<std::string>
TFLiteModelLoader::getOperatorType(const tflite::Operator *op) {
  tflite::BuiltinOperator opCode;
  ASSIGN_VALUE_OR_RETURN_ERR(opCode, getOperatorCode(op));
  return std::string(tflite::EnumNameBuiltinOperator(opCode));
}

Expected<std::string>
TFLiteModelLoader::getOperatorName(const tflite::Operator *op) {
  std::string opType;
  ASSIGN_VALUE_OR_RETURN_ERR(opType, getOperatorType(op));
  const auto *opOutputs = op->outputs();
  // If operator has no outputs then we return the operator type name.
  if (opOutputs->size() == 0) {
    return opType;
  }
  // If the first output tensor corresponds to an output placeholder then we use
  // the operator type name in order to preserve the output placeholder name.
  size_t opOutIdx = static_cast<size_t>((*opOutputs)[0]);
  const auto *graphOutputs = graph_->outputs();
  for (auto graphOutIdx : (*graphOutputs)) {
    if (opOutIdx == graphOutIdx) {
      return opType;
    }
  }
  // Return the name of the first output tensor.
  const tflite::Tensor *tensor;
  ASSIGN_VALUE_OR_RETURN_ERR(tensor, getTensorByIndex(opOutIdx));
  return getTensorName(tensor);
}

Expected<size_t>
TFLiteModelLoader::getOperatorInputTensorIdx(const tflite::Operator *op,
                                             size_t inputIdx) {
  std::string opType;
  ASSIGN_VALUE_OR_RETURN_ERR(opType, getOperatorType(op));
  const auto *opInputs = op->inputs();
  RETURN_ERR_IF_NOT(opInputs,
                    strFormat("TensorFlowLite: Operator '%s' has no inputs!",
                              opType.c_str()));
  RETURN_ERR_IF_NOT(inputIdx < opInputs->size(),
                    strFormat("TensorFlowLite: Operator '%s' input index %zu "
                              "is out of range! Operator has %d inputs!",
                              opType.c_str(), inputIdx, opInputs->size()));
  return static_cast<size_t>((*opInputs)[inputIdx]);
}

Expected<size_t>
TFLiteModelLoader::getOperatorOutputTensorIdx(const tflite::Operator *op,
                                              size_t outputIdx) {
  std::string opType;
  ASSIGN_VALUE_OR_RETURN_ERR(opType, getOperatorType(op));
  const auto *opOutputs = op->outputs();
  RETURN_ERR_IF_NOT(opOutputs,
                    strFormat("TensorFlowLite: Operator '%s' has no outputs!",
                              opType.c_str()));
  RETURN_ERR_IF_NOT(outputIdx < opOutputs->size(),
                    strFormat("TensorFlowLite: Operator '%s' output index %zu "
                              "is out of range! Operator has %d outputs!",
                              opType.c_str(), outputIdx, opOutputs->size()));
  return static_cast<size_t>((*opOutputs)[outputIdx]);
}

Expected<bool>
TFLiteModelLoader::isOperatorOutputFinalTensor(const tflite::Operator *op,
                                               size_t outputIdx) {
  size_t tensorIdx;
  ASSIGN_VALUE_OR_RETURN_ERR(tensorIdx,
                             getOperatorOutputTensorIdx(op, outputIdx));
  const auto *graphOutputs = graph_->outputs();
  for (auto graphOutIdx : (*graphOutputs)) {
    if (tensorIdx == graphOutIdx) {
      return true;
    }
  }
  return false;
}

Expected<NodeValue> TFLiteModelLoader::getNodeValueByIndex(size_t index) {
  RETURN_ERR_IF_NOT(!nodeValueByIndex_.empty(),
                    "TensorFlowLite: Node value array not initialized!");
  RETURN_ERR_IF_NOT(
      index < nodeValueByIndex_.size(),
      strFormat("TensorFlowLite: Node value index %zu is out of range!",
                index));
  NodeValue nodeValue = nodeValueByIndex_[index];
  RETURN_ERR_IF_NOT(nodeValue.getNode(),
                    strFormat("TensorFlowLite: Node value with index %zu is "
                              "null (not initialized)!",
                              index));
  return nodeValue;
}

Error TFLiteModelLoader::setNodeValueByIndex(size_t index,
                                             NodeValue nodeValue) {
  RETURN_ERR_IF_NOT(!nodeValueByIndex_.empty(),
                    "TensorFlowLite: Node value array not initialized!");
  RETURN_ERR_IF_NOT(
      index < nodeValueByIndex_.size(),
      strFormat("TensorFlowLite: Node value index %zu is out of range!",
                index));
  nodeValueByIndex_[index] = nodeValue;
  return Error::success();
}

Expected<NodeValue>
TFLiteModelLoader::getInputNodeValue(const tflite::Operator *op,
                                     size_t inputIdx) {
  size_t tensorIdx;
  ASSIGN_VALUE_OR_RETURN_ERR(tensorIdx,
                             getOperatorInputTensorIdx(op, inputIdx));
  return getNodeValueByIndex(tensorIdx);
}

Error TFLiteModelLoader::setOutputNodeValue(const tflite::Operator *op,
                                            NodeValue nodeValue) {
  std::vector<NodeValue> nodeValues = {nodeValue};
  return setOutputNodeValues(op, nodeValues);
}

Error TFLiteModelLoader::setOutputNodeValues(
    const tflite::Operator *op, llvm::ArrayRef<NodeValue> nodeValues) {
  std::string opType;
  ASSIGN_VALUE_OR_RETURN_ERR(opType, getOperatorType(op));
  const auto *opOutputs = op->outputs();
  RETURN_ERR_IF_NOT(
      opOutputs->size() == nodeValues.size(),
      strFormat("TensorFlowLite: Operator '%s' has %d outputs but %zu are set!",
                opType.c_str(), opOutputs->size(), nodeValues.size()));
  for (size_t idx = 0, idxEnd = nodeValues.size(); idx < idxEnd; ++idx) {
    NodeValue outNodeValue = nodeValues[idx];
    // Verify the output type of the node value matches the type registered in
    // the model with the exception of the final tensors which are allowed to
    // be modified (for example for the Softmax output when it is a final node).
    TypeRef outTy;
    ASSIGN_VALUE_OR_RETURN_ERR(outTy, getOutputType(op, idx));
    bool isFinal;
    ASSIGN_VALUE_OR_RETURN_ERR(isFinal, isOperatorOutputFinalTensor(op, idx));
    RETURN_ERR_IF_NOT(isFinal || outTy->isEqual(outNodeValue.getType()),
                      strFormat("TensorFlowLite: Operator '%s' modifies the "
                                "output type registered in the model!",
                                opType.c_str()));
    // Register the output node value.
    size_t tensorIdx = static_cast<size_t>((*opOutputs)[idx]);
    RETURN_IF_ERR(setNodeValueByIndex(tensorIdx, outNodeValue));
  }
  return Error::success();
}

Expected<TypeRef> TFLiteModelLoader::getOutputType(const tflite::Operator *op,
                                                   size_t outputIndex) {
  size_t tensorIdx;
  ASSIGN_VALUE_OR_RETURN_ERR(tensorIdx,
                             getOperatorOutputTensorIdx(op, outputIndex));
  const tflite::Tensor *tensor;
  ASSIGN_VALUE_OR_RETURN_ERR(tensor, getTensorByIndex(tensorIdx));
  Type type;
  ASSIGN_VALUE_OR_RETURN_ERR(type, getTensorType(tensor));
  return mod_.uniqueType(type);
}

void TFLiteModelLoader::initializeNodeValues() {
  auto numTensors = graph_->tensors()->size();
  nodeValueByIndex_ = std::vector<NodeValue>(numTensors, nullptr);
}

Error TFLiteModelLoader::loadInputPlaceholders() {
  for (auto inpIdx : *(graph_->inputs())) {
    // Get input placeholder name and type.
    const tflite::Tensor *tensor;
    ASSIGN_VALUE_OR_RETURN_ERR(tensor, getTensorByIndex(inpIdx));
    std::string name = getTensorName(tensor);
    Type type;
    ASSIGN_VALUE_OR_RETURN_ERR(type, getTensorType(tensor));
    // Create input placeholder. If the input type is quantized and a float
    // input is requested then we create a float placeholder and a Quantize
    // node, otherwise we create directly the quantized placeholder.
    Placeholder *inpPH;
    NodeValue inpNV;
    if (tfliteFloatInputsOpt && type.isQuantizedType()) {
      TypeRef floatType = mod_.uniqueType(ElemKind::FloatTy, type.dims());
      inpPH = mod_.createPlaceholder(floatType, name, /*isTrainable*/ false,
                                     ANY_LAYOUT);
      inpNV = F_->createQuantize(name + ".Quantize", inpPH, &type);
    } else {
      inpPH = mod_.createPlaceholder(&type, name, /*isTrainable*/ false,
                                     ANY_LAYOUT);
      inpNV = inpPH;
    }
    // Register placeholder by model input name.
    inputPlaceholderByName_.try_emplace(name, inpPH);
    // Set input node value.
    RETURN_IF_ERR(setNodeValueByIndex(inpIdx, inpNV));
  }
  return Error::success();
}

Error TFLiteModelLoader::loadConstants() {
  const auto *tensors = graph_->tensors();
  for (size_t idx = 0, idxEnd = tensors->size(); idx < idxEnd; ++idx) {
    // Get tensor data and size. A TensorFlowLite model tensor is a constant
    // if it has data stored in the model.
    const tflite::Tensor *tensor = (*tensors)[idx];
    std::pair<const char *, size_t> dataAndSize;
    ASSIGN_VALUE_OR_RETURN_ERR(dataAndSize, getTensorDataAndSize(tensor));
    if (dataAndSize.first == nullptr) {
      continue;
    }
    // Create tensor and initialize data.
    std::string name = getTensorName(tensor);
    Type type;
    ASSIGN_VALUE_OR_RETURN_ERR(type, getTensorType(tensor));
    RETURN_ERR_IF_NOT(
        type.getSizeInBytes() == dataAndSize.second,
        strFormat("TensorFlowLite: Tensor '%s' mismatch between shape based "
                  "size (%zu bytes) and actual data size (%lu bytes)!",
                  name.c_str(), type.getSizeInBytes(), dataAndSize.second));
    Tensor T = Tensor(type);
    T.copyRawFrom(dataAndSize.first);
    // Convert UINT8 data to INT8 data.
    if (tfliteUint8ToInt8Opt && (tensor->type() == tflite::TensorType_UINT8)) {
      convertUint8ToInt8(reinterpret_cast<uint8_t *>(T.getUnsafePtr()),
                         reinterpret_cast<int8_t *>(T.getUnsafePtr()),
                         dataAndSize.second);
    }
    // Create constant.
    Constant *node = mod_.createConstant(name, std::move(T), ANY_LAYOUT);
    // Register node value.
    RETURN_IF_ERR(setNodeValueByIndex(idx, node->getOutput()));
  }
  return Error::success();
}

Error TFLiteModelLoader::loadOperators() {
  OperatorInfo opInfo;
  auto *graphOperators = graph_->operators();
  for (size_t opIdx = 0, opIdxEnd = graphOperators->size(); opIdx < opIdxEnd;
       ++opIdx) {
    // Get operator meta data.
    const tflite::Operator *op = (*graphOperators)[opIdx];
    ASSIGN_VALUE_OR_RETURN_ERR(opInfo.name, getOperatorName(op));
    ASSIGN_VALUE_OR_RETURN_ERR(opInfo.type, getOperatorType(op));
    ASSIGN_VALUE_OR_RETURN_ERR(opInfo.code, getOperatorCode(op));
    ASSIGN_VALUE_OR_RETURN_ERR(opInfo.version, getOperatorVersion(op));
    opInfo.index = opIdx;
    // Load operator.
    mod_.registerOriginalName(opInfo.name);
    RETURN_IF_ERR(loadOperator(op, opInfo));
  }
  return Error::success();
}

Error TFLiteModelLoader::saveOutputPlaceholders() {
  for (auto outIdx : *(graph_->outputs())) {
    // Get placeholder name.
    const tflite::Tensor *tensor;
    ASSIGN_VALUE_OR_RETURN_ERR(tensor, getTensorByIndex(outIdx));
    std::string name = getTensorName(tensor);
    // Save output placeholder. If the output type is quantized and a float
    // output is requested then we create create a Dequantize node and save
    // into a float placeholder, otherwise we save the quantized placeholder.
    NodeValue outNodeValue;
    ASSIGN_VALUE_OR_RETURN_ERR(outNodeValue, getNodeValueByIndex(outIdx));
    if (tfliteFloatOutputsOpt && outNodeValue.getType()->isQuantizedType()) {
      outNodeValue = F_->createDequantize(name + ".Dequantize", outNodeValue,
                                          ElemKind::FloatTy);
    }
    auto *saveNode = F_->createSave(name, outNodeValue);
    // Register placeholder by model output name.
    outputPlaceholderByName_.try_emplace(name, saveNode->getPlaceholder());
  }
  return Error::success();
}

Error TFLiteModelLoader::addActivation(NodeValue &value,
                                       tflite::ActivationFunctionType type) {
  std::string nodeName = value.getNode()->getName().str();
  std::string actType = EnumNameActivationFunctionType(type);
  std::string actName = nodeName + "." + actType;
  if (type == tflite::ActivationFunctionType_NONE) {
    return Error::success();
  }
  if (type == tflite::ActivationFunctionType_RELU) {
    value = F_->createRELU(actName, value);
    return Error::success();
  }
  if (type == tflite::ActivationFunctionType_RELU_N1_TO_1) {
    value = F_->createClip(actName, value, -1.0, 1.0);
    return Error::success();
  }
  if (type == tflite::ActivationFunctionType_RELU6) {
    value = F_->createClip(actName, value, 0.0, 6.0);
    return Error::success();
  }
  if (type == tflite::ActivationFunctionType_TANH) {
    value = F_->createTanh(actName, value);
    return Error::success();
  }
  RETURN_ERR(strFormat("TensorFlowLite: Activation type '%s' is not supported!",
                       actType.c_str()));
}

const std::string TFLiteModelLoader::opErrMsg(const OperatorInfo &opInfo,
                                              const std::string &errMsg) {
  return strFormat("TensorFlowLite: Operator '%s' (Index %zu, Code %u): %s",
                   opInfo.type.c_str(), opInfo.index, opInfo.code,
                   errMsg.c_str());
}

template <typename T>
Expected<T> TFLiteModelLoader::loadAxis(const OperatorInfo &opInfo,
                                        NodeValue axis, NodeValue value) {
  Constant *axisC = llvm::dyn_cast<Constant>(axis.getNode());
  RETURN_ERR_IF_NOT(axisC,
                    opErrMsg(opInfo, "Non constant axis not supported!"));
  RETURN_ERR_IF_NOT(axisC->getType()->size() == 1,
                    opErrMsg(opInfo, "Axis should have 1 element!"));
  T axisVal;
  auto elemType = axisC->getType()->getElementType();
  if (elemType == ElemKind::Int32ITy) {
    auto axisH = axisC->getPayload().getHandle<int32_t>();
    ASSIGN_VALUE_OR_RETURN_ERR(
        axisVal, getPositiveAxis<T>(static_cast<int>(axisH.raw(0)), value));
  } else if (elemType == ElemKind::Int64ITy) {
    auto axisH = axisC->getPayload().getHandle<int64_t>();
    ASSIGN_VALUE_OR_RETURN_ERR(
        axisVal, getPositiveAxis<T>(static_cast<int>(axisH.raw(0)), value));
  } else {
    RETURN_ERR(opErrMsg(opInfo, "Axis should have INT32 or INT64 type!"));
  }
  return axisVal;
}

template <typename T>
Expected<std::vector<T>> TFLiteModelLoader::loadAxes(const OperatorInfo &opInfo,
                                                     NodeValue axes,
                                                     NodeValue value) {
  Constant *axesC = llvm::dyn_cast<Constant>(axes.getNode());
  RETURN_ERR_IF_NOT(axesC,
                    opErrMsg(opInfo, "Non constant axis not supported!"));
  RETURN_ERR_IF_NOT(axesC->getType()->size() >= 1,
                    opErrMsg(opInfo, "Axis should have at least 1 element!"));
  std::vector<T> axesVal = std::vector<T>(axesC->getType()->size());
  auto elemType = axesC->getType()->getElementType();
  for (size_t idx = 0; idx < axesC->getType()->size(); ++idx) {
    if (elemType == ElemKind::Int32ITy) {
      auto axesH = axesC->getPayload().getHandle<int32_t>();
      ASSIGN_VALUE_OR_RETURN_ERR(
          axesVal[idx],
          getPositiveAxis<T>(static_cast<int>(axesH.raw(idx)), value));
    } else if (elemType == ElemKind::Int64ITy) {
      auto axesH = axesC->getPayload().getHandle<int64_t>();
      ASSIGN_VALUE_OR_RETURN_ERR(
          axesVal[idx],
          getPositiveAxis<T>(static_cast<int>(axesH.raw(idx)), value));
    } else {
      RETURN_ERR(opErrMsg(opInfo, "Axis should have INT32 or INT64 type!"));
    }
  }
  return axesVal;
}

template <typename T>
Expected<std::vector<T>>
TFLiteModelLoader::loadArray(const OperatorInfo &opInfo, NodeValue value) {
  Constant *valueC = llvm::dyn_cast<Constant>(value.getNode());
  RETURN_ERR_IF_NOT(valueC,
                    opErrMsg(opInfo, "Non constant array not supported!"));
  auto valueSize = valueC->getType()->size();
  RETURN_ERR_IF_NOT(valueSize >= 1,
                    opErrMsg(opInfo, "Array should have at least 1 element!"));
  std::vector<T> valueV = std::vector<T>(valueSize);
  auto elemType = valueC->getType()->getElementType();
  for (size_t idx = 0; idx < valueSize; ++idx) {
    if (elemType == ElemKind::FloatTy) {
      auto valueH = valueC->getPayload().getHandle<float>();
      valueV[idx] = static_cast<T>(valueH.raw(idx));
    } else if (elemType == ElemKind::Int32ITy) {
      auto valueH = valueC->getPayload().getHandle<int32_t>();
      valueV[idx] = static_cast<T>(valueH.raw(idx));
    } else if (elemType == ElemKind::Int64ITy) {
      auto valueH = valueC->getPayload().getHandle<int64_t>();
      valueV[idx] = static_cast<T>(valueH.raw(idx));
    } else {
      RETURN_ERR(opErrMsg(opInfo, "Array type not supported!"));
    }
  }
  return valueV;
}

Expected<bool> TFLiteModelLoader::isConv2DPerAxisQuantized(
    const tflite::Operator *op, const OperatorInfo &opInfo,
    Constant *&filterScalesC, Constant *&filterOffsetsC, Constant *&biasScalesC,
    Constant *&biasOffsetsC) {
  // Get filter/bias tensors.
  NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(input, getInputNodeValue(op, 0));
  TypeRef outTy;
  ASSIGN_VALUE_OR_RETURN_ERR(outTy, getOutputType(op, 0));
  size_t filterTensorIdx;
  ASSIGN_VALUE_OR_RETURN_ERR(filterTensorIdx, getOperatorInputTensorIdx(op, 1));
  size_t biasTensorIdx;
  ASSIGN_VALUE_OR_RETURN_ERR(biasTensorIdx, getOperatorInputTensorIdx(op, 2));
  const tflite::Tensor *filterTensor;
  ASSIGN_VALUE_OR_RETURN_ERR(filterTensor, getTensorByIndex(filterTensorIdx));
  const tflite::Tensor *biasTensor;
  ASSIGN_VALUE_OR_RETURN_ERR(biasTensor, getTensorByIndex(biasTensorIdx));

  bool isPerAxisQuantized = isTensorPerAxisQuantized(filterTensor) &&
                            isTensorPerAxisQuantized(biasTensor);

  // If it is not per-axis quantized return directly.
  if (!isPerAxisQuantized) {
    filterScalesC = nullptr;
    filterOffsetsC = nullptr;
    biasScalesC = nullptr;
    biasOffsetsC = nullptr;
    return false;
  }

  dim_t numChannels = outTy->dims().back();

  // Get filter/bias quantization parameters.
  std::vector<float> filterScalesV;
  ASSIGN_VALUE_OR_RETURN_ERR(filterScalesV, getTensorScales(filterTensor));
  std::vector<int32_t> filterOffsetsV;
  ASSIGN_VALUE_OR_RETURN_ERR(filterOffsetsV, getTensorOffsets(filterTensor));
  std::vector<float> biasScalesV;
  ASSIGN_VALUE_OR_RETURN_ERR(biasScalesV, getTensorScales(biasTensor));
  std::vector<int32_t> biasOffsetsV;
  ASSIGN_VALUE_OR_RETURN_ERR(biasOffsetsV, getTensorOffsets(biasTensor));

  // Create filter/bias quantization parameters graph constants.
  filterScalesC =
      mod_.createConstant(ElemKind::FloatTy, {numChannels}, "filterScales");
  filterOffsetsC =
      mod_.createConstant(ElemKind::Int32ITy, {numChannels}, "filterOffsets");
  biasScalesC =
      mod_.createConstant(ElemKind::FloatTy, {numChannels}, "biasScales");
  biasOffsetsC =
      mod_.createConstant(ElemKind::Int32ITy, {numChannels}, "biasOffsets");

  RETURN_ERR_IF_NOT(
      filterScalesV.size() == numChannels,
      opErrMsg(opInfo,
               "Weights scales length should match the output channels!"));
  RETURN_ERR_IF_NOT(
      filterOffsetsV.size() == numChannels,
      opErrMsg(opInfo,
               "Weights offsets length should match the output channels!"));
  RETURN_ERR_IF_NOT(
      biasScalesV.size() == numChannels,
      opErrMsg(opInfo, "Bias scales length should match the output channels!"));
  RETURN_ERR_IF_NOT(
      biasOffsetsV.size() == numChannels,
      opErrMsg(opInfo,
               "Bias offsets length should match the output channels!"));

  filterScalesC->getPayloadMutable().copyRawFrom(
      reinterpret_cast<const char *>(filterScalesV.data()));
  filterOffsetsC->getPayloadMutable().copyRawFrom(
      reinterpret_cast<const char *>(filterOffsetsV.data()));
  biasScalesC->getPayloadMutable().copyRawFrom(
      reinterpret_cast<const char *>(biasScalesV.data()));
  biasOffsetsC->getPayloadMutable().copyRawFrom(
      reinterpret_cast<const char *>(biasOffsetsV.data()));

  // Validate filter/bias quantization parameters.
  float inputScale = input.getType()->getScale();
  auto filterScalesH = filterScalesC->getPayload().getHandle<float>();
  auto filterOffsetsH = filterOffsetsC->getPayload().getHandle<int32_t>();
  auto biasScalesH = biasScalesC->getPayload().getHandle<float>();
  auto biasOffsetsH = biasOffsetsC->getPayload().getHandle<int32_t>();
  for (size_t idx = 0; idx < numChannels; ++idx) {
    // TensorFlowLite mandates that filterOffset and biasOffset are 0.
    RETURN_ERR_IF_NOT(filterOffsetsH.raw(idx) == 0,
                      opErrMsg(opInfo, "Filter offset was expected to be 0!"));
    RETURN_ERR_IF_NOT(biasOffsetsH.raw(idx) == 0,
                      opErrMsg(opInfo, "Bias offset was expected to be 0!"));

    float filterScale = filterScalesH.raw(idx);
    float matMulScale = inputScale * filterScale;
    float biasScale = biasScalesH.raw(idx);

    // Check bias scale relative error to inputScale * filterScale.
    if (biasScale != matMulScale) {
      float relErr = std::abs(matMulScale - biasScale) / matMulScale;
      llvm::errs() << opErrMsg(
          opInfo,
          strFormat("WARNING: Bias scale value was expected "
                    "to be exactly %E (inputScale * weightsScale) but found "
                    "%E instead! Relative absolute error is %E!\n",
                    matMulScale, biasScale, relErr));
      if (relErr < tfliteBiasScaleCheckMaxErrorOpt) {
        // Modify bias scale.
        biasScalesH.raw(idx) = matMulScale;
      } else if (tfliteBiasScaleCheckThrowErrorOpt) {
        RETURN_ERR(opErrMsg(
            opInfo,
            strFormat("ERROR: Bias scale value was expected "
                      "to be exactly %E (inputScale * weightsScale) but found "
                      "%E instead! Relative absolute error is %E!\n",
                      matMulScale, biasScale, relErr)));
      }
    }
  }

  return true;
}

Error TFLiteModelLoader::loadOperator(const tflite::Operator *op,
                                      const OperatorInfo &opInfo) {
  // Opcodes are treated in increasing order to allow easy tracking
  // for which operators are supported and which are not.
  auto opCode = opInfo.code;
  if (opCode == tflite::BuiltinOperator_ADD) {
    return loadBinaryArithmetic(op, opInfo);
  }
  if (opCode == tflite::BuiltinOperator_AVERAGE_POOL_2D) {
    return loadPool2D(op, opInfo);
  }
  if (opCode == tflite::BuiltinOperator_CONCATENATION) {
    return loadConcat(op, opInfo);
  }
  if (opCode == tflite::BuiltinOperator_CONV_2D) {
    return loadConv2D(op, opInfo);
  }
  if (opCode == tflite::BuiltinOperator_DEPTHWISE_CONV_2D) {
    return loadDepthwiseConv2D(op, opInfo);
  }
  if (opCode == tflite::BuiltinOperator_DEQUANTIZE) {
    return loadUnaryArithmetic(op, opInfo);
  }
  if (opCode == tflite::BuiltinOperator_FLOOR) {
    return loadUnaryArithmetic(op, opInfo);
  }
  if (opCode == tflite::BuiltinOperator_FULLY_CONNECTED) {
    return loadFullyConnected(op, opInfo);
  }
  if (opCode == tflite::BuiltinOperator_LOGISTIC) {
    return loadUnaryArithmetic(op, opInfo);
  }
  if (opCode == tflite::BuiltinOperator_MAX_POOL_2D) {
    return loadPool2D(op, opInfo);
  }
  if (opCode == tflite::BuiltinOperator_MUL) {
    return loadBinaryArithmetic(op, opInfo);
  }
  if (opCode == tflite::BuiltinOperator_RELU) {
    return loadUnaryArithmetic(op, opInfo);
  }
  if (opCode == tflite::BuiltinOperator_RELU_N1_TO_1) {
    return loadUnaryArithmetic(op, opInfo);
  }
  if (opCode == tflite::BuiltinOperator_RELU6) {
    return loadUnaryArithmetic(op, opInfo);
  }
  if (opCode == tflite::BuiltinOperator_RESHAPE) {
    return loadReshape(op, opInfo);
  }
  if (opCode == tflite::BuiltinOperator_SOFTMAX) {
    return loadSoftmax(op, opInfo);
  }
  if (opCode == tflite::BuiltinOperator_TANH) {
    return loadUnaryArithmetic(op, opInfo);
  }
  if (opCode == tflite::BuiltinOperator_PAD) {
    return loadPad(op, opInfo);
  }
  if (opCode == tflite::BuiltinOperator_TRANSPOSE) {
    return loadTranspose(op, opInfo);
  }
  if (opCode == tflite::BuiltinOperator_MEAN) {
    return loadReduce(op, opInfo);
  }
  if (opCode == tflite::BuiltinOperator_SUB) {
    return loadBinaryArithmetic(op, opInfo);
  }
  if (opCode == tflite::BuiltinOperator_DIV) {
    return loadBinaryArithmetic(op, opInfo);
  }
  if (opCode == tflite::BuiltinOperator_SQUEEZE) {
    return loadReshape(op, opInfo);
  }
  if (opCode == tflite::BuiltinOperator_EXP) {
    return loadUnaryArithmetic(op, opInfo);
  }
  if (opCode == tflite::BuiltinOperator_SPLIT) {
    return loadSplit(op, opInfo);
  }
  if (opCode == tflite::BuiltinOperator_PRELU) {
    return loadBinaryArithmetic(op, opInfo);
  }
  if (opCode == tflite::BuiltinOperator_MAXIMUM) {
    return loadBinaryArithmetic(op, opInfo);
  }
  if (opCode == tflite::BuiltinOperator_ARG_MAX) {
    return loadArg(op, opInfo);
  }
  if (opCode == tflite::BuiltinOperator_MINIMUM) {
    return loadBinaryArithmetic(op, opInfo);
  }
  if (opCode == tflite::BuiltinOperator_LESS) {
    return loadBinaryArithmetic(op, opInfo);
  }
  if (opCode == tflite::BuiltinOperator_NEG) {
    return loadUnaryArithmetic(op, opInfo);
  }
  if (opCode == tflite::BuiltinOperator_GREATER) {
    return loadBinaryArithmetic(op, opInfo);
  }
  if (opCode == tflite::BuiltinOperator_GREATER_EQUAL) {
    return loadBinaryArithmetic(op, opInfo);
  }
  if (opCode == tflite::BuiltinOperator_LESS_EQUAL) {
    return loadBinaryArithmetic(op, opInfo);
  }
  if (opCode == tflite::BuiltinOperator_SLICE) {
    return loadSlice(op, opInfo);
  }
  if (opCode == tflite::BuiltinOperator_SIN) {
    return loadUnaryArithmetic(op, opInfo);
  }
  if (opCode == tflite::BuiltinOperator_TILE) {
    return loadTile(op, opInfo);
  }
  if (opCode == tflite::BuiltinOperator_EXPAND_DIMS) {
    return loadReshape(op, opInfo);
  }
  if (opCode == tflite::BuiltinOperator_EQUAL) {
    return loadBinaryArithmetic(op, opInfo);
  }
  if (opCode == tflite::BuiltinOperator_NOT_EQUAL) {
    return loadBinaryArithmetic(op, opInfo);
  }
  if (opCode == tflite::BuiltinOperator_LOG) {
    return loadUnaryArithmetic(op, opInfo);
  }
  if (opCode == tflite::BuiltinOperator_SQRT) {
    return loadUnaryArithmetic(op, opInfo);
  }
  if (opCode == tflite::BuiltinOperator_RSQRT) {
    return loadUnaryArithmetic(op, opInfo);
  }
  if (opCode == tflite::BuiltinOperator_POW) {
    return loadBinaryArithmetic(op, opInfo);
  }
  if (opCode == tflite::BuiltinOperator_ARG_MIN) {
    return loadArg(op, opInfo);
  }
  if (opCode == tflite::BuiltinOperator_PACK) {
    return loadPack(op, opInfo);
  }
  if (opCode == tflite::BuiltinOperator_LOGICAL_OR) {
    return loadBinaryArithmetic(op, opInfo);
  }
  if (opCode == tflite::BuiltinOperator_LOGICAL_AND) {
    return loadBinaryArithmetic(op, opInfo);
  }
  if (opCode == tflite::BuiltinOperator_LOGICAL_NOT) {
    return loadUnaryArithmetic(op, opInfo);
  }
  if (opCode == tflite::BuiltinOperator_UNPACK) {
    return loadUnpack(op, opInfo);
  }
  if (opCode == tflite::BuiltinOperator_SQUARE) {
    return loadUnaryArithmetic(op, opInfo);
  }
  if (opCode == tflite::BuiltinOperator_LEAKY_RELU) {
    return loadUnaryArithmetic(op, opInfo);
  }
  if (opCode == tflite::BuiltinOperator_ABS) {
    return loadUnaryArithmetic(op, opInfo);
  }
  if (opCode == tflite::BuiltinOperator_CEIL) {
    return loadUnaryArithmetic(op, opInfo);
  }
  if (opCode == tflite::BuiltinOperator_COS) {
    return loadUnaryArithmetic(op, opInfo);
  }
  if (opCode == tflite::BuiltinOperator_QUANTIZE) {
    return loadUnaryArithmetic(op, opInfo);
  }
  if (opCode == tflite::BuiltinOperator_ROUND) {
    return loadUnaryArithmetic(op, opInfo);
  }

  RETURN_ERR(strFormat("TensorFlowLite: Operator type '%s' is not supported!",
                       opInfo.type.c_str()));
}

Error TFLiteModelLoader::loadUnaryArithmetic(const tflite::Operator *op,
                                             const OperatorInfo &opInfo) {
  NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(input, getInputNodeValue(op, 0));
  TypeRef outTy;
  ASSIGN_VALUE_OR_RETURN_ERR(outTy, getOutputType(op, 0));

  auto opCode = opInfo.code;
  NodeValue output;
  if (opCode == tflite::BuiltinOperator_LOGISTIC) {
    output = F_->createSigmoid(opInfo.name, outTy, input);
  } else if (opCode == tflite::BuiltinOperator_RELU) {
    output = F_->createRELU(opInfo.name, input, outTy);
  } else if (opCode == tflite::BuiltinOperator_RELU_N1_TO_1) {
    output = F_->createClip(opInfo.name, input, outTy, -1.0, 1.0);
  } else if (opCode == tflite::BuiltinOperator_RELU6) {
    output = F_->createClip(opInfo.name, input, outTy, 0.0, 6.0);
  } else if (opCode == tflite::BuiltinOperator_TANH) {
    output = F_->createTanh(opInfo.name, outTy, input);
  } else if (opCode == tflite::BuiltinOperator_EXP) {
    output = F_->createExp(opInfo.name, input);
  } else if (opCode == tflite::BuiltinOperator_LOG) {
    output = F_->createLog(opInfo.name, input, outTy);
  } else if (opCode == tflite::BuiltinOperator_LEAKY_RELU) {
    const auto *opts = op->builtin_options_as_LeakyReluOptions();
    float alpha = opts->alpha();
    output = F_->createLeakyRELU(opInfo.name, outTy, input, alpha);
  } else if (opCode == tflite::BuiltinOperator_SQUARE) {
    output = F_->createSquare(opInfo.name, outTy, input);
  } else if (opCode == tflite::BuiltinOperator_ABS) {
    output = F_->createAbs(opInfo.name, outTy, input);
  } else if (opCode == tflite::BuiltinOperator_NEG) {
    output = F_->createNeg(opInfo.name, outTy, input);
  } else if (opCode == tflite::BuiltinOperator_FLOOR) {
    output = F_->createFloor(opInfo.name, outTy, input);
  } else if (opCode == tflite::BuiltinOperator_CEIL) {
    output = F_->createCeil(opInfo.name, outTy, input);
  } else if (opCode == tflite::BuiltinOperator_ROUND) {
    output = F_->createRound(opInfo.name, outTy, input);
  } else if (opCode == tflite::BuiltinOperator_SQRT) {
    output = F_->createSqrt(opInfo.name, outTy, input);
  } else if (opCode == tflite::BuiltinOperator_RSQRT) {
    output = F_->createRsqrt(opInfo.name, outTy, input);
  } else if (opCode == tflite::BuiltinOperator_SIN) {
    output = F_->createSin(opInfo.name, outTy, input);
  } else if (opCode == tflite::BuiltinOperator_COS) {
    output = F_->createCos(opInfo.name, outTy, input);
  } else if (opCode == tflite::BuiltinOperator_LOGICAL_NOT) {
    output = F_->createNot(opInfo.name, input);
  } else if (opCode == tflite::BuiltinOperator_QUANTIZE) {
    output = F_->createQuantize(opInfo.name, input, outTy);
  } else if (opCode == tflite::BuiltinOperator_DEQUANTIZE) {
    output = F_->createDequantize(opInfo.name, input, outTy);
  } else {
    RETURN_ERR(opErrMsg(opInfo, "Unsupported unary arithmetic operator!"));
  }
  return setOutputNodeValue(op, output);
}

Error TFLiteModelLoader::loadBinaryArithmetic(const tflite::Operator *op,
                                              const OperatorInfo &opInfo) {
  NodeValue LHS;
  ASSIGN_VALUE_OR_RETURN_ERR(LHS, getInputNodeValue(op, 0));
  NodeValue RHS;
  ASSIGN_VALUE_OR_RETURN_ERR(RHS, getInputNodeValue(op, 1));
  TypeRef outTy;
  ASSIGN_VALUE_OR_RETURN_ERR(outTy, getOutputType(op, 0));

  // LHS operand broadcasting.
  if (LHS.dims().size() < RHS.dims().size()) {
    unsigned_t axis = RHS.dims().size() - LHS.dims().size();
    LHS =
        F_->createBroadcast(opInfo.name + ".Broadcast", LHS, RHS.dims(), axis);
  }

  // RHS operand broadcasting.
  if (RHS.dims().size() < LHS.dims().size()) {
    unsigned_t axis = LHS.dims().size() - RHS.dims().size();
    RHS =
        F_->createBroadcast(opInfo.name + ".Broadcast", RHS, LHS.dims(), axis);
  }

  auto opCode = opInfo.code;
  NodeValue output;
  if (opCode == tflite::BuiltinOperator_ADD) {
    const auto *opts = op->builtin_options_as_AddOptions();
    output = F_->createAdd(opInfo.name, outTy, LHS, RHS);
    RETURN_IF_ERR(addActivation(output, opts->fused_activation_function()));
  } else if (opCode == tflite::BuiltinOperator_MUL) {
    const auto *opts = op->builtin_options_as_MulOptions();
    output = F_->createMul(opInfo.name, outTy, LHS, RHS);
    RETURN_IF_ERR(addActivation(output, opts->fused_activation_function()));
  } else if (opCode == tflite::BuiltinOperator_SUB) {
    const auto *opts = op->builtin_options_as_SubOptions();
    output = F_->createSub(opInfo.name, outTy, LHS, RHS);
    RETURN_IF_ERR(addActivation(output, opts->fused_activation_function()));
  } else if (opCode == tflite::BuiltinOperator_DIV) {
    const auto *opts = op->builtin_options_as_DivOptions();
    output = F_->createDiv(opInfo.name, outTy, LHS, RHS);
    RETURN_IF_ERR(addActivation(output, opts->fused_activation_function()));
  } else if (opCode == tflite::BuiltinOperator_POW) {
    output = F_->createPow(opInfo.name, outTy, LHS, RHS);
  } else if (opCode == tflite::BuiltinOperator_PRELU) {
    NodeValue slope =
        F_->createReshape(opInfo.name + ".reshape", RHS, outTy->dims());
    output = F_->createPRELU(opInfo.name, LHS, slope, outTy);
  } else if (opCode == tflite::BuiltinOperator_MAXIMUM) {
    output = F_->createMax(opInfo.name, outTy, LHS, RHS);
  } else if (opCode == tflite::BuiltinOperator_MINIMUM) {
    output = F_->createMin(opInfo.name, outTy, LHS, RHS);
  } else if (opCode == tflite::BuiltinOperator_EQUAL) {
    output = F_->createCmpEQ(opInfo.name, LHS, RHS);
  } else if (opCode == tflite::BuiltinOperator_NOT_EQUAL) {
    output = F_->createCmpNEQ(opInfo.name, LHS, RHS);
  } else if (opCode == tflite::BuiltinOperator_LESS) {
    output = F_->createCmpLT(opInfo.name, LHS, RHS);
  } else if (opCode == tflite::BuiltinOperator_LESS_EQUAL) {
    output = F_->createCmpLTE(opInfo.name, LHS, RHS);
  } else if (opCode == tflite::BuiltinOperator_GREATER) {
    output = F_->createCmpGT(opInfo.name, LHS, RHS);
  } else if (opCode == tflite::BuiltinOperator_GREATER_EQUAL) {
    output = F_->createCmpGTE(opInfo.name, LHS, RHS);
  } else if (opCode == tflite::BuiltinOperator_LOGICAL_AND) {
    output = F_->createAnd(opInfo.name, LHS, RHS);
  } else if (opCode == tflite::BuiltinOperator_LOGICAL_OR) {
    output = F_->createOr(opInfo.name, LHS, RHS);
  } else {
    RETURN_ERR(opErrMsg(opInfo, "Unsupported binary arithmetic operator!"));
  }
  return setOutputNodeValue(op, output);
}

Error TFLiteModelLoader::loadPool2D(const tflite::Operator *op,
                                    const OperatorInfo &opInfo) {
  const auto *opts = op->builtin_options_as_Pool2DOptions();
  NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(input, getInputNodeValue(op, 0));
  TypeRef outTy;
  ASSIGN_VALUE_OR_RETURN_ERR(outTy, getOutputType(op, 0));

  ShapeNHWC inputShape = ShapeNHWC(input.dims());
  ShapeNHWC outputShape = ShapeNHWC(outTy->dims());

  std::vector<unsigned_t> kernels = {
      static_cast<unsigned_t>(opts->filter_height()),
      static_cast<unsigned_t>(opts->filter_width())};

  std::vector<unsigned_t> strides = {
      static_cast<unsigned_t>(opts->stride_h()),
      static_cast<unsigned_t>(opts->stride_w()),
  };

  std::vector<unsigned_t> pads;
  if (opts->padding() == tflite::Padding::Padding_VALID) {
    pads = {0, 0, 0, 0};
  } else if (opts->padding() == tflite::Padding::Padding_SAME) {
    auto padsTB =
        getSamePads(inputShape.h, outputShape.h, kernels[0], strides[0]);
    auto padsLR =
        getSamePads(inputShape.w, outputShape.w, kernels[1], strides[1]);
    pads = {padsTB.first, padsLR.first, padsTB.second, padsLR.second};
  } else {
    RETURN_ERR(opErrMsg(opInfo, "Padding parameter invalid!"));
  }

  auto opCode = opInfo.code;
  NodeValue output;
  if (opCode == tflite::BuiltinOperator_AVERAGE_POOL_2D) {
    // TFLite AvgPool does NOT include padded regions when normalizing.
    auto *node = F_->createAvgPool(opInfo.name, input, kernels, strides, pads,
                                   ConvolutionLayout::NHWC,
                                   /* countIncludePads */ false);
    output = node->getResult();
  } else if (opCode == tflite::BuiltinOperator_MAX_POOL_2D) {
    auto *node = F_->createMaxPool(opInfo.name, input, kernels, strides, pads);
    output = node->getResult();
  } else {
    RETURN_ERR(opErrMsg(opInfo, "Unsupported Pool2D operator!"));
  }

  RETURN_IF_ERR(addActivation(output, opts->fused_activation_function()));
  return setOutputNodeValue(op, output);
}

Error TFLiteModelLoader::loadConcat(const tflite::Operator *op,
                                    const OperatorInfo &opInfo) {
  const auto *opts = op->builtin_options_as_ConcatenationOptions();
  TypeRef outTy;
  ASSIGN_VALUE_OR_RETURN_ERR(outTy, getOutputType(op, 0));

  const size_t numInputs = op->inputs()->size();
  llvm::SmallVector<NodeValue, 4> inputs;
  inputs.reserve(numInputs);
  for (size_t idx = 0; idx < numInputs; ++idx) {
    NodeValue input;
    ASSIGN_VALUE_OR_RETURN_ERR(input, getInputNodeValue(op, idx));
    inputs.push_back(input);
  }

  // If this node is quantized and there is a mismatch between the input and
  // output quantization parameters then we pull Rescale nodes from the inputs
  // to match the output quantization parameters.
  if (outTy->isQuantizedType()) {
    for (size_t idx = 0; idx < numInputs; ++idx) {
      NodeValue input = inputs[idx];
      TypeRef inpTy = input.getType();
      RETURN_ERR_IF_NOT(
          inpTy->isQuantizedType(),
          opErrMsg(opInfo, "Mixed precision for input/output not supported!"));
      if ((inpTy->getScale() != outTy->getScale()) ||
          (inpTy->getOffset() != outTy->getOffset())) {
        TypeRef inpTyNew = mod_.uniqueTypeWithNewShape(outTy, inpTy->dims());
        auto *rescaleNode = F_->createRescaleQuantized(
            opInfo.name + ".Rescale" + std::to_string(idx), input, inpTyNew);
        inputs[idx] = rescaleNode->getResult();
      }
    }
  }

  unsigned_t axis;
  ASSIGN_VALUE_OR_RETURN_ERR(
      axis, getPositiveAxis<unsigned_t>(opts->axis(), outTy->dims().size()));

  NodeValue output = F_->createConcat(opInfo.name, inputs, axis, outTy);
  RETURN_IF_ERR(addActivation(output, opts->fused_activation_function()));
  return setOutputNodeValue(op, output);
}

Error TFLiteModelLoader::loadConv2D(const tflite::Operator *op,
                                    const OperatorInfo &opInfo) {
  const auto *opts = op->builtin_options_as_Conv2DOptions();
  NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(input, getInputNodeValue(op, 0));
  NodeValue filter;
  ASSIGN_VALUE_OR_RETURN_ERR(filter, getInputNodeValue(op, 1));
  NodeValue bias;
  ASSIGN_VALUE_OR_RETURN_ERR(bias, getInputNodeValue(op, 2));
  TypeRef outTy;
  ASSIGN_VALUE_OR_RETURN_ERR(outTy, getOutputType(op, 0));

  ShapeNHWC inputShape = ShapeNHWC(input.dims());
  ShapeNHWC filterShape = ShapeNHWC(filter.dims());
  ShapeNHWC outputShape = ShapeNHWC(outTy->dims());

  std::vector<unsigned_t> kernels = {static_cast<unsigned_t>(filterShape.h),
                                     static_cast<unsigned_t>(filterShape.w)};

  std::vector<unsigned_t> strides = {
      static_cast<unsigned_t>(opts->stride_h()),
      static_cast<unsigned_t>(opts->stride_w()),
  };

  std::vector<unsigned_t> dilations = {
      static_cast<unsigned_t>(opts->dilation_h_factor()),
      static_cast<unsigned_t>(opts->dilation_w_factor()),
  };

  std::vector<unsigned_t> pads;
  if (opts->padding() == tflite::Padding::Padding_VALID) {
    pads = {0, 0, 0, 0};
  } else if (opts->padding() == tflite::Padding::Padding_SAME) {
    auto padsTB = getSamePads(inputShape.h, outputShape.h, kernels[0],
                              strides[0], dilations[0]);
    auto padsLR = getSamePads(inputShape.w, outputShape.w, kernels[1],
                              strides[1], dilations[1]);
    pads = {padsTB.first, padsLR.first, padsTB.second, padsLR.second};
  } else {
    RETURN_ERR(opErrMsg(opInfo, "Padding parameter invalid!"));
  }

  // TODO: Remove this when Conv2D supports multiple dilations.
  RETURN_ERR_IF_NOT(dilations[0] == dilations[1],
                    opErrMsg(opInfo, "Non square dilation not supported!"));
  unsigned_t dilation = dilations[0];

  // There are TensorFlowLite models which have only the weights quantized
  // to INT8 (the rest of the operands being FLOAT32). Since Glow does not
  // support mixed precision operation we dequantize the weights.
  if (input.getType()->isFPType() && filter.getType()->isQuantizedType() &&
      bias.getType()->isFPType() && outTy->isFPType()) {
    filter = F_->createDequantize(opInfo.name + ".Dequantize", filter,
                                  outTy->getElementType());
  }

  // Check whether this operator is quantized per axis.
  bool isPerAxisQuantized;
  Constant *filterScales = nullptr;
  Constant *filterOffsets = nullptr;
  Constant *biasScales = nullptr;
  Constant *biasOffsets = nullptr;
  ASSIGN_VALUE_OR_RETURN_ERR(isPerAxisQuantized,
                             isConv2DPerAxisQuantized(op, opInfo, filterScales,
                                                      filterOffsets, biasScales,
                                                      biasOffsets));

  // Create convolution node.
  NodeValue output;
  if (isPerAxisQuantized) {
    // Check that filter and bias are constants.
    RETURN_ERR_IF_NOT(llvm::dyn_cast<Constant>(filter.getNode()),
                      opErrMsg(opInfo, "Filter must be constant!"));
    RETURN_ERR_IF_NOT(llvm::dyn_cast<Constant>(bias.getNode()),
                      opErrMsg(opInfo, "Bias must be constant!"));
    // Create ChannelwiseQuantizedConvolution node.
    output = F_->createChannelwiseQuantizedConv(
        opInfo.name, input, filter, bias, filterScales, filterOffsets,
        biasScales, biasOffsets, outTy, kernels, strides, pads, /* group */ 1,
        dilation, /* quantizeFilter */ false, /* quantizeBias */ false);
  } else {
    // Check bias quantization parameters.
    RETURN_IF_ERR(checkBiasQuantizationParams(mod_, input, filter, bias));
    // Create Convolution node.
    output = F_->createConv(opInfo.name, input, filter, bias, outTy, kernels,
                            strides, pads, /* group */ 1, dilation);
  }

  RETURN_IF_ERR(addActivation(output, opts->fused_activation_function()));
  return setOutputNodeValue(op, output);
}

Error TFLiteModelLoader::loadDepthwiseConv2D(const tflite::Operator *op,
                                             const OperatorInfo &opInfo) {
  const auto *opts = op->builtin_options_as_DepthwiseConv2DOptions();
  NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(input, getInputNodeValue(op, 0));
  NodeValue filter;
  ASSIGN_VALUE_OR_RETURN_ERR(filter, getInputNodeValue(op, 1));
  NodeValue bias;
  ASSIGN_VALUE_OR_RETURN_ERR(bias, getInputNodeValue(op, 2));
  TypeRef outTy;
  ASSIGN_VALUE_OR_RETURN_ERR(outTy, getOutputType(op, 0));

  ShapeNHWC inputShape = ShapeNHWC(input.dims());
  ShapeNHWC filterShape = ShapeNHWC(filter.dims());
  ShapeNHWC outputShape = ShapeNHWC(outTy->dims());

  std::vector<unsigned_t> kernels = {static_cast<unsigned_t>(filterShape.h),
                                     static_cast<unsigned_t>(filterShape.w)};

  std::vector<unsigned_t> strides = {
      static_cast<unsigned_t>(opts->stride_h()),
      static_cast<unsigned_t>(opts->stride_w()),
  };

  std::vector<unsigned_t> dilations = {1, 1};
  if (opInfo.version >= 2) {
    dilations = {static_cast<unsigned_t>(opts->dilation_h_factor()),
                 static_cast<unsigned_t>(opts->dilation_w_factor())};
  }

  std::vector<unsigned_t> pads;
  if (opts->padding() == tflite::Padding::Padding_VALID) {
    pads = {0, 0, 0, 0};
  } else if (opts->padding() == tflite::Padding::Padding_SAME) {
    auto padsTB = getSamePads(inputShape.h, outputShape.h, kernels[0],
                              strides[0], dilations[0]);
    auto padsLR = getSamePads(inputShape.w, outputShape.w, kernels[1],
                              strides[1], dilations[1]);
    pads = {padsTB.first, padsLR.first, padsTB.second, padsLR.second};
  } else {
    RETURN_ERR(opErrMsg(opInfo, "Padding parameter invalid!"));
  }

  // TODO: Remove this when Conv2D supports multiple dilations.
  RETURN_ERR_IF_NOT(dilations[0] == dilations[1],
                    opErrMsg(opInfo, "Non square dilation not supported!"));
  unsigned_t dilation = dilations[0];

  // Convolution group is inputChannels / filterChannels = inputChannels.
  unsigned_t group = input.dims().back();

  // There are TensorFlowLite models which have only the weights quantized
  // to INT8 (the rest of the operands being FLOAT32). Since Glow does not
  // support mixed precision operation we dequantize the weights.
  if (input.getType()->isFPType() && filter.getType()->isQuantizedType() &&
      bias.getType()->isFPType() && outTy->isFPType()) {
    filter = F_->createDequantize(opInfo.name + ".Dequantize", filter,
                                  outTy->getElementType());
  }

  // Check whether this operator is quantized per axis.
  bool isPerAxisQuantized;
  Constant *filterScales = nullptr;
  Constant *filterOffsets = nullptr;
  Constant *biasScales = nullptr;
  Constant *biasOffsets = nullptr;
  ASSIGN_VALUE_OR_RETURN_ERR(isPerAxisQuantized,
                             isConv2DPerAxisQuantized(op, opInfo, filterScales,
                                                      filterOffsets, biasScales,
                                                      biasOffsets));

  // Transpose filter from CHWN to NHWC in-place without using a Reshape
  // node because further down the ChannelwiseQuantizedConvolution requires
  // the filter to be a Constant.
  RETURN_ERR_IF_NOT(filter.dims().size() == 4,
                    opErrMsg(opInfo, "Filter should be 4D!"));
  if (isPerAxisQuantized) {
    Constant *filterC = llvm::dyn_cast<Constant>(filter.getNode());
    RETURN_ERR_IF_NOT(filterC, opErrMsg(opInfo, "Filter must be constant!"));
    TypeRef filterTy = filterC->getType();
    auto filterDims = filterTy->dims();
    TypeRef newFilterTy = mod_.uniqueTypeWithNewShape(
        filterTy, {filterDims[3], filterDims[1], filterDims[2], filterDims[0]});
    Tensor newFilterT = Tensor(newFilterTy);
    filterC->getPayload().transpose(&newFilterT, {3, 1, 2, 0});
    Constant *newFilterC = mod_.createConstant(
        filterC->getName().str() + ".Reshape", std::move(newFilterT), "NHWC");
    filter = newFilterC->getOutput();
  } else {
    filter = F_->createTranspose(opInfo.name + ".Transpose", filter,
                                 {3, 1, 2, 0}, "NHWC");
  }
  RETURN_ERR_IF_NOT(filter.dims().back() == 1,
                    opErrMsg(opInfo, "Filter should have 1 channel!"));

  // Create convolution node.
  NodeValue output;
  if (isPerAxisQuantized) {
    // Check that filter and bias are constants.
    RETURN_ERR_IF_NOT(llvm::dyn_cast<Constant>(filter.getNode()),
                      opErrMsg(opInfo, "Filter must be constant!"));
    RETURN_ERR_IF_NOT(llvm::dyn_cast<Constant>(bias.getNode()),
                      opErrMsg(opInfo, "Bias must be constant!"));
    // Create ChannelwiseQuantizedConvolution node.
    output = F_->createChannelwiseQuantizedConv(
        opInfo.name, input, filter, bias, filterScales, filterOffsets,
        biasScales, biasOffsets, outTy, kernels, strides, pads, group, dilation,
        /* quantizeFilter */ false, /* quantizeBias */ false);
  } else {
    // Check bias quantization parameters.
    RETURN_IF_ERR(checkBiasQuantizationParams(mod_, input, filter, bias));
    // Create Convolution node.
    output = F_->createConv(opInfo.name, input, filter, bias, outTy, kernels,
                            strides, pads, group, dilation);
  }

  RETURN_IF_ERR(addActivation(output, opts->fused_activation_function()));
  return setOutputNodeValue(op, output);
}

Error TFLiteModelLoader::loadFullyConnected(const tflite::Operator *op,
                                            const OperatorInfo &opInfo) {
  const auto *opts = op->builtin_options_as_FullyConnectedOptions();
  NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(input, getInputNodeValue(op, 0));
  NodeValue weights;
  ASSIGN_VALUE_OR_RETURN_ERR(weights, getInputNodeValue(op, 1));
  NodeValue bias;
  ASSIGN_VALUE_OR_RETURN_ERR(bias, getInputNodeValue(op, 2));
  TypeRef outTy;
  ASSIGN_VALUE_OR_RETURN_ERR(outTy, getOutputType(op, 0));

  RETURN_IF_ERR(checkBiasQuantizationParams(mod_, input, weights, bias));

  // There are TensorFlowLite models which have only the weights quantized
  // to INT8 (the rest of the operands being FLOAT32). Since Glow does not
  // support mixed precision operation we dequantize the weights.
  if (input.getType()->isFPType() && weights.getType()->isQuantizedType() &&
      bias.getType()->isFPType() && outTy->isFPType()) {
    weights = F_->createDequantize(opInfo.name + ".Dequantize", weights,
                                   outTy->getElementType());
  }

  if (opInfo.version >= 2) {
    RETURN_ERR_IF_NOT(
        opts->weights_format() ==
            tflite::FullyConnectedOptionsWeightsFormat_DEFAULT,
        opErrMsg(opInfo, "Only default weights format is supported!"));
  }

  bool keepDims = false;
  if (opInfo.version >= 5) {
    keepDims = opts->keep_num_dims();
  }

  // Transpose weights.
  RETURN_ERR_IF_NOT(weights.dims().size() == 2,
                    opErrMsg(opInfo, "Weights should be 2D!"));
  weights = F_->createTranspose(opInfo.name + ".Transpose", weights, {1, 0});

  // For an input with shape [D(0), D(1), ... , D(N-1)] if:
  // keep_num_dims is FALSE then we flatten the input into:
  //   [D(0), D(1) x D(2) x ... x D(N-1)] (axis = 1).
  // keep_num_dims options is TRUE then we flatten the input into:
  //   [D(0) x D(1) x ... x D(N-2), D(N-1)] (axis = N-1).
  unsigned_t axis = keepDims ? (input.dims().size() - 1) : 1;
  NodeValue output =
      F_->createFullyConnected(opInfo.name, input, weights, bias, outTy, axis);
  RETURN_IF_ERR(addActivation(output, opts->fused_activation_function()));

  // Expand output dims if necessary.
  if (keepDims) {
    std::vector<dim_t> outputDims = input.dims();
    outputDims.back() = output.dims().back();
    output = F_->createReshape(opInfo.name + ".Reshape", output, outputDims);
  }
  return setOutputNodeValue(op, output);
}

Error TFLiteModelLoader::loadReshape(const tflite::Operator *op,
                                     const OperatorInfo &opInfo) {
  NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(input, getInputNodeValue(op, 0));
  TypeRef outTy;
  ASSIGN_VALUE_OR_RETURN_ERR(outTy, getOutputType(op, 0));
  // Note: The Reshape node has a second input operand which provides
  // the new shape but the documentation states that is should be ignored
  // and the 'new_shape' attribute should be used instead. Moreover, in
  // this case we are not using not even the 'new_shape' attribute because
  // we have the output type directly available. We are using this logic
  // also for loading other operators: Squeeze, ExpandDims.
  NodeValue output = F_->createReshape(opInfo.name, input, outTy->dims());
  return setOutputNodeValue(op, output);
}

Error TFLiteModelLoader::loadSoftmax(const tflite::Operator *op,
                                     const OperatorInfo &opInfo) {
  const auto *opts = op->builtin_options_as_SoftmaxOptions();
  NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(input, getInputNodeValue(op, 0));
  TypeRef outTy;
  ASSIGN_VALUE_OR_RETURN_ERR(outTy, getOutputType(op, 0));

  RETURN_ERR_IF_NOT(input.dims().size() >= 2,
                    opErrMsg(opInfo, "Input rank must be >= 2!"));
  float beta = opts->beta();

  // Create a constant to store labels to be used in SoftMaxGradNode.
  auto selected =
      mod_.createConstant(ElemKind::Int64ITy, {input.dims()[0], 1}, "selected");

  NodeValue output;
  if (tfliteFloatSoftmaxOpt) {
    // We dequantize the input if it is quantized type.
    if (input.getType()->isQuantizedType()) {
      input = F_->createDequantize(opInfo.name + ".Dequantize", input,
                                   ElemKind::FloatTy);
    }

    // Create float Softmax regardless of the type defined in the model.
    output = F_->createSoftMax(opInfo.name, input, selected, nullptr, beta);

    // If target output type is quantized we quantize the float output of the
    // Softmax but only if it is not an output placeholder in which case we
    // allow the output placeholder to remain float even though it was defined
    // as quantized in the original model.
    bool isFinal;
    ASSIGN_VALUE_OR_RETURN_ERR(isFinal, isOperatorOutputFinalTensor(op, 0));
    if (outTy->isQuantizedType() && !isFinal) {
      output = F_->createQuantize(opInfo.name + ".Quantize", output, outTy);
    }
  } else {
    output = F_->createSoftMax(opInfo.name, input, selected, outTy, beta);
  }
  return setOutputNodeValue(op, output);
}

Error TFLiteModelLoader::loadPad(const tflite::Operator *op,
                                 const OperatorInfo &opInfo) {
  NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(input, getInputNodeValue(op, 0));
  NodeValue pads;
  ASSIGN_VALUE_OR_RETURN_ERR(pads, getInputNodeValue(op, 1));
  TypeRef outTy;
  ASSIGN_VALUE_OR_RETURN_ERR(outTy, getOutputType(op, 0));

  // Validate paddings shape.
  auto numDims = input.dims().size();
  RETURN_ERR_IF_NOT(pads.dims().size() == 2,
                    opErrMsg(opInfo, "Paddings should be 2D!"));
  RETURN_ERR_IF_NOT(pads.dims()[0] == numDims,
                    opErrMsg(opInfo, "Paddings 1st dimension should match the "
                                     "input rank!"));
  RETURN_ERR_IF_NOT(pads.dims()[1] == 2,
                    opErrMsg(opInfo, "Paddings 2nd dimensions should be 2!"));

  // TFLite paddings are stored as start(D1),stop(D1),start(D2),stop(D2),etc.
  Constant *padsC = llvm::dyn_cast<Constant>(pads.getNode());
  RETURN_ERR_IF_NOT(padsC,
                    opErrMsg(opInfo, "Non constant 'paddings' not supported!"));
  RETURN_ERR_IF_NOT(padsC->getType()->getElementType() == ElemKind::Int32ITy,
                    opErrMsg(opInfo, "Paddings should have INT32 type!"));
  auto padsH = padsC->getPayload().getHandle<int32_t>();
  std::vector<int> padsVec(padsH.size());
  for (dim_t dim = 0; dim < numDims; ++dim) {
    auto padStart = padsH.at({dim, 0});
    auto padStop = padsH.at({dim, 1});
    RETURN_ERR_IF_NOT((padStart >= 0) && (padStop >= 0),
                      opErrMsg(opInfo, "Invalid negative padding value!"));
    padsVec[0 * numDims + dim] = static_cast<int>(padStart);
    padsVec[1 * numDims + dim] = static_cast<int>(padStop);
  }

  NodeValue output = F_->createPad(opInfo.name, input, outTy,
                                   PaddingMode::CONSTANT, padsVec, 0.f);
  return setOutputNodeValue(op, output);
}

Error TFLiteModelLoader::loadTranspose(const tflite::Operator *op,
                                       const OperatorInfo &opInfo) {
  NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(input, getInputNodeValue(op, 0));
  NodeValue perm;
  ASSIGN_VALUE_OR_RETURN_ERR(perm, getInputNodeValue(op, 1));

  Constant *permC = llvm::dyn_cast<Constant>(perm.getNode());
  RETURN_ERR_IF_NOT(
      permC, opErrMsg(opInfo, "Non constant permutation not supported!"));
  RETURN_ERR_IF_NOT(permC->getType()->getElementType() == ElemKind::Int32ITy,
                    opErrMsg(opInfo, "Permutation should have INT32 type!"));
  auto permH = permC->getPayload().getHandle<int32_t>();

  std::vector<unsigned_t> shuffle;
  for (size_t idx = 0; idx < permH.size(); ++idx) {
    int32_t dim = permH.raw(idx);
    RETURN_ERR_IF_NOT(dim >= 0, opErrMsg(opInfo, "Invalid permutation value!"));
    shuffle.push_back(static_cast<unsigned_t>(dim));
  }

  NodeValue output = F_->createTranspose(opInfo.name, input, shuffle);
  return setOutputNodeValue(op, output);
}

Error TFLiteModelLoader::loadReduce(const tflite::Operator *op,
                                    const OperatorInfo &opInfo) {
  const auto *opts = op->builtin_options_as_ReducerOptions();
  NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(input, getInputNodeValue(op, 0));
  NodeValue axes;
  ASSIGN_VALUE_OR_RETURN_ERR(axes, getInputNodeValue(op, 1));
  TypeRef outTy;
  ASSIGN_VALUE_OR_RETURN_ERR(outTy, getOutputType(op, 0));

  std::vector<unsigned_t> axesVal;
  ASSIGN_VALUE_OR_RETURN_ERR(axesVal,
                             loadAxes<unsigned_t>(opInfo, axes, input));

  bool keepDims = opts->keep_dims();

  // Currently the Glow reduce operators do not support multiple axes so we
  // create chained reduce operators with single axis.
  // TODO: When Glow supports reduce operators with multiple axes remove this!
  auto opCode = opInfo.code;
  NodeValue output = input;
  for (size_t idx = 0, end = axesVal.size(); idx < end; ++idx) {
    // Current axis value.
    unsigned_t axisVal = axesVal[idx];
    if (!keepDims) {
      axisVal = axisVal - idx;
    }
    // Current output type.
    ShapeVector outDimsCurr(output.dims().begin(), output.dims().end());
    outDimsCurr.erase(outDimsCurr.begin() + axisVal);
    auto outTypeCurr = mod_.uniqueTypeWithNewShape(outTy, outDimsCurr);
    // Create reduce operator.
    if (opCode == tflite::BuiltinOperator_MEAN) {
      output = F_->createBatchedReduceMean(opInfo.name, outTypeCurr, output,
                                           {axisVal});
      // The BatchedReduceMean reduces the output dimension and hence we expand
      // the output dimensions if keepDims is true.
      if (keepDims) {
        output =
            F_->createExpandDims(opInfo.name + ".Expand", output, {axisVal});
      }
    } else {
      RETURN_ERR(opErrMsg(opInfo, "Unsupported Reduce operator!"));
    }
  }
  return setOutputNodeValue(op, output);
}

Error TFLiteModelLoader::loadSplit(const tflite::Operator *op,
                                   const OperatorInfo &opInfo) {
  const auto *opts = op->builtin_options_as_SplitOptions();
  NodeValue axis;
  ASSIGN_VALUE_OR_RETURN_ERR(axis, getInputNodeValue(op, 0));
  NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(input, getInputNodeValue(op, 1));

  unsigned_t axisVal;
  ASSIGN_VALUE_OR_RETURN_ERR(axisVal,
                             loadAxis<unsigned_t>(opInfo, axis, input));

  unsigned_t numSplits = static_cast<unsigned_t>(opts->num_splits());
  RETURN_ERR_IF_NOT(
      input.dims()[axisVal] % numSplits == 0,
      opErrMsg(
          opInfo,
          "Input dimension should be divisible by 'num_splits' along axis!"));

  std::vector<SliceNode *> outputNodes;
  F_->createSplit(opInfo.name, input, numSplits, axisVal, {}, outputNodes);
  std::vector<NodeValue> outputNodeValues(outputNodes.size(), nullptr);
  for (size_t idx = 0, end = outputNodes.size(); idx < end; ++idx) {
    outputNodeValues[idx] = outputNodes[idx]->getResult();
  }
  return setOutputNodeValues(op, outputNodeValues);
}

Error TFLiteModelLoader::loadArg(const tflite::Operator *op,
                                 const OperatorInfo &opInfo) {
  NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(input, getInputNodeValue(op, 0));
  NodeValue axis;
  ASSIGN_VALUE_OR_RETURN_ERR(axis, getInputNodeValue(op, 1));
  TypeRef outTy;
  ASSIGN_VALUE_OR_RETURN_ERR(outTy, getOutputType(op, 0));

  unsigned_t axisVal;
  ASSIGN_VALUE_OR_RETURN_ERR(axisVal,
                             loadAxis<unsigned_t>(opInfo, axis, input));

  auto opCode = opInfo.code;
  NodeValue output = nullptr;
  if (opCode == tflite::BuiltinOperator_ARG_MAX) {
    output = F_->createArgMax(opInfo.name, input, axisVal, /* keepDims */ false,
                              outTy->getElementType());
  } else if (opCode == tflite::BuiltinOperator_ARG_MIN) {
    output = F_->createArgMin(opInfo.name, input, axisVal, /* keepDims */ false,
                              outTy->getElementType());
  } else {
    RETURN_ERR(opErrMsg(opInfo, "Unsupported Arg operator!"));
  }
  return setOutputNodeValue(op, output);
}

Error TFLiteModelLoader::loadSlice(const tflite::Operator *op,
                                   const OperatorInfo &opInfo) {
  NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(input, getInputNodeValue(op, 0));
  NodeValue begin;
  ASSIGN_VALUE_OR_RETURN_ERR(begin, getInputNodeValue(op, 1));
  TypeRef outTy;
  ASSIGN_VALUE_OR_RETURN_ERR(outTy, getOutputType(op, 0));
  // Note: Slice has a third input operand 'size' which provides the size of
  // the output slice. We derive here the output slice size based on outTy.

  Constant *beginC = llvm::dyn_cast<Constant>(begin.getNode());
  RETURN_ERR_IF_NOT(beginC,
                    opErrMsg(opInfo, "Non constant begin not supported!"));
  RETURN_ERR_IF_NOT(beginC->getType()->getElementType() == ElemKind::Int32ITy,
                    opErrMsg(opInfo, "Begin should have INT32 type!"));
  auto beginH = beginC->getPayload().getHandle<int32_t>();

  std::vector<dim_t> start;
  for (size_t idx = 0; idx < beginH.size(); ++idx) {
    int32_t dimStart = beginH.raw(idx);
    RETURN_ERR_IF_NOT(dimStart >= 0, opErrMsg(opInfo, "Invalid begin value!"));
    start.push_back(static_cast<dim_t>(dimStart));
  }

  NodeValue output = F_->createSlice(opInfo.name, input, start, outTy);
  return setOutputNodeValue(op, output);
}

Error TFLiteModelLoader::loadTile(const tflite::Operator *op,
                                  const OperatorInfo &opInfo) {
  NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(input, getInputNodeValue(op, 0));
  NodeValue multiples;
  ASSIGN_VALUE_OR_RETURN_ERR(multiples, getInputNodeValue(op, 1));

  auto numDims = input.getType()->dims().size();
  std::vector<unsigned_t> numTiles;
  ASSIGN_VALUE_OR_RETURN_ERR(numTiles,
                             loadArray<unsigned_t>(opInfo, multiples));
  RETURN_ERR_IF_NOT(numTiles.size() == numDims,
                    opErrMsg(opInfo, "Input operand 'multiples' length should "
                                     "match the number of input dimensions!"));

  NodeValue output = input;
  for (unsigned_t axis = 0; axis < numDims; ++axis) {
    unsigned_t tiles = numTiles[axis];
    if (tiles != 1) {
      output = F_->createTile(opInfo.name + std::to_string(axis), output, tiles,
                              axis);
    }
  }
  return setOutputNodeValue(op, output);
}

Error TFLiteModelLoader::loadPack(const tflite::Operator *op,
                                  const OperatorInfo &opInfo) {
  const auto *opts = op->builtin_options_as_PackOptions();
  TypeRef outTy;
  ASSIGN_VALUE_OR_RETURN_ERR(outTy, getOutputType(op, 0));

  const size_t numInputs = op->inputs()->size();
  RETURN_ERR_IF_NOT(numInputs == opts->values_count(),
                    opErrMsg(opInfo, "Attribute 'values_count' does not match "
                                     "the number of operator inputs!"));
  llvm::SmallVector<NodeValue, 4> inputs;
  inputs.reserve(numInputs);
  for (size_t idx = 0; idx < numInputs; ++idx) {
    NodeValue input;
    ASSIGN_VALUE_OR_RETURN_ERR(input, getInputNodeValue(op, idx));
    inputs.push_back(input);
  }

  unsigned_t axis;
  ASSIGN_VALUE_OR_RETURN_ERR(
      axis, getPositiveAxis<unsigned_t>(opts->axis(), outTy->dims().size()));

  // Validate that all inputs have same shape.
  for (size_t idx = 0; idx < numInputs; ++idx) {
    RETURN_ERR_IF_NOT(
        inputs[idx].getType()->isEqual(inputs[0].getType()),
        opErrMsg(opInfo, "Operator inputs do not have same type/shape!"));
  }

  // Reshape all inputs from [D0,D1,...,DN] to [D0,D1,...,1,...,DN] where the
  // the singular dimension 1 is on the axis position.
  std::vector<dim_t> inputDimsReshaped = inputs[0].dims();
  inputDimsReshaped.insert(inputDimsReshaped.begin() + axis, 1);
  for (size_t idx = 0; idx < numInputs; ++idx) {
    inputs[idx] =
        F_->createReshape(opInfo.name + ".Reshape" + std::to_string(idx),
                          inputs[idx], inputDimsReshaped);
  }

  // Concatenate all inputs along axis.
  NodeValue output =
      F_->createConcat(opInfo.name + ".Concat", inputs, axis, outTy);
  return setOutputNodeValue(op, output);
}

Error TFLiteModelLoader::loadUnpack(const tflite::Operator *op,
                                    const OperatorInfo &opInfo) {
  const auto *opts = op->builtin_options_as_UnpackOptions();
  NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(input, getInputNodeValue(op, 0));
  TypeRef outTy;
  ASSIGN_VALUE_OR_RETURN_ERR(outTy, getOutputType(op, 0));

  unsigned_t axis;
  ASSIGN_VALUE_OR_RETURN_ERR(axis,
                             getPositiveAxis<unsigned_t>(opts->axis(), input));

  unsigned_t num = static_cast<unsigned_t>(opts->num());
  RETURN_ERR_IF_NOT(
      num == input.dims()[axis],
      opErrMsg(opInfo,
               "Attribute 'num' should be equal to input size along axis!"));

  // Split input.
  std::vector<SliceNode *> outputNodes;
  F_->createSplit(opInfo.name, input, num, axis, {}, outputNodes);

  // Reshape outputs.
  std::vector<NodeValue> outputNodeValues(outputNodes.size(), nullptr);
  for (size_t idx = 0, end = outputNodes.size(); idx < end; ++idx) {
    outputNodeValues[idx] = outputNodes[idx]->getResult();
    outputNodeValues[idx] =
        F_->createReshape(opInfo.name + ".Reshape" + std::to_string(idx),
                          outputNodeValues[idx], outTy->dims());
  }
  return setOutputNodeValues(op, outputNodeValues);
}

TFLiteModelLoader::TFLiteModelLoader(const std::string &modelFilename,
                                     Function *F)
    : F_(F), mod_(*F->getParent()) {
  auto setup = [&]() -> Error {
    // Read model.
    std::vector<char> modelData;
    ASSIGN_VALUE_OR_RETURN_ERR(model_, readModel(modelData, modelFilename));

    // TODO: Verify model integrity using flatbuffers::Verifier class.

    // Get model info.
    modelVersion_ = model_->version();
    modelDescription_ = model_->description()->str();

    // Get model graph.
    const auto *modelGraphs = model_->subgraphs();
    RETURN_ERR_IF_NOT(
        modelGraphs->size() == 1,
        "TensorFlowLite: Only one model subgraph is currently supported!");
    graph_ = (*modelGraphs)[0];

    // Initialize graph node values.
    initializeNodeValues();

    // Load graph input placeholders.
    RETURN_IF_ERR(loadInputPlaceholders());

    // Load graph constants.
    RETURN_IF_ERR(loadConstants());

    // Load graph operators.
    RETURN_IF_ERR(loadOperators());

    // Save graph output placeholders.
    RETURN_IF_ERR(saveOutputPlaceholders());

    // Verify function.
    RETURN_ERR_IF_NOT(F_->verify(),
                      "TensorFlowLite: Function verification failed!");

    return Error::success();
  };

  EXIT_ON_ERR(setup());
}
