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

#include "PyTorchModelLoader.h"

#include "glow/Support/Error.h"
#include "glow/Support/Support.h"

#include <torch/csrc/jit/ir.h>

namespace {
/// \returns a corresponding Glow Type for a given PyTorch CompleteTensorType \p
/// ptType.
inline glow::Type ptTypeToGlowType(const at::CompleteTensorType &ptType) {
  // TODO: get correct ElemKind
  std::vector<size_t> dims;
  for (auto &size : ptType.sizes()) {
    dims.push_back(size);
  }
  return glow::Type(glow::ElemKind::FloatTy, dims);
}

/// Downcast a double to a float.
inline llvm::Expected<float> to32Bit(double val) {
  RETURN_ERR_IF_NOT(val <= std::numeric_limits<float>::max() ||
                        val >= std::numeric_limits<float>::lowest(),
                    glow::strFormat("Value %f is out of limit.", val));
  return llvm::Expected<float>(static_cast<float>(val));
}

/// Downcast an int64_t to a int32_t.
inline llvm::Expected<int32_t> to32Bit(int64_t val) {
  RETURN_ERR_IF_NOT(val <= std::numeric_limits<int32_t>::max() ||
                        val >= std::numeric_limits<int32_t>::lowest(),
                    glow::strFormat("Value %lld is out of limit.",
                                    static_cast<long long int>(val)));
  return llvm::Expected<int32_t>(static_cast<int32_t>(val));
}

/// Given a Glow Node \p glowNode which must be a Glow Constant, returns a Glow
/// Handle to that Constant's payload.
template <typename T>
llvm::Expected<glow::Handle<T>>
getGlowConstantPayload(const glow::Node *glowNode) {
  RETURN_ERR_IF_NOT(glowNode,
                    glow::strFormat("Expected not null input glow node."));

  const glow::Constant *glowConstant;
  RETURN_ERR_IF_NOT(glowConstant = llvm::dyn_cast<glow::Constant>(glowNode),
                    glow::strFormat("Expected glow node has Constant type."));

  auto type = glowConstant->getElementType();
  if (std::is_same<int32_t, T>::value) {
    RETURN_ERR_IF_NOT(type == glow::ElemKind::Int32ITy,
                      glow::strFormat("Expected Int32ITy type, got %s",
                                      glow::Type::getElementName(type).data()));
  } else if (std::is_same<float, T>::value) {
    RETURN_ERR_IF_NOT(type == glow::ElemKind::FloatTy,
                      glow::strFormat("Expected FloatTy type, got %s",
                                      glow::Type::getElementName(type).data()));
  } else if (std::is_same<bool, T>::value) {
    RETURN_ERR_IF_NOT(type == glow::ElemKind::BoolTy,
                      glow::strFormat("Expected BoolTy type, got %s",
                                      glow::Type::getElementName(type).data()));
  } else {
    RETURN_ERR(glow::strFormat("Unsupported type %s",
                               glow::Type::getElementName(type).data()));
  }
  return llvm::Expected<glow::Handle<T>>(
      glowConstant->getPayload().getHandle<T>());
}

/// Given a Glow Handle \p handle and \p targetSize, will return an std::vector
/// of the elements in the Tensor the Handle manages if the size of that Tensor
/// is targetSize or if the Tensor has only one element in it, will replicate
/// that element targetSize times.
template <typename T, typename OutT>
llvm::Error expandParamIfNeeded(const glow::Handle<T> &handle,
                                size_t targetSize, std::vector<OutT> &out) {
  RETURN_ERR_IF_NOT(
      handle.dims().size() == 1,
      glow::strFormat("Expected a 1d Glow Tensor, got %lu dimensions.",
                      handle.dims().size()));
  RETURN_ERR_IF_NOT(
      handle.size() == 1 || handle.size() == targetSize,
      glow::strFormat(
          "Expected input size to be either 1 or the %lu size, got %lu",
          targetSize, handle.size()));
  out.resize(targetSize);
  for (size_t i = 0; i < targetSize; ++i) {
    out[i] =
        static_cast<OutT>(handle.size() == 1 ? handle.raw(0) : handle.raw(i));
  }
  return llvm::Error::success();
}

/// Given a Glow Handle \p Handle returns the first element in the Tensor
/// of that Handle, asserting it is equivalent to all other elements in the
/// Tensor.
template <typename T, typename OutT>
llvm::Error contractParamIfNeeded(const glow::Handle<T> &handle,
                                  OutT &firstElem) {
  RETURN_ERR_IF_NOT(
      handle.size() > 0,
      glow::strFormat("Handle size must be > 0, got %lu.", handle.size()));
  firstElem = handle.raw(0);
  for (size_t i = 1; i < handle.size(); i++) {
    RETURN_ERR_IF_NOT(handle.raw(i) == firstElem,
                      "All elements must be equal.");
  }
  return llvm::Error::success();
}

/// Given Node inputs and outputs, check the expected sizes.
template <typename T>
llvm::Error checkInputAndOutputSizes(const T &inputs, size_t inputsSize,
                                     const T &outputs, size_t outputsSize) {
  RETURN_ERR_IF_NOT(inputs.size() == inputsSize,
                    glow::strFormat("Expected number inputs %lu, got %lu.",
                                    inputsSize, inputs.size()));
  RETURN_ERR_IF_NOT(outputs.size() == outputsSize,
                    glow::strFormat("Expected number outputs %lu, got %lu.",
                                    outputsSize, outputs.size()));
  return llvm::Error::success();
}

} // namespace

// static
const PyTorchModelLoader::MappingOfMemberFunctions &
PyTorchModelLoader::getSymbolsMapping() {
  /// Static map of the above mappings PyTorch operator -> loader method.
  static const MappingOfMemberFunctions symbolLoaderMapping = {
      {at::Symbol::fromQualString("prim::Constant"),
       &PyTorchModelLoader::loadConstant},
      {at::Symbol::fromQualString("aten::mul"), &PyTorchModelLoader::loadMul},
      {at::Symbol::fromQualString("aten::mul_"), &PyTorchModelLoader::loadMul},
      {at::Symbol::fromQualString("aten::div"), &PyTorchModelLoader::loadDiv},
      {at::Symbol::fromQualString("aten::div_"), &PyTorchModelLoader::loadDiv},
      {at::Symbol::fromQualString("aten::add"), &PyTorchModelLoader::loadAdd},
      {at::Symbol::fromQualString("aten::add_"), &PyTorchModelLoader::loadAdd},
      {at::Symbol::fromQualString("aten::sub"), &PyTorchModelLoader::loadSub},
      {at::Symbol::fromQualString("aten::sub_"), &PyTorchModelLoader::loadSub},
      {at::Symbol::fromQualString("aten::relu"), &PyTorchModelLoader::loadRelu},
      {at::Symbol::fromQualString("aten::relu_"),
       &PyTorchModelLoader::loadRelu},
      {at::Symbol::fromQualString("aten::reciprocal"),
       &PyTorchModelLoader::loadReciprocal},
      {at::Symbol::fromQualString("aten::reciprocal_"),
       &PyTorchModelLoader::loadReciprocal},
      {at::Symbol::fromQualString("aten::_convolution"),
       &PyTorchModelLoader::loadConvolution},
      {at::Symbol::fromQualString("aten::batch_norm"),
       &PyTorchModelLoader::loadBatchNorm},
      {at::Symbol::fromQualString("aten::max_pool2d"),
       &PyTorchModelLoader::loadMaxPool2d},
      {at::Symbol::fromQualString("aten::adaptive_avg_pool2d"),
       &PyTorchModelLoader::loadAdaptiveAvgPool2d},
      {at::Symbol::fromQualString("aten::t"),
       &PyTorchModelLoader::loadTranspose},
      {at::Symbol::fromQualString("aten::t_"),
       &PyTorchModelLoader::loadTranspose},
      {at::Symbol::fromQualString("aten::min"), &PyTorchModelLoader::loadMin},
      {at::Symbol::fromQualString("aten::max"), &PyTorchModelLoader::loadMax},
      {at::Symbol::fromQualString("aten::exp"), &PyTorchModelLoader::loadExp},
      {at::Symbol::fromQualString("aten::sqrt"), &PyTorchModelLoader::loadSqrt},
      {at::Symbol::fromQualString("aten::sqrt_"),
       &PyTorchModelLoader::loadSqrt},
  };

  return symbolLoaderMapping;
}

// static
bool PyTorchModelLoader::isNodeSupported(const torch::jit::Node *node) {
  const auto &mapping = getSymbolsMapping();
  return mapping.count(node->kind()) != 0;
}

llvm::Error PyTorchModelLoader::loadNode(const torch::jit::Node *node) {
  const auto &mapping = getSymbolsMapping();
  auto it = mapping.find(node->kind());
  if (it == mapping.end()) {
    RETURN_ERR(glow::strFormat("Node kind %s is not supported by Glow",
                               node->kind().toDisplayString()));
  }
  return (this->*it->second)(node);
}

llvm::Expected<glow::NodeValue>
PyTorchModelLoader::getGlowNodeValue(const torch::jit::Value *value) const {
  const auto it = valueMap_.find(value);
  if (it == valueMap_.end()) {
    RETURN_ERR(glow::strFormat("Value %s is not found.",
                               value->debugNameBase().c_str()));
  }
  return llvm::Expected<glow::NodeValue>(it->second);
}

bool PyTorchModelLoader::hasGlowNodeValue(
    const torch::jit::Value *value) const {
  return valueMap_.count(value) != 0;
}

llvm::Error PyTorchModelLoader::addGlowNodeValue(const torch::jit::Value *value,
                                                 glow::NodeValue nodeValue) {
  auto p = valueMap_.insert({value, nodeValue});
  RETURN_ERR_IF_NOT(p.second, glow::strFormat("Value %s is already in the map.",
                                              value->debugNameBase().c_str()));
  return llvm::Error::success();
}

template <typename T>
llvm::Expected<glow::Handle<T>> PyTorchModelLoader::getGlowConstantHandle(
    const torch::jit::Value *value) const {
  glow::NodeValue nodeValue;
  ASSIGN_VALUE_OR_RETURN_ERR(nodeValue, getGlowNodeValue(value));
  return getGlowConstantPayload<T>(nodeValue.getNode());
}

llvm::Expected<glow::Placeholder *>
PyTorchModelLoader::loadValue(const torch::jit::Value *value) {
  RETURN_ERR_IF_NOT(value->isCompleteTensor(),
                    glow::strFormat("Value %s must have CompleteTensor type.",
                                    value->debugNameBase().c_str()));
  auto ptType = value->type()->cast<at::CompleteTensorType>();
  auto glowType = ptTypeToGlowType(*ptType.get());
  glow::Placeholder *ph = F_.getParent()->createPlaceholder(
      &glowType, "input", /*isTrainable*/ false);
  RETURN_IF_ERR(addGlowNodeValue(value, ph->getOutput()));
  return llvm::Expected<glow::Placeholder *>(ph);
}

llvm::Error PyTorchModelLoader::loadMul(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 2, outputs, 1));

  glow::NodeValue lhs;
  ASSIGN_VALUE_OR_RETURN_ERR(lhs, getGlowNodeValue(inputs[0]));
  glow::NodeValue rhs;
  ASSIGN_VALUE_OR_RETURN_ERR(rhs, getGlowNodeValue(inputs[1]));

  glow::MulNode *glowNode = F_.createMul("mul", lhs, rhs);
  return addGlowNodeValue(outputs[0], glowNode->getResult());
}

llvm::Error PyTorchModelLoader::loadDiv(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 2, outputs, 1));

  glow::NodeValue lhs;
  ASSIGN_VALUE_OR_RETURN_ERR(lhs, getGlowNodeValue(inputs[0]));
  glow::NodeValue rhs;
  ASSIGN_VALUE_OR_RETURN_ERR(rhs, getGlowNodeValue(inputs[1]));

  glow::DivNode *glowNode = F_.createDiv("div", lhs, rhs);
  return addGlowNodeValue(outputs[0], glowNode->getResult());
}

llvm::Error PyTorchModelLoader::loadAdd(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 3, outputs, 1));

  // TODO: extend this to allow non-constant scalars.
  // Check scalar is 1, any other value is not supported.
  auto scalarHandle = glow::Handle<int32_t>::createInvalidHandle();
  ASSIGN_VALUE_OR_RETURN_ERR(scalarHandle,
                             getGlowConstantHandle<int32_t>(inputs[2]));
  RETURN_ERR_IF_NOT(
      scalarHandle.size() == 1,
      glow::strFormat("Second input must be 1D tensor, got %lu dimensions.",
                      scalarHandle.size()));
  RETURN_ERR_IF_NOT(scalarHandle.raw(0) == 1,
                    glow::strFormat("Scalar must have value equal 1."));

  glow::NodeValue lhs;
  ASSIGN_VALUE_OR_RETURN_ERR(lhs, getGlowNodeValue(inputs[0]));
  glow::NodeValue rhs;
  ASSIGN_VALUE_OR_RETURN_ERR(rhs, getGlowNodeValue(inputs[1]));

  glow::AddNode *glowNode = F_.createAdd("add", lhs, rhs);
  return addGlowNodeValue(outputs[0], glowNode->getResult());
}

llvm::Error PyTorchModelLoader::loadSub(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 3, outputs, 1));

  // TODO: extend this to allow non-constant scalars.
  // Check scalar is 1, any other value is not supported.
  auto scalarHandle = glow::Handle<int32_t>::createInvalidHandle();
  ASSIGN_VALUE_OR_RETURN_ERR(scalarHandle,
                             getGlowConstantHandle<int32_t>(inputs[2]));
  RETURN_ERR_IF_NOT(
      scalarHandle.size() == 1,
      glow::strFormat("Second input must be 1D tensor, got %lu dimensions.",
                      scalarHandle.size()));
  RETURN_ERR_IF_NOT(scalarHandle.raw(0) == 1,
                    glow::strFormat("Scalar must have value equal 1."));

  glow::NodeValue lhs;
  ASSIGN_VALUE_OR_RETURN_ERR(lhs, getGlowNodeValue(inputs[0]));
  glow::NodeValue rhs;
  ASSIGN_VALUE_OR_RETURN_ERR(rhs, getGlowNodeValue(inputs[1]));

  glow::SubNode *glowNode = F_.createSub("sub", lhs, rhs);
  return addGlowNodeValue(outputs[0], glowNode->getResult());
}

llvm::Error PyTorchModelLoader::loadMax(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 2, outputs, 1));

  glow::NodeValue lhs;
  ASSIGN_VALUE_OR_RETURN_ERR(lhs, getGlowNodeValue(inputs[0]));
  glow::NodeValue rhs;
  ASSIGN_VALUE_OR_RETURN_ERR(rhs, getGlowNodeValue(inputs[1]));

  glow::MaxNode *glowNode = F_.createMax("max", lhs, rhs);
  return addGlowNodeValue(outputs[0], glowNode->getResult());
}

llvm::Error PyTorchModelLoader::loadRelu(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 1, outputs, 1));

  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(input, getGlowNodeValue(inputs[0]));

  glow::ReluNode *glowNode = F_.createRELU("relu", input);
  return addGlowNodeValue(outputs[0], glowNode->getResult());
}

llvm::Error PyTorchModelLoader::loadExp(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 1, outputs, 1));

  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(input, getGlowNodeValue(inputs[0]));

  glow::ExpNode *glowNode = F_.createExp("exp", input);
  return addGlowNodeValue(outputs[0], glowNode->getResult());
}

llvm::Error PyTorchModelLoader::loadSqrt(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 1, outputs, 1));

  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(input, getGlowNodeValue(inputs[0]));

  glow::PowNode *glowNode = F_.createPow("sqrt", input, /*exp=*/0.5);
  return addGlowNodeValue(outputs[0], glowNode->getResult());
}

llvm::Error PyTorchModelLoader::loadReciprocal(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 1, outputs, 1));

  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(input, getGlowNodeValue(inputs[0]));
  glow::PowNode *glowNode = F_.createPow("reciprocal", input, /*exp=*/-1);
  return addGlowNodeValue(outputs[0], glowNode->getResult());
}

llvm::Error
PyTorchModelLoader::loadConvolution(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 12, outputs, 1));

  // Indexes of aten::_convolution inputs.
  struct Inputs {
    enum {
      input = 0, // NCHW
      weights = 1,
      bias = 2,
      stride = 3,
      padding = 4,
      dilation = 5,
      transposed = 6,
      output_padding = 7,
      groups = 8,
      benchmark = 9,
      deterministic = 10,
      cudnn_enabled = 11
    };
  };

  // Glow expects conv inputs to be in NHWC but PyTorch keeps them in NCHW so
  // we tranpose them.
  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(input, getGlowNodeValue(inputs[Inputs::input]));

  input = F_.createTranspose("conv_input_transposed", input, NCHW2NHWC);
  glow::ShapeNHWC inputShape(input.dims());

  // Glow expects conv weights to be in CRSK but PyTorch keeps them in CKRS
  // so we tranpose them. C - output_depth, R - filter_height, S -
  // filter_width, K - input_depth.
  glow::NodeValue weights;
  ASSIGN_VALUE_OR_RETURN_ERR(weights,
                             getGlowNodeValue(inputs[Inputs::weights]));
  weights = F_.createTranspose("conv_weights_transposed", weights, NCHW2NHWC);
  glow::ShapeNHWC weightsShape(weights.dims());

  // If a bias was provided then use it otherwise create a 0 bias.
  glow::NodeValue bias;
  if (hasGlowNodeValue(inputs[Inputs::bias])) {
    ASSIGN_VALUE_OR_RETURN_ERR(bias, getGlowNodeValue(inputs[Inputs::bias]));
  } else {
    glow::Tensor biasT(glow::ElemKind::FloatTy, {weightsShape.n});
    biasT.zero();
    glow::Constant *biasConstant =
        F_.getParent()->createConstant("conv_bias", std::move(biasT));
    bias = biasConstant->getOutput();
  }

  auto stridesHandle = glow::Handle<int32_t>::createInvalidHandle();
  ASSIGN_VALUE_OR_RETURN_ERR(
      stridesHandle, getGlowConstantHandle<int32_t>(inputs[Inputs::stride]));
  std::vector<uint32_t> strides;
  RETURN_IF_ERR(expandParamIfNeeded(stridesHandle, 2, strides));

  auto padsHandle = glow::Handle<int32_t>::createInvalidHandle();
  ASSIGN_VALUE_OR_RETURN_ERR(
      padsHandle, getGlowConstantHandle<int32_t>(inputs[Inputs::padding]));
  uint32_t pad;
  RETURN_IF_ERR(contractParamIfNeeded(padsHandle, pad));
  std::vector<uint32_t> pads = {pad, pad, pad, pad};

  auto dilationHandle = glow::Handle<int32_t>::createInvalidHandle();
  ASSIGN_VALUE_OR_RETURN_ERR(
      dilationHandle, getGlowConstantHandle<int32_t>(inputs[Inputs::dilation]));
  uint32_t dilation;
  RETURN_IF_ERR(contractParamIfNeeded(dilationHandle, dilation));

  // Don't support transposed convolutions yet.
  auto transposedHandle = glow::Handle<bool>::createInvalidHandle();
  ASSIGN_VALUE_OR_RETURN_ERR(transposedHandle, getGlowConstantHandle<bool>(
                                                   inputs[Inputs::transposed]));
  RETURN_ERR_IF_NOT(transposedHandle.size() == 1 && !transposedHandle.raw(0),
                    "Transposed convolutions not supported.");

  auto groupsHandle = glow::Handle<int32_t>::createInvalidHandle();
  ASSIGN_VALUE_OR_RETURN_ERR(
      groupsHandle, getGlowConstantHandle<int32_t>(inputs[Inputs::groups]));
  RETURN_ERR_IF_NOT(groupsHandle.size() == 1,
                    glow::strFormat("Groups size must be equal to 1, got %lu",
                                    groupsHandle.size()));
  uint32_t groups = groupsHandle.raw(0);

  std::vector<uint32_t> kernels = {static_cast<uint32_t>(weightsShape.h),
                                   static_cast<uint32_t>(weightsShape.w)};

  auto outSz = glow::calculateConvPoolOutputDims(
      inputShape.h, inputShape.w, kernels, strides, pads, dilation);
  std::array<size_t, 4> outDims = {
      {input.dims()[0], outSz.first, outSz.second, weightsShape.n}};
  glow::TypeRef outTy =
      F_.getParent()->uniqueType(glow::ElemKind::FloatTy, outDims);

  glow::ConvolutionNode *conv =
      F_.createConv("conv", input, weights, bias, outTy, kernels, strides, pads,
                    groups, dilation);
  glow::TransposeNode *output = F_.createTranspose(
      "conv_output_transposed", conv->getResult(), NHWC2NCHW);

  return addGlowNodeValue(outputs[0], output->getResult());
}

llvm::Error PyTorchModelLoader::loadBatchNorm(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 9, outputs, 1));

  // Indexes of aten::batch_norm inputs.
  struct Inputs {
    enum {
      input = 0, // NCHW
      weights = 1,
      bias = 2,
      running_mean = 3,
      running_var = 4,
      training = 5,
      momentum = 6,
      eps = 7,
      cuddnn_enabled = 8,
    };
  };

  // Don't support training yet.
  auto trainingHandle = glow::Handle<bool>::createInvalidHandle();
  ASSIGN_VALUE_OR_RETURN_ERR(
      trainingHandle, getGlowConstantHandle<bool>(inputs[Inputs::training]));
  RETURN_ERR_IF_NOT(trainingHandle.size() == 1, "Training must be 1D vector.");
  RETURN_ERR_IF_NOT(trainingHandle.raw(0) == false,
                    "Training must be scalar with false value.");

  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(input, getGlowNodeValue(inputs[Inputs::input]));
  RETURN_ERR_IF_NOT(
      input.dims().size() == 4,
      glow::strFormat("Number input dimensions must be equal to 4, got %lu",
                      input.dims().size()));

  size_t numChannels = input.dims()[1];

  glow::NodeValue weights;
  if (hasGlowNodeValue(inputs[Inputs::weights])) {
    ASSIGN_VALUE_OR_RETURN_ERR(weights,
                               getGlowNodeValue(inputs[Inputs::weights]));
  } else {
    glow::Tensor weightsT(glow::ElemKind::FloatTy, {numChannels});
    weightsT.init(glow::Tensor::InitKind::Broadcast, 1,
                  F_.getParent()->getPRNG());
    glow::Constant *weightsConstant = F_.getParent()->createConstant(
        "batchnorm_weights", std::move(weightsT));
    weights = weightsConstant->getOutput();
  }

  glow::NodeValue bias;
  if (hasGlowNodeValue(inputs[Inputs::bias])) {
    ASSIGN_VALUE_OR_RETURN_ERR(bias, getGlowNodeValue(inputs[Inputs::bias]));
  } else {
    glow::Tensor biasT(glow::ElemKind::FloatTy, {numChannels});
    biasT.zero();
    glow::Constant *biasConstant =
        F_.getParent()->createConstant("batchnorm_bias", std::move(biasT));
    bias = biasConstant->getOutput();
  }

  glow::NodeValue mean;
  ASSIGN_VALUE_OR_RETURN_ERR(mean,
                             getGlowNodeValue(inputs[Inputs::running_mean]));
  glow::NodeValue var;
  ASSIGN_VALUE_OR_RETURN_ERR(var,
                             getGlowNodeValue(inputs[Inputs::running_var]));

  auto momentumHandle = glow::Handle<float>::createInvalidHandle();
  ASSIGN_VALUE_OR_RETURN_ERR(
      momentumHandle, getGlowConstantHandle<float>(inputs[Inputs::momentum]));
  RETURN_ERR_IF_NOT(momentumHandle.size() == 1, "Momentum must be 1D vector.");
  float momentum = momentumHandle.raw(0);

  auto epsilonHandle = glow::Handle<float>::createInvalidHandle();
  ASSIGN_VALUE_OR_RETURN_ERR(epsilonHandle,
                             getGlowConstantHandle<float>(inputs[Inputs::eps]));
  RETURN_ERR_IF_NOT(
      epsilonHandle.size() == 1,
      glow::strFormat("Number epsilon dimensions must be equal to 1, got %lu",
                      epsilonHandle.size()));
  float epsilon = epsilonHandle.raw(0);

  // Input is in NCHW.
  uint32_t channelIdx = 1;

  glow::BatchNormalizationNode *bn =
      F_.createBatchNormalization("batchnorm", input, bias, weights, mean, var,
                                  channelIdx, epsilon, momentum);
  return addGlowNodeValue(outputs[0], bn->getResult());
}

llvm::Error PyTorchModelLoader::loadMaxPool2d(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 6, outputs, 1));

  // Indexes of aten::max_pool2d inputs.
  struct Inputs {
    enum {
      input = 0, // NCHW
      kernel_size = 1,
      stride = 2,
      padding = 3,
      dilation = 4,
      ceil_mode = 5,
    };
  };

  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(input, getGlowNodeValue(inputs[Inputs::input]));
  input = F_.createTranspose("maxpool2d_input_transposed", input, NCHW2NHWC);

  auto kernelsHandle = glow::Handle<int32_t>::createInvalidHandle();
  ASSIGN_VALUE_OR_RETURN_ERR(kernelsHandle, getGlowConstantHandle<int32_t>(
                                                inputs[Inputs::kernel_size]));
  std::vector<uint32_t> kernels;
  RETURN_IF_ERR(expandParamIfNeeded(kernelsHandle, 2, kernels));

  auto padsHandle = glow::Handle<int32_t>::createInvalidHandle();
  ASSIGN_VALUE_OR_RETURN_ERR(
      padsHandle, getGlowConstantHandle<int32_t>(inputs[Inputs::padding]));
  std::vector<uint32_t> padsPair;
  RETURN_IF_ERR(expandParamIfNeeded(padsHandle, 2, padsPair));
  std::vector<uint32_t> pads = {padsPair[0], padsPair[1], padsPair[0],
                                padsPair[1]};

  // Stride defaults to kernel_size.
  std::vector<uint32_t> strides;
  if (hasGlowNodeValue(inputs[2])) {
    auto stridesHandle = glow::Handle<int32_t>::createInvalidHandle();
    ASSIGN_VALUE_OR_RETURN_ERR(
        stridesHandle, getGlowConstantHandle<int32_t>(inputs[Inputs::stride]));
    RETURN_IF_ERR(expandParamIfNeeded(stridesHandle, 2, strides));
  } else {
    strides = kernels;
  }

  // Glow doesn't support maxpool dilation.
  auto dilationHandle = glow::Handle<int32_t>::createInvalidHandle();
  ASSIGN_VALUE_OR_RETURN_ERR(
      dilationHandle, getGlowConstantHandle<int32_t>(inputs[Inputs::dilation]));
  for (size_t i = 0; i < dilationHandle.size(); ++i) {
    RETURN_ERR_IF_NOT(
        dilationHandle.raw(i) == 1,
        glow::strFormat("Dilation value must be equal to 1, got %d",
                        dilationHandle.raw(i)));
  }

  // Glow doesn't support maxpool ceil mode.
  auto ceilModeHandle = glow::Handle<bool>::createInvalidHandle();
  ASSIGN_VALUE_OR_RETURN_ERR(
      ceilModeHandle, getGlowConstantHandle<bool>(inputs[Inputs::ceil_mode]));
  RETURN_ERR_IF_NOT(ceilModeHandle.size() == 1, "ceilMode must be 1D vector.");
  RETURN_ERR_IF_NOT(ceilModeHandle.raw(0) == false,
                    "ceilMode must be scalar with false value.");

  glow::MaxPoolNode *mp =
      F_.createMaxPool("maxpool2d", input, kernels, strides, pads);
  glow::NodeValue output = mp->getResult();
  output = F_.createTranspose("maxpool2d_output_transposed", output, NHWC2NCHW);
  return addGlowNodeValue(outputs[0], output);
}

llvm::Error
PyTorchModelLoader::loadAdaptiveAvgPool2d(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 2, outputs, 1));

  // Indexes of aten::adaptive_avg_pool2d inputs.
  struct Inputs {
    enum {
      input = 0, // NCHW
      output_size = 1,
    };
  };

  // Glow expects inputs to be in NHWC but PyTorch keeps them in NCHW so we
  // tranpose them.
  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(input, getGlowNodeValue(inputs[Inputs::input]));

  input = F_.createTranspose("adaptive_avg_pool2d_input_transposed", input,
                             NCHW2NHWC);

  auto outputSizeHandle = glow::Handle<int32_t>::createInvalidHandle();
  ASSIGN_VALUE_OR_RETURN_ERR(
      outputSizeHandle,
      getGlowConstantHandle<int32_t>(inputs[Inputs::output_size]));
  std::vector<uint32_t> outputSize;
  RETURN_IF_ERR(expandParamIfNeeded(outputSizeHandle, 2, outputSize));

  auto idim = glow::ShapeNHWC(input.dims());
  auto outTy = F_.getParent()->uniqueTypeWithNewShape(
      input.getType(), {idim.n, outputSize[0], outputSize[1], idim.c});

  glow::NodeValue output =
      F_.createAdaptiveAvgPool("adaptive_avg_pool2d", input, outTy);
  output = F_.createTranspose("adaptive_avg_pool2d_output_transposed", output,
                              NHWC2NCHW);
  return addGlowNodeValue(outputs[0], output);
}

llvm::Error PyTorchModelLoader::loadTranspose(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 1, outputs, 1));

  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(input, getGlowNodeValue(inputs[0]));

  glow::NodeValue output;
  if (input.dims().size() == 1) {
    output = input;
  } else if (input.dims().size() == 2) {
    output = F_.createTranspose("transpose", input, {1, 0});
  } else {
    RETURN_ERR("Transpose requires input to have rank <= 2");
  }
  return addGlowNodeValue(outputs[0], output);
}

llvm::Error PyTorchModelLoader::loadMin(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 2, outputs, 1));

  glow::NodeValue lhs;
  ASSIGN_VALUE_OR_RETURN_ERR(lhs, getGlowNodeValue(inputs[0]));
  glow::NodeValue rhs;
  ASSIGN_VALUE_OR_RETURN_ERR(rhs, getGlowNodeValue(inputs[1]));

  auto output = F_.createMin("min", lhs, rhs);
  return addGlowNodeValue(outputs[0], output);
}

llvm::Error PyTorchModelLoader::loadConstant(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 0, outputs, 1));

  auto optionalIValue = torch::jit::toIValue(outputs[0]);
  RETURN_ERR_IF_NOT(optionalIValue.has_value(),
                    "Constants should have IValue outputs.");
  const torch::jit::IValue iVal = *optionalIValue;

  glow::Module *mod = F_.getParent();

  glow::Tensor t;

  if (iVal.isInt()) {
    int64_t val = iVal.toInt();
    t.reset(glow::ElemKind::Int32ITy, {1});
    ASSIGN_VALUE_OR_RETURN_ERR(t.getHandle<int32_t>().raw(0), to32Bit(val));
  } else if (iVal.isIntList()) {
    const auto vals = iVal.toIntListRef();
    if (vals.empty()) {
      return llvm::Error::success();
    }
    t.reset(glow::ElemKind::Int32ITy, {vals.size()});
    auto handle = t.getHandle<int32_t>();
    for (size_t i = 0; i < vals.size(); ++i) {
      ASSIGN_VALUE_OR_RETURN_ERR(handle.raw(i), to32Bit(vals[i]));
    }
  } else if (iVal.isDouble()) {
    double val = iVal.toDouble();
    t.reset(glow::ElemKind::FloatTy, {1});
    ASSIGN_VALUE_OR_RETURN_ERR(t.getHandle<float>().raw(0), to32Bit(val));
  } else if (iVal.isDoubleList()) {
    const auto vals = iVal.toDoubleListRef();
    if (vals.empty()) {
      return llvm::Error::success();
    }
    t.reset(glow::ElemKind::FloatTy, {vals.size()});
    auto handle = t.getHandle<float>();
    for (size_t i = 0; i < vals.size(); ++i) {
      ASSIGN_VALUE_OR_RETURN_ERR(handle.raw(i), to32Bit(vals[i]));
    }
  } else if (iVal.isBool()) {
    bool val = iVal.toBool();
    t.reset(glow::ElemKind::BoolTy, {1});
    t.getHandle<bool>().raw(0) = val;
  } else if (iVal.isBoolList()) {
    const auto &valsList = iVal.toBoolList();
    const auto &vals = c10::impl::toVector(valsList);
    if (vals.empty()) {
      return llvm::Error::success();
    }
    t.reset(glow::ElemKind::BoolTy, {vals.size()});
    auto handle = t.getHandle<bool>();
    for (size_t i = 0; i < vals.size(); ++i) {
      handle.raw(i) = vals[i];
    }
  } else if (iVal.isTensor()) {
    RETURN_ERR("Not yet implemented tensor constants.");
  } else if (iVal.isNone()) {
    // Nothing to do for None
    return llvm::Error::success();
  } else {
    RETURN_ERR("Unsupported constant type.");
  }

  glow::Constant *glowConstant = mod->createConstant("constant", std::move(t));
  return addGlowNodeValue(outputs[0], glowConstant->getOutput());
}

/*static*/
llvm::Error PyTorchModelLoader::loadJITGraph(
    glow::Function &F, torch::jit::Graph &subgraph,
    at::ArrayRef<torch::jit::IValue> &inputs,
    std::vector<glow::Placeholder *> &inputPlaceholders,
    std::vector<glow::Placeholder *> &outputPlaceholders) {
  llvm::Error error = llvm::Error::success();
  MARK_ERR_CHECKED(error);
  PyTorchModelLoader loader(F, subgraph, inputs, inputPlaceholders,
                            outputPlaceholders, error);
  return error;
}

PyTorchModelLoader::PyTorchModelLoader(
    glow::Function &F, torch::jit::Graph &subgraph,
    at::ArrayRef<torch::jit::IValue> &inputs,
    std::vector<glow::Placeholder *> &inputPlaceholders,
    std::vector<glow::Placeholder *> &outputPlaceholders, llvm::Error &error)
    : F_(F) {
  auto subgraphInputValues = subgraph.inputs();

  if (inputs.size() != subgraphInputValues.size()) {
    error =
        MAKE_ERR(glow::strFormat("Number of Graph inputs %lu must match the "
                                 "number of provided inputs %lu.",
                                 subgraphInputValues.size(), inputs.size()));
    return;
  }

  for (size_t i = 0; i < subgraphInputValues.size(); ++i) {
    torch::jit::Value *inputValue = subgraphInputValues[i];
    const c10::IValue inputIValue = inputs.at(i);
    if (inputIValue.isTensor()) {
      inputValue->inferTypeFrom(inputIValue.toTensor());
      glow::Placeholder *PH;
      if (auto resOrErr = loadValue(inputValue)) {
        PH = std::move(*resOrErr);
      } else {
        error = resOrErr.takeError();
        return;
      }
      inputPlaceholders.push_back(PH);
    }
  }

  // Nodes are topologically sorted.
  for (auto node : subgraph.nodes()) {
    if ((error = loadNode(node))) {
      return;
    }
  }

  for (torch::jit::Value *output : subgraph.outputs()) {
    glow::NodeValue outputNodeValue;
    if (auto resOrErr = getGlowNodeValue(output)) {
      outputNodeValue = std::move(*resOrErr);
    } else {
      error = resOrErr.takeError();
      return;
    }

    auto *save = F_.createSave("save", outputNodeValue);
    outputPlaceholders.push_back(save->getPlaceholder());
  }
  error = llvm::Error::success();
}
