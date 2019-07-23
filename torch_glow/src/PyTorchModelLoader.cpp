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

#include "glow/Support/Support.h"

#include <torch/csrc/jit/ir.h>

namespace {
/// \returns a corresponding Glow Type for a given PyTorch CompleteTensorType \p
/// ptType.
glow::Type ptTypeToGlowType(const at::CompleteTensorType &ptType) {
  // TODO: get correct ElemKind
  std::vector<size_t> dims;
  for (auto &size : ptType.sizes()) {
    dims.push_back(size);
  }
  return glow::Type(glow::ElemKind::FloatTy, dims);
}

/// Downcast a double to a float.
float to32Bit(double val) {
  assert(val <= std::numeric_limits<float>::max());
  assert(val >= std::numeric_limits<float>::lowest());
  return static_cast<float>(val);
}

/// Downcast an int64_t to a int32_t.
int32_t to32Bit(int64_t val) {
  assert(val <= std::numeric_limits<int32_t>::max());
  assert(val >= std::numeric_limits<int32_t>::lowest());
  return static_cast<int32_t>(val);
}

/// Given a Glow Node \p glowNode which must be a Glow Constant, returns a Glow
/// Handle to that Constant's payload.
template <typename T>
const glow::Handle<T> getGlowConstantPayload(const glow::Node *glowNode) {
  assert(glowNode);
  const glow::Constant *glowConstant = llvm::dyn_cast<glow::Constant>(glowNode);
  assert(glowConstant);
  if (std::is_same<int32_t, T>::value) {
    assert(glowConstant->getElementType() == glow::ElemKind::Int32ITy);
  } else if (std::is_same<float, T>::value) {
    assert(glowConstant->getElementType() == glow::ElemKind::FloatTy);
  } else if (std::is_same<bool, T>::value) {
    assert(glowConstant->getElementType() == glow::ElemKind::BoolTy);
  } else {
    assert(false && "Unsupported type");
  }
  return glowConstant->getPayload().getHandle<T>();
}

/// Given a Glow Handle \p handle and \p targetSize, will return an std::vector
/// of the elements in the Tensor the Handle manages if the size of that Tensor
/// is targetSize or if the Tensor has only one element in it, will replicate
/// that element targetSize times.
template <typename T, typename OutT = T>
std::vector<OutT> expandParamIfNeeded(const glow::Handle<T> &handle,
                                      size_t targetSize) {
  assert(handle.dims().size() == 1 && "Expected a 1d Glow Tensor");
  assert(handle.size() == 1 ||
         handle.size() == targetSize &&
             "Expected input size to be either 1 or the target size");
  std::vector<OutT> out;
  out.resize(targetSize);
  for (size_t i = 0; i < targetSize; ++i) {
    out[i] =
        static_cast<OutT>(handle.size() == 1 ? handle.raw(0) : handle.raw(i));
  }
  return out;
}

/// Given a Glow Handle \p Handle returns the first element in the Tensor
/// of that Handle, asserting it is equivalent to all other elements in the
/// Tensor.
template <typename T> T contractParamIfNeeded(const glow::Handle<T> &handle) {
  assert(handle.size() > 0);
  const T firstElem = handle.raw(0);
  for (size_t i = 1; i < handle.size(); i++) {
    assert(handle.raw(i) == firstElem);
  }
  return firstElem;
}
} // namespace

glow::NodeValue
PyTorchModelLoader::getGlowNodeValue(const torch::jit::Value *value) const {
  const auto it = valueMap_.find(value);
  assert(it != valueMap_.end());
  return it->second;
}

bool PyTorchModelLoader::hasGlowNodeValue(
    const torch::jit::Value *value) const {
  const auto it = valueMap_.find(value);
  return it != valueMap_.end();
}

void PyTorchModelLoader::addGlowNodeValue(const torch::jit::Value *value,
                                          glow::NodeValue nodeValue) {
  assert(valueMap_.count(value) == 0);
  valueMap_[value] = nodeValue;
}

template <typename T>
glow::Handle<T> PyTorchModelLoader::getGlowConstantHandle(
    const torch::jit::Value *value) const {
  glow::NodeValue nodeValue = getGlowNodeValue(value);
  return getGlowConstantPayload<T>(nodeValue.getNode());
}

glow::Placeholder *
PyTorchModelLoader::loadValue(const torch::jit::Value *value) {
  assert(value->isCompleteTensor());
  auto ptType = value->type()->cast<at::CompleteTensorType>();
  auto glowType = ptTypeToGlowType(*ptType.get());
  glow::Placeholder *ph =
      mod_->createPlaceholder(&glowType, "input", /*isTrainable*/ false);
  addGlowNodeValue(value, ph->getOutput());
  return ph;
}

void PyTorchModelLoader::loadMul(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  assert(inputs.size() == 2);
  assert(outputs.size() == 1);

  glow::NodeValue rhs = getGlowNodeValue(inputs[0]);
  glow::NodeValue lhs = getGlowNodeValue(inputs[1]);
  glow::MulNode *glowNode = f_->createMul("mul", rhs, lhs);
  addGlowNodeValue(outputs[0], glowNode->getResult());
}

void PyTorchModelLoader::loadDiv(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  assert(inputs.size() == 2);
  assert(outputs.size() == 1);

  glow::NodeValue rhs = getGlowNodeValue(inputs[0]);
  glow::NodeValue lhs = getGlowNodeValue(inputs[1]);
  glow::DivNode *glowNode = f_->createDiv("div", rhs, lhs);
  addGlowNodeValue(outputs[0], glowNode->getResult());
}

void PyTorchModelLoader::loadAdd(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  assert(inputs.size() == 3);
  assert(outputs.size() == 1);

  // TODO: extend this to allow non-constant scalars.
  // Check scalar is 1, any other value is not supported.
  const auto scalarHandle = getGlowConstantHandle<int32_t>(inputs[2]);
  assert(scalarHandle.size() == 1);
  assert(scalarHandle.raw(0) == 1);
  (void)scalarHandle;

  glow::NodeValue rhs = getGlowNodeValue(inputs[0]);
  glow::NodeValue lhs = getGlowNodeValue(inputs[1]);
  glow::AddNode *glowNode = f_->createAdd("add", rhs, lhs);
  addGlowNodeValue(outputs[0], glowNode->getResult());
}

void PyTorchModelLoader::loadSub(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  assert(inputs.size() == 3);
  assert(outputs.size() == 1);

  // TODO: extend this to allow non-constant scalars.
  // Check scalar is 1, any other value is not supported.
  const auto scalarHandle = getGlowConstantHandle<int32_t>(inputs[2]);
  assert(scalarHandle.size() == 1);
  assert(scalarHandle.raw(0) == 1);
  (void)scalarHandle;

  glow::NodeValue rhs = getGlowNodeValue(inputs[0]);
  glow::NodeValue lhs = getGlowNodeValue(inputs[1]);
  glow::SubNode *glowNode = f_->createSub("sub", rhs, lhs);
  addGlowNodeValue(outputs[0], glowNode->getResult());
}

void PyTorchModelLoader::loadRelu(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  assert(inputs.size() == 1);
  assert(outputs.size() == 1);

  glow::NodeValue input = getGlowNodeValue(inputs[0]);
  glow::ReluNode *glowNode = f_->createRELU("relu", input);
  addGlowNodeValue(outputs[0], glowNode->getResult());
}

void PyTorchModelLoader::loadConvolution(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  assert(inputs.size() == 12);
  assert(outputs.size() == 1);

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

  // Glow expects conv inputs to be in NHWC but PyTorch keeps them in NCHW so we
  // tranpose them.
  glow::NodeValue input = getGlowNodeValue(inputs[Inputs::input]);
  input = f_->createTranspose("conv_input_transposed", input, NCHW2NHWC);
  glow::ShapeNHWC inputShape(input.dims());

  // Glow expects conv weights to be in CRSK but PyTorch keeps them in CKRS
  // so we tranpose them. C - output_depth, R - filter_height, S -
  // filter_width, K - input_depth.
  glow::NodeValue weights = getGlowNodeValue(inputs[Inputs::weights]);
  weights = f_->createTranspose("conv_weights_transposed", weights, NCHW2NHWC);
  glow::ShapeNHWC weightsShape(weights.dims());

  // If a bias was provided then use it otherwise create a 0 bias.
  glow::NodeValue bias;
  if (hasGlowNodeValue(inputs[Inputs::bias])) {
    bias = getGlowNodeValue(inputs[Inputs::bias]);
  } else {
    glow::Tensor biasT(glow::ElemKind::FloatTy, {weightsShape.n});
    biasT.zero();
    glow::Constant *biasConstant =
        f_->getParent()->createConstant("conv_bias", std::move(biasT));
    bias = biasConstant->getOutput();
  }

  glow::Handle<int32_t> stridesHandle =
      getGlowConstantHandle<int32_t>(inputs[Inputs::stride]);
  std::vector<uint32_t> strides =
      expandParamIfNeeded<int32_t, uint32_t>(stridesHandle, 2);

  glow::Handle<int32_t> padsHandle =
      getGlowConstantHandle<int32_t>(inputs[Inputs::padding]);
  uint32_t pad = static_cast<uint32_t>(contractParamIfNeeded(padsHandle));
  std::vector<uint32_t> pads = {pad, pad, pad, pad};

  glow::Handle<int32_t> dilationHandle =
      getGlowConstantHandle<int32_t>(inputs[Inputs::dilation]);
  uint32_t dilation =
      static_cast<uint32_t>(contractParamIfNeeded(dilationHandle));

  // Don't support transposed convolutions yet.
  glow::Handle<bool> transposedHandle =
      getGlowConstantHandle<bool>(inputs[Inputs::transposed]);
  assert(transposedHandle.size() == 1 && !transposedHandle.raw(0) &&
         "Transposed convolutions not supported");
  (void)transposedHandle;

  glow::Handle<int32_t> groupsHandle =
      getGlowConstantHandle<int32_t>(inputs[Inputs::groups]);
  assert(groupsHandle.size() == 1);
  uint32_t groups = groupsHandle.raw(0);

  std::vector<uint32_t> kernels = {static_cast<uint32_t>(weightsShape.h),
                                   static_cast<uint32_t>(weightsShape.w)};

  auto outSz = glow::calculateConvPoolOutputDims(
      inputShape.h, inputShape.w, kernels, strides, pads, dilation);
  std::array<size_t, 4> outDims = {
      {input.dims()[0], outSz.first, outSz.second, weightsShape.n}};
  glow::TypeRef outTy =
      f_->getParent()->uniqueType(glow::ElemKind::FloatTy, outDims);

  glow::ConvolutionNode *conv =
      f_->createConv("conv", input, weights, bias, outTy, kernels, strides,
                     pads, groups, dilation);
  glow::TransposeNode *output = f_->createTranspose(
      "conv_output_transposed", conv->getResult(), NHWC2NCHW);

  addGlowNodeValue(outputs[0], output->getResult());
}

void PyTorchModelLoader::loadBatchNorm(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  assert(inputs.size() == 9);
  assert(outputs.size() == 1);

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
  const auto trainingHandle =
      getGlowConstantHandle<bool>(inputs[Inputs::training]);
  assert(trainingHandle.size() == 1);
  assert(trainingHandle.raw(0) == false);
  (void)trainingHandle;

  glow::NodeValue input = getGlowNodeValue(inputs[Inputs::input]);
  assert(input.dims().size() == 4);

  size_t numChannels = input.dims()[1];

  glow::NodeValue weights;
  if (hasGlowNodeValue(inputs[Inputs::weights])) {
    weights = getGlowNodeValue(inputs[Inputs::weights]);
  } else {
    glow::Tensor weightsT(glow::ElemKind::FloatTy, {numChannels});
    weightsT.init(glow::Tensor::InitKind::Broadcast, 1,
                  f_->getParent()->getPRNG());
    glow::Constant *weightsConstant = f_->getParent()->createConstant(
        "batchnorm_weights", std::move(weightsT));
    weights = weightsConstant->getOutput();
  }

  glow::NodeValue bias;
  if (hasGlowNodeValue(inputs[Inputs::bias])) {
    bias = getGlowNodeValue(inputs[Inputs::bias]);
  } else {
    glow::Tensor biasT(glow::ElemKind::FloatTy, {numChannels});
    biasT.zero();
    glow::Constant *biasConstant =
        f_->getParent()->createConstant("batchnorm_bias", std::move(biasT));
    bias = biasConstant->getOutput();
  }

  glow::NodeValue mean = getGlowNodeValue(inputs[Inputs::running_mean]);
  glow::NodeValue var = getGlowNodeValue(inputs[Inputs::running_var]);

  const auto momentumHandle =
      getGlowConstantHandle<float>(inputs[Inputs::momentum]);
  assert(momentumHandle.size() == 1);
  float momentum = momentumHandle.raw(0);

  const auto epsilonHandle = getGlowConstantHandle<float>(inputs[Inputs::eps]);
  assert(epsilonHandle.size() == 1);
  float epsilon = epsilonHandle.raw(0);

  // Input is in NCHW.
  uint32_t channelIdx = 1;

  glow::BatchNormalizationNode *bn =
      f_->createBatchNormalization("batchnorm", input, bias, weights, mean, var,
                                   channelIdx, epsilon, momentum);
  addGlowNodeValue(outputs[0], bn->getResult());
}

void PyTorchModelLoader::loadMaxPool2d(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  assert(inputs.size() == 6);
  assert(outputs.size() == 1);

  // Indexes of aten::max_pool2d inputs.
  struct Inputs {
    enum {
      input = 1, // NCHW
      kernel_size = 1,
      stride = 2,
      padding = 3,
      dilation = 4,
      ceil_mode = 5,
    };
  };

  glow::NodeValue input = getGlowNodeValue(inputs[Inputs::input]);
  input = f_->createTranspose("maxpool2d_input_transposed", input, NCHW2NHWC);

  const auto kernelsHandle =
      getGlowConstantHandle<int32_t>(inputs[Inputs::kernel_size]);
  std::vector<uint32_t> kernels =
      expandParamIfNeeded<int32_t, uint32_t>(kernelsHandle, 2);

  const auto padsHandle =
      getGlowConstantHandle<int32_t>(inputs[Inputs::padding]);
  std::vector<uint32_t> padsPair =
      expandParamIfNeeded<int32_t, uint32_t>(padsHandle, 2);
  std::vector<uint32_t> pads = {padsPair[0], padsPair[1], padsPair[0],
                                padsPair[1]};

  // Stride defaults to kernel_size.
  std::vector<uint32_t> strides;
  if (hasGlowNodeValue(inputs[2])) {
    const auto stridesHandle =
        getGlowConstantHandle<int32_t>(inputs[Inputs::stride]);
    strides = expandParamIfNeeded<int32_t, uint32_t>(stridesHandle, 2);
  } else {
    strides = kernels;
  }

  // Glow doesn't support maxpool dilation.
  const auto dilationHandle =
      getGlowConstantHandle<int32_t>(inputs[Inputs::dilation]);
  for (size_t i = 0; i < dilationHandle.size(); ++i) {
    assert(dilationHandle.raw(i) == 1);
  }

  // Glow doesn't support maxpool ceil mode.
  const auto ceilModeHandle =
      getGlowConstantHandle<bool>(inputs[Inputs::ceil_mode]);
  assert(ceilModeHandle.size() == 1);
  assert(ceilModeHandle.raw(0) == false);
  (void)ceilModeHandle;

  glow::MaxPoolNode *mp =
      f_->createMaxPool("maxpool2d", input, kernels, strides, pads);
  glow::NodeValue output = mp->getResult();
  output =
      f_->createTranspose("maxpool2d_output_transposed", output, NHWC2NCHW);
  addGlowNodeValue(outputs[0], output);
}

void PyTorchModelLoader::loadReshape(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  assert(inputs.size() == 2);
  assert(outputs.size() == 1);

  // Indexes of aten::reshape inputs.
  struct Inputs {
	enum {
	  input = 0,
      output_shape = 1,
	};
  };

  glow::NodeValue input = getGlowNodeValue(inputs[Inputs::input]);

  const auto shapeHandle =
	  getGlowConstantHandle<uint32_t>(inputs[Inputs::output_shape]);
  // Hardcoded shape size, always assume input is reshape to a 4-dims (NCHW or NHWC) tensor.
  std::vector<size_t> output_shape =
	  expandParamIfNeeded<uint32_t, size_t>(shapeHandle, 4);

  glow::NodeValue output = f_->createReshape("reshape", input, output_shape);

  addGlowNodeValue(outputs[0], output);
}

void PyTorchModelLoader::loadAdaptiveAvgPool2d(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  assert(inputs.size() == 2);
  assert(outputs.size() == 1);

  // Indexes of aten::adaptive_avg_pool2d inputs.
  struct Inputs {
    enum {
      input = 0, // NCHW
      output_size = 1,
    };
  };

  // Glow expects inputs to be in NHWC but PyTorch keeps them in NCHW so we
  // tranpose them.
  glow::NodeValue input = getGlowNodeValue(inputs[Inputs::input]);
  input = f_->createTranspose("adaptive_avg_pool2d_input_transposed", input,
                              NCHW2NHWC);

  glow::Handle<int32_t> outputSizeHandle =
      getGlowConstantHandle<int32_t>(inputs[Inputs::output_size]);
  std::vector<uint32_t> outputSize =
      expandParamIfNeeded<int32_t, uint32_t>(outputSizeHandle, 2);

  auto idim = glow::ShapeNHWC(input.dims());
  auto outTy = f_->getParent()->uniqueTypeWithNewShape(
      input.getType(), {idim.n, outputSize[0], outputSize[1], idim.c});

  glow::NodeValue output =
      f_->createAdaptiveAvgPool("adaptive_avg_pool2d", input, outTy);
  output = f_->createTranspose("adaptive_avg_pool2d_output_transposed", output,
                               NHWC2NCHW);
  addGlowNodeValue(outputs[0], output);
}

void PyTorchModelLoader::loadConstant(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  assert(inputs.size() == 0);
  assert(outputs.size() == 1);
  (void)inputs;

  auto optionalIValue = torch::jit::toIValue(outputs[0]);
  assert(optionalIValue.has_value() && "Constants should have IValue outputs");
  const torch::jit::IValue iVal = *optionalIValue;

  glow::Module *mod = f_->getParent();

  glow::Tensor t;

  if (iVal.isInt()) {
    int64_t val = iVal.toInt();
    t.reset(glow::ElemKind::Int32ITy, {1});
    t.getHandle<int32_t>().raw(0) = to32Bit(val);
  } else if (iVal.isIntList()) {
    const auto vals = iVal.toIntListRef();
    if (vals.empty()) {
      return;
    }
    t.reset(glow::ElemKind::Int32ITy, {vals.size()});
    auto handle = t.getHandle<int32_t>();
    for (size_t i = 0; i < vals.size(); ++i) {
      handle.raw(i) = to32Bit(vals[i]);
    }
  } else if (iVal.isDouble()) {
    double val = iVal.toDouble();
    t.reset(glow::ElemKind::FloatTy, {1});
    t.getHandle<float>().raw(0) = to32Bit(val);
  } else if (iVal.isDoubleList()) {
    const auto vals = iVal.toDoubleListRef();
    if (vals.empty()) {
      return;
    }
    t.reset(glow::ElemKind::FloatTy, {vals.size()});
    auto handle = t.getHandle<float>();
    for (size_t i = 0; i < vals.size(); ++i) {
      handle.raw(i) = to32Bit(vals[i]);
    }
  } else if (iVal.isBool()) {
    bool val = iVal.toBool();
    t.reset(glow::ElemKind::BoolTy, {1});
    t.getHandle<bool>().raw(0) = val;
  } else if (iVal.isBoolList()) {
    const auto &valsList = iVal.toBoolList();
    const auto &vals = c10::impl::toVector(valsList);
    if (vals.empty()) {
      return;
    }
    t.reset(glow::ElemKind::BoolTy, {vals.size()});
    auto handle = t.getHandle<bool>();
    for (size_t i = 0; i < vals.size(); ++i) {
      handle.raw(i) = vals[i];
    }
  } else if (iVal.isTensor()) {
    assert(false && "not yet implemented tensor constants");
  } else if (iVal.isNone()) {
    // Nothing to do for None
    return;
  } else {
    assert(false && "Unsupported constant type");
  }

  glow::Constant *glowConstant = mod->createConstant("constant", std::move(t));
  addGlowNodeValue(outputs[0], glowConstant->getOutput());
}

void PyTorchModelLoader::populateNodeLoaderMapping() {
  nodeLoaderMapping_[at::Symbol::fromQualString("prim::Constant")] =
      [this](const torch::jit::Node *node) { return loadConstant(node); };

  nodeLoaderMapping_[at::Symbol::fromQualString("aten::mul")] =
      [this](const torch::jit::Node *node) { return loadMul(node); };
  nodeLoaderMapping_[at::Symbol::fromQualString("aten::mul_")] =
      [this](const torch::jit::Node *node) { return loadMul(node); };

  nodeLoaderMapping_[at::Symbol::fromQualString("aten::div")] =
      [this](const torch::jit::Node *node) { return loadDiv(node); };
  nodeLoaderMapping_[at::Symbol::fromQualString("aten::div_")] =
      [this](const torch::jit::Node *node) { return loadDiv(node); };

  nodeLoaderMapping_[at::Symbol::fromQualString("aten::add")] =
      [this](const torch::jit::Node *node) { return loadAdd(node); };
  nodeLoaderMapping_[at::Symbol::fromQualString("aten::add_")] =
      [this](const torch::jit::Node *node) { return loadAdd(node); };

  nodeLoaderMapping_[at::Symbol::fromQualString("aten::sub")] =
      [this](const torch::jit::Node *node) { return loadSub(node); };
  nodeLoaderMapping_[at::Symbol::fromQualString("aten::sub_")] =
      [this](const torch::jit::Node *node) { return loadSub(node); };

  nodeLoaderMapping_[at::Symbol::fromQualString("aten::relu")] =
      [this](const torch::jit::Node *node) { return loadRelu(node); };
  nodeLoaderMapping_[at::Symbol::fromQualString("aten::relu_")] =
      [this](const torch::jit::Node *node) { return loadRelu(node); };

  nodeLoaderMapping_[at::Symbol::fromQualString("aten::_convolution")] =
      [this](const torch::jit::Node *node) { return loadConvolution(node); };

  nodeLoaderMapping_[at::Symbol::fromQualString("aten::batch_norm")] =
      [this](const torch::jit::Node *node) { return loadBatchNorm(node); };

  nodeLoaderMapping_[at::Symbol::fromQualString("aten::max_pool2d")] =
      [this](const torch::jit::Node *node) { return loadMaxPool2d(node); };

  nodeLoaderMapping_[at::Symbol::fromQualString("aten::adaptive_avg_pool2d")] =
      [this](const torch::jit::Node *node) {
        return loadAdaptiveAvgPool2d(node);
      };

  nodeLoaderMapping_[at::Symbol::fromQualString("aten::reshape")] =
	  [this](const torch::jit::Node *node) { return loadReshape(node); };


}

void PyTorchModelLoader::loadNode(const torch::jit::Node *node) {
  auto kind = node->kind();
  assert(nodeLoaderMapping_.count(kind));
  return nodeLoaderMapping_.at(kind)(node);
}

/// knownSymbols contains the list of jit Symbols that are known to
/// PyTorchModelLoader. This is populated the first time \ref
/// PyTorchModelLoader::isNodeSupported is called and referenced on subsequent
/// calls.
static std::unordered_set<torch::jit::Symbol> knownSymbols;

// static
bool PyTorchModelLoader::isNodeSupported(const torch::jit::Node *node) {
  static std::once_flag flag;
  std::call_once(flag, [&]() {
    PyTorchModelLoader loader;
    for (const auto &kv : loader.nodeLoaderMapping_) {
      knownSymbols.insert(kv.first);
    }
  });
  return knownSymbols.count(node->kind());
}

void PyTorchModelLoader::load() {
  assert(f_ == nullptr && "This model loader has already been used.");

  static std::atomic<size_t> nextFuncId{0};
  f_ = mod_->createFunction(
      glow::strFormat("PyTorchFunction_%lu", nextFuncId++));

  auto subgraphInputValues = subgraph_->inputs();

  assert(inputs_->size() == subgraphInputValues.size());

  for (size_t i = 0; i < subgraphInputValues.size(); ++i) {
    torch::jit::Value *inputValue = subgraphInputValues[i];
    const c10::IValue inputIValue = inputs_->at(i);
    inputValue->inferTypeFrom(inputIValue.toTensor());
    inputPlaceholders_.push_back(loadValue(inputValue));
  }

  // Nodes are topologically sorted.
  for (auto node : subgraph_->nodes()) {
    loadNode(node);
  }

  for (torch::jit::Value *output : subgraph_->outputs()) {
    glow::NodeValue outputNodeValue = getGlowNodeValue(output);
    auto *save = f_->createSave("save", outputNodeValue);
    outputPlaceholders_.push_back(save->getPlaceholder());
  }
}

PyTorchModelLoader::PyTorchModelLoader() { populateNodeLoaderMapping(); }

PyTorchModelLoader::PyTorchModelLoader(glow::Module *mod,
                                       torch::jit::Graph *subgraph,
                                       at::ArrayRef<torch::jit::IValue> *inputs)
    : mod_(mod), subgraph_(subgraph), inputs_(inputs) {
  populateNodeLoaderMapping();
}
