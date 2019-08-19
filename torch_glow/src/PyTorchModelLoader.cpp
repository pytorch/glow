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

#include <ATen/ATen.h>
#include <torch/csrc/jit/ir.h>

namespace glow {

namespace {

/// \returns a corresponding Glow Type for a given PyTorch CompleteTensorType \p
/// ptType.
inline glow::Type ptTypeToGlowType(const c10::ProfiledTensorType &ptType) {
  // TODO: get correct ElemKind
  DCHECK_EQ(*ptType.scalarType(), at::kFloat)
      << "Only float type supported currently.";

  const auto concreteSizes = ptType.sizes().concrete_sizes().value();

  std::vector<size_t> dims;
  for (const auto &size : concreteSizes) {
    dims.push_back(static_cast<size_t>(size));
  }

  return glow::Type(glow::ElemKind::FloatTy, dims);
}

/// \returns a Glow tensor with the same type and underlying data as the given
/// PyTorch tensor \p ptT.
llvm::Expected<glow::Tensor> ptTensorToGlowTensor(const at::Tensor &ptT) {
  // TODO: get correct ElemKind.
  DCHECK_EQ(ptT.scalar_type(), at::kFloat)
      << "Only float tensors supported currently.";
  std::vector<size_t> dims;
  for (const auto d : ptT.sizes()) {
    dims.push_back(d);
  }

  glow::Type ty(glow::ElemKind::FloatTy, dims);
  glow::Tensor glowT(ptT.data_ptr(), &ty);

  return glowT;
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

/// Indexes of aten::_convolution inputs.
struct ConvInputs {
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

/// Indexes of aten::batch_norm inputs.
struct BNInputs {
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

/// Indexes of aten::avg_pool2d inputs.
struct AvgPoolInputs {
  enum {
    input = 0,
    kernel_size = 1,
    stride = 2,
    padding = 3,
    ceil_mode = 4,
    count_include_pad = 5,
    divisor_override = 6,
  };
};

/// Indexes of aten::max_pool2d inputs.
struct MaxPoolInputs {
  enum {
    input = 0, // NCHW
    kernel_size = 1,
    stride = 2,
    padding = 3,
    dilation = 4,
    ceil_mode = 5,
  };
};

/// Indexes of aten::adaptive_avg_pool2d inputs.
struct AdaptiveAvgPoolInputs {
  enum {
    input = 0, // NCHW
    output_size = 1,
  };
};

/// Indexes of aten::linear inputs.
struct LinearInputs {
  enum {
    input = 0,
    weights = 1,
    bias = 2,
  };
};

/// Indexes of aten::prelu inputs.
struct PReluInputs {
  enum {
    input = 0,
    weight = 1,
  };
};

/// Indexes of aten::softmax inputs.
struct SoftMaxInputs {
  enum {
    input = 0,
    dim = 1,
    dtype = 2,
  };
};

/// Indexes of aten::flatten inputs.
struct FlattenInputs {
  enum {
    input = 0,
    start_dim = 1,
    end_dim = 2,
  };
};

/// Indexes of aten::topk inputs.
struct TopKInputs {
  enum {
    input = 0,
    k = 1,
    dim = 2,
    largest = 3,
    sorted = 4,
  };
};
} // namespace

c10::ScalarType PyTorchModelLoader::convertGlowType(glow::TypeRef ty) {
  switch (ty->getElementType()) {
  case ElemKind::FloatTy:
    return at::kFloat;
  case ElemKind::Float16Ty:
    return at::kHalf;
  case ElemKind::Int32ITy:
    return at::kInt;
  case ElemKind::Int64ITy:
    return at::kLong;
  case ElemKind::BoolTy:
    return at::kBool;
  case ElemKind::UInt8FusedQTy:
  case ElemKind::UInt8FusedFP16QTy:
  case ElemKind::Int8QTy:
  case ElemKind::UInt8QTy:
  case ElemKind::Int16QTy:
  case ElemKind::Int32QTy:
    LOG(DFATAL) << "Not supported yet.";
    return at::kLong;
  }
}

// static
const PyTorchModelLoader::MappingOfMemberFunctions &
PyTorchModelLoader::getSymbolsMapping() {
  /// Static map of the set of PyTorch symbols to load, the PyTorchModelLoader
  /// for loading these symbols, and the set of inputs that should be considered
  /// immutable between inference invocations by Glow and loaded as Constants
  /// instead of Placeholders.
  static auto symbolLoaderMapping = MappingOfMemberFunctions(
      {{{"prim::Constant"}, &PyTorchModelLoader::loadConstant, {}},
       {{"aten::mul", "aten::mul_"}, &PyTorchModelLoader::loadMul, {}},
       {{"aten::div", "aten::div_"}, &PyTorchModelLoader::loadDiv, {}},
       {{"aten::add", "aten::add_"}, &PyTorchModelLoader::loadAdd, {}},
       {{"aten::sigmoid", "aten::sigmoid_"},
        &PyTorchModelLoader::loadSigmoid,
        {}},
       {{"aten::sub", "aten::sub_"}, &PyTorchModelLoader::loadSub, {}},
       {{"aten::relu", "aten::relu_"}, &PyTorchModelLoader::loadRelu, {}},
       {{"aten::t, aten::t_"}, &PyTorchModelLoader::loadTranspose, {}},
       {{"aten::min"}, &PyTorchModelLoader::loadMin, {}},
       {{"aten::max"}, &PyTorchModelLoader::loadMax, {}},
       {{"aten::exp"}, &PyTorchModelLoader::loadExp, {}},
       {{"aten::sqrt", "aten::sqrt_"}, &PyTorchModelLoader::loadSqrt, {}},
       {{"aten::reciprocal", "aten::reciprocal_"},
        &PyTorchModelLoader::loadReciprocal,
        {}},
       {{"aten::adaptive_avg_pool2d"},
        &PyTorchModelLoader::loadAdaptiveAvgPool2d,
        {AdaptiveAvgPoolInputs::output_size}},
       {{"aten::linear"},
        &PyTorchModelLoader::loadLinear,
        {LinearInputs::weights, LinearInputs::bias}},
       {{"aten::_convolution"},
        &PyTorchModelLoader::loadConvolution,
        {
            ConvInputs::weights,
            ConvInputs::bias,
            ConvInputs::stride,
            ConvInputs::padding,
            ConvInputs::dilation,
            ConvInputs::transposed,
            ConvInputs::output_padding,
            ConvInputs::groups,
            ConvInputs::benchmark,
            ConvInputs::deterministic,
            ConvInputs::cudnn_enabled,
        }},
       {{"aten::batch_norm"},
        &PyTorchModelLoader::loadBatchNorm,
        {
            BNInputs::weights,
            BNInputs::bias,
            BNInputs::running_mean,
            BNInputs::running_var,
            BNInputs::training,
            BNInputs::momentum,
            BNInputs::eps,
            BNInputs::cuddnn_enabled,
        }},
       {{"aten::max_pool2d"},
        &PyTorchModelLoader::loadMaxPool2d,
        {
            MaxPoolInputs::kernel_size,
            MaxPoolInputs::stride,
            MaxPoolInputs::padding,
            MaxPoolInputs::dilation,
            MaxPoolInputs::ceil_mode,
        }},
       {{"aten::avg_pool2d"},
        &PyTorchModelLoader::loadAvgPool2d,
        {
            AvgPoolInputs::kernel_size,
            AvgPoolInputs::stride,
            AvgPoolInputs::padding,
            AvgPoolInputs::ceil_mode,
            AvgPoolInputs::count_include_pad,
            AvgPoolInputs::divisor_override,
        }},
       {{"aten::mm"}, &PyTorchModelLoader::loadMatMul, {}},
       {{"aten::flatten"},
        &PyTorchModelLoader::loadFlatten,
        {
            FlattenInputs::start_dim,
            FlattenInputs::end_dim,
        }},
       {{"aten::prelu"},
        &PyTorchModelLoader::loadPRelu,
        {
            PReluInputs::weight,
        }},
       {{"aten::softmax"},
        &PyTorchModelLoader::loadSoftMax,
        {
            SoftMaxInputs::dim,
            SoftMaxInputs::dtype,
        }},
       {{"aten::topk"},
        &PyTorchModelLoader::loadTopK,
        {
            TopKInputs::k,
            TopKInputs::dim,
            TopKInputs::largest,
            TopKInputs::sorted,
        }}});

  return symbolLoaderMapping;
}

// static
bool PyTorchModelLoader::isNodeSupported(const torch::jit::Node *node) {
  const auto &mapping = getSymbolsMapping();
  return mapping.count(node->kind()) != 0;
}

llvm::Expected<PyTorchModelLoader::ValueMap>
PyTorchModelLoader::createConstantPhReplacements(
    const torch::jit::Node *ptNode,
    const std::unordered_set<size_t> &constInputs) {
  const auto inputs = ptNode->inputs();
  PyTorchModelLoader::ValueMap valueMapReplacements;

  for (size_t i = 0; i < inputs.size(); i++) {
    // Skip inputs that are not marked as constant input values that should be
    // represented as Constants in Glow.
    if (!constInputs.count(i)) {
      continue;
    }

    // Skip node inputs that don't have a Glow value.
    if (!hasGlowNodeValue(inputs[i])) {
      continue;
    }

    glow::NodeValue phNodeValue;
    ASSIGN_VALUE_OR_RETURN_ERR(phNodeValue, getGlowNodeValue(inputs[i]));

    glow::Placeholder *ph =
        llvm::dyn_cast<glow::Placeholder>(phNodeValue.getNode());

    // Skip node inputs that aren't placeholders.
    if (!ph) {
      continue;
    }

    size_t inputIndex = inputPlaceholdersReverseIndex_.at(ph);

    const auto inputIVal = inputs_.at(inputIndex);

    glow::Constant *glowConstant;
    ASSIGN_VALUE_OR_RETURN_ERR(glowConstant,
                               createGlowConstantFromIValue(inputIVal));

    assert(glowConstant &&
           "Failed to create a Glow Constant from PyTorch input.");

    valueMapReplacements[inputs[i]] = glowConstant->getOutput();
  }

  return valueMapReplacements;
}

llvm::Error PyTorchModelLoader::loadNode(const torch::jit::Node *node,
                                         bool weightFreezingEnabled) {
  const auto &mapping = getSymbolsMapping();
  auto it = mapping.find(node->kind());

  RETURN_ERR_IF_NOT(it != mapping.end(),
                    glow::strFormat("Node kind %s is not supported by Glow",
                                    node->kind().toDisplayString()));
  if (weightFreezingEnabled) {
    ValueMap valueMapReplacements;
    ASSIGN_VALUE_OR_RETURN_ERR(
        valueMapReplacements,
        createConstantPhReplacements(node, it->second.constInputs));

    for (auto &kv : valueMapReplacements) {
      std::swap(valueMapReplacements[kv.first], valueMap_[kv.first]);
    }

    auto err = (this->*it->second.loadFn)(node);

    for (auto &kv : valueMapReplacements) {
      std::swap(valueMapReplacements[kv.first], valueMap_[kv.first]);
    }

    return err;
  } else {
    return (this->*it->second.loadFn)(node);
  }
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
  auto glowType =
      ptTypeToGlowType(*value->type()->expect<at::ProfiledTensorType>());
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

  glow::MulNode *glowNode =
      F_.createNodeWithBroadcast<glow::MulNode>("mul", /*axis*/ -1, lhs, rhs);

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

  glow::DivNode *glowNode =
      F_.createNodeWithBroadcast<glow::DivNode>("div", /*axis*/ -1, lhs, rhs);

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

  glow::AddNode *glowNode =
      F_.createNodeWithBroadcast<glow::AddNode>("add", /*axis*/ -1, lhs, rhs);
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

  glow::SubNode *glowNode =
      F_.createNodeWithBroadcast<glow::SubNode>("sub", /*axis*/ -1, lhs, rhs);
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

llvm::Error PyTorchModelLoader::loadSigmoid(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 1, outputs, 1));

  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(input, getGlowNodeValue(inputs[0]));

  glow::SigmoidNode *glowNode = F_.createSigmoid("sigmoid", input);
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

  // Glow expects conv inputs to be in NHWC but PyTorch keeps them in NCHW so
  // we tranpose them.
  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(input,
                             getGlowNodeValue(inputs[ConvInputs::input]));

  input = F_.createTranspose("conv_input_transposed", input, NCHW2NHWC);
  glow::ShapeNHWC inputShape(input.dims());

  // Glow expects conv weights to be in CRSK but PyTorch keeps them in CKRS
  // so we tranpose them. C - output_depth, R - filter_height, S -
  // filter_width, K - input_depth.
  glow::NodeValue weights;
  ASSIGN_VALUE_OR_RETURN_ERR(weights,
                             getGlowNodeValue(inputs[ConvInputs::weights]));
  weights = F_.createTranspose("conv_weights_transposed", weights, NCHW2NHWC);
  glow::ShapeNHWC weightsShape(weights.dims());

  // If a bias was provided then use it otherwise create a 0 bias.
  glow::NodeValue bias;
  if (hasGlowNodeValue(inputs[ConvInputs::bias])) {
    ASSIGN_VALUE_OR_RETURN_ERR(bias,
                               getGlowNodeValue(inputs[ConvInputs::bias]));
  } else {
    glow::Tensor biasT(glow::ElemKind::FloatTy, {weightsShape.n});
    biasT.zero();
    glow::Constant *biasConstant =
        F_.getParent()->createConstant("conv_bias", std::move(biasT));
    bias = biasConstant->getOutput();
  }

  auto stridesHandle = glow::Handle<int32_t>::createInvalidHandle();
  ASSIGN_VALUE_OR_RETURN_ERR(stridesHandle, getGlowConstantHandle<int32_t>(
                                                inputs[ConvInputs::stride]));
  std::vector<uint32_t> strides;
  RETURN_IF_ERR(expandParamIfNeeded(stridesHandle, 2, strides));

  auto padsHandle = glow::Handle<int32_t>::createInvalidHandle();
  ASSIGN_VALUE_OR_RETURN_ERR(
      padsHandle, getGlowConstantHandle<int32_t>(inputs[ConvInputs::padding]));
  uint32_t pad;
  RETURN_IF_ERR(contractParamIfNeeded(padsHandle, pad));
  std::vector<uint32_t> pads = {pad, pad, pad, pad};

  auto dilationHandle = glow::Handle<int32_t>::createInvalidHandle();
  ASSIGN_VALUE_OR_RETURN_ERR(dilationHandle, getGlowConstantHandle<int32_t>(
                                                 inputs[ConvInputs::dilation]));
  uint32_t dilation;
  RETURN_IF_ERR(contractParamIfNeeded(dilationHandle, dilation));

  // Don't support transposed convolutions yet.
  auto transposedHandle = glow::Handle<bool>::createInvalidHandle();
  ASSIGN_VALUE_OR_RETURN_ERR(
      transposedHandle,
      getGlowConstantHandle<bool>(inputs[ConvInputs::transposed]));
  RETURN_ERR_IF_NOT(transposedHandle.size() == 1 && !transposedHandle.raw(0),
                    "Transposed convolutions not supported.");

  auto groupsHandle = glow::Handle<int32_t>::createInvalidHandle();
  ASSIGN_VALUE_OR_RETURN_ERR(
      groupsHandle, getGlowConstantHandle<int32_t>(inputs[ConvInputs::groups]));
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

llvm::Error PyTorchModelLoader::loadLinear(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  assert(inputs.size() == 3);
  assert(outputs.size() == 1);
  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(input,
                             getGlowNodeValue(inputs[LinearInputs::input]));
  glow::NodeValue weights;
  ASSIGN_VALUE_OR_RETURN_ERR(weights,
                             getGlowNodeValue(inputs[LinearInputs::weights]));

  // Currently in Pytorch => Glow, we only translate t()+addmm(), t()+mm() and
  // t()+matmul() to linear, Which means we need a transpose anyway, and also
  // since addmm and mm only works for 2-D, we dont need to consider other
  // transpose implementation.
  auto *weights_t = F_.createTranspose("weights_t", weights, {1, 0});
  auto *mul = F_.createMatMul("linear_mul", input, weights_t->getResult());

  if (hasGlowNodeValue(inputs[LinearInputs::bias])) {
    glow::NodeValue bias;
    ASSIGN_VALUE_OR_RETURN_ERR(bias,
                               getGlowNodeValue(inputs[LinearInputs::bias]));
    auto *output = F_.createAdd("linear_add", mul->getResult(), bias);
    return addGlowNodeValue(outputs[0], output->getResult());
  } else {
    return addGlowNodeValue(outputs[0], mul->getResult());
  }
}

llvm::Error PyTorchModelLoader::loadBatchNorm(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 9, outputs, 1));

  // Don't support training yet.
  auto trainingHandle = glow::Handle<bool>::createInvalidHandle();
  ASSIGN_VALUE_OR_RETURN_ERR(
      trainingHandle, getGlowConstantHandle<bool>(inputs[BNInputs::training]));
  RETURN_ERR_IF_NOT(trainingHandle.size() == 1, "Training must be 1D vector.");
  RETURN_ERR_IF_NOT(trainingHandle.raw(0) == false,
                    "Training must be scalar with false value.");

  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(input, getGlowNodeValue(inputs[BNInputs::input]));
  RETURN_ERR_IF_NOT(
      input.dims().size() == 4,
      glow::strFormat("Number input dimensions must be equal to 4, got %lu",
                      input.dims().size()));

  size_t numChannels = input.dims()[1];

  glow::NodeValue weights;
  if (hasGlowNodeValue(inputs[BNInputs::weights])) {
    ASSIGN_VALUE_OR_RETURN_ERR(weights,
                               getGlowNodeValue(inputs[BNInputs::weights]));
  } else {
    glow::Tensor weightsT(glow::ElemKind::FloatTy, {numChannels});
    weightsT.init(glow::Tensor::InitKind::Broadcast, 1,
                  F_.getParent()->getPRNG());
    glow::Constant *weightsConstant = F_.getParent()->createConstant(
        "batchnorm_weights", std::move(weightsT));
    weights = weightsConstant->getOutput();
  }

  glow::NodeValue bias;
  if (hasGlowNodeValue(inputs[BNInputs::bias])) {
    ASSIGN_VALUE_OR_RETURN_ERR(bias, getGlowNodeValue(inputs[BNInputs::bias]));
  } else {
    glow::Tensor biasT(glow::ElemKind::FloatTy, {numChannels});
    biasT.zero();
    glow::Constant *biasConstant =
        F_.getParent()->createConstant("batchnorm_bias", std::move(biasT));
    bias = biasConstant->getOutput();
  }

  glow::NodeValue mean;
  ASSIGN_VALUE_OR_RETURN_ERR(mean,
                             getGlowNodeValue(inputs[BNInputs::running_mean]));
  glow::NodeValue var;
  ASSIGN_VALUE_OR_RETURN_ERR(var,
                             getGlowNodeValue(inputs[BNInputs::running_var]));

  auto momentumHandle = glow::Handle<float>::createInvalidHandle();
  ASSIGN_VALUE_OR_RETURN_ERR(
      momentumHandle, getGlowConstantHandle<float>(inputs[BNInputs::momentum]));
  RETURN_ERR_IF_NOT(momentumHandle.size() == 1, "Momentum must be 1D vector.");
  float momentum = momentumHandle.raw(0);

  auto epsilonHandle = glow::Handle<float>::createInvalidHandle();
  ASSIGN_VALUE_OR_RETURN_ERR(
      epsilonHandle, getGlowConstantHandle<float>(inputs[BNInputs::eps]));
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

  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(input,
                             getGlowNodeValue(inputs[MaxPoolInputs::input]));
  input = F_.createTranspose("maxpool2d_input_transposed", input, NCHW2NHWC);

  auto kernelsHandle = glow::Handle<int32_t>::createInvalidHandle();
  ASSIGN_VALUE_OR_RETURN_ERR(
      kernelsHandle,
      getGlowConstantHandle<int32_t>(inputs[MaxPoolInputs::kernel_size]));
  std::vector<uint32_t> kernels;
  RETURN_IF_ERR(expandParamIfNeeded(kernelsHandle, 2, kernels));

  auto padsHandle = glow::Handle<int32_t>::createInvalidHandle();
  ASSIGN_VALUE_OR_RETURN_ERR(padsHandle, getGlowConstantHandle<int32_t>(
                                             inputs[MaxPoolInputs::padding]));
  std::vector<uint32_t> padsPair;
  RETURN_IF_ERR(expandParamIfNeeded(padsHandle, 2, padsPair));
  std::vector<uint32_t> pads = {padsPair[0], padsPair[1], padsPair[0],
                                padsPair[1]};

  // Stride defaults to kernel_size.
  std::vector<uint32_t> strides;
  if (hasGlowNodeValue(inputs[2])) {
    auto stridesHandle = glow::Handle<int32_t>::createInvalidHandle();
    ASSIGN_VALUE_OR_RETURN_ERR(
        stridesHandle,
        getGlowConstantHandle<int32_t>(inputs[MaxPoolInputs::stride]));
    RETURN_IF_ERR(expandParamIfNeeded(stridesHandle, 2, strides));
  } else {
    strides = kernels;
  }

  // Glow doesn't support maxpool dilation.
  auto dilationHandle = glow::Handle<int32_t>::createInvalidHandle();
  ASSIGN_VALUE_OR_RETURN_ERR(
      dilationHandle,
      getGlowConstantHandle<int32_t>(inputs[MaxPoolInputs::dilation]));
  for (size_t i = 0; i < dilationHandle.size(); ++i) {
    RETURN_ERR_IF_NOT(
        dilationHandle.raw(i) == 1,
        glow::strFormat("Dilation value must be equal to 1, got %d",
                        dilationHandle.raw(i)));
  }

  // Glow doesn't support maxpool ceil mode.
  auto ceilModeHandle = glow::Handle<bool>::createInvalidHandle();
  ASSIGN_VALUE_OR_RETURN_ERR(
      ceilModeHandle,
      getGlowConstantHandle<bool>(inputs[MaxPoolInputs::ceil_mode]));
  RETURN_ERR_IF_NOT(ceilModeHandle.size() == 1, "ceilMode must be 1D vector.");
  RETURN_ERR_IF_NOT(ceilModeHandle.raw(0) == false,
                    "ceilMode must be scalar with false value.");

  glow::MaxPoolNode *mp =
      F_.createMaxPool("maxpool2d", input, kernels, strides, pads);
  glow::NodeValue output = mp->getResult();
  output = F_.createTranspose("maxpool2d_output_transposed", output, NHWC2NCHW);
  return addGlowNodeValue(outputs[0], output);
}

llvm::Error PyTorchModelLoader::loadAvgPool2d(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 7, outputs, 1));

  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(input,
                             getGlowNodeValue(inputs[AvgPoolInputs::input]));
  input = F_.createTranspose("avgpool2d_input_transposed", input, NCHW2NHWC);

  auto kernelsHandle = glow::Handle<int32_t>::createInvalidHandle();
  ASSIGN_VALUE_OR_RETURN_ERR(
      kernelsHandle,
      getGlowConstantHandle<int32_t>(inputs[AvgPoolInputs::kernel_size]));
  std::vector<uint32_t> kernels;
  RETURN_IF_ERR(expandParamIfNeeded(kernelsHandle, 2, kernels));

  auto padsHandle = glow::Handle<int32_t>::createInvalidHandle();
  ASSIGN_VALUE_OR_RETURN_ERR(padsHandle, getGlowConstantHandle<int32_t>(
                                             inputs[AvgPoolInputs::padding]));
  std::vector<uint32_t> padsPair;
  RETURN_IF_ERR(expandParamIfNeeded(padsHandle, 2, padsPair));
  std::vector<uint32_t> pads = {padsPair[0], padsPair[1], padsPair[0],
                                padsPair[1]};

  // Stride defaults to kernel_size.
  std::vector<uint32_t> strides;
  if (hasGlowNodeValue(inputs[AvgPoolInputs::stride])) {
    auto stridesHandle = glow::Handle<int32_t>::createInvalidHandle();
    ASSIGN_VALUE_OR_RETURN_ERR(
        stridesHandle,
        getGlowConstantHandle<int32_t>(inputs[AvgPoolInputs::stride]));
    RETURN_IF_ERR(expandParamIfNeeded(stridesHandle, 2, strides));
  } else {
    strides = kernels;
  }

  // Glow doesn't support avgpool ceil mode.
  auto ceilModeHandle = glow::Handle<bool>::createInvalidHandle();
  ASSIGN_VALUE_OR_RETURN_ERR(
      ceilModeHandle,
      getGlowConstantHandle<bool>(inputs[AvgPoolInputs::ceil_mode]));
  RETURN_ERR_IF_NOT(ceilModeHandle.size() == 1, "ceilMode must be 1D vector.");
  RETURN_ERR_IF_NOT(ceilModeHandle.raw(0) == false, "ceilMode not supported.");

  // Glow always includes zero-padding in the averaging calculation.
  auto countIncludePadHandle = glow::Handle<bool>::createInvalidHandle();
  ASSIGN_VALUE_OR_RETURN_ERR(
      countIncludePadHandle,
      getGlowConstantHandle<bool>(inputs[AvgPoolInputs::count_include_pad]));
  RETURN_ERR_IF_NOT(countIncludePadHandle.size() == 1,
                    "countIncludePad must be 1D vector.");
  RETURN_ERR_IF_NOT(countIncludePadHandle.raw(0) == true,
                    "countIncludePad must be scalar with true value.");

  glow::AvgPoolNode *ap =
      F_.createAvgPool("avgpool2d", input, kernels, strides, pads);
  glow::NodeValue output = ap->getResult();
  output = F_.createTranspose("avgpool2d_output_transposed", output, NHWC2NCHW);
  return addGlowNodeValue(outputs[0], output);
}

llvm::Error
PyTorchModelLoader::loadAdaptiveAvgPool2d(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 2, outputs, 1));

  // Glow expects inputs to be in NHWC but PyTorch keeps them in NCHW so we
  // tranpose them.
  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(
      input, getGlowNodeValue(inputs[AdaptiveAvgPoolInputs::input]));

  input = F_.createTranspose("adaptive_avg_pool2d_input_transposed", input,
                             NCHW2NHWC);

  auto outputSizeHandle = glow::Handle<int32_t>::createInvalidHandle();
  ASSIGN_VALUE_OR_RETURN_ERR(outputSizeHandle,
                             getGlowConstantHandle<int32_t>(
                                 inputs[AdaptiveAvgPoolInputs::output_size]));
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

llvm::Error PyTorchModelLoader::loadMatMul(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 2, outputs, 1));

  glow::NodeValue lhs;
  ASSIGN_VALUE_OR_RETURN_ERR(lhs, getGlowNodeValue(inputs[0]));
  glow::NodeValue rhs;
  ASSIGN_VALUE_OR_RETURN_ERR(rhs, getGlowNodeValue(inputs[1]));

  auto *glowNode = F_.createMatMul("MatMul", lhs, rhs);
  return addGlowNodeValue(outputs[0], glowNode);
}

llvm::Error PyTorchModelLoader::loadPRelu(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 2, outputs, 1));

  glow::NodeValue in;
  ASSIGN_VALUE_OR_RETURN_ERR(in, getGlowNodeValue(inputs[PReluInputs::input]));
  glow::NodeValue weight;
  ASSIGN_VALUE_OR_RETURN_ERR(weight,
                             getGlowNodeValue(inputs[PReluInputs::weight]));

  // Do broadcasting.
  auto targetDim = in.dims();
  auto weightDim = weight.dims();
  RETURN_ERR_IF_NOT(
      weightDim.size() == targetDim.size() || weightDim.size() == 1,
      glow::strFormat(
          "Weight dimensions must be 1, or the number of channels, got %lu",
          weightDim.size()));
  // Sets the axis of each inputs so that the trailing-most dimensions of
  // input tensors and the target shape are aligned.
  int axis = targetDim.size() - weight.dims().size();
  auto *slope = F_.createBroadcast("broadcast", weight, targetDim, axis);
  auto *glowNode = F_.createPRELU("prelu", in, slope);
  return addGlowNodeValue(outputs[0], glowNode);
}

llvm::Error PyTorchModelLoader::loadSoftMax(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 3, outputs, 1));

  glow::NodeValue in;
  ASSIGN_VALUE_OR_RETURN_ERR(in,
                             getGlowNodeValue(inputs[SoftMaxInputs::input]));

  // Dim.
  auto dimHandle = glow::Handle<int32_t>::createInvalidHandle();
  ASSIGN_VALUE_OR_RETURN_ERR(
      dimHandle, getGlowConstantHandle<int32_t>(inputs[SoftMaxInputs::dim]));
  RETURN_ERR_IF_NOT(dimHandle.size() == 1, "dim must be 1D vector.");
  glow::unsigned_t dim = static_cast<glow::unsigned_t>(dimHandle.raw(0));

  // Dtype (optional).
  if (hasGlowNodeValue(inputs[SoftMaxInputs::dtype])) {
    auto dtypeHandle = glow::Handle<int32_t>::createInvalidHandle();
    ASSIGN_VALUE_OR_RETURN_ERR(dtypeHandle, getGlowConstantHandle<int32_t>(
                                                inputs[SoftMaxInputs::dtype]));
    RETURN_ERR_IF_NOT(dtypeHandle.size() == 1, "dim must be 1D vector.");
    glow::unsigned_t dtype = static_cast<glow::unsigned_t>(dtypeHandle.raw(0));
    RETURN_ERR_IF_NOT(
        dtype == static_cast<glow::unsigned_t>(at::ScalarType::Float),
        glow::strFormat(
            "Dtype parameter must have value torch::float(1), got %u", dtype));
  }

  auto selected = F_.getParent()->createConstant(glow::ElemKind::Int64ITy,
                                                 {in.dims()[0], 1}, "selected");

  auto *FN = F_.createFlatten("reshapeInput", in, dim);
  auto *SM = F_.createSoftMax("SoftMax", FN, selected);
  auto origInDims = in.getType()->dims();
  auto *glowNode = F_.createReshape("reshapeOutput", SM, origInDims);
  return addGlowNodeValue(outputs[0], glowNode);
}

llvm::Error PyTorchModelLoader::loadFlatten(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 3, outputs, 1));

  glow::NodeValue in;
  ASSIGN_VALUE_OR_RETURN_ERR(in,
                             getGlowNodeValue(inputs[FlattenInputs::input]));

  auto startHandle = glow::Handle<int32_t>::createInvalidHandle();
  ASSIGN_VALUE_OR_RETURN_ERR(
      startHandle,
      getGlowConstantHandle<int32_t>(inputs[FlattenInputs::start_dim]));
  RETURN_ERR_IF_NOT(startHandle.size() == 1, "start_dim must be 1D vector.");
  uint32_t startDim = static_cast<uint32_t>(startHandle.raw(0));

  auto endHandle = glow::Handle<int32_t>::createInvalidHandle();
  ASSIGN_VALUE_OR_RETURN_ERR(endHandle, getGlowConstantHandle<int32_t>(
                                            inputs[FlattenInputs::end_dim]));
  RETURN_ERR_IF_NOT(endHandle.size() == 1, "end_dim must be 1D vector.");
  uint32_t endDim = static_cast<uint32_t>(endHandle.raw(0));
  RETURN_ERR_IF_NOT(endDim == -1, "only -1 value for end_dim is supported.");

  auto xDim = glow::flattenCdr(in.dims(), startDim);
  auto *glowNode = F_.createReshape("flatten", in, {xDim.first, xDim.second});
  return addGlowNodeValue(outputs[0], glowNode);
}

llvm::Error PyTorchModelLoader::loadTopK(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 5, outputs, 2));

  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(input,
                             getGlowNodeValue(inputs[TopKInputs::input]));

  auto kHandle = glow::Handle<int32_t>::createInvalidHandle();
  ASSIGN_VALUE_OR_RETURN_ERR(
      kHandle, getGlowConstantHandle<int32_t>(inputs[TopKInputs::k]));
  RETURN_ERR_IF_NOT(kHandle.size() == 1, "k must be 1D vector.");
  glow::unsigned_t k = static_cast<glow::unsigned_t>(kHandle.raw(0));

  auto dimHandle = glow::Handle<int32_t>::createInvalidHandle();
  ASSIGN_VALUE_OR_RETURN_ERR(
      dimHandle, getGlowConstantHandle<int32_t>(inputs[TopKInputs::dim]));
  RETURN_ERR_IF_NOT(dimHandle.size() == 1, "dim must be 1D vector.");

  auto largestHandle = glow::Handle<bool>::createInvalidHandle();
  ASSIGN_VALUE_OR_RETURN_ERR(
      largestHandle, getGlowConstantHandle<bool>(inputs[TopKInputs::largest]));
  RETURN_ERR_IF_NOT(largestHandle.size() == 1, "largest must be 1D vector.");

  auto sortedHandle = glow::Handle<bool>::createInvalidHandle();
  ASSIGN_VALUE_OR_RETURN_ERR(
      sortedHandle, getGlowConstantHandle<bool>(inputs[TopKInputs::sorted]));
  RETURN_ERR_IF_NOT(sortedHandle.size() == 1, "sorted must be 1D vector.");

  auto *glowNode = F_.createTopK("TopK", input, k);

  RETURN_IF_ERR(addGlowNodeValue(outputs[0], glowNode->getValues()));
  RETURN_IF_ERR(addGlowNodeValue(outputs[1], glowNode->getIndices()));
  return llvm::Error::success();
}

llvm::Error PyTorchModelLoader::loadConstant(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 0, outputs, 1));

  auto optionalIValue = torch::jit::toIValue(outputs[0]);
  RETURN_ERR_IF_NOT(optionalIValue.has_value(),
                    "Constants should have IValue outputs.");
  const torch::jit::IValue iVal = *optionalIValue;

  glow::Constant *constant;
  ASSIGN_VALUE_OR_RETURN_ERR(constant, createGlowConstantFromIValue(iVal));

  if (!constant) {
    return llvm::Error::success();
  }

  return addGlowNodeValue(outputs[0], constant->getOutput());
}

llvm::Expected<glow::Constant *>
PyTorchModelLoader::createGlowConstantFromIValue(
    const torch::jit::IValue &iVal) {
  glow::Tensor t;

  if (iVal.isInt()) {
    int64_t val = iVal.toInt();
    t.reset(glow::ElemKind::Int32ITy, {1});
    ASSIGN_VALUE_OR_RETURN_ERR(t.getHandle<int32_t>().raw(0), to32Bit(val));
  } else if (iVal.isIntList()) {
    const auto vals = iVal.toIntListRef();
    if (vals.empty()) {
      return nullptr;
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
      return nullptr;
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
      return nullptr;
    }
    t.reset(glow::ElemKind::BoolTy, {vals.size()});
    auto handle = t.getHandle<bool>();
    for (size_t i = 0; i < vals.size(); ++i) {
      handle.raw(i) = vals[i];
    }
  } else if (iVal.isTensor()) {
    const at::Tensor &val = iVal.toTensor();
    ASSIGN_VALUE_OR_RETURN_ERR(t, ptTensorToGlowTensor(val));
  } else if (iVal.isNone()) {
    // Nothing to do for None
    return nullptr;
  } else {
    RETURN_ERR("Unsupported constant type.");
  }

  return F_.getParent()->createConstant("constant", std::move(t));
}

/*static*/
llvm::Error PyTorchModelLoader::loadJITGraph(
    glow::Function &F, torch::jit::Graph &subgraph,
    at::ArrayRef<torch::jit::IValue> &inputs,
    std::vector<glow::Placeholder *> &inputPlaceholders,
    std::vector<glow::Placeholder *> &outputPlaceholders,
    const PyTorchLoaderSettings &settings) {
  llvm::Error error = llvm::Error::success();
  MARK_ERR_CHECKED(error);
  PyTorchModelLoader loader(F, subgraph, inputs, inputPlaceholders,
                            outputPlaceholders, error, settings);
  return error;
}

PyTorchModelLoader::PyTorchModelLoader(
    glow::Function &F, torch::jit::Graph &subgraph,
    at::ArrayRef<torch::jit::IValue> &inputs,
    std::vector<glow::Placeholder *> &inputPlaceholders,
    std::vector<glow::Placeholder *> &outputPlaceholders, llvm::Error &error,
    const PyTorchLoaderSettings &settings)
    : F_(F), inputs_(inputs) {
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
      inputPlaceholdersReverseIndex_[PH] = i;
    }
  }

  bool weightFreezingEnabled = settings.weightFreezingEnabled;

  // Nodes are topologically sorted.
  for (auto node : subgraph.nodes()) {
    if ((error = loadNode(node, weightFreezingEnabled))) {
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

} // namespace glow
