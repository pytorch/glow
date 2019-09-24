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
/// For the quantized PyTorch ops, the activations are quantized to uint_8.
/// In Glow, the activations are quantized to int_8. Therefore, for the offset
/// read from quantized pytorch model, we need to subtract 128(i.e. INT8_MIN) to
/// make the activations becomes int8_t.
/// For Glow: -128 <= orig_fp32/scale_1 + offset_1 <= 127
/// For PyTorch: 0 <= orig_fp32/scale_2 + offset_2 <= 255
/// Therefore, we can make scale_1 == scale_2, and offset_1 = offset2 - 128
const int32_t OFFSETSHIFT = 128;

/// Downcast a double to a float.
Expected<float> to32Bit(double val) {
  RETURN_ERR_IF_NOT(val <= std::numeric_limits<float>::max() ||
                        val >= std::numeric_limits<float>::lowest(),
                    glow::strFormat("Value %f is out of limit.", val));
  return Expected<float>(static_cast<float>(val));
}

/// Unwrap a Expected and call to32Bit(double) or any contained return
/// Error.
Expected<float> to32Bit(Expected<double> expectedVal) {
  if (expectedVal) {
    return to32Bit(*expectedVal);
  } else {
    return expectedVal.takeError();
  }
}

/// Given a GlowIValue \p glowIVal and \p size, will return an std::vector
/// of the GlowIValue in the case it's a IntList or Tuple of Ints checking there
/// are exactly size elements or if the GlowIValue is an Int then it will
/// replicate it size times then return that.
Expected<std::vector<int64_t>> expandIntIValIfNeeded(const GlowIValue &glowIVal,
                                                     size_t size) {
  // If the GlowIValue is a single int then make size copies of it.
  if (glowIVal.isInt()) {
    std::vector<int64_t> out;
    int64_t elem;
    ASSIGN_VALUE_OR_RETURN_ERR(elem, glowIVal.toInt());
    for (size_t i = 0; i < size; ++i) {
      out.push_back(elem);
    }
    return out;
  }

  // If the GlowIValue is an IntList then check that its size is size then
  // return it.
  else if (glowIVal.isIntList()) {
    const std::vector<int64_t> *listPtr;
    ASSIGN_VALUE_OR_RETURN_ERR(listPtr, glowIVal.toIntList());
    RETURN_ERR_IF_NOT(
        listPtr->size() == size,
        strFormat("Expected a list of size %lu but found a list of size %lu",
                  size, listPtr->size()));
    return *listPtr;
  }

  // If the GlowIValue is a Tuple with size number of elements and all elements
  // are ints then put those ints in a vector.
  else if (glowIVal.isTuple()) {
    const std::vector<GlowIValue> *tuplePtr;
    ASSIGN_VALUE_OR_RETURN_ERR(tuplePtr, glowIVal.toTuple());
    RETURN_ERR_IF_NOT(
        tuplePtr->size() == size,
        strFormat("Expected a tuple of size %lu but found a tuple of size %lu",
                  size, tuplePtr->size()));
    std::vector<int64_t> out;
    for (const auto &ival : *tuplePtr) {
      int64_t elem;
      ASSIGN_VALUE_OR_RETURN_ERR(elem, ival.toInt());
      out.push_back(elem);
    }
    return out;
  }

  // Any other type of GlowIValue is invalid.
  else {
    RETURN_ERR(
        strFormat("Unexpected GlowIValue type: %s", glowIVal.getTagString()));
  }
}

/// Unwrap Expected<GlowIValue *> and call
/// expandIntIValIfNeeded(GlowIValue), propagates any Errors.
Expected<std::vector<int64_t>>
expandIntIValIfNeeded(Expected<GlowIValue *> expectedGlowIVal, size_t size) {
  if (expectedGlowIVal) {
    return expandIntIValIfNeeded(**expectedGlowIVal, size);
  } else {
    return expectedGlowIVal.takeError();
  }
}

/// Given a GlowIValue \p glowIVal, \returns if the GlowIValue is an Int return
/// it's value, if it's a IntList or Tuple of Ints then check that all elements
/// are the same then return the first one.
Expected<int64_t> contractIntIValIfNeeded(const GlowIValue &glowIVal) {
  if (glowIVal.isInt()) {
    return glowIVal.toInt();
  }

  // If the GlowIValue is an int list then check that its size is size then
  // return it.
  else if (glowIVal.isIntList()) {
    const std::vector<int64_t> *listPtr;
    ASSIGN_VALUE_OR_RETURN_ERR(listPtr, glowIVal.toIntList());
    RETURN_ERR_IF_NOT(!listPtr->empty(), "Unexpected empty list");
    int64_t value = (*listPtr)[0];
    for (size_t i = 1; i < listPtr->size(); ++i) {
      int64_t elem = (*listPtr)[i];
      RETURN_ERR_IF_NOT(value == elem,
                        "Expected all elements of list to be the same.");
    }
    return value;
  }

  // If the GlowIValue is a tuple with size number of elements and all elements
  // are ints then put those ints in a vector.
  else if (glowIVal.isTuple()) {
    const std::vector<GlowIValue> *tuplePtr;
    ASSIGN_VALUE_OR_RETURN_ERR(tuplePtr, glowIVal.toTuple());
    RETURN_ERR_IF_NOT(!tuplePtr->empty(), "Unexpected empty tuple");
    int64_t value;
    ASSIGN_VALUE_OR_RETURN_ERR(value, (*tuplePtr)[0].toInt());
    for (size_t i = 1; i < tuplePtr->size(); ++i) {
      int64_t elem;
      ASSIGN_VALUE_OR_RETURN_ERR(elem, (*tuplePtr)[i].toInt());
      RETURN_ERR_IF_NOT(value == elem,
                        "Expected all elements of tuple to be the same.");
    }
    return value;
  }

  // Any other type of GlowIValue is invalid.
  else {
    RETURN_ERR(
        strFormat("Unexpected GlowIValue type: %s", glowIVal.getTagString()));
  }
}

/// Unwrap a Expected<GlowIValue *> \p expectedGlowIVal and call
/// contractIntIValIfNeeded(GlowIValue), propogate any Errors.
Expected<int64_t>
contractIntIValIfNeeded(Expected<GlowIValue *> expectedGlowIVal) {
  if (expectedGlowIVal) {
    return contractIntIValIfNeeded(**expectedGlowIVal);
  } else {
    return expectedGlowIVal.takeError();
  }
}

/// Unwrap a Expected<GlowIValue *> \p expectedIVal and call toDouble,
/// propogate any Errors.
Expected<double> iValToDouble(Expected<GlowIValue *> expectedIVal) {
  if (expectedIVal) {
    return (*expectedIVal)->toDouble();
  } else {
    return expectedIVal.takeError();
  }
}

/// Unwrap a Expected<GlowIValue *> \p expectedIVal and call toInt,
/// propogate any Errors.
Expected<int64_t> iValToInt(Expected<GlowIValue *> expectedIVal) {
  if (expectedIVal) {
    return (*expectedIVal)->toInt();
  } else {
    return expectedIVal.takeError();
  }
}

/// Unwrap a Expected<GlowIValue *> \p expectedIVal and call toBool,
/// propogate any Errors.
Expected<bool> iValToBool(Expected<GlowIValue *> expectedIVal) {
  if (expectedIVal) {
    return (*expectedIVal)->toBool();
  } else {
    return expectedIVal.takeError();
  }
}

/// Unwrap a Expected<GlowIValue *> \p expectedIVal and call toIntList,
/// propogate any Errors.
Expected<std::vector<int64_t> *>
iValToIntList(Expected<GlowIValue *> expectedIVal) {
  if (expectedIVal) {
    return (*expectedIVal)->toIntList();
  } else {
    return expectedIVal.takeError();
  }
}

/// Given Node inputs and outputs, check the expected sizes. Negative size
/// indicates that the size should be equal to or greater than that size (for
/// example -2 means at least 2).
template <typename T>
Error checkInputAndOutputSizes(const T &inputs, int64_t inputsSize,
                               const T &outputs, int64_t outputsSize) {
  if (inputsSize >= 0) {
    RETURN_ERR_IF_NOT(inputs.size() == inputsSize,
                      glow::strFormat("Expected exactly %lu inputs, got %lu.",
                                      (size_t)inputsSize, inputs.size()));
  } else {
    inputsSize = inputsSize * -1;
    RETURN_ERR_IF_NOT(inputs.size() >= inputsSize,
                      glow::strFormat("Expected at least %lu inputs, got %lu.",
                                      (size_t)inputsSize, inputs.size()));
  }

  if (outputsSize >= 0) {
    RETURN_ERR_IF_NOT(outputs.size() == outputsSize,
                      glow::strFormat("Expected exactly %lu outputs, got %lu.",
                                      (size_t)outputsSize, outputs.size()));
  } else {
    outputsSize = outputsSize * -1;
    RETURN_ERR_IF_NOT(outputs.size() >= outputsSize,
                      glow::strFormat("Expected at least %lu outputs, got %lu.",
                                      (size_t)outputsSize, outputs.size()));
  }
  return Error::success();
}

/// Given a vector \p original containing elements of some type, \returns a
/// vector of each element cast to another type T.
template <typename T, typename OriginalT>
std::vector<T> castVector(const std::vector<OriginalT> &original) {
  std::vector<T> out;
  out.reserve(original.size());
  for (const auto &elem : original) {
    out.push_back(static_cast<T>(elem));
  }
  return out;
}

/// Unwrap a Expected<std::vector<>> \p originalExpected and calls
/// castVector() with the contents, propagates any Errors.
template <typename T, typename OriginalT>
Expected<std::vector<T>>
castVector(Expected<std::vector<OriginalT>> originalExpected) {
  if (originalExpected) {
    return castVector<T>(*originalExpected);
  } else {
    return originalExpected.takeError();
  }
}

/// Unwrap a Expected<OriginalT> \p originalExpected and calls
/// static_cast() with the contents, propagates any Errors.
template <typename T, typename OriginalT>
Expected<T> static_cast_expected(Expected<OriginalT> originalExpected) {
  if (originalExpected) {
    return static_cast<T>(*originalExpected);
  } else {
    return originalExpected.takeError();
  }
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

/// Indices of aten::clamp inputs.
struct ClampInputs {
  enum {
    input = 0,
    min = 1,
    max = 2,
  };
};

/// Indexes of aten::adaptive_avg_pool2d inputs.
struct AdaptiveAvgPoolInputs {
  enum {
    input = 0, // NCHW
    output_size = 1,
  };
};

/// Indexes of quantized::add inputs.
struct QuantizedAddInputs {
  enum {
    lhs = 0,
    rhs = 1,
    scale = 2,
    offset = 3,
  };
};

/// Indexes of aten::quantize_per_tensor inputs.
struct QuantizeInputs {
  enum {
    input = 0,
    scale = 1,
    offset = 2,
    dtype = 3,
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

/// Indexes of aten::size inputs.
struct SizeInputs {
  enum {
    input = 0,
    dim = 1,
  };
};

/// Indexes of aten::reshape inputs.
struct ReshapeInputs {
  enum {
    input = 0,
    shape = 1,
  };
};
} // namespace

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
       {{"aten::sub", "aten::sub_"}, &PyTorchModelLoader::loadSub, {}},
       {{"aten::sigmoid", "aten::sigmoid_"},
        &PyTorchModelLoader::loadSigmoid,
        {}},
       {{"aten::relu", "aten::relu_"}, &PyTorchModelLoader::loadRelu, {}},
       {{"aten::t", "aten::t_"}, &PyTorchModelLoader::loadTranspose, {}},
       {{"aten::min"}, &PyTorchModelLoader::loadMin, {}},
       {{"aten::max"}, &PyTorchModelLoader::loadMax, {}},
       {{"aten::exp"}, &PyTorchModelLoader::loadExp, {}},
       {{"aten::sqrt", "aten::sqrt_"}, &PyTorchModelLoader::loadSqrt, {}},
       {{"aten::clamp"},
        &PyTorchModelLoader::loadClamp,
        {
            ClampInputs::min,
            ClampInputs::max,
        }},
       {{"quantized::add"},
        &PyTorchModelLoader::loadQuantizedAdd,
        {QuantizedAddInputs::scale, QuantizedAddInputs::offset}},
       {{"aten::quantize_per_tensor"},
        &PyTorchModelLoader::loadQuantize,
        {QuantizeInputs::scale, QuantizeInputs::offset, QuantizeInputs::dtype}},
       {{"aten::dequantize"}, &PyTorchModelLoader::loadDequantize, {}},
       {{"aten::size"}, &PyTorchModelLoader::loadSize, {SizeInputs::dim}},
       // TODO: use -1 to freeze all inputs
       {{"prim::ListConstruct"}, &PyTorchModelLoader::loadListConstruct, {}},
       {{"aten::reciprocal", "aten::reciprocal_"},
        &PyTorchModelLoader::loadReciprocal,
        {}},
       {{"aten::adaptive_avg_pool2d"},
        &PyTorchModelLoader::loadAdaptiveAvgPool2d,
        {AdaptiveAvgPoolInputs::output_size}},
       {{"aten::linear"},
        &PyTorchModelLoader::loadLinear,
        {LinearInputs::weights, LinearInputs::bias}},
       {{"aten::reshape"},
        &PyTorchModelLoader::loadReshape,
        {ReshapeInputs::shape}},
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
bool PyTorchModelLoader::isNodeSupported(const torch::jit::Node *ptNode) {
  const auto &mapping = getSymbolsMapping();
  return mapping.count(ptNode->kind()) != 0;
}

Error PyTorchModelLoader::freezeWeights(const torch::jit::Node *ptNode) {
  const auto &mapping = getSymbolsMapping();
  const auto it = mapping.find(ptNode->kind());

  RETURN_ERR_IF_NOT(it != mapping.end(),
                    glow::strFormat("Node kind %s is not supported by Glow",
                                    ptNode->kind().toDisplayString()));

  const auto &inputsToFreeze = it->second.inputsToFreeze;

  const auto inputs = ptNode->inputs();

  std::vector<size_t> frozenInputIndices;

  for (size_t i = 0; i < inputs.size(); i++) {
    // Skip inputs that are not marked to be frozen
    if (!inputsToFreeze.count(i)) {
      continue;
    }

    // Skip node inputs that don't have a Glow NodeValue.
    if (!hasGlowNodeValueForValue(inputs[i])) {
      continue;
    }

    glow::NodeValue phNodeValue;
    ASSIGN_VALUE_OR_RETURN_ERR(phNodeValue,
                               getGlowNodeValueForValue(inputs[i]));

    glow::Placeholder *ph =
        llvm::dyn_cast<glow::Placeholder>(phNodeValue.getNode());

    // Skip inputs that aren't placeholders.
    if (!ph) {
      continue;
    }

    RETURN_ERR_IF_NOT(
        inputPlaceholdersReverseIndex_.count(ph),
        "Trying to freeze a NodeValue that did not come from the stack.");

    size_t inputIndex = inputPlaceholdersReverseIndex_.at(ph);

    const auto inputIVal = inputs_.at(inputIndex);

    GlowIValue glowIVal;
    RETURN_IF_ERR(glowIVal.fromIValue(inputIVal));

    removeValueMapping(inputs[i]);
    RETURN_IF_ERR(
        addValueMapping(inputs[i], std::move(glowIVal), /*wasFrozen*/ true));

    if (frozenInputIndices_) {
      frozenInputIndices_->insert(inputIndex);
    }
  }
  return Error::success();
}

Error PyTorchModelLoader::loadNode(const torch::jit::Node *node) {
  const auto &mapping = getSymbolsMapping();
  auto it = mapping.find(node->kind());

  RETURN_ERR_IF_NOT(it != mapping.end(),
                    glow::strFormat("Node kind %s is not supported by Glow",
                                    node->kind().toDisplayString()));
  return (this->*it->second.loadFn)(node);
}

Error PyTorchModelLoader::addValueMapping(const torch::jit::Value *value,
                                          glow::NodeValue nodeValue,
                                          bool wasFrozen) {

  ValueMapping mapping(std::move(nodeValue), wasFrozen);
  auto p = valueMap_.emplace(value, std::move(mapping));

  RETURN_ERR_IF_NOT(p.second, glow::strFormat("Value %s is already mapped",
                                              value->debugNameBase().c_str()));
  return Error::success();
}

void PyTorchModelLoader::removeValueMapping(const torch::jit::Value *value) {
  valueMap_.erase(value);
}

Error PyTorchModelLoader::addValueMapping(const torch::jit::Value *value,
                                          glow::GlowIValue glowIValue,
                                          bool wasFrozen) {
  glow::Constant *glowConstant = nullptr;
  if (glowIValue.isTensor()) {
    glow::Tensor *t;
    ASSIGN_VALUE_OR_RETURN_ERR(t, glowIValue.toTensor());
    glowConstant = F_.getParent()->createConstant("constant", std::move(*t));
    RETURN_IF_ERR(addValueMapping(value, glowConstant->getOutput(), wasFrozen));
  } else {
    ValueMapping mapping(std::move(glowIValue));
    auto p = valueMap_.emplace(value, std::move(mapping));

    RETURN_ERR_IF_NOT(p.second,
                      glow::strFormat("Value %s is already mapped",
                                      value->debugNameBase().c_str()));
  }

  return Error::success();
}

bool PyTorchModelLoader::hasGlowNodeValueForValue(
    const torch::jit::Value *value) const {
  const auto it = valueMap_.find(value);
  if (it == valueMap_.end()) {
    return false;
  }
  const auto mappingType = it->second.getMappingType();
  return mappingType != ValueMappingType::IValue;
}

bool PyTorchModelLoader::hasGlowIValueForValue(const torch::jit::Value *value,
                                               bool ignoreNones) const {
  const auto it = valueMap_.find(value);
  if (it == valueMap_.end()) {
    return false;
  }
  const auto mappingType = it->second.getMappingType();

  if (mappingType != ValueMappingType::IValue) {
    return false;
  }

  if (ignoreNones) {
    // Already checked ValueMappingType above.
    const auto *glowIVal = EXIT_ON_ERR(it->second.getMappedGlowIValue());
    return !glowIVal->isNone();
  }

  return true;
}

Expected<glow::NodeValue>
PyTorchModelLoader::getGlowNodeValueForValue(const torch::jit::Value *value) {
  auto it = valueMap_.find(value);
  if (it == valueMap_.end()) {
    RETURN_ERR(glow::strFormat("No mapping found fo Value %s",
                               value->debugNameBase().c_str()));
  }
  auto &mappingValue = it->second;
  if (mappingValue.getMappingType() == ValueMappingType::IValue) {
    RETURN_ERR(glow::strFormat("Did not find a NodeValue mapping for Value %s",
                               value->debugNameBase().c_str()));
  }

  return mappingValue.getMappedNodeValue();
}

Expected<glow::GlowIValue *>
PyTorchModelLoader::getGlowIValueForValue(const torch::jit::Value *value) {
  auto it = valueMap_.find(value);
  if (it == valueMap_.end()) {
    RETURN_ERR(glow::strFormat("No mapping found fo Value %s",
                               value->debugNameBase().c_str()));
  }
  auto &mappingValue = it->second;
  if (mappingValue.getMappingType() != ValueMappingType::IValue) {
    RETURN_ERR(glow::strFormat("Did not find a IValue mapping for Value %s",
                               value->debugNameBase().c_str()));
  }
  return mappingValue.getMappedGlowIValue();
}

Error PyTorchModelLoader::loadQuantizedAdd(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 4, outputs, 1));

  glow::NodeValue lhs;
  ASSIGN_VALUE_OR_RETURN_ERR(
      lhs, getGlowNodeValueForValue(inputs[QuantizedAddInputs::lhs]));
  glow::NodeValue rhs;
  ASSIGN_VALUE_OR_RETURN_ERR(
      rhs, getGlowNodeValueForValue(inputs[QuantizedAddInputs::rhs]));

  // scale
  float outScale;
  ASSIGN_VALUE_OR_RETURN_ERR(outScale, iValToDouble(getGlowIValueForValue(
                                           inputs[QuantizedAddInputs::scale])));

  // offset
  int32_t outOffset;
  ASSIGN_VALUE_OR_RETURN_ERR(
      outOffset,
      iValToInt(getGlowIValueForValue(inputs[QuantizedAddInputs::offset])));

  TypeRef inputType = lhs.getType();
  auto outDims = inputType->dims();
  auto outTy = F_.getParent()->uniqueType(ElemKind::Int8QTy, outDims, outScale,
                                          outOffset - OFFSETSHIFT);

  glow::AddNode *qan = F_.createAdd("quantized_add", outTy, lhs, rhs);
  return addValueMapping(outputs[0], qan->getResult());
}

Error PyTorchModelLoader::loadMul(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 2, outputs, 1));

  glow::NodeValue lhs;
  ASSIGN_VALUE_OR_RETURN_ERR(lhs, getGlowNodeValueForValue(inputs[0]));
  glow::NodeValue rhs;
  ASSIGN_VALUE_OR_RETURN_ERR(rhs, getGlowNodeValueForValue(inputs[1]));

  glow::MulNode *glowNode =
      F_.createNodeWithBroadcast<glow::MulNode>("mul", /*axis*/ -1, lhs, rhs);

  return addValueMapping(outputs[0], glowNode->getResult());
}

Error PyTorchModelLoader::loadDiv(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 2, outputs, 1));

  glow::NodeValue lhs;
  ASSIGN_VALUE_OR_RETURN_ERR(lhs, getGlowNodeValueForValue(inputs[0]));
  glow::NodeValue rhs;
  ASSIGN_VALUE_OR_RETURN_ERR(rhs, getGlowNodeValueForValue(inputs[1]));

  glow::DivNode *glowNode =
      F_.createNodeWithBroadcast<glow::DivNode>("div", /*axis*/ -1, lhs, rhs);

  return addValueMapping(outputs[0], glowNode->getResult());
}

Error PyTorchModelLoader::loadAdd(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 3, outputs, 1));

  // TODO: extend this to allow non-constant scalars.
  int64_t scalar;
  ASSIGN_VALUE_OR_RETURN_ERR(scalar,
                             iValToInt(getGlowIValueForValue(inputs[2])));
  RETURN_ERR_IF_NOT(scalar == 1,
                    glow::strFormat("Scalar must have value equal 1."));

  glow::NodeValue lhs;
  ASSIGN_VALUE_OR_RETURN_ERR(lhs, getGlowNodeValueForValue(inputs[0]));
  glow::NodeValue rhs;
  ASSIGN_VALUE_OR_RETURN_ERR(rhs, getGlowNodeValueForValue(inputs[1]));

  glow::AddNode *glowNode =
      F_.createNodeWithBroadcast<glow::AddNode>("add", /*axis*/ -1, lhs, rhs);
  return addValueMapping(outputs[0], glowNode->getResult());
}

Error PyTorchModelLoader::loadSub(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 3, outputs, 1));

  // TODO: extend this to allow non-constant scalars.
  int64_t scalar;
  ASSIGN_VALUE_OR_RETURN_ERR(scalar,
                             iValToInt(getGlowIValueForValue(inputs[2])));
  RETURN_ERR_IF_NOT(scalar == 1,
                    glow::strFormat("Scalar must have value equal 1."));

  glow::NodeValue lhs;
  ASSIGN_VALUE_OR_RETURN_ERR(lhs, getGlowNodeValueForValue(inputs[0]));
  glow::NodeValue rhs;
  ASSIGN_VALUE_OR_RETURN_ERR(rhs, getGlowNodeValueForValue(inputs[1]));

  glow::SubNode *glowNode =
      F_.createNodeWithBroadcast<glow::SubNode>("sub", /*axis*/ -1, lhs, rhs);
  return addValueMapping(outputs[0], glowNode->getResult());
}

Error PyTorchModelLoader::loadMax(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 2, outputs, 1));

  glow::NodeValue lhs;
  ASSIGN_VALUE_OR_RETURN_ERR(lhs, getGlowNodeValueForValue(inputs[0]));
  glow::NodeValue rhs;
  ASSIGN_VALUE_OR_RETURN_ERR(rhs, getGlowNodeValueForValue(inputs[1]));

  glow::MaxNode *glowNode = F_.createMax("max", lhs, rhs);
  return addValueMapping(outputs[0], glowNode->getResult());
}

Error PyTorchModelLoader::loadSize(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 2, outputs, 1));

  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(
      input, getGlowNodeValueForValue(inputs[SizeInputs::input]));

  int64_t dim;
  ASSIGN_VALUE_OR_RETURN_ERR(
      dim, iValToInt(getGlowIValueForValue(inputs[SizeInputs::dim])));

  if (dim == -1) {
    dim = input.dims().size() - 1;
  }

  // Index must a valid index of input.
  RETURN_ERR_IF_NOT(input.dims().size() - 1 >= dim,
                    strFormat("Trying to access the size of dim %d of a tensor "
                              "with only %d dimensions",
                              (int32_t)dim, (int32_t)input.dims().size()));

  GlowIValue glowIVal;
  glowIVal.fromInt(input.dims()[dim]);

  return addValueMapping(outputs[0], std::move(glowIVal));
}

Error PyTorchModelLoader::loadListConstruct(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  // Requires -1 because this requires at least one input.
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, -1, outputs, 1));

  // Get the Tag of the first input to use for the whole list.
  GlowIValue *firstInputIVal;
  ASSIGN_VALUE_OR_RETURN_ERR(firstInputIVal, getGlowIValueForValue(inputs[0]));
  auto tag = firstInputIVal->getTag();

  GlowIValue glowIVal;
  if (tag == GlowIValue::Tag::Double) {
    std::vector<double> doubles;
    for (size_t i = 0; i < inputs.size(); ++i) {
      double x;
      ASSIGN_VALUE_OR_RETURN_ERR(
          x, iValToDouble(getGlowIValueForValue(inputs[i])));
      doubles.push_back(x);
    }
    glowIVal.fromDoubleList(std::move(doubles));
  } else if (tag == GlowIValue::Tag::Int) {
    std::vector<int64_t> ints;
    for (size_t i = 0; i < inputs.size(); ++i) {
      int x;
      ASSIGN_VALUE_OR_RETURN_ERR(x,
                                 iValToInt(getGlowIValueForValue(inputs[i])));
      ints.push_back(x);
    }
    glowIVal.fromIntList(std::move(ints));
  } else if (tag == GlowIValue::Tag::Bool) {
    std::vector<bool> bools;
    for (size_t i = 0; i < inputs.size(); ++i) {
      bool x;
      ASSIGN_VALUE_OR_RETURN_ERR(x,
                                 iValToBool(getGlowIValueForValue(inputs[i])));
      bools.push_back(x);
    }
    glowIVal.fromBoolList(std::move(bools));
  } else {
    RETURN_ERR("Encountered an unsupported GlowIValue type for ListConstruct");
  }

  return addValueMapping(outputs[0], std::move(glowIVal));
}

Error PyTorchModelLoader::loadReshape(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 2, outputs, 1));

  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(
      input, getGlowNodeValueForValue(inputs[ReshapeInputs::input]));

  std::vector<int64_t> *shape;
  ASSIGN_VALUE_OR_RETURN_ERR(shape, iValToIntList(getGlowIValueForValue(
                                        inputs[ReshapeInputs::shape])));

  int64_t totalDims = 1;
  for (size_t dim : input.dims()) {
    totalDims *= dim;
  }

  std::vector<size_t> glowShape;
  for (size_t i = 0, e = shape->size(); i < e; ++i) {
    int64_t dim = (*shape)[i];

    RETURN_ERR_IF_NOT(dim >= 1 || dim == -1,
                      "Only positive values and -1 allowed in shape");

    if (dim == -1) {
      RETURN_ERR_IF_NOT(i == e - 1,
                        "-1 in shape only allowed in last position for now.");

      dim = totalDims;
    }

    totalDims /= dim;

    glowShape.push_back(dim);
  }

  RETURN_ERR_IF_NOT(totalDims == 1, "Expect reshape to preserve total size.");

  return addValueMapping(outputs[0],
                         F_.createReshape("reshape", input, glowShape));
}

Error PyTorchModelLoader::loadRelu(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 1, outputs, 1));

  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(input, getGlowNodeValueForValue(inputs[0]));

  glow::ReluNode *glowNode = F_.createRELU("relu", input);
  return addValueMapping(outputs[0], glowNode->getResult());
}

Error PyTorchModelLoader::loadExp(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 1, outputs, 1));

  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(input, getGlowNodeValueForValue(inputs[0]));

  glow::ExpNode *glowNode = F_.createExp("exp", input);
  return addValueMapping(outputs[0], glowNode->getResult());
}

Error PyTorchModelLoader::loadSqrt(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 1, outputs, 1));

  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(input, getGlowNodeValueForValue(inputs[0]));

  glow::PowNode *glowNode = F_.createPow("sqrt", input, /*exp=*/0.5);
  return addValueMapping(outputs[0], glowNode->getResult());
}

Error PyTorchModelLoader::loadSigmoid(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 1, outputs, 1));

  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(input, getGlowNodeValueForValue(inputs[0]));

  glow::SigmoidNode *glowNode = F_.createSigmoid("sigmoid", input);
  return addValueMapping(outputs[0], glowNode->getResult());
}

Error PyTorchModelLoader::loadReciprocal(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 1, outputs, 1));

  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(input, getGlowNodeValueForValue(inputs[0]));
  glow::PowNode *glowNode = F_.createPow("reciprocal", input, /*exp=*/-1);
  return addValueMapping(outputs[0], glowNode->getResult());
}

Error PyTorchModelLoader::loadConvolution(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 12, outputs, 1));

  // Glow expects conv inputs to be in NHWC but PyTorch keeps them in NCHW so
  // we tranpose them.
  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(
      input, getGlowNodeValueForValue(inputs[ConvInputs::input]));

  input = F_.createTranspose("conv_input_transposed", input, NCHW2NHWC);
  glow::ShapeNHWC inputShape(input.dims());

  // Glow expects conv weights to be in CRSK but PyTorch keeps them in CKRS
  // so we tranpose them. C - output_depth, R - filter_height, S -
  // filter_width, K - input_depth.
  glow::NodeValue weights;
  ASSIGN_VALUE_OR_RETURN_ERR(
      weights, getGlowNodeValueForValue(inputs[ConvInputs::weights]));
  weights = F_.createTranspose("conv_weights_transposed", weights, NCHW2NHWC);
  glow::ShapeNHWC weightsShape(weights.dims());

  // If a bias was provided then use it otherwise create a 0 bias.
  glow::NodeValue bias;
  if (hasGlowNodeValueForValue(inputs[ConvInputs::bias])) {
    ASSIGN_VALUE_OR_RETURN_ERR(
        bias, getGlowNodeValueForValue(inputs[ConvInputs::bias]));
  } else {
    glow::Tensor biasT(glow::ElemKind::FloatTy, {weightsShape.n});
    biasT.zero();
    glow::Constant *biasConstant =
        F_.getParent()->createConstant("conv_bias", std::move(biasT));
    bias = biasConstant->getOutput();
  }

  std::vector<glow::unsigned_t> strides;
  ASSIGN_VALUE_OR_RETURN_ERR(
      strides, castVector<glow::unsigned_t>(expandIntIValIfNeeded(
                   getGlowIValueForValue(inputs[ConvInputs::stride]), 2)));

  glow::unsigned_t pad;
  ASSIGN_VALUE_OR_RETURN_ERR(
      pad, static_cast_expected<glow::unsigned_t>(contractIntIValIfNeeded(
               getGlowIValueForValue(inputs[ConvInputs::padding]))));
  std::vector<glow::unsigned_t> pads = {pad, pad, pad, pad};

  glow::unsigned_t dilation;
  ASSIGN_VALUE_OR_RETURN_ERR(
      dilation, static_cast_expected<glow::unsigned_t>(contractIntIValIfNeeded(
                    getGlowIValueForValue(inputs[ConvInputs::dilation]))));

  // Don't support transposed convolutions yet.
  bool transposed;
  ASSIGN_VALUE_OR_RETURN_ERR(transposed, iValToBool(getGlowIValueForValue(
                                             inputs[ConvInputs::transposed])));
  RETURN_ERR_IF_NOT(!transposed, "Transposed convolutions not supported.");

  glow::unsigned_t groups;
  ASSIGN_VALUE_OR_RETURN_ERR(
      groups, static_cast_expected<glow::unsigned_t>(iValToInt(
                  getGlowIValueForValue(inputs[ConvInputs::groups]))));

  std::vector<glow::unsigned_t> kernels = {
      static_cast<glow::unsigned_t>(weightsShape.h),
      static_cast<glow::unsigned_t>(weightsShape.w)};

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

  return addValueMapping(outputs[0], output->getResult());
}

Error PyTorchModelLoader::loadLinear(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 3, outputs, 1));

  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(
      input, getGlowNodeValueForValue(inputs[LinearInputs::input]));

  glow::NodeValue weights;
  ASSIGN_VALUE_OR_RETURN_ERR(
      weights, getGlowNodeValueForValue(inputs[LinearInputs::weights]));

  // Transpose weights before inputing into FC (TODO).
  auto wDims = weights.dims();
  std::vector<unsigned_t> shuffle{1, 0};
  weights = F_.createTranspose("fc_weights_transposed", weights, shuffle);

  glow::NodeValue bias;
  if (hasGlowNodeValueForValue(inputs[LinearInputs::bias])) {
    ASSIGN_VALUE_OR_RETURN_ERR(
        bias, getGlowNodeValueForValue(inputs[LinearInputs::bias]));
  } else {
    glow::Tensor biasT(glow::ElemKind::FloatTy, {wDims[0]});
    biasT.zero();
    glow::Constant *biasConstant =
        F_.getParent()->createConstant("linear_bias", std::move(biasT));
    weights = biasConstant->getOutput();
  }

  glow::TypeRef outTy = F_.getParent()->uniqueType(
      glow::ElemKind::FloatTy, {input.dims()[0], bias.getType()->dims()[0]});

  return addValueMapping(
      outputs[0],
      F_.createFullyConnected("linear", input, weights, bias, outTy));
}

Error PyTorchModelLoader::loadBatchNorm(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 9, outputs, 1));

  bool training;
  ASSIGN_VALUE_OR_RETURN_ERR(
      training, iValToBool(getGlowIValueForValue(inputs[BNInputs::training])));
  RETURN_ERR_IF_NOT(training == false, "Don't support BatchNorm training yet.");

  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(input,
                             getGlowNodeValueForValue(inputs[BNInputs::input]));
  RETURN_ERR_IF_NOT(
      input.dims().size() == 4,
      glow::strFormat("Number input dimensions must be equal to 4, got %lu",
                      input.dims().size()));

  size_t numChannels = input.dims()[1];

  glow::NodeValue weights;
  if (hasGlowNodeValueForValue(inputs[BNInputs::weights])) {
    ASSIGN_VALUE_OR_RETURN_ERR(
        weights, getGlowNodeValueForValue(inputs[BNInputs::weights]));
  } else {
    glow::Tensor weightsT(glow::ElemKind::FloatTy, {numChannels});
    weightsT.init(glow::Tensor::InitKind::Broadcast, 1,
                  F_.getParent()->getPRNG());
    glow::Constant *weightsConstant = F_.getParent()->createConstant(
        "batchnorm_weights", std::move(weightsT));
    weights = weightsConstant->getOutput();
  }

  glow::NodeValue bias;
  if (hasGlowNodeValueForValue(inputs[BNInputs::bias])) {
    ASSIGN_VALUE_OR_RETURN_ERR(
        bias, getGlowNodeValueForValue(inputs[BNInputs::bias]));
  } else {
    glow::Tensor biasT(glow::ElemKind::FloatTy, {numChannels});
    biasT.zero();
    glow::Constant *biasConstant =
        F_.getParent()->createConstant("batchnorm_bias", std::move(biasT));
    bias = biasConstant->getOutput();
  }

  glow::NodeValue mean;
  ASSIGN_VALUE_OR_RETURN_ERR(
      mean, getGlowNodeValueForValue(inputs[BNInputs::running_mean]));

  glow::NodeValue var;
  ASSIGN_VALUE_OR_RETURN_ERR(
      var, getGlowNodeValueForValue(inputs[BNInputs::running_var]));

  float momentum;
  ASSIGN_VALUE_OR_RETURN_ERR(
      momentum,
      to32Bit(iValToDouble(getGlowIValueForValue(inputs[BNInputs::momentum]))));

  float epsilon;
  ASSIGN_VALUE_OR_RETURN_ERR(
      epsilon,
      to32Bit(iValToDouble(getGlowIValueForValue(inputs[BNInputs::eps]))));

  // Input is in NCHW.
  glow::unsigned_t channelIdx = 1;

  glow::BatchNormalizationNode *bn =
      F_.createBatchNormalization("batchnorm", input, bias, weights, mean, var,
                                  channelIdx, epsilon, momentum);
  return addValueMapping(outputs[0], bn->getResult());
}

Error PyTorchModelLoader::loadQuantize(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 4, outputs, 1));

  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(
      input, getGlowNodeValueForValue(inputs[QuantizeInputs::input]));

  // scale
  float outScale;
  ASSIGN_VALUE_OR_RETURN_ERR(outScale, iValToDouble(getGlowIValueForValue(
                                           inputs[QuantizeInputs::scale])));

  // offset
  int32_t outOffset;
  ASSIGN_VALUE_OR_RETURN_ERR(outOffset, iValToInt(getGlowIValueForValue(
                                            inputs[QuantizeInputs::offset])));

  // dtype, we only support quantize to int8 for now
  int32_t outDtype;
  ASSIGN_VALUE_OR_RETURN_ERR(outDtype, iValToInt(getGlowIValueForValue(
                                           inputs[QuantizeInputs::dtype])));

  // Right now pytorch only has quint8 quantization support, therefore we only
  // support uint8 as well.
  RETURN_ERR_IF_NOT(outDtype == (int)at::ScalarType::QUInt8,
                    "we only support to be quantized as uint8");

  TypeRef inputType = input.getType();
  auto outDims = inputType->dims();
  auto outTy = F_.getParent()->uniqueType(ElemKind::Int8QTy, outDims, outScale,
                                          outOffset - OFFSETSHIFT);

  glow::QuantizeNode *qn = F_.createQuantize("quantize", input, outTy);

  return addValueMapping(outputs[0], qn->getResult());
}

Error PyTorchModelLoader::loadDequantize(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 1, outputs, 1));

  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(input, getGlowNodeValueForValue(inputs[0]));

  glow::DequantizeNode *dn = F_.createDequantize("dequantize", input);
  return addValueMapping(outputs[0], dn->getResult());
}

Error PyTorchModelLoader::loadMaxPool2d(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 6, outputs, 1));

  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(
      input, getGlowNodeValueForValue(inputs[MaxPoolInputs::input]));
  input = F_.createTranspose("maxpool2d_input_transposed", input, NCHW2NHWC);

  std::vector<glow::unsigned_t> kernels;
  ASSIGN_VALUE_OR_RETURN_ERR(
      kernels,
      castVector<glow::unsigned_t>(expandIntIValIfNeeded(
          getGlowIValueForValue(inputs[MaxPoolInputs::kernel_size]), 2)));

  std::vector<glow::unsigned_t> padsPair;
  ASSIGN_VALUE_OR_RETURN_ERR(
      padsPair, castVector<glow::unsigned_t>(expandIntIValIfNeeded(
                    getGlowIValueForValue(inputs[MaxPoolInputs::padding]), 2)));
  std::vector<glow::unsigned_t> pads = {padsPair[0], padsPair[1], padsPair[0],
                                        padsPair[1]};

  // Stride defaults to kernel_size.
  std::vector<glow::unsigned_t> strides;
  if (hasGlowIValueForValue(inputs[MaxPoolInputs::stride])) {
    ASSIGN_VALUE_OR_RETURN_ERR(
        strides, castVector<glow::unsigned_t>(expandIntIValIfNeeded(
                     getGlowIValueForValue(inputs[MaxPoolInputs::stride]), 2)));
  } else {
    strides = kernels;
  }

  // Glow doesn't support maxpool dilation.
  int64_t dilation;
  ASSIGN_VALUE_OR_RETURN_ERR(
      dilation, contractIntIValIfNeeded(
                    getGlowIValueForValue(inputs[MaxPoolInputs::dilation])));
  RETURN_ERR_IF_NOT(
      dilation == 1,
      "Dilation value must be equal to 1, maxpool dilation not yet supported.");

  // Glow doesn't support maxpool ceil mode.
  bool ceilMode;
  ASSIGN_VALUE_OR_RETURN_ERR(ceilMode, iValToBool(getGlowIValueForValue(
                                           inputs[MaxPoolInputs::ceil_mode])));
  RETURN_ERR_IF_NOT(ceilMode == false,
                    "ceilMode must be scalar with false value.");

  glow::MaxPoolNode *mp =
      F_.createMaxPool("maxpool2d", input, kernels, strides, pads);
  glow::NodeValue output = mp->getResult();
  output = F_.createTranspose("maxpool2d_output_transposed", output, NHWC2NCHW);
  return addValueMapping(outputs[0], output);
}

Error PyTorchModelLoader::loadAvgPool2d(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 7, outputs, 1));

  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(
      input, getGlowNodeValueForValue(inputs[AvgPoolInputs::input]));
  input = F_.createTranspose("avgpool2d_input_transposed", input, NCHW2NHWC);

  std::vector<glow::unsigned_t> kernels;
  ASSIGN_VALUE_OR_RETURN_ERR(
      kernels,
      castVector<glow::unsigned_t>(expandIntIValIfNeeded(
          getGlowIValueForValue(inputs[AvgPoolInputs::kernel_size]), 2)));

  std::vector<glow::unsigned_t> padsPair;
  ASSIGN_VALUE_OR_RETURN_ERR(
      padsPair, castVector<glow::unsigned_t>(expandIntIValIfNeeded(
                    getGlowIValueForValue(inputs[AvgPoolInputs::padding]), 2)));
  std::vector<glow::unsigned_t> pads = {padsPair[0], padsPair[1], padsPair[0],
                                        padsPair[1]};

  // Stride defaults to kernel_size.
  std::vector<glow::unsigned_t> strides;
  if (hasGlowIValueForValue(inputs[AvgPoolInputs::stride])) {
    ASSIGN_VALUE_OR_RETURN_ERR(
        strides, castVector<glow::unsigned_t>(expandIntIValIfNeeded(
                     getGlowIValueForValue(inputs[AvgPoolInputs::stride]), 2)));
  } else {
    strides = kernels;
  }

  // Glow doesn't support avgpool ceil mode.
  bool ceilMode;
  ASSIGN_VALUE_OR_RETURN_ERR(ceilMode, iValToBool(getGlowIValueForValue(
                                           inputs[AvgPoolInputs::ceil_mode])));
  RETURN_ERR_IF_NOT(ceilMode == false,
                    "ceilMode must be scalar with false value.");

  // Glow always includes zero-padding in the averaging calculation.
  bool countIncludePad;
  ASSIGN_VALUE_OR_RETURN_ERR(countIncludePad,
                             iValToBool(getGlowIValueForValue(
                                 inputs[AvgPoolInputs::count_include_pad])));
  RETURN_ERR_IF_NOT(countIncludePad, "countIncludePad must be true.");

  glow::AvgPoolNode *ap =
      F_.createAvgPool("avgpool2d", input, kernels, strides, pads);
  glow::NodeValue output = ap->getResult();
  output = F_.createTranspose("avgpool2d_output_transposed", output, NHWC2NCHW);
  return addValueMapping(outputs[0], output);
}

Error PyTorchModelLoader::loadClamp(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 3, outputs, 1));

  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(
      input, getGlowNodeValueForValue(inputs[ClampInputs::input]));

  double minDouble;
  ASSIGN_VALUE_OR_RETURN_ERR(
      minDouble, iValToDouble(getGlowIValueForValue(inputs[ClampInputs::min])));
  float min;
  ASSIGN_VALUE_OR_RETURN_ERR(min, to32Bit(minDouble));

  double maxDouble;
  ASSIGN_VALUE_OR_RETURN_ERR(
      maxDouble, iValToDouble(getGlowIValueForValue(inputs[ClampInputs::max])));
  float max;
  ASSIGN_VALUE_OR_RETURN_ERR(max, to32Bit(maxDouble));

  auto output = F_.createClip("clip", input, min, max);
  return addValueMapping(outputs[0], output);
}

Error PyTorchModelLoader::loadAdaptiveAvgPool2d(
    const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 2, outputs, 1));

  // Glow expects inputs to be in NHWC but PyTorch keeps them in NCHW so we
  // tranpose them.
  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(
      input, getGlowNodeValueForValue(inputs[AdaptiveAvgPoolInputs::input]));

  size_t inputH = input.dims()[1];
  size_t inputW = input.dims()[2];

  input = F_.createTranspose("adaptive_avg_pool2d_input_transposed", input,
                             NCHW2NHWC);

  // OutputSize defaults to size of input if not provided.
  std::vector<size_t> outputSize;
  if (hasGlowIValueForValue(inputs[AdaptiveAvgPoolInputs::output_size])) {
    ASSIGN_VALUE_OR_RETURN_ERR(
        outputSize,
        castVector<size_t>(expandIntIValIfNeeded(
            getGlowIValueForValue(inputs[AdaptiveAvgPoolInputs::output_size]),
            2)));
  } else {
    outputSize = {inputH, inputW};
  }

  auto idim = glow::ShapeNHWC(input.dims());
  auto outTy = F_.getParent()->uniqueTypeWithNewShape(
      input.getType(), {idim.n, outputSize[0], outputSize[1], idim.c});

  glow::NodeValue output =
      F_.createAdaptiveAvgPool("adaptive_avg_pool2d", input, outTy);
  output = F_.createTranspose("adaptive_avg_pool2d_output_transposed", output,
                              NHWC2NCHW);
  return addValueMapping(outputs[0], output);
}

Error PyTorchModelLoader::loadTranspose(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 1, outputs, 1));

  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(input, getGlowNodeValueForValue(inputs[0]));

  glow::NodeValue output;
  if (input.dims().size() == 1) {
    output = input;
  } else if (input.dims().size() == 2) {
    output = F_.createTranspose("transpose", input, {1, 0});
  } else {
    RETURN_ERR("Transpose requires input to have rank <= 2");
  }
  return addValueMapping(outputs[0], output);
}

Error PyTorchModelLoader::loadMin(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 2, outputs, 1));

  glow::NodeValue lhs;
  ASSIGN_VALUE_OR_RETURN_ERR(lhs, getGlowNodeValueForValue(inputs[0]));
  glow::NodeValue rhs;
  ASSIGN_VALUE_OR_RETURN_ERR(rhs, getGlowNodeValueForValue(inputs[1]));

  auto output = F_.createMin("min", lhs, rhs);
  return addValueMapping(outputs[0], output);
}

Error PyTorchModelLoader::loadMatMul(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 2, outputs, 1));

  glow::NodeValue lhs;
  ASSIGN_VALUE_OR_RETURN_ERR(lhs, getGlowNodeValueForValue(inputs[0]));
  glow::NodeValue rhs;
  ASSIGN_VALUE_OR_RETURN_ERR(rhs, getGlowNodeValueForValue(inputs[1]));

  auto *glowNode = F_.createMatMul("MatMul", lhs, rhs);
  return addValueMapping(outputs[0], glowNode);
}

Error PyTorchModelLoader::loadPRelu(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 2, outputs, 1));

  glow::NodeValue in;
  ASSIGN_VALUE_OR_RETURN_ERR(
      in, getGlowNodeValueForValue(inputs[PReluInputs::input]));
  glow::NodeValue weight;
  ASSIGN_VALUE_OR_RETURN_ERR(
      weight, getGlowNodeValueForValue(inputs[PReluInputs::weight]));

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
  return addValueMapping(outputs[0], glowNode);
}

/// TODO: check Dtype is float (optional value).
Error PyTorchModelLoader::loadSoftMax(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 3, outputs, 1));

  glow::NodeValue in;
  ASSIGN_VALUE_OR_RETURN_ERR(
      in, getGlowNodeValueForValue(inputs[SoftMaxInputs::input]));

  glow::unsigned_t dim;
  ASSIGN_VALUE_OR_RETURN_ERR(
      dim, static_cast_expected<glow::unsigned_t>(
               iValToInt(getGlowIValueForValue(inputs[SoftMaxInputs::dim]))));

  auto selected = F_.getParent()->createConstant(
      glow::ElemKind::Int64ITy, {in.dims()[0], in.dims()[1]}, "selected");

  auto *FN = F_.createFlatten("reshapeInput", in, dim);
  auto *SM = F_.createSoftMax("SoftMax", FN, selected);
  auto origInDims = in.getType()->dims();
  auto *glowNode = F_.createReshape("reshapeOutput", SM, origInDims);
  return addValueMapping(outputs[0], glowNode);
}

Error PyTorchModelLoader::loadFlatten(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 3, outputs, 1));

  glow::NodeValue in;
  ASSIGN_VALUE_OR_RETURN_ERR(
      in, getGlowNodeValueForValue(inputs[FlattenInputs::input]));

  int64_t startDim;
  ASSIGN_VALUE_OR_RETURN_ERR(startDim, iValToInt(getGlowIValueForValue(
                                           inputs[FlattenInputs::start_dim])));

  int64_t endDim;
  ASSIGN_VALUE_OR_RETURN_ERR(
      endDim, iValToInt(getGlowIValueForValue(inputs[FlattenInputs::end_dim])));
  RETURN_ERR_IF_NOT(endDim == -1, "only -1 value for end_dim is supported.");

  auto xDim = glow::flattenCdr(in.dims(), startDim);
  auto *glowNode = F_.createReshape("flatten", in, {xDim.first, xDim.second});
  return addValueMapping(outputs[0], glowNode);
}

Error PyTorchModelLoader::loadTopK(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 5, outputs, 2));

  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(
      input, getGlowNodeValueForValue(inputs[TopKInputs::input]));

  int64_t k;
  ASSIGN_VALUE_OR_RETURN_ERR(
      k, iValToInt(getGlowIValueForValue(inputs[TopKInputs::k])));

  if (hasGlowIValueForValue(inputs[TopKInputs::dim],
                            /*ignoreNones*/ true)) {
    int64_t dim;
    ASSIGN_VALUE_OR_RETURN_ERR(
        dim, iValToInt(getGlowIValueForValue(inputs[TopKInputs::dim])));
    RETURN_ERR_IF_NOT(dim != input.dims().size() - 1,
                      "topk is only supported along the last dimension");
  }

  if (hasGlowIValueForValue(inputs[TopKInputs::largest],
                            /*ignoreNones*/ true)) {
    bool largest;
    ASSIGN_VALUE_OR_RETURN_ERR(largest, iValToBool(getGlowIValueForValue(
                                            inputs[TopKInputs::largest])));
    RETURN_ERR_IF_NOT(largest, "topk is only supported with largest");
  }

  auto *glowNode = F_.createTopK("TopK", input, (unsigned_t)k);

  RETURN_IF_ERR(addValueMapping(outputs[0], glowNode->getValues()));
  RETURN_IF_ERR(addValueMapping(outputs[1], glowNode->getIndices()));
  return Error::success();
}

Error PyTorchModelLoader::loadConstant(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 0, outputs, 1));

  auto optionalIValue = torch::jit::toIValue(outputs[0]);
  RETURN_ERR_IF_NOT(optionalIValue.has_value(),
                    "Constants should have IValue outputs.");
  const torch::jit::IValue iVal = *optionalIValue;

  GlowIValue glowIVal;
  RETURN_IF_ERR(glowIVal.fromIValue(iVal));

  // Consider empty lists as not existing because for example MaxPool2d
  // requires this.
  if (glowIVal.isIntList()) {
    std::vector<int64_t> *ints;
    ASSIGN_VALUE_OR_RETURN_ERR(ints, glowIVal.toIntList());
    if (ints->empty()) {
      return Error::success();
    }
  } else if (glowIVal.isDoubleList()) {
    std::vector<double> *doubles;
    ASSIGN_VALUE_OR_RETURN_ERR(doubles, glowIVal.toDoubleList());
    if (doubles->empty()) {
      return Error::success();
    }
  } else if (glowIVal.isBoolList()) {
    std::vector<bool> *bools;
    ASSIGN_VALUE_OR_RETURN_ERR(bools, glowIVal.toBoolList());
    if (bools->empty()) {
      return Error::success();
    }
  }

  if (glowIVal.isTensor()) {
    glow::Tensor *t;
    ASSIGN_VALUE_OR_RETURN_ERR(t, glowIVal.toTensor());
    glow::Constant *glowConstant =
        F_.getParent()->createConstant("constant", std::move(*t));
    if (copyTensorMemory_) {
      glowConstant->ensureIsOwned();
    }
    RETURN_IF_ERR(addValueMapping(outputs[0], glowConstant->getOutput()));
  } else {
    RETURN_IF_ERR(addValueMapping(outputs[0], std::move(glowIVal)));
  }

  return Error::success();
}

/*static*/
Error PyTorchModelLoader::loadJITGraph(
    glow::Function &F, const torch::jit::Graph &graph,
    const at::ArrayRef<torch::jit::IValue> inputs,
    std::vector<glow::Placeholder *> &inputPlaceholders,
    std::vector<glow::Placeholder *> &outputPlaceholders,
    const PyTorchLoaderSettings &settings) {
  Error error = Error::empty();
  PyTorchModelLoader loader(F, graph, inputs, inputPlaceholders,
                            outputPlaceholders, error, settings,
                            /*frozenInputIndices*/ nullptr);
  return error;
}

PyTorchModelLoader::PyTorchModelLoader(
    glow::Function &F, const torch::jit::Graph &graph,
    const at::ArrayRef<torch::jit::IValue> inputs,
    std::vector<glow::Placeholder *> &inputPlaceholders,
    std::vector<glow::Placeholder *> &outputPlaceholders, Error &error,
    const PyTorchLoaderSettings &settings, std::set<size_t> *frozenInputIndices)
    : F_(F), inputs_(inputs), frozenInputIndices_(frozenInputIndices),
      copyTensorMemory_(false) {
  auto loadFn = [&]() -> Error {
    auto graphInputValues = graph.inputs();

    RETURN_ERR_IF_NOT(
        inputs.size() == graphInputValues.size(),
        glow::strFormat("Number of Graph inputs %lu must match the "
                        "number of provided inputs %lu.",
                        graphInputValues.size(), inputs.size()));

    // Create Glow Placeholders for inputs.
    for (size_t i = 0; i < graphInputValues.size(); ++i) {
      const torch::jit::Value *inputValue = graphInputValues[i];
      const c10::IValue inputIValue = inputs.at(i);
      GlowIValue glowIVal;
      RETURN_IF_ERR(glowIVal.fromIValue(inputIValue));
      if (glowIVal.isTensor()) {
        glow::Tensor *t;
        ASSIGN_VALUE_OR_RETURN_ERR(t, glowIVal.toTensor());
        glow::Placeholder *ph = F_.getParent()->createPlaceholder(
            &t->getType(), "input", /*isTrainable*/ false);
        RETURN_IF_ERR(addValueMapping(inputValue, ph->getOutput()));
        inputPlaceholders.push_back(ph);
        inputPlaceholdersReverseIndex_[ph] = i;
      } else {
        RETURN_IF_ERR(addValueMapping(inputValue, std::move(glowIVal)));
      }
    }

    // If weight freezing is enabled then freeze all weights. This is done
    // before any nodes are loaded so all nodes see either frozen or unfrozen
    // view of inputs in case any input is shared.
    if (settings.weightFreezingEnabled) {
      for (const auto &node : graph.nodes()) {
        RETURN_IF_ERR(freezeWeights(node));
      }
    }

    // Nodes are topologically sorted.
    for (const auto &node : graph.nodes()) {
      RETURN_IF_ERR(loadNode(node));
    }

    // Create Glow Placeholders for outputs.
    for (const torch::jit::Value *output : graph.outputs()) {
      glow::NodeValue outputNodeValue;
      // Only allow tensor outputs from Glow subgraph.
      ASSIGN_VALUE_OR_RETURN_ERR(outputNodeValue,
                                 getGlowNodeValueForValue(output));
      auto *save = F_.createSave("save", outputNodeValue);
      outputPlaceholders.push_back(save->getPlaceholder());
    }

    return Error::success();
  };

  error = loadFn();
}

/*static*/
Error PyTorchModelLoader::loadJITGraphForOnnxTraining(
    glow::Function &F, const torch::jit::Graph &graph,
    const at::ArrayRef<torch::jit::IValue> inputs,
    const at::ArrayRef<std::shared_ptr<c10::TensorType>> parameters,
    std::vector<glow::Placeholder *> &inputPlaceholders,
    std::vector<glow::Placeholder *> &outputPlaceholders) {
  Error error = Error::empty();
  PyTorchModelLoader loader(F, graph, inputs, parameters, inputPlaceholders,
                            outputPlaceholders, error);
  return error;
}

PyTorchModelLoader::PyTorchModelLoader(
    glow::Function &F, const torch::jit::Graph &graph,
    const at::ArrayRef<torch::jit::IValue> inputs,
    const at::ArrayRef<std::shared_ptr<c10::TensorType>> parameters,
    std::vector<glow::Placeholder *> &inputPlaceholders,
    std::vector<glow::Placeholder *> &outputPlaceholders, Error &error)
    : F_(F), inputs_(inputs), copyTensorMemory_(true) {

  auto setup = [&]() -> Error {
    auto graphInputValues = graph.inputs();
    RETURN_ERR_IF_NOT(
        inputs.size() + parameters.size() == graphInputValues.size(),
        glow::strFormat("Number of Graph inputs %lu must match the "
                        "number of placeholders %lu + number of "
                        "provided inputs %lu.",
                        graphInputValues.size(), parameters.size(),
                        inputs.size()));

    size_t graphIdx = 0;

    // Create Glow Placeholders for inputs.
    for (size_t i = 0; i < inputs.size(); ++i, ++graphIdx) {
      const torch::jit::Value *inputValue = graphInputValues[graphIdx];
      const c10::IValue inputIValue = inputs.at(i);
      GlowIValue glowIVal;
      RETURN_IF_ERR(glowIVal.fromIValue(inputIValue));
      if (glowIVal.isTensor()) {
        glow::Tensor *t;
        ASSIGN_VALUE_OR_RETURN_ERR(t, glowIVal.toTensor());
        glow::Placeholder *ph = F_.getParent()->createPlaceholder(
            &t->getType(), "input", /*isTrainable*/ false);
        RETURN_IF_ERR(addValueMapping(inputValue, ph->getOutput()));
        inputPlaceholders.push_back(ph);
        inputPlaceholdersReverseIndex_[ph] = i;
      } else {
        RETURN_IF_ERR(addValueMapping(inputValue, std::move(glowIVal)));
      }
    }

    // Create Glow Placeholders for training parameters (don't put them in
    // inputPlaceholders though).
    for (size_t i = 0; i < parameters.size(); ++i, ++graphIdx) {
      auto glowType = ptTypeToGlowType(*parameters[i]);
      glow::Placeholder *ph = F_.getParent()->createPlaceholder(
          &glowType, "parameter", /*isTrainable*/ false);
      RETURN_IF_ERR(
          addValueMapping(graphInputValues[graphIdx], ph->getOutput()));
    }

    // Nodes are topologically sorted. Don't do any weight freezing first.
    for (const auto &node : graph.nodes()) {
      RETURN_IF_ERR(loadNode(node));
    }

    // Create Glow Placeholders for outputs.
    for (const torch::jit::Value *output : graph.outputs()) {
      glow::NodeValue outputNodeValue;
      // Only allow tensor outputs from Glow subgraph.
      ASSIGN_VALUE_OR_RETURN_ERR(outputNodeValue,
                                 getGlowNodeValueForValue(output));
      auto *save = F_.createSave("save", outputNodeValue);
      outputPlaceholders.push_back(save->getPlaceholder());
    }

    return Error::success();
  };

  error = setup();
}

ValueMappingType ValueMapping::getMappingType() const { return mappingType_; }

ValueMapping::ValueMapping(NodeValue nodeValue, bool wasFrozen) {
  mappingType_ = wasFrozen ? ValueMappingType::FrozenNodeValue
                           : ValueMappingType::NodeValue;
  nodeValue_ = std::move(nodeValue);
}

ValueMapping::ValueMapping(GlowIValue glowIValue) {
  mappingType_ = ValueMappingType::IValue;
  glowIValue_ = llvm::make_unique<GlowIValue>(std::move(glowIValue));
}

Expected<NodeValue> ValueMapping::getMappedNodeValue() {
  if (mappingType_ == ValueMappingType::IValue) {
    RETURN_ERR("ValueMapping doesn't contain a NodeValue");
  } else {
    return nodeValue_;
  }
}

Expected<GlowIValue *> ValueMapping::getMappedGlowIValue() {
  if (mappingType_ == ValueMappingType::IValue) {
    return glowIValue_.get();
  } else {
    RETURN_ERR("ValueMapping doesn't contain a GlowIValue");
  }
}

Expected<const GlowIValue *> ValueMapping::getMappedGlowIValue() const {
  if (mappingType_ == ValueMappingType::IValue) {
    return glowIValue_.get();
  } else {
    RETURN_ERR("ValueMapping doesn't contain a GlowIValue");
  }
}

} // namespace glow
