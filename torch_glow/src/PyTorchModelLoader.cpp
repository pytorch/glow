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

#include "PyTorchModelLoader.h"
#include "CustomPyTorchOpLoader.h"
#include "PyTorchCommon.h"

#include "glow/Exporter/ONNXModelWriter.h"
#include "glow/Quantization/Base/Base.h"
#include "glow/Support/Error.h"
#include "glow/Support/Support.h"

#include <ATen/ATen.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/native/c10_utils.h>
#include <torch/csrc/jit/ir/ir.h>

namespace glow {

namespace {
/// For the quantized PyTorch ops, the activations are quantized to uint_8.
/// In Glow, the activations are quantized to int_8. Therefore, for the offset
/// read from quantized pytorch model, we need to subtract 128(i.e. INT8_MIN) to
/// make the activations becomes int8_t.

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

/// Given the dimensions of two inputs of equal length and at least rank 3 \p
/// lhsDims and \p rhsDims, \returns the broadcast for each dimension with the
/// inner-most two dimensions set to 0 because they will not be broadcast for
/// matmul. This is a helper for loading matmul.
Expected<std::vector<glow::dim_t>>
computeBroadcastedMatMulTargetDims(llvm::ArrayRef<glow::dim_t> lhsDims,
                                   llvm::ArrayRef<glow::dim_t> rhsDims) {

  size_t lhsRank = lhsDims.size();
  size_t rhsRank = lhsDims.size();

  RETURN_ERR_IF_NOT(
      lhsRank == rhsRank,
      "Both inputs must have the same rank to compute broadcast.");

  RETURN_ERR_IF_NOT(lhsRank >= 3,
                    "Inputs must have at least rank 3 to compute broadcast.");

  std::vector<glow::dim_t> targetDims;

  // Reverse both inputs dims.
  auto lhsDimsRev = std::vector<glow::dim_t>(lhsDims.rbegin(), lhsDims.rend());
  auto rhsDimsRev = std::vector<glow::dim_t>(rhsDims.rbegin(), rhsDims.rend());

  // Insert 0 placeholders for the final two dims.
  targetDims.push_back(0);
  targetDims.push_back(0);

  // Start at index 2 because we don't broadcast the inner-most dims (these are
  // the first two dims in this case since these are reversed).
  for (size_t i = 2; i < lhsRank; ++i) {
    size_t lhsTarget = lhsDimsRev[i];
    size_t rhsTarget = rhsDimsRev[i];

    if (lhsTarget == rhsTarget || lhsTarget == 1 || rhsTarget == 1) {
      targetDims.push_back(std::max(lhsTarget, rhsTarget));
    } else {
      return MAKE_ERR(strFormat("Cannot broadcast dim %d with %d",
                                (int32_t)lhsTarget, (int32_t)rhsTarget));
    }
  }

  // Reverse the dimensions back before return.
  std::reverse(targetDims.begin(), targetDims.end());
  return targetDims;
}

/// Given dims \p inputDims, returns an expansion of these dims to rank \p
/// targetRank by prepending 1s. This is a helper for loading matmul.
std::vector<glow::dim_t> getExpandDims(llvm::ArrayRef<glow::dim_t> inputDims,
                                       size_t targetRank) {
  DCHECK_LE(inputDims.size(), targetRank)
      << "The rank of inputDims can't be expanded if it's already larger than "
         "targetRank.";

  std::vector<glow::dim_t> newShape;
  for (size_t i = 0, e = targetRank - inputDims.size(); i < e; ++i) {
    newShape.push_back(1);
  }
  for (size_t d : inputDims) {
    newShape.push_back(d);
  }
  return newShape;
}

/// Given dims \p inputDims, \returns these dims contracted to rank \p by
/// combining the outer most dimensions. This is a helper for loading matmul.
std::vector<glow::dim_t> getContractDims(llvm::ArrayRef<glow::dim_t> inputDims,
                                         size_t targetRank) {
  size_t inputRank = inputDims.size();

  DCHECK_GE(inputRank, targetRank)
      << "Can't contract dims if there are less than targetRank to begin with.";

  if (inputRank == targetRank) {
    return inputDims;
  }

  std::vector<glow::dim_t> newShape;

  size_t inputIndex = 0;

  // Combine outer dimension into a single dimension.
  newShape.push_back(1);
  for (size_t i = 0, e = inputRank - (targetRank - 1); i < e;
       ++i, ++inputIndex) {
    newShape[0] *= inputDims[inputIndex];
  }

  // Copy the inner dimensions.
  for (; inputIndex < inputRank; ++inputIndex) {
    newShape.push_back(inputDims[inputIndex]);
  }

  return newShape;
}

/// Helper function check that indices are valid and convert negative indices to
/// positive indices using Python's negative indexing. \p index is the raw
/// index, it could be positive or negative, dimSize is the size of the
/// container being indexed into.
Expected<int64_t> getPositiveIndex(int64_t index, int64_t dimSize) {
  RETURN_ERR_IF_NOT(dimSize > 0, "Can't index into an empty container");

  const int64_t minIndex = 0 - dimSize;
  const int64_t maxIndex = dimSize - 1;

  RETURN_ERR_IF_NOT(minIndex <= index && index <= maxIndex,
                    strFormat("Invalid index, expected to be in range of "
                              "[%" PRId64 ", %" PRId64 "], but got %" PRId64,
                              minIndex, maxIndex, index));

  return index >= 0 ? index : dimSize + index;
}

// TODO: replace this with PyTorch's cpp_custom_type_hack::isa
/// \returns true if output of \p node is used only by a packed quantized node.
bool isPackedQParamNode(const torch::jit::Node *node) {
  static std::unordered_set<torch::jit::Symbol> packedQuantNodeKinds = {
      torch::jit::Symbol::fromQualString("quantized::linear"),
      torch::jit::Symbol::fromQualString("quantized::conv2d"),
      torch::jit::Symbol::fromQualString("quantized::conv2d_relu"),
      torch::jit::Symbol::fromQualString("quantized::conv3d"),
      torch::jit::Symbol::fromQualString("quantized::conv3d_relu"),
  };

  const auto uses = node->output()->uses();
  if (uses.empty()) {
    return false;
  }

  const auto userKind = uses[0].user->kind();

  if (packedQuantNodeKinds.count(userKind)) {
    DCHECK_EQ(uses.size(), 1) << "Expected packed quantization parameters to "
                                 "only be used by one node";
    return true;
  }

  return false;
}

/// Writes the given Function \p F to file using ONNXModelWriter. If \p zipMode
/// is set then zipMode will be used for the writer. \returns an Error if one
/// occurred.
Error dumpOnnxModel(glow::Function &F, bool zipMode) {
  constexpr size_t kIrVer = 7, kOpsetVer = 9;
  std::string fileName = F.getName().str() + (zipMode ? ".zip" : ".onnxtxt");
  LOG(INFO) << "Writing ONNX model to " << fileName;
  Error err = Error::empty();
  ONNXModelWriter onnxWriter(fileName, F, kIrVer, kOpsetVer, &err,
                             /* textMode */ !zipMode, /* zipMode */ zipMode,
                             /* useGlowCustomOps */ true);
  return err;
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

/// Indexes of aten::mean inputs.
struct MeanInputs {
  enum {
    input = 0,
    axis = 1,
    keepdims = 2,
    output = 3,
  };
};

/// Indexes of aten::batch_norm inputs.
struct BatchNormInputs {
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

/// Indexes of aten::layer_norm inputs.
struct LayerNormInputs {
  enum {
    input = 0,
    normalized_shape = 1,
    weight = 2,
    bias = 3,
    eps = 4,
    cuddnn_enabled = 5,
  };
};

/// Indexes of aten::dropout inputs.
struct DropoutInputs {
  enum {
    input = 0,
    p = 1,
    training = 2,
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

/// Indices of aten::Pow inputs.
struct PowInputs {
  enum {
    input = 0,
    exponent = 1,
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

/// Indexes of glow::unpacked_quantized_conv2d inputs.
struct QuantizedUnpackedConv2dInputs {
  enum {
    input = 0, // NCHW
    weights = 1,
    bias = 2,
    stride = 3,
    padding = 4,
    dilation = 5,
    group = 6,
    scale = 7,
    zero_point = 8,
  };
};

/// Indexes of glow::unpacked_quantized_conv3d inputs.
struct QuantizedUnpackedConv3dInputs {
  enum {
    input = 0, // NCTHW
    weights = 1,
    bias = 2,
    stride = 3,
    padding = 4,
    dilation = 5,
    group = 6,
    scale = 7,
    zero_point = 8,
  };
};

/// Indexes of quantized::conv2d and quantized::conv2d_relu inputs.
struct QuantizedConv2dInputs {
  enum {
    input = 0, // NCHW
    packed_weights = 1,
    stride = 2,
    padding = 3,
    dilation = 4,
    group = 5,
    scale = 6,
    zero_point = 7,
  };
};

/// Indexes of quantized::conv3d and quantized::conv3d_relu inputs.
struct QuantizedConv3dInputs {
  enum {
    input = 0, // NCTHW
    packed_weights = 1,
    stride = 2,
    padding = 3,
    dilation = 4,
    group = 5,
    scale = 6,
    zero_point = 7,
  };
};

/// Indexes of quantized::add_relu inputs.
struct QuantizedAddReluInputs {
  enum {
    lhs = 0,
    rhs = 1,
    scale = 2,
    zero_point = 3,
  };
};

/// Indexes of quantized::add inputs.
struct QuantizedAddInputs {
  enum {
    lhs = 0,
    rhs = 1,
    scale = 2,
    zero_point = 3,
  };
};

/// Indexes of glow::unpacked_quantized_linear inputs.
struct QuantizedUnpackedLinearInputs {
  enum {
    input = 0,
    weight = 1,
    bias = 2,
    scale = 3,
    zero_point = 4,
  };
};

/// Indexes of quantized::linear inputs.
struct QuantizedLinearInputs {
  enum {
    input = 0,
    packed_weights = 1,
    scale = 2,
    zero_point = 3,
  };
};

/// Indexes of aten::quantize_per_tensor inputs.
struct QuantizeInputs {
  enum {
    input = 0,
    scale = 1,
    zero_point = 2,
    dtype = 3,
  };
};

/// Indexes of aten::prelu inputs.
struct PReluInputs {
  enum {
    input = 0,
    weight = 1,
  };
};

/// Indexes of aten::slice inputs.
struct SliceInputs {
  enum {
    input = 0,
    dim = 1,
    start = 2,
    end = 3,
    step = 4,
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

/// Indexes of aten::addmm inputs.
struct AddMMInputs {
  enum {
    input = 0,
    mat1 = 1,
    mat2 = 2,
    beta = 3,
    alpha = 4,
  };
};

/// Indexes of aten::transpose inputs.
struct TransposeInputs {
  enum {
    input = 0,
    dim0 = 1,
    dim1 = 2,
  };
};

/// Indexes of glow::fused_linear inputs.
struct GlowFusedLinearInputs {
  enum {
    input = 0,
    weights = 1,
    bias = 2,
    dim = 3,
    add_scalar = 4,
  };
};

/// Indexes of aten::embedding_bag inputs.
struct EmbeddingBagInputs {
  enum {
    weight,
    indices,
    offsets,
    scale_grad_by_freq,
    mode,
    sparse,
    per_sample_weights,
  };
};

/// Indexes of fb::embedding_bag_byte_rowwise_offsets inputs.
struct EmbeddingBagByteRowwiseOffsetsInputs {
  enum {
    weight,
    indices,
    offsets,
    scale_grad_by_freq,
    mode,
    sparse,
    per_sample_weights,
  };
};
} // namespace

// static
const PyTorchModelLoader::MappingOfMemberFunctions
PyTorchModelLoader::buildSymbolsMapping() {
  // First build mapping with standard PyTorch operators.
  auto symbolLoaderMapping = MappingOfMemberFunctions({
      {{"aten::type_as"}, &PyTorchModelLoader::loadTypeAs},
      {{"aten::contiguous"}, &PyTorchModelLoader::loadContiguous},
      {{"prim::Constant"}, &PyTorchModelLoader::loadConstant},
      {{"aten::mul", "aten::mul_"}, &PyTorchModelLoader::loadMul},
      {{"aten::div", "aten::div_"}, &PyTorchModelLoader::loadDiv},
      {{"aten::add", "aten::add_"}, &PyTorchModelLoader::loadAdd},
      {{"aten::sub", "aten::sub_"}, &PyTorchModelLoader::loadSub},
      {{"aten::rsub"}, &PyTorchModelLoader::loadRsub},
      {{"aten::sigmoid", "aten::sigmoid_"}, &PyTorchModelLoader::loadSigmoid},
      {{"aten::relu", "aten::relu_"}, &PyTorchModelLoader::loadRelu},
      {{"aten::gelu"}, &PyTorchModelLoader::loadGelu},
      {{"aten::tanh", "aten::tanh_"}, &PyTorchModelLoader::loadTanh},
      {{"aten::t", "aten::t_"}, &PyTorchModelLoader::loadT},
      {{"aten::permute"}, &PyTorchModelLoader::loadPermute},
      {{"aten::transpose", "aten::transpose_"},
       &PyTorchModelLoader::loadTranspose},
      {{"aten::min"}, &PyTorchModelLoader::loadMin},
      {{"aten::max"}, &PyTorchModelLoader::loadMax},
      {{"aten::exp"}, &PyTorchModelLoader::loadExp},
      {{"prim::FusedConcat"}, &PyTorchModelLoader::loadFusedConcat},
      {{"glow::fused_stack"}, &PyTorchModelLoader::loadFusedStack},
      {{"aten::mean"}, &PyTorchModelLoader::loadMean},
      {{"aten::pow"}, &PyTorchModelLoader::loadPow},
      {{"aten::dropout", "aten::dropout_"}, &PyTorchModelLoader::loadDropout},
      {{"aten::sqrt", "aten::sqrt_"}, &PyTorchModelLoader::loadSqrt},
      {{"aten::clamp"}, &PyTorchModelLoader::loadClamp},
      {{"quantized::add"}, &PyTorchModelLoader::loadQuantizedAdd},
      {{"quantized::add_relu"}, &PyTorchModelLoader::loadQuantizedAddRelu},
      {{"glow::fused_linear"}, &PyTorchModelLoader::loadGlowFusedLinear},
      {{"glow::unpacked_quantized_conv2d"},
       &PyTorchModelLoader::loadQuantizedConvUnpacked},
      {{"glow::unpacked_quantized_conv3d"},
       &PyTorchModelLoader::loadQuantizedConvUnpacked},
      {{"glow::unpacked_quantized_linear"},
       &PyTorchModelLoader::loadQuantizedLinearUnpacked},
      {{"quantized::linear"}, &PyTorchModelLoader::loadQuantizedLinear},
      {{"quantized::conv2d"}, &PyTorchModelLoader::loadQuantizedConv},
      {{"quantized::conv3d"}, &PyTorchModelLoader::loadQuantizedConv},
      {{"quantized::conv2d_relu"}, &PyTorchModelLoader::loadQuantizedConvRelu},
      {{"quantized::conv3d_relu"}, &PyTorchModelLoader::loadQuantizedConvRelu},
      {{"aten::quantize_per_tensor"}, &PyTorchModelLoader::loadQuantize},
      {{"aten::dequantize"}, &PyTorchModelLoader::loadDequantize},
      {{"aten::size"}, &PyTorchModelLoader::loadSize},
      {{"prim::ListConstruct"}, &PyTorchModelLoader::loadListConstruct},
      {{"aten::reciprocal", "aten::reciprocal_"},
       &PyTorchModelLoader::loadReciprocal},
      {{"aten::adaptive_avg_pool2d"},
       &PyTorchModelLoader::loadAdaptiveAvgPool2d},
      {{"aten::reshape"}, &PyTorchModelLoader::loadReshape},
      {{"aten::view"}, &PyTorchModelLoader::loadView},
      {{"aten::_convolution"}, &PyTorchModelLoader::loadConvolution},
      {{"aten::batch_norm"}, &PyTorchModelLoader::loadBatchNorm},
      {{"aten::layer_norm"}, &PyTorchModelLoader::loadLayerNorm},
      {{"aten::max_pool2d"}, &PyTorchModelLoader::loadMaxPool2d},
      {{"aten::avg_pool2d"}, &PyTorchModelLoader::loadAvgPool2d},
      {{"aten::matmul"}, &PyTorchModelLoader::loadMatMul},
      {{"aten::mm"}, &PyTorchModelLoader::loadMM},
      {{"aten::bmm"}, &PyTorchModelLoader::loadBmm},
      {{"aten::addmm"}, &PyTorchModelLoader::loadAddMM},
      {{"aten::flatten"}, &PyTorchModelLoader::loadFlatten},
      {{"aten::prelu"}, &PyTorchModelLoader::loadPRelu},
      {{"aten::slice"}, &PyTorchModelLoader::loadSlice},
      {{"aten::softmax"}, &PyTorchModelLoader::loadSoftMax},
      {{"aten::topk"}, &PyTorchModelLoader::loadTopK},
      {{"prim::ConstantChunk"}, &PyTorchModelLoader::loadConstantChunk},
      {{"aten::embedding_bag"}, &PyTorchModelLoader::loadEmbeddingBag},
      {{"fb::embedding_bag_byte_rowwise_offsets"},
       &PyTorchModelLoader::loadEmbeddingBagByteRowwiseOffsets},
  });

  // Add in custom operator loaders.
  for (const auto &symbolAndLoader : getCustomPyTorchOpLoaders()) {
    const char *symbolStr = symbolAndLoader.first.toQualString();
    MappingOfMemberFunctionsValue val({symbolStr},
                                      &PyTorchModelLoader::loadCustomOp);
    auto res = symbolLoaderMapping.insert({symbolAndLoader.first, val});
    DCHECK(res.second)
        << "Tried to create a custom op loader for a symbol that "
           "already has a registered loader: "
        << symbolStr;
  }

  return symbolLoaderMapping;
}

// static
const PyTorchModelLoader::MappingOfMemberFunctions &
PyTorchModelLoader::getSymbolsMapping() {
  /// Static map of the set of PyTorch symbols to load, the PyTorchModelLoader
  /// for loading these symbols, and the set of inputs that should be considered
  /// immutable between inference invocations by Glow and loaded as Constants
  /// instead of Placeholders.
  static auto symbolLoaderMapping = buildSymbolsMapping();

  return symbolLoaderMapping;
}

// static
bool PyTorchModelLoader::isNodeSupported(const torch::jit::Node *ptNode) {
  const auto kind = ptNode->kind();

  // Special case for prim::GetAttr, it's loaded separately from other ops.
  if (kind == torch::jit::prim::GetAttr) {
    return true;
  }

  const auto &mapping = getSymbolsMapping();

  return mapping.count(kind) != 0;
}

Error PyTorchModelLoader::loadNodes(const torch::jit::Graph &graph) {
  const auto &mapping = getSymbolsMapping();
  // Nodes are topologically sorted.
  for (const auto &node : graph.nodes()) {
    const auto kind = node->kind();
    // prim::GetAttr is loaded separately.
    if (kind == torch::jit::prim::GetAttr) {
      continue;
    }

    auto it = mapping.find(kind);

    RETURN_ERR_IF_NOT(it != mapping.end(),
                      glow::strFormat("Node kind %s is not supported by Glow",
                                      node->kind().toDisplayString()));

    RETURN_IF_ERR((this->*it->second.loadFn)(node));
  }

  return Error::success();
}

Error PyTorchModelLoader::getCorrectTypeMapping(c10::ScalarType &dest,
                                                const torch::jit::Value *src) {
  auto it = valueMap_.find(src);
  RETURN_ERR_IF_NOT(
      it != valueMap_.end(),
      glow::strFormat(
          "Cannot find value %s when trying to propagate its correct type",
          src->debugNameBase().c_str()));
  dest = it->second.getCorrectType();
  return Error::success();
}

Error PyTorchModelLoader::addValueMapping(const torch::jit::Value *value,
                                          glow::NodeValue nodeValue,
                                          c10::ScalarType correctType) {

  ValueMapping mapping(std::move(nodeValue));
  mapping.setCorrectType(correctType);
  auto p = valueMap_.emplace(value, std::move(mapping));

  RETURN_ERR_IF_NOT(p.second, glow::strFormat("Value %s is already mapped",
                                              value->debugNameBase().c_str()));
  return Error::success();
}

Error PyTorchModelLoader::addValueMapping(const torch::jit::Value *value,
                                          glow::NodeValue nodeValue) {
  auto correctType =
      elemKindToScalarType(nodeValue.getType()->getElementType());
  return addValueMapping(value, nodeValue, correctType);
}

void PyTorchModelLoader::removeValueMapping(const torch::jit::Value *value) {
  valueMap_.erase(value);
}

Error PyTorchModelLoader::addValueMapping(const torch::jit::Value *value,
                                          glow::GlowIValue glowIValue,
                                          c10::ScalarType correctType) {
  glow::Constant *glowConstant = nullptr;
  if (glowIValue.isTensor()) {
    glow::Tensor *t;
    ASSIGN_VALUE_OR_RETURN_ERR(t, glowIValue.toTensor());
    glowConstant = F_.getParent()->createConstant("constant", std::move(*t));
    RETURN_IF_ERR(
        addValueMapping(value, glowConstant->getOutput(), correctType));
  } else {
    ValueMapping mapping(std::move(glowIValue));
    mapping.setCorrectType(correctType);
    auto p = valueMap_.emplace(value, std::move(mapping));

    RETURN_ERR_IF_NOT(p.second,
                      glow::strFormat("Value %s is already mapped",
                                      value->debugNameBase().c_str()));
  }

  return Error::success();
}

Error PyTorchModelLoader::addValueMapping(const torch::jit::Value *value,
                                          glow::GlowIValue glowIValue) {
  auto correctType = c10::ScalarType::Undefined;
  // We dont propagate IValue's type unless the type is given,
  // since it has too many posibilities.
  return addValueMapping(value, std::move(glowIValue), correctType);
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
    RETURN_ERR(glow::strFormat(
        "Found a GlowIValue instead of a NodeValue for this Value: %s",
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
    RETURN_ERR(glow::strFormat(
        "Found a NodeValue instead of a GlowIValue for this Value: %s",
        value->debugNameBase().c_str()));
  }
  return mappingValue.getMappedGlowIValue();
}

glow::NodeValue PyTorchModelLoader::rescaleUIntToInt(glow::NodeValue input) {
  auto *inputTy = input.getType();
  if (inputTy->getElementType() == ElemKind::UInt8QTy) {
    auto dqInput = F_.createDequantize("dequantize", input);
    auto *outputTy = F_.getParent()->uniqueType(
        ElemKind::Int8QTy, inputTy->dims(), inputTy->getScale(),
        inputTy->getOffset() - OFFSETSHIFT);
    auto *qOut = F_.createQuantize("quantize", dqInput, outputTy);
    return qOut->getResult();
  } else {
    return input;
  }
}

glow::NodeValue PyTorchModelLoader::rescaleIntToUint(glow::NodeValue input) {
  auto *inputTy = input.getType();
  if (inputTy->getElementType() == ElemKind::Int8QTy) {
    auto dqInput = F_.createDequantize("dequantize", input);
    auto *outputTy = F_.getParent()->uniqueType(
        ElemKind::UInt8QTy, inputTy->dims(), inputTy->getScale(),
        inputTy->getOffset() + OFFSETSHIFT);
    auto *qOut = F_.createQuantize("quantize", dqInput, outputTy);
    return qOut->getResult();
  } else {
    return input;
  }
}

Expected<NodeValue>
PyTorchModelLoader::loadQuantizedConvImpl(const torch::jit::Node *ptNode,
                                          const bool isRelu) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  const glow::TransposeNode *output;

  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 8, outputs, 1));

  // input
  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(input, getGlowNodeValueForValue(inputs[0]));

  bool isConv3d = input.dims().size() == 5;
  if (isConv3d) {
    input = F_.createTranspose("qconv_input_transposed", input, NCTHW2NTHWC);
  } else {
    input = F_.createTranspose("qconv_input_transposed", input, NCHW2NHWC);
  }
  std::unordered_map<std::string, int8_t> input_mapping = {};
  if (isConv3d) {
    input_mapping["input"] = QuantizedConv3dInputs::input;
    input_mapping["packed_weights"] = QuantizedConv3dInputs::packed_weights;
    input_mapping["stride"] = QuantizedConv3dInputs::stride;
    input_mapping["padding"] = QuantizedConv3dInputs::padding;
    input_mapping["group"] = QuantizedConv3dInputs::group;
    input_mapping["scale"] = QuantizedConv3dInputs::scale;
    input_mapping["zero_point"] = QuantizedConv3dInputs::zero_point;

  } else {
    input_mapping["input"] = QuantizedConv2dInputs::input;
    input_mapping["packed_weights"] = QuantizedConv2dInputs::packed_weights;
    input_mapping["stride"] = QuantizedConv2dInputs::stride;
    input_mapping["padding"] = QuantizedConv2dInputs::padding;
    input_mapping["dilation"] = QuantizedConv2dInputs::dilation;
    input_mapping["group"] = QuantizedConv2dInputs::group;
    input_mapping["scale"] = QuantizedConv2dInputs::scale;
    input_mapping["zero_point"] = QuantizedConv2dInputs::zero_point;
  }

  // weight and bias
  at::Tensor *ptTensor;
  ASSIGN_VALUE_OR_RETURN_ERR(ptTensor,
                             iValToPTTensor(getGlowIValueForValue(
                                 inputs[input_mapping["packed_weights"]])));

  // groups
  glow::unsigned_t groups;
  ASSIGN_VALUE_OR_RETURN_ERR(
      groups, static_cast_expected<glow::unsigned_t>(iValToInt(
                  getGlowIValueForValue(inputs[input_mapping["group"]]))));

  std::string unpack_op_name;
  if (isConv3d) {
    unpack_op_name = "quantized::conv3d_unpack";
  } else {
    unpack_op_name = "quantized::conv2d_unpack";
  }
  auto op = c10::Dispatcher::singleton().findSchema({unpack_op_name, ""});

  CHECK(op.has_value());
  auto unpackedParams = callOp(*op, *ptTensor);
  const at::Tensor ptWeightTensor = unpackedParams[0].toTensor().contiguous();

  const c10::optional<at::Tensor> ptBiasTensorTmp =
      unpackedParams[1].toOptional<at::Tensor>();

  bool isPerChannelQuantized =
      ptWeightTensor.is_quantized() &&
      ptWeightTensor.qscheme() == at::kPerChannelAffine;

  // unpacked weights
  auto weightTensor = ptTensorToGlowTensor(ptWeightTensor);
  glow::Tensor weightTensorTransposed;
  std::string weightConstantName;
  if (isConv3d) {
    weightTensor.transpose(&weightTensorTransposed, NCTHW2NTHWC);
    weightConstantName = "quantized_conv3d_weights";
  } else {
    weightTensor.transpose(&weightTensorTransposed, NCHW2NHWC);
    weightConstantName = "quantized_conv2d_weights";
  }
  glow::Constant *weightConstant = F_.getParent()->createConstant(
      weightConstantName, std::move(weightTensorTransposed));
  weightConstant->ensureIsOwned();
  auto weight = weightConstant->getOutput();
  weight = rescaleUIntToInt(weight);

  // unpacked bias
  glow::Tensor biasTensor;
  if (ptBiasTensorTmp.has_value()) {
    auto ptBiasTensor = ptBiasTensorTmp.value().contiguous();
    biasTensor = ptTensorToGlowTensor(ptBiasTensor);
  } else {
    biasTensor = glow::Tensor(glow::ElemKind::FloatTy, {weight.dims()[0]});
    biasTensor.zero();
  }

  std::string biasConstantName;
  if (isConv3d) {
    biasConstantName = "quantized_conv3d_bias";
  } else {
    biasConstantName = "quantized_conv2d_bias";
  }
  glow::Constant *biasConstant =
      F_.getParent()->createConstant(biasConstantName, std::move(biasTensor));

  biasConstant->ensureIsOwned();
  glow::NodeValue bias = biasConstant->getOutput();

  // strides
  std::vector<glow::unsigned_t> strides;
  ASSIGN_VALUE_OR_RETURN_ERR(
      strides, castVector<glow::unsigned_t>(expandIntIValIfNeeded(
                   getGlowIValueForValue(inputs[input_mapping["stride"]]),
                   input.dims().size() - 2)));

  // pads
  std::vector<glow::unsigned_t> pads;
  if (isConv3d) {
    std::vector<glow::unsigned_t> pad;

    ASSIGN_VALUE_OR_RETURN_ERR(
        pad, castVector<glow::unsigned_t>(expandIntIValIfNeeded(
                 getGlowIValueForValue(inputs[input_mapping["padding"]]), 3)));
    pads = {pad[0], pad[0], pad[1], pad[1], pad[2], pad[2]};
  } else {
    glow::unsigned_t pad;
    ASSIGN_VALUE_OR_RETURN_ERR(
        pad, static_cast_expected<glow::unsigned_t>(contractIntIValIfNeeded(
                 getGlowIValueForValue(inputs[input_mapping["padding"]]))));
    pads = {pad, pad, pad, pad};
  }

  // quantized params
  float outScale;
  ASSIGN_VALUE_OR_RETURN_ERR(outScale, iValToDouble(getGlowIValueForValue(
                                           inputs[input_mapping["scale"]])));

  int32_t outOffset;
  ASSIGN_VALUE_OR_RETURN_ERR(
      outOffset,
      iValToInt(getGlowIValueForValue(inputs[input_mapping["zero_point"]])));

  // calc output type
  glow::unsigned_t dilation = 0;
  glow::TypeRef outTy;
  std::vector<glow::unsigned_t> kernels;
  if (isConv3d) {
    glow::ShapeNTHWC input3DShape(input.dims());
    glow::ShapeNTHWC weight3DShape(weight.dims());
    kernels = {static_cast<glow::unsigned_t>(weight3DShape.t),
               static_cast<glow::unsigned_t>(weight3DShape.h),
               static_cast<glow::unsigned_t>(weight3DShape.w)};
    auto outSz = glow::calculate3DConvPoolOutputDims(
        input3DShape.t, input3DShape.h, input3DShape.w, kernels, strides, pads);
    std::array<glow::dim_t, 5> outDims = {{input.dims()[0],
                                           outSz.temporal_frames, outSz.height,
                                           outSz.width, weight.dims()[0]}};
    outTy = F_.getParent()->uniqueType(glow::ElemKind::Int8QTy, outDims,
                                       outScale, outOffset - OFFSETSHIFT);

  } else {
    glow::ShapeNHWC inputShape(input.dims());
    glow::ShapeNHWC weightShape(weight.dims());
    kernels = {static_cast<glow::unsigned_t>(weightShape.h),
               static_cast<glow::unsigned_t>(weightShape.w)};
    ASSIGN_VALUE_OR_RETURN_ERR(
        dilation,
        static_cast_expected<glow::unsigned_t>(contractIntIValIfNeeded(
            getGlowIValueForValue(inputs[QuantizedConv2dInputs::dilation]))));
    auto outSz = glow::calculateConvPoolOutputDims(
        inputShape.h, inputShape.w, kernels, strides, pads, dilation);
    std::array<glow::dim_t, 4> outDims = {
        {input.dims()[0], outSz.first, outSz.second, weightShape.n}};
    outTy = F_.getParent()->uniqueType(glow::ElemKind::Int8QTy, outDims,
                                       outScale, outOffset - OFFSETSHIFT);
  }

  // create qconv
  glow::NodeValue output_not_transposed;
  if (isPerChannelQuantized) {
    if (!isConv3d) {
      RETURN_ERR_IF_NOT(
          dilation <= 1,
          "Dilation not supported for group quantized convolution");
    }

    // extract qparams from ptWeightTensor.
    // Notice since the memory of qparams may not be continous
    // we CANNOT use the data ptr of this chunk of memory and
    // convert them into glow tensor directly by using PtTensorToGlowTensor.
    // Instead, we extract them one after one.
    std::vector<float> scalesVector;
    std::vector<int32_t> offsetsVector;
    std::vector<glow::dim_t> dims;
    const int n = ptWeightTensor.q_per_channel_scales().size(0);
    dims.push_back(n);
    for (int i = 0; i < n; i++) {
      float scale =
          ptWeightTensor.q_per_channel_scales().to(at::kFloat)[i].item<float>();
      int32_t offset = ptWeightTensor.q_per_channel_zero_points()
                           .to(at::kInt)[i]
                           .item<int32_t>();
      scalesVector.push_back(scale);
      offsetsVector.push_back(offset);
    }

    // construct qparam constants
    auto scaleType = glow::Type(ElemKind::FloatTy, dims);
    auto offsetType = glow::Type(ElemKind::Int32ITy, dims);
    auto wScalesTensor = glow::Tensor(scalesVector.data(), &scaleType);
    auto wOffsetsTensor = glow::Tensor(offsetsVector.data(), &offsetType);

    auto wScales = F_.getParent()->createConstant(
        "channel_wised_scales_of_qconv", std::move(wScalesTensor));
    wScales->ensureIsOwned();
    auto wOffsets = F_.getParent()->createConstant(
        "channel_wised_offsets_of_qconv", std::move(wOffsetsTensor));
    wOffsets->ensureIsOwned();

    if (isConv3d) {
      auto qconv = F_.createChannelwiseQuantizedConv3D(
          "qconv_channel_wised", input, weightConstant, biasConstant, wScales,
          wOffsets, outTy, kernels, strides, pads, groups);
      output_not_transposed = qconv->getResult();
    } else {
      auto qconv = F_.createChannelwiseQuantizedConv(
          "qconv_channel_wised", input, weightConstant, biasConstant, wScales,
          wOffsets, outTy, kernels, strides, pads, groups);
      output_not_transposed = qconv->getResult();
    }
  } else {
    if (isConv3d) {
      auto qconv = F_.createConv3D("qconv", input, weight, bias, outTy, kernels,
                                   strides, pads, groups);
      output_not_transposed = qconv->getResult();
    } else {
      auto qconv = F_.createConv("qconv", input, weight, bias, outTy, kernels,
                                 strides, pads, groups, dilation);
      output_not_transposed = qconv->getResult();
    }
  }
  if (isRelu) {
    glow::ReluNode *qrelu = F_.createRELU("qconv_relu", output_not_transposed);
    output_not_transposed = qrelu->getResult();
  }
  if (isConv3d) {
    output = F_.createTranspose("channel_wised_qconv_relu_output_transposed",
                                output_not_transposed, NTHWC2NCTHW);
  } else {
    output = F_.createTranspose("channel_wised_qconv_relu_output_transposed",
                                output_not_transposed, NHWC2NCHW);
  }
  return Expected<NodeValue>(output->getResult());
}

template <typename T>
NodeValue PyTorchModelLoader::loadNodeValueOrCreateBroadcastedConstant(
    const torch::jit::Value *value, llvm::StringRef name, const Type &ty,
    const T &val) {
  glow::NodeValue nodeValue;
  if (hasGlowNodeValueForValue(value)) {
    return EXIT_ON_ERR(getGlowNodeValueForValue(value));
  } else {
    glow::Tensor t(ty);
    t.init(glow::Tensor::InitKind::Broadcast, val, F_.getParent()->getPRNG());
    return F_.getParent()->createConstant(name, std::move(t))->getOutput();
  }
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

  // zero_point
  int32_t outOffset;
  ASSIGN_VALUE_OR_RETURN_ERR(
      outOffset,
      iValToInt(getGlowIValueForValue(inputs[QuantizedAddInputs::zero_point])));

  TypeRef inputType = lhs.getType();
  auto outDims = inputType->dims();
  auto outTy = F_.getParent()->uniqueType(ElemKind::Int8QTy, outDims, outScale,
                                          outOffset - OFFSETSHIFT);

  glow::AddNode *qadd = F_.createAdd("quantized_add", outTy, lhs, rhs);
  auto output = qadd->getResult();

  c10::ScalarType dtype;
  RETURN_IF_ERR(getCorrectTypeMapping(dtype, inputs[QuantizedAddInputs::lhs]));
  return addValueMapping(outputs[0], output, dtype);
}

Error PyTorchModelLoader::loadQuantizedAddRelu(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 4, outputs, 1));

  glow::NodeValue lhs;
  ASSIGN_VALUE_OR_RETURN_ERR(
      lhs, getGlowNodeValueForValue(inputs[QuantizedAddReluInputs::lhs]));
  glow::NodeValue rhs;
  ASSIGN_VALUE_OR_RETURN_ERR(
      rhs, getGlowNodeValueForValue(inputs[QuantizedAddReluInputs::rhs]));

  // scale
  float outScale;
  ASSIGN_VALUE_OR_RETURN_ERR(outScale,
                             iValToDouble(getGlowIValueForValue(
                                 inputs[QuantizedAddReluInputs::scale])));

  // zero_point
  int32_t outOffset;
  ASSIGN_VALUE_OR_RETURN_ERR(outOffset,
                             iValToInt(getGlowIValueForValue(
                                 inputs[QuantizedAddReluInputs::zero_point])));

  TypeRef inputType = lhs.getType();
  auto outDims = inputType->dims();
  auto outTy = F_.getParent()->uniqueType(ElemKind::Int8QTy, outDims, outScale,
                                          outOffset - OFFSETSHIFT);

  glow::AddNode *qadd = F_.createAdd("quantized_add", outTy, lhs, rhs);
  glow::ReluNode *qrelu = F_.createRELU("quantized_relu", qadd);
  auto output = qrelu->getResult();

  c10::ScalarType dtype;
  RETURN_IF_ERR(
      getCorrectTypeMapping(dtype, inputs[QuantizedAddReluInputs::lhs]));
  return addValueMapping(outputs[0], output, dtype);
}

Error PyTorchModelLoader::loadQuantizedLinear(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 4, outputs, 1));

  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(
      input, getGlowNodeValueForValue(inputs[QuantizedLinearInputs::input]));

  at::Tensor *ptTensor;
  ASSIGN_VALUE_OR_RETURN_ERR(
      ptTensor, iValToPTTensor(getGlowIValueForValue(
                    inputs[QuantizedLinearInputs::packed_weights])));

  auto op =
      c10::Dispatcher::singleton().findSchema({"quantized::linear_unpack", ""});
  CHECK(op.has_value());
  auto unpackedParams = callOp(*op, *ptTensor);
  const at::Tensor ptWeightTensor = unpackedParams[0].toTensor().contiguous();
  const c10::optional<at::Tensor> ptBiasTensorTmp =
      unpackedParams[1].toOptional<at::Tensor>();

  bool isRowwiseQuantized = ptWeightTensor.is_quantized() &&
                            ptWeightTensor.qscheme() == at::kPerChannelAffine;

  // unpacked weights
  auto weightTensor = ptTensorToGlowTensor(ptWeightTensor);
  glow::Constant *weightConstant = F_.getParent()->createConstant(
      "quantized_linear_weights", std::move(weightTensor));
  weightConstant->ensureIsOwned();
  auto weight = weightConstant->getOutput();
  weight = rescaleUIntToInt(weight);
  RETURN_ERR_IF_NOT(weight.dims().size() == 2, "Expected 2d Linear weights");

  // unpacked bias
  glow::Tensor biasTensor;
  if (ptBiasTensorTmp.has_value()) {
    auto ptBiasTensor = ptBiasTensorTmp.value().contiguous();
    biasTensor = ptTensorToGlowTensor(ptBiasTensor);
  } else {
    biasTensor = glow::Tensor(glow::ElemKind::FloatTy, {weight.dims()[1]});
    biasTensor.zero();
  }

  // Choose bias quantization params and quantize it.
  glow::Constant *biasConstant = F_.getParent()->createConstant(
      "quantized_linear_bias", std::move(biasTensor));
  biasConstant->ensureIsOwned();
  RETURN_ERR_IF_NOT(biasConstant, "quantized::linear bias must be constant");
  const auto biasHandle = biasConstant->getPayload().getHandle<float>();
  const auto biasMinMaxIdx = biasHandle.minMaxArg();

  const auto biasQParams = chooseQuantizationParams(
      {biasHandle.raw(biasMinMaxIdx.first),
       biasHandle.raw(biasMinMaxIdx.second)},
      glow::quantization::Schema::Asymmetric, glow::ElemKind::Int32QTy);

  auto bias = biasConstant->getOutput();

  auto biasType =
      F_.getParent()->uniqueType(glow::ElemKind::Int32QTy, bias.dims(),
                                 biasQParams.scale, biasQParams.offset);
  bias = F_.createQuantize("quantize_bias", bias, biasType);

  float outScale;
  ASSIGN_VALUE_OR_RETURN_ERR(outScale,
                             to32Bit(iValToDouble(getGlowIValueForValue(
                                 inputs[QuantizedLinearInputs::scale]))));

  int64_t outZeroPoint;
  ASSIGN_VALUE_OR_RETURN_ERR(outZeroPoint,
                             iValToInt(getGlowIValueForValue(
                                 inputs[QuantizedLinearInputs::zero_point])));

  auto outTy = F_.getParent()->uniqueType(ElemKind::Int8QTy,
                                          {input.dims()[0], weight.dims()[0]},
                                          outScale, outZeroPoint - OFFSETSHIFT);
  if (isRowwiseQuantized) {
    // extract qparams from ptWeightTensor.
    // Notice since the memory of qparams may not be continous
    // we CANNOT use the data ptr of this chunk of memory and
    // convert them into glow tensor directly by using PtTensorToGlowTensor.
    // Instead, we extract them one after one.
    std::vector<float> scalesVector;
    std::vector<int32_t> offsetsVector;
    std::vector<glow::dim_t> dims;
    const int n = ptWeightTensor.q_per_channel_scales().size(0);
    dims.push_back(n);
    for (int i = 0; i < n; i++) {
      float scale =
          ptWeightTensor.q_per_channel_scales().to(at::kFloat)[i].item<float>();
      int32_t offset = ptWeightTensor.q_per_channel_zero_points()
                           .to(at::kInt)[i]
                           .item<int32_t>();
      scalesVector.push_back(scale);
      offsetsVector.push_back(offset);
    }

    // construct qparam constants
    auto scaleType = glow::Type(ElemKind::FloatTy, dims);
    auto offsetType = glow::Type(ElemKind::Int32ITy, dims);
    auto wScalesTensor = glow::Tensor(scalesVector.data(), &scaleType);
    auto wOffsetsTensor = glow::Tensor(offsetsVector.data(), &offsetType);

    auto wScales = F_.getParent()->createConstant(
        "channel_wised_scales_of_qlinear", std::move(wScalesTensor));
    wScales->ensureIsOwned();
    auto wOffsets = F_.getParent()->createConstant(
        "channel_wised_offsets_of_qlinear", std::move(wOffsetsTensor));
    wOffsets->ensureIsOwned();
    auto rowwise_fc = F_.createRowwiseQuantizedFullyConnected(
        "rowwise_quantized_fc", input, weightConstant, wScales, wOffsets, bias,
        outTy);
    return addValueMapping(outputs[0], rowwise_fc->getResult());
  } else {
    weight = F_.createTranspose("weight_transpose", weight, {1, 0});
    auto fc =
        F_.createFullyConnected("quantized_fc", input, weight, bias, outTy);

    c10::ScalarType dtype;
    RETURN_IF_ERR(
        getCorrectTypeMapping(dtype, inputs[QuantizedLinearInputs::input]));
    return addValueMapping(outputs[0], fc->getResult(), dtype);
  }
}

Error PyTorchModelLoader::loadQuantizedLinearUnpacked(
    const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 5, outputs, 1));

  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(
      input,
      getGlowNodeValueForValue(inputs[QuantizedUnpackedLinearInputs::input]));

  glow::NodeValue weight;
  ASSIGN_VALUE_OR_RETURN_ERR(
      weight,
      getGlowNodeValueForValue(inputs[QuantizedUnpackedLinearInputs::weight]));
  weight = rescaleUIntToInt(weight);

  RETURN_ERR_IF_NOT(weight.dims().size() == 2, "Expected 2d Linear weights");

  weight = F_.createTranspose("weight_transpose", weight, {1, 0});

  float outScale;
  ASSIGN_VALUE_OR_RETURN_ERR(
      outScale, to32Bit(iValToDouble(getGlowIValueForValue(
                    inputs[QuantizedUnpackedLinearInputs::scale]))));

  int64_t outZeroPoint;
  ASSIGN_VALUE_OR_RETURN_ERR(
      outZeroPoint, iValToInt(getGlowIValueForValue(
                        inputs[QuantizedUnpackedLinearInputs::zero_point])));

  auto outTy = F_.getParent()->uniqueType(ElemKind::Int8QTy,
                                          {input.dims()[0], weight.dims()[1]},
                                          outScale, outZeroPoint - OFFSETSHIFT);

  // Get bias or create a zero bias if no bias is found.
  glow::NodeValue bias = loadNodeValueOrCreateBroadcastedConstant(
      inputs[QuantizedUnpackedLinearInputs::bias], "quantized_linear_bias",
      glow::Type(ElemKind::FloatTy, {weight.dims()[1]}), 0.0);

  // Choose bias quantization params and quantize it.
  glow::Constant *biasConstant = llvm::dyn_cast<glow::Constant>(bias.getNode());

  const auto biasHandle = biasConstant->getPayload().getHandle<float>();
  const auto biasMinMaxIdx = biasHandle.minMaxArg();

  const auto biasQParams = chooseQuantizationParams(
      {biasHandle.raw(biasMinMaxIdx.first),
       biasHandle.raw(biasMinMaxIdx.second)},
      glow::quantization::Schema::Asymmetric, glow::ElemKind::Int32QTy);

  const auto biasType =
      F_.getParent()->uniqueType(glow::ElemKind::Int32QTy, bias.dims(),
                                 biasQParams.scale, biasQParams.offset);

  bias = F_.createQuantize("quantize_bias", bias, biasType);

  auto fc = F_.createFullyConnected("quantized_fc", input, weight, bias, outTy);

  c10::ScalarType dtype;
  RETURN_IF_ERR(getCorrectTypeMapping(
      dtype, inputs[QuantizedUnpackedLinearInputs::input]));
  return addValueMapping(outputs[0], fc->getResult(), dtype);
}

Error PyTorchModelLoader::loadGlowFusedLinear(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 5, outputs, 1));

  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(
      input, getGlowNodeValueForValue(inputs[GlowFusedLinearInputs::input]));

  glow::NodeValue weights;
  ASSIGN_VALUE_OR_RETURN_ERR(
      weights,
      getGlowNodeValueForValue(inputs[GlowFusedLinearInputs::weights]));

  glow::NodeValue bias;
  ASSIGN_VALUE_OR_RETURN_ERR(
      bias, getGlowNodeValueForValue(inputs[GlowFusedLinearInputs::bias]));

  int64_t dim;
  ASSIGN_VALUE_OR_RETURN_ERR(dim, iValToInt(getGlowIValueForValue(
                                      inputs[GlowFusedLinearInputs::dim])));

  int64_t addScalar;
  ASSIGN_VALUE_OR_RETURN_ERR(addScalar,
                             iValToInt(getGlowIValueForValue(
                                 inputs[GlowFusedLinearInputs::add_scalar])));

  RETURN_ERR_IF_NOT(addScalar == 1,
                    glow::strFormat("Scalar must have value equal 1."));

  glow::NodeValue output;
  if (input.dims().size() == dim) {
    weights = F_.createTranspose("weights_transposed", weights, {1, 0});
    auto mmOutput =
        F_.createMatMul("fused_linear_mm", input, weights)->getResult();
    output = F_.createAdd("fused_linear_add", bias, mmOutput);
  } else {
    weights = F_.createTranspose("weights_transposed", weights, {1, 0});
    glow::NodeValue matmulOutput;
    ASSIGN_VALUE_OR_RETURN_ERR(matmulOutput, loadMatMulImpl(input, weights));
    output = F_.createNodeWithBroadcast<glow::AddNode>("add", /*axis*/ -1,
                                                       matmulOutput, bias);
  }

  return addValueMapping(outputs[0], output);
}

Expected<NodeValue> PyTorchModelLoader::loadNodeValueOrBroadcastedIValue(
    const torch::jit::Value *value, llvm::ArrayRef<glow::dim_t> dims,
    bool makeFloat) {
  if (hasGlowNodeValueForValue(value)) {
    return getGlowNodeValueForValue(value);
  } else {
    GlowIValue *ival;
    ASSIGN_VALUE_OR_RETURN_ERR(ival, getGlowIValueForValue(value));

    if (makeFloat) {
      float constVal;
      if (ival->isInt()) {
        ASSIGN_VALUE_OR_RETURN_ERR(constVal,
                                   static_cast_expected<float>(ival->toInt()));
      } else {
        ASSIGN_VALUE_OR_RETURN_ERR(
            constVal, static_cast_expected<float>(ival->toDouble()));
      }
      glow::Tensor t(glow::ElemKind::FloatTy, dims);
      t.init(glow::Tensor::InitKind::Broadcast, constVal,
             F_.getParent()->getPRNG());
      return F_.getParent()
          ->createConstant("constant", std::move(t))
          ->getOutput();
    } else /* makeInt */ {
      int64_t constVal;
      if (ival->isInt()) {
        ASSIGN_VALUE_OR_RETURN_ERR(
            constVal, static_cast_expected<int64_t>(ival->toInt()));
      } else {
        ASSIGN_VALUE_OR_RETURN_ERR(
            constVal, static_cast_expected<int64_t>(ival->toDouble()));
      }
      glow::Tensor t(glow::ElemKind::Int64ITy, dims);
      t.init(glow::Tensor::InitKind::Broadcast, constVal,
             F_.getParent()->getPRNG());
      return F_.getParent()
          ->createConstant("constant", std::move(t))
          ->getOutput();
    }
  }
}

Error PyTorchModelLoader::loadTypeAs(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 2, outputs, 1));

  glow::NodeValue dataValue;
  glow::NodeValue typeNode;
  ASSIGN_VALUE_OR_RETURN_ERR(typeNode, getGlowNodeValueForValue(inputs[1]));
  ASSIGN_VALUE_OR_RETURN_ERR(dataValue, getGlowNodeValueForValue(inputs[0]));
  auto outType = typeNode.getType();

  glow::Node *bcast =
      F_.createBroadcast("typeas_broadcast", dataValue, typeNode.dims(), 0);
  glow::ConvertToNode *glowNode =
      F_.createConvertTo("typeas_convert", bcast, outType);

  return addValueMapping(outputs[0], glowNode->getResult());
}

Error PyTorchModelLoader::loadContiguous(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 2, outputs, 1));

  glow::NodeValue dataValue;
  ASSIGN_VALUE_OR_RETURN_ERR(dataValue, getGlowNodeValueForValue(inputs[0]));

  int64_t scalar;
  ASSIGN_VALUE_OR_RETURN_ERR(scalar,
                             iValToInt(getGlowIValueForValue(inputs[1])));
  RETURN_ERR_IF_NOT(scalar == (int64_t)at::MemoryFormat::Contiguous,
                    glow::strFormat("Scalar must have value equal 0."));

  return addValueMapping(outputs[0], dataValue);
}

template <typename GlowNode>
Expected<NodeValue>
PyTorchModelLoader::loadArithmeticNode(llvm::StringRef name,
                                       const torch::jit::Value *lhs,
                                       const torch::jit::Value *rhs) {
  glow::NodeValue lhsInput;
  glow::NodeValue rhsInput;

  if (hasGlowNodeValueForValue(lhs)) {
    ASSIGN_VALUE_OR_RETURN_ERR(lhsInput, getGlowNodeValueForValue(lhs));
    ASSIGN_VALUE_OR_RETURN_ERR(
        rhsInput,
        loadNodeValueOrBroadcastedIValue(
            rhs, lhsInput.dims(), isFloatElemKind(lhsInput.getElementType())));
  } else if (hasGlowNodeValueForValue(rhs)) {
    ASSIGN_VALUE_OR_RETURN_ERR(rhsInput, getGlowNodeValueForValue(rhs));
    ASSIGN_VALUE_OR_RETURN_ERR(
        lhsInput,
        loadNodeValueOrBroadcastedIValue(
            lhs, rhsInput.dims(), isFloatElemKind(rhsInput.getElementType())));
  } else {
    return MAKE_ERR("Either lhs or rhs of arithmetic node must be a tensor");
  }

  return F_
      .createNodeWithBroadcast<GlowNode>(name, /*axis*/ -1, lhsInput, rhsInput)
      ->getNthResult(0);
}

Error PyTorchModelLoader::loadMul(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 2, outputs, 1));

  glow::NodeValue res;
  ASSIGN_VALUE_OR_RETURN_ERR(
      res, loadArithmeticNode<glow::MulNode>("mul", inputs[0], inputs[1]));

  return addValueMapping(outputs[0], res);
}

Error PyTorchModelLoader::loadDiv(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 2, outputs, 1));

  glow::NodeValue res;
  ASSIGN_VALUE_OR_RETURN_ERR(
      res, loadArithmeticNode<glow::DivNode>("div", inputs[0], inputs[1]));

  return addValueMapping(outputs[0], res);
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

  glow::NodeValue res;
  ASSIGN_VALUE_OR_RETURN_ERR(
      res, loadArithmeticNode<glow::AddNode>("add", inputs[0], inputs[1]));

  return addValueMapping(outputs[0], res);
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

  glow::NodeValue res;
  ASSIGN_VALUE_OR_RETURN_ERR(
      res, loadArithmeticNode<glow::SubNode>("sub", inputs[0], inputs[1]));

  return addValueMapping(outputs[0], res);
}

Error PyTorchModelLoader::loadRsub(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 3, outputs, 1));

  // TODO: extend this to allow non-constant scalars.
  int64_t scalar;
  ASSIGN_VALUE_OR_RETURN_ERR(scalar,
                             iValToInt(getGlowIValueForValue(inputs[2])));
  RETURN_ERR_IF_NOT(scalar == 1,
                    glow::strFormat("Scalar must have value equal 1."));

  glow::NodeValue res;
  ASSIGN_VALUE_OR_RETURN_ERR(
      res, loadArithmeticNode<glow::SubNode>("sub", inputs[1], inputs[0]));

  return addValueMapping(outputs[0], res);
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

  // Convert negative dimension index into corresponding positive index
  auto origDim = dim;
  if (dim < 0) {
    dim += input.dims().size();
  }

  RETURN_ERR_IF_NOT(dim < input.dims().size() && dim >= 0,
                    strFormat("Dim value of %ld is out of range. Valid values "
                              "are in the range [-%ld, %ld]",
                              origDim, input.dims().size(),
                              input.dims().size() - 1));

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

Error PyTorchModelLoader::loadFusedConcat(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, -1, outputs, 1));

  // In the case of a single input, just return it.
  if (inputs.size() == 1) {
    glow::NodeValue input;
    ASSIGN_VALUE_OR_RETURN_ERR(input, getGlowNodeValueForValue(inputs[0]));
    return addValueMapping(outputs[0], input);
  }

  int64_t dim = ptNode->i(at::attr::dim);

  std::vector<glow::NodeValue> glowInputs;

  // Get number of input dimensions
  glow::NodeValue glowInput0;
  ASSIGN_VALUE_OR_RETURN_ERR(glowInput0, getGlowNodeValueForValue(inputs[0]));
  size_t numInputDims = glowInput0.dims().size();

  // Convert negative dimension index into corresponding positive index
  auto origDim = dim;
  if (dim < 0) {
    dim += numInputDims;
  }

  for (size_t i = 0; i < inputs.size(); ++i) {
    glow::NodeValue glowInput;
    ASSIGN_VALUE_OR_RETURN_ERR(glowInput, getGlowNodeValueForValue(inputs[i]));

    RETURN_ERR_IF_NOT(numInputDims == glowInput.dims().size(),
                      "All inputs must have the same number of dimensions.");

    RETURN_ERR_IF_NOT(dim < numInputDims && dim >= 0,
                      strFormat("Dim value of %ld is out of range. Valid "
                                "values are in the range [-%ld, %ld]",
                                origDim, numInputDims, numInputDims - 1));

    glowInputs.push_back(std::move(glowInput));
  }

  return addValueMapping(outputs[0], F_.createConcat("cat", glowInputs, dim));
}

Error PyTorchModelLoader::loadFusedStack(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, -1, outputs, 1));

  // In the case of a single input, just return it.
  if (inputs.size() == 1) {
    glow::NodeValue input;
    ASSIGN_VALUE_OR_RETURN_ERR(input, getGlowNodeValueForValue(inputs[0]));
    return addValueMapping(outputs[0], input);
  }

  int64_t dim = ptNode->i(at::attr::dim);

  RETURN_ERR_IF_NOT(dim >= 0, "Negative stack dims not supported yet.");

  std::vector<glow::NodeValue> glowInputs;
  for (size_t i = 0; i < inputs.size(); ++i) {
    glow::NodeValue glowInput;
    ASSIGN_VALUE_OR_RETURN_ERR(glowInput, getGlowNodeValueForValue(inputs[i]));

    // +1 because stack adds an extra dimension
    RETURN_ERR_IF_NOT(
        dim < glowInput.dims().size() + 1,
        "Dim must be less than the rank of inputs plus the added dimension");

    glowInputs.push_back(std::move(glowInput));
  }

  auto concat = F_.createConcat("stack_concat", glowInputs, dim)->getResult();
  auto concatDims = concat.dims();

  size_t numInputs = inputs.size();
  std::vector<glow::dim_t> reshapeDims;

  for (size_t i = 0; i < concatDims.size(); ++i) {
    if (i == dim) {
      reshapeDims.push_back(numInputs);
      reshapeDims.push_back(concatDims[i] / numInputs);
    } else {
      reshapeDims.push_back(concatDims[i]);
    }
  }

  // Handle the case when dim is the innermost dimension.
  if (reshapeDims.size() == concatDims.size()) {
    reshapeDims.back() /= numInputs;
    reshapeDims.push_back(numInputs);
  }

  auto reshape =
      F_.createReshape("stack_reshape", concat, reshapeDims)->getResult();

  return addValueMapping(outputs[0], reshape);
}

Error PyTorchModelLoader::loadReshape(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 2, outputs, 1));

  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(
      input, getGlowNodeValueForValue(inputs[ReshapeInputs::input]));

  std::vector<int64_t> *shapeOrignal;
  ASSIGN_VALUE_OR_RETURN_ERR(shapeOrignal, iValToIntList(getGlowIValueForValue(
                                               inputs[ReshapeInputs::shape])));

  // Copy shape so we can modify it.
  std::vector<int64_t> shape = *shapeOrignal;

  // Get total size of input.
  size_t inputTotalDims = input.getType()->size();

  // Get total size of shape, count -1 as 1, store index of -1 if found.
  int64_t negOneIndex = -1;
  size_t shapeTotalDims = 1;
  for (size_t i = 0; i < shape.size(); ++i) {
    int64_t val = shape[i];
    if (val > 0) {
      shapeTotalDims *= val;
    } else if (val == -1) {
      RETURN_ERR_IF_NOT(negOneIndex == -1,
                        "At most one negative value allowed in shape");
      negOneIndex = i;
    } else {
      return MAKE_ERR(
          strFormat("Found an invalid shape input value: % " PRId64, val));
    }
  }

  // If there was a negative index, replace it with the remaining dims in input.
  if (negOneIndex >= 0) {
    shape[negOneIndex] = inputTotalDims / shapeTotalDims;
  }

  c10::ScalarType dtype;
  RETURN_IF_ERR(getCorrectTypeMapping(dtype, inputs[ReshapeInputs::input]));
  return addValueMapping(
      outputs[0], F_.createReshape("reshape", input, castVector<dim_t>(shape)),
      dtype);
}

Error PyTorchModelLoader::loadView(const torch::jit::Node *ptNode) {
  // loadView is just like Reshape, except reshape should call contiguous
  // for non-contiguous data and view should fail
  return PyTorchModelLoader::loadReshape(ptNode);
}

Error PyTorchModelLoader::loadRelu(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 1, outputs, 1));

  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(input, getGlowNodeValueForValue(inputs[0]));

  glow::ReluNode *glowNode = F_.createRELU("relu", input);

  c10::ScalarType dtype;
  RETURN_IF_ERR(getCorrectTypeMapping(dtype, inputs[0]));
  return addValueMapping(outputs[0], glowNode->getResult(), dtype);
}

Error PyTorchModelLoader::loadGelu(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 1, outputs, 1));

  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(input, getGlowNodeValueForValue(inputs[0]));

  auto output = F_.createGELU("gelu", input)->getNthResult(0);
  return addValueMapping(outputs[0], output);
}

Error PyTorchModelLoader::loadTanh(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 1, outputs, 1));

  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(input, getGlowNodeValueForValue(inputs[0]));

  glow::TanhNode *glowNode = F_.createTanh("tanh", input);
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

Error PyTorchModelLoader::loadPow(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 2, outputs, 1));

  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(input, getGlowNodeValueForValue(inputs[0]));

  // NB: exponent may also be a Tensor. Will support if needed.
  float exponent;
  ASSIGN_VALUE_OR_RETURN_ERR(exponent,
                             iValToDouble(getGlowIValueForValue(inputs[1])));

  glow::PowNode *glowNode = F_.createPow("pow", input, exponent);
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
  // we transpose them.
  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(
      input, getGlowNodeValueForValue(inputs[ConvInputs::input]));

  // Glow expects conv weights to be in CRSK but PyTorch keeps them in CKRS
  // so we transpose them. C - output_depth, R - filter_height, S -
  // filter_width, K - input_depth.
  glow::NodeValue weights;
  ASSIGN_VALUE_OR_RETURN_ERR(
      weights, getGlowNodeValueForValue(inputs[ConvInputs::weights]));

  RETURN_ERR_IF_NOT((input.dims().size() == 4 || input.dims().size() == 5) &&
                        input.dims().size() == weights.dims().size(),
                    "Expect 4 dims in input and weights for conv2d and 5 dims "
                    "in input and weights for conv3d");
  bool isConv3d = input.dims().size() == 5;
  if (isConv3d) {
    input = F_.createTranspose("conv_input_transposed", input, NCTHW2NTHWC);
    weights =
        F_.createTranspose("conv_weights_transposed", weights, NCTHW2NTHWC);
  } else {
    input = F_.createTranspose("conv_input_transposed", input, NCHW2NHWC);
    weights = F_.createTranspose("conv_weights_transposed", weights, NCHW2NHWC);
  }

  // If a bias was provided then use it otherwise create a 0 bias.
  glow::dim_t biasDim = weights.dims()[0];
  glow::NodeValue bias = loadNodeValueOrCreateBroadcastedConstant(
      inputs[ConvInputs::bias], "conv_bias",
      glow::Type(ElemKind::FloatTy, {biasDim}), 0);

  std::vector<glow::unsigned_t> strides;
  ASSIGN_VALUE_OR_RETURN_ERR(
      strides, castVector<glow::unsigned_t>(expandIntIValIfNeeded(
                   getGlowIValueForValue(inputs[ConvInputs::stride]),
                   input.dims().size() - 2)));

  std::vector<glow::unsigned_t> pads;
  if (isConv3d) {
    std::vector<glow::unsigned_t> pad;

    ASSIGN_VALUE_OR_RETURN_ERR(
        pad, castVector<glow::unsigned_t>(expandIntIValIfNeeded(
                 getGlowIValueForValue(inputs[ConvInputs::padding]), 3)));
    pads = {pad[0], pad[0], pad[1], pad[1], pad[2], pad[2]};
  } else {
    glow::unsigned_t pad;
    ASSIGN_VALUE_OR_RETURN_ERR(
        pad, static_cast_expected<glow::unsigned_t>(contractIntIValIfNeeded(
                 getGlowIValueForValue(inputs[ConvInputs::padding]))));
    pads = {pad, pad, pad, pad};
  }

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
  std::vector<glow::unsigned_t> kernels;
  if (isConv3d) {
    glow::ShapeNTHWC weights3DShape(weights.dims());
    kernels = {static_cast<glow::unsigned_t>(weights3DShape.t),
               static_cast<glow::unsigned_t>(weights3DShape.h),
               static_cast<glow::unsigned_t>(weights3DShape.w)};
  } else {
    glow::ShapeNHWC weightsShape(weights.dims());
    kernels = {static_cast<glow::unsigned_t>(weightsShape.h),
               static_cast<glow::unsigned_t>(weightsShape.w)};
  }

  glow::TypeRef outTy;
  if (isConv3d) {
    glow::ShapeNTHWC input3DShape(input.dims());
    auto outSz = glow::calculate3DConvPoolOutputDims(
        input3DShape.t, input3DShape.h, input3DShape.w, kernels, strides, pads);
    std::array<glow::dim_t, 5> outDims = {{input.dims()[0],
                                           outSz.temporal_frames, outSz.height,
                                           outSz.width, weights.dims()[0]}};
    outTy = F_.getParent()->uniqueType(glow::ElemKind::FloatTy, outDims);
  } else {
    glow::ShapeNHWC inputShape(input.dims());
    auto outSz = glow::calculateConvPoolOutputDims(
        inputShape.h, inputShape.w, kernels, strides, pads, dilation);
    std::array<glow::dim_t, 4> outDims = {
        {input.dims()[0], outSz.first, outSz.second, weights.dims()[0]}};
    outTy = F_.getParent()->uniqueType(glow::ElemKind::FloatTy, outDims);
  }

  glow::TransposeNode *output = nullptr;
  if (isConv3d) {
    glow::Convolution3DNode *conv = F_.createConv3D(
        "conv3d", input, weights, bias, outTy, kernels, strides, pads, groups);
    output = F_.createTranspose("conv_output_transposed", conv->getResult(),
                                NTHWC2NCTHW);
  } else {
    glow::ConvolutionNode *conv =
        F_.createConv("conv", input, weights, bias, outTy, kernels, strides,
                      pads, groups, dilation);
    output = F_.createTranspose("conv_output_transposed", conv->getResult(),
                                NHWC2NCHW);
  }
  return addValueMapping(outputs[0], output->getResult());
}

Error PyTorchModelLoader::loadLayerNorm(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 6, outputs, 1));

  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(
      input, getGlowNodeValueForValue(inputs[LayerNormInputs::input]));

  float eps = 1e-5;
  if (hasGlowIValueForValue(inputs[LayerNormInputs::eps])) {
    ASSIGN_VALUE_OR_RETURN_ERR(
        eps, iValToDouble(getGlowIValueForValue(inputs[LayerNormInputs::eps])));
  }

  std::vector<int64_t> *normalizedShape;
  ASSIGN_VALUE_OR_RETURN_ERR(normalizedShape,
                             iValToIntList(getGlowIValueForValue(
                                 inputs[LayerNormInputs::normalized_shape])));

  std::vector<glow::dim_t> normalizedShapeCast =
      castVector<glow::dim_t>(*normalizedShape);

  glow::NodeValue weight = loadNodeValueOrCreateBroadcastedConstant(
      inputs[LayerNormInputs::weight], "layernorm_weight",
      glow::Type(ElemKind::FloatTy, normalizedShapeCast), 1.0);

  glow::NodeValue bias = loadNodeValueOrCreateBroadcastedConstant(
      inputs[LayerNormInputs::bias], "layernorm_bias",
      glow::Type(ElemKind::FloatTy, normalizedShapeCast), 0.0);

  auto output =
      F_.createLayerNormalization("layernorm", input, weight, bias, eps)
          ->getResult();

  return addValueMapping(outputs[0], output);
}

Error PyTorchModelLoader::loadBatchNorm(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 9, outputs, 1));

  bool training;
  ASSIGN_VALUE_OR_RETURN_ERR(training, iValToBool(getGlowIValueForValue(
                                           inputs[BatchNormInputs::training])));
  RETURN_ERR_IF_NOT(training == false, "Don't support BatchNorm training yet.");

  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(
      input, getGlowNodeValueForValue(inputs[BatchNormInputs::input]));
  RETURN_ERR_IF_NOT(
      input.dims().size() == 4,
      glow::strFormat("Number input dimensions must be equal to 4, got %lu",
                      input.dims().size()));

  size_t numChannels = input.dims()[1];

  glow::NodeValue weights = loadNodeValueOrCreateBroadcastedConstant(
      inputs[BatchNormInputs::weights], "batchnorm_weights",
      glow::Type(ElemKind::FloatTy, {numChannels}), 1.0);

  glow::NodeValue bias = loadNodeValueOrCreateBroadcastedConstant(
      inputs[BatchNormInputs::bias], "batchnorm_bias",
      glow::Type(ElemKind::FloatTy, {numChannels}), 0.0);

  glow::NodeValue mean;
  ASSIGN_VALUE_OR_RETURN_ERR(
      mean, getGlowNodeValueForValue(inputs[BatchNormInputs::running_mean]));

  glow::NodeValue var;
  ASSIGN_VALUE_OR_RETURN_ERR(
      var, getGlowNodeValueForValue(inputs[BatchNormInputs::running_var]));

  float momentum;
  ASSIGN_VALUE_OR_RETURN_ERR(
      momentum, to32Bit(iValToDouble(
                    getGlowIValueForValue(inputs[BatchNormInputs::momentum]))));

  float epsilon;
  ASSIGN_VALUE_OR_RETURN_ERR(
      epsilon, to32Bit(iValToDouble(
                   getGlowIValueForValue(inputs[BatchNormInputs::eps]))));

  // Input is in NCHW.
  glow::unsigned_t channelIdx = 1;

  glow::BatchNormalizationNode *bn =
      F_.createBatchNormalization("batchnorm", input, bias, weights, mean, var,
                                  channelIdx, epsilon, momentum);
  return addValueMapping(outputs[0], bn->getResult());
}

Error PyTorchModelLoader::loadDropout(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 3, outputs, 1));

  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(
      input, getGlowNodeValueForValue(inputs[DropoutInputs::input]));

  bool training;
  ASSIGN_VALUE_OR_RETURN_ERR(training, iValToBool(getGlowIValueForValue(
                                           inputs[DropoutInputs::training])));
  RETURN_ERR_IF_NOT(!training, "Glow doesn't support dropout training yet");

  // Dropout not in training mode is a noop.
  return addValueMapping(outputs[0], input);
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
  ASSIGN_VALUE_OR_RETURN_ERR(
      outScale, to32Bit(iValToDouble(
                    getGlowIValueForValue(inputs[QuantizeInputs::scale]))));

  // zero_point
  int32_t outOffset;
  ASSIGN_VALUE_OR_RETURN_ERR(
      outOffset,
      iValToInt(getGlowIValueForValue(inputs[QuantizeInputs::zero_point])));

  // dtype, we only support quantize to int8 for now
  int32_t outDtype;
  ASSIGN_VALUE_OR_RETURN_ERR(outDtype, iValToInt(getGlowIValueForValue(
                                           inputs[QuantizeInputs::dtype])));

  glow::TypeRef inputType = input.getType();
  auto outDims = inputType->dims();

  glow::TypeRef outTy;
  c10::ScalarType dtype;
  if (outDtype == (int32_t)at::ScalarType::QUInt8) {
    outTy = F_.getParent()->uniqueType(ElemKind::Int8QTy, outDims, outScale,
                                       outOffset - OFFSETSHIFT);
    dtype = c10::ScalarType::QUInt8;

  } else if (outDtype == (int32_t)at::ScalarType::QInt8) {
    outTy = F_.getParent()->uniqueType(ElemKind::Int8QTy, outDims, outScale,
                                       outOffset);
    dtype = c10::ScalarType::QInt8;
  } else {
    return MAKE_ERR("Quantize only supports QUInt8 and QInt8");
  }
  glow::QuantizeNode *qn = F_.createQuantize("quantize", input, outTy);

  return addValueMapping(outputs[0], qn->getResult(), dtype);
}

Error PyTorchModelLoader::loadDequantize(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 1, outputs, 1));

  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(input, getGlowNodeValueForValue(inputs[0]));

  glow::DequantizeNode *dn = F_.createDequantize("dequantize", input);

  c10::ScalarType dtype;
  RETURN_IF_ERR(getCorrectTypeMapping(dtype, inputs[0]));
  return addValueMapping(outputs[0], dn->getResult(), dtype);
}

Error PyTorchModelLoader::loadQuantizedConvRelu(
    const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  glow::NodeValue output;
  ASSIGN_VALUE_OR_RETURN_ERR(output,
                             loadQuantizedConvImpl(ptNode, true /* isRelu */));

  c10::ScalarType dtype;
  RETURN_IF_ERR(getCorrectTypeMapping(dtype, inputs[0]));
  return addValueMapping(outputs[0], output, dtype);
}

Error PyTorchModelLoader::loadQuantizedConv(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  glow::NodeValue output;
  ASSIGN_VALUE_OR_RETURN_ERR(output,
                             loadQuantizedConvImpl(ptNode, false /* isRelu */));

  c10::ScalarType dtype;
  RETURN_IF_ERR(getCorrectTypeMapping(dtype, inputs[0]));
  return addValueMapping(outputs[0], output, dtype);
}

Error PyTorchModelLoader::loadQuantizedConvUnpacked(
    const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 9, outputs, 1));

  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(input, getGlowNodeValueForValue(inputs[0]));

  bool isConv3d = input.dims().size() == 5;
  if (isConv3d) {
    input = F_.createTranspose("qconv_input_transposed", input, NCTHW2NTHWC);
  } else {
    input = F_.createTranspose("qconv_input_transposed", input, NCHW2NHWC);
  }
  std::unordered_map<std::string, int8_t> input_mapping = {};
  if (isConv3d) {
    input_mapping["input"] = QuantizedUnpackedConv3dInputs::input;
    input_mapping["weights"] = QuantizedUnpackedConv3dInputs::weights;
    input_mapping["bias"] = QuantizedUnpackedConv3dInputs::bias;
    input_mapping["stride"] = QuantizedUnpackedConv3dInputs::stride;
    input_mapping["padding"] = QuantizedUnpackedConv3dInputs::padding;
    input_mapping["group"] = QuantizedUnpackedConv3dInputs::group;
    input_mapping["scale"] = QuantizedUnpackedConv3dInputs::scale;
    input_mapping["zero_point"] = QuantizedUnpackedConv3dInputs::zero_point;

  } else {
    input_mapping["input"] = QuantizedUnpackedConv2dInputs::input;
    input_mapping["weights"] = QuantizedUnpackedConv2dInputs::weights;
    input_mapping["bias"] = QuantizedUnpackedConv2dInputs::bias;
    input_mapping["stride"] = QuantizedUnpackedConv2dInputs::stride;
    input_mapping["padding"] = QuantizedUnpackedConv2dInputs::padding;
    input_mapping["dilation"] = QuantizedUnpackedConv2dInputs::dilation;
    input_mapping["group"] = QuantizedUnpackedConv2dInputs::group;
    input_mapping["scale"] = QuantizedUnpackedConv2dInputs::scale;
    input_mapping["zero_point"] = QuantizedUnpackedConv2dInputs::zero_point;
  }

  glow::NodeValue weights;
  ASSIGN_VALUE_OR_RETURN_ERR(
      weights, getGlowNodeValueForValue(inputs[input_mapping["weights"]]));
  weights = rescaleUIntToInt(weights);
  if (isConv3d) {
    weights =
        F_.createTranspose("qconv_weights_transposed", weights, NCTHW2NTHWC);
  } else {
    weights =
        F_.createTranspose("qconv_weights_transposed", weights, NCHW2NHWC);
  }

  glow::NodeValue bias = loadNodeValueOrCreateBroadcastedConstant(
      inputs[input_mapping["bias"]], "qconv_bias",
      glow::Type(ElemKind::FloatTy, {weights.dims()[0]}), 0.0);

  auto biasType = F_.getParent()->uniqueType(
      glow::ElemKind::Int32QTy, bias.dims(),
      input.getType()->getScale() * weights.getType()->getScale(), 0);
  bias = F_.createQuantize("quantize_bias", bias, biasType);

  std::vector<glow::unsigned_t> strides;
  ASSIGN_VALUE_OR_RETURN_ERR(
      strides, castVector<glow::unsigned_t>(expandIntIValIfNeeded(
                   getGlowIValueForValue(inputs[input_mapping["stride"]]),
                   input.dims().size() - 2)));

  // pads
  std::vector<glow::unsigned_t> pads;
  if (isConv3d) {
    std::vector<glow::unsigned_t> pad;

    ASSIGN_VALUE_OR_RETURN_ERR(
        pad, castVector<glow::unsigned_t>(expandIntIValIfNeeded(
                 getGlowIValueForValue(inputs[input_mapping["padding"]]), 3)));
    pads = {pad[0], pad[0], pad[1], pad[1], pad[2], pad[2]};
  } else {
    glow::unsigned_t pad;
    ASSIGN_VALUE_OR_RETURN_ERR(
        pad, static_cast_expected<glow::unsigned_t>(contractIntIValIfNeeded(
                 getGlowIValueForValue(inputs[input_mapping["padding"]]))));
    pads = {pad, pad, pad, pad};
  }

  glow::unsigned_t groups;
  ASSIGN_VALUE_OR_RETURN_ERR(
      groups, static_cast_expected<glow::unsigned_t>(iValToInt(
                  getGlowIValueForValue(inputs[input_mapping["group"]]))));

  float outScale;
  ASSIGN_VALUE_OR_RETURN_ERR(outScale, iValToDouble(getGlowIValueForValue(
                                           inputs[input_mapping["scale"]])));

  int32_t outOffset;
  ASSIGN_VALUE_OR_RETURN_ERR(
      outOffset,
      iValToInt(getGlowIValueForValue(inputs[input_mapping["zero_point"]])));

  // calc output type
  glow::unsigned_t dilation = 0;
  glow::TypeRef outTy;
  std::vector<glow::unsigned_t> kernels;
  if (isConv3d) {
    glow::ShapeNTHWC input3DShape(input.dims());
    glow::ShapeNTHWC weight3DShape(weights.dims());
    kernels = {static_cast<glow::unsigned_t>(weight3DShape.t),
               static_cast<glow::unsigned_t>(weight3DShape.h),
               static_cast<glow::unsigned_t>(weight3DShape.w)};

    auto outSz = glow::calculate3DConvPoolOutputDims(
        input3DShape.t, input3DShape.h, input3DShape.w, kernels, strides, pads);
    std::array<glow::dim_t, 5> outDims = {{input.dims()[0],
                                           outSz.temporal_frames, outSz.height,
                                           outSz.width, weights.dims()[0]}};
    outTy = F_.getParent()->uniqueType(glow::ElemKind::Int8QTy, outDims,
                                       outScale, outOffset - OFFSETSHIFT);
  } else {
    glow::ShapeNHWC inputShape(input.dims());
    glow::ShapeNHWC weightShape(weights.dims());
    kernels = {static_cast<glow::unsigned_t>(weightShape.h),
               static_cast<glow::unsigned_t>(weightShape.w)};
    ASSIGN_VALUE_OR_RETURN_ERR(
        dilation, static_cast_expected<glow::unsigned_t>(
                      contractIntIValIfNeeded(getGlowIValueForValue(
                          inputs[QuantizedUnpackedConv2dInputs::dilation]))));
    auto outSz = glow::calculateConvPoolOutputDims(
        inputShape.h, inputShape.w, kernels, strides, pads, dilation);
    std::array<glow::dim_t, 4> outDims = {
        {input.dims()[0], outSz.first, outSz.second, weightShape.n}};
    outTy = F_.getParent()->uniqueType(glow::ElemKind::Int8QTy, outDims,
                                       outScale, outOffset - OFFSETSHIFT);
  }

  glow::TransposeNode *output;
  if (isConv3d) {
    glow::Convolution3DNode *qconv = F_.createConv3D(
        "qconv", input, weights, bias, outTy, kernels, strides, pads, groups);
    output = F_.createTranspose("qconv_output_transposed", qconv->getResult(),
                                NTHWC2NCTHW);
  } else {
    glow::ConvolutionNode *qconv =
        F_.createConv("qconv", input, weights, bias, outTy, kernels, strides,
                      pads, groups, dilation);
    output = F_.createTranspose("qconv_output_transposed", qconv->getResult(),
                                NHWC2NCHW);
  }

  c10::ScalarType dtype;
  RETURN_IF_ERR(getCorrectTypeMapping(dtype, inputs[0]));
  return addValueMapping(outputs[0], output->getResult(), dtype);
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
  RETURN_ERR_IF_NOT(dilation == 1, "Dilation value must be equal to 1, "
                                   "maxpool dilation not yet supported.");

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

  c10::ScalarType dtype;
  RETURN_IF_ERR(getCorrectTypeMapping(dtype, inputs[MaxPoolInputs::input]));
  return addValueMapping(outputs[0], output, dtype);
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

  c10::ScalarType dtype;
  RETURN_IF_ERR(getCorrectTypeMapping(dtype, inputs[AvgPoolInputs::input]));
  return addValueMapping(outputs[0], output, dtype);
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
  // transpose them.
  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(
      input, getGlowNodeValueForValue(inputs[AdaptiveAvgPoolInputs::input]));

  size_t inputH = input.dims()[1];
  size_t inputW = input.dims()[2];
  input = F_.createTranspose("adaptive_avg_pool2d_input_transposed", input,
                             NCHW2NHWC);

  // OutputSize defaults to size of input if not provided.
  std::vector<glow::dim_t> outputSize;
  if (hasGlowIValueForValue(inputs[AdaptiveAvgPoolInputs::output_size])) {
    ASSIGN_VALUE_OR_RETURN_ERR(
        outputSize,
        castVector<glow::dim_t>(expandIntIValIfNeeded(
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

  c10::ScalarType dtype;
  RETURN_IF_ERR(
      getCorrectTypeMapping(dtype, inputs[AdaptiveAvgPoolInputs::input]));
  return addValueMapping(outputs[0], output, dtype);
}

Error PyTorchModelLoader::loadT(const torch::jit::Node *ptNode) {
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

  c10::ScalarType dtype;
  RETURN_IF_ERR(getCorrectTypeMapping(dtype, inputs[0]));
  return addValueMapping(outputs[0], output, dtype);
}

Error PyTorchModelLoader::loadTranspose(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 3, outputs, 1));

  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(
      input, getGlowNodeValueForValue(inputs[TransposeInputs::input]));

  int64_t dim0;
  ASSIGN_VALUE_OR_RETURN_ERR(
      dim0, iValToInt(getGlowIValueForValue(inputs[TransposeInputs::dim0])));

  int64_t dim1;
  ASSIGN_VALUE_OR_RETURN_ERR(
      dim1, iValToInt(getGlowIValueForValue(inputs[TransposeInputs::dim1])));

  // Adjust dim0 for negative dimensions
  auto origDim0 = dim0;
  if (dim0 < 0) {
    dim0 += input.dims().size();
  }

  RETURN_ERR_IF_NOT(dim0 < input.dims().size() && dim0 >= 0,
                    strFormat("Dim0 value of %ld is out of range. Valid values "
                              "are in the range [-%ld, %ld]",
                              origDim0, input.dims().size(),
                              input.dims().size() - 1));

  // Adjust dim1 for negative dimensions
  auto origDim1 = dim1;
  if (dim1 < 0) {
    dim1 += input.dims().size();
  }

  RETURN_ERR_IF_NOT(dim1 < input.dims().size() && dim1 >= 0,
                    strFormat("Dim1 value of %ld is out of range. Valid values "
                              "are in the range [-%ld, %ld]",
                              origDim1, input.dims().size(),
                              input.dims().size() - 1));

  std::vector<glow::unsigned_t> shuffle(input.dims().size());
  std::iota(shuffle.begin(), shuffle.end(), 0);
  std::swap(shuffle[dim0], shuffle[dim1]);

  auto *output = F_.createTranspose("transpose", input, shuffle);

  c10::ScalarType dtype;
  RETURN_IF_ERR(getCorrectTypeMapping(dtype, inputs[TransposeInputs::input]));
  return addValueMapping(outputs[0], output->getResult(), dtype);
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

Error PyTorchModelLoader::loadMean(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, -2, outputs, 1));

  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(
      input, getGlowNodeValueForValue(inputs[MeanInputs::input]));

  std::vector<int64_t> *axis;
  if (hasGlowIValueForValue(inputs[MeanInputs::axis],
                            /*ignoreNones*/ true)) {
    ASSIGN_VALUE_OR_RETURN_ERR(
        axis, iValToIntList(getGlowIValueForValue(inputs[MeanInputs::axis])));
    std::sort(axis->begin(), axis->end(), std::greater<int64_t>());
    for (auto i : *axis) {
      input = F_.createBatchedReduceMean("mean", input, i);
    }
  } else {
    for (int i = input.dims().size() - 1; i >= 0; i--) {
      input = F_.createBatchedReduceMean("mean", input, i);
    }
  }

  if (inputs.size() > 2 &&
      hasGlowIValueForValue(inputs[MeanInputs::keepdims])) {
    bool keepdims;
    ASSIGN_VALUE_OR_RETURN_ERR(keepdims, iValToBool(getGlowIValueForValue(
                                             inputs[MeanInputs::keepdims])));
    if (keepdims == true) {
      RETURN_ERR("We don't currently support keeping dims");
    }
  }

  return addValueMapping(outputs[0], input);
}

Expected<glow::NodeValue>
PyTorchModelLoader::loadMatMulImpl(glow::NodeValue lhs, glow::NodeValue rhs) {
  glow::NodeValue output;

  auto lhsRank = lhs.dims().size();
  auto rhsRank = rhs.dims().size();

  if (lhsRank == 1 && rhsRank == 1) {
    // NOTE: Only Glow's 2d dotproduct operator accumulates so we prepend
    // 1 to dims to turn inputs into 2d.
    lhs = F_.createReshape("reshape_matmul_lhs", lhs, {1, lhs.dims()[0]});
    rhs = F_.createReshape("reshape_matmul_rhs", rhs, {1, rhs.dims()[0]});
    output = F_.createDotProduct("dotprod", lhs, rhs);
  } else if (lhsRank == 2 && rhsRank == 2) {
    output = F_.createMatMul("matmul", lhs, rhs);
  } else if (lhsRank == 1 && rhsRank == 2) {
    // Prepend a 1 to lhs's shape if it's 1d.
    lhs = F_.createReshape("reshape_matmul_lhs", lhs, {1, lhs.dims()[0]});
    output = F_.createMatMul("matmul", lhs, rhs);
    output = F_.createReshape("reshape_matmul_output", output, {rhs.dims()[1]});
  } else if (lhsRank == 2 && rhsRank == 1) {
    // Append a 1 to rhs's shape if it's 1d.
    rhs = F_.createReshape("reshape_matmul_rhs", rhs, {rhs.dims()[0], 1});
    output = F_.createMatMul("matmul", lhs, rhs);
    output = F_.createReshape("reshape_matmul_output", output, {lhs.dims()[0]});
  } else {
    // Prepend a 1 to lhs or append 1 to rhs so that they are both at least
    // rank 2.
    const bool lhsPrepend = lhsRank == 1;
    const bool rhsAppend = rhsRank == 1;
    if (lhsPrepend) {
      std::vector<glow::dim_t> newDims = lhs.dims();
      newDims.insert(newDims.begin(), 1);
      lhs = F_.createReshape("reshape_matmul_lhs", lhs, newDims);
    }
    if (rhsAppend) {
      std::vector<glow::dim_t> newDims = rhs.dims();
      newDims.push_back(1);
      rhs = F_.createReshape("reshape_matmul_rhs", rhs, newDims);
    }

    // Reshape inputs to be the same size, pad outer dimensions with 1s.
    size_t maxRank = std::max(lhsRank, rhsRank);
    lhs = F_.createReshape("reshape_matmul_lhs", lhs,
                           getExpandDims(lhs.dims(), maxRank));
    rhs = F_.createReshape("reshape_matmul_rhs", rhs,
                           getExpandDims(rhs.dims(), maxRank));

    // Compute target dims template (0s for the innermost dimensions)
    std::vector<glow::dim_t> targetDims;
    ASSIGN_VALUE_OR_RETURN_ERR(
        targetDims, computeBroadcastedMatMulTargetDims(lhs.dims(), rhs.dims()));

    // Compute the dimensions that lhs should be broadcast to.
    auto lhsTargetDims = targetDims;
    std::copy(lhs.dims().end() - 2, lhs.dims().end(), lhsTargetDims.end() - 2);

    // Compute the dimensions that rhs should be broadcast to.
    auto rhsTargetDims = targetDims;
    std::copy(rhs.dims().end() - 2, rhs.dims().end(), rhsTargetDims.end() - 2);

    // Compute the dimensions for the final output of batched matmul.
    auto outputTargetDims = targetDims;
    outputTargetDims[outputTargetDims.size() - 2] =
        lhsTargetDims[lhsTargetDims.size() - 2];
    outputTargetDims[outputTargetDims.size() - 1] =
        rhsTargetDims[rhsTargetDims.size() - 1];

    // Broadcast lhs and rhs so that they match their targetDims.
    lhs = F_.createBroadcast("lhsBroadcast", lhs, lhsTargetDims, 0);
    rhs = F_.createBroadcast("rhsBroadcast", rhs, rhsTargetDims, 0);

    // Reshape both inputs to be rank 3 by collapsing their outer dimensions
    // since BatchMatMul can only handle rank 3 inputs.
    lhs = F_.createReshape("reshape_matmul_lhs", lhs,
                           getContractDims(lhs.dims(), 3));
    rhs = F_.createReshape("reshape_matmul_rhs", rhs,
                           getContractDims(rhs.dims(), 3));

    // Perform BatchMatMul
    output = F_.createBatchMatMul("matmul", lhs, rhs);

    // Reshape the output of BatchMatMul to expand the outer dimensions again.
    output =
        F_.createReshape("matmul_output_reshape", output, outputTargetDims);

    // If a 1 was prepended to lhs or a 1 was appended to rhs at the beginning,
    // undo that now.
    if (lhsPrepend) {
      std::vector<glow::dim_t> newDims(output.dims().begin(),
                                       output.dims().end() - 2);
      newDims.push_back(*(output.dims().end() - 1));
      output = F_.createReshape("matmul_output_reshape", output, newDims);
    }
    if (rhsAppend) {
      std::vector<glow::dim_t> newDims(output.dims().begin(),
                                       output.dims().end() - 1);
      output = F_.createReshape("matmul_output_reshape", output, newDims);
    }
  }

  return output;
}

Error PyTorchModelLoader::loadMatMul(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 2, outputs, 1));

  glow::NodeValue lhs;
  ASSIGN_VALUE_OR_RETURN_ERR(lhs, getGlowNodeValueForValue(inputs[0]));
  glow::NodeValue rhs;
  ASSIGN_VALUE_OR_RETURN_ERR(rhs, getGlowNodeValueForValue(inputs[1]));

  glow::NodeValue output;
  ASSIGN_VALUE_OR_RETURN_ERR(output, loadMatMulImpl(lhs, rhs));

  return addValueMapping(outputs[0], output);
}

Error PyTorchModelLoader::loadMM(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 2, outputs, 1));

  glow::NodeValue lhs;
  ASSIGN_VALUE_OR_RETURN_ERR(lhs, getGlowNodeValueForValue(inputs[0]));
  glow::NodeValue rhs;
  ASSIGN_VALUE_OR_RETURN_ERR(rhs, getGlowNodeValueForValue(inputs[1]));

  // Check dimensions of inputs
  if (lhs.dims().size() != 2 || rhs.dims().size() != 2) {
    RETURN_ERR("aten::mm expects 2D matrices");
  }

  if (lhs.dims()[1] != rhs.dims()[0]) {
    RETURN_ERR("aten::mm does not broadcast");
  }

  auto output = F_.createMatMul("mm", lhs, rhs)->getResult();
  return addValueMapping(outputs[0], output);
}

Error PyTorchModelLoader::loadBmm(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 2, outputs, 1));

  glow::NodeValue lhs;
  ASSIGN_VALUE_OR_RETURN_ERR(lhs, getGlowNodeValueForValue(inputs[0]));
  glow::NodeValue rhs;
  ASSIGN_VALUE_OR_RETURN_ERR(rhs, getGlowNodeValueForValue(inputs[1]));

  // Check dimensions of inputs
  if (lhs.dims().size() != 3 || rhs.dims().size() != 3) {
    RETURN_ERR("aten::bmm expects 3D tensors");
  }

  if (lhs.dims()[2] != rhs.dims()[1]) {
    RETURN_ERR("aten::bmm does not broadcast");
  }

  auto output = F_.createBatchMatMul("bmm", lhs, rhs)->getResult();
  return addValueMapping(outputs[0], output);
}

Error PyTorchModelLoader::loadAddMM(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 5, outputs, 1));

  auto getScalar = [&](const torch::jit::Value *val) -> Expected<float> {
    if (!hasGlowIValueForValue(val)) {
      return 1.0;
    }

    GlowIValue *ival;
    ASSIGN_VALUE_OR_RETURN_ERR(ival, getGlowIValueForValue(val));

    if (ival->isDouble()) {
      return ival->toDouble();
    } else if (ival->isInt()) {
      return static_cast_expected<float>(ival->toInt());
    } else if (ival->isNone()) {
      return 1.0;
    } else {
      return MAKE_ERR("Unexpected scalar type");
    }
  };

  float alpha;
  ASSIGN_VALUE_OR_RETURN_ERR(alpha, getScalar(inputs[AddMMInputs::alpha]));

  float beta;
  ASSIGN_VALUE_OR_RETURN_ERR(beta, getScalar(inputs[AddMMInputs::beta]));

  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(
      input, getGlowNodeValueForValue(inputs[AddMMInputs::input]));

  if (beta != 1.0) {
    glow::Tensor t(ElemKind::FloatTy, input.dims());
    t.init(Tensor::InitKind::Broadcast, beta, F_.getParent()->getPRNG());
    auto *constant = F_.getParent()->createConstant("beta", std::move(t));
    input = F_.createMul("mul", constant, input)->getResult();
  }

  glow::NodeValue mat1;
  ASSIGN_VALUE_OR_RETURN_ERR(
      mat1, getGlowNodeValueForValue(inputs[AddMMInputs::mat1]));

  glow::NodeValue mat2;
  ASSIGN_VALUE_OR_RETURN_ERR(
      mat2, getGlowNodeValueForValue(inputs[AddMMInputs::mat2]));

  // Check dimensions of mat1 and mat2
  if (mat1.dims().size() != 2 || mat2.dims().size() != 2) {
    RETURN_ERR("aten::addmm expects 2D matrices");
  }

  if (mat1.dims()[1] != mat2.dims()[0]) {
    RETURN_ERR("aten::addmm does not broadcast mat1 or mat2");
  }

  auto matmul = F_.createMatMul("mm", mat1, mat2)->getResult();

  if (alpha != 1.0) {
    glow::Tensor t(ElemKind::FloatTy, matmul.dims());
    t.init(Tensor::InitKind::Broadcast, alpha, F_.getParent()->getPRNG());
    auto *constant = F_.getParent()->createConstant("alpha", std::move(t));
    matmul = F_.createMul("mul", constant, matmul)->getResult();
  }

  auto add = F_.createNodeWithBroadcast<glow::AddNode>("add", /*axis*/ -1,
                                                       input, matmul)
                 ->getResult();

  return addValueMapping(outputs[0], add);
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

Error PyTorchModelLoader::loadSlice(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 5, outputs, 1));

  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(
      input, getGlowNodeValueForValue(inputs[SliceInputs::input]));

  int64_t dim, start, end, step = 1;
  ASSIGN_VALUE_OR_RETURN_ERR(
      dim, iValToInt(getGlowIValueForValue(inputs[SliceInputs::dim])));
  ASSIGN_VALUE_OR_RETURN_ERR(
      start, iValToInt(getGlowIValueForValue(inputs[SliceInputs::start])));
  ASSIGN_VALUE_OR_RETURN_ERR(
      end, iValToInt(getGlowIValueForValue(inputs[SliceInputs::end])));
  ASSIGN_VALUE_OR_RETURN_ERR(
      step, iValToInt(getGlowIValueForValue(inputs[SliceInputs::step])));

  RETURN_ERR_IF_NOT(step == 1, "loadSlice only supports step == 1");
  int dimsSize = input.dims().size();
  std::vector<glow::dim_t> begins(dimsSize);
  std::vector<glow::dim_t> ends(dimsSize);

  for (int i = 0; i < dimsSize; ++i) {
    if (i == dim) {
      begins[i] = start;
      ends[i] = end > input.dims()[i] ? input.dims()[i] : end;
    } else {
      begins[i] = 0;
      ends[i] = input.dims()[i];
    }
  }
  auto *glowNode = F_.createSlice("sliceOutput", input, begins, ends);

  c10::ScalarType dtype;
  RETURN_IF_ERR(getCorrectTypeMapping(dtype, inputs[SliceInputs::input]));
  return addValueMapping(outputs[0], glowNode, dtype);
}

/// TODO: check Dtype is float (optional value).
Error PyTorchModelLoader::loadSoftMax(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 3, outputs, 1));

  glow::NodeValue in;
  ASSIGN_VALUE_OR_RETURN_ERR(
      in, getGlowNodeValueForValue(inputs[SoftMaxInputs::input]));

  int64_t dim;
  ASSIGN_VALUE_OR_RETURN_ERR(
      dim, iValToInt(getGlowIValueForValue(inputs[SoftMaxInputs::dim])));

  // Convert negative dimension index into corresponding positive index
  auto origDim = dim;
  if (dim < 0) {
    dim += in.dims().size();
  }

  RETURN_ERR_IF_NOT(dim < in.dims().size() && dim >= 0,
                    strFormat("Dim value of %ld is out of range. Valid values "
                              "are in the range [-%ld, %ld]",
                              origDim, in.dims().size(), in.dims().size() - 1));

  auto selected = F_.getParent()->createConstant(glow::ElemKind::Int64ITy,
                                                 {in.dims()[0], 1}, "selected");

  auto *FN = F_.createFlatten("reshapeInput", in, dim);
  auto *SM = F_.createSoftMax("SoftMax", FN, selected);
  auto origInDims = in.getType()->dims();
  auto *glowNode = F_.createReshape("reshapeOutput", SM, origInDims);
  return addValueMapping(outputs[0], glowNode);
}

Error PyTorchModelLoader::loadPermute(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 2, outputs, 1));

  glow::NodeValue in;
  ASSIGN_VALUE_OR_RETURN_ERR(in, getGlowNodeValueForValue(inputs[0]));
  size_t inDims = in.dims().size();

  std::vector<int64_t> *shuffle;
  ASSIGN_VALUE_OR_RETURN_ERR(shuffle,
                             iValToIntList(getGlowIValueForValue(inputs[1])));

  for (const int64_t dim : *shuffle) {
    RETURN_ERR_IF_NOT(dim >= 0,
                      "Negative shuffle dimensions not supported by Glow yet.");
    RETURN_ERR_IF_NOT(
        dim < inDims,
        "All shuffle dimensions must be less than the rank of the input.");
  }

  RETURN_ERR_IF_NOT(shuffle->size() == inDims,
                    "Shuffle for permute must has the same number of "
                    "dimensions as the input tensor.");

  auto output = F_.createTranspose("reshapeInput", in,
                                   castVector<glow::unsigned_t>(*shuffle))
                    ->getResult();

  c10::ScalarType dtype;
  RETURN_IF_ERR(getCorrectTypeMapping(dtype, inputs[0]));
  return addValueMapping(outputs[0], output, dtype);
}

Error PyTorchModelLoader::loadTo(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 5, outputs, 1));

  // TODO: use ConvertTo
  glow::NodeValue in;
  ASSIGN_VALUE_OR_RETURN_ERR(in, getGlowNodeValueForValue(inputs[0]));

  return addValueMapping(outputs[0], in);
}

Error PyTorchModelLoader::loadFlatten(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 3, outputs, 1));

  glow::NodeValue in;
  ASSIGN_VALUE_OR_RETURN_ERR(
      in, getGlowNodeValueForValue(inputs[FlattenInputs::input]));

  int64_t startDimRaw;
  ASSIGN_VALUE_OR_RETURN_ERR(
      startDimRaw,
      iValToInt(getGlowIValueForValue(inputs[FlattenInputs::start_dim])));
  int64_t startDim;
  ASSIGN_VALUE_OR_RETURN_ERR(startDim,
                             getPositiveIndex(startDimRaw, in.dims().size()));

  int64_t endDimRaw;
  ASSIGN_VALUE_OR_RETURN_ERR(endDimRaw, iValToInt(getGlowIValueForValue(
                                            inputs[FlattenInputs::end_dim])));

  int64_t endDim;
  ASSIGN_VALUE_OR_RETURN_ERR(endDim,
                             getPositiveIndex(endDimRaw, in.dims().size()));

  auto xDim = glow::flattenCdr(in.dims(), startDim);
  auto *glowNode = F_.createReshape("flatten", in, {xDim.first, xDim.second});

  c10::ScalarType dtype;
  RETURN_IF_ERR(getCorrectTypeMapping(dtype, inputs[FlattenInputs::input]));
  return addValueMapping(outputs[0], glowNode->getResult(), dtype);
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

Error PyTorchModelLoader::loadConstantChunk(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 1, outputs, -1));

  int64_t chunks = ptNode->i(at::attr::chunks);
  int64_t dimRaw = ptNode->i(at::attr::dim);

  RETURN_ERR_IF_NOT(chunks > 0, "There needs to be at least one chunk!");
  RETURN_ERR_IF_NOT(chunks == outputs.size(),
                    "Chunks must be equal to outputs.size()!");

  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(input, getGlowNodeValueForValue(inputs[0]));

  int64_t dim;
  ASSIGN_VALUE_OR_RETURN_ERR(dim,
                             getPositiveIndex(dimRaw, input.dims().size()));

  size_t dimsSize = input.dims().size();
  std::vector<glow::dim_t> begins(dimsSize);
  std::vector<glow::dim_t> ends(dimsSize);
  for (int i = 0; i < dimsSize; ++i) {
    begins[i] = 0;
    ends[i] = input.dims()[i];
  }

  // We can do this because chunks == output size. Otherwise this may not be
  // correct.
  size_t cur = 0;
  size_t end = input.dims()[dim];
  size_t step = ((end - cur) + (chunks - 1)) / chunks;

  for (int i = 0; i < outputs.size(); ++i) {
    begins[dim] = cur;
    cur = cur + step > end ? end : cur + step;
    ends[dim] = cur;
    RETURN_IF_ERR(addValueMapping(
        outputs[i], F_.createSlice("FusedChunkOut", input, begins, ends)));
  }
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

Error PyTorchModelLoader::loadCustomOp(const torch::jit::Node *ptNode) {
  CustomPyTorchOpLoader *customLoader =
      getCustomPyTorchOpLoaderForSymbol(ptNode->kind());

  RETURN_ERR_IF_NOT(
      customLoader,
      strFormat("Expected a custom loader to be found for symbol: %s",
                ptNode->kind().toQualString()));

  return customLoader->loadNode(*this, ptNode);
}

Error PyTorchModelLoader::loadEmbeddingBag(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, -7, outputs, 4));

  glow::NodeValue weight;
  ASSIGN_VALUE_OR_RETURN_ERR(
      weight, getGlowNodeValueForValue(inputs[EmbeddingBagInputs::weight]));
  glow::NodeValue indices;
  ASSIGN_VALUE_OR_RETURN_ERR(
      indices, getGlowNodeValueForValue(inputs[EmbeddingBagInputs::indices]));
  glow::NodeValue offsets;
  ASSIGN_VALUE_OR_RETURN_ERR(
      offsets, getGlowNodeValueForValue(inputs[EmbeddingBagInputs::offsets]));

  // If no indices are provided, replace the op with a zero Constant.
  if (indices.dims()[0] == 0) {
    glow::Tensor t(ElemKind::FloatTy, {offsets.dims()[0], weight.dims()[1]});
    t.zero();
    glow::Constant *glowConstant =
        F_.getParent()->createConstant("EmptyEmbeddingBag", std::move(t));
    return addValueMapping(outputs[0], glowConstant->getOutput());
  }

  glow::NodeValue perSampleWeights = loadNodeValueOrCreateBroadcastedConstant(
      inputs[EmbeddingBagInputs::per_sample_weights], "EmbeddingBag.ones",
      glow::Type(weight.getElementType(), {indices.dims()[0]}), 1.0);

  bool scaleGradByFreq;
  ASSIGN_VALUE_OR_RETURN_ERR(
      scaleGradByFreq, iValToBool(getGlowIValueForValue(
                           inputs[EmbeddingBagInputs::scale_grad_by_freq])));

  RETURN_ERR_IF_NOT(scaleGradByFreq == false,
                    "Currently only support scale_grad_by_freq == 'false'");

  int mode;
  ASSIGN_VALUE_OR_RETURN_ERR(
      mode, iValToInt(getGlowIValueForValue(inputs[EmbeddingBagInputs::mode])));

  RETURN_ERR_IF_NOT(mode == 0, "Currently only support mode='sum'");

  bool sparse;
  ASSIGN_VALUE_OR_RETURN_ERR(sparse, iValToBool(getGlowIValueForValue(
                                         inputs[EmbeddingBagInputs::sparse])));

  auto *EB = F_.createEmbeddingBag("EmbeddingBag", weight, perSampleWeights,
                                   indices, offsets);

  return addValueMapping(outputs[0], EB->getResult());
}

Error PyTorchModelLoader::loadEmbeddingBagByteRowwiseOffsets(
    const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, -7, outputs, 1));

  glow::NodeValue weight;
  ASSIGN_VALUE_OR_RETURN_ERR(
      weight, getGlowNodeValueForValue(
                  inputs[EmbeddingBagByteRowwiseOffsetsInputs::weight]));
  glow::NodeValue indices;
  ASSIGN_VALUE_OR_RETURN_ERR(
      indices, getGlowNodeValueForValue(
                   inputs[EmbeddingBagByteRowwiseOffsetsInputs::indices]));
  glow::NodeValue offsets;
  ASSIGN_VALUE_OR_RETURN_ERR(
      offsets, getGlowNodeValueForValue(
                   inputs[EmbeddingBagByteRowwiseOffsetsInputs::offsets]));

  bool scaleGradByFreq;
  ASSIGN_VALUE_OR_RETURN_ERR(
      scaleGradByFreq,
      iValToBool(getGlowIValueForValue(
          inputs[EmbeddingBagByteRowwiseOffsetsInputs::scale_grad_by_freq])));

  RETURN_ERR_IF_NOT(scaleGradByFreq == false,
                    "Currently only support scale_grad_by_freq == 'false'");

  int mode;
  ASSIGN_VALUE_OR_RETURN_ERR(
      mode, iValToInt(getGlowIValueForValue(
                inputs[EmbeddingBagByteRowwiseOffsetsInputs::mode])));

  RETURN_ERR_IF_NOT(mode == 0, "Currently only support mode='sum'");

  bool sparse;
  ASSIGN_VALUE_OR_RETURN_ERR(
      sparse, iValToBool(getGlowIValueForValue(
                  inputs[EmbeddingBagByteRowwiseOffsetsInputs::sparse])));

  RETURN_ERR_IF_NOT(sparse == false, "Currently only support sparse='false'");

  // If no indices are provided, replace the op with a zero Constant.
  if (indices.dims()[0] == 0) {
    glow::Tensor t(ElemKind::FloatTy,
                   {offsets.dims()[0], weight.dims()[1] - 2 * sizeof(float)});
    t.zero();
    glow::Constant *glowConstant = F_.getParent()->createConstant(
        "EmptyEmbeddingBagByteRowwiseOffsets", std::move(t));
    return addValueMapping(outputs[0], glowConstant->getOutput());
  }

  glow::NodeValue perSampleWeights = loadNodeValueOrCreateBroadcastedConstant(
      inputs[EmbeddingBagByteRowwiseOffsetsInputs::per_sample_weights],
      "EmbeddingBagByteRowwiseOffsets.ones",
      glow::Type(ElemKind::FloatTy, {indices.dims()[0]}), 1.0);

  glow::Constant *weightConstant =
      llvm::dyn_cast<glow::Constant>(weight.getNode());

  RETURN_ERR_IF_NOT(weightConstant,
                    strFormat("Expected Weight to be a Constant but found: %s",
                              weight.getNode()->getKindName()));

  TypeRef fusedTy = F_.getParent()->uniqueType(ElemKind::UInt8FusedQTy,
                                               weight.dims(), 0.0, 0);

  weightConstant->setType(Storage::OutputIdx, fusedTy);
  weightConstant->setPayloadType(fusedTy);

  auto *EB = F_.createEmbeddingBagByteRowwiseOffsets(
      "EmbeddingBagByteRowwiseOffsets", weightConstant->getOutput(),
      perSampleWeights, indices, offsets);

  return addValueMapping(outputs[0], EB->getResult());
}

Error PyTorchModelLoader::loadAttributes(
    const torch::jit::Graph &graph,
    const at::ArrayRef<torch::jit::IValue> inputs) {

  // Map from the Value in the Graph of an ivalue::Object to the Object and a
  // string representing it's place in the module hierarchy.
  std::unordered_map<const torch::jit::Value *,
                     std::pair<const c10::ivalue::Object *, std::string>>
      objectTree;

  // Load graph inputs that are Objects.
  auto graphInputValues = graph.inputs();
  for (size_t i = 0; i < inputs.size(); ++i) {
    const auto &input = inputs[i];
    if (!input.isObject()) {
      continue;
    }

    const auto &object = input.toObjectRef();

    objectTree[graphInputValues[i]] =
        std::make_pair(&object, object.type()->str().c_str());
  }

  // Load prim::GetAttr nodes.
  for (const auto &node : graph.nodes()) {
    if (node->kind() != torch::jit::prim::GetAttr) {
      continue;
    }

    RETURN_IF_ERR(
        checkInputAndOutputSizes(node->inputs(), 1, node->outputs(), 1));

    const auto *inputValue = node->input();
    const auto *outputValue = node->output();

    RETURN_ERR_IF_NOT(objectTree.count(inputValue),
                      "Missing input for prim::getAttr");

    const auto &parent = objectTree.at(inputValue);
    const auto *parentObject = parent.first;

    const auto attrName = node->s(torch::jit::attr::name);
    const auto ival = parentObject->getAttr(attrName);

    // Concatenation of names of Objects and fields referenced in the Module
    // tree.
    const auto &nameHierarchy = parent.second;
    const auto newNameHierarchy =
        strFormat("%s_%s", nameHierarchy.c_str(), attrName.c_str());

    if (ival.isObject()) {
      objectTree[outputValue] =
          std::make_pair(&ival.toObjectRef(), newNameHierarchy);
      continue;
    } else if (ival.isTensor()) {
      GlowIValue glowIVal;
      // PyTorch Tensor extracted type is kByte
      // indicate it is the address of stored weights of quantized
      // linear or conv.
      if (isPackedQParamNode(node)) {
        const auto ptTensor = ival.toTensor();
        CHECK(ptTensor.is_contiguous());
        glowIVal.fromPTTensor(ptTensor);
        RETURN_IF_ERR(addValueMapping(outputValue, std::move(glowIVal)));
      } else {
        RETURN_IF_ERR(glowIVal.fromIValue(ival));
        glow::Tensor *t;
        ASSIGN_VALUE_OR_RETURN_ERR(t, glowIVal.toTensor());
        glow::Constant *glowConstant =
            F_.getParent()->createConstant(newNameHierarchy, std::move(*t));

        if (copyTensorMemory_) {
          glowConstant->ensureIsOwned();
        }
        RETURN_IF_ERR(addValueMapping(outputValue, glowConstant->getOutput()));
      }
    } else {
      GlowIValue glowIVal;
      RETURN_IF_ERR(glowIVal.fromIValue(ival));
      RETURN_IF_ERR(addValueMapping(outputValue, std::move(glowIVal)));
    }
  }

  return Error::success();
}

/*static*/
Error PyTorchModelLoader::loadJITGraph(
    glow::Function &F, const torch::jit::Graph &graph,
    std::vector<glow::Placeholder *> &inputPlaceholders,
    std::vector<glow::Placeholder *> &outputPlaceholders,
    std::vector<c10::ScalarType> &outputCorrectType,
    const PyTorchLoaderSettings &settings,
    const at::ArrayRef<torch::jit::IValue> inputs,
    const std::vector<InputMeta> &inputMeta) {
  Error error = Error::empty();
  PyTorchModelLoader loader(F, graph, inputPlaceholders, outputPlaceholders,
                            outputCorrectType, error, settings, inputs,
                            inputMeta);
  return error;
}

PyTorchModelLoader::PyTorchModelLoader(
    glow::Function &F, const torch::jit::Graph &graph,
    std::vector<glow::Placeholder *> &inputPlaceholders,
    std::vector<glow::Placeholder *> &outputPlaceholders,
    std::vector<c10::ScalarType> &outputCorrectType, Error &error,
    const PyTorchLoaderSettings &settings,
    const at::ArrayRef<torch::jit::IValue> inputs,
    const std::vector<InputMeta> &inputMeta)
    : F_(F), inputs_(inputs), copyTensorMemory_(false) {
  auto loadFn = [&]() -> Error {
    auto graphInputValues = graph.inputs();

    RETURN_ERR_IF_NOT(
        inputs.size() == graphInputValues.size() ||
            inputMeta.size() == graphInputValues.size(),
        glow::strFormat("Number of Graph inputs %lu must match the "
                        "number of provided inputs %lu.",
                        graphInputValues.size(), inputs.size()));
    // Create Glow Placeholders for inputs.
    for (size_t i = 0; i < graphInputValues.size(); ++i) {
      const torch::jit::Value *inputValue = graphInputValues[i];
      c10::ScalarType inputScalarType;
      glow::Placeholder *ph;
      if (!inputMeta.empty()) {
        if (inputValue->type()->kind() == c10::TypeKind::TensorType) {
          inputScalarType = inputMeta[i].type;
          glow::ElemKind elemKind;
          if (inputMeta[i].type != at::kQUInt8) {
            elemKind = scalarTypeToElemKind(inputMeta[i].type);
          } else {
            elemKind = ElemKind::Int8QTy;
          }
          glow::Type t(elemKind, inputMeta[i].dims);

          ph = F_.getParent()->createPlaceholder(&t, "input",
                                                 /*isTrainable*/ false);

        } else {
          // Here we assume it's scalar type
          glow::Type t(typeKindToElemKind(inputValue->type()->kind()), {});
          ph = F_.getParent()->createPlaceholder(&t, "input", false);
          inputScalarType = elemKindToScalarType(t.getElementType());
        }
        RETURN_IF_ERR(
            addValueMapping(inputValue, ph->getOutput(), inputScalarType));
        inputPlaceholders.push_back(ph);
        inputPlaceholdersReverseIndex_[ph] = i;
      } else {
        const c10::IValue inputIValue = inputs.at(i);
        // Objects will be used to load model parameters.
        if (inputIValue.isObject()) {
          continue;
        }
        GlowIValue glowIVal;
        RETURN_IF_ERR(glowIVal.fromIValue(inputIValue));
        if (glowIVal.isTensor()) {
          glow::Tensor *t;
          ASSIGN_VALUE_OR_RETURN_ERR(t, glowIVal.toTensor());
          auto oldType = t->getType();
          if (oldType.getElementType() == ElemKind::UInt8QTy) {
            auto newType = glow::Type(ElemKind::Int8QTy, oldType.dims(),
                                      oldType.getScale(),
                                      oldType.getOffset() - OFFSETSHIFT);
            ph = F_.getParent()->createPlaceholder(&newType, "input",
                                                   /*isTrainable*/ false);
          } else {
            ph = F_.getParent()->createPlaceholder(&t->getType(), "input",
                                                   /*isTrainable*/ false);
          }
          inputScalarType = inputIValue.toTensor().scalar_type();
          RETURN_IF_ERR(
              addValueMapping(inputValue, ph->getOutput(), inputScalarType));
          inputPlaceholders.push_back(ph);
          inputPlaceholdersReverseIndex_[ph] = i;
        } else {

          RETURN_IF_ERR(addValueMapping(inputValue, std::move(glowIVal)));
        }
      }
    }

    RETURN_IF_ERR(loadAttributes(graph, inputs));
    RETURN_IF_ERR(loadNodes(graph));

    // Create Glow Placeholders for outputs.
    for (const torch::jit::Value *output : graph.outputs()) {
      glow::NodeValue outputNodeValue;
      // Only allow tensor outputs from Glow subgraph.
      ASSIGN_VALUE_OR_RETURN_ERR(outputNodeValue,
                                 getGlowNodeValueForValue(output));
      auto *save = F_.createSave("save", outputNodeValue);
      outputPlaceholders.push_back(save->getPlaceholder());

      c10::ScalarType outputScalarType;
      RETURN_IF_ERR(getCorrectTypeMapping(outputScalarType, output));
      outputCorrectType.push_back(outputScalarType);
    }

    if (settings.dumpGlowDag) {
      F_.dumpDAG(strFormat("%s.dot", F_.getName().data()));
    }

    if (settings.writeToOnnx) {
      RETURN_IF_ERR(dumpOnnxModel(F, /* zipMode */ false));
    }

    return Error::success();
  };
  error = loadFn();

  if (error) {
    DLOG(ERROR) << "Encountered error while loading graph:" << std::endl
                << graph << std::endl;
  }
}

/*static*/
Error PyTorchModelLoader::loadJITGraphForOnnxTraining(
    glow::Function &F, const torch::jit::Graph &graph,
    const at::ArrayRef<torch::jit::IValue> inputs,
    const std::vector<at::Tensor> &parameters,
    std::vector<glow::Placeholder *> &inputPlaceholders,
    std::vector<glow::Placeholder *> &outputPlaceholders) {
  Error error = Error::empty();
  PyTorchModelLoader loader(F, graph, parameters, inputPlaceholders,
                            outputPlaceholders, error, inputs);
  return error;
}

PyTorchModelLoader::PyTorchModelLoader(
    glow::Function &F, const torch::jit::Graph &graph,
    const std::vector<at::Tensor> &parameters,
    std::vector<glow::Placeholder *> &inputPlaceholders,
    std::vector<glow::Placeholder *> &outputPlaceholders, Error &error,
    const at::ArrayRef<torch::jit::IValue> inputs)
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
      glow::Constant *C = F_.getParent()->createConstant(
          "parameter", ptTensorToGlowTensor(parameters[i]));
      C->ensureIsOwned();

      RETURN_IF_ERR(
          addValueMapping(graphInputValues[graphIdx], C->getOutput()));
    }

    RETURN_IF_ERR(loadNodes(graph));

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

c10::ScalarType ValueMapping::getCorrectType() const { return correctType_; }

void ValueMapping::setCorrectType(c10::ScalarType dtype) {
  correctType_ = dtype;
}

ValueMapping::ValueMapping(NodeValue nodeValue) {
  mappingType_ = ValueMappingType::NodeValue;
  nodeValue_ = std::move(nodeValue);
}

ValueMapping::ValueMapping(GlowIValue glowIValue) {
  mappingType_ = ValueMappingType::IValue;
  glowIValue_ = glow::make_unique<GlowIValue>(std::move(glowIValue));
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
