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
#include <ATen/native/quantized/cpu/conv_packed_params.h>
#include <ATen/native/quantized/cpu/packed_params.h>
#include <torch/csrc/jit/ir/ir.h>
#include <unordered_map>

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
    RETURN_ERR(expectedVal.takeError());
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
    return MAKE_ERR(
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
    RETURN_ERR(expectedGlowIVal.takeError());
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
    return MAKE_ERR(
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
    RETURN_ERR(expectedGlowIVal.takeError());
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
    RETURN_ERR(originalExpected.takeError());
  }
}

std::vector<glow::unsigned_t>
castToGlowIntList(const torch::List<int64_t> &int_list) {
  std::vector<glow::unsigned_t> out;
  for (const auto &elem : int_list) {
    out.push_back(static_cast<glow::unsigned_t>(elem));
  }
  return out;
}

/// Unwrap a Expected<OriginalT> \p originalExpected and calls
/// static_cast() with the contents, propagates any Errors.
template <typename T, typename OriginalT>
Expected<T> static_cast_expected(Expected<OriginalT> originalExpected) {
  if (originalExpected) {
    return static_cast<T>(*originalExpected);
  } else {
    RETURN_ERR(originalExpected.takeError());
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

/// \returns true if output of \p node is used only by a quantized node.
bool isQParamWeightNode(const torch::jit::Node *node) {
  static std::unordered_set<torch::jit::Symbol> packedQuantNodeKinds = {
      torch::jit::Symbol::fromQualString("quantized::linear"),
      torch::jit::Symbol::fromQualString("quantized::conv2d"),
      torch::jit::Symbol::fromQualString("quantized::conv2d_relu"),
      torch::jit::Symbol::fromQualString("quantized::conv3d"),
      torch::jit::Symbol::fromQualString("quantized::conv3d_relu"),
      torch::jit::Symbol::fromQualString("glow::unpacked_quantized_linear"),
      torch::jit::Symbol::fromQualString("glow::unpacked_quantized_conv2d"),
      torch::jit::Symbol::fromQualString(
          "glow::unpacked_quantized_conv2d_relu"),
      torch::jit::Symbol::fromQualString("glow::unpacked_quantized_conv3d"),
      torch::jit::Symbol::fromQualString(
          "glow::unpacked_quantized_conv3d_relu"),
  };

  const auto uses = node->output()->uses();
  if (uses.empty()) {
    return false;
  }

  const auto userKind = uses[0].user->kind();

  if (packedQuantNodeKinds.count(userKind)) {
    return true;
  }

  return false;
}

/// \returns string representation of JIT node \p node.
std::string jitNodeToString(const torch::jit::Node *node) {
  std::stringstream ss;
  ss << *node;
  return ss.str();
}

/// Writes the given Function \p F to file using ONNXModelWriter. If \p zipMode
/// is set then zipMode will be used for the writer. \returns an Error if one
/// occurred.
Error dumpOnnxModel(glow::Function &F, bool zipMode,
                    const std::string &fileName, bool writeOnnxToTmp) {
  constexpr size_t kIrVer = 7, kOpsetVer = 9;
  std::string filepath;
  if (writeOnnxToTmp) {
    ASSIGN_VALUE_OR_RETURN_ERR(
        filepath,
        getTempFileLoc(fileName, (zipMode ? ".onnx.zip" : ".onnxtxt")));
  } else {
    filepath = fileName + (zipMode ? ".onnx.zip" : ".onnxtxt");
  }

  LOG(INFO) << "Writing ONNX model to " << filepath;
  Error err = Error::empty();
  ONNXModelWriter onnxWriter(filepath, F, kIrVer, kOpsetVer, &err,
                             /* textMode */ !zipMode, /* zipMode */ zipMode,
                             /* useGlowCustomOps */ true);
  return err;
}

/// Indexes of aten::zeros inputs.
struct ZerosInputs {
  enum {
    size = 0,
    dtype = 1,
    layout = 2,
    device = 3,
    pin_memory = 4,
  };
};

/// Indexes of aten::lstm inputs.
struct LSTMInputs {
  enum {
    input = 0,
    hx = 1,
    params = 2,
    has_biases = 3,
    num_layers = 4,
    dropout = 5,
    train = 6,
    bidirectional = 7,
    batch_first = 8,
  };
};

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
    cudnn_enabled = 11,
  };
};

/// Indexes of aten::select inputs.
struct SelectInputs {
  enum {
    input = 0,
    dim = 1,
    index = 2,
  };
};

struct Conv2DInputs {
  enum {
    input = 0, // NCHW
    weights = 1,
    bias = 2,
    stride = 3,
    padding = 4,
    dilation = 5,
    groups = 6,
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

/// Indices of quantized::batch_norm2d and quantized::batch_norm3d inputs.
struct QuantizedBatchNormInputs {
  enum {
    input = 0, // NCHW
    weights = 1,
    bias = 2,
    running_mean = 3,
    running_var = 4,
    eps = 5,
    output_scale = 6,
    output_zero_point = 7,
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

/// Indexes of glow::unpacked_quantized_conv2d/conv3d inputs.
struct QuantizedUnpackedConvInputs {
  enum {
    input = 0, // NCHW/NCTHW
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
    scale = 2,
    zero_point = 3,
  };
};

/// Indexes of quantized::conv3d and quantized::conv3d_relu inputs.
struct QuantizedConv3dInputs {
  enum {
    input = 0, // NCTHW
    packed_weights = 1,
    scale = 2,
    zero_point = 3,
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

/// Indexes of quantized::mul_scalar inputs.
/// Also used for quantized::mul when rhs is scalar.
struct QuantizedMulScalarInputs {
  enum {
    lhs = 0,
    rhs = 1,
  };
};

/// Indexes of quantized::mul inputs.
struct QuantizedMulInputs {
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

/// Indexes of aten::squeeze inputs.
struct SqueezeInputs {
  enum {
    input = 0,
    dim = 1,
  };
};

/// Indexes of aten::unsqueeze inputs.
struct UnsqueezeInputs {
  enum {
    input = 0,
    dim = 1,
  };
};

/// Indexes of aten::masked_fill inputs.
struct MaskedFillInputs {
  enum {
    input = 0,
    mask = 1,
    value = 2,
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

/// Indexes of aten::to.dtype_layout inputs.
struct ToDtypeLayoutInputs {
  enum {
    input = 0,
    dtype,
    layout,
    device,
    pin_memory,
    non_blocking,
    copy,
    memory_format,
  };
};

/// Indexes of aten::to.device inputs.
struct ToDeviceInputs {
  enum {
    input = 0,
    device,
    dtype,
    non_blocking,
    copy,
    memory_format,
  };
};

/// Indexes of aten::to.other inputs.
struct ToOtherInputs {
  enum {
    input = 0,
    other,
    non_blocking,
    copy,
    memory_format,
  };
};

/// Indexes of aten::to.prim_other inputs.
struct ToPrimOtherInputs {
  enum {
    input = 0,
    non_blocking,
    copy,
  };
};

/// Indexes of aten::to.prim_other inputs.
struct ToPrimDtypeInputs {
  enum {
    input = 0,
    dtype,
    non_blocking,
    copy,
  };
};

/// Indexes of aten::to.prim_other inputs.
struct ToPrimDeviceInputs {
  enum {
    input = 0,
    device,
    dtype,
    non_blocking,
    copy,
  };
};

/// Indexes of aten::to.dtype inputs.
struct ToDtypeInputs {
  enum {
    input = 0,
    dtype,
    non_blocking,
    copy,
    memory_format,
  };
};

/// Indexes of aten::embedding inputs.
struct EmbeddingInputs {
  enum {
    weights = 0,
    indices,
    padIdx,
    scale,
    sparse,
  };
};

/// Indexes of aten::embedding_bag inputs.
struct EmbeddingBagInputs {
  enum {
    weight = 0,
    indices,
    offsets,
    scale_grad_by_freq,
    mode,
    sparse,
    per_sample_weights,
    include_last_offset,
  };
};

/// Indexes of aten::index_select inputs.
struct IndexSelectInputs {
  enum {
    input = 0,
    dimension,
    index,
  };
};

/// Indexes of aten::clamp_min inputs.
struct ClampMinInputs {
  enum {
    input = 0,
    min,
  };
};

/// Indexes of aten::expand_as inputs.
struct ExpandAsInputs {
  enum {
    input = 0,
    other,
  };
};

/// Indexes of fb::glow_embedding_bag inputs.
struct GlowEmbeddingBagInputs {
  enum {
    indices = 0,
    offsets,
    weight_qualname,
    num_embeddings,
    embedding_dim,
    per_sample_weights,
    include_last_offset,
  };
};

/// Indexes of fb::lengths_range inputs.
struct LengthsRangeInputs {
  enum {
    lengths = 0,
    shapes,
  };
};

/// Indexes used for quantized::embedding_bag_byte_rowwise_offsets inputs.
struct EmbeddingBagByteRowwiseOffsetsInputs {
  enum {
    weight = 0,
    indices,
    offsets,
    scale_grad_by_freq,
    mode,
    sparse,
    per_sample_weights,
    compressed_indices_mapping,
    include_last_offset,
  };
};

/// Indexes used for quantized::embedding_bag_4bit_rowwise_offsets inputs.
struct EmbeddingBag4BitRowwiseOffsetsInputs {
  enum {
    weight = 0,
    indices,
    offsets,
    scale_grad_by_freq,
    mode,
    sparse,
    per_sample_weights,
    compressed_indices_mapping,
    include_last_offset,
  };
};

/// Indexes used for glow::fused_split inputs.
struct FusedSplitInputs {
  enum {
    input = 0,
    num_split,
    dim,
  };
};

/// Indexes used for fb::fast_gather inputs.
struct FastGatherInputs {
  enum {
    input = 0,
    indices,
  };
};

/// Indexes used for _caffe2::RoIAlign inputs
struct RoiAlignInputs {
  enum {
    features = 0,
    rois,
    layout,
    spatialScale,
    outputHeight,
    outputWidth,
    samplingRatio,
    aligned,
  };
};

/// Indexes used for _caffe2::BBoxTransform inputs
struct BBoxTransformInputs {
  enum {
    rois = 0,
    deltas,
    imInfo,
    weights,
    applyScale,
    rotated,
    angleBoundOn,
    angleBoundLo,
    angleBoundHi,
    clipAngleThresh,
    legacyPlusOne,
  };
};

/// Indexes used for _caffe2::BBoxTransform inputs
struct CompareInputs {
  enum {
    lhs = 0,
    rhs,
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
      {{"aten::detach"}, &PyTorchModelLoader::loadDetach},
      {{"prim::Constant"}, &PyTorchModelLoader::loadConstant},
      {{"prim::NumToTensor"}, &PyTorchModelLoader::loadNumToTensor},
      {{"aten::_shape_as_tensor"}, &PyTorchModelLoader::loadShapeAsTensor},
      {{"aten::Int"}, &PyTorchModelLoader::loadInt},
      {{"aten::arange"}, &PyTorchModelLoader::loadArange},
      {{"aten::zeros"}, &PyTorchModelLoader::loadZeros},
      {{"aten::mul", "aten::mul_"}, &PyTorchModelLoader::loadMul},
      {{"aten::div", "aten::div_"}, &PyTorchModelLoader::loadDiv},
      {{"aten::floor_divide", "aten::floor_divide_"},
       &PyTorchModelLoader::loadFloorDiv},
      {{"aten::add", "aten::add_"}, &PyTorchModelLoader::loadAdd},
      {{"aten::sub", "aten::sub_"}, &PyTorchModelLoader::loadSub},
      {{"aten::rsub"}, &PyTorchModelLoader::loadRsub},
      {{"aten::log"}, &PyTorchModelLoader::loadLog},
      {{"aten::sum"}, &PyTorchModelLoader::loadSum},
      {{"aten::sigmoid", "aten::sigmoid_"}, &PyTorchModelLoader::loadSigmoid},
      {{"aten::silu"}, &PyTorchModelLoader::loadSilu},
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
      {{"glow::fused_broadcast_cat"},
       &PyTorchModelLoader::loadFusedBroadcastConcat},
      {{"glow::fused_broadcast_stack"},
       &PyTorchModelLoader::loadFusedBroadcastStack},
      {{"aten::floor"}, &PyTorchModelLoader::loadFloor},
      {{"aten::ceil"}, &PyTorchModelLoader::loadCeil},
      {{"aten::mean"}, &PyTorchModelLoader::loadMean},
      {{"aten::pow"}, &PyTorchModelLoader::loadPow},
      {{"aten::logical_xor"}, &PyTorchModelLoader::loadLogicalXor},
      {{"aten::logical_or"}, &PyTorchModelLoader::loadLogicalOr},
      {{"aten::logical_and", "aten::__iand__"},
       &PyTorchModelLoader::loadLogicalAnd},
      {{"aten::logical_not"}, &PyTorchModelLoader::loadLogicalNot},
      {{"aten::dropout", "aten::dropout_"}, &PyTorchModelLoader::loadDropout},
      {{"aten::sqrt", "aten::sqrt_"}, &PyTorchModelLoader::loadSqrt},
      {{"aten::le", "aten::le_"}, &PyTorchModelLoader::loadCmp<CmpLTENode>},
      {{"aten::lt", "aten::lt_"}, &PyTorchModelLoader::loadCmp<CmpLTNode>},
      {{"aten::ne", "aten::ne_"}, &PyTorchModelLoader::loadCmp<CmpNEQNode>},
      {{"aten::eq", "aten::eq_"}, &PyTorchModelLoader::loadCmp<CmpEQNode>},
      {{"aten::ge", "aten::ge_"},
       &PyTorchModelLoader::loadCmp<CmpLTENode, true>},
      {{"aten::gt", "aten::gt_"},
       &PyTorchModelLoader::loadCmp<CmpLTNode, true>},
      {{"aten::clamp"}, &PyTorchModelLoader::loadClamp},
      {{"aten::cos"}, &PyTorchModelLoader::loadCos},
      {{"aten::sin"}, &PyTorchModelLoader::loadSin},
      {{"aten::acos"}, &PyTorchModelLoader::loadAcos},
      {{"aten::asin"}, &PyTorchModelLoader::loadAsin},
      {{"aten::atan"}, &PyTorchModelLoader::loadAtan},
      {{"quantized::add"}, &PyTorchModelLoader::loadQuantizedAdd},
      {{"quantized::add_relu"}, &PyTorchModelLoader::loadQuantizedAddRelu},
      {{"quantized::mul"}, &PyTorchModelLoader::loadQuantizedMul},
      {{"quantized::mul_scalar"}, &PyTorchModelLoader::loadQuantizedMul},
      {{"glow::fused_linear"}, &PyTorchModelLoader::loadGlowFusedLinear},
      {{"glow::unpacked_quantized_conv2d"},
       &PyTorchModelLoader::loadQuantizedConvUnpacked},
      {{"glow::unpacked_quantized_conv3d"},
       &PyTorchModelLoader::loadQuantizedConvUnpacked},
      {{"glow::unpacked_quantized_conv3d_relu"},
       &PyTorchModelLoader::loadQuantizedConvReluUnpacked},
      {{"glow::unpacked_quantized_conv2d_relu"},
       &PyTorchModelLoader::loadQuantizedConvReluUnpacked},
      {{"glow::unpacked_quantized_linear"},
       &PyTorchModelLoader::loadQuantizedLinearUnpacked},
      {{"glow::fused_split"}, &PyTorchModelLoader::loadFusedSplit},
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
      {{"aten::abs"}, &PyTorchModelLoader::loadAbs},
      {{"aten::upsample_nearest3d"}, &PyTorchModelLoader::loadUpsampleNearest},
      {{"aten::upsample_nearest2d"}, &PyTorchModelLoader::loadUpsampleNearest},
      {{"aten::view"}, &PyTorchModelLoader::loadView},
      {{"aten::_convolution"}, &PyTorchModelLoader::loadConvolution},
      {{"aten::lstm"}, &PyTorchModelLoader::loadLSTM},
      {{"aten::conv2d"}, &PyTorchModelLoader::loadConv2D},
      {{"aten::batch_norm"}, &PyTorchModelLoader::loadBatchNorm},
      {{"aten::norm", "aten::frobenius_norm"}, &PyTorchModelLoader::loadNorm},
      {{"quantized::batch_norm2d"},
       &PyTorchModelLoader::loadQuantizedBatchNorm2d},
      {{"quantized::batch_norm3d"},
       &PyTorchModelLoader::loadQuantizedBatchNorm3d},
      {{"quantized::batch_norm3d_relu"},
       &PyTorchModelLoader::loadQuantizedBatchNorm3dRelu},
      {{"aten::layer_norm"}, &PyTorchModelLoader::loadLayerNorm},
      {{"aten::max_pool2d", "aten::max_pool2d_with_indices"},
       &PyTorchModelLoader::loadMaxPool2d},
      {{"aten::avg_pool2d"}, &PyTorchModelLoader::loadAvgPool2d},
      {{"aten::avg_pool3d"}, &PyTorchModelLoader::loadAvgPool3d},
      {{"aten::matmul"}, &PyTorchModelLoader::loadMatMul},
      {{"aten::mm"}, &PyTorchModelLoader::loadMM},
      {{"aten::bmm"}, &PyTorchModelLoader::loadBmm},
      {{"aten::addmm"}, &PyTorchModelLoader::loadAddMM},
      {{"aten::select"}, &PyTorchModelLoader::loadSelect},
      {{"aten::flatten"}, &PyTorchModelLoader::loadFlatten},
      {{"aten::squeeze", "aten::squeeze_"}, &PyTorchModelLoader::loadSqueeze},
      {{"aten::unsqueeze", "aten::unsqueeze_"},
       &PyTorchModelLoader::loadUnsqueeze},
      {{"aten::masked_fill", "aten::masked_fill_"},
       &PyTorchModelLoader::loadMaskedFill},
      {{"aten::prelu"}, &PyTorchModelLoader::loadPRelu},
      {{"aten::slice"}, &PyTorchModelLoader::loadSlice},
      {{"aten::softmax"}, &PyTorchModelLoader::loadSoftMax},
      {{"aten::topk"}, &PyTorchModelLoader::loadTopK},
      {{"aten::to"}, &PyTorchModelLoader::loadTo},
      {{"prim::ConstantChunk"}, &PyTorchModelLoader::loadConstantChunk},
      {{"aten::embedding"}, &PyTorchModelLoader::loadEmbedding},
      {{"aten::embedding_bag"}, &PyTorchModelLoader::loadEmbeddingBag},
      {{"fb::simple_embedding_bag_sum"}, &PyTorchModelLoader::loadEmbeddingBag},
      {{"fb::glow_embedding_bag"}, &PyTorchModelLoader::loadGlowEmbeddingBag},
      {{"fb::glow_embedding_bag_byte_rowwise_offsets"},
       &PyTorchModelLoader::loadGlowEmbeddingBagByteRowwiseOffsets},
      {{"fb::glow_embedding_bag_4bit_rowwise_offsets"},
       &PyTorchModelLoader::loadGlowEmbeddingBag4bitRowwiseOffsets},
      // Disabled for now since this node needs extra information
      //{{"fb::lengths_range"}, &PyTorchModelLoader::loadLengthsRange},
      {{"_caffe2::BatchPermutation"},
       &PyTorchModelLoader::loadBatchPermutation},
      {{"quantized::embedding_bag_byte_rowwise_offsets"},
       &PyTorchModelLoader::loadEmbeddingBagByteRowwiseOffsets},
      {{"quantized::embedding_bag_4bit_rowwise_offsets"},
       &PyTorchModelLoader::loadEmbeddingBag4BitRowwiseOffsets},
      {{"fb::embedding_bag_byte_rowwise_offsets"},
       &PyTorchModelLoader::loadEmbeddingBagByteRowwiseOffsets},
      {{"fb::embedding_bag_4bit_rowwise_offsets"},
       &PyTorchModelLoader::loadEmbeddingBag4BitRowwiseOffsets},
      {{"fb::fast_gather"}, &PyTorchModelLoader::loadFastGather},
      {{"_caffe2::RoIAlign"}, &PyTorchModelLoader::loadRoiAlign},
      {{"_caffe2::RoIAlignRotated"}, &PyTorchModelLoader::loadRoiAlignRotated},
      {{"_caffe2::BBoxTransform"}, &PyTorchModelLoader::loadBBoxTransform},
      {{"aten::index_select"}, &PyTorchModelLoader::loadIndexSelect},
      {{"aten::clamp_min"}, &PyTorchModelLoader::loadClampMin},
      {{"aten::expand_as"}, &PyTorchModelLoader::loadExpandAs},
      {{"glow::nnckernel"}, &PyTorchModelLoader::loadNNCKernel},
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

// Check if node only has const inputs, which indicate output should be
// propagated from input when compiling.
static bool isConstNode(const glow::Node &glowNode,
                        const torch::jit::Node *const node) {
  // It is constant node already, dont need to propagate
  // And we dont want to do constant propagation for quantize node
  if (glowNode.getKind() == Kinded::Kind::ConstantKind ||
      glowNode.getKind() == Kinded::Kind::QuantizeNodeKind) {
    return false;
  }
  if (node->kind() == torch::jit::aten::matmul) {
    return false;
  }
  unsigned int n = glowNode.getNumInputs();
  for (int i = 0; i < n; i++) {
    auto ithInputNodeValue = glowNode.getNthInput(i);
    // Cannot propagate if not all inputs are constant
    if (ithInputNodeValue.getNode()->getKind() != Kinded::Kind::ConstantKind) {
      return false;
    }
  }
  // If all input nodes are constant, then this glowNode should propagate
  // its input to output and create a constant node to replace
  return true;
}

// This function creates and runs a graph which contains one node \p glowNode,
// and remaps its result as a constant back to original graph. If \p glowNode 's
// output is directly mapped to a jit node, then we modify the jit value
// mapping; If it is mapped to another glow nodevalue, usually when one jit node
// creates more than one glow nodes, then we find all other places that using
// this output, and replace it with our newly created constant. This process
// strictly relies on the topological order of the node in nodelist, therefore
// please do not change it during PyTorch model loading.
// Also, currently we dont support one jit node map to multiple glow node and
// having multiple outputs. The only scenario of this to be happening is
// prim::ConstantChunk, which we would like to skip running this.
Error PyTorchModelLoader::runAndRemapSingleNode(
    glow::Node &glowNode, const torch::jit::Node *const node,
    llvm::simple_ilist<glow::Node>::iterator nodeBeginPtr) {

  // Do not do constant propagation if jit node is prim:ConstantChunk
  // TODO reverse map of jit-glow node should resolve this problem.
  if (node->kind() == torch::jit::prim::ConstantChunk) {
    return Error::success();
  }

  std::vector<glow::Tensor *> outputTensors;
  PlaceholderBindings bindings;
  auto &nodelist = F_.getNodes();

  // Create a tmp Function to run the single node
  glow::Function *tmpF =
      F_.getParent()->createFunction("eval_const_propagating");
  unsigned int numResults = glowNode.getNumResults();
  auto tmpGlowNode = glowNode.clone();
  tmpF->addNode(tmpGlowNode);

  // Create output placeholders and tensors
  for (int i = 0; i < numResults; i++) {
    glow::NodeValue outputNodeValue = tmpGlowNode->getNthResult(i);
    auto *save = tmpF->createSave("save", outputNodeValue);
    auto *result = bindings.allocate(save->getPlaceholder());
    outputTensors.push_back(result);
  }

  // Evaluate the constant outputs using interpreter backend.
  std::unique_ptr<Backend> backend(createBackend("Interpreter"));
  CompilationContext cctx;
  cctx.compMode = CompilationMode::Infer;
  cctx.optimizationOpts.enableConstantFolding = false;
  cctx.backendOpts.collectConstants = true;
  cctx.verboseCompile = false;
  // Avoid constant folding on the subgraph. Constatnt deduplication still
  // happens when the full graph is being run. This helps speed up Glow
  // compilation.
  cctx.optimizationOpts.enableConstantDeduplication = false;
  RETURN_IF_ERR(executeConstantFunction(*backend, *tmpF, bindings, cctx, true));

  // Remap result back to original jit graph
  for (int i = 0; i < numResults; i++) {
    auto t = outputTensors[i];
    auto outputNodeValue = glowNode.getNthResult(i);
    auto nodePtr = nodeBeginPtr;

    auto glowConstant =
        tmpF->getParent()->createConstant("eval_graph_output", std::move(*t));

    glowConstant->ensureIsOwned();
    // Remap back to glow graph
    while (nodePtr != nodelist.end()) {
      glow::Node &checkNode = *nodePtr;
      for (int j = 0; j < checkNode.getNumInputs(); j++) {
        auto jthInputNodeValue = checkNode.getNthInput(j);
        if (jthInputNodeValue == outputNodeValue) {
          checkNode.setNthInput(j, glowConstant->getOutput());
        }
      }
      nodePtr++;
    }
    if (valueMapReverse_.count(outputNodeValue)) {
      auto *value = valueMapReverse_[outputNodeValue];
      RETURN_IF_ERR(removeValueMapping(value));
      auto scalarType =
          elemKindToScalarType(glowConstant->getType()->getElementType());
      RETURN_IF_ERR(
          addValueMapping(value, glowConstant->getOutput(), scalarType));
    }
  }

  // Remove tmp stuffs and return success
  F_.getParent()->eraseFunction(tmpF);
  return Error::success();
}

Error PyTorchModelLoader::loadNodes(const torch::jit::Graph &graph) {
  const auto &mapping = getSymbolsMapping();
  auto &nodelist = F_.getNodes();
  int nodeIdx = 0;

  // Lambda for loading a single node, used so any Error can be captured and
  // the string representation of the node appended to the Error stack.
  auto loadNode = [&](const torch::jit::Node *node) -> Error {
    const auto kind = node->kind();
    // prim::GetAttr is loaded separately.
    if (kind == torch::jit::prim::GetAttr) {
      return Error::success();
    }

    auto it = mapping.find(kind);

    RETURN_ERR_IF_NOT(it != mapping.end(),
                      glow::strFormat("Node kind %s is not supported by Glow",
                                      kind.toDisplayString()));

    RETURN_IF_ERR((this->*it->second.loadFn)(node));

    if (settings_.debugContinuouslyVerifyDuringModelLoading) {
      if (!F_.verify()) {
        F_.dumpDAG("failed.dot");
        return MAKE_ERR("Failed Function verification after loading a node");
      }
    }

    auto nodeItr = nodelist.begin();
    for (int j = 0; j < nodeIdx; j++) {
      nodeItr++;
    }
    // TODO we visited many redundent nodes during this process,
    // which makes while constant propagation to be a O(N^2) algorithm.
    // We should be able to improve this by improving nodelist structure.
    while (nodeItr != nodelist.end()) {
      glow::Node &glowNode = *nodeItr;
      if (isConstNode(glowNode, node)) {
        // Run glowNode and remap it result as a constant node as node's output.
        RETURN_IF_ERR(runAndRemapSingleNode(glowNode, node, nodeItr));
      }
      nodeIdx++;
      nodeItr++;
    }

    if (settings_.debugContinuouslyVerifyDuringModelLoading) {
      if (!F_.verify()) {
        F_.dumpDAG("failed.dot");
        return MAKE_ERR("Failed Function verification after constant "
                        "propagation while loading a node");
      }
    }

    return Error::success();
  };

  // Nodes are topologically sorted.
  for (const auto *node : graph.nodes()) {
    if (auto err = loadNode(node)) {
      ADD_MESSAGE_TO_ERR_STACK(
          err, strFormat("Encountered Error while loading node %s",
                         jitNodeToString(node).c_str()));
      return err;
    }
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

  ValueMapping mapping(nodeValue);
  mapping.setCorrectType(correctType);
  auto p = valueMap_.emplace(value, std::move(mapping));

  RETURN_ERR_IF_NOT(p.second, glow::strFormat("Value %s is already mapped",
                                              value->debugNameBase().c_str()));
  // Overwrite map, since sometimes we remap an exist nodeValue to a new jit
  // node.
  if (valueMapReverse_.count(nodeValue)) {
    valueMapReverse_.erase(nodeValue);
  }
  auto q = valueMapReverse_.insert({nodeValue, value});
  RETURN_ERR_IF_NOT(q.second,
                    glow::strFormat("Value %s's reversed mapping is occupied",
                                    value->debugNameBase().c_str()));
  return Error::success();
}

Error PyTorchModelLoader::addValueMapping(const torch::jit::Value *value,
                                          glow::NodeValue nodeValue) {
  auto correctType =
      elemKindToScalarType(nodeValue.getType()->getElementType());
  RETURN_ERR(addValueMapping(value, nodeValue, correctType));
}

Error PyTorchModelLoader::removeValueMapping(const torch::jit::Value *value) {
  auto it = valueMap_.find(value);
  if (it != valueMap_.end()) {
    // if this value is IValue, then just ignore it, since we dont have
    // valueMapReverse for IValue.
    if (it->second.getMappingType() == ValueMappingType::NodeValue) {
      glow::NodeValue n;
      ASSIGN_VALUE_OR_RETURN_ERR(n, it->second.getMappedNodeValue());
      valueMapReverse_.erase(n);
    }
  }
  valueMap_.erase(value);
  return Error::success();
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
  RETURN_ERR(addValueMapping(value, std::move(glowIValue), correctType));
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
    return MAKE_ERR(glow::strFormat("No mapping found for Value %s",
                                    value->debugNameBase().c_str()));
  }
  auto &mappingValue = it->second;
  if (mappingValue.getMappingType() == ValueMappingType::IValue) {
    return MAKE_ERR(glow::strFormat(
        "Found a GlowIValue instead of a NodeValue for this Value: %s",
        value->debugNameBase().c_str()));
  }

  return mappingValue.getMappedNodeValue();
}

Expected<glow::GlowIValue *>
PyTorchModelLoader::getGlowIValueForValue(const torch::jit::Value *value) {
  auto it = valueMap_.find(value);
  if (it == valueMap_.end()) {
    return MAKE_ERR(glow::strFormat("No mapping found fo Value %s",
                                    value->debugNameBase().c_str()));
  }
  auto &mappingValue = it->second;
  if (mappingValue.getMappingType() != ValueMappingType::IValue) {
    return MAKE_ERR(glow::strFormat(
        "Found a NodeValue instead of a GlowIValue for this Value: %s",
        value->debugNameBase().c_str()));
  }
  return mappingValue.getMappedGlowIValue();
}

glow::NodeValue PyTorchModelLoader::rescaleUIntToInt(glow::NodeValue input) {
  auto *inputTy = input.getType();
  if (inputTy->getElementType() == ElemKind::UInt8QTy) {
    auto dqInput = F_.createDequantize("dequantize", input, ElemKind::FloatTy);
    auto *outputTy = F_.getParent()->uniqueType(
        ElemKind::Int8QTy, inputTy->dims(), inputTy->getScale(),
        inputTy->getOffset() - UINT8_TO_INT8_SHIFT);
    auto *qOut = F_.createQuantize("quantize", dqInput, outputTy);
    return qOut->getResult();
  } else {
    return input;
  }
}

glow::NodeValue PyTorchModelLoader::rescaleIntToUint(glow::NodeValue input) {
  auto *inputTy = input.getType();
  if (inputTy->getElementType() == ElemKind::Int8QTy) {
    auto dqInput = F_.createDequantize("dequantize", input, ElemKind::FloatTy);
    auto *outputTy = F_.getParent()->uniqueType(
        ElemKind::UInt8QTy, inputTy->dims(), inputTy->getScale(),
        inputTy->getOffset() + UINT8_TO_INT8_SHIFT);
    auto *qOut = F_.createQuantize("quantize", dqInput, outputTy);
    return qOut->getResult();
  } else {
    return input;
  }
}

static std::pair<NodeValue, NodeValue>
extractChannelwiseQParams(Function &F, const at::Tensor &ptWeightTensor) {
  // extract qparams from ptWeightTensor.
  // Notice since the memory of qparams may not be continuous
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

  auto wScales = F.getParent()->createConstant("channelwise_scales_of_qconv",
                                               std::move(wScalesTensor));
  wScales->ensureIsOwned();
  auto wOffsets = F.getParent()->createConstant("channelwise_offsets_of_qconv",
                                                std::move(wOffsetsTensor));
  wOffsets->ensureIsOwned();

  return {wScales, wOffsets};
}

Expected<NodeValue>
PyTorchModelLoader::loadQuantizedConvImpl(const torch::jit::Node *ptNode,
                                          const bool isRelu) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 4, outputs, 1));

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
    input_mapping["scale"] = QuantizedConv3dInputs::scale;
    input_mapping["zero_point"] = QuantizedConv3dInputs::zero_point;
  } else {
    input_mapping["input"] = QuantizedConv2dInputs::input;
    input_mapping["packed_weights"] = QuantizedConv2dInputs::packed_weights;
    input_mapping["scale"] = QuantizedConv2dInputs::scale;
    input_mapping["zero_point"] = QuantizedConv2dInputs::zero_point;
  }

  at::Tensor ptWeightTensor;
  c10::optional<at::Tensor> ptBiasTensorTmp;
  std::vector<glow::unsigned_t> strides, pads, dilations;
  glow::unsigned_t groups;
  if (isConv3d) {
    auto packed_params = qparamsMap_.at(inputs[input_mapping["packed_weights"]])
                             .toCustomClass<ConvPackedParamsBase<3>>();
    std::tie(ptWeightTensor, ptBiasTensorTmp) = packed_params->unpack();
    // strides
    strides = castToGlowIntList(packed_params->stride());

    // dilations
    dilations = castToGlowIntList(packed_params->dilation());

    // pads
    std::vector<glow::unsigned_t> pad =
        castToGlowIntList(packed_params->padding());
    pads = {pad[0], pad[0], pad[1], pad[1], pad[2], pad[2]};

    // groups
    groups = static_cast<glow::unsigned_t>(packed_params->groups());
  } else {
    auto packed_params = qparamsMap_.at(inputs[input_mapping["packed_weights"]])
                             .toCustomClass<ConvPackedParamsBase<2>>();
    std::tie(ptWeightTensor, ptBiasTensorTmp) = packed_params->unpack();
    // strides
    strides = castToGlowIntList(packed_params->stride());

    // dilations
    dilations = castToGlowIntList(packed_params->dilation());

    // pads
    std::vector<glow::unsigned_t> pad =
        castToGlowIntList(packed_params->padding());
    DCHECK(pad[0] == pad[1]);
    pads = {pad[0], pad[0], pad[1], pad[1]};
    // groups
    groups = static_cast<glow::unsigned_t>(packed_params->groups());
  }

  bool isPerChannelQuantized =
      ptWeightTensor.is_quantized() &&
      ptWeightTensor.qscheme() == at::kPerChannelAffine;

  // unpacked weights
  auto ptWeightTensorContig = ptWeightTensor.contiguous();
  auto weightTensor = ptTensorToGlowTensor(ptWeightTensorContig);
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

  // quantized params
  float outScale;
  ASSIGN_VALUE_OR_RETURN_ERR(outScale, iValToDouble(getGlowIValueForValue(
                                           inputs[input_mapping["scale"]])));

  int32_t outOffset;
  ASSIGN_VALUE_OR_RETURN_ERR(
      outOffset,
      iValToInt(getGlowIValueForValue(inputs[input_mapping["zero_point"]])));

  // calc output type
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
    outTy =
        F_.getParent()->uniqueType(glow::ElemKind::Int8QTy, outDims, outScale,
                                   outOffset - UINT8_TO_INT8_SHIFT);

  } else {
    glow::ShapeNHWC inputShape(input.dims());
    glow::ShapeNHWC weightShape(weight.dims());
    kernels = {static_cast<glow::unsigned_t>(weightShape.h),
               static_cast<glow::unsigned_t>(weightShape.w)};
    auto outSz = glow::calculateConvPoolOutputDims(
        inputShape.h, inputShape.w, kernels, strides, pads, dilations);
    std::array<glow::dim_t, 4> outDims = {
        {input.dims()[0], outSz.first, outSz.second, weightShape.n}};
    outTy =
        F_.getParent()->uniqueType(glow::ElemKind::Int8QTy, outDims, outScale,
                                   outOffset - UINT8_TO_INT8_SHIFT);
  }

  // create qconv
  glow::NodeValue output;
  if (isPerChannelQuantized) {
    if (isConv3d) {
      RETURN_ERR_IF_NOT(
          std::all_of(dilations.cbegin(), dilations.cend(),
                      [](unsigned_t i) { return i == 1; }),
          "Dilation not supported for channelwise quantized conv3d");
    }

    NodeValue wScales, wOffsets;
    std::tie(wScales, wOffsets) = extractChannelwiseQParams(F_, ptWeightTensor);

    // Quantize the filter automatically (only if it is float). The bias is NOT
    // quantized automatically and is left at the disposal of each Backend to
    // quantize it later using custom logic.
    output = F_.createChannelwiseQuantizedConv(
                   "qconv_channelwise", input, weightConstant, biasConstant,
                   wScales, wOffsets, /* biasScales */ nullptr,
                   /* biasOffsets */ nullptr, outTy, kernels, strides, pads,
                   groups, dilations, /* quantizeFilter */ true,
                   /* quantizeBias */ false)
                 ->getResult();
  } else {
    if (isConv3d) {
      output = F_.createConv3D("qconv", input, weight, bias, outTy, kernels,
                               strides, pads, groups)
                   ->getResult();
    } else {
      output = F_.createConv("qconv", input, weight, bias, outTy, kernels,
                             strides, pads, groups, dilations)
                   ->getResult();
    }
  }
  if (isRelu) {
    output = F_.createRELU("qconv_output_relu", output)->getResult();
  }
  if (isConv3d) {
    output = F_.createTranspose("qconv_output_transpose", output, NTHWC2NCTHW)
                 ->getResult();
  } else {
    output = F_.createTranspose("qconv_output_transpose", output, NHWC2NCHW)
                 ->getResult();
  }
  return Expected<NodeValue>(output);
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
                                          outOffset - UINT8_TO_INT8_SHIFT);

  glow::AddNode *qadd = F_.createAdd("quantized_add", outTy, lhs, rhs);
  auto output = qadd->getResult();

  c10::ScalarType dtype;
  RETURN_IF_ERR(getCorrectTypeMapping(dtype, inputs[QuantizedAddInputs::lhs]));
  RETURN_ERR(addValueMapping(outputs[0], output, dtype));
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
                                          outOffset - UINT8_TO_INT8_SHIFT);

  glow::AddNode *qadd = F_.createAdd("quantized_add", outTy, lhs, rhs);
  glow::ReluNode *qrelu = F_.createRELU("quantized_relu", qadd);
  auto output = qrelu->getResult();

  c10::ScalarType dtype;
  RETURN_IF_ERR(
      getCorrectTypeMapping(dtype, inputs[QuantizedAddReluInputs::lhs]));
  RETURN_ERR(addValueMapping(outputs[0], output, dtype));
}

Error PyTorchModelLoader::loadQuantizedMul(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, -2, outputs, 1));

  if (inputs.size() == 2) {
    // Tensor * Scalar
    glow::NodeValue lhs;
    ASSIGN_VALUE_OR_RETURN_ERR(
        lhs, getGlowNodeValueForValue(inputs[QuantizedMulScalarInputs::lhs]));

    TypeRef inputType = lhs.getType();
    auto inputElemKind = inputType->getElementType();

    RETURN_ERR_IF_NOT(
        isQuantizedElemKind(inputElemKind),
        "For quantized::mul_scalar node in Glow, lhs must be quantized.");
    float scale = inputType->getScale();
    int32_t offset = inputType->getOffset();

    // qmin + qmax for int8 is
    // -128 + 127 = -1
    int32_t offsetMaxMin = -1;

    float rhsFloat;
    glow::NodeValue output;

    // rhs should be either a Double IValue or a Constant NodeValue which is a
    // Double Scalar.
    if (hasGlowIValueForValue(inputs[QuantizedMulScalarInputs::rhs])) {
      ASSIGN_VALUE_OR_RETURN_ERR(rhsFloat,
                                 iValToDouble(getGlowIValueForValue(
                                     inputs[QuantizedMulScalarInputs::rhs])));
    } else {
      glow::NodeValue rhsNodeValue;
      ASSIGN_VALUE_OR_RETURN_ERR(
          rhsNodeValue,
          getGlowNodeValueForValue(inputs[QuantizedMulScalarInputs::rhs]));

      auto rhsElementType = rhsNodeValue.getType()->getElementType();
      size_t rhsSize = rhsNodeValue.getType()->size();

      RETURN_ERR_IF_NOT(
          rhsNodeValue.getNode()->getKind() == Kinded::Kind::ConstantKind,
          "Expect rhs of quantized mul to be scalar or constant.");
      RETURN_ERR_IF_NOT(rhsElementType == glow::ElemKind::FloatTy,
                        "Expect rhs constant data type to be float.");
      RETURN_ERR_IF_NOT(rhsSize == 1, "Expect rhs constant to be a scalar");

      glow::Constant *rhsConstant =
          llvm::dyn_cast<glow::Constant>(rhsNodeValue.getNode());
      rhsFloat = rhsConstant->getPayload().getHandle<float>().at({0});
    }
    if (rhsFloat == 0) {
      // If mul's rhs is 0, we do not calc but just create a all-0-constant
      glow::Tensor t(glow::ElemKind::Int8QTy, lhs.dims(), 1.0, 0);
      t.zero();
      output =
          F_.getParent()->createConstant("constant", std::move(t))->getOutput();

    } else {
      float rhsScale, outScale;
      int32_t outOffset;
      float rhsQVal;
      if (rhsFloat > 0) {
        // positive
        rhsScale = rhsFloat;
        outScale = rhsFloat * scale;
        outOffset = offset;
        rhsQVal = 1;
      } else {
        // negative
        rhsScale = -rhsFloat;
        outScale = rhsFloat * scale * -1;
        outOffset = offsetMaxMin - offset;
        rhsQVal = -1;
      }

      auto rhsTy = glow::Type(inputElemKind, lhs.dims(), rhsScale, 0);
      glow::Tensor t(rhsTy);
      t.init(glow::Tensor::InitKind::Broadcast, rhsQVal,
             F_.getParent()->getPRNG());

      glow::NodeValue rhs =
          F_.getParent()
              ->createConstant("quantized_mul_scalar_rhs_constant",
                               std::move(t))
              ->getOutput();

      auto outTy = F_.getParent()->uniqueType(ElemKind::Int8QTy, lhs.dims(),
                                              outScale, outOffset);

      glow::MulNode *qmul =
          F_.createMul("quantized_mul_scalar", outTy, lhs, rhs);
      output = qmul->getResult();
    }
    c10::ScalarType dtype;
    RETURN_IF_ERR(
        getCorrectTypeMapping(dtype, inputs[QuantizedMulInputs::lhs]));
    RETURN_ERR(addValueMapping(outputs[0], output, dtype));
  } else {
    // Tensor * Tensor
    RETURN_ERR_IF_NOT(inputs.size() == 4,
                      "quantized::mul must have 4 inputs for tensor * tensor, "
                      "or 2 inputs for tensor * scalar.");
    glow::NodeValue lhs;
    ASSIGN_VALUE_OR_RETURN_ERR(
        lhs, getGlowNodeValueForValue(inputs[QuantizedMulInputs::lhs]));
    glow::NodeValue rhs;
    ASSIGN_VALUE_OR_RETURN_ERR(
        rhs, getGlowNodeValueForValue(inputs[QuantizedMulInputs::rhs]));

    // scale
    float outScale;
    ASSIGN_VALUE_OR_RETURN_ERR(
        outScale,
        iValToDouble(getGlowIValueForValue(inputs[QuantizedMulInputs::scale])));

    // zero_point
    int32_t outOffset;
    ASSIGN_VALUE_OR_RETURN_ERR(outOffset,
                               iValToInt(getGlowIValueForValue(
                                   inputs[QuantizedMulInputs::zero_point])));

    TypeRef inputType = lhs.getType();
    auto outDims = inputType->dims();
    auto outTy = F_.getParent()->uniqueType(
        ElemKind::Int8QTy, outDims, outScale, outOffset - UINT8_TO_INT8_SHIFT);

    RETURN_ERR_IF_NOT(
        lhs.dims().size() == rhs.dims().size(),
        glow::strFormat(
            "LHS and RHS must have number of dimensions, but LHS got "
            "%lu , RHS got %lu .",
            lhs.dims().size(), rhs.dims().size()));
    auto *bcast =
        F_.createBroadcast("broadcasted_rhs_quant_mul", rhs, lhs.dims(), 0);
    glow::MulNode *qmul = F_.createMul("quantized_mul", outTy, lhs, bcast);
    auto output = qmul->getResult();

    c10::ScalarType dtype;
    RETURN_IF_ERR(
        getCorrectTypeMapping(dtype, inputs[QuantizedMulInputs::lhs]));
    RETURN_ERR(addValueMapping(outputs[0], output, dtype));
  }
}

// implementation for per_tensor and per_channel quantized linear from either
// packed or unpacked linear
Error PyTorchModelLoader::loadQuantizedLinearImpl(
    NodeValue input, NodeValue weights, NodeValue bias, NodeValue wScales,
    NodeValue wOffsets, float outScale, int64_t outZeroPoint,
    const torch::jit::Value *outputValue, c10::ScalarType outputDtype) {
  bool isRowwiseQuantized = false;
  if (wScales) {
    RETURN_ERR_IF_NOT(wOffsets, "Expected both weight scales and offsets for "
                                "per_channel quantized linear");
    isRowwiseQuantized = true;
  } else {
    RETURN_ERR_IF_NOT(!wOffsets, "Expected neight weight scales nor offsets "
                                 "for per_tensor quantized linear");
  }

  // Flatten outer dims if necessary
  auto inputDims = input.dims();
  if (inputDims.size() > 2) {
    input = F_.createFlatten("flatten", input, inputDims.size() - 1);
  }

  auto outTy = F_.getParent()->uniqueType(
      ElemKind::Int8QTy, {input.dims()[0], weights.dims()[0]}, outScale,
      outZeroPoint - UINT8_TO_INT8_SHIFT);

  NodeValue output;
  if (isRowwiseQuantized) {
    auto rowwiseFC = F_.createRowwiseQuantizedFullyConnected(
        "rowwise_quantized_fc", input, weights,
        llvm::dyn_cast<glow::Constant>(wScales),
        llvm::dyn_cast<glow::Constant>(wOffsets), bias, outTy);
    output = rowwiseFC->getResult();
  } else {
    weights = F_.createTranspose("weight_transpose", weights, {1, 0});
    auto fc =
        F_.createFullyConnected("quantized_fc", input, weights, bias, outTy);
    output = fc->getResult();
  }

  // Restore original outer dims
  if (inputDims.size() > 2) {
    std::vector<dim_t> finalDims = inputDims.vec();
    finalDims.back() = output.dims().back();
    output = F_.createReshape("expand", output, finalDims);
  }

  RETURN_ERR(addValueMapping(outputValue, output, outputDtype));
}

Error PyTorchModelLoader::loadQuantizedLinear(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 4, outputs, 1));

  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(
      input, getGlowNodeValueForValue(inputs[QuantizedLinearInputs::input]));

  CHECK(qparamsMap_.count(inputs[QuantizedLinearInputs::packed_weights]));
  auto packedParams = qparamsMap_[inputs[QuantizedLinearInputs::packed_weights]]
                          .toCustomClass<LinearPackedParamsBase>();

  at::Tensor ptWeightTensor;
  c10::optional<at::Tensor> ptBiasTensorTmp;
  std::tie(ptWeightTensor, ptBiasTensorTmp) = packedParams->unpack();

  // unpacked weights
  auto weightTensor = ptTensorToGlowTensor(ptWeightTensor);
  glow::Constant *weightConstant = F_.getParent()->createConstant(
      "quantized_linear_weights", std::move(weightTensor));
  weightConstant->ensureIsOwned();
  RETURN_ERR_IF_NOT(weightConstant->dims().size() == 2,
                    "Expected 2d Linear weights");
  auto weights = weightConstant->getOutput();

  // unpacked bias
  glow::Tensor biasTensor;
  if (ptBiasTensorTmp.has_value()) {
    auto ptBiasTensor = ptBiasTensorTmp.value().contiguous();
    biasTensor = ptTensorToGlowTensor(ptBiasTensor);
  } else {
    biasTensor = glow::Tensor(glow::ElemKind::FloatTy, {weights.dims()[0]});
    biasTensor.zero();
  }

  glow::Constant *biasConstant = F_.getParent()->createConstant(
      "quantized_linear_bias", std::move(biasTensor));
  biasConstant->ensureIsOwned();
  auto bias = biasConstant->getOutput();

  float outScale;
  ASSIGN_VALUE_OR_RETURN_ERR(outScale,
                             to32Bit(iValToDouble(getGlowIValueForValue(
                                 inputs[QuantizedLinearInputs::scale]))));

  int64_t outZeroPoint;
  ASSIGN_VALUE_OR_RETURN_ERR(outZeroPoint,
                             iValToInt(getGlowIValueForValue(
                                 inputs[QuantizedLinearInputs::zero_point])));

  bool isRowwiseQuantized = ptWeightTensor.is_quantized() &&
                            ptWeightTensor.qscheme() == at::kPerChannelAffine;

  NodeValue wScales, wOffsets;
  if (isRowwiseQuantized) {
    std::tie(wScales, wOffsets) = extractChannelwiseQParams(F_, ptWeightTensor);
  }

  c10::ScalarType dtype;
  RETURN_IF_ERR(
      getCorrectTypeMapping(dtype, inputs[QuantizedLinearInputs::input]));

  return loadQuantizedLinearImpl(input, weights, bias, wScales, wOffsets,
                                 outScale, outZeroPoint, outputs[0], dtype);
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

  glow::NodeValue weights;
  ASSIGN_VALUE_OR_RETURN_ERR(
      weights,
      getGlowNodeValueForValue(inputs[QuantizedUnpackedLinearInputs::weight]));
  RETURN_ERR_IF_NOT(weights.dims().size() == 2, "Expected 2d Linear weights");

  float outScale;
  ASSIGN_VALUE_OR_RETURN_ERR(
      outScale, to32Bit(iValToDouble(getGlowIValueForValue(
                    inputs[QuantizedUnpackedLinearInputs::scale]))));

  int64_t outZeroPoint;
  ASSIGN_VALUE_OR_RETURN_ERR(
      outZeroPoint, iValToInt(getGlowIValueForValue(
                        inputs[QuantizedUnpackedLinearInputs::zero_point])));

  // Get bias or create a zero bias if no bias is found.
  glow::NodeValue bias = loadNodeValueOrCreateBroadcastedConstant(
      inputs[QuantizedUnpackedLinearInputs::bias], "quantized_linear_bias",
      glow::Type(ElemKind::FloatTy, {weights.dims()[0]}), 0.0);

  // Choose bias quantization params and quantize it.
  glow::Constant *biasConstant = llvm::dyn_cast<glow::Constant>(bias.getNode());

  c10::ScalarType dtype;
  RETURN_IF_ERR(getCorrectTypeMapping(
      dtype, inputs[QuantizedUnpackedLinearInputs::input]));

  auto ptWeightTensor =
      qparamsMap_.at(inputs[QuantizedUnpackedLinearInputs::weight]).toTensor();

  bool isRowwiseQuantized = ptWeightTensor.is_quantized() &&
                            ptWeightTensor.qscheme() == at::kPerChannelAffine;

  NodeValue wScales, wOffsets;
  if (isRowwiseQuantized) {
    std::tie(wScales, wOffsets) = extractChannelwiseQParams(F_, ptWeightTensor);
  }

  return loadQuantizedLinearImpl(input, weights, bias, wScales, wOffsets,
                                 outScale, outZeroPoint, outputs[0], dtype);
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

  RETURN_ERR(addValueMapping(outputs[0], output));
}

template <typename T, typename U>
Expected<NodeValue> iValueBroadcastingHelper(glow::Function &F_, U &&val,
                                             TypeRef type) {
  T constVal;
  ASSIGN_VALUE_OR_RETURN_ERR(constVal,
                             static_cast_expected<T>(std::forward<U>(val)));

  if (type->getElementType() == ElemKind::FloatTy) {
    return F_.createSplat("ivalConstSplat", type, constVal);
  } else {
    glow::Tensor t(type);
    t.init(glow::Tensor::InitKind::Broadcast, constVal,
           F_.getParent()->getPRNG());
    return F_.getParent()
        ->createConstant("ivalConstBcast", std::move(t))
        ->getOutput();
  }
}

Expected<NodeValue> PyTorchModelLoader::loadNodeValueOrBroadcastedIValue(
    const torch::jit::Value *value, TypeRef type) {
  if (hasGlowNodeValueForValue(value)) {
    return getGlowNodeValueForValue(value);
  } else {
    GlowIValue *ival;
    ASSIGN_VALUE_OR_RETURN_ERR(ival, getGlowIValueForValue(value));

    auto elemKind = type->getElementType();

    if (ival->isInt() && elemKind == ElemKind::Int64ITy) {
      return iValueBroadcastingHelper<int64_t>(F_, ival->toInt(), type);
    } else if (ival->isInt() && elemKind == ElemKind::Int32ITy) {
      return iValueBroadcastingHelper<int32_t>(F_, ival->toInt(), type);
    } else if (ival->isInt() && elemKind == ElemKind::FloatTy) {
      return iValueBroadcastingHelper<float>(F_, ival->toInt(), type);
    } else if (ival->isDouble() && elemKind == ElemKind::FloatTy) {
      return iValueBroadcastingHelper<float>(F_, ival->toDouble(), type);
    } else {
      return MAKE_ERR(strFormat(
          "unsupported IValue broadcasting from `%s` to a tensor of `%s`.",
          ival->getTagString(), type->getElementName().str().c_str()));
    }
  }
}

Error PyTorchModelLoader::loadTypeAs(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 2, outputs, 1));

  glow::NodeValue dataValue;
  glow::NodeValue typeNode;
  ASSIGN_VALUE_OR_RETURN_ERR(dataValue, getGlowNodeValueForValue(inputs[0]));
  ASSIGN_VALUE_OR_RETURN_ERR(typeNode, getGlowNodeValueForValue(inputs[1]));
  auto typeAsType = typeNode.getType();
  auto inputShape = dataValue.getType();

  if (typeAsType->getElementType() == inputShape->getElementType()) {
    // nop conversion
    RETURN_ERR(addValueMapping(outputs[0], dataValue));
  }

  auto outType = F_.getParent()->uniqueType(typeAsType->getElementType(),
                                            inputShape->dims());

  glow::ConvertToNode *glowNode =
      F_.createConvertTo("typeas", dataValue, outType);

  RETURN_ERR(addValueMapping(outputs[0], glowNode->getResult()));
}

Error PyTorchModelLoader::loadContiguous(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 2, outputs, 1));

  glow::NodeValue dataValue;
  ASSIGN_VALUE_OR_RETURN_ERR(dataValue, getGlowNodeValueForValue(inputs[0]));

  RETURN_ERR(addValueMapping(outputs[0], dataValue));
}

Error PyTorchModelLoader::loadDetach(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 1, outputs, 1));

  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(input, getGlowNodeValueForValue(inputs[0]));

  RETURN_ERR(addValueMapping(outputs[0], input));
}

template <typename GlowNode>
Expected<NodeValue> PyTorchModelLoader::loadArithmeticNode(
    llvm::StringRef name, const torch::jit::Value *lhs,
    const torch::jit::Value *rhs, bool convertToDefaultType) {
  glow::NodeValue lhsInput;
  glow::NodeValue rhsInput;

  if (hasGlowNodeValueForValue(lhs)) {
    ASSIGN_VALUE_OR_RETURN_ERR(lhsInput, getGlowNodeValueForValue(lhs));
    ASSIGN_VALUE_OR_RETURN_ERR(
        rhsInput, loadNodeValueOrBroadcastedIValue(rhs, lhsInput.getType()));
  } else if (hasGlowNodeValueForValue(rhs)) {
    ASSIGN_VALUE_OR_RETURN_ERR(rhsInput, getGlowNodeValueForValue(rhs));
    ASSIGN_VALUE_OR_RETURN_ERR(
        lhsInput, loadNodeValueOrBroadcastedIValue(lhs, rhsInput.getType()));
  } else {
    return MAKE_ERR("Either lhs or rhs of arithmetic node must be a tensor");
  }

  // For aten::div, it will promote the output to default scalar type if both
  // inputs are of integer type. However, Glow requires inputs and output have
  // the same type. In order to achieve same behavior as Pytorch div, we convert
  // the inputs to default scalar type if they are both integer.
  if (convertToDefaultType) {
    if (isNonQuantizedIntElemKind(rhsInput.getElementType()) &&
        isNonQuantizedIntElemKind(lhsInput.getElementType())) {
      auto glowElemKind = scalarTypeToElemKind(
          at::typeMetaToScalarType(at::get_default_dtype()));

      lhsInput = F_.createConvertTo(
          "lhs_to", lhsInput,
          F_.getParent()->uniqueType(glowElemKind, lhsInput.getType()->dims()));
      rhsInput = F_.createConvertTo(
          "rhs_to", rhsInput,
          F_.getParent()->uniqueType(glowElemKind, rhsInput.getType()->dims()));
    }
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

  RETURN_ERR(addValueMapping(outputs[0], res));
}

Error PyTorchModelLoader::loadDiv(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 2, outputs, 1));

  glow::NodeValue res;
  ASSIGN_VALUE_OR_RETURN_ERR(
      res, loadArithmeticNode<glow::DivNode>("div", inputs[0], inputs[1],
                                             /* convertToDefaultType */ true));

  RETURN_ERR(addValueMapping(outputs[0], res));
}

Error PyTorchModelLoader::loadFloorDiv(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 2, outputs, 1));

  auto lhs = inputs[0];
  auto rhs = inputs[1];
  glow::NodeValue lhsInput;
  glow::NodeValue rhsInput;

  if (hasGlowNodeValueForValue(lhs)) {
    ASSIGN_VALUE_OR_RETURN_ERR(lhsInput, getGlowNodeValueForValue(lhs));
    ASSIGN_VALUE_OR_RETURN_ERR(
        rhsInput, loadNodeValueOrBroadcastedIValue(rhs, lhsInput.getType()));
  } else if (hasGlowNodeValueForValue(rhs)) {
    ASSIGN_VALUE_OR_RETURN_ERR(rhsInput, getGlowNodeValueForValue(rhs));
    ASSIGN_VALUE_OR_RETURN_ERR(
        lhsInput, loadNodeValueOrBroadcastedIValue(lhs, rhsInput.getType()));
  } else {
    return MAKE_ERR("Either lhs or rhs of floorDiv node must be a tensor");
  }

  // Current Pytorch FloorDiv is actually TruncDiv
  // github.com/pytorch/pytorch/issues/43874
  auto res = F_.createFloorDivWithBroadcast(
      "floor_divide", /* axis */ -1, lhsInput, rhsInput, /* truncate */ true);
  RETURN_ERR(addValueMapping(outputs[0], res));
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

  RETURN_ERR(addValueMapping(outputs[0], res));
}

Error PyTorchModelLoader::loadIndexSelect(
    const torch::jit::Node *ptNode) noexcept {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 3, outputs, 1));

  glow::NodeValue inputInput;
  glow::NodeValue indexInput;

  const torch::jit::Value *input = inputs[IndexSelectInputs::input];
  int64_t dimension = 0;
  const torch::jit::Value *index = inputs[IndexSelectInputs::index];

  ASSIGN_VALUE_OR_RETURN_ERR(
      dimension,
      iValToInt(getGlowIValueForValue(inputs[IndexSelectInputs::dimension])));

  ASSIGN_VALUE_OR_RETURN_ERR(inputInput, getGlowNodeValueForValue(input));
  ASSIGN_VALUE_OR_RETURN_ERR(indexInput, getGlowNodeValueForValue(index));

  size_t indexSize = indexInput.getType()->dims().size();
  RETURN_ERR_IF_NOT(
      indexInput.getType()->dims().size() == 1,
      glow::strFormat("Index must be a 1-D tensor. dims: %ld", indexSize));

  RETURN_ERR_IF_NOT(indexInput.getType()->getElementType() ==
                        ElemKind::Int64ITy,
                    "Index must be a LongTensor");

  GatherNode *glowNode =
      F_.createGather("index_select", inputInput, indexInput, dimension);

  RETURN_ERR(addValueMapping(outputs[0], glowNode->getResult()));
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

  RETURN_ERR(addValueMapping(outputs[0], res));
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

  RETURN_ERR(addValueMapping(outputs[0], res));
}

Error PyTorchModelLoader::loadLog(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 1, outputs, 1));

  glow::NodeValue glowInput;
  ASSIGN_VALUE_OR_RETURN_ERR(glowInput, getGlowNodeValueForValue(inputs[0]));

  return addValueMapping(outputs[0], F_.createLog("log", glowInput));
}

Error PyTorchModelLoader::loadSum(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();

  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, -2, outputs, 1));

  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(input, getGlowNodeValueForValue(inputs[0]));

  GlowIValue *dtypeIVal = nullptr;
  ASSIGN_VALUE_OR_RETURN_ERR(
      dtypeIVal, getGlowIValueForValue(ptNode->namedInput("dtype")));

  std::vector<int64_t> *axes;
  std::vector<unsigned_t> glowAxes;

  bool keepDim = false;
  bool needsFlatten = false;
  // Load torch.sum(input, dtype)
  if (inputs.size() == 2) {
    glowAxes.push_back(0);
    needsFlatten = true;
  } else {
    // Load torch.sum(input, axis, keepdim, dtype)
    ASSIGN_VALUE_OR_RETURN_ERR(axes,
                               iValToIntList(getGlowIValueForValue(inputs[1])));
    RETURN_ERR_IF_NOT(axes->size() == 1,
                      "Only a single axis is supported for aten::sum.");

    GlowIValue *keepDimIVal;
    ASSIGN_VALUE_OR_RETURN_ERR(
        keepDimIVal, getGlowIValueForValue(ptNode->namedInput("keepdim")));
    if (keepDimIVal->getTag() != GlowIValue::Tag::None) {
      ASSIGN_VALUE_OR_RETURN_ERR(keepDim, iValToBool(keepDimIVal));
    }

    const auto inputRank = input.getType()->dims().size();
    for (auto axis : *axes) {
      RETURN_ERR_IF_NOT((axis < 0 ? axis >= -inputRank : axis < inputRank),
                        "Axis must be in the range [-r, r-1] for aten::sum.");
      // Convert negative axes to corresponding positive axes
      if (axis < 0) {
        axis += inputRank;
      }
      glowAxes.push_back(static_cast<unsigned_t>(axis));
    }
  }

  const bool needsConvertTo = dtypeIVal->getTag() != GlowIValue::Tag::None;
  ConvertToNode *toNode = nullptr;
  if (needsConvertTo) {
    int32_t dtype;
    ASSIGN_VALUE_OR_RETURN_ERR(dtype, iValToInt(dtypeIVal));
    auto glowElemKind =
        scalarTypeToElemKind(static_cast<c10::ScalarType>(dtype));
    auto toType =
        F_.getParent()->uniqueType(glowElemKind, input.getType()->dims());
    toNode = F_.createConvertTo("to", input, toType);
  }

  ReshapeNode *flattenNode = nullptr;
  if (needsFlatten) {
    flattenNode = F_.createFlatten(
        "flatten", needsConvertTo ? static_cast<NodeValue>(toNode) : input,
        needsConvertTo ? toNode->getResult().getType()->dims().size()
                       : input.dims().size());
  }

  auto batchedReduceAddNode = F_.createBatchedReduceAdd(
      "sum",
      needsFlatten     ? static_cast<NodeValue>(flattenNode)
      : needsConvertTo ? static_cast<NodeValue>(toNode)
                       : input,
      glowAxes);

  if (!keepDim) {
    return addValueMapping(outputs[0], batchedReduceAddNode);
  } else {
    // If keepDim is true we need to insert the removed dimension(s) manually by
    // reshaping
    std::vector<dim_t> shape =
        batchedReduceAddNode->getResult().getType()->dims();
    std::sort(glowAxes.begin(), glowAxes.end());
    for (const auto &axis : glowAxes) {
      shape.insert(shape.begin() + axis, static_cast<dim_t>(1));
    }
    auto reshapeNode = F_.createReshape("reshape", batchedReduceAddNode, shape);
    return addValueMapping(outputs[0], reshapeNode);
  }
}

Error PyTorchModelLoader::loadMax(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 2, outputs, 1));

  glow::NodeValue lhs;
  ASSIGN_VALUE_OR_RETURN_ERR(lhs, getGlowNodeValueForValue(inputs[0]));
  glow::NodeValue rhs;
  ASSIGN_VALUE_OR_RETURN_ERR(rhs, getGlowNodeValueForValue(inputs[1]));

  glow::MaxNode *glowNode =
      F_.createNodeWithBroadcast<MaxNode>("max", -1, lhs, rhs);
  RETURN_ERR(addValueMapping(outputs[0], glowNode->getResult()));
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

  RETURN_ERR(addValueMapping(outputs[0], std::move(glowIVal)));
}

Error PyTorchModelLoader::loadListConstruct(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  // Requires -1 because this requires at least one input.
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, -1, outputs, 1));
  GlowIValue glowIVal;
  // Get the Tag of the first input to use for the whole list.
  if (hasGlowIValueForValue(inputs[0])) {
    // If it is IValue
    GlowIValue *firstInputIVal;
    ASSIGN_VALUE_OR_RETURN_ERR(firstInputIVal,
                               getGlowIValueForValue(inputs[0]));
    auto tag = firstInputIVal->getTag();

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
        ASSIGN_VALUE_OR_RETURN_ERR(
            x, iValToBool(getGlowIValueForValue(inputs[i])));
        bools.push_back(x);
      }
      glowIVal.fromBoolList(std::move(bools));
    } else {
      return MAKE_ERR(
          "Encountered an unsupported GlowIValue type for ListConstruct");
    }
  } else if (hasGlowNodeValueForValue(inputs[0])) {
    // If it is a NodeValue, which we will store as a NodeValueList IValue.
    std::vector<glow::NodeValue> nodeValues;
    for (size_t i = 0; i < inputs.size(); i++) {
      glow::NodeValue x;
      ASSIGN_VALUE_OR_RETURN_ERR(x, getGlowNodeValueForValue(inputs[i]));
      nodeValues.push_back(x);
    }
    glowIVal.fromNodeValueList(std::move(nodeValues));
  } else {
    // Should never reach here
    return MAKE_ERR("Encountered unknown JIT Value mapping");
  }
  RETURN_ERR(addValueMapping(outputs[0], std::move(glowIVal)));
}

/// Mirroring the implementation in
/// caffe2/aten/src/ATen/native/TypeProperties.cpp
static inline c10::ScalarType promote_skip_undefined(c10::ScalarType a,
                                                     c10::ScalarType b) {
  if (a == c10::ScalarType::Undefined) {
    return b;
  }
  if (b == c10::ScalarType::Undefined) {
    return a;
  }
  return c10::promoteTypes(a, b);
}

/// Helper function to support upcasting in concat, we calculate the higher
/// type among a list of types. For example, the higher type of [half, float,
/// half, double] will be double. Similar to at::result_type().
static Expected<c10::ScalarType>
getHigherType(PyTorchModelLoader *loader,
              const c10::ArrayRef<const torch::jit::Value *> &values) {

  c10::ScalarType higherType = c10::ScalarType::Undefined;
  for (auto v : values) {
    c10::ScalarType dtype;
    RETURN_IF_ERR(loader->getCorrectTypeMapping(dtype, v));
    if (dtype != c10::ScalarType::QInt8 && dtype != c10::ScalarType::QUInt8) {
      higherType = promote_skip_undefined(higherType, dtype);
    }
  }
  return higherType;
}

/// Helper function for \p loadFusedConcat, \p
/// loadFusedBroadcastConcat, \p loadFusedStack and \p loadFusedBroadcastCat
/// with \p isStack to select whether to load stack or concat, and \p
/// doBroadcast to select whether broadcast is enabled for concat. \returns
/// error on failures, otherwise the created concat node reference.
static Expected<glow::ConcatNode *>
createConcatNode(PyTorchModelLoader *loader, Function &F,
                 const torch::jit::Node *ptNode, bool isStack,
                 bool doBroadcast) noexcept {

  auto inputs = ptNode->inputs();

  // Get number of input dimensions
  glow::NodeValue glowInput0;
  ASSIGN_VALUE_OR_RETURN_ERR(glowInput0,
                             loader->getGlowNodeValueForValue(inputs[0]));
  size_t numInputDims = glowInput0.dims().size();

  int64_t dim = ptNode->i(at::attr::dim);

  // Convert negative dimension index into corresponding positive index if the
  // node is concat
  auto origDim = dim;
  if (!isStack && dim < 0) {
    dim += numInputDims;
  }

  RETURN_ERR_IF_NOT(dim < numInputDims + isStack && dim >= 0,
                    strFormat("Dim value of %ld is out of range. Valid "
                              "values are in the range [-%ld, %ld]",
                              origDim, numInputDims,
                              numInputDims - 1 + isStack));

  c10::ScalarType higherType;
  glow::ElemKind higherKind;
  ASSIGN_VALUE_OR_RETURN_ERR(higherType, getHigherType(loader, inputs));
  if (higherType != c10::ScalarType::Undefined) {
    higherKind = scalarTypeToElemKind(higherType);
  }

  // Final shape for all tensors after broadcast
  std::vector<int64_t> bcastShape(numInputDims, -1);
  std::vector<bool> needBroadcast(inputs.size(), false);
  bool noBroadcastNeeded = true;

  // Use mulitple vectors for hierarchical concats, the first vector is the
  // final concat
  std::vector<glow::NodeValue> glowInputs;
  for (size_t i = 0; i < inputs.size(); ++i) {
    glow::NodeValue glowInput;
    ASSIGN_VALUE_OR_RETURN_ERR(glowInput,
                               loader->getGlowNodeValueForValue(inputs[i]));

    RETURN_ERR_IF_NOT(numInputDims == glowInput.dims().size(),
                      "All inputs must have the same number of dimensions.");

    // Record broadcast shapes to perform broadcasting
    for (int d = 0; d < glowInput.dims().size(); ++d) {
      // For stack we can broadcast every dim
      if (d != dim || isStack) {
        if (bcastShape[d] < 0 && glowInput.dims()[d] != 1) {
          // record first non-singleton size for dim_i as broadcast shape
          bcastShape[d] = glowInput.dims()[d];
        } else if (glowInput.dims()[d] == 1) {
          needBroadcast[i] = true;
          noBroadcastNeeded = false;
        }
      }
    }

    if (higherType != c10::ScalarType::Undefined &&
        !isQuantizedElemKind(higherKind) &&
        glowInput.getElementType() != higherKind) {
      glow::ConvertToNode *toNode =
          F.createConvertTo("upcastForConcat", glowInput, higherKind);
      glowInputs.emplace_back(toNode->getResult());
    } else {
      glowInputs.emplace_back(std::move(glowInput));
    }
  }

  if (!doBroadcast || noBroadcastNeeded) {
    return F.createConcat(isStack ? "stack_concat" : "cat", glowInputs, dim);
  }
  // For concat, we perform opportunistic concat before broadcast if
  // the adjacent nodes can be broadcast the same way. Doing this saves the
  // number total OPs. For stack, we cannot perform this optimization.
  std::vector<glow::NodeValue> finalConcatInputs;
  std::vector<glow::NodeValue> partialConcatInputs;
  std::string prevConcatKey = "";

  // Helper function for concat and using Tile ops to expand.
  auto addConcatAndBroadcastNode =
      [&](const std::vector<glow::NodeValue> &nodes, const std::string &key) {
        glow::NodeValue output;
        if (nodes.size() > 1) {
          auto *concatNode = F.createConcat("cat_" + key, nodes, dim);
          output = concatNode->getResult();
        } else if (nodes.size() == 1) {
          output = nodes[0];
        } else {
          // Should not come to this branch, trust downstream to handle errors
          return output;
        }

        for (int d = 0; d < numInputDims; ++d) {
          if ((d != dim || isStack) && output.dims()[d] == 1 &&
              bcastShape[d] > 1) {
            output = F.createTile("tile_" + key + "_" + std::to_string(d),
                                  output, bcastShape[d], /* tile */
                                  d /* axis */)
                         ->getResult();
          }
        }
        return output;
      };

  for (size_t i = 0; i < glowInputs.size(); ++i) {
    // Use dimensions as key so we can concat those of the same dimensions
    // For example, for [1, 32, 1] when we concat dim=1, the key is '1_1',
    // and for [1, 32, 4] the key is '1_4'.
    // We use '_' as delimiter to conveniently reuse the key for node name.
    std::stringstream ss;
    for (int d = 0; d < glowInputs[i].dims().size(); ++d) {
      if (d != dim || isStack) {
        ss << glowInputs[i].dims()[d] << "_";
      }
    }
    auto key = ss.str();
    if (!partialConcatInputs.empty() &&
        (!needBroadcast[i] || key != prevConcatKey)) {
      finalConcatInputs.emplace_back(
          addConcatAndBroadcastNode(partialConcatInputs, key));
      partialConcatInputs.clear();
    }
    if (needBroadcast[i]) {
      // Doing this guarantees we don't merge before Tile for stack
      if (!isStack) {
        prevConcatKey = key;
      }
      partialConcatInputs.emplace_back(glowInputs[i]);
    } else {
      prevConcatKey = "";
      finalConcatInputs.emplace_back(glowInputs[i]);
    }
  }
  // In cast the last nodes (1 or more) need concat and broadcast
  if (!partialConcatInputs.empty()) {
    finalConcatInputs.emplace_back(
        addConcatAndBroadcastNode(partialConcatInputs, prevConcatKey));
  }

  return F.createConcat(isStack ? "stack_concat" : "cat", finalConcatInputs,
                        dim);
}

Error PyTorchModelLoader::loadFusedConcat(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, -1, outputs, 1));

  // In the case of a single input, just return it.
  if (inputs.size() == 1) {
    glow::NodeValue input;
    ASSIGN_VALUE_OR_RETURN_ERR(input, getGlowNodeValueForValue(inputs[0]));
    RETURN_ERR(addValueMapping(outputs[0], input));
  }
  glow::Node *node;
  ASSIGN_VALUE_OR_RETURN_ERR(node, createConcatNode(this, F_, ptNode,
                                                    false /* isStack */,
                                                    false /* doBroadcast */));
  RETURN_ERR(addValueMapping(outputs[0], node));
}

Error PyTorchModelLoader::loadFusedBroadcastConcat(
    const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, -1, outputs, 1));

  // In the case of a single input, just return it.
  if (inputs.size() == 1) {
    glow::NodeValue input;
    ASSIGN_VALUE_OR_RETURN_ERR(input, getGlowNodeValueForValue(inputs[0]));
    RETURN_ERR(addValueMapping(outputs[0], input));
  }
  glow::Node *node;
  ASSIGN_VALUE_OR_RETURN_ERR(node, createConcatNode(this, F_, ptNode,
                                                    false /* isStack */,
                                                    true /* doBroadcast */));
  RETURN_ERR(addValueMapping(outputs[0], node));
}

static glow::Node *
createReshapeNodeForStack(glow::Function &F, const torch::jit::Node *ptNode,
                          const glow::ConcatNode *concatNode) {
  auto inputs = ptNode->inputs();
  int64_t dim = ptNode->i(at::attr::dim);

  auto concat = concatNode->getResult();
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

  return F.createReshape("stack_reshape", concat, reshapeDims)->getResult();
}

Error PyTorchModelLoader::loadFusedStack(const torch::jit::Node *ptNode) {

  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, -1, outputs, 1));

  // In the case of a single input, just return it.
  if (inputs.size() == 1) {
    glow::NodeValue input;
    ASSIGN_VALUE_OR_RETURN_ERR(input, getGlowNodeValueForValue(inputs[0]));
    RETURN_ERR(addValueMapping(outputs[0], input));
  }

  glow::ConcatNode *node;
  ASSIGN_VALUE_OR_RETURN_ERR(node, createConcatNode(this, F_, ptNode,
                                                    true /* isStack */,
                                                    false /* doBroadcast */));

  RETURN_ERR(
      addValueMapping(outputs[0], createReshapeNodeForStack(F_, ptNode, node)));
}

Error PyTorchModelLoader::loadFusedBroadcastStack(
    const torch::jit::Node *ptNode) {

  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, -1, outputs, 1));

  // In the case of a single input, just return it.
  if (inputs.size() == 1) {
    glow::NodeValue input;
    ASSIGN_VALUE_OR_RETURN_ERR(input, getGlowNodeValueForValue(inputs[0]));
    RETURN_ERR(addValueMapping(outputs[0], input));
  }

  glow::ConcatNode *node;
  ASSIGN_VALUE_OR_RETURN_ERR(node, createConcatNode(this, F_, ptNode,
                                                    true /* isStack */,
                                                    true /* doBroadcast */));

  RETURN_ERR(
      addValueMapping(outputs[0], createReshapeNodeForStack(F_, ptNode, node)));
}

Error PyTorchModelLoader::loadCos(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 1, outputs, 1));

  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(input, getGlowNodeValueForValue(inputs[0]));

  return addValueMapping(outputs[0], F_.createCos("Cos", input));
}

Error PyTorchModelLoader::loadSin(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 1, outputs, 1));

  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(input, getGlowNodeValueForValue(inputs[0]));

  return addValueMapping(outputs[0], F_.createSin("Sin", input));
}

Error PyTorchModelLoader::loadAcos(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 1, outputs, 1));

  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(input, getGlowNodeValueForValue(inputs[0]));

  return addValueMapping(outputs[0], F_.createAcos("Acos", input));
}

Error PyTorchModelLoader::loadAsin(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 1, outputs, 1));

  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(input, getGlowNodeValueForValue(inputs[0]));

  return addValueMapping(outputs[0], F_.createAsin("Asin", input));
}

Error PyTorchModelLoader::loadAtan(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 1, outputs, 1));

  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(input, getGlowNodeValueForValue(inputs[0]));

  return addValueMapping(outputs[0], F_.createAtan("Atan", input));
}

Error PyTorchModelLoader::loadNumToTensor(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 1, outputs, 1));

  glow::GlowIValue *glowIValue;
  ASSIGN_VALUE_OR_RETURN_ERR(glowIValue, getGlowIValueForValue(inputs[0]));
  glow::Tensor t;

  if (glowIValue->isInt()) {
    int64_t input;
    ASSIGN_VALUE_OR_RETURN_ERR(input, glowIValue->toInt());
    t = glow::Tensor(glow::ElemKind::Int64ITy, {1});
    t.init(glow::Tensor::InitKind::Broadcast, input, F_.getParent()->getPRNG());
  } else if (glowIValue->isDouble()) {
    double input;
    ASSIGN_VALUE_OR_RETURN_ERR(input, glowIValue->toDouble());
    t = glow::Tensor(glow::ElemKind::FloatTy, {1});
    t.init(glow::Tensor::InitKind::Broadcast, input, F_.getParent()->getPRNG());
  } else {
    // Not a number
    return MAKE_ERR(strFormat(
        "Expected integer/double GlowIValue type in NumToTensor, but get: %s",
        glowIValue->getTagString()));
  }
  auto output =
      F_.getParent()->createConstant("NumToTensor_output", std::move(t));
  RETURN_ERR(addValueMapping(outputs[0], output));
}

Error PyTorchModelLoader::loadShapeAsTensor(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 1, outputs, 1));

  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(input, getGlowNodeValueForValue(inputs[0]));

  auto dims = input.getType()->dims();
  std::vector<dim_t> outputValues{dims.begin(), dims.end()};

  auto type =
      F_.getParent()->uniqueType(glow::ElemKind::Int64ITy, outputValues.size());

  auto outputTensor = glow::Tensor(outputValues.data(), type);

  auto output = F_.getParent()->createConstant("ShapeAsTensor_output",
                                               std::move(outputTensor));

  output->ensureIsOwned(); // Prevents heap use after free
  RETURN_ERR(addValueMapping(ptNode->output(), output));
}

Error PyTorchModelLoader::loadInt(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 1, outputs, 1));

  // LoadInt receive an input constant node,
  // generate an glow iValue.
  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(input, getGlowNodeValueForValue(inputs[0]));
  auto inputElementType = input.getType()->getElementType();

  glow::Constant *intConstant = llvm::dyn_cast<glow::Constant>(input.getNode());
  RETURN_ERR_IF_NOT(
      intConstant,
      strFormat("Expected input to be a Constant in loadInt, but found: %s",
                input.getNode()->getKindName()));
  // Also need to check if intConstant is a scalar
  int value;

  if (inputElementType == glow::ElemKind::Int32ITy) {
    value = intConstant->getPayload().getHandle<int32_t>().at({0});
  } else if (inputElementType == glow::ElemKind::Int64ITy) {
    value = intConstant->getPayload().getHandle<int64_t>().at({0});
  } else if (inputElementType == glow::ElemKind::FloatTy) {
    auto value_f = intConstant->getPayload().getHandle<float>().at({0});
    value = static_cast<int>(value_f);
  } else {
    return MAKE_ERR("Expected integer/float tensor in loadInt");
  }
  glow::GlowIValue glowIVal;
  // No matter input is int32 or int64, it is int in glowIVal.
  // When using NumToTensor, this int will transformed into int64 again.
  glowIVal.fromInt(value);
  RETURN_ERR(addValueMapping(outputs[0], std::move(glowIVal)));
}

Error PyTorchModelLoader::loadZeros(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 5, outputs, 1));

  std::vector<int64_t> *inputSizePT;
  ASSIGN_VALUE_OR_RETURN_ERR(inputSizePT, iValToIntList(getGlowIValueForValue(
                                              inputs[ZerosInputs::size])));
  std::vector<glow::dim_t> inputSizeGlow;
  for (int i = 0; i < inputSizePT->size(); i++) {
    inputSizeGlow.push_back(static_cast<glow::dim_t>((*inputSizePT)[i]));
  }
  llvm::ArrayRef<glow::dim_t> inputSizeArrayRef(inputSizeGlow);

  int32_t dtype;
  ASSIGN_VALUE_OR_RETURN_ERR(
      dtype, iValToInt(getGlowIValueForValue(inputs[ZerosInputs::dtype])));
  auto glowElemKind = scalarTypeToElemKind(static_cast<c10::ScalarType>(dtype));

  auto output =
      F_.createSplat(
            "zeros",
            F_.getParent()->uniqueType(glowElemKind, inputSizeArrayRef), 0)
          ->getResult();
  return addValueMapping(outputs[0], output);
}

Error PyTorchModelLoader::loadArange(const torch::jit::Node *ptNode) {

  glow::GlowIValue defaultStartVal = glow::GlowIValue();
  glow::GlowIValue defaultStepVal = glow::GlowIValue();
  glow::GlowIValue *startIVal = &defaultStartVal;
  glow::GlowIValue *endIVal;
  glow::GlowIValue *stepIVal = &defaultStepVal;

  startIVal->fromInt(0);
  stepIVal->fromInt(1);

  ASSIGN_VALUE_OR_RETURN_ERR(endIVal,
                             getGlowIValueForValue(ptNode->namedInput("end")));
  if (ptNode->hasNamedInput("start")) {
    ASSIGN_VALUE_OR_RETURN_ERR(
        startIVal, getGlowIValueForValue(ptNode->namedInput("start")));
  }
  if (ptNode->hasNamedInput("step")) {
    ASSIGN_VALUE_OR_RETURN_ERR(
        stepIVal, getGlowIValueForValue(ptNode->namedInput("step")));
  }

  // If any of the input values are doubles, the outputs must also be.
  if (startIVal->isDouble() || stepIVal->isDouble() || endIVal->isDouble()) {
    float start;
    float end;
    float step;
    ASSIGN_VALUE_OR_RETURN_ERR(
        start, startIVal->isDouble()
                   ? to32Bit(startIVal->toDouble())
                   : static_cast_expected<float>(startIVal->toInt()));
    ASSIGN_VALUE_OR_RETURN_ERR(
        end, endIVal->isDouble()
                 ? to32Bit(endIVal->toDouble())
                 : static_cast_expected<float>(endIVal->toInt()));
    ASSIGN_VALUE_OR_RETURN_ERR(
        step, stepIVal->isDouble()
                  ? to32Bit(stepIVal->toDouble())
                  : static_cast_expected<float>(stepIVal->toInt()));
    std::vector<float> outputValues;
    auto span = std::abs(end - start);
    for (float offset = 0.0; std::abs(offset) < span; offset += step) {
      outputValues.push_back(start + offset);
    }
    auto type = F_.getParent()->uniqueType(glow::ElemKind::FloatTy,
                                           outputValues.size());
    auto outputTensor = glow::Tensor(outputValues.data(), type);
    auto output = F_.getParent()->createConstant("Arange_output",
                                                 std::move(outputTensor));
    output->ensureIsOwned(); // Prevents heap use after free
    RETURN_ERR(addValueMapping(ptNode->output(), output));
  } else {
    int64_t start;
    int64_t end;
    int64_t step;
    ASSIGN_VALUE_OR_RETURN_ERR(
        start, startIVal->isInt()
                   ? startIVal->toInt()
                   : static_cast_expected<int64_t>(startIVal->toDouble()));
    ASSIGN_VALUE_OR_RETURN_ERR(
        end, endIVal->isInt()
                 ? endIVal->toInt()
                 : static_cast_expected<int64_t>(endIVal->toDouble()));
    ASSIGN_VALUE_OR_RETURN_ERR(
        step, stepIVal->isInt()
                  ? stepIVal->toInt()
                  : static_cast_expected<int64_t>(stepIVal->toDouble()));
    std::vector<int64_t> outputValues;
    auto span = std::abs(end - start);
    for (int64_t offset = 0; std::abs(offset) < span; offset += step) {
      outputValues.push_back(start + offset);
    }
    auto type = F_.getParent()->uniqueType(glow::ElemKind::Int64ITy,
                                           outputValues.size());
    auto outputTensor = glow::Tensor(outputValues.data(), type);
    auto output = F_.getParent()->createConstant("Arange_output",
                                                 std::move(outputTensor));
    output->ensureIsOwned(); // Prevents heap use after free
    RETURN_ERR(addValueMapping(ptNode->output(), output));
  }
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
  RETURN_ERR(addValueMapping(
      outputs[0], F_.createReshape("reshape", input, castVector<dim_t>(shape)),
      dtype));
}

Error PyTorchModelLoader::loadUpsampleNearest(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(input, getGlowNodeValueForValue(inputs[0]));

  // dimSize = 4 for Upsample 2D and dimSize = 5 for Upsample 3D
  auto dimSize = input.dims().size();
  RETURN_ERR_IF_NOT(dimSize == 4 || dimSize == 5,
                    "Expecting 4D or 5D input Tensor");

  // inputs can be (inputTensor, outputSize, outputScale) or (inputTensor,
  // outputSize, outputScale_d (for 3D), outputScale_w, outputScale_h)
  RETURN_ERR_IF_NOT(inputs.size() == 3 || inputs.size() == dimSize,
                    glow::strFormat("Expected 3 or %zu arguments.  Got %zu.",
                                    dimSize, inputs.size()));
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, inputs.size(), outputs, 1));

  std::vector<int64_t> outputSizeBuf;
  std::vector<int64_t> *outputSize;
  glow::GlowIValue *outputSizeIValue;
  ASSIGN_VALUE_OR_RETURN_ERR(outputSizeIValue,
                             getGlowIValueForValue(inputs[1]));

  // outputSize is not specified then we should read outputScale instead
  if (!outputSizeIValue->isNone()) {
    // Explicit output size in upsample call.
    ASSIGN_VALUE_OR_RETURN_ERR(outputSize,
                               iValToIntList(getGlowIValueForValue(inputs[1])));
    RETURN_ERR_IF_NOT(
        (*outputSize).size() == dimSize - 2,
        glow::strFormat("Expecting %zuD output size", dimSize - 2));
  } else {
    // Node specifies scale factor.  Compute output size.
    std::vector<double> *scaleFactors;

    // outputScale is a tuple
    if (inputs.size() == 3) {
      ASSIGN_VALUE_OR_RETURN_ERR(
          scaleFactors, iValToDoubleList(getGlowIValueForValue(inputs[2])));
      RETURN_ERR_IF_NOT(scaleFactors->size() == dimSize - 2,
                        glow::strFormat("Expected %zu scale factors.  Got %zu.",
                                        dimSize - 2, scaleFactors->size()));
      for (int i = 0; i < dimSize - 2; ++i) {
        outputSizeBuf.push_back(input.dims()[i + 2] * scaleFactors->at(i));
      }
    } else { // outputScale is a separate value for each dim
      double scaleFactor;
      for (int i = 2; i < dimSize; i++) {
        ASSIGN_VALUE_OR_RETURN_ERR(
            scaleFactor, iValToDouble(getGlowIValueForValue(inputs[2])));
        outputSizeBuf.push_back(input.dims()[i] * scaleFactor);
      }
    }

    outputSize = &outputSizeBuf;
  }

  TypeRef outTy = nullptr;
  std::string name;

  // Upsample 2D
  if (dimSize == 5) {
    name = "upsample_nearest3d";
    dim_t ia = input.dims()[0];
    dim_t ib = input.dims()[1];
    dim_t ox = (dim_t)(*outputSize)[0];
    dim_t oy = (dim_t)(*outputSize)[1];
    dim_t oz = (dim_t)(*outputSize)[2];
    outTy = F_.getParent()->uniqueTypeWithNewShape(input.getType(),
                                                   {ia, ib, ox, oy, oz});
  } else { // Upsample 2D
    name = "upsample_nearest2d";
    dim_t iN = input.dims()[0];
    dim_t iC = input.dims()[1];
    dim_t oH = (dim_t)(*outputSize)[0];
    dim_t oW = (dim_t)(*outputSize)[1];

    outTy = F_.getParent()->uniqueTypeWithNewShape(input.getType(),
                                                   {iN, iC, oH, oW});
  }
  c10::ScalarType dtype;
  RETURN_IF_ERR(getCorrectTypeMapping(dtype, inputs[0]));
  RETURN_ERR(addValueMapping(
      outputs[0], F_.createResizeNearest(name, input, outTy), dtype));
}

Error PyTorchModelLoader::loadView(const torch::jit::Node *ptNode) {
  // loadView is just like Reshape, except reshape should call contiguous
  // for non-contiguous data and view should fail
  return PyTorchModelLoader::loadReshape(ptNode);
}

Error PyTorchModelLoader::loadFloor(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 1, outputs, 1));

  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(input, getGlowNodeValueForValue(inputs[0]));

  auto *glowNode = F_.createFloor("floor", input);
  c10::ScalarType dtype;
  RETURN_IF_ERR(getCorrectTypeMapping(dtype, inputs[0]));
  RETURN_ERR(addValueMapping(outputs[0], glowNode->getResult(), dtype));
}

Error PyTorchModelLoader::loadCeil(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 1, outputs, 1));

  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(input, getGlowNodeValueForValue(inputs[0]));

  auto *glowNode = F_.createCeil("ceil", input);
  c10::ScalarType dtype;
  RETURN_IF_ERR(getCorrectTypeMapping(dtype, inputs[0]));
  RETURN_ERR(addValueMapping(outputs[0], glowNode->getResult(), dtype));
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
  RETURN_ERR(addValueMapping(outputs[0], glowNode->getResult(), dtype));
}

Error PyTorchModelLoader::loadGelu(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 1, outputs, 1));

  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(input, getGlowNodeValueForValue(inputs[0]));

  auto output = F_.createGELU("gelu", input)->getNthResult(0);
  RETURN_ERR(addValueMapping(outputs[0], output));
}

Error PyTorchModelLoader::loadTanh(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 1, outputs, 1));

  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(input, getGlowNodeValueForValue(inputs[0]));

  glow::TanhNode *glowNode = F_.createTanh("tanh", input);
  RETURN_ERR(addValueMapping(outputs[0], glowNode->getResult()));
}

Error PyTorchModelLoader::loadExp(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 1, outputs, 1));

  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(input, getGlowNodeValueForValue(inputs[0]));

  glow::ExpNode *glowNode = F_.createExp("exp", input);
  RETURN_ERR(addValueMapping(outputs[0], glowNode->getResult()));
}

Error PyTorchModelLoader::loadPow(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 2, outputs, 1));

  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(input, getGlowNodeValueForValue(inputs[0]));

  glow::PowNode *PN = nullptr;
  if (hasGlowIValueForValue(inputs[1])) {
    float exponent;
    ASSIGN_VALUE_OR_RETURN_ERR(exponent,
                               iValToDouble(getGlowIValueForValue(inputs[1])));
    PN = F_.createPow("pow", input, exponent);
  } else {
    NodeValue expNV;
    ASSIGN_VALUE_OR_RETURN_ERR(expNV, getGlowNodeValueForValue(inputs[1]));
    PN = F_.createNodeWithBroadcast<PowNode>("pow", -1, input, expNV);
  }

  RETURN_ERR(addValueMapping(outputs[0], PN->getResult()));
}

Error PyTorchModelLoader::loadLogicalXor(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 2, outputs, 1));

  glow::NodeValue res;
  ASSIGN_VALUE_OR_RETURN_ERR(
      res, loadArithmeticNode<XorNode>("logical_xor", inputs[0], inputs[1]));

  RETURN_ERR(addValueMapping(outputs[0], res));
}

Error PyTorchModelLoader::loadLogicalOr(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 2, outputs, 1));

  glow::NodeValue res;
  ASSIGN_VALUE_OR_RETURN_ERR(
      res, loadArithmeticNode<OrNode>("logical_or", inputs[0], inputs[1]));

  RETURN_ERR(addValueMapping(outputs[0], res));
}

Error PyTorchModelLoader::loadLogicalAnd(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 2, outputs, 1));

  glow::NodeValue res;
  ASSIGN_VALUE_OR_RETURN_ERR(
      res, loadArithmeticNode<AndNode>("logical_and", inputs[0], inputs[1]));

  RETURN_ERR(addValueMapping(outputs[0], res));
}

Error PyTorchModelLoader::loadLogicalNot(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 1, outputs, 1));

  glow::NodeValue glowInput;
  ASSIGN_VALUE_OR_RETURN_ERR(glowInput, getGlowNodeValueForValue(inputs[0]));

  return addValueMapping(outputs[0], F_.createNot("logical_not", glowInput));
}

Error PyTorchModelLoader::loadSqrt(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 1, outputs, 1));

  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(input, getGlowNodeValueForValue(inputs[0]));

  glow::PowNode *glowNode = F_.createPow("sqrt", input, /*exp=*/0.5);
  RETURN_ERR(addValueMapping(outputs[0], glowNode->getResult()));
}

template <typename CmpType, bool invert>
Error PyTorchModelLoader::loadCmp(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  auto kind = ptNode->kind();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 2, outputs, 1));

  glow::NodeValue lhs;
  ASSIGN_VALUE_OR_RETURN_ERR(
      lhs, getGlowNodeValueForValue(inputs[CompareInputs::lhs]));

  glow::NodeValue rhs;
  ASSIGN_VALUE_OR_RETURN_ERR(
      rhs, loadNodeValueOrBroadcastedIValue(inputs[CompareInputs::rhs],
                                            lhs.getType()));

  constexpr int axis = -1;
  // Greater than and Greater or equal are mapped to inverted
  // (swap between LHS and RHS) Less or equal and Less than
  if constexpr (invert) {
    auto *glowNode = F_.createNodeWithBroadcast<CmpType>(kind.toUnqualString(),
                                                         axis, rhs, lhs);
    return addValueMapping(outputs[0], glowNode->getResult());
  } else {
    auto *glowNode = F_.createNodeWithBroadcast<CmpType>(kind.toUnqualString(),
                                                         axis, lhs, rhs);
    return addValueMapping(outputs[0], glowNode->getResult());
  }
}

Error PyTorchModelLoader::loadSigmoid(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 1, outputs, 1));

  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(input, getGlowNodeValueForValue(inputs[0]));

  glow::SigmoidNode *glowNode = F_.createSigmoid("sigmoid", input);
  c10::ScalarType dtype;
  RETURN_IF_ERR(getCorrectTypeMapping(dtype, inputs[0]));
  RETURN_ERR(addValueMapping(outputs[0], glowNode->getResult(), dtype));
}

Error PyTorchModelLoader::loadSilu(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 1, outputs, 1));

  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(input, getGlowNodeValueForValue(inputs[0]));

  glow::SwishNode *glowNode = F_.createSwish("swish", input);
  c10::ScalarType dtype;
  RETURN_IF_ERR(getCorrectTypeMapping(dtype, inputs[0]));
  RETURN_ERR(addValueMapping(outputs[0], glowNode->getResult(), dtype));
}

Error PyTorchModelLoader::loadReciprocal(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 1, outputs, 1));

  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(input, getGlowNodeValueForValue(inputs[0]));
  glow::PowNode *glowNode = F_.createPow("reciprocal", input, /*exp=*/-1);
  RETURN_ERR(addValueMapping(outputs[0], glowNode->getResult()));
}

Error PyTorchModelLoader::loadLSTM(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 9, outputs, 3));

  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(
      input, getGlowNodeValueForValue(inputs[LSTMInputs::input]));

  std::vector<glow::NodeValue> *hx;

  ASSIGN_VALUE_OR_RETURN_ERR(
      hx, iValToNodeValueList(getGlowIValueForValue(inputs[LSTMInputs::hx])));
  auto h03D = (*hx)[0];
  auto c03D = (*hx)[1];
  unsigned hiddenSize = h03D.dims()[2];

  bool hasBiases;
  ASSIGN_VALUE_OR_RETURN_ERR(hasBiases, iValToBool(getGlowIValueForValue(
                                            inputs[LSTMInputs::has_biases])));

  unsigned numLayers;
  ASSIGN_VALUE_OR_RETURN_ERR(numLayers, iValToInt(getGlowIValueForValue(
                                            inputs[LSTMInputs::num_layers])));
  RETURN_ERR_IF_NOT(numLayers == 1, "Stacked LSTM is not supported in Glow.");

  float dropout;
  ASSIGN_VALUE_OR_RETURN_ERR(dropout, iValToDouble(getGlowIValueForValue(
                                          inputs[LSTMInputs::dropout])));
  RETURN_ERR_IF_NOT(dropout == 0,
                    "Dropout is not allowed for inference in Glow.");

  bool train, bidirectional, batchFirst;
  ASSIGN_VALUE_OR_RETURN_ERR(
      train, iValToBool(getGlowIValueForValue(inputs[LSTMInputs::train])));
  ASSIGN_VALUE_OR_RETURN_ERR(
      bidirectional,
      iValToBool(getGlowIValueForValue(inputs[LSTMInputs::bidirectional])));
  ASSIGN_VALUE_OR_RETURN_ERR(batchFirst, iValToBool(getGlowIValueForValue(
                                             inputs[LSTMInputs::batch_first])));

  RETURN_ERR_IF_NOT(train == false,
                    "Training is not supported for LSTM in Glow.");

  RETURN_ERR_IF_NOT(batchFirst == false,
                    "batch_first is not supported for LSTM in Glow.");

  NodeValue hn, cn;
  std::vector<glow::NodeValue> *params;

  ASSIGN_VALUE_OR_RETURN_ERR(params, iValToNodeValueList(getGlowIValueForValue(
                                         inputs[LSTMInputs::params])));

  glow::dim_t paramsIdx = 0;
  auto inputElemKind = input.getType()->getElementType();
  auto getBias = [&](std::string constantName = "constant") {
    glow::Constant *res;
    if (hasBiases) {
      res = llvm::dyn_cast<glow::Constant>((*params)[paramsIdx++].getNode());
    } else {
      glow::Tensor t(inputElemKind, {4 * hiddenSize});
      t.zero();
      res = F_.getParent()->createConstant(constantName, std::move(t));
    }
    return res;
  };
  glow::Constant *Wx =
      llvm::dyn_cast<glow::Constant>((*params)[paramsIdx++].getNode());
  glow::Constant *Wh =
      llvm::dyn_cast<glow::Constant>((*params)[paramsIdx++].getNode());
  glow::Constant *Bx = getBias("Bx_Constant"), *Bh = getBias("Bh_Constant");

  NodeValue output;
  // W need to be transposed, in pt it is hiddenSize * inputSize,
  // in glow it is inputSize * hiddenSize.
  auto WxTransposed =
      F_.createTranspose("Wx_Transposed", Wx, {1, 0})->getResult();
  auto WhTransposed =
      F_.createTranspose("Wh_Transposed", Wh, {1, 0})->getResult();
  if (bidirectional) {
    hn = h03D;
    cn = c03D;
    glow::Constant *WxR =
        llvm::dyn_cast<glow::Constant>((*params)[paramsIdx++].getNode());
    glow::Constant *WhR =
        llvm::dyn_cast<glow::Constant>((*params)[paramsIdx++].getNode());
    glow::Constant *BxR = getBias("Bx_Reversed_Constant"),
                   *BhR = getBias("Bh_Reversed_Constant");

    // Same transpose for bidirectional LSTM's reversed weights
    auto WxRTransposed =
        F_.createTranspose("WxR_Transposed", WxR, {1, 0})->getResult();
    auto WhRTransposed =
        F_.createTranspose("WhR_Transposed", WhR, {1, 0})->getResult();
    F_.createPyTorchLSTM("lstm", input, WxTransposed, WhTransposed, Bx, Bh, hn,
                         cn, output, bidirectional, WxRTransposed,
                         WhRTransposed, BxR, BhR);
  } else {
    hn = F_.createReshape("reshape_H0", h03D, {h03D.dims()[1], h03D.dims()[2]})
             ->getResult();
    cn = F_.createReshape("reshape_C0", c03D, {c03D.dims()[1], c03D.dims()[2]})
             ->getResult();
    F_.createPyTorchLSTM("lstm", input, WxTransposed, WhTransposed, Bx, Bh, hn,
                         cn, output, bidirectional);
  }
  RETURN_IF_ERR(addValueMapping(outputs[0], output));
  RETURN_IF_ERR(addValueMapping(outputs[1], hn));
  RETURN_IF_ERR(addValueMapping(outputs[2], cn));
  return Error::success();
}

Error PyTorchModelLoader::loadConvolution(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, -12, outputs, 1));

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

  bool transposed;
  ASSIGN_VALUE_OR_RETURN_ERR(transposed, iValToBool(getGlowIValueForValue(
                                             inputs[ConvInputs::transposed])));

  bool isConv3d = input.dims().size() == 5;
  if (isConv3d) {
    input = F_.createTranspose("conv_input_transposed", input, NCTHW2NTHWC);
    weights =
        F_.createTranspose("conv_weights_transposed", weights, NCTHW2NTHWC);
  } else if (transposed) {
    input = F_.createTranspose("conv_input_transposed", input, NCHW2NHWC);
    weights = F_.createTranspose("conv_weights_transposed", weights, CNHW2NHWC);
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

  std::vector<glow::unsigned_t> dilations;
  if (isConv3d) {
    ASSIGN_VALUE_OR_RETURN_ERR(
        dilations,
        castVector<glow::unsigned_t>(expandIntIValIfNeeded(
            getGlowIValueForValue(inputs[ConvInputs::dilation]), 3)));

    // Currently conv3d doesn't support dilation
    RETURN_ERR_IF_NOT(std::all_of(dilations.cbegin(), dilations.cend(),
                                  [](unsigned_t i) { return i == 1; }),
                      "Dilation not supported for conv3d");
  } else {
    ASSIGN_VALUE_OR_RETURN_ERR(
        dilations,
        castVector<glow::unsigned_t>(expandIntIValIfNeeded(
            getGlowIValueForValue(inputs[ConvInputs::dilation]), 2)));
  }

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
        inputShape.h, inputShape.w, kernels, strides, pads, dilations);
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
  } else if (transposed) {
    glow::ConvTransposeNode *convTranspose =
        F_.createConvTranspose("convTranspose", input, weights, bias, outTy,
                               kernels, strides, pads, groups);
    output = F_.createTranspose("convTranpose_output_transposed",
                                convTranspose->getResult(), NHWC2NCHW);
  } else {
    glow::ConvolutionNode *conv =
        F_.createConv("conv", input, weights, bias, outTy, kernels, strides,
                      pads, groups, dilations);
    output = F_.createTranspose("conv_output_transposed", conv->getResult(),
                                NHWC2NCHW);
  }
  RETURN_ERR(addValueMapping(outputs[0], output->getResult()));
}

Error PyTorchModelLoader::loadConv2D(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 7, outputs, 1));

  // Glow expects conv inputs to be in NHWC but PyTorch keeps them in NCHW so
  // we transpose them.
  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(
      input, getGlowNodeValueForValue(inputs[Conv2DInputs::input]));

  // Glow expects conv weights to be in CRSK but PyTorch keeps them in CKRS
  // so we transpose them. C - output_depth, R - filter_height, S -
  // filter_width, K - input_depth.
  glow::NodeValue weights;
  ASSIGN_VALUE_OR_RETURN_ERR(
      weights, getGlowNodeValueForValue(inputs[Conv2DInputs::weights]));

  RETURN_ERR_IF_NOT(input.dims().size() == 4 &&
                        input.dims().size() == weights.dims().size(),
                    "Expect 4 dims in input and weights for conv2d");

  input = F_.createTranspose("conv_input_transposed", input, NCHW2NHWC);
  weights = F_.createTranspose("conv_weights_transposed", weights, NCHW2NHWC);

  // If a bias was provided then use it otherwise create a 0 bias.
  glow::dim_t biasDim = weights.dims()[0];
  glow::NodeValue bias = loadNodeValueOrCreateBroadcastedConstant(
      inputs[Conv2DInputs::bias], "conv_bias",
      glow::Type(ElemKind::FloatTy, {biasDim}), 0);

  std::vector<glow::unsigned_t> strides;
  ASSIGN_VALUE_OR_RETURN_ERR(
      strides, castVector<glow::unsigned_t>(expandIntIValIfNeeded(
                   getGlowIValueForValue(inputs[Conv2DInputs::stride]),
                   input.dims().size() - 2)));

  std::vector<glow::unsigned_t> pads;
  glow::unsigned_t pad;
  ASSIGN_VALUE_OR_RETURN_ERR(
      pad, static_cast_expected<glow::unsigned_t>(contractIntIValIfNeeded(
               getGlowIValueForValue(inputs[Conv2DInputs::padding]))));
  pads = {pad, pad, pad, pad};

  std::vector<glow::unsigned_t> dilations;
  ASSIGN_VALUE_OR_RETURN_ERR(
      dilations,
      castVector<glow::unsigned_t>(expandIntIValIfNeeded(
          getGlowIValueForValue(inputs[Conv2DInputs::dilation]), 2)));

  glow::unsigned_t groups;
  ASSIGN_VALUE_OR_RETURN_ERR(
      groups, static_cast_expected<glow::unsigned_t>(iValToInt(
                  getGlowIValueForValue(inputs[Conv2DInputs::groups]))));
  std::vector<glow::unsigned_t> kernels;
  glow::ShapeNHWC weightsShape(weights.dims());
  kernels = {static_cast<glow::unsigned_t>(weightsShape.h),
             static_cast<glow::unsigned_t>(weightsShape.w)};

  glow::TypeRef outTy;
  glow::ShapeNHWC inputShape(input.dims());
  auto outSz = glow::calculateConvPoolOutputDims(
      inputShape.h, inputShape.w, kernels, strides, pads, dilations);
  std::array<glow::dim_t, 4> outDims = {
      {input.dims()[0], outSz.first, outSz.second, weights.dims()[0]}};
  outTy = F_.getParent()->uniqueType(glow::ElemKind::FloatTy, outDims);

  glow::ConvolutionNode *conv =
      F_.createConv("conv", input, weights, bias, outTy, kernels, strides, pads,
                    groups, dilations);
  glow::TransposeNode *output = F_.createTranspose(
      "conv_output_transposed", conv->getResult(), NHWC2NCHW);
  RETURN_ERR(addValueMapping(outputs[0], output->getResult()));
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

  RETURN_ERR(addValueMapping(outputs[0], output));
}

Error PyTorchModelLoader::loadBatchNorm(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();

  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(
      input, getGlowNodeValueForValue(inputs[BatchNormInputs::input]));

  int numDims = input.dims().size() - 2;
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 9, outputs, 1));

  bool training;
  ASSIGN_VALUE_OR_RETURN_ERR(training, iValToBool(getGlowIValueForValue(
                                           inputs[BatchNormInputs::training])));
  RETURN_ERR_IF_NOT(training == false, "Don't support BatchNorm training yet.");

  RETURN_ERR_IF_NOT(
      numDims >= 0 && numDims <= 3,
      glow::strFormat("Only support 0D, 1D, 2D or 3D got %dD", numDims + 2));

  size_t numChannels = input.dims()[1];
  glow::NodeValue weights = loadNodeValueOrCreateBroadcastedConstant(
      inputs[BatchNormInputs::weights], "weight",
      glow::Type(ElemKind::FloatTy, {numChannels}), 1.0);
  glow::Constant *weightsC = llvm::dyn_cast<glow::Constant>(weights.getNode());

  glow::NodeValue bias = loadNodeValueOrCreateBroadcastedConstant(
      inputs[BatchNormInputs::bias], "bias",
      glow::Type(ElemKind::FloatTy, {numChannels}), 0.0);
  glow::Constant *biasC = llvm::dyn_cast<glow::Constant>(bias.getNode());

  glow::NodeValue mean;
  ASSIGN_VALUE_OR_RETURN_ERR(
      mean, getGlowNodeValueForValue(inputs[BatchNormInputs::running_mean]));
  glow::Constant *meanC = llvm::dyn_cast<glow::Constant>(mean.getNode());

  glow::NodeValue var;
  ASSIGN_VALUE_OR_RETURN_ERR(
      var, getGlowNodeValueForValue(inputs[BatchNormInputs::running_var]));
  glow::Constant *varC = llvm::dyn_cast<glow::Constant>(var.getNode());

  float epsilon;
  ASSIGN_VALUE_OR_RETURN_ERR(
      epsilon, to32Bit(iValToDouble(
                   getGlowIValueForValue(inputs[BatchNormInputs::eps]))));

  float momentum;
  ASSIGN_VALUE_OR_RETURN_ERR(
      momentum, to32Bit(iValToDouble(
                    getGlowIValueForValue(inputs[BatchNormInputs::momentum]))));

  // Input is in NCHW.
  glow::unsigned_t channelIdx = 1;
  glow::NodeValue output;
  // 0D. Currently NNPI only supports 2D, will remove this after it supports 0D.
  if (numDims == 0) {
    glow::ReshapeNode *twoD = F_.createReshape(
        "bn_NC2NCHW", input, {input.dims()[0], input.dims()[1], 1, 1});
    glow::BatchNormalizationNode *bn = F_.createBatchNormalization(
        "bn", twoD->getType(0), twoD, biasC, weightsC, meanC, varC, channelIdx,
        epsilon, momentum);
    glow::ReshapeNode *zeroD = F_.createReshape("bn_NCHW2NC", bn, input.dims());
    output = zeroD->getResult();
  } else { // 1D, 2D or 3D
    glow::BatchNormalizationNode *bn = F_.createBatchNormalization(
        "batchnorm", input.getType(), input, biasC, weightsC, meanC, varC,
        channelIdx, epsilon, momentum);
    output = bn->getResult();
  }

  c10::ScalarType dtype;
  RETURN_IF_ERR(getCorrectTypeMapping(dtype, inputs[0]));
  RETURN_ERR(addValueMapping(outputs[0], output, dtype));
}

Expected<NodeValue>
PyTorchModelLoader::loadQuantizedBatchNormImpl(const torch::jit::Node *ptNode,
                                               int numDims) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();

  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(
      input, getGlowNodeValueForValue(inputs[QuantizedBatchNormInputs::input]));

  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 8, outputs, 1));

  RETURN_ERR_IF_NOT(
      input.dims().size() == numDims + 2,
      glow::strFormat("Number input dimensions must be equal to %d, got %lu",
                      numDims + 2, input.dims().size()));

  size_t numChannels = input.dims()[1];

  glow::NodeValue weights = loadNodeValueOrCreateBroadcastedConstant(
      inputs[QuantizedBatchNormInputs::weights], "weight",
      glow::Type(ElemKind::FloatTy, {numChannels}), 1.0);
  glow::Constant *weightsC = llvm::dyn_cast<glow::Constant>(weights.getNode());

  glow::NodeValue bias = loadNodeValueOrCreateBroadcastedConstant(
      inputs[QuantizedBatchNormInputs::bias], "bias",
      glow::Type(ElemKind::FloatTy, {numChannels}), 0.0);
  glow::Constant *biasC = llvm::dyn_cast<glow::Constant>(bias.getNode());

  glow::NodeValue mean;
  ASSIGN_VALUE_OR_RETURN_ERR(
      mean,
      getGlowNodeValueForValue(inputs[QuantizedBatchNormInputs::running_mean]));
  glow::Constant *meanC = llvm::dyn_cast<glow::Constant>(mean.getNode());

  glow::NodeValue var;
  ASSIGN_VALUE_OR_RETURN_ERR(
      var,
      getGlowNodeValueForValue(inputs[QuantizedBatchNormInputs::running_var]));
  glow::Constant *varC = llvm::dyn_cast<glow::Constant>(var.getNode());

  float epsilon;
  ASSIGN_VALUE_OR_RETURN_ERR(epsilon,
                             to32Bit(iValToDouble(getGlowIValueForValue(
                                 inputs[QuantizedBatchNormInputs::eps]))));

  float momentum = 0.1;
  float output_scale;
  int32_t output_zero_point;
  ASSIGN_VALUE_OR_RETURN_ERR(
      output_scale, to32Bit(iValToDouble(getGlowIValueForValue(
                        inputs[QuantizedBatchNormInputs::output_scale]))));

  ASSIGN_VALUE_OR_RETURN_ERR(
      output_zero_point,
      iValToInt(getGlowIValueForValue(
          inputs[QuantizedBatchNormInputs::output_zero_point])));

  // Input is in NCHW.
  glow::unsigned_t channelIdx = 1;
  std::string opName;
  if (numDims == 3) {
    opName = "bn3d_quant";
  } else {
    opName = "bn2d_quant";
  }

  glow::BatchNormalizationNode *bn = F_.createBatchNormalization(
      opName, input.getType(), input, biasC, weightsC, meanC, varC, channelIdx,
      epsilon, momentum);
  return Expected<NodeValue>(bn->getResult());
}

Error PyTorchModelLoader::loadQuantizedBatchNorm2d(
    const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  glow::NodeValue output;
  ASSIGN_VALUE_OR_RETURN_ERR(output, loadQuantizedBatchNormImpl(ptNode, 2));

  c10::ScalarType dtype;
  RETURN_IF_ERR(getCorrectTypeMapping(dtype, inputs[0]));
  RETURN_ERR(addValueMapping(outputs[0], output, dtype));
}

Error PyTorchModelLoader::loadQuantizedBatchNorm3d(
    const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  glow::NodeValue output;
  ASSIGN_VALUE_OR_RETURN_ERR(output, loadQuantizedBatchNormImpl(ptNode, 3));

  c10::ScalarType dtype;
  RETURN_IF_ERR(getCorrectTypeMapping(dtype, inputs[0]));
  RETURN_ERR(addValueMapping(outputs[0], output, dtype));
}

Error PyTorchModelLoader::loadQuantizedBatchNorm3dRelu(
    const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  glow::NodeValue output;
  ASSIGN_VALUE_OR_RETURN_ERR(output, loadQuantizedBatchNormImpl(ptNode, 3));

  c10::ScalarType dtype;
  RETURN_IF_ERR(getCorrectTypeMapping(dtype, inputs[0]));
  output = F_.createRELU("quantized_relu_after_bn", output);
  RETURN_ERR(addValueMapping(outputs[0], output, dtype));
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
  RETURN_ERR(addValueMapping(outputs[0], input));
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
                                       outOffset - UINT8_TO_INT8_SHIFT);
    dtype = c10::ScalarType::QUInt8;

  } else if (outDtype == (int32_t)at::ScalarType::QInt8) {
    outTy = F_.getParent()->uniqueType(ElemKind::Int8QTy, outDims, outScale,
                                       outOffset);
    dtype = c10::ScalarType::QInt8;
  } else {
    return MAKE_ERR("Quantize only supports QUInt8 and QInt8");
  }
  glow::QuantizeNode *qn = F_.createQuantize("quantize", input, outTy);

  RETURN_ERR(addValueMapping(outputs[0], qn->getResult(), dtype));
}

Error PyTorchModelLoader::loadDequantize(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 1, outputs, 1));

  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(input, getGlowNodeValueForValue(inputs[0]));

  glow::DequantizeNode *dn =
      F_.createDequantize("dequantize", input, ElemKind::FloatTy);

  RETURN_ERR(addValueMapping(outputs[0], dn->getResult()));
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
  RETURN_ERR(addValueMapping(outputs[0], output, dtype));
}

Error PyTorchModelLoader::loadQuantizedConv(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  glow::NodeValue output;
  ASSIGN_VALUE_OR_RETURN_ERR(output,
                             loadQuantizedConvImpl(ptNode, false /* isRelu */));

  c10::ScalarType dtype;
  RETURN_IF_ERR(getCorrectTypeMapping(dtype, inputs[0]));
  RETURN_ERR(addValueMapping(outputs[0], output, dtype));
}

Error PyTorchModelLoader::loadQuantizedConvUnpacked(
    const torch::jit::Node *ptNode) {
  return loadQuantizedConvUnpackedImpl(ptNode, /*isRelu*/ false);
}

Error PyTorchModelLoader::loadQuantizedConvReluUnpacked(
    const torch::jit::Node *ptNode) {
  return loadQuantizedConvUnpackedImpl(ptNode, /*isRelu*/ true);
}

Error PyTorchModelLoader::loadQuantizedConvUnpackedImpl(
    const torch::jit::Node *ptNode, bool isRelu) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 9, outputs, 1));

  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(
      input,
      getGlowNodeValueForValue(inputs[QuantizedUnpackedConvInputs::input]));

  bool isConv3d = input.dims().size() == 5;
  if (isConv3d) {
    input = F_.createTranspose("qconv_input_transposed", input, NCTHW2NTHWC);
  } else {
    input = F_.createTranspose("qconv_input_transposed", input, NCHW2NHWC);
  }

  glow::NodeValue weights;
  ASSIGN_VALUE_OR_RETURN_ERR(
      weights,
      getGlowNodeValueForValue(inputs[QuantizedUnpackedConvInputs::weights]));
  if (isConv3d) {
    weights =
        F_.createTranspose("qconv_weights_transposed", weights, NCTHW2NTHWC);
  } else {
    weights =
        F_.createTranspose("qconv_weights_transposed", weights, NCHW2NHWC);
  }

  glow::NodeValue bias = loadNodeValueOrCreateBroadcastedConstant(
      inputs[QuantizedUnpackedConvInputs::bias], "qconv_bias",
      glow::Type(ElemKind::FloatTy, {weights.dims()[0]}), 0.0);

  std::vector<glow::unsigned_t> strides;
  ASSIGN_VALUE_OR_RETURN_ERR(
      strides,
      castVector<glow::unsigned_t>(expandIntIValIfNeeded(
          getGlowIValueForValue(inputs[QuantizedUnpackedConvInputs::stride]),
          input.dims().size() - 2)));

  // pads
  std::vector<glow::unsigned_t> pads;
  if (isConv3d) {
    std::vector<glow::unsigned_t> pad;

    ASSIGN_VALUE_OR_RETURN_ERR(
        pad,
        castVector<glow::unsigned_t>(expandIntIValIfNeeded(
            getGlowIValueForValue(inputs[QuantizedUnpackedConvInputs::padding]),
            3)));
    pads = {pad[0], pad[0], pad[1], pad[1], pad[2], pad[2]};
  } else {
    glow::unsigned_t pad;
    ASSIGN_VALUE_OR_RETURN_ERR(
        pad, static_cast_expected<glow::unsigned_t>(
                 contractIntIValIfNeeded(getGlowIValueForValue(
                     inputs[QuantizedUnpackedConvInputs::padding]))));
    pads = {pad, pad, pad, pad};
  }

  glow::unsigned_t groups;
  ASSIGN_VALUE_OR_RETURN_ERR(
      groups,
      static_cast_expected<glow::unsigned_t>(iValToInt(
          getGlowIValueForValue(inputs[QuantizedUnpackedConvInputs::group]))));

  float outScale;
  ASSIGN_VALUE_OR_RETURN_ERR(outScale,
                             iValToDouble(getGlowIValueForValue(
                                 inputs[QuantizedUnpackedConvInputs::scale])));

  int32_t outOffset;
  ASSIGN_VALUE_OR_RETURN_ERR(
      outOffset, iValToInt(getGlowIValueForValue(
                     inputs[QuantizedUnpackedConvInputs::zero_point])));

  // calc output type
  std::vector<glow::unsigned_t> dilations;
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
    outTy =
        F_.getParent()->uniqueType(glow::ElemKind::Int8QTy, outDims, outScale,
                                   outOffset - UINT8_TO_INT8_SHIFT);
  } else {
    glow::ShapeNHWC inputShape(input.dims());
    glow::ShapeNHWC weightShape(weights.dims());
    kernels = {static_cast<glow::unsigned_t>(weightShape.h),
               static_cast<glow::unsigned_t>(weightShape.w)};
    ASSIGN_VALUE_OR_RETURN_ERR(
        dilations, castVector<glow::unsigned_t>(expandIntIValIfNeeded(
                       getGlowIValueForValue(
                           inputs[QuantizedUnpackedConvInputs::dilation]),
                       2)));
    auto outSz = glow::calculateConvPoolOutputDims(
        inputShape.h, inputShape.w, kernels, strides, pads, dilations);
    std::array<glow::dim_t, 4> outDims = {
        {input.dims()[0], outSz.first, outSz.second, weightShape.n}};
    outTy =
        F_.getParent()->uniqueType(glow::ElemKind::Int8QTy, outDims, outScale,
                                   outOffset - UINT8_TO_INT8_SHIFT);
  }

  auto ptWeightTensor =
      qparamsMap_.at(inputs[QuantizedUnpackedConvInputs::weights]).toTensor();

  bool isPerChannelQuantized =
      ptWeightTensor.is_quantized() &&
      ptWeightTensor.qscheme() == at::kPerChannelAffine;

  glow::NodeValue output;

  if (isPerChannelQuantized) {
    NodeValue wScales, wOffsets;
    std::tie(wScales, wOffsets) = extractChannelwiseQParams(F_, ptWeightTensor);

    // Quantize the filter automatically (only if it is float). The bias is
    // NOT quantized automatically and is left at the disposal of each Backend
    // to quantize it later using custom logic.
    output = F_.createChannelwiseQuantizedConv(
                   "qconv_channelwise", input, weights, bias, wScales, wOffsets,
                   /* biasScales */ nullptr,
                   /* biasOffsets */ nullptr, outTy, kernels, strides, pads,
                   groups, dilations, /* quantizeFilter */ true,
                   /* quantizeBias */ false)
                 ->getResult();
  } else if (isConv3d) {
    output = F_.createConv3D("qconv", input, weights, bias, outTy, kernels,
                             strides, pads, groups)
                 ->getResult();
  } else {

    output = F_.createConv("qconv", input, weights, bias, outTy, kernels,
                           strides, pads, groups, dilations)
                 ->getResult();
  }

  if (isRelu) {
    output = F_.createRELU("qconv_relu", output)->getResult();
  }

  if (isConv3d) {
    output = F_.createTranspose("qconv_output_transposed", output, NTHWC2NCTHW)
                 ->getResult();
  } else {
    output = F_.createTranspose("qconv_output_transposed", output, NHWC2NCHW)
                 ->getResult();
  }

  c10::ScalarType dtype;
  RETURN_IF_ERR(getCorrectTypeMapping(dtype, inputs[0]));
  RETURN_ERR(addValueMapping(outputs[0], output, dtype));
}

Error PyTorchModelLoader::loadMaxPool2d(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 6, outputs, -1));

  bool withIndices = false;
  if (outputs.size() == 2) {
    withIndices = true;
  }

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

  bool ceilMode;
  ASSIGN_VALUE_OR_RETURN_ERR(ceilMode, iValToBool(getGlowIValueForValue(
                                           inputs[MaxPoolInputs::ceil_mode])));
  // For ceil mode, we add pads at the end of H and W,
  // to make the output size is effectively rounded up.
  if (ceilMode) {
    pads[2] += strides[0] - 1;
    pads[3] += strides[1] - 1;
  }

  glow::MaxPoolNode *mp =
      F_.createMaxPool("maxpool2d", input, kernels, strides, pads,
                       /* elemTyAMT */ ElemKind::Int64ITy, /* layout */ NHWC,
                       /* flattenIndices */ !withIndices);
  glow::NodeValue output = mp->getResult();
  output = F_.createTranspose("maxpool2d_output_transposed", output, NHWC2NCHW);

  TransposeNode *transposeNode = nullptr;
  if (withIndices) {
    transposeNode = F_.createTranspose("maxpool2d_indices_transposed",
                                       mp->getArgmax(), NHWC2NCHW);
    RETURN_IF_ERR(addValueMapping(outputs[1], transposeNode));
  }

  c10::ScalarType dtype;
  RETURN_IF_ERR(getCorrectTypeMapping(dtype, inputs[MaxPoolInputs::input]));
  RETURN_ERR(addValueMapping(outputs[0], output, dtype));
}

Expected<NodeValue>
PyTorchModelLoader::loadAvgPoolImpl(const torch::jit::Node *ptNode,
                                    int numDims) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 7, outputs, 1));

  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(
      input, getGlowNodeValueForValue(inputs[AvgPoolInputs::input]));
  bool is3d = (numDims == 3);
  std::string opName = is3d ? "avgpool3d" : "avgpool2d";

  if (is3d) {
    input =
        F_.createTranspose(opName + "_input_transposed", input, NCTHW2NTHWC);
  } else {
    input = F_.createTranspose(opName + "_input_transposed", input, NCHW2NHWC);
  }

  std::vector<glow::unsigned_t> kernels;
  ASSIGN_VALUE_OR_RETURN_ERR(
      kernels,
      castVector<glow::unsigned_t>(expandIntIValIfNeeded(
          getGlowIValueForValue(inputs[AvgPoolInputs::kernel_size]), numDims)));

  std::vector<glow::unsigned_t> padsPair;
  ASSIGN_VALUE_OR_RETURN_ERR(
      padsPair,
      castVector<glow::unsigned_t>(expandIntIValIfNeeded(
          getGlowIValueForValue(inputs[AvgPoolInputs::padding]), numDims)));
  RETURN_ERR_IF_NOT(padsPair.size() == numDims,
                    "Number of pad values is incorrect");
  std::vector<glow::unsigned_t> pads;
  if (is3d) {
    pads = {padsPair[0], padsPair[1], padsPair[2],
            padsPair[0], padsPair[1], padsPair[2]};
  } else {
    pads = {padsPair[0], padsPair[1], padsPair[0], padsPair[1]};
  }

  // Stride defaults to kernel_size.
  std::vector<glow::unsigned_t> strides;
  if (hasGlowIValueForValue(inputs[AvgPoolInputs::stride])) {
    ASSIGN_VALUE_OR_RETURN_ERR(
        strides,
        castVector<glow::unsigned_t>(expandIntIValIfNeeded(
            getGlowIValueForValue(inputs[AvgPoolInputs::stride]), numDims)));
  } else {
    strides = kernels;
  }

  // Glow doesn't support avgpool ceil mode.
  bool ceilMode;
  ASSIGN_VALUE_OR_RETURN_ERR(ceilMode, iValToBool(getGlowIValueForValue(
                                           inputs[AvgPoolInputs::ceil_mode])));
  RETURN_ERR_IF_NOT(ceilMode == false,
                    "ceilMode must be scalar with false value.");

  // CountIncludePad defaults to true.
  bool countIncludePads = true;
  if (hasGlowIValueForValue(inputs[AvgPoolInputs::count_include_pad])) {
    ASSIGN_VALUE_OR_RETURN_ERR(countIncludePads,
                               iValToBool(getGlowIValueForValue(
                                   inputs[AvgPoolInputs::count_include_pad])));
  }

  glow::AvgPoolNode *ap =
      F_.createAvgPool(opName, input, kernels, strides, pads,
                       (is3d ? NTHWC : NHWC), countIncludePads);
  glow::NodeValue ap_output = ap->getResult();
  const glow::TransposeNode *output;

  if (is3d) {
    output = F_.createTranspose(opName + "_output_transposed", ap_output,
                                NTHWC2NCTHW);
  } else {
    output =
        F_.createTranspose(opName + "_output_transposed", ap_output, NHWC2NCHW);
  }

  return Expected<NodeValue>(output->getResult());
}

Error PyTorchModelLoader::loadAvgPool2d(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  glow::NodeValue output;
  ASSIGN_VALUE_OR_RETURN_ERR(output, loadAvgPoolImpl(ptNode, 2 /* numDims */));

  c10::ScalarType dtype;
  RETURN_IF_ERR(getCorrectTypeMapping(dtype, inputs[0]));
  RETURN_ERR(addValueMapping(outputs[0], output, dtype));
}

Error PyTorchModelLoader::loadAvgPool3d(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  glow::NodeValue output;
  ASSIGN_VALUE_OR_RETURN_ERR(output, loadAvgPoolImpl(ptNode, 3 /* numDims */));

  c10::ScalarType dtype;
  RETURN_IF_ERR(getCorrectTypeMapping(dtype, inputs[0]));
  RETURN_ERR(addValueMapping(outputs[0], output, dtype));
}

Error PyTorchModelLoader::loadClamp(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 3, outputs, 1));

  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(
      input, getGlowNodeValueForValue(inputs[ClampInputs::input]));

  double minDouble = 0;
  float min = 0;
  NodeValue minSN = nullptr;

  if (hasGlowIValueForValue(inputs[ClampInputs::min], true)) {
    ASSIGN_VALUE_OR_RETURN_ERR(minDouble, iValToDouble(getGlowIValueForValue(
                                              inputs[ClampInputs::min])));
    ASSIGN_VALUE_OR_RETURN_ERR(min, to32Bit(minDouble));
    minSN = F_.createSplat("minValue", input.getType(), min);
  }

  double maxDouble = 0;
  float max = 0;
  NodeValue maxSN = nullptr;
  if (hasGlowIValueForValue(inputs[ClampInputs::max], true)) {
    ASSIGN_VALUE_OR_RETURN_ERR(maxDouble, iValToDouble(getGlowIValueForValue(
                                              inputs[ClampInputs::max])));
    ASSIGN_VALUE_OR_RETURN_ERR(max, to32Bit(maxDouble));
    maxSN = F_.createSplat("maxValue", input.getType(), max);
  }

  NodeValue output = input;
  if (minSN) {
    output = F_.createMax("Clamp_min", output, minSN);
  }
  if (maxSN) {
    output = F_.createMin("Clamp_max", output, maxSN);
  }
  RETURN_ERR_IF_NOT(output, "Failed to load aten::clamp");
  RETURN_ERR(addValueMapping(outputs[0], output));
}

Error PyTorchModelLoader::loadClampMin(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 2, outputs, 1));

  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(
      input, getGlowNodeValueForValue(inputs[ClampMinInputs::input]));

  double minDouble;
  ASSIGN_VALUE_OR_RETURN_ERR(minDouble, iValToDouble(getGlowIValueForValue(
                                            inputs[ClampMinInputs::min])));
  float min;
  ASSIGN_VALUE_OR_RETURN_ERR(min, to32Bit(minDouble));
  auto SN = F_.createSplat("minValue", input.getType(), min);

  auto output = F_.createMax("clamp_min", input, SN);
  RETURN_ERR(addValueMapping(outputs[0], output));
}

Error PyTorchModelLoader::loadExpandAs(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 2, outputs, 1));

  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(
      input, getGlowNodeValueForValue(inputs[ExpandAsInputs::input]));

  glow::NodeValue other;
  ASSIGN_VALUE_OR_RETURN_ERR(
      other, getGlowNodeValueForValue(inputs[ExpandAsInputs::other]));

  auto output = F_.createBroadcast("expand_as", input, other.dims(),
                                   other.dims().size() - input.dims().size());
  RETURN_ERR(addValueMapping(outputs[0], output));
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

  size_t inputH = input.dims()[2];
  size_t inputW = input.dims()[3];
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
  RETURN_ERR(addValueMapping(outputs[0], output, dtype));
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
    return MAKE_ERR("Transpose requires input to have rank <= 2");
  }

  c10::ScalarType dtype;
  RETURN_IF_ERR(getCorrectTypeMapping(dtype, inputs[0]));
  RETURN_ERR(addValueMapping(outputs[0], output, dtype));
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
  RETURN_ERR(addValueMapping(outputs[0], output->getResult(), dtype));
}

Error PyTorchModelLoader::loadAbs(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 1, outputs, 1));

  glow::NodeValue in;
  ASSIGN_VALUE_OR_RETURN_ERR(in, getGlowNodeValueForValue(inputs[0]));

  auto *resultNode = F_.createAbs("Abs", in);
  return addValueMapping(outputs[0], resultNode);
}

Error PyTorchModelLoader::loadMin(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 2, outputs, 1));

  glow::NodeValue lhs;
  ASSIGN_VALUE_OR_RETURN_ERR(lhs, getGlowNodeValueForValue(inputs[0]));
  glow::NodeValue rhs;
  ASSIGN_VALUE_OR_RETURN_ERR(rhs, getGlowNodeValueForValue(inputs[1]));

  auto output = F_.createNodeWithBroadcast<MinNode>("min", -1, lhs, rhs);
  RETURN_ERR(addValueMapping(outputs[0], output));
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
      return MAKE_ERR("We don't currently support keeping dims");
    }
  }

  RETURN_ERR(addValueMapping(outputs[0], input));
}

Error PyTorchModelLoader::loadNorm(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  // (1) Without p in torch.norm(input, dim), aten::norm(Tensor, axis[],
  // keepDim) is assumed in glow_graph, the input size will be 3.
  // (2) With p in torch.norm(input, dim, p), aten::norm(Tensor, p, axis[],
  // keepDim) is assumed in glow_graph, the input size will be 4.

  // the input size is at least 3, output size is 1
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, -3, outputs, 1));

  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(input, getGlowNodeValueForValue(inputs[0]));

  int64_t axis;
  int64_t p;

  GlowIValue *pOrAxis;
  ASSIGN_VALUE_OR_RETURN_ERR(pOrAxis, getGlowIValueForValue(inputs[1]));

  if (pOrAxis->isIntList()) {
    // Without p in torch.norm(Tensor, dim), inputs[1] is the list of int
    // representing axis/dim
    std::vector<int64_t> *axisList;
    ASSIGN_VALUE_OR_RETURN_ERR(axisList, iValToIntList(pOrAxis));
    RETURN_ERR_IF_NOT(axisList->size() == 1,
                      glow::strFormat("we currently only support 1 dimension "
                                      "of axis, but got dimension size = %lu",
                                      axisList->size()));
    axis = axisList->front();
    p = 2;
  } else {
    // With p in torch.norm(input, p,  dim), inputs[1] is the int representing p
    GlowIValue *pVal;
    ASSIGN_VALUE_OR_RETURN_ERR(pVal, getGlowIValueForValue(inputs[1]));

    // Check if p is int or is float without decimal digit.
    if (!pVal->isInt()) {
      double pDouble;
      ASSIGN_VALUE_OR_RETURN_ERR(pDouble, iValToDouble(pVal));
      p = static_cast<int64_t>(pDouble);
      RETURN_ERR_IF_NOT(p == pDouble, "We only support p as an integer input");
    } else {
      ASSIGN_VALUE_OR_RETURN_ERR(p, iValToInt(pVal));
    }

    // check if p is set to 2s
    RETURN_ERR_IF_NOT(
        p == 2,
        glow::strFormat("we currently only support p = 2, but got p = %lu", p));

    // With p in torch.norm(input, p,  dim), inputs[2] is the list of int
    // representing axis/dim
    std::vector<int64_t> *axisList;
    ASSIGN_VALUE_OR_RETURN_ERR(axisList,
                               iValToIntList(getGlowIValueForValue(inputs[2])));
    RETURN_ERR_IF_NOT(axisList->size() == 1,
                      glow::strFormat("we currently only support 1 dimension "
                                      "of axis, but got dimension size = %lu",
                                      axisList->size()));
    axis = axisList->front();
  }

  auto output = F_.createVectorNorm("norm", input, axis, p);

  RETURN_ERR(addValueMapping(outputs[0], output));
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

  RETURN_ERR(addValueMapping(outputs[0], output));
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
    return MAKE_ERR("aten::mm expects 2D matrices");
  }

  if (lhs.dims()[1] != rhs.dims()[0]) {
    return MAKE_ERR("aten::mm does not broadcast");
  }

  auto output = F_.createMatMul("mm", lhs, rhs)->getResult();
  RETURN_ERR(addValueMapping(outputs[0], output));
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
    return MAKE_ERR("aten::bmm expects 3D tensors");
  }

  if (lhs.dims()[2] != rhs.dims()[1]) {
    return MAKE_ERR("aten::bmm does not broadcast");
  }

  auto output = F_.createBatchMatMul("bmm", lhs, rhs)->getResult();
  RETURN_ERR(addValueMapping(outputs[0], output));
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
    return MAKE_ERR("aten::addmm expects 2D matrices");
  }

  if (mat1.dims()[1] != mat2.dims()[0]) {
    return MAKE_ERR("aten::addmm does not broadcast mat1 or mat2");
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

  RETURN_ERR(addValueMapping(outputs[0], add));
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
  RETURN_ERR(addValueMapping(outputs[0], glowNode));
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
  RETURN_ERR(addValueMapping(outputs[0], glowNode, dtype));
}

/// TODO: check Dtype is float (optional value).
Error PyTorchModelLoader::loadSoftMax(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 3, outputs, 1));

  glow::NodeValue in;
  ASSIGN_VALUE_OR_RETURN_ERR(
      in, getGlowNodeValueForValue(inputs[SoftMaxInputs::input]));

  const auto inDims = in.dims();

  int64_t dim;
  ASSIGN_VALUE_OR_RETURN_ERR(
      dim, iValToInt(getGlowIValueForValue(inputs[SoftMaxInputs::dim])));

  ASSIGN_VALUE_OR_RETURN_ERR(dim, getPositiveIndex(dim, inDims.size()));

  // transpose dim to inner dimension
  std::vector<unsigned_t> inTransposeShuffle;
  for (auto i = 0; i < inDims.size(); ++i) {
    inTransposeShuffle.push_back(i);
  }
  inTransposeShuffle.erase(inTransposeShuffle.begin() + dim);
  inTransposeShuffle.push_back(dim);
  in = F_.createTranspose("transpose_before_softmax", in, inTransposeShuffle)
           ->getResult();

  // flatten outer dims
  auto dimsBeforeFlatten = in.dims();
  in = F_.createFlatten("flatten_before_softmax", in, inDims.size() - 1);

  // Softmax
  auto selected = F_.getParent()->createConstant(
      glow::ElemKind::Int64ITy, {in.dims()[0], 1}, "softmax_selected");
  auto out = F_.createSoftMax("softmax", in, selected)->getResult();

  // unflatten
  out = F_.createReshape("reshape_after_softmax", out, dimsBeforeFlatten);

  // transpose dim back to where it started
  auto outTransposeShufle = getInverseTranspose(inTransposeShuffle);
  out = F_.createTranspose("transpose_after_softmax", out, outTransposeShufle)
            ->getResult();
  RETURN_ERR(addValueMapping(outputs[0], out));
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
  RETURN_ERR(addValueMapping(outputs[0], output, dtype));
}

Error PyTorchModelLoader::loadTo(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();

  // aten::to() takes at least 4 input arguments, and a single output
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, -4, outputs, 1));

  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(
      input, getGlowNodeValueForValue(inputs[ToDtypeLayoutInputs::input]));
  auto inputType = input.getType();

  // Argument index of the dtype
  int dtype_arg = -1;

  // - to.dtype_layout(Tensor self, ScalarType? dtype=None,
  //                   Layout? layout=None, Device? device=None,
  //                   bool? pin_memory=None, bool non_blocking=False,
  //                   bool copy=False, MemoryFormat? memory_format=None)
  if (inputs.size() == 8 && outputs.size() == 1) {
    dtype_arg = ToDtypeLayoutInputs::dtype;
  }
  // - to.device(Tensor self, Device device, ScalarType dtype,
  //             bool non_blocking=False, bool copy=False,
  //             MemoryFormat? memory_format=None)
  else if (inputs.size() == 6 && outputs.size() == 1) {
    dtype_arg = ToDeviceInputs::dtype;
  }
  // There are three alternatives with 5 inputs:
  else if (inputs.size() == 5 && outputs.size() == 1) {
    auto glowVal = getGlowIValueForValue(inputs[ToOtherInputs::other]);
    if (glowVal) {
      // - to.other(Tensor self, Tensor other, bool non_blocking=False,
      //            bool copy=False, MemoryFormat? memory_format=None)
      if ((*glowVal)->isTensor()) {
        return MAKE_ERR("aten::to.other is not supported.");
      }
      // - to.prim_device(Tensor self, Device device, ScalarType? dtype=None,
      //                  bool non_blocking=False, bool copy=False)
      else if ((*glowVal)->isString()) {
        auto glowDtypeVal =
            getGlowIValueForValue(inputs[ToPrimDeviceInputs::dtype]);
        // Check if the dtype is set, otherwise nop (device is "cpu")
        if (glowDtypeVal) {
          if ((*glowDtypeVal)->isNone()) {
            RETURN_ERR(addValueMapping(outputs[0], input));
          }
        } else {
          RETURN_ERR(glowVal.takeError());
        }
        dtype_arg = ToPrimDeviceInputs::dtype;
        return MAKE_ERR("aten::to.prim_device is not supported.");
      }
      // - to.dtype(Tensor self, ScalarType dtype, bool non_blocking=False,
      //            bool copy=False, MemoryFormat? memory_format=None)
      else {
        dtype_arg = ToDtypeInputs::dtype;
      }
    } else {
      RETURN_ERR(glowVal.takeError());
    }
  }
  // The 4 arguments version is similar to to.dtype, without a memory format
  // - to.prim_dtype(Tensor self, ScalarType dtype,
  //                 bool non_blocking=False, bool copy=False)
  else if (inputs.size() == 4 && outputs.size() == 1) {
    dtype_arg = ToPrimDtypeInputs::dtype;
  }
  // The 3 arguments version only uses unsupported non_blocking/copy flags
  // - to.prim_other(Tensor self, bool non_blocking=False, bool copy=False)
  else if (inputs.size() == 3 && outputs.size() == 1) {
    return MAKE_ERR("aten::to.prim_other is not supported.");
  }
  // Unsupported number of input/output arguments
  else {
    return MAKE_ERR("unsupported number of arguments in aten::to node.");
  }

  if (dtype_arg < 0) {
    return MAKE_ERR("unsupported aten::to form.");
  }

  int32_t dtype;
  ASSIGN_VALUE_OR_RETURN_ERR(
      dtype, iValToInt(getGlowIValueForValue(inputs[dtype_arg])));
  auto glowElemKind = scalarTypeToElemKind(static_cast<c10::ScalarType>(dtype));

  // No type conversion necessary
  if (glowElemKind == inputType->getElementType()) {
    RETURN_ERR(addValueMapping(outputs[0], input));
  }

  if (isQuantizedElemKind(glowElemKind) ||
      isQuantizedElemKind(inputType->getElementType())) {
    // We currently dont support aten::to to quantized tensors
    // Unless input dtype == output dtype
    return MAKE_ERR("Detected quantized type for aten::to node.");
  }

  // Create a convertTo node
  auto outType = F_.getParent()->uniqueType(glowElemKind, inputType->dims());
  glow::ConvertToNode *toNode = F_.createConvertTo("to", input, outType);
  RETURN_ERR(addValueMapping(outputs[0], toNode->getResult()));
}

Error PyTorchModelLoader::loadMaskedFill(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 3, outputs, 1));

  glow::NodeValue in;
  ASSIGN_VALUE_OR_RETURN_ERR(
      in, getGlowNodeValueForValue(inputs[MaskedFillInputs::input]));

  glow::NodeValue mask;
  ASSIGN_VALUE_OR_RETURN_ERR(
      mask, getGlowNodeValueForValue(inputs[MaskedFillInputs::mask]));

  size_t inSize = in.dims().size();
  size_t maskSize = mask.dims().size();

  RETURN_ERR_IF_NOT(
      inSize >= maskSize,
      strFormat("masked_fill must have inputs at least as large as mask got "
                "input of size %zu and mask of size %zu",
                inSize, maskSize));

  size_t maskBroadcastAxis = inSize - maskSize;
  mask = F_.createBroadcast("masked_fill.broadcast", mask, in.dims(),
                            maskBroadcastAxis)
             ->getNthResult(0);

  float value;
  ASSIGN_VALUE_OR_RETURN_ERR(value, iValToDouble(getGlowIValueForValue(
                                        inputs[MaskedFillInputs::value])));

  auto valueSplat =
      F_.createSplat("masked_fill_value",
                     F_.getParent()->uniqueType(ElemKind::FloatTy, in.dims()),
                     value)
          ->getResult();

  auto out = F_.createSelect("masked_fill", mask, valueSplat, in);
  RETURN_ERR(addValueMapping(outputs[0], out));
}

Error PyTorchModelLoader::loadSelect(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 3, outputs, 1));

  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(
      input, getGlowNodeValueForValue(inputs[SelectInputs::input]));

  int64_t axis;
  GlowIValue *axisVal;
  ASSIGN_VALUE_OR_RETURN_ERR(axisVal,
                             getGlowIValueForValue(inputs[SelectInputs::dim]));
  ASSIGN_VALUE_OR_RETURN_ERR(axis, iValToInt(axisVal));

  NodeValue index;
  std::vector<dim_t> indexDims = {1};
  // Gather expects NodeValue
  glow::Type type(ElemKind::Int64ITy, indexDims);
  ASSIGN_VALUE_OR_RETURN_ERR(index, loadNodeValueOrBroadcastedIValue(
                                        inputs[SelectInputs::index], &type));
  GatherNode *gatherOutput;

  /* If axis!=0 then reshape tensor and put the axis dimension as the
     outermost dimension. Do this because gather can only gather elements
     from the outermost dimension.
  */
  if (axis != 0) {
    dim_t axisDim;
    auto inputDims = input.dims();
    std::vector<dim_t> transDims;
    //
    for (auto it = inputDims.begin(); it != inputDims.end(); ++it) {
      if (std::distance(inputDims.begin(), it) == axis) {
        axisDim = *it;
      } else {
        transDims.push_back(*it);
      }
    }
    transDims.insert(transDims.begin(), axisDim);
    auto reshapeOutput = F_.createReshape("shape", input, transDims);
    gatherOutput = F_.createGather("select", reshapeOutput, index, axis);
  }

  // Perform gather to do the selection
  gatherOutput = F_.createGather("select", input, index, axis);

  // Remove the unary dimension that's added by Gather.
  auto gatherRes = gatherOutput->getResult();
  std::vector<dim_t> outDims = gatherRes.dims();
  for (auto it = outDims.begin(); it != outDims.end(); ++it) {
    if (std::distance(outDims.begin(), it) == axis) {
      outDims.erase(it);
      break;
    }
  }

  // Reshape to flatten unary dimension
  auto reshapeOutput = F_.createReshape("reshape", gatherRes, outDims);
  return addValueMapping(outputs[0], reshapeOutput);
}

Error PyTorchModelLoader::loadFlatten(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 3, outputs, 1));

  glow::NodeValue in;
  ASSIGN_VALUE_OR_RETURN_ERR(
      in, getGlowNodeValueForValue(inputs[FlattenInputs::input]));

  auto inDims = in.dims();

  int64_t startDimRaw;
  ASSIGN_VALUE_OR_RETURN_ERR(
      startDimRaw,
      iValToInt(getGlowIValueForValue(inputs[FlattenInputs::start_dim])));
  int64_t startDim;
  ASSIGN_VALUE_OR_RETURN_ERR(startDim,
                             getPositiveIndex(startDimRaw, inDims.size()));

  int64_t endDimRaw;
  ASSIGN_VALUE_OR_RETURN_ERR(endDimRaw, iValToInt(getGlowIValueForValue(
                                            inputs[FlattenInputs::end_dim])));
  int64_t endDim;
  ASSIGN_VALUE_OR_RETURN_ERR(endDim,
                             getPositiveIndex(endDimRaw, inDims.size()));

  RETURN_ERR_IF_NOT(
      startDim <= endDim,
      strFormat("start_dim (%d) must be less than or equal to end_dim (%d)",
                int(startDimRaw), int(endDimRaw)));

  std::vector<dim_t> outDims;

  for (auto i = 0; i < startDim; ++i) {
    outDims.push_back(inDims[i]);
  }

  // Note that the range from startDim to endDim is inclusive
  dim_t flattenedDims = 1;
  for (auto i = startDim; i <= endDim; ++i) {
    flattenedDims *= inDims[i];
  }

  outDims.push_back(flattenedDims);

  for (auto i = endDim + 1; i < inDims.size(); ++i) {
    outDims.push_back(inDims[i]);
  }

  auto res = F_.createReshape("flatten", in, outDims)->getResult();

  c10::ScalarType dtype;
  RETURN_IF_ERR(getCorrectTypeMapping(dtype, inputs[FlattenInputs::input]));
  RETURN_ERR(addValueMapping(outputs[0], res, dtype));
}

Error PyTorchModelLoader::loadSqueeze(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, -1, outputs, 1));

  glow::NodeValue in;
  ASSIGN_VALUE_OR_RETURN_ERR(
      in, getGlowNodeValueForValue(inputs[SqueezeInputs::input]));

  auto inDims = in.dims();

  // Load dim parameter if provided
  int64_t dim = 0;
  bool dimProvided = false;
  if (inputs.size() == 2) {
    dimProvided = true;
    int64_t dimRaw;
    ASSIGN_VALUE_OR_RETURN_ERR(
        dimRaw, iValToInt(getGlowIValueForValue(inputs[SqueezeInputs::dim])));
    ASSIGN_VALUE_OR_RETURN_ERR(dim, getPositiveIndex(dimRaw, inDims.size()));
  }

  std::vector<dim_t> outDims;

  for (auto i = 0; i < inDims.size(); ++i) {
    auto inDim = inDims[i];
    if (inDim != 1 || (dimProvided && i != dim)) {
      outDims.push_back(inDim);
    }
  }

  auto res = F_.createReshape("squeeze", in, outDims)->getResult();

  c10::ScalarType dtype;
  RETURN_IF_ERR(getCorrectTypeMapping(dtype, inputs[SqueezeInputs::input]));
  RETURN_ERR(addValueMapping(outputs[0], res, dtype));
}

Error PyTorchModelLoader::loadUnsqueeze(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 2, outputs, 1));

  glow::NodeValue in;
  ASSIGN_VALUE_OR_RETURN_ERR(
      in, getGlowNodeValueForValue(inputs[UnsqueezeInputs::input]));

  auto inDims = in.dims();

  int64_t dimRaw;
  ASSIGN_VALUE_OR_RETURN_ERR(
      dimRaw, iValToInt(getGlowIValueForValue(inputs[UnsqueezeInputs::dim])));
  int64_t dim;
  ASSIGN_VALUE_OR_RETURN_ERR(dim, getPositiveIndex(dimRaw, inDims.size() + 1));

  std::vector<dim_t> outDims(inDims.begin(), inDims.end());

  outDims.insert(outDims.begin() + dim, 1);

  auto res = F_.createReshape("unsqueeze", in, outDims)->getResult();

  c10::ScalarType dtype;
  RETURN_IF_ERR(getCorrectTypeMapping(dtype, inputs[UnsqueezeInputs::input]));
  RETURN_ERR(addValueMapping(outputs[0], res, dtype));
}

Error PyTorchModelLoader::loadBatchPermutation(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 3, outputs, 1));
  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(input, getGlowNodeValueForValue(inputs[0]));

  glow::NodeValue indices;
  ASSIGN_VALUE_OR_RETURN_ERR(indices, getGlowNodeValueForValue(inputs[1]));
  auto res = F_.createGather("BatchPermutation", input, indices)->getResult();
  RETURN_ERR(addValueMapping(outputs[0], res));
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
    if (dim < 0) {
      dim = input.dims().size() + dim;
    }
    RETURN_ERR_IF_NOT(dim == input.dims().size() - 1,
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

Expected<GlowIValue>
PyTorchModelLoader::getGenerictList(const torch::jit::IValue &iVal) {
  auto iValList = iVal.toListRef();
  GlowIValue glowIVal;
  if (iValList[0].isTensor()) {
    std::vector<NodeValue> constantNodeValueList;
    for (const auto &v : iValList) {
      RETURN_ERR_IF_NOT(v.isTensor(),
                        strFormat("Expect all ival in a PyTorch GenericList to "
                                  "be Tensor, but got %s.",
                                  v.tagKind().c_str()));
      glow::Tensor glowTensor =
          ptTensorToGlowTensor(v.toTensor().contiguous()).clone();
      auto glowConstantNodeValue =
          F_.getParent()
              ->createConstant("GenericList_created_constant",
                               std::move(glowTensor))
              ->getOutput();
      constantNodeValueList.push_back(glowConstantNodeValue);
    }
    glowIVal.fromNodeValueList(constantNodeValueList);
  } else if (iValList[0].isInt()) {
    std::vector<int64_t> intList;
    for (auto v : iValList) {
      RETURN_ERR_IF_NOT(
          v.isInt(),
          strFormat(
              "Expect all ival in a PyTorch GenericList to be Int, but got %s.",
              v.tagKind().c_str()));
      intList.push_back(v.toInt());
    }
    glowIVal.fromIntList(intList);
  } else if (iValList[0].isDouble()) {
    std::vector<double> doubleList;
    for (auto v : iValList) {
      RETURN_ERR_IF_NOT(v.isDouble(),
                        strFormat("Expect all ival in a PyTorch GenericList to "
                                  "be Double, but got %s.",
                                  v.tagKind().c_str()));
      doubleList.push_back(v.toDouble());
    }
    glowIVal.fromDoubleList(doubleList);
  } else {
    return MAKE_ERR(strFormat("Not supported GenericList data type: %s.",
                              iValList[0].tagKind().c_str()));
  }
  return glowIVal;
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
  // If iVal is a Generic list, it need to be handled separately.
  // Everything inside of a Generic should be same type.
  if (iVal.isList() &&
      !(iVal.isDoubleList() || iVal.isIntList() || iVal.isBoolList())) {
    ASSIGN_VALUE_OR_RETURN_ERR(glowIVal, getGenerictList(iVal));
  } else {
    RETURN_IF_ERR(glowIVal.fromIValue(iVal));
  }
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
    if (isQParamWeightNode(ptNode)) {
      qparamsMap_[outputs[0]] = iVal;
    }
    glow::Tensor *t;
    ASSIGN_VALUE_OR_RETURN_ERR(t, glowIVal.toTensor());
    glow::Constant *glowConstant =
        F_.getParent()->createConstant("constant", std::move(*t));
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

Error PyTorchModelLoader::loadEmbedding(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 5, outputs, 1));

  glow::NodeValue weights;
  ASSIGN_VALUE_OR_RETURN_ERR(
      weights, getGlowNodeValueForValue(inputs[EmbeddingInputs::weights]));

  glow::NodeValue indices;
  ASSIGN_VALUE_OR_RETURN_ERR(
      indices, getGlowNodeValueForValue(inputs[EmbeddingInputs::indices]));

  int64_t padIdx;
  ASSIGN_VALUE_OR_RETURN_ERR(padIdx, iValToInt(getGlowIValueForValue(
                                         inputs[EmbeddingInputs::padIdx])));

  bool scale;
  ASSIGN_VALUE_OR_RETURN_ERR(
      scale, iValToBool(getGlowIValueForValue(inputs[EmbeddingInputs::scale])));
  RETURN_ERR_IF_NOT(scale == false,
                    "Currently only support scale_grad_by_freq == 'false'");
  bool sparse;
  ASSIGN_VALUE_OR_RETURN_ERR(sparse, iValToBool(getGlowIValueForValue(
                                         inputs[EmbeddingInputs::sparse])));
  RETURN_ERR_IF_NOT(sparse == false,
                    "Currently only support sparse == 'false'");

  auto *resultNode =
      F_.createEmbedding("Embedding", weights, indices, padIdx, scale, sparse);
  return addValueMapping(outputs[0], resultNode);
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
    glow::Tensor t(
        ElemKind::FloatTy,
        {offsets.dims()[0] > 0 ? offsets.dims()[0] - 1 : 0, weight.dims()[1]});
    t.zero();
    glow::Constant *glowConstant =
        F_.getParent()->createConstant("EmptyEmbeddingBag", std::move(t));
    RETURN_ERR(addValueMapping(outputs[0], glowConstant->getOutput()));
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

  bool includeLastOffset;
  ASSIGN_VALUE_OR_RETURN_ERR(
      includeLastOffset, iValToBool(getGlowIValueForValue(
                             inputs[EmbeddingBagInputs::include_last_offset])));
  RETURN_ERR_IF_NOT(includeLastOffset,
                    "Currently only support include_last_offset='True'");

  auto *EB = F_.createEmbeddingBag("EmbeddingBag", weight, perSampleWeights,
                                   indices, offsets, includeLastOffset);

  RETURN_ERR(addValueMapping(outputs[0], EB->getResult()));
}

Error PyTorchModelLoader::loadGlowEmbeddingBag(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 7, outputs, 1));
  // get the shape (num_embeddings, embedding_dim) and qualName for the
  // embeddingBag, and create placeholder node
  glow::dim_t numEmbedding;
  ASSIGN_VALUE_OR_RETURN_ERR(
      numEmbedding, iValToInt(getGlowIValueForValue(
                        inputs[GlowEmbeddingBagInputs::num_embeddings])));
  glow::dim_t embeddingDim;
  ASSIGN_VALUE_OR_RETURN_ERR(
      embeddingDim, iValToInt(getGlowIValueForValue(
                        inputs[GlowEmbeddingBagInputs::embedding_dim])));
  std::string *weightQualName;
  ASSIGN_VALUE_OR_RETURN_ERR(
      weightQualName, iValToString(getGlowIValueForValue(
                          inputs[GlowEmbeddingBagInputs::weight_qualname])));
  std::vector<glow::dim_t> dims{numEmbedding, embeddingDim};
  glow::Type phType(ElemKind::FloatTy, dims);
  auto legalizedWeightQualName = glow::legalizeName(*weightQualName);
  auto ph = F_.getParent()->getPlaceholderByNameSlow(legalizedWeightQualName);
  if (!ph) {
    ph = F_.getParent()->createPlaceholder(&phType, legalizedWeightQualName,
                                           /*isTrainable*/ false);
    ph->setStatic(true);
  }
  glow::NodeValue indices;
  ASSIGN_VALUE_OR_RETURN_ERR(
      indices,
      getGlowNodeValueForValue(inputs[GlowEmbeddingBagInputs::indices]));
  glow::NodeValue offsets;
  ASSIGN_VALUE_OR_RETURN_ERR(
      offsets,
      getGlowNodeValueForValue(inputs[GlowEmbeddingBagInputs::offsets]));

  // If no indices are provided, replace the op with a zero Constant.
  if (indices.dims()[0] == 0) {
    glow::Tensor t(
        ElemKind::FloatTy,
        {offsets.dims()[0] > 0 ? offsets.dims()[0] - 1 : 0, embeddingDim});
    t.zero();
    glow::Constant *glowConstant =
        F_.getParent()->createConstant("EmptyEmbeddingBag", std::move(t));
    return addValueMapping(outputs[0], glowConstant->getOutput());
  }

  bool includeLastOffset;
  ASSIGN_VALUE_OR_RETURN_ERR(
      includeLastOffset,
      iValToBool(getGlowIValueForValue(
          inputs[GlowEmbeddingBagInputs::include_last_offset])));
  RETURN_ERR_IF_NOT(includeLastOffset,
                    "Currently only support include_last_offset='True'");
  glow::NodeValue perSampleWeights = loadNodeValueOrCreateBroadcastedConstant(
      inputs[GlowEmbeddingBagInputs::per_sample_weights], "EmbeddingBag.ones",
      glow::Type(ElemKind::FloatTy, {indices.dims()[0]}), 1.0);

  auto *EB = F_.createEmbeddingBag("GlowEmbeddingBag", ph->getOutput(),
                                   perSampleWeights, indices, offsets,
                                   includeLastOffset);
  return addValueMapping(outputs[0], EB->getResult());
}

Error PyTorchModelLoader::loadEmbeddingBagByteRowwiseOffsetsHelper(
    const torch::jit::Node *ptNode, bool is4Bit) {
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

  bool includeLastOffset;
  ASSIGN_VALUE_OR_RETURN_ERR(
      includeLastOffset,
      iValToBool(getGlowIValueForValue(
          inputs[is4Bit
                     ? EmbeddingBag4BitRowwiseOffsetsInputs::include_last_offset
                     : EmbeddingBagByteRowwiseOffsetsInputs::
                           include_last_offset])));
  RETURN_ERR_IF_NOT(includeLastOffset,
                    "Currently only support include_last_offset='True'");

  // If no indices are provided, replace the op with a zero Constant.
  if (indices.dims()[0] == 0) {
    // Assuming hasEndOffset = true, so the output.dims[0] should be
    // offsets.dims[0] - 1, if offsets is not empty
    glow::Tensor t(ElemKind::FloatTy,
                   {offsets.dims()[0] > 0 ? offsets.dims()[0] - 1 : 0,
                    (is4Bit ? weight.dims()[1] * 2 : weight.dims()[1]) -
                        2 * sizeof(float)});
    t.zero();
    glow::Constant *glowConstant = F_.getParent()->createConstant(
        "EmptyEmbeddingBagByteRowwiseOffsets", std::move(t));
    RETURN_ERR(addValueMapping(outputs[0], glowConstant->getOutput()));
  }

  glow::NodeValue perSampleWeights;
  if (is4Bit) {
    // Glow supported perSampleWeights is fp16 but PyTorch uses fp32,
    // therefore the input needs to be cast.
    auto node =
        inputs[EmbeddingBagByteRowwiseOffsetsInputs::per_sample_weights];
    if (hasGlowNodeValueForValue(node)) {
      glow::NodeValue gnode;
      ASSIGN_VALUE_OR_RETURN_ERR(gnode, getGlowNodeValueForValue(node));

      perSampleWeights =
          F_.createConvertTo(
                "ConvertEmbeddingBag4BitRowwiseOffsetsPerSampleWeights", gnode,
                ElemKind::Float16Ty)
              ->getResult();
    } else {
      glow::Tensor t(ElemKind::Float16Ty, {indices.dims()[0]});
      t.init(glow::Tensor::InitKind::Broadcast, 1.0, F_.getParent()->getPRNG());
      perSampleWeights =
          F_.getParent()
              ->createConstant("EmbeddingBag4BitRowwiseOffsets.ones",
                               std::move(t))
              ->getOutput();
    }
  } else {
    perSampleWeights = loadNodeValueOrCreateBroadcastedConstant(
        inputs[EmbeddingBagByteRowwiseOffsetsInputs::per_sample_weights],
        "EmbeddingBagByteRowwiseOffsets.ones",
        glow::Type((ElemKind::FloatTy), {indices.dims()[0]}), 1.0);
  }

  glow::Constant *weightConstant =
      llvm::dyn_cast<glow::Constant>(weight.getNode());

  RETURN_ERR_IF_NOT(weightConstant,
                    strFormat("Expected Weight to be a Constant but found: %s",
                              weight.getNode()->getKindName()));

  TypeRef fusedTy = F_.getParent()->uniqueType(
      (is4Bit ? ElemKind::UInt4FusedFP16QTy : ElemKind::UInt8FusedQTy),
      weight.dims(), 0.0, 0);

  weightConstant->setType(Storage::OutputIdx, fusedTy);
  weightConstant->setPayloadType(fusedTy);

  auto *EB = F_.createEmbeddingBagByteRowwiseOffsets(
      (is4Bit ? "EmbeddingBag4BitRowwiseOffsets"
              : "EmbeddingBagByteRowwiseOffsets"),
      weightConstant->getOutput(), perSampleWeights, indices, offsets, false,
      includeLastOffset);

  // Upcast EmbeddingBag4BitRowwiseOffsets to Float32 since its Glow output type
  // is Float16.
  if (is4Bit) {
    auto *CT = F_.createConvertTo("ConvertEmbeddingBag4BitRowwiseOffsetsOutput",
                                  EB, ElemKind::FloatTy);
    RETURN_ERR(addValueMapping(outputs[0], CT->getResult()));
  } else {
    RETURN_ERR(addValueMapping(outputs[0], EB->getResult()));
  }
}

Error PyTorchModelLoader::loadNNCKernel(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  std::string kernelName = "kernel_name";
  std::string kernelSrc = "kernel_source";
  RETURN_ERR_IF_NOT(ptNode->hasAttribute(at::Symbol::attr("nnc::kernel")),
                    "Doesn't have BLOCK kernel");
  kernelSrc = ptNode->s(at::Symbol::attr("nnc::kernel"));
  // Extract the name of the kernel without its signature.
  std::size_t pos0 = kernelSrc.find(" ");
  std::size_t pos1 = kernelSrc.find("(");
  kernelName = kernelSrc.substr(pos0 + 1, pos1 - pos0 - 1);

  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, -1, outputs, 1));
  std::vector<glow::NodeValue> glowInputs(inputs.size());
  for (unsigned idx = 0, e = inputs.size(); idx < e; ++idx) {
    ASSIGN_VALUE_OR_RETURN_ERR(glowInputs[idx],
                               getGlowNodeValueForValue(inputs[idx]));
  }

  // TODO: Use a proper type based on the JIT's output type.
  TypeRef outTy = nullptr;
  // Assume for now that the type of output is the same as type of inputs for
  // elementwise operations.
  outTy = glowInputs[0].getType();
  auto glowNode =
      F_.createExternalFunctionCall("external_function_call", outTy, glowInputs,
                                    kernelName, kernelSrc, "BLOCK");
  return addValueMapping(outputs[0], glowNode);
}

Error PyTorchModelLoader::loadEmbeddingBagByteRowwiseOffsets(
    const torch::jit::Node *ptNode) {
  return loadEmbeddingBagByteRowwiseOffsetsHelper(ptNode);
}

Error PyTorchModelLoader::loadEmbeddingBag4BitRowwiseOffsets(
    const torch::jit::Node *ptNode) {
  return loadEmbeddingBagByteRowwiseOffsetsHelper(ptNode, true);
}

Error PyTorchModelLoader::loadRowwiseQuantizedEmbeddingBagHelper(
    const torch::jit::Node *ptNode, bool is4Bit) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 7, outputs, 1));

  glow::dim_t numEmbedding;
  ASSIGN_VALUE_OR_RETURN_ERR(
      numEmbedding, iValToInt(getGlowIValueForValue(
                        inputs[GlowEmbeddingBagInputs::num_embeddings])));

  glow::dim_t embeddingDim;
  ASSIGN_VALUE_OR_RETURN_ERR(
      embeddingDim, iValToInt(getGlowIValueForValue(
                        inputs[GlowEmbeddingBagInputs::embedding_dim])));

  std::string *weightQualName;
  ASSIGN_VALUE_OR_RETURN_ERR(
      weightQualName, iValToString(getGlowIValueForValue(
                          inputs[GlowEmbeddingBagInputs::weight_qualname])));

  std::vector<glow::dim_t> dims{numEmbedding, embeddingDim};
  TypeRef fusedTy = F_.getParent()->uniqueType(
      (is4Bit ? ElemKind::UInt4FusedFP16QTy : ElemKind::UInt8FusedQTy), dims,
      0.0, 0);
  glow::Placeholder *ph =
      F_.getParent()->createPlaceholder(fusedTy, *weightQualName,
                                        /*isTrainable*/ false);
  ph->setStatic(true);

  glow::NodeValue indices;
  ASSIGN_VALUE_OR_RETURN_ERR(
      indices,
      getGlowNodeValueForValue(inputs[GlowEmbeddingBagInputs::indices]));

  glow::NodeValue offsets;
  ASSIGN_VALUE_OR_RETURN_ERR(
      offsets,
      getGlowNodeValueForValue(inputs[GlowEmbeddingBagInputs::offsets]));

  bool includeLastOffset;
  ASSIGN_VALUE_OR_RETURN_ERR(
      includeLastOffset,
      iValToBool(getGlowIValueForValue(
          inputs[GlowEmbeddingBagInputs::include_last_offset])));
  RETURN_ERR_IF_NOT(includeLastOffset,
                    "Currently only support include_last_offset='True'");

  // If no indices are provided, replace the op with a zero Constant.
  if (indices.dims()[0] == 0) {
    // Assuming hasEndOffset = true, so the output.dims[0] should be
    // offsets.dims[0] - 1, if offsets is not empty
    glow::Tensor t(
        (is4Bit ? ElemKind::UInt4FusedFP16QTy : ElemKind::UInt8FusedQTy),
        {offsets.dims()[0] > 0 ? offsets.dims()[0] - 1 : 0,
         // TODO: this really depends on how embedding_dim is specified during
         // materialization. Revist here once that's done.
         (is4Bit ? embeddingDim * 2 : embeddingDim) - 2 * sizeof(float)});
    t.zero();
    glow::Constant *glowConstant = F_.getParent()->createConstant(
        "EmptyRowwiseQuantizedEmbeddingBag", std::move(t));
    RETURN_ERR(addValueMapping(outputs[0], glowConstant->getOutput()));
  }

  glow::NodeValue perSampleWeights;
  if (is4Bit) {
    // Glow supported perSampleWeights is fp16 but PyTorch uses fp32,
    // therefore the input needs to be cast.
    auto node = inputs[GlowEmbeddingBagInputs::per_sample_weights];
    if (hasGlowNodeValueForValue(node)) {
      glow::NodeValue gnode;
      ASSIGN_VALUE_OR_RETURN_ERR(gnode, getGlowNodeValueForValue(node));

      perSampleWeights =
          F_.createConvertTo(
                "ConvertEmbeddingBag4BitRowwiseOffsetsPerSampleWeights", gnode,
                ElemKind::Float16Ty)
              ->getResult();
    } else {
      glow::Tensor t(ElemKind::Float16Ty, {indices.dims()[0]});
      t.init(glow::Tensor::InitKind::Broadcast, 1.0, F_.getParent()->getPRNG());
      perSampleWeights =
          F_.getParent()
              ->createConstant("EmbeddingBag4BitRowwiseOffsets.ones",
                               std::move(t))
              ->getOutput();
    }
  } else {
    perSampleWeights = loadNodeValueOrCreateBroadcastedConstant(
        inputs[GlowEmbeddingBagInputs::per_sample_weights],
        "EmbeddingBagByteRowwiseOffsets.ones",
        glow::Type((ElemKind::FloatTy), {indices.dims()[0]}), 1.0);
  }

  auto *EB = F_.createEmbeddingBagByteRowwiseOffsets(
      (is4Bit ? "EmbeddingBag4BitRowwiseOffsets"
              : "EmbeddingBagByteRowwiseOffsets"),
      ph->getOutput(), perSampleWeights, indices, offsets, false,
      includeLastOffset);

  // Upcast EmbeddingBag4BitRowwiseOffsets to Float32 since its Glow output type
  // is Float16.
  if (is4Bit) {
    auto *CT = F_.createConvertTo("ConvertEmbeddingBag4BitRowwiseOffsetsOutput",
                                  EB, ElemKind::FloatTy);
    RETURN_ERR(addValueMapping(outputs[0], CT->getResult()));
  } else {
    RETURN_ERR(addValueMapping(outputs[0], EB->getResult()));
  };
}

Error PyTorchModelLoader::loadGlowEmbeddingBagByteRowwiseOffsets(
    const torch::jit::Node *ptNode) {
  return loadRowwiseQuantizedEmbeddingBagHelper(ptNode, false);
}

Error PyTorchModelLoader::loadGlowEmbeddingBag4bitRowwiseOffsets(
    const torch::jit::Node *ptNode) {
  return loadRowwiseQuantizedEmbeddingBagHelper(ptNode, true);
}

Error PyTorchModelLoader::loadFusedSplit(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 3, outputs, -1));

  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(
      input, getGlowNodeValueForValue(inputs[FusedSplitInputs::input]));

  int num_split;
  ASSIGN_VALUE_OR_RETURN_ERR(
      num_split,
      iValToInt(getGlowIValueForValue(inputs[FusedSplitInputs::num_split])));

  RETURN_ERR_IF_NOT(num_split == outputs.size(),
                    "Number of splits not equal to output size!");

  int dim;
  ASSIGN_VALUE_OR_RETURN_ERR(
      dim, iValToInt(getGlowIValueForValue(inputs[FusedSplitInputs::dim])));

  std::vector<glow::SliceNode *> splitOutputs;
  F_.createSplit("EquallySplit", input, num_split, dim, {}, splitOutputs);
  for (size_t i = 0; i < splitOutputs.size(); ++i) {
    RETURN_IF_ERR(addValueMapping(outputs[i], splitOutputs[i]->getResult()));
  }
  return Error::success();
}

Error PyTorchModelLoader::loadFastGather(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 2, outputs, 1));

  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(
      input, getGlowNodeValueForValue(inputs[FastGatherInputs::input]));

  glow::NodeValue indices;
  ASSIGN_VALUE_OR_RETURN_ERR(
      indices, getGlowNodeValueForValue(inputs[FastGatherInputs::indices]));

  auto *g = F_.createGather("FastGather", input, indices);

  RETURN_ERR(addValueMapping(outputs[0], g->getResult()));
}

Error PyTorchModelLoader::loadLengthsRange(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 2, outputs, 1));

  glow::NodeValue lengths;
  ASSIGN_VALUE_OR_RETURN_ERR(
      lengths, getGlowNodeValueForValue(inputs[LengthsRangeInputs::lengths]));

  GlowIValue *shapes;
  ASSIGN_VALUE_OR_RETURN_ERR(
      shapes, getGlowIValueForValue(inputs[LengthsRangeInputs::shapes]));
  RETURN_ERR_IF_NOT(shapes->isNone() == true, "Expects shapes to be None");
  // TODO: fix UINT_MAX
  auto *LRF = F_.createLengthsRangeFill("LengthsRange", lengths, UINT_MAX);
  RETURN_ERR(addValueMapping(outputs[0], LRF->getResult()));
}

Error PyTorchModelLoader::loadRoiAlignImpl(const torch::jit::Node *ptNode,
                                           bool isRotated) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 9, outputs, 1));

  glow::NodeValue features;
  ASSIGN_VALUE_OR_RETURN_ERR(
      features, getGlowNodeValueForValue(inputs[RoiAlignInputs::features]));

  glow::NodeValue rois;
  ASSIGN_VALUE_OR_RETURN_ERR(
      rois, getGlowNodeValueForValue(inputs[RoiAlignInputs::rois]));

  std::string *layout;
  ASSIGN_VALUE_OR_RETURN_ERR(layout, iValToString(getGlowIValueForValue(
                                         inputs[RoiAlignInputs::layout])));

  bool needsTranspose = false;

  if (*layout == "NCHW") {
    needsTranspose = true;
  } else if (*layout == "NHWC") {
    needsTranspose = false;
  } else {
    return MAKE_ERR(strFormat("Invalid RoiAlign layout: %s", layout->c_str()));
  }

  float spatialScale;
  ASSIGN_VALUE_OR_RETURN_ERR(spatialScale,
                             iValToDouble(getGlowIValueForValue(
                                 inputs[RoiAlignInputs::spatialScale])));

  int64_t outputHeight;
  ASSIGN_VALUE_OR_RETURN_ERR(
      outputHeight,
      iValToInt(getGlowIValueForValue(inputs[RoiAlignInputs::outputHeight])));

  int64_t outputWidth;
  ASSIGN_VALUE_OR_RETURN_ERR(
      outputWidth,
      iValToInt(getGlowIValueForValue(inputs[RoiAlignInputs::outputWidth])));

  int64_t samplingRatio;
  ASSIGN_VALUE_OR_RETURN_ERR(
      samplingRatio,
      iValToInt(getGlowIValueForValue(inputs[RoiAlignInputs::samplingRatio])));

  bool aligned;
  ASSIGN_VALUE_OR_RETURN_ERR(aligned, iValToBool(getGlowIValueForValue(
                                          inputs[RoiAlignInputs::aligned])));

  if (needsTranspose) {
    features = F_.createTranspose("features_transposed", features, NCHW2NHWC)
                   ->getResult();
  }

  // Create a dummy BatchIndices tensor because this input is not used in
  // PyTorch/Caffe2.
  auto dummyBatchIndices =
      F_.getParent()
          ->createConstant(ElemKind::Int64ITy, {rois.dims()[0]},
                           "dummy_batch_indices")
          ->getOutput();

  NodeValue output =
      F_.createROIAlign("RoiAlign", features, rois, dummyBatchIndices,
                        outputHeight, outputWidth, samplingRatio, spatialScale,
                        aligned, isRotated)
          ->getResult();

  if (needsTranspose) {
    output =
        F_.createTranspose("roi_align_output_transposed", output, NHWC2NCHW);
  }

  RETURN_ERR(addValueMapping(outputs[0], output));
}

Error PyTorchModelLoader::loadRoiAlign(const torch::jit::Node *ptNode) {
  return loadRoiAlignImpl(ptNode, /*isRotated*/ false);
}

Error PyTorchModelLoader::loadRoiAlignRotated(const torch::jit::Node *ptNode) {
  return loadRoiAlignImpl(ptNode, /*isRotated*/ true);
}

Error PyTorchModelLoader::loadBBoxTransform(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 12, outputs, 2));

  glow::NodeValue rois;
  ASSIGN_VALUE_OR_RETURN_ERR(
      rois, getGlowNodeValueForValue(inputs[BBoxTransformInputs::rois]));

  glow::NodeValue deltas;
  ASSIGN_VALUE_OR_RETURN_ERR(
      deltas, getGlowNodeValueForValue(inputs[BBoxTransformInputs::deltas]));

  glow::NodeValue imInfo;
  ASSIGN_VALUE_OR_RETURN_ERR(
      imInfo, getGlowNodeValueForValue(inputs[BBoxTransformInputs::imInfo]));

  std::vector<double> *weightsDouble;
  ASSIGN_VALUE_OR_RETURN_ERR(weightsDouble,
                             iValToDoubleList(getGlowIValueForValue(
                                 inputs[BBoxTransformInputs::weights])));
  std::vector<float> weights;
  for (const auto &w : *weightsDouble) {
    weights.push_back(w);
  }

  bool applyScale;
  ASSIGN_VALUE_OR_RETURN_ERR(applyScale,
                             iValToBool(getGlowIValueForValue(
                                 inputs[BBoxTransformInputs::applyScale])));

  bool rotated;
  ASSIGN_VALUE_OR_RETURN_ERR(
      rotated,
      iValToBool(getGlowIValueForValue(inputs[BBoxTransformInputs::rotated])));

  bool angleBoundOn;
  ASSIGN_VALUE_OR_RETURN_ERR(angleBoundOn,
                             iValToBool(getGlowIValueForValue(
                                 inputs[BBoxTransformInputs::angleBoundOn])));

  int64_t angleBoundLo;
  ASSIGN_VALUE_OR_RETURN_ERR(angleBoundLo,
                             iValToInt(getGlowIValueForValue(
                                 inputs[BBoxTransformInputs::angleBoundLo])));

  int64_t angleBoundHi;
  ASSIGN_VALUE_OR_RETURN_ERR(angleBoundHi,
                             iValToInt(getGlowIValueForValue(
                                 inputs[BBoxTransformInputs::angleBoundHi])));

  float clipAngleThresh;
  ASSIGN_VALUE_OR_RETURN_ERR(
      clipAngleThresh, iValToDouble(getGlowIValueForValue(
                           inputs[BBoxTransformInputs::clipAngleThresh])));

  bool legacyPlusOne;
  ASSIGN_VALUE_OR_RETURN_ERR(legacyPlusOne,
                             iValToBool(getGlowIValueForValue(
                                 inputs[BBoxTransformInputs::legacyPlusOne])));

  auto *BBTN = F_.createBBoxTransform(
      "BBoxTransform", rois, deltas, imInfo, weights, applyScale, rotated,
      angleBoundOn, angleBoundLo, angleBoundHi, clipAngleThresh, legacyPlusOne);

  RETURN_IF_ERR(addValueMapping(outputs[0], BBTN->getBoxOut()));
  RETURN_IF_ERR(addValueMapping(outputs[1], BBTN->getRoiBatchSplits()));
  return Error::success();
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
      if (isQParamWeightNode(node)) {
        qparamsMap_[outputValue] = ival;
      } else {
        objectTree[outputValue] =
            std::make_pair(&ival.toObjectRef(), newNameHierarchy);
      }
      continue;
    } else if (ival.isTensor()) {
      GlowIValue glowIVal;
      // PyTorch Tensor extracted type is kByte
      // indicate it is the address of stored weights of quantized
      // linear or conv.
      if (isQParamWeightNode(node)) {
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
    const InputMetaStack &metaStack) {
  Error error = Error::empty();
  PyTorchModelLoader loader(F, graph, inputPlaceholders, outputPlaceholders,
                            outputCorrectType, error, settings, inputs,
                            metaStack);
  return error;
}

PyTorchModelLoader::PyTorchModelLoader(
    glow::Function &F, const torch::jit::Graph &graph,
    std::vector<glow::Placeholder *> &inputPlaceholders,
    std::vector<glow::Placeholder *> &outputPlaceholders,
    std::vector<c10::ScalarType> &outputCorrectType, Error &error,
    const PyTorchLoaderSettings &settings,
    const at::ArrayRef<torch::jit::IValue> inputs,
    const InputMetaStack &metaStack)
    : F_(F), settings_(settings), inputs_(inputs) {
  std::cerr << "loading PyTorch graph\n" << graph << std::endl;
  auto loadFn = [&]() -> Error {
    auto graphInputValues = graph.inputs();

    LOG(INFO) << "Using settings: " << settings_.toString();

    if (settings_.dumpFinalGlowGraph || settings_.dumpGlowDag) {
      const std::string fname = "preLoadGlowGraph.ir";
      LOG(INFO) << "Dumping pre load graph at " + fname;
      std::ofstream out;
      out.open(fname);
      graph.print(out);
      out.close();
    }

    RETURN_ERR_IF_NOT(
        inputs.size() == graphInputValues.size() ||
            metaStack.inputMetas.size() == graphInputValues.size(),
        glow::strFormat("Number of Graph inputs %lu must match the "
                        "number of provided inputs %lu.",
                        graphInputValues.size(), inputs.size()));
    // Create Glow Placeholders for inputs.
    for (size_t i = 0; i < graphInputValues.size(); ++i) {
      const torch::jit::Value *inputValue = graphInputValues[i];
      c10::ScalarType inputScalarType;
      glow::Placeholder *ph;
      if (!metaStack.inputMetas.empty()) {
        glow::Type t;
        if (inputValue->type()->kind() == c10::TypeKind::TensorType) {
          inputScalarType = metaStack.inputMetas[i].type;
          // TODO: Change Glow Type to use sdim_t to be consistent
          // with other places.
          std::vector<glow::dim_t> dims;
          for (auto d : metaStack.inputMetas[i].dims) {
            dims.push_back(static_cast<glow::dim_t>(d));
          }

          if (!c10::isQIntType(metaStack.inputMetas[i].type)) {
            t = glow::Type(scalarTypeToElemKind(metaStack.inputMetas[i].type),
                           dims);
          } else if (metaStack.inputMetas[i].type == at::kQUInt8) {
            t = glow::Type(
                ElemKind::Int8QTy, dims, metaStack.inputMetas[i].scale,
                metaStack.inputMetas[i].offset - UINT8_TO_INT8_SHIFT);
          } else {
            t = glow::Type(scalarTypeToElemKind(metaStack.inputMetas[i].type),
                           dims, metaStack.inputMetas[i].scale,
                           metaStack.inputMetas[i].offset);
          }
        } else {
          // Here we assume it's scalar type
          t = glow::Type(typeKindToElemKind(inputValue->type()->kind()), {});
          inputScalarType = elemKindToScalarType(t.getElementType());
        }
        ph =
            F_.getParent()->createPlaceholder(&t, strFormat("input_%d", int(i)),
                                              /*isTrainable*/ false);
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
            auto newType = glow::Type(
                ElemKind::Int8QTy, oldType.dims(), oldType.getScale(),
                oldType.getOffset() - UINT8_TO_INT8_SHIFT);
            ph = F_.getParent()->createPlaceholder(
                &newType, strFormat("input_%d", int(i)),
                /*isTrainable*/ false);
          } else {
            ph = F_.getParent()->createPlaceholder(
                &t->getType(), strFormat("input_%d", int(i)),
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
    const auto graphOutputs = graph.outputs();
    for (size_t i = 0; i < graphOutputs.size(); ++i) {
      const torch::jit::Value *output = graphOutputs[i];
      glow::NodeValue outputNodeValue;
      // Only allow tensor outputs from Glow subgraph.
      ASSIGN_VALUE_OR_RETURN_ERR(outputNodeValue,
                                 getGlowNodeValueForValue(output));
      auto *save =
          F_.createSave(strFormat("output_%d", int(i)), outputNodeValue);
      outputPlaceholders.push_back(save->getPlaceholder());

      c10::ScalarType outputScalarType;
      RETURN_IF_ERR(getCorrectTypeMapping(outputScalarType, output));
      outputCorrectType.push_back(outputScalarType);

      if (settings_.debugContinuouslyVerifyDuringModelLoading) {
        if (!F_.verify()) {
          F_.dumpDAG("failed.dot");
          return MAKE_ERR(
              "Failed Function verification while loading graph outputs.");
        }
      }
    }

    // When randomizing constants in graphs, don't randomize scales/offsets for
    // rowwise/channelwise ops.
    static std::map<Kinded::Kind, std::set<unsigned>>
        randomizeConstantsIgnoreSet = {
            {Kinded::Kind::ChannelwiseQuantizedConvolutionNodeKind,
             {ChannelwiseQuantizedConvolutionNode::InputIndices::
                  FilterOffsetsIdx,
              ChannelwiseQuantizedConvolutionNode::InputIndices::
                  FilterScalesIdx}},
            {Kinded::Kind::RowwiseQuantizedFullyConnectedNodeKind,
             {RowwiseQuantizedFullyConnectedNode::InputIndices::OffsetsIdx,
              RowwiseQuantizedFullyConnectedNode::InputIndices::ScalesIdx}},
        };

    if (settings_.randomizeConstants) {
      F_.randomizeConstants(randomizeConstantsIgnoreSet);
    }

    if (!F_.verify()) {
      F_.dumpDAG("failed.dot");
      return MAKE_ERR(
          "Failed Function verification after loading JIT graph. Enable the "
          "debugContinuouslyVerifyDuringModelLoading setting and run again to "
          "see which JIT node causes the failure.");
    }

    if (settings_.dumpGlowDag) {
      F_.dumpDAG(strFormat("%s.dot", F_.getName().data()));
    }

    if (settings_.writeToOnnx) {
      RETURN_ERR_IF_NOT(
          settings_.randomizeConstants || settings_.writeWithoutRandomize,
          "Write to Onnx without randomizing constants is not allowed! To "
          "allow this set flag `writeWithoutRandomize`.");
      LOG_IF(WARNING, !settings_.randomizeConstants)
          << "Write to Onnx without randomize constants!!!";
      std::string filename = settings_.onnxFileNamePrefix;
      if (filename.empty()) {
        filename = F.getName().str();
      }
      RETURN_IF_ERR(dumpOnnxModel(F, settings_.onnxZipMode, filename,
                                  settings_.writeOnnxToTmp));
    }

    return Error::success();
  };
  error = loadFn();

  if (error) {
    LOG(ERROR) << "Encountered an error while loading JIT graph:\n" << graph;
  }
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
    return MAKE_ERR("ValueMapping doesn't contain a NodeValue");
  } else {
    return nodeValue_;
  }
}

Expected<GlowIValue *> ValueMapping::getMappedGlowIValue() {
  if (mappingType_ == ValueMappingType::IValue) {
    return glowIValue_.get();
  } else {
    return MAKE_ERR("ValueMapping doesn't contain a GlowIValue");
  }
}

Expected<const GlowIValue *> ValueMapping::getMappedGlowIValue() const {
  if (mappingType_ == ValueMappingType::IValue) {
    return glowIValue_.get();
  } else {
    return MAKE_ERR("ValueMapping doesn't contain a GlowIValue");
  }
}

} // namespace glow
