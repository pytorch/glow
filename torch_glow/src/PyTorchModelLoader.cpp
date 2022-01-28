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

#include "c10/core/Scalar.h"
#include "c10/core/ScalarType.h"
#include "glow/Exporter/ONNXModelWriter.h"
#include "glow/Importer/CommonOperatorLoader.h"
#include "glow/Quantization/Base/Base.h"
#include "glow/Support/Error.h"
#include "glow/Support/Support.h"
#include "torch_glow/src/GlowIValue.h"
#include "torch_glow/src/ShapeInferenceEngine.h"

#include <ATen/ATen.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/native/quantized/cpu/conv_packed_params.h>
#include <ATen/native/quantized/cpu/packed_params.h>
#include <limits>
#include <torch/csrc/jit/ir/ir.h>
#include <type_traits>
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

/// Downcast a int64 to a int32.
Expected<int32_t> to32Bit(int64_t val) {
  RETURN_ERR_IF_NOT(val <= std::numeric_limits<int32_t>::max() ||
                        val >= std::numeric_limits<int32_t>::lowest(),
                    glow::strFormat("Value %ld is out of limit.", long(val)));
  return Expected<int32_t>(static_cast<int32_t>(val));
}

/// Unwrap a Expected and call to32Bit(int64_t) or any contained return
/// Error.
Expected<int32_t> to32Bit(Expected<int64_t> expectedVal) {
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
      torch::jit::Symbol::fromQualString("quantized::linear_dynamic"),
      torch::jit::Symbol::fromQualString("quantized::conv2d"),
      torch::jit::Symbol::fromQualString("quantized::conv2d_relu"),
      torch::jit::Symbol::fromQualString("quantized::conv3d"),
      torch::jit::Symbol::fromQualString("quantized::conv3d_relu"),
      torch::jit::Symbol::fromQualString("glow::unpacked_quantized_linear"),
      torch::jit::Symbol::fromQualString(
          "fb::quantized_linear_unpacked_weight"),
      torch::jit::Symbol::fromQualString(
          "fb::quantized_linear_unpacked_weight_v2"),
      torch::jit::Symbol::fromQualString("glow::unpacked_quantized_conv2d"),
      torch::jit::Symbol::fromQualString(
          "glow::unpacked_quantized_conv2d_relu"),
      torch::jit::Symbol::fromQualString("glow::unpacked_quantized_conv3d"),
      torch::jit::Symbol::fromQualString(
          "glow::unpacked_quantized_conv3d_relu"),
      torch::jit::Symbol::fromQualString("fb::quantize_per_tensor"),
      torch::jit::Symbol::fromQualString("fb::quantized_linear"),
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

/// Indexes of aten::empty_like inputs.
struct EmptyLikeInputs {
  enum {
    self = 0,
    dtype = 1,
    layout = 2,
    device = 3,
    pin_memory = 4,
    memory_format = 5,
  };
};

/// Indexes of aten::zeros_like inputs.
using ZerosLikeInputs = EmptyLikeInputs;

/// Indexes of aten::ones_like inputs.
using OnesLikeInputs = EmptyLikeInputs;

/// Indexes of aten::full_like inputs.
struct FullLikeInputs {
  enum {
    self = 0,
    fill_value = 1,
    dtype = 2,
    layout = 3,
    device = 4,
    pin_memory = 5,
    memory_format = 6,
  };
};

// Indexes of aten:arg_min and arg_max inputs.
struct ArgMaxMinInputs {
  enum {
    input = 0,
    axis = 1,
    keepDims = 2,
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

/// Indexes of quantized::layer_norm inputs.
struct QuantizedLayerNormInputs {
  enum {
    input = 0,
    normalized_shape = 1,
    weight = 2,
    bias = 3,
    eps = 4,
    output_scale = 5,
    output_zero_point = 6,
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

/// Indexes of quantized::leaky_relu inputs.
struct QuantizedLeakyReluInputs {
  enum {
    lhs = 0,
    alpha = 1,
    inplace = 2,
    scale = 3,
    zero_point = 4,
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
/// Also used for fb::quantized_linear_unpacked_weight
struct QuantizedUnpackedLinearInputs {
  enum {
    input = 0,
    weight = 1,
    bias = 2,
    scale = 3,
    zero_point = 4,
  };
};

/// Indexes of aten::linear inputs.
struct LinearInput {
  enum {
    input = 0,
    weight = 1,
    bias = 2,
  };
};

/// Indexes of quantized::linear_dynamic inputs.
struct DynQuantizedLinearInputs {
  enum {
    input = 0,
    packed_weights = 1,
    reduce_range = 2,
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

/// Indexes of aten::log_softmax inputs.
using LogSoftMaxInputs = SoftMaxInputs;

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

/// Indexes of fb::unsqueeze_n_times inputs.
struct UnsqueezeNTimesInputs {
  enum {
    input = 0,
    n = 1,
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

/// Indexes of aten::to.prim_dtype inputs.
struct ToPrimDtypeInputs {
  enum {
    input = 0,
    dtype,
    non_blocking,
    copy,
  };
};

/// Indexes of aten::to.prim_device inputs.
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

/// Indexes of aten::expand inputs.
struct ExpandInputs {
  enum {
    input = 0,
    shape,
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

/// Indexes of fb::xl_embedding_bag inputs.
struct XLEmbeddingBagInputs {
  enum {
    weight_id,
    indices,
    offsets,
    scale_grad_by_freq,
    mode,
    sparse,
    per_sample_weights,
    include_last_offset,
    num_embeddings,
    embedding_dim,
    avg_length,
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

struct XLEmbeddingBagRowwiseOffsetsInputs {
  enum {
    weight_id = 0,
    indices,
    offsets,
    scale_grad_by_freq,
    mode,
    pruned_weights,
    per_sample_weights,
    compressed_indices_mapping_id,
    include_last_offset,
    num_embeddings,
    embedding_dim,
    avg_length,
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

/// Indexes used for aten::gather inputs.
struct GatherElementsInputs {
  enum {
    input = 0,
    dim,
    indices,
    sparse,
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

/// Indexes of aten::cumsum inputs.
struct CumSumInputs {
  enum {
    input = 0,
    dim = 1,
    dtype = 2,
  };
};

/// Indexes used for aten::index_put inputs
struct IndexPutInputs {
  enum {
    input = 0,
    indices,
    value,
  };
};

/// Indexes of aten::index_add inputs.
struct IndexAddInputs {
  enum {
    input = 0,
    dim,
    index,
    source,
    alpha,
  };
};

/// Indexes used for aten::repeat inputs
struct RepeatInputs {
  enum {
    input = 0,
    repeats,
  };
};

/// Indexes used for fb::expand_dims inputs
struct ExpandDimsInputs {
  enum {
    input = 0,
    dims,
  };
};

/// Indexes used for aten::narrow inputs
struct NarrowInputs {
  enum { input = 0, dim, start, length };
};

///  Indexes used for aten::pixel_shuffle inputs
struct PixelShuffleInputs {
  enum {
    input = 0,
    upscale_factor,
  };
};

///  Indexes used for aten::pixel_shuffle inputs
struct PixelUnshuffleInputs {
  enum {
    input = 0,
    downscale_factor,
  };
};

///  Indexes used for fb::batched_unary_embeddings inputs
struct BatchedUnaryEmbeddingsBagsInputs {
  enum {
    weights = 0,
    tableOffsets,
    offsets,
    indices,
  };
};

/// Indexes used for fb::int_nbit_split_embedding_codegen_lookup_function inputs
struct IntNBitSplitEmbeddingBagsInputs {
  enum {
    dev_weights = 0,
    uvm_weights,
    weights_placements,
    weights_offsets,
    weights_tys,
    dimOffsets,
    totalDims,
    max_int2_D,
    max_int4_D,
    max_int8_D,
    max_float16_D,
    max_float32_D,
    indices,
    offsets,
    pooling_mode,
    indice_weights,
    output_dtype,
  };
};

} // namespace

// static
const PyTorchModelLoader::MappingOfMemberFunctions
PyTorchModelLoader::buildSymbolsMapping() {
// First build mapping with standard PyTorch operators.
#define UNARY_NODE_LOADER(NODE)                                                \
  static_cast<MappingOfMemberFunctionsValue::LoadFn>(                          \
      &PyTorchModelLoader::loadUnaryNode<glow::NODE##Node,                     \
                                         &glow::Function::create##NODE>)
  auto symbolLoaderMapping = MappingOfMemberFunctions({
      {{"aten::type_as"},
       &PyTorchModelLoader::loadTypeAs,
       &PyTorchModelLoader::getCorrectTypeFromInput<1>},
      {{"aten::contiguous"},
       &PyTorchModelLoader::loadCopy<2, false>,
       &PyTorchModelLoader::getCorrectTypeFromInput<0>},
      {{"aten::detach"},
       &PyTorchModelLoader::loadCopy<1, false>,
       &PyTorchModelLoader::getCorrectTypeFromInput<0>},
      {{"aten::copy_"},
       &PyTorchModelLoader::loadCopy<-2, true>,
       &PyTorchModelLoader::getCorrectTypeFromInput<0>},
      {{"aten::clone"},
       &PyTorchModelLoader::loadCopy<2, false>,
       &PyTorchModelLoader::getCorrectTypeFromInput<0>},
      {{"prim::Constant"},
       &PyTorchModelLoader::loadConstant,
       &PyTorchModelLoader::correctTypeAlreadySet},
      {{"prim::NumToTensor"},
       &PyTorchModelLoader::loadNumToTensor,
       &PyTorchModelLoader::correctTypeAlreadySet},
      {{"aten::_shape_as_tensor"},
       &PyTorchModelLoader::loadShapeAsTensor,
       &PyTorchModelLoader::makeCorrectType<at::kLong>},
      {{"aten::argmin"},
       &PyTorchModelLoader::loadArgMin,
       &PyTorchModelLoader::makeCorrectType<at::kLong>},
      {{"aten::argmax"},
       &PyTorchModelLoader::loadArgMax,
       &PyTorchModelLoader::makeCorrectType<at::kLong>},
      {{"aten::Int"},
       &PyTorchModelLoader::loadInt,
       &PyTorchModelLoader::correctTypeAlreadySet},
      {{"aten::arange"},
       &PyTorchModelLoader::loadArange,
       &PyTorchModelLoader::correctTypeAlreadySet},
      {{"aten::zeros"},
       &PyTorchModelLoader::loadZeros,
       &PyTorchModelLoader::makeCorrectType<at::kFloat>},
      {{"aten::empty_like"},
       &PyTorchModelLoader::loadEmptyLike,
       &PyTorchModelLoader::correctTypeAlreadySet},
      {{"aten::zeros_like"},
       &PyTorchModelLoader::loadZerosLike,
       &PyTorchModelLoader::correctTypeAlreadySet},
      {{"aten::ones_like"},
       &PyTorchModelLoader::loadOnesLike,
       &PyTorchModelLoader::correctTypeAlreadySet},
      {{"aten::full_like"},
       &PyTorchModelLoader::loadFullLike,
       &PyTorchModelLoader::correctTypeAlreadySet},
      {{"aten::mul", "aten::mul_"},
       &PyTorchModelLoader::loadMul,
       &PyTorchModelLoader::correctTypeAlreadySet},
      {{"aten::div", "aten::div_"},
       &PyTorchModelLoader::loadDiv,
       &PyTorchModelLoader::correctTypeAlreadySet},
      {{"aten::floor_divide", "aten::floor_divide_"},
       &PyTorchModelLoader::loadFloorDiv,
       &PyTorchModelLoader::correctTypeAlreadySet},
      {{"aten::fmod", "aten::fmod_"},
       &PyTorchModelLoader::loadFmod,
       &PyTorchModelLoader::correctTypeAlreadySet},
      {{"aten::add", "aten::add_"},
       &PyTorchModelLoader::loadAdd,
       &PyTorchModelLoader::correctTypeAlreadySet},
      {{"aten::sub", "aten::sub_"},
       &PyTorchModelLoader::loadSub,
       &PyTorchModelLoader::correctTypeAlreadySet},
      {{"aten::rsub"},
       &PyTorchModelLoader::loadRsub,
       &PyTorchModelLoader::correctTypeAlreadySet},
      {{"aten::log"},
       UNARY_NODE_LOADER(Log),
       &PyTorchModelLoader::getCorrectTypeFromInput<0>},
      {{"aten::sum"},
       &PyTorchModelLoader::loadSum,
       &PyTorchModelLoader::correctTypeAlreadySet},
      {{"aten::sigmoid", "aten::sigmoid_"},
       UNARY_NODE_LOADER(Sigmoid),
       &PyTorchModelLoader::getCorrectTypeFromInput<0>},
      {{"aten::silu"},
       UNARY_NODE_LOADER(Swish),
       &PyTorchModelLoader::getCorrectTypeFromInput<0>},
      {{"aten::relu", "aten::relu_"},
       UNARY_NODE_LOADER(Relu),
       &PyTorchModelLoader::getCorrectTypeFromInput<0>},
      {{"aten::leaky_relu", "aten::leaky_relu_"},
       &PyTorchModelLoader::loadLeakyRelu,
       &PyTorchModelLoader::getCorrectTypeFromInput<0>},
      {{"aten::gelu"},
       UNARY_NODE_LOADER(Gelu),
       &PyTorchModelLoader::getCorrectTypeFromInput<0>},
      {{"aten::tanh", "aten::tanh_"},
       UNARY_NODE_LOADER(Tanh),
       &PyTorchModelLoader::getCorrectTypeFromInput<0>},
      {{"aten::softplus"},
       &PyTorchModelLoader::loadSoftPlus,
       &PyTorchModelLoader::getCorrectTypeFromInput<0>},
      {{"aten::t", "aten::t_"},
       &PyTorchModelLoader::loadT,
       &PyTorchModelLoader::getCorrectTypeFromInput<0>},
      {{"aten::permute"},
       &PyTorchModelLoader::loadPermute,
       &PyTorchModelLoader::getCorrectTypeFromInput<0>},
      {{"aten::transpose", "aten::transpose_"},
       &PyTorchModelLoader::loadTranspose,
       &PyTorchModelLoader::getCorrectTypeFromInput<0>},
      {{"aten::min"},
       &PyTorchModelLoader::loadMin,
       &PyTorchModelLoader::getCorrectTypeFromInput<0>},
      {{"aten::max"},
       &PyTorchModelLoader::loadMax,
       &PyTorchModelLoader::correctTypeAlreadySet},
      {{"aten::exp"},
       UNARY_NODE_LOADER(Exp),
       &PyTorchModelLoader::getCorrectTypeFromInput<0>},
      {{"prim::FusedConcat"},
       &PyTorchModelLoader::loadFusedConcat,
       &PyTorchModelLoader::correctTypeAlreadySet},
      {{"glow::fused_stack"},
       &PyTorchModelLoader::loadFusedStack,
       &PyTorchModelLoader::correctTypeAlreadySet},
      {{"glow::fused_broadcast_cat"},
       &PyTorchModelLoader::loadFusedBroadcastConcat,
       &PyTorchModelLoader::correctTypeAlreadySet},
      {{"glow::fused_broadcast_stack"},
       &PyTorchModelLoader::loadFusedBroadcastStack,
       &PyTorchModelLoader::correctTypeAlreadySet},
      {{"aten::floor"},
       UNARY_NODE_LOADER(Floor),
       &PyTorchModelLoader::getCorrectTypeFromInput<0>},
      {{"aten::ceil"},
       UNARY_NODE_LOADER(Ceil),
       &PyTorchModelLoader::getCorrectTypeFromInput<0>},
      {{"aten::mean"},
       &PyTorchModelLoader::loadMean,
       &PyTorchModelLoader::getCorrectTypeFromInput<0>},
      {{"aten::pow"},
       &PyTorchModelLoader::loadPow,
       &PyTorchModelLoader::getCorrectTypeFromInput<0>},
      {{"aten::logical_xor", "aten::logical_xor_"},
       &PyTorchModelLoader::loadLogicalXor,
       &PyTorchModelLoader::correctTypeAlreadySet},
      {{"aten::logical_or", "aten::logical_or_"},
       &PyTorchModelLoader::loadLogicalOr,
       &PyTorchModelLoader::correctTypeAlreadySet},
      {{"aten::logical_and", "aten::logical_and_"},
       &PyTorchModelLoader::loadLogicalAnd,
       &PyTorchModelLoader::correctTypeAlreadySet},
      {{"aten::logical_not", "aten::logical_not_"},
       UNARY_NODE_LOADER(Not),
       &PyTorchModelLoader::getCorrectTypeFromInput<0>},
      {{"aten::bitwise_not"},
       &PyTorchModelLoader::loadBitwiseNot,
       &PyTorchModelLoader::getCorrectTypeFromInput<0>},
      {{"aten::__ixor__", "aten::__xor__", "aten::bitwise_xor",
        "aten::bitwise_xor_"},
       &PyTorchModelLoader::loadBitwiseOp<BitwiseXorNode>,
       &PyTorchModelLoader::correctTypeAlreadySet},
      {{"aten::__ior__", "aten::__or__", "aten::bitwise_or",
        "aten::bitwise_or_"},
       &PyTorchModelLoader::loadBitwiseOp<BitwiseOrNode>,
       &PyTorchModelLoader::correctTypeAlreadySet},
      {{"aten::__iand__", "aten::__and__", "aten::bitwise_and",
        "aten::bitwise_and_"},
       &PyTorchModelLoader::loadBitwiseOp<BitwiseAndNode>,
       &PyTorchModelLoader::correctTypeAlreadySet},
      {{"aten::index_put_", "aten::index_put"},
       &PyTorchModelLoader::loadIndexPut,
       &PyTorchModelLoader::getCorrectTypeFromInput<IndexPutInputs::input>},
      {{"aten::dropout", "aten::dropout_"},
       &PyTorchModelLoader::loadDropout,
       &PyTorchModelLoader::getCorrectTypeFromInput<DropoutInputs::input>},
      {{"aten::sqrt", "aten::sqrt_"},
       &PyTorchModelLoader::loadSqrt,
       &PyTorchModelLoader::getCorrectTypeFromInput<0>},
      {{"aten::le", "aten::le_"},
       &PyTorchModelLoader::loadCmp<CmpLTENode>,
       &PyTorchModelLoader::makeCorrectType<at::kBool>},
      {{"aten::lt", "aten::lt_"},
       &PyTorchModelLoader::loadCmp<CmpLTNode>,
       &PyTorchModelLoader::makeCorrectType<at::kBool>},
      {{"aten::ne", "aten::ne_"},
       &PyTorchModelLoader::loadCmp<CmpNEQNode>,
       &PyTorchModelLoader::makeCorrectType<at::kBool>},
      {{"aten::eq", "aten::eq_"},
       &PyTorchModelLoader::loadCmp<CmpEQNode>,
       &PyTorchModelLoader::makeCorrectType<at::kBool>},
      {{"aten::ge", "aten::ge_"},
       &PyTorchModelLoader::loadCmp<CmpLTENode, true>,
       &PyTorchModelLoader::makeCorrectType<at::kBool>},
      {{"aten::gt", "aten::gt_"},
       &PyTorchModelLoader::loadCmp<CmpLTNode, true>,
       &PyTorchModelLoader::makeCorrectType<at::kBool>},
      {{"aten::clamp"},
       &PyTorchModelLoader::loadClamp,
       &PyTorchModelLoader::getCorrectTypeFromInput<ClampInputs::input>},
      {{"aten::cos"},
       UNARY_NODE_LOADER(Cos),
       &PyTorchModelLoader::getCorrectTypeFromInput<0>},
      {{"aten::sin"},
       UNARY_NODE_LOADER(Sin),
       &PyTorchModelLoader::getCorrectTypeFromInput<0>},
      {{"aten::acos"},
       UNARY_NODE_LOADER(Acos),
       &PyTorchModelLoader::getCorrectTypeFromInput<0>},
      {{"aten::asin"},
       UNARY_NODE_LOADER(Asin),
       &PyTorchModelLoader::getCorrectTypeFromInput<0>},
      {{"aten::atan"},
       UNARY_NODE_LOADER(Atan),
       &PyTorchModelLoader::getCorrectTypeFromInput<0>},
      {{"quantized::add"},
       &PyTorchModelLoader::loadQuantizedAdd,
       &PyTorchModelLoader::getCorrectTypeFromInput<QuantizedAddInputs::lhs>},
      {{"quantized::add_relu"},
       &PyTorchModelLoader::loadQuantizedAddRelu,
       &PyTorchModelLoader::getCorrectTypeFromInput<
           QuantizedAddReluInputs::lhs>},
      {{"quantized::leaky_relu"},
       &PyTorchModelLoader::loadQuantizedLeakyRelu,
       &PyTorchModelLoader::getCorrectTypeFromInput<
           QuantizedLeakyReluInputs::lhs>},
      {{"quantized::mul"},
       &PyTorchModelLoader::loadQuantizedMul,
       &PyTorchModelLoader::getCorrectTypeFromInput<QuantizedMulInputs::lhs>},
      {{"quantized::cat"},
       &PyTorchModelLoader::loadQuantizedCat,
       &PyTorchModelLoader::correctTypeAlreadySet},
      {{"quantized::mul_scalar"},
       &PyTorchModelLoader::loadQuantizedMul,
       &PyTorchModelLoader::getCorrectTypeFromInput<
           QuantizedMulScalarInputs::lhs>},
      {{"glow::fused_linear"},
       &PyTorchModelLoader::loadGlowFusedLinear,
       &PyTorchModelLoader::getCorrectTypeFromInput<
           GlowFusedLinearInputs::input>},
      {{"glow::unpacked_quantized_conv2d"},
       &PyTorchModelLoader::loadQuantizedConvUnpacked,
       &PyTorchModelLoader::getCorrectTypeFromInput<
           QuantizedUnpackedConvInputs::input>},
      {{"glow::unpacked_quantized_conv3d"},
       &PyTorchModelLoader::loadQuantizedConvUnpacked,
       &PyTorchModelLoader::getCorrectTypeFromInput<
           QuantizedUnpackedConvInputs::input>},
      {{"glow::unpacked_quantized_conv3d_relu"},
       &PyTorchModelLoader::loadQuantizedConvReluUnpacked,
       &PyTorchModelLoader::getCorrectTypeFromInput<
           QuantizedUnpackedConvInputs::input>},
      {{"glow::unpacked_quantized_conv2d_relu"},
       &PyTorchModelLoader::loadQuantizedConvReluUnpacked,
       &PyTorchModelLoader::getCorrectTypeFromInput<
           QuantizedUnpackedConvInputs::input>},
      {{"glow::unpacked_quantized_linear",
        "fb::quantized_linear_unpacked_weight",
        "fb::quantized_linear_unpacked_weight_v2"},
       &PyTorchModelLoader::loadQuantizedLinearUnpacked,
       &PyTorchModelLoader::getCorrectTypeFromInput<
           QuantizedUnpackedLinearInputs::input>},
      {{"glow::fused_split"},
       &PyTorchModelLoader::loadFusedSplit,
       &PyTorchModelLoader::correctTypeAlreadySet},
      {{"aten::split"},
       &PyTorchModelLoader::loadSplit,
       &PyTorchModelLoader::correctTypeAlreadySet},
      {{"aten::split_with_sizes"},
       &PyTorchModelLoader::loadSplitWithSizes,
       &PyTorchModelLoader::correctTypeAlreadySet},
      {{"aten::linear"},
       &PyTorchModelLoader::loadLinear,
       &PyTorchModelLoader::getCorrectTypeFromInput<LinearInput::input>},
      {{"quantized::linear", "fb::quantized_linear"},
       &PyTorchModelLoader::loadQuantizedLinear,
       &PyTorchModelLoader::getCorrectTypeFromInput<
           QuantizedLinearInputs::input>},
      {{"quantized::linear_dynamic"},
       &PyTorchModelLoader::loadDynQuantizedLinear,
       &PyTorchModelLoader::getCorrectTypeFromInput<
           DynQuantizedLinearInputs::input>},
      {{"quantized::conv2d"},
       &PyTorchModelLoader::loadQuantizedConv,
       &PyTorchModelLoader::getCorrectTypeFromInput<Conv2DInputs::input>},
      {{"quantized::conv3d"},
       &PyTorchModelLoader::loadQuantizedConv,
       &PyTorchModelLoader::getCorrectTypeFromInput<
           QuantizedConv3dInputs::input>},
      {{"quantized::conv2d_relu"},
       &PyTorchModelLoader::loadQuantizedConvRelu,
       &PyTorchModelLoader::getCorrectTypeFromInput<
           QuantizedConv2dInputs::input>},
      {{"quantized::conv3d_relu"},
       &PyTorchModelLoader::loadQuantizedConvRelu,
       &PyTorchModelLoader::getCorrectTypeFromInput<0>},
      {{"aten::quantize_per_tensor", "fb::quantize_per_tensor"},
       &PyTorchModelLoader::loadQuantize,
       &PyTorchModelLoader::correctTypeAlreadySet},
      {{"aten::dequantize"},
       &PyTorchModelLoader::loadDequantize,
       &PyTorchModelLoader::makeCorrectType<at::kFloat>},
      {{"aten::size"},
       &PyTorchModelLoader::loadSize,
       &PyTorchModelLoader::correctTypeAlreadySet},
      {{"prim::ListConstruct"},
       &PyTorchModelLoader::loadListConstruct,
       &PyTorchModelLoader::correctTypeAlreadySet},
      {{"prim::ListUnpack"},
       &PyTorchModelLoader::loadListUnpack,
       &PyTorchModelLoader::correctTypeAlreadySet},
      {{"aten::reciprocal", "aten::reciprocal_"},
       &PyTorchModelLoader::loadReciprocal,
       &PyTorchModelLoader::getCorrectTypeFromInput<0>},
      {{"aten::adaptive_avg_pool2d"},
       &PyTorchModelLoader::loadAdaptiveAvgPool2d,
       &PyTorchModelLoader::getCorrectTypeFromInput<
           AdaptiveAvgPoolInputs::input>},
      {{"aten::reshape"},
       &PyTorchModelLoader::loadReshape,
       &PyTorchModelLoader::getCorrectTypeFromInput<0>},
      {{"aten::repeat"},
       &PyTorchModelLoader::loadRepeat,
       &PyTorchModelLoader::getCorrectTypeFromInput<0>},
      {{"aten::abs"},
       UNARY_NODE_LOADER(Abs),
       &PyTorchModelLoader::getCorrectTypeFromInput<0>},
      {{"aten::neg"},
       UNARY_NODE_LOADER(Neg),
       &PyTorchModelLoader::getCorrectTypeFromInput<0>},
      {{"aten::upsample_nearest3d"},
       &PyTorchModelLoader::loadUpsampleNearest,
       &PyTorchModelLoader::getCorrectTypeFromInput<0>},
      {{"aten::upsample_nearest2d"},
       &PyTorchModelLoader::loadUpsampleNearest,
       &PyTorchModelLoader::getCorrectTypeFromInput<0>},
      {{"aten::view"},
       &PyTorchModelLoader::loadView,
       &PyTorchModelLoader::getCorrectTypeFromInput<0>},
      {{"aten::_convolution"},
       &PyTorchModelLoader::loadConvolution,
       &PyTorchModelLoader::getCorrectTypeFromInput<Conv2DInputs::input>},
      {{"aten::lstm"},
       &PyTorchModelLoader::loadLSTM,
       &PyTorchModelLoader::correctTypeAlreadySet},
      {{"aten::conv2d"},
       &PyTorchModelLoader::loadConv2D,
       &PyTorchModelLoader::getCorrectTypeFromInput<Conv2DInputs::input>},
      {{"aten::batch_norm"},
       &PyTorchModelLoader::loadBatchNorm,
       &PyTorchModelLoader::getCorrectTypeFromInput<BatchNormInputs::input>},
      {{"aten::norm", "aten::frobenius_norm"},
       &PyTorchModelLoader::loadNorm,
       &PyTorchModelLoader::getCorrectTypeFromInput<0>},
      {{"quantized::batch_norm2d"},
       &PyTorchModelLoader::loadQuantizedBatchNorm2d,
       &PyTorchModelLoader::getCorrectTypeFromInput<
           QuantizedBatchNormInputs::input>},
      {{"quantized::batch_norm3d"},
       &PyTorchModelLoader::loadQuantizedBatchNorm3d,
       &PyTorchModelLoader::getCorrectTypeFromInput<
           QuantizedBatchNormInputs::input>},
      {{"quantized::batch_norm3d_relu"},
       &PyTorchModelLoader::loadQuantizedBatchNorm3dRelu,
       &PyTorchModelLoader::getCorrectTypeFromInput<
           QuantizedBatchNormInputs::input>},
      {{"aten::layer_norm"},
       &PyTorchModelLoader::loadLayerNorm,
       &PyTorchModelLoader::getCorrectTypeFromInput<LayerNormInputs::input>},
      {{"quantized::layer_norm"},
       &PyTorchModelLoader::loadQuantizedLayerNorm,
       &PyTorchModelLoader::getCorrectTypeFromInput<
           QuantizedLayerNormInputs::input>},
      {{"aten::max_pool2d"},
       &PyTorchModelLoader::loadMaxPool2d,
       &PyTorchModelLoader::getCorrectTypeFromInput<MaxPoolInputs::input>},
      {{"aten::avg_pool1d"},
       &PyTorchModelLoader::loadAvgPool<1>,
       &PyTorchModelLoader::getCorrectTypeFromInput<AvgPoolInputs::input>},
      {{"aten::avg_pool2d"},
       &PyTorchModelLoader::loadAvgPool<2>,
       &PyTorchModelLoader::getCorrectTypeFromInput<AvgPoolInputs::input>},
      {{"aten::avg_pool3d"},
       &PyTorchModelLoader::loadAvgPool<3>,
       &PyTorchModelLoader::getCorrectTypeFromInput<AvgPoolInputs::input>},
      {{"aten::matmul"},
       &PyTorchModelLoader::loadMatMul,
       &PyTorchModelLoader::getCorrectTypeFromInput<0>},
      {{"aten::mm"},
       &PyTorchModelLoader::loadMM,
       &PyTorchModelLoader::getCorrectTypeFromInput<0>},
      {{"aten::bmm"},
       &PyTorchModelLoader::loadBmm,
       &PyTorchModelLoader::getCorrectTypeFromInput<0>},
      {{"aten::addmm"},
       &PyTorchModelLoader::loadAddMM,
       &PyTorchModelLoader::getCorrectTypeFromInput<0>},
      {{"aten::select"},
       &PyTorchModelLoader::loadSelect,
       &PyTorchModelLoader::getCorrectTypeFromInput<SelectInputs::input>},
      {{"aten::flatten"},
       &PyTorchModelLoader::loadFlatten,
       &PyTorchModelLoader::getCorrectTypeFromInput<FlattenInputs::input>},
      {{"aten::squeeze", "aten::squeeze_"},
       &PyTorchModelLoader::loadSqueeze,
       &PyTorchModelLoader::getCorrectTypeFromInput<SqueezeInputs::input>},
      {{"aten::unsqueeze", "aten::unsqueeze_"},
       &PyTorchModelLoader::loadUnsqueeze,
       &PyTorchModelLoader::getCorrectTypeFromInput<UnsqueezeInputs::input>},
      {{"fb::unsqueeze_n_times"},
       &PyTorchModelLoader::loadUnsqueezeNTimes,
       &PyTorchModelLoader::getCorrectTypeFromInput<
           UnsqueezeNTimesInputs::input>},
      {{"aten::masked_fill", "aten::masked_fill_"},
       &PyTorchModelLoader::loadMaskedFill,
       &PyTorchModelLoader::getCorrectTypeFromInput<MaskedFillInputs::input>},
      {{"aten::prelu"},
       &PyTorchModelLoader::loadPRelu,
       &PyTorchModelLoader::getCorrectTypeFromInput<PReluInputs::input>},
      {{"aten::slice"},
       &PyTorchModelLoader::loadSlice,
       &PyTorchModelLoader::getCorrectTypeFromInput<SliceInputs::input>},
      {{"aten::softmax"},
       &PyTorchModelLoader::loadSoftMax,
       &PyTorchModelLoader::getCorrectTypeFromInput<SoftMaxInputs::input>},
      {{"aten::log_softmax"},
       &PyTorchModelLoader::loadLogSoftMax,
       &PyTorchModelLoader::getCorrectTypeFromInput<LogSoftMaxInputs::input>},
      {{"aten::topk"},
       &PyTorchModelLoader::loadTopK,
       &PyTorchModelLoader::correctTypeAlreadySet},
      {{"aten::argsort"},
       &PyTorchModelLoader::loadArgSort,
       &PyTorchModelLoader::makeCorrectType<at::kLong>},
      {{"aten::to"},
       &PyTorchModelLoader::loadTo,
       &PyTorchModelLoader::correctTypeAlreadySet},
      {{"prim::ConstantChunk"},
       &PyTorchModelLoader::loadConstantChunk,
       &PyTorchModelLoader::correctTypeAlreadySet},
      {{"aten::embedding"},
       &PyTorchModelLoader::loadEmbedding,
       &PyTorchModelLoader::getCorrectTypeFromInput<EmbeddingInputs::weights>},
      {{"aten::embedding_bag"},
       &PyTorchModelLoader::loadEmbeddingBag,
       &PyTorchModelLoader::getCorrectTypeFromInput<
           EmbeddingBagInputs::weight>},
      {{"fb::simple_embedding_bag_sum"},
       &PyTorchModelLoader::loadEmbeddingBag,
       &PyTorchModelLoader::getCorrectTypeFromInput<
           EmbeddingBagInputs::weight>},
      {{"fb::glow_embedding_bag"},
       &PyTorchModelLoader::loadGlowEmbeddingBag,
       &PyTorchModelLoader::makeCorrectType<at::kFloat>},
      {{"fb::glow_embedding_bag_byte_rowwise_offsets"},
       &PyTorchModelLoader::loadGlowEmbeddingBagByteRowwiseOffsets,
       &PyTorchModelLoader::makeCorrectType<at::kFloat>},
      {{"fb::glow_embedding_bag_4bit_rowwise_offsets"},
       &PyTorchModelLoader::loadGlowEmbeddingBag4bitRowwiseOffsets,
       &PyTorchModelLoader::makeCorrectType<at::kFloat>},
      // Disabled for now since this node needs extra information
      //{{"fb::lengths_range"}, &PyTorchModelLoader::loadLengthsRange},
      {{"fb::xl_embedding_bag"},
       &PyTorchModelLoader::loadXLEmbeddingBag,
       &PyTorchModelLoader::makeCorrectType<at::kFloat>},
      {{"fb::xl_embedding_bag_byte_rowwise_offsets"},
       &PyTorchModelLoader::loadXLEmbeddingBagByteRowwiseOffsets,
       &PyTorchModelLoader::makeCorrectType<at::kFloat>},
      {{"fb::xl_embedding_bag_4bit_rowwise_offsets"},
       &PyTorchModelLoader::loadXLEmbeddingBag4bitRowwiseOffsets,
       &PyTorchModelLoader::makeCorrectType<at::kFloat>},
      {{"quantized::embedding_bag_byte_rowwise_offsets"},
       &PyTorchModelLoader::loadEmbeddingBagByteRowwiseOffsets,
       &PyTorchModelLoader::makeCorrectType<at::kFloat>},
      {{"quantized::embedding_bag_4bit_rowwise_offsets"},
       &PyTorchModelLoader::loadEmbeddingBag4BitRowwiseOffsets,
       &PyTorchModelLoader::makeCorrectType<at::kFloat>},
      {{"fb::embedding_bag_byte_rowwise_offsets"},
       &PyTorchModelLoader::loadEmbeddingBagByteRowwiseOffsets,
       &PyTorchModelLoader::makeCorrectType<at::kFloat>},
      {{"fb::embedding_bag_4bit_rowwise_offsets"},
       &PyTorchModelLoader::loadEmbeddingBag4BitRowwiseOffsets,
       &PyTorchModelLoader::makeCorrectType<at::kFloat>},
      {{"_caffe2::BatchPermutation"},
       &PyTorchModelLoader::loadBatchPermutation,
       &PyTorchModelLoader::getCorrectTypeFromInput<0>},
      {{"fb::fast_gather"},
       &PyTorchModelLoader::loadFastGather,
       &PyTorchModelLoader::getCorrectTypeFromInput<FastGatherInputs::input>},
      {{"aten::gather"},
       &PyTorchModelLoader::loadGatherElements,
       &PyTorchModelLoader::getCorrectTypeFromInput<
           GatherElementsInputs::input>},
      {{"_caffe2::RoIAlign"},
       &PyTorchModelLoader::loadRoiAlign,
       &PyTorchModelLoader::getCorrectTypeFromInput<RoiAlignInputs::features>},
      {{"_caffe2::RoIAlignRotated"},
       &PyTorchModelLoader::loadRoiAlignRotated,
       &PyTorchModelLoader::getCorrectTypeFromInput<RoiAlignInputs::features>},
      {{"_caffe2::BBoxTransform"},
       &PyTorchModelLoader::loadBBoxTransform,
       &PyTorchModelLoader::correctTypeAlreadySet},
      {{"aten::index_select"},
       &PyTorchModelLoader::loadIndexSelect,
       &PyTorchModelLoader::getCorrectTypeFromInput<IndexSelectInputs::input>},
      {{"aten::clamp_min"},
       &PyTorchModelLoader::loadClampMin,
       &PyTorchModelLoader::getCorrectTypeFromInput<ClampMinInputs::input>},
      {{"aten::expand_as"},
       &PyTorchModelLoader::loadExpandAs,
       &PyTorchModelLoader::getCorrectTypeFromInput<ExpandAsInputs::input>},
      {{"aten::expand"},
       &PyTorchModelLoader::loadExpand,
       &PyTorchModelLoader::getCorrectTypeFromInput<ExpandInputs::input>},
      {{"glow::nnckernel"},
       &PyTorchModelLoader::loadNNCKernel,
       &PyTorchModelLoader::correctTypeAlreadySet},
      {{"aten::cumsum"},
       &PyTorchModelLoader::loadCumSum,
       &PyTorchModelLoader::correctTypeAlreadySet},
      {{"fb::equally_split"},
       &PyTorchModelLoader::loadEquallySplit,
       &PyTorchModelLoader::correctTypeAlreadySet},
      {{"fb::expand_dims"},
       &PyTorchModelLoader::loadExpandDims,
       &PyTorchModelLoader::correctTypeAlreadySet},
      {{"aten::narrow"},
       &PyTorchModelLoader::loadNarrow,
       &PyTorchModelLoader::correctTypeAlreadySet},
      {{"aten::pixel_shuffle"},
       &PyTorchModelLoader::loadPixelShuffle,
       &PyTorchModelLoader::getCorrectTypeFromInput<PixelShuffleInputs::input>},
      {{"aten::pixel_unshuffle"},
       &PyTorchModelLoader::loadPixelUnshuffle,
       &PyTorchModelLoader::getCorrectTypeFromInput<
           PixelUnshuffleInputs::input>},
      {{"aten::square"},
       &PyTorchModelLoader::loadSquare,
       &PyTorchModelLoader::getCorrectTypeFromInput<0>},
      {{"fb::scale_gradient"},
       &PyTorchModelLoader::loadScaleGradient,
       &PyTorchModelLoader::getCorrectTypeFromInput<0>},
      {{"aten::erf"},
       &PyTorchModelLoader::loadErf,
       &PyTorchModelLoader::getCorrectTypeFromInput<0>},
      {{"fbgemm_gpu::batched_unary_embeddings",
        "fbgemm::batched_unary_embeddings"},
       &PyTorchModelLoader::loadBatchedUnaryEmbeddingsBags,
       &PyTorchModelLoader::getCorrectTypeFromInput<
           BatchedUnaryEmbeddingsBagsInputs::weights>},
      {{"aten::sign"},
       &PyTorchModelLoader::loadSign,
       &PyTorchModelLoader::getCorrectTypeFromInput<0>},
      {{"aten::index_add_", "aten::index_add"},
       &PyTorchModelLoader::loadIndexAdd,
       &PyTorchModelLoader::getCorrectTypeFromInput<0>},
      {{"fb::int_nbit_split_embedding_codegen_lookup_function"},
       &PyTorchModelLoader::loadIntNBitSplitEmbeddingBags,
       &PyTorchModelLoader::getCorrectTypeFromInput<0>},
      {{"aten::amax"},
       &PyTorchModelLoader::loadAmax,
       &PyTorchModelLoader::getCorrectTypeFromInput<0>},
  });
#undef UNARY_NODE_LOADER

  // Add in custom operator loaders.
  for (const auto &symbolAndLoader : getCustomPyTorchOpLoaders()) {
    const char *symbolStr = symbolAndLoader.first.toQualString();
    MappingOfMemberFunctionsValue val(
        {symbolStr}, &PyTorchModelLoader::loadCustomOp,
        &PyTorchModelLoader::getCustomOpCorrectType);
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
  /// for loading these symbols, and the set of inputs that should be
  /// considered immutable between inference invocations by Glow and loaded as
  /// Constants instead of Placeholders.
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
      // Keep the original correct type
      at::ScalarType correctType;
      ASSIGN_VALUE_OR_RETURN_ERR(correctType, getCorrectTypeMapping(value));

      // Remove the old NodeValue mapping
      RETURN_IF_ERR(removeValueMapping(value));

      // Assign the new Constant NodeValue, make sure it has same correct type
      RETURN_IF_ERR(addValueMapping(value, glowConstant->getOutput()));
      RETURN_IF_ERR(setCorrectTypeMapping(value, correctType));
    }
  }

  // Remove tmp stuff and return success
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

    // Correct type(s) for each output. Usually each output has 1 correct type
    // unless it's a list of Tensors then it can have multiple correct types,
    // one for each tensor.
    std::vector<std::vector<at::ScalarType>> allCorrectTypes;
    ASSIGN_VALUE_OR_RETURN_ERR(allCorrectTypes,
                               (this->*it->second.correctTypeFn)(node));

    // Set correct types mapping for all the output we got correct types for
    for (size_t i = 0; i < allCorrectTypes.size(); ++i) {
      auto output = node->output(i);
      // Skip outputs we didn't load
      if (!valueMap_.count(output)) {
        continue;
      }
      const auto &correctTypes = allCorrectTypes[i];
      RETURN_IF_ERR(setCorrectTypesMapping(output, correctTypes));
    }

    // Verify all outputs that are mapped have valid correct types
    for (size_t i = 0; i < node->outputs().size(); ++i) {
      auto output = node->output(i);

      // Skip outputs we didn't load
      if (!valueMap_.count(output)) {
        continue;
      }

      if (auto err = valueMap_.at(output).verifyCorrectTypes()) {
        ADD_MESSAGE_TO_ERR_STACK(err, strFormat("Failed on output %d", int(i)));
        RETURN_ERR(err);
      }
    }

    // Do constant propagation
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
        // Run glowNode and remap it result as a constant node as node's
        // output.
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

  // Indentation
  auto errStringIndent = [](int layer, std::string &err) {
    if (!err.empty()) {
      err.push_back('\n');
    }
    for (int i = 0; i < layer; ++i) {
      err.push_back(' ');
    }
  };

  // If loadNode returns an error, following lambda will traverse
  // through parents and get node information upto debugLayers(configurable)
  std::function<void(const torch::jit::Node *, std::string &, int,
                     std::unordered_set<const torch::jit::Node *> &)>
      errStack = [&](const torch::jit::Node *nnode, std::string &err, int layer,
                     std::unordered_set<const torch::jit::Node *> &traversed) {
        if (layer <= 0 || nnode == nullptr || nnode->inputs().size() == 0 ||
            !traversed.insert(nnode).second) {
          return;
        }

        for (const auto parent : nnode->inputs()) {
          if (parent == nullptr) {
            continue;
          }

          errStack(parent->node(), err, layer - 1, traversed);
        }

        errStringIndent(layer, err);
        err.append(jitNodeToString(nnode));
      };

  // Nodes are topologically sorted.
  for (const auto *node : graph.nodes()) {
    VLOG(1) << "Loading node: " << jitNodeToString(node).c_str();
    if (auto err = loadNode(node)) {

      std::string errString;
      std::unordered_set<const torch::jit::Node *> traversed;
      if (settings_.debugLayers > 0) {
        errStack(node, errString, settings_.debugLayers, traversed);
      }

      ADD_MESSAGE_TO_ERR_STACK(err, errString.c_str());
      RETURN_ERR(err);
    }
  }

  return Error::success();
}

Expected<at::ScalarType>
PyTorchModelLoader::getCorrectTypeMapping(const torch::jit::Value *src) {
  auto it = valueMap_.find(src);
  RETURN_ERR_IF_NOT(
      it != valueMap_.end(),
      glow::strFormat(
          "Cannot find value %s when trying to get its correct type",
          src->debugNameBase().c_str()));
  return it->second.getCorrectType();
}

Expected<std::vector<at::ScalarType>>
PyTorchModelLoader::getCorrectTypesMapping(const torch::jit::Value *src) {
  auto it = valueMap_.find(src);
  RETURN_ERR_IF_NOT(
      it != valueMap_.end(),
      glow::strFormat(
          "Cannot find value %s when trying to get its correct type",
          src->debugNameBase().c_str()));
  return it->second.getCorrectTypes();
}

Error PyTorchModelLoader::setCorrectTypeMapping(const torch::jit::Value *src,
                                                at::ScalarType correctType) {
  auto it = valueMap_.find(src);
  RETURN_ERR_IF_NOT(
      it != valueMap_.end(),
      glow::strFormat(
          "Cannot find value %s when trying to set its correct type",
          src->debugNameBase().c_str()));
  RETURN_IF_ERR(it->second.setCorrectType(correctType));
  return Error::success();
}

Error PyTorchModelLoader::setCorrectTypesMapping(
    const torch::jit::Value *src,
    const std::vector<at::ScalarType> &correctTypes) {
  auto it = valueMap_.find(src);
  RETURN_ERR_IF_NOT(
      it != valueMap_.end(),
      glow::strFormat(
          "Cannot find value %s when trying to set its correct type",
          src->debugNameBase().c_str()));
  RETURN_IF_ERR(it->second.setCorrectTypes(correctTypes));
  return Error::success();
}

Error PyTorchModelLoader::setCorrectTypeMappingSameAs(
    const torch::jit::Value *src, const torch::jit::Value *other) {
  std::vector<at::ScalarType> otherCorrectTypes;
  ASSIGN_VALUE_OR_RETURN_ERR(otherCorrectTypes, getCorrectTypesMapping(other));
  RETURN_IF_ERR(setCorrectTypesMapping(src, otherCorrectTypes));
  return Error::success();
}

Error PyTorchModelLoader::addValueMapping(const torch::jit::Value *value,
                                          glow::NodeValue nodeValue) {
  ValueMapping mapping(nodeValue);
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
                                          glow::GlowIValue glowIValue) {
  glow::Constant *glowConstant = nullptr;
  if (glowIValue.isTensor()) {
    glow::Tensor *t;
    ASSIGN_VALUE_OR_RETURN_ERR(t, glowIValue.toTensor());
    glowConstant = F_.getParent()->createConstant("constant", std::move(*t));
    RETURN_IF_ERR(addValueMapping(value, glowConstant->getOutput()));
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

template <typename T>
Error PyTorchModelLoader::extractConstantFromNodeValue(
    const torch::jit::Value *value, glow::ElemKind elemKind, T &output) {
  glow::NodeValue nodeValue;
  ASSIGN_VALUE_OR_RETURN_ERR(nodeValue, getGlowNodeValueForValue(value));
  auto elementType = nodeValue.getType()->getElementType();
  size_t size = nodeValue.getType()->size();
  RETURN_ERR_IF_NOT(nodeValue.getNode()->getKind() ==
                        Kinded::Kind::ConstantKind,
                    "Expect scalar or constant value.");
  RETURN_ERR_IF_NOT(elementType == elemKind,
                    strFormat("Expected element type is %d, found %d.",
                              static_cast<int>(elemKind),
                              static_cast<int>(elementType)));
  RETURN_ERR_IF_NOT(size == 1, "Expect constant to be a scalar.");
  glow::Constant *constant =
      llvm::dyn_cast<glow::Constant>(nodeValue.getNode());
  RETURN_ERR_IF_NOT(constant != nullptr, "constant is null.");
  output = constant->getPayload().getHandle<T>().raw(0);
  return Error::success();
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

    // Quantize the filter automatically (only if it is float). The bias is
    // NOT quantized automatically and is left at the disposal of each Backend
    // to quantize it later using custom logic.
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
    auto NV = EXIT_ON_ERR(getGlowNodeValueForValue(value));
    CHECK(NV.getType()->isEqual(ty))
        << "Found NodeValue with type " << NV.getType()->toString()
        << ", expect " << ty.toString();
    return NV;
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

  auto broadcasted = F_.broadcastInputs(/* axis */ -1, {lhs, rhs});
  lhs = broadcasted[0];
  rhs = broadcasted[1];

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

  RETURN_ERR(addValueMapping(outputs[0], output));
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

  RETURN_ERR(addValueMapping(outputs[0], output));
}

Error PyTorchModelLoader::loadQuantizedLeakyRelu(
    const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 5, outputs, 1));

  glow::NodeValue lhs;
  ASSIGN_VALUE_OR_RETURN_ERR(
      lhs, getGlowNodeValueForValue(inputs[QuantizedLeakyReluInputs::lhs]));

  // negative_slope
  float alphaVal;
  ASSIGN_VALUE_OR_RETURN_ERR(alphaVal,
                             iValToDouble(getGlowIValueForValue(
                                 inputs[QuantizedLeakyReluInputs::alpha])));

  // inplace
  bool inplace;
  ASSIGN_VALUE_OR_RETURN_ERR(inplace,
                             iValToBool(getGlowIValueForValue(
                                 inputs[QuantizedLeakyReluInputs::inplace])));

  // scale
  float outScale;
  ASSIGN_VALUE_OR_RETURN_ERR(outScale,
                             iValToDouble(getGlowIValueForValue(
                                 inputs[QuantizedLeakyReluInputs::scale])));

  // zero_point
  int32_t outOffset;
  ASSIGN_VALUE_OR_RETURN_ERR(
      outOffset, iValToInt(getGlowIValueForValue(
                     inputs[QuantizedLeakyReluInputs::zero_point])));

  TypeRef inputType = lhs.getType();
  auto outDims = inputType->dims();
  auto outTy = F_.getParent()->uniqueType(ElemKind::Int8QTy, outDims, outScale,
                                          outOffset - UINT8_TO_INT8_SHIFT);

  glow::LeakyReluNode *qleakyrelu =
      F_.createLeakyRELU("quantized_leaky_relu", lhs, alphaVal);
  auto output = qleakyrelu->getResult();

  RETURN_ERR(addValueMapping(outputs[0], output));
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
    RETURN_ERR(addValueMapping(outputs[0], output));
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
    auto broadcasted = F_.broadcastInputs(/* axis */ -1, {lhs, rhs});
    lhs = broadcasted[0];
    rhs = broadcasted[1];
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
    glow::MulNode *qmul = F_.createMul("quantized_mul", outTy, lhs, rhs);
    auto output = qmul->getResult();

    RETURN_ERR(addValueMapping(outputs[0], output));
  }
}
Error PyTorchModelLoader::loadQuantizedCat(const torch::jit::Node *ptNode) {
  // Current strategy for quantized::cat is dequantize->cat->requantize.
  // TODO: Remove the quantization step to potentially improve performance.

  RETURN_IF_ERR(
      checkInputAndOutputSizes(ptNode->inputs(), -2, ptNode->outputs(), 1));

  std::vector<glow::NodeValue> *inputTensors;
  float quantizationScale;
  int32_t quantizationOffset;
  int64_t concatDimension;

  ASSIGN_VALUE_OR_RETURN_ERR(
      inputTensors,
      iValToNodeValueList(getGlowIValueForValue(ptNode->input(0))));
  ASSIGN_VALUE_OR_RETURN_ERR(
      quantizationScale, iValToDouble(getGlowIValueForValue(ptNode->input(2))));
  ASSIGN_VALUE_OR_RETURN_ERR(
      quantizationOffset, iValToInt(getGlowIValueForValue(ptNode->input(3))));
  ASSIGN_VALUE_OR_RETURN_ERR(
      concatDimension, iValToInt(getGlowIValueForValue(ptNode->input(1))));

  std::vector<glow::NodeValue> dequantizedInputs;
  for (glow::NodeValue input : *inputTensors) {
    // Legacy behavior suggests supporting concat of empty tensors, but only
    // for this specific shape. See the legacy_cat_wrap_dim function in
    // caffe2/aten/src/ATen/WrapDimUtils.h for more info.
    if (input.dims() == llvm::ArrayRef<dim_t>({0})) {
      continue;
    }
    dequantizedInputs.emplace_back(F_.createDequantize(
        "quantized_cat_dequantize", input, ElemKind::FloatTy));
  }

  glow::NodeValue concatResult =
      F_.createConcat("quantized_cat_nested_cat", dequantizedInputs,
                      concatDimension)
          ->getResult();

  auto *outputTy = F_.getParent()->uniqueType(
      ElemKind::Int8QTy, concatResult.dims(), quantizationScale,
      quantizationOffset - UINT8_TO_INT8_SHIFT);

  auto quantizedResult =
      F_.createQuantize("quantized_cat_requantize", concatResult, outputTy);

  RETURN_IF_ERR(addValueMapping(ptNode->output(0), quantizedResult));

  // Set the correct type of the output to be the same as the first element of
  // the list of inputs.
  std::vector<at::ScalarType> inputListCorrectTypes;
  ASSIGN_VALUE_OR_RETURN_ERR(inputListCorrectTypes,
                             getCorrectTypesMapping(ptNode->input(0)));
  RETURN_IF_ERR(
      setCorrectTypeMapping(ptNode->output(0), inputListCorrectTypes.at(0)));

  return Error::success();
}

Error PyTorchModelLoader::loadLinear(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 3, outputs, 1));

  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(
      input, getGlowNodeValueForValue(inputs[LinearInput::input]));

  // Flatten outer dims if necessary
  auto inputDims = input.dims();
  if (inputDims.size() > 2) {
    input = F_.createFlatten("flatten", input, inputDims.size() - 1);
  }

  // Expand dims if necessary
  if (inputDims.size() == 1) {
    input = F_.createReshape("reshape", input, {1, inputDims[0]});
  }

  glow::NodeValue weight;
  ASSIGN_VALUE_OR_RETURN_ERR(
      weight, getGlowNodeValueForValue(inputs[LinearInput::weight]));
  weight = F_.createTranspose("weight_transpose", weight, {1, 0});

  // Get bias or create a zero bias if no bias is found.
  glow::NodeValue bias = loadNodeValueOrCreateBroadcastedConstant(
      inputs[LinearInput::bias], "linear_bias",
      glow::Type(ElemKind::FloatTy, {weight.dims()[1]}), 0.0);

  NodeValue output = F_.createFullyConnected("linear", input, weight, bias);

  // Restore original outer dims
  if (inputDims.size() > 2) {
    std::vector<dim_t> finalDims = inputDims.vec();
    finalDims.back() = output.dims().back();
    output = F_.createReshape("linear_reshape", output, finalDims);
  }

  if (inputDims.size() == 1) {
    output = F_.createReshape("linear_reshape", output, {output.dims()[1]});
  }

  RETURN_ERR(addValueMapping(outputs[0], output));
}

// implementation for per_tensor and per_channel quantized linear from either
// packed or unpacked linear
Error PyTorchModelLoader::loadQuantizedLinearImpl(
    NodeValue input, NodeValue weights, NodeValue bias, NodeValue wScales,
    NodeValue wOffsets, float outScale, int64_t outZeroPoint,
    const torch::jit::Value *outputValue) {
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
    output = F_.createReshape("fc_reshape", output, finalDims);
  }

  RETURN_ERR(addValueMapping(outputValue, output));
}

Error PyTorchModelLoader::loadDynQuantizedLinear(
    const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 3, outputs, 1));

  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(
      input, getGlowNodeValueForValue(inputs[DynQuantizedLinearInputs::input]));

  RETURN_ERR_IF_NOT(
      qparamsMap_.count(inputs[DynQuantizedLinearInputs::packed_weights]),
      "Cannot find packed weights for DQFC");
  auto packedParams =
      qparamsMap_[inputs[DynQuantizedLinearInputs::packed_weights]]
          .toCustomClass<LinearPackedParamsBase>();

  at::Tensor ptWeightTensor;
  c10::optional<at::Tensor> ptBiasTensorTmp;
  std::tie(ptWeightTensor, ptBiasTensorTmp) = packedParams->unpack();

  // unpacked weights
  auto weightTensor = ptTensorToGlowTensor(ptWeightTensor);
  glow::Constant *weightConstant = F_.getParent()->createConstant(
      "dynamic_quantized_linear_weights", std::move(weightTensor));
  weightConstant->ensureIsOwned();
  RETURN_ERR_IF_NOT(weightConstant->dims().size() == 2,
                    "Expected 2d Linear weights");
  auto weights = weightConstant->getOutput();
  weights = F_.createTranspose("dynamic_quantized_weights_transposed", weights,
                               {1, 0});

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
      "dynamic_quantized_linear_bias", std::move(biasTensor));
  biasConstant->ensureIsOwned();
  auto bias = biasConstant->getOutput();

  bool isRowwiseQuantized = ptWeightTensor.is_quantized() &&
                            ptWeightTensor.qscheme() == at::kPerChannelAffine;

  // Flatten outer dims if necessary
  auto inputDims = input.dims();
  if (inputDims.size() > 2) {
    input = F_.createFlatten("flatten", input, inputDims.size() - 1);
  }

  NodeValue output;
  NodeValue wScales, wOffsets;
  if (isRowwiseQuantized) {
    std::tie(wScales, wOffsets) = extractChannelwiseQParams(F_, ptWeightTensor);
    auto fc = F_.createDynamicRowwiseQuantizedFullyConnected(
        "dynamic_rowwise_quantized_fc", input, weights, bias, wScales,
        wOffsets);
    output = fc->getResult();
  } else {
    auto fc = F_.createDynamicQuantizedFullyConnected("dynamic_quantized_fc",
                                                      input, weights, bias);
    output = fc->getResult();
  }
  // Restore original outer dims
  if (inputDims.size() > 2) {
    std::vector<dim_t> finalDims = inputDims.vec();
    finalDims.back() = output.dims().back();
    output = F_.createReshape("fc_reshape", output, finalDims);
  }

  RETURN_ERR(addValueMapping(outputs[0], output));
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
  if (hasGlowIValueForValue(inputs[QuantizedLinearInputs::scale])) {
    ASSIGN_VALUE_OR_RETURN_ERR(outScale,
                               to32Bit(iValToDouble(getGlowIValueForValue(
                                   inputs[QuantizedLinearInputs::scale]))));
  } else {
    float scaleConstant;
    RETURN_IF_ERR(extractConstantFromNodeValue<float>(
        inputs[QuantizedLinearInputs::scale], glow::ElemKind::FloatTy,
        scaleConstant));
    ASSIGN_VALUE_OR_RETURN_ERR(outScale, to32Bit((double)scaleConstant));
  }

  int64_t outZeroPoint;
  if (hasGlowIValueForValue(inputs[QuantizedLinearInputs::zero_point])) {
    ASSIGN_VALUE_OR_RETURN_ERR(outZeroPoint,
                               iValToInt(getGlowIValueForValue(
                                   inputs[QuantizedLinearInputs::zero_point])));
  } else {
    int32_t zeroPointConstant;
    RETURN_IF_ERR(extractConstantFromNodeValue<int32_t>(
        inputs[QuantizedLinearInputs::zero_point], glow::ElemKind::Int32ITy,
        zeroPointConstant));
    outZeroPoint = (int64_t)zeroPointConstant;
  }

  bool isRowwiseQuantized = ptWeightTensor.is_quantized() &&
                            ptWeightTensor.qscheme() == at::kPerChannelAffine;

  NodeValue wScales, wOffsets;
  if (isRowwiseQuantized) {
    std::tie(wScales, wOffsets) = extractChannelwiseQParams(F_, ptWeightTensor);
  }

  return loadQuantizedLinearImpl(input, weights, bias, wScales, wOffsets,
                                 outScale, outZeroPoint, outputs[0]);
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
  if (hasGlowIValueForValue(inputs[QuantizedUnpackedLinearInputs::scale])) {
    ASSIGN_VALUE_OR_RETURN_ERR(
        outScale, to32Bit(iValToDouble(getGlowIValueForValue(
                      inputs[QuantizedUnpackedLinearInputs::scale]))));
  } else {
    float scaleConstant;
    RETURN_IF_ERR(extractConstantFromNodeValue<float>(
        inputs[QuantizedUnpackedLinearInputs::scale], glow::ElemKind::FloatTy,
        scaleConstant));
    ASSIGN_VALUE_OR_RETURN_ERR(outScale, to32Bit((double)scaleConstant));
  }

  int64_t outZeroPoint;
  if (hasGlowIValueForValue(
          inputs[QuantizedUnpackedLinearInputs::zero_point])) {
    ASSIGN_VALUE_OR_RETURN_ERR(
        outZeroPoint, iValToInt(getGlowIValueForValue(
                          inputs[QuantizedUnpackedLinearInputs::zero_point])));
  } else {
    int32_t zeroPointConstant;
    RETURN_IF_ERR(extractConstantFromNodeValue<int32_t>(
        inputs[QuantizedUnpackedLinearInputs::zero_point],
        glow::ElemKind::Int32ITy, zeroPointConstant));
    outZeroPoint = (int64_t)zeroPointConstant;
  }

  // Get bias or create a zero bias if no bias is found.
  glow::NodeValue bias = loadNodeValueOrCreateBroadcastedConstant(
      inputs[QuantizedUnpackedLinearInputs::bias], "quantized_linear_bias",
      glow::Type(ElemKind::FloatTy, {weights.dims()[0]}), 0.0);

  // Choose bias quantization params and quantize it.
  glow::Constant *biasConstant = llvm::dyn_cast<glow::Constant>(bias.getNode());

  auto ptWeightTensor =
      qparamsMap_.at(inputs[QuantizedUnpackedLinearInputs::weight]).toTensor();

  bool isRowwiseQuantized = ptWeightTensor.is_quantized() &&
                            ptWeightTensor.qscheme() == at::kPerChannelAffine;

  NodeValue wScales, wOffsets;
  if (isRowwiseQuantized) {
    std::tie(wScales, wOffsets) = extractChannelwiseQParams(F_, ptWeightTensor);
  }

  return loadQuantizedLinearImpl(input, weights, bias, wScales, wOffsets,
                                 outScale, outZeroPoint, outputs[0]);
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

template <int64_t inputs_size, bool broadcast>
Error PyTorchModelLoader::loadCopy(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, inputs_size, outputs, 1));

  glow::NodeValue res;

  if (broadcast) {
    glow::NodeValue input;
    ASSIGN_VALUE_OR_RETURN_ERR(input, getGlowNodeValueForValue(inputs[0]));

    glow::NodeValue other;
    ASSIGN_VALUE_OR_RETURN_ERR(other, getGlowNodeValueForValue(inputs[1]));

    auto bn = F_.createBroadcast("copy_broadcast_", other, input.dims(),
                                 input.dims().size() - other.dims().size());
    res = bn->getNthResult(0);
  } else {
    ASSIGN_VALUE_OR_RETURN_ERR(res, getGlowNodeValueForValue(inputs[0]));
  }

  RETURN_ERR(addValueMapping(outputs[0], res));
}

template <size_t from_input_index>
Expected<std::vector<std::vector<at::ScalarType>>>
PyTorchModelLoader::getCorrectTypeFromInput(const torch::jit::Node *ptNode) {
  std::vector<std::vector<at::ScalarType>> outputTypes;

  at::ScalarType correctType;
  ASSIGN_VALUE_OR_RETURN_ERR(
      correctType, getCorrectTypeMapping(ptNode->input(from_input_index)));
  outputTypes.push_back({correctType});

  return outputTypes;
}

template <at::ScalarType scalar_type>
Expected<std::vector<std::vector<at::ScalarType>>>
PyTorchModelLoader::makeCorrectType(const torch::jit::Node *) {
  std::vector<std::vector<at::ScalarType>> outputTypes;
  outputTypes.push_back({scalar_type});
  return outputTypes;
}

template <at::ScalarType scalar_type_one, at::ScalarType scalar_type_two>
Expected<std::vector<std::vector<at::ScalarType>>>
PyTorchModelLoader::makeCorrectTypes(const torch::jit::Node *) {
  std::vector<std::vector<at::ScalarType>> outputTypes;
  outputTypes.push_back({scalar_type_one});
  outputTypes.push_back({scalar_type_two});
  return outputTypes;
}

template <at::ScalarType scalar_type_one, at::ScalarType scalar_type_two,
          at::ScalarType scalar_type_three>
Expected<std::vector<std::vector<at::ScalarType>>>
PyTorchModelLoader::makeCorrectTypes(const torch::jit::Node *) {
  std::vector<std::vector<at::ScalarType>> outputTypes;
  outputTypes.push_back({scalar_type_one});
  outputTypes.push_back({scalar_type_two});
  outputTypes.push_back({scalar_type_three});
  return outputTypes;
}

Expected<std::vector<std::vector<at::ScalarType>>>
PyTorchModelLoader::correctTypeAlreadySet(const torch::jit::Node *) {
  std::vector<std::vector<at::ScalarType>> outputTypes;
  return outputTypes;
}

template <typename GlowNode>
Expected<std::pair<NodeValue, at::ScalarType>>
PyTorchModelLoader::loadArithmeticNode(llvm::StringRef name,
                                       const torch::jit::Value *lhs,
                                       const torch::jit::Value *rhs,
                                       bool convertToDefaultType) {
  glow::NodeValue lhsInput;
  glow::NodeValue rhsInput;

  at::ScalarType correctType;

  if (hasGlowNodeValueForValue(lhs)) {
    ASSIGN_VALUE_OR_RETURN_ERR(lhsInput, getGlowNodeValueForValue(lhs));
    ASSIGN_VALUE_OR_RETURN_ERR(correctType, getCorrectTypeMapping(lhs));
    ASSIGN_VALUE_OR_RETURN_ERR(
        rhsInput, loadNodeValueOrBroadcastedIValue(rhs, lhsInput.getType()));
  } else if (hasGlowNodeValueForValue(rhs)) {
    ASSIGN_VALUE_OR_RETURN_ERR(rhsInput, getGlowNodeValueForValue(rhs));
    ASSIGN_VALUE_OR_RETURN_ERR(correctType, getCorrectTypeMapping(rhs));
    ASSIGN_VALUE_OR_RETURN_ERR(
        lhsInput, loadNodeValueOrBroadcastedIValue(lhs, rhsInput.getType()));
  } else {
    return MAKE_ERR("Either lhs or rhs of arithmetic node must be a tensor");
  }

  // For aten::div, it will promote the output to default scalar type if both
  // inputs are of integer type. However, Glow requires inputs and output have
  // the same type. In order to achieve same behavior as Pytorch div, we
  // convert the inputs to default scalar type if they are both integer.
  if (convertToDefaultType) {
    if (isNonQuantizedIntElemKind(rhsInput.getElementType()) &&
        isNonQuantizedIntElemKind(lhsInput.getElementType())) {
      auto defaultScalarType =
          at::typeMetaToScalarType(at::get_default_dtype());
      auto glowElemKind = scalarTypeToElemKind(defaultScalarType);

      correctType = defaultScalarType;

      lhsInput = F_.createConvertTo(
          "lhs_to", lhsInput,
          F_.getParent()->uniqueType(glowElemKind, lhsInput.getType()->dims()));
      rhsInput = F_.createConvertTo(
          "rhs_to", rhsInput,
          F_.getParent()->uniqueType(glowElemKind, rhsInput.getType()->dims()));
    }
  }

  std::pair<NodeValue, at::ScalarType> pp = {
      F_.createNodeWithBroadcast<GlowNode>(name.str(), /*axis*/ -1, lhsInput,
                                           rhsInput)
          ->getNthResult(0),
      correctType};

  return pp;
}

Error PyTorchModelLoader::loadMul(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 2, outputs, 1));

  std::pair<NodeValue, at::ScalarType> nodeValueAndCorrectType;
  ASSIGN_VALUE_OR_RETURN_ERR(
      nodeValueAndCorrectType,
      loadArithmeticNode<glow::MulNode>("mul", inputs[0], inputs[1]));

  RETURN_IF_ERR(addValueMapping(outputs[0], nodeValueAndCorrectType.first));
  RETURN_IF_ERR(
      setCorrectTypeMapping(outputs[0], nodeValueAndCorrectType.second));
  return Error::success();
}

Error PyTorchModelLoader::loadDiv(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();

  // div should take at least two arguments
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, -2, outputs, 1));

  // TODO: implement 3rd argument rounding_mode option
  LOG_IF(WARNING, inputs.size() > 2)
      << "Ignoring rounding argument for aten::div";

  std::pair<NodeValue, at::ScalarType> nodeValueAndCorrectType;
  ASSIGN_VALUE_OR_RETURN_ERR(
      nodeValueAndCorrectType,
      loadArithmeticNode<glow::DivNode>("div", inputs[0], inputs[1],
                                        /* convertToDefaultType */ true));

  RETURN_IF_ERR(addValueMapping(outputs[0], nodeValueAndCorrectType.first));
  RETURN_IF_ERR(
      setCorrectTypeMapping(outputs[0], nodeValueAndCorrectType.second));
  return Error::success();
}

Error PyTorchModelLoader::loadFloorDiv(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 2, outputs, 1));

  auto lhs = inputs[0];
  auto rhs = inputs[1];
  glow::NodeValue lhsInput;
  glow::NodeValue rhsInput;

  at::ScalarType correctType;

  if (hasGlowNodeValueForValue(lhs)) {
    ASSIGN_VALUE_OR_RETURN_ERR(lhsInput, getGlowNodeValueForValue(lhs));
    ASSIGN_VALUE_OR_RETURN_ERR(
        rhsInput, loadNodeValueOrBroadcastedIValue(rhs, lhsInput.getType()));
    ASSIGN_VALUE_OR_RETURN_ERR(correctType, getCorrectTypeMapping(lhs));
  } else if (hasGlowNodeValueForValue(rhs)) {
    ASSIGN_VALUE_OR_RETURN_ERR(rhsInput, getGlowNodeValueForValue(rhs));
    ASSIGN_VALUE_OR_RETURN_ERR(
        lhsInput, loadNodeValueOrBroadcastedIValue(lhs, rhsInput.getType()));
    ASSIGN_VALUE_OR_RETURN_ERR(correctType, getCorrectTypeMapping(rhs));
  } else {
    return MAKE_ERR("Either lhs or rhs of floorDiv node must be a tensor");
  }

  // Current Pytorch FloorDiv is actually TruncDiv
  // github.com/pytorch/pytorch/issues/43874
  auto res = F_.createFloorDivWithBroadcast(
      "floor_divide", /* axis */ -1, lhsInput, rhsInput, /* truncate */ true);
  RETURN_IF_ERR(addValueMapping(outputs[0], res));
  RETURN_IF_ERR(setCorrectTypeMapping(outputs[0], correctType));
  return Error::success();
}

Error PyTorchModelLoader::loadFmod(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 2, outputs, 1));

  std::pair<NodeValue, at::ScalarType> nodeValueAndCorrectType;
  ASSIGN_VALUE_OR_RETURN_ERR(
      nodeValueAndCorrectType,
      loadArithmeticNode<glow::FmodNode>("fmod", inputs[0], inputs[1],
                                         /* convertToDefaultType */
                                         true));

  // if both input tensors are of type int/long
  // then output tensor should have same data type as input
  glow::NodeValue lhsInput;
  at::ScalarType lhsType;
  ASSIGN_VALUE_OR_RETURN_ERR(lhsInput, getGlowNodeValueForValue(inputs[0]));
  ASSIGN_VALUE_OR_RETURN_ERR(lhsType, getCorrectTypeMapping(inputs[0]));

  glow::NodeValue rhsInput;
  at::ScalarType rhsType;
  if (hasGlowNodeValueForValue(inputs[1])) {
    ASSIGN_VALUE_OR_RETURN_ERR(rhsInput, getGlowNodeValueForValue(inputs[1]));
    ASSIGN_VALUE_OR_RETURN_ERR(rhsType, getCorrectTypeMapping(inputs[1]));
  } else {
    ASSIGN_VALUE_OR_RETURN_ERR(rhsInput, loadNodeValueOrBroadcastedIValue(
                                             inputs[1], lhsInput.getType()));
    ASSIGN_VALUE_OR_RETURN_ERR(rhsType, getCorrectTypeMapping(inputs[0]));
  }

  if (isNonQuantizedIntElemKind(lhsInput.getElementType()) &&
      isNonQuantizedIntElemKind(rhsInput.getElementType())) {
    at::ScalarType resultType =
        ((lhsType == rhsType) && (lhsType == at::kInt)) ? lhsType : at::kLong;
    auto glowElemKind = scalarTypeToElemKind(resultType);
    auto result = F_.createConvertTo(
        "result_to", nodeValueAndCorrectType.first,
        F_.getParent()->uniqueType(glowElemKind, lhsInput.getType()->dims()));
    RETURN_IF_ERR(addValueMapping(outputs[0], result));
    RETURN_IF_ERR(setCorrectTypeMapping(outputs[0], resultType));
  } else {
    RETURN_IF_ERR(addValueMapping(outputs[0], nodeValueAndCorrectType.first));
    RETURN_IF_ERR(
        setCorrectTypeMapping(outputs[0], nodeValueAndCorrectType.second));
  }
  return Error::success();
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

  std::pair<NodeValue, at::ScalarType> nodeValueAndCorrectType;
  ASSIGN_VALUE_OR_RETURN_ERR(
      nodeValueAndCorrectType,
      loadArithmeticNode<glow::AddNode>("add", inputs[0], inputs[1]));

  RETURN_IF_ERR(addValueMapping(outputs[0], nodeValueAndCorrectType.first));
  RETURN_IF_ERR(
      setCorrectTypeMapping(outputs[0], nodeValueAndCorrectType.second));
  return Error::success();
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

  std::pair<NodeValue, at::ScalarType> nodeValueAndCorrectType;
  ASSIGN_VALUE_OR_RETURN_ERR(
      nodeValueAndCorrectType,
      loadArithmeticNode<glow::SubNode>("sub", inputs[0], inputs[1]));

  RETURN_IF_ERR(addValueMapping(outputs[0], nodeValueAndCorrectType.first));
  RETURN_IF_ERR(
      setCorrectTypeMapping(outputs[0], nodeValueAndCorrectType.second));
  return Error::success();
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

  std::pair<NodeValue, at::ScalarType> nodeValueAndCorrectType;
  ASSIGN_VALUE_OR_RETURN_ERR(
      nodeValueAndCorrectType,
      loadArithmeticNode<glow::SubNode>("sub", inputs[1], inputs[0]));

  RETURN_IF_ERR(addValueMapping(outputs[0], nodeValueAndCorrectType.first));
  RETURN_IF_ERR(
      setCorrectTypeMapping(outputs[0], nodeValueAndCorrectType.second));
  return Error::success();
}

template <typename GlowNode>
Error PyTorchModelLoader::loadBitwiseOp(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 2, outputs, 1));

  for (int i = 0; i <= 1; i++) {
    glow::NodeValue input;
    ASSIGN_VALUE_OR_RETURN_ERR(input, getGlowNodeValueForValue(inputs[i]));
    auto inputElementType = input.getType()->getElementType();
    switch (inputElementType) {
    case ElemKind::Int32ITy:
    case ElemKind::Int64ITy:
    case ElemKind::BoolTy:
      break;
    default:
      return MAKE_ERR("Bitwise ops are only supported on Int and Bool");
    }
  }
  std::pair<NodeValue, at::ScalarType> nodeValueAndCorrectType;
  ASSIGN_VALUE_OR_RETURN_ERR(
      nodeValueAndCorrectType,
      loadArithmeticNode<GlowNode>(
          glow::strFormat("bitwise_%s", glow::getNodeName<GlowNode>()),
          inputs[0], inputs[1]));

  RETURN_IF_ERR(addValueMapping(outputs[0], nodeValueAndCorrectType.first));
  RETURN_IF_ERR(
      setCorrectTypeMapping(outputs[0], nodeValueAndCorrectType.second));
  return Error::success();
}

Error PyTorchModelLoader::loadLogicalXor(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 2, outputs, 1));

  std::pair<NodeValue, at::ScalarType> nodeValueAndCorrectType;
  ASSIGN_VALUE_OR_RETURN_ERR(
      nodeValueAndCorrectType,
      loadArithmeticNode<XorNode>("logical_xor", inputs[0], inputs[1]));

  RETURN_IF_ERR(addValueMapping(outputs[0], nodeValueAndCorrectType.first));
  RETURN_IF_ERR(
      setCorrectTypeMapping(outputs[0], nodeValueAndCorrectType.second));
  return Error::success();
}

Error PyTorchModelLoader::loadLogicalOr(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 2, outputs, 1));

  std::pair<NodeValue, at::ScalarType> nodeValueAndCorrectType;
  ASSIGN_VALUE_OR_RETURN_ERR(
      nodeValueAndCorrectType,
      loadArithmeticNode<OrNode>("logical_or", inputs[0], inputs[1]));

  RETURN_IF_ERR(addValueMapping(outputs[0], nodeValueAndCorrectType.first));
  RETURN_IF_ERR(
      setCorrectTypeMapping(outputs[0], nodeValueAndCorrectType.second));
  return Error::success();
}

Error PyTorchModelLoader::loadLogicalAnd(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 2, outputs, 1));

  std::pair<NodeValue, at::ScalarType> nodeValueAndCorrectType;
  ASSIGN_VALUE_OR_RETURN_ERR(
      nodeValueAndCorrectType,
      loadArithmeticNode<AndNode>("logical_and", inputs[0], inputs[1]));

  RETURN_IF_ERR(addValueMapping(outputs[0], nodeValueAndCorrectType.first));
  RETURN_IF_ERR(
      setCorrectTypeMapping(outputs[0], nodeValueAndCorrectType.second));
  return Error::success();
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

  GatherNode *glowNode =
      F_.createGather("index_select", inputInput, indexInput, dimension);

  RETURN_ERR(addValueMapping(outputs[0], glowNode->getResult()));
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
    RETURN_ERR_IF_NOT(axes->size() >= 1,
                      "Must have at least one axis for aten::sum.");

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
  at::ScalarType correctType = elemKindToScalarType(input.getElementType());

  glow::NodeValue batchedReduceAdd = input;
  if (needsConvertTo) {
    int32_t dtype;
    ASSIGN_VALUE_OR_RETURN_ERR(dtype, iValToInt(dtypeIVal));
    correctType = static_cast<at::ScalarType>(dtype);
    auto glowElemKind = correctType == at::kLong
                            ? ElemKind::Int32ITy
                            : scalarTypeToElemKind(correctType);
    auto toType =
        F_.getParent()->uniqueType(glowElemKind, input.getType()->dims());
    batchedReduceAdd =
        F_.createConvertTo("to", batchedReduceAdd, toType)->getResult();
  }

  if (needsFlatten) {
    auto flatten =
        F_.createFlatten("flatten", batchedReduceAdd, input.dims().size())
            ->getResult();
    batchedReduceAdd =
        F_.createBatchedReduceAdd("sum", flatten, glowAxes)->getResult();
  } else {
    auto inDims = input.dims();
    std::vector<dim_t> outDims(inDims.begin(), inDims.end());
    auto numReducedAxes = 0;
    std::sort(glowAxes.begin(), glowAxes.end());
    for (auto axis : glowAxes) {
      batchedReduceAdd =
          F_.createBatchedReduceAdd(strFormat("sum_axis%u", axis),
                                    batchedReduceAdd, axis - numReducedAxes)
              ->getResult();
      RETURN_ERR_IF_NOT(axis < outDims.size(),
                        strFormat("Axis %u is greater than num dims %lu", axis,
                                  outDims.size()));
      if (keepDim) {
        outDims[axis] = 1;
        batchedReduceAdd =
            F_.createReshape("reshape", batchedReduceAdd, outDims);
      } else {
        numReducedAxes += 1;
      }
    }
  }

  RETURN_IF_ERR(addValueMapping(outputs[0], batchedReduceAdd));
  RETURN_IF_ERR(setCorrectTypeMapping(outputs[0], correctType));
  return Error::success();
}

Error PyTorchModelLoader::loadMax(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, -1, outputs, -1));

  if (inputs.size() == 2) {
    // Binary elementwise max, return a tensor has same type to input
    glow::NodeValue lhs;
    ASSIGN_VALUE_OR_RETURN_ERR(lhs, getGlowNodeValueForValue(inputs[0]));
    glow::NodeValue rhs;
    ASSIGN_VALUE_OR_RETURN_ERR(rhs, getGlowNodeValueForValue(inputs[1]));

    glow::MaxNode *glowNode =
        F_.createNodeWithBroadcast<MaxNode>("max", -1, lhs, rhs);
    RETURN_IF_ERR(addValueMapping(outputs[0], glowNode->getResult()));
    RETURN_IF_ERR(setCorrectTypeMappingSameAs(outputs[0], inputs[0]));
  } else if (inputs.size() == 3) {
    // aten::max(Tensor self, int dim, bool keepdim) -> (Tensor, Tensor)
    glow::NodeValue input;
    ASSIGN_VALUE_OR_RETURN_ERR(input, getGlowNodeValueForValue(inputs[0]));
    auto inDims = input.dims();

    int64_t dimRaw;
    ASSIGN_VALUE_OR_RETURN_ERR(dimRaw,
                               iValToInt(getGlowIValueForValue(inputs[1])));
    int dim = 0;
    ASSIGN_VALUE_OR_RETURN_ERR(dim,
                               getPositiveIndex(dimRaw, input.dims().size()));
    bool keepDim = false;
    ASSIGN_VALUE_OR_RETURN_ERR(keepDim,
                               iValToBool(getGlowIValueForValue(inputs[2])));
    glow::NodeValue max;
    max = F_.createBatchedReduceMax("max", input, dim)->getResult();
    if (keepDim) {
      std::vector<dim_t> shape(inDims.begin(), inDims.end());
      RETURN_ERR_IF_NOT(
          dim < shape.size(),
          strFormat("dim %d out of bounds for input shape of size %lu", dim,
                    shape.size()));
      shape[dim] = 1;
      max = F_.createReshape("max_keep_dim", max, shape)->getResult();
    }
    auto argMax = F_.createArgMax("max_indices", input, dim, keepDim);

    RETURN_IF_ERR(addValueMapping(outputs[0], max));
    RETURN_IF_ERR(addValueMapping(outputs[1], argMax));
    RETURN_IF_ERR(setCorrectTypeMappingSameAs(outputs[0], inputs[0]));
    RETURN_IF_ERR(setCorrectTypeMapping(outputs[1], at::kLong));
  } else {
    // Unary max, return a scalar contains the biggest element in input
    glow::NodeValue input;
    ASSIGN_VALUE_OR_RETURN_ERR(input, getGlowNodeValueForValue(inputs[0]));

    auto n = input.dims().size();
    size_t length = 1;
    for (int i = 0; i < n; i++) {
      length *= input.dims()[i];
    }
    auto reshaped = F_.createReshape("reshaped_flatten_input", input, {length})
                        ->getResult();
    reshaped = F_.createBatchedReduceMax("batched_reduce_max_for_unary_max",
                                         reshaped, 0)
                   ->getResult();
    RETURN_IF_ERR(addValueMapping(outputs[0], reshaped));
    RETURN_IF_ERR(setCorrectTypeMappingSameAs(outputs[0], inputs[0]));
  }
  return Error::success();
}

Error PyTorchModelLoader::loadAmax(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, -2, outputs, 1));

  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(input, getGlowNodeValueForValue(inputs[0]));
  glow::GlowIValue *dimsIValue;
  ASSIGN_VALUE_OR_RETURN_ERR(dimsIValue, getGlowIValueForValue(inputs[1]));
  std::vector<unsigned_t> glowAxes;

  if (dimsIValue->isInt()) {
    int64_t dimRaw;
    ASSIGN_VALUE_OR_RETURN_ERR(dimRaw, iValToInt(dimsIValue));
    int dim = 0;
    ASSIGN_VALUE_OR_RETURN_ERR(dim,
                               getPositiveIndex(dimRaw, input.dims().size()));
    glowAxes.push_back(static_cast<unsigned_t>(dim));
  } else { // dimsIValue->isIntList()
    std::vector<int64_t> *axes;
    ASSIGN_VALUE_OR_RETURN_ERR(axes,
                               iValToIntList(getGlowIValueForValue(inputs[1])));

    for (auto axisRaw : *axes) {
      int axis = 0;
      ASSIGN_VALUE_OR_RETURN_ERR(
          axis, getPositiveIndex(axisRaw, input.dims().size()));
      glowAxes.push_back(static_cast<unsigned_t>(axis));
    }
  }
  RETURN_ERR_IF_NOT(glowAxes.size() >= 1, "Empty dims for reduction");
  auto amax = F_.createBatchedReduceMax("batched_reduce_max_for_unary_max",
                                        input, glowAxes)
                  ->getResult();
  RETURN_ERR(addValueMapping(outputs[0], amax));
}

/*
  aten::size(Tensor self, int dim) -> int"
  aten::size(Tensor self) -> int[]
*/
Error PyTorchModelLoader::loadSize(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, -1, outputs, 1));

  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(
      input, getGlowNodeValueForValue(inputs[SizeInputs::input]));
  GlowIValue glowIVal;
  if (ptNode->inputs().size() > 1) {
    int64_t dim;
    ASSIGN_VALUE_OR_RETURN_ERR(
        dim, iValToInt(getGlowIValueForValue(inputs[SizeInputs::dim])));

    // Convert negative dimension index into corresponding positive index
    auto origDim = dim;
    if (dim < 0) {
      dim += input.dims().size();
    }

    RETURN_ERR_IF_NOT(
        dim < input.dims().size() && dim >= 0,
        strFormat("Dim value of %ld is out of range. Valid values "
                  "are in the range [-%ld, %ld]",
                  origDim, input.dims().size(), input.dims().size() - 1));

    glowIVal.fromInt(input.dims()[dim]);
  } else {
    std::vector<int64_t> intList{input.dims().begin(), input.dims().end()};
    glowIVal.fromIntList(intList);
  }

  RETURN_ERR(addValueMapping(outputs[0], std::move(glowIVal)));
}

Error PyTorchModelLoader::loadListConstruct(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  // Requires -1 because this requires at least one input.
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, -1, outputs, 1));
  GlowIValue glowIVal;

  // Only used for lists of tensors
  std::vector<at::ScalarType> correctTypes;

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

      at::ScalarType correctType;
      ASSIGN_VALUE_OR_RETURN_ERR(correctType, getCorrectTypeMapping(inputs[i]));
      correctTypes.push_back(correctType);
    }
    glowIVal.fromNodeValueList(std::move(nodeValues));
  } else {
    // Should never reach here
    return MAKE_ERR("Encountered unknown JIT Value mapping");
  }
  RETURN_IF_ERR(addValueMapping(outputs[0], std::move(glowIVal)));
  if (!correctTypes.empty()) {
    RETURN_IF_ERR(setCorrectTypesMapping(outputs[0], correctTypes));
  }

  return Error::success();
}

Error PyTorchModelLoader::loadListUnpack(const torch::jit::Node *ptNode) {
  RETURN_ERR_IF_NOT(ptNode->inputs().size() == 1,
                    "ListUnpack only supports a single input");
  std::vector<glow::NodeValue> *inputs;
  ASSIGN_VALUE_OR_RETURN_ERR(
      inputs, iValToNodeValueList(getGlowIValueForValue(ptNode->input(0))));

  std::vector<at::ScalarType> correctTypes;
  ASSIGN_VALUE_OR_RETURN_ERR(correctTypes,
                             getCorrectTypesMapping(ptNode->input(0)));

  RETURN_ERR_IF_NOT(ptNode->outputs().size() == inputs->size(),
                    "ListUnpack inputs and outputs must be of same size.");

  RETURN_ERR_IF_NOT(ptNode->outputs().size() == correctTypes.size(),
                    "Must have one correct type per ListUnpack output");

  for (int idx = 0; idx < inputs->size(); idx++) {
    RETURN_IF_ERR(addValueMapping(ptNode->output(idx), inputs->at(idx)));
    RETURN_IF_ERR(
        setCorrectTypeMapping(ptNode->output(idx), correctTypes.at(idx)));
  }
  return Error::success();
}

/// Mirroring the implementation in
/// caffe2/aten/src/ATen/native/TypeProperties.cpp
static inline at::ScalarType promote_skip_undefined(at::ScalarType a,
                                                    at::ScalarType b) {
  if (a == at::ScalarType::Undefined) {
    return b;
  }
  if (b == at::ScalarType::Undefined) {
    return a;
  }
  return c10::promoteTypes(a, b);
}

/// Helper function to support upcasting in concat, we calculate the higher
/// type among a list of types. For example, the higher type of [half, float,
/// half, double] will be double. Similar to at::result_type().
static Expected<at::ScalarType>
getHigherType(PyTorchModelLoader *loader,
              const c10::ArrayRef<const torch::jit::Value *> &values) {
  at::ScalarType higherType = at::ScalarType::Undefined;
  for (auto v : values) {
    at::ScalarType dtype;
    ASSIGN_VALUE_OR_RETURN_ERR(dtype, loader->getCorrectTypeMapping(v));
    if (dtype != at::ScalarType::QInt8 && dtype != at::ScalarType::QUInt8) {
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
static Expected<std::pair<NodeValue, at::ScalarType>>
createConcatNode(PyTorchModelLoader *loader, Function &F,
                 const torch::jit::Node *ptNode, bool isStack,
                 bool doBroadcast) noexcept {
  std::vector<glow::NodeValue> inputs;
  for (auto *ptInput : ptNode->inputs()) {
    glow::NodeValue nodeVal;
    ASSIGN_VALUE_OR_RETURN_ERR(nodeVal,
                               loader->getGlowNodeValueForValue(ptInput));
    // Legacy behavior suggests supporting concat of empty tensors, but only
    // for this specific shape. See the legacy_cat_wrap_dim function in
    // caffe2/aten/src/ATen/WrapDimUtils.h for more info.
    if (nodeVal.dims() == llvm::ArrayRef<dim_t>({0})) {
      continue;
    }
    inputs.push_back(nodeVal);
  }

  RETURN_ERR_IF_NOT(inputs.size() > 0, "No non-empty inputs were provided!");

  // Get number of input dimensions
  int64_t numInputDims = inputs[0].dims().size();
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

  at::ScalarType higherType = at::ScalarType::Undefined;
  glow::ElemKind higherKind;
  ASSIGN_VALUE_OR_RETURN_ERR(higherType,
                             getHigherType(loader, ptNode->inputs()));
  if (higherType != at::ScalarType::Undefined) {
    higherKind = scalarTypeToElemKind(higherType);
  }

  // Final shape for all tensors after broadcast
  std::vector<int64_t> bcastShape(numInputDims, -1);
  std::vector<bool> needBroadcast(inputs.size(), false);
  bool noBroadcastNeeded = true;

  // Use multiple vectors for hierarchical concats, the first vector is the
  // final concat
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto input = inputs[i];
    if (numInputDims != input.dims().size()) {
      std::ostringstream ss;
      ss << "All inputs must have the same number of dimensions, but got";
      for (const auto &inp : inputs) {
        ss << " [" << folly::join(",", inp.dims()) << "]";
      }
      return MAKE_ERR(ss.str());
    }

    // Record broadcast shapes to perform broadcasting
    for (int d = 0; d < input.dims().size(); ++d) {
      // For stack we can broadcast every dim
      if (d != dim || isStack) {
        if (bcastShape[d] < 0 && input.dims()[d] != 1) {
          // record first non-singleton size for dim_i as broadcast shape
          bcastShape[d] = input.dims()[d];
        } else if (input.dims()[d] == 1) {
          needBroadcast[i] = true;
          noBroadcastNeeded = false;
        }
      }
    }

    if (higherType == at::ScalarType::Undefined) {
      continue;
    } else if (isQuantizedElemKind(higherKind)) {
      continue;
    } else if (input.getElementType() == higherKind) {
      continue;
    } else {
      inputs.at(i) =
          F.createConvertTo("upcastForConcat", input, higherKind)->getResult();
    }
  }

  if (!doBroadcast || noBroadcastNeeded) {
    std::pair<NodeValue, at::ScalarType> pp = {
        F.createConcat(isStack ? "stack_concat" : "cat", inputs,
                       std::min(dim, numInputDims - 1))
            ->getResult(),
        higherType};
    return pp;
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

  for (size_t i = 0; i < inputs.size(); ++i) {
    // Use dimensions as key so we can concat those of the same dimensions
    // For example, for [1, 32, 1] when we concat dim=1, the key is '1_1',
    // and for [1, 32, 4] the key is '1_4'.
    // We use '_' as delimiter to conveniently reuse the key for node name.
    std::stringstream ss;
    for (int d = 0; d < inputs[i].dims().size(); ++d) {
      if (d != dim || isStack) {
        ss << inputs[i].dims()[d] << "_";
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
      partialConcatInputs.emplace_back(inputs[i]);
    } else {
      prevConcatKey = "";
      finalConcatInputs.emplace_back(inputs[i]);
    }
  }
  // In cast the last nodes (1 or more) need concat and broadcast
  if (!partialConcatInputs.empty()) {
    finalConcatInputs.emplace_back(
        addConcatAndBroadcastNode(partialConcatInputs, prevConcatKey));
  }

  std::pair<NodeValue, at::ScalarType> pp = {
      F.createConcat(isStack ? "stack_concat" : "cat", finalConcatInputs,
                     std::min(dim, numInputDims - 1))
          ->getResult(),
      higherType};

  return pp;
}

Error PyTorchModelLoader::loadFusedConcat(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, -1, outputs, 1));

  std::pair<NodeValue, at::ScalarType> concatAndHigherType;
  ASSIGN_VALUE_OR_RETURN_ERR(concatAndHigherType,
                             createConcatNode(this, F_, ptNode,
                                              false /* isStack */,
                                              false /* doBroadcast */));
  RETURN_IF_ERR(addValueMapping(outputs[0], concatAndHigherType.first));

  if (concatAndHigherType.second != at::ScalarType::Undefined) {
    RETURN_IF_ERR(
        setCorrectTypeMapping(outputs[0], concatAndHigherType.second));
  } else {
    RETURN_IF_ERR(setCorrectTypeMappingSameAs(outputs[0], inputs[0]));
  }

  return Error::success();
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
    RETURN_IF_ERR(addValueMapping(outputs[0], input));
    RETURN_IF_ERR(setCorrectTypeMappingSameAs(outputs[0], inputs[0]));
    return Error::success();
  }
  std::pair<NodeValue, at::ScalarType> concatAndHigherType;
  ASSIGN_VALUE_OR_RETURN_ERR(concatAndHigherType,
                             createConcatNode(this, F_, ptNode,
                                              false /* isStack */,
                                              true /* doBroadcast */));
  RETURN_IF_ERR(addValueMapping(outputs[0], concatAndHigherType.first));

  if (concatAndHigherType.second != at::ScalarType::Undefined) {
    RETURN_IF_ERR(
        setCorrectTypeMapping(outputs[0], concatAndHigherType.second));
  } else {
    RETURN_IF_ERR(setCorrectTypeMappingSameAs(outputs[0], inputs[0]));
  }

  return Error::success();
}

static glow::Node *createReshapeNodeForStack(glow::Function &F,
                                             const torch::jit::Node *ptNode,
                                             const NodeValue concat) {
  auto inputs = ptNode->inputs();
  int64_t dim = ptNode->i(at::attr::dim);

  auto concatDims = concat.dims();
  uint ndims = concatDims.size();
  auto numInputs = inputs.size();
  std::vector<glow::dim_t> reshapeDims;

  // if dim == a.dim(), then the correct calculation should be:
  // torch.stack([a, a], dim=dim)
  // <==> torch.stack([a, a], dim=dim-1).transpose(dim, dim - 1)

  for (size_t i = 0; i < ndims; ++i) {
    if ((dim == ndims && i == dim - 1) || i == dim) {
      reshapeDims.push_back(numInputs);
      reshapeDims.push_back(concatDims[i] / numInputs);
    } else {
      reshapeDims.push_back(concatDims[i]);
    }
  }

  auto reshapeNode =
      F.createReshape("stack_reshape", concat, reshapeDims)->getResult();

  std::vector<glow::unsigned_t> shuffle(reshapeDims.size());
  std::iota(shuffle.begin(), shuffle.end(), 0);
  std::swap(shuffle[ndims], shuffle[ndims - 1]);

  if (dim == ndims) {
    return F.createTranspose("stack_transpose", reshapeNode, shuffle)
        ->getResult();
  } else {
    return reshapeNode;
  }
}

Error PyTorchModelLoader::loadFusedStack(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, -1, outputs, 1));

  // In the case of a single input, just return it.
  if (inputs.size() == 1) {
    glow::NodeValue input;
    ASSIGN_VALUE_OR_RETURN_ERR(input, getGlowNodeValueForValue(inputs[0]));

    int64_t dimRaw = ptNode->i(at::attr::dim);
    int64_t dim = 0;
    auto inDims = input.dims();
    ASSIGN_VALUE_OR_RETURN_ERR(dim, getPositiveIndex(dimRaw, inDims.size()));

    std::vector<dim_t> outDims(inDims.begin(), inDims.end());
    outDims.insert(outDims.begin() + dim, 1);
    auto res = F_.createReshape("stack", input, outDims)->getResult();

    RETURN_IF_ERR(addValueMapping(outputs[0], res));
    RETURN_IF_ERR(setCorrectTypeMappingSameAs(outputs[0], inputs[0]));
    return Error::success();
  }

  std::pair<NodeValue, at::ScalarType> concatAndHigherType;
  ASSIGN_VALUE_OR_RETURN_ERR(concatAndHigherType,
                             createConcatNode(this, F_, ptNode,
                                              true /* isStack */,
                                              false /* doBroadcast */));

  RETURN_IF_ERR(addValueMapping(
      outputs[0],
      createReshapeNodeForStack(F_, ptNode, concatAndHigherType.first)));

  if (concatAndHigherType.second != at::ScalarType::Undefined) {
    RETURN_IF_ERR(
        setCorrectTypeMapping(outputs[0], concatAndHigherType.second));
  } else {
    RETURN_IF_ERR(setCorrectTypeMappingSameAs(outputs[0], inputs[0]));
  }

  return Error::success();
}

Error PyTorchModelLoader::loadFusedBroadcastStack(
    const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, -1, outputs, 1));

  std::pair<NodeValue, at::ScalarType> concatAndHigherType;
  ASSIGN_VALUE_OR_RETURN_ERR(concatAndHigherType,
                             createConcatNode(this, F_, ptNode,
                                              true /* isStack */,
                                              true /* doBroadcast */));

  RETURN_IF_ERR(addValueMapping(
      outputs[0],
      createReshapeNodeForStack(F_, ptNode, concatAndHigherType.first)));

  if (concatAndHigherType.second != at::ScalarType::Undefined) {
    RETURN_IF_ERR(
        setCorrectTypeMapping(outputs[0], concatAndHigherType.second));
  } else {
    RETURN_IF_ERR(setCorrectTypeMappingSameAs(outputs[0], inputs[0]));
  }

  return Error::success();
}

Error PyTorchModelLoader::loadNumToTensor(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 1, outputs, 1));

  glow::GlowIValue *glowIValue;
  ASSIGN_VALUE_OR_RETURN_ERR(glowIValue, getGlowIValueForValue(inputs[0]));
  glow::Tensor t;

  at::ScalarType correctType = at::ScalarType::Undefined;

  if (glowIValue->isInt()) {
    int32_t input;
    ASSIGN_VALUE_OR_RETURN_ERR(input, to32Bit(glowIValue->toInt()));
    t = glow::Tensor(glow::ElemKind::Int32ITy, {1});
    t.init(glow::Tensor::InitKind::Broadcast, input, F_.getParent()->getPRNG());
    correctType = at::kLong;
  } else if (glowIValue->isDouble()) {
    float input;
    ASSIGN_VALUE_OR_RETURN_ERR(input, to32Bit(glowIValue->toDouble()));
    t = glow::Tensor(glow::ElemKind::FloatTy, {1});
    t.init(glow::Tensor::InitKind::Broadcast, input, F_.getParent()->getPRNG());
    correctType = at::kFloat;
  } else {
    // Not a number
    return MAKE_ERR(strFormat(
        "Expected integer/double GlowIValue type in NumToTensor, but get: %s",
        glowIValue->getTagString()));
  }
  auto output =
      F_.getParent()->createConstant("NumToTensor_output", std::move(t));

  RETURN_IF_ERR(addValueMapping(outputs[0], output));
  RETURN_IF_ERR(setCorrectTypeMapping(outputs[0], correctType));
  return Error::success();
}

Error PyTorchModelLoader::loadShapeAsTensor(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 1, outputs, 1));

  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(input, getGlowNodeValueForValue(inputs[0]));

  auto dims = input.getType()->dims();
  std::vector<int32_t> outputValues{dims.begin(), dims.end()};

  auto type =
      F_.getParent()->uniqueType(glow::ElemKind::Int32ITy, outputValues.size());

  auto outputTensor = glow::Tensor(outputValues.data(), type);

  auto output = F_.getParent()->createConstant("ShapeAsTensor_output",
                                               std::move(outputTensor));

  output->ensureIsOwned(); // Prevents heap use after free
  RETURN_IF_ERR(addValueMapping(ptNode->output(), output));
  return Error::success();
}

Error PyTorchModelLoader::loadArgMin(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, -2, outputs, 1));
  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(
      input, getGlowNodeValueForValue(inputs[ArgMaxMinInputs::input]));

  glow::unsigned_t axis = 0;
  if (hasGlowIValueForValue(inputs[ArgMaxMinInputs::axis],
                            /*ignoreNones*/ true)) {
    ASSIGN_VALUE_OR_RETURN_ERR(
        axis, static_cast_expected<glow::unsigned_t>(contractIntIValIfNeeded(
                  getGlowIValueForValue(inputs[ArgMaxMinInputs::axis]))));
  }

  bool keepDims = true;
  if (inputs.size() > 2 &&
      hasGlowIValueForValue(inputs[ArgMaxMinInputs::keepDims])) {
    ASSIGN_VALUE_OR_RETURN_ERR(
        keepDims,
        iValToBool(getGlowIValueForValue(inputs[ArgMaxMinInputs::keepDims])));
  }

  auto output = F_.createArgMin("argmin", input, axis, keepDims);
  RETURN_IF_ERR(addValueMapping(ptNode->output(), output));
  return Error::success();
}

Error PyTorchModelLoader::loadArgMax(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, -2, outputs, 1));
  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(
      input, getGlowNodeValueForValue(inputs[ArgMaxMinInputs::input]));

  glow::unsigned_t axis = 0;
  if (hasGlowIValueForValue(inputs[ArgMaxMinInputs::axis],
                            /*ignoreNones*/ true)) {
    ASSIGN_VALUE_OR_RETURN_ERR(
        axis, static_cast_expected<glow::unsigned_t>(contractIntIValIfNeeded(
                  getGlowIValueForValue(inputs[ArgMaxMinInputs::axis]))));
  }

  bool keepDims = true;
  if (inputs.size() > 2 &&
      hasGlowIValueForValue(inputs[ArgMaxMinInputs::keepDims])) {
    ASSIGN_VALUE_OR_RETURN_ERR(
        keepDims,
        iValToBool(getGlowIValueForValue(inputs[ArgMaxMinInputs::keepDims])));
  }
  auto output = F_.createArgMax("argmax", input, axis, keepDims);
  RETURN_IF_ERR(addValueMapping(ptNode->output(), output));
  return Error::success();
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
  dim_t input_len = intConstant->getPayload().size();
  RETURN_ERR_IF_NOT(
      input_len == 1,
      strFormat("Expected input to have length 1, but found: %lu", input_len));
  // Also need to check if intConstant is a scalar
  int value;

  if (inputElementType == glow::ElemKind::Int32ITy) {
    value = intConstant->getPayload().getHandle<int32_t>().raw(0);
  } else if (inputElementType == glow::ElemKind::Int64ITy) {
    value = intConstant->getPayload().getHandle<int64_t>().raw(0);
  } else if (inputElementType == glow::ElemKind::FloatTy) {
    auto value_f = intConstant->getPayload().getHandle<float>().raw(0);
    value = static_cast<int>(value_f);
  } else {
    return MAKE_ERR("Expected integer/float tensor in loadInt");
  }
  glow::GlowIValue glowIVal;
  // No matter input is int32 or int64, it is int in glowIVal.
  // When using NumToTensor, this int will transformed into int64 again.
  glowIVal.fromInt(value);
  RETURN_IF_ERR(addValueMapping(outputs[0], std::move(glowIVal)));
  return Error::success();
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
  auto glowElemKind = scalarTypeToElemKind(static_cast<at::ScalarType>(dtype));

  auto output =
      F_.createSplat(
            "zeros",
            F_.getParent()->uniqueType(glowElemKind, inputSizeArrayRef), 0)
          ->getResult();
  return addValueMapping(outputs[0], output);
}

Error PyTorchModelLoader::loadEmptyLike(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 6, outputs, 1));

  glow::ElemKind outputGlowElemKind;
  ASSIGN_VALUE_OR_RETURN_ERR(outputGlowElemKind,
                             getExpectedType(inputs[EmptyLikeInputs::self],
                                             inputs[EmptyLikeInputs::dtype]));

  if (outputGlowElemKind == ElemKind::Int32ITy) {
    const auto fillValue = at::nullopt;
    return loadFullLikeImpl<int>("empty_like", inputs[EmptyLikeInputs::self],
                                 outputGlowElemKind, fillValue, outputs[0]);
  } else {
    const auto fillValue = at::nullopt;
    return loadFullLikeImpl<double>("empty_like", inputs[EmptyLikeInputs::self],
                                    outputGlowElemKind, fillValue, outputs[0]);
  }
}

Error PyTorchModelLoader::loadZerosLike(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 6, outputs, 1));

  glow::ElemKind outputGlowElemKind;
  ASSIGN_VALUE_OR_RETURN_ERR(outputGlowElemKind,
                             getExpectedType(inputs[ZerosLikeInputs::self],
                                             inputs[ZerosLikeInputs::dtype]));

  if (outputGlowElemKind == ElemKind::Int32ITy) {
    const int fillValue = 0;
    return loadFullLikeImpl<int>("zeros_like", inputs[ZerosLikeInputs::self],
                                 outputGlowElemKind, fillValue, outputs[0]);
  } else {
    const double fillValue = 0.0;
    return loadFullLikeImpl<double>("zeros_like", inputs[ZerosLikeInputs::self],
                                    outputGlowElemKind, fillValue, outputs[0]);
  }
}

Error PyTorchModelLoader::loadOnesLike(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 6, outputs, 1));

  glow::ElemKind outputGlowElemKind;
  ASSIGN_VALUE_OR_RETURN_ERR(outputGlowElemKind,
                             getExpectedType(inputs[OnesLikeInputs::self],
                                             inputs[OnesLikeInputs::dtype]));

  if (outputGlowElemKind == ElemKind::Int32ITy) {
    const int fillValue = 1;
    return loadFullLikeImpl<int>("ones_like", inputs[OnesLikeInputs::self],
                                 outputGlowElemKind, fillValue, outputs[0]);
  } else {
    const double fillValue = 1.0;
    return loadFullLikeImpl<double>("ones_like", inputs[OnesLikeInputs::self],
                                    outputGlowElemKind, fillValue, outputs[0]);
  }
}

Error PyTorchModelLoader::loadFullLike(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 7, outputs, 1));

  glow::ElemKind outputGlowElemKind;
  ASSIGN_VALUE_OR_RETURN_ERR(outputGlowElemKind,
                             getExpectedType(inputs[FullLikeInputs::self],
                                             inputs[FullLikeInputs::dtype]));

  if (outputGlowElemKind == ElemKind::Int32ITy) {
    int fillValue;
    ASSIGN_VALUE_OR_RETURN_ERR(
        fillValue,
        iValToInt(getGlowIValueForValue(inputs[FullLikeInputs::fill_value])));

    return loadFullLikeImpl<int>("full_like", inputs[FullLikeInputs::self],
                                 outputGlowElemKind, fillValue, outputs[0]);
  } else {
    double fillValue;
    ASSIGN_VALUE_OR_RETURN_ERR(fillValue,
                               iValToDouble(getGlowIValueForValue(
                                   inputs[FullLikeInputs::fill_value])));

    return loadFullLikeImpl<double>("full_like", inputs[FullLikeInputs::self],
                                    outputGlowElemKind, fillValue, outputs[0]);
  }
}

Expected<ElemKind>
PyTorchModelLoader::getExpectedType(const torch::jit::Value *inputTensorValue,
                                    const torch::jit::Value *dtypeValue) {
  glow::NodeValue inputTensor;
  ASSIGN_VALUE_OR_RETURN_ERR(inputTensor,
                             getGlowNodeValueForValue(inputTensorValue));

  glow::GlowIValue *dtypeGlowValue;
  ASSIGN_VALUE_OR_RETURN_ERR(dtypeGlowValue, getGlowIValueForValue(dtypeValue));

  auto isNoneType = dtypeGlowValue->isNone();

  // if dtype not specified, default dtype to the same type as input tensor
  auto glowElemKind = inputTensor.getType()->getElementType();
  auto correctType = elemKindToScalarType(glowElemKind);

  if (!isNoneType) {
    int32_t dtype;
    ASSIGN_VALUE_OR_RETURN_ERR(dtype,
                               iValToInt(getGlowIValueForValue(dtypeValue)));

    correctType = static_cast<at::ScalarType>(dtype);
  }

  glowElemKind = correctType == at::kLong ? ElemKind::Int32ITy
                                          : scalarTypeToElemKind(correctType);

  return Expected<ElemKind>(glowElemKind);
}

template <class DType>
Error PyTorchModelLoader::loadFullLikeImpl(
    llvm::StringRef name, const torch::jit::Value *inputTensorValue,
    const glow::ElemKind outputGlowElemKind, at::optional<DType> fillValue,
    const torch::jit::Value *outputValue) {
  glow::NodeValue inputTensor;
  ASSIGN_VALUE_OR_RETURN_ERR(inputTensor,
                             getGlowNodeValueForValue(inputTensorValue));

  llvm::ArrayRef<glow::dim_t> selfDims(inputTensor.getType()->dims());
  auto correctType = elemKindToScalarType(outputGlowElemKind);

  glow::NodeValue outputTensor;
  if (fillValue.has_value()) {
    outputTensor =
        F_.createSplat(name,
                       F_.getParent()->uniqueType(outputGlowElemKind, selfDims),
                       fillValue.value())
            ->getResult();
  } else {
    outputTensor = F_.getParent()
                       ->createConstant(outputGlowElemKind, selfDims, name)
                       ->getOutput();
  }

  RETURN_IF_ERR(addValueMapping(outputValue, outputTensor));
  RETURN_IF_ERR(setCorrectTypeMapping(outputValue, correctType));
  return Error::success();
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
    RETURN_IF_ERR(addValueMapping(ptNode->output(), output));
    // TODO: allow correct type mapping from double to float
    RETURN_IF_ERR(setCorrectTypeMapping(ptNode->output(), at::kFloat));
  } else {
    int32_t start;
    int32_t end;
    int32_t step;
    ASSIGN_VALUE_OR_RETURN_ERR(
        start, startIVal->isInt()
                   ? to32Bit(startIVal->toInt())
                   : static_cast_expected<int32_t>(startIVal->toDouble()));
    ASSIGN_VALUE_OR_RETURN_ERR(
        end, endIVal->isInt()
                 ? to32Bit(endIVal->toInt())
                 : static_cast_expected<int32_t>(endIVal->toDouble()));
    ASSIGN_VALUE_OR_RETURN_ERR(
        step, stepIVal->isInt()
                  ? to32Bit(stepIVal->toInt())
                  : static_cast_expected<int32_t>(stepIVal->toDouble()));
    std::vector<int32_t> outputValues;
    auto span = std::abs(end - start);
    for (int32_t offset = 0; std::abs(offset) < span; offset += step) {
      outputValues.push_back(start + offset);
    }
    auto type = F_.getParent()->uniqueType(glow::ElemKind::Int32ITy,
                                           outputValues.size());
    auto outputTensor = glow::Tensor(outputValues.data(), type);
    auto output = F_.getParent()->createConstant("Arange_output",
                                                 std::move(outputTensor));
    output->ensureIsOwned(); // Prevents heap use after free
    RETURN_IF_ERR(addValueMapping(ptNode->output(), output));
    RETURN_IF_ERR(setCorrectTypeMapping(ptNode->output(), at::kLong));
  }
  return Error::success();
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

  // If there was a negative index, replace it with the remaining dims in
  // input.
  if (negOneIndex >= 0) {
    shape[negOneIndex] = inputTotalDims / shapeTotalDims;
  }

  RETURN_ERR(
      addValueMapping(outputs[0], F_.createReshape("reshape", input,
                                                   castVector<dim_t>(shape))));
}

Error PyTorchModelLoader::loadRepeat(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 2, outputs, 1));

  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(
      input, getGlowNodeValueForValue(inputs[RepeatInputs::input]));

  std::vector<int64_t> *repeats;
  ASSIGN_VALUE_OR_RETURN_ERR(repeats, iValToIntList(getGlowIValueForValue(
                                          inputs[RepeatInputs::repeats])));
  std::vector<int64_t> repeatsCast = castVector<int64_t>(*repeats);
  RETURN_ERR_IF_NOT(repeatsCast.size() >= input.dims().size(),
                    "The rank of the input tensor must be less than or "
                    "equal to the size of repeats vector for aten::repeat");

  const bool needsExpand = input.dims().size() < repeatsCast.size();
  if (needsExpand) {
    const auto diff = repeatsCast.size() - input.dims().size();
    std::vector<dim_t> newDims;
    for (size_t i = 0; i < diff; ++i) {
      newDims.push_back(1);
    }
    newDims.insert(newDims.end(), input.dims().begin(), input.dims().end());
    input = F_.createReshape("reshape", input, newDims)->getResult();
  }

  NodeValue output = input;
  for (size_t i = 0; i < repeatsCast.size(); ++i) {
    const auto repeat = repeatsCast[i];
    RETURN_ERR_IF_NOT(
        repeat > 0,
        "The value of repeat must be greater than 0 for aten::repeat");
    output = F_.createTile("tile." + std::to_string(i), output, repeat, i)
                 ->getResult();
  }

  RETURN_ERR(addValueMapping(outputs[0], output));
}

template <typename Node,
          Node *(glow::Function::*CreateFn)(llvm::StringRef, glow::NodeValue)>
Error PyTorchModelLoader::loadUnaryNode(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 1, outputs, 1));

  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(input, getGlowNodeValueForValue(inputs[0]));

  Node *output = (F_.*CreateFn)(glow::getNodeName<Node>(), input);
  return addValueMapping(outputs[0], output);
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

  glow::NodeValue input_transposed;
  glow::NodeValue output;
  // Upsample 3D
  if (dimSize == 5) {
    // pytorch default layout is NCTHW: retrieve dimensions and transposed to
    // Glow's layout NTHWC
    name = "upsample_nearest3d";
    dim_t ia = input.dims()[0];
    dim_t ib = input.dims()[1];
    dim_t ox = (dim_t)(*outputSize)[0];
    dim_t oy = (dim_t)(*outputSize)[1];
    dim_t oz = (dim_t)(*outputSize)[2];
    input_transposed = F_.createTranspose("upsample_nearest3d_input_transpose",
                                          input, NCTHW2NTHWC)
                           ->getResult();

    outTy = F_.getParent()->uniqueTypeWithNewShape(input_transposed.getType(),
                                                   {ia, ox, oy, oz, ib});
    output = F_.createResizeNearest(name, input_transposed, outTy)->getResult();
    output = F_.createTranspose("upsample_nearest3d_output_transpose", output,
                                NTHWC2NCTHW)
                 ->getResult();
  } else { // Upsample 2D
    name = "upsample_nearest2d";
    dim_t iN = input.dims()[0];
    dim_t iC = input.dims()[1];
    dim_t oH = (dim_t)(*outputSize)[0];
    dim_t oW = (dim_t)(*outputSize)[1];
    input_transposed = F_.createTranspose("upsample_nearest2d_input_transpose",
                                          input, NCHW2NHWC);
    outTy = F_.getParent()->uniqueTypeWithNewShape(input_transposed.getType(),
                                                   {iN, oH, oW, iC});
    output = F_.createResizeNearest(name, input_transposed, outTy)->getResult();
    output = F_.createTranspose("upsample_nearest2d_output_transpose", output,
                                NHWC2NCHW)
                 ->getResult();
  }
  RETURN_ERR(addValueMapping(outputs[0], output));
}

Error PyTorchModelLoader::loadView(const torch::jit::Node *ptNode) {
  // loadView is just like Reshape, except reshape should call contiguous
  // for non-contiguous data and view should fail
  return PyTorchModelLoader::loadReshape(ptNode);
}

Error PyTorchModelLoader::loadLeakyRelu(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, -2, outputs, 1));

  RETURN_ERR_IF_NOT(inputs.size() <= 3,
                    "Expected at most 3 inputs to aten::leaky_relu");

  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(input, getGlowNodeValueForValue(inputs[0]));

  float negativeSlope;
  ASSIGN_VALUE_OR_RETURN_ERR(negativeSlope,
                             to32Bit(iValToDouble(getGlowIValueForValue(
                                 ptNode->namedInput("negative_slope")))));

  auto *output = F_.createLeakyRELU("leaky_relu", input, negativeSlope);
  RETURN_ERR(addValueMapping(outputs[0], output->getResult()));
}

Error PyTorchModelLoader::loadPow(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 2, outputs, 1));

  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(input, getGlowNodeValueForValue(inputs[0]));

  glow::PowNode *PN = nullptr;
  if (hasGlowIValueForValue(inputs[1])) {
    glow::GlowIValue *exponentIValue;
    ASSIGN_VALUE_OR_RETURN_ERR(exponentIValue,
                               getGlowIValueForValue(inputs[1]));

    float exponent;
    if (exponentIValue->isDouble()) {
      ASSIGN_VALUE_OR_RETURN_ERR(exponent, iValToDouble(exponentIValue));
    } else if (exponentIValue->isInt()) {
      ASSIGN_VALUE_OR_RETURN_ERR(exponent, iValToInt(exponentIValue));
    } else {
      std::ostringstream ss;
      ss << "Unsupported type for exponent in node " << *ptNode;
      return MAKE_ERR(ss.str());
    }
    PN = F_.createPow("pow", input, exponent);
  } else {
    NodeValue expNV;
    ASSIGN_VALUE_OR_RETURN_ERR(expNV, getGlowNodeValueForValue(inputs[1]));
    PN = F_.createNodeWithBroadcast<PowNode>("pow", -1, input, expNV);
  }

  RETURN_ERR(addValueMapping(outputs[0], PN->getResult()));
}

Error PyTorchModelLoader::loadIndexPut(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();

  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(
      input, getGlowNodeValueForValue(inputs[IndexPutInputs::input]));
  std::vector<glow::NodeValue> *indices;
  ASSIGN_VALUE_OR_RETURN_ERR(indices, iValToNodeValueList(getGlowIValueForValue(
                                          inputs[IndexPutInputs::indices])));
  glow::NodeValue value;
  ASSIGN_VALUE_OR_RETURN_ERR(
      value, getGlowNodeValueForValue(inputs[IndexPutInputs::value]));

  std::vector<glow::NodeValue> reshapes;
  for (auto idx : *indices) {
    auto *rs = F_.createReshape("reshape", idx, {idx.dims()[0], 1});
    reshapes.push_back(rs);
  }
  auto *concatNode = F_.createConcat("concat", reshapes, 1);
  auto idxDims = concatNode->getResult().dims();

  bool cumulative = false;
  ASSIGN_VALUE_OR_RETURN_ERR(
      cumulative,
      iValToBool(getGlowIValueForValue(ptNode->namedInput("accumulate"))));

  auto expectedValueDims = input.dims().drop_front(idxDims[1]).vec();
  expectedValueDims.insert(expectedValueDims.begin(), idxDims[0]);
  glow::Tensor t(value.getElementType(), expectedValueDims);
  auto *valueConstant =
      F_.getParent()->createConstant("valueConstant", std::move(t));
  value = F_.broadcastInputs(/* axis */ -1, {value, valueConstant})[0];

  const bool needsValueReshape = idxDims[0] != value.dims()[0];
  if (needsValueReshape) {
    value = F_.createReshape("reshape", value, expectedValueDims)->getResult();
  }

  auto *scatterNode = F_.createScatterData("scatter_data", input, concatNode,
                                           value, cumulative);

  RETURN_ERR(addValueMapping(outputs[0], scatterNode));
}

Error PyTorchModelLoader::loadBitwiseNot(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 1, outputs, 1));

  glow::NodeValue glowInput;

  ASSIGN_VALUE_OR_RETURN_ERR(glowInput, getGlowNodeValueForValue(inputs[0]));

  auto inputElementType = glowInput.getType()->getElementType();

  if (inputElementType == glow::ElemKind::BoolTy) {
    // if bool, use existing logical_not implementation
    return addValueMapping(outputs[0], F_.createNot("logical_not", glowInput));
  } else if (inputElementType == glow::ElemKind::Int32ITy ||
             inputElementType == glow::ElemKind::Int64ITy) {
    return addValueMapping(outputs[0],
                           F_.createBitwiseNot("bitwise_not", glowInput));
  } else {
    return MAKE_ERR("Expected integer/boolean tensor in loadBitwiseNot");
  }
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
  RETURN_ERR_IF_NOT(
      (numLayers == 1 && bidirectional) || (numLayers >= 1 && !bidirectional),
      "multiple layer bidirectional LSTM is not supported in Glow");

  if (batchFirst) {
    input = F_.createTranspose("Input_BatchFirst_Transpose", input, {1, 0, 2})
                ->getResult();
  }

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

  std::vector<NodeValue> WxTransposedVector, WhTransposedVector, BxVector,
      BhVector;
  for (int layer = 0; layer < numLayers; layer++) {
    auto *Wx = llvm::dyn_cast<glow::Constant>((*params)[paramsIdx++].getNode());
    auto *Wh = llvm::dyn_cast<glow::Constant>((*params)[paramsIdx++].getNode());
    glow::Constant *Bx = getBias("Bx_Constant"), *Bh = getBias("Bh_Constant");

    // W need to be transposed, in pt it is hiddenSize * inputSize,
    // in glow it is inputSize * hiddenSize.
    auto WxTransposed =
        F_.createTranspose("Wx_Transposed", Wx, {1, 0})->getResult();
    auto WhTransposed =
        F_.createTranspose("Wh_Transposed", Wh, {1, 0})->getResult();
    WxTransposedVector.push_back(WxTransposed);
    WhTransposedVector.push_back(WhTransposed);
    BxVector.emplace_back(Bx);
    BhVector.emplace_back(Bh);
  }

  NodeValue output;
  hn = h03D;
  cn = c03D;
  if (bidirectional) {
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
    F_.createPyTorchLSTM("lstm", input, WxTransposedVector, WhTransposedVector,
                         BxVector, BhVector, hn, cn, output, bidirectional,
                         WxRTransposed, WhRTransposed, BxR, BhR);
  } else {
    F_.createPyTorchLSTM("lstm", input, WxTransposedVector, WhTransposedVector,
                         BxVector, BhVector, hn, cn, output, bidirectional);
  }
  if (batchFirst) {
    output =
        F_.createTranspose("Output_BatchFirst_Transpose", output, {1, 0, 2})
            ->getResult();
  }
  RETURN_IF_ERR(addValueMapping(outputs[0], output));
  RETURN_IF_ERR(addValueMapping(outputs[1], hn));
  RETURN_IF_ERR(addValueMapping(outputs[2], cn));

  // Set all outputs to have the same correct type as the input
  at::ScalarType inputCorrectType;
  ASSIGN_VALUE_OR_RETURN_ERR(inputCorrectType,
                             getCorrectTypeMapping(inputs[LSTMInputs::input]));
  RETURN_IF_ERR(setCorrectTypeMapping(outputs[0], inputCorrectType));
  RETURN_IF_ERR(setCorrectTypeMapping(outputs[1], inputCorrectType));
  RETURN_IF_ERR(setCorrectTypeMapping(outputs[2], inputCorrectType));

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
    std::vector<glow::unsigned_t> pad;
    ASSIGN_VALUE_OR_RETURN_ERR(
        pad, castVector<glow::unsigned_t>(expandIntIValIfNeeded(
                 getGlowIValueForValue(inputs[ConvInputs::padding]), 2)));
    pads = {pad[0], pad[1], pad[0], pad[1]};
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

  auto output = F_.createLayerNormalization("layernorm", input.getType(), input,
                                            weight, bias, eps)
                    ->getResult();

  RETURN_ERR(addValueMapping(outputs[0], output));
}

Error PyTorchModelLoader::loadQuantizedLayerNorm(
    const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 7, outputs, 1));

  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(
      input, getGlowNodeValueForValue(inputs[QuantizedLayerNormInputs::input]));

  float eps = 1e-5;
  if (hasGlowIValueForValue(inputs[LayerNormInputs::eps])) {
    ASSIGN_VALUE_OR_RETURN_ERR(
        eps, iValToDouble(
                 getGlowIValueForValue(inputs[QuantizedLayerNormInputs::eps])));
  }

  std::vector<int64_t> *normalizedShape;
  ASSIGN_VALUE_OR_RETURN_ERR(
      normalizedShape,
      iValToIntList(getGlowIValueForValue(
          inputs[QuantizedLayerNormInputs::normalized_shape])));

  std::vector<glow::dim_t> normalizedShapeCast =
      castVector<glow::dim_t>(*normalizedShape);

  glow::NodeValue weight = loadNodeValueOrCreateBroadcastedConstant(
      inputs[LayerNormInputs::weight], "layernorm_weight",
      glow::Type(ElemKind::FloatTy, normalizedShapeCast), 1.0);

  glow::NodeValue bias = loadNodeValueOrCreateBroadcastedConstant(
      inputs[LayerNormInputs::bias], "layernorm_bias",
      glow::Type(ElemKind::FloatTy, normalizedShapeCast), 0.0);

  float outScale;
  ASSIGN_VALUE_OR_RETURN_ERR(
      outScale, iValToDouble(getGlowIValueForValue(
                    inputs[QuantizedLayerNormInputs::output_scale])));

  int32_t outOffset;
  ASSIGN_VALUE_OR_RETURN_ERR(
      outOffset, iValToInt(getGlowIValueForValue(
                     inputs[QuantizedLayerNormInputs::output_zero_point])));

  auto outTy =
      F_.getParent()->uniqueType(ElemKind::Int8QTy, input.dims(), outScale,
                                 outOffset - UINT8_TO_INT8_SHIFT);

  auto output =
      F_.createLayerNormalization("layernorm", outTy, input, weight, bias, eps)
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
  // 0D. Currently NNPI only supports 2D, will remove this after it supports
  // 0D.
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

  RETURN_ERR(addValueMapping(outputs[0], output));
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
  float outScale;
  int32_t outOffset;
  ASSIGN_VALUE_OR_RETURN_ERR(
      outScale, to32Bit(iValToDouble(getGlowIValueForValue(
                    inputs[QuantizedBatchNormInputs::output_scale]))));

  ASSIGN_VALUE_OR_RETURN_ERR(
      outOffset, iValToInt(getGlowIValueForValue(
                     inputs[QuantizedBatchNormInputs::output_zero_point])));

  // Input is in NCHW.
  glow::unsigned_t channelIdx = 1;
  std::string opName;
  if (numDims == 3) {
    opName = "bn3d_quant";
  } else {
    opName = "bn2d_quant";
  }

  glow::TypeRef outTy =
      F_.getParent()->uniqueType(glow::ElemKind::Int8QTy, input.dims(),
                                 outScale, outOffset - UINT8_TO_INT8_SHIFT);
  glow::BatchNormalizationNode *bn =
      F_.createBatchNormalization(opName, outTy, input, biasC, weightsC, meanC,
                                  varC, channelIdx, epsilon, momentum);
  return Expected<NodeValue>(bn->getResult());
}

Error PyTorchModelLoader::loadQuantizedBatchNorm2d(
    const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  glow::NodeValue output;
  ASSIGN_VALUE_OR_RETURN_ERR(output, loadQuantizedBatchNormImpl(ptNode, 2));

  RETURN_ERR(addValueMapping(outputs[0], output));
}

Error PyTorchModelLoader::loadQuantizedBatchNorm3d(
    const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  glow::NodeValue output;
  ASSIGN_VALUE_OR_RETURN_ERR(output, loadQuantizedBatchNormImpl(ptNode, 3));

  RETURN_ERR(addValueMapping(outputs[0], output));
}

Error PyTorchModelLoader::loadQuantizedBatchNorm3dRelu(
    const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  glow::NodeValue output;
  ASSIGN_VALUE_OR_RETURN_ERR(output, loadQuantizedBatchNormImpl(ptNode, 3));

  output = F_.createRELU("quantized_relu_after_bn", output);
  RETURN_ERR(addValueMapping(outputs[0], output));
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
  if (hasGlowIValueForValue(inputs[QuantizeInputs::scale])) {
    ASSIGN_VALUE_OR_RETURN_ERR(
        outScale, to32Bit(iValToDouble(
                      getGlowIValueForValue(inputs[QuantizeInputs::scale]))));
  } else {
    float scaleConstant;
    RETURN_IF_ERR(extractConstantFromNodeValue<float>(
        inputs[QuantizeInputs::scale], glow::ElemKind::FloatTy, scaleConstant));
    ASSIGN_VALUE_OR_RETURN_ERR(outScale, to32Bit((double)scaleConstant));
  }

  // zero_point
  int32_t outOffset;
  if (hasGlowIValueForValue(inputs[QuantizeInputs::zero_point])) {
    ASSIGN_VALUE_OR_RETURN_ERR(
        outOffset,
        iValToInt(getGlowIValueForValue(inputs[QuantizeInputs::zero_point])));
  } else {
    int32_t offsetConstant;
    RETURN_IF_ERR(extractConstantFromNodeValue<int32_t>(
        inputs[QuantizeInputs::zero_point], glow::ElemKind::Int32ITy,
        offsetConstant));
    ASSIGN_VALUE_OR_RETURN_ERR(outOffset, Expected<int32_t>(offsetConstant));
  }

  // dtype, we only support quantize to int8 for now
  int32_t outDtype;
  ASSIGN_VALUE_OR_RETURN_ERR(outDtype, iValToInt(getGlowIValueForValue(
                                           inputs[QuantizeInputs::dtype])));

  glow::TypeRef inputType = input.getType();
  auto outDims = inputType->dims();

  glow::TypeRef outTy;
  at::ScalarType correctType;
  if (outDtype == (int32_t)at::ScalarType::QUInt8) {
    outTy = F_.getParent()->uniqueType(ElemKind::Int8QTy, outDims, outScale,
                                       outOffset - UINT8_TO_INT8_SHIFT);
    correctType = at::ScalarType::QUInt8;

  } else if (outDtype == (int32_t)at::ScalarType::QInt8) {
    outTy = F_.getParent()->uniqueType(ElemKind::Int8QTy, outDims, outScale,
                                       outOffset);
    correctType = at::ScalarType::QInt8;
  } else {
    return MAKE_ERR("Quantize only supports QUInt8 and QInt8");
  }
  glow::QuantizeNode *qn = F_.createQuantize("quantize", input, outTy);

  RETURN_IF_ERR(addValueMapping(outputs[0], qn->getResult()));
  RETURN_IF_ERR(setCorrectTypeMapping(outputs[0], correctType));
  return Error::success();
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

  RETURN_ERR(addValueMapping(outputs[0], output));
}

Error PyTorchModelLoader::loadQuantizedConv(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  glow::NodeValue output;
  ASSIGN_VALUE_OR_RETURN_ERR(output,
                             loadQuantizedConvImpl(ptNode, false /* isRelu */));

  RETURN_ERR(addValueMapping(outputs[0], output));
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

  RETURN_ERR(addValueMapping(outputs[0], output));
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
      F_.createMaxPool("maxpool2d", input, kernels, strides, pads);
  glow::NodeValue output = mp->getResult();
  output = F_.createTranspose("maxpool2d_output_transposed", output, NHWC2NCHW);

  RETURN_ERR(addValueMapping(outputs[0], output));
}

Expected<NodeValue>
PyTorchModelLoader::loadAvgPoolImpl(const torch::jit::Node *ptNode,
                                    int numDims) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, -6, outputs, 1));

  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(
      input, getGlowNodeValueForValue(inputs[AvgPoolInputs::input]));
  bool is3d = (numDims == 3);
  bool is1d = (numDims == 1);
  // aten::avg_pool1d does not have 7th input: divisor_override
  if (is1d) {
    RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 6, outputs, 1));
  } else {
    RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 7, outputs, 1));
  }
  std::string opName = is3d ? "avgpool3d" : (is1d ? "avgpool1d" : "avgpool2d");

  // For 1d avgpool N*M*L, we reshape it to N*M*L*1, and run avgpool2d on it,
  // and finally reshape it back.
  if (is1d) {
    input = F_.createReshape(
        opName + "_reshape_to_2d", input,
        {input.dims()[0], input.dims()[1], input.dims()[2], 1});
  }

  if (is3d) {
    input =
        F_.createTranspose(opName + "_input_transposed", input, NCTHW2NTHWC);
  } else {
    input = F_.createTranspose(opName + "_input_transposed", input, NCHW2NHWC);
  }

  // We also change the kernel, pads and stride to 2d for avgpool1d.
  std::vector<glow::unsigned_t> kernels;
  ASSIGN_VALUE_OR_RETURN_ERR(
      kernels,
      castVector<glow::unsigned_t>(expandIntIValIfNeeded(
          getGlowIValueForValue(inputs[AvgPoolInputs::kernel_size]), numDims)));
  if (is1d) {
    kernels.push_back(1);
  }

  std::vector<glow::unsigned_t> padsPair;
  ASSIGN_VALUE_OR_RETURN_ERR(
      padsPair,
      castVector<glow::unsigned_t>(expandIntIValIfNeeded(
          getGlowIValueForValue(inputs[AvgPoolInputs::padding]), numDims)));
  RETURN_ERR_IF_NOT(padsPair.size() == numDims,
                    "Number of pad values is incorrect");
  if (is1d) {
    padsPair.push_back(0);
  }
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
    if (is1d) {
      strides.push_back(1);
    }
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
  glow::NodeValue output = ap->getResult();

  if (is3d) {
    output =
        F_.createTranspose(opName + "_output_transposed", output, NTHWC2NCTHW)
            ->getResult();
  } else {
    output =
        F_.createTranspose(opName + "_output_transposed", output, NHWC2NCHW)
            ->getResult();
  }
  // Reshape back to 1d
  if (is1d) {
    output = F_.createReshape(
        opName + "_reshape_to_1d", output,
        {output.dims()[0], output.dims()[1], output.dims()[2]});
  }

  return Expected<NodeValue>(output);
}

template <size_t numDims>
Error PyTorchModelLoader::loadAvgPool(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  glow::NodeValue output;
  ASSIGN_VALUE_OR_RETURN_ERR(output, loadAvgPoolImpl(ptNode, numDims));
  RETURN_ERR(addValueMapping(outputs[0], output));
}

Expected<NodeValue>
PyTorchModelLoader::getClampMinMax(const torch::jit::Value *value,
                                   const glow::NodeValue input,
                                   const std::string name) {
  if (hasGlowIValueForValue(value, true)) {
    glow::GlowIValue *glowIValue;
    ASSIGN_VALUE_OR_RETURN_ERR(glowIValue, getGlowIValueForValue(value));
    if (glowIValue->isDouble()) {
      double vDouble;
      float vFloat;
      ASSIGN_VALUE_OR_RETURN_ERR(vDouble, iValToDouble(glowIValue));
      ASSIGN_VALUE_OR_RETURN_ERR(vFloat, to32Bit(vDouble));
      glow::NodeValue SN = F_.createSplat(name, input.getType(), vFloat);
      return SN;
    } else if (glowIValue->isInt()) {
      int vInt;
      ASSIGN_VALUE_OR_RETURN_ERR(vInt, iValToInt(glowIValue));
      glow::NodeValue SN = F_.createSplat(name, input.getType(), vInt);
      return SN;
    } else {
      return MAKE_ERR(strFormat("Unexpected Min/Max dtype: %s",
                                glowIValue->getTagString()));
    }
  } else {
    return nullptr;
  }
}

Error PyTorchModelLoader::loadClamp(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 3, outputs, 1));

  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(
      input, getGlowNodeValueForValue(inputs[ClampInputs::input]));

  NodeValue minSN, maxSN;
  ASSIGN_VALUE_OR_RETURN_ERR(
      minSN, getClampMinMax(inputs[ClampInputs::min], input, "minValue"));
  ASSIGN_VALUE_OR_RETURN_ERR(
      maxSN, getClampMinMax(inputs[ClampInputs::max], input, "maxValue"));

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

Error PyTorchModelLoader::loadExpand(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 3, outputs, 1));

  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(
      input, getGlowNodeValueForValue(inputs[ExpandInputs::input]));
  auto shapeOriginal = input.dims();

  std::vector<int64_t> *shapeInput;
  ASSIGN_VALUE_OR_RETURN_ERR(shapeInput, iValToIntList(getGlowIValueForValue(
                                             inputs[ExpandInputs::shape])));

  // Copy shapeInput so we can modify it.
  std::vector<int64_t> shapeDesired = *shapeInput;

  if (shapeDesired.size() < shapeOriginal.size()) {
    return MAKE_ERR(
        strFormat("Expanded shape may not have fewer dimensions than original "
                  "shape : %ld vs %ld",
                  shapeDesired.size(), shapeOriginal.size()));
  }

  // -1 for some dimension means keep its original size, so loop through
  // shapeDesired and perform this lookup where needed. Also check for
  // invalid input (<0 && !=-1). Note that the desired shape is aligned
  // to the *end* of the original shape.
  auto iterDesired = shapeDesired.rbegin();
  auto iterOriginal = shapeOriginal.rbegin();
  while (iterDesired != shapeDesired.rend()) {
    if (*iterDesired == 0 || *iterDesired < -1 ||
        (*iterDesired == -1 && iterOriginal == shapeOriginal.rend())) {
      return MAKE_ERR(
          strFormat("Could not determine a size for dimension %" PRId64,
                    std::distance(iterDesired, shapeDesired.rend())));
    }
    if (iterOriginal != shapeOriginal.rend()) {
      if (*iterDesired == -1) {
        *iterDesired = *iterOriginal;
      }
      ++iterOriginal;
    }
    ++iterDesired;
  }

  auto output =
      F_.createBroadcast("expand", input, castVector<dim_t>(shapeDesired),
                         shapeDesired.size() - shapeOriginal.size());
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

  RETURN_ERR(addValueMapping(outputs[0], output));
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
  RETURN_ERR(addValueMapping(outputs[0], output));
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

  RETURN_ERR(addValueMapping(outputs[0], output->getResult()));
}

Error PyTorchModelLoader::loadMin(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, -1, outputs, 1));
  if (inputs.size() == 2) {
    // Binary elementwise min, return a tensor has same type to input
    glow::NodeValue lhs;
    ASSIGN_VALUE_OR_RETURN_ERR(lhs, getGlowNodeValueForValue(inputs[0]));
    glow::NodeValue rhs;
    ASSIGN_VALUE_OR_RETURN_ERR(rhs, getGlowNodeValueForValue(inputs[1]));

    glow::MinNode *glowNode =
        F_.createNodeWithBroadcast<MinNode>("min", -1, lhs, rhs);
    RETURN_ERR(addValueMapping(outputs[0], glowNode->getResult()));
  } else {
    // Unary min, return a scalar contains the biggest element in input
    glow::NodeValue input;
    ASSIGN_VALUE_OR_RETURN_ERR(input, getGlowNodeValueForValue(inputs[0]));

    auto n = input.dims().size();
    size_t length = 1;
    for (int i = 0; i < n; i++) {
      length *= input.dims()[i];
    }
    auto reshaped = F_.createReshape("reshaped_flatten_input", input, {length})
                        ->getResult();
    reshaped = F_.createBatchedReduceMin("batched_reduce_min_for_unary_min",
                                         reshaped, 0)
                   ->getResult();
    RETURN_ERR(addValueMapping(outputs[0], reshaped));
  }
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
      // Wrap dimension if it is negative.
      if (i < 0) {
        i += input.dims().size();
      }
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
      std::vector<dim_t> shape = input.dims();
      std::sort(axis->begin(), axis->end());
      for (auto i : *axis) {
        shape.insert(shape.begin() + i, static_cast<dim_t>(1));
      }
      input = F_.createReshape("reshape", input, shape)->getResult();
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
  bool keepDim;

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
    ASSIGN_VALUE_OR_RETURN_ERR(keepDim,
                               iValToBool(getGlowIValueForValue(inputs[2])));
  } else {
    // With p in torch.norm(input, p,  dim), inputs[1] is the int representing
    // p
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

    // With p in torch.norm(input, p, dim), inputs[2] is the list of int
    // representing axis/dim
    std::vector<int64_t> *axisList;
    ASSIGN_VALUE_OR_RETURN_ERR(axisList,
                               iValToIntList(getGlowIValueForValue(inputs[2])));
    RETURN_ERR_IF_NOT(axisList->size() == 1,
                      glow::strFormat("we currently only support 1 dimension "
                                      "of axis, but got dimension size = %lu",
                                      axisList->size()));
    axis = axisList->front();

    ASSIGN_VALUE_OR_RETURN_ERR(keepDim,
                               iValToBool(getGlowIValueForValue(inputs[3])));
  }

  NodeValue output = F_.createVectorNorm("norm", input, axis, p);

  if (keepDim) {
    std::vector<uint64_t> newShape;
    for (int64_t i = 0; i < input.dims().size(); i++) {
      newShape.push_back(i == axis ? 1 : input.dims()[i]);
    }
    output = F_.createReshape("norm_reshape", output, newShape);
  }

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

    // If a 1 was prepended to lhs or a 1 was appended to rhs at the
    // beginning, undo that now.
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

  if (hasGlowIValueForValue(inputs[SliceInputs::start], true)) {
    ASSIGN_VALUE_OR_RETURN_ERR(
        start, iValToInt(getGlowIValueForValue(inputs[SliceInputs::start])));
  } else {
    start = 0;
  }

  if (hasGlowIValueForValue(inputs[SliceInputs::end], true)) {
    ASSIGN_VALUE_OR_RETURN_ERR(
        end, iValToInt(getGlowIValueForValue(inputs[SliceInputs::end])));
  } else {
    end = std::numeric_limits<int64_t>::max();
  }

  ASSIGN_VALUE_OR_RETURN_ERR(
      step, iValToInt(getGlowIValueForValue(inputs[SliceInputs::step])));

  RETURN_ERR_IF_NOT(step == 1, "loadSlice only supports step == 1");
  glow::dim_t rank = input.dims().size();
  std::vector<glow::dim_t> begins(rank);
  std::vector<glow::dim_t> ends(rank);

  for (glow::dim_t i = 0; i < rank; ++i) {
    const glow::dim_t dimSize = input.dims()[i];

    // Only slice along dim
    if (i != dim) {
      begins[i] = 0;
      ends[i] = dimSize;
      continue;
    }

    int64_t startPositive, endPositive;
    ASSIGN_VALUE_OR_RETURN_ERR(startPositive, getPositiveIndex(start, dimSize));

    if (end < 0) {
      ASSIGN_VALUE_OR_RETURN_ERR(endPositive, getPositiveIndex(end, dimSize));
    } else if (end > dimSize) {
      // Ensure that end is no bigger than dimSize, if end isn't provided then
      // it will be set to int64 maximum.
      endPositive = dimSize;
    } else {
      endPositive = end;
    }

    RETURN_ERR_IF_NOT(
        endPositive > startPositive,
        strFormat(
            "End index must be greater than start index got start=%ld "
            "end=%ld "
            "which was translated into positive range as start=%ld end=%ld",
            start, end, startPositive, endPositive)
            .c_str());

    begins[i] = startPositive;
    ends[i] = endPositive;
  }
  auto *glowNode = F_.createSlice("sliceOutput", input, begins, ends);

  RETURN_ERR(addValueMapping(outputs[0], glowNode));
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

Error PyTorchModelLoader::loadLogSoftMax(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 3, outputs, 1));

  glow::NodeValue in;
  ASSIGN_VALUE_OR_RETURN_ERR(
      in, getGlowNodeValueForValue(inputs[LogSoftMaxInputs::input]));

  const auto inDims = in.dims();

  int64_t dim;
  ASSIGN_VALUE_OR_RETURN_ERR(
      dim, iValToInt(getGlowIValueForValue(inputs[LogSoftMaxInputs::dim])));

  ASSIGN_VALUE_OR_RETURN_ERR(dim, getPositiveIndex(dim, inDims.size()));

  // transpose dim to inner dimension
  std::vector<unsigned_t> inTransposeShuffle;
  for (auto i = 0; i < inDims.size(); ++i) {
    inTransposeShuffle.push_back(i);
  }
  inTransposeShuffle.erase(inTransposeShuffle.begin() + dim);
  inTransposeShuffle.push_back(dim);
  in =
      F_.createTranspose("transpose_before_log_softmax", in, inTransposeShuffle)
          ->getResult();

  // flatten outer dims
  auto dimsBeforeFlatten = in.dims();
  in = F_.createFlatten("flatten_before_log_softmax", in, inDims.size() - 1);

  // LogSoftmax
  auto selected = F_.getParent()->createConstant(
      glow::ElemKind::Int64ITy, {in.dims()[0], 1}, "log_softmax_selected");
  auto out = F_.createLogSoftMax("log_softmax", in, selected)->getResult();

  // unflatten
  out = F_.createReshape("reshape_after_log_softmax", out, dimsBeforeFlatten);

  // transpose dim back to where it started
  auto outTransposeShufle = getInverseTranspose(inTransposeShuffle);
  out =
      F_.createTranspose("transpose_after_log_softmax", out, outTransposeShufle)
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

  RETURN_ERR(addValueMapping(outputs[0], output));
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
            RETURN_IF_ERR(addValueMapping(outputs[0], input));
            RETURN_IF_ERR(setCorrectTypeMappingSameAs(
                outputs[0], inputs[ToDtypeLayoutInputs::input]));
            return Error::success();
          }
        } else {
          RETURN_ERR(glowDtypeVal.takeError());
        }
        dtype_arg = ToPrimDeviceInputs::dtype;
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
  auto correctType = static_cast<at::ScalarType>(dtype);
  auto glowElemKind = correctType == at::kLong
                          ? ElemKind::Int32ITy
                          : scalarTypeToElemKind(correctType);

  // No type conversion necessary
  if (glowElemKind == inputType->getElementType()) {
    RETURN_IF_ERR(addValueMapping(outputs[0], input));
    RETURN_IF_ERR(setCorrectTypeMapping(outputs[0], correctType));
    return Error::success();
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
  RETURN_IF_ERR(addValueMapping(outputs[0], toNode->getResult()));
  RETURN_IF_ERR(setCorrectTypeMapping(outputs[0], correctType));
  return Error::success();
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

  RETURN_ERR(addValueMapping(outputs[0], res));
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

  RETURN_ERR(addValueMapping(outputs[0], res));
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

  RETURN_ERR(addValueMapping(outputs[0], res));
}

Error PyTorchModelLoader::loadUnsqueezeNTimes(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 2, outputs, 1));

  glow::NodeValue in;
  ASSIGN_VALUE_OR_RETURN_ERR(
      in, getGlowNodeValueForValue(inputs[UnsqueezeNTimesInputs::input]));
  auto inDims = in.dims();

  int64_t n = 0;
  ASSIGN_VALUE_OR_RETURN_ERR(
      n, iValToInt(getGlowIValueForValue(inputs[UnsqueezeNTimesInputs::n])));

  std::vector<dim_t> outDims(inDims.begin(), inDims.end());
  for (int i = 0; i < n; i++) {
    // according to unsqueeze_n_times definition:
    // for i in range(0, n):
    //   x = torch::unsqueeze(x, -1)
    // we should always insert a dimension of size one at the end
    outDims.push_back(1);
  }
  auto res =
      F_.createReshape("reshape_unsqueeze_n_times", in, outDims)->getResult();
  RETURN_ERR(addValueMapping(outputs[0], res));
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

  RETURN_IF_ERR(setCorrectTypeMappingSameAs(outputs[0], inputs[0]));
  RETURN_IF_ERR(setCorrectTypeMapping(outputs[1], at::kLong));
  return Error::success();
}

Error PyTorchModelLoader::loadArgSort(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 3, outputs, 1));

  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(input, getGlowNodeValueForValue(inputs[0]));

  int64_t dim;
  ASSIGN_VALUE_OR_RETURN_ERR(
      dim, iValToInt(getGlowIValueForValue(ptNode->namedInput("dim"))));
  if (dim < 0)
    dim += input.dims().size();

  RETURN_ERR_IF_NOT(dim == input.dims().size() - 1,
                    "aten::argsort is only supported along the last dimension");

  bool descending = true;
  ASSIGN_VALUE_OR_RETURN_ERR(
      descending,
      iValToBool(getGlowIValueForValue(ptNode->namedInput("descending"))));

  auto output = F_.createTopK("argsort", input, /* k = */ input.dims().back())
                    ->getIndices();

  if (!descending) {
    output = F_.createFlip("flip_output", output, dim)->getResult();
  }

  RETURN_ERR(addValueMapping(outputs[0], output));
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
    RETURN_IF_ERR(setCorrectTypeMappingSameAs(outputs[i], inputs[0]));
  }
  return Error::success();
}

static NodeValue convertInt64ConstantToInt32(NodeValue constant,
                                             glow::Function &F) {
  LOG(WARNING) << "Loading PyTorch int64 Tensor Constant as int32 "
                  "because int64 isn't supported";
  // For int64 constants, convert them to int32 since many accelerators
  // don't support int64
  return F.createConvertTo("int64_to_int32", constant, ElemKind::Int32ITy)
      ->getResult();
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
      if (glowConstantNodeValue.getElementType() == ElemKind::Int64ITy) {
        glowConstantNodeValue =
            convertInt64ConstantToInt32(glowConstantNodeValue, F_);
      }
      constantNodeValueList.push_back(glowConstantNodeValue);
    }
    glowIVal.fromNodeValueList(constantNodeValueList);
  } else if (iValList[0].isInt()) {
    std::vector<int64_t> intList;
    for (auto v : iValList) {
      RETURN_ERR_IF_NOT(v.isInt(),
                        strFormat("Expect all ival in a PyTorch GenericList "
                                  "to be Int, but got %s.",
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

  // If it is a qparam node, we deal with it separately.
  if (iVal.isObject()) {
    if (isQParamWeightNode(ptNode)) {
      qparamsMap_[ptNode->output()] = iVal;
      return Error::success();
    }
  }

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

    at::ScalarType correctType =
        elemKindToScalarType(glowConstant->getElementType());
    NodeValue out = glowConstant->getOutput();
    if (glowConstant->getElementType() == ElemKind::Int64ITy) {
      out = convertInt64ConstantToInt32(out, F_);
      correctType = at::kLong;
    }

    RETURN_IF_ERR(addValueMapping(outputs[0], out));
    RETURN_IF_ERR(setCorrectTypeMapping(outputs[0], correctType));
  } else if (glowIVal.isNodeValueList()) {
    std::vector<glow::NodeValue> *nodeValues;
    ASSIGN_VALUE_OR_RETURN_ERR(nodeValues, glowIVal.toNodeValueList());
    std::vector<at::ScalarType> correctTypes;
    for (const auto &nodeValue : *nodeValues) {
      // If it's not a constant node then it must be a convert node which
      // convert the constant from int64 to int32. In this case, the correct
      // type is at::kLong.
      if (nodeValue.getNode()->getKind() == Kinded::Kind::ConstantKind) {
        correctTypes.push_back(
            elemKindToScalarType(nodeValue.getElementType()));
      } else {
        correctTypes.push_back(at::kLong);
      }
    }
    RETURN_IF_ERR(addValueMapping(outputs[0], std::move(glowIVal)));
    RETURN_IF_ERR(setCorrectTypesMapping(outputs[0], correctTypes));
  } else {
    RETURN_ERR(addValueMapping(outputs[0], std::move(glowIVal)));
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

Expected<std::vector<std::vector<at::ScalarType>>>
PyTorchModelLoader::getCustomOpCorrectType(const torch::jit::Node *ptNode) {
  CustomPyTorchOpLoader *customLoader =
      getCustomPyTorchOpLoaderForSymbol(ptNode->kind());

  RETURN_ERR_IF_NOT(
      customLoader,
      strFormat("Expected a custom loader to be found for symbol: %s",
                ptNode->kind().toQualString()));

  return customLoader->getCorrectType(*this, ptNode);
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

Error PyTorchModelLoader::loadXLEmbeddingBag(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, -10, outputs, 4));
  // get the shape (num_embeddings, embedding_dim) and qualName for the
  // embeddingBag, and create placeholder node
  glow::dim_t numEmbedding;
  ASSIGN_VALUE_OR_RETURN_ERR(
      numEmbedding, iValToInt(getGlowIValueForValue(
                        inputs[XLEmbeddingBagInputs::num_embeddings])));
  glow::dim_t embeddingDim;
  ASSIGN_VALUE_OR_RETURN_ERR(embeddingDim,
                             iValToInt(getGlowIValueForValue(
                                 inputs[XLEmbeddingBagInputs::embedding_dim])));
  std::string *weightID;
  ASSIGN_VALUE_OR_RETURN_ERR(weightID,
                             iValToString(getGlowIValueForValue(
                                 inputs[XLEmbeddingBagInputs::weight_id])));
  std::vector<glow::dim_t> dims{numEmbedding, embeddingDim};
  glow::Type phType(ElemKind::FloatTy, dims);
  auto legalizedWeightID = glow::legalizeName(*weightID);
  auto ph = F_.getParent()->getPlaceholderByNameSlow(legalizedWeightID);
  if (!ph) {
    ph = F_.getParent()->createPlaceholder(&phType, legalizedWeightID,
                                           /*isTrainable*/ false);
    ph->setStatic(true);
  }
  glow::NodeValue indices;
  ASSIGN_VALUE_OR_RETURN_ERR(
      indices, getGlowNodeValueForValue(inputs[XLEmbeddingBagInputs::indices]));
  glow::NodeValue offsets;
  ASSIGN_VALUE_OR_RETURN_ERR(
      offsets, getGlowNodeValueForValue(inputs[XLEmbeddingBagInputs::offsets]));

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
          inputs[XLEmbeddingBagInputs::include_last_offset])));
  RETURN_ERR_IF_NOT(includeLastOffset,
                    "Currently only support include_last_offset='True'");
  glow::NodeValue perSampleWeights = loadNodeValueOrCreateBroadcastedConstant(
      inputs[XLEmbeddingBagInputs::per_sample_weights], "EmbeddingBag.ones",
      glow::Type(ElemKind::FloatTy, {indices.dims()[0]}), 1.0);

  float avgLength = NAN;
  if (inputs.size() > XLEmbeddingBagInputs::avg_length) {
    double avgLenValue;
    ASSIGN_VALUE_OR_RETURN_ERR(avgLenValue,
                               iValToDouble(getGlowIValueForValue(
                                   inputs[XLEmbeddingBagInputs::avg_length])));
    LOG(INFO) << "Loading avg length " << avgLenValue
              << " for xlEmbeddingBag with weightID " << legalizedWeightID;
    if (avgLenValue >= 0) {
      avgLength = avgLenValue;
    }
  }

  auto *EB = F_.createEmbeddingBag(
      "XLEmbeddingBag", ph->getOutput(), perSampleWeights, indices, offsets,
      includeLastOffset, LengthsMode::Variable, avgLength);
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

  // Upcast EmbeddingBag4BitRowwiseOffsets to Float32 since its Glow output
  // type is Float16.
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
  RETURN_IF_ERR(addValueMapping(outputs[0], glowNode));

  // Also assume that the correct type of the output ist he same as the input
  RETURN_IF_ERR(setCorrectTypeMappingSameAs(outputs[0], inputs[0]));

  return Error::success();
}

Error PyTorchModelLoader::loadCumSum(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 3, outputs, 1));

  glow::NodeValue in;
  ASSIGN_VALUE_OR_RETURN_ERR(
      in, getGlowNodeValueForValue(inputs[CumSumInputs::input]));

  int64_t dim;
  ASSIGN_VALUE_OR_RETURN_ERR(
      dim, iValToInt(getGlowIValueForValue(inputs[CumSumInputs::dim])));

  int64_t nDims = in.dims().size();
  RETURN_ERR_IF_NOT(
      (dim >= -nDims) && (dim < nDims),
      glow::strFormat("cumsum dim (%ld) is expected to be in range [%ld, %ld]",
                      dim, -nDims, nDims - 1));

  // Convert to supplied dtype is specified
  GlowIValue *dtypeIVal = nullptr;
  ASSIGN_VALUE_OR_RETURN_ERR(
      dtypeIVal, getGlowIValueForValue(inputs[CumSumInputs::dtype]));

  RETURN_IF_ERR(addValueMapping(
      outputs[0], F_.createCumSum("cumsum", in, dim)->getResult()));

  // Default value for dtype is None
  if (dtypeIVal->getTag() != GlowIValue::Tag::None) {
    int32_t dtype;
    ASSIGN_VALUE_OR_RETURN_ERR(dtype, iValToInt(dtypeIVal));
    auto correctType = static_cast<at::ScalarType>(dtype);
    const auto glowElemKind = correctType == at::kLong
                                  ? ElemKind::Int32ITy
                                  : scalarTypeToElemKind(correctType);
    in = F_.createConvertTo("cast", in, glowElemKind)->getResult();
    RETURN_IF_ERR(setCorrectTypeMapping(outputs[0], correctType));
  } else {
    RETURN_IF_ERR(setCorrectTypeMappingSameAs(outputs[0], inputs[0]));
  }

  return Error::success();
}

Error PyTorchModelLoader::loadEquallySplit(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 3, outputs, 1));

  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(
      input, getGlowNodeValueForValue(inputs[FusedSplitInputs::input]));

  int num_split;
  ASSIGN_VALUE_OR_RETURN_ERR(
      num_split,
      iValToInt(getGlowIValueForValue(inputs[FusedSplitInputs::num_split])));

  int dimRaw;
  ASSIGN_VALUE_OR_RETURN_ERR(
      dimRaw, iValToInt(getGlowIValueForValue(inputs[FusedSplitInputs::dim])));
  int dim = 0;
  ASSIGN_VALUE_OR_RETURN_ERR(dim,
                             getPositiveIndex(dimRaw, input.dims().size()));

  std::vector<glow::SliceNode *> splitOutputs;
  F_.createSplit("EquallySplit", input, num_split, dim, {}, splitOutputs);

  std::vector<glow::NodeValue> outputNodeValues;
  for (auto o : splitOutputs) {
    outputNodeValues.emplace_back(o);
  }
  GlowIValue glowIVal;
  glowIVal.fromNodeValueList(std::move(outputNodeValues));
  RETURN_IF_ERR(addValueMapping(outputs[0], std::move(glowIVal)));

  // Each output tensor in the vector should have the same correct type as the
  // input.
  at::ScalarType inputCorrectType;
  ASSIGN_VALUE_OR_RETURN_ERR(inputCorrectType,
                             getCorrectTypeMapping(inputs[0]));
  std::vector<at::ScalarType> outputCorrectTypes(num_split, inputCorrectType);
  RETURN_IF_ERR(setCorrectTypesMapping(outputs[0], outputCorrectTypes));

  return Error::success();
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

  // Note that QuantizedGlowEBB infers dims directly from embeddingDim, while
  // QuantizeEBB infers dims from weights. To reuse QuantizeEBB's loading
  // logic, we scale the embedding dim of the static PH to match the second
  // dims of the corresponding weight tensor of a QuantizeEBB:
  // For 4bit: (weightShape[1] - 4) * 2 = embeddingDim
  //           =>  scaledEmbeddingDim = embeddingDim / 2 + 4
  // For byte: weightShape[1] - 8 = embeddingDim
  //           =>  scaledEmbeddingDim = embeddingDim + 8
  glow::dim_t scaledEmbeddingDim =
      is4Bit ? embeddingDim / 2 + 4 : embeddingDim + 8;

  std::vector<glow::dim_t> dims{numEmbedding, scaledEmbeddingDim};
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

  // Upcast EmbeddingBag4BitRowwiseOffsets to Float32 since its Glow output
  // type is Float16.
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

Error PyTorchModelLoader::loadRowwiseQuantizedXLEmbeddingBagHelper(
    const torch::jit::Node *ptNode, bool is4Bit) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, -11, outputs, 1));

  glow::dim_t numEmbedding;
  ASSIGN_VALUE_OR_RETURN_ERR(
      numEmbedding,
      iValToInt(getGlowIValueForValue(
          inputs[XLEmbeddingBagRowwiseOffsetsInputs::num_embeddings])));

  glow::dim_t embeddingDim;
  ASSIGN_VALUE_OR_RETURN_ERR(
      embeddingDim,
      iValToInt(getGlowIValueForValue(
          inputs[XLEmbeddingBagRowwiseOffsetsInputs::embedding_dim])));

  std::string *weightID;
  ASSIGN_VALUE_OR_RETURN_ERR(
      weightID, iValToString(getGlowIValueForValue(
                    inputs[XLEmbeddingBagRowwiseOffsetsInputs::weight_id])));

  // Note that QuantizedXLEBB infers dims directly from embeddingDim, while
  // QuantizeEBB infers dims from weights. To reuse QuantizeEBB's loading
  // logic, we scale the embedding dim of the static PH to match the second
  // dims of the corresponding weight tensor of a QuantizeEBB:
  // For 4bit: (weightShape[1] - 4) * 2 = embeddingDim
  //           =>  scaledEmbeddingDim = embeddingDim / 2 + 4
  // For byte: weightShape[1] - 8 = embeddingDim
  //           =>  scaledEmbeddingDim = embeddingDim + 8
  glow::dim_t scaledEmbeddingDim =
      is4Bit ? embeddingDim / 2 + 4 : embeddingDim + 8;

  std::vector<glow::dim_t> dims{numEmbedding, scaledEmbeddingDim};
  TypeRef fusedTy = F_.getParent()->uniqueType(
      (is4Bit ? ElemKind::UInt4FusedFP16QTy : ElemKind::UInt8FusedQTy), dims,
      0.0, 0);
  glow::Placeholder *ph =
      F_.getParent()->createPlaceholder(fusedTy, *weightID,
                                        /*isTrainable*/ false);
  ph->setStatic(true);

  glow::NodeValue indices;
  ASSIGN_VALUE_OR_RETURN_ERR(
      indices, getGlowNodeValueForValue(
                   inputs[XLEmbeddingBagRowwiseOffsetsInputs::indices]));

  glow::NodeValue offsets;
  ASSIGN_VALUE_OR_RETURN_ERR(
      offsets, getGlowNodeValueForValue(
                   inputs[XLEmbeddingBagRowwiseOffsetsInputs::offsets]));

  bool includeLastOffset;
  ASSIGN_VALUE_OR_RETURN_ERR(
      includeLastOffset,
      iValToBool(getGlowIValueForValue(
          inputs[XLEmbeddingBagRowwiseOffsetsInputs::include_last_offset])));
  RETURN_ERR_IF_NOT(includeLastOffset,
                    "Currently only support include_last_offset='True'");

  // If no indices are provided, replace the op with a zero Constant.
  if (indices.dims()[0] == 0) {
    // Assuming hasEndOffset = true, so the output.dims[0] should be
    // offsets.dims[0] - 1, if offsets is not empty
    glow::Tensor t(
        (is4Bit ? ElemKind::UInt4FusedFP16QTy : ElemKind::UInt8FusedQTy),
        {offsets.dims()[0] > 0 ? offsets.dims()[0] - 1 : 0,
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
    auto node = inputs[XLEmbeddingBagRowwiseOffsetsInputs::per_sample_weights];
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
        inputs[XLEmbeddingBagInputs::per_sample_weights],
        "EmbeddingBagByteRowwiseOffsets.ones",
        glow::Type((ElemKind::FloatTy), {indices.dims()[0]}), 1.0);
  }

  float avgLength = NAN;
  if (inputs.size() > XLEmbeddingBagRowwiseOffsetsInputs::avg_length) {
    double avgLenValue;
    ASSIGN_VALUE_OR_RETURN_ERR(
        avgLenValue,
        iValToDouble(getGlowIValueForValue(
            inputs[XLEmbeddingBagRowwiseOffsetsInputs::avg_length])));
    LOG(INFO) << "Loading avg length " << avgLenValue
              << " for quantized xlEmbeddingBag with weightID " << weightID;
    if (avgLenValue >= 0) {
      avgLength = avgLenValue;
    }
  }

  auto *EB = F_.createEmbeddingBagByteRowwiseOffsets(
      (is4Bit ? "EmbeddingBag4BitRowwiseOffsets"
              : "EmbeddingBagByteRowwiseOffsets"),
      ph->getOutput(), perSampleWeights, indices, offsets, false,
      includeLastOffset, LengthsMode::Variable, avgLength);

  // Upcast EmbeddingBag4BitRowwiseOffsets to Float32 since its Glow output
  // type is Float16.
  if (is4Bit) {
    auto *CT = F_.createConvertTo("ConvertEmbeddingBag4BitRowwiseOffsetsOutput",
                                  EB, ElemKind::FloatTy);
    RETURN_ERR(addValueMapping(outputs[0], CT->getResult()));
  } else {
    RETURN_ERR(addValueMapping(outputs[0], EB->getResult()));
  };
}

Error PyTorchModelLoader::loadXLEmbeddingBagByteRowwiseOffsets(
    const torch::jit::Node *ptNode) {
  return loadRowwiseQuantizedXLEmbeddingBagHelper(ptNode, false);
}

Error PyTorchModelLoader::loadXLEmbeddingBag4bitRowwiseOffsets(
    const torch::jit::Node *ptNode) {
  return loadRowwiseQuantizedXLEmbeddingBagHelper(ptNode, true);
}

Error PyTorchModelLoader::loadSplit(const torch::jit::Node *ptNode) {
  RETURN_IF_ERR(
      checkInputAndOutputSizes(ptNode->inputs(), 3, ptNode->outputs(), 1));
  glow::NodeValue input;
  int64_t dimension;
  ASSIGN_VALUE_OR_RETURN_ERR(input, getGlowNodeValueForValue(ptNode->input(0)));
  ASSIGN_VALUE_OR_RETURN_ERR(
      dimension, iValToInt(getGlowIValueForValue(ptNode->input(2))));

  unsigned int dimensionPositive;
  ASSIGN_VALUE_OR_RETURN_ERR(dimensionPositive,
                             getPositiveIndex(dimension, input.dims().size()));

  uint64_t chunkSize;
  ASSIGN_VALUE_OR_RETURN_ERR(
      chunkSize, iValToInt(getGlowIValueForValue(ptNode->input(1))));
  std::vector<glow::NodeValue> chunks;
  ASSIGN_VALUE_OR_RETURN_ERR(
      chunks, loadSplitImpl(input, dimensionPositive, chunkSize));
  size_t numChunks = chunks.size();
  glow::GlowIValue output;
  output.fromNodeValueList(chunks);
  RETURN_IF_ERR(addValueMapping(ptNode->output(0), std::move(output)));

  // Each output tensor in the vector should have the same correct type as the
  // input.
  at::ScalarType inputCorrectType;
  ASSIGN_VALUE_OR_RETURN_ERR(inputCorrectType,
                             getCorrectTypeMapping(ptNode->input(0)));
  std::vector<at::ScalarType> outputCorrectTypes(numChunks, inputCorrectType);
  RETURN_IF_ERR(setCorrectTypesMapping(ptNode->output(0), outputCorrectTypes));

  return Error::success();
}

Error PyTorchModelLoader::loadSplitWithSizes(const torch::jit::Node *ptNode) {
  RETURN_IF_ERR(
      checkInputAndOutputSizes(ptNode->inputs(), 3, ptNode->outputs(), 1));
  glow::NodeValue input;
  int64_t dimension;
  ASSIGN_VALUE_OR_RETURN_ERR(input, getGlowNodeValueForValue(ptNode->input(0)));
  ASSIGN_VALUE_OR_RETURN_ERR(
      dimension, iValToInt(getGlowIValueForValue(ptNode->input(2))));

  unsigned int dimensionPositive;
  ASSIGN_VALUE_OR_RETURN_ERR(dimensionPositive,
                             getPositiveIndex(dimension, input.dims().size()));

  std::vector<int64_t> *signedSizes;
  ASSIGN_VALUE_OR_RETURN_ERR(
      signedSizes, iValToIntList(getGlowIValueForValue(ptNode->input(1))));
  std::vector<uint64_t> sizes;
  for (uint64_t size : *signedSizes) {
    sizes.push_back(size);
  }
  std::vector<glow::NodeValue> chunks;
  ASSIGN_VALUE_OR_RETURN_ERR(chunks,
                             loadSplitImpl(input, dimensionPositive, sizes));
  size_t numChunks = chunks.size();
  glow::GlowIValue output;
  output.fromNodeValueList(chunks);
  RETURN_IF_ERR(addValueMapping(ptNode->output(0), std::move(output)));

  at::ScalarType inputCorrectType;
  ASSIGN_VALUE_OR_RETURN_ERR(inputCorrectType,
                             getCorrectTypeMapping(ptNode->input(0)));
  std::vector<at::ScalarType> outputCorrectTypes(numChunks, inputCorrectType);
  RETURN_IF_ERR(setCorrectTypesMapping(ptNode->output(0), outputCorrectTypes));

  return Error::success();
}

Expected<std::vector<glow::NodeValue>>
PyTorchModelLoader::loadSplitImpl(const glow::NodeValue &glowInput,
                                  const uint64_t dimension,
                                  const uint64_t size) {
  uint64_t chunkCount = glowInput.dims()[dimension] / size;
  std::vector<uint64_t> sizes(chunkCount, size);
  if (chunkCount * size < glowInput.dims()[dimension]) {
    sizes.push_back(glowInput.dims()[dimension] - chunkCount * size);
  }
  return loadSplitImpl(glowInput, dimension, sizes);
}

Expected<std::vector<glow::NodeValue>>
PyTorchModelLoader::loadSplitImpl(const glow::NodeValue &glowInput,
                                  const uint64_t dimension,
                                  const std::vector<uint64_t> &sizes) {
  uint64_t currentStartIndex = 0;
  uint64_t chunkNumber = 0;
  std::vector<glow::NodeValue> chunks;
  std::vector<glow::dim_t> startIndices(glowInput.dims().size(), 0);
  std::vector<glow::dim_t> endIndices = glowInput.dims();
  RETURN_ERR_IF_NOT(!startIndices.empty(), "Input tensor is empty!");
  for (auto chunkSize : sizes) {
    startIndices[dimension] = currentStartIndex;
    endIndices[dimension] = currentStartIndex + chunkSize;
    RETURN_ERR_IF_NOT(
        endIndices[dimension] <= glowInput.dims()[dimension],
        strFormat("Given split sections sum to %lu which is greater than the "
                  "length of the tensor (%lu) in the given dimension (%lu)!",
                  endIndices[dimension], glowInput.dims()[dimension],
                  dimension));
    chunks.push_back(F_.createSlice(strFormat("split_%lu", chunkNumber),
                                    glowInput, startIndices, endIndices)
                         ->getResult());
    currentStartIndex += chunkSize;
    chunkNumber++;
  }
  return chunks;
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

  // wrap if dim is negative
  if (dim < 0) {
    dim += input.dims().size();
  }

  at::ScalarType inputCorrectType;
  ASSIGN_VALUE_OR_RETURN_ERR(inputCorrectType,
                             getCorrectTypeMapping(ptNode->input(0)));

  std::vector<glow::SliceNode *> splitOutputs;
  F_.createSplit("EquallySplit", input, num_split, dim, {}, splitOutputs);
  for (size_t i = 0; i < splitOutputs.size(); ++i) {
    RETURN_IF_ERR(addValueMapping(outputs[i], splitOutputs[i]->getResult()));
    RETURN_IF_ERR(setCorrectTypeMapping(outputs[i], inputCorrectType));
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

Error PyTorchModelLoader::loadGatherElements(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();

  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, -4, outputs, 1));
  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(
      input, getGlowNodeValueForValue(inputs[GatherElementsInputs::input]));
  int64_t dim;
  ASSIGN_VALUE_OR_RETURN_ERR(
      dim, iValToInt(getGlowIValueForValue(inputs[GatherElementsInputs::dim])));
  glow::NodeValue indices;
  ASSIGN_VALUE_OR_RETURN_ERR(
      indices, getGlowNodeValueForValue(inputs[GatherElementsInputs::indices]));
  bool sparse_grad;
  ASSIGN_VALUE_OR_RETURN_ERR(sparse_grad,
                             iValToBool(getGlowIValueForValue(inputs[3])));
  RETURN_ERR_IF_NOT(!sparse_grad, "Currently only supports sparse_grad=false");

  ASSIGN_VALUE_OR_RETURN_ERR(dim, getPositiveIndex(dim, input.dims().size()));
  auto *g = F_.createGatherElements("GatherElements", input, indices, dim);

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
          ->createConstant(ElemKind::Int32ITy, {rois.dims()[0]},
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

  RETURN_IF_ERR(setCorrectTypeMappingSameAs(outputs[0], inputs[0]));
  RETURN_IF_ERR(setCorrectTypeMappingSameAs(outputs[1], inputs[0]));
  return Error::success();
}

Error PyTorchModelLoader::loadExpandDims(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 2, outputs, 1));

  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(
      input, getGlowNodeValueForValue(inputs[ExpandDimsInputs::input]));

  std::vector<int64_t> *dimValues;
  ASSIGN_VALUE_OR_RETURN_ERR(dimValues, iValToIntList(getGlowIValueForValue(
                                            inputs[ExpandDimsInputs::dims])));
  std::vector<dim_t> dims;
  for (dim_t dim : *dimValues) {
    dims.push_back(dim);
  }
  auto *g = F_.createExpandDims("ExpandDims", input, dims);

  RETURN_IF_ERR(addValueMapping(outputs[0], g->getResult()));

  RETURN_IF_ERR(setCorrectTypeMappingSameAs(outputs[0], inputs[0]));

  return Error::success();
}

Error PyTorchModelLoader::loadNarrow(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 4, outputs, 1));

  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(
      input, getGlowNodeValueForValue(inputs[NarrowInputs::input]));
  int64_t dim;
  ASSIGN_VALUE_OR_RETURN_ERR(
      dim, iValToInt(getGlowIValueForValue(inputs[NarrowInputs::dim])));
  int64_t start;
  ASSIGN_VALUE_OR_RETURN_ERR(
      start, iValToInt(getGlowIValueForValue(inputs[NarrowInputs::start])));
  int64_t length;
  ASSIGN_VALUE_OR_RETURN_ERR(
      length, iValToInt(getGlowIValueForValue(inputs[NarrowInputs::length])));

  std::vector<dim_t> starts;
  std::vector<dim_t> ends;
  for (auto d : input.dims()) {
    starts.push_back(0);
    ends.push_back(d); // ends are exclusive
  }
  RETURN_ERR_IF_NOT(
      starts.size() > dim,
      strFormat("Expected input node to have at least %lu dims", dim + 1));
  starts[dim] = start;

  RETURN_ERR_IF_NOT(
      ends.size() > dim,
      strFormat("Expected input node to have at least %lu dims", dim + 1));
  ends[dim] = start + length;

  auto *g = F_.createSlice("Narrow", input, starts, ends);

  RETURN_IF_ERR(addValueMapping(outputs[0], g->getResult()));

  RETURN_IF_ERR(setCorrectTypeMappingSameAs(outputs[0], inputs[0]));

  return Error::success();
}

Error PyTorchModelLoader::loadPixelShuffle(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 2, outputs, 1));

  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(
      input, getGlowNodeValueForValue(inputs[PixelShuffleInputs::input]));

  RETURN_ERR_IF_NOT(input.dims().size() > 3,
                    "Input must have at least 3 dimensions");

  dim_t upscaleFactor;
  ASSIGN_VALUE_OR_RETURN_ERR(upscaleFactor,
                             iValToInt(getGlowIValueForValue(
                                 inputs[PixelShuffleInputs::upscale_factor])));

  RETURN_ERR_IF_NOT(upscaleFactor > 0, "upscale_factor must be > 0");

  const auto NUM_NON_BATCH_DIMS = 3;
  const auto sizesBatchEnd = input.dims().end() - NUM_NON_BATCH_DIMS;

  std::vector<dim_t> shape = input.dims().vec();

  dim_t iC = shape[shape.size() - 3];
  dim_t iH = shape[shape.size() - 2];
  dim_t iW = shape[shape.size() - 1];

  dim_t upscaleFactorSquared = upscaleFactor * upscaleFactor;

  RETURN_ERR_IF_NOT(
      iC % upscaleFactorSquared == 0,
      "Channel dimension must be divisible by the square of upscale_factor");

  dim_t oC = iC / upscaleFactorSquared;
  dim_t oH = iH * upscaleFactor;
  dim_t oW = iW * upscaleFactor;

  // First, reshape to split the channels dim from c into 3 separate dims:
  // (oC, upscaleFactor, upscaleFactor). This allows shuffling to be done next
  // by permuting dims.
  std::vector<dim_t> addedDimsShape(input.dims().begin(), sizesBatchEnd);
  addedDimsShape.insert(addedDimsShape.end(),
                        {oC, upscaleFactor, upscaleFactor, iH, iW});

  auto *inputReshaped = F_.createReshape("reshape", input, addedDimsShape);

  // Next, shuffle by permuting the new upscaleFactor dims alongside the
  // height and width dims.
  std::vector<dim_t> permutation(input.dims().begin(), sizesBatchEnd);
  const auto idx = permutation.size();
  std::iota(permutation.begin(), permutation.end(), 0);
  permutation.insert(permutation.end(), {
                                            idx,     /* oC */
                                            idx + 3, /* iH */
                                            idx + 1, /* upscaleFactor */
                                            idx + 4, /* iW */
                                            idx + 2  /* upscaleFactor */
                                        });

  const auto inputPermuted = F_.createTranspose(
      "reshapeInput", inputReshaped, castVector<glow::unsigned_t>(permutation));

  // Finally, upscale by collapsing (iH, upscaleFactor) -> a single dim (oH)
  // and (iW, upscaleFactor) -> a single dim (oW).
  std::vector<dim_t> finalShape(input.dims().begin(), sizesBatchEnd);
  finalShape.insert(finalShape.end(), {oC, oH, oW});

  auto output =
      F_.createReshape("reshape", inputPermuted, finalShape)->getResult();

  RETURN_ERR(addValueMapping(outputs[0], output));
}

Error PyTorchModelLoader::loadPixelUnshuffle(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 2, outputs, 1));

  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(
      input, getGlowNodeValueForValue(inputs[PixelUnshuffleInputs::input]));

  RETURN_ERR_IF_NOT(input.dims().size() > 3,
                    "Input must have at least 3 dimensions");

  dim_t downscaleFactor;
  ASSIGN_VALUE_OR_RETURN_ERR(
      downscaleFactor, iValToInt(getGlowIValueForValue(
                           inputs[PixelUnshuffleInputs::downscale_factor])));

  RETURN_ERR_IF_NOT(downscaleFactor > 0, "downscale_factor must be > 0");

  const auto NUM_NON_BATCH_DIMS = 3;
  const auto sizesBatchEnd = input.dims().end() - NUM_NON_BATCH_DIMS;

  std::vector<dim_t> shape = input.dims().vec();

  dim_t iC = shape[shape.size() - 3];
  dim_t iH = shape[shape.size() - 2];
  dim_t iW = shape[shape.size() - 1];

  RETURN_ERR_IF_NOT(iH % downscaleFactor == 0,
                    "height must be evenly divisible by downscale_factor");

  RETURN_ERR_IF_NOT(iW % downscaleFactor == 0,
                    "width must be evenly divisible by downscale_factor");

  dim_t downscaleFactorSquared = downscaleFactor * downscaleFactor;

  dim_t oC = iC * downscaleFactorSquared;
  dim_t oH = iH / downscaleFactor;
  dim_t oW = iW / downscaleFactor;

  // First, reshape to split height dim into (oH, downscaleFactor) dims and
  // width dim into (oW, downscaleFactor) dims. This allows unshuffling to be
  // done next by permuting dims.
  std::vector<dim_t> addedDimsShape(input.dims().begin(), sizesBatchEnd);
  addedDimsShape.insert(addedDimsShape.end(),
                        {iC, oH, downscaleFactor, oW, downscaleFactor});

  auto *inputReshaped = F_.createReshape("reshape", input, addedDimsShape);

  // unshuffle by permuting the downscaleFactor dims alongside the channel
  // dim.
  std::vector<dim_t> permutation(input.dims().begin(), sizesBatchEnd);
  const auto idx = permutation.size();
  std::iota(permutation.begin(), permutation.end(), 0);
  permutation.insert(permutation.end(), {
                                            idx,     /* iC */
                                            idx + 2, /* downscaleFactor */
                                            idx + 4, /* downscaleFactor */
                                            idx + 1, /* oH */
                                            idx + 3  /* oW */
                                        });

  const auto inputPermuted = F_.createTranspose(
      "reshapeInput", inputReshaped, castVector<glow::unsigned_t>(permutation));

  // Finally, downscale by collapsing (iC, downscaleFactor, downscaleFactor)
  // -> a single dim (oC), resulting in height=oH and width=oW.
  std::vector<dim_t> finalShape(input.dims().begin(), sizesBatchEnd);
  finalShape.insert(finalShape.end(), {oC, oH, oW});

  auto output =
      F_.createReshape("reshape", inputPermuted, finalShape)->getResult();

  RETURN_ERR(addValueMapping(outputs[0], output));
}

Error PyTorchModelLoader::loadSquare(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 1, outputs, 1));

  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(input, getGlowNodeValueForValue(inputs[0]));

  auto node = F_.createMul("square", input, input);

  RETURN_IF_ERR(addValueMapping(outputs[0], node->getResult()));

  return Error::success();
}

Error PyTorchModelLoader::loadScaleGradient(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 2, outputs, 1));

  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(input, getGlowNodeValueForValue(inputs[0]));

  // Currently PyTorch importer only supports inference,
  // therefore return the input as is.
  RETURN_ERR(addValueMapping(outputs[0], input));
}

Error PyTorchModelLoader::loadErf(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 1, outputs, 1));

  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(input, getGlowNodeValueForValue(inputs[0]));

  auto node = F_.createErf("erf", input)->getResult();

  RETURN_IF_ERR(addValueMapping(outputs[0], node));

  return Error::success();
}
Error PyTorchModelLoader::loadSign(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 1, outputs, 1));

  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(input, getGlowNodeValueForValue(inputs[0]));

  auto zeroes = F_.createSplat("zeroes", input.getType(), 0.f);

  auto isPos = F_.createCmpLT("isPos", zeroes, input);
  auto isNeg = F_.createCmpLT("isNeg", input, zeroes);

  auto posOnes = F_.createSplat("posOnes", input.getType(), 1);
  auto negOnes = F_.createSplat("negOnes", input.getType(), -1);

  auto fillPositive = F_.createSelect("fillPos", isPos, posOnes, zeroes);
  auto fillNegative = F_.createSelect("fillNeg", isNeg, negOnes, fillPositive);

  RETURN_IF_ERR(addValueMapping(outputs[0], fillNegative->getResult()));

  return Error::success();
}

Error PyTorchModelLoader::loadSoftPlus(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 3, outputs, 1));

  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(input, getGlowNodeValueForValue(inputs[0]));
  GlowIValue *betaIValue;
  ASSIGN_VALUE_OR_RETURN_ERR(betaIValue, getGlowIValueForValue(inputs[1]));
  float beta;
  if (betaIValue->isInt()) {
    ASSIGN_VALUE_OR_RETURN_ERR(beta, iValToInt(betaIValue));
  } else {
    ASSIGN_VALUE_OR_RETURN_ERR(beta, iValToDouble(betaIValue));
  }
  GlowIValue *thresholdIValue;
  ASSIGN_VALUE_OR_RETURN_ERR(thresholdIValue, getGlowIValueForValue(inputs[2]));
  float threshold;
  if (thresholdIValue->isInt()) {
    ASSIGN_VALUE_OR_RETURN_ERR(threshold, iValToInt(thresholdIValue));
  } else {
    ASSIGN_VALUE_OR_RETURN_ERR(threshold, iValToDouble(thresholdIValue));
  }

  glow::NodeValue softplus;
  glow::NodeValue linear;
  if (beta == 1) {
    softplus = F_.createSoftPlus("softplus", input)->getResult();
    linear = input;
  } else {
    auto betaType =
        F_.getParent()->uniqueType(ElemKind::FloatTy, {input.dims()});
    auto betas = F_.createSplat("betas", betaType, beta);
    linear = F_.createMul("mult", input, betas)->getResult();
    auto exp = F_.createExp("exp", linear);
    auto ones = F_.createSplat("ones", input.getType(), 1);
    auto sum = F_.createAdd("sum", exp, ones);
    auto log = F_.createLog("log", sum);
    softplus = F_.createDiv("div", log, betas)->getResult();
  }
  auto thresholds = F_.createSplat("thresholds", input.getType(), threshold);
  auto overThreshold = F_.createCmpLT("overThreshold", thresholds, linear);
  auto linearOverThreshold =
      F_.createSelect("linearOverThreshold", overThreshold, linear, softplus);
  RETURN_ERR(addValueMapping(outputs[0], linearOverThreshold->getResult()));
}

Error PyTorchModelLoader::loadBatchedUnaryEmbeddingsBags(
    const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, 4, outputs, 1));

  glow::NodeValue weights;
  ASSIGN_VALUE_OR_RETURN_ERR(
      weights, getGlowNodeValueForValue(
                   inputs[BatchedUnaryEmbeddingsBagsInputs::weights]));
  glow::NodeValue tableOffsets;
  ASSIGN_VALUE_OR_RETURN_ERR(
      tableOffsets,
      getGlowNodeValueForValue(
          inputs[BatchedUnaryEmbeddingsBagsInputs::tableOffsets]));
  glow::NodeValue indices;
  ASSIGN_VALUE_OR_RETURN_ERR(
      indices, getGlowNodeValueForValue(
                   inputs[BatchedUnaryEmbeddingsBagsInputs::indices]));
  glow::NodeValue offsets;
  ASSIGN_VALUE_OR_RETURN_ERR(
      offsets, getGlowNodeValueForValue(
                   inputs[BatchedUnaryEmbeddingsBagsInputs::offsets]));

  auto *EB = F_.createBatchedUnaryEmbeddingsBags(
      "BatchedUnaryEmbeddingsBags", weights, tableOffsets, indices, offsets);

  RETURN_ERR(addValueMapping(outputs[0], EB->getResult()));
}

Error PyTorchModelLoader::loadIntNBitSplitEmbeddingBags(
    const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, -15, outputs, 1));

  glow::NodeValue devWeights;
  ASSIGN_VALUE_OR_RETURN_ERR(
      devWeights, getGlowNodeValueForValue(
                      inputs[IntNBitSplitEmbeddingBagsInputs::dev_weights]));
  glow::NodeValue uvmWeights;
  ASSIGN_VALUE_OR_RETURN_ERR(
      uvmWeights, getGlowNodeValueForValue(
                      inputs[IntNBitSplitEmbeddingBagsInputs::uvm_weights]));
  glow::NodeValue weightsPlacements;
  ASSIGN_VALUE_OR_RETURN_ERR(
      weightsPlacements,
      getGlowNodeValueForValue(
          inputs[IntNBitSplitEmbeddingBagsInputs::weights_placements]));
  glow::NodeValue weightsOffsets;
  ASSIGN_VALUE_OR_RETURN_ERR(
      weightsOffsets,
      getGlowNodeValueForValue(
          inputs[IntNBitSplitEmbeddingBagsInputs::weights_offsets]));
  glow::NodeValue weightsTys;
  ASSIGN_VALUE_OR_RETURN_ERR(
      weightsTys, getGlowNodeValueForValue(
                      inputs[IntNBitSplitEmbeddingBagsInputs::weights_tys]));
  glow::NodeValue dimOffsets;
  ASSIGN_VALUE_OR_RETURN_ERR(
      dimOffsets, getGlowNodeValueForValue(
                      inputs[IntNBitSplitEmbeddingBagsInputs::dimOffsets]));
  int64_t totalDims;
  ASSIGN_VALUE_OR_RETURN_ERR(
      totalDims, iValToInt(getGlowIValueForValue(
                     inputs[IntNBitSplitEmbeddingBagsInputs::totalDims])));
  glow::NodeValue indices;
  ASSIGN_VALUE_OR_RETURN_ERR(
      indices, getGlowNodeValueForValue(
                   inputs[IntNBitSplitEmbeddingBagsInputs::indices]));
  glow::NodeValue offsets;
  ASSIGN_VALUE_OR_RETURN_ERR(
      offsets, getGlowNodeValueForValue(
                   inputs[IntNBitSplitEmbeddingBagsInputs::offsets]));
  int64_t poolingMode;
  ASSIGN_VALUE_OR_RETURN_ERR(
      poolingMode, iValToInt(getGlowIValueForValue(
                       inputs[IntNBitSplitEmbeddingBagsInputs::pooling_mode])));
  if (poolingMode < 0 ||
      poolingMode >=
          static_cast<int64_t>(SplitEmbeddingPoolingMode::EP_TOTAL)) {
    return MAKE_ERR(
        "Invalid pooling mode when loading IntNBitSplitEmbeddingBags");
  }
  glow::NodeValue indiceWeights;
  if (hasGlowNodeValueForValue(
          inputs[IntNBitSplitEmbeddingBagsInputs::indice_weights])) {
    auto indiceWeightsValue = getGlowNodeValueForValue(
        inputs[IntNBitSplitEmbeddingBagsInputs::indice_weights]);
    indiceWeights = std::move(indiceWeightsValue.get());
  }
  int64_t outputDType;
  ASSIGN_VALUE_OR_RETURN_ERR(
      outputDType, iValToInt(getGlowIValueForValue(
                       inputs[IntNBitSplitEmbeddingBagsInputs::output_dtype])));
  if (outputDType < 0 ||
      outputDType >=
          static_cast<int64_t>(SplitEmbeddingSparseType::EST_TOTAL)) {
    return MAKE_ERR(
        "Invalid output data type when loading IntNBitSplitEmbeddingBags");
  }
  if (!indiceWeights) {
    auto *EB = F_.createIntNBitSplitEmbeddingBags(
        "IntNBitSplitEmbeddingBags", devWeights, uvmWeights, weightsPlacements,
        weightsOffsets, weightsTys, dimOffsets, totalDims, indices, offsets,
        static_cast<SplitEmbeddingPoolingMode>(poolingMode),
        static_cast<SplitEmbeddingSparseType>(outputDType));
    RETURN_ERR(addValueMapping(outputs[0], EB->getResult()));
  } else {
    auto *EB = F_.createIntNBitSplitEmbeddingWeightedBags(
        "IntNBitSplitEmbeddingWeightedBags", devWeights, uvmWeights,
        weightsPlacements, weightsOffsets, weightsTys, dimOffsets, totalDims,
        indices, offsets, static_cast<SplitEmbeddingPoolingMode>(poolingMode),
        static_cast<SplitEmbeddingSparseType>(outputDType), indiceWeights);
    RETURN_ERR(addValueMapping(outputs[0], EB->getResult()));
  }
}

Error PyTorchModelLoader::loadIndexAdd(const torch::jit::Node *ptNode) {
  auto inputs = ptNode->inputs();
  auto outputs = ptNode->outputs();
  RETURN_IF_ERR(checkInputAndOutputSizes(inputs, -4, outputs, 1));

  glow::NodeValue input;
  ASSIGN_VALUE_OR_RETURN_ERR(
      input, getGlowNodeValueForValue(inputs[IndexAddInputs::input]));

  int dim = 0;
  ASSIGN_VALUE_OR_RETURN_ERR(
      dim, iValToInt(getGlowIValueForValue(inputs[IndexAddInputs::dim])));
  RETURN_ERR_IF_NOT(dim == 0,
                    "Only adding along dim = 0 is currently supported. ");
  glow::NodeValue index;
  ASSIGN_VALUE_OR_RETURN_ERR(
      index, getGlowNodeValueForValue(inputs[IndexAddInputs::index]));

  glow::NodeValue source;
  ASSIGN_VALUE_OR_RETURN_ERR(
      source, getGlowNodeValueForValue(inputs[IndexAddInputs::source]));

  float alpha = 1.0;
  if (hasGlowIValueForValue(inputs[IndexAddInputs::alpha])) {
    GlowIValue *ival;
    ASSIGN_VALUE_OR_RETURN_ERR(
        ival, getGlowIValueForValue(inputs[IndexAddInputs::alpha]));
    if (ival->isDouble()) {
      ASSIGN_VALUE_OR_RETURN_ERR(alpha, ival->toDouble());
    } else if (ival->isInt()) {
      ASSIGN_VALUE_OR_RETURN_ERR(alpha,
                                 static_cast_expected<float>(ival->toInt()));
    } else if (!ival->isNone()) {
      return MAKE_ERR("Unexpected scalar type");
    }
  }

  RETURN_ERR_IF_NOT(index.dims().size() == 1,
                    "Indices should be 1-dimensional");
  RETURN_ERR_IF_NOT(source.dims().size() > dim,
                    "dim must be less than the number of dims of source");
  RETURN_ERR_IF_NOT(index.dims()[0] == source.dims()[dim],
                    "The dim-th dimension of source must have the same size as "
                    "the length of index");
  for (auto i = 0; i < index.dims().size(); i++) {
    RETURN_ERR_IF_NOT(i == dim || input.dims()[i] == source.dims()[i],
                      "Every dimension of source has to match that of input "
                      "other than dim");
  }

  auto scaleType =
      F_.getParent()->uniqueType(ElemKind::FloatTy, {source.dims()});
  auto scales = F_.createSplat("scales", scaleType, alpha);
  Node *scaledSource = F_.createMul("scaledSource", source, scales);
  auto indices2D = F_.createReshape("indices2D", index, {index.dims()[0], 1});
  auto scatterNode =
      F_.createScatterData("scatter", index, indices2D, scaledSource, true);
  RETURN_ERR(addValueMapping(outputs[0], scatterNode));
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

        at::ScalarType correctType =
            elemKindToScalarType(glowConstant->getElementType());
        NodeValue out = glowConstant->getOutput();
        if (glowConstant->getElementType() == ElemKind::Int64ITy) {
          // For int64 constants, convert them to int32 since many
          // accelerators don't support int64
          LOG(WARNING) << "Loading PyTorch int64 Tensor Attribute as int32 "
                          "because int64 isn't supported";
          out = F_.createConvertTo("int64_to_int32", out, ElemKind::Int32ITy)
                    ->getResult();
          correctType = at::kLong;
        }

        RETURN_IF_ERR(addValueMapping(outputValue, out));
        RETURN_IF_ERR(setCorrectTypeMapping(outputValue, correctType));
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
    std::vector<at::ScalarType> &outputCorrectTypes,
    const PyTorchLoaderSettings &settings,
    const at::ArrayRef<torch::jit::IValue> inputs,
    const InputMetaStack &metaStack) {
  Error error = Error::empty();
  bool loadGraph = false;
  PyTorchModelLoader loader(F, graph, inputPlaceholders, outputPlaceholders,
                            outputCorrectTypes, error, settings, inputs,
                            metaStack, loadGraph);
  return error;
}

PyTorchModelLoader::PyTorchModelLoader(
    glow::Function &F, const torch::jit::Graph &graph,
    std::vector<glow::Placeholder *> &inputPlaceholders,
    std::vector<glow::Placeholder *> &outputPlaceholders,
    std::vector<at::ScalarType> &outputCorrectTypes, Error &error,
    const PyTorchLoaderSettings &settings,
    const at::ArrayRef<torch::jit::IValue> inputs,
    const InputMetaStack &metaStack, bool loadGraph)
    : F_(F), settings_(settings), inputs_(inputs) {
  if (loadGraph) {
    std::cerr << "loading PyTorch graph\n" << graph << std::endl;
  }
  auto loadFn = [&]() -> Error {
    auto graphInputValues = graph.inputs();

    LOG(INFO) << "Using settings: " << settings_.toString();

    if (settings_.dumpFinalGlowGraph || settings_.dumpGlowDag) {
      const std::string fname = "preLoadGlowGraph-" + F.getName().str() + "ir";
      LOG(INFO) << "Dumping pre load graph at " + fname;
      std::ofstream out;
      out.open(fname);
      graph.print(out);
      out.close();
      glow::dumpOperatorStats(graph);
    }

    RETURN_ERR_IF_NOT(
        inputs.size() == graphInputValues.size() ||
            metaStack.inputMetas.size() == graphInputValues.size(),
        glow::strFormat("Number of Graph inputs %lu must match the "
                        "number of provided inputs %lu or inputMeta %lu.",
                        graphInputValues.size(), inputs.size(),
                        metaStack.inputMetas.size()));
    // Create Glow Placeholders for inputs.
    for (size_t i = 0; i < graphInputValues.size(); ++i) {
      const torch::jit::Value *inputValue = graphInputValues[i];
      at::ScalarType inputCorrectType;
      glow::Placeholder *ph;
      if (!metaStack.inputMetas.empty()) {
        glow::Type t;
        if (inputValue->type()->kind() == c10::TypeKind::TensorType) {
          inputCorrectType = metaStack.inputMetas[i].type;
          if (inputCorrectType == at::ScalarType::Undefined) {
            // Handle None input;
            continue;
          }
          // TODO: Change Glow Type to use sdim_t to be consistent
          // with other places.
          std::vector<glow::dim_t> dims;
          for (auto d : metaStack.inputMetas[i].dims) {
            dims.push_back(static_cast<glow::dim_t>(d));
          }

          if (metaStack.inputMetas[i].type == at::kLong) {
            // Load int64 as int32 because many backends don't support int64
            LOG(WARNING) << "Loading input " << i
                         << " as int32 because int64 isn't supported";
            t = glow::Type(ElemKind::Int32ITy, dims);
          } else if (!c10::isQIntType(metaStack.inputMetas[i].type)) {
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
          inputCorrectType = elemKindToScalarType(t.getElementType());
        }
        ph =
            F_.getParent()->createPlaceholder(&t, strFormat("input_%d", int(i)),
                                              /*isTrainable*/ false);
        RETURN_IF_ERR(addValueMapping(inputValue, ph->getOutput()));
        RETURN_IF_ERR(setCorrectTypeMapping(inputValue, inputCorrectType));
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
          } else if (oldType.getElementType() == ElemKind::Int64ITy) {
            auto newType = glow::Type(ElemKind::Int32ITy, oldType.dims());
            LOG(WARNING) << "Loading input " << i
                         << " as int32 because int64 isn't supported";
            ph = F_.getParent()->createPlaceholder(
                &newType, strFormat("input_%d", int(i)),
                /*isTrainable*/ false);
          } else {
            ph = F_.getParent()->createPlaceholder(
                &t->getType(), strFormat("input_%d", int(i)),
                /*isTrainable*/ false);
          }
          inputCorrectType = inputIValue.toTensor().scalar_type();
          RETURN_IF_ERR(addValueMapping(inputValue, ph->getOutput()));
          RETURN_IF_ERR(setCorrectTypeMapping(inputValue, inputCorrectType));
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

      // Check that each output has a valid correct type mapping
      if (auto err = valueMap_.at(output).verifyCorrectTypes()) {
        ADD_MESSAGE_TO_ERR_STACK(
            err, strFormat("Failed on graph output %d", int(i)));
        RETURN_ERR(err);
      }

      at::ScalarType outputScalarType;
      ASSIGN_VALUE_OR_RETURN_ERR(outputScalarType,
                                 getCorrectTypeMapping(output));
      outputCorrectTypes.push_back(outputScalarType);

      glow::NodeValue outputNodeValue;
      // Only allow tensor outputs from Glow subgraph.
      ASSIGN_VALUE_OR_RETURN_ERR(outputNodeValue,
                                 getGlowNodeValueForValue(output));
      auto *save =
          F_.createSave(strFormat("output_%d", int(i)), outputNodeValue);
      outputPlaceholders.push_back(save->getPlaceholder());

      if (settings_.debugContinuouslyVerifyDuringModelLoading) {
        if (!F_.verify()) {
          F_.dumpDAG("failed.dot");
          return MAKE_ERR(
              "Failed Function verification while loading graph outputs.");
        }
      }
    }

    // When randomizing constants in graphs, don't randomize scales/offsets
    // for rowwise/channelwise ops.
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
            {Kinded::Kind::DynamicRowwiseQuantizedFullyConnectedNodeKind,
             {DynamicRowwiseQuantizedFullyConnectedNode::InputIndices::
                  OffsetsIdx,
              DynamicRowwiseQuantizedFullyConnectedNode::InputIndices::
                  ScalesIdx}},
        };

    if (settings_.randomizeConstants) {
      F_.randomizeConstants(randomizeConstantsIgnoreSet);
    }

    if (!F_.verify()) {
      F_.dumpDAG("failed.dot");
      return MAKE_ERR(
          "Failed Function verification after loading JIT graph. Enable the "
          "debugContinuouslyVerifyDuringModelLoading setting and run again "
          "to "
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

Expected<at::ScalarType> ValueMapping::getCorrectType() const {
  RETURN_ERR_IF_NOT(
      correctTypes_.size() == 1,
      strFormat("Expected there to be 1 correct type mapped but found %d",
                int(correctTypes_.size())));
  return correctTypes_[0];
}

std::vector<at::ScalarType> ValueMapping::getCorrectTypes() const {
  return correctTypes_;
}

Error ValueMapping::setCorrectType(at::ScalarType dtype) {
  RETURN_ERR_IF_NOT(correctTypes_.empty(),
                    "Shouldn't be resetting correct type");
  correctTypes_ = {dtype};
  return Error::success();
}

Error ValueMapping::setCorrectTypes(const std::vector<at::ScalarType> &dtypes) {
  RETURN_ERR_IF_NOT(correctTypes_.empty(),
                    "Shouldn't be resetting correct type");
  correctTypes_ = dtypes;
  return Error::success();
}

Error ValueMapping::verifyCorrectTypes() {
  // Verifies that the correctType is valid for the given NodeValue.
  auto verifyCorrectTypeForNodeValue = [](const NodeValue &nodeValue,
                                          at::ScalarType correctType) -> Error {
    auto scalarType = elemKindToScalarType(nodeValue.getElementType());
    // If the correct type matches the actual type then nothing else to check
    if (correctType == scalarType) {
      return Error::success();
    }

    if (correctType == at::kQUInt8) {
      RETURN_ERR_IF_NOT(
          scalarType == at::kQInt8,
          strFormat("Only QInt8 can stand in for QUInt8, found %s",
                    c10::toString(scalarType)));
      return Error::success();
    }

    if (correctType == at::kLong) {
      RETURN_ERR_IF_NOT(scalarType == at::kInt,
                        strFormat("Only Int can stand in for Long, found %s",
                                  c10::toString(scalarType)));
      return Error::success();
    }

    return MAKE_ERR(
        strFormat("Found an unsupported correct type mapping %s has the "
                  "correct type %s but that isn't supported",
                  c10::toString(scalarType), c10::toString(correctType)));
  };

  // Handle single tensor case
  if (mappingType_ == ValueMappingType::NodeValue) {
    RETURN_ERR_IF_NOT(
        correctTypes_.size() == 1,
        strFormat("Expected 1 correct type for NodeValue, found %d",
                  int(correctTypes_.size())));
    RETURN_IF_ERR(
        verifyCorrectTypeForNodeValue(nodeValue_, correctTypes_.at(0)));
    return Error::success();
  }

  DCHECK(glowIValue_ != nullptr);

  if (glowIValue_->isNodeValueList()) {
    // For list of NodeValues, check each NodeValue has a sensible correct
    // type
    std::vector<glow::NodeValue> *nodeValues;
    ASSIGN_VALUE_OR_RETURN_ERR(nodeValues, glowIValue_->toNodeValueList());

    RETURN_ERR_IF_NOT(nodeValues->size() == correctTypes_.size(),
                      strFormat("NodeValue list constains %d elements but have "
                                "correct types for %d elements",
                                int(nodeValues->size()),
                                int(correctTypes_.size())));
    for (size_t i = 0; i < nodeValues->size(); ++i) {
      RETURN_IF_ERR(verifyCorrectTypeForNodeValue(nodeValues->at(i),
                                                  correctTypes_.at(i)));
    }
  } else {
    // For all other GlowIValue types, no correct type should be set
    RETURN_ERR_IF_NOT(
        correctTypes_.empty(),
        strFormat("Found unexpected correct types for GlowIValue of type %s",
                  glowIValue_->getTagString()));
  }

  return Error::success();
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
