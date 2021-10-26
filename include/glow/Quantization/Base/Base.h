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

#ifndef GLOW_QUANTIZATION_BASE_BASE_H
#define GLOW_QUANTIZATION_BASE_BASE_H

#include "glow/Base/Tensor.h"
#include "glow/Base/Traits.h"
#include "glow/Base/Type.h"

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <limits>

namespace glow {

/// Dummy scale used for representing dummy quantization parameters that have
/// been loaded in place of real quantization parameters.
constexpr float dummyScale = 0.123456813395023345947265625;

/// Profiling parameters of a tensor consisting in the global minimum and global
/// maximum values and also the histogram obtained during profiling. To be noted
/// that the histogram is not normalized.
struct TensorProfilingParams {
  float min;
  float max;
  std::vector<float> histogram;

  TensorProfilingParams() = default;
  TensorProfilingParams(float min, float max) : min(min), max(max) {}
  TensorProfilingParams(float min, float max, const std::vector<float> &hist)
      : min(min), max(max), histogram(hist) {}
  TensorProfilingParams(float min, float max, const Tensor &hist)
      : min(min), max(max) {
    auto histH = hist.getHandle<float>();
    histogram = std::vector<float>(histH.size());
    for (dim_t idx = 0, e = histH.size(); idx < e; idx++) {
      histogram[idx] = histH.raw(idx);
    }
  }
};

/// Main attributes of a quantized tensor.
/// Scale and Offset allow quantization of a float tensor and dequantization of
/// integer tensor back to float one.
struct TensorQuantizationParams {
  float scale;
  int32_t offset;
};

/// A data structure that represents the 32-bit to 8-bit quantization
/// scaling operation. This data structure represents the transformation:
/// (((input >> pre) * scale) + rtn) >> post + offset.
struct QuantizationTransform32To8 {
  int pre;
  int post;
  int scale;
  int offset;

  /// Initializes the transformation based on the conversion formula (above).
  QuantizationTransform32To8(int pre, int post, int scale, int offset)
      : pre(pre), post(post), scale(scale), offset(offset) {}

  /// \returns the scaled integer.
  int32_t transform(int32_t input) {
    // The operation x >> post is rounded down to negative infinity. To get to
    // round-nearest we add (1 << (post - 1)) to the value prior to shifting.
    // Rounding is performed only when shifting right (pos > 0).
    int rtn = (post > 0) ? (1 << (post - 1)) : 0;
    return ((((input >> pre) * scale) + rtn) >> post) + offset;
  }
};

/// Tensor profiling parameters for a given node.
struct NodeProfilingInfo {
  std::string nodeOutputName_;
  TensorProfilingParams tensorProfilingParams_;

  NodeProfilingInfo() = default;
  NodeProfilingInfo(const std::string &nodeOutputName,
                    const TensorProfilingParams &tensorProfilingParams)
      : nodeOutputName_(nodeOutputName),
        tensorProfilingParams_(tensorProfilingParams) {}

  float min() const { return tensorProfilingParams_.min; }
  float max() const { return tensorProfilingParams_.max; }
  const std::vector<float> &histogram() const {
    return tensorProfilingParams_.histogram;
  }
};

/// Tensor quantization parameters for a given node.
struct NodeQuantizationInfo {
  std::string nodeOutputName_;
  TensorQuantizationParams tensorQuantizationParams_;

  NodeQuantizationInfo() = default;
  NodeQuantizationInfo(const std::string &nodeOutputName,
                       const TensorQuantizationParams &tensorQuantizationParams)
      : nodeOutputName_(nodeOutputName),
        tensorQuantizationParams_(tensorQuantizationParams) {}

  float scale() const { return tensorQuantizationParams_.scale; }
  int32_t offset() const { return tensorQuantizationParams_.offset; }
};

/// Primitive to encode an integer in 32-bit unsigned fixed-point format.
class FixedPointUInt32 {
private:
  /// Encoded fixed-point value.
  uint32_t val_;
  /// Number of integer bits.
  unsigned intBits_;
  /// Number of fractional bits.
  unsigned fracBits_;

public:
  /// Default constructor.
  FixedPointUInt32() = default;

  /// Construct a fixed-point representation of the floating-point value
  /// \p floatVal using the fixed-point configuration with minimum approximation
  /// error by using the least amount of integer bits and the highest amount
  /// of fractional bits.
  FixedPointUInt32(float floatVal) {
    assert(floatVal >= 0 && "Floating point value must be positive!");
    val_ = floatingToFixedPoint(floatVal, 32 - minBitsIntegerPart(floatVal));
    intBits_ = minBitsIntegerPart(floatVal);
    fracBits_ = 32 - intBits_;
  };

  /// Construct a fixed-point representation of the floating-point value
  /// \p floatVal using the given number of integer bits \p intBits.
  FixedPointUInt32(float floatVal, unsigned intBits) {
    assert(floatVal >= 0 && "Floating point value must be positive!");
    assert(intBits >= 0 && intBits <= 32 &&
           "Integer bits must be between 0 and 32");
    val_ = floatingToFixedPoint(floatVal, 32 - intBits);
    intBits_ = intBits;
    fracBits_ = 32 - intBits_;
  }

  /// \returns the encoded fixed-point value as integer.
  uint32_t getFixedVal() const { return val_; }

  /// \returns the encoded fixed-point value as float.
  float getFloatVal() const { return (float)(val_) / std::exp2(fracBits_); }

  /// \returns the number of integer bits.
  unsigned getIntBits() const { return intBits_; }

  /// \returns the number of fractional bits.
  unsigned getFracBits() const { return fracBits_; }

private:
  // \p number.
  // \returns the minimum number of bits representing the
  // integer part of the fixed point representation of a
  // floating point number.
  uint32_t minBitsIntegerPart(float number) {
    assert(number >= 0 && "Floating point value must be positive!");
    uint32_t aux = (uint32_t)number;
    uint32_t integerPart = 0;

    while (aux / 2 != 0 || aux % 2 != 0) {
      integerPart += 1;
      aux /= 2;
    }

    assert(integerPart >= 0 && integerPart <= 32 &&
           "Overflow caused by input number\n");
    return integerPart;
  }

  // \p elem.
  // \p fracPart representing number of bits for fixed point representation.
  // \returns the fixed point representation of the input floating point number
  // using the format Q(32- fracPart).fracPart.
  uint32_t floatingToFixedPoint(float elem, uint32_t fracPart) {
    assert(elem >= 0 && "Floating point value must be positive!");
    double result = (double)elem * (double)std::exp2((double)fracPart);
    assert(result >= (double)std::numeric_limits<uint32_t>::min() &&
           result <= (double)std::numeric_limits<uint32_t>::max() &&
           "Float to fix point conversion overflow\n");
    return round(result);
  }

public:
  /// \returns a string representation of the fixed-point value (e.g. "0.13").
  std::string toString() const { return std::to_string(getFloatVal()); }
};

namespace quantization {

/// Type definition for a float min/max range.
using FloatRange = std::pair<float, float>;

/// Type definition for a quantized min/max range.
using QuantizedRange = std::pair<int64_t, int64_t>;

/// Quantization schema which influences the way the quantization parameters
/// scale and offset are computed based on the target min/max dynamic range.
enum Schema {
  /// Asymmetric quantization produces ranges not necessarily centered on 0.
  Asymmetric,
  /// Symmetric quantization produces ranges centered on 0.
  Symmetric,
  /// Symmetric quantization produces ranges centered on 0 or -qmin, qmin being
  /// the minimum value of the quantized type.
  /// An offset of qmin (i.e., offset == -128 for int8) represents an unsigned
  /// version of the quantized type with an offset of zero:
  /// For example, int8 is [-128; 127] - (-128) == uint8 [0; 255] - 0
  SymmetricWithUnsigned,
  /// Quantization schema with:
  /// - range centered on 0 (symmetric): offset == 0.
  /// - scale parameter is a power of 2: scale = 2^E where E is a signed
  ///   exponent. Since the scale parameter is mostly subunitary, the
  ///   exponent is mostly negative.
  /// Since the scale parameter is stored as floating point, the values
  /// of E which are exactly representable range from -126 to 127.
  SymmetricWithPower2Scale,
};

/// Calibration mode which influences the way the dynamic range min/max obtained
/// during profiling is narrowed in order to have a more precise representation
/// for the majority of the values with the price of saturating the outliers.
enum Calibration {
  /// No calibration. The quantization parameters will be computed using the
  /// unaltered dynamic range min/max obtained during profiling such that all
  /// the profiled dynamic range will be representable without saturation.
  None,
  /// Calibration mode based on minimizing the Kullback-Leibler divergence.
  KLMinimization
};

/// Configuration for Profiling, passed into \ref profileQuantization().
struct ProfilingConfiguration {
  /// Number of bins used to compute the histogram during profiling.
  unsigned numHistogramBins{10};
};

/// Configuration for Quantization, passed into \ref quantizeFunction().
struct QuantizationConfiguration {
  /// Profiling infos to use when computing the scale and offset for all the
  /// Nodes inside the function being quantized, including the referenced
  /// Placeholders and Constants.
  std::vector<NodeProfilingInfo> infos{};

  /// The hash of the graph obtained during profiling in the pre lowering stage.
  /// This hash is used to verify during quantization that the graph being
  /// compiled matches the graph used for obtaining the profiling information.
  llvm::hash_code graphPreLowerHash{0};

  /// Whether to check the graph hash during quantization.
  bool checkGraphPreLowerHash{false};

  /// Precision to use when quantizing a Function.
  ElemKind precision{ElemKind::Int8QTy};

  /// Schema to use when quantizing a Function.
  Schema schema{Schema::Asymmetric};

  /// Calibration mode used when computing the quantization parameters.
  Calibration calibration{Calibration::None};

  /// Whether to enable the calibration for constant weights.
  bool calibrateConstants{false};

  /// Whether to use rowwise quantization when quantizing a Function.
  bool enableRowwise{false};

  /// Whether to use channelwise quantization when quantizing a Function.
  bool enableChannelwise{false};

  /// New name for the quantized function. If no name is given then
  /// \ref quantizeFunction() will generate a name.
  std::string newFuncName{""};

  /// If true, the quantizer will abort when encountering a node that it would
  /// like to quantize but the backend cannot support. Note that node kinds in
  /// doNotQuantizeKinds will skip this check and not cause an abort.
  bool assertAllNodesQuantized{false};

  /// Precision used for bias quantization for Convolution and FullyConnected.
  /// This allows specializing the bias quantization. Default is int32.
  ElemKind precisionBias{ElemKind::Int32QTy};

  /// If true, don't apply quantization to FC bias inputs.
  bool skipQuantizeFCBias{false};

  QuantizationConfiguration() = default;
  QuantizationConfiguration(llvm::ArrayRef<NodeProfilingInfo> i) : infos(i) {}
};

/// \returns the tensor average value based on the profiling info \p profParams.
float getTensorAverageValue(const TensorProfilingParams &profParams);

/// \returns the value \p in as clipped to the range of \p DestTy.
template <class SrcTy, class DestTy> DestTy clip(SrcTy in) {
  static_assert(sizeof(SrcTy) >= sizeof(DestTy), "Invalid types");

  auto mx = std::numeric_limits<DestTy>::max();
  auto mn = std::numeric_limits<DestTy>::min();
  return std::max<SrcTy>(mn, std::min<SrcTy>(mx, in));
}

/// Converts floating point value \p input to DestTy (quantized type) based
/// on the quantization parameters \p TQP.
template <class DestTy = int8_t>
inline DestTy quantize(float input, const TensorQuantizationParams &TQP) {
  float result = input / TQP.scale + TQP.offset;
  // Note: use int64_t since casts of large values might be wrapped around
  // before clipping, for example for result = 2147483648.00 (float).
  return quantization::clip<int64_t, DestTy>((int64_t)nearbyintf(result));
}

/// Converts floating point value \p input to \p DestTy (quantized type) based
/// on the quantization parameters \p TQP. The value is returned as int64.
inline int64_t quantize(float input, const TensorQuantizationParams &TQP,
                        ElemKind DestTy) {
  if (DestTy == ElemKind::Int8QTy) {
    return quantize<int8_t>(input, TQP);
  } else if (DestTy == ElemKind::Int16QTy) {
    return quantize<int16_t>(input, TQP);
  } else if (DestTy == ElemKind::Int32QTy) {
    return quantize<int32_t>(input, TQP);
  } else if (DestTy == ElemKind::Int64QTy) {
    return quantize<int64_t>(input, TQP);
  } else {
    llvm_unreachable("Precision not supported!");
  }
}

/// Converts a quantized value (type eTy) to floating point based on the
/// quantization parameters \p TQP.
/// Note: use int64_t to cover the 'symmetric int32 with unsigned' case.
template <class eTy = int8_t>
inline float dequantize(eTy input, const TensorQuantizationParams &TQP) {
  return TQP.scale * ((int64_t)input - TQP.offset);
}

/// Converts floating point value to DestTy (quantized type) based on the
/// quantization parameters \p scale and \p offset. If the destination type is
/// int8_t then an offset of 128 is subtracted to convert to int8_t. If the
/// destination type is int16_t then an offset of 32768 is subtracted to convert
/// to int16_t.
template <class DestTy>
inline DestTy quantizeWithFloatOffset(float input, float scale, float offset) {
  uint16_t d = static_cast<uint16_t>(std::round((input - offset) / scale));
  if (std::is_same<int8_t, DestTy>::value) {
    d -= 128;
  } else if (std::is_same<int16_t, DestTy>::value) {
    d -= 32768;
  }
  return static_cast<DestTy>(d);
}

/// Converts floating point value \p input to 4-bit quantization based on the
/// quantization parameters \p scale and \p offset.
inline uint8_t quantize4BitsWithFloatOffset(float input, float scale,
                                            float offset) {
  uint8_t d = std::max(
      0, std::min(static_cast<int>(std::round((input - offset) / scale)), 15));
  return d;
}

/// Converts a quantized value (type eTy) to floating point based on the
/// quantization parameters \p scale and \p offset. If the input type is int8_t,
/// then an offset of 128 is added to convert to uint8_t. If the input type is
/// int16_t, then an offset of 32768 is added to convert to uint16_t.
template <class eTy>
inline float dequantizeWithFloatOffset(eTy input, float scale, float offset) {
  uint16_t d = static_cast<uint16_t>(input);
  if (std::is_same<int8_t, eTy>::value) {
    d += 128;
  } else if (std::is_same<int16_t, eTy>::value) {
    d += 32768;
  }
  return (d * scale) + offset;
}

/// Converts a 4-bit quantized value, which is stored in \p input (MSB if \p
/// isMSB is true, otherwise LSB), to floating point based on the quantization
/// parameters \p scale and \p offset.
inline float dequantize4BitWithFloatOffset(uint8_t input, float scale,
                                           float offset, bool isMSB) {
  if (isMSB) {
    input >>= 4;
  }
  input &= 0x0f;
  return (input * scale) + offset;
}

/// Converts a floating point \p tensor to quantized tensor based on the
/// quantization parameters \p TQP and \p Ty.
Tensor quantizeTensor(const Tensor &tensor, const TensorQuantizationParams &TQP,
                      ElemKind Ty = ElemKind::Int8QTy);

/// Converts quantized tensor \p tensor to floating point tensor of type \p Ty
/// floatKind.
Tensor dequantizeTensor(const Tensor &tensor, ElemKind floatKind);

/// Dequantize 4-bit fused quantized tensor \p input. \returns the float type
/// output.
Tensor tensor4BitsFusedRowwiseDequantization(const Tensor &input);

/// Convert the floating point quantization parameters \p scale and \p offset
/// into the integer sequence of:
/// result = ((input >> pre) * scale) >> post + offset.
/// This scales a 32-bit signed integer word into an 8-bit signed integer.
/// \returns transformation parameters.
QuantizationTransform32To8 quantizeScaleOffset32To8(float scale,
                                                    int32_t offset);

/// Function to get the quantized range for a given precision type \p qTy.
/// \returns the range as a (min, max) pair.
QuantizedRange getQuantizedRange(ElemKind qTy);

/// Function to validate that the given quantization parameters \p qParams
/// comply with the given quantization \p schema and precision \p qTy.
void validateQuantizationParams(TensorQuantizationParams qParams, Schema schema,
                                ElemKind qTy);

/// Calculate the TensorQuantizationParams from the TensorProfilingParams
/// \p profParams using the quantization type \p qTy and the quantization
/// method described by \p schema. The calibration of the quantization
/// parameters will be done using the method given by \p calibration.
TensorQuantizationParams
chooseQuantizationParams(TensorProfilingParams profParams,
                         Schema schema = Asymmetric,
                         ElemKind qTy = ElemKind::Int8QTy,
                         Calibration calibration = Calibration::None);

/// Function to specialize the TensorQuantizationParams of the bias operand
/// for nodes like Convolution and FullyConnected given the initially computed
/// parameters \p biasTQP and the parameters of the input \p inputTQP and the
/// weights \p weightsTQP, for given quantization schema \p schema and bias type
/// \p biasQTy. The parameter \p biasZero provides the information whether bias
/// data is zero. The bias operand requires a more thoughtful quantization since
/// every bias value has a higher impact on the precision of the output value
/// than any particular weight value. The specialization logic is:
/// - for INT32 bias quantization: since the dynamic range of INT32 is large we
///   can always force symmetric quantization (offset = 0). This allows a faster
///   implementation since no offset subtraction is required at run-time.
/// - for INT8/INT16 bias quantization: since the dynamic range is small we
///   will keep the original offset.
/// - regardless of precision, we try to force the bias scale parameter to
///   bias_scale = input_scale * weights_scale since this has a performance
///   benefit by specializing the parameters to biasPre = 0, biasPost = 0,
///   biasScale = 1. We must verify that by changing the bias scale we don`t
///   saturate the bias data. This is also equivalent to forcing the effective
///   scale applied at run-time (bias_scale / (input_scale * weights_scale))
///   to be always greater than or equal to 1.0 which is a common constraint
///   for the bias for most libraries with quantized implementations.
TensorQuantizationParams
specializeBiasQuantizationParams(const TensorQuantizationParams &biasTQP,
                                 const TensorQuantizationParams &inputTQP,
                                 const TensorQuantizationParams &weightsTQP,
                                 Schema schema, ElemKind biasQTy,
                                 bool biasZero = false);

/// Function similar to \ref specializeBiasQuantizationParams with the main
/// distinction that this function is also allowed to change the quantization
/// parameters of the weights. The modification is done in place. This function
/// is used for per-channel quantization. When the requested bias precision is
/// INT32 this function ensures that bias_scale = input_scale * weights_scale
/// while making sure the bias data is not saturated by changing both the bias
/// and weights quantization parameters.
void specializeBiasWeightsQuantizationParams(
    TensorQuantizationParams &biasTQP, const TensorQuantizationParams &inputTQP,
    TensorQuantizationParams &weightsTQP, Schema schema, ElemKind biasQTy,
    bool biasZero = false);

/// \returns an integer mapping from the \p inTy to the \p outTy given the
/// floating-point function \p func.
/// \pre inTy and outTy must be quantized types.
template <typename T = int8_t>
std::vector<T> createMapping(TypeRef inTy, TypeRef outTy,
                             std::function<float(float)> func) {
  assert(inTy->isQuantizedType() && "Input type must be quantized!");
  assert(outTy->isQuantizedType() && "Output type must be quantized!");
  assert(outTy->isType<T>() && "Output type must match template type!");

  // Calculate the step which will be added to the currInputVal repeatedly in
  // order to cover the input range of the input type.
  auto inputRange = inTy->getQuantizedValueRange();
  const float step = inTy->getQuantizedValueStep();
  float currInputVal = inputRange.first;

  // Calculate the output int value for each possible input value.
  std::vector<T> mapping(inTy->getQuantizedValueCount());
  TensorQuantizationParams outputTQP{outTy->getScale(), outTy->getOffset()};
  for (size_t i = 0; i < mapping.size(); i++, currInputVal += step) {
    float currOutputVal = func(currInputVal);
    mapping[i] = quantization::quantize<T>(currOutputVal, outputTQP);
  }
  return mapping;
}

/// Row-wise quantize the tensor \p input. \p scales and \p offsets are
/// generated by each row of \p input, \p output is tensor of the same shape as
/// input, quantized from \p input using \p scales and \p offsets for each
/// row. Note that the shape of input/output can be any non-zero number of
/// dimensions; row refers to all data in the first dimension of the shape.
/// Template parameter \p ScaleT and OffsetT represent the type to use for the
/// scales and offsets for quantization respectively. Template parameter \p QP
/// represents quantization precision, typically int8_t or uint8_t.
template <typename ScaleT, typename OffsetT, typename QP>
void tensorRowwiseQuantization(const Tensor &input, Tensor &output,
                               Tensor &scales, Tensor &offsets,
                               quantization::Schema schema) {
  constexpr bool offsetIsFP = std::is_same<float, OffsetT>::value ||
                              std::is_same<float16_t, OffsetT>::value;
  constexpr bool offsetIsInt32 = std::is_same<int32_t, OffsetT>::value;
  static_assert((offsetIsInt32 && std::is_same<float, ScaleT>::value) ||
                    (offsetIsFP && std::is_same<ScaleT, OffsetT>::value),
                "Invalid combination of Scale/Offset types.");

  const auto fDims = flattenCdr(input.dims());
  Tensor finalIn = input.getUnowned({fDims.first, fDims.second});
  Tensor finalOut = output.getUnowned({fDims.first, fDims.second});
  ShapeHW idim(finalIn.dims());

  auto srcH = finalIn.getHandle<float>();
  auto destH = finalOut.getHandle<QP>();
  auto scalesH = scales.getHandle<ScaleT>();
  auto offsetsH = offsets.getHandle<OffsetT>();
  for (dim_t i = 0; i < idim.height; i++) {
    auto slice = srcH.extractSlice(i);
    auto rSrc = slice.getHandle<float>();
    auto res = rSrc.minMaxArg();
    float min = rSrc.raw(res.first);
    float max = rSrc.raw(res.second);

    // Handle rowwise quantization for FCs.
    if (offsetIsInt32) {
      TensorQuantizationParams qParams =
          chooseQuantizationParams({min, max}, schema);
      for (dim_t j = 0; j < idim.width; j++) {
        destH.at({i, j}) = quantization::quantize(srcH.at({i, j}), qParams);
      }
      scalesH.raw(i) = qParams.scale;
      offsetsH.raw(i) = qParams.offset;
    } else if (offsetIsFP) {
      // Handle rowwise quantization for Rowwise quantized SLS.
      constexpr float kEqualityThreshold = 1e-10f;
      const float scale = ((max - min) < kEqualityThreshold)
                              ? 1.0
                              : ((double)max - (double)min) / 255.0;
      float offset = min;

      for (dim_t j = 0; j < idim.width; j++) {
        destH.at({i, j}) = quantization::quantizeWithFloatOffset<QP>(
            srcH.at({i, j}), scale, offset);
      }
      scalesH.raw(i) = static_cast<ScaleT>(scale);
      offsetsH.raw(i) = static_cast<OffsetT>(offset);
    } else {
      llvm_unreachable("Unsupported offset type.");
    }
  }
}

/// Fused-rowwise quantize the tensor \p input. Scales and offsets are generated
/// from each row of \p input. This function supports 8-bits quantization (i.e.
/// each quantized data uses 8 bits) and 4-bits quantization(i.e. each quantized
/// data uses 4 bits).
/// For 8-bits quantization, \p output is tensor of the same shape as input but
/// with extra columns for storing fused scales. Template parameter \p T
/// represents the datatype used for storing the scale and offset in the row
/// |   .... int8 data ...    |   scale   |  offset   |
/// |num_of_input_columns * 1B| sizeof(T) | sizeof(T) |
/// For 4-bits quantization, in \p output, 1 byte will contain 2 quantized data.
/// Template parameter \p T here could be either float or float16_t.
/// |   .... int4 data ...       | scale        |   offset      |
/// |num_of_input_columns * 0.5B |  sizeof(T)   |   sizeof(T)   |
/// \pre input.dims().size() == 2
/// \pre output.dims().size() == 2
/// For 8-bits quantization:
/// \pre input.dims()[1] + 2 * sizeof(T) == output.dims()[1]
/// For 4-bits quantization:
/// \pre input.dims()[1] % 2 == 0
/// \pre input.dims()[1] / 2 + 2 * sizeof(T) == output.dims()[1]
template <typename T>
void tensorFusedRowwiseQuantization(const Tensor &input, Tensor &output) {
  // We are fusing the scale and offset onto the end of each row. Thus input and
  // output must both be 2 dimensional, with output having 2*sizeof(T) extra
  // columns for the scale and offset.
  auto outputType = output.getElementType();
  assert(input.dims().size() == 2 && output.dims().size() == 2 &&
         "Input and output must be 2 dimensional.");
  if (outputType == ElemKind::UInt8FusedFP16QTy ||
      outputType == ElemKind::UInt8FusedQTy) {
    assert(input.dims()[1] + 2 * sizeof(T) == output.dims()[1] &&
           "Output must have 2*sizeof(T) more columns than input for 8-bits "
           "quantization.");
  } else if (outputType == ElemKind::UInt4FusedFP16QTy ||
             outputType == ElemKind::UInt4FusedQTy) {
    assert(
        input.dims()[1] % 2 == 0 &&
        "4-bits fused quantization only works for the number of input column "
        "a multiple of 2");
    assert(
        input.dims()[1] / 2 + 2 * sizeof(T) == output.dims()[1] &&
        "Output must have 2*sizeof(T) more columns than half of input columns "
        "for 4-bits quantization.");
  }

  auto srcH = input.getHandle<float>();
  auto destH = output.getHandle<uint8_t>();
  for (dim_t i = 0, e = input.dims()[0]; i < e; i++) {
    auto slice = srcH.extractSlice(i);
    auto rSrc = slice.getHandle<float>();
    auto res = rSrc.minMaxArg();
    float min = rSrc.raw(res.first);
    float max = rSrc.raw(res.second);

    float range;
    switch (outputType) {
    case ElemKind::UInt8FusedQTy:
    case ElemKind::UInt8FusedFP16QTy:
      range = 255.0;
      break;
    case ElemKind::UInt4FusedFP16QTy:
    case ElemKind::UInt4FusedQTy:
      range = 15.0;
      break;
    default:
      llvm_unreachable("Not yet supported");
    }

    // This matches the Caffe2 implementation for FloatToRowwiseQuantized8BitsOp
    // found in operators/lengths_reducer_rowwise_8bit_ops.h.
    constexpr float kEqualityThreshold = 1e-10f;
    const float scale = ((max - min) < kEqualityThreshold)
                            ? 1.0
                            : ((double)max - (double)min) / range;
    const float offset = min;

    for (dim_t j = 0, f = input.dims()[1]; j < f; j++) {
      if (outputType == ElemKind::UInt8FusedFP16QTy ||
          outputType == ElemKind::UInt8FusedQTy) {
        destH.at({i, j}) = quantization::quantizeWithFloatOffset<uint8_t>(
            srcH.at({i, j}), scale, offset);
      } else if (outputType == ElemKind::UInt4FusedFP16QTy ||
                 outputType == ElemKind::UInt4FusedQTy) {
        uint8_t quantized = quantization::quantize4BitsWithFloatOffset(
            srcH.at({i, j}), scale, offset);
        if (j % 2 == 0) {
          // Even columns use LSB 4-bit.
          destH.at({i, j / 2}) = quantized;
        } else {
          // Odd columns use MSB 4-bit.
          destH.at({i, j / 2}) |= quantized << 4;
        }
      } else {
        llvm_unreachable("Not yet supported");
      }
    }

    // Now set the scale/offset at the end of each row.
    destH.setFusedScaleOffsetInRow<T>(i, scale, offset);
  }
}

/// Generic function to compute the quantization parameters for an input
/// floating-point tensor \p tensor with given schema \p qSchema and type
/// \p qTy. A separate set of quantization parameters (scale, offset) will
/// be computed for each group of \p qStep indices along the \p qDim dimension.
/// This allows quantizing a given tensor with finer granularity (e.g. rowwise
/// or channelwise).
/// For example, for a tensor of size [4, 6, 8, 10], qDim = 1 and qStep = 3:
/// -> one set of quantization parameters will be computed for [:,0:2,:,:].
/// -> one set of quantization parameters will be computed for [:,3:5,:,:].
/// The number of sets of computed quantization parameters (scale, offset) is
/// tensor.dims()[qDim] / qStep. \returns the set of quantization parameters.
std::vector<TensorQuantizationParams>
getTensorQuantizationParams(const Tensor &tensor, Schema qSchema = Asymmetric,
                            ElemKind qTy = ElemKind::Int8QTy, dim_t qDim = 0,
                            dim_t qStep = 1);

/// Similar function to the one above with the difference that the quantization
/// parameters scales and offsets are written into separate tensors \p scales
/// and \p offsets which are assummed allocated with the correct type and size.
void getTensorQuantizationParams(const Tensor &tensor, Tensor &scales,
                                 Tensor &offsets, Schema qSchema = Asymmetric,
                                 ElemKind qTy = ElemKind::Int8QTy,
                                 dim_t qDim = 0, dim_t qStep = 1);

/// Generic function to quantize a given input floating-point tensor \p tensor
/// with given tensor quantization parameters \p TQP and type \p qTy. A separate
/// set of quantization parameters (scale, offset) is provided for each group
/// of \p qStep indices along the \p qDim dimension and can be obtained using
/// the function \ref getTensorQuantizationParams. This allows quantizing a
/// given tensor with finer granularity (e.g. rowwise or channelwise).
/// For example, for a tensor of size [4, 6, 8, 10], qDim = 1 and qStep = 3:
/// -> one set of quantization parameters will be provided for [:,0:2,:,:].
/// -> one set of quantization parameters will be provided for [:,3:5,:,:].
/// The number of sets of provided quantization parameters (scale, offset) is
/// tensor.dims()[qDim] / qStep. \returns the quantized tensor.
Tensor quantizeTensor(const Tensor &tensor,
                      llvm::ArrayRef<TensorQuantizationParams> TQP,
                      ElemKind qTy = ElemKind::Int8QTy, dim_t qDim = 0,
                      dim_t qStep = 1);

/// Similar function to the one above with the difference that the quantization
/// parameters scales and offsets are loaded from separate tensors \p scales
/// and \p offsets.
Tensor quantizeTensor(const Tensor &tensor, const Tensor &scales,
                      const Tensor &offsets, ElemKind qTy = ElemKind::Int8QTy,
                      dim_t qDim = 0, dim_t qStep = 1);

/// Verify if float is an exact power of 2 (mantissa is exactly 1.0).
bool isFloatPowerOf2(float val);

/// Get float 2's exponent.
int getFloat2Exp(float val);

} // namespace quantization
} // namespace glow

#endif // GLOW_QUANTIZATION_BASE_BASE_H
