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

/// Tensor quantization parameters for a given node.
struct NodeQuantizationInfo {
  std::string nodeOutputName_;
  TensorQuantizationParams tensorQuantizationParams_;

  NodeQuantizationInfo() = default;
  NodeQuantizationInfo(const std::string &nodeOutputName,
                       const TensorQuantizationParams &tensorQuantizationParams)
      : nodeOutputName_(nodeOutputName),
        tensorQuantizationParams_(tensorQuantizationParams) {}

  float Scale() const { return tensorQuantizationParams_.scale; }
  int32_t Offset() const { return tensorQuantizationParams_.offset; }
};

namespace quantization {

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

/// Configuration for Quantization, passed into \ref quantizeFunction().
struct QuantizationConfiguration {
  /// Infos to use when determining scale and offset for all Nodes inside, and
  /// Placeholders and Constants referenced by, a Function being quantized.
  std::vector<NodeQuantizationInfo> infos{};

  /// Precision to use when quantizing a Function.
  ElemKind precision{ElemKind::Int8QTy};

  /// Schema to use when quantizing a Function.
  quantization::Schema schema{quantization::Schema::Asymmetric};

  /// Whether to use rowwise quantization when quantizing a Function.
  bool enableRowwise{false};

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

  QuantizationConfiguration() = default;
  QuantizationConfiguration(llvm::ArrayRef<NodeQuantizationInfo> i)
      : infos(i) {}
};

/// \returns the value \p in as clipped to the range of \p DestTy.
template <class SrcTy, class DestTy> DestTy clip(SrcTy in) {
  static_assert(sizeof(SrcTy) >= sizeof(DestTy), "Invalid types");

  auto mx = std::numeric_limits<DestTy>::max();
  auto mn = std::numeric_limits<DestTy>::min();
  return std::max<SrcTy>(mn, std::min<SrcTy>(mx, in));
}

/// Converts floating point value to DestTy (quantized type) based on the
/// quantization parameters \p TQP.
template <class DestTy = int8_t>
inline DestTy quantize(float input, const TensorQuantizationParams &TQP) {
  float result = input / TQP.scale + TQP.offset;
  // Note: use int64_t since casts of large values might be wrapped around
  // before clipping, for example for result = 2147483648.00 (float).
  return quantization::clip<int64_t, DestTy>((int64_t)nearbyintf(result));
}

/// Converts a quantized value (type eTy) to floating point based on the
/// quantization parameters \p TQP.
/// Note: use int64_t to cover the 'symmetric int32 with unsigned' case.
template <class eTy = int8_t>
inline float dequantize(eTy input, const TensorQuantizationParams &TQP) {
  return TQP.scale * ((int64_t)input - TQP.offset);
}

/// Converts floating point value to DestTy (quantized type) based on the
/// quantization parameters \p scale and \p offset. If the dest type is int8_t,
/// then an offset of 128 is substracted to convert to int8_t.
template <class DestTy>
inline DestTy quantizeWithFloatOffset(float input, float scale, float offset) {
  uint8_t d = static_cast<uint8_t>(std::round((input - offset) / scale));
  if (std::is_same<int8_t, DestTy>::value) {
    d -= 128;
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
/// then an offset of 128 is added to convert to uint8_t.
template <class eTy>
inline float dequantizeWithFloatOffset(eTy input, float scale, float offset) {
  uint8_t d = static_cast<uint8_t>(input);
  if (std::is_same<int8_t, eTy>::value) {
    d += 128;
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
std::pair<int64_t, int64_t> getQuantizationRange(ElemKind qTy);

/// Function to validate that the given quantization parameters \p qParams
/// comply with the given quantization \p schema and precision \p qTy.
void validateQuantizationParams(TensorQuantizationParams qParams, Schema schema,
                                ElemKind qTy);

/// Calculate TensorQuantizationParams based on the clipped \p min and \p max
/// floating point range and using the base quantization type \p qTy and the
/// quantization method described by \p schema.
TensorQuantizationParams
chooseQuantizationParams(float min, float max, Schema schema = Asymmetric,
                         ElemKind qTy = ElemKind::Int8QTy);

/// \returns an int8 vector mapping from the \p inTy to the \p outTy given the
/// function \p f.
/// \pre inTy and outTy should be Int8QTy.
std::vector<int8_t> createMapping(TypeRef inTy, TypeRef outTy,
                                  std::function<float(float)> f);

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
    // Expand the range to include 0.0f so that 0 is exactly representable.
    min = std::min(min, 0.0f);
    max = std::max(max, 0.0f);

    // Handle rowwise quantization for FCs.
    if (offsetIsInt32) {
      TensorQuantizationParams qParams =
          chooseQuantizationParams(min, max, schema);
      for (dim_t j = 0; j < idim.width; j++) {
        destH.at({i, j}) = quantization::quantize(srcH.at({i, j}), qParams);
      }
      scalesH.raw(i) = qParams.scale;
      offsetsH.raw(i) = qParams.offset;
    } else if (offsetIsFP) {
      // Handle rowwise quantization for Rowwise quantized SLS.
      float scale = ((double)max - (double)min) / 255.0;
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
/// Template parameter \p T here must be float16_t.
/// |   .... int4 data ...       | scale | offset |
/// |num_of_input_columns * 0.5B |  2B   |   2B   |
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
  } else if (outputType == ElemKind::UInt4FusedFP16QTy) {
    constexpr bool scaleIsFP16 = std::is_same<float16_t, T>::value;
    (void)scaleIsFP16;
    assert(scaleIsFP16 && "Only float16_t scale and offset are supported "
                          "in 4-bit fused quantization");
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

    min = std::min(min, 0.0f);
    max = std::max(max, 0.0f);

    float range;
    switch (outputType) {
    case ElemKind::UInt8FusedQTy:
    case ElemKind::UInt8FusedFP16QTy:
      range = 255.0;
      break;
    case ElemKind::UInt4FusedFP16QTy:
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
      } else if (outputType == ElemKind::UInt4FusedFP16QTy) {
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

/// Verify if float is an exact power of 2 (mantissa is exactly 1.0).
bool isFloatPowerOf2(float val);

/// Get float 2's exponent.
int getFloat2Exp(float val);

} // namespace quantization
} // namespace glow

#endif // GLOW_QUANTIZATION_BASE_BASE_H
