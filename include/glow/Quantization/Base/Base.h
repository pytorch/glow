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

#ifndef GLOW_QUANTIZATION_BASE_BASE_H
#define GLOW_QUANTIZATION_BASE_BASE_H

#include "glow/Base/Type.h"

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <limits>

namespace glow {

class Tensor;

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
    // The operation x >> y is rounded down to negative infinity. To get to
    // round-nearest we add (1 << (shift - 1)) to the value prior to shifting.
    int rtn = (1 << (post - 1));
    return ((((input >> pre) * scale) + rtn) >> post) + offset;
  }
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
  return quantization::clip<int32_t, DestTy>((int32_t)nearbyintf(result));
}

/// Converts a quantized value (type eTy) to floating point based on the
/// quantization parameters \p TQP.
/// Note: use int64_t to cover the 'symmetric int32 with unsigned' case.
template <class eTy = int8_t>
inline float dequantize(eTy input, const TensorQuantizationParams &TQP) {
  return TQP.scale * ((int64_t)input - TQP.offset);
}

/// Converts a floating point \p tensor to quantized tensor based on the
/// quantization parameters \p TQP and \p Ty.
Tensor quantizeTensor(const Tensor &tensor, const TensorQuantizationParams &TQP,
                      ElemKind Ty = ElemKind::Int8QTy);

/// Converts quantized tensor \p tensor to floating point tensor of type \p Ty
/// floatKind.
Tensor dequantizeTensor(const Tensor &tensor, ElemKind floatKind);

/// Convert the floating point quantization parameters \p scale and \p offset
/// into the integer sequence of:
/// result = ((input >> pre) * scale) >> post + offset.
/// This scales a 32-bit signed integer word into an 8-bit signed integer.
/// \returns transformation parameters.
QuantizationTransform32To8 quantizeScaleOffset32To8(float scale,
                                                    int32_t offset);

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

/// Row-wise quantize the tensor \p input. The param \p input is a 2D
/// tensor (i.e. M * N), \p scales and \p offsets are generated by each row of
/// \p input, \p output is 2D tensor quantized from \p input using \p scales
/// and \p offsets for each row.
void tensorRowwiseQuantization(const Tensor &input, Tensor &output,
                               Tensor &scales, Tensor &offsets);

} // namespace quantization
} // namespace glow

#endif // GLOW_QUANTIZATION_BASE_BASE_H
