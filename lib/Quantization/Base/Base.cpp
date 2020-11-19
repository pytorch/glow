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

#include "glow/Quantization/Base/Base.h"
#include "glow/Base/Tensor.h"
#include "glow/Quantization/Base/Calibration.h"
#include "glow/Quantization/Base/Profile.h"

#include <cmath>

namespace glow {
namespace quantization {

float getTensorAverageValue(const TensorProfilingParams &profParams) {
  size_t numBins = profParams.histogram.size();
  assert(numBins > 0 && "Histogram is empty!");
  float histDelta = (profParams.max - profParams.min) / (float)(numBins);
  float histOff = profParams.min + histDelta / 2.0;
  float histAvg = 0.0;
  float histSum = 0.0;
  for (size_t idx = 0; idx < numBins; ++idx) {
    float histBinCenter = histOff + histDelta * (float)idx;
    float histBinCount = profParams.histogram[idx];
    histAvg += histBinCenter * histBinCount;
    histSum += histBinCount;
  }
  histAvg /= histSum;
  return histAvg;
}

template <class eTy = int8_t>
static void quantizeTensorUtil(Tensor *dest, const Tensor &src) {
  auto destH = dest->getHandle<eTy>();
  TensorQuantizationParams TQP{dest->getType().getScale(),
                               dest->getType().getOffset()};
  switch (src.getElementType()) {
  case ElemKind::FloatTy: {
    auto srcHandle = src.getHandle<float>();
    for (size_t i = 0, e = destH.size(); i < e; ++i) {
      destH.raw(i) = quantization::quantize<eTy>(
          static_cast<float>(srcHandle.raw(i)), TQP);
    }
    break;
  }
  case ElemKind::Float16Ty: {
    auto srcHandle = src.getHandle<float16_t>();
    for (size_t i = 0, e = destH.size(); i < e; ++i) {
      destH.raw(i) = quantization::quantize<eTy>(
          static_cast<float>(srcHandle.raw(i)), TQP);
    }
    break;
  }
  case ElemKind::BFloat16Ty: {
    auto srcHandle = src.getHandle<bfloat16_t>();
    for (size_t i = 0, e = destH.size(); i < e; ++i) {
      destH.raw(i) = quantization::quantize<eTy>(
          static_cast<float>(srcHandle.raw(i)), TQP);
    }
    break;
  }
  default:
    llvm_unreachable("Cannot quantize a type");
  }
}

Tensor quantizeTensor(const Tensor &tensor, const TensorQuantizationParams &TQP,
                      ElemKind Ty) {
  Tensor tmp(Ty, tensor.dims(), TQP.scale, TQP.offset);
  assert(tensor.getType().isFPType() && "Type not supported yet");
  if (Ty == ElemKind::Int8QTy) {
    quantizeTensorUtil<int8_t>(&tmp, tensor);
  } else if (Ty == ElemKind::UInt8QTy) {
    quantizeTensorUtil<uint8_t>(&tmp, tensor);
  } else if (Ty == ElemKind::Int16QTy) {
    quantizeTensorUtil<int16_t>(&tmp, tensor);
  } else if (Ty == ElemKind::Int32QTy) {
    quantizeTensorUtil<int32_t>(&tmp, tensor);
  } else {
    llvm_unreachable("Quantized type not supported");
  }
  return tmp;
}

template <class eTy = int8_t>
static void dequantizeTensorUtil(Tensor *dest, const Tensor &src) {
  TensorQuantizationParams TQP{src.getType().getScale(),
                               src.getType().getOffset()};
  auto srcHandle = src.getHandle<eTy>();
  switch (dest->getElementType()) {
  case ElemKind::FloatTy: {
    auto destH = dest->getHandle<float>();
    for (size_t i = 0, e = destH.size(); i < e; ++i) {
      destH.raw(i) = quantization::dequantize<eTy>(
          static_cast<eTy>(srcHandle.raw(i)), TQP);
    }
    break;
  }
  case ElemKind::Float16Ty: {
    auto destH = dest->getHandle<float16_t>();
    for (size_t i = 0, e = destH.size(); i < e; ++i) {
      destH.raw(i) = quantization::dequantize<eTy>(
          static_cast<eTy>(srcHandle.raw(i)), TQP);
    }
    break;
  }
  case ElemKind::BFloat16Ty: {
    auto destH = dest->getHandle<bfloat16_t>();
    for (size_t i = 0, e = destH.size(); i < e; ++i) {
      destH.raw(i) = quantization::dequantize<eTy>(
          static_cast<eTy>(srcHandle.raw(i)), TQP);
    }
    break;
  }
  default:
    llvm_unreachable("Cannot dequantize to the given type");
  }
}

/// Helper for dequantizing UInt8FusedQTy \p src to \p dest.
template <typename DeqElemTy, typename ScaleOffsetTy>
static void dequantizeFusedRowwiseTensorUtil(Tensor &dest, const Tensor &src) {
  auto dims = dest.dims();
  auto srcH = src.getHandle<uint8_t>();
  auto destH = dest.getHandle<DeqElemTy>();
  for (dim_t i = 0, e = dims[0]; i < e; ++i) {
    float scale, offset;
    std::tie(scale, offset) = srcH.getFusedScaleOffsetFromRow<ScaleOffsetTy>(i);
    for (dim_t j = 0, f = dims[1]; j < f; ++j) {
      destH.at({i, j}) =
          static_cast<DeqElemTy>(quantization::dequantizeWithFloatOffset(
              srcH.at({i, j}), scale, offset));
    }
  }
}

Tensor dequantizeTensor(const Tensor &tensor, ElemKind floatKind) {
  assert(isFloatElemKind(floatKind) &&
         "Non supported output floating point type");
  auto Ty = tensor.getType().getElementType();

  if (Ty == ElemKind::UInt8FusedQTy || Ty == ElemKind::UInt8FusedFP16QTy) {
    const bool scaleOffsetFP16 = Ty == ElemKind::UInt8FusedFP16QTy;
    const dim_t scaleOffsetSize =
        scaleOffsetFP16 ? sizeof(float16_t) : sizeof(float);
    assert(tensor.dims().size() == 2 && "Fused tensors should be 2D");
    assert(tensor.dims()[1] > 2 * scaleOffsetSize &&
           "Expected space for per-row scale/offset");
    Tensor tmp(floatKind, {tensor.dims()[0],
                           tensor.dims()[1] - (dim_t)(2 * scaleOffsetSize)});
    switch (floatKind) {
    case ElemKind::FloatTy:
      if (scaleOffsetFP16) {
        dequantizeFusedRowwiseTensorUtil<float, float16_t>(tmp, tensor);
      } else {
        dequantizeFusedRowwiseTensorUtil<float, float>(tmp, tensor);
      }
      break;
    case ElemKind::Float16Ty:
      if (scaleOffsetFP16) {
        dequantizeFusedRowwiseTensorUtil<float16_t, float16_t>(tmp, tensor);
      } else {
        dequantizeFusedRowwiseTensorUtil<float16_t, float>(tmp, tensor);
      }
      break;
    default:
      llvm_unreachable("Cannot dequantize to the given type");
    }
    return tmp;
  }

  Tensor tmp(floatKind, tensor.dims());
  if (Ty == ElemKind::Int8QTy) {
    dequantizeTensorUtil<int8_t>(&tmp, tensor);
  } else if (Ty == ElemKind::UInt8QTy) {
    dequantizeTensorUtil<uint8_t>(&tmp, tensor);
  } else if (Ty == ElemKind::Int16QTy) {
    dequantizeTensorUtil<int16_t>(&tmp, tensor);
  } else if (Ty == ElemKind::Int32QTy) {
    dequantizeTensorUtil<int32_t>(&tmp, tensor);
  } else {
    llvm_unreachable("Input quantized type not supported");
  }
  return tmp;
}

template <class T = float16_t>
static Tensor tensor4BitsFusedRowwiseDequantizationUtil(const Tensor &input) {
  assert(input.dims().size() == 2 && "Input must be 2 dimensional.");
  // The output tensor should have the same raw as input tensor. Since the
  // quantized tensor is in the following format: | 4bit quantized data |
  // T scale | T offset| The columns of dequantized float data
  // should be (input.dims()[1] - 2*sizeof(T)) * 2.
  Tensor output(
      ElemKind::FloatTy,
      {input.dims()[0], (dim_t)(input.dims()[1] - 2 * sizeof(T)) * 2});
  auto srcH = input.getHandle<uint8_t>();
  auto destH = output.getHandle<float>();
  for (dim_t i = 0; i < input.dims()[0]; i++) {
    T scale, offset;
    std::tie(scale, offset) = srcH.getFusedScaleOffsetFromRow<T>(i);
    for (dim_t j = 0; j < output.dims()[1]; j++) {
      bool isMSB = (j % 2 == 1);
      destH.at({i, j}) = dequantize4BitWithFloatOffset(
          srcH.at({i, j / 2}), static_cast<float>(scale),
          static_cast<float>(offset), isMSB);
    }
  }
  return output;
}

Tensor tensor4BitsFusedRowwiseDequantization(const Tensor &input) {
  auto Ty = input.getType().getElementType();
  assert((Ty == ElemKind::UInt4FusedFP16QTy || Ty == ElemKind::UInt4FusedQTy) &&
         "Unsupported 4bits fused rw quantization type.");
  if (Ty == ElemKind::UInt4FusedFP16QTy) {
    return tensor4BitsFusedRowwiseDequantizationUtil<float16_t>(input);
  }
  return tensor4BitsFusedRowwiseDequantizationUtil<float>(input);
}

QuantizationTransform32To8 quantizeScaleOffset32To8(float scale,
                                                    int32_t offset) {
  // In this function we compute an efficient way to convert signed 32-bit
  // integers into signed 8-bit integers without the use of floating-point
  // multiplication. Instead, we represent the original calculation:
  //
  //    result = (x * scale + offset)
  //
  // as the following sequence of integer calculations:
  //
  //    ((x >> pre_scale  * integer_scale) >> post_scale) + offset
  //
  // This function converts the floating-point scale and offset values to the
  // constants in the integer formula.
  //
  // In this method we assume that any signed 32-bit integer in the input word
  // must be mapped into an 8-bit integer. If the scale factor is 2X, then the
  // number 1000 won't be a legal input because after scaling the result would
  // fall outside of the signed 8-bit range. Any 32-bit number that falls
  // outside of signed the 8-bit output integer will be clipped. This gives us
  // the ability to perform 32-bit arithmetic, as explained below.
  //
  // We can't accurately represent fraction scales (in the range zero to one),
  // because the lowest integer multiplication value is one. For example, the
  // scaling factor 0.25 must be represented as integer multiplication of either
  // zero or one, which would result in an highly inaccurate output.
  // Similarly, rounding the scaling factor of 1.6 to 2.0 would produce
  // inaccurate results because drop a significant part of the number.
  //
  // The solution here is to scale (increase in size) the signed integer scalar,
  // and divide the result by shifting it to the right hand side. For example,
  // the floating-point scalar 0.41 is multiplied by 32x (to 13.12, rounded to
  // 13). Then the signed 32-bit integer input is multiplied by 13, and then
  // shifted 5 times to the right (to shrink the result back). The output of
  // this calculation is (13.0 / 32), which is about ~0.4.
  //
  // This approach works well for some scale values. Notice that the modified
  // integer multiplication requires more bits because the intermediate result
  // is larger. Notice that it's always safe to promote the scalar value from a
  // fraction up to one. When you multiply by the integer value one, the
  // intermediate result does not overflow (does not require more bits).
  //
  // It is actually always safe to perform 16-bit post-multiplication
  // right-shifts. Let's consider two cases. If the value of the floating-point
  // scale is greater than 1.0 then we know that at most 8 of the 32-bits in the
  // input register are used, because the result must fit in 8-bits. The result
  // of 8-bit times 8-bit multiplication is 16-bits, which leaves another 16
  // bits that are unused. We can use these 16-bits to increase the size of the
  // integer scale, and shift the result, as described above, without
  // overflowing the register.
  // The second case is where the scalar value is smaller than 1.0.
  // Multiplication of any number by zero or one does not increase the number of
  // bits which are used by the number.
  //
  // Now, we need to consider another problem. In the previous section we
  // described how we scaled small fractions into a number that's close to one.
  // But scaling to around 1.0 is not accurate enough. Rounding a scale factor
  // like 0.6 to integer would give a very high error rate. Generally, we can't
  // increase the size of the integer multiplier without a limit because this
  // would overflow large values that are close to the upper signed 32-bit
  // limit.
  //
  // To solve the accuracy problem we need to continue to increase the size of
  // the integer scalar without overflowing the signed 32-bit register.
  // The solution here is to perform right-shift on the input, in addition to
  // the output. The idea here is that by performing the post-multiplication
  // right-shift we pick the high bits from the result of the multiplication,
  // and the low bits are ignored. This means that we can continue to increase
  // the size of the integer multiplier and continue to increase the accuracy of
  // the calculation by pre-shifting the 32-bit input. Shifting the input to the
  // right would flip some input bits to zero, but the accuracy loss would be
  // minimal.
  //
  // If the floating point scale factor small then it spans a small part of the
  // 32-bit word. For example, a scale factor of 0.125 (1/8) scales some range
  // into the signed 8-bit result. This range is 8 + 3 bits. This means that we
  // can shift as much as 32-11 bits without overflowing the register. This is
  // a net win because we get to increase the accuracy of the floating point
  // scale factor. For very small scale factors, the used range is very large
  // and can take up the whole 32-bit register, so overflow is a real problem.
  // Here we can use the post-shift value to estimate how many bits will be
  // discarded from the after the multiplication operation and figure out how
  // many bits we can take from the bottom of the input word by shifting it to
  // the right and add more precision to the integer scale multiplier.
  int preShift = 0;
  int postShift = 0;

  // We treat first the particular case when scale is a power of 2 (2 ^ exp,
  // where exp is a signed integer exponent). The operation is specialized as:
  // - for positive 2's exponent:
  //     x * scale + offset (pre = 0, post = 0, scale = (int)scale).
  // - for negative 2's exponent:
  //     x >> post + offset (pre = 0, post = -exp, scale = 1).
  if (isFloatPowerOf2(scale)) {
    int exp = getFloat2Exp(scale);
    if (exp > 0) {
      return QuantizationTransform32To8(0,                       // pre
                                        0,                       // post
                                        static_cast<int>(scale), // scale
                                        offset);                 // offset
    } else {
      return QuantizationTransform32To8(0,       // pre
                                        -exp,    // post
                                        1,       // scale
                                        offset); // offset
    }
  }

  // Calculate the post-shift value. It's always safe to increase scale as long
  // as it's below one, and it's always legal to shift at least 15 bits for
  // small scale values.
  while (scale < 0.5 || (scale < 256 && postShift < 15)) {
    scale *= 2;
    postShift++;
  }

  // Calculate the pre-multiplication shift. Estimate how many bits we can take
  // from the input number and pass to the integer scale.
  while (scale < 255 && preShift < (postShift / 2)) {
    scale *= 2;
    preShift++;
  }

  return QuantizationTransform32To8(preShift, postShift, std::round(scale),
                                    offset);
}

QuantizedRange getQuantizedRange(ElemKind qTy) {
  // Pick int64_t in order to cover the uint32_t range.
  int64_t qmin;
  int64_t qmax;

  switch (qTy) {
  case ElemKind::Int8QTy: {
    qmin = std::numeric_limits<int8_t>::min();
    qmax = std::numeric_limits<int8_t>::max();
    break;
  }
  case ElemKind::UInt8QTy: {
    qmin = std::numeric_limits<uint8_t>::min();
    qmax = std::numeric_limits<uint8_t>::max();
    break;
  }
  case ElemKind::Int16QTy: {
    qmin = std::numeric_limits<int16_t>::min();
    qmax = std::numeric_limits<int16_t>::max();
    break;
  }
  case ElemKind::Int32QTy: {
    // A corner case is when quantizing the bias tensor which is later used in
    // arithmetic computations as (int32)(bias[idx] - biasOffset) (e.g. in the
    // LIBJIT function "libjit_scale_i32i8"). To avoid overflow we must restrict
    // the quantization range such that the subtraction result fits int32. Since
    // both bias[idx] and biasOffset are within the range [qmin, qmax] we will
    // impose: min(int32) <= qmin - qmax and qmax - qmin <= max(int32). In other
    // words we will restrict the quantized dynamic range to int31. Furthermore,
    // since scale is computed as scale = (max - min) / (qmax - qmin) where
    // (qmax - qmin) is large (~2^31) the scale computation has large errors.
    // We will further limit the quantized range to int30 (one extra bit) in
    // order for the computed scale to provide safe quantization within the
    // intended range.
    qmin = std::numeric_limits<int32_t>::min() >> 2;
    qmax = std::numeric_limits<int32_t>::max() >> 2;
    break;
  }
  default:
    llvm_unreachable("Quantized type not supported");
  }
  return QuantizedRange(qmin, qmax);
}

void validateQuantizationParams(TensorQuantizationParams qParams, Schema schema,
                                ElemKind qTy) {

  // Get the quantized range.
  auto minMaxPair = getQuantizedRange(qTy);
  int64_t qmin = minMaxPair.first;
  int64_t qmax = minMaxPair.second;

  // Validate params.
  (void)(qmin);
  (void)(qmax);
  assert((qmin <= qParams.offset) && (qParams.offset <= qmax) &&
         "The offset must be within the quantized range");
  if (schema == quantization::Schema::Symmetric) {
    assert((qParams.offset == 0) &&
           "Symmetric quantization should have offset 0");
  } else if (schema == quantization::Schema::SymmetricWithUnsigned) {
    assert((qParams.offset == qmin || qParams.offset == 0) &&
           "SymmetricWithUnsigned quantization should have offset 0 or qmin");
  } else if (schema == quantization::Schema::SymmetricWithPower2Scale) {
    assert((qParams.offset == 0) &&
           "SymmetricWithPower2Scale quantization should have offset 0");
    assert(isFloatPowerOf2(qParams.scale) &&
           "SymmetricWithPower2Scale quantization parameter should be a power "
           "of 2");
  }
}

TensorQuantizationParams
chooseQuantizationParams(TensorProfilingParams profParams, Schema schema,
                         ElemKind qTy, Calibration calibration) {
  float min = profParams.min;
  float max = profParams.max;
  assert(min <= max && "min must not be bigger than max");

  // Get the quantized range.
  auto minMaxPair = getQuantizedRange(qTy);
  int64_t qmin = minMaxPair.first;
  int64_t qmax = minMaxPair.second;

  // We extend the [min, max] interval to ensure that it contains 0.
  // Otherwise, we would not meet the requirement that 0 be an exactly
  // representable value.
  min = std::min(min, 0.f);
  max = std::max(max, 0.f);

  if (schema == quantization::Schema::SymmetricWithUnsigned) {
    // Check if the range we try to encode is purely positive.
    // If not, we cannot use the Unsigned mapping and we fall back
    // to the symmetric schema.
    if (min >= 0.f) {
      // By construction we always have zero to our range.
      // Since min is >= 0 and 0 is in our range, min is
      // actually zero.
      // Therefore zero is going to be mapped to the first
      // element of the quantized range qmin and thus the
      // offset is going to be qmin.
      assert(min <= std::numeric_limits<float>::epsilon() &&
             "Our range should start at zero");
    } else {
      schema = quantization::Schema::Symmetric;
    }
  }
  if (schema == quantization::Schema::Symmetric ||
      schema == quantization::Schema::SymmetricWithPower2Scale) {
    // Check which end saturates the output dynamic range earlier
    // and extend the other end to map the zero-point to quantized 0.
    assert(qmin < 0 && "Symmetric schema incompatible with unsigned range");
    double rmin = min / (double)qmin;
    double rmax = max / (double)qmax;
    if (rmin > rmax) {
      max = rmin * qmax;
    } else {
      min = rmax * qmin;
    }
  }

  min = std::max(min, std::numeric_limits<float>::lowest());
  max = std::min(max, std::numeric_limits<float>::max());

  // Calibrate the min/max range (for non-zero ranges only).
  if ((profParams.min != profParams.max) && (min != max) &&
      (calibration == Calibration::KLMinimization)) {

    // Rescale the profiled histogram with the new constrained min/max range.
    auto histRescaled = rescaleHistogram(profParams.histogram, profParams.min,
                                         profParams.max, min, max);

    // Number of quantized bins. Default value from TVM / MXNet.
    const size_t numQuantizedBins = 255;

    // Check symmetric schema.
    const bool symmetric = (schema != Asymmetric);

    // Optimize the range.
    FloatRange rangeOpt =
        optimizeKL(histRescaled, min, max, numQuantizedBins, symmetric);

    // Update the min/max range with the optimized range.
    min = rangeOpt.first;
    max = rangeOpt.second;
  }

  // Compute scale.
  double scale = ((double)max - min) / ((double)qmax - qmin);

  // Dequantization uses the following formula scale * (X - offset), so
  // scale should not be equal to zero.
  // If scale is 0, we arbitrary adjust the scale to 0.1.
  if (scale == 0) {
    scale = 0.1;
  }

  assert(scale > 0 && "Scale must be non negative");

  // Zero-point computation.
  // First the initial floating-point computation. The zero-point can be
  // determined from solving an affine equation for any known pair
  // (real value, corresponding quantized value).
  // We know two such pairs: (rmin, qmin) and (rmax, qmax).
  // The arithmetic error on the zero point computed from either pair
  // will be roughly machine_epsilon * (sum of absolute values of terms)
  // so we want to use the variant that adds the smaller terms.
  double zeroPointFromMin = qmin - min / scale;
  double zeroPointFromMax = qmax - max / scale;
  double zeroPointFromMinError = std::abs(qmin) + std::abs(min / scale);
  double zeroPointFromMaxError = std::abs(qmax) + std::abs(max / scale);
  double initialZeroPoint = zeroPointFromMinError < zeroPointFromMaxError
                                ? zeroPointFromMin
                                : zeroPointFromMax;

  // For symmetric quantization, if min == -max, force the zero point to be 0.
  float difference = std::abs(max + min);
  if (difference <= std::numeric_limits<float>::epsilon()) {
    initialZeroPoint = 0;
  }

  // Now we need to nudge the zero point to be an integer (our zero points are
  // integer, and this is motivated by the requirement to be able to represent
  // the real value "0" exactly as a quantized value, which is required in
  // multiple places, for example in Im2col with SAME padding).
  int32_t nudgedZeroPoint = 0;
  if (initialZeroPoint < qmin) {
    nudgedZeroPoint = qmin;
  } else if (initialZeroPoint > qmax) {
    nudgedZeroPoint = qmax;
  } else {
    nudgedZeroPoint = static_cast<int32_t>(round(initialZeroPoint));
  }

  // For SymmetricWithPower2Scale, round scale to nearest higher power of 2.
  if (schema == quantization::Schema::SymmetricWithPower2Scale) {
    scale = std::exp2(std::ceil(std::log2(scale)));
  }

  TensorQuantizationParams result{static_cast<float>(scale), nudgedZeroPoint};
  validateQuantizationParams(result, schema, qTy);
  return result;
}

TensorQuantizationParams
specializeBiasQuantizationParams(const TensorQuantizationParams &biasTQP,
                                 const TensorQuantizationParams &inputTQP,
                                 const TensorQuantizationParams &weightsTQP,
                                 Schema schema, ElemKind biasQTy) {
  // Choose bias offset. For int32 bias we always force offset 0 in order
  // to simplify the implementation since the dynamic range allows it.
  int32_t biasOffset = biasTQP.offset;
  if (biasQTy == ElemKind::Int32QTy) {
    biasOffset = 0;
  }
  // Choose bias scale. We try to force the bias scale value to the product
  // scaleInput * scaleWeights but only if the resulting scale is larger
  // (in order to avoid bias data saturation).
  float scaleInput = inputTQP.scale;
  float scaleWeights = weightsTQP.scale;
  float biasScale = biasTQP.scale;
  if (scaleInput * scaleWeights > biasScale) {
    biasScale = scaleInput * scaleWeights;
  }
  // Validate new bias TQP and return.
  TensorQuantizationParams biasTQPNew = {biasScale, biasOffset};
  validateQuantizationParams(biasTQPNew, schema, biasQTy);
  return biasTQPNew;
}

std::vector<int8_t> createMapping(TypeRef inTy, TypeRef outTy,
                                  std::function<float(float)> f) {
  assert(inTy->getElementType() == outTy->getElementType() &&
         "Input and output type must have same element kind.");
  assert(inTy->isQuantizedType() && "Must pass quantized types.");
  assert(inTy->getElementType() == ElemKind::Int8QTy &&
         "Currently only support int8 for this method.");

  // Calculate the step which will be added to the currInputVal repeatedly in
  // order to cover the input range of the input type.
  auto inputRange = inTy->getQuantizedValueRange();
  const float step = (inputRange.second - inputRange.first) / 255;
  float currInputVal = inputRange.first;

  // Calculate the output int value for each possible input value.
  std::vector<int8_t> mapping(256);
  TensorQuantizationParams outputTQP{outTy->getScale(), outTy->getOffset()};
  for (size_t i = 0; i < 256; i++, currInputVal += step) {
    float currOutputVal = f(currInputVal);
    mapping[i] = quantization::quantize(currOutputVal, outputTQP);
  }

  return mapping;
}

std::vector<TensorQuantizationParams>
getTensorQuantizationParams(const Tensor &tensor, Schema qSchema, ElemKind qTy,
                            dim_t qDim, dim_t qStep) {

  // Validate tensor parameters.
  assert(qDim < tensor.dims().size() &&
         "Quantization dimension  exceeds max tensor dimension!");
  assert(qStep > 0 &&
         "Quantization step (granularity) must be greater than 0!");
  assert((tensor.dims()[qDim] % qStep) == 0 &&
         "Quantization step must divide dimension length!");
  assert(tensor.getElementType() == ElemKind::FloatTy &&
         "Tensor type should be float!");
  dim_t groupNum = tensor.dims()[qDim] / qStep;

  // Get tensor view with max of 6 dimensions.
  auto dimsMax = expandDimsToMax(tensor.dims());
  Tensor tensorMax = tensor.getUnowned(dimsMax);
  auto tensorH = tensorMax.getHandle<float>();

  // Find min/max for each quantization group.
  std::vector<float> minArray(groupNum, std::numeric_limits<float>::max());
  std::vector<float> maxArray(groupNum, std::numeric_limits<float>::lowest());
  assert(dimsMax.size() == 6 &&
         "Invalid number of dimensions for tensor expansion!");
  for (dim_t idx0 = 0; idx0 < dimsMax[0]; idx0++) {
    for (dim_t idx1 = 0; idx1 < dimsMax[1]; idx1++) {
      for (dim_t idx2 = 0; idx2 < dimsMax[2]; idx2++) {
        for (dim_t idx3 = 0; idx3 < dimsMax[3]; idx3++) {
          for (dim_t idx4 = 0; idx4 < dimsMax[4]; idx4++) {
            for (dim_t idx5 = 0; idx5 < dimsMax[5]; idx5++) {

              // Current sample multidimensional index.
              std::array<dim_t, 6> sampleIdx{
                  {idx0, idx1, idx2, idx3, idx4, idx5}};

              // Find quantization group to which this sample belongs.
              dim_t groupIdx = (dim_t)(sampleIdx[qDim] / qStep);

              // Adjust min/max for current group.
              if (tensorH.at(sampleIdx) < minArray[groupIdx]) {
                minArray[groupIdx] = tensorH.at(sampleIdx);
              }
              if (tensorH.at(sampleIdx) > maxArray[groupIdx]) {
                maxArray[groupIdx] = tensorH.at(sampleIdx);
              }
            }
          }
        }
      }
    }
  }

  // Compute the quantization parameters for each group.
  std::vector<TensorQuantizationParams> TQP;
  for (dim_t groupIdx = 0; groupIdx < groupNum; groupIdx++) {
    TQP.push_back(chooseQuantizationParams(
        {minArray[groupIdx], maxArray[groupIdx]}, qSchema, qTy));
  }
  return TQP;
}

void getTensorQuantizationParams(const Tensor &tensor, Tensor &scales,
                                 Tensor &offsets, Schema qSchema, ElemKind qTy,
                                 dim_t qDim, dim_t qStep) {
  auto TQP = getTensorQuantizationParams(tensor, qSchema, qTy, qDim, qStep);
  assert(scales.size() == TQP.size() && "Scales tensor size invalid!");
  assert(offsets.size() == TQP.size() && "Offsets tensor size invalid!");
  auto scalesH = scales.getHandle<float>();
  auto offsetsH = offsets.getHandle<int32_t>();
  for (dim_t idx = 0; idx < TQP.size(); idx++) {
    scalesH.raw(idx) = TQP[idx].scale;
    offsetsH.raw(idx) = TQP[idx].offset;
  }
}

template <class eTy = int8_t>
static void quantizeTensorImpl(Tensor *dest, const Tensor &src,
                               llvm::ArrayRef<TensorQuantizationParams> TQP,
                               dim_t qDim, dim_t qStep) {

  // Validate tensor parameters.
  assert(qDim < src.dims().size() &&
         "Quantization dimension  exceeds max tensor dimension!");
  assert(qStep > 0 &&
         "Quantization step (granularity) must be greater than 0!");
  assert((src.dims()[qDim] % qStep) == 0 &&
         "Quantization step must divide dimension length!");
  assert(src.getElementType() == ElemKind::FloatTy &&
         "Tensor type should be float!");
  assert(TQP.size() == (src.dims()[qDim] / qStep) &&
         "TensorQuantizationParams array size invalid!");

  // Get tensor views with maximum dimensions.
  auto dimsMax = expandDimsToMax(src.dims());
  Tensor srcMax = src.getUnowned(dimsMax);
  auto srcH = srcMax.getHandle<float>();
  Tensor destMax = dest->getUnowned(dimsMax);
  auto destH = destMax.getHandle<eTy>();

  // Perform quantization for each group.
  assert(dimsMax.size() == 6 &&
         "Invalid number of dimensions for tensor expansion!");
  for (dim_t idx0 = 0; idx0 < dimsMax[0]; idx0++) {
    for (dim_t idx1 = 0; idx1 < dimsMax[1]; idx1++) {
      for (dim_t idx2 = 0; idx2 < dimsMax[2]; idx2++) {
        for (dim_t idx3 = 0; idx3 < dimsMax[3]; idx3++) {
          for (dim_t idx4 = 0; idx4 < dimsMax[4]; idx4++) {
            for (dim_t idx5 = 0; idx5 < dimsMax[5]; idx5++) {

              // Current sample multidimensional index.
              std::array<dim_t, 6> sampleIdx{
                  {idx0, idx1, idx2, idx3, idx4, idx5}};

              // Find quantization group to which this sample belongs.
              dim_t groupIdx = sampleIdx[qDim] / qStep;

              // Quantize current sample with group specific quantization
              // parameters.
              destH.at(sampleIdx) = quantization::quantize<eTy>(
                  srcH.at(sampleIdx), TQP[groupIdx]);
            }
          }
        }
      }
    }
  }
}

Tensor quantizeTensor(const Tensor &tensor,
                      llvm::ArrayRef<TensorQuantizationParams> TQP,
                      ElemKind qTy, dim_t qDim, dim_t qStep) {
  Tensor tensorQ(qTy, tensor.dims(), 1.0, 0);
  assert(tensor.getType().isFPType() && "Type not supported yet");
  if (qTy == ElemKind::Int8QTy) {
    quantizeTensorImpl<int8_t>(&tensorQ, tensor, TQP, qDim, qStep);
  } else if (qTy == ElemKind::UInt8QTy) {
    quantizeTensorImpl<uint8_t>(&tensorQ, tensor, TQP, qDim, qStep);
  } else if (qTy == ElemKind::Int16QTy) {
    quantizeTensorImpl<int16_t>(&tensorQ, tensor, TQP, qDim, qStep);
  } else if (qTy == ElemKind::Int32QTy) {
    quantizeTensorImpl<int32_t>(&tensorQ, tensor, TQP, qDim, qStep);
  } else {
    llvm_unreachable("Quantization type not supported");
  }
  return tensorQ;
}

Tensor quantizeTensor(const Tensor &tensor, const Tensor &scales,
                      const Tensor &offsets, ElemKind qTy, dim_t qDim,
                      dim_t qStep) {
  assert(scales.size() == offsets.size() &&
         "Scales/Offsets tensor size invalid!");
  auto scalesH = scales.getHandle<float>();
  auto offsetsH = offsets.getHandle<int32_t>();
  std::vector<TensorQuantizationParams> TQP;
  for (dim_t idx = 0; idx < scales.size(); idx++) {
    TQP.push_back({scalesH.raw(idx), offsetsH.raw(idx)});
  }
  return quantizeTensor(tensor, TQP, qTy, qDim, qStep);
}

bool isFloatPowerOf2(float val) {
  // frexp returns mantissa normalized in [0.5,1) so compare with 0.5.
  int exp;
  return (std::abs(std::frexp(val, &exp)) == 0.5);
}

int getFloat2Exp(float val) { return std::ilogb(val); }

} // namespace quantization
} // namespace glow
