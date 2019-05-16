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

#include "glow/Quantization/Base/Base.h"
#include "glow/Base/Tensor.h"

#include <cmath>

namespace glow {
namespace quantization {

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
    auto srcHandle = src.getHandle<float16>();
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
    auto destH = dest->getHandle<float16>();
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

Tensor dequantizeTensor(const Tensor &tensor, ElemKind floatKind) {
  assert(((floatKind == ElemKind::FloatTy) ||
          (floatKind == ElemKind::Float16Ty)) &&
         "Non supported output floating point type");
  Tensor tmp(floatKind, tensor.dims());
  auto Ty = tensor.getType().getElementType();
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

TensorQuantizationParams chooseQuantizationParams(float min, float max,
                                                  Schema schema, ElemKind qTy) {
  assert(min <= max && "min must not be bigger than max");

  // Compute the quantized int range.
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
    qmin = std::numeric_limits<int32_t>::min();
    qmax = std::numeric_limits<int32_t>::max();
    break;
  }
  default:
    llvm_unreachable("Quantized type not supported");
  }

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
  if (schema == quantization::Schema::Symmetric) {
    // Check which end saturates the output dynamic range earlier
    // and extend the other end to map the zero-point to quantized 0.
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

  TensorQuantizationParams result{static_cast<float>(scale), nudgedZeroPoint};
  // The only valid offset for symmetric quantization is 0.
  assert((result.offset == 0 || schema != quantization::Schema::Symmetric) &&
         "Symmetric quantization should be centered on 0");

  // The only valid offsets for symmetric quantization with unsigned support are
  // 0 and qmin.
  assert((result.offset == qmin || result.offset == 0 ||
          schema != quantization::Schema::SymmetricWithUnsigned) &&
         "Symmetric quantization with unsigned should be centered on 0 or on "
         "-qmin");
  return result;
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

void tensorFusedRowwiseQuantization(const Tensor &input, Tensor &output) {
  // We are fusing the float scale and int32_t offset onto the end of each
  // row. Thus input and output must both be 2 dimensional, with output having 8
  // extra columns for 4 bytes for float scale, and 4 bytes for int32_t offset.
  assert(input.dims().size() == 2 && output.dims().size() == 2 &&
         "Input and output must be 2 dimensional.");
  assert(input.dims()[1] + 8 == output.dims()[1] &&
         "Output must have 8 more columns than input.");

  const size_t outWidth = output.dims()[1];
  char *dataBasePtr = output.getUnsafePtr();

  auto srcH = input.getHandle<float>();
  auto destH = output.getHandle<uint8_t>();
  for (size_t i = 0, e = input.dims()[0]; i < e; i++) {
    auto slice = srcH.extractSlice(i);
    auto rSrc = slice.getHandle<float>();
    auto res = rSrc.minMaxArg();
    float min = rSrc.raw(res.first);
    float max = rSrc.raw(res.second);

    min = std::min(min, 0.0f);
    max = std::max(max, 0.0f);

    // This matches the Caffe2 implementation for FloatToRowwiseQuantized8BitsOp
    // found in operators/lengths_reducer_rowwise_8bit_ops.h.
    constexpr float kEqualityThreshold = 1e-10f;
    const float scale = ((max - min) < kEqualityThreshold)
                            ? 1.0
                            : ((double)max - (double)min) / 255.0;
    const float offset = min;

    for (size_t j = 0, f = input.dims()[1]; j < f; j++) {
      destH.at({i, j}) = quantization::quantizeWithFloatOffset<uint8_t>(
          srcH.at({i, j}), scale, offset);
    }

    // Now set the scale/offset at the end of each row.
    char *currRowScaleOffsetPtr =
        dataBasePtr + (i + 1) * outWidth - 2 * sizeof(float);
    memcpy(currRowScaleOffsetPtr, &scale, sizeof(float));
    memcpy(currRowScaleOffsetPtr + sizeof(float), &offset, sizeof(float));
  }
}

} // namespace quantization
} // namespace glow
