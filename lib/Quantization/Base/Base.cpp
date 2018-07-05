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

#include <cmath>

namespace glow {
namespace quantization {

int8_t quantize(float input, const TensorQuantizationParams &TQP) {
  float result = input / TQP.scale_ + TQP.offset_;
  return quantization::clip<int32_t, int8_t>((int32_t)nearbyintf(result));
}

float dequantize(int8_t input, const TensorQuantizationParams &TQP) {
  return TQP.scale_ * (input - TQP.offset_);
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

} // namespace quantization
} // namespace glow
