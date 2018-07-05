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

#ifndef GLOW_QUANTIZATION_BASE_H
#define GLOW_QUANTIZATION_BASE_H

#include <cstdlib>
#include <cassert>
#include <limits>
#include <algorithm>

namespace glow {

/// Main attributes of a quantized tensor.
/// Scale and Offset allow quantization of a float tensor and dequantization of
/// integer tensor back to float one.
struct TensorQuantizationParams {
  float scale_;
  int32_t offset_;
};

namespace quantization {

/// Converts floating point value to int8 based on the quantization
/// parameters \p TQP.
int8_t quantize(float input, const TensorQuantizationParams &TQP);

/// Converts int8 quantized value back to floating point number based on
/// the quantization parameters \p TQP.
float dequantize(int8_t input, const TensorQuantizationParams &TQP);

/// \returns the value \p in as clipped to the range of \p DestTy.
template <class SrcTy, class DestTy> DestTy clip(SrcTy in) {
  assert(sizeof(SrcTy) >= sizeof(DestTy) && "Invalid types");

  auto mx = std::numeric_limits<DestTy>::max();
  auto mn = std::numeric_limits<DestTy>::min();
  return std::max<SrcTy>(mn, std::min<SrcTy>(mx, in));
}

} // quantization
} // glow

#endif 
