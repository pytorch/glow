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
#ifndef GLOW_SUPPORT_FLOAT16_H
#define GLOW_SUPPORT_FLOAT16_H

#include "fp16.h"

#include <cstdint>
#include <iostream>

namespace glow {

/// Use a proxy type in case we need to change it in the future.
using Float16Storage = uint16_t;
class float16 {
  Float16Storage data_;

public:
  float16(float data = 0.0) { data_ = fp16_ieee_from_fp32_value(data); }

  /// Arithmetic operators.
  float16 operator*(const float16 &b) const {
    return float16(operator float() * float(b));
  }
  float16 operator/(const float16 &b) const {
    return float16(operator float() / float(b));
  }
  float16 operator+(const float16 &b) const {
    return float16(operator float() + float(b));
  }
  float16 operator-(const float16 &b) const {
    return float16(operator float() - float(b));
  }
  float16 operator+=(const float16 &b) {
    *this = *this + b;
    return *this;
  }
  float16 operator-=(const float16 &b) {
    *this = *this - b;
    return *this;
  }

  /// Comparisons.
  bool operator<(const float16 &b) const { return operator float() < float(b); }
  bool operator>(const float16 &b) const { return operator float() > float(b); }
  bool operator==(const float16 &b) const {
    return operator float() == float(b);
  }
  bool operator>=(const float16 &b) const { return !(operator<(b)); }
  bool operator<=(const float16 &b) const { return !(operator>(b)); }

  /// Cast operators.
  operator double() const { return double(operator float()); }
  operator float() const { return fp16_ieee_to_fp32_value(data_); }
  operator int64_t() const {
    return static_cast<int64_t>(fp16_ieee_to_fp32_value(data_));
  }
  operator int32_t() const {
    return static_cast<int32_t>(fp16_ieee_to_fp32_value(data_));
  }
}; // End class float16.

/// Allow float16_t to be passed to an ostream.
inline std::ostream &operator<<(std::ostream &os, const float16 &b) {
  os << float(b);
  return os;
}

} // End namespace glow.

#endif // GLOW_SUPPORT_FLOAT16_H
