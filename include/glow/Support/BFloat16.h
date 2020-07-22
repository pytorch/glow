#ifndef GLOW_SUPPORT_BFLOAT16_H
#define GLOW_SUPPORT_BFLOAT16_H

#include <cmath>
#include <cstdint>
#include <iostream>

namespace glow {

/// Soft bfloat16.
/// This implementation uses single-precision floating point.
class alignas(2) bfloat16 {
  uint16_t storage_;

public:
  static uint16_t float32_to_bfloat16_storage(float rhs) {
    const float &rhs_ref = rhs;
    uint32_t rhsu = *reinterpret_cast<const uint32_t *>(&rhs_ref);
    uint16_t lhs = static_cast<uint16_t>(rhsu >> 16);

    if (std::isnan(rhs) && (lhs & 0x7fu) == 0) {
      lhs = 0x7fc0u; // qNaN
    }

    return lhs;
  }

  static float bfloat16_to_float32(bfloat16 rhs) {
    const uint32_t lhsu = static_cast<uint32_t>(rhs.storage_) << 16;
    const uint32_t &lhsu_ref = lhsu;
    float lhs = *reinterpret_cast<const float *>(&lhsu_ref);
    return lhs;
  }

  static bfloat16 bfloat16_from_uint16_storage(uint16_t rhs) {
    bfloat16 lhs;
    lhs.storage_ = rhs;
    return lhs;
  }

  static int fpclassify(bfloat16 rhs) {
    return std::fpclassify(bfloat16_to_float32(rhs));
  }

  static bool isfinite(bfloat16 rhs) {
    return (rhs.storage_ & 0x7f80u) != 0x7f80u;
  }

  static bool isinf(bfloat16 rhs) {
    return (rhs.storage_ & 0x7fffu) == 0x7f80u;
  }

  static bool isnan(bfloat16 rhs) { return fpclassify(rhs) == FP_NAN; }

  static bool isnormal(bfloat16 rhs) { return fpclassify(rhs) == FP_NORMAL; }

  static bool signbit(bfloat16 rhs) {
    return (rhs.storage_ & 0x8000u) == 0x8000u;
  }

  bfloat16() : storage_{0} {}

  bfloat16(const bfloat16 &rhs) : storage_{rhs.storage_} {}

  bfloat16(float rhs) { storage_ = float32_to_bfloat16_storage(rhs); }

  uint16_t storage() const { return storage_; }

  operator float() const { return bfloat16_to_float32(*this); }

  operator double() const {
    return static_cast<double>(static_cast<float>(*this));
  }

  operator bool() const { return static_cast<bool>(static_cast<float>(*this)); }

  operator int8_t() const {
    return static_cast<int8_t>(static_cast<float>(*this));
  }

  operator int16_t() const {
    return static_cast<int16_t>(static_cast<float>(*this));
  }

  operator int32_t() const {
    return static_cast<int32_t>(static_cast<float>(*this));
  }

  operator int64_t() const {
    return static_cast<int64_t>(static_cast<float>(*this));
  }

  operator uint8_t() const {
    return static_cast<uint8_t>(static_cast<float>(*this));
  }

  operator uint16_t() const {
    return static_cast<uint16_t>(static_cast<float>(*this));
  }

  operator uint32_t() const {
    return static_cast<uint32_t>(static_cast<float>(*this));
  }

  operator uint64_t() const {
    return static_cast<uint64_t>(static_cast<float>(*this));
  }

  bfloat16 operator-() {
    bfloat16 lhs =
        bfloat16_from_uint16_storage(static_cast<uint16_t>(storage_ ^ 0x8000u));
    return lhs;
  }

  bfloat16 &operator+=(const bfloat16 &rhs) {
    storage_ = float32_to_bfloat16_storage(static_cast<float>(*this) +
                                           static_cast<float>(rhs));
    return *this;
  }

  bfloat16 &operator-=(const bfloat16 &rhs) {
    storage_ = float32_to_bfloat16_storage(static_cast<float>(*this) -
                                           static_cast<float>(rhs));
    return *this;
  }

  bfloat16 &operator*=(const bfloat16 &rhs) {
    storage_ = float32_to_bfloat16_storage(static_cast<float>(*this) *
                                           static_cast<float>(rhs));
    return *this;
  }

  bfloat16 &operator/=(const bfloat16 &rhs) {
    storage_ = float32_to_bfloat16_storage(static_cast<float>(*this) /
                                           static_cast<float>(rhs));
    return *this;
  }

  bfloat16 &operator+=(const float &rhs) {
    storage_ = float32_to_bfloat16_storage(static_cast<float>(*this) + rhs);
    return *this;
  }

  bfloat16 &operator-=(const float &rhs) {
    storage_ = float32_to_bfloat16_storage(static_cast<float>(*this) - rhs);
    return *this;
  }

  bfloat16 &operator*=(const float &rhs) {
    storage_ = float32_to_bfloat16_storage(static_cast<float>(*this) * rhs);
    return *this;
  }

  bfloat16 &operator/=(const float &rhs) {
    storage_ = float32_to_bfloat16_storage(static_cast<float>(*this) / rhs);
    return *this;
  }

  friend bfloat16 operator+(bfloat16 lhs, const bfloat16 &rhs) {
    lhs += rhs;
    return lhs;
  }

  friend bfloat16 operator-(bfloat16 lhs, const bfloat16 &rhs) {
    lhs -= rhs;
    return lhs;
  }

  friend bfloat16 operator*(bfloat16 lhs, const bfloat16 &rhs) {
    lhs *= rhs;
    return lhs;
  }

  friend bfloat16 operator/(bfloat16 lhs, const bfloat16 &rhs) {
    lhs /= rhs;
    return lhs;
  }

  friend float operator+(bfloat16 lhs, const float &rhs) {
    return static_cast<float>(lhs) + rhs;
  }

  friend float operator-(bfloat16 lhs, const float &rhs) {
    return static_cast<float>(lhs) - rhs;
  }

  friend float operator*(bfloat16 lhs, const float &rhs) {
    return static_cast<float>(lhs) * rhs;
  }

  friend float operator/(bfloat16 lhs, const float &rhs) {
    return static_cast<float>(lhs) / rhs;
  }

  friend float operator+(float lhs, const bfloat16 &rhs) {
    return lhs + static_cast<float>(rhs);
  }

  friend float operator-(float lhs, const bfloat16 &rhs) {
    return lhs - static_cast<float>(rhs);
  }

  friend float operator*(float lhs, const bfloat16 &rhs) {
    return lhs * static_cast<float>(rhs);
  }

  friend float operator/(float lhs, const bfloat16 &rhs) {
    return lhs / static_cast<float>(rhs);
  }

  friend bool operator<(const bfloat16 &lhs, const bfloat16 &rhs) {
    return static_cast<float>(lhs) < static_cast<float>(rhs);
  }

  friend bool operator>(const bfloat16 &lhs, const bfloat16 &rhs) {
    return rhs < lhs;
  }

  friend bool operator<=(const bfloat16 &lhs, const bfloat16 &rhs) {
    return static_cast<float>(lhs) <= static_cast<float>(rhs);
  }

  friend bool operator>=(const bfloat16 &lhs, const bfloat16 &rhs) {
    return rhs <= lhs;
  }

  friend bool operator==(const bfloat16 &lhs, const bfloat16 &rhs) {
    return static_cast<float>(lhs) == static_cast<float>(rhs);
  }

  friend bool operator!=(const bfloat16 &lhs, const bfloat16 &rhs) {
    return static_cast<float>(lhs) != static_cast<float>(rhs);
  }
};

/// Allow bfloat16_t to be passed to an ostream.
inline std::ostream &operator<<(std::ostream &os, const bfloat16 &b) {
  os << static_cast<float>(b);
  return os;
}

using bfloat16_t = bfloat16;
static_assert(sizeof(bfloat16_t) == 2, "bfloat16_t must be 16 bits wide");

} // namespace glow

#endif // GLOW_SUPPORT_BFLOAT16_H
