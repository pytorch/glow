#ifndef NOETHER_TENSOR_H
#define NOETHER_TENSOR_H

#include <numeric>
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <iostream>
#include <cmath>


namespace noether {

const size_t numRandomVals = 32;
const double randomVals[] = {0.4414916532837977, 0.9746009401393961, 0.7644974627030178, 0.27827514759492433, 0.2823586355251567, 0.2506456727980517, 0.42572714689894375, 0.0336972057297974, 0.243959576298854, 0.18966169453269455, 0.5286137478346916, 0.6222636988212916, 0.2957554911353527, 0.620880111550055, 0.6401712669183511, 0.7998884586830336, 0.5631086934294972, 0.2538295602954759, 0.8832542169407436, 0.48075360869679007, 0.5971563343845181, 0.15586842213087104, 0.11915981553150845, 0.6615953736728364, 0.9890390729567575, 0.12803209414922834, 0.6681364032106246, 0.9624442797641373, 0.9968422482663746, 0.9591749207255753, 0.7215510646287491, 0.9200771645764693};

/// A 3D tensor.
template <class ElemTy> class Array3D final {
  size_t sx_{0}, sy_{0}, sz_{0};
  ElemTy *data_{nullptr};

  /// \returns the offset of the element in the tensor.
  size_t getElementIdx(size_t x, size_t y, size_t z) const {
    assert(isInBounds(x, y, z) && "Out of bounds");
    return (sx_ * y + x) * sz_ + z;
  }

public:
  /// \returns True if the coordinate is within the array.
  bool isInBounds(size_t x, size_t y, size_t z) const {
    return x < sx_ && y < sy_ && z < sz_;
  }

  void clear(ElemTy value = 0) {
      std::fill(&data_[0], &data_[0] + size(), value);
  }

  /// Fill the array with random data that's close to zero.
  void randomize() {
    static int offset = 0;
    double scale = std::sqrt(double(size()));
    for (size_t i = 0, e = size(); i < e; ++i) {
      data_[i] = randomVals[(offset + i) % numRandomVals] / scale;
    }

    offset++;
  }

  /// \returns the dimension of the tensor.
  std::tuple<size_t, size_t, size_t> dims() const {
    return std::make_tuple(sx_, sy_, sz_);
  }

  /// \returns the number of elements in the array.
  size_t size() const { return sx_ * sy_ * sz_; }

  /// Initialize an empty tensor.
  Array3D() = default;

  /// Initialize a new tensor.
  Array3D(size_t x, size_t y, size_t z) : sx_(x), sy_(y), sz_(z) {
    data_ = new ElemTy[size()];
    clear();
  }

  /// Copy ctor.
  Array3D (const Array3D& other) = delete;

  // Move ctor.
  Array3D (Array3D&& other) noexcept {
    data_ = other.data_;
    sx_ = other.sx_;
    sy_ = other.sy_;
    sz_ = other.sz_;
    other.data_ = nullptr;
    other.sx_ = 0;
    other.sy_ = 0;
    other.sz_ = 0;
  }

  Array3D& operator= (const Array3D& other) = delete;

  /// Move assignment operator.
  Array3D& operator= (Array3D&& other) noexcept {
    data_ = other.data_;
    sx_ = other.sx_;
    sy_ = other.sy_;
    sz_ = other.sz_;
    other.data_ = nullptr;
    other.sx_ = 0;
    other.sy_ = 0;
    other.sz_ = 0;
    return *this;
  }

  /// Assigns a new shape to the tensor and allocates a new buffer.
  void reset(std::tuple<size_t, size_t, size_t> dim) {
    size_t x, y, z;
    std::tie(x, y, z) = dim;
    reset(x,y,z);
  }

  /// Assigns a new shape to the tensor and allocates a new buffer.
  void reset(size_t x, size_t y, size_t z) {
    sx_ = x;
    sy_ = y;
    sz_ = z;
    delete[] data_;
    data_ = new ElemTy[size()];
    clear();
  }

  ~Array3D() { delete[] data_; }

  /// Add all of the elements in the array.
  ElemTy sum() {
    return std::accumulate(&data_[0], &data_[size()], ElemTy(0));
  }

  ElemTy &at(size_t x, size_t y, size_t z) {
    return data_[getElementIdx(x, y, z)];
  }

  void dump(std::string title = "", std::string suffix = "") {
    std::cout<< title << "[";
    for (int x = 0; x < sx_; x++) {
      std::cout<<"[";
      for (int y = 0; y < sy_; y++) {
        std::cout<<"[";
        for (int z = 0; z < sz_; z++) {
            std::cout<< at(x, y, z) << " ";
        }
        std::cout<<"]";
      }
      std::cout<<"]";
    }
    std::cout<<"]" << suffix;
  }

};

}

#endif // NOETHER_TENSOR_H
