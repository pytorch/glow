#include <cstddef>
#include <cstdint>
#include <vector>
#include <iostream>
#include <string>
#include <cassert>

/// A 3D tensor.
template <class ElemTy>
class Array3D final {
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

  /// \returns the dimension of the tensor.
  std::tuple<size_t, size_t, size_t> dims() const { return {sx_, sy_, sz_}; }

  /// \returns the number of elements in the array.
  size_t size() const { return sx_ * sy_ * sz_; }

  /// Initialize an empty tensor.
  Array3D() = default;

  /// Initialize a new tensor.
  Array3D(size_t x, size_t y, size_t z) : sx_(x), sy_(y), sz_(z) {
    data_ = new ElemTy[size()];
  }

  /// Assigns a new shape to the tensor and allocates a new buffer.
  void reset(size_t x, size_t y, size_t z) {
    sx_ = x;
    sy_ = y;
    sz_ = z;
    delete data_;
    data_ = new ElemTy[size()];
  }

  ~Array3D() { delete data_; }

  ElemTy &get(size_t x, size_t y, size_t z) const {
    return data_[getElementIdx(x,y,z)];
  }
};

/// Represents a node in the network compute graph.
template <class ElemTy>
class Layer {

  /// \returns a descriptive name for the operation.
  virtual std::string getName() = 0;

  /// \returns the output of a node in the compute graph.
  virtual Array3D<ElemTy> &getOutput() = 0;

  /// \returns the dimension of the tensor.
  std::tuple<size_t, size_t, size_t> dims() const { return getOutput().dims(); }

  /// Does the forward propagation.
  void forward() = 0;

  /// Does the backwards propagation.
  void backward() = 0;
};


template <class ElemTy>
class ConvLayer final : public Layer<ElemTy> {
  Layer<ElemTy> *input_;
  /// A list of convolution filters.
  std::vector<Array3D<ElemTy>> filters_;
  /// The convolution bias.
  Array3D<ElemTy> bias_;
  /// The convolution output.
  Array3D<ElemTy> output_;

  size_t filterSize_;
  size_t stride_;
  size_t pad_;

  ConvLayer(Layer<ElemTy> *input, size_t outDepth, size_t filterSize, size_t stride, size_t pad) :
    input_(input), filterSize_(filterSize), stride_(stride), pad_(pad) {
    assert(pad == 0 && "Unsupported pad size");
    assert(input && "Invalid input layer");
    size_t inx, iny, inz;
    std::tie(inx, iny, inz) = input_->dims();

    size_t outsx = ((inx + pad_ * 2 - filterSize) / stride + 1);
    size_t outsy = ((iny + pad_ * 2 - filterSize) / stride + 1);

    output_.reset(outsx, outsy, outDepth);
    bias_.reset(1, 1, outDepth);

    for (size_t i = 0 ; i < outDepth; i++) {
      filters_.emplace(filterSize, filterSize, inz);
    }
  }

  void forward() {
    size_t outx, outy, outz;
    std::tie(outx, outy, outz) = output_->dims();
    size_t inx, iny, inz;
    std::tie(inx, iny, inz) = input_->dims();

    auto &inputBuffer = input_->getOutput();

    // For each layer in the output tensor:
    for (size_t d = 0; d < outz; d++) {
      auto &currFilter = filters_[d];

      // For each convolution 'jump' in the input tensor:
      size_t y = 0;
      for (size_t ay = 0; ay<outy ; y+=stride_, ay++) {
        size_t x = 0;
        for (size_t ax = 0; ax<outy ; x+=stride_, ax++) {

          ElemTy sum = 0;
          for (size_t fy = 0; fy < filterSize_; fy++) {
            for (size_t fx = 0; fx < filterSize_; fx++) {
              auto ox = x + fx;
              auto oy = y + fy;

              if (output_.isInBounds(ox, oy)) {
                for (size_t fd = 0; fd < inz; fd++) {
                  sum += currFilter.get(fx, fy, fd) * inputBuffer.get(ox, oy, fd);
                }
              }

            }
          }

          sum += bias_.get(d);
          output_.get(ax, ay, d) = sum;
        }
      }

    }
  }

};


int main() {
  Array3D<float> X(320,200,3);
  X.get(10u,10u,2u) = 2;

}
