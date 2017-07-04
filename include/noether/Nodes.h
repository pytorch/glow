#ifndef NOETHER_NODES_H
#define NOETHER_NODES_H

#include "noether/Tensor.h"
#include "noether/Node.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <cmath>

namespace noether {

template <class ElemTy> class ConvNode final : public Node<ElemTy> {
  Node<ElemTy> *input_;
  /// A list of convolution filters.
  std::vector<DerivData<ElemTy>> filters_;
  /// The convolution bias.
  DerivData<ElemTy> bias_;

  size_t filterSize_;
  size_t stride_;
  size_t pad_;

public:

  ConvNode(Network *N, Node<ElemTy> *input, size_t outDepth, size_t filterSize,
            size_t stride, size_t pad)
      : Node<ElemTy>(N), input_(input), filterSize_(filterSize), stride_(stride), pad_(pad) {
    assert(pad == 0 && "Unsupported pad size");
    assert(input && input_->size() && "Invalid input");
    N->addNodeDependency(this, input);
    size_t inx, iny, inz;
    std::tie(inx, iny, inz) = input_->dims();

    size_t outsx = ((inx + pad_ * 2 - filterSize) / stride + 1);
    size_t outsy = ((iny + pad_ * 2 - filterSize) / stride + 1);

    this->output_.reset(outsx, outsy, outDepth);
    bias_.reset(1, 1, outDepth);

    for (size_t i = 0; i < outDepth; i++) {
      filters_.emplace_back(filterSize, filterSize, inz);
    }

    for (size_t i = 0; i < outDepth; i++) {
      N->registerDerivTensor(this, &filters_[i]);
    }

    N->registerDerivTensor(this, &bias_);
  }

  virtual void forward() override {
    size_t outx, outy, outz;
    std::tie(outx, outy, outz) = this->output_.dims();
    size_t inx, iny, inz;
    std::tie(inx, iny, inz) = input_->dims();
    auto &inputBuffer = input_->getOutput();

    // For each layer in the output tensor:
    for (size_t d = 0; d < outz; d++) {
      auto &currFilter = filters_[d];

      // For each convolution 'jump' in the input tensor:
      size_t y = 0;
      for (size_t ay = 0; ay < outy; y += stride_, ay++) {
        size_t x = 0;
        for (size_t ax = 0; ax < outy; x += stride_, ax++) {

          // For each element in the convolution-filter:
          ElemTy sum = 0;
          for (size_t fy = 0; fy < filterSize_; fy++) {
            for (size_t fx = 0; fx < filterSize_; fx++) {
              auto ox = x + fx;
              auto oy = y + fy;

              if (this->output_.isInBounds(ox, oy, 0)) {
                for (size_t fd = 0; fd < inz; fd++) {
                  sum += currFilter.weight_.at(fx, fy, fd) *
                    inputBuffer.weight_.at(ox, oy, fd);
                }
              }
            }
          }

          sum += bias_.weight_.at(0,0,d);
          this->output_.weight_.at(ax, ay, d) = sum;
        }
      }
    }
  }

  virtual void backward() override {
    size_t outx, outy, outz;
    std::tie(outx, outy, outz) = this->output_.dims();
    size_t inx, iny, inz;
    std::tie(inx, iny, inz) = input_->dims();
    auto &inputBuffer = input_->getOutput();

    // Zero the gradient of the input.
    inputBuffer.gradient_.randomize();

    // Compute the gradient. For each layer in the output tensor:
    for (size_t d = 0; d < outz; d++) {
      auto &currFilter = filters_[d];

      // For each convolution 'jump' in the input tensor:
      size_t y = 0;
      for (size_t ay = 0; ay < outy; y += stride_, ay++) {
        size_t x = 0;
        for (size_t ax = 0; ax < outy; x += stride_, ax++) {

          ElemTy chainGrad = this->output_.gradient_.at(ax,ay, d);

          // For each element in the convolution-filter:
          for (size_t fy = 0; fy < filterSize_; fy++) {
            for (size_t fx = 0; fx < filterSize_; fx++) {
              auto ox = x + fx;
              auto oy = y + fy;

              if (this->output_.isInBounds(ox, oy, 0)) {
                for (size_t fd = 0; fd < inz; fd++) {
                  currFilter.gradient_.at(fx, fy, fd) += chainGrad * inputBuffer.weight_.at(ox, oy, fd);
                  inputBuffer.gradient_.at(ox, oy, fd) += currFilter.weight_.at(fx, fy, fd);
                }
              }
            }
          }

          bias_.gradient_.at(0,0,d) += chainGrad;
        }
      }
    }
  }

  virtual std::string getName() const override { return "ConvNode"; }
};

template <class ElemTy> class FullyConnectedNode final : public Node<ElemTy> {
  /// A reference to the layer input.
  Node<ElemTy> *input_;
  /// A list of filters.
  std::vector<DerivData<ElemTy>> filters_;
  /// The biases.
  DerivData<ElemTy> bias_;

public:
  FullyConnectedNode(Network *N, Node<ElemTy> *input, size_t outDepth)
      : Node<ElemTy>(N), input_(input) {
    assert(input && input_->size() && "Invalid input");
    N->addNodeDependency(this, input);
    size_t inx, iny, inz;
    std::tie(inx, iny, inz) = input_->dims();

    this->output_.reset(1, 1, outDepth);
    bias_.reset(1, 1, outDepth);

    for (size_t i = 0; i < outDepth; i++) {
      filters_.emplace_back(inx, iny, inz);
    }

    for (size_t i = 0; i < outDepth; i++) {
      N->registerDerivTensor(this, &filters_[i]);
    }

    N->registerDerivTensor(this, &bias_);
  }

  virtual void forward() override {
    size_t inx, iny, inz;
    std::tie(inx, iny, inz) = input_->dims();
    size_t outx, outy, outz;
    std::tie(outx, outy, outz) = this->output_.dims();
    auto &inputBuffer = input_->getOutput();

    for (size_t i = 0; i < outz; i++) {
      auto &currFilter = filters_[i];
      ElemTy sum = 0;

      for (size_t x = 0; x < inx; x++) {
        for (size_t y = 0; y < iny; y++) {
          for (size_t z = 0; z < inz; z++) {
            sum += inputBuffer.weight_.at(x,y,z) * currFilter.weight_.at(x,y,z);
          }
        }
      }
      sum += bias_.weight_.at(0,0,i);
      this->output_.weight_.at(0,0,i) = sum;
    }
  }

  virtual void backward() override {
    size_t inx, iny, inz;
    std::tie(inx, iny, inz) = input_->dims();
    size_t outx, outy, outz;
    std::tie(outx, outy, outz) = this->output_.dims();
    auto &inputBuffer = input_->getOutput();

    // Zero the gradient of the input.
    inputBuffer.gradient_.randomize();

    // Compute the gradient:
    for (size_t i = 0; i < outz; i++) {
      auto &currFilter = filters_[i];
      ElemTy chainGrad = this->output_.gradient_.at(0,0,i);

      for (size_t x = 0; x < inx; x++) {
        for (size_t y = 0; y < iny; y++) {
          for (size_t z = 0; z < inz; z++) {
            // Input gradient:
            inputBuffer.gradient_.at(x,y,z) += currFilter.weight_.at(x,y,z) * chainGrad;
            // Param gradient:
            currFilter.gradient_.at(x,y,z) += inputBuffer.weight_.at(x,y,z) * chainGrad;
          }
        }
      }

      this->bias_.gradient_.at(0,0,i) += chainGrad;
    }
  }

  virtual std::string getName() const override { return "FullyConnectedNode"; }
};


template <class ElemTy> class RELUNode final : public Node<ElemTy> {
  /// A reference to the layer input.
  Node<ElemTy> *input_;

public:
  RELUNode(Network *N, Node<ElemTy> *input)
  : Node<ElemTy>(N), input_(input) {
    assert(input && input_->size() && "Invalid input");
    this->output_.reset(input_->dims());
    N->addNodeDependency(this, input);
  }

  virtual void forward() override {
    size_t inx, iny, inz;
    std::tie(inx, iny, inz) = input_->dims();
    auto &inputBuffer = input_->getOutput();

    auto &OutW = this->output_.weight_;
    auto &InW = inputBuffer.weight_;

    for (size_t x = 0; x < inx; x++) {
      for (size_t y = 0; y < iny; y++) {
        for (size_t z = 0; z < inz; z++) {
          ElemTy val = InW.at(x,y,z);
          OutW.at(x,y,z) = val < 0 ? 0 : val;
        }
      }
    }
  }

  virtual void backward() override {
    size_t inx, iny, inz;
    std::tie(inx, iny, inz) = input_->dims();
    auto &inputBuffer = input_->getOutput();

    auto &OutW = this->output_.weight_;
    auto &OutDW = this->output_.gradient_;
    auto &InDW = inputBuffer.gradient_;

    for (size_t x = 0; x < inx; x++) {
      for (size_t y = 0; y < iny; y++) {
        for (size_t z = 0; z < inz; z++) {
          ElemTy val = OutW.at(x,y,z);
          InDW.at(x,y,z) = (val < 0 ? 0 : OutDW.at(x,y,z));
        }
      }
    }
  }

  virtual std::string getName() const override { return "RELUNode"; }
};


template <class ElemTy> class SoftMaxNode final : public Node<ElemTy> {
  /// A reference to the node input.
  Node<ElemTy> *input_;
  /// The selected one-hot value from the softmax function.
  size_t selected_;

  /// A temporary array for storing the subexpression (e ^ (a[i] - max)).
  Array3D<ElemTy> e_;

public:
  /// Ctor - \p is the input layer that must be of shape (1 x 1 x N).
  /// And \p selected that's the selected one-hot representation of the
  /// softmax function.
  SoftMaxNode(Network *N, Node<ElemTy> *input, size_t selected = 0)
  : Node<ElemTy>(N), input_(input), selected_(selected) {
    assert(input && input_->size() && "Invalid input");
    N->addNodeDependency(this, input);
    size_t inx, iny, inz;
    std::tie(inx, iny, inz) = input_->dims();
    assert(inx == 1 && iny == 1 && "Softmax input must be 1x1xN");
    this->output_.reset(1, 1, inz);
    e_.reset(1,1,inz);
  }

  virtual void forward() override {
    size_t inx, iny, inz;
    std::tie(inx, iny, inz) = input_->dims();
    auto &inputBuffer = input_->getOutput();

    auto &OutW = this->output_.weight_;
    auto &InW = inputBuffer.weight_;

    ElemTy max = InW.at(0, 0, 0);

    // Find Max.
    for (size_t z = 0; z < inz; z++) {
      max = std::max(max, InW.at(0, 0, z));
    }

    ElemTy sum = 0;

    // Compute exp.
    for (size_t z = 0; z < inz; z++) {
      ElemTy e = std::exp(InW.at(0, 0, z) - max);
      sum += e;
      e_.at(0,0,z) = e;
    }

    // Normalize the output.
    for (size_t z = 0; z < inz; z++) {
      e_.at(0, 0, z) /= sum;
      OutW.at(0, 0, z) = e_.at(0, 0, z);
    }
  }

  void setSelected(size_t selected) {
    size_t inx, iny, inz;
    std::tie(inx, iny, inz) = input_->dims();
    assert(inx == 1 && iny == 1 && selected < inz && "Invalid selection");
    selected_ = selected;
  }

  virtual void backward() override {
    size_t inx, iny, inz;
    std::tie(inx, iny, inz) = input_->dims();
    auto &inputBuffer = input_->getOutput();
    auto &InDW = inputBuffer.gradient_;

    InDW.randomize();

    for (size_t z = 0; z < inz; z++) {
      ElemTy indicator = (selected_ == z ? 1 : 0);
      ElemTy mul = -(indicator - e_.at(0,0,z));
      InDW.at(0,0,z) = mul;
    }
  }

  ElemTy loss () {
    // The loss is the class' negative log likelihood.
    return -std::log(e_.at(0,0,selected_));
  }

  virtual std::string getName() const override { return "SoftMaxNode"; }
};

}

#endif // NOETHER_NODES_H
