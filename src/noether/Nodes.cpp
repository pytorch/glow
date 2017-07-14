#include "noether/Nodes.h"
#include "noether/Tensor.h"

using namespace noether;

ConvNode::ConvNode(Network *N, TrainableNode *input, size_t outDepth,
                   size_t filterSize, size_t stride, size_t pad)
    : TrainableNode(N), input_(input), filterSize_(filterSize), stride_(stride),
      pad_(pad) {
  assert(input && input_->size() && "Invalid input");
  N->addNodeDependency(this, input);
  size_t inx, iny, inz;
  std::tie(inx, iny, inz) = input_->dims();
  assert(inx > filterSize && iny > filterSize &&
         "buffer too small for selected stride");

  size_t outsx = ((inx + pad_ * 2 - filterSize) / stride + 1);
  size_t outsy = ((iny + pad_ * 2 - filterSize) / stride + 1);

  this->output_.reset(outsx, outsy, outDepth);
  bias_.reset(1, 1, outDepth);

  // RELUs like small positive bias to get gradients early in the training
  // process, otherwise the RELU units may never turn on and turn into a
  // "dead RELU".
  for (size_t i = 0; i < outDepth; i++) {
    bias_.weight_.at(0, 0, i) = 0.1;
  }

  for (size_t i = 0; i < outDepth; i++) {
    filters_.emplace_back(filterSize, filterSize, inz);
  }

  for (size_t i = 0; i < outDepth; i++) {
    N->registerDerivTensor(this, &filters_[i]);
  }

  N->registerDerivTensor(this, &bias_);
}

void ConvNode::forward() {
  size_t outx, outy, outz;
  std::tie(outx, outy, outz) = this->output_.dims();
  size_t inx, iny, inz;
  std::tie(inx, iny, inz) = input_->dims();
  auto &inputBuffer = input_->getOutput();

  // For each layer in the output tensor:
  for (size_t d = 0; d < outz; d++) {
    auto &currFilter = filters_[d];

    // For each convolution 'jump' in the input tensor:
    ssize_t y = -ssize_t(pad_);
    for (size_t ay = 0; ay < outy; y += stride_, ay++) {
      ssize_t x = -ssize_t(pad_);
      for (size_t ax = 0; ax < outy; x += stride_, ax++) {

        // For each element in the convolution-filter:
        FloatTy sum = 0;
        for (size_t fy = 0; fy < filterSize_; fy++) {
          for (size_t fx = 0; fx < filterSize_; fx++) {
            ssize_t ox = x + fx;
            ssize_t oy = y + fy;

            // Ignore index access below zero (this is due to padding).
            if (ox < 0 || oy < 0)
              continue;

            if (this->output_.isInBounds(ox, oy, 0)) {
              for (size_t fd = 0; fd < inz; fd++) {
                sum += currFilter.weight_.at(fx, fy, fd) *
                       inputBuffer.weight_.at(ox, oy, fd);
              }
            }
          }
        }

        sum += bias_.weight_.at(0, 0, d);
        this->output_.weight_.at(ax, ay, d) = sum;
      }
    }
  }
}

void ConvNode::backward() {
  size_t outx, outy, outz;
  std::tie(outx, outy, outz) = this->output_.dims();
  size_t inx, iny, inz;
  std::tie(inx, iny, inz) = input_->dims();
  auto &inputBuffer = input_->getOutput();

  // Zero the gradient of the input.
  inputBuffer.gradient_.clear();

  // Compute the gradient. For each layer in the output tensor:
  for (size_t d = 0; d < outz; d++) {
    auto &currFilter = filters_[d];

    // For each convolution 'jump' in the input tensor:
    ssize_t y = -ssize_t(pad_);
    for (size_t ay = 0; ay < ssize_t(outy); y += stride_, ay++) {
      ssize_t x = -ssize_t(pad_);
      for (size_t ax = 0; ax < ssize_t(outx); x += stride_, ax++) {

        FloatTy chainGrad = this->output_.gradient_.at(ax, ay, d);

        // For each element in the convolution-filter:
        for (size_t fy = 0; fy < filterSize_; fy++) {
          for (size_t fx = 0; fx < filterSize_; fx++) {
            ssize_t ox = x + fx;
            ssize_t oy = y + fy;

            // Ignore index access below zero (this is due to padding).
            if (ox < 0 || oy < 0)
              continue;

            if (this->output_.isInBounds(ox, oy, 0)) {
              for (size_t fd = 0; fd < inz; fd++) {
                currFilter.gradient_.at(fx, fy, fd) += inputBuffer.weight_.at(ox, oy, fd) * chainGrad;
                inputBuffer.gradient_.at(ox, oy, fd) += currFilter.weight_.at(fx, fy, fd) * chainGrad;
              }
            }
          }
        }

        bias_.gradient_.at(0, 0, d) += chainGrad;
      }
    }
  }
}

MaxPoolNode::MaxPoolNode(Network *N, TrainableNode *input, size_t filterSize,
                         size_t stride, size_t pad)
    : TrainableNode(N), input_(input), filterSize_(filterSize), stride_(stride),
      pad_(pad) {
  assert(input && input_->size() && "Invalid input");
  N->addNodeDependency(this, input);
  size_t inx, iny, inz;
  std::tie(inx, iny, inz) = input_->dims();
  assert(inx > filterSize && iny > filterSize &&
         "buffer too small for selected stride");

  size_t outsx = ((inx + pad_ * 2 - filterSize) / stride + 1);
  size_t outsy = ((iny + pad_ * 2 - filterSize) / stride + 1);

  this->output_.reset(outsx, outsy, inz);

  // Resize the arrays that store the x and y coordinates of the incoming
  // gradient.
  srcX_.reset(outsx, outsy, inz);
  srcY_.reset(outsx, outsy, inz);
}

void MaxPoolNode::forward() {
  size_t outx, outy, outz;
  std::tie(outx, outy, outz) = this->output_.dims();
  size_t inx, iny, inz;
  std::tie(inx, iny, inz) = input_->dims();
  auto &inputBuffer = input_->getOutput();

  // For each layer in the output tensor:
  for (size_t z = 0; z < inz; z++) {
    // For each convolution 'jump' in the input tensor:
    ssize_t y = -ssize_t(pad_);
    for (size_t ay = 0; ay < outy; y += stride_, ay++) {
      ssize_t x = -ssize_t(pad_);
      for (size_t ax = 0; ax < outx; x += stride_, ax++) {
        size_t maxX = x;
        size_t maxY = y;

        bool first = true;
        FloatTy max = 0;

        for (size_t fy = 0; fy < filterSize_; fy++) {
          for (size_t fx = 0; fx < filterSize_; fx++) {
            ssize_t ox = x + fx;
            ssize_t oy = y + fy;

            // Ignore index access below zero (this is due to padding).
            if (ox < 0 || oy < 0)
              continue;

            if (inputBuffer.isInBounds(ox, oy, z)) {
              FloatTy val = inputBuffer.weight_.at(ox, oy, z);

              if (first || (val >= max)) {
                first = false;
                max = val;
                maxX = ox;
                maxY = oy;
              }
            }
          }
        }

        assert(!first && "Max value is uninitialized");
        srcX_.at(ax, ay, z) = maxX;
        srcY_.at(ax, ay, z) = maxY;
        this->output_.weight_.at(ax, ay, z) = max;
      }
    }
  }
}

void MaxPoolNode::backward() {
  size_t outx, outy, outz;
  std::tie(outx, outy, outz) = this->output_.dims();
  size_t inx, iny, inz;
  std::tie(inx, iny, inz) = input_->dims();
  auto &inputBuffer = input_->getOutput();

  // Zero the gradient of the input.
  inputBuffer.gradient_.clear();

  // Compute the gradient. For each layer in the output tensor:
  for (size_t z = 0; z < outz; z++) {

    // For each convolution 'jump' in the input tensor:
    ssize_t y = -ssize_t(pad_);
    for (size_t ay = 0; ay < ssize_t(outy); y += stride_, ay++) {
      ssize_t x = -ssize_t(pad_);
      for (size_t ax = 0; ax < ssize_t(outy); x += stride_, ax++) {

        FloatTy chainGrad = this->output_.gradient_.at(ax, ay, z);

        size_t maxX = srcX_.at(ax, ay, z);
        size_t maxY = srcY_.at(ax, ay, z);

        inputBuffer.gradient_.at(maxX, maxY, z) += chainGrad;
      }
    }
  }
}

FullyConnectedNode::FullyConnectedNode(Network *N, TrainableNode *input,
                                       size_t outDepth)
    : TrainableNode(N), input_(input) {
  assert(input && input_->size() && "Invalid input");
  N->addNodeDependency(this, input);
  size_t inx, iny, inz;
  std::tie(inx, iny, inz) = input_->dims();

  this->output_.reset(1, 1, outDepth);
  bias_.reset(1, 1, outDepth);

  // RELUs like small positive bias to get gradients early in the training
  // process, otherwise the RELU units may never turn on and turn into a
  // "dead RELU".
  for (size_t i = 0; i < outDepth; i++) {
    bias_.weight_.at(0, 0, i) = 0.1;
  }

  for (size_t i = 0; i < outDepth; i++) {
    filters_.emplace_back(inx, iny, inz);
  }

  for (auto &filter : filters_) {
    filter.weight_.randomize();
  }

  for (size_t i = 0; i < outDepth; i++) {
    N->registerDerivTensor(this, &filters_[i]);
  }

  N->registerDerivTensor(this, &bias_);
}

void FullyConnectedNode::forward() {
  size_t inx, iny, inz;
  std::tie(inx, iny, inz) = input_->dims();
  size_t outx, outy, outz;
  std::tie(outx, outy, outz) = this->output_.dims();
  auto &inputBuffer = input_->getOutput();

  for (size_t i = 0; i < outz; i++) {
    auto &currFilter = filters_[i];
    FloatTy sum = 0;

    for (size_t x = 0; x < inx; x++) {
      for (size_t y = 0; y < iny; y++) {
        for (size_t z = 0; z < inz; z++) {
          sum +=
              inputBuffer.weight_.at(x, y, z) * currFilter.weight_.at(x, y, z);
        }
      }
    }
    sum += bias_.weight_.at(0, 0, i);
    this->output_.weight_.at(0, 0, i) = sum;
  }
}

void FullyConnectedNode::backward() {
  size_t inx, iny, inz;
  std::tie(inx, iny, inz) = input_->dims();
  size_t outx, outy, outz;
  std::tie(outx, outy, outz) = this->output_.dims();
  auto &inputBuffer = input_->getOutput();

  // Zero the gradient of the input.
  inputBuffer.gradient_.clear();

  // Compute the gradient:
  for (size_t i = 0; i < outz; i++) {
    auto &currFilter = filters_[i];
    FloatTy chainGrad = this->output_.gradient_.at(0, 0, i);

    for (size_t x = 0; x < inx; x++) {
      for (size_t y = 0; y < iny; y++) {
        for (size_t z = 0; z < inz; z++) {
          // Input gradient:
          inputBuffer.gradient_.at(x, y, z) +=
              currFilter.weight_.at(x, y, z) * chainGrad;
          // Param gradient:
          currFilter.gradient_.at(x, y, z) +=
              inputBuffer.weight_.at(x, y, z) * chainGrad;
        }
      }
    }

    this->bias_.gradient_.at(0, 0, i) += chainGrad;
  }
}

RELUNode::RELUNode(Network *N, TrainableNode *input)
    : TrainableNode(N), input_(input) {
  assert(input && input_->size() && "Invalid input");
  this->output_.reset(input_->dims());
  N->addNodeDependency(this, input);
}

void RELUNode::forward() {
  size_t inx, iny, inz;
  std::tie(inx, iny, inz) = input_->dims();
  auto &inputBuffer = input_->getOutput();

  auto &OutW = this->output_.weight_;
  auto &InW = inputBuffer.weight_;

  for (size_t x = 0; x < inx; x++) {
    for (size_t y = 0; y < iny; y++) {
      for (size_t z = 0; z < inz; z++) {
        FloatTy val = InW.at(x, y, z);
        OutW.at(x, y, z) = val < 0 ? 0 : val;
      }
    }
  }
}

void RELUNode::backward() {
  size_t inx, iny, inz;
  std::tie(inx, iny, inz) = input_->dims();
  auto &inputBuffer = input_->getOutput();

  auto &OutW = this->output_.weight_;
  auto &OutDW = this->output_.gradient_;
  auto &InDW = inputBuffer.gradient_;

  for (size_t x = 0; x < inx; x++) {
    for (size_t y = 0; y < iny; y++) {
      for (size_t z = 0; z < inz; z++) {
        FloatTy val = OutW.at(x, y, z);
        InDW.at(x, y, z) = (val <= 0 ? 0 : OutDW.at(x, y, z));
      }
    }
  }
}

SigmoidNode::SigmoidNode(Network *N, TrainableNode *input)
    : TrainableNode(N), input_(input) {
  assert(input && input_->size() && "Invalid input");
  this->output_.reset(input_->dims());
  N->addNodeDependency(this, input);
}

void SigmoidNode::forward() {
  size_t inx, iny, inz;
  std::tie(inx, iny, inz) = input_->dims();
  auto &inputBuffer = input_->getOutput();

  auto &OutW = this->output_.weight_;
  auto &InW = inputBuffer.weight_;

  for (size_t x = 0; x < inx; x++) {
    for (size_t y = 0; y < iny; y++) {
      for (size_t z = 0; z < inz; z++) {
        FloatTy val = InW.at(x, y, z);
        OutW.at(x, y, z) = 1 / (1 + std::exp(-val));
      }
    }
  }
}

void SigmoidNode::backward() {
  size_t inx, iny, inz;
  std::tie(inx, iny, inz) = input_->dims();
  auto &inputBuffer = input_->getOutput();

  auto &OutW = this->output_.weight_;
  auto &OutDW = this->output_.gradient_;
  auto &InDW = inputBuffer.gradient_;

  for (size_t x = 0; x < inx; x++) {
    for (size_t y = 0; y < iny; y++) {
      for (size_t z = 0; z < inz; z++) {
        FloatTy val = OutW.at(x, y, z);
        InDW.at(x, y, z) = val * (1 - val) * OutDW.at(x, y, z);
      }
    }
  }
}

SoftMaxNode::SoftMaxNode(Network *N, TrainableNode *input, size_t selected)
    : TrainableNode(N), input_(input), selected_(selected) {
  assert(input && input_->size() && "Invalid input");
  N->addNodeDependency(this, input);
  size_t inx, iny, inz;
  std::tie(inx, iny, inz) = input_->dims();
  assert(inx == 1 && iny == 1 && "Softmax input must be 1x1xN");
  this->output_.reset(1, 1, inz);
  e_.reset(1, 1, inz);
}

void SoftMaxNode::forward() {
  size_t inx, iny, inz;
  std::tie(inx, iny, inz) = input_->dims();
  auto &inputBuffer = input_->getOutput();

  auto &OutW = this->output_.weight_;
  auto &InW = inputBuffer.weight_;

  FloatTy max = InW.at(0, 0, 0);

  // Find Max.
  for (size_t z = 0; z < inz; z++) {
    max = std::max(max, InW.at(0, 0, z));
  }

  FloatTy sum = 0;

  // Compute exp.
  for (size_t z = 0; z < inz; z++) {
    FloatTy e = std::exp(InW.at(0, 0, z) - max);
    sum += e;
    e_.at(0, 0, z) = e;
  }

  // Normalize the output.
  for (size_t z = 0; z < inz; z++) {
    e_.at(0, 0, z) /= sum;
    OutW.at(0, 0, z) = e_.at(0, 0, z);
  }
}

void SoftMaxNode::backward() {
  size_t inx, iny, inz;
  std::tie(inx, iny, inz) = input_->dims();
  auto &inputBuffer = input_->getOutput();
  auto &InDW = inputBuffer.gradient_;

  for (size_t z = 0; z < inz; z++) {
    FloatTy indicator = (selected_ == z ? 1 : 0);
    FloatTy mul = -(indicator - e_.at(0, 0, z));
    InDW.at(0, 0, z) = mul;
  }
}

size_t SoftMaxNode::maxArg() const {
  size_t inx, iny, inz;
  std::tie(inx, iny, inz) = input_->dims();

  auto &OutW = this->output_.weight_;
  FloatTy max = OutW.at(0, 0, 0);
  size_t idx = 0;

  for (size_t i = 1; i < inz; i++) {
    FloatTy val = OutW.at(0, 0, i);
    if (val > max) {
      max = val;
      idx = i;
    }
  }
  return idx;
}

void SoftMaxNode::setSelected(size_t selected) {
  size_t inx, iny, inz;
  std::tie(inx, iny, inz) = input_->dims();
  assert(inx == 1 && iny == 1 && selected < inz && "Invalid selection");
  selected_ = selected;
}

RegressionNode::RegressionNode(Network *N, TrainableNode *input)
    : TrainableNode(N), input_(input) {
  assert(input && input_->size() && "Invalid input");
  N->addNodeDependency(this, input);
  size_t inx, iny, inz;
  std::tie(inx, iny, inz) = input_->dims();
  assert(inx == 1 && iny == 1 && "input must be 1x1xN");
  expected_.reset(1, 1, inz);
  this->output_.reset(1, 1, inz);
}

void RegressionNode::forward() {
  assert(expected_.dims() == input_->dims() && "invalid expected dims");
  size_t inx, iny, inz;
  std::tie(inx, iny, inz) = input_->dims();
  auto &inputBuffer = input_->getOutput();

  auto &OutW = this->output_.weight_;
  auto &InW = inputBuffer.weight_;

  for (size_t z = 0; z < inz; z++) {
    OutW.at(0, 0, z) = InW.at(0, 0, z);
  }
}

void RegressionNode::backward() {
  assert(expected_.dims() == input_->dims() && "invalid expected dims");

  size_t inx, iny, inz;
  std::tie(inx, iny, inz) = input_->dims();
  auto &inputBuffer = input_->getOutput();
  auto &InDW = inputBuffer.gradient_;

  FloatTy loss = 0;

  for (size_t z = 0; z < inz; z++) {
    FloatTy dy = (inputBuffer.weight_.at(0, 0, z) - expected_.at(0, 0, z));
    InDW.at(0, 0, z) = dy;
    loss += 0.5 * dy * dy;
  }
}

MaxNode::MaxNode(Network *N, TrainableNode *input)
    : TrainableNode(N), input_(input) {
  assert(input && input_->size() && "Invalid input");
  N->addNodeDependency(this, input);
  size_t inx, iny, inz;
  std::tie(inx, iny, inz) = input_->dims();
  this->output_.reset(inx, iny, inz);
}

void MaxNode::forward() {
  size_t inx, iny, inz;
  std::tie(inx, iny, inz) = input_->dims();
  auto &inputBuffer = input_->getOutput();
  auto &OutW = this->output_.weight_;
  auto &InW = inputBuffer.weight_;

  for (size_t x = 0; x < inz; x++) {
    for (size_t y = 0; y < inz; y++) {
      for (size_t z = 0; z < inz; z++) {
        OutW.at(x, y, z) = InW.at(x, y, z);
      }
    }
  }
}

void MaxNode::backward() {
  size_t inx, iny, inz;
  std::tie(inx, iny, inz) = input_->dims();
  auto &inputBuffer = input_->getOutput();
  auto &InDW = inputBuffer.gradient_;

  for (size_t x = 0; x < inz; x++) {
    for (size_t y = 0; y < inz; y++) {
      for (size_t z = 0; z < inz; z++) {
        FloatTy dy = inputBuffer.weight_.at(x, y, z);
        InDW.at(x, y, z) = dy > 0 ? -1 : 1;
      }
    }
  }
}
