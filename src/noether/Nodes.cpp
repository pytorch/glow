#include "noether/Nodes.h"
#include "noether/Network.h"
#include "noether/Tensor.h"

using namespace noether;

ConvNode::ConvNode(Network *N, TrainableNode *input, size_t outDepth,
                   size_t filterSize, size_t stride, size_t pad)
    : TrainableNode(N), input_(input), filterSize_(filterSize), stride_(stride),
      pad_(pad) {
  assert(input && input_->size() && "Invalid input");
  auto idim = input_->dims();
  assert(idim[0] > filterSize && idim[1] > filterSize &&
         "buffer too small for selected stride");

  unsigned outsx = ((idim[0] + pad_ * 2 - filterSize) / stride + 1);
  unsigned outsy = ((idim[1] + pad_ * 2 - filterSize) / stride + 1);

  this->output_.reset({outsx, outsy, (unsigned)outDepth});
  bias_.reset({1, 1, (unsigned)outDepth});

  // RELUs like small positive bias to get gradients early in the training
  // process, otherwise the RELU units may never turn on and turn into a
  // "dead RELU".
  auto biasWeights = bias_.weight_.getHandle();
  for (unsigned i = 0; i < outDepth; i++) {
    biasWeights.at({0, 0, i}) = 0.1;
  }

  for (size_t i = 0; i < outDepth; i++) {
    auto dims = {(unsigned)filterSize, (unsigned)filterSize, idim[2]};
    filters_.emplace_back(dims);
  }

  for (auto &filter : filters_) {
    filter.weight_.randomize();
  }

  for (size_t i = 0; i < outDepth; i++) {
    N->registerDerivTensor(this, &filters_[i]);
  }

  N->registerDerivTensor(this, &bias_);
}

void ConvNode::forward() {
  auto odim = this->output_.dims();
  auto idim = input_->dims();

  auto inW = input_->getOutput().weight_.getHandle();
  auto outW = this->output_.weight_.getHandle();
  auto biasW = bias_.weight_.getHandle();

  // For each layer in the output tensor:
  for (unsigned d = 0; d < odim[2]; d++) {
    auto currFilterW = filters_[d].weight_.getHandle();


    // For each convolution 'jump' in the input tensor:
    ssize_t y = -ssize_t(pad_);
    for (unsigned ay = 0; ay < odim[1]; y += stride_, ay++) {
      ssize_t x = -ssize_t(pad_);
      for (unsigned ax = 0; ax < odim[0]; x += stride_, ax++) {

        // For each element in the convolution-filter:
        FloatTy sum = 0;
        for (unsigned fy = 0; fy < filterSize_; fy++) {
          for (unsigned fx = 0; fx < filterSize_; fx++) {
            ssize_t ox = x + fx;
            ssize_t oy = y + fy;

            // Ignore index access below zero (this is due to padding).
            if (ox < 0 || oy < 0)
              continue;

            if (outW.isInBounds({(unsigned)ox,(unsigned)oy, 0})) {
              for (unsigned fd = 0; fd < idim[2]; fd++) {
                sum += currFilterW.at({fx, fy, fd}) *
                  inW.at({(unsigned)ox, (unsigned)oy, fd});
              }
            }
          }
        }

        sum += biasW.at({0, 0, d});
        outW.at({ax, ay, d}) = sum;
      }
    }
  }
}

void ConvNode::backward() {
  auto odim = this->output_.dims();
  auto idim = input_->dims();
  auto &inputBuffer = input_->getOutput();
  auto inW = inputBuffer.weight_.getHandle();
  auto inG = inputBuffer.gradient_.getHandle();
  auto outG = this->output_.gradient_.getHandle();
  auto biasG = bias_.gradient_.getHandle();

  // Zero the gradient of the input.
  inG.clear();

  // Compute the gradient. For each layer in the output tensor:
  for (unsigned d = 0; d < odim[2]; d++) {
    auto currFilterG = filters_[d].gradient_.getHandle();
    auto currFilterW = filters_[d].weight_.getHandle();

    // For each convolution 'jump' in the input tensor:
    ssize_t y = -ssize_t(pad_);
    for (unsigned ay = 0; ay < ssize_t(odim[1]); y += stride_, ay++) {
      ssize_t x = -ssize_t(pad_);
      for (unsigned ax = 0; ax < ssize_t(odim[0]); x += stride_, ax++) {

        FloatTy chainGrad = outG.at({ax, ay, d});

        // For each element in the convolution-filter:
        for (unsigned fy = 0; fy < filterSize_; fy++) {
          for (unsigned fx = 0; fx < filterSize_; fx++) {
            ssize_t ox = x + fx;
            ssize_t oy = y + fy;

            // Ignore index access below zero (this is due to padding).
            if (ox < 0 || oy < 0)
              continue;

            if (outG.isInBounds({(unsigned)ox, (unsigned)oy, 0u})) {
              for (unsigned fd = 0; fd < idim[2]; fd++) {
                currFilterG.at({fx, fy, fd}) +=
                inW.at({(unsigned)ox, (unsigned)oy, fd}) * chainGrad;
                inG.at({(unsigned)ox, (unsigned)oy, fd}) +=
                currFilterW.at({fx, fy, fd}) * chainGrad;
              }
            }
          }
        }

        biasG.at({0, 0, d}) += chainGrad;
      }
    }
  }
}

MaxPoolNode::MaxPoolNode(Network *N, TrainableNode *input, size_t filterSize,
                         size_t stride, size_t pad)
    : TrainableNode(N), input_(input), filterSize_(filterSize), stride_(stride),
      pad_(pad) {
  assert(input && input_->size() && "Invalid input");
  auto idim = input_->dims();
  assert(idim[0] > filterSize && idim[0] > filterSize &&
         "buffer too small for selected stride");

  unsigned outsx = ((idim[0] + pad_ * 2 - filterSize) / stride + 1);
  unsigned outsy = ((idim[1] + pad_ * 2 - filterSize) / stride + 1);

  this->output_.reset({outsx, outsy, idim[2]});

  // Resize the arrays that store the x and y coordinates of the incoming
  // gradient.
  srcX_.reset({outsx, outsy, idim[2]});
  srcY_.reset({outsx, outsy, idim[2]});
}

void MaxPoolNode::forward() {
  auto odim = this->output_.dims();
  auto idim = input_->dims();
  auto inW = input_->getOutput().weight_.getHandle();
  auto outW = this->output_.weight_.getHandle();

  auto SX = srcX_.getHandle();
  auto SY = srcY_.getHandle();

  // For each layer in the output tensor:
  for (unsigned z = 0; z < idim[2]; z++) {
    // For each convolution 'jump' in the input tensor:
    ssize_t y = -ssize_t(pad_);
    for (unsigned ay = 0; ay < odim[1]; y += stride_, ay++) {
      ssize_t x = -ssize_t(pad_);
      for (unsigned ax = 0; ax < odim[0]; x += stride_, ax++) {
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

            if (inW.isInBounds({(unsigned)ox, (unsigned)oy, z})) {
              FloatTy val = inW.at({(unsigned)ox, (unsigned)oy, z});

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
        SX.at({ax, ay, z}) = maxX;
        SY.at({ax, ay, z}) = maxY;
        outW.at({ax, ay, z}) = max;
      }
    }
  }
}

void MaxPoolNode::backward() {
  auto odim = this->output_.dims();
  auto inG = input_->getOutput().gradient_.getHandle();
  auto outG = this->output_.gradient_.getHandle();

  auto SX = srcX_.getHandle();
  auto SY = srcY_.getHandle();

  // Zero the gradient of the input.
  inG.clear();

  // Compute the gradient. For each layer in the output tensor:
  for (unsigned z = 0; z < odim[2]; z++) {

    // For each convolution 'jump' in the input tensor:
    for (unsigned ay = 0; ay < ssize_t(odim[1]); ay++) {
      for (unsigned ax = 0; ax < ssize_t(odim[0]); ax++) {

        FloatTy chainGrad = outG.at({ax, ay, z});

        unsigned maxX = SX.at({ax, ay, z});
        unsigned maxY = SY.at({ax, ay, z});

        inG.at({maxX, maxY, z}) += chainGrad;
      }
    }
  }
}

FullyConnectedNode::FullyConnectedNode(Network *N, TrainableNode *input,
                                       size_t outDepth)
    : TrainableNode(N), input_(input) {
  assert(input && input_->size() && "Invalid input");
  auto idim = input_->dims();

  this->output_.reset({1, 1, (unsigned)outDepth});
  bias_.reset({1, 1, (unsigned)outDepth});

  // RELUs like small positive bias to get gradients early in the training
  // process, otherwise the RELU units may never turn on and turn into a
  // "dead RELU".
  auto biasW = bias_.weight_.getHandle();
  for (unsigned i = 0; i < outDepth; i++) {
    biasW.at({0, 0, i}) = 0.1;
  }

  for (size_t i = 0; i < outDepth; i++) {
    auto newDim = {idim[0], idim[1], idim[2]};
    filters_.emplace_back(newDim);
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
  auto idim = input_->dims();
  auto odim = this->output_.dims();
  auto inW = input_->getOutput().weight_.getHandle();
  auto biasW = bias_.weight_.getHandle();
  auto outW = this->output_.weight_.getHandle();

  for (unsigned i = 0; i < odim[2]; i++) {
    auto currFilterW = filters_[i].weight_.getHandle();
    FloatTy sum = 0;

    for (unsigned x = 0; x < idim[0]; x++) {
      for (unsigned y = 0; y < idim[1]; y++) {
        for (unsigned z = 0; z < idim[2]; z++) {
          sum += inW.at({x, y, z}) * currFilterW.at({x, y, z});
        }
      }
    }
    sum += biasW.at({0, 0, i});
    outW.at({0, 0, i}) = sum;
  }
}

void FullyConnectedNode::backward() {
  auto idim = input_->dims();
  auto odim = this->output_.dims();
  auto &inputBuffer = input_->getOutput();
  auto outG = this->output_.gradient_.getHandle();
  auto inG = inputBuffer.gradient_.getHandle();
  auto inW = inputBuffer.weight_.getHandle();
  auto biasG = this->bias_.gradient_.getHandle();

  // Zero the gradient of the input.
  inputBuffer.gradient_.clear();

  // Compute the gradient:
  for (unsigned i = 0; i < odim[2]; i++) {
    auto filterG = filters_[i].gradient_.getHandle();
    auto filterW = filters_[i].weight_.getHandle();

    FloatTy chainGrad = outG.at({0, 0, i});

    for (unsigned x = 0; x < idim[0]; x++) {
      for (unsigned y = 0; y < idim[1]; y++) {
        for (unsigned z = 0; z < idim[2]; z++) {
          // Input gradient:
          inG.at({x, y, z}) += filterW.at({x, y, z}) * chainGrad;
          // Param gradient:
          filterG.at({x, y, z}) += inW.at({x, y, z}) * chainGrad;
        }
      }
    }

    biasG.at({0, 0, i}) += chainGrad;
  }
}

RELUNode::RELUNode(Network *N, TrainableNode *input)
    : TrainableNode(N), input_(input) {
  assert(input && input_->size() && "Invalid input");
  this->output_.reset(input_->dims());
}

void RELUNode::forward() {
  auto idim = input_->dims();
  auto &inputBuffer = input_->getOutput();

  auto outW = this->output_.weight_.getHandle();
  auto inW = inputBuffer.weight_.getHandle();

  for (unsigned x = 0; x < idim[0]; x++) {
    for (unsigned y = 0; y < idim[1]; y++) {
      for (unsigned z = 0; z < idim[2]; z++) {
        FloatTy val = inW.at({x, y, z});
        outW.at({x, y, z}) = val < 0 ? 0 : val;
      }
    }
  }
}

void RELUNode::backward() {
  auto idim = input_->dims();
  auto &inputBuffer = input_->getOutput();

  auto outW = this->output_.weight_.getHandle();
  auto outDW = this->output_.gradient_.getHandle();
  auto inDW = inputBuffer.gradient_.getHandle();

  for (unsigned x = 0; x < idim[0]; x++) {
    for (unsigned y = 0; y < idim[1]; y++) {
      for (unsigned z = 0; z < idim[2]; z++) {
        FloatTy val = outW.at({x, y, z});
        inDW.at({x, y, z}) = (val <= 0 ? 0 : outDW.at({x, y, z}));
      }
    }
  }
}

SigmoidNode::SigmoidNode(Network *N, TrainableNode *input)
    : TrainableNode(N), input_(input) {
  assert(input && input_->size() && "Invalid input");
  this->output_.reset(input_->dims());
}

void SigmoidNode::forward() {
  auto idim = input_->dims();
  auto &inputBuffer = input_->getOutput();

  auto outW = this->output_.weight_.getHandle();
  auto inW = inputBuffer.weight_.getHandle();

  for (unsigned x = 0; x < idim[0]; x++) {
    for (unsigned y = 0; y < idim[1]; y++) {
      for (unsigned z = 0; z < idim[2]; z++) {
        FloatTy val = inW.at({x, y, z});
        outW.at({x, y, z}) = 1 / (1 + std::exp(-val));
      }
    }
  }
}

void SigmoidNode::backward() {
  auto idim = input_->dims();
  auto &inputBuffer = input_->getOutput();

  auto outW = this->output_.weight_.getHandle();
  auto outDW = this->output_.gradient_.getHandle();
  auto inDW = inputBuffer.gradient_.getHandle();

  for (unsigned x = 0; x < idim[0]; x++) {
    for (unsigned y = 0; y < idim[1]; y++) {
      for (unsigned z = 0; z < idim[2]; z++) {
        FloatTy val = outW.at({x, y, z});
        inDW.at({x, y, z}) = val * (1 - val) * outDW.at({x, y, z});
      }
    }
  }
}

SoftMaxNode::SoftMaxNode(Network *N, TrainableNode *input)
    : TrainableNode(N), input_(input), selected_(0) {
  assert(input && input_->size() && "Invalid input");
  auto idim = input_->dims();
  assert(idim[0] == 1 && idim[1] == 1 && "Softmax input must be 1x1xN");
  this->output_.reset({1, 1, idim[2]});
  e_.reset({1, 1, idim[2]});
}

void SoftMaxNode::forward() {
  auto idim = input_->dims();
  auto outW = this->output_.weight_.getHandle();
  auto inW = input_->getOutput().weight_.getHandle();

  FloatTy max = inW.at({0, 0, 0});

  // Find Max.
  for (unsigned z = 0; z < idim[2]; z++) {
    max = std::max(max, inW.at({0, 0, z}));
  }

  FloatTy sum = 0;

  auto EH = e_.getHandle();
  // Compute exp.
  for (unsigned z = 0; z < idim[2]; z++) {
    FloatTy e = std::exp(inW.at({0, 0, z}) - max);
    sum += e;
    EH.at({0, 0, z}) = e;
  }

  // Normalize the output.
  for (unsigned z = 0; z < idim[2]; z++) {
    EH.at({0, 0, z}) /= sum;
    outW.at({0, 0, z}) = EH.at({0, 0, z});
  }
}

void SoftMaxNode::backward() {
  auto idim = input_->dims();
  auto inDW = input_->getOutput().gradient_.getHandle();
  auto ex = e_.getHandle();

  for (unsigned z = 0; z < idim[2]; z++) {
    FloatTy indicator = (selected_ == z ? 1 : 0);
    FloatTy mul = -(indicator - ex.at({0, 0, z}));
    inDW.at({0, 0, z}) = mul;
  }
}

size_t SoftMaxNode::maxArg() const {
  auto idim = input_->dims(); (void) idim;
  assert(idim[0] == 1 && idim[1] == 1 && "Invalid softmax shape!");

  auto &outW = this->output_.weight_;
  FloatTy max = outW.at(0);
  size_t idx = 0;

  for (size_t i = 1; i < outW.size(); i++) {
    FloatTy val = outW.at(i);
    if (val > max) {
      max = val;
      idx = i;
    }
  }
  return idx;
}

void SoftMaxNode::setSelected(size_t selected) {
  auto idim = input_->dims();
  (void)idim;
  assert(idim[0] == 1 && idim[1] == 1 && selected < idim[2] &&
         "Invalid selection");
  selected_ = selected;
}

RegressionNode::RegressionNode(Network *N, TrainableNode *input)
    : TrainableNode(N), input_(input) {
  assert(input && input_->size() && "Invalid input");
  auto idim = input_->dims();
  assert(idim[0] == 1 && idim[1] == 1 && "input must be 1x1xN");
  expected_.reset({1, 1, idim[2]});
  this->output_.reset({1, 1, idim[2]});
}

void RegressionNode::forward() {
  assert(expected_.dims() == input_->dims() && "invalid expected dims");
  auto idim = input_->dims();
  auto &inputBuffer = input_->getOutput();

  auto outW = this->output_.weight_.getHandle();
  auto inW = inputBuffer.weight_.getHandle();

  for (unsigned z = 0; z < idim[2]; z++) {
    outW.at({0, 0, z}) = inW.at({0, 0, z});
  }
}

void RegressionNode::backward() {
  assert(expected_.dims() == input_->dims() && "invalid expected dims");

  auto idim = input_->dims();
  auto inW = input_->getOutput().weight_.getHandle();
  auto inG = input_->getOutput().gradient_.getHandle();

  auto e = expected_.getHandle();

  for (unsigned z = 0; z < idim[2]; z++) {
    FloatTy dy = (inW.at({0, 0, z}) - e.at({0, 0, z}));
    inG.at({0, 0, z}) = dy;
  }
}

MaxNode::MaxNode(Network *N, TrainableNode *input)
    : TrainableNode(N), input_(input) {
  assert(input && input_->size() && "Invalid input");
  auto idim = input_->dims();
  this->output_.reset(idim);
}

void MaxNode::forward() {
  auto idim = input_->dims();
  auto &inputBuffer = input_->getOutput();
  auto outW = this->output_.weight_.getHandle();
  auto inW = inputBuffer.weight_.getHandle();

  for (unsigned x = 0; x < idim[0]; x++) {
    for (unsigned y = 0; y < idim[1]; y++) {
      for (unsigned z = 0; z < idim[2]; z++) {
        outW.at({x, y, z}) = inW.at({x, y, z});
      }
    }
  }
}

void MaxNode::backward() {
  auto idim = input_->dims();
  auto &inputBuffer = input_->getOutput();
  auto inW = inputBuffer.weight_.getHandle();
  auto inG = inputBuffer.gradient_.getHandle();

  for (unsigned x = 0; x < idim[0]; x++) {
    for (unsigned y = 0; y < idim[1]; y++) {
      for (unsigned z = 0; z < idim[2]; z++) {
        FloatTy dy = inW.at({x, y, z});
        inG.at({x, y, z}) = dy > 0 ? -1 : 1;
      }
    }
  }
}

// Define the node visitor for all nodes in the graph that have a single
// incoming node.

#define DEFINE_CLASS_VISITOR(CLASS_NAME) \
void CLASS_NAME::visit(NodeVisitor *visitor) { \
visitor->pre(this); \
input_->visit(visitor); \
visitor->post(this); \
} \

DEFINE_CLASS_VISITOR(ConvNode)
DEFINE_CLASS_VISITOR(MaxPoolNode)
DEFINE_CLASS_VISITOR(FullyConnectedNode)
DEFINE_CLASS_VISITOR(RELUNode)
DEFINE_CLASS_VISITOR(SigmoidNode)
DEFINE_CLASS_VISITOR(SoftMaxNode)
DEFINE_CLASS_VISITOR(RegressionNode)
DEFINE_CLASS_VISITOR(MaxNode)

void ArrayNode::visit(NodeVisitor *visitor) {
  visitor->pre(this);
  visitor->post(this);
}
