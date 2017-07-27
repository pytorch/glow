#include "noether/Nodes.h"
#include "noether/Network.h"
#include "noether/Tensor.h"

using namespace noether;

/// \returns a handle to the gradient tensor \p XX.
#define GRADHANDLE(XX) \
  Handle<FloatTy>((network_->getGradientTensor(ctx, XX )))


ConvNode::ConvNode(Network *N, TrainableNode *input, size_t outDepth,
                   size_t filterSize, size_t stride, size_t pad)
    : TrainableNode(N), input_(input), filterSize_(filterSize), stride_(stride),
      pad_(pad) {
  assert(input && input_->size() && "Invalid input");
  auto idim = input_->dims();
  assert(idim[0] > filterSize && idim[1] > filterSize &&
         "buffer too small for selected stride");

  size_t outsx = ((idim[0] + pad_ * 2 - filterSize) / stride + 1);
  size_t outsy = ((idim[1] + pad_ * 2 - filterSize) / stride + 1);

  output_.reset(ElemKind::FloatTy, {outsx, outsy, outDepth});
  bias_.reset(ElemKind::FloatTy, {1, 1, outDepth});

  // RELUs like small positive bias to get gradients early in the training
  // process, otherwise the RELU units may never turn on and turn into a
  // "dead RELU".
  auto biasWeights = bias_.getHandle<FloatTy>();
  for (size_t i = 0; i < outDepth; i++) {
    biasWeights.at({0, 0, i}) = 0.1;
  }

  filters_.reset(ElemKind::FloatTy, {outDepth, filterSize, filterSize, idim[2]});
  size_t fanIn = filterSize * filterSize * idim[2];
  filters_.getHandle<FloatTy>().randomize(fanIn);

  N->allocateGradientTensor(&filters_);
  N->allocateGradientTensor(&bias_);
}

void ConvNode::forward(Context *ctx) {
  auto odim = output_.dims();
  auto idim = input_->dims();

  auto inW = input_->getOutput().getHandle<FloatTy>();
  auto outW = output_.getHandle<FloatTy>();
  auto biasW = bias_.getHandle<FloatTy>();
  auto filterW = filters_.getHandle<FloatTy>();

  // For each layer in the output tensor:
  for (size_t d = 0; d < odim[2]; d++) {

    // For each convolution 'jump' in the input tensor:
    ssize_t y = -ssize_t(pad_);
    for (size_t ay = 0; ay < odim[1]; y += stride_, ay++) {
      ssize_t x = -ssize_t(pad_);
      for (size_t ax = 0; ax < odim[0]; x += stride_, ax++) {

        // For each element in the convolution-filter:
        FloatTy sum = 0;
        for (size_t fy = 0; fy < filterSize_; fy++) {
          for (size_t fx = 0; fx < filterSize_; fx++) {
            ssize_t ox = x + fx;
            ssize_t oy = y + fy;

            // Ignore index access below zero (this is due to padding).
            if (ox < 0 || oy < 0)
              continue;

            if (outW.isInBounds({(size_t)ox, (size_t)oy, 0})) {
              for (size_t fd = 0; fd < idim[2]; fd++) {
                sum += filterW.at({d, fx, fy, fd}) *
                       inW.at({(size_t)ox, (size_t)oy, fd});
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

void ConvNode::backward(Context *ctx) {
  auto odim = output_.dims();
  auto idim = input_->dims();
  auto &inputWeights = input_->getOutput();
  auto inW = inputWeights.getHandle<FloatTy>();
  auto inG = GRADHANDLE(&inputWeights);
  auto outG = GRADHANDLE(&output_);
  auto biasG = GRADHANDLE(&bias_);
  auto filterG = GRADHANDLE(&filters_);
  auto filterW = filters_.getHandle<FloatTy>();

  // Zero the gradient of the input.
  inG.clear();

  // Compute the gradient. For each layer in the output tensor:
  for (size_t d = 0; d < odim[2]; d++) {

    // For each convolution 'jump' in the input tensor:
    ssize_t y = -ssize_t(pad_);
    for (size_t ay = 0; ay < ssize_t(odim[1]); y += stride_, ay++) {
      ssize_t x = -ssize_t(pad_);
      for (size_t ax = 0; ax < ssize_t(odim[0]); x += stride_, ax++) {

        FloatTy chainGrad = outG.at({ax, ay, d});

        // For each element in the convolution-filter:
        for (size_t fy = 0; fy < filterSize_; fy++) {
          for (size_t fx = 0; fx < filterSize_; fx++) {
            ssize_t ox = x + fx;
            ssize_t oy = y + fy;

            // Ignore index access below zero (this is due to padding).
            if (ox < 0 || oy < 0)
              continue;

            if (outG.isInBounds({(size_t)ox, (size_t)oy, 0u})) {
              for (size_t fd = 0; fd < idim[2]; fd++) {
                filterG.at({d, fx, fy, fd}) +=
                    inW.at({(size_t)ox, (size_t)oy, fd}) * chainGrad;
                inG.at({(size_t)ox, (size_t)oy, fd}) +=
                    filterW.at({d, fx, fy, fd}) * chainGrad;
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

  size_t outsx = ((idim[0] + pad_ * 2 - filterSize) / stride + 1);
  size_t outsy = ((idim[1] + pad_ * 2 - filterSize) / stride + 1);

  output_.reset(ElemKind::FloatTy, {outsx, outsy, idim[2]});

  // Resize the arrays that store the x and y coordinates of the incoming
  // gradient.
  srcX_.reset(ElemKind::IndexTy, {outsx, outsy, idim[2]});
  srcY_.reset(ElemKind::IndexTy, {outsx, outsy, idim[2]});
}

void MaxPoolNode::forward(Context *ctx) {
  auto odim = output_.dims();
  auto idim = input_->dims();
  auto inW = input_->getOutput().getHandle<FloatTy>();
  auto outW = output_.getHandle<FloatTy>();

  auto SX = srcX_.getHandle<size_t>();
  auto SY = srcY_.getHandle<size_t>();

  // For each layer in the output tensor:
  for (size_t z = 0; z < idim[2]; z++) {
    // For each convolution 'jump' in the input tensor:
    ssize_t y = -ssize_t(pad_);
    for (size_t ay = 0; ay < odim[1]; y += stride_, ay++) {
      ssize_t x = -ssize_t(pad_);
      for (size_t ax = 0; ax < odim[0]; x += stride_, ax++) {
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

            if (inW.isInBounds({(size_t)ox, (size_t)oy, z})) {
              FloatTy val = inW.at({(size_t)ox, (size_t)oy, z});

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

void MaxPoolNode::backward(Context *ctx) {
  auto odim = output_.dims();
  auto inG = GRADHANDLE(&input_->getOutput());
  auto outG = GRADHANDLE(&output_);

  auto SX = srcX_.getHandle<size_t>();
  auto SY = srcY_.getHandle<size_t>();

  // Zero the gradient of the input.
  inG.clear();

  // Compute the gradient. For each layer in the output tensor:
  for (size_t z = 0; z < odim[2]; z++) {

    // For each convolution 'jump' in the input tensor:
    for (size_t ay = 0; ay < odim[1]; ay++) {
      for (size_t ax = 0; ax < odim[0]; ax++) {

        FloatTy chainGrad = outG.at({(size_t)ax, (size_t)ay, z});

        size_t maxX = SX.at({(size_t)ax, (size_t)ay, z});
        size_t maxY = SY.at({(size_t)ax, (size_t)ay, z});

        inG.at({maxX, maxY, z}) += chainGrad;
      }
    }
  }
}

FullyConnectedNode::FullyConnectedNode(Network *N, TrainableNode *input,
                                       size_t outDepth)
    : TrainableNode(N), input_(input) {
  assert(input && input_->size() && "Invalid input");

  output_.reset(ElemKind::FloatTy, {outDepth});
  bias_.reset(ElemKind::FloatTy, {outDepth});

  // RELUs like small positive bias to get gradients early in the training
  // process, otherwise the RELU units may never turn on and turn into a
  // "dead RELU".
  auto biasW = bias_.getHandle<FloatTy>();
  for (size_t i = 0; i < outDepth; i++) {
    biasW.at({i}) = 0.1;
  }

  filters_.reset(ElemKind::FloatTy, {outDepth, input_->size()});
  filters_.getHandle<FloatTy>().randomize(input_->size());

  N->allocateGradientTensor(&filters_);
  N->allocateGradientTensor(&bias_);
}

void FullyConnectedNode::forward(Context *ctx) {
  auto &inputBuffer = input_->getOutput();
  auto odim = output_.dims();
  auto inW = input_->getOutput().getHandle<FloatTy>();
  auto biasW = bias_.getHandle<FloatTy>();
  auto outW = output_.getHandle<FloatTy>();
  auto currFilterW = filters_.getHandle<FloatTy>();

  size_t inputSize = inputBuffer.size();
  for (size_t i = 0; i < odim[0]; i++) {
    FloatTy sum = 0;
    for (size_t j = 0; j < inputSize; j++) {
      sum += inW.raw(j) * currFilterW.at({i, j});
    }

    sum += biasW.at({i});
    outW.at({i}) = sum;
  }
}

void FullyConnectedNode::backward(Context *ctx) {
  auto odim = output_.dims();
  auto &inputBuffer = input_->getOutput();
  auto outG = GRADHANDLE(&output_);
  auto inG = GRADHANDLE(&inputBuffer);
  auto inW = inputBuffer.getHandle<FloatTy>();
  auto biasG = GRADHANDLE(&bias_);

  // Zero the gradient of the input.
  inG.clear();

  auto filterG = GRADHANDLE(&filters_);
  auto filterW = filters_.getHandle<FloatTy>();

  size_t inSize = inG.size();
  // Compute the gradient:
  for (size_t i = 0; i < odim[0]; i++) {
    FloatTy chainGrad = outG.at({i});

    for (size_t j = 0, e = inSize; j < e; j++) {
      // Input gradient:
      inG.raw(j) += filterW.at({i, j}) * chainGrad;
      // Param gradient:
      filterG.at({i, j}) += inW.raw(j) * chainGrad;
    }

    biasG.at({i}) += chainGrad;
  }
}

RELUNode::RELUNode(Network *N, TrainableNode *input)
    : TrainableNode(N), input_(input) {
  assert(input && input_->size() && "Invalid input");
  output_.reset(ElemKind::FloatTy, input_->dims());
}

void RELUNode::forward(Context *ctx) {
  auto &inputBuffer = input_->getOutput();
  auto outW = output_.getHandle<FloatTy>();
  auto inW = inputBuffer.getHandle<FloatTy>();

  for (size_t i = 0, e = inW.size(); i < e; i++) {
    FloatTy val = inW.raw(i);
    outW.raw(i) = val < 0 ? 0 : val;
  }
}

void RELUNode::backward(Context *ctx) {
  auto &inputBuffer = input_->getOutput();

  auto outW = output_.getHandle<FloatTy>();
  auto outDW = GRADHANDLE(&output_);
  auto inDW = GRADHANDLE(&inputBuffer);

  for (size_t i = 0, e = outW.size(); i < e; i++) {
    FloatTy val = outW.raw(i);
    inDW.raw(i) = (val <= 0 ? 0 : outDW.raw(i));
  }
}

SigmoidNode::SigmoidNode(Network *N, TrainableNode *input)
    : TrainableNode(N), input_(input) {
  assert(input && input_->size() && "Invalid input");
  output_.reset(ElemKind::FloatTy, input_->dims());
}

void SigmoidNode::forward(Context *ctx) {
  auto &inputBuffer = input_->getOutput();

  auto outW = output_.getHandle<FloatTy>();
  auto inW = inputBuffer.getHandle<FloatTy>();

  for (size_t i = 0, e = outW.size(); i < e; i++) {
    FloatTy val = inW.raw(i);
    outW.raw(i) = 1 / (1 + std::exp(-val));
  }
}

void SigmoidNode::backward(Context *ctx) {
  auto &inputBuffer = input_->getOutput();

  auto outW = output_.getHandle<FloatTy>();
  auto outDW = GRADHANDLE(&output_);
  auto inDW = GRADHANDLE(&inputBuffer);

  for (size_t i = 0, e = outW.size(); i < e; i++) {
    FloatTy val = outW.raw(i);
    inDW.raw(i) = val * (1 - val) * outDW.raw(i);
  }
}

SoftMaxNode::SoftMaxNode(Network *N, TrainableNode *input)
    : TrainableNode(N), input_(input), selected_(0) {
  assert(input && input_->size() && "Invalid input");
  auto idim = input_->dims();
  assert(idim.size() == 1 && "Softmax input must be a simple vector.");
  output_.reset(ElemKind::FloatTy, {idim[0]});
  e_.reset(ElemKind::FloatTy, {idim[0]});
}

void SoftMaxNode::forward(Context *ctx) {
  auto idim = input_->dims();
  auto outW = output_.getHandle<FloatTy>();
  auto inW = input_->getOutput().getHandle<FloatTy>();

  FloatTy max = inW.at({0});

  // Find Max.
  for (size_t i = 0; i < idim[0]; i++) {
    max = std::max(max, inW.at({i}));
  }

  FloatTy sum = 0;

  auto EH = e_.getHandle<FloatTy>();
  // Compute exp.
  for (size_t i = 0; i < idim[0]; i++) {
    FloatTy e = std::exp(inW.at({i}) - max);
    sum += e;
    EH.at({i}) = e;
  }

  // Normalize the output.
  for (size_t i = 0; i < idim[0]; i++) {
    EH.at({i}) /= sum;
    outW.at({i}) = EH.at({i});
  }
}

void SoftMaxNode::backward(Context *ctx) {
  auto idim = input_->dims();
  auto inDW = GRADHANDLE(&input_->getOutput());
  auto ex = e_.getHandle<FloatTy>();

  for (size_t i = 0; i < idim[0]; i++) {
    FloatTy indicator = (selected_ == i ? 1 : 0);
    FloatTy mul = -(indicator - ex.at({i}));
    inDW.at({i}) = mul;
  }
}

size_t SoftMaxNode::maxArg() {
  auto idim = input_->dims();
  (void)idim;
  assert(idim.size() && "Invalid softmax shape!");

  auto outW = output_.getHandle<FloatTy>();
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
  assert(idim.size() == 1 && selected < idim[0] && "Invalid selection");
  selected_ = selected;
}

RegressionNode::RegressionNode(Network *N, TrainableNode *input)
    : TrainableNode(N), input_(input) {
  assert(input && input_->size() && "Invalid input");
  auto idim = input_->dims();
  assert(idim.size() == 1 && "input must be a simple vector.");
  expected_.reset(ElemKind::FloatTy, {idim[0]});
  output_.reset(ElemKind::FloatTy, {idim[0]});
}

void RegressionNode::forward(Context *ctx) {
  assert(expected_.dims() == input_->dims() && "invalid expected dims");
  auto idim = input_->dims();
  auto &inputBuffer = input_->getOutput();

  auto outW = output_.getHandle<FloatTy>();
  auto inW = inputBuffer.getHandle<FloatTy>();

  for (size_t i = 0; i < idim[0]; i++) {
    outW.at({i}) = inW.at({i});
  }
}

void RegressionNode::backward(Context *ctx) {
  assert(expected_.dims() == input_->dims() && "invalid expected dims");

  auto idim = input_->dims();
  auto inW = input_->getOutput().getHandle<FloatTy>();
  auto inG = GRADHANDLE(&input_->getOutput());

  auto e = expected_.getHandle<FloatTy>();

  for (size_t i = 0; i < idim[0]; i++) {
    FloatTy dy = inW.at({i}) - e.at({i});
    inG.at({i}) = dy;
  }
}

MaxNode::MaxNode(Network *N, TrainableNode *input)
    : TrainableNode(N), input_(input) {
  assert(input && input_->size() && "Invalid input");
  auto idim = input_->dims();
  output_.reset(ElemKind::FloatTy, idim);
}

void MaxNode::forward(Context *ctx) {
  auto &inputBuffer = input_->getOutput();
  auto outW = output_.getHandle<FloatTy>();
  auto inW = inputBuffer.getHandle<FloatTy>();

  for (size_t i = 0, e = outW.size(); i < e; i++) {
    outW.raw(i) = inW.raw(i);
  }
}

void MaxNode::backward(Context *ctx) {
  auto &inputBuffer = input_->getOutput();
  auto inW = inputBuffer.getHandle<FloatTy>();
  auto inG = GRADHANDLE(&inputBuffer);

  for (size_t i = 0, e = inG.size(); i < e; i++) {
    FloatTy dy = inW.raw(i);
    inG.raw(i) = dy > 0 ? -1 : 1;
  }
}

// Define the node visitor for all nodes in the graph that have a single
// incoming node.

#define DEFINE_CLASS_VISITOR(CLASS_NAME)                                       \
  void CLASS_NAME::visit(NodeVisitor *visitor) {                               \
    visitor->pre(this);                                                        \
    input_->visit(visitor);                                                    \
    visitor->post(this);                                                       \
  }

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
