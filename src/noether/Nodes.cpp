#include "noether/Nodes.h"
#include "noether/Network.h"
#include "noether/Tensor.h"

using namespace noether;


ConvNode::ConvNode(Network *N, NodeBase *input, size_t outDepth,
                   size_t filterSize, size_t stride, size_t pad)
    : NodeBase(), input_(input), filterSize_(filterSize), stride_(stride),
  pad_(pad), outDepth_(outDepth) {}

void ConvNode::init(Context *ctx) const {
  assert(input_ && input_->size(ctx) && "Invalid input");
  auto idim = input_->dims(ctx);
  assert(idim[0] > filterSize_ && idim[1] > filterSize_ &&
         "buffer too small for selected stride");

  size_t outsx = ((idim[0] + pad_ * 2 - filterSize_) / stride_ + 1);
  size_t outsy = ((idim[1] + pad_ * 2 - filterSize_) / stride_ + 1);

  ctx->allocateTrainable(&output_, false, {outsx, outsy, outDepth_});
  ctx->allocateTrainable(&bias_, true, {1, 1, outDepth_});
  ctx->allocateTrainable(&filters_, true, {outDepth_, filterSize_, filterSize_, idim[2]});

  // RELUs like small positive bias to get gradients early in the training
  // process, otherwise the RELU units may never turn on and turn into a
  // "dead RELU".
  auto biasWeights = ctx->getWeightHandle(&bias_);
  for (size_t i = 0; i < outDepth_; i++) {
    biasWeights.at({0, 0, i}) = 0.1;
  }

  size_t fanIn = filterSize_ * filterSize_ * idim[2];
  ctx->getWeightHandle(&filters_).randomize(fanIn);
}

void ConvNode::forward(Context *ctx) const {
  auto odim = getOutput(ctx)->dims();
  auto idim = input_->dims(ctx);

  auto inW = input_->getWeightHandle(ctx);
  auto outW = getWeightHandle(ctx);
  auto biasW = ctx->getWeightHandle(&bias_);
  auto filterW = ctx->getWeightHandle(&filters_);

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

void ConvNode::backward(Context *ctx) const {
  auto odim = getOutput(ctx)->dims();
  auto idim = input_->getOutput(ctx)->dims();
  auto inW = input_->getWeightHandle(ctx);
  auto inG = input_->getGradHandle(ctx);
  auto outG = getGradHandle(ctx);
  auto biasG = ctx->getGradHandle(&bias_);
  auto filterG = ctx->getGradHandle(&filters_);
  auto filterW = ctx->getWeightHandle(&filters_);

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

MaxPoolNode::MaxPoolNode(Network *N, NodeBase *input, size_t filterSize,
                         size_t stride, size_t pad)
    : NodeBase(), input_(input), filterSize_(filterSize), stride_(stride),
pad_(pad) { }

void MaxPoolNode::init(Context *ctx) const {
  assert(input_ && input_->size(ctx) && "Invalid input");
  auto idim = input_->dims(ctx);
  assert(idim[0] > filterSize_ && idim[0] > filterSize_ &&
         "buffer too small for selected stride");

  size_t outsx = ((idim[0] + pad_ * 2 - filterSize_) / stride_ + 1);
  size_t outsy = ((idim[1] + pad_ * 2 - filterSize_) / stride_ + 1);

  ctx->allocateTrainable(&output_, false, {outsx, outsy, idim[2]});

  // Resize the arrays that store the x and y coordinates of the incoming
  // gradient.
  ctx->allocateTensor(&srcX_, ElemKind::IndexTy, {outsx, outsy, idim[2]});
  ctx->allocateTensor(&srcY_, ElemKind::IndexTy, {outsx, outsy, idim[2]});
}

void MaxPoolNode::forward(Context *ctx) const {
  auto odim = dims(ctx);
  auto idim = input_->dims(ctx);
  auto inW = input_->getWeightHandle(ctx);
  auto outW = getWeightHandle(ctx);

  auto SX = ctx->getTensor(&srcX_)->getHandle<size_t>();
  auto SY = ctx->getTensor(&srcY_)->getHandle<size_t>();

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

void MaxPoolNode::backward(Context *ctx) const {
  auto odim = dims(ctx);
  auto inG = input_->getGradHandle(ctx);
  auto outG = getGradHandle(ctx);

  auto SX = ctx->getTensor(&srcX_)->getHandle<size_t>();
  auto SY = ctx->getTensor(&srcX_)->getHandle<size_t>();

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

FullyConnectedNode::FullyConnectedNode(Network *N, NodeBase *input,
                                       size_t outDepth)
: NodeBase(), input_(input), outDepth_(outDepth) {}

void FullyConnectedNode::init(Context *ctx) const {
  assert(input_ && input_->size(ctx) && "Invalid input");

  ctx->allocateTrainable(&output_, false, {outDepth_});
  ctx->allocateTrainable(&bias_, true, {outDepth_});

  // RELUs like small positive bias to get gradients early in the training
  // process, otherwise the RELU units may never turn on and turn into a
  // "dead RELU".
  auto biasW = ctx->getWeightHandle(&bias_);
  for (size_t i = 0; i < outDepth_; i++) {
    biasW.at({i}) = 0.1;
  }

  ctx->allocateTrainable(&filters_, true, {outDepth_, input_->size(ctx)});
  ctx->getWeightHandle(&filters_).randomize(input_->size(ctx));
}

void FullyConnectedNode::forward(Context *ctx) const {
  auto odim = dims(ctx);
  auto inW = input_->getWeightHandle(ctx);
  auto biasW = ctx->getWeightHandle(&bias_);
  auto outW = getWeightHandle(ctx);
  auto currFilterW = ctx->getWeightHandle(&filters_);

  size_t inputSize = inW.size();
  for (size_t i = 0; i < odim[0]; i++) {
    FloatTy sum = 0;
    for (size_t j = 0; j < inputSize; j++) {
      sum += inW.raw(j) * currFilterW.at({i, j});
    }

    sum += biasW.at({i});
    outW.at({i}) = sum;
  }
}

void FullyConnectedNode::backward(Context *ctx) const {
  auto odim = dims(ctx);
  auto outG = getGradHandle(ctx);
  auto inG = input_->getGradHandle(ctx);
  auto inW = input_->getWeightHandle(ctx);
  auto biasG = ctx->getGradHandle(&bias_);

  // Zero the gradient of the input.
  inG.clear();

  auto filterG = ctx->getGradHandle(&filters_);
  auto filterW = ctx->getWeightHandle(&filters_);

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

RELUNode::RELUNode(Network *N, NodeBase *input)
    : NodeBase(), input_(input) {}

void RELUNode::init(Context *ctx) const {
  assert(input_ && input_->size(ctx) && "Invalid input");
  ctx->allocateTrainable(&output_, false, input_->dims(ctx));
}

void RELUNode::forward(Context *ctx) const {
  auto outW = getWeightHandle(ctx);
  auto inW = input_->getWeightHandle(ctx);

  for (size_t i = 0, e = inW.size(); i < e; i++) {
    FloatTy val = inW.raw(i);
    outW.raw(i) = val < 0 ? 0 : val;
  }
}

void RELUNode::backward(Context *ctx) const {
  auto outW = getWeightHandle(ctx);
  auto outG = getGradHandle(ctx);
  auto inG = input_->getGradHandle(ctx);

  for (size_t i = 0, e = outW.size(); i < e; i++) {
    FloatTy val = outW.raw(i);
    inG.raw(i) = (val <= 0 ? 0 : outG.raw(i));
  }
}

SigmoidNode::SigmoidNode(Network *N, NodeBase *input)
    : NodeBase(), input_(input) {}

void SigmoidNode::init(Context *ctx) const {
  assert(input_ && input_->size(ctx) && "Invalid input");
  ctx->allocateTrainable(&output_, false, input_->dims(ctx));
}

void SigmoidNode::forward(Context *ctx) const {
  auto outW = getWeightHandle(ctx);
  auto inW = input_->getWeightHandle(ctx);

  for (size_t i = 0, e = outW.size(); i < e; i++) {
    FloatTy val = inW.raw(i);
    outW.raw(i) = 1 / (1 + std::exp(-val));
  }
}

void SigmoidNode::backward(Context *ctx) const {
  auto outW = getWeightHandle(ctx);
  auto outG = getGradHandle(ctx);
  auto inG = input_->getGradHandle(ctx);


  for (size_t i = 0, e = outW.size(); i < e; i++) {
    FloatTy val = outW.raw(i);
    inG.raw(i) = val * (1 - val) * outG.raw(i);
  }
}

SoftMaxNode::SoftMaxNode(Network *N, NodeBase *input)
: NodeBase(), input_(input) {}

void SoftMaxNode::init(Context *ctx) const {
  assert(input_ && input_->size(ctx) && "Invalid input");
  auto idim = input_->dims(ctx);
  assert(idim.size() == 1 && "Softmax input must be a simple vector.");
  ctx->allocateTrainable(&output_, false, {idim[0]});


  ctx->allocateTensor(&e_, ElemKind::FloatTy, {idim[0]});
  ctx->allocateTensor(&selected_, ElemKind::IndexTy, {1});
}

void SoftMaxNode::forward(Context *ctx) const {
  auto outW = getWeightHandle(ctx);
  auto idim = input_->dims(ctx);
  auto inW = input_->getWeightHandle(ctx);

  FloatTy max = inW.at({0});

  // Find Max.
  for (size_t i = 0; i < idim[0]; i++) {
    max = std::max(max, inW.at({i}));
  }

  FloatTy sum = 0;

  auto EH = ctx->getTensor(&e_)->getHandle<FloatTy>();
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

void SoftMaxNode::backward(Context *ctx) const {
  auto idim = input_->dims(ctx);
  auto inG = input_->getGradHandle(ctx);
  auto EH = ctx->getTensor(&e_)->getHandle<FloatTy>();
  auto selectedH = ctx->getTensor(&selected_)->getHandle<size_t>();
  size_t selected = selectedH.at({0});

  for (size_t i = 0; i < idim[0]; i++) {
    FloatTy indicator = (selected == i ? 1 : 0);
    FloatTy mul = -(indicator - EH.at({i}));
    inG.at({i}) = mul;
  }
}

void SoftMaxNode::setSelected(Context *ctx, size_t selected) {
  auto selectedH = ctx->getTensor(&selected_)->getHandle<size_t>();
  selectedH.at({0}) = selected;
}

void SoftMaxNode::updateInputs(Context *ctx, Tensor *batch, size_t sampleIdx)  {
  auto dim = batch->dims();
  assert(dim.size() == 1 && "Invalid input shape");


  // Take the n'th slice in the input vector. The slice must be a scalar.
  size_t val = batch->getHandle<size_t>().at({sampleIdx % dim[0]});
  auto selectedH = ctx->getTensor(&selected_)->getHandle<size_t>();
  selectedH.at({0}) = val;
  assert(val < dims(ctx)[0] && "Invalid selected value");
}

void SoftMaxNode::updateInput(Context *ctx, Tensor *var) {
  auto dim = var->dims();
  (void)dim;
  assert(dim.size() == 1 && "Invalid input shape");

  size_t val = var->getHandle<size_t>().at({0});

  auto selectedH = ctx->getTensor(&selected_)->getHandle<size_t>();
  selectedH.at({0}) = val;

  assert(val < dims(ctx)[0] && "Invalid selected value");
}


RegressionNode::RegressionNode(Network *N, NodeBase *input)
: NodeBase(), input_(input) {}

void RegressionNode::init(Context *ctx) const {
  assert(input_ && input_->size(ctx) && "Invalid input");
  auto idim = input_->dims(ctx);
  assert(idim.size() == 1 && "input must be a simple vector.");
  ctx->allocateTrainable(&output_, false, {idim[0]});
  ctx->allocateTensor(&expected_, ElemKind::FloatTy, {idim[0]});
}

void RegressionNode::forward(Context *ctx) const {
  assert(dims(ctx) == input_->dims(ctx) && "invalid expected dims");
  auto idim = input_->dims(ctx);
  auto outW = getWeightHandle(ctx);
  auto inW = input_->getWeightHandle(ctx);

  for (size_t i = 0; i < idim[0]; i++) {
    outW.at({i}) = inW.at({i});
  }
}

void RegressionNode::backward(Context *ctx) const {
  assert(dims(ctx) == input_->dims(ctx) && "invalid expected dims");
  auto idim = input_->dims(ctx);
  auto inW = input_->getWeightHandle(ctx);
  auto inG = input_->getGradHandle(ctx);

  auto e = ctx->getTensor(&expected_)->getHandle<FloatTy>();

  for (size_t i = 0; i < idim[0]; i++) {
    FloatTy dy = inW.at({i}) - e.at({i});
    inG.at({i}) = dy;
  }
}

void RegressionNode::updateInputs(Context *ctx, Tensor *batch, size_t sampleIdx)  {
  auto dim = batch->dims();
  auto *E = ctx->getTensor(&expected_);

  assert(E->getElementType() == batch->getElementType() &&
         "invalid input type");
  assert(dims(ctx) == dim.drop_front() && "Invalid batch size");

  // Extract the n'th slice. The slice must be a tensor.
  *E = batch->getHandle<FloatTy>().extractSlice(sampleIdx % dim[0]);
}

void RegressionNode::updateInput(Context *ctx, Tensor *var)  {
  auto *E = ctx->getTensor(&expected_);

  assert(E->dims() == var->dims() && "Invalid input size");
  assert(E->getElementType() == var->getElementType() &&
         "invalid input type");
  *E = var->clone();
}

MaxNode::MaxNode(Network *N, NodeBase *input)
    : NodeBase(), input_(input) {}

void MaxNode::init(Context *ctx) const {
  assert(input_ && input_->size(ctx) && "Invalid input");
  auto idim = input_->dims(ctx);
  ctx->allocateTrainable(&output_, false, idim);
}

void MaxNode::forward(Context *ctx) const {
  auto inW = input_->getWeightHandle(ctx);
  auto outW = getWeightHandle(ctx);

  for (size_t i = 0, e = outW.size(); i < e; i++) {
    outW.raw(i) = inW.raw(i);
  }
}

void MaxNode::backward(Context *ctx) const {
  auto inW = input_->getWeightHandle(ctx);
  auto inG = input_->getGradHandle(ctx);

  for (size_t i = 0, e = inG.size(); i < e; i++) {
    FloatTy dy = inW.raw(i);
    inG.raw(i) = dy > 0 ? -1 : 1;
  }
}

ArrayNode::ArrayNode(Network *N, ArrayRef<size_t> dims) : NodeBase(),
dims_(dims.begin(), dims.end()) {}

void ArrayNode::init(Context *ctx) const {
  ctx->allocateTrainable(&output_, false, dims_);
}

void ArrayNode::updateInputs(Context *ctx, Tensor *batch, size_t sampleIdx)  {
  auto dim = batch->dims();
  assert(dims(ctx) == dim.drop_front() && "Invalid batch size");
  /// Extract the n'th slice, that must be a tensor.
  size_t slc = sampleIdx % dim[0];
  getOutput(ctx)->weights_ = batch->getHandle<FloatTy>().extractSlice(slc);
}

void ArrayNode::updateInput(Context *ctx, Tensor *var)  {
  auto &w = getOutput(ctx)->weights_;
  (void)w;
  assert(w.dims() == var->dims() && "Invalid input size");
  assert(w.getElementType() == var->getElementType() && "invalid input type");
  getOutput(ctx)->weights_ = var->clone();
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
