#include "glow/Network/Nodes.h"
#include "glow/Network/Network.h"
#include "glow/Network/Tensor.h"
#include "glow/Support/Support.h"

using namespace glow;

/// Colllapse a tensor shape into two sizes: the first dimension and the size
/// of the rest of the dimensions.
/// For example, [7, 3, 4, 2] -> [7, 24]
static std::pair<size_t, size_t> flattenCdr(ArrayRef<size_t> dims) {
  assert(dims.size() > 1);
  size_t first = dims[0];
  size_t rest = dims[1];
  for (size_t i = 2; i < dims.size(); i++) {
    rest *= dims[i];
  }

  return {first, rest};
}

ConvNode::ConvNode(Network *N, NodeBase *input, size_t outDepth,
                   size_t filterSize, size_t stride, size_t pad)
    : input_(input), filterSize_(filterSize), stride_(stride), pad_(pad),
      outDepth_(outDepth) {}

void ConvNode::init(Context *ctx) const {
  assert(input_ && input_->size(ctx) && "Invalid input");
  ShapeNHWC idim = input_->dims(ctx);
  assert(idim.h >= filterSize_ && idim.w >= filterSize_ &&
         "buffer too small for selected stride");

  size_t outsx = ((idim.h + pad_ * 2 - filterSize_) / stride_ + 1);
  size_t outsy = ((idim.w + pad_ * 2 - filterSize_) / stride_ + 1);

  ctx->allocateTensor(&outputWeight_, ElemKind::FloatTy,
                      {idim.n, outsx, outsy, outDepth_});
  ctx->allocateTensor(&outputGrad_, ElemKind::FloatTy,
                      {idim.n, outsx, outsy, outDepth_});

  std::vector<size_t> biasDim = {outDepth_};
  ctx->allocateTensor(&biasW_, ElemKind::FloatTy, biasDim,
                      Context::ShareKind::kSharedTensor);
  ctx->allocateTensor(&biasG_, ElemKind::FloatTy, biasDim);

  std::vector<size_t> ftlrDim = {outDepth_, filterSize_, filterSize_, idim.c};
  ctx->allocateTensor(&filtersW_, ElemKind::FloatTy, ftlrDim,
                      Context::ShareKind::kSharedTensor);
  ctx->allocateTensor(&filtersG_, ElemKind::FloatTy, ftlrDim);

  // RELUs like small positive bias to get gradients early in the training
  // process, otherwise the RELU units may never turn on and turn into a
  // "dead RELU".
  if (ctx->hasTensor(&biasW_)) {
    auto biasWeights = ctx->getHandle(&biasW_);
    biasWeights.clear(0.1);
  }
  if (ctx->hasTensor(&filtersW_)) {
    size_t fanIn = filterSize_ * filterSize_ * idim.c;
    ctx->getHandle(&filtersW_).randomize(fanIn);
  }

  ctx->addTensorPair({&filtersW_, &filtersG_});
  ctx->addTensorPair({&biasW_, &biasG_});
}

std::string ConvNode::getDebugRepr(Context *ctx) const {
  DescriptionBuilder db(getName());
  db.addDim("input", input_->dims(ctx));
  db.addDim("output", getOutputWeight(ctx)->dims());
  db.addDim("filter", ctx->getHandle(&filtersW_).dims());
  db.addDim("bias", ctx->getHandle(&biasW_).dims());
  db.addParam("filter size", filterSize_);
  db.addParam("stride", stride_);
  db.addParam("pad", pad_);
  db.addParam("depth", outDepth_);
  return db;
}

void ConvNode::forward(Context *ctx, PassKind kind) const {
  ShapeNHWC odim = dims(ctx);
  ShapeNHWC idim = input_->dims(ctx);

  auto inW = input_->getWeightHandle(ctx);
  auto outW = getWeightHandle(ctx);
  auto biasW = ctx->getHandle(&biasW_);
  auto filterW = ctx->getHandle(&filtersW_);

  // For each input in the batch:
  for (size_t n = 0; n < idim.n; n++) {

    // For each layer in the output tensor:
    for (size_t d = 0; d < odim.c; d++) {

      // For each convolution 'jump' in the input tensor:
      ssize_t y = -ssize_t(pad_);
      for (size_t ay = 0; ay < odim.w; y += stride_, ay++) {
        ssize_t x = -ssize_t(pad_);
        for (size_t ax = 0; ax < odim.h; x += stride_, ax++) {

          // For each element in the convolution-filter:
          FloatTy sum = 0;
          for (size_t fy = 0; fy < filterSize_; fy++) {
            for (size_t fx = 0; fx < filterSize_; fx++) {
              ssize_t ox = x + fx;
              ssize_t oy = y + fy;

              // Ignore index access below zero (this is due to padding).
              if (ox < 0 || oy < 0 || ox >= odim.h || oy >= odim.w)
                continue;

              for (size_t fd = 0; fd < idim.c; fd++) {
                sum += filterW.at({d, fx, fy, fd}) *
                       inW.at({n, (size_t)ox, (size_t)oy, fd});
              }
            }
          }

          sum += biasW.at({d});
          outW.at({n, ax, ay, d}) = sum;
        } // H
      }   // W
    }     // C
  }       // N
}

void ConvNode::backward(Context *ctx) const {
  ShapeNHWC odim = dims(ctx);
  ShapeNHWC idim = input_->dims(ctx);
  auto inW = input_->getWeightHandle(ctx);
  auto inG = input_->getGradHandle(ctx);
  auto outG = getGradHandle(ctx);
  auto biasG = ctx->getHandle(&biasG_);
  auto filterG = ctx->getHandle(&filtersG_);
  auto filterW = ctx->getHandle(&filtersW_);

  // For each input in the batch:
  for (size_t n = 0; n < odim.n; n++) {

    // Compute the gradient. For each layer in the output tensor:
    for (size_t d = 0; d < odim.c; d++) {

      // For each convolution 'jump' in the input tensor:
      ssize_t y = -ssize_t(pad_);
      for (size_t ay = 0; ay < ssize_t(odim.w); y += stride_, ay++) {
        ssize_t x = -ssize_t(pad_);
        for (size_t ax = 0; ax < ssize_t(odim.h); x += stride_, ax++) {

          FloatTy chainGrad = outG.at({n, ax, ay, d});

          // For each element in the convolution-filter:
          for (size_t fy = 0; fy < filterSize_; fy++) {
            for (size_t fx = 0; fx < filterSize_; fx++) {
              ssize_t ox = x + fx;
              ssize_t oy = y + fy;

              // Ignore index access below zero (this is due to padding).
              if (ox < 0 || oy < 0 || ox >= odim.h || oy >= odim.w)
                continue;

              for (size_t fd = 0; fd < idim.c; fd++) {
                filterG.at({d, fx, fy, fd}) +=
                    inW.at({0u, (size_t)ox, (size_t)oy, fd}) * chainGrad;
                inG.at({n, (size_t)ox, (size_t)oy, fd}) +=
                    filterW.at({d, fx, fy, fd}) * chainGrad;
              }
            }
          }

          biasG.at({d}) += chainGrad;
        } // H
      }   // W
    }     // C
  }       // N
}

void ConvNode::updateWeights(Network *N_, Tensor *filter, Tensor *bias) {
  N_->updateTensor(&filtersW_, filter);
  N_->updateTensor(&biasW_, bias);
}

MaxPoolNode::MaxPoolNode(Network *N, NodeBase *input, OpKind kind,
                         size_t filterSize, size_t stride, size_t pad)
    : kind_(kind), input_(input), filterSize_(filterSize), stride_(stride),
      pad_(pad) {}

void MaxPoolNode::init(Context *ctx) const {
  assert(input_ && input_->size(ctx) && "Invalid input");
  ShapeNHWC idim = input_->dims(ctx);
  assert(idim.w >= filterSize_ && idim.h >= filterSize_ &&
         "buffer too small for selected stride");

  size_t outsx = ((idim.w + pad_ * 2 - filterSize_) / stride_ + 1);
  size_t outsy = ((idim.h + pad_ * 2 - filterSize_) / stride_ + 1);

  ctx->allocateTensor(&outputWeight_, ElemKind::FloatTy,
                      {idim.n, outsx, outsy, idim.c});
  ctx->allocateTensor(&outputGrad_, ElemKind::FloatTy,
                      {idim.n, outsx, outsy, idim.c});

  // Allocate cache arrays that store the x and y coordinates of the incoming
  // gradient for each max element.
  if (kind_ == OpKind::kMax) {
    ctx->allocateTensor(&srcX_, ElemKind::IndexTy,
                        {idim.n, outsx, outsy, idim.c});
    ctx->allocateTensor(&srcY_, ElemKind::IndexTy,
                        {idim.n, outsx, outsy, idim.c});
  }
}

std::string MaxPoolNode::getDebugRepr(Context *ctx) const {
  DescriptionBuilder db(getName());
  db.addDim("input", input_->dims(ctx));
  db.addDim("output", getOutputWeight(ctx)->dims());

  db.addParam("filter size", filterSize_);
  db.addParam("stride", stride_);
  db.addParam("pad", pad_);
  db.addParam("operation", kind_ == OpKind::kMax ? "Max" : "Avg");
  return db;
}

void MaxPoolNode::forward(Context *ctx, PassKind kind) const {
  if (kind_ == OpKind::kMax) {
    return forwardMax(ctx);
  }

  return forwardAvg(ctx);
}

void MaxPoolNode::backward(Context *ctx) const {
  if (kind_ == OpKind::kMax) {
    return backwardMax(ctx);
  }

  return backwardAvg(ctx);
}

void MaxPoolNode::forwardMax(Context *ctx) const {
  ShapeNHWC odim = dims(ctx);
  ShapeNHWC idim = input_->dims(ctx);
  auto inW = input_->getWeightHandle(ctx);
  auto outW = getWeightHandle(ctx);

  auto SX = ctx->getTensor(&srcX_)->getHandle<size_t>();
  auto SY = ctx->getTensor(&srcY_)->getHandle<size_t>();

  // For each input in the batch:
  for (size_t n = 0; n < odim.n; n++) {

    // For each layer in the output tensor:
    for (size_t z = 0; z < idim.c; z++) {
      // For each convolution 'jump' in the input tensor:
      ssize_t y = -ssize_t(pad_);
      for (size_t ay = 0; ay < odim.w; y += stride_, ay++) {
        ssize_t x = -ssize_t(pad_);
        for (size_t ax = 0; ax < odim.h; x += stride_, ax++) {
          size_t maxX = x;
          size_t maxY = y;

          bool first = true;
          FloatTy max = 0;

          for (size_t fy = 0; fy < filterSize_; fy++) {
            for (size_t fx = 0; fx < filterSize_; fx++) {
              ssize_t ox = x + fx;
              ssize_t oy = y + fy;

              // Ignore index access below zero (this is due to padding).
              if (ox < 0 || oy < 0 || ox >= idim.h || oy >= idim.w)
                continue;

              FloatTy val = inW.at({n, (size_t)ox, (size_t)oy, z});

              if (first || (val >= max)) {
                first = false;
                max = val;
                maxX = ox;
                maxY = oy;
              }
            }
          }

          assert(!first && "Max value is uninitialized");
          SX.at({n, ax, ay, z}) = maxX;
          SY.at({n, ax, ay, z}) = maxY;
          outW.at({n, ax, ay, z}) = max;
        } // H
      }   // W
    }     // C
  }       // N
}

void MaxPoolNode::backwardMax(Context *ctx) const {
  ShapeNHWC odim = dims(ctx);
  auto inG = input_->getGradHandle(ctx);
  auto outG = getGradHandle(ctx);

  auto SX = ctx->getTensor(&srcX_)->getHandle<size_t>();
  auto SY = ctx->getTensor(&srcY_)->getHandle<size_t>();

  // For each input in the batch:
  for (size_t n = 0; n < odim.n; n++) {

    // Compute the gradient. For each layer in the output tensor:
    for (size_t z = 0; z < odim.c; z++) {

      // For each convolution 'jump' in the input tensor:
      for (size_t ay = 0; ay < odim.w; ay++) {
        for (size_t ax = 0; ax < odim.h; ax++) {

          FloatTy chainGrad = outG.at({n, (size_t)ax, (size_t)ay, z});

          size_t maxX = SX.at({n, (size_t)ax, (size_t)ay, z});
          size_t maxY = SY.at({n, (size_t)ax, (size_t)ay, z});

          inG.at({n, maxX, maxY, z}) += chainGrad;
        } // H
      }   // W
    }     // C
  }       // N
}

void MaxPoolNode::forwardAvg(Context *ctx) const {
  // Implement the avg pooling operation as defined here:
  // https://arxiv.org/abs/1312.4400

  ShapeNHWC odim = dims(ctx);
  ShapeNHWC idim = input_->dims(ctx);
  auto inW = input_->getWeightHandle(ctx);
  auto outW = getWeightHandle(ctx);

  FloatTy filterArea = filterSize_ * filterSize_;

  // For each input in the batch:
  for (size_t n = 0; n < odim.n; n++) {

    // For each layer in the output tensor:
    for (size_t z = 0; z < idim.c; z++) {
      // For each convolution 'jump' in the input tensor:
      ssize_t y = -ssize_t(pad_);
      for (size_t ay = 0; ay < odim.w; y += stride_, ay++) {
        ssize_t x = -ssize_t(pad_);
        for (size_t ax = 0; ax < odim.h; x += stride_, ax++) {
          FloatTy sum = 0;

          for (size_t fy = 0; fy < filterSize_; fy++) {
            for (size_t fx = 0; fx < filterSize_; fx++) {
              ssize_t ox = x + fx;
              ssize_t oy = y + fy;

              // Ignore index access below zero (this is due to padding).
              if (ox < 0 || oy < 0 || ox >= idim.h || oy >= idim.w)
                continue;

              sum += inW.at({n, (size_t)ox, (size_t)oy, z});
            }
          }
          outW.at({n, ax, ay, z}) = sum / filterArea;
        } // H
      }   // W
    }     // C

  } // N
}

void MaxPoolNode::backwardAvg(Context *ctx) const {
  ShapeNHWC odim = dims(ctx);
  ShapeNHWC idim = input_->dims(ctx);
  auto inG = input_->getGradHandle(ctx);
  auto outG = getGradHandle(ctx);
  FloatTy filterArea = filterSize_ * filterSize_;

  // For each input in the batch:
  for (size_t n = 0; n < odim.n; n++) {

    // For each layer in the output tensor:
    for (size_t z = 0; z < odim.c; z++) {
      // For each convolution 'jump' in the input tensor:
      ssize_t y = -ssize_t(pad_);
      for (size_t ay = 0; ay < odim.w; y += stride_, ay++) {
        ssize_t x = -ssize_t(pad_);
        for (size_t ax = 0; ax < odim.h; x += stride_, ax++) {
          FloatTy dy = outG.at({n, ax, ay, z}) / filterArea;

          for (size_t fy = 0; fy < filterSize_; fy++) {
            for (size_t fx = 0; fx < filterSize_; fx++) {
              ssize_t ox = x + fx;
              ssize_t oy = y + fy;

              // Ignore index access below zero (this is due to padding).
              if (ox < 0 || oy < 0 || ox >= idim.h || oy >= idim.w)
                continue;

              inG.at({n, (size_t)ox, (size_t)oy, z}) += dy;
            }
          }
        } // H
      }   // W
    }     // C
  }       // N
}

FullyConnectedNode::FullyConnectedNode(Network *N, NodeBase *input,
                                       size_t outDepth)
    : input_(input), outDepth_(outDepth) {}

void FullyConnectedNode::init(Context *ctx) const {
  assert(input_ && input_->size(ctx) && "Invalid input");

  auto idim = flattenCdr(input_->dims(ctx));

  ctx->allocateTensor(&outputWeight_, ElemKind::FloatTy,
                      {idim.first, outDepth_});
  ctx->allocateTensor(&outputGrad_, ElemKind::FloatTy, {idim.first, outDepth_});

  std::vector<size_t> biasDim = {outDepth_};
  ctx->allocateTensor(&biasW_, ElemKind::FloatTy, biasDim,
                      Context::ShareKind::kSharedTensor);
  ctx->allocateTensor(&biasG_, ElemKind::FloatTy, biasDim);

  std::vector<size_t> ftlrDim = {outDepth_, idim.second};
  ctx->allocateTensor(&filtersW_, ElemKind::FloatTy, ftlrDim,
                      Context::ShareKind::kSharedTensor);
  ctx->allocateTensor(&filtersG_, ElemKind::FloatTy, ftlrDim);

  // RELUs like small positive bias to get gradients early in the training
  // process, otherwise the RELU units may never turn on and turn into a
  // "dead RELU".
  if (ctx->hasTensor(&biasW_)) {
    auto biasWeights = ctx->getHandle(&biasW_);
    biasWeights.clear(0.1);
  }

  if (ctx->hasTensor(&filtersW_)) {
    size_t fanIn = idim.second;
    ctx->getHandle(&filtersW_).randomize(fanIn);
  }

  ctx->addTensorPair({&filtersW_, &filtersG_});
  ctx->addTensorPair({&biasW_, &biasG_});
}

std::string FullyConnectedNode::getDebugRepr(Context *ctx) const {
  DescriptionBuilder db(getName());
  db.addDim("input", input_->dims(ctx));
  db.addDim("output", getOutputWeight(ctx)->dims());
  db.addDim("filter", ctx->getHandle(&filtersW_).dims());
  db.addDim("bias", ctx->getHandle(&biasW_).dims());
  db.addParam("depth", outDepth_);
  return db;
}

void FullyConnectedNode::forward(Context *ctx, PassKind kind) const {
  auto odim = flattenCdr(dims(ctx));
  auto idim = flattenCdr(input_->dims(ctx));
  auto inW = input_->getWeightHandle(ctx);
  auto biasW = ctx->getHandle(&biasW_);
  auto outW = getWeightHandle(ctx);
  auto currFilterW = ctx->getHandle(&filtersW_);

  size_t inputSize = idim.second;

  for (size_t n = 0; n < odim.first; n++) {
    size_t base = inW.getElementPtr({n});

    for (size_t i = 0; i < odim.second; i++) {

      FloatTy sum = 0;
      for (size_t j = 0; j < inputSize; j++) {
        sum += inW.raw(base + j) * currFilterW.at({i, j});
      }

      sum += biasW.at({i});
      outW.at({n, i}) = sum;
    }
  } // N
}

void FullyConnectedNode::backward(Context *ctx) const {
  auto odim = flattenCdr(dims(ctx));
  auto idim = flattenCdr(input_->dims(ctx));
  auto outG = getGradHandle(ctx);
  auto inG = input_->getGradHandle(ctx);
  auto inW = input_->getWeightHandle(ctx);
  auto biasG = ctx->getHandle(&biasG_);

  auto filterG = ctx->getHandle(&filtersG_);
  auto filterW = ctx->getHandle(&filtersW_);

  size_t inSize = idim.second;

  for (size_t n = 0; n < odim.first; n++) {
    size_t base = inW.getElementPtr({n});

    // Compute the gradient:
    for (size_t i = 0; i < odim.second; i++) {
      FloatTy chainGrad = outG.at({n, i});

      for (size_t j = 0, e = inSize; j < e; j++) {
        // Input gradient:
        inG.raw(base + j) += filterW.at({i, j}) * chainGrad;
        // Param gradient:
        filterG.at({i, j}) += inW.raw(base + j) * chainGrad;
      }

      biasG.at({i}) += chainGrad;
    }
  } // N
}

RELUNode::RELUNode(Network *N, NodeBase *input) : input_(input) {}

void RELUNode::init(Context *ctx) const {
  assert(input_ && input_->size(ctx) && "Invalid input");

  ctx->allocateTensor(&outputWeight_, ElemKind::FloatTy, input_->dims(ctx));
  ctx->allocateTensor(&outputGrad_, ElemKind::FloatTy, input_->dims(ctx));
}

void RELUNode::forward(Context *ctx, PassKind kind) const {
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
    inG.raw(i) += (val <= 0 ? 0 : outG.raw(i));
  }
}

SigmoidNode::SigmoidNode(Network *N, NodeBase *input) : input_(input) {}

void SigmoidNode::init(Context *ctx) const {
  assert(input_ && input_->size(ctx) && "Invalid input");

  ctx->allocateTensor(&outputWeight_, ElemKind::FloatTy, input_->dims(ctx));
  ctx->allocateTensor(&outputGrad_, ElemKind::FloatTy, input_->dims(ctx));
}

void SigmoidNode::forward(Context *ctx, PassKind kind) const {
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
    inG.raw(i) += val * (1 - val) * outG.raw(i);
  }
}

SoftMaxNode::SoftMaxNode(Network *N, NodeBase *input, NodeBase *selected)
    : input_(input), selected_(selected) {}

void SoftMaxNode::init(Context *ctx) const {
  assert(input_ && input_->size(ctx) && "Invalid input");
  auto idim = input_->dims(ctx);
  auto sdim = selected_->dims(ctx);
  (void) sdim;
  assert(idim.size() == 2 && sdim.size() == 2 &&
         "Softmax input must be a 2D matrix.");
  assert(idim[0] == sdim[0] && "Batch size does not match");
  assert(sdim[1] == 1 && "The selected vector must contain a single index for "
         "each batch element");

  ctx->allocateTensor(&outputWeight_, ElemKind::FloatTy, idim);
  ctx->allocateTensor(&outputGrad_, ElemKind::FloatTy, idim);

  ctx->allocateTensor(&e_, ElemKind::FloatTy, idim);
}

std::string SoftMaxNode::getDebugRepr(Context *ctx) const {
  DescriptionBuilder db(getName());
  db.addDim("input", input_->dims(ctx));
  db.addDim("selected", selected_->dims(ctx));
  db.addDim("output", getOutputWeight(ctx)->dims());
  return db;
}

void SoftMaxNode::forward(Context *ctx, PassKind kind) const {
  auto outW = getWeightHandle(ctx);
  auto idim = input_->dims(ctx);
  auto inW = input_->getWeightHandle(ctx);

  auto EH = ctx->getTensor(&e_)->getHandle<FloatTy>();

  for (size_t n = 0; n < idim[0]; n++) {
    FloatTy max = inW.at({n, 0});

    // Find Max.
    for (size_t i = 0; i < idim[1]; i++) {
      max = std::max(max, inW.at({n, i}));
    }

    FloatTy sum = 0;

    // Compute exp.
    for (size_t i = 0; i < idim[1]; i++) {
      FloatTy e = std::exp(inW.at({n, i}) - max);
      sum += e;
      EH.at({n, i}) = e;
    }

    // Normalize the output.
    for (size_t i = 0; i < idim[1]; i++) {
      EH.at({n, i}) /= sum;
      outW.at({n, i}) = EH.at({n, i});
    }

  } // N
}

void SoftMaxNode::backward(Context *ctx) const {
  auto idim = input_->dims(ctx);
  auto inG = input_->getGradHandle(ctx);
  auto EH = ctx->getTensor(&e_)->getHandle<FloatTy>();
  auto selectedH = selected_->getOutputWeight(ctx)->getHandle<size_t>();

  // http://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
  // https://stats.stackexchange.com/questions/79454/softmax-layer-in-a-neural-network
  for (size_t n = 0; n < idim[0]; n++) {
    for (size_t i = 0; i < idim[1]; i++) {
      FloatTy delta = (selectedH.at({n, 0}) == i);
      FloatTy sigma = (EH.at({n, i}) - delta);
      inG.at({n, i}) += sigma;
    }
  }
}

RegressionNode::RegressionNode(Network *N, NodeBase *input, NodeBase *expected)
    : input_(input), expected_(expected) {}

void RegressionNode::init(Context *ctx) const {
  assert(input_ && input_->size(ctx) && "Invalid input");
  auto idim = input_->dims(ctx);
  assert(idim.size() == 2 && "Regression input must be a 2D matrix.");
  ctx->allocateTensor(&outputWeight_, ElemKind::FloatTy, idim);
  ctx->allocateTensor(&outputGrad_, ElemKind::FloatTy, idim);
}

std::string RegressionNode::getDebugRepr(Context *ctx) const {
  DescriptionBuilder db(getName());
  db.addDim("input", input_->dims(ctx));
  db.addDim("expected", expected_->dims(ctx));
  db.addDim("output", getOutputWeight(ctx)->dims());
  return db;
}

void RegressionNode::forward(Context *ctx, PassKind kind) const {
  assert(dims(ctx) == input_->dims(ctx) && "invalid expected dims");
  auto outW = getWeightHandle(ctx);
  auto inW = input_->getWeightHandle(ctx);

  for (size_t i = 0, e = size(ctx); i < e; i++) {
    outW.raw(i) = inW.raw(i);
  }
}

void RegressionNode::backward(Context *ctx) const {
  assert(dims(ctx) == input_->dims(ctx) && "invalid expected dims");
  auto idim = input_->dims(ctx);
  auto inW = input_->getWeightHandle(ctx);
  auto inG = input_->getGradHandle(ctx);
  assert(idim.size() == 2 && "Input is expected to be a vector per input");

  auto e = expected_->getOutputWeight(ctx)->getHandle<FloatTy>();

  // For each input in the batch:
  for (size_t n = 0; n < idim[0]; n++) {

    for (size_t i = 0; i < idim[1]; i++) {
      FloatTy dy = inW.at({n, i}) - e.at({n, i});
      inG.at({n, i}) += dy;
    }
  } // N
}

MaxNode::MaxNode(Network *N, NodeBase *input) : input_(input) {}

void MaxNode::init(Context *ctx) const {
  assert(input_ && input_->size(ctx) && "Invalid input");
  auto idim = input_->dims(ctx);
  ctx->allocateTensor(&outputWeight_, ElemKind::FloatTy, idim);
  ctx->allocateTensor(&outputGrad_, ElemKind::FloatTy, idim);
}

void MaxNode::forward(Context *ctx, PassKind kind) const {
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
    inG.raw(i) += dy > 0 ? -1 : 1;
  }
}

Variable::Variable(Network *N, ArrayRef<size_t> dims, ElemKind elemTy)
    : dims_(dims.begin(), dims.end()), elemTy_(elemTy) {}

void Variable::init(Context *ctx) const {
  ctx->allocateTensor(&outputWeight_, elemTy_, dims_);
  ctx->allocateTensor(&outputGrad_, elemTy_, dims_);
}

std::string Variable::getDebugRepr(Context *ctx) const {
  DescriptionBuilder db(getName());
  db.addDim("output", getOutputWeight(ctx)->dims());
  return db;
}

void Variable::updateInputs(Context *ctx, Tensor *batch, size_t sampleIdx) {
  auto dim = batch->dims();
  assert(dims(ctx).drop_front() == dim.drop_front() && "Invalid slice size");
  /// Extract the n'th slice, that must be a tensor.
  size_t slc = sampleIdx % dim[0];
  ctx->getTensor(&outputWeight_)->copyConsecutiveSlices(batch, slc);
}

ConcatNode::ConcatNode(Network *N, ArrayRef<NodeBase *> inputs,
                       unsigned dimension)
    : inputs_(inputs.begin(), inputs.end()), dimension_(dimension) {}

void ConcatNode::init(Context *ctx) const {
  assert(inputs_.size() > 1 && "invalid number of inputs");
  auto dims0 = inputs_[0]->dims(ctx);
  assert(dimension_ < dims0.size() && "Invalid concat dimension");

  size_t sizeNthDim = dims0[dimension_];
  for (int i = 1, e = inputs_.size(); i < e; i++) {
    auto dimsI = inputs_[i]->dims(ctx);

    // Count how many layers are in the nth dimension.
    sizeNthDim += dimsI[dimension_];

    // Validate that the rest of the dimensions are identical.
    assert(dims0.size() == dimsI.size() && "Invalid number of dimensions");
    for (int i = 0; i < dims0.size(); i++) {
      if (i == dimension_)
        continue;
      assert(dimsI[i] == dims0[i] && "Invalid dimension");
    }
  }

  // Create the shape of the new vector.
  std::vector<size_t> shape(dims0.begin(), dims0.end());
  shape[dimension_] = sizeNthDim;

  ctx->allocateTensor(&outputWeight_, ElemKind::FloatTy, shape);
  ctx->allocateTensor(&outputGrad_, ElemKind::FloatTy, shape);
}

std::string ConcatNode::getDebugRepr(Context *ctx) const {
  DescriptionBuilder db(getName());
  for (int i = 0, e = inputs_.size(); i < e; i++) {
    db.addDim("input", inputs_[i]->dims(ctx));
  }
  db.addDim("output", getOutputWeight(ctx)->dims());
  db.addParam("dimension_", (size_t)dimension_);
  return db;
}

void ConcatNode::forward(Context *ctx, PassKind kind) const {
  auto outW = getWeightHandle(ctx);

  /// Insert the tensors at this coordinate. Start at zero.
  std::vector<size_t> offset(outW.size(), 0);

  for (int i = 0, e = inputs_.size(); i < e; i++) {
    auto inW = inputs_[i]->getWeightHandle(ctx);

    // Insert the tensor.
    insertTensors(inW, outW, offset);

    // The next tensor starts after this one ends.
    offset[dimension_] += inW.dims()[dimension_];
  }
}

void ConcatNode::backward(Context *ctx) const {
  auto outG = getGradHandle(ctx);

  /// Insert the tensors at this coordinate. Start at zero.
  std::vector<size_t> offset(outG.size(), 0);

  for (int i = 0, e = inputs_.size(); i < e; i++) {
    auto inG = inputs_[i]->getGradHandle(ctx);

    // Insert the tensor.
    extractTensors(inG, outG, offset);

    // TODO: this code assumes that input[i] has only one user, because it
    // zeros the gradient before extracting the tensor.

    // The next tensor starts after this one ends.
    offset[dimension_] += inG.dims()[dimension_];
  }
}

ReshapeNode::ReshapeNode(Network *N, NodeBase *input, ArrayRef<size_t> shape)
    : input_(input), shape_(shape.vec()) {}

void ReshapeNode::init(Context *ctx) const {
  assert(input_ && input_->size(ctx) && "Invalid input");

  auto newSize = std::accumulate(shape_.begin(), shape_.end(), 1,
                                 std::multiplies<size_t>());
  (void)newSize;
  assert(input_->size(ctx) == newSize && "New shape must be of the same size.");

  ctx->allocateTensor(&outputWeight_, ElemKind::FloatTy, shape_);
  ctx->allocateTensor(&outputGrad_, ElemKind::FloatTy, shape_);
}

void ReshapeNode::forward(Context *ctx, PassKind kind) const {
  auto outW = getWeightHandle(ctx);
  auto inW = input_->getWeightHandle(ctx);

  for (size_t i = 0, e = inW.size(); i < e; i++) {
    outW.raw(i) = inW.raw(i);
  }
}

void ReshapeNode::backward(Context *ctx) const {
  auto outG = getGradHandle(ctx);
  auto inG = input_->getGradHandle(ctx);
  for (size_t i = 0, e = outG.size(); i < e; i++) {
    inG.raw(i) += outG.raw(i);
  }
}

BatchNormalizationNode::BatchNormalizationNode(Network *N, NodeBase *input,
                                               size_t channelIdx,
                                               FloatTy epsilon,
                                               FloatTy momentum)
    : input_(input), channelIdx_(channelIdx), epsilon_(epsilon),
      momentum_(momentum) {}

void BatchNormalizationNode::init(Context *ctx) const {
  assert(input_ && input_->size(ctx) && "Invalid input");
  assert(input_->dims(ctx).size() > 2 && "Invalid dimensions.");

  // Allocate the output buffer:
  ctx->allocateTensor(&outputWeight_, ElemKind::FloatTy, input_->dims(ctx));
  ctx->allocateTensor(&outputGrad_, ElemKind::FloatTy, input_->dims(ctx));

  // Figure out how many channels are in the tensor.
  size_t channels = input_->dims(ctx)[channelIdx_];

  // Allocate the learnable parameters beta and gamma.
  std::vector<size_t> paramDim = {channels};
  ctx->allocateTensor(&betaW_, ElemKind::FloatTy, paramDim,
                      Context::ShareKind::kSharedTensor);
  ctx->allocateTensor(&betaG_, ElemKind::FloatTy, paramDim);
  ctx->allocateTensor(&gammaW_, ElemKind::FloatTy, paramDim,
                      Context::ShareKind::kSharedTensor);
  ctx->allocateTensor(&gammaG_, ElemKind::FloatTy, paramDim);

  // Allocate the batch-local storage for mean and var.
  ctx->allocateTensor(&mean_, ElemKind::FloatTy, paramDim);
  ctx->allocateTensor(&variance_, ElemKind::FloatTy, paramDim);

  ctx->getHandle(&betaW_).clear();
  ctx->getHandle(&gammaW_).clear(1.0);

  // Tie the gradient and weight tensors and register them for sgd update.
  ctx->addTensorPair({&betaW_, &betaG_});
  ctx->addTensorPair({&gammaW_, &gammaG_});
}

std::string BatchNormalizationNode::getDebugRepr(Context *ctx) const {
  DescriptionBuilder db(getName());
  db.addDim("input", input_->dims(ctx));
  db.addDim("output", getOutputWeight(ctx)->dims());
  db.addDim("beta", ctx->getHandle(&betaW_).dims());
  db.addDim("gamma", ctx->getHandle(&gammaW_).dims());

  db.addParam("channelIdx", channelIdx_);
  db.addParam("epsilon", epsilon_);
  db.addParam("momentum", momentum_);
  return db;
}

void BatchNormalizationNode::forward(Context *ctx, PassKind kind) const {

  if (kind == PassKind::kInference) {
    return forwardInfer(ctx);
  }

  return forwardTrain(ctx);
}

void BatchNormalizationNode::forwardInfer(Context *ctx) const {
  auto betaWH = ctx->getTensor(&betaW_)->getHandle<FloatTy>();
  auto gammaWH = ctx->getTensor(&gammaW_)->getHandle<FloatTy>();
  auto varH = ctx->getTensor(&variance_)->getHandle<FloatTy>();
  auto meanH = ctx->getTensor(&mean_)->getHandle<FloatTy>();

  auto outW = getWeightHandle(ctx);
  auto inW = input_->getWeightHandle(ctx);

  // http://cthorey.github.io./backpropagation/
  //
  // mu = 1/N*np.sum(h,axis =0)
  // sigma2 = 1/N*np.sum((h-mu)**2)
  // hath = (h-mu)*(sigma2+epsilon)**(-1./2.)
  // y = gamma*hath+beta

  // In inference mode just apply the transformation:
  // y[i] = (x - mu) * gamma / stdvar + beta;
  for (size_t i = 0, e = inW.size(); i < e; i++) {
    size_t channelId = inW.getDimForPtr(channelIdx_, i);
    FloatTy x = inW.raw(i);

    FloatTy mu = meanH.at({channelId});
    FloatTy var = varH.at({channelId});

    FloatTy stdvar = 1.0 / std::sqrt(var + epsilon_);

    FloatTy gamma = gammaWH.at({channelId});
    FloatTy beta = betaWH.at({channelId});

    outW.raw(i) = (x - mu) * gamma * stdvar + beta;
  }
}

void BatchNormalizationNode::forwardTrain(Context *ctx) const {
  auto varH = ctx->getTensor(&variance_)->getHandle<FloatTy>();
  auto meanH = ctx->getTensor(&mean_)->getHandle<FloatTy>();
  auto inW = input_->getWeightHandle(ctx);

  Tensor localMean(ElemKind::FloatTy, meanH.dims());
  Tensor localVar(ElemKind::FloatTy, varH.dims());
  auto localMeanH = localMean.getHandle<FloatTy>();
  auto localVarH = localVar.getHandle<FloatTy>();

  // The number of different channels.
  const size_t numChannels = input_->dims(ctx)[channelIdx_];
  // THe number of elements that each channel holds.
  const size_t samplesPerChannel = inW.size() / numChannels;

  // Calculate Mean:

  // sum(in[i])
  for (size_t i = 0, e = inW.size(); i < e; i++) {
    size_t channelId = inW.getDimForPtr(channelIdx_, i);
    FloatTy v = inW.raw(i);
    localMeanH.raw(channelId) += v;
  }
  // Mean = sum(in[i]) / N
  for (size_t i = 0, e = localMeanH.size(); i < e; i++) {
    localMeanH.at({i}) /= samplesPerChannel;
  }

  // Calculate Variance:

  // sum((x - mu) ^ 2)
  for (size_t i = 0, e = inW.size(); i < e; i++) {
    size_t channelId = inW.getDimForPtr(channelIdx_, i);
    FloatTy v = inW.raw(i) - localMeanH.at({channelId});
    localVarH.raw(channelId) += v * v;
  }
  // Var = sum((x - mu) ^ 2) / N
  for (size_t i = 0, e = localMeanH.size(); i < e; i++) {
    localVarH.at({i}) /= samplesPerChannel;
  }

  // Update the global variance and mean:
  for (size_t i = 0, e = localMeanH.size(); i < e; i++) {
    auto P = momentum_;
    meanH.at({i}) = P * localMeanH.at({i}) + (1 - P) * meanH.at({i});
    varH.at({i}) = P * localVarH.at({i}) + (1 - P) * varH.at({i});
  }

  // TODO: should we be using the running mean or the local mean?
  forwardInfer(ctx);
}

void BatchNormalizationNode::backward(Context *ctx) const {
  auto gammaWH = ctx->getTensor(&gammaW_)->getHandle<FloatTy>();
  auto betaGH = ctx->getTensor(&betaG_)->getHandle<FloatTy>();
  auto gammaGH = ctx->getTensor(&gammaG_)->getHandle<FloatTy>();
  auto varH = ctx->getTensor(&variance_)->getHandle<FloatTy>();
  auto meanH = ctx->getTensor(&mean_)->getHandle<FloatTy>();

  auto inW = input_->getWeightHandle(ctx);
  auto outG = getGradHandle(ctx);
  auto inG = input_->getGradHandle(ctx);

  // Update the gradient of the incoming buffer:
  Tensor dyhmu(ElemKind::FloatTy, meanH.dims());
  Tensor sumDy(ElemKind::FloatTy, meanH.dims());
  auto dyhmuH = dyhmu.getHandle<FloatTy>();
  auto sumDyH = sumDy.getHandle<FloatTy>();

  // The number of different channels.
  const size_t numChannels = input_->dims(ctx)[channelIdx_];
  // THe number of elements that each channel holds.
  const size_t samplesPerChannel = inW.size() / numChannels;

  // Calculate: sum(dy * (h - mu))
  for (size_t i = 0, e = inW.size(); i < e; i++) {
    size_t channelId = inW.getDimForPtr(channelIdx_, i);
    // x - mean.
    FloatTy cx = inW.raw(i) - meanH.at({channelId});
    // dy * (h - mu)
    dyhmuH.at({channelId}) += outG.raw(i) * cx;
  }

  // Calculate: sum(dy)
  for (size_t i = 0, e = inW.size(); i < e; i++) {
    size_t channelId = inW.getDimForPtr(channelIdx_, i);
    sumDyH.at({channelId}) += outG.raw(i);
  }

  // http://cthorey.github.io./backpropagation/
  //
  // mu = 1./N*np.sum(h)
  // var = 1./N*np.sum((h-mu)**2)
  // dbeta = np.sum(dy)
  // dgamma = np.sum((h - mu) * (var + eps)**(-1. / 2.) * dy)
  // dh = (1. / N) * gamma * (var + eps)**(-1. / 2.) *
  //     (N * dy - np.sum(dy) - (h - mu) * 1/(var + eps) *
  //     np.sum(dy * (h - mu)))
  //
  for (size_t i = 0, e = inW.size(); i < e; i++) {
    size_t channelId = inW.getDimForPtr(channelIdx_, i);

    FloatTy invN = (1. / samplesPerChannel);
    FloatTy gamma = gammaWH.at({channelId});
    FloatTy var = varH.at({channelId});
    FloatTy mu = meanH.at({channelId});
    FloatTy invVarSqrt = 1. / std::sqrt(var + epsilon_);
    FloatTy invVar = 1. / (var + epsilon_);

    FloatTy dy = outG.raw(i);
    FloatTy hmu = inW.raw(i) - mu;
    FloatTy sdy = sumDyH.at(channelId);
    FloatTy sdyhmu = dyhmuH.at(channelId);
    inG.raw(i) += invN * gamma * invVarSqrt *
                  (samplesPerChannel * dy - sdy - hmu * invVar * sdyhmu);
  }

  // Update the gradient of beta and gamma.
  for (size_t i = 0, e = inW.size(); i < e; i++) {
    size_t channelId = inW.getDimForPtr(channelIdx_, i);

    FloatTy mu = meanH.at({channelId});
    FloatTy var = varH.at({channelId});
    FloatTy invVarSqrt = 1. / std::sqrt(var + epsilon_);

    betaGH.at({channelId}) += outG.raw(i);
    gammaGH.at({channelId}) += (inW.raw(i) - mu) * invVarSqrt * outG.raw(i);
  }
}

ArithmeticNode::ArithmeticNode(Network *N, NodeBase *LHS, NodeBase *RHS,
                               OpKind op)
    : LHS_(LHS), RHS_(RHS), op_(op) {}

void ArithmeticNode::init(Context *ctx) const {
  assert(LHS_ && LHS_->size(ctx) && "Invalid LHS");
  assert(RHS_ && RHS_->size(ctx) && "Invalid RHS");
  assert(RHS_->dims(ctx) == LHS_->dims(ctx) && "Operand sizes does not match.");

  ctx->allocateTensor(&outputWeight_, ElemKind::FloatTy, LHS_->dims(ctx));
  ctx->allocateTensor(&outputGrad_, ElemKind::FloatTy, LHS_->dims(ctx));
}

std::string ArithmeticNode::getDebugRepr(Context *ctx) const {
  DescriptionBuilder db(getName());
  db.addDim("LHS", LHS_->dims(ctx));
  db.addDim("RHS", RHS_->dims(ctx));
  db.addDim("output", getOutputWeight(ctx)->dims());
  db.addParam("operation", op_ == OpKind::kAdd ? "add" : "mul");
  return db;
}

void ArithmeticNode::forward(Context *ctx, PassKind kind) const {
  auto outW = getWeightHandle(ctx);
  auto LHSW = LHS_->getWeightHandle(ctx);
  auto RHSW = RHS_->getWeightHandle(ctx);

  switch (op_) {
  case OpKind::kAdd:
    for (size_t i = 0, e = outW.size(); i < e; i++) {
      outW.raw(i) = LHSW.raw(i) + RHSW.raw(i);
    }
    return;
    break;

  case OpKind::kMul:
    for (size_t i = 0, e = outW.size(); i < e; i++) {
      outW.raw(i) = LHSW.raw(i) * RHSW.raw(i);
    }
    return;
    break;
  }
}

void ArithmeticNode::backward(Context *ctx) const {
  auto LHSW = LHS_->getWeightHandle(ctx);
  auto RHSW = RHS_->getWeightHandle(ctx);
  auto outG = getGradHandle(ctx);
  auto LHSG = LHS_->getGradHandle(ctx);
  auto RHSG = RHS_->getGradHandle(ctx);

  switch (op_) {
  case OpKind::kAdd:
    for (size_t i = 0, e = outG.size(); i < e; i++) {
      LHSG.raw(i) = outG.raw(i);
      RHSG.raw(i) = outG.raw(i);
    }
    return;
    break;

  case OpKind::kMul:
    for (size_t i = 0, e = outG.size(); i < e; i++) {
      LHSG.raw(i) = RHSW.raw(i) * outG.raw(i);
      RHSG.raw(i) = LHSW.raw(i) * outG.raw(i);
    }
    return;
    break;
  }
}

// Define the node visitor for all nodes in the graph that have a single
// incoming node.

#define DEFINE_CLASS_VISITOR(CLASS_NAME)                                       \
  void CLASS_NAME::visit(NodeBase *parent, NodeVisitor *visitor) {             \
    if (!visitor->shouldVisit(parent, this))                                   \
      return;                                                                  \
    visitor->pre(parent, this);                                                \
    input_->visit(this, visitor);                                              \
    visitor->post(parent, this);                                               \
  }

DEFINE_CLASS_VISITOR(ConvNode)
DEFINE_CLASS_VISITOR(MaxPoolNode)
DEFINE_CLASS_VISITOR(FullyConnectedNode)
DEFINE_CLASS_VISITOR(RELUNode)
DEFINE_CLASS_VISITOR(ReshapeNode)
DEFINE_CLASS_VISITOR(SigmoidNode)
DEFINE_CLASS_VISITOR(RegressionNode)
DEFINE_CLASS_VISITOR(MaxNode)
DEFINE_CLASS_VISITOR(BatchNormalizationNode)

#define DEFINE_CLASS_REPR(CLASS_NAME)                                          \
  std::string CLASS_NAME::getDebugRepr(Context *ctx) const {                   \
    DescriptionBuilder db(getName());                                          \
    db.addDim("input", input_->dims(ctx));                                     \
    db.addDim("output", getOutputWeight(ctx)->dims());                         \
    return db;                                                                 \
  }

DEFINE_CLASS_REPR(RELUNode);
DEFINE_CLASS_REPR(ReshapeNode);
DEFINE_CLASS_REPR(SigmoidNode);
DEFINE_CLASS_REPR(MaxNode);

void ArithmeticNode::visit(NodeBase *parent, NodeVisitor *visitor) {
  if (!visitor->shouldVisit(parent, this)) {
    return;
  }
  visitor->pre(parent, this);
  LHS_->visit(this, visitor);
  RHS_->visit(this, visitor);
  visitor->post(parent, this);
}

void SoftMaxNode::visit(NodeBase *parent, NodeVisitor *visitor) {
  if (!visitor->shouldVisit(parent, this))
    return;
  visitor->pre(parent, this);
  input_->visit(this, visitor);
  selected_->visit(this, visitor);
  visitor->post(parent, this);
}

void ConcatNode::visit(NodeBase *parent, NodeVisitor *visitor) {
  if (!visitor->shouldVisit(parent, this))
    return;
  visitor->pre(parent, this);
  for (auto &I : inputs_) {
    I->visit(this, visitor);
  }
  visitor->post(parent, this);
}

void Variable::visit(NodeBase *parent, NodeVisitor *visitor) {
  if (!visitor->shouldVisit(parent, this))
    return;
  visitor->pre(parent, this);
  visitor->post(parent, this);
}
