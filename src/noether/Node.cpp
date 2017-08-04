#include "noether/Node.h"
#include "noether/Network.h"
#include "noether/Tensor.h"

using namespace noether;

Tensor *NodeBase::getOutputWeight(Context *ctx) const {
  return ctx->getTensor(&outputWeight_);
}

Tensor *NodeBase::getOutputGrad(Context *ctx) const {
  return ctx->getTensor(&outputGrad_);
}

ArrayRef<size_t> NodeBase::dims(Context *ctx) const {
  return getOutputWeight(ctx)->dims();
}

size_t NodeBase::size(Context *ctx) const { return getOutputWeight(ctx)->size(); }

Handle<FloatTy> NodeBase::getWeightHandle(Context *ctx) const {
  return getOutputWeight(ctx)->getHandle<FloatTy>();
}

Handle<FloatTy> NodeBase::getGradHandle(Context *ctx) const {
  return getOutputGrad(ctx)->getHandle<FloatTy>();
}
