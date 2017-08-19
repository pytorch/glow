#include "glow/Node.h"
#include "glow/Network.h"
#include "glow/Tensor.h"

using namespace glow;

Tensor *NodeBase::getOutputWeight(Context *ctx) const {
  return ctx->getTensor(&outputWeight_);
}

Tensor *NodeBase::getOutputGrad(Context *ctx) const {
  return ctx->getTensor(&outputGrad_);
}

/// Zeros the output gradient of this node.
void NodeBase::clearOutputGrad(Context *ctx) const {
  getOutputGrad(ctx)->zero();
}

ArrayRef<size_t> NodeBase::dims(Context *ctx) const {
  return getOutputWeight(ctx)->dims();
}

size_t NodeBase::size(Context *ctx) const {
  return getOutputWeight(ctx)->size();
}

Handle<FloatTy> NodeBase::getWeightHandle(Context *ctx) const {
  return getOutputWeight(ctx)->getHandle<FloatTy>();
}

Handle<FloatTy> NodeBase::getGradHandle(Context *ctx) const {
  return getOutputGrad(ctx)->getHandle<FloatTy>();
}
