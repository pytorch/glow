#include "noether/Node.h"
#include "noether/Network.h"
#include "noether/Tensor.h"

using namespace noether;

TrainableData *NodeBase::getOutput(Context *ctx) {
  return ctx->getTrainable(&output_);
}

const TrainableData *NodeBase::getOutput(Context *ctx) const {
  return ctx->getTrainable(&output_);
}

ArrayRef<size_t> NodeBase::dims(Context *ctx) const {
  return getOutput(ctx)->dims();
}

size_t NodeBase::size(Context *ctx) const {
  return getOutput(ctx)->size();
}

Handle<FloatTy> NodeBase::getWeightHandle(Context *ctx)  const {
  return ctx->getTrainable(&output_)->getWeightHandle();
}

Handle<FloatTy> NodeBase::getGradHandle(Context *ctx) const {
  return ctx->getTrainable(&output_)->getGradHandle();
}
