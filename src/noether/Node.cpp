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

/// \returns the dimension of the tensor.
ArrayRef<size_t> NodeBase::dims(Context *ctx) const {
  return getOutput(ctx)->dims();
}

/// \returns the number of elements in the tensor.
size_t NodeBase::size(Context *ctx) const {
  return getOutput(ctx)->size();
}
