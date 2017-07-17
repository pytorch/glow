#include "noether/Node.h"
#include "noether/Network.h"
#include "noether/Tensor.h"

using namespace noether;

TrainableNode::TrainableNode(Network *N) {
  N->registerDerivTensor(this, &output_);
}
