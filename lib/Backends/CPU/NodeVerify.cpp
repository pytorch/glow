#include "glow/Graph/Nodes.h"

using namespace glow;

void CPUBackend__MaxZeroNode::verify() const {
  assert(getInput().getType() == getResult().getType() && "Invalid type");
  assert(getInput().dims() == getResult().dims() && "Invalid shape");
}
