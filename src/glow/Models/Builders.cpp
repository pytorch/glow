#include "glow/Models/Builders.h"

#include "glow/Network/Image.h"
#include "glow/Network/Network.h"
#include "glow/Network/Nodes.h"
#include "glow/Network/Tensor.h"

using namespace glow;

namespace {

/// Resnet implementation based on:
/// http://ethereon.github.io/netscope/#/gist/db945b393d40bfa26006

NodeBase *createConv(Network &N, NodeBase *data, size_t num_filter,
                     size_t kernel, size_t stride, size_t pad, bool addRelu) {
  NodeBase *conv = N.createConvNode(data, num_filter, kernel, stride, pad);
  conv = N.createBatchNormalizationNode(conv, 3);
  if (addRelu) {
    conv = N.createRELUNode(conv);
  }
  return conv;
}

NodeBase *createResidualBlock(Network &N, NodeBase *data,
                              ArrayRef<size_t> num_filter, bool diverge,
                              bool shrink) {
  // Some residual nodes shrink the input by a factor of two:
  int shrinkStride = shrink ? 2 : 1;

  NodeBase *R = data;
  NodeBase *L = data;
  R = createConv(N, R, num_filter[0], 1, shrinkStride, 0, true);
  R = createConv(N, R, num_filter[1], 3, 1, 1, true);
  R = createConv(N, R, num_filter[2], 1, 1, 0, false);

  if (diverge) {
    L = createConv(N, L, num_filter[2], 1, shrinkStride, 0, true);
  }

  NodeBase *res = N.createArithmeticNode(R, L, ArithmeticNode::OpKind::kAdd);
  return N.createRELUNode(res);
}

} // namespace

NodeBase *glow::createResnet(Network &N, NodeBase *data,
                             NodeBase *expected_softmax, unsigned resLayers) {
  NodeBase *O = N.createConvNode(data, 64, 7, 2, 3);
  O = N.createBatchNormalizationNode(O, 3);
  O = N.createRELUNode(O);
  O = N.createMaxPoolNode(O, MaxPoolNode::OpKind::kMax, 3, 2, 1);

  O = createResidualBlock(N, O, {64, 64, 256}, true, false);
  O = createResidualBlock(N, O, {64, 64, 256}, false, false);
  O = createResidualBlock(N, O, {64, 64, 256}, false, false);

  O = createResidualBlock(N, O, {128, 128, 512}, true, true);
  O = createResidualBlock(N, O, {128, 128, 512}, false, false);
  O = createResidualBlock(N, O, {128, 128, 512}, false, false);
  O = createResidualBlock(N, O, {128, 128, 512}, false, false);

  O = createResidualBlock(N, O, {256, 256, 1024}, true, true);
  O = createResidualBlock(N, O, {256, 256, 1024}, false, false);
  O = createResidualBlock(N, O, {256, 256, 1024}, false, false);
  O = createResidualBlock(N, O, {256, 256, 1024}, false, false);
  O = createResidualBlock(N, O, {256, 256, 1024}, false, false);
  O = createResidualBlock(N, O, {256, 256, 1024}, false, false);

  O = createResidualBlock(N, O, {512, 512, 2048}, true, true);
  O = createResidualBlock(N, O, {512, 512, 2048}, false, false);
  O = createResidualBlock(N, O, {512, 512, 2048}, false, false);

  NodeBase *pool = N.createMaxPoolNode(O, MaxPoolNode::OpKind::kAvg, 7, 1, 1);
  NodeBase *fc = N.createFullyConnectedNode(pool, 10);
  return N.createSoftMaxNode(fc, expected_softmax);
}

NodeBase *glow::createSimpleNet(Network &N, NodeBase *input,
                                NodeBase *expected) {
  auto *CV0 = N.createConvNode(input, 16, 5, 1, 2);
  auto *RL0 = N.createRELUNode(CV0);
  auto *MP0 = N.createMaxPoolNode(RL0, MaxPoolNode::OpKind::kMax, 2, 2, 0);

  auto *CV1 = N.createConvNode(MP0, 20, 5, 1, 2);
  auto *RL1 = N.createRELUNode(CV1);
  auto *MP1 = N.createMaxPoolNode(RL1, MaxPoolNode::OpKind::kMax, 2, 2, 0);

  auto *CV2 = N.createConvNode(MP1, 20, 5, 1, 2);
  auto *RL2 = N.createRELUNode(CV2);
  auto *MP2 = N.createMaxPoolNode(RL2, MaxPoolNode::OpKind::kMax, 2, 2, 0);

  auto *FCL1 = N.createFullyConnectedNode(MP2, 10);
  auto *RL3 = N.createRELUNode(FCL1);
  auto *SM = N.createSoftMaxNode(RL3, expected);
  return SM;
}
