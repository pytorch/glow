#ifndef GLOW_MODELS_BUILDERS_H
#define GLOW_MODELS_BUILDERS_H

namespace glow {

class NodeBase;
class Network;

/// Create a very simple network that's based on CaffeNet. Add the nodes into \p
/// N, where the data is read from \p input and the expected softmax labels are
/// compared to \p expected.
NodeBase *createSimpleNet(Network &N, NodeBase *input, NodeBase *expected);

/// Create the Resnet network. Add the nodes into \p N, where the data is read
/// from \p input and the expected softmax labels are compared to \p expected.
/// The network will contain \p resLayers layers.
NodeBase *createResnet(Network &N, NodeBase *data, NodeBase *expected_softmax,
                       unsigned resLayers = 9);

} // namespace glow

#endif // GLOW_MODELS_BUILDERS_H
