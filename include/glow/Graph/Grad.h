#ifndef GLOW_GRAPH_GRAD_H
#define GLOW_GRAPH_GRAD_H

#include "glow/Graph/Node.h"

#include <unordered_map>

namespace glow {

class Graph;

/// A Helper class that manages the mapping between gradients and activations,
/// and helps to accumulate gradients into variables.
class GraphGradMapper {
  /// The graph that we mutate.
  Graph &G_;
  /// Maps activation values to their gradient values.
  UnownedNodeValueMap map_;

public:
  GraphGradMapper(Graph &G) : G_(G) {}

  /// \register the node \p grad as the grad of \p activation. If the node is
  /// already registered then create an 'add' node that accumulates the gradient
  /// into the grad buffer.
  void addGradient(NodeValue activation, NodeValue grad);

  /// \returns the node that \p activation is mapped to.
  NodeValue getGradient(NodeValue activation);
};

} // namespace glow

#endif // GLOW_GRAPH_GRAD_H
