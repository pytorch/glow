#ifndef GLOW_GRAPH_GRAPH_H
#define GLOW_GRAPH_GRAPH_H

namespace glow {

class Node;

/// Represents the compute graph.
class Graph final {
  /// A list of nodes that the graph owns.
  std::vector<Node *> nodes_;

public:
  Graph() = default;
  ~Graph();
};

} // namespace glow

#endif // GLOW_GRAPH_GRAPH_H
