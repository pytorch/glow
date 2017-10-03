#ifndef GLOW_GRAPH_NODES_H
#define GLOW_GRAPH_NODES_H

#include "glow/IR/IR.h"

#include "glow/Graph/Node.h"

namespace glow {

class Relu : public Node {

public:
  Relu(Node *in, llvm::StringRef name = "")
      : Node(Kinded::Kind::ReluInstKind, in->getType(), name) {}
};

} // namespace glow

#endif // GLOW_GRAPH_NODES_H
