#ifndef GLOW_GRAPH_NODE_H
#define GLOW_GRAPH_NODE_H

#include "llvm/ADT/StringRef.h"

#include "glow/IR/Traits.h"
#include "glow/IR/Type.h"

namespace glow {

/// Represents a node in the compute graph.
class Node : public Kinded, public Typed, public Named {

public:
  Node(Kinded::Kind k, TypeRef Ty, llvm::StringRef name = "")
      : Kinded(k), Typed(Ty) {
    setName(name);
  }
};

} // namespace glow

#endif // GLOW_GRAPH_NODE_H
