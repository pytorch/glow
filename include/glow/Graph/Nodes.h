#ifndef GLOW_GRAPH_NODES_H
#define GLOW_GRAPH_NODES_H

#include "glow/Base/Tensor.h"
#include "glow/Graph/Node.h"

#include "llvm/ADT/Hashing.h"
#include "llvm/Support/Casting.h"

namespace glow {

class Variable final : public Node {
public:
  enum class InitKind {
    Extern,    // No initialization.
    Broadcast, // Broadcast a single value to all elements.
    Xavier,    // Init the tensor with random values using the Xavier method.
  };

private:
  /// The value to use during initialization. This can be the value to splat or
  /// a parameter to specify the range of the random values.
  float val_;
  /// The initialization mode.
  InitKind initKind_;
  /// The tensor payload that the variable holds.
  Tensor payload_;

  /// Initialize the content of the tensor. If an 'extern' is set then the user
  /// of the graph is responsible for updating the tensor externally.
  void initPayload();

public:
  Variable(llvm::StringRef name, TypeRef Ty, InitKind initKind, float val)
      : Node(Kinded::Kind::VariableNodeKind, name), val_(val),
        initKind_(initKind) {
    addResult(Ty);
    initPayload();
  }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::VariableNodeKind;
  }

  const char *getInitKindStr() const;

  static const char *getInitKindStr(InitKind kind);

  /// \returns the original initialization mode of the variable.
  InitKind getInitKind() const { return initKind_; }

  Tensor &getPayload() { return payload_; }

  template <class ElemTy = float> Handle<ElemTy> getHandle() {
    return getPayload().getHandle<ElemTy>();
  }

  void copyFrom(Tensor *t) { payload_.copyFrom(t); }

  std::string getDebugDesc() const override;

  void visit(Node *parent, NodeWalker *visitor) override;

  bool isEqual(const Variable &other) const;

  llvm::hash_code getHash() const;
};

using VariableNode = Variable;

/// Calculate the size of the output tensor based on the convolution
/// parameters.
inline std::pair<size_t, size_t> calculateConvOutputDims(size_t sx, size_t sy,
                                                         size_t pad,
                                                         size_t filterSize,
                                                         size_t stride) {
  size_t outsx = ((sx + pad * 2 - filterSize) / stride + 1);
  size_t outsy = ((sy + pad * 2 - filterSize) / stride + 1);
  return {outsx, outsy};
}

/// Support for hashing the Nodes. This is required for using llvm::hash_combine.
class Node;
class Tensor;
struct Type;
struct NodeValue;

/// Convert a float into an unsigned integer binary representation.
/// FIXME: This is a workaround, because defining the hash_code hash_value(float)
/// does not work for some reason.
size_t toBinary(float f);
llvm::hash_code hash_value(const glow::Tensor &T);

llvm::hash_code hash_value(const glow::Type *T);

llvm::hash_code hash_value(glow::Node *T);

llvm::hash_code hash_value(const glow::NodeValue &T);

} // namespace glow

// The rest of the nodes are auto-generated into this file:
#include "AutoGenNodes.h"

namespace glow {

/// A helper class for all the Node visitors.
/// You probably shouldn't use this directly.
template <typename ImplClass> class NodeVisitorBase {
public:
  ImplClass &asImpl() { return static_cast<ImplClass &>(*this); }
};

/// A visitor that visits only nodes. It does not recursively
/// visit any children of nodes.
template <typename ImplClass, typename RetTy = void, typename... ArgTys>
class NodeVisitor : public NodeVisitorBase<ImplClass> {
  using super = NodeVisitorBase<ImplClass>;

public:
  using super::asImpl;

  // Perform any required pre-processing before visiting.
  // Sub-classes can override it to provide their custom
  // pre-processing steps.
  void pre(Node *N) {}
  void post(Node *N) {}

  RetTy visit(Node *N, ArgTys... args) {
    asImpl().pre(N, args...);

    switch (N->getKind()) {
#define DEF_NODE(CLASS, NAME)                                                  \
  case glow::Kinded::Kind::CLASS##Kind:                                        \
    return asImpl().visit##CLASS(static_cast<CLASS *>(N),                      \
                                 std::forward<ArgTys>(args)...);
#include "AutoGenNodes.def"

#define DEF_INSTR(CLASS, NAME) case glow::Kinded::Kind::CLASS##Kind:
#define DEF_VALUE(CLASS, NAME) case glow::Kinded::Kind::CLASS##Kind:
#include "AutoGenInstr.def"

      llvm_unreachable(
          "Not reachable, values and instructions are not handled here");
    }
    llvm_unreachable("Not reachable, all cases handled");
  }

// Define default dispatcher implementations chain to parent nodes.
#define DEF_NODE(CLASS, NAME)                                                  \
  RetTy visit##CLASS(CLASS *N, ArgTys... args) {                               \
    auto Ret = asImpl().visit##PARENT(N, std::forward<ArgTys>(args)...);       \
    asImpl().post(N, args...);                                                 \
    return Ret;                                                                \
  }
#include "AutoGenNodes.def"
};

} // namespace glow

#endif // GLOW_GRAPH_NODES_H
