#ifndef GLOW_GRAPH_NODES_H
#define GLOW_GRAPH_NODES_H

#include "glow/Base/Tensor.h"
#include "glow/Graph/Grad.h"
#include "glow/Graph/Node.h"

#include "llvm/ADT/Hashing.h"
#include "llvm/Support/Casting.h"

namespace glow {

class Variable final : public Node {
public:
  /// Specifies the kind of training and initialization for the variable.
  /// Nodes that are marked as 'none' are not modified during the training
  /// process. Other nodes are trained with the inititial value specified by
  /// this enum.
  enum class TrainKind {
    None,      // The variable is not trainable. It is initialized to zero.
    Broadcast, // Broadcast a single value to all elements.
    Xavier,    // Init the variable with random values using the Xavier method.
  };

  /// Specifies the visibility of the variable. Public nodes can't be
  /// optimized because they are visibile to external users that may hold
  /// a reference or handles.
  enum class VisibilityKind {
    Public,  // The variable is visibile from outside the graph.
    Private, // The variable isn't visibile from outside the graph.
  };

private:
  /// The value to use during initialization. This can be the value to splat or
  /// a parameter to specify the range of the random values.
  float val_;
  /// Specifies if the variable is trainable and how it's initialized.
  TrainKind train_;
  /// Specifies the visibility of the variable.
  VisibilityKind visibility_;
  /// The tensor payload that the variable holds.
  Tensor payload_;

  /// Initialize the content of the tensor. If an 'extern' is set then the user
  /// of the graph is responsible for updating the tensor externally.
  void initPayload();

public:
  Variable(llvm::StringRef name, TypeRef Ty, VisibilityKind visibility,
           TrainKind train, float val)
      : Node(Kinded::Kind::VariableNodeKind, name), val_(val), train_(train),
        visibility_(visibility) {
    addResult(Ty);
    initPayload();
  }

  /// \returns True if the Variable is initialized to be in training mode.
  bool isTraining() const { return train_ != TrainKind::None; }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::VariableNodeKind;
  }

  /// \returns the original training mode of the variable.
  TrainKind getTrainKind() const { return train_; }

  /// \returns the visibility of the variable.
  VisibilityKind getVisibilityKind() const { return visibility_; }

  Tensor &getPayload() { return payload_; }

  template <class ElemTy = float> Handle<ElemTy> getHandle() {
    return getPayload().getHandle<ElemTy>();
  }

  void copyFrom(Tensor *t) { payload_.copyFrom(t); }

  unsigned getNumInputs() const;
  llvm::StringRef getInputName(unsigned idx) const;
  NodeValue &getNthInput(unsigned idx);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const;
  std::string getDebugDesc() const;

  void visit(Node *parent, NodeWalker *visitor);
  void visit(const Node *parent, NodeWalker *visitor) const;

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

/// Support for hashing the Nodes. This is required for using
/// llvm::hash_combine.
class Node;
class Tensor;
struct Type;
struct NodeValue;

/// Convert a float into an unsigned integer binary representation.
/// FIXME: This is a workaround, because defining the hash_code
/// hash_value(float) does not work for some reason.
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
