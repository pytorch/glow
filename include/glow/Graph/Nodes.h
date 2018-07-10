/**
 * Copyright (c) 2017-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef GLOW_GRAPH_NODES_H
#define GLOW_GRAPH_NODES_H

#include "glow/Base/Tensor.h"
#include "glow/Base/Traits.h"
#include "glow/Graph/Grad.h"
#include "glow/Graph/Node.h"

#include "llvm/ADT/Hashing.h"
#include "llvm/Support/Casting.h"

#include <tuple>

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

  /// Initialize the content of the tensor.
  /// Payload is initialized to zero for 'None' TrainKind, and user
  /// of the graph is responsible for updating the tensor externally.
  void initPayload(PseudoRNG &PRNG);

public:
  /// Create a new variable and initialize its payload.
  Variable(llvm::StringRef name, TypeRef Ty, VisibilityKind visibility,
           TrainKind train, float val, PseudoRNG &PRNG)
      : Node(Kinded::Kind::VariableNodeKind, name), val_(val), train_(train),
        visibility_(visibility) {
    addResult(Ty);
    initPayload(PRNG);
  }

  Variable(llvm::StringRef name, VisibilityKind visibility, Tensor &&payload)
      : Node(Kinded::Kind::VariableNodeKind, name), val_(0.0),
        train_(TrainKind::None), visibility_(visibility),
        payload_(std::move(payload)) {
    addResult(&payload_.getType());
  }

  /// \returns True if the Variable is initialized to be in training mode.
  bool isTraining() const { return train_ != TrainKind::None; }

  /// \returns True if the Variable is private.
  bool isPrivate() const { return visibility_ == VisibilityKind::Private; }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::VariableNodeKind;
  }

  /// \returns the original training mode of the variable.
  TrainKind getTrainKind() const { return train_; }

  /// \returns result type of the variable.
  TypeRef getType() const { return Node::getType(0); }

  /// Methods that forward to the result type (that must be valid):
  /// @{
  ElemKind getElementType() const { return getType()->getElementType(); };
  llvm::ArrayRef<size_t> dims() const { return getType()->dims(); };
  /// @}

  /// \returns the value used during initialization.
  float getValue() const { return val_; }

  /// \returns the visibility of the variable.
  VisibilityKind getVisibilityKind() const { return visibility_; }

  Tensor &getPayload() { return payload_; }

  template <class ElemTy = float> Handle<ElemTy> getHandle() {
    return getPayload().getHandle<ElemTy>();
  }

  void copyFrom(const Tensor *t) { payload_.copyFrom(t); }

  unsigned getNumInputs() const;
  llvm::StringRef getInputName(unsigned idx) const;
  NodeValue &getNthInput(unsigned idx);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const;
  std::string getDebugDesc() const;
  Node *clone() const;

  void visit(Node *parent, NodeWalker *visitor);
  void visit(const Node *parent, NodeWalker *visitor) const;

  bool isEqual(const Variable &other) const;

  llvm::hash_code getHash() const;
};

using VariableNode = Variable;

/// Calculate the size of the output tensor based on the convolution parameters.
inline std::pair<size_t, size_t>
calculateConvOutputDims(size_t sx, size_t sy, size_t filterSize, size_t stride,
                        llvm::ArrayRef<size_t> pads) {
  PaddingTLBR pdim(pads);
  size_t outsx = ((sx + pdim.top + pdim.bottom - filterSize) / stride + 1);
  size_t outsy = ((sy + pdim.left + pdim.right - filterSize) / stride + 1);
  return {outsx, outsy};
}

/// Calculate the size of the output tensor based on the pooling parameters.
inline std::pair<size_t, size_t> calculatePoolOutputDims(size_t sx, size_t sy,
                                                         size_t filterSize,
                                                         size_t stride,
                                                         size_t pad) {
  size_t outsx = ((sx + 2 * pad - filterSize) / stride + 1);
  size_t outsy = ((sy + 2 * pad - filterSize) / stride + 1);
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
#define DEF_BACKEND_SPECIFIC_INSTR(CLASS, NAME) DEF_INSTR(CLASS, NAME)
#define DEF_VALUE(CLASS, NAME) DEF_INSTR(CLASS, NAME)
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
