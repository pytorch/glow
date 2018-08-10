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

class Variable : public Node {
  /// Specifies if the variable is trainable.
  bool isTrainable_;
  /// Specifies the visibility of the variable.
  VisibilityKind visibility_;
  /// The tensor payload that the variable holds.
  Tensor payload_;

public:
  /// Create a new variable and initialize its payload.
  Variable(llvm::StringRef name, TypeRef Ty, VisibilityKind visibility,
           bool isTrainable)
      : Node(Kinded::Kind::VariableKind, name), isTrainable_(isTrainable),
        visibility_(visibility) {
    addResult(Ty);
    payload_.reset(*Ty);
  }

  Variable(llvm::StringRef name, VisibilityKind visibility, Tensor &&payload)
      : Node(Kinded::Kind::VariableKind, name), isTrainable_(false),
        visibility_(visibility), payload_(std::move(payload)) {
    addResult(&payload_.getType());
  }

  /// \returns True if the Variable is initialized to be in training mode.
  bool isTraining() const { return isTrainable_; }

  /// \returns True if the Variable is private.
  bool isPrivate() const { return visibility_ == VisibilityKind::Private; }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::VariableKind;
  }

  /// \returns result type of the variable.
  TypeRef getType() const { return Node::getType(0); }

  /// Methods that forward to the result type (that must be valid):
  /// @{
  ElemKind getElementType() const { return getType()->getElementType(); };
  llvm::ArrayRef<size_t> dims() const { return getType()->dims(); };
  /// @}

  /// \returns the visibility of the variable.
  VisibilityKind getVisibilityKind() const { return visibility_; }

  Tensor &getPayload() { return payload_; }

  const Tensor &getPayload() const { return payload_; }

  template <class ElemTy = float> Handle<ElemTy> getHandle() {
    return getPayload().getHandle<ElemTy>();
  }

  void copyFrom(const Tensor *t) { payload_.copyFrom(t); }

  /// \returns the output NodeValue from the Variable. Variables only have a
  /// single output.
  NodeValue getOutput() { return getNthResult(0); }

  unsigned getNumInputs() const;
  llvm::StringRef getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const;
  std::string getDebugDesc() const;
  Node *clone() const;

  void visit(Node *parent, NodeWalker *visitor);
  void visit(const Node *parent, NodeWalker *visitor) const;

  bool isEqual(const Variable &other) const;

  llvm::hash_code getHash() const;
};

/// Calculate the size of the output tensor based on the convolution/pooling
/// parameters.
inline std::pair<size_t, size_t> calculateConvPoolOutputDims(
    size_t sx, size_t sy, llvm::ArrayRef<unsigned> kernels,
    llvm::ArrayRef<unsigned> strides, llvm::ArrayRef<unsigned> pads) {
  PaddingTLBR pdim(pads);
  ShapeHW kdim(kernels);
  ShapeHW sdim(strides);
  size_t outsx =
      ((sx + pdim.top + pdim.bottom - kdim.height) / sdim.height + 1);
  size_t outsy = ((sy + pdim.left + pdim.right - kdim.width) / sdim.width + 1);
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
llvm::hash_code hash_value(const glow::NodeHandle &T);

} // namespace glow

// The rest of the nodes are auto-generated into this file:
#include "glow/AutoGenNodes.h"

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
#include "glow/AutoGenNodes.def"

#define DEF_INSTR(CLASS, NAME) case glow::Kinded::Kind::CLASS##Kind:
#define DEF_BACKEND_SPECIFIC_INSTR(CLASS, NAME) DEF_INSTR(CLASS, NAME)
#define DEF_VALUE(CLASS, NAME) DEF_INSTR(CLASS, NAME)
#include "glow/AutoGenInstr.def"

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
#include "glow/AutoGenNodes.def"
};

} // namespace glow

#endif // GLOW_GRAPH_NODES_H
