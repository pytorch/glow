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

// Storage is the base class for Constants, which are bound to tensors, and
// Placeholder nodes which are unbound.
class Storage : public Node {
public:
  enum ResultIndices {
    OutputIdx = 0,
  };

  Storage(Kinded::Kind k, llvm::StringRef name) : Node(k, name) {}

  /// \return the single output value of the node.
  NodeValue getOutput() { return getNthResult(0); }

  /// Declare the standard Node methods.
  /// @{
  void visit(Node *parent, NodeWalker *visitor);
  void visit(const Node *parent, NodeWalker *visitor) const;
  bool isEqual(const Storage &other) const;
  unsigned getNumInputs() const;
  std::string getInputName(unsigned idx) const;
  NodeValue getNthInput(unsigned idx);
  llvm::StringRef getOutputName(unsigned idx) const;
  bool hasSideEffects() const;
  Node *clone() const;
  /// @}

  /// \returns result type of the storage.
  TypeRef getType() const { return Node::getType(0); }

  /// Methods that forward to the result type (that must be valid):
  /// @{
  ElemKind getElementType() const { return getType()->getElementType(); };
  llvm::ArrayRef<size_t> dims() const { return getType()->dims(); };
  /// @}

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::ConstantKind ||
           k->getKind() == Kinded::Kind::PlaceholderKind;
  }
};

class Constant : public Storage {
  /// The tensor payload that the constant holds.
  Tensor payload_;

public:
  /// Create a new constant and initialize its payload.
  Constant(llvm::StringRef name, TypeRef Ty)
      : Storage(Kinded::Kind::ConstantKind, name) {
    addResult(Ty);
    payload_.reset(*Ty);
  }

  Constant(llvm::StringRef name, Tensor &&payload)
      : Storage(Kinded::Kind::ConstantKind, name),
        payload_(std::move(payload)) {
    addResult(&payload_.getType());
  }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::ConstantKind;
  }

  /// \returns a mutable reference to the payload tensor. If the payload tensor
  /// is unowned then it will be converted to an owned copy before returning.
  Tensor &getPayloadMutable() {
    // If payload is unowned, make an owned copy of the payload for
    // modification.
    if (payload_.isUnowned()) {
      payload_ = payload_.clone();
    }
    assert(!payload_.isUnowned() &&
           "Can only modify Constants with owned payloads");
    return payload_;
  }

  // Get an immutable reference to the payload tensor.
  const Tensor &getPayload() const { return payload_; }

  template <class ElemTy = float> Handle<ElemTy> getHandle() {
    return getPayload().getHandle<ElemTy>();
  }

  void assign(const Tensor *t) { payload_.assign(t); }

  void setPayloadType(TypeRef ty) { payload_.setType(ty); }

  std::string getDebugDesc() const;

  llvm::hash_code getHash() const;

  void clearPayload() { payload_.release(); }

  bool verify() const;
};

/// Placeholder nodes are unbound-storage. The content tensors are attached to
/// this node at runtime. Placeholders are used as inputs and output nodes to
/// the network.
class Placeholder : public Storage {
  /// Specifies if the placeholder is trainable.
  bool isTrainable_;

  /// Specifies if associated Tensors should be zeroed when allocated.
  bool allocZero_{false};

public:
  /// Create a new placeholder.
  Placeholder(llvm::StringRef name, TypeRef Ty, bool isTrainable)
      : Storage(Kinded::Kind::PlaceholderKind, name),
        isTrainable_(isTrainable) {
    addResult(Ty);
  }

  /// \returns True if the placeholder are trainable during
  /// differentiation.
  bool isTraining() const { return isTrainable_; }

  /// \returns True if associated Tensors should be zeroed when allocated.
  bool allocZero() const { return allocZero_; }

  /// Sets whether or not associated Tensors should be zeroed.
  void setAllocZero(bool on = true) { allocZero_ = on; }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::PlaceholderKind;
  }

  std::string getDebugDesc() const;

  llvm::hash_code getHash() const;
};

/// Calculate the size of the output tensor based on the convolution/pooling
/// parameters.
inline std::pair<size_t, size_t> calculateConvPoolOutputDims(
    size_t sx, size_t sy, llvm::ArrayRef<unsigned_t> kernels,
    llvm::ArrayRef<unsigned_t> strides, llvm::ArrayRef<unsigned_t> pads,
    unsigned_t dilation = 1) {
  PaddingTLBR pdim(pads);
  ShapeHW kdim(kernels);
  ShapeHW sdim(strides);
  size_t outsx = ((sx + pdim.top + pdim.bottom - kdim.height -
                   (kdim.height - 1) * (dilation - 1)) /
                      sdim.height +
                  1);
  size_t outsy = ((sy + pdim.left + pdim.right - kdim.width -
                   (kdim.width - 1) * (dilation - 1)) /
                      sdim.width +
                  1);
  return {outsx, outsy};
}

/// Calculate the size of the output tensor based on the 3D convolution/pooling
/// parameters \p inH \p inW, \p inT which are the input's height, width, and
/// depth respectively.
inline ShapeHWD calculate3DConvPoolOutputDims(
    size_t inH, size_t inW, size_t inD, llvm::ArrayRef<unsigned_t> kernels,
    llvm::ArrayRef<unsigned_t> strides, llvm::ArrayRef<unsigned_t> pads) {
  PaddingTLNBRF pdim(pads);
  ShapeHWD kdim(kernels);
  ShapeHWD sdim(strides);

  size_t outH =
      ((inH + pdim.top + pdim.bottom - kdim.height) / sdim.height + 1);
  size_t outW = ((inW + pdim.left + pdim.right - kdim.width) / sdim.width + 1);
  size_t outD = ((inD + pdim.near + pdim.far - kdim.depth) / sdim.depth + 1);

  llvm::SmallVector<size_t, 3> outDims{outH, outW, outD};
  return ShapeHWD(llvm::makeArrayRef(outDims));
}

/// Modes of the padding operation.
enum PaddingMode { CONSTANT = 0, REFLECT, EDGE };

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
