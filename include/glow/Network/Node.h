#ifndef GLOW_NODE_H
#define GLOW_NODE_H

#include "glow/Network/Tensor.h"
#include "glow/Network/Train.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace glow {

class Network;
class NodeBase;
class Context;

class NodeVisitor {
public:
  /// This callback is called before visiting the children of \p N.
  virtual void pre(NodeBase *N) {}

  /// This callback is called after visiting the children of \p N.
  virtual void post(NodeBase *N) {}

  /// This callback is called before processing the graph. If the method returns
  /// false then we skip this node.
  virtual bool shouldVisit(NodeBase *N) { return true; }
};

/// Represents a node in the network compute graph.
class NodeBase {
public:
  /// Describes the kind of pass that the network is executing.
  enum class PassKind {
    /// The network is in inference mode.
    kInference,
    /// The network is training.
    kTraining,
  };

protected:
  /// The filter output weight.
  TensorToken outputWeight_;

  /// The filter output gradient.
  TensorToken outputGrad_;

public:
  /// \returns a descriptive name for the operation.
  virtual std::string getName() const = 0;

  /// Initialize the node.
  virtual void init(Context *ctx) const = 0;

  /// Does the forward propagation. If \p kind describes the kind of
  /// operation that the network is doing (training, inference, etc).
  virtual void forward(Context *ctx, PassKind kind) const = 0;

  /// Does the backwards propagation.
  virtual void backward(Context *ctx) const = 0;

  /// This method implements the visitor pattern that scans the compute DAG top
  /// to bottom.
  virtual void visit(NodeVisitor *visitor) = 0;

  /// Dtor.
  virtual ~NodeBase() = default;

  /// \returns the output (weights) of a node in the compute graph.
  Tensor *getOutputWeight(Context *ctx) const;

  /// \returns the output (gradient) of a node in the compute graph.
  Tensor *getOutputGrad(Context *ctx) const;

  /// Zeros the output gradient of this node.
  void clearOutputGrad(Context *ctx) const;

  /// \returns the dimension of the tensor.
  ArrayRef<size_t> dims(Context *ctx) const;

  /// \returns the number of elements in the tensor.
  size_t size(Context *ctx) const;

  /// \returns the weight handle for the node output.
  Handle<FloatTy> getWeightHandle(Context *ctx) const;

  /// \returns the gradient handle for the node output.
  Handle<FloatTy> getGradHandle(Context *ctx) const;
};
} // namespace glow

#endif // GLOW_NODE_H
