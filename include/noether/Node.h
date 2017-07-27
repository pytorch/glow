#ifndef NOETHER_NODE_H
#define NOETHER_NODE_H

#include "noether/Tensor.h"
#include "noether/Train.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace noether {

class Network;
class NodeBase;
class Context;

class NodeVisitor {
public:
  /// This callback is called before visiting the children of \p N.
  virtual void pre(NodeBase *N) {}

  /// This callback is called after visiting the children of \p N.
  virtual void post(NodeBase *N) {}
};

/// This is the non-templated part of the compute node.
class NodeBase {
public:
  /// \returns a descriptive name for the operation.
  virtual std::string getName() const = 0;

  /// Does the forward propagation.
  virtual void forward(Context *ctx) = 0;

  /// Does the backwards propagation.
  virtual void backward(Context *ctx) = 0;

  /// Update the input or expected output variables of the node with data from
  /// \p batch. Select inputs from the slice specified by \p payload.
  virtual void updateInputs(Tensor *batch, size_t sampleIdx) {}

  /// Update the input or expected output variables of the node with data from
  /// \p var.
  virtual void updateInput(Tensor *var) {}

  /// This method implements the visitor pattern that scans the compute DAG top
  /// to bottom.
  virtual void visit(NodeVisitor *visitor) = 0;

  /// \returns true if the node weights can be changed during the training
  /// process.
  virtual bool isTrainable() { return true; }

  /// Dtor.
  virtual ~NodeBase() {}
};

/// Represents a node in the network compute graph.
class TrainableNode : public NodeBase {
protected:
  /// The filter output.
  Tensor output_;

  /// A pointer to the network that owns the node.
  Network *network_;

public:
  TrainableNode(Network *N);

  /// \returns the output (weights) of a node in the compute graph.
  Tensor &getOutput() { return output_; }

  /// \returns the dimension of the tensor.
  ArrayRef<size_t> dims() const { return output_.dims(); }

  /// \returns the number of elements in the tensor.
  size_t size() const { return output_.size(); }
};
}

#endif // NOETHER_NODE_H
