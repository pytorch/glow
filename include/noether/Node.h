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

/// Represents a node in the network compute graph.
class NodeBase {
protected:
  /// The filter output.
  TensorToken output_;

public:
  /// \returns a descriptive name for the operation.
  virtual std::string getName() const = 0;

  /// Initialize the node.
  virtual void init(Context *ctx) = 0;

  /// Does the forward propagation.
  virtual void forward(Context *ctx) = 0;

  /// Does the backwards propagation.
  virtual void backward(Context *ctx) = 0;

  /// Update the input or expected output variables of the node with data from
  /// \p batch. Select inputs from the slice specified by \p payload.
  virtual void updateInputs(Context *ctx, Tensor *batch, size_t sampleIdx) {}

  /// Update the input or expected output variables of the node with data from
  /// \p var.
  virtual void updateInput(Context *ctx, Tensor *var) {}

  /// This method implements the visitor pattern that scans the compute DAG top
  /// to bottom.
  virtual void visit(NodeVisitor *visitor) = 0;

  /// Dtor.
  virtual ~NodeBase() {}

  /// \returns the output (weights) of a node in the compute graph.
  TrainableData *getOutput(Context *ctx);

  /// \returns the output (weights) of a node in the compute graph.
  const TrainableData *getOutput(Context *ctx) const;

  /// \returns the dimension of the tensor.
  ArrayRef<size_t> dims(Context *ctx) const;

  /// \returns the number of elements in the tensor.
  size_t size(Context *ctx) const;

  /// \returns the weight handle for the node output.
  Handle<FloatTy> getWeightHandle(Context *ctx);

  /// \returns the gradient handle for the node output.
  Handle<FloatTy> getGradHandle(Context *ctx);
};

}

#endif // NOETHER_NODE_H
