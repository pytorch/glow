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
  virtual void forward() = 0;

  /// Does the backwards propagation.
  virtual void backward() = 0;

  /// If the node is bound to some input or expected output variable then
  /// copy the data now. The parameter \p sampleIdx specifies which input
  /// to load.
  virtual void updateBoundInputs(size_t sampleIdx) {}

  /// This method implements the visitor pattern that scans the compute DAG top
  /// to bottom.
  virtual void visit(NodeVisitor *visitor) = 0;

  /// Dtor.
  virtual ~NodeBase() {}
};

/// Represents a node in the network compute graph.
class TrainableNode : public NodeBase {
protected:
  /// The filter output.
  TrainableData output_;

public:
  TrainableNode(Network *N);

  /// \returns the output of a node in the compute graph.
  TrainableData &getOutput() { return output_; }

  /// \returns the dimension of the tensor.
  ArrayRef<size_t> dims() const { return output_.dims(); }

  /// \returns the number of elements in the tensor.
  size_t size() const { return output_.size(); }
};
}

#endif // NOETHER_NODE_H
