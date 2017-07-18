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
  virtual void updateBoundInputs(size_t sampleIdx) { }

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
  std::tuple<size_t, size_t, size_t> dims() const { return output_.dims(); }

  /// \returns the number of elements in the tensor.
  size_t size() const { return output_.size(); }
};
}

#endif // NOETHER_NODE_H
