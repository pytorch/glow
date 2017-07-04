#ifndef NOETHER_NETWORK_H
#define NOETHER_NETWORK_H

#include <vector>
#include <map>

namespace noether {

class LayerBase;

class TrainableData;

class Network {
  /// A list of dependencies.
  std::map<LayerBase*, std::vector<LayerBase*>> deps_;

  /// A list of buffers to train as part of the backwards prop pass.
  std::vector<TrainableData*> trainableBuffers_;

  /// Generate a topological order of the nodes in the network.
  void sortNetwork(std::vector<LayerBase*> &order);
public:
  Network();

  /// Add \p dep as a dependency (prerequisite) for \p layer.
  void addLayerDependency(LayerBase *node, LayerBase *dep);

  /// Registers the derivable data \p weights (weights and gradient) as
  /// belonging to the node \p node.
  void registerDerivTensor(LayerBase *node, TrainableData *weights);

  /// Train the network on a single input.
  void train();

  /// Infer data for a single input.
  void infer();
};

}

#endif // NOETHER_NETWORK_H
