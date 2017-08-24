#ifndef GLOW_IMPORTER_CAFFE2_H
#define GLOW_IMPORTER_CAFFE2_H

#include "glow/Support/ADT.h"

#include "caffe/caffe.pb.h"

#include <unordered_map>

namespace glow {

class Network;
class NodeBase;
class Tensor;

/// Loads caffe2 models.
class caffe2ModelLoader {
  /// The caffe2 weights that we are deserializing.
  caffe2::NetDef weights_;
  /// The caffe2 network descriptor that we are deserializing.
  caffe2::NetDef network_;
  /// The network that we are building.
  glow::Network &N_;
  /// Saves network nodes by name.
  std::unordered_map<std::string, NodeBase *> nodeByName_;
  /// A list of weight tensors indexed by name.
  std::unordered_map<std::string, Tensor *> tensors_;

  /// Load the weight tensors from the 'init' file and register them in the map
  /// \p tensors.
  void loadWeights();

  /// Load the structure of the network from the 'net' file.
  void loadNetwork();

  /// \returns the tensor that was registered under the name \p name.
  Tensor *getTensorByName(const std::string &name);

  /// \returns the node  that was registered with the name \p name.
  NodeBase *getOrCreateNodeByName(const std::string &name);

  /// Load the operator \p op into the network. This creates one or more nodes
  /// in the network.
  void loadOperator(const caffe2::OperatorDef &op);

  /// Reads a network (weights or structure) from the serialized protocol buffer
  /// file.
  bool loadProtoFile(caffe2::NetDef &net, const std::string &filename);

public:
  /// Loads the caffe2 model that's represnted by a network description file,
  /// serialized in \p netDescFilename, and weights file, serialized in
  /// \p netWeightFilename, and populates the network in \p N.
  /// The tensors in \p tensors are stored with the names in the list of names
  /// \p names and used as inputs to the network.
  caffe2ModelLoader(const std::string &netDescFilename,
                    const std::string &netWeightFilename,
                    ArrayRef<const char *> names, ArrayRef<Tensor *> tensors,
                    glow::Network &N);
};

} // namespace glow

#endif // GLOW_IMPORTER_CAFFE2_H
