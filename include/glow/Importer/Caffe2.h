#ifndef GLOW_IMPORTER_CAFFE2_H
#define GLOW_IMPORTER_CAFFE2_H

#include "glow/Graph/Graph.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"

#include <string>
#include <unordered_map>
#include <unordered_set>

namespace caffe2 {
class OperatorDef;
class NetDef;
} // namespace caffe2

namespace glow {

class Tensor;
class Value;

/// Loads caffe2 models.
class caffe2ModelLoader {
  /// The graph that we are constructing.
  Function &G_;
  /// Saves network nodes by name.
  std::unordered_map<std::string, Node *> nodeByName_;
  /// A list of weight tensors indexed by name.
  std::unordered_map<std::string, Tensor *> tensors_;
  /// The external output of the network.
  SaveNode *root_{nullptr};

  /// Load the weight tensors from the 'init' file and register them in the map
  /// \p tensors.
  void loadWeights(caffe2::NetDef &net);

  /// Load the structure of the network from the 'net' file.
  void loadNetwork(caffe2::NetDef &net);

  /// \returns the tensor that was registered under the name \p name.
  Tensor *getTensorByName(const std::string &name);

  /// Load the operator \p op into the network. This creates one or more nodes
  /// in the network.
  void loadOperator(const caffe2::OperatorDef &op);

  /// Reads a network (weights or structure) from the serialized protocol buffer
  /// file.
  bool loadProtoFile(caffe2::NetDef &net, const std::string &filename);

public:
  /// \returns the node that was registered with the name \p name.
  Node *getNodeByName(const std::string &name);

  /// \returns the node that was registered with the name \p name or create a
  /// new Variable node for a tensor with this name.
  Node *getOrCreateNodeByName(const std::string &name);

  /// \returns True if the node that's registered using \p name exists.
  bool hasNodeByName(const std::string &name) const;

  /// Loads the caffe2 model that's represented by a network description file,
  /// serialized in \p netDescFilename, and weights file, serialized in
  /// \p netWeightFilename, and populates the network in \p N.
  /// The tensors in \p tensors are stored with the names in the list of names
  /// \p names and used as inputs to the network.
  caffe2ModelLoader(const std::string &netDescFilename,
                    const std::string &netWeightFilename,
                    llvm::ArrayRef<const char *> names,
                    llvm::ArrayRef<Tensor *> tensors, Function &G);

  ~caffe2ModelLoader();

  /// \returns the output of the network. This is usually the result of the last
  /// softmax or regression layer.
  SaveNode *getRoot() { return root_; }
};

} // namespace glow

#endif // GLOW_IMPORTER_CAFFE2_H
