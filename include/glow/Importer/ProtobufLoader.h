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

#ifndef GLOW_IMPORTER_PROTOBUFLOADER_H
#define GLOW_IMPORTER_PROTOBUFLOADER_H

#include "glow/Base/Tensor.h"
#include "glow/Graph/Graph.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <google/protobuf/text_format.h>

#include <string>
#include <unordered_map>
#include <vector>

namespace glow {

/// Returns true iff all elements of \p a are the same.
bool isArrayConstant(const llvm::ArrayRef<size_t> a);

/// Prints a single serialized protocol buffer node. This method is useful for
/// debugging the network and printing errors.
template <typename T>
void unexpectedNodeError(const T &node, llvm::StringRef message) {
  std::string str;
  google::protobuf::TextFormat::PrintToString(node, &str);
  llvm::outs() << message << "\n" << str << "\n";
}

/// Reads a single integer.
template <typename T> static int loadInt(const T *arg) {
  assert(arg->has_i() && "Node has no Int value");
  return arg->i();
}

/// Reads a single float.
template <typename T> static float loadFloat(const T *arg) {
  assert(arg->has_f() && "Node has no float value");
  return arg->f();
}

/// Reads a single string.
template <typename T> static const std::string &loadStr(const T *arg) {
  assert(arg->has_s() && "Node has no str value");
  return arg->s();
}

/// Load the 'shape' record into a vector of sizes.
template <typename ElemTy = size_t, typename AttrType>
std::vector<ElemTy> getShape(const AttrType *arg) {
  std::vector<ElemTy> dim;
  for (auto i : arg->ints()) {
    dim.push_back(i);
  }
  return dim;
}

/// Loads array and checks that all elements are the same. Returns array[0].
template <typename T> size_t getConstantArrayHead(const T *arg) {
  auto dim = getShape(arg);
  assert(isArrayConstant(dim) &&
         "Only equal values along each dimensions are supported");
  return dim[0];
}

/// Returns canonical name for a given operator: either \p name() from proto,
/// or its first output's name.
template <typename T> std::string loadOperatorName(const T &op) {
  return op.name().length() ? op.name() : op.output(0);
}

/// Loads model: graph and weights.
class ProtobufLoader {
protected:
  /// The graph that we are constructing.
  Function &G_;
  /// Saves network nodes by name.
  std::unordered_map<std::string, Node *> nodeByName_;
  /// A list of weight tensors indexed by name.
  std::unordered_map<std::string, Tensor *> tensors_;
  /// The external output of the network.
  SaveNode *root_{nullptr};

  /// \returns the tensor that was registered under the name \p name.
  Tensor *getTensorByName(llvm::StringRef name);

  /// Create a new variable \p name initialized with \p tensor and load it.
  /// The load is also registered under \p name.
  /// \returns A load to the newly created variable.
  /// \pre !hasNodeByName(name)
  Node *createAndRememberVariable(
      llvm::StringRef name, Tensor &tensor,
      VisibilityKind visibilityKind = VisibilityKind::Private,
      Variable::TrainKind trainKind = Variable::TrainKind::Broadcast);

  /// \returns the node that was registered with the name \p name or nullptr
  /// if no node has been registered with this name.
  Node *getNodeByNameOrNull(llvm::StringRef name) const;

public:
  /// \returns the node that was registered with the name \p name.
  /// \pre hasNodeByName(name)
  Node *getNodeByName(llvm::StringRef name) const;

  /// \returns the node that was registered with the name \p name or create a
  /// new Variable node for a tensor with this name and load it.
  /// In case a new variable is returned, this method registers
  /// the returned load node, not the variable itself.
  /// Only save and load node are allowed to use variable directly.
  Node *getOrCreateVariableByName(llvm::StringRef name);

  /// \returns The variable referenced by the load registered
  /// under \p name.
  /// \pre isa<LoadNode>(getNodeByName(name))
  Variable *getVariableByName(llvm::StringRef name) const;

  /// Create a new variable \p name initialized with \p tensor.
  /// \returns the newly created variable.
  Node *createVariable(
      llvm::StringRef name, Tensor &tensor,
      VisibilityKind visibilityKind = VisibilityKind::Private,
      Variable::TrainKind trainKind = Variable::TrainKind::Broadcast);

  /// \returns True if the node that's registered using \p name exists.
  bool hasNodeByName(llvm::StringRef name) const;

  /// Constructs new ProtobufLoader object. It will populate the network into \p
  /// F. The tensors in \p tensors are stored with the names in the list of
  /// names \p names and used as inputs to the network.
  ProtobufLoader(llvm::ArrayRef<const char *> names,
                 llvm::ArrayRef<Tensor *> tensors, Function &F);

  virtual ~ProtobufLoader();

  /// \returns the output of the network. This is usually the result of the last
  /// softmax or regression layer.
  SaveNode *getRoot() { return root_; }
};

} // namespace glow

#endif // GLOW_IMPORTER_PROTOBUFLOADER_H
