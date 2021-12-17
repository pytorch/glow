/**
 * Copyright (c) Glow Contributors. See CONTRIBUTORS file.
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

#include "glow/Exporter/ONNXModelWriter.h"
#include "glow/Graph/Utils.h"
#include "glow/Runtime/RuntimeTypes.h"
#include "glow/Support/ZipUtils.h"

#include <stack>

#include "miniz.h"
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl_lite.h>

using namespace glow::runtime;
using google::protobuf::RepeatedPtrField;

namespace glow {
#ifdef FACEBOOK_INTERNAL
extern const char *revisionHash;
#endif /* FACEBOOK_INTERNAL */

#define NUM_FLOAT_DIGS 30

namespace {
template <bool IsInteger, bool IsEnum, typename T> struct AttributeAssigner {
  static void assign(ONNX_NAMESPACE::AttributeProto *attr, const T &container);
};

// Specialization for llvm::ArrayRef<T> container types
template <typename T>
struct AttributeAssigner<false, false, llvm::ArrayRef<T>> {
  static void assign(ONNX_NAMESPACE::AttributeProto *attr,
                     const llvm::ArrayRef<T> &container) {
    attr->set_type(ONNX_NAMESPACE::AttributeProto::INTS);
    for (auto value : container) {
      attr->add_ints(value);
    }
  }
};

// Specialization for string type
template <> struct AttributeAssigner<false, false, std::string> {
  static void assign(ONNX_NAMESPACE::AttributeProto *attr,
                     const std::string &container) {
    attr->set_type(ONNX_NAMESPACE::AttributeProto::STRING);
    attr->set_s(container);
  }
};

// Specialization for StringRef type
template <> struct AttributeAssigner<false, false, llvm::StringRef> {
  static void assign(ONNX_NAMESPACE::AttributeProto *attr,
                     const llvm::StringRef container) {
    attr->set_type(ONNX_NAMESPACE::AttributeProto::STRING);
    attr->set_s(container.str());
  }
};

// Specialization for vector of strings.
template <> struct AttributeAssigner<false, false, std::vector<std::string>> {
  static void assign(ONNX_NAMESPACE::AttributeProto *attr,
                     const std::vector<std::string> &container) {
    attr->set_type(ONNX_NAMESPACE::AttributeProto::STRINGS);
    for (auto &str : container) {
      attr->add_strings(str);
    }
  }
};

// Specialization for float type
template <> struct AttributeAssigner<false, false, float> {
  static void assign(ONNX_NAMESPACE::AttributeProto *attr,
                     const float &container) {
    attr->set_type(ONNX_NAMESPACE::AttributeProto::FLOAT);
    attr->set_f(container);
  }
};

// Specialization for NodeValueArrayRef.
template <> struct AttributeAssigner<false, false, NodeValueArrayRef> {
  static void assign(ONNX_NAMESPACE::AttributeProto *attr,
                     const NodeValueArrayRef &container) {
    attr->set_type(ONNX_NAMESPACE::AttributeProto::STRINGS);
    for (size_t i = 0, e = container.size(); i < e; i++) {
      attr->add_strings(container[i].generateNodeOutputName(
          /* stripResNoFor0thInput */ true));
    }
  }
};

// Specialization for llvm::ArrayRef<float>.
template <> struct AttributeAssigner<false, false, llvm::ArrayRef<float>> {
  static void assign(ONNX_NAMESPACE::AttributeProto *attr,
                     const llvm::ArrayRef<float> &container) {
    attr->set_type(ONNX_NAMESPACE::AttributeProto::FLOATS);
    for (auto value : container) {
      attr->add_floats(value);
    }
  }
};

// Specialization for int type
template <typename T> struct AttributeAssigner<true, false, T> {
  static void assign(ONNX_NAMESPACE::AttributeProto *attr, const T &container) {
    attr->set_type(ONNX_NAMESPACE::AttributeProto::INT);
    attr->set_i(container);
  }
};

// Specialization for enums.
template <typename T> struct AttributeAssigner<false, true, T> {
  static void assign(ONNX_NAMESPACE::AttributeProto *attr, const T &container) {
    attr->set_type(ONNX_NAMESPACE::AttributeProto::STRING);
    std::string storage;
    llvm::raw_string_ostream stream(storage);
    stream << container;
    attr->set_s(stream.str());
  }
};

template <typename T>
void addValueAttribute(ONNX_NAMESPACE::NodeProto *proto,
                       const std::string &name, const T &container) {
  auto *attr = proto->add_attribute();
  attr->set_name(name);
  AttributeAssigner<std::numeric_limits<T>::is_integer, std::is_enum<T>::value,
                    T>::assign(attr, container);
}

/// Adds the type attributes from \p ty to \p proto. \p ioNum, \p isInput, and
/// \p addPrefix are used to format the name of the attribute.
void addTypeAttributes(ONNX_NAMESPACE::NodeProto *proto, TypeRef ty,
                       unsigned ioNum, bool isInput,
                       const std::string &addPrefix = "") {
  // Add ElemKind.
  auto *elemKindAttr = proto->add_attribute();
  elemKindAttr->set_name(
      getTypeAttrID(ioNum, elemKindSignifier, isInput, addPrefix));
  AttributeAssigner<false, false, llvm::StringRef>::assign(
      elemKindAttr, ty->getElementName());

  // Add Shape.
  addValueAttribute(proto,
                    getTypeAttrID(ioNum, shapeSignifier, isInput, addPrefix),
                    ty->dims());

  // Non-standard strides need to be serialized.
  if (!ty->hasStandardStrides()) {
    addValueAttribute(
        proto, getTypeAttrID(ioNum, stridesSignifier, isInput, addPrefix),
        ty->strides());
  }

  // Write out scale/offset if quantized ElemKind.
  if (isQuantizedElemKind(ty->getElementType())) {
    addValueAttribute(proto,
                      getTypeAttrID(ioNum, qScaleSignifier, isInput, addPrefix),
                      ty->getScale());
    addValueAttribute(
        proto, getTypeAttrID(ioNum, qOffsetSignifier, isInput, addPrefix),
        ty->getOffset());
  }
}

/// Adds the type attributes from \p NV to \p proto. \p ioNum, \p isInput, and
/// \p addPrefix are used to format the name of the attribute.
void addTypeAttributes(ONNX_NAMESPACE::NodeProto *proto, const NodeValue &NV,
                       unsigned ioNum, bool isInput,
                       const std::string &addPrefix = "") {
  addTypeAttributes(proto, NV.getType(), ioNum, isInput, addPrefix);
}

/// Add the type attributes from the \p ioNum number input or output (depending
/// on \p isInput) of \p N to \p proto. This includes the ElemKind, the Shape,
/// and scale/offset if ElemKind is quantized. Note that 'i' or 'o' along with
/// \p ioNum is prefixed onto the specific attribute being appended, as ops may
/// have multiple inputs/outputs.
void addTypeAttributes(ONNX_NAMESPACE::NodeProto *proto, const Node *N,
                       unsigned ioNum, bool isInput) {
  NodeValue NV = isInput ? N->getNthInput(ioNum) : N->getNthResult(ioNum);
  return addTypeAttributes(proto, NV, ioNum, isInput);
}

/// Helper function to recursively rewind Tile \p node.
/// Optionally, if provided fills out the repeats \p repeats.
/// Returns the first Tile in a chain of Tiles.
const TileNode *unwindTile(const TileNode *node, std::vector<size_t> *repeats,
                           ReportedNodes &reporter) {
  // unwind Tile
  // Keep track of detected <axis, count> pairs.
  std::vector<std::pair<unsigned_t, unsigned_t>> info;
  const TileNode *tile = node;
  while (tile) {
    // Insert counts and axises in reverse order,
    // cause rewind algorithm navigates from the bottom to the top.
    info.insert(info.begin(), {tile->getAxis(), tile->getCount()});

    if (const auto *TN = llvm::dyn_cast<TileNode>(tile->getInput().getNode())) {
      reporter.insert(TN);
      tile = TN;
    } else {
      break;
    }
  }

  if (repeats) {
    unsigned_t numDims = tile->getInput().dims().size();
    // axis is in a normal case will have [0, 1, ..., numDims - 1] values,
    // in extreme case []. Find missing indices and insert count 1.
    auto aB = info.begin();

    for (unsigned_t i = 0; i < numDims; ++i, ++aB) {
      if (aB == info.end() || aB->first != i) {
        aB = info.insert(aB, {i, 1});
      }
    }

    for (size_t b = 0, e = info.size(); b < e; ++b) {
      repeats->push_back(info[b].second);
    }
  }
  return tile;
}

/// Writes all outputs from Node \p node to protobuf \p proto.
void findOutputNames(const Node *node, ONNX_TRAITS::GraphProto &graph,
                     std::function<void(const std::string &name)> &&callback) {
  // Check if user is SaveNode
  std::set<unsigned> saveResNo;
  std::vector<std::pair<const SaveNode *, unsigned>> saveOutputs;
  std::vector<int> resultUsers(node->getNumResults(), 0);
  for (const auto &use : node->getUsers()) {
    const auto *user = use.getUser();
    unsigned resNo = 0;
    for (unsigned b = 0, e = user->getNumInputs(); b < e; ++b) {
      auto UNV = user->getNthInput(b);
      if (node == UNV.getNode()) {
        resNo = UNV.getResNo();
        resultUsers[resNo]++;
        break;
      }
    }

    if (user->getKind() == Kinded::Kind::SaveNodeKind) {
      // Use the associated placeholder's name.
      const SaveNode *SN = llvm::cast<SaveNode>(user);
      saveOutputs.emplace_back(SN, resNo);
    }
  }

  // If saveNode is the only user of a result, we can just use save name as
  // output name. Otherwise, we have to insert a Identity node to relay this
  // output to save output
  for (const auto &p : saveOutputs) {
    if (resultUsers[p.second] == 1) {
      callback(p.first->getPlaceholder()->getName().str());
      saveResNo.insert(p.second);
    } else {
      auto *proto = graph.add_node();
      proto->set_name(node->getName().str() + "_copy_" +
                      std::to_string(p.second));
      proto->set_op_type("Identity");
      proto->add_input(p.second == 0 ? node->getName().str()
                                     : (node->getName().str() + "_out_" +
                                        std::to_string(p.second)));
      proto->add_output(p.first->getPlaceholder()->getName().str());
    }
  }

  // write the other outputs, if any
  for (unsigned b = 0, e = node->getNumResults(); b < e; ++b) {
    if (saveResNo.count(b)) {
      continue;
    }
    if (b == 0) {
      callback(node->getName().str());
    } else {
      callback(node->getName().str() + "_out_" + std::to_string(b));
    }
  }
}

/// Writes all outputs from Node \p node to protobuf \p proto.
void outputsToProto(const Node *node, ONNX_TRAITS::GraphProto &graph,
                    ONNX_NAMESPACE::NodeProto *proto) {
  findOutputNames(node, graph,
                  [&](const std::string &name) { proto->add_output(name); });
}

/// Writes all inputs from Node \p node to protobuf \p proto.
void inputsToProto(const Node *node, ONNX_NAMESPACE::NodeProto *proto) {
  for (unsigned b = 0, e = node->getNumInputs(); b < e; ++b) {
    const auto NV = node->getNthInput(b);
    auto resNo = NV.getResNo();
    auto name = NV.getNode()->getName().str();
    if (resNo) {
      proto->add_input(name + "_out_" + std::to_string(b));
    } else {
      proto->add_input(name);
    }
  }
}

/// Write the output of the provided type only of node outputs.
bool outputKindToProto(Kinded::Kind kind, const Node *node,
                       ONNX_TRAITS::GraphProto &graph,
                       ONNX_NAMESPACE::NodeProto *proto) {
  bool found = false;
  for (const auto &use : node->getUsers()) {
    const auto *user = use.getUser();
    if (user->getKind() == Kinded::Kind::SaveNodeKind) {
      found = true;
      const SaveNode *SN = llvm::cast<SaveNode>(user);
      proto->add_output(SN->getPlaceholder()->getName().str());
      break;
    } else if (user->getKind() == kind) {
      found = true;
      outputsToProto(user, graph, proto);
    }
  }
  return found;
}

/// Writes MatMul operators from Node \p node into
/// provided graph protobuf \p graph, optionally reports intermediate nodes as
/// visited, signaling that such nodes must be ignored, Depending on \p
/// nodeKind, we can write either MatMul or BatchMatMul. \returns error.
template <typename T>
Error writeMatMulKind(const T *node, ONNX_TRAITS::GraphProto &graph,
                      const std::string &nodeKind) {
  auto *proto = graph.add_node();
  proto->set_name(node->getName().str());
  proto->set_op_type(nodeKind);

  Node *LHS = node->getLHS().getNode();
  proto->add_input(LHS->getName().str());
  Node *RHS = node->getRHS().getNode();
  proto->add_input(RHS->getName().str());

  outputsToProto(node, graph, proto);
  return Error::success();
}

// Creates a Transpose Node as the Result of the \p node.
// Reuses given \p proto pointer and create a new proto adding it to \p graph.
// The permutation argument enables use of a different permutation. E.g. - for
// tranposing 3D convolution with NCTHW2NTHWC.
template <typename T>
void writeTransposeResult(const T *node, ONNX_NAMESPACE::NodeProto *&proto,
                          ONNX_TRAITS::GraphProto &graph,
                          llvm::ArrayRef<unsigned_t> permutation = NCHW2NHWC) {
  // Add dictionary entries.
  llvm::ArrayRef<unsigned_t> container(permutation);
  addValueAttribute(proto, "perm", container);
  // Re-use proto for Transpose node.
  auto newName = node->getName().str() + "_out_transpose";
  proto->set_name(newName);
  proto->set_op_type("Transpose");
  proto->add_input(newName);

  proto->add_output(node->getName().str());

  // T node proto.
  proto = graph.add_node();
  proto->add_output(newName);
}

// Creates a Transpose Node as the Input \p input of the \p node.
// Reuses given \p proto pointer and create a new proto adding it to \p graph.
// The permutation argument enables use of a different permutation. E.g. - for
// tranposing 3D convolution with NCTHW2NTHWC.
void writeTransposeInput(const Node *node, const Node *input,
                         ONNX_NAMESPACE::NodeProto *proto,
                         ONNX_TRAITS::GraphProto &graph,
                         llvm::ArrayRef<unsigned_t> permutation = NHWC2NCHW) {
  // Write "mirror" Transform input, i.e. NHWC2NCHW
  auto newName =
      node->getName().str() + "_" + input->getName().str() + "_in_transpose";
  auto *transformProto = graph.add_node();
  transformProto->set_name(newName);
  transformProto->set_op_type("Transpose");

  // Add dictionary entries.
  llvm::ArrayRef<unsigned_t> container(permutation);
  addValueAttribute(transformProto, "perm", container);
  transformProto->add_input(input->getName().str());
  transformProto->add_output(newName);
  proto->add_input(newName);
}

/// Writes Arithmetic operators with name \p opName from Node \p node into
/// provided graph protobuf \p graph. Arithmetic node may have been broadcasted,
/// \p hasMultidirectionalBroadcast indicates the node can be multidirectional
/// broadcast, if that's the case do not specify the axis or broadcast flag in
/// protobuf, optionally reports intermediate nodes as visited, signaling that
/// such nodes must be ignored, \returns error.
template <typename T>
Error writeArithmetic(const std::string &opName, const T *node,
                      ONNX_TRAITS::GraphProto &graph, ReportedNodes &reporter,
                      bool hasMultidirectionalBroadcast) {
  auto *proto = graph.add_node();
  proto->set_name(node->getName().str());
  proto->set_op_type(opName);
  outputsToProto(node, graph, proto);

  auto LHS = node->getLHS();
  if (const BroadcastNode *BN = llvm::dyn_cast<BroadcastNode>(LHS.getNode())) {
    reporter.insert(BN);
    LHS = BN->getInput();
  }

  int axis = -1;
  auto RHS = node->getRHS();
  if (const BroadcastNode *BN = llvm::dyn_cast<BroadcastNode>(RHS.getNode())) {
    reporter.insert(BN);
    RHS = BN->getInput();
    axis = BN->getAxis();
  }

  proto->add_input(LHS.getNode()->getName().str());
  proto->add_input(RHS.getNode()->getName().str());

  // Check if the shapes of LHS and RHS are different and broadcast attribute is
  // required.
  if (LHS.dims() != RHS.dims() && !hasMultidirectionalBroadcast) {
    addValueAttribute(proto, "axis", axis);
    addValueAttribute(proto, "broadcast", 1UL);
  }

  return Error::success();
}

void tensorShapeFromInput(const std::string &name, TypeRef ty,
                          ONNX_NAMESPACE::ValueInfoProto *valueProto) {
  valueProto->set_name(name);
  auto *type = valueProto->mutable_type();
  auto *tensorType = type->mutable_tensor_type();
  tensorType->set_elem_type(ONNXModelWriter::convertType(*ty));
  auto *tensorShape = tensorType->mutable_shape();
  const auto &dims = ty->dims();
  for (unsigned b = 0, e = dims.size(); b < e; ++b) {
    auto *tensorDims = tensorShape->add_dim();
    tensorDims->set_dim_value(dims[b]);
  }
}

/// Creates the list Nodes in the reverse order of the required order for ONNX.
class ReverseGraphWalker {
  /// A post-order list of nodes.
  std::vector<const Node *> reverseOrder_;
  /// A set of visited nodes.
  std::unordered_set<const Node *> visited_;

  void visit(Function &F) {
    // Write constants first, even they should be placed at the end of the list,
    // we can visit them first, cause they will be written into ONNX
    // initializers, not nodes.
    for (const auto *C : F.findConstants()) {
      reverseOrder_.push_back(C);
      visited_.insert(C);
    }
    // Start visiting all root nodes, i.e. nodes that do not have any users.
    for (auto &N : F.getNodes()) {
      if (N.getNumUsers() == 0) {
        visitIteratively(&N);
      }
    }
  }

  void visitIteratively(const Node *rootNode) {
    std::stack<const Node *> st;
    st.push(rootNode);
    while (st.size()) {
      auto *N = st.top();
      st.pop();

      // Check is node has been visited already.
      if (visited_.count(N)) {
        continue;
      }

      if (N->getKind() == Kinded::Kind::PlaceholderKind) {
        // For Placeholder don't visit uses.
        visited_.insert(N);
        reverseOrder_.push_back(N);
        continue;
      }

      // First visit all users, if any.
      bool continueEarly = false;
      for (const auto &use : N->getUsers()) {
        const auto *UN = use.getUser();
        // Check vacancy.
        if (visited_.count(UN)) {
          continue;
        }
        st.push(UN);
        continueEarly = true;
      }
      if (continueEarly) {
        continue;
      }

      // Visit current node, if it's still vacant.
      if (!visited_.count(N)) {
        visited_.insert(N);
        reverseOrder_.push_back(N);
      }

      // And then visit inputs of the current node.
      for (unsigned b = 0, e = N->getNumInputs(); b < e; ++b) {
        auto *UN = N->getNthInput(b).getNode();
        if (visited_.count(UN)) {
          continue;
        }
        st.push(UN);
      }

      // Additionally visit the predicate input if it exists.
      if (N->hasPredicate()) {
        auto *UN = N->getPredicate().getNode();
        if (!visited_.count(UN)) {
          st.push(UN);
        }
      }
    }
  }

public:
  explicit ReverseGraphWalker(Function &F) { visit(F); }

  llvm::ArrayRef<const Node *> getNodes() const { return reverseOrder_; }
};

template <typename T>
static void addAttrToDocString(T *proto, const std::string &attrName,
                               llvm::StringRef attrVal) {
  *(proto->mutable_doc_string()) += std::string(1, startChar) + attrName +
                                    std::string(1, sepChar) + attrVal.str();
}

} // namespace

Error ONNXModelWriter::insertLoaderNameUniqueOffsetMetadata(
    llvm::StringMap<std::string> &extraMetadataProps,
    const OriginNameToTQPMap &map) {
  RETURN_ERR_IF_NOT(!extraMetadataProps.count("OriginNameToTQPMap"),
                    "Already had OriginNameToTQPMap");
  std::string str;
  for (const auto &nameTQP : map) {
    str += nameTQP.first + offsetSepSig +
           std::to_string(nameTQP.second.offset) + offsetEndSig;
  }
  extraMetadataProps.try_emplace(originNameToUniqueOffsetMappingSignifier, str);
  return Error::success();
}

bool ONNXModelWriter::isIntermediatePHForDAG(const Placeholder *PH) {
  if (!dagMode_) {
    return false;
  }

  bool isInputPH = false, isOutputPH = false;
  for (const auto &use : PH->getUsers()) {
    const auto *userN = use.getUser();
    // Only consider users from Functions in the DAG.
    const Function *userF = userN->getParent();
    if (!functionsFromDAG_.count(userF)) {
      continue;
    }
    const bool currIsInputPH = isInput(PH, *userF);
    const bool currIsOutputPH = isOutput(PH, *userF);
    // Check this is not quantization profiling or training cases.
    assert(
        !(currIsInputPH && currIsOutputPH) &&
        "Do not support PHs that are input and output to a single Function.");
    if (currIsInputPH) {
      isInputPH = true;
    }
    if (currIsOutputPH) {
      isOutputPH = true;
    }
  }

  // If the PH is both an input and an output for the Functions in the DAG then
  // it must be an intermediate.
  return isInputPH && isOutputPH;
}

/// Reverses the order of the nodes in \p nodes.
static void
reverseNodesOrder(RepeatedPtrField<ONNX_NAMESPACE::NodeProto> &nodes) {
  for (size_t i = 0, n = nodes.size(); i < n / 2; ++i) {
    nodes.SwapElements(i, n - i - 1);
  }
}

bool ONNXModelWriter::isWritingConstFoldSubgraph() {
  return graphProtoRoot_ != graphProto_;
}

Error ONNXModelWriter::writeConstantFoldingSubgraph(const Constant *C,
                                                    SaveNode *SN) {
  Function *constFoldFunction = SN->getParent();

  // If we already wrote out this Function we can return early.
  if (!processedConstFoldFunctions_.insert(constFoldFunction).second) {
    return Error::success();
  }

  // Create new constant folding Node, which we add the subgraph to. Always add
  // to the root as these are all loaded before loading any ops.
  auto *constFoldNodeProto = graphProtoRoot_->add_node();
  constFoldNodeProto->set_op_type(constFoldSubgraphNodeName);
  const char *constFoldNodeName = constFoldFunction->getName().data();
  constFoldNodeProto->set_name(constFoldNodeName);

  // Now add the constant folding subgraph to the node.
  auto *constFoldNodeSubgraph = constFoldNodeProto->add_attribute();
  constFoldNodeSubgraph->set_name("ConstFoldSubgraph");
  constFoldNodeSubgraph->set_type(ONNX_NAMESPACE::AttributeProto::GRAPH);

  // Temporarily swap in the constant folding function and the graph proto from
  // the constant folding subgraph node so we can write into it.
  ONNX_TRAITS::GraphProto *origGraphProto = graphProto_;
  graphProto_ = constFoldNodeSubgraph->mutable_g();
  Function *origF = F_;
  F_ = constFoldFunction;
  // Make sure to restore original state of the writer when exiting this scope.
  ScopeGuard restoreOrigStateGuard([&]() {
    graphProto_ = origGraphProto;
    F_ = origF;
  });

  // Now that we have setup the constant folding Function and proto to write
  // into, write the function in.
  RETURN_IF_ERR(writeFunction());

  // Now set the output of the ConstFoldSubgraph node. Only output is the
  // Constant it generates. Note that there are no inputs, as Constant inputs
  // are self-contained to this graph as initializers.
  constFoldNodeProto->add_output(C->getName().data());
  addTypeAttributes(constFoldNodeProto, SN->getOutput(), SaveNode::OutputIdx,
                    /* isInput */ false);

  // Reverse the order of Nodes since we wrote them in reverse order.
  reverseNodesOrder(*graphProto_->mutable_node());

  return Error::success();
}

Error ONNXModelWriter::writeFunction() {
  // Use pre order graph traversal.
  // If node is constant with uses, turned into "input" and create a tensor
  // If node is placeholder with uses, turned into "input" with tensor shape,
  // except the case when placeholder has use as SaveNode.
  // Otherwise call common operators method or special operators and write
  // protobuf inputs from node inputs and protobuf outputs from uses.

  ReverseGraphWalker visitor(*F_);
  for (const auto *N : visitor.getNodes()) {
    if (reportedNodes_.count(N)) {
      continue;
    }

    const auto kind = N->getKind();
    // Handle placeholders cases.
    if (kind == Kinded::Kind::PlaceholderKind) {
      const auto *PH = llvm::cast<Placeholder>(N);
      if (!isInput(PH, *F_)) {
        // Storage as an input to SaveNode - ignore it.
        continue;
      }
      // Skip Placeholders that are only intermediates -- these are
      // understood/recreated during reimporting based on an op's partition.
      if (isIntermediatePHForDAG(PH)) {
        continue;
      }
      // Write global input, output only tensor shape.
      RETURN_IF_EXPECTED_IS_ERR(createProtoForIO(PH, /* isInput */ true));
    } else if (kind == Kinded::Kind::ConstantKind) {
      // Write global initializer, output tensor bytes.
      const auto *C = llvm::cast<Constant>(N);

      // Check if this constant came from constant folding that we recorded and
      // want to serialize in the model. If so then we process it before the
      // Constant itself so that it will be loaded first.
      auto constFoldRecordIt = constFoldRecord_.find(C);
      if (constFoldRecordIt != constFoldRecord_.end()) {
        RETURN_IF_ERR(
            writeConstantFoldingSubgraph(C, constFoldRecordIt->second));
      }

      // Note: Always add initializers to the root graph proto.
      auto *tensorProto = addInitializer(*graphProtoRoot_);

      // If we added a constant folding Node recording the constant folding that
      // generated this initializer then point to it in the initializer.
      if (constFoldRecordIt != constFoldRecord_.end()) {
        SaveNode *SN = constFoldRecordIt->second;
        // Point the original initializerProto to this Node so that it knows
        // where to find the Function to replay the constant folding, along with
        // the resNo needed from the Function.
        auto *constFoldNodeNameProto = tensorProto->add_external_data();
        constFoldNodeNameProto->set_key("ConstFoldNodeName");
        constFoldNodeNameProto->set_value(SN->getParent()->getName().data());
        auto *resNoProto = tensorProto->add_external_data();
        resNoProto->set_key("ConstFoldResNo");
        resNoProto->set_value(std::to_string(SN->getInput().getResNo()));
      }

      // When using useGlowCustomOps we always use generateNodeOutputName for
      // all inputs and outputs.
      tensorProto->set_name(useGlowCustomOps_
                                ? C->getOutput().generateNodeOutputName(
                                      /* stripResNoFor0thInput */ true)
                                : C->getName().str());
      writeTensor(C->getPayload(), tensorProto, useGlowCustomOps_,
                  includeConstantData_);
      if (useGlowCustomOps_) {
        // Also include the layout in the initializer to be loaded later.
        addAttrToDocString(tensorProto, layoutSignifier, C->getLayout());
      }
    } else if (kind == Kinded::Kind::SaveNodeKind) {
      // Save node case, find input and use its name as a global output,
      // output only shape.
      const SaveNode *SN = llvm::cast<SaveNode>(N);
      const auto *PH = SN->getPlaceholder();

      // If useGlowCustomOps then we need to add an Identity to map the name
      // from generateNodeOutputName() to the name of the Placeholder.
      if (useGlowCustomOps_) {
        auto *proto = graphProto_->add_node();
        proto->set_op_type("Identity");
        proto->set_name(SN->getName().data());
        proto->add_input(SN->getInput().generateNodeOutputName(
            /* stripResNoFor0thInput */ true));
        proto->add_output(PH->getName().data());
        addTypeAttributes(proto, SN->getInput(), SaveNode::InputIdx,
                          /* isInput */ true);
        addTypeAttributes(proto, SN->getOutput(), SaveNode::OutputIdx,
                          /* isInput */ false);
        // If dumping a DAG then add partition names to each op that's written.
        // Also only do so when not writing a const fold subgraph.
        if (dagMode_ && !isWritingConstFoldSubgraph()) {
          addValueAttribute(proto, "partitionName", F_->getName().str());
          // Skip writing Placeholders that are only intermediates -- these are
          // understood/recreated during reimporting based on an op's partition.
          if (isIntermediatePHForDAG(PH)) {
            addValueAttribute(proto, "isIntermediateOutputForDAG", true);
            continue;
          }
        }
      }

      ONNX_NAMESPACE::ValueInfoProto *out;
      ASSIGN_VALUE_OR_RETURN_ERR(out,
                                 createProtoForIO(PH, /* isInput */ false));

      // Use the doc string to specify the name that should be used for the
      // SaveNode to ensure it's kept the same between export and import.
      addAttrToDocString(out, saveNameSignifier, SN->getName());
    } else if (useGlowCustomOps_) {
      RETURN_IF_ERR(writeGlowCustomOperator(N, *graphProto_));
    } else {
      RETURN_IF_ERR(writeOperator(N, *graphProto_));
    }
    reportedNodes_.insert(N);
  }

  return Error::success();
}

void ONNXModelWriter::setupNewProto() {
  modelProto_.set_ir_version(irVersion_);
  modelProto_.set_producer_name(useGlowCustomOps_ ? "GlowONNXModelWriter"
                                                  : "ONNXModelWriter");
  auto *opsetImportProto = modelProto_.add_opset_import();
  opsetImportProto->set_version(opsetVersion_);
  graphProto_ = modelProto_.mutable_graph();
  graphProto_->set_name("glow");
  graphProtoRoot_ = graphProto_;
}

static Error writeModelToString(const ::google::protobuf::Message &modelProto,
                                bool textMode, std::string *outputStringPtr) {
  if (textMode) {
    RETURN_ERR_IF_NOT(google::protobuf::TextFormat::PrintToString(
                          modelProto, outputStringPtr),
                      "Error writing to string");
  } else {
    ::google::protobuf::io::StringOutputStream zeroCopyOutput(outputStringPtr);
    ::google::protobuf::io::CodedOutputStream codedOutput(&zeroCopyOutput);
    modelProto.SerializeToCodedStream(&codedOutput);
    RETURN_ERR_IF_NOT(!codedOutput.HadError(),
                      "Can't write to the output string",
                      ErrorValue::ErrorCode::MODEL_WRITER_SERIALIZATION_ERROR);
  }
  return Error::success();
}

Error ONNXModelWriter::finalizeAndWriteProto(llvm::StringRef name) {
  // Nodes have been added in a reverse order from SaveNode up to the inputs,
  // we need to rearrange all nodes in the reverse order before serialization.
  auto *nodes = graphProto_->mutable_node();
  reverseNodesOrder(*nodes);

  // useGlowCustomOps uses Identities differently than the normal writer, so
  // we do not want to do this if so.
  if (!useGlowCustomOps_) {
    // We need to swap back Identity node with the next non-Identity node
    // since we append Identity node to tap out the intermediate results
    for (int i = 0, n = nodes->size(); i < n - 1; ++i) {
      if (nodes->Get(i).op_type() == "Identity") {
        int k = 1;
        while (i + k < n) {
          if (nodes->Get(i + k).op_type() != "Identity") {
            break;
          }
          ++k;
        }
        nodes->SwapElements(i, i + k);
        i += k;
      }
    }
  } else {
#ifdef FACEBOOK_INTERNAL
    addMetadataProp("GlowRevisionHash", revisionHash);
#endif /* FACEBOOK_INTERNAL */
  }

  // If we have loadedPHNames_, then we buffered the non-static PH IO protobuf
  // in inputValueInfos_ and outputValueInfos_. Now we write it all out in order
  // according to indices provided in loadedPHNames_.
  if (loadedPHNames_) {
    const bool ioNumMismatch =
        (inputValueInfos_.size() + outputValueInfos_.size() !=
         loadedPHNames_->size());

    // If total number of inputs and outputs doesn't match the number of
    // placeholders, then log an error message and let the next for-loop find
    // the culprits.
    if (ioNumMismatch) {
      LOG(ERROR) << "Number of buffered inputs and outputs "
                 << (inputValueInfos_.size() + outputValueInfos_.size())
                 << " didn't match the number of loadedPHNames "
                 << loadedPHNames_->size();
    }

    // If we have the loaded PH names map, then we need to reorder the inputs
    // and outputs to follow the same order as provided in the loadedPHNames_.
    std::vector<const Placeholder *> orderedInputs(inputValueInfos_.size());
    std::vector<const Placeholder *> orderedOutputs(outputValueInfos_.size());
    for (const auto &pair : *loadedPHNames_) {
      const Placeholder *PH = pair.first;
      const unsigned orderIdx = pair.second.second;
      if (inputValueInfos_.count(PH)) {
        orderedInputs[orderIdx] = PH;
      } else if (outputValueInfos_.count(PH)) {
        orderedOutputs[orderIdx] = PH;
      } else {
        return MAKE_ERR("PH must either be in inputs or outputs: " +
                        PH->getName().str());
      }
    }

    // If didn't find bad placeholders, then it must be some bad inputs/outputs.
    if (ioNumMismatch) {
      return MAKE_ERR("Found some inputs/outputs that don't have corresponding "
                      "placeholders");
    }

    // Now have IO in order matching loadedPHNames_, so finally write them out.
    for (const Placeholder *PH : orderedInputs) {
      auto *inputProto = graphProto_->add_input();
      inputProto->MergeFrom(inputValueInfos_[PH]);
    }
    for (const Placeholder *PH : orderedOutputs) {
      auto *outputProto = graphProto_->add_output();
      outputProto->MergeFrom(outputValueInfos_[PH]);
    }
  }

  if (zipMode_) {
    RETURN_ERR_IF_NOT(
        outputStringPtr_ == nullptr,
        "OnnxModelWriter write to string for zip mode not supported");
    const bool compressed = false;
    ZipWriter zip(&ff_, name.str());
    std::stringstream ss;
    ss << initializers_.size() << "\n";
    zip.writeRecord("weights", ss.str().c_str(), ss.str().size(), compressed);
    std::string largeBuffer;
    int i = 0;
    // This part is probably quite inefficient as we are deserializing the
    // protobuf to a char buffer and then put it to zip stream. I didn't dig
    // enough to see if we can deserialize it into zip stream directly.
    for (const auto &t : initializers_) {
      std::stringstream nm;
      nm << "weight_" << i++;
      t.SerializeToString(&largeBuffer);
      zip.writeRecord(nm.str(), largeBuffer.c_str(), largeBuffer.size(),
                      compressed);
    }
    if (textMode_) {
      google::protobuf::TextFormat::PrintToString(modelProto_, &largeBuffer);
    } else {
      modelProto_.SerializeToString(&largeBuffer);
    }
    zip.writeRecord("model", largeBuffer.c_str(), largeBuffer.size(),
                    compressed);
    zip.writeEndOfFile();
    ff_.flush();
    ff_.close();
    return Error::success();
  } else {
    if (outputStringPtr_ != nullptr) {
      return writeModelToString(modelProto_, textMode_, outputStringPtr_);
    } else {
      return writeModel(modelProto_, textMode_);
    }
  }
}

ONNXModelWriter::ONNXModelWriter(
    const std::string &modelFilename, Function &F, size_t irVersion,
    size_t opsetVersion, Error *errPtr, bool textMode, bool zipMode,
    bool useGlowCustomOps, bool includeConstantData,
    const llvm::StringMap<std::string> &extraMetadataProps,
    const ConstantFoldingRecordMap &constFoldRecord,
    const BackendSpecificNodeInfo &backendSpecificNodeInfo,
    std::string *outputStringPtr)
    : CommonOperatorWriter(modelFilename, &F, errPtr,
                           outputStringPtr == nullptr),
      irVersion_(irVersion), opsetVersion_(opsetVersion), zipMode_(zipMode),
      textMode_(textMode), includeConstantData_(includeConstantData),
      extraMetadataProps_(extraMetadataProps),
      useGlowCustomOps_(useGlowCustomOps), dagMode_(false),
      constFoldRecord_(constFoldRecord),
      backendSpecificNodeInfo_(backendSpecificNodeInfo),
      loadedPHNames_(nullptr), staticPlaceholderTypes_(nullptr),
      outputStringPtr_(outputStringPtr) {
  // If errPtr already contains an error then don't continue with constructor.
  if (errPtr && *errPtr) {
    return;
  }

  // Lambda to setup the ONNXModelWriter and return any Errors that were raised.
  auto setup = [&]() -> Error {
    setupNewProto();
    for (auto &prop : extraMetadataProps_) {
      addMetadataProp(prop.getKey().str(), prop.second);
    }

    RETURN_IF_ERR(writeFunction());

    return finalizeAndWriteProto(F_->getName());
  };

  if (errPtr) {
    *errPtr = setup();
  } else {
    EXIT_ON_ERR(setup());
  }
}

/// Collect nodes from the DAG from \p root in post order in \p postOrder.
/// Gather visited nodes in \p visited.
static void collectNodesPostOrder(const DAGNode *root,
                                  std::unordered_set<const DAGNode *> &visited,
                                  std::vector<const DAGNode *> &postOrder) {
  if (root == nullptr) {
    return;
  }
  visited.insert(root);
  for (auto &c : root->children) {
    if (visited.count(c) == 0) {
      collectNodesPostOrder(c, visited, postOrder);
    }
  }
  postOrder.push_back(root);
}

void ONNXModelWriter::addMetadataProp(const std::string &key,
                                      const std::string &val) {
  auto *prop = modelProto_.add_metadata_props();
  prop->set_key(key);
  prop->set_value(val);
}

Error ONNXModelWriter::writePartitionAndMetadataProps(
    Module &mod, llvm::ArrayRef<const DAGNode *> postOrder) {
  // Add number of partitions to proto.
  addMetadataProp("numPartitions", std::to_string(postOrder.size()));

  for (size_t i = 0, e = postOrder.size(); i < e; i++) {
    const auto *dagNode = postOrder[i];
    F_ = mod.getFunction(dagNode->name);
    RETURN_ERR_IF_NOT(F_, "Function was not valid from DAGList");

    // Write the nodes of the Function.
    RETURN_IF_ERR(writeFunction());

    // Add to the proto the partition name and related meta info:
    const std::string partIdPrefix = getPartitionIdPrefix(i);

    // name of partition:
    addMetadataProp(partIdPrefix + nameSignifier, dagNode->name);

    // logicalDevices of partition:
    addMetadataProp(partIdPrefix + numLogicalDevicesSignifier,
                    std::to_string(dagNode->logicalDevices.size()));
    for (size_t j = 0, f = dagNode->logicalDevices.size(); j < f; j++) {
      addMetadataProp(partIdPrefix + getLogicalDeviceSignfier(j),
                      std::to_string(dagNode->logicalDevices[j]));
    }

    // backendName of partition:
    addMetadataProp(partIdPrefix + backendNameSignifier, dagNode->backendName);

    // size of partition:
    addMetadataProp(partIdPrefix + sizeSignifier,
                    std::to_string(dagNode->size));

    // backendHints.executionUnits of partition:
    addMetadataProp(partIdPrefix + executionUnitsSignifier,
                    std::to_string(dagNode->backendHints.executionUnits));

    // backendHints.SRAMPrioritization of partition not supported:
    assert(dagNode->backendHints.SRAMPrioritization.size() == 0 &&
           "Do not support SRAMPrioritization saving from DAGNode");

    // backendSpecificOpts of partition:
    addMetadataProp(partIdPrefix + numBackendSpecificOptsSignifier,
                    std::to_string(dagNode->backendSpecificOpts.size()));
    size_t j = 0;
    for (const auto &keyVal : dagNode->backendSpecificOpts) {
      addMetadataProp(partIdPrefix + getBackendSpecificOptKeySignifier(j),
                      keyVal.first);
      addMetadataProp(partIdPrefix + getBackendSpecificOptValSignifier(j),
                      keyVal.second);
      j += 1;
    }

    // replicationCount of partition:
    addMetadataProp(partIdPrefix + replicationCountSignifier,
                    std::to_string(dagNode->replicationCount));
  }

  return Error::success();
}

ONNXModelWriter::ONNXModelWriter(
    const std::string &modelFilename, DAGListTy &dagList, size_t irVersion,
    size_t opsetVersion, Error *errPtr, bool textMode, bool zipMode,
    bool includeConstantData,
    const llvm::StringMap<std::string> &extraMetadataProps,
    const ConstantFoldingRecordMap &constFoldRecord,
    const BackendSpecificNodeInfo &backendSpecificNodeInfo,
    const LoadedPlaceholderNameMap *loadedPHNames,
    const std::map<std::string, Type> *staticPlaceholderTypes,
    std::string *outputStringPtr)
    : CommonOperatorWriter(modelFilename, nullptr, errPtr,
                           outputStringPtr == nullptr),
      irVersion_(irVersion), opsetVersion_(opsetVersion), zipMode_(zipMode),
      textMode_(textMode), includeConstantData_(includeConstantData),
      extraMetadataProps_(extraMetadataProps), useGlowCustomOps_(true),
      dagMode_(true), constFoldRecord_(constFoldRecord),
      backendSpecificNodeInfo_(backendSpecificNodeInfo),
      loadedPHNames_(loadedPHNames),
      staticPlaceholderTypes_(staticPlaceholderTypes),
      outputStringPtr_(outputStringPtr) {
  // If errPtr already contains an error then don't continue with constructor.
  if (errPtr && *errPtr) {
    return;
  }

  // Lambda to setup the ONNXModelWriter and return any Errors that were raised.
  auto setup = [&]() -> Error {
    setupNewProto();
    for (auto &prop : extraMetadataProps_) {
      addMetadataProp(prop.getKey().str(), prop.second);
    }

    RETURN_ERR_IF_NOT(dagList.size() == 1, "Expect only one DAG.");
    const auto &dag = *dagList.begin();

    Module &mod = *dag.root->module;

    // Iterate over the DAG in post-order; Nodes per Function are written in
    // reverse order and reversed at the end, so this follows suit.
    std::unordered_set<const DAGNode *> visited;
    std::vector<const DAGNode *> postOrder;
    collectNodesPostOrder(dag.root.get(), visited, postOrder);
    // Remove the root node from the list as we don't care about it.
    postOrder.pop_back();
    for (const DAGNode *dagNode : postOrder) {
      functionsFromDAG_.insert(mod.getFunction(dagNode->name));
    }

    RETURN_IF_ERR(writePartitionAndMetadataProps(mod, postOrder));

    return finalizeAndWriteProto(dag.root->name);
  };

  if (errPtr) {
    *errPtr = setup();
  } else {
    EXIT_ON_ERR(setup());
  }
}

ONNXModelWriter::TensorType *ONNXModelWriter::addInitializer(GraphType &g) {
  if (zipMode_) {
    initializers_.emplace_back();
    return &initializers_.back();
  } else {
    return g.add_initializer();
  }
}

ONNXModelWriter::TensorType::DataType
ONNXModelWriter::convertType(const Type &glowType) {
  switch (glowType.getElementType()) {
  case ElemKind::FloatTy:
    return TensorType::FLOAT;
  case ElemKind::Float16Ty:
    return TensorType::FLOAT16;
  case ElemKind::BFloat16Ty:
    return TensorType::BFLOAT16;
  case ElemKind::Float64Ty:
    return TensorType::DOUBLE;
  case ElemKind::Int8QTy:
    return TensorType::INT8;
  case ElemKind::UInt8FusedQTy:
  case ElemKind::UInt8FusedFP16QTy:
  case ElemKind::UInt4FusedFP16QTy:
  case ElemKind::UInt4FusedQTy:
  case ElemKind::UInt8QTy:
  case ElemKind::UInt8ITy:
    return TensorType::UINT8;
  case ElemKind::Int16QTy:
    return TensorType::INT16;
  case ElemKind::Int32QTy:
  case ElemKind::Int32ITy:
    return TensorType::INT32;
  case ElemKind::Int64QTy:
  case ElemKind::Int64ITy:
    return TensorType::INT64;
  case ElemKind::BoolTy:
    return TensorType::BOOL;
  }
  LOG(DFATAL) << "Cannot reach here.";
  return TensorType::UNDEFINED; // Avoids a compilation warning.
}

/// Add quantization parameters to the doc_string in \p out based on \p type.
template <typename T>
static void addQuantParamsToDocString(T *out, const Type &type) {
  addAttrToDocString(out, qScaleSignifier,
                     strFormat("%.*f", NUM_FLOAT_DIGS, type.getScale()));
  addAttrToDocString(out, qOffsetSignifier, std::to_string(type.getOffset()));
}

/// Add strides to the doc_string in \p out based on \p type.
template <typename T>
static void addStridesToDocString(T *out, const Type &type) {
  // Non-standard strides need to be serialized.
  if (type.hasStandardStrides()) {
    return;
  }
  const auto &strides = type.strides();
  std::string stridesStr;
  std::string delim;
  for (const auto &stride : strides) {
    stridesStr.append(delim);
    stridesStr.append(std::to_string(stride));
    delim = ",";
  }
  addAttrToDocString(out, stridesSignifier, stridesStr);
}

void ONNXModelWriter::writeTensor(const Tensor &T, TensorType *out,
                                  bool useGlowCustomOps, bool includeData) {
  const auto &type = T.getType();
  out->set_data_type(convertType(type));
  const auto &dims = type.dims();
  for (unsigned b = 0, e = dims.size(); b < e; ++b) {
    out->add_dims(dims[b]);
  }

  if (includeData) {
    out->set_raw_data(T.getUnsafePtr(), type.getSizeInBytes());
  }

  if (useGlowCustomOps) {
    addAttrToDocString(out, elemKindSignifier, type.getElementName());
    addStridesToDocString(out, type);
  }

  if (type.isQuantizedType()) {
    if (useGlowCustomOps) {
      addQuantParamsToDocString(out, type);
    } else {
      // Format is ElemKind:scale:offset.
      out->set_doc_string(strFormat("%s:%.*f:%d", type.getElementName().data(),
                                    NUM_FLOAT_DIGS, T.getType().getScale(),
                                    T.getType().getOffset()));
    }
  }
}

Expected<ONNX_NAMESPACE::ValueInfoProto *>
ONNXModelWriter::createProtoForIO(const Placeholder *PH, bool isInput) {
  // If loadedPHNames_ then we have a specific order we need to write out IO
  // protos. If so, buffer non-static IO that is not part of a constant folding
  // subgraph into inputValueInfos_/outputValueInfos_ to later be written out in
  // order inside finalizeAndWriteProto() based on loadedPHNames_.
  ONNX_NAMESPACE::ValueInfoProto *valueProto;
  if (!loadedPHNames_ || isWritingConstFoldSubgraph() || PH->isStatic()) {
    valueProto = isInput ? graphProto_->add_input() : graphProto_->add_output();
  } else {
    valueProto = isInput ? &inputValueInfos_[PH] : &outputValueInfos_[PH];
  }

  tensorShapeFromInput(PH->getName().str(), PH->getType(), valueProto);

  if (useGlowCustomOps_) {
    // Write out any meta information we need to for the Placeholder.
    addStridesToDocString(valueProto, *PH->getType());
    addAttrToDocString(valueProto, staticSignifier,
                       std::to_string(PH->isStatic()));
    addAttrToDocString(valueProto, trainableSignifier,
                       std::to_string(PH->isTraining()));
    addAttrToDocString(valueProto, layoutSignifier, PH->getLayout());
    addAttrToDocString(valueProto, elemKindSignifier,
                       PH->getType()->getElementName());

    // If we're writing out a Placeholder from the original input Function, then
    // expect to find a corresponding input loaded PH name if they are
    // provided. This is expected when the PH is not static (as otherwise it's
    // input as a weight), when the Function being written isn't a constant
    // folding subgraph (then we have PHs that are used just to save the const
    // folding result), and when the PH isn't intermediate (then it's only
    // visible/used by Glow when executing partitioned DAGs).
    if (loadedPHNames_ && !PH->isStatic() && !isWritingConstFoldSubgraph() &&
        !isIntermediatePHForDAG(PH)) {
      auto it = loadedPHNames_->find(PH);
      RETURN_ERR_IF_NOT(it != loadedPHNames_->end(),
                        "Did not find associated loader name for " +
                            PH->getName().str() + " while writing Function " +
                            F_->getName().str());
      addAttrToDocString(valueProto, loaderNameSignifier, it->second.first);
    }

    // If we have a type that was used for loading a static Placeholder, then
    // serialize that type into a dummy node.
    if (staticPlaceholderTypes_ && PH->isStatic()) {
      auto it = staticPlaceholderTypes_->find(PH->getName().data());
      RETURN_ERR_IF_NOT(it != staticPlaceholderTypes_->end(),
                        "Did not find associated type for static PH " +
                            PH->getName().str() + " while writing Function " +
                            F_->getName().str());

      // Create new static PH dummy node that carries the type that the static
      // PH was loaded with. Note it has no inputs or outputs, howeverr there is
      // a type appended for the output idx, and the node has the same name as
      // the static PH to use when reloading.
      auto *staticPHDummyNodeProto = graphProto_->add_node();
      staticPHDummyNodeProto->set_op_type(staticPHDummyNodeName);
      staticPHDummyNodeProto->set_name(PH->getName().data());

      // Set the output type to be the one we found in staticPlaceholderTypes_.
      addTypeAttributes(staticPHDummyNodeProto, &it->second, Storage::OutputIdx,
                        /* isInput */ false);
    }

    // Also include quantization params if necessary.
    if (PH->getType()->isQuantizedType()) {
      addQuantParamsToDocString(valueProto, *PH->getType());
    }
  }
  return valueProto;
}

Error ONNXModelWriter::writeAllWithNode(const std::string &opName,
                                        const Node *node, GraphType &graph,
                                        NodeType *proto) {
  proto->set_name(node->getName().str());
  proto->set_op_type(opName);
  inputsToProto(node, proto);
  outputsToProto(node, graph, proto);
  return Error::success();
}

Error ONNXModelWriter::writeAll(const std::string &opName, const Node *node,
                                GraphType &graph) {
  return writeAllWithNode(opName, node, graph, graph.add_node());
}

//===-----------------------------------------------------------------===//
//                    Operators Supported by ONNX
//===-----------------------------------------------------------------===//
Error ONNXModelWriter::writePad(const PadNode *node, GraphType &graph) {
  auto *proto = graph.add_node();
  // Add dictionary entries.
  switch (node->getMode()) {
  case PaddingMode::CONSTANT:
    addValueAttribute(proto, "mode", std::string("constant"));
    break;
  case PaddingMode::REFLECT:
    addValueAttribute(proto, "mode", std::string("reflect"));
    break;
  case PaddingMode::EDGE:
    addValueAttribute(proto, "mode", std::string("edge"));
    break;
  default:
    return MAKE_ERR("Pad: Invalid mode",
                    ErrorValue::ErrorCode::MODEL_WRITER_SERIALIZATION_ERROR);
  }

  if (opsetVersion_ <= 10) {
    addValueAttribute(proto, "pads", node->getPads());
    float value = node->getValue();
    if (value != .0f) {
      addValueAttribute(proto, "value", value);
    }
    return writeAllWithNode("Pad", node, graph, proto);
  } else {
    proto->set_name(node->getName().str());
    proto->set_op_type("Pad");
    // Input for data.
    inputsToProto(node, proto);

    // Input for pads.
    auto pads = node->getPads();
    Tensor oneDimTensorPads(ElemKind::Int64ITy, {(dim_t)pads.size()});
    auto oneDimTensorPadsH = oneDimTensorPads.getHandle<int64_t>();
    for (size_t b = 0, e = oneDimTensorPads.size(); b < e; ++b) {
      oneDimTensorPadsH.raw(b) = pads[b];
    }
    auto *tensorProto = addInitializer(graph);
    tensorProto->set_name(node->getName().str() + "_pads");
    writeTensor(oneDimTensorPads, tensorProto);
    proto->add_input(node->getName().str() + "_pads");

    // Input for value.
    Tensor value(ElemKind::FloatTy, {1});
    auto valueH = value.getHandle();
    valueH.raw(0) = node->getValue();
    tensorProto = addInitializer(graph);
    tensorProto->set_name(node->getName().str() + "_value");
    writeTensor(value, tensorProto);
    proto->add_input(node->getName().str() + "_value");
    // Output
    outputsToProto(node, graph, proto);
    return Error::success();
  }
}

Error ONNXModelWriter::writeConcat(const ConcatNode *node, GraphType &graph) {
  auto *proto = graph.add_node();
  // Add dictionary entries.
  addValueAttribute(proto, "axis", node->getDim());

  return writeAllWithNode("Concat", node, graph, proto);
}

Error ONNXModelWriter::writeTranspose(const TransposeNode *node,
                                      GraphType &graph) {
  // Some nodes create transpose for outputs.
  auto *input = node->getInput().getNode();
  if (llvm::dyn_cast<ConvolutionNode>(input) ||
      llvm::dyn_cast<Convolution3DNode>(input) ||
      llvm::dyn_cast<AvgPoolNode>(input) ||
      llvm::dyn_cast<MaxPoolNode>(input) ||
      llvm::dyn_cast<SpaceToDepthNode>(input)) {
    return Error::success();
  }

  auto *proto = graph.add_node();
  // Add dictionary entries.
  addValueAttribute(proto, "perm", node->getShuffle());

  return writeAllWithNode("Transpose", node, graph, proto);
}

Error ONNXModelWriter::writeCollectRpnProposals(
    const CollectRpnProposalsNode *node, GraphType &graph) {
  return writeAllWithNode("CollectRpnProposals", node, graph, graph.add_node());
}

Error ONNXModelWriter::writeFlip(const FlipNode *node, GraphType &graph) {
  auto *proto = graph.add_node();
  // Add dictionary entries.
  addValueAttribute(proto, "axis", node->getAxis());

  return writeAllWithNode("Flip", node, graph, proto);
}

Error ONNXModelWriter::writeAudioSpectrogram(const AudioSpectrogramNode *node,
                                             GraphType &graph) {
  auto *proto = graph.add_node();

  addValueAttribute(proto, "window_size", node->getWindowSize());
  addValueAttribute(proto, "stride", node->getWindowStride());
  addValueAttribute(proto, "magnitude_squared", node->getMagnitudeSquared());

  return writeAllWithNode("AudioSpectrogram", node, graph, proto);
}

Error ONNXModelWriter::writeMFCC(const MFCCNode *node, GraphType &graph) {
  auto *proto = graph.add_node();

  addValueAttribute(proto, "sample_rate", node->getSampleRate());
  addValueAttribute(proto, "lower_frequency_limit", node->getLowerFrequency());
  addValueAttribute(proto, "upper_frequency_limit", node->getUpperFrequency());
  addValueAttribute(proto, "filterbank_channel_count",
                    node->getFilterBankCount());
  addValueAttribute(proto, "dct_coefficient_count", node->getNumCoefficients());

  return writeAllWithNode("MFCC", node, graph, proto);
}

Error ONNXModelWriter::writeROIAlign(const ROIAlignNode *node,
                                     GraphType &graph) {
  auto *proto = graph.add_node();
  switch (node->getMode()) {
  case PoolingMode::AVG:
    addValueAttribute(proto, "mode", std::string("avg"));
    break;
  case PoolingMode::MAX:
    addValueAttribute(proto, "mode", std::string("max"));
    break;
  }
  addValueAttribute(proto, "output_height", node->getOutputHeight());
  addValueAttribute(proto, "output_width", node->getOutputWidth());
  addValueAttribute(proto, "sampling_ratio", node->getSamplingRatio());
  addValueAttribute(proto, "spatial_scale", node->getSpatialScale());
  addValueAttribute(proto, "aligned", node->getAligned());
  addValueAttribute(proto, "rotated", node->getRotated());
  return writeAllWithNode("ROIAlign", node, graph, proto);
}

Error ONNXModelWriter::writeBBoxTransform(const BBoxTransformNode *node,
                                          GraphType &graph) {
  auto *proto = graph.add_node();
  addValueAttribute(proto, "ApplyScale", node->getApplyScale());
  addValueAttribute(proto, "Rotated", node->getRotated());
  addValueAttribute(proto, "AngleBoundOn", node->getAngleBoundOn());
  addValueAttribute(proto, "AngleBoundLo", node->getAngleBoundLo());
  addValueAttribute(proto, "AngleBoundHi", node->getAngleBoundHi());
  addValueAttribute(proto, "ClipAngleThresh", node->getClipAngleThresh());
  return writeAllWithNode("BBoxTransform", node, graph, proto);
}

Error ONNXModelWriter::writeConvolution(const ConvolutionNode *node,
                                        GraphType &graph) {
  // Loading convolution creates a sandwich with Transpose nodes for Input,
  // Weights, and Result. The lowering algorithm can remove Transpose nodes and
  // replace one set of nodes with another ones. When saving a graph to ONNX
  // format, keep in mind that when it will be loaded again a Transpose nodes
  // sandwich will be created again. The steps will be:
  // Remove Transpose nodes for Input and Weights, if such Transpose are not
  // found (they are supposed to be NCHW2NHWC then create a "mirror"
  // Transpose, i.e. NHWC2NCHW for correspondent Input or/and Weights.
  // The similar algorithm will be applied for Result. If Transpose NHWC2NCHW
  // node is found for Result user then remove it, otherwise create a "mirror"
  // Transpose, i.e. NCHW2NHWC.
  assert(node->getLayout() == NHWC && "can only write NHWC Convolutions");

  // Delegate writing quantized Convs to writeTensorwiseQuantizedConvolution.
  if (isQuantizedElemKind(node->getInput().getElementType())) {
    return writeTensorwiseQuantizedConvolution(node, graph);
  }

  auto *proto = graph.add_node();

  // Use the output of transpose node.
  if (!outputKindToProto(Kinded::Kind::TransposeNodeKind, node, graph, proto)) {
    // Apparently Result Transpose has been removed, add NCHW2NHWC Transpose.
    writeTransposeResult(node, proto, graph);
  }

  // Add dictionary entries.
  addValueAttribute(proto, "strides", node->getStrides());
  addValueAttribute(proto, "pads", node->getPads());
  addValueAttribute(proto, "group", node->getGroup());
  addValueAttribute(proto, "dilations", node->getDilation());

  const Node *input = node->getInput().getNode();
  if (const TransposeNode *TN = llvm::dyn_cast<TransposeNode>(input)) {
    proto->add_input(TN->getInput().getNode()->getName().str());
    reportedNodes_.insert(TN);
  } else if (const ReshapeNode *RSN = llvm::dyn_cast<ReshapeNode>(input)) {
    proto->add_input(RSN->getInput().getNode()->getName().str());
    reportedNodes_.insert(RSN);
  } else {
    writeTransposeInput(node, input, proto, graph);
  }

  const Node *filter = node->getFilter().getNode();
  if (const TransposeNode *TN = llvm::dyn_cast<TransposeNode>(filter)) {
    proto->add_input(TN->getInput().getNode()->getName().str());
    reportedNodes_.insert(TN);
  } else if (const ReshapeNode *RSN = llvm::dyn_cast<ReshapeNode>(filter)) {
    proto->add_input(RSN->getInput().getNode()->getName().str());
    reportedNodes_.insert(RSN);
  } else {
    writeTransposeInput(node, filter, proto, graph);
  }

  proto->add_input(node->getBias().getNode()->getName().str());

  proto->set_name(node->getName().str());
  proto->set_op_type("Conv");

  return Error::success();
}

Error ONNXModelWriter::writeTensorwiseQuantizedConvolution(
    const ConvolutionNode *node, GraphType &graph) {
  auto *proto = graph.add_node();

  // Add dictionary entries.
  addValueAttribute(proto, "kernel_shape", node->getKernels());
  addValueAttribute(proto, "strides", node->getStrides());
  addValueAttribute(proto, "pads", node->getPads());
  addValueAttribute(proto, "group", node->getGroup());
  addValueAttribute(proto, "dilation", node->getDilation());

  addValueAttribute(proto, "out_scale",
                    node->getType(ConvolutionNode::ResultIdx)->getScale());
  addValueAttribute(proto, "out_offset",
                    node->getType(ConvolutionNode::ResultIdx)->getOffset());

  return writeAllWithNode("Conv", node, graph, proto);
}

Error ONNXModelWriter::writeChannelwiseQuantizedConvolution(
    const ChannelwiseQuantizedConvolutionNode *node, GraphType &graph) {
  auto *proto = graph.add_node();

  // Add dictionary entries.
  addValueAttribute(proto, "kernel_shape", node->getKernels());
  addValueAttribute(proto, "strides", node->getStrides());
  addValueAttribute(proto, "pads", node->getPads());
  addValueAttribute(proto, "group", node->getGroup());

  addValueAttribute(
      proto, "out_scale",
      node->getType(ChannelwiseQuantizedConvolutionNode::ResultIdx)
          ->getScale());
  addValueAttribute(
      proto, "out_offset",
      node->getType(ChannelwiseQuantizedConvolutionNode::ResultIdx)
          ->getOffset());

  return writeAllWithNode("ChannelwiseQuantizedConvolution", node, graph,
                          proto);
}

Error ONNXModelWriter::writeBatchedReduceMean(const BatchedReduceMeanNode *node,
                                              GraphType &graph) {
  auto *proto = graph.add_node();
  // Add dictionary entries.
  addValueAttribute(proto, "axes", node->getAxes());

  proto->set_name(node->getName().str());
  proto->set_op_type("ReduceMean");
  inputsToProto(node, proto);

  addValueAttribute(proto, "keepdims", 0);
  outputsToProto(node, graph, proto);

  return Error::success();
}

Error ONNXModelWriter::writeBatchedReduceAdd(const BatchedReduceAddNode *node,
                                             GraphType &graph) {
  auto *proto = graph.add_node();
  // Add dictionary entries.
  unsigned_t axis = node->getAxis();
  llvm::ArrayRef<unsigned_t> axes(axis);
  addValueAttribute(proto, "axes", axes);

  proto->set_name(node->getName().str());
  proto->set_op_type("ReduceSum");
  inputsToProto(node, proto);

  addValueAttribute(proto, "keepdims", 0);
  outputsToProto(node, graph, proto);

  return Error::success();
}

Error ONNXModelWriter::writeBatchedReduceSumSquare(
    const BatchedReduceSumSquareNode *node, GraphType &graph) {
  auto *proto = graph.add_node();
  // Add dictionary entries.
  unsigned_t axis = node->getAxis();
  llvm::ArrayRef<unsigned_t> axes(axis);
  addValueAttribute(proto, "axes", axes);

  proto->set_name(node->getName().str());
  proto->set_op_type("ReduceSum");
  inputsToProto(node, proto);

  addValueAttribute(proto, "keepdims", 0);
  outputsToProto(node, graph, proto);

  return Error::success();
}

Error ONNXModelWriter::writeBatchedReduceMax(const BatchedReduceMaxNode *node,
                                             GraphType &graph) {
  auto *proto = graph.add_node();
  // Find dictionary entries.
  addValueAttribute(proto, "axes", node->getAxes());

  return writeAllWithNode("ReduceMax", node, graph, proto);
}

Error ONNXModelWriter::writeBatchedReduceMin(const BatchedReduceMinNode *node,
                                             GraphType &graph) {
  auto *proto = graph.add_node();
  // Find dictionary entries.
  addValueAttribute(proto, "axes", node->getAxes());

  return writeAllWithNode("ReduceMin", node, graph, proto);
}

Error ONNXModelWriter::writeBatchedReduceProd(const BatchedReduceProdNode *node,
                                              GraphType &graph) {
  auto *proto = graph.add_node();
  // Add dictionary entries.
  unsigned_t axis = node->getAxis();
  llvm::ArrayRef<unsigned_t> axes(axis);
  addValueAttribute(proto, "axes", axes);

  proto->set_name(node->getName().str());
  proto->set_op_type("ReduceProd");
  inputsToProto(node, proto);

  addValueAttribute(proto, "keepdims", 0);
  outputsToProto(node, graph, proto);

  return Error::success();
}

Error ONNXModelWriter::writeBatchNormalization(
    const BatchNormalizationNode *node, GraphType &graph) {
  auto *proto = graph.add_node();
  // Add dictionary entries.
  addValueAttribute(proto, "epsilon", node->getEpsilon());
  addValueAttribute(proto, "momentum", node->getMomentum());

  proto->set_name(node->getName().str());
  proto->set_op_type("BatchNormalization");

  proto->add_input(node->getInput().getNode()->getName().str());
  proto->add_input(node->getScale().getNode()->getName().str());
  proto->add_input(node->getBias().getNode()->getName().str());
  proto->add_input(node->getMean().getNode()->getName().str());
  proto->add_input(node->getVar().getNode()->getName().str());

  outputsToProto(node, graph, proto);
  return Error::success();
}

Error ONNXModelWriter::writeInstanceNormalization(
    const InstanceNormalizationNode *node, GraphType &graph) {
  auto *proto = graph.add_node();
  // Add dictionary entries.
  addValueAttribute(proto, "epsilon", node->getEpsilon());

  proto->set_name(node->getName().str());
  proto->set_op_type("InstanceNormalization");

  proto->add_input(node->getInput().getNode()->getName().str());
  proto->add_input(node->getScale().getNode()->getName().str());
  proto->add_input(node->getBias().getNode()->getName().str());

  outputsToProto(node, graph, proto);
  return Error::success();
}

Error ONNXModelWriter::writeLayerNormalization(
    const LayerNormalizationNode *node, GraphType &graph) {
  auto *proto = graph.add_node();
  // Add dictionary entries.
  addValueAttribute(proto, "epsilon", node->getEpsilon());

  proto->set_name(node->getName().str());
  proto->set_op_type("LayerNormalization");

  proto->add_input(node->getInput().getNode()->getName().str());
  proto->add_input(node->getScale().getNode()->getName().str());
  proto->add_input(node->getBias().getNode()->getName().str());

  outputsToProto(node, graph, proto);
  return Error::success();
}

Error ONNXModelWriter::writeMeanVarNormalization(
    const MeanVarNormalizationNode *node, GraphType &graph) {
  auto *proto = graph.add_node();
  // Add dictionary entries.
  addValueAttribute(proto, "channel", node->getChannelIdx());
  addValueAttribute(proto, "momentum", node->getMomentum());

  proto->set_name(node->getName().str());
  proto->set_op_type("MeanVarianceNormalization");

  inputsToProto(node, proto);
  outputsToProto(node, graph, proto);
  return Error::success();
}

Error ONNXModelWriter::writeSlice(const SliceNode *node, GraphType &graph) {
  auto *proto = graph.add_node();
  // Add dictionary entries.
  auto starts = node->getStart();
  auto outs = node->getResult().dims();
  RETURN_ERR_IF_NOT(starts.size() == outs.size(),
                    "Mismatch starts and result dimensions.");

  RETURN_IF_ERR(writeAllWithNode("Slice", node, graph, proto));

  if (opsetVersion_ >= 10) {
    Tensor oneDimTensorStarts(ElemKind::Int64ITy, {(dim_t)starts.size()});
    auto handleStarts = oneDimTensorStarts.getHandle<int64_t>();
    Tensor oneDimTensorEnds(ElemKind::Int64ITy, {(dim_t)starts.size()});
    auto handleEnds = oneDimTensorEnds.getHandle<int64_t>();

    for (size_t b = 0, e = starts.size(); b < e; ++b) {
      handleStarts.raw(b) = starts[b];
      handleEnds.raw(b) = outs[b] + starts[b];
    }

    auto *tensorProto = addInitializer(graph);
    tensorProto->set_name(node->getName().str() + "_starts");
    writeTensor(oneDimTensorStarts, tensorProto, useGlowCustomOps_);
    proto->add_input(node->getName().str() + "_starts");

    tensorProto = addInitializer(graph);
    tensorProto->set_name(node->getName().str() + "_ends");
    writeTensor(oneDimTensorEnds, tensorProto, useGlowCustomOps_);
    proto->add_input(node->getName().str() + "_ends");
  } else {
    auto *attrStarts = proto->add_attribute();
    attrStarts->set_name("starts");
    attrStarts->set_type(AttrType::INTS);
    auto *attrEnds = proto->add_attribute();
    attrEnds->set_name("ends");
    attrEnds->set_type(AttrType::INTS);

    for (unsigned b = 0, e = starts.size(); b < e; ++b) {
      attrStarts->add_ints(starts[b]);
      attrEnds->add_ints(outs[b] + starts[b]);
    }
  }
  return Error::success();
}

Error ONNXModelWriter::writePow(const PowNode *node, GraphType &graph) {
  auto *proto = graph.add_node();
  proto->set_name(node->getName().str());
  proto->add_input(node->getLHS().getNode()->getName().str());
  outputsToProto(node, graph, proto);

  // Find exponent from splat node
  const auto *RHSN = node->getRHS().getNode();
  switch (RHSN->getKind()) {
  case Kinded::Kind::SplatNodeKind: {
    const auto *SN = llvm::cast<SplatNode>(RHSN);
    float value = SN->getValue();
    if (value == 0.5f) {
      proto->set_op_type("Sqrt");
    } else if (value == -1.0f) {
      proto->set_op_type("Reciprocal");
    } else if (value == 2.0f) {
      proto->set_op_type("Sqr");
    } else {
      return MAKE_ERR("Splat Node Value is invalid.");
    }
    break;
  }
  default:
    proto->add_input(RHSN->getName().str());
    break;
  }

  reportedNodes_.insert(RHSN);
  return Error::success();
}

Error ONNXModelWriter::writeTopK(const TopKNode *node, GraphType &graph) {
  auto *proto = graph.add_node();

  Tensor scalar(ElemKind::Int64ITy, {1});
  auto handle = scalar.getHandle<int64_t>();
  handle.raw(0) = node->getK();

  auto *tensorProto = addInitializer(graph);
  tensorProto->set_name("k");
  writeTensor(scalar, tensorProto, useGlowCustomOps_);

  RETURN_IF_ERR(writeAllWithNode("TopK", node, graph, proto));

  proto->add_input("k");
  return Error::success();
}

Error ONNXModelWriter::writeArgMax(const ArgMaxNode *node, GraphType &graph) {
  auto *proto = graph.add_node();

  Tensor axis(ElemKind::Int64ITy, {1});
  Tensor keepDims(ElemKind::BoolTy, {1});
  auto axisH = axis.getHandle<int64_t>();
  auto keepDimsH = keepDims.getHandle<int8_t>();
  axisH.raw(0) = node->getAxis();
  keepDimsH.raw(0) = node->getKeepDims();

  auto *tensorProto = addInitializer(graph);
  tensorProto->set_name("axis");
  writeTensor(axis, tensorProto, useGlowCustomOps_);

  tensorProto = addInitializer(graph);
  tensorProto->set_name("keepDims");
  writeTensor(keepDims, tensorProto, useGlowCustomOps_);
  RETURN_IF_ERR(writeAllWithNode("ArgMax", node, graph, proto));

  return Error::success();
}

Error ONNXModelWriter::writeArgMin(const ArgMinNode *node, GraphType &graph) {
  auto *proto = graph.add_node();

  Tensor axis(ElemKind::Int64ITy, {1});
  Tensor keepDims(ElemKind::BoolTy, {1});
  auto axisH = axis.getHandle<int64_t>();
  auto keepDimsH = keepDims.getHandle<int8_t>();
  axisH.raw(0) = node->getAxis();
  keepDimsH.raw(0) = node->getKeepDims();

  auto *tensorProto = addInitializer(graph);
  tensorProto->set_name("axis");
  writeTensor(axis, tensorProto, useGlowCustomOps_);

  tensorProto = addInitializer(graph);
  tensorProto->set_name("keepDims");
  writeTensor(keepDims, tensorProto, useGlowCustomOps_);
  RETURN_IF_ERR(writeAllWithNode("ArgMin", node, graph, proto));

  return Error::success();
}

Error ONNXModelWriter::writePRelu(const PReluNode *node, GraphType &graph) {
  auto *proto = graph.add_node();
  proto->set_name(node->getName().str());
  proto->set_op_type("PRelu");
  proto->add_input(node->getInput().getNode()->getName().str());

  const auto *slope = node->getSlope().getNode();
  if (const auto *BN = llvm::dyn_cast<BroadcastNode>(slope)) {
    proto->add_input(BN->getInput().getNode()->getName().str());
    reportedNodes_.insert(BN);
  } else if (const SplatNode *SN = llvm::dyn_cast<SplatNode>(slope)) {
    // Conversion a scalar to a tensor is required.
    Tensor scalar = {SN->getValue()};
    auto *tensorProto = addInitializer(graph);
    tensorProto->set_name(SN->getName().str());
    writeTensor(scalar, tensorProto, useGlowCustomOps_);
    proto->add_input(SN->getName().str());
    reportedNodes_.insert(SN);
  } else {
    return MAKE_ERR("Can't find Splat/Broadcast Node as part of PRelu Node.");
  }

  outputsToProto(node, graph, proto);
  return Error::success();
}

Error ONNXModelWriter::writeGather(const GatherNode *node, GraphType &graph) {
  auto *proto = graph.add_node();
  // Add dictionary entries.
  auto axis = node->getBatchDims();

  if (axis != 0) {
    addValueAttribute(proto, "axis", axis);
    return writeAllWithNode("BatchGather", node, graph, proto);
  } else {
    return writeAllWithNode("Gather", node, graph, proto);
  }
}

Error ONNXModelWriter::writeGatherElements(const GatherElementsNode *node,
                                           GraphType &graph) {
  auto *proto = graph.add_node();
  // Add dictionary entries.
  return writeAllWithNode("GatherElements", node, graph, proto);
}

Error ONNXModelWriter::writeGatherND(const GatherNDNode *node,
                                     GraphType &graph) {
  auto *proto = graph.add_node();
  // Add dictionary entries.
  return writeAllWithNode("GatherND", node, graph, proto);
}

Error ONNXModelWriter::writeMatMul(const MatMulNode *node, GraphType &graph) {
  return writeMatMulKind(node, graph, "MatMul");
}

Error ONNXModelWriter::writeBatchMatMul(const BatchMatMulNode *node,
                                        GraphType &graph) {
  auto dimSize = node->getLHS().dims().size();
  if (dimSize == 2) {
    return writeMatMulKind(node, graph, "MatMul");
  } else {
    return writeMatMulKind(node, graph, "BatchMatMul");
  }
}

Error ONNXModelWriter::writeReshape(const ReshapeNode *node, GraphType &graph) {
  auto *proto = graph.add_node();

  // Converting arrayRef scale to a constant node
  auto dims = node->getDims();
  Tensor dimsTensor(ElemKind::Int64ITy, {(dim_t)dims.size()});
  auto handleDims = dimsTensor.getHandle<int64_t>();
  for (size_t b = 0, e = dims.size(); b < e; ++b) {
    handleDims.raw(b) = dims[b];
  }

  auto *tensorProto = addInitializer(graph);
  tensorProto->set_name(node->getName().str() + "_shape");
  writeTensor(dimsTensor, tensorProto, useGlowCustomOps_);

  RETURN_IF_ERR(writeAllWithNode("Reshape", node, graph, proto));
  proto->add_input(node->getName().str() + "_shape");
  return Error::success();
}

Error ONNXModelWriter::writeBucketize(const BucketizeNode *node,
                                      GraphType &graph) {
  auto *proto = graph.add_node();
  // Add dictionary entries.
  addValueAttribute(proto, "boundaries", node->getBoundaries());

  return writeAllWithNode("Bucketize", node, graph, proto);
}

Error ONNXModelWriter::writeResizeNearest(const ResizeNearestNode *node,
                                          GraphType &graph) {
  auto *proto = graph.add_node();
  // Converting arrayRef scale to a constant node
  auto scale = node->getScale();
  Tensor scaleTensor(ElemKind::FloatTy, {(dim_t)scale.size()});
  auto handleScale = scaleTensor.getHandle<float>();
  for (size_t b = 0, e = scale.size(); b < e; ++b) {
    handleScale.raw(b) = scale[b];
  }

  auto *tensorProto = addInitializer(graph);
  tensorProto->set_name(node->getName().str() + "_scale");
  writeTensor(scaleTensor, tensorProto, useGlowCustomOps_);

  // Add dictionary entries.
  addValueAttribute(proto, "coordinate_transformation_mode",
                    std::string("asymmetric"));
  addValueAttribute(proto, "mode", std::string("nearest"));
  addValueAttribute(proto, "nearest_mode", std::string("floor"));

  RETURN_IF_ERR(writeAllWithNode("Resize", node, graph, proto));
  proto->add_input(node->getName().str() + "_scale");
  return Error::success();
}

Error ONNXModelWriter::writeResizeBilinear(const ResizeBilinearNode *node,
                                           GraphType &graph) {
  auto *proto = graph.add_node();
  // Converting arrayRef scale to a constant node
  auto scale = node->getScale();
  Tensor scaleTensor(ElemKind::FloatTy, {(dim_t)scale.size()});
  auto handleScale = scaleTensor.getHandle<float>();
  for (size_t b = 0, e = scale.size(); b < e; ++b) {
    handleScale.raw(b) = scale[b];
  }

  auto *tensorProto = addInitializer(graph);
  tensorProto->set_name(node->getName().str() + "_scale");
  writeTensor(scaleTensor, tensorProto, useGlowCustomOps_);

  // Add dictionary entries.
  addValueAttribute(proto, "coordinate_transformation_mode",
                    std::string("asymmetric"));
  addValueAttribute(proto, "mode", std::string("linear"));

  RETURN_IF_ERR(writeAllWithNode("Resize", node, graph, proto));
  proto->add_input(node->getName().str() + "_scale");
  return Error::success();
}

Error ONNXModelWriter::writeSoftMax(const SoftMaxNode *node, GraphType &graph) {
  auto *proto = graph.add_node();
  proto->set_name(node->getName().str());
  proto->set_op_type("Softmax");
  outputsToProto(node, graph, proto);
  // Find input from Reshape node
  proto->add_input(node->getInput().getNode()->getName().str());

  // Mark selected input as visited.
  reportedNodes_.insert(node->getSelected().getNode());
  return Error::success();
}

Error ONNXModelWriter::writeLogSoftMax(const LogSoftMaxNode *node,
                                       GraphType &graph) {
  auto *proto = graph.add_node();
  proto->set_name(node->getName().str());
  proto->set_op_type("LogSoftmax");
  outputsToProto(node, graph, proto);
  // Find input from Reshape node
  proto->add_input(node->getInput().getNode()->getName().str());

  // Mark selected input as visited.
  reportedNodes_.insert(node->getSelected().getNode());
  return Error::success();
}

Error ONNXModelWriter::writeReplaceNaN(const ReplaceNaNNode *node,
                                       GraphType &graph) {
  auto *proto = graph.add_node();
  // Add dictionary entries.
  float value = node->getValue();
  if (value != 0.0f) {
    addValueAttribute(proto, "value", value);
  }
  return writeAllWithNode("ReplaceNaN", node, graph, proto);
}

Error ONNXModelWriter::writeGatherRanges(const GatherRangesNode *node,
                                         GraphType &graph) {
  auto *proto = graph.add_node();
  // Add dictionary entries.
  addValueAttribute(proto, "maxOutputSize", node->getOutput().dims()[0]);

  return writeAllWithNode("GatherRanges", node, graph, proto);
}

Error ONNXModelWriter::writeSparseToDenseMask(const SparseToDenseMaskNode *node,
                                              GraphType &graph) {
  auto *proto = graph.add_node();
  // Add dictionary entries.
  addValueAttribute(proto, "mask", node->getMask());

  return writeAllWithNode("SparseToDenseMask", node, graph, proto);
}

Error ONNXModelWriter::writeAdaptiveAvgPool(const AdaptiveAvgPoolNode *node,
                                            GraphType &graph) {
  auto *proto = graph.add_node();

  // Add dictionary entries.
  const auto outShape = ShapeNHWC(node->getResult().dims());
  std::vector<size_t> output_size{outShape.h, outShape.w};
  addValueAttribute(proto, "output_size", llvm::makeArrayRef(output_size));

  auto err = writeAllWithNode("AdaptiveAvgPool", node, graph, proto);
  return err;
}

Error ONNXModelWriter::writeLocalResponseNormalization(
    const LocalResponseNormalizationNode *node, GraphType &graph) {
  auto *proto = graph.add_node();
  proto->set_name(node->getName().str());
  proto->set_op_type("LRN");
  outputsToProto(node, graph, proto);
  // Find input from Transpose node
  const TransposeNode *TN =
      llvm::dyn_cast<TransposeNode>(node->getInput().getNode());
  RETURN_ERR_IF_NOT(
      TN,
      "Can't find Transpose Node as part of LocalResponseNormalization Node.");
  proto->add_input(TN->getInput().getNode()->getName().str());
  reportedNodes_.insert(TN);
  // Add dictionary entries.
  addValueAttribute(proto, "size", 2 * node->getHalfWindowSize());
  addValueAttribute(proto, "alpha", node->getAlpha());
  addValueAttribute(proto, "beta", node->getBeta());
  addValueAttribute(proto, "bias", node->getK());

  return Error::success();
}

Error ONNXModelWriter::writeBatchBoxCox(const BatchBoxCoxNode *node,
                                        GraphType &graph) {
  auto *proto = graph.add_node();
  addValueAttribute(proto, "epsilon", node->getEpsilon());
  return writeAllWithNode("BatchBoxCox", node, graph, proto);
}

//===-----------------------------------------------------------------===//
//                    Operators Supported by Glow only
//===-----------------------------------------------------------------===//
Error ONNXModelWriter::writeModulo(const ModuloNode *node, GraphType &graph) {
  auto *proto = graph.add_node();
  // Add dictionary entries.
  addValueAttribute(proto, "divisor", node->getDivisor());
  addValueAttribute(proto, "sign_follow_divisor",
                    node->getSignFollowDivisor() ? 1 : 0);

  return writeAllWithNode("Modulo", node, graph, proto);
}

namespace {
template <typename T>
void writeTensorwiseQuantizedPool(const T *node, const std::string &op,
                                  ONNX_TRAITS::GraphProto &graph,
                                  ReportedNodes &) {
  assert(node->getLayout() == NHWC && "can only write NHWC Pools");

  auto *proto = graph.add_node();

  // Add dictionary entries.
  addValueAttribute(proto, "kernel_shape", node->getKernels());
  addValueAttribute(proto, "strides", node->getStrides());
  addValueAttribute(proto, "pads", node->getPads());

  if (auto *APN = llvm::dyn_cast<AvgPoolNode>(node)) {
    addValueAttribute(proto, "count_include_pad", APN->getCountIncludePads());
    addValueAttribute(proto, "out_scale",
                      APN->getType(AvgPoolNode::ResultIdx)->getScale());
    addValueAttribute(proto, "out_offset",
                      APN->getType(AvgPoolNode::ResultIdx)->getOffset());
  } else if (auto *MPN = llvm::dyn_cast<MaxPoolNode>(node)) {
    addValueAttribute(proto, "out_scale",
                      MPN->getType(MaxPoolNode::ResultIdx)->getScale());
    addValueAttribute(proto, "out_offset",
                      MPN->getType(MaxPoolNode::ResultIdx)->getOffset());
  }

  proto->add_input(node->getInput().getNode()->getName().str());
  outputsToProto(node, graph, proto);

  proto->set_name(node->getName().str());
  proto->set_op_type(op);
}

template <typename T>
void writePool(const T *node, const std::string &op,
               ONNX_TRAITS::GraphProto &graph, ReportedNodes &reporter) {
  // Delegate writing quantized pool ops to writeTensorwiseQuantizedPool.
  if (isQuantizedElemKind(node->getInput().getElementType())) {
    return writeTensorwiseQuantizedPool(node, op, graph, reporter);
  }

  // Loading pools creates a sandwich with Transpose nodes for Input
  // and Result. The lowering algorithm can remove Transpose nodes and
  // replace one set of nodes with another ones. When saving a graph to ONNX
  // format, keep in mind that when it will be loaded again a Transpose nodes
  // sandwich will be created again. The steps will be:
  // Remove Transpose node for Input, if such Transpose is not
  // found (they are supposed to be NCHW2NHWC then create a "mirror"
  // Transpose, i.e. NHWC2NCHW for correspondent Input or/and Weights.
  // The similar algorithm will be applied for Result. If Transpose NHWC2NCHW
  // node is found for Result user then remove it, otherwise create a "mirror"
  // Transpose, i.e. NCHW2NHWC.
  assert((node->getLayout() == NHWC || node->getLayout() == NTHWC) &&
         "can only write NHWC (2D) or NTHWC (3D) Pool ops");

  auto *proto = graph.add_node();

  // Use the output of transpose node.
  if (!outputKindToProto(Kinded::Kind::TransposeNodeKind, node, graph, proto)) {
    // Apparently Result Transpose has been removed, add NCHW2NHWC Transpose.
    writeTransposeResult(node, proto, graph);
  }

  // Add dictionary entries.
  addValueAttribute(proto, "kernel_shape", node->getKernels());
  addValueAttribute(proto, "strides", node->getStrides());
  addValueAttribute(proto, "pads", node->getPads());

  if (auto *APN = llvm::dyn_cast<AvgPoolNode>(node)) {
    addValueAttribute(proto, "count_include_pad", APN->getCountIncludePads());
  }

  const Node *input = node->getInput().getNode();
  if (const TransposeNode *TN = llvm::dyn_cast<TransposeNode>(input)) {
    proto->add_input(TN->getInput().getNode()->getName().str());
    reporter.insert(TN);
  } else if (const ReshapeNode *RSN = llvm::dyn_cast<ReshapeNode>(input)) {
    proto->add_input(RSN->getInput().getNode()->getName().str());
    reporter.insert(RSN);
  } else {
    writeTransposeInput(node, input, proto, graph);
  }

  proto->set_name(node->getName().str());
  proto->set_op_type(op);
}
} // namespace

Error ONNXModelWriter::writeAvgPool(const AvgPoolNode *node, GraphType &graph) {
  writePool(node, "AveragePool", graph, reportedNodes_);
  return Error::success();
}

Error ONNXModelWriter::writeMaxPool(const MaxPoolNode *node, GraphType &graph) {
  writePool(node, "MaxPool", graph, reportedNodes_);
  return Error::success();
}

Error ONNXModelWriter::writeConvolution3D(const Convolution3DNode *node,
                                          GraphType &graph) {
  // Loading convolution creates a sandwich with Transpose nodes for Input,
  // Weights, and Result. The lowering algorithm can remove Transpose nodes and
  // replace one set of nodes with another ones. When saving a graph to ONNX
  // format, keep in mind that when it will be loaded again a Transpose nodes
  // sandwich will be created again. The steps will be:
  // Remove Transpose nodes for Input and Weights, if such Transpose are not
  // found (they are supposed to be NCTHW2NTHWC then create a "mirror"
  // Transpose, i.e. NTHWC2NCTHW for correspondent Input or/and Weights.
  // The similar algorithm will be applied for Result. If Transpose NTHWC2NCTHW
  // node is found for Result user then remove it, otherwise create a "mirror"
  // Transpose, i.e. NCTHW2NTHWC.
  // assert(node->getLayout() == NTHWC && "can only write NTHWC Convolutions");

  // Delegate writing quantized Convs to writeTensorwiseQuantizedConvolution.
  if (isQuantizedElemKind(node->getInput().getElementType())) {
    return MAKE_ERR("Not implemented");
    // return writeTensorwiseQuantizedConvolution(node, graph);
  }

  auto *proto = graph.add_node();

  // Use the output of transpose node.
  if (!outputKindToProto(Kinded::Kind::TransposeNodeKind, node, graph, proto)) {
    // Apparently Result Transpose has been removed, add NCTHW2NTHWC Transpose.
    writeTransposeResult(node, proto, graph, NCTHW2NTHWC);
  }

  // Add dictionary entries.
  addValueAttribute(proto, "kernel_shape", node->getKernels());
  addValueAttribute(proto, "strides", node->getStrides());
  addValueAttribute(proto, "pads", node->getPads());
  addValueAttribute(proto, "group", node->getGroup());
  // addValueAttribute(proto, "dilations", node->getDilation());

  const Node *input = node->getInput().getNode();
  if (const TransposeNode *TN = llvm::dyn_cast<TransposeNode>(input)) {
    proto->add_input(TN->getInput().getNode()->getName().str());
    reportedNodes_.insert(TN);
  } else if (const ReshapeNode *RSN = llvm::dyn_cast<ReshapeNode>(input)) {
    proto->add_input(RSN->getInput().getNode()->getName().str());
    reportedNodes_.insert(RSN);
  } else {
    writeTransposeInput(node, input, proto, graph, NTHWC2NCTHW);
  }

  const Node *filter = node->getFilter().getNode();
  if (const TransposeNode *TN = llvm::dyn_cast<TransposeNode>(filter)) {
    proto->add_input(TN->getInput().getNode()->getName().str());
    reportedNodes_.insert(TN);
  } else if (const ReshapeNode *RSN = llvm::dyn_cast<ReshapeNode>(filter)) {
    proto->add_input(RSN->getInput().getNode()->getName().str());
    reportedNodes_.insert(RSN);
  } else {
    writeTransposeInput(node, filter, proto, graph, NTHWC2NCTHW);
  }

  proto->add_input(node->getBias().getNode()->getName().str());

  proto->set_name(node->getName().str());
  proto->set_op_type("Conv");

  return Error::success();
}

Error ONNXModelWriter::writeSpaceToDepth(const SpaceToDepthNode *node,
                                         GraphType &graph) {
  auto *proto = graph.add_node();

  // Find input from Transpose node
  const TransposeNode *TN =
      llvm::dyn_cast<TransposeNode>(node->getInput().getNode());
  RETURN_ERR_IF_NOT(TN,
                    "Can't find Transpose Node as part of SpaceToDepth Node.");
  proto->add_input(TN->getInput().getNode()->getName().str());
  reportedNodes_.insert(TN);

  proto->set_name(node->getName().str());
  proto->set_op_type("SpaceToDepth");
  // Add dictionary entries.
  addValueAttribute(proto, "blocksize", node->getBlockSize());

  // Use the output of transpose node, if any.
  if (!outputKindToProto(Kinded::Kind::TransposeNodeKind, node, graph, proto)) {
    outputsToProto(node, graph, proto);
  }
  return Error::success();
}

Error ONNXModelWriter::writeChannelShuffle(const ChannelShuffleNode *node,
                                           GraphType &graph) {
  auto *proto = graph.add_node();
  // Add dictionary entries.
  addValueAttribute(proto, "group", node->getGroup());
  addValueAttribute(proto, "kernel", node->getKernel());

  return writeAllWithNode("ChannelShuffle", node, graph, proto);
}

Error ONNXModelWriter::writeQuantizationProfile(
    const QuantizationProfileNode *node, GraphType &graph) {
  auto *proto = graph.add_node();
  // Add dictionary entries.
  addValueAttribute(proto, "name", node->getProfiledNodeName());
  addValueAttribute(proto, "number", node->getProfiledOutputNumber());

  return writeAllWithNode("QuantizationProfile", node, graph, proto);
}

Error ONNXModelWriter::writeTraceEvent(const TraceEventNode *node,
                                       GraphType &graph) {
  auto *proto = graph.add_node();
  // Add dictionary entries.
  addValueAttribute(proto, "name", node->getEventName());
  addValueAttribute(proto, "type", node->getEventType());
  addValueAttribute(proto, "index", node->getIndex());

  return writeAllWithNode("TraceEvent", node, graph, proto);
}

Error ONNXModelWriter::writeInsertTensor(const InsertTensorNode *node,
                                         GraphType &graph) {
  auto *proto = graph.add_node();
  // Add dictionary entries.
  addValueAttribute(proto, "start", node->getStart());
  addValueAttribute(proto, "count", node->getCount());
  addValueAttribute(proto, "axis", node->getAxis());

  return writeAllWithNode("InsertTensor", node, graph, proto);
}

Error ONNXModelWriter::writeSplat(const SplatNode *node, GraphType &graph) {
  // Conversion a scalar to a tensor is required.
  Tensor tensor(ElemKind::FloatTy, node->getResult().dims());
  auto handle = tensor.getHandle<>();
  float value = node->getValue();
  for (size_t b = 0, e = tensor.size(); b < e; ++b) {
    handle.raw(b) = value;
  }

  auto *tensorProto = addInitializer(graph);

  findOutputNames(node, graph, [&](const std::string &name) {
    tensorProto->set_name(name);
  });

  writeTensor(tensor, tensorProto, useGlowCustomOps_);
  reportedNodes_.insert(node);

  return Error::success();
}

Error ONNXModelWriter::writeTouch(const TouchNode *node, GraphType &graph) {
  auto *proto = graph.add_node();
  return writeAllWithNode("Touch", node, graph, proto);
}

// Exporting arithmetic node which may involve broadcasting.
// Broadcast Node will be unwind.
#define ARITHMETIC_NODE_WRITER(ONNXNAME, GLOWNAME)                             \
  Error ONNXModelWriter::write##GLOWNAME(const GLOWNAME##Node *node,           \
                                         GraphType &graph) {                   \
    return writeArithmetic(#ONNXNAME, node, graph, reportedNodes_,             \
                           hasMultidirectionalBroadcast(#ONNXNAME));           \
  }

ARITHMETIC_NODE_WRITER(Add, Add);
ARITHMETIC_NODE_WRITER(Sub, Sub);
ARITHMETIC_NODE_WRITER(Mul, Mul);
ARITHMETIC_NODE_WRITER(Div, Div);
ARITHMETIC_NODE_WRITER(Equal, CmpEQ)
ARITHMETIC_NODE_WRITER(And, And)
ARITHMETIC_NODE_WRITER(Or, Or)
ARITHMETIC_NODE_WRITER(Xor, Xor)
ARITHMETIC_NODE_WRITER(Less, CmpLT)

// Ops that Onnx doesn't have
ARITHMETIC_NODE_WRITER(CmpLTE, CmpLTE)
ARITHMETIC_NODE_WRITER(FloorDiv, FloorDiv);
ARITHMETIC_NODE_WRITER(Fmod, Fmod)
ARITHMETIC_NODE_WRITER(BitwiseAnd, BitwiseAnd)
ARITHMETIC_NODE_WRITER(BitwiseOr, BitwiseOr)
ARITHMETIC_NODE_WRITER(BitwiseXor, BitwiseXor)
#undef ARITHMETIC_NODE_WRITER

// Default exporting algorithm.
#define DEF_ALL_WRITER_NODE(NAME)                                              \
  Error ONNXModelWriter::write##NAME(const NAME##Node *node,                   \
                                     GraphType &graph) {                       \
    return writeAll(#NAME, node, graph);                                       \
  }

// ONNX nodes with default exporting algorithm.
DEF_ALL_WRITER_NODE(Not)
DEF_ALL_WRITER_NODE(Abs)
DEF_ALL_WRITER_NODE(Neg)
DEF_ALL_WRITER_NODE(Floor)
DEF_ALL_WRITER_NODE(Sign)
DEF_ALL_WRITER_NODE(Ceil)
DEF_ALL_WRITER_NODE(Round)
DEF_ALL_WRITER_NODE(Sqrt)
DEF_ALL_WRITER_NODE(Rsqrt)
DEF_ALL_WRITER_NODE(Reciprocal)
DEF_ALL_WRITER_NODE(Sin)
DEF_ALL_WRITER_NODE(Cos)
DEF_ALL_WRITER_NODE(LSTMUnit)
DEF_ALL_WRITER_NODE(DynamicQuantizedFullyConnected)
DEF_ALL_WRITER_NODE(DynamicRowwiseQuantizedFullyConnected)
DEF_ALL_WRITER_NODE(Erf)
DEF_ALL_WRITER_NODE(Min)
DEF_ALL_WRITER_NODE(Max)
DEF_ALL_WRITER_NODE(Log)
DEF_ALL_WRITER_NODE(Asin)
DEF_ALL_WRITER_NODE(Acos)
DEF_ALL_WRITER_NODE(Atan)
DEF_ALL_WRITER_NODE(Exp)
DEF_ALL_WRITER_NODE(Relu)
DEF_ALL_WRITER_NODE(LeakyRelu)
DEF_ALL_WRITER_NODE(Gelu)
DEF_ALL_WRITER_NODE(Tanh)
DEF_ALL_WRITER_NODE(IsNaN)
DEF_ALL_WRITER_NODE(Sigmoid)
DEF_ALL_WRITER_NODE(Swish)
DEF_ALL_WRITER_NODE(SoftPlus)
DEF_ALL_WRITER_NODE(LengthsSum)
DEF_ALL_WRITER_NODE(BatchOneHot)
DEF_ALL_WRITER_NODE(LengthsToRanges)
DEF_ALL_WRITER_NODE(SparseLengthsSum)
DEF_ALL_WRITER_NODE(SparseLengthsWeightedSum)
DEF_ALL_WRITER_NODE(EmbeddingBag)
DEF_ALL_WRITER_NODE(Embedding)
DEF_ALL_WRITER_NODE(BitwiseNot)
DEF_ALL_WRITER_NODE(GaussianFill)
DEF_ALL_WRITER_NODE(NonZero)
DEF_ALL_WRITER_NODE(BatchSparseToDense)
DEF_ALL_WRITER_NODE(FillExamplesWithIndicator)

// Glow nodes with default exporting algorithm.
DEF_ALL_WRITER_NODE(CmpNEQ)
DEF_ALL_WRITER_NODE(BatchedAdd)
DEF_ALL_WRITER_NODE(BatchedMul)
DEF_ALL_WRITER_NODE(Dequantize)
DEF_ALL_WRITER_NODE(Regression)
DEF_ALL_WRITER_NODE(RowwiseQuantizedSparseLengthsWeightedSum)
DEF_ALL_WRITER_NODE(FusedRowwiseQuantizedSparseLengthsSum)
DEF_ALL_WRITER_NODE(EmbeddingBagByteRowwiseOffsets)
DEF_ALL_WRITER_NODE(FusedRowwiseQuantizedSparseLengthsWeightedSum)
DEF_ALL_WRITER_NODE(NonMaxSuppression)
DEF_ALL_WRITER_NODE(TFLiteDetectionPostProcess)
DEF_ALL_WRITER_NODE(HardSwish)
DEF_ALL_WRITER_NODE(ConvTranspose)
DEF_ALL_WRITER_NODE(Logit)
DEF_ALL_WRITER_NODE(Truncate)
DEF_ALL_WRITER_NODE(BatchedUnaryEmbeddingsBags)
DEF_ALL_WRITER_NODE(IntNBitSplitEmbeddingBags)
DEF_ALL_WRITER_NODE(IntNBitSplitEmbeddingWeightedBags)

Error ONNXModelWriter::writeClip(const ClipNode *node, GraphType &graph) {
  auto *proto = graph.add_node();
  addValueAttribute(proto, "min", node->getMin());
  addValueAttribute(proto, "max", node->getMax());
  return writeAllWithNode("Clip", node, graph, proto);
}

Error ONNXModelWriter::writeConvertTo(const ConvertToNode *node,
                                      GraphType &graph) {
  auto *proto = graph.add_node();

  // Add dictionary entries.
  TensorType ttype;
  for (auto d : node->getResult().dims()) {
    ttype.add_dims(d);
  }
  ttype.set_data_type(convertType(*node->getResult().getType()));
  auto *attr = proto->add_attribute();
  attr->set_name("shape");
  attr->mutable_t()->CopyFrom(ttype);

  return writeAllWithNode("ConvertTo", node, graph, proto);
}

Error ONNXModelWriter::writeSelect(const SelectNode *node, GraphType &graph) {
  auto *proto = graph.add_node();
  // Add dictionary entries.
  addValueAttribute(proto, "shape", node->getResult().dims());

  return writeAllWithNode("Select", node, graph, proto);
}

Error ONNXModelWriter::writeQuantize(const QuantizeNode *node,
                                     GraphType &graph) {
  auto *proto = graph.add_node();
  auto outTy = node->getResult().getType();

  // Add dictionary entries.
  addValueAttribute(proto, "scale", outTy->getScale());
  addValueAttribute(proto, "offset", outTy->getOffset());
  addValueAttribute(proto, "elem_kind", outTy->getElementName());

  return writeAllWithNode("Quantize", node, graph, proto);
}

Error ONNXModelWriter::writeIntLookupTable(const IntLookupTableNode *node,
                                           GraphType &graph) {
  auto *proto = graph.add_node();
  // Add dictionary entries.
  addValueAttribute(proto, "shape", node->getResult().dims());
  NodeValue mapping = node->getMapping();
  if (Constant *c = llvm::dyn_cast<Constant>(mapping.getNode())) {
    auto handle = c->getHandle<int8_t>();
    auto begin = &handle.raw(0);
    addValueAttribute(
        proto, "values",
        llvm::ArrayRef<int8_t>(begin, begin + handle.actualSize()));
  } else {
    return MAKE_ERR("Mapping must be a constant type.");
  }

  return writeAllWithNode("IntLookupTable", node, graph, proto);
}

Error ONNXModelWriter::writeLookupTable(const LookupTableNode *node,
                                        GraphType &graph) {
  auto *proto = graph.add_node();
  // Add dictionary entries.
  addValueAttribute(proto, "shape", node->getResult().dims());
  NodeValue table = node->getTable();
  if (Constant *c = llvm::dyn_cast<Constant>(table.getNode())) {
    auto handle = c->getHandle<int8_t>();
    auto begin = &handle.raw(0);
    addValueAttribute(
        proto, "values",
        llvm::ArrayRef<int8_t>(begin, begin + handle.actualSize()));
  } else {
    return MAKE_ERR("Mapping must be a constant type.");
  }

  return writeAllWithNode("LookupTable", node, graph, proto);
}

Error ONNXModelWriter::writeLengthsRangeFill(const LengthsRangeFillNode *node,
                                             GraphType &graph) {
  auto *proto = graph.add_node();
  // Add dictionary entries.
  addValueAttribute(proto, "size", node->getResult().dims()[0]);

  return writeAllWithNode("LengthsRangeFill", node, graph, proto);
}

Error ONNXModelWriter::writeRescaleQuantized(const RescaleQuantizedNode *node,
                                             GraphType &graph) {
  auto *proto = graph.add_node();
  auto outTy = node->getResult().getType();
  // Add dictionary entries.
  addValueAttribute(proto, "scale", outTy->getScale());
  addValueAttribute(proto, "offset", outTy->getOffset());

  return writeAllWithNode("RescaleQuantized", node, graph, proto);
}

Error ONNXModelWriter::writeGemm(const GemmNode *node, GraphType &graph) {
  auto *proto = graph.add_node();
  proto->set_name(node->getName().str());
  proto->set_op_type("Gemm");

  proto->add_input(node->getA().getNode()->getName().str());
  proto->add_input(node->getB().getNode()->getName().str());
  if (node->getC().getNode()) {
    proto->add_input(node->getC().getNode()->getName().str());
  }

  addValueAttribute(proto, "alpha", node->getAlpha());
  addValueAttribute(proto, "beta", node->getBeta());
  addValueAttribute(proto, "transA", node->getTransposeA());
  addValueAttribute(proto, "transB", node->getTransposeB());

  outputsToProto(node, graph, proto);
  return Error::success();
}

Error ONNXModelWriter::writeFullyConnected(const FullyConnectedNode *node,
                                           GraphType &graph) {
  auto *proto = graph.add_node();
  proto->set_name(node->getName().str());
  proto->set_op_type("FullyConnected");

  if (node->getInput().dims().size() != 2) {
    return MAKE_ERR("Don't support input dim other than 2");
  }
  proto->add_input(node->getInput().getNode()->getName().str());
  proto->add_input(node->getWeights().getNode()->getName().str());
  proto->add_input(node->getBias().getNode()->getName().str());
  outputsToProto(node, graph, proto);
  return Error::success();
}

Error ONNXModelWriter::writeVectorNorm(const VectorNormNode *node,
                                       GraphType &graph) {
  auto *proto = graph.add_node();

  // Add dictionary entries.
  addValueAttribute(proto, "axis", node->getAxis());

  proto->set_name(node->getName().str());
  proto->set_op_type("VectorNorm");
  inputsToProto(node, proto);

  // currently support p = 2 (Frobenius or i2)
  addValueAttribute(proto, "p", node->getP());

  outputsToProto(node, graph, proto);

  return Error::success();
}

Error ONNXModelWriter::writeRowwiseQuantizedFullyConnected(
    const RowwiseQuantizedFullyConnectedNode *node, GraphType &graph) {
  auto *proto = graph.add_node();

  // Add dictionary entries.
  addValueAttribute(
      proto, "out_scale",
      node->getType(RowwiseQuantizedFullyConnectedNode::ResultIdx)->getScale());
  addValueAttribute(proto, "out_offset",
                    node->getType(RowwiseQuantizedFullyConnectedNode::ResultIdx)
                        ->getOffset());

  return writeAllWithNode("RowwiseQuantizedFullyConnected", node, graph, proto);
}

Error ONNXModelWriter::writeTile(const TileNode *node, GraphType &graph) {
  auto *proto = graph.add_node();

  // unwind Tile
  std::vector<size_t> repeats;
  const TileNode *tile = unwindTile(node, &repeats, reportedNodes_);

  proto->set_name("Tile");
  proto->set_op_type("Tile");
  // Use inputs from top tile.
  inputsToProto(tile, proto);
  // Use outputs from bottom tile.
  outputsToProto(node, graph, proto);

  // Add node indices
  auto *indices = graph.add_node();
  indices->set_name("Constant");
  indices->set_op_type("Constant");
  indices->add_output(tile->getName().str() + "_indices");

  unsigned_t numDims = tile->getInput().dims().size();

  DCHECK(repeats.size() == numDims);

  // Add Tensor type attribute.
  addValueAttribute(indices, "value", llvm::makeArrayRef(repeats));
  // Add indices as input to the Tile node
  proto->add_input(tile->getName().str() + "_indices");

  return Error::success();
}

Error ONNXModelWriter::writeCumSum(const CumSumNode *node, GraphType &graph) {
  auto *proto = graph.add_node();
  // Add dictionary entries.
  addValueAttribute(proto, "axis", 0);
  addValueAttribute(proto, "exclusive", node->getExclusive());
  addValueAttribute(proto, "reverse", node->getReverse());

  return writeAllWithNode("CumSum", node, graph, proto);
}

Error ONNXModelWriter::writeScatterData(const ScatterDataNode *node,
                                        GraphType &graph) {
  auto *proto = graph.add_node();

  return writeAllWithNode("ScatterData", node, graph, proto);
}

// Unsupported for export Glow nodes.
#define DEF_UNSUPPORTED_STORAGE(NAME)                                          \
  Error ONNXModelWriter::write##NAME(const NAME *node, GraphType &) {          \
    return writeUnexpectedKind(node);                                          \
  }

// Helper nodes.
DEF_UNSUPPORTED_STORAGE(Placeholder)
DEF_UNSUPPORTED_STORAGE(Constant)
DEF_UNSUPPORTED_STORAGE(Storage)

// Unsupported for export Glow nodes.
#define DEF_UNSUPPORTED_NODE(NAME)                                             \
  Error ONNXModelWriter::write##NAME(const NAME##Node *node, GraphType &) {    \
    return writeUnexpectedKind(node);                                          \
  }

DEF_UNSUPPORTED_NODE(BatchedPairwiseDotProduct)
DEF_UNSUPPORTED_NODE(Broadcast)
DEF_UNSUPPORTED_NODE(SGD)
DEF_UNSUPPORTED_NODE(SparseLabelSplit)
// Artificial node.
DEF_UNSUPPORTED_NODE(Save)
DEF_UNSUPPORTED_NODE(ExternalFunctionCall)
// Gradient nodes.
DEF_UNSUPPORTED_NODE(AddGrad)
DEF_UNSUPPORTED_NODE(DivGrad)
DEF_UNSUPPORTED_NODE(MulGrad)
DEF_UNSUPPORTED_NODE(SubGrad)
DEF_UNSUPPORTED_NODE(ReluGrad)
DEF_UNSUPPORTED_NODE(TanhGrad)
DEF_UNSUPPORTED_NODE(AvgPoolGrad)
DEF_UNSUPPORTED_NODE(MaxPoolGrad)
DEF_UNSUPPORTED_NODE(SigmoidGrad)
DEF_UNSUPPORTED_NODE(SoftMaxGrad)
DEF_UNSUPPORTED_NODE(LogSoftMaxGrad)
DEF_UNSUPPORTED_NODE(RegressionGrad)
DEF_UNSUPPORTED_NODE(ConvolutionGrad)
DEF_UNSUPPORTED_NODE(CrossEntropyLoss)
DEF_UNSUPPORTED_NODE(Convolution3DGrad)
DEF_UNSUPPORTED_NODE(FullyConnectedGrad)
DEF_UNSUPPORTED_NODE(CrossEntropyLossGrad)
DEF_UNSUPPORTED_NODE(BatchNormalizationGrad)
DEF_UNSUPPORTED_NODE(SparseLengthsSumGrad)
DEF_UNSUPPORTED_NODE(SparseLengthsWeightedSumGrad)
DEF_UNSUPPORTED_NODE(SigmoidCrossEntropyWithLogits)
DEF_UNSUPPORTED_NODE(LocalResponseNormalizationGrad)
DEF_UNSUPPORTED_NODE(AdaptiveAvgPoolGrad)
DEF_UNSUPPORTED_NODE(BatchedPairwiseDotProductGrad)

// Include backend-specific ONNX model writers.
#include "glow/ONNXModelWriterIncludes.h"

Error ONNXModelWriter::writeGlowCustomOperator(const Node *node,
                                               GraphType &graph) {
  ONNX_NAMESPACE::NodeProto *opProto = nullptr;

  switch (node->getKind()) {
#include "glow/AutoGenNodesExport.h"
  default:
    return MAKE_ERR(
        strFormat("Unhandled Node for export: %s", node->getName().data()));
  }
  RETURN_ERR_IF_NOT(opProto, "Did not have valid opProto.");

  // If dumping a DAG then add partition names to each op that's written.
  if (dagMode_ && !isWritingConstFoldSubgraph()) {
    addValueAttribute(opProto, "partitionName", F_->getName().str());
  }

  // Check if there is backendSpecificNodeInfo for node, and if so include it.
  auto itF = backendSpecificNodeInfo_.find(node->getParent());
  if (itF != backendSpecificNodeInfo_.end()) {
    auto itN = itF->second.find(node);
    if (itN != itF->second.end()) {
      // We found backend-specific node info, so add it to the opProto.
      for (const auto &optValPair : itN->second) {
        addValueAttribute(opProto,
                          std::string(nodeOptSignifier) + "_" +
                              optValPair.getKey().data(),
                          optValPair.getValue());
      }
    }
  }

  return Error::success();
}

bool ONNXModelWriter::hasMultidirectionalBroadcast(
    const llvm::StringRef typeName) {
  // Before opset 7, broadcasting was unidirectional.
  if (opsetVersion_ > 6) {
    // List of ops that support multidirectional broadcast can be found at
    // https://github.com/onnx/onnx/blob/master/docs/Broadcasting.md
    if ((typeName == "Add") || (typeName == "Sub") || (typeName == "Mul") ||
        (typeName == "Div") || (typeName == "Equal") ||
        (typeName == "Greater") || (typeName == "Less") ||
        (typeName == "Max") || (typeName == "Mean") || (typeName == "Min") ||
        (typeName == "Or") || (typeName == "Pow") || (typeName == "Sum") ||
        (typeName == "Xor")) {
      return true;
    }
  }
  return false;
}

} // namespace glow
