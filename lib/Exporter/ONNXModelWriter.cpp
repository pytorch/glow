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
#include "glow/Support/ZipUtils.h"

#include "miniz.h"

namespace glow {
namespace {
template <bool IsInteger, typename T> struct AttributeAssigner {
  static void assign(ONNX_NAMESPACE::AttributeProto *attr, const T &container);
};

// Specialization for llvm::ArrayRef<T> container types
template <typename T> struct AttributeAssigner<false, llvm::ArrayRef<T>> {
  static void assign(ONNX_NAMESPACE::AttributeProto *attr,
                     const llvm::ArrayRef<T> &container) {
    attr->set_type(ONNX_NAMESPACE::AttributeProto::INTS);
    for (auto value : container) {
      attr->add_ints(value);
    }
  }
};

// Specialization for 1D, 1 element Tensor container types
template <> struct AttributeAssigner<false, Tensor> {
  static void assign(ONNX_NAMESPACE::AttributeProto *attr, const Tensor &T) {
    auto *proto = attr->mutable_t();
    ONNXModelWriter::writeTensor(T, proto);
    attr->set_type(ONNX_NAMESPACE::AttributeProto::TENSOR);
  }
};

// Specialization for string type
template <> struct AttributeAssigner<false, std::string> {
  static void assign(ONNX_NAMESPACE::AttributeProto *attr,
                     const std::string &container) {
    attr->set_type(ONNX_NAMESPACE::AttributeProto::STRING);
    attr->set_s(container);
  }
};

// Specialization for StringRef type
template <> struct AttributeAssigner<false, llvm::StringRef> {
  static void assign(ONNX_NAMESPACE::AttributeProto *attr,
                     const llvm::StringRef container) {
    attr->set_type(ONNX_NAMESPACE::AttributeProto::STRING);
    attr->set_s(container.str());
  }
};

// Specialization for float type
template <> struct AttributeAssigner<false, float> {
  static void assign(ONNX_NAMESPACE::AttributeProto *attr,
                     const float &container) {
    attr->set_type(ONNX_NAMESPACE::AttributeProto::FLOAT);
    attr->set_f(container);
  }
};

// Specialization for int type
template <typename T> struct AttributeAssigner<true, T> {
  static void assign(ONNX_NAMESPACE::AttributeProto *attr, const T &container) {
    attr->set_type(ONNX_NAMESPACE::AttributeProto::INT);
    attr->set_i(container);
  }
};

template <typename T>
void addValueAttribute(ONNX_NAMESPACE::NodeProto *proto,
                       const std::string &name, const T &container) {
  auto *attr = proto->add_attribute();
  attr->set_name(name);
  AttributeAssigner<std::numeric_limits<T>::is_integer, T>::assign(attr,
                                                                   container);
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

/// Helper function to recursively get inputs for the broadcast node.
/// Broadcast node get constructed as a chain of Reshape->Tile->...->Tile.
const Node *unwindBroadcastInput(const TileNode *tile,
                                 std::vector<size_t> *repeats,
                                 ReportedNodes &reporter) {
  tile = unwindTile(tile, repeats, reporter);
  DCHECK(tile);

  reporter.insert(tile);
  if (const ReshapeNode *RN =
          llvm::dyn_cast<ReshapeNode>(tile->getInput().getNode())) {
    return RN;
  } else {
    return nullptr;
  }
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
      callback(p.first->getPlaceholder()->getName());
      saveResNo.insert(p.second);
    } else {
      auto *proto = graph.add_node();
      proto->set_name(node->getName().str() + "_copy_" +
                      std::to_string(p.second));
      proto->set_op_type("Identity");
      proto->add_input(p.second == 0 ? node->getName().str()
                                     : (node->getName().str() + "_out_" +
                                        std::to_string(p.second)));
      proto->add_output(p.first->getPlaceholder()->getName());
    }
  }

  // write the other outputs, if any
  for (unsigned b = 0, e = node->getNumResults(); b < e; ++b) {
    if (saveResNo.count(b)) {
      continue;
    }
    if (b == 0) {
      callback(node->getName());
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
    auto name = NV.getNode()->getName();
    if (resNo) {
      proto->add_input(name.str() + "_out_" + std::to_string(b));
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
      proto->add_output(SN->getPlaceholder()->getName());
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
  proto->set_name(node->getName());
  proto->set_op_type(nodeKind);

  Node *LHS = node->getLHS().getNode();
  proto->add_input(LHS->getName());
  Node *RHS = node->getRHS().getNode();
  proto->add_input(RHS->getName());

  outputsToProto(node, graph, proto);
  return Error::success();
}

// Creates a Transpose Node as the Result of the \p node.
// Reuses given \p proto pointer and create a new proto adding it to \p graph.
template <typename T>
void writeTransposeResult(const T *node, ONNX_NAMESPACE::NodeProto *&proto,
                          ONNX_TRAITS::GraphProto &graph) {
  // Add dictionary entries.
  std::vector<unsigned_t> shuffle(NCHW2NHWC);
  llvm::ArrayRef<unsigned_t> container(shuffle);
  addValueAttribute(proto, "perm", container);
  // Re-use proto for Transpose node.
  auto newName = node->getName().str() + "_out_transpose";
  proto->set_name(newName);
  proto->set_op_type("Transpose");
  proto->add_input(newName);

  proto->add_output(node->getName());

  // T node proto.
  proto = graph.add_node();
  proto->add_output(newName);
}

// Creates a Transpose Node as the Input \p input of the \p node.
// Reuses given \p proto pointer and create a new proto adding it to \p graph.
void writeTransposeInput(const Node *node, const Node *input,
                         ONNX_NAMESPACE::NodeProto *proto,
                         ONNX_TRAITS::GraphProto &graph) {
  // Write "mirror" Transform input, i.e. NHWC2NCHW
  auto newName =
      node->getName().str() + "_" + input->getName().str() + "_in_transpose";
  auto *transformProto = graph.add_node();
  transformProto->set_name(newName);
  transformProto->set_op_type("Transpose");

  // Add dictionary entries.
  std::vector<unsigned_t> shuffle(NHWC2NCHW);
  llvm::ArrayRef<unsigned_t> container(shuffle);
  addValueAttribute(transformProto, "perm", container);
  transformProto->add_input(input->getName());
  transformProto->add_output(newName);
  proto->add_input(newName);
}
/// Writes Arithmetic operators with name \p opName from Node \p node into
/// provided graph protobuf \p graph, optionally reports intermediate nodes as
/// visited, signaling that such nodes must be ignored,
/// \returns error.
template <typename T>
Error writeArithmetic(const std::string &opName, const T *node,
                      ONNX_TRAITS::GraphProto &graph, ReportedNodes &reporter) {
  auto *proto = graph.add_node();
  proto->set_name(node->getName());
  proto->set_op_type(opName);
  outputsToProto(node, graph, proto);

  auto LHS = node->getLHS();
  if (const TileNode *TN = llvm::dyn_cast<TileNode>(LHS.getNode())) {
    reporter.insert(TN);
    const ReshapeNode *RN = llvm::dyn_cast<ReshapeNode>(
        unwindBroadcastInput(TN, nullptr /* repeats */, reporter));
    RETURN_ERR_IF_NOT(RN, "Can't unwind Tile node.");
    reporter.insert(RN);
    LHS = RN->getInput();
  }

  int axis = -1;
  auto RHS = node->getRHS();
  if (const TileNode *TN = llvm::dyn_cast<TileNode>(RHS.getNode())) {
    reporter.insert(TN);
    // unwind broadcast with tiles repeats.
    std::vector<size_t> repeats;
    const ReshapeNode *RN = llvm::dyn_cast<ReshapeNode>(
        unwindBroadcastInput(TN, &repeats, reporter));
    RETURN_ERR_IF_NOT(RN, "Can't unwind Tile node.");
    reporter.insert(RN);
    RHS = RN->getInput();

    if (LHS.dims() != RHS.dims()) {
      // Extract axis from available shapes, ie. input origin,
      // reshape and target repeats.
      llvm::ArrayRef<dim_t> origin = RHS.dims();
      llvm::ArrayRef<dim_t> reshape = RN->getDims();
      DCHECK(reshape.size() == repeats.size());
      DCHECK(repeats.size() >= origin.size());

      // Replace target repeats dimension if it equal 1 and the correspondent
      // reshape dimension isn't.
      for (size_t b = 0, e = repeats.size(); b < e; ++b) {
        if (repeats[b] == 1 && reshape[b] != 1) {
          repeats[b] = reshape[b];
        }
      }

      for (int b = 0, e = repeats.size() - origin.size(); b <= e && axis == -1;
           ++b) {
        axis = b;
        for (size_t i = 0; i < origin.size(); ++i) {
          if (origin[i] != repeats[i + b] &&
              (origin[i] != 1 || reshape[i + b] != 1)) {
            axis = -1;
            break;
          }
        }
      }
    }
  }

  proto->add_input(LHS.getNode()->getName());
  proto->add_input(RHS.getNode()->getName());

  // Check if the shapes of LHS and RHS are different and broadcast attribute is
  // required.
  if (LHS.dims() != RHS.dims()) {
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
    for (const auto *V : F.getParent()->getConstants()) {
      reverseOrder_.push_back(V);
      visited_.insert(V);
    }
    // Start visiting all root nodes, i.e. nodes that do not have any users.
    for (auto &N : F.getNodes()) {
      if (N.getNumUsers() == 0) {
        visitRecursively(&N);
      }
    }
  }

  void visitRecursively(const Node *N) {
    // Check is node has been visited already.
    if (visited_.count(N)) {
      return;
    }

    if (N->getKind() == Kinded::Kind::PlaceholderKind) {
      // For Placeholder don't visit uses.
      visited_.insert(N);
      reverseOrder_.push_back(N);
      return;
    }

    // First visit all users, if any.
    for (const auto &use : N->getUsers()) {
      const auto *UN = use.getUser();
      // Check vacancy.
      if (visited_.count(UN)) {
        continue;
      }

      visitRecursively(UN);

      if (!visited_.count(UN)) {
        visited_.insert(UN);
        reverseOrder_.push_back(UN);
      }
    }

    // Visit current node, if it's still vacant.
    if (!visited_.count(N)) {
      visited_.insert(N);
      reverseOrder_.push_back(N);
    }

    // And then visit inputs of the current node.
    for (unsigned b = 0, e = N->getNumInputs(); b < e; ++b) {
      visitRecursively(N->getNthInput(b).getNode());
    }
  }

public:
  explicit ReverseGraphWalker(Function &F) { visit(F); }

  llvm::ArrayRef<const Node *> getNodes() const { return reverseOrder_; }
};
} // namespace

ONNXModelWriter::ONNXModelWriter(const std::string &modelFilename, Function &F,
                                 size_t irVersion, size_t opsetVersion,
                                 Error *errPtr, bool textMode, bool zipMode)
    : CommonOperatorWriter(modelFilename, F, errPtr),
      opsetVersion_(opsetVersion), zipMode_(zipMode) {
  // If errPtr already contains an error then don't continue with constructor.
  if (errPtr && *errPtr) {
    return;
  }

  // Lambda to setup the ONNXModelWriter and return any Errors that were
  // raised.
  auto setup = [&]() -> Error {
    // Loop through all nodes, output Graph to Model protobuf.
    ONNX_NAMESPACE::ModelProto modelProto;
    modelProto.set_ir_version(irVersion);
    modelProto.set_producer_name("ONNXModelWriter");
    auto *opsetImportProto = modelProto.add_opset_import();
    opsetImportProto->set_version(opsetVersion);
    auto *graphProto = modelProto.mutable_graph();
    graphProto->set_name("glow");
    // Use pre order graph traversal.
    // If node is constant with uses, turned into "input" and create a tensor
    // If node is placeholder with uses, turned into "input" with tensor shape,
    // except the case when placeholder has use as SaveNode.
    // Otherwise call common operators method or special operators and write
    // protobuf inputs from node inputs and protobuf outputs from uses.

    ReverseGraphWalker visitor(G_);
    for (const auto *N : visitor.getNodes()) {
      if (reportedNodes_.count(N)) {
        continue;
      }

      const auto kind = N->getKind();
      // Handle placeholders cases.
      if (kind == Kinded::Kind::PlaceholderKind) {
        if (hasUsesOfKind(N, Kinded::Kind::SaveNodeKind)) {
          // Storage as an input to SaveNode - ignore it.
          continue;
        }
        const auto *PH = llvm::cast<Placeholder>(N);
        // Write global input, output only tensor shape.
        auto *inputProto = graphProto->add_input();
        tensorShapeFromPlaceholder(PH, inputProto);
      } else if (kind == Kinded::Kind::ConstantKind) {
        // Write global initializer, output tensor bytes.
        const auto *C = llvm::cast<Constant>(N);
        auto *tensorProto = addInitializer(*graphProto);
        tensorProto->set_name(C->getName());
        writeTensor(C->getPayload(), tensorProto);
      } else if (kind == Kinded::Kind::SaveNodeKind) {
        // Save node case, find input and use its name as a global output,
        // output only shape.
        const SaveNode *SN = llvm::cast<SaveNode>(N);
        const auto *PH = SN->getPlaceholder();
        auto *out = graphProto->add_output();
        tensorShapeFromPlaceholder(PH, out);
      } else {
        RETURN_IF_ERR(writeOperator(N, *graphProto));
      }
      reportedNodes_.insert(N);
    }

    // Nodes has been added in a reverse order from SaveNode up to the inputs,
    // we need to rearrange all nodes in the reverse order before serialization.
    auto *nodes = graphProto->mutable_node();
    for (size_t i = 0, n = nodes->size(); i < n / 2; ++i) {
      nodes->SwapElements(i, n - i - 1);
    }
    // We need to swap back Identity node with the next non-Identity node since
    // we append Identity node to tap out the intermediate results
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

    if (zipMode_) {
      const bool compressed = false;
      ZipWriter zip(&ff_, F.getName());
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
      if (textMode) {
        google::protobuf::TextFormat::PrintToString(modelProto, &largeBuffer);
      } else {
        modelProto.SerializeToString(&largeBuffer);
      }
      zip.writeRecord("model", largeBuffer.c_str(), largeBuffer.size(),
                      compressed);
      zip.writeEndOfFile();
      ff_.flush();
      ff_.close();
      return Error::success();
    } else {
      return writeModel(modelProto, textMode);
    }
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
  case ElemKind::Int8QTy:
    return TensorType::INT8;
  case ElemKind::UInt8FusedQTy:
  case ElemKind::UInt8FusedFP16QTy:
  case ElemKind::UInt4FusedFP16QTy:
  case ElemKind::UInt8QTy:
    return TensorType::UINT8;
  case ElemKind::Int16QTy:
    return TensorType::INT16;
  case ElemKind::Int32QTy:
  case ElemKind::Int32ITy:
    return TensorType::INT32;
  case ElemKind::Int64ITy:
    return TensorType::INT64;
  case ElemKind::BoolTy:
    return TensorType::BOOL;
  }
  LOG(DFATAL) << "Cannot reach here.";
}

void ONNXModelWriter::writeTensor(const Tensor &T, TensorType *out) {
  const auto &type = T.getType();
  out->set_data_type(convertType(type));
  const auto &dims = type.dims();
  for (unsigned b = 0, e = dims.size(); b < e; ++b) {
    out->add_dims(dims[b]);
  }

  out->set_raw_data(T.getUnsafePtr(), type.getSizeInBytes());

  if (type.isQuantizedType()) {
    // Format is ElemKind:scale:offset
    out->set_doc_string(strFormat("%s:%f:%d", type.getElementName().data(),
                                  T.getType().getScale(),
                                  T.getType().getOffset()));
  }
}

void ONNXModelWriter::tensorShapeFromPlaceholder(const Placeholder *PH,
                                                 ValueInfoType *valueProto) {
  tensorShapeFromInput(PH->getName(), PH->getType(), valueProto);
  if (PH->isStatic()) {
    valueProto->set_doc_string("offline");
  }
}

Error ONNXModelWriter::writeAllWithNode(const std::string &opName,
                                        const Node *node, GraphType &graph,
                                        NodeType *proto) {
  proto->set_name(node->getName());
  proto->set_op_type(opName);
  inputsToProto(node, proto);
  outputsToProto(node, graph, proto);
  return Error::success();
}

Error ONNXModelWriter::writeAll(const std::string &opName, const Node *node,
                                GraphType &graph) {
  return writeAllWithNode(opName, node, graph, graph.add_node());
}

bool ONNXModelWriter::hasUsesOfKind(const Node *node, Kinded::Kind kind) {
  for (const auto &use : node->getUsers()) {
    if (use.getUser()->getKind() == kind) {
      return true;
    }
  }
  return false;
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
    RETURN_ERR("Pad: Invalid mode",
               ErrorValue::ErrorCode::MODEL_WRITER_SERIALIZATION_ERROR);
  }

  addValueAttribute(proto, "pads", node->getPads());
  float value = node->getValue();
  if (value != .0f) {
    addValueAttribute(proto, "value", value);
  }

  return writeAllWithNode("Pad", node, graph, proto);
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

Error ONNXModelWriter::writeFlip(const FlipNode *node, GraphType &graph) {
  auto *proto = graph.add_node();
  // Add dictionary entries.
  addValueAttribute(proto, "axis", node->getAxis());

  return writeAllWithNode("Flip", node, graph, proto);
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
  std::vector<unsigned_t> buffer(2, node->getDilation());
  llvm::ArrayRef<unsigned_t> container(buffer);
  addValueAttribute(proto, "dilations", container);

  const Node *input = node->getInput().getNode();
  if (const TransposeNode *TN = llvm::dyn_cast<TransposeNode>(input)) {
    proto->add_input(TN->getInput().getNode()->getName());
    reportedNodes_.insert(TN);
  } else if (const ReshapeNode *RSN = llvm::dyn_cast<ReshapeNode>(input)) {
    proto->add_input(RSN->getInput().getNode()->getName());
    reportedNodes_.insert(RSN);
  } else {
    writeTransposeInput(node, input, proto, graph);
  }

  const Node *filter = node->getFilter().getNode();
  if (const TransposeNode *TN = llvm::dyn_cast<TransposeNode>(filter)) {
    proto->add_input(TN->getInput().getNode()->getName());
    reportedNodes_.insert(TN);
  } else if (const ReshapeNode *RSN = llvm::dyn_cast<ReshapeNode>(filter)) {
    proto->add_input(RSN->getInput().getNode()->getName());
    reportedNodes_.insert(RSN);
  } else {
    writeTransposeInput(node, filter, proto, graph);
  }

  proto->add_input(node->getBias().getNode()->getName());

  proto->set_name(node->getName());
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

  proto->set_name(node->getName());
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

  proto->set_name(node->getName());
  proto->set_op_type("ReduceSum");
  inputsToProto(node, proto);

  addValueAttribute(proto, "keepdims", 0);
  outputsToProto(node, graph, proto);

  return Error::success();
}

Error ONNXModelWriter::writeBatchedReduceMin(const BatchedReduceMinNode *node,
                                             GraphType &graph) {
  auto *proto = graph.add_node();
  // Find dictionary entries.
  addValueAttribute(proto, "axes", node->getAxes());

  return writeAllWithNode("ReduceMin", node, graph, proto);
}

Error ONNXModelWriter::writeBatchNormalization(
    const BatchNormalizationNode *node, GraphType &graph) {
  auto *proto = graph.add_node();
  // Add dictionary entries.
  addValueAttribute(proto, "epsilon", node->getEpsilon());
  addValueAttribute(proto, "momentum", node->getMomentum());

  proto->set_name(node->getName());
  proto->set_op_type("BatchNormalization");

  proto->add_input(node->getInput().getNode()->getName());
  proto->add_input(node->getScale().getNode()->getName());
  proto->add_input(node->getBias().getNode()->getName());
  proto->add_input(node->getMean().getNode()->getName());
  proto->add_input(node->getVar().getNode()->getName());

  outputsToProto(node, graph, proto);
  return Error::success();
}

Error ONNXModelWriter::writeLayerNormalization(
    const LayerNormalizationNode *node, GraphType &graph) {
  auto *proto = graph.add_node();
  // Add dictionary entries.
  addValueAttribute(proto, "epsilon", node->getEpsilon());

  proto->set_name(node->getName());
  proto->set_op_type("LayerNormalization");

  proto->add_input(node->getInput().getNode()->getName());
  proto->add_input(node->getScale().getNode()->getName());
  proto->add_input(node->getBias().getNode()->getName());

  outputsToProto(node, graph, proto);
  return Error::success();
}

Error ONNXModelWriter::writeMeanVarNormalization(
    const MeanVarNormalizationNode *node, GraphType &graph) {
  auto *proto = graph.add_node();
  // Add dictionary entries.
  addValueAttribute(proto, "channel", node->getChannelIdx());
  addValueAttribute(proto, "momentum", node->getMomentum());

  proto->set_name(node->getName());
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
    writeTensor(oneDimTensorStarts, tensorProto);
    proto->add_input(node->getName().str() + "_starts");

    tensorProto = addInitializer(graph);
    tensorProto->set_name(node->getName().str() + "_ends");
    writeTensor(oneDimTensorEnds, tensorProto);
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
  proto->set_name(node->getName());
  proto->add_input(node->getLHS().getNode()->getName());
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
      RETURN_ERR("Splat Node Value is invalid.");
    }
    break;
  }
  default:
    proto->add_input(RHSN->getName());
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
  writeTensor(scalar, tensorProto);

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
  writeTensor(axis, tensorProto);

  tensorProto = addInitializer(graph);
  tensorProto->set_name("keepDims");
  writeTensor(keepDims, tensorProto);
  RETURN_IF_ERR(writeAllWithNode("ArgMax", node, graph, proto));

  return Error::success();
}

Error ONNXModelWriter::writePRelu(const PReluNode *node, GraphType &graph) {
  auto *proto = graph.add_node();
  proto->set_name(node->getName());
  proto->set_op_type("PRelu");
  proto->add_input(node->getInput().getNode()->getName());

  const auto *slope = node->getSlope().getNode();
  if (const auto *tile = llvm::dyn_cast<TileNode>(slope)) {
    reportedNodes_.insert(tile);
    slope = unwindBroadcastInput(tile, nullptr /* repeats */, reportedNodes_);
  }
  if (const SplatNode *SN = llvm::dyn_cast<SplatNode>(slope)) {
    // Conversion a scalar to a tensor is required.
    Tensor scalar = {SN->getValue()};
    auto *tensorProto = addInitializer(graph);
    tensorProto->set_name(SN->getName());
    writeTensor(scalar, tensorProto);
    proto->add_input(SN->getName());
    reportedNodes_.insert(SN);
  } else if (const ReshapeNode *RN = llvm::dyn_cast<ReshapeNode>(slope)) {
    proto->add_input(RN->getInput().getNode()->getName());
    reportedNodes_.insert(RN);
  } else {
    RETURN_ERR("Can't find Splat/Reshape Node as part of PRelu Node.");
  }

  outputsToProto(node, graph, proto);
  return Error::success();
}

Error ONNXModelWriter::writeGather(const GatherNode *node, GraphType &graph) {
  auto *proto = graph.add_node();
  // Add dictionary entries.
  auto batchDims = node->getBatchDims();

  if (batchDims != 0) {
    addValueAttribute(proto, "axis", batchDims);
    return writeAllWithNode("BatchGather", node, graph, proto);
  } else {
    return writeAllWithNode("Gather", node, graph, proto);
  }
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

  // Add ints type attribute.
  addValueAttribute(proto, "shape", node->getDims());

  return writeAllWithNode("Reshape", node, graph, proto);
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
  // Find dictionary entries.
  addValueAttribute(proto, "height_scale", node->getHeightScale());
  addValueAttribute(proto, "width_scale", node->getWidthScale());

  return writeAllWithNode(node->getName(), node, graph, proto);
}

Error ONNXModelWriter::writeSoftMax(const SoftMaxNode *node, GraphType &graph) {
  auto *proto = graph.add_node();
  proto->set_name(node->getName());
  proto->set_op_type("Softmax");
  outputsToProto(node, graph, proto);
  // Find input from Reshape node
  proto->add_input(node->getInput().getNode()->getName());

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

  // Use the output of transpose node.
  if (!outputKindToProto(Kinded::Kind::TransposeNodeKind, node, graph, proto)) {
    // Apparently Result Transpose has been removed, add NCHW2NHWC Transpose.
    writeTransposeResult(node, proto, graph);
  }

  // Add dictionary entries.
  const auto outShape = ShapeNHWC(node->getResult().dims());
  std::vector<size_t> output_size{outShape.h, outShape.w};
  addValueAttribute(proto, "output_size", llvm::makeArrayRef(output_size));

  const Node *input = node->getInput().getNode();
  if (const TransposeNode *TN = llvm::dyn_cast<TransposeNode>(input)) {
    proto->add_input(TN->getInput().getNode()->getName());
    reportedNodes_.insert(TN);
  } else if (const ReshapeNode *RSN = llvm::dyn_cast<ReshapeNode>(input)) {
    proto->add_input(RSN->getInput().getNode()->getName());
    reportedNodes_.insert(RSN);
  } else {
    writeTransposeInput(node, input, proto, graph);
  }

  proto->set_name(node->getName());
  proto->set_op_type("AdaptiveAvgPool");

  return Error::success();
}

Error ONNXModelWriter::writeLocalResponseNormalization(
    const LocalResponseNormalizationNode *node, GraphType &graph) {
  auto *proto = graph.add_node();
  proto->set_name(node->getName());
  proto->set_op_type("LRN");
  outputsToProto(node, graph, proto);
  // Find input from Transpose node
  const TransposeNode *TN =
      llvm::dyn_cast<TransposeNode>(node->getInput().getNode());
  RETURN_ERR_IF_NOT(
      TN,
      "Can't find Transpose Node as part of LocalResponseNormalization Node.");
  proto->add_input(TN->getInput().getNode()->getName());
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
void writePool(const T *node, const std::string &op,
               ONNX_TRAITS::GraphProto &graph, ReportedNodes &reporter) {
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
  assert(node->getLayout() == NHWC && "can only write NHWC Pools");
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

  const Node *input = node->getInput().getNode();
  if (const TransposeNode *TN = llvm::dyn_cast<TransposeNode>(input)) {
    proto->add_input(TN->getInput().getNode()->getName());
    reporter.insert(TN);
  } else if (const ReshapeNode *RSN = llvm::dyn_cast<ReshapeNode>(input)) {
    proto->add_input(RSN->getInput().getNode()->getName());
    reporter.insert(RSN);
  } else {
    writeTransposeInput(node, input, proto, graph);
  }

  proto->set_name(node->getName());
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
  auto *proto = graph.add_node();
  // Add dictionary entries.
  addValueAttribute(proto, "kernel_shape", node->getKernels());
  addValueAttribute(proto, "strides", node->getStrides());
  addValueAttribute(proto, "pads", node->getPads());
  addValueAttribute(proto, "group", node->getGroup());

  return writeAllWithNode("Convolution3D", node, graph, proto);
}

Error ONNXModelWriter::writeSpaceToDepth(const SpaceToDepthNode *node,
                                         GraphType &graph) {
  auto *proto = graph.add_node();

  // Find input from Transpose node
  const TransposeNode *TN =
      llvm::dyn_cast<TransposeNode>(node->getInput().getNode());
  RETURN_ERR_IF_NOT(TN,
                    "Can't find Transpose Node as part of SpaceToDepth Node.");
  proto->add_input(TN->getInput().getNode()->getName());
  reportedNodes_.insert(TN);

  proto->set_name(node->getName());
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

  writeTensor(tensor, tensorProto);
  reportedNodes_.insert(node);

  return Error::success();
}

Error ONNXModelWriter::writeAdd(const AddNode *node, GraphType &graph) {
  return writeArithmetic("Add", node, graph, reportedNodes_);
}

Error ONNXModelWriter::writeDiv(const DivNode *node, GraphType &graph) {
  return writeArithmetic("Div", node, graph, reportedNodes_);
}

Error ONNXModelWriter::writeMul(const MulNode *node, GraphType &graph) {
  return writeArithmetic("Mul", node, graph, reportedNodes_);
}

Error ONNXModelWriter::writeSub(const SubNode *node, GraphType &graph) {
  return writeArithmetic("Sub", node, graph, reportedNodes_);
}

// Default exporting algorithm.
#define DEF_ALL_WRITER_NODE(NAME)                                              \
  Error ONNXModelWriter::write##NAME(const NAME##Node *node,                   \
                                     GraphType &graph) {                       \
    return writeAll(#NAME, node, graph);                                       \
  }

// ONNX nodes with default exporting algorithm.
DEF_ALL_WRITER_NODE(Min)
DEF_ALL_WRITER_NODE(Max)
DEF_ALL_WRITER_NODE(Log)
DEF_ALL_WRITER_NODE(Exp)
DEF_ALL_WRITER_NODE(Relu)
DEF_ALL_WRITER_NODE(Tanh)
DEF_ALL_WRITER_NODE(IsNaN)
DEF_ALL_WRITER_NODE(Sigmoid)
DEF_ALL_WRITER_NODE(LengthsSum)
DEF_ALL_WRITER_NODE(BatchOneHot)
DEF_ALL_WRITER_NODE(LengthsToRanges)
DEF_ALL_WRITER_NODE(SparseLengthsSum)
DEF_ALL_WRITER_NODE(SparseLengthsWeightedSum)
DEF_ALL_WRITER_NODE(EmbeddingBag)

// Glow nodes with default exporting algorithm.
DEF_ALL_WRITER_NODE(CmpEQ)
DEF_ALL_WRITER_NODE(CmpLTE)
DEF_ALL_WRITER_NODE(CmpLT)
DEF_ALL_WRITER_NODE(BatchedAdd)
DEF_ALL_WRITER_NODE(Dequantize)
DEF_ALL_WRITER_NODE(Regression)
DEF_ALL_WRITER_NODE(RowwiseQuantizedFullyConnected)
DEF_ALL_WRITER_NODE(RowwiseQuantizedSparseLengthsWeightedSum)
DEF_ALL_WRITER_NODE(FusedRowwiseQuantizedSparseLengthsSum)
DEF_ALL_WRITER_NODE(EmbeddingBagByteRowwiseOffsets)
DEF_ALL_WRITER_NODE(FusedRowwiseQuantizedSparseLengthsWeightedSum)
DEF_ALL_WRITER_NODE(NonMaxSuppression)

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
    RETURN_ERR("Mapping must be a constant type.");
  }

  return writeAllWithNode("IntLookupTable", node, graph, proto);
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

Error ONNXModelWriter::writeFullyConnected(const FullyConnectedNode *node,
                                           GraphType &graph) {
  auto *proto = graph.add_node();
  proto->set_name(node->getName());
  proto->set_op_type("FullyConnected");

  if (node->getInput().dims().size() != 2) {
    RETURN_ERR("Don't support input dim other than 2");
  }
  proto->add_input(node->getInput().getNode()->getName());
  proto->add_input(node->getWeights().getNode()->getName());
  proto->add_input(node->getBias().getNode()->getName());
  outputsToProto(node, graph, proto);
  return Error::success();
}

Error ONNXModelWriter::writeSparseToDense(const SparseToDenseNode *node,
                                          GraphType &graph) {
  auto *proto = graph.add_node();

  RETURN_IF_ERR(writeAllWithNode("SparseToDense", node, graph, proto));

  // Write dataToInferDim as additional input with initialization.
  auto values = node->getValues();
  auto out = node->getResult();
  ShapeVector outDims(values.dims().begin(), values.dims().end());
  outDims[0] = out.dims()[0];

  auto outTy =
      G_.getParent()->uniqueTypeWithNewShape(values.getType(), outDims);
  auto *inputProto = graph.add_input();
  tensorShapeFromInput("dataToInferDim", outTy, inputProto);
  proto->add_input("dataToInferDim");
  return Error::success();
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
  // Create and populate Tensor.
  Tensor oneDimTensor(ElemKind::Int64ITy, {numDims});
  auto handle = oneDimTensor.getHandle<int64_t>();

  for (size_t b = 0, e = repeats.size(); b < e; ++b) {
    handle.raw(b) = repeats[b];
  }

  // Add Tensor type attribute.
  addValueAttribute(indices, "value", oneDimTensor);
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

DEF_UNSUPPORTED_NODE(SGD)
// Artificial node.
DEF_UNSUPPORTED_NODE(Save)
// TODO: Turn to ScatterNd when it is supported in ONNX.
DEF_UNSUPPORTED_NODE(ScatterData)
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

// Include backend-specific ONNX model writers.
#include "glow/ONNXModelWriterIncludes.h"
} // namespace glow
