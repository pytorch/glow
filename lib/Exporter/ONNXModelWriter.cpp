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

#include "glow/Exporter/ONNXModelWriter.h"
#include "glow/Graph/Utils.h"

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
void outputsToProto(const Node *node, ONNX_NAMESPACE::NodeProto *proto) {
  for (const auto &use : node->getUsers()) {
    const auto *user = use.getUser();
    if (user->getKind() == Kinded::Kind::SaveNodeKind) {
      proto->add_output(user->getName());
    } else {
      // Find the user input that matches input node parameter.
      for (unsigned b = 0, e = user->getNumInputs(); b < e; ++b) {
        auto *UIN = user->getNthInput(b).getNode();
        if (UIN == node) {
          proto->add_output(UIN->getName());
          break;
        }
      }
    }
  }
}

/// Write the output of the provided type only of node outputs.
bool outputKindToProto(Kinded::Kind kind, const Node *node,
                       ONNX_NAMESPACE::NodeProto *proto) {
  bool found = false;
  for (const auto &use : node->getUsers()) {
    const auto *user = use.getUser();
    if (user->getKind() == Kinded::Kind::SaveNodeKind) {
      proto->add_output(user->getName());
    } else if (user->getKind() == kind) {
      found = true;
      outputsToProto(user, proto);
    }
  }
  return found;
}

/// Writes all inputs from Node \p node to protobuf \p proto.
void inputsToProto(const Node *node, ONNX_NAMESPACE::NodeProto *proto) {
  for (unsigned b = 0, e = node->getNumInputs(); b < e; ++b) {
    proto->add_input(node->getNthInput(b).getNode()->getName());
  }
}

/// Writes MatMul operators from Node \p node into
/// provided graph protobuf \p graph, optionally reports intermediate nodes as
/// visited, signaling that such nodes must be ignored,
/// \returns error.
template <typename T>
llvm::Error writeMatMulKind(const T *node, ONNX_TRAITS::GraphProto &graph,
                            ReportedNodes &reporter) {
  auto *proto = graph.add_node();
  proto->set_name(node->getName());
  proto->set_op_type("MatMul");

  // Check if LHS/RHS are Transpose node.
  Node *LHS = node->getLHS().getNode();
  TransposeNode *TLHS = llvm::dyn_cast<TransposeNode>(LHS);
  if (TLHS) {
    proto->add_input(TLHS->getInput().getNode()->getName());
    addValueAttribute(proto, "trans_a", 1U);
    reporter.insert(TLHS);
  } else {
    proto->add_input(LHS->getName());
  }

  Node *RHS = node->getRHS().getNode();
  TransposeNode *TRHS = llvm::dyn_cast<TransposeNode>(RHS);
  if (TRHS) {
    proto->add_input(TRHS->getInput().getNode()->getName());
    addValueAttribute(proto, "trans_b", 1U);
    reporter.insert(TRHS);
  } else {
    proto->add_input(RHS->getName());
  }

  outputsToProto(node, proto);
  return llvm::Error::success();
}

/// Writes Arithmetic operators with name \p opName from Node \p node into
/// provided graph protobuf \p graph, optionally reports intermediate nodes as
/// visited, signaling that such nodes must be ignored,
/// \returns error.
template <typename T>
llvm::Error writeArithmetic(const std::string &opName, const T *node,
                            ONNX_TRAITS::GraphProto &graph,
                            ReportedNodes &reporter) {
  auto *proto = graph.add_node();
  proto->set_name(node->getName());
  proto->set_op_type(opName);
  outputsToProto(node, proto);

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
      llvm::ArrayRef<size_t> origin = RHS.dims();
      llvm::ArrayRef<size_t> reshape = RN->getDims();
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

  return llvm::Error::success();
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

} // namespace

ONNXModelWriter::ONNXModelWriter(const std::string &modelFilename, Function &F,
                                 size_t irVersion, size_t opsetVersion,
                                 llvm::Error *errPtr, bool textMode)
    : CommonOperatorWriter(modelFilename, F, errPtr),
      opsetVersion_(opsetVersion) {
  // If errPtr already contains an error then don't continue with constructor.
  if (errPtr && *errPtr) {
    return;
  }

  // Lambda to setup the ONNXModelWriter and return any llvm::Errors that were
  // raised.
  auto setup = [&]() -> llvm::Error {
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
    GraphPreOrderVisitor visitor(G_);
    for (const auto *N : visitor.getPreOrder()) {
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
        auto *tensorProto = graphProto->add_initializer();
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

    return writeModel(modelProto, textMode);
  };

  if (errPtr) {
    *errPtr = setup();
  } else {
    EXIT_ON_ERR(setup());
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
}

void ONNXModelWriter::writeTensor(const Tensor &T, TensorType *out) {
  const auto &type = T.getType();
  out->set_data_type(convertType(type));
  const auto &dims = type.dims();
  for (unsigned b = 0, e = dims.size(); b < e; ++b) {
    out->add_dims(dims[b]);
  }

  out->set_raw_data(std::string(T.getUnsafePtr(), type.getSizeInBytes()));
}

void ONNXModelWriter::tensorShapeFromPlaceholder(const Placeholder *PH,
                                                 ValueInfoType *valueProto) {
  tensorShapeFromInput(PH->getName(), PH->getType(), valueProto);
}

llvm::Error ONNXModelWriter::writeAllWithNode(const std::string &opName,
                                              const Node *node,
                                              NodeType *proto) {
  proto->set_name(node->getName());
  proto->set_op_type(opName);
  inputsToProto(node, proto);
  outputsToProto(node, proto);
  return llvm::Error::success();
}

llvm::Error ONNXModelWriter::writeAll(const std::string &opName,
                                      const Node *node, GraphType &graph) {
  return writeAllWithNode(opName, node, graph.add_node());
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
llvm::Error ONNXModelWriter::writePad(const PadNode *node, GraphType &graph) {
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
               GlowErr::ErrorCode::MODEL_WRITER_SERIALIZATION_ERROR);
  }

  addValueAttribute(proto, "pads", node->getPads());
  float value = node->getValue();
  if (value != .0f) {
    addValueAttribute(proto, "value", value);
  }

  return writeAllWithNode("Pad", node, proto);
}

llvm::Error ONNXModelWriter::writeConcat(const ConcatNode *node,
                                         GraphType &graph) {
  auto *proto = graph.add_node();
  // Add dictionary entries.
  addValueAttribute(proto, "axis", node->getDim());

  return writeAllWithNode("Concat", node, proto);
}

llvm::Error ONNXModelWriter::writeTranspose(const TransposeNode *node,
                                            GraphType &graph) {
  // Some nodes create transpose for outputs.
  auto *input = node->getInput().getNode();
  if (llvm::dyn_cast<ConvolutionNode>(input) ||
      llvm::dyn_cast<AvgPoolNode>(input) ||
      llvm::dyn_cast<MaxPoolNode>(input) ||
      llvm::dyn_cast<SpaceToDepthNode>(input)) {
    return llvm::Error::success();
  }

  auto *proto = graph.add_node();
  // Add dictionary entries.
  addValueAttribute(proto, "perm", node->getShuffle());

  return writeAllWithNode("Transpose", node, proto);
}

llvm::Error ONNXModelWriter::writeConvolution(const ConvolutionNode *node,
                                              GraphType &graph) {
  assert(node->getLayout() == NHWC && "can only write NHWC Convolutions");
  auto *proto = graph.add_node();
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
  } else {
    proto->add_input(input->getName());
  }

  const Node *filter = node->getFilter().getNode();
  if (const TransposeNode *TN = llvm::dyn_cast<TransposeNode>(filter)) {
    proto->add_input(TN->getInput().getNode()->getName());
  } else {
    proto->add_input(filter->getName());
  }

  proto->add_input(node->getBias().getNode()->getName());

  proto->set_name(node->getName());
  proto->set_op_type("Conv");

  // Use the output of transpose node.
  outputKindToProto(Kinded::Kind::TransposeNodeKind, node, proto);

  return llvm::Error::success();
}

llvm::Error
ONNXModelWriter::writeBatchedReduceMean(const BatchedReduceMeanNode *node,
                                        GraphType &graph) {
  auto *proto = graph.add_node();
  // Add dictionary entries.
  addValueAttribute(proto, "axes", node->getAxes());

  proto->set_name(node->getName());
  proto->set_op_type("ReduceMean");
  inputsToProto(node, proto);

  // Use the output of reshape node.
  if (outputKindToProto(Kinded::Kind::ReshapeNodeKind, node, proto)) {
    // Add dictionary entries.
    addValueAttribute(proto, "keepdims", 1);
  }

  return llvm::Error::success();
}

llvm::Error
ONNXModelWriter::writeBatchedReduceAdd(const BatchedReduceAddNode *node,
                                       GraphType &graph) {
  auto *proto = graph.add_node();
  // Add dictionary entries.
  unsigned_t axis = node->getAxis();
  llvm::ArrayRef<unsigned_t> axes(axis);
  addValueAttribute(proto, "axes", axes);

  proto->set_name(node->getName());
  proto->set_op_type("ReduceSum");
  inputsToProto(node, proto);

  // Use the output of reshape node.
  if (outputKindToProto(Kinded::Kind::ReshapeNodeKind, node, proto)) {
    // Add dictionary entries.
    addValueAttribute(proto, "keepdims", 1);
  }

  return llvm::Error::success();
}

llvm::Error
ONNXModelWriter::writeBatchedReduceMin(const BatchedReduceMinNode *node,
                                       GraphType &graph) {
  auto *proto = graph.add_node();
  // Find dictionary entries.
  addValueAttribute(proto, "axes", node->getAxes());

  return writeAllWithNode("ReduceMin", node, proto);
}

llvm::Error
ONNXModelWriter::writeBatchNormalization(const BatchNormalizationNode *node,
                                         GraphType &graph) {
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

  outputsToProto(node, proto);
  return llvm::Error::success();
}

llvm::Error
ONNXModelWriter::writeMeanVarNormalization(const MeanVarNormalizationNode *node,
                                           GraphType &graph) {
  auto *proto = graph.add_node();
  // Add dictionary entries.
  addValueAttribute(proto, "channel", node->getChannelIdx());
  addValueAttribute(proto, "momentum", node->getMomentum());

  proto->set_name(node->getName());
  proto->set_op_type("MeanVarianceNormalization");

  inputsToProto(node, proto);
  outputsToProto(node, proto);
  return llvm::Error::success();
}

llvm::Error ONNXModelWriter::writeSlice(const SliceNode *node,
                                        GraphType &graph) {
  auto *proto = graph.add_node();
  // Add dictionary entries.
  auto starts = node->getStart();
  auto outs = node->getInput().dims();
  RETURN_ERR_IF_NOT(starts.size() == outs.size(),
                    "Mismatch starts and result dimensions.");

  RETURN_IF_ERR(writeAllWithNode("Slice", node, proto));

  if (opsetVersion_ >= 10) {
    Tensor oneDimTensorStarts(ElemKind::Int64ITy, {starts.size()});
    auto handleStarts = oneDimTensorStarts.getHandle<int64_t>();
    Tensor oneDimTensorEnds(ElemKind::Int64ITy, {starts.size()});
    auto handleEnds = oneDimTensorEnds.getHandle<int64_t>();

    for (size_t b = 0, e = starts.size(); b < e; ++b) {
      handleStarts.raw(b) = starts[b];
      handleEnds.raw(b) = outs[b] + starts[b];
    }

    auto *tensorProto = graph.add_initializer();
    tensorProto->set_name(node->getName().str() + "_starts");
    writeTensor(oneDimTensorStarts, tensorProto);
    proto->add_input(node->getName().str() + "_starts");

    tensorProto = graph.add_initializer();
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
  return llvm::Error::success();
}

llvm::Error ONNXModelWriter::writePow(const PowNode *node, GraphType &graph) {
  auto *proto = graph.add_node();
  proto->set_name(node->getName());
  proto->add_input(node->getLHS().getNode()->getName());
  outputsToProto(node, proto);

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
  return llvm::Error::success();
}

llvm::Error ONNXModelWriter::writeTopK(const TopKNode *node, GraphType &graph) {
  auto *proto = graph.add_node();

  Tensor scalar(ElemKind::Int64ITy, {1});
  auto handle = scalar.getHandle<int64_t>();
  handle.raw(0) = node->getK();

  auto *tensorProto = graph.add_initializer();
  tensorProto->set_name("k");
  writeTensor(scalar, tensorProto);

  RETURN_IF_ERR(writeAllWithNode("TopK", node, proto));

  proto->add_input("k");
  return llvm::Error::success();
}

llvm::Error ONNXModelWriter::writeArgMax(const ArgMaxNode *node,
                                         GraphType &graph) {
  auto *proto = graph.add_node();

  Tensor axis(ElemKind::Int64ITy, {1});
  Tensor keepDims(ElemKind::BoolTy, {1});
  auto axisH = axis.getHandle<int64_t>();
  auto keepDimsH = keepDims.getHandle<int8_t>();
  axisH.raw(0) = node->getAxis();
  keepDimsH.raw(0) = node->getKeepDims();

  auto *tensorProto = graph.add_initializer();
  tensorProto->set_name("axis");
  writeTensor(axis, tensorProto);

  tensorProto = graph.add_initializer();
  tensorProto->set_name("keepDims");
  writeTensor(keepDims, tensorProto);
  RETURN_IF_ERR(writeAllWithNode("ArgMax", node, proto));

  return llvm::Error::success();
}

llvm::Error ONNXModelWriter::writePRelu(const PReluNode *node,
                                        GraphType &graph) {
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
    auto *tensorProto = graph.add_initializer();
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

  outputsToProto(node, proto);
  return llvm::Error::success();
}

llvm::Error ONNXModelWriter::writeGather(const GatherNode *node,
                                         GraphType &graph) {
  auto *proto = graph.add_node();
  // Add dictionary entries.
  auto batchDims = node->getBatchDims();

  if (batchDims != 0) {
    addValueAttribute(proto, "axis", batchDims);
    return writeAllWithNode("BatchGather", node, proto);
  } else {
    return writeAllWithNode("Gather", node, proto);
  }
}

llvm::Error ONNXModelWriter::writeMatMul(const MatMulNode *node,
                                         GraphType &graph) {
  return writeMatMulKind(node, graph, reportedNodes_);
}

llvm::Error ONNXModelWriter::writeBatchMatMul(const BatchMatMulNode *node,
                                              GraphType &graph) {
  return writeMatMulKind(node, graph, reportedNodes_);
}

llvm::Error ONNXModelWriter::writeReshape(const ReshapeNode *node,
                                          GraphType &graph) {
  // ReduceMean/ReduceSum nodes create reshape for the output.
  // Therefore check if this reshape has BatchedReduceMean/BatchedReduceAdd
  // node as input.
  const Node *input = node->getInput().getNode();
  if (llvm::dyn_cast<BatchedReduceMeanNode>(input) ||
      llvm::dyn_cast<BatchedReduceAddNode>(input)) {
    return llvm::Error::success();
  }

  auto *proto = graph.add_node();

  // Add ints type attribute.
  addValueAttribute(proto, "shape", node->getDims());

  return writeAllWithNode("Reshape", node, proto);
}

llvm::Error ONNXModelWriter::writeBucketize(const BucketizeNode *node,
                                            GraphType &graph) {
  auto *proto = graph.add_node();
  // Add dictionary entries.
  addValueAttribute(proto, "boundaries", node->getBoundaries());

  return writeAllWithNode("Bucketize", node, proto);
}

llvm::Error ONNXModelWriter::writeResizeNearest(const ResizeNearestNode *node,
                                                GraphType &graph) {
  auto *proto = graph.add_node();
  // Find dictionary entries.
  addValueAttribute(proto, "height_scale", node->getHeightScale());
  addValueAttribute(proto, "width_scale", node->getWidthScale());

  return writeAllWithNode(node->getName(), node, proto);
}

llvm::Error ONNXModelWriter::writeSoftMax(const SoftMaxNode *node,
                                          GraphType &graph) {
  auto *proto = graph.add_node();
  proto->set_name(node->getName());
  proto->set_op_type("Softmax");
  outputsToProto(node, proto);
  // Find input from Reshape node
  proto->add_input(node->getInput().getNode()->getName());
  return llvm::Error::success();
}

llvm::Error ONNXModelWriter::writeReplaceNaN(const ReplaceNaNNode *node,
                                             GraphType &graph) {
  auto *proto = graph.add_node();
  // Add dictionary entries.
  float value = node->getValue();
  if (value != 0.0f) {
    addValueAttribute(proto, "value", value);
  }
  return writeAllWithNode("ReplaceNaN", node, proto);
}

llvm::Error ONNXModelWriter::writeGatherRanges(const GatherRangesNode *node,
                                               GraphType &graph) {
  auto *proto = graph.add_node();
  // Add dictionary entries.
  addValueAttribute(proto, "maxOutputSize", node->getOutput().dims()[0]);

  return writeAllWithNode("GatherRanges", node, proto);
}

llvm::Error
ONNXModelWriter::writeSparseToDenseMask(const SparseToDenseMaskNode *node,
                                        GraphType &graph) {
  auto *proto = graph.add_node();
  // Add dictionary entries.
  addValueAttribute(proto, "mask", node->getMask());

  return writeAllWithNode("SparseToDenseMask", node, proto);
}

llvm::Error
ONNXModelWriter::writeAdaptiveAvgPool(const AdaptiveAvgPoolNode *node,
                                      GraphType &graph) {
  auto *proto = graph.add_node();

  const auto outShape = ShapeNHWC(node->getResult().dims());
  std::vector<size_t> output_size{outShape.h, outShape.w};
  addValueAttribute(proto, "output_size", llvm::makeArrayRef(output_size));

  return writeAllWithNode("AdaptiveAvgPool", node, proto);
}

llvm::Error ONNXModelWriter::writeLocalResponseNormalization(
    const LocalResponseNormalizationNode *node, GraphType &graph) {
  auto *proto = graph.add_node();
  proto->set_name(node->getName());
  proto->set_op_type("LRN");
  outputsToProto(node, proto);
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

  return llvm::Error::success();
}

llvm::Error ONNXModelWriter::writeBatchBoxCox(const BatchBoxCoxNode *node,
                                              GraphType &graph) {
  auto *proto = graph.add_node();
  addValueAttribute(proto, "epsilon", node->getEpsilon());
  return writeAllWithNode("BatchBoxCox", node, proto);
}

//===-----------------------------------------------------------------===//
//                    Operators Supported by Glow only
//===-----------------------------------------------------------------===//
llvm::Error ONNXModelWriter::writeModulo(const ModuloNode *node,
                                         GraphType &graph) {
  auto *proto = graph.add_node();
  // Add dictionary entries.
  addValueAttribute(proto, "divisor", node->getDivisor());
  addValueAttribute(proto, "sign_follow_divisor",
                    node->getSignFollowDivisor() ? 1 : 0);

  return writeAllWithNode("Modulo", node, proto);
}

namespace {
template <typename T>
void writePool(const T *node, ONNX_NAMESPACE::NodeProto *proto) {
  assert(node->getLayout() == NHWC && "can only write NHWC Pools");
  // Add dictionary entries.
  addValueAttribute(proto, "kernel_shape", node->getKernels());
  addValueAttribute(proto, "strides", node->getStrides());
  addValueAttribute(proto, "pads", node->getPads());

  proto->set_name(node->getName());

  const Node *input = node->getInput().getNode();
  if (const TransposeNode *TN = llvm::dyn_cast<TransposeNode>(input)) {
    proto->add_input(TN->getInput().getNode()->getName());
  } else {
    proto->add_input(input->getName());
  }

  // Use the output of transpose node, if any.
  outputKindToProto(Kinded::Kind::TransposeNodeKind, node, proto);
}
} // namespace

llvm::Error ONNXModelWriter::writeAvgPool(const AvgPoolNode *node,
                                          GraphType &graph) {
  auto *proto = graph.add_node();
  proto->set_op_type("AveragePool");
  writePool(node, proto);
  return llvm::Error::success();
}

llvm::Error ONNXModelWriter::writeMaxPool(const MaxPoolNode *node,
                                          GraphType &graph) {
  auto *proto = graph.add_node();
  proto->set_op_type("MaxPool");
  writePool(node, proto);
  return llvm::Error::success();
}

llvm::Error ONNXModelWriter::writeConvolution3D(const Convolution3DNode *node,
                                                GraphType &graph) {
  auto *proto = graph.add_node();
  // Add dictionary entries.
  addValueAttribute(proto, "kernel_shape", node->getKernels());
  addValueAttribute(proto, "strides", node->getStrides());
  addValueAttribute(proto, "pads", node->getPads());
  addValueAttribute(proto, "group", node->getGroup());

  return writeAllWithNode("Convolution3D", node, proto);
}

llvm::Error ONNXModelWriter::writeSpaceToDepth(const SpaceToDepthNode *node,
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
  outputKindToProto(Kinded::Kind::TransposeNodeKind, node, proto);
  return llvm::Error::success();
}

llvm::Error ONNXModelWriter::writeChannelShuffle(const ChannelShuffleNode *node,
                                                 GraphType &graph) {
  auto *proto = graph.add_node();
  // Add dictionary entries.
  addValueAttribute(proto, "group", node->getGroup());
  addValueAttribute(proto, "kernel", node->getKernel());

  return writeAllWithNode("ChannelShuffle", node, proto);
}

llvm::Error
ONNXModelWriter::writeQuantizationProfile(const QuantizationProfileNode *node,
                                          GraphType &graph) {
  auto *proto = graph.add_node();
  // Add dictionary entries.
  addValueAttribute(proto, "name", node->getProfiledNodeName());
  addValueAttribute(proto, "number", node->getProfiledOutputNumber());

  return writeAllWithNode("QuantizationProfile", node, proto);
}

llvm::Error ONNXModelWriter::writeTraceEvent(const TraceEventNode *node,
                                             GraphType &graph) {
  auto *proto = graph.add_node();
  // Add dictionary entries.
  addValueAttribute(proto, "name", node->getEventName());
  addValueAttribute(proto, "type", node->getEventType());
  addValueAttribute(proto, "index", node->getIndex());

  return writeAllWithNode("TraceEvent", node, proto);
}

llvm::Error ONNXModelWriter::writeInsertTensor(const InsertTensorNode *node,
                                               GraphType &graph) {
  auto *proto = graph.add_node();
  // Add dictionary entries.
  addValueAttribute(proto, "start", node->getStart());
  addValueAttribute(proto, "count", node->getCount());
  addValueAttribute(proto, "axis", node->getAxis());

  return writeAllWithNode("InsertTensor", node, proto);
}

llvm::Error ONNXModelWriter::writeChannelwiseQuantizedConvolution(
    const ChannelwiseQuantizedConvolutionNode *node, GraphType &graph) {
  auto *proto = graph.add_node();
  // Add dictionary entries.
  addValueAttribute(proto, "kernel_shape", node->getKernels());
  addValueAttribute(proto, "strides", node->getStrides());
  addValueAttribute(proto, "pads", node->getPads());
  addValueAttribute(proto, "group", node->getGroup());
  addValueAttribute(proto, "group_wise", node->getGroupwise() ? 1 : 0);

  return writeAllWithNode("ChannelwiseQuantizedConvolution", node, proto);
}

llvm::Error ONNXModelWriter::writeSplat(const SplatNode *node,
                                        GraphType &graph) {
  auto *proto = graph.add_node();
  // Convert value to tensor with result shape
  Tensor tensor(ElemKind::FloatTy, node->getResult().dims());
  auto handle = tensor.getHandle<>();
  float value = node->getValue();
  for (size_t b = 0, e = tensor.size(); b < e; ++b) {
    handle.raw(b) = value;
  }

  // Add dictionary entries.
  addValueAttribute(proto, "value", tensor);

  return writeAllWithNode("Splat", node, proto);
}

llvm::Error ONNXModelWriter::writeAdd(const AddNode *node, GraphType &graph) {
  return writeArithmetic("Add", node, graph, reportedNodes_);
}

llvm::Error ONNXModelWriter::writeDiv(const DivNode *node, GraphType &graph) {
  return writeArithmetic("Div", node, graph, reportedNodes_);
}

llvm::Error ONNXModelWriter::writeMul(const MulNode *node, GraphType &graph) {
  return writeArithmetic("Mul", node, graph, reportedNodes_);
}

llvm::Error ONNXModelWriter::writeSub(const SubNode *node, GraphType &graph) {
  return writeArithmetic("Sub", node, graph, reportedNodes_);
}

// Default exporting algorithm.
#define DEF_ALL_WRITER_NODE(NAME)                                              \
  llvm::Error ONNXModelWriter::write##NAME(const NAME##Node *node,             \
                                           GraphType &graph) {                 \
    return writeAll(#NAME, node, graph);                                       \
  }

// ONNX nodes with default exporting algorithm.
DEF_ALL_WRITER_NODE(Max)
DEF_ALL_WRITER_NODE(Min)
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
DEF_ALL_WRITER_NODE(FusedRowwiseQuantizedSparseLengthsWeightedSum)

llvm::Error ONNXModelWriter::writeConvertTo(const ConvertToNode *node,
                                            GraphType &graph) {
  auto *proto = graph.add_node();
  // Add dictionary entries.
  addValueAttribute(proto, "shape", node->getResult().dims());

  return writeAllWithNode("ConvertTo", node, proto);
}

llvm::Error ONNXModelWriter::writeSelect(const SelectNode *node,
                                         GraphType &graph) {
  auto *proto = graph.add_node();
  // Add dictionary entries.
  addValueAttribute(proto, "shape", node->getResult().dims());

  return writeAllWithNode("Select", node, proto);
}

llvm::Error ONNXModelWriter::writeQuantize(const QuantizeNode *node,
                                           GraphType &graph) {
  auto *proto = graph.add_node();
  auto outTy = node->getResult().getType();
  // Add dictionary entries.
  addValueAttribute(proto, "scale", outTy->getScale());
  addValueAttribute(proto, "offset", outTy->getOffset());

  return writeAllWithNode("Quantize", node, proto);
}

llvm::Error ONNXModelWriter::writeIntLookupTable(const IntLookupTableNode *node,
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

  return writeAllWithNode("IntLookupTable", node, proto);
}

llvm::Error
ONNXModelWriter::writeLengthsRangeFill(const LengthsRangeFillNode *node,
                                       GraphType &graph) {
  auto *proto = graph.add_node();
  // Add dictionary entries.
  addValueAttribute(proto, "size", node->getResult().dims()[0]);

  return writeAllWithNode("LengthsRangeFill", node, proto);
}

llvm::Error
ONNXModelWriter::writeRescaleQuantized(const RescaleQuantizedNode *node,
                                       GraphType &graph) {
  auto *proto = graph.add_node();
  auto outTy = node->getResult().getType();
  // Add dictionary entries.
  addValueAttribute(proto, "scale", outTy->getScale());
  addValueAttribute(proto, "offset", outTy->getOffset());

  return writeAllWithNode("RescaleQuantized", node, proto);
}

llvm::Error ONNXModelWriter::writeFullyConnected(const FullyConnectedNode *node,
                                                 GraphType &graph) {
  auto *proto = graph.add_node();
  proto->set_name(node->getName());
  proto->set_op_type("FCTransposed");

  const auto *input = node->getInput().getNode();

  if (const auto *reshape = llvm::dyn_cast<ReshapeNode>(input)) {
    proto->add_input(reshape->getInput().getNode()->getName());
    reportedNodes_.insert(reshape);
  } else {
    proto->add_input(input->getName());
  }

  proto->add_input(node->getWeights().getNode()->getName());
  proto->add_input(node->getBias().getNode()->getName());
  outputsToProto(node, proto);
  return llvm::Error::success();
}

llvm::Error ONNXModelWriter::writeSparseToDense(const SparseToDenseNode *node,
                                                GraphType &graph) {
  auto *proto = graph.add_node();

  RETURN_IF_ERR(writeAllWithNode("SparseToDense", node, proto));

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
  return llvm::Error::success();
}

llvm::Error ONNXModelWriter::writeTile(const TileNode *node, GraphType &graph) {
  auto *proto = graph.add_node();

  // unwind Tile
  std::vector<size_t> repeats;
  const TileNode *tile = unwindTile(node, &repeats, reportedNodes_);

  proto->set_name("Tile");
  proto->set_op_type("Tile");
  // Use inputs from top tile.
  inputsToProto(tile, proto);
  // Use outputs from bottom tile.
  outputsToProto(node, proto);

  // Add node indices
  auto *indices = graph.add_node();
  indices->set_name("Constant");
  indices->set_op_type("Constant");
  indices->add_output("indices");

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
  proto->add_input("indices");

  return llvm::Error::success();
}

// Unsupported for export Glow nodes.
#define DEF_UNSUPPORTED_STORAGE(NAME)                                          \
  llvm::Error ONNXModelWriter::write##NAME(const NAME *node, GraphType &) {    \
    return writeUnexpectedKind(node);                                          \
  }

// Helper nodes.
DEF_UNSUPPORTED_STORAGE(Placeholder)
DEF_UNSUPPORTED_STORAGE(Constant)
DEF_UNSUPPORTED_STORAGE(Storage)

// Unsupported for export Glow nodes.
#define DEF_UNSUPPORTED_NODE(NAME)                                             \
  llvm::Error ONNXModelWriter::write##NAME(const NAME##Node *node,             \
                                           GraphType &) {                      \
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
DEF_UNSUPPORTED_NODE(SparseLengthsWeightedSumGrad)
DEF_UNSUPPORTED_NODE(SigmoidCrossEntropyWithLogits)
DEF_UNSUPPORTED_NODE(LocalResponseNormalizationGrad)
DEF_UNSUPPORTED_NODE(AdaptiveAvgPoolGrad)

#ifdef GLOW_WITH_CPU

llvm::Error ONNXModelWriter::writeCPUMaxSplat(const CPUMaxSplatNode *node,
                                              GraphType &graph) {
  auto *proto = graph.add_node();
  // Add dictionary entries.
  addValueAttribute(proto, "value", node->getSplatValue());

  return writeAllWithNode("CPUMaxSplat", node, proto);
}

llvm::Error ONNXModelWriter::writeCPUConvDKKC8(const CPUConvDKKC8Node *node,
                                               GraphType &graph) {
  auto *proto = graph.add_node();
  // Add dictionary entries.
  addValueAttribute(proto, "kernel_shape", node->getKernels());
  addValueAttribute(proto, "strides", node->getStrides());
  addValueAttribute(proto, "pads", node->getPads());
  addValueAttribute(proto, "group", node->getGroup());

  return writeAllWithNode("CPUConvDKKC8", node, proto);
}

#endif // GLOW_WITH_CPU

#ifdef GLOW_WITH_OPENCL

llvm::Error
ONNXModelWriter::writeOCLBatchedReduceAdd(const OCLBatchedReduceAddNode *node,
                                          GraphType &graph) {
  auto *proto = graph.add_node();
  // Add dictionary entries.
  addValueAttribute(proto, "axis", node->getAxis());
  addValueAttribute(proto, "source_axis", node->getAxisSrcSliceSize());

  return writeAllWithNode("OCLBatchedReduceAdd", node, proto);
}

#endif // GLOW_WITH_OPENCL
} // namespace glow
