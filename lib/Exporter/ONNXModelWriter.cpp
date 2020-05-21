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

#include <float.h>

#include "miniz.h"

using namespace glow::runtime;
using google::protobuf::RepeatedPtrField;

namespace glow {
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

/// Adds the type attributes from \p NV to \p proto. \p ioNum, \p isInput, and
/// \p addPrefix are used to format the name of the attribute.
void addTypeAttributes(ONNX_NAMESPACE::NodeProto *proto, const NodeValue NV,
                       unsigned ioNum, bool isInput,
                       const std::string &addPrefix = "") {
  const TypeRef ty = NV.getType();

  // Add ElemKind.
  auto *elemKindAttr = proto->add_attribute();
  elemKindAttr->set_name(
      getTypeAttrID(ioNum, elemKindSignifier, isInput, addPrefix));
  AttributeAssigner<false, false, llvm::StringRef>::assign(
      elemKindAttr, ty->getElementName());

  // Add Shape.
  addValueAttribute(proto,
                    getTypeAttrID(ioNum, shapeSignifier, isInput, addPrefix),
                    NV.dims());

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

    // Additionally visit the predicate input if it exists.
    if (N->hasPredicate()) {
      visitRecursively(N->getPredicate().getNode());
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

Error ONNXModelWriter::writeModel(const ::google::protobuf::Message &modelProto,
                                  bool textMode) {
  {
    ::google::protobuf::io::OstreamOutputStream zeroCopyOutput(&ff_);
    // Write the content.
    if (textMode) {
      RETURN_ERR_IF_NOT(
          google::protobuf::TextFormat::Print(modelProto, &zeroCopyOutput),
          "Can't write to the output file name",
          ErrorValue::ErrorCode::MODEL_WRITER_SERIALIZATION_ERROR);
    } else {
      ::google::protobuf::io::CodedOutputStream codedOutput(&zeroCopyOutput);
      modelProto.SerializeToCodedStream(&codedOutput);
      RETURN_ERR_IF_NOT(
          !codedOutput.HadError(), "Can't write to the output file name",
          ErrorValue::ErrorCode::MODEL_WRITER_SERIALIZATION_ERROR);
    }
  }
  ff_.flush();
  ff_.close();
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

Error ONNXModelWriter::writeFunction() {
  // Use pre order graph traversal.
  // If node is constant with uses, turned into "input" and create a tensor
  // If node is placeholder with uses, turned into "input" with tensor shape,
  // except the case when placeholder has use as SaveNode.
  // Otherwise call common operators method or special operators and write
  // protobuf inputs from node inputs and protobuf outputs from uses.

  // Keep track of already visited or processed nodes.
  std::unordered_set<const Node *> reportedNodes;

  ReverseGraphWalker visitor(*F_);
  for (const auto *N : visitor.getNodes()) {
    if (reportedNodes.count(N)) {
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
      auto *inputProto = graphProto_->add_input();
      tensorShapeFromPlaceholder(PH, inputProto);
    } else if (kind == Kinded::Kind::ConstantKind) {
      // Write global initializer, output tensor bytes.
      const auto *C = llvm::cast<Constant>(N);
      auto *tensorProto = addInitializer(*graphProto_);
      // Always use generateNodeOutputName for all inputs and outputs.
      tensorProto->set_name(C->getOutput().generateNodeOutputName(
          /* stripResNoFor0thInput */ true));
      writeTensor(C->getPayload(), tensorProto, includeConstantData_);
      // Also include the layout in the initializer to be loaded later.
      addAttrToDocString(tensorProto, layoutSignifier, C->getLayout());
    } else if (kind == Kinded::Kind::SaveNodeKind) {
      // Save node case, find input and use its name as a global output,
      // output only shape.
      const SaveNode *SN = llvm::cast<SaveNode>(N);
      const auto *PH = SN->getPlaceholder();

      // We need to add an Identity to map the name from
      // generateNodeOutputName() to the name of the Placeholder.
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
      if (dagMode_) {
        addValueAttribute(proto, "partitionName", F_->getName().str());
        // Skip writing Placeholders that are only intermediates -- these are
        // understood/recreated during reimporting based on an op's partition.
        if (isIntermediatePHForDAG(PH)) {
          addValueAttribute(proto, "isIntermediateOutputForDAG", true);
          continue;
        }
      }

      auto *out = graphProto_->add_output();
      tensorShapeFromPlaceholder(PH, out);

      // Use the doc string to specify the name that should be used for the
      // SaveNode to ensure it's kept the same between export and import.
      addAttrToDocString(out, saveNameSignifier, SN->getName());
    } else {
      RETURN_IF_ERR(writeGlowCustomOperator(N, *graphProto_));
    }
    reportedNodes.insert(N);
  }

  return Error::success();
}

Error ONNXModelWriter::setupNewProto(const std::string &modelFilename,
                                     const size_t irVersion,
                                     const size_t opsetVersion) {
  // Try to open file for write
  ff_.open(modelFilename, std::ios::out | std::ios::trunc | std::ios::binary);
  RETURN_ERR_IF_NOT(ff_, "Can't find the output file name: " + modelFilename,
                    ErrorValue::ErrorCode::MODEL_WRITER_INVALID_FILENAME);

  modelProto_.set_ir_version(irVersion);
  modelProto_.set_producer_name("GlowONNXModelWriter");
  auto *opsetImportProto = modelProto_.add_opset_import();
  opsetImportProto->set_version(opsetVersion);
  graphProto_ = modelProto_.mutable_graph();
  graphProto_->set_name("glow");
  return Error::success();
}

Error ONNXModelWriter::finalizeAndWriteProto(llvm::StringRef name) {
  // Nodes have been added in a reverse order from SaveNode up to the inputs,
  // we need to rearrange all nodes in the reverse order before serialization.
  auto *nodes = graphProto_->mutable_node();
  for (size_t i = 0, n = nodes->size(); i < n / 2; ++i) {
    nodes->SwapElements(i, n - i - 1);
  }

  if (zipMode_) {
    const bool compressed = false;
    ZipWriter zip(&ff_, name);
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
  }

  return writeModel(modelProto_, textMode_);
}

ONNXModelWriter::ONNXModelWriter(const std::string &modelFilename, Function &F,
                                 size_t irVersion, size_t opsetVersion,
                                 Error *errPtr, bool textMode, bool zipMode,
                                 bool includeConstantData)
    : F_(&F), zipMode_(zipMode), textMode_(textMode),
      includeConstantData_(includeConstantData), dagMode_(false) {
  // Verify that the version of the library that we linked against is
  // compatible with the version of the headers we compiled against.
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  // If errPtr already contains an error then don't continue with constructor.
  if (errPtr && *errPtr) {
    return;
  }

  // Lambda to setup the ONNXModelWriter and return any Errors that were raised.
  auto setup = [&]() -> Error {
    RETURN_IF_ERR(setupNewProto(modelFilename, irVersion, opsetVersion));

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

ONNXModelWriter::ONNXModelWriter(const std::string &modelFilename,
                                 DAGListTy &dagList, size_t irVersion,
                                 size_t opsetVersion, Error *errPtr,
                                 bool textMode, bool zipMode,
                                 bool includeConstantData)
    : F_(nullptr), zipMode_(zipMode), textMode_(textMode),
      includeConstantData_(includeConstantData), dagMode_(true) {
  // If errPtr already contains an error then don't continue with constructor.
  if (errPtr && *errPtr) {
    return;
  }

  // Lambda to setup the ONNXModelWriter and return any Errors that were raised.
  auto setup = [&]() -> Error {
    RETURN_IF_ERR(setupNewProto(modelFilename, irVersion, opsetVersion));

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

/// Add quantization parameters to the doc_string in \p out based on \p type.
template <typename T>
static void addQuantParamsToDocString(T *out, const Type &type) {
  addAttrToDocString(out, qScaleSignifier,
                     strFormat("%.*f", DBL_DIG - 1, type.getScale()));
  addAttrToDocString(out, qOffsetSignifier, std::to_string(type.getOffset()));
}

void ONNXModelWriter::writeTensor(const Tensor &T, TensorType *out,
                                  bool includeData) {
  const auto &type = T.getType();
  out->set_data_type(convertType(type));
  const auto &dims = type.dims();
  for (unsigned b = 0, e = dims.size(); b < e; ++b) {
    out->add_dims(dims[b]);
  }

  if (includeData) {
    out->set_raw_data(T.getUnsafePtr(), type.getSizeInBytes());
  }

  addAttrToDocString(out, elemKindSignifier, type.getElementName());

  if (type.isQuantizedType()) {
    // Note the use of DBL_DIG is to ensure all digits of the scale are printed.
    addQuantParamsToDocString(out, type);
  }
}

void ONNXModelWriter::tensorShapeFromPlaceholder(const Placeholder *PH,
                                                 ValueInfoType *valueProto) {
  tensorShapeFromInput(PH->getName(), PH->getType(), valueProto);

  // Write out any meta information we need to for the Placeholder.
  addAttrToDocString(valueProto, staticSignifier,
                     std::to_string(PH->isStatic()));
  addAttrToDocString(valueProto, trainableSignifier,
                     std::to_string(PH->isTraining()));
  addAttrToDocString(valueProto, layoutSignifier, PH->getLayout());
  addAttrToDocString(valueProto, elemKindSignifier,
                     PH->getType()->getElementName());

  // Also include quantization params if necessary.
  if (PH->getType()->isQuantizedType()) {
    addQuantParamsToDocString(valueProto, *PH->getType());
  }
}

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
  if (dagMode_) {
    addValueAttribute(opProto, "partitionName", F_->getName().str());
  }

  return Error::success();
}

} // namespace glow
