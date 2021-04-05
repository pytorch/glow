// Copyright 2004-present Facebook. All Rights Reserved.

#include <ATen/WrapDimUtils.h>
#include <iostream>
#include <string>
#include <torch/script.h>
#include <unordered_set>
#include <vector>

#include "ShapeInferenceEngine.h"

#include "glow/Support/Error.h"
#include "glow/Support/Support.h"

DEFINE_string(shapeInferenceOpBlocklist, "", "Ops to skip shape inference");
DEFINE_int32(max_feature_length, -1, "max feature length");
DEFINE_bool(print_shape_inference_graph, false,
            "print graph for shape inference debugging");
DEFINE_bool(skipReferOperatorsOnCpu, false,
            "Skip referring shapes running on CPU");

namespace glow {

static std::vector<std::string> splitStr(const std::string &s,
                                         const char delimiter = ',') {
  std::vector<std::string> substrings;
  size_t start = 0;
  bool lastWasSplit = true;
  for (size_t i = 0; i < s.size(); i++) {
    if (lastWasSplit && s[i] == ' ') {
      start = i + 1;
      continue;
    }
    lastWasSplit = false;
    if (s[i] == delimiter) {
      substrings.push_back(s.substr(start, i - start));
      start = i + 1;
      lastWasSplit = true;
    }
  }

  if (start < s.size() - 1) {
    substrings.push_back(s.substr(start, s.size() - start));
  }

  return substrings;
}

ShapeInferenceEngine::ShapeInferenceEngine(
    const torch::jit::Graph &graph, const at::ArrayRef<at::IValue> &inputs,
    const std::string &fusionNodeSymbol, const bool &compilationMode)
    : graph_(graph), inputs_(inputs), fusionNodeSymbol_(fusionNodeSymbol),
      compilationMode_(compilationMode) {
  if (!FLAGS_shapeInferenceOpBlocklist.empty()) {
    auto ret = splitStr(FLAGS_shapeInferenceOpBlocklist);
    for (const auto &s : ret) {
      blockList_.insert(s);
    }
  }
};

bool ShapeInferenceEngine::getNodeInputShape(const torch::jit::Node *node,
                                             MetaStack &inputMetas) {
  for (auto input : node->inputs()) {
    auto it = shapeMap_.find(input);
    if (it == shapeMap_.end()) {
      LOG(ERROR) << "Node: " << node->kind().toDisplayString() << " input "
                 << input->debugName();
      return false;
    }
    inputMetas.emplace_back(shapeMap_[input]);
  }
  return true;
}

const MetaStack &ShapeInferenceEngine::getGraphOutputShape() {
  return outputShape_;
}

const std::unordered_map<const torch::jit::Value *, VariableMeta> &
ShapeInferenceEngine::getVariableMap() {
  return shapeMap_;
}

Error ShapeInferenceEngine::shapeOnNode(const torch::jit::Node *node) {

  /// Get op symbol
  const auto kind = node->kind();
  const std::string symbol = kind.toQualString();
  if (blockList_.count(symbol)) {
    // Skip shape inference for this node. If other nodes have dependency
    // on this one then later their shape inference would fail explicitly.
    LOG(INFO) << "Skip shape inference for " << symbol;
    return Error::success();
  }

  // TODO(T88296130): cleanup this part to reuse the code with the if/switch
  const std::unordered_set<std::string> white_list = {
      "glow::fused_stack",
      "glow::fused_broadcast_stack",
      "glow::fused_broadcast_cat",
      "glow::fused_split",
      "quantized::embedding_bag_byte_rowwise_offsets",
      "quantized::embedding_bag_4bit_rowwise_offsets",
      "glow::unpacked_quantized_linear",
      "fb::lengths_to_offsets",
      "fb::simple_embedding_bag_sum",
      "fb::glow_embedding_bag",
      "fb::glow_embedding_bag_byte_rowwise_offsets",
      "fb::glow_embedding_bag_4bit_rowwise_offsets",
      "fb::xl_embedding_bag",
      "fb::xl_embedding_bag_byte_rowwise_offsets",
      "fb::xl_embedding_bag_4bit_rowwise_offsets",
      "fb::fast_gather",
      "fb::lengths_range",
      "fb::lengths_range_w_truncation_size",
      "aten::quantize_per_tensor",
      "aten::dequantize",
      "quantized::mul",
      "prim::Constant",
      "aten::tanh",
      "aten::relu",
      "aten::sigmoid",
      "aten::sub",
      "aten::pow",
      "aten::mul",
      "aten::add",
      "aten::div",
      "aten::rsub",
      "aten::mm",
      "aten::addmm",
      "aten::bmm",
      "aten::t",
      "aten::transpose",
      "aten::flatten",
      "prim::FusedConcat",
      "prim::ConstantChunk",
      "aten::chunk",
      "prim::ListConstruct",
      "aten::slice",
      "aten::reshape",
      "aten::cat",
      "aten::permute",
      "aten::embedding_bag",
      "aten::matmul",
      "aten::layer_norm",
      "aten::linear",
      "aten::stack",
      "aten::to",
      "aten::sum",
      "prim::dtype",
      "prim::ListUnpack"};

  if (!white_list.count(symbol)) {
    // Skip shape inference for this node. If other nodes have dependency
    // on this one then later their shape inference would fail explicitly.
    LOG(INFO) << "Skip shape inference for " << symbol;
    return Error::success();
  }

  /// Extract shapes of inputs from shape mapping
  MetaStack inputMetas;

  /// The outputs of each Op shape function include the shape and data
  /// type, and the shape could be either the shape or int value
  /// generated by prim::consant or prim::ListContruct.
  TensorOutput tensorOutput;
  TensorListOutput tensorListOutput;

  bool ret = getNodeInputShape(node, inputMetas);
  if (!ret) {
    LOG(WARNING) << "Skip shape inference for " << symbol
                 << " due to prior missing shapes";
    return Error::success();
  }

  // Get output shape or int value of the ops without actual computation
  if (symbol == "glow::fused_stack") {
    int64_t dim = node->i(at::attr::dim);
    ASSIGN_VALUE_OR_RETURN_ERR(tensorOutput, fusedStack(inputMetas, dim));
  } else if (symbol == "glow::fused_broadcast_stack") {
    int64_t dim = node->i(at::attr::dim);
    ASSIGN_VALUE_OR_RETURN_ERR(tensorOutput,
                               fusedBroadcastStack(inputMetas, dim));
  } else if (symbol == "glow::fused_broadcast_cat") {
    int64_t dim = node->i(at::attr::dim);
    ASSIGN_VALUE_OR_RETURN_ERR(tensorOutput,
                               fusedBroadcastConcat(inputMetas, dim));
  } else if (symbol == "glow::fused_split") {
    ASSIGN_VALUE_OR_RETURN_ERR(tensorListOutput, fusedSplit(inputMetas));
  } else if (symbol == "quantized::embedding_bag_byte_rowwise_offsets") {
    ASSIGN_VALUE_OR_RETURN_ERR(
        tensorOutput, quantizedEmbeddingBagByteRowwiseOffsets(inputMetas));
  } else if (symbol == "quantized::embedding_bag_4bit_rowwise_offsets") {
    ASSIGN_VALUE_OR_RETURN_ERR(tensorOutput,
                               embeddingBag4BitRowwiseOffsets(inputMetas));
  } else if (symbol == "glow::unpacked_quantized_linear") {
    ASSIGN_VALUE_OR_RETURN_ERR(tensorOutput,
                               glowUnpackedQuantizedLinear(inputMetas));
  } else if (symbol == "fb::lengths_to_offsets") {
    ASSIGN_VALUE_OR_RETURN_ERR(tensorOutput, lengthsToOffsets(inputMetas));
  } else if (symbol == "fb::simple_embedding_bag_sum") {
    ASSIGN_VALUE_OR_RETURN_ERR(tensorOutput, embeddingBag(inputMetas));
  } else if (symbol == "fb::glow_embedding_bag") {
    ASSIGN_VALUE_OR_RETURN_ERR(tensorOutput, glowEmbeddingBag(inputMetas));
  } else if (symbol == "fb::glow_embedding_bag_byte_rowwise_offsets") {
    ASSIGN_VALUE_OR_RETURN_ERR(
        tensorOutput, quantizedGlowEmbeddingBagByteRowwiseOffsets(inputMetas));
  } else if (symbol == "fb::glow_embedding_bag_4bit_rowwise_offsets") {
    ASSIGN_VALUE_OR_RETURN_ERR(
        tensorOutput, quantizedGlowEmbeddingBag4BitRowwiseOffsets(inputMetas));
  } else if (symbol == "fb::xl_embedding_bag") {
    ASSIGN_VALUE_OR_RETURN_ERR(tensorOutput, xlEmbeddingBag(inputMetas));
  } else if (symbol == "fb::xl_embedding_bag_byte_rowwise_offsets") {
    ASSIGN_VALUE_OR_RETURN_ERR(
        tensorOutput, quantizedXLEmbeddingBagByteRowwiseOffsets(inputMetas));
  } else if (symbol == "fb::xl_embedding_bag_4bit_rowwise_offsets") {
    ASSIGN_VALUE_OR_RETURN_ERR(
        tensorOutput, quantizedXLEmbeddingBag4BitRowwiseOffsets(inputMetas));
  } else if (symbol == "fb::fast_gather") {
    ASSIGN_VALUE_OR_RETURN_ERR(tensorOutput, fastGather(inputMetas));
  } else if (symbol == "fb::lengths_range") {
    ASSIGN_VALUE_OR_RETURN_ERR(tensorOutput, lengthsRange(inputMetas));
  } else if (symbol == "fb::lengths_range_w_truncation_size") {
    // Current shape inference function can handle both cases.
    ASSIGN_VALUE_OR_RETURN_ERR(tensorOutput, lengthsRange(inputMetas));
  } else if (symbol == "aten::quantize_per_tensor") {
    ASSIGN_VALUE_OR_RETURN_ERR(tensorOutput, quantizePerTensor(inputMetas));
  } else if (symbol == "aten::dequantize") {
    ASSIGN_VALUE_OR_RETURN_ERR(tensorOutput, dequantize(inputMetas));
  } else if (symbol == "quantized::mul") {
    ASSIGN_VALUE_OR_RETURN_ERR(tensorOutput, quantizedMul(inputMetas));
  } else {
    switch (kind) {
    case c10::prim::Constant: {
      ASSIGN_VALUE_OR_RETURN_ERR(tensorOutput, primConstant(node));
      break;
    }
    case c10::aten::tanh:
    case c10::aten::relu:
    case c10::aten::sigmoid: {
      RETURN_ERR_IF_NOT(inputMetas.size() == 1,
                        "Expected 1 input shape for operators.");
      tensorOutput.shapeOrIntValues = inputMetas[0].shape<TensorShape>(),
      tensorOutput.dtype = inputMetas[0].dtype;
      break;
    }
    case c10::aten::sub:
    case c10::aten::pow:
    case c10::aten::mul:
    case c10::aten::add:
    case c10::aten::div:
    case c10::aten::rsub: {
      ASSIGN_VALUE_OR_RETURN_ERR(tensorOutput, binaryOp(inputMetas));
      break;
    }
    case c10::aten::mm: {
      ASSIGN_VALUE_OR_RETURN_ERR(tensorOutput, mm(inputMetas));
      break;
    }
    case c10::aten::addmm: {
      ASSIGN_VALUE_OR_RETURN_ERR(tensorOutput, addmm(inputMetas));
      break;
    }
    case c10::aten::bmm: {
      ASSIGN_VALUE_OR_RETURN_ERR(tensorOutput, bmm(inputMetas));
      break;
    }
    case c10::aten::t: {
      ASSIGN_VALUE_OR_RETURN_ERR(tensorOutput, t(inputMetas));
      break;
    }
    case c10::aten::transpose: {
      ASSIGN_VALUE_OR_RETURN_ERR(tensorOutput, transpose(inputMetas));
      break;
    }
    case c10::aten::flatten: {
      ASSIGN_VALUE_OR_RETURN_ERR(tensorOutput, flatten(inputMetas));
      break;
    }
    case c10::prim::FusedConcat: {
      int64_t dim = node->i(at::attr::dim);
      ASSIGN_VALUE_OR_RETURN_ERR(tensorOutput, fusedConcat(inputMetas, dim));
      break;
    }
    case c10::prim::ConstantChunk: {
      int64_t chunks = node->i(at::attr::chunks);
      int64_t dim = node->i(at::attr::dim);
      ASSIGN_VALUE_OR_RETURN_ERR(tensorListOutput,
                                 constantChunk(inputMetas, chunks, dim));
      break;
    }
    case c10::aten::chunk: {
      ASSIGN_VALUE_OR_RETURN_ERR(tensorListOutput, chunk(inputMetas));
      break;
    }
    case c10::prim::ListConstruct: {
      ASSIGN_VALUE_OR_RETURN_ERR(tensorListOutput, listConstruct(inputMetas));
      break;
    }
    case c10::aten::slice: {
      ASSIGN_VALUE_OR_RETURN_ERR(tensorOutput, slice(inputMetas));
      break;
    }
    case c10::aten::reshape: {
      ASSIGN_VALUE_OR_RETURN_ERR(tensorOutput, reshape(inputMetas));
      break;
    }
    case c10::aten::cat: {
      ASSIGN_VALUE_OR_RETURN_ERR(tensorOutput, cat(inputMetas));
      break;
    }
    case c10::aten::permute: {
      ASSIGN_VALUE_OR_RETURN_ERR(tensorOutput, permute(inputMetas));
      break;
    }
    case c10::aten::embedding_bag: {
      ASSIGN_VALUE_OR_RETURN_ERR(tensorOutput, embeddingBag(inputMetas));
      break;
    }
    case c10::aten::matmul: {
      ASSIGN_VALUE_OR_RETURN_ERR(tensorOutput, matmul(inputMetas));
      break;
    }
    case c10::aten::layer_norm: {
      ASSIGN_VALUE_OR_RETURN_ERR(tensorOutput, layerNorm(inputMetas));
      break;
    }
    case c10::aten::linear: {
      ASSIGN_VALUE_OR_RETURN_ERR(tensorOutput, linear(inputMetas));
      break;
    }
    case c10::aten::stack: {
      ASSIGN_VALUE_OR_RETURN_ERR(tensorOutput, stack(inputMetas));
      break;
    }
    case c10::aten::to: {
      ASSIGN_VALUE_OR_RETURN_ERR(tensorOutput, to(inputMetas));
      break;
    }
    case c10::aten::sum: {
      ASSIGN_VALUE_OR_RETURN_ERR(tensorOutput, sum(inputMetas));
      break;
    }
    case c10::prim::dtype: {
      ASSIGN_VALUE_OR_RETURN_ERR(tensorOutput, primDtype(inputMetas));
      break;
    }
    case c10::prim::ListUnpack: {
      ASSIGN_VALUE_OR_RETURN_ERR(tensorListOutput, listUnpack(inputMetas));
      break;
    }
    default: {
      return MAKE_ERR(strFormat("Node's operator %s is not supported",
                                kind.toQualString()));
    }
    }
  }

  /// Put output into map
  /// For \p prim::Constant, the output could be either Tensor or NumberType.
  /// If the output is TensorType, store the \p outputShapesOrValues
  /// into VariableMeta.listOfShape;
  /// Else store the \p outputShapesOrValues into VariableMeta.intValue.
  /// For \p prim::ListConstruct, if the output is a Scalar[], Bool[],
  /// Store the shape of \p outputShapesOrValues into VariableMeta.listOfShape
  /// store the value of \p outputShapesOrValues into VariableMeta.intValue
  /// Else the output is Tensor[], Store the list of shape
  /// \p outputShapesOrValues into VariableMeta.listOfShape
  /// For \p aten::embedding_bag, since the output is a std::tuple<Tensor,
  /// Tensor, Tensor, Tensor>(ret, offset2bag, bag_size, bag_size), and for now,
  /// only the ret tensor shape needed, the embeddingBag() only generate the ret
  /// shape.
  /// For \p c10::aten::chunk, the output is tensor[],
  /// Store the shapes \p outputShapesOrValues into VariableMeta.listOfShape
  if (kind == c10::prim::Constant || kind == c10::prim::dtype) {
    if (node->output()->type()->isSubtypeOf(at::TensorType::get())) {
      shapeMap_[node->output()].listOfShape.emplace_back(
          std::move(tensorOutput.shapeOrIntValues));
      shapeMap_[node->output()].dtype = tensorOutput.dtype;
    } else {
      shapeMap_[node->output()].listOfShape.emplace_back((TensorShape){1});
      shapeMap_[node->output()].intValue =
          std::move(tensorOutput.shapeOrIntValues);
      shapeMap_[node->output()].dtype = tensorOutput.dtype;
    }
  } else if (kind == c10::prim::ListConstruct) {
    auto elem_type =
        node->output()->type()->cast<c10::ListType>()->getElementType();
    if (elem_type->kind() == at::TensorType::Kind ||
        (elem_type->kind() == at::OptionalType::Kind &&
         elem_type->cast<c10::OptionalType>()->getElementType()->kind() ==
             at::TensorType::Kind)) {
      shapeMap_[node->output()].listOfShape.emplace_back(
          std::move(tensorListOutput.shape));
      shapeMap_[node->output()].dtype = tensorListOutput.dtype;
    } else {
      shapeMap_[node->output()].listOfShape.emplace_back((TensorShape){
          static_cast<long>(tensorListOutput.shape[0].size()), 1});
      shapeMap_[node->output()].intValue = std::move(tensorListOutput.shape[0]);
      shapeMap_[node->output()].dtype = tensorListOutput.dtype;
    }
  } else if (symbol == "fb::glow_embedding_bag") {
    shapeMap_[node->output()].listOfShape.emplace_back(
        std::move(tensorOutput.shapeOrIntValues));
    shapeMap_[node->output()].dtype = tensorOutput.dtype;
  } else if (symbol == "fb::xl_embedding_bag") {
    shapeMap_[node->output(0)].listOfShape.emplace_back(
        std::move(tensorOutput.shapeOrIntValues));
    shapeMap_[node->output(0)].dtype = tensorOutput.dtype;
  } else if (kind == c10::aten::embedding_bag ||
             symbol == "fb::simple_embedding_bag_sum") {
    shapeMap_[node->output(0)].listOfShape.emplace_back(
        std::move(tensorOutput.shapeOrIntValues));
    shapeMap_[node->output(0)].dtype = tensorOutput.dtype;
  } else if (kind == c10::aten::chunk) {
    shapeMap_[node->output()].listOfShape.emplace_back(
        std::move(tensorListOutput.shape));
    shapeMap_[node->output()].dtype = tensorListOutput.dtype;
  } else if (tensorOutput.shapeOrIntValues.size() > 0) {
    shapeMap_[node->output()].listOfShape.emplace_back(
        std::move(tensorOutput.shapeOrIntValues));
    shapeMap_[node->output()].dtype = tensorOutput.dtype;
  } else {
    for (int i = 0; i < node->outputs().size(); i++) {
      shapeMap_[node->output(i)].listOfShape.emplace_back(
          std::move(tensorListOutput.shape[i]));
      shapeMap_[node->output(i)].dtype = tensorListOutput.dtype;
    }
  }
  return Error::success();
}

Error ShapeInferenceEngine::runSubGraph(
    const torch::jit::Graph &graph,
    const at::ArrayRef<torch::jit::IValue> &inputs) {
  RETURN_IF_ERR(getGraphInputShapeType(graph, inputs));
  for (auto *node : graph.nodes()) {
    CHECK(!node->hasAttribute(torch::jit::attr::Subgraph));
    RETURN_IF_ERR(shapeOnNode(node));
  }
  return Error::success();
}

Error ShapeInferenceEngine::runGraph(
    const torch::jit::Graph &graph,
    const at::ArrayRef<torch::jit::IValue> &inputs) {
  // Populate input shapes
  RETURN_IF_ERR(getGraphInputShapeType(graph, inputs));

  int totalFusionNodes = 0;
  for (auto *node : graph.nodes()) {
    if (node->kind().toQualString() == fusionNodeSymbol_) {
      totalFusionNodes += 1;
    }
  }
  int fusionNodeIndex = 0;
  /// Run shape inference for each node
  for (auto *node : graph.nodes()) {
    if (node->hasAttribute(torch::jit::attr::Subgraph)) {
      std::string kind = node->kind().toQualString();
      CHECK_EQ(kind.find(fusionNodeSymbol_), 0);
      // After fusion the input Value of the subgraph and
      // input Value of the fusion node are different
      // in memory objects. Therefore we populate inputMeta
      // beforehand and pass it to recursive run.
      std::vector<torch::jit::IValue> subgraphInputs;
      for (auto i : node->inputs()) {
        auto it = shapeMap_.find(i);
        CHECK(it != shapeMap_.end()) << "missing input " << i->debugName();
        // Only support tensor input for now
        // TODO Add support for other input types, e.g., tensor list
        subgraphInputs.emplace_back(
            torch::empty(it->second.shape<TensorShape>(),
                         torch::TensorOptions().dtype(it->second.dtype)));
      }
      const at::ArrayRef<torch::jit::IValue> inputRefs(subgraphInputs);

      auto subgraph = node->g(torch::jit::attr::Subgraph);
      RETURN_IF_ERR(runSubGraph(*subgraph, subgraphInputs));

      CHECK_EQ(subgraph->outputs().size(), node->outputs().size());
      for (int i = 0; i < subgraph->outputs().size(); ++i) {
        shapeMap_[node->outputs()[i]] = shapeMap_[subgraph->outputs()[i]];
      }
      fusionNodeIndex += 1;
    } else {
      if (compilationMode_ && fusionNodeIndex == totalFusionNodes &&
          FLAGS_skipReferOperatorsOnCpu) {
        LOG(INFO)
            << "Skip shape inference for node after fusion groups with kind: "
            << node->kind().toQualString();
        continue;
      } else {
        RETURN_IF_ERR(shapeOnNode(node));
      }
    }
  }
  return Error::success();
}

Error ShapeInferenceEngine::run() {
  RETURN_ERR_IF_NOT(
      inputs_.size() == graph_.inputs().size() ||
          (inputs_.size() + 1 == graph_.inputs().size() &&
           graph_.inputs()[0]->type()->is_module()),
      "Number of inputs mismatch between Graph and actual inputs");
  if (FLAGS_print_shape_inference_graph) {
    printGraph(graph_, 0);
  }
  /// Put graph input into shape mapping
  RETURN_IF_ERR(runGraph(graph_, inputs_));
  if (!compilationMode_) {
    /// Extract output from shape mapping
    RETURN_IF_ERR(generateGraphOutputShape());
  }
  return Error::success();
}

void ShapeInferenceEngine::printShapeMap() {
  for (auto elem : shapeMap_) {
    std::cout << elem.first->debugName() << ":[ ";
    if (elem.second.listOfShape[0].type() == typeid(TensorShape)) {
      const TensorShape &shape = elem.second.shape<TensorShape>();
      for (auto value : shape) {
        std::cout << value << " ";
      }
    } else if (elem.second.listOfShape[0].type() == typeid(TensorListShape)) {
      const TensorListShape &shapes = elem.second.shape<TensorListShape>();
      for (auto shape : shapes) {
        std::cout << "[ ";
        for (auto value : shape) {
          std::cout << value << " ";
        }
        std::cout << "]";
      }
    } else {
      std::cout << "Type doesn't support yet.";
    }
    std::cout << "]" << std::endl;
  }
}

void ShapeInferenceEngine::printGraph(const torch::jit::Graph &graph,
                                      int64_t level) {
  int index = 0;
  for (auto *node : graph.nodes()) {
    if (node->hasAttribute(torch::jit::attr::Subgraph)) {
      auto subgraph = node->g(torch::jit::attr::Subgraph);
      LOG(INFO) << "graph level " << level << " node(fusion group) " << index
                << " " << node->kind().toQualString();
      printGraph(*subgraph, level + 1);
    } else {
      LOG(INFO) << "graph level " << level << " node(leaf) " << index << " "
                << node->kind().toQualString();
      for (int i = 0; i < node->inputs().size(); i++) {
        LOG(INFO) << "  input " << i << ": " << node->output(i)->debugName();
      }
      for (int i = 0; i < node->outputs().size(); i++) {
        LOG(INFO) << "  output " << i << ": " << node->output(i)->debugName();
      }
    }
    index++;
  }
}

/// If the input is tensor, store the shape info only;
/// Else If the input is bool or int, store the value, and set shape as 1.
/// Else if the input is intlist, store the intlist, and set shape as [sizeof
/// intlist, 1]
/// Else return an error
Error ShapeInferenceEngine::getGraphInputShapeType(
    const torch::jit::Graph &graph,
    const at::ArrayRef<torch::jit::IValue> &inputs) {
  int has_self = 0;
  if (!graph.inputs().empty() && graph.inputs()[0]->type()->is_module()) {
    has_self = 1;
  }
  for (auto i = 0; i < inputs.size(); i++) {
    auto gInName = graph.inputs()[i + has_self];
    auto input = inputs[i];
    TensorShape shape = {};
    std::vector<int64_t> intValue = {};
    c10::ScalarType dtype;

    if (input.isTensor()) {
      auto &ptTensor = input.toTensor();
      for (auto s : ptTensor.sizes()) {
        shape.emplace_back(s);
      }
      dtype = ptTensor.scalar_type();
    } else if (input.isBool() || input.isInt()) {
      shape = {1};
      intValue = {input.toInt()};
      dtype = input.isBool() ? c10::ScalarType::Bool : c10::ScalarType::Int;
    } else if (input.isIntList()) {
      intValue = input.toIntVector();
      shape = {static_cast<long>(intValue.size()), 1};
      dtype = c10::ScalarType::Int;
    } else {
      return MAKE_ERR("Input type doesn't support yet.");
    }
    shapeMap_[gInName].listOfShape.emplace_back(std::move(shape));
    shapeMap_[gInName].intValue = intValue;
    shapeMap_[gInName].dtype = dtype;
  }
  return Error::success();
}

Error ShapeInferenceEngine::generateGraphOutputShape() {
  for (auto output : graph_.outputs()) {
    auto it = shapeMap_.find(output);
    if (it == shapeMap_.end()) {
      LOG(WARNING) << "Some output shape is missing. Likely due to "
                      "blockList. Clearing the output shape vector.";
      outputShape_.clear();
      return Error::success();
    }
    outputShape_.emplace_back(it->second);
  }
  return Error::success();
}

/// The \p prim::Constant may have multiple types of output, eg.
/// int = prim::Constant[value=0]()
/// Float(1:1) = prim::Constant[value={0}]()
/// bool = prim::Constant[value=0]()
/// None = prim::Constant()
/// int[] = prim::Constant[value=[1,2,3]]()
/// Tensor = prim::Constant[value= <Tensor>]()
/// If the output is a tensor, return shape info and dtype;
/// Else, return the value and dtype.
Expected<TensorOutput>
ShapeInferenceEngine::primConstant(const torch::jit::Node *node) {

  TensorShape shapeOrValue;
  c10::ScalarType dtype;
  at::TypePtr type = node->output()->type();

  if (type->isSubtypeOf(at::FloatType::get())) {
    /// The float type will not affect the shape
    /// Set value as 1
    shapeOrValue = {1};
    dtype = c10::ScalarType::Float;
  } else if (type->isSubtypeOf(at::IntType::get())) {
    shapeOrValue = {node->i(at::attr::value)};
    dtype = c10::ScalarType::Int;
  } else if (type->isSubtypeOf(at::BoolType::get())) {
    shapeOrValue = {node->i(at::attr::value)};
    dtype = c10::ScalarType::Bool;
  } else if (type->isSubtypeOf(at::NoneType::get())) {
    shapeOrValue = {};
    dtype = c10::ScalarType::Undefined;
  } else if (type->isSubtypeOf(at::TensorType::get())) {
    at::Tensor t = node->t(at::attr::value);
    for (auto s : t.sizes()) {
      shapeOrValue.emplace_back(s);
    }
    dtype = t.scalar_type();
  } else if (type->isSubtypeOf(at::ListType::ofInts())) {
    dtype = c10::ScalarType::Int;
    shapeOrValue = node->ival(at::attr::value).toIntVector();
  } else if (type->isSubtypeOf(at::StringType::get())) {
    shapeOrValue = {1};
    dtype = c10::ScalarType::Char;
  } else {
    LOG(ERROR) << "Got " << *type;
    return MAKE_ERR("Type not supported");
  }
  TensorOutput output;
  output.shapeOrIntValues = shapeOrValue;
  output.dtype = dtype;
  return output;
}

/**
 * aten::add(Tensor self, Tensor or Scalar other, Scalar alpha=1) -> Tensor
 * aten::pow(Tensor self, Tensor or Scalar other, Scalar alpha=1) -> Tensor
 * aten::mul(Tensor self, Tensor or Scalar other, Scalar alpha=1) -> Tensor
 * variableMetas: 0: self, 1: other
 */
Expected<TensorOutput>
ShapeInferenceEngine::binaryOp(const MetaStack &variableMetas) {

  if (variableMetas.size() != 2 && variableMetas.size() != 3) {
    return MAKE_ERR("Expected two or three inputs shapes of this operation.");
  }

  const TensorShape &t0 = variableMetas[0].shape<TensorShape>();
  const TensorShape &t1 = variableMetas[1].shape<TensorShape>();

  auto d0 = t0.size();
  auto d1 = t1.size();

  TensorOutput output;
  output.dtype = variableMetas[0].dtype;

  /// One input is Scalar
  if (d1 == 1) {
    output.shapeOrIntValues = t0;
    return output;
  }

  size_t dim = std::max(d0, d1);
  TensorShape shape(dim);

  for (auto i = 0; i < dim; i++) {
    auto j = -1 - i;
    if (i >= d0 || t0[d0 + j] == 1) {
      shape[dim + j] = t1[d1 + j];
    } else if (i >= d1 || t1[d1 + j] == 1) {
      shape[dim + j] = t0[d0 + j];
    } else {
      if (t1[d1 + j] != t0[d0 + j]) {
        return MAKE_ERR(
            strFormat("The size of tensor a (%zu) must match the size of "
                      "tensor b (%zu)at non-singleton dimension 1.",
                      t0[d0 + j], t1[d1 + j]));
      }

      shape[dim + j] = t1[d1 + j];
    }
  }

  output.shapeOrIntValues = shape;
  return output;
}

/**
 * aten::mm(Tensor self, Tensor mat2) -> Tensor
 * variableMetas: 0: self, 1: mat2
 */
Expected<TensorOutput>
ShapeInferenceEngine::mm(const MetaStack &variableMetas) {

  RETURN_ERR_IF_NOT(variableMetas.size() == 2,
                    "Expected two inputs shapes of this operation.");

  const TensorShape &t0 = variableMetas[0].shape<TensorShape>();
  const TensorShape &t1 = variableMetas[1].shape<TensorShape>();

  if (!(t1.size() == 2 && t0.size() == 2)) {
    return MAKE_ERR("Expected 2-dimensional tensor.");
  }

  if (t0[1] != t1[0]) {
    return MAKE_ERR(
        strFormat("The size of tensor a (%zu) at dimension 1 must match the "
                  "size of tensor b (%zu) at dimension 0.",
                  t0[1], t1[0]));
  }

  TensorOutput output;
  TensorShape shape = {t0[0], t1[1]};
  output.shapeOrIntValues = shape;
  output.dtype = variableMetas[0].dtype;
  return output;
}

/**
 * aten::bmm(Tensor self, Tensor mat2) -> Tensor
 * variableMetas: 0: self, 1: mat2
 */
Expected<TensorOutput>
ShapeInferenceEngine::bmm(const MetaStack &variableMetas) {

  if (variableMetas.size() != 2) {
    return MAKE_ERR("Expected two inputs shapes of this operation.");
  }

  const TensorShape &t0 = variableMetas[0].shape<TensorShape>();
  const TensorShape &t1 = variableMetas[1].shape<TensorShape>();

  if (!(t0.size() == 3 && t1.size() == 3)) {
    return MAKE_ERR("Expected 3-dimensional tensor.");
  }

  if (t0[0] != t1[0]) {
    return MAKE_ERR("Expected tensors to have same size at dimension 0");
  }

  if (t0[2] != t1[1]) {
    return MAKE_ERR(strFormat("The size of tensor a (%zu) at dimension 2 must"
                              "match the size of tensor b (%zu) at dimension 1",
                              t0[2], t1[1]));
  }
  TensorOutput output;
  TensorShape shape = {t0[0], t0[1], t1[2]};
  output.shapeOrIntValues = shape;
  output.dtype = variableMetas[0].dtype;
  return output;
}

/**
 * aten::addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar
   alpha=1) -> Tensor
 * variableMetas: 0: self, 1: mat1, 2: mat2
 */
Expected<TensorOutput>
ShapeInferenceEngine::addmm(const MetaStack &variableMetas) {

  RETURN_ERR_IF_NOT(variableMetas.size() >= 3,
                    strFormat("Expected at least three inputs shapes, got %zu.",
                              variableMetas.size()));

  const VariableMeta &t0 = variableMetas[0];
  const VariableMeta &t1 = variableMetas[1];
  const VariableMeta &t2 = variableMetas[2];
  VariableMeta t;

  // For Scalar type, the shape.size() is 1
  if (t2.shape<TensorShape>().size() == 1) {
    t = variableMetas[1];
  } else {
    const MetaStack &mmShape = {t1, t2};
    TensorOutput mmOutput;
    ASSIGN_VALUE_OR_RETURN_ERR(mmOutput, mm(mmShape));
    t.listOfShape.emplace_back(std::move(mmOutput.shapeOrIntValues));
  }

  return binaryOp({t0, std::move(t)});
}

/**
 * aten::t(Tensor self) -> Tensor
 * refer to https://pytorch.org/docs/master/generated/torch.t
 */
Expected<TensorOutput> ShapeInferenceEngine::t(const MetaStack &variableMetas) {

  RETURN_ERR_IF_NOT(
      variableMetas.size() == 1,
      strFormat("Expected one input, got %zu.", variableMetas.size()));

  const TensorShape &t0 = variableMetas[0].shape<TensorShape>();

  auto d0 = t0.size();
  TensorOutput output;
  output.dtype = variableMetas[0].dtype;

  /// 0-D or 1-D tensor: Same shape
  if (d0 == 1) {
    output.shapeOrIntValues = t0;
    return output;
    /// 2-D tensor: Transpose
  } else if (d0 == 2) {
    TensorShape shape{t0[1], t0[0]};
    output.shapeOrIntValues = shape;
    return output;
    /// >2-D tensor: Invalid input
  } else {
    return MAKE_ERR(strFormat("Expected tensor <= 2-D, got %zu-D.", d0));
  }
}

Expected<TensorOutput>
ShapeInferenceEngine::sum(const MetaStack &variableMetas) {
  RETURN_ERR_IF_NOT(
      variableMetas.size() == 4,
      strFormat("Expected Four input, got %zu.", variableMetas.size()));
  // TODO: @hwwang T80910607 Only support None dtype (4th argument)
  RETURN_ERR_IF_NOT(variableMetas[3].intValue.size() == 0 and
                        variableMetas[3].dtype == c10::ScalarType::Undefined,
                    "Only support 4th arugment of aten::sum operator is None");
  const auto &t0 = variableMetas[0].shape<TensorShape>();
  auto dims = variableMetas[1].intValue;
  bool include_dim = variableMetas[2].intValue[0];

  TensorShape shape;
  for (int i = 0; i < t0.size(); i++) {
    if (std::find(dims.begin(), dims.end(), i) != dims.end()) {
      if (include_dim) {
        shape.push_back(1);
      } else {
        continue;
      }
    } else {
      shape.push_back(t0[i]);
    }
  }
  TensorOutput output;
  output.dtype = variableMetas[0].dtype;
  output.shapeOrIntValues = shape;
  return output;
}

/**
 * aten::transpose(Tensor self, int dim0, int dim1) => Tensor
 * variableMetas: 0: self, 1: dim0, 2: dim1
 * refer to https://pytorch.org/docs/master/generated/torch.transpose
 **/
Expected<TensorOutput>
ShapeInferenceEngine::transpose(const MetaStack &variableMetas) {
  if (variableMetas.size() != 3) {
    return MAKE_ERR(
        strFormat("Expect 3 inputs, get %zu", variableMetas.size()));
  }
  RETURN_ERR_IF_NOT(variableMetas[1].intValue.size() == 1,
                    "Expect 1 int dimension");
  RETURN_ERR_IF_NOT(variableMetas[2].intValue.size() == 1,
                    "Expect 1 int dimension");

  TensorShape shape = variableMetas[0].shape<TensorShape>();

  int64_t inDims = shape.size();
  int64_t dim0 = variableMetas[1].intValue[0];
  int64_t dim1 = variableMetas[2].intValue[0];

  // convert to positive dimension
  dim0 = at::maybe_wrap_dim(dim0, inDims);
  dim1 = at::maybe_wrap_dim(dim1, inDims);

  std::swap(shape[dim0], shape[dim1]);

  TensorOutput output;
  output.shapeOrIntValues = shape;
  output.dtype = variableMetas[0].dtype;
  return output;
}

/**
 * aten::cat(Tensors tensors, int dim=0) => Tensor
 * 0:variableMetas, 1: dim
 * refer to https://pytorch.org/docs/master/generated/torch.cat
 **/
Expected<TensorOutput>
ShapeInferenceEngine::cat(const MetaStack &variableMetas) {
  RETURN_ERR_IF_NOT(
      variableMetas.size() == 2,
      strFormat("Expected 2 inputs, got %zu.", variableMetas.size()));

  const TensorListShape &tensorListShapes =
      variableMetas[0].shape<TensorListShape>();
  std::vector<int64_t> shape = tensorListShapes[0];
  TensorOutput output;
  output.dtype = variableMetas[0].dtype;

  // Hanlde the single input case
  if (tensorListShapes.size() == 1) {
    output.shapeOrIntValues = shape;
    return output;
  }

  // Convert negtive dimension to positive, then check the dim range.
  int64_t dim = variableMetas[1].intValue[0];
  int64_t inDims = shape.size();
  dim = at::maybe_wrap_dim(dim, inDims);

  // Handle multiple input cases.
  // Verify all inputs dimenions are the same execpt the dimension applies cat.
  for (int i = 1; i < tensorListShapes.size(); ++i) {
    RETURN_ERR_IF_NOT(inDims == tensorListShapes[i].size(),
                      "All inputs must have the same number of dimensions.");
    for (int j = 0; j < inDims; j++) {
      if (j == dim) {
        continue;
      } else {
        RETURN_ERR_IF_NOT(
            shape[j] == tensorListShapes[i][j],
            strFormat("Sizes of tensors must match except in dimension %zu.",
                      dim));
      }
    }
  }
  for (int i = 1; i < tensorListShapes.size(); ++i)
    shape[dim] += tensorListShapes[i][dim];

  output.shapeOrIntValues = shape;
  return output;
}

/**
 * aten::flatten(Tensor self, int start_dim, int end_dim) => Tensor
 * variableMetas: 0: self, 1: start_dim, 2: end_dim
 * refer to: https://pytorch.org/docs/master/generated/torch.flatten
 **/
Expected<TensorOutput>
ShapeInferenceEngine::flatten(const MetaStack &variableMetas) {
  if (variableMetas.size() != 3) {
    return MAKE_ERR(
        strFormat("Expect 3 inputs, get %zu", variableMetas.size()));
  }
  RETURN_ERR_IF_NOT(variableMetas[1].intValue.size() == 1,
                    "Expect 1 int dimension");
  RETURN_ERR_IF_NOT(variableMetas[2].intValue.size() == 1,
                    "Expect 1 int dimension");

  const TensorShape &t = variableMetas[0].shape<TensorShape>();

  int64_t inDims = t.size();
  int64_t startDim = variableMetas[1].intValue[0];
  int64_t endDim = variableMetas[2].intValue[0];

  // convert to positive dimension
  startDim = at::maybe_wrap_dim(startDim, inDims);
  endDim = at::maybe_wrap_dim(endDim, inDims);

  if (startDim > endDim) {
    return MAKE_ERR("start dimension should not be larger than end dimension");
  }

  TensorShape shape;
  for (int i = 0; i < startDim; i++) {
    shape.push_back(t[i]);
  }
  int64_t flattenDim = 1;
  for (int i = startDim; i <= endDim; i++) {
    flattenDim *= t[i];
  }
  shape.push_back(flattenDim);
  for (int i = endDim + 1; i < inDims; i++) {
    shape.push_back(t[i]);
  }

  TensorOutput output;
  output.dtype = variableMetas[0].dtype;
  output.shapeOrIntValues = shape;
  return output;
}

/**
 * prim::ConstantChunk[int chunks, int dim](Tensor self) -> Tensors
 * variableMetas: 0: self
 */
Expected<TensorListOutput>
ShapeInferenceEngine::constantChunk(const MetaStack &variableMetas,
                                    int64_t chunks, int64_t dim) {

  RETURN_ERR_IF_NOT(
      variableMetas.size() == 1,
      strFormat("Expected one input, got %zu.", variableMetas.size()));

  const TensorShape &t = variableMetas[0].shape<TensorShape>();

  /// Convert dim into positive
  int64_t inDims = t.size();
  dim = at::maybe_wrap_dim(dim, inDims);

  /// For constant chunk, the size of the last chunk one may smaller than the
  /// others
  int64_t c = (t[dim] + chunks - 1) / chunks;
  int64_t r = t[dim] - c * (chunks - 1);

  TensorListShape resShapes;
  for (int i = 0; i < chunks; i++) {
    TensorShape shape = t;
    shape[dim] = (i == chunks - 1) ? r : c;
    resShapes.emplace_back(shape);
  }

  TensorListOutput output;
  output.dtype = variableMetas[0].dtype;
  output.shape = resShapes;
  return output;
}

static inline c10::ScalarType promote_skip_undefined(c10::ScalarType a,
                                                     c10::ScalarType b) {
  if (a == c10::ScalarType::Undefined) {
    return b;
  }
  if (b == c10::ScalarType::Undefined) {
    return a;
  }
  return c10::promoteTypes(a, b);
}

/**
 * prim::FusedConcat[int dim](Tensor self, Tensor mat1, Tensor mat2, ...) ->
 * Tensor variableMetas: 0: self, 1: mat1, 2: mat2, ...
 */
Expected<TensorOutput>
ShapeInferenceEngine::fusedConcat(const MetaStack &variableMetas, int64_t dim) {

  RETURN_ERR_IF_NOT(
      variableMetas.size() >= 1,
      strFormat("Expected at least 1 inputs, got %zu.", variableMetas.size()));

  TensorOutput output;
  output.dtype = variableMetas[0].dtype;

  if (variableMetas.size() == 1) {
    output.shapeOrIntValues = variableMetas[0].shape<TensorShape>();
    return output;
  }

  TensorShape shape = variableMetas[0].shape<TensorShape>();
  /// Convert negtive dimension to positive, then check the dim range.
  int64_t inDims = shape.size();
  dim = at::maybe_wrap_dim(dim, inDims);

  /// Handle multiple inputs cases
  for (int i = 1; i < variableMetas.size(); ++i) {
    const TensorShape &t = variableMetas[i].shape<TensorShape>();
    RETURN_ERR_IF_NOT(inDims == t.size(),
                      "All inputs must have the same number of dimensions.");
    for (int j = 0; j < inDims; j++) {
      if (j == dim) {
        shape[dim] += t[dim];
      } else {
        RETURN_ERR_IF_NOT(shape[j] == t[j],
                          strFormat("Sizes of tensors %lu != %lu must match "
                                    "except in dimension %zu.",
                                    shape[j], t[j], dim));
      }
    }
  }
  output.shapeOrIntValues = shape;
  return output;
}

/**
 * prim::FusedBroadcastConcat[int dim](Tensor self, Tensor mat1, Tensor mat2,
 * ...) -> Tensor variableMetas: 0: self, 1: mat1, 2: mat2, ...
 */
Expected<TensorOutput>
ShapeInferenceEngine::fusedBroadcastConcat(const MetaStack &variableMetas,
                                           int64_t dim) {
  RETURN_ERR_IF_NOT(
      variableMetas.size() >= 1,
      strFormat("Expected at least 1 inputs, got %zu.", variableMetas.size()));

  TensorOutput output;
  output.shapeOrIntValues = variableMetas[0].shape<TensorShape>();
  output.dtype = variableMetas[0].dtype;
  if (variableMetas.size() == 1) {
    return output;
  }

  /// Convert negtive dimension to positive, then check the dim range.
  int64_t inDims = output.shapeOrIntValues.size();
  dim = at::maybe_wrap_dim(dim, inDims);

  /// Handle multiple inputs cases
  for (int i = 1; i < variableMetas.size(); ++i) {
    const TensorShape &s = variableMetas[i].shape<TensorShape>();
    output.dtype = promote_skip_undefined(output.dtype, variableMetas[i].dtype);
    for (int j = 0; j < inDims; j++) {
      if (j == dim) {
        output.shapeOrIntValues[j] += s[j];
      } else if (s[j] != 1) {
        output.shapeOrIntValues[j] = s[j];
      }
    }
  }
  return output;
}

/**
 * aten::slice(Tensor self, int dim, int start, int end, int step)
 * variableMetas: 0: self, 1: dim, 2: start, 3: end, 4: step.
 */
Expected<TensorOutput>
ShapeInferenceEngine::slice(const MetaStack &variableMetas) {

  RETURN_ERR_IF_NOT(
      variableMetas.size() == 5,
      strFormat("Expected 5 inputs, got %zu.", variableMetas.size()));

  for (int i = 1; i < 5; i++) {
    RETURN_ERR_IF_NOT(variableMetas[i].intValue.size() == 1,
                      "Expected int in Slice.");
  }

  int64_t dim = variableMetas[1].intValue[0];
  int64_t start = variableMetas[2].intValue[0];
  int64_t end = variableMetas[3].intValue[0];
  int64_t step = variableMetas[4].intValue[0];

  TensorShape shape = variableMetas[0].shape<TensorShape>();
  int64_t inDims = shape[dim];

  TensorOutput output;
  output.dtype = variableMetas[0].dtype;

  /// Check if the start or end dim out of the input dimension
  if (start >= inDims || end <= -inDims) {
    shape[dim] = 0;
    output.shapeOrIntValues = shape;
    return output;
  }

  /// Convert start dim into positive
  if (start <= -inDims) {
    start = 0;
  } else if (start > -inDims && start < 0) {
    start += inDims;
  }

  /// Convert end dim into positive
  if (end > inDims) {
    end = inDims;
  } else if (end > -inDims && end < 0) {
    end += inDims;
  }

  if (start >= end) {
    shape[dim] = 0;
    output.shapeOrIntValues = shape;
    return output;
  }

  shape[dim] = (end - start) / step;
  if ((end - start) % step) {
    shape[dim] += 1;
  }
  output.shapeOrIntValues = shape;
  return output;
}

/**
 * aten::reshape(Tensor self, int[] shape) -> Tensor
 * variableMetas: 0: self, 1: shape
 */
Expected<TensorOutput>
ShapeInferenceEngine::reshape(const MetaStack &variableMetas) {

  RETURN_ERR_IF_NOT(
      variableMetas.size() == 2,
      strFormat("Expected two inputs shapes, got %zu.", variableMetas.size()));

  int64_t s0 = 1;
  int64_t s1 = 1;

  const TensorShape &t = variableMetas[0].shape<TensorShape>();

  /// Flag for multiple negative index
  int64_t negIndex = -1;
  for (auto i : t) {
    s0 *= i;
  }

  for (int i = 0; i < variableMetas[1].intValue.size(); i++) {
    s1 *= variableMetas[1].intValue[i];
    if (variableMetas[1].intValue[i] == -1) {
      if (negIndex == -1) {
        negIndex = i;
      } else {
        return MAKE_ERR("Unable to infer undetermined dimension");
      }
    }
  }

  RETURN_ERR_IF_NOT(s0 % s1 == 0, "Reshape size is invalid for input size.");

  TensorShape shape = variableMetas[1].intValue;

  if (negIndex != -1) {
    shape[negIndex] = -s0 / s1;
  }

  TensorOutput output;
  output.dtype = variableMetas[0].dtype;
  output.shapeOrIntValues = shape;
  return output;
}

/**
 * aten::permute(Tensor self, int[] shape) -> Tensor
 * variableMetas: 0: self, 1: shape
 */
Expected<TensorOutput>
ShapeInferenceEngine::permute(const MetaStack &variableMetas) {

  RETURN_ERR_IF_NOT(
      variableMetas.size() == 2,
      strFormat("Expected two inputs shapes, got %zu.", variableMetas.size()));

  const TensorShape &t = variableMetas[0].shape<TensorShape>();

  int64_t inDims = t.size();

  RETURN_ERR_IF_NOT(inDims == variableMetas[1].intValue.size(),
                    "Shuffle for permute must has the same number of "
                    "dimensions as the input tensor.");

  TensorShape shape;

  for (int64_t dim : variableMetas[1].intValue) {
    RETURN_ERR_IF_NOT(dim >= 0,
                      "Negative shuffle dimensions not supported by Glow yet.");
    RETURN_ERR_IF_NOT(
        dim < inDims,
        "All shuffle dimensions must be less than the rank of the input.");
    shape.emplace_back(t[dim]);
  }

  TensorOutput output;
  output.dtype = variableMetas[0].dtype;
  output.shapeOrIntValues = shape;
  return output;
}

/**
 * prim::ListContruct(Scalar/Bool/Tensor self, Scalar/Bool/Tensor v1,
 * Scalar/Bool/Tensor v2, ...) -> Scalar[]/Bool[]/Tensor[]
 * variableMetas: 0: self, 1: v1, 2: v2, ...
 */
Expected<TensorListOutput>
ShapeInferenceEngine::listConstruct(const MetaStack &variableMetas) {

  RETURN_ERR_IF_NOT(
      variableMetas.size() >= 1,
      strFormat("Expected at least 1 inputs, got %zu.", variableMetas.size()));

  TensorListShape listValueOrShape(1);
  if (variableMetas[0].intValue.size() == 1) {
    // scalar or bool
    for (auto ele : variableMetas) {
      RETURN_ERR_IF_NOT(ele.intValue.size() == 1,
                        "Expected int type input in listConstruct.");
      listValueOrShape[0].emplace_back(ele.intValue[0]);
    }
  } else {
    // tensor
    listValueOrShape.resize(variableMetas.size());
    for (int i = 0; i < variableMetas.size(); i++) {
      listValueOrShape[i] = variableMetas[i].shape<TensorShape>();
    }
  }

  TensorListOutput output;
  output.dtype = variableMetas[0].dtype;
  output.shape = listValueOrShape;
  return output;
}

/**
 * glow::fused_stack[dim=1](Tensor self, Tensor mat1, Tensor mat2, ...)
 * variableMetas: 0: self, 1: mat1, 2: mat2, ...
 */
Expected<TensorOutput>
ShapeInferenceEngine::fusedStack(const MetaStack &variableMetas, int64_t dim) {

  RETURN_ERR_IF_NOT(
      variableMetas.size() >= 1,
      strFormat("Expected at least 1 inputs, got %zu.", variableMetas.size()));

  TensorOutput output;
  TensorShape shape = variableMetas[0].shape<TensorShape>();
  output.dtype = variableMetas[0].dtype;

  if (variableMetas.size() == 1) {
    output.shapeOrIntValues = shape;
    return output;
  }
  int64_t inDims = shape.size();
  /// glow::fused_stack will add one more dim
  dim = at::maybe_wrap_dim(dim, inDims + 1);

  for (auto eleShape : variableMetas) {
    RETURN_ERR_IF_NOT(eleShape.shape<TensorShape>() == shape,
                      "All inputs must have same shape");
  }

  shape.insert(shape.begin() + dim, variableMetas.size());

  output.shapeOrIntValues = shape;
  return output;
}

/**
 * glow::fused_stack[dim=1](Tensor self, Tensor mat1, Tensor mat2, ...)
 * variableMetas: 0: self, 1: mat1, 2: mat2, ...
 */
Expected<TensorOutput>
ShapeInferenceEngine::fusedBroadcastStack(const MetaStack &variableMetas,
                                          int64_t dim) {
  RETURN_ERR_IF_NOT(
      variableMetas.size() >= 1,
      strFormat("Expected at least 1 inputs, got %zu.", variableMetas.size()));

  TensorOutput output;
  output.shapeOrIntValues = variableMetas[0].shape<TensorShape>();
  output.dtype = variableMetas[0].dtype;
  if (variableMetas.size() == 1) {
    return output;
  }

  int64_t inDims = output.shapeOrIntValues.size();

  /// Handle multiple inputs cases
  for (int i = 1; i < variableMetas.size(); ++i) {
    const TensorShape &s = variableMetas[i].shape<TensorShape>();
    output.dtype = promote_skip_undefined(output.dtype, variableMetas[i].dtype);
    for (int j = 0; j < inDims; j++) {
      if (s[j] != 1) {
        output.shapeOrIntValues[j] = s[j];
      }
    }
  }
  output.shapeOrIntValues.insert(output.shapeOrIntValues.begin() + dim,
                                 variableMetas.size());
  return output;
}

/**
 * glow::fused_split(Tensor input, int num_splits, int dim) -> Tensor[]
 */
Expected<TensorListOutput>
ShapeInferenceEngine::fusedSplit(const MetaStack &variableMetas) {
  RETURN_ERR_IF_NOT(
      variableMetas.size() == 3,
      strFormat("Expected Three input, got %zu.", variableMetas.size()));
  int64_t numSplit = variableMetas[1].intValue[0];
  int64_t dim = variableMetas[2].intValue[0];

  const auto &inputTensorShape = variableMetas[0].shape<TensorShape>();

  /// Convert dim into positive
  int64_t inDimSize = inputTensorShape.size();
  dim = at::maybe_wrap_dim(dim, inDimSize);

  RETURN_ERR_IF_NOT(
      inputTensorShape[dim] % numSplit == 0,
      strFormat("Expected dimension size could be evenly "
                "divieded by numSplit, got dimSize %long and numSplit %long",
                inputTensorShape[dim], numSplit));

  RETURN_ERR_IF_NOT(numSplit > 0,
                    strFormat("Expected numSplit is larger than 0"));

  TensorShape elementShape = inputTensorShape;
  elementShape[dim] = inputTensorShape[dim] / numSplit;
  TensorListShape resShapes(numSplit, elementShape);

  TensorListOutput output;
  output.dtype = variableMetas[0].dtype;
  output.shape = resShapes;
  return output;
}

/**
 * aten::_embedding_bag(Tensor weight,
 *                      Tensor indices,
 *                      Tensor offsets,
 *                      bool scale_grad_by_freq=False,
 *                      int mode=0,
 *                      bool sparse=False,
 *                      Tensor? per_sample_weights=None,
 *                      bool include_last_offset=False)
 *                      -> (Tensor, Tensor, Tensor, Tensor)
 */
/// Since the first output tensor is the result, and we only need the shape of
/// result Return the shape of the first tensor only
/// In glow, the include_last_offset is always True.
Expected<TensorOutput>
ShapeInferenceEngine::embeddingBag(const MetaStack &variableMetas) {

  RETURN_ERR_IF_NOT(
      variableMetas.size() == 8,
      strFormat("Expected 8 inputs, got %zu.", variableMetas.size()));

  TensorShape shape;

  const TensorShape &t0 = variableMetas[0].shape<TensorShape>();

  const TensorShape &t1 = variableMetas[1].shape<TensorShape>();

  const TensorShape &t2 = variableMetas[2].shape<TensorShape>();

  if (t1.size() == 1) {
    RETURN_ERR_IF_NOT(t2.size() == 1,
                      strFormat("Expected 1D offset, got %zu.", t2.size()));
    shape = {t2[0] - static_cast<int>(((hasEndOffset_) ? 1 : 0)), t0[1]};
  } else if (t1.size() == 2) {
    shape = {t1[0], t0[1]};
  } else {
    return MAKE_ERR("Only support 1D and 2D Input in Embedding bag.");
  }

  TensorOutput output;
  output.shapeOrIntValues = shape;
  output.dtype = c10::ScalarType::Float;
  return output;
}

/**
 * fb::glow_embedding_bag(Tensor indices,
 *                        Tensor offsets,
 *                        string? weight_qualname=None,
 *                        int num_embeddings,
 *                        int embedding_dim,
 *                        Tensor? per_sample_weights=None,
 *                        bool include_last_offset=True)
 *                        -> Tensor
 */
/// In glow, the include_last_offset is always True.
Expected<TensorOutput>
ShapeInferenceEngine::glowEmbeddingBag(const MetaStack &variableMetas) {

  RETURN_ERR_IF_NOT(
      variableMetas.size() == 7,
      strFormat("Expected 7 inputs, got %zu.", variableMetas.size()));

  TensorShape shape;

  const auto &indicesShape = variableMetas[0].shape<TensorShape>();

  const auto &offsetSahpe = variableMetas[1].shape<TensorShape>();

  int64_t embeddingDim = variableMetas[4].intValue[0];

  RETURN_ERR_IF_NOT(
      indicesShape.size() == 1,
      strFormat("Expected 1D input, got %zu.", indicesShape.size()));

  RETURN_ERR_IF_NOT(
      offsetSahpe.size() == 1,
      strFormat("Expected 1D offset, got %zu.", offsetSahpe.size()));

  shape = {offsetSahpe[0] - static_cast<int>(((hasEndOffset_) ? 1 : 0)),
           embeddingDim};

  TensorOutput output;
  output.shapeOrIntValues = shape;
  output.dtype = c10::ScalarType::Float;
  return output;
}

/**
 * fb::glow_embedding_bag_byte_rowwise_offsets(Tensor indices,
 *                        Tensor offsets,
 *                        string? weight_qualname=None,
 *                        int num_embeddings,
 *                        int embedding_dim,
 *                        Tensor? per_sample_weights=None,
 *                        bool include_last_offset=True)
 *                        -> Tensor
 */
/// In glow, the include_last_offset is always True.
Expected<TensorOutput>
ShapeInferenceEngine::quantizedGlowEmbeddingBagByteRowwiseOffsets(
    const MetaStack &variableMetas) {

  RETURN_ERR_IF_NOT(
      variableMetas.size() == 7,
      strFormat("Expected 7 inputs, got %zu.", variableMetas.size()));

  TensorShape shape;

  const auto &offsetShape = variableMetas[1].shape<TensorShape>();

  int64_t embeddingDim = variableMetas[4].intValue[0];

  shape = {offsetShape[0] - static_cast<int>(((hasEndOffset_) ? 1 : 0)),
           embeddingDim};

  TensorOutput output;
  output.shapeOrIntValues = shape;
  output.dtype = c10::ScalarType::Float;
  return output;
}

/**
 * fb::glow_embedding_bag_4bit_rowwise_offsets(Tensor indices,
 *                        Tensor offsets,
 *                        string? weight_qualname=None,
 *                        int num_embeddings,
 *                        int embedding_dim,
 *                        Tensor? per_sample_weights=None,
 *                        bool include_last_offset=True)
 *                        -> Tensor
 */
/// In glow, the include_last_offset is always True.
Expected<TensorOutput>
ShapeInferenceEngine::quantizedGlowEmbeddingBag4BitRowwiseOffsets(
    const MetaStack &variableMetas) {

  RETURN_ERR_IF_NOT(
      variableMetas.size() == 7,
      strFormat("Expected 7 inputs, got %zu.", variableMetas.size()));

  TensorShape shape;

  const auto &offsetShape = variableMetas[1].shape<TensorShape>();

  int64_t embeddingDim = variableMetas[4].intValue[0];

  shape = {offsetShape[0] - static_cast<int>(((hasEndOffset_) ? 1 : 0)),
           embeddingDim};

  TensorOutput output;
  output.shapeOrIntValues = shape;
  output.dtype = c10::ScalarType::Float;
  return output;
}

/**
 * fb::xl_embedding_bag(string? weight_id,
 *                      Tensor indices,
 *                      Tensor offsets,
 *                      bool scale_grad_by_freq=False,
 *                      int mode=0,
 *                      bool sparse=False,
 *                      Tensor? per_sample_weights=None,
 *                      bool include_last_offset=True,
 *                      int num_embeddings,
 *                      int embedding_dim,
 *                      -> (Tensor, Tensor, Tensor, Tensor)
 */
/// Since the first output tensor is the result, and we only need the shape of
/// result Return the shape of the first tensor only
/// In glow, the include_last_offset is always True.
Expected<TensorOutput>
ShapeInferenceEngine::xlEmbeddingBag(const MetaStack &variableMetas) {

  RETURN_ERR_IF_NOT(
      variableMetas.size() == 10,
      strFormat("Expected 10 inputs, got %zu.", variableMetas.size()));

  TensorShape shape;

  const auto &indicesShape = variableMetas[1].shape<TensorShape>();

  const auto &offsetShape = variableMetas[2].shape<TensorShape>();

  int64_t embeddingDim = variableMetas[9].intValue[0];

  RETURN_ERR_IF_NOT(
      indicesShape.size() == 1,
      strFormat("Expected 1D input, got %zu.", indicesShape.size()));

  RETURN_ERR_IF_NOT(
      offsetShape.size() == 1,
      strFormat("Expected 1D offset, got %zu.", offsetShape.size()));

  shape = {offsetShape[0] - static_cast<int>(((hasEndOffset_) ? 1 : 0)),
           embeddingDim};

  TensorOutput output;
  output.shapeOrIntValues = shape;
  output.dtype = c10::ScalarType::Float;
  return output;
}

/**
 * fb::xl_embedding_bag_byte_rowwise_offsets(string weight_id,
 *                        Tensor indices,
 *                        Tensor? offset_in=None,
 *                        bool? scale_grad_by_freq=False,
 *                        int mode=0,
 *                        bool pruned_weights=False,
 *                        Tensor? per_sample_weights=None,
 *                        str? compressed_indices_mapping_id=None,
 *                        bool include_last_offset=False,
 *                        int num_embeddings=0,
 *                        int embedding_dim=0)
 *                        -> Tensor
 */
/// In glow, the include_last_offset is always True.
Expected<TensorOutput>
ShapeInferenceEngine::quantizedXLEmbeddingBagByteRowwiseOffsets(
    const MetaStack &variableMetas) {

  RETURN_ERR_IF_NOT(
      variableMetas.size() == 11,
      strFormat("Expected 11 inputs, got %zu.", variableMetas.size()));

  TensorShape shape;

  const auto &offsetShape = variableMetas[2].shape<TensorShape>();

  int64_t embeddingDim = variableMetas[10].intValue[0];

  shape = {offsetShape[0] - static_cast<int>(((hasEndOffset_) ? 1 : 0)),
           embeddingDim};

  TensorOutput output;
  output.shapeOrIntValues = shape;
  output.dtype = c10::ScalarType::Float;
  return output;
}

/**
 * fb::xl_embedding_bag_4bit_rowwise_offsets(string weight_id,
 *                        Tensor indices,
 *                        Tensor? offset_in=None,
 *                        bool? scale_grad_by_freq=False,
 *                        int mode=0,
 *                        bool pruned_weights=False,
 *                        Tensor? per_sample_weights=None,
 *                        str? compressed_indices_mapping_id=None,
 *                        bool include_last_offset=False,
 *                        int num_embeddings=0,
 *                        int embedding_dim=0)
 *                        -> Tensor
 */
/// In glow, the include_last_offset is always True.
/// Remark: We have exactly the same input format and shape inference function
/// between xl_embedding_bag_4bit_rowwise_offsets and
/// xl_embedding_bag_byte_rowwise_offsets. Reuse the code here.
Expected<TensorOutput>
ShapeInferenceEngine::quantizedXLEmbeddingBag4BitRowwiseOffsets(
    const MetaStack &variableMetas) {
  return quantizedXLEmbeddingBagByteRowwiseOffsets(variableMetas);
}

/**
 * quantized::embedding_bag_byte_rowwise_offsets(Tensor weight,
 *                                        Tensor indices,
 *                                        Tensor offsets,
 *                                        bool scale_grad_by_freq=False,
 *                                        int mode=0,
 *                                        bool sparse=False,
 *                                        Tensor? per_sample_weights=None,
 *                                        Tensor? compressed_indices_mapping,
 *                                        bool include_last_offset=True)
 *                                        -> Tensor;
 */
/// In glow, the include_last_offset is always True.
Expected<TensorOutput>
ShapeInferenceEngine::quantizedEmbeddingBagByteRowwiseOffsets(
    const MetaStack &variableMetas) {

  RETURN_ERR_IF_NOT(
      variableMetas.size() == 9,
      strFormat("Expected 9 inputs, got %zu.", variableMetas.size()));

  const TensorShape &t0 = variableMetas[0].shape<TensorShape>();

  const TensorShape &t2 = variableMetas[2].shape<TensorShape>();

  /// variableMetas[0].shape[1] - 8 is to account for scale and bias
  /// 4-byte scale, 4-byte zero_offset
  TensorShape shape = {t2[0] - static_cast<int>(((hasEndOffset_) ? 1 : 0)),
                       t0[1] - 8};

  TensorOutput output;
  output.shapeOrIntValues = shape;
  output.dtype = c10::ScalarType::Float;
  return output;
}

/**
 * aten::chuck(Tensor self, int chunks, int dim) -> Tensor[]
 * refer to: https://pytorch.org/docs/master/generated/torch.chunk
 */
Expected<TensorListOutput>
ShapeInferenceEngine::chunk(const MetaStack &variableMetas) {

  RETURN_ERR_IF_NOT(
      variableMetas.size() == 3,
      strFormat("Expected one input, got %zu.", variableMetas.size()));

  const TensorShape &t = variableMetas[0].shape<TensorShape>();

  int64_t chunks = variableMetas[1].intValue[0];
  int64_t dim = variableMetas[2].intValue[0];

  /// Convert dim into positive
  int64_t inDims = t.size();
  dim = at::maybe_wrap_dim(dim, inDims);

  /// For constant chunk, the size of the last chunk one may smaller than the
  /// others
  int64_t c = (t[dim] + chunks - 1) / chunks;
  int64_t r = t[dim] - c * (chunks - 1);

  TensorListShape resShapes;
  for (int i = 0; i < chunks; i++) {
    TensorShape shape = t;
    shape[dim] = (i == chunks - 1) ? r : c;
    resShapes.emplace_back(shape);
  }

  TensorListOutput output;
  output.shape = resShapes;
  output.dtype = variableMetas[0].dtype;
  return output;
}

/*
 * glow::unpacked_quantized_linear(Tensor a_quant, Tensor w_quant, Tensor "
      "b, float r_scale, int r_zero_point) -> Tensor";

Input: (N, *, in_features) where * means any number of
additional dimensions
Weight: (out_features, in_features)
Bias: (out_features)
Output: (N, *, out_features)

 */
Expected<TensorOutput> ShapeInferenceEngine::glowUnpackedQuantizedLinear(
    const MetaStack &variableMetas) {
  RETURN_ERR_IF_NOT(
      variableMetas.size() == 5,
      strFormat("Expected 5 inputs,got %zu", variableMetas.size()));

  TensorShape outputShape;
  const TensorShape &inputShape = variableMetas[0].shape<TensorShape>();
  const int64_t &weightShape = variableMetas[1].shape<TensorShape>()[0];

  outputShape = inputShape;
  // Replace last element with weightShape
  if (outputShape.size() > 0) {
    outputShape.back() = weightShape;
  }

  TensorOutput output;
  output.shapeOrIntValues = outputShape;
  output.dtype = c10::ScalarType::QUInt8;
  return output;
}

/*
 * fb::embedding_bag_4bit_rowwise_offsets(Tensor weight,
 *                                        Tensor indices,
 *                                        Tensor offsets,
 *                                        bool scale_grad_by_freq=False,
 *                                        int mode=0,
 *                                        bool sparse=False,
 *                                        Tensor? per_sample_weights=None,
 *                                        Tensor? compressed_indices_mapping,
 *                                        bool include_last_offset=True)
 *                                        -> Tensor;
 */
/// In glow, the include_last_offset is always True.
Expected<TensorOutput> ShapeInferenceEngine::embeddingBag4BitRowwiseOffsets(
    const MetaStack &variableMetas) {

  RETURN_ERR_IF_NOT(
      variableMetas.size() == 9,
      strFormat("Expected 9 inputs, got %zu.", variableMetas.size()));

  /// variableMetas[0].shape[1] - 4 is to account for scale and offsets
  /// Note: 2-byte fp16 scale and 2-byte zero_offset
  /// *2 which accounts for the packed fp16 weights
  const TensorShape &weightShape = variableMetas[0].shape<TensorShape>();
  const TensorShape &offsetsShape = variableMetas[2].shape<TensorShape>();

  TensorOutput output;
  output.shapeOrIntValues = {offsetsShape[0] -
                                 static_cast<int>(((hasEndOffset_) ? 1 : 0)),
                             (weightShape[1] - 4) * 2};
  output.dtype = c10::ScalarType::Float;
  return output;
}

/**
 * aten::stack(Tensor[] tensors, int dim) -> Tensor
 * refer to: https://pytorch.org/docs/stable/generated/torch.stack
 */
Expected<TensorOutput>
ShapeInferenceEngine::stack(const MetaStack &variableMetas) {

  RETURN_ERR_IF_NOT(
      variableMetas.size() == 2,
      strFormat("Expected 2 input, got %zu.", variableMetas.size()));

  const TensorListShape &shapes = variableMetas[0].shape<TensorListShape>();
  TensorShape shape = shapes[0];

  // Convert negtive dimension to positive, then check the dim range.
  int64_t dim = variableMetas[1].intValue[0];
  int64_t inDims = shape.size();
  dim = at::maybe_wrap_dim(dim, inDims);

  // Verify the shapes of all input tensors.
  for (int i = 1; i < shapes.size(); i++) {
    RETURN_ERR_IF_NOT(shape == shapes[i],
                      "All tensors need to be of the same shape.");
  }

  shape.insert(shape.begin() + dim, shapes.size());

  TensorOutput output;
  output.shapeOrIntValues = shape;
  output.dtype = variableMetas[0].dtype;
  return output;
}

/**
 * prim::ListUnpack(Tensor[] tensors) -> Tensor, ..., Tensor
 */
Expected<TensorListOutput>
ShapeInferenceEngine::listUnpack(const MetaStack &variableMetas) {

  RETURN_ERR_IF_NOT(
      variableMetas.size() == 1,
      strFormat("Expected 1 input, got %zu.", variableMetas.size()));

  std::vector<TensorShape> shapes;
  const TensorListShape &t = variableMetas[0].shape<TensorListShape>();

  for (int i = 0; i < t.size(); i++) {
    shapes.emplace_back(t[i]);
  }

  TensorListOutput output;
  output.shape = shapes;
  output.dtype = variableMetas[0].dtype;
  return output;
}

/*
 * aten::to(Tensor input, int dtype, bool non_block, bool copy,
 * MemoryFormat? memory_format) -> Tensor
 */
Expected<TensorOutput>
ShapeInferenceEngine::to(const MetaStack &variableMetas) {

  RETURN_ERR_IF_NOT(
      variableMetas.size() == 4 || variableMetas.size() == 5,
      strFormat("Expected 4 or 5 input, got %zu.", variableMetas.size()));

  const TensorShape &t = variableMetas[0].shape<TensorShape>(); // input shape
  int32_t dtype = variableMetas[1].intValue[0];

  TensorOutput output;
  output.shapeOrIntValues = t;
  output.dtype = static_cast<c10::ScalarType>(dtype);
  return output;
}

/*
 * fb::lengths_to_offsets(Tensor lengths, bool include_last_offset) -> Tensor,
 */
Expected<TensorOutput>
ShapeInferenceEngine::lengthsToOffsets(const MetaStack &variableMetas) {

  RETURN_ERR_IF_NOT(
      variableMetas.size() == 2,
      strFormat("Expected 2 input, got %zu.", variableMetas.size()));

  const TensorShape &t = variableMetas[0].shape<TensorShape>(); // input shape
  RETURN_ERR_IF_NOT(t.size() == 1,
                    strFormat("Expected input dim is 1, got %zu.", t.size()));

  bool include_last_offset = variableMetas[1].intValue[0];
  RETURN_ERR_IF_NOT(include_last_offset == true,
                    strFormat("Expected include_last_offset is true, got %d.",
                              include_last_offset));

  TensorOutput output;
  output.shapeOrIntValues = t;
  output.shapeOrIntValues[0] += 1; // include last offset
  output.dtype = variableMetas[0].dtype;
  return output;
}

/*
 * prim::dtype(Tensor input) -> Int,
 */
Expected<TensorOutput>
ShapeInferenceEngine::primDtype(const MetaStack &variableMetas) {

  RETURN_ERR_IF_NOT(
      variableMetas.size() == 1,
      strFormat("Expected 1 input, got %zu.", variableMetas.size()));

  int dtype = static_cast<int>(variableMetas[0].dtype);

  TensorOutput output;
  output.shapeOrIntValues = {dtype};
  output.dtype = c10::ScalarType::Int;
  return output;
}

/*
 * fb::fast_gather(Tensor input, Tensor indices) -> Tensor
 */
Expected<TensorOutput>
ShapeInferenceEngine::fastGather(const MetaStack &variableMetas) {
  RETURN_ERR_IF_NOT(
      variableMetas.size() == 2,
      strFormat("Expected 2 input, got %zu.", variableMetas.size()));

  const auto &t0 = variableMetas[0].shape<TensorShape>();
  const auto &t1 = variableMetas[1].shape<TensorShape>();

  // suppose t0 = [d1, d2, ..., dm], t1 = [D1, D2, ..., Dn]
  // the result shape will be [D1, D2, ..., Dn, d2, ..., dm]
  TensorShape shape = t1;
  for (int i = 1; i < t0.size(); i++) {
    shape.emplace_back(t0[i]);
  }

  TensorOutput output;
  output.shapeOrIntValues = shape;
  output.dtype = variableMetas[0].dtype;
  return output;
}

/*
 * fb::lengths_range(Tensor input, int[]? shape, int? truncation_size) -> Int,
 * e.g. max_feature_length = 200
 * input: [2, 3]
 * original output: [0, 1, 0, 1, 2]
 * output after update: [0, 1, ..., 200, ] * 2
 */
Expected<TensorOutput>
ShapeInferenceEngine::lengthsRange(const MetaStack &variableMetas) {
  RETURN_ERR_IF_NOT(
      variableMetas.size() == 3,
      strFormat("Expected 3 inputs, got %zu.", variableMetas.size()));

  int max_feature_length;
  if (variableMetas[2].intValue.size() == 1) {
    max_feature_length = variableMetas[2].intValue[0];
  } else {
    max_feature_length = FLAGS_max_feature_length;
  }

  RETURN_ERR_IF_NOT(max_feature_length > 0,
                    strFormat("Expected max_feature_length > 0, got %d.",
                              max_feature_length));
  TensorOutput output;
  output.shapeOrIntValues = {variableMetas[0].shape<TensorShape>()[0] *
                             max_feature_length};
  output.dtype = variableMetas[0].dtype;
  return output;
}

/*
 * quantize_per_tensor(Tensor self, float scale, int zero_point, ScalarType
 * dtype) -> Tensor
 */
Expected<TensorOutput>
ShapeInferenceEngine::quantizePerTensor(const MetaStack &variableMetas) {
  RETURN_ERR_IF_NOT(
      variableMetas.size() == 4,
      strFormat("Expected 4 inputs, got %zu.", variableMetas.size()));
  const auto &inputShape = variableMetas[0].shape<TensorShape>();
  const int convertedTypeValue = variableMetas[3].intValue[0];
  TensorOutput output;
  output.shapeOrIntValues = inputShape;
  output.dtype = static_cast<c10::ScalarType>(convertedTypeValue);
  return output;
}

/*
 * aten::dequantize(Tensor qtensor) -> Tensor
 */
Expected<TensorOutput>
ShapeInferenceEngine::dequantize(const MetaStack &variableMetas) {
  RETURN_ERR_IF_NOT(
      variableMetas.size() == 1,
      strFormat("Expected 1 inputs, got %zu.", variableMetas.size()));
  const auto &inputShape = variableMetas[0].shape<TensorShape>();
  TensorOutput output;
  output.shapeOrIntValues = inputShape;
  output.dtype = c10::ScalarType::Float;
  return output;
}

/*
 * quantized::mul(%a_quant, %b_quant, %scale, %zero_point) -> Tensor
 */
Expected<TensorOutput>
ShapeInferenceEngine::quantizedMul(const MetaStack &variableMetas) {
  RETURN_ERR_IF_NOT(
      variableMetas.size() == 4,
      strFormat("Expected 4 inputs, got %zu.", variableMetas.size()));
  const auto &inputShape = variableMetas[0].shape<TensorShape>();
  const auto &weightShape = variableMetas[1].shape<TensorShape>();
  RETURN_ERR_IF_NOT(inputShape.size() > 0,
                    "Expected input shape size is larger than 0");
  RETURN_ERR_IF_NOT(weightShape.size() == 2,
                    "Expected weight is two dimensional tension");
  RETURN_ERR_IF_NOT(
      inputShape.back() == weightShape[1],
      "Expected the last dimension matches between input and weight");
  TensorOutput output;
  output.shapeOrIntValues = inputShape;
  output.shapeOrIntValues.back() = weightShape[0];
  output.dtype = variableMetas[0].dtype;
  return output;
}

/*
 * aten::matmul(Tensor input, Tensor other) -> Tensor
 */
Expected<TensorOutput>
ShapeInferenceEngine::matmul(const MetaStack &variableMetas) {
  RETURN_ERR_IF_NOT(
      variableMetas.size() == 2,
      strFormat("Expected 2 inputs, got %zu.", variableMetas.size()));
  const auto &inputOneShape = variableMetas[0].shape<TensorShape>();
  const auto &inputTwoShape = variableMetas[1].shape<TensorShape>();
  RETURN_ERR_IF_NOT(inputOneShape.size() == 3,
                    strFormat("Only support input as 3-d tensor, got %zu.",
                              inputOneShape.size()));
  RETURN_ERR_IF_NOT(inputTwoShape.size() == 3,
                    strFormat("Only support input as 3-d tensor, got %zu.",
                              inputTwoShape.size()));
  RETURN_ERR_IF_NOT(inputOneShape[2] == inputTwoShape[1],
                    "The 3rd dim of first input should be the same as 2nd dim "
                    "of second input.");
  TensorShape shapes;
  // TODO hwwang T81654300, add support for inputs with differnt dimensions.
  shapes.emplace_back(std::max(inputOneShape[0], inputTwoShape[0]));
  shapes.emplace_back(inputOneShape[1]);
  shapes.emplace_back(inputTwoShape[2]);
  TensorOutput output;
  output.shapeOrIntValues = shapes;
  output.dtype = variableMetas[0].dtype;
  return output;
}

/*
 * aten::layer_norm(Tensor input, int[] normalized_shape, Tensor? weight,
 * Tensor? bias, float eps, bool cudnn_enable) -> Tensor
 */
Expected<TensorOutput>
ShapeInferenceEngine::layerNorm(const MetaStack &variableMetas) {
  RETURN_ERR_IF_NOT(
      variableMetas.size() == 6,
      strFormat("Expected 6 inputs, got %zu.", variableMetas.size()));
  // The output is the same shape as input
  const auto &inputShape = variableMetas[0].shape<TensorShape>();
  TensorOutput output;
  output.shapeOrIntValues = inputShape;
  output.dtype = variableMetas[0].dtype;
  return output;
}

/*
 * aten::linear(Tensor input, Tensor weight, Tensor? bias) -> Tensor
 */
Expected<TensorOutput>
ShapeInferenceEngine::linear(const MetaStack &variableMetas) {
  RETURN_ERR_IF_NOT(
      variableMetas.size() == 3,
      strFormat("Expected 3 inputs, got %zu.", variableMetas.size()));
  const auto &inputShape = variableMetas[0].shape<TensorShape>();
  const auto &weightShape = variableMetas[1].shape<TensorShape>();
  RETURN_ERR_IF_NOT(inputShape.size() > 0,
                    "Expected input shape size is larger than 0");
  RETURN_ERR_IF_NOT(weightShape.size() == 2,
                    strFormat("Only support weight as 2-d tensor, got %zu.",
                              weightShape.size()));
  RETURN_ERR_IF_NOT(
      inputShape.back() == weightShape[1],
      "The last dim of input should be the same as 2nd dim of weight");
  TensorShape outputShape = inputShape;
  outputShape.back() = weightShape[0];
  TensorOutput output;
  output.shapeOrIntValues = outputShape;
  output.dtype = variableMetas[0].dtype;
  return output;
}

} // namespace glow
