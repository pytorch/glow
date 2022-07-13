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

#include <ATen/WrapDimUtils.h>
#include <iostream>
#include <string>
#include <torch/script.h>
#include <unordered_set>
#include <vector>

#include "ShapeInferenceEngine.h"

#include "folly/Overload.h"
#include "folly/String.h"
#include "glow/Support/Error.h"
#include "glow/Support/Support.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/FileSystem.h"

DEFINE_string(shapeInferenceOpBlocklist, "", "Ops to skip shape inference");
DEFINE_int32(max_feature_length, -1, "max feature length");
DEFINE_bool(print_shape_inference_graph, false,
            "print graph for shape inference debugging");
DEFINE_bool(skipReferOperatorsOnCpu, false,
            "Skip referring shapes running on CPU");

namespace glow {

namespace {
/* Print graph/shapes in GraphViz dot format for debugging */
class GraphDrawer {
public:
  GraphDrawer(const torch::jit::Graph &graph,
              const std::unordered_map<const torch::jit::Value *, VariableMeta>
                  &shapeMap)
      : graph_(graph), shapeMap_(shapeMap) {}

  void dump(std::ostream &os) {
    DCHECK(os) << "Failed to create file";
    drawNodes();
    os << "digraph DAG {\n\trankdir=TB;\n";
    // Dump vertices:
    for (auto &n : nodes_) {
      os << n << "\n";
    }

    // Dump edges:
    for (auto &e : edges_) {
      os << e << ";\n";
    }
    os << "}";
  }

private:
  void dumpShape(VariableMeta vm, std::ostream &os) {
    os << "[";
    DCHECK(vm.listOfShape.size() > 0);
    folly::variant_match(
        vm.listOfShape[0],
        [&](const TensorShape &shape) {
          for (auto i = 0; i < shape.size(); i++) {
            if (i) {
              os << ", ";
            }
            os << shape[i];
          }
        },
        [&](const TensorListShape &shapes) {
          for (auto shape : shapes) {
            os << "[";
            for (auto i = 0; i < shape.size(); i++) {
              if (i) {
                os << ", ";
              }
              os << shape[i];
            }
            os << "]";
          }
        },
        [&](const auto &) { os << "Unsupported shape type"; });
    os << "], ";
    os << "dtype: " << vm.dtype;
  }

  void dumpInputsOutputs(const torch::jit::Node *node, bool dumpInputs,
                         std::ostream &os) {
    auto label = dumpInputs ? "\\lInputs" : "\\lOutputs";
    auto values = dumpInputs ? node->inputs() : node->outputs();
    auto first = true;
    for (auto value : values) {
      if (first) {
        os << label << "\\l";
        first = false;
      } else {
        os << "\\l";
      }
      os << value->debugName() << " : ";
      if (shapeMap_.count(value)) {
        auto vm = shapeMap_.find(value)->second;
        if (vm.listOfShape.size() > 0) {
          dumpShape(vm, os);
        } else {
          os << "Missing shape";
        }
      } else {
        os << "Missing shape";
      }
      os << " (" << *value->type() << ")\\l";
    }
  }

  void colorNode(const torch::jit::Node *node, std::ostream &ss) {
    if (node->kind() == at::prim::Constant) {
      ss << "\tfillcolor=grey\n";
    } else {
      size_t colorHash =
          llvm::hash_value(llvm::StringRef(node->kind().toQualString()));
      ss << "\tfillcolor=" << glow::getDotFileNodeColor(colorHash) << "\n";
    }
  }

  void dumpNode(const torch::jit::Node *node, std::ostream &ss) {
    if (node_id_map_.count(node)) {
      return;
    }
    size_t node_id = node_id_map_.size();
    node_id_map_.insert({node, node_id});
    ss << node_id << "[\n";
    ss << "\tlabel = \"{{<Kind>" << node->kind().toQualString() << "} | {";
    dumpInputsOutputs(node, true, ss);
    dumpInputsOutputs(node, false, ss);
    for (auto *input : node->inputs()) {
      std::ostringstream edge;
      if (node_id_map_.count(input->node())) {
        auto from_id = node_id_map_.find(input->node())->second;
        edge << from_id << ":Result -> " << node_id << ":Kind";
        edges_.insert(edge.str());
      } else if (input->node()->kind() != at::prim::Param) {
        LOG(INFO) << "Missing node " << input->node()->kind().toQualString();
      }
    }
    ss << "}|{<Result>}}";
    ss << "\"\n";
    ss << "\tshape = \"record\"\n";
    ss << "\tstyle=\"filled,rounded\"\n";
    colorNode(node, ss);
    ss << "penwidth = 2];\n";
  }

  void addNode(const torch::jit::Node *node) {
    std::ostringstream ss;
    dumpNode(node, ss);
    nodes_.emplace_back(ss.str());
  }

  void drawNodes() {
    for (auto *node : graph_.nodes()) {
      addNode(node);
    }
  }

private:
  std::vector<std::string> nodes_;
  std::unordered_set<std::string> edges_;
  std::unordered_map<const torch::jit::Node *, int> node_id_map_;
  const torch::jit::Graph &graph_;
  const std::unordered_map<const torch::jit::Value *, VariableMeta> &shapeMap_;
};

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
} // namespace

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
  for (size_t i = 0; i < node->inputs().size(); ++i) {
    auto &input = node->inputs()[i];
    auto it = shapeMap_.find(input);
    if (it == shapeMap_.end()) {
      LOG(WARNING) << "Missing input #" << i << " '" << input->debugName()
                   << "' for node: " << *node;
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

  auto &mapping = getShapeSymbolMapping();
  if (!mapping.count(symbol)) {
    LOG(WARNING) << "Skip shape inference for unsupported op '" << symbol
                 << "' at " << *node;
    return Error::success();
  }

  if (blockList_.count(symbol)) {
    // Skip shape inference for this node. If other nodes have dependency
    // on this one then later their shape inference would fail explicitly.
    LOG(INFO) << "Skip shape inference for " << symbol << " due to block list";
    return Error::success();
  }
  /// Extract shapes of inputs from shape mapping
  MetaStack inputMetas;

  /// The outputs of each Op shape function include the shape and data
  /// type, and the shape could be either the shape or int value
  /// generated by prim::consant or prim::ListContruct.
  bool ret = getNodeInputShape(node, inputMetas);
  if (!ret) {
    LOG(WARNING) << "Skip shape inference for " << symbol
                 << " due to prior missing shapes";
    return Error::success();
  }

  auto symbolItem = mapping.find(symbol);
  if (symbolItem != mapping.end()) {
    return symbolItem->second.infer(this, inputMetas, node);
  }
  return Error::success();
}

Error ShapeInferenceEngine::ShapeInference::infer(
    ShapeInferenceEngine *engine, const MetaStack &meta,
    const torch::jit::Node *node) const {
  auto matchTensorOutputFn = folly::overload(
      [&](const InferenceFn0 &metastackInferFn) -> Expected<TensorOutput> {
        return metastackInferFn(meta);
      },
      [&](const InferenceFn1 &nodeInferFn) -> Expected<TensorOutput> {
        return nodeInferFn(node);
      },
      [&](const InferenceFn2 &metastackNodeInferFn) -> Expected<TensorOutput> {
        return metastackNodeInferFn(meta, node);
      },
      [&](const auto &) -> Expected<TensorOutput> {
        return MAKE_ERR("Shape inference misconfiguration. Inference function "
                        "expected to return a TensorOutput. ");
      });
  auto matchTensorListOutputFn = folly::overload(
      [&](const InferenceFn3 &metastackInferFn) -> Expected<TensorListOutput> {
        return metastackInferFn(meta);
      },
      [&](const InferenceFn4 &nodeInferFn) -> Expected<TensorListOutput> {
        return nodeInferFn(node);
      },
      [&](const InferenceFn5 &metastackNodeInferFn)
          -> Expected<TensorListOutput> {
        return metastackNodeInferFn(meta, node);
      },
      [&](const auto &) -> Expected<TensorListOutput> {
        return MAKE_ERR("Shape inference misconfiguration. Inference function "
                        "expected to return a TensorListOutput.");
      });
  auto matchElemOutputFn = folly::overload(
      [&](const InferenceFn6 &metastackInferFn) -> Expected<ElemOutput> {
        return metastackInferFn(meta);
      },
      [&](const auto &) -> Expected<ElemOutput> {
        return (MAKE_ERR("Shape inference misconfiguration. Inference function "
                         "expected to return an ElemOutput."));
      });
  return folly::variant_match(
      addShapeFn,
      [&](const AddShapeFn0 &addTensorOutput) -> Error {
        TensorOutput output;
        ASSIGN_VALUE_OR_RETURN_ERR(
            output, boost::apply_visitor(matchTensorOutputFn, inferenceFn));
        (engine->*(addTensorOutput))(node, output);
        return Error::success();
      },
      [&](const AddShapeFn1 &addTensorListOutput) -> Error {
        TensorListOutput output;
        ASSIGN_VALUE_OR_RETURN_ERR(
            output, boost::apply_visitor(matchTensorListOutputFn, inferenceFn));
        (engine->*(addTensorListOutput))(node, output);
        return Error::success();
      },
      [&](const AddShapeFn2 &addElemOutput) -> Error {
        ElemOutput output;
        ASSIGN_VALUE_OR_RETURN_ERR(
            output, boost::apply_visitor(matchElemOutputFn, inferenceFn));
        (engine->*(addElemOutput))(node, output);
        return Error::success();
      },
      [&](const auto &) {
        return MAKE_ERR("Unsupported types for addShapeFn");
      });
}

ShapeInferenceEngine::SymbolToFunctionMap
ShapeInferenceEngine::buildShapeSymbolMapping() {

  using SI = ShapeInferenceEngine;

  auto map = SymbolToFunctionMap({
      {"glow::fused_stack", ShapeInference(&fusedStack, &SI::addShapeDefault)},
      {"glow::fused_stack", ShapeInference(&fusedStack, &SI::addShapeDefault)},
      {"glow::fused_broadcast_stack",
       ShapeInference(&fusedBroadcastStack, &SI::addShapeDefault)},
      {"glow::fused_broadcast_stack_rc",
       ShapeInference(&fusedBroadcastStackRC, &SI::addShapeDefault)},
      {"glow::fused_broadcast_cat",
       ShapeInference(&fusedBroadcastConcat, &SI::addShapeDefault)},
      {"glow::fused_broadcast_cat_rc",
       ShapeInference(&fusedBroadcastConcatRC, &SI::addShapeDefault)},
      {"glow::fused_split",
       ShapeInference(&fusedSplit, &SI::addShapeDefaultList)},
      {"quantized::embedding_bag_byte_rowwise_offsets",
       ShapeInference(&quantizedEmbeddingBagByteRowwiseOffsets,
                      &SI::addShapeDefault)},
      {"quantized::embedding_bag_4bit_rowwise_offsets",
       ShapeInference(&embeddingBag4BitRowwiseOffsets, &SI::addShapeDefault)},
      {"glow::unpacked_quantized_linear",
       ShapeInference(&glowUnpackedQuantizedLinear, &SI::addShapeDefault)},
      {"fb::quantized_linear_unpacked_weight",
       ShapeInference(&glowUnpackedQuantizedLinear, &SI::addShapeDefault)},
      {"fb::quantized_linear_unpacked_weight_v2",
       ShapeInference(&glowUnpackedQuantizedLinear, &SI::addShapeDefault)},
      {"fb::lengths_to_offsets",
       ShapeInference(&lengthsToOffsets, &SI::addShapeDefault)},
      {"fb::simple_embedding_bag_sum",
       ShapeInference(&embeddingBag, &SI::addShapeBag)},
      {"fb::glow_embedding_bag",
       ShapeInference(&glowEmbeddingBag, &SI::addShapeDefault)},
      {"fb::glow_embedding_bag_byte_rowwise_offsets",
       ShapeInference(&quantizedGlowEmbeddingBagByteRowwiseOffsets,
                      &SI::addShapeDefault)},
      {"fb::glow_embedding_bag_4bit_rowwise_offsets",
       ShapeInference(&quantizedGlowEmbeddingBag4BitRowwiseOffsets,
                      &SI::addShapeDefault)},
      {"fb::xl_embedding_bag",
       ShapeInference(&xlEmbeddingBag, &SI::addShapeBag)},
      {"fb::xl_embedding_bag_byte_rowwise_offsets",
       ShapeInference(&quantizedXLEmbeddingBagByteRowwiseOffsets,
                      &SI::addShapeBag)},
      {"fb::xl_embedding_bag_4bit_rowwise_offsets",
       ShapeInference(&quantizedXLEmbeddingBag4BitRowwiseOffsets,
                      &SI::addShapeBag)},
      {"fb::fast_gather", ShapeInference(&fastGather, &SI::addShapeDefault)},
      {"fb::lengths_range",
       ShapeInference(&lengthsRange, &SI::addShapeDefault)},
      // Current shape inference function can handle both cases.
      {"fb::lengths_range_w_truncation_size",
       ShapeInference(&lengthsRange, &SI::addShapeDefault)},
      {"fb::quantize_per_tensor",
       ShapeInference(&quantizePerTensor, &SI::addShapeDefault)},
      {"aten::quantize_per_tensor",
       ShapeInference(&quantizePerTensor, &SI::addShapeDefault)},
      {"aten::dequantize", ShapeInference(&dequantize, &SI::addShapeDefault)},
      {"quantized::mul", ShapeInference(&quantizedMul, &SI::addShapeDefault)},
      {"prim::Constant", ShapeInference(&primConstant, &SI::addShapeConstant)},
      {"aten::tanh", ShapeInference(&unaryOp, &SI::addShapeDefault)},
      {"aten::relu", ShapeInference(&unaryOp, &SI::addShapeDefault)},
      {"aten::sigmoid", ShapeInference(&unaryOp, &SI::addShapeDefault)},
      {"aten::sign", ShapeInference(&unaryOp, &SI::addShapeDefault)},
      {"aten::abs", ShapeInference(&unaryOp, &SI::addShapeDefault)},
      {"aten::log1p", ShapeInference(&unaryOp, &SI::addShapeDefault)},
      {"aten::square", ShapeInference(&unaryOp, &SI::addShapeDefault)},
      {"aten::sqrt", ShapeInference(&unaryOp, &SI::addShapeDefault)},
      {"aten::sub", ShapeInference(&binaryOp, &SI::addShapeDefault)},
      {"aten::pow", ShapeInference(&binaryOp, &SI::addShapeDefault)},
      {"aten::mul", ShapeInference(&mul, &SI::addShapeDefault)},
      {"aten::add", ShapeInference(&add, &SI::addShapeDefault)},
      {"aten::div", ShapeInference(&binaryOp, &SI::addShapeDefault)},
      {"aten::rsub", ShapeInference(&binaryOp, &SI::addShapeDefault)},
      {"aten::mm", ShapeInference(&mm, &SI::addShapeDefault)},
      {"aten::addmm", ShapeInference(&addmm, &SI::addShapeDefault)},
      {"aten::bmm", ShapeInference(&bmm, &SI::addShapeDefault)},
      {"aten::t", ShapeInference(&t, &SI::addShapeDefault)},
      {"aten::transpose", ShapeInference(&transpose, &SI::addShapeDefault)},
      {"aten::flatten", ShapeInference(&flatten, &SI::addShapeDefault)},
      {"prim::FusedConcat", ShapeInference(&fusedConcat, &SI::addShapeDefault)},
      {"prim::ConstantChunk",
       ShapeInference(&constantChunk, &SI::addShapeDefaultList)},
      {"aten::chunk", ShapeInference(&chunk, &SI::addShapeChunk)},
      {"prim::ListConstruct",
       ShapeInference(&listConstruct, &SI::addShapeListConstruct)},
      {"aten::slice", ShapeInference(&slice, &SI::addShapeSlice)},
      {"aten::reshape", ShapeInference(&reshape, &SI::addShapeDefault)},
      {"aten::cat", ShapeInference(&cat, &SI::addShapeDefault)},
      {"aten::permute", ShapeInference(&permute, &SI::addShapeDefault)},
      {"aten::embedding_bag", ShapeInference(&embeddingBag, &SI::addShapeBag)},
      {"aten::matmul", ShapeInference(&matmul, &SI::addShapeDefault)},
      {"aten::layer_norm", ShapeInference(&layerNorm, &SI::addShapeDefault)},
      {"aten::linear", ShapeInference(&linear, &SI::addShapeDefault)},
      {"aten::stack", ShapeInference(&stack, &SI::addShapeDefault)},
      {"aten::to", ShapeInference(&to, &SI::addShapeDefault)},
      {"aten::sum", ShapeInference(&reduceOp, &SI::addShapeDefault)},
      {"aten::mean", ShapeInference(&reduceOp, &SI::addShapeDefault)},
      {"prim::dtype", ShapeInference(&primDtype, &SI::addShapeConstant)},
      {"prim::ListUnpack",
       ShapeInference(&listUnpack, &SI::addShapeDefaultList)},
      {"fbgemm_gpu::Fused8BitRowwiseQuantizedToFloat",
       ShapeInference(&fused8BitRowwiseQuantizedToFloat, &SI::addShapeDefault)},
      {"fb::compressed_indices_remap",
       ShapeInference(&compressedIndicesRemap, &SI::addShapeDefaultList)},
      {"fb::xl_compressed_indices_remap",
       ShapeInference(&compressedIndicesRemap, &SI::addShapeDefaultList)},
      {"quantized::embedding_bag_byte_unpack",
       ShapeInference(&embeddingBagByteUnpack, &SI::addShapeDefault)},
      {"fb::unsqueeze_n_times",
       ShapeInference(&unsqueezeNTimes, &SI::addShapeDefault)},
      {"fb::equally_split",
       ShapeInference(&equallySplit, &SI::addShapeDefaultList)},
      {"aten::squeeze", ShapeInference(&squeeze, &SI::addShapeDefault)},
      {"aten::narrow", ShapeInference(&narrow, &SI::addShapeDefault)},
      {"fb::index_hash", ShapeInference(&indexHash, &SI::addShapeDefault)},
      {"fb::bucketize", ShapeInference(&bucketize, &SI::addShapeDefault)},
      {"fb::expand_dims", ShapeInference(&expandDims, &SI::addShapeDefault)},
      {"aten::split_with_sizes",
       ShapeInference(&splitWithSizes, &SI::addShapeListConstruct)},
      {"aten::Int", ShapeInference(&inferInt, &SI::addShapeDefault)},
      {"prim::NumToTensor", ShapeInference(&numToTensor, &SI::addShapeDefault)},
      {"aten::size", ShapeInference(&size, &SI::addShapeConstant)},
      {"fb::scale_gradient",
       ShapeInference(&scaleGradient, &SI::addShapeDefault)},
      {"fb::to_lengths_to_offsets",
       ShapeInference(&toLengthsToOffsets, &SI::addShapeDefault)},
      {"aten::repeat", ShapeInference(&repeat, &SI::addShapeDefault)},
      {"aten::softmax", ShapeInference(&softmax, &SI::addShapeDefault)},
      {"aten::unsqueeze", ShapeInference(&unsqueeze, &SI::addShapeDefault)},
      {"aten::clamp_min", ShapeInference(&clampMin, &SI::addShapeDefault)},
      {"aten::norm", ShapeInference(&norm, &SI::addShapeDefault)},
      {"aten::expand_as", ShapeInference(&expandAs, &SI::addShapeDefault)},
      {"aten::argmin", ShapeInference(&argmin, &SI::addShapeDefault)},
      {"fb::sigrid_hash_precompute",
       ShapeInference(&sigridHashPrecompute, &SI::addShapeDefault)},
      {"aten::full_like", ShapeInference(&fullLike, &SI::addShapeDefault)},
  });
  return map;
}

const ShapeInferenceEngine::SymbolToFunctionMap &
ShapeInferenceEngine::getShapeSymbolMapping() {
  static const auto mapping = buildShapeSymbolMapping();
  return mapping;
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

void ShapeInferenceEngine::addShapeConstant(const torch::jit::Node *node,
                                            TensorOutput &output) {
  if (node->output()->type()->isSubtypeOf(at::TensorType::get())) {
    shapeMap_[node->output()].listOfShape.emplace_back(
        std::move(output.shapeOrIntValues));
    shapeMap_[node->output()].dtype = output.dtype;
  } else {
    shapeMap_[node->output()].listOfShape.emplace_back((TensorShape){1});
    shapeMap_[node->output()].intValue = std::move(output.shapeOrIntValues);
    shapeMap_[node->output()].dtype = output.dtype;
  }
}

void ShapeInferenceEngine::addShapeListConstruct(const torch::jit::Node *node,
                                                 TensorListOutput &output) {
  auto elem_type =
      node->output()->type()->cast<c10::ListType>()->getElementType();
  if (elem_type->kind() == at::TensorType::Kind ||
      (elem_type->kind() == at::OptionalType::Kind &&
       elem_type->cast<c10::OptionalType>()->getElementType()->kind() ==
           at::TensorType::Kind)) {
    shapeMap_[node->output()].listOfShape.emplace_back(std::move(output.shape));
    shapeMap_[node->output()].dtype = output.dtype;
  } else {
    shapeMap_[node->output()].listOfShape.emplace_back(
        (TensorShape){static_cast<long>(output.shape[0].size()), 1});
    shapeMap_[node->output()].intValue = std::move(output.shape[0]);
    shapeMap_[node->output()].dtype = output.dtype;
  }
}

void ShapeInferenceEngine::addShapeBag(const torch::jit::Node *node,
                                       TensorOutput &output) {
  shapeMap_[node->output(0)].listOfShape.emplace_back(
      std::move(output.shapeOrIntValues));
  shapeMap_[node->output(0)].dtype = output.dtype;
}

void ShapeInferenceEngine::addShapeChunk(const torch::jit::Node *node,
                                         TensorListOutput &output) {
  shapeMap_[node->output()].listOfShape.emplace_back(std::move(output.shape));
  shapeMap_[node->output()].dtype = output.dtype;
}

void ShapeInferenceEngine::addShapeDefault(const torch::jit::Node *node,
                                           TensorOutput &output) {
  if (output.scalar) {
    CHECK_EQ(output.shapeOrIntValues.size(), 1);
    shapeMap_[node->output()].listOfShape.emplace_back((TensorShape){1});
    shapeMap_[node->output()].intValue = std::move(output.shapeOrIntValues);
    shapeMap_[node->output()].dtype = output.dtype;
  } else {
    shapeMap_[node->output()].listOfShape.emplace_back(
        std::move(output.shapeOrIntValues));
    shapeMap_[node->output()].dtype = output.dtype;
  }
}

void ShapeInferenceEngine::addShapeDefaultList(const torch::jit::Node *node,
                                               TensorListOutput &output) {
  for (int i = 0; i < node->outputs().size(); i++) {
    shapeMap_[node->output(i)].listOfShape.emplace_back(
        std::move(output.shape[i]));
    if (output.dtypeList.size() > 0) {
      shapeMap_[node->output(i)].dtype = output.dtypeList[i];
    } else {
      shapeMap_[node->output(i)].dtype = output.dtype;
    }
  }
}

void ShapeInferenceEngine::addShapeSlice(const torch::jit::Node *node,
                                         ElemOutput &output) {
  folly::variant_match(
      output,
      [&](TensorOutput &tensorOutput) { addShapeDefault(node, tensorOutput); },
      [&](TensorListOutput &tensorListOutput) {
        shapeMap_[node->output()].listOfShape.emplace_back(
            std::move(tensorListOutput.shape));
        shapeMap_[node->output()].dtype = tensorListOutput.dtype;
      });
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
  {
    std::ofstream f{"shape_graph.txt"};
    f << graph;
  }
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
        if (it == shapeMap_.end()) {
          dumpGraph(graph);
          LOG(FATAL) << "missing input " << i->debugName().c_str();
        }
        // Only support tensor input for now
        // TODO Add support for other input types, e.g., tensor list
        if (it->second.dtype == at::ScalarType::QUInt8) {
          auto emptyTQ = at::_empty_affine_quantized(
              it->second.shape<TensorShape>(), at::ScalarType::QUInt8, 0, 1);
          subgraphInputs.emplace_back(emptyTQ);
        } else {
          subgraphInputs.emplace_back(
              torch::empty(it->second.shape<TensorShape>(),
                           torch::TensorOptions().dtype(it->second.dtype)));
        }
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

bool ShapeInferenceEngine::isSupportedNodeSymbol(const torch::jit::Node *node) {
  const auto kind = node->kind();
  const std::string symbol = kind.toQualString();
  auto &mapping = getShapeSymbolMapping();
  return mapping.find(symbol) != mapping.end();
}

std::unordered_set<std::string>
ShapeInferenceEngine::findUnsupportedGraphSymbols(bool skipLastFusionNode) {
  std::unordered_set<std::string> unsupportedSymbols;
  findUnsupportedGraphSymbols(graph_, unsupportedSymbols, skipLastFusionNode);
  return unsupportedSymbols;
}

void ShapeInferenceEngine::findUnsupportedGraphSymbols(
    const torch::jit::Graph &graph,
    std::unordered_set<std::string> &unsupportedSymbols,
    bool skipLastFusionNode) {

  int totalFusionNodes = 0;
  for (auto *node : graph.nodes()) {
    if (node->kind().toQualString() == fusionNodeSymbol_) {
      totalFusionNodes += 1;
    }
  }
  int fusionNodeIndex = 0;
  for (auto *node : graph.nodes()) {
    if (node->hasAttribute(torch::jit::attr::Subgraph)) {
      auto subgraph = node->g(torch::jit::attr::Subgraph);
      findUnsupportedGraphSymbols(*subgraph, unsupportedSymbols, false);
      fusionNodeIndex += 1;
    } else {
      if (fusionNodeIndex == totalFusionNodes && skipLastFusionNode) {
        LOG(INFO)
            << "Skip shape inference for node after fusion groups with kind: "
            << node->kind().toQualString();
        continue;
      } else if (!isSupportedNodeSymbol(node)) {
        unsupportedSymbols.insert(node->kind().toQualString());
      }
    }
  }
}

void ShapeInferenceEngine::printShapeMap() {
  for (auto elem : shapeMap_) {
    std::cout << elem.first->debugName() << ":[ ";
    folly::variant_match(
        elem.second.listOfShape[0],
        [&](const TensorShape &shape) {
          for (auto value : shape) {
            std::cout << value << " ";
          }
        },
        [&](const TensorListShape &shapes) {
          for (auto shape : shapes) {
            std::cout << "[ ";
            for (auto value : shape) {
              std::cout << value << " ";
            }
            std::cout << "]";
          }
        },
        [&](const auto &) { std::cout << "Type doesn't support yet."; });
    std::cout << "]" << std::endl;
  }
}

void ShapeInferenceEngine::dumpGraph(const torch::jit::Graph &graph) {
  std::string graphPath = "debug_shapes.dot";
  LOG(INFO) << "Dumping graph dot file to " << graphPath;
  GraphDrawer GD(graph, shapeMap_);
  std::ofstream file(graphPath);
  if (file) {
    GD.dump(file);
  } else {
    LOG(ERROR) << "Unable to open " << graphPath << "\n " << strerror(errno);
  }
  file.close();
  auto groupId = 0;
  std::vector<const torch::jit::Node *> fusions;
  for (auto *node : graph.nodes()) {
    if (node->hasAttribute(torch::jit::attr::Subgraph)) {
      std::string dotFileName =
          "debug_shapes_fusion_group_" + std::to_string(groupId) + ".dot";
      {
        GraphDrawer SGD(*node->g(torch::jit::attr::Subgraph), shapeMap_);
        LOG(INFO) << "Dumping fusion subgraph dot file to " << dotFileName;
        std::ofstream subgraphFile(dotFileName);
        if (subgraphFile) {
          SGD.dump(subgraphFile);
        } else {
          LOG(ERROR) << "Unable to open " << dotFileName << "\n "
                     << strerror(errno);
        }
      }

      std::string txtFileName =
          "debug_shapes_fusion_group_" + std::to_string(groupId) + ".txt";
      {
        LOG(INFO) << "Dumping fusion subgraph txt file to " << txtFileName;
        std::ofstream subgraphFile(txtFileName);
        if (subgraphFile) {
          subgraphFile << *node->g(torch::jit::attr::Subgraph);
        } else {
          LOG(ERROR) << "Unable to open " << txtFileName << "\n "
                     << strerror(errno);
        }
      }

      groupId++;
      fusions.push_back(node);
    }
  }

  // If more than one group, then we can find unsupported nodes in between
  if (fusions.size() > 1) {
    LOG(INFO) << "Found more than one fusion group";
    std::unordered_set<const torch::jit::Node *> allFusionOutputs;
    for (auto f : fusions) {
      for (auto o : f->outputs()) {
        for (auto use : o->uses()) {
          allFusionOutputs.insert(use.user);
        }
      }
    }

    std::ostringstream ss;
    for (auto n : allFusionOutputs) {
      ss << "\n" << *n;
    }
    LOG(INFO) << "Got " << allFusionOutputs.size()
              << " nodes following fusion groups" << ss.str();
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
        LOG(INFO) << "  input " << i << ": " << node->input(i)->debugName();
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
    } else if (input.isNone()) {
      dtype = c10::ScalarType::Undefined;
    } else {
      return MAKE_ERR("Input type isn't supported yet.");
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

bool ShapeInferenceEngine::isScalarInt(const VariableMeta &vm) {
  const auto &shape = vm.shape<TensorShape>();
  return shape.size() == 1 && shape[0] == 1 && vm.intValue.size() == 1 &&
         vm.dtype == c10::ScalarType::Int;
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
    std::ostringstream ss;
    ss << "Type '" << *type << "' is not supported.\nIt's coming from node "
       << *node;
    LOG(ERROR) << ss.str();
    return MAKE_ERR(ss.str());
  }
  TensorOutput output;
  output.shapeOrIntValues = shapeOrValue;
  output.dtype = dtype;
  return output;
}

// Shape inference for aten::tanh, aten::relu, aten::sigmoid, aten::abs,
// aten::sign, aten::log1p
Expected<TensorOutput>
ShapeInferenceEngine::unaryOp(const MetaStack &variableMetas) {
  RETURN_ERR_IF_NOT(variableMetas.size() == 1,
                    "Expected 1 input shape for operators.");
  TensorOutput output;
  output.shapeOrIntValues = variableMetas[0].shape<TensorShape>(),
  output.dtype = variableMetas[0].dtype;
  return output;
}

/**
 * aten::add(Tensor self, Tensor or Scalar other, Scalar alpha=1) -> Tensor
 * aten::pow(Tensor self, Tensor or Scalar other, Scalar alpha=1) -> Tensor
 * variableMetas: 0: self, 1: other
 */
Expected<TensorOutput>
ShapeInferenceEngine::binaryOp(const MetaStack &variableMetas,
                               const torch::jit::Node *ptNode) {

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
        std::ostringstream ss;
        ss << *ptNode;
        return MAKE_ERR(strFormat(
            "The size of tensor a (%zu) must match the size of "
            "tensor b (%zu)at non-singleton dimension 1. Offending op: %s",
            t0[d0 + j], t1[d1 + j], ss.str().c_str()));
      }

      shape[dim + j] = t1[d1 + j];
    }
  }

  output.shapeOrIntValues = shape;
  return output;
}

/**
 * aten::add(Tensor self, Tensor or Scalar other, Scalar alpha=1) -> Tensor
 * variableMetas: 0: self, 1: other
 */
Expected<TensorOutput>
ShapeInferenceEngine::add(const MetaStack &variableMetas,
                          const torch::jit::Node *ptNode) {
  RETURN_ERR_IF_NOT(
      variableMetas.size() == 2 || variableMetas.size() == 3,
      strFormat("Expected 2 or 3 inputs, got %zu", variableMetas.size()));

  if (isScalarInt(variableMetas[0]) && isScalarInt(variableMetas[1])) {
    TensorOutput output;
    output.shapeOrIntValues = {variableMetas[0].intValue[0] +
                               variableMetas[1].intValue[0]};
    output.dtype = c10::ScalarType::Int;
    output.scalar = true;
    return output;
  }

  return binaryOp(variableMetas, ptNode);
}

/**
 * aten::mul(Tensor self, Tensor or Scalar other, Scalar alpha=1) -> Tensor
 * variableMetas: 0: self, 1: other
 */
Expected<TensorOutput>
ShapeInferenceEngine::mul(const MetaStack &variableMetas,
                          const torch::jit::Node *ptNode) {
  if (variableMetas.size() == 2 && isScalarInt(variableMetas[0]) &&
      isScalarInt(variableMetas[1])) {
    TensorOutput output;
    output.shapeOrIntValues = {variableMetas[0].intValue[0] *
                               variableMetas[1].intValue[0]};
    output.dtype = c10::ScalarType::Int;
    output.scalar = true;
    return output;
  }

  return binaryOp(variableMetas, ptNode);
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
ShapeInferenceEngine::addmm(const MetaStack &variableMetas,
                            const torch::jit::Node *ptNode) {

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

  return binaryOp({t0, std::move(t)}, ptNode);
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
ShapeInferenceEngine::reduceOp(const MetaStack &variableMetas) {
  RETURN_ERR_IF_NOT(
      variableMetas.size() == 4,
      strFormat("Expected 4 inputs, got %zu.", variableMetas.size()));
  // TODO: @hwwang T80910607 Only support None dtype (4th argument)
  RETURN_ERR_IF_NOT(
      variableMetas[3].intValue.size() == 0 &&
          variableMetas[3].dtype == c10::ScalarType::Undefined,
      "Only support 4th argument of aten::sum/aten::mean operator is None");
  const auto &t0 = variableMetas[0].shape<TensorShape>();
  auto dims = variableMetas[1].intValue;
  bool includeDim = variableMetas[2].intValue[0];

  RETURN_ERR_IF_NOT(dims.size() == 1,
                    "Currently support only single axis for reduction");
  int dim = dims[0];
  if (dim < 0) {
    dim += t0.size();
  }

  TensorShape shape;
  for (int i = 0; i < t0.size(); i++) {
    if (dim == i) {
      if (includeDim) {
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
  // Verify all inputs dimenions are the same except the dimension applies
  // cat.
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
                                    const torch::jit::Node *node) {

  int64_t chunks = node->i(at::attr::chunks);
  int64_t dim = node->i(at::attr::dim);

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
ShapeInferenceEngine::fusedConcat(const MetaStack &variableMetas,
                                  const torch::jit::Node *node) {

  int64_t dim = node->i(at::attr::dim);

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
                                           const torch::jit::Node *node) {

  int64_t dim = node->i(at::attr::dim);

  RETURN_ERR_IF_NOT(
      variableMetas.size() >= 1,
      strFormat("Expected at least 1 inputs, got %zu.", variableMetas.size()));

  TensorOutput output;
  output.shapeOrIntValues = variableMetas[0].shape<TensorShape>();
  output.dtype = variableMetas[0].dtype;
  if (variableMetas.size() == 1) {
    return output;
  }

  /// Convert negative dimension to positive, then check the dim range.
  int64_t inDims = output.shapeOrIntValues.size();
  dim = at::maybe_wrap_dim(dim, inDims);

  /// Validating the input shapes
  bool goodShapes = true;
  for (int i = 1; i < variableMetas.size(); ++i) {
    const auto &s = variableMetas[i].shape<TensorShape>();
    if (output.shapeOrIntValues.size() != s.size()) {
      goodShapes = false;
    } else {
      for (int j = 0; j < s.size(); ++j) {
        if (j != dim && output.shapeOrIntValues[j] != s[j] &&
            output.shapeOrIntValues[j] != 1 && s[j] != 1) {
          goodShapes = false;
          break;
        }
      }
    }
    if (!goodShapes) {
      break;
    }
  }

  if (!goodShapes) {
    std::ostringstream ss;
    ss << "Got bad input shapes:";
    for (const auto &vm : variableMetas) {
      ss << "  [" << folly::join(",", vm.shape<TensorShape>()) << "]";
    }
    ss << " at node " << *node;
    return MAKE_ERR(ss.str());
  }

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

Expected<TensorOutput>
ShapeInferenceEngine::fusedBroadcastConcatRC(const MetaStack &variableMetas,
                                             const torch::jit::Node *node) {

  int64_t dim = node->i(at::attr::dim);

  RETURN_ERR_IF_NOT(
      variableMetas.size() >= 1,
      strFormat("Expected at least 1 inputs, got %zu.", variableMetas.size()));

  TensorOutput output;
  output.shapeOrIntValues = variableMetas[0].shape<TensorShape>();
  output.dtype = variableMetas[0].dtype;
  if (variableMetas.size() == 1) {
    return output;
  }

  /// Convert negative dimension to positive, then check the dim range.
  int64_t inDims = output.shapeOrIntValues.size();
  dim = at::maybe_wrap_dim(dim, inDims);
  auto batchIndicesShape =
      variableMetas[variableMetas.size() - 1].shape<TensorShape>();
  RETURN_ERR_IF_NOT(batchIndicesShape.size() == 1,
                    "Should have batchIndices input");
  auto requestCoalescing = batchIndicesShape[0] > 1;
  /// Validating the input shapes
  bool goodShapes = true;
  for (int i = 1; i < variableMetas.size() - 1; ++i) {
    const auto &s = variableMetas[i].shape<TensorShape>();
    if (output.shapeOrIntValues.size() != s.size()) {
      goodShapes = false;
    } else {
      for (int j = 0; j < s.size(); ++j) {
        if (j != dim && output.shapeOrIntValues[j] != s[j] &&
            output.shapeOrIntValues[j] != 1 && s[j] != 1 &&
            (!requestCoalescing || (requestCoalescing && j != 0))) {
          goodShapes = false;
          break;
        }
      }
      if (requestCoalescing && s[0] > batchIndicesShape[0]) {
        goodShapes = false;
        break;
      }
    }
    if (!goodShapes) {
      break;
    }
  }

  if (!goodShapes) {
    std::ostringstream ss;
    ss << "Got bad input shapes:";
    for (const auto &vm : variableMetas) {
      ss << "  [" << folly::join(",", vm.shape<TensorShape>()) << "]";
    }
    ss << " at node " << *node;
    return MAKE_ERR(ss.str());
  }

  /// Handle multiple inputs cases
  for (int i = 1; i < variableMetas.size() - 1; ++i) {
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
  if (requestCoalescing) {
    output.shapeOrIntValues[0] = batchIndicesShape[0];
  }
  return output;
}

/**
 * aten::slice(Tensor self, int dim, int start, int end, int step)
 * aten::slice(t[] l, int start, int end, int step) -> t[]
 * variableMetas: 0: self, 1: dim, 2: start, 3: end, 4: step.
 */
Expected<ElemOutput>
ShapeInferenceEngine::slice(const MetaStack &variableMetas) {

  RETURN_ERR_IF_NOT(
      variableMetas.size() == 5 || variableMetas.size() == 4,
      strFormat("Expected 5 inputs, got %zu.", variableMetas.size()));
  auto argSize = variableMetas.size();
  for (int i = 1; i < argSize; i++) {
    auto isNone = variableMetas[i].dtype == c10::ScalarType::Undefined;
    auto isInt = variableMetas[i].intValue.size() == 1;
    RETURN_ERR_IF_NOT(isNone || isInt, "Expected int in Slice.");
  }
  RETURN_ERR_IF_NOT(variableMetas[argSize - 1].intValue.size() == 1,
                    "Expected int in Slice.");

  int64_t dim = 0;
  int64_t start = 0;
  int64_t end = std::numeric_limits<long>::max();
  int64_t step = 1;
  ElemOutput output;
  if (argSize == 5) {
    dim = variableMetas[1].intValue[0];
    if (variableMetas[2].intValue.size() == 1) {
      start = variableMetas[2].intValue[0];
    }
    if (variableMetas[3].intValue.size() == 1) {
      end = variableMetas[3].intValue[0];
    }
    step = variableMetas[4].intValue[0];
    TensorOutput outputTensor;
    TensorShape shape = variableMetas[0].shape<TensorShape>();
    RETURN_ERR_IF_NOT(shape.size() > 0, "Shape expected to be nonempty.");
    int64_t inDims = shape[dim];
    outputTensor.dtype = variableMetas[0].dtype;

    /// Check if the start or end dim out of the input dimension
    if (start >= inDims || end <= -inDims) {
      shape[dim] = 0;
      outputTensor.shapeOrIntValues = shape;
      return outputTensor;
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
      outputTensor.shapeOrIntValues = shape;
      return outputTensor;
    }

    shape[dim] = (end - start) / step;
    if ((end - start) % step) {
      shape[dim] += 1;
    }
    outputTensor.shapeOrIntValues = shape;
    output = outputTensor;
  } else if (argSize == 4) {
    if (variableMetas[1].intValue.size() == 1) {
      start = variableMetas[1].intValue[0];
    }
    if (variableMetas[2].intValue.size() == 1) {
      end = variableMetas[2].intValue[0];
    }
    step = variableMetas[3].intValue[0];
    TensorListShape shape = variableMetas[0].shape<TensorListShape>();
    TensorListOutput outputTensorList;

    outputTensorList.dtype = variableMetas[0].dtype;
    int64_t inDims = shape.size();
    /// Check if the start or end dim out of the input dimension
    RETURN_ERR_IF_NOT(start < inDims && end > -inDims,
                      strFormat("Invalid start or end dims. Start is %ld and "
                                "end is %ld, while total dims are %ld",
                                start, end, inDims));

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

    RETURN_ERR_IF_NOT(
        start < end || (start == end && end < inDims),
        strFormat("Invalid start and end dims. Start is %ld and end is %ld",
                  start, end));

    TensorListShape outShapes;
    RETURN_ERR_IF_NOT(shape.size() >= end,
                      strFormat("Expected the shape size to be at least end. "
                                "Shape size is %ld and end is %ld",
                                shape.size(), end));
    for (auto i = start; i < end; i += step) {
      outShapes.emplace_back(shape[i]);
    }
    outputTensorList.shape = outShapes;
    output = outputTensorList;
  }
  return output;
}

/**
 * aten::reshape(Tensor self, int[] shape) -> Tensor
 * variableMetas: 0: self, 1: shape
 */
Expected<TensorOutput>
ShapeInferenceEngine::reshape(const MetaStack &variableMetas,
                              const torch::jit::Node *node) {
  RETURN_ERR_IF_NOT(
      variableMetas.size() == 2,
      strFormat("Expected two inputs, got %zu.", variableMetas.size()));

  int64_t s0 = 1;
  int64_t s1 = 1;

  const TensorShape &t = variableMetas[0].shape<TensorShape>();

  for (auto i : t) {
    s0 *= i;
  }

  int64_t negIndex = -1;
  for (int i = 0; i < variableMetas[1].intValue.size(); i++) {
    if (variableMetas[1].intValue[i] == -1) {
      if (negIndex != -1) {
        std::ostringstream ss;
        ss << "Found more than one negative index ["
           << folly::join(",", variableMetas[1].intValue) << "] in node "
           << *node;
        return MAKE_ERR(ss.str());
      }
      negIndex = i;
    } else {
      s1 *= variableMetas[1].intValue[i];
    }
  }

  if ((negIndex != -1 && s0 % s1 != 0) || (negIndex == -1 && s0 != s1)) {
    std::ostringstream ss;
    ss << "Invalid reshape from [" << folly::join(",", t) << "] to ["
       << folly::join(",", variableMetas[1].intValue) << "] in node " << *node;
    return MAKE_ERR(ss.str());
  }

  TensorShape shape = variableMetas[1].intValue;

  if (negIndex != -1) {
    shape[negIndex] = s0 / s1;
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
ShapeInferenceEngine::listConstruct(const MetaStack &variableMetas,
                                    const torch::jit::Node *node) {

  RETURN_ERR_IF_NOT(
      variableMetas.size() >= 1,
      strFormat("Expected at least 1 inputs, got %zu.", variableMetas.size()));

  TensorListShape listValueOrShape(1);
  if (variableMetas[0].intValue.size() == 1) {
    // scalar or bool
    for (auto ele : variableMetas) {
      RETURN_ERR_IF_NOT(ele.intValue.size() == 1,
                        strFormat("Expected int type input in listConstruct, "
                                  "but got %zu on node %s",
                                  ele.intValue.size(),
                                  node->kind().toDisplayString()));
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
ShapeInferenceEngine::fusedStack(const MetaStack &variableMetas,
                                 const torch::jit::Node *node) {

  int64_t dim = node->i(at::attr::dim);

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

Error validateConcatShape(const MetaStack &variableMetas, TensorOutput output,
                          const torch::jit::Node *node,
                          bool enableRequestCoalescing) {
  int64_t dim = node->i(at::attr::dim);
  auto num_inputs =
      enableRequestCoalescing ? variableMetas.size() - 1 : variableMetas.size();
  bool goodShapes = true;
  TensorShape batchIndicesShape;
  if (enableRequestCoalescing) {
    batchIndicesShape =
        variableMetas[variableMetas.size() - 1].shape<TensorShape>();
    RETURN_ERR_IF_NOT(batchIndicesShape.size() == 1,
                      "Should have batchIndices input");
  }
  for (int i = 1; i < num_inputs; ++i) {
    const auto &s = variableMetas[i].shape<TensorShape>();
    if (output.shapeOrIntValues.size() != s.size()) {
      goodShapes = false;
    } else {
      for (int j = 0; j < s.size(); ++j) {
        if (j != dim && output.shapeOrIntValues[j] != s[j] &&
            output.shapeOrIntValues[j] != 1 && s[j] != 1 &&
            (!enableRequestCoalescing || (enableRequestCoalescing && j != 0))) {
          goodShapes = false;
          break;
        }
      }
      if (enableRequestCoalescing) {
        if (s[0] > batchIndicesShape[0]) {
          goodShapes = false;
          break;
        }
      }
    }
    if (!goodShapes) {
      break;
    }
  }

  if (!goodShapes) {
    std::ostringstream ss;
    ss << "Got bad input shapes:";
    for (const auto &vm : variableMetas) {
      ss << " [" << folly::join(",", vm.shape<TensorShape>()) << "]";
    }
    ss << " at node " << *node;
    return MAKE_ERR(ss.str());
  }
  return Error::success();
}

/**
 * glow::fused_broadcast_stack[dim=1](Tensor self, Tensor mat1, Tensor mat2,
 * ...) variableMetas: 0: self, 1: mat1, 2: mat2, ...
 */
Expected<TensorOutput>
ShapeInferenceEngine::fusedBroadcastStack(const MetaStack &variableMetas,
                                          const torch::jit::Node *node) {

  int64_t dim = node->i(at::attr::dim);

  RETURN_ERR_IF_NOT(
      variableMetas.size() >= 1,
      strFormat("Expected at least 1 inputs, got %zu.", variableMetas.size()));

  TensorOutput output;
  output.shapeOrIntValues = variableMetas[0].shape<TensorShape>();
  output.dtype = variableMetas[0].dtype;

  /// Validating the input shapes
  RETURN_IF_ERR(validateConcatShape(variableMetas, output, node, false));

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
 * glow::fused_broadcast_stack_rc[dim=1](Tensor self, Tensor mat1, Tensor mat2,
 * ...) variableMetas: 0: self, 1: mat1, 2: mat2, ...
 */
Expected<TensorOutput>
ShapeInferenceEngine::fusedBroadcastStackRC(const MetaStack &variableMetas,
                                            const torch::jit::Node *node) {

  int64_t dim = node->i(at::attr::dim);
  RETURN_ERR_IF_NOT(
      variableMetas.size() >= 1,
      strFormat("Expected at least 1 inputs, got %zu.", variableMetas.size()));
  auto batchIndicesShape =
      variableMetas[variableMetas.size() - 1].shape<TensorShape>();
  RETURN_ERR_IF_NOT(batchIndicesShape.size() == 1,
                    "Expected 1d batchIndices input");
  TensorOutput output;
  output.shapeOrIntValues = variableMetas[0].shape<TensorShape>();
  output.dtype = variableMetas[0].dtype;

  /// Validating the input shapes
  RETURN_IF_ERR(validateConcatShape(variableMetas, output, node, true));

  /// Handle multiple inputs cases
  int64_t inDims = output.shapeOrIntValues.size();
  for (int i = 1; i < variableMetas.size() - 1; ++i) {
    const auto &s = variableMetas[i].shape<TensorShape>();
    output.dtype = promote_skip_undefined(output.dtype, variableMetas[i].dtype);
    for (int j = 0; j < inDims; j++) {
      if (s[j] != 1) {
        output.shapeOrIntValues[j] = s[j];
      }
    }
  }
  output.shapeOrIntValues[0] = batchIndicesShape[0];
  output.shapeOrIntValues.insert(output.shapeOrIntValues.begin() + dim,
                                 variableMetas.size() - 1);
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
      variableMetas.size() == 10 || variableMetas.size() == 11,
      strFormat("Expected 10 or 11 inputs, got %zu.", variableMetas.size()));

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
      variableMetas.size() == 11 || variableMetas.size() == 12,
      strFormat("Expected 11 or 12 inputs, got %zu.", variableMetas.size()));

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
 * fb::quantized_linear_unpacked_weight(Tensor a_quant, Tensor w_quant, "
      "Tensor b, float r_scale, int r_zero_point) -> Tensor";
 * fb::quantized_linear_unpacked_weight_v2(Tensor a_quant, Tensor w_quant, "
      "Tensor b, Tensor r_scale, Tensor r_zero_point) -> Tensor";
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
 * fb::to_lengths_to_offsets.prim_dtype(Tensor lengths, bool
 * include_last_offset, int? dtype) -> Tensor,
 * fb::to_lengths_to_offsets.dtype(Tensor lengths, bool include_last_offset,
 * ScalarType dtype) -> Tensor, Does not support Other variant since we can't
 * distinguish this input in shape inference from the prim None type input, and
 * the current shape support for aten::to does not support the Other overload.
 * Only supports include_last_offset=true per the current shape inference
 * support for lengths_to_offsets
 */
Expected<TensorOutput>
ShapeInferenceEngine::toLengthsToOffsets(const MetaStack &variableMetas) {
  RETURN_ERR_IF_NOT(
      variableMetas.size() == 3,
      strFormat("Expected 3 input, got %zu.", variableMetas.size()));

  const TensorShape &t = variableMetas[0].shape<TensorShape>(); // input shape
  RETURN_ERR_IF_NOT(t.size() == 1,
                    strFormat("Expected input dim is 1, got %zu.", t.size()));

  // only support include_last_offset=true, per the original shape inference
  // implementation for lengths_to_offsets
  bool include_last_offset = variableMetas[1].intValue[0];
  RETURN_ERR_IF_NOT(include_last_offset == true,
                    strFormat("Expected include_last_offset is true, got %d.",
                              include_last_offset));

  auto &dtype_int = variableMetas[2].intValue;

  TensorOutput output;
  output.shapeOrIntValues = t;
  output.shapeOrIntValues[0] += 1; // include last offset
  if (dtype_int.size() == 0) {
    output.dtype = variableMetas[0].dtype;
  } else {
    output.dtype = static_cast<c10::ScalarType>(dtype_int[0]);
  }
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
  output.scalar = true;
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
      variableMetas.size() >= 2,
      strFormat("Expected 2 or more inputs, got %zu.", variableMetas.size()));

  int max_feature_length;
  if (variableMetas.size() > 2 && variableMetas[2].intValue.size() == 1) {
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
 * aten::quantize_per_tensor(Tensor self, float scale, int zero_point,
 * ScalarType dtype) -> Tensor fb::quantize_per_tensor(Tensor self, Tensor
 * scale, Tensor zero_point, ScalarType dtype) -> Tensor
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
ShapeInferenceEngine::matmul(const MetaStack &variableMetas,
                             const torch::jit::Node *node) {
  RETURN_ERR_IF_NOT(
      variableMetas.size() == 2,
      strFormat("Expected 2 inputs, got %zu.", variableMetas.size()));
  const auto &inputOneShape = variableMetas[0].shape<TensorShape>();
  const auto &inputTwoShape = variableMetas[1].shape<TensorShape>();

  if (inputOneShape.size() != 2 && inputOneShape.size() != 3 &&
      inputOneShape.size() != 4) {
    std::ostringstream ss;
    ss << "Only support 2D, 3D or 4D for the first input, got "
       << inputOneShape.size() << " on node '" << *node << "'";
    return MAKE_ERR(ss.str());
  }
  if (inputTwoShape.size() != 2 && inputTwoShape.size() != 3 &&
      inputTwoShape.size() != 4) {
    std::ostringstream ss;
    ss << "Only support 2D, 3D or 4D for the second input, got "
       << inputTwoShape.size() << " on node '" << *node << "'";
    return MAKE_ERR(ss.str());
  }
  RETURN_ERR_IF_NOT(inputOneShape[inputOneShape.size() - 1] ==
                        inputTwoShape[inputTwoShape.size() - 2],
                    "The last dim of the first input should be the same as the "
                    "second last dim "
                    "of the second input.");

  // Populating the shape in reverse order
  TensorShape shapes;
  shapes.push_back(inputTwoShape[inputTwoShape.size() - 1]);
  shapes.push_back(inputOneShape[inputOneShape.size() - 2]);

  int indexOne = inputOneShape.size() - 3;
  int indexTwo = inputTwoShape.size() - 3;

  while (indexOne >= 0 && indexTwo >= 0) {
    auto dimOne = inputOneShape[indexOne];
    auto dimTwo = inputTwoShape[indexTwo];

    if (dimOne != 1 && dimTwo != 1 && dimOne != dimTwo) {
      return MAKE_ERR(strFormat("Bad input shapes: [%s] and [%s]",
                                folly::join(",", inputOneShape).c_str(),
                                folly::join(",", inputTwoShape).c_str()));
    }

    shapes.push_back(std::max(dimOne, dimTwo));
    --indexOne;
    --indexTwo;
  }

  while (indexOne >= 0) {
    shapes.push_back(inputOneShape[indexOne]);
    --indexOne;
  }

  while (indexTwo >= 0) {
    shapes.push_back(inputTwoShape[indexTwo]);
    --indexTwo;
  }

  std::reverse(shapes.begin(), shapes.end());

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

/*
 * fb::Fused8BitRowwiseQuantizedToFloat(Tensor input) -> Tensor
 */
Expected<TensorOutput> ShapeInferenceEngine::fused8BitRowwiseQuantizedToFloat(
    const MetaStack &variableMetas) {
  RETURN_ERR_IF_NOT(
      variableMetas.size() == 1,
      strFormat("Expected 1 inputs, got %zu.", variableMetas.size()));
  const auto &inputShape = variableMetas[0].shape<TensorShape>();
  RETURN_ERR_IF_NOT(inputShape.size() > 0,
                    "Expected input shape size is larger than 0");
  TensorShape outputShape = inputShape;
  // Substract zero_point and scale_size which are float number,
  // Each of them is float number which equals to 4 of int8 number.
  outputShape.back() -= (2 * 4);
  TensorOutput output;
  output.shapeOrIntValues = outputShape;
  output.dtype = c10::ScalarType::Float;
  return output;
}

Expected<TensorListOutput>
ShapeInferenceEngine::compressedIndicesRemap(const MetaStack &variableMetas) {
  RETURN_ERR_IF_NOT(
      variableMetas.size() == 4,
      strFormat("Expected 4 inputs, got %zu.", variableMetas.size()));
  const auto &indicesShape = variableMetas[0].shape<TensorShape>();
  const auto &offsetShape = variableMetas[2].shape<TensorShape>();
  const auto &weightShape = variableMetas[3].shape<TensorShape>();

  RETURN_ERR_IF_NOT(indicesShape.size() > 0,
                    "Expected indices shape size is larger than 0");
  RETURN_ERR_IF_NOT(offsetShape.size() > 0,
                    "Expected offset shape size is larger than 0");

  TensorListShape resShapes;

  resShapes.emplace_back(indicesShape);
  resShapes.emplace_back(offsetShape);
  resShapes.emplace_back(weightShape);

  TensorListOutput output;
  output.dtypeList.emplace_back(variableMetas[0].dtype);
  output.dtypeList.emplace_back(variableMetas[2].dtype);
  output.dtypeList.emplace_back(variableMetas[3].dtype);

  output.shape = resShapes;
  return output;
}

/*
 * quantized::embedding_bag_byte_unpack(Tensor weight) -> Tensor
 */
Expected<TensorOutput>
ShapeInferenceEngine::embeddingBagByteUnpack(const MetaStack &variableMetas) {
  RETURN_ERR_IF_NOT(
      variableMetas.size() == 1,
      strFormat("Expected 1 inputs, got %zu.", variableMetas.size()));
  const auto &inputShape = variableMetas[0].shape<TensorShape>();
  RETURN_ERR_IF_NOT(inputShape.size() > 0,
                    "Expected input shape size is larger than 0");
  TensorShape outputShape = inputShape;
  // Quantized tensor contains zero_point and scale_size which are float
  // number, Each of them is float number which equals to 4 of int8 number.
  // Unpacking will remove these two numbers.
  outputShape.back() -= (2 * 4);
  TensorOutput output;
  output.shapeOrIntValues = outputShape;
  output.dtype = c10::ScalarType::Float;
  return output;
}

/*
 * fb::unsqueeze_n_times(Tensor input, int n) -> Tensor
 * implementation logic:
 * for i in range(0, n):
 *   x = torch::unsqueeze(x, -1)
 */
Expected<TensorOutput>
ShapeInferenceEngine::unsqueezeNTimes(const MetaStack &variableMetas) {
  RETURN_ERR_IF_NOT(
      variableMetas.size() == 2,
      strFormat("Expected 2 input, got %zu.", variableMetas.size()));

  const auto &t = variableMetas[0].shape<TensorShape>();
  int64_t n = variableMetas[1].intValue[0];

  TensorShape shape = t;
  for (int i = 0; i < n; i++) {
    // according to unsqueeze_n_times definition, we always insert
    // a dimension of size one at the end
    shape.push_back(1);
  }
  TensorOutput output;
  output.shapeOrIntValues = shape;
  output.dtype = variableMetas[0].dtype;
  return output;
}

/*
 * fb::equally_split(Tensor input, int num_split, int dim) -> Tensor
 */
Expected<TensorListOutput>
ShapeInferenceEngine::equallySplit(const MetaStack &variableMetas) {
  RETURN_ERR_IF_NOT(
      variableMetas.size() == 3,
      strFormat("Expected 3 input, got %zu", variableMetas.size()));
  int64_t numSplit = variableMetas[1].intValue[0];
  int64_t dim = variableMetas[2].intValue[0];

  const auto &inputShape = variableMetas[0].shape<TensorShape>();
  RETURN_ERR_IF_NOT(inputShape.size() > 0,
                    "Expected input shape size is larger than 0");

  // Convert dim to positive
  dim = at::maybe_wrap_dim(dim, inputShape.size());
  RETURN_ERR_IF_NOT(
      inputShape[dim] % numSplit == 0,
      strFormat("Expected dimension size could be evenly divided by numSplit, "
                "got dimSize %long and numSplit %long",
                inputShape[dim], numSplit));

  TensorShape sliceShape = inputShape;
  sliceShape[dim] = inputShape[dim] / numSplit;
  TensorListShape outputShape(numSplit, sliceShape);

  TensorListOutput output;
  output.shape = outputShape;
  output.dtype = variableMetas[0].dtype;
  return output;
}

Expected<TensorOutput>
ShapeInferenceEngine::squeeze(const MetaStack &variableMetas) {
  RETURN_ERR_IF_NOT(
      variableMetas.size() == 2 || variableMetas.size() == 1,
      strFormat("Expected 1 or 2 input, got %zu.", variableMetas.size()));

  const auto &t = variableMetas[0].shape<TensorShape>();
  // Load dim parameter if provided
  int64_t dim = 0;
  bool dimProvided = false;
  if (variableMetas.size() == 2) {
    dimProvided = true;
    dim = variableMetas[1].intValue[0];
    if (dim < 0) {
      dim += t.size();
    }
  }

  TensorShape shape;
  for (int i = 0; i < t.size(); i++) {
    if (t[i] != 1 || (dimProvided && i != dim)) {
      shape.push_back(t[i]);
    }
  }
  TensorOutput output;
  output.shapeOrIntValues = shape;
  output.dtype = variableMetas[0].dtype;
  return output;
}

Expected<TensorOutput>
ShapeInferenceEngine::narrow(const MetaStack &variableMetas) {
  RETURN_ERR_IF_NOT(
      variableMetas.size() == 4,
      strFormat("Expected 4 input, got %zu.", variableMetas.size()));

  TensorShape shape = variableMetas[0].shape<TensorShape>();
  int64_t dim = variableMetas[1].intValue[0];
  int64_t length = variableMetas[3].intValue[0];

  TensorOutput output;
  RETURN_ERR_IF_NOT(
      shape.size() > dim,
      strFormat("Expected shape does not have dimension %zu.", dim));
  shape[dim] = length;
  output.shapeOrIntValues = shape;
  output.dtype = variableMetas[0].dtype;

  return output;
}

Expected<TensorOutput>
ShapeInferenceEngine::indexHash(const MetaStack &variableMetas) {
  RETURN_ERR_IF_NOT(
      variableMetas.size() == 3,
      strFormat("Expected 3 input, got %zu.", variableMetas.size()));

  TensorShape shape = variableMetas[0].shape<TensorShape>();

  TensorOutput output;
  output.shapeOrIntValues = shape;
  output.dtype = variableMetas[0].dtype;

  return output;
}

Expected<TensorOutput>
ShapeInferenceEngine::bucketize(const MetaStack &variableMetas) {
  RETURN_ERR_IF_NOT(
      variableMetas.size() == 2,
      strFormat("Expected 2 input, got %zu.", variableMetas.size()));

  TensorShape shape = variableMetas[0].shape<TensorShape>();

  TensorOutput output;
  output.shapeOrIntValues = shape;
  output.dtype = c10::ScalarType::Int;
  return output;
}

Expected<TensorOutput>
ShapeInferenceEngine::expandDims(const MetaStack &variableMetas) {
  RETURN_ERR_IF_NOT(
      variableMetas.size() == 2,
      strFormat("Expected 2 input, got %zu.", variableMetas.size()));

  TensorShape shape = variableMetas[0].shape<TensorShape>();
  std::vector<int64_t> dims = variableMetas[1].intValue;

  TensorShape outputShape;
  outputShape.resize(shape.size() + dims.size());
  for (auto dim : dims) {
    RETURN_ERR_IF_NOT(
        outputShape.size() > dim,
        strFormat("Output shape does not have dimension %zu.", dim));
    outputShape[dim] = 1;
  }

  auto iter = shape.begin();
  for (auto &i : outputShape) {
    if (i == 1) {
      continue;
    }
    i = *iter;
    ++iter;
  }
  TensorOutput output;
  output.shapeOrIntValues = outputShape;
  output.dtype = variableMetas[0].dtype;

  return output;
}

/*
 * aten::split_with_sizes(Tensor input, int[] chunk_sizes, int dim) ->
 * Tensor[]
 */
Expected<TensorListOutput>
ShapeInferenceEngine::splitWithSizes(const MetaStack &variableMetas) {
  RETURN_ERR_IF_NOT(
      variableMetas.size() == 3,
      strFormat("Expected 3 inputs, got %zu", variableMetas.size()));

  const auto &inputShape = variableMetas[0].shape<TensorShape>();

  const auto &chunkSizes = variableMetas[1].intValue;
  RETURN_ERR_IF_NOT(chunkSizes.size() > 0,
                    "Expected to have at least one chunk size");

  auto dim = variableMetas[2].intValue[0];
  // Convert dim to positive
  dim = at::maybe_wrap_dim(dim, inputShape.size());

  int64_t sumOfChunkSizes = 0;
  TensorListShape outputShape;
  for (auto chunkSize : chunkSizes) {
    TensorShape shape{inputShape};
    RETURN_ERR_IF_NOT(
        shape.size() > dim,
        strFormat("Expected shape does not have dimension %zu.", dim));
    shape[dim] = chunkSize;
    outputShape.push_back(std::move(shape));
    sumOfChunkSizes += chunkSize;
  }

  RETURN_ERR_IF_NOT(
      inputShape[dim] == sumOfChunkSizes,
      strFormat("Expected dimension size should be equal to the sum of chunk "
                "sizes, "
                "got dimSize %long and sum of chunk sizes %long",
                inputShape[dim], sumOfChunkSizes));

  TensorListOutput output;
  output.shape = outputShape;
  output.dtype = variableMetas[0].dtype;
  return output;
}

/*
 * aten::Int(Tensor input) -> int
 */
Expected<TensorOutput>
ShapeInferenceEngine::inferInt(const MetaStack &variableMetas) {
  RETURN_ERR_IF_NOT(isScalarInt(variableMetas[0]),
                    "Currently support only scalar int inputs");
  TensorOutput output;
  output.shapeOrIntValues = variableMetas[0].intValue;
  output.dtype = variableMetas[0].dtype;
  output.scalar = true;
  return output;
}

/*
 * prim::NumToTensor(int num) -> Tensor
 */
Expected<TensorOutput>
ShapeInferenceEngine::numToTensor(const MetaStack &variableMetas,
                                  const torch::jit::Node *node) {
  RETURN_ERR_IF_NOT(isScalarInt(variableMetas[0]),
                    "Currently support only scalar int inputs");
  TensorOutput output;
  output.shapeOrIntValues = variableMetas[0].intValue;
  output.dtype = variableMetas[0].dtype;
  output.scalar = true;

  return output;
}

/*
 * aten::size(Tensor input, int dim) -> int
 */
Expected<TensorOutput>
ShapeInferenceEngine::size(const MetaStack &variableMetas) {
  RETURN_ERR_IF_NOT(
      variableMetas.size() == 2,
      strFormat("Expected 2 inputs, got %zu", variableMetas.size()));

  const auto &inputShape = variableMetas[0].shape<TensorShape>();

  auto dim = variableMetas[1].intValue[0];
  // Convert dim to positive
  dim = at::maybe_wrap_dim(dim, inputShape.size());

  TensorOutput output;
  output.shapeOrIntValues = {inputShape[dim]};
  output.dtype = c10::ScalarType::Int;

  return output;
}

/*
 * fb::scale_gradient(Tensor input, float scale) -> Tensor
 */
Expected<TensorOutput>
ShapeInferenceEngine::scaleGradient(const MetaStack &variableMetas) {
  RETURN_ERR_IF_NOT(
      variableMetas.size() == 2,
      strFormat("Expected 2 inputs, got %zu", variableMetas.size()));

  TensorShape shape = variableMetas[0].shape<TensorShape>();

  TensorOutput output;
  output.shapeOrIntValues = shape;
  output.dtype = variableMetas[0].dtype;
  return output;
}

/*
 * aten::repeat(Tensor input, int[] repeats) -> Tensor
 */
Expected<TensorOutput>
ShapeInferenceEngine::repeat(const MetaStack &variableMetas) {
  RETURN_ERR_IF_NOT(
      variableMetas.size() == 2,
      strFormat("Expected 2 inputs, got %zu", variableMetas.size()));

  const TensorShape &t = variableMetas[0].shape<TensorShape>();
  auto repeats = variableMetas[1].intValue;

  RETURN_ERR_IF_NOT(
      repeats.size() >= t.size(),
      strFormat("Number of dimensions of repeat dims can not be smaller "
                "than number of dimensions of tensor: %zu vs %zu",
                repeats.size(), t.size()));

  std::vector<int64_t> resultShape;
  size_t diff = repeats.size() - t.size();
  for (int i = 0; i < repeats.size(); ++i) {
    if (i < diff) {
      resultShape.push_back(repeats[i]);
    } else {
      resultShape.push_back(repeats[i] * t[i - diff]);
    }
  }

  TensorOutput output;
  output.dtype = variableMetas[0].dtype;
  output.shapeOrIntValues = resultShape;
  return output;
}

/*
 * aten::softmax(Tensor input, int dim, torch.dtype dtype) -> Tensor
 */
Expected<TensorOutput>
ShapeInferenceEngine::softmax(const MetaStack &variableMetas) {
  RETURN_ERR_IF_NOT(
      variableMetas.size() == 2 || variableMetas.size() == 3,
      strFormat("Expected 2 or 3 inputs, got %zu", variableMetas.size()));

  TensorShape shape = variableMetas[0].shape<TensorShape>();

  TensorOutput output;
  output.shapeOrIntValues = shape;
  output.dtype = variableMetas[0].dtype;
  return output;
}

/*
 * aten::unsqueeze(Tensor input, int dim) -> Tensor
 *
 * E.g. x.size() = torch.Size([2, 4])
 * torch.unsqueeze(x, 0).size() = torch.Size([1, 2, 4])
 * torch.unsqueeze(x, 1).size() = torch.Size([2, 1, 4])
 * torch.unsqueeze(x, 2).size() = torch.Size([2, 4, 1])
 */
Expected<TensorOutput>
ShapeInferenceEngine::unsqueeze(const MetaStack &variableMetas) {
  RETURN_ERR_IF_NOT(
      variableMetas.size() == 2,
      strFormat("Expected 2 inputs, got %zu", variableMetas.size()));

  const auto inputShape = variableMetas[0].shape<TensorShape>();

  int dim = variableMetas[1].intValue[0];
  // Wrapping for aten::unsqueeze works differently than for most of the ops.
  if (dim < 0) {
    dim += (inputShape.size() + 1);
  }

  // Add 1 at dim-th index.
  // E.g. x.size() = torch.Size([2, 4]), dim = 1
  // shapes = torch.Size([2, 1, 4])
  TensorShape shape = inputShape;
  shape.insert(shape.begin() + dim, 1);

  TensorOutput output;
  output.shapeOrIntValues = shape;
  output.dtype = variableMetas[0].dtype;
  return output;
}

/*
 * aten::clamp_min(Tensor self, Scalar min) -> Tensor
 */
Expected<TensorOutput>
ShapeInferenceEngine::clampMin(const MetaStack &variableMetas) {
  RETURN_ERR_IF_NOT(
      variableMetas.size() == 2,
      strFormat("Expected 2 inputs, got %zu", variableMetas.size()));

  TensorShape shape = variableMetas[0].shape<TensorShape>();

  TensorOutput output;
  output.shapeOrIntValues = shape;
  output.dtype = variableMetas[0].dtype;
  return output;
}

/*
 * For now supporting only this version:
 * aten::norm.ScalarOpt_dim(Tensor self, Scalar? p, int[1] dim, bool
 * keepdim=False) -> Tensor
 */
Expected<TensorOutput>
ShapeInferenceEngine::norm(const MetaStack &variableMetas) {
  RETURN_ERR_IF_NOT(
      variableMetas.size() == 4,
      strFormat("Expected 4 inputs, got %zu", variableMetas.size()));

  TensorShape shape = variableMetas[0].shape<TensorShape>();

  const auto normDims = variableMetas[2].intValue;
  RETURN_ERR_IF_NOT(
      normDims.size() == 1,
      strFormat("Expected only one dim, got %zu", normDims.size()));
  RETURN_ERR_IF_NOT(
      normDims[0] >= 0 && normDims[0] < shape.size(),
      strFormat("Expected dim to be in range [0..%zu)", shape.size()));

  const auto keepDimValues = variableMetas[3].intValue;
  RETURN_ERR_IF_NOT(keepDimValues.size() == 1,
                    strFormat("Expected only one value for keepDim, got %zu",
                              normDims.size()));
  RETURN_ERR_IF_NOT(keepDimValues[0] == 1,
                    "Only supporting keepdim=True for now");

  TensorOutput output;
  output.shapeOrIntValues = shape;
  output.shapeOrIntValues[normDims[0]] = 1;
  output.dtype = variableMetas[0].dtype;
  return output;
}

/*
 * aten::expand_as(Tensor(a) self, Tensor other) -> Tensor(a)
 */
Expected<TensorOutput>
ShapeInferenceEngine::expandAs(const MetaStack &variableMetas) {
  RETURN_ERR_IF_NOT(
      variableMetas.size() == 2,
      strFormat("Expected 2 inputs, got %zu", variableMetas.size()));

  TensorShape shape = variableMetas[1].shape<TensorShape>();

  TensorOutput output;
  output.shapeOrIntValues = shape;
  output.dtype = variableMetas[0].dtype;
  return output;
}

/*
 * aten::argmin(Tensor self, int? dim=None, bool keepdim=False) -> Tensor
 */
Expected<TensorOutput>
ShapeInferenceEngine::argmin(const MetaStack &variableMetas) {
  RETURN_ERR_IF_NOT(
      variableMetas.size() == 3,
      strFormat("Expected 3 inputs, got %zu", variableMetas.size()));

  TensorShape shape = variableMetas[0].shape<TensorShape>();

  const auto dimValues = variableMetas[1].intValue;
  RETURN_ERR_IF_NOT(
      dimValues.size() == 1,
      strFormat("Expected only one dim value, got %zu", dimValues.size()));

  const auto keepDimValues = variableMetas[2].intValue;
  RETURN_ERR_IF_NOT(keepDimValues.size() == 1,
                    strFormat("Expected only one value for keepDim, got %zu",
                              keepDimValues.size()));
  RETURN_ERR_IF_NOT(keepDimValues[0] == 0,
                    "Only supporting keepdim=False for now");

  TensorOutput output;
  output.shapeOrIntValues = shape;
  output.shapeOrIntValues.erase(output.shapeOrIntValues.begin() + dimValues[0]);
  output.dtype = at::ScalarType::Long;
  return output;
}

/*
 * fb::sigrid_hash_precompute(
 *    Tensor input,
 *    int salt,
 *    int maxValue,
 *    Tensor multiplier_shift,
 *    bool hashIntoInt32
 * ) -> Tensor
 *
 *
 */
Expected<TensorOutput>
ShapeInferenceEngine::sigridHashPrecompute(const MetaStack &variableMetas) {
  RETURN_ERR_IF_NOT(
      variableMetas.size() == 5,
      strFormat("Expected 5 inputs, got %zu", variableMetas.size()));

  TensorShape shape = variableMetas[0].shape<TensorShape>();

  const auto hashToInt32Values = variableMetas[4].intValue;
  RETURN_ERR_IF_NOT(
      hashToInt32Values.size() == 1,
      strFormat("Expected only one value for hashIntoInt32, got %zu",
                hashToInt32Values.size()));

  TensorOutput output;
  output.shapeOrIntValues = shape;
  if (hashToInt32Values[0]) {
    output.dtype = at::ScalarType::Int;
  } else {
    output.dtype = variableMetas[0].dtype;
  }
  return output;
}

/*
 * aten::full_like(
 *    Tensor self,
 *    Scalar fill_value,
 *    *,
 *    ScalarType? dtype=None,
 *    Layout? layout=None,
 *    Device? device=None,
 *    bool? pin_memory=None,
 *    MemoryFormat? memory_format=None
 * ) -> Tensor"))) {
 *
 */
Expected<TensorOutput>
ShapeInferenceEngine::fullLike(const MetaStack &variableMetas) {
  RETURN_ERR_IF_NOT(
      variableMetas.size() == 7,
      strFormat("Expected 7 inputs, got %zu", variableMetas.size()));

  TensorShape shape = variableMetas[0].shape<TensorShape>();

  const auto dtypeValues = variableMetas[2].intValue;
  RETURN_ERR_IF_NOT(
      dtypeValues.size() == 1,
      strFormat("Expected only one dtype value, got %zu", dtypeValues.size()));

  TensorOutput output;
  output.shapeOrIntValues = shape;
  output.dtype = static_cast<at::ScalarType>(dtypeValues[0]);
  return output;
}

} // namespace glow
