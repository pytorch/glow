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

#include "glow/Graph/FXIRUtils.h"

#include "glow/Graph/FXIRWrapper.h"
#include "llvm/Support/Casting.h"

#include <folly/DynamicConverter.h>

using namespace glow;

namespace {
const std::unordered_map<std::string, ElemKind> stringToElemKind = {
    // 64-bit float type
    {"torch.float64", ElemKind::Float64Ty},
    // 32-bit float type
    {"torch.float32", ElemKind::FloatTy},
    {"torch.float", ElemKind::FloatTy},
    // 16-bit float type
    {"torch.float16", ElemKind::Float16Ty},
    {"torch.half", ElemKind::Float16Ty},
    {"torch.bfloat16", ElemKind::BFloat16Ty},
    // 64-bit int type
    {"torch.int64", ElemKind::Int64ITy},
    // 64-bit int type
    {"torch.long", ElemKind::Int64ITy},
    // Unsigned 8 bit Int type
    {"torch.uint8", ElemKind::UInt8ITy},
    // 8 bit Quantized int type
    {"torch.qint8", ElemKind::Int8QTy},
    // Unsigned 8 bit Quantized int type
    {"torch.quint8", ElemKind::UInt8QTy},
    // 32-bit int type
    {"torch.int32", ElemKind::Int32ITy},
    {"torch.qint32", ElemKind::Int32QTy},
    // 8-bit fused quantize type
    {"acc.uint8fused", ElemKind::UInt8FusedQTy},
    // 4-bit fused quantize type
    {"acc.uint4fused", ElemKind::UInt4FusedQTy},
    // bool type
    {"torch.bool", ElemKind::BoolTy},
};
}

ElemKind glow::getElemKind(const std::string &dtypeStr) {
  const auto &dtypeElt = stringToElemKind.find(dtypeStr);
  CHECK(dtypeElt != stringToElemKind.end()) << dtypeStr << " is not supported!";
  return dtypeElt->second;
}

std::string glow::getNodeOpCode(const folly::dynamic &node) {
  CHECK(node.find("op_code") != node.items().end())
      << "op_code field doesn't exist in node " << node;
  return node["op_code"].getString();
}

std::string glow::getNodeName(const folly::dynamic &node) {
  CHECK(node.find("name") != node.items().end())
      << "name field doesn't exist in node " << node;
  return node["name"].getString();
}

std::string glow::getNodeTarget(const folly::dynamic &node) {
  CHECK(node.find("target") != node.items().end())
      << "target field doesn't exist in node " << node;
  return node["target"].getString();
}

ElemKind glow::getNodeDataType(const folly::dynamic &node, int idx) {
  CHECK(node.find("dtype") != node.items().end())
      << "dtype field doesn't exist in node " << node;
  auto s = idx < 0 ? node.at("dtype").getString()
                   : node.at("dtype").at(idx).getString();
  return getElemKind(s);
}

double glow::getNodeScale(const folly::dynamic &node) {
  CHECK(node.find("q_scale") != node.items().end())
      << "q_scale field doesn't exist in node " << node;
  return node.at("q_scale").getDouble();
}

int glow::getNodeZeroPoint(const folly::dynamic &node) {
  CHECK(node.find("q_zero_point") != node.items().end())
      << "q_zero_point field doesn't exist in node " << node;
  return node.at("q_zero_point").getInt();
}

const folly::dynamic &glow::getNodeArgs(const folly::dynamic &node) {
  CHECK(node.find("args") != node.items().end())
      << "args field doesn't exist in node " << node;
  return node["args"];
}

const folly::dynamic &glow::getNodeKwargs(const folly::dynamic &node) {
  CHECK(node.find("kwargs") != node.items().end())
      << "args field doesn't exist in node " << node;
  return node["kwargs"];
}

bool glow::isNodePadded(const folly::dynamic &node) {
  auto shape = getNodeShape<glow::dim_t>(node);
  auto stride = getNodeStride<glow::dim_t>(node);
  if (stride.size() >= 2) {
    CHECK_EQ(shape.size(), stride.size());
    return stride[stride.size() - 2] > shape[shape.size() - 1];
  }
  return false;
}

Value *glow::valueForNode(
    const std::string &nodeName,
    const std::unordered_map<std::string, Value *> &storageNodeNameToDest,
    const std::unordered_map<std::string, Value *> &nonStorageNodeNameToDest) {
  Value *value = nullptr;
  if (storageNodeNameToDest.find(nodeName) != storageNodeNameToDest.end()) {
    value = storageNodeNameToDest.at(nodeName);
  } else if (nonStorageNodeNameToDest.find(nodeName) !=
             nonStorageNodeNameToDest.end()) {
    value = nonStorageNodeNameToDest.at(nodeName);
  }
  CHECK(value != nullptr) << "IR was not generated for the node with name: "
                          << nodeName;
  return value;
}

bool glow::hasFxOutTensorView(const folly::dynamic &node) {
  const auto &kwargs = getNodeKwargs(node);
  return kwargs.find("out_memref") != kwargs.items().end();
}

const folly::dynamic &glow::getFxOutTensorView(const folly::dynamic &node,
                                               int idx) {
  const auto &kwargs = getNodeKwargs(node);
  CHECK(hasFxOutTensorView(node)) << "Node must have 'out_memref'\n";
  const auto &out = kwargs["out_memref"];
  if (idx < 0) {
    CHECK(out.isObject())
        << "Expected Node object given unspecified multi-output idx";
    return out;
  }
  CHECK(out.isArray() && idx < out.size());
  CHECK(out.at(idx).isObject());
  return out.at(idx);
}

std::vector<dim_t> glow::getOffsets(const folly::dynamic &node) {
  const auto &kwargs = getNodeKwargs(node);
  const std::string shape = glow::getNodeShapeAsString(node);
  auto count = std::count(shape.begin(), shape.end(), ',') + 1;
  std::vector<dim_t> offsets(count, 0);
  auto dim = kwargs["dim"].asInt();
  auto start = kwargs["start"].asInt();
  offsets[dim] = start;
  return offsets;
}

//======================================================================
std::string glow::getNodeShapeAsString(const folly::dynamic &node, int idx) {
  return glow::getNodeItemAsString(node, "shape", idx);
}

//======================================================================
std::string glow::getNodeStrideAsString(const folly::dynamic &node, int idx) {
  return getNodeItemAsString(node, "stride", idx);
}

//======================================================================
// Offset is introduced with tensor views, so node must be a tensor view
// node, not a compute node.
//======================================================================
std::string glow::getNodeOffsetsAsString(const folly::dynamic &node) {
  CHECK(node.find("is_tensor_view") != node.items().end() &&
        node["is_tensor_view"].asBool())
      << "Node must be tensor view\n";
  CHECK(node.find("offset") != node.items().end())
      << "Missing field 'offset' in tensor view\n";
  return node.at("offset").getString();
}

//======================================================================
// Prior to introduction of tensor views (out_memref is a tensor view
// for writing the output to an alloc), shape and stride were taken
// from the output/destination alloc node (that a compute node writes to).
// The two if statements handle the case with tensor views and the final
// returns the shape from the destination node.
//======================================================================
std::string glow::getNodeItemAsString(const folly::dynamic &node,
                                      const char *itemName, int idx) {
  if (node.find("kwargs") != node.items().end()) {
    const auto &kwargs = getNodeKwargs(node);
    if (kwargs.find("out_memref") != kwargs.items().end()) {
      const auto &out_memref = kwargs["out_memref"]; // out tensor view
      if (idx > -1) {
        return out_memref.at(idx).at(itemName).getString();
      }
      return out_memref.at(itemName).getString();
    }
  }
  CHECK(node.find(itemName) != node.items().end())
      << "Neither " << itemName << " nor out_memref exists in node " << node
      << "\n";
  if (idx > -1) {
    return node.at(itemName).at(idx).getString();
  }
  return node.at(itemName).getString();
}

bool glow::isInputFXNode(const folly::dynamic &node) {
  return (node["op_code"].getString() == "placeholder" &&
          (node.count("ph_type") == 0 ||
           node["ph_type"].getString() == "input_ph"));
}

bool glow::isOutputFXNode(const folly::dynamic &node) {
  return (node["op_code"].getString() == "output" ||
          (node["op_code"].getString() == "placeholder" &&
           (node.count("ph_type") != 0 &&
            node["ph_type"].getString() == "output_ph")));
}

bool glow::isConstantWeightFXNode(const folly::dynamic &node) {
  return node["op_code"].getString() == "get_attr" ||
         (node["op_code"].getString() == "call_function" &&
          node["target"].getString() == "acc_ops.xl_weight");
}

bool glow::isActivationFXNode(const folly::dynamic &node) {
  return (node["op_code"].getString() == "call_function" &&
          node["target"].getString() == "fba_ops.fba_alloc_activation");
}
