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
    // 32-bit float type
    {"torch.float32", ElemKind::FloatTy},
    {"torch.float", ElemKind::FloatTy},
    // 16-bit float type
    {"torch.float16", ElemKind::Float16Ty},
    {"torch.half", ElemKind::Float16Ty},
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

ElemKind glow::getNodeDataType(const folly::dynamic &node) {
  CHECK(node.find("dtype") != node.items().end())
      << "dtype field doesn't exist in node " << node;
  return getElemKind(node.at("dtype").getString());
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

std::vector<dim_t> glow::getOffsets(const folly::dynamic &node) {
  const auto &inputs = getNodeKwargs(node);
  auto shape = node["shape"].asString();
  auto count = std::count(shape.begin(), shape.end(), ',') + 1;
  std::vector<dim_t> offsets(count, 0);
  auto dims = folly::convertTo<std::vector<dim_t>>(inputs["dims"]);
  auto starts = folly::convertTo<std::vector<dim_t>>(inputs["starts"]);
  for (int i = 0; i < dims.size(); i++) {
    offsets[dims[i]] = starts[i];
  }
  return offsets;
}
