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

#ifndef GLOW_IR_FXIRUTILS_H
#define GLOW_IR_FXIRUTILS_H

#include "glow/Graph/FXIRWrapper.h"
#include "glow/IR/IR.h"
#include "glow/IR/Instrs.h"

#include <cstdint>
#include <folly/Conv.h>
#include <folly/String.h>
#include <folly/json/dynamic.h>
#include <type_traits>

namespace glow {

/// Get ElemKind from typeStr.
ElemKind getElemKind(const std::string &dtypeStr);

/// Get the kwargs of the node.
const folly::dynamic &getNodeKwargs(const folly::dynamic &node);

/// Helper function to convert \p intArrayStr like "[1, 2, 3]" or "(1, 2, 3)"
/// to vector [1, 2, 3]. If \p length is greater than 0, append the vector with
/// last element to such length. \returns a vector of length size.
template <class T>
std::vector<T> toIntegerArray(std::string intArrayStr,
                              const uint32_t length = 0) {
  // If intArrayStr is a number, make it a single element array.
  if (isdigit(intArrayStr.front())) {
    intArrayStr = folly::to<std::string>("[", intArrayStr, "]");
  }

  // If arrayStr is "[]", make it "[1]".
  if (intArrayStr == "[]") {
    intArrayStr = "[1]";
  }

  // Tokenizing by ", ".
  std::vector<T> vec;
  std::vector<std::string> tokens;
  folly::split(", ", intArrayStr.substr(1, intArrayStr.length() - 2), tokens);
  for (const auto &token : tokens) {
    vec.push_back(folly::to<T>(token));
  }

  CHECK(!vec.empty()) << "Empty dimension size!";
  // Expand vec to length with the last element in vec.
  while (vec.size() < length) {
    vec.push_back(vec.back());
  }

  return vec;
}

template <class T>
std::vector<T> toArray(const folly::dynamic &dyn, const uint32_t &length = 0) {
  static_assert(std::is_floating_point<T>() || std::is_integral<T>(),
                "Currently only support float and int types");
  auto isType = [](const auto &a) {
    return std::is_floating_point<T>() ? a.isDouble() : a.isInt();
  };
  auto getType = [](const auto &a) {
    return std::is_floating_point<T>() ? a.getDouble() : a.getInt();
  };
  std::vector<T> vec;
  if (isType(dyn)) {
    vec.emplace_back(getType(dyn));
  } else if (dyn.isArray()) {
    for (auto &e : dyn) {
      if (isType(e)) {
        vec.emplace_back(getType(e));
      } else {
        LOG(FATAL) << "Mismatch between specified type for toArray and found "
                      "type in the vector in json";
      }
    }
  } else {
    LOG(FATAL) << "Expected single element or vector of specified isArray type";
  }

  CHECK(!vec.empty()) << "Empty dimension size!";
  // Expand vec to length with the last element in vec.
  while (vec.size() < length) {
    vec.push_back(vec.back());
  }

  return vec;
}

/// Get the opCode of the node.
std::string getNodeOpCode(const folly::dynamic &node);

/// Get the name of the node.
std::string getNodeName(const folly::dynamic &node);

/// Get the target of the node.
std::string getNodeTarget(const folly::dynamic &node);

/// Get the data type of the node. \p idx represents which output to get a
/// result for in the case that the node has multiple outputs; if it's a single
/// output then should be left to -1.
ElemKind getNodeDataType(const folly::dynamic &node, int idx = -1);

bool hasFxOutTensorView(const folly::dynamic &node);

int countFxOutTensorView(const folly::dynamic &node);

/// Get out tensorview for \p node. If \p idx is non-negative then assume this
/// is a multi-output node, so get the tensorview output for that specific idx.
/// \p idx represents which output to get a result for in the case that the node
/// has multiple outputs; if it's a single output then should be left to -1.
const folly::dynamic &getFxOutTensorView(const folly::dynamic &node,
                                         int idx = -1);

/// \returns specific item \p itemName from \p node. \p idx represents which
/// output to get a result for in the case that the node has multiple outputs;
/// if it's a single output then should be left to -1.
std::string getNodeItemAsString(const folly::dynamic &node,
                                const char *itemName, int idx = -1);
std::string getNodeShapeAsString(const folly::dynamic &node, int idx = -1);
std::string getNodeStrideAsString(const folly::dynamic &node, int idx = -1);

template <class T>
std::vector<T> getNodeItem(const folly::dynamic &node, const char *itemName,
                           int idx = -1) {
  const std::string itemString = getNodeItemAsString(node, itemName, idx);
  return toIntegerArray<glow::dim_t>(itemString);
}

/// \returns the shape from \p node. \p idx represents which output to get a
/// result for in the case that the node has multiple outputs; if it's a single
/// output then should be left to -1.
template <class T>
std::vector<T> getNodeShape(const folly::dynamic &node, int idx = -1) {
  const std::string shapeString = getNodeShapeAsString(node, idx);
  return toIntegerArray<glow::dim_t>(shapeString);
}

/// \returns the stride from \p node. \p idx represents which output to get a
/// result for in the case that the node has multiple outputs; if it's a single
/// output then should be left to -1.
template <class T>
std::vector<T> getNodeStride(const folly::dynamic &node, int idx = -1) {
  const std::string strideString = getNodeStrideAsString(node, idx);
  return toIntegerArray<glow::dim_t>(strideString);
}

std::string getNodeOffsetsAsString(const folly::dynamic &node);

template <class T> std::vector<T> getNodeOffsets(const folly::dynamic &node) {
  const std::string offsetString = getNodeOffsetsAsString(node);
  return toIntegerArray<glow::dim_t>(offsetString);
}

/// Checks if node's padded.
bool isNodePadded(const folly::dynamic &node);

/// Get the arg of the node.
const folly::dynamic &getNodeArgs(const folly::dynamic &node);

template <class T> std::vector<T> getConvStride(const folly::dynamic &node) {
  const auto &inputs = getNodeKwargs(node);
  CHECK(inputs.find("stride") != inputs.items().end())
      << "stride field doesn't exist in Conv Inputs " << node;
  return toArray<uint32_t>(inputs["stride"], 2);
}

template <class T> std::vector<T> getConvPads(const folly::dynamic &node) {
  const auto &inputs = getNodeKwargs(node);
  CHECK(inputs.find("padding") != inputs.items().end())
      << "padding field doesn't exist in Conv Inputs " << node;
  return toArray<uint32_t>(inputs["padding"], 2);
}

template <class T> std::vector<T> getConvKernels(const folly::dynamic &node) {
  const auto &inputs = getNodeKwargs(node);
  CHECK(inputs.find("kernel_size") != inputs.items().end())
      << "kernel_size field doesn't exist in Conv Inputs " << node;
  return toArray<uint32_t>(inputs["kernel_size"], 2);
}

template <class T>
std::vector<T> getConvKernelsFromWeightNode(const folly::dynamic &node) {
  const auto weightShape = getNodeShape<glow::dim_t>(node);
  CHECK_GE(weightShape.size(), 2) << "Expected weight at least 2D";
  return std::vector<T>(weightShape[weightShape.size() - 2],
                        weightShape[weightShape.size() - 1]);
}

template <class T>
std::vector<T> getTransposeShuffle(const folly::dynamic &node) {
  const auto &inputs = getNodeKwargs(node);
  CHECK(inputs.find("permutation") != inputs.items().end())
      << "field transposed_dims doesn't exist in Conv Inputs " << node;
  return toArray<uint32_t>(inputs["permutation"], 2);
}

template <class T> std::vector<T> getMeanDims(const folly::dynamic &node) {
  auto &inputs = getNodeKwargs(node);
  CHECK(inputs.find("dim") != inputs.items().end())
      << "field dims doesn't exist in Mean Inputs " << node;
  return toArray<uint32_t>(inputs["dim"]);
}

/// Search \p storageNodeNameToDest and \p nonStorageNodeNameToDest for
/// nodeName.
Value *valueForNode(
    const std::string &nodeName,
    const std::unordered_map<std::string, Value *> &storageNodeNameToDest,
    const std::unordered_map<std::string, Value *> &nonStorageNodeNameToDest);

/// Get the scale for quantized node.
double getNodeScale(const folly::dynamic &node);

/// Get the zero point for quantized node.
int getNodeZeroPoint(const folly::dynamic &node);

/// Get vector of offsets for Extract operations.
std::vector<dim_t> getOffsets(const folly::dynamic &node);

bool isInputFXNode(const folly::dynamic &node);
bool isOutputFXNode(const folly::dynamic &node);
bool isConstantWeightFXNode(const folly::dynamic &node);
bool isActivationFXNode(const folly::dynamic &node);

} // namespace glow

#endif // GLOW_IR_FXIRUTILS_H
