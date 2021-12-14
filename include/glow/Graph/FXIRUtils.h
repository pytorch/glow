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

#include <folly/Conv.h>
#include <folly/String.h>
#include <folly/dynamic.h>

namespace glow {

/// Get ElemKind from typeStr.
ElemKind getElemKind(const std::string &dtypeStr);

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
std::vector<T> toIntegerArray(const folly::dynamic &dyn,
                              const uint32_t &length = 0) {
  std::vector<T> vec;
  if (dyn.isInt()) {
    vec.emplace_back(dyn.getInt());
  } else if (dyn.isArray()) {
    for (auto &e : dyn) {
      if (e.isInt()) {
        vec.emplace_back(e.getInt());
      } else {
        LOG(FATAL) << "Non-integer vector unhandled";
      }
    }
  } else {
    LOG(FATAL) << "Only supporting integer/vec<integer>";
  }

  CHECK(!vec.empty()) << "Empty dimension size!";
  // Expand vec to length with the last element in vec.
  while (vec.size() < length) {
    vec.push_back(vec.back());
  }

  return vec;
}

template <class T> std::vector<T> getNodeStride(const folly::dynamic &node) {
  CHECK(node.find("stride") != node.items().end())
      << "stride field doesn't exist in node " << node;
  return toIntegerArray<glow::dim_t>(node.at("stride").getString());
}

/// Get the opCode of the node.
std::string getNodeOpCode(const folly::dynamic &node);

/// Get the name of the node.
std::string getNodeName(const folly::dynamic &node);

/// Get the target of the node.
std::string getNodeTarget(const folly::dynamic &node);

/// Get the data type of the node.
ElemKind getNodeDataType(const folly::dynamic &node);

template <class T> std::vector<T> getNodeShape(const folly::dynamic &node) {
  CHECK(node.find("shape") != node.items().end())
      << "shape field doesn't exist in node " << node;
  return toIntegerArray<glow::dim_t>(node.at("shape").getString());
}

/// Checks if node's padded.
bool isNodePadded(const folly::dynamic &node);

/// Get the arg of the node.
const folly::dynamic &getNodeArgs(const folly::dynamic &node);

/// Get the kwargs of the node.
const folly::dynamic &getNodeKwargs(const folly::dynamic &node);

template <class T> std::vector<T> getConvStride(const folly::dynamic &node) {
  const auto &inputs = getNodeKwargs(node);
  CHECK(inputs.find("stride") != inputs.items().end())
      << "stride field doesn't exist in Conv Inputs " << node;
  return toIntegerArray<uint32_t>(inputs["stride"], 2);
}

template <class T> std::vector<T> getConvPads(const folly::dynamic &node) {
  const auto &inputs = getNodeKwargs(node);
  CHECK(inputs.find("padding") != inputs.items().end())
      << "padding field doesn't exist in Conv Inputs " << node;
  return toIntegerArray<uint32_t>(inputs["padding"], 2);
}

template <class T> std::vector<T> getConvKernels(const folly::dynamic &node) {
  const auto &inputs = getNodeKwargs(node);
  CHECK(inputs.find("kernel_size") != inputs.items().end())
      << "kernel_size field doesn't exist in Conv Inputs " << node;
  return toIntegerArray<uint32_t>(inputs["kernel_size"], 2);
}

template <class T>
std::vector<T> getTransposeShuffle(const folly::dynamic &node) {
  const auto &inputs = getNodeKwargs(node);
  CHECK(inputs.find("permutation") != inputs.items().end())
      << "field transposed_dims doesn't exist in Conv Inputs " << node;
  return toIntegerArray<uint32_t>(inputs["permutation"], 2);
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

} // namespace glow

#endif // GLOW_IR_FXIRUTILS_H
