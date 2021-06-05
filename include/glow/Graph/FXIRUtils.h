// (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

#ifndef GLOW_IR_FXIRUTILS_H
#define GLOW_IR_FXIRUTILS_H

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

/// Get the opCode of the node.
std::string getNodeOpCode(const folly::dynamic &node);

/// Get the name of the node.
std::string getNodeName(const folly::dynamic &node);

/// Get the target of the node.
std::string getNodeTarget(const folly::dynamic &node);

/// Get the data type of the node.
ElemKind getNodeDataType(const folly::dynamic &node);

/// Get the shape of the node.
std::vector<glow::dim_t> getNodeShape(const folly::dynamic &node);

/// Get the arg of the node.
const folly::dynamic &getNodeArgs(const folly::dynamic &node);

/// Get the kwargs of the node.
const folly::dynamic &getNodeKwargs(const folly::dynamic &node);

/// Search \p storageNodeNameToDest and \p nonStorageNodeNameToDest for
/// nodeName.
Value *valueForNode(
    const std::string &nodeName,
    const std::unordered_map<std::string, Value *> &storageNodeNameToDest,
    const std::unordered_map<std::string, Value *> &nonStorageNodeNameToDest);

} // namespace glow

#endif // GLOW_IR_FXIRUTILS_H
