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
/// This file describes the API used for graph verification.
/// These are mainly helper class/functions for printing errors and the related
/// context and doing checks.
#ifndef GLOW_GRAPH_VERIFIERHELPER_H
#define GLOW_GRAPH_VERIFIERHELPER_H

#include "glow/Base/Type.h"
#include "glow/Graph/Graph.h"
#include "glow/Graph/Node.h"
#include "glow/Support/Support.h"
#include "llvm/ADT/StringRef.h"

namespace glow {

//===----------------------------------------------------------------------===//
//                       Printing
//===----------------------------------------------------------------------===//

/// Default reportContext function used to print \p a.
/// The default implementation relies on operator<< being available.
/// The actual printing is done calling glow::report.
template <typename Ty> void reportContext(const Ty &a) {
  std::string storage;
  llvm::raw_string_ostream stringStream(storage);
  stringStream << a;
  report(stringStream.str());
}

template <typename Ty> void reportContext(const llvm::ArrayRef<Ty> &arrayRef) {
  bool isFirst = true;
  report("{");
  for (auto elt : arrayRef) {
    if (!isFirst) {
      report(", ");
    }
    isFirst = false;
    reportContext(elt);
  }
  report("}");
}

void reportContext(ElemKind Ty);
void reportContext(const ShapeNHWC &shapeNHWC);
void reportContext(const ShapeNCHW &shapeNCHW);
void reportContext(const Node *node);
void reportContext(const Function *function);

//===----------------------------------------------------------------------===//
//                       Checks
//===----------------------------------------------------------------------===//

/// Wrapper around comparison operators.
/// They are used to specify the behavior of expectCompareTrue
/// and provide a pretty printer of the operator used when
/// things fail.
/// @{

/// Interface that the comparison operator must implement.
template <typename Ty> struct CompareWithName {
  virtual ~CompareWithName() {}
  /// Binary comparison operation.
  virtual bool operator()(const Ty &a, const Ty &b) const = 0;
  /// Name of the operator used for pretty printing.
  virtual llvm::StringRef getCompareName() const = 0;
};

/// Operator ==.
template <typename Ty>
struct CompareOperatorEqual : public CompareWithName<Ty> {
  bool operator()(const Ty &a, const Ty &b) const override { return a == b; }
  llvm::StringRef getCompareName() const override { return "Equal"; }
};

/// Operator >=.
template <typename Ty>
struct CompareOperatorGreaterEqual : public CompareWithName<Ty> {
  bool operator()(const Ty &a, const Ty &b) const override { return a >= b; }
  llvm::StringRef getCompareName() const override { return "GreaterEqual"; }
};

/// Operator >.
template <typename Ty>
struct CompareOperatorGreaterThan : public CompareWithName<Ty> {
  bool operator()(const Ty &a, const Ty &b) const override { return a > b; }
  llvm::StringRef getCompareName() const override { return "GreaterThan"; }
};

/// Operator <=.
template <typename Ty>
struct CompareOperatorLessEqual : public CompareWithName<Ty> {
  bool operator()(const Ty &a, const Ty &b) const override { return a <= b; }
  llvm::StringRef getCompareName() const override { return "LessEqual"; }
};

/// Operator <.
template <typename Ty> struct CompareOperatorLess : public CompareWithName<Ty> {
  bool operator()(const Ty &a, const Ty &b) const override { return a < b; }
  llvm::StringRef getCompareName() const override { return "Less"; }
};
/// @}

/// Main API of the verifier.
/// Check whether \p comp(\p a, \p b) is true.
/// If that check fails, \p msg is printed out using glow::report
/// and \p parent (if not nullptr), \p a, and \p b are printed out
/// using glow::reportContext.
/// \returns \p comp(\p a, \p b).
template <typename InputTy, typename ParentTy>
bool expectCompareTrue(
    const char *msg, const InputTy &a, const InputTy &b, const ParentTy *parent,
    const CompareWithName<InputTy> &comp = CompareOperatorEqual<InputTy>()) {
  if (comp(a, b)) {
    return true;
  }
  if (parent) {
    reportContext(parent);
    report("\n");
  }
  report(msg);
  report("\nFor comparison `LHS ");
  report(comp.getCompareName());
  report(" RHS` with:");
  report("\nLHS: ");
  reportContext(a);
  report("\nRHS: ");
  reportContext(b);
  report("\n");
  return false;
}

/// Check whether $V_{0,n}{comp(\p a, \p b_i)}$ is true.
/// If that check fails, \p msg is printed out using glow::report
/// and \p parent (if not nullptr), \p a, and \p b are printed out
/// using glow::reportContext.
/// \returns \p comp(\p a, \p b_0) v ... v comp(\p a, \p b_i).
template <typename InputTy>
bool expectCompareTrue(
    const char *msg, const InputTy &a, llvm::ArrayRef<InputTy> b,
    const Node *parent,
    const CompareWithName<InputTy> &comp = CompareOperatorEqual<InputTy>()) {
  bool result = false;
  for (const auto &bi : b) {
    result |= comp(a, bi);
  }
  if (result) {
    return true;
  }
  if (parent) {
    reportContext(parent);
  }
  report(msg);
  report("\nFor comparison `LHS ");
  report(comp.getCompareName());
  report(" RHS` with:");
  report("\nLHS: ");
  reportContext(a);
  report("\nRHS: ");
  for (const auto &bi : b) {
    reportContext(bi);
    report(", ");
  }
  report("\n");
  return false;
}

/// Check that the type of the first operand \p A matches the type of the second
/// operand \p B. \p parent is used to print the context of that check
/// in case the it fails.
/// \see expectCompareTrue for more details.
bool checkSameType(NodeValue A, NodeValue B, const Node *parent);

/// Check that the shape of the first operand \p A matches the shape of the
/// second operand \p B. \p parent is used to print the context of that check
/// in case the it fails.
/// \see expectCompareTrue for more details.
bool checkSameShape(NodeValue A, NodeValue B, const Node *parent);

/// Check that the element type of the operand \p A matches expected type \p
/// expectedType. \p parent is used to print the context of that check
/// in case the it fails.
/// \see expectCompareTrue for more details.
bool checkType(NodeValue A, ElemKind expectedType, const Node *parent);

/// Check that the element type of the operand \p A matches any of the expected
/// types \p expectedTypes. \p parent is used to print the context of that
/// check in case the it fails. \see expectCompareTrue for more details.
bool checkType(NodeValue A, llvm::ArrayRef<ElemKind> expectedTypes,
               const Node *parent);

/// Check if \p A and \p B have the same value for isQuantized. \p parent is
/// used to print the context of that check in case the it fails.
/// \see expectCompareTrue for more details.
bool checkSameIsQuantized(const TypeRef A, const TypeRef B, const Node *parent);

/// \return True if \p A is not quantized or has its quantization parameters
/// match \p scale and \p offset. False otherwise. \p parent is used to print
/// the context of that check in case the it fails.
/// \see expectCompareTrue for more details.
bool checkNotQuantizedOrSameParams(const TypeRef A, float scale, int32_t offset,
                                   const Node *parent);

/// \return True if \p A is not quantized or matches \p B quantization
/// parameters. False otherwise.
/// In particular, this returns false if \p A is quantized and \p B
/// is not. The opposite is not true.
/// \p parent is used to print the context of that check
/// in case the it fails.
/// \see expectCompareTrue for more details.
bool checkNotQuantizedOrSameParams(const TypeRef A, const TypeRef B,
                                   const Node *parent);

/// Check that the type of the first operand \p A matches the type of the second
/// operand \p B but ignore the actual shape. Use only element type and
/// quantization parameters in comparison.
/// \p parent is used to print the context of that check
/// in case the it fails.
/// \see expectCompareTrue for more details.
bool checkTypeIgnoreShape(NodeValue A, NodeValue B, const Node *parent);
} // namespace glow
#endif // End of GLOW_GRAPH_VERIFIERHELPER_H.
