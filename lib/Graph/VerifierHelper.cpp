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

#include "glow/Graph/VerifierHelper.h"

using namespace glow;

//===----------------------------------------------------------------------===//
//                       Printing
//===----------------------------------------------------------------------===//

void glow::reportContext(ElemKind Ty) { report(Type::getElementName(Ty)); }

void glow::reportContext(const ShapeNHWC &shapeNHWC) {
  report("NHWC: ");
  reportContext(llvm::ArrayRef<size_t>(
      {shapeNHWC.n, shapeNHWC.h, shapeNHWC.w, shapeNHWC.c}));
}

void glow::reportContext(const ShapeNCHW &shapeNCHW) {
  report("NCHW: ");
  reportContext(llvm::ArrayRef<size_t>(
      {shapeNCHW.n, shapeNCHW.c, shapeNCHW.h, shapeNCHW.w}));
}

void glow::reportContext(const Node *node) {
  report("In '");
  report(node->getName());
  report("'");
  if (const Function *function = node->getParent()) {
    report(" ");
    reportContext(function);
  }
}

void glow::reportContext(const Function *function) {
  report("From '");
  report(function->getName());
  report("'");
}

//===----------------------------------------------------------------------===//
//                       Checks
//===----------------------------------------------------------------------===//

bool glow::checkSameType(NodeValue A, NodeValue B, const Node *parent) {
  return expectCompareTrue("Mismatching type", *A.getType(), *B.getType(),
                           parent);
}

bool glow::checkSameShape(NodeValue A, NodeValue B, const Node *parent) {
  return expectCompareTrue("Mismatching dimensions", A.dims(), B.dims(),
                           parent);
}

bool glow::checkType(NodeValue A, ElemKind expectedType, const Node *parent) {
  return expectCompareTrue("Mismatching element type", A.getElementType(),
                           expectedType, parent);
}

bool glow::checkType(NodeValue A, llvm::ArrayRef<ElemKind> expectedTypes,
                     const Node *parent) {
  return expectCompareTrue("Mismatching element type", A.getElementType(),
                           expectedTypes, parent);
}

bool glow::checkSameIsQuantized(const TypeRef A, const TypeRef B,
                                const Node *parent) {
  return expectCompareTrue("Mismatching isQuantized", A->isQuantizedType(),
                           B->isQuantizedType(), parent);
}

bool glow::checkNotQuantizedOrSameParams(const TypeRef A, float scale,
                                         int32_t offset, const Node *parent) {
  if (A->isQuantizedType()) {
    if (!expectCompareTrue("Mismatching scale", A->getScale(), scale, parent) ||
        !expectCompareTrue("Mismatching offset", A->getOffset(), offset,
                           parent)) {
      return false;
    }
  }
  return true;
}

bool glow::checkNotQuantizedOrSameParams(const TypeRef A, const TypeRef B,
                                         const Node *parent) {
  if (!B->isQuantizedType()) {
    return checkSameIsQuantized(A, B, parent);
  }
  return checkNotQuantizedOrSameParams(A, B->getScale(), B->getOffset(),
                                       parent);
}

bool glow::checkTypeIgnoreShape(NodeValue A, NodeValue B, const Node *parent) {
  bool isValid = checkType(A, B.getElementType(), parent);
  isValid &= checkSameIsQuantized(A.getType(), B.getType(), parent);
  isValid &= checkNotQuantizedOrSameParams(A.getType(), B.getType(), parent);
  return isValid;
}
