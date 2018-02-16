// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "glow/Graph/Graph.h"
#include "glow/Graph/Nodes.h"
#include "glow/IR/IR.h"
#include "glow/Support/Support.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

#include <fstream>
#include <iostream>
#include <unordered_set>

using namespace glow;
using llvm::dyn_cast;
using llvm::isa;

Graph::~Graph() {
  // Delete all of the variables.
  for (auto it = vars_.begin(), e = vars_.end(); it != e;) {
    auto cur = it++;
    eraseVariable(*cur);
  }
}

TypeRef Graph::uniqueType(ElemKind elemTy, llvm::ArrayRef<size_t> dims) {
  return uniqueType(Type(elemTy, dims));
}

TypeRef Graph::uniqueType(ElemKind elemTy, llvm::ArrayRef<size_t> dims,
                          float scale, int32_t offset) {
  return uniqueType(Type(elemTy, dims, scale, offset));
}

TypeRef Graph::uniqueTypeWithNewShape(TypeRef T, llvm::ArrayRef<size_t> dims) {
  if (T->isQuantizedType()) {
    return uniqueType(
        Type(T->getElementType(), dims, T->getScale(), T->getOffset()));

  } else {
    return uniqueType(Type(T->getElementType(), dims));
  }
}

TypeRef Graph::uniqueType(const Type &T) {
  for (auto &tp : types_) {
    if (T.isEqual(tp)) {
      return &tp;
    }
  }

  return &*types_.insert(types_.begin(), T);
}

TypeRef Graph::getVoidTy() { return uniqueType(Type()); }

//===----------------------------------------------------------------------===//
//                       Node builders
//===----------------------------------------------------------------------===//

Variable *Graph::createVariable(TypeRef T, llvm::StringRef name,
                                Variable::VisibilityKind visibility,
                                Variable::TrainKind train, float val) {
  auto FT = uniqueType(*T);
  return addVar(new Variable(name, FT, visibility, train, val));
}

Variable *Graph::createVariable(ElemKind T, llvm::ArrayRef<size_t> dims,
                                llvm::StringRef name,
                                Variable::VisibilityKind visibility,
                                Variable::TrainKind train, float val) {
  auto FT = uniqueType(T, dims);
  return createVariable(FT, name, visibility, train, val);
}

Variable *Graph::createVariable(ElemKind T, llvm::ArrayRef<size_t> dims,
                                float scale, int32_t offset,
                                llvm::StringRef name,
                                Variable::VisibilityKind visibility,
                                Variable::TrainKind train, float val) {
  auto FT = uniqueType(T, dims, scale, offset);
  return createVariable(FT, name, visibility, train, val);
}

/// Form a unique name based on the original non-uniqued \p Name.
///
/// This is done by taking the original non-uniqued name
/// (i.e. the part of the name before the first occurrence of "__")
/// and concatenating it with "__N", where N is a unique numeric
/// suffix.
///
/// The "__" suffix is used as a delimeter and therefore it should
/// not be used by names of user-defined variables.
///
/// If the compiler needs to auto-generate some node names, it should
/// never add any suffix anywhere after "__", because it will get
/// stripped by uniqueName. Instead, all such auto-generated pieces of
/// a name should be added somewhere before "__", e.g. as a prefix.
std::string Graph::uniqueName(llvm::StringRef name) {
  // First, remove everything starting with the __ delimiter.
  auto delimPos = name.find("__", 0);
  if (delimPos != llvm::StringRef::npos) {
    name = name.substr(0, delimPos);
  }
  std::string UniqueName{name};
  UniqueName += "__";
  UniqueName += std::to_string(uniqueIdx_);
  uniqueIdx_++;
  return UniqueName;
}

void Graph::uniquifyName(Node *N) { N->setName(uniqueName(N->getName())); }

void Graph::eraseVariable(VariablesList::iterator I) {
  if (I == vars_.end())
    return;
  delete *I;
  vars_.erase(I);
}

Variable *Graph::getVariableByName(llvm::StringRef name) {
  for (auto *V : getVars()) {
    if (V->getName() == name)
      return V;
  }
  return nullptr;
}

void Graph::eraseVariable(Variable *N) {
  auto I = std::find(vars_.begin(), vars_.end(), N);
  eraseVariable(I);
}
