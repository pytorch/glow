/**
 * Copyright (c) 2017-present, Facebook, Inc.
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
#ifndef GLOW_TOOLS_CLASSGEN_MEMBERTYPE_H
#define GLOW_TOOLS_CLASSGEN_MEMBERTYPE_H

#include <cassert>
#include <string>
#include <unordered_map>

enum class MemberType : unsigned {
  TypeRef,
  Float,
  Unsigned,
  Boolean,
  String,
  VectorFloat,
  VectorSigned,
  VectorUnsigned,
  VectorSizeT,
  VectorNodeValue,
  Enum,
};

/// This struct encapsulates all of the information NodeBuilder needs about the
/// type of a member in order to generate a cloner, hasher, equator, etc.
struct MemberTypeInfo {
  MemberType type;
  std::string returnTypename;
  std::string storageTypename;
  std::string ctorArgTypename;
  std::string forwardDecl;
};

/// These are instances of MemberTypeInfo for commonly used types.
extern MemberTypeInfo kTypeRefTypeInfo;
extern MemberTypeInfo kFloatTypeInfo;
extern MemberTypeInfo kUnsignedTypeInfo;
extern MemberTypeInfo kBooleanTypeInfo;
extern MemberTypeInfo kStringTypeInfo;
extern MemberTypeInfo kVectorFloatTypeInfo;
extern MemberTypeInfo kVectorUnsignedTypeInfo;
extern MemberTypeInfo kVectorSignedTypeInfo;
extern MemberTypeInfo kVectorSizeTTypeInfo;
extern MemberTypeInfo kVectorNodeValueTypeInfo;
extern MemberTypeInfo kEnumTypeInfo;

inline const char *getReturnTypename(const MemberTypeInfo *typeInfo) {
  return typeInfo->returnTypename.c_str();
}

inline const char *getStorageTypename(const MemberTypeInfo *typeInfo) {
  return typeInfo->storageTypename.c_str();
}

inline const char *getCtorArgTypename(const MemberTypeInfo *typeInfo) {
  return typeInfo->ctorArgTypename.c_str();
}

/// TODO: Remove after modifying InstrGen to use MemberTypeInfo as well?
inline const char *getReturnTypename(MemberType type) {
  const char *returnTypes[] = {"TypeRef",
                               "float",
                               "unsigned_t",
                               "bool",
                               "llvm::StringRef",
                               "llvm::ArrayRef<float>",
                               "llvm::ArrayRef<int>",
                               "llvm::ArrayRef<unsigned_t>",
                               "llvm::ArrayRef<size_t>",
                               "NodeValueArrayRef",
                               nullptr};
  return returnTypes[(int)type];
}

inline const char *getStorageTypename(MemberType type) {
  const char *storageTypes[] = {"TypeRef",
                                "float",
                                "unsigned_t",
                                "bool",
                                "std::string",
                                "std::vector<float>",
                                "std::vector<int>",
                                "std::vector<unsigned_t>",
                                "std::vector<size_t>",
                                "std::vector<NodeHandle>",
                                nullptr};
  return storageTypes[(int)type];
}

inline const char *getCtorArgTypename(MemberType type) {
  const char *ctorArgTypes[] = {"TypeRef",
                                "float",
                                "unsigned_t",
                                "bool",
                                "std::string",
                                "std::vector<float>",
                                "std::vector<int>",
                                "std::vector<unsigned_t>",
                                "std::vector<size_t>",
                                "std::vector<NodeValue>",
                                nullptr};
  return ctorArgTypes[(int)type];
}

#endif // GLOW_TOOLS_CLASSGEN_MEMBERTYPE_H
