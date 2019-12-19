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
#ifndef GLOW_TOOLS_CLASSGEN_MEMBERTYPE_H
#define GLOW_TOOLS_CLASSGEN_MEMBERTYPE_H

#include <cassert>
#include <string>
#include <unordered_map>

/// Define a MemberTypeInfo for the type T. The definition of type T does not
/// need to be known when compiling the NodeGen.cpp, but it should be known when
/// compiling the generated files. T can be a simple type like CustomType or
/// have a more complex form like e.g. CustomType * or CustomType &.
#define MEMBER_TYPE_INFO(T)                                                    \
  MemberTypeInfo { MemberType::UserDefinedType, #T, #T, #T, "" }

/// Define a MemberTypeInfo for the type T that uses a custom comparator instead
/// of ==. The definition of type T does not need to be known when compiling the
/// NodeGen.cpp, but it should be known when compiling the generated files. T
/// can be a simple type like CustomType or have a more complex form like e.g.
/// CustomType * or CustomType &.
#define MEMBER_TYPE_INFO_WITH_CMP(T, CMP_FN)                                   \
  MemberTypeInfo { MemberType::UserDefinedType, #T, #T, #T, "", CMP_FN }

enum class MemberType : unsigned {
  TypeRef,
  Float,
  Unsigned,
  Boolean,
  Int64,
  String,
  VectorFloat,
  VectorSigned,
  VectorUnsigned,
  VectorInt64,
  VectorSizeT,
  VectorDimT,
  VectorNodeValue,
  Enum,
  UserDefinedType,
};

/// This struct encapsulates all of the information NodeBuilder needs about the
/// type of a member in order to generate a cloner, hasher, equator, etc.
struct MemberTypeInfo {
  /// Kind of the member type.
  MemberType type;
  /// Type to be used when this member type is returned.
  std::string returnTypename;
  /// Type to be used for storing members of this type.
  std::string storageTypename;
  /// Type to be used when providing this member type as a constructor arguments
  /// of a node.
  std::string ctorArgTypename;
  /// Forward declaration of this member type.
  std::string forwardDecl;
  /// Comparator function to be used for comparing members of this type.
  std::string cmpFn = "==";
  /// Whether to include a setter for this member type.
  bool addSetter{false};
};

/// These are instances of MemberTypeInfo for commonly used types.
extern MemberTypeInfo kTypeRefTypeInfo;
extern MemberTypeInfo kFloatTypeInfo;
extern MemberTypeInfo kUnsignedTypeInfo;
extern MemberTypeInfo kBooleanTypeInfo;
extern MemberTypeInfo kInt64TypeInfo;
extern MemberTypeInfo kStringTypeInfo;
extern MemberTypeInfo kVectorFloatTypeInfo;
extern MemberTypeInfo kVectorUnsignedTypeInfo;
extern MemberTypeInfo kVectorInt64TypeInfo;
extern MemberTypeInfo kVectorSignedTypeInfo;
extern MemberTypeInfo kVectorSizeTTypeInfo;
extern MemberTypeInfo kVectorDimTTypeInfo;
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
                               "int64_t",
                               "llvm::StringRef",
                               "llvm::ArrayRef<float>",
                               "llvm::ArrayRef<int>",
                               "llvm::ArrayRef<unsigned_t>",
                               "llvm::ArrayRef<int64_t>",
                               "llvm::ArrayRef<size_t>",
                               "llvm::ArrayRef<dim_t>",
                               "NodeValueArrayRef",
                               nullptr,
                               nullptr};
  return returnTypes[(int)type];
}

inline const char *getStorageTypename(MemberType type) {
  const char *storageTypes[] = {"TypeRef",
                                "float",
                                "unsigned_t",
                                "bool",
                                "int64_t",
                                "std::string",
                                "std::vector<float>",
                                "std::vector<int>",
                                "std::vector<unsigned_t>",
                                "std::vector<int64_t>",
                                "std::vector<size_t>",
                                "std::vector<dim_t>",
                                "std::vector<NodeHandle>",
                                nullptr,
                                nullptr};
  return storageTypes[(int)type];
}

inline const char *getCtorArgTypename(MemberType type) {
  const char *ctorArgTypes[] = {"TypeRef",
                                "float",
                                "unsigned_t",
                                "bool",
                                "int64_t",
                                "std::string",
                                "std::vector<float>",
                                "std::vector<int>",
                                "std::vector<unsigned_t>",
                                "std::vector<int64_t>",
                                "std::vector<size_t>",
                                "std::vector<dim_t>",
                                "std::vector<NodeValue>",
                                nullptr,
                                nullptr};
  return ctorArgTypes[(int)type];
}

#endif // GLOW_TOOLS_CLASSGEN_MEMBERTYPE_H
