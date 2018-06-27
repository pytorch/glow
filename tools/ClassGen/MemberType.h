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
  SizeT,
  String,
  VectorFloat,
  VectorUnsigned,
  VectorSizeT,
  VectorNodeValue,
};

inline const char *getReturnTypename(MemberType type) {
  const char *returnTypes[] = {"TypeRef",
                               "float",
                               "unsigned",
                               "size_t",
                               "llvm::StringRef",
                               "llvm::ArrayRef<float>",
                               "llvm::ArrayRef<unsigned>",
                               "llvm::ArrayRef<size_t>",
                               "NodeValueArrayRef",
                               nullptr};
  return returnTypes[(int)type];
}

inline const char *getStorageTypename(MemberType type) {
  const char *storageTypes[] = {
      "TypeRef",
      "float",
      "unsigned",
      "size_t",
      "std::string",
      "std::vector<float>",
      "std::vector<unsigned>",
      "std::vector<size_t>",
      "std::vector<PrivateNodeTypes::NodeValueHandle>",
      nullptr};
  return storageTypes[(int)type];
}

inline const char *getCtorArgTypename(MemberType type) {
  const char *ctorArgTypes[] = {"TypeRef",
                                "float",
                                "unsigned",
                                "size_t",
                                "std::string",
                                "std::vector<float>",
                                "std::vector<unsigned>",
                                "std::vector<size_t>",
                                "std::vector<NodeValue>",
                                nullptr};
  return ctorArgTypes[(int)type];
}

#endif // GLOW_TOOLS_CLASSGEN_MEMBERTYPE_H
