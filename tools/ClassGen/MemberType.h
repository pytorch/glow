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
  VoidStar,
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
                               "llvm::ArrayRef<NodeValue>",
                               "void*",
                               nullptr};
  return returnTypes[(int)type];
}

inline const char *getStorageTypename(MemberType type) {
  const char *storageTypes[] = {"TypeRef",
                                "float",
                                "unsigned",
                                "size_t",
                                "std::string",
                                "std::vector<float>",
                                "std::vector<unsigned>",
                                "std::vector<size_t>",
                                "std::vector<NodeValue>",
                                "void*",
                                nullptr};
  return storageTypes[(int)type];
}

#endif // GLOW_TOOLS_CLASSGEN_MEMBERTYPE_H
