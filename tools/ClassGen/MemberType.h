#ifndef GLOW_TOOLS_CLASSGEN_MEMBERTYPE_H
#define GLOW_TOOLS_CLASSGEN_MEMBERTYPE_H

#include <cassert>
#include <string>
#include <unordered_map>

enum MemberType {
  TypeRef,
  Float,
  Unsigned,
  SizeT,
  VectorFloat,
  VectorUnsigned,
  VectorSizeT,
};

struct TypeStr {
  std::string storageType;
  std::string returnType;
};

static const std::unordered_map<MemberType, TypeStr> kMemberTypeStrMap = {
    {MemberType::TypeRef, {"TypeRef", "TypeRef"}},
    {MemberType::Float, {"float", "float"}},
    {MemberType::Unsigned, {"unsigned", "unsigned"}},
    {MemberType::SizeT, {"size_t", "size_t"}},
    {MemberType::VectorFloat, {"std::vector<float>", "llvm::ArrayRef<float>"}},
    {MemberType::VectorUnsigned,
     {"std::vector<unsigned>", "llvm::ArrayRef<unsigned>"}},
    {MemberType::VectorSizeT,
     {"std::vector<size_t>", "llvm::ArrayRef<size_t>"}},
};

inline std::string getStorageTypename(MemberType type) {
  return kMemberTypeStrMap.at(type).storageType;
}

inline std::string getReturnTypename(MemberType type) {
  return kMemberTypeStrMap.at(type).returnType;
}

#endif // GLOW_TOOLS_CLASSGEN_MEMBERTYPE_H
