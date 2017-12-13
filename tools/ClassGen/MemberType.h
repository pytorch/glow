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
  VectorFloat,
  VectorUnsigned,
  VectorSizeT,
  VectorNodeValue,
  CustomType,
};

/// A helper type for handling members with custom types.
class MemberTypeProvider {
  /// A mapping of members names to their custom types.
  mutable std::unordered_map<std::string, std::string> customTypeMembersTypes_;

public:
  void addCustomTypeMember(const std::string &typeName,
                           const std::string &name) {
    customTypeMembersTypes_.insert({name, typeName});
  }

  inline const char *getReturnTypename(MemberType type,
                                       const std::string &name) const {
    static const char *returnTypes[] = {"TypeRef",
                                        "float",
                                        "unsigned",
                                        "size_t",
                                        "llvm::ArrayRef<float>",
                                        "llvm::ArrayRef<unsigned>",
                                        "llvm::ArrayRef<size_t>",
                                        "llvm::ArrayRef<NodeValue>",
                                        "CustomType",
                                        nullptr};
    if (type != MemberType::CustomType)
      return returnTypes[(int)type];
    auto found = customTypeMembersTypes_.find(name);
    assert(found != customTypeMembersTypes_.end() &&
           "typename should be defined");
    return (*found).second.c_str();
  }

  inline const char *getStorageTypename(MemberType type,
                                        const std::string &name) const {
    static const char *storageTypes[] = {"TypeRef",
                                         "float",
                                         "unsigned",
                                         "size_t",
                                         "std::vector<float>",
                                         "std::vector<unsigned>",
                                         "std::vector<size_t>",
                                         "std::vector<NodeValue>",
                                         "CustomType",
                                         nullptr};
    if (type != MemberType::CustomType)
      return storageTypes[(int)type];
    auto found = customTypeMembersTypes_.find(name);
    assert(found != customTypeMembersTypes_.end() &&
           "typename should be defined");
    return (*found).second.c_str();
  }
};
#endif // GLOW_TOOLS_CLASSGEN_MEMBERTYPE_H
