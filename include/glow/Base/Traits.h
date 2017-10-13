#ifndef GLOW_BASE_TRAITS_H
#define GLOW_BASE_TRAITS_H

#include "glow/Base/Type.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

namespace glow {

/// This add the capability to name subclasses.
class Named {
  std::string name_{};

public:
  Named() = default;

  /// \returns the name of the instruction.
  llvm::StringRef getName() const { return name_; }

  /// \returns the name of the instruction.
  bool hasName() const { return !name_.empty(); }

  /// Set the name of the instruction to \p name.
  void setName(llvm::StringRef name) { name_ = name; }
};

/// Subclasses of this class have a type associated with them.
class Typed {
private:
  TypeRef Ty_{};

public:
  explicit Typed(TypeRef Ty) : Ty_(Ty){};

  TypeRef getType() const { return Ty_; }

  llvm::ArrayRef<size_t> dims() const { return Ty_->dims(); }

  ElemKind getElementType() const { return Ty_->getElementType(); }

  bool isType(TypeRef T) { return Ty_ == T; }
};

/// Subclasses of Value have an enum that describe their kind.
class Kinded {
public:
  enum class Kind {
#define DEF_INSTR(CLASS, NAME) CLASS##Kind,
#define DEF_VALUE(CLASS, NAME) CLASS##Kind,
#define DEF_NODE(CLASS, NAME) CLASS##Kind,
#include "glow/IR/Instrs.def"
  };

  static const char *getKindName(Kind IK) {
    const char *names[] = {
#define DEF_INSTR(CLASS, NAME) #NAME,
#define DEF_VALUE(CLASS, NAME) #NAME,
#define DEF_NODE(CLASS, NAME) #NAME,
#include "glow/IR/Instrs.def"
        nullptr};
    return names[(int)IK];
  }

private:
  /// The kind of the value.
  Kind kind_;

public:
  /// Ctor.
  explicit Kinded(Kind vk) : kind_(vk) {}

  /// Returns the kind of the instruction.
  Kind getKind() const { return kind_; }

  const char *getKindName() const { return getKindName(kind_); }
};

} // namespace glow

#endif // GLOW_BASE_TRAITS_H
