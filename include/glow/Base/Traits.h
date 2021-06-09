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
#ifndef GLOW_BASE_TRAITS_H
#define GLOW_BASE_TRAITS_H

#include "glow/Base/Type.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringRef.h"

namespace glow {

/// This add the capability to name subclasses.
class Named {
  std::string name_{};

public:
  explicit Named(llvm::StringRef name) : name_(name) {}

  /// \returns the name of the instruction.
  llvm::StringRef getName() const { return name_; }

  /// \returns the name of the instruction.
  bool hasName() const { return !name_.empty(); }

  /// Set the name of the instruction to \p name.
  void setName(llvm::StringRef name) { name_ = name.str(); }

  /// Compares by names, \returns true if name_ < x.name_.
  bool compareByName(const Named &x) const {
    return name_.compare(x.name_) > 0;
  }
};

/// Use to sort named classes by their name.
struct SortNamed {
  inline bool operator()(const Named *named1, const Named *named2) const {
    return named1->compareByName(*named2);
  }
};

/// Subclasses of this class have a type associated with them.
class Typed {
private:
  TypeRef Ty_{};

public:
  explicit Typed(TypeRef Ty) : Ty_(Ty) {}

  TypeRef getType() const { return Ty_; }

  void setType(TypeRef Ty) { Ty_ = Ty; }

  llvm::ArrayRef<dim_t> dims() const { return Ty_->dims(); }

  size_t size() const { return Ty_->size(); }

  size_t getSizeInBytes() const { return Ty_->getSizeInBytes(); }

  ElemKind getElementType() const { return Ty_->getElementType(); }

  bool isType(TypeRef T) { return Ty_ == T; }
};

/// Subclasses of Value have an enum that describe their kind.
class Kinded {
public:
  enum class Kind {
#define DEF_INSTR(CLASS, NAME) CLASS##Kind,
#define DEF_BACKEND_SPECIFIC_INSTR(CLASS, NAME) DEF_INSTR(CLASS, NAME)
#define DEF_VALUE(CLASS, NAME) DEF_INSTR(CLASS, NAME)
#include "glow/AutoGenInstr.def"
#define DEF_NODE(CLASS, NAME) CLASS##Kind,
#include "glow/AutoGenNodes.def"
  };

  static const char *getKindName(Kind IK) {
    const char *names[] = {
#define DEF_INSTR(CLASS, NAME) #NAME,
#define DEF_BACKEND_SPECIFIC_INSTR(CLASS, NAME) DEF_INSTR(CLASS, NAME)
#define DEF_VALUE(CLASS, NAME) DEF_INSTR(CLASS, NAME)
#include "glow/AutoGenInstr.def"
#define DEF_NODE(CLASS, NAME) #NAME,
#include "glow/AutoGenNodes.def"
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

using KindSet = llvm::SmallSet<Kinded::Kind, 6>;

/// Kind of the IR.
enum class IRKind {
  /// Glow high level graph IR.
  GlowGraphIRKind,
  /// Glow low level instruction IR.
  GlowInstructionIRKind,
  /// Glow FX IR.
  GlowFXIRKind,
};

class Module;

/// Subclasses of this class represent an IR container, e.g. a function or a
/// module.
class IRContainer : public Named {
public:
  IRContainer(llvm::StringRef name) : Named(name) {}
  virtual ~IRContainer() = default;
  virtual IRKind getIRKind() const = 0;
  virtual Module *getParent() = 0;
  virtual const Module *getParent() const = 0;
};

} // namespace glow

#endif // GLOW_BASE_TRAITS_H
