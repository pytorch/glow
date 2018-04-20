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
  explicit Named(llvm::StringRef name) : name_(name) {}

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
  explicit Typed(TypeRef Ty) : Ty_(Ty) {}

  TypeRef getType() const { return Ty_; }

  llvm::ArrayRef<size_t> dims() const { return Ty_->dims(); }

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
#include "AutoGenInstr.def"
#define DEF_NODE(CLASS, NAME) CLASS##Kind,
#include "AutoGenNodes.def"
  };

  static const char *getKindName(Kind IK) {
    const char *names[] = {
#define DEF_INSTR(CLASS, NAME) #NAME,
#define DEF_BACKEND_SPECIFIC_INSTR(CLASS, NAME) DEF_INSTR(CLASS, NAME)
#define DEF_VALUE(CLASS, NAME) DEF_INSTR(CLASS, NAME)
#include "AutoGenInstr.def"
#define DEF_NODE(CLASS, NAME) #NAME,
#include "AutoGenNodes.def"
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

/// Specifies the visibility of a Variable or WeightVar. Public nodes can't be
/// optimized because they are visible to external users that may hold a
/// reference or handles. Their equivalent WeightVar must be left Mutable at the
/// Instr level.
enum class VisibilityKind {
  Public,  // The variable is visible from outside the graph.
  Private, // The variable isn't visible from outside the graph.
};

} // namespace glow

#endif // GLOW_BASE_TRAITS_H
