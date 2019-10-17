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
#ifndef GLOW_IR_INSTRS_H
#define GLOW_IR_INSTRS_H

#include "glow/Base/Traits.h"
#include "glow/Base/Type.h"
#include "glow/Graph/Nodes.h"
#include "glow/IR/IR.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Casting.h"

namespace glow {

class WeightVar : public Value {
public:
  enum class MutabilityKind {
    Constant, // A read-only region of memory.
    Mutable,  // A read/write region of memory.
  };

private:
  /// The mutability mode.
  MutabilityKind mut_;

public:
  WeightVar(llvm::StringRef name, TypeRef Ty, MutabilityKind mut)
      : Value(name, Ty, Kinded::Kind::WeightVarKind), mut_(mut) {}

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::WeightVarKind;
  }

  static const char *getMutabilityStr(MutabilityKind mut);

  const char *getMutabilityStr() const;

  /// \returns true if the mutability kind of the weight is constant.
  bool isConstant() const {
    return getMutability() == MutabilityKind::Constant;
  }

  /// \returns the mutability kind of the weight.
  MutabilityKind getMutability() const { return mut_; }

  /// Updates the mutability kind of the weight to \p mut.
  void setMutability(MutabilityKind mut) { mut_ = mut; }

  void dump(llvm::raw_ostream &os) const;
  void verify() const {}
};

} // namespace glow

// The rest of the nodes are auto-generated into this file:
#include "glow/AutoGenInstr.h"

#endif // GLOW_IR_INSTRS_H
