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
#include <algorithm>

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

/// This is a manually written instruction to represent a Group of Fused
/// instructions. The class works by receieved
class FusionGroupInst final : public Instruction {
public:
  FusionGroupInst(llvm::StringRef name, Instruction *instr1,
                  Instruction *instr2)
      : Instruction(name, Kinded::Kind::FusionGroupInstKind, nullptr, {}) {
    transferOperands(instr1);
    transferOperands(instr2);
    instrs_.push_back(instr1);
    instrs_.push_back(instr2);
    instr1->setParentGroupFusionInstr(this);
    instr2->setParentGroupFusionInstr(this);
  }

  FusionGroupInst(llvm::StringRef name,
                  llvm::SmallVectorImpl<Instruction *> &instrs)
      : Instruction(name, Kinded::Kind::FusionGroupInstKind, nullptr, {}) {
    for (Instruction *instr : instrs) {
      transferOperands(instr);
      instrs_.push_back(instr);
      instr->setParentGroupFusionInstr(this);
    }
  }

  static bool classof(const Kinded *k) {
    return k->getKind() == Kinded::Kind::FusionGroupInstKind;
  }

  /// Returns a refernce to the vector of instructions making up the fused
  /// instruction.
  const std::vector<Instruction *> &getInstrs() const { return instrs_; }

  /// Dump out the Fused instruction with its operands to \p os stream.
  void dump(llvm::raw_ostream &os) const {
    os << "%" << (std::string)getName() << " = " << getKindName() << " ";
    dumpOperands(os);
    // Dump the Instruction the fusion is composed of
    os << "\n  {";
    for (auto *I : instrs_) {
      os << "\n    ";
      I->dump(os);
    }
    os << "\n  }";
  }

  /// Dump out the Fused instruction with its operands to default stream stream.
  void dump() const { dump(llvm::outs()); }

  bool verify() const { return true; }

  /// \p Returns true if \p instr is a member of this fusion group instr
  bool contains(Instruction *inst) const {
    auto result = std::find(instrs_.begin(), instrs_.end(), inst);
    return (result != std::end(instrs_));
  }

private:
  /// Instructions that make up this fused instruction.
  std::vector<Instruction *> instrs_;
  std::set<glow::Value *> fusedOperands_;

  /// Transfers the operands from \p instr to the newly created fusion
  /// instruction.
  void transferOperands(Instruction *instr) {
    for (auto &op : instr->getOperands()) {
      // Push the operand to the fused Instruction if it is not already added
      // before from an earlier instruction.
      if (fusedOperands_.count(op.first)) {
        continue;
      }
      pushOperand(op);
      fusedOperands_.insert(op.first);
    }
  }
};

} // namespace glow

// The rest of the nodes are auto-generated into this file:
#include "glow/AutoGenInstr.h"

#endif // GLOW_IR_INSTRS_H
