// Copyright 2017 Facebook Inc.  All Rights Reserved.
#define DEBUG_TYPE "ir-optimizer"

#include "glow/Graph/Graph.h"
#include "glow/IR/IR.h"
#include "glow/IR/IRBuilder.h"
#include "glow/IR/IRUtils.h"
#include "glow/IR/Instrs.h"
#include "glow/Optimizer/Optimizer.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <iostream>
#include <unordered_map>
#include <unordered_set>

static llvm::cl::opt<bool>
    instrumentDebug("instrument-debug",
                    llvm::cl::desc("Instrument the IR for debugging"),
                    llvm::cl::init(false), llvm::cl::Hidden);
static llvm::cl::opt<bool> optimizeIR("optimize-ir",
                                      llvm::cl::desc("Enable IR optimizations"),
                                      llvm::cl::init(true), llvm::cl::Hidden);

using namespace glow;

using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;

/// Live interval of a memory buffer.
/// It represents a sequence of instructions [begin, end) where this buffer
/// holds a value.
struct Interval {
  /// Index of the interval begin.
  size_t begin_;
  /// Index of the interval end.
  size_t end_;
  /// True if the value may change between begin and end, e.g.
  /// due to @inout use.
  bool sameValue_{true};

  bool operator==(const Interval &other) const {
    return begin_ == other.begin_ && end_ == other.end_ &&
           sameValue_ == other.sameValue_;
  }

  std::string str() const {
    std::string s;
    llvm::raw_string_ostream sb{s};
    sb << "[" << begin_ << ", " << end_ << ", " << sameValue_ << ")";
    return sb.str();
  }
};

#ifndef NDEBUG
llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const Interval &I) {
  os << I.str();
  return os;
}
#endif

/// Set of intervals.
using Intervals = llvm::SmallVector<Interval, 4>;
/// Maping from a value to its live intervals.
using LiveIntervalsMap = std::unordered_map<Value *, Intervals>;
/// Set of instructions.
using Instructions = std::unordered_set<Instruction *>;

/// Hoists Dealloc instructions right after their last use.
static void hoistDealloc(Module &M) {
  // Maps activation instructions to their last non-dealloc user.
  std::unordered_map<Value *, InstrIterator> lastUser;
  auto &instrs = M.getInstrs();

  // Record the last use of each dealloc.
  for (auto it = instrs.begin(), e = instrs.end(); it != e; ++it) {
    if (isa<DeallocActivationInst>(*it))
      continue;

    if (auto alloc = dyn_cast<AllocActivationInst>(*it)) {
      lastUser[alloc] = it;
      continue;
    }

    for (int i = 0, e = (*it)->getNumOperands(); i < e; i++) {
      auto op = (*it)->getOperand(i).first;
      // Consider any use of a tensor_view to be also a use
      // of its source tensor. This is required to make
      // sure that a lifetime of a tensor_view is always
      // enclosed inside the lifetime of its source tensor.
      if (auto *alloc = getAllocationOrigin(op)) {
        lastUser[alloc] = it;
        continue;
      }
    }
  }

  // Now that we've found the last user we can hoist the instruction.
  for (auto it = instrs.begin(), e = instrs.end(); it != e;
       /* increment below */) {
    auto curr = it;
    ++it;
    auto *da = dyn_cast<DeallocActivationInst>(*curr);
    if (!da) {
      continue;
    }

    auto *alloc = cast<AllocActivationInst>(getOrigin(da->getOperand(0).first));
    auto where = lastUser[alloc];
    if (std::next(where) == curr) {
      // No need to move the instruction, because the last use was
      // right before the deallocation.
      continue;
    }
    ++where;
    M.moveInstruction(where, da);
  }
}

/// Sink Alloc instructions right before their first use.
static void sinkAllocas(Module &M) {
  /// A list of allocas to reschedule.
  std::unordered_set<AllocActivationInst *> allocs;
  auto &instrs = M.getInstrs();

  // Remove all of the allocas.
  for (auto it = instrs.begin(), e = instrs.end(); it != e;) {
    auto curr = it;
    auto *aa = dyn_cast<AllocActivationInst>(*curr);
    if (!aa) {
      ++it;
      continue;
    }

    allocs.insert(aa);
    it = M.removeInstruction(curr);
  }

  // Place all of the allocas in the right place:
  for (auto it = instrs.begin(), e = instrs.end(); it != e; ++it) {
    for (int i = 0, e = (*it)->getNumOperands(); i < e; i++) {
      auto op = (*it)->getOperand(i).first;
      auto aa = dyn_cast<AllocActivationInst>(op);
      if (!aa) {
        continue;
      }
      auto A = allocs.find(aa);
      if (A == allocs.end()) {
        continue;
      }
      allocs.erase(A);
      M.insertInstruction(it, aa);
      if (allocs.empty())
        return;
    }
  }

  assert(allocs.empty() && "Forgot to insert some allocas!");
}

/// Delete alloc instructions that have no readers or writers.
static void deleteDeadAllocs(Module &M) {
  auto &instrs = M.getInstrs();

  llvm::SmallVector<Instruction *, 16> ErasedInstructions{};

  // Remove all unused tenstorviews.
  std::copy_if(instrs.begin(), instrs.end(),
               std::back_inserter(ErasedInstructions),
               [](const Instruction *I) -> bool {
                 return (isa<TensorViewInst>(I) && I->getNumUsers() == 0);
               });

  for (auto I : ErasedInstructions) {
    M.eraseInstruction(I);
  }
  ErasedInstructions.clear();

  // Remove all of the DeallocActivationInst that close unused allocs.
  std::copy_if(
      instrs.begin(), instrs.end(), std::back_inserter(ErasedInstructions),
      [](const Instruction *I) -> bool {
        if (const auto *DA = dyn_cast<const DeallocActivationInst>(I)) {
          return DA->getAlloc()->getNumUsers() < 2;
        }
        return false;
      });

  for (auto I : ErasedInstructions) {
    M.eraseInstruction(I);
  }

  ErasedInstructions.clear();
  // Remove the unused allocs.
  std::copy_if(instrs.begin(), instrs.end(),
               std::back_inserter(ErasedInstructions),
               [](const Instruction *I) -> bool {
                 if (isa<const AllocActivationInst>(I)) {
                   return I->getNumUsers() < 2;
                 }
                 return false;
               });

  for (auto I : ErasedInstructions) {
    M.eraseInstruction(I);
  }
}

// Replace all users of some value with another value, but don't touch the
// dealloc instruction, because we need to preserve the well formdness of the
// IR.
static void replaceAllNonDeallocUsersWith(Value *val, Value *with) {
  assert(val != with && "Replacing value with self");
  auto &users = val->getUsers();
  auto *withOrigin = getOrigin(with);
  // We use a vector here because changing the operands of the user changes the
  // uselist, and this invalidates the iterator.
  llvm::SmallVector<Use, 6> usersVec(users.begin(), users.end());
  for (auto &U : usersVec) {
    auto *I = U.get();
    auto &M = *I->getParent();
    IRBuilder B(&M);
    // Ignore dealloc instrs.
    if (isa<DeallocActivationInst>(I)) {
      continue;
    }

    auto Op = U.getOperand();
    auto replacement = with;
    if (Op.first->getType() != replacement->getType())
      replacement = withOrigin;
    if (Op.first->getType() != replacement->getType()) {
      // Perform a cast if required.
      auto *tv = B.createTensorViewInst(I->getName(), replacement,
                                        Op.first->getType());
      M.moveInstruction(I, tv);
      replacement = tv;
    }
    U.setOperand(replacement);
  }
}

/// \returns the pointer to the single writer that writes into this value, or
/// nullptr if the number of writers is not exactly one.
static Instruction *getSingleWriter(const Value *V) {
  Instruction *singleUser = nullptr;
  for (const auto &U : ValueUses(V)) {
    Instruction *user = U.get();

    // Ignore deallocs.
    if (isa<DeallocActivationInst>(user))
      continue;

    auto op = U.getOperand();

    // Ignore the readers.
    if (op.second == OperandKind::In) {
      continue;
    }

    // Multiple users.
    if (singleUser) {
      return nullptr;
    }

    singleUser = user;
  }

  return singleUser;
}

void makeWeightsConst(Module &M) {
  // For each weight:
  for (auto *W : M.getWeights()) {
    bool readOnly = true;
    // For each instruction that uses the weight:
    for (const auto &U : ValueUses(W)) {
      auto kind = U.getOperand().second;
      // Check if all of the users are read-only.
      if (kind != OperandKind::In) {
        readOnly = false;
        break;
      }
    }

    // Mark the variable as read only.
    if (readOnly) {
      W->setMutability(WeightVar::MutabilityKind::Constant);
    } else {
      W->setMutability(WeightVar::MutabilityKind::Mutable);
    }
  }
}

#ifndef NDEBUG
/// Dump a live intervals map.
static void LLVM_ATTRIBUTE_UNUSED dump(Module &M,
                                       LiveIntervalsMap &IntervalsMap) {
  llvm::outs() << "\nDumping live intervals map:\n";
  for (const auto &I : IntervalsMap) {
    llvm::outs() << "\nValue " << I.first->getName();
    llvm::outs() << "\n";
    for (const auto &Interval : I.second) {
      llvm::outs() << Interval << " ";
    }
    llvm::outs() << "\n";
  }
}
#endif

/// Compute live intervals for each mutable location represented by
/// Value which is either an AllocActivationInst or a WeightVar.
/// Each such value is mapped to a list of intervals where it is alive.
/// Each interval starts at the point of definition and ends at last use
/// of the current value, which is assigned at the beginning of the current
/// interval. If there are multiple writes to the same mutable memory
/// location, then each such assignment would result in a new interval.
static void calculateLiveIntervals(Module &M, LiveIntervalsMap &liveness) {
  assert(liveness.empty() &&
         "This function should be called with empty liveness map");
  auto &instrs = M.getInstrs();
  unsigned instIdx = 0;

  // Compute the [start..end) intervals for each alloc activation in our basic
  // block. Notice that we ignore Dealloc instructions in our analysis.
  for (auto it = instrs.begin(), e = instrs.end(); it != e;
       ++it, instIdx += InstructionNumbering::MAX_SLOT) {
    auto *I = *it;
    // Ignore deallocations in our liveness calculation.
    if (isa<DeallocActivationInst>(I)) {
      continue;
    }

    // Ignore tensorview instructions, because they are just aliases
    // and do not represent a read or write, even though formally they
    // are reads due to the @in src parameter.
    if (isa<TensorViewInst>(I))
      continue;

    auto InstOperands = I->getOperands();
    llvm::SmallVector<Instruction::Operand, 8> SortedOperands(
        InstOperands.begin(), InstOperands.end());

    // Sort operands so that:
    // - all operands referencing the same Value are grouped together.
    // - operands related to the same Value are always in the following
    // order: In, InOut, Out.
    //
    // This ordering ensures that we process reads before writes.
    std::sort(SortedOperands.begin(), SortedOperands.end());

    for (int i = 0, e = SortedOperands.size(); i < e; i++) {
      auto op = SortedOperands[i].first;
      auto opKind = SortedOperands[i].second;
      // Look through tensorviews. As a result, all operations
      // on tensorviews are accounted as operations on their
      // origins.
      auto opOrigin = getOrigin(op);
      Value *loc = dyn_cast<AllocActivationInst>(opOrigin);
      if (!loc) {
        loc = dyn_cast<WeightVar>(opOrigin);
      }
      // Bail if the operand is not an AllocActivationInst or a WeightVar.
      if (!loc) {
        continue;
      }

      auto opIdx = instIdx;
      if (opKind == OperandKind::Out) {
        opIdx = InstructionNumbering::getInstrWriteSlotNumber(instIdx);
      } else {
        opIdx = InstructionNumbering::getInstrReadSlotNumber(instIdx);
      }

      auto found = liveness.find(loc);
      if (found == liveness.end()) {
        // Create a new interval.
        liveness[loc].push_back({opIdx, opIdx + 1});
        // If it is a first use, it should be either an input variable or
        // a write.
        // FIXME: Remove InOut!
        assert((isa<TensorViewInst>(I) || isa<WeightVar>(opOrigin) ||
                opKind == OperandKind::Out || opKind == OperandKind::InOut) &&
               "First reference inside a live interval should be either an "
               "input variable or a write");
        continue;
      }

      auto &Intervals = found->second;
      // Extend the interval but only if current use is not a write or
      // if it is a write, but we have seen a read before.
      if (opKind != OperandKind::Out)
        Intervals.back().end_ = opIdx + 1;

      // How @inout operands should be handled?
      // They cannot be treated as an end of an interval and a beginning of a
      // new one, because this would imply that this new interval completely
      // overwrites the buffer, which is not true in general.
      // @inout operands cannot be considered to be simple reads, because it
      // would mean that the value does not change for the duration of the
      // interval, which is not the case. To handle this, @inout operands are
      // considered to be a part of the existing interval, but the sameValue_
      // flag is set to false to indicate that the value is not guaranteed to be
      // the same inside the interval.
      if (opKind == OperandKind::InOut)
        Intervals.back().sameValue_ = false;

      // No need to create a new interval if it is not a write.
      if (opKind != OperandKind::Out) //|| opKind == OperandKind::InOut)
        continue;
      opIdx = InstructionNumbering::getInstrWriteSlotNumber(instIdx);
      // This instruction modifies the memory location.
      // Therefore, end the current active live interval
      // for this memory location and begin a new one.
      Intervals.push_back({opIdx, opIdx + 1});
    }
  }

  for (auto &Entry : liveness) {
    auto *ML = Entry.first;
    auto &IL = Entry.second;
    if (isa<WeightVar>(ML)) {
      assert(!IL.empty() && "Live interval list cannot be empty");
      // Extend the last interval till the end of the program
      // to express that all mutable weights are used outside.
      IL.back().end_ = instIdx;
    }
  }
}

/// Provided a set of intervals, return the interval covering
/// a given instruction.
static Intervals::iterator getEnclosingInterval(Intervals &LiveIntervals,
                                                size_t instIdx) {
  for (auto I = LiveIntervals.begin(), E = LiveIntervals.end(); I != E; ++I) {
    if (I->begin_ <= instIdx && instIdx < I->end_)
      return I;
  }
  return LiveIntervals.end();
}

/// Returns true if RHS is enclosed inside LHS.
static bool isEnclosedInside(Interval &LHS, Interval &RHS) {
  return LHS.begin_ < RHS.begin_ && RHS.end_ <= LHS.end_;
}

/// \returns true of any intervals from \p Ints overlap with interval \p I.
static bool hasOverlappingIntervals(Intervals &Ints, Interval I) {
  for (const auto &CurI : Ints) {
    if (std::max(CurI.begin_, I.begin_) < std::min(CurI.end_, I.end_))
      return true;
  }
  return false;
}

/// Moves an interval from one interval list to another.
static void moveInterval(Intervals &from, Intervals &to, Interval &interval) {
  auto fromIt = std::find(from.begin(), from.end(), interval);
  assert(fromIt != from.end() && "Interval should exist in the from list");
  // Nothing to do if interval is enclosed into one of to intervals.
  // Add to the to list.
  bool isEnclosed = false;
  for (auto &I : to) {
    if (isEnclosedInside(I, interval)) {
      isEnclosed = true;
      break;
    }
  }
  if (!isEnclosed) {
    to.push_back(interval);
    // Let sort find a right position for it.
    // std::sort(to.begin(), to.end());
  }
  // Delete from the from list.
  from.erase(fromIt);
}

/// Replace all uses of \p val by \p with inside interval \p liveInterval.
static void
replaceAllUsesInsideIntervalWith(Value *val, Value *with,
                                 Interval &liveInterval, Module &M,
                                 InstructionNumbering &InstrNumbering) {
  auto &instrs = M.getInstrs();
  auto valOrigin = getOrigin(val);
  auto withOrigin = getOrigin(with);
  int instIdx = 0;
  IRBuilder B(&M);
  for (auto it = InstrNumbering.getInstr(liveInterval.begin_), e = instrs.end();
       it != e && instIdx <= liveInterval.end_; ++it) {
    auto *I = *it;
    if (isa<DeallocActivationInst>(I))
      continue;
    // Ignore any new instructions which were not present as the instruction
    // numbering was performed.
    auto instNum = InstrNumbering.getInstrNumber(I);
    if (instNum >= 0)
      instIdx = instNum;
    if (instNum < 0)
      continue;

    // This is an instruction inside the interval.
    // Iterate over all operands and perform replacements.
    for (int i = 0, e = I->getNumOperands(); i < e; i++) {
      auto op = I->getOperand(i).first;
      auto opOrigin = getOrigin(op);
      auto opKind = I->getOperand(i).second;
      // Is the operand the value we are looking for?
      if (opOrigin != valOrigin)
        continue;
      auto opIdx = (opKind == OperandKind::In)
                       ? InstructionNumbering::getInstrReadSlotNumber(instIdx)
                       : InstructionNumbering::getInstrWriteSlotNumber(instIdx);
      // Skip operands outside of the interval.
      if (opIdx < liveInterval.begin_ || opIdx >= liveInterval.end_)
        continue;
      auto replacement = with;
      if (op->getType() != replacement->getType())
        replacement = withOrigin;
      if (op->getType() != replacement->getType()) {
        // Perform a cast if required.
        auto *tv = B.createTensorViewInst((*it)->getName(), replacement,
                                          op->getType());
        M.moveInstruction(it, tv);
        replacement = tv;
      }

      DEBUG(llvm::dbgs() << "Replacing inside instruction " << opIdx << "\n";
            llvm::dbgs() << "before: "; I->dump(llvm::dbgs());
            llvm::dbgs() << "\n");
      // Replace the old value by the new value.
      I->setOperand(i, replacement);
      DEBUG(llvm::dbgs() << "after: "; I->dump(llvm::dbgs());
            llvm::dbgs() << "\n");
    }
  }
}

/// Erase all instructions from the \p ErasedInstructions set.
/// If \p forceErase is true, no additional checks are performed.
/// Otherwise, copies into weight variables cannot be erased.
static void eraseInstructions(Module &M, Instructions &ErasedInstructions) {
  for (auto it : ErasedInstructions) {
    DEBUG(llvm::dbgs() << "Deleting instruction :"; it->dump(llvm::dbgs());
          llvm::dbgs() << "\n");
    M.eraseInstruction(it);
  }
}

/// \returns true if writes into this memory location are observable from
/// outside.
static bool isObservable(Value *V) { return isa<WeightVar>(getOrigin(V)); }

/// Substitute all uses of \p oldBuffer by \p newBuffer inside the
/// \p oldInterval.
static void reuseBufferInsideInterval(Value *oldBuffer, Value *newBuffer,
                                      Interval oldInterval, Module &M,
                                      InstructionNumbering &InstrNumbering,
                                      Intervals &oldIntervals,
                                      Intervals &newIntervals) {
  DEBUG(llvm::dbgs() << "\n\nReuse buffers: use buffer of "
                     << newBuffer->getName() << " as a buffer for "
                     << oldBuffer->getName() << "\n"
                     << "in live interval " << oldInterval << "\n");
  // Replace src by dest.
  replaceAllUsesInsideIntervalWith(oldBuffer, newBuffer, oldInterval, M,
                                   InstrNumbering);
  // srcInterval does not belong to srcIntervals anymore. It should belong
  // to destIntervals now.
  moveInterval(oldIntervals, newIntervals, oldInterval);
}

/// Tries to share a buffer for two operands of the same instruction.
/// An operand X cannot reuse the buffer of another operand Y,
/// if the live interval of X overlaps with any live intervals of Y.
/// The only exception is the copy instruction, where the live interval
/// of the X may be enclosed into a live interval of Y because they have
/// the same value after the copy instruction.
static void tryToShareBuffersForInstr(LiveIntervalsMap &IntervalsMap,
                                      InstructionNumbering &InstrNumbering,
                                      Instruction *I, int InstIdx) {
  Module &M = *I->getParent();
  for (unsigned first = 0, e = I->getNumOperands(); first < e; first++) {
    auto destOp = I->getOperand(first);
    Value *dest = getAllocationOrigin(destOp.first);
    if (!dest)
      dest = destOp.first;
    for (unsigned second = first + 1; second < e; second++) {
      auto srcOp = I->getOperand(second);
      Value *src = getAllocationOrigin(srcOp.first);
      if (!src)
        src = srcOp.first;
      // Operands must be different, but of the same type.
      if (dest->getType() != src->getType()) {
        continue;
      }

      if (dest == src) {
        // Bail if operands are the same and are combined already.
        if (Instruction::isInplaceOp(I, first, second))
          return;
        continue;
      }

      // Bail if the instruction does not allow for reuse of buffers.
      if (!Instruction::isInplaceOp(I, first, second)) {
        continue;
      }

      auto srcOrigin = getOrigin(src);
      auto destOrigin = getOrigin(dest);

      auto &srcIntervals = IntervalsMap[srcOrigin];
      auto &destIntervals = IntervalsMap[destOrigin];

      // Bail if information about live intervals is not known.
      assert(!srcIntervals.empty() &&
             "Set of live intervals for a memory buffer cannot be empty");
      assert(!destIntervals.empty() &&
             "Set of live intervals for a memory buffer cannot be empty");

      // Find the Src live interval that encloses instIdx.
      auto srcIntervalIt = getEnclosingInterval(
          srcIntervals, InstructionNumbering::getInstrReadSlotNumber(InstIdx));
      assert(srcIntervalIt != srcIntervals.end() &&
             "Cannot share buffers: cannot "
             "find enclosing src interval");

      // Find the Dest live interval that encloses instIdx.
      auto destIntervalIt = getEnclosingInterval(
          destIntervals,
          InstructionNumbering::getInstrWriteSlotNumber(InstIdx));
      assert(destIntervalIt != destIntervals.end() &&
             "Cannot share buffers: cannot "
             "find enclosing src interval");

      auto &srcInterval = *srcIntervalIt;
      auto &destInterval = *destIntervalIt;

      // Live interval destInterval is adjacent to live interval srcInterval,
      // because they are connected via the current instruction.

      // A helper logic for selecting which buffer can be reused for the other
      // one. Returns a buffer that can be reused or nullptr, if reuse is not
      // possible.
      auto GetBufferToBeReused = [&]() -> Value * {
        // Do not try to combine observables.
        if (isObservable(destOrigin) && isObservable(srcOrigin))
          return nullptr;
        // Check if dest or src live interval is the last live interval of
        // an observable memory location.
        bool isDestLastIntervalOfObservable =
            isObservable(destOrigin) && destInterval == destIntervals.back();
        bool isSrcLastIntervalOfObservable =
            isObservable(srcOrigin) && srcInterval == srcIntervals.back();

        // A value X cannot reuse the buffer of another value Y,
        // if the live interval of X overlaps with any live intervals of Y.
        // The only exception is the copy instruction, where the live interval
        // of the destination may be enclosed into a live interval of the source
        // because they have the same value.
        bool canCopyPropagate = isa<CopyInst>(I) &&
                                isEnclosedInside(srcInterval, destInterval) &&
                                srcInterval.sameValue_;
        bool destIntvalCannotBeReplaced =
            !canCopyPropagate &&
            hasOverlappingIntervals(srcIntervals, destInterval);
        bool srcIntervalCannotBeReplaced =
            hasOverlappingIntervals(destIntervals, srcInterval);

        if (!isDestLastIntervalOfObservable && !isSrcLastIntervalOfObservable &&
            !destIntvalCannotBeReplaced && !srcIntervalCannotBeReplaced) {
          // There are no restrictions and intervals can be combined on any
          // order. Try to use a heuristic to pick the right way to combine
          // them.

          // Try to reuse the interval of an observable memory location, because
          // it does not increase the memory usage.
          // TODO: If it would introduce a last write into an observable, do not
          // do it.
          if (isObservable(srcOrigin) && !isObservable(destOrigin)) {
            // Use src buffer for dest.
            return srcOrigin;
          }

          if (isObservable(destOrigin) && !isObservable(srcOrigin)) {
            // Use dest buffer for src.
            return destOrigin;
          }

          // Avoid sharing a buffer if there is a single
          // live interval in the interval list. After replacement
          // this whole buffer can be eliminated.
          if (srcIntervals.size() == 1 && destIntervals.size() != 1) {
            // Use dest buffer for src.
            return destOrigin;
          }

          if (destIntervals.size() == 1 && srcIntervals.size() != 1) {
            // Use src buffer for dest.
            return srcOrigin;
          }

          // Just use src buffer for dest by default.
          return srcOrigin;
        }

        // Try to check if buffers can be shared by using
        // src instead of dst inside the live interval of dest.
        // This is possible if src is not live after the current instruction and
        // until the end of the current Dest's live interval.
        if (isDestLastIntervalOfObservable || destIntvalCannotBeReplaced) {
          // Dest cannot be replaced by src because src is being mutated while
          // dest is alive or because dest contains the last write into
          // an observable memory location.

          // Try to replace src by dest in the live interval of src.
          // This is possible if Src is not live anywhere inside the current
          // Dest's live interval ending at the current instruction.

          // Bail, because src cannot be replaced by dest because dest is being
          // mutated while src is alive or because src contains the last write
          // into an observable memory location.
          if (isSrcLastIntervalOfObservable || srcIntervalCannotBeReplaced)
            return nullptr;
          return destOrigin;
        }

        return srcOrigin;
      };

      auto *bufferToReuse = GetBufferToBeReused();
      if (!bufferToReuse)
        continue;

      // TODO: May be disallow usage of dest interval for src?
      // This would avoid extending dest lifetime to start earlier.
      // But it would also miss some opportunities for sharing buffers.
      // if (bufferToReuse == destOrigin)
      //   continue;

      if (bufferToReuse == destOrigin) {
        // This operation may extend the lifetime of dest's buffer.
        // shareBuffers will fix it once it's done.
        reuseBufferInsideInterval(srcOrigin, dest, srcInterval, M,
                                  InstrNumbering, srcIntervals, destIntervals);
        return;
      }

      reuseBufferInsideInterval(destOrigin, src, destInterval, M,
                                InstrNumbering, destIntervals, srcIntervals);
      return;
    }
  }
}

/// Sharing of buffers
///
/// The purpose of this optimization is to reuse memory usage by
/// reusing memory buffers as much as possible.
///
/// The overall idea is that it is fine to combine storage for two live
/// intervals if they do not overlap and if for at least one of them the writes
/// do not need to be observable. Typically, two live intervals are considred as
/// candidates for sharing if they occur in the same instruction, but it is not
/// strictly necessary.
static void shareBuffers(Module &M) {
  Instructions ErasedInstructions;
  // Build a list of live intervals for each memory location
  // which is either a WeightVar or a an Allocation.
  LiveIntervalsMap IntervalsMap;
  calculateLiveIntervals(M, IntervalsMap);
  InstructionNumbering InstrNumbering(M);

  // Get the source of the copy. This memory location may have been
  // modified by any instruction that used it as an @out or @inout
  // parameter.
  auto &instrs = M.getInstrs();

  // For each instruction, in reverse order.
  for (auto it = instrs.rbegin(), e = instrs.rend(); it != e; ++it) {
    Instruction *I = *it;
    auto instIdx = InstrNumbering.getInstrNumber(I);
    if (instIdx < 0)
      continue;
    // Try to reuse the operand memory buffers.
    tryToShareBuffersForInstr(IntervalsMap, InstrNumbering, I, instIdx);
  }

  // Fix eventual issues with allocs and deallocs that shareBuffers may
  // introduce by extending live interval lifetimes.
  hoistDealloc(M);
  sinkAllocas(M);
}

/// Dead Store Elimination.
///
/// Perform a backwards pass:
/// - For each location remember the last seen read.
/// - When a write is detected:
///   - If there is no last seen read, it is safe to remove this write
///   - Remember this last seen write, reset the last seen read.
/// A single pass is enough because currently there is just a single basic
/// basic block.
static void eliminateDeadStores(Module &M) {
  auto &instrs = M.getInstrs();
  // Instructions to be erased.
  Instructions ErasedInstructions;
  /// Representation of the analysis state.
  struct MemoryLocationState {
    /// Instruction that contained a last seen read.
    Instruction *lastSeenRead_{nullptr};
    /// Instruction that contained a last seen write.
    Instruction *lastSeenWrite_{nullptr};
  };

  // Maps each memory location to its analysis state.
  std::unordered_map<Value *, MemoryLocationState> memoryState;

  // Create a fake last read for each of the weight variables,
  // to indicate that WeightVars are live at the end of the BB.
  // This ensures that last stored into WeightVars are not
  // eliminated.
  for (auto *WV : M.getWeights()) {
    memoryState[WV].lastSeenRead_ = *std::prev(instrs.end());
  }

  // Iterate over instructions in reversed order.
  for (auto it = instrs.rbegin(), e = instrs.rend(); it != e; ++it) {
    auto *I = *it;
    if (isa<DeallocActivationInst>(I) || isa<AllocActivationInst>(I) ||
        isa<TensorViewInst>(I))
      continue;
    size_t NumMutatedOperands = 0;
    size_t NumNonReadMutatedOperands = 0;
    // Process all operand writes.
    for (const auto &Op : I->getOperands()) {
      auto OpOrigin = getOrigin(Op.first);
      auto OpKind = Op.second;
      auto &State = memoryState[OpOrigin];
      if (OpKind != OperandKind::In) {
        NumMutatedOperands++;
        // If it a write that was not read and it is not a last write into
        // a WeightVar (i.e. an observable effect), then is can be eliminated.
        // If there are multiple writes in this instruction, all of them
        // should satisfy this property for the instruction to be removed.
        if (!State.lastSeenRead_) {
          NumNonReadMutatedOperands++;
        }
        State.lastSeenWrite_ = I;
        State.lastSeenRead_ = nullptr;
      }
    }

    // It is safe to remove an instruction if all of its mutated operands
    // are not read afterwards.
    if (NumMutatedOperands > 0 &&
        NumMutatedOperands == NumNonReadMutatedOperands) {
      ErasedInstructions.insert(I);
      // Do not process any reads of operands, because
      // this instruction will be eliminated.
      continue;
    }

    // Process all operand reads.
    for (const auto &Op : I->getOperands()) {
      auto OpOrigin = getOrigin(Op.first);
      auto OpKind = Op.second;
      auto &State = memoryState[OpOrigin];
      if (OpKind != OperandKind::Out) {
        State.lastSeenRead_ = I;
      }
    }
  }

  eraseInstructions(M, ErasedInstructions);
}

/// Instrument the code to make it easier to debug issues.
/// Add dumping of inputs before each instruction and
/// dumping of outputs after each instruction.
/// For each input/output tensor its name and its value are dumped.
static void performDebugInstrumentation(Module &M) {
  if (!instrumentDebug)
    return;

  auto &instrs = M.getInstrs();
  for (auto it = instrs.begin(), e = instrs.end(); it != e;) {
    auto next = std::next(it);
    if (isa<DebugPrintInst>(*it) || isa<AllocActivationInst>(*it) ||
        isa<DeallocActivationInst>(*it)) {
      it = next;
      continue;
    }
    auto instrName = (*it)->getName();
    for (const auto &Op : (*it)->getOperands()) {
      // Dump inputs of the current instruction before the instruction.
      if (Op.second != OperandKind::Out) {
        std::string name = "debug_print.before.";
        name += Op.first->getName();
        name += ".";
        name += instrName;
        auto *dumpInstr = new DebugPrintInst(&M, name, Op.first);
        M.insertInstruction(it, dumpInstr);
      }

      // Dump outputs of the current instruction after the instruction.
      if (Op.second != OperandKind::In) {
        std::string name = "debug_print.after.";
        name += Op.first->getName();
        name += ".";
        name += instrName;
        auto *dumpInstr = new DebugPrintInst(&M, name, Op.first);
        M.insertInstruction(next, dumpInstr);
      }
    }
    it = next;
  }
}

/// Perform peephole optimizations.
void performPeepholeOptimizations(Module &M) {
  auto &instrs = M.getInstrs();
  IRBuilder B(&M);
  for (auto it = instrs.begin(), e = instrs.end(); it != e;) {
    auto cur = it;
    auto *I = *cur;
    it = std::next(it);
    // PoolMaxWithXYInst -> PoolMaxInst.
    if (auto *PMI = dyn_cast<PoolMaxWithXYInst>(I)) {
      auto *SrcXY = PMI->getSrcXY();
      // Optimize only if the cache is an allocation and
      // it has exactly 2 users: the current instruction and
      // a deallocation.
      if (!isa<AllocActivationInst>(SrcXY) || SrcXY->getNumUsers() != 2)
        continue;

      auto *NewPMI = B.createPoolMaxInst(PMI->getName(), PMI->getDest(),
                                         PMI->getSrc(), PMI->getKernel(),
                                         PMI->getStride(), PMI->getPad());
      it = M.moveInstruction(cur, NewPMI);
      M.eraseInstruction(cur);
      continue;
    }

    // SoftMaxWithXYInst -> SoftMaxInst.
    if (auto *SMI = dyn_cast<SoftMaxWithEInst>(I)) {
      auto *E = SMI->getE();
      // Optimize only if the cache is read exactly once,
      // namely by this instruction.
      bool isUsedE = false;
      for (auto &U : ValueUses(getOrigin(E))) {
        if (U.getOperand().second != OperandKind::Out && U.get() != SMI) {
          isUsedE = true;
        }
      }
      if (isUsedE)
        continue;

      auto *NewSMI = B.createSoftMaxInst(SMI->getName(), SMI->getDest(),
                                         SMI->getSrc(), SMI->getSelected());
      it = M.moveInstruction(cur, NewSMI);
      M.eraseInstruction(cur);
      continue;
    }

    // reshape -> tensorview, copy
    if (auto *RI = dyn_cast<ReshapeInst>(I)) {
      auto *TVI = B.createTensorViewInst(RI->getName(), RI->getSrc(),
                                         RI->getDest()->getType());
      it = M.moveInstruction(cur, TVI);
      auto *CI = B.createCopyInst(RI->getName(), RI->getDest(), TVI);
      M.moveInstruction(cur, CI);
      M.eraseInstruction(cur);
      continue;
    }

    // tranpose dest, splat (src), ... -> copy dest, tensorview (splat (src))
    // This is safe, because transpose of a splat does not change any elements.
    // It changes only types.
    if (auto *TI = dyn_cast<TransposeInst>(I)) {
      auto Src = TI->getSrc();
      auto Dest = TI->getDest();
      if (auto W = getSingleWriter(Src)) {
        if (isa<SplatInst>(W)) {
          if (Src->getType() != Dest->getType()) {
            auto *TVI =
                B.createTensorViewInst(TI->getName(), Src, Dest->getType());
            M.moveInstruction(cur, TVI);
            Src = TVI;
          }
          auto *CI = B.createCopyInst(TI->getName(), TI->getDest(), Src);
          it = M.moveInstruction(cur, CI);
          M.eraseInstruction(cur);
          continue;
        }
      }
    }

    // Convert element_max instruction into a canonical form,
    // where the splat (i.e. the constant) argument is the last one.
    if (auto *EM = dyn_cast<ElementMaxInst>(I)) {
      auto *LHS = EM->getLHS();
      auto *RHS = EM->getRHS();
      auto *WLHS = getSingleWriter(LHS);
      if (!WLHS)
        continue;
      if (!isa<SplatInst>(WLHS))
        continue;
      // If RHS is a splat already, there is nothing to do.
      auto *WRHS = getSingleWriter(RHS);
      if (WRHS && isa<SplatInst>(WRHS))
        continue;
      auto *NewEM =
          B.createElementMaxInst(EM->getName(), EM->getDest(), RHS, LHS);
      it = M.moveInstruction(cur, NewEM);
      M.eraseInstruction(cur);
      continue;
    }

    // tensorview that does not change the type is equivalent to its source
    // operand.
    if (auto *TV = dyn_cast<TensorViewInst>(I)) {
      if (TV->getType() == TV->getSrc()->getType()) {
        replaceAllNonDeallocUsersWith(TV, TV->getSrc());
      }
      continue;
    }

    // Remove useless copies.
    if (auto *CI = dyn_cast<CopyInst>(I)) {
      if (getOrigin(CI->getSrc()) == getOrigin(CI->getDest()))
        M.eraseInstruction(cur);
      continue;
    }
  }
}

void glow::optimize(Module &M, CompilationMode mode) {
  M.verify();
  if (!optimizeIR)
    return;

  performPeepholeOptimizations(M);

  eliminateDeadStores(M);

  // Reuse buffers from previous operations.
  shareBuffers(M);

  performPeepholeOptimizations(M);

  // Shorten the lifetime of buffers.
  hoistDealloc(M);
  sinkAllocas(M);

  // Perform Dead Store Elimination.
  eliminateDeadStores(M);

  deleteDeadAllocs(M);

  // Turn read-only weights into constant weights.
  makeWeightsConst(M);

  // Perform a debug instrumentation if required.
  performDebugInstrumentation(M);

  M.verify();
}
