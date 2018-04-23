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
  /// Index of the interval begin. Typically this is the index of the
  /// instruction, which overwrites the buffer.
  size_t begin_;
  /// Index of the interval end. Typically, it is the last use of the current
  /// value held in the buffer.
  size_t end_;
  /// True if the value may change between begin and end, e.g.
  /// due to @inout use. In most cases, the value does not change for the
  /// duration of a single live interval.
  bool sameValue_{true};

  Interval(size_t begin, size_t end, bool sameValue = true)
      : begin_(begin), end_(end), sameValue_(sameValue) {}

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

/// A helper class used for instructions numbering used by live intervals.
/// It follows the LLVM's linear scan register allocator approach and assigns
/// different numbers to read and write slots of the same instruction, which
/// allows for an easy construction of a very precise set of live intervals.
class LiveIntervalsInstructionNumbering {
  using NumberedInstructionMap = std::vector<InstrIterator>;
  using InstructionNumbersMap = std::unordered_map<Instruction *, size_t>;
  /// Maps the number to an instruction.
  NumberedInstructionMap numToInstr_;
  /// Maps an instruction to its number.
  InstructionNumbersMap instrToNum_;

public:
  /// Virtual slot number to be used for instructions numbering. It helps to
  /// distinguish reads from writes and makes comparision of live intervals
  /// easier. LLVM used a similar approach for the linear scan register
  /// allocator.
  ///
  /// For an instruction with number N, its @in operands would be considered
  /// to be at (N+READ_SLOT), its @out operands would be at (N+WRITE_SLOT).
  enum SLOTS {
    READ_SLOT = 0,
    WRITE_SLOT = 2,
    MAX_SLOT = 4,
  };

  LiveIntervalsInstructionNumbering(IRFunction &M) {
    auto &instrs = M.getInstrs();
    size_t instIdx = 0;
    numToInstr_.reserve(instrs.size());

    for (auto it = instrs.begin(), e = instrs.end(); it != e;
         instIdx += MAX_SLOT, ++it) {
      numToInstr_.push_back(it);
      instrToNum_[*it] = instIdx;
    }
  }

  /// \returns the base number of the instruction.
  /// It is the same for all slots of a given instruction.
  static int64_t getInstrBaseNumber(int64_t idx) {
    return idx / MAX_SLOT * MAX_SLOT;
  }

  /// \returns true if \p idx is the instruction number of the read slot of the
  /// instruction.
  static bool isReadSlotNumber(int64_t idx) {
    return idx % MAX_SLOT == READ_SLOT;
  }

  /// \returns true if \p idx is the instruction number of a write slot of the
  /// instruction.
  static bool isWriteSlotNumber(int64_t idx) {
    return idx % MAX_SLOT == WRITE_SLOT;
  }

  /// \returns the instruction number of a read slot of instruction with number
  /// \p idx.
  static int64_t getInstrReadSlotNumber(int64_t idx) {
    return getInstrBaseNumber(idx) + READ_SLOT;
  }

  /// \returns the instruction number of a write slot of instruction with number
  /// \p idx.
  static int64_t getInstrWriteSlotNumber(int64_t idx) {
    return getInstrBaseNumber(idx) + WRITE_SLOT;
  }

  /// \returns the number of the instruction, or -1 if the instruction is not
  /// numbered.
  int64_t getInstrNumber(Instruction *I) const {
    auto result = instrToNum_.find(I);
    if (result == instrToNum_.end())
      return -1;
    return (int64_t)result->second;
  }

  /// \returns the number of the instruction, or -1 if the instruction is not
  /// numbered.
  int64_t getInstrNumber(InstrIterator IT) const { return getInstrNumber(*IT); }

  /// \returns the instruction with a given number.
  InstrIterator getInstr(size_t instrNumber) const {
    assert(instrNumber / MAX_SLOT < numToInstr_.size());
    return numToInstr_[instrNumber / MAX_SLOT];
  }
};

#ifndef NDEBUG
llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const Interval &I) {
  os << I.str();
  return os;
}
#endif

/// Set of intervals for a single memory buffer. If there is only one write into
/// a memory buffer, it would contain a single interval. If there are multiple
/// writes, it would contain multiple live intervals, one per write.
using Intervals = llvm::SmallVector<Interval, 4>;
/// Maping from a memory buffer to its live intervals.
using LiveIntervalsMap = std::unordered_map<Value *, Intervals>;
/// Set of instructions.
using Instructions = std::unordered_set<Instruction *>;

/// Hoists Dealloc instructions right after their last use.
static void hoistDealloc(IRFunction &M) {
  // Maps activation instructions to their last non-dealloc user.
  std::unordered_map<Value *, InstrIterator> lastUser;
  // Dealloc instructions in the current function.
  llvm::SmallVector<InstrIterator, 64> deallocs;
  auto &instrs = M.getInstrs();

  // Record the last use of each dealloc.
  for (auto it = instrs.begin(), e = instrs.end(); it != e; ++it) {
    if (isa<DeallocActivationInst>(*it)) {
      // Collect dealloc instructions.
      deallocs.push_back(it);
      continue;
    }

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

  // Now that we've found the last user of each allocated buffer, we can hoist
  // the dealloc instructions.
  for (auto it = deallocs.begin(), e = deallocs.end(); it != e;
       /* increment below */) {
    auto curr = *it;
    ++it;
    auto *da = dyn_cast<DeallocActivationInst>(*curr);
    if (!da) {
      continue;
    }

    auto *alloc = cast<AllocActivationInst>(getOrigin(da->getSrc()));
    auto where = lastUser[alloc];
    if (std::next(where) == curr) {
      // No need to move the instruction, because the last use was
      // right before the deallocation.
      continue;
    }
    ++where;
    M.moveInstruction(where, curr);
  }
}

/// Sink Alloc instructions right before their first use.
static void sinkAllocas(IRFunction &M) {
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
      auto aa = dyn_cast<AllocActivationInst>(getOrigin(op));
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

/// Sink tensorview instructions right before their first use.
static void sinkTensorViews(IRFunction &M) {
  // A set of tensorviews to reschedule.
  std::unordered_set<TensorViewInst *> tensorviews;
  auto &instrs = M.getInstrs();

  // Remove all of the tensorviews.
  for (auto it = instrs.begin(), e = instrs.end(); it != e;) {
    auto curr = it;
    auto *tv = dyn_cast<TensorViewInst>(*curr);
    if (!tv) {
      ++it;
      continue;
    }

    // Ignore tensorviews that are unused.
    if (!tv->hasUsers()) {
      ++it;
      continue;
    }

    tensorviews.insert(tv);
    it = M.removeInstruction(curr);
  }

  // Place all of the tensorviews in the right place:
  for (auto it = instrs.begin(), e = instrs.end(); it != e;) {
    // Holds the next value for the iterator.
    auto nextIt = instrs.end();
    for (int i = 0, e = (*it)->getNumOperands(); i < e; i++) {
      auto op = (*it)->getOperand(i).first;
      auto tv = dyn_cast<TensorViewInst>(op);
      if (!tv) {
        continue;
      }
      auto TV = tensorviews.find(tv);
      if (TV == tensorviews.end()) {
        continue;
      }
      auto inserted = M.insertInstruction(it, tv);
      tensorviews.erase(TV);
      if (tensorviews.empty())
        return;
      if (nextIt == instrs.end()) {
        // Remember and re-scan the first inserted instruction as it may use
        // another tensor_view.
        nextIt = inserted;
      }
    }
    // If no insertions were made, move to the next instruction.
    if (nextIt == instrs.end()) {
      nextIt = ++it;
    }
    it = nextIt;
  }

  assert(tensorviews.empty() && "Forgot to insert some tensorviews!");
}

/// Delete alloc instructions that have no readers or writers.
static void deleteDeadAllocs(IRFunction &M) {
  auto &instrs = M.getInstrs();

  llvm::SmallVector<Instruction *, 16> erasedInstructions{};

  // Remove all unused tensorviews.
  for (auto &I : instrs) {
    if (isa<TensorViewInst>(I) && I->getNumUsers() == 0) {
      erasedInstructions.push_back(I);
    }
  }

  for (auto I : erasedInstructions) {
    M.eraseInstruction(I);
  }

  erasedInstructions.clear();

  // Remove all of the DeallocActivationInst that close unused allocs.
  for (auto &I : instrs) {
    const auto *DA = dyn_cast<const DeallocActivationInst>(I);
    if (DA && DA->getAlloc()->getNumUsers() < 2) {
      erasedInstructions.push_back(I);
    }
  }

  for (auto I : erasedInstructions) {
    M.eraseInstruction(I);
  }
  erasedInstructions.clear();

  // Remove the unused allocs.
  for (auto &I : instrs) {
    if (isa<const AllocActivationInst>(I) && I->getNumUsers() < 2) {
      erasedInstructions.push_back(I);
    }
  }

  for (auto I : erasedInstructions) {
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

/// Marks non-mutable weights as constants.
void makeWeightsConst(IRFunction &M) {
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
static void LLVM_ATTRIBUTE_UNUSED dump(IRFunction &M,
                                       LiveIntervalsMap &intervalsMap) {
  llvm::outs() << "\nDumping live intervals map:\n";
  for (const auto &I : intervalsMap) {
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
static void calculateLiveIntervals(IRFunction &M, LiveIntervalsMap &liveness) {
  assert(liveness.empty() &&
         "This function should be called with empty liveness map");
  auto &instrs = M.getInstrs();
  unsigned instIdx = 0;

  // Compute the [start..end) intervals for each alloc activation in our basic
  // block. Notice that we ignore Dealloc instructions in our analysis.
  for (auto it = instrs.begin(), e = instrs.end(); it != e;
       ++it, instIdx += LiveIntervalsInstructionNumbering::MAX_SLOT) {
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

    auto instOperands = I->getOperands();
    llvm::SmallVector<Instruction::Operand, 8> sortedOperands(
        instOperands.begin(), instOperands.end());

    // Sort operands so that:
    // - all operands referencing the same Value are grouped together.
    // - operands related to the same Value are always in the following
    // order: In, InOut, Out.
    //
    // This ordering ensures that we process reads before writes.
    std::sort(sortedOperands.begin(), sortedOperands.end());

    for (int i = 0, e = sortedOperands.size(); i < e; i++) {
      auto op = sortedOperands[i].first;
      auto opKind = sortedOperands[i].second;
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
        opIdx =
            LiveIntervalsInstructionNumbering::getInstrWriteSlotNumber(instIdx);
      } else {
        opIdx =
            LiveIntervalsInstructionNumbering::getInstrReadSlotNumber(instIdx);
      }

      auto found = liveness.find(loc);
      if (found == liveness.end()) {
        // Create a new interval.
        liveness[loc].push_back(Interval(opIdx, opIdx + 1));
        // If it is a first use, it should be either an input variable or
        // a write.
        // FIXME: Remove InOut!
        assert((isa<TensorViewInst>(I) || isa<WeightVar>(opOrigin) ||
                opKind == OperandKind::Out || opKind == OperandKind::InOut) &&
               "First reference inside a live interval should be either an "
               "input variable or a write");
        continue;
      }

      auto &intervals = found->second;
      // Extend the interval but only if current use is not a write or
      // if it is a write, but we have seen a read before.
      if (opKind != OperandKind::Out)
        intervals.back().end_ = opIdx + 1;

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
        intervals.back().sameValue_ = false;

      // No need to create a new interval if it is not a write.
      if (opKind != OperandKind::Out)
        continue;
      opIdx =
          LiveIntervalsInstructionNumbering::getInstrWriteSlotNumber(instIdx);
      // This instruction modifies the memory location.
      // Therefore, end the current active live interval
      // for this memory location and begin a new one.
      intervals.push_back(Interval(opIdx, opIdx + 1));
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
static Intervals::iterator getEnclosingInterval(Intervals &liveIntervals,
                                                size_t instIdx) {
  for (auto I = liveIntervals.begin(), E = liveIntervals.end(); I != E; ++I) {
    if (I->begin_ <= instIdx && instIdx < I->end_)
      return I;
  }
  return liveIntervals.end();
}

/// Returns true if RHS is enclosed inside LHS.
static bool isEnclosedInside(Interval &lhs, Interval &rhs) {
  return lhs.begin_ < rhs.begin_ && rhs.end_ <= lhs.end_;
}

/// \returns true of any intervals from \p Ints overlap with interval \p I.
static bool hasOverlappingIntervals(Intervals &intervals, Interval I) {
  for (const auto &curI : intervals) {
    if (std::max(curI.begin_, I.begin_) < std::min(curI.end_, I.end_))
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
static void replaceAllUsesInsideIntervalWith(
    Value *val, Value *with, Interval &liveInterval, IRFunction &M,
    LiveIntervalsInstructionNumbering &instrNumbering) {
  auto &instrs = M.getInstrs();
  auto valOrigin = getOrigin(val);
  auto withOrigin = getOrigin(with);
  unsigned instIdx = 0;
  IRBuilder B(&M);
  for (auto it = instrNumbering.getInstr(liveInterval.begin_), e = instrs.end();
       it != e && instIdx <= liveInterval.end_; ++it) {
    auto *I = *it;
    if (isa<DeallocActivationInst>(I))
      continue;
    // Ignore any new instructions which were not present as the instruction
    // numbering was performed.
    auto instNum = instrNumbering.getInstrNumber(I);
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
      size_t opIdx = static_cast<size_t>(
          (opKind == OperandKind::In)
              ? LiveIntervalsInstructionNumbering::getInstrReadSlotNumber(
                    instIdx)
              : LiveIntervalsInstructionNumbering::getInstrWriteSlotNumber(
                    instIdx));
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
static void eraseInstructions(IRFunction &M, Instructions &erasedInstructions) {
  for (auto it : erasedInstructions) {
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
static void
reuseBufferInsideInterval(Value *oldBuffer, Value *newBuffer,
                          Interval oldInterval, IRFunction &M,
                          LiveIntervalsInstructionNumbering &instrNumbering,
                          Intervals &oldIntervals, Intervals &newIntervals) {
  DEBUG(llvm::dbgs() << "\n\nReuse buffers: use buffer of "
                     << newBuffer->getName() << " as a buffer for "
                     << oldBuffer->getName() << "\n"
                     << "in live interval " << oldInterval << "\n");
  // Replace src by dest.
  replaceAllUsesInsideIntervalWith(oldBuffer, newBuffer, oldInterval, M,
                                   instrNumbering);
  // srcInterval does not belong to srcIntervals anymore. It should belong
  // to destIntervals now.
  moveInterval(oldIntervals, newIntervals, oldInterval);
}

/// A helper class for performing a sharing of buffers used by a given
/// instruction.
class BufferSharingOptimizer {
  /// Current function.
  IRFunction &M_;
  /// The instruction numbering to be used.
  LiveIntervalsInstructionNumbering &instrNumbering_;
  /// Current instruction.
  Instruction *instr_;
  /// The number of the current instruction.
  size_t instrIdx_;
  /// The source argument.
  Value *src_;
  /// The destination argument.
  Value *dest_;
  /// The origin of the source argument.
  Value *srcOrigin_;
  /// The origin of the destination argument.
  Value *destOrigin_;

  /// List of live intervals for the source buffer.
  Intervals &srcIntervals_;
  /// List of live intervals for the destination buffer.
  Intervals &destIntervals_;
  /// The live interval of the source buffer, which covers the current
  /// instruction.
  Interval *srcInterval_;
  /// The live interval of the destination buffer, which covers the current
  /// instruction.
  Interval *destInterval_;

  /// Pick the buffer that can be reused. To make a decision, check
  /// which intervals intersect with each other. In most cases, the buffers
  /// can be combined if their live intervals do not overlap.
  ///
  /// \returns the buffer that can be
  /// reused, or nullptr if none of the buffers can be reused.
  Value *getBufferToBeReused() {
    // Do not try to combine observables.
    if (isObservable(destOrigin_) && isObservable(srcOrigin_))
      return nullptr;

    // Check if dest or src live interval is the last live interval of
    // an observable memory location.
    bool isDestLastIntervalOfObservable =
        isObservable(destOrigin_) && *destInterval_ == destIntervals_.back();
    bool isSrcLastIntervalOfObservable =
        isObservable(srcOrigin_) && *srcInterval_ == srcIntervals_.back();

    // A value X cannot reuse the buffer of another value Y,
    // if the live interval of X overlaps with any live intervals of Y.
    // The only exception is the copy instruction, where the live interval
    // of the destination may be enclosed into a live interval of the source,
    // because they have the same value.
    bool canCopyPropagate = isa<CopyInst>(instr_) &&
                            isEnclosedInside(*srcInterval_, *destInterval_) &&
                            srcInterval_->sameValue_;
    // If dest interval overlaps with any srcIntervals, it cannot be replaced.
    bool destIntvalCannotBeReplaced =
        !canCopyPropagate &&
        hasOverlappingIntervals(srcIntervals_, *destInterval_);
    // If src interval overlaps with any dest Intervals, it cannot be replaced.
    bool srcIntervalCannotBeReplaced =
        hasOverlappingIntervals(destIntervals_, *srcInterval_);

    if (!isDestLastIntervalOfObservable && !isSrcLastIntervalOfObservable &&
        !destIntvalCannotBeReplaced && !srcIntervalCannotBeReplaced) {
      // There are no restrictions and intervals can be combined on any
      // order. Try to use a heuristic to pick the right way to combine
      // them.

      // Try to reuse the interval of an observable memory location, because
      // it does not increase the memory usage.
      // TODO: If it would introduce a last write into an observable, do not
      // do it.
      if (isObservable(srcOrigin_) && !isObservable(destOrigin_)) {
        // Use src buffer for dest.
        return srcOrigin_;
      }

      if (isObservable(destOrigin_) && !isObservable(srcOrigin_)) {
        // Use dest buffer for src.
        return destOrigin_;
      }

      // Avoid sharing a buffer if there is a single
      // live interval in the interval list. After replacement
      // this whole buffer can be eliminated.
      if (srcIntervals_.size() == 1 && destIntervals_.size() != 1) {
        // Use dest buffer for src.
        return destOrigin_;
      }

      if (destIntervals_.size() == 1 && srcIntervals_.size() != 1) {
        // Use src buffer for dest.
        return srcOrigin_;
      }

      // Just use src buffer for dest by default.
      return srcOrigin_;
    }

    // Try to check if buffers can be shared by using
    // src instead of dest inside the live interval of dest.
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
      return destOrigin_;
    }

    return srcOrigin_;
  }

public:
  /// Initialize the state of the shared buffers optimizer.
  BufferSharingOptimizer(IRFunction &M, LiveIntervalsMap &intervalsMap,
                         LiveIntervalsInstructionNumbering &instrNumbering,
                         Instruction *instr, size_t instrIdx, Value *dest,
                         Value *src)
      : M_(M), instrNumbering_(instrNumbering), instr_(instr),
        instrIdx_(instrIdx), src_(src), dest_(dest), srcOrigin_(getOrigin(src)),
        destOrigin_(getOrigin(dest)), srcIntervals_(intervalsMap[srcOrigin_]),
        destIntervals_(intervalsMap[destOrigin_]) {
    // Bail if information about live intervals is not known.
    assert(!srcIntervals_.empty() &&
           "Set of live intervals for a memory buffer cannot be empty");
    assert(!destIntervals_.empty() &&
           "Set of live intervals for a memory buffer cannot be empty");
    // Find the Src live interval that encloses the current instruction.
    auto srcIntervalIt = getEnclosingInterval(
        srcIntervals_,
        LiveIntervalsInstructionNumbering::getInstrReadSlotNumber(instrIdx_));
    assert(srcIntervalIt != srcIntervals_.end() &&
           "Cannot share buffers: cannot "
           "find enclosing src interval");

    // Find the Dest live interval that encloses the current instruction.
    auto destIntervalIt = getEnclosingInterval(
        destIntervals_,
        LiveIntervalsInstructionNumbering::getInstrWriteSlotNumber(instrIdx_));
    assert(destIntervalIt != destIntervals_.end() &&
           "Cannot share buffers: cannot "
           "find enclosing src interval");

    // Remember the found src and dest intervals.
    srcInterval_ = srcIntervalIt;
    destInterval_ = destIntervalIt;
  }

  /// Try to share buffers used by src and dest.
  /// \returns true, if it was possible to reuse buffers.
  bool tryToShareBuffers() {
    // Pick the buffer that can be reused. It can be either the src or the dest
    // buffer.
    auto *bufferToReuse = getBufferToBeReused();
    // Bail if none of the buffers can be reused.
    if (!bufferToReuse)
      return false;

    // TODO: May be disallow usage of dest interval for src?
    // This would avoid extending dest lifetime to start earlier.
    // But it would also miss some opportunities for sharing buffers.
    // if (bufferToReuse == destOrigin)
    //   continue;

    // Check if it is the destination buffer that can be reused.
    if (bufferToReuse == destOrigin_) {
      // This operation may extend the lifetime of dest's buffer.
      // shareBuffers will fix it once it's done.
      reuseBufferInsideInterval(srcOrigin_, dest_, *srcInterval_, M_,
                                instrNumbering_, srcIntervals_, destIntervals_);
      return true;
    }

    // Source buffer can be reused.
    reuseBufferInsideInterval(destOrigin_, src_, *destInterval_, M_,
                              instrNumbering_, destIntervals_, srcIntervals_);
    return true;
  }
};

/// Tries to share a buffer for two operands of the same instruction.
/// An operand X cannot reuse the buffer of another operand Y,
/// if the live interval of X overlaps with any live intervals of Y.
/// The only exception is the copy instruction, where the live interval
/// of the X may be enclosed into a live interval of Y because they have
/// the same value after the copy instruction.
static void
tryToShareBuffersForInstr(LiveIntervalsMap &intervalsMap,
                          LiveIntervalsInstructionNumbering &instrNumbering,
                          Instruction *I, unsigned instIdx) {
  IRFunction &M = *I->getParent();
  // Consider all pair of operands. Check if their respective buffers can be
  // reused in principle.
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
      if (destOp.first->getType() != srcOp.first->getType()) {
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
      // The buffers can be reused in principle, thus try to share the buffers.
      BufferSharingOptimizer opt(M, intervalsMap, instrNumbering, I, instIdx,
                                 dest, src);
      if (opt.tryToShareBuffers())
        return;
    }
  }
}

/// Sharing of buffers
///
/// The purpose of this optimization is to reduce the memory usage by
/// reusing memory buffers as much as possible.
///
/// The overall idea is that it is fine to combine storage for two live
/// intervals if they do not overlap and if for at least one of them the writes
/// do not need to be observable. Typically, two live intervals are considred as
/// candidates for sharing if they occur in the same instruction, but it is not
/// strictly necessary.
static void shareBuffers(IRFunction &M) {
  Instructions erasedInstructions;
  // Build a list of live intervals for each memory location
  // which is either a WeightVar or a an Allocation.
  LiveIntervalsMap intervalsMap;
  calculateLiveIntervals(M, intervalsMap);
  LiveIntervalsInstructionNumbering instrNumbering(M);

  // Get the source of the copy. This memory location may have been
  // modified by any instruction that used it as an @out or @inout
  // parameter.
  auto &instrs = M.getInstrs();

  // For each instruction, in reverse order.
  for (auto it = instrs.rbegin(), e = instrs.rend(); it != e; ++it) {
    Instruction *I = *it;
    auto instIdx = instrNumbering.getInstrNumber(I);
    if (instIdx < 0)
      continue;
    // Try to reuse the operand memory buffers.
    tryToShareBuffersForInstr(intervalsMap, instrNumbering, I, instIdx);
  }

  // Fix eventual issues with allocs and deallocs that shareBuffers may
  // introduce by extending live interval lifetimes.
  hoistDealloc(M);
  sinkAllocas(M);
  // Fix eventual issues with tensorviews that shareBuffers may
  // introduce by extending live interval lifetimes.
  sinkTensorViews(M);
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
static void eliminateDeadStores(IRFunction &M) {
  auto &instrs = M.getInstrs();
  // Instructions to be erased.
  Instructions erasedInstructions;
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
    size_t numMutatedOperands = 0;
    size_t numNonReadMutatedOperands = 0;
    // Process all operand writes.
    for (const auto &Op : I->getOperands()) {
      auto OpOrigin = getOrigin(Op.first);
      auto OpKind = Op.second;
      auto &State = memoryState[OpOrigin];
      if (OpKind != OperandKind::In) {
        numMutatedOperands++;
        // If it a write that was not read and it is not a last write into
        // a WeightVar (i.e. an observable effect), then is can be eliminated.
        // If there are multiple writes in this instruction, all of them
        // should satisfy this property for the instruction to be removed.
        if (!State.lastSeenRead_) {
          numNonReadMutatedOperands++;
        }
        State.lastSeenWrite_ = I;
        State.lastSeenRead_ = nullptr;
      }
    }

    // It is safe to remove an instruction if all of its mutated operands
    // are not read afterwards.
    if (numMutatedOperands > 0 &&
        numMutatedOperands == numNonReadMutatedOperands) {
      erasedInstructions.insert(I);
      // Do not process any reads of operands, because
      // this instruction will be eliminated.
      continue;
    }

    // Process all operand reads and the predicate.
    for (const auto &Op : I->getOperands()) {
      auto opOrigin = getOrigin(Op.first);
      auto opKind = Op.second;
      auto &state = memoryState[opOrigin];
      if (opKind != OperandKind::Out) {
        state.lastSeenRead_ = I;
      }
    }
  }

  eraseInstructions(M, erasedInstructions);
}

/// Instrument the code to make it easier to debug issues.
/// Add dumping of inputs before each instruction and
/// dumping of outputs after each instruction.
/// For each input/output tensor its name and its value are dumped.
static void performDebugInstrumentation(IRFunction &M) {
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
void performPeepholeOptimizations(IRFunction &M) {
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

      B.createPoolMaxInst(PMI->getName(), PMI->getDest(), PMI->getSrc(),
                          PMI->getKernel(), PMI->getStride(), PMI->getPad());
      it = M.moveInstruction(cur, instrs.back());
      M.eraseInstruction(cur);
      continue;
    }

    // tranpose dest, splat (src), ... -> copy dest, tensorview (splat (src))
    // This is safe, because transpose of a splat does not change any elements.
    // It changes only types.
    if (auto *TI = dyn_cast<TransposeInst>(I)) {
      auto src = TI->getSrc();
      auto dest = TI->getDest();
      if (auto W = getSingleWriter(src)) {
        if (isa<SplatInst>(W)) {
          if (src->getType() != dest->getType()) {
            auto *TVI =
                B.createTensorViewInst(TI->getName(), src, dest->getType());
            M.moveInstruction(cur, instrs.back());
            src = TVI;
          }
          B.createCopyInst(TI->getName(), TI->getDest(), src);
          it = M.moveInstruction(cur, instrs.back());
          M.eraseInstruction(cur);
          continue;
        }
      }
    }

    // Convert element_max instruction into a canonical form,
    // where the splat (i.e. the constant) argument is the last one.
    if (auto *EM = dyn_cast<ElementMaxInst>(I)) {
      auto *lhs = EM->getLHS();
      auto *rhs = EM->getRHS();
      auto *wlhs = getSingleWriter(lhs);
      if (!wlhs)
        continue;
      if (!isa<SplatInst>(wlhs))
        continue;
      // If RHS is a splat already, there is nothing to do.
      auto *wrhs = getSingleWriter(rhs);
      if (wrhs && isa<SplatInst>(wrhs))
        continue;
      B.createElementMaxInst(EM->getName(), EM->getDest(), rhs, lhs);
      it = M.moveInstruction(cur, instrs.back());
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

/// Perform optimizations on the IR representation.
void glow::optimize(IRFunction &M, CompilationMode mode) {
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
