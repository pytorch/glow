// Copyright 2017 Facebook Inc.  All Rights Reserved.
#define DEBUG_TYPE "ir-optimizer"

#include "glow/Graph/Graph.h"
#include "glow/IR/IR.h"
#include "glow/IR/IRBuilder.h"
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

using namespace glow;

using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;

/// A live interval is represented as [begin, end).
using Interval = std::pair<unsigned, unsigned>;
using Intervals = std::list<Interval>;
using LivenessMap = std::unordered_map<Value *, Interval>;
using LiveIntervalsMap = std::unordered_map<Value *, Intervals>;
/// Set of instruction numbers.
using InstructionNumbers = std::unordered_set<size_t>;

static AllocActivationInst *getAllocationOrigin(Value *V) {
  if (auto *AI = dyn_cast<AllocActivationInst>(V))
    return AI;
  if (auto *TVI = dyn_cast<TensorViewInst>(V))
    return getAllocationOrigin(TVI->getSrc());
  return nullptr;
}

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
    auto *da = dyn_cast<DeallocActivationInst>(*curr);
    if (!da) {
      ++it;
      continue;
    }

    auto *alloc = cast<AllocActivationInst>(da->getOperand(0).first);

    it = M.removeInstruction(curr);
    auto &where = lastUser[alloc];
    where++;
    M.insertInstruction(where, da);
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
    }
  }

  assert(allocs.empty() && "Forgot to insert some allocas!");
}

/// Delete alloc instructions that have no readers or writers.
static void deleteDeadAllocs(Module &M) {
  auto &instrs = M.getInstrs();

  llvm::SmallVector<Instruction *, 16> ErasedInstructions{};
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
  // We use a vector here because changing the operands of the user changes the
  // uselist, and this invalidates the iterator.
  llvm::SmallVector<Use, 6> usersVec(users.begin(), users.end());
  for (auto &U : usersVec) {
    // Ignore dealloc instrs.
    if (isa<DeallocActivationInst>(U.get())) {
      continue;
    }

    U.setOperand(with);
  }
}

/// Optimize the input/output buffer for the instruction \p I, based on the
/// liveness information in \p liveBuffers.
static void
tryToShareBuffersForInstr(const std::unordered_set<Value *> &liveBuffers,
                          Instruction *I) {
  // At this point <out> variables are marked as dead, and <in> variables have
  // not been marked alive yet.

  for (unsigned first = 0, e = I->getNumOperands(); first < e; first++) {
    for (unsigned second = first + 1; second < e; second++) {
      auto destOp = I->getOperand(first);
      auto srcOp = I->getOperand(second);
      Value *dest = getAllocationOrigin(destOp.first);
      Value *src = getAllocationOrigin(srcOp.first);
      if (!dest)
        dest = destOp.first;
      if (!src)
        src = srcOp.first;
      // Operands must be different, but of the same type.
      if (dest->getType() != src->getType() || dest == src) {
        continue;
      }

      if (!Instruction::isInplaceOp(I, first, second)) {
        continue;
      }

      // If both the src and the dest operands are dead, this means that we can
      // reuse the buffer storage!
      if (!liveBuffers.count(dest) && !liveBuffers.count(src)) {
        replaceAllNonDeallocUsersWith(dest, src);
        return;
      }
    }
  }
}

static void shareBuffers(Module &M) {
  auto &instrs = M.getInstrs();

  // The live set stores allocations that are known to contain information
  // that's used by some user. These buffers can't be clobbered.
  std::unordered_set<Value *> liveBuffers;

  // All of the weights are alive. We can't touch them.
  for (auto *W : M.getWeights()) {
    liveBuffers.insert(W);
  }

  // Output buffers of the current instruction.
  std::unordered_set<Value *> outBuffers;

  // For each instruction, in reverse order.
  for (auto it = instrs.rbegin(), e = instrs.rend(); it != e; ++it) {
    Instruction *I = *it;

    outBuffers.clear();

    // Remove <out> dependencies from the live set, because this instruction
    // writes into them. This means that the buffer is unused before the write
    // point.
    for (unsigned op = 0, ope = I->getNumOperands(); op < ope; op++) {
      auto O = I->getOperand(op);
      // Find the origin of the operand.
      Value *ai = getAllocationOrigin(O.first);
      if (!ai) {
        continue;
      }

      // <Out> dependency means that the buffer is being killed. Remove from the
      // live list.
      if (O.second == OperandKind::Out) {
        auto it = liveBuffers.find(ai);
        if (it != liveBuffers.end()) {
          liveBuffers.erase(it);
          outBuffers.insert(ai);
        }
        continue;
      }
      // The <InOut> means that the value of the buffer is being consumed,
      // which means that it is alive. Add to the live set.
      if (ai && O.second == OperandKind::InOut) {
        liveBuffers.insert(ai);
      }
      // The <In> use of a buffer that is also used as an <Out> means that the
      // value of the buffer is being consumed, which means that it is alive.
      // Add to the live set.
      if (ai && O.second == OperandKind::In && outBuffers.count(ai) > 0) {
        liveBuffers.insert(ai);
      }
    }

    // Now that we've calculated the liveness for the exact location of the
    // buffer we can try to reuse the operand memory buffers.
    tryToShareBuffersForInstr(liveBuffers, I);

    // Now, before we are moving to the next instruction, insert the input
    // operand-buffers into the live set, because this instruction needs them
    // alive.
    for (unsigned op = 0, ope = I->getNumOperands(); op < ope; op++) {
      auto O = I->getOperand(op);
      auto ai = getAllocationOrigin(O.first);
      if (!ai) {
        continue;
      }

      // The <In> means that the value of the buffer is being consumed,
      // which means that it is alive. Add to the live set.
      if (O.second != OperandKind::Out) {
        liveBuffers.insert(ai);
      }
    }
  }
}

/// \returns the pointer to the single writer that writes into this value, or
/// nullptr if the number of writers is not exactly one.
static Instruction *getSingleWriter(Value *V) {
  Instruction *singleUser = nullptr;
  for (auto U : V->getUsers()) {
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
    for (auto &U : W->getUsers()) {
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
  for (auto &I : IntervalsMap) {
    llvm::outs() << "\nValue " << I.first->getName();
    llvm::outs() << "\n";
    for (auto &Interval : I.second) {
      llvm::outs() << " (" << Interval.first << ", " << Interval.second << ")";
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
  for (auto it = instrs.begin(), e = instrs.end(); it != e; ++it, ++instIdx) {
    // Ignore deallocations in our liveness calculation.
    if (isa<DeallocActivationInst>(*it)) {
      continue;
    }

    auto InstOperands = (*it)->getOperands();
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
      Value *loc = dyn_cast<AllocActivationInst>(op);
      if (!loc) {
        loc = dyn_cast<WeightVar>(op);
        // No need to track constants. They are always read-only.
        if (loc && dyn_cast<WeightVar>(op)->getMutability() ==
                       WeightVar::MutabilityKind::Constant)
          continue;
      }
      // Bail if the operand is not an AllocActivationInst or a WeightVar.
      if (!loc) {
        continue;
      }

      auto I = liveness.find(loc);
      if (I == liveness.end()) {
        // Create a new interval.
        liveness[loc].push_back({instIdx, instIdx});
        // If it is a first use, it should be either an input variable or
        // a write.
        // FIXME: Remove InOut!
        assert((isa<TensorViewInst>(*it) || isa<WeightVar>(op) ||
                opKind == OperandKind::Out || opKind == OperandKind::InOut) &&
               "First reference inside a live interval should be either an "
               "input variable or a write");
        continue;
      }

      auto &Intervals = I->second;
      // Extend the interval but only if current use is not a write or
      // if it is a write, but we have seen a read before.
      if (opKind != OperandKind::Out ||
          Intervals.back().second != Intervals.back().first)
        Intervals.back().second = instIdx;

      // No need to create a new interval unless it is a write.
      if (opKind == OperandKind::In || opKind == OperandKind::InOut)
        continue;

      // This instruction modifies the memory location.
      // Therefore, end the current active live interval
      // for this memory location and begin a new one.
      Intervals.push_back({instIdx, instIdx});
    }
  }

  for (auto &Entry : liveness) {
    auto *ML = Entry.first;
    auto &IL = Entry.second;
    if (isa<WeightVar>(ML)) {
      assert(!IL.empty() && "Live interval list cannot be empty");
      // Extend the last interval till the end of the program
      // to express that all mutable weights are used outside.
      IL.back().second = instIdx;
    }
  }
}

/// Provided a set of intervals, return the interval covering
/// a given instruction.
static Intervals::iterator getEnclosingInterval(Intervals &LiveIntervals,
                                                size_t instIdx) {
  for (auto I = LiveIntervals.begin(), E = LiveIntervals.end(); I != E; ++I) {
    if (I->first <= instIdx && instIdx <= I->second)
      return I;
  }
  return LiveIntervals.end();
}

/// Returns true if RHS is enclosed inside LHS.
static bool isEnclosedInside(Interval &LHS, Interval &RHS) {
  return LHS.first < RHS.first && RHS.second <= LHS.second;
}

static void replaceAllUsesWith(Value *val, Value *with, Interval &I, Module &M,
                               std::vector<Instruction *> &ChangedInstrs) {
  auto &instrs = M.getInstrs();
  size_t instIdx = 0;
  for (auto it = instrs.begin(), e = instrs.end();
       it != e && instIdx <= I.second; ++instIdx, ++it) {
    if (instIdx < I.first)
      continue;
    // This is an instruction inside the interval.
    for (int i = 0, e = (*it)->getNumOperands(); i < e; i++) {
      auto op = (*it)->getOperand(i).first;
      auto kind = (*it)->getOperand(i).second;
      if (op != val)
        continue;
      if (instIdx == I.first && kind != OperandKind::Out)
        continue;
      DEBUG(llvm::outs() << "Replacing inside instruction " << instIdx << "\n");
      // Replace the old value by the new value.
      (*it)->setOperand(i, with);
      ChangedInstrs.push_back(*it);
    }
  }
}

static void eraseInstructions(Module &M,
                              InstructionNumbers &ErasedInstructions) {
  auto &instrs = M.getInstrs();
  size_t instIdx = 0;
  for (auto it = instrs.begin(), e = instrs.end(); it != e; ++instIdx) {
    if (ErasedInstructions.count(instIdx) == 0) {
      ++it;
      continue;
    }
    it = M.eraseInstruction(it);
  }
}

/// Perform a copy propagation.
void copyPropagation(Module &M) {
  auto &instrs = M.getInstrs();

  InstructionNumbers ErasedInstructions;
  // Build a list of live intervals for each memory location
  // which is either a WeightVar or a an Allocation.
  LiveIntervalsMap IntervalsMap;
  calculateLiveIntervals(M, IntervalsMap);

  size_t instIdx = 0;
  // Go over instructions and loop for copy instructions.
  for (auto it = instrs.begin(), e = instrs.end(); it != e; ++instIdx) {
    auto curr = it;
    auto *ci = dyn_cast<CopyInst>(*curr);
    // We need only copy instructions.
    if (!ci) {
      ++it;
      continue;
    }

    // Get the source of the copy. This memory location may have been
    // modified by any instruction that used it as an @out or @inout
    // parameter.
    auto *Src = ci->getSrc();
    auto *Dest = ci->getDest();
    assert(Src->getType() == Dest->getType() &&
           "Both src and dest of copy should have the same type");
    DEBUG(llvm::dbgs() << "Instruction " << instIdx << ": Found a copy from "
                       << Src->getName() << " to " << Dest->getName() << ":\n";
          ci->dump(llvm::dbgs()); llvm::dbgs() << "\n");

    // We plan to replace the assignments to Src by assignments
    // to Dest and replace all uses of Src to use Dest to get rid of the copy.
    // But before we can do it, we need to check some preconditions.

    // Check if writes to Src are allowed to be replaced by writes
    // to Dest.
    if (auto *WV = dyn_cast<WeightVar>(Src)) {
      // Writes into an output variable should not be transformed,
      // because it would change the observable effect of the write.
      // So, bail if:
      // - Src is a mutable WeightVar or
      // - Src it is a WeightVar constant, but Dest is assigned multiple times.
      // - or Dest is assigned once, but it is an output variable and thus
      //   assignments to it cannot be removed.
      if (WV->getMutability() == WeightVar::MutabilityKind::Mutable ||
          getSingleWriter(Dest) != ci ||
          Dest->getKind() == Kinded::Kind::WeightVarKind) {
        DEBUG(llvm::outs() << "Cannot copy propagate if src is a WeightVar\n");
        ++it;
        continue;
      }
      // There is only one write into Dest and it is this copy instruction.
      // Therefore it is safe to replace all uses of Dest by Src.
      replaceAllNonDeallocUsersWith(Dest, Src);
      ErasedInstructions.insert(instIdx);
      DEBUG(llvm::outs() << "Can replace this copy by forward "
                            "propagating its value\n");
      ++it;
      continue;
    }

    auto &SrcIntervals = IntervalsMap[Src];
    auto &DestIntervals = IntervalsMap[Dest];
    // Bail if information about live intervals is not known.
    if (SrcIntervals.empty() || DestIntervals.empty()) {
      DEBUG(llvm::outs() << "Cannot copy propagate because "
                            "cannot find live intervals\n");
      ++it;
      continue;
    }

    // Find the Src live interval that encloses instIdx
    auto SrcInterval = getEnclosingInterval(SrcIntervals, instIdx);
    if (SrcInterval == SrcIntervals.end()) {
      DEBUG(llvm::outs() << "Cannot copy propagate: cannot "
                            "find enclosing src interval\n";
            llvm::outs() << "instruction idx = " << instIdx << "\n");
      ++it;
      continue;
    }

    // Find the Dest live interval that encloses instIdx.
    auto DestInterval = getEnclosingInterval(DestIntervals, instIdx);
    if (DestInterval == DestIntervals.end()) {
      DEBUG(llvm::outs() << "Cannot copy propagate: cannot "
                            "find enclosing dest interval\n");
      ++it;
      continue;
    }

    // If the Src interval ends before the Dest interval starts,
    // it means that the copy instruction is the last use of Src.
    // After this copy, Dest would be equal to Src.
    // Thus, it is safe to replace all uses of Src inside the Src
    // interval by Dest. In particular, the instruction that
    // initializes Src will now initialize Dest.
    // This would have the effect of shrinking Src's lifetime
    // and extending the Dest's lifetime.
    //
    // So, basically:
    // src <- val
    // use1_src
    // dest <- src
    // use2_dest
    //
    // is transformed into:
    //
    // dest <- val
    // use1_dest
    // use2_dest

    // Another possible case is that Dest interval is enclosed
    // into Src interval.
    //
    // In this case, we get:
    // src <- val
    // use1_src
    // dest <- src
    // use2_src
    // use3_dest // Last use of dest
    // use4_src
    //
    // is transformed into:
    //
    // dest <- val
    // use1_dest
    // use2_dest
    // user3_dest
    // use4_dest

    // Check if SrcInterval ends before DestInterval starts or
    // that DestInterval is enclosed inside the SrcInterval.
    bool canPropagate = SrcInterval->second <= DestInterval->first ||
                        isEnclosedInside(*SrcInterval, *DestInterval);
    if (!canPropagate) {
      DEBUG(llvm::outs() << "Cannot copy propagate: "
                         << "DstInterval"
                         << "(" << DestInterval->first << ","
                         << DestInterval->second << ")"
                         << " is not enclosed inside SrcInterval"
                         << "(" << SrcInterval->first << ","
                         << SrcInterval->second << ")"
                         << "\n");
      ++it;
      continue;
    }

    // It is safe to replace all references to Src inside SrcInterval
    // by references to Dest.
    std::vector<Instruction *> ChangedInstrs;
    replaceAllUsesWith(Src, Dest, *SrcInterval, M, ChangedInstrs);
    /// TODO: Do we need to update the information about Src and Dest in the
    /// live intervals map?
    assert(!ChangedInstrs.empty() &&
           "Some instructions should have been changed");
    DEBUG(llvm::dbgs() << "Can replace this copy by producing instruction:\n";
          ChangedInstrs[0]->dump(llvm::dbgs()); llvm::dbgs() << "\n");
    assert(ci->getSrc() == ci->getDest() && "Src and Dest of a copy "
                                            "instruction should be the same "
                                            "after copy propagation");
    // Remove the obsolete copy instruction.
    ErasedInstructions.insert(instIdx);
    ++it;
  }

  // Erase instructions.
  eraseInstructions(M, ErasedInstructions);
}

/// Get a set of operands updated by an instruction.
static void
getUpdatedOperands(InstrIterator I,
                   llvm::SmallVectorImpl<Instruction::Operand> &Operands) {
  assert(Operands.empty());
  for (int i = 0, e = (*I)->getNumOperands(); i < e; i++) {
    auto Op = (*I)->getOperand(i);
    // Collect only operands that are updated.
    if (Op.second == OperandKind::In)
      continue;
    // Add to the set if it is not present there yet.
    if (std::find(Operands.begin(), Operands.end(), Op) == Operands.end())
      Operands.push_back(Op);
  }
}

/// Dead Store Elimination.
static void eliminateDeadStores(Module &M) {
  InstructionNumbers ErasedInstructions;
  // Build a list of live intervals for each memory location
  // which is either a WeightVar or a an Allocation.
  InstructionNumbering InstrNumbering(M);
  LiveIntervalsMap IntervalsMap;
  calculateLiveIntervals(M, IntervalsMap);

  // For each memory location from the intervals map
  // - Find intervals which whose start and end are the same and the first
  // instruction is a write. It means that they are only written to, but
  // never read.
  // - If this first instruction writes only into the memory location in
  // question and does not write/update any other memory locations,
  // this instruction can be removed.
  for (auto const &Entry : IntervalsMap) {
    auto *ML = Entry.first;
    (void)ML;
    auto &IL = Entry.second;
    for (auto &LI : IL) {
      auto IntervalBegin = LI.first;
      auto IntervalEnd = LI.second;
      if (IntervalBegin != IntervalEnd)
        continue;
      // Get the defining instruction for this interval.
      auto I = InstrNumbering.getInstr(IntervalBegin);
      if (isa<AllocActivationInst>(*I) || isa<DeallocActivationInst>(*I))
        continue;
      // Check which memory locations are being changed by this instruction.
      llvm::SmallVector<Instruction::Operand, 8> Operands;
      getUpdatedOperands(I, Operands);
      if (Operands.size() > 1) {
        // Instruction updates more than one memory location.
        // TODO: If none of those locations are used afterwards, then it is fine
        // to remove it.
        continue;
      }
      DEBUG(llvm::dbgs() << "Found a dead store into " << ML->getName().str()
                         << ": ";
            (*I)->dump(llvm::dbgs()); llvm::dbgs() << "\n";
            llvm::dbgs() << "Operand being updated is: "
                         << Operands[0].first->getName().str() << "\n";
            llvm::dbgs() << "Interval is: "
                         << "(" << IntervalBegin << ", " << IntervalEnd << ")"
                         << "\n");
      assert(Operands[0].first == ML);
      // Only the current memory location is being updated by this instruction.
      ErasedInstructions.insert(IntervalBegin);
    }
  }
  if (!ErasedInstructions.empty()) {
    eraseInstructions(M, ErasedInstructions);
  }
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
    for (auto const &Op : (*it)->getOperands()) {
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

/// Try to use non-caching versions of some IR instructions like PoolMax and
/// SoftMax.
void optimizeUsingNonCachingInstructions(Module &M) {
  auto &instrs = M.getInstrs();
  IRBuilder B(&M);
  for (auto it = instrs.begin(), e = instrs.end(); it != e;) {
    auto cur = it;
    it = std::next(it);
    if (auto *PMI = dyn_cast<PoolMaxWithXYInst>(*cur)) {
      auto *SrcXY = PMI->getSrcXY();
      // Optimize only if the cache is an allocation and
      // it has exactly 2 users: the current instruction and
      // a deallocation.
      if (!isa<AllocActivationInst>(SrcXY) || SrcXY->getNumUsers() != 2)
        continue;

      auto *NewPMI = B.createPoolMaxInst(PMI->getName(), PMI->getDest(),
                                         PMI->getSrc(), PMI->getKernel(),
                                         PMI->getStride(), PMI->getPad());
      M.moveInstruction(cur, NewPMI);
      M.eraseInstruction(cur);
      continue;
    }

    if (auto *SMI = dyn_cast<SoftMaxWithEInst>(*cur)) {
      auto *E = SMI->getE();
      // Optimize only if the cache is an allocation and
      // it has exactly 2 users: the current instruction and
      // a deallocation.
      if (!isa<AllocActivationInst>(E) || E->getNumUsers() != 2)
        continue;

      auto *NewSMI = B.createSoftMaxInst(SMI->getName(), SMI->getDest(),
                                         SMI->getSrc(), SMI->getSelected());
      M.moveInstruction(cur, NewSMI);
      M.eraseInstruction(cur);
      continue;
    }
  }
}

void glow::optimize(Module &M, CompilationMode mode) {
  M.verify();

  optimizeUsingNonCachingInstructions(M);

  // Reuse buffers from previous operations.
  shareBuffers(M);

  // Remove unused allocations.
  deleteDeadAllocs(M);

  // Shorten the lifetime of buffers.
  hoistDealloc(M);

  sinkAllocas(M);

  // Turn read-only weights into constant weights.
  makeWeightsConst(M);

  // Perform copy propagation.
  copyPropagation(M);

  deleteDeadAllocs(M);

  // Perform Dead Store Elimination.
  eliminateDeadStores(M);

  deleteDeadAllocs(M);

  // Perform a debug instrumentation if required.
  performDebugInstrumentation(M);

  M.verify();
}
