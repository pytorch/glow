// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "glow/IR/IR.h"
#include "glow/IR/Instrs.h"
#include "glow/Optimizer/Optimizer.h"

#include "llvm/Support/Casting.h"

#include <unordered_map>
#include <unordered_set>

using namespace glow;

using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;

using Interval = std::pair<unsigned, unsigned>;
using LivenessMap = std::unordered_map<Value *, Interval>;
static void calculateLiveness(Module &M, LivenessMap &liveness) {
  auto &instrs = M.getInstrs();
  unsigned instIdx = 0;

  // Compute the [start..end) intervals for each alloc activation in our basic
  // block. Notice that we ignore Dealloc instructions in our analysis.
  for (auto it = instrs.begin(), e = instrs.end(); it != e; ++it) {
    instIdx++;
    // Ignore deallocations in our liveness calculation.
    if (isa<DeallocActivationInst>(*it)) {
      continue;
    }

    for (int i = 0, e = (*it)->getNumOperands(); i < e; i++) {
      auto op = (*it)->getOperand(i).first;
      auto aa = dyn_cast<AllocActivationInst>(op);
      if (!aa) {
        continue;
      }

      auto I = liveness.find(aa);
      if (I == liveness.end()) {
        // Create a new interval.
        liveness[aa] = {instIdx, instIdx};
        continue;
      }

      // Increase the size of the interval.
      I->second.second = instIdx;
    }
  }
}

/// Hoists Dealloc instructions right after their last use.
static void hoistDealloc(Module &M) {
  using iterator = Module::InstListTy::iterator;
  // Maps activation instructions to their last non-dealloc user.
  std::unordered_map<AllocActivationInst *, iterator> lastUser;
  auto &instrs = M.getInstrs();

  // Record the last use of each dealloc.
  for (auto it = instrs.begin(), e = instrs.end(); it != e; ++it) {
    if (isa<DeallocActivationInst>(*it))
      continue;

    for (int i = 0, e = (*it)->getNumOperands(); i < e; i++) {
      auto op = (*it)->getOperand(i).first;
      if (auto alloc = dyn_cast<AllocActivationInst>(op)) {
        lastUser[alloc] = it;
      }
    }
  }

  // Now that we've found the last user we can hoist the instruction.
  for (auto it = instrs.begin(), e = instrs.end(); it != e;
       /* increment below */) {
    iterator curr = it;
    auto *da = dyn_cast<DeallocActivationInst>(*curr);
    if (!da) {
      ++it;
      continue;
    }

    auto *alloc = cast<AllocActivationInst>(da->getOperand(0).first);

    it = instrs.erase(curr);
    auto &where = lastUser[alloc];
    where++;
    instrs.insert(where, da);
  }
}

/// Sink Alloc instructions right before their first use.
static void sinkAllocas(Module &M) {
  using iterator = Module::InstListTy::iterator;
  /// A list of allocas to reschedule.
  std::unordered_set<AllocActivationInst *> allocs;
  auto &instrs = M.getInstrs();

  // Remove all of the allocas.
  for (auto it = instrs.begin(), e = instrs.end(); it != e;) {
    iterator curr = it;
    auto *aa = dyn_cast<AllocActivationInst>(*curr);
    if (!aa) {
      ++it;
      continue;
    }

    allocs.insert(aa);
    it = instrs.erase(curr);
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
      instrs.insert(it, aa);
    }
  }

  assert(allocs.empty() && "Forgot to insert some allocas!");
}

/// Delete alloc instructions that have no readers or writers.
static void deleteDeadAllocs(Module &M) {
  auto &instrs = M.getInstrs();

  // Remove all of the DeallocActivationInst that close unused allocs.
  instrs.erase(
      std::remove_if(instrs.begin(), instrs.end(),
                     [](const Instruction *I) -> bool {
                       if (const auto *DA =
                               dyn_cast<const DeallocActivationInst>(I)) {
                         return DA->getAlloc()->getNumUsers() < 2;
                       }
                       return false;
                     }),
      instrs.end());

  // Remove the unused allocs.
  instrs.erase(std::remove_if(instrs.begin(), instrs.end(),
                              [](const Instruction *I) -> bool {
                                if (isa<const AllocActivationInst>(I)) {
                                  return I->getNumUsers() < 2;
                                }
                                return false;
                              }),
               std::end(instrs));
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
      // Operands must be different, but of the same type.
      if (destOp.first->getType() != srcOp.first->getType() ||
          destOp.first == srcOp.first) {
        continue;
      }

      if (!Instruction::isInplaceOp(I, first, second)) {
        continue;
      }

      // If both the src and the dest operands are dead, this means that we can
      // reuse the buffer storage!
      if (!liveBuffers.count(destOp.first) && !liveBuffers.count(srcOp.first)) {
        replaceAllNonDeallocUsersWith(destOp.first, srcOp.first);
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

  // For each instruction, in reverse order.
  for (auto it = instrs.rbegin(), e = instrs.rend(); it != e; ++it) {
    Instruction *I = *it;

    // Remove <out> dependencies from the live set, because this instruction
    // writes into them. This means that the buffer is unused before the write
    // point.
    for (unsigned op = 0, ope = I->getNumOperands(); op < ope; op++) {
      auto O = I->getOperand(op);
      auto ai = dyn_cast<AllocActivationInst>(O.first);
      if (!ai) {
        continue;
      }

      // <Out> dependency means that the buffer is being killed. Remove from the
      // live list.
      if (O.second == OperandKind::Out) {
        auto it = liveBuffers.find(ai);
        if (it != liveBuffers.end()) {
          liveBuffers.erase(it);
        }
        continue;
      }
      // The <InOut> means that the value of the buffer is being consumed,
      // which means that it is alive. Add to the live set.
      if (ai && O.second == OperandKind::InOut) {
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
      auto ai = dyn_cast<AllocActivationInst>(O.first);
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

/// This optimization is based on the paper:
/// "Training Deep Nets with Sublinear Memory Cost" Arxiv 1604.06174
/// The idea is that instead of keeping a buffer around for a really long time
/// we can simply recompute some nodes. There is a tradeoff here between compute
/// and memory usage. The idea is to reduce memory usage significantly at a low
/// compute cost. Only apply this optimization when two parallel lifetimes are
/// reduces to one, for at least a part of the program.
void rematerializeCompute(Module &M) {
  auto &instrs = M.getInstrs();

  // Don't rematerialize if the distance between the original calculation and
  // the use is below this number of instructions.
  const unsigned rematerializeDistance = 5;

  unsigned instIdx = 0;

  // Calculate the liveness of the allocas in the block. This does not include
  // the alloc/dealloc instructions because they will be shrinked later on.
  LivenessMap liveness;
  calculateLiveness(M, liveness);

  // This map maps the destination buffers to the single writer instruction
  // that stores into them. The map also saves the index of the writer in the
  // basic block, starting from the top.
  std::unordered_map<Value *, std::pair<ReluInst *, unsigned>> writers;
  // Maps the original values to the re-calculated one. It's always better to
  // use the recalculated write because it is closer.
  std::unordered_map<Value *, Value *> rewrites;

  // Do an initial pass that collects all of the available RELUs.
  for (auto it = instrs.begin(), e = instrs.end(); it != e; ++it) {
    instIdx++;
    auto RL = dyn_cast<ReluInst>(*it);
    if (!RL) {
      continue;
    }

    // Ignore RL instructions that are writing to a shared buffer.
    if (RL != getSingleWriter(RL->getDest())) {
      continue;
    }

    writers[RL->getDest()] = {RL, instIdx};
  }

  // Do a second pass that rematerializes the RELUs.
  instIdx = 0;
  for (auto it = instrs.begin(), e = instrs.end(); it != e; ++it) {
    Instruction *I = *it;
    instIdx++;

    // Try to optimize the operands of each instruction that we encounter.
    for (unsigned op = 0, ope = I->getNumOperands(); op < ope; op++) {
      auto O = I->getOperand(op);
      // Ignore write operands.
      if (O.second != OperandKind::In) {
        continue;
      }

      // Ignore operands that don't touch known allocas.
      auto reluIter = writers.find(O.first);
      if (reluIter == writers.end()) {
        continue;
      }

      // Ignore very recently allocated .
      unsigned indexOfPrevReluWriter = reluIter->second.second;
      if ((instIdx - indexOfPrevReluWriter) < rematerializeDistance) {
        continue;
      }

      // If we've already rematerialized this computation then we can use the
      // cache.
      auto cacheIter = rewrites.find(O.first);
      if (cacheIter != rewrites.end()) {
        I->setOperand(op, cacheIter->second);
        continue;
      }

      // Check if the lifetime of the thing that feeds into the original relu is
      // still alive. If it's not aloive the copying the relu would extend it's
      // lifetime for no good reason.
      ReluInst *prevRelu = reluIter->second.first;

      auto LI = liveness.find(prevRelu->getSrc());
      if (LI == liveness.end()) {
        // Cound not find liveness for the original relu operand. Is it not an
        // alloca?
        continue;
      }

      // This is the interval of the thing that flows into the RELU.
      Interval origReluSrcInterval = LI->second;
      assert(origReluSrcInterval.first < instIdx && "Invalid start index");

      // Don't perform this optimization if it extends the lifetime of the
      // inputs of the relu.
      if (origReluSrcInterval.second < instIdx) {
        continue;
      }

      // Recompute the relu locally.
      auto *A = new AllocActivationInst(O.first->getName().str() + ".re",
                                        O.first->getType());
      instrs.insert(it, A);
      auto *R = new ReluInst("re.Relu", A, prevRelu->getSrc());
      instrs.insert(it, R);
      auto *D = new DeallocActivationInst("re", A);
      instrs.push_back(D);

      I->setOperand(op, A);
      rewrites[O.first] = A;
      break;
    }
  }
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
    }
  }
}

void glow::optimize(Module &M, CompilationMode mode) {
  M.verify();

  // Try to recompute instead of carying large buffers for a while.
  rematerializeCompute(M);

  // Reuse buffers from previous operations.
  shareBuffers(M);

  // Remove unused allocations.
  deleteDeadAllocs(M);

  // Shorten the lifetime of buffers.
  hoistDealloc(M);
  sinkAllocas(M);

  // Turn read-only weights into constant weights.
  makeWeightsConst(M);

  M.verify();
}
