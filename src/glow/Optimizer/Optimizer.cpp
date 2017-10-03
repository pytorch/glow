// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "glow/Optimizer/Optimizer.h"
#include "glow/IR/IR.h"
#include "glow/IR/Instrs.h"
#include "glow/Support/Casting.h"

#include <unordered_map>
#include <unordered_set>

using namespace glow;

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
  std::vector<Value::Use> usersVec(users.begin(), users.end());
  for (auto &U : usersVec) {
    // Ignore dealloc instrs.
    if (isa<DeallocActivationInst>(U.second)) {
      continue;
    }

    U.second->setOperand(U.first, with);
  }
}

/// Optimize the input/output buffer for the instruction \p I, based on the
/// liveness information in \p liveBuffers.
static void shareBuffersForInstr(const std::unordered_set<Value *> &liveBuffers,
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

      // If both the src and the dest operands are dead, this means that we can
      // reuse the buffer storage.
      if (!liveBuffers.count(destOp.first) && !liveBuffers.count(srcOp.first)) {
        replaceAllNonDeallocUsersWith(destOp.first, srcOp.first);
        return;
      }
    }
  }
}

static void shareBuffers(Module &M) {
  auto &instrs = M.getInstrs();
  std::unordered_set<Value *> liveBuffers;

  // All the weights are alive, because they are persistent.
  for (auto *W : M.getWeights()) {
    liveBuffers.insert(W);
  }

  // For each instruction, in reverse order.
  for (auto it = instrs.rbegin(), e = instrs.rend(); it != e; ++it) {
    Instruction *I = *it;

    // Remove <out> dependencies from the live set, because this instruction
    // writes into them.
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

    if (Instruction::mayShareBuffers(I))
      shareBuffersForInstr(liveBuffers, I);

    // Insert the input buffers into the live set.
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

void glow::optimize(Module &M, OptimizationMode mode) {
  M.verify();

  if (mode == OptimizationMode::None) {
    return;
  }

  // Sharing buffers is only legal in training mode because it kills the
  // backprop.
  if (mode == OptimizationMode::Infer) {
    shareBuffers(M);
    M.verify();
  }

  // Remove unused allocations.
  deleteDeadAllocs(M);
  M.verify();

  // Shorten the lifetime of buffers.
  hoistDealloc(M);
  M.verify();
}
