// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "glow/IR/IR.h"
#include "glow/IR/Instrs.h"
#include "glow/Optimizer/Optimizer.h"
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

void glow::optimize(Module &M, CompilationMode mode) {
  M.verify();

  // Remove unused allocations.
  deleteDeadAllocs(M);

  // Shorten the lifetime of buffers.
  hoistDealloc(M);
  sinkAllocas(M);
  M.verify();
}
