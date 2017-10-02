#include "glow/Optimizer/Optimizer.h"
#include "glow/IR/IR.h"
#include "glow/IR/Instrs.h"
#include "glow/Support/Casting.h"

#include <unordered_map>

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

void glow::optimize(Module &M) {
  M.verify();

  hoistDealloc(M);

  M.verify();
}
