// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "CPUBackend.h"

#include "glow/CodeGen/MemoryAllocator.h"
#include "glow/Graph/Graph.h"
#include "glow/Graph/Nodes.h"
#include "glow/IR/Instrs.h"
#include "glow/Optimizer/Optimizer.h"

using namespace glow;
using llvm::StringRef;
using llvm::dyn_cast;
using llvm::isa;

static bool isZeroNode(NodeValue N) {
  SplatNode *splat = dyn_cast<SplatNode>(N);
  if (!splat)
    return false;

  return splat->getValue() == 0;
}

bool CPUBackend::transform(Function *F) {
  bool changed = false;
  for (auto node : F->getNodes()) {
    if (auto *MN = dyn_cast<MaxNode>(node)) {
      if (isZeroNode(MN->getLHS())) {
        auto MZN = F->addNode(new CPUMaxZeroNode(MN->getName(), MN->getRHS()));
        NodeValue(node, 0).replaceAllUsesOfWith(MZN);
        changed = true;
        continue;
      }
      if (isZeroNode(MN->getRHS())) {
        auto MZN = F->addNode(new CPUMaxZeroNode(MN->getName(), MN->getRHS()));
        NodeValue(node, 0).replaceAllUsesOfWith(MZN);
        changed = true;
        continue;
      }
    }
  }

  return changed;
}
