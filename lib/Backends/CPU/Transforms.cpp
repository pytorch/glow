// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "CPUBackend.h"

#include "glow/CodeGen/MemoryAllocator.h"
#include "glow/Graph/Graph.h"
#include "glow/Graph/Nodes.h"
#include "glow/IR/Instrs.h"
#include "glow/Optimizer/Optimizer.h"

using namespace glow;
using llvm::dyn_cast;
using llvm::isa;

bool CPUBackend::transformPreLowering(Function *F) {
  bool changed = false;
  for (auto node : F->getNodes()) {
    if (isa<FullyConnectedNode>(node)) {
      node->setShouldLower(false);
      changed = true;
    }
  }
  return changed;
}

bool CPUBackend::transformPostLowering(Function *F) {
  bool changed = false;
  for (auto node : F->getNodes()) {
    if (auto *MN = dyn_cast<MaxNode>(node)) {
      if (auto *splat = dyn_cast<SplatNode>(MN->getLHS())) {
        auto MSN = F->addNode(new CPUMaxSplatNode(MN->getName(), MN->getRHS(),
                                                  splat->getValue()));
        NodeValue(node, 0).replaceAllUsesOfWith(MSN);
        changed = true;
        continue;
      }
      if (auto *splat = dyn_cast<SplatNode>(MN->getRHS())) {
        auto MSN = F->addNode(new CPUMaxSplatNode(MN->getName(), MN->getLHS(),
                                                  splat->getValue()));
        NodeValue(node, 0).replaceAllUsesOfWith(MSN);
        changed = true;
        continue;
      }
    }
  }

  return changed;
}
