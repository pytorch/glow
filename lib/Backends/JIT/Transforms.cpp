// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "JIT.h"

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

bool JITBackend::transform(Function *F) {
  bool changed = false;
  for (auto node : F->getNodes()) {
    if (auto *AN = dyn_cast<ArithmeticNode>(node)) {
      if (AN->getMode() == ArithmeticNode::Mode::Max) {
        if (isZeroNode(AN->getLHS())) {
          auto I = F->createIntrinsicNode(AN->getName(), "jit.max0",
                                          {AN->getRHS()}, {AN->getType()});
          NodeValue(node, 0).replaceAllUsesOfWith(I);
          changed = true;
          continue;
        }
        if (isZeroNode(AN->getRHS())) {
          auto I = F->createIntrinsicNode(AN->getName(), "jit.max0",
                                          {AN->getLHS()}, {AN->getType()});
          NodeValue(node, 0).replaceAllUsesOfWith(I);
          changed = true;
          continue;
        }
      }
    }
  }

  return changed;
}
