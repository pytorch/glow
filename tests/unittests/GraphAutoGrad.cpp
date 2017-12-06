// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Graph/Graph.h"
#include "glow/IR/IR.h"
#include "glow/IR/IRBuilder.h"
#include "glow/IR/Instrs.h"

#include "gtest/gtest.h"

#include <cassert>
#include <string>

using namespace glow;

TEST(GraphAutoGrad, autoGrad) {
  ExecutionEngine EE;

  // Construct the network:
  EE.getConfig().learningRate = 0.001;
  EE.getConfig().momentum = 0.9;
  EE.getConfig().L2Decay = 0.001;

  auto &G = EE.getGraph();

  Variable *A = G.createVariable(ElemKind::FloatTy, {10, 28, 28, 1}, "input",
                                 Variable::InitKind::Extern);

  auto *CV0 = G.createConv("conv1", A, 16, 5, 1, 2);
  auto *RL0 = G.createRELU("relu1", CV0);
  auto *MP0 = G.createPool("pool1", RL0, PoolNode::Mode::Max, 3, 3, 0);

  auto *CV1 = G.createConv("conv2", MP0, 16, 5, 1, 2);
  auto *RL1 = G.createRELU("conv23", CV1);
  auto *MP1 = G.createPool("pool2", RL1, PoolNode::Mode::Max, 3, 3, 0);

  auto *FCL1 = G.createFullyConnected("fc3", MP1, 10);
  auto *RL2 = G.createRELU("relu3", FCL1);
  Variable *selected = G.createVariable(ElemKind::IndexTy, {10, 1}, "selected",
                                        Variable::InitKind::Extern);

  auto *SM = G.createSoftMax("sm", RL2, selected);

  auto *result = G.createSave("return", SM);
  (void)result;

  G.dump();
  G.dumpDAG();

  EE.compile(CompilationMode::Train);
}
