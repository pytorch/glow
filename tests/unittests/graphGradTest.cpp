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

  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  Variable *A = mod.createVariable(ElemKind::FloatTy, {10, 28, 28, 1}, "input",
                                   Variable::VisibilityKind::Public,
                                   Variable::TrainKind::None);

  auto *CV0 = F->createConv("conv1", A, 16, 5, 1, 2);
  auto *RL0 = F->createRELU("relu1", CV0);
  auto *MP0 = F->createPool("pool1", RL0, PoolNode::Mode::Max, 3, 3, 0);

  auto *CV1 = F->createConv("conv2", MP0, 16, 5, 1, 2);
  auto *RL1 = F->createRELU("conv23", CV1);
  auto *MP1 = F->createPool("pool2", RL1, PoolNode::Mode::Max, 3, 3, 0);

  auto *FCL1 = F->createFullyConnected("fc3", MP1, 10);
  auto *RL2 = F->createRELU("relu3", FCL1);
  Variable *selected = mod.createVariable(
      ElemKind::IndexTy, {10, 1}, "selected", Variable::VisibilityKind::Public,
      Variable::TrainKind::None);

  auto *SM = F->createSoftMax("sm", RL2, selected);

  auto *result = F->createSave("return", SM);
  (void)result;

  Function *TF = glow::differentiate(F, EE.getConfig());
  EE.compile(CompilationMode::Train, TF);
  EE.compile(CompilationMode::Infer, F);
}

TEST(GraphAutoGrad, checkLRNGen) {
  ExecutionEngine EE;

  // Construct the network:
  EE.getConfig().learningRate = 0.001;
  EE.getConfig().momentum = 0.9;
  EE.getConfig().L2Decay = 0.001;

  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  Variable *A = mod.createVariable(ElemKind::FloatTy, {10, 28, 28, 1}, "input",
                                   Variable::VisibilityKind::Public,
                                   Variable::TrainKind::None);
  auto *CV0 = F->createLocalResponseNormalization("LRN", A);
  auto *FCL1 = F->createFullyConnected("fc3", CV0, 10);
  auto *RL2 = F->createRELU("relu3", FCL1);
  Variable *selected = mod.createVariable(
      ElemKind::IndexTy, {10, 1}, "selected", Variable::VisibilityKind::Public,
      Variable::TrainKind::None);

  auto *SM = F->createSoftMax("sm", RL2, selected);

  auto *result = F->createSave("return", SM);
  (void)result;
  Function *TF = glow::differentiate(F, EE.getConfig());
  EE.compile(CompilationMode::Train, TF);
  EE.compile(CompilationMode::Infer, F);
}
