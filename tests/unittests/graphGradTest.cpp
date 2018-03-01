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
  auto *MP0 = F->createPoolMax("pool1", RL0, 3, 3, 0);

  auto *CV1 = F->createConv("conv2", MP0, 16, 5, 1, 2);
  auto *RL1 = F->createRELU("conv23", CV1);
  auto *MP1 = F->createPoolMax("pool2", RL1, 3, 3, 0);

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

TEST(GraphAutoGrad, cloneAndDiff) {
  // The test ensures that unused variables are not touched in differentiation.
  ExecutionEngine EE;

  Module M;

  auto *F = M.createFunction("main");
  Node *A = M.createVariable(ElemKind::FloatTy, {1}, "A",
                             Variable::VisibilityKind::Private);
  Node *B = M.createVariable(ElemKind::FloatTy, {1}, "B",
                             Variable::VisibilityKind::Private);
  Node *AplusB_F = F->createAdd("AplusB", A, B);

  EXPECT_EQ(M.getVars().size(), 2);

  auto *G = F->clone("G");

  EXPECT_EQ(M.getVars().size(), 2);
  EXPECT_EQ(G->getNodes().size(), 1);

  Node *C = M.createVariable(ElemKind::FloatTy, {1}, "C",
                             Variable::VisibilityKind::Private);
  Node *AplusB_G = G->getNodes().back();
  G->createAdd("totalSum", AplusB_G, C);

  EXPECT_EQ(M.getVars().size(), 3);

  Node *label = M.createVariable(ElemKind::FloatTy, {1}, "label",
                                 Variable::VisibilityKind::Public,
                                 Variable::TrainKind::None);
  Node *reg = F->createRegression("reg", AplusB_F, label);
  F->createSave("return", reg);

  EXPECT_EQ(M.getVars().size(), 5);

  auto *diffF = differentiate(F, EE.getConfig());

  diffF->verify();

  EXPECT_EQ(M.getFunctions().size(), 3);
  EXPECT_EQ(M.getVars().size(), 7);
}
