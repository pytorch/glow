/**
 * Copyright (c) 2017-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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

  TrainingConfig TC;

  // Construct the network:
  TC.learningRate = 0.001;
  TC.momentum = 0.9;
  TC.L2Decay = 0.001;
  TC.L1Decay = 0.001;

  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  Variable *A = mod.createVariable(ElemKind::FloatTy, {10, 28, 28, 1}, "input",
                                   VisibilityKind::Public, false);

  auto *CV0 = F->createConv("conv1", A, 16, 5, 1, 2, 1);
  auto *RL0 = F->createRELU("relu1", CV0);
  auto *MP0 = F->createMaxPool("pool1", RL0, 3, 3, 0);

  auto *CV1 = F->createConv("conv2", MP0, 16, 5, 1, 2, 1);
  auto *RL1 = F->createRELU("conv23", CV1);
  auto *MP1 = F->createMaxPool("pool2", RL1, 3, 3, 0);

  auto *FCL1 = F->createFullyConnected("fc3", MP1, 10);
  auto *RL2 = F->createRELU("relu3", FCL1);
  Variable *selected = mod.createVariable(
      ElemKind::Int64ITy, {10, 1}, "selected", VisibilityKind::Public, false);

  auto *SM = F->createSoftMax("sm", RL2, selected);

  auto *result = F->createSave("return", SM);
  (void)result;

  Function *TF = glow::differentiate(F, TC);
  EE.compile(CompilationMode::Train, TF);
  EE.compile(CompilationMode::Infer, F);
}

TEST(GraphAutoGrad, checkLRNGen) {
  ExecutionEngine EE;
  TrainingConfig TC;

  // Construct the network:
  TC.learningRate = 0.001;
  TC.momentum = 0.9;
  TC.L2Decay = 0.001;

  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  Variable *A = mod.createVariable(ElemKind::FloatTy, {10, 28, 28, 1}, "input",
                                   VisibilityKind::Public, false);
  auto *CV0 = F->createLocalResponseNormalization("LRN", A);
  auto *FCL1 = F->createFullyConnected("fc3", CV0, 10);
  auto *RL2 = F->createRELU("relu3", FCL1);
  Variable *selected = mod.createVariable(
      ElemKind::Int64ITy, {10, 1}, "selected", VisibilityKind::Public, false);

  auto *SM = F->createSoftMax("sm", RL2, selected);

  auto *result = F->createSave("return", SM);
  (void)result;
  Function *TF = glow::differentiate(F, TC);
  EE.compile(CompilationMode::Train, TF);
  EE.compile(CompilationMode::Infer, F);
}

TEST(GraphAutoGrad, cloneAndDiff) {
  // The test ensures that unused variables are not touched in differentiation.
  ExecutionEngine EE;
  TrainingConfig TC;

  Module M;

  auto *F = M.createFunction("main");
  Node *A =
      M.createVariable(ElemKind::FloatTy, {1}, "A", VisibilityKind::Private);
  Node *B =
      M.createVariable(ElemKind::FloatTy, {1}, "B", VisibilityKind::Private);
  Node *AplusB_F = F->createAdd("AplusB", A, B);

  EXPECT_EQ(M.getVars().size(), 2);

  auto *G = F->clone("G");

  EXPECT_EQ(M.getVars().size(), 2);
  EXPECT_EQ(G->getNodes().size(), 1);

  Node *C =
      M.createVariable(ElemKind::FloatTy, {1}, "C", VisibilityKind::Private);
  Node *AplusB_G = &G->getNodes().back();
  G->createAdd("totalSum", AplusB_G, C);

  EXPECT_EQ(M.getVars().size(), 3);

  Node *label = M.createVariable(ElemKind::FloatTy, {1}, "label",
                                 VisibilityKind::Public, false);
  Node *reg = F->createRegression("reg", AplusB_F, label);
  F->createSave("return", reg);

  EXPECT_EQ(M.getVars().size(), 5);

  auto *diffF = differentiate(F, TC);

  diffF->verify();

  EXPECT_EQ(M.getFunctions().size(), 3);
  EXPECT_EQ(M.getVars().size(), 5);
  // Check that we have as many SGD node as variables that need to be trained.
  unsigned nbSGDs = 0;
  unsigned nbSGDA = 0;
  unsigned nbSGDB = 0;
  for (auto &node : diffF->getNodes()) {
    SGDNode *SGD = llvm::dyn_cast<SGDNode>(&node);
    if (!SGD)
      continue;
    ++nbSGDs;
    if (A == SGD->getWeight())
      ++nbSGDA;
    else if (B == SGD->getWeight())
      ++nbSGDB;
  }
  EXPECT_EQ(nbSGDs, 2);
  EXPECT_EQ(nbSGDA, 1);
  EXPECT_EQ(nbSGDB, 1);
}
