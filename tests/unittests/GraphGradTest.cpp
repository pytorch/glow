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
  PlaceholderBindings bindings;

  TrainingConfig TC;

  // Construct the network:
  TC.learningRate = 0.001;
  TC.momentum = 0.9;
  TC.L2Decay = 0.001;
  TC.L1Decay = 0.001;

  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  auto *A =
      mod.createPlaceholder(ElemKind::FloatTy, {10, 28, 28, 1}, "input", false);

  auto *CV0 = F->createConv(bindings, "conv1", A, 16, 5, 1, 2, 1);
  auto *RL0 = F->createRELU("relu1", CV0);
  auto *MP0 = F->createMaxPool("pool1", RL0, 3, 3, 0);

  auto *CV1 = F->createConv(bindings, "conv2", MP0, 16, 5, 1, 2, 1);
  auto *RL1 = F->createRELU("conv23", CV1);
  auto *MP1 = F->createMaxPool("pool2", RL1, 3, 3, 0);

  auto *FCL1 = F->createFullyConnected(bindings, "fc3", MP1, 10);
  auto *RL2 = F->createRELU("relu3", FCL1);
  auto *selected =
      mod.createPlaceholder(ElemKind::Int64ITy, {10, 1}, "selected", false);

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
  PlaceholderBindings bindings;

  // Construct the network:
  TC.learningRate = 0.001;
  TC.momentum = 0.9;
  TC.L2Decay = 0.001;

  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  auto *A =
      mod.createPlaceholder(ElemKind::FloatTy, {10, 28, 28, 1}, "input", false);
  auto *CV0 = F->createLocalResponseNormalization("LRN", A);
  auto *FCL1 = F->createFullyConnected(bindings, "fc3", CV0, 10);
  auto *RL2 = F->createRELU("relu3", FCL1);
  auto *selected =
      mod.createPlaceholder(ElemKind::Int64ITy, {10, 1}, "selected", false);

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
  PlaceholderBindings bindings;
  Module M;

  auto *F = M.createFunction("main");
  Node *A = M.createPlaceholder(ElemKind::FloatTy, {1}, "A", true);
  Node *B = M.createPlaceholder(ElemKind::FloatTy, {1}, "B", true);
  Node *AplusB_F = F->createAdd("AplusB", A, B);

  EXPECT_EQ(M.getPlaceholders().size(), 2);

  auto *G = F->clone("G");

  EXPECT_EQ(M.getPlaceholders().size(), 2);
  EXPECT_EQ(G->getNodes().size(), 1);

  Node *C = M.createPlaceholder(ElemKind::FloatTy, {1}, "C", true);
  Node *AplusB_G = &G->getNodes().back();
  G->createAdd("totalSum", AplusB_G, C);

  EXPECT_EQ(M.getPlaceholders().size(), 3);

  Node *label = M.createPlaceholder(ElemKind::FloatTy, {1}, "label", false);
  Node *reg = F->createRegression("reg", AplusB_F, label);
  F->createSave("return", reg);

  EXPECT_EQ(M.getPlaceholders().size(), 5);

  auto *diffF = differentiate(F, TC);

  EXPECT_TRUE(diffF->verify());

  EXPECT_EQ(M.getFunctions().size(), 3);
  EXPECT_EQ(M.getPlaceholders().size(), 5);
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

/// Check that we can differentiate functions that update Placeholder graphs.
TEST(GraphAutoGrad, checkPlaceholderGradTest) {
  ExecutionEngine EE;
  TrainingConfig TC;
  PlaceholderBindings bindings;

  // Construct the network:
  TC.learningRate = 0.001;

  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  Placeholder *A =
      mod.createPlaceholder(ElemKind::FloatTy, {10, 28, 28, 1}, "input", true);
  auto *RL = F->createRELU("relu", A);
  F->createSave("return", RL);

  // Expect a single user to the trainable input placeholder.
  EXPECT_EQ(A->getNumUsers(), 1);

  Function *TF = glow::differentiate(F, TC);
  EE.compile(CompilationMode::Train, TF);
  EE.compile(CompilationMode::Infer, F);

  // Check that the Placeholder has multiple users, because at least one write
  // node will be added.
  EXPECT_GT(A->getNumUsers(), 1);
}

/// Check that we can differentiate functions that use ConvertToNode.
TEST(GraphAutoGrad, checkConvertToGradTest) {
  ExecutionEngine EE;
  TrainingConfig TC;
  PlaceholderBindings bindings;

  // Construct the network:
  TC.learningRate = 0.001;

  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  auto *A = mod.createPlaceholder(ElemKind::FloatTy, {20, 13}, "A", false);
  auto inputHandle = bindings.allocate(A)->getHandle<float>();
  inputHandle.randomize(-3.0, 3.0, mod.getPRNG());

  TypeRef outTy = mod.uniqueType(ElemKind::Float16Ty, A->dims());

  auto *convertTo = F->createConvertTo("convertTo", A, outTy);
  auto *result = F->createSave("save", convertTo);
  bindings.allocate(result->getPlaceholder());

  Function *TF = glow::differentiate(F, TC);
  EE.compile(CompilationMode::Train, TF);
  EE.compile(CompilationMode::Infer, F);
}

/// Check that we can differentiate functions that use MatMulNode.
TEST(GraphAutoGrad, checkMatMulGradTest) {
  ExecutionEngine EE;
  TrainingConfig TC;
  PlaceholderBindings Bindings;

  // Construct the network:
  TC.learningRate = 0.001;

  auto &Mod = EE.getModule();
  Function *F = Mod.createFunction("main");

  auto *A = Mod.createPlaceholder(ElemKind::FloatTy, {20, 13}, "A", false);
  auto HandleA = Bindings.allocate(A)->getHandle<float>();
  HandleA.randomize(-3.0, 3.0, Mod.getPRNG());

  auto *B = Mod.createPlaceholder(ElemKind::FloatTy, {13, 30}, "B", false);
  auto HandleB = Bindings.allocate(B)->getHandle<float>();
  HandleB.randomize(-3.0, 3.0, Mod.getPRNG());

  auto *MatMul = F->createMatMul("matMul", A, B);
  auto *R = F->createSave("save", MatMul);
  Bindings.allocate(R->getPlaceholder());

  Function *TF = glow::differentiate(F, TC);
  EE.compile(CompilationMode::Train, TF);
  EE.compile(CompilationMode::Infer, F);
}

/// Check that we can differentiate functions that use BatchedReduceAddNode.
TEST(GraphAutoGrad, checkBatchedReduceAddGradTest) {
  ExecutionEngine EE;
  TrainingConfig TC;
  PlaceholderBindings Bindings;

  auto &Mod = EE.getModule();
  Function *F = Mod.createFunction("main");

  TypeRef Ty = Mod.uniqueType(ElemKind::FloatTy, {1, 10});
  auto *A = Mod.createPlaceholder(ElemKind::FloatTy, {10, 10}, "A", false);
  auto HandleA = Bindings.allocate(A)->getHandle<float>();
  HandleA.randomize(-3.0, 3.0, Mod.getPRNG());

  auto *BRA = F->createBatchedReduceAdd("BRA", Ty, A, 0 /*axis*/);
  auto *R = F->createSave("save", BRA);
  Bindings.allocate(R->getPlaceholder());

  Function *TF = glow::differentiate(F, TC);
  EE.compile(CompilationMode::Train, TF);
  EE.compile(CompilationMode::Infer, F);
}

/// Check that we can differentiate functions that use GatherNode.
TEST(GraphAutoGrad, checkGatherGrad1DIndexTest) {
  ExecutionEngine EE;
  TrainingConfig TC;
  PlaceholderBindings Bindings;

  auto &Mod = EE.getModule();
  Function *F = Mod.createFunction("main");

  auto *Data = Mod.createPlaceholder(ElemKind::FloatTy, {3, 4}, "Data", false);
  auto *Indices =
      Mod.createPlaceholder(ElemKind::Int64ITy, {2}, "Indices", false);

  auto HandleData = Bindings.allocate(Data)->getHandle<float>();
  HandleData.randomize(-3.0, 3.0, Mod.getPRNG());

  Bindings.allocate(Indices)->getHandle<int64_t>() = {0, 2};

  auto *G = F->createGather("gather", Data, Indices, 0 /*batchDims*/);
  auto *R = F->createSave("save", G);
  Bindings.allocate(R->getPlaceholder());

  Function *TF = glow::differentiate(F, TC);
  EE.compile(CompilationMode::Train, TF);
  EE.compile(CompilationMode::Infer, F);
}

TEST(GraphAutoGrad, checkGatherGrad2DIndexTest) {
  ExecutionEngine EE;
  TrainingConfig TC;
  PlaceholderBindings Bindings;

  auto &Mod = EE.getModule();
  Function *F = Mod.createFunction("main");

  auto *Data = Mod.createPlaceholder(ElemKind::FloatTy, {8, 4}, "Data", false);
  auto *Indices =
      Mod.createPlaceholder(ElemKind::Int64ITy, {2, 2}, "Indices", false);

  auto HandleData = Bindings.allocate(Data)->getHandle<float>();
  HandleData.randomize(-3.0, 3.0, Mod.getPRNG());

  Bindings.allocate(Indices)->getHandle<int64_t>() = {0, 2, 1, 3};

  auto *G = F->createGather("gather", Data, Indices, 0 /*batchDims*/);
  auto *R = F->createSave("save", G);
  Bindings.allocate(R->getPlaceholder());

  Function *TF = glow::differentiate(F, TC);
  EE.compile(CompilationMode::Train, TF);
  EE.compile(CompilationMode::Infer, F);
}

TEST(GraphAutoGrad, checkGatherGrad3DIndexTest) {
  ExecutionEngine EE;
  TrainingConfig TC;
  PlaceholderBindings Bindings;

  auto &Mod = EE.getModule();
  Function *F = Mod.createFunction("main");

  auto *Data = Mod.createPlaceholder(ElemKind::FloatTy, {8, 4}, "Data", false);
  auto *Indices =
      Mod.createPlaceholder(ElemKind::Int64ITy, {2, 2, 2}, "Indices", false);

  auto HandleData = Bindings.allocate(Data)->getHandle<float>();
  HandleData.randomize(-3.0, 3.0, Mod.getPRNG());

  Bindings.allocate(Indices)->getHandle<int64_t>() = {0, 2, 1, 3, 4, 5, 7, 6};

  auto *G = F->createGather("gather", Data, Indices, 0 /*batchDims*/);
  auto *R = F->createSave("save", G);
  Bindings.allocate(R->getPlaceholder());

  Function *TF = glow::differentiate(F, TC);
  EE.compile(CompilationMode::Train, TF);
  EE.compile(CompilationMode::Infer, F);
}
