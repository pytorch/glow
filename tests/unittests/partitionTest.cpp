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

#include "BackendTestUtils.h"

#include "glow/Graph/Graph.h"
#include "glow/Optimizer/Partition.h"

#include "gtest/gtest.h"

#include <llvm/ADT/ArrayRef.h>

using namespace glow;

class PartitionTest : public ::testing::Test {
public:
  PartitionTest() : F_(mod_.createFunction("main")) {}

protected:
  Module mod_;
  Function *F_;
};

/// Execute a graph of functions serially, which is the simplest approach.
static void executeSerial(const FunctionGraph &G,
                          llvm::ArrayRef<Variable *> vars,
                          llvm::ArrayRef<Tensor *> inputs) {
  for (auto *F : G.getFunctions()) {
    ExecutionEngine EE;
    EE.compile(CompilationMode::Infer, F);
    EE.run(vars, inputs);
  }
}

TEST_F(PartitionTest, SerialExecution) {
  auto *input =
      mod_.createVariable(ElemKind::FloatTy, {1, 32}, "input",
                          VisibilityKind::Public, Variable::TrainKind::None);

  // Initial FC.
  Node *I = F_->createFullyConnected("initial_fc", input, 16);
  I = F_->createSigmoid("initial_sigmoid", I);

  // Left branch.
  Node *L = F_->createFullyConnected("left_fc1", I, 16);
  L = F_->createSigmoid("left_sigmoid1", L);
  L = F_->createFullyConnected("left_fc2", L, 8);
  L = F_->createSigmoid("left_sigmoid2", L);

  // Right branch.
  Node *R = F_->createFullyConnected("right_fc1", I, 16);
  R = F_->createSigmoid("right_sigmoid1", R);
  R = F_->createFullyConnected("right_fc2", R, 8);
  R = F_->createSigmoid("right_sigmoid2", R);

  // Join branches.
  auto *mul = F_->createMul("mul", L, R);
  auto *save = F_->createSave("ret", mul);
  auto &res = save->getVariable()->getPayload();

  auto G = glow::partition(F_);
  ASSERT_EQ(G.getFunctions().size(), 4);

  auto it = G.getFunctions().begin();
  {
    auto *F = *it++;
    EXPECT_EQ(F->getNodes().size(), 3);
    EXPECT_EQ(G.getDependencies(F).size(), 0);
  }
  {
    auto *F = *it++;
    EXPECT_EQ(F->getNodes().size(), 5);
    EXPECT_EQ(G.getDependencies(F).size(), 1);
  }
  {
    auto *F = *it++;
    EXPECT_EQ(F->getNodes().size(), 5);
    EXPECT_EQ(G.getDependencies(F).size(), 1);
  }
  {
    auto *F = *it++;
    EXPECT_EQ(F->getNodes().size(), 2);
    EXPECT_EQ(G.getDependencies(F).size(), 2);
  }

  // Infer using the un-partitioned graph.
  Tensor in(ElemKind::FloatTy, {1, 32});
  ExecutionEngine EE;
  EE.compile(CompilationMode::Infer, F_);
  EE.run({input}, {&in});
  Tensor ref = res.clone();

  // Infer using the partitioned graph.
  executeSerial(G, {input}, {&in});
  Tensor test = res.clone();
  EXPECT_TRUE(ref.isEqual(test));
}

TEST_F(PartitionTest, Branchover) {
  auto *input =
      mod_.createVariable(ElemKind::FloatTy, {1, 8}, "input",
                          VisibilityKind::Public, Variable::TrainKind::None);
  auto *FC1 = F_->createFullyConnected("fc1", input, 8);
  auto *FC2 = F_->createFullyConnected("fc2", FC1, 8);
  auto *add = F_->createAdd("add", FC1, FC2);
  auto *save = F_->createSave("save", add);
  auto &res = save->getVariable()->getPayload();

  auto G = glow::partition(F_);
  ASSERT_EQ(G.getFunctions().size(), 3);

  auto it = G.getFunctions().begin();
  {
    auto *F = *it++;
    EXPECT_EQ(F->getNodes().size(), 2);
    EXPECT_EQ(G.getDependencies(F).size(), 0);
  }
  {
    auto *F = *it++;
    EXPECT_EQ(F->getNodes().size(), 2);
    EXPECT_EQ(G.getDependencies(F).size(), 1);
  }
  {
    auto *F = *it++;
    EXPECT_EQ(F->getNodes().size(), 2);
    EXPECT_EQ(G.getDependencies(F).size(), 2);
  }

  // Infer using the un-partitioned graph.
  Tensor in(ElemKind::FloatTy, {1, 8});
  ExecutionEngine EE;
  EE.compile(CompilationMode::Infer, F_);
  EE.run({input}, {&in});
  Tensor ref = res.clone();

  // Infer using the partitioned graph.
  executeSerial(G, {input}, {&in});
  Tensor test = res.clone();
  EXPECT_TRUE(ref.isEqual(test));
}

TEST_F(PartitionTest, Train) {
  auto *input =
      mod_.createVariable(ElemKind::FloatTy, {1, 8}, "input",
                          VisibilityKind::Public, Variable::TrainKind::None);
  auto *FC = F_->createFullyConnected("fc", input, 8);
  F_->createSave("save", FC);
  auto *TF = glow::differentiate(F_, TrainingConfig());
  auto G = glow::partition(TF);
  ASSERT_EQ(G.getFunctions().size(), 6);
}

TEST_F(PartitionTest, VerifyTopo) {
  auto *F1 = mod_.createFunction("F1");
  auto *F2 = mod_.createFunction("F2");
  FunctionList functions{F1, F2};
  FunctionGraph G(functions);
  G.add(F2, F1);
  EXPECT_TRUE(G.verify());
}

TEST_F(PartitionTest, VerifyTopoFails) {
  auto *F1 = mod_.createFunction("F1");
  auto *F2 = mod_.createFunction("F2");
  FunctionList functions{F1, F2};
  FunctionGraph G(functions);
  G.add(F1, F2);
  EXPECT_FALSE(G.verify());
}

TEST_F(PartitionTest, VerifyCyclicFails) {
  auto *F1 = mod_.createFunction("F1");
  FunctionList functions{F1};
  FunctionGraph G(functions);
  G.add(F1, F1);
  EXPECT_FALSE(G.verify());
}
