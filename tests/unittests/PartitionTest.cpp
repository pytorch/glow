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

#include "gtest/gtest.h"

using namespace glow;

class PartitionTest : public ::testing::Test {
public:
  PartitionTest() : F_(mod_.createFunction("main")) {}

protected:
  Module mod_;
  Function *F_;
};

TEST_F(PartitionTest, Simple) {
  auto *input = mod_.createVariable(ElemKind::FloatTy, {1, 32}, "input",
                                    VisibilityKind::Public,
                                    Variable::TrainKind::Broadcast);

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

  auto *mul = F_->createMul("mul", L, R);
  auto *save = F_->createSave("ret", mul);
  auto &res = llvm::cast<Variable>(save->getOutput())->getPayload();

  // Infer using un-partitioned graph.
  Tensor in(ElemKind::FloatTy, {1, 32});
  ExecutionEngine EE(BackendKind::CPU);
  EE.compile(CompilationMode::Infer, F_);
  EE.run({input}, {&in});
  Tensor out1 = res.clone();

  // Infer using the partitioned graph.
  res.zero();
  EE.setBackend(BackendKind::MultiCPU);
  EE.compile(CompilationMode::Infer, F_);
  EE.run({input}, {&in});
  Tensor out2 = res.clone();

  EXPECT_TRUE(out1.isEqual(out2));
}
