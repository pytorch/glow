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
#include "glow/Partitioner/Partitioner.h"
#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Graph/Graph.h"

#include "gtest/gtest.h"

using namespace glow;

class PartitionerTest : public ::testing::Test {
public:
  PartitionerTest() : F_(mod_.createFunction("main")) {}

protected:
  Module mod_;
  Function *F_;
  Context ctx_;
};

/// Execute a graph of functions based on the given DAG.
static void executeDAG(DAGNode *G, Module &mod, Context &ctx,
                       llvm::ArrayRef<Placeholder *> vars,
                       llvm::ArrayRef<Tensor *> inputs) {
  std::unordered_map<std::string, Function *> name2func;

  for (auto *F : mod.getFunctions()) {
    name2func[F->getName()] = F;
  }

  std::vector<DAGNode *> exeList;
  int endPt = 0;
  int curPt = 0;
  // The first node is always the dummy node.
  exeList.push_back(G);
  endPt++;
  while (curPt < endPt) {
    DAGNode *dag = exeList.at(curPt);
    // The root in a G is always a dummy function.
    if (curPt > 0) {
      ExecutionEngine EE;
      Function *func = name2func[dag->name];
      EE.compile(CompilationMode::Infer, func);
      updateInputPlaceholders(ctx, vars, inputs);
      EE.run(ctx);
    }
    for (int i = 0, e = dag->children.size(); i < e; i++) {
      exeList.push_back(dag->children.at(i));
      endPt++;
    }
    curPt++;
  }
}

TEST_F(PartitionerTest, test1) {
  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 32}, "input", false);
  ctx_.allocate(input);

  // Initial FC.
  Node *I = F_->createFullyConnected(ctx_, "initial_fc", input, 16);
  I = F_->createSigmoid("initial_sigmoid", I);

  // Left branch.
  Node *L = F_->createFullyConnected(ctx_, "left_fc1", I, 16);
  L = F_->createSigmoid("left_sigmoid1", L);
  L = F_->createFullyConnected(ctx_, "left_fc2", L, 8);
  L = F_->createSigmoid("left_sigmoid2", L);

  // Right branch.
  Node *R = F_->createFullyConnected(ctx_, "right_fc1", I, 16);
  R = F_->createSigmoid("right_sigmoid1", R);
  R = F_->createFullyConnected(ctx_, "right_fc2", R, 8);
  R = F_->createSigmoid("right_sigmoid2", R);

  // Join branches.
  auto *mul = F_->createMul("mul", L, R);
  auto *save = F_->createSave("ret", mul);
  auto &res = *ctx_.allocate(save->getPlaceholder());

  // Infer using the un-partitioned graph.
  Tensor in(ElemKind::FloatTy, {1, 32});
  ExecutionEngine EE;

  EE.compile(CompilationMode::Infer, F_);
  updateInputPlaceholders(ctx_, {input}, {&in});
  EE.run(ctx_);
  Tensor ref = res.clone();

  std::vector<DeviceInfo> devices;
  Partitioner myPartitioner(&mod_, devices);

  DAGNodeList myList = std::move(myPartitioner.Partition());
  ASSERT_EQ(mod_.getFunctions().size(), 3);
  ASSERT_EQ(myList.roots.size(), 1);

  // Run the paritioned graph and compare the results.
  ctx_.allocate(mod_.getPlaceholders());
  for (auto it = myList.roots.begin(); it != myList.roots.end(); ++it) {
    ctx_.allocate(mod_.getPlaceholders());
    executeDAG((*it).get(), mod_, ctx_, {input}, {&in});
    Tensor test = res.clone();
    EXPECT_TRUE(ref.isEqual(test));
  }
}
