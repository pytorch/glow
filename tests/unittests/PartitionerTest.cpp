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
  PlaceholderBindings bindings_;
};

/// Execute a graph of functions based on the given DAG.
static void executeDAG(DAGNode *G, Module &mod, PlaceholderBindings &bindings,
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
      updateInputPlaceholders(bindings, vars, inputs);
      EE.run(bindings);
    }
    for (int i = 0, e = dag->children.size(); i < e; i++) {
      exeList.push_back(dag->children.at(i));
      endPt++;
    }
    curPt++;
  }
}

/// \returns true if all the functions have the valid save node format: i.e. no
/// such pattern Save->Save.
static bool checkSaveNode(Module &mod) {
  for (auto F : mod.getFunctions()) {
    for (const Node &N : F->getNodes()) {
      if (N.getKind() != Kinded::Kind::SaveNodeKind) {
        continue;
      }
      Placeholder *ph = llvm::dyn_cast<Placeholder>(N.getNthInput(0).getNode());
      if (!ph) {
        continue;
      }
      // If this SaveNode use the output of another SaveNode, it is an illegal
      // pattern.
      for (auto &user : ph->getUsers()) {
        if (user.getUser() == &N || !llvm::dyn_cast<SaveNode>(user.getUser())) {
          continue;
        }
        return false;
      }
    }
  }
  return true;
}

/// This one tests the model with this feature: after BFS, the memory
/// consumption of all the nodes in each level won't exceed the device memory
/// constraints.
TEST_F(PartitionerTest, Basic1) {
  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 32}, "input", false);
  auto *w1 = mod_.createConstant(ElemKind::FloatTy, {32, 16}, "w1");
  auto *b1 = mod_.createConstant(ElemKind::FloatTy, {16}, "b1");
  bindings_.allocate(input);
  w1->getHandle<>().randomize(-2.0, 2.0, mod_.getPRNG());
  b1->getHandle<>().randomize(-2.0, 2.0, mod_.getPRNG());

  // Initial FC.
  Node *I = F_->createFullyConnected("initial_fc", input, w1, b1);
  I = F_->createSigmoid("initial_sigmoid", I);

  // Left branch.
  auto *w2 = mod_.createConstant(ElemKind::FloatTy, {16, 16}, "w2");
  auto *b2 = mod_.createConstant(ElemKind::FloatTy, {16}, "b2");
  w2->getHandle<>().randomize(-2.0, 2.0, mod_.getPRNG());
  b2->getHandle<>().randomize(-2.0, 2.0, mod_.getPRNG());
  Node *L = F_->createFullyConnected("left_fc1", I, w2, b2);
  L = F_->createSigmoid("left_sigmoid1", L);
  auto *w3 = mod_.createConstant(ElemKind::FloatTy, {16, 8}, "w3");
  auto *b3 = mod_.createConstant(ElemKind::FloatTy, {8}, "b3");
  w3->getHandle<>().randomize(-2.0, 2.0, mod_.getPRNG());
  b3->getHandle<>().randomize(-2.0, 2.0, mod_.getPRNG());
  L = F_->createFullyConnected("left_fc2", L, w3, b3);
  L = F_->createSigmoid("left_sigmoid2", L);

  // Right branch.
  auto *w4 = mod_.createConstant(ElemKind::FloatTy, {16, 16}, "w4");
  auto *b4 = mod_.createConstant(ElemKind::FloatTy, {16}, "b4");
  w4->getHandle<>().randomize(-2.0, 2.0, mod_.getPRNG());
  b4->getHandle<>().randomize(-2.0, 2.0, mod_.getPRNG());
  Node *R = F_->createFullyConnected("right_fc1", I, w4, b4);
  R = F_->createSigmoid("right_sigmoid1", R);
  auto *w5 = mod_.createConstant(ElemKind::FloatTy, {16, 8}, "w5");
  auto *b5 = mod_.createConstant(ElemKind::FloatTy, {8}, "b5");
  w5->getHandle<>().randomize(-2.0, 2.0, mod_.getPRNG());
  b5->getHandle<>().randomize(-2.0, 2.0, mod_.getPRNG());
  R = F_->createFullyConnected("right_fc2", R, w5, b5);
  R = F_->createSigmoid("right_sigmoid2", R);

  // Join branches.
  auto *mul = F_->createMul("mul", L, R);
  auto *save = F_->createSave("ret", mul);
  auto &res = *bindings_.allocate(save->getPlaceholder());

  // Infer using the un-partitioned graph.
  Tensor in(ElemKind::FloatTy, {1, 32});
  ExecutionEngine EE;

  EE.compile(CompilationMode::Infer, F_);
  updateInputPlaceholders(bindings_, {input}, {&in});
  EE.run(bindings_);
  Tensor ref = res.clone();

  std::vector<DeviceInfo> devices = {{3072, BackendKind::Interpreter},
                                     {3072, BackendKind::Interpreter},
                                     {3072, BackendKind::Interpreter}};
  Partitioner myPartitioner(&mod_, devices, false, true);
  CompilationContext cctx;
  auto err = myPartitioner.Partition(cctx);
  EXPECT_FALSE(errToBool(std::move(err)));
  DAGListTy myList = std::move(myPartitioner.getPartitionResult());
  ASSERT_EQ(mod_.getFunctions().size(), 3);
  ASSERT_EQ(myList.size(), 1);
  ASSERT_TRUE(checkSaveNode(mod_));

  // Run the paritioned graph and compare the results.
  bindings_.allocate(mod_.getPlaceholders());
  for (auto it = myList.begin(); it != myList.end(); ++it) {
    bindings_.allocate(mod_.getPlaceholders());
    executeDAG((*it).root.get(), mod_, bindings_, {input}, {&in});
    Tensor test = res.clone();
    EXPECT_TRUE(ref.isEqual(test));
  }
}

/// This one tests the model with this feature: after BFS, there is one level,
/// the  memory comsumption of all the nodes in which exceeds the device memory
/// constraints.
TEST_F(PartitionerTest, Basic2) {
  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 16}, "input", false);
  auto *input1 =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 16}, "input1", false);
  bindings_.allocate(input);
  bindings_.allocate(input1);
  // Left branch.
  auto *w2 = mod_.createConstant(ElemKind::FloatTy, {16, 16}, "w2");
  auto *b2 = mod_.createConstant(ElemKind::FloatTy, {16}, "b2");
  w2->getHandle<>().randomize(-2.0, 2.0, mod_.getPRNG());
  b2->getHandle<>().randomize(-2.0, 2.0, mod_.getPRNG());
  Node *L = F_->createFullyConnected("left_fc1", input, w2, b2);
  L = F_->createSigmoid("left_sigmoid1", L);
  auto *w3 = mod_.createConstant(ElemKind::FloatTy, {16, 8}, "w3");
  auto *b3 = mod_.createConstant(ElemKind::FloatTy, {8}, "b3");
  w3->getHandle<>().randomize(-2.0, 2.0, mod_.getPRNG());
  b3->getHandle<>().randomize(-2.0, 2.0, mod_.getPRNG());
  L = F_->createFullyConnected("left_fc2", L, w3, b3);
  L = F_->createSigmoid("left_sigmoid2", L);

  // Right branch.
  auto *w4 = mod_.createConstant(ElemKind::FloatTy, {16, 16}, "w4");
  auto *b4 = mod_.createConstant(ElemKind::FloatTy, {16}, "b4");
  w4->getHandle<>().randomize(-2.0, 2.0, mod_.getPRNG());
  b4->getHandle<>().randomize(-2.0, 2.0, mod_.getPRNG());
  Node *R = F_->createFullyConnected("right_fc1", input1, w4, b4);
  R = F_->createSigmoid("right_sigmoid1", R);
  auto *w5 = mod_.createConstant(ElemKind::FloatTy, {16, 8}, "w5");
  auto *b5 = mod_.createConstant(ElemKind::FloatTy, {8}, "b5");
  w5->getHandle<>().randomize(-2.0, 2.0, mod_.getPRNG());
  b5->getHandle<>().randomize(-2.0, 2.0, mod_.getPRNG());
  R = F_->createFullyConnected("right_fc2", R, w5, b5);
  R = F_->createSigmoid("right_sigmoid2", R);

  // Join branches.
  auto *mul = F_->createMul("mul", L, R);
  auto *save = F_->createSave("ret", mul);
  auto &res = *bindings_.allocate(save->getPlaceholder());

  // Infer using the un-partitioned graph.
  Tensor in(ElemKind::FloatTy, {1, 16});
  ExecutionEngine EE;

  EE.compile(CompilationMode::Infer, F_);
  updateInputPlaceholders(bindings_, {input, input1}, {&in, &in});
  EE.run(bindings_);
  Tensor ref = res.clone();

  std::vector<DeviceInfo> devices = {{2048, BackendKind::Interpreter},
                                     {2048, BackendKind::Interpreter},
                                     {2048, BackendKind::Interpreter},
                                     {2048, BackendKind::Interpreter}};
  Partitioner myPartitioner(&mod_, devices, /* saturateHost */ true);
  CompilationContext cctx;
  auto err = myPartitioner.Partition(cctx);
  EXPECT_FALSE(errToBool(std::move(err)));
  DAGListTy myList = std::move(myPartitioner.getPartitionResult());
  ASSERT_EQ(mod_.getFunctions().size(), 2);
  ASSERT_EQ(myList.size(), 1);
  ASSERT_TRUE(checkSaveNode(mod_));

  for (auto &dag : myList) {
    for (auto &node : dag.nodes) {
      // Since saturateHost is set true, in this case, there should be 2 copys
      // of the partitions.
      ASSERT_EQ(node->logicalDevices.size(), 2);
    }
  }

  // Run the paritioned graph and compare the results.
  bindings_.allocate(mod_.getPlaceholders());
  for (auto it = myList.begin(); it != myList.end(); ++it) {
    bindings_.allocate(mod_.getPlaceholders());
    executeDAG((*it).root.get(), mod_, bindings_, {input}, {&in});
    Tensor test = res.clone();
    EXPECT_TRUE(ref.isEqual(test));
  }
}

/// This one tests the error msg: if the number of partitions is larger than
/// given number of devices, report an error.
TEST_F(PartitionerTest, Error1) {
  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 16}, "input", false);
  auto *input1 =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 16}, "input1", false);
  bindings_.allocate(input);
  bindings_.allocate(input1);
  // Left branch.
  auto *w2 = mod_.createConstant(ElemKind::FloatTy, {16, 16}, "w2");
  auto *b2 = mod_.createConstant(ElemKind::FloatTy, {16}, "b2");
  w2->getHandle<>().randomize(-2.0, 2.0, mod_.getPRNG());
  b2->getHandle<>().randomize(-2.0, 2.0, mod_.getPRNG());
  Node *L = F_->createFullyConnected("left_fc1", input, w2, b2);
  L = F_->createSigmoid("left_sigmoid1", L);
  auto *w3 = mod_.createConstant(ElemKind::FloatTy, {16, 8}, "w3");
  auto *b3 = mod_.createConstant(ElemKind::FloatTy, {8}, "b3");
  w3->getHandle<>().randomize(-2.0, 2.0, mod_.getPRNG());
  b3->getHandle<>().randomize(-2.0, 2.0, mod_.getPRNG());
  L = F_->createFullyConnected("left_fc2", L, w3, b3);
  L = F_->createSigmoid("left_sigmoid2", L);

  // Right branch.
  auto *w4 = mod_.createConstant(ElemKind::FloatTy, {16, 16}, "w4");
  auto *b4 = mod_.createConstant(ElemKind::FloatTy, {16}, "b4");
  w4->getHandle<>().randomize(-2.0, 2.0, mod_.getPRNG());
  b4->getHandle<>().randomize(-2.0, 2.0, mod_.getPRNG());
  Node *R = F_->createFullyConnected("right_fc1", input1, w4, b4);
  R = F_->createSigmoid("right_sigmoid1", R);
  auto *w5 = mod_.createConstant(ElemKind::FloatTy, {16, 8}, "w5");
  auto *b5 = mod_.createConstant(ElemKind::FloatTy, {8}, "b5");
  w5->getHandle<>().randomize(-2.0, 2.0, mod_.getPRNG());
  b5->getHandle<>().randomize(-2.0, 2.0, mod_.getPRNG());
  R = F_->createFullyConnected("right_fc2", R, w5, b5);
  R = F_->createSigmoid("right_sigmoid2", R);

  // Join branches.
  auto *mul = F_->createMul("mul", L, R);
  auto *save = F_->createSave("ret", mul);
  auto &res = *bindings_.allocate(save->getPlaceholder());

  // Infer using the un-partitioned graph.
  Tensor in(ElemKind::FloatTy, {1, 16});
  ExecutionEngine EE;

  EE.compile(CompilationMode::Infer, F_);
  updateInputPlaceholders(bindings_, {input, input1}, {&in, &in});
  EE.run(bindings_);
  Tensor ref = res.clone();

  std::vector<DeviceInfo> devices = {{2048}};
  Partitioner myPartitioner(&mod_, devices);
  CompilationContext cctx;
  auto err = myPartitioner.Partition(cctx);
  EXPECT_TRUE(errToBool(std::move(err)));
}

/// This one tests the roofline computed with compute, memory and communication
/// costs
TEST_F(PartitionerTest, Basic1Roofline) {
  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 32}, "input", false);
  auto *w1 = mod_.createConstant(ElemKind::FloatTy, {32, 16}, "w1");
  auto *b1 = mod_.createConstant(ElemKind::FloatTy, {16}, "b1");
  bindings_.allocate(input);
  w1->getHandle<>().randomize(-2.0, 2.0, mod_.getPRNG());
  b1->getHandle<>().randomize(-2.0, 2.0, mod_.getPRNG());

  // Initial FC.
  Node *I = F_->createFullyConnected("initial_fc", input, w1, b1);
  I = F_->createSigmoid("initial_sigmoid", I);

  // Left branch.
  auto *w2 = mod_.createConstant(ElemKind::FloatTy, {16, 16}, "w2");
  auto *b2 = mod_.createConstant(ElemKind::FloatTy, {16}, "b2");
  w2->getHandle<>().randomize(-2.0, 2.0, mod_.getPRNG());
  b2->getHandle<>().randomize(-2.0, 2.0, mod_.getPRNG());
  Node *L = F_->createFullyConnected("left_fc1", I, w2, b2);
  L = F_->createSigmoid("left_sigmoid1", L);
  auto *w3 = mod_.createConstant(ElemKind::FloatTy, {16, 8}, "w3");
  auto *b3 = mod_.createConstant(ElemKind::FloatTy, {8}, "b3");
  w3->getHandle<>().randomize(-2.0, 2.0, mod_.getPRNG());
  b3->getHandle<>().randomize(-2.0, 2.0, mod_.getPRNG());
  L = F_->createFullyConnected("left_fc2", L, w3, b3);
  L = F_->createSigmoid("left_sigmoid2", L);

  // Right branch.
  auto *w4 = mod_.createConstant(ElemKind::FloatTy, {16, 16}, "w4");
  auto *b4 = mod_.createConstant(ElemKind::FloatTy, {16}, "b4");
  w4->getHandle<>().randomize(-2.0, 2.0, mod_.getPRNG());
  b4->getHandle<>().randomize(-2.0, 2.0, mod_.getPRNG());
  Node *R = F_->createFullyConnected("right_fc1", I, w4, b4);
  R = F_->createSigmoid("right_sigmoid1", R);
  auto *w5 = mod_.createConstant(ElemKind::FloatTy, {16, 8}, "w5");
  auto *b5 = mod_.createConstant(ElemKind::FloatTy, {8}, "b5");
  w5->getHandle<>().randomize(-2.0, 2.0, mod_.getPRNG());
  b5->getHandle<>().randomize(-2.0, 2.0, mod_.getPRNG());
  R = F_->createFullyConnected("right_fc2", R, w5, b5);
  R = F_->createSigmoid("right_sigmoid2", R);

  // Join branches.
  auto *mul = F_->createMul("mul", L, R);
  auto *save = F_->createSave("ret", mul);
  auto &res = *bindings_.allocate(save->getPlaceholder());

  // Infer using the un-partitioned graph.
  Tensor in(ElemKind::FloatTy, {1, 32});
  ExecutionEngine EE;

  EE.compile(CompilationMode::Infer, F_);
  updateInputPlaceholders(bindings_, {input}, {&in});
  EE.run(bindings_);
  Tensor ref = res.clone();

  std::unordered_map<Node *, std::string> nodeNamesMap;
  for (auto &node : F_->getNodes()) {
    nodeNamesMap[&node] = node.getName();
  }

  std::vector<DeviceInfo> devices = {
      {3072, BackendKind::Interpreter, 100, 10, 0.1, 1, 0.05},
      {3072, BackendKind::Interpreter, 100, 10, 0.1, 1, 0.05},
      {3072, BackendKind::Interpreter, 100, 10, 0.1, 1, 0.05}};
  Partitioner myPartitioner(&mod_, devices);
  CompilationContext cctx;
  auto err = myPartitioner.Partition(cctx);
  EXPECT_FALSE(errToBool(std::move(err)));

  DAGListTy myList = std::move(myPartitioner.getPartitionResult());

  // check compute costs
  std::unordered_map<std::string, float> expectedComputeTime{
      {"initial_sigmoid", 128},
      {"left_sigmoid2", 64},
      {"fc_add_bias3", 192},
      {"right_sigmoid1", 128},
      {"mul", 96},
      {"fc_add_bias2", 96},
      {"ret", 0},
      {"fc_dot", 21760},
      {"left_sigmoid1", 128},
      {"fc_add_bias", 192},
      {"fc_dot1", 10240},
      {"right_sigmoid2", 64},
      {"fc_add_bias1", 192},
      {"fc_dot2", 5120},
      {"fc_dot3", 10240},
      {"fc_dot4", 5120},
      {"fc_add_bias4", 96},
  };
  ASSERT_EQ(myPartitioner.getComputeTime().size(), expectedComputeTime.size());
  for (auto &el : myPartitioner.getComputeTime()) {
    Node *n = el.first;
    float expected = expectedComputeTime[nodeNamesMap[n].c_str()];
    float res = el.second;
    ASSERT_EQ(expected, res);
  }

  ASSERT_EQ(mod_.getFunctions().size(), 3);
  ASSERT_EQ(myList.size(), 1);
}

TEST_F(PartitionerTest, SelectRepFunc) {
  auto *inA = mod_.createConstant(ElemKind::FloatTy, {2}, "A");
  auto *inB = mod_.createConstant(ElemKind::FloatTy, {2}, "B");
  inA->getHandle<>().randomize(-2.0, 2.0, mod_.getPRNG());
  inB->getHandle<>().randomize(-2.0, 2.0, mod_.getPRNG());

  auto *plus = F_->createAdd("AplusB", inA, inB);
  F_->createSave("save", plus);

  Partitioner myPartitioner(&mod_, {{1000000}, {1000000}, {1000000}});

  CompilationContext cctx;
  auto err = myPartitioner.Partition(cctx);
  EXPECT_FALSE(errToBool(std::move(err)));
}

/// Create a mock backend with \p kind, and rewrite the isOpSupported function
/// to un-support the op \p unsupportedOpKind.
template <BackendKind kind, glow::Kinded::Kind unsupportedOpKind>
class MockBackend : public Backend {
  class MockFunction : public CompiledFunction {
  public:
    MockFunction(const runtime::RuntimeBundle &bundle)
        : CompiledFunction(bundle) {}

    llvm::Error execute(ExecutionContext *) override {
      return llvm::Error::success();
    }

    BackendKind getCompileBackendKind() const override {
      return BackendKind::Interpreter;
    }
  };

  BackendKind getBackendKind() const override { return kind; }

  std::string getBackendName() const override { return "MockBackend"; }

  std::unique_ptr<CompiledFunction>
  compile(Function *F, const BackendOptions &opts) const override {
    return llvm::make_unique<MockFunction>(runtime::RuntimeBundle::create(*F));
  }

  bool isOpSupported(const NodeInfo &NI) const override {
    if (NI.getKind() == unsupportedOpKind) {
      return false;
    }
    return true;
  }

  bool shouldLower(const Node *N) const override { return false; }

  bool generateInst(Node *N, IRGenVisitor &irgen) const override {
    return false;
  }
};

class BackendWithoutSub
    : public MockBackend<BackendKind::CPU, Kinded::Kind::SubNodeKind> {};
class BackendWithoutMul
    : public MockBackend<BackendKind::Interpreter, Kinded::Kind::MulNodeKind> {
};

static void createSimpleModule(Module &mod) {
  mod.clear();
  auto *F = mod.createFunction("test");
  auto *input1 =
      mod.createPlaceholder(ElemKind::FloatTy, {16}, "input1", false);
  auto *input2 =
      mod.createPlaceholder(ElemKind::FloatTy, {16}, "input2", false);
  auto *sub = F->createSub("sub", input1, input2);
  auto *mul = F->createMul("mul", input1, input2);
  auto *sum = F->createMul("add", sub, mul);
  auto *save = F->createSave("ret", sum);
  (void)save;
}

TEST_F(PartitionerTest, SimpleHeterogeneousPartitioning) {
  {
    createSimpleModule(mod_);
    BackendWithoutSub backendWithoutSub1, backendWithoutSub2;
    BackendWithoutMul backendWithoutMul1, backendWithoutMul2;
    // Create two backends which support different ops, then do the partition by
    // assigning the ops to the corresponding abackends.
    std::vector<Backend *> backends;
    backends.emplace_back(&backendWithoutSub1);
    backends.emplace_back(&backendWithoutSub2);
    backends.emplace_back(&backendWithoutMul1);
    backends.emplace_back(&backendWithoutMul2);
    std::vector<DeviceInfo> devices = {{3072, BackendKind::CPU},
                                       {3072, BackendKind::CPU},
                                       {3072, BackendKind::Interpreter},
                                       {3072, BackendKind::Interpreter}};
    auto partitioner =
        Partitioner(&mod_, devices, backends, /* saturateHost */ true);
    CompilationContext cctx;
    auto err = partitioner.Partition(cctx);
    EXPECT_FALSE(errToBool(std::move(err)));
    DAGListTy myList = std::move(partitioner.getPartitionResult());
    ASSERT_EQ(mod_.getFunctions().size(), 2);
    ASSERT_EQ(myList.size(), 1);
    ASSERT_TRUE(checkSaveNode(mod_));

    for (auto &dag : myList) {
      for (auto &node : dag.nodes) {
        // Although the saturateHost is set true, no saturating the host in
        // heterogeneous partiton.
        ASSERT_EQ(node->logicalDevices.size(), 1);
      }
    }
    mod_.clear();
  }
}
