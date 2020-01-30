/**
 * Copyright (c) Glow Contributors. See CONTRIBUTORS file.
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
#include "glow/Optimizer/GraphOptimizer/GraphOptimizer.h"
#include "glow/Partitioner/PartitionerUtils.h"

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
                       llvm::ArrayRef<Tensor *> inputs, ExecutionEngine *EE) {
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
      updateInputPlaceholders(bindings, vars, inputs);
      EE->run(bindings, dag->name);
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
      auto *ph = llvm::dyn_cast<Placeholder>(N.getNthInput(0).getNode());
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
  ExecutionEngine EER, EEP;
  constexpr float range = 2.0;
  std::vector<ExecutionEngine *> engines{&EER, &EEP};
  // Since compiling modifies the module and partitioning modifies the function,
  // setup two EEs with identical functions for validation.
  for (auto EE : engines) {
    auto mod = &EE->getModule();
    F_ = mod->createFunction("main");
    auto *input =
        mod->createPlaceholder(ElemKind::FloatTy, {1, 32}, "input", false);
    auto *w1 = mod->createConstant(ElemKind::FloatTy, {32, 16}, "w1");
    auto *b1 = mod->createConstant(ElemKind::FloatTy, {16}, "b1");
    bindings_.allocate(input);
    w1->getHandle<>().randomize(-range, range, mod->getPRNG());
    b1->getHandle<>().randomize(-range, range, mod->getPRNG());

    // Initial FC.
    Node *I = F_->createFullyConnected("initial_fc", input, w1, b1);
    I = F_->createSigmoid("initial_sigmoid", I);

    // Left branch.
    auto *w2 = mod->createConstant(ElemKind::FloatTy, {16, 16}, "w2");
    auto *b2 = mod->createConstant(ElemKind::FloatTy, {16}, "b2");
    w2->getHandle<>().randomize(-range, range, mod->getPRNG());
    b2->getHandle<>().randomize(-range, range, mod->getPRNG());
    Node *L = F_->createFullyConnected("left_fc1", I, w2, b2);
    L = F_->createSigmoid("left_sigmoid1", L);
    auto *w3 = mod->createConstant(ElemKind::FloatTy, {16, 8}, "w3");
    auto *b3 = mod->createConstant(ElemKind::FloatTy, {8}, "b3");
    w3->getHandle<>().randomize(-range, range, mod->getPRNG());
    b3->getHandle<>().randomize(-range, range, mod->getPRNG());
    L = F_->createFullyConnected("left_fc2", L, w3, b3);
    L = F_->createSigmoid("left_sigmoid2", L);

    // Right branch.
    auto *w4 = mod->createConstant(ElemKind::FloatTy, {16, 16}, "w4");
    auto *b4 = mod->createConstant(ElemKind::FloatTy, {16}, "b4");
    w4->getHandle<>().randomize(-range, range, mod->getPRNG());
    b4->getHandle<>().randomize(-range, range, mod->getPRNG());
    Node *R = F_->createFullyConnected("right_fc1", I, w4, b4);
    R = F_->createSigmoid("right_sigmoid1", R);
    auto *w5 = mod->createConstant(ElemKind::FloatTy, {16, 8}, "w5");
    auto *b5 = mod->createConstant(ElemKind::FloatTy, {8}, "b5");
    w5->getHandle<>().randomize(-range, range, mod->getPRNG());
    b5->getHandle<>().randomize(-range, range, mod->getPRNG());
    R = F_->createFullyConnected("right_fc2", R, w5, b5);
    R = F_->createSigmoid("right_sigmoid2", R);

    // Join branches.
    auto *mul = F_->createMul("mul", L, R);
    F_->createSave("ret", mul);
  }

  // Infer using the un-partitioned graph.
  Tensor in(ElemKind::FloatTy, {1, 32});
  in.getHandle<>().randomize(-range, range, EER.getModule().getPRNG());

  EER.compile(CompilationMode::Infer);
  bindings_.clear();
  bindings_.allocate(EER.getModule().getPlaceholders());
  updateInputPlaceholders(bindings_, {bindings_.getPlaceholderByName("input")},
                          {&in});
  EER.run(bindings_);
  Tensor ref = bindings_.get(bindings_.getPlaceholderByName("ret"))->clone();

  std::vector<DeviceInfo> devices = {
      {3072, "Interpreter"}, {3072, "Interpreter"}, {3072, "Interpreter"}};
  Partitioner myPartitioner(&EEP.getModule(), devices, false, true);
  CompilationContext cctx;
  auto dagList = myPartitioner.partition(cctx);
  ASSERT_TRUE((bool)dagList);
  EXPECT_EQ(EEP.getModule().getFunctions().size(), 3);
  EXPECT_EQ(dagList->size(), 1);
  EXPECT_TRUE(checkSaveNode(EEP.getModule()));

  // Run the paritioned graph and compare the results.
  bindings_.clear();
  bindings_.allocate(EEP.getModule().getPlaceholders());
  EEP.compile(cctx);
  for (auto it = dagList->begin(); it != dagList->end(); ++it) {
    executeDAG((*it).root.get(), EEP.getModule(), bindings_,
               {bindings_.getPlaceholderByName("input")}, {&in}, &EEP);
    Tensor test = bindings_.get(bindings_.getPlaceholderByName("ret"))->clone();
    EXPECT_TRUE(ref.isEqual(test));
  }
}

/// This one tests the model with this feature: after BFS, there is one level,
/// the memory consumption of all the nodes in which exceeds the device memory
/// constraints.
TEST_F(PartitionerTest, Basic2) {

  ExecutionEngine EER, EEP;
  constexpr float range = 2.0;
  std::vector<ExecutionEngine *> engines{&EER, &EEP};
  for (auto EE : engines) {
    auto mod = &EE->getModule();
    F_ = mod->createFunction("main");
    auto *input =
        mod->createPlaceholder(ElemKind::FloatTy, {1, 16}, "input", false);
    auto *input1 =
        mod->createPlaceholder(ElemKind::FloatTy, {1, 16}, "input1", false);
    bindings_.allocate(input);
    bindings_.allocate(input1);
    // Left branch.
    auto *w2 = mod->createConstant(ElemKind::FloatTy, {16, 16}, "w2");
    auto *b2 = mod->createConstant(ElemKind::FloatTy, {16}, "b2");
    w2->getHandle<>().randomize(-range, range, mod->getPRNG());
    b2->getHandle<>().randomize(-range, range, mod->getPRNG());
    Node *L = F_->createFullyConnected("left_fc1", input, w2, b2);
    L = F_->createSigmoid("left_sigmoid1", L);
    auto *w3 = mod->createConstant(ElemKind::FloatTy, {16, 8}, "w3");
    auto *b3 = mod->createConstant(ElemKind::FloatTy, {8}, "b3");
    w3->getHandle<>().randomize(-range, range, mod->getPRNG());
    b3->getHandle<>().randomize(-range, range, mod->getPRNG());
    L = F_->createFullyConnected("left_fc2", L, w3, b3);
    L = F_->createSigmoid("left_sigmoid2", L);

    // Right branch.
    auto *w4 = mod->createConstant(ElemKind::FloatTy, {16, 16}, "w4");
    auto *b4 = mod->createConstant(ElemKind::FloatTy, {16}, "b4");
    w4->getHandle<>().randomize(-range, range, mod->getPRNG());
    b4->getHandle<>().randomize(-range, range, mod->getPRNG());
    Node *R = F_->createFullyConnected("right_fc1", input1, w4, b4);
    R = F_->createSigmoid("right_sigmoid1", R);
    auto *w5 = mod->createConstant(ElemKind::FloatTy, {16, 8}, "w5");
    auto *b5 = mod->createConstant(ElemKind::FloatTy, {8}, "b5");
    w5->getHandle<>().randomize(-range, range, mod->getPRNG());
    b5->getHandle<>().randomize(-range, range, mod->getPRNG());
    R = F_->createFullyConnected("right_fc2", R, w5, b5);
    R = F_->createSigmoid("right_sigmoid2", R);

    // Join branches.
    auto *mul = F_->createMul("mul", L, R);
    F_->createSave("ret", mul);
  }

  // Infer using the un-partitioned graph.
  Tensor in(ElemKind::FloatTy, {1, 16});
  in.getHandle<>().randomize(-range, range, EER.getModule().getPRNG());
  EER.compile(CompilationMode::Infer);
  bindings_.clear();
  bindings_.allocate(EER.getModule().getPlaceholders());
  updateInputPlaceholders(bindings_,
                          {bindings_.getPlaceholderByName("input"),
                           bindings_.getPlaceholderByName("input1")},
                          {&in, &in});
  EER.run(bindings_);
  Tensor ref = bindings_.get(bindings_.getPlaceholderByName("ret"))->clone();

  std::vector<DeviceInfo> devices = {{2048, "Interpreter"},
                                     {2048, "Interpreter"},
                                     {2048, "Interpreter"},
                                     {2048, "Interpreter"}};
  Partitioner myPartitioner(&EEP.getModule(), devices, /* saturateHost */ true);
  CompilationContext cctx;
  runtime::DAGListTy dagList;
  ASSIGN_VALUE_OR_FAIL_TEST(dagList, myPartitioner.partition(cctx));
  EXPECT_EQ(EEP.getModule().getFunctions().size(), 2);
  EXPECT_EQ(dagList.size(), 1);
  ASSERT_TRUE(checkSaveNode(EEP.getModule()));

  for (auto &dag : dagList) {
    for (auto &node : dag.nodes) {
      // Since saturateHost is set true, in this case, there should be 2 copys
      // of the partitions.
      EXPECT_EQ(node->logicalDevices.size(), 2);
    }
  }

  // Run the paritioned graph and compare the results.
  bindings_.clear();
  bindings_.allocate(EEP.getModule().getPlaceholders());
  EEP.compile(cctx);
  for (auto it = dagList.begin(); it != dagList.end(); ++it) {
    updateInputPlaceholders(bindings_,
                            {bindings_.getPlaceholderByName("input"),
                             bindings_.getPlaceholderByName("input1")},
                            {&in, &in});
    executeDAG((*it).root.get(), EEP.getModule(), bindings_,
               {bindings_.getPlaceholderByName("input")}, {&in}, &EEP);
    Tensor test = bindings_.get(bindings_.getPlaceholderByName("ret"))->clone();
    ASSERT_TRUE(ref.isEqual(test));
  }
}

/// This one tests the error msg: if the number of partitions is larger than
/// given number of devices, report an error.
TEST_F(PartitionerTest, Error1) {
  ExecutionEngine EER, EEP;
  constexpr float range = 2.0;
  std::vector<ExecutionEngine *> engines{&EER, &EEP};
  for (auto EE : engines) {
    auto mod = &EE->getModule();
    F_ = mod->createFunction("main");
    auto *input =
        mod->createPlaceholder(ElemKind::FloatTy, {1, 16}, "input", false);
    auto *input1 =
        mod->createPlaceholder(ElemKind::FloatTy, {1, 16}, "input1", false);
    bindings_.allocate(input);
    bindings_.allocate(input1);
    // Left branch.
    auto *w2 = mod->createConstant(ElemKind::FloatTy, {16, 16}, "w2");
    auto *b2 = mod->createConstant(ElemKind::FloatTy, {16}, "b2");
    w2->getHandle<>().randomize(-range, range, mod->getPRNG());
    b2->getHandle<>().randomize(-range, range, mod->getPRNG());
    Node *L = F_->createFullyConnected("left_fc1", input, w2, b2);
    L = F_->createSigmoid("left_sigmoid1", L);
    auto *w3 = mod->createConstant(ElemKind::FloatTy, {16, 8}, "w3");
    auto *b3 = mod->createConstant(ElemKind::FloatTy, {8}, "b3");
    w3->getHandle<>().randomize(-range, range, mod->getPRNG());
    b3->getHandle<>().randomize(-range, range, mod->getPRNG());
    L = F_->createFullyConnected("left_fc2", L, w3, b3);
    L = F_->createSigmoid("left_sigmoid2", L);

    // Right branch.
    auto *w4 = mod->createConstant(ElemKind::FloatTy, {16, 16}, "w4");
    auto *b4 = mod->createConstant(ElemKind::FloatTy, {16}, "b4");
    w4->getHandle<>().randomize(-range, range, mod->getPRNG());
    b4->getHandle<>().randomize(-range, range, mod->getPRNG());
    Node *R = F_->createFullyConnected("right_fc1", input1, w4, b4);
    R = F_->createSigmoid("right_sigmoid1", R);
    auto *w5 = mod->createConstant(ElemKind::FloatTy, {16, 8}, "w5");
    auto *b5 = mod->createConstant(ElemKind::FloatTy, {8}, "b5");
    w5->getHandle<>().randomize(-range, range, mod->getPRNG());
    b5->getHandle<>().randomize(-range, range, mod->getPRNG());
    R = F_->createFullyConnected("right_fc2", R, w5, b5);
    R = F_->createSigmoid("right_sigmoid2", R);

    // Join branches.
    auto *mul = F_->createMul("mul", L, R);
    F_->createSave("ret", mul);
  }

  // Infer using the un-partitioned graph.
  Tensor in(ElemKind::FloatTy, {1, 16});
  in.getHandle<>().randomize(-range, range, EER.getModule().getPRNG());

  EER.compile(CompilationMode::Infer);
  bindings_.clear();
  bindings_.allocate(EER.getModule().getPlaceholders());
  updateInputPlaceholders(bindings_,
                          {bindings_.getPlaceholderByName("input"),
                           bindings_.getPlaceholderByName("input1")},
                          {&in, &in});
  EER.run(bindings_);

  std::vector<DeviceInfo> devices = {{2048, "Interpreter"}};
  Partitioner myPartitioner(&EEP.getModule(), devices);
  CompilationContext cctx;
  auto dagList = myPartitioner.partition(cctx);
  EXPECT_TRUE(ERR_TO_BOOL(dagList.takeError()));
}

/// This one tests the roofline computed with compute, memory and
/// communication costs
TEST_F(PartitionerTest, Basic1Roofline) {
  ExecutionEngine EEP;
  constexpr float range = 2.0;

  auto mod = &EEP.getModule();
  F_ = mod->createFunction("main");
  auto *input =
      mod->createPlaceholder(ElemKind::FloatTy, {1, 32}, "input", false);
  auto *w1 = mod->createConstant(ElemKind::FloatTy, {32, 16}, "w1");
  auto *b1 = mod->createConstant(ElemKind::FloatTy, {16}, "b1");
  bindings_.allocate(input);
  w1->getHandle<>().randomize(-range, range, mod->getPRNG());
  b1->getHandle<>().randomize(-range, range, mod->getPRNG());

  // Initial FC.
  Node *I = F_->createFullyConnected("initial_fc", input, w1, b1);
  I = F_->createSigmoid("initial_sigmoid", I);

  // Left branch.
  auto *w2 = mod->createConstant(ElemKind::FloatTy, {16, 16}, "w2");
  auto *b2 = mod->createConstant(ElemKind::FloatTy, {16}, "b2");
  w2->getHandle<>().randomize(-range, range, mod->getPRNG());
  b2->getHandle<>().randomize(-range, range, mod->getPRNG());
  Node *L = F_->createFullyConnected("left_fc1", I, w2, b2);
  L = F_->createSigmoid("left_sigmoid1", L);
  auto *w3 = mod->createConstant(ElemKind::FloatTy, {16, 8}, "w3");
  auto *b3 = mod->createConstant(ElemKind::FloatTy, {8}, "b3");
  w3->getHandle<>().randomize(-range, range, mod->getPRNG());
  b3->getHandle<>().randomize(-range, range, mod->getPRNG());
  L = F_->createFullyConnected("left_fc2", L, w3, b3);
  L = F_->createSigmoid("left_sigmoid2", L);

  // Right branch.
  auto *w4 = mod->createConstant(ElemKind::FloatTy, {16, 16}, "w4");
  auto *b4 = mod->createConstant(ElemKind::FloatTy, {16}, "b4");
  w4->getHandle<>().randomize(-range, range, mod->getPRNG());
  b4->getHandle<>().randomize(-range, range, mod->getPRNG());
  Node *R = F_->createFullyConnected("right_fc1", I, w4, b4);
  R = F_->createSigmoid("right_sigmoid1", R);
  auto *w5 = mod->createConstant(ElemKind::FloatTy, {16, 8}, "w5");
  auto *b5 = mod->createConstant(ElemKind::FloatTy, {8}, "b5");
  w5->getHandle<>().randomize(-range, range, mod->getPRNG());
  b5->getHandle<>().randomize(-range, range, mod->getPRNG());
  R = F_->createFullyConnected("right_fc2", R, w5, b5);
  R = F_->createSigmoid("right_sigmoid2", R);

  // Join branches.
  auto *mul = F_->createMul("mul", L, R);
  F_->createSave("ret", mul);

  // Since the partitioner will look at all nodesin the function post
  // optimization and lowering, we need to do so here for the same list of
  // nodes.
  std::unique_ptr<Backend> backend(createBackend(EEP.getBackendName()));
  CompilationContext cctx;
  EXIT_ON_ERR(optimizeFunctionBeforeLowering(
      EEP.getModule().getFunction("main"), cctx));
  EXIT_ON_ERR(::glow::optimizeFunction(EEP.getModule().getFunction("main"),
                                       *backend, cctx));
  std::unordered_map<Node *, std::string> nodeNamesMap;
  for (auto &node : EEP.getModule().getFunction("main")->getNodes()) {
    nodeNamesMap[&node] = node.getName();
  }

  // check compute costs
  std::unordered_map<std::string, float> expectedComputeTime{
      {"initial_sigmoid", 128}, {"left_sigmoid2", 64},
      {"right_sigmoid1", 128},  {"mul", 96},
      {"ret_save", 0},          {"initial_fc", 21760},
      {"left_fc1", 10240},      {"left_fc2", 5120},
      {"left_sigmoid1", 128},   {"right_fc1", 10240},
      {"right_fc2", 5120},      {"right_sigmoid2", 64},
  };

  BackendInfo backendInfo;
  backendInfo.sramCapacity = 100;
  backendInfo.peakCompute = 10;
  backendInfo.peakDramBw = 0.1;
  backendInfo.peakSramBw = 1;
  backendInfo.peakPCIeBw = 0.05;
  for (auto const &p : nodeNamesMap) {
    auto *N = p.first;
    EXPECT_EQ(getNodeComputeTime(N, backendInfo),
              expectedComputeTime[p.second]);
  }
}

TEST_F(PartitionerTest, SelectRepFunc) {
  auto *inA = mod_.createConstant(ElemKind::FloatTy, {2}, "A");
  auto *inB = mod_.createConstant(ElemKind::FloatTy, {2}, "B");
  inA->getHandle<>().randomize(-2.0, 2.0, mod_.getPRNG());
  inB->getHandle<>().randomize(-2.0, 2.0, mod_.getPRNG());

  auto *plus = F_->createAdd("AplusB", inA, inB);
  F_->createSave("save", plus);

  Partitioner myPartitioner(&mod_, {{1000000, "Interpreter"},
                                    {1000000, "Interpreter"},
                                    {1000000, "Interpreter"}});

  CompilationContext cctx;
  auto dagList = myPartitioner.partition(cctx);
  ASSERT_TRUE((bool)dagList);
}

/// Create a mock backend and rewrite the isOpSupported function
/// to un-support the op \p unsupportedOpKind.
template <glow::Kinded::Kind unsupportedOpKind>
class MockBackend : public Backend {
public:
  std::string backendName;

  class MockFunction : public CompiledFunction {
  public:
    MockFunction(llvm::StringRef backendName, runtime::RuntimeBundle &&bundle)
        : CompiledFunction(std::move(bundle)), backendName(backendName) {}

    Error execute(ExecutionContext *) override { return Error::success(); }

    std::string getCompileBackendName() const override { return backendName; }

    std::string backendName;
  };

  std::string getBackendName() const override { return backendName; }

  Expected<std::unique_ptr<CompiledFunction>>
  compile(Function *F, const BackendOptions &opts) const override {
    return glow::make_unique<MockFunction>(backendName,
                                           runtime::RuntimeBundle::create(*F));
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

  runtime::DeviceManager *
  createDeviceManager(const runtime::DeviceConfig &deviceConfig) override {
    return nullptr;
  }
};

class BackendWithoutSub : public MockBackend<Kinded::Kind::SubNodeKind> {
public:
  BackendWithoutSub() { backendName = "CPU"; }
};
class BackendWithoutMul : public MockBackend<Kinded::Kind::MulNodeKind> {
public:
  BackendWithoutMul() { backendName = "Interpreter"; }
};

static void createSimpleModule(Module &mod) {
  mod.clear();
  auto *F = mod.createFunction("test");
  auto *input1 =
      mod.createPlaceholder(ElemKind::FloatTy, {16}, "input1", false);
  auto *input2 =
      mod.createPlaceholder(ElemKind::FloatTy, {16}, "input2", false);
  auto *input3 =
      mod.createPlaceholder(ElemKind::FloatTy, {16}, "input3", false);
  auto *sub = F->createSub("sub", input1, input2);
  auto *mul = F->createMul("mul", input1, input2);
  auto *sum = F->createAdd("add", sub, mul);
  auto *sub2 = F->createSub("sub1", sum, input3);
  auto *save = F->createSave("ret", sub2);
  (void)save;
}

static void createSimpleSparseNNModule(Module &mod, bool shareSplatWeights) {
  mod.clear();
  auto *F = mod.createFunction("test");

  // Create SLS inputs
  std::vector<NodeValue> slsOutputs;
  dim_t tableWidth = 16;
  dim_t numIndices = 80;
  dim_t batchSize = 32;
  dim_t tableEntries = 10;

  NodeValue weights;
  if (shareSplatWeights) {
    auto ty =
        F->getParent()->uniqueType(ElemKind::FloatTy, {numIndices * batchSize});
    weights = F->createSplat("ones", ty, 1.0)->getResult();
  }

  // Create SLS portion
  for (int table = 0; table < 5; table++) {
    Tensor data(ElemKind::FloatTy, {tableEntries, tableWidth});
    auto *indices = mod.createPlaceholder(
        IndexElemKind, {numIndices * batchSize}, "indices", false);
    if (!shareSplatWeights) {
      weights = mod.createPlaceholder(ElemKind::FloatTy,
                                      {numIndices * batchSize}, "w", false)
                    ->getOutput();
    }
    auto *lengths = mod.createPlaceholder(ElemKind::Int32ITy, {batchSize},
                                          "lengths", false);
    auto *output = F->createFusedRowwiseQuantizedSparseLengthsWeightedSum(
        "SLS", data, weights, indices, lengths, ElemKind::UInt8FusedQTy, false);
    slsOutputs.push_back(output);
  }

  // Create Concat
  auto *concat = F->createConcat("concat", slsOutputs, 1);
  Node *cur = (Node *)concat;

  // Create FC portion
  for (dim_t layer = 0; layer < 4; layer++) {
    Tensor FCWeights(ElemKind::FloatTy, {5 * tableWidth, 5 * tableWidth});
    Constant *weights = mod.createConstant("FCWeights", FCWeights);
    Tensor FCBias(ElemKind::FloatTy, {5 * tableWidth});
    Constant *bias = mod.createConstant("FCBias", FCBias);

    auto *FC = F->createFullyConnected("FC", cur, weights, bias);
    cur = (Node *)FC;
  }

  auto *save = F->createSave("ret", cur);
  (void)save;
}

/// \returns true if there is \p nodeKind kind of nodes in \p func.
static bool findNodeInFunction(const Function *func,
                               const Kinded::Kind nodeKind) {
  for (const Node &N : func->getNodes()) {
    if (N.getKind() == nodeKind) {
      return true;
    }
  }
  return false;
}

/// To check if the generated DAG is correct for the SparseNN Partiton
/// unnittests. The network used for check is generated from function static
/// void createSimpleSparseNNModule(Module &mod).
static void sparseNNPartitionValidation(const DAGListTy &dagList, Module &mod,
                                        bool shareSplatWeights) {
  int numOfCPUBackends = 0;
  int numOfSLSNodes = 0;
  int numOfFCNodes = 0;
  for (auto &dag : dagList) {
    for (auto &node : dag.nodes) {
      if (node->backendName == "CPU") {
        numOfCPUBackends++;
        auto *func = mod.getFunction(node->name);
        if (findNodeInFunction(
                func,
                Kinded::Kind::
                    FusedRowwiseQuantizedSparseLengthsWeightedSumNodeKind)) {
          numOfSLSNodes++;
          EXPECT_EQ(node->logicalDevices.size(), 1);
          if (shareSplatWeights) {
            for (const Node &N : func->getNodes()) {
              if (const auto *SLWS = llvm::dyn_cast<
                      FusedRowwiseQuantizedSparseLengthsWeightedSumNode>(&N)) {
                EXPECT_TRUE(llvm::isa<SplatNode>(SLWS->getWeights()));
              }
            }
          }
        } else {
          numOfFCNodes++;
          EXPECT_EQ(node->logicalDevices.size(), 3);
        }
      }
    }
  }
  // 4 partitions (3 SLS + 1 FC)
  EXPECT_EQ(numOfCPUBackends, 4);
  EXPECT_EQ(numOfSLSNodes, 3);
  EXPECT_EQ(numOfFCNodes, 1);
}

static void testSimpleSparseNNPartitioning(Module &mod,
                                           bool shareSplatWeights) {
  createSimpleSparseNNModule(mod, shareSplatWeights);
  BackendWithoutSub backend1, backend2, backend3;
  std::vector<Backend *> backends;
  backends.emplace_back(&backend1);
  backends.emplace_back(&backend2);
  backends.emplace_back(&backend3);
  std::vector<DeviceInfo> devices = {
      {400000, "CPU"}, {400000, "CPU"}, {400000, "CPU"}};
  Partitioner partitioner(&mod, devices, backends, /* saturateHost */ false);
  CompilationContext cctx;
  cctx.optimizationOpts.useSparseNNPartitioningScheme = true;
  cctx.optimizationOpts.sparseNNPartitioningSchemeNumCards = 3;
  cctx.optimizationOpts.sparseNNPartitioningSchemeSLSTableKBytesPerCard = 200;
  auto dagList = partitioner.partition(cctx);
  ASSERT_TRUE((bool)dagList);
  EXPECT_EQ(mod.getFunctions().size(), 4);
  EXPECT_EQ(dagList->size(), 1);
  ASSERT_TRUE(checkSaveNode(mod));
  sparseNNPartitionValidation(dagList.get(), mod, shareSplatWeights);
  mod.clear();
}

/// Test using user-defined backends for SparseNN partition.
TEST_F(PartitionerTest, SimpleSparseNNPartitioning) {
  testSimpleSparseNNPartitioning(mod_, /*shareSplatWeights*/ false);
}

/// Test using user-defined backends for SparseNN partition when weights are
/// shared Splats by all SLSs.
TEST_F(PartitionerTest, SimpleSparseNNPartitioning_SharedWeights) {
  testSimpleSparseNNPartitioning(mod_, /*shareSplatWeights*/ true);
}

/// To check if the generated DAG is correct for the Heterogeneous Partiton
/// unnittests. The network used for check is generated from function static
/// void createSimpleModule(Module &mod).
static void heterogeneousPartitionValidation(const DAGListTy &dagList,
                                             Module &mod) {
  int numOfInterpreterBackends = 0;
  int numOfCPUBackends = 0;
  int numOfMulNodes = 0;
  int numOfSubNodes = 0;
  for (auto &dag : dagList) {
    for (auto &node : dag.nodes) {
      // Although the saturateHost is set true, no saturating the host in
      // heterogeneous partiton.
      EXPECT_EQ(node->logicalDevices.size(), 1);
      if (node->backendName == "CPU") {
        numOfCPUBackends++;
        auto func = mod.getFunction(node->name);
        // Sub Node should not be assigned to CPU backend.
        EXPECT_FALSE(findNodeInFunction(func, Kinded::Kind::SubNodeKind));
        numOfMulNodes +=
            (findNodeInFunction(func, Kinded::Kind::MulNodeKind) == true);
      }
      if (node->backendName == "Interpreter") {
        numOfInterpreterBackends++;
        auto func = mod.getFunction(node->name);
        // Mul Node should not be assigned to Interpreter backend.
        EXPECT_FALSE(findNodeInFunction(func, Kinded::Kind::MulNodeKind));
        numOfSubNodes +=
            (findNodeInFunction(func, Kinded::Kind::SubNodeKind) == true);
      }
    }
  }
  EXPECT_EQ(numOfInterpreterBackends, 2);
  EXPECT_EQ(numOfCPUBackends, 1);
  EXPECT_EQ(numOfSubNodes, 2);
  EXPECT_EQ(numOfMulNodes, 1);
}

/// Test using user-defined backends for heterogeneous partition.
TEST_F(PartitionerTest, SimpleHeterogeneousPartitioning) {
  createSimpleModule(mod_);
  BackendWithoutSub backendWithoutSub1;
  BackendWithoutMul backendWithoutMul1, backendWithoutMul2;
  // Create two backends which support different ops, then do the partition by
  // assigning the ops to the corresponding abackends.
  std::vector<Backend *> backends;
  backends.emplace_back(&backendWithoutMul1);
  backends.emplace_back(&backendWithoutMul2);
  backends.emplace_back(&backendWithoutSub1);
  std::vector<DeviceInfo> devices = {
      {3072, "Interpreter"}, {3072, "Interpreter"}, {3072, "CPU"}};
  Partitioner partitioner(&mod_, devices, backends, /* saturateHost */ true);
  CompilationContext cctx;
  auto dagList = partitioner.partition(cctx);
  ASSERT_TRUE((bool)dagList);
  EXPECT_EQ(mod_.getFunctions().size(), 3);
  EXPECT_EQ(dagList->size(), 1);
  ASSERT_TRUE(checkSaveNode(mod_));
  heterogeneousPartitionValidation(dagList.get(), mod_);

  mod_.clear();
}

/// Test pre-defined non-supported ops used for choosing backend in
/// Heterogeneous Partition. In this test, "Mul" is not supported in
/// Interpreter backend, and "Sub" is not supported in CPU backend.
TEST_F(PartitionerTest, heterogeneousPartitioningWithNonSupportedNodes) {
#ifndef GLOW_WITH_CPU
  return;
#endif
  createSimpleModule(mod_);
  std::vector<DeviceInfo> devices = {{3072, "Interpreter", "Mul"},
                                     {3072, "Interpreter", "Mul"},
                                     {3072, "CPU", "Sub"}};
  Partitioner partitioner(&mod_, devices);
  CompilationContext cctx;
  auto dagList = partitioner.partition(cctx);
  ASSERT_TRUE((bool)dagList);
  EXPECT_EQ(mod_.getFunctions().size(), 3);
  EXPECT_EQ(dagList->size(), 1);
  ASSERT_TRUE(checkSaveNode(mod_));
  heterogeneousPartitionValidation(dagList.get(), mod_);

  mod_.clear();
}

/// Test pre-defined supported ops used for choosing backend in Heterogeneous
/// Partition. In this test, "Mul" is not supported in Interpreter backend,
/// and "Sub" is not supported in CPU backend. "Sub,Add,Save" can be supported
/// in Interpreter backend and "Mul,Add,Save" can be supported in CPU backend.
TEST_F(PartitionerTest, heterogeneousPartitioningWithSupportedNodes) {
#ifndef GLOW_WITH_CPU
  return;
#endif
  createSimpleModule(mod_);
  std::vector<DeviceInfo> devices = {
      // {memory size, backend, non-supported nodes, supported nodes}
      {3072, "Interpreter", "", "Sub,Add,Save"},
      {3072, "Interpreter", "", "Sub,Add,Save"},
      {3072, "CPU", "", "Mul,Add,Save"}};
  Partitioner partitioner(&mod_, devices);
  CompilationContext cctx;
  auto dagList = partitioner.partition(cctx);
  ASSERT_TRUE((bool)dagList);
  EXPECT_EQ(mod_.getFunctions().size(), 3);
  EXPECT_EQ(dagList->size(), 1);
  ASSERT_TRUE(checkSaveNode(mod_));
  heterogeneousPartitionValidation(dagList.get(), mod_);

  mod_.clear();
}

/// Test assigning more than one partitions in to one device for single
/// backendName.
TEST_F(PartitionerTest, logicalIDTest0) {
  auto *input1 =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 10}, "input1", false);
  auto *input2 =
      mod_.createPlaceholder(ElemKind::FloatTy, {10, 16}, "input2", false);
  auto *input3 =
      mod_.createPlaceholder(ElemKind::FloatTy, {16, 20}, "input3", false);
  auto *input4 =
      mod_.createPlaceholder(ElemKind::FloatTy, {20, 1}, "input4", false);
  auto *input5 =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 50}, "input5", false);
  auto *mul0 = F_->createMatMul("mul0", input1, input2);
  auto *mul1 = F_->createMatMul("mul1", mul0, input3);
  auto *mul2 = F_->createMatMul("mul2", mul1, input4);
  auto *mul3 = F_->createMatMul("mul3", mul2, input5);
  auto *save = F_->createSave("ret", mul3);
  (void)save;
  std::vector<DeviceInfo> devices = {{1500, "Interpreter"},
                                     {1500, "Interpreter"}};
  // Create two backends which support different ops, then do the partition by
  // assigning the ops to the corresponding abackends.
  Partitioner partitioner(&mod_, devices, /* saturateHost */ true);
  CompilationContext cctx;
  auto dagList = partitioner.partition(cctx);
  ASSERT_TRUE((bool)dagList);
  // Check there are 3 partitions.
  EXPECT_EQ(mod_.getFunctions().size(), 3);
  EXPECT_EQ(dagList->size(), 1);
  ASSERT_TRUE(checkSaveNode(mod_));

  for (auto &dag : dagList.get()) {
    // Check number of logical devices;
    llvm::SmallSet<DeviceIDTy, 4> usedID;
    for (auto &node : dag.nodes) {
      EXPECT_EQ(node->logicalDevices.size(), 1);
      usedID.insert(node->logicalDevices[0]);
    }
    // Check there are 2 logical devices.
    EXPECT_EQ(usedID.size(), 2);
  }
  mod_.clear();
}

/// Test assigning more than one partitions in to one device in Heterogeneous
/// partition.
TEST_F(PartitionerTest, logicalIDTest1) {
  createSimpleModule(mod_);
  BackendWithoutSub backendWithoutSub1, backendWithoutSub2;
  BackendWithoutMul backendWithoutMul1, backendWithoutMul2;
  // Create two backends which support different ops, then do the partition by
  // assigning the ops to the corresponding abackends.
  std::vector<Backend *> backends;
  backends.emplace_back(&backendWithoutMul1);
  backends.emplace_back(&backendWithoutSub1);
  std::vector<DeviceInfo> devices = {{3072, "Interpreter"}, {3072, "CPU"}};
  Partitioner partitioner(&mod_, devices, backends, /* saturateHost */ true);
  CompilationContext cctx;
  auto dagList = partitioner.partition(cctx);
  ASSERT_TRUE((bool)dagList);
  EXPECT_EQ(mod_.getFunctions().size(), 3);
  EXPECT_EQ(dagList->size(), 1);
  ASSERT_TRUE(checkSaveNode(mod_));

  for (auto &dag : dagList.get()) {
    // Check number of logical devices;
    llvm::SmallSet<DeviceIDTy, 4> usedID;
    for (auto &node : dag.nodes) {
      // Although the saturateHost is set true, no saturating the host in
      // heterogeneous partiton.
      EXPECT_EQ(node->logicalDevices.size(), 1);
      usedID.insert(node->logicalDevices[0]);
    }
    EXPECT_EQ(usedID.size(), 2);
  }
  mod_.clear();
}

/// Check the function getGraphMemInfo and updateGraphMemInfo to handle more
/// than one outputs of a single Node in PartitionerUtils.cpp
TEST_F(PartitionerTest, graphMemInfoCalculation1) {
  // TODO: The values are too large when dim_t is 32b. Figure out how it's
  // computed and ensure it's computed correctly.
  if (DIM_T_BITWIDTH == 32)
    return;
  auto *inp1 =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 1, 3}, "input", false);
  auto *inp2 =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 1, 3}, "input", false);
  auto *indices =
      mod_.createPlaceholder(IndexElemKind, {4, 1, 2}, "indices", false);

  auto *R1 = F_->createTopK("TopK1", inp1, 2);
  auto *R2 = F_->createTopK("TopK2", inp2, 2);

  // Concat the values and indices separately, both on the 0th dimension,
  // matching the shapes of the values and indices variables above.
  auto *CV =
      F_->createConcat("Concat.Values", {R1->getValues(), R2->getValues()}, 0);
  auto *CI = F_->createConcat("Concat.Indices",
                              {R1->getIndices(), R2->getIndices()}, 0);

  auto *saveValues = F_->createSave("Save.Values", CV);
  auto *saveIndices = F_->createSave("Save.Indices", CI, indices);

  std::set<Node *> nodes1;
  GraphMemInfo res;
  res = updateGraphMemInfoByAddingNode(nodes1, res, R1);
  EXPECT_EQ(res, GraphMemInfo(24, 48, 0));
  nodes1.insert(R1);

  res = updateGraphMemInfoByAddingNode(nodes1, res, R2);
  EXPECT_EQ(res, GraphMemInfo(48, 96, 0));
  nodes1.insert(R2);

  res = updateGraphMemInfoByAddingNode(nodes1, res, CV);
  EXPECT_EQ(res, GraphMemInfo(48, 96, 0));
  nodes1.insert(CV);

  res = updateGraphMemInfoByAddingNode(nodes1, res, CI);
  EXPECT_EQ(res, GraphMemInfo(48, 96, 0));
  nodes1.insert(CI);

  res = updateGraphMemInfoByAddingNode(nodes1, res, saveValues);
  EXPECT_EQ(res, GraphMemInfo(48, 96, 0));
  nodes1.insert(saveValues);

  res = updateGraphMemInfoByAddingNode(nodes1, res, saveIndices);
  EXPECT_EQ(res, GraphMemInfo(48, 96, 0));
  nodes1.insert(saveIndices);

  std::set<Node *> nodes2, nodes3;
  nodes2.insert(R1);
  nodes2.insert(R2);
  nodes3.insert(CV);
  nodes3.insert(CI);
  nodes3.insert(saveValues);
  nodes3.insert(saveIndices);
  GraphMemInfo res1 = getGraphMemInfo(nodes2);
  GraphMemInfo res2 = getGraphMemInfo(nodes3);
  GraphMemInfo ref1(48, 96, 0);
  GraphMemInfo ref2(96, 96, 0);
  EXPECT_EQ(res1, ref1);
  EXPECT_EQ(res2, ref2);
}

/// Check the function updateGraphMemInfoByAddingNode and getGraphMemInfo to
/// handle shared Storage node in PartitionerUtils.cpp
TEST_F(PartitionerTest, graphMemInfoCalculation2) {
  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 16}, "input", false);

  // Left branch.
  auto *w2 = mod_.createConstant(ElemKind::FloatTy, {16, 16}, "w2");
  auto *b2 = mod_.createConstant(ElemKind::FloatTy, {16}, "b2");
  auto *L = F_->createFullyConnected("left_fc1", input, w2, b2);
  auto *L1 = F_->createSigmoid("left_sigmoid1", L);
  auto *w3 = mod_.createConstant(ElemKind::FloatTy, {16, 8}, "w3");
  auto *b3 = mod_.createConstant(ElemKind::FloatTy, {8}, "b3");
  auto *L2 = F_->createFullyConnected("left_fc2", L1, w3, b3);
  auto *L3 = F_->createSigmoid("left_sigmoid2", L2);

  // Right branch.
  auto *R = F_->createFullyConnected("right_fc1", input, w2, b2);
  auto *R1 = F_->createSigmoid("right_sigmoid1", R);
  auto *w5 = mod_.createConstant(ElemKind::FloatTy, {16, 8}, "w5");
  auto *b5 = mod_.createConstant(ElemKind::FloatTy, {8}, "b5");
  auto *R2 = F_->createFullyConnected("right_fc2", R1, w5, b5);
  auto *R3 = F_->createSigmoid("right_sigmoid2", R2);

  // Join branches.
  auto *mul = F_->createMul("mul", L3, R3);
  auto *save = F_->createSave("ret", mul);

  std::set<Node *> nodes1, nodes2;
  GraphMemInfo res1, res2;
  res1 = updateGraphMemInfoByAddingNode(nodes1, res1, L);
  EXPECT_EQ(res1, GraphMemInfo(64, 64, 1088));
  nodes1.insert(L);

  res1 = updateGraphMemInfoByAddingNode(nodes1, res1, R);
  EXPECT_EQ(res1, GraphMemInfo(64, 128, 1088));
  nodes1.insert(R);

  res1 = updateGraphMemInfoByAddingNode(nodes1, res1, R1);
  EXPECT_EQ(res1, GraphMemInfo(64, 128, 1088));
  nodes1.insert(R1);

  res1 = updateGraphMemInfoByAddingNode(nodes1, res1, R2);
  EXPECT_EQ(res1, GraphMemInfo(64, 96, 1632));
  nodes1.insert(R2);

  res1 = getGraphMemInfo(nodes1);
  EXPECT_EQ(res1, GraphMemInfo(64, 96, 1632));

  res2 = updateGraphMemInfoByAddingNode(nodes2, res2, L1);
  EXPECT_EQ(res2, GraphMemInfo(64, 64, 0));
  nodes2.insert(L1);

  res2 = updateGraphMemInfoByAddingNode(nodes2, res2, L2);
  EXPECT_EQ(res2, GraphMemInfo(64, 32, 544));
  nodes2.insert(L2);

  res2 = updateGraphMemInfoByAddingNode(nodes2, res2, L3);
  EXPECT_EQ(res2, GraphMemInfo(64, 32, 544));
  nodes2.insert(L3);

  res2 = updateGraphMemInfoByAddingNode(nodes2, res2, mul);
  EXPECT_EQ(res2, GraphMemInfo(96, 32, 544));
  nodes2.insert(mul);

  res2 = updateGraphMemInfoByAddingNode(nodes2, res2, R3);
  EXPECT_EQ(res2, GraphMemInfo(96, 32, 544));
  nodes2.insert(R3);

  res2 = updateGraphMemInfoByAddingNode(nodes2, res2, save);
  EXPECT_EQ(res2, GraphMemInfo(96, 32, 544));
  nodes2.insert(save);

  res2 = getGraphMemInfo(nodes2);
  EXPECT_EQ(res2, GraphMemInfo(96, 32, 544));
}

/// Check the function getFunctionMemory in PartitionerUtils.cpp to compute
/// memory consumption of a simple function with same inputs used for multiple
/// nodes.
TEST_F(PartitionerTest, funcMemInfoCalculation1) {
  auto *input1 =
      mod_.createPlaceholder(ElemKind::FloatTy, {16}, "input1", false);
  auto *input2 =
      mod_.createPlaceholder(ElemKind::FloatTy, {16}, "input2", false);
  auto *input3 =
      mod_.createPlaceholder(ElemKind::FloatTy, {16}, "input3", false);
  auto *sub = F_->createSub("sub", input1, input2);
  auto *mul = F_->createMul("mul", input1, input2);
  auto *sum = F_->createAdd("add", sub, mul);
  auto *sub2 = F_->createSub("sub1", sum, input3);
  auto *save = F_->createSave("ret", sub2);
  (void)save;

  GraphMemInfo info = getFunctionMemory(F_);
  // 3x input Tensors of 16 fp32 each and 1x output of 16 fp32 values.
  GraphMemInfo res(192, 64, 0);
  EXPECT_EQ(res, info);
}

/// Check the function getFunctionMemory in PartitionerUtils.cpp to compute
/// memory consumption of a function with constants.
TEST_F(PartitionerTest, funcMemInfoCalculation2) {
  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 16}, "input", false);

  // Left branch.
  auto *w2 = mod_.createConstant(ElemKind::FloatTy, {16, 16}, "w2");
  auto *b2 = mod_.createConstant(ElemKind::FloatTy, {16}, "b2");
  auto *L = F_->createFullyConnected("left_fc1", input, w2, b2);
  auto *L1 = F_->createSigmoid("left_sigmoid1", L);
  auto *w3 = mod_.createConstant(ElemKind::FloatTy, {16, 8}, "w3");
  auto *b3 = mod_.createConstant(ElemKind::FloatTy, {8}, "b3");
  auto *L2 = F_->createFullyConnected("left_fc2", L1, w3, b3);
  auto *L3 = F_->createSigmoid("left_sigmoid2", L2);

  // Right branch.
  auto *R = F_->createFullyConnected("right_fc1", input, w2, b2);
  auto *R1 = F_->createSigmoid("right_sigmoid1", R);
  auto *w5 = mod_.createConstant(ElemKind::FloatTy, {16, 8}, "w5");
  auto *b5 = mod_.createConstant(ElemKind::FloatTy, {8}, "b5");
  auto *R2 = F_->createFullyConnected("right_fc2", R1, w5, b5);
  auto *R3 = F_->createSigmoid("right_sigmoid2", R2);

  // Join branches.
  auto *mul = F_->createMul("mul", L3, R3);
  auto *save = F_->createSave("ret", mul);
  (void)save;

  GraphMemInfo info = getFunctionMemory(F_);
  // single input tensor (1*16) fp32
  // single output tensor (1*8) fp32
  // constants fp32 (16*16 + 16 + 16*8 + 8 + 16*8 + 8)
  GraphMemInfo res(64, 32, 2176);
  EXPECT_EQ(res, info);
}

/// Check the function getFunctionMemory in PartitionerUtils.cpp to compute
/// memory consumption of a function with same inputs used for multiple
/// nodes.
TEST_F(PartitionerTest, funcMemInfoCalculation3) {
  auto *input1 =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 10}, "input1", false);
  auto *input2 =
      mod_.createPlaceholder(ElemKind::FloatTy, {10, 16}, "input2", false);
  auto *input3 =
      mod_.createPlaceholder(ElemKind::FloatTy, {16, 20}, "input3", false);
  auto *input4 =
      mod_.createPlaceholder(ElemKind::FloatTy, {20, 1}, "input4", false);
  auto *input5 =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 50}, "input5", false);
  auto *mul0 = F_->createMatMul("mul0", input1, input2);
  auto *mul1 = F_->createMatMul("mul1", mul0, input3);
  auto *mul2 = F_->createMatMul("mul2", mul1, input4);
  auto *mul3 = F_->createMatMul("mul3", mul2, input5);
  auto *save = F_->createSave("ret", mul3);
  (void)save;

  GraphMemInfo info = getFunctionMemory(F_);
  // input consists of 5 tensors (2*10 + 10*16 + 16*20 + 20*1 + 1*50 = 570) fp32
  // output is tensor of 2*50 = 100 fp32
  GraphMemInfo res(2280, 400, 0);
  EXPECT_EQ(res, info);
}

/// This one test the memoryUsageValidation in Partitioner : the memory usage
/// of one single node is larger than the given device memory.
TEST_F(PartitionerTest, memoryUsageValidation1) {
  auto *input1 =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 10}, "input1", false);
  auto *input2 =
      mod_.createPlaceholder(ElemKind::FloatTy, {10, 16}, "input2", false);
  auto *mul0 = F_->createMatMul("mul0", input1, input2);
  F_->createSave("ret", mul0);

  std::vector<DeviceInfo> devices = {{500, "Interpreter"},
                                     {500, "Interpreter"}};
  Partitioner myPartitioner(&mod_, devices);
  CompilationContext cctx;
  auto dagList = myPartitioner.partition(cctx);
  EXPECT_TRUE(ERR_TO_BOOL(dagList.takeError()));
}

/// This one test dagValidation in partitioner : p1->p2, p2->p1.
TEST_F(PartitionerTest, dagValidationWithBackendHints) {
  auto *input1 =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 10}, "input1", false);
  auto *input2 =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 10}, "input2", false);
  auto *input3 =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 10}, "input3", false);
  auto *add1 = F_->createAdd("add1", input1, input2);
  auto *add2 = F_->createAdd("add2", add1, input3);
  auto *sub1 = F_->createSub("sub1", add1, add2);
  F_->createSave("save", sub1);

  std::vector<DeviceInfo> devices = {{3072, "Interpreter"},
                                     {3072, "Interpreter"}};

  // User-defined partition: p1->p2, p2->p1.
  PartitionConfig partitionConfig;
  partitionConfig.funcName = "main";
  partitionConfig.numOfPartitions = 2;
  BackendHints bh1, bh2;
  bh1.executionUnits = 2;
  bh2.executionUnits = 3;
  partitionConfig.backendHints = {bh1, bh2};
  partitionConfig.backendNames = {"Interpreter", "Interpreter"};
  partitionConfig.partitionNames = {"p1", "p2"};
  partitionConfig.nodeToPartition = {{"add2", 0}};
  auto partitioner = Partitioner(&mod_, devices, false, false, partitionConfig);
  CompilationContext cctx;
  auto dagList = partitioner.partition(cctx);
  EXPECT_TRUE(ERR_TO_BOOL(dagList.takeError()));
}

/// This one test dagValidation in partitioner : p1->p2, p2->p1.
TEST_F(PartitionerTest, dagValidation1) {
  auto *input1 =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 10}, "input1", false);
  auto *input2 =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 10}, "input2", false);
  auto *input3 =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 10}, "input3", false);
  auto *add1 = F_->createAdd("add1", input1, input2);
  auto *add2 = F_->createAdd("add2", add1, input3);
  auto *sub1 = F_->createSub("sub1", add1, add2);
  F_->createSave("save", sub1);

  std::vector<DeviceInfo> devices = {{3072, "Interpreter"},
                                     {3072, "Interpreter"}};

  // User-defined partition: p1->p2, p2->p1.
  PartitionConfig partitionConfig;
  partitionConfig.funcName = "main";
  partitionConfig.numOfPartitions = 2;
  partitionConfig.backendNames = {"Interpreter", "Interpreter"};
  partitionConfig.partitionNames = {"p1", "p2"};
  partitionConfig.nodeToPartition = {{"add2", 0}};
  auto partitioner = Partitioner(&mod_, devices, false, false, partitionConfig);
  CompilationContext cctx;
  auto dagList = partitioner.partition(cctx);
  EXPECT_TRUE(ERR_TO_BOOL(dagList.takeError()));
}

/// This one test dagValidation in partitioner: p0->p1, p1->p2, p2->p1.
TEST_F(PartitionerTest, dagValidation2) {
  auto *input1 =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 10}, "input1", false);
  auto *input2 =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 10}, "input2", false);
  auto *input3 =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 10}, "input3", false);
  auto *input4 =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 10}, "input4", false);
  auto *add0 = F_->createAdd("add0", input1, input2);
  auto *add1 = F_->createAdd("add1", add0, input3);
  auto *add2 = F_->createAdd("add2", add1, input4);
  auto *sub1 = F_->createSub("sub1", add1, add2);
  F_->createSave("save", sub1);

  std::vector<DeviceInfo> devices = {
      {3072, "Interpreter"}, {3072, "Interpreter"}, {3072, "Interpreter"}};

  // User-defined partition: p0->p1, p1->p2, p2->p1.
  PartitionConfig partitionConfig;
  partitionConfig.funcName = "main";
  partitionConfig.numOfPartitions = 3;
  partitionConfig.backendNames = {"Interpreter", "Interpreter", "Interpreter"};
  partitionConfig.partitionNames = {"p0", "p1", "p2"};
  partitionConfig.nodeToPartition = {{"add0", 0}, {"add2", 2}};
  auto partitioner = Partitioner(&mod_, devices, false, false, partitionConfig);
  CompilationContext cctx;
  auto dagList = partitioner.partition(cctx);
  EXPECT_TRUE(ERR_TO_BOOL(dagList.takeError()));
}

/// This one tests partition from a user-defined config.
TEST_F(PartitionerTest, partitionFromConfig) {
#ifndef GLOW_WITH_CPU
  return;
#endif
  createSimpleModule(mod_);
  std::vector<DeviceInfo> devices = {
      {3072, "Interpreter"}, {3072, "Interpreter"}, {3072, "CPU"}};

  // User-defined partition: 3 partitions (2 interpreter, 1 cpu), Mul nodes to
  // CPU, others to Interpreter.
  PartitionConfig partitionConfig;
  partitionConfig.funcName = "test";
  partitionConfig.numOfPartitions = 3;
  partitionConfig.backendNames = {"Interpreter", "CPU", "Interpreter"};
  partitionConfig.partitionNames = {"p1", "p2", "p3"};
  partitionConfig.nodeToPartition = {{"sub", 0}, {"mul", 1}};
  Partitioner partitioner(&mod_, devices, false, false, partitionConfig);
  CompilationContext cctx;
  auto dagList = partitioner.partition(cctx);
  ASSERT_TRUE((bool)dagList);
  EXPECT_EQ(mod_.getFunctions().size(), 3);
  EXPECT_EQ(dagList->size(), 1);
  ASSERT_TRUE(checkSaveNode(mod_));
  heterogeneousPartitionValidation(dagList.get(), mod_);
}

/// Test user-defined partition with user specified logical devices through
/// compilationContext.
TEST_F(PartitionerTest, partitionFromConfigWithLogicalDevices) {
  auto *input1 =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 10}, "input1", false);
  auto *input2 =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 10}, "input2", false);
  auto *input3 =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 10}, "input3", false);
  auto *add1 = F_->createAdd("add1", input1, input2);
  auto *add2 = F_->createAdd("add2", add1, input3);
  auto *sub1 = F_->createSub("sub1", add1, add2);
  F_->createSave("save", sub1);

  std::vector<DeviceInfo> devices = {
      {3072, "Interpreter"}, {3072, "Interpreter"}, {3072, "Interpreter"}};

  // User-defined partition: p0->p1, p1->p2, p2->p1.
  PartitionConfig partitionConfig;
  partitionConfig.funcName = "main";
  partitionConfig.numOfPartitions = 3;
  partitionConfig.backendNames = {"Interpreter", "Interpreter", "Interpreter"};
  partitionConfig.partitionNames = {"p0", "p1", "p2"};
  partitionConfig.nodeToPartition = {{"add1", 0}, {"add2", 2}};
  partitionConfig.logicalIDs = {{0}, {1}, {0, 1}};
  auto partitioner = Partitioner(&mod_, devices, /*SaturateHost*/ false);
  CompilationContext cctx;
  cctx.partitionConfig = &partitionConfig;
  auto result = partitioner.partition(cctx);
  DAGListTy nodeList;
  EXPECT_FALSE(ERR_TO_BOOL(result.takeError()));
  nodeList = std::move(result.get());
  // Check that p2 has both 0 and 1 for logicalDevices.
  EXPECT_EQ(nodeList[0].nodes[2]->logicalDevices[0], 0);
  EXPECT_EQ(nodeList[0].nodes[2]->logicalDevices[1], 1);
}

/// Test user-defined partition with user specified logical devices through
/// compilationContext using fp16.
TEST_F(PartitionerTest, partitionFromConfigWithLogicalDevicesFp16) {
  auto *input1 =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 10}, "input1", false);
  auto *input2 =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 10}, "input2", false);
  auto *input3 =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 10}, "input3", false);
  auto *add1 = F_->createAdd("add1", input1, input2);
  auto *add2 = F_->createAdd("add2", add1, input3);
  auto *sub1 = F_->createSub("sub1", add1, add2);
  F_->createSave("save", sub1);

  std::vector<DeviceInfo> devices = {
      {3072, "Interpreter"}, {3072, "Interpreter"}, {3072, "Interpreter"}};

  // User-defined partition: p0->p1, p1->p2, p2->p1.
  PartitionConfig partitionConfig;
  partitionConfig.funcName = "main";
  partitionConfig.numOfPartitions = 3;
  partitionConfig.backendNames = {"Interpreter", "Interpreter", "Interpreter"};
  partitionConfig.partitionNames = {"p0", "p1", "p2"};
  partitionConfig.nodeToPartition = {{"add1", 0}, {"add2", 2}};
  partitionConfig.logicalIDs = {{0}, {1}, {0, 1}};
  auto partitioner = Partitioner(&mod_, devices, /*SaturateHost*/ false);
  CompilationContext cctx;
  cctx.partitionConfig = &partitionConfig;
  PrecisionConfiguration pc;
  pc.convertToFP16 = true;
  cctx.precisionConfig = pc;
  auto result = partitioner.partition(cctx);

  // Do optimization
  for (dim_t i = 0; i < partitionConfig.numOfPartitions; i++) {
    for (auto &func : mod_.getFunctions()) {
      std::unique_ptr<Backend> backend(createBackend("Interpreter"));
      auto err = ::glow::optimizeFunction(func, *backend, cctx);
      EXPECT_FALSE(err);
    }
  }

  DAGListTy nodeList;
  EXPECT_FALSE(ERR_TO_BOOL(result.takeError()));
  nodeList = std::move(result.get());
  // Check that p2 has both 0 and 1 for logicalDevices.
  EXPECT_EQ(nodeList[0].nodes[2]->logicalDevices[0], 0);
  EXPECT_EQ(nodeList[0].nodes[2]->logicalDevices[1], 1);
  // Check that the inputs and outputs of add1, add2 and sub1 are in fp16
  for (auto const &F : mod_.getFunctions()) {
    for (auto const &N : F->getNodes()) {
      auto NI = NodeInfo(N);
      if (NI.getKind() != Kinded::Kind::SaveNodeKind &&
          NI.getKind() != Kinded::Kind::ConvertToNodeKind) {
        EXPECT_TRUE(
            NI.allInputsAndOutputsHaveSameElemKind({ElemKind::Float16Ty}));
      }
    }
  }
}

/// This one tests calling PartitionFromConfig directly.
TEST_F(PartitionerTest, partitionFromConfigDirectCall) {
#ifndef GLOW_WITH_CPU
  return;
#endif
  createSimpleModule(mod_);
  std::vector<DeviceInfo> devices = {
      {3072, "Interpreter"}, {3072, "Interpreter"}, {3072, "CPU"}};

  // User-defined partition: 3 partitions (2 interpreter, 1 cpu), Mul nodes to
  // CPU, others to Interpreter.
  PartitionConfig partitionConfig;
  partitionConfig.funcName = "test";
  partitionConfig.numOfPartitions = 3;
  partitionConfig.backendNames = {"Interpreter", "CPU", "Interpreter"};
  partitionConfig.partitionNames = {"p1", "p2", "p3"};
  partitionConfig.nodeToPartition = {{"sub", 0}, {"mul", 1}};
  Partitioner partitioner(&mod_, devices);
  CompilationContext cctx;
  auto dagList = partitioner.partitionFromConfig(partitionConfig, cctx);
  ASSERT_TRUE((bool)dagList);
  EXPECT_EQ(mod_.getFunctions().size(), 3);
  EXPECT_EQ(dagList->size(), 1);
  ASSERT_TRUE(checkSaveNode(mod_));
  heterogeneousPartitionValidation(dagList.get(), mod_);
}

/// This one test load-balanced partition flow.
TEST_F(PartitionerTest, loadBalancedPartition) {
  ExecutionEngine EER, EEP;
  constexpr float range = 2.0;
  std::vector<ExecutionEngine *> engines{&EER, &EEP};
  // Since compiling modifies the module and partitioning modifies the
  // function, setup two EEs with identical functions for validation.
  for (auto EE : engines) {
    auto mod = &EE->getModule();
    F_ = mod->createFunction("main");
    auto *input =
        mod->createPlaceholder(ElemKind::FloatTy, {1, 32}, "input", false);
    auto *w1 = mod->createConstant(ElemKind::FloatTy, {32, 16}, "w1");
    auto *b1 = mod->createConstant(ElemKind::FloatTy, {16}, "b1");
    bindings_.allocate(input);
    w1->getHandle<>().randomize(-range, range, mod->getPRNG());
    b1->getHandle<>().randomize(-range, range, mod->getPRNG());

    // Initial FC.
    Node *I = F_->createFullyConnected("initial_fc", input, w1, b1);
    I = F_->createSigmoid("initial_sigmoid", I);

    // Left branch.
    auto *w2 = mod->createConstant(ElemKind::FloatTy, {16, 16}, "w2");
    auto *b2 = mod->createConstant(ElemKind::FloatTy, {16}, "b2");
    w2->getHandle<>().randomize(-range, range, mod->getPRNG());
    b2->getHandle<>().randomize(-range, range, mod->getPRNG());
    Node *L = F_->createFullyConnected("left_fc1", I, w2, b2);
    L = F_->createSigmoid("left_sigmoid1", L);
    auto *w3 = mod->createConstant(ElemKind::FloatTy, {16, 8}, "w3");
    auto *b3 = mod->createConstant(ElemKind::FloatTy, {8}, "b3");
    w3->getHandle<>().randomize(-range, range, mod->getPRNG());
    b3->getHandle<>().randomize(-range, range, mod->getPRNG());
    L = F_->createFullyConnected("left_fc2", L, w3, b3);
    L = F_->createSigmoid("left_sigmoid2", L);

    // Right branch.
    auto *w4 = mod->createConstant(ElemKind::FloatTy, {16, 16}, "w4");
    auto *b4 = mod->createConstant(ElemKind::FloatTy, {16}, "b4");
    w4->getHandle<>().randomize(-range, range, mod->getPRNG());
    b4->getHandle<>().randomize(-range, range, mod->getPRNG());
    Node *R = F_->createFullyConnected("right_fc1", I, w4, b4);
    R = F_->createSigmoid("right_sigmoid1", R);
    auto *w5 = mod->createConstant(ElemKind::FloatTy, {16, 8}, "w5");
    auto *b5 = mod->createConstant(ElemKind::FloatTy, {8}, "b5");
    w5->getHandle<>().randomize(-range, range, mod->getPRNG());
    b5->getHandle<>().randomize(-range, range, mod->getPRNG());
    R = F_->createFullyConnected("right_fc2", R, w5, b5);
    R = F_->createSigmoid("right_sigmoid2", R);

    // Join branches.
    auto *mul = F_->createMul("mul", L, R);
    F_->createSave("ret", mul);
  }

  // Infer using the un-partitioned graph.
  Tensor in(ElemKind::FloatTy, {1, 32});
  in.getHandle<>().randomize(-range, range, EER.getModule().getPRNG());

  EER.compile(CompilationMode::Infer);
  bindings_.clear();
  bindings_.allocate(EER.getModule().getPlaceholders());
  updateInputPlaceholders(bindings_, {bindings_.getPlaceholderByName("input")},
                          {&in});
  EER.run(bindings_);
  Tensor ref = bindings_.get(bindings_.getPlaceholderByName("ret"))->clone();

  std::vector<DeviceInfo> devices = {
      {3072, "Interpreter"}, {3072, "Interpreter"}, {3072, "Interpreter"}};
  Partitioner myPartitioner(&EEP.getModule(), devices, false, true);
  CompilationContext cctx;
  auto dagList = myPartitioner.loadBalancedPartition(cctx);
  ASSERT_TRUE((bool)dagList);
  EXPECT_EQ(EEP.getModule().getFunctions().size(), 3);
  EXPECT_EQ(dagList->size(), 1);
  EXPECT_TRUE(checkSaveNode(EEP.getModule()));

  // Run the paritioned graph and compare the results.
  bindings_.clear();
  bindings_.allocate(EEP.getModule().getPlaceholders());
  EEP.compile(cctx);
  for (auto it = dagList->begin(); it != dagList->end(); ++it) {
    executeDAG((*it).root.get(), EEP.getModule(), bindings_,
               {bindings_.getPlaceholderByName("input")}, {&in}, &EEP);
    Tensor test = bindings_.get(bindings_.getPlaceholderByName("ret"))->clone();
    EXPECT_TRUE(ref.isEqual(test));
  }
}
