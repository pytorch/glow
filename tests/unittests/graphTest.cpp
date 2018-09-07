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

#include "glow/Graph/Graph.h"
#include "BackendTestUtils.h"
#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Graph/Node.h"
#include "glow/Graph/Nodes.h"
#include "glow/Graph/Utils.h"
#include "glow/IR/IR.h"

#include "llvm/ADT/SmallPtrSet.h"

#include "gtest/gtest.h"

using namespace glow;

TEST(Graph, testVariableErasure) {
  Module MD;
  auto &vars = MD.getVars();
  EXPECT_EQ(vars.size(), 0);
  EXPECT_EQ(std::distance(vars.begin(), vars.end()), vars.size());

  Variable *V = MD.createVariable(ElemKind::FloatTy, {1, 1}, "dummy",
                                  VisibilityKind::Public);
  EXPECT_EQ(vars.size(), 1);
  EXPECT_EQ(std::distance(vars.begin(), vars.end()), vars.size());

  MD.eraseVariable(V);
  EXPECT_EQ(vars.size(), 0);
  EXPECT_EQ(std::distance(vars.begin(), vars.end()), vars.size());
}

TEST(Graph, simpleTestConv) {
  Module MD;
  Function *F = MD.createFunction("F");
  IRFunction M(F);
  Node *K = MD.createVariable(ElemKind::FloatTy, {4, 320, 200, 3}, "input");
  Node *S = MD.createVariable(ElemKind::Int64ITy, {4, 1}, "select");

  K = F->createConv("Conv1", K, 16, 3, 2, 3, 1);
  K = F->createRELU("Relu", K);
  K = F->createSoftMax("SoftMax", K, S);
  F->createSave("Save", K);
  F->dump();
  F->dumpDAG();
  lower(F, MockBackend());
  ::optimize(F, CompilationMode::Train);
  M.generateIR();
  M.dump();
  EXPECT_GT(M.getInstrs().size(), 0);
}

/// Test that our use lists are correctly reflecting the state of the IR
/// and in particular that it is not polluted by temporary variable.
TEST(Graph, useList) {
  Module MD;
  Function *F = MD.createFunction("F");
  IRFunction M(F);
  Variable *K = MD.createVariable(ElemKind::FloatTy, {4, 320, 200, 3}, "input");

  EXPECT_EQ(K->getNumUsers(), 0);

  ConvolutionNode *conv = F->createConv("Conv1", K, 16, 3, 2, 3, 1);

  EXPECT_TRUE(K->hasOneUse());
  EXPECT_EQ(K->getNumUsers(), 1);
  EXPECT_EQ(conv->getNumUsers(), 0);

  // Although the filter of the convolution is only used by the convolution
  // node, calling getFilter creates a temporary NodeValue that messes up
  // with the actual use list.
  // Therefore those checks are currently inverted but should be
  // fixed eventually.
  // Test with implicit temporary NodeValue.
  EXPECT_TRUE(conv->getFilter().getNode()->hasOneUse());
  EXPECT_EQ(conv->getFilter().getNode()->getNumUsers(), 1);

  // Test with explicit temporary NodeValue.
  Node *nodeFilter;
  {
    NodeValue tmp = conv->getFilter();
    EXPECT_TRUE(tmp.getNode()->hasOneUse());
    EXPECT_EQ(tmp.getNode()->getNumUsers(), 1);
    nodeFilter = tmp.getNode();
    // Test with NodeValue still around.
    EXPECT_TRUE(nodeFilter->hasOneUse());
    EXPECT_EQ(nodeFilter->getNumUsers(), 1);
  }

  // Test with NodeValue took out.
  EXPECT_TRUE(nodeFilter->hasOneUse());
  EXPECT_EQ(nodeFilter->getNumUsers(), 1);

  // Same kind of test but with the convolution node itself.
  {
    NodeValue tmpConvRes(conv, 0);
    EXPECT_EQ(conv->getNumUsers(), 0);
    EXPECT_EQ(tmpConvRes.getNode()->getNumUsers(), 0);
  }

  // Add a couple of uses to conv and make sure it reflects on its use list.
  F->createSave("Save", conv, K);

  EXPECT_FALSE(K->hasOneUse());
  EXPECT_EQ(K->getNumUsers(), 2);
  EXPECT_EQ(conv->getNumUsers(), 1);
  EXPECT_TRUE(conv->hasOneUse());

  {
    NodeValue tmpConvRes(conv, 0);
    EXPECT_TRUE(tmpConvRes.getNode()->hasOneUse());
    EXPECT_TRUE(conv->hasOneUse());
    EXPECT_EQ(conv->getNumUsers(), 1);
    EXPECT_EQ(tmpConvRes.getNode()->getNumUsers(), 1);
  }

  F->createSave("Save", conv, K);

  EXPECT_FALSE(K->hasOneUse());
  EXPECT_EQ(K->getNumUsers(), 3);
  EXPECT_EQ(conv->getNumUsers(), 2);
  EXPECT_FALSE(conv->hasOneUse());

  {
    NodeValue tmpConvRes(conv, 0);
    EXPECT_FALSE(tmpConvRes.getNode()->hasOneUse());
    EXPECT_FALSE(conv->hasOneUse());
    EXPECT_EQ(conv->getNumUsers(), 2);
    EXPECT_EQ(tmpConvRes.getNode()->getNumUsers(), 2);
  }
}

TEST(Graph, useListIteration) {
  Module MD;
  Function *F = MD.createFunction("F");
  IRFunction M(F);
  Node *K = MD.createVariable(ElemKind::FloatTy, {4, 320, 200, 3}, "input");

  EXPECT_EQ(K->getNumUsers(), 0);

  ConvolutionNode *conv1 = F->createConv("Conv1", K, 16, 3, 2, 3, 1);
  ConvolutionNode *conv2 = F->createConv("Conv2", K, 16, 3, 2, 3, 1);
  // Check the number of users for different nodes.
  EXPECT_EQ(K->getNumUsers(), 2);
  EXPECT_EQ(conv1->getNumUsers(), 0);
  EXPECT_TRUE(conv2->getFilter().getNode()->hasOneUse());
  EXPECT_EQ(conv1->getFilter().getNode()->getNumUsers(), 1);
  // Check that the first user of K is conv1.
  EXPECT_EQ(K->getUsers().begin()->getUser(), conv1);
  // Check that the second user of K is conv2.
  EXPECT_EQ((++K->getUsers().begin())->getUser(), conv2);
}

TEST(Graph, simpleTestFC) {
  unsigned numInputs = 10;
  Module MD;
  Function *F = MD.createFunction("F");
  IRFunction M(F);

  auto *A = MD.createVariable(ElemKind::FloatTy, {numInputs, 2}, "A");
  auto *Ex = MD.createVariable(ElemKind::FloatTy, {numInputs, 1}, "Ex");

  Node *O = F->createFullyConnected("FC1", A, 6);
  O = F->createRELU("RELU1", O);
  O = F->createFullyConnected("FC2", O, 1);
  O = F->createRELU("RELU2", O);
  O = F->createRegression("Regression", O, Ex);
  F->createSave("Save", O);
  F->dump();
  F->dumpDAG();
  lower(F, MockBackend());
  ::optimize(F, CompilationMode::Train);
  M.generateIR();
  M.dump();
  EXPECT_GT(M.getInstrs().size(), 0);
}

TEST(Graph, QuantizationProfileNodes) {
  unsigned numInputs = 10;
  Module MD;
  Function *F = MD.createFunction("F");
  IRFunction M(F);

  auto *A = MD.createVariable(ElemKind::FloatTy, {numInputs, 2}, "A");

  // Add non float operation, which should not be profiled.
  auto *outQTy = F->getParent()->uniqueType(glow::ElemKind::Int8QTy,
                                            {numInputs, 2}, 1.5, 6);
  auto *quantize = F->createQuantize("quantize", A, outQTy);
  // Make sure that quantize is not optimized away.
  F->createSave("save", quantize);

  // Multiple nodes read from the same variable.
  // Only one Quantization Profile node should be created for the output
  // from the variable.
  Node *O = F->createFullyConnected("FC1", A, 6);
  Node *C = F->createFullyConnected("FC2", A, 6);
  O = F->createRELU("RELU1", O);
  F->createSave("save", O);
  F->createSave("save", C);

  // Simulate actual usage.
  ::optimize(F, CompilationMode::Infer);
  F = ::glow::profileQuantization(F);
  lower(F, MockBackend());
  ::optimize(F, CompilationMode::Infer);

  size_t numberOfProfileNodes =
      std::count_if(F->getNodes().begin(), F->getNodes().end(), [](Node &node) {
        return llvm::isa<QuantizationProfileNode>(&node);
      });

  EXPECT_EQ(10, numberOfProfileNodes);
}

TEST(Graph, simpleQuant) {
  ExecutionEngine EE;
  auto &MD = EE.getModule();
  auto *F = MD.createFunction("main");

  unsigned depth = 16;
  llvm::SmallVector<unsigned_t, 2> kernels = {5, 5};
  llvm::SmallVector<unsigned_t, 4> pads = {0, 0, 0, 0};
  llvm::SmallVector<unsigned_t, 2> steps = {1, 1};
  unsigned width = 224;

  auto *input = MD.createVariable(ElemKind::Int8QTy, {1, width, width, 3}, 0.4,
                                  2, "Input", VisibilityKind::Public);

  // Calculate the size and allocate the output buffer.
  std::array<size_t, 4> filterDim = {{depth, kernels[0], kernels[1], 3}};
  auto *filter = MD.createVariable(ElemKind::Int8QTy, filterDim, 3.3, 4, "F",
                                   VisibilityKind::Private);
  auto *bias = MD.createVariable(ElemKind::Int8QTy, {depth}, 1.3, 5, "B",
                                 VisibilityKind::Private);

  // Calculate the size and allocate the output buffer.
  auto outSz = calculateConvPoolOutputDims(width, width, kernels, steps, pads);
  std::array<size_t, 4> outDims = {{1, outSz.first, outSz.second, 16}};
  auto t = F->getParent()->uniqueType(glow::ElemKind::Int8QTy, outDims, 1.5, 6);

  auto *conv =
      F->createConv("conv", input, filter, bias, t, kernels, steps, pads, 1);

  auto s = conv->getResult().getType()->size();
  auto *fcFilter = MD.createVariable(ElemKind::Int8QTy, {s, 6}, 0.4, 2, "F");
  auto *fcBias = MD.createVariable(ElemKind::Int8QTy, {6}, 0.4, 2, "B");
  Node *O = F->createFullyConnected("fc1", conv, fcFilter, fcBias);
  F->createSave("ret", O);
  EE.compile(CompilationMode::Infer, F);
}

TEST(Graph, quantizeDequantizeNodes) {
  ExecutionEngine EE;
  auto &MD = EE.getModule();
  auto F = MD.createFunction("main");

  auto *input = MD.createVariable(ElemKind::FloatTy, {1, 3}, "Input");
  auto qType = F->getParent()->uniqueType(ElemKind::Int8QTy, {1, 3}, 0.3, 5);

  auto *Q = F->createQuantize("quantize", input, qType);

  auto transform =
      F->getParent()->uniqueType(ElemKind::Int8QTy, {1, 3}, 1.4, 3);
  auto *A = F->createRescaleQuantized("rescale", Q, transform);

  auto *D = F->createDequantize("dequantize", A);
  F->createSave("ret", D);
  EE.compile(CompilationMode::Infer, F);
}

TEST(Graph, quantizeGather) {
  ExecutionEngine EE;
  auto &mod = EE.getModule();
  auto *F = mod.createFunction("main");
  auto *input = mod.createVariable(ElemKind::Int8QTy, {2, 2}, 0.4, 2, "input",
                                   VisibilityKind::Public);
  auto *indices = mod.createVariable(ElemKind::Int64ITy, {1}, "index",
                                     VisibilityKind::Public);
  auto *gather = F->createGather("gather", input, indices);
  F->createSave("ret", gather);
  EE.compile(CompilationMode::Infer, F);
}

TEST(Graph, cloneTest) {
  Module M;

  Function *F = M.createFunction("main");
  Node *K = M.createVariable(ElemKind::FloatTy, {4, 320, 200, 3}, "input");
  Node *S = M.createVariable(ElemKind::Int64ITy, {4, 1}, "select");
  Node *conv = F->createConv("Conv1", K, 16, 3, 2, 3, 1);
  Node *relu = F->createRELU("Relu", conv);
  Node *SM = F->createSoftMax("SoftMax", relu, S);
  F->createSave("Save", SM);

  auto *newConv = F->addNode(conv->clone());
  auto *newRelu = F->addNode(relu->clone());
  auto *newSM = F->addNode(SM->clone());

  EXPECT_TRUE(newConv != conv && conv->isEqual(*newConv));
  EXPECT_TRUE(newRelu != relu && relu->isEqual(*newRelu));
  EXPECT_TRUE(newSM != SM && SM->isEqual(*newSM));
}

TEST(Graph, moduleTest) {
  Module M;
  M.createFunction("one");
  M.createFunction("two");
  M.createVariable(ElemKind::FloatTy, {4, 320, 200, 3}, "V1");
  M.createVariable(ElemKind::FloatTy, {4, 320, 200, 3}, "V2");
  EXPECT_TRUE(M.hasFunction("one"));
  EXPECT_TRUE(M.hasFunction("two"));
  EXPECT_FALSE(M.hasFunction("four"));
  M.dumpDAG();
}

TEST(Graph, functionDependenciesTest) {
  Module M;
  auto F1 = M.createFunction("one");
  auto F2 = M.createFunction("two");
  auto V1 = M.createVariable(ElemKind::FloatTy, {4, 320, 200, 3}, "V1");
  auto V2 = M.createVariable(ElemKind::FloatTy, {4, 320, 200, 3}, "V2");
  auto V3 = M.createVariable(ElemKind::FloatTy, {4, 320, 200, 3}, "V3");
  M.createVariable(ElemKind::FloatTy, {4, 320, 200, 3}, "V4");

  auto sum = F1->createSub("1_sub_2", V1, V2);
  F1->createSave("sv", sum, V1);
  F2->createSave("sv", V3, V2);

  EXPECT_TRUE(M.hasFunction("one"));
  EXPECT_TRUE(M.hasFunction("two"));
  EXPECT_FALSE(M.hasFunction("four"));
  M.dumpDAG();
}

TEST(Graph, cloneTest2) {
  Module M;

  auto *F = M.createFunction("main");
  Node *K = M.createVariable(ElemKind::FloatTy, {4, 320, 200, 3}, "input");
  Node *S = M.createVariable(ElemKind::Int64ITy, {4, 1}, "select");
  Node *conv = F->createConv("Conv1", K, 16, 3, 2, 3, 1);
  Node *relu = F->createRELU("Relu", conv);
  Node *concat = F->createConcat("concat", {relu, relu, relu}, 0);

  Node *SM = F->createSoftMax("SoftMax", concat, S);
  F->createSave("Save", SM);

  auto *newF = F->clone("new_main");
  newF->verify();
  F->dump();
  newF->dump();

  EXPECT_EQ(newF->getNodes().size(), F->getNodes().size());
  EXPECT_EQ(newF->getParent(), F->getParent());
}

TEST(Graph, NodeValue) {
  ExecutionEngine EE;
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  auto *inputX = mod.createVariable(ElemKind::FloatTy, {1}, "input",
                                    VisibilityKind::Public, true);
  inputX->getPayload().init(Tensor::InitKind::Broadcast, 3.0, mod.getPRNG());

  NodeValue a = F->createAdd("x2", inputX, inputX);
  a = F->createAdd("x4", a, a);
  a = F->createAdd("x8", a, a);
  auto S = F->createSave("Save", a);

  EE.compile(CompilationMode::Infer, F);
  EE.run();

  EXPECT_EQ(
      llvm::cast<Variable>(S->getOutput())->getPayload().getHandle().raw(0),
      24);
}

/// Check that by deleting one function, the variables that refernced
/// by this function, will reduce its number of uses by one.
TEST(Graph, deleteFunction) {
  ExecutionEngine EE;
  auto &mod = EE.getModule();
  Function *F1 = mod.createFunction("f1");
  auto *inputX = mod.createVariable(ElemKind::FloatTy, {1}, "input",
                                    VisibilityKind::Public, true);
  F1->createLog("log1", inputX);
  Function *F2 = mod.createFunction("f2");
  F2->createLog("log2", inputX);
  // We check the number of user of inputX to be 2 as only F1 and F2 are
  // using it.
  EXPECT_EQ(inputX->getNumUsers(), 2);
  // Erase this function here to see if we can see the number of user of inputX
  // reduce to 1.
  mod.eraseFunction(F1);
  EXPECT_EQ(inputX->getNumUsers(), 1);
}

TEST(Graph, nodesWithPredicates) {
  ExecutionEngine EE;

  Tensor inputs(ElemKind::FloatTy, {1, 32, 32, 3});

  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  F->setName("interpret");
  auto *input = mod.createVariable(ElemKind::FloatTy, {1, 32, 32, 3}, "input",
                                   VisibilityKind::Public);

  auto *ex = mod.createVariable(ElemKind::Int64ITy, {1, 1}, "exp");

  Variable *pred = mod.createVariable(ElemKind::Int64ITy, {1}, "predicate",
                                      VisibilityKind::Private, false);

  auto *CV0 = F->createConv("conv1", input, 16, 5, 1, 2, 1);
  auto *RL0 = F->createRELU("relu1", CV0);
  auto *MP0 = F->createMaxPool("pool1", RL0, 2, 2, 0);

  CV0->setPredicate(pred);
  RL0->setPredicate(pred);
  MP0->setPredicate(pred);

  auto *FCL1 = F->createFullyConnected("fc", MP0, 10);
  auto *RL3 = F->createRELU("relu4", FCL1);
  auto *SM = F->createSoftMax("sm", RL3, ex);
  F->createSave("ret", SM);

  EE.compile(CompilationMode::Infer, F);
  EE.updateVariables({input}, {&inputs});
  EE.run();
}

// Return the number of ConvolutionNode after lower.
unsigned getConvNodeSize(BackendKind kind) {
  Module mod;
  Function *F = mod.createFunction("main");
  IRFunction M(F);
  auto *input = mod.createVariable(ElemKind::FloatTy, {1, 2, 1, 32}, "input");
  ConvolutionNode *CN = F->createConv("conv", input, 6, 1, 1, 0, 2);
  F->createSave("save", CN);

  std::unique_ptr<Backend> backend(createBackend(kind));
  lower(F, *backend);

  unsigned count = 0;
  for (auto &n : F->getNodes()) {
    if (n.getKind() == Kinded::Kind::ConvolutionNodeKind) {
      count++;
    }
  }

  if (kind == BackendKind::Interpreter) {
    EXPECT_EQ(count, 1);
  }

  return count;
}

// Check the unrolling grouped convolution opt status:
// -- disabled for Interpreter, CPU and OpenCL backend,
TEST(Graph, disableUnrollingGroupConv) {
  unsigned numberOfNodesInterpreter = getConvNodeSize(BackendKind::Interpreter);
  (void)numberOfNodesInterpreter;

#ifdef GLOW_WITH_CPU
  unsigned numberOfNodesCPU = getConvNodeSize(BackendKind::CPU);
  EXPECT_EQ(numberOfNodesCPU, numberOfNodesInterpreter);
#endif // GLOW_WITH_CPU

#ifdef GLOW_WITH_OPENCL
  unsigned numberOfNodesOpenCL = getConvNodeSize(BackendKind::OpenCL);
  EXPECT_EQ(numberOfNodesOpenCL, numberOfNodesInterpreter);
#endif // GLOW_WITH_OPENCL
}

/// Check that save nodes are properly scheduled.
/// That is, they happen after the last use of the related variable.
/// In that test, the order of the creation of the nodes give a valid schedule.
TEST(Graph, schedulingOfSavesOrderProvided) {
  ExecutionEngine EE;

  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  auto *A = mod.createVariable(ElemKind::FloatTy, {3, 32}, "A",
                               VisibilityKind::Public, true);
  auto *B = mod.createVariable(A->getType(), "B", VisibilityKind::Public, true);
  auto *zero =
      mod.createVariable(A->getType(), "zero", VisibilityKind::Public, true);

  A->getPayload().init(Tensor::InitKind::Xavier, 1.0, mod.getPRNG());
  B->getPayload().init(Tensor::InitKind::Xavier, 1.0, mod.getPRNG());
  zero->getPayload().init(Tensor::InitKind::Broadcast, 0.0, mod.getPRNG());

  auto *addAB = F->createAdd("addAB", A, B);

  auto *saveNode = F->createSave("ret", addAB);
  F->createSave("resetA", zero, A);

  // Copy the value of A.
  Tensor AOrig = A->getPayload().clone();

  EE.compile(CompilationMode::Infer, F);
  EE.run();
  auto *ret = saveNode->getVariable();
  auto handleAOrig = AOrig.getHandle<>();
  auto handleB = B->getPayload().getHandle<>();
  auto handleRet = ret->getPayload().getHandle<>();
  bool allEqual = true;
  for (unsigned row = 0; row != 3; ++row) {
    for (unsigned column = 0; column != 32; ++column) {
      allEqual &= handleAOrig.at({row, column}) + handleB.at({row, column}) ==
                  handleRet.at({row, column});
    }
  }
  EXPECT_TRUE(A->getPayload().isEqual(zero->getPayload(), 0.0));
  EXPECT_TRUE(allEqual);
}

/// Same as schedulingOfSavesOrderProvided except the order in which the nodes
/// are added to the function don't form a valid schedule.
/// In other words, the scheduler won't get away with scheduling
/// using only the order of the nodes in the list of nodes.
TEST(Graph, schedulingOfSaves) {
  ExecutionEngine EE;

  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  auto *A = mod.createVariable(ElemKind::FloatTy, {3, 32}, "A",
                               VisibilityKind::Public, true);
  auto *B = mod.createVariable(A->getType(), "B", VisibilityKind::Public, true);
  auto *zero =
      mod.createVariable(A->getType(), "zero", VisibilityKind::Public, true);
  F->createSave("resetA", zero, A);

  A->getPayload().init(Tensor::InitKind::Xavier, 1.0, mod.getPRNG());
  B->getPayload().init(Tensor::InitKind::Xavier, 1.0, mod.getPRNG());
  zero->getPayload().init(Tensor::InitKind::Broadcast, 0.0, mod.getPRNG());

  auto *addAB = F->createAdd("addAB", A, B);

  auto *saveNode = F->createSave("ret", addAB);

  // Copy the value of A.
  Tensor AOrig = A->getPayload().clone();

  EE.compile(CompilationMode::Infer, F);
  EE.run();
  auto *ret = saveNode->getVariable();
  auto handleAOrig = AOrig.getHandle<>();
  auto handleB = B->getHandle<>();
  auto handleRet = ret->getHandle<>();
  bool allEqual = true;
  for (unsigned row = 0; row != 3; ++row) {
    for (unsigned column = 0; column != 32; ++column) {
      allEqual &= handleAOrig.at({row, column}) + handleB.at({row, column}) ==
                  handleRet.at({row, column});
    }
  }
  EXPECT_TRUE(A->getPayload().isEqual(zero->getPayload(), 0.0));
  EXPECT_TRUE(allEqual);
}

/// Check that the parent link is properly updated while tweaking
/// nodes and their function.
TEST(Graph, parentLink) {
  ExecutionEngine EE;

  auto &mod = EE.getModule();
  Variable *V = new Variable("V", mod.uniqueType(ElemKind::FloatTy, {3, 32}),
                             VisibilityKind::Private, true);

  // Variables don't belong to any function...
  EXPECT_EQ(V->getParent(), nullptr);
  // Even when we create them from a module...
  Variable *V2 = mod.createVariable(V->getType(), "V2");
  EXPECT_EQ(V2->getParent(), nullptr);
  // Or add them to a module.
  mod.addVar(V);
  EXPECT_EQ(V->getParent(), nullptr);

  Function *F = mod.createFunction("main");

  // Nodes created with function helper belong to the related function.
  auto *addNode = F->createAdd("addnode", V, V2);
  EXPECT_EQ(addNode->getParent(), F);

  // Nodes created directly don't belong to any function.
  auto *addNode2 = new AddNode("addnode2", V->getType(), addNode, addNode);
  EXPECT_EQ(addNode2->getParent(), nullptr);

  // Nodes added to a function belong to that function.
  F->addNode(addNode2);
  EXPECT_EQ(addNode2->getParent(), F);

  // Cloned nodes don't belong to anything.
  auto *clonedAddNode = addNode->clone();
  EXPECT_EQ(clonedAddNode->getParent(), nullptr);

  // Check that the setter properly sets things.
  clonedAddNode->setParent(F);
  EXPECT_EQ(clonedAddNode->getParent(), F);
  clonedAddNode->setParent(nullptr);
  EXPECT_EQ(clonedAddNode->getParent(), nullptr);

  // Add the cloned node to F so that the memory is properly
  // cleaned at the end of the test.
  F->addNode(clonedAddNode);
  EXPECT_EQ(clonedAddNode->getParent(), F);
}

/// Check that Cmp nodes are created with proper output types.
TEST(Graph, cmpOutputTypes) {
  ExecutionEngine EE;

  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  // Define two different quntized types.
  auto qType1 = F->getParent()->uniqueType(ElemKind::Int8QTy, {1, 3}, 0.3, 5);
  auto qType2 = F->getParent()->uniqueType(ElemKind::Int8QTy, {1, 3}, 0.4, 5);
  // Define two variables of quantized types.
  Variable *qv1 = mod.createVariable(qType1, "V1", VisibilityKind::Private);
  Variable *qv2 = mod.createVariable(qType2, "V2", VisibilityKind::Private);
  // Create cmp nodes using quantized inputs.
  auto *cmpNode1 = F->createCmpEQ("cmpeq", qv1, qv2);
  auto *cmpNode2 = F->createCmpLTE("cmplte", qv1, qv2);
  // Check that the output type of cmp nodes is quantized, has scale 1.0 and
  // offset 0.
  EXPECT_TRUE(cmpNode1->getResult().getType()->isQuantizedType());
  EXPECT_EQ(cmpNode1->getResult().getType()->getScale(), 1.0);
  EXPECT_EQ(cmpNode1->getResult().getType()->getOffset(), 0);
  EXPECT_TRUE(cmpNode2->getResult().getType()->isQuantizedType());
  EXPECT_EQ(cmpNode2->getResult().getType()->getScale(), 1.0);
  EXPECT_EQ(cmpNode2->getResult().getType()->getOffset(), 0);

  // Define a non-quantized type.
  auto nqType3 = F->getParent()->uniqueType(ElemKind::FloatTy, {1, 3});
  // Define two variables of non-quantized types.
  Variable *nqv3 = mod.createVariable(nqType3, "V3", VisibilityKind::Private);
  Variable *nqv4 = mod.createVariable(nqType3, "V4", VisibilityKind::Private);
  // Create cmp nodes using non-quantized inputs.
  auto *cmpNode3 = F->createCmpEQ("cmpeq", nqv3, nqv4);
  auto *cmpNode4 = F->createCmpLTE("cmplte", nqv3, nqv4);
  // Check that output of cmp nodes is a non-quantized type matching the type of
  // inputs.
  EXPECT_FALSE(cmpNode3->getResult().getType()->isQuantizedType());
  EXPECT_EQ(cmpNode3->getResult().getType(), nqv3->getType());
  EXPECT_FALSE(cmpNode4->getResult().getType()->isQuantizedType());
  EXPECT_EQ(cmpNode4->getResult().getType(), nqv3->getType());
}

/// Check that the users of value are equal to expectedUsers.
static bool
hasAllTheseUses(const llvm::SmallPtrSetImpl<const Node *> &expectedUsers,
                const NodeValue &value) {
  llvm::SmallPtrSet<const Node *, 4> uses;
  for (const NodeUse &use : value.getUsers()) {
    const Node *user = use.getUser();
    if (!expectedUsers.count(user)) {
      // We found a user that wasn't on the list.
      return false;
    }
    uses.insert(user);
  }
  return expectedUsers.size() == uses.size();
}

/// Check that our uses lists are correct for nodes with multiple results.
TEST(Graph, usesListsWithSeveralResult) {
  ExecutionEngine EE;

  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  auto *input = mod.createVariable(ElemKind::FloatTy, {3, 32}, "input",
                                   VisibilityKind::Public, true);
  auto *topK = F->createTopK("topK", input, 12);
  EXPECT_EQ(topK->getNumUsers(), 0);

  NodeValue values = topK->getValues();
  NodeValue indices = topK->getIndices();
  llvm::SmallPtrSet<const Node *, 4> savesOfValues;
  llvm::SmallPtrSet<const Node *, 4> savesOfIndices;

  EXPECT_EQ(indices.getNumUsers(), 0);
  EXPECT_EQ(values.getNumUsers(), 0);

  EXPECT_FALSE(indices.hasOneUse());
  EXPECT_FALSE(values.hasOneUse());

  EXPECT_TRUE(hasAllTheseUses(savesOfIndices, indices));
  EXPECT_TRUE(hasAllTheseUses(savesOfValues, values));

  // Now add a user to only one result of the topK node.
  savesOfValues.insert(F->createSave("saveValues1", values));

  // The whole node should inherit the uses of each of its results.
  EXPECT_EQ(topK->getNumUsers(), 1);

  // Each result should have its own use list.
  EXPECT_EQ(indices.getNumUsers(), 0);
  EXPECT_EQ(values.getNumUsers(), 1);

  EXPECT_FALSE(indices.hasOneUse());
  EXPECT_TRUE(values.hasOneUse());

  EXPECT_TRUE(hasAllTheseUses(savesOfIndices, indices));
  EXPECT_TRUE(hasAllTheseUses(savesOfValues, values));

  // Add a user to the other result of the topK node.
  savesOfIndices.insert(F->createSave("saveIndices1", indices));

  // The whole node should inherit the uses of each of its results.
  EXPECT_EQ(topK->getNumUsers(), 2);

  // Each result should have its own use list.
  EXPECT_EQ(indices.getNumUsers(), 1);
  EXPECT_EQ(values.getNumUsers(), 1);

  EXPECT_TRUE(indices.hasOneUse());
  EXPECT_TRUE(values.hasOneUse());

  EXPECT_TRUE(hasAllTheseUses(savesOfIndices, indices));
  EXPECT_TRUE(hasAllTheseUses(savesOfValues, values));

  // Add a couple more users of values and indices.
  // Interleaves the insertions in the uses list for both values and indices.
  savesOfValues.insert(F->createSave("saveValues2", values));
  savesOfValues.insert(F->createSave("saveValues3", values));
  savesOfIndices.insert(F->createSave("saveIndices2", indices));

  EXPECT_EQ(topK->getNumUsers(), 5);

  EXPECT_EQ(indices.getNumUsers(), 2);
  EXPECT_EQ(values.getNumUsers(), 3);

  EXPECT_FALSE(indices.hasOneUse());
  EXPECT_FALSE(values.hasOneUse());

  EXPECT_TRUE(hasAllTheseUses(savesOfIndices, indices));
  EXPECT_TRUE(hasAllTheseUses(savesOfValues, values));
}

/// Check that our uses lists are correct when accessed through
/// NodeValue.
TEST(Graph, usesListsThroughNodeValues) {
  ExecutionEngine EE;

  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  auto *input = mod.createVariable(ElemKind::FloatTy, {3, 32}, "input",
                                   VisibilityKind::Public, true);
  auto *reLU = F->createRELU("reLU", input);
  EXPECT_EQ(reLU->getNumUsers(), 0);

  NodeValue values = reLU->getResult();
  llvm::SmallPtrSet<const Node *, 4> savesOfValues;

  EXPECT_EQ(values.getNumUsers(), 0);

  EXPECT_FALSE(values.hasOneUse());

  EXPECT_TRUE(hasAllTheseUses(savesOfValues, values));

  // Now add a user to only one result of the reLU node.
  savesOfValues.insert(F->createSave("saveValues1", values));

  // The whole node should inherit the uses of each of its results.
  EXPECT_EQ(reLU->getNumUsers(), 1);

  // The NodeValue should match.
  EXPECT_EQ(values.getNumUsers(), 1);
  EXPECT_TRUE(values.hasOneUse());
  EXPECT_TRUE(hasAllTheseUses(savesOfValues, values));

  // Add one more use.
  savesOfValues.insert(F->createSave("saveValues2", values));

  // The whole node should inherit the uses of each of its results.
  EXPECT_EQ(reLU->getNumUsers(), 2);

  EXPECT_EQ(values.getNumUsers(), 2);
  EXPECT_FALSE(values.hasOneUse());
  EXPECT_TRUE(hasAllTheseUses(savesOfValues, values));

  // Add a couple more users.
  savesOfValues.insert(F->createSave("saveValues3", values));
  savesOfValues.insert(F->createSave("saveValues4", values));

  EXPECT_EQ(reLU->getNumUsers(), 4);

  EXPECT_EQ(values.getNumUsers(), 4);
  EXPECT_FALSE(values.hasOneUse());
  EXPECT_TRUE(hasAllTheseUses(savesOfValues, values));
}

/// Verify that the pre-order visitor works correctly.
TEST(Graph, PreOrderTest) {
  Module M;
  auto *F = M.createFunction("main");

  Variable *input1 = M.createVariable(ElemKind::FloatTy, {4, 10}, "input1",
                                      VisibilityKind::Public);
  Variable *input2 = M.createVariable(ElemKind::FloatTy, {4, 10}, "input2",
                                      VisibilityKind::Public);
  SplatNode *zero = F->createSplat("zero", input1->getType(), 0.);
  MulNode *mul1 = F->createMul("mul1", zero, input1);
  MulNode *mul2 = F->createMul("mul2", zero, input2);
  MulNode *mul3 = F->createMul("mul3", mul1, mul2);
  SaveNode *ret1 = F->createSave("ret1", mul3);

  SplatNode *one = F->createSplat("one", input2->getType(), 1.0);
  AddNode *add1 = F->createAdd("add1", input2, one);
  AddNode *add2 = F->createAdd("add2", add1, one);
  AddNode *add3 = F->createAdd("add3", add2, one);
  SaveNode *ret2 = F->createSave("ret2", add2);

  GraphPreOrderVisitor visitor(*F);
  auto order = visitor.getPreOrder();

  ASSERT_EQ(order.size(), 14);
  EXPECT_EQ(order[0], ret1);
  EXPECT_EQ(order[1], mul3);
  EXPECT_EQ(order[2], mul1);
  EXPECT_EQ(order[3], zero);
  EXPECT_EQ(order[4], input1);
  EXPECT_EQ(order[5], mul2);
  EXPECT_EQ(order[6], input2);
  EXPECT_EQ(order[7], ret1->getOutput());
  EXPECT_EQ(order[8], add3);
  EXPECT_EQ(order[9], add2);
  EXPECT_EQ(order[10], add1);
  EXPECT_EQ(order[11], one);
  EXPECT_EQ(order[12], ret2);
  EXPECT_EQ(order[13], ret2->getOutput());
}

/// Verify that the post-order visitor works correctly.
TEST(Graph, PostOrderTest) {
  Module M;
  auto *F = M.createFunction("main");

  Variable *input1 = M.createVariable(ElemKind::FloatTy, {4, 10}, "input1",
                                      VisibilityKind::Public);
  Variable *input2 = M.createVariable(ElemKind::FloatTy, {4, 10}, "input2",
                                      VisibilityKind::Public);
  SplatNode *zero = F->createSplat("zero", input1->getType(), 0.);
  MulNode *mul1 = F->createMul("mul1", zero, input1);
  MulNode *mul2 = F->createMul("mul2", zero, input2);
  MulNode *mul3 = F->createMul("mul3", mul1, mul2);
  SaveNode *ret1 = F->createSave("ret1", mul3);

  SplatNode *one = F->createSplat("one", input2->getType(), 1.0);
  AddNode *add1 = F->createAdd("add1", input2, one);
  AddNode *add2 = F->createAdd("add2", add1, one);
  AddNode *add3 = F->createAdd("add3", add2, one);
  SaveNode *ret2 = F->createSave("ret2", add2);

  GraphPostOrderVisitor visitor(*F);
  auto order = visitor.getPostOrder();

  ASSERT_EQ(order.size(), 14);
  EXPECT_EQ(order[0], zero);
  EXPECT_EQ(order[1], input1);
  EXPECT_EQ(order[2], mul1);
  EXPECT_EQ(order[3], input2);
  EXPECT_EQ(order[4], mul2);
  EXPECT_EQ(order[5], mul3);
  EXPECT_EQ(order[6], ret1->getOutput());
  EXPECT_EQ(order[7], ret1);
  EXPECT_EQ(order[8], one);
  EXPECT_EQ(order[9], add1);
  EXPECT_EQ(order[10], add2);
  EXPECT_EQ(order[11], add3);
  EXPECT_EQ(order[12], ret2->getOutput());
  EXPECT_EQ(order[13], ret2);
}

TEST(Graph, placeholder) {
  Module MD;
  Function *F = MD.createFunction("F");
  IRFunction M(F);
  Node *K = MD.createPlaceholder(ElemKind::FloatTy, {4, 320, 200, 3}, "input");
  Node *S = MD.createPlaceholder(ElemKind::Int64ITy, {4, 1}, "select");

  K = F->createFullyConnected("FC", K, 10); 
  K = F->createRELU("Relu", K);
  K = F->createSoftMax("SoftMax", K, S);
  F->createSave("Save", K);
}
