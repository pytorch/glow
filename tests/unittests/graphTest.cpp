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
#include "glow/IR/IR.h"

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
  Node *S = MD.createVariable(ElemKind::IndexTy, {4, 1}, "select");

  K = F->createConv("Conv1", K, 16, 3, 2, 3, 1);
  K = F->createRELU("Relu", K);
  K = F->createSoftMax("SoftMax", K, S);
  F->createSave("Save", K);
  F->dump();
  F->dumpDAG();
  lower(F, CompilationMode::Train, MockBackend());
  ::optimize(F, CompilationMode::Train);
  M.generateIR();
  M.dump();
  EXPECT_GT(M.getInstrs().size(), 0);
}

TEST(Graph, useList) {
  Module MD;
  Function *F = MD.createFunction("F");
  IRFunction M(F);
  Node *K = MD.createVariable(ElemKind::FloatTy, {4, 320, 200, 3}, "input");

  EXPECT_EQ(K->getNumUsers(), 0);

  ConvolutionNode *conv = F->createConv("Conv1", K, 16, 3, 2, 3, 1);

  EXPECT_TRUE(K->hasOneUse());
  EXPECT_EQ(K->getNumUsers(), 1);
  EXPECT_EQ(conv->getNumUsers(), 0);
  EXPECT_TRUE(conv->getFilter()->hasOneUse());
  EXPECT_EQ(conv->getFilter()->getNumUsers(), 1);
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
  lower(F, CompilationMode::Train, MockBackend());
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
  lower(F, CompilationMode::Infer, MockBackend());
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
  unsigned kernel = 5;
  llvm::SmallVector<size_t, 4> pads = {0, 0, 0, 0};
  unsigned step = 1;
  unsigned width = 224;

  auto *input = MD.createVariable(ElemKind::Int8QTy, {1, width, width, 3}, 0.4,
                                  2, "Input", VisibilityKind::Public);

  // Calculate the size and allocate the output buffer.
  std::array<size_t, 4> filterDim = {{depth, kernel, kernel, 3}};
  auto *filter = MD.createVariable(ElemKind::Int8QTy, filterDim, 3.3, 4, "F",
                                   VisibilityKind::Private);
  auto *bias = MD.createVariable(ElemKind::Int8QTy, {depth}, 1.3, 5, "B",
                                 VisibilityKind::Private);

  // Calculate the size and allocate the output buffer.
  auto outSz = calculateConvOutputDims(width, width, kernel, step, pads);
  std::array<size_t, 4> outDims = {{1, outSz.first, outSz.second, 16}};
  auto t = F->getParent()->uniqueType(glow::ElemKind::Int8QTy, outDims, 1.5, 6);

  auto *conv =
      F->createConv("conv", input, filter, bias, t, kernel, step, pads, 1);

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
  auto *indices = mod.createVariable(ElemKind::IndexTy, {1}, "index",
                                     VisibilityKind::Public);
  auto *gather = F->createGather("gather", input, indices);
  F->createSave("ret", gather);
  EE.compile(CompilationMode::Infer, F);
}

TEST(Graph, cloneTest) {
  Module M;

  Function *F = M.createFunction("main");
  Node *K = M.createVariable(ElemKind::FloatTy, {4, 320, 200, 3}, "input");
  Node *S = M.createVariable(ElemKind::IndexTy, {4, 1}, "select");
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
  Node *S = M.createVariable(ElemKind::IndexTy, {4, 1}, "select");
  Node *conv = F->createConv("Conv1", K, 16, 3, 2, 3, 1);
  Node *relu = F->createRELU("Relu", conv);
  Node *concat = F->createConcat("concat", {relu, relu, relu}, 0);

  Node *SM = F->createSoftMax("SoftMax", concat, S);
  F->createSave("Save", SM);

  auto *newF = F->clone("new_main");
  newF->verify();
  EXPECT_EQ(newF->getNodes().size(), F->getNodes().size());
  EXPECT_EQ(newF->getParent(), F->getParent());
}

TEST(Graph, NodeValue) {
  ExecutionEngine EE;
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  auto *inputX = mod.createVariable(ElemKind::FloatTy, {1}, "input",
                                    VisibilityKind::Public,
                                    Variable::TrainKind::Broadcast, 3.0);
  NodeValue a = F->createAdd("x2", inputX, inputX);
  a = F->createAdd("x4", a, a);
  a = F->createAdd("x8", a, a);
  auto S = F->createSave("Save", a);

  EE.compile(CompilationMode::Infer, F);
  EE.run({}, {});

  EXPECT_EQ(
      llvm::cast<Variable>(S->getOutput())->getPayload().getHandle().raw(0),
      24);
}

TEST(Graph, nodesWithPredicates) {
  ExecutionEngine EE;

  Tensor inputs(ElemKind::FloatTy, {1, 32, 32, 3});

  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  F->setName("interpret");
  auto *input = mod.createVariable(ElemKind::FloatTy, {1, 32, 32, 3}, "input",
                                   VisibilityKind::Public);

  auto *ex = mod.createVariable(ElemKind::IndexTy, {1, 1}, "exp");

  Variable *pred =
      mod.createVariable(ElemKind::IndexTy, {1}, "predicate",
                         VisibilityKind::Private, Variable::TrainKind::None);

  auto *CV0 = F->createConv("conv1", input, 16, 5, 1, 2, 1);
  auto *RL0 = F->createRELU("relu1", CV0);
  auto *MP0 = F->createPoolMax("pool1", RL0, 2, 2, 0);

  CV0->setPredicate(pred);
  RL0->setPredicate(pred);
  MP0->setPredicate(pred);

  auto *FCL1 = F->createFullyConnected("fc", MP0, 10);
  auto *RL3 = F->createRELU("relu4", FCL1);
  auto *SM = F->createSoftMax("sm", RL3, ex);
  F->createSave("ret", SM);

  EE.compile(CompilationMode::Infer, F);
  EE.run({input}, {&inputs});
}

// Return the number of ConvolutionNode after lower.
unsigned getConvNodeSize(BackendKind kind) {
  Module mod;
  Function *F = mod.createFunction("main");
  IRFunction M(F);
  auto *input = mod.createVariable(ElemKind::FloatTy, {1, 2, 1, 32}, "input");
  ConvolutionNode *CN = F->createConv("conv", input, 6, 1, 1, 0, 2);
  F->createSave("save", CN);

  std::unique_ptr<Backend> backend(createBackend(kind, &M));
  lower(F, CompilationMode::Infer, *backend);

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
// -- disabled for Interpreter and CPU backend,
// -- enabled for openCL backend.
TEST(Graph, disableUnrollingGroupConv) {
  unsigned numberOfNodesInterpreter = getConvNodeSize(BackendKind::Interpreter);
  (void)numberOfNodesInterpreter;

#ifdef GLOW_WITH_CPU
  unsigned numberOfNodesCPU = getConvNodeSize(BackendKind::CPU);
  EXPECT_EQ(numberOfNodesCPU, numberOfNodesInterpreter);
#endif // GLOW_WITH_CPU

#ifdef GLOW_WITH_OPENCL
  unsigned numberOfNodesOpenCL = getConvNodeSize(BackendKind::OpenCL);
  EXPECT_GT(numberOfNodesOpenCL, numberOfNodesInterpreter);
#endif // GLOW_WITH_OPENCL
}
