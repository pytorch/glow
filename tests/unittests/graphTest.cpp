// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Graph/Graph.h"
#include "glow/Graph/Node.h"
#include "glow/Graph/Nodes.h"
#include "glow/IR/IR.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>

using namespace glow;

TEST(Graph, simpleTest) {
  {
    Module MD;
    Function &G = *MD.createFunction("F");
    IRFunction M(&G);
    Node *K = MD.createVariable(ElemKind::FloatTy, {4, 320, 200, 3}, "input");
    Node *S = MD.createVariable(ElemKind::IndexTy, {4, 1}, "select");

    K = G.createConv("Conv1", K, 16, 3, 2, 3);
    K = G.createRELU("Relu", K);
    K = G.createSoftMax("SoftMax", K, S);
    G.createSave("Save", K);
    G.dump();
    G.dumpDAG();
    lower(G, CompilationMode::Train);
    ::optimize(&G, CompilationMode::Train);
    M.generateIR(CompilationMode::Train);
    M.dump();
    EXPECT_GT(M.getInstrs().size(), 0);
  }

  {
    unsigned numInputs = 10;
    Module MD;
    Function &G = *MD.createFunction("F");
    IRFunction M(&G);

    auto *A = MD.createVariable(ElemKind::FloatTy, {numInputs, 2}, "A");
    auto *Ex = MD.createVariable(ElemKind::FloatTy, {numInputs, 1}, "Ex");

    Node *O = G.createFullyConnected("FC1", A, 6);
    O = G.createRELU("RELU1", O);
    O = G.createFullyConnected("FC2", O, 1);
    O = G.createRELU("RELU2", O);
    O = G.createRegression("Regression", O, Ex);
    G.createSave("Save", O);
    G.dump();
    G.dumpDAG();
    lower(G, CompilationMode::Train);
    ::optimize(&G, CompilationMode::Train);
    M.generateIR(CompilationMode::Train);
    M.dump();
    EXPECT_GT(M.getInstrs().size(), 0);
  }
}

TEST(Graph, QuantizationProfileNodes) {
  unsigned numInputs = 10;
  Module MD;
  Function &G = *MD.createFunction("F");
  IRFunction M(&G);

  auto *A = MD.createVariable(ElemKind::FloatTy, {numInputs, 2}, "A");
  auto *Ex = MD.createVariable(ElemKind::FloatTy, {numInputs, 1}, "Ex");

  // Create two nodes reading from the same variable.
  // Only one Quantization Profile node should be created for the output
  // from the variable.
  Node *O = G.createFullyConnected("FC1", A, 6);
  Node *C = G.createFullyConnected("FC2", A, 6);
  (void)C;
  O = G.createRELU("RELU1", O);
  G.createRegression("Regression", O, Ex);

  ::glow::profileQuantization(G);
  lower(G, CompilationMode::Infer);
  ::optimize(&G, CompilationMode::Infer);
  M.generateIR(CompilationMode::Infer);

  size_t numberOfProfileNodes =
      std::count_if(G.getNodes().begin(), G.getNodes().end(), [](Node *node) {
        return llvm::isa<QuantizationProfileNode>(node);
      });

  EXPECT_EQ(8, numberOfProfileNodes);
}

TEST(Graph, simpleQuant) {
  ExecutionEngine EE;
  auto &MD = EE.getModule();
  auto &G = *MD.createFunction("main");

  unsigned depth = 16;
  unsigned kernel = 5;
  unsigned pad = 0;
  unsigned step = 1;
  unsigned width = 224;

  auto *input = MD.createVariable(ElemKind::Int8QTy, {1, width, width, 3}, 0.4,
                                  2, "Input", Variable::VisibilityKind::Public);

  // Calculate the size and allocate the output buffer.
  std::array<size_t, 4> filterDim = {{depth, kernel, kernel, 3}};
  auto *filter = MD.createVariable(ElemKind::Int8QTy, filterDim, 3.3, 4, "F",
                                   Variable::VisibilityKind::Private);
  auto *bias = MD.createVariable(ElemKind::Int8QTy, {depth}, 1.3, 5, "B",
                                 Variable::VisibilityKind::Private);

  // Calculate the size and allocate the output buffer.
  auto outSz = calculateConvOutputDims(width, width, kernel, step, pad);
  std::array<size_t, 4> outDims = {{1, outSz.first, outSz.second, 16}};
  auto t = G.getParent()->uniqueType(glow::ElemKind::Int8QTy, outDims, 1.5, 6);

  auto *conv =
      G.createConv("conv", input, filter, bias, t, depth, kernel, step, pad);

  auto s = conv->getType()->size();
  auto *fcFilter = MD.createVariable(ElemKind::Int8QTy, {s, 6}, 0.4, 2, "F");
  auto *fcBias = MD.createVariable(ElemKind::Int8QTy, {6}, 0.4, 2, "B");
  Node *O = G.createFullyConnected("fc1", conv, fcFilter, fcBias);
  G.createSave("ret", O);
  EE.compile(CompilationMode::Infer, &G);
}

TEST(Graph, quantizeDequantizeNodes) {
  ExecutionEngine EE;
  auto &MD = EE.getModule();
  auto &G = *MD.createFunction("main");

  auto *input = MD.createVariable(ElemKind::FloatTy, {1, 3}, "Input");
  auto qType = G.getParent()->uniqueType(ElemKind::Int8QTy, {1, 3}, 0.3, 5);

  auto *Q = G.createQuantize("quantize", input, qType);

  auto transform = G.getParent()->uniqueType(ElemKind::Int8QTy, {1, 3}, 1.4, 3);
  auto *A = G.createRescaleQuantized("rescale", Q, transform);

  auto *D = G.createDequantize("dequantize", A);
  G.createSave("ret", D);
  EE.compile(CompilationMode::Infer, &G);
}

TEST(Graph, cloneTest) {
  Module M;

  Function &G = *M.createFunction("main");
  Node *K = M.createVariable(ElemKind::FloatTy, {4, 320, 200, 3}, "input");
  Node *S = M.createVariable(ElemKind::IndexTy, {4, 1}, "select");
  Node *conv = G.createConv("Conv1", K, 16, 3, 2, 3);
  Node *relu = G.createRELU("Relu", conv);
  Node *SM = G.createSoftMax("SoftMax", relu, S);
  G.createSave("Save", SM);

  auto *newConv = G.addNode(conv->clone());
  auto *newRelu = G.addNode(relu->clone());
  auto *newSM = G.addNode(SM->clone());

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

  auto sum = F1->createArithmetic("1_sub_2", V1, V2, ArithmeticNode::Mode::Sub);
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
  Node *conv = F->createConv("Conv1", K, 16, 3, 2, 3);
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
                                    Variable::VisibilityKind::Public,
                                    Variable::TrainKind::Broadcast, 3.0);
  NodeValue a =
      F->createArithmetic("x2", inputX, inputX, ArithmeticNode::Mode::Add);
  a = F->createArithmetic("x4", a, a, ArithmeticNode::Mode::Add);
  a = F->createArithmetic("x8", a, a, ArithmeticNode::Mode::Add);
  auto S = F->createSave("Save", a);

  EE.compile(CompilationMode::Infer, F);
  EE.run({}, {});

  EXPECT_EQ(
      llvm::cast<Variable>(S->getOutput())->getPayload().getHandle().raw(0),
      24);
}
