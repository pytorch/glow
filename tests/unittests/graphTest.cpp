// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "glow/Graph/Graph.h"
#include "glow/ExecutionEngine/ExecutionEngine.h"
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
    Graph G;
    Module M(&G);
    Node *K = G.createVariable(ElemKind::FloatTy, {4, 320, 200, 3}, "input");
    Node *S = G.createVariable(ElemKind::IndexTy, {4, 1}, "select");

    K = G.createConv("Conv1", K, 16, 3, 2, 3);
    K = G.createRELU("Relu", K);
    K = G.createSoftMax("SoftMax", K, S);
    G.createSave("Save", K);
    G.dump();
    G.dumpDAG();
    lower(G, CompilationMode::Train);
    ::optimize(G, CompilationMode::Train);
    M.generateIR(CompilationMode::Train);
    M.dump();
    EXPECT_GT(M.getInstrs().size(), 0);
  }

  {
    unsigned numInputs = 10;
    Graph G;
    Module M(&G);

    auto *A = G.createVariable(ElemKind::FloatTy, {numInputs, 2}, "A");
    auto *Ex = G.createVariable(ElemKind::FloatTy, {numInputs, 1}, "Ex");

    Node *O = G.createFullyConnected("FC1", A, 6);
    O = G.createRELU("RELU1", O);
    O = G.createFullyConnected("FC2", O, 1);
    O = G.createRELU("RELU2", O);
    O = G.createRegression("Regression", O, Ex);
    G.createSave("Save", O);
    G.dump();
    G.dumpDAG();
    lower(G, CompilationMode::Train);
    ::optimize(G, CompilationMode::Train);
    M.generateIR(CompilationMode::Train);
    M.dump();
    EXPECT_GT(M.getInstrs().size(), 0);
  }
}

TEST(Graph, multipleReturnType) {
  {
    Graph G;
    Module M(&G);
    Variable *K =
        G.createVariable(ElemKind::FloatTy, {4, 320, 200, 3}, "input");
    auto *D = new DistributeNode("dist", K);
    G.getNodes().push_back(D);
    auto *V = G.createArithmetic("add", D->getLeft(), D->getRight(),
                                 ArithmeticNode::Mode::Add);
    G.createSave("S1", V);
    G.dump();
    G.dumpDAG();
  }
}

TEST(Graph, QuantizationProfileNodes) {
  unsigned numInputs = 10;
  Graph G;
  Module M(&G);

  auto *A = G.createVariable(ElemKind::FloatTy, {numInputs, 2}, "A");
  auto *Ex = G.createVariable(ElemKind::FloatTy, {numInputs, 1}, "Ex");

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
  ::optimize(G, CompilationMode::Infer);
  M.generateIR(CompilationMode::Infer);

  size_t numberOfProfileNodes =
      std::count_if(G.getNodes().begin(), G.getNodes().end(), [](Node *node) {
        return llvm::isa<QuantizationProfileNode>(node);
      });

  EXPECT_EQ(8, numberOfProfileNodes);
}

TEST(Graph, simpleQuant) {
  ExecutionEngine EE;
  auto &G = EE.getGraph();

  unsigned depth = 16;
  unsigned kernel = 5;
  unsigned pad = 0;
  unsigned step = 1;
  unsigned width = 224;

  auto *input =
      G.createVariable(ElemKind::Int8QTy, {1, width, width, 3}, 0.4, 0.2,
                       "Input", Variable::VisibilityKind::Public);

  // Calculate the size and allocate the output buffer.
  std::array<size_t, 4> filterDim = {{depth, kernel, kernel, 3}};
  auto *filter = G.createVariable(ElemKind::Int8QTy, filterDim, 3.3, 0.4, "F",
                                  Variable::VisibilityKind::Private);
  auto *bias = G.createVariable(ElemKind::Int8QTy, {depth}, 1.3, 5.6, "B",
                                Variable::VisibilityKind::Private);

  // Calculate the size and allocate the output buffer.
  auto outSz = calculateConvOutputDims(width, width, kernel, step, pad);
  std::array<size_t, 4> outDims = {{1, outSz.first, outSz.second, 16}};
  auto t = G.uniqueType(glow::ElemKind::Int8QTy, outDims, 1.5, 6.7);

  auto *conv =
      G.createConv("conv", input, filter, bias, t, depth, kernel, step, pad);

  auto s = conv->getType()->size();
  auto *fcFilter = G.createVariable(ElemKind::Int8QTy, {s, 6}, 0.4, 0.2, "F");
  auto *fcBias = G.createVariable(ElemKind::Int8QTy, {6}, 0.4, 0.2, "B");
  Node *O = G.createFullyConnected("fc1", conv, fcFilter, fcBias, 6);
  G.createSave("ret", O);
  EE.compile(CompilationMode::Infer);
}

TEST(Graph, quantizeDequantizeNodes) {
  ExecutionEngine EE;
  auto &G = EE.getGraph();

  auto *input = G.createVariable(ElemKind::FloatTy, {1, 3}, "Input");
  auto qType = G.uniqueType(ElemKind::Int8QTy, {1, 3}, 0.3, 0.5);

  auto *Q = G.createQuantize("quantize", input, qType);

  auto transform = G.uniqueType(ElemKind::Int8QTy, {1, 3}, 1.4, 3);
  auto *A = G.createRescaleQuantized("rescale", Q, transform);

  auto *D = G.createDequantize("dequantize", A);
  G.createSave("ret", D);
  EE.compile(CompilationMode::Infer);
}
