// Copyright 2017 Facebook Inc.  All Rights Reserved.

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
    Graph G;
    Module M(&G);
    Node *K = G.createVariable(ElemKind::FloatTy, {4, 320, 200, 3}, "input");
    Node *S = G.createVariable(ElemKind::IndexTy, {4, 1}, "select");

    K = G.createConv("Conv1", K, 16, 3, 2, 3);
    K = G.createRELU("Relu", K);
    K = G.createSoftMax("SoftMax", K, S);
    G.dump();
    G.dumpDAG();
    M.generateIR(CompilationMode::Train);
    M.dump();
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
    G.createRegression("Regression", O, Ex);
    G.dump();
    G.dumpDAG();
    M.generateIR(CompilationMode::Train);
    M.dump();
  }
}
