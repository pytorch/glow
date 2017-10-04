// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "glow/Graph/Graph.h"
#include "glow/Graph/Node.h"
#include "glow/Graph/Nodes.h"

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
    Module M;
    Graph G(M);
    Node *K = G.createVariable(ElemKind::FloatTy, {4, 320, 200, 3}, "input");
    Node *S = G.createVariable(ElemKind::IndexTy, {4, 1}, "select");

    K = G.createConv(K, 16, 3, 2, 3);
    K = G.createRELU(K);
    K = G.createSoftMax(K, S);
  }

  {
    unsigned numInputs = 10;
    Module M;
    Graph G(M);
    auto *A = G.createVariable(ElemKind::FloatTy, {numInputs, 2}, "A");
    auto *Ex = G.createVariable(ElemKind::FloatTy, {numInputs, 1}, "Ex");

    Node *O = G.createFullyConnected(A, 6);
    O = G.createRELU(O);
    O = G.createFullyConnected(O, 1);
    O = G.createRELU(O);
    G.createRegression(O, Ex);
  }
}
