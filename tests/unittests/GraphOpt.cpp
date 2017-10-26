// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "glow/Graph/Graph.h"
#include "glow/Graph/Node.h"
#include "glow/Graph/Nodes.h"
#include "glow/IR/IR.h"
#include "glow/Optimizer/Optimizer.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>

using namespace glow;

TEST(GraphOptz, DCE) {

  Graph G;
  Module M(&G);
  Node *K = G.createVariable(ElemKind::FloatTy, {4, 320, 200, 3}, "input");

  for (int i = 0; i < 40; i++) {
    K = G.createRELU("relu", K);
    // Add a graph structure that diverges and converges, to catch algorithms
    // that perform a dump recursive scan.
    K = G.createArithmetic("arith", K, K, ArithmeticNode::Mode::Add);
  }

  // Check that we know how many knows we've created.
  EXPECT_EQ(G.getNodes().size(), 80);

  // Optimize all of the dead code.
  ::glow::optimize(G, CompilationMode::Infer);

  //  All of the nodes are gone.
  EXPECT_EQ(G.getNodes().size(), 0);
}
