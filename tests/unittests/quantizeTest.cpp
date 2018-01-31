// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Graph/Graph.h"
#include "glow/Graph/Node.h"
#include "glow/Graph/Nodes.h"
#include "glow/IR/IR.h"

#include "gtest/gtest.h"

#include <cassert>
#include <cstddef>
#include <cstdint>

using namespace glow;

TEST(QuantizeTest, simpleQuant) {
  ExecutionEngine EE;
  auto &G = EE.getGraph();

  auto *input = G.createVariable(ElemKind::Int8QTy, {10, 2}, 0.4, 0.2, "Input",
                                 Variable::VisibilityKind::Public);
  auto *filter =
      G.createVariable(ElemKind::Int8QTy, {2, 6}, 0.4, 0.2, "Filter");
  auto *bias = G.createVariable(ElemKind::Int8QTy, {6}, 0.4, 0.2, "Bias");
  Node *O = G.createFullyConnected("fc1", input, filter, bias, 6);
  G.createSave("ret", O);
  EE.compile(CompilationMode::Infer);
}
