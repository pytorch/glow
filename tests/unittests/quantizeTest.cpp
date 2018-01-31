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
  auto outSz = calculateConvOutputDims(width, width, pad, kernel, step);
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
