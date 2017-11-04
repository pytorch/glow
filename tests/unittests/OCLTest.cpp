// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Graph/Graph.h"
#include "glow/IR/IR.h"
#include "glow/IR/IRBuilder.h"
#include "glow/IR/Instrs.h"

#include "gtest/gtest.h"

#include <cassert>
#include <string>

using namespace glow;

TEST(Interpreter, interpret) {
  ExecutionEngine EE(BackendKind::OpenCL);

  Tensor inputs(ElemKind::FloatTy, {1, 6, 6});

  auto &G = EE.getGraph();
  auto *input = G.createVariable(ElemKind::FloatTy, {1, 6, 6}, "input");
  inputs.getHandle().randomize(1);

  auto *RL0 = G.createRELU("relu", input);
  auto *RL1 = G.createRELU("relu", RL0);
  auto result = G.createSave("ret", RL1);

  EE.compile(CompilationMode::Infer);

  EE.getModule().dump();
  inputs.getHandle().dump("Inputs", "\n");
  result->getOutput()->getHandle().randomize(1);
  result->getOutput()->getHandle().dump("before", "\n");

  EE.infer({input}, {&inputs});
  result->getOutput()->getHandle().dump("after", "\n");
}
