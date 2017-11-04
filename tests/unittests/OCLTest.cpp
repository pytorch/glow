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

#if WITH_OPENCL

TEST(Interpreter, interpret) {
  ExecutionEngine EE(BackendKind::OpenCL);

  Tensor inputs(ElemKind::FloatTy, {1, 6, 6});

  auto &G = EE.getGraph();
  auto *input = G.createVariable(ElemKind::FloatTy, {1, 6, 6}, "input");
  inputs.getHandle().randomize(1);

  auto *RL0 = G.createRELU("relu", input);
  auto *S1 = G.createSigmoid("sig", RL0);
  auto *T1 = G.createTanh("tanh", S1);
  auto *RL2 = G.createRELU("relu", T1);
  auto *A1 = G.createArithmetic("add", S1, RL2, ArithmeticNode::Mode::Add);
  auto *M1 = G.createArithmetic("add", A1, T1, ArithmeticNode::Mode::Mul);
  auto result = G.createSave("ret", M1);

  EE.compile(CompilationMode::Infer);

  EE.getModule().dump();
  inputs.getHandle().dump("Inputs", "\n");
  result->getOutput()->getHandle().randomize(1);

  EE.infer({input}, {&inputs});
  result->getOutput()->getHandle().dump("after", "\n");
}

#endif // WITH_OPENCL
