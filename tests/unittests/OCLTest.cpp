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
using llvm::cast;

void inferReluNet(Tensor *inputs, Tensor *out, BackendKind kind) {
  ExecutionEngine EE(kind);
  auto &G = EE.getGraph();
  auto *var =
      G.createVariable(inputs->getElementType(), inputs->dims(), "input");
  auto *relu = G.createRELU("relu", var);
  auto result = G.createSave("ret", relu);
  EE.compile(CompilationMode::Infer);
  EE.infer({var}, {inputs});
  out->copyFrom(&result->getVariable()->getPayload());
}

void inferBasicConvNet(Tensor *inputs, Tensor *out, BackendKind kind) {
  ExecutionEngine EE(kind);
  auto &G = EE.getGraph();
  auto *var =
      G.createVariable(inputs->getElementType(), inputs->dims(), "input");
  auto *tr = G.createTranspose("tr", var, {0, 2, 3, 1});
  auto *conv = G.createConv("conv", tr, 4, 5, 2, 1);
  cast<Variable>(conv->getFilter())->getHandle().clear(2);
  cast<Variable>(conv->getBias())->getHandle().clear(2);
  auto *pool = G.createPool("pool", conv, PoolNode::Mode::Max, 2, 2, 0);
  auto result = G.createSave("ret", pool);
  EE.compile(CompilationMode::Infer);
  EE.infer({var}, {inputs});
  out->copyFrom(&result->getVariable()->getPayload());
}

void inferBasicFCNet(Tensor *inputs, Tensor *out, BackendKind kind) {
  ExecutionEngine EE(kind);
  auto &G = EE.getGraph();
  auto *var =
      G.createVariable(inputs->getElementType(), inputs->dims(), "input");
  auto *tr = G.createTranspose("tr", var, {0, 2, 3, 1});
  auto *fc = G.createFullyConnected("fc", tr, 16);
  auto *rl0 = G.createRELU("relu", fc);
  auto *fc2 = G.createFullyConnected("fc2", rl0, 8);
  auto *rl1 = G.createRELU("relu", fc);
  cast<Variable>(fc->getFilter())->getHandle().clear(0.8);
  cast<Variable>(fc2->getFilter())->getHandle().clear(1.5);
  auto result = G.createSave("ret", rl1);
  EE.compile(CompilationMode::Infer);
  EE.infer({var}, {inputs});
  out->copyFrom(&result->getVariable()->getPayload());
}

void inferMixedNet(Tensor *inputs, Tensor *out, BackendKind kind) {
  ExecutionEngine EE(kind);
  auto &G = EE.getGraph();
  auto *var =
      G.createVariable(inputs->getElementType(), inputs->dims(), "input");
  auto *selected = G.createVariable(ElemKind::IndexTy, {2, 1}, "selected");

  auto *tr = G.createTranspose("tr", var, {0, 2, 3, 1});
  auto *fc = G.createFullyConnected("fc", tr, 16);
  auto *th0 = G.createTanh("tanh", fc);
  auto *sg0 = G.createSigmoid("sig", fc);
  auto *A1 = G.createArithmetic("add", th0, sg0, ArithmeticNode::Mode::Add);
  auto *fc2 = G.createFullyConnected("fc2", A1, 16);

  auto *R = G.createRegression("reg", fc2, fc2);
  auto *SM = G.createSoftMax("SM", R, selected);
  auto result = G.createSave("ret", SM);

  cast<Variable>(fc->getFilter())->getHandle().clear(0.4);
  cast<Variable>(fc2->getFilter())->getHandle().clear(3.5);

  EE.compile(CompilationMode::Infer);
  EE.infer({var}, {inputs});
  out->copyFrom(&result->getVariable()->getPayload());
}

TEST(OpenCLCorrectnessTest, ReluTest) {
  Tensor inputs(ElemKind::FloatTy, {2, 16});
  inputs.getHandle().randomize(1);
  Tensor out1;
  Tensor out2;

  inferReluNet(&inputs, &out1, BackendKind::OpenCL);
  inferReluNet(&inputs, &out2, BackendKind::Interpreter);
  auto H1 = out1.getHandle();
  auto H2 = out2.getHandle();

  EXPECT_TRUE(H1.isEqual(H2));
  if (!H1.isEqual(H2)) {
    H1.dump();
    H2.dump();
  }
}

TEST(OpenCLCorrectnessTest, convOps) {
  Tensor inputs(ElemKind::FloatTy, {2, 3, 16, 16});
  inputs.getHandle().randomize(1);
  Tensor out1;
  Tensor out2;

  inferBasicConvNet(&inputs, &out1, BackendKind::OpenCL);
  inferBasicConvNet(&inputs, &out2, BackendKind::Interpreter);
  auto H1 = out1.getHandle();
  auto H2 = out2.getHandle();

  EXPECT_TRUE(H1.isEqual(H2));
  if (!H1.isEqual(H2)) {
    H1.dump();
    H2.dump();
  }
}

TEST(OpenCLCorrectnessTest, basicFCNet) {
  Tensor inputs(ElemKind::FloatTy, {2, 3, 16, 16});
  inputs.getHandle().randomize(1);
  Tensor out1;
  Tensor out2;

  inferBasicFCNet(&inputs, &out1, BackendKind::OpenCL);
  inferBasicFCNet(&inputs, &out2, BackendKind::Interpreter);
  auto H1 = out1.getHandle();
  auto H2 = out2.getHandle();

  EXPECT_TRUE(H1.isEqual(H2));
  if (!H1.isEqual(H2)) {
    H1.dump();
    H2.dump();
  }
}

TEST(OpenCLCorrectnessTest, inferMixedNet) {
  Tensor inputs(ElemKind::FloatTy, {2, 3, 16, 16});
  inputs.getHandle().randomize(1);
  Tensor out1;
  Tensor out2;

  inferMixedNet(&inputs, &out1, BackendKind::OpenCL);
  inferMixedNet(&inputs, &out2, BackendKind::Interpreter);
  auto H1 = out1.getHandle();
  auto H2 = out2.getHandle();

  EXPECT_TRUE(H1.isEqual(H2));
  if (!H1.isEqual(H2)) {
    H1.dump();
    H2.dump();
  }
}
