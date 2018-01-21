// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "AutoGenInferFunc.h"
#include "BackendTestUtils.h"

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

TEST(JITCorrectnessTest, reluTest) {
  Tensor inputs(ElemKind::FloatTy, {2, 16});
  inputs.getHandle().randomize(1);
  Tensor out1;
  Tensor out2;

  inferReluNet(&inputs, &out1, BackendKind::JIT);
  inferReluNet(&inputs, &out2, BackendKind::Interpreter);
  auto H1 = out1.getHandle();
  auto H2 = out2.getHandle();

  EXPECT_TRUE(H1.isEqual(H2));
}

TEST(JITCorrectnessTest, sigmoidTest) {
  Tensor inputs(ElemKind::FloatTy, {11, 4, 5, 2});
  inputs.getHandle().randomize(1);
  Tensor out1;
  Tensor out2;

  inferSigmoidNet(&inputs, &out1, BackendKind::JIT);
  inferSigmoidNet(&inputs, &out2, BackendKind::Interpreter);
  auto H1 = out1.getHandle();
  auto H2 = out2.getHandle();

  EXPECT_TRUE(H1.isEqual(H2));
}

TEST(JITCorrectnessTest, tanhTest) {
  Tensor inputs(ElemKind::FloatTy, {4, 7, 3, 3});
  inputs.getHandle().randomize(1);
  Tensor out1;
  Tensor out2;

  inferTanhNet(&inputs, &out1, BackendKind::JIT);
  inferTanhNet(&inputs, &out2, BackendKind::Interpreter);
  auto H1 = out1.getHandle();
  auto H2 = out2.getHandle();

  EXPECT_TRUE(H1.isEqual(H2));
}

TEST(JITCorrectnessTest, convOps) {
  Tensor inputs(ElemKind::FloatTy, {2, 3, 16, 16});
  inputs.getHandle().randomize(1);
  Tensor out1;
  Tensor out2;

  inferBasicConvNet(&inputs, &out1, BackendKind::JIT);
  inferBasicConvNet(&inputs, &out2, BackendKind::Interpreter);
  auto H1 = out1.getHandle();
  auto H2 = out2.getHandle();

  EXPECT_TRUE(H1.isEqual(H2));
}

TEST(JITCorrectnessTest, basicFCNet) {
  Tensor inputs(ElemKind::FloatTy, {2, 3, 16, 16});
  inputs.getHandle().randomize(1);
  Tensor out1;
  Tensor out2;

  inferBasicFCNet(&inputs, &out1, BackendKind::JIT);
  inferBasicFCNet(&inputs, &out2, BackendKind::Interpreter);
  auto H1 = out1.getHandle();
  auto H2 = out2.getHandle();

  EXPECT_TRUE(H1.isEqual(H2));
}
