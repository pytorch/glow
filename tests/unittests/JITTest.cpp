// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "BackendTestUtils.h"

#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Graph/Graph.h"
#include "glow/IR/IR.h"
#include "glow/IR/IRBuilder.h"
#include "glow/IR/Instrs.h"
#include "glow/Support/Random.h"

#include "gtest/gtest.h"

#include <cassert>
#include <string>

using namespace glow;
using llvm::cast;

TEST(JITCorrectnessTest, maxTest) {
  Tensor inputs1(ElemKind::FloatTy, {3, 8, 2});
  Tensor inputs2(ElemKind::FloatTy, {3, 8, 2});
  inputs1.getHandle().randomize(1);
  inputs2.getHandle().randomize(1);
  Tensor out1;
  Tensor out2;

  inferMaxNet(&inputs1, &inputs2, &out1, BackendKind::JIT);
  inferMaxNet(&inputs1, &inputs2, &out2, BackendKind::Interpreter);
  auto H1 = out1.getHandle();
  auto H2 = out2.getHandle();

  EXPECT_TRUE(H1.isEqual(H2));
}

TEST(JITCorrectnessTest, minTest) {
  Tensor inputs1(ElemKind::FloatTy, {10, 3, 12, 8});
  Tensor inputs2(ElemKind::FloatTy, {10, 3, 12, 8});
  inputs1.getHandle().randomize(1);
  inputs2.getHandle().randomize(1);
  Tensor out1;
  Tensor out2;

  inferMinNet(&inputs1, &inputs2, &out1, BackendKind::JIT);
  inferMinNet(&inputs1, &inputs2, &out2, BackendKind::Interpreter);
  auto H1 = out1.getHandle();
  auto H2 = out2.getHandle();

  EXPECT_TRUE(H1.isEqual(H2));
}

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

TEST(JITCorrectnessTest, reshapeTest) {
  Tensor inputs(ElemKind::FloatTy, {12, 6, 8, 12});
  inputs.getHandle().randomize(1);
  Tensor out1;
  Tensor out2;

  inferReshapeNet(&inputs, {18, 4, 24, 4}, &out1, BackendKind::JIT);
  inferReshapeNet(&inputs, {18, 4, 24, 4}, &out2, BackendKind::Interpreter);
  auto H1 = out1.getHandle();
  auto H2 = out2.getHandle();

  EXPECT_TRUE(H1.isEqual(H2));
}

TEST(JITCorrectnessTest, selectTest) {
  Tensor cond(ElemKind::FloatTy, {5, 3, 9, 2});
  Tensor inputs1(ElemKind::FloatTy, {5, 3, 9, 2});
  Tensor inputs2(ElemKind::FloatTy, {5, 3, 9, 2});
  auto condH = cond.getHandle();
  for (size_t i = 0; i < 5 * 3 * 9 * 2; ++i) {
    condH.raw(i) = nextRandInt01();
  }
  inputs1.getHandle().randomize(1);
  inputs2.getHandle().randomize(1);
  Tensor out1;
  Tensor out2;

  inferSelectNet(&cond, &inputs1, &inputs2, &out1, BackendKind::JIT);
  inferSelectNet(&cond, &inputs1, &inputs2, &out2, BackendKind::Interpreter);
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
