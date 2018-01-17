// Copyright 2017 Facebook Inc.  All Rights Reserved.

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

TEST(OpenCLCorrectnessTest, reluTest) {
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
