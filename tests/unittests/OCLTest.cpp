/**
 * Copyright (c) 2017-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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
  inputs.getHandle().initXavier(1);
  Tensor out1;
  Tensor out2;

  inferReluNet(&inputs, &out1, BackendKind::OpenCL);
  inferReluNet(&inputs, &out2, BackendKind::Interpreter);

  EXPECT_TRUE(out1.isEqual(out2));
}

TEST(OpenCLCorrectnessTest, convOps) {
  Tensor inputs(ElemKind::FloatTy, {2, 3, 16, 16});
  inputs.getHandle().initXavier(1);
  Tensor out1;
  Tensor out2;

  inferBasicConvNet(&inputs, &out1, BackendKind::OpenCL, 4);
  inferBasicConvNet(&inputs, &out2, BackendKind::Interpreter, 4);

  EXPECT_TRUE(out1.isEqual(out2));
}

TEST(OpenCLCorrectnessTest, basicFCNet) {
  Tensor inputs(ElemKind::FloatTy, {2, 3, 16, 16});
  inputs.getHandle().initXavier(1);
  Tensor out1;
  Tensor out2;

  inferBasicFCNet(&inputs, &out1, BackendKind::OpenCL);
  inferBasicFCNet(&inputs, &out2, BackendKind::Interpreter);

  EXPECT_TRUE(out1.isEqual(out2));
}

TEST(OpenCLCorrectnessTest, inferMixedNet) {
  Tensor inputs(ElemKind::FloatTy, {2, 3, 16, 16});
  inputs.getHandle().initXavier(1);
  Tensor out1;
  Tensor out2;

  inferMixedNet(&inputs, &out1, BackendKind::OpenCL);
  inferMixedNet(&inputs, &out2, BackendKind::Interpreter);

  EXPECT_TRUE(out1.isEqual(out2));
}
