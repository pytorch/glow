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

#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Graph/Graph.h"
#include "glow/IR/IRBuilder.h"

#include "gtest/gtest.h"

#include "llvm/Support/Casting.h"

using namespace glow;

class BackendTest : public ::testing::TestWithParam<BackendKind> {
public:
  ExecutionEngine EE_{GetParam()};
};


TEST(Interpreter, NotImplementedSave) {
  // Interpreter backend does not support a save method.
  // Exercise it and make sure that it fails.
  ExecutionEngine EE;
  auto &mod = EE.getModule();

  // Create a few nodes to make sure IR can be normally generated.
  Function *F = mod.createFunction("main");
  F->createSave("save", mod.createVariable(ElemKind::FloatTy, {2}, "A",
                                           VisibilityKind::Public,
                                           Variable::TrainKind::None));

  EXPECT_DEATH(EE.save(CompilationMode::Infer, F, "output"), "");
}

TEST(Interpreter, profileQuantizationForANetwork) {
  ExecutionEngine EE;
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  Tensor inputs(ElemKind::FloatTy, {1, 4});
  inputs.getHandle() = {1, 1.2f, 0.5f, 1.3f};

  auto *A = mod.createVariable(ElemKind::FloatTy, {1, 4}, "A",
                               VisibilityKind::Public);
  auto *Ex = mod.createVariable(ElemKind::FloatTy, {1, 4}, "E",
                                VisibilityKind::Public);
  Node *O = F->createFullyConnected("fc", A, 4);
  O = F->createRELU("relu", O);
  O = F->createRegression("reg", O, Ex);

  F = ::glow::profileQuantization(F);

  EE.compile(CompilationMode::Infer, F);

  // TODO: Verify histogram itself, for now just verify min and max.
  // Run inference first time and capture tensor stats.
  EE.run({A}, {&inputs});

  QuantizationProfileNode *profile{nullptr};
  // Find QPN for node A.
  for (auto &node : F->getNodes()) {
    if (QuantizationProfileNode *QPN =
            llvm::dyn_cast<QuantizationProfileNode>(&node)) {
      Node *observedNode = node.getNthInput(0).getNode();
      if (observedNode == A) {
        profile = QPN;
        break;
      }
    }
  }

  EXPECT_TRUE(profile != nullptr);

  auto CI = profile->getComputationInfoVar()->getHandle<float>();
  float min = CI.raw(0);
  float max = CI.raw(1);
  EXPECT_NEAR(0.5, min, 0.00001);
  EXPECT_NEAR(1.3, max, 0.00001);

  // Run inference for the second time with new min and max.
  inputs.getHandle() = {0.2f, 1.6f, 0.5f, 1.3f};
  EE.run({A}, {&inputs});
  min = CI.raw(0);
  max = CI.raw(1);
  EXPECT_NEAR(0.2, min, 0.00001);
  EXPECT_NEAR(1.6, max, 0.00001);
}

TEST_P(BackendTest, simpleInference) {
  Tensor inputs(ElemKind::FloatTy, {1, 32, 32, 3});

  auto &mod = EE_.getModule();
  Function *F = mod.createFunction("main");
  F->setName("interpret");
  auto *input = mod.createVariable(ElemKind::FloatTy, {1, 32, 32, 3}, "input",
                                   VisibilityKind::Public);

  auto *ex = mod.createVariable(ElemKind::IndexTy, {1, 1}, "exp");

  auto *CV0 = F->createConv("conv1", input, 16, 5, 1, 2, 1);
  auto *RL0 = F->createRELU("relu1", CV0);
  auto *MP0 = F->createPoolMax("pool1", RL0, 2, 2, 0);

  auto *CV1 = F->createConv("conv2", MP0, 20, 5, 1, 2, 1);
  auto *RL1 = F->createRELU("relu2", CV1);
  auto *MP1 = F->createPoolMax("pool2", RL1, 2, 2, 0);

  auto *CV2 = F->createConv("conv3", MP1, 20, 5, 1, 2, 1);
  auto *RL2 = F->createRELU("relu3", CV2);
  auto *MP2 = F->createPoolMax("pool3", RL2, 2, 2, 0);

  auto *FCL1 = F->createFullyConnected("fc", MP2, 10);
  auto *RL3 = F->createRELU("relu4", FCL1);
  auto *SM = F->createSoftMax("sm", RL3, ex);
  F->createSave("ret", SM);

  EE_.compile(CompilationMode::Infer, F);

  /// Add a debug_action instruction to check that it can be
  /// processed by the interpreter.
  auto &M = EE_.getIR();
  IRBuilder builder(&M);
  builder.createDebugPrintInst("print1", *M.getWeights().begin());

  EE_.run({input}, {&inputs});
}

INSTANTIATE_TEST_CASE_P(Interpreter, BackendTest,
                        ::testing::Values(BackendKind::Interpreter));

#ifdef GLOW_WITH_CPU
INSTANTIATE_TEST_CASE_P(JIT, BackendTest, ::testing::Values(BackendKind::CPU));
#endif

#ifdef GLOW_WITH_OPENCL
INSTANTIATE_TEST_CASE_P(OpenCL, BackendTest, ::testing::Values(BackendKind::OpenCL));
#endif
