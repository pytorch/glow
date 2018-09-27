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
#include "glow/Graph/Context.h"
#include "glow/Graph/Graph.h"
#include "glow/IR/IRBuilder.h"

#include "gtest/gtest.h"

#include "llvm/ADT/STLExtras.h"
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
  Context ctx;
  auto &mod = EE.getModule();

  // Create a few nodes to make sure IR can be normally generated.
  Function *F = mod.createFunction("main");
  F->createSave(ctx, "save",
                mod.createPlaceholder(ElemKind::FloatTy, {2}, "A", false));

  EXPECT_DEATH(EE.save(CompilationMode::Infer, F, "output", "network"), "");
}

TEST(Interpreter, profileQuantizationForANetwork) {
  ExecutionEngine EE;
  Context ctx;
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  Tensor inputs(ElemKind::FloatTy, {1, 4});
  inputs.getHandle() = {1, 1.2f, 0.5f, 1.3f};

  auto *A = mod.createPlaceholder(ElemKind::FloatTy, {1, 4}, "A", false);
  auto *Ex = mod.createPlaceholder(ElemKind::FloatTy, {1, 4}, "E", false);
  Node *O = F->createFullyConnected(ctx, "fc", A, 4);
  O = F->createRELU("relu", O);
  O = F->createRegression("reg", O, Ex);

  F = ::glow::profileQuantization(F);

  ctx.allocate(A);
  ctx.allocate(Ex);
  EE.compile(CompilationMode::Infer, F, ctx);

  // TODO: Verify histogram itself, for now just verify min and max.
  // Run inference first time and capture tensor stats.
  updateVariables(ctx, {A}, {&inputs});
  EE.run();

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
  updateVariables(ctx, {A}, {&inputs});
  EE.run();
  min = CI.raw(0);
  max = CI.raw(1);
  EXPECT_NEAR(0.2, min, 0.00001);
  EXPECT_NEAR(1.6, max, 0.00001);
}

TEST_P(BackendTest, simpleInference) {
  Tensor inputs(ElemKind::FloatTy, {1, 32, 32, 3});
  Context ctx;

  auto &mod = EE_.getModule();
  Function *F = mod.createFunction("main");
  F->setName("interpret");
  auto *input =
      mod.createPlaceholder(ElemKind::FloatTy, {1, 32, 32, 3}, "input", false);

  auto *ex = mod.createPlaceholder(ElemKind::Int64ITy, {1, 1}, "exp", false);

  auto *CV0 = F->createConv(ctx, "conv1", input, 16, 5, 1, 2, 1);
  auto *RL0 = F->createRELU("relu1", CV0);
  auto *MP0 = F->createMaxPool("pool1", RL0, 2, 2, 0);

  auto *CV1 = F->createConv(ctx, "conv2", MP0, 20, 5, 1, 2, 1);
  auto *RL1 = F->createRELU("relu2", CV1);
  auto *MP1 = F->createMaxPool("pool2", RL1, 2, 2, 0);

  auto *CV2 = F->createConv(ctx, "conv3", MP1, 20, 5, 1, 2, 1);
  auto *RL2 = F->createRELU("relu3", CV2);
  auto *MP2 = F->createMaxPool("pool3", RL2, 2, 2, 0);

  auto *FCL1 = F->createFullyConnected(ctx, "fc", MP2, 10);
  auto *RL3 = F->createRELU("relu4", FCL1);
  auto *SM = F->createSoftMax("sm", RL3, ex);
  auto *S = F->createSave(ctx, "ret", SM);

  ctx.allocate(input);
  ctx.allocate(ex);
  ctx.allocate(S->getPlaceholder());
  EE_.compile(CompilationMode::Infer, F, ctx);

  updateVariables(ctx, {input}, {&inputs});
  EE_.run();
}

/// Test that the DebugPrint instruction works correctly for the backend. Note
/// that the backend being tested must inherit from BackendUsingGlowIR and
/// implement the compileIR() function for this test to work.
TEST_P(BackendTest, debugPrint) {
  Tensor input{0.0, 1.0, 2.0, 3.0};
  Module mod;
  Context ctx;
  Function *F = mod.createFunction("main");
  auto *IV = mod.createPlaceholder(input.getElementType(), input.dims(),
                                   "input", false);
  auto *IVTensor = ctx.allocate(IV);
  IVTensor->assign(&input);

  auto IR = llvm::make_unique<IRFunction>(F);
  IR->generateIR();
  IRBuilder(IR.get()).createDebugPrintInst("print", *IR->getWeights().begin());

  std::unique_ptr<BackendUsingGlowIR> backend(
      static_cast<BackendUsingGlowIR *>(createBackend(GetParam())));
  auto function = backend->compileIR(std::move(IR), ctx);
  function->execute();
}

/// This test checks that we can compile a function without depending on the
/// graph representation. We compile some function and then delete the function.
/// Later we execute the code and check that things work.
TEST_P(BackendTest, decoupleCodegenFromGraph) {
  Module mod;
  Context ctx;

  Function *F = mod.createFunction("main");
  auto *X = mod.createPlaceholder(ElemKind::FloatTy, {3}, "X", false);
  auto *XTensor = ctx.allocate(X);
  XTensor->getHandle() = {1., 2., 3.};
  auto *pow = F->createPow("Pow1", X, 2.0);
  auto *save = F->createSave(ctx, "save", pow);
  auto *saveTensor = ctx.allocate(save->getPlaceholder());
  EE_.compile(CompilationMode::Infer, F, ctx);

  // Erase all of the functions to ensure that the compiled code does not
  // depend on the graph.
  mod.eraseFunctions();

  // We can run the compiled code without having the graph representation
  // around.
  EE_.run();

  auto HX = saveTensor->getHandle();
  EXPECT_NEAR(HX.at({0}), 1, 1E-5);
  EXPECT_NEAR(HX.at({1}), 4, 1E-5);
  EXPECT_NEAR(HX.at({2}), 9, 1E-5);
}

/// Check that we can pass information to the execution engine using Placeholder
/// variables and read it back using Save nodes (in variables).
TEST_P(BackendTest, simplePlaceholderValue) {
  Tensor data{99.0, 35.0, 2.0, 3.0};
  auto &mod = EE_.getModule();
  Function *F = mod.createFunction("main");
  auto *input = mod.createPlaceholder(ElemKind::FloatTy, {4}, "input", false);
  Context ctx({input}, {&data});
  SaveNode *S = F->createSave(ctx, "ret", input);
  auto *STensor = ctx.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer, F, ctx);
  EE_.run();
  EXPECT_TRUE(STensor->isEqual(data));
}

/// Test the basic functionality of the context.
TEST(Context, basicContextTest) {
  Module mod;
  TypeRef ty = mod.uniqueType(ElemKind::FloatTy, {1, 32, 32, 3});

  Tensor T1(ty);

  // Create a context for some threaded execution.
  Context C;

  // Create a simple graph, just to have a few placeholders.
  Function *F = mod.createFunction("main");
  auto *input1 = mod.createPlaceholder(ty, "input1", false);
  auto *input2 = mod.createPlaceholder(ty, "input2", false);
  auto *input3 = mod.createPlaceholder(ty, "input3", false);
  auto *add = F->createAdd("add", input1, input2);
  auto *save = F->createSave(C, "ret", add);
  auto *savePlaceholder = save->getPlaceholder();
  C.allocate(savePlaceholder);

  C.insert(input1, std::move(T1));
  Tensor *I2 = C.allocate(input2);

  // Check that the right placeholders are found.
  EXPECT_TRUE(C.count(input1));
  EXPECT_TRUE(C.count(input2));
  EXPECT_TRUE(C.count(savePlaceholder));
  EXPECT_FALSE(C.count(nullptr));

  // Try to fetch some placeholders that exist and some that don't.
  auto *V1 = C.get(input1);
  auto *V2 = C.get(input2);
  auto *V3 = C.get(input3);
  auto *S = C.get(savePlaceholder);
  EXPECT_NE(V1, nullptr);
  EXPECT_NE(V2, nullptr);
  EXPECT_EQ(V3, nullptr);
  EXPECT_NE(S, nullptr);

  // The tensor that we got while allocating T2 is the same one that we got
  // while searching the context.
  EXPECT_EQ(I2, V2);

  // Check that all of the placeholders are allocated.
  C.allocate(input3);
  EXPECT_EQ(nullptr, C.getFirstUnallocated(mod.getPlaceholders()));

  // Check that some placeholders are unallocated.
  C.clear();
  EXPECT_NE(nullptr, C.getFirstUnallocated(mod.getPlaceholders()));

  // Check that all of the placeholders are allocated.
  C.allocate(mod.getPlaceholders());
  EXPECT_EQ(nullptr, C.getFirstUnallocated(mod.getPlaceholders()));
}

INSTANTIATE_TEST_CASE_P(Interpreter, BackendTest,
                        ::testing::Values(BackendKind::Interpreter));

#ifdef GLOW_WITH_CPU
INSTANTIATE_TEST_CASE_P(JIT, BackendTest, ::testing::Values(BackendKind::CPU));
#endif

#ifdef GLOW_WITH_OPENCL
INSTANTIATE_TEST_CASE_P(OpenCL, BackendTest,
                        ::testing::Values(BackendKind::OpenCL));
#endif
