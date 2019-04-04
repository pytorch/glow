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
#include "glow/Graph/PlaceholderBindings.h"
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
  PlaceholderBindings ctx;
  auto &mod = EE.getModule();

  // Create a few nodes to make sure IR can be normally generated.
  Function *F = mod.createFunction("main");
  F->createSave("save",
                mod.createPlaceholder(ElemKind::FloatTy, {2}, "A", false));

  CompilationOptions opts;
  opts.mode = CompilationMode::Infer;
  EXPECT_DEATH(EE.save(F, opts, "output", "network"), "");
}

TEST(Interpreter, profileQuantizationForANetwork) {
  ExecutionEngine EE;
  PlaceholderBindings ctx;
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  Tensor inputs(ElemKind::FloatTy, {1, 4});
  inputs.getHandle() = {1, 1.2f, 0.5f, 1.3f};

  auto *A = mod.createPlaceholder(ElemKind::FloatTy, {1, 4}, "A", false);
  auto *Ex = mod.createPlaceholder(ElemKind::FloatTy, {1, 4}, "E", false);
  Node *O = F->createFullyConnected(ctx, "fc", A, 4);
  O = F->createRELU("relu", O);
  O = F->createRegression("reg", O, Ex);

  F = ::glow::profileQuantization(ctx, F);

  ctx.allocate(A);
  ctx.allocate(Ex);
  EE.compile(CompilationMode::Infer, F);

  // TODO: Verify histogram itself, for now just verify min and max.
  // Run inference first time and capture tensor stats.
  updateInputPlaceholders(ctx, {A}, {&inputs});
  EE.run(ctx);

  QuantizationProfileNode *profile{nullptr};
  // Find QPN for node A.
  for (auto &node : F->getNodes()) {
    if (QuantizationProfileNode *QPN =
            llvm::dyn_cast<QuantizationProfileNode>(&node)) {
      Node *observedNode = QPN->getInput().getNode();
      if (observedNode == A) {
        profile = QPN;
        break;
      }
    }
  }

  EXPECT_TRUE(profile != nullptr);

  auto CI =
      ctx.get(profile->getComputationInfoPlaceholder())->getHandle<float>();
  float min = CI.raw(0);
  float max = CI.raw(1);
  EXPECT_NEAR(0.5, min, 0.00001);
  EXPECT_NEAR(1.3, max, 0.00001);

  // Run inference for the second time with new min and max.
  inputs.getHandle() = {0.2f, 1.6f, 0.5f, 1.3f};
  updateInputPlaceholders(ctx, {A}, {&inputs});
  EE.run(ctx);
  min = CI.raw(0);
  max = CI.raw(1);
  EXPECT_NEAR(0.2, min, 0.00001);
  EXPECT_NEAR(1.6, max, 0.00001);
}

TEST_P(BackendTest, simpleInference) {
  Tensor inputs(ElemKind::FloatTy, {1, 32, 32, 3});
  PlaceholderBindings ctx;

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
  auto *S = F->createSave("ret", SM);

  ctx.allocate(input);
  ctx.allocate(ex);
  ctx.allocate(S->getPlaceholder());
  EE_.compile(CompilationMode::Infer, F);

  updateInputPlaceholders(ctx, {input}, {&inputs});
  EE_.run(ctx);
}

/// Test that the DebugPrint instruction works correctly for the backend. Note
/// that the backend being tested must inherit from BackendUsingGlowIR and
/// implement the compileIR() function for this test to work.
TEST_P(BackendTest, debugPrint) {
  Tensor input{0.0, 1.0, 2.0, 3.0};
  Module mod;
  auto ctx = llvm::make_unique<ExecutionContext>();
  Function *F = mod.createFunction("main");
  auto *IV = mod.createPlaceholder(input.getElementType(), input.dims(),
                                   "input", false);
  auto *IVTensor = ctx->getPlaceholderBindings()->allocate(IV);
  IVTensor->assign(&input);
  auto *save = F->createSave("save", IV);
  ctx->getPlaceholderBindings()->allocate(save->getPlaceholder());

  std::unique_ptr<BackendUsingGlowIR> backend(
      static_cast<BackendUsingGlowIR *>(createBackend(GetParam())));
  auto IR = llvm::make_unique<IRFunction>(F);
  IR->generateIR(*backend.get());
  IRBuilder(IR.get()).createDebugPrintInst("print", *IR->getWeights().begin());

  auto function = backend->compileIR(std::move(IR));
  EE_.insertCompiledFunction("main", std::move(function));
  EE_.run(*ctx.get());
}

/// Test the compile method on the backend completes without error when
/// collectConstants is false.
TEST_P(BackendTest, CompileWithoutConstants) {
  Module mod;
  PlaceholderBindings ctx;
  Function *F = mod.createFunction("main");
  auto *X = mod.createPlaceholder(ElemKind::FloatTy, {3}, "X", false);
  auto *XTensor = ctx.allocate(X);
  XTensor->getHandle() = {1., 2., 3.};
  auto *pow = F->createPow("Pow1", X, 2.0);
  auto *save = F->createSave("save", pow);
  ctx.allocate(save->getPlaceholder());
  std::unique_ptr<Backend> backend(createBackend(GetParam()));
  CompilationOptions opts;
  opts.collectConstants = false;
  auto function = backend->compile(F, opts);
}

/// Test that the runtimeBundle includes only symbols from its function and not
/// the whole module.
TEST_P(BackendTest, BundleFunctionSymbolsOnly) {
  Module mod;
  PlaceholderBindings bindings;
  Function *F = mod.createFunction("main");
  auto *X = mod.createConstant(ElemKind::FloatTy, {3}, "X");
  X->getHandle() = {1., 2., 3.};
  auto *pow = F->createPow("Pow1", X, 2.0);
  auto *save = F->createSave("save", pow);
  bindings.allocate(save->getPlaceholder());
  PlaceholderBindings bindings2;
  Function *F2 = mod.createFunction("main2");
  auto *X2 = mod.createConstant(ElemKind::FloatTy, {3}, "X2");
  X2->getHandle() = {1., 2., 3.};
  auto *pow2 = F2->createPow("Pow2", X2, 2.0);
  auto *save2 = F2->createSave("save2", pow2);
  bindings2.allocate(save2->getPlaceholder());

  std::unique_ptr<Backend> backend(createBackend(GetParam()));
  auto function = backend->compile(F);
  auto function2 = backend->compile(F2);
  auto table1 = function->getRuntimeBundle().getSymbolTable();
  auto table2 = function2->getRuntimeBundle().getSymbolTable();
  /// Make sure no symbol in table1 is in table2.
  for (auto sym : table1) {
    auto it = table2.find(sym.first);
    EXPECT_TRUE(it == table2.end());
  }
}

/// Test that a shared constant is in the bundle of both functions.
TEST_P(BackendTest, BundleSharedConstant) {
  Module mod;
  PlaceholderBindings bindings;
  Function *F = mod.createFunction("main");
  auto *X = mod.createConstant(ElemKind::FloatTy, {3}, "X");
  X->getHandle() = {1., 2., 3.};
  auto *pow = F->createPow("Pow1", X, 2.0);
  auto *save = F->createSave("save", pow);
  bindings.allocate(save->getPlaceholder());
  PlaceholderBindings bindings2;
  Function *F2 = mod.createFunction("main2");
  auto *pow2 = F2->createPow("Pow2", X, 2.0);
  auto *save2 = F2->createSave("save2", pow2);
  bindings2.allocate(save2->getPlaceholder());

  std::unique_ptr<Backend> backend(createBackend(GetParam()));
  auto function = backend->compile(F);
  auto function2 = backend->compile(F2);
  auto table1 = function->getRuntimeBundle().getSymbolTable();
  auto table2 = function2->getRuntimeBundle().getSymbolTable();
  /// Make sure X is in both tables.
  auto it = table1.find(X->getName());
  auto it2 = table2.find(X->getName());
  EXPECT_TRUE(it != table1.end());
  EXPECT_TRUE(it2 != table2.end());
}

/// Test compiling a vector of functions completes without error.
TEST_P(BackendTest, compileVectorOfFunctions) {
  Module mod;
  std::vector<Function *> functions;
  for (unsigned int i = 0; i < 3; i++) {
    Function *F = mod.createFunction("function" + std::to_string(i));
    auto *X = mod.createPlaceholder(ElemKind::FloatTy, {3},
                                    "X" + std::to_string(i), false);
    auto *pow = F->createPow("Pow" + std::to_string(i), X, 2.0);
    F->createSave("save" + std::to_string(i), pow);
    functions.push_back(F);
  }
  std::unique_ptr<Backend> backend(createBackend(GetParam()));
  CompilationOptions opts;
  auto function = backend->compileFunctions(functions, opts);
}

/// This test checks that we can compile a function without depending on the
/// graph representation. We compile some function and then delete the function.
/// Later we execute the code and check that things work.
TEST_P(BackendTest, decoupleCodegenFromGraph) {
  Module mod;
  PlaceholderBindings ctx;

  Function *F = mod.createFunction("main");
  auto *X = mod.createPlaceholder(ElemKind::FloatTy, {3}, "X", false);
  auto *XTensor = ctx.allocate(X);
  XTensor->getHandle() = {1., 2., 3.};
  auto *pow = F->createPow("Pow1", X, 2.0);
  auto *save = F->createSave("save", pow);
  auto *saveTensor = ctx.allocate(save->getPlaceholder());
  EE_.compile(CompilationMode::Infer, F);

  // Collect constants to fill out the RuntimeBundle.
  EE_.getCompiledFunction().collectConstants(&mod);

  // Erase all of the functions to ensure that the compiled code does not
  // depend on the graph.
  mod.eraseFunctions();

  // We can run the compiled code without having the graph representation
  // around.
  EE_.run(ctx);

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
  PlaceholderBindings ctx({input}, {&data});
  SaveNode *S = F->createSave("ret", input);
  auto *STensor = ctx.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer, F);
  EE_.run(ctx);
  EXPECT_TRUE(STensor->isEqual(data));
}

/// Add and compile a network, then add and compile another so that the first
/// CompiledFunction does not know about every Placeholder in the module.
TEST_P(BackendTest, compileThenAddNetwork) {
  PlaceholderBindings bindings1, bindings2;

  auto &mod = EE_.getModule();
  Tensor inputs(ElemKind::FloatTy, {1, 10, 10, 3});
  inputs.getHandle().randomize(-2, 2, mod.getPRNG());

  // Create a simple graph that uses some placeholders.
  Function *F = mod.createFunction("main");
  auto *input =
      mod.createPlaceholder(ElemKind::FloatTy, {1, 10, 10, 3}, "in", false);

  auto *ex = mod.createPlaceholder(ElemKind::Int64ITy, {1, 1}, "exp", false);

  auto *FC = F->createFullyConnected(bindings1, "FC", input, 30);
  auto *RL = F->createRELU("RL2", FC);
  auto *SM = F->createSoftMax("sm", RL, ex);
  auto *S = F->createSave("ret", SM);

  Placeholder *FC_weights =
      llvm::dyn_cast<Placeholder>(FC->getWeights().getNode());

  EE_.compile(CompilationMode::Infer, F);

  // Recreate that graph in a different Function.
  Function *F2 = mod.createFunction("other");
  auto *input2 =
      mod.createPlaceholder(ElemKind::FloatTy, {1, 10, 10, 3}, "in", false);

  auto *ex2 = mod.createPlaceholder(ElemKind::Int64ITy, {1, 1}, "exp", false);
  auto *FC2 = F2->createFullyConnected(bindings2, "FC", input2, 30);

  // FC2 now has random weights we replace with FC1's weights so the output is
  // the same.
  Placeholder *FC2_weights =
      llvm::dyn_cast<Placeholder>(FC2->getWeights().getNode());
  bindings2.get(FC2_weights)->assign(bindings1.get(FC_weights));

  auto *RL2 = F2->createRELU("RL2", FC2);
  auto *SM2 = F2->createSoftMax("sm", RL2, ex2);
  auto *S2 = F2->createSave("ret", SM2);

  EE_.compile(CompilationMode::Infer, F2, /* clearOtherFunctions */ false);

  // Allocate all placeholders.
  bindings1.allocate(mod.getPlaceholders());
  bindings2.allocate(mod.getPlaceholders());
  updateInputPlaceholders(bindings1, {input}, {&inputs});
  updateInputPlaceholders(bindings2, {input2}, {&inputs});

  EE_.run(bindings1, "main");
  EE_.run(bindings2, "other");

  // The graphs were the same so their outputs should be as well.
  EXPECT_TRUE(bindings1.get(S->getPlaceholder())
                  ->isEqual(*bindings2.get(S2->getPlaceholder())));
}

/// Test the basic functionality of the bindings.
TEST(PlaceholderBindings, basicPlaceholderBindingsTest) {
  Module mod;
  TypeRef ty = mod.uniqueType(ElemKind::FloatTy, {1, 32, 32, 3});

  Tensor T1(ty);

  // Create a bindings for some threaded execution.
  PlaceholderBindings C;

  // Create a simple graph, just to have a few placeholders.
  Function *F = mod.createFunction("main");
  auto *input1 = mod.createPlaceholder(ty, "input1", false);
  auto *input2 = mod.createPlaceholder(ty, "input2", false);
  auto *input3 = mod.createPlaceholder(ty, "input3", false);
  auto *add = F->createAdd("add", input1, input2);
  auto *save = F->createSave("ret", add);
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
  // while searching the bindings.
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
#endif // GLOW_WITH_CPU

#ifdef GLOW_WITH_OPENCL
INSTANTIATE_TEST_CASE_P(OpenCL, BackendTest,
                        ::testing::Values(BackendKind::OpenCL));
#endif // GLOW_WITH_OPENCL
