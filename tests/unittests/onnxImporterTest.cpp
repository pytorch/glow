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
#include "ImporterTestUtils.h"
#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Graph/Context.h"
#include "glow/Graph/Graph.h"
#include "glow/Importer/ONNXModelLoader.h"
#include "gtest/gtest.h"

using namespace glow;

/// Test loading conv op from a ONNX model.
/// The input is N*C*H*W (1*1*3*3), the kernels is {2, 2},
/// strides is {1, 1}, pads is {1, 1, 1, 1}, group is 1.
TEST(onnx, importConv) {
  ExecutionEngine EE{BackendKind::Interpreter};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetFilename("tests/models/onnxModels/simpleConv.onnxtxt");

  Context ctx;
  Placeholder *graphOutputVar;
  // Destroy the loader after the graph is loaded since the following execution
  // will not depend on anyting from the loader.
  {
    Tensor data;
    getNCHWData(&data, 1, 1, 3, 3);
    ONNXModelLoader onnxLD(NetFilename, {"data"}, {&data.getType()}, *F);
    graphOutputVar = onnxLD.getSingleOutput();
    ctx.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(ctx, &mod, {"data"}, {&data});
  }

  // ONNX importer loads a conv node and converts it to 4 ops:
  // Transpose (input)   -> Conv -> Transpose
  // Transpose (filter) ->
  // A save node is added in the network as well. Therefore there are 5 nodes:
  // Transpose (input)   -> Conv -> Transpose -> Save
  // Transpose (filter) ->
  // Note that in case the convolution filter is a constant tensor, the filter
  // transpose node will be later optimized out by the optimizer.
  EXPECT_EQ(F->getNodes().size(), 5);
  EXPECT_EQ(mod.getPlaceholders().size(), 2);
  EXPECT_EQ(mod.getConstants().size(), 2);

  auto *saveNode = getSaveNodeFromDest(graphOutputVar);
  auto *node = saveNode->getInput().getNode();

  EXPECT_TRUE(node->getKind() == Kinded::Kind::TransposeNodeKind);
  auto *convNode = llvm::dyn_cast<TransposeNode>(node)->getInput().getNode();

  EXPECT_TRUE(convNode->getKind() == Kinded::Kind::ConvolutionNodeKind);
  auto *tInNode =
      llvm::dyn_cast<ConvolutionNode>(convNode)->getInput().getNode();
  auto *tFilterNode =
      llvm::dyn_cast<ConvolutionNode>(convNode)->getFilter().getNode();
  EXPECT_TRUE(tInNode->getKind() == Kinded::Kind::TransposeNodeKind);
  EXPECT_TRUE(tFilterNode->getKind() == Kinded::Kind::TransposeNodeKind);

  EE.compile(CompilationMode::Infer, F, ctx);
  EE.run();
  auto result = ctx.get(graphOutputVar)->getHandle();
  std::vector<size_t> expectedDims = {1, 1, 4, 4};
  std::vector<float> expectedValues = {2,  3,  5,  4,  5, 10, 14, 9,
                                       11, 22, 26, 15, 8, 15, 17, 10};
  EXPECT_TRUE(result.dims().vec() == expectedDims);
  for (size_t i = 0; i < 4 * 4; i++)
    EXPECT_FLOAT_EQ(result.raw(i), expectedValues[i]);
}

TEST(onnx, importAveragePool3D) {
  ExecutionEngine EE{BackendKind::Interpreter};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetFilename("tests/models/onnxModels/averagePool3D.onnxtxt");

  // Destroy the loader after the graph is loaded since the following execution
  // will not depend on anyting from the loader.
  {
    Tensor data(ElemKind::FloatTy, {1, 3, 32, 32, 32});
    EXPECT_DEATH(ONNXModelLoader(NetFilename, {"x"}, {&data.getType()}, *F),
                 "");
  }
}

/// Test loading clip op from an ONNX model.
/// Test with arg min = 20.0 max = 60.0
TEST(onnx, importClip) {
  ExecutionEngine EE{BackendKind::Interpreter};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string netFilename("tests/models/onnxModels/clip.onnxtxt");

  Context ctx;
  Placeholder *output;
  {
    Tensor x(ElemKind::FloatTy, {3, 3});
    x.getHandle() = {1, 2, 3, 40, 5, 6, 7, 8, 90};
    ONNXModelLoader onnxLD(netFilename, {"x"}, {&x.getType()}, *F);
    output = onnxLD.getSingleOutput();
    ctx.allocate(mod.getPlaceholders());

    updateInputPlaceholdersByName(ctx, &mod, {"x"}, {&x});
  }

  auto *res = ctx.get(output);
  EE.compile(CompilationMode::Infer, F, ctx);
  EE.run();

  auto result = res->getHandle();
  std::vector<size_t> expectedDims = {3, 3};
  std::vector<float> expectedValues = {20, 20, 20, 40, 20, 20, 20, 20, 60};

  EXPECT_TRUE(result.dims().vec() == expectedDims);
  for (size_t i = 0; i < 3 * 3; i++) {
    EXPECT_FLOAT_EQ(result.raw(i), expectedValues[i]);
  }
}

/// Test loading BatchBoxCox op from an ONNX model.
TEST(onnx, importBatchBoxCox) {
  ExecutionEngine EE{BackendKind::Interpreter};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string netFilename("tests/models/onnxModels/batchBoxCox.onnxtxt");

  Context ctx;
  Placeholder *output;

  // Make input tensors.
  const size_t kRows = 3;
  const size_t kCols = 3;
  Tensor data(ElemKind::FloatTy, {kRows, kCols});
  Tensor lambda1(ElemKind::FloatTy, {kCols});
  Tensor lambda2(ElemKind::FloatTy, {kCols});
  auto dataH = data.getHandle();
  auto lambda1H = lambda1.getHandle();
  auto lambda2H = lambda2.getHandle();

  // Fill inputs with random positive values.
  dataH.randomize(0.0, 5.0, mod.getPRNG());
  lambda1H.randomize(1.0, 2.0, mod.getPRNG());
  lambda2H.randomize(1.0, 2.0, mod.getPRNG());

  // Zero out every other element to lambda1 to test that case of the transform.
  for (size_t i = 0; i < kCols; i += 2) {
    lambda1H.at({i}) = 0;
  }

  {
    ONNXModelLoader onnxLD(
        netFilename, {"data", "lambda1", "lambda2"},
        {&data.getType(), &lambda1.getType(), &lambda2.getType()}, *F);
    output = onnxLD.getSingleOutput();
    ctx.allocate(mod.getPlaceholders());

    updateInputPlaceholdersByName(ctx, &mod, {"data", "lambda1", "lambda2"},
                                  {&data, &lambda1, &lambda2});
  }

  auto *res = ctx.get(output);
  EE.compile(CompilationMode::Infer, F, ctx);
  EE.run();

  auto result = res->getHandle();

  // Output should have the same dims as the inputs.
  EXPECT_TRUE(result.dims().vec() == data.dims().vec());

  // Compute elementwise Box-Cox transform and compare with corresponding
  // element of result.
  for (size_t i = 0; i < kRows; ++i) {
    for (size_t j = 0; j < kCols; ++j) {
      float d = dataH.at({i, j});
      float l1 = lambda1H.at({j});
      float l2 = lambda2H.at({j});

      float tmp = std::max(d + l2, 1e-6f);
      float y = 0;

      if (l1 == 0) {
        // Clip argument to log and pow at 1e-6 to avoid saturation.
        y = std::log(tmp);
      } else {
        y = (std::pow(tmp, l1) - 1) / l1;
      }

      EXPECT_FLOAT_EQ(y, result.at({i, j}));
    }
  }
}

/// Test loading DotProduct op from an ONNX model.
TEST(onnx, importDotProduct) {
  ExecutionEngine EE{BackendKind::Interpreter};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string netFilename("tests/models/onnxModels/dot_product.onnxtxt");

  Placeholder *output;
  {
    Tensor x(ElemKind::FloatTy, {3, 3});
    Tensor y(ElemKind::FloatTy, {3, 3});

    ONNXModelLoader onnxLD(netFilename, {"x", "y"},
                           {&x.getType(), &y.getType()}, *F);
    output = onnxLD.getSingleOutput();
  }

  // Just verify the structure.
  // SaveNode + MulNode + BatchedReduceAddNode.
  ASSERT_EQ(3, F->getNodes().size());
  auto *saveNode = getSaveNodeFromDest(output);
  auto *saveInput = saveNode->getInput().getNode();
  ASSERT_TRUE(llvm::isa<BatchedReduceAddNode>(saveInput));

  auto *batchedReduceAdd = llvm::cast<BatchedReduceAddNode>(saveInput);
  ASSERT_TRUE(llvm::isa<MulNode>(batchedReduceAdd->getBatch()));
}
