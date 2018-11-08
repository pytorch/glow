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
  EE.run(ctx);
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
  EE.run(ctx);

  auto result = res->getHandle();
  std::vector<size_t> expectedDims = {3, 3};
  std::vector<float> expectedValues = {20, 20, 20, 40, 20, 20, 20, 20, 60};

  EXPECT_TRUE(result.dims().vec() == expectedDims);
  for (size_t i = 0; i < 3 * 3; i++) {
    EXPECT_FLOAT_EQ(result.raw(i), expectedValues[i]);
  }
}

/// Test loading BatchMatMul op from an ONNX model.
TEST(onnx, importBatchMatMul) {
  ExecutionEngine EE{BackendKind::Interpreter};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  std::string netFilename("tests/models/onnxModels/batch_matmul.onnxtxt");

  Context ctx;
  Placeholder *output;
  {
    Tensor inputs_0(ElemKind::FloatTy, {20, 7, 40});
    Tensor inputs_1(ElemKind::FloatTy, {20, 7, 40});
    inputs_0.getHandle().randomize(-3.0, 3.0, mod.getPRNG());
    inputs_1.getHandle().randomize(-3.0, 3.0, mod.getPRNG());
    ONNXModelLoader onnxLD(netFilename, {"inputs_0", "inputs_1"},
                           {&inputs_0.getType(), &inputs_1.getType()}, *F);
    output = onnxLD.getSingleOutput();

    ctx.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(ctx, &mod, {"inputs_0", "inputs_1"},
                                  {&inputs_0, &inputs_1});
  }
  auto *res = ctx.get(output);
  EE.compile(CompilationMode::Infer, F, ctx);
  EE.run(ctx);

  auto result = res->getHandle();
  std::vector<size_t> expectedDims = {20, 7, 7};
  EXPECT_EQ(result.dims().vec(), expectedDims);

  // High level check on the content of the graph.
  // We have 1 transpose, 20 * (matmul + 2 slices, 2 reshapes), 1 concat, 1
  // reshape, 1 save.
  EXPECT_EQ(F->getNodes().size(), 1 + 20 * 5 + 3);
  // With have 2 inputs and one outputs.
  EXPECT_EQ(mod.getPlaceholders().size(), 3);
  // Check that the graph has the expected shape,
  // starting from the output.
  // Batched matmul with broadcasted RHS are lowered
  // to a regular matmul, where LHS is reshaped from
  // a 3D tensor to a flattened matrix.
  auto *saveNode = getSaveNodeFromDest(output);
  auto *reshapeResult =
      llvm::dyn_cast<ReshapeNode>(saveNode->getInput().getNode());
  ASSERT_TRUE(reshapeResult);
  auto *concat =
      llvm::dyn_cast<ConcatNode>(reshapeResult->getInput().getNode());
  ASSERT_TRUE(concat);
  for (size_t i = 0; i < 20; ++i) {
    auto *matmulI =
        llvm::dyn_cast<MatMulNode>(concat->getNthInput(i).getNode());
    ASSERT_TRUE(matmulI);
    for (size_t j = 0; j < 2; ++j) {
      auto *reshape0 =
          llvm::dyn_cast<ReshapeNode>(matmulI->getNthInput(j).getNode());
      ASSERT_TRUE(reshape0);
      auto *slice0 = llvm::dyn_cast<SliceNode>(reshape0->getInput().getNode());
      ASSERT_TRUE(slice0);
    }
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
  EE.run(ctx);

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

/// Test loading Sum with more than 2 inputs
TEST(onnx, importSumN) {
  ExecutionEngine EE{BackendKind::Interpreter};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  std::string netFilename("tests/models/onnxModels/sumN.onnxtxt");

  Context ctx;
  Placeholder *output;
  {
    Tensor i0(ElemKind::FloatTy, {3});
    i0.getHandle() = {1, 2, 3};
    Tensor i1(ElemKind::FloatTy, {3});
    i1.getHandle() = {4, 5, 6};
    Tensor i2(ElemKind::FloatTy, {3});
    i2.getHandle() = {7, 8, 9};

    ONNXModelLoader onnxLD(netFilename, {"i0", "i1", "i2"},
                           {&i0.getType(), &i1.getType(), &i2.getType()}, *F);
    output = onnxLD.getSingleOutput();

    ctx.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(ctx, &mod, {"i0", "i1", "i2"},
                                  {&i0, &i1, &i2});
  }

  auto *res = ctx.get(output);
  EE.compile(CompilationMode::Infer, F, ctx);
  EE.run(ctx);

  auto result = res->getHandle();
  std::vector<size_t> expectedDims = {3};
  std::vector<float> expectedValues = {12, 15, 18};

  EXPECT_EQ(result.dims().vec(), expectedDims);
  for (size_t i = 0; i < 3; i++) {
    EXPECT_FLOAT_EQ(result.raw(i), expectedValues[i]);
  }

  // Verify the structure
  // Reshape x 3 -> Concat -> batchedReduceAdd -> Save
  ASSERT_EQ(6, F->getNodes().size());
  auto *saveNode = getSaveNodeFromDest(output);
  auto *batchedReduceAdd =
      llvm::dyn_cast<BatchedReduceAddNode>(saveNode->getInput().getNode());
  ASSERT_TRUE(batchedReduceAdd);
  auto *concat =
      llvm::dyn_cast<ConcatNode>(batchedReduceAdd->getNthInput(0).getNode());
  ASSERT_TRUE(concat);
  for (size_t i = 0; i < 3; ++i) {
    auto *reshape =
        llvm::dyn_cast<ReshapeNode>(concat->getNthInput(i).getNode());
    ASSERT_TRUE(reshape);
  }
}

/// Test loading Sum with one input and one output
TEST(onnx, importSum1) {
  ExecutionEngine EE{BackendKind::Interpreter};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  std::string netFilename("tests/models/onnxModels/sum1.onnxtxt");

  Context ctx;
  Placeholder *output;
  {
    Tensor x(ElemKind::FloatTy, {3});
    x.getHandle() = {1, 2, 3};
    ONNXModelLoader onnxLD(netFilename, {"x"}, {&x.getType()}, *F);
    output = onnxLD.getSingleOutput();

    ctx.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(ctx, &mod, {"x"}, {&x});
  }

  auto *res = ctx.get(output);
  EE.compile(CompilationMode::Infer, F, ctx);
  EE.run(ctx);

  auto result = res->getHandle();
  std::vector<size_t> expectedDims = {3};
  std::vector<float> expectedValues = {1, 2, 3};

  EXPECT_EQ(result.dims().vec(), expectedDims);
  for (size_t i = 0; i < 3; i++) {
    EXPECT_FLOAT_EQ(result.raw(i), expectedValues[i]);
  }

  // Verify structure: input -> Save -> output
  ASSERT_EQ(mod.getPlaceholders().size(), 2);
  ASSERT_EQ(F->getNodes().size(), 1);
  auto *save = getSaveNodeFromDest(output);
  ASSERT_TRUE(llvm::isa<Placeholder>(save->getInput().getNode()));
}

/// Test loading LengthsToRanges from an ONNX model.
TEST(onnx, importLengthsToRanges) {
  ExecutionEngine EE;
  auto &mod = EE.getModule();
  auto *F = mod.createFunction("main");
  std::string netFilename("tests/models/onnxModels/lengths_to_ranges.onnxtxt");
  Placeholder *output;
  {
    Tensor lengths(ElemKind::Int64ITy, {4});
    ONNXModelLoader onnxLD(netFilename, {"lengths"}, {&lengths.getType()}, *F);
    output = onnxLD.getSingleOutput();
  }
  // Verify structure: PH -> LengthsToRanges -> Save -> PH.
  ASSERT_EQ(mod.getPlaceholders().size(), 2);
  ASSERT_EQ(F->getNodes().size(), 2);
  auto *save = getSaveNodeFromDest(output);
  auto *LTR = llvm::dyn_cast<LengthsToRangesNode>(save->getInput().getNode());
  ASSERT_TRUE(LTR);
  ASSERT_TRUE(llvm::isa<Placeholder>(LTR->getLengths()));
}

/// Test loading ReplaceNaN op from an ONNX model.
/// Test with arg value = 1.0.
TEST(onnx, importReplaceNaN) {
  ExecutionEngine EE{BackendKind::Interpreter};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string netFilename("tests/models/onnxModels/replaceNaN.onnxtxt");

  Context ctx;
  Placeholder *output;
  Tensor x(ElemKind::FloatTy, {3, 3});

  {
    ONNXModelLoader onnxLD(netFilename, {"x"}, {&x.getType()}, *F);
    output = onnxLD.getSingleOutput();
    ctx.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(ctx, &mod, {"x"}, {&x});
  }

  // Verify structure: Input, IsNan, Splat -> Select -> Save.
  ASSERT_EQ(mod.getPlaceholders().size(), 2);
  ASSERT_EQ(F->getNodes().size(), 4);
  auto *save = getSaveNodeFromDest(output);
  auto *select = llvm::dyn_cast<SelectNode>(save->getInput().getNode());
  ASSERT_TRUE(select);
  auto *isNaN = llvm::dyn_cast<IsNaNNode>(select->getNthInput(0).getNode());
  ASSERT_TRUE(isNaN);
  auto *splat = llvm::dyn_cast<SplatNode>(select->getNthInput(1).getNode());
  ASSERT_TRUE(splat);
  auto *input = llvm::dyn_cast<Placeholder>(select->getNthInput(2).getNode());
  ASSERT_EQ(input, mod.getPlaceholderByName("x"));
}

/// Test loading SparseToDense op from an ONNX model.
TEST(onnx, importSparseToDense) {
  ExecutionEngine EE{BackendKind::Interpreter};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string netFilename("tests/models/onnxModels/sparseToDense.onnxtxt");

  Context ctx;
  Placeholder *output;

  // Create inputs.
  constexpr size_t kNumIndices = 5;
  constexpr size_t kMaxIndex = 20;
  constexpr size_t kRows = 10;
  constexpr size_t kCols = 5;
  Tensor indices(ElemKind::Int64ITy, {kNumIndices});
  Tensor values(ElemKind::FloatTy, {kNumIndices, kRows, kCols});
  Tensor dataToInferDim(ElemKind::FloatTy, {kMaxIndex, kRows, kCols});

  // Load model.
  {
    ONNXModelLoader onnxLD(
        netFilename, {"indices", "values", "dataToInferDim"},
        {&indices.getType(), &values.getType(), &dataToInferDim.getType()}, *F);
    output = onnxLD.getSingleOutput();
  }

  // Verify structure: Inputs -> SparseToDense -> Save.
  ASSERT_EQ(mod.getPlaceholders().size(), 4);
  ASSERT_EQ(F->getNodes().size(), 2);

  auto *save = getSaveNodeFromDest(output);
  auto *out = save->getPlaceholder();
  EXPECT_TRUE(out->dims().vec() == dataToInferDim.dims().vec());

  auto *STD = llvm::dyn_cast<SparseToDenseNode>(save->getInput().getNode());
  ASSERT_TRUE(STD);
  auto *idx = llvm::dyn_cast<Placeholder>(STD->getIndices().getNode());
  EXPECT_EQ(idx, mod.getPlaceholderByName("indices"));
  auto *vals = llvm::dyn_cast<Placeholder>(STD->getValues().getNode());
  EXPECT_EQ(vals, mod.getPlaceholderByName("values"));
}

/// Test loading LengthsSum from an ONNX model.
TEST(onnx, importLengthsSum) {
  ExecutionEngine EE;
  auto &mod = EE.getModule();
  auto *F = mod.createFunction("main");
  std::string netFilename("tests/models/onnxModels/lengths_sum.onnxtxt");
  Placeholder *output;
  {
    Tensor data(ElemKind::FloatTy, {10, 2, 3});
    Tensor lengths(ElemKind::Int64ITy, {5});
    ONNXModelLoader onnxLD(netFilename, {"data", "lengths"},
                           {&data.getType(), &lengths.getType()}, *F);
    output = onnxLD.getSingleOutput();
  }
  // Verify structure: PH, PH -> LengthsSum -> Save -> PH.
  ASSERT_EQ(mod.getPlaceholders().size(), 3);
  ASSERT_EQ(F->getNodes().size(), 2);
  auto *save = getSaveNodeFromDest(output);
  auto *LS = llvm::dyn_cast<LengthsSumNode>(save->getInput().getNode());
  ASSERT_TRUE(LS);
  ASSERT_TRUE(llvm::isa<Placeholder>(LS->getData()));
  ASSERT_TRUE(llvm::isa<Placeholder>(LS->getLengths()));
}

/// Test loading a FCTransposed node: I * W + B, where I is need to be flatten.
TEST(onnx, FCTransposedWithFlatten) {
  ExecutionEngine EE{BackendKind::Interpreter};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string netFilename("tests/models/onnxModels/FCTransposed.onnxtxt");

  Placeholder *output;

  {
    Tensor data(ElemKind::FloatTy, {2, 1, 3});
    data.getHandle() = {1, 2, 3, 4, 5, 6};
    ONNXModelLoader onnxLD(netFilename, {"data"}, {&data.getType()}, *F);
    output = onnxLD.getSingleOutput();
  }

  // High level check on the content of the graph. We have 1 reshape, 1 FC,
  // and 1 save.
  EXPECT_EQ(F->getNodes().size(), 3);
  auto *saveNode = getSaveNodeFromDest(output);
  auto *fcNode =
      llvm::dyn_cast<FullyConnectedNode>(saveNode->getInput().getNode());
  ASSERT_TRUE(fcNode);
  auto *reshape = llvm::dyn_cast<ReshapeNode>(fcNode->getInput());
  ASSERT_TRUE(reshape);
}

/// Test loading ExpandDims from an ONNX model.
TEST(onnx, expandDims) {
  ExecutionEngine EE;
  auto &mod = EE.getModule();
  auto *F = mod.createFunction("main");
  std::string netFilename("tests/models/onnxModels/expandDims.onnxtxt");
  Placeholder *output;
  {
    Tensor x(ElemKind::FloatTy, {2, 2});
    ONNXModelLoader onnxLD(netFilename, {"x"}, {&x.getType()}, *F);
    output = onnxLD.getSingleOutput();
  }

  // Verify structure: PH -> Reshape -> Save -> PH.
  ASSERT_EQ(mod.getPlaceholders().size(), 2);
  ASSERT_EQ(F->getNodes().size(), 2);
  auto *save = getSaveNodeFromDest(output);
  auto *reshape = llvm::dyn_cast<ReshapeNode>(save->getInput().getNode());
  ASSERT_TRUE(reshape);
  EXPECT_TRUE(reshape->getDims().equals({1, 2, 2, 1}));
}
