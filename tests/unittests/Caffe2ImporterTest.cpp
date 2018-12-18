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
#include "glow/Graph/Graph.h"
#include "glow/Importer/Caffe2ModelLoader.h"
#include "gtest/gtest.h"

using namespace glow;

/// Test loading conv op from a Caffe2 model.
/// The input is N*C*H*W (1*1*3*3), the kernel is 2,
/// stride is 1, pad is 1, group is 1.
TEST(caffe2, importConv) {
  ExecutionEngine EE{BackendKind::Interpreter};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename("tests/models/caffe2Models/predict_net.pbtxt");
  std::string NetWeightFilename("tests/models/caffe2Models/init_net.pbtxt");

  Placeholder *output;
  Context ctx;

  // Destroy the loader after the graph is loaded since the following execution
  // will not depend on anyting from the loader.
  {
    Tensor data;
    getNCHWData(&data, 1, 1, 3, 3);
    Caffe2ModelLoader caffe2LD(NetDescFilename, NetWeightFilename, {"data"},
                               {&data.getType()}, *F);
    output = EXIT_ON_ERR(caffe2LD.getSingleOutput());

    ctx.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(ctx, &mod, {"data"}, {&data});
  }

  auto res = ctx.get(output);
  EE.compile(CompilationMode::Infer, F);

  EE.run(ctx);
  auto result = res->getHandle();
  std::vector<size_t> expectedDims = {1, 1, 4, 4};
  std::vector<float> expectedValues = {2,  3,  5,  4,  5, 10, 14, 9,
                                       11, 22, 26, 15, 8, 15, 17, 10};
  EXPECT_TRUE(result.dims().vec() == expectedDims);
  for (size_t i = 0; i < 4 * 4; i++)
    EXPECT_FLOAT_EQ(result.raw(i), expectedValues[i]);
}

/// Test loading conv op from a Caffe2 model.
/// The input is N*H*W*C (1*3*3*1), the kernel is 2,
/// stride is 1, pad is 1, group is 1.
TEST(caffe2, convNHWC) {
  ExecutionEngine EE{BackendKind::Interpreter};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename(
      "tests/models/caffe2Models/conv_nhwc_predict_net.pbtxt");
  std::string NetWeightFilename(
      "tests/models/caffe2Models/conv_nhwc_init_net.pbtxt");

  Placeholder *output;
  Context ctx;

  Tensor inputs(ElemKind::FloatTy, {1, 3, 3, 1});

  // Destroy the loader after the graph is loaded since the following execution
  // will not depend on anyting from the loader.
  {
    Caffe2ModelLoader caffe2LD(NetDescFilename, NetWeightFilename, {"inputs"},
                               {&inputs.getType()}, *F);
    output = EXIT_ON_ERR(caffe2LD.getSingleOutput());
  }

  // High level check on the content of the graph. We have 1 conv and 1 save.
  EXPECT_EQ(F->getNodes().size(), 2);
  auto *saveNode = getSaveNodeFromDest(output);
  auto *convNode =
      llvm::dyn_cast<ConvolutionNode>(saveNode->getInput().getNode());
  ASSERT_TRUE(convNode);

  // We have 2 placeholders:  1 input and 1 output.
  EXPECT_EQ(mod.getPlaceholders().size(), 2);
  // We have 2 constants: Weights and bias.
  EXPECT_EQ(mod.getConstants().size(), 2);
}

/// Test loading MaxPool with NHWC order input.
TEST(caffe2, maxPoolNHWC) {
  ExecutionEngine EE{BackendKind::Interpreter};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename(
      "tests/models/caffe2Models/maxpool_nhwc_predict_net.pbtxt");
  std::string NetWeightFilename(
      "tests/models/caffe2Models/empty_init_net.pbtxt");

  Placeholder *output;
  Context ctx;

  Tensor inputs(ElemKind::FloatTy, {1, 3, 3, 1});

  // Destroy the loader after the graph is loaded since the following execution
  // will not depend on anyting from the loader.
  {
    Caffe2ModelLoader caffe2LD(NetDescFilename, NetWeightFilename, {"inputs"},
                               {&inputs.getType()}, *F);
    output = EXIT_ON_ERR(caffe2LD.getSingleOutput());
  }

  // High level check on the content of the graph. We have 1 maxpool and 1 save.
  EXPECT_EQ(F->getNodes().size(), 2);
  auto *saveNode = getSaveNodeFromDest(output);
  auto *maxPoolNode =
      llvm::dyn_cast<MaxPoolNode>(saveNode->getInput().getNode());
  ASSERT_TRUE(maxPoolNode);

  // We have 2 placeholders:  1 input and 1 output.
  EXPECT_EQ(mod.getPlaceholders().size(), 2);
  // We have 0 constants.
  EXPECT_EQ(mod.getConstants().size(), 0);
}

/// Test loading MaxPool with default NCHW order input.
TEST(caffe2, maxPool) {
  ExecutionEngine EE{BackendKind::Interpreter};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename(
      "tests/models/caffe2Models/maxpool_predict_net.pbtxt");
  std::string NetWeightFilename(
      "tests/models/caffe2Models/empty_init_net.pbtxt");

  Placeholder *output;
  Context ctx;

  Tensor inputs(ElemKind::FloatTy, {1, 3, 3, 1});

  // Destroy the loader after the graph is loaded since the following execution
  // will not depend on anyting from the loader.
  {
    Caffe2ModelLoader caffe2LD(NetDescFilename, NetWeightFilename, {"inputs"},
                               {&inputs.getType()}, *F);
    output = EXIT_ON_ERR(caffe2LD.getSingleOutput());
  }

  // High level check on the content of the graph. We have 1 maxpool, 1 save
  // and 2 transpose.
  EXPECT_EQ(F->getNodes().size(), 4);
  auto *saveNode = getSaveNodeFromDest(output);
  auto *transNode1 =
      llvm::dyn_cast<TransposeNode>(saveNode->getInput().getNode());
  ASSERT_TRUE(transNode1);
  auto *maxPoolNode =
      llvm::dyn_cast<MaxPoolNode>(transNode1->getInput().getNode());
  ASSERT_TRUE(maxPoolNode);
  auto *transNode2 =
      llvm::dyn_cast<TransposeNode>(maxPoolNode->getInput().getNode());
  ASSERT_TRUE(transNode2);

  // We have 2 placeholders:  1 input and 1 output.
  EXPECT_EQ(mod.getPlaceholders().size(), 2);
  // We have 0 constants.
  EXPECT_EQ(mod.getConstants().size(), 0);
}

/// Test loading AvgPool with NHWC order input.
TEST(caffe2, avgPoolNHWC) {
  ExecutionEngine EE{BackendKind::Interpreter};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename(
      "tests/models/caffe2Models/avgpool_nhwc_predict_net.pbtxt");
  std::string NetWeightFilename(
      "tests/models/caffe2Models/empty_init_net.pbtxt");

  Placeholder *output;
  Context ctx;

  Tensor inputs(ElemKind::FloatTy, {1, 3, 3, 1});

  // Destroy the loader after the graph is loaded since the following execution
  // will not depend on anyting from the loader.
  {
    Caffe2ModelLoader caffe2LD(NetDescFilename, NetWeightFilename, {"inputs"},
                               {&inputs.getType()}, *F);
    output = EXIT_ON_ERR(caffe2LD.getSingleOutput());
  }

  // High level check on the content of the graph. We have 1 maxpool and 1 save.
  EXPECT_EQ(F->getNodes().size(), 2);
  auto *saveNode = getSaveNodeFromDest(output);
  auto *avgPoolNode =
      llvm::dyn_cast<AvgPoolNode>(saveNode->getInput().getNode());
  ASSERT_TRUE(avgPoolNode);

  // We have 2 placeholders:  1 input and 1 output.
  EXPECT_EQ(mod.getPlaceholders().size(), 2);
  // We have 0 constants.
  EXPECT_EQ(mod.getConstants().size(), 0);
}

/// Test loading AveragePool with default NCHW order input.
TEST(caffe2, avgPool) {
  ExecutionEngine EE{BackendKind::Interpreter};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename(
      "tests/models/caffe2Models/avgpool_predict_net.pbtxt");
  std::string NetWeightFilename(
      "tests/models/caffe2Models/empty_init_net.pbtxt");

  Placeholder *output;
  Context ctx;

  Tensor inputs(ElemKind::FloatTy, {1, 3, 3, 1});

  // Destroy the loader after the graph is loaded since the following execution
  // will not depend on anyting from the loader.
  {
    Caffe2ModelLoader caffe2LD(NetDescFilename, NetWeightFilename, {"inputs"},
                               {&inputs.getType()}, *F);
    output = EXIT_ON_ERR(caffe2LD.getSingleOutput());
  }

  // High level check on the content of the graph. We have 1 maxpool, 1 save
  // and 2 transpose.
  EXPECT_EQ(F->getNodes().size(), 4);
  auto *saveNode = getSaveNodeFromDest(output);
  auto *transNode1 =
      llvm::dyn_cast<TransposeNode>(saveNode->getInput().getNode());
  ASSERT_TRUE(transNode1);
  auto *avgPoolNode =
      llvm::dyn_cast<AvgPoolNode>(transNode1->getInput().getNode());
  ASSERT_TRUE(avgPoolNode);
  auto *transNode2 =
      llvm::dyn_cast<TransposeNode>(avgPoolNode->getInput().getNode());
  ASSERT_TRUE(transNode2);

  // We have 2 placeholders:  1 input and 1 output.
  EXPECT_EQ(mod.getPlaceholders().size(), 2);
  // We have 0 constants.
  EXPECT_EQ(mod.getConstants().size(), 0);
}

/// Test loading a concat node with add_axis.
/// Concat nodes with add_axis have a different semantic
/// than the plain glow concat.
/// concat A(dim0, dim1), B(dim0, dim1), ... 1, add_axis = 1
/// res = A, B...
/// C2 shape: dim0, #input, dim1, i.e., three dimensions.
/// Glow shape: dim0, #input x dim1, i.e., two dimensions.
///
/// To fill the gap between the two, glow issues a reshape
/// right after its concat.
TEST(caffe2, concatAddAxis) {
  ExecutionEngine EE{BackendKind::Interpreter};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename(
      "tests/models/caffe2Models/concat_add_axis_predict_net.pbtxt");
  std::string NetWeightFilename(
      "tests/models/caffe2Models/empty_init_net.pbtxt");

  Context ctx;

  Placeholder *output;
  Tensor inputs_0(ElemKind::FloatTy, {10, 7});
  Tensor inputs_1(ElemKind::FloatTy, {10, 7});
  Tensor inputs_2(ElemKind::FloatTy, {10, 7});
  inputs_0.getHandle().randomize(-3.0, 3.0, mod.getPRNG());
  inputs_1.getHandle().randomize(-3.0, 3.0, mod.getPRNG());
  inputs_2.getHandle().randomize(-3.0, 3.0, mod.getPRNG());
  // Destroy the loader after the graph is loaded since the following execution
  // will not depend on anyting from the loader.
  {
    Caffe2ModelLoader caffe2LD(
        NetDescFilename, NetWeightFilename,
        {"inputs_0", "inputs_1", "inputs_2"},
        {&inputs_0.getType(), &inputs_1.getType(), &inputs_2.getType()}, *F);
    output = EXIT_ON_ERR(caffe2LD.getSingleOutput());

    ctx.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(ctx, &mod,
                                  {"inputs_0", "inputs_1", "inputs_2"},
                                  {&inputs_0, &inputs_1, &inputs_2});
  }

  // Check that the shape of the output matches what Caffe2 expects.
  std::vector<size_t> expectedDims = {10, 3, 7};
  EXPECT_TRUE(output->dims().vec() == expectedDims);

  auto res = ctx.get(output);
  EE.compile(CompilationMode::Infer, F);

  EE.run(ctx);
  // High level check on the content of the graph.
  // We have 1 reshape, 1 concat, and 1 save.
  EXPECT_EQ(F->getNodes().size(), 3);
  // With have three inputs and one outputs.
  EXPECT_EQ(mod.getPlaceholders().size(), 4);

  // Check that the graph has the expected shape,
  // starting from the output.
  auto *saveNode = getSaveNodeFromDest(output);
  auto *reshape = llvm::dyn_cast<ReshapeNode>(saveNode->getInput().getNode());
  ASSERT_TRUE(reshape);
  auto *concat = llvm::dyn_cast<ConcatNode>(reshape->getInput());
  ASSERT_TRUE(concat);
  // We will check that the inputs are correct within
  // the next loop.

  auto result = res->getHandle();

  // Check that the output matches the concatenation of
  // all the inputs.
  Tensor *inputs[] = {&inputs_0, &inputs_1, &inputs_2};
  for (size_t i = 0; i < 3; ++i) {
    const auto inputsHandle = inputs[i]->getHandle();
    ASSERT_TRUE(llvm::isa<Placeholder>(concat->getInputs()[i]));

    for (size_t row = 0; row < 10; ++row) {
      for (size_t column = 0; column < 7; ++column) {
        EXPECT_FLOAT_EQ(result.at({row, i, column}),
                        inputsHandle.at({row, column}));
      }
    }
  }
}

/// Test loading a regular concat node.
TEST(caffe2, concat) {
  ExecutionEngine EE{BackendKind::Interpreter};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename(
      "tests/models/caffe2Models/concat_predict_net.pbtxt");
  std::string NetWeightFilename(
      "tests/models/caffe2Models/empty_init_net.pbtxt");

  Context ctx;
  Placeholder *output;
  Tensor inputs_0(ElemKind::FloatTy, {10, 7});
  Tensor inputs_1(ElemKind::FloatTy, {10, 12});
  Tensor inputs_2(ElemKind::FloatTy, {10, 5});
  inputs_0.getHandle().randomize(-3.0, 3.0, mod.getPRNG());
  inputs_1.getHandle().randomize(-3.0, 3.0, mod.getPRNG());
  inputs_2.getHandle().randomize(-3.0, 3.0, mod.getPRNG());
  // Destroy the loader after the graph is loaded since the following execution
  // will not depend on anyting from the loader.
  {
    Caffe2ModelLoader caffe2LD(
        NetDescFilename, NetWeightFilename,
        {"inputs_0", "inputs_1", "inputs_2"},
        {&inputs_0.getType(), &inputs_1.getType(), &inputs_2.getType()}, *F);
    output = EXIT_ON_ERR(caffe2LD.getSingleOutput());

    ctx.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(ctx, &mod,
                                  {"inputs_0", "inputs_1", "inputs_2"},
                                  {&inputs_0, &inputs_1, &inputs_2});
  }

  // Check that the shape of the output matches what Caffe2 expects.
  std::vector<size_t> expectedDims = {10, 24};
  EXPECT_TRUE(output->dims().vec() == expectedDims);

  ctx.allocate(mod.getPlaceholders());
  auto res = ctx.get(output);
  EE.compile(CompilationMode::Infer, F);

  EE.run(ctx);
  // High level check on the content of the graph.
  // We have 1 concat, and 1 save.
  EXPECT_EQ(F->getNodes().size(), 2);
  // With have three inputs and one outputs.
  EXPECT_EQ(mod.getPlaceholders().size(), 4);

  auto result = res->getHandle();

  // Check that the graph has the expected shape,
  // starting from the output.
  auto *saveNode = getSaveNodeFromDest(output);
  auto *concat = llvm::dyn_cast<ConcatNode>(saveNode->getInput());
  ASSERT_TRUE(concat);
  // We will check that the inputs are correct within
  // the next loop.

  // Check that the output matches the concatenation of
  // all the inputs.
  Tensor *inputs[] = {&inputs_0, &inputs_1, &inputs_2};
  size_t columnsChecked = 0;
  for (size_t i = 0; i < 3; ++i) {
    const auto inputsHandle = inputs[i]->getHandle();
    ASSERT_TRUE(llvm::isa<Placeholder>(concat->getInputs()[i]));

    size_t currentColumnWidth = inputs[i]->dims()[1];
    for (size_t row = 0; row < 10; ++row) {
      for (size_t column = 0; column < currentColumnWidth; ++column) {
        EXPECT_FLOAT_EQ(result.at({row, columnsChecked + column}),
                        inputsHandle.at({row, column}));
      }
    }
    columnsChecked += currentColumnWidth;
  }
}

/// Test loading a batched matmul with transpose on RHS.
TEST(caffe2, batchedMatmulRHS) {
  ExecutionEngine EE{BackendKind::Interpreter};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  std::string NetDescFilename(
      "tests/models/caffe2Models/matmul_trans_RHS_predict_net.pbtxt");
  std::string NetWeightFilename(
      "tests/models/caffe2Models/empty_init_net.pbtxt");

  Placeholder *output;
  Tensor inputs_0(ElemKind::FloatTy, {3, 10, 7});
  Tensor inputs_1(ElemKind::FloatTy, {10, 7});
  inputs_0.getHandle().randomize(-3.0, 3.0, mod.getPRNG());
  inputs_1.getHandle().randomize(-3.0, 3.0, mod.getPRNG());
  // Destroy the loader after the graph is loaded since the following execution
  // will not depend on anyting from the loader.
  {
    Caffe2ModelLoader caffe2LD(NetDescFilename, NetWeightFilename,
                               {"inputs_0", "inputs_1"},
                               {&inputs_0.getType(), &inputs_1.getType()}, *F);
    output = EXIT_ON_ERR(caffe2LD.getSingleOutput());
  }

  // Check that the shape of the output matches what Caffe2 expects.
  std::vector<size_t> expectedDims = {3, 10, 10};
  EXPECT_TRUE(output->dims().vec() == expectedDims);
  // High level check on the content of the graph.
  // We have 1 transpose, 1 matmul, 1 save, and 2 reshapes.
  EXPECT_EQ(F->getNodes().size(), 5);
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
  auto *matmul =
      llvm::dyn_cast<MatMulNode>(reshapeResult->getInput().getNode());
  ASSERT_TRUE(matmul);
  const size_t matmulDims[] = {30, 10};
  EXPECT_EQ(matmul->dims(0), llvm::makeArrayRef(matmulDims));
  auto *lhs = llvm::dyn_cast<ReshapeNode>(matmul->getLHS().getNode());
  ASSERT_TRUE(lhs);
  auto *lhsInput = lhs->getInput().getNode();
  ASSERT_TRUE(llvm::isa<Placeholder>(lhsInput));
  auto *transpose = llvm::dyn_cast<TransposeNode>(matmul->getRHS().getNode());
  ASSERT_TRUE(transpose);
  ASSERT_TRUE(llvm::isa<Placeholder>(transpose->getInput().getNode()));
  // Check that the last two dimensions are swapped.
  const unsigned_t shuffle[] = {1, 0};
  EXPECT_EQ(transpose->getShuffle(), llvm::makeArrayRef(shuffle));
  // We don't actually check that the output is correct, because this
  // should be covered in the OperatorTest for MatMul already.
}

/// Test loading a parallel batched matmul.
TEST(caffe2, parallelBatchedMatmulRHS) {
  ExecutionEngine EE{BackendKind::Interpreter};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  std::string NetDescFilename(
      "tests/models/caffe2Models/parallel_matmul_predict_net.pbtxt");
  std::string NetWeightFilename(
      "tests/models/caffe2Models/empty_init_net.pbtxt");

  Placeholder *output;
  Tensor inputs_0(ElemKind::FloatTy, {3, 10, 7});
  Tensor inputs_1(ElemKind::FloatTy, {3, 7, 10});
  inputs_0.getHandle().randomize(-3.0, 3.0, mod.getPRNG());
  inputs_1.getHandle().randomize(-3.0, 3.0, mod.getPRNG());
  // Destroy the loader after the graph is loaded since the following execution
  // will not depend on anyting from the loader.
  {
    Caffe2ModelLoader caffe2LD(NetDescFilename, NetWeightFilename,
                               {"inputs_0", "inputs_1"},
                               {&inputs_0.getType(), &inputs_1.getType()}, *F);
    output = EXIT_ON_ERR(caffe2LD.getSingleOutput());
  }

  // Check that the shape of the output matches what Caffe2 expects.
  std::vector<size_t> expectedDims = {3, 10, 10};
  EXPECT_TRUE(output->dims().vec() == expectedDims);
  // High level check on the content of the graph.
  // We have 6 slices, 3 matmuls, 1 concat, 7 reshapes, 1 save.
  EXPECT_EQ(F->getNodes().size(), 18);
  // With have 2 inputs and one outputs.
  EXPECT_EQ(mod.getPlaceholders().size(), 3);
  // Check that the graph has the expected shape,
  // starting from the output.
  // Parallel Batched matmul is lowered to a sequence of slices, reshapes and
  // regular matmuls.
  auto *saveNode = getSaveNodeFromDest(output);
  auto *reshapeResult =
      llvm::dyn_cast<ReshapeNode>(saveNode->getInput().getNode());
  ASSERT_TRUE(reshapeResult);
  auto *concat =
      llvm::dyn_cast<ConcatNode>(reshapeResult->getInput().getNode());
  ASSERT_TRUE(concat);
  for (size_t i = 0; i < 3; i++) {
    auto *matmul = llvm::dyn_cast<MatMulNode>(concat->getNthInput(i).getNode());
    ASSERT_TRUE(matmul);
    const size_t matmulDims[] = {10, 10};
    EXPECT_EQ(matmul->dims(0), llvm::makeArrayRef(matmulDims));

    const size_t sliceStart[] = {i, 0, 0};
    // LHS
    auto *lhsReshape = llvm::dyn_cast<ReshapeNode>(matmul->getLHS().getNode());
    ASSERT_TRUE(lhsReshape);
    const size_t lhsReshapeDims[] = {10, 7};
    EXPECT_EQ(lhsReshape->getDims(), llvm::makeArrayRef(lhsReshapeDims));
    auto *lhsSlice =
        llvm::dyn_cast<SliceNode>(lhsReshape->getInput().getNode());
    ASSERT_TRUE(lhsSlice);
    EXPECT_EQ(lhsSlice->getStart(), llvm::makeArrayRef(sliceStart));
    auto *lhsInput =
        llvm::dyn_cast<Placeholder>(lhsSlice->getInput().getNode());
    ASSERT_TRUE(lhsInput);
    // RHS
    auto *rhsReshape = llvm::dyn_cast<ReshapeNode>(matmul->getRHS().getNode());
    ASSERT_TRUE(rhsReshape);
    const size_t rhsReshapeDims[] = {7, 10};
    EXPECT_EQ(rhsReshape->getDims(), llvm::makeArrayRef(rhsReshapeDims));
    auto *rhsSlice =
        llvm::dyn_cast<SliceNode>(rhsReshape->getInput().getNode());
    ASSERT_TRUE(rhsSlice);
    EXPECT_EQ(rhsSlice->getStart(), llvm::makeArrayRef(sliceStart));
    auto *rhsInput =
        llvm::dyn_cast<Placeholder>(rhsSlice->getInput().getNode());
    ASSERT_TRUE(rhsInput);
  }
  // We don't actually check that the output is correct, because this
  // should be covered in the OperatorTest for MatMul already.
}

/// Test loading a FC node : I * transpose(W) + B.
TEST(caffe2, FC) {
  ExecutionEngine EE{BackendKind::Interpreter};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename("tests/models/caffe2Models/fc_predict_net.pbtxt");
  std::string NetWeightFilename("tests/models/caffe2Models/fc_init_net.pbtxt");

  Placeholder *output;
  Context ctx;
  // Destroy the loader after the graph is loaded since the following execution
  // will not depend on anyting from the loader.
  {
    Tensor inputs(ElemKind::FloatTy, {2, 3});
    inputs.getHandle() = {1, 2, 3, 4, 5, 6};

    // Weights and bias are read from NetWeightFilename. And the values are:
    // weights : {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    // bias : {0.1f, 0.2f, 0.3f, 0.4f};
    Caffe2ModelLoader caffe2LD(NetDescFilename, NetWeightFilename, {"inputs"},
                               {&inputs.getType()}, *F);
    output = EXIT_ON_ERR(caffe2LD.getSingleOutput());
    ctx.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(ctx, &mod, {"inputs"}, {&inputs});
  }

  EE.compile(CompilationMode::Infer, F);
  EE.run(ctx);

  auto result = ctx.get(output)->getHandle();
  std::vector<size_t> expectedDims = {2, 4};
  std::vector<float> expectedValues = {14.1f, 32.2f, 50.3f,  68.4f,
                                       32.1f, 77.2f, 122.3f, 167.4f};
  EXPECT_TRUE(result.dims().vec() == expectedDims);
  for (size_t i = 0; i < 2 * 4; i++)
    EXPECT_FLOAT_EQ(result.raw(i), expectedValues[i]);
}

/// Test loading a FC node : I * transpose(W) + B, where I is need to be
/// flatten.
TEST(caffe2, FCWithFlatten) {
  ExecutionEngine EE{BackendKind::Interpreter};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename("tests/models/caffe2Models/fc_predict_net.pbtxt");
  std::string NetWeightFilename("tests/models/caffe2Models/fc_init_net.pbtxt");

  Placeholder *output;
  Context ctx;

  {
    Tensor inputs(ElemKind::FloatTy, {2, 1, 3});
    inputs.getHandle() = {1, 2, 3, 4, 5, 6};

    // Weights and bias are read from NetWeightFilename. And the values are:
    // weights : {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    // bias : {0.1f, 0.2f, 0.3f, 0.4f};
    Caffe2ModelLoader caffe2LD(NetDescFilename, NetWeightFilename, {"inputs"},
                               {&inputs.getType()}, *F);
    output = EXIT_ON_ERR(caffe2LD.getSingleOutput());
    ctx.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(ctx, &mod, {"inputs"}, {&inputs});
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

  EE.compile(CompilationMode::Infer, F);
  EE.run(ctx);
  auto result = ctx.get(output)->getHandle();
  std::vector<size_t> expectedDims = {2, 4};
  std::vector<float> expectedValues = {14.1f, 32.2f, 50.3f,  68.4f,
                                       32.1f, 77.2f, 122.3f, 167.4f};
  result = ctx.get(output)->getHandle();
  EXPECT_TRUE(result.dims().vec() == expectedDims);
  for (size_t i = 0; i < 2 * 4; i++)
    EXPECT_FLOAT_EQ(result.raw(i), expectedValues[i]);
}

/// Test loading a FCTransposed node: I * W + B
TEST(caffe2, FCTransposed) {
  ExecutionEngine EE{BackendKind::Interpreter};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename(
      "tests/models/caffe2Models/fcTransposed_predict_net.pbtxt");
  std::string NetWeightFilename(
      "tests/models/caffe2Models/fcTransposed_init_net.pbtxt");

  Placeholder *output;
  Context ctx;

  // Destroy the loader after the graph is loaded since the following execution
  // will not depend on anyting from the loader.
  {
    Tensor inputs(ElemKind::FloatTy, {2, 3});
    inputs.getHandle() = {1, 2, 3, 4, 5, 6};

    // Weights and bias are read from NetWeightFilename. And the values are:
    // weights : {1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12};
    // bias : {0.1f, 0.2f, 0.3f, 0.4f};
    Caffe2ModelLoader caffe2LD(NetDescFilename, NetWeightFilename, {"inputs"},
                               {&inputs.getType()}, *F);
    output = EXIT_ON_ERR(caffe2LD.getSingleOutput());
    ctx.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(ctx, &mod, {"inputs"}, {&inputs});
  }

  // High level check on the content of the graph. We have 1 FC and 1 save,
  EXPECT_EQ(F->getNodes().size(), 2);
  auto *saveNode = getSaveNodeFromDest(output);
  auto *fcNode =
      llvm::dyn_cast<FullyConnectedNode>(saveNode->getInput().getNode());
  ASSERT_TRUE(fcNode);

  EE.compile(CompilationMode::Infer, F);
  EE.run(ctx);

  auto result = ctx.get(output)->getHandle();
  std::vector<size_t> expectedDims = {2, 4};
  std::vector<float> expectedValues = {14.1f, 32.2f, 50.3f,  68.4f,
                                       32.1f, 77.2f, 122.3f, 167.4f};
  EXPECT_TRUE(result.dims().vec() == expectedDims);
  for (size_t i = 0; i < 2 * 4; i++)
    EXPECT_FLOAT_EQ(result.raw(i), expectedValues[i]);
}

/// Test loading a FCTransposed node: I * W + B, where I is need to be flatten.
TEST(caffe2, FCTransposedWithFlatten) {
  ExecutionEngine EE{BackendKind::Interpreter};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename(
      "tests/models/caffe2Models/fcTransposed_predict_net.pbtxt");
  std::string NetWeightFilename(
      "tests/models/caffe2Models/fcTransposed_init_net.pbtxt");

  Placeholder *output;
  Context ctx;

  {
    Tensor inputs(ElemKind::FloatTy, {2, 1, 3});
    inputs.getHandle() = {1, 2, 3, 4, 5, 6};

    // Weights and bias are read from NetWeightFilename. And the values are:
    // weights : {1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12};
    // bias : {0.1f, 0.2f, 0.3f, 0.4f};
    Caffe2ModelLoader caffe2LD(NetDescFilename, NetWeightFilename, {"inputs"},
                               {&inputs.getType()}, *F);
    output = EXIT_ON_ERR(caffe2LD.getSingleOutput());
    ctx.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(ctx, &mod, {"inputs"}, {&inputs});
  }

  // High level check on the content of the graph. We have 1 reshape, 1 FC,
  // and 1 save.
  EXPECT_EQ(F->getNodes().size(), 3);
  auto *saveNode1 = getSaveNodeFromDest(output);
  auto *fcNode1 =
      llvm::dyn_cast<FullyConnectedNode>(saveNode1->getInput().getNode());
  ASSERT_TRUE(fcNode1);
  auto *reshape = llvm::dyn_cast<ReshapeNode>(fcNode1->getInput());
  ASSERT_TRUE(reshape);

  EE.compile(CompilationMode::Infer, F);
  EE.run(ctx);
  auto result = ctx.get(output)->getHandle();
  std::vector<size_t> expectedDims = {2, 4};
  std::vector<float> expectedValues = {14.1f, 32.2f, 50.3f,  68.4f,
                                       32.1f, 77.2f, 122.3f, 167.4f};
  result = ctx.get(output)->getHandle();
  EXPECT_TRUE(result.dims().vec() == expectedDims);
  for (size_t i = 0; i < 2 * 4; i++)
    EXPECT_FLOAT_EQ(result.raw(i), expectedValues[i]);
}

/// Test loading clip op from a Caffe2 model.
/// Test with arg min = 20.0 max = 60.0
TEST(caffe2, importClip) {
  ExecutionEngine EE{BackendKind::Interpreter};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename("tests/models/caffe2Models/clip_op_net.pbtxt");
  std::string NetWeightFilename(
      "tests/models/caffe2Models/empty_init_net.pbtxt");

  Context ctx;
  Placeholder *output;
  Tensor inputs_0(ElemKind::FloatTy, {5, 5});
  inputs_0.getHandle<>() = {45.0, 16.0, 59.0, 99.0, 48.0, 12.0, 44.0,
                            46.0, 82.0, 28.0, 1.0,  91.0, 18.0, 9.0,
                            71.0, 24.0, 37.0, 61.0, 12.0, 81.0, 36.0,
                            38.0, 30.0, 84.0, 40.0};
  // Destroy the loader after the graph is loaded since the following execution
  // will not depend on anyting from the loader.
  {
    Caffe2ModelLoader caffe2LD(NetDescFilename, NetWeightFilename, {"inputs_0"},
                               {&inputs_0.getType()}, *F);
    output = EXIT_ON_ERR(caffe2LD.getSingleOutput());
    ctx.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(ctx, &mod, {"inputs_0"}, {&inputs_0});
  }

  auto *res = ctx.get(output);
  EE.compile(CompilationMode::Infer, F);
  EE.run(ctx);

  auto result = res->getHandle();
  std::vector<size_t> expectedDims = {5, 5};
  std::vector<float> expectedValues = {45.0, 20.0, 59.0, 60.0, 48.0, 20.0, 44.0,
                                       46.0, 60.0, 28.0, 20.0, 60.0, 20.0, 20.0,
                                       60.0, 24.0, 37.0, 60.0, 20.0, 60.0, 36.0,
                                       38.0, 30.0, 60.0, 40.0};
  EXPECT_TRUE(result.dims().vec() == expectedDims);
  for (size_t i = 0; i < 5 * 5; i++) {
    EXPECT_FLOAT_EQ(result.raw(i), expectedValues[i]);
  }
}

/// Test loading clip op from a Caffe2 model with default arg values:
/// min = std::numeric_limits<float>::lowest()
/// max = std::numeric_limits<float>::max()
TEST(caffe2, importClipDefault) {
  ExecutionEngine EE{BackendKind::Interpreter};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename(
      "tests/models/caffe2Models/clip_op_default_net.pbtxt");
  std::string NetWeightFilename(
      "tests/models/caffe2Models/empty_init_net.pbtxt");

  Context ctx;
  Placeholder *output;
  Tensor inputs_0(ElemKind::FloatTy, {5, 5});
  inputs_0.getHandle<>() = {45.0, 16.0, 59.0, 99.0, 48.0, 12.0, 44.0,
                            46.0, 82.0, 28.0, 1.0,  91.0, 18.0, 9.0,
                            71.0, 24.0, 37.0, 61.0, 12.0, 81.0, 36.0,
                            38.0, 30.0, 84.0, 40.0};

  // Destroy the loader after the graph is loaded since the following execution
  // will not depend on anyting from the loader.
  {
    Caffe2ModelLoader caffe2LD(NetDescFilename, NetWeightFilename, {"inputs_0"},
                               {&inputs_0.getType()}, *F);
    output = EXIT_ON_ERR(caffe2LD.getSingleOutput());
    ctx.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(ctx, &mod, {"inputs_0"}, {&inputs_0});
  }

  auto *res = ctx.get(output);
  EE.compile(CompilationMode::Infer, F);
  EE.run(ctx);

  auto result = res->getHandle();
  std::vector<size_t> expectedDims = {5, 5};
  EXPECT_TRUE(result.dims().vec() == expectedDims);
  for (size_t i = 0; i < 5 * 5; i++) {
    EXPECT_FLOAT_EQ(result.raw(i), inputs_0.getHandle().raw(i));
  }
}

/// Test loading a ReplaceNaN operator.
TEST(caffe2, replaceNaN) {
  ExecutionEngine EE{BackendKind::Interpreter};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename(
      "tests/models/caffe2Models/replace_nan_predict_net.pbtxt");
  std::string NetWeightFilename(
      "tests/models/caffe2Models/empty_init_net.pbtxt");

  Context ctx;
  Placeholder *output;
  Tensor input(ElemKind::FloatTy, {10, 10});
  auto inputHandle = input.getHandle();

  // Fill input by alternating between NAN and random values.
  inputHandle.randomize(-3.0, 3.0, mod.getPRNG());
  for (size_t i = 0; i < inputHandle.size(); ++i) {
    if (i & 0x1) {
      inputHandle.raw(i) = NAN;
    }
  }

  // Destroy the loader after the graph is loaded since the following execution
  // will not depend on anyting from the loader.
  {
    Caffe2ModelLoader caffe2LD(NetDescFilename, NetWeightFilename, {"input"},
                               {&input.getType()}, *F);
    output = EXIT_ON_ERR(caffe2LD.getSingleOutput());
    ctx.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(ctx, &mod, {"input"}, {&input});
  }

  // Check that the shape of the output matches the input.
  std::vector<size_t> expectedDims = {10, 10};
  EXPECT_TRUE(output->dims().vec() == expectedDims);

  // Compile and run the model.
  auto *res = ctx.get(output);
  EE.compile(CompilationMode::Infer, F);
  EE.run(ctx);

  auto result = res->getHandle();

  // High level checks on the content of the graph.
  // We have 1 IsNaN, 1 Splat, 1 Select and 1 Output.
  EXPECT_EQ(F->getNodes().size(), 4);
  // With have one input and one output.
  EXPECT_EQ(mod.getPlaceholders().size(), 2);

  // Check that the output tensor is the same as the input tensor except for
  // NaNs, which should have been replaced with 1 (the value specified in
  // replace_nan_predict_net.pbtxt).
  for (size_t i = 0; i < result.size(); ++i) {
    if (std::isnan(inputHandle.raw(i)))
      EXPECT_EQ(result.raw(i), 1);
    else {
      EXPECT_EQ(result.raw(i), inputHandle.raw(i));
    }
  }
}

/// Test loading a DotProduct operator with 1D inputs.
TEST(caffe2, dotProduct1D) {
  ExecutionEngine EE{BackendKind::Interpreter};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename(
      "tests/models/caffe2Models/dot_product_predict_net.pbtxt");
  std::string NetWeightFilename(
      "tests/models/caffe2Models/empty_init_net.pbtxt");

  Placeholder *output;
  Context ctx;

  // Input tensors.
  const size_t kDataSize = 10;
  Tensor X(ElemKind::FloatTy, {kDataSize});
  Tensor Y(ElemKind::FloatTy, {kDataSize});
  auto XH = X.getHandle();
  auto YH = Y.getHandle();

  // Fill inputs with random values.
  XH.randomize(-3.0, 3.0, mod.getPRNG());
  YH.randomize(-3.0, 3.0, mod.getPRNG());

  // Compute expected output.
  Tensor O(ElemKind::FloatTy, {kDataSize});
  auto OH = O.getHandle();

  for (size_t i = 0; i < kDataSize; ++i) {
    OH.at({i}) = XH.at({i}) * YH.at({i});
  }

  // Destroy the loader after the graph is loaded since the following execution
  // will not depend on anyting from the loader.
  {
    Caffe2ModelLoader caffe2LD(NetDescFilename, NetWeightFilename, {"X", "Y"},
                               {&X.getType(), &Y.getType()}, *F);
    output = EXIT_ON_ERR(caffe2LD.getSingleOutput());
    ctx.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(ctx, &mod, {"X", "Y"}, {&X, &Y});
  }

  // Check that the shape of the output matches that of the expected output.
  EXPECT_TRUE(output->dims().vec() == OH.dims().vec());

  // High level checks on the content of the graph.
  // We have 1 Mul and 1 Output.
  EXPECT_EQ(F->getNodes().size(), 2);

  // Check that the graph has the expected shape (Mul -> Save),
  // starting from the output.
  auto *saveNode = getSaveNodeFromDest(output);
  auto *MN = llvm::dyn_cast<MulNode>(saveNode->getInput());
  ASSERT_TRUE(MN);

  // With have two inputs and one output.
  EXPECT_EQ(mod.getPlaceholders().size(), 3);

  // Compile and run the model.
  ctx.allocate(mod.getPlaceholders());
  auto *res = ctx.get(output);
  EE.compile(CompilationMode::Infer, F);
  EE.run(ctx);

  auto result = res->getHandle();
  // Check that the output tensor is the same as the expected output.
  for (size_t i = 0; i < result.size(); ++i) {
    EXPECT_NEAR(result.raw(i), OH.raw(i), 0.00001);
  }
}

// Test loading a DotProduct operator with 2D inputs.
TEST(caffe2, dotProduct2D) {
  ExecutionEngine EE{BackendKind::Interpreter};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename(
      "tests/models/caffe2Models/dot_product_predict_net.pbtxt");
  std::string NetWeightFilename(
      "tests/models/caffe2Models/empty_init_net.pbtxt");

  Context ctx;
  Placeholder *output;

  // Input tensors.
  const size_t kRows = 10;
  const size_t kCols = 20;
  Tensor X(ElemKind::FloatTy, {kRows, kCols});
  Tensor Y(ElemKind::FloatTy, {kRows, kCols});
  auto XH = X.getHandle();
  auto YH = Y.getHandle();

  // Fill inputs with random values.
  XH.randomize(-3.0, 3.0, mod.getPRNG());
  YH.randomize(-3.0, 3.0, mod.getPRNG());

  // Compute expected output.
  Tensor O(ElemKind::FloatTy, {kRows});
  auto OH = O.getHandle();

  for (size_t i = 0; i < kRows; ++i) {
    auto dotProduct = 0.0f;

    // Compute dot product of the i-th row of X and Y.
    for (size_t j = 0; j < kCols; ++j) {
      dotProduct += (XH.at({i, j}) * YH.at({i, j}));
    }

    OH.at({i}) = dotProduct;
  }

  // Destroy the loader after the graph is loaded since the following execution
  // will not depend on anyting from the loader.
  {
    Caffe2ModelLoader caffe2LD(NetDescFilename, NetWeightFilename, {"X", "Y"},
                               {&X.getType(), &Y.getType()}, *F);
    output = EXIT_ON_ERR(caffe2LD.getSingleOutput());
    ctx.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(ctx, &mod, {"X", "Y"}, {&X, &Y});
  }

  // Check that the shape of the output matches that of the expected output.
  EXPECT_TRUE(output->dims().vec() == OH.dims().vec());

  // High level checks on the content of the graph.
  // We have 1 Mul, 1 BatchedReduceAdd and 1 Output.
  EXPECT_EQ(F->getNodes().size(), 3);

  // Check that the graph has the expected shape
  /// (Mul -> BatchedReduceAdd -> Save), starting from the output.
  auto *saveNode = getSaveNodeFromDest(output);
  auto *BRA = llvm::dyn_cast<BatchedReduceAddNode>(saveNode->getInput());
  ASSERT_TRUE(BRA);
  ASSERT_EQ(BRA->getNumInputs(), 1);

  auto *MN = llvm::dyn_cast<MulNode>(BRA->getBatch());
  ASSERT_TRUE(MN);

  // With have two inputs and one output.
  EXPECT_EQ(mod.getPlaceholders().size(), 3);

  // Compile and run the model.
  auto *res = ctx.get(output);
  EE.compile(CompilationMode::Infer, F);
  EE.run(ctx);

  auto result = res->getHandle();

  // Check that the output tensor is the same as the expected output.
  for (size_t i = 0; i < result.size(); ++i) {
    EXPECT_NEAR(result.raw(i), OH.raw(i), 0.00001);
  }
}

// Test loading a BatchBoxCox operator.
TEST(caffe2, batchBoxCox) {
  ExecutionEngine EE{BackendKind::Interpreter};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename(
      "tests/models/caffe2Models/batch_box_cox_predict_net.pbtxt");
  std::string NetWeightFilename(
      "tests/models/caffe2Models/empty_init_net.pbtxt");

  Context ctx;
  Placeholder *output;

  // Input tensors.
  const size_t kRows = 10;
  const size_t kCols = 5;
  Tensor data(ElemKind::FloatTy, {kRows, kCols});
  Tensor lambda1(ElemKind::FloatTy, {kCols});
  Tensor lambda2(ElemKind::FloatTy, {kCols});
  auto dataH = data.getHandle();
  auto lambda1H = lambda1.getHandle();
  auto lambda2H = lambda2.getHandle();

  // Fill inputs with random values.
  dataH.randomize(0.0, 5.0, mod.getPRNG());
  lambda1H.randomize(1.0, 2.0, mod.getPRNG());
  lambda2H.randomize(1.0, 2.0, mod.getPRNG());

  // Zero out every other element to lambda1 to test that case of the transform.
  for (size_t i = 0; i < kCols; i += 2) {
    lambda1H.at({i}) = 0;
  }

  // Compute expected output.
  Tensor O(ElemKind::FloatTy, {kRows, kCols});
  auto OH = O.getHandle();

  for (size_t i = 0; i < kRows; ++i) {
    for (size_t j = 0; j < kCols; ++j) {
      float d = dataH.at({i, j});
      float l1 = lambda1H.at({j});
      float l2 = lambda2H.at({j});

      // Compute elementwise Box-Cox transform.
      float tmp = std::max(d + l2, 1e-6f);
      if (l1 == 0) {
        // Clip argument to log and pow at 1e-6 to avoid saturation.
        OH.at({i, j}) = std::log(tmp);
      } else {
        OH.at({i, j}) = (std::pow(tmp, l1) - 1) / l1;
      }
    }
  }

  // Destroy the loader after the graph is loaded since the following execution
  // will not depend on anyting from the loader.
  {
    Caffe2ModelLoader caffe2LD(
        NetDescFilename, NetWeightFilename, {"data", "lambda1", "lambda2"},
        {&data.getType(), &lambda1.getType(), &lambda2.getType()}, *F);
    output = EXIT_ON_ERR(caffe2LD.getSingleOutput());
    ctx.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(ctx, &mod, {"data", "lambda1", "lambda2"},
                                  {&data, &lambda1, &lambda2});
  }

  // Check that the shape of the output matches that of the expected output.
  EXPECT_TRUE(output->dims().vec() == OH.dims().vec());

  // High level checks on the content of the graph.
  // We should have 2 Broadcast (2 Reshape and 2 Tile), 2 Add, 4 Splat,
  // 1 Max, 1 Log, 1 Pow, 1 Sub, 1 Div, 1 CmpEQ, 1 Select and 1 Output =
  // 18 nodes in total.
  //
  EXPECT_EQ(F->getNodes().size(), 18);

  // Check that the graph has the expected shape:
  //
  //        (input) Broadcast          Broadcast -----
  //              \   |        ________/ /           |
  //              v   v       v         /            |
  //       Splat   Add  -> Pow  Splat  /             |
  //           \    |   |   |   /     /              |
  //            v   v   |   v  v     v               |
  //             Max ----   Sub    Add <-- Splat     |
  //              |          |    /                  |
  //              v          v   v                   |
  //             Log         Div                     |
  //                \       /         _______________|
  //                 v     v         v
  //                 Select  <--- CmpEQ <--- Splat
  //                   |
  //                   v
  //                 Output
  //
  // Search in a breadth-first fashion starting from the output.
  // Broadcast consists of (Reshape -> Tile), so cast to TileNode
  // when checking for Broadcast.

  // Output.
  auto *saveNode = getSaveNodeFromDest(output);
  ASSERT_TRUE(saveNode);

  // Select.
  auto *selectNode = llvm::dyn_cast<SelectNode>(saveNode->getInput());
  ASSERT_TRUE(selectNode);

  // CmpEQ, Log, Div.
  auto *CEQ = llvm::dyn_cast<CmpEQNode>(selectNode->getCond());
  ASSERT_TRUE(CEQ);
  auto *LN = llvm::dyn_cast<LogNode>(selectNode->getLHS());
  ASSERT_TRUE(LN);
  auto *DN = llvm::dyn_cast<DivNode>(selectNode->getRHS());
  ASSERT_TRUE(DN);

  // Splat, Broadcast, Max, Sub, Add.
  ASSERT_TRUE(llvm::dyn_cast<SplatNode>(CEQ->getRHS()));
  auto *BN1 = llvm::dyn_cast<TileNode>(CEQ->getLHS());
  ASSERT_TRUE(BN1);
  auto *MN = llvm::dyn_cast<MaxNode>(LN->getInput());
  ASSERT_TRUE(MN);
  auto *subNode = llvm::dyn_cast<SubNode>(DN->getLHS());
  ASSERT_TRUE(subNode);
  auto *AN1 = llvm::dyn_cast<AddNode>(DN->getRHS());
  ASSERT_TRUE(AN1);

  // Splat, Splat, Splat. Add, Pow, Broadcast.
  ASSERT_TRUE(llvm::dyn_cast<SplatNode>(MN->getRHS()));
  ASSERT_TRUE(llvm::dyn_cast<SplatNode>(subNode->getRHS()));
  ASSERT_TRUE(llvm::dyn_cast<SplatNode>(AN1->getRHS()));
  auto *AN2 = llvm::dyn_cast<AddNode>(MN->getLHS());
  ASSERT_TRUE(AN2);
  auto *PN = llvm::dyn_cast<PowNode>(subNode->getLHS());
  ASSERT_TRUE(PN);
  EXPECT_EQ(MN, llvm::dyn_cast<MaxNode>(PN->getLHS()));
  EXPECT_EQ(BN1, llvm::dyn_cast<TileNode>(AN1->getLHS()));

  // Broadcast, Broadcast.
  EXPECT_EQ(BN1, llvm::dyn_cast<TileNode>(PN->getRHS()));
  auto *BN2 = llvm::dyn_cast<TileNode>(AN2->getRHS());
  EXPECT_TRUE(BN2);

  // There are three inputs and one output.
  EXPECT_EQ(mod.getPlaceholders().size(), 4);

  // Compile and run the model.
  auto *res = ctx.get(output);
  EE.compile(CompilationMode::Infer, F);
  EE.run(ctx);

  auto result = res->getHandle();

  // Check that the output tensor is the same as the expected output.
  for (size_t i = 0; i < result.size(); ++i) {
    EXPECT_FLOAT_EQ(result.raw(i), OH.raw(i));
  }
}

// Test loading a EQ operator with 1D inputs.
TEST(caffe2, EQ1D) {
  ExecutionEngine EE{BackendKind::Interpreter};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename("tests/models/caffe2Models/eq_op_net.pbtxt");
  std::string NetWeightFilename(
      "tests/models/caffe2Models/empty_init_net.pbtxt");

  Placeholder *output;
  Context ctx;

  // Input tensors.
  const size_t kDataSize = 10;
  Tensor X(ElemKind::FloatTy, {kDataSize});
  Tensor Y(ElemKind::FloatTy, {kDataSize});

  // Destroy the loader after the graph is loaded
  {
    Caffe2ModelLoader caffe2LD(NetDescFilename, NetWeightFilename, {"X", "Y"},
                               {&X.getType(), &Y.getType()}, *F);
    output = EXIT_ON_ERR(caffe2LD.getSingleOutput());
  }

  // High level checks on the content of the graph.
  // We have 1 EQ and 1 Output.
  EXPECT_EQ(F->getNodes().size(), 2);

  // Check that the graph has the expected shape (EQ -> Save),
  // starting from the output.
  auto *saveNode = getSaveNodeFromDest(output);
  auto *EQN = llvm::dyn_cast<CmpEQNode>(saveNode->getInput());
  ASSERT_TRUE(EQN);

  // Graph has two inputs and one output.
  EXPECT_EQ(mod.getPlaceholders().size(), 3);
}

// Test loading a LengthsToRanges operator.
TEST(caffe2, LengthsToRanges) {
  ExecutionEngine EE{BackendKind::Interpreter};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename(
      "tests/models/caffe2Models/lengths_to_ranges.pbtxt");
  std::string NetWeightFilename(
      "tests/models/caffe2Models/lengths_to_ranges_init_net.pbtxt");

  Placeholder *output;

  // Destroy the loader after the graph is loaded
  {
    Caffe2ModelLoader caffe2LD(NetDescFilename, NetWeightFilename, {}, {}, *F);
    output = EXIT_ON_ERR(caffe2LD.getSingleOutput());
  }

  // High level checks on the content of the graph.
  // We have 1 LengthsToRanges and 1 Save.
  EXPECT_EQ(F->getNodes().size(), 2);

  // Check that the graph has the expected shape (LengthsToRanges -> Save),
  // starting from the output.
  auto *saveNode = getSaveNodeFromDest(output);
  auto *N = llvm::dyn_cast<LengthsToRangesNode>(saveNode->getInput());
  ASSERT_TRUE(N);

  // Graph has one output.
  EXPECT_EQ(mod.getPlaceholders().size(), 1);
}

// Test loading Logit operator from a Caffe2 model.
TEST(caffe2, Logit) {
  ExecutionEngine EE{BackendKind::Interpreter};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename("tests/models/caffe2Models/logit_op_net.pbtxt");
  std::string NetWeightFilename(
      "tests/models/caffe2Models/empty_init_net.pbtxt");

  Placeholder *output;

  // Input tensors.
  const std::size_t kDataSize = 10;
  Tensor X(ElemKind::FloatTy, {kDataSize});

  // Destroy the loader after the graph is loaded
  {
    Caffe2ModelLoader caffe2LD(NetDescFilename, NetWeightFilename, {"inputs_0"},
                               {&X.getType()}, *F);
    output = EXIT_ON_ERR(caffe2LD.getSingleOutput());
  }

  // Check that the shape of the output matches what Caffe2 expects.
  std::vector<size_t> expectedDims = {kDataSize};
  EXPECT_EQ(output->dims().vec(), expectedDims);

  // High level checks on the content of the graph.
  // We have 1 Clip (1 Splat, 1 Max, 1 Splat, 1 Min),
  // 1 Splat, 1 Sub, 1 Div, 1 Log, and 1 Output.
  EXPECT_EQ(F->getNodes().size(), 9);

  // Graph has one input and one output.
  EXPECT_EQ(mod.getPlaceholders().size(), 2);
}

// Test loading a SparseToDense operator.
TEST(caffe2, sparseToDense) {
  ExecutionEngine EE{BackendKind::Interpreter};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename(
      "tests/models/caffe2Models/sparse_to_dense.pbtxt");
  std::string NetWeightFilename(
      "tests/models/caffe2Models/empty_init_net.pbtxt");

  Placeholder *output;
  Context ctx;

  // Create inputs.
  constexpr size_t kNumIndices = 5;
  constexpr size_t kMaxIndex = 20;
  constexpr size_t kRows = 10;
  constexpr size_t kCols = 5;
  Tensor indices(ElemKind::Int64ITy, {kNumIndices});
  Tensor values(ElemKind::FloatTy, {kNumIndices, kRows, kCols});
  Tensor dataToInferDim(ElemKind::FloatTy, {kMaxIndex, kRows, kCols});

  // Destroy the loader after the graph is loaded since the following execution
  // will not depend on anyting from the loader.
  {
    Caffe2ModelLoader caffe2LD(
        NetDescFilename, NetWeightFilename,
        {"indices", "values", "dataToInferDim"},
        {&indices.getType(), &values.getType(), &dataToInferDim.getType()}, *F);
    output = EXIT_ON_ERR(caffe2LD.getSingleOutput());
    ctx.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(ctx, &mod, {"indices", "values"},
                                  {&indices, &values});
  }

  // Check that the shape of the output matches that of the expected output.
  EXPECT_TRUE(output->dims().vec() == dataToInferDim.dims().vec());

  // High level checks on the content of the graph.
  // We should have 1 SparseToDense and 1 Output node = 2 nodes in total.
  EXPECT_EQ(F->getNodes().size(), 2);

  // Check that the graph has the expected shape (SparseToDense -> Save),
  // starting from the output.
  auto *saveNode = getSaveNodeFromDest(output);
  auto *STDN = llvm::dyn_cast<SparseToDenseNode>(saveNode->getInput());
  ASSERT_TRUE(STDN);

  // Graph has three inputs and one output.
  EXPECT_EQ(mod.getPlaceholders().size(), 4);
}

/// Test loading NCHW2NHWC op.
TEST(caffe2, testNCHW2NHWC) {
  ExecutionEngine EE{BackendKind::Interpreter};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename(
      "tests/models/caffe2Models/NCHW2NHWC_predict_net.pbtxt");
  std::string NetWeightFilename(
      "tests/models/caffe2Models/empty_init_net.pbtxt");

  Placeholder *output;
  Context ctx;

  Tensor inputs(ElemKind::FloatTy, {1, 2, 3, 4});

  // Destroy the loader after the graph is loaded since the following execution
  // will not depend on anyting from the loader.
  {
    Caffe2ModelLoader caffe2LD(NetDescFilename, NetWeightFilename, {"inputs"},
                               {&inputs.getType()}, *F);
    output = EXIT_ON_ERR(caffe2LD.getSingleOutput());
    ctx.allocate(mod.getPlaceholders());
  }

  // Check output shape.
  auto res = ctx.get(output);
  std::vector<size_t> expectedDims = {1, 3, 4, 2};
  EXPECT_TRUE(res->getHandle<float>().dims().vec() == expectedDims);

  // High level check on the content of the graph. We have 1 transpose and 1
  // save.
  EXPECT_EQ(F->getNodes().size(), 2);
  auto *saveNode = getSaveNodeFromDest(output);
  auto *transNode =
      llvm::dyn_cast<TransposeNode>(saveNode->getInput().getNode());
  ASSERT_TRUE(transNode);

  // We have 2 placeholders:  1 input and 1 output.
  EXPECT_EQ(mod.getPlaceholders().size(), 2);
  // We have 0 constants.
  EXPECT_EQ(mod.getConstants().size(), 0);
}

// Test loading a LengthsSum operator.
TEST(caffe2, lengthsSum) {
  ExecutionEngine EE{BackendKind::Interpreter};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename("tests/models/caffe2Models/lengths_sum.pbtxt");
  std::string NetWeightFilename(
      "tests/models/caffe2Models/empty_init_net.pbtxt");

  Placeholder *output;
  Context ctx;

  // Create inputs.
  Tensor data(ElemKind::Int64ITy, {10, 2, 3});
  Tensor lengths(ElemKind::FloatTy, {5});

  // Destroy the loader after the graph is loaded since the following execution
  // will not depend on anyting from the loader.
  {
    Caffe2ModelLoader caffe2LD(NetDescFilename, NetWeightFilename,
                               {"data", "lengths"},
                               {&data.getType(), &lengths.getType()}, *F);
    output = EXIT_ON_ERR(caffe2LD.getSingleOutput());
  }

  // Check that the shape of the output matches that of the expected output.
  std::vector<size_t> expectedShape{5, 2, 3};
  EXPECT_TRUE(output->dims().vec() == expectedShape);

  // High level checks on the content of the graph.
  // We should have 1 LengthsSum and 1 Output node = 2 nodes in total.
  EXPECT_EQ(F->getNodes().size(), 2);

  // Check that the graph has the expected shape (LengthsSum -> Save),
  // starting from the output.
  auto *saveNode = getSaveNodeFromDest(output);
  auto *LSN = llvm::dyn_cast<LengthsSumNode>(saveNode->getInput());
  ASSERT_TRUE(LSN);

  // Graph has two inputs and one output.
  EXPECT_EQ(mod.getPlaceholders().size(), 3);
}

/// Verify that different fill types are loaded with the correct types.
TEST(caffe2, tensorFillsTest) {
  ExecutionEngine EE{BackendKind::Interpreter};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename(
      "tests/models/caffe2Models/empty_predict_net.pbtxt");
  std::string NetWeightFilename(
      "tests/models/caffe2Models/fill_test_init_net.pbtxt");

  Constant *tensorFillFloat, *tensorFillInt, *tensorIntFill, *tensorInt64Fill;

  // Destroy the loader after the graph is loaded since the following execution
  // will not depend on anything from the loader.
  {
    // Loaded protos must have at least one external output, so load an unused
    // output and type to satisfy it. It is named unused_output in
    // empty_predict_net.pbtxt.
    Type unusedTy = Type(ElemKind::FloatTy, {1});
    Caffe2ModelLoader caffe2LD(NetDescFilename, NetWeightFilename,
                               {"unused_output"}, {&unusedTy}, *F);
    tensorFillFloat = llvm::dyn_cast<Constant>(EXIT_ON_ERR(
        caffe2LD.getNodeValueOrCreateConstantByName("tensor_fill_float")));
    tensorFillInt = llvm::dyn_cast<Constant>(EXIT_ON_ERR(
        caffe2LD.getNodeValueOrCreateConstantByName("tensor_fill_int")));
    tensorIntFill = llvm::dyn_cast<Constant>(EXIT_ON_ERR(
        caffe2LD.getNodeValueOrCreateConstantByName("tensor_int_fill")));
    tensorInt64Fill = llvm::dyn_cast<Constant>(EXIT_ON_ERR(
        caffe2LD.getNodeValueOrCreateConstantByName("tensor_int64_fill")));
  }

  ASSERT_TRUE(tensorFillFloat);
  ASSERT_TRUE(tensorFillInt);
  ASSERT_TRUE(tensorIntFill);
  ASSERT_TRUE(tensorInt64Fill);

  // All fills in fill_test_init_net.pbtxt use shape {2, 2}.
  const std::vector<size_t> expectedDims = {2, 2};
  ASSERT_TRUE(tensorFillFloat->dims().equals(expectedDims));
  ASSERT_TRUE(tensorFillInt->dims().equals(expectedDims));
  ASSERT_TRUE(tensorIntFill->dims().equals(expectedDims));
  ASSERT_TRUE(tensorInt64Fill->dims().equals(expectedDims));

  auto tensorFillFloatH = tensorFillFloat->getPayload().getHandle<float>();
  auto tensorFillIntH = tensorFillInt->getPayload().getHandle<int64_t>();
  auto tensorIntFillH = tensorIntFill->getPayload().getHandle<int32_t>();
  auto tensorInt64FillH = tensorInt64Fill->getPayload().getHandle<int64_t>();

  // All fills in fill_test_init_net.pbtxt are set to 0 through 3.
  for (size_t i = 0, e = 4; i < e; i++) {
    EXPECT_FLOAT_EQ(tensorFillFloatH.raw(i), (float)i);
    EXPECT_EQ(tensorFillIntH.raw(i), (int64_t)i);
    EXPECT_EQ(tensorIntFillH.raw(i), (int32_t)i);
    EXPECT_EQ(tensorInt64FillH.raw(i), (int64_t)i);
  }
}
