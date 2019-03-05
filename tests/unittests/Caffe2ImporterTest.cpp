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

#ifndef GLOW_DATA_PATH
#define GLOW_DATA_PATH
#endif

using namespace glow;

/// Test loading conv op from a Caffe2 model.
/// The input is N*C*H*W (1*1*3*3), the kernel is 2,
/// stride is 1, pad is 1, group is 1.
TEST(caffe2, importConv) {
  ExecutionEngine EE{BackendKind::Interpreter};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename(GLOW_DATA_PATH
                              "tests/models/caffe2Models/predict_net.pbtxt");
  std::string NetWeightFilename(GLOW_DATA_PATH
                                "tests/models/caffe2Models/init_net.pbtxt");

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
      GLOW_DATA_PATH "tests/models/caffe2Models/conv_nhwc_predict_net.pbtxt");
  std::string NetWeightFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/conv_nhwc_init_net.pbtxt");

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
      GLOW_DATA_PATH
      "tests/models/caffe2Models/maxpool_nhwc_predict_net.pbtxt");
  std::string NetWeightFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/empty_init_net.pbtxt");

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
      GLOW_DATA_PATH "tests/models/caffe2Models/maxpool_predict_net.pbtxt");
  std::string NetWeightFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/empty_init_net.pbtxt");

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
      GLOW_DATA_PATH
      "tests/models/caffe2Models/avgpool_nhwc_predict_net.pbtxt");
  std::string NetWeightFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/empty_init_net.pbtxt");

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
      GLOW_DATA_PATH "tests/models/caffe2Models/avgpool_predict_net.pbtxt");
  std::string NetWeightFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/empty_init_net.pbtxt");

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
      GLOW_DATA_PATH
      "tests/models/caffe2Models/concat_add_axis_predict_net.pbtxt");
  std::string NetWeightFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/empty_init_net.pbtxt");

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
      GLOW_DATA_PATH "tests/models/caffe2Models/concat_predict_net.pbtxt");
  std::string NetWeightFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/empty_init_net.pbtxt");

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
      GLOW_DATA_PATH
      "tests/models/caffe2Models/matmul_trans_RHS_predict_net.pbtxt");
  std::string NetWeightFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/empty_init_net.pbtxt");

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
      GLOW_DATA_PATH
      "tests/models/caffe2Models/parallel_matmul_predict_net.pbtxt");
  std::string NetWeightFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/empty_init_net.pbtxt");

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

  std::string NetDescFilename(GLOW_DATA_PATH
                              "tests/models/caffe2Models/fc_predict_net.pbtxt");
  std::string NetWeightFilename(GLOW_DATA_PATH
                                "tests/models/caffe2Models/fc_init_net.pbtxt");

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

  // High level check on the content of the graph. We have 1 FC node and 1 save.
  EXPECT_EQ(F->getNodes().size(), 2);
  auto *saveNode = getSaveNodeFromDest(output);
  auto *fcNode =
      llvm::dyn_cast<FullyConnectedNode>(saveNode->getInput().getNode());
  EXPECT_TRUE(fcNode);

  // Check the numerical values of the weights and biases.
  {
    const Constant *constant = mod.getConstantByName("weights");
    ASSERT_TRUE(constant);
    const Tensor &weights = constant->getPayload();
    const std::vector<size_t> expectedDimensions = {3, 4};
    const std::vector<float> expectedValues = {1.0f, 4.0f, 7.0f, 10.0f, //
                                               2.0f, 5.0f, 8.0f, 11.0f, //
                                               3.0f, 6.0f, 9.0f, 12.0f};
    EXPECT_EQ(expectedDimensions, weights.dims().vec());
    ASSERT_EQ(expectedValues.size(), weights.size());
    const auto elements = weights.getHandle();
    for (size_t i = 0; i < expectedValues.size(); ++i) {
      EXPECT_FLOAT_EQ(expectedValues.at(i), elements.raw(i))
          << "Where i = " << i;
    }
  }
  {
    const Constant *constant = mod.getConstantByName("biases");
    ASSERT_TRUE(constant);
    const Tensor &bias = constant->getPayload();
    const std::vector<size_t> expectedDimensions = {4};
    const std::vector<float> expectedValues = {0.1f, 0.2f, 0.3f, 0.4f};
    EXPECT_EQ(expectedDimensions, bias.dims().vec());
    ASSERT_EQ(expectedValues.size(), bias.size());
    const auto elements = bias.getHandle();
    for (size_t i = 0; i < expectedValues.size(); ++i) {
      EXPECT_FLOAT_EQ(expectedValues.at(i), elements.raw(i))
          << "Where i = " << i;
    }
  }

  // We don't actually check that the output is correct, because this is
  // already covered in the Operator.FC/* tests.
}

/// Test loading a FC node : I * transpose(W) + B, where I is need to be
/// flatten.
TEST(caffe2, FCWithFlatten) {
  ExecutionEngine EE{BackendKind::Interpreter};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename(GLOW_DATA_PATH
                              "tests/models/caffe2Models/fc_predict_net.pbtxt");
  std::string NetWeightFilename(GLOW_DATA_PATH
                                "tests/models/caffe2Models/fc_init_net.pbtxt");

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

  // Check the numerical values of the weights and biases.
  {
    const Constant *constant = mod.getConstantByName("weights");
    ASSERT_TRUE(constant);
    const Tensor &weights = constant->getPayload();
    const std::vector<size_t> expectedDimensions = {3, 4};
    const std::vector<float> expectedValues = {1.0f, 4.0f, 7.0f, 10.0f, //
                                               2.0f, 5.0f, 8.0f, 11.0f, //
                                               3.0f, 6.0f, 9.0f, 12.0f};
    EXPECT_EQ(expectedDimensions, weights.dims().vec());
    ASSERT_EQ(expectedValues.size(), weights.size());
    const auto elements = weights.getHandle();
    for (size_t i = 0; i < expectedValues.size(); ++i) {
      EXPECT_FLOAT_EQ(expectedValues.at(i), elements.raw(i))
          << "Where i = " << i;
    }
  }
  {
    const Constant *constant = mod.getConstantByName("biases");
    ASSERT_TRUE(constant);
    const Tensor &bias = constant->getPayload();
    const std::vector<size_t> expectedDimensions = {4};
    const std::vector<float> expectedValues = {0.1f, 0.2f, 0.3f, 0.4f};
    EXPECT_EQ(expectedDimensions, bias.dims().vec());
    ASSERT_EQ(expectedValues.size(), bias.size());
    const auto elements = bias.getHandle();
    for (size_t i = 0; i < expectedValues.size(); ++i) {
      EXPECT_FLOAT_EQ(expectedValues.at(i), elements.raw(i))
          << "Where i = " << i;
    }
  }

  // We don't actually check that the output is correct, because this is
  // already covered in the Operator.FCWithFlatten/* tests.
}

/// Test loading a FCTransposed node: I * W + B
TEST(caffe2, FCTransposed) {
  ExecutionEngine EE{BackendKind::Interpreter};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename(
      GLOW_DATA_PATH
      "tests/models/caffe2Models/fcTransposed_predict_net.pbtxt");
  std::string NetWeightFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/fcTransposed_init_net.pbtxt");

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

  // Check the numerical values of the weights and biases.
  {
    const Constant *constant = mod.getConstantByName("weights");
    ASSERT_TRUE(constant);
    const Tensor &weights = constant->getPayload();
    const std::vector<size_t> expectedDimensions = {3, 4};
    const std::vector<float> expectedValues = {1.0f, 4.0f, 7.0f, 10.0f, //
                                               2.0f, 5.0f, 8.0f, 11.0f, //
                                               3.0f, 6.0f, 9.0f, 12.0f};
    EXPECT_EQ(expectedDimensions, weights.dims().vec());
    ASSERT_EQ(expectedValues.size(), weights.size());
    const auto elements = weights.getHandle();
    for (size_t i = 0; i < expectedValues.size(); ++i) {
      EXPECT_FLOAT_EQ(expectedValues.at(i), elements.raw(i))
          << "Where i = " << i;
    }
  }
  {
    const Constant *constant = mod.getConstantByName("biases");
    ASSERT_TRUE(constant);
    const Tensor &bias = constant->getPayload();
    const std::vector<size_t> expectedDimensions = {4};
    const std::vector<float> expectedValues = {0.1f, 0.2f, 0.3f, 0.4f};
    EXPECT_EQ(expectedDimensions, bias.dims().vec());
    ASSERT_EQ(expectedValues.size(), bias.size());
    const auto elements = bias.getHandle();
    for (size_t i = 0; i < expectedValues.size(); ++i) {
      EXPECT_FLOAT_EQ(expectedValues.at(i), elements.raw(i))
          << "Where i = " << i;
    }
  }

  // We don't actually check that the output is correct, because this is
  // already covered in the Operator.FCWithFlatten/* tests.
}

/// Test loading a FCTransposed node: I * W + B, where I is need to be flatten.
TEST(caffe2, FCTransposedWithFlatten) {
  ExecutionEngine EE{BackendKind::Interpreter};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename(
      GLOW_DATA_PATH
      "tests/models/caffe2Models/fcTransposed_predict_net.pbtxt");
  std::string NetWeightFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/fcTransposed_init_net.pbtxt");

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

  // Check the numerical values of the weights and biases.
  {
    const Constant *constant = mod.getConstantByName("weights");
    ASSERT_TRUE(constant);
    const Tensor &weights = constant->getPayload();
    const std::vector<size_t> expectedDimensions = {3, 4};
    const std::vector<float> expectedValues = {1.0f, 4.0f, 7.0f, 10.0f, //
                                               2.0f, 5.0f, 8.0f, 11.0f, //
                                               3.0f, 6.0f, 9.0f, 12.0f};
    EXPECT_EQ(expectedDimensions, weights.dims().vec());
    ASSERT_EQ(expectedValues.size(), weights.size());
    const auto elements = weights.getHandle();
    for (size_t i = 0; i < expectedValues.size(); ++i) {
      EXPECT_FLOAT_EQ(expectedValues.at(i), elements.raw(i))
          << "Where i = " << i;
    }
  }
  {
    const Constant *constant = mod.getConstantByName("biases");
    ASSERT_TRUE(constant);
    const Tensor &bias = constant->getPayload();
    const std::vector<size_t> expectedDimensions = {4};
    const std::vector<float> expectedValues = {0.1f, 0.2f, 0.3f, 0.4f};
    EXPECT_EQ(expectedDimensions, bias.dims().vec());
    ASSERT_EQ(expectedValues.size(), bias.size());
    const auto elements = bias.getHandle();
    for (size_t i = 0; i < expectedValues.size(); ++i) {
      EXPECT_FLOAT_EQ(expectedValues.at(i), elements.raw(i))
          << "Where i = " << i;
    }
  }

  // We don't actually check that the output is correct, because this is
  // already covered in the Operator.FCWithFlatten/* tests.
}

/// Test loading clip op from a Caffe2 model.
/// Test with arg min = 20.0 max = 60.0
TEST(caffe2, importClip) {
  ExecutionEngine EE{BackendKind::Interpreter};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename(GLOW_DATA_PATH
                              "tests/models/caffe2Models/clip_op_net.pbtxt");
  std::string NetWeightFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/empty_init_net.pbtxt");

  Context ctx;
  Placeholder *output;
  Tensor inputs_0(ElemKind::FloatTy, {5, 5});
  // Destroy the loader after the graph is loaded since the following execution
  // will not depend on anyting from the loader.
  {
    Caffe2ModelLoader caffe2LD(NetDescFilename, NetWeightFilename, {"inputs_0"},
                               {&inputs_0.getType()}, *F);
    output = EXIT_ON_ERR(caffe2LD.getSingleOutput());
    ctx.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(ctx, &mod, {"inputs_0"}, {&inputs_0});
  }

  EXPECT_EQ(F->getNodes().size(), 5);
  auto *saveNode = getSaveNodeFromDest(output);
  auto *minNode = llvm::dyn_cast<MinNode>(saveNode->getInput().getNode());
  ASSERT_TRUE(minNode);
  auto *maxNode = llvm::dyn_cast<MaxNode>(minNode->getLHS().getNode());
  ASSERT_TRUE(maxNode);
  auto *maxSplatNode = llvm::dyn_cast<SplatNode>(minNode->getRHS().getNode());
  ASSERT_TRUE(maxSplatNode);
  EXPECT_EQ(maxSplatNode->getValue(), 60.0);
  auto *minSplatNode = llvm::dyn_cast<SplatNode>(maxNode->getRHS().getNode());
  ASSERT_TRUE(minSplatNode);
  EXPECT_EQ(minSplatNode->getValue(), 20.0);
  auto *inputNode = llvm::dyn_cast<Placeholder>(maxNode->getLHS().getNode());
  ASSERT_EQ(inputNode, mod.getPlaceholderByName("inputs_0"));
  // We have one input and one output.
  EXPECT_EQ(mod.getPlaceholders().size(), 2);
}

/// Test loading clip op from a Caffe2 model with default arg values:
/// min = std::numeric_limits<float>::lowest()
/// max = std::numeric_limits<float>::max()
TEST(caffe2, importClipDefault) {
  ExecutionEngine EE{BackendKind::Interpreter};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/clip_op_default_net.pbtxt");
  std::string NetWeightFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/empty_init_net.pbtxt");

  Context ctx;
  Placeholder *output;
  Tensor inputs_0(ElemKind::FloatTy, {5, 5});

  // Destroy the loader after the graph is loaded since the following execution
  // will not depend on anyting from the loader.
  {
    Caffe2ModelLoader caffe2LD(NetDescFilename, NetWeightFilename, {"inputs_0"},
                               {&inputs_0.getType()}, *F);
    output = EXIT_ON_ERR(caffe2LD.getSingleOutput());
    ctx.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(ctx, &mod, {"inputs_0"}, {&inputs_0});
  }
  EXPECT_EQ(F->getNodes().size(), 5);
  auto *saveNode = getSaveNodeFromDest(output);
  auto *minNode = llvm::dyn_cast<MinNode>(saveNode->getInput().getNode());
  ASSERT_TRUE(minNode);
  auto *maxNode = llvm::dyn_cast<MaxNode>(minNode->getLHS().getNode());
  ASSERT_TRUE(maxNode);
  auto *maxSplatNode = llvm::dyn_cast<SplatNode>(minNode->getRHS().getNode());
  ASSERT_TRUE(maxSplatNode);
  EXPECT_EQ(maxSplatNode->getValue(), std::numeric_limits<float>::max());
  auto *minSplatNode = llvm::dyn_cast<SplatNode>(maxNode->getRHS().getNode());
  ASSERT_TRUE(minSplatNode);
  EXPECT_EQ(minSplatNode->getValue(), std::numeric_limits<float>::lowest());
  auto *inputNode = llvm::dyn_cast<Placeholder>(maxNode->getLHS().getNode());
  ASSERT_EQ(inputNode, mod.getPlaceholderByName("inputs_0"));
  // We have one input and one output.
  EXPECT_EQ(mod.getPlaceholders().size(), 2);
}

/// Test loading a ReplaceNaN operator.
TEST(caffe2, replaceNaN) {
  ExecutionEngine EE{BackendKind::Interpreter};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/replace_nan_predict_net.pbtxt");
  std::string NetWeightFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/empty_init_net.pbtxt");

  Context ctx;
  Placeholder *output;
  Tensor input(ElemKind::FloatTy, {10, 10});

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

  // High level checks on the content of the graph.
  // We have 1 IsNaN, 1 Splat, 1 Select and 1 Output.
  EXPECT_EQ(F->getNodes().size(), 4);
  auto *saveNode = getSaveNodeFromDest(output);
  auto *selectNode = llvm::dyn_cast<SelectNode>(saveNode->getInput().getNode());
  ASSERT_TRUE(selectNode);
  auto *isNaNNode = llvm::dyn_cast<IsNaNNode>(selectNode->getCond().getNode());
  ASSERT_TRUE(isNaNNode);
  auto *splatNode = llvm::dyn_cast<SplatNode>(selectNode->getLHS().getNode());
  ASSERT_TRUE(splatNode);
  auto *inputNode = llvm::dyn_cast<Placeholder>(selectNode->getRHS().getNode());
  ASSERT_EQ(inputNode, mod.getPlaceholderByName("input"));

  // We have one input and one output.
  EXPECT_EQ(mod.getPlaceholders().size(), 2);
}

/// Test loading a DotProduct operator with 1D inputs.
TEST(caffe2, dotProduct1D) {
  ExecutionEngine EE{BackendKind::Interpreter};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/dot_product_predict_net.pbtxt");
  std::string NetWeightFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/empty_init_net.pbtxt");

  Placeholder *output;

  // Input tensors.
  constexpr std::size_t kDataSize = 10;
  auto type = mod.uniqueType(ElemKind::FloatTy, {kDataSize});

  // Destroy the loader after the graph is loaded to ensure the function F
  // does not depend on anything stored in it.
  {
    Caffe2ModelLoader caffe2LD(NetDescFilename, NetWeightFilename, {"X", "Y"},
                               {type, type}, *F);
    output = EXIT_ON_ERR(caffe2LD.getSingleOutput());
  }

  // Check that the shape of the output matches that of the expected output.
  EXPECT_TRUE(output->dims().equals({kDataSize}));

  // High level checks on the content of the graph.
  // We have 1 Mul and 1 Output.
  EXPECT_EQ(F->getNodes().size(), 2);

  // Check that the graph has the expected shape (Mul -> Save),
  // starting from the output.
  auto *saveNode = getSaveNodeFromDest(output);
  auto *MN = llvm::dyn_cast<MulNode>(saveNode->getInput());
  ASSERT_TRUE(MN);

  // We have two inputs and one output.
  EXPECT_EQ(mod.getPlaceholders().size(), 3);
}

// Test loading a DotProduct operator with 2D inputs.
TEST(caffe2, dotProduct2D) {
  ExecutionEngine EE{BackendKind::Interpreter};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/dot_product_predict_net.pbtxt");
  std::string NetWeightFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/empty_init_net.pbtxt");

  Placeholder *output;

  // Input tensors.
  constexpr std::size_t kRows = 10;
  constexpr std::size_t kCols = 20;
  auto type = mod.uniqueType(ElemKind::FloatTy, {kRows, kCols});

  // Destroy the loader after the graph is loaded to ensure the function F
  // does not depend on anything stored in it.
  {
    Caffe2ModelLoader caffe2LD(NetDescFilename, NetWeightFilename, {"X", "Y"},
                               {type, type}, *F);
    output = EXIT_ON_ERR(caffe2LD.getSingleOutput());
  }

  // Check that the shape of the output matches that of the expected output.
  EXPECT_TRUE(output->dims().equals({kRows}));

  // High level checks on the content of the graph.
  // We have 1 Mul, 1 BatchedReduceAdd and 1 Output.
  EXPECT_EQ(F->getNodes().size(), 3);

  // Check that the graph has the expected shape
  // (Mul -> BatchedReduceAdd -> Save), starting from the output.
  auto *saveNode = getSaveNodeFromDest(output);
  auto *BRA = llvm::dyn_cast<BatchedReduceAddNode>(saveNode->getInput());
  ASSERT_TRUE(BRA);
  ASSERT_EQ(BRA->getNumInputs(), 1);

  auto *MN = llvm::dyn_cast<MulNode>(BRA->getBatch());
  ASSERT_TRUE(MN);

  // We have two inputs and one output.
  EXPECT_EQ(mod.getPlaceholders().size(), 3);
}

// Test loading a BatchBoxCox operator.
TEST(caffe2, batchBoxCox) {
  ExecutionEngine EE{BackendKind::Interpreter};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename(
      GLOW_DATA_PATH
      "tests/models/caffe2Models/batch_box_cox_predict_net.pbtxt");
  std::string NetWeightFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/empty_init_net.pbtxt");

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

  std::string NetDescFilename(GLOW_DATA_PATH
                              "tests/models/caffe2Models/eq_op_net.pbtxt");
  std::string NetWeightFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/empty_init_net.pbtxt");

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
      GLOW_DATA_PATH "tests/models/caffe2Models/lengths_to_ranges.pbtxt");
  std::string NetWeightFilename(
      GLOW_DATA_PATH
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

  std::string NetDescFilename(GLOW_DATA_PATH
                              "tests/models/caffe2Models/logit_op_net.pbtxt");
  std::string NetWeightFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/empty_init_net.pbtxt");

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
      GLOW_DATA_PATH "tests/models/caffe2Models/sparse_to_dense.pbtxt");
  std::string NetWeightFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/empty_init_net.pbtxt");

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

TEST(caffe2, SparseToDenseMask) {
  ExecutionEngine EE{BackendKind::Interpreter};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename(
      GLOW_DATA_PATH
      "tests/models/caffe2Models/sparse_to_dense_mask_op_net.pbtxt");
  std::string NetWeightFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/empty_init_net.pbtxt");

  Placeholder *output;
  Context ctx;

  Tensor indices(ElemKind::Int64ITy, {4});
  Tensor values(ElemKind::FloatTy, {4, 10, 20, 30});
  Tensor defaultValue(ElemKind::FloatTy, {10, 20, 30});

  // Destroy the loader after the graph is loaded since the following execution
  // will not depend on anything from the loader.
  {
    // Loaded protos must have at least one external output, so load an unused
    // output and type to satisfy it. It is named unused_output in
    // empty_predict_net.pbtxt.
    Caffe2ModelLoader caffe2LD(
        NetDescFilename, NetWeightFilename,
        {"indices", "values", "defaultValue"},
        {&indices.getType(), &values.getType(), &defaultValue.getType()}, *F);
    output = EXIT_ON_ERR(caffe2LD.getSingleOutput());
  }

  ASSERT_TRUE(output);

  // Graph has 2 nodes: Save and SparseToDenseMask
  EXPECT_EQ(F->getNodes().size(), 2);

  // One constant was created for implicit Lengths input
  EXPECT_EQ(mod.getConstants().size(), 1);

  // Net has 3 inputs.
  EXPECT_EQ(mod.getPlaceholders().size(), 4);

  auto *saveNode = getSaveNodeFromDest(output);
  auto *N = llvm::dyn_cast<SparseToDenseMaskNode>(saveNode->getInput());
  ASSERT_TRUE(N);

  // Check that no batch dimension was added because Lengths was not given.
  EXPECT_TRUE(N->dims(0).equals({6, 10, 20, 30}));
  // Check that mask was read correctly.
  EXPECT_TRUE(N->getMask().equals({42, 100, 300, 1, 0, 312}));
}

/// Test loading NCHW2NHWC op.
TEST(caffe2, testNCHW2NHWC) {
  ExecutionEngine EE{BackendKind::Interpreter};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/NCHW2NHWC_predict_net.pbtxt");
  std::string NetWeightFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/empty_init_net.pbtxt");

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

/// Test loading a LengthsSum operator.
TEST(caffe2, lengthsSum) {
  ExecutionEngine EE{BackendKind::Interpreter};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename(GLOW_DATA_PATH
                              "tests/models/caffe2Models/lengths_sum.pbtxt");
  std::string NetWeightFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/empty_init_net.pbtxt");

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

/// Test loading a GatherRanges op.
TEST(caffe2, gatherRanges) {
  ExecutionEngine EE;
  auto &mod = EE.getModule();
  auto *F = mod.createFunction("main");

  std::string NetDescFilename(GLOW_DATA_PATH
                              "tests/models/caffe2Models/gather_ranges.pbtxt");
  std::string NetWeightFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/empty_init_net.pbtxt");

  Placeholder *output;
  Tensor data(ElemKind::FloatTy, {6});
  Tensor ranges(ElemKind::Int32ITy, {2, 2, 2});

  {
    Caffe2ModelLoader caffe2LD(NetDescFilename, NetWeightFilename,
                               {"data", "ranges"},
                               {&data.getType(), &ranges.getType()}, *F);
    output = EXIT_ON_ERR(caffe2LD.getOutputByName("output"));
  }

  // Verify structure: PH/PH -> GatherRanges -> Save -> PH/PH.
  ASSERT_EQ(mod.getPlaceholders().size(), 4);
  ASSERT_EQ(F->getNodes().size(), 3);
  auto *save = getSaveNodeFromDest(output);
  auto *gatherRanges =
      llvm::dyn_cast<GatherRangesNode>(save->getInput().getNode());
  ASSERT_TRUE(gatherRanges);
  EXPECT_TRUE(gatherRanges->getOutput().dims().equals({5}));
  EXPECT_TRUE(gatherRanges->getLengths().dims().equals({2}));
}

/// Verify that different fill types are loaded with the correct types.
TEST(caffe2, tensorFillsTest) {
  ExecutionEngine EE{BackendKind::Interpreter};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/empty_predict_net.pbtxt");
  std::string NetWeightFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/fill_test_init_net.pbtxt");

  Constant *tensorFillFloat, *tensorIntFill, *tensorInt64Fill,
      *tensorStringToUInt8Fill;

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
    tensorIntFill = llvm::dyn_cast<Constant>(EXIT_ON_ERR(
        caffe2LD.getNodeValueOrCreateConstantByName("tensor_int_fill")));
    tensorInt64Fill = llvm::dyn_cast<Constant>(EXIT_ON_ERR(
        caffe2LD.getNodeValueOrCreateConstantByName("tensor_int64_fill")));
    tensorStringToUInt8Fill = llvm::dyn_cast<Constant>(
        EXIT_ON_ERR(caffe2LD.getNodeValueOrCreateConstantByName(
            "tensor_string_to_uint8_fill")));
  }

  ASSERT_TRUE(tensorFillFloat);
  ASSERT_TRUE(tensorIntFill);
  ASSERT_TRUE(tensorInt64Fill);
  ASSERT_TRUE(tensorStringToUInt8Fill);

  // All fills in fill_test_init_net.pbtxt use shape {2, 2}.
  const std::vector<size_t> expectedDims = {2, 2};
  ASSERT_TRUE(tensorFillFloat->dims().equals(expectedDims));
  ASSERT_TRUE(tensorIntFill->dims().equals(expectedDims));
  ASSERT_TRUE(tensorInt64Fill->dims().equals(expectedDims));
  ASSERT_TRUE(tensorStringToUInt8Fill->dims().equals(expectedDims));

  auto tensorFillFloatH = tensorFillFloat->getPayload().getHandle<float>();
  auto tensorIntFillH = tensorIntFill->getPayload().getHandle<int32_t>();
  auto tensorInt64FillH = tensorInt64Fill->getPayload().getHandle<int64_t>();
  // We load GivenTensorByteStringToUInt8Fill as Int8QTy with dummy scale/offset
  // for now, because it's only used for rowwise-quantized tensors.
  auto tensorStringToUInt8FillH =
      tensorStringToUInt8Fill->getPayload().getHandle<int8_t>();

  // All fills in fill_test_init_net.pbtxt are set to 0 through 3.
  for (size_t i = 0; i < 4; i++) {
    EXPECT_FLOAT_EQ(tensorFillFloatH.raw(i), (float)i);
    EXPECT_EQ(tensorIntFillH.raw(i), (int32_t)i);
    EXPECT_EQ(tensorInt64FillH.raw(i), (int64_t)i);
    EXPECT_EQ(tensorStringToUInt8FillH.raw(i), (int8_t)i);
  }
}

TEST(caffe2, Alias) {
  ExecutionEngine EE{BackendKind::Interpreter};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  llvm::StringRef NetDescFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/alias_op_net.pbtxt");
  std::string NetWeightFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/empty_init_net.pbtxt");

  Placeholder *output;
  Context ctx;

  Tensor input(ElemKind::FloatTy, {1, 2, 3, 4});

  // Destroy the loader after the graph is loaded since the following execution
  // will not depend on anything from the loader.
  {
    // Loaded protos must have at least one external output, so load an unused
    // output and type to satisfy it. It is named unused_output in
    // empty_predict_net.pbtxt.
    Caffe2ModelLoader caffe2LD(NetDescFilename, NetWeightFilename, {"X"},
                               {&input.getType()}, *F);
    output = EXIT_ON_ERR(caffe2LD.getSingleOutput());
  }

  ASSERT_TRUE(output);

  // The only node is Save.
  EXPECT_EQ(F->getNodes().size(), 1);

  auto *saveNode = getSaveNodeFromDest(output);
  auto *N = llvm::dyn_cast<Placeholder>(saveNode->getInput());
  EXPECT_TRUE(N);
}

TEST(caffe2, Modulo) {
  ExecutionEngine EE{BackendKind::Interpreter};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename(GLOW_DATA_PATH
                              "tests/models/caffe2Models/modulo_op_net.pbtxt");
  std::string NetWeightFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/fill_test_init_net.pbtxt");

  Placeholder *output;
  Context ctx;

  Tensor data(ElemKind::Int64ITy, {7});

  // Destroy the loader after the graph is loaded since the following execution
  // will not depend on anything from the loader.
  {
    // Loaded protos must have at least one external output, so load an unused
    // output and type to satisfy it. It is named unused_output in
    // empty_predict_net.pbtxt.
    Caffe2ModelLoader caffe2LD(NetDescFilename, NetWeightFilename, {"data"},
                               {&data.getType()}, *F);
    output = EXIT_ON_ERR(caffe2LD.getSingleOutput());
  }

  ASSERT_TRUE(output);

  // Graph has 2 nodes: Save and Modulo.
  EXPECT_EQ(F->getNodes().size(), 2);

  // Net has 1 inputs.
  EXPECT_EQ(mod.getPlaceholders().size(), 2);

  // Input to save node is ModuloNode.
  auto *saveNode = getSaveNodeFromDest(output);
  auto *N = llvm::dyn_cast<ModuloNode>(saveNode->getInput());
  ASSERT_TRUE(N);
}

/// Test loading an ElementwiseLinear operator.
TEST(caffe2, elementwiseLinear) {
  ExecutionEngine EE{BackendKind::Interpreter};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/elementwise_linear_net.pbtxt");
  std::string NetWeightFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/empty_init_net.pbtxt");

  Context ctx;
  Placeholder *output;
  Tensor X(ElemKind::FloatTy, {10, 5});
  Tensor w(ElemKind::FloatTy, {10}), b(ElemKind::FloatTy, {10});

  // Destroy the loader after the graph is loaded since the following execution
  // will not depend on anyting from the loader.
  {
    Caffe2ModelLoader caffe2LD(NetDescFilename, NetWeightFilename,
                               {"X", "w", "b"},
                               {&X.getType(), &w.getType(), &b.getType()}, *F);
    output = EXIT_ON_ERR(caffe2LD.getSingleOutput());
  }

  // Check that the shape of the output matches that of the input.
  std::vector<size_t> expectedDims = {10, 5};
  EXPECT_TRUE(output->dims().vec() == expectedDims);

  // High level checks on the content of the graph.
  // It should look like this:
  //
  //            X           w            b
  //            |           |            |
  //            |           v            v
  //            |        Reshape      Reshape
  //            |           |            |
  //            |           v            v
  //            |         Tile         Tile
  //            |         /             /
  //            v  v------             /
  //            Mul                   /
  //             |   /---------------
  //             v  v
  //             Add
  //              |
  //              v
  //             Save

  EXPECT_EQ(F->getNodes().size(), 7);
  auto *save = getSaveNodeFromDest(output);
  auto *add = llvm::dyn_cast<AddNode>(save->getInput().getNode());
  ASSERT_TRUE(add);
  auto *mul = llvm::dyn_cast<MulNode>(add->getLHS().getNode());
  ASSERT_TRUE(mul);
  auto *bTile = llvm::dyn_cast<TileNode>(add->getRHS().getNode());
  ASSERT_TRUE(bTile);
  EXPECT_EQ(bTile->getAxis(), 1);
  auto *XPH = llvm::dyn_cast<Placeholder>(mul->getRHS().getNode());
  EXPECT_EQ(XPH, mod.getPlaceholderByName("X"));
  auto *wTile = llvm::dyn_cast<TileNode>(mul->getLHS().getNode());
  ASSERT_TRUE(wTile);
  EXPECT_EQ(wTile->getAxis(), 1);
  auto *bReshape = llvm::dyn_cast<ReshapeNode>(bTile->getInput().getNode());
  ASSERT_TRUE(bReshape);
  auto *wReshape = llvm::dyn_cast<ReshapeNode>(wTile->getInput().getNode());
  ASSERT_TRUE(wReshape);
  auto *wPH = llvm::dyn_cast<Placeholder>(wReshape->getInput().getNode());
  EXPECT_EQ(wPH, mod.getPlaceholderByName("w"));
  auto *bPH = llvm::dyn_cast<Placeholder>(bReshape->getInput().getNode());
  EXPECT_EQ(bPH, mod.getPlaceholderByName("b"));

  // We have three inputs and one output.
  EXPECT_EQ(mod.getPlaceholders().size(), 4);
}

/// Test loading an ElementwiseLinear operator with no axis specified.
TEST(caffe2, elementwiseLinearUnspecifiedAxis) {
  ExecutionEngine EE{BackendKind::Interpreter};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename(
      GLOW_DATA_PATH
      "tests/models/caffe2Models/elementwise_linear_default_net.pbtxt");
  std::string NetWeightFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/empty_init_net.pbtxt");

  Context ctx;
  Placeholder *output;

  // Since the loader will assume that axis = 1, the 0th dim of the shapes of w
  // and b must match the 1st dim of X.
  Tensor X(ElemKind::FloatTy, {5, 10});
  Tensor w(ElemKind::FloatTy, {10}), b(ElemKind::FloatTy, {10});

  // Destroy the loader after the graph is loaded since the following execution
  // will not depend on anyting from the loader.
  {
    Caffe2ModelLoader caffe2LD(NetDescFilename, NetWeightFilename,
                               {"X", "w", "b"},
                               {&X.getType(), &w.getType(), &b.getType()}, *F);
    output = EXIT_ON_ERR(caffe2LD.getSingleOutput());
  }

  // Check that the shape of the output matches that of the input.
  std::vector<size_t> expectedDims = {5, 10};
  EXPECT_TRUE(output->dims().vec() == expectedDims);

  // High level checks on the content of the graph.
  // It should look like this:
  //
  //            X           w            b
  //            |           |            |
  //            |           v            v
  //            |        Reshape      Reshape
  //            |           |            |
  //            |           v            v
  //            |         Tile         Tile
  //            |         /             /
  //            v  v------             /
  //            Mul                   /
  //             |   /---------------
  //             v  v
  //             Add
  //              |
  //              v
  //             Save

  EXPECT_EQ(F->getNodes().size(), 7);
  auto *save = getSaveNodeFromDest(output);
  auto *add = llvm::dyn_cast<AddNode>(save->getInput().getNode());
  ASSERT_TRUE(add);
  auto *mul = llvm::dyn_cast<MulNode>(add->getLHS().getNode());
  ASSERT_TRUE(mul);
  auto *bTile = llvm::dyn_cast<TileNode>(add->getRHS().getNode());
  ASSERT_TRUE(bTile);
  EXPECT_EQ(bTile->getAxis(), 0);
  auto *XPH = llvm::dyn_cast<Placeholder>(mul->getRHS().getNode());
  EXPECT_EQ(XPH, mod.getPlaceholderByName("X"));
  auto *wTile = llvm::dyn_cast<TileNode>(mul->getLHS().getNode());
  ASSERT_TRUE(wTile);
  EXPECT_EQ(wTile->getAxis(), 0);
  auto *bReshape = llvm::dyn_cast<ReshapeNode>(bTile->getInput().getNode());
  ASSERT_TRUE(bReshape);
  auto *wReshape = llvm::dyn_cast<ReshapeNode>(wTile->getInput().getNode());
  ASSERT_TRUE(wReshape);
  auto *wPH = llvm::dyn_cast<Placeholder>(wReshape->getInput().getNode());
  EXPECT_EQ(wPH, mod.getPlaceholderByName("w"));
  auto *bPH = llvm::dyn_cast<Placeholder>(bReshape->getInput().getNode());
  EXPECT_EQ(bPH, mod.getPlaceholderByName("b"));

  // We have three inputs and one output.
  EXPECT_EQ(mod.getPlaceholders().size(), 4);
}

/// Test loading SparseLengthsWeightedSum8BitsRowwise. This is created as a
/// RowwiseQuantizedSparseLengthsWeightedSumNode. The following inputs/outputs
/// are used/expected for this test. Note that the DATA input is
/// rowwise-quantized in the init_net proto. Scales/offsets are loaded in a
/// separate tensor scales_bias. The C2 loader will copy the scales/offsets into
/// separate Constants for use by RowwiseQuantizedSparseLengthsWeightedSumNode.
///    DATA  =   [[2.0, -0.5, 13]]
///    WEIGHTS = [3, 1, 0, 0, 0, 0, 2, -0.5]
///    INDICES = [1, 0, 2, 0, 1, 2, 2, 0]
///    LENGTHS = [3, 0, 3, 2]
///    OUTPUT =  [[0.5, 0, 0, 25]]
TEST(caffe2, SparseLengthsWeightedSum8BitsRowwise) {
  ExecutionEngine EE{BackendKind::Interpreter};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename(
      GLOW_DATA_PATH
      "tests/models/caffe2Models/"
      "rowwise_quantized_sparse_lengths_weighted_sum_predict_net.pbtxt");
  std::string NetWeightFilename(
      GLOW_DATA_PATH
      "tests/models/caffe2Models/"
      "rowwise_quantized_sparse_lengths_weighted_sum_init_net.pbtxt");

  Placeholder *output, *indices, *lengths;
  Context ctx;

  TypeRef indicesType = F->getParent()->uniqueType(ElemKind::Int64ITy, {8});
  TypeRef lengthsType = F->getParent()->uniqueType(ElemKind::Int32ITy, {4});

  // Destroy the loader after the graph is loaded since the following execution
  // will not depend on anyting from the loader.
  {
    Caffe2ModelLoader caffe2LD(NetDescFilename, NetWeightFilename,
                               {"indices", "lengths"},
                               {indicesType, lengthsType}, *F);

    indices = llvm::dyn_cast<Placeholder>(
        EXIT_ON_ERR(caffe2LD.getNodeValueByName("indices")));
    lengths = llvm::dyn_cast<Placeholder>(
        EXIT_ON_ERR(caffe2LD.getNodeValueByName("lengths")));
    output = EXIT_ON_ERR(caffe2LD.getSingleOutput());
  }

  ASSERT_TRUE(indices);
  ASSERT_TRUE(lengths);

  ctx.allocate(indices)->getHandle<int64_t>() = {
      1, 0, 2, 0, 1, 2, 2, 0,
  };
  ctx.allocate(lengths)->getHandle<int32_t>() = {
      3,
      0,
      3,
      2,
  };

  // High level check on the content of the graph. We have 1 rowwise-quantized
  // SLWS and 1 save.
  EXPECT_EQ(F->getNodes().size(), 2);
  SaveNode *saveNode = getSaveNodeFromDest(output);
  RowwiseQuantizedSparseLengthsWeightedSumNode *RWQSLWS =
      llvm::dyn_cast<RowwiseQuantizedSparseLengthsWeightedSumNode>(
          saveNode->getInput().getNode());
  ASSERT_TRUE(RWQSLWS);
  // Check that the weights input is a Constant node.
  Constant *weights = llvm::dyn_cast<Constant>(RWQSLWS->getWeights().getNode());
  ASSERT_TRUE(weights);

  // We have 3 placeholders: 1 for save, and then indices and lengths.
  EXPECT_EQ(mod.getPlaceholders().size(), 3);

  // We have 5 constants: originally fused data (no longer used), data, scales,
  // offsets, and weights.
  EXPECT_EQ(mod.getConstants().size(), 5);

  EE.compile(CompilationMode::Infer, F);

  // Post compile, DCE should have gotten rid of the originally fused data
  // Constant, as it is no longer used.
  EXPECT_EQ(mod.getConstants().size(), 4);

  EE.run(ctx);

  Tensor &result = *ctx.get(output);
  Tensor expected(ElemKind::FloatTy, {4, 1});
  expected.getHandle() = {
      0.5,
      0,
      0,
      25,
  };

  EXPECT_TRUE(expected.isEqual(result, 0.03f));
}

/// Test loading SparseLengthsSum8BitsRowwise. This is created as a
/// RowwiseQuantizedSparseLengthsWeightedSumNode. The following inputs/outputs
/// are used/expected for this test. Note that the DATA input is
/// rowwise-quantized in the init_net proto. Scales/offsets are loaded in a
/// separate tensor scales_bias. The C2 loader will copy the scales/offsets into
/// separate Constants for use by RowwiseQuantizedSparseLengthsSumNode.
///    DATA  = [
///        [1.0, 1.2],
///        [2.3, 3.4],
///        [4.5, 5.7],
///    ]
///    INDICES = [2, 0, 1, 2, 0, 0, 0, 0]
///    LENGTHS = [2, 0, 2, 1, 3]
///    OUTPUT = [
///        [5.5, 6.9],
///        [0.0, 0.0],
///        [6.8, 9.1],
///        [1.0, 1.2],
///        [3.0, 3.6],
///    ]
TEST(caffe2, SparseLengthsSum8BitsRowwise) {
  ExecutionEngine EE{BackendKind::Interpreter};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/"
                     "rowwise_quantized_sparse_lengths_sum_predict_net.pbtxt");
  std::string NetWeightFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/"
                     "rowwise_quantized_sparse_lengths_sum_init_net.pbtxt");

  Placeholder *output, *indices, *lengths;
  Context ctx;

  TypeRef indicesType = F->getParent()->uniqueType(ElemKind::Int64ITy, {8});
  TypeRef lengthsType = F->getParent()->uniqueType(ElemKind::Int32ITy, {5});

  // Destroy the loader after the graph is loaded since the following execution
  // will not depend on anyting from the loader.
  {
    Caffe2ModelLoader caffe2LD(NetDescFilename, NetWeightFilename,
                               {"indices", "lengths"},
                               {indicesType, lengthsType}, *F);

    indices = llvm::dyn_cast<Placeholder>(
        EXIT_ON_ERR(caffe2LD.getNodeValueByName("indices")));
    lengths = llvm::dyn_cast<Placeholder>(
        EXIT_ON_ERR(caffe2LD.getNodeValueByName("lengths")));
    output = EXIT_ON_ERR(caffe2LD.getSingleOutput());
  }

  ASSERT_TRUE(indices);
  ASSERT_TRUE(lengths);

  ctx.allocate(indices)->getHandle<int64_t>() = {
      2, 0, 1, 2, 0, 0, 0, 0,
  };
  ctx.allocate(lengths)->getHandle<int32_t>() = {
      2, 0, 2, 1, 3,
  };

  // High level check on the content of the graph. We have 1 rowwise-quantized
  // SLWS (which implements SLS), 1 Splat for the weights, and 1 save.
  EXPECT_EQ(F->getNodes().size(), 3);
  SaveNode *saveNode = getSaveNodeFromDest(output);
  RowwiseQuantizedSparseLengthsWeightedSumNode *RWQSLS =
      llvm::dyn_cast<RowwiseQuantizedSparseLengthsWeightedSumNode>(
          saveNode->getInput().getNode());
  ASSERT_TRUE(RWQSLS);
  SplatNode *splatNode =
      llvm::dyn_cast<SplatNode>(RWQSLS->getWeights().getNode());
  ASSERT_TRUE(splatNode);
  EXPECT_EQ(splatNode->getValue(), 1.0f);

  // We have 3 placeholders: 1 for save, and then indices and lengths.
  EXPECT_EQ(mod.getPlaceholders().size(), 3);

  // We have 5 constants: originally fused data (no longer used), data, scales,
  // and offsets.
  EXPECT_EQ(mod.getConstants().size(), 4);

  EE.compile(CompilationMode::Infer, F);

  // Post compile, DCE should have gotten rid of the originally fused data
  // Constant, as it is no longer used.
  EXPECT_EQ(mod.getConstants().size(), 3);

  EE.run(ctx);

  Tensor &result = *ctx.get(output);
  Tensor expected(ElemKind::FloatTy, {5, 2});
  expected.getHandle() = {
      5.5f, 6.9f, 0.0f, 0.0f, 6.8f, 9.1f, 1.0f, 1.2f, 3.0f, 3.6f,
  };

  EXPECT_TRUE(expected.isEqual(result, 0.02f));
}

/// Test loading SparseLengthsWeightedSumFused8BitRowwise. This is created as a
/// RowwiseQuantizedSparseLengthsWeightedSumNode. The following inputs/outputs
/// are used/expected for this test. Note that the DATA input is
/// rowwise-quantized in the init_net proto.
///    DATA  =   [[2.0, -0.5, 13]]
///    WEIGHTS = [3, 1, 0, 0, 0, 0, 2, -0.5]
///    INDICES = [1, 0, 2, 0, 1, 2, 2, 0]
///    LENGTHS = [3, 0, 3, 2]
///    OUTPUT =  [[0.5, 0, 0, 25]]
TEST(caffe2, SparseLengthsWeightedSumFused8BitRowwise) {
  ExecutionEngine EE{BackendKind::Interpreter};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename(
      GLOW_DATA_PATH
      "tests/models/caffe2Models/"
      "fused_rowwise_quantized_sparse_lengths_weighted_sum_predict_net.pbtxt");
  std::string NetWeightFilename(
      GLOW_DATA_PATH
      "tests/models/caffe2Models/"
      "fused_rowwise_quantized_sparse_lengths_weighted_sum_init_net.pbtxt");

  Placeholder *output, *indices, *lengths;
  Context ctx;

  TypeRef indicesType = F->getParent()->uniqueType(ElemKind::Int64ITy, {8});
  TypeRef lengthsType = F->getParent()->uniqueType(ElemKind::Int32ITy, {4});

  // Destroy the loader after the graph is loaded since the following execution
  // will not depend on anyting from the loader.
  {
    Caffe2ModelLoader caffe2LD(NetDescFilename, NetWeightFilename,
                               {"indices", "lengths"},
                               {indicesType, lengthsType}, *F);

    indices = llvm::dyn_cast<Placeholder>(
        EXIT_ON_ERR(caffe2LD.getNodeValueByName("indices")));
    lengths = llvm::dyn_cast<Placeholder>(
        EXIT_ON_ERR(caffe2LD.getNodeValueByName("lengths")));
    output = EXIT_ON_ERR(caffe2LD.getSingleOutput());
  }

  ASSERT_TRUE(indices);
  ASSERT_TRUE(lengths);

  ctx.allocate(indices)->getHandle<int64_t>() = {
      1, 0, 2, 0, 1, 2, 2, 0,
  };
  ctx.allocate(lengths)->getHandle<int32_t>() = {
      3,
      0,
      3,
      2,
  };

  // High level check on the content of the graph. We have 1 rowwise-quantized
  // SLWS and 1 save.
  EXPECT_EQ(F->getNodes().size(), 2);
  SaveNode *saveNode = getSaveNodeFromDest(output);
  FusedRowwiseQuantizedSparseLengthsWeightedSumNode *FRWQSLWS =
      llvm::dyn_cast<FusedRowwiseQuantizedSparseLengthsWeightedSumNode>(
          saveNode->getInput().getNode());
  ASSERT_TRUE(FRWQSLWS);
  // Check that the weights input is a Constant node.
  Constant *weights =
      llvm::dyn_cast<Constant>(FRWQSLWS->getWeights().getNode());
  ASSERT_TRUE(weights);
  // Check that the data input is a Constant node with expected ElemKind.
  Constant *data = llvm::dyn_cast<Constant>(FRWQSLWS->getData().getNode());
  ASSERT_TRUE(data);
  EXPECT_TRUE(data->getElementType() == ElemKind::UInt8FusedQTy);

  // We have 3 placeholders: 1 for save, and then indices and lengths.
  EXPECT_EQ(mod.getPlaceholders().size(), 3);

  // We have 2 constants: data and weights.
  EXPECT_EQ(mod.getConstants().size(), 2);

  EE.compile(CompilationMode::Infer, F);

  EE.run(ctx);

  Tensor &result = *ctx.get(output);
  Tensor expected(ElemKind::FloatTy, {4, 1});
  expected.getHandle() = {
      0.5,
      0,
      0,
      25,
  };

  EXPECT_TRUE(expected.isEqual(result, 0.02f));
}

/// Test loading SparseLengthsSumFused8BitRowwise. This is created as a
/// RowwiseQuantizedSparseLengthsWeightedSumNode. The following inputs/outputs
/// are used/expected for this test. Note that the DATA input is
/// rowwise-quantized in the init_net proto.
///    DATA  = [
///        [1.0, 1.2],
///        [2.3, 3.4],
///        [4.5, 5.7],
///    ]
///    INDICES = [2, 0, 1, 2, 0, 0, 0, 0]
///    LENGTHS = [2, 0, 2, 1, 3]
///    OUTPUT = [
///        [5.5, 6.9],
///        [0.0, 0.0],
///        [6.8, 9.1],
///        [1.0, 1.2],
///        [3.0, 3.6],
///    ]
TEST(caffe2, SparseLengthsSumFused8BitRowwise) {
  ExecutionEngine EE{BackendKind::Interpreter};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename(
      GLOW_DATA_PATH
      "tests/models/caffe2Models/"
      "fused_rowwise_quantized_sparse_lengths_sum_predict_net.pbtxt");
  std::string NetWeightFilename(
      GLOW_DATA_PATH
      "tests/models/caffe2Models/"
      "fused_rowwise_quantized_sparse_lengths_sum_init_net.pbtxt");

  Placeholder *output, *indices, *lengths;
  Context ctx;

  TypeRef indicesType = F->getParent()->uniqueType(ElemKind::Int64ITy, {8});
  TypeRef lengthsType = F->getParent()->uniqueType(ElemKind::Int32ITy, {5});

  // Destroy the loader after the graph is loaded since the following execution
  // will not depend on anyting from the loader.
  {
    Caffe2ModelLoader caffe2LD(NetDescFilename, NetWeightFilename,
                               {"indices", "lengths"},
                               {indicesType, lengthsType}, *F);

    indices = llvm::dyn_cast<Placeholder>(
        EXIT_ON_ERR(caffe2LD.getNodeValueByName("indices")));
    lengths = llvm::dyn_cast<Placeholder>(
        EXIT_ON_ERR(caffe2LD.getNodeValueByName("lengths")));
    output = EXIT_ON_ERR(caffe2LD.getSingleOutput());
  }

  ASSERT_TRUE(indices);
  ASSERT_TRUE(lengths);

  ctx.allocate(indices)->getHandle<int64_t>() = {
      2, 0, 1, 2, 0, 0, 0, 0,
  };
  ctx.allocate(lengths)->getHandle<int32_t>() = {
      2, 0, 2, 1, 3,
  };

  // High level check on the content of the graph. We have 1 rowwise-quantized
  // SLWS (which implements SLS), 1 Splat for the weights, and 1 save.
  EXPECT_EQ(F->getNodes().size(), 3);
  SaveNode *saveNode = getSaveNodeFromDest(output);
  FusedRowwiseQuantizedSparseLengthsWeightedSumNode *FRWQSLS =
      llvm::dyn_cast<FusedRowwiseQuantizedSparseLengthsWeightedSumNode>(
          saveNode->getInput().getNode());
  ASSERT_TRUE(FRWQSLS);
  SplatNode *splatNode =
      llvm::dyn_cast<SplatNode>(FRWQSLS->getWeights().getNode());
  ASSERT_TRUE(splatNode);
  EXPECT_EQ(splatNode->getValue(), 1.0f);
  // Check that the data input is a Constant node with expected ElemKind.
  Constant *data = llvm::dyn_cast<Constant>(FRWQSLS->getData().getNode());
  ASSERT_TRUE(data);
  EXPECT_TRUE(data->getElementType() == ElemKind::UInt8FusedQTy);

  // We have 3 placeholders: 1 for save, and then indices and lengths.
  EXPECT_EQ(mod.getPlaceholders().size(), 3);

  // We have 1 constant: data.
  EXPECT_EQ(mod.getConstants().size(), 1);

  EE.compile(CompilationMode::Infer, F);

  EE.run(ctx);

  Tensor &result = *ctx.get(output);
  Tensor expected(ElemKind::FloatTy, {5, 2});
  expected.getHandle() = {
      5.5f, 6.9f, 0.0f, 0.0f, 6.8f, 9.1f, 1.0f, 1.2f, 3.0f, 3.6f,
  };

  EXPECT_TRUE(expected.isEqual(result, 0.02f));
}
