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
#include "glow/Importer/Caffe2.h"
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

  SaveNode *output;
  // Destroy the loader after the graph is loaded since the following execution
  // will not depend on anyting from the loader.
  {
    Tensor data;
    getNCHWData(&data, 1, 1, 3, 3);
    caffe2ModelLoader caffe2LD(NetDescFilename, NetWeightFilename, {"data"},
                               {&data}, *F);
    output = caffe2LD.getSingleOutput();
  }

  EE.compile(CompilationMode::Infer, F);
  EE.run({}, {});
  auto result = output->getVariable()->getHandle();
  std::vector<size_t> expectedDims = {1, 1, 4, 4};
  std::vector<float> expectedValues = {2,  3,  5,  4,  5, 10, 14, 9,
                                       11, 22, 26, 15, 8, 15, 17, 10};
  EXPECT_TRUE(result.dims().vec() == expectedDims);
  for (size_t i = 0; i < 4 * 4; i++)
    EXPECT_FLOAT_EQ(result.raw(i), expectedValues[i]);
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

  SaveNode *output;
  Tensor inputs_0(ElemKind::FloatTy, {10, 7});
  Tensor inputs_1(ElemKind::FloatTy, {10, 7});
  Tensor inputs_2(ElemKind::FloatTy, {10, 7});
  inputs_0.getHandle().randomize(-3.0, 3.0, mod.getPRNG());
  inputs_1.getHandle().randomize(-3.0, 3.0, mod.getPRNG());
  inputs_2.getHandle().randomize(-3.0, 3.0, mod.getPRNG());
  // Destroy the loader after the graph is loaded since the following execution
  // will not depend on anyting from the loader.
  {
    caffe2ModelLoader caffe2LD(NetDescFilename, NetWeightFilename,
                               {"inputs_0", "inputs_1", "inputs_2"},
                               {&inputs_0, &inputs_1, &inputs_2}, *F);
    output = caffe2LD.getSingleOutput();
  }

  auto result = output->getVariable()->getHandle();
  // Check that the shape of the output matches what Caffe2 expects.
  std::vector<size_t> expectedDims = {10, 3, 7};
  EXPECT_TRUE(result.dims().vec() == expectedDims);

  EE.compile(CompilationMode::Infer, F);
  EE.run({}, {});
  // High level check on the content of the graph.
  // We have 1 reshape, 1 concat, and 1 save.
  EXPECT_EQ(F->getNodes().size(), 3);
  // With have three inputs and one outputs.
  EXPECT_EQ(mod.getVars().size(), 4);

  // Check that the graph has the expected shape,
  // starting from the output.
  auto *reshape = llvm::dyn_cast<ReshapeNode>(output->getInput().getNode());
  ASSERT_TRUE(reshape);
  auto *concat = llvm::dyn_cast<ConcatNode>(reshape->getInput());
  ASSERT_TRUE(concat);
  // We will check that the inputs are correct within
  // the next loop.

  // Check that the output matches the concatenation of
  // all the inputs.
  Tensor *inputs[] = {&inputs_0, &inputs_1, &inputs_2};
  for (size_t i = 0; i < 3; ++i) {
    const auto inputsHandle = inputs[i]->getHandle();
    ASSERT_TRUE(llvm::isa<Variable>(concat->getInputs()[i]));
    EXPECT_TRUE(llvm::cast<Variable>(concat->getInputs()[i])
                    ->getPayload()
                    .isEqual(*inputs[i]));

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

  SaveNode *output;
  Tensor inputs_0(ElemKind::FloatTy, {10, 7});
  Tensor inputs_1(ElemKind::FloatTy, {10, 12});
  Tensor inputs_2(ElemKind::FloatTy, {10, 5});
  inputs_0.getHandle().randomize(-3.0, 3.0, mod.getPRNG());
  inputs_1.getHandle().randomize(-3.0, 3.0, mod.getPRNG());
  inputs_2.getHandle().randomize(-3.0, 3.0, mod.getPRNG());
  // Destroy the loader after the graph is loaded since the following execution
  // will not depend on anyting from the loader.
  {
    caffe2ModelLoader caffe2LD(NetDescFilename, NetWeightFilename,
                               {"inputs_0", "inputs_1", "inputs_2"},
                               {&inputs_0, &inputs_1, &inputs_2}, *F);
    output = caffe2LD.getSingleOutput();
  }

  auto result = output->getVariable()->getHandle();
  // Check that the shape of the output matches what Caffe2 expects.
  std::vector<size_t> expectedDims = {10, 24};
  EXPECT_TRUE(result.dims().vec() == expectedDims);

  EE.compile(CompilationMode::Infer, F);
  EE.run({}, {});
  // High level check on the content of the graph.
  // We have 1 concat, and 1 save.
  EXPECT_EQ(F->getNodes().size(), 2);
  // With have three inputs and one outputs.
  EXPECT_EQ(mod.getVars().size(), 4);

  // Check that the graph has the expected shape,
  // starting from the output.
  auto *concat = llvm::dyn_cast<ConcatNode>(output->getInput());
  ASSERT_TRUE(concat);
  // We will check that the inputs are correct within
  // the next loop.

  // Check that the output matches the concatenation of
  // all the inputs.
  Tensor *inputs[] = {&inputs_0, &inputs_1, &inputs_2};
  size_t columnsChecked = 0;
  for (size_t i = 0; i < 3; ++i) {
    const auto inputsHandle = inputs[i]->getHandle();
    ASSERT_TRUE(llvm::isa<Variable>(concat->getInputs()[i]));
    EXPECT_TRUE(llvm::cast<Variable>(concat->getInputs()[i])
                    ->getPayload()
                    .isEqual(*inputs[i]));

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
  SaveNode *output;
  Tensor inputs_0(ElemKind::FloatTy, {3, 10, 7});
  Tensor inputs_1(ElemKind::FloatTy, {10, 7});
  inputs_0.getHandle().randomize(-3.0, 3.0, mod.getPRNG());
  inputs_1.getHandle().randomize(-3.0, 3.0, mod.getPRNG());
  // Destroy the loader after the graph is loaded since the following execution
  // will not depend on anyting from the loader.
  {
    caffe2ModelLoader caffe2LD(NetDescFilename, NetWeightFilename,
                               {"inputs_0", "inputs_1"}, {&inputs_0, &inputs_1},
                               *F);
    output = caffe2LD.getSingleOutput();
  }
  auto result = output->getVariable()->getHandle();
  // Check that the shape of the output matches what Caffe2 expects.
  std::vector<size_t> expectedDims = {3, 10, 10};
  EXPECT_TRUE(result.dims().vec() == expectedDims);
  // High level check on the content of the graph.
  // We have 1 transpose, 1 matmul, 1 save, and 2 reshapes.
  EXPECT_EQ(F->getNodes().size(), 5);
  // With have 2 inputs and one outputs.
  EXPECT_EQ(mod.getVars().size(), 3);
  // Check that the graph has the expected shape,
  // starting from the output.
  // Batched matmul with broadcasted RHS are lowered
  // to a regular matmul, where LHS is reshaped from
  // a 3D tensor to a flattened matrix.
  auto *reshapeResult =
      llvm::dyn_cast<ReshapeNode>(output->getInput().getNode());
  ASSERT_TRUE(reshapeResult);
  auto *matmul =
      llvm::dyn_cast<MatMulNode>(reshapeResult->getInput().getNode());
  ASSERT_TRUE(matmul);
  const size_t matmulDims[] = {30, 10};
  EXPECT_EQ(matmul->dims(0), llvm::makeArrayRef(matmulDims));
  auto *lhs = llvm::dyn_cast<ReshapeNode>(matmul->getLHS().getNode());
  ASSERT_TRUE(lhs);
  auto *lhsInput = lhs->getInput().getNode();
  ASSERT_TRUE(llvm::isa<Variable>(lhsInput));
  EXPECT_TRUE(llvm::cast<Variable>(lhsInput)->getPayload().isEqual(inputs_0));
  auto *transpose = llvm::dyn_cast<TransposeNode>(matmul->getRHS().getNode());
  ASSERT_TRUE(transpose);
  ASSERT_TRUE(llvm::isa<Variable>(transpose->getInput().getNode()));
  EXPECT_TRUE(llvm::cast<Variable>(transpose->getInput().getNode())
                  ->getPayload()
                  .isEqual(inputs_1));
  // Check that the last two dimensions are swapped.
  const unsigned_t shuffle[] = {1, 0};
  EXPECT_EQ(transpose->getShuffle(), llvm::makeArrayRef(shuffle));
  // We don't actually check that the output is correct, because this
  // should be covered in the OperatorTest for MatMul already.
}
