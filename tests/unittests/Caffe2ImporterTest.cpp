/**
 * Copyright (c) Glow Contributors. See CONTRIBUTORS file.
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

class Caffe2ImporterTest : public ::testing::Test {
protected:
  // By default constant folding at load time is enabled in general, but we do
  // many tests here loading Constants, so keep it false during these tests by
  // default.
  void SetUp() override { glow::setConstantFoldLoaderOpsFlag(false); }
  void TearDown() override { glow::setConstantFoldLoaderOpsFlag(true); }
};

using namespace glow;
/// Test loading of Elementwise Unary Ops floating point.
static void testEltwiseUnaryOpFloat(std::string fileName,
                                    llvm::ArrayRef<dim_t> inputShape,
                                    std::string input_name, float delta,
                                    const std::function<float(float)> &op) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  std::string NetDescFilename =
      std::string(GLOW_DATA_PATH "tests/models/caffe2Models/") + fileName;
  std::string NetWeightFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/empty_init_net.pbtxt");

  PlaceholderBindings bindings;
  Placeholder *graphOutputVar;
  Type input_type(ElemKind::FloatTy, inputShape);
  Caffe2ModelLoader caffe2LD(NetDescFilename, NetWeightFilename,
                             {input_name.c_str()}, {&input_type}, *F);
  graphOutputVar = EXIT_ON_ERR(caffe2LD.getSingleOutput());
  auto PH = mod.getPlaceholderByName(input_name);
  auto *inTensor = bindings.allocate(PH);
  inTensor->getHandle().randomize(-10.0, 10.0, mod.getPRNG());
  EE.compile(CompilationMode::Infer);
  bindings.allocate(mod.getPlaceholders());
  EE.run(bindings);
  auto result = bindings.get(graphOutputVar)->getHandle();
  auto inHandle = inTensor->getHandle();
  ASSERT_TRUE(result.dims() == inputShape);
  for (size_t i = 0; i < result.getType().size(); i++) {
    EXPECT_NEAR(result.raw(i), op(inHandle.raw(i)), delta);
  }
}

TEST_F(Caffe2ImporterTest, importExp) {
  testEltwiseUnaryOpFloat("exp_op_net.pbtxt", {1, 2, 4, 3}, "data", 0.002,
                          [](float a) { return std::exp(a); });
}

/// Test loading conv op from a Caffe2 model.
/// The input is N*C*H*W (1*1*3*3), the kernel is 2,
/// stride is 1, pad is 1, group is 1.
TEST_F(Caffe2ImporterTest, importConv) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename(GLOW_DATA_PATH
                              "tests/models/caffe2Models/predict_net.pbtxt");
  std::string NetWeightFilename(GLOW_DATA_PATH
                                "tests/models/caffe2Models/init_net.pbtxt");

  Placeholder *output;
  PlaceholderBindings bindings;

  // Destroy the loader after the graph is loaded since the following execution
  // will not depend on anything from the loader.
  {
    Tensor data;
    getNCHWData(&data, 1, 1, 3, 3);
    Caffe2ModelLoader caffe2LD(NetDescFilename, NetWeightFilename,
                               {"gpu_0/data_0"}, {&data.getType()}, *F);
    output = EXIT_ON_ERR(caffe2LD.getSingleOutput());

    bindings.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(bindings, &mod, {"gpu_0/data_0"}, {&data});
  }

  auto res = bindings.get(output);
  EE.compile(CompilationMode::Infer);

  EE.run(bindings);
  auto result = res->getHandle();
  std::vector<dim_t> expectedDims = {1, 1, 4, 4};
  std::vector<float> expectedValues = {2,  3,  5,  4,  5, 10, 14, 9,
                                       11, 22, 26, 15, 8, 15, 17, 10};
  EXPECT_TRUE(result.dims().vec() == expectedDims);
  for (size_t i = 0; i < 4 * 4; i++)
    EXPECT_FLOAT_EQ(result.raw(i), expectedValues[i]);
}

/// Test loading ConvRelu op from a Caffe2 model.
/// The input is N*C*H*W (1*1*3*3), the kernel is 2,
/// stride is 1, pad is 1, group is 1.
TEST_F(Caffe2ImporterTest, importConvRelu) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/convrelu_pred_net.pbtxt");
  std::string NetWeightFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/convrelu_init_net.pbtxt");

  Placeholder *output;
  PlaceholderBindings bindings;

  // Destroy the loader after the graph is loaded since the following execution
  // will not depend on anything from the loader.
  {
    Tensor data;
    getNCHWData(&data, 1, 1, 3, 3);
    Caffe2ModelLoader caffe2LD(NetDescFilename, NetWeightFilename,
                               {"gpu_0/data_0"}, {&data.getType()}, *F);
    output = EXIT_ON_ERR(caffe2LD.getSingleOutput());

    bindings.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(bindings, &mod, {"gpu_0/data_0"}, {&data});
  }

  // High level check on the content of the graph. We should have
  // transpose => conv => relu => transpose => save
  EXPECT_EQ(F->getNodes().size(), 5);
  auto *saveNode = getSaveNodeFromDest(output);

  auto *transNode1 =
      llvm::dyn_cast<TransposeNode>(saveNode->getInput().getNode());
  ASSERT_TRUE(transNode1);
  auto *reluNode = llvm::dyn_cast<ReluNode>(transNode1->getInput().getNode());
  ASSERT_TRUE(reluNode);
  auto *convNode =
      llvm::dyn_cast<ConvolutionNode>(reluNode->getInput().getNode());
  ASSERT_TRUE(convNode);
  auto *transNode2 =
      llvm::dyn_cast<TransposeNode>(convNode->getInput().getNode());
  ASSERT_TRUE(transNode2);

  auto res = bindings.get(output);
  EE.compile(CompilationMode::Infer);

  EE.run(bindings);
  auto result = res->getHandle();
  std::vector<dim_t> expectedDims = {1, 1, 4, 4};
  std::vector<float> expectedValues = {2,  3,  5,  4,  5, 10, 14, 9,
                                       11, 22, 26, 15, 8, 15, 17, 10};
  EXPECT_TRUE(result.dims().vec() == expectedDims);
  for (size_t i = 0; i < 4 * 4; i++)
    EXPECT_FLOAT_EQ(result.raw(i), expectedValues[i]);
}

/// Test loading conv op from a Caffe2 model.
/// The input is N*H*W*C (1*3*3*1), the kernel is 2,
/// stride is 1, pad is 1, group is 1.
TEST_F(Caffe2ImporterTest, convNHWC) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/conv_nhwc_predict_net.pbtxt");
  std::string NetWeightFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/conv_nhwc_init_net.pbtxt");

  Placeholder *output;
  PlaceholderBindings bindings;

  Tensor inputs(ElemKind::FloatTy, {1, 3, 3, 1});

  // Destroy the loader after the graph is loaded since the following execution
  // will not depend on anything from the loader.
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

/// Test loading ChannelwiseQuantizedConvolutionNode op from a Caffe2 model.
/// The input is N*H*W*C (1*1*1*4), the kernel is 1,
/// stride is 1, pad is 1, group is 2.
TEST_F(Caffe2ImporterTest, convGroupQuantized) {
  // TODO Due to https://github.com/pytorch/glow/pull/3877
  // the API of channelwise quantized conv has been changed
  // this test is skipped for now and should be enbaled once
  // we fixed.
  GTEST_SKIP();
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename(
      GLOW_DATA_PATH
      "tests/models/caffe2Models/conv_group_quantized_pred_net.pbtxt");
  std::string NetWeightFilename(
      GLOW_DATA_PATH
      "tests/models/caffe2Models/conv_group_quantized_init_net.pbtxt");

  Placeholder *output;
  PlaceholderBindings bindings;

  Tensor input(ElemKind::Int8QTy, {1, 1, 1, 4}, 1.0, 0);

  // Destroy the loader after the graph is loaded since the following execution
  // will not depend on anything from the loader.
  {
    Caffe2ModelLoader caffe2LD(NetDescFilename, NetWeightFilename, {"input"},
                               {&input.getType()}, *F);
    output = EXIT_ON_ERR(caffe2LD.getSingleOutput());
  }

  // High level check on the content of the graph. We have 1
  // ChannelwiseQuantizedConvolutionNode and 1 save.
  EXPECT_EQ(F->getNodes().size(), 2);
  auto *saveNode = getSaveNodeFromDest(output);
  auto *groupwiseConv = llvm::dyn_cast<ChannelwiseQuantizedConvolutionNode>(
      saveNode->getInput().getNode());
  ASSERT_TRUE(groupwiseConv);

  // Check params.
  std::vector<unsigned> expectedKernelsAndStrides = {1, 1};
  std::vector<unsigned> expectedPads = {1, 1, 1, 1};
  EXPECT_EQ(groupwiseConv->getKernels(),
            llvm::makeArrayRef(expectedKernelsAndStrides));
  EXPECT_EQ(groupwiseConv->getStrides(),
            llvm::makeArrayRef(expectedKernelsAndStrides));
  EXPECT_EQ(groupwiseConv->getPads(), llvm::makeArrayRef(expectedPads));
  EXPECT_EQ(groupwiseConv->getGroup(), 2);

  // Check constant inputs.
  Constant *filterConstant =
      llvm::dyn_cast<Constant>(groupwiseConv->getFilter().getNode());
  Constant *biasConstant =
      llvm::dyn_cast<Constant>(groupwiseConv->getBias().getNode());
  Constant *scalesConstant =
      llvm::dyn_cast<Constant>(groupwiseConv->getScales().getNode());
  Constant *offsetsConstant =
      llvm::dyn_cast<Constant>(groupwiseConv->getOffsets().getNode());

  ASSERT_TRUE(filterConstant);
  ASSERT_TRUE(biasConstant);
  ASSERT_TRUE(scalesConstant);
  ASSERT_TRUE(offsetsConstant);

  const auto filterH = filterConstant->getPayload().getHandle<int8_t>();
  const auto biasH = biasConstant->getPayload().getHandle<float>();
  const auto scalesH = scalesConstant->getPayload().getHandle<float>();
  const auto offsetsH = offsetsConstant->getPayload().getHandle<int32_t>();

  for (size_t i = 0; i < filterH.size(); ++i) {
    EXPECT_EQ(filterH.raw(i), i % 2);
  }

  for (size_t i = 0; i < biasH.size(); ++i) {
    EXPECT_EQ(biasH.raw(i), 7);
  }

  for (size_t i = 0; i < scalesH.size(); ++i) {
    EXPECT_EQ(scalesH.raw(i), 6);
  }

  for (size_t i = 0; i < offsetsH.size(); ++i) {
    EXPECT_EQ(offsetsH.raw(i), 5);
  }

  // We have 2 placeholders: 1 input and 1 output.
  EXPECT_EQ(mod.getPlaceholders().size(), 2);
  // We have 4 constants: Bias, Weights, and Weights' separate scales and
  // offsets.
  EXPECT_EQ(mod.getConstants().size(), 4);
}

/// Test loading MaxPool with NHWC order input.
TEST_F(Caffe2ImporterTest, maxPoolNHWC) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename(
      GLOW_DATA_PATH
      "tests/models/caffe2Models/maxpool_nhwc_predict_net.pbtxt");
  std::string NetWeightFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/empty_init_net.pbtxt");

  Placeholder *output;
  PlaceholderBindings bindings;

  Tensor inputs(ElemKind::FloatTy, {1, 3, 3, 1});

  // Destroy the loader after the graph is loaded since the following execution
  // will not depend on anything from the loader.
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

/// Test that loading MaxPool with legacy padding terminates early.
TEST_F(Caffe2ImporterTest, maxPoolLegacyPadding) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename(
      GLOW_DATA_PATH
      "tests/models/caffe2Models/maxpool_legacy_padding_predict_net.pbtxt");
  std::string NetWeightFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/empty_init_net.pbtxt");

  Tensor inputs(ElemKind::FloatTy, {1, 3, 3, 1});

  Error err(Error::success());
  Caffe2ModelLoader caffe2LD(NetDescFilename, NetWeightFilename, {"inputs"},
                             {&inputs.getType()}, *F, &err);

  // Test that the error is the expected one.
  auto msg = ERR_TO_STRING(std::move(err));
  ASSERT_NE(msg.find("MaxPool nodes with legacy caffe padding are "
                     "deprecated and not supported."),
            std::string::npos);
}

/// Test loading MaxPool with default NCHW order input.
TEST_F(Caffe2ImporterTest, maxPool) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/maxpool_predict_net.pbtxt");
  std::string NetWeightFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/empty_init_net.pbtxt");

  Placeholder *output;
  PlaceholderBindings bindings;

  Tensor inputs(ElemKind::FloatTy, {1, 3, 3, 1});

  // Destroy the loader after the graph is loaded since the following execution
  // will not depend on anything from the loader.
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
TEST_F(Caffe2ImporterTest, avgPoolNHWC) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename(
      GLOW_DATA_PATH
      "tests/models/caffe2Models/avgpool_nhwc_predict_net.pbtxt");
  std::string NetWeightFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/empty_init_net.pbtxt");

  Placeholder *output;
  PlaceholderBindings bindings;

  Tensor inputs(ElemKind::FloatTy, {1, 3, 3, 1});

  // Destroy the loader after the graph is loaded since the following execution
  // will not depend on anything from the loader.
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
TEST_F(Caffe2ImporterTest, avgPool) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/avgpool_predict_net.pbtxt");
  std::string NetWeightFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/empty_init_net.pbtxt");

  Placeholder *output;
  PlaceholderBindings bindings;

  Tensor inputs(ElemKind::FloatTy, {1, 3, 3, 1});

  // Destroy the loader after the graph is loaded since the following execution
  // will not depend on anything from the loader.
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
TEST_F(Caffe2ImporterTest, concatAddAxis) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename(
      GLOW_DATA_PATH
      "tests/models/caffe2Models/concat_add_axis_predict_net.pbtxt");
  std::string NetWeightFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/empty_init_net.pbtxt");

  PlaceholderBindings bindings;

  Placeholder *output;
  Tensor inputs_0(ElemKind::FloatTy, {10, 7});
  Tensor inputs_1(ElemKind::FloatTy, {10, 7});
  Tensor inputs_2(ElemKind::FloatTy, {10, 7});
  inputs_0.getHandle().randomize(-3.0, 3.0, mod.getPRNG());
  inputs_1.getHandle().randomize(-3.0, 3.0, mod.getPRNG());
  inputs_2.getHandle().randomize(-3.0, 3.0, mod.getPRNG());
  // Destroy the loader after the graph is loaded since the following execution
  // will not depend on anything from the loader.
  {
    Caffe2ModelLoader caffe2LD(
        NetDescFilename, NetWeightFilename,
        {"inputs_0", "inputs_1", "inputs_2"},
        {&inputs_0.getType(), &inputs_1.getType(), &inputs_2.getType()}, *F);
    output = EXIT_ON_ERR(caffe2LD.getSingleOutput());

    bindings.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(bindings, &mod,
                                  {"inputs_0", "inputs_1", "inputs_2"},
                                  {&inputs_0, &inputs_1, &inputs_2});
  }

  // Check that the shape of the output matches what Caffe2 expects.
  std::vector<dim_t> expectedDims = {10, 3, 7};
  EXPECT_TRUE(output->dims().vec() == expectedDims);

  auto res = bindings.get(output);
  EE.compile(CompilationMode::Infer);

  EE.run(bindings);
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
  for (dim_t i = 0; i < 3; ++i) {
    const auto inputsHandle = inputs[i]->getHandle();
    ASSERT_TRUE(llvm::isa<Placeholder>(concat->getInputs()[i]));

    for (dim_t row = 0; row < 10; ++row) {
      for (dim_t column = 0; column < 7; ++column) {
        EXPECT_FLOAT_EQ(result.at({row, i, column}),
                        inputsHandle.at({row, column}));
      }
    }
  }
}

/// Test loading a regular concat node.
TEST_F(Caffe2ImporterTest, concat) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/concat_predict_net.pbtxt");
  std::string NetWeightFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/empty_init_net.pbtxt");

  PlaceholderBindings bindings;
  Placeholder *output;
  Tensor inputs_0(ElemKind::FloatTy, {10, 7});
  Tensor inputs_1(ElemKind::FloatTy, {10, 12});
  Tensor inputs_2(ElemKind::FloatTy, {10, 5});
  inputs_0.getHandle().randomize(-3.0, 3.0, mod.getPRNG());
  inputs_1.getHandle().randomize(-3.0, 3.0, mod.getPRNG());
  inputs_2.getHandle().randomize(-3.0, 3.0, mod.getPRNG());
  // Destroy the loader after the graph is loaded since the following execution
  // will not depend on anything from the loader.
  {
    Caffe2ModelLoader caffe2LD(
        NetDescFilename, NetWeightFilename,
        {"inputs_0", "inputs_1", "inputs_2"},
        {&inputs_0.getType(), &inputs_1.getType(), &inputs_2.getType()}, *F);
    output = EXIT_ON_ERR(caffe2LD.getSingleOutput());

    bindings.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(bindings, &mod,
                                  {"inputs_0", "inputs_1", "inputs_2"},
                                  {&inputs_0, &inputs_1, &inputs_2});
  }

  // Check that the shape of the output matches what Caffe2 expects.
  std::vector<dim_t> expectedDims = {10, 24};
  EXPECT_TRUE(output->dims().vec() == expectedDims);

  bindings.allocate(mod.getPlaceholders());
  auto res = bindings.get(output);
  EE.compile(CompilationMode::Infer);

  EE.run(bindings);
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
  dim_t columnsChecked = 0;
  for (size_t i = 0; i < 3; ++i) {
    const auto inputsHandle = inputs[i]->getHandle();
    ASSERT_TRUE(llvm::isa<Placeholder>(concat->getInputs()[i]));

    dim_t currentColumnWidth = inputs[i]->dims()[1];
    for (dim_t row = 0; row < 10; ++row) {
      for (dim_t column = 0; column < currentColumnWidth; ++column) {
        EXPECT_FLOAT_EQ(result.at({row, columnsChecked + column}),
                        inputsHandle.at({row, column}));
      }
    }
    columnsChecked += currentColumnWidth;
  }
}

/// Test loading a batched matmul with transpose on RHS.
TEST_F(Caffe2ImporterTest, batchedMatmulRHS) {
  ExecutionEngine EE{};
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
  // will not depend on anything from the loader.
  {
    Caffe2ModelLoader caffe2LD(NetDescFilename, NetWeightFilename,
                               {"inputs_0", "inputs_1"},
                               {&inputs_0.getType(), &inputs_1.getType()}, *F);
    output = EXIT_ON_ERR(caffe2LD.getSingleOutput());
  }

  // Check that the shape of the output matches what Caffe2 expects.
  std::vector<dim_t> expectedDims = {3, 10, 10};
  EXPECT_TRUE(output->dims().vec() == expectedDims);
  // High level check on the content of the graph.
  // We have 1 transpose, 1 matmul, 1 save, and 2 reshapes.
  EXPECT_EQ(F->getNodes().size(), 5);
  // With have 2 inputs and one outputs.
  EXPECT_EQ(mod.getPlaceholders().size(), 3);
  // Check that the graph has the expected shape,
  // starting from the output.
  auto *saveNode = getSaveNodeFromDest(output);
  auto *BMMN = llvm::dyn_cast<BatchMatMulNode>(saveNode->getInput().getNode());
  ASSERT_TRUE(BMMN);
  const dim_t batchMatmulDims[] = {3, 10, 10};
  EXPECT_EQ(BMMN->getResult().dims(), llvm::makeArrayRef(batchMatmulDims));
  EXPECT_TRUE(llvm::isa<Placeholder>(BMMN->getLHS()));
  auto *tileRHS = llvm::dyn_cast<TileNode>(BMMN->getRHS());
  ASSERT_TRUE(tileRHS);
  auto *reshapeRHS = llvm::dyn_cast<ReshapeNode>(tileRHS->getInput());
  ASSERT_TRUE(reshapeRHS);
  auto *transposeRHS = llvm::dyn_cast<TransposeNode>(reshapeRHS->getInput());
  ASSERT_TRUE(transposeRHS);
  EXPECT_TRUE(llvm::isa<Placeholder>(transposeRHS->getInput()));
  // Check that the last two dimensions are swapped.
  const unsigned_t shuffle[] = {1, 0};
  EXPECT_EQ(transposeRHS->getShuffle(), llvm::makeArrayRef(shuffle));
  // We don't actually check that the output is correct, because this
  // should be covered in the OperatorTest for MatMul already.
}

/// Test loading a parallel batched matmul.
TEST_F(Caffe2ImporterTest, parallelBatchedMatmulRHS) {
  ExecutionEngine EE{};
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
  // will not depend on anything from the loader.
  {
    Caffe2ModelLoader caffe2LD(NetDescFilename, NetWeightFilename,
                               {"inputs_0", "inputs_1"},
                               {&inputs_0.getType(), &inputs_1.getType()}, *F);
    output = EXIT_ON_ERR(caffe2LD.getSingleOutput());
  }

  // High level check on the content of the graph.
  // We have a BatchMatMul and a Save.
  EXPECT_EQ(F->getNodes().size(), 2);
  // With have 2 inputs and one outputs.
  EXPECT_EQ(mod.getPlaceholders().size(), 3);
  // Check that the graph has the expected shape,
  // starting from the output.
  // Parallel Batched matmul is lowered to a sequence of slices, reshapes and
  // regular matmuls.
  auto *saveNode = getSaveNodeFromDest(output);
  auto *BMMN = llvm::dyn_cast<BatchMatMulNode>(saveNode->getInput());
  ASSERT_TRUE(BMMN);

  const dim_t lhsDims[] = {3, 10, 7};
  EXPECT_EQ(BMMN->getLHS().dims(), llvm::makeArrayRef(lhsDims));
  const dim_t rhsDims[] = {3, 7, 10};
  EXPECT_EQ(BMMN->getRHS().dims(), llvm::makeArrayRef(rhsDims));
  const dim_t resultDims[] = {3, 10, 10};
  EXPECT_EQ(BMMN->getResult().dims(), llvm::makeArrayRef(resultDims));

  // We don't actually check that the output is correct, because this
  // should be covered in the OperatorTest for MatMul already.
}

/// Test loading a FC node : I * transpose(W) + B.
TEST_F(Caffe2ImporterTest, FC) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename(GLOW_DATA_PATH
                              "tests/models/caffe2Models/fc_predict_net.pbtxt");
  std::string NetWeightFilename(GLOW_DATA_PATH
                                "tests/models/caffe2Models/fc_init_net.pbtxt");

  Placeholder *output;
  PlaceholderBindings bindings;
  // Destroy the loader after the graph is loaded since the following execution
  // will not depend on anything from the loader.
  {
    Tensor inputs(ElemKind::FloatTy, {2, 3});
    inputs.getHandle() = {1, 2, 3, 4, 5, 6};

    // Weights and bias are read from NetWeightFilename. And the values are:
    // weights : {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    // bias : {0.1f, 0.2f, 0.3f, 0.4f};
    Caffe2ModelLoader caffe2LD(NetDescFilename, NetWeightFilename, {"inputs"},
                               {&inputs.getType()}, *F);
    output = EXIT_ON_ERR(caffe2LD.getSingleOutput());
    bindings.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(bindings, &mod, {"inputs"}, {&inputs});
  }

  // High level check on the content of the graph. We have 1 FC node and 1 save.
  EXPECT_EQ(F->getNodes().size(), 2);
  auto *saveNode = getSaveNodeFromDest(output);
  auto *fcNode =
      llvm::dyn_cast<FullyConnectedNode>(saveNode->getInput().getNode());
  EXPECT_TRUE(fcNode);

  // Check the numerical values of the weights and biases.
  {
    // NOTE: this is weights1 because the weights constant was transposed
    const Constant *constant = mod.getConstantByName("weights__1");
    ASSERT_TRUE(constant);
    const Tensor &weights = constant->getPayload();
    const std::vector<dim_t> expectedDimensions = {3, 4};
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
    const Constant *constant = mod.getConstantByName("bias");
    ASSERT_TRUE(constant);
    const Tensor &bias = constant->getPayload();
    const std::vector<dim_t> expectedDimensions = {4};
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
TEST_F(Caffe2ImporterTest, FCWithFlatten) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/fc_4d_predict_net.pbtxt");
  std::string NetWeightFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/fc_4d_init_net.pbtxt");

  Placeholder *output;
  PlaceholderBindings bindings;

  {
    Tensor inputs(ElemKind::FloatTy, {1, 1, 1, 2048});

    // Weights and bias are read from NetWeightFilename
    Caffe2ModelLoader caffe2LD(NetDescFilename, NetWeightFilename, {"inputs"},
                               {&inputs.getType()}, *F);
    output = EXIT_ON_ERR(caffe2LD.getSingleOutput());
    bindings.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(bindings, &mod, {"inputs"}, {&inputs});
  }

  // High level check on the content of the graph. We have a reshape, an FC,
  // another reshape, and a save.
  EXPECT_EQ(F->getNodes().size(), 4);

  auto finalShape = output->getType()->dims();
  std::vector<dim_t> expectedOutput{1, 1, 1, 9190};
  EXPECT_EQ(finalShape, llvm::makeArrayRef(expectedOutput));

  auto *saveNode = getSaveNodeFromDest(output);
  auto *reshapeAfterNode =
      llvm::dyn_cast<ReshapeNode>(saveNode->getInput().getNode());
  ASSERT_TRUE(reshapeAfterNode);
  auto *fcNode = llvm::dyn_cast<FullyConnectedNode>(
      reshapeAfterNode->getInput().getNode());
  ASSERT_TRUE(fcNode);
  auto *reshape = llvm::dyn_cast<ReshapeNode>(fcNode->getInput());
  ASSERT_TRUE(reshape);

  // We don't actually check that the output is correct, because this is
  // already covered in the Operator.FCWithFlatten/* tests.
}

/// Test loading a FCTransposed node: I * W + B
TEST_F(Caffe2ImporterTest, FCTransposed) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename(
      GLOW_DATA_PATH
      "tests/models/caffe2Models/fcTransposed_predict_net.pbtxt");
  std::string NetWeightFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/fcTransposed_init_net.pbtxt");

  Placeholder *output;
  PlaceholderBindings bindings;

  // Destroy the loader after the graph is loaded since the following execution
  // will not depend on anything from the loader.
  {
    Tensor inputs(ElemKind::FloatTy, {2, 3});
    inputs.getHandle() = {1, 2, 3, 4, 5, 6};

    // Weights and bias are read from NetWeightFilename. And the values are:
    // weights : {1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12};
    // bias : {0.1f, 0.2f, 0.3f, 0.4f};
    Caffe2ModelLoader caffe2LD(NetDescFilename, NetWeightFilename, {"inputs"},
                               {&inputs.getType()}, *F);
    output = EXIT_ON_ERR(caffe2LD.getSingleOutput());
    bindings.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(bindings, &mod, {"inputs"}, {&inputs});
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
    const std::vector<dim_t> expectedDimensions = {3, 4};
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
    const Constant *constant = mod.getConstantByName("bias");
    ASSERT_TRUE(constant);
    const Tensor &bias = constant->getPayload();
    const std::vector<dim_t> expectedDimensions = {4};
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
TEST_F(Caffe2ImporterTest, FCTransposedWithFlatten) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename(
      GLOW_DATA_PATH
      "tests/models/caffe2Models/fcTransposed_4d_predict_net.pbtxt");
  std::string NetWeightFilename(
      GLOW_DATA_PATH
      "tests/models/caffe2Models/fcTransposed_4d_init_net.pbtxt");

  Placeholder *output;
  PlaceholderBindings bindings;

  {
    Tensor inputs(ElemKind::FloatTy, {1, 1, 1, 2048});

    // Weights and bias are read from NetWeightFilename.
    Caffe2ModelLoader caffe2LD(NetDescFilename, NetWeightFilename, {"inputs"},
                               {&inputs.getType()}, *F);
    output = EXIT_ON_ERR(caffe2LD.getSingleOutput());
    bindings.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(bindings, &mod, {"inputs"}, {&inputs});
  }

  // High level check on the content of the graph. We have a reshape, an FC,
  // another reshape, and a save.
  EXPECT_EQ(F->getNodes().size(), 4);

  auto finalShape = output->getType()->dims();
  std::vector<dim_t> expectedOutput{1, 1, 1, 9190};
  EXPECT_EQ(finalShape, llvm::makeArrayRef(expectedOutput));

  auto *saveNode = getSaveNodeFromDest(output);
  auto *reshapeAfterNode =
      llvm::dyn_cast<ReshapeNode>(saveNode->getInput().getNode());
  ASSERT_TRUE(reshapeAfterNode);
  auto *fcNode = llvm::dyn_cast<FullyConnectedNode>(
      reshapeAfterNode->getInput().getNode());
  ASSERT_TRUE(fcNode);
  auto *reshape = llvm::dyn_cast<ReshapeNode>(fcNode->getInput());
  ASSERT_TRUE(reshape);

  // We don't actually check that the output is correct, because this is
  // already covered in the Operator.FCWithFlatten/* tests.
}

/// Test loading bucketize op from a Caffe2 model.
/// Test with arg boundaries = [0.1, 2.5]
TEST_F(Caffe2ImporterTest, importBucketize) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/bucketize_op_net.pbtxt");
  std::string NetWeightFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/empty_init_net.pbtxt");

  PlaceholderBindings bindings;
  Placeholder *output;
  Tensor inputs_0(ElemKind::FloatTy, {3, 2});
  // Destroy the loader after the graph is loaded since the following execution
  // will not depend on anything from the loader.
  {
    Caffe2ModelLoader caffe2LD(NetDescFilename, NetWeightFilename, {"input_0"},
                               {&inputs_0.getType()}, *F);
    output = EXIT_ON_ERR(caffe2LD.getSingleOutput());
    bindings.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(bindings, &mod, {"input_0"}, {&inputs_0});
  }

  EXPECT_EQ(F->getNodes().size(), 2);
  auto *saveNode = getSaveNodeFromDest(output);
  auto *bucketizeNode =
      llvm::dyn_cast<BucketizeNode>(saveNode->getInput().getNode());
  ASSERT_TRUE(bucketizeNode);
  auto boundriesVec = bucketizeNode->getBoundaries();
  ASSERT_EQ(boundriesVec.size(), 2);
  EXPECT_NEAR(boundriesVec[0], 0.1, 0.00001);
  EXPECT_NEAR(boundriesVec[1], 2.5, 0.00001);
  // We have one input and one output.
  EXPECT_EQ(mod.getPlaceholders().size(), 2);
}

/// Test loading ResizeNearest op from a Caffe2 model.
/// Test with NHWC order, 2.0 height scale and 1.5 width scale
TEST_F(Caffe2ImporterTest, importResizeNearest) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/resize_nearest_op_net.pbtxt");
  std::string NetWeightFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/empty_init_net.pbtxt");

  Placeholder *output;
  PlaceholderBindings bindings;

  {
    Tensor input(ElemKind::FloatTy, {1, 2, 2, 1});

    Caffe2ModelLoader caffe2LD(NetDescFilename, NetWeightFilename, {"input_0"},
                               {&input.getType()}, *F);
    output = EXIT_ON_ERR(caffe2LD.getSingleOutput());
    bindings.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(bindings, &mod, {"input_0"}, {&input});
  }

  EXPECT_EQ(F->getNodes().size(), 2);
  auto *saveNode = getSaveNodeFromDest(output);
  auto *resizeNearestNode =
      llvm::dyn_cast<ResizeNearestNode>(saveNode->getInput().getNode());
  ASSERT_TRUE(resizeNearestNode);
  // We have one input and one output.
  EXPECT_EQ(mod.getPlaceholders().size(), 2);
  auto heightScale = resizeNearestNode->getHeightScale();
  auto widthScale = resizeNearestNode->getWidthScale();
  EXPECT_NEAR(heightScale, 2.0, 0.00001);
  EXPECT_NEAR(widthScale, 1.5, 0.00001);
}

/// Test loading clip op from a Caffe2 model.
/// Test with arg min = 20.0 max = 60.0
TEST_F(Caffe2ImporterTest, importClip) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename(GLOW_DATA_PATH
                              "tests/models/caffe2Models/clip_op_net.pbtxt");
  std::string NetWeightFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/empty_init_net.pbtxt");

  PlaceholderBindings bindings;
  Placeholder *output;
  Tensor inputs_0(ElemKind::FloatTy, {5, 5});
  // Destroy the loader after the graph is loaded since the following execution
  // will not depend on anything from the loader.
  {
    Caffe2ModelLoader caffe2LD(NetDescFilename, NetWeightFilename, {"inputs_0"},
                               {&inputs_0.getType()}, *F);
    output = EXIT_ON_ERR(caffe2LD.getSingleOutput());
    bindings.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(bindings, &mod, {"inputs_0"}, {&inputs_0});
  }

  EXPECT_EQ(F->getNodes().size(), 2);
  auto *saveNode = getSaveNodeFromDest(output);
  auto *clipNode = llvm::dyn_cast<ClipNode>(saveNode->getInput().getNode());
  ASSERT_TRUE(clipNode);
  EXPECT_EQ(clipNode->getMax(), 60.0);
  EXPECT_EQ(clipNode->getMin(), 20.0);
  auto *inputNode = llvm::dyn_cast<Placeholder>(clipNode->getInput());
  ASSERT_EQ(inputNode, mod.getPlaceholderByName("inputs_0"));
  // We have one input and one output.
  EXPECT_EQ(mod.getPlaceholders().size(), 2);
}

/// Test loading clip op from a Caffe2 model with default arg values:
/// min = std::numeric_limits<float>::lowest()
/// max = std::numeric_limits<float>::max()
TEST_F(Caffe2ImporterTest, importClipDefault) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/clip_op_default_net.pbtxt");
  std::string NetWeightFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/empty_init_net.pbtxt");

  PlaceholderBindings bindings;
  Placeholder *output;
  Tensor inputs_0(ElemKind::FloatTy, {5, 5});

  // Destroy the loader after the graph is loaded since the following execution
  // will not depend on anything from the loader.
  {
    Caffe2ModelLoader caffe2LD(NetDescFilename, NetWeightFilename, {"inputs_0"},
                               {&inputs_0.getType()}, *F);
    output = EXIT_ON_ERR(caffe2LD.getSingleOutput());
    bindings.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(bindings, &mod, {"inputs_0"}, {&inputs_0});
  }
  EXPECT_EQ(F->getNodes().size(), 2);
  auto *saveNode = getSaveNodeFromDest(output);
  auto *clipNode = llvm::dyn_cast<ClipNode>(saveNode->getInput().getNode());
  EXPECT_EQ(clipNode->getMax(), std::numeric_limits<float>::max());
  EXPECT_EQ(clipNode->getMin(), std::numeric_limits<float>::lowest());
  auto *inputNode = llvm::dyn_cast<Placeholder>(clipNode->getInput().getNode());
  ASSERT_EQ(inputNode, mod.getPlaceholderByName("inputs_0"));
  // We have one input and one output.
  EXPECT_EQ(mod.getPlaceholders().size(), 2);
}

/// Test loading a ReplaceNaN operator.
TEST_F(Caffe2ImporterTest, replaceNaN) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/replace_nan_predict_net.pbtxt");
  std::string NetWeightFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/empty_init_net.pbtxt");

  PlaceholderBindings bindings;
  Placeholder *output;
  Tensor input(ElemKind::FloatTy, {10, 10});

  // Destroy the loader after the graph is loaded since the following execution
  // will not depend on anything from the loader.
  {
    Caffe2ModelLoader caffe2LD(NetDescFilename, NetWeightFilename, {"input"},
                               {&input.getType()}, *F);
    output = EXIT_ON_ERR(caffe2LD.getSingleOutput());
    bindings.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(bindings, &mod, {"input"}, {&input});
  }

  // Check that the shape of the output matches the input.
  std::vector<dim_t> expectedDims = {10, 10};
  EXPECT_TRUE(output->dims().vec() == expectedDims);

  // High level checks on the content of the graph.
  // We have 1 ReplaceNaN and 1 Output.
  EXPECT_EQ(F->getNodes().size(), 2);
  auto *saveNode = getSaveNodeFromDest(output);
  auto *replaceNaNNode =
      llvm::dyn_cast<ReplaceNaNNode>(saveNode->getInput().getNode());
  EXPECT_EQ(replaceNaNNode->getValue(), 1.0f);
  auto *inputNode =
      llvm::dyn_cast<Placeholder>(replaceNaNNode->getInput().getNode());
  ASSERT_EQ(inputNode, mod.getPlaceholderByName("input"));

  // We have one input and one output.
  EXPECT_EQ(mod.getPlaceholders().size(), 2);
}

/// Test loading a DotProduct operator with 1D inputs.
TEST_F(Caffe2ImporterTest, dotProduct1D) {
  ExecutionEngine EE{};
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
TEST_F(Caffe2ImporterTest, dotProduct2D) {
  ExecutionEngine EE{};
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
TEST_F(Caffe2ImporterTest, batchBoxCox) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename(
      GLOW_DATA_PATH
      "tests/models/caffe2Models/batch_box_cox_predict_net.pbtxt");
  std::string NetWeightFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/empty_init_net.pbtxt");

  PlaceholderBindings bindings;
  Placeholder *output;

  // Input tensors.
  const size_t kRows = 10;
  const size_t kCols = 5;
  Tensor data(ElemKind::FloatTy, {kRows, kCols});
  Tensor lambda1(ElemKind::FloatTy, {kCols});
  Tensor lambda2(ElemKind::FloatTy, {kCols});
  Tensor O(ElemKind::FloatTy, {kRows, kCols});
  // Destroy the loader after the graph is loaded since the following execution
  // will not depend on anything from the loader.
  {
    Caffe2ModelLoader caffe2LD(
        NetDescFilename, NetWeightFilename, {"data", "lambda1", "lambda2"},
        {&data.getType(), &lambda1.getType(), &lambda2.getType()}, *F);
    output = EXIT_ON_ERR(caffe2LD.getSingleOutput());
    bindings.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(bindings, &mod,
                                  {"data", "lambda1", "lambda2"},
                                  {&data, &lambda1, &lambda2});
  }

  EXPECT_EQ(F->getNodes().size(), 2);

  // Output.
  auto *saveNode = getSaveNodeFromDest(output);
  ASSERT_TRUE(saveNode);

  // Select.
  auto *BBCN = llvm::dyn_cast<BatchBoxCoxNode>(saveNode->getInput());
  ASSERT_TRUE(BBCN);

  // There are three inputs and one output.
  EXPECT_EQ(mod.getPlaceholders().size(), 4);
}

// Test loading a EQ operator with 1D inputs.
TEST_F(Caffe2ImporterTest, EQ1D) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename(GLOW_DATA_PATH
                              "tests/models/caffe2Models/eq_op_net.pbtxt");
  std::string NetWeightFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/empty_init_net.pbtxt");

  Placeholder *output;
  PlaceholderBindings bindings;

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
TEST_F(Caffe2ImporterTest, LengthsToRanges) {
  ExecutionEngine EE{};
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
TEST_F(Caffe2ImporterTest, Logit) {
  ExecutionEngine EE{};
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
  std::vector<dim_t> expectedDims = {kDataSize};
  EXPECT_EQ(output->dims().vec(), expectedDims);

  // High level checks on the content of the graph.
  // We have 1 Clip (1 Splat, 1 Max, 1 Splat, 1 Min),
  // 1 Splat, 1 Sub, 1 Div, 1 Log, and 1 Output.
  EXPECT_EQ(F->getNodes().size(), 6);

  // Graph has one input and one output.
  EXPECT_EQ(mod.getPlaceholders().size(), 2);
}

// Test loading a SparseToDense operator.
TEST_F(Caffe2ImporterTest, sparseToDense) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/sparse_to_dense.pbtxt");
  std::string NetWeightFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/empty_init_net.pbtxt");

  Placeholder *output;
  PlaceholderBindings bindings;

  // Create inputs.
  constexpr size_t kNumIndices = 5;
  constexpr size_t kMaxIndex = 20;
  constexpr size_t kRows = 10;
  constexpr size_t kCols = 5;
  Tensor indices(IndexElemKind, {kNumIndices});
  Tensor values(ElemKind::FloatTy, {kNumIndices, kRows, kCols});
  Tensor dataToInferDim(ElemKind::FloatTy, {kMaxIndex, kRows, kCols});

  // Destroy the loader after the graph is loaded since the following execution
  // will not depend on anything from the loader.
  {
    Caffe2ModelLoader caffe2LD(
        NetDescFilename, NetWeightFilename,
        {"indices", "values", "dataToInferDim"},
        {&indices.getType(), &values.getType(), &dataToInferDim.getType()}, *F);
    output = EXIT_ON_ERR(caffe2LD.getSingleOutput());
    bindings.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(bindings, &mod, {"indices", "values"},
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

TEST_F(Caffe2ImporterTest, SparseToDenseMask) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename(
      GLOW_DATA_PATH
      "tests/models/caffe2Models/sparse_to_dense_mask_op_net.pbtxt");
  std::string NetWeightFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/empty_init_net.pbtxt");

  Placeholder *output;
  PlaceholderBindings bindings;

  Tensor indices(IndexElemKind, {4});
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
  EXPECT_TRUE(N->getResult().dims().equals({6, 10, 20, 30}));
  // Check that mask was read correctly.
  EXPECT_TRUE(N->getMask().equals({42, 100, 300, 1, 0, 312}));
}

/// Test loading NCHW2NHWC op.
TEST_F(Caffe2ImporterTest, testNCHW2NHWC) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/NCHW2NHWC_predict_net.pbtxt");
  std::string NetWeightFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/empty_init_net.pbtxt");

  Placeholder *output;
  PlaceholderBindings bindings;

  Tensor inputs(ElemKind::FloatTy, {1, 2, 3, 4});

  // Destroy the loader after the graph is loaded since the following execution
  // will not depend on anything from the loader.
  {
    Caffe2ModelLoader caffe2LD(NetDescFilename, NetWeightFilename, {"inputs"},
                               {&inputs.getType()}, *F);
    output = EXIT_ON_ERR(caffe2LD.getSingleOutput());
    bindings.allocate(mod.getPlaceholders());
  }

  // Check output shape.
  auto res = bindings.get(output);
  std::vector<dim_t> expectedDims = {1, 3, 4, 2};
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
TEST_F(Caffe2ImporterTest, lengthsSum) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename(GLOW_DATA_PATH
                              "tests/models/caffe2Models/lengths_sum.pbtxt");
  std::string NetWeightFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/empty_init_net.pbtxt");

  Placeholder *output;
  PlaceholderBindings bindings;

  // Create inputs.
  Tensor data(ElemKind::Int64ITy, {10, 2, 3});
  Tensor lengths(ElemKind::FloatTy, {5});

  // Destroy the loader after the graph is loaded since the following execution
  // will not depend on anything from the loader.
  {
    Caffe2ModelLoader caffe2LD(NetDescFilename, NetWeightFilename,
                               {"data", "lengths"},
                               {&data.getType(), &lengths.getType()}, *F);
    output = EXIT_ON_ERR(caffe2LD.getSingleOutput());
  }

  // Check that the shape of the output matches that of the expected output.
  std::vector<dim_t> expectedShape{5, 2, 3};
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
TEST_F(Caffe2ImporterTest, gatherRanges) {
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

/// Test loading Gather ops with constant folding from an Caffe2 model.
TEST_F(Caffe2ImporterTest, gatherConstantFoldingAndReshape) {
  // This test verifies that Gather gets constant-folded, so that the argument
  // of the reshape becomes constant.
  ExecutionEngine EE;
  auto &mod = EE.getModule();

  std::string netDescFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/gather_const_fold.pbtxt");
  std::string netWeightFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/gather_const_fold_init.pbtxt");
  PlaceholderBindings bindings;
  auto *F = mod.createFunction("main");
  Placeholder *output;
  Tensor data(ElemKind::FloatTy, {1, 2, 4, 3});
  // This test is testing constant folding during loading, so enable it
  // explicitly.
  setConstantFoldLoaderOpsFlag(true);
  {
    Caffe2ModelLoader caffe2LD(netDescFilename, netWeightFilename, {"data"},
                               {&data.getType()}, *F);
    output = EXIT_ON_ERR(caffe2LD.getOutputByName("result"));
    bindings.allocate(mod.getPlaceholders());
  }
  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  auto result = bindings.get(output)->getHandle();
  std::vector<dim_t> expectedDims = {1, 4, 3, 2};
  EXPECT_TRUE(result.dims().vec() == expectedDims);
}
/// Test loading a LengthsRangeFill op.
TEST_F(Caffe2ImporterTest, LengthsRangeFill) {
  ExecutionEngine EE;
  auto &mod = EE.getModule();
  auto *F = mod.createFunction("main");

  std::string NetDescFilename(
      GLOW_DATA_PATH
      "tests/models/caffe2Models/lengths_range_fill_predict_net.pbtxt");
  std::string NetWeightFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/empty_init_net.pbtxt");

  Placeholder *output;
  Tensor lengths(ElemKind::Int32ITy, {3});

  {
    Caffe2ModelLoader caffe2LD(NetDescFilename, NetWeightFilename, {"lengths"},
                               {&lengths.getType()}, *F);
    output = EXIT_ON_ERR(caffe2LD.getOutputByName("result"));
  }

  // Verify structure: PH -> LengthsRangeFill -> Save -> PH.
  ASSERT_EQ(mod.getPlaceholders().size(), 2);
  ASSERT_EQ(F->getNodes().size(), 2);
  auto *save = getSaveNodeFromDest(output);
  auto *LRF = llvm::dyn_cast<LengthsRangeFillNode>(save->getInput().getNode());
  ASSERT_TRUE(LRF);
  EXPECT_TRUE(LRF->getLengths().dims().equals({3}));
  EXPECT_EQ(LRF->getResult().dims().size(), 1);
  // Proto specifies the max output size is 8.
  EXPECT_TRUE(LRF->getResult().dims().equals({8}));
}

/// Verify that different fill types are loaded with the correct types.
TEST_F(Caffe2ImporterTest, tensorFillsTest) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/fill_test_predict_net.pbtxt");
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
    Type unusedTy = Type(ElemKind::FloatTy, {4});
    Caffe2ModelLoader caffe2LD(
        NetDescFilename, NetWeightFilename,
        {"tensor_fill_float_eq", "tensor_int_fill_eq", "tensor_int64_fill_eq",
         "tensor_string_to_uint8_fill_eq"},
        {&unusedTy, &unusedTy, &unusedTy, &unusedTy}, *F);
    tensorFillFloat = llvm::dyn_cast<Constant>(
        EXIT_ON_ERR(caffe2LD.getNodeValueByName("tensor_fill_float")));
    tensorIntFill = llvm::dyn_cast<Constant>(
        EXIT_ON_ERR(caffe2LD.getNodeValueByName("tensor_int_fill")));
    tensorInt64Fill = llvm::dyn_cast<Constant>(
        EXIT_ON_ERR(caffe2LD.getNodeValueByName("tensor_int64_fill")));
    tensorStringToUInt8Fill = llvm::dyn_cast<Constant>(EXIT_ON_ERR(
        caffe2LD.getNodeValueByName("tensor_string_to_uint8_fill")));
  }

  ASSERT_TRUE(tensorFillFloat);
  ASSERT_TRUE(tensorIntFill);
  ASSERT_TRUE(tensorInt64Fill);
  ASSERT_TRUE(tensorStringToUInt8Fill);

  // All fills in fill_test_init_net.pbtxt use shape {2, 2}.
  const std::vector<dim_t> expectedDims = {2, 2};
  ASSERT_TRUE(tensorFillFloat->dims().equals(expectedDims));
  ASSERT_TRUE(tensorIntFill->dims().equals(expectedDims));
  ASSERT_TRUE(tensorInt64Fill->dims().equals(expectedDims));
  ASSERT_TRUE(tensorStringToUInt8Fill->dims().equals(expectedDims));

  auto tensorFillFloatH = tensorFillFloat->getPayload().getHandle<float>();
  auto tensorIntFillH = tensorIntFill->getPayload().getHandle<int32_t>();
  auto tensorInt64FillH = tensorInt64Fill->getPayload().getHandle<int64_t>();
  // We load GivenTensorByteStringToUInt8Fill as UInt8QTy with dummy
  // scale/offset for now, because it's only used for rowwise-quantized tensors.
  auto tensorStringToUInt8FillH =
      tensorStringToUInt8Fill->getPayload().getHandle<uint8_t>();

  // All fills in fill_test_init_net.pbtxt are set to 0 through 3.
  for (size_t i = 0; i < 4; i++) {
    EXPECT_FLOAT_EQ(tensorFillFloatH.raw(i), (float)i);
    EXPECT_EQ(tensorIntFillH.raw(i), (int32_t)i);
    EXPECT_EQ(tensorInt64FillH.raw(i), (int64_t)i);
    EXPECT_EQ(tensorStringToUInt8FillH.raw(i), (uint8_t)(i + 128));
  }
}

TEST_F(Caffe2ImporterTest, HalfToFloat) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  llvm::StringRef NetDescFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/halftofloat_op_net.pbtxt");
  std::string NetWeightFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/empty_init_net.pbtxt");

  Placeholder *output;
  PlaceholderBindings bindings;

  Tensor input(ElemKind::Float16Ty, {1, 2, 3, 4});

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

  // Graph has 2 nodes: Save and ConvertTo
  EXPECT_EQ(F->getNodes().size(), 2);

  // Input to save node is ConvertToNode.
  auto *saveNode = getSaveNodeFromDest(output);
  auto *N = llvm::dyn_cast<ConvertToNode>(saveNode->getInput());
  EXPECT_TRUE(N);
  EXPECT_EQ(N->getResult().getElementType(), ElemKind::FloatTy);
}

TEST_F(Caffe2ImporterTest, Alias) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  llvm::StringRef NetDescFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/alias_op_net.pbtxt");
  std::string NetWeightFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/empty_init_net.pbtxt");

  Placeholder *output;
  PlaceholderBindings bindings;

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

TEST_F(Caffe2ImporterTest, Modulo) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename(GLOW_DATA_PATH
                              "tests/models/caffe2Models/modulo_op_net.pbtxt");
  std::string NetWeightFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/fill_test_init_net.pbtxt");

  Placeholder *output;
  PlaceholderBindings bindings;

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
TEST_F(Caffe2ImporterTest, elementwiseLinear) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/elementwise_linear_net.pbtxt");
  std::string NetWeightFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/empty_init_net.pbtxt");

  PlaceholderBindings bindings;
  Placeholder *output;
  Tensor X(ElemKind::FloatTy, {10, 5});
  Tensor w(ElemKind::FloatTy, {10}), b(ElemKind::FloatTy, {10});

  // Destroy the loader after the graph is loaded since the following execution
  // will not depend on anything from the loader.
  {
    Caffe2ModelLoader caffe2LD(NetDescFilename, NetWeightFilename,
                               {"X", "w", "b"},
                               {&X.getType(), &w.getType(), &b.getType()}, *F);
    output = EXIT_ON_ERR(caffe2LD.getSingleOutput());
  }

  // Check that the shape of the output matches that of the input.
  std::vector<dim_t> expectedDims = {10, 5};
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
TEST_F(Caffe2ImporterTest, elementwiseLinearUnspecifiedAxis) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename(
      GLOW_DATA_PATH
      "tests/models/caffe2Models/elementwise_linear_default_net.pbtxt");
  std::string NetWeightFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/empty_init_net.pbtxt");

  PlaceholderBindings bindings;
  Placeholder *output;

  // Since the loader will assume that axis = 1, the 0th dim of the shapes of w
  // and b must match the 1st dim of X.
  Tensor X(ElemKind::FloatTy, {5, 10});
  Tensor w(ElemKind::FloatTy, {10}), b(ElemKind::FloatTy, {10});

  // Destroy the loader after the graph is loaded since the following execution
  // will not depend on anything from the loader.
  {
    Caffe2ModelLoader caffe2LD(NetDescFilename, NetWeightFilename,
                               {"X", "w", "b"},
                               {&X.getType(), &w.getType(), &b.getType()}, *F);
    output = EXIT_ON_ERR(caffe2LD.getSingleOutput());
  }

  // Check that the shape of the output matches that of the input.
  std::vector<dim_t> expectedDims = {5, 10};
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
TEST_F(Caffe2ImporterTest, SparseLengthsWeightedSum8BitsRowwise) {
  ExecutionEngine EE{};
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
  PlaceholderBindings bindings;

  TypeRef indicesType = F->getParent()->uniqueType(IndexElemKind, {8});
  TypeRef lengthsType = F->getParent()->uniqueType(ElemKind::Int32ITy, {4});

  // Destroy the loader after the graph is loaded since the following execution
  // will not depend on anything from the loader.
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

  bindings.allocate(indices)->getHandle<sdim_t>() = {
      1, 0, 2, 0, 1, 2, 2, 0,
  };
  bindings.allocate(lengths)->getHandle<int32_t>() = {
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

  // We have 4 constants: data, scales, offsets, and weights. Originally fused
  // data is no longer used and is removed by loader.
  EXPECT_EQ(mod.getConstants().size(), 4);

  EE.compile(CompilationMode::Infer);
  bindings.allocate(mod.getPlaceholders());

  // Post compile, DCE should have gotten rid of the originally fused data
  // Constant, as it is no longer used.
  EXPECT_EQ(mod.getConstants().size(), 4);

  EE.run(bindings);

  Tensor &result = *bindings.get(output);
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
TEST_F(Caffe2ImporterTest, SparseLengthsSum8BitsRowwise) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/"
                     "rowwise_quantized_sparse_lengths_sum_predict_net.pbtxt");
  std::string NetWeightFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/"
                     "rowwise_quantized_sparse_lengths_sum_init_net.pbtxt");

  Placeholder *output, *indices, *lengths;
  PlaceholderBindings bindings;

  TypeRef indicesType = F->getParent()->uniqueType(IndexElemKind, {8});
  TypeRef lengthsType = F->getParent()->uniqueType(ElemKind::Int32ITy, {5});

  // Destroy the loader after the graph is loaded since the following execution
  // will not depend on anything from the loader.
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

  bindings.allocate(indices)->getHandle<sdim_t>() = {
      2, 0, 1, 2, 0, 0, 0, 0,
  };
  bindings.allocate(lengths)->getHandle<int32_t>() = {
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

  // We have 5 constants: Data, scales, and offsets. Originally fused data is no
  // longer used and is removed by loader.
  EXPECT_EQ(mod.getConstants().size(), 3);

  EE.compile(CompilationMode::Infer);
  bindings.allocate(mod.getPlaceholders());

  // Post compile, DCE should have gotten rid of the originally fused data
  // Constant, as it is no longer used.
  EXPECT_EQ(mod.getConstants().size(), 3);

  EE.run(bindings);

  Tensor &result = *bindings.get(output);
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
TEST_F(Caffe2ImporterTest, SparseLengthsWeightedSumFused8BitRowwise) {
  ExecutionEngine EE{};
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
  PlaceholderBindings bindings;

  TypeRef indicesType = F->getParent()->uniqueType(IndexElemKind, {8});
  TypeRef lengthsType = F->getParent()->uniqueType(ElemKind::Int32ITy, {4});

  // Destroy the loader after the graph is loaded since the following execution
  // will not depend on anything from the loader.
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

  bindings.allocate(indices)->getHandle<sdim_t>() = {
      1, 0, 2, 0, 1, 2, 2, 0,
  };
  bindings.allocate(lengths)->getHandle<int32_t>() = {
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

  EE.compile(CompilationMode::Infer);
  bindings.allocate(mod.getPlaceholders());

  EE.run(bindings);

  Tensor &result = *bindings.get(output);
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
TEST_F(Caffe2ImporterTest, SparseLengthsSumFused8BitRowwise) {
  ExecutionEngine EE{};
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
  PlaceholderBindings bindings;

  TypeRef indicesType = F->getParent()->uniqueType(IndexElemKind, {8});
  TypeRef lengthsType = F->getParent()->uniqueType(ElemKind::Int32ITy, {5});

  // Destroy the loader after the graph is loaded since the following execution
  // will not depend on anything from the loader.
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

  bindings.allocate(indices)->getHandle<sdim_t>() = {
      2, 0, 1, 2, 0, 0, 0, 0,
  };
  bindings.allocate(lengths)->getHandle<int32_t>() = {
      2, 0, 2, 1, 3,
  };

  // High level check on the content of the graph. We have 1 rowwise-quantized
  // SLS and 1 save.
  EXPECT_EQ(F->getNodes().size(), 2);
  SaveNode *saveNode = getSaveNodeFromDest(output);
  FusedRowwiseQuantizedSparseLengthsSumNode *FRWQSLS =
      llvm::dyn_cast<FusedRowwiseQuantizedSparseLengthsSumNode>(
          saveNode->getInput().getNode());
  ASSERT_TRUE(FRWQSLS);
  // Check that the data input is a Constant node with expected ElemKind.
  Constant *data = llvm::dyn_cast<Constant>(FRWQSLS->getData().getNode());
  ASSERT_TRUE(data);
  EXPECT_TRUE(data->getElementType() == ElemKind::UInt8FusedQTy);

  // We have 3 placeholders: 1 for save, and then indices and lengths.
  EXPECT_EQ(mod.getPlaceholders().size(), 3);

  // We have 1 constant: data.
  EXPECT_EQ(mod.getConstants().size(), 1);

  EE.compile(CompilationMode::Infer);
  bindings.allocate(mod.getPlaceholders());

  EE.run(bindings);

  Tensor &result = *bindings.get(output);
  Tensor expected(ElemKind::FloatTy, {5, 2});
  expected.getHandle() = {
      5.5f, 6.9f, 0.0f, 0.0f, 6.8f, 9.1f, 1.0f, 1.2f, 3.0f, 3.6f,
  };

  EXPECT_TRUE(expected.isEqual(result, 0.02f));
}

/// Test loading SparseLengthsSumFused4BitRowwise.
TEST_F(Caffe2ImporterTest, SparseLengthsSumFused4BitRowwise) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename(
      GLOW_DATA_PATH
      "tests/models/caffe2Models/"
      "4bit_fused_rowwise_quantized_sparse_lengths_sum_predict_net.pbtxt");
  std::string NetWeightFilename(
      GLOW_DATA_PATH
      "tests/models/caffe2Models/"
      "4bit_fused_rowwise_quantized_sparse_lengths_sum_init_net.pbtxt");

  Placeholder *output, *indices, *lengths;
  PlaceholderBindings bindings;

  TypeRef indicesType = F->getParent()->uniqueType(IndexElemKind, {8});
  TypeRef lengthsType = F->getParent()->uniqueType(ElemKind::Int32ITy, {5});

  // Destroy the loader after the graph is loaded since the following execution
  // will not depend on anything from the loader.
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

  // High level check on the content of the graph. We have 1 rowwise-quantized
  // SLS, 1 convertTo and 1 save.
  EXPECT_EQ(F->getNodes().size(), 3);
  SaveNode *saveNode = getSaveNodeFromDest(output);
  ConvertToNode *C =
      llvm::dyn_cast<ConvertToNode>(saveNode->getInput().getNode());
  ASSERT_TRUE(C);
  FusedRowwiseQuantizedSparseLengthsSumNode *FRWQSLS =
      llvm::dyn_cast<FusedRowwiseQuantizedSparseLengthsSumNode>(
          C->getInput().getNode());
  ASSERT_TRUE(FRWQSLS);
  // Check that the data input is a Constant node with expected ElemKind.
  Constant *data = llvm::dyn_cast<Constant>(FRWQSLS->getData().getNode());
  ASSERT_TRUE(data);
  EXPECT_TRUE(data->getElementType() == ElemKind::UInt4FusedFP16QTy);

  // Check the output dim
  const auto out_node = saveNode->getOutput();
  EXPECT_EQ(out_node.getElementType(), ElemKind::FloatTy);
  const auto dims = out_node.dims();
  EXPECT_EQ(dims.size(), 2);
  EXPECT_EQ(dims[0], 5);
  EXPECT_EQ(dims[1], 10);

  // We have 3 placeholders: 1 for save, and then indices and lengths.
  EXPECT_EQ(mod.getPlaceholders().size(), 3);

  // We have 1 constant: data.
  EXPECT_EQ(mod.getConstants().size(), 1);
}

/// Load big enough model and validate node order.
TEST_F(Caffe2ImporterTest, validateNodeOrder) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  std::string NetDescFilename(
      GLOW_DATA_PATH
      "tests/models/caffe2Models/batch_box_cox_predict_net.pbtxt");
  std::string NetWeightFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/empty_init_net.pbtxt");

  PlaceholderBindings bindings;

  // Input tensors.
  const size_t kRows = 10;
  const size_t kCols = 5;
  Tensor data(ElemKind::FloatTy, {kRows, kCols});
  Tensor lambda1(ElemKind::FloatTy, {kCols});
  Tensor lambda2(ElemKind::FloatTy, {kCols});
  Tensor O(ElemKind::FloatTy, {kRows, kCols});
  // Destroy the loader after the graph is loaded since the following execution
  // will not depend on anything from the loader.
  {
    Caffe2ModelLoader caffe2LD(
        NetDescFilename, NetWeightFilename, {"data", "lambda1", "lambda2"},
        {&data.getType(), &lambda1.getType(), &lambda2.getType()}, *F);
    bindings.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(bindings, &mod,
                                  {"data", "lambda1", "lambda2"},
                                  {&data, &lambda1, &lambda2});
  }

  EXPECT_EQ(F->getNodes().size(), 2);
  // Make sure that nodes are sorted by name.
  EXPECT_TRUE(std::is_sorted(
      F->getNodes().begin(), F->getNodes().end(),
      [](const Node &a, const Node &b) { return a.getName() < b.getName(); }));
}

TEST_F(Caffe2ImporterTest, importInt8ConvRelu) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/int8convrelu_pred_net.pbtxt");
  std::string NetWeightFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/int8convrelu_init_net.pbtxt");

  Placeholder *output;
  PlaceholderBindings bindings;

  // Destroy the loader after the graph is loaded since the following execution
  // will not depend on anything from the loader.
  {
    Tensor data(ElemKind::Int8QTy, {1, 1, 3, 3}, 1, 0);
    Caffe2ModelLoader caffe2LD(NetDescFilename, NetWeightFilename,
                               {"gpu_0/data_0"}, {&data.getType()}, *F);
    output = EXIT_ON_ERR(caffe2LD.getSingleOutput());

    bindings.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(bindings, &mod, {"gpu_0/data_0"}, {&data});
  }

  // High level check on the content of the graph. We should have
  // transpose => conv => relu => transpose => save
  EXPECT_EQ(F->getNodes().size(), 5);
  auto *saveNode = getSaveNodeFromDest(output);

  auto *transNode1 =
      llvm::dyn_cast<TransposeNode>(saveNode->getInput().getNode());
  ASSERT_TRUE(transNode1);
  auto *reluNode = llvm::dyn_cast<ReluNode>(transNode1->getInput().getNode());
  ASSERT_TRUE(reluNode);
  auto *convNode =
      llvm::dyn_cast<ConvolutionNode>(reluNode->getInput().getNode());
  ASSERT_TRUE(convNode);
  auto *transNode2 =
      llvm::dyn_cast<TransposeNode>(convNode->getInput().getNode());
  ASSERT_TRUE(transNode2);

  EE.compile(CompilationMode::Infer);
}

TEST_F(Caffe2ImporterTest, importInt8SumRelu) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/int8sumrelu_pred_net.pbtxt");
  std::string NetWeightFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/int8sumrelu_init_net.pbtxt");

  Placeholder *output;
  PlaceholderBindings bindings;

  // Destroy the loader after the graph is loaded since the following execution
  // will not depend on anything from the loader.
  {
    Tensor data(ElemKind::Int8QTy, {4, 2}, 1, 0);
    Caffe2ModelLoader caffe2LD(NetDescFilename, NetWeightFilename,
                               {"gpu_0/data_0"}, {&data.getType()}, *F);
    output = EXIT_ON_ERR(caffe2LD.getSingleOutput());

    bindings.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(bindings, &mod, {"gpu_0/data_0"}, {&data});
  }

  // High level check on the content of the graph. We should have
  // input-=> add => relu => save
  // const/
  EXPECT_EQ(F->getNodes().size(), 3);
  auto *save = getSaveNodeFromDest(output);

  auto *relu = llvm::dyn_cast<ReluNode>(save->getInput().getNode());
  ASSERT_TRUE(relu);
  auto *add = llvm::dyn_cast<AddNode>(relu->getInput().getNode());
  ASSERT_TRUE(add);
  auto *input = llvm::dyn_cast<Placeholder>(add->getLHS().getNode());
  ASSERT_TRUE(input);
  auto *val = llvm::dyn_cast<Constant>(add->getRHS().getNode());
  ASSERT_TRUE(val);

  EE.compile(CompilationMode::Infer);
}

TEST_F(Caffe2ImporterTest, importNames) {
  std::string NetDescFilename(GLOW_DATA_PATH
                              "tests/models/caffe2Models/sigmoid.pbtxt");
  std::string NetWeightFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/empty_init_net.pbtxt");
  ExecutionEngine EE;
  auto &mod = EE.getModule();
  auto *F = mod.createFunction("main");
  Tensor input(ElemKind::FloatTy, {6});
  Caffe2ModelLoader caffe2LD(NetDescFilename, NetWeightFilename,
                             {"sigmoid_test_input"}, {&input.getType()}, *F);
  EXPECT_TRUE(mod.getPlaceholderByName("sigmoid_test_output"));
  EXPECT_TRUE(F->getNodeByName("sigmoid_test_output__1"));
}

TEST_F(Caffe2ImporterTest, importSqr) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/sqr_predict_net.pbtxt");
  std::string NetWeightFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/empty_init_net.pbtxt");

  Placeholder *output;
  PlaceholderBindings bindings;

  // Destroy the loader after the graph is loaded since the following execution
  // will not depend on anything from the loader.
  {
    Tensor data(ElemKind::FloatTy, {4, 2});
    Caffe2ModelLoader caffe2LD(NetDescFilename, NetWeightFilename, {"input"},
                               {&data.getType()}, *F);
    output = EXIT_ON_ERR(caffe2LD.getSingleOutput());

    bindings.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(bindings, &mod, {"input"}, {&data});
  }

  // High level check on the content of the graph. We should have
  // save(pow(input, splat(2)))
  EXPECT_EQ(F->getNodes().size(), 3);
  auto *save = getSaveNodeFromDest(output);
  ASSERT_TRUE(save);
  auto *pow = llvm::dyn_cast<PowNode>(save->getInput().getNode());
  ASSERT_TRUE(pow);
  auto *input = llvm::dyn_cast<Placeholder>(pow->getLHS().getNode());
  ASSERT_TRUE(input);
  auto *splat = llvm::dyn_cast<SplatNode>(pow->getRHS().getNode());
  ASSERT_TRUE(splat);
  EXPECT_EQ(splat->getValue(), 2);

  EE.compile(CompilationMode::Infer);
}
