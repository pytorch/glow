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
  auto PH = mod.getPlaceholderByNameSlow(input_name);
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

/// Test loading PReLU op from a Caffe2 model.
/// The input is N*C*H*W (1*2*3*3), the slope is 2.
TEST_F(Caffe2ImporterTest, importPReLU) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename(GLOW_DATA_PATH
                              "tests/models/caffe2Models/prelu.pbtxt");
  std::string NetWeightFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/empty_init_net.pbtxt");

  Placeholder *output;
  PlaceholderBindings bindings;

  // Destroy the loader after the graph is loaded since the following execution
  // will not depend on anything from the loader.
  {
    Tensor data(ElemKind::FloatTy, {1, 2, 3, 3});
    data.getHandle() = {-2.0, -0.5, 0, 1, 2, 3,  4,  5,  6,
                        -1.5, -2.5, 7, 8, 9, 10, 11, 12, 13};
    Tensor slope(ElemKind::FloatTy, {2});
    slope.getHandle() = {0.1, 0.2};
    Caffe2ModelLoader caffe2LD(NetDescFilename, NetWeightFilename,
                               {"prelu_test_input", "slope"},
                               {&data.getType(), &slope.getType()}, *F);
    output = EXIT_ON_ERR(caffe2LD.getSingleOutput());

    bindings.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(bindings, &mod, {"prelu_test_input", "slope"},
                                  {&data, &slope});
  }

  auto res = bindings.get(output);
  EE.compile(CompilationMode::Infer);

  EE.run(bindings);
  auto result = res->getHandle();
  std::vector<dim_t> expectedDims = {1, 2, 3, 3};
  std::vector<float> expectedValues = {-0.2, -0.05, 0, 1, 2, 3,  4,  5,  6,
                                       -0.3, -0.5,  7, 8, 9, 10, 11, 12, 13};
  EXPECT_TRUE(result.dims().vec() == expectedDims);
  for (size_t i = 0; i < expectedValues.size(); i++)
    EXPECT_FLOAT_EQ(result.raw(i), expectedValues[i]);
}

/// Test loading conv op from a Caffe2 model.
/// The input is N*C*H*W (1*1*3*3), the kernel is 2,
/// stride is 1, pad is 1, group is 1, dilation is 2.
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
  std::vector<dim_t> expectedDims = {1, 1, 3, 3};
  std::vector<float> expectedValues = {6, 10, 6, 10, 18, 10, 6, 10, 6};
  EXPECT_TRUE(result.dims().vec() == expectedDims);
  for (size_t i = 0; i < 3 * 3; i++)
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
  // {transpose, transpose} => conv => relu => transpose => save
  EXPECT_EQ(F->getNodes().size(), 6);
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
  auto *transNode3 =
      llvm::dyn_cast<TransposeNode>(convNode->getFilter().getNode());
  ASSERT_TRUE(transNode3);

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

  // High level check on the content of the graph. We have 1 conv, 1 transpose,
  // and1 save.
  EXPECT_EQ(F->getNodes().size(), 3);
  auto *saveNode = getSaveNodeFromDest(output);
  auto *convNode =
      llvm::dyn_cast<ConvolutionNode>(saveNode->getInput().getNode());
  ASSERT_TRUE(convNode);
  auto *transposeNode = llvm::dyn_cast<TransposeNode>(convNode->getFilter());
  ASSERT_TRUE(transposeNode);

  // We have 2 placeholders:  1 input and 1 output.
  EXPECT_EQ(mod.getPlaceholders().size(), 2);
  // We have 2 constants: Weights and bias.
  EXPECT_EQ(mod.getConstants().size(), 2);
}

/// Test loading ChannelwiseQuantizedConvolutionNode op from a Caffe2 model.
/// The input is N*H*W*C (1*1*1*4), the kernel is 1, stride is 1, pad is 1,
/// group is 2.
TEST_F(Caffe2ImporterTest, convGroupQuantized) {
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
  std::vector<unsigned> expectedKernelsStridesAndDilation = {1, 1};
  std::vector<unsigned> expectedPads = {1, 1, 1, 1};
  EXPECT_EQ(groupwiseConv->getKernels(),
            llvm::makeArrayRef(expectedKernelsStridesAndDilation));
  EXPECT_EQ(groupwiseConv->getStrides(),
            llvm::makeArrayRef(expectedKernelsStridesAndDilation));
  EXPECT_EQ(groupwiseConv->getPads(), llvm::makeArrayRef(expectedPads));
  EXPECT_EQ(groupwiseConv->getGroup(), 2);
  EXPECT_EQ(groupwiseConv->getDilation(),
            llvm::makeArrayRef(expectedKernelsStridesAndDilation));

  // Check constant inputs.
  Constant *filterConstant =
      llvm::dyn_cast<Constant>(groupwiseConv->getFilter().getNode());
  Constant *biasConstant =
      llvm::dyn_cast<Constant>(groupwiseConv->getBias().getNode());
  Constant *filterScalesConstant =
      llvm::dyn_cast<Constant>(groupwiseConv->getFilterScales().getNode());
  Constant *filterOffsetsConstant =
      llvm::dyn_cast<Constant>(groupwiseConv->getFilterOffsets().getNode());
  Constant *biasScalesConstant =
      llvm::dyn_cast<Constant>(groupwiseConv->getBiasScales().getNode());
  Constant *biasOffsetsConstant =
      llvm::dyn_cast<Constant>(groupwiseConv->getBiasOffsets().getNode());

  ASSERT_TRUE(filterConstant);
  ASSERT_TRUE(biasConstant);
  ASSERT_TRUE(filterScalesConstant);
  ASSERT_TRUE(filterOffsetsConstant);
  ASSERT_TRUE(biasScalesConstant);
  ASSERT_TRUE(biasOffsetsConstant);

  const auto filterH = filterConstant->getPayload().getHandle<int8_t>();
  const auto biasH = biasConstant->getPayload().getHandle<float>();
  const auto filterScalesH =
      filterScalesConstant->getPayload().getHandle<float>();
  const auto filterOffsetsH =
      filterOffsetsConstant->getPayload().getHandle<int32_t>();
  const auto biasScalesH = biasScalesConstant->getPayload().getHandle<float>();
  const auto biasOffsetsH =
      biasOffsetsConstant->getPayload().getHandle<int32_t>();

  for (size_t i = 0; i < filterH.size(); ++i) {
    EXPECT_EQ(filterH.raw(i), i % 2);
  }

  for (size_t i = 0; i < biasH.size(); ++i) {
    EXPECT_EQ(biasH.raw(i), 7.0);
  }

  for (size_t i = 0; i < filterScalesH.size(); ++i) {
    EXPECT_EQ(filterScalesH.raw(i), 6.0f);
  }

  for (size_t i = 0; i < filterOffsetsH.size(); ++i) {
    EXPECT_EQ(filterOffsetsH.raw(i), 5);
  }

  for (size_t i = 0; i < biasScalesH.size(); ++i) {
    float matmulScale = filterScalesH.raw(i) * input.getType().getScale();
    EXPECT_EQ(biasScalesH.raw(i), matmulScale);
  }

  for (size_t i = 0; i < biasOffsetsH.size(); ++i) {
    EXPECT_EQ(biasOffsetsH.raw(i), 0);
  }

  // We have 2 placeholders: 1 input and 1 output.
  EXPECT_EQ(mod.getPlaceholders().size(), 2);
  // We have 6 constants: Bias, Filter, FilterScales, FilterOffsets, BiasScales
  // and BiasOffsets.
  EXPECT_EQ(mod.getConstants().size(), 6);
}

/// Helper method to run the ConvTranspose operator test cases.
/// \p filename contains the model .onnxtxt.
/// \p expectedDims: output Tensor dimensions.
/// \p expectedValues : output Tensor values expected.
/// The input is N*C*H*W (1*1*2*2), the kernels is {3, 3},
/// strides is {1, 1}, group is 1. Pads can vary.
static void convTransposeTestHelper(std::string &netname, std::string &initname,
                                    llvm::ArrayRef<dim_t> expectedDims,
                                    llvm::ArrayRef<float> expectedValues) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename =
      std::string(GLOW_DATA_PATH "tests/models/caffe2Models/") + netname;

  std::string NetWeightFilename =
      std::string(GLOW_DATA_PATH "tests/models/caffe2Models/") + initname;

  Placeholder *output;
  PlaceholderBindings bindings;

  // Destroy the loader after the graph is loaded since the following execution
  // will not depend on anything from the loader.
  {
    Tensor data;
    getNCHWData(&data, 1, 1, 2, 2);
    data.getHandle() = {2., 3., 4., 5.};

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

  EXPECT_TRUE(result.dims() == expectedDims);
  for (dim_t i = 0, e = expectedValues.size(); i < e; i++) {
    EXPECT_FLOAT_EQ(result.raw(i), expectedValues[i]);
  }
}

/// Test loading ConvTranspose op from a ONNX model.
/// The input is N*C*H*W (1*1*2*2), the kernels is {3, 3},
/// strides is {1, 1}, pads is {0, 0, 0, 0}, group is 1.
TEST(caffe2, importConvTranspose) {
  std::string netname("convtranspose.pbtxt");
  std::string initname("convtranspose_init.pbtxt");
  std::vector<dim_t> expectedDims = {1, 1, 4, 4};
  std::vector<float> expectedValues = {5,  13, 18,  13, 19, 50, 64, 42,
                                       37, 92, 106, 66, 33, 77, 86, 51};
  convTransposeTestHelper(netname, initname, expectedDims, expectedValues);
}

/// Test loading ConvTranspose op from a ONNX model.
/// The input is N*C*H*W (1*1*2*2), the kernels is {3, 3},
/// strides is {1, 1}, pads is {1, 1, 1, 1}, group is 1.
TEST(onnx, importConvTransposePads) {
  std::string netname("convtranspose_pads.pbtxt");
  std::string initname("convtranspose_init.pbtxt");
  std::vector<dim_t> expectedDims = {1, 1, 2, 2};
  std::vector<float> expectedValues = {50, 64, 92, 106};
  convTransposeTestHelper(netname, initname, expectedDims, expectedValues);
}

/// Test loading conv op from a Caffe2 model.
/// The input is N*H*W*C (1*3*3*1), the kernel is 2,
/// stride is 1, pad is 1, group is 1.
TEST(caffe2, convTransposeNHWC) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/convtranspose_nhwc.pbtxt");
  std::string NetWeightFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/convtranspose_nhwc_init.pbtxt");

  Placeholder *output;
  PlaceholderBindings bindings;

  Tensor inputs(ElemKind::FloatTy, {1, 2, 2, 1});
  inputs.getHandle() = {2., 3., 4., 5.};

  // Destroy the loader after the graph is loaded since the following execution
  // will not depend on anything from the loader.
  {
    Caffe2ModelLoader caffe2LD(NetDescFilename, NetWeightFilename, {"inputs"},
                               {&inputs.getType()}, *F);
    output = EXIT_ON_ERR(caffe2LD.getSingleOutput());
  }

  // High level check on the content of the graph. We have 1 conv, 1 Transpose,
  // and 1 save.
  EXPECT_EQ(F->getNodes().size(), 3);
  auto *saveNode = getSaveNodeFromDest(output);
  auto *convTransposeNode =
      llvm::dyn_cast<ConvTransposeNode>(saveNode->getInput().getNode());
  ASSERT_TRUE(convTransposeNode);
  auto *transposeNode =
      llvm::dyn_cast<TransposeNode>(convTransposeNode->getFilter());
  ASSERT_TRUE(transposeNode);

  // We have 2 placeholders:  1 input and 1 output.
  EXPECT_EQ(mod.getPlaceholders().size(), 2);
  // We have 2 constants: Weights and bias.
  EXPECT_EQ(mod.getConstants().size(), 2);
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
  ASSERT_FALSE(avgPoolNode->getCountIncludePads());

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
  ASSERT_TRUE(avgPoolNode->getCountIncludePads());
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

TEST_F(Caffe2ImporterTest, concatAddAxisAtEdge) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/concat_add_axis_at_edge.pbtxt");
  std::string NetWeightFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/empty_init_net.pbtxt");

  PlaceholderBindings bindings;

  Placeholder *output;
  const std::array<dim_t, 4> inputShape{7, 11, 13, 5};
  Tensor inputs_0(ElemKind::FloatTy, inputShape);
  Tensor inputs_1(ElemKind::FloatTy, inputShape);
  Tensor inputs_2(ElemKind::FloatTy, inputShape);
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

  const std::vector<Tensor *> inputs{&inputs_0, &inputs_1, &inputs_2};

  // Check that the shape of the output matches what Caffe2 expects.
  std::vector<dim_t> expectedDims{inputShape.begin(), inputShape.end()};
  expectedDims.push_back(inputs.size());
  EXPECT_EQ(expectedDims, output->dims().vec());

  auto res = bindings.get(output);
  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  auto result = res->getHandle();

  // Check that the output matches the concatenation of all inputs.
  LOG(INFO) << "inputs_0=" << inputs_0.toString();
  LOG(INFO) << "inputs_1=" << inputs_1.toString();
  LOG(INFO) << "inputs_2=" << inputs_2.toString();
  LOG(INFO) << "result=" << result.clone().toString();

  for (dim_t i = 0; i < inputs.size(); ++i) {
    const auto inputsHandle = inputs[i]->getHandle();

    for (dim_t dim1 = 0; dim1 < inputShape[0]; ++dim1) {
      for (dim_t dim2 = 0; dim2 < inputShape[1]; ++dim2) {
        for (dim_t dim3 = 0; dim3 < inputShape[2]; ++dim3) {
          for (dim_t dim4 = 0; dim4 < inputShape[3]; ++dim4) {
            EXPECT_FLOAT_EQ(result.at({dim1, dim2, dim3, dim4, i}),
                            inputsHandle.at({dim1, dim2, dim3, dim4}));
          }
        }
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

TEST_F(Caffe2ImporterTest, batchMatMulManyDims) {
  const std::string NetDescFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/batched_matmul.pbtxt");
  const std::string NetWeightFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/empty_init_net.pbtxt");

  // Set of test cases: (lhs shape, rhs shape, expected result shape)
  const std::vector<std::vector<std::vector<dim_t>>> shape_cases{
      {{2, 2, 3, 4}, {2, 2, 4, 5}, {2, 2, 3, 5}},
      {{2, 2, 3, 4}, {2, 1, 4, 5}, {2, 2, 3, 5}},
      {{2, 2, 3, 4}, {2, 2, 2, 4, 5}, {2, 2, 2, 3, 5}},
      {{2, 2, 2, 3, 4}, {3, 1, 2, 2, 4, 5}, {3, 2, 2, 2, 3, 5}},
      {{2, 4, 5}, {3, 2, 5, 6}, {3, 2, 4, 6}},
  };

  for (const auto &shapes : shape_cases) {
    ExecutionEngine EE{};
    auto &mod = EE.getModule();
    Function *F = mod.createFunction("main");

    PlaceholderBindings bindings;
    Placeholder *output;

    Tensor lhs{ElemKind::FloatTy, shapes[0]};
    const auto lhsSize = std::accumulate(lhs.dims().begin(), lhs.dims().end(),
                                         1, std::multiplies<>());
    std::vector<float> lhsData(lhsSize);
    std::iota(lhsData.begin(), lhsData.end(), 0);
    lhs.getHandle() = lhsData;

    Tensor rhs{ElemKind::FloatTy, shapes[1]};
    const auto rhsSize = std::accumulate(rhs.dims().begin(), rhs.dims().end(),
                                         1, std::multiplies<>());
    std::vector<float> rhsData(rhsSize);
    std::iota(rhsData.begin(), rhsData.end(), 10);
    rhs.getHandle() = rhsData;

    // Destroy the loader after the graph is loaded since the following
    // execution will not depend on anything from the loader.
    {
      Caffe2ModelLoader caffe2LD(NetDescFilename, NetWeightFilename,
                                 {"lhs", "rhs"},
                                 {&lhs.getType(), &rhs.getType()}, *F);
      output = EXIT_ON_ERR(caffe2LD.getSingleOutput());

      bindings.allocate(mod.getPlaceholders());
      updateInputPlaceholdersByName(bindings, &mod, {"lhs", "rhs"},
                                    {&lhs, &rhs});
    }

    EXPECT_EQ(shapes[2], output->dims().vec());

    auto res = bindings.get(output);
    EE.compile(CompilationMode::Infer);
    EE.run(bindings);

    auto result = res->getHandle();

    const auto resultSize = std::accumulate(
        output->dims().begin(), output->dims().end(), 1, std::multiplies<>());
    result.dump(llvm::errs(), resultSize);
  }
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

  // High level check on the content of the graph. We have 1 FC node,
  // 1 transpose, and 1 save.
  EXPECT_EQ(F->getNodes().size(), 3);
  auto *saveNode = getSaveNodeFromDest(output);
  auto *fcNode =
      llvm::dyn_cast<FullyConnectedNode>(saveNode->getInput().getNode());
  ASSERT_TRUE(fcNode);
  auto *transposeNode = llvm::dyn_cast<TransposeNode>(fcNode->getWeights());
  ASSERT_TRUE(transposeNode);

  // Check the numerical values of the weights and biases.
  {
    const Constant *constant =
        llvm::dyn_cast<Constant>(transposeNode->getInput());
    ASSERT_TRUE(constant);
    const Tensor &weights = constant->getPayload();
    const std::vector<dim_t> expectedDimensions = {4, 3};
    const std::vector<float> expectedValues = {1.0f, 2.0f,  3.0f,  4.0f,
                                               5.0f, 6.0f,  7.0f,  8.0f,
                                               9.0f, 10.0f, 11.0f, 12.0f};
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

  // High level check on the content of the graph. We have two Splats for
  // weights and bias, a reshape, Transpose for FC weights, an FC, another
  // reshape, and a save.
  EXPECT_EQ(F->getNodes().size(), 7);

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
  auto *transpose = llvm::dyn_cast<TransposeNode>(fcNode->getWeights());
  ASSERT_TRUE(transpose);

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

  // High level check on the content of the graph. We have two Splats for
  // weights and bias, a reshape, an FC, another reshape, and a save.
  EXPECT_EQ(F->getNodes().size(), 6);

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

TEST_F(Caffe2ImporterTest, FC3DSecondAxis) {
  const std::string NetDescFilename(
      GLOW_DATA_PATH
      "tests/models/caffe2Models/fc_3d_second_axis_predict.pbtxt");
  const std::string NetWeightFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/fc_3d_second_axis_init.pbtxt");

  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  PlaceholderBindings bindings;
  Placeholder *output;

  const std::vector<dim_t> inputShape{2, 3, 4};

  // Create and populate input tensor
  Tensor input{ElemKind::FloatTy, inputShape};
  std::vector<float> inputData;
  inputData.resize(input.size());
  std::iota(inputData.begin(), inputData.end(), 0);
  input.getHandle() = inputData;

  // Destroy the loader after the graph is loaded since the following
  // execution will not depend on anything from the loader.
  {
    Caffe2ModelLoader caffe2LD(NetDescFilename, NetWeightFilename, {"input"},
                               {&input.getType()}, *F);
    output = EXIT_ON_ERR(caffe2LD.getSingleOutput());

    bindings.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(bindings, &mod, {"input"}, {&input});
  }

  auto res = bindings.get(output);
  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  std::vector<dim_t> expectedShape{inputShape.begin(), inputShape.end() - 1};
  expectedShape.push_back(5); // bias size as defined in .pbtxt file
  EXPECT_EQ(expectedShape, res->dims().vec());

  auto result = res->getHandle();

  // Data is based on Caffe2 results
  std::vector<std::vector<std::vector<float>>> expectedData{
      {{14, 39, 64, 89, 114},
       {38, 127, 216, 305, 394},
       {62, 215, 368, 521, 674}},
      {{86, 303, 520, 737, 954},
       {110, 391, 672, 953, 1234},
       {134, 479, 824, 1169, 1514}}};

  for (dim_t dim1 = 0; dim1 < expectedShape[0]; ++dim1) {
    for (dim_t dim2 = 0; dim2 < expectedShape[0]; ++dim2) {
      for (dim_t dim3 = 0; dim3 < expectedShape[0]; ++dim3) {
        EXPECT_FLOAT_EQ(expectedData[dim1][dim2][dim3],
                        result.at({dim1, dim2, dim3}));
      }
    }
  }
}

TEST_F(Caffe2ImporterTest, FC4DFirstAxis) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename(
      GLOW_DATA_PATH
      "tests/models/caffe2Models/fc_4d_first_axis_predict_net.pbtxt");
  std::string NetWeightFilename(
      GLOW_DATA_PATH
      "tests/models/caffe2Models/fc_4d_first_axis_init_net.pbtxt");

  Placeholder *output;
  PlaceholderBindings bindings;

  {
    Tensor inputs(ElemKind::FloatTy, {1, 5, 1, 1});

    // Weights and bias are read from NetWeightFilename
    Caffe2ModelLoader caffe2LD(NetDescFilename, NetWeightFilename, {"inputs"},
                               {&inputs.getType()}, *F);
    output = EXIT_ON_ERR(caffe2LD.getSingleOutput());
    bindings.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(bindings, &mod, {"inputs"}, {&inputs});
  }

  auto finalShape = output->getType()->dims();
  std::vector<dim_t> expectedOutput{1, 3};
  EXPECT_EQ(finalShape, llvm::makeArrayRef(expectedOutput));
}

TEST_F(Caffe2ImporterTest, FbFCPacked3DSecondAxis) {
  const std::string NetDescFilename(
      GLOW_DATA_PATH
      "tests/models/caffe2Models/fbfcpacked_3d_second_axis_predict.pbtxt");
  const std::string NetWeightFilename(
      GLOW_DATA_PATH
      "tests/models/caffe2Models/fbfcpacked_3d_second_axis_init.pbtxt");

  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  PlaceholderBindings bindings;
  Placeholder *output;

  const std::vector<dim_t> inputShape{2, 3, 4};

  // Create and populate input tensor
  Tensor input{ElemKind::FloatTy, inputShape};
  std::vector<float> inputData;
  inputData.resize(input.size());
  std::iota(inputData.begin(), inputData.end(), 0);
  input.getHandle() = inputData;

  // Destroy the loader after the graph is loaded since the following
  // execution will not depend on anything from the loader.
  {
    Caffe2ModelLoader caffe2LD(NetDescFilename, NetWeightFilename, {"input"},
                               {&input.getType()}, *F);
    output = EXIT_ON_ERR(caffe2LD.getSingleOutput());

    bindings.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(bindings, &mod, {"input"}, {&input});
  }

  auto res = bindings.get(output);
  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  std::vector<dim_t> expectedShape{inputShape.begin(), inputShape.end() - 1};
  expectedShape.push_back(5); // bias size as defined in .pbtxt file
  EXPECT_EQ(expectedShape, res->dims().vec());

  auto result = res->getHandle();

  // Data is based on Caffe2 results
  std::vector<std::vector<std::vector<float>>> expectedData{
      {{14, 39, 64, 89, 114},
       {38, 127, 216, 305, 394},
       {62, 215, 368, 521, 674}},
      {{86, 303, 520, 737, 954},
       {110, 391, 672, 953, 1234},
       {134, 479, 824, 1169, 1514}}};

  for (dim_t dim1 = 0; dim1 < expectedShape[0]; ++dim1) {
    for (dim_t dim2 = 0; dim2 < expectedShape[0]; ++dim2) {
      for (dim_t dim3 = 0; dim3 < expectedShape[0]; ++dim3) {
        EXPECT_FLOAT_EQ(expectedData[dim1][dim2][dim3],
                        result.at({dim1, dim2, dim3}));
      }
    }
  }
}

TEST_F(Caffe2ImporterTest, FbFCPacked4DFirstAxis) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename(
      GLOW_DATA_PATH
      "tests/models/caffe2Models/fbfcpacked_4d_first_axis_predict_net.pbtxt");
  std::string NetWeightFilename(
      GLOW_DATA_PATH
      "tests/models/caffe2Models/fbfcpacked_4d_first_axis_init_net.pbtxt");

  Placeholder *output;
  PlaceholderBindings bindings;

  {
    Tensor input(ElemKind::FloatTy, {1, 5, 1, 1});

    // Weights and bias are read from NetWeightFilename
    Caffe2ModelLoader caffe2LD(NetDescFilename, NetWeightFilename, {"input"},
                               {&input.getType()}, *F);
    output = EXIT_ON_ERR(caffe2LD.getSingleOutput());
    bindings.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(bindings, &mod, {"input"}, {&input});
  }

  auto finalShape = output->getType()->dims();
  std::vector<dim_t> expectedOutput{1, 3};
  EXPECT_EQ(finalShape, llvm::makeArrayRef(expectedOutput));
}

TEST_F(Caffe2ImporterTest, Int8FC3DSecondAxis) {
  const std::string NetDescFilename(
      GLOW_DATA_PATH
      "tests/models/caffe2Models/int8fc_3d_second_axis_predict.pbtxt");
  const std::string NetWeightFilename(
      GLOW_DATA_PATH
      "tests/models/caffe2Models/int8fc_3d_second_axis_init.pbtxt");

  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  PlaceholderBindings bindings;
  Placeholder *output;

  const std::vector<dim_t> inputShape{2, 2, 3};

  // Create and populate input tensor
  Tensor input{ElemKind::Int8QTy, inputShape, 1, 0};
  std::vector<int8_t> inputData;
  inputData.resize(input.size());
  std::iota(inputData.begin(), inputData.end(), 0);
  input.getHandle<int8_t>() = inputData;

  // Destroy the loader after the graph is loaded since the following
  // execution will not depend on anything from the loader.
  {
    Caffe2ModelLoader caffe2LD(NetDescFilename, NetWeightFilename, {"input"},
                               {&input.getType()}, *F);
    output = EXIT_ON_ERR(caffe2LD.getSingleOutput());

    bindings.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(bindings, &mod, {"input"}, {&input});
  }

  auto res = bindings.get(output);
  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  std::vector<dim_t> expectedShape{inputShape.begin(), inputShape.end() - 1};
  expectedShape.push_back(2); // bias size as defined in .pbtxt file
  EXPECT_EQ(expectedShape, res->dims().vec());

  auto result = res->getHandle<float>();

  // Data is based on Caffe2 results
  std::vector<std::vector<std::vector<float>>> expectedData{
      {{4.956894, 14.890197}, {14.003877, 51.125492}},
      {{22.921618, 86.84314}, {31.9686, 123.07844}}};

  for (dim_t dim1 = 0; dim1 < expectedShape[0]; ++dim1) {
    for (dim_t dim2 = 0; dim2 < expectedShape[0]; ++dim2) {
      for (dim_t dim3 = 0; dim3 < expectedShape[0]; ++dim3) {
        EXPECT_NEAR(expectedData[dim1][dim2][dim3],
                    result.at({dim1, dim2, dim3}), 0.2);
      }
    }
  }
}

TEST_F(Caffe2ImporterTest, Int8FC4DFirstAxis) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename(
      GLOW_DATA_PATH
      "tests/models/caffe2Models/int8fc_4d_first_axis_predict_net.pbtxt");
  std::string NetWeightFilename(
      GLOW_DATA_PATH
      "tests/models/caffe2Models/int8fc_4d_first_axis_init_net.pbtxt");

  Placeholder *output;
  PlaceholderBindings bindings;

  {
    Tensor input(ElemKind::FloatTy, {1, 5, 1, 1});

    // Weights and bias are read from NetWeightFilename
    Caffe2ModelLoader caffe2LD(NetDescFilename, NetWeightFilename, {"input"},
                               {&input.getType()}, *F);
    output = EXIT_ON_ERR(caffe2LD.getSingleOutput());
    bindings.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(bindings, &mod, {"input"}, {&input});
  }

  auto finalShape = output->getType()->dims();
  std::vector<dim_t> expectedOutput{1, 3};
  EXPECT_EQ(finalShape, llvm::makeArrayRef(expectedOutput));
}

TEST_F(Caffe2ImporterTest, Int8FCDynamicQuantized) {
  const std::string NetDescFilename(
      GLOW_DATA_PATH
      "tests/models/caffe2Models/dynamic_quantized_fc_predict_net.pbtxt");
  const std::string NetWeightFilename(
      GLOW_DATA_PATH
      "tests/models/caffe2Models/dynamic_quantized_fc_init.pbtxt");

  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  PlaceholderBindings bindings;
  Placeholder *output;

  const std::vector<dim_t> inputShape{2, 3};

  // Create and populate input tensor
  Tensor input{ElemKind::Float16Ty, inputShape};
  std::vector<float16_t> inputData;
  inputData.resize(input.size());
  std::iota(inputData.begin(), inputData.end(), 1);
  input.getHandle<float16_t>() = inputData;

  // Destroy the loader after the graph is loaded since the following
  // execution will not depend on anything from the loader.
  {
    Caffe2ModelLoader caffe2LD(NetDescFilename, NetWeightFilename, {"input"},
                               {&input.getType()}, *F);
    output = EXIT_ON_ERR(caffe2LD.getSingleOutput());

    bindings.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(bindings, &mod, {"input"}, {&input});
  }

  auto res = bindings.get(output);
  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  std::vector<dim_t> expectedShape{2, 4};
  EXPECT_EQ(expectedShape, res->dims().vec());

  auto result = res->getHandle<float16_t>();

  std::vector<std::vector<float>> expectedData{{8, 27, 46, 65},
                                               {17, 63, 109, 155}};

  for (dim_t dim1 = 0; dim1 < expectedShape[0]; ++dim1) {
    for (dim_t dim2 = 0; dim2 < expectedShape[1]; ++dim2) {
      EXPECT_NEAR(expectedData[dim1][dim2], result.at({dim1, dim2}), 0.3);
    }
  }
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

  const std::vector<dim_t> inputShape{3, 2};
  PlaceholderBindings bindings;
  Placeholder *outputPH;
  Tensor inputs_0(ElemKind::FloatTy, inputShape);
  inputs_0.getHandle().randomize(-3.0, 3.0, mod.getPRNG());
  // Destroy the loader after the graph is loaded since the following execution
  // will not depend on anything from the loader.
  {
    Caffe2ModelLoader caffe2LD(NetDescFilename, NetWeightFilename, {"input_0"},
                               {&inputs_0.getType()}, *F);
    outputPH = EXIT_ON_ERR(caffe2LD.getSingleOutput());
    bindings.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(bindings, &mod, {"input_0"}, {&inputs_0});
  }

  EXPECT_EQ(F->getNodes().size(), 2);
  auto *saveNode = getSaveNodeFromDest(outputPH);
  auto *bucketizeNode =
      llvm::dyn_cast<BucketizeNode>(saveNode->getInput().getNode());
  ASSERT_TRUE(bucketizeNode);
  auto boundriesVec = bucketizeNode->getBoundaries();
  ASSERT_EQ(boundriesVec.size(), 2);
  EXPECT_NEAR(boundriesVec[0], 0.1, 0.00001);
  EXPECT_NEAR(boundriesVec[1], 2.5, 0.00001);
  // We have one input and one output.
  EXPECT_EQ(mod.getPlaceholders().size(), 2);

  auto output = bindings.get(outputPH);
  EXPECT_EQ(inputShape, output->dims().vec());

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  auto outputH = output->getHandle<int32_t>();

  // From the bucketize_op_net.pbtxt file
  std::array<float, 2> boundaries{0.1, 2.5};
  for (dim_t d1 = 0; d1 < inputShape[0]; ++d1) {
    for (dim_t d2 = 0; d2 < inputShape[1]; ++d2) {
      auto inputVal = inputs_0.getHandle().at({d1, d2});
      auto exp =
          std::lower_bound(boundaries.begin(), boundaries.end(), inputVal) -
          boundaries.begin();
      EXPECT_EQ(exp, outputH.at({d1, d2}));
    }
  }
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
  auto scale = resizeNearestNode->getScale();
  EXPECT_EQ(scale[0], 1);
  auto heightScale = scale[1];
  auto widthScale = scale[2];
  EXPECT_EQ(scale[3], 1);
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
  ASSERT_EQ(inputNode, mod.getPlaceholderByNameSlow("inputs_0"));
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
  ASSERT_EQ(inputNode, mod.getPlaceholderByNameSlow("inputs_0"));
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
  ASSERT_EQ(inputNode, mod.getPlaceholderByNameSlow("input"));

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
  constexpr dim_t kDataSize = 10;
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
  constexpr dim_t kRows = 10;
  constexpr dim_t kCols = 20;
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
  const dim_t kRows = 10;
  const dim_t kCols = 5;
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
  const dim_t kDataSize = 10;
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
  const dim_t kDataSize = 10;
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
  // We have 1 Logit, 1 Save.
  EXPECT_EQ(F->getNodes().size(), 2);

  // Graph has one input and one output.
  EXPECT_EQ(mod.getPlaceholders().size(), 2);
}

// Test loading Logit operator from a Caffe2 model.
TEST_F(Caffe2ImporterTest, Swish) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename(GLOW_DATA_PATH
                              "tests/models/caffe2Models/swish_op_net.pbtxt");
  std::string NetWeightFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/empty_init_net.pbtxt");

  PlaceholderBindings bindings;
  Placeholder *output;

  // Input tensors.
  Tensor X(ElemKind::FloatTy, {10});

  // Destroy the loader after the graph is loaded
  {
    Caffe2ModelLoader caffe2LD(NetDescFilename, NetWeightFilename, {"input"},
                               {&X.getType()}, *F);
    output = EXIT_ON_ERR(caffe2LD.getSingleOutput());
    bindings.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(bindings, &mod, {"input"}, {&X});
  }

  // Check that the type of the output matches the input.
  EXPECT_TRUE(output->getType()->isEqual(X.getType()));

  // High level checks on the content of the graph.
  EXPECT_EQ(F->getNodes().size(), 2); // Save and Swish
  auto *saveNode = getSaveNodeFromDest(output);
  auto *swish = llvm::dyn_cast<SwishNode>(saveNode->getInput());
  ASSERT_TRUE(swish);

  // Graph has one input and one output.
  EXPECT_EQ(mod.getPlaceholders().size(), 2);

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);
}

void testSparseToDense(llvm::ArrayRef<dim_t> indicesShape,
                       llvm::ArrayRef<dim_t> valuesShape,
                       llvm::ArrayRef<dim_t> dataToInferDimShape,
                       int expectedNumberOfNodes) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/sparse_to_dense.pbtxt");
  std::string NetWeightFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/empty_init_net.pbtxt");

  Placeholder *outputPH;
  PlaceholderBindings bindings;

  // Create inputs.
  Tensor indices(ElemKind::Int64ITy, indicesShape);
  Tensor values(ElemKind::FloatTy, valuesShape);
  Tensor dataToInferDim(ElemKind::FloatTy, dataToInferDimShape);

  indices.getHandle<int64_t>().randomize(0, dataToInferDimShape[0] - 1,
                                         mod.getPRNG());
  values.getHandle().randomize(-3.0, 3.0, mod.getPRNG());
  // Destroy the loader after the graph is loaded since the following execution
  // will not depend on anything from the loader.
  {
    Caffe2ModelLoader caffe2LD(
        NetDescFilename, NetWeightFilename,
        {"indices", "values", "dataToInferDim"},
        {&indices.getType(), &values.getType(), &dataToInferDim.getType()}, *F);
    outputPH = EXIT_ON_ERR(caffe2LD.getSingleOutput());
    bindings.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(bindings, &mod, {"indices", "values"},
                                  {&indices, &values});
  }

  // Check that the shape of the output matches that of the expected output.
  const std::vector<dim_t> expectedOutputShape{dataToInferDimShape[0],
                                               valuesShape[1], valuesShape[2]};
  EXPECT_EQ(expectedOutputShape, outputPH->dims().vec());

  // High level checks on the content of the graph.
  // We should have 1 Splat, 1 optional Reshape, 1 ScatterData and 1 Output
  // node, i.e. 4 nodes in total.
  EXPECT_EQ(F->getNodes().size(), expectedNumberOfNodes);

  // Check that the graph has the expected shape (SparseToDense -> Save),
  // starting from the output.
  auto *saveNode = getSaveNodeFromDest(outputPH);
  auto *STDN = llvm::dyn_cast<ScatterDataNode>(saveNode->getInput());
  ASSERT_TRUE(STDN);

  // Graph has three inputs and one output.
  EXPECT_EQ(mod.getPlaceholders().size(), 4);

  auto output = bindings.get(outputPH);

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  auto outputH = output->getHandle();

  // Initialize with zeroes
  std::vector<std::vector<std::vector<float>>> expected(
      expectedOutputShape[0],
      std::vector<std::vector<float>>(
          expectedOutputShape[1],
          std::vector<float>(expectedOutputShape[2], 0)));
  for (dim_t d1 = 0; d1 < valuesShape[0]; ++d1) {
    dim_t dest_d1 = indices.getHandle<int64_t>().at(d1);
    for (dim_t d2 = 0; d2 < valuesShape[1]; ++d2) {
      for (dim_t d3 = 0; d3 < valuesShape[2]; ++d3) {
        expected[dest_d1][d2][d3] += values.getHandle().at({d1, d2, d3});
      }
    }
  }

  for (dim_t d1 = 0; d1 < expectedOutputShape[0]; ++d1) {
    for (dim_t d2 = 0; d2 < expectedOutputShape[1]; ++d2) {
      for (dim_t d3 = 0; d3 < expectedOutputShape[2]; ++d3) {
        EXPECT_NEAR(expected[d1][d2][d3], outputH.at({d1, d2, d3}), 1e-3);
      }
    }
  }
}

// Test loading a SparseToDense operator.
TEST_F(Caffe2ImporterTest, sparseToDense_indices1D) {
  constexpr dim_t kNumIndices = 40;
  constexpr dim_t kMaxIndex = 20;
  constexpr dim_t kRows = 10;
  constexpr dim_t kCols = 5;

  testSparseToDense({kNumIndices}, {kNumIndices, kRows, kCols}, {kMaxIndex}, 4);
}

TEST_F(Caffe2ImporterTest, sparseToDense_indices2D) {
  constexpr dim_t kNumIndices = 40;
  constexpr dim_t kMaxIndex = 20;
  constexpr dim_t kRows = 10;
  constexpr dim_t kCols = 5;

  testSparseToDense({kNumIndices, 1}, {kNumIndices, kRows, kCols}, {kMaxIndex},
                    3);
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
  EXPECT_TRUE(N->getResult().dims().equals({6, 10, 20, 30}));
  // Check that mask was read correctly.
  EXPECT_TRUE(N->getMask().equals({42, 100, 300, 1, 0, 312}));
}

// Test loading a FillExamplesWithIndicator
TEST_F(Caffe2ImporterTest, FillExamplesWithIndicator) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename(
      GLOW_DATA_PATH
      "tests/models/caffe2Models/fill_examples_with_indicator.pbtxt");
  std::string NetWeightFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/empty_init_net.pbtxt");

  Placeholder *outputPH;
  PlaceholderBindings bindings;
  // Create inputs.
  constexpr dim_t n = 20;
  constexpr dim_t d1 = 3;
  constexpr dim_t d2 = 4;
  Tensor indicator(ElemKind::Int32ITy, {n});
  auto indicatorH = indicator.getHandle<int32_t>();
  indicatorH.randomize(0, 1, mod.getPRNG());

  dim_t m = 0;
  for (size_t i = 0, s = indicatorH.actualSize(); i < s; i++) {
    m += indicatorH.at(i);
  }
  Tensor data(ElemKind::FloatTy, {m, d1, d2});
  data.zero();
  data.getHandle().randomize(-3.0, 3.0, mod.getPRNG());
  // Destroy the loader after the graph is loaded since the following execution
  // will not depend on anything from the loader.
  {
    Caffe2ModelLoader caffe2LD(NetDescFilename, NetWeightFilename,
                               {"data", "indicator"},
                               {&data.getType(), &indicator.getType()}, *F);
    outputPH = EXIT_ON_ERR(caffe2LD.getSingleOutput());
    bindings.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(bindings, &mod, {"data", "indicator"},
                                  {&data, &indicator});
  }

  // Check that the shape of the output matches that of the expected output.
  const std::vector<dim_t> expectedOutputShape{n, d1, d2};
  EXPECT_EQ(expectedOutputShape, outputPH->dims().vec());
  // Graph has 9 nodes: 2 Reshapes, 2 Converts, Nonzero, Slice, Splat,
  // ScatterData, Output
  // NonZero is translated into multiple ops.
  EXPECT_EQ(F->getNodes().size(), 21);

  // Graph has two inputs and one output.
  EXPECT_EQ(mod.getPlaceholders().size(), 3);

  auto output = bindings.get(outputPH);

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  auto outputH = output->getHandle();

  // Initialize with zeroes
  std::vector<std::vector<std::vector<float>>> expected(
      n, std::vector<std::vector<float>>(d1, std::vector<float>(d2)));
  for (dim_t d = 0, v = 0; d < n; ++d) {
    if (indicatorH.at(d)) {
      for (dim_t e = 0; e < d1; ++e) {
        for (dim_t f = 0; f < d2; ++f) {
          expected[d][e][f] = data.getHandle().at({v, e, f});
        }
      }
      v++;
    }
  }

  for (dim_t d = 0; d < n; ++d) {
    for (dim_t e = 0; e < d1; ++e) {
      for (dim_t f = 0; f < d2; ++f) {
        EXPECT_NEAR(expected[d][e][f], outputH.at({d, e, f}), 1e-3);
      }
    }
  }
}

// Test loading a BatchSparseToDense operator w/ second dimension = 1.
TEST_F(Caffe2ImporterTest, BatchSparseToDense_lastdim1) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename(
      GLOW_DATA_PATH
      "tests/models/caffe2Models/batch_sparse_to_dense_last_dim_1.pbtxt");
  std::string NetWeightFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/empty_init_net.pbtxt");

  Placeholder *outputPH;
  PlaceholderBindings bindings;
  // Create inputs.
  constexpr dim_t numBatches = 100;
  Tensor lengths(ElemKind::Int32ITy, {numBatches});
  auto lengthsH = lengths.getHandle<int32_t>();
  lengthsH.randomize(0, 1, mod.getPRNG());

  // Calculate number of nonzero indices.
  dim_t numIndices = 0;
  for (size_t i = 0, n = lengthsH.actualSize(); i < n; i++) {
    numIndices += lengthsH.at(i);
  }
  Tensor indices(ElemKind::Int64ITy, {numIndices});
  Tensor values(ElemKind::FloatTy, {numIndices});
  indices.zero();
  values.getHandle().randomize(-3.0, 3.0, mod.getPRNG());
  // Destroy the loader after the graph is loaded since the following execution
  // will not depend on anything from the loader.
  {
    Caffe2ModelLoader caffe2LD(
        NetDescFilename, NetWeightFilename, {"lengths", "indices", "values"},
        {&lengths.getType(), &indices.getType(), &values.getType()}, *F);
    outputPH = EXIT_ON_ERR(caffe2LD.getSingleOutput());
    bindings.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(bindings, &mod,
                                  {"lengths", "indices", "values"},
                                  {&lengths, &indices, &values});
  }

  // Check that the shape of the output matches that of the expected output.
  const std::vector<dim_t> expectedOutputShape{numBatches, 1};
  EXPECT_EQ(expectedOutputShape, outputPH->dims().vec());
  EXPECT_EQ(F->getNodes().size(), 20);

  // Graph has three inputs and one output.
  EXPECT_EQ(mod.getPlaceholders().size(), 4);

  auto output = bindings.get(outputPH);

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  auto outputH = output->getHandle();

  // Initialize with zeroes
  std::vector<std::vector<float>> expected(numBatches, std::vector<float>(1));
  for (dim_t d = 0, v = 0; d < numBatches; ++d) {
    if (lengthsH.at(d) == 1) {
      expected[d][0] = values.getHandle().at({v});
      v++;
    }
  }

  for (dim_t d = 0; d < numBatches; ++d) {
    EXPECT_NEAR(expected[d][0], outputH.at({d, 0}), 1e-3);
  }
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
    Caffe2ModelLoader caffe2LD(NetDescFilename.str(), NetWeightFilename, {"X"},
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
    Caffe2ModelLoader caffe2LD(NetDescFilename.str(), NetWeightFilename, {"X"},
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
  //            |       Broadcast    Broadcast
  //            |           |            |
  //            |           |            |
  //            |           /            /
  //            v  v-------             /
  //            Mul                    /
  //             |   /----------------
  //             v  v
  //             Add
  //              |
  //              v
  //             Save

  EXPECT_EQ(F->getNodes().size(), 5);
  auto *save = getSaveNodeFromDest(output);
  auto *add = llvm::dyn_cast<AddNode>(save->getInput().getNode());
  ASSERT_TRUE(add);
  auto *mul = llvm::dyn_cast<MulNode>(add->getLHS().getNode());
  ASSERT_TRUE(mul);
  auto *bBN = llvm::dyn_cast<BroadcastNode>(add->getRHS().getNode());
  ASSERT_TRUE(bBN);
  EXPECT_EQ(bBN->getAxis(), 0);
  auto *XPH = llvm::dyn_cast<Placeholder>(mul->getRHS().getNode());
  EXPECT_EQ(XPH, mod.getPlaceholderByNameSlow("X"));
  auto *wBN = llvm::dyn_cast<BroadcastNode>(mul->getLHS().getNode());
  ASSERT_TRUE(wBN);
  EXPECT_EQ(wBN->getAxis(), 0);
  auto *wPH = llvm::dyn_cast<Placeholder>(wBN->getInput().getNode());
  EXPECT_EQ(wPH, mod.getPlaceholderByNameSlow("w"));
  auto *bPH = llvm::dyn_cast<Placeholder>(bBN->getInput().getNode());
  EXPECT_EQ(bPH, mod.getPlaceholderByNameSlow("b"));

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
  //            |       Broadcast    Broadcast
  //            |           |            |
  //            |           |            |
  //            |           /            /
  //            v  v-------             /
  //            Mul                    /
  //             |   /----------------
  //             v  v
  //             Add
  //              |
  //              v
  //             Save

  EXPECT_EQ(F->getNodes().size(), 5);
  auto *save = getSaveNodeFromDest(output);
  auto *add = llvm::dyn_cast<AddNode>(save->getInput().getNode());
  ASSERT_TRUE(add);
  auto *mul = llvm::dyn_cast<MulNode>(add->getLHS().getNode());
  ASSERT_TRUE(mul);
  auto *bBN = llvm::dyn_cast<BroadcastNode>(add->getRHS().getNode());
  ASSERT_TRUE(bBN);
  EXPECT_EQ(bBN->getAxis(), 1);
  auto *XPH = llvm::dyn_cast<Placeholder>(mul->getRHS().getNode());
  EXPECT_EQ(XPH, mod.getPlaceholderByNameSlow("X"));
  auto *wBN = llvm::dyn_cast<BroadcastNode>(mul->getLHS().getNode());
  ASSERT_TRUE(wBN);
  EXPECT_EQ(wBN->getAxis(), 1);
  auto *wPH = llvm::dyn_cast<Placeholder>(wBN->getInput().getNode());
  EXPECT_EQ(wPH, mod.getPlaceholderByNameSlow("w"));
  auto *bPH = llvm::dyn_cast<Placeholder>(bBN->getInput().getNode());
  EXPECT_EQ(bPH, mod.getPlaceholderByNameSlow("b"));

  // We have three inputs and one output.
  EXPECT_EQ(mod.getPlaceholders().size(), 4);
}

/// Test loading an ElementwiseLinear operator with implicit broadcast
TEST_F(Caffe2ImporterTest, elementwiseImplicitBroadcast) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename(
      GLOW_DATA_PATH
      "tests/models/caffe2Models/elementwise_linear_broadcast_net.pbtxt");
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
  //            |       Broadcast    Broadcast
  //            |           |            |
  //            |           |            |
  //            |           /            /
  //            v  v-------             /
  //            Mul                    /
  //             |   /----------------
  //             v  v
  //             Add
  //              |
  //              v
  //             Save

  EXPECT_EQ(F->getNodes().size(), 5);
  auto *save = getSaveNodeFromDest(output);
  auto *add = llvm::dyn_cast<AddNode>(save->getInput().getNode());
  ASSERT_TRUE(add);
  auto *mul = llvm::dyn_cast<MulNode>(add->getLHS().getNode());
  ASSERT_TRUE(mul);
  auto *bBN = llvm::dyn_cast<BroadcastNode>(add->getRHS().getNode());
  ASSERT_TRUE(bBN);
  EXPECT_EQ(bBN->getAxis(), 1);
  auto *XPH = llvm::dyn_cast<Placeholder>(mul->getLHS().getNode());
  EXPECT_EQ(XPH, mod.getPlaceholderByNameSlow("X"));
  auto *wBN = llvm::dyn_cast<BroadcastNode>(mul->getRHS().getNode());
  ASSERT_TRUE(wBN);
  EXPECT_EQ(wBN->getAxis(), 1);
  auto *wPH = llvm::dyn_cast<Placeholder>(wBN->getInput().getNode());
  EXPECT_EQ(wPH, mod.getPlaceholderByNameSlow("w"));
  auto *bPH = llvm::dyn_cast<Placeholder>(bBN->getInput().getNode());
  EXPECT_EQ(bPH, mod.getPlaceholderByNameSlow("b"));

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

  TypeRef indicesType = F->getParent()->uniqueType(ElemKind::Int64ITy, {8});
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

  bindings.allocate(indices)->getHandle<int64_t>() = {
      1, 0, 2, 0, 1, 2, 2, 0,
  };
  bindings.allocate(lengths)->getHandle<int32_t>() = {
      3,
      0,
      3,
      2,
  };

  // High level check on the content of the graph. We have 1 rowwise-quantized
  // SLWS and 1 save, along with 2 Slices and 2 Reshapes to extract out
  // scales/biases from the loaded Constant.
  EXPECT_EQ(F->getNodes().size(), 6);
  SaveNode *saveNode = getSaveNodeFromDest(output);
  RowwiseQuantizedSparseLengthsWeightedSumNode *RWQSLWS =
      llvm::dyn_cast<RowwiseQuantizedSparseLengthsWeightedSumNode>(
          saveNode->getInput().getNode());
  ASSERT_TRUE(RWQSLWS);
  // Check that the weights input is a Constant node.
  Constant *weights = llvm::dyn_cast<Constant>(RWQSLWS->getWeights().getNode());
  ASSERT_TRUE(weights);

  // Check that we have a Reshape(Slice(Constant)) for Scales/Offsets.
  ReshapeNode *reshapeScales =
      llvm::dyn_cast<ReshapeNode>(RWQSLWS->getScales());
  ASSERT_TRUE(reshapeScales);
  SliceNode *sliceScales = llvm::dyn_cast<SliceNode>(reshapeScales->getInput());
  ASSERT_TRUE(sliceScales);
  ReshapeNode *reshapeOffsets =
      llvm::dyn_cast<ReshapeNode>(RWQSLWS->getOffsets());
  ASSERT_TRUE(reshapeOffsets);
  SliceNode *sliceOffsets =
      llvm::dyn_cast<SliceNode>(reshapeOffsets->getInput());
  ASSERT_TRUE(sliceOffsets);
  EXPECT_EQ(sliceScales->getInput(), sliceOffsets->getInput());
  EXPECT_TRUE(llvm::isa<Constant>(sliceScales->getInput()));

  // We have 3 placeholders: 1 for save, and then indices and lengths.
  EXPECT_EQ(mod.getPlaceholders().size(), 3);

  // We have 3 constants: data, scales+offsets, and weights. Originally fused
  // data is no longer used and is removed by loader.
  EXPECT_EQ(mod.getConstants().size(), 3);

  EE.compile(CompilationMode::Infer);
  bindings.allocate(mod.getPlaceholders());

  // Post compile, should have folded the Slice and Reshape into the
  // Scales/Biases. Also, DCE should have gotten rid of the originally fused
  // data Constant, as it is no longer used.
  EXPECT_EQ(F->getNodes().size(), 2);
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

  TypeRef indicesType = F->getParent()->uniqueType(ElemKind::Int64ITy, {8});
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

  bindings.allocate(indices)->getHandle<int64_t>() = {
      2, 0, 1, 2, 0, 0, 0, 0,
  };
  bindings.allocate(lengths)->getHandle<int32_t>() = {
      2, 0, 2, 1, 3,
  };

  // High level check on the content of the graph. We have 1 rowwise-quantized
  // SLWS (which implements SLS), 1 Splat for the weights, and 1 save. For SLS
  // scales/bias, we have 2 Slices and 2 Reshapes to extract out scales/biases
  // from the loaded Constant.
  EXPECT_EQ(F->getNodes().size(), 7);
  SaveNode *saveNode = getSaveNodeFromDest(output);
  RowwiseQuantizedSparseLengthsWeightedSumNode *RWQSLS =
      llvm::dyn_cast<RowwiseQuantizedSparseLengthsWeightedSumNode>(
          saveNode->getInput().getNode());
  ASSERT_TRUE(RWQSLS);
  SplatNode *splatNode =
      llvm::dyn_cast<SplatNode>(RWQSLS->getWeights().getNode());
  ASSERT_TRUE(splatNode);
  EXPECT_EQ(splatNode->getValue(), 1.0f);

  // Check that we have a Reshape(Slice(Constant)) for Scales/Offsets.
  ReshapeNode *reshapeScales = llvm::dyn_cast<ReshapeNode>(RWQSLS->getScales());
  ASSERT_TRUE(reshapeScales);
  SliceNode *sliceScales = llvm::dyn_cast<SliceNode>(reshapeScales->getInput());
  ASSERT_TRUE(sliceScales);
  ReshapeNode *reshapeOffsets =
      llvm::dyn_cast<ReshapeNode>(RWQSLS->getOffsets());
  ASSERT_TRUE(reshapeOffsets);
  SliceNode *sliceOffsets =
      llvm::dyn_cast<SliceNode>(reshapeOffsets->getInput());
  ASSERT_TRUE(sliceOffsets);
  EXPECT_EQ(sliceScales->getInput(), sliceOffsets->getInput());
  EXPECT_TRUE(llvm::isa<Constant>(sliceScales->getInput()));

  // We have 3 placeholders: 1 for save, and then indices and lengths.
  EXPECT_EQ(mod.getPlaceholders().size(), 3);

  // We have 2 constants: Data and fused scales+offsets.
  EXPECT_EQ(mod.getConstants().size(), 2);

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
static void testFRWQSLWS(float avgLength) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename(
      std::isnan(avgLength) ? GLOW_DATA_PATH
          "tests/models/caffe2Models/"
          "fused_rowwise_quantized_sparse_lengths_weighted_sum_predict_net."
          "pbtxt"
                            : GLOW_DATA_PATH
          "tests/models/caffe2Models/"
          "fused_rowwise_quantized_sparse_lengths_weighted_sum_avg_length_"
          "predict_net.pbtxt");
  std::string NetWeightFilename(
      GLOW_DATA_PATH
      "tests/models/caffe2Models/"
      "fused_rowwise_quantized_sparse_lengths_weighted_sum_init_net.pbtxt");

  Placeholder *output, *indices, *lengths;
  PlaceholderBindings bindings;

  TypeRef indicesType = F->getParent()->uniqueType(ElemKind::Int64ITy, {8});
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

  bindings.allocate(indices)->getHandle<int64_t>() = {
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
  if (std::isnan(avgLength)) {
    EXPECT_TRUE(std::isnan(FRWQSLWS->getAvgLength()));
  } else {
    EXPECT_EQ(FRWQSLWS->getAvgLength(), avgLength);
  }
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

TEST_F(Caffe2ImporterTest, SparseLengthsWeightedSumFused8BitRowwise) {
  testFRWQSLWS(NAN);
}

TEST_F(Caffe2ImporterTest, SparseLengthsWeightedSumFused8BitRowwiseAvgLength) {
  testFRWQSLWS(5.0f);
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

  TypeRef indicesType = F->getParent()->uniqueType(ElemKind::Int64ITy, {8});
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

  bindings.allocate(indices)->getHandle<int64_t>() = {
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

/// Test loading SparseLengthsSumFused8BitRowwise with all lookup lengths equal
/// to one. This is created as a RowwiseQuantizedSparseLengthsWeightedSumNode
/// with `AllLengthsOne=true`. The following inputs/outputs are used/expected
/// for this test. Note that the DATA input is rowwise-quantized in the init_net
/// proto.
///    DATA  = [
///        [1.0, 1.2],
///        [2.3, 3.4],
///        [4.5, 5.7],
///    ]
///    INDICES = [2, 0, 1, 2, 0]
///    LENGTHS = [1, 1, 1, 1, 1]
///    OUTPUT = [
///        [4.5, 5.7],
///        [1.0, 1.2],
///        [2.3, 3.4],
///        [4.5, 5.7],
///        [1.0, 1.2],
///    ]
TEST_F(Caffe2ImporterTest, SparseLengthsSumFused8BitRowwiseAllLengthsOne) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename(
      GLOW_DATA_PATH
      "tests/models/caffe2Models/"
      "fused_rowwise_quantized_sparse_lengths_sum_predict_net_length1.pbtxt");
  std::string NetWeightFilename(
      GLOW_DATA_PATH
      "tests/models/caffe2Models/"
      "fused_rowwise_quantized_sparse_lengths_sum_init_net.pbtxt");

  Placeholder *output, *indices, *lengths;
  PlaceholderBindings bindings;

  TypeRef indicesType = F->getParent()->uniqueType(ElemKind::Int64ITy, {5});
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

  bindings.allocate(indices)->getHandle<int64_t>() = {
      2, 0, 1, 2, 0,
  };
  bindings.allocate(lengths)->getHandle<int32_t>() = {
      1, 1, 1, 1, 1,
  };

  // High level check on the content of the graph. We have 1 rowwise-quantized
  // SLS and 1 save.
  EXPECT_EQ(F->getNodes().size(), 2);
  SaveNode *saveNode = getSaveNodeFromDest(output);
  FusedRowwiseQuantizedSparseLengthsSumNode *FRWQSLS =
      llvm::dyn_cast<FusedRowwiseQuantizedSparseLengthsSumNode>(
          saveNode->getInput().getNode());
  ASSERT_TRUE(FRWQSLS);
  EXPECT_EQ(FRWQSLS->getLengthsMode(), LengthsMode::AllOne);
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
      4.5f, 5.7f, 1.0f, 1.2f, 2.3f, 3.4f, 4.5f, 5.7f, 1.0f, 1.2f,
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

  TypeRef indicesType = F->getParent()->uniqueType(ElemKind::Int64ITy, {8});
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
  const dim_t kRows = 10;
  const dim_t kCols = 5;
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
  // {transpose, transpose} => conv => relu => transpose => save
  EXPECT_EQ(F->getNodes().size(), 6);
  auto *saveNode = getSaveNodeFromDest(output);

  auto *transNode1 =
      llvm::dyn_cast<TransposeNode>(saveNode->getInput().getNode());
  ASSERT_TRUE(transNode1);
  auto *reluNode = llvm::dyn_cast<ReluNode>(transNode1->getInput().getNode());
  ASSERT_TRUE(reluNode);
  EXPECT_TRUE(reluNode->getResult().getType()->isQuantizedType());
  EXPECT_EQ(reluNode->getResult().getType()->getScale(), 1.5f);
  EXPECT_EQ(reluNode->getResult().getType()->getOffset(),
            7 - UINT8_TO_INT8_SHIFT);
  auto *convNode =
      llvm::dyn_cast<ConvolutionNode>(reluNode->getInput().getNode());
  ASSERT_TRUE(convNode);
  EXPECT_TRUE(convNode->getResult().getType()->isQuantizedType());
  EXPECT_EQ(convNode->getResult().getType()->getScale(), 1.5f);
  EXPECT_EQ(convNode->getResult().getType()->getOffset(),
            7 - UINT8_TO_INT8_SHIFT);
  EXPECT_TRUE(convNode->getFilter().getType()->isQuantizedType());
  EXPECT_EQ(convNode->getFilter().getType()->getScale(), 2.f);
  EXPECT_EQ(convNode->getFilter().getType()->getOffset(),
            10 - UINT8_TO_INT8_SHIFT);
  EXPECT_TRUE(convNode->getBias().getType()->isQuantizedType());
  EXPECT_EQ(convNode->getBias().getType()->getScale(), 10.f);
  // This one is loaded int32, so has no shift.
  EXPECT_EQ(convNode->getBias().getType()->getOffset(), 4);
  auto *transNode2 =
      llvm::dyn_cast<TransposeNode>(convNode->getInput().getNode());
  ASSERT_TRUE(transNode2);
  auto *transNode3 =
      llvm::dyn_cast<TransposeNode>(convNode->getFilter().getNode());
  ASSERT_TRUE(transNode3);

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
  EXPECT_TRUE(mod.getPlaceholderByNameSlow("sigmoid_test_output"));
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

/// \returns whether \p val is found in \p vec.
static bool vecContainsVal(const std::vector<runtime::DeviceIDTy> &vec,
                           runtime::DeviceIDTy val) {
  return std::find(vec.begin(), vec.end(), val) != vec.end();
}

/// Verify that different fill types are loaded with the correct types into
/// their respective partitions specified in the C2 proto.
TEST_F(Caffe2ImporterTest, PrePartitionedTensorFillsTest) {
  ExecutionEngine EE("Interpreter", /* deviceMemory (16GB) */ 0x400000000,
                     /* ignoreUserDeviceConfig */ false, /* numDevices */ 3);
  auto &mod = EE.getModule();

  std::string NetDescFilename(
      GLOW_DATA_PATH
      "tests/models/caffe2Models/pre_partitioned_fill_test_predict_net.pbtxt");
  std::string NetWeightFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/fill_test_init_net.pbtxt");

  Constant *tensorFillFloat, *tensorIntFill, *tensorInt64Fill,
      *tensorStringToUInt8Fill;

  // Destroy the loader after the graph is loaded since the following execution
  // will not depend on anything from the loader.
  runtime::PrePartitionedConfig PPC;
  {
    // Loaded protos must have at least one external output, so load an unused
    // output and type to satisfy it. It is named unused_output in
    // empty_predict_net.pbtxt.
    Type unusedTy = Type(ElemKind::FloatTy, {4});
    Caffe2ModelLoader caffe2LD(
        NetDescFilename, NetWeightFilename,
        {"tensor_fill_float_eq", "tensor_int_fill_eq", "tensor_int64_fill_eq",
         "tensor_string_to_uint8_fill_eq"},
        {&unusedTy, &unusedTy, &unusedTy, &unusedTy}, mod, "main", &PPC);
    tensorFillFloat = llvm::dyn_cast<Constant>(
        EXIT_ON_ERR(caffe2LD.getNodeValueByName("tensor_fill_float")));
    tensorIntFill = llvm::dyn_cast<Constant>(
        EXIT_ON_ERR(caffe2LD.getNodeValueByName("tensor_int_fill")));
    tensorInt64Fill = llvm::dyn_cast<Constant>(
        EXIT_ON_ERR(caffe2LD.getNodeValueByName("tensor_int64_fill")));
    tensorStringToUInt8Fill = llvm::dyn_cast<Constant>(EXIT_ON_ERR(
        caffe2LD.getNodeValueByName("tensor_string_to_uint8_fill")));
  }

  ASSERT_EQ(mod.getFunctions().size(), 3);
  Function *P0 = nullptr, *P1 = nullptr, *P2 = nullptr;
  for (size_t i = 0, e = PPC.funcs.size(); i < e; i++) {
    // Find the expected Function, and check that the logical device IDs were
    // correctly loaded.
    Function *F = PPC.funcs[i];
    if (F->getName() == "main_p0") {
      P0 = F;
      ASSERT_EQ(PPC.logicalIDs[i].size(), 2);
      EXPECT_TRUE(vecContainsVal(PPC.logicalIDs[i], 0));
      EXPECT_TRUE(vecContainsVal(PPC.logicalIDs[i], 2));
    } else if (F->getName() == "main_p1") {
      P1 = F;
      ASSERT_EQ(PPC.logicalIDs[i].size(), 1);
      EXPECT_TRUE(vecContainsVal(PPC.logicalIDs[i], 1));
    } else if (F->getName() == "main_p2") {
      P2 = F;
    } else {
      FAIL() << "Unknown Function found.";
      ASSERT_EQ(PPC.logicalIDs[i].size(), 1);
      EXPECT_TRUE(vecContainsVal(PPC.logicalIDs[i], 2));
    }

    // Check that the function was also found in the module.
    auto &modFuns = mod.getFunctions();
    ASSERT_NE(std::find(modFuns.begin(), modFuns.end(), F), modFuns.end());
  }
  ASSERT_TRUE(P0);
  ASSERT_TRUE(P1);
  ASSERT_TRUE(P2);

  ASSERT_TRUE(tensorFillFloat);
  ASSERT_TRUE(tensorIntFill);
  ASSERT_TRUE(tensorInt64Fill);
  ASSERT_TRUE(tensorStringToUInt8Fill);

  // Note: Only user is a no-op Reshape, which is fed into a Save.
  ASSERT_EQ(tensorFillFloat->getNumUsers(), 1);
  ASSERT_EQ(tensorIntFill->getNumUsers(), 1);
  ASSERT_EQ(tensorInt64Fill->getNumUsers(), 1);
  ASSERT_EQ(tensorStringToUInt8Fill->getNumUsers(), 1);

  // Check that the parent Functions of the Reshapes match expected partitions.
  EXPECT_EQ(tensorFillFloat->getUsers().front().getUser()->getParent(), P0);
  EXPECT_EQ(tensorIntFill->getUsers().front().getUser()->getParent(), P1);
  EXPECT_EQ(tensorInt64Fill->getUsers().front().getUser()->getParent(), P2);
  EXPECT_EQ(tensorStringToUInt8Fill->getUsers().front().getUser()->getParent(),
            P0);

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

  CompilationContext cctx;
  cctx.prepartitionedConfig = &PPC;
  EE.compile(cctx);
  PlaceholderBindings bindings;
  bindings.allocate(mod.getPlaceholders());
  EE.run(bindings);
}

/// Verify that multiple ops loaded into different pre-partitioned Functions
/// with a non-trivial dependence between them works correctly.
/// Note: DAG of the partitions looks like: F0 -> F1
///                                           \   |
///                                            v  v
///                                             F2
TEST_F(Caffe2ImporterTest, PrePartitionedMultiOpTest) {
  ExecutionEngine EE("Interpreter", /* deviceMemory (16GB) */ 0x400000000,
                     /* ignoreUserDeviceConfig */ false, /* numDevices */ 3);
  auto &mod = EE.getModule();

  const std::string NetDescFilename(
      GLOW_DATA_PATH
      "tests/models/caffe2Models/pre_partitioned_multi_op_predict_net.pbtxt");
  const std::string NetWeightFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/empty_init_net.pbtxt");

  Placeholder *outputPH;
  Tensor *resultPartitionedT;
  PlaceholderBindings bindingsU;
  PlaceholderBindings bindingsP;

  // Destroy the loader after the graph is loaded since the following execution
  // will not depend on anything from the loader.
  runtime::PrePartitionedConfig PPC;
  Tensor mmIn0T(ElemKind::FloatTy, {10, 10});
  Tensor mmIn1T(ElemKind::FloatTy, {10, 10});
  Tensor addInT(ElemKind::FloatTy, {10, 10});
  mmIn0T.getHandle().randomize(-3.0, 3.0, mod.getPRNG());
  mmIn1T.getHandle().randomize(-3.0, 3.0, mod.getPRNG());
  addInT.getHandle().randomize(-3.0, 3.0, mod.getPRNG());
  Placeholder *mmIn0P = nullptr, *mmIn1P = nullptr, *addInP = nullptr;
  {
    Caffe2ModelLoader caffe2LD(
        NetDescFilename, NetWeightFilename, {"mm0_in", "mm1_in", "add_in"},
        {&mmIn0T.getType(), &mmIn1T.getType(), &addInT.getType()}, mod, "main",
        &PPC);
    outputPH = EXIT_ON_ERR(caffe2LD.getSingleOutput());
    NodeValue mmIn0NV;
    ASSIGN_VALUE_OR_FAIL_TEST(mmIn0NV, caffe2LD.getNodeValueByName("mm0_in"));
    mmIn0P = llvm::dyn_cast<Placeholder>(mmIn0NV);
    NodeValue mmIn1NV;
    ASSIGN_VALUE_OR_FAIL_TEST(mmIn1NV, caffe2LD.getNodeValueByName("mm1_in"));
    mmIn1P = llvm::dyn_cast<Placeholder>(mmIn1NV);
    NodeValue addInNV;
    ASSIGN_VALUE_OR_FAIL_TEST(addInNV, caffe2LD.getNodeValueByName("add_in"));
    addInP = llvm::dyn_cast<Placeholder>(addInNV);
  }

  // First we are going to make sure the structure of the pre-partitioned Module
  // is set up as expected, and run it with random inputs to get some results.
  {
    ASSERT_TRUE(mmIn0P);
    ASSERT_TRUE(mmIn1P);
    ASSERT_TRUE(addInP);

    ASSERT_EQ(mod.getFunctions().size(), 3);
    Function *P0 = nullptr, *P1 = nullptr, *P2 = nullptr;
    for (size_t i = 0, e = PPC.funcs.size(); i < e; i++) {
      // Find the expected Function, and check that the logical device IDs were
      // correctly loaded.
      Function *F = PPC.funcs[i];
      if (F->getName() == "main_p0") {
        P0 = F;
        ASSERT_EQ(PPC.logicalIDs[i].size(), 1);
        EXPECT_TRUE(vecContainsVal(PPC.logicalIDs[i], 2));
        EXPECT_EQ(PPC.backendSpecificOpts[i].size(), 0);
      } else if (F->getName() == "main_p1") {
        P1 = F;
        ASSERT_EQ(PPC.logicalIDs[i].size(), 2);
        EXPECT_TRUE(vecContainsVal(PPC.logicalIDs[i], 0));
        EXPECT_TRUE(vecContainsVal(PPC.logicalIDs[i], 1));
        EXPECT_EQ(PPC.backendSpecificOpts[i].size(), 0);
      } else if (F->getName() == "main_p2") {
        P2 = F;
        ASSERT_EQ(PPC.logicalIDs[i].size(), 1);
        EXPECT_TRUE(vecContainsVal(PPC.logicalIDs[i], 2));
        EXPECT_EQ(PPC.backendSpecificOpts[i].size(), 3);
        ASSERT_TRUE(PPC.backendSpecificOpts[i].count("BackendA_opt1"));
        EXPECT_EQ(PPC.backendSpecificOpts[i].at("BackendA_opt1"), "val1");
        ASSERT_TRUE(PPC.backendSpecificOpts[i].count("BackendA_opt2"));
        EXPECT_EQ(PPC.backendSpecificOpts[i].at("BackendA_opt2"), "val2");
        ASSERT_TRUE(PPC.backendSpecificOpts[i].count("BackendB_opt3"));
        EXPECT_EQ(PPC.backendSpecificOpts[i].at("BackendB_opt3"), "val3");
      } else {
        FAIL() << "Unknown Function found.";
      }

      // Check that the function was also found in the module.
      auto &modFuns = mod.getFunctions();
      ASSERT_NE(std::find(modFuns.begin(), modFuns.end(), F), modFuns.end());
    }
    ASSERT_TRUE(P0);
    ASSERT_TRUE(P1);
    ASSERT_TRUE(P2);

    // Verify P0:
    auto *finalSave = getSaveNodeFromDest(outputPH);
    ASSERT_TRUE(finalSave);
    EXPECT_EQ(finalSave->getParent(), P0);
    SubNode *sub = llvm::dyn_cast<SubNode>(finalSave->getInput());
    ASSERT_TRUE(sub);
    Placeholder *intermedAddOut = llvm::dyn_cast<Placeholder>(sub->getRHS());
    ASSERT_TRUE(intermedAddOut);
    MulNode *mul = llvm::dyn_cast<MulNode>(sub->getLHS());
    ASSERT_TRUE(mul);
    Placeholder *intermedMMOut = llvm::dyn_cast<Placeholder>(mul->getRHS());
    ASSERT_TRUE(intermedMMOut);
    Placeholder *mmIn0 = llvm::dyn_cast<Placeholder>(mul->getLHS());
    ASSERT_TRUE(mmIn0);

    // Verify P2:
    Node *userFromP2 = nullptr;
    for (auto &U : intermedAddOut->getUsers()) {
      if (U.getUser()->getParent() == P2) {
        ASSERT_FALSE(userFromP2);
        userFromP2 = U.getUser();
      }
    }
    ASSERT_TRUE(userFromP2);
    SaveNode *saveIntermedP2Out = llvm::dyn_cast<SaveNode>(userFromP2);
    ASSERT_TRUE(saveIntermedP2Out);
    AddNode *add = llvm::dyn_cast<AddNode>(saveIntermedP2Out->getInput());
    ASSERT_TRUE(add);
    Placeholder *addIn = llvm::dyn_cast<Placeholder>(add->getRHS());
    ASSERT_TRUE(addIn);
    EXPECT_EQ(add->getLHS().getNode(), intermedMMOut);

    // Verify P1:
    Node *userFromP1 = nullptr;
    for (auto &U : intermedMMOut->getUsers()) {
      if (U.getUser()->getParent() == P1) {
        ASSERT_FALSE(userFromP1);
        userFromP1 = U.getUser();
      }
    }
    ASSERT_TRUE(userFromP1);
    SaveNode *saveIntermedP1Out = llvm::dyn_cast<SaveNode>(userFromP1);
    ASSERT_TRUE(saveIntermedP1Out);
    MatMulNode *matMul =
        llvm::dyn_cast<MatMulNode>(saveIntermedP1Out->getInput());
    ASSERT_TRUE(matMul);
    EXPECT_EQ(matMul->getLHS().getNode(), mmIn0);
    Placeholder *matMulIn = llvm::dyn_cast<Placeholder>(matMul->getRHS());
    ASSERT_TRUE(matMulIn);

    // Now that we've verifed the shape of the Module, run it and keep around
    // the pointer to the result.
    CompilationContext cctx;
    cctx.prepartitionedConfig = &PPC;
    EE.compile(cctx);
    bindingsP.insert(mmIn0P, mmIn0T.getUnowned());
    bindingsP.insert(mmIn1P, mmIn1T.getUnowned());
    bindingsP.insert(addInP, addInT.getUnowned());
    bindingsP.allocate(mod.getPlaceholders());
    EE.run(bindingsP);

    resultPartitionedT = bindingsP.get(outputPH);
  }

  // Now that we have the model result from pre-partitioned execution, execute
  // the model ignoring the pre-partitioning and bitwise compare results.
  EE.setBackendName(EE.getBackendName());

  Module &modU = EE.getModule();
  {
    Function *F = modU.createFunction("main");
    Caffe2ModelLoader caffe2LD(
        NetDescFilename, NetWeightFilename, {"mm0_in", "mm1_in", "add_in"},
        {&mmIn0T.getType(), &mmIn1T.getType(), &addInT.getType()}, *F);
    outputPH = EXIT_ON_ERR(caffe2LD.getSingleOutput());
    NodeValue mmIn0NV;
    ASSIGN_VALUE_OR_FAIL_TEST(mmIn0NV, caffe2LD.getNodeValueByName("mm0_in"));
    mmIn0P = llvm::dyn_cast<Placeholder>(mmIn0NV);
    NodeValue mmIn1NV;
    ASSIGN_VALUE_OR_FAIL_TEST(mmIn1NV, caffe2LD.getNodeValueByName("mm1_in"));
    mmIn1P = llvm::dyn_cast<Placeholder>(mmIn1NV);
    NodeValue addInNV;
    ASSIGN_VALUE_OR_FAIL_TEST(addInNV, caffe2LD.getNodeValueByName("add_in"));
    addInP = llvm::dyn_cast<Placeholder>(addInNV);
  }

  Tensor *resultUnpartitonedT;

  {
    ASSERT_TRUE(mmIn0P);
    ASSERT_TRUE(mmIn1P);
    ASSERT_TRUE(addInP);
    ASSERT_EQ(modU.getFunctions().size(), 1);

    EE.compile(CompilationMode::Infer);
    bindingsU.insert(mmIn0P, mmIn0T.getUnowned());
    bindingsU.insert(mmIn1P, mmIn1T.getUnowned());
    bindingsU.insert(addInP, addInT.getUnowned());
    bindingsU.allocate(modU.getPlaceholders());
    EE.run(bindingsU);

    resultUnpartitonedT = bindingsU.get(outputPH);
  }

  EXPECT_TRUE(resultPartitionedT->isBitwiseEqual(*resultUnpartitonedT,
                                                 /* verbose */ true));
}

/// Test importing a Caffe2 LayerNorm without weights and bias provided but with
/// epsilon or axis.
TEST_F(Caffe2ImporterTest, importLayerNormNoWeightBias) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/layernorm_pred_net.pbtxt");
  std::string NetWeightFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/empty_init_net.pbtxt");

  Placeholder *output;
  PlaceholderBindings bindings;

  const ShapeVector inShape({4, 2, 5, 5});

  // Destroy the loader after the graph is loaded since the following execution
  // will not depend on anything from the loader.
  {
    Tensor data(ElemKind::FloatTy, inShape);
    data.getHandle().randomize(-3.0, 3.0, mod.getPRNG());
    Caffe2ModelLoader caffe2LD(NetDescFilename, NetWeightFilename, {"input"},
                               {&data.getType()}, *F);
    output = EXIT_ON_ERR(caffe2LD.getSingleOutput());

    bindings.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(bindings, &mod, {"input"}, {&data});
  }

  // High level check on the content of the graph. We should have
  // {Placeholder, Splat, Splat} => LayerNorm => Save
  EXPECT_EQ(F->getNodes().size(), 4);
  SaveNode *save = getSaveNodeFromDest(output);

  auto *LN = llvm::dyn_cast<LayerNormalizationNode>(save->getInput().getNode());
  ASSERT_TRUE(LN);
  EXPECT_EQ(LN->getEpsilon(), 0.05f);
  EXPECT_TRUE(LN->getInput().dims().equals(inShape));
  EXPECT_TRUE(LN->getResult().dims().equals(inShape));

  auto *scale = llvm::dyn_cast<SplatNode>(LN->getScale().getNode());
  ASSERT_TRUE(scale);
  EXPECT_EQ(scale->getValue(), 1.0f);

  auto *bias = llvm::dyn_cast<SplatNode>(LN->getBias().getNode());
  ASSERT_TRUE(bias);
  EXPECT_EQ(bias->getValue(), 0.0f);

  // Axis is 2, so check shape with second and third dims of inShape.
  EXPECT_TRUE(scale->getResult().dims().equals({inShape[2], inShape[3]}));
  EXPECT_TRUE(bias->getResult().dims().equals({inShape[2], inShape[3]}));

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);
}

/// Test importing a Caffe2 LayerNorm with weights and bias provided but no
/// epsilon or axis.
TEST_F(Caffe2ImporterTest, importLayerNormWithWeightBias) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename(
      GLOW_DATA_PATH
      "tests/models/caffe2Models/layernorm_weight_bias_pred_net.pbtxt");
  std::string NetWeightFilename(
      GLOW_DATA_PATH
      "tests/models/caffe2Models/layernorm_weight_bias_init_net.pbtxt");

  Placeholder *output;
  PlaceholderBindings bindings;

  const ShapeVector inShape({5, 4, 3});

  // Destroy the loader after the graph is loaded since the following execution
  // will not depend on anything from the loader.
  {
    Tensor data(ElemKind::FloatTy, inShape);
    data.getHandle().randomize(-3.0, 3.0, mod.getPRNG());
    Caffe2ModelLoader caffe2LD(NetDescFilename, NetWeightFilename, {"input"},
                               {&data.getType()}, *F);
    output = EXIT_ON_ERR(caffe2LD.getSingleOutput());

    bindings.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(bindings, &mod, {"input"}, {&data});
  }

  // High level check on the content of the graph. We should have
  // {Placeholder, Constant, Constant} => LayerNorm => Save
  EXPECT_EQ(F->getNodes().size(), 2);
  SaveNode *save = getSaveNodeFromDest(output);

  auto *LN = llvm::dyn_cast<LayerNormalizationNode>(save->getInput().getNode());
  ASSERT_TRUE(LN);
  EXPECT_EQ(LN->getEpsilon(), 0.001f); // Caffe2 default.
  EXPECT_TRUE(LN->getInput().dims().equals(inShape));
  EXPECT_TRUE(LN->getResult().dims().equals(inShape));

  auto *scale = llvm::dyn_cast<Constant>(LN->getScale().getNode());
  ASSERT_TRUE(scale);

  auto *bias = llvm::dyn_cast<Constant>(LN->getBias().getNode());
  ASSERT_TRUE(bias);

  // Default axis is 1 and it was unspecified in the input proto, so check shape
  // with first and second dims of inShape.
  EXPECT_TRUE(scale->getOutput().dims().equals({inShape[1], inShape[2]}));
  EXPECT_TRUE(bias->getOutput().dims().equals({inShape[1], inShape[2]}));

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);
}

/// Test importing a Caffe2 LayerNorm with negative axis
TEST_F(Caffe2ImporterTest, importLayerNormNegativeAxis) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename(
      GLOW_DATA_PATH
      "tests/models/caffe2Models/layernorm_neg_axis_pred_net.pbtxt");
  std::string NetWeightFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/empty_init_net.pbtxt");

  Placeholder *output;
  PlaceholderBindings bindings;

  const ShapeVector inShape({4, 2, 5, 5});

  // Destroy the loader after the graph is loaded since the following execution
  // will not depend on anything from the loader.
  {
    Tensor data(ElemKind::FloatTy, inShape);
    data.getHandle().randomize(-3.0, 3.0, mod.getPRNG());
    Caffe2ModelLoader caffe2LD(NetDescFilename, NetWeightFilename, {"input"},
                               {&data.getType()}, *F);
    output = EXIT_ON_ERR(caffe2LD.getSingleOutput());

    bindings.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(bindings, &mod, {"input"}, {&data});
  }

  // High level check on the content of the graph. We should have
  // {Placeholder, Splat, Splat} => LayerNorm => Save
  EXPECT_EQ(F->getNodes().size(), 4);
  SaveNode *save = getSaveNodeFromDest(output);

  auto *LN = llvm::dyn_cast<LayerNormalizationNode>(save->getInput().getNode());
  ASSERT_TRUE(LN);
  EXPECT_EQ(LN->getEpsilon(), 0.05f);
  EXPECT_TRUE(LN->getInput().dims().equals(inShape));
  EXPECT_TRUE(LN->getResult().dims().equals(inShape));

  auto *scale = llvm::dyn_cast<SplatNode>(LN->getScale().getNode());
  ASSERT_TRUE(scale);
  EXPECT_EQ(scale->getValue(), 1.0f);

  auto *bias = llvm::dyn_cast<SplatNode>(LN->getBias().getNode());
  ASSERT_TRUE(bias);
  EXPECT_EQ(bias->getValue(), 0.0f);

  // Axis is -2, so check shape with second and third dims of inShape.
  EXPECT_TRUE(scale->getResult().dims().equals({inShape[2], inShape[3]}));
  EXPECT_TRUE(bias->getResult().dims().equals({inShape[2], inShape[3]}));

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);
}

static void testImportTrackedQParams(bool loadUniquedDummyQParams) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/int8convrelu_pred_net.pbtxt");
  std::string NetWeightFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/int8convrelu_init_net.pbtxt");

  Placeholder *output;
  PlaceholderBindings bindings;
  OriginNameToTQPMap originNameToTQPMap;

  // Destroy the loader after the graph is loaded since the following execution
  // will not depend on anything from the loader.
  {
    Tensor data(ElemKind::Int8QTy, {1, 1, 3, 3}, 1, 0);
    Caffe2ModelLoader caffe2LD(NetDescFilename, NetWeightFilename,
                               {"gpu_0/data_0"}, {&data.getType()}, *F,
                               /* errPtr */ nullptr, &originNameToTQPMap,
                               loadUniquedDummyQParams);
    output = EXIT_ON_ERR(caffe2LD.getSingleOutput());

    bindings.allocate(mod.getPlaceholders());
  }

  // High level check on the content of the graph. We should have
  // {transpose, transpose} => conv => relu => transpose => save
  EXPECT_EQ(F->getNodes().size(), 6);
  auto *saveNode = getSaveNodeFromDest(output);

  EXPECT_EQ(originNameToTQPMap.size(), 4);
  TensorQuantizationParams convOut, convBias, convWeight, convInput;
  for (const auto &nameTQP : originNameToTQPMap) {
    if (nameTQP.first == "conv_out") {
      convOut = nameTQP.second;
    } else if (nameTQP.first == "conv_w") {
      convWeight = nameTQP.second;
    } else if (nameTQP.first == "conv_b") {
      convBias = nameTQP.second;
    } else if (nameTQP.first == "gpu_0/data_0") {
      convInput = nameTQP.second;
    } else {
      FAIL();
    }
  }

  if (loadUniquedDummyQParams) {
    // Dummies should have unique offsets 0->3.
    EXPECT_EQ(convInput.offset, 0);
    EXPECT_EQ(convWeight.offset, 1);
    EXPECT_EQ(convBias.offset, 2);
    EXPECT_EQ(convOut.offset, 3);

    // All dummmies should have dummy scale.
    EXPECT_EQ(convInput.scale, dummyScale);
    EXPECT_EQ(convWeight.scale, dummyScale);
    EXPECT_EQ(convBias.scale, dummyScale);
    EXPECT_EQ(convOut.scale, dummyScale);
  } else {
    // This one was provided as an input PH with a type already based on Glow
    // Int8QTy, so don't shift.
    EXPECT_EQ(convInput.offset, 0);
    EXPECT_EQ(convWeight.offset, 10 - UINT8_TO_INT8_SHIFT);
    // This one is loaded int32, so has no shift.
    EXPECT_EQ(convBias.offset, 4);
    EXPECT_EQ(convOut.offset, 7 - UINT8_TO_INT8_SHIFT);

    EXPECT_EQ(convInput.scale, 1.f);
    EXPECT_EQ(convWeight.scale, 2.f);
    EXPECT_EQ(convBias.scale, 10.f);
    EXPECT_EQ(convOut.scale, 1.5f);
  }

  auto *transNode1 =
      llvm::dyn_cast<TransposeNode>(saveNode->getInput().getNode());
  ASSERT_TRUE(transNode1);
  auto *reluNode = llvm::dyn_cast<ReluNode>(transNode1->getInput().getNode());
  ASSERT_TRUE(reluNode);
  ASSERT_TRUE(reluNode);
  EXPECT_TRUE(reluNode->getResult().getType()->isQuantizedType());
  EXPECT_EQ(reluNode->getResult().getType()->getScale(), convOut.scale);
  EXPECT_EQ(reluNode->getResult().getType()->getOffset(), convOut.offset);
  auto *convNode =
      llvm::dyn_cast<ConvolutionNode>(reluNode->getInput().getNode());
  ASSERT_TRUE(convNode);
  EXPECT_TRUE(convNode->getResult().getType()->isQuantizedType());
  EXPECT_EQ(convNode->getResult().getType()->getScale(), convOut.scale);
  EXPECT_EQ(convNode->getResult().getType()->getOffset(), convOut.offset);
  EXPECT_TRUE(convNode->getFilter().getType()->isQuantizedType());
  EXPECT_EQ(convNode->getFilter().getType()->getScale(), convWeight.scale);
  EXPECT_EQ(convNode->getFilter().getType()->getOffset(), convWeight.offset);
  EXPECT_TRUE(convNode->getBias().getType()->isQuantizedType());
  EXPECT_EQ(convNode->getBias().getType()->getScale(), convBias.scale);
  EXPECT_EQ(convNode->getBias().getType()->getOffset(), convBias.offset);
  ASSERT_TRUE(convNode);
  auto *transNode2 =
      llvm::dyn_cast<TransposeNode>(convNode->getInput().getNode());
  ASSERT_TRUE(transNode2);
  auto *transNode3 =
      llvm::dyn_cast<TransposeNode>(convNode->getFilter().getNode());
  ASSERT_TRUE(transNode3);

  EE.compile(CompilationMode::Infer);
}

/// Test that when we load a pre-quantized model when providing
/// OriginNameToTQPMap that the quant params are discarded and unique offsets
/// are used to track the mapping to names they came from.
TEST_F(Caffe2ImporterTest, importInt8ConvReluTrackedDummyQParams) {
  testImportTrackedQParams(/* loadUniquedDummyQParams */ true);
}

/// Test that when we load a pre-quantized model when providing
/// OriginNameToTQPMap, but we don't enable loading unique dummy qparams, that
/// we correctly have mapped the quant params to the name it came from.
TEST_F(Caffe2ImporterTest, importInt8ConvReluTrackedRealQParams) {
  testImportTrackedQParams(/* loadUniquedDummyQParams */ false);
}

/// Check that we clip a Node with 0.f scale to kMinScaleFP16 correctly.
TEST_F(Caffe2ImporterTest, ClipZeroScaleFP16QuantOpt) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename(
      GLOW_DATA_PATH
      "tests/models/caffe2Models/int8sumrelu_tiny_scale_pred_net.pbtxt");
  std::string NetWeightFilename(
      GLOW_DATA_PATH
      "tests/models/caffe2Models/int8sumrelu_tiny_scale_init_net.pbtxt");

  Placeholder *output;
  PlaceholderBindings bindings;

  // Destroy the loader after the graph is loaded since the following execution
  // will not depend on anything from the loader.
  {
    Tensor data(ElemKind::Int8QTy, {4, 2}, 1.f, 0);
    Caffe2ModelLoader caffe2LD(
        NetDescFilename, NetWeightFilename, {"gpu_0/data_0"}, {&data.getType()},
        *F, /* errPtr */ nullptr, /* originNameToTQPMap */ nullptr,
        /* loadUniquedDummyQParams */ false,
        /* zeroScaleFP16Clip */ true);
    output = EXIT_ON_ERR(caffe2LD.getSingleOutput());

    bindings.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(bindings, &mod, {"gpu_0/data_0"}, {&data});
  }

  // High level check on the content of the graph. We should have
  // input-=> add => relu => save
  // const/
  EXPECT_EQ(F->getNodes().size(), 3);
  auto *save = getSaveNodeFromDest(output);

  // Verify that the structure is as expected *except* that the tiny scales that
  // are loaded have been replaced by kMinScaleFP16.
  auto *relu = llvm::dyn_cast<ReluNode>(save->getInput().getNode());
  ASSERT_TRUE(relu);
  EXPECT_EQ(relu->getResult().getType()->getScale(), kMinScaleFP16);
  EXPECT_EQ(relu->getResult().getType()->getOffset(), 5 - UINT8_TO_INT8_SHIFT);
  auto *add = llvm::dyn_cast<AddNode>(relu->getInput().getNode());
  ASSERT_TRUE(add);
  EXPECT_EQ(add->getResult().getType()->getScale(), kMinScaleFP16);
  EXPECT_EQ(add->getResult().getType()->getOffset(), 5 - UINT8_TO_INT8_SHIFT);
  auto *input = llvm::dyn_cast<Placeholder>(add->getLHS().getNode());
  ASSERT_TRUE(input);
  auto *val = llvm::dyn_cast<Constant>(add->getRHS().getNode());
  ASSERT_TRUE(val);
  EXPECT_EQ(val->getOutput().getType()->getScale(), kMinScaleFP16);
  EXPECT_EQ(val->getOutput().getType()->getOffset(), 13 - UINT8_TO_INT8_SHIFT);

  EE.compile(CompilationMode::Infer);
}

/// Check that we clip a Node with 0.f scale to kMinScaleFP16 correctly.
TEST_F(Caffe2ImporterTest, ClipLargeQRangeToFP16) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename(
      GLOW_DATA_PATH
      "tests/models/caffe2Models/int8sumrelu_large_range_pred_net.pbtxt");
  std::string NetWeightFilename(
      GLOW_DATA_PATH
      "tests/models/caffe2Models/int8sumrelu_large_range_init_net.pbtxt");

  Placeholder *output;
  PlaceholderBindings bindings;

  // Destroy the loader after the graph is loaded since the following execution
  // will not depend on anything from the loader.
  {
    Tensor data(ElemKind::FloatTy, {4, 2});
    Caffe2ModelLoader caffe2LD(
        NetDescFilename, NetWeightFilename, {"gpu_0/data_0"}, {&data.getType()},
        *F, /* errPtr */ nullptr, /* originNameToTQPMap */ nullptr,
        /* loadUniquedDummyQParams */ false,
        /* zeroScaleFP16Clip */ false, /* clipQuantRangeToFP16 */ true);
    output = EXIT_ON_ERR(caffe2LD.getSingleOutput());

    bindings.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(bindings, &mod, {"gpu_0/data_0"}, {&data});
  }

  // High level check on the content of the graph. We should have
  // input=> quant-=> add => relu => save
  //         const/
  EXPECT_EQ(F->getNodes().size(), 4);
  auto *save = getSaveNodeFromDest(output);

  auto clippedQP = quantization::chooseQuantizationParams({kMinFP16, kMaxFP16});

  // Verify that the structure is as expected *except* that the range is
  // adjusted for clipping to fp16.
  auto *relu = llvm::dyn_cast<ReluNode>(save->getInput().getNode());
  ASSERT_TRUE(relu);
  EXPECT_EQ(relu->getResult().getType()->getScale(), clippedQP.scale);
  EXPECT_EQ(relu->getResult().getType()->getOffset(), clippedQP.offset);
  auto *add = llvm::dyn_cast<AddNode>(relu->getInput().getNode());
  ASSERT_TRUE(add);
  EXPECT_EQ(add->getResult().getType()->getScale(), clippedQP.scale);
  EXPECT_EQ(add->getResult().getType()->getOffset(), clippedQP.offset);
  auto *quant = llvm::dyn_cast<QuantizeNode>(add->getLHS().getNode());
  ASSERT_TRUE(quant);
  EXPECT_EQ(quant->getResult().getType()->getScale(), 1.f);
  EXPECT_EQ(quant->getResult().getType()->getOffset(), 0 - UINT8_TO_INT8_SHIFT);
  EXPECT_TRUE(llvm::isa<Placeholder>(quant->getInput().getNode()));
  auto *C = llvm::dyn_cast<Constant>(add->getRHS().getNode());
  ASSERT_TRUE(C);
  EXPECT_EQ(C->getOutput().getType()->getScale(), clippedQP.scale);
  EXPECT_EQ(C->getOutput().getType()->getOffset(), clippedQP.offset);

  EE.compile(CompilationMode::Infer);
}

// Here we use a shape that is provided as an argument
TEST_F(Caffe2ImporterTest, gaussianFillUseProvidedShape) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename(
      GLOW_DATA_PATH
      "tests/models/caffe2Models/gaussian_fill_use_provided_shape.pbtxt");
  std::string NetWeightFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/empty_init_net.pbtxt");

  PlaceholderBindings bindings;

  Placeholder *output;
  // Destroy the loader after the graph is loaded since the following execution
  // will not depend on anything from the loader.
  {
    Caffe2ModelLoader caffe2LD(NetDescFilename, NetWeightFilename, {}, {}, *F);
    output = EXIT_ON_ERR(caffe2LD.getSingleOutput());

    bindings.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(bindings, &mod, {}, {});
  }

  // Shape is defined in .pbtxt file
  const std::vector<dim_t> expectedShape{4, 5, 6, 7};
  EXPECT_EQ(expectedShape, output->dims().vec());

  auto res = bindings.get(output);
  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  auto result = res->getHandle();
  EXPECT_NEAR(0, result.calculateMeanVariance().first, 3);
}

// Here we use expect input to be 1D tensor and act like shape
TEST_F(Caffe2ImporterTest, gaussianFillInputAsShape) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename(
      GLOW_DATA_PATH
      "tests/models/caffe2Models/gaussian_fill_input_as_shape.pbtxt");
  std::string NetWeightFilename(
      GLOW_DATA_PATH
      "tests/models/caffe2Models/gaussian_fill_input_as_shape_init.pbtxt");

  PlaceholderBindings bindings;

  Placeholder *output;
  // Destroy the loader after the graph is loaded since the following execution
  // will not depend on anything from the loader.
  {
    Caffe2ModelLoader caffe2LD(NetDescFilename, NetWeightFilename, {}, {}, *F);
    output = EXIT_ON_ERR(caffe2LD.getSingleOutput());

    bindings.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(bindings, &mod, {}, {});
  }

  // Shape is defined in .pbtxt file
  const std::vector<dim_t> expectedShape{4, 5, 6, 7};
  EXPECT_EQ(expectedShape, output->dims().vec());

  auto res = bindings.get(output);
  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  auto result = res->getHandle();
  EXPECT_NEAR(0, result.calculateMeanVariance().first, 3);
}

// Here we use input's shape for the result
TEST_F(Caffe2ImporterTest, gaussianFillUseInputShape) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename(
      GLOW_DATA_PATH
      "tests/models/caffe2Models/gaussian_fill_use_input_shape.pbtxt");
  std::string NetWeightFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/empty_init_net.pbtxt");

  PlaceholderBindings bindings;

  Placeholder *output;
  const std::vector<dim_t> inputShape{4, 5, 6, 7};
  Tensor inputs_0(ElemKind::Int64ITy, inputShape);
  inputs_0.getHandle<int64_t>().randomize(-10, 10, mod.getPRNG());
  // Destroy the loader after the graph is loaded since the following execution
  // will not depend on anything from the loader.
  {
    Caffe2ModelLoader caffe2LD(NetDescFilename, NetWeightFilename, {"inputs_0"},
                               {&inputs_0.getType()}, *F);
    output = EXIT_ON_ERR(caffe2LD.getSingleOutput());

    bindings.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(bindings, &mod, {"inputs_0"}, {&inputs_0});
  }

  // Check that the shape of the output matches what Caffe2 expects.
  EXPECT_EQ(inputShape, output->dims().vec());

  auto res = bindings.get(output);
  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  auto result = res->getHandle();
  EXPECT_NEAR(0, result.calculateMeanVariance().first, 3);
}

// Set the extra_shape argument with input provided
TEST_F(Caffe2ImporterTest, gaussianFillExtraShape) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/"
                     "gaussian_fill_use_input_shape_with_extra_shape.pbtxt");
  std::string NetWeightFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/empty_init_net.pbtxt");

  PlaceholderBindings bindings;

  Placeholder *output;
  const std::vector<dim_t> inputShape{4, 5, 6, 7};
  Tensor inputs_0(ElemKind::Int64ITy, inputShape);
  inputs_0.getHandle<int64_t>().randomize(-10, 10, mod.getPRNG());
  // Destroy the loader after the graph is loaded since the following execution
  // will not depend on anything from the loader.
  {
    Caffe2ModelLoader caffe2LD(NetDescFilename, NetWeightFilename, {"inputs_0"},
                               {&inputs_0.getType()}, *F);
    output = EXIT_ON_ERR(caffe2LD.getSingleOutput());

    bindings.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(bindings, &mod, {"inputs_0"}, {&inputs_0});
  }

  // Check that the shape of the output matches what Caffe2 expects.
  const std::vector<dim_t> expectedShape{4, 5, 6, 7, 2, 3};
  EXPECT_EQ(expectedShape, output->dims().vec());

  auto res = bindings.get(output);
  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  auto result = res->getHandle();
  EXPECT_NEAR(0, result.calculateMeanVariance().first, 3);
}

// Here we use expect input to be 1D tensor and act like shape
TEST_F(Caffe2ImporterTest, uniformFillInputAsShape) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename(
      GLOW_DATA_PATH
      "tests/models/caffe2Models/uniform_fill_input_as_shape.pbtxt");
  std::string NetWeightFilename(
      GLOW_DATA_PATH
      "tests/models/caffe2Models/gaussian_fill_input_as_shape_init.pbtxt");

  PlaceholderBindings bindings;

  Placeholder *output;
  // Destroy the loader after the graph is loaded since the following execution
  // will not depend on anything from the loader.
  {
    Caffe2ModelLoader caffe2LD(NetDescFilename, NetWeightFilename, {}, {}, *F);
    output = EXIT_ON_ERR(caffe2LD.getSingleOutput());

    bindings.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(bindings, &mod, {}, {});
  }

  // Shape is defined in .pbtxt file
  const std::vector<dim_t> expectedShape{4, 5, 6, 7};
  EXPECT_EQ(expectedShape, output->dims().vec());

  auto res = bindings.get(output);
  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  auto result = res->getHandle();
  for (dim_t dim1 = 0; dim1 < expectedShape[0]; ++dim1) {
    for (dim_t dim2 = 0; dim2 < expectedShape[1]; ++dim2) {
      for (dim_t dim3 = 0; dim3 < expectedShape[2]; ++dim3) {
        for (dim_t dim4 = 0; dim4 < expectedShape[3]; ++dim4) {
          // As defined in the .pbtxt file
          EXPECT_LE(0.0f, result.at({dim1, dim2, dim3, dim4}));
          EXPECT_GT(10.0f, result.at({dim1, dim2, dim3, dim4}));
        }
      }
    }
  }
}

// Here we use input's shape for the result
TEST_F(Caffe2ImporterTest, uniformFillUseInputShape) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename(
      GLOW_DATA_PATH
      "tests/models/caffe2Models/uniform_fill_use_input_shape.pbtxt");
  std::string NetWeightFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/empty_init_net.pbtxt");

  PlaceholderBindings bindings;

  Placeholder *output;
  const std::vector<dim_t> inputShape{4, 5, 6, 7};
  Tensor inputs_0(ElemKind::Int64ITy, inputShape);
  inputs_0.getHandle<int64_t>().randomize(-10, 10, mod.getPRNG());
  // Destroy the loader after the graph is loaded since the following execution
  // will not depend on anything from the loader.
  {
    Caffe2ModelLoader caffe2LD(NetDescFilename, NetWeightFilename, {"inputs_0"},
                               {&inputs_0.getType()}, *F);
    output = EXIT_ON_ERR(caffe2LD.getSingleOutput());

    bindings.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(bindings, &mod, {"inputs_0"}, {&inputs_0});
  }

  // Check that the shape of the output matches what Caffe2 expects.
  EXPECT_EQ(inputShape, output->dims().vec());

  auto res = bindings.get(output);
  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  auto result = res->getHandle();
  for (dim_t dim1 = 0; dim1 < inputShape[0]; ++dim1) {
    for (dim_t dim2 = 0; dim2 < inputShape[1]; ++dim2) {
      for (dim_t dim3 = 0; dim3 < inputShape[2]; ++dim3) {
        for (dim_t dim4 = 0; dim4 < inputShape[3]; ++dim4) {
          // As defined in the .pbtxt file
          EXPECT_LE(0.0f, result.at({dim1, dim2, dim3, dim4}));
          EXPECT_GT(10.0f, result.at({dim1, dim2, dim3, dim4}));
        }
      }
    }
  }
}

// Here we use a shape that is provided as an argument
TEST_F(Caffe2ImporterTest, constantFillUseProvidedShape) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename(
      GLOW_DATA_PATH
      "tests/models/caffe2Models/constant_fill_use_provided_shape.pbtxt");
  std::string NetWeightFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/empty_init_net.pbtxt");

  PlaceholderBindings bindings;

  Placeholder *output;
  // Destroy the loader after the graph is loaded since the following execution
  // will not depend on anything from the loader.
  {
    Caffe2ModelLoader caffe2LD(NetDescFilename, NetWeightFilename, {}, {}, *F);
    output = EXIT_ON_ERR(caffe2LD.getSingleOutput());

    bindings.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(bindings, &mod, {}, {});
  }

  // Shape is defined in .pbtxt file
  const std::vector<dim_t> expectedShape{7, 11, 13, 17};
  EXPECT_EQ(expectedShape, output->dims().vec());

  auto res = bindings.get(output);
  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  auto result = res->getHandle();
  for (dim_t dim1 = 0; dim1 < expectedShape[0]; ++dim1) {
    for (dim_t dim2 = 0; dim2 < expectedShape[1]; ++dim2) {
      for (dim_t dim3 = 0; dim3 < expectedShape[2]; ++dim3) {
        for (dim_t dim4 = 0; dim4 < expectedShape[3]; ++dim4) {
          // As defined in the .pbtxt file
          EXPECT_FLOAT_EQ(3.0f, result.at({dim1, dim2, dim3, dim4}));
        }
      }
    }
  }
}

// Here we use expect input to be 1D tensor and act like shape
TEST_F(Caffe2ImporterTest, constantFillInputAsShape) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename(
      GLOW_DATA_PATH
      "tests/models/caffe2Models/constant_fill_input_as_shape.pbtxt");
  std::string NetWeightFilename(
      GLOW_DATA_PATH
      "tests/models/caffe2Models/constant_fill_input_as_shape_init.pbtxt");

  PlaceholderBindings bindings;

  Placeholder *output;
  // Destroy the loader after the graph is loaded since the following execution
  // will not depend on anything from the loader.
  {
    Caffe2ModelLoader caffe2LD(NetDescFilename, NetWeightFilename, {}, {}, *F);
    output = EXIT_ON_ERR(caffe2LD.getSingleOutput());

    bindings.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(bindings, &mod, {}, {});
  }

  // Shape is defined in .pbtxt file
  const std::vector<dim_t> expectedShape{17, 23, 29, 31};
  EXPECT_EQ(expectedShape, output->dims().vec());

  auto res = bindings.get(output);
  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  auto result = res->getHandle();
  for (dim_t dim1 = 0; dim1 < expectedShape[0]; ++dim1) {
    for (dim_t dim2 = 0; dim2 < expectedShape[1]; ++dim2) {
      for (dim_t dim3 = 0; dim3 < expectedShape[2]; ++dim3) {
        for (dim_t dim4 = 0; dim4 < expectedShape[3]; ++dim4) {
          // As defined in the .pbtxt file
          EXPECT_FLOAT_EQ(3.0f, result.at({dim1, dim2, dim3, dim4}));
        }
      }
    }
  }
}

// Here we use input's shape for the result
TEST_F(Caffe2ImporterTest, constantFillUseInputShape) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetDescFilename(
      GLOW_DATA_PATH
      "tests/models/caffe2Models/constant_fill_use_input_shape.pbtxt");
  std::string NetWeightFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/empty_init_net.pbtxt");

  PlaceholderBindings bindings;

  Placeholder *output;
  const std::vector<dim_t> inputShape{7, 11, 13, 17};
  Tensor inputs_0(ElemKind::FloatTy, inputShape);
  inputs_0.getHandle().randomize(-3.0, 3.0, mod.getPRNG());
  // Destroy the loader after the graph is loaded since the following execution
  // will not depend on anything from the loader.
  {
    Caffe2ModelLoader caffe2LD(NetDescFilename, NetWeightFilename, {"inputs_0"},
                               {&inputs_0.getType()}, *F);
    output = EXIT_ON_ERR(caffe2LD.getSingleOutput());

    bindings.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(bindings, &mod, {"inputs_0"}, {&inputs_0});
  }

  // Check that the shape of the output matches what Caffe2 expects.
  EXPECT_EQ(inputShape, output->dims().vec());

  auto res = bindings.get(output);
  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  auto result = res->getHandle();
  for (dim_t dim1 = 0; dim1 < inputShape[0]; ++dim1) {
    for (dim_t dim2 = 0; dim2 < inputShape[1]; ++dim2) {
      for (dim_t dim3 = 0; dim3 < inputShape[2]; ++dim3) {
        for (dim_t dim4 = 0; dim4 < inputShape[3]; ++dim4) {
          // As defined in the .pbtxt file
          EXPECT_FLOAT_EQ(3.0f, result.at({dim1, dim2, dim3, dim4}));
        }
      }
    }
  }
}

TEST_F(Caffe2ImporterTest, reduceBackSum) {
  const std::string NetDescFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/reduce_back_sum.pbtxt");
  const std::string NetWeightFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/empty_init_net.pbtxt");

  // Shape validations
  {
    const std::vector<std::vector<dim_t>> inputShapes{
        {3, 2},
        {64, 64, 11},
        {2, 3, 4, 5},
    };
    for (const auto &inputShape : inputShapes) {
      ExecutionEngine EE{};
      auto &mod = EE.getModule();
      Function *F = mod.createFunction("main");

      PlaceholderBindings bindings;
      Placeholder *output;

      Tensor inputs_0{ElemKind::FloatTy, inputShape};
      inputs_0.getHandle().randomize(-3.0, 3.0, mod.getPRNG());
      // Destroy the loader after the graph is loaded since the following
      // execution will not depend on anything from the loader.
      {
        Caffe2ModelLoader caffe2LD(NetDescFilename, NetWeightFilename,
                                   {"inputs_0"}, {&inputs_0.getType()}, *F);
        output = EXIT_ON_ERR(caffe2LD.getSingleOutput());

        bindings.allocate(mod.getPlaceholders());
        updateInputPlaceholdersByName(bindings, &mod, {"inputs_0"},
                                      {&inputs_0});
      }

      // Check that the shape of the output matches what Caffe2 expects.
      const std::vector<dim_t> expectedOutputShape{inputShape.begin(),
                                                   inputShape.end() - 1};
      EXPECT_EQ(expectedOutputShape, output->dims().vec());
    }
  }

  // Numeric validations
  {
    ExecutionEngine EE{};
    auto &mod = EE.getModule();
    Function *F = mod.createFunction("main");

    PlaceholderBindings bindings;
    Placeholder *output;

    std::vector<dim_t> inputShape{11, 13, 17};

    Tensor inputs_0{ElemKind::FloatTy, inputShape};
    inputs_0.getHandle().randomize(-3.0, 3.0, mod.getPRNG());
    // Destroy the loader after the graph is loaded since the following
    // execution will not depend on anything from the loader.
    {
      Caffe2ModelLoader caffe2LD(NetDescFilename, NetWeightFilename,
                                 {"inputs_0"}, {&inputs_0.getType()}, *F);
      output = EXIT_ON_ERR(caffe2LD.getSingleOutput());

      bindings.allocate(mod.getPlaceholders());
      updateInputPlaceholdersByName(bindings, &mod, {"inputs_0"}, {&inputs_0});
    }

    auto res = bindings.get(output);
    EE.compile(CompilationMode::Infer);
    EE.run(bindings);

    auto result = res->getHandle();
    for (dim_t dim1 = 0; dim1 < inputShape[0]; ++dim1) {
      for (dim_t dim2 = 0; dim2 < inputShape[1]; ++dim2) {
        float sum = 0;
        for (dim_t dim3 = 0; dim3 < inputShape[2]; ++dim3) {
          sum += inputs_0.getHandle().at({dim1, dim2, dim3});
        }
        EXPECT_FLOAT_EQ(sum, result.at({dim1, dim2}));
      }
    }
  }
}

TEST_F(Caffe2ImporterTest, rmsNorm) {
  const std::string NetDescFilename(GLOW_DATA_PATH
                                    "tests/models/caffe2Models/rmsnorm.pbtxt");
  const std::string NetWeightFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/empty_init_net.pbtxt");

  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  PlaceholderBindings bindings;
  Placeholder *y;
  Placeholder *rrms;

  const std::vector<dim_t> xShape{3, 4};
  const std::vector<dim_t> gammaShape{4};
  const std::vector<dim_t> betaShape{4};

  Tensor x{ElemKind::FloatTy, xShape};
  Tensor gamma{ElemKind::FloatTy, gammaShape};
  Tensor beta{ElemKind::FloatTy, betaShape};

  x.getHandle() = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  gamma.getHandle() = {1, 2, 3, 4};
  beta.getHandle() = {1, 2, 3, 4};

  // Destroy the loader after the graph is loaded since the following
  // execution will not depend on anything from the loader.
  {
    Caffe2ModelLoader caffe2LD(
        NetDescFilename, NetWeightFilename, {"x", "gamma", "beta"},
        {&x.getType(), &gamma.getType(), &beta.getType()}, *F);
    y = EXIT_ON_ERR(caffe2LD.getOutputByName("y"));
    rrms = EXIT_ON_ERR(caffe2LD.getOutputByName("rrms"));

    bindings.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(bindings, &mod, {"x", "gamma", "beta"},
                                  {&x, &gamma, &beta});
  }

  EXPECT_NE(y, nullptr);
  EXPECT_NE(rrms, nullptr);

  auto yRes = bindings.get(y);
  auto rrmsRes = bindings.get(rrms);

  const std::vector<dim_t> yShapeExpected{xShape};
  const std::vector<dim_t> rrmsShapeExpected{xShape[0]};

  EXPECT_EQ(yShapeExpected, yRes->dims().vec());
  EXPECT_EQ(rrmsShapeExpected, rrmsRes->dims().vec());

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  auto yHandle = yRes->getHandle();

  // Results are based on Caffe2 implementation
  const std::vector<std::vector<float>> yExpected{
      {1.3429972, 3.3719888, 6.0869746, 9.487955},
      {1.7495317, 3.798876, 6.148033, 8.797003},
      {1.8485281, 3.8856182, 6.11127, 8.525484},
  };

  for (dim_t d1 = 0; d1 < xShape[0]; ++d1) {
    for (dim_t d2 = 0; d2 < xShape[1]; ++d2) {
      EXPECT_NEAR(yExpected[d1][d2], yHandle.at({d1, d2}), 1e-5);
    }
  }
}

TEST_F(Caffe2ImporterTest, mean) {
  const std::string NetDescFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/mean_3inputs.pbtxt");
  const std::string NetWeightFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/empty_init_net.pbtxt");

  const std::vector<dim_t> inputShape{5, 6};

  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  PlaceholderBindings bindings;
  Placeholder *outputPH;

  Tensor input1{ElemKind::FloatTy, inputShape};
  Tensor input2{ElemKind::FloatTy, inputShape};
  Tensor input3{ElemKind::FloatTy, inputShape};
  input1.getHandle().randomize(-3.0, 3.0, mod.getPRNG());
  input2.getHandle().randomize(-3.0, 3.0, mod.getPRNG());
  input3.getHandle().randomize(-3.0, 3.0, mod.getPRNG());

  // Destroy the loader after the graph is loaded since the following
  // execution will not depend on anything from the loader.
  {
    Caffe2ModelLoader caffe2LD(
        NetDescFilename, NetWeightFilename, {"input1", "input2", "input3"},
        {&input1.getType(), &input2.getType(), &input3.getType()}, *F);
    outputPH = EXIT_ON_ERR(caffe2LD.getOutputByName("output"));

    bindings.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(bindings, &mod,
                                  {"input1", "input2", "input3"},
                                  {&input1, &input2, &input3});
  }

  auto output = bindings.get(outputPH);

  EXPECT_EQ(inputShape, output->dims().vec());

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  auto outputH = output->getHandle();

  for (dim_t d1 = 0; d1 < inputShape[0]; ++d1) {
    for (dim_t d2 = 0; d2 < inputShape[1]; ++d2) {
      auto val =
          (input1.getHandle().at({d1, d2}) + input2.getHandle().at({d1, d2}) +
           input3.getHandle().at({d1, d2})) /
          3;
      EXPECT_NEAR(val, outputH.at({d1, d2}), 1e-5);
    }
  }
}

TEST_F(Caffe2ImporterTest, negative) {
  const std::string NetDescFilename(GLOW_DATA_PATH
                                    "tests/models/caffe2Models/negative.pbtxt");
  const std::string NetWeightFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/empty_init_net.pbtxt");

  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  PlaceholderBindings bindings;
  Placeholder *outputPH;

  const std::vector<dim_t> shape{5, 6};
  Tensor input{ElemKind::FloatTy, shape};
  input.getHandle().randomize(-3.0, 3.0, mod.getPRNG());

  // Destroy the loader after the graph is loaded since the following
  // execution will not depend on anything from the loader.
  {
    Caffe2ModelLoader caffe2LD(NetDescFilename, NetWeightFilename, {"input"},
                               {&input.getType()}, *F);
    outputPH = EXIT_ON_ERR(caffe2LD.getOutputByName("output"));

    bindings.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(bindings, &mod, {"input"}, {&input});
  }

  auto output = bindings.get(outputPH);

  EXPECT_EQ(shape, output->dims().vec());

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  auto outputH = output->getHandle();

  for (dim_t d1 = 0; d1 < shape[0]; ++d1) {
    for (dim_t d2 = 0; d2 < shape[1]; ++d2) {
      EXPECT_NEAR(-1 * input.getHandle().at({d1, d2}), outputH.at({d1, d2}),
                  1e-5);
    }
  }
}

TEST_F(Caffe2ImporterTest, lpNorm) {
  const std::vector<dim_t> inputShape{5, 6};

  const auto runTest =
      [&inputShape](const std::string &NetDescFilename,
                    const std::string &NetWeightFilename,
                    std::function<float(const Tensor &input)> refImpl) {
        ExecutionEngine EE{};
        auto &mod = EE.getModule();
        Function *F = mod.createFunction("main");

        PlaceholderBindings bindings;
        Placeholder *outputPH;

        Tensor input{ElemKind::FloatTy, inputShape};
        input.getHandle().randomize(-3.0, 3.0, mod.getPRNG());

        // Destroy the loader after the graph is loaded since the following
        // execution will not depend on anything from the loader.
        {
          Caffe2ModelLoader caffe2LD(NetDescFilename, NetWeightFilename,
                                     {"input"}, {&input.getType()}, *F);
          outputPH = EXIT_ON_ERR(caffe2LD.getOutputByName("output"));

          bindings.allocate(mod.getPlaceholders());
          updateInputPlaceholdersByName(bindings, &mod, {"input"}, {&input});
        }

        auto output = bindings.get(outputPH);

        const std::vector<dim_t> expectedOutputShape{1};
        EXPECT_EQ(expectedOutputShape, output->dims().vec());

        EE.compile(CompilationMode::Infer);
        EE.run(bindings);

        auto outputH = output->getHandle();

        EXPECT_NEAR(refImpl(input), outputH.at({0}), 1e-5);
      };

  const auto refImplP1 = [&inputShape](const Tensor &input) {
    float sum = 0;
    for (dim_t d1 = 0; d1 < inputShape[0]; ++d1) {
      for (dim_t d2 = 0; d2 < inputShape[1]; ++d2) {
        auto val = input.getHandle().at({d1, d2});
        sum += ((val >= 0) ? val : -val);
      }
    }
    return sum;
  };

  const auto refImplP2 = [&inputShape](const Tensor &input) {
    float sum = 0;
    for (dim_t d1 = 0; d1 < inputShape[0]; ++d1) {
      for (dim_t d2 = 0; d2 < inputShape[1]; ++d2) {
        auto val = input.getHandle().at({d1, d2});
        sum += val * val;
      }
    }
    return sum;
  };

  runTest(GLOW_DATA_PATH "tests/models/caffe2Models/lpnorm_p1.pbtxt",
          GLOW_DATA_PATH "tests/models/caffe2Models/empty_init_net.pbtxt",
          refImplP1);

  runTest(GLOW_DATA_PATH "tests/models/caffe2Models/lpnorm_p2.pbtxt",
          GLOW_DATA_PATH "tests/models/caffe2Models/empty_init_net.pbtxt",
          refImplP2);
}

TEST_F(Caffe2ImporterTest, argMin) {
  const std::string NetDescFilename(GLOW_DATA_PATH
                                    "tests/models/caffe2Models/argmin.pbtxt");
  const std::string NetWeightFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/empty_init_net.pbtxt");

  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  PlaceholderBindings bindings;
  Placeholder *output;

  std::vector<dim_t> inputShape{1, 50};

  Tensor input{ElemKind::FloatTy, {inputShape}};
  input.getHandle().randomize(-3.0, 3.0, mod.getPRNG());
  // Destroy the loader after the graph is loaded since the following
  // execution will not depend on anything from the loader.
  {
    Caffe2ModelLoader caffe2LD(NetDescFilename, NetWeightFilename, {"input"},
                               {&input.getType()}, *F);
    output = EXIT_ON_ERR(caffe2LD.getSingleOutput());

    bindings.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(bindings, &mod, {"input"}, {&input});
  }

  auto res = bindings.get(output);
  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  auto result = res->getHandle<int64_t>();
  std::vector<dim_t> expectedShape{1};
  EXPECT_EQ(expectedShape, result.dims().vec());

  dim_t minIndex = 0;
  for (dim_t d = 1; d < inputShape[1]; ++d) {
    if (input.getHandle().at({0, d}) < input.getHandle().at({0, minIndex})) {
      minIndex = d;
    }
  }
  EXPECT_EQ(minIndex, result.at({0}));
}

TEST_F(Caffe2ImporterTest, scale) {
  const std::string NetDescFilename(GLOW_DATA_PATH
                                    "tests/models/caffe2Models/scale.pbtxt");
  const std::string NetWeightFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/empty_init_net.pbtxt");

  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  PlaceholderBindings bindings;
  Placeholder *outputPH;

  std::vector<dim_t> inputShape{10, 30};

  Tensor input{ElemKind::FloatTy, {inputShape}};
  input.getHandle().randomize(-10.0, 10.0, mod.getPRNG());
  // Destroy the loader after the graph is loaded since the following
  // execution will not depend on anything from the loader.
  {
    Caffe2ModelLoader caffe2LD(NetDescFilename, NetWeightFilename, {"input"},
                               {&input.getType()}, *F);
    outputPH = EXIT_ON_ERR(caffe2LD.getSingleOutput());

    bindings.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(bindings, &mod, {"input"}, {&input});
  }

  auto output = bindings.get(outputPH);
  EXPECT_EQ(inputShape, output->dims().vec());

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  auto outputH = output->getHandle();

  for (dim_t d1 = 1; d1 < inputShape[0]; ++d1) {
    for (dim_t d2 = 1; d2 < inputShape[1]; ++d2) {
      auto val = input.getHandle().at({d1, d2});
      auto exp = 2 * val;
      EXPECT_NEAR(exp, outputH.at({d1, d2}), 1e-3);
    }
  }
}

TEST_F(Caffe2ImporterTest, sign) {
  const std::string NetDescFilename(GLOW_DATA_PATH
                                    "tests/models/caffe2Models/sign.pbtxt");
  const std::string NetWeightFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/empty_init_net.pbtxt");

  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  PlaceholderBindings bindings;
  Placeholder *outputPH;

  std::vector<dim_t> inputShape{20, 50};

  Tensor input{ElemKind::FloatTy, {inputShape}};
  input.getHandle().randomize(-3.0, 3.0, mod.getPRNG());
  // Destroy the loader after the graph is loaded since the following
  // execution will not depend on anything from the loader.
  {
    Caffe2ModelLoader caffe2LD(NetDescFilename, NetWeightFilename, {"input"},
                               {&input.getType()}, *F);
    outputPH = EXIT_ON_ERR(caffe2LD.getSingleOutput());

    bindings.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(bindings, &mod, {"input"}, {&input});
  }

  auto output = bindings.get(outputPH);
  EXPECT_EQ(inputShape, output->dims().vec());

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  auto outputH = output->getHandle();

  for (dim_t d1 = 1; d1 < inputShape[0]; ++d1) {
    for (dim_t d2 = 1; d2 < inputShape[1]; ++d2) {
      auto val = input.getHandle().at({d1, d2});
      auto exp = (val > 0) ? 1 : (val == 0 ? 0 : -1);
      EXPECT_EQ(exp, outputH.at({d1, d2}));
    }
  }
}

TEST_F(Caffe2ImporterTest, softplus) {
  const std::string NetDescFilename(GLOW_DATA_PATH
                                    "tests/models/caffe2Models/softplus.pbtxt");
  const std::string NetWeightFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/empty_init_net.pbtxt");

  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  PlaceholderBindings bindings;
  Placeholder *outputPH;

  std::vector<dim_t> inputShape{20, 50};

  Tensor input{ElemKind::FloatTy, {inputShape}};
  input.getHandle().randomize(-3.0, 3.0, mod.getPRNG());
  // Destroy the loader after the graph is loaded since the following
  // execution will not depend on anything from the loader.
  {
    Caffe2ModelLoader caffe2LD(NetDescFilename, NetWeightFilename, {"input"},
                               {&input.getType()}, *F);
    outputPH = EXIT_ON_ERR(caffe2LD.getSingleOutput());

    bindings.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(bindings, &mod, {"input"}, {&input});
  }

  auto output = bindings.get(outputPH);
  EXPECT_EQ(inputShape, output->dims().vec());

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  auto outputH = output->getHandle();

  for (dim_t d1 = 1; d1 < inputShape[0]; ++d1) {
    for (dim_t d2 = 1; d2 < inputShape[1]; ++d2) {
      auto val = input.getHandle().at({d1, d2});
      auto exp = std::log(std::exp(val) + 1);
      EXPECT_NEAR(exp, outputH.at({d1, d2}), 1e-3);
    }
  }
}

TEST_F(Caffe2ImporterTest, topk) {
  const std::string NetDescFilename(GLOW_DATA_PATH
                                    "tests/models/caffe2Models/topk_k2.pbtxt");
  const std::string NetWeightFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/empty_init_net.pbtxt");

  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  PlaceholderBindings bindings;
  Placeholder *valuesPH;
  Placeholder *indicesPH;

  std::vector<dim_t> inputShape{20, 50};

  Tensor input{ElemKind::FloatTy, {inputShape}};
  input.getHandle().randomize(-3.0, 3.0, mod.getPRNG());
  // Destroy the loader after the graph is loaded since the following
  // execution will not depend on anything from the loader.
  {
    Caffe2ModelLoader caffe2LD(NetDescFilename, NetWeightFilename, {"input"},
                               {&input.getType()}, *F);
    valuesPH = EXIT_ON_ERR(caffe2LD.getOutputByName("values"));
    indicesPH = EXIT_ON_ERR(caffe2LD.getOutputByName("indices"));

    bindings.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(bindings, &mod, {"input"}, {&input});
  }

  auto values = bindings.get(valuesPH);
  auto indices = bindings.get(indicesPH);
  std::vector<dim_t> expectedShape{20, 2};
  EXPECT_EQ(expectedShape, values->dims().vec());
  EXPECT_EQ(expectedShape, indices->dims().vec());

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  auto valuesH = values->getHandle();
  auto indicesH = indices->getHandle<int32_t>();

  for (dim_t d1 = 0; d1 < inputShape[0]; ++d1) {
    dim_t firstLargestIndex = 0;
    dim_t secondLargestIndex = 1;
    if (input.getHandle().at({d1, 1}) > input.getHandle().at({d1, 0})) {
      std::swap(firstLargestIndex, secondLargestIndex);
    }
    float firstLargestValue = input.getHandle().at({d1, firstLargestIndex});
    float secondLargestValue = input.getHandle().at({d1, secondLargestIndex});

    for (dim_t d2 = 2; d2 < inputShape[1]; ++d2) {
      auto val = input.getHandle().at({d1, d2});
      if (val > firstLargestValue) {
        secondLargestIndex = firstLargestIndex;
        secondLargestValue = firstLargestValue;
        firstLargestIndex = d2;
        firstLargestValue = val;
      } else if (val > secondLargestValue) {
        secondLargestIndex = d2;
        secondLargestValue = val;
      }
    }

    EXPECT_EQ(indicesH.at({d1, 0}), firstLargestIndex);
    EXPECT_EQ(indicesH.at({d1, 1}), secondLargestIndex);
    EXPECT_EQ(valuesH.at({d1, 0}), firstLargestValue);
    EXPECT_EQ(valuesH.at({d1, 1}), secondLargestValue);
  }
}

TEST_F(Caffe2ImporterTest, SparseLabelSplit) {
  const std::string NetDescFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/sparselabelsplit.pbtxt");
  const std::string NetWeightFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/empty_init_net.pbtxt");

  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  PlaceholderBindings bindings;
  Placeholder *labelValues0PH;
  Placeholder *labelValues1PH;
  Placeholder *labelValues2PH;
  Placeholder *labelValues3PH;
  Placeholder *exampleIds0PH;
  Placeholder *exampleIds1PH;
  Placeholder *exampleIds2PH;
  Placeholder *exampleIds3PH;
  Placeholder *gradientOffsetMapPH;

  Tensor lengths{ElemKind::Int32ITy, {4}};
  Tensor indices{ElemKind::Int64ITy, {8}};
  Tensor values{ElemKind::FloatTy, {8}};

  lengths.getHandle<int32_t>() = {1, 3, 2, 2};
  indices.getHandle<int64_t>() = {3, 1, 2, 0, 0, 2, 1, 3};
  values.getHandle() = {1.2, 2.3, 3.1, 6.7, 8.3, 9.0, 3.7, 8.8};

  // Destroy the loader after the graph is loaded since the following
  // execution will not depend on anything from the loader.
  {
    Caffe2ModelLoader caffe2LD(
        NetDescFilename, NetWeightFilename, {"lengths", "indices", "values"},
        {&lengths.getType(), &indices.getType(), &values.getType()}, *F);
    labelValues0PH = EXIT_ON_ERR(caffe2LD.getOutputByName("label_values_0"));
    labelValues1PH = EXIT_ON_ERR(caffe2LD.getOutputByName("label_values_1"));
    labelValues2PH = EXIT_ON_ERR(caffe2LD.getOutputByName("label_values_2"));
    labelValues3PH = EXIT_ON_ERR(caffe2LD.getOutputByName("label_values_3"));
    exampleIds0PH = EXIT_ON_ERR(caffe2LD.getOutputByName("example_ids_0"));
    exampleIds1PH = EXIT_ON_ERR(caffe2LD.getOutputByName("example_ids_1"));
    exampleIds2PH = EXIT_ON_ERR(caffe2LD.getOutputByName("example_ids_2"));
    exampleIds3PH = EXIT_ON_ERR(caffe2LD.getOutputByName("example_ids_3"));
    gradientOffsetMapPH =
        EXIT_ON_ERR(caffe2LD.getOutputByName("gradient_offset_map"));

    bindings.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(bindings, &mod,
                                  {"lengths", "indices", "values"},
                                  {&lengths, &indices, &values});
  }

  EXPECT_NE(labelValues0PH, nullptr);
  EXPECT_NE(labelValues1PH, nullptr);
  EXPECT_NE(labelValues2PH, nullptr);
  EXPECT_NE(labelValues3PH, nullptr);
  EXPECT_NE(exampleIds0PH, nullptr);
  EXPECT_NE(exampleIds1PH, nullptr);
  EXPECT_NE(exampleIds2PH, nullptr);
  EXPECT_NE(exampleIds3PH, nullptr);
  EXPECT_NE(gradientOffsetMapPH, nullptr);

  auto labelValues0 = bindings.get(labelValues0PH);
  auto labelValues1 = bindings.get(labelValues1PH);
  auto labelValues2 = bindings.get(labelValues2PH);
  auto labelValues3 = bindings.get(labelValues3PH);
  auto exampleIds0 = bindings.get(exampleIds0PH);
  auto exampleIds1 = bindings.get(exampleIds1PH);
  auto exampleIds2 = bindings.get(exampleIds2PH);
  auto exampleIds3 = bindings.get(exampleIds3PH);
  auto gradientOffsetMap = bindings.get(gradientOffsetMapPH);

  std::vector<dim_t> expectedOutputShape{2};
  EXPECT_EQ(expectedOutputShape, labelValues0->dims().vec());
  EXPECT_EQ(expectedOutputShape, labelValues1->dims().vec());
  EXPECT_EQ(expectedOutputShape, labelValues2->dims().vec());
  EXPECT_EQ(expectedOutputShape, labelValues3->dims().vec());
  EXPECT_EQ(expectedOutputShape, exampleIds0->dims().vec());
  EXPECT_EQ(expectedOutputShape, exampleIds1->dims().vec());
  EXPECT_EQ(expectedOutputShape, exampleIds2->dims().vec());
  EXPECT_EQ(expectedOutputShape, exampleIds3->dims().vec());

  std::vector<dim_t> expectedGradientOffsetMapShape{8};
  EXPECT_EQ(expectedGradientOffsetMapShape, gradientOffsetMap->dims().vec());

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  auto labelValues0H = labelValues0->getHandle();
  auto labelValues1H = labelValues1->getHandle();
  auto labelValues2H = labelValues2->getHandle();
  auto labelValues3H = labelValues3->getHandle();
  auto exampleIds0H = exampleIds0->getHandle<int32_t>();
  auto exampleIds1H = exampleIds1->getHandle<int32_t>();
  auto exampleIds2H = exampleIds2->getHandle<int32_t>();
  auto exampleIds3H = exampleIds3->getHandle<int32_t>();
  auto gradientOffsetMapH = gradientOffsetMap->getHandle<int32_t>();

  EXPECT_NEAR(6.7, labelValues0H.at(0), 1e-3);
  EXPECT_NEAR(8.3, labelValues0H.at(1), 1e-3);
  EXPECT_NEAR(2.3, labelValues1H.at(0), 1e-3);
  EXPECT_NEAR(3.7, labelValues1H.at(1), 1e-3);
  EXPECT_NEAR(3.1, labelValues2H.at(0), 1e-3);
  EXPECT_NEAR(9.0, labelValues2H.at(1), 1e-3);
  EXPECT_NEAR(1.2, labelValues3H.at(0), 1e-3);
  EXPECT_NEAR(8.8, labelValues3H.at(1), 1e-3);

  EXPECT_EQ(1, exampleIds0H.at(0));
  EXPECT_EQ(2, exampleIds0H.at(1));
  EXPECT_EQ(1, exampleIds1H.at(0));
  EXPECT_EQ(3, exampleIds1H.at(1));
  EXPECT_EQ(1, exampleIds2H.at(0));
  EXPECT_EQ(2, exampleIds2H.at(1));
  EXPECT_EQ(0, exampleIds3H.at(0));
  EXPECT_EQ(3, exampleIds3H.at(1));

  const std::vector<int32_t> expectedGradientOffsetMap{0, 0, 0, 0, 1, 1, 1, 1};
  for (dim_t d = 0; d < 8; ++d) {
    EXPECT_EQ(expectedGradientOffsetMap[d], gradientOffsetMapH.at(d));
  }
}

TEST_F(Caffe2ImporterTest, Log1p) {
  const std::string NetDescFilename(GLOW_DATA_PATH
                                    "tests/models/caffe2Models/log1p.pbtxt");
  const std::string NetWeightFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/empty_init_net.pbtxt");

  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  PlaceholderBindings bindings;
  Placeholder *outputPH;

  std::vector<dim_t> inputShape{3, 4};

  Tensor input{ElemKind::FloatTy, {inputShape}};
  input.getHandle() = {0.5, 0.1, 0.01, 1, 2, 3, 10, 20, 30, 1.1, 1.01, 1.001};
  // Destroy the loader after the graph is loaded since the following
  // execution will not depend on anything from the loader.
  {
    Caffe2ModelLoader caffe2LD(NetDescFilename, NetWeightFilename, {"input"},
                               {&input.getType()}, *F);
    outputPH = EXIT_ON_ERR(caffe2LD.getSingleOutput());

    bindings.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(bindings, &mod, {"input"}, {&input});
  }

  auto output = bindings.get(outputPH);
  EXPECT_EQ(inputShape, output->dims().vec());

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  auto outputH = output->getHandle();

  for (dim_t d1 = 1; d1 < inputShape[0]; ++d1) {
    for (dim_t d2 = 1; d2 < inputShape[1]; ++d2) {
      auto val = input.getHandle().at({d1, d2});
      auto exp = std::log(val + 1);
      EXPECT_NEAR(exp, outputH.at({d1, d2}), 1e-3) << "Bad value " << val;
    }
  }
}

TEST_F(Caffe2ImporterTest, ReduceBackMean) {
  const std::string NetDescFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/reducebackmean.pbtxt");
  const std::string NetWeightFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/empty_init_net.pbtxt");

  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  PlaceholderBindings bindings;
  Placeholder *outputPH;

  std::vector<dim_t> inputShape{3, 4};

  Tensor input{ElemKind::FloatTy, {inputShape}};
  input.getHandle() = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  // Destroy the loader after the graph is loaded since the following
  // execution will not depend on anything from the loader.
  {
    Caffe2ModelLoader caffe2LD(NetDescFilename, NetWeightFilename, {"input"},
                               {&input.getType()}, *F);
    outputPH = EXIT_ON_ERR(caffe2LD.getSingleOutput());

    bindings.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(bindings, &mod, {"input"}, {&input});
  }

  const std::vector<dim_t> expectedShape{3};
  auto output = bindings.get(outputPH);
  EXPECT_EQ(expectedShape, output->dims().vec());

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  auto outputH = output->getHandle();

  for (dim_t d1 = 0; d1 < inputShape[0]; ++d1) {
    float expected = 0;
    for (dim_t d2 = 0; d2 < inputShape[1]; ++d2) {
      expected += input.getHandle().at({d1, d2});
    }
    expected /= inputShape[1];
    EXPECT_NEAR(expected, outputH.at(d1), 1e-3);
  }
}

// Test that dropout is a no-op
TEST_F(Caffe2ImporterTest, Dropout) {
  const std::string NetDescFilename(GLOW_DATA_PATH
                                    "tests/models/caffe2Models/dropout.pbtxt");
  const std::string NetWeightFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/empty_init_net.pbtxt");

  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  PlaceholderBindings bindings;
  Placeholder *outputPH;

  std::vector<dim_t> inputShape{20, 50};
  Tensor input{ElemKind::FloatTy, {inputShape}};
  input.getHandle().randomize(-3.0, 3.0, mod.getPRNG());
  // Destroy the loader after the graph is loaded since the following
  // execution will not depend on anything from the loader.
  {
    Caffe2ModelLoader caffe2LD(NetDescFilename, NetWeightFilename, {"input"},
                               {&input.getType()}, *F);
    outputPH = EXIT_ON_ERR(caffe2LD.getSingleOutput());

    bindings.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(bindings, &mod, {"input"}, {&input});
  }

  // Expect graph to have one node (1 save node)
  EXPECT_EQ(F->getNodes().size(), 1);
  auto *save = getSaveNodeFromDest(outputPH);
  ASSERT_TRUE(save);

  auto output = bindings.get(outputPH);
  EXPECT_EQ(inputShape, output->dims().vec());

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  auto inputH = input.getHandle();
  auto outputH = output->getHandle();
  for (dim_t d1 = 0; d1 < inputShape[0]; ++d1) {
    for (dim_t d2 = 0; d2 < inputShape[1]; ++d2) {
      EXPECT_EQ(inputH.at(d1), outputH.at(d1));
    }
  }
}

TEST_F(Caffe2ImporterTest, CastInt64ToInt64) {
  const std::string NetDescFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/cast_int64_to_int64.pbtxt");
  const std::string NetWeightFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/empty_init_net.pbtxt");

  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  PlaceholderBindings bindings;
  Placeholder *outputPH;

  std::vector<dim_t> inputShape{3, 4};

  Tensor input{ElemKind::Int64ITy, {inputShape}};
  input.getHandle<int64_t>() = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  // Destroy the loader after the graph is loaded since the following
  // execution will not depend on anything from the loader.
  {
    Caffe2ModelLoader caffe2LD(NetDescFilename, NetWeightFilename, {"input"},
                               {&input.getType()}, *F);
    outputPH = EXIT_ON_ERR(caffe2LD.getSingleOutput());

    bindings.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(bindings, &mod, {"input"}, {&input});
  }

  auto output = bindings.get(outputPH);
  EXPECT_EQ(inputShape, output->dims().vec());

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  auto outputH = output->getHandle<int64_t>();

  for (dim_t d1 = 0; d1 < inputShape[0]; ++d1) {
    for (dim_t d2 = 0; d2 < inputShape[1]; ++d2) {
      auto val = input.getHandle<int64_t>().at({d1, d2});
      auto exp = static_cast<int64_t>(val);
      EXPECT_EQ(exp, outputH.at({d1, d2}));
    }
  }
}

// Gelu test
TEST_F(Caffe2ImporterTest, gelu) {
  const std::string NetDescFilename(GLOW_DATA_PATH
                                    "tests/models/caffe2Models/gelu.pbtxt");
  const std::string NetWeightFilename(
      GLOW_DATA_PATH "tests/models/caffe2Models/empty_init_net.pbtxt");

  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  PlaceholderBindings bindings;
  Placeholder *outputPH;

  std::vector<dim_t> inputShape{10, 30};

  Tensor input{ElemKind::FloatTy, {inputShape}};
  input.getHandle().randomize(-10.0, 10.0, mod.getPRNG());
  // Destroy the loader after the graph is loaded since the following
  // execution will not depend on anything from the loader.
  {
    Caffe2ModelLoader caffe2LD(NetDescFilename, NetWeightFilename, {"input"},
                               {&input.getType()}, *F);
    outputPH = EXIT_ON_ERR(caffe2LD.getSingleOutput());

    bindings.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(bindings, &mod, {"input"}, {&input});
  }

  auto output = bindings.get(outputPH);
  EXPECT_EQ(inputShape, output->dims().vec());

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  auto outputH = output->getHandle();

  for (dim_t d1 = 1; d1 < inputShape[0]; ++d1) {
    for (dim_t d2 = 1; d2 < inputShape[1]; ++d2) {
      auto val = input.getHandle().at({d1, d2});
      // Gaussian Error Linear Unit. An activation function used in the most
      // recent Transformers – Google's BERT and OpenAI's GPT-2. This activation
      // function takes the form of this equation:
      // GELU(x) = 0.5𝑥(1+erf(𝑥/√2))
      // Knowing that erf(𝑥) is very close to tanh(𝑥)
      // GELU(x) ≃ 0.5x(1+tanh(√2/π(x+0.044715x3)))
      // = 0.5𝑥(1 + tanh(0.797885𝑥+0.035677𝑥3))
      auto exp =
          0.5 * val * (1 + tanh(0.797885 * val + 0.035677 * pow(val, 3)));
      EXPECT_NEAR(exp, outputH.at({d1, d2}), 1e-3);
    }
  }
}
