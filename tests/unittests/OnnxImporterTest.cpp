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
#include "glow/Graph/Nodes.h"
#include "glow/Graph/PlaceholderBindings.h"
#include "glow/Importer/ONNXModelLoader.h"
#include "llvm/Support/FileSystem.h"
#include "gtest/gtest.h"

#ifndef GLOW_DATA_PATH
#define GLOW_DATA_PATH
#endif

using namespace glow;

#include <fstream>
using namespace std;

class OnnxImporterTest : public ::testing::Test {
protected:
  // By default constant folding at load time is enabled in general, but we do
  // many tests here loading Constants, so keep it false during these tests by
  // default.
  void SetUp() override { glow::setConstantFoldLoaderOpsFlag(false); }
  void TearDown() override { glow::setConstantFoldLoaderOpsFlag(true); }
};

/// Loads onnxtxt model file \p filename and \returns ModelProto object.
Expected<ONNX_NAMESPACE::ModelProto> loadProto(const std::string &filename) {
  std::ifstream ff(filename, std::ios::in | std::ios::binary);
  RETURN_ERR_IF_NOT(ff,
                    strFormat("Can't find the model or network files for %s.",
                              filename.c_str()),
                    ErrorValue::ErrorCode::MODEL_LOADER_INVALID_PROTOBUF);
  if (filename.find(".onnxtxt") != std::string::npos) {
    std::string str((std::istreambuf_iterator<char>(ff)),
                    std::istreambuf_iterator<char>());
    ONNX_NAMESPACE::ModelProto MP;
    bool parseNet = google::protobuf::TextFormat::ParseFromString(str, &MP);
    RETURN_ERR_IF_NOT(parseNet, "Failed to parse ModelProto",
                      ErrorValue::ErrorCode::MODEL_LOADER_INVALID_PROTOBUF);
    return MP;
  }
  RETURN_ERR("Can't load proto file");
}

/// Saves ModelProto object \p model as onnxtxt model file \p filename
/// and \returns true if successful.
Expected<bool> saveProto(const std::string &filename,
                         ONNX_NAMESPACE::ModelProto &model) {
  std::ofstream ff(filename, std::ios::out);
  RETURN_ERR_IF_NOT(ff, "Can't write the proto file.",
                    ErrorValue::ErrorCode::RUNTIME_ERROR);
  if (filename.find(".onnxtxt") != std::string::npos) {
    std::string onnx_message = model.DebugString();
    ff << onnx_message;
    ff.close();
    return true;
  }
  ff.close();
  return false;
}

/// Replaces placeholders with names \p tensorNames in model proto object \p
/// model with initializers of same  name and values specified in input tensor
/// array \p tensors and \returns true if successful.
Expected<bool>
replacePlaceholderWithConstant(ONNX_NAMESPACE::ModelProto &model,
                               llvm::ArrayRef<const char *> tensorNames,
                               llvm::ArrayRef<Tensor *> tensors) {
  ONNX_NAMESPACE::NodeProto np;
  ONNX_NAMESPACE::GraphProto *gp = model.mutable_graph();
  RETURN_ERR_IF_NOT(gp, "Can't get mutable graph.",
                    ErrorValue::ErrorCode::RUNTIME_ERROR);
  for (size_t i = 0; i < tensorNames.size(); i++) {
    for (int j = 0; j < gp->input_size(); j++) {
      ONNX_NAMESPACE::ValueInfoProto *valueInfo = gp->mutable_input(j);
      const std::string &inputName = valueInfo->name();
      if (inputName != tensorNames[i]) {
        continue;
      }
      std::string newName = "dummy_input" + std::to_string(i);
      valueInfo->set_name(newName);
      auto RH = tensors[i]->getHandle<>();
      ONNX_NAMESPACE::TensorProto *tp = gp->add_initializer();
      tp->set_name(tensorNames[i]);
      for (size_t k = 0; k < tensors[i]->dims().size(); k++) {
        tp->add_dims(tensors[i]->dims()[k]);
      }
      switch (RH.getElementType()) {
      case ElemKind::FloatTy:
        tp->set_data_type(ONNX_NAMESPACE::TensorProto::FLOAT);
        for (size_t k = 0; k < tensors[i]->size(); k++) {
          tp->add_float_data(RH.raw(k));
        }
        break;
      case ElemKind::Int64ITy:
        tp->set_data_type(ONNX_NAMESPACE::TensorProto::INT64);
        for (size_t k = 0; k < tensors[i]->size(); k++) {
          tp->add_int64_data(RH.raw(k));
        }
        break;
      case ElemKind::Int32ITy:
        tp->set_data_type(ONNX_NAMESPACE::TensorProto::INT32);
        for (size_t k = 0; k < tensors[i]->size(); k++) {
          tp->add_int32_data(RH.raw(k));
        }
        break;
      default:
        std::cout << "Unsupported datatype";
        return false;
      }
    }
  }
  gp->clear_input();
  return true;
}

/// Performs constant folding test on the given model file \p NetFilename
/// by replacing input tensors with name \p tensorNames, and values \p tensors
/// and then checking against expected output expectedTensors. \returns true
/// if the test completes without error.
Error checkConstFoldedOutput(std::string NetFilename,
                             llvm::ArrayRef<const char *> tensorNames,
                             llvm::ArrayRef<Tensor *> tensors,
                             llvm::ArrayRef<Tensor *> expectedTensors) {
  ONNX_NAMESPACE::ModelProto modelDef;
  llvm::SmallVector<char, 64> resultPath;
  llvm::sys::fs::createTemporaryFile("dummy", "onnxtxt", resultPath);
  std::string netFilename(resultPath.begin(), resultPath.end());

  ASSIGN_VALUE_OR_RETURN_ERR(modelDef, loadProto(NetFilename));
  // Replace placeholders in the original onnx model with constants.
  RETURN_IF_ERR(replacePlaceholderWithConstant(modelDef, tensorNames, tensors)
                    .takeError());
  RETURN_IF_ERR(saveProto(netFilename, modelDef).takeError());
  setConstantFoldLoaderOpsFlag(true);

  // It is expected that loading will fold the whole graph and output
  // nodes will become constants during the loading process.
  ExecutionEngine EE;
  Module &mod = EE.getModule();
  Function *F = mod.createFunction("temp");
  ONNXModelLoader onnxLD(netFilename, {}, {}, *F);
  setConstantFoldLoaderOpsFlag(false);

  // The folded output tensors are expected to be constants and should
  // match the expectedTensors passed in.
  for (int i = 0; i < modelDef.graph().output_size(); i++) {
    NodeValue NV;
    ASSIGN_VALUE_OR_RETURN_ERR(
        NV, onnxLD.getNodeValueByName(modelDef.graph().output(i).name()));
    auto *constOut = llvm::dyn_cast<Constant>(NV.getNode());
    RETURN_ERR_IF_NOT(constOut, "Failed cast to Constant");
    EXPECT_TRUE(expectedTensors[i]->isEqual(constOut->getPayload()));
  }
  return Error::success();
}

template <class OpType>
static void
importArithMultiBroadcastTest(std::string fileName,
                              llvm::ArrayRef<dim_t> inputShape, bool multi,
                              int numLeftTile, int numRightTile,
                              const std::function<float(float, float)> &op) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetFilename =
      std::string(GLOW_DATA_PATH "tests/models/onnxModels/") + fileName;
  PlaceholderBindings bindings;
  Placeholder *graphOutputVar;
  // Destroy the loader after the graph is loaded since the following execution
  // will not depend on anyting from the loader.
  Tensor data;
  getNCHWData(&data, inputShape[0], inputShape[1], inputShape[2],
              inputShape[3]);
  {
    ONNXModelLoader onnxLD(NetFilename, {"data"}, {&data.getType()}, *F);
    graphOutputVar = EXIT_ON_ERR(onnxLD.getSingleOutput());
    bindings.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(bindings, &mod, {"data"}, {&data});
  }
  // ONNX importer loads an arithmetic node and inserts:
  // - a Reshape node for each broadcasted operand
  // - a Tile node for each boardcasted dimension
  // Check the graph structure
  auto *saveNode = getSaveNodeFromDest(graphOutputVar);
  auto *node = saveNode->getInput().getNode();
  auto *opNode = llvm::dyn_cast<OpType>(node);
  EXPECT_NE(nullptr, opNode);

  // Left operand (numLeftTile dimensions to broadcast)
  if (numLeftTile > 0) {
    TileNode *tileNode = llvm::dyn_cast<TileNode>(opNode->getLHS().getNode());
    EXPECT_NE(nullptr, tileNode);
    for (int i = 1; i < numLeftTile; i++) {
      tileNode = llvm::dyn_cast<TileNode>(tileNode->getInput().getNode());
      EXPECT_NE(nullptr, tileNode);
    }
    auto *reshapeNode =
        llvm::dyn_cast<ReshapeNode>(tileNode->getInput().getNode());
    EXPECT_NE(nullptr, reshapeNode);
  }

  // Right operand (numRightTile dimensions to broadcast)
  if (numRightTile > 0) {
    TileNode *tileNode = llvm::dyn_cast<TileNode>(opNode->getRHS().getNode());
    EXPECT_NE(nullptr, tileNode);
    for (int i = 1; i < numRightTile; i++) {
      tileNode = llvm::dyn_cast<TileNode>(tileNode->getInput().getNode());
      EXPECT_NE(nullptr, tileNode);
    }
    auto *reshapeNode =
        llvm::dyn_cast<ReshapeNode>(tileNode->getInput().getNode());
    EXPECT_NE(nullptr, reshapeNode);
  }

  // Compile&run the graph, and check the output
  EE.compile(CompilationMode::Infer);
  EE.run(bindings);
  auto result = bindings.get(graphOutputVar)->getHandle();
  std::vector<dim_t> expectedDims = {1, 3, 4, 2};
  std::vector<float> expectedValues;

  if (multi) {
    expectedValues = {op(0, 2), op(1, 2), op(0, 2), op(1, 2), op(0, 2),
                      op(1, 2), op(0, 2), op(1, 2), op(2, 2), op(3, 2),
                      op(2, 2), op(3, 2), op(2, 2), op(3, 2), op(2, 2),
                      op(3, 2), op(4, 2), op(5, 2), op(4, 2), op(5, 2),
                      op(4, 2), op(5, 2), op(4, 2), op(5, 2)};
  } else {
    expectedValues = {op(0, 2),  op(1, 2),  op(2, 2),  op(3, 2),  op(4, 2),
                      op(5, 2),  op(6, 2),  op(7, 2),  op(8, 2),  op(9, 2),
                      op(10, 2), op(11, 2), op(12, 2), op(13, 2), op(14, 2),
                      op(15, 2), op(16, 2), op(17, 2), op(18, 2), op(19, 2),
                      op(20, 2), op(21, 2), op(22, 2), op(23, 2)};
  }
  EXPECT_TRUE(result.dims().vec() == expectedDims);
  for (size_t i = 0; i < result.getType().size(); i++) {
    EXPECT_FLOAT_EQ(result.raw(i), expectedValues[i]);
  }
  // Constant Folding Test.
  FAIL_TEST_IF_ERR(checkConstFoldedOutput(NetFilename, {"data"}, {&data},
                                          {bindings.get(graphOutputVar)}));
}

/// Test loading LeakyRelu op from an ONNX model.
TEST_F(OnnxImporterTest, leakyRelu) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string netFilename(GLOW_DATA_PATH
                          "tests/models/onnxModels/leakyRelu.onnxtxt");

  PlaceholderBindings bindings;
  Placeholder *output;
  {
    Tensor x(ElemKind::FloatTy, {7});
    x.getHandle() = {0, -1, -2, -3, 4, 5, 6};

    ONNXModelLoader onnxLD(netFilename, {"x"}, {&x.getType()}, *F);
    output = EXIT_ON_ERR(onnxLD.getSingleOutput());
  }

  auto *save = getSaveNodeFromDest(output);
  PReluNode *PRL = llvm::dyn_cast<PReluNode>(save->getInput().getNode());
  ASSERT_TRUE(PRL);
  NodeValue slopeN = PRL->getSlope();
  SplatNode *splatN = llvm::dyn_cast<SplatNode>(slopeN.getNode());
  ASSERT_TRUE(splatN);
  EXPECT_FLOAT_EQ(splatN->getValue(), 0.100000001);
}

/// Test Loading LeakyRelu op from an ONNX model with default alpha.
TEST_F(OnnxImporterTest, leakyReluDefault) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string netFilename(GLOW_DATA_PATH
                          "tests/models/onnxModels/leakyReluDefault.onnxtxt");

  PlaceholderBindings bindings;
  Placeholder *output;
  {
    Tensor x(ElemKind::FloatTy, {7});
    x.getHandle() = {0, -1, -2, -3, 4, 5, 6};

    ONNXModelLoader onnxLD(netFilename, {"x"}, {&x.getType()}, *F);
    output = EXIT_ON_ERR(onnxLD.getSingleOutput());
  }

  auto *save = getSaveNodeFromDest(output);
  PReluNode *PRL = llvm::dyn_cast<PReluNode>(save->getInput().getNode());
  ASSERT_TRUE(PRL);
  NodeValue slopeN = PRL->getSlope();
  SplatNode *splatN = llvm::dyn_cast<SplatNode>(slopeN.getNode());
  ASSERT_TRUE(splatN);
  EXPECT_FLOAT_EQ(splatN->getValue(), 0.01);
}

TEST_F(OnnxImporterTest, importAddMultiBroadcastOp7) {
  importArithMultiBroadcastTest<AddNode>(
      "addMultiBroadcastOp7.onnxtxt", {1, 3, 1, 2}, true, 1, 2,
      [](float a, float b) { return a + b; });
}

TEST_F(OnnxImporterTest, importAddUniBroadcastOp6NoAxis) {
  importArithMultiBroadcastTest<AddNode>(
      "addUniBroadcastOp6NoAxis.onnxtxt", {1, 3, 4, 2}, false, 0, 2,
      [](float a, float b) { return a + b; });
}

TEST_F(OnnxImporterTest, importAddUniBroadcastOp6Axis) {
  importArithMultiBroadcastTest<AddNode>(
      "addUniBroadcastOp6Axis.onnxtxt", {1, 3, 4, 2}, false, 0, 2,
      [](float a, float b) { return a + b; });
}

TEST_F(OnnxImporterTest, importSubMultiBroadcastOp7) {
  importArithMultiBroadcastTest<SubNode>(
      "subMultiBroadcastOp7.onnxtxt", {1, 3, 1, 2}, true, 1, 2,
      [](float a, float b) { return a - b; });
}

TEST_F(OnnxImporterTest, importSubUniBroadcastOp6NoAxis) {
  importArithMultiBroadcastTest<SubNode>(
      "subUniBroadcastOp6NoAxis.onnxtxt", {1, 3, 4, 2}, false, 0, 2,
      [](float a, float b) { return a - b; });
}

TEST_F(OnnxImporterTest, importSubUniBroadcastOp6Axis) {
  importArithMultiBroadcastTest<SubNode>(
      "subUniBroadcastOp6Axis.onnxtxt", {1, 3, 4, 2}, false, 0, 2,
      [](float a, float b) { return a - b; });
}

TEST_F(OnnxImporterTest, importMulMultiBroadcastOp7) {
  importArithMultiBroadcastTest<MulNode>(
      "mulMultiBroadcastOp7.onnxtxt", {1, 3, 1, 2}, true, 1, 2,
      [](float a, float b) { return a * b; });
}

TEST_F(OnnxImporterTest, importMulUniBroadcastOp6NoAxis) {
  importArithMultiBroadcastTest<MulNode>(
      "mulUniBroadcastOp6NoAxis.onnxtxt", {1, 3, 4, 2}, false, 0, 2,
      [](float a, float b) { return a * b; });
}

TEST_F(OnnxImporterTest, importMulUniBroadcastOp6Axis) {
  importArithMultiBroadcastTest<MulNode>(
      "mulUniBroadcastOp6Axis.onnxtxt", {1, 3, 4, 2}, false, 0, 2,
      [](float a, float b) { return a * b; });
}

TEST_F(OnnxImporterTest, importDivMultiBroadcastOp7) {
  importArithMultiBroadcastTest<DivNode>(
      "divMultiBroadcastOp7.onnxtxt", {1, 3, 1, 2}, true, 1, 2,
      [](float a, float b) { return a / b; });
}

TEST_F(OnnxImporterTest, importDivUniBroadcastOp6NoAxis) {
  importArithMultiBroadcastTest<DivNode>(
      "divUniBroadcastOp6NoAxis.onnxtxt", {1, 3, 4, 2}, false, 0, 2,
      [](float a, float b) { return a / b; });
}

TEST_F(OnnxImporterTest, importDivUniBroadcastOp6Axis) {
  importArithMultiBroadcastTest<DivNode>(
      "divUniBroadcastOp6Axis.onnxtxt", {1, 3, 4, 2}, false, 0, 2,
      [](float a, float b) { return a / b; });
}

/// This tests reproduces issue #2135.
TEST_F(OnnxImporterTest, importUniBroadcastMultiOutput) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetFilename = std::string(
      GLOW_DATA_PATH "tests/models/onnxModels/UniBroadcastIssue2135.onnxtxt");
  Tensor data(ElemKind::FloatTy, {20});
  ONNXModelLoader onnxLD(NetFilename, {"data"}, {&data.getType()}, *F);
  (void)onnxLD;
}

/// Test loading of Elementwise Unary Ops floating point.
static void testEltwiseUnaryOpFloat(std::string fileName,
                                    llvm::ArrayRef<dim_t> inputShape,
                                    std::string input_name, float delta,
                                    const std::function<float(float)> &op) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  std::string NetFilename =
      std::string(GLOW_DATA_PATH "tests/models/onnxModels/") + fileName;
  PlaceholderBindings bindings;
  Placeholder *graphOutputVar;
  Type input_type(ElemKind::FloatTy, inputShape);
  ONNXModelLoader onnxLD(NetFilename, {input_name.c_str()}, {&input_type}, *F);
  graphOutputVar = EXIT_ON_ERR(onnxLD.getSingleOutput());
  auto PH = mod.getPlaceholderByName(input_name);
  auto *inTensor = bindings.allocate(PH);
  inTensor->getHandle().randomize(-10.0, 10.0, mod.getPRNG());
  // Compile&run the graph, and check the output
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

TEST_F(OnnxImporterTest, importExp) {
  testEltwiseUnaryOpFloat("exp.onnxtxt", {1, 2, 4, 3}, "data", 0.002,
                          [](float a) { return std::exp(a); });
}

static void testImportPRelu(std::string filename,
                            llvm::ArrayRef<dim_t> inputShape,
                            std::vector<float> expectedSlope) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetFileName =
      std::string(GLOW_DATA_PATH "tests/models/onnxModels/") + filename;

  PlaceholderBindings bindings;
  Placeholder *graphOutputVar;
  // Destroy the loader after the graph is loaded since the following execution
  // will not depend on anyting from the loader.
  Tensor data(ElemKind::FloatTy, inputShape);
  data.getHandle().randomize(-4.0, 4.0, mod.getPRNG());
  {
    ONNXModelLoader onnxLoader(NetFileName, {"data"}, {&data.getType()}, *F);
    graphOutputVar = EXIT_ON_ERR(onnxLoader.getSingleOutput());
    bindings.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(bindings, &mod, {"data"}, {&data});
  }

  // Compile&run the graph, and check the output.
  EE.compile(CompilationMode::Infer);
  EE.run(bindings);
  auto dataH = bindings.get(bindings.getPlaceholderByName("data"))->getHandle();
  auto result = bindings.get(graphOutputVar)->getHandle();
  std::vector<dim_t> expectedDims = {inputShape[0], inputShape[1],
                                     inputShape[2], inputShape[3]};

  EXPECT_TRUE(result.dims().vec() == expectedDims);
  for (size_t i = 0; i < dataH.size(); i++) {
    float expectedVal = expectedSlope[i] * std::min<float>(0, dataH.raw(i)) +
                        std::max<float>(0, dataH.raw(i));
    EXPECT_FLOAT_EQ(result.raw(i), expectedVal);
  }

  // Constant Folding Test.
  FAIL_TEST_IF_ERR(checkConstFoldedOutput(NetFileName, {"data"}, {&data},
                                          {bindings.get(graphOutputVar)}));
}

TEST_F(OnnxImporterTest, importPreluSlopeHasSameShape) {
  // The expected slope values correspond to the pre-broadcast
  // initializer values in the model file.
  std::vector<float> expectedSlope = {1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0,
                                      3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0};
  testImportPRelu("preluSlopeHasSameShape.onnxtxt", {1, 4, 2, 2},
                  expectedSlope);
}

TEST_F(OnnxImporterTest, importPReluBroadcastSlope) {
  // The expected slope values correspond to the pre-broadcast
  // initializer values in the model file.
  std::vector<float> expectedSlope = {1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0,
                                      3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0};
  testImportPRelu("preluBroadcastSlope.onnxtxt", {1, 4, 2, 2}, expectedSlope);
}

/// Expects failure to load PRelu in case of invalid slope shape.
TEST_F(OnnxImporterTest, importPReluInvalidBroadcastSlope) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetFileName =
      std::string(GLOW_DATA_PATH
                  "tests/models/onnxModels/preluInvalidBroadcastSlope.onnxtxt");

  // Destroy the loader after the graph is loaded since the following execution
  // will not depend on anyting from the loader.
  {
    Tensor data(ElemKind::FloatTy, {1, 4, 2, 2});
    EXPECT_DEATH(ONNXModelLoader(NetFileName, {"data"}, {&data.getType()}, *F),
                 "");
  }
}

/// Helper method to run the Conv operator test cases.
/// \p filename contains the model .onnxtxt.
/// \p expectedDims: output Tensor dimensions.
/// \p expectedValues : output Tensor values expected.
/// The input is N*C*H*W (1*1*3*3), the kernels is {2, 2},
/// strides is {1, 1}, group is 1. Pads can vary.
static void convTestHelper(std::string &filename,
                           const llvm::ArrayRef<dim_t> expectedDims,
                           const llvm::ArrayRef<float> expectedValues) {

  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetFilename =
      std::string(GLOW_DATA_PATH "tests/models/onnxModels/") + filename;

  PlaceholderBindings bindings;
  Placeholder *graphOutputVar;
  // Destroy the loader after the graph is loaded since the following execution
  // will not depend on anyting from the loader.
  {
    Tensor data;
    getNCHWData(&data, 1, 1, 3, 3);
    ONNXModelLoader onnxLD(NetFilename, {"data"}, {&data.getType()}, *F);
    graphOutputVar = EXIT_ON_ERR(onnxLD.getSingleOutput());
    bindings.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(bindings, &mod, {"data"}, {&data});
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

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);
  auto result = bindings.get(graphOutputVar)->getHandle();
  EXPECT_TRUE(result.dims() == expectedDims);
  for (size_t i = 0, e = expectedValues.size(); i < e; i++) {
    EXPECT_FLOAT_EQ(result.raw(i), expectedValues[i]);
  }
}

/// Test loading conv op from a ONNX model.
/// The input is N*C*H*W (1*1*3*3), the kernels is {2, 2},
/// strides is {1, 1}, pads is {1, 1, 1, 1}, group is 1.
TEST_F(OnnxImporterTest, importConv) {
  std::string filename("simpleConv.onnxtxt");
  std::vector<dim_t> expectedDims = {1, 1, 4, 4};
  std::vector<float> expectedValues = {2,  3,  5,  4,  5, 10, 14, 9,
                                       11, 22, 26, 15, 8, 15, 17, 10};
  convTestHelper(filename, expectedDims, expectedValues);
}

/// Test loading conv op from a ONNX model.
/// The input is N*C*H*W (1*1*3*3), the kernels is {2, 2},
/// strides is {1, 1}, auto_pad VALID (i.e. no padding), group is 1.
TEST_F(OnnxImporterTest, importConvAutoPadValid) {
  std::string filename("simpleConvAutoPadValid.onnxtxt");
  std::vector<dim_t> expectedDims = {1, 1, 2, 2};
  std::vector<float> expectedValues = {10, 14, 22, 26};
  convTestHelper(filename, expectedDims, expectedValues);
}

/// Test loading conv op from a ONNX model.
/// The input is N*C*H*W (1*1*3*3), the kernels is {2, 2},
/// strides is {1, 1}, auto_pad SAME_UPPER, group is 1.
TEST_F(OnnxImporterTest, importConvAutoPadSameUpper) {
  std::string filename("simpleConvAutoPadSameUpper.onnxtxt");
  std::vector<dim_t> expectedDims = {1, 1, 3, 3};
  std::vector<float> expectedValues = {10, 14, 9, 22, 26, 15, 15, 17, 10};
  convTestHelper(filename, expectedDims, expectedValues);
}

/// Test loading conv op from a ONNX model.
/// The input is N*C*H*W (1*1*3*3), the kernels is {2, 2},
/// strides is {1, 1}, auto_pad SAME_LOWER, group is 1.
TEST_F(OnnxImporterTest, importConvAutoPadSameLower) {
  std::string filename("simpleConvAutoPadSameLower.onnxtxt");
  std::vector<dim_t> expectedDims = {1, 1, 3, 3};
  std::vector<float> expectedValues = {2, 3, 5, 5, 10, 14, 11, 22, 26};
  convTestHelper(filename, expectedDims, expectedValues);
}

/// Test to ensure error handling for missing bias
/// input is handled correctly. Remaining input is
/// still sane to make sure it only fails for the
/// intended case.
TEST_F(OnnxImporterTest, importConvBiasFail) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetFilename(GLOW_DATA_PATH
                          "tests/models/onnxModels/simpleConvBiasFail.onnxtxt");

  // Destroy the loader after the graph is loaded since the following execution
  // will not depend on anyting from the loader.
  {
    Tensor data;
    getNCHWData(&data, 1, 1, 3, 3);

    EXPECT_DEATH(ONNXModelLoader(NetFilename, {"data"}, {&data.getType()}, *F),
                 "");
  }
}

/// Helper method to run the AveragePool operator test cases.
/// \p filename contains the model .onnxtxt.
/// \p expectedDims: output Tensor dimensions.
/// \p expectedValues : output Tensor values expected.
/// \p global: GlobalAveragePool if true, AveragePool if false.
/// The input is N*C*H*W (1*1*3*3), the kernels is {2, 2},
/// strides is {1, 1}, group is 1. Pads can vary in filename.
static void averagePoolTestHelper(std::string &filename,
                                  const llvm::ArrayRef<dim_t> expectedDims,
                                  const llvm::ArrayRef<float> expectedValues) {

  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetFilename =
      std::string(GLOW_DATA_PATH "tests/models/onnxModels/") + filename;

  PlaceholderBindings bindings;
  Placeholder *graphOutputVar;
  // Destroy the loader after the graph is loaded since the following execution
  // will not depend on anyting from the loader.
  Tensor data;
  getNCHWData(&data, 1, 1, 3, 3);
  {
    ONNXModelLoader onnxLD(NetFilename, {"x"}, {&data.getType()}, *F);
    graphOutputVar = EXIT_ON_ERR(onnxLD.getSingleOutput());
    bindings.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(bindings, &mod, {"x"}, {&data});
  }

  // ONNX importer loads a AveragePool node and converts it to 4 ops:
  // Transpose (input)   -> AveragePool -> Transpose -> Save
  EXPECT_EQ(F->getNodes().size(), 4);
  EXPECT_EQ(mod.getPlaceholders().size(), 2);

  auto *saveNode = getSaveNodeFromDest(graphOutputVar);
  auto *node = saveNode->getInput().getNode();

  EXPECT_TRUE(node->getKind() == Kinded::Kind::TransposeNodeKind);
  auto *poolNode = llvm::dyn_cast<TransposeNode>(node)->getInput().getNode();

  EXPECT_TRUE(poolNode->getKind() == Kinded::Kind::AvgPoolNodeKind);
  auto *tInNode = llvm::dyn_cast<AvgPoolNode>(poolNode)->getInput().getNode();

  EXPECT_TRUE(tInNode->getKind() == Kinded::Kind::TransposeNodeKind);

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);
  auto result = bindings.get(graphOutputVar)->getHandle();
  EXPECT_TRUE(result.dims() == expectedDims);
  for (size_t i = 0, e = expectedValues.size(); i < e; i++) {
    EXPECT_FLOAT_EQ(result.raw(i), expectedValues[i]);
  }

  // Constant Folding Test.
  FAIL_TEST_IF_ERR(checkConstFoldedOutput(NetFilename, {"x"}, {&data},
                                          {bindings.get(graphOutputVar)}));
}

/// Test loading AveragePool op from a ONNX model.
/// The input is N*C*H*W (1*1*3*3), the kernels is {2, 2},
/// strides is {1, 1}, pads is auto_pad VALID (no padding), group is 1.
TEST_F(OnnxImporterTest, importAveragePool2DAutoPadValid) {
  std::string filename("averagePool2DAutoPadValid.onnxtxt");
  std::vector<dim_t> expectedDims = {1, 1, 2, 2};
  std::vector<float> expectedValues = {2, 3, 5, 6};
  averagePoolTestHelper(filename, expectedDims, expectedValues);
}

/// Test loading AveragePool op from a ONNX model.
/// The input is N*C*H*W (1*1*3*3), the kernels is {2, 2},
/// strides is {1, 1}, pads is auto_pad SAME_UPPER, group is 1.
TEST_F(OnnxImporterTest, importAveragePool2DAutoPadSameUpper) {
  std::string filename("averagePool2DAutoPadSameUpper.onnxtxt");
  std::vector<dim_t> expectedDims = {1, 1, 3, 3};
  std::vector<float> expectedValues = {2, 3, 1.75, 5, 6, 3.25, 3.25, 3.75, 2};
  averagePoolTestHelper(filename, expectedDims, expectedValues);
}

/// Test loading AveragePool op from a ONNX model.
/// The input is N*C*H*W (1*1*3*3), the kernels is {2, 2},
/// strides is {1, 1}, pads is auto_pad SAME_LOWER, group is 1.
TEST_F(OnnxImporterTest, importAveragePool2DAutoPadSameLower) {
  std::string filename("averagePool2DAutoPadSameLower.onnxtxt");
  std::vector<dim_t> expectedDims = {1, 1, 3, 3};
  std::vector<float> expectedValues = {0, 0.25, 0.75, 0.75, 2, 3, 2.25, 5, 6};
  averagePoolTestHelper(filename, expectedDims, expectedValues);
}

TEST_F(OnnxImporterTest, importAveragePool3D) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetFilename(GLOW_DATA_PATH
                          "tests/models/onnxModels/averagePool3D.onnxtxt");

  // Destroy the loader after the graph is loaded since the following execution
  // will not depend on anyting from the loader.
  {
    Tensor data(ElemKind::FloatTy, {1, 3, 32, 32, 32});
    EXPECT_DEATH(ONNXModelLoader(NetFilename, {"x"}, {&data.getType()}, *F),
                 "");
  }
}

/// Test loading ReduceMean op from a ONNX model.
/// Input shape is 4D, one dimension is reduced, and output shape is 3D.
TEST_F(OnnxImporterTest, reduceMean4Dto3D) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string netFilename(GLOW_DATA_PATH
                          "tests/models/onnxModels/reduceMean4Dto3D.onnxtxt");

  PlaceholderBindings bindings;
  Placeholder *output;
  Tensor x(ElemKind::FloatTy, {2, 2, 2, 2});
  x.getHandle() = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  {
    ONNXModelLoader onnxLD(netFilename, {"x"}, {&x.getType()}, *F);
    output = EXIT_ON_ERR(onnxLD.getSingleOutput());
    bindings.allocate(mod.getPlaceholders());

    updateInputPlaceholdersByName(bindings, &mod, {"x"}, {&x});
  }

  auto *res = bindings.get(output);
  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  auto result = res->getHandle();
  std::vector<dim_t> expectedDims = {2, 2, 2};
  std::vector<float> expectedValues = {
      1.5, 3.5, 5.5, 7.5, 9.5, 11.5, 13.5, 15.5,
  };

  EXPECT_TRUE(result.dims().vec() == expectedDims);
  for (size_t i = 0; i < 8; i++) {
    EXPECT_FLOAT_EQ(result.raw(i), expectedValues[i]);
  }

  // Constant Folding Test.
  FAIL_TEST_IF_ERR(
      checkConstFoldedOutput(netFilename, {"x"}, {&x}, {bindings.get(output)}));
}

/// Test loading ReduceMean op from a ONNX model.
/// Input shape is 4D, one dimension is reduced, and output shape stays 4D.
TEST_F(OnnxImporterTest, reduceMean4Dto4D) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string netFilename(GLOW_DATA_PATH
                          "tests/models/onnxModels/reduceMean4Dto4D.onnxtxt");

  PlaceholderBindings bindings;
  Placeholder *output;
  Tensor x(ElemKind::FloatTy, {2, 2, 2, 2});
  x.getHandle() = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

  {

    ONNXModelLoader onnxLD(netFilename, {"x"}, {&x.getType()}, *F);
    output = EXIT_ON_ERR(onnxLD.getSingleOutput());
    bindings.allocate(mod.getPlaceholders());

    updateInputPlaceholdersByName(bindings, &mod, {"x"}, {&x});
  }

  auto *res = bindings.get(output);
  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  auto result = res->getHandle();
  std::vector<dim_t> expectedDims = {2, 2, 2, 1};
  std::vector<float> expectedValues = {
      1.5, 3.5, 5.5, 7.5, 9.5, 11.5, 13.5, 15.5,
  };

  EXPECT_TRUE(result.dims().vec() == expectedDims);
  for (size_t i = 0; i < 8; i++) {
    EXPECT_FLOAT_EQ(result.raw(i), expectedValues[i]);
  }

  // Constant Folding Test.
  FAIL_TEST_IF_ERR(
      checkConstFoldedOutput(netFilename, {"x"}, {&x}, {bindings.get(output)}));
}

/// Test loading ReduceSum op from a ONNX model.
/// Input shape is 4D, one dimension is reduced, and output shape is 4D.
TEST_F(OnnxImporterTest, reduceSum4D) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string netFilename(GLOW_DATA_PATH
                          "tests/models/onnxModels/reduceSum4D.onnxtxt");

  PlaceholderBindings bindings;
  Placeholder *output;
  Tensor x(ElemKind::FloatTy, {2, 2, 2, 2});
  x.getHandle() = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

  {

    ONNXModelLoader onnxLD(netFilename, {"x"}, {&x.getType()}, *F);
    output = EXIT_ON_ERR(onnxLD.getSingleOutput());
    bindings.allocate(mod.getPlaceholders());

    updateInputPlaceholdersByName(bindings, &mod, {"x"}, {&x});
  }

  auto *res = bindings.get(output);
  EE.compile(CompilationMode::Infer);
  EE.run(bindings);
  auto result = res->getHandle();
  std::vector<dim_t> expectedDims = {2, 2, 2, 1};
  std::vector<float> expectedValues = {3, 7, 11, 15, 19, 23, 27, 31};

  EXPECT_TRUE(result.dims().vec() == expectedDims);
  for (size_t i = 0; i < 8; i++) {
    EXPECT_FLOAT_EQ(result.raw(i), expectedValues[i]);
  }
  // Constant Folding Test.
  FAIL_TEST_IF_ERR(
      checkConstFoldedOutput(netFilename, {"x"}, {&x}, {bindings.get(output)}));
}

/// Test loading ReduceMean op from a ONNX model.
/// Input shape is 4D, two dimensions are reduced, targeting ReduceMean
/// optimization using AvgPool. Output shape is 4D.
TEST_F(OnnxImporterTest, reduceMean2AvgPoolKeepDims) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string netFilename(GLOW_DATA_PATH
                          "tests/models/onnxModels/reduceMean2AvgPool.onnxtxt");

  PlaceholderBindings bindings;
  Placeholder *output;
  Tensor x(ElemKind::FloatTy, {2, 2, 2, 2});
  x.getHandle() = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

  {

    ONNXModelLoader onnxLD(netFilename, {"x"}, {&x.getType()}, *F);
    output = EXIT_ON_ERR(onnxLD.getSingleOutput());
    bindings.allocate(mod.getPlaceholders());

    updateInputPlaceholdersByName(bindings, &mod, {"x"}, {&x});
  }

  auto *res = bindings.get(output);
  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  auto result = res->getHandle();
  std::vector<dim_t> expectedDims = {2, 2, 1, 1};
  std::vector<float> expectedValues = {
      2.5,
      6.5,
      10.5,
      14.5,
  };

  EXPECT_TRUE(result.dims().vec() == expectedDims);
  for (size_t i = 0; i < 4; i++) {
    EXPECT_FLOAT_EQ(result.raw(i), expectedValues[i]);
  }
  // Constant Folding Test.
  FAIL_TEST_IF_ERR(
      checkConstFoldedOutput(netFilename, {"x"}, {&x}, {bindings.get(output)}));
}

/// Test loading ReduceMean op from a ONNX model.
/// Input shape is 4D, two dimensions are reduced, targeting ReduceMean
/// optimization using AvgPool. Output shape is 2D.
TEST_F(OnnxImporterTest, reduceMean2AvgPoolNoKeepDims) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string netFilename(
      GLOW_DATA_PATH
      "tests/models/onnxModels/reduceMean2AvgPoolNoKeep.onnxtxt");

  PlaceholderBindings bindings;
  Placeholder *output;
  Tensor x(ElemKind::FloatTy, {2, 2, 2, 2});
  x.getHandle() = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

  {

    ONNXModelLoader onnxLD(netFilename, {"x"}, {&x.getType()}, *F);
    output = EXIT_ON_ERR(onnxLD.getSingleOutput());
    bindings.allocate(mod.getPlaceholders());

    updateInputPlaceholdersByName(bindings, &mod, {"x"}, {&x});
  }

  auto *res = bindings.get(output);
  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  auto result = res->getHandle();
  std::vector<dim_t> expectedDims = {2, 2};
  std::vector<float> expectedValues = {
      2.5,
      6.5,
      10.5,
      14.5,
  };

  EXPECT_TRUE(result.dims().vec() == expectedDims);
  for (size_t i = 0; i < 4; i++) {
    EXPECT_FLOAT_EQ(result.raw(i), expectedValues[i]);
  }

  // Constant Folding Test.
  FAIL_TEST_IF_ERR(
      checkConstFoldedOutput(netFilename, {"x"}, {&x}, {bindings.get(output)}));
}

/// Test loading ReduceMin op from a ONNX model.
/// Input shape is 4D, two dimensions are reduced,Output shape is 4D.
TEST_F(OnnxImporterTest, reduceMinKeepDims) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string netFilename(GLOW_DATA_PATH
                          "tests/models/onnxModels/reduceMin.onnxtxt");

  PlaceholderBindings bindings;
  Placeholder *output;
  Tensor x(ElemKind::FloatTy, {2, 2, 2, 2});
  x.getHandle() = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

  {

    ONNXModelLoader onnxLD(netFilename, {"x"}, {&x.getType()}, *F);
    output = EXIT_ON_ERR(onnxLD.getSingleOutput());
    bindings.allocate(mod.getPlaceholders());

    updateInputPlaceholdersByName(bindings, &mod, {"x"}, {&x});
  }

  auto *res = bindings.get(output);
  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  auto result = res->getHandle();
  std::vector<dim_t> expectedDims = {2, 2, 1, 1};
  std::vector<float> expectedValues = {1, 5, 9, 13};

  EXPECT_TRUE(result.dims().vec() == expectedDims);
  for (dim_t i = 0; i < result.size(); i++) {
    EXPECT_FLOAT_EQ(result.raw(i), expectedValues[i]);
  }
}

/// Test loading ReduceMean op from a ONNX model.
/// Input shape is 4D, two dimensions are reduced, targeting ReduceMean
/// optimization using AvgPool. Output shape is 2D.
TEST_F(OnnxImporterTest, reduceMinNoKeepDims) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string netFilename(GLOW_DATA_PATH
                          "tests/models/onnxModels/reduceMinNoKeep.onnxtxt");

  PlaceholderBindings bindings;
  Placeholder *output;
  Tensor x(ElemKind::FloatTy, {2, 2, 2, 2});
  x.getHandle() = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

  {

    ONNXModelLoader onnxLD(netFilename, {"x"}, {&x.getType()}, *F);
    output = EXIT_ON_ERR(onnxLD.getSingleOutput());
    bindings.allocate(mod.getPlaceholders());

    updateInputPlaceholdersByName(bindings, &mod, {"x"}, {&x});
  }

  auto *res = bindings.get(output);
  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  auto result = res->getHandle();
  std::vector<dim_t> expectedDims = {2, 2};
  std::vector<float> expectedValues = {
      1,
      5,
      9,
      13,
  };

  EXPECT_TRUE(result.dims().vec() == expectedDims);
  for (dim_t i = 0; i < result.size(); i++) {
    EXPECT_FLOAT_EQ(result.raw(i), expectedValues[i]);
  }
}

/// Test loading ReduceMin op from a ONNX model.
/// Input shape is 4D, two dimensions are reduced,Output shape is 4D.
TEST_F(OnnxImporterTest, reduceMinKeepDimsDefaultAxis) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string netFilename(
      GLOW_DATA_PATH "tests/models/onnxModels/reduceMinDefaultAxis.onnxtxt");

  PlaceholderBindings bindings;
  Placeholder *output;
  Tensor x(ElemKind::FloatTy, {2, 2, 2, 2});
  x.getHandle() = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

  {

    ONNXModelLoader onnxLD(netFilename, {"x"}, {&x.getType()}, *F);
    output = EXIT_ON_ERR(onnxLD.getSingleOutput());
    bindings.allocate(mod.getPlaceholders());

    updateInputPlaceholdersByName(bindings, &mod, {"x"}, {&x});
  }

  auto *res = bindings.get(output);
  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  auto result = res->getHandle();
  std::vector<dim_t> expectedDims = {1, 1, 1, 1};
  std::vector<float> expectedValues = {1};

  EXPECT_TRUE(result.dims().vec() == expectedDims);
  for (size_t i = 0; i < result.size(); i++) {
    EXPECT_FLOAT_EQ(result.raw(i), expectedValues[i]);
  }
}

/// Test loading SpaceToDepth op from an ONNX model.
TEST_F(OnnxImporterTest, spaceToDepth) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string netFilename(GLOW_DATA_PATH
                          "tests/models/onnxModels/spaceToDepth.onnxtxt");

  PlaceholderBindings bindings;
  Placeholder *output;
  {
    Tensor x(ElemKind::FloatTy, {1, 2, 4, 4});
    x.zero();

    ONNXModelLoader onnxLD(netFilename, {"x"}, {&x.getType()}, *F);
    output = EXIT_ON_ERR(onnxLD.getSingleOutput());
  }

  auto *save = getSaveNodeFromDest(output);
  TransposeNode *TRN =
      llvm::dyn_cast<TransposeNode>(save->getInput().getNode());
  ASSERT_TRUE(TRN);
  SpaceToDepthNode *STDN =
      llvm::dyn_cast<SpaceToDepthNode>(TRN->getInput().getNode());
  ASSERT_TRUE(STDN);
  unsigned blockSize = STDN->getBlockSize();
  EXPECT_EQ(blockSize, 2);
}

/// Test loading clip op from an ONNX model.
/// Test with arg min = 20.0 max = 60.0
TEST_F(OnnxImporterTest, importClip) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string netFilename(GLOW_DATA_PATH
                          "tests/models/onnxModels/clip.onnxtxt");

  PlaceholderBindings bindings;
  Placeholder *output;
  Tensor x(ElemKind::FloatTy, {3, 3});
  x.getHandle() = {1, 2, 3, 40, 5, 6, 7, 8, 90};

  {
    ONNXModelLoader onnxLD(netFilename, {"x"}, {&x.getType()}, *F);
    output = EXIT_ON_ERR(onnxLD.getSingleOutput());
    bindings.allocate(mod.getPlaceholders());

    updateInputPlaceholdersByName(bindings, &mod, {"x"}, {&x});
  }

  auto *res = bindings.get(output);
  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  auto result = res->getHandle();
  std::vector<dim_t> expectedDims = {3, 3};
  std::vector<float> expectedValues = {20, 20, 20, 40, 20, 20, 20, 20, 60};

  EXPECT_TRUE(result.dims().vec() == expectedDims);
  for (size_t i = 0; i < 3 * 3; i++) {
    EXPECT_FLOAT_EQ(result.raw(i), expectedValues[i]);
  }

  // Constant Folding Test.
  FAIL_TEST_IF_ERR(
      checkConstFoldedOutput(netFilename, {"x"}, {&x}, {bindings.get(output)}));
}

/// Test loading BatchMatMul op from an ONNX model.
TEST_F(OnnxImporterTest, importBatchMatMul) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  std::string netFilename(GLOW_DATA_PATH
                          "tests/models/onnxModels/batch_matmul.onnxtxt");

  PlaceholderBindings bindings;
  Placeholder *output;
  Tensor inputs_0(ElemKind::FloatTy, {20, 40, 7});
  Tensor inputs_1(ElemKind::FloatTy, {20, 7, 40});
  auto data_0 = inputs_0.getHandle();
  auto data_1 = inputs_1.getHandle();
  // Fill inputs with random positive values.
  data_0.randomize(0.0, 5.0, mod.getPRNG());
  data_1.randomize(1.0, 2.0, mod.getPRNG());
  {
    ONNXModelLoader onnxLD(netFilename, {"inputs_0", "inputs_1"},
                           {&inputs_0.getType(), &inputs_1.getType()}, *F);
    output = EXIT_ON_ERR(onnxLD.getSingleOutput());
    bindings.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(bindings, &mod, {"inputs_0", "inputs_1"},
                                  {&inputs_0, &inputs_1});
  }
  auto *res = bindings.get(output);
  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  auto result = res->getHandle();
  std::vector<dim_t> expectedDims = {20, 7, 7};
  EXPECT_EQ(result.dims().vec(), expectedDims);

  // High level check on the content of the graph.
  // We have 2 transpose, 20 * (matmul, 2 slices, 2 reshapes), 1 concat, 1
  // reshape, 1 save.
  EXPECT_EQ(F->getNodes().size(), 2 + 20 * 5 + 3);
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
  // Constant Folding Test.
  FAIL_TEST_IF_ERR(checkConstFoldedOutput(netFilename, {"inputs_0", "inputs_1"},
                                          {&inputs_0, &inputs_1},
                                          {bindings.get(output)}));
}

/// Test loading BatchBoxCox op from an ONNX model.
TEST_F(OnnxImporterTest, importBatchBoxCox) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string netFilename(GLOW_DATA_PATH
                          "tests/models/onnxModels/batchBoxCox.onnxtxt");

  PlaceholderBindings bindings;
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
  for (dim_t i = 0; i < kCols; i += 2) {
    lambda1H.at({i}) = 0;
  }

  {
    ONNXModelLoader onnxLD(
        netFilename, {"data", "lambda1", "lambda2"},
        {&data.getType(), &lambda1.getType(), &lambda2.getType()}, *F);
    output = EXIT_ON_ERR(onnxLD.getSingleOutput());
    bindings.allocate(mod.getPlaceholders());

    updateInputPlaceholdersByName(bindings, &mod,
                                  {"data", "lambda1", "lambda2"},
                                  {&data, &lambda1, &lambda2});
  }

  auto *res = bindings.get(output);
  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  auto result = res->getHandle();

  // Output should have the same dims as the inputs.
  EXPECT_TRUE(result.dims().vec() == data.dims().vec());

  // Compute elementwise Box-Cox transform and compare with corresponding
  // element of result.
  for (dim_t i = 0; i < kRows; ++i) {
    for (dim_t j = 0; j < kCols; ++j) {
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

  // Constant Folding Test.
  FAIL_TEST_IF_ERR(checkConstFoldedOutput(
      netFilename, {"data", "lambda1", "lambda2"}, {&data, &lambda1, &lambda2},
      {bindings.get(output)}));
}

/// Test loading DotProduct op from an ONNX model.
TEST_F(OnnxImporterTest, importDotProduct) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string netFilename(GLOW_DATA_PATH
                          "tests/models/onnxModels/dot_product.onnxtxt");

  Placeholder *output;
  {
    Tensor x(ElemKind::FloatTy, {3, 3});
    Tensor y(ElemKind::FloatTy, {3, 3});

    ONNXModelLoader onnxLD(netFilename, {"x", "y"},
                           {&x.getType(), &y.getType()}, *F);
    output = EXIT_ON_ERR(onnxLD.getSingleOutput());
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
TEST_F(OnnxImporterTest, importSumN) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  std::string netFilename(GLOW_DATA_PATH
                          "tests/models/onnxModels/sumN.onnxtxt");

  PlaceholderBindings bindings;
  Placeholder *output;
  Tensor i0(ElemKind::FloatTy, {3});
  i0.getHandle() = {1, 2, 3};
  Tensor i1(ElemKind::FloatTy, {3});
  i1.getHandle() = {4, 5, 6};
  Tensor i2(ElemKind::FloatTy, {3});
  i2.getHandle() = {7, 8, 9};
  {

    ONNXModelLoader onnxLD(netFilename, {"i0", "i1", "i2"},
                           {&i0.getType(), &i1.getType(), &i2.getType()}, *F);
    output = EXIT_ON_ERR(onnxLD.getSingleOutput());

    bindings.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(bindings, &mod, {"i0", "i1", "i2"},
                                  {&i0, &i1, &i2});
  }

  auto *res = bindings.get(output);
  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  auto result = res->getHandle();
  std::vector<dim_t> expectedDims = {3};
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
      llvm::dyn_cast<ConcatNode>(batchedReduceAdd->getBatch().getNode());
  ASSERT_TRUE(concat);
  for (size_t i = 0; i < 3; ++i) {
    auto *reshape =
        llvm::dyn_cast<ReshapeNode>(concat->getNthInput(i).getNode());
    ASSERT_TRUE(reshape);
  }

  // Constant Folding Test.
  FAIL_TEST_IF_ERR(checkConstFoldedOutput(netFilename, {"i0", "i1", "i2"},
                                          {&i0, &i1, &i2},
                                          {bindings.get(output)}));
}

/// Test loading Sum with one input and one output
TEST_F(OnnxImporterTest, importSum1) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  std::string netFilename(GLOW_DATA_PATH
                          "tests/models/onnxModels/sum1.onnxtxt");

  PlaceholderBindings bindings;
  Placeholder *output;
  Tensor x(ElemKind::FloatTy, {3});
  x.getHandle() = {1, 2, 3};

  {
    ONNXModelLoader onnxLD(netFilename, {"x"}, {&x.getType()}, *F);
    output = EXIT_ON_ERR(onnxLD.getSingleOutput());

    bindings.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(bindings, &mod, {"x"}, {&x});
  }

  auto *res = bindings.get(output);
  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  auto result = res->getHandle();
  std::vector<dim_t> expectedDims = {3};
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

  // Constant Folding Test.
  FAIL_TEST_IF_ERR(
      checkConstFoldedOutput(netFilename, {"x"}, {&x}, {bindings.get(output)}));
}

/// Test loading LengthsToRanges from an ONNX model.
TEST_F(OnnxImporterTest, importLengthsToRanges) {
  ExecutionEngine EE;
  auto &mod = EE.getModule();
  auto *F = mod.createFunction("main");
  std::string netFilename(GLOW_DATA_PATH
                          "tests/models/onnxModels/lengths_to_ranges.onnxtxt");
  Placeholder *output;
  {
    Tensor lengths(ElemKind::Int32ITy, {4});
    ONNXModelLoader onnxLD(netFilename, {"lengths"}, {&lengths.getType()}, *F);
    output = EXIT_ON_ERR(onnxLD.getSingleOutput());
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
TEST_F(OnnxImporterTest, importReplaceNaN) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string netFilename(GLOW_DATA_PATH
                          "tests/models/onnxModels/replaceNaN.onnxtxt");

  PlaceholderBindings bindings;
  Placeholder *output;
  Tensor x(ElemKind::FloatTy, {3, 3});

  {
    ONNXModelLoader onnxLD(netFilename, {"x"}, {&x.getType()}, *F);
    output = EXIT_ON_ERR(onnxLD.getSingleOutput());
    bindings.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(bindings, &mod, {"x"}, {&x});
  }

  // Verify structure: Input -> ReplaceNaN -> Save.
  EXPECT_EQ(F->getNodes().size(), 2);
  auto *saveNode = getSaveNodeFromDest(output);
  auto *replaceNaNNode =
      llvm::dyn_cast<ReplaceNaNNode>(saveNode->getInput().getNode());
  EXPECT_EQ(replaceNaNNode->getValue(), 1.0f);
  auto *inputNode =
      llvm::dyn_cast<Placeholder>(replaceNaNNode->getInput().getNode());
  ASSERT_EQ(inputNode, mod.getPlaceholderByName("x"));

  // We have one input and one output.
  EXPECT_EQ(mod.getPlaceholders().size(), 2);
}

/// Test loading SparseToDense op from an ONNX model.
TEST_F(OnnxImporterTest, importSparseToDense) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string netFilename(GLOW_DATA_PATH
                          "tests/models/onnxModels/sparseToDense.onnxtxt");

  PlaceholderBindings bindings;
  Placeholder *output;

  // Create inputs.
  constexpr dim_t kNumIndices = 5;
  constexpr dim_t kMaxIndex = 20;
  constexpr dim_t kRows = 10;
  constexpr dim_t kCols = 5;
  Tensor indices(IndexElemKind, {kNumIndices});
  Tensor values(ElemKind::FloatTy, {kNumIndices, kRows, kCols});
  Tensor dataToInferDim(ElemKind::FloatTy, {kMaxIndex, kRows, kCols});

  // Load model.
  {
    ONNXModelLoader onnxLD(
        netFilename, {"indices", "values", "dataToInferDim"},
        {&indices.getType(), &values.getType(), &dataToInferDim.getType()}, *F);
    output = EXIT_ON_ERR(onnxLD.getSingleOutput());
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

/// Test loading SparseLengthsSum from an ONNX model.
TEST_F(OnnxImporterTest, importSparseLengthsSum) {
  ExecutionEngine EE;
  auto &mod = EE.getModule();
  auto *F = mod.createFunction("main");
  std::string netFilename(GLOW_DATA_PATH
                          "tests/models/onnxModels/sparseLengthsSum.onnxtxt");
  Placeholder *output;
  {
    Tensor data(ElemKind::FloatTy, {2, 1});
    Tensor indices(IndexElemKind, {2});
    Tensor lengths(ElemKind::Int32ITy, {2});
    ONNXModelLoader onnxLD(
        netFilename, {"data", "indices", "lengths"},
        {&data.getType(), &indices.getType(), &lengths.getType()}, *F);
    output = EXIT_ON_ERR(onnxLD.getSingleOutput());
  }
  // Verify structure: PH, PH ->  SparseLengthsSum -> Save -> PH.
  //                  PH -> Splat /
  ASSERT_EQ(mod.getPlaceholders().size(), 4);
  ASSERT_EQ(F->getNodes().size(), 2);
  auto *save = getSaveNodeFromDest(output);
  auto *LS = llvm::dyn_cast<SparseLengthsSumNode>(save->getInput().getNode());
  ASSERT_TRUE(LS);
  ASSERT_TRUE(llvm::isa<Placeholder>(LS->getData()));
  ASSERT_TRUE(llvm::isa<Placeholder>(LS->getIndices()));
  ASSERT_TRUE(llvm::isa<Placeholder>(LS->getLengths()));
}

/// Test loading LengthsSum from an ONNX model.
TEST_F(OnnxImporterTest, importLengthsSum) {
  ExecutionEngine EE;
  auto &mod = EE.getModule();
  auto *F = mod.createFunction("main");
  std::string netFilename(GLOW_DATA_PATH
                          "tests/models/onnxModels/lengths_sum.onnxtxt");
  Placeholder *output;
  {
    Tensor data(ElemKind::FloatTy, {10, 2, 3});
    Tensor lengths(ElemKind::Int32ITy, {5});
    ONNXModelLoader onnxLD(netFilename, {"data", "lengths"},
                           {&data.getType(), &lengths.getType()}, *F);
    output = EXIT_ON_ERR(onnxLD.getSingleOutput());
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

/// Test loading CumSum from an ONNX model.
TEST_F(OnnxImporterTest, importCumSum) {
  ExecutionEngine EE;
  auto &mod = EE.getModule();
  auto *F = mod.createFunction("main");
  std::string netFilename(GLOW_DATA_PATH
                          "tests/models/onnxModels/cumsum.onnxtxt");
  Placeholder *output;
  {
    Tensor lengths(ElemKind::FloatTy, {10});
    lengths.getHandle() = {10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
    ONNXModelLoader onnxLD(netFilename, {"lengths"}, {&lengths.getType()}, *F);
    output = EXIT_ON_ERR(onnxLD.getSingleOutput());
  }
  // Verify structure: PH -> CumSum -> Save -> PH.
  ASSERT_EQ(mod.getPlaceholders().size(), 2);
  ASSERT_EQ(F->getNodes().size(), 2);
  auto *save = getSaveNodeFromDest(output);
  auto *CS = llvm::dyn_cast<CumSumNode>(save->getInput().getNode());
  ASSERT_TRUE(CS);
  ASSERT_TRUE(llvm::isa<Placeholder>(CS->getInput()));
  ASSERT_FALSE(CS->getExclusive());
  ASSERT_TRUE(CS->getReverse());
}

/// Test loading a FCTransposed node: I * W + B, where I is need to be flatten.
TEST_F(OnnxImporterTest, FCTransposedWithFlatten) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string netFilename(GLOW_DATA_PATH
                          "tests/models/onnxModels/FCTransposed.onnxtxt");

  Placeholder *output;

  {
    Tensor data(ElemKind::FloatTy, {2, 1, 3});
    data.getHandle() = {1, 2, 3, 4, 5, 6};
    ONNXModelLoader onnxLD(netFilename, {"data"}, {&data.getType()}, *F);
    output = EXIT_ON_ERR(onnxLD.getSingleOutput());
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

/// Test loading Constant from an ONNX model.
TEST_F(OnnxImporterTest, constant) {
  ExecutionEngine EE;
  auto &mod = EE.getModule();
  auto *F = mod.createFunction("main");
  std::string netFilename(GLOW_DATA_PATH
                          "tests/models/onnxModels/constant.onnxtxt");
  Placeholder *output;
  {
    ONNXModelLoader onnxLD(netFilename, {}, {}, *F);
    output = EXIT_ON_ERR(onnxLD.getSingleOutput());
    EXPECT_NE(output, nullptr);
  }
  // Constant -> Save -> PH
  ASSERT_EQ(mod.getPlaceholders().size(), 1);
  ASSERT_EQ(F->getNodes().size(), 1);
}

/// Test loading of testConstantOfShape.
template <class ElemType>
static void testConstantOfShape(std::string fileName, ElemType ref) {
  ExecutionEngine EE;
  auto &mod = EE.getModule();
  auto *F = mod.createFunction("main");
  PlaceholderBindings bindings;

  std::string netFilename =
      std::string(GLOW_DATA_PATH "tests/models/onnxModels/") + fileName;
  Placeholder *output;
  {
    ONNXModelLoader onnxLD(netFilename, {}, {}, *F);
    output = EXIT_ON_ERR(onnxLD.getSingleOutput());
    EXPECT_NE(output, nullptr);
  }
  // ConstantOfShape -> Save -> PH
  ASSERT_EQ(mod.getPlaceholders().size(), 1);
  ASSERT_EQ(F->getNodes().size(), 2);

  EE.compile(CompilationMode::Infer);
  bindings.allocate(mod.getPlaceholders());
  EE.run(bindings);

  auto result = bindings.get(output)->getHandle<ElemType>();
  for (size_t i = 0; i < result.getType().size(); i++) {
    ElemType val = result.raw(i);
    EXPECT_EQ(val, ref);
  }
}

/// Test loading of testConstantOfShape.
template <class ElemType>
static void testConstantOfShapeFailure(std::string fileName) {
  ExecutionEngine EE;
  auto &mod = EE.getModule();
  auto *F = mod.createFunction("main");
  std::string netFilename =
      std::string(GLOW_DATA_PATH "tests/models/onnxModels/") + fileName;
  ASSERT_DEATH(ONNXModelLoader(netFilename, {}, {}, *F), "losses");
}

TEST_F(OnnxImporterTest, importConstantOfShapeFloat) {
  testConstantOfShape<float>("constantOfShape.onnxtxt", 1.0F);
}

TEST_F(OnnxImporterTest, importConstantOfShapeInt32) {
  testConstantOfShape<int32_t>("constantOfShapeInt32.onnxtxt", 65535);
}

TEST_F(OnnxImporterTest, importConstantOfShapeInt64) {
  testConstantOfShape<int64_t>("constantOfShapeInt64.onnxtxt", 16777216LL);
}

TEST_F(OnnxImporterTest, importConstantOfShapeInt64LossFailure) {
  testConstantOfShapeFailure<int64_t>("constantOfShapeInt64Fail.onnxtxt");
}

TEST_F(OnnxImporterTest, importConstantOfShapeInt32LossFailure) {
  testConstantOfShapeFailure<int32_t>("constantOfShapeInt32Fail.onnxtxt");
}

/// Test loading ExpandDims from an ONNX model.
TEST_F(OnnxImporterTest, expandDims) {
  ExecutionEngine EE;
  auto &mod = EE.getModule();
  auto *F = mod.createFunction("main");
  std::string netFilename(GLOW_DATA_PATH
                          "tests/models/onnxModels/expandDims.onnxtxt");
  Placeholder *output;
  {
    Tensor x(ElemKind::FloatTy, {2, 2});
    ONNXModelLoader onnxLD(netFilename, {"x"}, {&x.getType()}, *F);
    output = EXIT_ON_ERR(onnxLD.getSingleOutput());
  }

  // Verify structure: PH -> Reshape -> Save -> PH.
  ASSERT_EQ(mod.getPlaceholders().size(), 2);
  ASSERT_EQ(F->getNodes().size(), 2);
  auto *save = getSaveNodeFromDest(output);
  auto *reshape = llvm::dyn_cast<ReshapeNode>(save->getInput().getNode());
  ASSERT_TRUE(reshape);
  EXPECT_TRUE(reshape->getDims().equals({1, 2, 2, 1}));
}

/// Test loading Gather from an ONNX model.
TEST_F(OnnxImporterTest, gather) {
  ExecutionEngine EE;
  auto &mod = EE.getModule();
  std::string netFilename(GLOW_DATA_PATH
                          "tests/models/onnxModels/gather.onnxtxt");
  auto *F = mod.createFunction("main");
  Placeholder *output;
  Tensor data(ElemKind::FloatTy, {3, 2});
  Tensor indices(ElemKind::Int32ITy, {2, 4});

  {
    ONNXModelLoader onnxLD(netFilename, {"data", "indices"},
                           {&data.getType(), &indices.getType()}, *F);
    output = EXIT_ON_ERR(onnxLD.getSingleOutput());
  }

  // Verify structure: PH/PH -> Gather -> Save -> PH.
  ASSERT_EQ(mod.getPlaceholders().size(), 3);
  ASSERT_EQ(F->getNodes().size(), 2);
  auto *save = getSaveNodeFromDest(output);
  auto *gather = llvm::dyn_cast<GatherNode>(save->getInput().getNode());
  ASSERT_TRUE(gather);
  EXPECT_TRUE(gather->getResult().dims().equals({2, 4, 2}));
}

/// Test loading GatherRanges from an ONNX model.
TEST_F(OnnxImporterTest, gatherRanges) {
  ExecutionEngine EE;
  auto &mod = EE.getModule();
  std::string netFilename(GLOW_DATA_PATH
                          "tests/models/onnxModels/gatherranges.onnxtxt");
  auto *F = mod.createFunction("main");
  Placeholder *output;
  Tensor data(ElemKind::FloatTy, {6});
  Tensor ranges(ElemKind::Int32ITy, {2, 2, 2});

  {
    ONNXModelLoader onnxLD(netFilename, {"data", "ranges"},
                           {&data.getType(), &ranges.getType()}, *F);
    output = EXIT_ON_ERR(onnxLD.getOutputByName("output"));
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

/// Test loading Gather ops with constant folding from an ONNX model.
TEST_F(OnnxImporterTest, gatherOpConstantFoldingAndReshape) {
  // This test verifies that Gather gets constant-folded, so that the argument
  // of the reshape becomes constant.
  ExecutionEngine EE;
  auto &mod = EE.getModule();
  std::string netFilename(
      GLOW_DATA_PATH "tests/models/onnxModels/gatherConstantFolding.onnxtxt");
  PlaceholderBindings bindings;
  auto *F = mod.createFunction("main");
  Placeholder *output;
  Tensor data(ElemKind::FloatTy, {1, 2, 4, 3});
  setConstantFoldLoaderOpsFlag(true);
  {
    ONNXModelLoader onnxLD(netFilename, {"input"}, {&data.getType()}, *F);
    output = EXIT_ON_ERR(onnxLD.getSingleOutput());
    bindings.allocate(mod.getPlaceholders());
  }
  EE.compile(CompilationMode::Infer);
  EE.run(bindings);
  setConstantFoldLoaderOpsFlag(false);

  auto result = bindings.get(output)->getHandle();
  std::vector<dim_t> expectedDims = {1, 4, 3, 2};
  EXPECT_TRUE(result.dims().vec() == expectedDims);
}

static void importSliceTest(std::string fileName, const char *inputName,
                            const llvm::ArrayRef<dim_t> inputShape,
                            const llvm::ArrayRef<dim_t> starts,
                            const llvm::ArrayRef<dim_t> outputShape) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetFilename =
      std::string(GLOW_DATA_PATH "tests/models/onnxModels/") + fileName;
  PlaceholderBindings bindings;
  Placeholder *graphOutputVar;
  // Destroy the loader after the graph is loaded since the following execution
  // will not depend on anyting from the loader.
  Tensor data;
  getNCHWData(&data, inputShape[0], inputShape[1], inputShape[2],
              inputShape[3]);
  {
    ONNXModelLoader onnxLD(NetFilename, {inputName}, {&data.getType()}, *F);
    graphOutputVar = EXIT_ON_ERR(onnxLD.getSingleOutput());
    bindings.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(bindings, &mod, {inputName}, {&data});
  }

  // ONNX importer loads an Slice operator and adds to the IR:
  // - a Slice node

  // Check the graph structure.
  auto *saveNode = getSaveNodeFromDest(graphOutputVar);
  auto *node = saveNode->getInput().getNode();
  auto *sliceNode = llvm::dyn_cast<SliceNode>(node);
  EXPECT_NE(nullptr, sliceNode);

  // Compile&run the graph, and check the output.
  EE.compile(CompilationMode::Infer);
  EE.run(bindings);
  auto result = bindings.get(graphOutputVar)->getHandle();
  EXPECT_TRUE(result.dims().vec() == outputShape.vec());
  dim_t wSliceSize = inputShape[3];
  dim_t hSliceSize = inputShape[2] * wSliceSize;
  dim_t cSliceSize = inputShape[1] * hSliceSize;
  dim_t indexOutput = 0;
  for (dim_t n = 0; n < outputShape[0]; n++) {
    for (dim_t c = 0; c < outputShape[1]; c++) {
      for (dim_t h = 0; h < outputShape[2]; h++) {
        for (dim_t w = 0; w < outputShape[3]; w++) {
          dim_t indexInput = (starts[0] + n) * cSliceSize +
                             (starts[1] + c) * hSliceSize +
                             (starts[2] + h) * wSliceSize + (starts[3] + w);
          EXPECT_FLOAT_EQ(result.raw(indexOutput++), indexInput);
        }
      }
    }
  }

  // Constant Folding Test.
  FAIL_TEST_IF_ERR(checkConstFoldedOutput(NetFilename, {inputName}, {&data},
                                          {bindings.get(graphOutputVar)}));
}

TEST_F(OnnxImporterTest, importSliceDynamicNoAxes) {
  importSliceTest("sliceDynamic.onnxtxt", "data", {2, 3, 3, 3} /* input */,
                  {0, 1, 1, 1} /* starts */, /* ends: {2, 2, 3, 3} */
                  {2, 1, 2, 2} /* output */);
}

TEST_F(OnnxImporterTest, importSliceAxesFull) {
  importSliceTest("sliceAxesFull.onnxtxt", "data", {2, 3, 3, 3} /* input */,
                  {0, 1, 1, 2} /* starts */, /* ends: {1, 2, 3, 3} */
                  {1, 1, 2, 1} /* output */);
}

TEST_F(OnnxImporterTest, importSliceAxesAnyOrder) {
  importSliceTest("sliceAxesAnyOrder.onnxtxt", "data", {2, 3, 3, 3} /* input */,
                  {1, 2, 0, 2} /* starts */, /* ends: {2, 3, 1, 3} */
                  {1, 1, 1, 1} /* output */);
}

TEST_F(OnnxImporterTest, importSliceAxesOverwrite) {
  importSliceTest("sliceAxesOverwrite.onnxtxt", "data",
                  {2, 3, 3, 3} /* input */,
                  {0, 1, 1, 2} /* starts */, /* ends: {1, 2, 3, 3} */
                  {1, 1, 2, 1} /* output */);
}

TEST_F(OnnxImporterTest, importSliceAxesPartial) {
  importSliceTest("sliceAxesPartial.onnxtxt", "data", {2, 3, 3, 3} /* input */,
                  {0, 1, 1, 0} /* starts */, /* ends: {2, 2, 3, 3} */
                  {2, 1, 2, 3} /* output */);
}

TEST_F(OnnxImporterTest, importSliceNoAxes) {
  importSliceTest("sliceNoAxes.onnxtxt", "data", {2, 3, 3, 3} /* input */,
                  {0, 1, 1, 1} /* starts */, /* ends: {2, 2, 3, 3} */
                  {2, 1, 2, 2} /* output */);
}

static void importCast(llvm::StringRef fileName, llvm::StringRef inputName,
                       const llvm::ArrayRef<dim_t> inputShape,
                       ElemKind outputKind) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetFilename =
      std::string(GLOW_DATA_PATH "tests/models/onnxModels/") + fileName.str();
  PlaceholderBindings bindings;
  Placeholder *graphOutputVar;
  {
    Tensor data;
    getNCHWData(&data, inputShape[0], inputShape[1], inputShape[2],
                inputShape[3]);
    ONNXModelLoader onnxLD(NetFilename, {inputName.str().c_str()},
                           {&data.getType()}, *F);
    graphOutputVar = EXIT_ON_ERR(onnxLD.getSingleOutput());
    bindings.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(bindings, &mod, {inputName}, {&data});
  }

  // ONNX importer loads a Cast operator and adds to the IR:
  // - a ConvertTo node

  // Check the graph structure.
  auto *saveNode = getSaveNodeFromDest(graphOutputVar);
  auto *node = saveNode->getInput().getNode();
  auto *castNode = llvm::dyn_cast<ConvertToNode>(node);
  ASSERT_NE(nullptr, castNode);

  // Check node output type.
  ASSERT_EQ(castNode->getResult().getType()->getElementType(), outputKind);
}

TEST_F(OnnxImporterTest, importCastToFloat) {
  importCast("castToFloat.onnxtxt", "data", {1, 2, 2, 2}, ElemKind::FloatTy);
}
TEST_F(OnnxImporterTest, importCastToFloat16) {
  importCast("castToFloat16.onnxtxt", "data", {1, 2, 2, 2},
             ElemKind::Float16Ty);
}
TEST_F(OnnxImporterTest, importCastToInt32) {
  importCast("castToInt32.onnxtxt", "data", {1, 2, 2, 2}, ElemKind::Int32ITy);
}
TEST_F(OnnxImporterTest, importCastToInt64) {
  importCast("castToInt64.onnxtxt", "data", {1, 2, 2, 2}, ElemKind::Int64ITy);
}

TEST_F(OnnxImporterTest, cast_32_64) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string netFilename(GLOW_DATA_PATH
                          "tests/models/onnxModels/castInt-32-64.onnxtxt");
  PlaceholderBindings bindings;
  Placeholder *graphOutputVar;
  std::vector<float> init(1 * 2 * 4 * 3);
  std::vector<float> expectedOut(1 * 2 * 4 * 3);
  for (size_t i = 0; i < init.size(); i++) {
    const float value = i * 12.345678f;
    init[i] = value;
    expectedOut[i] = int32_t(value);
  }
  {
    Tensor data(ElemKind::FloatTy, {1, 2, 4, 3});
    data.getHandle() = init;
    ONNXModelLoader onnxLD(netFilename, {"input"}, {&data.getType()}, *F);
    graphOutputVar = EXIT_ON_ERR(onnxLD.getSingleOutput());
    bindings.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(bindings, &mod, {"input"}, {&data});
  }

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);
  // Make sure that the optimizer did not eliminate float->int casts. They are
  // not NOOP. Conversions int32 -> int64 -> int32 are always NOOP, so they can
  // be optimized away.
  EXPECT_EQ(F->getNodes().size(), 3);
  auto result = bindings.get(graphOutputVar)->getHandle();
  std::vector<dim_t> expectedDims = {1, 2, 4, 3};

  EXPECT_TRUE(result.dims().vec() == expectedDims);
  for (size_t i = 0; i < expectedOut.size(); i++) {
    EXPECT_EQ(result.raw(i), expectedOut[i]);
  }
}

static void importPad(std::string fileName, const char *inputName,
                      const llvm::ArrayRef<dim_t> inputShape,
                      const llvm::ArrayRef<sdim_t> starts,
                      const llvm::ArrayRef<sdim_t> ends, PaddingMode mode,
                      float value, bool testOutput,
                      bool expectLoadError = false) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetFilename =
      std::string(GLOW_DATA_PATH "tests/models/onnxModels/") + fileName;
  PlaceholderBindings bindings;
  Placeholder *graphOutputVar;
  // Destroy the loader after the graph is loaded since the following execution
  // will not depend on anyting from the loader.
  {
    Tensor data;
    getNCHWData(&data, inputShape[0], inputShape[1], inputShape[2],
                inputShape[3]);
    if (expectLoadError) {
      Error err = Error::empty();
      ONNXModelLoader(NetFilename, {inputName}, {&data.getType()}, *F, &err);
      EXPECT_TRUE(ERR_TO_BOOL(std::move(err)));
      return;
    }
    ONNXModelLoader onnxLD(NetFilename, {inputName}, {&data.getType()}, *F);
    graphOutputVar = EXIT_ON_ERR(onnxLD.getSingleOutput());
    bindings.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(bindings, &mod, {inputName}, {&data});
  }

  // ONNX importer loads a Pad operator and adds to the IR:
  // - a Pad node

  // Check the graph structure.
  auto *saveNode = getSaveNodeFromDest(graphOutputVar);
  auto *node = saveNode->getInput().getNode();
  auto *padNode = llvm::dyn_cast<PadNode>(node);
  EXPECT_NE(nullptr, padNode);

  // Check Pad node properties.
  assert(padNode->getMode() == mode);
  if (mode == PaddingMode::CONSTANT) {
    EXPECT_EQ(value, padNode->getValue());
  }
  // Check the Pad node output shape.
  std::vector<dim_t> expectedOutputShape(inputShape.size());
  for (unsigned int i = 0; i < inputShape.size(); i++) {
    expectedOutputShape[i] =
        size_t(ssize_t(inputShape[i]) + starts[i] + ends[i]);
  }
  EXPECT_TRUE(padNode->getResult().dims().vec() == expectedOutputShape);

  // Currently, only constant with positive pads is supported at lowering.
  // We just consider this test case.
  if (testOutput && mode == PaddingMode::CONSTANT) {
    // Compile&run the graph, and check the output.
    EE.compile(CompilationMode::Infer);
    EE.run(bindings);
    auto result = bindings.get(graphOutputVar)->getHandle();
    EXPECT_TRUE(result.dims().vec() == expectedOutputShape);
    size_t indexOutput = 0;
    size_t indexinput = 0;
    for (size_t n = 0; n < expectedOutputShape[0]; n++) {
      for (size_t c = 0; c < expectedOutputShape[1]; c++) {
        for (size_t h = 0; h < expectedOutputShape[2]; h++) {
          for (size_t w = 0; w < expectedOutputShape[3]; w++) {
            float expectedValue = value;
            if ((n >= size_t(starts[0])) &&
                (n < (expectedOutputShape[0] - size_t(ends[0]))) &&
                (c >= size_t(starts[1])) &&
                (c < (expectedOutputShape[1] - size_t(ends[1]))) &&
                (h >= size_t(starts[2])) &&
                (h < (expectedOutputShape[2] - size_t(ends[2]))) &&
                (w >= size_t(starts[3])) &&
                (w < (expectedOutputShape[3] - size_t(ends[3])))) {
              // This is the way 'getNCHWData' initializes data.
              expectedValue = indexinput++;
            }
            EXPECT_FLOAT_EQ(result.raw(indexOutput++), expectedValue);
          }
        }
      }
    }
  }
}

TEST_F(OnnxImporterTest, importPadDefault) {
  importPad("padDefault.onnxtxt", "data", {4, 6, 5, 7} /* input */,
            {1, 2, -2, 0} /* starts */, {0, -2, 1, 2} /* ends */,
            PaddingMode::CONSTANT, 0.f, false);
}

TEST_F(OnnxImporterTest, importPadConstant) {
  importPad("padConstant.onnxtxt", "data", {4, 6, 5, 7} /* input */,
            {1, 2, -2, 0} /* starts */, {0, -2, 1, 2} /* ends */,
            PaddingMode::CONSTANT, 2.55f, false);
}

TEST_F(OnnxImporterTest, importPadReflect) {
  // Note: PaddingMode::REFLECT is not yet supported, so we assert death when
  // loading the model.
  importPad("padReflect.onnxtxt", "data", {4, 6, 5, 7} /* input */,
            {1, 2, -2, 0} /* starts */, {0, -2, 1, 2} /* ends */,
            PaddingMode::REFLECT, 0.f /* any */, false,
            /* expectLoadError */ true);
}

TEST_F(OnnxImporterTest, importPadEdge) {
  // Note: PaddingMode::EDGE is not yet supported, so we assert death when
  // loading the model.
  importPad("padEdge.onnxtxt", "data", {4, 6, 5, 7} /* input */,
            {1, 2, -2, 0} /* starts */, {0, -2, 1, 2} /* ends */,
            PaddingMode::EDGE, 0.f /* any */, false,
            /* expectLoadError */ true);
}

TEST_F(OnnxImporterTest, importPadConstantPositive) {
  importPad("padConstantPositive.onnxtxt", "data", {4, 6, 5, 7} /* input */,
            {1, 2, 3, 4} /* starts */, {0, 3, 1, 2} /* ends */,
            PaddingMode::CONSTANT, 2.55f, true);
}

/// Test loading BatchNorm with all optional outputs declared, but not used in
/// the model. Glow supports only the first mandatory output, but declaring
/// optional outputs while not using them in the model should not make the
/// import fail.
TEST_F(OnnxImporterTest, batchNormPR2304) {
  ExecutionEngine EE;
  auto &mod = EE.getModule();
  std::string netFilename(GLOW_DATA_PATH
                          "tests/models/onnxModels/batchNormPR2304.onnxtxt");
  auto *F = mod.createFunction("main");
  Placeholder *output;
  Tensor inputTensor(ElemKind::FloatTy, {1, 2, 10, 10});
  {
    ONNXModelLoader onnxLD(netFilename, {"input"}, {&inputTensor.getType()},
                           *F);
    output = EXIT_ON_ERR(onnxLD.getOutputByName("output"));
  }

  // Check the graph structure.
  auto *saveNode = getSaveNodeFromDest(output);
  auto *trNode = llvm::dyn_cast<TransposeNode>(saveNode->getInput().getNode());
  EXPECT_NE(nullptr, trNode);
  auto *bnNode =
      llvm::dyn_cast<BatchNormalizationNode>(trNode->getInput().getNode());
  EXPECT_NE(nullptr, bnNode);
}

/// Test constructor for auto loading inputs case.
TEST_F(OnnxImporterTest, autoLoadInputs) {
  ExecutionEngine EE;
  auto &mod = EE.getModule();
  std::string netFilename(GLOW_DATA_PATH
                          "tests/models/onnxModels/batchNormPR2304.onnxtxt");
  auto *F = mod.createFunction("main");
  Tensor inputTensor(ElemKind::FloatTy, {1, 2, 10, 10});
  llvm::StringRef inputName = "input";
  ONNXModelLoader onnxLD(netFilename, {}, {}, *F);
  auto inputs = onnxLD.getInputVarsMapping();
  EXPECT_EQ(inputs.size(), 1);
  EXPECT_TRUE(inputTensor.getType().isEqual(inputs[inputName]->getType()));
}

TEST_F(OnnxImporterTest, shape) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string netFilename(GLOW_DATA_PATH
                          "tests/models/onnxModels/shape.onnxtxt");

  PlaceholderBindings bindings;
  Placeholder *output;
  Tensor x(ElemKind::FloatTy, {2, 2, 2, 2});
  x.getHandle() = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

  {
    ONNXModelLoader onnxLD(netFilename, {"input"}, {&x.getType()}, *F);
    output = EXIT_ON_ERR(onnxLD.getSingleOutput());
    bindings.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(bindings, &mod, {"input"}, {&x});
  }

  auto *res = bindings.get(output);
  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  auto result = res->getHandle<int64_t>();
  std::vector<dim_t> expectedDims = {1};
  std::vector<int64_t> expectedValues = {4};

  EXPECT_TRUE(result.dims().vec() == expectedDims);
  for (size_t i = 0; i < expectedValues.size(); i++) {
    EXPECT_EQ(result.raw(i), expectedValues[i]);
  }

  // Constant Folding Test.
  FAIL_TEST_IF_ERR(checkConstFoldedOutput(netFilename, {"input"}, {&x},
                                          {bindings.get(output)}));
}

TEST_F(OnnxImporterTest, tile) {
  ExecutionEngine EE;
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string netFilename(GLOW_DATA_PATH
                          "tests/models/onnxModels/tile.onnxtxt");

  PlaceholderBindings bindings;
  Placeholder *output;
  {
    Tensor x(ElemKind::FloatTy, {1, 2, 2, 1});
    x.getHandle() = {1., 2., 3., 4.};

    ONNXModelLoader onnxLD(netFilename, {"input"}, {&x.getType()}, *F);
    output = EXIT_ON_ERR(onnxLD.getSingleOutput());
    bindings.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(bindings, &mod, {"input"}, {&x});
  }

  auto *res = bindings.get(output);
  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  auto result = res->getHandle();
  std::vector<dim_t> expectedDims = {1, 4, 4, 3};
  std::vector<float> expectedValues = {
      1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0,
      3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0,
      1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0,
      3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0,
  };

  EXPECT_TRUE(result.dims().vec() == expectedDims);
  for (size_t i = 0; i < expectedValues.size(); i++) {
    EXPECT_EQ(result.raw(i), expectedValues[i]);
  }
}

TEST_F(OnnxImporterTest, topK) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string netFilename(GLOW_DATA_PATH
                          "tests/models/onnxModels/TopK.onnxtxt");

  PlaceholderBindings bindings;
  Placeholder *output;
  Placeholder *index;
  Tensor x(ElemKind::FloatTy, {1, 3, 4});
  x.getHandle() = {1., 2., 3., 4., 8., 7., 7., 7., 11., 12., 11., 10.};

  {
    ONNXModelLoader onnxLD(netFilename, {"scores"}, {&x.getType()}, *F);
    output = EXIT_ON_ERR(onnxLD.getOutputByName("topscores"));
    index = EXIT_ON_ERR(onnxLD.getOutputByName("topindices"));
    bindings.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(bindings, &mod, {"scores"}, {&x});
  }

  auto *outputT = bindings.get(output);
  auto *indexT = bindings.get(index);

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  auto outputH = outputT->getHandle();
  auto indexH = indexT->getHandle<sdim_t>();
  std::vector<dim_t> expectedDims = {1, 3, 2};
  std::vector<float> expectedValues = {
      4., 3., 8., 7., 12, 11.,
  };
  std::vector<dim_t> expectedIndices = {3, 2, 0, 1, 1, 0};

  EXPECT_TRUE(outputH.dims().vec() == expectedDims);
  for (size_t i = 0; i < expectedValues.size(); i++) {
    EXPECT_EQ(outputH.raw(i), expectedValues[i]);
  }

  EXPECT_TRUE(indexH.dims().vec() == expectedDims);
  for (size_t i = 0; i < expectedIndices.size(); i++) {
    EXPECT_EQ(indexH.raw(i), expectedIndices[i]);
  }

  // Constant Folding Test.
  FAIL_TEST_IF_ERR(
      checkConstFoldedOutput(netFilename, {"scores"}, {&x}, {outputT, indexT}));
}

TEST_F(OnnxImporterTest, argMaxKeepDim) {
  ExecutionEngine EE;
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string netFilename(GLOW_DATA_PATH
                          "tests/models/onnxModels/ArgMaxKeepDim.onnxtxt");

  PlaceholderBindings bindings;
  Placeholder *argmaxPH;
  {
    Tensor inT(ElemKind::FloatTy, {2, 3, 4, 5});

    ONNXModelLoader onnxLD(netFilename, {"input"}, {&inT.getType()}, *F);
    argmaxPH = EXIT_ON_ERR(onnxLD.getOutputByName("argmax_scores"));
    bindings.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(bindings, &mod, {"input"}, {&inT});
  }

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  auto argmax = bindings.get(argmaxPH)->getHandle<sdim_t>();
  std::vector<dim_t> expectedDims = {2, 3, 1, 5};
  EXPECT_TRUE(argmax.dims().vec() == expectedDims);
}

TEST_F(OnnxImporterTest, argMaxNoKeepDim) {
  ExecutionEngine EE;
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string netFilename(GLOW_DATA_PATH
                          "tests/models/onnxModels/ArgMaxNoKeepDim.onnxtxt");

  PlaceholderBindings bindings;
  Placeholder *argmaxPH;
  {
    Tensor inT(ElemKind::FloatTy, {2, 3, 4, 5});

    ONNXModelLoader onnxLD(netFilename, {"input"}, {&inT.getType()}, *F);
    argmaxPH = EXIT_ON_ERR(onnxLD.getOutputByName("argmax_scores"));
    bindings.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(bindings, &mod, {"input"}, {&inT});
  }

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  auto argmax = bindings.get(argmaxPH)->getHandle<sdim_t>();
  std::vector<dim_t> expectedDims = {2, 4, 5};
  EXPECT_TRUE(argmax.dims().vec() == expectedDims);
}

TEST_F(OnnxImporterTest, argMaxDefault) {
  ExecutionEngine EE;
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string netFilename(GLOW_DATA_PATH
                          "tests/models/onnxModels/ArgMaxDefault.onnxtxt");

  PlaceholderBindings bindings;
  Placeholder *argmaxPH;
  {
    Tensor inT(ElemKind::FloatTy, {2, 3, 4, 5});

    ONNXModelLoader onnxLD(netFilename, {"input"}, {&inT.getType()}, *F);
    argmaxPH = EXIT_ON_ERR(onnxLD.getOutputByName("argmax_scores"));
    bindings.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(bindings, &mod, {"input"}, {&inT});
  }

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  auto argmax = bindings.get(argmaxPH)->getHandle<sdim_t>();
  std::vector<dim_t> expectedDims = {1, 3, 4, 5};
  EXPECT_TRUE(argmax.dims().vec() == expectedDims);
}

TEST_F(OnnxImporterTest, importMaxPoolWithArgmax) {
  ExecutionEngine EE;
  auto &mod = EE.getModule();
  std::string netFilename(GLOW_DATA_PATH
                          "tests/models/onnxModels/maxPoolWithArgmax.onnxtxt");
  auto *F = mod.createFunction("main");
  PlaceholderBindings bindings;
  Placeholder *resultPH, *indicesPH;
  Tensor inputTensor(ElemKind::FloatTy, {1, 3, 4, 4});

  // Execute the following scenario for MaxPool with Argmax output:
  // Input:
  // [[[[ 0. 47. 35. 23.]
  //    [11. 58. 46. 34.]
  //    [22. 10. 57. 45.]
  //    [33. 21.  9. 56.]]
  //
  //   [[44. 32. 20.  8.]
  //    [55. 43. 31. 19.]
  //    [ 7. 54. 42. 30.]
  //    [18.  6. 53. 41.]]
  //
  //   [[29. 17.  5. 52.]
  //    [40. 28. 16.  4.]
  //    [51. 39. 27. 15.]
  //    [ 3. 50. 38. 26.]]]]
  //
  // Result:
  // [[[[58. 46.]
  //    [33. 57.]]
  //
  //   [[55. 31.]
  //    [54. 53.]]
  //
  //   [[40. 52.]
  //    [51. 38.]]]]
  //
  // Argmax:
  // [[[[15 18]
  //    [36 30]]
  //
  //   [[13 19]
  //    [28 43]]
  //
  //   [[14 11]
  //    [26 44]]]]
  inputTensor.getHandle() = {
      0.0,  47.0, 35.0, 23.0, 11.0, 58.0, 46.0, 34.0, 22.0, 10.0, 57.0, 45.0,
      33.0, 21.0, 9.0,  56.0, 44.0, 32.0, 20.0, 8.0,  55.0, 43.0, 31.0, 19.0,
      7.0,  54.0, 42.0, 30.0, 18.0, 6.0,  53.0, 41.0, 29.0, 17.0, 5.0,  52.0,
      40.0, 28.0, 16.0, 4.0,  51.0, 39.0, 27.0, 15.0, 3.0,  50.0, 38.0, 26.0};

  {
    ONNXModelLoader onnxLD(netFilename, {"input"}, {&inputTensor.getType()},
                           *F);
    resultPH = EXIT_ON_ERR(onnxLD.getOutputByName("result"));
    indicesPH = EXIT_ON_ERR(onnxLD.getOutputByName("indices"));
    bindings.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(bindings, &mod, {"input"}, {&inputTensor});
  }

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  auto result = bindings.get(resultPH)->getHandle();
  auto indices = bindings.get(indicesPH)->getHandle<sdim_t>();
  std::vector<dim_t> expectedDims = {1, 3, 2, 2};

  EXPECT_TRUE(result.dims().vec() == expectedDims);
  EXPECT_TRUE(indices.dims().vec() == expectedDims);

  std::vector<float> expectedResult = {58.0, 46.0, 33.0, 57.0, 55.0, 31.0,
                                       54.0, 53.0, 40.0, 52.0, 51.0, 38.0};
  std::vector<sdim_t> expectedIndices = {15, 18, 36, 30, 13, 19,
                                         28, 43, 14, 11, 26, 44};

  for (size_t i = 0; i < expectedResult.size(); i++) {
    EXPECT_EQ(result.raw(i), expectedResult[i]);
    EXPECT_EQ(indices.raw(i), expectedIndices[i]);
  }
}

TEST_F(OnnxImporterTest, importWhere) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string netFilename(GLOW_DATA_PATH
                          "tests/models/onnxModels/Where.onnxtxt");

  Placeholder *out = nullptr;
  {
    Tensor condition(ElemKind::BoolTy, {1, 1, 4});
    Tensor X(ElemKind::FloatTy, {1, 4, 1});
    Tensor Y(ElemKind::FloatTy, {4, 1, 1});

    condition.zero();
    X.zero();
    Y.zero();

    ONNXModelLoader onnxLD(netFilename, {"Condition", "X", "Y"},
                           {&condition.getType(), &X.getType(), &Y.getType()},
                           *F);
    out = EXIT_ON_ERR(onnxLD.getOutputByName("Out"));
  }

  auto *save = getSaveNodeFromDest(out);

  SelectNode *WHR = llvm::dyn_cast<SelectNode>(save->getInput().getNode());

  ASSERT_TRUE(WHR);
  EXPECT_EQ(WHR->getResult().dims()[0], 4);
  EXPECT_EQ(WHR->getResult().dims()[1], 4);
  EXPECT_EQ(WHR->getResult().dims()[2], 4);
}

TEST_F(OnnxImporterTest, importLess) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string netFilename(GLOW_DATA_PATH
                          "tests/models/onnxModels/Less.onnxtxt");

  Placeholder *out = nullptr;
  {
    Tensor X(ElemKind::FloatTy, {1, 4, 1});
    Tensor Y(ElemKind::FloatTy, {4, 1, 1});
    X.zero();
    Y.zero();

    ONNXModelLoader onnxLD(netFilename, {"X", "Y"},
                           {&X.getType(), &Y.getType()}, *F);
    out = EXIT_ON_ERR(onnxLD.getOutputByName("Out"));
  }

  auto *save = getSaveNodeFromDest(out);

  CmpLTNode *CMPLT = llvm::dyn_cast<CmpLTNode>(save->getInput().getNode());

  ASSERT_TRUE(CMPLT);
  ASSERT_EQ(CMPLT->getResult().dims().size(), 3);
  EXPECT_EQ(CMPLT->getResult().dims()[0], 4);
  EXPECT_EQ(CMPLT->getResult().dims()[1], 4);
  EXPECT_EQ(CMPLT->getResult().dims()[2], 1);
}

/// Test loading NMS using initializer nodes op from an ONNX model.
TEST_F(OnnxImporterTest, importNMSInitializer) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string netFilename(GLOW_DATA_PATH
                          "tests/models/onnxModels/NonMaxSuppression.onnxtxt");

  PlaceholderBindings bindings;
  Placeholder *output;
  {
    Tensor boxes(ElemKind::FloatTy, {8, 4});
    boxes.zero();

    Tensor scores(ElemKind::FloatTy, {8});
    scores.zero();

    ONNXModelLoader onnxLD(netFilename, {"boxes", "scores"},
                           {&boxes.getType(), &scores.getType()}, *F);
    output = EXIT_ON_ERR(onnxLD.getOutputByName("indices"));
  }

  auto *save = getSaveNodeFromDest(output);
  NonMaxSuppressionNode *NMS =
      llvm::dyn_cast<NonMaxSuppressionNode>(save->getInput().getNode());
  ASSERT_TRUE(NMS);
  EXPECT_EQ(NMS->dims(0)[0], 3);
  EXPECT_EQ(NMS->getCenterPointBox(), 0);
}

/// Test loading NMS using Constant Tensors op from an ONNX model.
TEST_F(OnnxImporterTest, importNMSConstTensor) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string netFilename(
      GLOW_DATA_PATH "tests/models/onnxModels/NonMaxSuppressionSSD.onnxtxt");

  PlaceholderBindings bindings;
  Placeholder *output;
  {
    Tensor boxes(ElemKind::FloatTy, {8, 4});
    boxes.zero();

    Tensor scores(ElemKind::FloatTy, {8});
    scores.zero();

    ONNXModelLoader onnxLD(netFilename, {"boxes", "scores"},
                           {&boxes.getType(), &scores.getType()}, *F);
    output = EXIT_ON_ERR(onnxLD.getOutputByName("indices"));
  }

  auto *save = getSaveNodeFromDest(output);
  NonMaxSuppressionNode *NMS =
      llvm::dyn_cast<NonMaxSuppressionNode>(save->getInput().getNode());
  ASSERT_TRUE(NMS);
  EXPECT_EQ(NMS->dims(0)[0], 3);
  EXPECT_EQ(NMS->getCenterPointBox(), 1);
}

/// Test loading ONNX NMS using Constant Tensors op from an ONNX model.
TEST_F(OnnxImporterTest, importNMSONNXConstTensor) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string netFilename(
      GLOW_DATA_PATH
      "tests/models/onnxModels/NonMaxSuppressionSSD_ONNX.onnxtxt");

  PlaceholderBindings bindings;
  Placeholder *output;
  {
    Tensor boxes(ElemKind::FloatTy, {1, 8, 4});
    boxes.zero();

    Tensor scores(ElemKind::FloatTy, {1, 1, 8});
    scores.zero();

    ONNXModelLoader onnxLD(netFilename, {"boxes", "scores"},
                           {&boxes.getType(), &scores.getType()}, *F);
    output = EXIT_ON_ERR(onnxLD.getOutputByName("indices"));
  }

  auto *save = getSaveNodeFromDest(output);
  NonMaxSuppressionNode *NMS =
      llvm::dyn_cast<NonMaxSuppressionNode>(save->getInput().getNode());
  ASSERT_TRUE(NMS);
  EXPECT_EQ(NMS->dims(0)[0], 3);
  EXPECT_EQ(NMS->dims(0)[1], 3);
  EXPECT_EQ(NMS->getCenterPointBox(), 1);
}

TEST_F(OnnxImporterTest, importDimParamExplicit) {
  ExecutionEngine EE;
  auto &mod = EE.getModule();
  std::string netFilename(GLOW_DATA_PATH
                          "tests/models/onnxModels/dimParam.onnxtxt");
  auto *F = mod.createFunction("main");

  // Import ONNX model with explicit input information.
  {
    Tensor inputTensor(ElemKind::FloatTy, {1, 2});
    setOnnxDefineSymbol({"ONNXUndefinedSymbol,1"});
    ONNXModelLoader onnxLD(netFilename, {"input"}, {&inputTensor.getType()},
                           *F);
    setOnnxDefineSymbol({});
  }

  // Validate placeholder sizes.
  Placeholder *inputPH, *outputPH;
  inputPH = mod.getPlaceholderByName("input");
  outputPH = mod.getPlaceholderByName("output");
  EXPECT_TRUE(inputPH);
  EXPECT_TRUE(outputPH);
  EXPECT_EQ(inputPH->dims()[0], 1);
  EXPECT_EQ(inputPH->dims()[1], 2);
  EXPECT_EQ(outputPH->dims()[0], 1);
  EXPECT_EQ(outputPH->dims()[1], 2);
}

TEST_F(OnnxImporterTest, importDimParamImplicit) {
  ExecutionEngine EE;
  auto &mod = EE.getModule();
  std::string netFilename(GLOW_DATA_PATH
                          "tests/models/onnxModels/dimParam.onnxtxt");
  auto *F = mod.createFunction("main");

  // Import ONNX model with implicit input information.
  {
    setOnnxDefineSymbol({"ONNXUndefinedSymbol,1"});
    ONNXModelLoader onnxLD(netFilename, {}, {}, *F);
    setOnnxDefineSymbol({});
  }

  // Validate placeholder sizes.
  Placeholder *inputPH, *outputPH;
  inputPH = mod.getPlaceholderByName("input");
  outputPH = mod.getPlaceholderByName("output");
  EXPECT_TRUE(inputPH);
  EXPECT_TRUE(outputPH);
  EXPECT_EQ(inputPH->dims()[0], 1);
  EXPECT_EQ(inputPH->dims()[1], 2);
  EXPECT_EQ(outputPH->dims()[0], 1);
  EXPECT_EQ(outputPH->dims()[1], 2);
}

/// Test loading RNN from a ONNX model. The ONNX model already computes
/// the error compared to a PyTorch reference implementation.
static void importRNN(std::string fileName) {
  ExecutionEngine EE;
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  PlaceholderBindings bindings;
  {
    ONNXModelLoader onnxLD(fileName, {}, {}, *F);
    bindings.allocate(mod.getPlaceholders());
    auto Y_h_nv = EXIT_ON_ERR(onnxLD.getNodeValueByName("Y_h"));
    EXPECT_TRUE(Y_h_nv.getNode());
  }

  // Search RNN state placeholder and set to 0.
  Placeholder *Y_h_ph = nullptr;
  for (const auto &ph : mod.getPlaceholders()) {
    if (llvm::StringRef(ph->getName()).endswith("Y_h"))
      Y_h_ph = ph;
  }
  EXPECT_TRUE(Y_h_ph);
  bindings.get(Y_h_ph)->zero();

  // Compile and run.
  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  // Verify RNN error.
  Placeholder *Y_err_ph = mod.getPlaceholderByName("Y_err");
  EXPECT_TRUE(Y_err_ph);
  auto err = bindings.get(Y_err_ph)->getHandle();
  for (size_t idx = 0; idx < Y_err_ph->getType()->size(); idx++) {
    EXPECT_TRUE(std::abs(err.raw(idx)) < 1e-6);
  }
}

TEST_F(OnnxImporterTest, importRNNForward) {
  importRNN(GLOW_DATA_PATH "tests/models/onnxModels/rnnForward.onnxtxt");
}

TEST_F(OnnxImporterTest, importRNNReverse) {
  importRNN(GLOW_DATA_PATH "tests/models/onnxModels/rnnReverse.onnxtxt");
}

TEST_F(OnnxImporterTest, importRNNBidirectional) {
  importRNN(GLOW_DATA_PATH "tests/models/onnxModels/rnnBidirectional.onnxtxt");
}

TEST_F(OnnxImporterTest, importRNNForwardNoBias) {
  importRNN(GLOW_DATA_PATH "tests/models/onnxModels/rnnForwardNoBias.onnxtxt");
}

TEST_F(OnnxImporterTest, importRNNForwardNoState) {
  importRNN(GLOW_DATA_PATH "tests/models/onnxModels/rnnForwardNoState.onnxtxt");
}

/// Test loading GRU from a ONNX model. The ONNX model already computes
/// the error compared to a PyTorch reference implementation.
static void importGRU(std::string fileName) {
  ExecutionEngine EE;
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  PlaceholderBindings bindings;
  {
    ONNXModelLoader onnxLD(fileName, {}, {}, *F);
    bindings.allocate(mod.getPlaceholders());
    auto Y_h_nv = EXIT_ON_ERR(onnxLD.getNodeValueByName("Y_h"));
    EXPECT_TRUE(Y_h_nv.getNode());
  }

  // Search GRU state placeholder and set to 0.
  Placeholder *Y_h_ph = nullptr;
  for (const auto &ph : mod.getPlaceholders()) {
    if (llvm::StringRef(ph->getName()).endswith("Y_h"))
      Y_h_ph = ph;
  }
  EXPECT_TRUE(Y_h_ph);
  bindings.get(Y_h_ph)->zero();

  // Compile and run.
  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  // Verify GRU error.
  Placeholder *Y_err_ph = mod.getPlaceholderByName("Y_err");
  EXPECT_TRUE(Y_err_ph);
  auto err = bindings.get(Y_err_ph)->getHandle();
  for (size_t idx = 0; idx < Y_err_ph->getType()->size(); idx++) {
    EXPECT_TRUE(std::abs(err.raw(idx)) < 1e-6);
  }
}

TEST_F(OnnxImporterTest, importGRUForward) {
  importGRU(GLOW_DATA_PATH "tests/models/onnxModels/gruForward.onnxtxt");
}

TEST_F(OnnxImporterTest, importGRUReverse) {
  importGRU(GLOW_DATA_PATH "tests/models/onnxModels/gruReverse.onnxtxt");
}

TEST_F(OnnxImporterTest, importGRUBidirectional) {
  importGRU(GLOW_DATA_PATH "tests/models/onnxModels/gruBidirectional.onnxtxt");
}

TEST_F(OnnxImporterTest, importGRUForwardNoBias) {
  importGRU(GLOW_DATA_PATH "tests/models/onnxModels/gruForwardNoBias.onnxtxt");
}

TEST_F(OnnxImporterTest, importGRUForwardNoState) {
  importGRU(GLOW_DATA_PATH "tests/models/onnxModels/gruForwardNoState.onnxtxt");
}

TEST_F(OnnxImporterTest, importGRUForwardLinearBeforeReset) {
  importGRU(GLOW_DATA_PATH
            "tests/models/onnxModels/gruForwardLinearBeforeReset.onnxtxt");
}

/// Test loading LSTM from a ONNX model. The ONNX model already computes
/// the error compared to a PyTorch reference implementation.
static void importLSTM(std::string fileName) {
  ExecutionEngine EE;
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  PlaceholderBindings bindings;
  {
    ONNXModelLoader onnxLD(fileName, {}, {}, *F);
    bindings.allocate(mod.getPlaceholders());
    auto Y_h_nv = EXIT_ON_ERR(onnxLD.getNodeValueByName("Y_h"));
    auto Y_c_nv = EXIT_ON_ERR(onnxLD.getNodeValueByName("Y_c"));
    EXPECT_TRUE(Y_h_nv.getNode());
    EXPECT_TRUE(Y_c_nv.getNode());
  }

  // Search LSTM state placeholders and set to 0.
  Placeholder *Y_h_ph = nullptr;
  Placeholder *Y_c_ph = nullptr;
  for (const auto &ph : mod.getPlaceholders()) {
    if (llvm::StringRef(ph->getName()).endswith("Y_h"))
      Y_h_ph = ph;
    if (llvm::StringRef(ph->getName()).endswith("Y_c"))
      Y_c_ph = ph;
  }
  EXPECT_TRUE(Y_h_ph);
  EXPECT_TRUE(Y_c_ph);
  bindings.get(Y_h_ph)->zero();
  bindings.get(Y_c_ph)->zero();

  // Compile and run.
  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  // Verify LSTM error.
  Placeholder *Y_err_ph = mod.getPlaceholderByName("Y_err");
  EXPECT_TRUE(Y_err_ph);
  auto err = bindings.get(Y_err_ph)->getHandle();
  for (size_t idx = 0; idx < Y_err_ph->getType()->size(); idx++) {
    EXPECT_TRUE(std::abs(err.raw(idx)) < 1e-6);
  }
}

TEST_F(OnnxImporterTest, importLSTMForward) {
  importLSTM(GLOW_DATA_PATH "tests/models/onnxModels/lstmForward.onnxtxt");
}

TEST_F(OnnxImporterTest, importLSTMReverse) {
  importLSTM(GLOW_DATA_PATH "tests/models/onnxModels/lstmReverse.onnxtxt");
}

TEST_F(OnnxImporterTest, importLSTMBidirectional) {
  importLSTM(GLOW_DATA_PATH
             "tests/models/onnxModels/lstmBidirectional.onnxtxt");
}

TEST_F(OnnxImporterTest, importLSTMForwardNoBias) {
  importLSTM(GLOW_DATA_PATH
             "tests/models/onnxModels/lstmForwardNoBias.onnxtxt");
}

TEST_F(OnnxImporterTest, importLSTMForwardNoState) {
  importLSTM(GLOW_DATA_PATH
             "tests/models/onnxModels/lstmForwardNoState.onnxtxt");
}

TEST_F(OnnxImporterTest, importLSTMForwardWithPeephole) {
  importLSTM(GLOW_DATA_PATH
             "tests/models/onnxModels/lstmForwardWithPeephole.onnxtxt");
}

TEST_F(OnnxImporterTest, importLSTMForwardInputForget) {
  importLSTM(GLOW_DATA_PATH
             "tests/models/onnxModels/lstmForwardInputForget.onnxtxt");
}

/// Test loading Flip from a ONNX model. The ONNX model already computes
/// the error.
static void importFlip(std::string fileName) {
  ExecutionEngine EE;
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  PlaceholderBindings bindings;
  {
    ONNXModelLoader onnxLD(fileName, {}, {}, *F);
    bindings.allocate(mod.getPlaceholders());
  }

  // Compile and run.
  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  // Verify error.
  Placeholder *Y_err_ph = mod.getPlaceholderByName("Y_err");
  EXPECT_TRUE(Y_err_ph);
  auto err = bindings.get(Y_err_ph)->getHandle();
  for (size_t idx = 0; idx < Y_err_ph->getType()->size(); idx++) {
    EXPECT_EQ(err.raw(idx), 0);
  }
}

TEST_F(OnnxImporterTest, importFlipWithAxis) {
  importFlip(GLOW_DATA_PATH "tests/models/onnxModels/flipWithAxis.onnxtxt");
}

TEST_F(OnnxImporterTest, importFlipNoAxis) {
  importFlip(GLOW_DATA_PATH "tests/models/onnxModels/flipNoAxis.onnxtxt");
}

/// Test loading FRWQSparseLengthsWeightedSum from an ONNX model.
TEST_F(OnnxImporterTest, importFRWQSLWS) {
  ExecutionEngine EE;
  auto &mod = EE.getModule();
  auto *F = mod.createFunction("main");
  std::string netFilename(GLOW_DATA_PATH
                          "tests/models/onnxModels/fusedSLWS.onnxtxt");
  Placeholder *output;
  {
    Tensor weights(ElemKind::FloatTy, {8});
    Tensor indices(IndexElemKind, {8});
    Tensor lengths(ElemKind::Int32ITy, {5});
    ONNXModelLoader onnxLD(
        netFilename, {"weights", "indices", "lengths"},
        {&weights.getType(), &indices.getType(), &lengths.getType()}, *F);
    output = EXIT_ON_ERR(onnxLD.getSingleOutput());
  }

  // Verify structure: {Constant, PH, PH, PH} ->  FRWQSLWS -> Save -> PH.
  EXPECT_EQ(mod.getPlaceholders().size(), 4);
  // FRWQSLWS, Save nodes
  EXPECT_EQ(F->getNodes().size(), 2);
  auto *save = getSaveNodeFromDest(output);
  auto *FRWQSLWS =
      llvm::dyn_cast<FusedRowwiseQuantizedSparseLengthsWeightedSumNode>(
          save->getInput().getNode());
  ASSERT_TRUE(FRWQSLWS);
  auto *data = llvm::dyn_cast<Constant>(FRWQSLWS->getData());
  ASSERT_TRUE(data);
  EXPECT_EQ(data->dims().vec(), std::vector<dim_t>({3, 10}));
  EXPECT_EQ(data->getType()->getElementType(), ElemKind::UInt8FusedQTy);
  auto *weights = llvm::dyn_cast<Placeholder>(FRWQSLWS->getWeights());
  ASSERT_TRUE(weights);
  EXPECT_EQ(weights->dims().vec(), std::vector<dim_t>({8}));
  EXPECT_EQ(weights->getType()->getElementType(), ElemKind::FloatTy);
  auto *indices = llvm::dyn_cast<Placeholder>(FRWQSLWS->getIndices());
  ASSERT_TRUE(indices);
  EXPECT_EQ(indices->dims().vec(), std::vector<dim_t>({8}));
  EXPECT_EQ(indices->getType()->getElementType(), ElemKind::Int64ITy);
  auto *lengths = llvm::dyn_cast<Placeholder>(FRWQSLWS->getLengths());
  ASSERT_TRUE(lengths);
  EXPECT_EQ(lengths->dims().vec(), std::vector<dim_t>({5}));
  EXPECT_EQ(lengths->getType()->getElementType(), ElemKind::Int32ITy);
}
