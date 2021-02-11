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
  return MAKE_ERR("Can't load proto file");
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
/// with single output and then checking against expected values
/// \p expectedValues and \returns true if the test completes without error.
Error checkConstFoldLegalName(std::string NetFilename,
                              std::vector<float> expectedValues) {
  Tensor T(glow::ElemKind::FloatTy, {3, 2});
  T.getHandle<float>() = expectedValues;
  ONNX_NAMESPACE::ModelProto modelDef;
  ASSIGN_VALUE_OR_RETURN_ERR(modelDef, loadProto(NetFilename));
  setConstantFoldLoaderOpsFlag(true);

  // It is expected that loading will fold the whole graph and output
  // nodes will become constants during the loading process.
  ExecutionEngine EE;
  Module &mod = EE.getModule();
  Function *F = mod.createFunction("temp");
  ONNXModelLoader onnxLD(NetFilename, {}, {}, *F);

  setConstantFoldLoaderOpsFlag(false);

  // The folded output tensors are expected to be constants and should
  // match the expected values.
  NodeValue NV;
  ASSIGN_VALUE_OR_RETURN_ERR(
      NV, onnxLD.getNodeValueByName(modelDef.graph().output(0).name()));
  auto *constOut = llvm::dyn_cast<Constant>(NV.getNode());
  RETURN_ERR_IF_NOT(constOut, "Failed cast to Constant");
  EXPECT_TRUE(T.isEqual(constOut->getPayload()));
  return Error::success();
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

static void importReduceL2Test(const std::string &netFilename,
                               llvm::ArrayRef<float> inputValues,
                               llvm::ArrayRef<dim_t> inputShape,
                               llvm::ArrayRef<dim_t> outputShape,
                               llvm::ArrayRef<float> expectedValues) {
  float delta = 1e-08;
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  PlaceholderBindings bindings;
  Placeholder *graphOutputVar;

  // Load the .onnxtxt model.
  Type inputType(ElemKind::FloatTy, inputShape);
  ONNXModelLoader onnxLD(netFilename, {"input"}, {&inputType}, *F);
  graphOutputVar = EXIT_ON_ERR(onnxLD.getSingleOutput());
  auto PH = mod.getPlaceholderByNameSlow("input");
  auto *inTensor = bindings.allocate(PH);
  inTensor->getHandle() = inputValues;
  EE.compile(CompilationMode::Infer);
  bindings.allocate(mod.getPlaceholders());
  EE.run(bindings);
  auto result = bindings.get(graphOutputVar)->getHandle();
  ASSERT_TRUE(result.dims() == (llvm::ArrayRef<dim_t>)outputShape);
  for (size_t i = 0; i < result.getType().size(); i++) {
    EXPECT_NEAR(result.raw(i), expectedValues[i], delta);
  }
}

/// Test the utility function that gets the inputs name and glow types
/// from updated graph proto

TEST_F(OnnxImporterTest, getInputNamesAndTypes) {
  // Set onnx-define-symbol if present in model
  std::string inputSymbol = "batch_size,5";
  setOnnxDefineSymbol({inputSymbol});

  std::string netFilename(
      GLOW_DATA_PATH
      "tests/models/onnxModels/getInputsOnnxDefineSample.onnxtxt");

  bool isError = false;

  std::vector<std::string> names;
  std::vector<Type> types;

  std::vector<std::string> expectedNames = {"input"};
  std::vector<std::vector<dim_t>> expectedDims = {{5, 3, 224, 224}};

  isError = ERR_TO_BOOL(
      ONNXModelLoader::getInputsNamesAndTypes(names, types, netFilename));

  EXPECT_FALSE(isError);

  for (size_t i = 0; i < expectedNames.size(); i++) {
    EXPECT_TRUE(expectedNames[i] == names[i]);
    std::vector<dim_t> dims = types[i].dims();
    for (size_t j = 0; j < expectedDims[i].size(); j++) {
      EXPECT_EQ(expectedDims[i][j], dims[j]);
    }
  }
}

/// Test the utility function which wraps a negative axis.
TEST_F(OnnxImporterTest, getPositiveAxis) {
  int axisPos;
  ASSIGN_VALUE_OR_FAIL_TEST(axisPos, getPositiveAxis<int>(-3, 3));
  EXPECT_EQ(axisPos, 0);
  ASSIGN_VALUE_OR_FAIL_TEST(axisPos, getPositiveAxis<int>(-2, 3));
  EXPECT_EQ(axisPos, 1);
  ASSIGN_VALUE_OR_FAIL_TEST(axisPos, getPositiveAxis<int>(-1, 3));
  EXPECT_EQ(axisPos, 2);
  ASSIGN_VALUE_OR_FAIL_TEST(axisPos, getPositiveAxis<int>(0, 3));
  EXPECT_EQ(axisPos, 0);
  ASSIGN_VALUE_OR_FAIL_TEST(axisPos, getPositiveAxis<int>(1, 3));
  EXPECT_EQ(axisPos, 1);
  ASSIGN_VALUE_OR_FAIL_TEST(axisPos, getPositiveAxis<int>(2, 3));
  EXPECT_EQ(axisPos, 2);
}

/// Test loading reduceL2 op from an ONNX model
/// with axes = [].
TEST_F(OnnxImporterTest, reduceL2NoAxis) {
  std::vector<float> inputValues = {1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2};
  std::vector<dim_t> inputShape = {2, 3, 2};
  std::vector<dim_t> outputShape = {1, 1, 1};
  std::vector<float> expectedValues = {5.477226};
  std::string netFilename(GLOW_DATA_PATH
                          "tests/models/onnxModels/ReduceL2NoAxis.onnxtxt");
  importReduceL2Test(netFilename, inputValues, inputShape, outputShape,
                     expectedValues);
}

/// Test loading reduceL2 op from an ONNX model
/// with negative axis values.
TEST_F(OnnxImporterTest, reduceL2NegAxis) {
  std::vector<float> inputValues = {1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2};
  std::vector<dim_t> inputShape = {2, 3, 2};
  std::vector<dim_t> outputShape = {2, 1, 1};
  std::vector<float> expectedValues = {3.8729835, 3.8729835};
  std::string netFilename(GLOW_DATA_PATH
                          "tests/models/onnxModels/ReduceL2NegAxis.onnxtxt");
  importReduceL2Test(netFilename, inputValues, inputShape, outputShape,
                     expectedValues);
}

/// Test loading reduceL2 op from an ONNX model
/// with keepdims = True.
TEST_F(OnnxImporterTest, reduceL2KeepDims) {
  std::vector<float> inputValues = {1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2};
  std::vector<dim_t> inputShape = {2, 3, 2};
  std::vector<dim_t> outputShape = {2, 1, 1};
  std::vector<float> expectedValues = {3.8729835, 3.8729835};
  std::string netFilename(GLOW_DATA_PATH
                          "tests/models/onnxModels/ReduceL2KeepDims.onnxtxt");
  importReduceL2Test(netFilename, inputValues, inputShape, outputShape,
                     expectedValues);
}

/// Test loading reduceL2 op from an ONNX model
/// with keepdims = False.
TEST_F(OnnxImporterTest, reduceL2NoKeepDims) {
  std::vector<float> inputValues = {1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2};
  std::vector<dim_t> inputShape = {2, 3, 2};
  std::vector<dim_t> outputShape = {2};
  std::vector<float> expectedValues = {3.8729835, 3.8729835};
  std::string netFilename(GLOW_DATA_PATH
                          "tests/models/onnxModels/ReduceL2NoKeepDims.onnxtxt");
  importReduceL2Test(netFilename, inputValues, inputShape, outputShape,
                     expectedValues);
}

/// Test loading constant+relu ops with numeric input names from an ONNX model.
TEST_F(OnnxImporterTest, reluConstFoldLegalName) {
  std::string NetFilename(GLOW_DATA_PATH
                          "tests/models/onnxModels/constRelu.onnxtxt");
  FAIL_TEST_IF_ERR(
      checkConstFoldLegalName(NetFilename, {1.0, 0.0, 0.0, 1.0, 1.0, 1.0}));
}

template <class OpType>
static void
importArithMultiBroadcastTest(std::string fileName,
                              llvm::ArrayRef<dim_t> inputShape, bool multi,
                              bool leftBroadcast, bool rightBroadcast,
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
  // Check the graph structure
  auto *saveNode = getSaveNodeFromDest(graphOutputVar);
  auto *node = saveNode->getInput().getNode();
  auto *opNode = llvm::dyn_cast<OpType>(node);
  EXPECT_NE(nullptr, opNode);

  BroadcastNode *leftBN =
      llvm::dyn_cast<BroadcastNode>(opNode->getLHS().getNode());
  BroadcastNode *rightBN =
      llvm::dyn_cast<BroadcastNode>(opNode->getRHS().getNode());
  EXPECT_NE(leftBroadcast, leftBN == nullptr);
  EXPECT_NE(rightBroadcast, rightBN == nullptr);

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

static void importExpandTest(const std::string &netFilename,
                             llvm::ArrayRef<float> inputValues,
                             llvm::ArrayRef<dim_t> inputShape,
                             llvm::ArrayRef<dim_t> outputShape,
                             llvm::ArrayRef<float> expectedValues) {
  float delta = 1e-08;
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  PlaceholderBindings bindings;
  Placeholder *graphOutputVar;
  // Load the .onnxtxt model.
  Type inputType(ElemKind::FloatTy, inputShape);
  ONNXModelLoader onnxLD(netFilename, {"x"}, {&inputType}, *F);
  graphOutputVar = EXIT_ON_ERR(onnxLD.getSingleOutput());
  auto *PH = mod.getPlaceholderByNameSlow("x");
  auto *inTensor = bindings.allocate(PH);
  inTensor->getHandle() = inputValues;
  EE.compile(CompilationMode::Infer);
  bindings.allocate(mod.getPlaceholders());
  EE.run(bindings);
  auto result = bindings.get(graphOutputVar)->getHandle();
  ASSERT_TRUE(result.dims() == (llvm::ArrayRef<dim_t>)outputShape);
  for (size_t i = 0; i < result.getType().size(); i++) {
    EXPECT_NEAR(result.raw(i), expectedValues[i], delta);
  }
}

/// Import maxPool1D
static void importMaxPool1DTest(std::string &netFilename,
                                llvm::ArrayRef<float> inputValues,
                                llvm::ArrayRef<dim_t> inputShape,
                                llvm::ArrayRef<dim_t> outputShape,
                                llvm::ArrayRef<float> expectedValues) {
  float delta = 1e-08;
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  PlaceholderBindings bindings;
  Placeholder *graphOutputVar;

  Type input_type(ElemKind::FloatTy, inputShape);
  ONNXModelLoader onnxLD(netFilename, {"x"}, {&input_type}, *F);

  graphOutputVar = EXIT_ON_ERR(onnxLD.getSingleOutput());

  auto PH = mod.getPlaceholderByNameSlow("x");
  auto *inTensor = bindings.allocate(PH);
  inTensor->getHandle() = inputValues;

  EE.compile(CompilationMode::Infer);
  bindings.allocate(mod.getPlaceholders());
  EE.run(bindings);

  auto result = bindings.get(graphOutputVar)->getHandle();
  ASSERT_TRUE(result.dims() == (llvm::ArrayRef<dim_t>)outputShape);
  for (size_t i = 0; i < result.getType().size(); i++) {
    EXPECT_NEAR(result.raw(i), expectedValues[i], delta);
  }
}

/// Test loading expand op from an ONNX model
/// with different output shape.
TEST_F(OnnxImporterTest, expandDiffShape) {
  std::vector<float> inputValues = {1, 2, 3};
  std::vector<dim_t> inputShape = {3, 1};
  std::vector<dim_t> outputShape = {2, 3, 6};
  std::vector<float> expectedValues = {
      1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3,
      1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3,
  };
  std::string netFilename(
      GLOW_DATA_PATH "tests/models/onnxModels/expandnodeDiffShape.onnxtxt");
  importExpandTest(netFilename, inputValues, inputShape, outputShape,
                   expectedValues);
}

/// Test loading expand op from an ONNX model
/// with same output shape.
TEST_F(OnnxImporterTest, expandSameShape) {
  std::vector<float> inputValues = {1, 2, 3};
  std::vector<dim_t> inputShape = {3, 1};
  std::vector<dim_t> outputShape = {3, 4};
  std::vector<float> expectedValues = {
      1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
  };
  std::string netFilename(
      GLOW_DATA_PATH "tests/models/onnxModels/expandnodeSameShape.onnxtxt");
  importExpandTest(netFilename, inputValues, inputShape, outputShape,
                   expectedValues);
}

/// Test loading maxPool1D op from an ONNX model
/// with different output shape.
TEST_F(OnnxImporterTest, maxPool1D) {
  std::vector<float> inputValues = {
      1.4206449,  0.54408556, 1.3318906,  0.771925,   0.9450552,
      0.08600737, 0.30009857, 1.4206449,  0.54408556, 1.3318906,
      0.771925,   0.9450552,  0.08600737, 0.30009857};

  std::vector<dim_t> inputShape = {1, 2, 7};
  std::vector<dim_t> outputShape = {1, 2, 2};
  std::vector<float> expectedValues = {
      1.4206449,
      0.9450552,
      1.4206449,
      0.9450552,
  };
  std::string netFilename(GLOW_DATA_PATH
                          "tests/models/onnxModels/maxPool1D.onnxtxt");
  importMaxPool1DTest(netFilename, inputValues, inputShape, outputShape,
                      expectedValues);
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
  LeakyReluNode *LR = llvm::dyn_cast<LeakyReluNode>(save->getInput().getNode());
  ASSERT_TRUE(LR);
  EXPECT_FLOAT_EQ(LR->getAlpha(), 0.100000001);
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
  LeakyReluNode *LR = llvm::dyn_cast<LeakyReluNode>(save->getInput().getNode());
  ASSERT_TRUE(LR);
  EXPECT_FLOAT_EQ(LR->getAlpha(), 0.01);
}

TEST_F(OnnxImporterTest, importAddMultiBroadcastOp7) {
  importArithMultiBroadcastTest<AddNode>(
      "addMultiBroadcastOp7.onnxtxt", {1, 3, 1, 2}, /* multi */ true,
      /* leftBroadcast */ true, /* rightBroadcast */ true,
      [](float a, float b) { return a + b; });
}

TEST_F(OnnxImporterTest, importAddUniBroadcastOp6NoAxis) {
  importArithMultiBroadcastTest<AddNode>(
      "addUniBroadcastOp6NoAxis.onnxtxt", {1, 3, 4, 2}, /* multi */ false,
      /* leftBroadcast */ false, /* rightBroadcast */ true,
      [](float a, float b) { return a + b; });
}

TEST_F(OnnxImporterTest, importAddUniBroadcastOp6Axis) {
  importArithMultiBroadcastTest<AddNode>(
      "addUniBroadcastOp6Axis.onnxtxt", {1, 3, 4, 2}, /* multi */ false,
      /* leftBroadcast */ false, /* rightBroadcast */ true,
      [](float a, float b) { return a + b; });
}

TEST_F(OnnxImporterTest, importSubMultiBroadcastOp7) {
  importArithMultiBroadcastTest<SubNode>(
      "subMultiBroadcastOp7.onnxtxt", {1, 3, 1, 2}, /* multi */ true,
      /* leftBroadcast */ true, /* rightBroadcast */ true,
      [](float a, float b) { return a - b; });
}

TEST_F(OnnxImporterTest, importSubUniBroadcastOp6NoAxis) {
  importArithMultiBroadcastTest<SubNode>(
      "subUniBroadcastOp6NoAxis.onnxtxt", {1, 3, 4, 2}, /* multi */ false,
      /* leftBroadcast */ false, /* rightBroadcast */ true,
      [](float a, float b) { return a - b; });
}

TEST_F(OnnxImporterTest, importSubUniBroadcastOp6Axis) {
  importArithMultiBroadcastTest<SubNode>(
      "subUniBroadcastOp6Axis.onnxtxt", {1, 3, 4, 2}, /* multi */ false,
      /* leftBroadcast */ false, /* rightBroadcast */ true,
      [](float a, float b) { return a - b; });
}

TEST_F(OnnxImporterTest, importMulMultiBroadcastOp7) {
  importArithMultiBroadcastTest<MulNode>(
      "mulMultiBroadcastOp7.onnxtxt", {1, 3, 1, 2}, /* multi */ true,
      /* leftBroadcast */ true, /* rightBroadcast */ true,
      [](float a, float b) { return a * b; });
}

TEST_F(OnnxImporterTest, importMulUniBroadcastOp6NoAxis) {
  importArithMultiBroadcastTest<MulNode>(
      "mulUniBroadcastOp6NoAxis.onnxtxt", {1, 3, 4, 2}, /* multi */ false,
      /* leftBroadcast */ false, /* rightBroadcast */ true,
      [](float a, float b) { return a * b; });
}

TEST_F(OnnxImporterTest, importMulUniBroadcastOp6Axis) {
  importArithMultiBroadcastTest<MulNode>(
      "mulUniBroadcastOp6Axis.onnxtxt", {1, 3, 4, 2}, /* multi */ false,
      /* leftBroadcast */ false, /* rightBroadcast */ true,
      [](float a, float b) { return a * b; });
}

TEST_F(OnnxImporterTest, importDivMultiBroadcastOp7) {
  importArithMultiBroadcastTest<DivNode>(
      "divMultiBroadcastOp7.onnxtxt", {1, 3, 1, 2}, /* multi */ true,
      /* leftBroadcast */ true, /* rightBroadcast */ true,
      [](float a, float b) { return a / b; });
}

TEST_F(OnnxImporterTest, importDivUniBroadcastOp6NoAxis) {
  importArithMultiBroadcastTest<DivNode>(
      "divUniBroadcastOp6NoAxis.onnxtxt", {1, 3, 4, 2}, /* multi */ false,
      /* leftBroadcast */ false, /* rightBroadcast */ true,
      [](float a, float b) { return a / b; });
}

TEST_F(OnnxImporterTest, importDivUniBroadcastOp6Axis) {
  importArithMultiBroadcastTest<DivNode>(
      "divUniBroadcastOp6Axis.onnxtxt", {1, 3, 4, 2}, /* multi */ false,
      /* leftBroadcast */ false, /* rightBroadcast */ true,
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
  auto PH = mod.getPlaceholderByNameSlow(input_name);
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

TEST(onnx, importNeg) {
  testEltwiseUnaryOpFloat("neg.onnxtxt", {1, 2, 4, 3}, "data", 0.000,
                          [](float a) { return -a; });
}

TEST(onnx, importCeil) {
  testEltwiseUnaryOpFloat("ceil.onnxtxt", {1, 2, 4, 3}, "data", 0.000,
                          [](float a) { return std::ceil(a); });
}

TEST(onnx, importFloor) {
  testEltwiseUnaryOpFloat("floor.onnxtxt", {1, 2, 4, 3}, "data", 0.000,
                          [](float a) { return std::floor(a); });
}

TEST_F(OnnxImporterTest, importSin) {
  testEltwiseUnaryOpFloat("Sin.onnxtxt", {2, 3, 1}, "X", 0.002,
                          [](float a) { return std::sin(a); });
}

TEST_F(OnnxImporterTest, importCos) {
  testEltwiseUnaryOpFloat("Cos.onnxtxt", {2, 3, 1}, "X", 0.002,
                          [](float a) { return std::cos(a); });
}

TEST_F(OnnxImporterTest, importErf) {
  testEltwiseUnaryOpFloat("Erf.onnxtxt", {1, 3, 4, 5}, "input", 0.002,
                          [](float a) { return std::erf(a); });
}

TEST(onnx, importAbs) {
  testEltwiseUnaryOpFloat("abs.onnxtxt", {1, 2, 3, 2}, "input", 0.002,
                          [](float a) { return std::abs(a); });
}

// Tests log node for random positive values.
static void testImportLog(std::string fileName,
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
  auto PH = mod.getPlaceholderByNameSlow(input_name);
  auto *inTensor = bindings.allocate(PH);

  inTensor->getHandle().randomize(0, 500.0, mod.getPRNG());
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

/// Test loading of Elemenntwise Trigonometric Ops
/// Extendable for other ops in future
static void
testEltwiseTrigonometricOpFloat(std::string fileName,
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
  auto PH = mod.getPlaceholderByNameSlow(input_name);
  auto *inTensor = bindings.allocate(PH);

  // Range of Asin/Acos is -1 to 1
  inTensor->getHandle().randomize(-1.0, 1.0, mod.getPRNG());
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

TEST_F(OnnxImporterTest, importAsin) {
  testEltwiseTrigonometricOpFloat("Asin.onnxtxt", {1, 3, 4, 5}, "input", 0.002,
                                  [](float a) { return std::asin(a); });
}

TEST_F(OnnxImporterTest, importAcos) {
  testEltwiseTrigonometricOpFloat("Acos.onnxtxt", {1, 3, 4, 5}, "input", 0.002,
                                  [](float a) { return std::acos(a); });
}

TEST_F(OnnxImporterTest, importAtan) {
  testEltwiseTrigonometricOpFloat("Atan.onnxtxt", {1, 3, 4, 5}, "input", 0.002,
                                  [](float a) { return std::atan(a); });
}

TEST_F(OnnxImporterTest, importLog) {
  testImportLog("log.onnxtxt", {1, 2, 3, 2}, "data", 0.002,
                [](float a) { return std::log(a); });
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
  auto dataH =
      bindings.get(bindings.getPlaceholderByNameSlow("data"))->getHandle();
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
                           llvm::ArrayRef<dim_t> expectedDims,
                           llvm::ArrayRef<float> expectedValues) {

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
/// strides is {1, 1}, pads is {1, 1, 1, 1}, group is 1, dilation is {1, 2}.
TEST_F(OnnxImporterTest, importConvNonSquareDilation) {
  std::string filename("simpleConvNonSquareDilation.onnxtxt");
  std::vector<dim_t> expectedDims = {1, 1, 4, 3};
  std::vector<float> expectedValues = {3, 4, 3, 7, 12, 7, 13, 24, 13, 9, 16, 9};
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

/// Import conv1D
static void importConv1DTest(std::string &netFilename,
                             llvm::ArrayRef<float> inputXValues,
                             llvm::ArrayRef<dim_t> inputXShape,
                             llvm::ArrayRef<float> inputWValues,
                             llvm::ArrayRef<dim_t> inputWShape,
                             llvm::ArrayRef<dim_t> outputShape,
                             llvm::ArrayRef<float> expectedValues) {
  float delta = 1e-07;
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  PlaceholderBindings bindings;
  Placeholder *graphOutputVar;

  Type input_type_x(ElemKind::FloatTy, inputXShape);
  Type input_type_w(ElemKind::FloatTy, inputWShape);
  ONNXModelLoader onnxLD(netFilename, {"x", "w"},
                         {&input_type_x, &input_type_w}, *F);

  graphOutputVar = EXIT_ON_ERR(onnxLD.getSingleOutput());

  auto PHX = mod.getPlaceholderByNameSlow("x");
  auto *inTensorX = bindings.allocate(PHX);
  inTensorX->getHandle() = inputXValues;

  auto PHW = mod.getPlaceholderByNameSlow("w");
  auto *inTensorW = bindings.allocate(PHW);
  inTensorW->getHandle() = inputWValues;

  EE.compile(CompilationMode::Infer);
  bindings.allocate(mod.getPlaceholders());
  EE.run(bindings);

  auto result = bindings.get(graphOutputVar)->getHandle();
  ASSERT_TRUE(result.dims() == (llvm::ArrayRef<dim_t>)outputShape);
  for (size_t i = 0; i < result.getType().size(); i++) {
    EXPECT_NEAR(result.raw(i), expectedValues[i], delta);
  }
}

/// Test Conv1D
TEST_F(OnnxImporterTest, conv1D) {
  std::vector<float> inputXValues = {
      1.4206449,  -0.54408556, -1.3318906,  0.771925,   0.9450552,  0.08600737,
      0.30009857, -0.36060193, -0.33999684, -0.9809143, -1.0172559, -0.4921318,
      -1.0513021, 1.8671927,   -0.842103,   -0.8903683};
  std::vector<float> inputWValues = {0.16575365, -0.42219377, 0.55620337,
                                     -0.5700942, -1.1148645,  -0.33808824};
  std::vector<dim_t> inputXShape = {1, 2, 8};
  std::vector<dim_t> inputWShape = {3, 2, 1};
  std::vector<dim_t> outputShape = {1, 3, 8};
  std::vector<float> expectedValues = {
      0.3790216,  0.32395172, 0.20871338,  0.33572435, 0.6004995,   -0.7740611,
      0.40527308, 0.31613684, 0.9839977,   0.25659135, -0.16087033, 0.7099088,
      1.1249841,  -1.0166382, 0.6469939,   0.30702582, -1.4688776,  0.9382173,
      1.8287997,  -0.6942077, -0.69817555, -0.7271625, -0.04986412, 0.7030453};
  std::string netFilename(GLOW_DATA_PATH
                          "tests/models/onnxModels/conv1D.onnxtxt");
  importConv1DTest(netFilename, inputXValues, inputXShape, inputWValues,
                   inputWShape, outputShape, expectedValues);
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

/// Helper method to run the ConvTranspose operator test cases.
/// \p filename contains the model .onnxtxt.
/// \p expectedDims: output Tensor dimensions.
/// \p expectedValues : output Tensor values expected.
/// The input is N*C*H*W (1*1*2*2), the kernels is {3, 3},
/// strides is {1, 1}, group is 1. Pads can vary.
static void convTransposeTestHelper(std::string &filename,
                                    llvm::ArrayRef<dim_t> expectedDims,
                                    llvm::ArrayRef<float> expectedValues) {

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
    Tensor data(ElemKind::FloatTy, {1, 1, 2, 2});
    data.getHandle() = {2., 3., 4., 5.};

    ONNXModelLoader onnxLD(NetFilename, {"data"}, {&data.getType()}, *F);
    graphOutputVar = EXIT_ON_ERR(onnxLD.getSingleOutput());
    bindings.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(bindings, &mod, {"data"}, {&data});
  }

  // ONNX importer loads a ConvTranspose node and converts it to 4 ops:
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
  auto *convTrNode = llvm::dyn_cast<TransposeNode>(node)->getInput().getNode();

  EXPECT_TRUE(convTrNode->getKind() == Kinded::Kind::ConvTransposeNodeKind);
  auto *tInNode =
      llvm::dyn_cast<ConvTransposeNode>(convTrNode)->getInput().getNode();
  auto *tFilterNode =
      llvm::dyn_cast<ConvTransposeNode>(convTrNode)->getFilter().getNode();
  EXPECT_TRUE(tInNode->getKind() == Kinded::Kind::TransposeNodeKind);
  EXPECT_TRUE(tFilterNode->getKind() == Kinded::Kind::TransposeNodeKind);

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  EXPECT_EQ(F->getNodes().size(), 4);
  EXPECT_EQ(mod.getPlaceholders().size(), 2);
  EXPECT_EQ(mod.getConstants().size(), 2);

  auto result = bindings.get(graphOutputVar)->getHandle();
  EXPECT_TRUE(result.dims() == expectedDims);
  for (dim_t i = 0, e = expectedValues.size(); i < e; i++) {
    EXPECT_FLOAT_EQ(result.raw(i), expectedValues[i]);
  }
}

/// Test loading ConvTranspose op from a ONNX model, no pads.
TEST_F(OnnxImporterTest, importConvTranspose) {
  std::string filename("simpleConvTranspose.onnxtxt");
  std::vector<dim_t> expectedDims = {1, 1, 4, 4};
  std::vector<float> expectedValues = {5,  13, 18,  13, 19, 50, 64, 42,
                                       37, 92, 106, 66, 33, 77, 86, 51};
  convTransposeTestHelper(filename, expectedDims, expectedValues);
}

/// Test loading ConvTranspose op from a ONNX model, symmetric pads.
TEST_F(OnnxImporterTest, importConvTransposePads) {
  std::string filename("simpleConvTransposePads.onnxtxt");
  std::vector<dim_t> expectedDims = {1, 1, 3, 3};
  std::vector<float> expectedValues = {14., 19., 14.,  51., 65.,
                                       43., 93., 107., 67.};
  convTransposeTestHelper(filename, expectedDims, expectedValues);
}

/// Test loading ConvTranspose op from a ONNX model, auto_pad=VALID
TEST_F(OnnxImporterTest, importConvTransposeAutoPadValid) {
  std::string filename("simpleConvTransposeAutoPadValid.onnxtxt");
  std::vector<dim_t> expectedDims = {1, 1, 4, 4};
  std::vector<float> expectedValues = {4,  12, 17,  12, 18, 49, 63, 41,
                                       36, 91, 105, 65, 32, 76, 85, 50};
  convTransposeTestHelper(filename, expectedDims, expectedValues);
}

/// Test loading ConvTranspose op from a ONNX model, auto_pad=SAME_UPPER
TEST_F(OnnxImporterTest, importConvTransposeAutoPadSameUpper) {
  std::string filename("simpleConvTransposeAutoPadSameUpper.onnxtxt");
  std::vector<dim_t> expectedDims = {1, 1, 2, 2};
  std::vector<float> expectedValues = {49., 63., 91., 105.};
  convTransposeTestHelper(filename, expectedDims, expectedValues);
}

/// Test loading ConvTranspose op from a ONNX model, auto_pad=SAME_LOWER
TEST_F(OnnxImporterTest, importConvTransposeAutoPadSameLower) {
  std::string filename("simpleConvTransposeAutoPadSameLower.onnxtxt");
  std::vector<dim_t> expectedDims = {1, 1, 2, 2};
  std::vector<float> expectedValues = {49., 63., 91., 105.};
  convTransposeTestHelper(filename, expectedDims, expectedValues);
}

/// Test loading ConvTranspose op, explicit output_shape, auto_pad=SAME_UPPER.
TEST_F(OnnxImporterTest, importConvTransposeOutputShapeSameUpper) {
  std::string filename("simpleConvTransposeOutShapeSameUpper.onnxtxt");
  std::vector<dim_t> expectedDims = {1, 1, 4, 4};
  std::vector<float> expectedValues = {4,  12, 17,  12, 18, 49, 63, 41,
                                       36, 91, 105, 65, 32, 76, 85, 50};
  convTransposeTestHelper(filename, expectedDims, expectedValues);
}

/// Test loading deconv op, explicit output_shape, auto_pad=SAME_LOWER.
TEST_F(OnnxImporterTest, importConvTransposeOutputShapeSameLower) {
  std::string filename("simpleConvTransposeOutShapeSameLower.onnxtxt");
  std::vector<dim_t> expectedDims = {1, 1, 4, 4};
  std::vector<float> expectedValues = {4,  12, 17,  12, 18, 49, 63, 41,
                                       36, 91, 105, 65, 32, 76, 85, 50};
  convTransposeTestHelper(filename, expectedDims, expectedValues);
}

/// Test loading ConvTranspose op, explicit output_shape, auto_pad not set.
TEST_F(OnnxImporterTest, importConvTransposeOutputShape) {
  std::string filename("simpleConvTransposeOutShape.onnxtxt");
  std::vector<dim_t> expectedDims = {1, 1, 4, 4};
  std::vector<float> expectedValues = {4,  12, 17,  12, 18, 49, 63, 41,
                                       36, 91, 105, 65, 32, 76, 85, 50};
  convTransposeTestHelper(filename, expectedDims, expectedValues);
}

/// Helper method to run the Range operator test cases.
/// \p filename contains the model .onnxtxt.
/// \p expectedDims: output Tensor dimensions.
/// \p expectedValues : output Tensor values expected.
template <typename T>
static void rangeTestHelper(std::string &filename,
                            llvm::ArrayRef<dim_t> expectedDims,
                            llvm::ArrayRef<T> expectedValues) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetFilename =
      std::string(GLOW_DATA_PATH "tests/models/onnxModels/") + filename;

  PlaceholderBindings bindings;
  Placeholder *output;
  {
    ONNXModelLoader onnxLD(NetFilename, {}, {}, *F);
    output = EXIT_ON_ERR(onnxLD.getSingleOutput());
    bindings.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(bindings, &mod, {}, {});
  }
  auto *res = bindings.get(output);
  EE.compile(CompilationMode::Infer);
  EE.run(bindings);
  auto result = res->getHandle<T>();
  EXPECT_TRUE(result.dims() == expectedDims);
  for (dim_t i = 0, e = expectedValues.size(); i < e; i++) {
    EXPECT_FLOAT_EQ(result.raw(i), expectedValues[i]);
  }
}

/// Test loading Range with int32 datatype.
TEST(onnx, importRangeInt32) {
  std::string filename("RangeInt32.onnxtxt");
  std::vector<dim_t> expectedDims = {2};
  std::vector<int32_t> expectedValues = {10, 7};
  rangeTestHelper<int32_t>(filename, expectedDims, expectedValues);
}

/// Test loading Range with float datatype.
TEST(onnx, importRangeFloat) {
  std::string filename("RangeFloat.onnxtxt");
  std::vector<dim_t> expectedDims = {5};
  std::vector<float> expectedValues = {0.0, 1.0, 2.0, 3.0, 4.0};
  rangeTestHelper<float>(filename, expectedDims, expectedValues);
}

/// Test loading ConvTranspose, implicit kernel, multi-channel input/output,
/// asymmetric kernel and pads.
TEST(onnx, importDeconvAsymmetric) {

  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetFilename = std::string(
      GLOW_DATA_PATH "tests/models/onnxModels/convTransposeAsymmetric.onnxtxt");

  PlaceholderBindings bindings;
  Placeholder *output;
  {
    Tensor input(ElemKind::FloatTy, {1, 3, 4, 4});
    for (dim_t i = 0; i < 3 * 4 * 4; i++) {
      input.getHandle().raw(i) = i;
    }
    Tensor filter(ElemKind::FloatTy, {3, 2, 3, 2});
    for (dim_t i = 0; i < 3 * 2 * 3 * 2; i++) {
      filter.getHandle().raw(i) = i * 2;
    }
    ONNXModelLoader onnxLD(NetFilename, {"X", "W"},
                           {&input.getType(), &filter.getType()}, *F);
    output = EXIT_ON_ERR(onnxLD.getSingleOutput());
    bindings.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(bindings, &mod, {"X", "W"},
                                  {&input, &filter});
  }
  auto *res = bindings.get(output);
  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  auto result = res->getHandle();

  EXPECT_TRUE(result.dims() == llvm::ArrayRef<dim_t>({1, 2, 5, 3}));

  std::vector<float> expected = {
      2095.1,  2065.1,  2173.1,  4705.1, 4633.1, 4873.1,  7879.1,  7753.1,
      8149.1,  8959.1,  8761.1,  9229.1, 6697.1, 6553.1,  6889.1,  2708.2,
      2714.2,  2822.2,  6074.2,  6074.2, 6314.2, 10148.2, 10130.2, 10526.2,
      11660.2, 11570.2, 12038.2, 8642.2, 8570.2, 8906.2};

  for (dim_t i = 0, e = expected.size(); i < e; i++) {
    EXPECT_FLOAT_EQ(result.raw(i), expected[i]);
  }
}

// ConvTranspose test with Group>1
TEST(onnx, importDeconvGrouped) {

  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetFilename = std::string(
      GLOW_DATA_PATH "tests/models/onnxModels/convTransposeGroup.onnxtxt");

  PlaceholderBindings bindings;
  Placeholder *output;
  {
    Tensor input(ElemKind::FloatTy, {1, 2, 3, 3});
    for (dim_t i = 0; i < 2 * 3 * 3; i++) {
      input.getHandle().raw(i) = i;
    }
    Tensor filter(ElemKind::FloatTy, {2, 1, 2, 2});
    for (dim_t i = 0; i < 2 * 2 * 2; i++) {
      filter.getHandle().raw(i) = i * 2;
    }
    ONNXModelLoader onnxLD(NetFilename, {"X", "W"},
                           {&input.getType(), &filter.getType()}, *F);
    output = EXIT_ON_ERR(onnxLD.getSingleOutput());
    bindings.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(bindings, &mod, {"X", "W"},
                                  {&input, &filter});
  }
  auto *res = bindings.get(output);
  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  auto result = res->getHandle();

  EXPECT_TRUE(result.dims() == llvm::ArrayRef<dim_t>({1, 2, 6, 6}));

  std::vector<float> expected = {
      0,   0,   0,   2,   0,   4,   0,   0,   4,   6,   8,   12,  0,   6,   0,
      8,   0,   10,  12,  18,  16,  24,  20,  30,  0,   12,  0,   14,  0,   16,
      24,  36,  28,  42,  32,  48,  72,  90,  80,  100, 88,  110, 108, 126, 120,
      140, 132, 154, 96,  120, 104, 130, 112, 140, 144, 168, 156, 182, 168, 196,
      120, 150, 128, 160, 136, 170, 180, 210, 192, 224, 204, 238};

  for (dim_t i = 0, e = expected.size(); i < e; i++) {
    EXPECT_FLOAT_EQ(result.raw(i), expected[i]);
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
                                  llvm::ArrayRef<dim_t> expectedDims,
                                  llvm::ArrayRef<float> expectedValues) {

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

/// Test loading AveragePool op from a ONNX model.
/// The input is N*C*H*W (1*1*3*3), the kernels is {3, 3},
/// strides is {2, 2}, pads is {1, 1, 1, 1},
/// countIncludePads is false.
TEST_F(OnnxImporterTest, importAveragePool2DCountExcludePads) {
  std::string filename("averagePool2DCountExcludePads.onnxtxt");
  std::vector<dim_t> expectedDims = {1, 1, 2, 2};
  std::vector<float> expectedValues = {2, 3, 5, 6};
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

static void testReductionOps(std::string modelName,
                             const std::vector<dim_t> &expectedDims,
                             const std::vector<float> &expectedValues) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  // Input.
  Tensor x(ElemKind::FloatTy, {2, 2, 2, 2});
  x.getHandle() = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

  // Load model.
  std::string netFilename =
      std::string(GLOW_DATA_PATH "tests/models/onnxModels/") + modelName;
  ONNXModelLoader onnxLD(netFilename, {"x"}, {&x.getType()}, *F);
  Placeholder *output = EXIT_ON_ERR(onnxLD.getSingleOutput());

  // Allocate placeholders.
  PlaceholderBindings bindings;
  bindings.allocate(mod.getPlaceholders());
  updateInputPlaceholdersByName(bindings, &mod, {"x"}, {&x});

  auto *res = bindings.get(output);
  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  // Compare results.
  auto result = res->getHandle();
  EXPECT_TRUE(result.dims().vec() == expectedDims);
  for (dim_t i = 0; i < result.size(); i++) {
    EXPECT_FLOAT_EQ(result.raw(i), expectedValues[i]);
  }

  // Constant Folding Test.
  FAIL_TEST_IF_ERR(
      checkConstFoldedOutput(netFilename, {"x"}, {&x}, {bindings.get(output)}));
}

/// Test loading ReduceMean op from a ONNX model.
/// Input shape is 4D, one dimension is reduced, and output shape is 3D.
TEST_F(OnnxImporterTest, reduceMean4Dto3D) {
  testReductionOps("reduceMean4Dto3D.onnxtxt", {2, 2, 2},
                   {1.5, 3.5, 5.5, 7.5, 9.5, 11.5, 13.5, 15.5});
}

/// Test loading ReduceMean op from a ONNX model.
/// Input shape is 4D, one dimension is reduced, and output shape stays 4D.
TEST_F(OnnxImporterTest, reduceMean4Dto4D) {
  testReductionOps("reduceMean4Dto4D.onnxtxt", {2, 2, 2, 1},
                   {1.5, 3.5, 5.5, 7.5, 9.5, 11.5, 13.5, 15.5});
}

/// Test loading ReduceSum op from a ONNX model.
/// Input shape is 4D, one dimension is reduced, and output shape is 4D.
TEST_F(OnnxImporterTest, reduceSum4D) {
  testReductionOps("reduceSum4D.onnxtxt", {2, 2, 2, 1},
                   {3, 7, 11, 15, 19, 23, 27, 31});
}

/// Test loading ReduceMean op from a ONNX model.
/// Input shape is 4D, two dimensions are reduced, targeting ReduceMean
/// optimization using AvgPool. Output shape is 4D.
TEST_F(OnnxImporterTest, reduceMean2AvgPoolKeepDims) {
  testReductionOps("reduceMean2AvgPool.onnxtxt", {2, 2, 1, 1},
                   {2.5, 6.5, 10.5, 14.5});
}

/// Test loading ReduceMean op from a ONNX model.
/// Input shape is 4D, two dimensions are reduced, targeting ReduceMean
/// optimization using AvgPool. Output shape is 2D.
TEST_F(OnnxImporterTest, reduceMean2AvgPoolNoKeepDims) {
  testReductionOps("reduceMean2AvgPoolNoKeep.onnxtxt", {2, 2},
                   {2.5, 6.5, 10.5, 14.5});
}

/// Test loading ReduceMax op from a ONNX model.
/// Input shape is 4D, two dimensions are reduced,Output shape is 4D.
TEST_F(OnnxImporterTest, reduceMaxKeepDims) {
  testReductionOps("reduceMax.onnxtxt", {2, 2, 1, 1}, {4, 8, 12, 16});
}

/// Test loading ReduceMax op from a ONNX model.
/// Input shape is 4D, two dimensions are reduced, targeting ReduceMean
/// optimization using AvgPool. Output shape is 2D.
TEST_F(OnnxImporterTest, reduceMaxNoKeepDims) {
  testReductionOps("reduceMaxNoKeep.onnxtxt", {2, 2}, {4, 8, 12, 16});
}

/// Test loading ReduceMax op from a ONNX model.
/// Input shape is 4D, two dimensions are reduced,Output shape is 4D.
TEST_F(OnnxImporterTest, reduceMaxKeepDimsDefaultAxis) {
  testReductionOps("reduceMaxDefaultAxis.onnxtxt", {1, 1, 1, 1}, {16});
}

/// Test loading ReduceMin op from a ONNX model.
/// Input shape is 4D, two dimensions are reduced,Output shape is 4D.
TEST_F(OnnxImporterTest, reduceMinKeepDims) {
  testReductionOps("reduceMin.onnxtxt", {2, 2, 1, 1}, {1, 5, 9, 13});
}

/// Test loading ReduceMin op from a ONNX model.
/// Input shape is 4D, two dimensions are reduced, targeting ReduceMean
/// optimization using AvgPool. Output shape is 2D.
TEST_F(OnnxImporterTest, reduceMinNoKeepDims) {
  testReductionOps("reduceMinNoKeep.onnxtxt", {2, 2}, {1, 5, 9, 13});
}

/// Test loading ReduceMin op from a ONNX model.
/// Input shape is 4D, two dimensions are reduced,Output shape is 4D.
TEST_F(OnnxImporterTest, reduceMinKeepDimsDefaultAxis) {
  testReductionOps("reduceMinDefaultAxis.onnxtxt", {1, 1, 1, 1}, {1});
}

/// Test loading ReduceProd op from a ONNX model.
/// Input shape is 4D, one dimension is reduced, and output shape is 4D
TEST_F(OnnxImporterTest, reduceProd4D) {
  testReductionOps("reduceProd.onnxtxt", {2, 2, 2, 1},
                   {2, 12, 30, 56, 90, 132, 182, 240});
}

static void testDepthToSpace(std::string &filename,
                             const std::vector<dim_t> &expectedDims,
                             const std::vector<float> &expectedValues) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string netFilename =
      std::string(GLOW_DATA_PATH "tests/models/onnxModels/") + filename;

  PlaceholderBindings bindings;
  Placeholder *output;
  {
    // NCHW
    Tensor x(ElemKind::FloatTy, {1, 8, 2, 3});
    x.getHandle() = {0.,  1.,  2.,  3.,  4.,  5.,  9.,  10., 11., 12.,
                     13., 14., 18., 19., 20., 21., 22., 23., 27., 28.,
                     29., 30., 31., 32., 36., 37., 38., 39., 40., 41.,
                     45., 46., 47., 48., 49., 50., 54., 55., 56., 57.,
                     58., 59., 63., 64., 65., 66., 67., 68.};

    ONNXModelLoader onnxLD(netFilename, {"x"}, {&x.getType()}, *F);
    output = EXIT_ON_ERR(onnxLD.getSingleOutput());
    bindings.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(bindings, &mod, {"x"}, {&x});
  }

  auto *res = bindings.get(output);
  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  auto result = res->getHandle();
  EXPECT_TRUE(result.dims().vec() == expectedDims);
  for (size_t i = 0; i < result.size(); i++) {
    EXPECT_FLOAT_EQ(result.raw(i), expectedValues[i]);
  }
}

/// Test loading DepthToSpace with mode=CRD from an ONNX model.
TEST_F(OnnxImporterTest, depthToSpaceCRD) {
  std::string filename("depthToSpace_crd.onnxtxt");
  std::vector<dim_t> expectedDims = {1, 2, 4, 6};
  std::vector<float> expectedValues = {
      0,  9,  1,  10, 2,  11, 18, 27, 19, 28, 20, 29, 3,  12, 4,  13,
      5,  14, 21, 30, 22, 31, 23, 32, 36, 45, 37, 46, 38, 47, 54, 63,
      55, 64, 56, 65, 39, 48, 40, 49, 41, 50, 57, 66, 58, 67, 59, 68};
  testDepthToSpace(filename, expectedDims, expectedValues);
}

/// Test loading DepthToSpace with default mode(DCR) from an ONNX model.
TEST_F(OnnxImporterTest, depthToSpaceDCR) {
  std::string filename("depthToSpace.onnxtxt");
  std::vector<dim_t> expectedDims = {1, 2, 4, 6};
  std::vector<float> expectedValues = {
      0,  18, 1,  19, 2,  20, 36, 54, 37, 55, 38, 56, 3,  21, 4,  22,
      5,  23, 39, 57, 40, 58, 41, 59, 9,  27, 10, 28, 11, 29, 45, 63,
      46, 64, 47, 65, 12, 30, 13, 31, 14, 32, 48, 66, 49, 67, 50, 68,
  };
  testDepthToSpace(filename, expectedDims, expectedValues);
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

/// Test loading MatMul op from an ONNX model with dimension equal to 3
TEST_F(OnnxImporterTest, importMatMul) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  std::string netFilename(GLOW_DATA_PATH
                          "tests/models/onnxModels/matmul.onnxtxt");

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
  std::vector<dim_t> expectedDims = {20, 40, 40};
  EXPECT_EQ(result.dims().vec(), expectedDims);
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
  const dim_t kRows = 3;
  const dim_t kCols = 3;
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
  ASSERT_EQ(inputNode, mod.getPlaceholderByNameSlow("x"));

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
  Tensor indices(ElemKind::Int64ITy, {kNumIndices});
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
  EXPECT_EQ(idx, mod.getPlaceholderByNameSlow("indices"));
  auto *vals = llvm::dyn_cast<Placeholder>(STD->getValues().getNode());
  EXPECT_EQ(vals, mod.getPlaceholderByNameSlow("values"));
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
    Tensor indices(ElemKind::Int64ITy, {2});
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

/// Helper method to run the gather operator test cases.
/// \p filename contains the model .onnxtxt.
/// \p dataShape: data Tensor dimensions.
/// \p indicesShape: indices Tensor dimensions
/// \p expectedValues : output Tensor values expected.
template <class OpType>
static void gatherTestHelper(llvm::StringRef fileName,
                             llvm::ArrayRef<dim_t> dataShape,
                             llvm::ArrayRef<dim_t> indicesShape,
                             llvm::ArrayRef<dim_t> expectedDims) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  std::string netFilename =
      std::string(GLOW_DATA_PATH "tests/models/onnxModels/") + fileName.str();
  Placeholder *output;
  Tensor data(ElemKind::FloatTy, dataShape);
  Tensor indices(ElemKind::Int32ITy, indicesShape);

  {
    ONNXModelLoader onnxLD(netFilename, {"data", "indices"},
                           {&data.getType(), &indices.getType()}, *F);
    output = EXIT_ON_ERR(onnxLD.getSingleOutput());
  }

  // Verify structure: PH/PH -> Gather/GatherND -> Save -> PH.
  auto *saveNode = getSaveNodeFromDest(output);
  auto *node = saveNode->getInput().getNode();
  auto *nodeGather = llvm::dyn_cast<OpType>(node);
  ASSERT_TRUE(nodeGather);
  EXPECT_TRUE(nodeGather->getResult().dims().equals({expectedDims}));
}

/// Test loading gather op from a ONNX model.
TEST_F(OnnxImporterTest, importGather) {
  std::string filename("gather.onnxtxt");
  std::vector<dim_t> dataShape = {3, 2};
  std::vector<dim_t> indicesShape = {2, 4};
  std::vector<dim_t> expectedDims = {2, 4, 2};
  gatherTestHelper<GatherNode>(filename, dataShape, indicesShape, expectedDims);
}

/// Test loading gatherND op from a ONNX model.
TEST_F(OnnxImporterTest, importGatherND) {
  std::string filename("gatherND.onnxtxt");
  std::vector<dim_t> dataShape = {2, 2, 2};
  std::vector<dim_t> indicesShape = {2, 2};
  std::vector<dim_t> expectedDims = {2, 2};
  gatherTestHelper<GatherNDNode>(filename, dataShape, indicesShape,
                                 expectedDims);
}

/// Test loading ScatterND from an ONNX model.
// Simplified test
TEST_F(OnnxImporterTest, scatterND) {
  ExecutionEngine EE;
  auto &mod = EE.getModule();
  std::string netFilename(GLOW_DATA_PATH
                          "tests/models/onnxModels/scatterND.onnxtxt");
  auto *F = mod.createFunction("main");
  Placeholder *output;
  Tensor data(ElemKind::FloatTy, {8});
  Tensor indices(ElemKind::Int64ITy, {4, 1});
  Tensor updates(ElemKind::FloatTy, {4});

  ONNXModelLoader onnxLD(
      netFilename, {"data", "indices", "updates"},
      {&data.getType(), &indices.getType(), &updates.getType()}, *F);
  output = EXIT_ON_ERR(onnxLD.getSingleOutput());

  // Verify structure: PH/PH/PH -> ScatterND -> Save -> PH.
  ASSERT_EQ(mod.getPlaceholders().size(), 4);
  ASSERT_EQ(F->getNodes().size(), 2);
  auto *save = getSaveNodeFromDest(output);
  auto *scatter = llvm::dyn_cast<ScatterDataNode>(save->getInput().getNode());
  ASSERT_TRUE(scatter);
  EXPECT_TRUE(scatter->getResult().dims().equals({8}));
}

/// Test loading ScatterND from an ONNX model.
// multi-dim test
TEST_F(OnnxImporterTest, mscatterND) {
  ExecutionEngine EE;
  auto &mod = EE.getModule();
  std::string netFilename(GLOW_DATA_PATH
                          "tests/models/onnxModels/mscatterND.onnxtxt");
  auto *F = mod.createFunction("main");
  Placeholder *output;
  Tensor data(ElemKind::FloatTy, {4, 4, 4});
  Tensor indices(ElemKind::Int64ITy, {2, 1});
  Tensor updates(ElemKind::FloatTy, {2, 4, 4});

  ONNXModelLoader onnxLD(
      netFilename, {"data", "indices", "updates"},
      {&data.getType(), &indices.getType(), &updates.getType()}, *F);
  output = EXIT_ON_ERR(onnxLD.getSingleOutput());

  // Verify structure: PH/PH/PH -> ScatterND -> Save -> PH.
  ASSERT_EQ(mod.getPlaceholders().size(), 4);
  ASSERT_EQ(F->getNodes().size(), 2);
  auto *save = getSaveNodeFromDest(output);
  auto *scatter = llvm::dyn_cast<ScatterDataNode>(save->getInput().getNode());
  ASSERT_TRUE(scatter);
  EXPECT_TRUE(scatter->getResult().dims().equals({4, 4, 4}));
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
    EXPECT_EQ(mod.getPlaceholders().size(), 2);
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
                            llvm::ArrayRef<dim_t> inputShape,
                            llvm::ArrayRef<dim_t> starts,
                            llvm::ArrayRef<dim_t> outputShape) {
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
                       llvm::ArrayRef<dim_t> inputShape, ElemKind outputKind) {
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
TEST(onnx, importCastToBool) {
  importCast("castToBool.onnxtxt", "data", {1, 2, 2, 2}, ElemKind::BoolTy);
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
                      llvm::ArrayRef<dim_t> inputShape,
                      llvm::ArrayRef<sdim_t> starts,
                      llvm::ArrayRef<sdim_t> ends, PaddingMode mode,
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

TEST_F(OnnxImporterTest, importPadDefaultInputPads) {
  // This test Pad in opset v11 where "pads" is passed through the 2nd input.
  importPad("padDefaultInputPad.onnxtxt", "data", {4, 6, 5, 7} /* input */,
            {1, 2, -2, 0} /* starts */, {0, -2, 1, 2} /* ends */,
            PaddingMode::CONSTANT, 0.f, false);
}

TEST_F(OnnxImporterTest, importPadConstant) {
  importPad("padConstant.onnxtxt", "data", {4, 6, 5, 7} /* input */,
            {1, 2, -2, 0} /* starts */, {0, -2, 1, 2} /* ends */,
            PaddingMode::CONSTANT, 2.55f, false);
}

TEST_F(OnnxImporterTest, importPadConstantInput) {
  // This tests Pad in opset v11 where "pads" is passed through the 2nd input
  // and "value" through the 3rd input.
  importPad("padConstantInput.onnxtxt", "data", {4, 6, 5, 7} /* input */,
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

static void importPowTest(const std::string &netFilename, Tensor &x, Tensor &y,
                          std::vector<dim_t> &expectedDims,
                          std::vector<float> &expectedValues) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  PlaceholderBindings bindings;
  Placeholder *output;

  ONNXModelLoader onnxLD(netFilename, {"base", "exp"},
                         {&x.getType(), &y.getType()}, *F);
  output = EXIT_ON_ERR(onnxLD.getSingleOutput());
  bindings.allocate(mod.getPlaceholders());
  updateInputPlaceholdersByName(bindings, &mod, {"base"}, {&x});
  updateInputPlaceholdersByName(bindings, &mod, {"exp"}, {&y});

  auto *outputT = bindings.get(output);

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  auto outputH = outputT->getHandle();

  EXPECT_TRUE(outputH.dims().vec() == expectedDims);
  for (size_t i = 0; i < expectedValues.size(); i++) {
    EXPECT_EQ(outputH.raw(i), expectedValues[i]);
  }
}

TEST_F(OnnxImporterTest, pow_scalar_broadcast) {
  Tensor x(ElemKind::FloatTy, {2, 3});
  x.getHandle() = {1, 2, 3, 4, 5, 6};

  Tensor y(ElemKind::FloatTy, {1});
  y.getHandle() = {
      3,
  };

  std::string netFilename(
      GLOW_DATA_PATH "tests/models/onnxModels/pow_scalar_broadcast.onnxtxt");

  std::vector<dim_t> expectedDims = {2, 3};
  std::vector<float> expectedValues = {
      1., 8., 27., 64., 125, 216.,
  };

  importPowTest(netFilename, x, y, expectedDims, expectedValues);
}

TEST_F(OnnxImporterTest, pow_vector_broadcast) {
  Tensor x(ElemKind::FloatTy, {2, 3});
  x.getHandle() = {1, 2, 3, 4, 5, 6};

  Tensor y(ElemKind::FloatTy, {3});
  y.getHandle() = {
      1,
      2,
      3,
  };

  std::string netFilename(
      GLOW_DATA_PATH "tests/models/onnxModels/pow_array_broadcast.onnxtxt");

  std::vector<dim_t> expectedDims = {2, 3};
  std::vector<float> expectedValues = {
      1., 4., 27., 4., 25, 216.,
  };

  importPowTest(netFilename, x, y, expectedDims, expectedValues);
}

TEST_F(OnnxImporterTest, pow_element_wise) {
  Tensor x(ElemKind::FloatTy, {3});
  x.getHandle() = {1, 2, 3};

  Tensor y(ElemKind::FloatTy, {3});
  y.getHandle() = {4, 5, 6};

  std::string netFilename(GLOW_DATA_PATH
                          "tests/models/onnxModels/pow_element_wise.onnxtxt");

  std::vector<dim_t> expectedDims = {3};
  std::vector<float> expectedValues = {
      1.,
      32.,
      729.,
  };

  importPowTest(netFilename, x, y, expectedDims, expectedValues);
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
  auto indexH = indexT->getHandle<int64_t>();
  std::vector<dim_t> expectedDims = {1, 3, 2};
  std::vector<float> expectedValues = {
      4., 3., 8., 7., 12, 11.,
  };
  std::vector<int64_t> expectedIndices = {3, 2, 0, 1, 1, 0};

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

void testArgMinMax(llvm::StringRef filename, bool isMin,
                   const std::vector<dim_t> &expectedDims) {
  ExecutionEngine EE;
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string netFilename = std::string(GLOW_DATA_PATH) + filename.str();

  PlaceholderBindings bindings;
  Placeholder *PH;
  std::vector<dim_t> inDims = {2, 3, 4, 5};
  {
    Tensor inT(ElemKind::FloatTy, inDims);

    ONNXModelLoader onnxLD(netFilename, {"input"}, {&inT.getType()}, *F);
    PH = EXIT_ON_ERR(onnxLD.getOutputByName("scores"));
    bindings.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(bindings, &mod, {"input"}, {&inT});
  }

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  auto output = bindings.get(PH)->getHandle<int64_t>();
  EXPECT_TRUE(output.dims().vec() == expectedDims);

  auto *save = getSaveNodeFromDest(PH);
  if (isMin) {
    EXPECT_TRUE(llvm::isa<ArgMinNode>(save->getInput()));
  } else {
    EXPECT_TRUE(llvm::isa<ArgMaxNode>(save->getInput()));
  }
}

TEST_F(OnnxImporterTest, argMaxKeepDim) {
  testArgMinMax("tests/models/onnxModels/ArgMaxKeepDim.onnxtxt", false,
                {2, 3, 1, 5});
}

TEST_F(OnnxImporterTest, argMaxNoKeepDim) {
  testArgMinMax("tests/models/onnxModels/ArgMaxNoKeepDim.onnxtxt", false,
                {2, 4, 5});
}

TEST_F(OnnxImporterTest, argMaxDefault) {
  testArgMinMax("tests/models/onnxModels/ArgMaxDefault.onnxtxt", false,
                {1, 3, 4, 5});
}

TEST_F(OnnxImporterTest, argMinKeepDim) {
  testArgMinMax("tests/models/onnxModels/ArgMinKeepDim.onnxtxt", true,
                {2, 3, 1, 5});
}

TEST_F(OnnxImporterTest, argMinNoKeepDim) {
  testArgMinMax("tests/models/onnxModels/ArgMinNoKeepDim.onnxtxt", true,
                {2, 4, 5});
}

TEST_F(OnnxImporterTest, argMinDefault) {
  testArgMinMax("tests/models/onnxModels/ArgMinDefault.onnxtxt", true,
                {1, 3, 4, 5});
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
  auto indices = bindings.get(indicesPH)->getHandle<int64_t>();
  std::vector<dim_t> expectedDims = {1, 3, 2, 2};

  EXPECT_TRUE(result.dims().vec() == expectedDims);
  EXPECT_TRUE(indices.dims().vec() == expectedDims);

  std::vector<float> expectedResult = {58.0, 46.0, 33.0, 57.0, 55.0, 31.0,
                                       54.0, 53.0, 40.0, 52.0, 51.0, 38.0};
  std::vector<int64_t> expectedIndices = {15, 18, 36, 30, 13, 19,
                                          28, 43, 14, 11, 26, 44};

  for (size_t i = 0; i < expectedResult.size(); i++) {
    EXPECT_EQ(result.raw(i), expectedResult[i]);
    EXPECT_EQ(indices.raw(i), expectedIndices[i]);
  }
}

TEST_F(OnnxImporterTest, importMean) {
  ExecutionEngine EE;
  auto &mod = EE.getModule();
  std::string netFilename(GLOW_DATA_PATH
                          "tests/models/onnxModels/Mean.onnxtxt");
  auto *F = mod.createFunction("main");
  PlaceholderBindings bindings;
  Placeholder *resultPH;
  Tensor T0(ElemKind::FloatTy, {2, 3, 2});
  Tensor T1(ElemKind::FloatTy, {2, 3, 2});
  Tensor T2(ElemKind::FloatTy, {2, 3, 2});
  T0.getHandle() = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  T1.getHandle() = {11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
  T2.getHandle() = {2.5, 1, 2.5, 1, 2.5, 1, 2.5, 1, 2.5, 1, 0, 1};
  {
    ONNXModelLoader onnxLD(netFilename, {"T0", "T1", "T2"},
                           {&T0.getType(), &T1.getType(), &T2.getType()}, *F);
    resultPH = EXIT_ON_ERR(onnxLD.getOutputByName("Y"));
    bindings.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(bindings, &mod, {"T0", "T1", "T2"},
                                  {&T0, &T1, &T2});
  }
  EE.compile(CompilationMode::Infer);
  EE.run(bindings);
  auto result = bindings.get(resultPH)->getHandle();
  std::vector<dim_t> expectedDims = {2, 3, 2};
  EXPECT_TRUE(result.dims().vec() == expectedDims);
  std::vector<float> expectedResult = {4.5, 4, 4.5, 4, 4.5,      4,
                                       4.5, 4, 4.5, 4, 11.0 / 3, 4};
  for (size_t i = 0; i < expectedResult.size(); i++) {
    EXPECT_EQ(result.raw(i), expectedResult[i]);
  }
}

TEST_F(OnnxImporterTest, importMeanBroadcast) {
  ExecutionEngine EE;
  auto &mod = EE.getModule();
  std::string netFilename(GLOW_DATA_PATH
                          "tests/models/onnxModels/Mean_broadcast.onnxtxt");
  auto *F = mod.createFunction("main");
  PlaceholderBindings bindings;
  Placeholder *resultPH;
  Tensor T0(ElemKind::FloatTy, {1, 2, 1});
  Tensor T1(ElemKind::FloatTy, {3});
  Tensor T2(ElemKind::FloatTy, {1, 2, 3});
  T0.getHandle() = {0, 1};
  T1.getHandle() = {11, 10, 9};
  T2.getHandle() = {5, 4, 3, 2, 1, 0};

  {
    ONNXModelLoader onnxLD(netFilename, {"T0", "T1", "T2"},
                           {&T0.getType(), &T1.getType(), &T2.getType()}, *F);
    resultPH = EXIT_ON_ERR(onnxLD.getOutputByName("Y"));
    bindings.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(bindings, &mod, {"T0", "T1", "T2"},
                                  {&T0, &T1, &T2});
  }
  EE.compile(CompilationMode::Infer);
  EE.run(bindings);
  auto result = bindings.get(resultPH)->getHandle();
  std::vector<dim_t> expectedDims = {1, 2, 3};
  EXPECT_TRUE(result.dims().vec() == expectedDims);
  std::vector<float> expectedResult = {16.0 / 3, 14.0 / 3, 4.0,
                                       14.0 / 3, 4.0,      10.0 / 3};
  for (size_t i = 0; i < expectedResult.size(); i++) {
    EXPECT_EQ(result.raw(i), expectedResult[i]);
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

TEST_F(OnnxImporterTest, importLessEqual) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string netFilename(GLOW_DATA_PATH
                          "tests/models/onnxModels/CmpLTE.onnxtxt");

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

  CmpLTENode *CMPLTE = llvm::dyn_cast<CmpLTENode>(save->getInput().getNode());

  ASSERT_TRUE(CMPLTE);
  ASSERT_EQ(CMPLTE->getResult().dims().size(), 3);
  EXPECT_EQ(CMPLTE->getResult().dims()[0], 4);
  EXPECT_EQ(CMPLTE->getResult().dims()[1], 4);
  EXPECT_EQ(CMPLTE->getResult().dims()[2], 1);
}

TEST_F(OnnxImporterTest, importEqual) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string netFilename(GLOW_DATA_PATH
                          "tests/models/onnxModels/Equal.onnxtxt");

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

  CmpEQNode *CMPEQ = llvm::dyn_cast<CmpEQNode>(save->getInput().getNode());

  ASSERT_TRUE(CMPEQ);
  ASSERT_EQ(CMPEQ->getResult().dims().size(), 3);
  EXPECT_EQ(CMPEQ->getResult().dims()[0], 4);
  EXPECT_EQ(CMPEQ->getResult().dims()[1], 4);
  EXPECT_EQ(CMPEQ->getResult().dims()[2], 1);
}

static void importLogical(const std::string &netFilename,
                          llvm::ArrayRef<bool> LHS, llvm::ArrayRef<bool> RHS,
                          llvm::ArrayRef<dim_t> LHSShape,
                          llvm::ArrayRef<dim_t> RHSShape,
                          llvm::ArrayRef<dim_t> outputShape,
                          llvm::ArrayRef<bool> expectedValues) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  // Load the .onnxtxt model.
  Type LHSType(ElemKind::BoolTy, LHSShape);
  Type RHSType(ElemKind::BoolTy, RHSShape);
  ONNXModelLoader onnxLD(netFilename, {"LHS", "RHS"}, {&LHSType, &RHSType}, *F);

  // Get placeholder bindings
  PlaceholderBindings bindings;
  Placeholder *graphOutputVar;
  graphOutputVar = EXIT_ON_ERR(onnxLD.getSingleOutput());
  auto *LHSPH = mod.getPlaceholderByNameSlow("LHS");
  auto *LHSTensor = bindings.allocate(LHSPH);
  LHSTensor->getHandle<bool>() = LHS;
  auto *RHSPH = mod.getPlaceholderByNameSlow("RHS");
  auto *RHSTensor = bindings.allocate(RHSPH);
  RHSTensor->getHandle<bool>() = RHS;

  // Compile and run graph
  EE.compile(CompilationMode::Infer);
  bindings.allocate(mod.getPlaceholders());
  EE.run(bindings);
  auto result = bindings.get(graphOutputVar)->getHandle<bool>();

  // Validate results
  ASSERT_TRUE(result.dims() == (llvm::ArrayRef<dim_t>)outputShape);
  for (size_t i = 0; i < result.getType().size(); i++) {
    EXPECT_EQ(result.raw(i), (bool)expectedValues[i]);
  }
}

/// Test "and" operation of dimensions 4
TEST_F(OnnxImporterTest, importLogicAnd) {
  llvm::SmallVector<bool, 12> LHS = {true,  true,  false, false, true, true,
                                     false, false, false, false, true, true};
  llvm::SmallVector<bool, 12> RHS = {true,  true, false, true, false, true,
                                     false, true, true,  true, true,  true};
  std::vector<dim_t> LHSShape = {1, 2, 3, 2};
  std::vector<dim_t> RHSShape = {1, 2, 3, 2};
  std::vector<dim_t> outputShape = {1, 2, 3, 2};
  llvm::SmallVector<bool, 12> expectedValues = {true,  true,  false, false,
                                                false, true,  false, false,
                                                false, false, true,  true};
  std::string netFilename(GLOW_DATA_PATH
                          "tests/models/onnxModels/logicalAnd.onnxtxt");
  importLogical(netFilename, LHS, RHS, LHSShape, RHSShape, outputShape,
                expectedValues);
}

/// Test "broadcast and" of dimensions 4 and 2
TEST_F(OnnxImporterTest, importLogicBcastAnd) {
  llvm::SmallVector<bool, 12> LHS = {true,  true,  false, false, true, true,
                                     false, false, false, false, true, true};
  llvm::SmallVector<bool, 6> RHS = {false, true, true, true, true, false};
  std::vector<dim_t> LHSShape = {1, 2, 3, 2};
  std::vector<dim_t> RHSShape = {3, 2};
  std::vector<dim_t> outputShape = {1, 2, 3, 2};
  llvm::SmallVector<bool, 12> expectedValues = {false, true,  false, false,
                                                true,  false, false, false,
                                                false, false, true,  false};
  std::string netFilename(GLOW_DATA_PATH
                          "tests/models/onnxModels/logicalAndBcast.onnxtxt");
  importLogical(netFilename, LHS, RHS, LHSShape, RHSShape, outputShape,
                expectedValues);
}

/// Test "or" operation of dimensions 4
TEST_F(OnnxImporterTest, importLogicOr) {
  llvm::SmallVector<bool, 12> LHS = {true,  true,  false, false, true, true,
                                     false, false, false, false, true, true};
  llvm::SmallVector<bool, 12> RHS = {true,  true, false, true, false, true,
                                     false, true, true,  true, true,  true};
  std::vector<dim_t> LHSShape = {1, 2, 3, 2};
  std::vector<dim_t> RHSShape = {1, 2, 3, 2};
  std::vector<dim_t> outputShape = {1, 2, 3, 2};
  llvm::SmallVector<bool, 12> expectedValues = {
      true, true, false, true, true, true, false, true, true, true, true, true};
  std::string netFilename(GLOW_DATA_PATH
                          "tests/models/onnxModels/logicalOr.onnxtxt");
  importLogical(netFilename, LHS, RHS, LHSShape, RHSShape, outputShape,
                expectedValues);
}

/// Test "broadcast or" of dimensions 4 and 2
TEST_F(OnnxImporterTest, importLogicBcastOr) {
  llvm::SmallVector<bool, 12> LHS = {true,  true,  false, false, true, true,
                                     false, false, false, false, true, true};
  llvm::SmallVector<bool, 6> RHS = {false, true, true, true, true, false};
  std::vector<dim_t> LHSShape = {1, 2, 3, 2};
  std::vector<dim_t> RHSShape = {3, 2};
  std::vector<dim_t> outputShape = {1, 2, 3, 2};
  llvm::SmallVector<bool, 12> expectedValues = {
      true, true, true, true, true, true, false, true, true, true, true, true};
  std::string netFilename(GLOW_DATA_PATH
                          "tests/models/onnxModels/logicalOrBcast.onnxtxt");
  importLogical(netFilename, LHS, RHS, LHSShape, RHSShape, outputShape,
                expectedValues);
}

/// Test "xor" operation of dimensions 4
TEST_F(OnnxImporterTest, importLogicXor) {
  llvm::SmallVector<bool, 12> LHS = {true,  true,  false, false, true, true,
                                     false, false, false, false, true, true};
  llvm::SmallVector<bool, 12> RHS = {true,  true, false, true, false, true,
                                     false, true, true,  true, true,  true};
  std::vector<dim_t> LHSShape = {1, 2, 3, 2};
  std::vector<dim_t> RHSShape = {1, 2, 3, 2};
  std::vector<dim_t> outputShape = {1, 2, 3, 2};
  llvm::SmallVector<bool, 12> expectedValues = {false, false, false, true,
                                                true,  false, false, true,
                                                true,  true,  false, false};
  std::string netFilename(GLOW_DATA_PATH
                          "tests/models/onnxModels/logicalXor.onnxtxt");
  importLogical(netFilename, LHS, RHS, LHSShape, RHSShape, outputShape,
                expectedValues);
}

/// Test "broadcast xor" of dimensions 4 and 2
TEST_F(OnnxImporterTest, importLogicBcastXor) {
  llvm::SmallVector<bool, 12> LHS = {true,  true,  false, false, true, true,
                                     false, false, false, false, true, true};
  llvm::SmallVector<bool, 6> RHS = {false, true, true, true, true, false};
  std::vector<dim_t> LHSShape = {1, 2, 3, 2};
  std::vector<dim_t> RHSShape = {3, 2};
  std::vector<dim_t> outputShape = {1, 2, 3, 2};
  llvm::SmallVector<bool, 12> expectedValues = {true,  false, true,  true,
                                                false, true,  false, true,
                                                true,  true,  false, true};
  std::string netFilename(GLOW_DATA_PATH
                          "tests/models/onnxModels/logicalXorBcast.onnxtxt");
  importLogical(netFilename, LHS, RHS, LHSShape, RHSShape, outputShape,
                expectedValues);
}

/// Test not operation
TEST_F(OnnxImporterTest, importNot) {
  llvm::SmallVector<bool, 12> X = {true,  true,  false, false, true, true,
                                   false, false, false, false, true, true};
  std::vector<dim_t> XShape = {1, 2, 3, 2};
  std::vector<dim_t> YShape = {1, 2, 3, 2};
  llvm::SmallVector<bool, 12> expectedValues = {false, false, true,  true,
                                                false, false, true,  true,
                                                true,  true,  false, false};
  std::string netFilename(GLOW_DATA_PATH
                          "tests/models/onnxModels/logicalNot.onnxtxt");

  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  PlaceholderBindings bindings;
  Placeholder *graphOutputVar;

  // Load the .onnxtxt model.
  Type XType(ElemKind::BoolTy, XShape);
  ONNXModelLoader onnxLD(netFilename, {"X"}, {&XType}, *F);
  graphOutputVar = EXIT_ON_ERR(onnxLD.getSingleOutput());
  auto *XPH = mod.getPlaceholderByNameSlow("X");
  auto *XTensor = bindings.allocate(XPH);
  XTensor->getHandle<bool>() = X;

  // Compile and run the graph
  EE.compile(CompilationMode::Infer);
  bindings.allocate(mod.getPlaceholders());
  EE.run(bindings);

  // Validate results
  auto result = bindings.get(graphOutputVar)->getHandle<bool>();
  ASSERT_TRUE(result.dims() == (llvm::ArrayRef<dim_t>)YShape);
  for (size_t i = 0; i < result.getType().size(); i++) {
    EXPECT_EQ(result.raw(i), (bool)expectedValues[i]);
  }
}

/// Test loading NonZero from a ONNX model.
static void testNonZero(llvm::StringRef name,
                        const std::vector<dim_t> &expectedDims,
                        const std::vector<int64_t> &expVals) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  PlaceholderBindings bindings;
  Placeholder *out = nullptr;

  std::string netFilename(GLOW_DATA_PATH
                          "tests/models/onnxModels/NonZero.onnxtxt");
  {
    ONNXModelLoader onnxLD(netFilename, {}, {}, *F);
    out = EXIT_ON_ERR(onnxLD.getOutputByName(name));
    EXPECT_NE(out, nullptr);
  }

  // Constant -> NonZero -> PH (x2 for 3 models inside the file)
  ASSERT_EQ(mod.getPlaceholders().size(), 3);
  ASSERT_EQ(F->getNodes().size(), 3);

  EE.compile(CompilationMode::Infer);
  bindings.allocate(mod.getPlaceholders());
  EE.run(bindings);

  auto result = bindings.get(out)->getHandle<int64_t>();

  EXPECT_TRUE(result.dims().vec() == expectedDims);
  for (size_t i = 0; i < expVals.size(); i++) {
    EXPECT_EQ(result.raw(i), expVals[i]);
  }
}

/// Test loading NonZero using constant int32_t tensor initializer.
TEST_F(OnnxImporterTest, importNonZeroI32) {
  std::vector<int64_t> expVals = {
      0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3,
      3, 3, 3, 3, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1,
      0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 0, 1, 2, 0, 0, 1,
      2, 0, 1, 1, 0, 1, 1, 2, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0,
      0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0};
  testNonZero("out_i32", {5, 29}, expVals);
}

/// Test loading NonZero using constant float tensor initializer.
TEST_F(OnnxImporterTest, importNonZeroF) {
  std::vector<int64_t> expVals = {0,  1,  3,  4,  6,  8,  10,
                                  12, 14, 16, 18, 19, 21, 22};
  testNonZero("out_f", {1, 14}, expVals);
}

/// Test loading NonZero using constant float tensor initializer.
TEST_F(OnnxImporterTest, importNonZeroI64) {
  std::vector<int64_t> expVals = {0,  1,  3,  4,  6,  8,  10,
                                  12, 14, 16, 18, 19, 21, 22};
  testNonZero("out_i64", {1, 14}, expVals);
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

/// Test loading and inference of  ONNX ROIAlign of onnx example
TEST(onnx, ROIAlign_onnx) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  std::string netFilename(GLOW_DATA_PATH
                          "tests/models/onnxModels/ROIAlign_onnx.onnxtxt");
  PlaceholderBindings bindings;
  Placeholder *output;
  Tensor featureMap(ElemKind::FloatTy, {1, 1, 10, 10});
  Tensor boxes(ElemKind::FloatTy, {3, 4});
  Tensor batchedIndices(ElemKind::Int64ITy, {
                                                3,
                                            });

  featureMap.getHandle() = {
      0.2764, 0.7150, 0.1958, 0.3416, 0.4638, 0.0259, 0.2963, 0.6518, 0.4856,
      0.7250, 0.9637, 0.0895, 0.2919, 0.6753, 0.0234, 0.6132, 0.8085, 0.5324,
      0.8992, 0.4467, 0.3265, 0.8479, 0.9698, 0.2471, 0.9336, 0.1878, 0.4766,
      0.4308, 0.3400, 0.2162, 0.0206, 0.1720, 0.2155, 0.4394, 0.0653, 0.3406,
      0.7724, 0.3921, 0.2541, 0.5799, 0.4062, 0.2194, 0.4473, 0.4687, 0.7109,
      0.9327, 0.9815, 0.6320, 0.1728, 0.6119, 0.3097, 0.1283, 0.4984, 0.5068,
      0.4279, 0.0173, 0.4388, 0.0430, 0.4671, 0.7119, 0.1011, 0.8477, 0.4726,
      0.1777, 0.9923, 0.4042, 0.1869, 0.7795, 0.9946, 0.9689, 0.1366, 0.3671,
      0.7011, 0.6234, 0.9867, 0.5585, 0.6985, 0.5609, 0.8788, 0.9928, 0.5697,
      0.8511, 0.6711, 0.9406, 0.8751, 0.7496, 0.1650, 0.1049, 0.1559, 0.2514,
      0.7012, 0.4056, 0.7879, 0.3461, 0.0415, 0.2998, 0.5094, 0.3727, 0.5482,
      0.0502};

  boxes.getHandle() = {0, 0, 9, 9, 0, 5, 4, 9, 5, 5, 9, 9};

  batchedIndices.getHandle<int64_t>() = {0, 0, 0};
  std::vector<float> expectedResult = {
      0.4664, 0.4466, 0.3405, 0.5688, 0.6068, 0.3714, 0.4296, 0.3835, 0.5562,
      0.351,  0.2768, 0.4883, 0.5222, 0.5528, 0.4171, 0.4713, 0.4844, 0.6904,
      0.492,  0.8774, 0.6239, 0.7125, 0.6289, 0.3355, 0.3495,

      0.3022, 0.4305, 0.4696, 0.3978, 0.5423, 0.3656, 0.705,  0.5165, 0.3172,
      0.7015, 0.2912, 0.5059, 0.6476, 0.6235, 0.8299, 0.5916, 0.7389, 0.7048,
      0.8372, 0.8893, 0.6227, 0.6153, 0.7097, 0.6154, 0.4585,

      0.2384, 0.3379, 0.3717, 0.61,   0.7601, 0.3767, 0.3785, 0.7147, 0.9243,
      0.9727, 0.5749, 0.5826, 0.5709, 0.7619, 0.877,  0.5355, 0.2566, 0.2141,
      0.2796, 0.36,   0.4365, 0.3504, 0.2887, 0.3661, 0.2349,
  };

  ONNXModelLoader onnxLD(
      netFilename, {"featureMap", "boxes", "batchIndices"},
      {&featureMap.getType(), &boxes.getType(), &batchedIndices.getType()}, *F);

  bindings.allocate(mod.getPlaceholders());
  updateInputPlaceholdersByName(bindings, &mod,
                                {"featureMap", "boxes", "batchIndices"},
                                {&featureMap, &boxes, &batchedIndices});
  output = EXIT_ON_ERR(onnxLD.getOutputByName("result"));
  EE.compile(CompilationMode::Infer);
  EE.run(bindings);
  auto resultH = bindings.get(output)->getHandle<float>();
  std::vector<dim_t> outputShape = {3, 1, 5, 5};
  float delta = 1e-03;
  ASSERT_TRUE(resultH.dims() == (llvm::ArrayRef<dim_t>)outputShape);
  for (size_t i = 0; i < resultH.getType().size(); i++) {
    EXPECT_NEAR(resultH.raw(i), expectedResult[i], delta);
  }
}

/// Test loading and inference of ONNX MatMul operator with
/// 4D inputs.
TEST(onnx, MatMul4D) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  std::string netFilename(GLOW_DATA_PATH
                          "tests/models/onnxModels/MatMul4D.onnxtxt");
  PlaceholderBindings bindings;
  Placeholder *output;
  Placeholder *refOutput;

  ONNXModelLoader onnxLD(netFilename, {}, {}, *F);
  output = EXIT_ON_ERR(onnxLD.getOutputByName("Y"));
  refOutput = EXIT_ON_ERR(onnxLD.getOutputByName("Yref"));

  EE.compile(CompilationMode::Infer);
  bindings.allocate(mod.getPlaceholders());
  EE.run(bindings);
  auto resultH = bindings.get(output)->getHandle();
  auto refYH = bindings.get(refOutput)->getHandle();
  std::vector<dim_t> outputShape = {1, 2, 3, 3};
  float delta = 1e-03;
  ASSERT_TRUE(resultH.dims() == (llvm::ArrayRef<dim_t>)outputShape);
  for (size_t i = 0; i < resultH.getType().size(); i++) {
    EXPECT_NEAR(resultH.raw(i), refYH.raw(i), delta);
  }
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
  inputPH = mod.getPlaceholderByNameSlow("input");
  outputPH = mod.getPlaceholderByNameSlow("output");
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
  inputPH = mod.getPlaceholderByNameSlow("input");
  outputPH = mod.getPlaceholderByNameSlow("output");
  EXPECT_TRUE(inputPH);
  EXPECT_TRUE(outputPH);
  EXPECT_EQ(inputPH->dims()[0], 1);
  EXPECT_EQ(inputPH->dims()[1], 2);
  EXPECT_EQ(outputPH->dims()[0], 1);
  EXPECT_EQ(outputPH->dims()[1], 2);
}

static void importUnary(const std::string &netFilename,
                        llvm::ArrayRef<float> input,
                        llvm::ArrayRef<dim_t> inputShape,
                        llvm::ArrayRef<dim_t> outputShape,
                        llvm::ArrayRef<float> expectedValues) {

  float delta = 1e-08;
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  PlaceholderBindings bindings;
  Placeholder *graphOutputVar;
  // Load the .onnxtxt model
  Type inputType(ElemKind::FloatTy, inputShape);
  ONNXModelLoader onnxLD(netFilename, {"input"}, {&inputType}, *F);
  graphOutputVar = EXIT_ON_ERR(onnxLD.getSingleOutput());
  auto inputPH = mod.getPlaceholderByNameSlow("input");
  auto *inputTensor = bindings.allocate(inputPH);
  inputTensor->getHandle<float>() = input;
  EE.compile(CompilationMode::Infer);
  bindings.allocate(mod.getPlaceholders());
  EE.run(bindings);
  auto result = bindings.get(graphOutputVar)->getHandle<float>();
  ASSERT_TRUE(result.dims() == (llvm::ArrayRef<dim_t>)outputShape);
  for (size_t i = 0; i < result.getType().size(); i++) {
    EXPECT_NEAR(result.raw(i), (float)expectedValues[i], delta);
  }
}

TEST(onnx, importSign) {
  std::vector<float> input = {-1, -2, 0, -2, 1, 2, 1, 2, -10, 0, 0, -2};
  std::vector<dim_t> inputShape = {1, 2, 3, 2};
  std::vector<dim_t> outputShape = {1, 2, 3, 2};
  std::vector<float> expectedValues = {-1, -1, 0, -1, 1, 1, 1, 1, -1, 0, 0, -1};
  std::string netFilename(GLOW_DATA_PATH
                          "tests/models/onnxModels/sign.onnxtxt");
  importUnary(netFilename, input, inputShape, outputShape, expectedValues);
}

static void
testLoop(std::string &filename, const std::vector<dim_t> &expected_v_finalDims,
         const std::vector<dim_t> &expected_scan_output_finalDims,
         const std::vector<float> &expected_v_finalValues,
         const std::vector<float> &expectedscan_output_finalValues) {
  ExecutionEngine EE;
  auto &mod = EE.getModule();
  auto *F = mod.createFunction("main");

  std::string netFilename =
      std::string(GLOW_DATA_PATH "tests/models/onnxModels/") + filename;

  PlaceholderBindings bindings;
  Placeholder *v_final;
  Placeholder *scan_output_final;

  Tensor init_i(ElemKind::FloatTy, {1});
  init_i.getHandle() = {0};
  Tensor inc(ElemKind::FloatTy, {1});
  inc.getHandle() = {1};

  {
    ONNXModelLoader onnxLD(netFilename, {"init_i", "inc"},
                           {&init_i.getType(), &inc.getType()}, *F);

    v_final = EXIT_ON_ERR(onnxLD.getOutputByName("v_final"));
    scan_output_final =
        EXIT_ON_ERR(onnxLD.getOutputByName("scan_output_final"));

    bindings.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(bindings, &mod, {"init_i", "inc"},
                                  {&init_i, &inc});
  }

  auto *v_finalT = bindings.get(v_final);
  auto *scan_output_finalT = bindings.get(scan_output_final);

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  auto v_finalH = v_finalT->getHandle();
  auto scan_output_finalH = scan_output_finalT->getHandle();

  EXPECT_EQ(v_finalH.dims().vec(), expected_v_finalDims);
  EXPECT_EQ(scan_output_finalH.dims().vec(), expected_scan_output_finalDims);
  for (size_t i = 0; i < expected_v_finalValues.size(); i++) {
    EXPECT_FLOAT_EQ(v_finalH.raw(i), expected_v_finalValues[i]);
  }
  for (size_t i = 0; i < expectedscan_output_finalValues.size(); i++) {
    EXPECT_FLOAT_EQ(scan_output_finalH.raw(i),
                    expectedscan_output_finalValues[i]);
  }
}

TEST_F(OnnxImporterTest, importLoopStatic) {
  // In this loop, cond is not changed in the loop body.
  //
  // input (trip_count, cond)
  //
  // int max_trip_count = 10;
  // cond = true;
  // init_i = 0;
  // for (i=0; i< max_trip_count && cond; ++i){
  //   scan_output[i] = init_i;
  //   inti_i = init_i + inc;
  // }
  std::string filename("loop_static.onnxtxt");
  std::vector<dim_t> expected_v_finalDims = {1};
  std::vector<dim_t> expected_scan_output_finalDims = {10, 1};
  std::vector<float> expected_v_finalValues = {10.};
  std::vector<float> expectedscan_output_finalValues = {0., 1., 2., 3., 4.,
                                                        5., 6., 7., 8., 9.};
  testLoop(filename, expected_v_finalDims, expected_scan_output_finalDims,
           expected_v_finalValues, expectedscan_output_finalValues);
}

TEST_F(OnnxImporterTest, importLoopNoIteration) {
  // The loop should be zero iteration.
  //
  // input (trip_count, 0)
  //
  // int max_trip_count = 10;
  // cond = false;
  // init_i = 0;
  // for (i=0; i < max_trip_count && cond; ++i) {
  //   scan_output[i] = init_i;
  //   inti_i = init_i + inc;
  // }
  std::string filename("loop_no_iteration.onnxtxt");
  std::vector<dim_t> expected_v_finalDims = {1};
  std::vector<dim_t> expected_scan_output_finalDims = {1, 1};
  std::vector<float> expected_v_finalValues = {0.};
  std::vector<float> expectedscan_output_finalValues = {0.};
  testLoop(filename, expected_v_finalDims, expected_scan_output_finalDims,
           expected_v_finalValues, expectedscan_output_finalValues);
}

TEST(onnx, importLoopCond) {
  // In this loop, cond is updated in the loop body, but it should be folded
  // into a Constant during loading time.
  // The loop should exit by cond.
  //
  // input(trip_count, cond) :
  //
  // int max_trip_count = 9223372036854775807;
  // int reduce_i = 20;
  // for (i=0; i < max_trip_count && cond; ++i) {
  //   scan_output[i] = reduce_i;
  //   reduce_i = reduce_i - 1;
  //   cond = (bool)(reduce_i - 1);
  // }
  std::string filename("loop_cond.onnxtxt");
  std::vector<dim_t> expected_v_finalDims = {1};
  std::vector<dim_t> expected_scan_output_finalDims = {20, 1};
  std::vector<float> expected_v_finalValues = {0.};
  std::vector<float> expectedscan_output_finalValues = {
      20., 19., 18., 17., 16., 15., 14., 13., 12., 11.,
      10., 9.,  8.,  7.,  6.,  5.,  4.,  3.,  2.,  1.};
  testLoop(filename, expected_v_finalDims, expected_scan_output_finalDims,
           expected_v_finalValues, expectedscan_output_finalValues);
}

TEST(onnx, importLoopTripCount) {
  // The loop should exit by trip_count.
  //
  // input(trip_count, cond) :
  //
  // int max_trip_count = 20;
  // int reduce_i = 20;
  // for (i=0; i < max_trip_count && cond; ++i) {
  //   scan_output[i] = reduce_i;
  //   reduce_i = reduce_i - 1;
  //   cond = (bool)(reduce_i - 1);
  //  }
  std::string filename("loop_tripcount.onnxtxt");
  std::vector<dim_t> expected_v_finalDims = {1};
  std::vector<dim_t> expected_scan_output_finalDims = {20, 1};
  std::vector<float> expected_v_finalValues = {0.0};
  std::vector<float> expectedscan_output_finalValues = {
      20., 19., 18., 17., 16., 15., 14., 13., 12., 11.,
      10., 9.,  8.,  7.,  6.,  5.,  4.,  3.,  2.,  1.};
  testLoop(filename, expected_v_finalDims, expected_scan_output_finalDims,
           expected_v_finalValues, expectedscan_output_finalValues);
}

TEST(onnx, importLoopEmptyTripCount) {
  // The loop should ignore trip-count, so exit by cond.
  //
  // input ("", 1)
  //
  // int reduce_i = 10;
  // bool cond = true;
  // for (int i = 0; cond; ++i) {
  //   scan_output[i] = reduce_i;
  //   reduce_i = reduce_i - 1;
  //   cond = (bool)reduce_i;
  // }
  std::string filename("loop_empty_tripcount.onnxtxt");
  std::vector<dim_t> expected_v_finalDims = {1};
  std::vector<dim_t> expected_scan_output_finalDims = {10, 1};
  std::vector<float> expected_v_finalValues = {0.};
  std::vector<float> expectedscan_output_finalValues = {10., 9., 8., 7., 6.,
                                                        5.,  4., 3., 2., 1.};
  testLoop(filename, expected_v_finalDims, expected_scan_output_finalDims,
           expected_v_finalValues, expectedscan_output_finalValues);
}

TEST(onnx, importLoopEmptyCond) {
  // The loop should ignore cond, so exit by trip_count.
  //
  // input(trip_count, "") :
  //
  // int max_trip_count = 7;
  // int reduce_i = 5;
  // for (i=0; i < max_trip_count; ++i) {
  //   scan_output[i] = reduce_i;
  //   reduce_i = reduce_i - 1;
  //   cond = (bool)(reduce_i - 1); // ignored
  // }
  std::string filename("loop_emptycond.onnxtxt");
  std::vector<dim_t> expected_v_finalDims = {1};
  std::vector<dim_t> expected_scan_output_finalDims = {7, 1};
  std::vector<float> expected_v_finalValues = {-2.0};
  std::vector<float> expectedscan_output_finalValues = {5., 4., 3., 2.,
                                                        1., 0., -1.};
  testLoop(filename, expected_v_finalDims, expected_scan_output_finalDims,
           expected_v_finalValues, expectedscan_output_finalValues);
}

TEST(onnx, importLoopWithoutN) {
  // The loop should exit by trip_count.
  //
  // input(trip_count, cond) :
  // bool cond = true;
  // int max_trip_count = 10;
  // for (i=0; i < max_trip_count && cond; ++i) {
  //   scan_output[i] = i;
  //  }
  std::string filename("loop_withoutN.onnxtxt");
  std::vector<dim_t> expected_v_finalDims = {1};
  std::vector<dim_t> expected_scan_output_finalDims = {10, 1};
  std::vector<float> expected_v_finalValues = {0.0};
  std::vector<float> expectedscan_output_finalValues = {0., 1., 2., 3., 4.,
                                                        5., 6., 7., 8., 9.};
  testLoop(filename, expected_v_finalDims, expected_scan_output_finalDims,
           expected_v_finalValues, expectedscan_output_finalValues);
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
  }

  // Compile and run.
  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  // Verify RNN error.
  Placeholder *Y_err_ph = mod.getPlaceholderByNameSlow("Y_err");
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
  }

  // Compile and run.
  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  // Verify GRU error.
  Placeholder *Y_err_ph = mod.getPlaceholderByNameSlow("Y_err");
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
  }

  // Compile and run.
  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  // Verify LSTM error.
  Placeholder *Y_err_ph = mod.getPlaceholderByNameSlow("Y_err");
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
  Placeholder *Y_err_ph = mod.getPlaceholderByNameSlow("Y_err");
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
    Tensor indices(ElemKind::Int64ITy, {8});
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

/// Test loading AudioSpectrogram from an ONNX model. The ONNX model already
/// computes the error compared to a TensorFlow reference implementation.
static void importAudioSpectrogram(std::string fileName) {
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
  Placeholder *errPH = mod.getPlaceholderByNameSlow("spectrogram_err");
  EXPECT_TRUE(errPH);
  auto errH = bindings.get(errPH)->getHandle();
  auto fftLen = (errPH->getType()->dims()[1] - 1) * 2;
  for (size_t idx = 0; idx < errPH->getType()->size(); idx++) {
    float errVal = std::abs(errH.raw(idx)) / (float)(fftLen);
    EXPECT_TRUE(errVal < 1e-5);
  }
}

TEST_F(OnnxImporterTest, importAudioSpectrogramOneWindow) {
  importAudioSpectrogram(
      GLOW_DATA_PATH
      "tests/models/onnxModels/audioSpectrogramOneWindow.onnxtxt");
}

TEST_F(OnnxImporterTest, importAudioSpectrogramTwoWindow) {
  importAudioSpectrogram(
      GLOW_DATA_PATH
      "tests/models/onnxModels/audioSpectrogramTwoWindow.onnxtxt");
}

TEST_F(OnnxImporterTest, importAudioSpectrogramNonSquared) {
  importAudioSpectrogram(
      GLOW_DATA_PATH
      "tests/models/onnxModels/audioSpectrogramNonSquared.onnxtxt");
}

/// Test loading MFCC from an ONNX model. The ONNX model already computes
/// the error compared to a TensorFlow reference implementation.
static void importMFCC(std::string fileName) {
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
  Placeholder *errPH = mod.getPlaceholderByNameSlow("coefficients_err");
  EXPECT_TRUE(errPH);
  auto errH = bindings.get(errPH)->getHandle();
  for (size_t idx = 0; idx < errPH->getType()->size(); idx++) {
    EXPECT_TRUE(std::abs(errH.raw(idx)) < 1e-5);
  }
}

TEST_F(OnnxImporterTest, importMFCCOneWindow) {
  importMFCC(GLOW_DATA_PATH "tests/models/onnxModels/mfccOneWindow.onnxtxt");
}

TEST_F(OnnxImporterTest, importMFCCTwoWindow) {
  importMFCC(GLOW_DATA_PATH "tests/models/onnxModels/mfccTwoWindow.onnxtxt");
}

/// Test loading a custom ONNX Glow quantized TopK.
TEST_F(OnnxImporterTest, CustomGlowTopKQuantized) {
  ExecutionEngine EE;
  auto &mod = EE.getModule();
  auto *F = mod.createFunction("main");
  std::string netFilename(
      GLOW_DATA_PATH
      "tests/models/onnxModels/glow_custom_op_topk_quantized.onnxtxt");
  Placeholder *valuesPH, *indicesPH;
  {
    ONNXModelLoader onnxLD(netFilename, {}, {}, *F);
    valuesPH = EXIT_ON_ERR(onnxLD.getOutputByName("save_values"));
    indicesPH = EXIT_ON_ERR(onnxLD.getOutputByName("save_indices"));
  }

  // Verify structure: PH -> TopK -> Save -> PH.
  //                           |
  //                           v
  //                         Save -> PH
  EXPECT_EQ(mod.getPlaceholders().size(), 3);
  // TopK, Save nodes
  EXPECT_EQ(F->getNodes().size(), 3);

  auto *values = getSaveNodeFromDest(valuesPH);
  ASSERT_TRUE(values);
  EXPECT_EQ(values->getInput().getType()->getElementType(), ElemKind::Int8QTy);
  EXPECT_EQ(values->getInput().getType()->getScale(), 1.2f);
  EXPECT_EQ(values->getInput().getType()->getOffset(), 5);
  EXPECT_EQ(values->getInput().dims().vec(), std::vector<dim_t>({3, 1, 3}));

  auto *indices = getSaveNodeFromDest(indicesPH);
  ASSERT_TRUE(indices);
  EXPECT_EQ(indices->getInput().getType()->getElementType(),
            ElemKind::Int64ITy);
  EXPECT_EQ(indices->getInput().dims().vec(), std::vector<dim_t>({3, 1, 3}));

  EXPECT_EQ(indices->getInput().getNode(), values->getInput().getNode());

  auto *TKN = llvm::dyn_cast<TopKNode>(indices->getInput());
  ASSERT_TRUE(TKN);
  EXPECT_EQ(TKN->getK(), 3);

  auto *input = llvm::dyn_cast<Placeholder>(TKN->getInput());
  ASSERT_TRUE(input);
  EXPECT_EQ(input->dims().vec(), std::vector<dim_t>({3, 1, 5}));
  EXPECT_EQ(input->getType()->getElementType(), ElemKind::Int8QTy);
  EXPECT_EQ(input->getType()->getScale(), 1.2f);
  EXPECT_EQ(input->getType()->getOffset(), 5);
}

/// Test loading a custom ONNX Glow ChannelwiseQuantizedGroupConvolution.
TEST_F(OnnxImporterTest, CustomGlowChannelwiseQuantizedGroupConvolution) {
  ExecutionEngine EE;
  auto &mod = EE.getModule();
  auto *F = mod.createFunction("main");
  std::string netFilename(
      GLOW_DATA_PATH "tests/models/onnxModels/"
                     "glow_custom_op_channelwise_quantized_group_conv.onnxtxt");
  Placeholder *outputPH;
  {
    ONNXModelLoader onnxLD(netFilename, {}, {}, *F);
    outputPH = EXIT_ON_ERR(onnxLD.getSingleOutput());
  }

  // Verify structure:
  // {(PH -> Quantize), Constant, Constant, Constant, Constant} ->
  // ChannelwiseQuantizedConvolution -> Save -> PH.
  EXPECT_EQ(mod.getPlaceholders().size(), 2);
  EXPECT_EQ(mod.getConstants().size(), 6);
  // ChannelwiseQuantizedConvolution, Save, Quantize, Dequantize
  EXPECT_EQ(F->getNodes().size(), 4);

  auto *save = getSaveNodeFromDest(outputPH);
  ASSERT_TRUE(save);

  auto *DQN = llvm::dyn_cast<DequantizeNode>(save->getInput());
  ASSERT_TRUE(DQN);
  EXPECT_EQ(DQN->getInput().getType()->getElementType(), ElemKind::Int8QTy);
  EXPECT_EQ(DQN->getInput().getType()->getScale(), 1.0f);
  EXPECT_EQ(DQN->getInput().getType()->getOffset(), 0);
  EXPECT_EQ(DQN->getInput().dims().vec(), std::vector<dim_t>({1, 1, 3, 4}));

  auto *CN =
      llvm::dyn_cast<ChannelwiseQuantizedConvolutionNode>(DQN->getInput());
  ASSERT_TRUE(CN);
  EXPECT_EQ(CN->getKernels().vec(), std::vector<unsigned_t>({2, 1}));
  EXPECT_EQ(CN->getStrides().vec(), std::vector<unsigned_t>({1, 1}));
  EXPECT_EQ(CN->getPads().vec(), std::vector<unsigned_t>({0, 0, 0, 0}));
  EXPECT_EQ(CN->getGroup(), 2);
  EXPECT_EQ(CN->getDilation().vec(), std::vector<unsigned_t>({1, 1}));

  auto *QN = llvm::dyn_cast<QuantizeNode>(CN->getInput());
  ASSERT_TRUE(QN);
  EXPECT_EQ(QN->getResult().getType()->getElementType(), ElemKind::Int8QTy);
  EXPECT_EQ(QN->getResult().getType()->getScale(), 1.0f);
  EXPECT_EQ(QN->getResult().getType()->getOffset(), 0);
  EXPECT_EQ(QN->getResult().dims().vec(), std::vector<dim_t>({1, 2, 3, 2}));
  EXPECT_TRUE(llvm::isa<Placeholder>(QN->getInput()));

  auto *filter = llvm::dyn_cast<Constant>(CN->getFilter());
  ASSERT_TRUE(filter);
  EXPECT_EQ(filter->getOutput().getType()->getElementType(), ElemKind::Int8QTy);
  EXPECT_EQ(filter->getOutput().dims().vec(), std::vector<dim_t>({4, 2, 1, 1}));

  auto *bias = llvm::dyn_cast<Constant>(CN->getBias());
  ASSERT_TRUE(bias);
  EXPECT_EQ(bias->getOutput().getType()->getElementType(), ElemKind::Int32QTy);
  EXPECT_EQ(bias->getOutput().dims().vec(), std::vector<dim_t>({4}));

  auto *filterScales = llvm::dyn_cast<Constant>(CN->getFilterScales());
  ASSERT_TRUE(filterScales);
  EXPECT_EQ(filterScales->getOutput().getType()->getElementType(),
            ElemKind::FloatTy);
  EXPECT_EQ(filterScales->getOutput().dims().vec(), std::vector<dim_t>({4}));

  auto *filterOffsets = llvm::dyn_cast<Constant>(CN->getFilterOffsets());
  ASSERT_TRUE(filterOffsets);
  EXPECT_EQ(filterOffsets->getOutput().getType()->getElementType(),
            ElemKind::Int32ITy);
  EXPECT_EQ(filterOffsets->getOutput().dims().vec(), std::vector<dim_t>({4}));

  auto *biasScales = llvm::dyn_cast<Constant>(CN->getBiasScales());
  ASSERT_TRUE(biasScales);
  EXPECT_EQ(biasScales->getOutput().getType()->getElementType(),
            ElemKind::FloatTy);
  EXPECT_EQ(biasScales->getOutput().dims().vec(), std::vector<dim_t>({4}));

  auto *biasOffsets = llvm::dyn_cast<Constant>(CN->getBiasOffsets());
  ASSERT_TRUE(biasOffsets);
  EXPECT_EQ(biasOffsets->getOutput().getType()->getElementType(),
            ElemKind::Int32ITy);
  EXPECT_EQ(biasOffsets->getOutput().dims().vec(), std::vector<dim_t>({4}));
}

/// Upsample Test Helper
static void importUpsampleTest(std::string &netFilename) {
  ExecutionEngine EE;
  auto &mod = EE.getModule();
  auto *F = mod.createFunction("main");
  PlaceholderBindings bindings;
  Placeholder *resultPH;
  Tensor inputTensor(ElemKind::FloatTy, {1, 1, 2, 2});

  inputTensor.getHandle() = {1, 2, 3, 4};

  ONNXModelLoader onnxLD(netFilename, {"input"}, {&inputTensor.getType()}, *F);
  resultPH = EXIT_ON_ERR(onnxLD.getOutputByName("Y"));
  bindings.allocate(mod.getPlaceholders());
  updateInputPlaceholdersByName(bindings, &mod, {"input"}, {&inputTensor});

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  auto result = bindings.get(resultPH)->getHandle();
  std::vector<dim_t> expectedDims = {1, 1, 4, 6};

  EXPECT_TRUE(result.dims().vec() == expectedDims);

  std::vector<float> expectedResult = {1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2,
                                       3, 3, 3, 4, 4, 4, 3, 3, 3, 4, 4, 4};

  for (dim_t i = 0; i < expectedResult.size(); i++) {
    EXPECT_EQ(result.raw(i), expectedResult[i]);
  }
}

TEST_F(OnnxImporterTest, importUpsampleOpset7) {
  std::string netFilename(GLOW_DATA_PATH
                          "tests/models/onnxModels/upsampleOpset7.onnxtxt");
  importUpsampleTest(netFilename);
}

TEST_F(OnnxImporterTest, importUpsampleOpset9) {
  std::string netFilename(GLOW_DATA_PATH
                          "tests/models/onnxModels/upsampleOpset9.onnxtxt");
  importUpsampleTest(netFilename);
}

static void testIf(std::string filename, float inputVal, float outputVal) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string netFilename = std::string(GLOW_DATA_PATH) + filename;

  PlaceholderBindings bindings;
  Placeholder *output;
  {
    Tensor x(ElemKind::FloatTy, {1});
    x.getHandle() = {inputVal};

    ONNXModelLoader onnxLD(netFilename, {"input"}, {&x.getType()}, *F);
    output = EXIT_ON_ERR(onnxLD.getSingleOutput());
    bindings.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(bindings, &mod, {"input"}, {&x});
  }

  auto *res = bindings.get(output);
  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  auto result = res->getHandle();

  std::vector<float> expectedValues = {outputVal};
  for (size_t i = 0; i < expectedValues.size(); i++) {
    EXPECT_EQ(result.raw(i), expectedValues[i]);
  }
}

TEST(onnx, testIfConstantTrue) {
  testIf("tests/models/onnxModels/if_true.onnxtxt", 3.0f, 6.0f);
}

TEST(onnx, testIfConstantFalse) {
  testIf("tests/models/onnxModels/if_false.onnxtxt", 3.0f, 9.0f);
}

/// ResizeNearest Test Helper
static void importResizeNearest(std::string filename) {
  ExecutionEngine EE;
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string netFilename(filename);

  PlaceholderBindings bindings;
  Placeholder *output;
  Tensor in(ElemKind::FloatTy, {2, 2, 2, 2});
  in.getHandle() = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  {
    ONNXModelLoader onnxLD(netFilename, {"in"}, {&in.getType()}, *F);
    output = EXIT_ON_ERR(onnxLD.getSingleOutput());

    bindings.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(bindings, &mod, {"in"}, {&in});
  }

  auto *res = bindings.get(output);
  EE.compile(CompilationMode::Infer);
  EE.run(bindings);
  ASSERT_EQ(2, F->getNodes().size());

  auto *saveNode = getSaveNodeFromDest(output);
  auto *RN = llvm::dyn_cast<ResizeNearestNode>(saveNode->getInput());
  ASSERT_TRUE(RN);

  auto result = res->getHandle();
  std::vector<dim_t> expectedDims = {2, 2, 4, 4};
  EXPECT_EQ(result.dims().vec(), expectedDims);

  std::vector<float> expectedValues = {
      1.0,  1.0,  2.0,  2.0,  1.0,  1.0,  2.0,  2.0,  3.0,  3.0,  4.0,
      4.0,  3.0,  3.0,  4.0,  4.0,  5.0,  5.0,  6.0,  6.0,  5.0,  5.0,
      6.0,  6.0,  7.0,  7.0,  8.0,  8.0,  7.0,  7.0,  8.0,  8.0,  9.0,
      9.0,  10.0, 10.0, 9.0,  9.0,  10.0, 10.0, 11.0, 11.0, 12.0, 12.0,
      11.0, 11.0, 12.0, 12.0, 13.0, 13.0, 14.0, 14.0, 13.0, 13.0, 14.0,
      14.0, 15.0, 15.0, 16.0, 16.0, 15.0, 15.0, 16.0, 16.0};

  for (dim_t i = 0; i < 64; i++) {
    EXPECT_FLOAT_EQ(result.raw(i), expectedValues[i]);
  }

  // Constant Folding Test.
  FAIL_TEST_IF_ERR(checkConstFoldedOutput(netFilename, {"in"}, {&in},
                                          {bindings.get(output)}));
}

/// Test ONNX Resize mode=nearest.
TEST(onnx, importResizeNearest) {
  std::string netFilename(GLOW_DATA_PATH
                          "tests/models/onnxModels/resizeNearest.onnxtxt");
  importResizeNearest(netFilename);
}

/// Test ONNX Resize V11 mode=nearest that is compatible with V10 spec
TEST(onnx, importResizeNearestV11compat) {
  std::string netFilename(
      GLOW_DATA_PATH "tests/models/onnxModels/resizeNearestV11compat.onnxtxt");
  importResizeNearest(netFilename);
}

/// Test ONNX Resize V11 mode=nearest that is compatible with V10 spec
/// except that scales are inferred from sizes input.
TEST(onnx, importResizeNearestV11compat_sizes) {
  std::string netFilename(
      GLOW_DATA_PATH
      "tests/models/onnxModels/resizeNearestV11compat_sizes.onnxtxt");
  importResizeNearest(netFilename);
}

static void importResizeBilinear(std::string filename) {
  ExecutionEngine EE;
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  std::string netFilename(filename);

  PlaceholderBindings bindings;
  Placeholder *output;
  Tensor in(ElemKind::FloatTy, {1, 1, 2, 2});
  in.getHandle() = {1, 2, 3, 4};
  {
    ONNXModelLoader onnxLD(netFilename, {"in"}, {&in.getType()}, *F);
    output = EXIT_ON_ERR(onnxLD.getSingleOutput());

    bindings.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(bindings, &mod, {"in"}, {&in});
  }

  auto *res = bindings.get(output);
  EE.compile(CompilationMode::Infer);
  EE.run(bindings);
  ASSERT_EQ(4, F->getNodes().size());

  auto *saveNode = getSaveNodeFromDest(output);
  auto *TR = llvm::dyn_cast<ReshapeNode>(saveNode->getInput().getNode());
  ASSERT_TRUE(TR);
  auto *RN = llvm::dyn_cast<ResizeBilinearNode>(TR->getInput());
  ASSERT_TRUE(RN);

  auto result = res->getHandle();
  std::vector<dim_t> expectedDims = {1, 1, 4, 4};
  EXPECT_EQ(result.dims().vec(), expectedDims);

  std::vector<float> expectedValues = {1.0, 1.5, 2.0, 2.0, 2.0, 2.5, 3.0, 3.0,
                                       3.0, 3.5, 4.0, 4.0, 3.0, 3.5, 4.0, 4.0};

  for (dim_t i = 0; i < 16; i++) {
    EXPECT_FLOAT_EQ(result.raw(i), expectedValues[i]);
  }

  // Constant Folding Test.
  FAIL_TEST_IF_ERR(checkConstFoldedOutput(netFilename, {"in"}, {&in},
                                          {bindings.get(output)}));
}

TEST_F(OnnxImporterTest, importBoolFromInt) {
  ExecutionEngine EE;
  auto &mod = EE.getModule();
  std::string netFilename(GLOW_DATA_PATH
                          "tests/models/onnxModels/bool_from_int.onnxtxt");
  auto *F = mod.createFunction("main");
  PlaceholderBindings bindings;
  Placeholder *output;
  {
    ONNXModelLoader onnxLD(netFilename, {}, {}, *F);
    output = EXIT_ON_ERR(onnxLD.getSingleOutput());
    ASSERT_TRUE(output);
  }

  EE.compile(CompilationMode::Infer);
  bindings.allocate(mod.getPlaceholders());
  EE.run(bindings);

  std::vector<bool> expectedOut = {true, false, true};
  auto result = bindings.get(output)->getHandle<bool>();
  for (size_t i = 0; i < result.getType().size(); i++)
    EXPECT_EQ(result.raw(i), expectedOut[i]);
}

/// ResizeNearest Test Helper.
TEST(onnx, importResizeBilinear) {
  std::string netFilename(GLOW_DATA_PATH
                          "tests/models/onnxModels/resizeBilinear.onnxtxt");
  importResizeBilinear(netFilename);
}

/// Test ONNX Resize V11 mode=nearest that is compatible with V10 spec
TEST(onnx, importResizeBilinearV11compat) {
  std::string netFilename(
      GLOW_DATA_PATH "tests/models/onnxModels/resizeBilinearV11compat.onnxtxt");
  importResizeBilinear(netFilename);
}

/// Test ONNX Resize V11 mode=bilinear that is compatible with V10 spec
/// except that scales are inferred from sizes input.
TEST(onnx, importResizeBilinearV11compat_sizes) {
  std::string netFilename(
      GLOW_DATA_PATH
      "tests/models/onnxModels/resizeBilinearV11compat_sizes.onnxtxt");
  importResizeBilinear(netFilename);
}

/// Test loading a custom ONNX Glow net with NodeOpts.
TEST_F(OnnxImporterTest, CustomGlowWithNodeOpts) {
  ExecutionEngine EE;
  auto &mod = EE.getModule();
  auto *F = mod.createFunction("main");
  std::string netFilename(
      GLOW_DATA_PATH
      "tests/models/onnxModels/glow_custom_op_node_opts.onnxtxt");
  Placeholder *outputPH;
  BackendSpecificNodeInfo funNodeInfo;
  {
    ONNXModelLoader onnxLD(netFilename, {}, {}, *F, /* errPtr */ nullptr,
                           /* zipMode */ false, &funNodeInfo);
    outputPH = EXIT_ON_ERR(onnxLD.getSingleOutput());
  }

  auto itF = funNodeInfo.find(F);
  ASSERT_NE(itF, funNodeInfo.end());
  auto &nodeInfo = itF->second;

  SaveNode *save = getSaveNodeFromDest(outputPH);
  ASSERT_TRUE(save);
  // Verify that there are no options specified for the Save.
  EXPECT_EQ(nodeInfo.find(save), nodeInfo.end());

  // Verify that the options for the MatMul are loaded correctly.
  MatMulNode *MN = llvm::dyn_cast<MatMulNode>(save->getInput());
  auto itMN = nodeInfo.find(MN);
  ASSERT_NE(itMN, nodeInfo.end());
  llvm::StringMap<std::vector<std::string>> &opts = itMN->second;

  // attribute {
  //   name: "NodeOpt_BackendA_Option1"
  //   strings: "1"
  //   strings: "2"
  //   type: STRINGS
  // }
  auto itOpt1 = opts.find("BackendA_Option1");
  ASSERT_NE(itOpt1, opts.end());
  EXPECT_EQ(itOpt1->second.size(), 2);
  EXPECT_EQ(itOpt1->second[0], "1");
  EXPECT_EQ(itOpt1->second[1], "2");

  // attribute {
  //   name: "NodeOpt_BackendA_Option2"
  //   strings: "3"
  //   type: STRINGS
  // }
  auto itOpt2 = opts.find("BackendA_Option2");
  ASSERT_NE(itOpt2, opts.end());
  EXPECT_EQ(itOpt2->second.size(), 1);
  EXPECT_EQ(itOpt2->second[0], "3");

  // attribute {
  //   name: "NodeOpt_BackendB_Option3"
  //   strings: "4"
  //   strings: "5"
  //   type: STRINGS
  // }
  auto itOpt3 = opts.find("BackendB_Option3");
  ASSERT_NE(itOpt3, opts.end());
  EXPECT_EQ(itOpt3->second.size(), 2);
  EXPECT_EQ(itOpt3->second[0], "4");
  EXPECT_EQ(itOpt3->second[1], "5");
}

static bool vecContainsVal(const std::vector<runtime::DeviceIDTy> &vec,
                           runtime::DeviceIDTy val) {
  return std::find(vec.begin(), vec.end(), val) != vec.end();
}

/// Test loading a custom ONNX Glow net that has been already partitioned,
/// turned into a DAG, and then exported.
TEST_F(OnnxImporterTest, CustomGlowDAGMultiOp) {
  ExecutionEngine EE("Interpreter", /* deviceMemory (16GB) */ 0x400000000,
                     /* ignoreUserDeviceConfig */ false, /* numDevices */ 3);
  auto &mod = EE.getModule();
  std::string netFilename(
      GLOW_DATA_PATH
      "tests/models/onnxModels/glow_custom_dag_multi_op.onnxtxt");

  Placeholder *outputPH;
  Tensor *resultPartitionedT;
  PlaceholderBindings bindingsU;
  PlaceholderBindings bindingsP;

  runtime::PrePartitionedConfig PPC;
  Tensor mmIn0T(ElemKind::FloatTy, {10, 10});
  Tensor mmIn1T(ElemKind::FloatTy, {10, 10});
  Tensor addInT(ElemKind::FloatTy, {10, 10});
  mmIn0T.getHandle().randomize(-3.0, 3.0, mod.getPRNG());
  mmIn1T.getHandle().randomize(-3.0, 3.0, mod.getPRNG());
  addInT.getHandle().randomize(-3.0, 3.0, mod.getPRNG());
  Placeholder *mmIn0P = nullptr, *mmIn1P = nullptr, *addInP = nullptr;
  {
    ONNXModelLoader onnxLD(netFilename, {}, {}, mod, "main", &PPC,
                           /* errPtr */ nullptr, /* zipMode */ false);
    outputPH = EXIT_ON_ERR(onnxLD.getSingleOutput());
    NodeValue mmIn0NV;
    ASSIGN_VALUE_OR_FAIL_TEST(mmIn0NV, onnxLD.getNodeValueByName("mm0_in"));
    mmIn0P = llvm::dyn_cast<Placeholder>(mmIn0NV);
    NodeValue mmIn1NV;
    ASSIGN_VALUE_OR_FAIL_TEST(mmIn1NV, onnxLD.getNodeValueByName("mm1_in"));
    mmIn1P = llvm::dyn_cast<Placeholder>(mmIn1NV);
    NodeValue addInNV;
    ASSIGN_VALUE_OR_FAIL_TEST(addInNV, onnxLD.getNodeValueByName("add_in"));
    addInP = llvm::dyn_cast<Placeholder>(addInNV);
  }

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
    ONNXModelLoader onnxLD(netFilename, {}, {}, *F);
    outputPH = EXIT_ON_ERR(onnxLD.getSingleOutput());
    NodeValue mmIn0NV;
    ASSIGN_VALUE_OR_FAIL_TEST(mmIn0NV, onnxLD.getNodeValueByName("mm0_in"));
    mmIn0P = llvm::dyn_cast<Placeholder>(mmIn0NV);
    NodeValue mmIn1NV;
    ASSIGN_VALUE_OR_FAIL_TEST(mmIn1NV, onnxLD.getNodeValueByName("mm1_in"));
    mmIn1P = llvm::dyn_cast<Placeholder>(mmIn1NV);
    NodeValue addInNV;
    ASSIGN_VALUE_OR_FAIL_TEST(addInNV, onnxLD.getNodeValueByName("add_in"));
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

/// Utility function to test ONNX Gemm import.
static void importGemm(std::string filename, bool hasC, bool batchedC,
                       bool transA, bool transB) {
  ExecutionEngine EE;
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  std::string netFilename(filename);

  PlaceholderBindings bindings;
  Placeholder *output;

  Tensor tensorA;
  if (transA) {
    tensorA = Tensor(ElemKind::FloatTy, {3, 2});
    tensorA.getHandle() = {1, 4, 2, 5, 3, 6};
  } else {
    tensorA = Tensor(ElemKind::FloatTy, {2, 3});
    tensorA.getHandle() = {1, 2, 3, 4, 5, 6};
  }

  Tensor tensorB;
  if (transB) {
    tensorB = Tensor(ElemKind::FloatTy, {4, 3});
    tensorB.getHandle() = {1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4};
  } else {
    tensorB = Tensor(ElemKind::FloatTy, {3, 4});
    tensorB.getHandle() = {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4};
  }

  Tensor tensorC;
  if (batchedC) {
    tensorC = Tensor(ElemKind::FloatTy, {2, 4});
    tensorC.getHandle() = {1, 2, 3, 4, 1, 2, 3, 4};
  } else {
    tensorC = Tensor(ElemKind::FloatTy, {4});
    tensorC.getHandle() = {1, 2, 3, 4};
  }

  {
    ONNXModelLoader onnxLD(netFilename, {}, {}, *F);
    output = EXIT_ON_ERR(onnxLD.getSingleOutput());
    bindings.allocate(mod.getPlaceholders());
    if (hasC) {
      updateInputPlaceholdersByName(bindings, &mod, {"A", "B", "C"},
                                    {&tensorA, &tensorB, &tensorC});
    } else {
      updateInputPlaceholdersByName(bindings, &mod, {"A", "B"},
                                    {&tensorA, &tensorB});
    }
  }

  auto *saveNode = getSaveNodeFromDest(output);
  auto *GN = llvm::dyn_cast<GemmNode>(saveNode->getInput().getNode());
  ASSERT_TRUE(GN);

  auto *res = bindings.get(output);
  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  // Check output size.
  auto result = res->getHandle();
  std::vector<dim_t> expectedDims = {2, 4};
  EXPECT_EQ(result.dims().vec(), expectedDims);

  // Check output values.
  std::vector<float> expectedValues(8);
  if (hasC) {
    expectedValues = {7.0, 14.0, 21.0, 28.0, 16.0, 32.0, 48.0, 64.0};
  } else {
    expectedValues = {6.0, 12.0, 18.0, 24.0, 15.0, 30.0, 45.0, 60.0};
  }
  for (dim_t i = 0; i < 8; i++) {
    EXPECT_FLOAT_EQ(result.raw(i), expectedValues[i]);
  }
}

/// Test ONNX Gemm.
TEST_F(OnnxImporterTest, importGemmNoC) {
  std::string netFilename(GLOW_DATA_PATH
                          "tests/models/onnxModels/gemmNoC.onnxtxt");
  importGemm(netFilename, /* hasC */ false, /* batchedC */ false,
             /* transA */ false, /* transB */ false);
}

TEST_F(OnnxImporterTest, importGemmSingleC) {
  std::string netFilename(GLOW_DATA_PATH
                          "tests/models/onnxModels/gemmSingleC.onnxtxt");
  importGemm(netFilename, /* hasC */ true, /* batchedC */ false,
             /* transA */ false, /* transB */ false);
}

TEST_F(OnnxImporterTest, importGemmBatchedC) {
  std::string netFilename(GLOW_DATA_PATH
                          "tests/models/onnxModels/gemmBatchedC.onnxtxt");
  importGemm(netFilename, /* hasC */ true, /* batchedC */ true,
             /* transA */ false, /* transB */ false);
}

TEST_F(OnnxImporterTest, importGemmTransA) {
  std::string netFilename(GLOW_DATA_PATH
                          "tests/models/onnxModels/gemmTransA.onnxtxt");
  importGemm(netFilename, /* hasC */ true, /* batchedC */ false,
             /* transA */ true, /* transB */ false);
}

TEST_F(OnnxImporterTest, importGemmTransB) {
  std::string netFilename(GLOW_DATA_PATH
                          "tests/models/onnxModels/gemmTransB.onnxtxt");
  importGemm(netFilename, /* hasC */ true, /* batchedC */ false,
             /* transA */ false, /* transB */ true);
}

TEST(onnx, importTransposeNullPerm) {
  ExecutionEngine EE;
  auto &mod = EE.getModule();
  std::string netFilename(
      GLOW_DATA_PATH "tests/models/onnxModels/transpose_null_perm.onnxtxt");
  auto *F = mod.createFunction("main");
  PlaceholderBindings bindings;
  Placeholder *output_0;

  Tensor input_0(ElemKind::Int32ITy, {1, 2, 3, 4});
  input_0.getHandle<int32_t>() = {1, 2, 3, 6, 4, 5, 6, 3, 1, 2, 3, 6,
                                  4, 5, 6, 3, 7, 8, 9, 2, 3, 5, 7, 1};
  {
    ONNXModelLoader onnxLD(netFilename, {"X1"}, {&input_0.getType()}, *F);

    output_0 = EXIT_ON_ERR(onnxLD.getOutputByName("output0"));

    bindings.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(bindings, &mod, {"X1"}, {&input_0});
  }

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  std::vector<dim_t> expectedDims = {4, 3, 2, 1};
  std::vector<int32_t> expectedValues = {1, 4, 4, 7, 1, 3, 2, 5, 5, 8, 2, 5,
                                         3, 6, 6, 9, 3, 7, 6, 3, 3, 2, 6, 1};

  auto result = bindings.get(output_0)->getHandle<int32_t>();

  EXPECT_EQ(result.dims().vec(), expectedDims);
  for (dim_t i = 0; i < 24; i++) {
    EXPECT_FLOAT_EQ(result.raw(i), expectedValues[i]);
  }
}

TEST(onnx, importNames) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  std::string NetFilename(GLOW_DATA_PATH
                          "tests/models/onnxModels/legalizeNames.onnxtxt");

  PlaceholderBindings bindings;
  Placeholder *graphOutputVar;
  Type input_type(ElemKind::FloatTy, {1, 2, 4, 3});
  ONNXModelLoader onnxLD(NetFilename, {"data"}, {&input_type}, *F);
  graphOutputVar = EXIT_ON_ERR(onnxLD.getSingleOutput());
  auto PH = mod.getPlaceholderByNameSlow("data");
  auto *inTensor = bindings.allocate(PH);
  inTensor->getHandle().randomize(-10.0, 10.0, mod.getPRNG());
  // Compile&run the graph, and check the output
  EE.compile(CompilationMode::Infer);
  vector<std::string> origNames = {"a__1",  "a__1", "a__3__3", "a__2",
                                   "a__1_", "a__b", "a"};
  auto *currNode = (Node *)getSaveNodeFromDest(graphOutputVar);
  for (size_t i = 0; i < origNames.size(); i++) {
    auto *prevNode = currNode->getNthInput(0).getNode();
    // Make sure original names are retained in the legalized names.
    EXPECT_EQ(prevNode->getName().find(origNames[i]), 0);
    currNode = prevNode;
  }
}

TEST(onnx, importClipV11) {
  // Test loading Clip in opset v11 format where min(-2) and max(2) are passed
  // as inputs.
  ExecutionEngine EE;
  auto &mod = EE.getModule();
  std::string netFilename(GLOW_DATA_PATH
                          "tests/models/onnxModels/clipv11.onnxtxt");
  auto *F = mod.createFunction("main");
  PlaceholderBindings bindings;
  Placeholder *output_0;

  Tensor X(ElemKind::FloatTy, {1, 2, 2, 2});
  X.getHandle() = {-3, -2, -1, 0, 1, 2, 3, 4};

  {
    ONNXModelLoader onnxLD(netFilename, {"X"}, {&X.getType()}, *F);
    output_0 = EXIT_ON_ERR(onnxLD.getOutputByName("output0"));
    bindings.allocate(mod.getPlaceholders());
    updateInputPlaceholdersByName(bindings, &mod, {"X"}, {&X});
  }

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  std::vector<dim_t> expectedDims = {1, 2, 2, 2};
  std::vector<float> expectedValues = {-2, -2, -1, 0, 1, 2, 2, 2};
  auto result = bindings.get(output_0)->getHandle();
  EXPECT_EQ(result.dims().vec(), expectedDims);

  for (size_t i = 0; i < 8; i++) {
    EXPECT_FLOAT_EQ(result.raw(i), expectedValues[i]);
  }
}

// Utility function to test ONNX Softmax
static void testSoftmax(const std::string &modelName,
                        const std::vector<dim_t> &expectedDims,
                        const std::vector<float> &expectedValues) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  // Input.
  Tensor x(ElemKind::FloatTy, {2, 2, 2, 2});
  x.getHandle() = {0.0, 1.0, 2.0,  3.0,  4.0,  5.0,  6.0,  7.0,
                   8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0};

  // Load model.
  std::string netFilename =
      std::string(GLOW_DATA_PATH "tests/models/onnxModels/") + modelName;
  ONNXModelLoader onnxLD(netFilename, {"x"}, {&x.getType()}, *F);
  Placeholder *output = EXIT_ON_ERR(onnxLD.getSingleOutput());

  // Allocate placeholders.
  PlaceholderBindings bindings;
  bindings.allocate(mod.getPlaceholders());
  updateInputPlaceholdersByName(bindings, &mod, {"x"}, {&x});

  auto *res = bindings.get(output);
  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  // Compare results.
  auto result = res->getHandle();
  EXPECT_TRUE(result.dims().vec() == expectedDims);
  for (dim_t i = 0; i < result.size(); i++) {
    EXPECT_FLOAT_EQ(result.raw(i), expectedValues[i]);
  }
}

/// Test loading Softmax from a ONNX model.
TEST_F(OnnxImporterTest, softmax) {
  testSoftmax("softmax11.onnxtxt", {2, 2, 2, 2},
              {5.7661277e-04, 1.5673960e-03, 4.2606238e-03, 1.1581578e-02,
               3.1481992e-02, 8.5576929e-02, 2.3262219e-01, 6.3233274e-01,
               5.7661277e-04, 1.5673960e-03, 4.2606238e-03, 1.1581578e-02,
               3.1481992e-02, 8.5576929e-02, 2.3262219e-01, 6.3233274e-01});
}
/// Test loading Softmax opset13 from a ONNX model.
TEST_F(OnnxImporterTest, softmax13) {
  testSoftmax("softmax13.onnxtxt", {2, 2, 2, 2},
              {0.11920292, 0.11920292, 0.880797, 0.880797, 0.11920292,
               0.11920292, 0.880797, 0.880797, 0.11920292, 0.11920292, 0.880797,
               0.880797, 0.11920292, 0.11920292, 0.880797, 0.880797});
}
