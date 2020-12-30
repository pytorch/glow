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

#include "BackendTestUtils.h"

#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Graph/CustomOpData.h"
#include "glow/Graph/Graph.h"
#include "glow/Graph/OpRepository.h"
#include "glow/Quantization/Base/Base.h"

#include "gtest/gtest.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <functional>
#include <numeric>

using namespace glow;
using namespace std;

class CustomOpTest : public ::testing::TestWithParam<std::string> {
public:
  ~CustomOpTest() { OpRepository::get()->clear(); };

protected:
  PlaceholderBindings bindings_;
  ExecutionEngine EE_{GetParam()};
  Module &mod_ = EE_.getModule();
  Function *F_ = mod_.createFunction("main");
};

#ifndef GLOW_DATA_PATH
#define GLOW_DATA_PATH
#endif

// Utility for registering CustomOp.
static bool registerCustomOp(std::string &opType, std::string &packageName,
                             std::vector<ParamInfo> &paramInfo,
                             std::vector<NodeIOInfo> &inputIOs,
                             std::vector<NodeIOInfo> &outputIOs,
                             std::vector<ImplementationInfo> &implInfo,
                             std::string &verificationLibrary) {
  // Register operation.
  OpRepository *opRepo = OpRepository::get();
  OperationInfo scaleOpInfo(opType, packageName, paramInfo, inputIOs, outputIOs,
                            implInfo, verificationLibrary);
  return ERR_TO_BOOL(opRepo->registerOperation(scaleOpInfo));
}

TEST_P(CustomOpTest, CustomOpRelu) {
  // Op info.
  std::string opName = "CustomRelu";
  std::string opPackage = "OpPackage";
  std::string verificationLibrary{GLOW_DATA_PATH
                                  "tests/unittests/libCustomRelu.so"};
  const std::string cppFileName{GLOW_DATA_PATH
                                "tests/unittests/customop_examples.cpp"};
  // Parameter info.
  ParamInfo alphaInfo("alpha", CustomOpDataType::DTFloat32,
                      true /* isScalar */);
  ParamInfo betaInfo("beta", CustomOpDataType::DTFloat32, true /* isScalar */);
  std::vector<ParamInfo> paramInfo({alphaInfo, betaInfo});

  // IO info..
  std::vector<NodeIOInfo> inputIOs({NodeIOInfo("input", 1)});
  std::vector<NodeIOInfo> outputIOs({NodeIOInfo("reluOut", 1)});

  // Implementation info.
  std::vector<ImplementationInfo> implInfo;
  implInfo.push_back(ImplementationInfo("Interpreter", "customReluExecute",
                                        (void *)(&verificationLibrary)));

  // Register operation in OpRepository.
  EXPECT_FALSE(registerCustomOp(opName, opPackage, paramInfo, inputIOs,
                                outputIOs, implInfo, verificationLibrary));

  // Create inputs and output types.
  // In application, this info is provided by the model.
  std::vector<NodeValue> inputs;
  auto *I1 = mod_.createPlaceholder(ElemKind::FloatTy, {10}, "I1",
                                    false /* isTrainable */);
  auto inputH = bindings_.allocate(I1)->getHandle<float>();
  inputH.randomize(-1.0, 1.0, mod_.getPRNG());
  inputs.push_back(I1);

  std::vector<TypeRef> outputTypes;
  outputTypes.push_back(mod_.uniqueType(ElemKind::FloatTy, {10}));

  // Populate custom op data. This is typically provided in the model.
  auto opInfo = OpRepository::get()->getOperationInfo(opName, opPackage);
  assert(opInfo && "Could not retrieve OperationInfo.");

  CustomOpData relu1Data("relu1", opName, opPackage, opInfo->getParamInfo());
  float alpha = 0.5f, beta = 0.2f;

  relu1Data.addParam("alpha", alpha);
  relu1Data.addParam("beta", beta);

  // Create graph.
  CustomOpNode *customOp = F_->createCustomOp(
      relu1Data.getName(), relu1Data.getTypeName(), relu1Data.getPackageName(),
      inputs, outputTypes, relu1Data);
  SaveNode *S1 = F_->createSave("S1", customOp->getNthResult(0));

  bindings_.allocate(S1->getPlaceholder());

  // Compile and run.
  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto resultH = bindings_.get(S1->getPlaceholder())->getHandle<float>();
  for (int i = 0; i < inputH.size(); i++) {
    float v = inputH.raw(i) * alpha + beta;
    float r = v > 0 ? (v < 1 ? v : 1) : 0;
    EXPECT_EQ(resultH.raw(i), r);
  }
}

/// This example intends to test following:
/// - Multiple inputs and outputs, each of different type.
/// - Multiple data types for inputs and outputs.
/// - CustomOpCopy (TestPackage::TestOp) copies nth input to nth output.
TEST_P(CustomOpTest, CustomOpCopy) {
  // Op info.
  std::string typeName = "TestOp";
  std::string packageName = "TestPackage";
  std::string functionsLibrary{GLOW_DATA_PATH
                               "tests/unittests/libCustomOpCopy.so"};
  const std::string cppFileName{GLOW_DATA_PATH
                                "tests/unittests/customop_examples.cpp"};
  // Parameter info.
  ParamInfo floatInfo("floatParam", CustomOpDataType::DTFloat32,
                      true /* isScalar */);
  ParamInfo intInfo("intParam", CustomOpDataType::DTIInt32,
                    true /* isScalar */);
  ParamInfo floatVecInfo("floatVecParam", CustomOpDataType::DTFloat32,
                         false /* isScalar */, 5);
  ParamInfo intVecInfo("intVecParam", CustomOpDataType::DTIInt32,
                       false /* isScalar */, 5);
  std::vector<ParamInfo> paramInfo(
      {floatInfo, intInfo, floatVecInfo, intVecInfo});

  // IO info.
  std::vector<NodeIOInfo> inputIOs(
      {NodeIOInfo("floatIn", 3), NodeIOInfo("float16In", 3),
       NodeIOInfo("int16In", 3), NodeIOInfo("int8In", 3),
       NodeIOInfo("int32IIn", 3), NodeIOInfo("int64IIn", 3)});
  std::vector<NodeIOInfo> outputIOs(
      {NodeIOInfo("floatOut", 3), NodeIOInfo("float16Out", 3),
       NodeIOInfo("int16Out", 3), NodeIOInfo("int8Out", 3),
       NodeIOInfo("int32IOut", 3), NodeIOInfo("int64IOut", 3)});

  // Implementation info for Interpreter backend.
  ImplementationInfo dummyCopyImpl("Interpreter", "TestCustomOp",
                                   (void *)&functionsLibrary);
  std::vector<ImplementationInfo> implInfo({dummyCopyImpl});

  // Register operation in OpRepository.
  EXPECT_FALSE(registerCustomOp(typeName, packageName, paramInfo, inputIOs,
                                outputIOs, implInfo, functionsLibrary));

  // Create Graph.
  std::vector<dim_t> dims = {5};
  TypeRef tyfp32 = mod_.uniqueType(ElemKind::FloatTy, dims);
  TypeRef tyfp16 = mod_.uniqueType(ElemKind::Float16Ty, dims);
  TypeRef tyint8 = mod_.uniqueType(ElemKind::Int8QTy, dims, 0.02, -127);
  TypeRef tyuint8 = mod_.uniqueType(ElemKind::UInt8QTy, dims, 0.04, -127);
  TypeRef tyintI32 = mod_.uniqueType(ElemKind::Int32ITy, dims);
  TypeRef tyintI64 = mod_.uniqueType(ElemKind::Int64ITy, dims);
  std::vector<TypeRef> types = {tyfp32,  tyfp16,   tyint8,
                                tyuint8, tyintI32, tyintI64};

  auto *floatIn = mod_.createPlaceholder(tyfp32, "floatIn", false);
  auto *float16In = mod_.createPlaceholder(tyfp16, "float16In", false);
  auto *int8In = mod_.createPlaceholder(tyint8, "int8In", false);
  auto *uint8In = mod_.createPlaceholder(tyuint8, "uint8In", false);
  auto *int32IIn = mod_.createPlaceholder(tyintI32, "int32IIn", false);
  auto *int64IIn = mod_.createPlaceholder(tyintI64, "int64IIn", false);
  bindings_.allocate(floatIn)->getHandle<float>().randomize(-1.0, 1.0,
                                                            mod_.getPRNG());
  bindings_.allocate(float16In)->getHandle<float16_t>().randomize(
      -100.0, 100.0, mod_.getPRNG());
  bindings_.allocate(int8In)->getHandle<int8_t>().randomize(-128, 127,
                                                            mod_.getPRNG());
  bindings_.allocate(uint8In)->getHandle<uint8_t>().randomize(0, 255,
                                                              mod_.getPRNG());
  bindings_.allocate(int32IIn)->getHandle<int32_t>().randomize(-200, 300,
                                                               mod_.getPRNG());
  bindings_.allocate(int64IIn)->getHandle<int64_t>().randomize(-6455, 6555,
                                                               mod_.getPRNG());

  std::vector<NodeValue> inputs = {floatIn, float16In, int8In,
                                   uint8In, int32IIn,  int64IIn};

  auto opInfo = OpRepository::get()->getOperationInfo(typeName, packageName);
  assert(opInfo && "Could not retrieve OperationInfo.");

  // Parameter names passed to CustomOpData must be in the order that they were
  // registered.
  float floatP = 4.0f;
  int intP = 2;
  std::vector<float> floatVecP({0.5, 0.5, 0.5});
  std::vector<int> intVecP({7, 7, 7, 7, 7});
  CustomOpData dummyMetaData("scale1", typeName, packageName,
                             opInfo->getParamInfo());
  dummyMetaData.addParam("floatParam", floatP);
  dummyMetaData.addParam("intParam", intP);
  dummyMetaData.addParam("floatVecParam", floatVecP);
  dummyMetaData.addParam("intVecParam", intVecP);

  CustomOpNode *customOp = F_->createCustomOp("Copy1", typeName, packageName,
                                              inputs, types, dummyMetaData);

  SaveNode *S0 = F_->createSave("S0", customOp->getNthResult(0));
  SaveNode *S1 = F_->createSave("S1", customOp->getNthResult(1));
  SaveNode *S2 = F_->createSave("S2", customOp->getNthResult(2));
  SaveNode *S3 = F_->createSave("S3", customOp->getNthResult(3));
  SaveNode *S4 = F_->createSave("S4", customOp->getNthResult(4));
  SaveNode *S5 = F_->createSave("S5", customOp->getNthResult(5));
  bindings_.allocate(mod_.getPlaceholders());

  // Compile and run.
  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  // Check.
  size_t size = dims[0];

#define COMPARE_IO(in, out, type)                                              \
  {                                                                            \
    auto inH = bindings_.get(in)->getHandle<type>();                           \
    auto outH = bindings_.get(out->getPlaceholder())->getHandle<type>();       \
    for (dim_t z = 0; z < size; z++)                                           \
      EXPECT_EQ(outH.at({z}), inH.at({z}));                                    \
  }
  COMPARE_IO(floatIn, S0, float)
  COMPARE_IO(float16In, S1, float16_t)
  COMPARE_IO(int8In, S2, int8_t)
  COMPARE_IO(uint8In, S3, uint8_t)
  COMPARE_IO(int32IIn, S4, int32_t)
  COMPARE_IO(int64IIn, S5, int64_t)

#undef COMPARE_IO
}

TEST_P(CustomOpTest, CustomOpConvertLayout) {
  // Op info.
  std::string opName = "ConvertLayout";
  std::string opPackage = "OpPackage";
  std::string verificationLibrary{GLOW_DATA_PATH
                                  "tests/unittests/libCustomConvertLayout.so"};
  const std::string cppFileName{GLOW_DATA_PATH
                                "tests/unittests/customop_examples.cpp"};
  // Parameter info.
  ParamInfo inLayInfo("inLayout", CustomOpDataType::DTString,
                      true /* isScalar */);
  ParamInfo outLayInfo("outLayout", CustomOpDataType::DTString,
                       true /* isScalar */);
  std::vector<ParamInfo> paramInfo({inLayInfo, outLayInfo});

  // IO info..
  std::vector<NodeIOInfo> inputIOs({NodeIOInfo("input", 1)});
  std::vector<NodeIOInfo> outputIOs({NodeIOInfo("Out1", 1)});

  // Implementation info.
  std::vector<ImplementationInfo> implInfo;
  implInfo.push_back(ImplementationInfo("Interpreter", "customConvertLayout",
                                        (void *)(&verificationLibrary)));

  // Register operation in OpRepository.
  EXPECT_FALSE(registerCustomOp(opName, opPackage, paramInfo, inputIOs,
                                outputIOs, implInfo, verificationLibrary));

  // Create inputs and output types.
  int32_t inshape_NHWC[] = {16, 28, 14, 3};
  int32_t inshape_NCHW[] = {16, 3, 28, 14};

  std::vector<NodeValue> inputs;
  auto *I1 = mod_.createPlaceholder(ElemKind::Int32ITy, {4}, "I1",
                                    false /* isTrainable */);
  bindings_.allocate(I1)->getHandle<int32_t>() = inshape_NHWC;
  inputs.push_back(I1);

  std::vector<TypeRef> outputTypes;
  outputTypes.push_back(mod_.uniqueType(ElemKind::Int32ITy, {4}));

  // Populate custom op data. This is typically provided in the model.
  auto opInfo = OpRepository::get()->getOperationInfo(opName, opPackage);
  assert(opInfo && "Could not retrieve OperationInfo.");

  CustomOpData convertLayData("covertLayout1", opName, opPackage,
                              opInfo->getParamInfo());

  convertLayData.addParam("inLayout", "NHWC");
  convertLayData.addParam("outLayout", "NCHW");

  // Create graph.
  CustomOpNode *customOp = F_->createCustomOp(
      convertLayData.getName(), convertLayData.getTypeName(),
      convertLayData.getPackageName(), inputs, outputTypes, convertLayData);
  SaveNode *S0 = F_->createSave("S0", customOp->getNthResult(0));

  bindings_.allocate(S0->getPlaceholder());

  // Compile and run.
  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto result0 = bindings_.get(S0->getPlaceholder())->getHandle<int32_t>();

  for (int i = 0; i < 4; i++) {
    EXPECT_EQ(result0.at(i), inshape_NCHW[i]);
  }

  // Clear OpRepository.
  OpRepository::get()->clear();

} // CustomOpConvertLayoutTest

GLOW_INSTANTIATE_TEST_SUITE_P(Interpreter, CustomOpTest,
                              ::testing::Values("Interpreter"));
