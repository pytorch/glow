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

#include "gtest/gtest.h"

#include "glow/Backend/Backend.h"
#include "glow/Backend/BackendOpRepository.h"
#include "glow/Graph/OpRepository.h"

#include "BackendTestUtils.h"

#ifndef GLOW_DATA_PATH
#define GLOW_DATA_PATH
#endif

namespace glow {

// GoogleTest Fixture class for OpRepository.
// The Fixture implements a backendOpRepository for MockBackend.
// The tests using this Fixture can register and unregister MockBackend and
// MockBackendOpRepository on-demand.
class OpRepositoryTest : public ::testing::Test {
protected:
  const std::string mockBackendString = "MockBackend";

  /// Implement a BackendOpRepository for MockBackend.
  class MockBackendOpRepository : public BackendOpRepository {

  private:
    using OpKey = std::tuple<std::string, std::string>;
    OpKey makeOpKey(llvm::StringRef opTypeName,
                    llvm::StringRef packageName) const {
      return std::make_tuple(opTypeName, packageName);
    }

    class OpImplsInfo {
    public:
      std::map<std::string, void *> implMap_{};
      customOpSelectImpl_t select_{};
    };
    std::map<OpKey, OpImplsInfo> repo_;

  public:
    MockBackendOpRepository() = default;
    ~MockBackendOpRepository() { clear(); };

    // Get BackendOpRepository Name.
    std::string getBackendOpRepositoryName() const override {
      return Named::getName().empty() ? getName() : Named::getName().str();
    }

    // Get BackendOpRepository Name. Must be the same as Backend Name.
    static std::string getName() { return "MockBackend"; }

    // Store Implementations and the selection function.
    Error registerImplementations(llvm::StringRef opTypeName,
                                  llvm::StringRef opPackageName,
                                  llvm::ArrayRef<ImplementationInfo> impls,
                                  customOpSelectImpl_t implSelFunc) override {

      OpImplsInfo opImpls;
      opImpls.select_ = implSelFunc;
      for (const auto &impl : impls) {
        RETURN_ERR_IF_NOT(impl.getImplementation() != nullptr,
                          "Implementation cannot be nullptr.");
        opImpls.implMap_.insert({impl.getType(), impl.getImplementation()});
      }

      OpKey opkey = makeOpKey(opTypeName, opPackageName);
      repo_.insert({opkey, opImpls});
      return Error::success();
    }

    void clear() override { repo_.clear(); }

  }; // class MockBackendOpRepository

  // The Backend and BackendOpRepo are normally registered via the macro.
  // Breaking the macro in two part for the test: 1. creating a class Factory
  // 2. instead of static object, register only when the register-function
  // is called. This way the lifetime of these class is limited to this test
  // fixture and not throughout the process (macro creates a static object).

  // REGISTER_GLOW_BACKEND_FACTORY(MockBackendFactory, MockBackend);
  class MockBackendFactory : public BaseFactory<std::string, Backend> {
  public:
    Backend *create() override { return new MockBackend(); }
    std::string getRegistrationKey() const override {
      return MockBackend::getName();
    }
    unsigned numDevices() const override { return MockBackend::numDevices(); }
  }; // class MockBackendFactory

  // REGISTER_GLOW_BACKEND_OP_REPOSITORY_FACTORY(MockBackendOpRepoFactory,MockBackendOpRepository)
  class MockBackendOpRepoFactory
      : public BaseFactory<std::string, BackendOpRepository> {
  public:
    BackendOpRepository *create() override {
      if (backendOpRepo_ == nullptr) {
        backendOpRepo_ = std::make_shared<MockBackendOpRepository>();
      }
      return backendOpRepo_.get();
    }
    std::string getRegistrationKey() const override {
      return MockBackendOpRepository::getName();
    }
    /*Not valid for BackendOpRepository */
    unsigned numDevices() const override { return 0; }

  private:
    std::shared_ptr<MockBackendOpRepository> backendOpRepo_;
  }; // class MockBackendOpRepoFactory

  OpRepositoryTest(){};

  ~OpRepositoryTest() override{};

  std::string getMockBackendName() const { return mockBackendString; }

  // Clean up.
  void TearDown() override {
    unregisterMockBackend();
    unregisterMockBackendOpRepository();
    OpRepository::get()->clear();
  }

  std::unique_ptr<RegisterFactory<std::string, MockBackendFactory, Backend>>
      backendFactoryPtr;
  std::unique_ptr<RegisterFactory<std::string, MockBackendOpRepoFactory,
                                  BackendOpRepository>>
      backendOpRepoFactoryPtr;

  // Utility for registering Mock Backend.
  void registerMockBackend() {
    backendFactoryPtr.reset(
        new RegisterFactory<std::string, MockBackendFactory, Backend>());
  };

  void unregisterMockBackend() { backendFactoryPtr.reset(); }

  // Utility for registering Mock BackendOpRepository.
  void registerMockBackendOpRepository() {
    backendOpRepoFactoryPtr.reset(
        new RegisterFactory<std::string, MockBackendOpRepoFactory,
                            BackendOpRepository>());
  };

  void unregisterMockBackendOpRepository() { backendOpRepoFactoryPtr.reset(); }

  // Other utilities.
  bool isBackendRegistered(std::string name) {
    std::vector<std::string> names = getAvailableBackends();
    return find(names.begin(), names.end(), name) != names.end();
  }

  bool isBackendOpRepositoryRegistered(std::string name) {
    std::vector<std::string> names = getAvailableBackendOpRepositories();
    return find(names.begin(), names.end(), name) != names.end();
  }

}; // OpRepositoryTest GTest Fixture

// Test utility functions for BackendOpRepository.
// getAvailableBackendOpRepositories
// getBackendOpRepository(name)
TEST_F(OpRepositoryTest, UtilityFunctions) {
  std::string mockBackendName = getMockBackendName();

  // Assert the Mock backend and backendOpRepo is not available.
  ASSERT_FALSE(isBackendRegistered(mockBackendName));
  ASSERT_FALSE(isBackendOpRepositoryRegistered(mockBackendName));
  ASSERT_EQ(nullptr, getBackendOpRepository(mockBackendName));

  // Register Mock Backend and BackendOpRepo.
  registerMockBackend();
  registerMockBackendOpRepository();

  // Assert the Mock backend and backendOpRepo are available.
  ASSERT_TRUE(isBackendRegistered(mockBackendName));
  ASSERT_TRUE(isBackendOpRepositoryRegistered(mockBackendName));
  ASSERT_NE(nullptr, getBackendOpRepository(mockBackendName));
}

// Test OpRepository APIs.
// TODO: Define ImplementationInfo with a generic identifier string
// rather than name of ElementKind.
TEST_F(OpRepositoryTest, RegisterOp) {
  // Register MOCK Backend and Backend Op Repository.
  std::string mockBackendName = getMockBackendName();
  registerMockBackend();
  registerMockBackendOpRepository();

  std::vector<ParamInfo> parameters;
  std::vector<NodeIOInfo> inputs, outputs;
  std::string dummyImpl = "dummy";
  ImplementationInfo implTest(mockBackendName, "M1", &dummyImpl);
  std::vector<ImplementationInfo> impls = {implTest};

  std::string functionLibrary{};
  OperationInfo customReluX("ReluX", "Test", parameters, inputs, outputs, impls,
                            functionLibrary);

  OpRepository *opRepo = OpRepository::get();

  // Returns Error. Invalid Verification Library.
  ASSERT_TRUE(ERR_TO_BOOL(opRepo->registerOperation(customReluX)));

  functionLibrary = GLOW_DATA_PATH "tests/unittests/libCustomOpFunctions.so";
  OperationInfo customRelu("Relu", "Test", parameters, inputs, outputs, impls,
                           functionLibrary);

  // No Error. Registered successfully.
  ASSERT_FALSE(ERR_TO_BOOL(opRepo->registerOperation(customRelu)));
  // Returns Error. Double registeration.
  ASSERT_TRUE(ERR_TO_BOOL(opRepo->registerOperation(customRelu)));

  OperationInfo customX("X", "Test", parameters, inputs, outputs, impls,
                        functionLibrary);

  // customX is not registered.
  ASSERT_FALSE(opRepo->isOpRegistered("X", "Test"));
  // No Error. Registered successfully.
  ASSERT_FALSE(ERR_TO_BOOL(opRepo->registerOperation(customX)));
  // Now returns true.
  ASSERT_TRUE(opRepo->isOpRegistered("X", "Test"));
  // Returns false.
  ASSERT_FALSE(opRepo->isOpRegistered("Test", "X"));

  // Returns ptr to the correct OpInfo.
  auto opinfo1 = opRepo->getOperationInfo("X", "Test");
  ASSERT_NE(opinfo1, nullptr);
  ASSERT_EQ(opinfo1->getTypeName(), "X");
  ASSERT_EQ(opinfo1->getPackageName(), "Test");
  ASSERT_EQ(opinfo1->getParamInfo().size(), 0);
  ASSERT_EQ(opinfo1->getInputInfo().size(), 0);
  ASSERT_EQ(opinfo1->getOutputInfo().size(), 0);

  // Returns ptr to the correct OpInfo.
  auto opinfo2 = opRepo->getOperationInfo("Relu", "Test");
  ASSERT_NE(opinfo2, nullptr);
  ASSERT_EQ(opinfo2->getTypeName(), "Relu");
  ASSERT_EQ(opinfo2->getPackageName(), "Test");
  ASSERT_EQ(opinfo2->getParamInfo().size(), 0);
  ASSERT_EQ(opinfo2->getInputInfo().size(), 0);
  ASSERT_EQ(opinfo2->getOutputInfo().size(), 0);

  // Returns nullptr, no such OpInfo registered.
  auto opinfo3 = opRepo->getOperationInfo("TEST", "X");
  ASSERT_EQ(opinfo3, nullptr);
}

// Test OpInfoRegisteration wrt the registered backends and
// backendOpRepositories.
TEST_F(OpRepositoryTest, RegisterImpl) {
  std::string mockBackendName = getMockBackendName();
  std::vector<ParamInfo> parameters;
  std::vector<NodeIOInfo> inputs, outputs;
  std::vector<ImplementationInfo> impls;
  std::string functionLibrary{GLOW_DATA_PATH
                              "tests/unittests/libCustomOpFunctions.so"};

  ImplementationInfo implTest1(mockBackendName, "M1", nullptr);
  impls.push_back(implTest1);
  OperationInfo customRelu("Relu1", "Test", parameters, inputs, outputs, impls,
                           functionLibrary);

  OpRepository *opRepo = OpRepository::get();

  // Returns Error. No Backend "MockBackend" supported.
  ASSERT_TRUE(ERR_TO_BOOL(opRepo->registerOperation(customRelu)));

  // Register Mock Backend.
  registerMockBackend();

  // Returns Error. No BackendOpRepository registered for "Mockbackend".
  ASSERT_TRUE(ERR_TO_BOOL(opRepo->registerOperation(customRelu)));

  // Register Mock Backend Op Repository.
  registerMockBackendOpRepository();

  // Returns Error. Mock BackendOpRepository does not register implementation
  // that is nullptr.
  ASSERT_TRUE(ERR_TO_BOOL(opRepo->registerOperation(customRelu)));

  std::string dummyImpl = "DummyImpl";
  ImplementationInfo implTest2(mockBackendName, "M2", &dummyImpl);
  impls.clear();
  impls.push_back(implTest2);
  OperationInfo customX("X2", "Test", parameters, inputs, outputs, impls,
                        functionLibrary);

  // Registers successfully.
  ASSERT_FALSE(ERR_TO_BOOL(opRepo->registerOperation(customX)));
}

static std::string castVoidPtrToString(void *ptr) {
  return *(std::string *)(ptr);
}

TEST(OpInfoParserTest, ParseOpInfo) {
  std::string configFile(GLOW_DATA_PATH "tests/CustomOpConfig.yaml");

  std::vector<OperationInfo> opinfos;
  ASSERT_FALSE(ERR_TO_BOOL(deserializeOpInfoFromYaml(configFile, opinfos)));
  ASSERT_EQ(opinfos.size(), 2);

  // Validate op 1
  auto opInfo = opinfos[0];
  ASSERT_EQ(opInfo.getTypeName(), "X");
  ASSERT_EQ(opInfo.getPackageName(), "Example");

  auto params = opInfo.getParamInfo();
  ASSERT_EQ(params.size(), 2);
  ASSERT_EQ(params[0].getName(), "paramX1");
  ASSERT_EQ(params[0].getDataType(), CustomOpDataType::DTBool);
  ASSERT_TRUE(params[0].isScalar());
  ASSERT_FALSE(params[0].isArray());
  ASSERT_EQ(params[0].getSize(), 0);

  ASSERT_EQ(params[1].getName(), "paramX2");
  ASSERT_EQ(params[1].getDataType(), CustomOpDataType::DTIInt32);
  ASSERT_FALSE(params[1].isScalar());
  ASSERT_TRUE(params[1].isArray());
  ASSERT_EQ(params[1].getSize(), 5);

  auto inputs = opInfo.getInputInfo();
  ASSERT_EQ(inputs.size(), 2);
  ASSERT_EQ(inputs[0].getName(), "inputX1");
  ASSERT_EQ(inputs[0].getMaxDims(), 1);
  ASSERT_FALSE(inputs[0].isConstant());
  ASSERT_EQ(inputs[1].getName(), "inputX2");
  ASSERT_EQ(inputs[1].getMaxDims(), 5);
  ASSERT_TRUE(inputs[1].isConstant());

  auto outputs = opInfo.getOutputInfo();
  ASSERT_EQ(outputs.size(), 1);
  ASSERT_EQ(outputs[0].getName(), "outputX1");
  ASSERT_EQ(outputs[0].getMaxDims(), 1);
  ASSERT_FALSE(outputs[0].isConstant());

  ASSERT_EQ(opInfo.getFunctionLibraryPath(), "/tmp/customop_functions.so");

  auto impls = opInfo.getImplementations();
  ASSERT_EQ(impls.size(), 3);

  ASSERT_EQ(impls[0].getBackendName(), "Interpreter");
  ASSERT_EQ(impls[0].getType(), "float");
  ASSERT_EQ(impls[0].getConfig(), "");
  ASSERT_EQ(castVoidPtrToString(impls[0].getImplementation()), "X_I1.so");

  ASSERT_EQ(impls[1].getBackendName(), "AIC");
  ASSERT_EQ(impls[1].getType(), "i32");
  ASSERT_EQ(impls[1].getConfig(), "ConfigAIC");
  ASSERT_EQ(castVoidPtrToString(impls[1].getImplementation()), "X_AIC1.cpp");

  ASSERT_EQ(impls[2].getBackendName(), "Interpreter");
  ASSERT_EQ(impls[2].getType(), "bool");
  ASSERT_EQ(impls[2].getConfig(), "");
  ASSERT_EQ(castVoidPtrToString(impls[2].getImplementation()), "X_I2.so");

  // Validate op2
  opInfo = opinfos[1];
  ASSERT_EQ(opInfo.getTypeName(), "Y");
  ASSERT_EQ(opInfo.getPackageName(), "Example");
  params = opInfo.getParamInfo();
  ASSERT_EQ(params.size(), 1);
  ASSERT_EQ(params[0].getName(), "alphaY");
  ASSERT_EQ(params[0].getDataType(), CustomOpDataType::DTFloat32);
  ASSERT_TRUE(params[0].isScalar());
  ASSERT_FALSE(params[0].isArray());
  ASSERT_EQ(params[0].getSize(), 0);

  inputs = opInfo.getInputInfo();
  ASSERT_EQ(inputs.size(), 1);
  ASSERT_EQ(inputs[0].getName(), "inputY");
  ASSERT_EQ(inputs[0].getMaxDims(), 1);
  ASSERT_FALSE(inputs[0].isConstant());

  outputs = opInfo.getOutputInfo();
  ASSERT_EQ(outputs.size(), 2);
  ASSERT_EQ(outputs[0].getName(), "outputY1");
  ASSERT_EQ(outputs[0].getMaxDims(), 1);
  ASSERT_FALSE(outputs[0].isConstant());
  ASSERT_EQ(outputs[1].getName(), "outputY2");
  ASSERT_EQ(outputs[1].getMaxDims(), 5);
  ASSERT_FALSE(outputs[1].isConstant());

  impls = opInfo.getImplementations();
  ASSERT_EQ(impls.size(), 2);
  ASSERT_EQ(impls[0].getBackendName(), "Interpreter");
  ASSERT_EQ(impls[0].getType(), "float");
  ASSERT_EQ(impls[0].getConfig(), "ConfigINTP");
  ASSERT_EQ(castVoidPtrToString(impls[0].getImplementation()), "Y_I1.so");

  ASSERT_EQ(impls[1].getBackendName(), "AIC");
  ASSERT_EQ(impls[1].getType(), "i32");
  ASSERT_EQ(impls[1].getConfig(), "");
  ASSERT_EQ(castVoidPtrToString(impls[1].getImplementation()), "Y_AIC1.cpp");

  ASSERT_EQ(opInfo.getFunctionLibraryPath(), "/tmp/customop_functions.so");
}

TEST_F(OpRepositoryTest, RegisterOpConfigFile) {
  // Register MOCK Backend and Backend Op Repository.
  std::string mockBackendName = getMockBackendName();
  registerMockBackend();
  registerMockBackendOpRepository();

  std::string configFile{GLOW_DATA_PATH "tests/CustomOpReluMock.yaml"};

  OpRepository *opRepo = OpRepository::get();
  ASSERT_FALSE(ERR_TO_BOOL(opRepo->registerOperation(configFile)));
  ASSERT_TRUE(opRepo->isOpRegistered("CustomReluOp", "Example"));

  // Test "CustomReluOp" parsing.
  // CustomOpConfig.yaml's parsing has been tested in
  // "OpInfoParserTest.ParseOpInfo".
  auto opInfo = opRepo->getOperationInfo("CustomReluOp", "Example");
  ASSERT_EQ(opInfo->getTypeName(), "CustomReluOp");
  ASSERT_EQ(opInfo->getPackageName(), "Example");
  auto params = opInfo->getParamInfo();
  ASSERT_EQ(params.size(), 2);
  ASSERT_EQ(params[0].getName(), "alpha");
  ASSERT_EQ(params[0].getDataType(), CustomOpDataType::DTFloat32);
  ASSERT_EQ(params[1].getName(), "beta");
  ASSERT_FALSE(params[1].isScalar());

  auto inputs = opInfo->getInputInfo();
  ASSERT_EQ(inputs.size(), 1);
  ASSERT_EQ(inputs[0].getName(), "input");

  auto outputs = opInfo->getOutputInfo();
  ASSERT_EQ(outputs.size(), 1);
  ASSERT_EQ(outputs[0].getName(), "reluOut");

  auto impls = opInfo->getImplementations();
  ASSERT_EQ(impls.size(), 2);
  ASSERT_EQ(impls[0].getBackendName(), "MockBackend");
  ASSERT_EQ(castVoidPtrToString(impls[0].getImplementation()),
            "mock/custom_relu.so");

  ASSERT_EQ(impls[1].getBackendName(), "MockBackend");
  ASSERT_EQ(impls[1].getType(), "i32");
  ASSERT_EQ(castVoidPtrToString(impls[1].getImplementation()),
            "mock/libjit_custom.cpp");

  ASSERT_EQ(opInfo->getFunctionLibraryPath(),
            "tests/unittests/libCustomOpFunctions.so");
}

} // namespace glow
