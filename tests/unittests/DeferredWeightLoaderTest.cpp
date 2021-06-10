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

#include "glow/Runtime/DeferredWeightLoader.h"
#include "BackendTestUtils.h"
#include "glow/ExecutionContext/ExecutionContext.h"
#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Runtime/Provisioner/Provisioner.h"

#include "gtest/gtest.h"

using namespace glow;
using namespace glow::runtime;

class TestDeferredWeightLoader : public DeferredWeightLoader {
public:
  Error loadNextWeight() override {
    position_++;
    if (position_ < names_.size() && names_[position_] == "fail") {
      return MAKE_ERR(ErrorValue::ErrorCode::RUNTIME_DEFERRED_WEIGHT_ERROR,
                      "Fail to load weight.");
    }
    return Error::success();
  }
  Error setSrc(void *loaderObject) override { return Error::success(); }
  void addWeight(Tensor *weight) { weights_.push_back(weight); }
  void addName(std::string name) { names_.push_back(name); }
  void setTypeInfo(std::map<std::string, Type> info) override {}

  std::string getName() override {
    for (auto na : names_) {
    }
    if (position_ >= int(names_.size())) {
      return "";
    }
    return names_[position_];
  }

  Tensor *getTensor() override {
    if (position_ >= int(weights_.size())) {
      return nullptr;
    }
    return weights_[position_];
  }

private:
  std::vector<Tensor *> weights_{};
  std::vector<std::string> names_{};
  int position_{-1};
};

class DeferredWeightLoaderTest : public ::testing::TestWithParam<std::string> {
};

std::unique_ptr<HostManager>
createHostManager(llvm::StringRef backendName,
                  HostConfig hostConfig = HostConfig()) {
  std::vector<std::unique_ptr<DeviceConfig>> configs;
  auto deviceConfig = glow::make_unique<DeviceConfig>(backendName);
  configs.push_back(std::move(deviceConfig));
  std::unique_ptr<HostManager> hostManager =
      glow::make_unique<HostManager>(std::move(configs), hostConfig);
  return hostManager;
}

TEST_P(DeferredWeightLoaderTest, cleanupFailedDeferred) {
  // We want this provisioning to fail after loading a deferred weight, then
  // verify that the network is cleaned up properly.
  CHECK_IF_ENABLED();
  std::unique_ptr<Module> module = glow::make_unique<Module>();
  auto F = module->createFunction("main");
  auto *X = module->createPlaceholder(ElemKind::FloatTy, {1}, "X", false);

  auto *Y = module->createPlaceholder(ElemKind::FloatTy, {1}, "Y", false);
  auto *Z = module->createPlaceholder(ElemKind::FloatTy, {1}, "Z", false);
  auto *output =
      module->createPlaceholder(ElemKind::FloatTy, {1}, "output", false);
  // Set X and Y as static.
  X->setStatic(true);
  Y->setStatic(true);
  auto pow1 = F->createPow("pow", X, Y);
  auto pow2 = F->createPow("pow2", Z, pow1);
  F->createSave("save", pow2, output);
  std::vector<Tensor> staticInputs;
  auto xTensor = Tensor(X->getType());
  auto yTensor = Tensor(Y->getType());
  auto zTensor = Tensor(Z->getType());
  xTensor.getHandle().clear(2.0);
  yTensor.getHandle().clear(3.0);
  zTensor.getHandle().clear(2.0);

  TestDeferredWeightLoader loader;
  loader.addWeight(&xTensor);
  loader.addWeight(&yTensor);
  loader.addName("fail");
  loader.addName("fail");
  DeferredLoader()->registerLoader(&loader);

  CompilationContext cctx;
  cctx.deferredWeightLoader = &loader;
  cctx.optimizationOpts.foldStaticPlaceholderConversions = true;

  DeviceConfig config(GetParam());
  std::unique_ptr<DeviceManager> device(
      DeviceManager::createDeviceManager(config));
  EXPECT_FALSE(ERR_TO_BOOL(device->init()));

  DeviceManagerMapTy devices;
  devices.emplace(0, std::move(device));

  DAGListTy partitions;

  DAGNodePtrVec nodes;
  auto rootNode = glow::make_unique<DAGNode>();
  auto firstNode = glow::make_unique<DAGNode>();
  rootNode->name = "root";
  rootNode->children.push_back(firstNode.get());
  firstNode->name = "main";
  firstNode->logicalDevices = {0};
  firstNode->backendName = GetParam();
  nodes.push_back(std::move(firstNode));
  partitions.push_back({std::move(rootNode), std::move(nodes)});

  Provisioner provisioner(devices);
  auto err = provisioner.provision(partitions, *module.get(), cctx);
  // Expect that there was an Error when provisioning
  EXPECT_TRUE(ERR_TO_BOOL(std::move(err)));

  // Setup a new loader with correct info.
  TestDeferredWeightLoader loaderNew;
  loaderNew.addWeight(&xTensor);
  loaderNew.addWeight(&yTensor);
  loaderNew.addName("X");
  loaderNew.addName("Y");
  DeferredLoader()->registerLoader(&loaderNew);
  cctx.deferredWeightLoader = &loaderNew;
  auto err2 = provisioner.provision(partitions, *module.get(), cctx);
  // Verify provisioning completes correctly.
  EXPECT_FALSE(ERR_TO_BOOL(std::move(err2)));
}

TEST_P(DeferredWeightLoaderTest, staticPlaceholderInference) {
  CHECK_IF_ENABLED();
  auto hostmanager = createHostManager(GetParam());
  ExecutionEngine EE{GetParam()};
  auto &module = EE.getModule();
  auto F = module.createFunction("main");
  auto *X = module.createPlaceholder(ElemKind::FloatTy, {1}, "X", false);

  auto *Y = module.createPlaceholder(ElemKind::FloatTy, {1}, "Y", false);
  auto *Z = module.createPlaceholder(ElemKind::FloatTy, {1}, "Z", false);
  auto *output =
      module.createPlaceholder(ElemKind::FloatTy, {1}, "output", false);
  // Set X and Y as static.
  X->setStatic(true);
  Y->setStatic(true);
  auto pow1 = F->createPow("pow", X, Y);
  auto pow2 = F->createPow("pow2", Z, pow1);
  F->createSave("save", pow2, output);
  std::vector<Tensor> staticInputs;
  auto xTensor = Tensor(X->getType());
  auto yTensor = Tensor(Y->getType());
  auto zTensor = Tensor(Z->getType());
  xTensor.getHandle().clear(2.0);
  yTensor.getHandle().clear(3.0);
  zTensor.getHandle().clear(2.0);

  TestDeferredWeightLoader loader;
  loader.addWeight(&xTensor);
  loader.addWeight(&yTensor);
  loader.addName("X");
  loader.addName("Y");
  DeferredLoader()->registerLoader(&loader);

  CompilationContext cctx;
  cctx.deferredWeightLoader = &loader;
  cctx.optimizationOpts.foldStaticPlaceholderConversions = true;
  EE.compile(cctx);
  PlaceholderBindings pBindings;
  pBindings.allocate(Z);
  pBindings.allocate(output);
  updateInputPlaceholders(pBindings, {Z}, {&zTensor});
  EE.run(pBindings);
  auto resHandle = pBindings.get(output)->getHandle();
  EXPECT_NEAR(resHandle.at({0}), 256.0, 1E-5);
}

TEST_P(DeferredWeightLoaderTest, FP16StaticPlaceholderInference) {
  CHECK_IF_ENABLED();
  auto hostmanager = createHostManager(GetParam());
  ExecutionEngine EE{GetParam()};
  auto &module = EE.getModule();
  auto F = module.createFunction("main");
  auto *X = module.createPlaceholder(ElemKind::FloatTy, {1}, "X", false);

  auto *Y = module.createPlaceholder(ElemKind::FloatTy, {1}, "Y", false);
  auto *Z = module.createPlaceholder(ElemKind::FloatTy, {1}, "Z", false);
  auto *output =
      module.createPlaceholder(ElemKind::FloatTy, {1}, "output", false);
  // Set X and Y as static.
  X->setStatic(true);
  Y->setStatic(true);
  auto mul1 = F->createMul("mul", X, Y);
  auto mul2 = F->createMul("mul2", Z, mul1);
  F->createSave("save", mul2, output);
  std::vector<Tensor> staticInputs;
  auto xTensor = Tensor(X->getType());
  auto yTensor = Tensor(Y->getType());
  auto zTensor = Tensor(Z->getType());
  xTensor.getHandle().clear(2.0);
  yTensor.getHandle().clear(3.0);
  zTensor.getHandle().clear(2.0);

  TestDeferredWeightLoader loader;
  loader.addWeight(&xTensor);
  loader.addWeight(&yTensor);
  loader.addName("X");
  loader.addName("Y");
  DeferredLoader()->registerLoader(&loader);

  PlaceholderBindings pBindings;

  CompilationContext cctx;
  cctx.deferredWeightLoader = &loader;
  cctx.optimizationOpts.foldStaticPlaceholderConversions = true;
  cctx.precisionConfig.convertToFP16 = true;

  EE.compile(cctx);

  pBindings.allocate(Z);
  pBindings.allocate(output);
  updateInputPlaceholders(pBindings, {Z}, {&zTensor});
  EE.run(pBindings);
  auto resHandle = pBindings.get(output)->getHandle();
  EXPECT_NEAR(resHandle.at({0}), 12.0, 1E-5);
}

INSTANTIATE_BACKEND_TEST(DeferredWeightLoaderTest);
