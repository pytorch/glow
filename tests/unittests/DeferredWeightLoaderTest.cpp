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

#include "gtest/gtest.h"

using namespace glow;
using namespace glow::runtime;

class TestDeferredWeightLoader : public DeferredWeightLoader {
public:
  Error loadNextWeight() override {
    position_++;
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
  EXPECT_NEAR(resHandle.at({0}), 256.0, 1E-5);
}

INSTANTIATE_BACKEND_TEST(DeferredWeightLoaderTest);
