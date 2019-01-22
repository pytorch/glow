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
#include "glow/Runtime/Provisioner/Provisioner.h"
#include "CPUDeviceManager.h"

#include "gtest/gtest.h"

using namespace glow;
using namespace glow::runtime;

class ProvisionerTest : public ::testing::Test {};
std::unique_ptr<Module> setupModule(uint functionCount) {
  std::unique_ptr<Module> module = std::make_unique<Module>();
  for (int i = 0; i < functionCount; i++) {
    Function *F = module->createFunction("function" + std::to_string(i));
    auto *X = module->createPlaceholder(ElemKind::FloatTy, {3},
                                        "X" + std::to_string(i), false);
    auto *pow = F->createPow("Pow1", X, 2.0);
    F->createSave("save", pow);
  }
  return module;
}

std::vector<DAGNode> setupDAG(Module *module, uint familyCount,
                              uint childCount) {
  std::vector<DAGNode> networks;
  uint currentFunction = 0;
  for (int family = 0; family < familyCount; family++) {
    auto familyNode = DAGNode();
    auto firstNode = new DAGNode();
    familyNode.name = "family" + std::to_string(family);
    familyNode.children.push_back(firstNode);
    firstNode->name = "function" + std::to_string(currentFunction);
    currentFunction++;
    for (int child = 0; child < childCount; child++) {
      auto newChild = new DAGNode();
      newChild->name = "function" + std::to_string(currentFunction);
      currentFunction++;
    }
    networks.push_back(familyNode);
  }
  return networks;
}

TEST_F(ProvisionerTest, provisionDag) {
  auto mod = setupModule(6);
  auto networks = setupDAG(mod.get(), 2, 3);
  auto provisioner = Provisioner();
  std::map<DeviceIDTy, std::unique_ptr<DeviceManager>> devices;
  for (int i = 0; i < 6; i++) {
    std::unique_ptr<DeviceManager> device(new CPUDeviceManager);
    devices.emplace(i, std::move(device));
  }
  auto result = provisioner.provision(networks, devices, *mod.get());
  EXPECT_EQ(result, Ready);
}