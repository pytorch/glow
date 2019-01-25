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
using DAGNodePairTy = std::pair<std::vector<std::unique_ptr<DAGNode>>,
                                std::vector<std::unique_ptr<DAGNode>>>;

class ProvisionerTest : public ::testing::Test {};
std::unique_ptr<Module> setupModule(uint functionCount) {
  std::unique_ptr<Module> module = llvm::make_unique<Module>();
  for (unsigned int i = 0; i < functionCount; i++) {
    Function *F = module->createFunction("function" + std::to_string(i));
    auto *X = module->createPlaceholder(ElemKind::FloatTy, {3},
                                        "X" + std::to_string(i), false);
    auto *pow = F->createPow("Pow" + std::to_string(i), X, 2.0);
    F->createSave("save" + std::to_string(i), pow);
  }
  return module;
}

DAGNodePairTy setupDAG(uint rootCount, uint childCount) {
  std::vector<std::unique_ptr<DAGNode>> networks;
  std::vector<std::unique_ptr<DAGNode>> children;
  uint currentFunction = 0;
  for (unsigned int root = 0; root < rootCount; root++) {
    auto rootNode = llvm::make_unique<DAGNode>();
    auto firstNode = llvm::make_unique<DAGNode>();
    rootNode->name = "root" + std::to_string(root);
    rootNode->children.push_back(firstNode.get());
    firstNode->name = "function" + std::to_string(currentFunction);
    currentFunction++;
    for (unsigned int child = 0; child < childCount; child++) {
      auto newChild = llvm::make_unique<DAGNode>();
      newChild->name = "function" + std::to_string(currentFunction);
      currentFunction++;
      firstNode->children.push_back(newChild.get());
      children.push_back(std::move(newChild));
    }
    networks.push_back(std::move(rootNode));
    children.push_back(std::move(firstNode));
  }
  return std::make_pair(std::move(networks), std::move(children));
}

TEST_F(ProvisionerTest, provisionDag) {
  auto mod = setupModule(6);
  auto networks = setupDAG(2, 0);
  auto provisioner = Provisioner();
  std::map<DeviceIDTy, std::unique_ptr<DeviceManager>> devices;
  for (int i = 0; i < 6; i++) {
    std::unique_ptr<DeviceManager> device(new CPUDeviceManager);
    devices.emplace(i, std::move(device));
  }
  auto result = provisioner.provision(networks.first, devices, *mod.get());
  EXPECT_EQ(result, ResultCode::Ready);
}
