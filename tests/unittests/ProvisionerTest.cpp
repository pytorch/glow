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
std::unique_ptr<Module> setupModule(unsigned functionCount) {
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

DAGListTy setupDAG(unsigned rootCount, unsigned childCount) {
  DAGListTy partitions;
  unsigned currentFunction = 0;
  for (unsigned int root = 0; root < rootCount; root++) {
    nodesDAGNodeTy nodes;
    auto rootNode = llvm::make_unique<DAGNode>();
    auto firstNode = llvm::make_unique<DAGNode>();
    rootNode->name = "root" + std::to_string(root);
    rootNode->children.push_back(firstNode.get());
    firstNode->name = "function" + std::to_string(currentFunction);
    firstNode->logicalDevices = {0};
    currentFunction++;
    for (unsigned int child = 0; child < childCount; child++) {
      auto newChild = llvm::make_unique<DAGNode>();
      newChild->name = "function" + std::to_string(currentFunction);
      newChild->logicalDevices = {0};
      currentFunction++;
      firstNode->children.push_back(newChild.get());
      nodes.push_back(std::move(newChild));
    }
    nodes.push_back(std::move(firstNode));
    partitions.push_back({std::move(rootNode), std::move(nodes)});
  }
  return partitions;
}

TEST_F(ProvisionerTest, provisionDag) {
  auto mod = setupModule(6);
  auto networks = setupDAG(2, 0);

  DeviceManagerMapTy devices;
  for (int i = 0; i < 6; i++) {
    std::unique_ptr<DeviceManager> device(new CPUDeviceManager);
    devices.emplace(i, std::move(device));
  }
  auto provisioner = Provisioner(devices);
  auto err = provisioner.provision(networks, *mod.get());
  // Expect that there was no Error when provisioning
  EXPECT_FALSE(errToBool(std::move(err)));
}
