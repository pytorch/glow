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
#include "glow/Runtime/Provisioner/Provisioner.h"
#include "../../lib/Backends/CPU/CPUDeviceManager.h"
#include "glow/Optimizer/GraphOptimizer/GraphOptimizer.h"

#include "gtest/gtest.h"

using namespace glow;
using namespace glow::runtime;

class ProvisionerTest : public ::testing::Test {};

std::unique_ptr<Module> setupModule(unsigned functionCount) {
  auto mod = glow::make_unique<Module>();
  for (unsigned int i = 0; i < functionCount; i++) {
    auto *F = mod->createFunction("function" + std::to_string(i));
    auto *X = mod->createPlaceholder(ElemKind::FloatTy, {16, 1024}, "X", false);
    auto *W = mod->createConstant(ElemKind::FloatTy, {1024, 1024}, "W");
    auto *B = mod->createConstant(ElemKind::FloatTy, {1024}, "B");
    auto *FC = F->createFullyConnected("FC", X, W, B);
    F->createSave("save", FC);
    CompilationContext cctx;
    lower(F, cctx);
  }
  return mod;
}

DAGListTy setupDAG(unsigned rootCount, unsigned childCount) {
  DAGListTy partitions;
  unsigned currentFunction = 0;
  for (unsigned int root = 0; root < rootCount; root++) {
    DAGNodePtrVec nodes;
    auto rootNode = glow::make_unique<DAGNode>();
    auto firstNode = glow::make_unique<DAGNode>();
    rootNode->name = "root" + std::to_string(root);
    rootNode->children.push_back(firstNode.get());
    firstNode->name = "function" + std::to_string(currentFunction);
    firstNode->logicalDevices = {0, 1};
    firstNode->backendName = "CPU";
    currentFunction++;
    for (unsigned int child = 0; child < childCount; child++) {
      auto newChild = glow::make_unique<DAGNode>();
      newChild->name = "function" + std::to_string(currentFunction);
      newChild->logicalDevices = {0};
      newChild->backendName = "CPU";
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
    std::unique_ptr<DeviceManager> device(
        new CPUDeviceManager(DeviceConfig("CPU")));
    devices.emplace(i, std::move(device));
  }

  CompilationContext cctx;
  Provisioner provisioner(devices);
  auto err = provisioner.provision(networks, *mod.get(), cctx);
  // Expect that there was no Error when provisioning
  EXPECT_FALSE(ERR_TO_BOOL(std::move(err)));
}

TEST_F(ProvisionerTest, provisionDagFail) {
  auto mod = setupModule(6);
  auto networks = setupDAG(2, 0);

  DeviceManagerMapTy devices;
  for (int i = 0; i < 6; i++) {
    auto config = DeviceConfig("CPU");
    config.setDeviceMemory(1000);
    std::unique_ptr<DeviceManager> device(new CPUDeviceManager(config));
    devices.emplace(i, std::move(device));
  }

  CompilationContext cctx;
  Provisioner provisioner(devices);
  auto err = provisioner.provision(networks, *mod.get(), cctx);
  // Expect that there was an Error when provisioning
  EXPECT_TRUE(ERR_TO_BOOL(std::move(err)));
}

TEST_F(ProvisionerTest, provisionFailCleanup) {
  // We want this provisioning to fail after adding the first partition
  // successfully. This is to test that cleanup properly evicts networks.
  DeviceConfig configBig("CPU");
  DeviceConfig configSmall("CPU");
  configSmall.setDeviceMemory(1);
  std::unique_ptr<DeviceManager> deviceBig(new CPUDeviceManager(configBig));
  std::unique_ptr<DeviceManager> deviceSmall(new CPUDeviceManager(configSmall));
  DeviceManagerMapTy devices;
  devices.emplace(0, std::move(deviceBig));
  devices.emplace(1, std::move(deviceSmall));

  auto mod = setupModule(2);
  auto networks = setupDAG(2, 0);
  CompilationContext cctx;
  Provisioner provisioner(devices);
  auto err = provisioner.provision(networks, *mod.get(), cctx);
  // Expect that there was an Error when provisioning
  EXPECT_TRUE(ERR_TO_BOOL(std::move(err)));
}
