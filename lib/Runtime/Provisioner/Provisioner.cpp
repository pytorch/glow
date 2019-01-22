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
#include "glow/Backends/BackendUtils.h"
#include "glow/Backends/CompiledFunction.h"
#include "glow/Graph/Graph.h"

#include <future>
#include <queue>

using namespace glow;
using namespace runtime;
using DeviceID = unsigned int;
ResultCode Provisioner::provision(
    std::vector<DAGNode> &networks,
    std::map<DeviceIDTy, std::unique_ptr<DeviceManager>> &devices,
    Module &module) {
  // For the first pass we will just assign and load devices in order and update
  // the deviceID field of the node.
  std::queue<std::vector<DAGNode *>> nextNode;
  // Process head node, this does not contain a function but serves as an entry
  // point for the network. We build a vector of nodes, containing all family
  // members of a sub-function.
  for (int i = 0; i < networks[0].children.size(); i++) {
    std::vector<DAGNode *> newSet;
    for (auto node : networks) {
      newSet.push_back(node.children[i]);
    }
    nextNode.push(newSet);
  }
  while (!nextNode.empty()) {
    std::map<std::string, CompiledFunction *> compiledFunctions;
    auto nodes = nextNode.front();
    nextNode.pop();
    // Add child nodes to the queue.
    for (int i = 0; i < nodes[0]->children.size(); i++) {
      std::vector<DAGNode *> newSet;
      for (auto node : nodes) {
        newSet.push_back(node->children[i]);
      }
      nextNode.push(newSet);
    }
    // Assign collection of nodes to a device, compile and load the device.
    // We will do a round robin assignment of nodes. If there is not space we
    // will return an error.
    // TODO Add ability to try against another device when currDevice has
    // insufficient space.
    auto currDevice = devices.begin();
    // Set backend to match the device.
    backend_.reset(createBackend(currDevice->second->getBackendKind()));
    // Iterate over the nodes, compile them and add them to compiledFunctions.
    for (auto node : nodes) {
      node->deviceID = currDevice->first;
      Function *function = module.getFunction(node->name);
      auto compiled = backend_->compile(function);
      node->runtimeBundle = compiled->getRuntimeBundle();
      compiledFunctions.emplace(node->name, compiled.get());
    }
    // Check if sufficient space on device. Currently requiring a 10% buffer
    // over the size of constants.
    auto availableMemory = currDevice->second->getAvailableMemory();
    if (availableMemory < NETWORK_PADDING_FACTOR *
                              nodes[0]->runtimeBundle.getConstantWeightSize()) {
      return ResultCode::Failed;
    }
    // Load functions on device.
    std::promise<bool> addNetwork;
    auto ready = addNetwork.get_future();
    currDevice->second->addNetwork(
        &module, compiledFunctions,
        [&addNetwork](const Module *, ResultCode result) {
          if (result == Ready) {
            addNetwork.set_value(true);
          } else {
            addNetwork.set_value(false);
          }
        });
    auto result = ready.get();
    if (!result) {
      return ResultCode::Failed;
    }
    currDevice++;
    // Handle wrapping around to start of devices again.
    if (currDevice == devices.end()) {
      currDevice = devices.begin();
    }
  }
  return ResultCode::Ready;
};
