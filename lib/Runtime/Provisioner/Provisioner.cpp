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

Provisioner::Provisioner(DeviceIDtoManagerMapTy &devices) {
  for (auto &device : devices) {
    devices_.push_back(device.second);
  }
}

ResultCode
Provisioner::provision(std::vector<std::unique_ptr<DAGNode>> &networks,
                       Module &module) {
  // For the first pass we will just assign and load devices in order and update
  // the deviceID field of the node.
  std::queue<std::vector<DAGNode *>> nextNode;
  // Process head node, this does not contain a function but serves as an entry
  // point for the network. We build a vector of nodes, containing all
  // sub-functions that use the same constants. Later we will group by
  // logicalDevice.
  for (unsigned int i = 0; i < networks[0]->children.size(); i++) {
    std::vector<DAGNode *> newSet;
    for (auto &node : networks) {
      newSet.push_back(node->children[i]);
    }
    nextNode.push(newSet);
  }
  while (!nextNode.empty()) {
    FunctionMapTy compiledFunctions;
    auto nodes = nextNode.front();
    nextNode.pop();

    // Add child nodes to the queue.
    for (unsigned int i = 0; i < nodes[0]->children.size(); i++) {
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

    // Get the next Device to be loaded.
    auto &device = devices_[nextDevice_];
    // Set backend to match the device.
    backend_.reset(createBackend(device->getBackendKind()));

    // Iterate over the nodes, compile them and add them to compiledFunctions.
    for (auto node : nodes) {
      node->deviceID = nextDevice_;
      Function *function = module.getFunction(node->name);
      auto compiled = backend_->compile(function);
      node->runtimeBundle = compiled->getRuntimeBundle();
      node->runtimeBundle.setInputsandOutputs();
      compiledFunctions.emplace(node->name, compiled.get());
      functions_.emplace(node->name, std::move(compiled));
    }

    // Check if sufficient space on device. Currently requiring a buffer
    // over the size of constants determined by NETWORK_PADDING_FACTOR.
    auto availableMemory = device->getAvailableMemory();
    auto requiredMemory = NETWORK_PADDING_FACTOR *
                          nodes[0]->runtimeBundle.getConstantWeightSize();
    if (availableMemory < requiredMemory) {
      return ResultCode::Failed;
    }

    // Load functions on device. Create a promise that is passed into the
    // callback that is passed to the deviceManager alongside a reference to the
    // module and a map of the functions. Then wait on the future for the
    // results.
    std::promise<bool> addNetwork;
    auto ready = addNetwork.get_future();
    device->addNetwork(&module, compiledFunctions,
                       [&addNetwork](const Module *, ResultCode result) {
                         addNetwork.set_value(result == ResultCode::Ready);
                       });
    auto result = ready.get();
    if (!result) {
      return ResultCode::Failed;
    }

    nextDevice_ = (nextDevice_ + 1) % devices_.size();
  }
  return ResultCode::Ready;
};
