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

#include "glow/Runtime/HostManager/HostManager.h"
#include "glow/Backends/DeviceManager.h"
#include "glow/Graph/Context.h"
#include "glow/Partitioner/Partitioner.h"
#include "glow/Runtime/Executor/Executor.h"
#include "glow/Runtime/Provisioner/Provisioner.h"
#include "glow/Runtime/RuntimeTypes.h"

#include <future>
#include <queue>

using namespace glow;
using namespace runtime;

HostManager::HostManager(const std::vector<DeviceConfig> &configs) {
  DeviceIDTy deviceCount = 0;

  if (configs.size() > 0) {
    backend_.reset(createBackend(configs[0].backendKind));
  }
  for (auto &config : configs) {
    devices_[deviceCount] =
        std::unique_ptr<DeviceManager>(DeviceManager::createDeviceManager(
            config.backendKind, config.deviceName));
    ResultCode response = devices_[deviceCount]->init();
    assert(response == ResultCode::Executed && "Failed to initialize device.");
    (void)response;
    deviceCount++;
  }
  provisioner_.reset(new Provisioner(devices_));
  executor_.reset(createExecutor(devices_));
}

HostManager::~HostManager() { clearHost(); }

ResultCode HostManager::addNetwork(Module *M) {
  std::lock_guard<std::mutex> networkLock(networkLock_);
  std::vector<DeviceInfo> deviceInfo;
  for (auto &device : devices_) {
    DeviceInfo info = DeviceInfo();
    info.availableMemory = device.second->getAvailableMemory();
    deviceInfo.push_back(info);
  }
  // Optimize functions before passing to partitioner.
  // Currently hardcoding inference.
  if (backend_) {
    CompilationOptions opts;
    opts.mode = CompilationMode::Infer;
    for (auto F : M->getFunctions()) {
      backend_->optimizeFunction(F, opts);
    }
  }
  auto partitioner = Partitioner(M, deviceInfo);
  auto nodeList = std::move(partitioner.Partition());
  auto result = provisioner_->provision(nodeList.roots, *M);

  for (auto &node : nodeList.roots) {
    roots_.emplace(node->name, std::move(node));
  }
  for (auto &node : nodeList.nodes) {
    networks_.emplace(node->name, std::move(node));
  }
  return result;
}

void HostManager::removeNetwork(llvm::StringRef networkName) {
  // Walk the tree for the given function, calling evict function for each node
  // before removing it from networks_ which frees the node.
  std::lock_guard<std::mutex> networkLock(networkLock_);
  if (roots_.find(networkName) == roots_.end()) {
    return;
  }
  std::queue<std::string> nodes;
  std::set<std::string> allNodes;
  for (auto &child : roots_[networkName]->children) {
    nodes.push(child->name);
    allNodes.insert(child->name);
  }

  while (!nodes.empty()) {
    auto nodeName = nodes.front();
    nodes.pop();
    auto it = networks_.find(nodeName);
    if (it != networks_.end()) {
      for (auto &node : it->second->children) {
        auto name = node->name;
        if (allNodes.find(node->name) == allNodes.end()) {
          nodes.push(node->name);
          allNodes.insert(node->name);
        }
      }
    }
  }
  roots_.erase(roots_.find(networkName));
  for (auto &networkName : allNodes) {
    auto networkIterator = networks_.find(networkName);
    if (networkIterator != networks_.end()) {
      devices_[networkIterator->second->deviceID]->evictNetwork(
          networkName, /*evictCB=*/nullptr);
      networks_.erase(networkIterator);
    }
  }
}

bool HostManager::networkAdded(llvm::StringRef networkName) {
  std::lock_guard<std::mutex> networkLock(networkLock_);
  return roots_.find(networkName) != roots_.end();
}

void HostManager::clearHost() {
  // shutdown the executor, blocking on any current inflight and prevent new
  // requests from being serviced.
  executor_->shutdown();
  assert(activeRequestCount_ == 0 &&
         "All requests should be finished when shutting down HostManager.");

  std::lock_guard<std::mutex> networkLock(networkLock_);
  for (auto &it : devices_) {
    it.second->stop();
  }
  for (auto &network : networks_) {
    devices_[network.second->deviceID]->evictNetwork(network.second->name,
                                                     /*evictCB=*/nullptr);
  }
  networks_.clear();
  roots_.clear();
}

RunIdentifierTy HostManager::runNetwork(llvm::StringRef networkName,
                                        std::unique_ptr<Context> context,
                                        ResultCBTy callback) {

  auto currentRun = totalRequestCount_++;
  std::lock_guard<std::mutex> networkLock(networkLock_);
  if (roots_.find(networkName) == roots_.end()) {
    callback(currentRun, ResultCode::Failed, std::move(context));
    return currentRun;
  }
  if (activeRequestCount_ >= activeRequestLimit_) {
    callback(currentRun, ResultCode::Canceled, std::move(context));
    return currentRun;
  }
  activeRequestCount_++;
  executor_->run(roots_[networkName].get(), std::move(context), currentRun,
                 [&activeRequest = this->activeRequestCount_,
                  callback](RunIdentifierTy runID, ResultCode result,
                            std::unique_ptr<Context> context) {
                   --activeRequest;
                   callback(runID, result, std::move(context));
                 });
  return currentRun;
}
