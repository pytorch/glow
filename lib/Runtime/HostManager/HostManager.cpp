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

#include "glow/Runtime/HostManager/HostManager.h"
#include "glow/Backends/DeviceManager.h"
#include "glow/Graph/PlaceholderBindings.h"
#include "glow/Optimizer/GraphOptimizer/GraphOptimizer.h"
#include "glow/Partitioner/Partitioner.h"
#include "glow/Runtime/Executor/ThreadPoolExecutor.h"
#include "glow/Runtime/Provisioner/Provisioner.h"
#include "glow/Runtime/RuntimeTypes.h"
#include "glow/Support/Support.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"

#include <glog/logging.h>

#include <future>
#include <queue>
#include <shared_mutex>

using namespace glow;
using namespace runtime;

namespace {
llvm::cl::OptionCategory hostManagerCat("HostManager Options");

llvm::cl::opt<std::string> loadBackendSpecificOptionsOpt(
    "load-backend-specific-opts",
    llvm::cl::desc("Load backend-specific options for compilation."),
    llvm::cl::value_desc("options.yaml"), llvm::cl::Optional,
    llvm::cl::cat(hostManagerCat));
} // namespace

/// The device configs file used for Runtime.
llvm::cl::opt<std::string> loadDeviceConfigsFileOpt(
    "load-device-configs",
    llvm::cl::desc("Load device configs used in Runtime"),
    llvm::cl::value_desc("configs.yaml"), llvm::cl::Optional,
    llvm::cl::cat(hostManagerCat));

HostManager::HostManager()
    : config_(), statsExporterRegistry_(StatsExporterRegistry::Stats()) {}

HostManager::HostManager(const HostConfig &hostConfig)
    : config_(hostConfig),
      statsExporterRegistry_(StatsExporterRegistry::Stats()) {}

HostManager::HostManager(
    std::vector<std::unique_ptr<DeviceConfig>> deviceConfigs)
    : config_(), statsExporterRegistry_(StatsExporterRegistry::Stats()) {
  // TODO: move all initialization out of constructor.
  EXIT_ON_ERR(init(std::move(deviceConfigs)));
}

HostManager::HostManager(
    std::vector<std::unique_ptr<DeviceConfig>> deviceConfigs,
    const HostConfig &hostConfig)
    : config_(hostConfig),
      statsExporterRegistry_(StatsExporterRegistry::Stats()) {
  // TODO: move all initialization out of constructor.
  EXIT_ON_ERR(init(std::move(deviceConfigs)));
}

Expected<DAG *> HostManager::getNetworkDAG(llvm::StringRef network) {
  auto it = networks_.find(network);
  if (it == networks_.end()) {
    return MAKE_ERR(ErrorValue::ErrorCode::RUNTIME_ERROR, "Network not found.");
  }
  return &it->second.dag;
}

Error HostManager::init(std::vector<std::unique_ptr<DeviceConfig>> configs) {
  DeviceIDTy deviceCount = 0;

  for (auto &config : configs) {
    config->deviceID = deviceCount;
    if (!config->hasName()) {
      config->name = "config" + std::to_string(deviceCount);
    }

    devices_[deviceCount] = std::unique_ptr<DeviceManager>(
        DeviceManager::createDeviceManager(*config));

    RETURN_IF_ERR(devices_[deviceCount]->init());

    deviceCount++;
  }
  provisioner_.reset(new Provisioner(devices_));
  executor_.reset(new ThreadPoolExecutor(devices_, config_.executorThreads));
  exportMemoryCounters();
  return Error::success();
}

void HostManager::exportMemoryCounters() {
  uint64_t maxMem = 0;
  uint64_t availableMem = 0;
  for (auto &dev : devices_) {
    maxMem += dev.second->getMaximumMemory();
    availableMem += dev.second->getAvailableMemory();
  }
  statsExporterRegistry_->setCounter(kDeviceMemoryUsed, maxMem - availableMem);
  statsExporterRegistry_->setCounter(kDeviceMemoryAvailable, availableMem);
  statsExporterRegistry_->setCounter(kDeviceMemoryMax, maxMem);
}

HostManager::~HostManager() {
  ERR_TO_VOID(clearHost());
  exportMemoryCounters();
}

void HostManager::cleanupAddNetwork(llvm::ArrayRef<std::string> names) {
  for (auto &name : names) {
    processingNetworks_.erase(name);
  }
  exportMemoryCounters();
}

Error HostManager::addNetwork(std::unique_ptr<Module> module,
                              CompilationContext &cctx, bool saturateHost) {
  std::vector<std::string> names;
  {
    std::unique_lock<std::shared_timed_mutex> networkLock(networkLock_);
    auto functions = module->getFunctions();
    for (auto &F : functions) {
      std::string name = F->getName();
      auto it = networks_.find(name);
      if (it != networks_.end() ||
          processingNetworks_.find(name) != processingNetworks_.end()) {
        cleanupAddNetwork(names);
        return MAKE_ERR(
            ErrorValue::ErrorCode::RUNTIME_ERROR,
            "Failed to add network: already have a function called " + name);
      }
      // Add the network to processingNetworks_ so we know it's being worked on.
      processingNetworks_.insert(name);
      names.push_back(name);
    }
  }
  // Load backend-specific options if specified.
  if (!loadBackendSpecificOptionsOpt.empty()) {
    if (cctx.backendOpts.backendSpecificOpts.size() != 0) {
      VLOG_EVERY_N(1, 1000) << "Warning: backendSpecificOpts is set via the "
                               "HostManager, ignoring previously set options.";
    }
    cctx.backendOpts.backendSpecificOpts =
        deserializeStrStrMapFromYaml(loadBackendSpecificOptionsOpt);
  }

  std::vector<DeviceInfo> deviceInfo;
  {
    std::unique_lock<std::shared_timed_mutex> networkLock(networkLock_);
    for (auto &device : devices_) {
      DeviceInfo info = device.second->getDeviceInfo();
      info.availableMemory = device.second->getAvailableMemory();
      info.backendName = device.second->getBackendName();
      info.nonSupportedNodes =
          device.second->getParamByName("nonSupportedNodes");
      info.supportedNodes = device.second->getParamByName("supportedNodes");
      deviceInfo.push_back(info);
    }
  }
  // Perform a round of target-independent graph optimizations. This helps the
  // partitioner to do its job more efficiently.
  for (Function *F : module->getFunctions()) {
    auto err = optimizeFunctionBeforeLowering(F, cctx);
    if (err) {
      {
        std::unique_lock<std::shared_timed_mutex> networkLock(networkLock_);
        cleanupAddNetwork(names);
      }
      return err;
    }
  }
  Partitioner partitioner(module.get(), deviceInfo, saturateHost);
  DAGListTy nodeList;
  auto result = partitioner.partition(cctx);
  if (result) {
    nodeList = std::move(result.get());
  } else {
    std::unique_lock<std::shared_timed_mutex> networkLock(networkLock_);
    cleanupAddNetwork(names);
    return result.takeError();
  }

  if (cctx.precisionConfig.quantMode == QuantizationMode::Profile) {
    // Since for profiling the provisioner will be reset, we only allow one
    // network in one HM.
    if (networks_.size() > 0) {
      return MAKE_ERR(ErrorValue::ErrorCode::RUNTIME_ERROR,
                      "For quantization profiling flow, there can't be other "
                      "registered networks before this one");
    }
    // For profiling, we use CPU backend. Overwrite Provisioner and Executor
    // to force the network is compiled and run in profilingBackend. backend.
    size_t devicesNum = devices_.size();
    for (size_t i = 0; i < devicesNum; i++) {
      auto name = devices_[i]->getDeviceConfig().name;
      auto config = glow::make_unique<DeviceConfig>(profilingBackend, name);
      devices_[i] = std::unique_ptr<DeviceManager>(
          DeviceManager::createDeviceManager(*config));
      RETURN_IF_ERR(devices_[i]->init());
    }
    provisioner_.reset(new Provisioner(devices_));
    executor_.reset(new ThreadPoolExecutor(devices_, config_.executorThreads));
  }

  auto err = provisioner_->provision(nodeList, *module, cctx);
  if (err) {
    {
      std::unique_lock<std::shared_timed_mutex> networkLock(networkLock_);
      cleanupAddNetwork(names);
    }
    return err;
  }

  // Clear constants contents from the module then put it in a
  // shared_ptr to be shared between all of the networks created from each
  // function in the module.
  module->strip();
  auto sharedModule = std::shared_ptr<Module>(std::move(module));
  {
    std::unique_lock<std::shared_timed_mutex> networkLock(networkLock_);
    for (auto &node : nodeList) {
      auto &networkData = networks_[(node.root)->name];
      networkData.dag = std::move(node);
      networkData.module = sharedModule;
    }
    cleanupAddNetwork(names);
  }
  return Error::success();
}

Error HostManager::removeNetwork(llvm::StringRef networkName) {
  std::unique_lock<std::shared_timed_mutex> networkLock(networkLock_);
  auto networkIterator = networks_.find(networkName);
  if (networkIterator == networks_.end()) {
    return Error::success();
  }

  if (processingNetworks_.find(networkName) != processingNetworks_.end()) {
    // Return an error, the network is in an incomplete state likely because
    // it is still being added by a different call.
    return MAKE_ERR(ErrorValue::ErrorCode::RUNTIME_NET_BUSY,
                    llvm::formatv("Cannot remove the network {0}, as it is "
                                  "currently being modified.",
                                  networkName)
                        .str());
  }

  // Issue an error as there are outstanding runs for the network
  if (networkIterator->second.refcount != 0) {
    return MAKE_ERR(ErrorValue::ErrorCode::RUNTIME_NET_BUSY,
                    llvm::formatv("Cannot remove the network {0}, as there are "
                                  "still outstanding runs",
                                  networkName)
                        .str());
  }

  OneErrOnly err;
  auto &nodes = networkIterator->second.dag.nodes;
  for (auto &node : nodes) {
    for (auto device : node->deviceIDs) {
      Error evictErr = provisioner_->evictFunction(node->name, device);
      err.set(std::move(evictErr));
    }
    // Also remove compiledFunction from Provisioner.
    err.set(provisioner_->removeFunction(node->name));
  }
  networks_.erase(networkIterator);
  exportMemoryCounters();
  return err.get();
}

bool HostManager::networkAdded(llvm::StringRef networkName) {
  std::shared_lock<std::shared_timed_mutex> networkLock(networkLock_);
  return networks_.find(networkName) != networks_.end();
}

Error HostManager::clearHost() {
  // shutdown the executor, blocking on any current inflight and prevent new
  // requests from being serviced.
  executor_->shutdown();

  DCHECK_EQ(activeRequestCount_, 0)
      << "All requests should be finished when shutting down HostManager.";

  // Remove all networks from the host and device(s).
  while (networks_.size() != 0) {
    RETURN_IF_ERR(removeNetwork(networks_.begin()->first));
  }

  // Now it's safe to stop the DeviceManagers.
  std::unique_lock<std::shared_timed_mutex> networkLock(networkLock_);
  OneErrOnly errContainer;
  for (auto &it : devices_) {
    errContainer.set(it.second->stop());
  }
  // Zero out counters.
  statsExporterRegistry_->setCounter(kDeviceMemoryUsed, 0);
  statsExporterRegistry_->setCounter(kDeviceMemoryAvailable, 0);
  statsExporterRegistry_->setCounter(kDeviceMemoryMax, 0);

  return errContainer.get();
}

Error HostManager::runNetworkBlocking(llvm::StringRef networkName,
                                      PlaceholderBindings &bindings) {
  std::unique_ptr<PlaceholderBindings> phBindings(&bindings);
  std::unique_ptr<ExecutionContext> context =
      glow::make_unique<ExecutionContext>(std::move(phBindings));
  std::promise<void> runPromise;
  auto fut = runPromise.get_future();
  std::unique_ptr<Error> runErr;
  runNetwork(
      networkName, std::move(context),
      [&runPromise, &runErr](runtime::RunIdentifierTy, Error err,
                             std::unique_ptr<ExecutionContext> contextPtr) {
        // Don't delete ph bindings since they were created from a passed in
        // reference.
        std::unique_ptr<PlaceholderBindings> phBind =
            contextPtr->movePlaceholderBindings();
        phBind.release();

        runErr = glow::make_unique<Error>(std::move(err));
        runPromise.set_value();
      });

  fut.wait();
  return std::move(*DCHECK_NOTNULL(runErr.get()));
}

Error HostManager::runNetworkBlocking(
    llvm::StringRef networkName, std::unique_ptr<ExecutionContext> context) {
  std::promise<void> runPromise;
  auto fut = runPromise.get_future();
  std::unique_ptr<Error> runErr;
  runNetwork(
      networkName, std::move(context),
      [&runPromise, &runErr](runtime::RunIdentifierTy, Error err,
                             std::unique_ptr<ExecutionContext> contextPtr) {
        runErr = glow::make_unique<Error>(std::move(err));
        runPromise.set_value();
      });

  fut.wait();
  return std::move(*DCHECK_NOTNULL(runErr.get()));
}

void HostManager::dispatchNextRun() {
  int requestId = -1;
  llvm::Optional<InferRequest> pRequest;
  std::shared_lock<std::shared_timed_mutex> networkLock(networkLock_);
  {
    // hmm this lock is hot but I still have it as a unique lock because
    // we always need to pop inferQueue and inferQueue is not thread safe
    std::unique_lock<std::shared_timed_mutex> queueLock(inferQueueLock_);
    if (inferQueue_.size()) {
      // Get the next request, unfortunately priority_queue only
      // provides a const ref to the top element, since we need to move
      // it we first cast it to remove the const.
      pRequest = std::move(const_cast<InferRequest &>(inferQueue_.top()));
      requestId = static_cast<int>(pRequest->requestID);
      inferQueue_.pop();
    } else {
      // Decrement the activeRequest counter so new requests can
      // launched.
      --activeRequestCount_;
      return;
    }
  }

  assert(pRequest.hasValue());
  InferRequest request = std::move(pRequest.getValue());
  TRACE_EVENT_CREATE(beginTraceEvent, TraceLevel::REQUEST,
                     "network_execution_e2e", TraceEvent::AsyncBeginType,
                     requestId);
  auto startTime = TraceEvent::now();
  executor_->run(
      networks_[request.networkName].dag.root.get(), std::move(request.context),
      request.requestID,
      [this, callback = request.callback, name = request.networkName, startTime,
       requestId, beginTraceEvent = std::move(beginTraceEvent)](
          RunIdentifierTy runID, Error err,
          std::unique_ptr<ExecutionContext> context) mutable {
        {
          std::shared_lock<std::shared_timed_mutex> netLock(networkLock_);
          auto it = networks_.find(name);
          if (it != networks_.end()) {
            it->second.refcount--;
          }
        }
        TRACE_EVENT_LOG_PRE_CREATED(context->getTraceContext(),
                                    beginTraceEvent);
        TRACE_EVENT_LOG_ID(context->getTraceContext(), TraceLevel::REQUEST,
                           "network_execution_e2e", TraceEvent::AsyncEndType,
                           TraceEvent::now(), requestId);
        updateExecutionStats(startTime, context);
        callback(runID, std::move(err), std::move(context));
        dispatchNextRun();
      });
}

RunIdentifierTy
HostManager::runNetwork(llvm::StringRef networkName,
                        std::unique_ptr<ExecutionContext> context,
                        ResultCBTy callback, uint64_t priority) {
  DCHECK(callback != nullptr);

  TRACE_EVENT_SCOPE(context->getTraceContext(), TraceLevel::RUNTIME,
                    "HostManager::runNetwork");
  if (context->getTraceContext()) {
    context->getTraceContext()->setThreadName("Caller");
  }
  auto currentRun = totalRequestCount_++;

  NetworkData *network = nullptr;
  {
    std::shared_lock<std::shared_timed_mutex> networkLock(networkLock_);
    auto it = networks_.find(networkName);
    if (it != networks_.end()) {
      network = &it->second;
      network->refcount++;
    }

    if (network == nullptr) {
      callback(
          currentRun,
          MAKE_ERR(ErrorValue::ErrorCode::RUNTIME_NET_NOT_FOUND,
                   llvm::formatv("Function {0} not found", networkName).str()),
          std::move(context));
      return currentRun;
    }
    // Setup the request
    InferRequest queuedRequest(networkName, std::move(context), callback,
                               priority, currentRun);
    // Put the request in the queue.
    {
      std::shared_lock<std::shared_timed_mutex> lock(inferQueueLock_);
      auto queueSize = inferQueue_.size();
      if (queueSize >= config_.maxQueueSize) {
        // The queue is full, return an error.
        network->refcount--;
        callback(
            currentRun,
            MAKE_ERR(
                ErrorValue::ErrorCode::RUNTIME_REQUEST_REFUSED,
                strFormat(
                    "The number of allowed queued requests has been exceeded. "
                    "queued requests: %lu allowed requests: %zu",
                    queueSize, config_.maxQueueSize)),
            std::move(context));
        return currentRun;
      }
    }
    {
      std::unique_lock<std::shared_timed_mutex> lock(inferQueueLock_);
      inferQueue_.push(std::move(queuedRequest));
    }
  }

  // If we haven't reached maxActiveRequests kick off next request.
  size_t activeRequestCount = activeRequestCount_++;
  if (activeRequestCount < config_.maxActiveRequests) {
    dispatchNextRun();
    return currentRun;
  }
  activeRequestCount_--;
  return currentRun;
}

/// Helper to update execution stats
void HostManager::updateExecutionStats(
    uint64_t startTime, std::unique_ptr<ExecutionContext> &context) {
  auto duration = TraceEvent::now() - startTime;
  statsExporterRegistry_->addTimeSeriesValue("network_execution_e2e", duration);
  statsExporterRegistry_->incrementCounter("network_execution");
  if (context && context->getPlaceholderBindings() && duration > 0) {
    statsExporterRegistry_->addTimeSeriesValue(
        "network_execution_throughput",
        context->getPlaceholderBindings()->getDataSize() * 1000000 / duration);
  }
}

/// Helper to get the parameters in DeviceConfig from \p str. The \p str has
/// multiple lines, and each line with this format : "str1" : "str2".
static llvm::StringMap<std::string> getBackendParams(std::string &str) {
  llvm::StringMap<std::string> ret{};
  std::string s;
  std::istringstream f(str.c_str());
  while (getline(f, s, '\n')) {
    // Abstract the mapping from each line's string:
    // ""str1" : "str2"" => ret["str1"] = "str2";
    size_t pos1, pos2, pos3, pos4;
    pos1 = s.find('"');
    assert(pos1 != std::string::npos && "invalid string format");
    pos2 = s.find('"', pos1 + 1);
    assert(pos2 != std::string::npos && "invalid string format");
    pos3 = s.find('"', pos2 + 1);
    assert(pos3 != std::string::npos && "invalid string format");
    pos4 = s.find('"', pos3 + 1);
    assert(pos4 != std::string::npos && "invalid string format");
    ret[s.substr(pos1 + 1, pos2 - pos1 - 1)] =
        s.substr(pos3 + 1, pos4 - pos3 - 1);
  }
  return ret;
}

/// If the device config file \p loadDeviceDoncfigsFile available, load \p
/// configs from the file. Otherwise, create \p numDevices number of devices
/// based on \p backendName.
std::vector<std::unique_ptr<runtime::DeviceConfig>>
runtime::generateDeviceConfigs(unsigned int numDevices,
                               llvm::StringRef backendName, size_t memSize) {
  std::vector<std::unique_ptr<runtime::DeviceConfig>> configs;
  if (!loadDeviceConfigsFromFile(configs, memSize)) {
    // If there is no device config file, use numDevices to generate the
    // configs.
    for (unsigned int i = 0; i < numDevices; ++i) {
      auto config = glow::make_unique<runtime::DeviceConfig>(backendName);
      config->setDeviceMemory(memSize);
      config->deviceID = i;
      configs.push_back(std::move(config));
    }
  }
  return configs;
}

bool runtime::loadDeviceConfigsFromFile(
    std::vector<std::unique_ptr<runtime::DeviceConfig>> &configs,
    size_t memSize) {
  if (loadDeviceConfigsFileOpt.empty()) {
    return false;
  }

  std::vector<DeviceConfigHelper> lists;
  lists = deserializeDeviceConfigFromYaml(loadDeviceConfigsFileOpt);
  for (unsigned int i = 0; i < lists.size(); ++i) {
    std::string configBackendName = lists[i].backendName_;
    std::string name = lists[i].name_;
    auto parameters = getBackendParams(lists[i].parameters_.str);
    auto config = glow::make_unique<runtime::DeviceConfig>(configBackendName,
                                                           name, parameters);
    config->setDeviceMemory(memSize);
    configs.push_back(std::move(config));
  }
  return true;
}

Backend &HostManager::getBackend(llvm::StringRef backendName) const {
  return provisioner_->getBackend(backendName);
}
