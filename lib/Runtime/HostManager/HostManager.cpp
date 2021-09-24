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
#include "glow/Exporter/ONNXModelWriter.h"
#include "glow/Flags/Flags.h"
#include "glow/Graph/PlaceholderBindings.h"
#include "glow/Optimizer/GraphOptimizer/GraphOptimizer.h"
#include "glow/Partitioner/Partitioner.h"
#include "glow/Runtime/DeferredWeightLoader.h"
#include "glow/Runtime/DeviceHealthMonitor.h"
#include "glow/Runtime/ErrorReporter.h"
#include "glow/Runtime/Executor/ThreadPoolExecutor.h"
#include "glow/Runtime/Provisioner/Provisioner.h"
#include "glow/Runtime/RequestData.h"
#include "glow/Runtime/RuntimeTypes.h"
#include "glow/Support/Support.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"

#include <glog/logging.h>

#include "folly/String.h"
#include "folly/executors/CPUThreadPoolExecutor.h"

#include <algorithm>
#include <future>
#include <queue>
#include <shared_mutex>

constexpr uint64_t P2PInputLimit = 256;
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

namespace glow {

#if FACEBOOK_INTERNAL
Error optimizeDAG(DAGListTy &nodeList, const Provisioner &provisioner,
                  Module &mod, const std::vector<DeviceInfo> &devices,
                  CompilationContext &cctx,
                  ConstantFoldingRecordMap &constFoldRecord);
extern const char *revisionHash;
#endif /* FACEBOOK_INTERNAL */
} // namespace glow

/// The device configs file used for Runtime.
llvm::cl::opt<std::string> loadDeviceConfigsFileOpt(
    "load-device-configs",
    llvm::cl::desc("Load device configs used in Runtime"),
    llvm::cl::value_desc("configs.yaml"), llvm::cl::Optional,
    llvm::cl::cat(hostManagerCat));

/// The value that should be used for device initialization timeout, default:
/// 5000 milliseconds.
llvm::cl::opt<unsigned, /* ExternalStorage */ true> deviceInitTimeout(
    "device_init_timeout_ms",
    llvm::cl::desc("Set device init timout in milliseconds"),
    llvm::cl::Optional,
    llvm::cl::location(glow::runtime::flags::DeviceInitTimeoutMs),
    llvm::cl::cat(hostManagerCat));

HostManager::HostManager() : HostManager(HostConfig{}) {}

HostManager::HostManager(const HostConfig &hostConfig)
    : config_(hostConfig),
      statsExporterRegistry_(StatsExporterRegistry::Stats()) {
  statsExporterRegistry_->setCounter(kMaxQueueSize, hostConfig.maxQueueSize);
}

HostManager::HostManager(
    std::vector<std::unique_ptr<DeviceConfig>> deviceConfigs)
    : HostManager(std::move(deviceConfigs), HostConfig{}) {}

HostManager::HostManager(
    std::vector<std::unique_ptr<DeviceConfig>> deviceConfigs,
    const HostConfig &hostConfig)
    : config_(hostConfig),
      statsExporterRegistry_(StatsExporterRegistry::Stats()) {
  // TODO: move all initialization out of constructor.

  REPORT_AND_EXIT_ON_ERR(init(std::move(deviceConfigs)));
  statsExporterRegistry_->setCounter(kMaxQueueSize, hostConfig.maxQueueSize);
}

Expected<DAG *> HostManager::getNetworkDAG(llvm::StringRef network) {
  auto it = networks_.find(network.str());
  if (it == networks_.end()) {
    return MAKE_ERR(ErrorValue::ErrorCode::RUNTIME_ERROR, "Network not found.");
  }
  return &it->second.dag;
}

Error HostManager::startDeviceTrace() {
  LOG(INFO) << "start device tracing" << std::endl;
  for (auto &dev : devices_) {
    Error err = dev.second->startDeviceTrace(hostTraceContext_.get());
    RETURN_IF_ERR(err);
  }
  return Error::success();
}

Error HostManager::stopDeviceTrace() {

  auto *traceContext = hostTraceContext_.get();
  if (!traceContext) {
    LOG(INFO) << "No HostManager TraceContext registered, skipping call to "
                 "stopDeviceTrace";
    return Error::success();
  } else {
    LOG(INFO) << "stop device tracing";
  }
  for (auto &dev : devices_) {
    Error err = dev.second->stopDeviceTrace(traceContext);
    RETURN_IF_ERR(err);
  }
  return Error::success();
}

Error HostManager::init(std::vector<std::unique_ptr<DeviceConfig>> configs) {
  static std::once_flag monitorFlag;
  std::call_once(monitorFlag, []() {
    auto monitors = DeviceHealthMonitorRegistry::Monitors();
    if (monitors) {
      monitors->start();
    }
  });

  DeviceIDTy deviceCount = 0;
  for (auto &config : configs) {
    if (!config->hasName()) {
      config->name = "device_" + std::to_string(deviceCount);
    }

    devices_[deviceCount] = std::unique_ptr<DeviceManager>(
        DeviceManager::createDeviceManager(*config));

    std::promise<Error> devPromise;
    auto devFuture = devPromise.get_future();
    auto *dev = devices_[deviceCount].get();
    threadPool_.submit([&devPromise, dev] {
      auto err = dev->init();
      devPromise.set_value(std::move(err));
    });
    if (devFuture.wait_for(std::chrono::milliseconds(
            flags::DeviceInitTimeoutMs)) != std::future_status::timeout) {
      RETURN_IF_ERR(devFuture.get());
    } else {
      // Device initialization is taking longer than expected, return an error.
      return MAKE_ERR(ErrorValue::ErrorCode::RUNTIME_ERROR,
                      "Timeout encountered when initializing device: " +
                          std::string(config->name));
    }
    availableDevices_.push_back(deviceCount);
    deviceCount++;
  }
#ifdef FACEBOOK_INTERNAL
  LOG(INFO) << "Initialized " << deviceCount << " device(s)";
#endif

  provisioner_.reset(new Provisioner(devices_));
  executor_.reset(
      new ThreadPoolExecutor(devices_, config_.executorThreads, "HostManager"));
  exportMemoryCounters();
  if (flags::AvailableDevices.length()) {
    std::vector<unsigned> devices;
    folly::split<char, std::string, unsigned>(',', flags::AvailableDevices,
                                              devices,
                                              /* ignoreEmpty */ true);
    std::vector<runtime::DeviceIDTy> convertedDevs(devices.begin(),
                                                   devices.end());
    setAvailableDevices(convertedDevs);
  }
  // If no HostManager is registered yet, register this one.
  if (!ManagerRegistry()->getHostManager()) {
    ManagerRegistry()->registerHostManager(this);
  }

  return Error::success();
}

void HostManager::setAvailableDevices(const std::vector<DeviceIDTy> &devices) {
  // Validate new device list.
  availableDevices_.clear();
  std::vector<DeviceIDTy> mapping;
  std::vector<DeviceManager *> availableDevices;
  // Grab a lock to prevent devices_ getting changed concurrently.
  std::unique_lock<std::shared_timed_mutex> networkLock(networkLock_);
  for (auto dev : devices) {
    auto it = devices_.find(dev);
    if (it != devices_.end()) {
      availableDevices_.push_back(dev);
      availableDevices.push_back(devices_[dev].get());
      mapping.push_back(it->first);
    }
  }
  // Update the provisioner.
  provisioner_->updateAvailableDevices(availableDevices, mapping);
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
  LOG(INFO) << "Destroying host manager...";
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
                              CompilationContext &cctx) {
#ifdef FACEBOOK_INTERNAL
  LOG(INFO) << "Adding Glow network built with revision hash: " << revisionHash;
#endif /* FACEBOOK_INTERNAL */
  VLOG(1) << "addNetwork";
  ScopeGuard debugDumpDAGGuard([&]() {
    if (cctx.dumpFinalGraph) {
      for (Function *F : module->getFunctions()) {
        auto fname = strFormat("%sfinal_graph_dbg_err_%s.dot",
                               cctx.dumpGraphPath.c_str(), F->getName().data());
        LOG(INFO) << "Dumping final graph due to error to " << fname;
        F->dumpDAG(fname);
      }
    }
  });

  /// If specified in the cctx, this will prevent Constants from being modified
  /// until the current scope ends or the preventer is dismissed. Does so by
  /// swapping in temporary Placeholders instead of Constants.
  ConstantModificationPreventer constModPreventer(*module, cctx);
  if (cctx.optimizationOpts.delayAndRecordConstantModification) {
    constModPreventer.activate();
  }

  std::vector<std::string> names;
  {
    std::unique_lock<std::shared_timed_mutex> networkLock(networkLock_);
    auto functions = module->getFunctions();
    for (auto &F : functions) {
      std::string name = F->getName().str();
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

  // Issue a warning when loading backend specific options from the command line
  // and the compile context also contains backend specific options.
  if (!loadBackendSpecificOptionsOpt.empty()) {
    if (cctx.backendOpts.backendSpecificOpts.size() != 0) {
      VLOG_EVERY_N(1, 1000) << "Warning: backendSpecificOpts is set via the "
                               "HostManager, ignoring previously set options.";
    }
    cctx.backendOpts.backendSpecificOpts =
        deserializeStrStrMapFromYaml(loadBackendSpecificOptionsOpt);
  } else {
    auto ctxLoadBackendSpecificOpt =
        cctx.backendOpts.backendSpecificOpts.find("loadBackendSpecificOptions");

    if (ctxLoadBackendSpecificOpt !=
        cctx.backendOpts.backendSpecificOpts.end()) {
      cctx.backendOpts.backendSpecificOpts =
          deserializeStrStrMapFromYaml(ctxLoadBackendSpecificOpt->second);
    }
  }

  std::vector<DeviceInfo> deviceInfo;
  {
    std::unique_lock<std::shared_timed_mutex> networkLock(networkLock_);
    for (auto &device : availableDevices_) {
      DeviceInfo info = devices_[device]->getDeviceInfo();
      info.availableMemory = devices_[device]->getAvailableMemory();
      info.backendName = devices_[device]->getBackendName().str();
      info.nonSupportedNodes =
          devices_[device]->getParamByName("nonSupportedNodes").str();
      info.supportedNodes =
          devices_[device]->getParamByName("supportedNodes").str();
      // If p2p is enabled update the inputCount limit.
      if (cctx.enableP2P) {
        info.inputCountMax = P2PInputLimit;
      }
      deviceInfo.push_back(info);
    }
  }

  // Optimize Functions only if we don't have any backendSpecificNodeInfo,
  // because if we do then the Functions were already optimized and Nodes had
  // extra info mapped to them, so we don't want to mutate the Function. Also
  // skip optimizations if we're loading an AOT optimized model.
  const bool skipOptimizations =
      cctx.loadingAOTModel || !cctx.backendOpts.backendSpecificNodeInfo.empty();

  // Perform a round of target-independent graph optimizations. This helps the
  // partitioner to do its job more efficiently.
  if (!skipOptimizations) {
    for (Function *F : module->getFunctions()) {
      auto err = optimizeFunctionBeforeLowering(F, cctx);
      if (err) {
        {
          std::unique_lock<std::shared_timed_mutex> networkLock(networkLock_);
          cleanupAddNetwork(names);
        }
        RETURN_ERR(err);
      }
    }
  }
  VLOG(1) << "Before partitioner";
  Partitioner partitioner(module.get(), deviceInfo, skipOptimizations);
  auto backendName = devices_[0]->getBackendName();
  const auto &backend = provisioner_->getBackend(backendName);
  auto contextCount = backend.getContextCount(cctx);
  partitioner.setContextCount(contextCount);
  DAGListTy nodeList;
  auto result = partitioner.partition(cctx);
  VLOG(1) << "After partitioner";
  if (result) {
    nodeList = std::move(result.get());
  } else {
    std::unique_lock<std::shared_timed_mutex> networkLock(networkLock_);
    cleanupAddNetwork(names);
    RETURN_ERR(result.takeError());
  }
  VLOG(1) << "Before quantmode";
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

  VLOG(1) << "Before replace dummy TQPs";
  // Now that we've partitioned and optimized, do some verification based on the
  // dummy mode we're using, if any.
  if (cctx.precisionConfig.replaceDummyTQPs ||
      cctx.precisionConfig.loadUniquedDummyQParams) {
    RETURN_IF_ERR(module->verifyDummyQParams(
        cctx.precisionConfig.loadUniquedDummyQParams));
  }

  // If we are loading an AOT model where we are replacing dummy TQPs, then we
  // may need to update Relu output types on FCs, since they should be set to
  // use zero as min but the correct qparams could not be calculated AOT.
  if (cctx.loadingAOTModel && cctx.precisionConfig.replaceDummyTQPs) {
    LOG(INFO) << "Updating quantized Relu types given real TQPs";
    for (Function *F : module->getFunctions()) {
      updateQuantReluTypes(F);
    }
  }

  VLOG(1) << "Before constant folding";
  // If we prevented constant modification then run constant folding with
  // recording now. Record so that if we are going to serialize we can embed the
  // constant folding subgraphs in the Glow ONNX model.
  ConstantFoldingRecordMap record;
  if (cctx.optimizationOpts.delayAndRecordConstantModification) {
    constModPreventer.deactivateAndCleanup();

    RETURN_ERR_IF_NOT(nodeList.size() == 1, "Expect only one DAG.");
    const auto &dag = *nodeList.begin();
    for (auto &dagNode : dag.nodes) {
      Function *F = module->getFunction(dagNode->name);
      RETURN_ERR_IF_NOT(
          F, strFormat("Function %s not found", dagNode->name.data()));

      ConstantFoldingRecordMap currRecord = constantFoldAndRecord(F, cctx);
      record.insert(currRecord.begin(), currRecord.end());
      runDCEPass(F, cctx);

      // Verify the Function is valid after constant folding takes place.
      Backend &B = provisioner_->getBackend(dagNode->backendName);
      RETURN_ERR_IF_NOT(
          B.verify(*F, cctx.verboseCompile),
          "Unsupported node(s) found after delayed constant folding Function " +
              F->getName().str() + " for backend " + B.getBackendName());
    }
  }
  VLOG(1) << "Before loading AOT";
  if (!cctx.loadingAOTModel) {
    if (cctx.callDAGOptimizer) {
#if FACEBOOK_INTERNAL
      auto optDagErr = optimizeDAG(nodeList, *provisioner_, *module, deviceInfo,
                                   cctx, record);
      if (optDagErr) {
        std::unique_lock<std::shared_timed_mutex> networkLock(networkLock_);
        cleanupAddNetwork(names);
        RETURN_ERR(optDagErr);
      }
#endif /* FACEBOOK_INTERNAL */
    } else {
      // If not using the DAG optimizer, iterate over the DAGs and call
      // transformPostOptPipeline() on the Functions.
      VLOG(1) << "No DAG optimizer";
      for (const auto &dag : nodeList) {
        for (auto &dagNode : dag.nodes) {
          Function *F = module->getFunction(dagNode->name);
          RETURN_ERR_IF_NOT(
              F, strFormat("Function %s not found", dagNode->name.data()));

          if (cctx.optimizationOpts.onlyLowerFuns.count(F)) {
            continue;
          }

          Backend &B = provisioner_->getBackend(dagNode->backendName);
          RETURN_IF_EXPECTED_IS_ERR(B.transformPostOptPipeline(F, cctx));

          RETURN_ERR_IF_NOT(
              B.verify(*F, cctx.verboseCompile),
              "Unsupported node(s) found after transformPostOptPipeline() " +
                  F->getName().str() + " for backend " + B.getBackendName());
        }
      }
    }
  }

  VLOG(1) << "Before serialize compile DAG";
  // If requested, serialize the resulting DAG that was just optimized and
  // partitioned.
  if (cctx.serializeCompiledDAG) {
    std::string loc;
    char *envSpecifiedSerializationPath = getenv("GLOW_DAG_SERIALIZATION_LOC");
    if (!envSpecifiedSerializationPath) {
      loc = nodeList.begin()->root->name + ".onnxtxt";
    } else {
      loc = std::string(envSpecifiedSerializationPath);
    }

    LOG(INFO) << "Serializing final compiled DAG to " << loc;
    {
      llvm::StringMap<std::string> extraMetadataProps;
      if (cctx.precisionConfig.originNameToTQPMap) {
        RETURN_IF_ERR(ONNXModelWriter::insertLoaderNameUniqueOffsetMetadata(
            extraMetadataProps, *cctx.precisionConfig.originNameToTQPMap));
      }
      if (cctx.precisionConfig.clipQuantRangeToFP16) {
        extraMetadataProps[clipQuantRangeToFP16Key] = "1";
      }
      Error writeErr = Error::empty();
      // Note: If cctx.skipProvisioning then we want to serialize all meta info
      // as we are likely doing AOT optimization. Otherwise do not provide the
      // meta info as the model does not need to be reloaded.
      ONNXModelWriter onnxWR(
          loc, nodeList, 7, 9, &writeErr,
          /* textMode */ true,
          /* zipMode */ cctx.useZipModeForSerializeCompiledDAG,
          /* includeConstantData */ cctx.saveConstantInSerializeCompiledDAG,
          extraMetadataProps, record, cctx.backendOpts.backendSpecificNodeInfo,
          cctx.skipProvisioning ? &cctx.loadedPHNames : nullptr,
          cctx.skipProvisioning ? &cctx.staticPlaceholderTypesForAOT : nullptr,
          cctx.returnGlowSerializedModelStr
              ? cctx.glowAOTSerializationModelStrPtr.get()
              : nullptr);
      RETURN_IF_ERR(writeErr);
    }

    // If we're using AOT DAG optimizer then skip provisioning.
    if (cctx.skipProvisioning ||
        (cctx.callDAGOptimizer && cctx.useDAGOptimizerAOTMode)) {
      LOG(INFO) << "Host manager skipping provisioning";
      {
        std::unique_lock<std::shared_timed_mutex> networkLock(networkLock_);
        cleanupAddNetwork(names);
      }
      debugDumpDAGGuard.dismiss();
      cleanupConstantFolding(*module, record);
      if (cctx.dumpFinalGraph) {
        for (Function *F : module->getFunctions()) {
          auto fname =
              strFormat("%sfinal_graph_aot_%s.dot", cctx.dumpGraphPath.c_str(),
                        F->getName().data());
          LOG(INFO) << "Dumping final graph to " << fname;
          F->dumpDAG(fname);
        }
      }
      return Error::success();
    }
  }

  // Now that we've serialized the model if requested, cleanup the temporary
  // Functions and PHs used for constant folding.
  cleanupConstantFolding(*module, record);
  VLOG(1) << "Before provisioning";
  auto err = provisioner_->provision(nodeList, *module, cctx);
  if (err) {
    if (err.peekErrorValue()->isFatalError()) {
      statsExporterRegistry_->setCounter(kDeviceFatalError, 1);
    }
    {
      std::unique_lock<std::shared_timed_mutex> networkLock(networkLock_);
      cleanupAddNetwork(names);
    }
    RETURN_ERR(err);
  }
  debugDumpDAGGuard.dismiss();
  VLOG(1) << "Calculation of maxActiveRequests";
  {
    std::unique_lock<std::shared_timed_mutex> networkLock(networkLock_);
    /// Calculate networkMaxActive requests. Then update
    /// config_.maxActiveRequests This will be maxActiveRequestsPerInstance *
    /// instanceCount * minReplications or config_.maxActiveRequests whichever
    /// is smaller.

    // Find the minimum on device replication.
    unsigned minReplications{1};
    for (auto &node : nodeList) {
      for (auto &dag : node.nodes) {
        minReplications = std::min(dag->replicationCount, minReplications);
      }
    }
    unsigned product{0};
    if (nodeList.size() && nodeList[0].nodes.size()) {
      product = nodeList[0].nodes[0]->instanceCount *
                cctx.maxActiveRequestsPerInstance * minReplications;
    } else {
      return MAKE_ERR(ErrorValue::ErrorCode::RUNTIME_ERROR,
                      "NodeList is empty.");
    }
    unsigned maxActiveRequests = config_.maxActiveRequests;
    config_.maxActiveRequests = std::min(product, maxActiveRequests);

    // Create pool of cachedExecutionStates.
    for (auto &node : nodeList) {
      // Note: currently getNextNetworkExecutionState assumes that pool size is
      // >= currentInFlight requests, so we set pool size to maxActiveRequests.
      executor_->createPool(node.root.get(), config_.maxActiveRequests,
                            cctx.enableP2P, cctx.enableDRT);
    }
  }
  // Clear constants contents from the module then put it in a
  // shared_ptr to be shared between all of the networks created from each
  // function in the module.
  auto targetBackendName = std::string(devices_[0]->getBackendName());
  const auto &targetBackend = provisioner_->getBackend(targetBackendName);
  if (targetBackend.shouldStripModule() && !cctx.skipModuleStrip) {
    module->strip();
  }
  VLOG(1) << "Cleanup";
  auto sharedModule = std::shared_ptr<Module>(std::move(module));
  {
    std::unique_lock<std::shared_timed_mutex> networkLock(networkLock_);
    for (auto &node : nodeList) {
#if FACEBOOK_INTERNAL
      LOG(INFO) << "Successfully compiled and provisioned " << node.root->name;
#endif
      auto &networkData = networks_[(node.root)->name];
      networkData.dag = std::move(node);
      networkData.module = sharedModule;
    }
    cleanupAddNetwork(names);
  }
  VLOG(1) << "After cleanup";
  return Error::success();
}

#if FACEBOOK_INTERNAL
Error HostManager::addNetworkFX(
    std::unique_ptr<Module> module, CompilationContext &cctx,
    DAGListTy &networks, const folly::dynamic &FXIR,
    const llvm::StringMap<const void *> &constants) {

  LOG(INFO) << "Adding Glow network built with revision hash: " << revisionHash;
  VLOG(1) << "addNetwork";

  std::vector<std::string> names;
  {
    std::unique_lock<std::shared_timed_mutex> networkLock(networkLock_);
    auto functions = module->getFunctions();
    for (auto &F : functions) {
      const auto name = F->getName().str();
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

  // Issue a warning when loading backend specific options from the command line
  // and the compile context also contains backend specific options.
  if (!loadBackendSpecificOptionsOpt.empty()) {
    if (cctx.backendOpts.backendSpecificOpts.size() != 0) {
      VLOG_EVERY_N(1, 1000) << "Warning: backendSpecificOpts is set via the "
                               "HostManager, ignoring previously set options.";
    }
    cctx.backendOpts.backendSpecificOpts =
        deserializeStrStrMapFromYaml(loadBackendSpecificOptionsOpt);
  } else {
    auto ctxLoadBackendSpecificOpt =
        cctx.backendOpts.backendSpecificOpts.find("loadBackendSpecificOptions");

    if (ctxLoadBackendSpecificOpt !=
        cctx.backendOpts.backendSpecificOpts.end()) {
      cctx.backendOpts.backendSpecificOpts =
          deserializeStrStrMapFromYaml(ctxLoadBackendSpecificOpt->second);
    }
  }

  std::vector<DeviceInfo> deviceInfo;
  {
    std::unique_lock<std::shared_timed_mutex> networkLock(networkLock_);
    for (auto &device : availableDevices_) {
      DeviceInfo info = devices_[device]->getDeviceInfo();
      info.availableMemory = devices_[device]->getAvailableMemory();
      info.backendName = devices_[device]->getBackendName();
      info.nonSupportedNodes =
          devices_[device]->getParamByName("nonSupportedNodes");
      info.supportedNodes = devices_[device]->getParamByName("supportedNodes");
      // If p2p is enabled update the inputCount limit.
      if (cctx.enableP2P) {
        info.inputCountMax = P2PInputLimit;
      }
      deviceInfo.push_back(info);
    }
  }

  VLOG(1) << "Before provisioning";
  auto err =
      provisioner_->provisionFX(networks, *module, FXIR, constants, cctx);
  if (err) {
    if (err.peekErrorValue()->isFatalError()) {
      statsExporterRegistry_->setCounter(kDeviceFatalError, 1);
    }
    {
      std::unique_lock<std::shared_timed_mutex> networkLock(networkLock_);
      cleanupAddNetwork(names);
    }
    RETURN_ERR(err);
  }

  VLOG(1) << "Calculation of maxActiveRequests";
  {
    std::unique_lock<std::shared_timed_mutex> networkLock(networkLock_);
    /// Calculate networkMaxActive requests. Then update
    /// config_.maxActiveRequests This will be maxActiveRequestsPerInstance *
    /// instanceCount * minReplications or config_.maxActiveRequests whichever
    /// is smaller.

    // Find the minimum on device replication.
    unsigned minReplications{1};
    for (auto &node : networks) {
      for (auto &dag : node.nodes) {
        minReplications = std::min(dag->replicationCount, minReplications);
      }
    }
    unsigned product{0};
    if (networks.size() && networks[0].nodes.size()) {
      product = networks[0].nodes[0]->instanceCount *
                cctx.maxActiveRequestsPerInstance * minReplications;
    } else {
      return MAKE_ERR(ErrorValue::ErrorCode::RUNTIME_ERROR,
                      "NodeList is empty.");
    }
    unsigned maxActiveRequests = config_.maxActiveRequests;
    config_.maxActiveRequests = std::min(product, maxActiveRequests);

    // Create pool of cachedExecutionStates.
    for (auto &node : networks) {
      // Note: currently getNextNetworkExecutionState assumes that pool size is
      // >= currentInFlight requests, so we set pool size to maxActiveRequests.
      executor_->createPool(node.root.get(), config_.maxActiveRequests,
                            cctx.enableP2P, cctx.enableDRT);
    }
  }
  // Clear constants contents from the module then put it in a
  // shared_ptr to be shared between all of the networks created from each
  // function in the module.
  auto targetBackendName = std::string(devices_[0]->getBackendName());
  const auto &targetBackend = provisioner_->getBackend(targetBackendName);
  if (targetBackend.shouldStripModule() && !cctx.skipModuleStrip) {
    module->strip();
  }
  VLOG(1) << "Cleanup";
  auto sharedModule = std::shared_ptr<Module>(std::move(module));
  {
    std::unique_lock<std::shared_timed_mutex> networkLock(networkLock_);
    for (auto &node : networks) {
      LOG(INFO) << "Successfully compiled and provisioned " << node.root->name;
      auto &networkData = networks_[(node.root)->name];
      networkData.dag = std::move(node);
      networkData.module = sharedModule;
    }
    cleanupAddNetwork(names);
  }
  VLOG(1) << "After cleanup";
  return Error::success();
}
#endif

std::unordered_map<std::string, std::vector<DeviceIDTy>>
HostManager::getDevicePartitionMapping(llvm::StringRef network) {
  std::unordered_map<std::string, std::vector<DeviceIDTy>> mapping;
  auto it = networks_.find(network.str());
  if (it != networks_.end()) {
    auto &nodeList = it->second.dag.nodes;
    for (auto &node : nodeList) {
      std::vector<DeviceIDTy> devices;
      for (auto &dev : node->deviceRuntimeInfos) {
        devices.push_back(dev.first);
      }
      mapping[node->name] = devices;
    }
  }
  return mapping;
}

Error HostManager::removeNetwork(llvm::StringRef networkName) {
  std::unique_lock<std::shared_timed_mutex> networkLock(networkLock_);
  auto networkIterator = networks_.find(networkName.str());
  if (networkIterator == networks_.end()) {
    return Error::success();
  }

  if (processingNetworks_.find(networkName.str()) !=
      processingNetworks_.end()) {
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
  // Free the pool of executionStates.
  executor_->freePool(networkIterator->second.dag.root.get());
  for (auto &node : nodes) {
    for (auto device : node->deviceRuntimeInfos) {
      Error evictErr = provisioner_->evictFunction(
          node->name, devices_[device.first].get(), node->replicationCount);
      err.set(std::move(evictErr));
    }
    // Also remove compiledFunction from Provisioner.
    err.set(provisioner_->removeFunction(node->name));
  }
  networks_.erase(networkIterator);
  exportMemoryCounters();
  RETURN_ERR(err.get());
}

bool HostManager::networkAdded(llvm::StringRef networkName) {
  std::shared_lock<std::shared_timed_mutex> networkLock(networkLock_);
  return networks_.find(networkName.str()) != networks_.end();
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

  RETURN_ERR(errContainer.get());
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
    llvm::StringRef networkName, std::unique_ptr<ExecutionContext> &context) {
  std::promise<void> runPromise;
  auto fut = runPromise.get_future();
  Error runErr = Error::empty();
  std::unique_ptr<ExecutionContext> tempContext;

  runNetwork(networkName, std::move(context),
             [&runPromise, &runErr,
              &tempContext](runtime::RunIdentifierTy, Error err,
                            std::unique_ptr<ExecutionContext> resultCtxt) {
               runErr = std::move(err);
               tempContext = std::move(resultCtxt);
               runPromise.set_value();
             });

  fut.wait();
  context = std::move(tempContext);
  return runErr;
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
  auto startTime = TraceEvent::now();
  auto requestReceived = request.startTime;
  executor_->run(
      networks_[request.networkName].dag.root.get(), std::move(request.context),
      request.requestID,
      [this, callback = request.callback, name = request.networkName, startTime,
       requestReceived](RunIdentifierTy runID, Error err,
                        std::unique_ptr<ExecutionContext> context) mutable {
        {
          std::shared_lock<std::shared_timed_mutex> netLock(networkLock_);
          auto it = networks_.find(name);
          if (it != networks_.end()) {
            it->second.refcount--;
          }
        }

        updateExecutionStats(startTime, context, name, err);
        // Update request runtime.
        auto requestData = ::glow::runtime::RequestData::get();
        if (requestData) {
          uint64_t end = TraceEvent::now();
          requestData->startTime = requestReceived;
          requestData->stopTime = end;
        }

        callback(runID, std::move(err), std::move(context));
        dispatchNextRun();
      });
}

RunIdentifierTy
HostManager::runNetwork(llvm::StringRef networkName,
                        std::unique_ptr<ExecutionContext> context,
                        ResultCBTy callback, uint64_t priority) {
  DCHECK(callback != nullptr);

  TRACE_EVENT_SCOPE_NAMED(context->getTraceContext(), TraceLevel::RUNTIME,
                          "HostManager::runNetwork", traceBlock);
  auto currentRun = totalRequestCount_++;
  traceBlock.addArg("glowRequestId", llvm::formatv("{0}", currentRun).str());
  uint64_t requestReceived = TraceEvent::now();
  size_t queueSize = 0;

  NetworkData *network = nullptr;
  {
    std::shared_lock<std::shared_timed_mutex> networkLock(networkLock_);
    auto it = networks_.find(networkName.str());
    if (it != networks_.end()) {
      network = &it->second;
      network->refcount++;
    }

    if (network == nullptr) {
      TRACE_EVENT_SCOPE_END_NAMED(traceBlock);
      callback(
          currentRun,
          MAKE_ERR(ErrorValue::ErrorCode::RUNTIME_NET_NOT_FOUND,
                   llvm::formatv("Function {0} not found", networkName).str()),
          std::move(context));
      return currentRun;
    }
    // Put the request in the queue.
    {
      std::shared_lock<std::shared_timed_mutex> lock(inferQueueLock_);
      queueSize = inferQueue_.size();
      if (queueSize >= config_.maxQueueSize) {
        // The queue is full, return an error.
        network->refcount--;
        TRACE_EVENT_SCOPE_END_NAMED(traceBlock);
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
    reportCurrentQueueSize(queueSize);
    // Setup the request
    InferRequest queuedRequest(networkName.str(), std::move(context), callback,
                               priority, currentRun, requestReceived);
    {
      std::unique_lock<std::shared_timed_mutex> lock(inferQueueLock_);
      TRACE_EVENT_SCOPE_END_NAMED(traceBlock);
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

/// Helper to report current queue size
void HostManager::reportCurrentQueueSize(int32_t queueSize) {
  statsExporterRegistry_->setCounter(
      kCurrentQueueSize10k, static_cast<float>(queueSize) /
                                static_cast<float>(config_.maxQueueSize) *
                                100000);
}

/// Helper to update execution stats
void HostManager::updateExecutionStats(
    uint64_t startTime, std::unique_ptr<ExecutionContext> &context,
    llvm::StringRef networkName, const Error &error) {
  auto duration = TraceEvent::now() - startTime;
  auto updateCountersFn = [&](llvm::StringRef s) {
    statsExporterRegistry_->addTimeSeriesValue(
        ("glow.execution_duration_e2e." + s).str(), duration);
    statsExporterRegistry_->incrementCounter(
        ("glow.requests_processed." + s).str());
    if (error.peekErrorValue()) {
      statsExporterRegistry_->incrementCounter(
          ("glow.requests_failed." + s).str());
    } else {
      statsExporterRegistry_->incrementCounter(
          ("glow.requests_succeeded." + s).str());
    }
  };
  updateCountersFn(networkName);
  updateCountersFn("global");
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
    std::vector<unsigned> available_device_ids;
    if (glow::flags::ScanDevices) {
      const auto &factories =
          FactoryRegistry<std::string, Backend>::factories();
      auto it = factories.find(backendName.str());
      if (it != factories.end()) {
        available_device_ids = it->second->scanDeviceIDs();
      }
      CHECK_GE(available_device_ids.size(), 0) << "No devices found.";
      CHECK_GE(available_device_ids.size(), numDevices)
          << "Not enough devices found.";
    }
    for (unsigned int i = 0; i < numDevices; ++i) {
      auto config = glow::make_unique<runtime::DeviceConfig>(backendName);
      config->setDeviceMemory(memSize);
      if (glow::flags::ScanDevices) {
        config->deviceID = available_device_ids.back();
        available_device_ids.pop_back();
      } else {
        config->deviceID = i;
      }
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
    config->deviceID = i;
    config->setDeviceMemory(memSize);
    configs.push_back(std::move(config));
  }
  return true;
}

Backend &HostManager::getBackend(llvm::StringRef backendName) const {
  return provisioner_->getBackend(backendName);
}

Expected<Backend *> HostManager::getBackend() const {
  return provisioner_->getBackend();
}

std::unique_ptr<
    std::unordered_map<std::string, std::unique_ptr<BlockStreamBase>>>
HostManager::getAllSerializedFunctions() {
  return provisioner_->getAllSerializedFunctionsMap();
}

HostManager *HostManagerRegistry::getHostManager() { return hostManager_; }

void HostManagerRegistry::registerHostManager(HostManager *hostManager) {
  hostManager_ = hostManager;
}

std::shared_ptr<HostManagerRegistry> glow::runtime::ManagerRegistry() {
  static auto hostManager = std::make_shared<HostManagerRegistry>();
  return hostManager;
}
