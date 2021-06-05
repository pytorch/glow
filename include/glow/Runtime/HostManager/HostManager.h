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
#ifndef GLOW_RUNTIME_HOSTMANAGERR_HOSTMANAGER_H
#define GLOW_RUNTIME_HOSTMANAGERR_HOSTMANAGER_H

#include "glow/Backend/Backend.h"
#include "glow/Backends/DeviceManager.h"
#include "glow/Graph/Graph.h"
#include "glow/Runtime/Executor/Executor.h"
#include "glow/Runtime/Provisioner/Provisioner.h"
#include "glow/Runtime/RuntimeTypes.h"
#include "glow/Runtime/StatsExporter.h"

#include <atomic>
#include <map>
#include <mutex>
#include <queue>
#include <shared_mutex>
#include <unordered_map>
#include <vector>

#if FACEBOOK_INTERNAL
namespace folly {
struct dynamic;
}
#endif

namespace glow {
namespace runtime {
/// The HostManager serves as an entry point into the Runtime environment. It
/// provides an interface to add, run, and evict networks from the host. It
/// handles DeviceManager initialization, houses the Executor, and calls into
/// the Partitioner and Provisioner for network initialization.
class HostManager final {
  /// NetworkData contains data about each network in HostManager that is needed
  /// by the runtime.
  struct NetworkData {
    DAG dag{};
    // Module that was used to create this network. Everything except
    // placeholders and types have been removed from it.
    std::shared_ptr<Module> module{nullptr};

    /// use an atomic refcount rather than just store a shared_ptr for thread
    /// safety.
    std::atomic<size_t> refcount{0};
  };
  /// Container for inference requests waiting in the queue.
  struct InferRequest {
    /// Name of the network the requested run is for.
    std::string networkName;

    /// The execution context for the request.
    std::unique_ptr<ExecutionContext> context;

    /// The user provided callback to run after execution finishes.
    ResultCBTy callback;

    /// The specified priority for the run.
    uint64_t priority;

    /// The runtime generated ID for the run request.
    uint64_t requestID;

    /// Timestamp for request creation.
    uint64_t startTime;

    // Define greater than operator to allow sorting in priority_heap for queue
    // reqests. If priority is the same fall back to order of submission.
    bool operator>(const InferRequest &inferReq) const {
      if (priority == inferReq.priority) {
        return requestID > inferReq.requestID;
      }
      return priority > inferReq.priority;
    }
    InferRequest(std::string networkName,
                 std::unique_ptr<ExecutionContext> context, ResultCBTy callback,
                 uint64_t priority, uint64_t requestID, uint64_t startTime = 0)
        : networkName{networkName}, context{std::move(context)},
          callback{callback}, priority{priority}, requestID{requestID},
          startTime{startTime} {}
  };

  /// Count of current in-flight networks being run. Atomic to allow
  /// concurrency in runNetwork.
  std::atomic<size_t> activeRequestCount_{0};

  /// Count of total requests, this is used as a run ID. Atomic to allow
  /// concurrency in runNetwork.
  std::atomic<size_t> totalRequestCount_{0};

  /// Priority queue for queued requests. This is a min-heap so lowest value is
  /// popped first.
  std::priority_queue<InferRequest, std::vector<InferRequest>,
                      std::greater<InferRequest>>
      inferQueue_;

  /// Lock for the priority queue above. Please make sure whenever you want to
  /// access inferQueue_, you take a lock. Usage is the same as
  /// std::shared_mutex
  std::shared_timed_mutex inferQueueLock_;

  /// Configuration parameters for this Runtime Host.
  HostConfig config_{};

  std::unique_ptr<TraceContext> hostTraceContext_;

  /// A map from a networkName to a network, which is represented by struct DAG.
  std::unordered_map<std::string, NetworkData> networks_;

  /// Mutex for networks_ since runNetwork, addNetwork, and
  /// removeNetwork can all be called concurrently, a guard is needed.
  std::shared_timed_mutex networkLock_;

  /// A map of DeviceManagers by deviceID. An ordered map is used here to allow
  /// a stable iteration order over devices.
  DeviceManagerMapTy devices_;

  /// A vector of devices available for new networks to be added to.
  std::vector<DeviceIDTy> availableDevices_;

  /// A single threaded threadpool used by init() when initializing devices.
  ThreadPool threadPool_{1};

  /// Executor class, this handles dispatching execution requests to the
  /// appropriate device managers for an inference request.
  std::unique_ptr<Executor> executor_;

  /// The provisioner owns the compiledFunctions and handles loading functions
  /// onto the devices.
  std::unique_ptr<Provisioner> provisioner_;

  /// String const for logging max queue size in glow
  static constexpr const char *kMaxQueueSize = "glow.queue.max.size";

  /// String const for logging total device memory usage.
  static constexpr const char *kDeviceMemoryUsed =
      "glow.devices.used_memory.total";

  /// String const for logging total available device memory.
  static constexpr const char *kDeviceMemoryAvailable =
      "glow.devices.available_memory.total";

  /// String const for logging total maximum device memory.
  static constexpr const char *kDeviceMemoryMax =
      "glow.devices.maximum_memory.total";

  /// String const for logging device fatal errors.
  static constexpr const char *kDeviceFatalError =
      "glow.devices.fatal_compilation_error";

  /// Helper function to handle cleanup if an error occurs during addNetwork.
  /// This must be called while holding the a lock on networkLock_.
  void cleanupAddNetwork(llvm::ArrayRef<std::string> names);

  /// Set of networks in the process of being added.
  std::set<std::string> processingNetworks_;

  /// Method to dispatch a new run to the executor.
  void dispatchNextRun();

  /// Method to calculate and export aggregate memory usage counters.
  void exportMemoryCounters();

  /// Queue size stat update
  void reportCurrentQueueSize(int32_t queueSize);

  /// Execution stats update.
  void updateExecutionStats(uint64_t startTime,
                            std::unique_ptr<ExecutionContext> &context,
                            llvm::StringRef name, const Error &error);

  /// Keeps the stats exporter registry object alive till destructor.
  std::shared_ptr<StatsExporterRegistry> statsExporterRegistry_;

  /// Default constructor.
  HostManager();

public:
  /// Constructor that takes configuration options.
  HostManager(const HostConfig &hostConfig);

  /// Constructor that takes a list of Devices to use.
  HostManager(std::vector<std::unique_ptr<DeviceConfig>> deviceConfigs);

  /// Constructor that takes both Devices and the configuration.
  HostManager(std::vector<std::unique_ptr<DeviceConfig>> deviceConfigs,
              const HostConfig &hostConfig);

  /// Adds the network to the host and does the necessary setup work. This
  /// includes partitioning, provisioning, compiling and initializing
  /// backends. Additionally DAGs are created for each function and stored in
  /// networks_. \returns an Error containing the results of the
  /// operation. This function consumes the \p module so any pointers to data
  /// contained within the module should be considered invalid. The function is
  /// optimized based on \p cctx.
  Error addNetwork(std::unique_ptr<Module> module, CompilationContext &cctx);

/// Adds the already partitioned FX \p FXIR network to the host and does the
/// necessary setup work. This includes provisioning, compiling and
/// initializing backends. Requires a  DAG \p networks to be provided.
/// \returns an Error containing the results of the operation. This function
/// consumes the \p module so any pointers to data contained within the module
/// should be considered invalid. The function is optimized based on \p cctx.
/// Constants are provided with a stringmap \p constants.
#if FACEBOOK_INTERNAL
  Error addNetworkFX(std::unique_ptr<Module> module, CompilationContext &cctx,
                     DAGListTy &networks, const folly::dynamic &FXIR,
                     const llvm::StringMap<const void *> &constants);
#endif

  /// Given \p networkName removes that network from the host. This also
  /// removes the network from any backends setup to execute it.
  /// \returns an Error indicating success or failure of the operation.
  Error removeNetwork(llvm::StringRef networkName);

  /// Update the list of available devices.
  void setAvailableDevices(const std::vector<DeviceIDTy> &devices);

  /// Returns a string map containing the name and block-stream for all
  /// serialized functions.
  std::unique_ptr<
      std::unordered_map<std::string, std::unique_ptr<BlockStreamBase>>>
  getAllSerializedFunctions();

  /// For a given \p network returns all partitions of that network and the
  /// devices each partition is assigned to.
  std::unordered_map<std::string, std::vector<DeviceIDTy>>
  getDevicePartitionMapping(llvm::StringRef network);

  /// Returns true if \p networkName is already added to the host.
  bool networkAdded(llvm::StringRef networkName);

  /// Removes all networks from the host, and stops execution on all devices.
  Error clearHost();

  /// Runs the network specified by \p networkName using
  /// the provided \p context, returns a runIdentifier which refers to the
  /// specic inference request. Calls \p callback with the results when
  /// inference is done.
  /// Note: This method is intended to be thread-safe, it will be called
  /// concurrently from multiple threads.
  /// Returns -1 if networkName not found or too many active requests.
  /// The parameter \p priority is used to indicate queueing priority, priority
  /// is lowest number first and in case of a tie the request that was submitted
  /// first will go first.
  RunIdentifierTy runNetwork(llvm::StringRef networkName,
                             std::unique_ptr<ExecutionContext> context,
                             ResultCBTy callback, uint64_t priority = 0);

  /// A wrapper around runNetwork that provides a blocking interface for an
  /// inference request. Runs the network provided in \p networkName using \p
  /// context. \returns an Error indicating success or failure. Upon return,
  /// regardless of success or failure, \p context will be filled with the
  /// return context from running the network.
  Error runNetworkBlocking(llvm::StringRef networkName,
                           std::unique_ptr<ExecutionContext> &context);

  /// A wrapper around runNetwork that provides a blocking interface for an
  /// inference request. Runs the network provided in \p networkName using \p
  /// bindings for placeholder bindings. \returns an Error indicating
  /// success or failure.
  Error runNetworkBlocking(llvm::StringRef networkName,
                           PlaceholderBindings &bindings);

  /// Initialize the HostManager with the given \p configs creating one
  /// DeviceManager for each config listed.
  Error init(std::vector<std::unique_ptr<DeviceConfig>> configs);

  /// Get the network DAG for \p network if it exists.
  Expected<DAG *> getNetworkDAG(llvm::StringRef network);

  /// \returns a non-owning pointer to the TraceContext.
  TraceContext *getTraceContext() { return hostTraceContext_.get(); }

  /// Sets the TraceContext and \returns the existing value.
  std::unique_ptr<TraceContext>
  setTraceContext(std::unique_ptr<TraceContext> traceContext) {
    std::swap(hostTraceContext_, traceContext);
    return traceContext;
  }

  /// Triggers start tracing of all active devices \returns Error if fails.
  Error startDeviceTrace();

  /// Triggers stop tracing of all active devices \returns Error if fails.
  Error stopDeviceTrace();

  /// \returns a reference to the backend with name \p backendName owned by the
  /// Provisioner.
  Backend &getBackend(llvm::StringRef backendName) const;

  /// \returns a reference to the Backend if only one Backend is found,
  /// otherwise returns an Error.
  Expected<Backend *> getBackend() const;

  /// \returns the number of devices the HostManager owns.
  size_t numDevices() const { return devices_.size(); }

  ~HostManager();

  /// String const for logging current queue size in glow
  static constexpr const char *kCurrentQueueSize10k =
      "glow.queue.current.occupancy.10k";
};

/// If the device config file specified in loadDeviceConfigsFileOpt is
/// available, load \p configs from the file. Otherwise, create \p numDevices
/// number of devices based on \p backendName.
std::vector<std::unique_ptr<runtime::DeviceConfig>>
generateDeviceConfigs(unsigned int numDevices, llvm::StringRef backendName,
                      size_t memSize = 0);

/// Attempts to load user-specified DeviceConfigs file
/// \ref loadDeviceConfigsFileOpt. If the path exists then \p configs will be
/// loaded with DeviceConfigs given that file and \p memSize, and the function
/// \returns true. Otherwise \returns false with \p configs untouched.
bool loadDeviceConfigsFromFile(
    std::vector<std::unique_ptr<runtime::DeviceConfig>> &configs,
    size_t memSize);

/// Registry singleton for aquiring a HostManager.
class HostManagerRegistry final {
public:
  void registerHostManager(HostManager *hostManager);
  HostManager *getHostManager();

private:
  HostManager *hostManager_{nullptr};
};

/// Global singleton.
std::shared_ptr<HostManagerRegistry> ManagerRegistry();

} // namespace runtime
} // namespace glow
#endif // GLOW_RUNTIME_HOSTMANAGERR_HOSTMANAGER_H
