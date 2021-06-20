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

// Silence Apple's warning about the deprecation of OpenCL.
#define CL_SILENCE_DEPRECATION

// Silence warnings about using deprecated OpenCL 1.2 functions.
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include "OpenCLDeviceManager.h"
#include "OpenCL.h"

#include "glow/Runtime/StatsExporter.h"
#include "glow/Support/Debug.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "opencl"

using namespace glow;
using namespace glow::runtime;

// This defines kernels for most operations.
static const unsigned char kernels_cl_src[] = {
#include "glow/OpenCL/kernels.cl.inc"
};
static const size_t kernels_cl_src_size = sizeof(kernels_cl_src);

extern llvm::cl::opt<unsigned> clPlatformId;
extern llvm::cl::opt<int> clDeviceId;
extern llvm::cl::opt<bool> clDoProfile;

namespace glow {
namespace runtime {

DeviceManager *createOCLDeviceManager(const DeviceConfig &config) {
  return new OpenCLDeviceManager(config);
}

OpenCLBuffer::~OpenCLBuffer() { clReleaseMemObject(buffer_); }
} // namespace runtime
} // namespace glow

/// Helper method to parse a string parameter to an unsigned. \returns
/// Expected with either the value or an error.
static Expected<unsigned> parseInputAsUnsigned(std::string input) {
  char *end;
  auto parsed = strtol(input.c_str(), &end, 10);
  if (end == input.c_str() || *end != '\0') {
    return MAKE_ERR(ErrorValue::ErrorCode::RUNTIME_ERROR,
                    "Invalid input expected integer got: " + input);
  }
  return parsed;
}

OpenCLCommandQueuePool::~OpenCLCommandQueuePool() {
  // Make sure all queues have been returned to the pool.
  DCHECK_EQ(queuesAllocated_, queuesAvailable_)
      << "OpenCLCommandQueue destroyed before all queues returned!";
  // For each properties -> vector of queues pair:
  for (auto &kv : queues_) {
    // For each queue in each vector:
    for (auto &q : kv.second) {
      // Release the backing queue.
      cl_int err = clReleaseCommandQueue(q.backingQueue);
      DCHECK_EQ(err, CL_SUCCESS)
          << "clReleaseCommandQueue failed with error code " << err;
    }
  }
}

Expected<OpenCLCommandQueue> OpenCLCommandQueuePool::requestCommandQueue(
    cl_command_queue_properties properties) {
  OpenCLCommandQueue ret;
  // Get the vector that has queues with the desired properties.
  std::vector<OpenCLCommandQueue> &srcVec = queues_[properties];

  if (srcVec.empty()) {
    // The vector is empty. This means a new queus must be created.
    cl_int err;
    ret.props = properties;
    ret.backingQueue =
        clCreateCommandQueue(context_, device_, properties, &err);
    RETURN_ERR_IF_NOT(err == CL_SUCCESS,
                      strFormat("Unable to create command queue: %d", err));

    // If queue creation succeeds, increment the total number of queues. Do not
    // increment the number of available queues since the newly created one is
    // about to be given away.
    ++queuesAllocated_;
    ++queuesAllocatedByProps_[properties];
  } else {
    ret = srcVec.back();
    srcVec.pop_back();
    --queuesAvailable_;
    --queuesAvailableByProps_[properties];
  }

  return ret;
}

void OpenCLCommandQueuePool::returnCommandQueue(OpenCLCommandQueue &queue) {
  // Check that the number of available queues is less than the number of
  // allocated queues.
  DCHECK_LE(queuesAvailable_, queuesAllocated_)
      << "Available queues must be less than allocated queues";

  // Get the vector that has queues with the desired properties.
  std::vector<OpenCLCommandQueue> &destVec = queues_[queue.props];
  ++queuesAvailable_;
  ++queuesAvailableByProps_[queue.props];
  destVec.emplace_back(std::move(queue));
}

unsigned OpenCLCommandQueuePool::getNumAllocatedQueuesForProperties(
    cl_command_queue_properties props) const {
  auto it = queuesAllocatedByProps_.find(props);
  return it != queuesAllocatedByProps_.end() ? it->second : 0;
}

unsigned OpenCLCommandQueuePool::getNumQueuesAvailableForProperties(
    cl_command_queue_properties props) const {
  auto it = queuesAvailableByProps_.find(props);
  return it != queuesAvailableByProps_.end() ? it->second : 0;
}

Expected<cl_mem> OpenCLDeviceManager::allocDeviceBuffer(uint64_t size) {
  const uint64_t alignment = 128;
  // Always allocate buffers properly aligned to hold values of any type.
  size = alignedSize(size, alignment);
  auto buf =
      clCreateBuffer(context_, CL_MEM_READ_WRITE, size, nullptr, nullptr);
  RETURN_ERR_IF_NOT(buf, "Allocation failed!");
  return buf;
}
OpenCLDeviceManager::OpenCLDeviceManager(const DeviceConfig &config)
    : QueueBackedDeviceManager(config) {}

Error OpenCLDeviceManager::parseConfig() {
  auto it = config_.parameters.find("deviceId");
  unsigned value{0};
  if (it != config_.parameters.end()) {
    ASSIGN_VALUE_OR_RETURN_ERR(value, parseInputAsUnsigned(it->second));
    clDeviceId = value;
  }
  it = config_.parameters.find("platformId");
  if (it != config_.parameters.end()) {
    ASSIGN_VALUE_OR_RETURN_ERR(value, parseInputAsUnsigned(it->second));
    clPlatformId = value;
  }
  it = config_.parameters.find("doProfile");
  if (it != config_.parameters.end()) {
    if (it->second == "true") {
      clDoProfile = true;
    } else if (it->second == "false") {
      clDoProfile = false;
    } else {
      return MAKE_ERR(ErrorValue::ErrorCode::RUNTIME_ERROR,
                      "Invalid input expected true or false got: " +
                          it->second);
    }
  }
  return Error::success();
}

Error OpenCLDeviceManager::findBestDevice(cl_platform_id platformId,
                                          int deviceId) {
  cl_uint num{0};
  cl_int err = clGetDeviceIDs(platformId, CL_DEVICE_TYPE_ALL, 0, nullptr, &num);
  if (err != CL_SUCCESS) {
    return MAKE_ERR("clGetDeviceIDs Failed");
  }
  if ((deviceId > 0 && num < deviceId) || num == 0) {
    return MAKE_ERR("Should have at least one GPU/CPU/FPGA for running OpenCL");
  }

  // Enumerate all available devices.
  std::vector<cl_device_id> devices(num);
  err = clGetDeviceIDs(platformId, CL_DEVICE_TYPE_ALL, num, devices.data(),
                       nullptr);
  if (err != CL_SUCCESS) {
    return MAKE_ERR("clGetDeviceIDs Failed");
  }

  // If the deviceId was set on the command line, use that.
  if (deviceId >= 0) {
    deviceId_ = devices[deviceId];
  } else {
    cl_device_id chosen = devices[0];
    cl_device_type chosen_type = 0;

    // Otherwise loop through all devices.
    for (auto id : devices) {
      cl_bitfield type;
      // Get the device type (bitmask of 1 for cpu, 2 for gpu, 4 for
      // accelerator).
      err = clGetDeviceInfo(id, CL_DEVICE_TYPE, sizeof(cl_bitfield), &type,
                            nullptr);
      if (err != CL_SUCCESS) {
        return MAKE_ERR("clGetDeviceInfo Failed");
      }

      // prefer the highest device type, and tiebreak on the the highest device
      // id.
      if (type >= chosen_type) {
        chosen = id;
        chosen_type = type;
      }
    }

    deviceId_ = chosen;
  }

  char name[100];
  err = clGetDeviceInfo(deviceId_, CL_DEVICE_NAME, 100, &name, nullptr);
  DEBUG_GLOW(llvm::dbgs() << "Using OpenCL device " << name << "\n");

  return Error::success();
}

Error OpenCLDeviceManager::init() {
  // The OpenCL Backend defines three command line options: doProfile, deviceId,
  // and platformId. If the parameter is not provided we use the CL
  // options from the OpenCl Backend.

  // Check if parameters are in map.
  RETURN_IF_ERR(parseConfig());

  cl_uint numPlatforms{0};
  cl_int err = clGetPlatformIDs(0, NULL, &numPlatforms);
  if (err != CL_SUCCESS) {
    return MAKE_ERR("clGetPlatformIDs Failed.");
  }

  if (numPlatforms < clPlatformId) {
    return MAKE_ERR("Should have at least one platform for running OpenCL");
  }

  std::vector<cl_platform_id> platform_ids(numPlatforms);
  err = clGetPlatformIDs(numPlatforms, platform_ids.data(), NULL);

  cl_platform_id platform_id_used = platform_ids[clPlatformId];

  Error deviceErr = findBestDevice(platform_id_used, clDeviceId);
  if (deviceErr) {
    return deviceErr;
  }
  context_ = clCreateContext(nullptr, 1, &deviceId_, nullptr, nullptr, nullptr);
  if (!context_) {
    return MAKE_ERR("clCreateContext Failed");
  }

  cl_ulong mem_size;
  err = clGetDeviceInfo(deviceId_, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong),
                        &mem_size, NULL);
  if (err != CL_SUCCESS) {
    return MAKE_ERR("Error getting device memory limit");
  }

  // If limited by deviceConfig, should allow less deviceMemory
  if (config_.getDeviceMemory() != 0 && config_.getDeviceMemory() < mem_size) {
    maxMemoryBytes_ = config_.getDeviceMemory();
  } else {
    maxMemoryBytes_ = mem_size;
  }

  localMemSize_ = 0;
  cl_device_local_mem_type localMemType;
  err = clGetDeviceInfo(deviceId_, CL_DEVICE_LOCAL_MEM_TYPE,
                        sizeof(localMemType), &localMemType, NULL);
  if (err != CL_SUCCESS) {
    return MAKE_ERR("Error getting device local memory type.");
  }
  if (localMemType == CL_LOCAL) {
    cl_ulong localMemSize;
    err = clGetDeviceInfo(deviceId_, CL_DEVICE_LOCAL_MEM_SIZE,
                          sizeof(localMemSize), &localMemSize, NULL);
    if (err != CL_SUCCESS) {
      return MAKE_ERR("Error getting device local memory type.");
    }
    localMemSize_ = localMemSize;
  }

  commandQueuePool_.setContext(context_);
  commandQueuePool_.setDevice(deviceId_);

  statsExporterRegistry_->incrementCounter(kDevicesUsedOpenCL);
  exportMemoryCounters();

  threads::getThreadId();
  workThread_.submit([this] {
    /// Prime thread ids for this device.
    threads::getThreadId();
    /// It looks nicer if the host thread is before the device thread, so
    /// prevent reordering.
    std::atomic_signal_fence(std::memory_order_seq_cst);
    deviceTid_ = threads::createThreadId();
  });
  return Error::success();
}

OpenCLDeviceManager::~OpenCLDeviceManager() {
  clReleaseContext(context_);
  buffers_.clear();
  statsExporterRegistry_->incrementCounter(kDevicesUsedOpenCL, -1);
  zeroMemoryCounters();
}

uint64_t OpenCLDeviceManager::getMaximumMemory() const {
  return maxMemoryBytes_;
}

uint64_t OpenCLDeviceManager::getAvailableMemory() const {
  return maxMemoryBytes_ - usedMemoryBytes_;
}

bool OpenCLDeviceManager::isMemoryAvailable(uint64_t estimate) const {
  return maxMemoryBytes_ >= (usedMemoryBytes_ + estimate);
}

void OpenCLDeviceManager::addNetworkImpl(const Module *module,
                                         FunctionMapTy functions,
                                         ReadyCBTy readyCB) {
  DCHECK(readyCB != nullptr);

  // First check for uniqueness of the function name.
  for (const auto &func : functions) {
    if (functions_.count(func.first) != 0) {
      readyCB(
          module,
          MAKE_ERR(
              llvm::formatv(
                  "Failed to add network: already have a function called {}",
                  func.first)
                  .str()));
      return;
    }

    if (func.second->getCompileBackendName() != "OpenCL") {
      readyCB(module,
              MAKE_ERR(llvm::formatv("Failed to add network: function {} is "
                                     "not a OpenCL Function",
                                     func.first)
                           .str()));
    }

    auto &bundle = func.second->getRuntimeBundle();
    if (bundle.getConstants() == nullptr) {
      bundle.collectConstants(module);
    }
    size_t sizeInBytes = bundle.getConstantWeightSize();
    if (usedMemoryBytes_ + sizeInBytes > maxMemoryBytes_) {
      // Free the constants.
      bundle.freeConstants();
      readyCB(module,
              MAKE_ERR(ErrorValue::ErrorCode::RUNTIME_OUT_OF_DEVICE_MEMORY,
                       "Failed to add network: not enough memory"));
      return;
    }

    // Create a command queue to copy constants to the device and compile the
    // function.
    cl_int err;
    cl_command_queue commands =
        clCreateCommandQueue(context_, deviceId_, 0, &err);
    if (!commands) {
      readyCB(module,
              MAKE_ERR(
                  ErrorValue::ErrorCode::RUNTIME_OUT_OF_DEVICE_MEMORY,
                  "Failed to add network: could not create CL command queue."));
      return;
    }

    // Copy constants to device.
    auto size = bundle.getConstantWeightSize() + bundle.getMutableWeightSize() +
                bundle.getActivationsSize();
    cl_mem deviceBuffer;
    if (auto autoDeviceBufferOrErr = allocDeviceBuffer(size)) {
      deviceBuffer = *autoDeviceBufferOrErr;
    } else {
      readyCB(module, autoDeviceBufferOrErr.takeError());
      return;
    }

    auto buffer = std::make_shared<OpenCLBuffer>(deviceBuffer, size);
    if (bundle.getConstants()) {
      auto buf = bundle.getConstants();
      size_t valueOffset = 0;
      cl_event event{nullptr};
      err = clEnqueueWriteBuffer(
          commands, buffer->getBuffer(), /* blocking_write */ CL_FALSE,
          valueOffset, sizeInBytes, buf, /* num_events_in_wait_list */ 0,
          /* event_list */ nullptr,
          /* event */ doProfile_ ? &event : nullptr);
      if (err != CL_SUCCESS) {
        readyCB(module, MAKE_ERR("Unable to copy data to the device"));
        return;
      }
      clFinish(commands);
    }
    usedMemoryBytes_ += sizeInBytes;
    // Compile the CL program.
    // Add to the function name lookup map.
    // Add shared pointer to the buffer to buffers. This way the buffer will
    // be freed after the last reference is removed.

    // Configure the kernels by providing the size of size_t on the host size.
    // This is required to e.g. properly pass struct parameters of types like
    // ShapeNHWC, ShapeNCHW, etc. The definitions of these types on the host
    // side use size_t for their members and they should be defined on the
    // OpenCL's side using integer types of the same width.
    std::vector<std::string> options;
    addIntOption(options, "SIZEOF_HOST_SIZE_T", sizeof(size_t));
    // Create the program from the source.
    std::string source(reinterpret_cast<const char *>(kernels_cl_src),
                       kernels_cl_src_size);
    OpenCLFunction *function = static_cast<OpenCLFunction *>(func.second);
    auto program = function->createProgram(source, options, commands);
    programs_.emplace(func.first, program);
    functions_.emplace(func.first, func.second);
    buffers_.emplace(func.first, buffer);
    buffer->incrementUsers();

    // Add function name to map for static placeholders.
    for (auto PH : function->getIR()->getGraph()->findPlaceholders()) {
      if (PH->isStatic()) {
        staticPlaceholderToFunctions_[PH].push_back(func.first);
      }
    }

    DCHECK_LE(usedMemoryBytes_, maxMemoryBytes_);
    clReleaseCommandQueue(commands);
  }

  // Export change in memory usage.
  exportMemoryCounters();

  // Fire the ready CB.
  readyCB(module, Error::success());
}

void OpenCLDeviceManager::evictNetworkImpl(std::string functionName,
                                           EvictFunctionCBTy evictCB) {
  DCHECK(evictCB != nullptr);

  if (functions_.erase(functionName)) {
    auto buffer = buffers_[functionName];
    auto users = buffer->decrementUsers();
    auto size = buffer->getSize();
    buffers_.erase(functionName);
    if (users == 0) {
      DCHECK_GE(usedMemoryBytes_, size);
      usedMemoryBytes_ -= size;
    }
  } else {
    evictCB(functionName,
            MAKE_ERR(ErrorValue::ErrorCode::RUNTIME_NET_NOT_FOUND,
                     strFormat("Could not find function with name %s to evict",
                               functionName.c_str())));
    return;
  }
  exportMemoryCounters();
  evictCB(functionName, Error::success());
}

Expected<OpenCLCommandQueue>
OpenCLDeviceManager::requestRunCommandQueue(CompiledFunction *function) {
  auto traceInfo = function->getTraceInfo();
  cl_command_queue_properties props =
      clDoProfile || traceInfo.enabled ? CL_QUEUE_PROFILING_ENABLE : 0;
  return commandQueuePool_.requestCommandQueue(props);
}

void OpenCLDeviceManager::returnRunCommandQueue(OpenCLCommandQueue &queue) {
  commandQueuePool_.returnCommandQueue(queue);
}

void OpenCLDeviceManager::transferStaticPlaceholderToDevice(
    Placeholder *PH, Tensor *T, std::function<void(Error)> resultCB) {
  auto it = staticPlaceholderToFunctions_.find(PH);
  if (it == staticPlaceholderToFunctions_.end()) {
    resultCB(MAKE_ERR(
        ErrorValue::ErrorCode::RUNTIME_ERROR,
        llvm::formatv("Unable to transfer PH: {0}", PH->getName()).str()));
    return;
  }
  for (auto functionName : it->second) {
    // Tranfer for each function that needs PH.
    auto buffer = buffers_[functionName]->getBuffer();

    auto funcIt = functions_.find(functionName);
    if (funcIt == functions_.end()) {
      resultCB(
          MAKE_ERR(ErrorValue::ErrorCode::RUNTIME_NET_NOT_FOUND,
                   llvm::formatv("Function {} not found", functionName).str()));
      return;
    }

    OpenCLFunction *func = static_cast<OpenCLFunction *>(funcIt->second);

    OpenCLCommandQueue queue;
    auto queueOrError = requestRunCommandQueue(func);

    if (queueOrError) {
      queue = std::move(queueOrError.get());
    } else {
      resultCB(queueOrError.takeError());
      return;
    }
    auto symbolTable = func->getRuntimeBundle().getSymbolTable();
    auto symbolIt = symbolTable.find(PH->getName().str());
    if (symbolIt == symbolTable.end()) {
      resultCB(
          MAKE_ERR(ErrorValue::ErrorCode::RUNTIME_ERROR,
                   llvm::formatv("Symbol {} not found", PH->getName()).str()));
      return;
    }
    auto offset = symbolIt->second.offset;
    auto size = symbolIt->second.size;

    // Issue a non-blocking command to copy the buffer to the device.
    cl_int err = clEnqueueWriteBuffer(queue.backingQueue, buffer,
                                      /* blocking_write */ CL_FALSE, offset,
                                      size, T->getUnsafePtr(),
                                      /* num_events_in_wait_list */ 0,
                                      /* event_list */ nullptr,
                                      /* event */ nullptr);
    if (err != CL_SUCCESS) {
      resultCB(MAKE_ERR(
          ErrorValue::ErrorCode::RUNTIME_ERROR,
          llvm::formatv("Copying Symbol: {} to device failed.", PH->getName())
              .str()));
    }
    clFinish(queue.backingQueue);
    returnRunCommandQueue(queue);
  }

  resultCB(Error::success());
}

void OpenCLDeviceManager::copyInputsToDevice(
    const RuntimeBundle &runtimeBundle, ExecutionContext *context,
    runtime::OpenCLDeviceBindings *devBindings, bool traceEnabled) {
  TRACE_EVENT_SCOPE(context->getTraceContext(), TraceLevel::RUNTIME,
                    "copyInputsToDevice");

  bool profilingEnabled =
      traceEnabled && context->getTraceContext() &&
      (context->getTraceContext()->getTraceLevel() & TraceLevel::COPY);

  auto &symbolTable = runtimeBundle.getSymbolTable();
  for (auto &PH : context->getPlaceholderBindings()->pairs()) {
    auto it = symbolTable.find(PH.first->getName().str());
    if (it == symbolTable.end()) {
      continue;
    }
    // If the PH is marked as static do not copy it.
    if (PH.first->isStatic()) {
      continue;
    }
    auto symbolInfo = it->second;
    auto addr = symbolInfo.offset;
    auto numBytes = PH.second.getUnpaddedSizeInBytes();
    // Issue a non-blocking command to copy the buffer to the device.
    auto buf = PH.second.getUnsafePtr();
    cl_event event{nullptr};

    cl_int err = clEnqueueWriteBuffer(
        devBindings->commandQueue, devBindings->deviceBuffer,
        /* blocking_write */ CL_FALSE, addr, numBytes, buf,
        /* num_events_in_wait_list */ 0,
        /* event_list */ nullptr,
        /* event */ profilingEnabled ? &event : nullptr);
    CHECK_EQ(err, CL_SUCCESS) << "Unable to copy data to the device";
    if (profilingEnabled) {
      std::string name = ("(H2D) " + PH.first->getName()).str();
      devBindings->kernelLaunches.emplace_back(name, "copy", event);
    }
  }

  // Do it!
  clFinish(devBindings->commandQueue);
}

void OpenCLDeviceManager::copyOutputsFromDevice(
    const RuntimeBundle &runtimeBundle, ExecutionContext *context,
    runtime::OpenCLDeviceBindings *devBindings, bool traceEnabled) {
  TRACE_EVENT_SCOPE(context->getTraceContext(), TraceLevel::RUNTIME,
                    "copyOutputsFromDevice");

  bool profilingEnabled =
      traceEnabled && context->getTraceContext() &&
      (context->getTraceContext()->getTraceLevel() & TraceLevel::COPY);

  auto &symbolTable = runtimeBundle.getSymbolTable();
  for (auto &PH : context->getPlaceholderBindings()->pairs()) {
    auto it = symbolTable.find(PH.first->getName().str());
    if (it == symbolTable.end()) {
      continue;
    }
    auto symbolInfo = it->second;
    auto addr = symbolInfo.offset;
    auto numBytes = PH.second.getUnpaddedSizeInBytes();
    // Issue a non-blocking command to copy the buffer to the device.
    auto buf = PH.second.getUnsafePtr();
    cl_event event{nullptr};

    cl_int err = clEnqueueReadBuffer(
        devBindings->commandQueue, devBindings->deviceBuffer,
        /* blocking_read */ CL_FALSE, addr, numBytes, buf,
        /* num_events_in_wait_list */ 0,
        /* event_list */ nullptr,
        /* event */ profilingEnabled ? &event : nullptr);
    CHECK_EQ(err, CL_SUCCESS) << "Unable to copy data from the device";
    if (profilingEnabled) {
      std::string name = ("(D2H) " + PH.first->getName()).str();
      devBindings->kernelLaunches.emplace_back(name, "copy", event);
    }
  }

  // Do it!
  clFinish(devBindings->commandQueue);
}

void OpenCLDeviceManager::translateTraceEvents(
    ManualEventMap &manualTraceEvents, ExecutionContext *context,
    runtime::OpenCLDeviceBindings *devBindings) {
  if (context->getTraceContext() == nullptr) {
    return;
  }

  TRACE_EVENT_SCOPE(context->getTraceContext(), TraceLevel::RUNTIME,
                    "processInstrumentation");
  cl_ulong total = 0;

  context->getTraceContext()->setThreadName("OCL DeviceManager");
  context->getTraceContext()->setThreadName(deviceTid_, "OCL Device");

  // The device uses a different clock domain, so we'll assume that the last
  // timestamp and now are close and get the difference between the two
  // timestamps, which we can use to pull event timestamps in to the
  // steady_clock domain.
  // TODO: synchronize clocks better, this can be off the thread was yielded
  // since getting the timestamp in updatePlaceholders.
  int64_t tsOffset = std::chrono::duration_cast<std::chrono::microseconds>(
                         std::chrono::steady_clock().now().time_since_epoch())
                         .count();

  if (!devBindings->kernelLaunches.empty()) {
    auto &event = devBindings->kernelLaunches.back().event_;
    cl_ulong time_end;
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end),
                            &time_end, NULL);

    // Get the difference between the last event's end and the tsOffset
    // timestamp.
    tsOffset -= (time_end / 1000);
  }

  std::unordered_map<std::string, cl_ulong> kernelToDuration;
  auto &traceEvents = context->getTraceContext()->getTraceEvents();
  std::vector<cl_ulong> manualTimestamps;

  for (auto &kl : devBindings->kernelLaunches) {
    auto &event = kl.event_;
    if (event == nullptr) {
      continue;
    }
    clWaitForEvents(1, &event);

    auto &name = kl.name_;
    auto &type = kl.type_;
    DCHECK(!name.empty()) << "Kernel name cannot be empty";
    cl_ulong timeStart;
    cl_ulong timeEnd;

    if (clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START,
                                sizeof(timeStart), &timeStart,
                                NULL) != CL_SUCCESS) {
      continue;
    }
    if (clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END,
                                sizeof(timeEnd), &timeEnd,
                                NULL) != CL_SUCCESS) {
      continue;
    }

    if (type == "checkpoint") {
      const auto &it = manualTraceEvents.find(name);
      if (it == manualTraceEvents.end()) {
        DEBUG_GLOW(llvm::dbgs() << "warning: found manual trace event (" << name
                                << ") with no metadata (OCL)\n");
      } else {
        auto handle = context->getPlaceholderBindings()
                          ->get(it->second.first)
                          ->getHandle<int64_t>();
        const TraceInfo::Event *ev = it->second.second;

        // Convert into usec and move into steady_clock domain.
        auto timestamp = (timeEnd / 1000) + tsOffset;

        handle.at({(dim_t)ev->startIndex, 0}) = timestamp;
        traceEvents.push_back({ev->name,
                               TraceLevel::DEBUG,
                               timestamp,
                               ev->type,
                               deviceTid_,
                               {{"kind", ev->kind}}});
      }
    } else {
      // Duration should be usec.
      auto duration = (timeEnd - timeStart) / 1000;
      // Convert into usec and move into steady_clock domain.
      auto startUs = (timeStart / 1000) + tsOffset;

      TraceLevel level =
          (type == "copy") ? TraceLevel::COPY : TraceLevel::OPERATOR;

      traceEvents.push_back(
          {name, level, startUs, duration, deviceTid_, {{"type", type}}});
    }

    if (clDoProfile) {
      // Duration (in nanoseconds).
      double duration = timeEnd - timeStart;
      kernelToDuration[type] += duration;
      total += duration;
      LOG(INFO) << "OpenCl execution time for a launch of kernel " << type
                << strFormat(" is: %0.3f milliseconds\n", duration / 1000000.0);
    }
  }

  if (!clDoProfile) {
    return;
  }
  llvm::outs() << llvm::format(
      "OpenCl total execution time is: %0.3f milliseconds \n",
      total / 1000000.0);

  // Build a sorted list of kernel durations.
  std::vector<std::pair<cl_ulong, std::string>> sortedKernelDurations;
  sortedKernelDurations.reserve(kernelToDuration.size());
  for (auto kv : kernelToDuration) {
    sortedKernelDurations.push_back(std::make_pair(kv.second, kv.first));
  }
  std::sort(sortedKernelDurations.begin(), sortedKernelDurations.end());

  llvm::outs() << "\n\nSummary information per kernel:\n";
  for (auto k : sortedKernelDurations) {
    llvm::outs() << "OpenCl total execution time for kernel " << k.second
                 << llvm::format(" is: %0.3f milliseconds (%lu%%)\n",
                                 k.first / 1000000.0,
                                 (unsigned long)(k.first * 100 / total));
  }
}

void OpenCLDeviceManager::runFunctionImpl(
    RunIdentifierTy id, std::string function,
    std::unique_ptr<ExecutionContext> context, ResultCBTy resultCB) {
  DCHECK(resultCB != nullptr);
  TRACE_EVENT_SCOPE_NAMED(context->getTraceContext(), TraceLevel::RUNTIME,
                          "DeviceManager::run", dmRun);
  /// OpenCL DeviceManager doesn't support Device Resident Tensors.
  context->getPlaceholderBindings()->ensureOnHost();

  auto funcIt = functions_.find(function);
  if (funcIt == functions_.end()) {
    dmRun.addArg("reason", "function not found");
    TRACE_EVENT_SCOPE_END_NAMED(dmRun);
    resultCB(id,
             MAKE_ERR(ErrorValue::ErrorCode::RUNTIME_NET_NOT_FOUND,
                      llvm::formatv("Function {} not found", function).str()),
             std::move(context));
    return;
  }

  OpenCLFunction *func = static_cast<OpenCLFunction *>(funcIt->second);
  bool traceEnabled = func->getTraceInfo().enabled || clDoProfile;

  // Get a command queue for this run.
  OpenCLCommandQueue queue;
  {
    TRACE_EVENT_SCOPE(context->getTraceContext(),
                      TraceEvent::TraceLevel::RUNTIME,
                      "requestRunCommandQueue");
    auto queueOrError = requestRunCommandQueue(func);

    if (queueOrError) {
      queue = std::move(queueOrError.get());
    } else {
      TRACE_EVENT_SCOPE_END();
      resultCB(id, queueOrError.takeError(), std::move(context));
    }
  }
  // Create and set deviceBindings for call. This contains all the state
  // needed for the function to run on a device.
  auto program = programs_[function];
  auto clBindings = glow::make_unique<runtime::OpenCLDeviceBindings>(
      buffers_[function]->getBuffer(), queue.backingQueue, deviceId_, context_,
      program);

  // Copy inputs to the device.
  copyInputsToDevice(func->getRuntimeBundle(), context.get(), clBindings.get(),
                     traceEnabled);

  // Run that function.
  context->setDeviceBindings(std::move(clBindings));
  auto executeErr = func->execute(context.get());

  auto devBindings = static_cast<runtime::OpenCLDeviceBindings *>(
      context->getDeviceBindings());
  copyOutputsFromDevice(func->getRuntimeBundle(), context.get(), devBindings,
                        traceEnabled);

  // Output profiling information.
  translateTraceEvents(func->getManualTraceEvents(), context.get(),
                       devBindings);

  {
    TRACE_EVENT_SCOPE(context->getTraceContext(), TraceLevel::RUNTIME,
                      "releaseKernels");
    for (auto &kl : devBindings->kernelLaunches) {
      clReleaseKernel(kl.kernel_);
    }
    devBindings->kernelLaunches.clear();
  }

  // Return the command queue.
  returnRunCommandQueue(queue);

  // End the TraceEvent early to avoid time in the CB.
  TRACE_EVENT_SCOPE_END_NAMED(dmRun);

  // Fire the resultCB.
  resultCB(id, std::move(executeErr), std::move(context));
}

DeviceInfo OpenCLDeviceManager::getDeviceInfo() const {
  DeviceInfo info;
  info.availableLocalMemory = localMemSize_;
  return info;
}
