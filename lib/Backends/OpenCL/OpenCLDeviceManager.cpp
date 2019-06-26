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

// Silence Apple's warning about the deprecation of OpenCL.
#define CL_SILENCE_DEPRECATION

// Silence warnings about using deprecated OpenCL 1.2 functions.
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include "OpenCLDeviceManager.h"
#include "OpenCL.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CommandLine.h"
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
extern llvm::cl::opt<unsigned> clDeviceId;
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
/// llvm::Expected with either the value or an error.
static llvm::Expected<unsigned> parseInputAsUnsigned(std::string input) {
  char *end;
  auto parsed = strtol(input.c_str(), &end, 10);
  if (end == input.c_str() || *end != '\0') {
    return MAKE_ERR(GlowErr::ErrorCode::RUNTIME_ERROR,
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

llvm::Expected<OpenCLCommandQueue> OpenCLCommandQueuePool::requestCommandQueue(
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

llvm::Expected<cl_mem> OpenCLDeviceManager::allocDeviceBuffer(uint64_t size) {
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

llvm::Error OpenCLDeviceManager::parseConfig() {
  auto it = config_.parameters.find("deviceId");
  unsigned value;
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
      return MAKE_ERR(GlowErr::ErrorCode::RUNTIME_ERROR,
                      "Invalid input expected true or false got: " +
                          it->second);
    }
  }
  return llvm::Error::success();
}

llvm::Error OpenCLDeviceManager::init() {
  // The OpenCL Backend defines three command line options: doProfile, deviceId,
  // and platformId. If the parameter is not provided we use the CL
  // options from the OpenCl Backend.

  // Check if parameters are in map.
  RETURN_IF_ERR(parseConfig());

  cl_uint numPlatforms{0};
  cl_int err = clGetPlatformIDs(0, NULL, &numPlatforms);
  if (err != CL_SUCCESS) {
    RETURN_ERR("clGetPlatformIDs Failed.");
  }

  if (numPlatforms < clPlatformId) {
    RETURN_ERR("Should have at least one platform for running OpenCL");
  }

  std::vector<cl_platform_id> platform_ids(numPlatforms);
  err = clGetPlatformIDs(numPlatforms, platform_ids.data(), NULL);

  cl_platform_id platform_id_used = platform_ids[clPlatformId];
  cl_uint num{0};
  err = clGetDeviceIDs(platform_id_used, CL_DEVICE_TYPE_ALL, 0, nullptr, &num);
  if (err != CL_SUCCESS) {
    RETURN_ERR("clGetDeviceIDs Failed");
  }
  if (num < clDeviceId) {
    RETURN_ERR("Should have at least one GPU/CPU/FPGA for running OpenCL");
  }
  std::vector<cl_device_id> devices(num);
  err = clGetDeviceIDs(platform_id_used, CL_DEVICE_TYPE_ALL, num,
                       devices.data(), nullptr);
  if (err != CL_SUCCESS) {
    RETURN_ERR("clGetDeviceIDs Failed");
  }
  deviceId_ = devices[clDeviceId];
  context_ = clCreateContext(nullptr, 1, &deviceId_, nullptr, nullptr, nullptr);
  if (!context_) {
    RETURN_ERR("clCreateContext Failed");
  }

  cl_ulong mem_size;
  err = clGetDeviceInfo(deviceId_, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong),
                        &mem_size, NULL);
  if (err != CL_SUCCESS) {
    RETURN_ERR("Error getting device memory limit");
  }

  // If limited by deviceConfig, should allow less deviceMemory
  if (config_.getDeviceMemory() != 0 && config_.getDeviceMemory() < mem_size) {
    maxMemoryBytes_ = config_.getDeviceMemory();
  } else {
    maxMemoryBytes_ = mem_size;
  }

  commandQueuePool_.setContext(context_);
  commandQueuePool_.setDevice(deviceId_);

  return llvm::Error::success();
}

OpenCLDeviceManager::~OpenCLDeviceManager() {
  clReleaseContext(context_);
  buffers_.clear();
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
      readyCB(
          module,
          MAKE_ERR(
              llvm::formatv(
                  "Failed to add network: function {} is not a OpenCL Function",
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
      readyCB(module, MAKE_ERR(GlowErr::ErrorCode::RUNTIME_OUT_OF_DEVICE_MEMORY,
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
                  GlowErr::ErrorCode::RUNTIME_OUT_OF_DEVICE_MEMORY,
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
          /* event_list */ nullptr, /* event */ doProfile_ ? &event : nullptr);
      if (err != CL_SUCCESS) {
        readyCB(module, MAKE_ERR("Unable to copy data to the device"));
        return;
      }
      clFinish(commands);
    }
    usedMemoryBytes_ += sizeInBytes;
    // Compile the CL program.
    // Add to the function name lookup map.
    // Add shared pointer to the buffer to buffers. This way the buffer will be
    // freed after the last reference is removed.

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
    function->createProgram(source, options, commands);
    functions_.emplace(func.first, func.second);
    buffers_.emplace(func.first, buffer);
    buffer->incrementUsers();

    DCHECK_LE(usedMemoryBytes_, maxMemoryBytes_);
    clReleaseCommandQueue(commands);
  }
  // Fire the ready CB.
  readyCB(module, llvm::Error::success());
}

void OpenCLDeviceManager::evictNetworkImpl(std::string functionName,
                                           EvictFunctionCBTy evictCB) {
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
            MAKE_ERR(GlowErr::ErrorCode::RUNTIME_NET_NOT_FOUND,
                     strFormat("Could not find function with name %s to evict",
                               functionName.c_str())));
    return;
  }
  evictCB(functionName, llvm::Error::success());
}

llvm::Expected<OpenCLCommandQueue>
OpenCLDeviceManager::requestRunCommandQueue(CompiledFunction *function) {
  auto traceInfo = function->getTraceInfo();
  cl_command_queue_properties props =
      clDoProfile || traceInfo.enabled ? CL_QUEUE_PROFILING_ENABLE : 0;
  return commandQueuePool_.requestCommandQueue(props);
}

void OpenCLDeviceManager::returnRunCommandQueue(OpenCLCommandQueue &queue) {
  commandQueuePool_.returnCommandQueue(queue);
}

void OpenCLDeviceManager::runFunctionImpl(
    RunIdentifierTy id, std::string function,
    std::unique_ptr<ExecutionContext> context, ResultCBTy resultCB) {
  TRACE_EVENT_SCOPE_NAMED(context->getTraceContext(), TraceLevel::RUNTIME,
                          "DeviceManager::run", dmRun);
  auto funcIt = functions_.find(function);
  if (funcIt == functions_.end()) {
    dmRun.addArg("reason", "function not found");
    TRACE_EVENT_SCOPE_END_NAMED(dmRun);
    resultCB(id,
             MAKE_ERR(GlowErr::ErrorCode::RUNTIME_NET_NOT_FOUND,
                      llvm::formatv("Function {} not found", function).str()),
             std::move(context));
    return;
  }

  CompiledFunction *func = funcIt->second;

  TRACE_EVENT_SCOPE(context->getTraceContext(), TraceEvent::TraceLevel::RUNTIME,
                    "requestRunCommandQueue");
  // Get a command queue for this run.
  OpenCLCommandQueue queue;
  auto queueOrError = requestRunCommandQueue(func);

  if (queueOrError) {
    queue = std::move(queueOrError.get());
  } else {
    resultCB(id, queueOrError.takeError(), std::move(context));
  }

  TRACE_EVENT_SCOPE_END();
  // Create and set deviceBindings for call. This contains all the state needed
  // for the function to run on a device.
  auto clBindings = llvm::make_unique<runtime::OpenCLDeviceBindings>(
      buffers_[function]->getBuffer(), queue.backingQueue, deviceId_, context_);
  context->setDeviceBindings(std::move(clBindings));

  // Run that function.
  auto executeErr = func->execute(context.get());

  // Return the command queue.
  returnRunCommandQueue(queue);

  // End the TraceEvent early to avoid time in the CB.
  TRACE_EVENT_SCOPE_END_NAMED(dmRun);

  // Fire the resultCB.
  resultCB(id, std::move(executeErr), std::move(context));
}
