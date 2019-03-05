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
#define DEBUG_TYPE "opencl"

// Silence Apple's warning about the deprecation of OpenCL.
#define CL_SILENCE_DEPRECATION

// Silence warnings about using deprecated OpenCL 1.2 functions.
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include "OpenCLDeviceManager.h"
#include "OpenCL.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"

using namespace glow;
using namespace glow::runtime;

namespace glow {
namespace runtime {
DeviceManager *createOCLDeviceManager(llvm::StringRef name) {
  return new OpenCLDeviceManager(name);
}
} // namespace runtime
} // namespace glow

cl_mem OpenCLDeviceManager::allocDeviceBuffer(uint64_t size) {
  const uint64_t alignment = 128;
  // Always allocate buffers properly aligned to hold values of any type.
  size = alignedSize(size, alignment);
  auto buf =
      clCreateBuffer(context_, CL_MEM_READ_WRITE, size, nullptr, nullptr);
  GLOW_ASSERT(buf && "Allocation failed!");
  return buf;
}
OpenCLDeviceManager::OpenCLDeviceManager(llvm::StringRef name)
    : QueueBackedDeviceManager(BackendKind::OpenCL, name) {
  name_ = name.str();
}

ResultCode OpenCLDeviceManager::init() {
  // For now we have a string, if the first digit is
  // an int use it, otherwise
  // use 0 as the default.
  // There are flags in the OpenCLBackend to select devices. Once we refactor
  // more functionality out of the OCLCompiledFunction we can move those flags
  // here.
  auto deviceId{0};
  if (llvm::isDigit(name_[0])) {
    deviceId = std::stoi(name_.substr(0, 1));
  }
  cl_uint numPlatforms{0};
  cl_int err = clGetPlatformIDs(0, NULL, &numPlatforms);
  if (err != CL_SUCCESS) {
    llvm::outs() << "clGetPlatformIDs Failed. \n";
    return ResultCode::Failed;
  }
  std::vector<cl_platform_id> platform_ids(numPlatforms);
  err = clGetPlatformIDs(numPlatforms, platform_ids.data(), NULL);
  // Take the first platform.
  cl_platform_id platform_id_used = platform_ids[0];
  cl_uint num{0};
  err = clGetDeviceIDs(platform_id_used, CL_DEVICE_TYPE_ALL, 0, nullptr, &num);
  if (err != CL_SUCCESS) {
    llvm::outs() << "clGetDeviceIDs Failed.\n";
    return ResultCode::Failed;
  }
  if (num < deviceId) {
    llvm::outs()
        << "Should have at least one GPU/CPU/FPGA for running OpenCL\n";
    return ResultCode::Failed;
  }
  std::vector<cl_device_id> devices(num);
  err = clGetDeviceIDs(platform_id_used, CL_DEVICE_TYPE_ALL, num,
                       devices.data(), nullptr);
  if (err != CL_SUCCESS) {
    llvm::outs() << "clGetDeviceIDs Failed.\n";
    return ResultCode::Failed;
  }
  deviceId_ = devices[deviceId];
  context_ = clCreateContext(nullptr, 1, &deviceId_, nullptr, nullptr, nullptr);
  if (!context_) {
    llvm::outs() << "clCreateContext Failed.\n";
    return ResultCode::Failed;
  }
  commands_ = clCreateCommandQueue(
      context_, deviceId_, (doProfile_) ? CL_QUEUE_PROFILING_ENABLE : 0, &err);
  if (!commands_) {
    llvm::outs() << "clCreateCommandQueue Failed.\n";
    return ResultCode::Failed;
  }
  cl_ulong mem_size;
  err = clGetDeviceInfo(deviceId_, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong),
                        &mem_size, NULL);
  if (err != CL_SUCCESS) {
    llvm::outs() << "Error getting device memory limit.\n";
    return ResultCode::Failed;
  }
  maxMemoryBytes_ = mem_size;
  return ResultCode::Executed;
}

OpenCLDeviceManager::~OpenCLDeviceManager() {
  clReleaseCommandQueue(commands_);
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
      llvm::errs() << "Failed to add network: already have a function called "
                   << func.first << ".\n";
      readyCB(module, ResultCode::Failed);
      return;
    }

    if (func.second->getCompileBackendKind() != BackendKind::OpenCL) {
      llvm::errs() << "Failed to add network: function " << func.first
                   << " is not an OpenCL Function.\n";
      readyCB(module, ResultCode::Failed);
    }
  }
  // Collect constants once, since currently the bundle grabs everything in the
  // module.
  auto &bundle = functions.begin()->second->getRuntimeBundle();
  if (bundle.getConstants() == nullptr) {
    bundle.collectConstants(module);
  }
  size_t sizeInBytes = bundle.getConstantWeightSize();
  if (usedMemoryBytes_ + sizeInBytes > maxMemoryBytes_) {
    llvm::errs() << "Failed to add network: not enough memory.\n";
    // Free the constants.
    bundle.freeConstants();
    readyCB(module, ResultCode::Failed);
    return;
  }

  // Copy constants to device.
  auto size = bundle.getConstantWeightSize() + bundle.getMutableWeightSize() +
              bundle.getActivationsSize();
  auto deviceBuffer = allocDeviceBuffer(size);
  auto buffer = std::make_shared<OpenCLBuffer>(deviceBuffer, size);
  if (bundle.getConstants()) {
    auto buf = bundle.getConstants();
    size_t valueOffset = 0;
    cl_event event{nullptr};
    cl_int err = clEnqueueWriteBuffer(
        commands_, buffer->getBuffer(), /* blocking_write */ CL_FALSE,
        valueOffset, sizeInBytes, buf, /* num_events_in_wait_list */ 0,
        /* event_list */ nullptr, /* event */ doProfile_ ? &event : nullptr);
    GLOW_ASSERT(err == CL_SUCCESS && "Unable to copy data to the device");
    clFinish(commands_);
  }
  usedMemoryBytes_ += sizeInBytes;
  // Add to the function name lookup map.
  // Add shared pointer to buffer to buffers. This way buffer will be freed
  // after last reference is removed.
  for (const auto &func : functions) {
    functions_.emplace(func.first, func.second);
    buffers_.emplace(func.first, buffer);
    buffer->incrementUsers();
  }

  assert(usedMemoryBytes_ <= maxMemoryBytes_);

  // Fire the ready CB.
  readyCB(module, ResultCode::Ready);
}

void OpenCLDeviceManager::evictNetworkImpl(std::string functionName,
                                           EvictFunctionCBTy evictCB) {
  ResultCode resultCode = ResultCode::Failed;

  if (functions_.erase(functionName)) {
    auto buffer = buffers_[functionName];
    auto users = buffer->decrementUsers();
    auto size = buffer->getSize();
    buffers_.erase(functionName);
    if (users == 0) {
      assert(usedMemoryBytes_ >= size);
      usedMemoryBytes_ -= size;
    }
    resultCode = ResultCode::Executed;
  }

  if (evictCB) {
    evictCB(functionName, resultCode);
  }
}

void OpenCLDeviceManager::runFunctionImpl(RunIdentifierTy id,
                                          std::string function,
                                          std::unique_ptr<Context> ctx,
                                          ResultCBTy resultCB) {
  auto funcIt = functions_.find(function);
  if (funcIt == functions_.end()) {
    llvm::errs() << "Failed to run function: name " << function
                 << " not found.\n";
    resultCB(id, ResultCode::Failed, std::move(ctx));
    return;
  }

  CompiledFunction *func = funcIt->second;

  // Run that function.
  // Until we have executionInfo object need to call setup/teardown and pin to
  // single device.
  func->setupRuns();
  func->beforeRun(*ctx.get());
  func->execute(ctx.get());
  func->afterRun(*ctx.get());

  // Fire the resultCB.
  resultCB(id, ResultCode::Executed, std::move(ctx));
}
