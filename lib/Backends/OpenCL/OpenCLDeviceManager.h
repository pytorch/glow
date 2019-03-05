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
#ifndef GLOW_BACKENDS_OPENCL_OPENCLDEVICEMANAGER_H
#define GLOW_BACKENDS_OPENCL_OPENCLDEVICEMANAGER_H

#include "glow/Backends/QueueBackedDeviceManager.h"

#if defined(__APPLE__) || defined(__MACOSX)
#include "OpenCL/opencl.h"
#else
#include <CL/cl.h>
#endif

namespace glow {
namespace runtime {

/// A class that contains an openCL device buffer. It frees the buffer when it
/// is destroyed. Can be extended to store multiple buffers and rotate through
/// them. Also tracks number of functions using this buffer. Since adds/evicts
/// are serialized by the DeviceManager this does not need multithreading
/// protection.
class OpenCLBuffer {
  /// The OpenCL buffer being stored.
  cl_mem buffer_;
  /// Count of functions using this buffer.
  unsigned int users_{0};
  /// Size of the buffer in bytes.
  const size_t size_{0};

public:
  ~OpenCLBuffer() { clReleaseMemObject(buffer_); }
  OpenCLBuffer(cl_mem &buffer, size_t size) : buffer_(buffer), size_(size) {}
  /// Returns the stored buffer.
  cl_mem getBuffer() { return buffer_; }
  /// Increment user count by 1 and return new count.
  unsigned int incrementUsers() { return users_++; }
  /// Decrement user count by 1 and return new count.
  unsigned int decrementUsers() { return users_--; }
  /// Get size of buffer in bytes.
  size_t getSize() { return size_; }
};

/// A class controlling a single OpenCL device. Many OpenCLFunctions may be
/// added, but only one inference is executed at a time.
class OpenCLDeviceManager : public QueueBackedDeviceManager {
  /// Compiled function list by name.
  FunctionMapTy functions_;

  /// Maximum available memory on the device.
  uint64_t maxMemoryBytes_{0};

  /// Amount of memory used by all models.
  uint64_t usedMemoryBytes_{0};

  /// CL compute device id.
  cl_device_id deviceId_;
  /// CL compute context.
  cl_context context_;
  /// CL compute command queue.
  cl_command_queue commands_;

  /// Enable profiling flag.
  bool doProfile_{false};

  /// A pointer to the on-device memory buffer.
  std::map<std::string, std::shared_ptr<OpenCLBuffer>> buffers_;

  /// Allocate a device buffer of required \p size.
  cl_mem allocDeviceBuffer(uint64_t size);

  /// Device name.
  std::string name_;

public:
  OpenCLDeviceManager(llvm::StringRef name = "unnamed");

  ~OpenCLDeviceManager();

  ResultCode init() override;

  /// Returns the amount of memory in bytes available on the device when no
  /// models are loaded.
  uint64_t getMaximumMemory() const override;

  /// Returns the amount of memory in bytes currently availbe on the device.
  uint64_t getAvailableMemory() const override;

  /// Returns true if a function requiring the \p estimate size will fit on the
  /// device. This is not a promise as memory cost could vary due to alignment,
  /// etc.
  bool isMemoryAvailable(uint64_t estimate) const override;

protected:
  /// Adds functions to the device. Calls to this are serialized so concurrency
  /// is not an issue.
  void addNetworkImpl(const Module *module, FunctionMapTy functions,
                      ReadyCBTy cb) override;

  /// Remove network from the device. Also serialized so concurrency is not an
  /// issue.
  void evictNetworkImpl(std::string functionName,
                        EvictFunctionCBTy evictCB) override;

  /// Run the function on the device, there is a single thread of execution so
  /// only one function can execute at a time.
  void runFunctionImpl(runtime::RunIdentifierTy id, std::string functionName,
                       std::unique_ptr<Context> ctx, ResultCBTy cb) override;
};
} // namespace runtime
} // namespace glow

#endif // GLOW_BACKENDS_OPENCL_OPENCLDEVICEMANAGER_H
