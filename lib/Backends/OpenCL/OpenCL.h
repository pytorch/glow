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
#ifndef GLOW_OPENCL_BACKEND_H
#define GLOW_OPENCL_BACKEND_H

#include "glow/Backends/Backend.h"
#include "glow/Base/Tensor.h"
#include "glow/Base/Traits.h"
#include "glow/CodeGen/MemoryAllocator.h"

#include "llvm/ADT/ArrayRef.h"

#include <unordered_map>

#if defined(__APPLE__) || defined(__MACOSX)
#include "OpenCL/opencl.h"
#else
#include <CL/cl.hpp>
#endif

namespace glow {
class IRFunction;
class Backend;
/// A helper struct with information about kernels launches.
struct KernelLaunch {
  /// Kernel that was launched.
  cl_kernel kernel_;
  /// The name of the kernel that was launched.
  std::string name_;
  /// Event associated with the start of the kernel.
  /// Used only when profiling is enabled.
  cl_event event_;
  /// Constructor to be used by launching Glow's CL kernels.
  KernelLaunch(cl_kernel kernel, std::string name, cl_event event)
      : kernel_(kernel), name_(name), event_(event) {}
  /// Constructor to be used when launching an "external" CL kernel, e.g.
  /// provided by such libraries like CLBlast, etc.
  KernelLaunch(const std::string &name, cl_event event)
      : kernel_(nullptr), name_(name), event_(event) {}
};

/// This is the OpenCL backend.
class OCLBackend final : public Backend {
  /// The Module that holds the IR. This does not own the module.
  IRFunction *F_;
  /// The allocator assigns device memory addresses to the buffers.
  MemoryAllocator allocator_;
  /// Maps values to on-device buffers. This list includes both weights and
  /// activations.
  std::unordered_map<const Value *, size_t> tensors_;
  /// Maps values to Tensors, that are *not* owned by this class.
  std::unordered_map<const Value *, Tensor *> externalTensors_;
  /// CL compute device id.
  cl_device_id deviceId_;
  /// CL compute context.
  cl_context context_;
  /// CL compute command queue.
  cl_command_queue commands_;
  // Stores the compiled kernel bank.
  cl_program program_;
  // A pointer to the on-device memory buffer.
  cl_mem deviceBuffer_{0};

public:
  /// Ctor.
  explicit OCLBackend(IRFunction *M);

  /// @name Backend methods.
  /// This is the implementation of the Backend interface.
  ///@{
  ~OCLBackend() override;

  void clear() override;

  void init() override;

  void doForwardPass() override;

  bool isOpSupported(Kinded::Kind opKind, ElemKind elementTy) const override {
    if (elementTy == ElemKind::Int8QTy) {
      return false;
    }

    return true;
  };
  /// @}

private:
  void copyWeightsToDevice();

  void copyWeightsFromDevice();
  /// Enqueue a \p kernel on a provided \p commands queue. 
  void enqueueKernel(cl_command_queue commands, cl_kernel kernel,
                     cl_device_id device, llvm::ArrayRef<size_t> global,
                     std::vector<KernelLaunch> &kernelLaunches);

  /// \returns a pointer to the tensor that is saved under \p v.
  Tensor *getTensor(const Value *v) const;
};

} // namespace glow

namespace glow {

Backend *createOCLBackend(IRFunction *M);

} // namespace glow

#endif // GLOW_OPENCL_BACKEND_H
