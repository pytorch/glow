#ifndef GLOW_OPENCL_BACKEND_H
#define GLOW_OPENCL_BACKEND_H

#include "glow/Backends/Backend.h"
#include "glow/Base/Tensor.h"
#include "glow/Base/Traits.h"
#include "glow/CodeGen/MemoryAllocator.h"

#include "llvm/ADT/ArrayRef.h"

#include <unordered_map>

#if WITH_OPENCL

#if defined(__APPLE__) || defined(__MACOSX)
#include "OpenCL/opencl.h"
#else
#include <CL/cl.hpp>
#endif

namespace glow {
class Module;
class Backend;

/// This is the OpenCL backend.
class OCLBackend final : public Backend {
  /// The Module that holds the IR. This does not own the module.
  Module *M_;
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
  explicit OCLBackend(Module *M);

  /// @name Backend methods.
  /// This is the implementation of the Backend interface.
  ///@{
  ~OCLBackend() override;

  void clear() override;

  void init() override;

  void doForwardPass(bool isTrain) override;
  /// @}

private:
  void copyWeightsToDevice();

  void copyWeightsFromDevice();

  /// \returns a pointer to the tensor that is saved under \p v.
  Tensor *getTensor(const Value *v) const;
};

} // namespace glow

#endif // WITH_OPENCL

namespace glow {

Backend *createOCLBackend(Module *M);

} // namespace glow

#endif // GLOW_OPENCL_BACKEND_H
