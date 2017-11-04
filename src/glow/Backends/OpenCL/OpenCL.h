#ifndef GLOW_OPENCL_BACKEND_H
#define GLOW_OPENCL_BACKEND_H

#include "glow/Backends/Backend.h"
#include "glow/Base/Tensor.h"
#include "glow/Base/Traits.h"

#include "llvm/ADT/ArrayRef.h"

#include <unordered_map>

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
  /// Maps values to on-device buffers. This list includes both weights and
  /// activations.
  std::unordered_map<const Value *, cl_mem> tensors_;
  /// Maps values to Tensors, that are *not* owned by this class.
  std::unordered_map<const Value *, Tensor *> externalTensors_;
  /// CL compute device id.
  cl_device_id deviceId_;
  /// CL compute context.
  cl_context context_;
  /// CL compute command queue.
  cl_command_queue commands_;
  // A list of compiled shaders.
  std::unordered_map<Kinded::Kind, cl_program> programs_;

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

  void registerGraphTensor(const Value *v, Tensor *t) override;

  Tensor *getTensor(const Variable *v) const override;

  Tensor *getGradTensor(const Variable *v) const override;
  /// @}

private:
  /// \returns a pointer to the tensor that is saved under \p v.
  Tensor *getTensor(const Value *v) const;

  /// Allocate a tensor to back the value \p v. Do not allocate anything if a
  /// tensor is already allocated for \p v.
  /// \returns a tensor for \p v.
  Tensor *getOrCreateTensor(const Value *v);

  /// If a tensor is allocated for \p v then delete it.
  void deleteTensor(const Value *v);

  /// \returns a float-handle to the tensor that is stored at \p v.
  Handle<float> getWeightHandle(Value *v) const;

  /// \returns a float-handle to the gradient tensor that matches \p v. Notice
  /// that this API is only valid when the module is compiled in training mode.
  Handle<float> getGradHandle(Value *v) const;

  /// \returns a pointer to the gradient tensor that matches \p v. Notice
  /// that this API is only valid when the module is compiled in training mode.
  Tensor *getGradTensor(const Value *v) const;

  /// \returns True if the value \p has an associated gradient tensor.
  bool hasGradTensor(const Value *v) const;
};

Backend *createOCLBackend(Module *M);

} // namespace glow

#endif // GLOW_OPENCL_BACKEND_H
