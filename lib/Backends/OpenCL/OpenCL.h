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
#ifndef GLOW_BACKENDS_OPENCL_OPENCL_H
#define GLOW_BACKENDS_OPENCL_OPENCL_H

#include "glow/Backend/Backend.h"
#include "glow/Backend/BackendUtils.h"
#include "glow/Backend/CompiledFunction.h"
#include "glow/Base/Tensor.h"
#include "glow/Base/Traits.h"
#include "glow/ExecutionContext/ExecutionContext.h"
#include "glow/Graph/Node.h"
#include "glow/IR/IR.h"

#include "llvm/ADT/ArrayRef.h"

#include <unordered_map>

#if defined(__APPLE__) || defined(__MACOSX)
#include "OpenCL/opencl.h"
#else
#include <CL/cl.h>
#endif

namespace glow {

class ConvolutionInst;
class Value;
namespace runtime {
struct OpenCLDeviceBindings;
}

/// A helper struct with information about kernels launches.
struct KernelLaunch {
  /// Kernel that was launched.
  cl_kernel kernel_;
  /// The name of the kernel that was launched.
  std::string name_;
  /// The type of the kernel that was launched.
  std::string type_;
  /// Event associated with the start of the kernel.
  /// Used only when profiling is enabled.
  cl_event event_;
  /// Constructor to be used by launching Glow's CL kernels.
  KernelLaunch(cl_kernel kernel, std::string name, std::string type,
               cl_event event)
      : kernel_(kernel), name_(name), type_(type), event_(event) {}
  /// Constructor to be used when launching an "external" CL kernel, e.g.
  /// provided by such libraries like CLBlast, etc.
  KernelLaunch(const std::string &name, std::string type, cl_event event)
      : kernel_(nullptr), name_(name), type_(type), event_(event) {}
};

/// Add an macro definition with an integer value to the set of options.
template <typename T>
static void addIntOption(std::vector<std::string> &options,
                         const std::string &name, const T value) {
  options.push_back("-D" + name + "=" + std::to_string(value));
}

using ManualEventMap =
    std::map<std::string, std::pair<Placeholder *, const TraceInfo::Event *>>;

/// A Glow IR function compiled for OpenCL.
class OpenCLFunction final : public CompiledFunction {
  /// A helper type representing a key for the program's cache.
  /// Each compiled program is uniquely identified by its source code, set of
  /// compiler options that were used and the device it was compiled for.
  using ProgramKey = std::tuple<const std::string, const std::string,
                                const cl_device_id, const cl_context>;
  struct ProgramKeyHash {
    std::size_t operator()(const ProgramKey &K) const noexcept {
      return llvm::hash_combine(std::get<0>(K), std::get<1>(K), std::get<2>(K));
    }
  };
  /// The IR to be executed.
  std::unique_ptr<IRFunction> F_;

  /// Cache of compiled programs.
  /// The same source code can be compile with different options (e.g. with
  /// different set of macro definitions) and/or for a different device and
  /// would result in different programs.
  std::unordered_map<ProgramKey, cl_program, ProgramKeyHash> programsCache_;

  /// is kernel level profiling (autoInstrumentation) enabled.
  bool kernelProfiling_{false};
  /// Manual trace events:
  ManualEventMap manualTraceEvents_;

public:
  /// Ctor.
  explicit OpenCLFunction(std::unique_ptr<IRFunction> F,
                          runtime::RuntimeBundle &&bundle, TraceInfo traceInfo);

  /// @name CompiledFunction interface
  ///@{
  ~OpenCLFunction() override;

  Error execute(ExecutionContext *context) override;

  void freeCompilationResources() override;

  /// Collects constants for runtime.
  void collectConstants(const Module *module) override;

  /// \returns the backend used to compile this function.
  std::string getCompileBackendName() const override { return "OpenCL"; }
  ///@}

  /// Returns IR function pointer.
  IRFunction *getIR() { return F_.get(); }

  /// Create a program from the \p source using provided \p options.
  cl_program createProgram(const std::string &source,
                           const std::vector<std::string> &options,
                           cl_command_queue queue);

  /// Returns metadata about manual TraceEvents defined in this function.
  ManualEventMap &getManualTraceEvents() { return manualTraceEvents_; }

private:
  /// Returns the directory of cached pre-built programs for the given device.
  /// \returns the directory as given by the user.
  std::string deviceProgramCacheDir(cl_device_id deviceId);

  /// Returns the (hashed) file name of a cached pre-built program for the
  /// given source and set of build options.
  /// \returns the filename (without directory).
  std::string diskCacheProgramFileName(cl_device_id deviceId,
                                       const std::string &source,
                                       const std::string &options);

  /// (Tries to) load a program with the given (hashed) filename
  /// from the disk cache.
  /// \returns pointer to the program, if found, nullptr otherwise.
  cl_program loadProgramFromDiskCache(std::string cacheDirectory,
                                      std::string programFileName,
                                      cl_context ctx, cl_device_id device);

  /// Save the given program to the disk cache.
  void saveProgramToDiskCache(std::string cacheDirectory,
                              std::string programFilename, cl_program program,
                              cl_context ctx, cl_device_id deviceId);

  /// Copy the value from a device to a provided buffer.
  /// \returns number of copied bytes.
  uint64_t copyValueFromDevice(const Value *v,
                               runtime::OpenCLDeviceBindings *devBindings,
                               void *buf = nullptr);
  /// Copy value from the provided buffer to the device.
  /// \returns number of copied bytes.
  uint64_t copyValueToDevice(const Value *v,
                             runtime::OpenCLDeviceBindings *devBindings,
                             void *buf = nullptr);
  /// Fill the device \p buffer with a given \p value.
  /// \param len number of buffer elements to be filled by the \p value.
  /// Elements are considered to be of the type described by \p elemKind.
  void fillBuffer(cl_mem buffer, uint64_t start, uint64_t len, float value,
                  ElemKind elemKind,
                  runtime::OpenCLDeviceBindings *devBindings);

  /// Execution a convolution instruction which uses NCHW format.
  void executeNCHWConvolution(const ConvolutionInst *CC,
                              ExecutionContext *executionContext,
                              runtime::OpenCLDeviceBindings *devBindings);
  /// Allocate a device buffer of required \p size.
  cl_mem allocDeviceBuffer(uint64_t size, cl_context clContext);
  /// Frees a device buffer.
  void freeDeviceBuffer(cl_mem buf);

  /// Create kernel with a given \p name from a \p program.
  /// If \p program is nullptr, try to find the kernel with a given \p name
  /// in any of compiled programs.
  cl_kernel createKernel(const std::string &name, cl_program program);

  /// Enqueue a \p kernel on a provided \p commands queue.
  void enqueueKernel(llvm::StringRef name, cl_command_queue commands,
                     cl_kernel kernel, cl_device_id device,
                     llvm::ArrayRef<size_t> global,
                     std::vector<KernelLaunch> &kernelLaunches);
  /// Enqueue a \p kernel on a provided \p commands queue using specified \p
  /// global and \p local work sizes.
  void enqueueKernel(llvm::StringRef name, cl_command_queue commands,
                     cl_kernel kernel, cl_device_id device,
                     llvm::ArrayRef<size_t> global,
                     llvm::ArrayRef<size_t> local,
                     std::vector<KernelLaunch> &kernelLaunches);

  /// Load outputs from the device into \p bindings.
  void updatePlaceholders(PlaceholderBindings *bindings,
                          runtime::OpenCLDeviceBindings *devBindings);

  /// Read trace events out of this func and write them into /p bindings
  void translateTraceEventsCL(ExecutionContext *context,
                              runtime::OpenCLDeviceBindings *devBindings);
};

/// This is the OpenCL backend.
class OCLBackend final : public BackendUsingGlowIR {
public:
  /// Ctor.
  OCLBackend() = default;

  /// @name Backend methods.
  /// This is the implementation of the Backend interface.
  ///@{
  ~OCLBackend() override = default;

  std::string getBackendName() const override { return getName(); }
  static std::string getName() { return "OpenCL"; }
  static unsigned numDevices() { return 1; }

  std::unique_ptr<CompiledFunction>
  compileIR(std::unique_ptr<IRFunction> IR) const override;

  Expected<std::unique_ptr<CompiledFunction>>
  compile(Function *F, const BackendOptions &opts) const override;

  bool transformPostLowering(
      Function *F, CompilationContext &cctx,
      const glow::runtime::DeviceInfo *devInfo) const override;

  bool isOpSupported(const NodeInfo &NI) const override;

  bool verify(const Function &F, bool verbose = true) const override;
  bool verify(const IRFunction &IR) const override;

  TensorLayoutCommon &getTensorLayoutRequirements() const override;

  bool shouldLower(const Node *N) const override {
    // The group convolution is supported in OpenCL slow convolution kernel.
    if (N->getKind() == Kinded::Kind::ConvolutionNodeKind)
      return false;
    return true;
  }

  /// Size of each TraceEvent (for manual events).
  size_t getTraceEventDataSize() const override { return sizeof(uint64_t); }

  runtime::DeviceManager *
  createDeviceManager(const runtime::DeviceConfig &deviceConfig) override;

private:
  /// Parses the graph \F and builds a TraceInfo structure from any found
  /// TraceEventNodes.
  TraceInfo buildManualTraceInfo(Function *F) const;

  /// Enables kernel profiling to generate TraceEvents after run.
  void autoInstrument(TraceInfo &traceInfo, IRFunction *IR) const;

  /// @}
};

namespace runtime {
/// OpenCLDeviceBindings inherits from DeviceBindings, it contains per run
/// device specific information used to run a compiled function on a specific
/// device.
struct OpenCLDeviceBindings : DeviceBindings {
  OpenCLDeviceBindings(cl_mem buffer, cl_command_queue commands,
                       cl_device_id device, cl_context ctx, cl_program prog)
      : DeviceBindings(OCLBackend::getName()), deviceBuffer{buffer},
        commandQueue{commands}, deviceId{device}, context{ctx}, program{prog} {}

  /// CL memory buffer. Currently this contains both mutable and immutable
  /// weights, the buffer is allocated once when the network is added.
  cl_mem deviceBuffer;

  /// CL compute command queue. A per run queue for the specific device.
  ///
  cl_command_queue commandQueue;

  /// CL compute device id. Identifies the CL device to be used.
  ///
  cl_device_id deviceId;

  /// CL compute context. Identifies a context on the CL device the computation
  /// will take place in.
  ///
  cl_context context;

  /// CL program which was compiled at addNetwork.
  cl_program program;

  /// A list of kernels and their associated events.
  std::vector<KernelLaunch> kernelLaunches;
};
} // namespace runtime
} // namespace glow

#endif // GLOW_BACKENDS_OPENCL_OPENCL_H
