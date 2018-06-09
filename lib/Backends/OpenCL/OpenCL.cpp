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

#include "OpenCL.h"

#include "glow/Graph/Graph.h"
#include "glow/Graph/Nodes.h"
#include "glow/IR/IRUtils.h"
#include "glow/IR/Instrs.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

using namespace glow;
using llvm::format;

using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;

typedef uint32_t cl_size_t;

// This defines the string "SHADER_CODE".
#include "kernels.cl"
// This defines kernels for optimized convolutions.
#include "kernels_fwd_conv.cl"

namespace {
llvm::cl::OptionCategory OpenCLBackendCat("Glow OpenCL Backend Options");

static llvm::cl::opt<unsigned>
    deviceId("device", llvm::cl::desc("OpenCL device to be used"),
             llvm::cl::init(0), llvm::cl::cat(OpenCLBackendCat));
static llvm::cl::opt<bool> doProfile("opencl-profile",
                                     llvm::cl::desc("Profile OpenCL kernels"),
                                     llvm::cl::init(false),
                                     llvm::cl::cat(OpenCLBackendCat));
} // namespace

Backend *glow::createOCLBackend(IRFunction *F) { return new OCLBackend(F); }

using Kind = Kinded::Kind;
using kernelSrcEnum = struct {
  Kind kind;
  const char *funcName;
};

static void dumpCompileLog(cl_device_id dev, cl_program prog) {
#ifndef NDEBUG
  // Determine the size of the log.
  size_t logSize;
  clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);

  // Allocate memory for the log.
  auto *log = (char *)malloc(logSize);

  // Get the log.
  clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, logSize, log, nullptr);

  // Print the log.
  llvm::outs() << log << "\n";
  free(log);
#endif
}

OCLBackend::OCLBackend(IRFunction *F) : F_(F), allocator_(0xFFFFFFFF) {
  cl_uint num{0};
  cl_int err = clGetDeviceIDs(nullptr, CL_DEVICE_TYPE_ALL, 0, nullptr, &num);
  GLOW_ASSERT(err == CL_SUCCESS && "clGetDeviceIDs Failed.");
  GLOW_ASSERT(num > deviceId &&
              "Should have at least one GPU for running OpenCL");
  cl_device_id devices[num];
  err = clGetDeviceIDs(nullptr, CL_DEVICE_TYPE_ALL, num, devices, nullptr);
  GLOW_ASSERT(err == CL_SUCCESS && "clGetDeviceIDs Failed.");
  deviceId_ = devices[deviceId];
  context_ = clCreateContext(nullptr, 1, &deviceId_, nullptr, nullptr, nullptr);
  GLOW_ASSERT(context_ && "clCreateContext Failed.");
  commands_ = clCreateCommandQueue(
      context_, deviceId_, (doProfile) ? CL_QUEUE_PROFILING_ENABLE : 0, &err);
  GLOW_ASSERT(commands_ && "clCreateCommandQueue Failed.");

  err = CL_SUCCESS;
  /// Create the program from the source.
  createProgram(SHADER_CODE, {}, commands_);
}

OCLBackend::~OCLBackend() {
  for (auto &kv : programsCache_) {
    auto prog = kv.second;
    clReleaseProgram(prog);
  }
  clReleaseCommandQueue(commands_);
  clReleaseContext(context_);
  if (deviceBuffer_) {
    freeDeviceBuffer(deviceBuffer_);
    deviceBuffer_ = nullptr;
  }
  clear();
}

static std::string getKernelName(const char *baseName, ElemKind elemTy) {
  std::string name = baseName;
  switch (elemTy) {
  case ElemKind::FloatTy:
    return name + "W";
  case ElemKind::Int8QTy:
    return name + "_i8W";
  case ElemKind::Int32QTy:
    return name + "_i32W";
  case ElemKind::IndexTy:
    return name + "_uW";
  default:
    GLOW_ASSERT("Unsupported element type");
  }
  return "";
}

cl_kernel OCLBackend::createKernel(const std::string &name,
                                   cl_program program) {
  cl_int err = CL_SUCCESS;
  cl_kernel kernel = nullptr;
  if (program) {
    cl_kernel kernel = clCreateKernel(program, name.c_str(), &err);
    GLOW_ASSERT((kernel && err == CL_SUCCESS) && "clCreateKernel Failed.");
    return kernel;
  }
  // Inspect all programs.
  for (auto &kv : programsCache_) {
    auto prog = kv.second;
    cl_kernel kernel = clCreateKernel(prog, name.c_str(), &err);
    if (err == CL_SUCCESS) {
      return kernel;
    }
  }
  GLOW_ASSERT(kernel && "clCreateKernel Failed.");
  return kernel;
}

cl_program OCLBackend::createProgram(const std::string &source,
                                     const std::vector<std::string> &options,
                                     cl_command_queue queue) {
  const char *src = source.c_str();
  cl_context ctx;
  cl_int err = clGetCommandQueueInfo(queue, CL_QUEUE_CONTEXT, sizeof(ctx), &ctx,
                                     nullptr);
  GLOW_ASSERT(err == CL_SUCCESS && "clGetCommandQueueInfo Failed.");
  cl_device_id deviceId;
  err = clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE, sizeof(deviceId),
                              &deviceId, nullptr);
  GLOW_ASSERT(err == CL_SUCCESS && "clGetCommandQueueInfo Failed.");

  // Check if this program was compiled with the same parameters for the
  // provided context and device.
  std::string combinedOptions;
  for (auto &opt : options) {
    combinedOptions.append(opt).append(" ");
  }

  ProgramKey key = std::make_tuple(source, combinedOptions, deviceId);
  cl_program &program = programsCache_[key];
  if (program) {
    return program;
  }
  // Create a new compiled program.
  program = clCreateProgramWithSource(context_, 1, &src, nullptr, &err);
  GLOW_ASSERT(program && "clCreateProgramWithSource Failed.");
  err = clBuildProgram(program, 0, nullptr, combinedOptions.c_str(), nullptr,
                       nullptr);
  if (err) {
    dumpCompileLog(deviceId, program);
  }
  GLOW_ASSERT(err == CL_SUCCESS && "clBuildProgram Failed.");
  // Add this program to the program cache.
  return program;
}

template <class T>
void setKernelArg(cl_kernel kernel, unsigned argIdx, T value) {
  cl_int err = clSetKernelArg(kernel, argIdx, sizeof(T), &value);
  GLOW_ASSERT(err == CL_SUCCESS && "Unable to set parameter");
}

void OCLBackend::fillBuffer(cl_mem buffer, size_t start, size_t len,
                            float value, ElemKind elemKind) {
  auto kernel = createKernel(getKernelName("splat", elemKind));
  setKernelArg(kernel, 0, buffer);
  setKernelArg<cl_uint>(kernel, 1, start);
  setKernelArg(kernel, 2, value);
  enqueueKernel(commands_, kernel, deviceId_, {len}, kernelLaunches_);
}

/// \returns the max local workgroup size for each dimension, under the
/// opencl constraints, with the global workgroup sizes of \p global;
void getMaxLocalWorkgroupSize(cl_kernel kernel, cl_device_id device,
                              llvm::ArrayRef<size_t> global,
                              llvm::MutableArrayRef<size_t> local) {

  // Figure out the max size of the workgroup.
  size_t L;
  auto err = clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE,
                                      sizeof(L), &L, nullptr);

  size_t WIS[3];
  clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(WIS), &WIS,
                  nullptr);

  size_t WGS;
  clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(WGS), &WGS,
                  nullptr);

  GLOW_ASSERT(err == CL_SUCCESS && "Error in clGetKernelWorkGroupInfo.");
  // The global workgroup size must be a multiple of the local workgroup size,
  // and less than the max size for the specific dimension. Also, the
  // multiplication of all dimensions (size of total local work) needs to be
  // less than WSG. In here we find the highest L that divides the global
  // workgroup size. This is our naive implementation of gcd, with the other
  // constraints:
  size_t totalWorkPrevDims = 1;
  for (int i = 0, e = global.size(); i < e; i++) {
    local[i] = L;

    while (global[i] % local[i] || L % local[i] || local[i] > WIS[i] ||
           local[i] * totalWorkPrevDims > WGS) {
      local[i]--;
    }

    // Remember how much work we are doing in this dimension. Use it to make
    // sure that the next dimenstions don't exceed the total allowed workgroup
    // size.
    totalWorkPrevDims *= local[i];
  }
}

void OCLBackend::enqueueKernel(cl_command_queue commands, cl_kernel kernel,
                               cl_device_id device,
                               llvm::ArrayRef<size_t> global,
                               llvm::ArrayRef<size_t> local,
                               std::vector<KernelLaunch> &kernelLaunches) {
  char kernelName[128];
  size_t retSize;
  cl_int err = clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME,
                               sizeof(kernelName), &kernelName, &retSize);
  GLOW_ASSERT(err == CL_SUCCESS && "Error in clGetKernelInfo.");

  cl_event event{nullptr};
  err = clEnqueueNDRangeKernel(commands, kernel, global.size(), nullptr,
                               &global[0], &local[0], 0, nullptr,
                               doProfile ? &event : nullptr);
  GLOW_ASSERT(err == CL_SUCCESS && "Error in clEnqueueNDRangeKernel.");
  kernelLaunches.push_back(KernelLaunch(kernel, kernelName, event));
}

/// Enqueue a \p kernel for execution on the command queue \p commands on a
/// given \p device. The information about the launched kernel will be added to
/// \p kernelLaunches list.
void OCLBackend::enqueueKernel(cl_command_queue commands, cl_kernel kernel,
                               cl_device_id device,
                               llvm::ArrayRef<size_t> global,
                               std::vector<KernelLaunch> &kernelLaunches) {
  llvm::SmallVector<size_t, 4> local(global.size(), 0);
  getMaxLocalWorkgroupSize(kernel, device, global, local);
  char kernelName[128];
  size_t retSize;
  cl_int err = clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME,
                               sizeof(kernelName), &kernelName, &retSize);
  GLOW_ASSERT(err == CL_SUCCESS && "Error in clGetKernelInfo.");

  cl_event event{nullptr};
  err = clEnqueueNDRangeKernel(commands, kernel, global.size(), nullptr,
                               &global[0], &local[0], 0, nullptr,
                               doProfile ? &event : nullptr);
  GLOW_ASSERT(err == CL_SUCCESS && "Error in clEnqueueNDRangeKernel.");
  kernelLaunches.push_back(KernelLaunch(kernel, kernelName, event));
}

/// Analyze and dump the collected profiling information about the execution of
/// OpenCL kernels.
static void dumpProfileInfo(const std::vector<KernelLaunch> &kernelLaunches) {
  if (!doProfile)
    return;
  cl_ulong total = 0;

  std::unordered_map<std::string, cl_ulong> kernelToDuration;

  for (auto &kl : kernelLaunches) {
    auto &event = kl.event_;
    clWaitForEvents(1, &event);
    auto name = kl.name_;
    assert(!name.empty() && "Kernel name cannot be empty");
    cl_ulong time_start;
    cl_ulong time_end;

    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START,
                            sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end),
                            &time_end, NULL);
    // Duration (in nanoseconds).
    double duration = time_end - time_start;
    kernelToDuration[name] += duration;
    total += duration;
    llvm::outs() << "OpenCl execution time for a launch of kernel " << name
                 << format(" is: %0.3f milliseconds\n", duration / 1000000.0);
  }
  llvm::outs() << format(
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
                 << format(" is: %0.3f milliseconds (%lu%%)\n",
                           k.first / 1000000.0,
                           (unsigned long)(k.first * 100 / total));
  }
}

/// Add an macro definition with an integer value to the set of options.
template <typename T>
static void addIntOption(std::vector<std::string> &options,
                         const std::string &name, const T value) {
  options.push_back("-D" + name + "=" + std::to_string(value));
}

/// Add an macro definition with a string value to the set of options.
static void addStringOption(std::vector<std::string> &options,
                            const std::string &name, const std::string &value) {
  options.push_back("-D" + name + "=" + value);
}

void OCLBackend::executeConvolution(const OCLConvolutionInst *CC) {
  auto input = CC->getSrc();
  auto output = CC->getDest();
  auto bias = CC->getBias();
  auto weights = CC->getFilter();
  auto odim = ShapeNCHW(CC->getDest()->getType()->dims());
  auto idim = ShapeNCHW(CC->getSrc()->getType()->dims());
  auto fdim = ShapeNCHW(CC->getFilter()->getType()->dims());
  // Create options for compiling the program.
  // Don't use names M, N, K as they are defined in precompiled headers.

  std::vector<std::string> options;
  // Number of spacial axes.
  addIntOption(options, "v_nax", 2);
  // Number of groups.
  addIntOption(options, "v_g", 1);
  // Parameters for kernel size, padding and stride
  addIntOption(options, "v_k_0", CC->getKernel());
  addIntOption(options, "v_k_1", CC->getKernel());
  addIntOption(options, "v_p_0", CC->getPad());
  addIntOption(options, "v_p_1", CC->getPad());
  addIntOption(options, "v_s_0", CC->getStride());
  addIntOption(options, "v_s_1", CC->getStride());

  // Dilation.
  addIntOption(options, "v_d_0", 1);
  addIntOption(options, "v_d_1", 1);

  // Number of kernel input channels.
  addIntOption(options, "v_fin", fdim.c);
  // Number of kernel output channels.
  addIntOption(options, "v_fout", fdim.n);
  // Bias multiplier.
  addStringOption(options, "v_bmul", "(float)1");

  // Spacial dimensions of input.
  addIntOption(options, "v_imsi_0", idim.h);
  addIntOption(options, "v_imsi_1", idim.w);

  // Spacial dimensions of output.
  addIntOption(options, "v_imso_0", odim.h);
  addIntOption(options, "v_imso_1", odim.w);

  // Padding required for tiles.
  addStringOption(options, "v_pad_A", "0");
  addStringOption(options, "v_pad_B", "0");

  // Determine the work groups sizes along h and w.
  size_t WIS[3];
  cl_int err = clGetDeviceInfo(deviceId_, CL_DEVICE_MAX_WORK_ITEM_SIZES,
                               sizeof(WIS), &WIS, nullptr);
  GLOW_ASSERT(err == CL_SUCCESS && "Could not execute clGetDeviceInfo");
  size_t wg_size[3];
  for (int id = 0; id < 2; ++id) {
    size_t defaultVal = 16;
    // Special case on CPUs devices, where a workgroup size could be 1,
    // e.g. in case of Apple's OpenCL driver for CPUs.
    if (WIS[id] < defaultVal || (id == 0 && WIS[1] < defaultVal)) {
      defaultVal = WIS[1];
    }
    addIntOption(options, "workgroup_size_" + std::to_string(id), defaultVal);
    wg_size[id] = defaultVal;
  }
  // The tile-size in dimension K.
  // Should be tunable.
  addStringOption(options, "TSK", "4");
  addIntOption(options, "TSK_UNROLL", 1);

  // WPTN and WPTM should be tunable.
  size_t WPTN = 4;
  size_t WPTM = 4;
  // The work-per-thread in dimension N.
  addIntOption(options, "WPTN", WPTN);
  // The work-per-thread in dimension M.
  addIntOption(options, "WPTM", WPTM);

  // Vector width in dimensions M and M.
  // VWN and VWM should be tunable.
  addStringOption(options, "VWM", "4");
  addStringOption(options, "VWN", "4");

  // Generate a tailor-made convolution kernel using the provided options based
  // on the parameters of the current convolution.
  auto prog = createProgram(FWD_CONV_CODE, options, commands_);
  auto kernel = createKernel("conv_forward_mem", prog);
  setKernelArg(kernel, 0, deviceBuffer_);
  setKernelArg<cl_uint>(kernel, 1, tensors_[input]);
  setKernelArg<cl_uint>(kernel, 2, tensors_[weights]);
  setKernelArg<cl_uint>(kernel, 3, tensors_[bias]);
  setKernelArg<cl_uint>(kernel, 4, tensors_[output]);

  // Compute proper parameters for global work and workgroups.
  int group = 1;
  auto fw_wgs0 = wg_size[0];
  auto fw_wgs1 = wg_size[1];
  int fw_wptn = WPTN;
  int fw_wptm = WPTM;
  int fw_div_N = fw_wptn * fw_wgs0;
  int fw_div_M = fw_wptm * fw_wgs1;
  int N_FW_ = idim.h * idim.w;
  int M_FW_ = odim.c / group;
  size_t L;
  clGetKernelWorkGroupInfo(kernel, deviceId_, CL_KERNEL_WORK_GROUP_SIZE,
                           sizeof(L), &L, nullptr);
  GLOW_ASSERT(fw_wgs0 * fw_wgs1 <= L && "Bad workgroup size");

  // Set the size of a workgroup.
  std::vector<size_t> local = {fw_wgs0, fw_wgs1, 1};

  // Set the global work size.
  std::vector<size_t> global = {((N_FW_ - 1) / fw_div_N + 1) * fw_wgs0,
                                ((M_FW_ - 1) / fw_div_M + 1) * fw_wgs1,
                                idim.n * group};

  enqueueKernel(commands_, kernel, deviceId_, global, local, kernelLaunches_);
}

void OCLBackend::doForwardPass() {
  auto copiedToDeviceBytes = copyMutableWeightsToDevice();
  (void)copiedToDeviceBytes;
  DEBUG(llvm::dbgs() << "Copied " << copiedToDeviceBytes
                     << " bytes to OpenCL device\n");

  for (const auto &I : F_->getInstrs()) {
    // The kernels are named after the name of the instruction, plus the "W"
    // suffix to prevent name colissions for functions like 'tanh' that are also
    // a part of the OpenCL runtime.
    auto elemTy = I.getNumOperands() ? I.getOperand(0).first->getElementType()
                                     : ElemKind::FloatTy;
    std::string kernelName = getKernelName(I.getKindName(), elemTy);

    // Skip memory allocation instructions as they are NOPs.
    if (isa<AllocActivationInst>(I) || isa<DeallocActivationInst>(I) ||
        isa<TensorViewInst>(I)) {
      continue;
    }

    // Element-wise operations, except the copy instruction.
    if (I.isDataParallel() && !isa<CopyInst>(I)) {
      // Figure out how many element-wise elements are there to process:
      size_t global;
      if (I.isDataParallel()) {
        global = I.getOperand(0).first->getType()->size();
        if (global % 16 == 0) {
          // Start less kernels and let each kernel do more work using vector
          // instructions.
          global /= 16;
          kernelName += "16";
        } else if (global % 8 == 0) {
          // Start less kernels and let each kernel do more work using vector
          // instructions.
          global /= 8;
          kernelName += "8";
        }
      } else {
        GLOW_UNREACHABLE("Invalid instruction.");
      }

      cl_kernel kernel = createKernel(kernelName);
      setKernelArg(kernel, 0, deviceBuffer_);
      unsigned numArgs = I.getNumOperands();

      for (unsigned arg = 0, e = I.getNumOperands(); arg < e; arg++) {
        setKernelArg<cl_uint>(kernel, arg + 1,
                              tensors_[I.getOperand(arg).first]);
      }

      if (auto *SI = dyn_cast<SplatInst>(&I)) {
        // Pass the splat as a parameter.
        setKernelArg(kernel, numArgs + 1, SI->getValue());
      } else if (auto *EPI = dyn_cast<ElementPowInst>(&I)) {
        // Pass the exp as a parameter.
        setKernelArg(kernel, numArgs + 1, EPI->getExp());
      }

      enqueueKernel(commands_, kernel, deviceId_, {global}, kernelLaunches_);
      continue;
    }

    if (auto *SM = dyn_cast<SoftMaxInst>(&I)) {
      // Implement Softmax by parallelizing the batch dimension. Each sample in
      // the batch is processed by a different parallel 'thread'.
      cl_kernel kernel = createKernel(kernelName);

      setKernelArg(kernel, 0, deviceBuffer_);

      unsigned numArgs = I.getNumOperands();
      for (unsigned arg = 0; arg < numArgs; arg++) {
        setKernelArg<cl_uint>(kernel, arg + 1,
                              tensors_[I.getOperand(arg).first]);
      }

      // This is the number of elements for each slice. There are N slices in
      // our batch.
      auto inputDims = SM->getSrc()->getType()->dims();
      size_t numSlices = inputDims[0];

      // Pass the slice size (size of each sample in the batch) as a parameter.
      setKernelArg<cl_uint>(kernel, numArgs + 1, flattenCdr(inputDims).second);

      enqueueKernel(commands_, kernel, deviceId_, {numSlices}, kernelLaunches_);
      continue;
    }

    if (auto *SM = dyn_cast<SoftMaxGradInst>(&I)) {
      // Implement Softmax by parallelizing the batch dimension. Each sample in
      // the batch is processed by a different parallel 'thread'.
      cl_kernel kernel = createKernel(kernelName);

      setKernelArg(kernel, 0, deviceBuffer_);

      unsigned numArgs = I.getNumOperands();
      for (unsigned arg = 0; arg < numArgs; arg++) {
        setKernelArg<cl_uint>(kernel, arg + 1,
                              tensors_[I.getOperand(arg).first]);
      }

      // This is the number of elements for each slice. There are N slices in
      // our batch.
      auto inputDims = SM->getSrcGrad()->getType()->dims();
      size_t numSlices = inputDims[0];

      // Pass the slice size (size of each sample in the batch) as a parameter.
      setKernelArg<cl_uint>(kernel, numArgs + 1, flattenCdr(inputDims).second);

      enqueueKernel(commands_, kernel, deviceId_, {numSlices}, kernelLaunches_);
      continue;
    }

    if (auto *ET = dyn_cast<ExtractTensorInst>(&I)) {
      cl_kernel kernel = createKernel(kernelName);
      setKernelArg(kernel, 0, deviceBuffer_);

      unsigned numArgs = I.getNumOperands();
      for (unsigned arg = 0; arg < numArgs; arg++) {
        setKernelArg<cl_uint>(kernel, arg + 1,
                              tensors_[I.getOperand(arg).first]);
      }

      // Currently support tensors up to 4 dimensions.
      // TODO: Handle other dimensions.
      const size_t numDimensions = ET->getDest()->getType()->dims().size();
      ShapeNHWC odim = ShapeNHWC::empty();
      ShapeNHWC idim = ShapeNHWC::empty();
      ShapeNHWC offset = ShapeNHWC::empty();

      if (numDimensions == 1) {
        odim = ShapeNHWC::fromX(ET->getDest()->getType()->dims());
        idim = ShapeNHWC::fromX(ET->getSrc()->getType()->dims());
        offset = ShapeNHWC::fromXY(ET->getOffsets());
      } else if (numDimensions == 2) {
        odim = ShapeNHWC::fromXY(ET->getDest()->getType()->dims());
        idim = ShapeNHWC::fromXY(ET->getSrc()->getType()->dims());
        offset = ShapeNHWC::fromXY(ET->getOffsets());
      } else if (numDimensions == 3) {
        odim = ShapeNHWC::fromXYZ(ET->getDest()->getType()->dims());
        idim = ShapeNHWC::fromXYZ(ET->getSrc()->getType()->dims());
        offset = ShapeNHWC::fromXYZ(ET->getOffsets());
      } else if (numDimensions == 4) {
        odim = ShapeNHWC(ET->getDest()->getType()->dims());
        idim = ShapeNHWC(ET->getSrc()->getType()->dims());
        offset = ShapeNHWC(ET->getOffsets());
      } else {
        assert(false && "Unsupported tensor dimension");
      }

      setKernelArg(kernel, 3, odim);
      setKernelArg(kernel, 4, idim);
      setKernelArg(kernel, 5, offset);
      enqueueKernel(commands_, kernel, deviceId_, {odim.n, odim.h},
                    kernelLaunches_);
      continue;
    }

    if (auto *IT = dyn_cast<InsertTensorInst>(&I)) {
      cl_kernel kernel = createKernel(kernelName);
      setKernelArg(kernel, 0, deviceBuffer_);

      unsigned numArgs = I.getNumOperands();
      for (unsigned arg = 0; arg < numArgs; arg++) {
        setKernelArg<cl_uint>(kernel, arg + 1,
                              tensors_[I.getOperand(arg).first]);
      }

      // Currently support tensors of up to 4 dimensions.
      // TODO: Handle other dimensions.
      const size_t numDimensions = IT->getDest()->getType()->dims().size();
      ShapeNHWC odim = ShapeNHWC::empty();
      ShapeNHWC idim = ShapeNHWC::empty();
      ShapeNHWC offset = ShapeNHWC::empty();
      if (numDimensions == 1) {
        odim = ShapeNHWC::fromX(IT->getDest()->getType()->dims());
        idim = ShapeNHWC::fromX(IT->getSrc()->getType()->dims());
        offset = ShapeNHWC::fromX(IT->getOffsets());
      } else if (numDimensions == 2) {
        odim = ShapeNHWC::fromXY(IT->getDest()->getType()->dims());
        idim = ShapeNHWC::fromXY(IT->getSrc()->getType()->dims());
        offset = ShapeNHWC::fromXY(IT->getOffsets());
      } else if (numDimensions == 3) {
        odim = ShapeNHWC::fromXYZ(IT->getDest()->getType()->dims());
        idim = ShapeNHWC::fromXYZ(IT->getSrc()->getType()->dims());
        offset = ShapeNHWC::fromXYZ(IT->getOffsets());
      } else if (numDimensions == 4) {
        odim = ShapeNHWC(IT->getDest()->getType()->dims());
        idim = ShapeNHWC(IT->getSrc()->getType()->dims());
        offset = ShapeNHWC(IT->getOffsets());
      } else {
        assert(false && "Unsupported tensor dimension");
      }

      setKernelArg(kernel, 3, odim);
      setKernelArg(kernel, 4, idim);
      setKernelArg(kernel, 5, offset);
      setKernelArg<cl_uint>(kernel, 6, IT->getCount());
      setKernelArg<cl_uint>(kernel, 7, IT->getAxis());
      enqueueKernel(commands_, kernel, deviceId_, {idim.n, idim.h},
                    kernelLaunches_);
      continue;
    }

    if (auto *BMM = dyn_cast<MatMulInst>(&I)) {
// Size of the tile to be used for matrix multiplication.
#define TILE_DIM ((size_t)8)
      // Determine max work groups sizes.
      size_t WIS[3];
      cl_int err = clGetDeviceInfo(deviceId_, CL_DEVICE_MAX_WORK_ITEM_SIZES,
                                   sizeof(WIS), &WIS, nullptr);
      GLOW_ASSERT(err == CL_SUCCESS && "Could not execute clGetDeviceInfo");
      // True if the tiled matrix multiplication kernel can be used. This is
      // only possible if the device allows workgroups with sizes which are at
      // least as big as a tile.
      bool useTiledMatMul = (WIS[0] >= TILE_DIM && WIS[1] >= TILE_DIM);
      cl_kernel kernel =
          createKernel(useTiledMatMul ? "matmul_tiled" : kernelName);
      setKernelArg(kernel, 0, deviceBuffer_);

      unsigned numArgs = I.getNumOperands();
      for (unsigned arg = 0; arg < numArgs; arg++) {
        setKernelArg<cl_uint>(kernel, arg + 1,
                              tensors_[I.getOperand(arg).first]);
      }

      auto ddim = ShapeNHWC::fromXY(BMM->getDest()->getType()->dims());
      auto ldim = ShapeNHWC::fromXY(BMM->getLHS()->getType()->dims());
      auto rdim = ShapeNHWC::fromXY(BMM->getRHS()->getType()->dims());

      setKernelArg(kernel, 4, ddim);
      setKernelArg(kernel, 5, ldim);
      setKernelArg(kernel, 6, rdim);

      if (useTiledMatMul) {
        std::vector<size_t> local{TILE_DIM, TILE_DIM};
        std::vector<size_t> global{(ddim.n / local[0] + 1) * local[0],
                                   (ddim.h / local[1] + 1) * local[1]};

        enqueueKernel(commands_, kernel, deviceId_, global, local,
                      kernelLaunches_);
      } else {
        enqueueKernel(commands_, kernel, deviceId_, {ddim.n, ddim.h, ddim.w},
                      kernelLaunches_);
      }
#undef TILE_DIM
      continue;
    }

    if (auto *BA = dyn_cast<BatchedAddInst>(&I)) {
      cl_kernel kernel = createKernel(kernelName);
      setKernelArg(kernel, 0, deviceBuffer_);

      unsigned numArgs = I.getNumOperands();
      for (unsigned arg = 0; arg < numArgs; arg++) {
        setKernelArg<cl_uint>(kernel, arg + 1,
                              tensors_[I.getOperand(arg).first]);
      }

      auto bdim = flattenCdr(BA->getBatch()->dims());
      setKernelArg<cl_uint>(kernel, 4, bdim.first);
      setKernelArg<cl_uint>(kernel, 5, bdim.second);

      // Parallelize on each element in the slice.
      enqueueKernel(commands_, kernel, deviceId_, {bdim.second},
                    kernelLaunches_);
      continue;
    }

    if (auto *BRA = dyn_cast<BatchedReduceAddInst>(&I)) {
      cl_kernel kernel = createKernel(kernelName);
      setKernelArg(kernel, 0, deviceBuffer_);

      unsigned numArgs = I.getNumOperands();
      for (unsigned arg = 0; arg < numArgs; arg++) {
        setKernelArg<cl_uint>(kernel, arg + 1,
                              tensors_[I.getOperand(arg).first]);
      }

      auto bdim = flattenCdr(BRA->getBatch()->dims());
      setKernelArg<cl_uint>(kernel, 3, bdim.first);
      setKernelArg<cl_uint>(kernel, 4, bdim.second);

      // Parallelize on each element in the slice.
      enqueueKernel(commands_, kernel, deviceId_, {bdim.second},
                    kernelLaunches_);
      continue;
    }

    if (auto *CC = dyn_cast<OCLConvolutionInst>(&I)) {
      executeConvolution(CC);
      continue;
    }

    if (auto *CC = dyn_cast<ConvolutionInst>(&I)) {
      // This is a naive implementation that parallelizes using three dims:
      // the X and the Y in the output filter.
      cl_kernel kernel = createKernel(kernelName);
      setKernelArg(kernel, 0, deviceBuffer_);

      unsigned numArgs = I.getNumOperands();
      for (unsigned arg = 0; arg < numArgs; arg++) {
        setKernelArg<cl_uint>(kernel, arg + 1,
                              tensors_[I.getOperand(arg).first]);
      }

      auto odim = ShapeNHWC(CC->getDest()->getType()->dims());
      auto idim = ShapeNHWC(CC->getSrc()->getType()->dims());

      setKernelArg<cl_uint>(kernel, 5, CC->getKernel());
      setKernelArg<cl_uint>(kernel, 6, CC->getStride());
      setKernelArg<cl_uint>(kernel, 7, CC->getPad());
      setKernelArg(kernel, 8, odim);
      setKernelArg(kernel, 9, idim);
      setKernelArg(kernel, 10, ShapeNHWC(CC->getFilter()->getType()->dims()));

      // Use a 3D grid where the first dimension is the depth and the second
      // dimension is the slice index in the batch.
      enqueueKernel(commands_, kernel, deviceId_, {odim.h, odim.w, odim.c},
                    kernelLaunches_);
      continue;
    }

    if (auto *CG = dyn_cast<ConvolutionGradInst>(&I)) {
      auto *src = CG->getSrc();
      auto *filter = CG->getFilter();
      auto *destGrad = CG->getDestGrad();
      auto *srcGrad = CG->getSrcGrad();
      auto *filterGrad = CG->getFilterGrad();
      auto *biasGrad = CG->getBiasGrad();
      cl_kernel kernel = createKernel(kernelName);
      setKernelArg(kernel, 0, deviceBuffer_);

      unsigned numArgs = CG->getNumOperands();
      for (unsigned arg = 0; arg < numArgs; arg++) {
        setKernelArg<cl_uint>(kernel, arg + 1,
                              tensors_[CG->getOperand(arg).first]);
      }

      auto destGradDim = ShapeNHWC(destGrad->dims());
      auto srcDim = ShapeNHWC(src->dims());
      auto filterGradDim = ShapeNHWC(filterGrad->dims());

      setKernelArg<cl_uint>(kernel, numArgs + 1, CG->getKernel());
      setKernelArg<cl_uint>(kernel, numArgs + 2, CG->getStride());
      setKernelArg<cl_uint>(kernel, numArgs + 3, CG->getPad());
      setKernelArg(kernel, numArgs + 4, srcDim);
      setKernelArg(kernel, numArgs + 5, destGradDim);
      setKernelArg(kernel, numArgs + 6, filterGradDim);

      // Zero memory for the output buffers.
      fillBuffer(deviceBuffer_, tensors_[srcGrad], srcGrad->size(), 0,
                 srcGrad->getElementType());
      fillBuffer(deviceBuffer_, tensors_[filterGrad], filterGrad->size(), 0,
                 filterGrad->getElementType());
      fillBuffer(deviceBuffer_, tensors_[biasGrad], biasGrad->size(), 0,
                 biasGrad->getElementType());

      (void)filter;
      assert(filter->dims() == filterGrad->dims() && "Dims should be the same");
      assert(src->dims() == srcGrad->dims() && "Dims should be the same");

      enqueueKernel(commands_, kernel, deviceId_,
                    {destGradDim.h, destGradDim.w, destGradDim.c},
                    kernelLaunches_);
      continue;
    }

    if (auto *PM = dyn_cast<PoolMaxInst>(&I)) {
      // This is a naive implementation that parallelizes using three dims:
      // the X and the Y in the output filter.
      cl_kernel kernel = createKernel(kernelName);
      setKernelArg(kernel, 0, deviceBuffer_);

      unsigned numArgs = I.getNumOperands();
      for (unsigned arg = 0; arg < numArgs; arg++) {
        setKernelArg<cl_uint>(kernel, arg + 1,
                              tensors_[I.getOperand(arg).first]);
      }

      auto odim = ShapeNHWC(PM->getDest()->getType()->dims());
      auto idim = ShapeNHWC(PM->getSrc()->getType()->dims());

      setKernelArg<cl_uint>(kernel, numArgs + 1, PM->getKernel());
      setKernelArg<cl_uint>(kernel, numArgs + 2, PM->getStride());
      setKernelArg<cl_uint>(kernel, numArgs + 3, PM->getPad());
      setKernelArg(kernel, numArgs + 4, odim);
      setKernelArg(kernel, numArgs + 5, idim);

      enqueueKernel(commands_, kernel, deviceId_, {odim.h, odim.w, odim.c},
                    kernelLaunches_);
      continue;
    }

    if (auto *PM = dyn_cast<PoolMaxWithXYInst>(&I)) {
      // This is a naive implementation that parallelizes using three dims:
      // the X and the Y in the output filter.
      cl_kernel kernel = createKernel(kernelName);
      setKernelArg(kernel, 0, deviceBuffer_);

      unsigned numArgs = I.getNumOperands();
      for (unsigned arg = 0; arg < numArgs; arg++) {
        setKernelArg<cl_uint>(kernel, arg + 1,
                              tensors_[I.getOperand(arg).first]);
      }

      auto odim = ShapeNHWC(PM->getDest()->getType()->dims());
      auto idim = ShapeNHWC(PM->getSrc()->getType()->dims());

      setKernelArg<cl_uint>(kernel, numArgs + 1, PM->getKernel());
      setKernelArg<cl_uint>(kernel, numArgs + 2, PM->getStride());
      setKernelArg<cl_uint>(kernel, numArgs + 3, PM->getPad());
      setKernelArg(kernel, numArgs + 4, odim);
      setKernelArg(kernel, numArgs + 5, idim);

      enqueueKernel(commands_, kernel, deviceId_, {odim.h, odim.w, odim.c},
                    kernelLaunches_);
      continue;
    }

    if (auto *PMG = dyn_cast<PoolMaxWithXYGradInst>(&I)) {
      cl_kernel kernel = createKernel(kernelName);
      setKernelArg(kernel, 0, deviceBuffer_);

      unsigned numArgs = I.getNumOperands();
      for (unsigned arg = 0; arg < numArgs; arg++) {
        setKernelArg<cl_uint>(kernel, arg + 1,
                              tensors_[I.getOperand(arg).first]);
      }

      auto destGradDim = ShapeNHWC(PMG->getDestGrad()->dims());
      auto srcGradDim = ShapeNHWC(PMG->getSrcGrad()->dims());

      setKernelArg<cl_uint>(kernel, numArgs + 1, PMG->getKernel());
      setKernelArg<cl_uint>(kernel, numArgs + 2, PMG->getStride());
      setKernelArg<cl_uint>(kernel, numArgs + 3, PMG->getPad());
      setKernelArg(kernel, numArgs + 4, srcGradDim);
      setKernelArg(kernel, numArgs + 5, destGradDim);

      assert(srcGradDim.n == destGradDim.n && "batch size is wrong");
      assert(srcGradDim.c == destGradDim.c && "depth size is wrong");

      enqueueKernel(commands_, kernel, deviceId_, {srcGradDim.n},
                    kernelLaunches_);
      continue;
    }

    if (auto *PA = dyn_cast<PoolAvgInst>(&I)) {
      // This is a naive implementation that parallelizes using three dims:
      // the X and the Y in the output filter.
      cl_kernel kernel = createKernel(kernelName);
      setKernelArg(kernel, 0, deviceBuffer_);

      unsigned numArgs = I.getNumOperands();
      for (unsigned arg = 0; arg < numArgs; arg++) {
        setKernelArg<cl_uint>(kernel, arg + 1,
                              tensors_[I.getOperand(arg).first]);
      }

      auto odim = ShapeNHWC(PA->getDest()->getType()->dims());
      auto idim = ShapeNHWC(PA->getSrc()->getType()->dims());

      setKernelArg<cl_uint>(kernel, 3, PA->getKernel());
      setKernelArg<cl_uint>(kernel, 4, PA->getStride());
      setKernelArg<cl_uint>(kernel, 5, PA->getPad());
      setKernelArg(kernel, 6, odim);
      setKernelArg(kernel, 7, idim);

      enqueueKernel(commands_, kernel, deviceId_, {odim.h, odim.w, odim.c},
                    kernelLaunches_);
      continue;
    }

    if (auto *TR = dyn_cast<TransposeInst>(&I)) {
      // This is a naive implementation that parallelizes using one dimension,
      // the N (batch size).
      GLOW_ASSERT(TR->getShuffle().size() <= 4 &&
                  "This code supports only 4 and lower dimensional transposes");

      cl_kernel kernel = createKernel(kernelName);
      setKernelArg(kernel, 0, deviceBuffer_);

      unsigned numArgs = I.getNumOperands();
      for (unsigned arg = 0; arg < numArgs; arg++) {
        setKernelArg<cl_uint>(kernel, arg + 1,
                              tensors_[I.getOperand(arg).first]);
      }

      // Temporary hack to support 3-dim transposes.
      // TODO: support any dimensional transposes.
      std::vector<size_t> odim_vec = TR->getDest()->getType()->dims();
      std::vector<size_t> idim_vec = TR->getSrc()->getType()->dims();
      std::vector<unsigned> mask = TR->getShuffle();
      while (mask.size() < 4) {
        odim_vec.push_back(1);
        idim_vec.push_back(1);
        mask.push_back(mask.size());
        continue;
      }

      auto odim = ShapeNHWC(odim_vec);
      auto idim = ShapeNHWC(idim_vec);

      setKernelArg(kernel, 3, odim);
      setKernelArg(kernel, 4, idim);

      ShapeNHWC shuff(mask[0], mask[1], mask[2], mask[3]);
      setKernelArg(kernel, 5, shuff);
      enqueueKernel(commands_, kernel, deviceId_, {idim.n, idim.h},
                    kernelLaunches_);
      continue;
    }

    if (auto *C = dyn_cast<CopyInst>(&I)) {
      Value *dest, *src;
      dest = C->getDest();
      src = C->getSrc();
      if (src == dest) {
        continue;
      }
      size_t destOff = tensors_[dest];
      size_t srcOff = tensors_[src];
      size_t sizeInBytes = dest->getSizeInBytes();

      cl_int err =
          clEnqueueCopyBuffer(commands_, deviceBuffer_, deviceBuffer_, srcOff,
                              destOff, sizeInBytes, 0, nullptr, nullptr);
      GLOW_ASSERT(err == CL_SUCCESS && "Error in clEnqueueCopyBuffer.");
      continue;
    }

    if (auto *GI = dyn_cast<GatherInst>(&I)) {
      cl_kernel kernel = createKernel(kernelName);
      setKernelArg(kernel, 0, deviceBuffer_);

      unsigned numArgs = I.getNumOperands();
      for (unsigned arg = 0; arg < numArgs; arg++) {
        setKernelArg<cl_uint>(kernel, arg + 1,
                              tensors_[I.getOperand(arg).first]);
      }

      auto *data = GI->getData();
      size_t dataSliceSize = data->size() / data->dims()[0];
      size_t numIndices = GI->getIndices()->size();
      setKernelArg<cl_uint>(kernel, numArgs + 1, numIndices);
      setKernelArg<cl_uint>(kernel, numArgs + 2, dataSliceSize);

      enqueueKernel(commands_, kernel, deviceId_, {numIndices},
                    kernelLaunches_);
      continue;
    }

    if (auto *DP = dyn_cast<DebugPrintInst>(&I)) {
      clFinish(commands_);
      auto *V = DP->getSrc();
      // Allocate a temporary tensor to hold the value.
      Tensor T(V->getType());
      // Load the current value of the variable into host memory.
      copyValueFromDevice(V, T.getUnsafePtr());
      clFinish(commands_);
      llvm::outs() << I.getName() << ": ";
      // Dump the content of a value.
      V->dump();
      llvm::outs() << "\n";
      dumpImpl(&T);
      llvm::outs() << "\n";
      llvm::outs().flush();
      continue;
    }

    if (auto PA = dyn_cast<OCLPoolAvgInst>(&I)) {
      cl_kernel kernel = createKernel(kernelName);
      setKernelArg(kernel, 0, deviceBuffer_);

      unsigned numArgs = I.getNumOperands();
      for (unsigned arg = 0; arg < numArgs; arg++) {
        setKernelArg<cl_uint>(kernel, arg + 1,
                              tensors_[I.getOperand(arg).first]);
      }

      auto odim = ShapeNCHW(PA->getDest()->getType()->dims());
      auto idim = ShapeNCHW(PA->getSrc()->getType()->dims());

      setKernelArg<cl_uint>(kernel, 3, PA->getKernel());
      setKernelArg<cl_uint>(kernel, 4, PA->getStride());
      setKernelArg<cl_uint>(kernel, 5, PA->getPad());
      setKernelArg(kernel, 6, odim);
      setKernelArg(kernel, 7, idim);

      enqueueKernel(commands_, kernel, deviceId_, {odim.h, odim.w, odim.c},
                    kernelLaunches_);
      continue;
    }

    if (auto *PM = dyn_cast<OCLPoolMaxInst>(&I)) {
      cl_kernel kernel = createKernel(kernelName);
      setKernelArg(kernel, 0, deviceBuffer_);

      unsigned numArgs = I.getNumOperands();
      for (unsigned arg = 0; arg < numArgs; arg++) {
        setKernelArg<cl_uint>(kernel, arg + 1,
                              tensors_[I.getOperand(arg).first]);
      }

      auto odim = ShapeNCHW(PM->getDest()->getType()->dims());
      auto idim = ShapeNCHW(PM->getSrc()->getType()->dims());

      setKernelArg<cl_uint>(kernel, numArgs + 1, PM->getKernel());
      setKernelArg<cl_uint>(kernel, numArgs + 2, PM->getStride());
      setKernelArg<cl_uint>(kernel, numArgs + 3, PM->getPad());
      setKernelArg(kernel, numArgs + 4, odim);
      setKernelArg(kernel, numArgs + 5, idim);

      enqueueKernel(commands_, kernel, deviceId_, {odim.h, odim.w, odim.c},
                    kernelLaunches_);
      continue;
    }

    llvm::errs() << "Cannot select: " << I.getKindName() << "\n";
    GLOW_UNREACHABLE("compilation failed");
  }

  clFinish(commands_);

  // Output profiling information.
  dumpProfileInfo(kernelLaunches_);

  for (auto &kl : kernelLaunches_) {
    clReleaseKernel(kl.kernel_);
  }
  kernelLaunches_.clear();

  auto copiedFromDeviceBytes = copyMutableWeightsFromDevice();
  (void)copiedFromDeviceBytes;
  DEBUG(llvm::dbgs() << "Copied " << copiedFromDeviceBytes
                     << " bytes from OpenCL device\n");
}

size_t OCLBackend::copyValueToDevice(const Value *v, void *buf) {
  size_t copiedBytes = 0;
  auto it = tensors_.find(v);
  assert(it != tensors_.end() && "Unknown value");
  size_t sizeInBytes = v->getType()->getSizeInBytes();
  // Issue a non-blocking command to copy the buffer to the device.
  if (sizeInBytes) {
    if (!buf) {
      Tensor *T = externalTensors_[v];
      assert(T && "Expected an external tensor");
      buf = T->getUnsafePtr();
    }
    size_t valueOffset = it->second;
    cl_int err = clEnqueueWriteBuffer(
        commands_, deviceBuffer_, /* blocking_read */ CL_FALSE, valueOffset,
        sizeInBytes, buf, /* num_events_in_wait_list */ 0,
        /* event_list */ nullptr, /* event */ nullptr);
    GLOW_ASSERT(err == CL_SUCCESS && "Unable to copy data to the device");
    copiedBytes += sizeInBytes;
  }
  return copiedBytes;
}

size_t OCLBackend::copyValueFromDevice(const Value *v, void *buf) {
  size_t copiedBytes = 0;
  auto it = tensors_.find(v);
  assert(it != tensors_.end() && "Unknown value");
  size_t sizeInBytes = v->getType()->getSizeInBytes();
  // Issue a non-blocking command to copy the buffer from the device.
  if (sizeInBytes) {
    if (!buf) {
      Tensor *T = externalTensors_[v];
      assert(T && "Expected an external tensor");
      buf = T->getUnsafePtr();
    }
    size_t valueOffset = it->second;
    cl_int err = clEnqueueReadBuffer(
        commands_, deviceBuffer_, /* blocking_read */ CL_FALSE, valueOffset,
        sizeInBytes, buf, /* num_events_in_wait_list */ 0,
        /* event_list */ nullptr, /* event */ nullptr);
    GLOW_ASSERT(err == CL_SUCCESS && "Unable to copy from the device");
    DEBUG(llvm::dbgs() << "Copied the value from device: "
                       << it->first->getName() << "\n");
    copiedBytes += sizeInBytes;
  }
  return copiedBytes;
}

size_t OCLBackend::copyMutableWeightsToDevice() {
  size_t copiedBytes = 0;
  for (auto it : tensors_) {
    if (!externalTensors_.count(it.first)) {
      continue;
    }
    if (auto *W = dyn_cast<WeightVar>(it.first)) {
      if (W->getMutability() == WeightVar::MutabilityKind::Constant)
        continue;
    }
    copiedBytes += copyValueToDevice(it.first);
  }
  // Do it!
  clFinish(commands_);
  return copiedBytes;
}

size_t OCLBackend::copyConstantWeightsToDevice() {
  size_t copiedBytes = 0;
  for (auto it : tensors_) {
    if (!externalTensors_.count(it.first)) {
      continue;
    }
    if (auto *W = dyn_cast<WeightVar>(it.first)) {
      if (W->getMutability() != WeightVar::MutabilityKind::Constant)
        continue;
    }
    copiedBytes += copyValueToDevice(it.first);
  }
  // Do it!
  clFinish(commands_);
  return copiedBytes;
}

size_t OCLBackend::copyMutableWeightsFromDevice() {
  size_t copiedBytes = 0;
  clFinish(commands_);

  for (auto it : tensors_) {
    if (!externalTensors_.count(it.first)) {
      continue;
    }
    if (auto *W = dyn_cast<WeightVar>(it.first)) {
      if (W->getMutability() == WeightVar::MutabilityKind::Constant)
        continue;
    }
    copiedBytes += copyValueFromDevice(it.first);
  }
  clFinish(commands_);
  return copiedBytes;
}

void OCLBackend::init() {
  for (auto &v : F_->getGraph()->getParent()->getVars()) {
    auto *w = F_->getWeightForNode(v);
    assert(!externalTensors_.count(w) && "The tensor is already registered");
    externalTensors_[w] = &v->getPayload();
  }

  // Assign device-space addresses to the weights.
  for (auto it : externalTensors_) {
    Tensor *T = it.second;
    size_t sizeInBytes = T->getType().getSizeInBytes();
    size_t addr = allocator_.allocate(sizeInBytes);
    // Associate the new buffer with the weight value.
    tensors_[it.first] = addr;
  }

  // Assign device-space addresses to the activations.
  for (const auto &I : F_->getInstrs()) {
    if (auto *A = llvm::dyn_cast<AllocActivationInst>(&I)) {
      auto numBytes = I.getSizeInBytes();
      size_t addr = allocator_.allocate(numBytes);
      assert(!tensors_.count(A) && "Allocation already made!");
      tensors_[A] = addr;
      continue;
    }

    if (auto *TV = llvm::dyn_cast<TensorViewInst>(&I)) {
      // Calculate and store the length of the offset into the base, using the
      // source of the tensorview.
      assert(!tensors_.count(TV) && "Allocation already made!");
      size_t offsetLength = TV->getOffsets()[0];
      auto *tvSource = TV->getSrc();
      if (tvSource->dims().size() > 1) {
        for (size_t i = 1; i < tvSource->dims().size(); ++i) {
          offsetLength *= tvSource->dims()[i];
        }
      }
      assert(tensors_.count(tvSource) && "Source allocation not found!");
      tensors_[TV] =
          tensors_[tvSource] + (offsetLength * TV->getType()->getElementSize());
      continue;
    }

    if (auto *D = llvm::dyn_cast<DeallocActivationInst>(&I)) {
      auto *A = D->getAlloc();
      assert(tensors_.count(A) && "Invalid deallocation!");
      allocator_.deallocate(tensors_[A]);
      continue;
    }
  }

  // Ask the memory allocator how much memory is required. What was the high
  // watermark for this program.
  size_t requiredSpace = allocator_.getMaxMemoryUsage();
  DEBUG(llvm::dbgs() << "Allocated GPU memory block of size: " << requiredSpace
                     << "\n");

  // Release the memory from the previous run.
  if (deviceBuffer_) {
    freeDeviceBuffer(deviceBuffer_);
    deviceBuffer_ = nullptr;
  }

  deviceBuffer_ = allocDeviceBuffer(requiredSpace);
  // Copy constant weights just once.
  copyConstantWeightsToDevice();
}

void OCLBackend::clear() { externalTensors_.clear(); }

Tensor *OCLBackend::getTensor(const Value *v) const {
  assert(externalTensors_.count(v) && "Unknown value");
  auto ie = externalTensors_.find(v);
  return ie->second;
}

cl_mem OCLBackend::allocDeviceBuffer(size_t size) {
  const size_t alignment = 128;
  // Always allocate buffers properly aligned to hold values of any type.
  size = alignedSize(size, alignment);
  auto buf =
      clCreateBuffer(context_, CL_MEM_READ_WRITE, size, nullptr, nullptr);
  GLOW_ASSERT(buf && "Allocation failed!");
  return buf;
}

void OCLBackend::freeDeviceBuffer(cl_mem buf) { clReleaseMemObject(buf); }
