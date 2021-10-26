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

#include "OpenCL.h"
#include "OpenCLDeviceManager.h"
#include "OpenCLTensorLayout.h"

#include "glow/Backend/BackendUtils.h"
#include "glow/CodeGen/MemoryAllocator.h"
#include "glow/Graph/Graph.h"
#include "glow/Graph/Nodes.h"
#include "glow/IR/IRUtils.h"
#include "glow/IR/Instrs.h"
#include "glow/Optimizer/IROptimizer/IROptimizer.h"
#include "glow/Quantization/Base/Base.h"
#include "glow/Support/Debug.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

#include <fstream>

#define DEBUG_TYPE "opencl"

using namespace glow;
using llvm::format;

using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;

typedef uint32_t cl_size_t;

/// We convert the OpenCL source code of kernels into an include file using an
/// external utility (include-bin). The resulting file is included here to
/// compile these sources into our the OpenCL backend. During the execution
/// these sources are compiled by the OpenCL driver.

// This defines kernels for optimized convolutions.
static const unsigned char kernels_fwd_conv_cl_src[] = {
#include "glow/OpenCL/kernels_fwd_conv.cl.inc"
};
static const size_t kernels_fwd_conv_cl_src_size =
    sizeof(kernels_fwd_conv_cl_src);

// This defines kernels for quantized optimized convolutions.
static const unsigned char kernels_fwd_quantized_conv_cl_src[] = {
#include "glow/OpenCL/kernels_fwd_quantized_conv.cl.inc"
};
static const size_t kernels_fwd_quantized_conv_cl_src_size =
    sizeof(kernels_fwd_quantized_conv_cl_src);

// Non-local memory using convolution for aggressive compile time
// specialization.
static const unsigned char kernels_specialized_no_local_mem_conv_cl_src[] = {
#include "glow/OpenCL/kernels_specialized_no_local_mem_conv.cl.inc"
};
static const size_t kernels_specialized_no_local_mem_conv_cl_src_size =
    sizeof(kernels_specialized_no_local_mem_conv_cl_src);

static llvm::cl::OptionCategory OpenCLBackendCat("Glow OpenCL Backend Options");

llvm::cl::opt<unsigned>
    clPlatformId("platform", llvm::cl::desc("OpenCL platform to be used"),
                 llvm::cl::init(0), llvm::cl::cat(OpenCLBackendCat));
llvm::cl::opt<int> clDeviceId(
    "device",
    llvm::cl::desc(
        "OpenCL device to be used. Default is to guess best device."),
    llvm::cl::init(-1), llvm::cl::cat(OpenCLBackendCat));
llvm::cl::opt<bool> clDoProfile("opencl-profile",
                                llvm::cl::desc("Profile OpenCL kernels"),
                                llvm::cl::init(false),
                                llvm::cl::cat(OpenCLBackendCat));

// Since conv_forward_mem is always specialized, this actually affects only
// the non-local memory using non-quantized one currently, but can be extended
// later to cover also the quantized one.
llvm::cl::opt<bool> clSpecializeConvolution(
    "opencl-specialize-convolution",
    llvm::cl::desc("Aggressively specialize convolution kernel launches."),
    llvm::cl::init(false), llvm::cl::cat(OpenCLBackendCat));

llvm::cl::opt<std::string> clDeviceProgramCacheDir(
    "opencl-program-cache-dir",
    llvm::cl::desc("The program disk cache directory for the "
                   "used device. If empty, disk caching "
                   "disabled."),
    llvm::cl::init(""), llvm::cl::cat(OpenCLBackendCat));

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

/// Add an macro definition with a string value to the set of options.
static void addStringOption(std::vector<std::string> &options,
                            const std::string &name, const std::string &value) {
  options.push_back("-D" + name + "=" + value);
}

OpenCLFunction::OpenCLFunction(std::unique_ptr<IRFunction> F,
                               runtime::RuntimeBundle &&bundle,
                               TraceInfo traceInfo)
    : CompiledFunction(std::move(bundle)), F_(std::move(F)) {
  // We need to go through the TraceInfo and pull out some info about manual
  // TraceEvents.
  for (const auto &backingPair : traceInfo.events) {
    Placeholder *backing = backingPair.first;
    for (const auto &event : backingPair.second) {
      // Context is the name of the TraceEventNode.
      manualTraceEvents_.emplace(event.context,
                                 std::make_pair(backing, &event));
    }
  }
  setTraceInfo(std::move(traceInfo));
}

void OpenCLFunction::freeCompilationResources() {
  runtimeBundle_.freeConstants();
}

OpenCLFunction::~OpenCLFunction() {
  for (auto &kv : programsCache_) {
    auto prog = kv.second;
    clReleaseProgram(prog);
  }
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
  case ElemKind::Int64ITy:
    return name + "_uW";
  case ElemKind::BoolTy:
    return name + "_bW";
  default:
    LOG(FATAL) << "Unsupported data type: "
               << Type::getElementName(elemTy).str();
  }
}

cl_kernel OpenCLFunction::createKernel(const std::string &name,
                                       cl_program program) {
  DCHECK(program) << "program cannot be null.";
  cl_int err = CL_SUCCESS;
  cl_kernel kernel = clCreateKernel(program, name.c_str(), &err);
  CHECK(kernel) << "clCreateKernel Failed for " << name;
  CHECK_EQ(err, CL_SUCCESS) << "clCreateKernel Failed.";
  return kernel;
}

std::string OpenCLFunction::deviceProgramCacheDir(cl_device_id deviceId) {
  // Support only a single device and device program cache directory
  // for now.
  return clDeviceProgramCacheDir;
}

std::string
OpenCLFunction::diskCacheProgramFileName(cl_device_id deviceId,
                                         const std::string &source,
                                         const std::string &options) {
  std::ostringstream hashString;
  hashString << source << options;
  return std::to_string(std::hash<std::string>{}(hashString.str())) + ".clb";
}

cl_program OpenCLFunction::loadProgramFromDiskCache(std::string cacheDirectory,
                                                    std::string programFileName,
                                                    cl_context ctx,
                                                    cl_device_id device) {
  std::string programPath = cacheDirectory + '/' + programFileName;
  if (!llvm::sys::fs::exists(programPath))
    return nullptr;

  uint64_t binarySize;
  std::error_code errc = llvm::sys::fs::file_size(programPath, binarySize);
  CHECK(!errc) << "Error getting the file size of " << programPath << ".";

  size_t binarySizeST = binarySize;

  std::ifstream binFile(programPath.c_str(), std::ios::binary);
  CHECK(binFile) << "Error opening " << programPath << " for reading.";

  auto binary = glow::make_unique<unsigned char[]>(binarySize);
  binFile.read((char *)binary.get(), binarySize);
  CHECK(binFile) << "Could not read the binary.";
  binFile.close();

  const unsigned char *binPtr = binary.get();

  cl_int status;
  cl_int err;
  cl_program prog = clCreateProgramWithBinary(ctx, 1, &device, &binarySizeST,
                                              &binPtr, &status, &err);
  if (err != CL_SUCCESS) {
    // The binary might be corrupted (e.g. process killed during write,
    // or incompatible OpenCL driver update). Just delete the cached
    // binary silently so we generate a fresh one.
    llvm::sys::fs::remove(programPath);
    return nullptr;
  }
  return prog;
}

void OpenCLFunction::saveProgramToDiskCache(std::string cacheDirectory,
                                            std::string programFilename,
                                            cl_program program, cl_context ctx,
                                            cl_device_id deviceId) {

  std::string programPath = cacheDirectory + '/' + programFilename;
  if (!llvm::sys::fs::is_directory(cacheDirectory)) {
    std::error_code err = llvm::sys::fs::create_directories(cacheDirectory);
    CHECK(!err) << "Could not create the OpenCL program disk cache directory "
                << cacheDirectory << ".";
  }
  cl_uint numDevices = 0;
  cl_int errC = clGetProgramInfo(program, CL_PROGRAM_NUM_DEVICES,
                                 sizeof(cl_uint), &numDevices, nullptr);
  CHECK_EQ(errC, CL_SUCCESS)
      << "clGetProgramInfo for CL_PROGRAM_NUM_DEVICES failed";
  CHECK_EQ(numDevices, 1) << "Only one OpenCL device supported";

  size_t binSize = 0;
  errC = clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t *),
                          &binSize, nullptr);
  CHECK_EQ(errC, CL_SUCCESS)
      << "clGetProgramInfo for CL_PROGRAM_BINARY_SIZES failed";

  std::unique_ptr<unsigned char[]> bin =
      glow::make_unique<unsigned char[]>(binSize);
  unsigned char *binPtr = bin.get();
  errC =
      clGetProgramInfo(program, CL_PROGRAM_BINARIES, binSize, &binPtr, nullptr);
  CHECK_EQ(errC, CL_SUCCESS)
      << "clGetProgramInfo for CL_PROGRAM_BINARIES failed.";

  std::ofstream binFile(programPath.c_str(),
                        std::ios::binary | std::ios::trunc);
  CHECK(binFile) << "Error opening " << programPath << " for writing.";

  binFile.write((const char *)bin.get(), binSize);
  CHECK(binFile) << "Could not write binary to " << programPath << ".";
  binFile.close();
}

cl_program
OpenCLFunction::createProgram(const std::string &source,
                              const std::vector<std::string> &options,
                              cl_command_queue queue) {
  const char *src = source.c_str();
  cl_context ctx;
  cl_int err = clGetCommandQueueInfo(queue, CL_QUEUE_CONTEXT, sizeof(ctx), &ctx,
                                     nullptr);
  CHECK_EQ(err, CL_SUCCESS) << "clGetCommandQueueInfo Failed.";
  cl_device_id deviceId;
  err = clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE, sizeof(deviceId),
                              &deviceId, nullptr);
  CHECK_EQ(err, CL_SUCCESS) << "clGetCommandQueueInfo Failed.";

  // Check if this program was compiled with the same parameters for the
  // provided context and device.
  std::string combinedOptions;
  for (auto &opt : options) {
    combinedOptions.append(opt).append(" ");
  }

  if (DIM_T_BITWIDTH == 32) {
    combinedOptions.append("-Ddim_t=uint -Dsdim_t=int");
  } else if (DIM_T_BITWIDTH == 64) {
    combinedOptions.append("-Ddim_t=ulong -Dsdim_t=long");
  } else {
    static_assert(DIM_T_BITWIDTH == 32 || DIM_T_BITWIDTH == 64,
                  "Unsupported dim_t width.");
  }

  ProgramKey key = std::make_tuple(source, combinedOptions, deviceId, ctx);
  cl_program &program = programsCache_[key];
  if (program) {
    return program;
  }

  const bool useDiskCache = deviceProgramCacheDir(deviceId) != "";
  bool loadedFromCache = false;

  std::string cacheDir;
  std::string programFileName;
  if (useDiskCache) {
    cacheDir = deviceProgramCacheDir(deviceId);
    programFileName =
        diskCacheProgramFileName(deviceId, source, combinedOptions);
    program =
        loadProgramFromDiskCache(cacheDir, programFileName, ctx, deviceId);
    loadedFromCache = program != nullptr;
  }

  if (program == nullptr) {
    // Create a new compiled program from the source. This will also add the
    // program to the in-memory program cache because 'program' is a reference
    // to an existing cache item.
    program = clCreateProgramWithSource(ctx, 1, &src, nullptr, &err);
    CHECK(program) << "clCreateProgramWithSource failed.";
  }
  err = clBuildProgram(program, 0, nullptr, combinedOptions.c_str(), nullptr,
                       nullptr);
  if (err) {
    dumpCompileLog(deviceId, program);
  }
  CHECK_EQ(err, CL_SUCCESS) << "clBuildProgram Failed.";

  if (useDiskCache && !loadedFromCache) {
    saveProgramToDiskCache(cacheDir, programFileName, program, ctx, deviceId);
  }

  return program;
}

template <class T>
static void setKernelArg(cl_kernel kernel, unsigned argIdx, T value) {
  cl_int err = clSetKernelArg(kernel, argIdx, sizeof(T), &value);
  CHECK_EQ(err, CL_SUCCESS) << "Unable to set parameter";
}

/// Set OpenCL \p kernel arguments using the buffer operands of the
/// instruction \p I. The first of these arguments should be passed to the \p
/// kernel at index \p nextKernelArgIdx. The \p bundle provides symbolTable, a
/// mapping from Values to on-device buffer offsets of these values.
///
/// \returns the index of the last set OpenCL kernel argument.
static size_t setKernelArgsForBuffers(cl_kernel kernel, const Instruction &I,
                                      size_t nextKernelArgIdx,
                                      runtime::RuntimeBundle &bundle) {
  // Number of instruction operands.
  auto numArgs = I.getNumOperands();
  // The predicate of the instruction if available.
  Value *predicate = I.hasPredicate() ? I.getPredicate() : nullptr;
  // The index of the kernel argument to be set.
  unsigned kernelArgIdx = nextKernelArgIdx;
  // Go over all operands and pass buffer operands to the kernel.
  for (unsigned arg = 0; arg < numArgs; arg++) {
    auto *value = I.getOperand(arg).first;
    // Ignore predicate operands as they are not supported by the OpenCL backend
    // yet.
    if (value == predicate)
      continue;
    // The value is a buffer that should be passed as a kernel argument.
    setKernelArg<cl_uint>(kernel, kernelArgIdx, bundle.getValueOffset(value));
    kernelArgIdx++;
  }
  return kernelArgIdx - 1;
}

/// \returns the preferred (intra) vector width for the given OpenCL \p device,
/// and the given \p elementType.
static unsigned getPreferredVectorWidth(cl_device_id device,
                                        ElemKind elementType) {
  cl_uint width;
  cl_device_info paramName;
  switch (elementType) {
  case ElemKind::FloatTy:
    paramName = CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT;
    break;
  case ElemKind::BoolTy:
  case ElemKind::Int8QTy:
    paramName = CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR;
    break;
  case ElemKind::Int32QTy:
    paramName = CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT;
    break;
  case ElemKind::Int64ITy:
    paramName = CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG;
    break;
  default:
    LOG(FATAL) << "Unsupported vector data type: "
               << Type::getElementName(elementType).str();
  }
  clGetDeviceInfo(device, paramName, sizeof(width), &width, NULL);
  return width;
}

void OpenCLFunction::fillBuffer(cl_mem buffer, uint64_t start, uint64_t len,
                                float value, ElemKind elemKind,
                                runtime::OpenCLDeviceBindings *devBindings) {
  auto kernel =
      createKernel(getKernelName("splat", elemKind), devBindings->program);
  setKernelArg(kernel, 0, buffer);
  setKernelArg<cl_uint>(kernel, 1, start);
  setKernelArg(kernel, 2, value);
  enqueueKernel("splat", devBindings->commandQueue, kernel,
                devBindings->deviceId, {(size_t)len},
                devBindings->kernelLaunches);
}

/// \returns the max local workgroup size for each dimension, under the
/// opencl constraints, with the global workgroup sizes of \p global;
static void getMaxLocalWorkgroupSize(cl_kernel kernel, cl_device_id device,
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

  CHECK_EQ(err, CL_SUCCESS) << "Error in clGetKernelWorkGroupInfo.";
  // The global workgroup size must be a multiple of the local workgroup size,
  // and less than the max size for the specific dimension. Also, the
  // multiplication of all dimensions (size of total local work) needs to be
  // less than WSG. In here we find the highest L that divides the global
  // workgroup size. This is our naive implementation of gcd, with the other
  // constraints:
  size_t totalWorkPrevDims = 1;
  for (int i = 0, e = global.size(); i < e; i++) {
    local[i] = std::min(L, WIS[i]);
    local[i] = std::min(local[i], WGS / totalWorkPrevDims);

    while (global[i] % local[i] || L % local[i]) {
      local[i]--;
    }

    // Remember how much work we are doing in this dimension. Use it to make
    // sure that the next dimensions don't exceed the total allowed workgroup
    // size.
    totalWorkPrevDims *= local[i];
  }
}

void OpenCLFunction::enqueueKernel(llvm::StringRef name,
                                   cl_command_queue commands, cl_kernel kernel,
                                   cl_device_id device,
                                   llvm::ArrayRef<size_t> global,
                                   llvm::ArrayRef<size_t> local,
                                   std::vector<KernelLaunch> &kernelLaunches) {
  char kernelType[128];
  size_t retSize;
  cl_int err = clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME,
                               sizeof(kernelType), &kernelType, &retSize);
  CHECK_EQ(err, CL_SUCCESS) << "Error in clGetKernelInfo.";

  cl_event event{nullptr};
  bool profile = kernelProfiling_;
  err = clEnqueueNDRangeKernel(commands, kernel, global.size(), nullptr,
                               &global[0], &local[0], 0, nullptr,
                               profile ? &event : nullptr);
  CHECK_EQ(err, CL_SUCCESS) << "Error in clEnqueueNDRangeKernel.";
  kernelLaunches.push_back(KernelLaunch(kernel, name.str(), kernelType, event));
}

/// Enqueue a \p kernel for execution on the command queue \p commands on a
/// given \p device. The information about the launched kernel will be added to
/// \p kernelLaunches list.
void OpenCLFunction::enqueueKernel(llvm::StringRef name,
                                   cl_command_queue commands, cl_kernel kernel,
                                   cl_device_id device,
                                   llvm::ArrayRef<size_t> global,
                                   std::vector<KernelLaunch> &kernelLaunches) {
  llvm::SmallVector<size_t, 4> local(global.size(), 0);
  getMaxLocalWorkgroupSize(kernel, device, global, local);
  char kernelType[128];
  size_t retSize;
  cl_int err = clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME,
                               sizeof(kernelType), &kernelType, &retSize);
  CHECK_EQ(err, CL_SUCCESS) << "Error in clGetKernelInfo.";

  cl_event event{nullptr};
  bool profile = kernelProfiling_;
  err = clEnqueueNDRangeKernel(commands, kernel, global.size(), nullptr,
                               &global[0], &local[0], 0, nullptr,
                               profile ? &event : nullptr);
  CHECK_EQ(err, CL_SUCCESS) << "Error in clEnqueueNDRangeKernel.";
  kernelLaunches.push_back(
      KernelLaunch(kernel, name.str(), kernelType, profile ? event : nullptr));
}

void OpenCLFunction::executeNCHWConvolution(
    const ConvolutionInst *CC, ExecutionContext *executionContext,
    runtime::OpenCLDeviceBindings *devBindings) {
  auto input = CC->getSrc();
  auto output = CC->getDest();
  auto bias = CC->getBias();
  auto weights = CC->getFilter();
  auto odim = ShapeNCHW(CC->getDest()->getType()->dims());
  auto idim = ShapeNCHW(CC->getSrc()->getType()->dims());
  auto fdim = ShapeNCHW(CC->getFilter()->getType()->dims());
  bool isQuantized = output->getType()->isQuantizedType();
  PaddingTLBR pads(CC->getPads());
  ShapeHW kdim(CC->getKernels());
  ShapeHW sdim(CC->getStrides());
  unsigned group = CC->getGroup();
  auto dilation = CC->getDilation();
  // So far, we don't support fast convolution kernel if group > 1.
  // For group convolution, the slow convolution kernel should be invoked.
  // The following debug checks should be removed once the group > 1 is
  // supported in fast convolution kernel.
  DCHECK_EQ(group, 1)
      << "Group Convolution is not supported by OpenCL backend's "
         "fast convolution kernel.";

  // Create options for compiling the program.
  // Don't use names M, N, K as they are defined in precompiled headers.

  std::vector<std::string> options;
  // Number of spacial axes.
  addIntOption(options, "v_nax", 2);
  // Number of groups.
  addIntOption(options, "v_g", group);
  // Parameters for kernel size, padding and stride
  addIntOption(options, "v_k_0", kdim.height);
  addIntOption(options, "v_k_1", kdim.width);
  addIntOption(options, "v_p_0", pads.top);
  addIntOption(options, "v_p_1", pads.left);
  addIntOption(options, "v_s_0", sdim.height);
  addIntOption(options, "v_s_1", sdim.width);

  // Dilation.
  addIntOption(options, "v_d_0", dilation[0]);
  addIntOption(options, "v_d_1", dilation[1]);

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

  if (CC->getFusedActivation() == FusedActivation::RELU) {
    addIntOption(options, "v_fuse_relu", 1);
  } else {
    addIntOption(options, "v_fuse_relu", 0);
  }

  // Determine the work groups sizes along h and w.
  size_t WIS[3];
  cl_int err =
      clGetDeviceInfo(devBindings->deviceId, CL_DEVICE_MAX_WORK_ITEM_SIZES,
                      sizeof(WIS), &WIS, nullptr);
  CHECK_EQ(err, CL_SUCCESS) << "Could not execute clGetDeviceInfo";

  size_t dev_max_wg_size;
  err = clGetDeviceInfo(devBindings->deviceId, CL_DEVICE_MAX_WORK_GROUP_SIZE,
                        sizeof(dev_max_wg_size), &dev_max_wg_size, nullptr);
  CHECK_EQ(err, CL_SUCCESS) << "Could not execute clGetDeviceInfo";

  size_t wg_size[3];

  for (int id = 0; id < 2; ++id) {
    size_t defaultVal = 16;
    // Special case on CPUs devices, where a workgroup size could be 1,
    // e.g. in case of Apple's OpenCL driver for CPUs.
    if (WIS[id] < defaultVal || (id == 0 && WIS[1] < defaultVal)) {
      defaultVal = WIS[1];
    }
    if (id == 1 && defaultVal * wg_size[0] > dev_max_wg_size)
      defaultVal = dev_max_wg_size / wg_size[0];
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
  std::string src;
  if (isQuantized) {
    src.append(
        reinterpret_cast<const char *>(kernels_fwd_quantized_conv_cl_src),
        kernels_fwd_quantized_conv_cl_src_size);
  } else {
    src.append(reinterpret_cast<const char *>(kernels_fwd_conv_cl_src),
               kernels_fwd_conv_cl_src_size);
  }

  TRACE_EVENT_SCOPE_NAMED(executionContext->getTraceContext(),
                          TraceLevel::RUNTIME, "convCreateProgram", cpEvent);
  auto prog = createProgram(src, options, devBindings->commandQueue);
  TRACE_EVENT_SCOPE_END_NAMED(cpEvent);

  auto kernelName = isQuantized ? "conv_forward_mem_i8" : "conv_forward_mem";
  auto kernel = createKernel(kernelName, prog);
  setKernelArg(kernel, 0, devBindings->deviceBuffer);
  setKernelArg<cl_uint>(kernel, 1, runtimeBundle_.getValueOffset(input));
  setKernelArg<cl_uint>(kernel, 2, runtimeBundle_.getValueOffset(weights));
  setKernelArg<cl_uint>(kernel, 3, runtimeBundle_.getValueOffset(bias));
  setKernelArg<cl_uint>(kernel, 4, runtimeBundle_.getValueOffset(output));

  // Extra options for quantized kernel
  if (isQuantized) {
    auto inputTy = CC->getSrc()->getType();
    auto outputTy = CC->getDest()->getType();
    auto biasTy = CC->getBias()->getType();
    auto weightsTy = CC->getFilter()->getType();
    setKernelArg(kernel, 5, weightsTy->getOffset());
    setKernelArg(kernel, 6, weightsTy->getScale());
    setKernelArg(kernel, 7, inputTy->getOffset());
    setKernelArg(kernel, 8, inputTy->getScale());
    setKernelArg(kernel, 9, outputTy->getOffset());
    setKernelArg(kernel, 10, outputTy->getScale());
    setKernelArg(kernel, 11, biasTy->getOffset());
    setKernelArg(kernel, 12, biasTy->getScale());
  }

  // Compute proper parameters for global work and workgroups.
  auto fw_wgs0 = wg_size[0];
  auto fw_wgs1 = wg_size[1];
  int fw_wptn = WPTN;
  int fw_wptm = WPTM;
  int fw_div_N = fw_wptn * fw_wgs0;
  int fw_div_M = fw_wptm * fw_wgs1;
  int N_FW_ = odim.h * odim.w;
  int M_FW_ = odim.c / group;
  size_t max_kern_wg_size;
  clGetKernelWorkGroupInfo(kernel, devBindings->deviceId,
                           CL_KERNEL_WORK_GROUP_SIZE, sizeof(max_kern_wg_size),
                           &max_kern_wg_size, nullptr);
  CHECK_LE(fw_wgs0 * fw_wgs1, max_kern_wg_size) << "Bad workgroup size";

  // Set the size of a workgroup.
  std::vector<size_t> local = {fw_wgs0, fw_wgs1, 1};

  // Set the global work size.
  std::vector<size_t> global = {((N_FW_ - 1) / fw_div_N + 1) * fw_wgs0,
                                ((M_FW_ - 1) / fw_div_M + 1) * fw_wgs1,
                                idim.n * group};

  enqueueKernel(CC->getName(), devBindings->commandQueue, kernel,
                devBindings->deviceId, global, local,
                devBindings->kernelLaunches);
}

/// This method is copied from InterpreterNodes.cpp. Please be aware that
/// they should be in sync.
template <typename T>
static void topK(Tensor &outW, Tensor &indW, Tensor &inW, size_t k) {
  auto values = outW.getHandle<T>();
  auto indices = indW.getHandle<int64_t>();
  auto in = inW.getHandle<T>();
  size_t n = in.dims().back();

  size_t in_p = 0, out_p = 0;
  size_t tensor_end = in.size();
  using pairType = std::pair<float, size_t>;
  std::vector<pairType> buf(n);

  while (in_p < tensor_end) {
    for (size_t i = 0; i < n; i++) {
      buf[i].first = in.raw(in_p++);
      buf[i].second = i;
    }
    // NOTE: it's possible to do N + KlogK, while this version is NlogN
    std::sort(buf.begin(), buf.end(), [](const pairType &a, const pairType &b) {
      if (a.first != b.first)
        return a.first > b.first;
      return a.second < b.second;
    });
    for (size_t i = 0; i < k; i++) {
      values.raw(out_p) = buf[i].first;
      indices.raw(out_p) = buf[i].second;
      out_p++;
    }
  }
}

static ShapeNHWC shapeFromDims(llvm::ArrayRef<dim_t> arr) {
  assert(arr.size() <= 4);
  llvm::SmallVector<dim_t, 4> ones(4, 1);
  std::copy(arr.begin(), arr.end(), ones.begin());
  return ShapeNHWC(llvm::ArrayRef<dim_t>(ones));
};

Error OpenCLFunction::execute(ExecutionContext *context) {
  auto clBindings = static_cast<runtime::OpenCLDeviceBindings *>(
      context->getDeviceBindings());

  auto deviceBuffer = clBindings->deviceBuffer;
  auto deviceId = clBindings->deviceId;
  auto commands = clBindings->commandQueue;
  auto program = clBindings->program;
  std::vector<KernelLaunch> &kernelLaunches = clBindings->kernelLaunches;

  kernelProfiling_ = clDoProfile || getTraceInfo().autoInstrumented;

  TRACE_EVENT_SCOPE_NAMED(context, TraceLevel::RUNTIME, "enqueueKernels",
                          enqueueEvent);
  for (const auto &I : F_->getInstrs()) {
    // Skip memory allocation instructions as they are NOPs.
    if (isa<AllocActivationInst>(I) || isa<DeallocActivationInst>(I) ||
        isa<TensorViewInst>(I) || isa<TouchInst>(I)) {
      continue;
    }
    // The kernels are named after the name of the instruction, plus the "W"
    // suffix to prevent name colissions for functions like 'tanh' that are also
    // a part of the OpenCL runtime.
    auto elemTy = I.getNumOperands() ? I.getOperand(0).first->getElementType()
                                     : ElemKind::FloatTy;

    // If ElementCmpLTEInst then the first operand is always bool, so instead
    // set the element type based on the LHS input.
    if (auto *LTE = dyn_cast<ElementCmpLTEInst>(&I)) {
      elemTy = LTE->getLHS()->getElementType();
    }

    std::string kernelName = getKernelName(I.getKindName(), elemTy);

    //  Check if the instruction is quantized. Consider an instruction to be
    //  quantized if its destination or source operands are quantized.
    bool isQuantized = I.getNumOperands() &&
                       (I.getOperand(0).first->getType()->isQuantizedType() ||
                        I.getOperand(I.getNumOperands() - 1)
                            .first->getType()
                            ->isQuantizedType());

    // Element-wise operations, except the copy instruction.
    if (I.isDataParallel() && !isa<CopyInst>(I)) {
      // Figure out how many element-wise elements are there to process:
      size_t global;
      if (I.isDataParallel()) {
        global = I.getOperand(0).first->getType()->size();
        // The check for quantization below is a temporary workaround until the
        // corresponding kernels are implemented for the quantized operations.
        if (!isQuantized) {
          if (getPreferredVectorWidth(deviceId, elemTy) == 1) {
            // If the device prefers not to use vector data types, let's not.
          } else if (global % 16 == 0) {
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
        }
      } else {
        LOG(FATAL) << "Invalid instruction: " << I.getName().str();
      }

      cl_kernel kernel = createKernel(kernelName, program);
      setKernelArg(kernel, 0, deviceBuffer);
      auto numArgs = setKernelArgsForBuffers(kernel, I, 1, runtimeBundle_);
      auto numMandatoryArgs = numArgs;
      (void)numMandatoryArgs;

      if (auto *SI = dyn_cast<SplatInst>(&I)) {
        // Pass the splat as a parameter.
        if (!isQuantized) {
          setKernelArg(kernel, ++numArgs, SI->getValue());
        } else {
          auto *destTy = SI->getDest()->getType();
          TensorQuantizationParams destQ{destTy->getScale(),
                                         destTy->getOffset()};
          float val = SI->getValue();
          int8_t int8Val = quantization::quantize(val, destQ);
          setKernelArg<float>(kernel, ++numArgs, static_cast<float>(int8Val));
        }
      }

      if (isQuantized) {
        if (isa<ElementAddInst>(I) || isa<ElementSubInst>(I) ||
            isa<ElementMulInst>(I) || isa<ElementDivInst>(I) ||
            isa<ElementMinInst>(I) || isa<ElementMaxInst>(I)) {
          int32_t destOffset = I.getOperand(0).first->getType()->getOffset();
          float destScale = I.getOperand(0).first->getType()->getScale();

          auto LHSTy = I.getOperand(1).first->getType();
          auto RHSTy = I.getOperand(2).first->getType();

          auto lhsScaleParams = quantization::quantizeScaleOffset32To8(
              LHSTy->getScale() / destScale, LHSTy->getOffset());
          auto rhsScaleParams = quantization::quantizeScaleOffset32To8(
              RHSTy->getScale() / destScale, RHSTy->getOffset());
          setKernelArg(kernel, ++numArgs, destOffset);
          setKernelArg(kernel, ++numArgs, lhsScaleParams);
          setKernelArg(kernel, ++numArgs, rhsScaleParams);
          if (isa<ElementMulInst>(I) || isa<ElementDivInst>(I)) {
            float resultScale =
                isa<ElementMulInst>(I)
                    ? LHSTy->getScale() * RHSTy->getScale() / destScale
                    : LHSTy->getScale() / (RHSTy->getScale() * destScale);
            auto resultScaleParams =
                quantization::quantizeScaleOffset32To8(resultScale, 0);
            setKernelArg(kernel, ++numArgs, resultScaleParams);
          }
        } else if (auto *RI = dyn_cast<ReluInst>(&I)) {
          int32_t destOffset = RI->getDest()->getType()->getOffset();
          float destScale = RI->getDest()->getType()->getScale();

          auto srcTy = RI->getSrc()->getType();

          auto srcScaleParams = quantization::quantizeScaleOffset32To8(
              srcTy->getScale() / destScale, srcTy->getOffset());
          setKernelArg(kernel, ++numArgs, destOffset);
          setKernelArg(kernel, ++numArgs, srcScaleParams);
        }
        // Quantize floating point tensor. Scale and Offset are based on return
        // type of the instruction \p I.
        if (auto *QI = dyn_cast<QuantizeInst>(&I)) {
          float destTensorQuantizationScale =
              QI->getDest()->getType()->getScale();
          int32_t destTensorQuantizationOffset =
              QI->getDest()->getType()->getOffset();
          setKernelArg(kernel, ++numArgs, destTensorQuantizationScale);
          setKernelArg(kernel, ++numArgs, destTensorQuantizationOffset);
        }
        // Rescale quantized tensor. Scale and Offset are based on return type
        // of the instruction \p I.
        if (auto *RQI = dyn_cast<RescaleQuantizedInst>(&I)) {
          auto *dest = RQI->getDest();
          auto *src = RQI->getSrc();
          auto *destType = dest->getType();
          auto *srcType = src->getType();
          auto rescaleParams = quantization::quantizeScaleOffset32To8(
              srcType->getScale() / destType->getScale(), srcType->getOffset());

          setKernelArg(kernel, ++numArgs, destType->getOffset());
          setKernelArg(kernel, ++numArgs, srcType->getOffset());
          setKernelArg(kernel, ++numArgs, rescaleParams);
        }
        // Dequantize integer tensor. Scale and Offset are based
        // on the source tensor type.
        if (auto *QI = dyn_cast<DequantizeInst>(&I)) {
          float srcTensorQuantizationScale =
              QI->getSrc()->getType()->getScale();
          int32_t srcTensorQuantizationOffset =
              QI->getSrc()->getType()->getOffset();
          setKernelArg(kernel, ++numArgs, srcTensorQuantizationScale);
          setKernelArg(kernel, ++numArgs, srcTensorQuantizationOffset);
        }
      }

      if (isQuantized) {
        DCHECK_GT(numArgs, numMandatoryArgs) << "Not enough kernel arguments";
      }
      enqueueKernel(I.getName(), commands, kernel, deviceId, {global},
                    kernelLaunches);
      continue;
    }

    if (auto *SM = dyn_cast<SoftMaxInst>(&I)) {
      // Implement Softmax by parallelizing the batch dimension. Each sample in
      // the batch is processed by a different parallel 'thread'.
      cl_kernel kernel = createKernel(kernelName, program);
      setKernelArg(kernel, 0, deviceBuffer);
      auto numArgs = setKernelArgsForBuffers(kernel, I, 1, runtimeBundle_);

      // This is the number of elements for each slice. There are N slices in
      // our batch.
      auto inputDims = SM->getSrc()->getType()->dims();
      size_t numSlices = inputDims[0];

      // Pass the slice size (size of each sample in the batch) as a parameter.
      setKernelArg<cl_uint>(kernel, numArgs + 1, flattenCdr(inputDims).second);

      enqueueKernel(I.getName(), commands, kernel, deviceId, {numSlices},
                    kernelLaunches);
      continue;
    }

    if (auto *SM = dyn_cast<SoftMaxGradInst>(&I)) {
      // Implement Softmax by parallelizing the batch dimension. Each sample in
      // the batch is processed by a different parallel 'thread'.
      cl_kernel kernel = createKernel(kernelName, program);
      setKernelArg(kernel, 0, deviceBuffer);
      auto numArgs = setKernelArgsForBuffers(kernel, I, 1, runtimeBundle_);

      // This is the number of elements for each slice. There are N slices in
      // our batch.
      auto inputDims = SM->getSrcGrad()->getType()->dims();
      size_t numSlices = inputDims[0];

      // Pass the slice size (size of each sample in the batch) as a parameter.
      setKernelArg<cl_uint>(kernel, numArgs + 1, flattenCdr(inputDims).second);

      enqueueKernel(I.getName(), commands, kernel, deviceId, {numSlices},
                    kernelLaunches);
      continue;
    }

    if (auto *ET = dyn_cast<ExtractTensorInst>(&I)) {
      cl_kernel kernel = createKernel(kernelName, program);
      setKernelArg(kernel, 0, deviceBuffer);
      auto numArgs = setKernelArgsForBuffers(kernel, I, 1, runtimeBundle_);

      // Currently support tensors up to 4 dimensions.
      // TODO: Handle other dimensions.
      assert(ET->getDest()->getType()->dims().size() <= 4);

      ShapeNHWC odim = shapeFromDims(ET->getDest()->getType()->dims());
      ShapeNHWC idim = shapeFromDims(ET->getSrc()->getType()->dims());
      ShapeNHWC offset = shapeFromDims(ET->getOffsets());

      setKernelArg(kernel, numArgs + 1, odim);
      setKernelArg(kernel, numArgs + 2, idim);
      setKernelArg(kernel, numArgs + 3, offset);
      enqueueKernel(I.getName(), commands, kernel, deviceId, {odim.n, odim.h},
                    kernelLaunches);
      continue;
    }

    if (auto *IT = dyn_cast<InsertTensorInst>(&I)) {
      cl_kernel kernel = createKernel(kernelName, program);
      setKernelArg(kernel, 0, deviceBuffer);
      auto numArgs = setKernelArgsForBuffers(kernel, I, 1, runtimeBundle_);

      // Currently support tensors of up to 4 dimensions.
      // TODO: Handle other dimensions.
      assert(IT->getDest()->getType()->dims().size() <= 4);

      ShapeNHWC odim = shapeFromDims(IT->getDest()->getType()->dims());
      ShapeNHWC idim = shapeFromDims(IT->getSrc()->getType()->dims());
      ShapeNHWC offset = shapeFromDims(IT->getOffsets());

      setKernelArg(kernel, numArgs + 1, odim);
      setKernelArg(kernel, numArgs + 2, idim);
      setKernelArg(kernel, numArgs + 3, offset);
      setKernelArg<cl_uint>(kernel, numArgs + 4, IT->getCount());
      setKernelArg<cl_uint>(kernel, numArgs + 5, IT->getAxis());
      enqueueKernel(I.getName(), commands, kernel, deviceId, {idim.n, idim.h},
                    kernelLaunches);
      continue;
    }

    if (auto *BMM = dyn_cast<MatMulInst>(&I)) {
      // Size of the tile to be used for matrix multiplication.
      constexpr size_t TILE_DIM = 8;

      // Determine max work groups sizes.
      size_t WIS[3];
      cl_int err = clGetDeviceInfo(deviceId, CL_DEVICE_MAX_WORK_ITEM_SIZES,
                                   sizeof(WIS), &WIS, nullptr);
      CHECK_EQ(err, CL_SUCCESS) << "Could not execute clGetDeviceInfo";
      // True if the tiled matrix multiplication kernel can be used. This is
      // only possible if the device allows workgroups with sizes which are at
      // least as big as a tile.
      bool useTiledMatMul = (WIS[0] >= TILE_DIM && WIS[1] >= TILE_DIM);
      auto tiledKernelName = isQuantized ? "matmul_tiled_i8" : "matmul_tiled";
      cl_kernel kernel =
          createKernel(useTiledMatMul ? tiledKernelName : kernelName, program);
      setKernelArg(kernel, 0, deviceBuffer);
      auto numArgs = setKernelArgsForBuffers(kernel, I, 1, runtimeBundle_);

      ShapeNHWC ddim = shapeFromDims(BMM->getDest()->getType()->dims());
      ShapeNHWC ldim = shapeFromDims(BMM->getLHS()->getType()->dims());
      ShapeNHWC rdim = shapeFromDims(BMM->getRHS()->getType()->dims());

      setKernelArg(kernel, numArgs + 1, ddim);
      setKernelArg(kernel, numArgs + 2, ldim);
      setKernelArg(kernel, numArgs + 3, rdim);
      if (isQuantized) {
        auto lhsTy = BMM->getLHS()->getType();
        auto rhsTy = BMM->getRHS()->getType();
        auto destTy = BMM->getDest()->getType();
        auto destScaleParams = quantization::quantizeScaleOffset32To8(
            lhsTy->getScale() * rhsTy->getScale() / destTy->getScale(), 0);
        setKernelArg(kernel, numArgs + 4, lhsTy->getOffset());
        setKernelArg(kernel, numArgs + 5, rhsTy->getOffset());
        setKernelArg(kernel, numArgs + 6, destTy->getOffset());
        setKernelArg(kernel, numArgs + 7, destScaleParams);
      }

      if (useTiledMatMul) {
        std::vector<size_t> local{TILE_DIM, TILE_DIM};
        std::vector<size_t> global{(ddim.n / local[0] + 1) * local[0],
                                   (ddim.h / local[1] + 1) * local[1]};

        enqueueKernel(I.getName(), commands, kernel, deviceId, global, local,
                      kernelLaunches);
      } else {
        enqueueKernel(I.getName(), commands, kernel, deviceId,
                      {ddim.n, ddim.h, ddim.w}, kernelLaunches);
      }
      continue;
    }

    if (auto *BA = dyn_cast<BatchedAddInst>(&I)) {
      if (isQuantized &&
          BA->getSlice()->getType()->getElementType() == ElemKind::Int32QTy) {
        kernelName += "_32";
      }
      cl_kernel kernel = createKernel(kernelName, program);
      setKernelArg(kernel, 0, deviceBuffer);
      auto numArgs = setKernelArgsForBuffers(kernel, I, 1, runtimeBundle_);

      auto bdim = flattenCdr(BA->getBatch()->dims());
      setKernelArg<cl_uint>(kernel, numArgs + 1, bdim.first);
      setKernelArg<cl_uint>(kernel, numArgs + 2, bdim.second);

      if (isQuantized) {
        auto *destTy = BA->getDest()->getType();
        auto *batchTy = BA->getBatch()->getType();
        auto *sliceTy = BA->getSlice()->getType();

        setKernelArg(kernel, numArgs + 3, destTy->getOffset());

        float destScale = destTy->getScale();
        auto batchScaleParams = quantization::quantizeScaleOffset32To8(
            batchTy->getScale() / destScale, batchTy->getOffset());
        auto sliceScaleParams = quantization::quantizeScaleOffset32To8(
            sliceTy->getScale() / destScale, sliceTy->getOffset());

        setKernelArg(kernel, numArgs + 4, batchScaleParams);
        setKernelArg(kernel, numArgs + 5, sliceScaleParams);
      }

      // Parallelize on each element in the slice.
      enqueueKernel(I.getName(), commands, kernel, deviceId, {bdim.second},
                    kernelLaunches);
      continue;
    }

    if (auto *BRA = dyn_cast<OCLBatchedReduceAddInst>(&I)) {
      auto axis = BRA->getAxis();
      auto axisSrcSliceSize = BRA->getAxisSrcSliceSize();

      // Determine and store the slice sizes of each input dimension excluding
      // the reduce axis into batchSliceSizes. Determine also the slice size on
      // the reduce axis and store that separately. These are used by the kernel
      // to index correctly into the input buffer. If the input has one
      // dimension (that is also the reduce axis), store one slice of size 1
      // into batchSliceSizes.
      auto batchDims = BRA->getSrc()->getType()->dims();

      // Determine and store the slice sizes of each output dimension excluding
      // the reduce axis into destSliceSizes. These are used by the kernel to
      // index correctly into the output buffer. If the output has zero
      // dimensions store one slice of size 1 into destSliceSizes.
      auto destDims = BRA->getDest()->getType()->dims();
      std::vector<size_t> destDimsVec(destDims.begin(), destDims.end());
      if (destDims.empty()) {
        destDimsVec.emplace_back(1);
      }

      // Create kernel and set arguments.
      cl_kernel kernel = createKernel(kernelName, program);
      setKernelArg(kernel, 0, deviceBuffer);
      auto numArgs = setKernelArgsForBuffers(kernel, I, 1, runtimeBundle_);

      setKernelArg<cl_uint>(kernel, numArgs + 1, batchDims[axis]);
      setKernelArg<cl_uint>(kernel, numArgs + 2, axisSrcSliceSize);

      // Parallelize on each element in the slice.
      enqueueKernel(I.getName(), commands, kernel, deviceId, destDimsVec,
                    kernelLaunches);
      continue;
    }

    if (auto *LRN = dyn_cast<LocalResponseNormalizationGradInst>(&I)) {
      cl_kernel kernel = createKernel(kernelName, program);
      setKernelArg(kernel, 0, deviceBuffer);

      size_t numArgs = setKernelArgsForBuffers(kernel, I, 1, runtimeBundle_);
      ShapeNHWC dim(LRN->getDest()->getType()->dims());

      uint32_t halfWindowSize = LRN->getHalfWindowSize();
      uint32_t windowSize = 2 * halfWindowSize + 1;
      setKernelArg(kernel, ++numArgs, dim);
      setKernelArg(kernel, ++numArgs, halfWindowSize);
      setKernelArg(kernel, ++numArgs, LRN->getK());
      setKernelArg(kernel, ++numArgs, LRN->getBeta());
      setKernelArg(kernel, ++numArgs, LRN->getAlpha() / windowSize);

      enqueueKernel(I.getName(), commands, kernel, deviceId,
                    {dim.n, dim.h, dim.w}, kernelLaunches);
      continue;
    }

    if (auto *LRN = dyn_cast<LocalResponseNormalizationInst>(&I)) {
      cl_kernel kernel = createKernel(kernelName, program);
      setKernelArg(kernel, 0, deviceBuffer);

      size_t numArgs = setKernelArgsForBuffers(kernel, I, 1, runtimeBundle_);
      ShapeNHWC dim(LRN->getDest()->getType()->dims());

      uint32_t halfWindowSize = LRN->getHalfWindowSize();
      uint32_t windowSize = 2 * halfWindowSize + 1;
      setKernelArg(kernel, ++numArgs, dim);
      setKernelArg(kernel, ++numArgs, halfWindowSize);
      setKernelArg(kernel, ++numArgs, LRN->getK());
      setKernelArg(kernel, ++numArgs, LRN->getBeta());
      setKernelArg(kernel, ++numArgs, LRN->getAlpha() / windowSize);

      enqueueKernel(I.getName(), commands, kernel, deviceId,
                    {dim.h, dim.w, dim.c}, kernelLaunches);
      continue;
    }

    if (auto *CC = dyn_cast<ConvolutionInst>(&I)) {
      // For OpenCL backend, only NCHW convolution support non-square dilation
      if (CC->getLayout() == NCHW) {
        executeNCHWConvolution(CC, context, clBindings);
        continue;
      }

      if (CC->getFusedActivation() == FusedActivation::RELU) {
        kernelName += "_ReLU";
      }

      // This is a naive implementation that parallelizes using three dims:
      // the X and the Y in the output filter.
      cl_program prog = program;
      auto idim = ShapeNHWC(CC->getSrc()->getType()->dims());
      ShapeHW kdim(CC->getKernels());
      ShapeHW sdim(CC->getStrides());
      auto odim = ShapeNHWC(CC->getDest()->getType()->dims());
      ShapeNHWC kernelSize(CC->getFilter()->getType()->dims());
      auto pads = PaddingTLBR(CC->getPads());

      CHECK_EQ(CC->getDilation()[0], CC->getDilation()[1])
          << "Currently not support non-square dilation here";

      const bool specialize = clSpecializeConvolution && !isQuantized;
      std::string src;
      if (specialize) {
        // Specialize the kernel related to Conv node parameters to enable
        // aggressive constant propagation and other optimizations.
        std::vector<std::string> options;
        addIntOption(options, "CONVK_GROUP", CC->getGroup());
        addIntOption(options, "CONVK_BATCHES", idim.n);
        addIntOption(options, "CONVK_DILATION", CC->getDilation()[0]);
        addIntOption(options, "CONVK_KERNEL_W", kdim.width);
        addIntOption(options, "CONVK_KERNEL_H", kdim.height);
        addIntOption(options, "CONVK_STRIDES_W", sdim.width);
        addIntOption(options, "CONVK_STRIDES_H", sdim.height);
        addIntOption(options, "CONVK_IDIM_W", idim.w);
        addIntOption(options, "CONVK_IDIM_H", idim.h);
        addIntOption(options, "CONVK_IDIM_C", idim.c);
        addIntOption(options, "CONVK_ODIM_W", odim.w);
        addIntOption(options, "CONVK_ODIM_H", odim.h);
        addIntOption(options, "CONVK_ODIM_C", odim.c);
        addIntOption(options, "CONVK_PADS_TOP", pads.top);
        addIntOption(options, "CONVK_PADS_LEFT", pads.left);
        addIntOption(options, "CONVK_FILTER_W", kernelSize.w);
        addIntOption(options, "CONVK_FILTER_H", kernelSize.h);
        addIntOption(options, "CONVK_FILTER_C", kernelSize.c);
        src.append(reinterpret_cast<const char *>(
                       kernels_specialized_no_local_mem_conv_cl_src),
                   kernels_specialized_no_local_mem_conv_cl_src_size);
        prog = createProgram(src, options, commands);
      }

      cl_kernel kernel = createKernel(kernelName, prog);
      setKernelArg(kernel, 0, deviceBuffer);
      auto numArgs = setKernelArgsForBuffers(kernel, I, 1, runtimeBundle_);

      if (!specialize) {
        setKernelArg(kernel, numArgs + 1, kdim);
        setKernelArg(kernel, numArgs + 2, sdim);
        setKernelArg(kernel, numArgs + 3, pads);
        setKernelArg(kernel, numArgs + 4, CC->getGroup());
        setKernelArg(kernel, numArgs + 5, CC->getDilation()[0]);
        setKernelArg(kernel, numArgs + 6, odim);
        setKernelArg(kernel, numArgs + 7, idim);
        setKernelArg(kernel, numArgs + 8, kernelSize);

        if (isQuantized) {
          auto srcTy = CC->getSrc()->getType();
          auto destTy = CC->getDest()->getType();
          auto filterTy = CC->getFilter()->getType();
          auto biasTy = CC->getBias()->getType();
          setKernelArg(kernel, numArgs + 9, destTy->getOffset());
          setKernelArg(kernel, numArgs + 10, destTy->getScale());
          setKernelArg(kernel, numArgs + 11, srcTy->getOffset());
          setKernelArg(kernel, numArgs + 12, srcTy->getScale());
          setKernelArg(kernel, numArgs + 13, filterTy->getOffset());
          setKernelArg(kernel, numArgs + 14, filterTy->getScale());
          setKernelArg(kernel, numArgs + 15, biasTy->getOffset());
          setKernelArg(kernel, numArgs + 16, biasTy->getScale());
        }
      }

      // Use a 3D grid where the first dimension is the depth and the second
      // dimension is the slice index in the batch.
      enqueueKernel(I.getName(), commands, kernel, deviceId,
                    {odim.h, odim.w, odim.c}, kernelLaunches);
      continue;
    }

    if (auto *CG = dyn_cast<ConvolutionGradInst>(&I)) {
      auto *src = CG->getSrc();
      auto *destGrad = CG->getDestGrad();
      auto *srcGrad = CG->getSrcGrad();
      auto *filterGrad = CG->getFilterGrad();
      auto *biasGrad = CG->getBiasGrad();
      cl_kernel kernel = createKernel(kernelName, program);
      setKernelArg(kernel, 0, deviceBuffer);
      auto numArgs = setKernelArgsForBuffers(kernel, I, 1, runtimeBundle_);

      auto destGradDim = ShapeNHWC(destGrad->dims());
      auto srcDim = ShapeNHWC(src->dims());
      auto filterGradDim = ShapeNHWC(filterGrad->dims());
      auto pads = PaddingTLBR(CG->getPads());

      CHECK_EQ(CG->getDilation()[0], CG->getDilation()[1])
          << "Currently not support non-square dilation.";

      ShapeHW kdim(CG->getKernels());
      ShapeHW sdim(CG->getStrides());
      setKernelArg(kernel, numArgs + 1, kdim);
      setKernelArg(kernel, numArgs + 2, sdim);
      setKernelArg(kernel, numArgs + 3, pads);
      setKernelArg(kernel, numArgs + 4, CG->getGroup());
      setKernelArg(kernel, numArgs + 5, CG->getDilation()[0]);
      setKernelArg(kernel, numArgs + 6, srcDim);
      setKernelArg(kernel, numArgs + 7, destGradDim);
      setKernelArg(kernel, numArgs + 8, filterGradDim);
      // Zero memory for the output buffers.
      fillBuffer(deviceBuffer, runtimeBundle_.getValueOffset(srcGrad),
                 srcGrad->size(), 0, srcGrad->getElementType(), clBindings);
      fillBuffer(deviceBuffer, runtimeBundle_.getValueOffset(filterGrad),
                 filterGrad->size(), 0, filterGrad->getElementType(),
                 clBindings);
      fillBuffer(deviceBuffer, runtimeBundle_.getValueOffset(biasGrad),
                 biasGrad->size(), 0, biasGrad->getElementType(), clBindings);

      enqueueKernel(I.getName(), commands, kernel, deviceId,
                    {destGradDim.h, destGradDim.w, destGradDim.c},
                    kernelLaunches);
      continue;
    }

    if (auto *PM = dyn_cast<MaxPoolInst>(&I)) {
      bool isNCHW = PM->getLayout() == NCHW;

      if (isNCHW) {
        kernelName = "ocl" + kernelName;
      }

      cl_kernel kernel = createKernel(kernelName, program);
      setKernelArg(kernel, 0, deviceBuffer);
      auto numArgs = setKernelArgsForBuffers(kernel, I, 1, runtimeBundle_);

      ShapeHW kdim(PM->getKernels());
      ShapeHW sdim(PM->getStrides());
      setKernelArg<cl_uint>(kernel, numArgs + 1, kdim.height);
      setKernelArg<cl_uint>(kernel, numArgs + 2, sdim.height);
      auto pads = PaddingTLBR(PM->getPads());
      setKernelArg(kernel, numArgs + 3, pads);

      std::array<size_t, 3> global;
      if (isNCHW) {
        ShapeNCHW odim(PM->getDest()->getType()->dims());
        ShapeNCHW idim(PM->getSrc()->getType()->dims());

        setKernelArg(kernel, numArgs + 4, odim);
        setKernelArg(kernel, numArgs + 5, idim);
        global = {{odim.h, odim.w, odim.c}};
      } else {
        ShapeNHWC odim(PM->getDest()->getType()->dims());
        ShapeNHWC idim(PM->getSrc()->getType()->dims());
        setKernelArg(kernel, numArgs + 4, odim);
        setKernelArg(kernel, numArgs + 5, idim);
        global = {{odim.h, odim.w, odim.c}};
      }

      enqueueKernel(I.getName(), commands, kernel, deviceId, global,
                    kernelLaunches);
      continue;
    }

    if (auto *PM = dyn_cast<MaxPoolWithArgmaxInst>(&I)) {
      // This is a naive implementation that parallelizes using three dims:
      // the X and the Y in the output filter.
      cl_kernel kernel = createKernel(kernelName, program);
      setKernelArg(kernel, 0, deviceBuffer);
      auto numArgs = setKernelArgsForBuffers(kernel, I, 1, runtimeBundle_);

      auto odim = ShapeNHWC(PM->getDest()->getType()->dims());
      auto idim = ShapeNHWC(PM->getSrc()->getType()->dims());
      auto pads = PaddingTLBR(PM->getPads());
      ShapeHW kdim(PM->getKernels());
      ShapeHW sdim(PM->getStrides());
      setKernelArg<cl_uint>(kernel, numArgs + 1, kdim.height);
      setKernelArg<cl_uint>(kernel, numArgs + 2, sdim.height);
      setKernelArg(kernel, numArgs + 3, pads);
      setKernelArg(kernel, numArgs + 4, odim);
      setKernelArg(kernel, numArgs + 5, idim);

      enqueueKernel(I.getName(), commands, kernel, deviceId,
                    {odim.h, odim.w, odim.c}, kernelLaunches);
      continue;
    }

    if (auto *PMG = dyn_cast<MaxPoolWithArgmaxGradInst>(&I)) {
      cl_kernel kernel = createKernel(kernelName, program);
      setKernelArg(kernel, 0, deviceBuffer);
      auto numArgs = setKernelArgsForBuffers(kernel, I, 1, runtimeBundle_);

      auto destGradDim = ShapeNHWC(PMG->getDestGrad()->dims());
      auto srcGradDim = ShapeNHWC(PMG->getSrcGrad()->dims());
      auto pads = PaddingTLBR(PMG->getPads());
      ShapeHW kdim(PMG->getKernels());
      ShapeHW sdim(PMG->getStrides());
      setKernelArg<cl_uint>(kernel, numArgs + 1, kdim.height);
      setKernelArg<cl_uint>(kernel, numArgs + 2, sdim.height);
      setKernelArg(kernel, numArgs + 3, pads);
      setKernelArg(kernel, numArgs + 4, srcGradDim);
      setKernelArg(kernel, numArgs + 5, destGradDim);

      enqueueKernel(I.getName(), commands, kernel, deviceId, {srcGradDim.n},
                    kernelLaunches);
      continue;
    }

    if (auto *PA = dyn_cast<AvgPoolInst>(&I)) {
      bool isNCHW = PA->getLayout() == NCHW;

      if (isNCHW) {
        kernelName = "ocl" + kernelName;
      }

      cl_kernel kernel = createKernel(kernelName, program);
      setKernelArg(kernel, 0, deviceBuffer);
      auto numArgs = setKernelArgsForBuffers(kernel, I, 1, runtimeBundle_);

      ShapeHW kdim(PA->getKernels());
      ShapeHW sdim(PA->getStrides());
      setKernelArg<cl_uint>(kernel, numArgs + 1, kdim.height);
      setKernelArg<cl_uint>(kernel, numArgs + 2, sdim.height);
      auto pads = PaddingTLBR(PA->getPads());
      setKernelArg(kernel, numArgs + 3, pads);

      std::array<size_t, 3> global;
      if (isNCHW) {
        ShapeNCHW odim(PA->getDest()->getType()->dims());
        ShapeNCHW idim(PA->getSrc()->getType()->dims());

        setKernelArg(kernel, numArgs + 4, odim);
        setKernelArg(kernel, numArgs + 5, idim);
        global = {{odim.h, odim.w, odim.c}};
      } else {
        ShapeNHWC odim(PA->getDest()->getType()->dims());
        ShapeNHWC idim(PA->getSrc()->getType()->dims());
        setKernelArg(kernel, numArgs + 4, odim);
        setKernelArg(kernel, numArgs + 5, idim);
        global = {{odim.h, odim.w, odim.c}};
      }

      if (isNCHW && isQuantized) {
        auto srcTy = PA->getSrc()->getType();
        auto destTy = PA->getDest()->getType();
        auto destScaleParam = quantization::quantizeScaleOffset32To8(
            srcTy->getScale() / destTy->getScale() /
                (PA->getKernels()[0] * PA->getKernels()[0]),
            destTy->getOffset());
        setKernelArg(kernel, numArgs + 6, srcTy->getOffset());
        setKernelArg(kernel, numArgs + 7, destScaleParam);
      }

      enqueueKernel(I.getName(), commands, kernel, deviceId, global,
                    kernelLaunches);
      continue;
    }

    if (auto *TR = dyn_cast<TransposeInst>(&I)) {
      // This is a naive implementation that parallelizes using one dimension,
      // the N (batch size).
      CHECK_LE(TR->getShuffle().size(), 4)
          << "This code supports only 4 and lower dimensional transposes";

      cl_kernel kernel = createKernel(kernelName, program);
      setKernelArg(kernel, 0, deviceBuffer);
      auto numArgs = setKernelArgsForBuffers(kernel, I, 1, runtimeBundle_);

      // Temporary hack to support 3-dim transposes.
      // TODO: support any dimensional transposes.
      std::vector<dim_t> odim_vec = TR->getDest()->getType()->dims();
      std::vector<dim_t> idim_vec = TR->getSrc()->getType()->dims();
      std::vector<unsigned_t> mask = TR->getShuffle();
      while (mask.size() < 4) {
        odim_vec.push_back(1);
        idim_vec.push_back(1);
        mask.push_back(mask.size());
        continue;
      }

      auto odim = ShapeNHWC(llvm::makeArrayRef(odim_vec));
      auto idim = ShapeNHWC(llvm::makeArrayRef(idim_vec));

      setKernelArg(kernel, numArgs + 1, odim);
      setKernelArg(kernel, numArgs + 2, idim);

      ShapeNHWC shuff(mask[0], mask[1], mask[2], mask[3]);
      setKernelArg(kernel, numArgs + 3, shuff);
      enqueueKernel(I.getName(), commands, kernel, deviceId, {idim.n, idim.h},
                    kernelLaunches);
      continue;
    }

    if (auto *C = dyn_cast<CopyInst>(&I)) {
      Value *dest, *src;
      dest = C->getDest();
      src = C->getSrc();
      if (src == dest) {
        continue;
      }
      size_t destOff = runtimeBundle_.getValueOffset(dest);
      size_t srcOff = runtimeBundle_.getValueOffset(src);
      size_t sizeInBytes = dest->getSizeInBytes();
      cl_event event{nullptr};
      cl_int err = clEnqueueCopyBuffer(commands, deviceBuffer, deviceBuffer,
                                       srcOff, destOff, sizeInBytes, 0, nullptr,
                                       kernelProfiling_ ? &event : nullptr);
      if (kernelProfiling_) {
        kernelLaunches.emplace_back(
            KernelLaunch(I.getName().str(), "copy", event));
      }
      CHECK_EQ(err, CL_SUCCESS) << "Error in clEnqueueCopyBuffer.";
      continue;
    }

    if (auto *GI = dyn_cast<GatherInst>(&I)) {
      cl_kernel kernel = createKernel(kernelName, program);
      setKernelArg(kernel, 0, deviceBuffer);
      auto numArgs = setKernelArgsForBuffers(kernel, I, 1, runtimeBundle_);
      unsigned_t axis = GI->getBatchDims();

      auto *data = GI->getData();

      TypeRef dataType = data->getType();
      size_t numIndices = GI->getIndices()->size();

      // The size of the sample in the batch.
      size_t sliceSize = dataType->getSliceSize(axis + 1);
      // The size of the slices that we gather.
      size_t srcSampleSize = dataType->getSliceSize(axis);
      // The size of the slices that we pack.
      size_t destSampleSize = numIndices * sliceSize;
      // The size of each sample in the batch.
      size_t numSamples = dataType->size() / srcSampleSize;

      setKernelArg<cl_uint>(kernel, numArgs + 1, numIndices);
      setKernelArg<cl_uint>(kernel, numArgs + 2, sliceSize);

      // Batch arguments:
      setKernelArg<cl_uint>(kernel, numArgs + 3, numSamples);
      setKernelArg<cl_uint>(kernel, numArgs + 4, destSampleSize);
      setKernelArg<cl_uint>(kernel, numArgs + 5, srcSampleSize);

      enqueueKernel(I.getName(), commands, kernel, deviceId, {numIndices},
                    kernelLaunches);
      continue;
    }

    if (auto *SDI = dyn_cast<ScatterDataInst>(&I)) {
      cl_kernel kernel = createKernel(kernelName, program);
      setKernelArg(kernel, 0, deviceBuffer);
      auto numArgs = setKernelArgsForBuffers(kernel, I, 1, runtimeBundle_);

      auto *data = SDI->getData();
      size_t dataSliceSize = data->size() / data->dims()[0];
      size_t numIndices = SDI->getIndices()->size();
      setKernelArg<cl_uint>(kernel, numArgs + 1, dataSliceSize);

      enqueueKernel(I.getName(), commands, kernel, deviceId, {numIndices},
                    kernelLaunches);
      continue;
    }

    if (auto *SLWS = dyn_cast<SparseLengthsWeightedSumInst>(&I)) {
      cl_kernel kernel = createKernel(kernelName, program);
      // Set the device buffer as the first argument.
      setKernelArg(kernel, 0, deviceBuffer);
      // Set all buffer arguments from the instruction (data, dest, weights,
      // indices, lengths) as subsequent arguments.
      auto numArgs = setKernelArgsForBuffers(kernel, I, 1, runtimeBundle_);

      // Set the size of one slice of data as the last argument.
      auto *data = SLWS->getData();
      size_t dataSliceSize = data->size() / data->dims()[0];
      setKernelArg<cl_uint>(kernel, numArgs + 1, dataSliceSize);

      // Zero the destination buffer so that the kernel can accumulate (+=) into
      // it.
      auto *dest = SLWS->getDest();
      fillBuffer(deviceBuffer, runtimeBundle_.getValueOffset(dest),
                 dest->size(), 0, dest->getElementType(), clBindings);

      // Get the number of segments. The output for each segment will be
      // computed in parallel by setting the global size equal to the number of
      // segments.
      size_t segments = SLWS->getLengths()->size();

      // Enqueue the kernel.
      enqueueKernel(I.getName(), commands, kernel, deviceId, {segments},
                    kernelLaunches);
      continue;
    }

    if (auto *SLWSG = dyn_cast<SparseLengthsWeightedSumGradInst>(&I)) {
      cl_kernel kernel = createKernel(kernelName, program);
      // Set the device buffer as the first argument.
      setKernelArg(kernel, 0, deviceBuffer);
      // Set all buffer arguments from the instruction (dataGrad, destGrad,
      // weights, indices, lengths) as subsequent arguments.
      auto numArgs = setKernelArgsForBuffers(kernel, I, 1, runtimeBundle_);

      // Set the number of segments as the second last argument.
      auto *lengths = SLWSG->getLengths();
      size_t segments = lengths->size();
      setKernelArg<cl_uint>(kernel, numArgs + 1, segments);

      // Set the size of one slice of destGrad as the last argument.
      auto *destGrad = SLWSG->getDestGrad();
      size_t destGradSliceSize = destGrad->size() / destGrad->dims()[0];
      setKernelArg<cl_uint>(kernel, numArgs + 2, destGradSliceSize);

      // Zero the data gradient buffer so that the kernel can accumulate (+=)
      // into it.
      auto *dataGrad = SLWSG->getDataGrad();
      fillBuffer(deviceBuffer, runtimeBundle_.getValueOffset(dataGrad),
                 dataGrad->size(), 0, dataGrad->getElementType(), clBindings);

      // Enqueue the kernel. Set the global size to 1 so that all segments are
      // processed sequentially to avoid two kernel instances accumulating into
      // the same data gradient slice. This could potentially be relaxed by
      // using an atomic add in the kernel.
      enqueueKernel(I.getName(), commands, kernel, deviceId, {1},
                    kernelLaunches);
      continue;
    }

    if (auto *DP = dyn_cast<DebugPrintInst>(&I)) {
      clFinish(commands);
      auto *V = DP->getSrc();
      // Allocate a temporary tensor to hold the value.
      Tensor T(V->getType());
      // Load the current value of the variable into host memory.
      copyValueFromDevice(V, clBindings, T.getUnsafePtr());
      clFinish(commands);
      llvm::outs() << I.getName() << ": ";
      // Dump the content of a value.
      V->dump();
      llvm::outs() << "\n";
      dumpImpl(&T);
      llvm::outs() << "\n";
      llvm::outs().flush();
      continue;
    }

    if (auto *TE = dyn_cast<TraceEventInst>(&I)) {
      cl_kernel kernel = createKernel("checkpoint", program);
      setKernelArg(kernel, 0, deviceBuffer);

      llvm::SmallVector<size_t, 1> global = {1};
      llvm::SmallVector<size_t, 4> local(global.size(), 0);
      getMaxLocalWorkgroupSize(kernel, deviceId, global, local);

      cl_event event;
      cl_int err =
          clEnqueueNDRangeKernel(commands, kernel, global.size(), nullptr,
                                 &global[0], &local[0], 0, nullptr, &event);
      CHECK_EQ(err, CL_SUCCESS) << "Error in clEnqueueNDRangeKernel.";
      kernelLaunches.push_back(
          KernelLaunch(kernel, TE->getName().str(), "checkpoint", event));
      continue;
    }

    // For TopKInst, we perform the computation on the host side, as sorting on
    // GPU is complex and we may not get too much benefit from it. We copy the
    // tensor from GPU memory to host memory, perform the computation, and then
    // copy the results back to GPU memory.
    if (auto *TK = dyn_cast<TopKInst>(&I)) {
      clFinish(commands);
      auto *destDev = TK->getValues();
      auto *indDev = TK->getIndices();
      auto *srcDev = TK->getInput();
      Tensor destT(destDev->getType());
      Tensor indT(indDev->getType());
      Tensor srcT(srcDev->getType());
      size_t k = TK->getK();

      copyValueFromDevice(srcDev, clBindings, srcT.getUnsafePtr());
      clFinish(commands);

      if (srcDev->getType()->isQuantizedType() ||
          destDev->getType()->isQuantizedType()) {
        topK<int8_t>(destT, indT, srcT, k);
      } else {
        topK<float>(destT, indT, srcT, k);
      }
      copyValueToDevice(destDev, clBindings, destT.getUnsafePtr());
      copyValueToDevice(indDev, clBindings, indT.getUnsafePtr());
      clFinish(commands);
      continue;
    }

    LOG(FATAL) << "Compilation failed, cannot select: " << I.getKindName();
  }

  enqueueEvent.end();

  clFinish(commands);

  return Error::success();
}

uint64_t OpenCLFunction::copyValueToDevice(
    const Value *v, runtime::OpenCLDeviceBindings *devBindings, void *buf) {
  uint64_t copiedBytes = 0;
  auto symbolInfo = runtimeBundle_.getSymbolInfo(v);
  size_t sizeInBytes = v->getType()->getSizeInBytes();
  // Issue a non-blocking command to copy the buffer to the device.
  if (sizeInBytes) {
    size_t valueOffset = symbolInfo.offset;
    cl_event event{nullptr};
    cl_int err = clEnqueueWriteBuffer(
        devBindings->commandQueue, devBindings->deviceBuffer,
        /* blocking_write */ CL_FALSE, valueOffset, sizeInBytes, buf,
        /* num_events_in_wait_list */ 0,
        /* event_list */ nullptr,
        /* event */ kernelProfiling_ ? &event : nullptr);
    CHECK_EQ(err, CL_SUCCESS) << "Unable to copy data to the device";
    if (kernelProfiling_) {
      devBindings->kernelLaunches.emplace_back(
          KernelLaunch("copyValueToDevice", "copy", event));
    }
    copiedBytes += sizeInBytes;
  }
  return copiedBytes;
}

uint64_t OpenCLFunction::copyValueFromDevice(
    const Value *v, runtime::OpenCLDeviceBindings *devBindings, void *buf) {
  uint64_t copiedBytes = 0;
  auto symbolInfo = runtimeBundle_.getSymbolInfo(v);
  size_t sizeInBytes = v->getType()->getSizeInBytes();
  // Issue a non-blocking command to copy the buffer from the device.
  if (sizeInBytes) {
    size_t valueOffset = symbolInfo.offset;
    cl_event event{nullptr};
    cl_int err = clEnqueueReadBuffer(
        devBindings->commandQueue, devBindings->deviceBuffer,
        /* blocking_read */ CL_FALSE, valueOffset, sizeInBytes, buf,
        /* num_events_in_wait_list */ 0,
        /* event_list */ nullptr,
        /* event */ kernelProfiling_ ? &event : nullptr);
    CHECK_EQ(err, CL_SUCCESS) << "Unable to copy from the device";
    DEBUG_GLOW(llvm::dbgs()
               << "Copied the value from device: " << v->getName() << "\n");
    if (kernelProfiling_) {
      devBindings->kernelLaunches.emplace_back(
          KernelLaunch("copyValueFromDevice", "copyValueFromDevice", event));
    }
    copiedBytes += sizeInBytes;
  }
  return copiedBytes;
}

cl_mem OpenCLFunction::allocDeviceBuffer(uint64_t size, cl_context clContext) {
  const uint64_t alignment = 128;
  // Always allocate buffers properly aligned to hold values of any type.
  size = alignedSize(size, alignment);
  auto buf =
      clCreateBuffer(clContext, CL_MEM_READ_WRITE, size, nullptr, nullptr);
  CHECK(buf) << "Allocation failed!";
  return buf;
}

void OpenCLFunction::freeDeviceBuffer(cl_mem buf) { clReleaseMemObject(buf); }

void OpenCLFunction::collectConstants(const Module *module) {
  runtimeBundle_.collectConstants(module);
}

std::unique_ptr<CompiledFunction>
OCLBackend::compileIR(std::unique_ptr<IRFunction> IR) const {
  auto *module = IR->getParent();
  TraceInfo traceInfo;

  MemoryAllocator allocator("GPU", 0xFFFFFFFF);
  runtime::RuntimeBundle bundle =
      runtime::RuntimeBundle::create(*IR, allocator);
  std::unique_ptr<CompiledFunction> function =
      glow::make_unique<OpenCLFunction>(std::move(IR), std::move(bundle),
                                        std::move(traceInfo));
  auto OCLFunction = static_cast<OpenCLFunction *>(function.get());
  OCLFunction->collectConstants(module);
  return function;
}

Expected<std::unique_ptr<CompiledFunction>>
OCLBackend::compile(Function *F, const BackendOptions &opts) const {
  TraceInfo traceInfo = buildManualTraceInfo(F);

  auto IR = generateAndOptimizeIR(F, *this, shouldShareBuffers());

  if (opts.autoInstrument) {
    autoInstrument(traceInfo, IR.get());
  }

  MemoryAllocator allocator("GPU", 0xFFFFFFFF);
  runtime::RuntimeBundle bundle =
      runtime::RuntimeBundle::create(*IR, allocator);

  if (opts.collectConstants) {
    bundle.collectConstants(F->getParent());
  }

  std::unique_ptr<CompiledFunction> compiledFunc =
      glow::make_unique<OpenCLFunction>(std::move(IR), std::move(bundle),
                                        std::move(traceInfo));

  return Expected<std::unique_ptr<CompiledFunction>>(std::move(compiledFunc));
}

bool OCLBackend::isOpSupported(const NodeInfo &NI) const {
  switch (NI.getKind()) {
  case Kinded::Kind::SplatNodeKind:
  case Kinded::Kind::TouchNodeKind:
  case Kinded::Kind::TransposeNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
        {ElemKind::FloatTy, ElemKind::Int8QTy, ElemKind::Int64ITy});

  case Kinded::Kind::PowNodeKind:
  case Kinded::Kind::LocalResponseNormalizationNodeKind:
  case Kinded::Kind::LocalResponseNormalizationGradNodeKind:
  case Kinded::Kind::BatchedReduceAddNodeKind:
  case Kinded::Kind::TanhNodeKind:
  case Kinded::Kind::SigmoidNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind({ElemKind::FloatTy});

  case Kinded::Kind::AddNodeKind:
  case Kinded::Kind::SubNodeKind:
  case Kinded::Kind::MulNodeKind:
  case Kinded::Kind::DivNodeKind:
  case Kinded::Kind::MaxNodeKind:
  case Kinded::Kind::MinNodeKind:
  case Kinded::Kind::MatMulNodeKind:
  case Kinded::Kind::ConcatNodeKind:
  case Kinded::Kind::SliceNodeKind:
  case Kinded::Kind::InsertTensorNodeKind:
  case Kinded::Kind::AvgPoolNodeKind:
  case Kinded::Kind::ReluNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
        {ElemKind::FloatTy, ElemKind::Int8QTy});

  case Kinded::Kind::MaxPoolNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
               {ElemKind::FloatTy, ElemKind::Int8QTy}, {},
               {MaxPoolNode::ArgmaxIdx}) &&
           (NI.getOutElemTy(MaxPoolNode::ArgmaxIdx) == ElemKind::Int64ITy);

  case Kinded::Kind::ConvolutionNodeKind:
    if (!NI.getInTy(ConvolutionNode::InputIdx)->isQuantizedType()) {
      return NI.allInputsAndOutputsHaveSameElemKind({ElemKind::FloatTy});
    }
    return NI.allInputsAndOutputsHaveSameElemKind({ElemKind::Int8QTy},
                                                  {ConvolutionNode::BiasIdx}) &&
           (NI.getInElemTy(ConvolutionNode::BiasIdx) == ElemKind::Int32QTy);

  case Kinded::Kind::TopKNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
               {ElemKind::FloatTy, ElemKind::Int8QTy}, {},
               {TopKNode::IndicesIdx}) &&
           (NI.getOutElemTy(TopKNode::IndicesIdx) == ElemKind::Int64ITy);

  case Kinded::Kind::BatchedAddNodeKind:
    if (!NI.getInTy(BatchedAddNode::BatchIdx)->isQuantizedType()) {
      return NI.allInputsAndOutputsHaveSameElemKind({ElemKind::FloatTy});
    }
    return NI.allInputsAndOutputsHaveSameElemKind({ElemKind::Int8QTy},
                                                  {BatchedAddNode::SliceIdx}) &&
           ((NI.getInElemTy(BatchedAddNode::SliceIdx) == ElemKind::Int8QTy) ||
            (NI.getInElemTy(BatchedAddNode::SliceIdx) == ElemKind::Int32QTy));

  case Kinded::Kind::GatherNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind({ElemKind::FloatTy},
                                                  {GatherNode::IndicesIdx}) &&
           (NI.getInElemTy(GatherNode::IndicesIdx) == ElemKind::Int64ITy);

  case Kinded::Kind::ScatterDataNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
               {ElemKind::FloatTy}, {ScatterDataNode::IndicesIdx}) &&
           (NI.getInElemTy(ScatterDataNode::IndicesIdx) == ElemKind::Int64ITy);

  case Kinded::Kind::SparseLengthsWeightedSumNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
               {ElemKind::FloatTy},
               {SparseLengthsWeightedSumNode::IndicesIdx,
                SparseLengthsWeightedSumNode::LengthsIdx}) &&
           (NI.getInElemTy(SparseLengthsWeightedSumNode::IndicesIdx) ==
            ElemKind::Int64ITy) &&
           (NI.getInElemTy(SparseLengthsWeightedSumNode::LengthsIdx) ==
            ElemKind::Int32ITy);

  case Kinded::Kind::MaxPoolGradNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
               {ElemKind::FloatTy},
               {MaxPoolGradNode::OriginalOutputForArgmaxIdx,
                MaxPoolGradNode::GradOfOriginalOutputNamedArgmaxIdx}) &&
           (NI.getInElemTy(MaxPoolGradNode::OriginalOutputForArgmaxIdx) ==
            ElemKind::Int64ITy) &&
           (NI.getInElemTy(
                MaxPoolGradNode::GradOfOriginalOutputNamedArgmaxIdx) ==
            ElemKind::Int64ITy);

  case Kinded::Kind::SparseLengthsWeightedSumGradNodeKind:
    // GradOfInputNamedIndicesIdx and GradOfInputNamedLengthsIdx do not need to
    // be checked because they are not used.
    return NI.allInputsAndOutputsHaveSameElemKind(
               {ElemKind::FloatTy},
               {SparseLengthsWeightedSumGradNode::IndicesIdx,
                SparseLengthsWeightedSumGradNode::LengthsIdx},
               {SparseLengthsWeightedSumGradNode::GradOfInputNamedIndicesIdx,
                SparseLengthsWeightedSumGradNode::
                    GradOfInputNamedLengthsIdx}) &&
           (NI.getInElemTy(SparseLengthsWeightedSumGradNode::IndicesIdx) ==
            ElemKind::Int64ITy) &&
           (NI.getInElemTy(SparseLengthsWeightedSumGradNode::LengthsIdx) ==
            ElemKind::Int32ITy);

  case Kinded::Kind::QuantizeNodeKind:
    return (NI.getInElemTy(QuantizeNode::InputIdx) == ElemKind::FloatTy) &&
           ((NI.getOutElemTy(QuantizeNode::ResultIdx) == ElemKind::Int8QTy) ||
            (NI.getOutElemTy(QuantizeNode::ResultIdx) == ElemKind::Int32QTy));

  case Kinded::Kind::DequantizeNodeKind:
    return (NI.getInElemTy(DequantizeNode::InputIdx) == ElemKind::Int8QTy) &&
           (NI.getOutElemTy(DequantizeNode::ResultIdx) == ElemKind::FloatTy);

  case Kinded::Kind::RescaleQuantizedNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind({ElemKind::Int8QTy});

  case Kinded::Kind::SelectNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind({ElemKind::FloatTy},
                                                  {SelectNode::CondIdx}) &&
           (NI.getInElemTy(SelectNode::CondIdx) == ElemKind::BoolTy);

  case Kinded::Kind::CmpLTENodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind({ElemKind::FloatTy}, {},
                                                  {CmpLTENode::ResultIdx}) &&
           (NI.getOutElemTy(CmpLTENode::ResultIdx) == ElemKind::BoolTy);

  // We just clip 64 to 32 SelectedIdx silently with the SoftMax
  // SelectedIdx in case dim_t is 32b.
  case Kinded::Kind::SoftMaxNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind({ElemKind::FloatTy},
                                                  {SoftMaxNode::SelectedIdx}) &&
           (NI.getInElemTy(SoftMaxNode::SelectedIdx) == ElemKind::Int32ITy ||
            NI.getInElemTy(SoftMaxNode::SelectedIdx) == ElemKind::Int64ITy);

  case Kinded::Kind::ConvolutionGradNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
        {ElemKind::FloatTy}, {},
        {ConvolutionGradNode::GradOfInputNamedInputIdx});

  case Kinded::Kind::SoftMaxGradNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
               {ElemKind::FloatTy}, {SoftMaxGradNode::SelectedIdx},
               {SoftMaxGradNode::GradOfInputNamedSelectedIdx}) &&
           (NI.getInElemTy(SoftMaxGradNode::SelectedIdx) == ElemKind::Int64ITy);

  case Kinded::Kind::SaveNodeKind:
  case Kinded::Kind::ReshapeNodeKind:
  case Kinded::Kind::OCLBatchedReduceAddNodeKind:
  case Kinded::Kind::TraceEventNodeKind:
    // These work regardless of the underlying type.
    return true;

  default:
    return false;
  }
}

/// If \p I got square shaped kernels and strides \returns true.
template <class T> static bool checkSquare(const T &I) {
  ShapeHW kdim(I.getKernels());
  ShapeHW sdim(I.getStrides());
  if (!kdim.isSquare()) {
    report("Only square kernel is supported");
    return false;
  }
  if (!sdim.isSquare()) {
    report("Only square stride is supported");
    return false;
  }
  return true;
}

bool OCLBackend::verify(const Function &F, bool verbose) const {
  if (!F.verify(this)) {
    return false;
  }
  if (!checkAllNodesSupported(F, verbose)) {
    return false;
  }
  for (const Node &N : F.getNodes()) {
    if (!(N.getKind() == Kinded::Kind::ConvolutionNodeKind &&
          llvm::cast<ConvolutionNode>(&N)->getFusedActivation() ==
              FusedActivation::RELU) &&
        !checkNoFusionForNode(N)) {
      return false;
    }
    switch (N.getKind()) {
    case Kinded::Kind::ScatterDataNodeKind: {
      auto *SD = llvm::cast<ScatterDataNode>(&N);
      if (SD->getCumulative()) {
        report("Cumulative assign not supported!");
        return false;
      }
      if (SD->getIndices().dims()[1] != 1) {
        report("Only one-dimensional indices are supported");
        return false;
      }
      continue;
    }
    case Kinded::Kind::OCLBatchedReduceAddNodeKind: {
      auto *BRA = llvm::cast<OCLBatchedReduceAddNode>(&N);
      auto destDims = BRA->getResult().getType()->dims();
      if (destDims.size() > 3) {
        report("OpenCL BatchedReduceAdd supports max 3 output dimensions");
        return false;
      }
      continue;
    }
    case Kinded::Kind::MaxPoolNodeKind: {
      auto *MP = llvm::cast<MaxPoolNode>(&N);
      if (!checkSquare(*MP)) {
        return false;
      }
      continue;
    }
    case Kinded::Kind::MaxPoolGradNodeKind: {
      auto *MPG = llvm::cast<MaxPoolGradNode>(&N);
      if (!checkSquare(*MPG)) {
        return false;
      }
      continue;
    }
    case Kinded::Kind::AvgPoolNodeKind: {
      auto *AP = llvm::cast<AvgPoolNode>(&N);
      if (!checkSquare(*AP)) {
        return false;
      }
      continue;
    }
    default:
      continue;
    }
  }
  return true;
}

bool OCLBackend::verify(const IRFunction &IR) const {
  for (const auto &I : IR.getInstrs()) {
    // Only support convolution+relu fusions for now.
    if (!(I.getKind() == Kinded::Kind::ConvolutionInstKind &&
          llvm::cast<ConvolutionInst>(&I)->getFusedActivation() ==
              FusedActivation::RELU) &&
        !checkNoFusionForInstr(I)) {
      return false;
    }
    switch (I.getKind()) {
    case Kinded::Kind::ScatterDataInstKind: {
      auto *SD = llvm::cast<ScatterDataInst>(&I);
      if (SD->getCumulative()) {
        report("Cumulative assign not supported!");
        return false;
      }
      if (SD->getIndices()->dims()[1] != 1) {
        report("Only one-dimensional indices are supported");
        return false;
      }
      continue;
    }
    case Kinded::Kind::OCLBatchedReduceAddInstKind: {
      auto *BRA = llvm::cast<OCLBatchedReduceAddInst>(&I);
      auto destDims = BRA->getDest()->getType()->dims();
      if (destDims.size() > 3) {
        report("OpenCL BatchedReduceAdd supports max 3 output dimensions");
        return false;
      }
      continue;
    }
    case Kinded::Kind::ConvolutionGradInstKind: {
      auto *CG = llvm::cast<ConvolutionGradInst>(&I);
      auto *src = CG->getSrc();
      auto *filter = CG->getFilter();
      auto *srcGrad = CG->getSrcGrad();
      auto *filterGrad = CG->getFilterGrad();
      if (filter->dims() != filterGrad->dims() ||
          src->dims() != srcGrad->dims()) {
        report("Dims should be the same");
        return false;
      }
      continue;
    }
    case Kinded::Kind::MaxPoolInstKind: {
      auto *MP = llvm::cast<MaxPoolInst>(&I);
      if (!checkSquare(*MP)) {
        return false;
      }
      continue;
    }
    case Kinded::Kind::MaxPoolWithArgmaxInstKind: {
      auto *MPWA = llvm::cast<MaxPoolWithArgmaxInst>(&I);
      if (!checkSquare(*MPWA)) {
        return false;
      }
      continue;
    }
    case Kinded::Kind::MaxPoolWithArgmaxGradInstKind: {
      auto *MPWAG = llvm::cast<MaxPoolWithArgmaxGradInst>(&I);
      if (!checkSquare(*MPWAG)) {
        return false;
      }
      auto destGradDim = ShapeNHWC(MPWAG->getDestGrad()->dims());
      auto srcGradDim = ShapeNHWC(MPWAG->getSrcGrad()->dims());
      if (srcGradDim.n != destGradDim.n) {
        report("batch size is wrong");
        return false;
      }
      if (srcGradDim.c != destGradDim.c) {
        report("depth size is wrong");
        return false;
      }
      continue;
    }
    case Kinded::Kind::AvgPoolInstKind: {
      auto *AP = llvm::cast<AvgPoolInst>(&I);
      if (!checkSquare(*AP)) {
        return false;
      }
      continue;
    }
    case Kinded::Kind::GatherInstKind: {
      auto *G = llvm::cast<GatherInst>(&I);
      auto *data = G->getData();
      if (data->getElementType() != ElemKind::FloatTy) {
        report("Gather: At the moment only floats are supported");
        return false;
      }
      continue;
    }
    default:
      continue;
    }
  }
  return true;
}

runtime::DeviceManager *
OCLBackend::createDeviceManager(const runtime::DeviceConfig &deviceConfig) {
  return createOCLDeviceManager(deviceConfig);
}

TensorLayoutCommon &OCLBackend::getTensorLayoutRequirements() const {
  return OpenCLTensorLayout::getInstance();
}

TraceInfo OCLBackend::buildManualTraceInfo(Function *F) const {
  TraceInfo info(false, getTraceEventDataSize());

  const auto &nodes = F->getNodes();
  for (const auto &node : nodes) {
    if (const TraceEventNode *TEN = llvm::dyn_cast<TraceEventNode>(&node)) {

      Placeholder *backing =
          llvm::dyn_cast<Placeholder>(TEN->getData().getNode());
      char type = TraceEvent::InstantType;
      if (!TEN->getEventType().empty()) {
        type = TEN->getEventType()[0];
      }
      info.add(backing, TEN->getIndex(), TEN->getEventName(), type,
               TEN->getName().str());
      info.enabled = true;
    }
  }

  return info;
}

void OCLBackend::autoInstrument(TraceInfo &traceInfo, IRFunction *IR) const {
  traceInfo.enabled = true;
  traceInfo.autoInstrumented = true;

  // On OpenCL we don't insert instructions into the IR, and we don't need
  // entries in TraceInfo to interpret them.
}
