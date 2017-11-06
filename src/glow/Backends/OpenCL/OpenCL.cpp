// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "OpenCL.h"

#include "glow/Graph/Graph.h"
#include "glow/Graph/Nodes.h"
#include "glow/IR/Instrs.h"
#include "glow/Optimizer/Optimizer.h"

#include "llvm/Support/Casting.h"

using namespace glow;

using llvm::dyn_cast;
using llvm::isa;

// This defines the string "SHADER_CODE".
#include "kernels.cl"

#if WITH_OPENCL

Backend *glow::createOCLBackend(Module *M) { return new OCLBackend(M); }

using Kind = Kinded::Kind;
using kernelSrcEnum = struct {
  Kind kind;
  const char *funcName;
};

static void dumpCompileLog(cl_device_id dev, cl_program prog) {
#ifndef NDEBUG
  // Determine the size of the log.
  size_t log_size;
  clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);

  // Allocate memory for the log.
  char *log = (char *)malloc(log_size);

  // Get the log.
  clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, log_size, log,
                        nullptr);

  // Print the log.
  std::cout << log << "\n";
  free(log);
#endif
}

OCLBackend::OCLBackend(Module *M) : M_(M) {
  cl_int err =
      clGetDeviceIDs(nullptr, CL_DEVICE_TYPE_DEFAULT, 1, &deviceId_, nullptr);
  GLOW_ASSERT(err == CL_SUCCESS && "clGetDeviceIDs Failed.");

  context_ = clCreateContext(0, 1, &deviceId_, nullptr, nullptr, &err);
  GLOW_ASSERT(context_ && "clCreateContext Failed.");

  commands_ = clCreateCommandQueue(context_, deviceId_, 0, &err);
  GLOW_ASSERT(commands_ && "clCreateCommandQueue Failed.");

  err = CL_SUCCESS;
  program_ =
      clCreateProgramWithSource(context_, 1, &SHADER_CODE, nullptr, &err);
  GLOW_ASSERT(program_ && "clCreateProgramWithSource Failed.");
  err = clBuildProgram(program_, 0, nullptr, nullptr, nullptr, nullptr);
  if (err) {
    dumpCompileLog(deviceId_, program_);
  }
  GLOW_ASSERT(err == CL_SUCCESS && "clBuildProgram Failed.");
}

OCLBackend::~OCLBackend() {
  clReleaseProgram(program_);
  clReleaseCommandQueue(commands_);
  clReleaseContext(context_);
  clear();
}

cl_kernel createKernel(cl_program program, const std::string &name) {
  cl_int err = CL_SUCCESS;
  cl_kernel kernel = clCreateKernel(program, name.c_str(), &err);
  GLOW_ASSERT((kernel && err == CL_SUCCESS) && "clCreateKernel Failed.");
  return kernel;
}

template <class T>
void setKernelArg(cl_kernel kernel, unsigned argIdx, T value) {
  cl_int err = clSetKernelArg(kernel, argIdx, sizeof(T), &value);
  GLOW_ASSERT(err == CL_SUCCESS && "Unable to set parameter");
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
  // less than WSG. In here we find the highest L that divides the globa
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

void enqueueKernel(cl_command_queue commands, cl_kernel kernel,
                   cl_device_id device, llvm::ArrayRef<size_t> global) {
  std::vector<size_t> local(global.size(), 0);
  getMaxLocalWorkgroupSize(kernel, device, global, local);

  auto err = clEnqueueNDRangeKernel(commands, kernel, global.size(), nullptr,
                                    &global[0], &local[0], 0, nullptr, nullptr);
  GLOW_ASSERT(err == CL_SUCCESS && "Error in clEnqueueNDRangeKernel.");
}

void OCLBackend::doForwardPass(bool isTrain) {
  copyWeightsToDevice();
  std::vector<cl_kernel> kernels;

  for (auto &I : M_->getInstrs()) {
    // The kernels are named after the name of the instruction, plus the "K"
    // suffix to prevent name colissions for functions like 'tanh' that are also
    // a part of the OpenCL runtime.
    std::string kernelName = std::string(I->getKindName()) + "K";

    if (auto *A = llvm::dyn_cast<AllocActivationInst>(I)) {
      auto numBytes = I->getType()->getSizeInBytes();
      auto buff = clCreateBuffer(context_, CL_MEM_READ_WRITE, numBytes, nullptr,
                                 nullptr);
      assert(!tensors_.count(A) && "Allocation already made!");
      tensors_[A] = buff;
      continue;
    }

    if (auto *D = llvm::dyn_cast<DeallocActivationInst>(I)) {
      auto *A = D->getAlloc();
      assert(tensors_.count(A) && "Invalid deallocation!");
      clReleaseMemObject(tensors_[A]);
      continue;
    }

    // Element-wise operations:
    if (isa<ReluInst>(I) || isa<SigmoidInst>(I) || isa<TanhInst>(I) ||
        isa<RegressionInst>(I) || isa<ReluInst>(I) || isa<ElementAddInst>(I) ||
        isa<ElementMulInst>(I)) {

      cl_kernel kernel = createKernel(program_, kernelName);

      for (unsigned arg = 0, e = I->getNumOperands(); arg < e; arg++) {
        setKernelArg(kernel, arg, tensors_[I->getOperand(arg).first]);
      }

      // Figure out how many element-wise elements are there to process:
      size_t global = I->getOperand(0).first->getType()->size();

      enqueueKernel(commands_, kernel, deviceId_, {global});
      kernels.push_back(kernel);
      continue;
    }

    if (auto *SM = dyn_cast<SoftMaxInst>(I)) {
      // Implement Softmax by parallelizing the batsh dimension. Each sample in
      // the batch is processed by a different parallel 'thread'.
      cl_kernel kernel = createKernel(program_, kernelName);

      unsigned numArgs = I->getNumOperands();
      for (unsigned arg = 0; arg < numArgs; arg++) {
        setKernelArg(kernel, arg, tensors_[I->getOperand(arg).first]);
      }

      // This is the number of elements for each slice. There are N slices in
      // our batch.
      auto inputDims = SM->getSrc()->getType()->dims();
      size_t numSlices = inputDims[0];

      // Pass the slice size (size of each sample in the batch) as a parameter.
      setKernelArg(kernel, numArgs, flattenCdr(inputDims).second);

      enqueueKernel(commands_, kernel, deviceId_, {numSlices});
      kernels.push_back(kernel);
      continue;
    }

    if (auto *FC = dyn_cast<FullyConnectedInst>(I)) {
      // This is a naive implementation of sgemm that's based on this algorithm:
      // https://cnugteren.github.io/tutorial/pages/page3.html
      cl_kernel kernel = createKernel(program_, kernelName);

      unsigned numArgs = I->getNumOperands();
      for (unsigned arg = 0; arg < numArgs; arg++) {
        setKernelArg(kernel, arg, tensors_[I->getOperand(arg).first]);
      }

      // This is the number of elements for each slice. There are N slices in
      // our batch.
      auto inputDims = FC->getSrc()->getType()->dims();
      // This is the batch size (number of slices/samples in the batch).
      size_t numSlices = inputDims[0];
      size_t depth = FC->getDepth();

      // Pass the slice size (size of each sample in the batch) as a parameter.
      setKernelArg(kernel, numArgs, flattenCdr(inputDims).second);
      setKernelArg(kernel, numArgs + 1, FC->getDepth());

      // Use a 2D grid where the first dimension is the depth and the second
      // dimension is the slice index in the batch.
      enqueueKernel(commands_, kernel, deviceId_, {depth, numSlices});
      kernels.push_back(kernel);
      continue;
    }

    if (auto *CC = dyn_cast<ConvolutionInst>(I)) {
      // This is a naive implementation that parallelizes using three dims:
      // the X and the Y in the output filter.
      cl_kernel kernel = createKernel(program_, kernelName);

      unsigned numArgs = I->getNumOperands();
      for (unsigned arg = 0; arg < numArgs; arg++) {
        setKernelArg(kernel, arg, tensors_[I->getOperand(arg).first]);
      }

      auto odim = ShapeNHWC(CC->getDest()->getType()->dims());
      auto idim = ShapeNHWC(CC->getSrc()->getType()->dims());

      setKernelArg<size_t>(kernel, 4, CC->getKernel());
      setKernelArg(kernel, 5, CC->getPad());
      setKernelArg(kernel, 6, CC->getStride());
      setKernelArg(kernel, 7, odim);
      setKernelArg(kernel, 8, idim);
      setKernelArg(kernel, 9, ShapeNHWC(CC->getFilter()->getType()->dims()));

      auto depth = CC->getDepth();

      // Use a 3D grid where the first dimension is the depth and the second
      // dimension is the slice index in the batch.
      enqueueKernel(commands_, kernel, deviceId_, {odim.h, odim.w, depth});
      kernels.push_back(kernel);
      continue;
    }

    if (auto *PM = dyn_cast<PoolMaxInst>(I)) {
      // This is a naive implementation that parallelizes using three dims:
      // the X and the Y in the output filter.
      cl_kernel kernel = createKernel(program_, kernelName);

      unsigned numArgs = I->getNumOperands();
      for (unsigned arg = 0; arg < numArgs; arg++) {
        setKernelArg(kernel, arg, tensors_[I->getOperand(arg).first]);
      }

      auto odim = ShapeNHWC(PM->getDest()->getType()->dims());
      auto idim = ShapeNHWC(PM->getSrc()->getType()->dims());

      setKernelArg<size_t>(kernel, 3, PM->getKernel());
      setKernelArg(kernel, 4, PM->getPad());
      setKernelArg(kernel, 5, PM->getStride());
      setKernelArg(kernel, 6, odim);
      setKernelArg(kernel, 7, idim);

      enqueueKernel(commands_, kernel, deviceId_, {odim.h, odim.w, odim.c});
      kernels.push_back(kernel);
      continue;
    }

    if (auto *PA = dyn_cast<PoolAvgInst>(I)) {
      // This is a naive implementation that parallelizes using three dims:
      // the X and the Y in the output filter.
      cl_kernel kernel = createKernel(program_, kernelName);

      unsigned numArgs = I->getNumOperands();
      for (unsigned arg = 0; arg < numArgs; arg++) {
        setKernelArg(kernel, arg, tensors_[I->getOperand(arg).first]);
      }

      auto odim = ShapeNHWC(PA->getDest()->getType()->dims());
      auto idim = ShapeNHWC(PA->getSrc()->getType()->dims());

      setKernelArg<size_t>(kernel, 2, PA->getKernel());
      setKernelArg(kernel, 3, PA->getPad());
      setKernelArg(kernel, 4, PA->getStride());
      setKernelArg(kernel, 5, odim);
      setKernelArg(kernel, 6, idim);

      enqueueKernel(commands_, kernel, deviceId_, {odim.h, odim.w, odim.c});
      kernels.push_back(kernel);
      continue;
    }

    if (auto *TR = dyn_cast<TransposeInst>(I)) {
      // This is a naive implementation that parallelizes using one dimension,
      // the N (batch size).
      GLOW_ASSERT(TR->getShuffle().size() == 4 &&
                  "This code supports only 4-dim transposes");

      cl_kernel kernel = createKernel(program_, kernelName);

      unsigned numArgs = I->getNumOperands();
      for (unsigned arg = 0; arg < numArgs; arg++) {
        setKernelArg(kernel, arg, tensors_[I->getOperand(arg).first]);
      }

      auto odim = ShapeNHWC(TR->getDest()->getType()->dims());
      auto idim = ShapeNHWC(TR->getSrc()->getType()->dims());

      setKernelArg(kernel, 2, odim);
      setKernelArg(kernel, 3, idim);

      auto mask = TR->getShuffle();
      ShapeNHWC shuff(mask[0], mask[1], mask[2], mask[3]);
      setKernelArg(kernel, 4, shuff);

      enqueueKernel(commands_, kernel, deviceId_, {idim.n});
      kernels.push_back(kernel);
      continue;
    }

    if (isa<CopyInst>(I) || isa<ReshapeInst>(I)) {
      auto *dest = I->getOperand(0).first;
      auto *src = I->getOperand(1).first;
      if (src == dest) {
        continue;
      }
      auto *destPtr = tensors_[dest];
      auto *srcPtr = tensors_[src];
      size_t sizeInBytes = dest->getType()->getSizeInBytes();
      cl_int err = clEnqueueCopyBuffer(commands_, srcPtr, destPtr, 0, 0,
                                       sizeInBytes, 0, 0, 0);
      GLOW_ASSERT(err == CL_SUCCESS && "Error in clEnqueueCopyBuffer.");
      continue;
    }
    std::cout << "unknown node " << I->getKindName() << "\n";
    // assert(false && "Unexpected node");
  }

  clFinish(commands_);

  for (auto &k : kernels) {
    clReleaseKernel(k);
  }

  copyWeightsFromDevice();
}

void OCLBackend::copyWeightsToDevice() {
  for (auto it : tensors_) {
    assert(externalTensors_.count(it.first) && "Unknown weight!");
    Tensor *T = externalTensors_[it.first];
    size_t sizeInBytes = T->getType().getSizeInBytes();
    // Issue a non-blocking command to copy the buffer to the device.
    cl_int err =
        clEnqueueWriteBuffer(commands_, it.second, CL_FALSE, 0, sizeInBytes,
                             T->getUnsafePtr(), 0, nullptr, nullptr);
    GLOW_ASSERT(err == CL_SUCCESS && "Unable to copy data to the device");
  }
  // Do it!
  clFinish(commands_);
}

void OCLBackend::copyWeightsFromDevice() {
  clFinish(commands_);

  for (auto it : tensors_) {
    if (!externalTensors_.count(it.first))
      continue;

    Tensor *T = externalTensors_[it.first];
    size_t sizeInBytes = T->getType().getSizeInBytes();

    // Issue a non-blocking command to copy the buffer from the device.
    cl_int err =
        clEnqueueReadBuffer(commands_, it.second, CL_FALSE, 0, sizeInBytes,
                            T->getUnsafePtr(), 0, nullptr, nullptr);
    GLOW_ASSERT(err == CL_SUCCESS && "Unable to copy from the device");
  }
  clFinish(commands_);
}

void OCLBackend::registerGraphTensor(const Value *v, Tensor *t) {
  assert(!externalTensors_.count(v) && "The tensor is already registered");
  externalTensors_[v] = t;
}

void OCLBackend::init() {
  // Copy the weights into the device.

  // For each weight:
  for (auto it : externalTensors_) {
    Tensor *T = it.second;
    size_t sizeInBytes = T->getType().getSizeInBytes();

    // Allocate a new buffer, unless it was allocated in previous compilations.
    cl_mem buff;
    auto tt = tensors_.find(it.first);
    if (tt == tensors_.end()) {
      buff = clCreateBuffer(context_, CL_MEM_READ_WRITE, sizeInBytes, nullptr,
                            nullptr);
    } else {
      buff = tt->second;
    }
    GLOW_ASSERT(buff && "Allocation failed!");
    // Associate the new buffer with the weight value.
    tensors_[it.first] = buff;
  }
}

void OCLBackend::clear() { externalTensors_.clear(); }

Tensor *OCLBackend::getGradTensor(const Value *v) const {
  auto &map = M_->getGradientMap();
  auto it = map.find(v);
  assert(it != map.end() && "Gradient tensor unavailable");
  return getTensor(it->second);
}

Tensor *OCLBackend::getGradTensor(const Variable *v) const {
  auto *W = M_->getWeightForNode(v);
  return getGradTensor(W);
}

Tensor *OCLBackend::getTensor(const Variable *v) const {
  auto *W = M_->getWeightForNode(v);
  return getTensor(W);
}

Tensor *OCLBackend::getTensor(const Value *v) const {
  assert(externalTensors_.count(v) && "Unknown Value");
  auto ie = externalTensors_.find(v);
  return ie->second;
}

#else

Backend *glow::createOCLBackend(Module *M) {
  GLOW_ASSERT(false && "Glow is compiled without OpenCL support");
}

#endif // WITH_OPENCL
