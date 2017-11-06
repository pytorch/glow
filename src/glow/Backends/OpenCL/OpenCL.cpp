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
  clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

  // Allocate memory for the log.
  char *log = (char *)malloc(log_size);

  // Get the log.
  clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

  // Print the log.
  std::cout << log << "\n";
  free(log);
#endif
}

OCLBackend::OCLBackend(Module *M) : M_(M) {
  cl_int err =
      clGetDeviceIDs(NULL, CL_DEVICE_TYPE_DEFAULT, 1, &deviceId_, NULL);
  GLOW_ASSERT(err == CL_SUCCESS && "clGetDeviceIDs Failed.");

  context_ = clCreateContext(0, 1, &deviceId_, NULL, NULL, &err);
  GLOW_ASSERT(context_ && "clCreateContext Failed.");

  commands_ = clCreateCommandQueue(context_, deviceId_, 0, &err);
  GLOW_ASSERT(commands_ && "clCreateCommandQueue Failed.");

  err = CL_SUCCESS;
  program_ = clCreateProgramWithSource(context_, 1, &SHADER_CODE, NULL, &err);
  GLOW_ASSERT(program_ && "clCreateProgramWithSource Failed.");
  err = clBuildProgram(program_, 0, NULL, NULL, NULL, NULL);
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

size_t getMaxLocalWorkgroupSize(cl_kernel kernel, cl_device_id device,
                                size_t globalWGSize0,
                                size_t globalWGSize1 = 0) {
  // Figure out the max size of the workgroup.
  size_t L;
  auto err = clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE,
                                      sizeof(L), &L, NULL);
  GLOW_ASSERT(err == CL_SUCCESS && "Error in clGetKernelWorkGroupInfo.");
  size_t res = L;
  // The global workgroup size must be a multiple of the local workgroup size.
  // In here we find the highest L that divides the global workgroup size.
  // This is our naive implementation of gcd:
  while (globalWGSize0 % res || globalWGSize1 % res || L % res) {
    res--;
  }
  return res;
}

void enqueueKernel(cl_command_queue commands, cl_kernel kernel,
                   llvm::ArrayRef<size_t> global,
                   llvm::ArrayRef<size_t> local) {
  assert(global.size() == local.size());
  auto err = clEnqueueNDRangeKernel(commands, kernel, global.size(), NULL,
                                    &global[0], &local[0], 0, NULL, NULL);
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
      // Figure out the size of the workgroup.
      size_t local = getMaxLocalWorkgroupSize(kernel, deviceId_, global);

      enqueueKernel(commands_, kernel, {global}, {local});
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

      // Figure out the max size of the workgroup.
      size_t L = getMaxLocalWorkgroupSize(kernel, deviceId_, numSlices);

      const size_t local = L;
      const size_t global = numSlices;
      enqueueKernel(commands_, kernel, {global}, {local});
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

      // Figure out the max size of the workgroup.
      size_t L = getMaxLocalWorkgroupSize(kernel, deviceId_, numSlices, depth);

      // Use a 2D grid where the first dimension is the depth and the second
      // dimension is the slice index in the batch.
      enqueueKernel(commands_, kernel, {depth, numSlices}, {L, L});
      kernels.push_back(kernel);
      continue;
    }

    if (auto *CC = dyn_cast<ConvolutionInst>(I)) {
      // This is a naive implementation that parallelizes using two dimensions:
      // the X and the Y in the output filter.
      cl_kernel kernel = createKernel(program_, kernelName);

      unsigned numArgs = I->getNumOperands();
      for (unsigned arg = 0; arg < numArgs; arg++) {
        setKernelArg(kernel, arg, tensors_[I->getOperand(arg).first]);
      }

      setKernelArg<size_t>(kernel, 4, CC->getKernel());
      setKernelArg(kernel, 5, CC->getPad());
      setKernelArg(kernel, 6, CC->getStride());
      setKernelArg(kernel, 7, ShapeNHWC(CC->getDest()->getType()->dims()));
      setKernelArg(kernel, 8, ShapeNHWC(CC->getSrc()->getType()->dims()));
      setKernelArg(kernel, 9, ShapeNHWC(CC->getFilter()->getType()->dims()));

      auto odim = ShapeNHWC(CC->getDest()->getType()->dims());

      // Figure out the max size of the workgroup.
      size_t L = getMaxLocalWorkgroupSize(kernel, deviceId_, odim.h, odim.w);

      // Use a 2D grid where the first dimension is the depth and the second
      // dimension is the slice index in the batch.
      enqueueKernel(commands_, kernel, {odim.h, odim.w}, {L, L});
      kernels.push_back(kernel);
      continue;
    }

    if (isa<CopyInst>(I) || isa<ReshapeInst>(I))  {
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

    assert(false && "Unexpected node");
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
