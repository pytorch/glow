// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "OpenCL.h"

#include "glow/Graph/Graph.h"
#include "glow/Graph/Nodes.h"
#include "glow/IR/Instrs.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

using namespace glow;

using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;

// This defines the string "SHADER_CODE".
#include "kernels.cl"

namespace {
llvm::cl::OptionCategory OpenCLBackendCat("Glow OpenCL Backend Options");

static llvm::cl::opt<int> deviceId("device",
                                   llvm::cl::desc("OpenCL device to be used"),
                                   llvm::cl::init(0),
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
  if (deviceBuffer_) {
    clReleaseMemObject(deviceBuffer_);
    deviceBuffer_ = nullptr;
  }
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
  llvm::SmallVector<size_t, 4> local(global.size(), 0);
  getMaxLocalWorkgroupSize(kernel, device, global, local);

  auto err = clEnqueueNDRangeKernel(commands, kernel, global.size(), nullptr,
                                    &global[0], &local[0], 0, nullptr, nullptr);
  GLOW_ASSERT(err == CL_SUCCESS && "Error in clEnqueueNDRangeKernel.");
}

void OCLBackend::doForwardPass() {
  copyWeightsToDevice();
  std::vector<cl_kernel> kernels;

  for (auto &I : F_->getInstrs()) {
    // The kernels are named after the name of the instruction, plus the "W"
    // suffix to prevent name colissions for functions like 'tanh' that are also
    // a part of the OpenCL runtime.
    std::string kernelName = std::string(I->getKindName()) + "W";

    if (isa<AllocActivationInst>(I) || isa<DeallocActivationInst>(I)) {
      continue;
    }

    // Element-wise operations:
    if (isa<SigmoidInst>(I) || isa<TanhInst>(I) || isa<ElementAddInst>(I) ||
        isa<ElementSubInst>(I) || isa<ElementMaxInst>(I) ||
        isa<ElementMinInst>(I) || isa<ElementCmpLTEInst>(I) ||
        isa<ElementMulInst>(I) || isa<ElementDivInst>(I)) {

      cl_kernel kernel = createKernel(program_, kernelName);
      setKernelArg(kernel, 0, deviceBuffer_);

      for (unsigned arg = 0, e = I->getNumOperands(); arg < e; arg++) {
        setKernelArg(kernel, arg + 1, tensors_[I->getOperand(arg).first]);
      }

      // Figure out how many element-wise elements are there to process:
      size_t global;
      if (auto *tmpInst = dyn_cast<SigmoidInst>(I)) {
        global = tmpInst->getDest()->getType()->size();
      } else if (auto *tmpInst = dyn_cast<TanhInst>(I)) {
        global = tmpInst->getDest()->getType()->size();
      } else if (auto *tmpInst = dyn_cast<ElementAddInst>(I)) {
        global = tmpInst->getDest()->getType()->size();
      } else if (auto *tmpInst = dyn_cast<ElementSubInst>(I)) {
        global = tmpInst->getDest()->getType()->size();
      } else if (auto *tmpInst = dyn_cast<ElementMaxInst>(I)) {
        global = tmpInst->getDest()->getType()->size();
      } else if (auto *tmpInst = dyn_cast<ElementMinInst>(I)) {
        global = tmpInst->getDest()->getType()->size();
      } else if (auto *tmpInst = dyn_cast<ElementCmpLTEInst>(I)) {
        global = tmpInst->getDest()->getType()->size();
      } else if (auto *tmpInst = dyn_cast<ElementMulInst>(I)) {
        global = tmpInst->getDest()->getType()->size();
      } else if (auto *tmpInst = dyn_cast<ElementDivInst>(I)) {
        global = tmpInst->getDest()->getType()->size();
      } else {
        GLOW_UNREACHABLE("Invalid instruction.");
      }

      enqueueKernel(commands_, kernel, deviceId_, {global});
      kernels.push_back(kernel);
      continue;
    }

    // Element-wise operations:
    if (auto *SI = dyn_cast<SplatInst>(I)) {
      cl_kernel kernel = createKernel(program_, kernelName);
      setKernelArg(kernel, 0, deviceBuffer_);
      unsigned numArgs = I->getNumOperands();

      for (unsigned arg = 0; arg < numArgs; arg++) {
        setKernelArg(kernel, arg + 1, tensors_[I->getOperand(arg).first]);
      }

      // Pass the splat as a parameter.
      setKernelArg(kernel, numArgs + 1, SI->getValue());

      // Figure out how many element-wise elements are there to process:
      size_t global = SI->getDest()->getType()->size();

      enqueueKernel(commands_, kernel, deviceId_, {global});
      kernels.push_back(kernel);
      continue;
    }

    if (auto *SM = dyn_cast<SoftMaxInst>(I)) {
      // Implement Softmax by parallelizing the batch dimension. Each sample in
      // the batch is processed by a different parallel 'thread'.
      cl_kernel kernel = createKernel(program_, kernelName);

      setKernelArg(kernel, 0, deviceBuffer_);

      unsigned numArgs = I->getNumOperands();
      for (unsigned arg = 0; arg < numArgs; arg++) {
        setKernelArg(kernel, arg + 1, tensors_[I->getOperand(arg).first]);
      }

      // This is the number of elements for each slice. There are N slices in
      // our batch.
      auto inputDims = SM->getSrc()->getType()->dims();
      size_t numSlices = inputDims[0];

      // Pass the slice size (size of each sample in the batch) as a parameter.
      setKernelArg(kernel, numArgs + 1, flattenCdr(inputDims).second);

      enqueueKernel(commands_, kernel, deviceId_, {numSlices});
      kernels.push_back(kernel);
      continue;
    }

    if (auto *CI = dyn_cast<InsertTensorInst>(I)) {
      cl_kernel kernel = createKernel(program_, kernelName);
      setKernelArg(kernel, 0, deviceBuffer_);

      unsigned numArgs = I->getNumOperands();
      for (unsigned arg = 0; arg < numArgs; arg++) {
        setKernelArg(kernel, arg + 1, tensors_[I->getOperand(arg).first]);
      }

      // Currently support tensors of 2 and 4 dimensions.
      // TODO: Handle other dimensions.
      const size_t numDimensions = CI->getDest()->getType()->dims().size();
      ShapeNHWC odim = ShapeNHWC::empty();
      ShapeNHWC idim = ShapeNHWC::empty();
      ShapeNHWC offset = ShapeNHWC::empty();

      if (numDimensions == 4) {
        odim = ShapeNHWC(CI->getDest()->getType()->dims());
        idim = ShapeNHWC(CI->getSrc()->getType()->dims());
        offset = ShapeNHWC(CI->getOffsets());
      } else if (numDimensions == 2) {
        odim = ShapeNHWC::fromXY(CI->getDest()->getType()->dims());
        idim = ShapeNHWC::fromXY(CI->getSrc()->getType()->dims());
        offset = ShapeNHWC::fromXY(CI->getOffsets());
      } else {
        assert(false && "Unsupported tensor dimension");
      }

      setKernelArg(kernel, 3, odim);
      setKernelArg(kernel, 4, idim);
      setKernelArg(kernel, 5, offset);
      enqueueKernel(commands_, kernel, deviceId_, {idim.n});
      kernels.push_back(kernel);

      continue;
    }

    if (auto *BMM = dyn_cast<MatMulInst>(I)) {
      // This is a naive implementation that parallelizes using three dims:
      // batch, X and Y in the output filter.
      cl_kernel kernel = createKernel(program_, kernelName);
      setKernelArg(kernel, 0, deviceBuffer_);

      unsigned numArgs = I->getNumOperands();
      for (unsigned arg = 0; arg < numArgs; arg++) {
        setKernelArg(kernel, arg + 1, tensors_[I->getOperand(arg).first]);
      }

      auto ddim = ShapeNHWC::fromXY(BMM->getDest()->getType()->dims());
      auto ldim = ShapeNHWC::fromXY(BMM->getLHS()->getType()->dims());
      auto rdim = ShapeNHWC::fromXY(BMM->getRHS()->getType()->dims());

      setKernelArg(kernel, 4, ddim);
      setKernelArg(kernel, 5, ldim);
      setKernelArg(kernel, 6, rdim);

      // Use a 3D grid where the first dimension is the N and the second and
      // third dimensions are the X and Y in the output buffer.
      enqueueKernel(commands_, kernel, deviceId_, {ddim.n, ddim.h, ddim.w});
      kernels.push_back(kernel);
      continue;
    }

    if (auto *BA = dyn_cast<BatchedAddInst>(I)) {
      cl_kernel kernel = createKernel(program_, kernelName);
      setKernelArg(kernel, 0, deviceBuffer_);

      unsigned numArgs = I->getNumOperands();
      for (unsigned arg = 0; arg < numArgs; arg++) {
        setKernelArg(kernel, arg + 1, tensors_[I->getOperand(arg).first]);
      }

      auto bdim = flattenCdr(BA->getBatch()->dims());
      setKernelArg(kernel, 4, bdim.first);
      setKernelArg(kernel, 5, bdim.second);

      // Parallelize on each element in the slice.
      enqueueKernel(commands_, kernel, deviceId_, {bdim.second});
      kernels.push_back(kernel);
      continue;
    }

    if (auto *BRA = dyn_cast<BatchedReduceAddInst>(I)) {
      cl_kernel kernel = createKernel(program_, kernelName);
      setKernelArg(kernel, 0, deviceBuffer_);

      unsigned numArgs = I->getNumOperands();
      for (unsigned arg = 0; arg < numArgs; arg++) {
        setKernelArg(kernel, arg + 1, tensors_[I->getOperand(arg).first]);
      }

      auto bdim = flattenCdr(BRA->getBatch()->dims());
      setKernelArg(kernel, 3, bdim.first);
      setKernelArg(kernel, 4, bdim.second);

      // Parallelize on each element in the slice.
      enqueueKernel(commands_, kernel, deviceId_, {bdim.second});
      kernels.push_back(kernel);
      continue;
    }

    if (auto *CC = dyn_cast<ConvolutionInst>(I)) {
      // This is a naive implementation that parallelizes using three dims:
      // the X and the Y in the output filter.
      cl_kernel kernel = createKernel(program_, kernelName);
      setKernelArg(kernel, 0, deviceBuffer_);

      unsigned numArgs = I->getNumOperands();
      for (unsigned arg = 0; arg < numArgs; arg++) {
        setKernelArg(kernel, arg + 1, tensors_[I->getOperand(arg).first]);
      }

      auto odim = ShapeNHWC(CC->getDest()->getType()->dims());
      auto idim = ShapeNHWC(CC->getSrc()->getType()->dims());

      setKernelArg<size_t>(kernel, 5, CC->getKernel());
      setKernelArg(kernel, 6, CC->getStride());
      setKernelArg(kernel, 7, CC->getPad());
      setKernelArg(kernel, 8, odim);
      setKernelArg(kernel, 9, idim);
      setKernelArg(kernel, 10, ShapeNHWC(CC->getFilter()->getType()->dims()));

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
      setKernelArg(kernel, 0, deviceBuffer_);

      unsigned numArgs = I->getNumOperands();
      for (unsigned arg = 0; arg < numArgs; arg++) {
        setKernelArg(kernel, arg + 1, tensors_[I->getOperand(arg).first]);
      }

      auto odim = ShapeNHWC(PM->getDest()->getType()->dims());
      auto idim = ShapeNHWC(PM->getSrc()->getType()->dims());

      setKernelArg<size_t>(kernel, numArgs + 1, PM->getKernel());
      setKernelArg(kernel, numArgs + 2, PM->getStride());
      setKernelArg(kernel, numArgs + 3, PM->getPad());
      setKernelArg(kernel, numArgs + 4, odim);
      setKernelArg(kernel, numArgs + 5, idim);

      enqueueKernel(commands_, kernel, deviceId_, {odim.h, odim.w, odim.c});
      kernels.push_back(kernel);
      continue;
    }

    if (auto *PM = dyn_cast<PoolMaxWithXYInst>(I)) {
      // This is a naive implementation that parallelizes using three dims:
      // the X and the Y in the output filter.
      cl_kernel kernel = createKernel(program_, kernelName);
      setKernelArg(kernel, 0, deviceBuffer_);

      unsigned numArgs = I->getNumOperands();
      for (unsigned arg = 0; arg < numArgs; arg++) {
        setKernelArg(kernel, arg + 1, tensors_[I->getOperand(arg).first]);
      }

      auto odim = ShapeNHWC(PM->getDest()->getType()->dims());
      auto idim = ShapeNHWC(PM->getSrc()->getType()->dims());

      setKernelArg<size_t>(kernel, numArgs + 1, PM->getKernel());
      setKernelArg(kernel, numArgs + 2, PM->getStride());
      setKernelArg(kernel, numArgs + 3, PM->getPad());
      setKernelArg(kernel, numArgs + 4, odim);
      setKernelArg(kernel, numArgs + 5, idim);

      enqueueKernel(commands_, kernel, deviceId_, {odim.h, odim.w, odim.c});
      kernels.push_back(kernel);
      continue;
    }

    if (auto *PA = dyn_cast<PoolAvgInst>(I)) {
      // This is a naive implementation that parallelizes using three dims:
      // the X and the Y in the output filter.
      cl_kernel kernel = createKernel(program_, kernelName);
      setKernelArg(kernel, 0, deviceBuffer_);

      unsigned numArgs = I->getNumOperands();
      for (unsigned arg = 0; arg < numArgs; arg++) {
        setKernelArg(kernel, arg + 1, tensors_[I->getOperand(arg).first]);
      }

      auto odim = ShapeNHWC(PA->getDest()->getType()->dims());
      auto idim = ShapeNHWC(PA->getSrc()->getType()->dims());

      setKernelArg<size_t>(kernel, 3, PA->getKernel());
      setKernelArg(kernel, 4, PA->getStride());
      setKernelArg(kernel, 5, PA->getPad());
      setKernelArg(kernel, 6, odim);
      setKernelArg(kernel, 7, idim);

      enqueueKernel(commands_, kernel, deviceId_, {odim.h, odim.w, odim.c});
      kernels.push_back(kernel);
      continue;
    }

    if (auto *TR = dyn_cast<TransposeInst>(I)) {
      // This is a naive implementation that parallelizes using one dimension,
      // the N (batch size).
      GLOW_ASSERT(TR->getShuffle().size() <= 4 &&
                  "This code supports only 4 and lower dimensional transposes");

      cl_kernel kernel = createKernel(program_, kernelName);
      setKernelArg(kernel, 0, deviceBuffer_);

      unsigned numArgs = I->getNumOperands();
      for (unsigned arg = 0; arg < numArgs; arg++) {
        setKernelArg(kernel, arg + 1, tensors_[I->getOperand(arg).first]);
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

      enqueueKernel(commands_, kernel, deviceId_, {idim.n});
      kernels.push_back(kernel);
      continue;
    }

    if (auto *TV = dyn_cast<TensorViewInst>(I)) {
      assert(tensors_[TV] == tensors_[TV->getSrc()] &&
             "Memory address for a tensor_view should be the same as the "
             "address of its origin");
      (void)TV;
      continue;
    }

    if (auto *C = dyn_cast<CopyInst>(I)) {
      Value *dest, *src;
      dest = C->getDest();
      src = C->getSrc();
      if (src == dest) {
        continue;
      }
      size_t destOff = tensors_[dest];
      size_t srcOff = tensors_[src];
      size_t sizeInBytes = dest->getType()->getSizeInBytes();

      cl_int err =
          clEnqueueCopyBuffer(commands_, deviceBuffer_, deviceBuffer_, srcOff,
                              destOff, sizeInBytes, 0, nullptr, nullptr);
      GLOW_ASSERT(err == CL_SUCCESS && "Error in clEnqueueCopyBuffer.");
      continue;
    }

    llvm::errs() << "Cannot select: " << I->getKindName() << "\n";
    llvm::report_fatal_error("compilation failed");
  }

  clFinish(commands_);

  for (auto &k : kernels) {
    clReleaseKernel(k);
  }

  copyWeightsFromDevice();
}

void OCLBackend::copyWeightsToDevice() {
  for (auto it : tensors_) {
    if (!externalTensors_.count(it.first)) {
      continue;
    }
    Tensor *T = externalTensors_[it.first];
    size_t sizeInBytes = T->getType().getSizeInBytes();
    // Issue a non-blocking command to copy the buffer to the device.
    if (sizeInBytes) {
      cl_int err = clEnqueueWriteBuffer(commands_, deviceBuffer_, CL_FALSE,
                                        it.second, sizeInBytes,
                                        T->getUnsafePtr(), 0, nullptr, nullptr);
      GLOW_ASSERT(err == CL_SUCCESS && "Unable to copy data to the device");
    }
  }
  // Do it!
  clFinish(commands_);
}

void OCLBackend::copyWeightsFromDevice() {
  clFinish(commands_);

  for (auto it : tensors_) {
    if (!externalTensors_.count(it.first)) {
      continue;
    }

    Tensor *T = externalTensors_[it.first];
    size_t sizeInBytes = T->getType().getSizeInBytes();

    // Issue a non-blocking command to copy the buffer from the device.
    cl_int err = clEnqueueReadBuffer(commands_, deviceBuffer_, CL_FALSE,
                                     it.second, sizeInBytes, T->getUnsafePtr(),
                                     0, nullptr, nullptr);
    GLOW_ASSERT(err == CL_SUCCESS && "Unable to copy from the device");
  }
  clFinish(commands_);
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
  for (auto &I : F_->getInstrs()) {
    if (auto *A = llvm::dyn_cast<AllocActivationInst>(I)) {
      auto numBytes = I->getType()->getSizeInBytes();
      size_t addr = allocator_.allocate(numBytes);
      assert(!tensors_.count(A) && "Allocation already made!");
      tensors_[A] = addr;
      continue;
    }

    if (auto *TV = llvm::dyn_cast<TensorViewInst>(I)) {
      assert(!tensors_.count(TV) && "Allocation already made!");
      tensors_[TV] = tensors_[TV->getSrc()];
      continue;
    }

    if (auto *D = llvm::dyn_cast<DeallocActivationInst>(I)) {
      auto *A = D->getAlloc();
      assert(tensors_.count(A) && "Invalid deallocation!");
      allocator_.deallocate(tensors_[A]);
      continue;
    }
  }

  // Ask the memory allocator how much memory is required. What was the high
  // watermark for this program.
  size_t requiredSpace = allocator_.getMaxMemoryUsage();

  // Release the memory from the previous run.
  if (deviceBuffer_) {
    clReleaseMemObject(deviceBuffer_);
    deviceBuffer_ = nullptr;
  }

  deviceBuffer_ = clCreateBuffer(context_, CL_MEM_READ_WRITE, requiredSpace,
                                 nullptr, nullptr);
  GLOW_ASSERT(deviceBuffer_ && "Allocation failed!");
}

void OCLBackend::clear() { externalTensors_.clear(); }

Tensor *OCLBackend::getTensor(const Value *v) const {
  assert(externalTensors_.count(v) && "Unknown Value");
  auto ie = externalTensors_.find(v);
  return ie->second;
}
