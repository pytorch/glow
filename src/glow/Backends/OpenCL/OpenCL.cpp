// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "OpenCL.h"

#include "glow/Graph/Graph.h"
#include "glow/Graph/Nodes.h"
#include "glow/IR/Instrs.h"
#include "glow/Optimizer/Optimizer.h"

#include "llvm/Support/Casting.h"

using namespace glow;

Backend *glow::createOCLBackend(Module *M) { return new OCLBackend(M); }

const char *ReluSrc =
    "__kernel void op(__global float* dest, __global float* src)"
    "{ size_t i = get_global_id(0); dest[i] = fmax(src[i], 0); }";
const char *RegressionSrc =
    "__kernel void op(__global float* dest, __global float* src)"
    "{ size_t i = get_global_id(0); dest[i] = src[i]; }";

using Kind = Kinded::Kind;
using kernelSrcEnum = struct {
  Kind kind;
  const char *src;
};
kernelSrcEnum shaders[] = {{Kind::ReluInstKind, ReluSrc},
                           {Kind::RegressionInstKind, RegressionSrc}};

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

using namespace glow;
OCLBackend::OCLBackend(Module *M) : M_(M) {
  cl_int err =
      clGetDeviceIDs(NULL, CL_DEVICE_TYPE_DEFAULT, 1, &deviceId_, NULL);
  GLOW_ASSERT(err == CL_SUCCESS && "clGetDeviceIDs Failed.");

  context_ = clCreateContext(0, 1, &deviceId_, NULL, NULL, &err);
  GLOW_ASSERT(context_ && "clCreateContext Failed.");

  commands_ = clCreateCommandQueue(context_, deviceId_, 0, &err);
  GLOW_ASSERT(commands_ && "clCreateCommandQueue Failed.");

  // Compile all of the shaders:
  for (auto SH : shaders) {
    err = CL_SUCCESS;
    cl_program program =
        clCreateProgramWithSource(context_, 1, &SH.src, NULL, &err);
    GLOW_ASSERT(program && "clCreateProgramWithSource Failed.");
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err) {
      dumpCompileLog(deviceId_, program);
    }
    GLOW_ASSERT(err == CL_SUCCESS && "clBuildProgram Failed.");
    assert(!programs_.count(SH.kind) && "Opcode already registered");
    programs_[SH.kind] = program;
  }
}

OCLBackend::~OCLBackend() {
  // Free all of the compiled programs.
  for (auto &p : programs_) {
    clReleaseProgram(p.second);
  }
  programs_.clear();

  clReleaseCommandQueue(commands_);
  clReleaseContext(context_);
  clear();
}

void OCLBackend::doForwardPass(bool isTrain) {
  copyWeightsToDevice();

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
  clFlush(commands_);
}

void OCLBackend::copyWeightsFromDevice() {
  for (auto it : tensors_) {
    assert(externalTensors_.count(it.first) && "Unknown weight!");
    Tensor *T = externalTensors_[it.first];
    size_t sizeInBytes = T->getType().getSizeInBytes();
    // Issue a non-blocking command to copy the buffer from the device.
    cl_int err =
        clEnqueueReadBuffer(commands_, it.second, CL_FALSE, 0, sizeInBytes,
                            T->getUnsafePtr(), 0, nullptr, nullptr);
    GLOW_ASSERT(err == CL_SUCCESS && "Unable to copy data to the device");
  }
  // Do it!
  clFlush(commands_);
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
