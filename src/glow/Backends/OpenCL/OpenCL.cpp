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
    "{ size_t i = get_global_id(0); output[i] = max(input[i], 0); }";
const char *RegressionSrc =
    "__kernel void op(__global float* dest, __global float* src)"
    "{ size_t i = get_global_id(0); output[i] = input[i]; }";

using Kind = Kinded::Kind;
using kernelSrcEnum = struct {
  Kind kind;
  const char *src;
};
kernelSrcEnum shaders[] = {{Kind::ReluInstKind, ReluSrc},
                           {Kind::RegressionInstKind, RegressionSrc}};

using namespace glow;
OCLBackend::OCLBackend(Module *M) : M_(M) {
  cl_int err =
      clGetDeviceIDs(NULL, CL_DEVICE_TYPE_DEFAULT, 1, &deviceId_, NULL);
  GLOW_ASSERT(err == CL_SUCCESS && "clGetDeviceIDs Failed.");

  context_ = clCreateContext(0, 1, &deviceId_, NULL, NULL, &err);
  GLOW_ASSERT(context_ && "clCreateContext Failed.");

  commands_ = clCreateCommandQueue(context_, deviceId_, 0, &err);
  GLOW_ASSERT(commands_ && "clCreateCommandQueue Failed.");

  for (auto SH : shaders) {
    err = CL_SUCCESS;
    cl_program program =
        clCreateProgramWithSource(context_, 1, &SH.src, NULL, &err);
    GLOW_ASSERT(program && "clCreateProgramWithSource Failed.");
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
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

void OCLBackend::doForwardPass(bool isTrain) {}

void OCLBackend::registerGraphTensor(const Value *v, Tensor *t) {
  assert(!externalTensors_.count(v) && "The tensor is already registered");
  externalTensors_[v] = t;
}

void OCLBackend::init() {}

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
