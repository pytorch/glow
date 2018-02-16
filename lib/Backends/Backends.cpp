// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "Interpreter/Interpreter.h"
#if defined(GLOW_WITH_JIT)
#include "JIT/JIT.h"
#endif
#if defined(GLOW_WITH_OPENCL)
#include "OpenCL/OpenCL.h"
#endif

#include "glow/Backends/Backend.h"
#include "glow/Graph/Nodes.h"
#include "glow/IR/Instrs.h"
#include "glow/Optimizer/Optimizer.h"

#include "llvm/Support/Casting.h"

using namespace glow;

Backend *glow::createBackend(BackendKind backendKind, Module *M) {
  switch (backendKind) {
  case BackendKind::Interpreter:
    return createInterpreter(M);
#if defined(GLOW_WITH_OPENCL)
  case BackendKind::OpenCL:
    return createOCLBackend(M);
#endif
#if defined(GLOW_WITH_JIT)
  case BackendKind::JIT:
    return createJIT(M);
#endif
  default:
    // Unknown execution backend.
    llvm_unreachable("Invalid backend kind.");
    break;
  }
}
