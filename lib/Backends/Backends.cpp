// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "Interpreter/Interpreter.h"
#if defined(GLOW_WITH_JIT)
#include "CPUBackend/CPUBackend.h"
#endif
#if defined(GLOW_WITH_OPENCL)
#include "OpenCL/OpenCL.h"
#endif

#include "glow/Backends/Backend.h"
#include "glow/Graph/Graph.h"
#include "glow/Graph/Nodes.h"
#include "glow/IR/Instrs.h"
#include "glow/Optimizer/Optimizer.h"

#include "llvm/Support/Casting.h"

using namespace glow;

Backend *glow::createBackend(BackendKind backendKind, IRFunction *F) {
  switch (backendKind) {
  case BackendKind::Interpreter:
    return createInterpreter(F);
#if defined(GLOW_WITH_OPENCL)
  case BackendKind::OpenCL:
    return createOCLBackend(F);
#endif
#if defined(GLOW_WITH_JIT)
  case BackendKind::JIT:
    return createJIT(F);
#endif
  default:
    // Unknown execution backend.
    llvm_unreachable("Invalid backend kind.");
    break;
  }
}

void Backend::save(llvm::StringRef outputDir) {
  llvm_unreachable("Saving a bundle is not supported by the backend");
}
