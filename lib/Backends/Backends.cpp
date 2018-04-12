// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "Interpreter/Interpreter.h"
#if defined(GLOW_WITH_CPU)
#include "CPU/CPUBackend.h"
#endif
#if defined(GLOW_WITH_OPENCL)
#include "OpenCL/OpenCL.h"
#endif

#include "glow/Backends/Backend.h"
#include "glow/Graph/Graph.h"
#include "glow/Graph/Nodes.h"
#include "glow/IR/Instrs.h"

#include "llvm/Support/Casting.h"

using namespace glow;

Backend *glow::createBackend(BackendKind backendKind, IRFunction *F) {
  switch (backendKind) {
  case BackendKind::Interpreter:
    return createInterpreter(F);
  case BackendKind::OpenCL:
    #ifndef GLOW_WITH_OPENCL
      GLOW_UNREACHABLE("Must compile with OpenCL support");
    #else
      return createOCLBackend(F);
    #endif
  case BackendKind::CPU:
    #ifndef GLOW_WITH_CPU
      GLOW_UNREACHABLE("Must compile with CPU support");
    #else
      return createCPUBackend(F);
    #endif
  }
}

void Backend::save(llvm::StringRef outputDir) {
  GLOW_UNREACHABLE("Saving a bundle is not supported by the backend");
}
