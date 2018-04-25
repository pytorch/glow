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

  // This is to make compiler happy. It can never reach this point as switch
  // always covers all possible values.
  llvm_unreachable("unreachable");
}

void Backend::save(llvm::StringRef outputDir) {
  GLOW_UNREACHABLE("Saving a bundle is not supported by the backend");
}
