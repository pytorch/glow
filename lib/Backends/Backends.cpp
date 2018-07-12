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

#include "glow/Backends/Backend.h"
#include "glow/Graph/Graph.h"
#include "glow/Graph/Nodes.h"
#include "glow/IR/Instrs.h"

#include "llvm/Support/Casting.h"

using namespace glow;

namespace glow {
/// NOTE: Please add a declaration of a backend-specific `create` method here
/// when you define a new backend.

/// Create a new instance of the interpreter backend.
Backend *createInterpreter();

#if defined(GLOW_WITH_CPU)
/// Create a new instance of the CPUBackend backend.
Backend *createCPUBackend();
Backend *createMultiCPUBackend();
#else
Backend *createCPUBackend() {
  GLOW_UNREACHABLE("Must compile with CPU support");
}

Backend *createMultiCPUBackend() {
  GLOW_UNREACHABLE("Must compile with CPU support");
}
#endif

#if defined(GLOW_WITH_OPENCL)
/// Create a new instance of the OpenCL backend.
Backend *createOCLBackend();
#else
Backend *createOCLBackend() {
  GLOW_UNREACHABLE("Must compile with OpenCL support");
}
#endif
} // namespace glow

Backend *glow::createBackend(BackendKind backendKind) {
  switch (backendKind) {
  case BackendKind::Interpreter:
    return createInterpreter();
  case BackendKind::OpenCL:
    return createOCLBackend();
  case BackendKind::CPU:
    return createCPUBackend();
  case BackendKind::MultiCPU:
    return createMultiCPUBackend();
  }

  // This is to make compiler happy. It can never reach this point as switch
  // always covers all possible values.
  llvm_unreachable("unreachable");
}
