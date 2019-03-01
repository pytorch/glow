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
#include "glow/Backends/Backend.h"
#include "glow/Graph/Graph.h"
#include "glow/Graph/Nodes.h"

#if defined(GLOW_WITH_OPENCL)
#include "OpenCL/OpenCL.h"
#endif

#if defined(GLOW_WITH_CPU)
#include "CPU/CPUBackend.h"
#endif

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
#else
Backend *createCPUBackend() {
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
  }

  // This is to make compiler happy. It can never reach this point as switch
  // always covers all possible values.
  llvm_unreachable("unreachable");
}

bool glow::transformPostLoweringStatic(BackendKind backendKind, Function *F,
                                       const CompilationOptions &opts) {
  switch (backendKind) {
  case BackendKind::Interpreter:
    return Interpreter::transformPostLoweringStatic(F, opts);

  case BackendKind::OpenCL:
#if defined(GLOW_WITH_OPENCL)
    return OCLBackend::transformPostLoweringStatic(F, opts);
#else
    GLOW_UNREACHABLE("Must compile with OpenCL support");
#endif

  case BackendKind::CPU:
#if defined(GLOW_WITH_CPU)
    return CPUBackend::transformPostLoweringStatic(F, opts);
#else
    GLOW_UNREACHABLE("Must compile with CPU support");
#endif
  }

  // This is to make compiler happy. It can never reach this point as switch
  // always covers all possible values.
  llvm_unreachable("unreachable");
}

bool glow::isOpSupportedStatic(BackendKind backendKind, const NodeInfo &NI) {
  switch (backendKind) {
  case BackendKind::Interpreter:
    return Interpreter::isOpSupportedStatic(NI);

  case BackendKind::OpenCL:
#if defined(GLOW_WITH_OPENCL)
    return OCLBackend::isOpSupportedStatic(NI);
#else
    GLOW_UNREACHABLE("Must compile with OpenCL support");
#endif

  case BackendKind::CPU:
#if defined(GLOW_WITH_CPU)
    return CPUBackend::isOpSupportedStatic(NI);
#else
    GLOW_UNREACHABLE("Must compile with CPU support");
#endif
  }

  // This is to make compiler happy. It can never reach this point as switch
  // always covers all possible values.
  llvm_unreachable("unreachable");
}

bool glow::shouldLowerStatic(BackendKind backendKind, const Node *N) {
  switch (backendKind) {
  case BackendKind::Interpreter:
    return Interpreter::shouldLowerStatic(N);

  case BackendKind::OpenCL:
#if defined(GLOW_WITH_OPENCL)
    return OCLBackend::shouldLowerStatic(N);
#else
    GLOW_UNREACHABLE("Must compile with OpenCL support");
#endif

  case BackendKind::CPU:
#if defined(GLOW_WITH_CPU)
    return CPUBackend::shouldLowerStatic(N);
#else
    GLOW_UNREACHABLE("Must compile with CPU support");
#endif
  }

  // This is to make compiler happy. It can never reach this point as switch
  // always covers all possible values.
  llvm_unreachable("unreachable");
}