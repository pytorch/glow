/**
 * Copyright (c) Glow Contributors. See CONTRIBUTORS file.
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

#ifndef GLOW_OPTIMIZER_IROPTIMIZER_COMMANDLINE_H
#define GLOW_OPTIMIZER_IROPTIMIZER_COMMANDLINE_H

#include "llvm/Support/CommandLine.h"

/// Option to optimize the IR (default true).
extern llvm::cl::opt<bool> optimizeIR;

/// Option to dump the IR (default false).
extern llvm::cl::opt<bool> dumpIR;

/// Option to enable the IR debug instrumentation (default false).
extern llvm::cl::opt<bool> instrumentDebug;

/// Option to choose the IR debug instrumentation directory.
/// Default directory is 'debug'.
extern llvm::cl::opt<std::string> instrumentDebugDir;

/// Option to choose the IR debug instrumentation format. The supported formats
/// are 'console', 'bin', 'txt', 'rawbin' and 'rawtxt'.
extern llvm::cl::opt<std::string> instrumentDebugFormat;

/// Option to choose to instrument only the listed instructions.
extern llvm::cl::list<std::string> instrumentDebugOnly;

/// Option to enable the IR instrumentation (default false).
extern llvm::cl::opt<bool> instrumentIR;

/// Option to choose to instrument only the listed instructions.
extern llvm::cl::list<std::string> instrumentIROnly;

#endif // GLOW_OPTIMIZER_IROPTIMIZER_COMMANDLINE_H
