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

#include "glow/Optimizer/IROptimizer/CommandLine.h"

static llvm::cl::OptionCategory IROptimizerCat("IR Optimizer Options");

llvm::cl::opt<bool> optimizeIR("optimize-ir",
                               llvm::cl::desc("Enable IR optimizations"),
                               llvm::cl::init(true),
                               llvm::cl::cat(IROptimizerCat));

llvm::cl::opt<bool> dumpIR("dump-ir", llvm::cl::desc("Prints IR to stdout"),
                           llvm::cl::init(false),
                           llvm::cl::cat(IROptimizerCat));

llvm::cl::opt<bool>
    instrumentDebug("instrument-debug",
                    llvm::cl::desc("Instrument the IR for debugging"),
                    llvm::cl::init(false), llvm::cl::cat(IROptimizerCat));

llvm::cl::opt<std::string> instrumentDebugDir(
    "instrument-debug-dir",
    llvm::cl::desc("The directory where the file dumps will be written!\n"),
    llvm::cl::init("debug"), llvm::cl::cat(IROptimizerCat));

llvm::cl::opt<std::string> instrumentDebugFormat(
    "instrument-debug-format",
    llvm::cl::desc(
        "The format of the IR debugging instrumentation:                     \n"
        "- 'console': The tensors are dumped in text format in the console.  \n"
        "- 'bin': The tensors are dumped in binary format in separate files. \n"
        "         Each file will contain the tensor type and tensor data.    \n"
        "- 'txt': The tensors are dumped in text format in separate files.   \n"
        "         Each file will contain the tensor type and tensor data.    \n"
        "- 'rawbin': The tensors are dumped in raw binary format in separate \n"
        "            files. Each file will contain ONLY the tensor data.     \n"
        "- 'rawtxt': The tensors are dumped in raw text format in separate   \n"
        "            files. Each file will contain ONLY the tensor data.\n"),
    llvm::cl::init("console"), llvm::cl::cat(IROptimizerCat));

llvm::cl::list<std::string> instrumentDebugOnly(
    "instrument-debug-only",
    llvm::cl::desc(
        "Instrument the IR for debugging, but only the listed instructions"),
    llvm::cl::CommaSeparated, llvm::cl::cat(IROptimizerCat));

llvm::cl::opt<bool>
    instrumentIR("instrument-ir",
                 llvm::cl::desc("Instrument the IR instructions"),
                 llvm::cl::init(false), llvm::cl::cat(IROptimizerCat));

llvm::cl::list<std::string> instrumentIROnly(
    "instrument-ir-only",
    llvm::cl::desc("Instrument the IR but only the listed instructions names"),
    llvm::cl::CommaSeparated, llvm::cl::cat(IROptimizerCat));
