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

#include "CommandLine.h"

llvm::cl::OptionCategory &getLLVMBackendCat() {
  static llvm::cl::OptionCategory cpuBackendCat("Glow CPU Backend Options");
  return cpuBackendCat;
}

llvm::cl::opt<std::string> llvmTarget("target",
                                      llvm::cl::desc("LLVM target to be used"));

llvm::cl::opt<std::string>
    llvmArch("march", llvm::cl::desc("LLVM architecture to be used"));

llvm::cl::opt<std::string> llvmCPU("mcpu",
                                   llvm::cl::desc("LLVM CPU to be used"));

llvm::cl::list<std::string>
    llvmTargetFeatures("target-feature",
                       llvm::cl::desc("LLVM target/CPU features to be used"),
                       llvm::cl::CommaSeparated, llvm::cl::ZeroOrMore);

llvm::cl::alias llvmMAttr("mattr", llvm::cl::desc("Alias for -target-feature"),
                          llvm::cl::aliasopt(llvmTargetFeatures));

llvm::cl::opt<std::string>
    llvmCompiler("llvm-compiler",
                 llvm::cl::desc("External LLVM compiler (e.g. llc) to use for "
                                "compiling LLVM bitcode into machine code"));

llvm::cl::list<std::string> llvmCompilerOptions(
    "llvm-compiler-opt",
    llvm::cl::desc("Options to pass to the external LLVM compiler"),
    llvm::cl::ZeroOrMore);

llvm::cl::opt<llvm::FloatABI::ABIType>
    floatABI("float-abi", llvm::cl::desc("Option to set float ABI type"),
             llvm::cl::values(clEnumValN(llvm::FloatABI::Default, "default",
                                         "Default float ABI type"),
                              clEnumValN(llvm::FloatABI::Soft, "soft",
                                         "Soft float ABI (softfp)"),
                              clEnumValN(llvm::FloatABI::Hard, "hard",
                                         "Hard float ABI (hardfp)")),
             llvm::cl::init(llvm::FloatABI::Default));
