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

#include "glow/LLVMIRCodeGen/CommandLine.h"

llvm::cl::OptionCategory &getLLVMBackendCat() {
  static llvm::cl::OptionCategory cpuBackendCat("Glow CPU Backend Options");
  return cpuBackendCat;
}

llvm::cl::opt<std::string>
    llvmTarget("target", llvm::cl::desc("LLVM target triple to be used"));

llvm::cl::opt<std::string>
    llvmArch("march", llvm::cl::desc("LLVM architecture to be used"));

llvm::cl::opt<std::string> llvmCPU("mcpu",
                                   llvm::cl::desc("LLVM CPU to be used"));

llvm::cl::opt<std::string> llvmABI("mabi",
                                   llvm::cl::desc("Machine ABI to be used"));

llvm::cl::opt<llvm::CodeModel::Model> llvmCodeModel(
    "code-model",
    llvm::cl::desc("Specify which code model to use on the target machine"),
    llvm::cl::values(
        clEnumValN(llvm::CodeModel::Model::Small, "small", "Small code model"),
        clEnumValN(llvm::CodeModel::Model::Medium, "medium",
                   "Medium code model"),
        clEnumValN(llvm::CodeModel::Model::Large, "large", "Large code model")),
    llvm::cl::init(llvm::CodeModel::Model::Large),
    llvm::cl::cat(getLLVMBackendCat()));

llvm::cl::opt<llvm::CodeModel::Model> llvmBundleCodeModel(
    "bundle-code-model",
    llvm::cl::desc("Specify which code model to use for a bundle"),
    llvm::cl::values(
        clEnumValN(llvm::CodeModel::Model::Small, "small", "Small code model"),
        clEnumValN(llvm::CodeModel::Model::Medium, "medium",
                   "Medium code model"),
        clEnumValN(llvm::CodeModel::Model::Large, "large", "Large code model")),
    llvm::cl::init(llvm::CodeModel::Model::Small),
    llvm::cl::cat(getLLVMBackendCat()));

llvm::cl::opt<llvm::Reloc::Model> llvmRelocModel(
    "relocation-model",
    llvm::cl::desc(
        "Specify which relocation model to use on the target machine"),
    llvm::cl::values(
        clEnumValN(llvm::Reloc::Static, "static", "Non-relocatable code"),
        clEnumValN(llvm::Reloc::PIC_, "pic", "Position independent code")),
    llvm::cl::init(llvm::Reloc::Static), llvm::cl::cat(getLLVMBackendCat()));

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

llvm::cl::opt<std::string>
    llvmOpt("llvm-opt",
            llvm::cl::desc("External LLVM-Opt compiler (opt) to use for "
                           "optimizing LLVM bitcode."));

llvm::cl::list<std::string> llvmCompilerOptions(
    "llvm-compiler-opt",
    llvm::cl::desc("Options to pass to the external LLVM compiler"),
    llvm::cl::ZeroOrMore);

llvm::cl::opt<bool> llvmSaveAsm(
    "llvm-save-asm",
    llvm::cl::desc("Create and save asm file along with Bundle object file."),
    llvm::cl::init(false), llvm::cl::cat(getLLVMBackendCat()));

llvm::cl::opt<llvm::FloatABI::ABIType>
    floatABI("float-abi", llvm::cl::desc("Option to set float ABI type"),
             llvm::cl::values(clEnumValN(llvm::FloatABI::Default, "default",
                                         "Default float ABI type"),
                              clEnumValN(llvm::FloatABI::Soft, "soft",
                                         "Soft float ABI (softfp)"),
                              clEnumValN(llvm::FloatABI::Hard, "hard",
                                         "Hard float ABI (hardfp)")),
             llvm::cl::init(llvm::FloatABI::Default));

static llvm::cl::OptionCategory bundleSaverCat("Bundle Options");

llvm::cl::opt<glow::BundleApiType>
    bundleAPI("bundle-api", llvm::cl::desc("Specify which bundle API to use."),
              llvm::cl::Optional,
              llvm::cl::values(clEnumValN(glow::BundleApiType::Dynamic,
                                          "dynamic", "Dynamic API"),
                               clEnumValN(glow::BundleApiType::Static, "static",
                                          "Static API")),
              llvm::cl::init(glow::BundleApiType::Static),
              llvm::cl::cat(bundleSaverCat));

llvm::cl::opt<bool> bundleAPIVerbose(
    "bundle-api-verbose",
    llvm::cl::desc("Print more details in the bundle API header file"),
    llvm::cl::init(false), llvm::cl::cat(bundleSaverCat));

llvm::cl::list<std::string> bundleObjectsOpt(
    "bundle-objects",
    llvm::cl::desc("Comma separated list of names of other object files which "
                   "should be archived into the bundle. The object files are "
                   "pre registered during Glow build. "),
    llvm::cl::CommaSeparated, llvm::cl::ZeroOrMore);
