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

#include "glow/LLVMIRCodeGen/BundleSaver.h"
#include "glow/LLVMIRCodeGen/LLVMBackend.h"

#include "glow/Graph/Graph.h"
#include "glow/Graph/PlaceholderBindings.h"
#include "glow/IR/Instrs.h"
#include "glow/Support/Debug.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <glog/logging.h>

#define DEBUG_TYPE "jit"

using namespace glow;
using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;

/// Header file string template.
static const char *headerFileTemplate =
    R"RAW(%s
#ifndef _GLOW_BUNDLE_%s_H
#define _GLOW_BUNDLE_%s_H

#include <stdint.h>

// ---------------------------------------------------------------
//                       Common definitions
// ---------------------------------------------------------------
#ifndef _GLOW_BUNDLE_COMMON_DEFS
#define _GLOW_BUNDLE_COMMON_DEFS

// Glow bundle error code for correct execution.
#define GLOW_SUCCESS 0
%s
#endif

// ---------------------------------------------------------------
//                          Bundle API
// ---------------------------------------------------------------
%s
// NOTE: Placeholders are allocated within the "mutableWeight"
// buffer and are identified using an offset relative to base.
// ---------------------------------------------------------------
#ifdef __cplusplus
extern "C" {
#endif
%s%s
#ifdef __cplusplus
}
#endif
#endif
)RAW";

/// Function to print the header file using the template.
static void printHeader(llvm::StringRef headerFileName,
                        llvm::StringRef bundleName,
                        llvm::StringRef commonDefines,
                        llvm::StringRef modelInfo, llvm::StringRef modelApi,
                        llvm::StringRef headerExtra) {
  std::error_code EC;
  llvm::raw_fd_ostream headerFile(headerFileName, EC,
                                  llvm::sys::fs::OpenFlags::F_Text);
  CHECK(!EC) << "Could not open header file!";
  std::string header;
  header += "// Bundle API auto-generated header file. Do not edit!\n";
#ifdef GLOW_BUILD_DATE
  header += "// Glow Tools version: " + std::string(GLOW_BUILD_DATE) + "\n";
#endif
  headerFile << strFormat(headerFileTemplate, header.c_str(),
                          bundleName.upper().data(), bundleName.upper().data(),
                          commonDefines.data(), modelInfo.data(),
                          modelApi.data(), headerExtra.data());
  headerFile.close();
}

/// Header file common definitions for dynamic API.
static const char *dynamicApiCommonDefines = R"RAW(
// Type describing a symbol table entry of a generated bundle.
struct SymbolTableEntry {
  // Name of a variable.
  const char *name;
  // Offset of the variable inside the memory area.
  uint64_t offset;
  // The number of elements inside this variable.
  uint64_t size;
  // Variable kind: 1 if it is a mutable variable, 0 otherwise.
  char kind;
};

// Type describing the config of a generated bundle.
struct BundleConfig {
  // Size of the constant weight variables memory area.
  uint64_t constantWeightVarsMemSize;
  // Size of the mutable weight variables memory area.
  uint64_t mutableWeightVarsMemSize;
  // Size of the activations memory area.
  uint64_t activationsMemSize;
  // Alignment to be used for weights and activations.
  uint64_t alignment;
  // Number of symbols in the symbol table.
  uint64_t numSymbols;
  // Symbol table.
  const SymbolTableEntry *symbolTable;
};
)RAW";

/// Header file common definitions for static API.
static const char *staticApiCommonDefines = R"RAW(
// Memory alignment definition with given alignment size
// for static allocation of memory.
#define GLOW_MEM_ALIGN(size)  __attribute__((aligned(size)))

// Macro function to get the absolute address of a
// placeholder using the base address of the mutable
// weight buffer and placeholder offset definition.
#define GLOW_GET_ADDR(mutableBaseAddr, placeholderOff)  (((uint8_t*)(mutableBaseAddr)) + placeholderOff)
)RAW";

/// Utility function to serialize a binary file to text file as a C array.
static void serializeBinaryToText(llvm::StringRef binFileName,
                                  llvm::StringRef txtFileName) {
  FILE *inpFile = fopen(binFileName.str().c_str(), "rb");
  CHECK(inpFile) << "Could not open binary input file: " << binFileName.str();
  FILE *outFile = fopen(txtFileName.str().c_str(), "w");
  CHECK(outFile) << "Could not open text output file: " << txtFileName.str();
  const size_t numBytesPerLine = 20;
  for (size_t i = 0;; i++) {
    int ch = fgetc(inpFile);
    if (ch == EOF) {
      break;
    }
    fprintf(outFile, " 0X%02X,", ch);
    if ((i % numBytesPerLine) == (numBytesPerLine - 1)) {
      fprintf(outFile, "\n");
    }
  }
  fprintf(outFile, "\n");
  fclose(inpFile);
  fclose(outFile);
}

BundleSaver::BundleSaver(const LLVMBackend &llvmBackend,
                         llvm::StringRef outputDir, llvm::StringRef bundleName)
    : irgen_(llvmBackend.createIRGen(nullptr, allocationsInfo_)),
      bundleAPI_(llvmBackend.getOptions().getBundleAPI()) {
  llvm::SmallVector<std::string, 8> targetFeatures(llvmTargetFeatures.begin(),
                                                   llvmTargetFeatures.end());
  irgen_->setBundleName(bundleName);
  irgen_->setOutputDir(outputDir);
  // Use the bundle code model as a code model for the TargetMachine.
  auto opts = llvmBackend.getOptions();
  opts.setCodeModel(opts.getBundleCodeModel());
  irgen_->initTargetMachine(opts);
  irgen_->initCodeGen();
}

void BundleSaver::setIRFunction(llvm::StringRef mainEntryName,
                                const IRFunction *F) {
  irgen_->setIRFunction(F);
  if (F) {
    savedIRFunctions_.push_back(SavedIRFunction{mainEntryName, F});
  }
}

bool BundleSaver::WeightAddrComparator::
operator()(const WeightInfo &LHS, const WeightInfo &RHS) const {
  auto lhsAddr =
      bundleSaver_->allocationsInfo_.allocatedAddress_.lookup(LHS.first);
  auto rhsAddr =
      bundleSaver_->allocationsInfo_.allocatedAddress_.lookup(RHS.first);
  return lhsAddr < rhsAddr;
}

std::set<BundleSaver::WeightInfo, BundleSaver::WeightAddrComparator>
BundleSaver::findConstantWeights() const {
  std::set<BundleSaver::WeightInfo, BundleSaver::WeightAddrComparator>
      constants(WeightAddrComparator(*const_cast<BundleSaver *>(this)));
  for (auto &savedIRFunction : savedIRFunctions_) {
    for (auto *c : savedIRFunction.savedF->findConstants()) {
      auto *w = cast<WeightVar>(savedIRFunction.savedF->getWeightForNode(c));
      constants.insert({w, c});
    }
  }
  return constants;
}

std::set<const Placeholder *> BundleSaver::findPlaceholders() const {
  std::set<const Placeholder *> placeholders;
  for (auto &savedIRFunction : savedIRFunctions_) {
    for (auto *ph : savedIRFunction.savedF->findPlaceholders()) {
      placeholders.insert(ph);
    }
  }
  return placeholders;
}

Value *BundleSaver::getWeightForNode(const Storage *V) const {
  for (auto &savedIRFunction : savedIRFunctions_) {
    if (auto *W = savedIRFunction.savedF->getWeightForNode(V)) {
      return W;
    }
  }
  return nullptr;
}

void BundleSaver::saveWeights(llvm::StringRef weightsFileName) {
  std::error_code EC;
  llvm::raw_fd_ostream weightsFile(weightsFileName, EC, llvm::sys::fs::F_None);
  CHECK(!EC) << "Could not open the output file for saving the bundle weights "
                "with file name: "
             << weightsFileName.str();
  // Serialize only constant weights.
  // Do not serialize mutable weights representing inputs and outputs, because
  // it should be configurable and set by the client.
  size_t pos = 0;
  size_t maxPos = 0;
  for (auto &weightInfo : findConstantWeights()) {
    auto *w = weightInfo.first;
    auto *c = weightInfo.second;
    auto numBytes = w->getSizeInBytes();
    auto payload = c->getPayload().getUnsafePtr();
    auto addr = allocationsInfo_.allocatedAddress_[weightInfo.first];
    if (addr < pos) {
      // The payload was written already. It aliases something we have seen
      // already.
      continue;
    }
    weightsFile.seek(addr);
    CHECK(!weightsFile.has_error()) << "Could not set file write position";
    weightsFile.write(payload, numBytes);
    CHECK(!weightsFile.has_error()) << "Could not write bytes";
    pos = addr + numBytes;
    maxPos = std::max(pos, maxPos);
  }
  // Make sure that the file is as long as the constantWeightVarsMemSize_.
  // This is needed to properly handle alignments.
  weightsFile.seek(maxPos);
  for (size_t endPos = irgen_->getAllocationsInfo().constantWeightVarsMemSize_;
       maxPos < endPos; maxPos++) {
    weightsFile.write(0);
  }
  weightsFile.close();
}

void BundleSaver::saveHeader(llvm::StringRef headerFileName) {
  auto bundleName = irgen_->getBundleName();
  auto bundleNameUpper = llvm::StringRef(bundleName).upper();
  auto constMemSize = irgen_->getAllocationsInfo().constantWeightVarsMemSize_;
  auto mutableMemSize = irgen_->getAllocationsInfo().mutableWeightVarsMemSize_;
  auto activationsMemSize = irgen_->getAllocationsInfo().activationsMemSize_;
  auto memAlignSize = TensorAlignment;
  auto totMemSize = constMemSize + mutableMemSize + activationsMemSize;

  // Format common bundle definitions.
  auto commonDefines = (bundleAPI_ == BundleApiType::Dynamic)
                           ? dynamicApiCommonDefines
                           : staticApiCommonDefines;

  // Format model description.
  std::string modelInfo = strFormat("// Model name: \"%s\"\n"
                                    "// Total data size: %lu (bytes)\n",
                                    bundleName.data(), totMemSize);
  // Print placeholders (mandatory).
  modelInfo += "// Placeholders:\n";
  auto placeholders = findPlaceholders();
  for (auto &v : placeholders) {
    auto *w = cast<WeightVar>(getWeightForNode(v));
    // Get placeholder properties.
    auto name = w->getName();
    auto type = w->getType();
    auto typeName = type->toString();
    auto sizeElem = type->size();
    auto sizeByte = type->getSizeInBytes();
    auto offset = allocationsInfo_.allocatedAddress_[w];
    modelInfo += strFormat("//\n"
                           "//   Name: \"%s\"\n"
                           "//   Type: %s\n"
                           "//   Size: %" PRIuDIM " (elements)\n"
                           "//   Size: %zu (bytes)\n"
                           "//   Offset: %lu (bytes)\n",
                           name.data(), typeName.c_str(), sizeElem, sizeByte,
                           (unsigned long)offset);
  }
  // Print constants (optional).
  if (bundleAPIVerbose) {
    modelInfo += "//\n"
                 "// Constants:\n";
    auto constantWeights = findConstantWeights();
    for (auto &weightInfo : constantWeights) {
      auto *w = weightInfo.first;
      // Get constant properties.
      auto name = w->getName();
      auto type = w->getType();
      auto typeName = type->toString();
      auto sizeElem = type->size();
      auto sizeByte = type->getSizeInBytes();
      auto offset = allocationsInfo_.allocatedAddress_[w];
      modelInfo += strFormat("//\n"
                             "//   Name: \"%s\"\n"
                             "//   Type: %s\n"
                             "//   Size: %" PRIuDIM " (elements)\n"
                             "//   Size: %zu (bytes)\n"
                             "//   Offset: %lu (bytes)\n",
                             name.data(), typeName.c_str(), sizeElem, sizeByte,
                             (unsigned long)offset);
    }
  }
  modelInfo += "//";

  std::string modelApi = "\n";
  if (bundleAPI_ == BundleApiType::Dynamic) {
    // Print bundle memory configuration.
    modelApi += strFormat("// Bundle memory configuration (memory layout).\n"
                          "extern BundleConfig %s_config;\n"
                          "\n",
                          bundleName.data());

  } else {
    // Get placeholder names and offsets. Compute also the maximum placeholder
    // name length for print purposes.
    unsigned nameMaxLen = 0;
    std::vector<std::pair<llvm::StringRef, unsigned>> nameAddrPairs;
    for (auto &v : placeholders) {
      auto *w = cast<WeightVar>(getWeightForNode(v));
      auto name = w->getName();
      auto addr = allocationsInfo_.allocatedAddress_[w];
      nameMaxLen = name.size() > nameMaxLen ? name.size() : nameMaxLen;
      nameAddrPairs.push_back(std::pair<llvm::StringRef, unsigned>(name, addr));
    }

    // Print placeholder address offsets.
    modelApi +=
        "// Placeholder address offsets within mutable buffer (bytes).\n";
    for (auto &pair : nameAddrPairs) {
      modelApi += strFormat(
          "#define %s_%s%s  %u\n", bundleNameUpper.data(), pair.first.data(),
          std::string(nameMaxLen - pair.first.size(), ' ').c_str(),
          pair.second);
    }
    modelApi += "\n";

    // Print memory sizes and memory alignment.
    modelApi +=
        strFormat("// Memory sizes (bytes).\n"
                  "#define %s_CONSTANT_MEM_SIZE     %lu\n"
                  "#define %s_MUTABLE_MEM_SIZE      %lu\n"
                  "#define %s_ACTIVATIONS_MEM_SIZE  %lu\n"
                  "\n"
                  "// Memory alignment (bytes).\n"
                  "#define %s_MEM_ALIGN  %d\n"
                  "\n",
                  bundleNameUpper.data(), constMemSize, bundleNameUpper.data(),
                  mutableMemSize, bundleNameUpper.data(), activationsMemSize,
                  bundleNameUpper.data(), memAlignSize);
  }

  // Print bundle entry functions.
  for (auto &savedIRFunction : savedIRFunctions_) {
    modelApi +=
        strFormat("// Bundle entry point (inference function). Returns 0\n"
                  "// for correct execution or some error code otherwise.\n"
                  "int %s("
                  "uint8_t *constantWeight, "
                  "uint8_t *mutableWeight, "
                  "uint8_t *activations"
                  ");\n",
                  savedIRFunction.entryName.c_str());
  }

  // Get bundle header extra content.
  std::string headerExtra = irgen_->getBundleHeaderExtra();

  // Print header file.
  printHeader(headerFileName, bundleName, commonDefines, modelInfo, modelApi,
              headerExtra);
}

void BundleSaver::emitSymbolTable() {
  // Define a struct for symbol table entries:
  // struct SymbolTableEntry {
  //  const char *name;
  //  uint64_t offset;
  //  uint64_t size;
  //  char kind;
  // };
  auto *charTy = llvm::Type::getInt8Ty(irgen_->getLLVMContext());
  auto *uint64TTy =
      llvm::Type::getIntNTy(irgen_->getLLVMContext(), sizeof(uint64_t) * 8);
  auto symbolTableEntryTy = llvm::StructType::get(
      irgen_->getLLVMContext(),
      {charTy->getPointerTo(), uint64TTy, uint64TTy, charTy});
  // Set of entries in the symbol table.
  llvm::SmallVector<llvm::Constant *, 128> entries;
  // Iterate over all Placeholders and record information about their names,
  // offset, size and kind.
  for (auto &v : findPlaceholders()) {
    auto *w = cast<WeightVar>(getWeightForNode(v));
    auto size = w->getType()->size();
    auto addr = allocationsInfo_.allocatedAddress_[w];
    // Create an SymbolTableEntry.
    auto *entry = llvm::ConstantStruct::get(
        symbolTableEntryTy,
        {// name.
         dyn_cast<llvm::Constant>(irgen_->getBuilder().CreateBitCast(
             irgen_->emitStringConst(irgen_->getBuilder(), w->getName()),
             charTy->getPointerTo())),
         // offset.
         llvm::ConstantInt::get(uint64TTy, addr),
         // size.
         llvm::ConstantInt::get(uint64TTy, size),
         // 1 for Mutable Kind
         llvm::ConstantInt::get(charTy, 1)});
    entries.push_back(entry);
  }

  // Create a constant array with these entries.
  auto *arr = llvm::ConstantArray::get(
      llvm::ArrayType::get(symbolTableEntryTy, entries.size()), entries);
  // Create a global variable and initialize it with the constructed array.
  new llvm::GlobalVariable(irgen_->getModule(), arr->getType(), true,
                           llvm::GlobalValue::InternalLinkage, arr,
                           irgen_->getBundleName() + "SymbolTable");
}

void BundleSaver::produceBundle() {
  DCHECK(!isSaved_) << "produceBundle can be invoked only once";
  isSaved_ = true;
  // Emit entry functions.
  for (auto &savedFunction : savedIRFunctions_) {
    emitBundleEntryFunction(savedFunction);
  }
  // Finish code generation.
  irgen_->finishCodeGen();
  setIRFunction("<noname>", nullptr);
  // Emit symbol table and bundle config only for dynamic API
  if (bundleAPI_ == BundleApiType::Dynamic) {
    // Emit the symbol table for weight variables.
    emitSymbolTable();
    // Emit the config for the bundle.
    emitBundleConfig();
  }

  auto &M = irgen_->getModule();
  auto outputDir = irgen_->getOutputDir();
  auto bundleName = irgen_->getBundleName();
  std::string extension = (llvmCompiler.empty()) ? ".o" : ".bc";
  auto bundleCodeOutput = (outputDir + "/" + bundleName + extension).str();
  auto bundleWeightsBinOut =
      (outputDir + "/" + bundleName + ".weights.bin").str();
  auto bundleHeaderOutput = (outputDir + "/" + bundleName + ".h").str();
  DEBUG_GLOW(llvm::dbgs() << "Producing a bundle:\n"
                          << "bundle name: " << bundleName << "\n"
                          << "bundle code: " << bundleCodeOutput << "\n"
                          << "bundle weights:" << bundleWeightsBinOut << "\n"
                          << "header file: " << bundleHeaderOutput << "\n");
  llvm::StringRef fileName = bundleCodeOutput;
  std::error_code EC;
  llvm::raw_fd_ostream outputFile(fileName, EC, llvm::sys::fs::F_None);
  CHECK(!EC) << "Could not open the output file for saving the bundle "
                "code with file name: "
             << fileName.str();
  if (fileName.endswith(".bc")) {
    // Emit the bitcode file.
    llvm::WriteBitcodeToFile(M, outputFile);
    outputFile.flush();
    if (!llvmCompiler.empty()) {
      // Compile bitcode using an external LLVM compiler.
      std::string cmd = llvmCompiler;
      for (auto option : llvmCompilerOptions) {
        cmd += " " + option + " ";
      }
      cmd += " " + bundleCodeOutput;
      std::string bundleObjectCodeOutputOpt =
          " -o " + (outputDir + "/" + bundleName + ".o").str();
      cmd += bundleObjectCodeOutputOpt;
      CHECK(!system(cmd.c_str()))
          << "Error running external LLVM compiler: " << cmd;
    }
  } else if (fileName.endswith(".o")) {
    // Emit the object file.
    llvm::legacy::PassManager PM;
    auto &TM = irgen_->getTargetMachine();

#if FACEBOOK_INTERNAL && LLVM_VERSION_MAJOR < 8
    TM.addPassesToEmitFile(
        PM, outputFile, llvm::TargetMachine::CodeGenFileType::CGFT_ObjectFile);
#elif LLVM_VERSION_MAJOR < 10
    TM.addPassesToEmitFile(
        PM, outputFile, nullptr,
        llvm::TargetMachine::CodeGenFileType::CGFT_ObjectFile);
#else
    TM.addPassesToEmitFile(PM, outputFile, nullptr, llvm::CGFT_ObjectFile);
#endif

    PM.run(M);
  }
  outputFile.close();
  // Output weights.
  saveWeights(bundleWeightsBinOut);
  // Header file.
  saveHeader(bundleHeaderOutput);
  // Save weights also in text format for Static API.
  if (bundleAPI_ == BundleApiType::Static) {
    auto bundleWeightsTxtOut =
        (outputDir + "/" + bundleName + ".weights.txt").str();
    serializeBinaryToText(bundleWeightsBinOut, bundleWeightsTxtOut);
  }
}

/// Emit the entry function for the bundle. It simply calls the main entry of
/// the module and forwards its arguments to it. As the last argument it
/// provides the constant array of offsets. Since these offsets are constants,
/// the LLVM optimizer will constant propagate them into relative addressing
/// computations and the like and produce a very efficient code that uses
/// absolute addressing whenever possible.
void BundleSaver::emitBundleEntryFunction(
    BundleSaver::SavedIRFunction &savedF) {
  // The bundle entry point has the following API:
  // int entry(uint8_t *constantWeight,
  //           uint8_t *mutableWeight,
  //           uint8_t *activations);
  auto int8PtrTy = llvm::Type::getInt8PtrTy(irgen_->getLLVMContext());
  llvm::Type *retTy = llvm::Type::getIntNTy(irgen_->getLLVMContext(),
                                            irgen_->getLibjitIntWidth());
  llvm::FunctionType *bundleFuncTy =
      llvm::FunctionType::get(retTy, {int8PtrTy, int8PtrTy, int8PtrTy}, false);
  auto *func =
      llvm::Function::Create(bundleFuncTy, llvm::Function::ExternalLinkage,
                             savedF.entryName, &irgen_->getModule());
  llvm::BasicBlock *entry_bb =
      llvm::BasicBlock::Create(irgen_->getLLVMContext(), "entry", func);
  llvm::IRBuilder<> builder(entry_bb);
  // Add a provisional terminator to make the function well-formed.
  auto *zero = builder.getIntN(irgen_->getLibjitIntWidth(), 0);
  auto *ret = builder.CreateRet(zero);
  builder.SetInsertPoint(ret);

  // Prepare arguments for the "main" function.
  llvm::SmallVector<llvm::Value *, 4> initFunctionCallArgs;
  initFunctionCallArgs.push_back(func->args().begin());
  initFunctionCallArgs.push_back(func->args().begin() + 1);
  initFunctionCallArgs.push_back(func->args().begin() + 2);
  // Now form the offsets array and pass it as the last argument.
  auto offsetsArray = irgen_->emitConstOffsetsArray(builder, allocationsInfo_);
  initFunctionCallArgs.push_back(offsetsArray);
  // Invoke the main entry with constant arguments and let LLVM optimizer make
  // use of it.
  auto *entryF = savedF.llvmF;
  entryF->setLinkage(llvm::Function::InternalLinkage);
  auto *result = irgen_->createCall(builder, entryF, initFunctionCallArgs);
  // Terminate the function.
  builder.CreateRet(result);
  // Remove the provisional terminator.
  ret->eraseFromParent();
  // Create the debug info for the bundle entry point function.
  irgen_->generateFunctionDebugInfo(func);
}

// Create a config for this network. It will be exposed to the clients,
// so that they know how much memory they need to allocate, etc.
// Config consists of the following fields:
// struct BundleConfig {
//   uint64_t constantWeightVarsMemSize;
//   uint64_t mutableWeightVarsMemSize;
//   uint64_t activationsMemSize;
//   uint64_t alignment;
//   uint64_t numSymbols;
//   SymbolTableEntry *symbolTable;
// };
void BundleSaver::emitBundleConfig() {
  auto symbolTableName = irgen_->getBundleName().str() + "SymbolTable";
  auto symbolTable =
      irgen_->getModule().getGlobalVariable(symbolTableName, true);
  CHECK(symbolTable)
      << "Expected to find a symbol table for the AOT bundle with name: "
      << symbolTableName;
  // Get the integer type having the same size in bits as uint64_t.
  auto *uint64TType = irgen_->getBuilder().getIntNTy(sizeof(uint64_t) * 8);
  auto symbolTableEntryTy = symbolTable->getType()->getPointerElementType();
  auto *bundleConfigTy =
      llvm::StructType::get(irgen_->getLLVMContext(),
                            {uint64TType, uint64TType, uint64TType, uint64TType,
                             uint64TType, symbolTableEntryTy->getPointerTo()});
  auto config = new llvm::GlobalVariable(
      irgen_->getModule(), bundleConfigTy, /* isConst */ true,
      llvm::GlobalValue::LinkageTypes::ExternalLinkage, nullptr,
      irgen_->getBundleName().str() + "_config");
  config->setInitializer(llvm::ConstantStruct::get(
      bundleConfigTy,
      llvm::ConstantInt::get(
          uint64TType, irgen_->getAllocationsInfo().constantWeightVarsMemSize_),
      llvm::ConstantInt::get(
          uint64TType, irgen_->getAllocationsInfo().mutableWeightVarsMemSize_),
      llvm::ConstantInt::get(uint64TType,
                             irgen_->getAllocationsInfo().activationsMemSize_),

      llvm::ConstantInt::get(uint64TType, TensorAlignment),
      llvm::ConstantInt::get(uint64TType, findPlaceholders().size()),

      symbolTable));
}

void BundleSaver::performBundleMemoryAllocation() {
  // Perform memory allocation for the current function.
  auto *F = savedIRFunctions_.back().savedF;
  allocationsInfo_.numberValues(F);
  allocationsInfo_.allocateActivations(F);
  // Tell the allocateWeightVars to not reuse any existing addresses for weights
  // and to assign new ones.
  allocationsInfo_.allocateWeightVars(F);
  allocationsInfo_.allocateTensorViews(F);
}

void BundleSaver::save(llvm::StringRef mainEntryName, const IRFunction *F) {
  // Object files generation works properly only in small mode.
  irgen_->setMainEntryName(mainEntryName);
  // Set current IRFunction using the legalized name.
  setIRFunction(irgen_->getMainEntryName(), F);
  // irgen_->initCodeGen();
  // Perform the address assignment for activations and WeightVars.
  performBundleMemoryAllocation();
  // Emit the code for the body of the entry function.
  irgen_->performCodeGen();
  savedIRFunctions_.back().llvmF = irgen_->getLLVMFunction();
}
