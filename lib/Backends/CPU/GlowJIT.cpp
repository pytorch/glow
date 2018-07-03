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

#include "GlowJIT.h"
#include "CommandLine.h"
#include "llvm/Object/SymbolSize.h"

using GlowJIT = llvm::orc::GlowJIT;

namespace {
/// An option to enabling the dump of the symbol information for the JITted
/// functions. It dumps e.g. the names of the functions, their start addresses
/// and their end addresses.
static llvm::cl::opt<bool> dumpJITSymbolInfo(
    "dump-jit-symbol-info",
    llvm::cl::desc("Dump the load addresses and sizes of JITted symbols"),
    llvm::cl::init(false), llvm::cl::cat(CPUBackendCat));

/// This is a callback that is invoked when an LLVM module is compiled and
/// loaded by the JIT for execution.
class NotifyLoadedFunctor {
  /// JIT object whose events are received by this listener.
  GlowJIT *jit_;
  /// Dump symbol information for symbols defined by the object file.
  void dumpSymbolInfo(const llvm::object::ObjectFile &loadedObj,
                      const llvm::RuntimeDyld::LoadedObjectInfo &objInfo) {
    if (!dumpJITSymbolInfo)
      return;
    // Dump information about symbols.
    for (auto symSizePair : llvm::object::computeSymbolSizes(loadedObj)) {
      auto sym = symSizePair.first;
      auto size = symSizePair.second;
      auto symName = sym.getName();
      // Skip any unnamed symbols.
      if (!symName || symName->empty())
        continue;
      // The relative address of the symbol inside its section.
      auto symAddr = sym.getAddress();
      if (!symAddr)
        continue;
      // The address the functions was loaded at.
      auto loadedSymAddress = *symAddr;
      auto symbolSection = sym.getSection();
      if (symbolSection) {
        // Compute the load address of the symbol by adding the section load
        // address.
        loadedSymAddress += objInfo.getSectionLoadAddress(*symbolSection.get());
      }
      llvm::outs() << llvm::format("Address range: [%12p, %12p]",
                                   loadedSymAddress, loadedSymAddress + size)
                   << "\tSymbol: " << *symName << "\n";
    }
  }

public:
  NotifyLoadedFunctor(GlowJIT *jit) : jit_(jit) {}
  void operator()(llvm::orc::RTDyldObjectLinkingLayerBase::ObjHandleT,
                  const llvm::orc::RTDyldObjectLinkingLayerBase::ObjectPtr &obj,
                  const llvm::RuntimeDyld::LoadedObjectInfo &objInfo) {
    auto loadedObj = obj->getBinary();
    // Dump symbol information for the JITed symbols.
    dumpSymbolInfo(*loadedObj, objInfo);
  }
};
} // namespace

GlowJIT::GlowJIT(llvm::TargetMachine &TM)
    : TM_(TM), DL_(TM_.createDataLayout()),
      objectLayer_([]() { return std::make_shared<SectionMemoryManager>(); },
                   NotifyLoadedFunctor(this)),
      compileLayer_(objectLayer_, SimpleCompiler(TM)) {
  llvm::sys::DynamicLibrary::LoadLibraryPermanently(nullptr);
}

GlowJIT::ModuleHandle GlowJIT::addModule(std::unique_ptr<Module> M) {
  // Build our symbol resolver:
  // Lambda 1: Look back into the JIT itself to find symbols that are part of
  //           the same "logical dylib".
  // Lambda 2: Search for external symbols in the host process.
  auto resolver = createLambdaResolver(
      [&](const std::string &name) {
        if (auto sym = compileLayer_.findSymbol(name, false))
          return sym;
        return JITSymbol(nullptr);
      },
      [](const std::string &name) {
        if (auto symAddr = RTDyldMemoryManager::getSymbolAddressInProcess(name))
          return JITSymbol(symAddr, JITSymbolFlags::Exported);
        return JITSymbol(nullptr);
      });

  // Add the set to the JIT with the resolver we created above and a newly
  // created SectionMemoryManager.
  return cantFail(compileLayer_.addModule(std::move(M), std::move(resolver)));
}

llvm::JITSymbol GlowJIT::findSymbol(const std::string name) {
  std::string mangledName;
  raw_string_ostream MangledNameStream(mangledName);
  Mangler::getNameWithPrefix(MangledNameStream, name, DL_);
  return compileLayer_.findSymbol(MangledNameStream.str(), true);
}

void GlowJIT::removeModule(GlowJIT::ModuleHandle H) {
  cantFail(compileLayer_.removeModule(H));
}
