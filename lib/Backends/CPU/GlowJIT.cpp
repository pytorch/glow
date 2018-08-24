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
#include "llvm/ExecutionEngine/JITEventListener.h"
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

#if LLVM_VERSION_MAJOR <= 6
/// This is a callback that is invoked when an LLVM module is compiled and
/// loaded by the JIT for execution.
class NotifyLoadedFunctor {
  /// The listener for debugger events. It is used to provide debuggers with the
  /// information about JITted code.
  llvm::JITEventListener *dbgRegistrationListener_;
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
  NotifyLoadedFunctor(GlowJIT *jit)
      : dbgRegistrationListener_(
            llvm::JITEventListener::createGDBRegistrationListener()) {}
  void operator()(llvm::orc::RTDyldObjectLinkingLayerBase::ObjHandleT,
                  const llvm::orc::RTDyldObjectLinkingLayerBase::ObjectPtr &obj,
                  const llvm::RuntimeDyld::LoadedObjectInfo &objInfo) {
    auto loadedObj = obj->getBinary();
    // Inform the debugger about the loaded object file. This should allow for
    // more complete stack traces under debugger. And even it should even enable
    // the stepping functionality on platforms supporting it.
    dbgRegistrationListener_->NotifyObjectEmitted(*loadedObj, objInfo);
    // Dump symbol information for the JITed symbols.
    dumpSymbolInfo(*loadedObj, objInfo);
  }
};
#endif

} // namespace

GlowJIT::GlowJIT(llvm::TargetMachine &TM)
    : TM_(TM), DL_(TM_.createDataLayout()),
#if LLVM_VERSION_MAJOR > 6
      SSP_(new SymbolStringPool()),
      ES_(SSP_),
      resolver_(createLegacyLookupResolver(
          ES_,
          [this](const std::string &Name) -> JITSymbol {
            if (auto Sym = compileLayer_.findSymbol(Name, false))
              return Sym;
            else if (auto Err = Sym.takeError())
              return std::move(Err);
            if (auto SymAddr =
                    RTDyldMemoryManager::getSymbolAddressInProcess(Name))
              return JITSymbol(SymAddr, JITSymbolFlags::Exported);
            return nullptr;
          },
          [](Error Err) { cantFail(std::move(Err), "lookupFlags failed"); })),
      objectLayer_(ES_,
                   [this](llvm::orc::VModuleKey) {
                     return RTDyldObjectLinkingLayer::Resources{
                         std::make_shared<SectionMemoryManager>(), resolver_};
                   }),
#else
      objectLayer_([]() { return std::make_shared<SectionMemoryManager>(); },
                   NotifyLoadedFunctor(this)),
#endif
      compileLayer_(objectLayer_, SimpleCompiler(TM_)) {
  llvm::sys::DynamicLibrary::LoadLibraryPermanently(nullptr);
}

GlowJIT::ModuleHandle GlowJIT::addModule(std::unique_ptr<llvm::Module> M) {
// Add the set to the JIT with the resolver and a newly created
// SectionMemoryManager.
#if LLVM_VERSION_MAJOR > 6
  auto K = ES_.allocateVModule();
  cantFail(compileLayer_.addModule(K, std::move(M)));
  return K;
#else
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

  return cantFail(compileLayer_.addModule(std::move(M), std::move(resolver)));
#endif
}

void GlowJIT::removeModule(GlowJIT::ModuleHandle H) {
  cantFail(compileLayer_.removeModule(H));
}

llvm::JITSymbol GlowJIT::findSymbol(const std::string name) {
  std::string mangledName;
  raw_string_ostream MangledNameStream(mangledName);
  Mangler::getNameWithPrefix(MangledNameStream, name, DL_);
  return compileLayer_.findSymbol(MangledNameStream.str(), true);
}
