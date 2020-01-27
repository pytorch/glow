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

#include "glow/LLVMIRCodeGen/GlowJIT.h"
#include "glow/Support/Debug.h"

#include "llvm/ExecutionEngine/JITEventListener.h"
#include "llvm/Object/SymbolSize.h"

#define DEBUG_TYPE "jit-engine"

using GlowJIT = llvm::orc::GlowJIT;

namespace {
/// An option to enabling the dump of the symbol information for the JITted
/// functions. It dumps e.g. the names of the functions, their start addresses
/// and their end addresses.
static llvm::cl::opt<bool> dumpJITSymbolInfo(
    "dump-jit-symbol-info",
    llvm::cl::desc("Dump the load addresses and sizes of JITted symbols"),
    llvm::cl::init(false), llvm::cl::cat(getLLVMBackendCat()));

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

  void operator()(llvm::orc::VModuleKey key,
                  const llvm::object::ObjectFile &obj,
                  const llvm::RuntimeDyld::LoadedObjectInfo &objInfo) {
    auto &loadedObj = obj;
    // Inform the debugger about the loaded object file. This should allow for
    // more complete stack traces under debugger. And even it should even enable
    // the stepping functionality on platforms supporting it.
#if LLVM_VERSION_MAJOR == 7 || FACEBOOK_INTERNAL
    // This fails sometimes with the following assertion:
    // lib/ExecutionEngine/GDBRegistrationListener.cpp:168: virtual void
    // {anonymous}::GDBJITRegistrationListener::NotifyObjectEmitted(const
    // llvm::object::ObjectFile&, const llvm::RuntimeDyld::LoadedObjectInfo&):
    // Assertion `ObjectBufferMap.find(Key) == ObjectBufferMap.end() && "Second
    // attempt to perform debug registration."' failed.
    // dbgRegistrationListener_->NotifyObjectEmitted(loadedObj, objInfo);
#else
    dbgRegistrationListener_->notifyObjectLoaded(
        (llvm::JITEventListener::ObjectKey)&loadedObj, loadedObj, objInfo);
#endif

    // Dump symbol information for the JITed symbols.
    dumpSymbolInfo(loadedObj, objInfo);
  }
};

} // namespace

GlowJIT::GlowJIT(llvm::TargetMachine &TM)
    : TM_(TM), DL_(TM_.createDataLayout()),
#if FACEBOOK_INTERNAL && LLVM_VERSION_MAJOR < 8
      ES_(SSP_),
      resolver_(createLegacyLookupResolver(
          [this](const std::string &Name) -> JITSymbol {
            // Search for symbols which may not be exported.  On PE/COFF targets
            // (i.e. Windows), not all symbols are implicitly exported.  If the
            // symbols is not marked as DLLExport, it is not considered
            // exported, and the symbol lookup may fail.  This may also occur on
            // ELF/MachO targets if built with hidden visibility.  The JIT
            // however maintains a list of all symbols and can find unexported
            // symbols as well.
            if (auto Sym = compileLayer_.findSymbol(
                    Name, /*ExportedSymbolsOnly=*/false))
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
      SSP_(std::make_shared<SymbolStringPool>()), ES_(SSP_),
      cxxSymbolOverride_(
          [this](const std::string &name) { return mangle(name); }),
      resolver_(createLegacyLookupResolver(
          ES_,
          [this](const std::string &name) -> JITSymbol {
            if (auto localSym = compileLayer_.findSymbol(name, false)) {
              return localSym;
            } else if (auto Err = localSym.takeError()) {
              return std::move(Err);
            }
            // Some symbols are overridden, in particular __dso_handle and
            // __cxa_atexit .
            if (auto overriddenSym = cxxSymbolOverride_.searchOverrides(name)) {
              return overriddenSym;
            }
            // FIXME: looking for symbols external to libjit in the process is
            // dangerous because it can be environment dependent. For example,
            // we get cases where a symbol is found in the Linux environment,
            // but not in the Windows environment.
            if (auto processSymAddr =
                    RTDyldMemoryManager::getSymbolAddressInProcess(name)) {
              return JITSymbol(processSymAddr, JITSymbolFlags::Exported);
            }
            // The symbol was not resolved. This will make the retreival of
            // 'main' function symbol fail later without much information about
            // the source of the problem. Then, we dump an error message now to
            // ease debugging.
            DEBUG_GLOW(llvm::dbgs()
                       << "JIT: Error resolving symbol '" << name << "'\n");
            // Return a 'symbol not found' JITSymbol object (nullptr).
            return nullptr;
          },
          [](Error Err) { cantFail(std::move(Err), "lookupFlags failed"); })),
#if LLVM_VERSION_MAJOR == 7 || FACEBOOK_INTERNAL
      objectLayer_(ES_,
                   [this](llvm::orc::VModuleKey) {
                     return RTDyldObjectLinkingLayer::Resources{
                         std::make_shared<SectionMemoryManager>(), resolver_};
                   },
                   NotifyLoadedFunctor(this)),
#else
      objectLayer_(ES_,
                   [this](llvm::orc::VModuleKey) {
                     return LegacyRTDyldObjectLinkingLayer::Resources{
                         std::make_shared<SectionMemoryManager>(), resolver_};
                   },
                   NotifyLoadedFunctor(this)),
#endif
#endif
      compileLayer_(objectLayer_, SimpleCompiler(TM_)) {
  //  When passing a null pointer to LoadLibraryPermanently, we request to
  //  'load' the host process itself, making its exported symbols available for
  //  execution.
  llvm::sys::DynamicLibrary::LoadLibraryPermanently(nullptr);
}

GlowJIT::~GlowJIT() {
  // Run any destructor registered with __cxa_atexit.
  cxxSymbolOverride_.runDestructors();
  // Run any destructor discovered in the LLVM IR of the JIT modules.
  for (auto &dtorRunner : irStaticDestructorRunners_) {
    cantFail(dtorRunner.runViaLayer(compileLayer_));
  }
}

GlowJIT::ModuleHandle GlowJIT::addModule(std::unique_ptr<llvm::Module> M) {
  // Add the set to the JIT with the resolver and a newly created
  // SectionMemoryManager.

  auto K = ES_.allocateVModule();

  // Record the static constructors and destructors. We have to do this before
  // we hand over ownership of the module to the JIT.
  // Note: This code is based on the LLI/OrcLazyJIT LLVM tool code that is based
  // on the ORCv1 API (see
  // https://github.com/llvm-mirror/llvm/blob/release_60/tools/lli/OrcLazyJIT.cpp)
  // In recent LLVM versions (7+), LLJIT uses the newer ORCv2 API (see
  // https://github.com/llvm-mirror/llvm/blob/release_70/lib/ExecutionEngine/Orc/LLJIT.cpp).
  std::vector<std::string> ctorNames, dtorNames;
  for (auto ctor : orc::getConstructors(*M))
    ctorNames.push_back(mangle(ctor.Func->getName()));
  for (auto dtor : orc::getDestructors(*M))
    dtorNames.push_back(mangle(dtor.Func->getName()));

  cantFail(compileLayer_.addModule(K, std::move(M)));

#if LLVM_VERSION_MAJOR == 7 || FACEBOOK_INTERNAL
  CtorDtorRunner<decltype(compileLayer_)> ctorRunner(std::move(ctorNames), K);
#else
  LegacyCtorDtorRunner<decltype(compileLayer_)> ctorRunner(std::move(ctorNames),
                                                           K);
#endif

  // Run the static constructors and register static destructors.
  consumeError(ctorRunner.runViaLayer(compileLayer_));
  irStaticDestructorRunners_.emplace_back(std::move(dtorNames), K);

  return K;
}

void GlowJIT::removeModule(GlowJIT::ModuleHandle H) {
  cantFail(compileLayer_.removeModule(H));
}

std::string GlowJIT::mangle(const std::string &name) {
  std::string mangledName;
  raw_string_ostream MangledNameStream(mangledName);
  Mangler::getNameWithPrefix(MangledNameStream, name, DL_);
  return MangledNameStream.str();
}

llvm::JITSymbol GlowJIT::findSymbol(const std::string &name) {
  return compileLayer_.findSymbol(mangle(name), false);
}
