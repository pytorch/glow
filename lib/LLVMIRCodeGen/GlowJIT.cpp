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
#if LLVM_VERSION_MAJOR == 7 || LLVM_VERSION_MAJOR == 10 ||                     \
    (LLVM_VERSION_MAJOR <= 8 && FACEBOOK_INTERNAL)
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

//==============================================================================
#if LLVM_VERSION_MAJOR < 8 && FACEBOOK_INTERNAL
//==============================================================================
llvm::JITSymbol GlowJIT::resolveSymbol(const std::string &name) {
  // Search for symbols which may not be exported.  On PE/COFF targets
  // (i.e. Windows), not all symbols are implicitly exported.  If the
  // symbols is not marked as DLLExport, it is not considered
  // exported, and the symbol lookup may fail.  This may also occur on
  // ELF/MachO targets if built with hidden visibility.  The JIT
  // however maintains a list of all symbols and can find unexported
  // symbols as well.
  if (auto Sym = compileLayer_.findSymbol(Name, /*ExportedSymbolsOnly=*/false))
    return Sym;
  else if (auto Err = Sym.takeError())
    return std::move(Err);
  if (auto SymAddr = RTDyldMemoryManager::getSymbolAddressInProcess(Name))
    return JITSymbol(SymAddr, JITSymbolFlags::Exported);
  return nullptr;
}

template <typename LegacyLookupFn>
static std::shared_ptr<llvm::orc::LegacyLookupFnResolver<LegacyLookupFn>>
createLookupResolver(llvm::orc::ExecutionSession &, LegacyLookupFn LegacyLookup,
                     std::function<void(llvm::Error)> ErrorReporter) {
  return createLegacyLookupResolver(std::move(LegacyLookup),
                                    std::move(ErrorReporter));
}

//==============================================================================
#elif LLVM_VERSION_MAJOR < 8
//==============================================================================
llvm::JITSymbol GlowJIT::resolveSymbol(const std::string &name) {
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
  DEBUG_GLOW(llvm::dbgs() << "JIT: Error resolving symbol '" << name << "'\n");
  // Return a 'symbol not found' JITSymbol object (nullptr).
  return nullptr;
}

template <typename LegacyLookupFn>
static std::shared_ptr<llvm::orc::LegacyLookupFnResolver<LegacyLookupFn>>
createLookupResolver(llvm::orc::ExecutionSession &ES,
                     LegacyLookupFn LegacyLookup,
                     std::function<void(llvm::Error)> ErrorReporter) {
  return createLegacyLookupResolver(ES, std::move(LegacyLookup),
                                    std::move(ErrorReporter));
}

//==============================================================================
#else // 8 <= LLVM_VERSION_MAJOR
//==============================================================================
static bool symbolFound(llvm::JITSymbol &s) {
  const llvm::JITSymbolFlags flags = s.getFlags();
  if (flags.getRawFlagsValue() || flags.getTargetFlags()) {
    return true;
  }

  llvm::Expected<llvm::JITTargetAddress> expAddr = s.getAddress();
  if (!expAddr) {
    return false; // should never get here since no flags are set
  }

  return expAddr.get() != 0;
}

llvm::JITSymbol GlowJIT::resolveSymbol(const std::string &name) {

  // Search accross all modules for a strong symbol. If no strong symbol is
  // found, return the first matching weak symbol found if any.
  bool weakFound = false;
  JITSymbol firstWeak(nullptr);
  for (auto k : vModKeys_) {
    JITSymbol localSym = compileLayer_.findSymbolIn(k, name, false);
    if (auto Err = localSym.takeError()) {
      return std::move(Err);
    }

    if (!symbolFound(localSym)) {
      continue;
    }

    JITSymbolFlags flags = localSym.getFlags();
    if (flags.isStrong()) {
      return localSym;
    }

    // This is a matching weak or common symbol. Remember the first one we find
    // in case we don't find a subsequent strong one.
    if (!weakFound) {
      firstWeak = std::move(localSym);
      weakFound = true;
    }
  }

#if !FACEBOOK_INTERNAL
  // Some symbols are overridden, in particular __dso_handle and
  // __cxa_atexit .
  if (auto overriddenSym = cxxSymbolOverride_.searchOverrides(name)) {
    return overriddenSym;
  }
#endif
  // FIXME: looking for symbols external to libjit in the process is
  // dangerous because it can be environment dependent. For example,
  // we get cases where a symbol is found in the Linux environment,
  // but not in the Windows environment.
  if (auto processSymAddr =
          RTDyldMemoryManager::getSymbolAddressInProcess(name)) {
    return JITSymbol(processSymAddr, JITSymbolFlags::Exported);
  }

  // No strong symbol found. Return a weak symbol if we found one.
  if (weakFound) {
    return firstWeak;
  }

  // The symbol was not resolved. This will make the retreival of
  // 'main' function symbol fail later without much information about
  // the source of the problem. Then, we dump an error message now to
  // ease debugging.
  DEBUG_GLOW(llvm::dbgs() << "JIT: Error resolving symbol '" << name << "'\n");
  // Return a 'symbol not found' JITSymbol object (nullptr).
  return nullptr;
}

namespace {
// In order to work around a bug in the llvm-provided
// 'getResponsibilitySetWithLegacyFn' involving the handling of weak symbols, we
// provide our own implementation, called indirectly through this implementation
// of the 'llvm::orc::SymbolResolver' interface.
template <typename LegacyLookupFn>
class LookupFnResolver final : public llvm::orc::SymbolResolver {
private:
  using Error = llvm::Error;
  using ErrorReporter = std::function<void(Error)>;
  using SymbolNameSet = llvm::orc::SymbolNameSet;
  using AsynchronousSymbolQuery = llvm::orc::AsynchronousSymbolQuery;
  using ExecutionSession = llvm::orc::ExecutionSession;
  using JITSymbol = llvm::JITSymbol;
  using JITSymbolFlags = llvm::JITSymbolFlags;
  using JITTargetAddress = llvm::JITTargetAddress;

  ExecutionSession &ES;
  LegacyLookupFn LegacyLookup;
  ErrorReporter ReportError;

  llvm::Expected<SymbolNameSet>
  getResponsibilitySetWithLegacyFn(const SymbolNameSet &Symbols) {
    SymbolNameSet Result;

    for (auto &S : Symbols) {
      // Note that we don't use Sym's operator bool() here since that returns
      // false for symbols with no address (which includes weak symbols).
      JITSymbol Sym = LegacyLookup(*S);
      if (auto Err = Sym.takeError()) {
        return std::move(Err);
      }
      if (!Sym.getFlags().isStrong()) {
        Result.insert(S);
      }
    }

    return Result;
  }

public:
  LookupFnResolver(ExecutionSession &ES, LegacyLookupFn LegacyLookup,
                   ErrorReporter ReportError)
      : ES(ES), LegacyLookup(std::move(LegacyLookup)),
        ReportError(std::move(ReportError)) {}

  SymbolNameSet lookup(std::shared_ptr<AsynchronousSymbolQuery> Query,
                       SymbolNameSet Symbols) final {
    return llvm::orc::lookupWithLegacyFn(ES, *Query, Symbols, LegacyLookup);
  }

  SymbolNameSet getResponsibilitySet(const SymbolNameSet &Symbols) final {
    auto ResponsibilitySet = getResponsibilitySetWithLegacyFn(Symbols);

    if (ResponsibilitySet) {
      return std::move(*ResponsibilitySet);
    }

    ReportError(ResponsibilitySet.takeError());
    return SymbolNameSet();
  }
};
} // namespace

template <typename LegacyLookupFn>
static std::shared_ptr<LookupFnResolver<LegacyLookupFn>>
createLookupResolver(llvm::orc::ExecutionSession &ES,
                     LegacyLookupFn LegacyLookup,
                     std::function<void(llvm::Error)> ErrorReporter) {
  return std::make_shared<LookupFnResolver<LegacyLookupFn>>(
      ES, std::move(LegacyLookup), std::move(ErrorReporter));
}
#endif

GlowJIT::GlowJIT(llvm::TargetMachine &TM)
    : TM_(TM), DL_(TM_.createDataLayout()),
#if FACEBOOK_INTERNAL && LLVM_VERSION_MAJOR < 8
      ES_(SSP_),
      resolver_(createLookupResolver(
          ES_,
          [this](const std::string &Name) -> JITSymbol {
            return this->resolveSymbol(Name);
          },
          [](Error Err) { cantFail(std::move(Err), "lookupFlags failed"); })),
      objectLayer_(ES_,
                   [this](llvm::orc::VModuleKey) {
                     return RTDyldObjectLinkingLayer::Resources{
                         std::make_shared<SectionMemoryManager>(), resolver_};
                   }),
#else
      SSP_(std::make_shared<SymbolStringPool>()), ES_(SSP_),
#if !FACEBOOK_INTERNAL
      cxxSymbolOverride_(
          [this](const std::string &name) { return mangle(name); }),
#endif
      resolver_(createLookupResolver(
          ES_,
          [this](const std::string &name) -> JITSymbol {
            return this->resolveSymbol(name);
          },
          [](Error Err) { cantFail(std::move(Err), "lookupFlags failed"); })),
#if LLVM_VERSION_MAJOR == 7 || (LLVM_VERSION_MAJOR <= 8 && FACEBOOK_INTERNAL)
      objectLayer_(
          ES_,
          [this](llvm::orc::VModuleKey) {
            return RTDyldObjectLinkingLayer::Resources{
                std::make_shared<SectionMemoryManager>(), resolver_};
          },
          NotifyLoadedFunctor(this)),
#else
      objectLayer_(
          ES_,
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
#if !FACEBOOK_INTERNAL
  // Run any destructor registered with __cxa_atexit.
  cxxSymbolOverride_.runDestructors();
#endif
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
  vModKeys_.insert(K);

#if LLVM_VERSION_MAJOR == 7 || (LLVM_VERSION_MAJOR <= 8 && FACEBOOK_INTERNAL)
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
  vModKeys_.erase(H);
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
