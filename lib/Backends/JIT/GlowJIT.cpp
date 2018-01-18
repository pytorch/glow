// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "GlowJIT.h"

using GlowJIT = llvm::orc::GlowJIT;

/// Generate the LLVM MAttr list of attributes.
static llvm::SmallVector<std::string, 0> getMachineAttributes() {
  llvm::SmallVector<std::string, 0> result;
  llvm::StringMap<bool> hostFeatures;
  if (llvm::sys::getHostCPUFeatures(hostFeatures)) {
    for (auto &feature : hostFeatures) {
      if (feature.second) {
        llvm::StringRef fn = feature.first();
        // Skip avx512 because LLVM does not support it well.
        if (fn.startswith("avx512")) {
          continue;
        }
        result.push_back(fn);
      }
    }
  }
  return result;
}

/// Returns the CPU hostname.
static llvm::StringRef getHostCpuName() {
  auto cpu_name = llvm::sys::getHostCPUName();
  // Skip avx512 because LLVM does not support it well.
  cpu_name.consume_back("-avx512");
  return cpu_name;
}

GlowJIT::GlowJIT()
    : TM(EngineBuilder().selectTarget(llvm::Triple(), "", getHostCpuName(),
                                      getMachineAttributes())),
      DL(TM->createDataLayout()),
      ObjectLayer([]() { return std::make_shared<SectionMemoryManager>(); }),
      CompileLayer(ObjectLayer, SimpleCompiler(*TM)) {
  llvm::sys::DynamicLibrary::LoadLibraryPermanently(nullptr);
}

GlowJIT::ModuleHandle GlowJIT::addModule(std::unique_ptr<Module> M) {
  // Build our symbol resolver:
  // Lambda 1: Look back into the JIT itself to find symbols that are part of
  //           the same "logical dylib".
  // Lambda 2: Search for external symbols in the host process.
  auto Resolver = createLambdaResolver(
      [&](const std::string &Name) {
        if (auto Sym = CompileLayer.findSymbol(Name, false))
          return Sym;
        return JITSymbol(nullptr);
      },
      [](const std::string &Name) {
        if (auto SymAddr = RTDyldMemoryManager::getSymbolAddressInProcess(Name))
          return JITSymbol(SymAddr, JITSymbolFlags::Exported);
        return JITSymbol(nullptr);
      });

  // Add the set to the JIT with the resolver we created above and a newly
  // created SectionMemoryManager.
  return cantFail(CompileLayer.addModule(std::move(M), std::move(Resolver)));
}

llvm::JITSymbol GlowJIT::findSymbol(const std::string Name) {
  std::string MangledName;
  raw_string_ostream MangledNameStream(MangledName);
  Mangler::getNameWithPrefix(MangledNameStream, Name, DL);
  return CompileLayer.findSymbol(MangledNameStream.str(), true);
}

void GlowJIT::removeModule(GlowJIT::ModuleHandle H) {
  cantFail(CompileLayer.removeModule(H));
}
