// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "GlowJIT.h"

using GlowJIT = llvm::orc::GlowJIT;

GlowJIT::GlowJIT(llvm::TargetMachine &TM)
    : TM_(TM), DL_(TM_.createDataLayout()),
      objectLayer_([]() { return std::make_shared<SectionMemoryManager>(); }),
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
