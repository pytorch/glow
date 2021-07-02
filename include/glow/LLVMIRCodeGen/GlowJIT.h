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
#ifndef GLOW_LLVMIRCODEGEN_GLOWJIT_H
#define GLOW_LLVMIRCODEGEN_GLOWJIT_H

#include "llvm/ADT/STLExtras.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/Orc/SymbolStringPool.h"
#include "llvm/ExecutionEngine/RTDyldMemoryManager.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Mangler.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#ifndef GLOW_JIT_ORC_VERSION
#if LLVM_VERSION_MAJOR < 11
#define GLOW_JIT_ORC_VERSION 1
#else
#define GLOW_JIT_ORC_VERSION 2
#endif
#endif

//##############################################################################
#if GLOW_JIT_ORC_VERSION == 1
//##############################################################################
namespace llvm {
namespace orc {

// A class that represents a simple LLVM-based Orc JIT. Based on the
// KaleidoscopeJIT example in the LLVM tree.
class GlowJIT {
private:
  std::unique_ptr<llvm::TargetMachine> TM_;
  const DataLayout DL_;
  std::unique_ptr<llvm::LLVMContext> ctx_;
#if FACEBOOK_INTERNAL && LLVM_VERSION_MAJOR < 8
  SymbolStringPool SSP_;
  ExecutionSession ES_;
  std::shared_ptr<SymbolResolver> resolver_;
#else
  std::shared_ptr<SymbolStringPool> SSP_;
  ExecutionSession ES_;
  /// Handles symbols that are overridden by the JIT engine (needed to manage
  /// C++ destructors for static objects).
#if !FACEBOOK_INTERNAL
#if LLVM_VERSION_MAJOR == 7 || FACEBOOK_INTERNAL
  LocalCXXRuntimeOverrides cxxSymbolOverride_;
#else
  LegacyLocalCXXRuntimeOverrides cxxSymbolOverride_;
#endif
#endif

  /// Set of VModuleKeys for all modules in the CompileLayer. Needed in order to
  /// iterate over available modules in some versions of resolveSymbol.
  llvm::DenseSet<llvm::orc::VModuleKey> vModKeys_;

  std::shared_ptr<SymbolResolver> resolver_;
#endif
#if LLVM_VERSION_MAJOR == 7 || (LLVM_VERSION_MAJOR <= 8 && FACEBOOK_INTERNAL)
  RTDyldObjectLinkingLayer objectLayer_;
  IRCompileLayer<decltype(objectLayer_), SimpleCompiler> compileLayer_;
  /// Records C++ constructor/destructor names of static objects.
  std::vector<llvm::orc::CtorDtorRunner<decltype(compileLayer_)>>
      irStaticDestructorRunners_;
#else
  LegacyRTDyldObjectLinkingLayer objectLayer_;
  LegacyIRCompileLayer<decltype(objectLayer_), SimpleCompiler> compileLayer_;
  /// Object that records static C++ constructor/destructor names.
  std::vector<LegacyCtorDtorRunner<decltype(compileLayer_)>>
      irStaticDestructorRunners_;
#endif

  /// \returns the JITSymbol for the symbol named \p name. Differs from
  /// findSymbol in that it handles resolving symbols outside of the
  /// CompileLayer (e.g. in process symbols, overridden symbols).
  /// Called indirectly by the ObjectLinkingLayer.
  JITSymbol resolveSymbol(const std::string &name);

  /// \returns the mangled name for the C++ global symbol \p name.
  std::string mangle(const std::string &name);

public:
  GlowJIT(std::unique_ptr<llvm::TargetMachine> TM);
  ~GlowJIT();

  TargetMachine &getTargetMachine() { return *TM_; }

  JITSymbol findSymbol(const std::string &name);

  using ModuleHandle = orc::VModuleKey;

  ModuleHandle addModule(std::unique_ptr<Module> M);

  void removeModule(ModuleHandle H);

  void setContext(std::unique_ptr<llvm::LLVMContext> ctx);
};

} // end namespace orc
} // end namespace llvm

namespace glow {
using GlowJIT = llvm::orc::GlowJIT;
}

//##############################################################################
#elif GLOW_JIT_ORC_VERSION == 2
//##############################################################################

namespace glow {

class GlowJITOrcV2 {
  friend class GlowJITDefGenerator;

  std::unique_ptr<llvm::TargetMachine> tm_;
  const llvm::DataLayout dl_;
  std::shared_ptr<llvm::orc::SymbolStringPool> ssp_;
  llvm::orc::ExecutionSession es_;
  llvm::orc::JITDylib &jd_;

  llvm::orc::RTDyldObjectLinkingLayer objectLayer_;
  llvm::orc::IRCompileLayer compileLayer_;
  llvm::orc::ThreadSafeContext ctx_;

  llvm::orc::MangleAndInterner mangler_;

  /// Handles symbols that are overridden by the JIT engine (needed to manage
  /// C++ destructors for static objects).
  llvm::orc::LocalCXXRuntimeOverrides cxxSymbolOverride_;

  /// Object that records static C++ constructor/destructor names.
  std::vector<llvm::orc::CtorDtorRunner> irStaticDestructorRunners_;

  /// \returns the JITSymbol for the symbol named \p name. Differs from
  /// findSymbol in that it handles resolving symbols outside of the
  /// CompileLayer (e.g. in process symbols, overridden symbols).
  /// Called indirectly by the ObjectLinkingLayer.
  llvm::Error tryToGenerate(llvm::orc::LookupKind K, llvm::orc::JITDylib &JD,
                            llvm::orc::JITDylibLookupFlags JDLookupFlags,
                            const llvm::orc::SymbolLookupSet &LookupSet);

public:
  GlowJITOrcV2(std::unique_ptr<llvm::TargetMachine> tm);
  virtual ~GlowJITOrcV2();

  llvm::JITSymbol findSymbol(const std::string &name);

  void setContext(std::unique_ptr<llvm::LLVMContext> ctx);
  void addModule(std::unique_ptr<llvm::Module> M);
};
using GlowJIT = GlowJITOrcV2;

} // namespace glow
#else
#error Unsupported GLOW_JIT_ORC_VERSION
#endif

#endif // GLOW_LLVMIRCODEGEN_GLOWJIT_H
