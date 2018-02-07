#ifndef GLOW_BACKENDS_JIT_GLOWJIT_H
#define GLOW_BACKENDS_JIT_GLOWJIT_H

#include "llvm/ADT/STLExtras.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/LambdaResolver.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
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

namespace llvm {
namespace orc {

// A class that represents a simple LLVM-based Orc JIT. Based on the
// KaleidoscopeJIT example in the LLVM tree.
class GlowJIT {
private:
  std::unique_ptr<TargetMachine> TM_;
  const DataLayout DL_;
  RTDyldObjectLinkingLayer objectLayer_;
  IRCompileLayer<decltype(objectLayer_), SimpleCompiler> compileLayer_;

public:
  using ModuleHandle = decltype(compileLayer_)::ModuleHandleT;

  GlowJIT();

  TargetMachine &getTargetMachine() { return *TM_; }

  ModuleHandle addModule(std::unique_ptr<Module> M);

  JITSymbol findSymbol(const std::string name);

  void removeModule(ModuleHandle H);
};

} // end namespace orc
} // end namespace llvm

#endif // GLOW_BACKENDS_JIT_GLOWJIT_H
