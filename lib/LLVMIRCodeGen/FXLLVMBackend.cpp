// (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

#include "glow/Graph/FXIRWrapper.h"
#include "glow/LLVMIRCodeGen/LLVMBackend.h"
#include "glow/Optimizer/IROptimizer/IROptimizer.h"

using namespace glow;

Expected<std::unique_ptr<CompiledFunction>>
LLVMBackend::compileFX(const folly::dynamic &FXIR, const std::string &submod,
                       const llvm::StringMap<const void *> &constants,
                       const BackendOptions &opts, Module *glowModule) const {
  // TODO: Add support for traceInfo for FXIR.
  // TraceInfo traceInfo = buildManualTraceInfo(F);
  auto fXIRWrapper =
      glow::make_unique<FXIRWrapper>(FXIR, submod, constants, glowModule);
  auto IR =
      generateAndOptimizeIR(fXIRWrapper.get(), *this, shouldShareBuffers());

  // autoInstrument invokes addMetadataPlaceholder which is not supported by
  // FXIRWrapper now.
  if (opts.autoInstrument) {
    // TODO: Add support for autoInstrument for FXIR.
    // autoInstrument(traceInfo, IR.get());
  }

  std::unique_ptr<CompiledFunction> compiledFunc;
  if (opts.collectConstants) {
    compiledFunc = compileIR(std::move(IR));
  } else {
    compiledFunc = compileIRWithoutConstants(IR.get());
  }

  // TODO: Add support for traceInfo for FXIR.
  // compiledFunc->setTraceInfo(std::move(traceInfo));
  return Expected<std::unique_ptr<CompiledFunction>>(std::move(compiledFunc));
}
