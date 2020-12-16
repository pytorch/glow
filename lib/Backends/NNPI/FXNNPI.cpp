// Copyright 2004-present Facebook. All Rights Reserved.

#include "glow/lib/Backends/NNPI/NNPI.h"
#include "glow/lib/Backends/NNPI/NNPICompiledFunction.h"
#include <folly/dynamic.h>

using namespace glow;

Expected<std::unique_ptr<CompiledFunction>>
NNPIBackend::compileFX(const folly::dynamic &FXIR, const std::string &submod,
                       const llvm::StringMap<const void *> &constants,
                       const BackendOptions &opts, Module *glowModule) const {
  auto compiledFunc = std::make_unique<NNPICompiledFunction>(
      FXIR, submod, constants, glowModule);
  auto compileHasError = compiledFunc->compileFX(FXIR, submod, constants, opts);

  return compileHasError ? std::move(compileHasError)
                         : Expected<std::unique_ptr<CompiledFunction>>(
                               std::move(compiledFunc));
}
