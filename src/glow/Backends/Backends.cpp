// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "Interpreter/Interpreter.h"

#include "glow/Graph/Graph.h"
#include "glow/Graph/Nodes.h"
#include "glow/IR/Instrs.h"
#include "glow/Optimizer/Optimizer.h"

#include "llvm/Support/Casting.h"

using namespace glow;

Backend *glow::createBackend(BackendKind backendKind, Module *M) {
  switch (backendKind) {
  case BackendKind::Interpreter:
    return createInterpreter(M);

  default:
    // Unknown execution backend.
    glow_unreachable();
    break;
  }
}
