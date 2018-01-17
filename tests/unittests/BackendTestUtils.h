// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Graph/Graph.h"
#include "glow/IR/IR.h"

void inferReluNet(glow::Tensor *inputs, glow::Tensor *out,
                  glow::BackendKind kind);

void inferBasicConvNet(glow::Tensor *inputs, glow::Tensor *out,
                       glow::BackendKind kind);

void inferBasicFCNet(glow::Tensor *inputs, glow::Tensor *out,
                     glow::BackendKind kind);

void inferMixedNet(glow::Tensor *inputs, glow::Tensor *out,
                   glow::BackendKind kind);
