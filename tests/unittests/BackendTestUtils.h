// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Graph/Graph.h"
#include "glow/IR/IR.h"

void inferMaxNet(glow::Tensor *inputs1, glow::Tensor *inputs2,
                 glow::Tensor *out, glow::BackendKind kind);

void inferMinNet(glow::Tensor *inputs1, glow::Tensor *inputs2,
                 glow::Tensor *out, glow::BackendKind kind);

void inferReluNet(glow::Tensor *inputs, glow::Tensor *out,
                  glow::BackendKind kind);

void inferReshapeNet(glow::Tensor *inputs, llvm::ArrayRef<size_t> shape,
                     glow::Tensor *out, glow::BackendKind kind);

void inferSelectNet(glow::Tensor *cond, glow::Tensor *inputs1,
                    glow::Tensor *inputs2, glow::Tensor *out,
                    glow::BackendKind kind);

void inferSigmoidNet(glow::Tensor *inputs, glow::Tensor *out,
                     glow::BackendKind kind);

void inferTanhNet(glow::Tensor *inputs, glow::Tensor *out,
                  glow::BackendKind kind);

void inferBasicConvNet(glow::Tensor *inputs, glow::Tensor *out,
                       glow::BackendKind kind);

void inferBasicFCNet(glow::Tensor *inputs, glow::Tensor *out,
                     glow::BackendKind kind);

void inferMixedNet(glow::Tensor *inputs, glow::Tensor *out,
                   glow::BackendKind kind);
