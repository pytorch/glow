// Copyright 2017-2018 Facebook. All Rights Reserved.

#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Graph/Graph.h"
#include "glow/IR/IR.h"

void inferBatchedAddNet(glow::Tensor *inputs1, glow::Tensor *inputs2,
                        glow::Tensor *out, glow::BackendKind kind);

void inferBatchedReduceAddNet(glow::Tensor *inputs, glow::Tensor *out,
                              glow::BackendKind kind);

void inferMaxNet(glow::Tensor *inputs1, glow::Tensor *inputs2,
                 glow::Tensor *out, glow::BackendKind kind);

void inferMinNet(glow::Tensor *inputs1, glow::Tensor *inputs2,
                 glow::Tensor *out, glow::BackendKind kind);

void inferPoolAvgNet(glow::Tensor *inputs, glow::Tensor *out,
                     glow::BackendKind kind);

void trainPoolAvgNet(glow::Tensor *inputs, glow::Tensor *selected,
                     llvm::ArrayRef<size_t> shape, glow::Tensor *out,
                     glow::BackendKind kind);

void inferReluNet(glow::Tensor *inputs, glow::Tensor *out,
                  glow::BackendKind kind);

void inferReshapeNet(glow::Tensor *inputs, llvm::ArrayRef<size_t> shape,
                     glow::Tensor *out, glow::BackendKind kind);

void inferSelectNet(glow::Tensor *cond, glow::Tensor *inputs1,
                    glow::Tensor *inputs2, glow::Tensor *out,
                    glow::BackendKind kind);

void inferSigmoidNet(glow::Tensor *inputs, glow::Tensor *out,
                     glow::BackendKind kind);

void inferSoftMaxNet(glow::Tensor *inputs, glow::Tensor *selected,
                     glow::Tensor *out, glow::BackendKind kind);

void trainSoftMaxNet(glow::Tensor *inputs, glow::Tensor *selected,
                     glow::Tensor *out, glow::BackendKind kind);

void inferTanhNet(glow::Tensor *inputs, glow::Tensor *out,
                  glow::BackendKind kind);

void inferBasicConvNet(glow::Tensor *inputs, glow::Tensor *out,
                       glow::BackendKind kind);

void inferBasicFCNet(glow::Tensor *inputs, glow::Tensor *out,
                     glow::BackendKind kind);

void inferMixedNet(glow::Tensor *inputs, glow::Tensor *out,
                   glow::BackendKind kind);

void inferComplexNet1(glow::Tensor *inputs1, glow::Tensor *inputs2,
                      glow::Tensor *inputs3, glow::Tensor *inputs4,
                      glow::Tensor *out, glow::BackendKind kind);
