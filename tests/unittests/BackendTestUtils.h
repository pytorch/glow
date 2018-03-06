// Copyright 2017-2018 Facebook. All Rights Reserved.

#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Graph/Graph.h"
#include "glow/IR/IR.h"

namespace glow {

void inferBatchedAddNet(Tensor *inputs1, Tensor *inputs2, Tensor *out,
                        BackendKind kind);

void inferBatchedReduceAddNet(Tensor *inputs, Tensor *out, BackendKind kind);

void inferConvNet(Tensor *inputs, Tensor *kernel, Tensor *bias, Tensor *out,
                  BackendKind kind);

void trainConvNet(Tensor *inputs, Tensor *kernel1, Tensor *bias1,
                  Tensor *kernel2, Tensor *bias2, Tensor *selected,
                  llvm::ArrayRef<size_t> shape1, llvm::ArrayRef<size_t> shape2,
                  Tensor *out, BackendKind kind);

void inferLocalResponseNormalizationNet(Tensor *inputs, Tensor *out,
                                        BackendKind kind);

void trainLocalResponseNormalizationNet(Tensor *inputs, Tensor *weights,
                                        Tensor *bias, Tensor *selected,
                                        llvm::ArrayRef<size_t> shape1,
                                        llvm::ArrayRef<size_t> shape2,
                                        Tensor *out, BackendKind kind);

void inferMaxNet(Tensor *inputs1, Tensor *inputs2, Tensor *out,
                 BackendKind kind);

void inferMinNet(Tensor *inputs1, Tensor *inputs2, Tensor *out,
                 BackendKind kind);

void inferPoolAvgNet(Tensor *inputs, Tensor *out, BackendKind kind);

void trainPoolAvgNet(Tensor *inputs, Tensor *weights, Tensor *bias,
                     Tensor *selected, llvm::ArrayRef<size_t> shape1,
                     llvm::ArrayRef<size_t> shape2, Tensor *out,
                     BackendKind kind);

void inferPoolMaxNet(Tensor *inputs, Tensor *out, BackendKind kind);

void trainPoolMaxNet(Tensor *inputs, Tensor *weights, Tensor *bias,
                     Tensor *selected, llvm::ArrayRef<size_t> shape1,
                     llvm::ArrayRef<size_t> shape2, Tensor *out,
                     BackendKind kind);

void inferQuantizeNet(Tensor *inputs, float scale, int32_t offset, Tensor *out,
                      BackendKind kind);

void inferReluNet(Tensor *inputs, Tensor *out, BackendKind kind);

void inferReshapeNet(Tensor *inputs, llvm::ArrayRef<size_t> shape, Tensor *out,
                     BackendKind kind);

void inferSelectNet(Tensor *cond, Tensor *inputs1, Tensor *inputs2, Tensor *out,
                    BackendKind kind);

void inferSigmoidNet(Tensor *inputs, Tensor *out, BackendKind kind);

void inferSoftMaxNet(Tensor *inputs, Tensor *selected, Tensor *out,
                     BackendKind kind);

void trainSoftMaxNet(Tensor *inputs, Tensor *weights, Tensor *bias,
                     Tensor *selected, Tensor *out, BackendKind kind);

void inferTanhNet(Tensor *inputs, Tensor *out, BackendKind kind);

void inferBasicConvNet(Tensor *inputs, Tensor *out, BackendKind kind);

void inferBasicFCNet(Tensor *inputs, Tensor *out, BackendKind kind);

void inferMixedNet(Tensor *inputs, Tensor *out, BackendKind kind);

void inferComplexNet1(Tensor *inputs1, Tensor *inputs2, Tensor *inputs3,
                      Tensor *inputs4, Tensor *out, BackendKind kind);

} // namespace glow
