/**
 * Copyright (c) 2017-present, Facebook, Inc.
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

#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Graph/Graph.h"
#include "glow/IR/IR.h"

namespace glow {

/// MockBackend used only for unit testing.
class MockBackend : public Backend {
  class MockFunction : public CompiledFunction {
    void execute() override {}
  };
  std::unique_ptr<CompiledFunction>
  compile(std::unique_ptr<IRFunction> IR) const override {
    return llvm::make_unique<MockFunction>();
  }
  bool isOpSupported(Kinded::Kind opKind, ElemKind elementTy) const override {
    return false;
  }
};

void inferBatchedAddNet(Tensor *inputs1, Tensor *inputs2, Tensor *out,
                        BackendKind kind);

void inferBatchedReduceAddNet(Tensor *inputs, Tensor *out, BackendKind kind);

void inferConvNet(Tensor *inputs, Tensor *filter, Tensor *bias, Tensor *out,
                  BackendKind kind);

void trainConvNet(Tensor *inputs, Tensor *kernel1, Tensor *bias1,
                  Tensor *kernel2, Tensor *bias2, Tensor *selected,
                  llvm::ArrayRef<size_t> shape1, llvm::ArrayRef<size_t> shape2,
                  Tensor *out, BackendKind kind);

void inferGatherNet(Tensor *data, Tensor *indices, Tensor *dest,
                    BackendKind kind);

void inferLocalResponseNormalizationNet(Tensor *inputs, Tensor *out,
                                        BackendKind kind);

void trainLocalResponseNormalizationNet(Tensor *inputs, Tensor *weights,
                                        Tensor *bias, Tensor *selected,
                                        llvm::ArrayRef<size_t> shape1,
                                        llvm::ArrayRef<size_t> shape2,
                                        Tensor *out, BackendKind kind);

void inferMatMulNet(Tensor *lhs, Tensor *rhs, Tensor *out, BackendKind kind);

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

void inferIntLookupTableNet(Tensor *input, Tensor *out,
                            llvm::ArrayRef<int8_t> table, BackendKind kind);

void inferQuantizeNet(Tensor *inputs, float scale, int32_t offset, Tensor *out,
                      BackendKind kind);

void inferReluNet(Tensor *inputs, Tensor *out, BackendKind kind);

void inferReshapeNet(Tensor *inputs, llvm::ArrayRef<size_t> shape, Tensor *out,
                     BackendKind kind);

void inferSelectNet(Tensor *cond, Tensor *inputs1, Tensor *inputs2, Tensor *out,
                    BackendKind kind);

void inferSigmoidNet(Tensor *inputs, Tensor *out, BackendKind kind);

void inferGroupConv(Tensor *out, BackendKind kind);

void inferNonSquarePaddingConv(Tensor *out, BackendKind kind);

void inferConvDKKC8(Tensor *out, BackendKind kind);

void inferSmallConv(Tensor *inputs, Tensor *out, BackendKind kind);

void inferSoftMaxNet(Tensor *inputs, Tensor *selected, Tensor *out,
                     BackendKind kind);

void trainSoftMaxNet(Tensor *inputs, Tensor *weights, Tensor *bias,
                     Tensor *selected, Tensor *out, BackendKind kind);

void inferTanhNet(Tensor *inputs, Tensor *out, BackendKind kind);

void inferTransposeNet(Tensor *inputs, Tensor *out, BackendKind kind);

void inferBasicConvNet(Tensor *inputs, Tensor *out, BackendKind kind,
                       size_t convDepth);

void inferTanhConcatNet(Tensor *input1, Tensor *input2, Tensor *input3,
                        Tensor *out, BackendKind kind);

void inferBasicFCNet(Tensor *inputs, Tensor *out, BackendKind kind);

void inferMixedNet(Tensor *inputs, Tensor *out, BackendKind kind);

void inferComplexNet1(Tensor *inputs1, Tensor *inputs2, Tensor *inputs3,
                      Tensor *inputs4, Tensor *out, BackendKind kind);

void inferTinyResnet(Tensor *input, Tensor *out, std::vector<Tensor> &weights,
                     BackendKind kind);

void inferExtract3D(Tensor *input, Tensor *out, BackendKind kind);

} // namespace glow
