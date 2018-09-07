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
#include "glow/IR/IRBuilder.h"
#include "glow/IR/Instrs.h"

namespace glow {

using llvm::cast;

#define VarFrom(T)                                                             \
  mod.createVariable(&T->getType(), #T, VisibilityKind::Public, false)

void inferBatchedAddNet(Tensor *batch, Tensor *slice, Tensor *out,
                        BackendKind kind) {
  ExecutionEngine EE(kind);
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  Variable *batchVar;
  Variable *sliceVar;
  Variable *outVar;
  TypeRef OT;
  if (batch->getType().isQuantizedType()) {
    auto &outType = out->getType();
    batchVar = VarFrom(batch);
    sliceVar = VarFrom(slice);
    outVar = VarFrom(out);
    OT = F->getParent()->uniqueType(out->getElementType(), out->dims(),
                                    outType.getScale(), outType.getOffset());
  } else {
    batchVar = VarFrom(batch);
    sliceVar = VarFrom(slice);
    outVar = VarFrom(out);
    OT = F->getParent()->uniqueType(out->getElementType(), out->dims());
  }
  auto *batchedadd = F->createBatchedAdd("batchedadd", OT, batchVar, sliceVar);
  auto result = F->createSave("ret", batchedadd, outVar);
  EE.compile(CompilationMode::Infer, F);
  updateVariables({batchVar, sliceVar}, {batch, slice});
  EE.run();

  out->assign(&result->getVariable()->getPayload());
}

void inferBatchedReduceAddNet(Tensor *inputs, Tensor *out, BackendKind kind) {
  ExecutionEngine EE(kind);
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  auto *var = VarFrom(inputs);
  auto *batchedreduce =
      F->createBatchedReduceAdd("batchedreduce", var, /* axis */ 0);
  auto result = F->createSave("ret", batchedreduce);
  EE.compile(CompilationMode::Infer, F);
  updateVariables({var}, {inputs});
  EE.run();

  out->assign(&result->getVariable()->getPayload());
}

void inferIntLookupTableNet(Tensor *input, Tensor *out,
                            llvm::ArrayRef<int8_t> table, BackendKind kind) {
  ExecutionEngine EE(kind);
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  auto *var = VarFrom(input);
  auto outTy = mod.uniqueType(ElemKind::Int8QTy, {input->size()}, 3, 3);
  auto *lookupTable = F->createIntLookupTable("lookuptable", var, table, outTy);
  auto result = F->createSave("ret", lookupTable);
  EE.compile(CompilationMode::Infer, F);
  updateVariables({var}, {input});
  EE.run();
  out->assign(&result->getVariable()->getPayload());
}

void inferConvNet(Tensor *inputs, Tensor *filter, Tensor *bias, Tensor *out,
                  BackendKind kind) {
  ExecutionEngine EE(kind);
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  Variable *inputVar;
  Variable *filterVar;
  Variable *biasVar;
  Variable *outVar;
  TypeRef OT;
  if (inputs->getType().isQuantizedType()) {
    auto &outType = out->getType();
    inputVar = VarFrom(inputs);
    filterVar = VarFrom(filter);
    biasVar = VarFrom(bias);
    outVar = VarFrom(out);
    OT = F->getParent()->uniqueType(out->getElementType(), out->dims(),
                                    outType.getScale(), outType.getOffset());
  } else {
    inputVar = VarFrom(inputs);
    filterVar = VarFrom(filter);
    biasVar = VarFrom(bias);
    outVar = VarFrom(out);
    OT = F->getParent()->uniqueType(out->getElementType(), out->dims());
  }
  auto *conv =
      F->createConv("conv", inputVar, filterVar, biasVar, OT, 5, 3, 4, 1);
  auto result = F->createSave("ret", conv, outVar);
  EE.compile(CompilationMode::Infer, F);
  updateVariables({inputVar, filterVar, biasVar}, {inputs, filter, bias});
  EE.run();
  out->assign(&result->getVariable()->getPayload());
}

void trainConvNet(Tensor *inputs, Tensor *kernel1, Tensor *bias1,
                  Tensor *kernel2, Tensor *bias2, Tensor *selected,
                  llvm::ArrayRef<size_t> shape1, llvm::ArrayRef<size_t> shape2,
                  Tensor *out, BackendKind kind) {
  ExecutionEngine EE(kind);
  TrainingConfig TC;

  // This variable records the number of the next sample to be used for
  // training.
  size_t sampleCounter = 0;

  TC.learningRate = 0.03;
  TC.momentum = 0.3;
  TC.L2Decay = 0.01;
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  auto *var1 = VarFrom(inputs);
  auto *var2 = VarFrom(selected);
  auto *conv1 =
      F->createConv("conv1", var1, 3, {5, 3}, {2, 1}, {2, 1, 2, 1}, 1);
  cast<Variable>(conv1->getFilter())->assign(kernel1);
  cast<Variable>(conv1->getBias())->assign(bias1);
  auto *reshape1 = F->createReshape("reshape1", conv1, shape1);
  auto *conv2 = F->createConv("conv2", reshape1, 2, 2, 2, 0, 1);
  cast<Variable>(conv2->getFilter())->assign(kernel2);
  cast<Variable>(conv2->getBias())->assign(bias2);
  auto *reshape2 = F->createReshape("reshape2", conv2, shape2);
  auto *softmax = F->createSoftMax("softmax", reshape2, var2);
  auto result = F->createSave("ret", softmax);

  Function *TF = glow::differentiate(F, TC);
  EE.compile(CompilationMode::Train, TF);

  runBatch(EE, 8, sampleCounter, {var1, var2}, {inputs, selected});
  EE.compile(CompilationMode::Infer, F);
  updateVariables({var1, var2}, {inputs, selected});
  EE.run();
  out->assign(&result->getVariable()->getPayload());
}

void inferGatherNet(Tensor *data, Tensor *indices, Tensor *dest,
                    BackendKind kind) {
  ExecutionEngine EE(kind);
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  auto *dataV = VarFrom(data);
  auto *indicesV = VarFrom(indices);
  auto *gather = F->createGather("gather", dataV, indicesV);
  auto *result = F->createSave("ret", gather);
  EE.compile(CompilationMode::Infer, F);
  updateVariables({dataV, indicesV}, {data, indices});
  EE.run();
  dest->assign(&result->getVariable()->getPayload());
}

void inferLocalResponseNormalizationNet(Tensor *inputs, Tensor *out,
                                        BackendKind kind) {
  ExecutionEngine EE(kind);
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  auto *var = VarFrom(inputs);
  auto *lrn = F->createLocalResponseNormalization("lrn", var, 5, 3.0, 0.5, 1.5);
  auto result = F->createSave("ret", lrn);
  EE.compile(CompilationMode::Infer, F);
  updateVariables({var}, {inputs});
  EE.run();
  out->assign(&result->getVariable()->getPayload());
}

void trainLocalResponseNormalizationNet(Tensor *inputs, Tensor *weights,
                                        Tensor *bias, Tensor *selected,
                                        llvm::ArrayRef<size_t> shape1,
                                        llvm::ArrayRef<size_t> shape2,
                                        Tensor *out, BackendKind kind) {
  ExecutionEngine EE(kind);
  TrainingConfig TC;

  // This variable records the number of the next sample to be used for
  // training.
  size_t sampleCounter = 0;

  TC.learningRate = 0.06;
  TC.momentum = 0.1;
  TC.L2Decay = 0.01;
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  auto *var1 = VarFrom(inputs);
  auto *var2 = VarFrom(selected);
  auto *fc = F->createFullyConnected("fc", var1, bias->dims()[0]);
  cast<Variable>(fc->getWeights())->assign(weights);
  cast<Variable>(fc->getBias())->assign(bias);
  auto *reshape1 = F->createReshape("reshape1", fc, shape1);
  auto *lrn =
      F->createLocalResponseNormalization("lrn", reshape1, 2, 2.0, 0.5, 1.0);
  auto *reshape2 = F->createReshape("reshape2", lrn, shape2);
  auto *softmax = F->createSoftMax("softmax", reshape2, var2);
  auto result = F->createSave("ret", softmax);

  Function *TF = glow::differentiate(F, TC);
  EE.compile(CompilationMode::Train, TF);
  runBatch(EE, 8, sampleCounter, {var1, var2}, {inputs, selected});

  EE.compile(CompilationMode::Infer, F);
  runBatch(EE, 1, sampleCounter, {var1, var2}, {inputs, selected});
  out->assign(&result->getVariable()->getPayload());
}

void inferMatMulNet(Tensor *lhs, Tensor *rhs, Tensor *out, BackendKind kind) {
  ExecutionEngine EE(kind);
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  Variable *lhsVar;
  Variable *rhsVar;
  Variable *outVar;
  TypeRef OT;
  if (lhs->getType().isQuantizedType()) {
    auto &outType = out->getType();
    lhsVar = VarFrom(lhs);
    rhsVar = VarFrom(rhs);
    outVar = VarFrom(out);
    OT = F->getParent()->uniqueType(out->getElementType(), out->dims(),
                                    outType.getScale(), outType.getOffset());
  } else {
    lhsVar = VarFrom(lhs);
    rhsVar = VarFrom(rhs);
    outVar = VarFrom(out);
    OT = F->getParent()->uniqueType(out->getElementType(), out->dims());
  }
  auto *matmul = F->createMatMul("matmul", OT, lhsVar, rhsVar);
  auto result = F->createSave("ret", matmul, outVar);
  EE.compile(CompilationMode::Infer, F);
  updateVariables({lhsVar, rhsVar}, {lhs, rhs});
  EE.run();
  out->assign(&result->getVariable()->getPayload());
}

void inferMaxNet(Tensor *inputs1, Tensor *inputs2, Tensor *out,
                 BackendKind kind) {
  ExecutionEngine EE(kind);
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  auto *var1 = VarFrom(inputs1);
  auto *var2 = VarFrom(inputs2);
  auto *max = F->createMax("max", var1, var2);
  auto result = F->createSave("ret", max);
  EE.compile(CompilationMode::Infer, F);
  updateVariables({var1, var2}, {inputs1, inputs2});
  EE.run();
  out->assign(&result->getVariable()->getPayload());
}

void inferMinNet(Tensor *inputs1, Tensor *inputs2, Tensor *out,
                 BackendKind kind) {
  ExecutionEngine EE(kind);
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  auto *var1 = VarFrom(inputs1);
  auto *var2 = VarFrom(inputs2);
  auto *min = F->createMin("min", var1, var2);
  auto result = F->createSave("ret", min);
  EE.compile(CompilationMode::Infer, F);
  updateVariables({var1, var2}, {inputs1, inputs2});
  EE.run();
  out->assign(&result->getVariable()->getPayload());
}

void inferAvgPoolNet(Tensor *inputs, Tensor *out, BackendKind kind) {
  ExecutionEngine EE(kind);
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  auto *var = VarFrom(inputs);
  auto *pool = F->createAvgPool("pool", var, 3, 3, 1);
  auto result = F->createSave("ret", pool);
  EE.compile(CompilationMode::Infer, F);
  updateVariables({var}, {inputs});
  EE.run();
  out->assign(&result->getVariable()->getPayload());
}

void trainAvgPoolNet(Tensor *inputs, Tensor *weights, Tensor *bias,
                     Tensor *selected, llvm::ArrayRef<size_t> shape1,
                     llvm::ArrayRef<size_t> shape2, Tensor *out,
                     BackendKind kind) {
  ExecutionEngine EE(kind);
  TrainingConfig TC;

  // This variable records the number of the next sample to be used for
  // training.
  size_t sampleCounter = 0;

  TC.learningRate = 0.01;
  TC.momentum = 0.4;
  TC.L2Decay = 0.01;
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  auto *var1 = VarFrom(inputs);
  auto *var2 = VarFrom(selected);
  auto *fc = F->createFullyConnected("fc", var1, bias->dims()[0]);
  cast<Variable>(fc->getWeights())->assign(weights);
  cast<Variable>(fc->getBias())->assign(bias);
  auto *reshape1 = F->createReshape("reshape1", fc, shape1);
  auto *pool = F->createAvgPool("pool", reshape1, 2, 2, 0);
  auto *reshape2 = F->createReshape("reshape2", pool, shape2);
  auto *softmax = F->createSoftMax("softmax", reshape2, var2);
  auto result = F->createSave("ret", softmax);

  Function *TF = glow::differentiate(F, TC);
  EE.compile(CompilationMode::Train, TF);

  runBatch(EE, 10, sampleCounter, {var1, var2}, {inputs, selected});
  EE.compile(CompilationMode::Infer, F);
  updateVariables({var1, var2}, {inputs, selected});
  EE.run();
  out->assign(&result->getVariable()->getPayload());
}

void inferMaxPoolNet(Tensor *inputs, Tensor *out, BackendKind kind) {
  ExecutionEngine EE(kind);
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  auto *var = VarFrom(inputs);
  auto *pool = F->createMaxPool("pool", var, 4, 2, 3);
  auto result = F->createSave("ret", pool);
  EE.compile(CompilationMode::Infer, F);
  updateVariables({var}, {inputs});
  EE.run();
  out->assign(&result->getVariable()->getPayload());
}

void trainMaxPoolNet(Tensor *inputs, Tensor *weights, Tensor *bias,
                     Tensor *selected, llvm::ArrayRef<size_t> shape1,
                     llvm::ArrayRef<size_t> shape2, Tensor *out,
                     BackendKind kind) {
  ExecutionEngine EE(kind);
  TrainingConfig TC;

  // This variable records the number of the next sample to be used for
  // training.
  size_t sampleCounter = 0;

  TC.learningRate = 0.03;
  TC.momentum = 0.3;
  TC.L2Decay = 0.003;
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  auto *var1 = VarFrom(inputs);
  auto *var2 = VarFrom(selected);
  auto *fc = F->createFullyConnected("fc", var1, bias->dims()[0]);
  cast<Variable>(fc->getWeights())->assign(weights);
  cast<Variable>(fc->getBias())->assign(bias);
  auto *reshape1 = F->createReshape("reshape1", fc, shape1);
  auto *pool = F->createMaxPool("pool", reshape1, 5, 3, 4);
  auto *reshape2 = F->createReshape("reshape2", pool, shape2);
  auto *softmax = F->createSoftMax("softmax", reshape2, var2);
  auto result = F->createSave("ret", softmax);

  Function *TF = glow::differentiate(F, TC);
  EE.compile(CompilationMode::Train, TF);

  runBatch(EE, 7, sampleCounter, {var1, var2}, {inputs, selected});
  EE.compile(CompilationMode::Infer, F);
  runBatch(EE, 1, sampleCounter, {var1, var2}, {inputs, selected});
  out->assign(&result->getVariable()->getPayload());
}

void inferQuantizeNet(Tensor *inputs, float scale, int32_t offset, Tensor *out,
                      BackendKind kind) {
  ExecutionEngine EE(kind);
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  auto *var = VarFrom(inputs);
  auto QT1 = F->getParent()->uniqueType(ElemKind::Int8QTy, inputs->dims(),
                                        scale, offset);
  auto QT2 = F->getParent()->uniqueType(ElemKind::Int8QTy, inputs->dims(),
                                        scale * 1.125, offset + 1);
  auto *quantize = F->createQuantize("quantize", var, QT1);
  auto *rescale = F->createRescaleQuantized("rescale", quantize, QT2);
  auto *dequantize = F->createDequantize("dequantize", rescale);
  auto result = F->createSave("ret", dequantize);
  EE.compile(CompilationMode::Infer, F);
  updateVariables({var}, {inputs});
  EE.run();
  out->assign(&result->getVariable()->getPayload());
}

void inferReluNet(Tensor *inputs, Tensor *out, BackendKind kind) {
  ExecutionEngine EE(kind);
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  auto *var = VarFrom(inputs);
  auto *relu = F->createRELU("relu", var);
  auto result = F->createSave("ret", relu);
  EE.compile(CompilationMode::Infer, F);
  updateVariables({var}, {inputs});
  EE.run();
  out->assign(&result->getVariable()->getPayload());
}

void inferReshapeNet(Tensor *inputs, llvm::ArrayRef<size_t> shape, Tensor *out,
                     BackendKind kind) {
  ExecutionEngine EE(kind);
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  auto *var = VarFrom(inputs);
  auto *reshape = F->createReshape("reshape", var, shape);
  auto result = F->createSave("ret", reshape);
  EE.compile(CompilationMode::Infer, F);
  updateVariables({var}, {inputs});
  EE.run();
  out->assign(&result->getVariable()->getPayload());
}

void inferSelectNet(Tensor *cond, Tensor *inputs1, Tensor *inputs2, Tensor *out,
                    BackendKind kind) {
  ExecutionEngine EE(kind);
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  auto *var1 = VarFrom(cond);
  auto *var2 = VarFrom(inputs1);
  auto *var3 = VarFrom(inputs2);
  auto *select = F->createSelect("cond", var1, var2, var3);
  auto result = F->createSave("ret", select);
  EE.compile(CompilationMode::Infer, F);
  updateVariables({var1, var2, var3}, {cond, inputs1, inputs2});
  EE.run();

  out->assign(&result->getVariable()->getPayload());
}

void inferSigmoidNet(Tensor *inputs, Tensor *out, BackendKind kind) {
  ExecutionEngine EE(kind);
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  auto *var = VarFrom(inputs);
  auto *sigmoid = F->createSigmoid("sigmoid", var);
  auto result = F->createSave("ret", sigmoid);
  EE.compile(CompilationMode::Infer, F);
  updateVariables({var}, {inputs});
  EE.run();

  EE.run();
  out->assign(&result->getVariable()->getPayload());
}

void inferSmallConv(Tensor *inputs, Tensor *out, BackendKind kind) {
  ExecutionEngine EE(kind);
  auto &mod = EE.getModule();
  auto *F = mod.createFunction("main");
  auto *in = VarFrom(inputs);
  auto *C = F->createConv("conv2a", in, 64, 1, 1, 0, 1);
  cast<Variable>(C->getFilter())->getHandle().clear(0.3);
  cast<Variable>(C->getBias())->getHandle().clear(0.4);
  auto *result = F->createSave("ret", C);
  EE.compile(CompilationMode::Infer, F);
  updateVariables({in}, {inputs});
  EE.run();

  out->assign(&result->getVariable()->getPayload());
}

void inferGroupConv(Tensor *out, BackendKind kind) {
  ExecutionEngine EE(kind);
  auto &mod = EE.getModule();
  auto *F = mod.createFunction("main");

  auto *input = mod.createVariable(ElemKind::FloatTy, {1, 2, 1, 32}, "input");
  auto IH = input->getHandle();
  for (size_t i = 0; i < 2 * 32; i++) {
    IH.raw(i) = (i + 1) / 10.0;
  }

  auto filter =
      mod.createVariable(ElemKind::FloatTy, {128, 1, 1, 16}, "filter");
  auto FH = filter->getHandle();
  for (size_t i = 0; i < 128; i++)
    for (size_t j = 0; j < 16; j++) {
      FH.at({i, 0, 0, j}) = (i + j) / 100.0;
    }
  auto *zeroBias = mod.createVariable(ElemKind::FloatTy, {128}, "bias");
  zeroBias->getPayload().zero();

  auto outTy = mod.uniqueType(ElemKind::FloatTy, {1, 2, 1, 128});

  ConvolutionNode *CN =
      F->createConv("Conv", input, filter, zeroBias, outTy, 1, 1, 0, 2);
  SaveNode *result = F->createSave("save", CN);

  EE.compile(CompilationMode::Infer, F);
  EE.run();
  out->assign(&result->getVariable()->getPayload());
}

void inferNonSquarePaddingConv(Tensor *out, BackendKind kind) {
  ExecutionEngine EE(kind);
  auto &mod = EE.getModule();
  auto *F = mod.createFunction("main");

  auto *input = mod.createVariable(ElemKind::FloatTy, {1, 2, 1, 32}, "input");
  auto IH = input->getHandle();
  for (size_t i = 0; i < 2 * 32; i++) {
    IH.raw(i) = (i + 1) / 10.0;
  }

  auto filter =
      mod.createVariable(ElemKind::FloatTy, {128, 1, 1, 32}, "filter");
  auto FH = filter->getHandle();
  for (size_t i = 0; i < 128; i++)
    for (size_t j = 0; j < 16; j++) {
      FH.at({i, 0, 0, j}) = (i + j) / 100.0;
    }
  auto *zeroBias = mod.createVariable(ElemKind::FloatTy, {128}, "bias");
  zeroBias->getPayload().zero();
  auto outTy = mod.uniqueType(ElemKind::FloatTy, {1, 4, 5, 128});

  ConvolutionNode *CN = F->createConv("Conv", input, filter, zeroBias, outTy,
                                      {1, 1}, {1, 1}, {0, 1, 2, 3}, 1);
  SaveNode *result = F->createSave("save", CN);

  EE.compile(CompilationMode::Infer, F);
  EE.run();
  out->assign(&result->getVariable()->getPayload());
}

void inferNonSquareKernelConv(Tensor *out, BackendKind kind) {
  ExecutionEngine EE(kind);
  auto &mod = EE.getModule();
  auto *F = mod.createFunction("main");

  auto *input = mod.createVariable(ElemKind::FloatTy, {1, 2, 1, 32}, "input");
  auto IH = input->getHandle();
  for (size_t i = 0; i < 2 * 32; i++) {
    IH.raw(i) = (i + 1) / 10.0;
  }

  auto filter =
      mod.createVariable(ElemKind::FloatTy, {128, 2, 1, 32}, "filter");
  auto FH = filter->getHandle();
  for (size_t i = 0; i < 128; i++)
    for (size_t j = 0; j < 2; j++)
      for (size_t k = 0; k < 32; k++) {
        FH.at({i, j, 0, k}) = (i + j + k) / 100.0;
      }
  auto *zeroBias = mod.createVariable(ElemKind::FloatTy, {128}, "bias");
  zeroBias->getPayload().zero();
  auto outTy = mod.uniqueType(ElemKind::FloatTy, {1, 3, 5, 128});

  ConvolutionNode *CN = F->createConv("Conv", input, filter, zeroBias, outTy,
                                      {2, 1}, {1, 1}, {0, 1, 2, 3}, 1);
  SaveNode *result = F->createSave("save", CN);

  EE.compile(CompilationMode::Infer, F);
  EE.run();
  out->assign(&result->getVariable()->getPayload());
}

void inferNonSquareStrideConv(Tensor *out, BackendKind kind) {
  ExecutionEngine EE(kind);
  auto &mod = EE.getModule();
  auto *F = mod.createFunction("main");

  auto *input = mod.createVariable(ElemKind::FloatTy, {1, 2, 1, 32}, "input");
  auto IH = input->getHandle();
  for (size_t i = 0; i < 2 * 32; i++) {
    IH.raw(i) = (i + 1) / 10.0;
  }

  auto filter =
      mod.createVariable(ElemKind::FloatTy, {128, 2, 1, 32}, "filter");
  auto FH = filter->getHandle();
  for (size_t i = 0; i < 128; i++)
    for (size_t j = 0; j < 2; j++)
      for (size_t k = 0; k < 32; k++) {
        FH.at({i, j, 0, k}) = (i + j + k) / 100.0;
      }
  auto *zeroBias = mod.createVariable(ElemKind::FloatTy, {128}, "bias");
  zeroBias->getPayload().zero();
  auto outTy = mod.uniqueType(ElemKind::FloatTy, {1, 2, 5, 128});

  ConvolutionNode *CN = F->createConv("Conv", input, filter, zeroBias, outTy,
                                      {2, 1}, {2, 1}, {0, 1, 2, 3}, 1);
  SaveNode *result = F->createSave("save", CN);

  EE.compile(CompilationMode::Infer, F);
  EE.run();
  out->assign(&result->getVariable()->getPayload());
}

void inferConvDKKC8(Tensor *out, BackendKind kind) {
  ExecutionEngine EE(kind);
  auto &mod = EE.getModule();
  auto *F = mod.createFunction("main");

  auto *input = mod.createVariable(ElemKind::FloatTy, {3, 3, 3, 32}, "input");
  auto IH = input->getHandle();
  for (size_t i = 0; i < 3 * 3 * 3 * 32; i++) {
    IH.raw(i) = (i + 1) / 10.0;
  }

  auto filter =
      mod.createVariable(ElemKind::FloatTy, {192, 3, 3, 32}, "filter");
  auto FH = filter->getHandle();
  for (size_t i = 0; i < 192; i++)
    for (size_t j = 0; j < 3; j++)
      for (size_t k = 0; k < 3; k++)
        for (size_t l = 0; l < 32; l++) {
          FH.at({i, j, k, k}) = (i + j + k + l) / 200.0;
        }
  auto *zeroBias = mod.createVariable(ElemKind::FloatTy, {192}, "bias");
  zeroBias->getPayload().zero();
  auto outTy = mod.uniqueType(ElemKind::FloatTy, {3, 3, 3, 192});

  ConvolutionNode *CN = F->createConv("Conv", input, filter, zeroBias, outTy,
                                      {3, 3}, {1, 1}, {1, 1, 1, 1}, 1);
  SaveNode *result = F->createSave("save", CN);

  EE.compile(CompilationMode::Infer, F);
  EE.run();
  out->assign(&result->getVariable()->getPayload());
}

void inferSoftMaxNet(Tensor *inputs, Tensor *selected, Tensor *out,
                     BackendKind kind) {
  ExecutionEngine EE(kind);
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  auto *var1 = VarFrom(inputs);
  auto *var2 = VarFrom(selected);
  auto *softmax = F->createSoftMax("softmax", var1, var2);
  auto result = F->createSave("ret", softmax);
  EE.compile(CompilationMode::Infer, F);
  updateVariables({var1, var2}, {inputs, selected});
  EE.run();
  out->assign(&result->getVariable()->getPayload());
}

void trainSoftMaxNet(Tensor *inputs, Tensor *weights, Tensor *bias,
                     Tensor *selected, Tensor *out, BackendKind kind) {
  ExecutionEngine EE(kind);
  TrainingConfig TC;

  // This variable records the number of the next sample to be used for
  // training.
  size_t sampleCounter = 0;

  TC.learningRate = 0.003;
  TC.momentum = 0.7;
  TC.L2Decay = 0.001;
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  auto *var1 = VarFrom(inputs);
  auto *var2 = VarFrom(selected);
  auto *fc = F->createFullyConnected("fc", var1, bias->dims()[0]);
  cast<Variable>(fc->getWeights())->assign(weights);
  cast<Variable>(fc->getBias())->assign(bias);
  auto *softmax = F->createSoftMax("softmax", fc, var2);
  auto result = F->createSave("ret", softmax);

  Function *TF = glow::differentiate(F, TC);
  EE.compile(CompilationMode::Train, TF);

  runBatch(EE, 30, sampleCounter, {var1, var2}, {inputs, selected});
  EE.compile(CompilationMode::Infer, F);
  updateVariables({var1, var2}, {inputs, selected});
  EE.run();
  out->assign(&result->getVariable()->getPayload());
}

void inferTanhNet(Tensor *inputs, Tensor *out, BackendKind kind) {
  ExecutionEngine EE(kind);
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  auto *var = VarFrom(inputs);
  auto *tanh = F->createTanh("tanh", var);
  auto result = F->createSave("ret", tanh);
  EE.compile(CompilationMode::Infer, F);
  updateVariables({var}, {inputs});
  EE.run();
  out->assign(&result->getVariable()->getPayload());
}

void inferTransposeNet(Tensor *inputs, Tensor *out, BackendKind kind) {
  ExecutionEngine EE(kind);
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  auto *var = VarFrom(inputs);
  auto *tr = F->createTranspose("tr", var, {1, 0});
  auto result = F->createSave("ret", tr);
  EE.compile(CompilationMode::Infer, F);
  updateVariables({var}, {inputs});
  EE.run();
  out->assign(&result->getVariable()->getPayload());
}

void inferTanhConcatNet(Tensor *input1, Tensor *input2, Tensor *input3,
                        Tensor *out, BackendKind kind) {
  ExecutionEngine EE(kind);
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  auto *var1 = VarFrom(input1);
  auto *var2 = VarFrom(input2);
  auto *var3 = VarFrom(input3);
  auto *T1 = F->createTanh("tanh1", var1);
  auto *T2 = F->createTanh("tanh2", var2);
  auto *T3 = F->createTanh("tanh3", var3);
  Node *C1 = F->createConcat("concat", {T1, T2}, 0);
  Node *C2 = F->createConcat("concat", {T2, T3, C1, T2}, 0);
  auto *result = F->createSave("ret", C2);
  EE.compile(CompilationMode::Infer, F);
  updateVariables({var1, var2, var3}, {input1, input2, input3});
  EE.run();
  out->assign(&result->getVariable()->getPayload());
}

void inferBasicConvNet(Tensor *inputs, Tensor *out, BackendKind kind,
                       size_t convDepth) {
  ExecutionEngine EE(kind);
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  auto *var = VarFrom(inputs);
  auto *tr = F->createTranspose("tr", var, NCHW2NHWC);
  auto *conv =
      F->createConv("conv", tr, convDepth, {5, 5}, {2, 2}, {1, 1, 1, 1}, 1);
  cast<Variable>(conv->getFilter())->getHandle().clear(2);
  cast<Variable>(conv->getBias())->getHandle().clear(2);
  auto *pool = F->createMaxPool("pool", conv, 2, 2, 0);
  auto result = F->createSave("ret", pool);
  EE.compile(CompilationMode::Infer, F);
  updateVariables({var}, {inputs});
  EE.run();
  out->assign(&result->getVariable()->getPayload());
}

void inferBasicFCNet(Tensor *inputs, Tensor *out, BackendKind kind) {
  ExecutionEngine EE(kind);
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  auto *var = VarFrom(inputs);
  auto *tr = F->createTranspose("tr", var, NCHW2NHWC);
  auto *fc = F->createFullyConnected("fc", tr, 16);
  auto *rl0 = F->createRELU("relu", fc);
  auto *fc2 = F->createFullyConnected("fc2", rl0, 8);
  auto *rl1 = F->createRELU("relu", fc2);
  cast<Variable>(fc->getWeights())->getHandle().clear(0.8);
  cast<Variable>(fc2->getWeights())->getHandle().clear(1.5);
  auto result = F->createSave("ret", rl1);
  EE.compile(CompilationMode::Infer, F);
  updateVariables({var}, {inputs});
  EE.run();
  out->assign(&result->getVariable()->getPayload());
}

void inferMixedNet(Tensor *inputs, Tensor *out, BackendKind kind) {
  ExecutionEngine EE(kind);
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  auto *var = VarFrom(inputs);
  auto *selected = mod.createVariable(ElemKind::Int64ITy, {2, 1}, "selected");

  auto *tr = F->createTranspose("tr", var, NCHW2NHWC);
  auto *fc = F->createFullyConnected("fc", tr, 16);
  auto *th0 = F->createTanh("tanh", fc);
  auto *sg0 = F->createSigmoid("sig", fc);
  auto *A1 = F->createAdd("add", th0, sg0);
  auto *fc2 = F->createFullyConnected("fc2", A1, 16);

  auto *R = F->createRegression("reg", fc2, fc2);
  auto *SM = F->createSoftMax("SM", R, selected);
  auto result = F->createSave("ret", SM);

  cast<Variable>(fc->getWeights())->getHandle().clear(0.4);
  cast<Variable>(fc2->getWeights())->getHandle().clear(3.5);

  EE.compile(CompilationMode::Infer, F);
  updateVariables({var}, {inputs});
  EE.run();
  out->assign(&result->getVariable()->getPayload());
}

void inferComplexNet1(Tensor *inputs1, Tensor *inputs2, Tensor *inputs3,
                      Tensor *inputs4, Tensor *out, BackendKind kind) {
  ExecutionEngine EE(kind);
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  auto *var1 = VarFrom(inputs1);
  auto *var2 = VarFrom(inputs2);
  auto *var3 = VarFrom(inputs3);
  auto *var4 = VarFrom(inputs4);
  auto *conv1 = F->createConv("conv1", var1, 6, 4, 1, 2, 1);
  cast<Variable>(conv1->getFilter())->getHandle().clear(0.5);
  cast<Variable>(conv1->getBias())->getHandle().clear(0.7);
  auto *sigmoid1 = F->createSigmoid("sigmoid1", conv1);
  auto *fc1 = F->createFullyConnected("fc1", var2, 2352);
  cast<Variable>(fc1->getWeights())->getHandle().clear(0.6);
  auto *reshape1 = F->createReshape("reshape1", fc1, {8, 14, 28, 6});
  auto *relu1 = F->createRELU("relu1", reshape1);
  auto *pool1 = F->createMaxPool("pool1", relu1, 2, 2, 1);
  auto *add = F->createAdd("add", sigmoid1, pool1);
  auto *tanh = F->createTanh("tanh", add);
  auto *fc2 = F->createFullyConnected("fc2", var3, 720);
  cast<Variable>(fc2->getWeights())->getHandle().clear(1.1);
  auto *reshape2 = F->createReshape("reshape2", fc2, {8, 8, 15, 6});
  auto *mul = F->createMul("mul", tanh, reshape2);
  auto *sigmoid2 = F->createSigmoid("sigmoid2", mul);
  auto *conv2 = F->createConv("conv2", sigmoid2, 7, 3, 2, 1, 1);
  cast<Variable>(conv2->getFilter())->getHandle().clear(0.3);
  cast<Variable>(conv2->getBias())->getHandle().clear(1.3);
  auto *reshape3 = F->createReshape("reshape3", conv2, {8, 8, 7, 4});
  auto *sub = F->createSub("sub", reshape3, var4);
  auto *relu2 = F->createRELU("relu2", sub);
  auto *pool2 = F->createAvgPool("pool2", relu2, 3, 2, 1);
  auto *sigmoid3 = F->createSigmoid("sigmoid3", pool2);
  auto result = F->createSave("ret", sigmoid3);
  EE.compile(CompilationMode::Infer, F);
  updateVariables({var1, var2, var3, var4},
                  {inputs1, inputs2, inputs3, inputs4});
  EE.run();
  out->assign(&result->getVariable()->getPayload());
}

namespace {
// Helper for initializing conv node filter/bias from input tensors.
static void initConv(ConvolutionNode *C, Tensor &filter, Tensor &bias) {
  cast<Variable>(C->getFilter())->getPayload().assign(&filter);
  cast<Variable>(C->getBias())->getPayload().assign(&bias);
}
} // namespace

void inferTinyResnet(Tensor *input, Tensor *out, std::vector<Tensor> &weights,
                     BackendKind kind) {
  ExecutionEngine EE(kind);
  auto &mod = EE.getModule();
  auto *F = mod.createFunction("main");

  auto *in = VarFrom(input);
  auto *conv1 = F->createConv("conv1", in, 256, 1, 1, 0, 1);
  auto *conv2a = F->createConv("conv2a", in, 64, 1, 1, 0, 1);
  auto *relu2a = F->createRELU("relu2a", conv2a);
  auto *conv2b = F->createConv("conv2b", relu2a, 64, 3, 1, 1, 1);
  auto *relu2b = F->createRELU("relu2b", conv2b);
  auto *conv2c = F->createConv("conv2c", relu2b, 256, 1, 1, 0, 1);
  auto *add = F->createAdd("add", conv2c, conv1);
  auto *relu = F->createRELU("res2a_relu", add);
  auto *result = F->createSave("ret", relu);

  initConv(conv1, weights[0], weights[1]);
  initConv(conv2a, weights[2], weights[3]);
  initConv(conv2b, weights[4], weights[5]);
  initConv(conv2c, weights[6], weights[7]);

  EE.compile(CompilationMode::Infer, F);

  updateVariables({in}, {input});
  EE.run();
  out->assign(&result->getVariable()->getPayload());
}

void inferExtract3D(Tensor *input, Tensor *out, BackendKind kind) {
  ExecutionEngine EE(kind);
  auto &mod = EE.getModule();
  auto *F = mod.createFunction("main");

  auto *inputs = VarFrom(input);

  auto *x1 = F->createSlice("ex1", inputs, {0, 5, 0}, {1, 100, 100});
  auto *x2 = F->createSlice("ex2", inputs, {1, 5, 0}, {2, 100, 100});
  auto *x3 = F->createSlice("ex3", inputs, {2, 5, 0}, {3, 100, 100});
  auto *x4 = F->createSlice("ex4", inputs, {3, 5, 0}, {4, 100, 100});

  auto *x12 = F->createConcat("x12", {x1, x2}, 1);
  auto *x34 = F->createConcat("x34", {x3, x4}, 1);
  auto *x13 = F->createConcat("x34", {x1, x3}, 1);
  auto *x24 = F->createConcat("x34", {x2, x4}, 1);

  auto *add1 = F->createAdd("add1", x12, x34);
  auto *add2 = F->createAdd("add1", x13, x24);
  auto *add3 = F->createAdd("add1", add1, add2);

  auto *e = F->createSlice("slice", add3, {0, 55, 50}, {1, 150, 100});
  auto *result = F->createSave("ret", e);

  EE.compile(CompilationMode::Infer, F);

  updateVariables({inputs}, {input});
  EE.run();
  out->assign(&result->getVariable()->getPayload());
}

void inferMaxSplat(Tensor *input, Tensor *out, BackendKind kind) {
  ExecutionEngine EE(kind);
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  auto *var = VarFrom(input);
  auto T = mod.uniqueType(ElemKind::Int8QTy, var->getType()->dims(),
                          2 * var->getType()->getScale(),
                          -var->getType()->getOffset());

  auto *rescale = F->createRescaleQuantized("rescale", var, T);

  auto *splat1 = F->createSplat("splat1", T, 0.0);
  auto *splat2 = F->createSplat("splat2", T, 5.0);

  auto *max1 = F->createMax("max1", rescale, splat1);
  auto *max2 = F->createMax("max2", splat2, max1);

  auto result = F->createSave("ret", max2);
  EE.compile(CompilationMode::Infer, F);
  updateVariables({var}, {input});
  EE.run();
  out->assign(&result->getVariable()->getPayload());
}

} // namespace glow
