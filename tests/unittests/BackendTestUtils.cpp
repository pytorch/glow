// Copyright 2017-2018 Facebook. All Rights Reserved.

#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Graph/Graph.h"
#include "glow/IR/IR.h"
#include "glow/IR/IRBuilder.h"
#include "glow/IR/Instrs.h"

namespace glow {

using llvm::cast;

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
    auto &batchType = batch->getType();
    auto &sliceType = slice->getType();
    auto &outType = out->getType();
    batchVar = mod.createVariable(batch->getElementType(), batch->dims(),
                                  batchType.getScale(), batchType.getOffset(),
                                  "batch", Variable::VisibilityKind::Public);
    sliceVar = mod.createVariable(slice->getElementType(), slice->dims(),
                                  sliceType.getScale(), sliceType.getOffset(),
                                  "slice", Variable::VisibilityKind::Public);
    outVar = mod.createVariable(out->getElementType(), out->dims(),
                                outType.getScale(), outType.getOffset(), "out",
                                Variable::VisibilityKind::Public);
    OT = F->getParent()->uniqueType(out->getElementType(), out->dims(),
                                    outType.getScale(), outType.getOffset());
  } else {
    batchVar = mod.createVariable(batch->getElementType(), batch->dims(),
                                  "batch", Variable::VisibilityKind::Public);
    sliceVar = mod.createVariable(slice->getElementType(), slice->dims(),
                                  "slice", Variable::VisibilityKind::Public);
    outVar = mod.createVariable(out->getElementType(), out->dims(), "out",
                                Variable::VisibilityKind::Public);
    OT = F->getParent()->uniqueType(out->getElementType(), out->dims());
  }
  auto *batchedadd = F->createBatchedAdd("batchedadd", OT, batchVar, sliceVar);
  auto result = F->createSave("ret", batchedadd, outVar);
  EE.compile(CompilationMode::Infer, F);
  EE.run({batchVar, sliceVar}, {batch, slice});
  out->copyFrom(&result->getVariable()->getPayload());
}

void inferBatchedReduceAddNet(Tensor *inputs, Tensor *out, BackendKind kind) {
  ExecutionEngine EE(kind);
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  auto *var = mod.createVariable(inputs->getElementType(), inputs->dims(),
                                 "input", Variable::VisibilityKind::Public);
  auto *batchedreduce = F->createBatchedReduceAdd("batchedreduce", var);
  auto result = F->createSave("ret", batchedreduce);
  EE.compile(CompilationMode::Infer, F);
  EE.run({var}, {inputs});
  out->copyFrom(&result->getVariable()->getPayload());
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
    auto &inputType = inputs->getType();
    auto &filterType = filter->getType();
    auto &biasType = bias->getType();
    auto &outType = out->getType();
    inputVar = mod.createVariable(inputs->getElementType(), inputs->dims(),
                                  inputType.getScale(), inputType.getOffset(),
                                  "input", Variable::VisibilityKind::Public);
    filterVar = mod.createVariable(
        filter->getElementType(), filter->dims(), filterType.getScale(),
        filterType.getOffset(), "filter", Variable::VisibilityKind::Public);
    biasVar = mod.createVariable(bias->getElementType(), bias->dims(),
                                 biasType.getScale(), biasType.getOffset(),
                                 "bias", Variable::VisibilityKind::Public);
    outVar = mod.createVariable(out->getElementType(), out->dims(),
                                outType.getScale(), outType.getOffset(), "out",
                                Variable::VisibilityKind::Public);
    OT = F->getParent()->uniqueType(out->getElementType(), out->dims(),
                                    outType.getScale(), outType.getOffset());
  } else {
    inputVar = mod.createVariable(inputs->getElementType(), inputs->dims(),
                                  "input", Variable::VisibilityKind::Public);
    filterVar = mod.createVariable(filter->getElementType(), filter->dims(),
                                   "filter", Variable::VisibilityKind::Public);
    biasVar = mod.createVariable(bias->getElementType(), bias->dims(), "bias",
                                 Variable::VisibilityKind::Public);
    outVar = mod.createVariable(out->getElementType(), out->dims(), "out",
                                Variable::VisibilityKind::Public);
    OT = F->getParent()->uniqueType(out->getElementType(), out->dims());
  }
  auto *conv =
      F->createConv("conv", inputVar, filterVar, biasVar, OT, 10, 5, 3, 4);
  auto result = F->createSave("ret", conv, outVar);
  EE.compile(CompilationMode::Infer, F);
  EE.run({inputVar, filterVar, biasVar}, {inputs, filter, bias});
  out->copyFrom(&result->getVariable()->getPayload());
}

void trainConvNet(Tensor *inputs, Tensor *kernel1, Tensor *bias1,
                  Tensor *kernel2, Tensor *bias2, Tensor *selected,
                  llvm::ArrayRef<size_t> shape1, llvm::ArrayRef<size_t> shape2,
                  Tensor *out, BackendKind kind) {
  ExecutionEngine EE(kind);
  EE.getConfig().learningRate = 0.03;
  EE.getConfig().momentum = 0.3;
  EE.getConfig().L2Decay = 0.01;
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  auto *var1 = mod.createVariable(inputs->getElementType(), inputs->dims(),
                                  "input", Variable::VisibilityKind::Public,
                                  Variable::TrainKind::None);
  auto *var2 = mod.createVariable(selected->getElementType(), selected->dims(),
                                  "selected", Variable::VisibilityKind::Public,
                                  Variable::TrainKind::None);
  auto *conv1 = F->createConv("conv1", var1, 3, 3, 2, 1);
  cast<Variable>(conv1->getFilter())->copyFrom(kernel1);
  cast<Variable>(conv1->getBias())->copyFrom(bias1);
  auto *reshape1 = F->createReshape("reshape1", conv1, shape1);
  auto *conv2 = F->createConv("conv2", reshape1, 2, 2, 2, 0);
  cast<Variable>(conv2->getFilter())->copyFrom(kernel2);
  cast<Variable>(conv2->getBias())->copyFrom(bias2);
  auto *reshape2 = F->createReshape("reshape2", conv2, shape2);
  auto *softmax = F->createSoftMax("softmax", reshape2, var2);
  auto result = F->createSave("ret", softmax);

  Function *TF = glow::differentiate(F, EE.getConfig());
  EE.compile(CompilationMode::Train, TF);

  EE.runBatch(8, {var1, var2}, {inputs, selected});
  EE.compile(CompilationMode::Infer, F);
  EE.run({var1, var2}, {inputs, selected});
  out->copyFrom(&result->getVariable()->getPayload());
}

void inferGatherNet(Tensor *data, Tensor *indices, Tensor *dest,
                    BackendKind kind) {
  ExecutionEngine EE(kind);
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  auto *dataV = mod.createVariable(data->getElementType(), data->dims(), "data",
                                   Variable::VisibilityKind::Public);
  auto *indicesV =
      mod.createVariable(indices->getElementType(), indices->dims(), "indices",
                         Variable::VisibilityKind::Public);
  auto *gather = F->createGather("gather", dataV, indicesV);
  auto *result = F->createSave("ret", gather);
  EE.compile(CompilationMode::Infer, F);
  EE.run({dataV, indicesV}, {data, indices});
  dest->copyFrom(&result->getVariable()->getPayload());
}

void inferLocalResponseNormalizationNet(Tensor *inputs, Tensor *out,
                                        BackendKind kind) {
  ExecutionEngine EE(kind);
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  auto *var = mod.createVariable(inputs->getElementType(), inputs->dims(),
                                 "input", Variable::VisibilityKind::Public);
  auto *lrn = F->createLocalResponseNormalization("lrn", var, 5, 3.0, 0.5, 1.5);
  auto result = F->createSave("ret", lrn);
  EE.compile(CompilationMode::Infer, F);
  EE.run({var}, {inputs});
  out->copyFrom(&result->getVariable()->getPayload());
}

void trainLocalResponseNormalizationNet(Tensor *inputs, Tensor *weights,
                                        Tensor *bias, Tensor *selected,
                                        llvm::ArrayRef<size_t> shape1,
                                        llvm::ArrayRef<size_t> shape2,
                                        Tensor *out, BackendKind kind) {
  ExecutionEngine EE(kind);
  EE.getConfig().learningRate = 0.06;
  EE.getConfig().momentum = 0.1;
  EE.getConfig().L2Decay = 0.01;
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  auto *var1 = mod.createVariable(inputs->getElementType(), inputs->dims(),
                                  "input", Variable::VisibilityKind::Public);
  auto *var2 = mod.createVariable(selected->getElementType(), selected->dims(),
                                  "selected", Variable::VisibilityKind::Public,
                                  Variable::TrainKind::None);
  auto *fc = F->createFullyConnected("fc", var1, bias->dims()[0]);
  cast<Variable>(fc->getWeights())->copyFrom(weights);
  cast<Variable>(fc->getBias())->copyFrom(bias);
  auto *reshape1 = F->createReshape("reshape1", fc, shape1);
  auto *lrn =
      F->createLocalResponseNormalization("lrn", reshape1, 2, 2.0, 0.5, 1.0);
  auto *reshape2 = F->createReshape("reshape2", lrn, shape2);
  auto *softmax = F->createSoftMax("softmax", reshape2, var2);
  auto result = F->createSave("ret", softmax);

  Function *TF = glow::differentiate(F, EE.getConfig());
  EE.compile(CompilationMode::Train, TF);
  EE.runBatch(8, {var1, var2}, {inputs, selected});

  EE.compile(CompilationMode::Infer, F);
  EE.runBatch(1, {var1, var2}, {inputs, selected});
  out->copyFrom(&result->getVariable()->getPayload());
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
    auto &lhsType = lhs->getType();
    auto &rhsType = rhs->getType();
    auto &outType = out->getType();
    lhsVar = mod.createVariable(lhs->getElementType(), lhs->dims(),
                                lhsType.getScale(), lhsType.getOffset(), "lhs",
                                Variable::VisibilityKind::Public);
    rhsVar = mod.createVariable(rhs->getElementType(), rhs->dims(),
                                rhsType.getScale(), rhsType.getOffset(), "rhs",
                                Variable::VisibilityKind::Public);
    outVar = mod.createVariable(out->getElementType(), out->dims(),
                                outType.getScale(), outType.getOffset(), "out",
                                Variable::VisibilityKind::Public);
    OT = F->getParent()->uniqueType(out->getElementType(), out->dims(),
                                    outType.getScale(), outType.getOffset());
  } else {
    lhsVar = mod.createVariable(lhs->getElementType(), lhs->dims(), "lhs",
                                Variable::VisibilityKind::Public);
    rhsVar = mod.createVariable(rhs->getElementType(), rhs->dims(), "rhs",
                                Variable::VisibilityKind::Public);
    outVar = mod.createVariable(out->getElementType(), out->dims(), "out",
                                Variable::VisibilityKind::Public);
    OT = F->getParent()->uniqueType(out->getElementType(), out->dims());
  }
  auto *matmul = F->createMatMul("matmul", OT, lhsVar, rhsVar);
  auto result = F->createSave("ret", matmul, outVar);
  EE.compile(CompilationMode::Infer, F);
  EE.run({lhsVar, rhsVar}, {lhs, rhs});
  out->copyFrom(&result->getVariable()->getPayload());
}

void inferMaxNet(Tensor *inputs1, Tensor *inputs2, Tensor *out,
                 BackendKind kind) {
  ExecutionEngine EE(kind);
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  auto *var1 = mod.createVariable(inputs1->getElementType(), inputs1->dims(),
                                  "input1", Variable::VisibilityKind::Public);
  auto *var2 = mod.createVariable(inputs2->getElementType(), inputs2->dims(),
                                  "input2", Variable::VisibilityKind::Public);
  auto *max = F->createMax("max", var1, var2);
  auto result = F->createSave("ret", max);
  EE.compile(CompilationMode::Infer, F);
  EE.run({var1, var2}, {inputs1, inputs2});
  out->copyFrom(&result->getVariable()->getPayload());
}

void inferMinNet(Tensor *inputs1, Tensor *inputs2, Tensor *out,
                 BackendKind kind) {
  ExecutionEngine EE(kind);
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  auto *var1 = mod.createVariable(inputs1->getElementType(), inputs1->dims(),
                                  "input1", Variable::VisibilityKind::Public);
  auto *var2 = mod.createVariable(inputs2->getElementType(), inputs2->dims(),
                                  "input2", Variable::VisibilityKind::Public);
  auto *min = F->createMin("min", var1, var2);
  auto result = F->createSave("ret", min);
  EE.compile(CompilationMode::Infer, F);
  EE.run({var1, var2}, {inputs1, inputs2});
  out->copyFrom(&result->getVariable()->getPayload());
}

void inferPoolAvgNet(Tensor *inputs, Tensor *out, BackendKind kind) {
  ExecutionEngine EE(kind);
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  auto *var = mod.createVariable(inputs->getElementType(), inputs->dims(),
                                 "input", Variable::VisibilityKind::Public);
  auto *pool = F->createPoolAvg("pool", var, 3, 3, 1);
  auto result = F->createSave("ret", pool);
  EE.compile(CompilationMode::Infer, F);
  EE.run({var}, {inputs});
  out->copyFrom(&result->getVariable()->getPayload());
}

void trainPoolAvgNet(Tensor *inputs, Tensor *weights, Tensor *bias,
                     Tensor *selected, llvm::ArrayRef<size_t> shape1,
                     llvm::ArrayRef<size_t> shape2, Tensor *out,
                     BackendKind kind) {
  ExecutionEngine EE(kind);
  EE.getConfig().learningRate = 0.01;
  EE.getConfig().momentum = 0.4;
  EE.getConfig().L2Decay = 0.01;
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  auto *var1 = mod.createVariable(inputs->getElementType(), inputs->dims(),
                                  "input", Variable::VisibilityKind::Public);
  auto *var2 = mod.createVariable(selected->getElementType(), selected->dims(),
                                  "selected", Variable::VisibilityKind::Public,
                                  Variable::TrainKind::None);
  auto *fc = F->createFullyConnected("fc", var1, bias->dims()[0]);
  cast<Variable>(fc->getWeights())->copyFrom(weights);
  cast<Variable>(fc->getBias())->copyFrom(bias);
  auto *reshape1 = F->createReshape("reshape1", fc, shape1);
  auto *pool = F->createPoolAvg("pool", reshape1, 2, 2, 0);
  auto *reshape2 = F->createReshape("reshape2", pool, shape2);
  auto *softmax = F->createSoftMax("softmax", reshape2, var2);
  auto result = F->createSave("ret", softmax);

  Function *TF = glow::differentiate(F, EE.getConfig());
  EE.compile(CompilationMode::Train, TF);

  EE.runBatch(10, {var1, var2}, {inputs, selected});
  EE.compile(CompilationMode::Infer, F);
  EE.run({var1, var2}, {inputs, selected});
  out->copyFrom(&result->getVariable()->getPayload());
}

void inferPoolMaxNet(Tensor *inputs, Tensor *out, BackendKind kind) {
  ExecutionEngine EE(kind);
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  auto *var = mod.createVariable(inputs->getElementType(), inputs->dims(),
                                 "input", Variable::VisibilityKind::Public);
  auto *pool = F->createPoolMax("pool", var, 4, 2, 3);
  auto result = F->createSave("ret", pool);
  EE.compile(CompilationMode::Infer, F);
  EE.run({var}, {inputs});
  out->copyFrom(&result->getVariable()->getPayload());
}

void trainPoolMaxNet(Tensor *inputs, Tensor *weights, Tensor *bias,
                     Tensor *selected, llvm::ArrayRef<size_t> shape1,
                     llvm::ArrayRef<size_t> shape2, Tensor *out,
                     BackendKind kind) {
  ExecutionEngine EE(kind);
  EE.getConfig().learningRate = 0.03;
  EE.getConfig().momentum = 0.3;
  EE.getConfig().L2Decay = 0.003;
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  auto *var1 = mod.createVariable(inputs->getElementType(), inputs->dims(),
                                  "input", Variable::VisibilityKind::Public);
  auto *var2 = mod.createVariable(selected->getElementType(), selected->dims(),
                                  "selected", Variable::VisibilityKind::Public,
                                  Variable::TrainKind::None);
  auto *fc = F->createFullyConnected("fc", var1, bias->dims()[0]);
  cast<Variable>(fc->getWeights())->copyFrom(weights);
  cast<Variable>(fc->getBias())->copyFrom(bias);
  auto *reshape1 = F->createReshape("reshape1", fc, shape1);
  auto *pool = F->createPoolMax("pool", reshape1, 5, 3, 4);
  auto *reshape2 = F->createReshape("reshape2", pool, shape2);
  auto *softmax = F->createSoftMax("softmax", reshape2, var2);
  auto result = F->createSave("ret", softmax);

  Function *TF = glow::differentiate(F, EE.getConfig());
  EE.compile(CompilationMode::Train, TF);

  EE.runBatch(7, {var1, var2}, {inputs, selected});
  EE.compile(CompilationMode::Infer, F);
  EE.runBatch(1, {var1, var2}, {inputs, selected});
  out->copyFrom(&result->getVariable()->getPayload());
}

void inferQuantizeNet(Tensor *inputs, float scale, int32_t offset, Tensor *out,
                      BackendKind kind) {
  ExecutionEngine EE(kind);
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  auto *var = mod.createVariable(inputs->getElementType(), inputs->dims(),
                                 "input", Variable::VisibilityKind::Public);
  auto QT1 = F->getParent()->uniqueType(ElemKind::Int8QTy, inputs->dims(),
                                        scale, offset);
  auto QT2 = F->getParent()->uniqueType(ElemKind::Int8QTy, inputs->dims(),
                                        scale * 1.125, offset + 1);
  auto *quantize = F->createQuantize("quantize", var, QT1);
  auto *rescale = F->createRescaleQuantized("rescale", quantize, QT2);
  auto *dequantize = F->createDequantize("dequantize", rescale);
  auto result = F->createSave("ret", dequantize);
  EE.compile(CompilationMode::Infer, F);
  EE.run({var}, {inputs});
  out->copyFrom(&result->getVariable()->getPayload());
}

void inferReluNet(Tensor *inputs, Tensor *out, BackendKind kind) {
  ExecutionEngine EE(kind);
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  auto *var = mod.createVariable(inputs->getElementType(), inputs->dims(),
                                 "input", Variable::VisibilityKind::Public);
  auto *relu = F->createRELU("relu", var);
  auto result = F->createSave("ret", relu);
  EE.compile(CompilationMode::Infer, F);
  EE.run({var}, {inputs});
  out->copyFrom(&result->getVariable()->getPayload());
}

void inferReshapeNet(Tensor *inputs, llvm::ArrayRef<size_t> shape, Tensor *out,
                     BackendKind kind) {
  ExecutionEngine EE(kind);
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  auto *var = mod.createVariable(inputs->getElementType(), inputs->dims(),
                                 "input", Variable::VisibilityKind::Public);
  auto *reshape = F->createReshape("reshape", var, shape);
  auto result = F->createSave("ret", reshape);
  EE.compile(CompilationMode::Infer, F);
  EE.run({var}, {inputs});
  out->copyFrom(&result->getVariable()->getPayload());
}

void inferSelectNet(Tensor *cond, Tensor *inputs1, Tensor *inputs2, Tensor *out,
                    BackendKind kind) {
  ExecutionEngine EE(kind);
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  auto *var1 = mod.createVariable(cond->getElementType(), cond->dims(), "cond",
                                  Variable::VisibilityKind::Public);
  auto *var2 = mod.createVariable(inputs1->getElementType(), inputs1->dims(),
                                  "input1", Variable::VisibilityKind::Public);
  auto *var3 = mod.createVariable(inputs2->getElementType(), inputs2->dims(),
                                  "input2", Variable::VisibilityKind::Public);
  auto *select = F->createSelect("cond", var1, var2, var3);
  auto result = F->createSave("ret", select);
  EE.compile(CompilationMode::Infer, F);
  EE.run({var1, var2, var3}, {cond, inputs1, inputs2});
  out->copyFrom(&result->getVariable()->getPayload());
}

void inferSigmoidNet(Tensor *inputs, Tensor *out, BackendKind kind) {
  ExecutionEngine EE(kind);
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  auto *var = mod.createVariable(inputs->getElementType(), inputs->dims(),
                                 "input", Variable::VisibilityKind::Public);
  auto *sigmoid = F->createSigmoid("sigmoid", var);
  auto result = F->createSave("ret", sigmoid);
  EE.compile(CompilationMode::Infer, F);
  EE.run({var}, {inputs});
  out->copyFrom(&result->getVariable()->getPayload());
}

void inferSoftMaxNet(Tensor *inputs, Tensor *selected, Tensor *out,
                     BackendKind kind) {
  ExecutionEngine EE(kind);
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  auto *var1 = mod.createVariable(inputs->getElementType(), inputs->dims(),
                                  "input", Variable::VisibilityKind::Public);
  auto *var2 = mod.createVariable(selected->getElementType(), selected->dims(),
                                  "selected", Variable::VisibilityKind::Public);
  auto *softmax = F->createSoftMax("softmax", var1, var2);
  auto result = F->createSave("ret", softmax);
  EE.compile(CompilationMode::Infer, F);
  EE.run({var1, var2}, {inputs, selected});
  out->copyFrom(&result->getVariable()->getPayload());
}

void trainSoftMaxNet(Tensor *inputs, Tensor *weights, Tensor *bias,
                     Tensor *selected, Tensor *out, BackendKind kind) {
  ExecutionEngine EE(kind);
  EE.getConfig().learningRate = 0.003;
  EE.getConfig().momentum = 0.7;
  EE.getConfig().L2Decay = 0.001;
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  auto *var1 = mod.createVariable(inputs->getElementType(), inputs->dims(),
                                  "input", Variable::VisibilityKind::Public);
  auto *var2 = mod.createVariable(selected->getElementType(), selected->dims(),
                                  "selected", Variable::VisibilityKind::Public,
                                  Variable::TrainKind::None);
  auto *fc = F->createFullyConnected("fc", var1, bias->dims()[0]);
  cast<Variable>(fc->getWeights())->copyFrom(weights);
  cast<Variable>(fc->getBias())->copyFrom(bias);
  auto *softmax = F->createSoftMax("softmax", fc, var2);
  auto result = F->createSave("ret", softmax);

  Function *TF = glow::differentiate(F, EE.getConfig());
  EE.compile(CompilationMode::Train, TF);

  EE.runBatch(30, {var1, var2}, {inputs, selected});
  EE.compile(CompilationMode::Infer, F);
  EE.run({var1, var2}, {inputs, selected});
  out->copyFrom(&result->getVariable()->getPayload());
}

void inferTanhNet(Tensor *inputs, Tensor *out, BackendKind kind) {
  ExecutionEngine EE(kind);
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  auto *var = mod.createVariable(inputs->getElementType(), inputs->dims(),
                                 "input", Variable::VisibilityKind::Public);
  auto *tanh = F->createTanh("tanh", var);
  auto result = F->createSave("ret", tanh);
  EE.compile(CompilationMode::Infer, F);
  EE.run({var}, {inputs});
  out->copyFrom(&result->getVariable()->getPayload());
}

void inferBasicConvNet(Tensor *inputs, Tensor *out, BackendKind kind,
                       size_t convDepth) {
  ExecutionEngine EE(kind);
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  auto *var = mod.createVariable(inputs->getElementType(), inputs->dims(),
                                 "input", Variable::VisibilityKind::Public);
  auto *tr = F->createTranspose("tr", var, {0, 2, 3, 1});
  auto *conv = F->createConv("conv", tr, convDepth, 5, 2, 1);
  cast<Variable>(conv->getFilter())->getHandle().clear(2);
  cast<Variable>(conv->getBias())->getHandle().clear(2);
  auto *pool = F->createPoolMax("pool", conv, 2, 2, 0);
  auto result = F->createSave("ret", pool);
  EE.compile(CompilationMode::Infer, F);
  EE.run({var}, {inputs});
  out->copyFrom(&result->getVariable()->getPayload());
}

void inferBasicFCNet(Tensor *inputs, Tensor *out, BackendKind kind) {
  ExecutionEngine EE(kind);
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  auto *var = mod.createVariable(inputs->getElementType(), inputs->dims(),
                                 "input", Variable::VisibilityKind::Public);
  auto *tr = F->createTranspose("tr", var, {0, 2, 3, 1});
  auto *fc = F->createFullyConnected("fc", tr, 16);
  auto *rl0 = F->createRELU("relu", fc);
  auto *fc2 = F->createFullyConnected("fc2", rl0, 8);
  auto *rl1 = F->createRELU("relu", fc);
  cast<Variable>(fc->getWeights())->getHandle().clear(0.8);
  cast<Variable>(fc2->getWeights())->getHandle().clear(1.5);
  auto result = F->createSave("ret", rl1);
  EE.compile(CompilationMode::Infer, F);
  EE.run({var}, {inputs});
  out->copyFrom(&result->getVariable()->getPayload());
}

void inferMixedNet(Tensor *inputs, Tensor *out, BackendKind kind) {
  ExecutionEngine EE(kind);
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  auto *var = mod.createVariable(inputs->getElementType(), inputs->dims(),
                                 "input", Variable::VisibilityKind::Public);
  auto *selected = mod.createVariable(ElemKind::IndexTy, {2, 1}, "selected");

  auto *tr = F->createTranspose("tr", var, {0, 2, 3, 1});
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
  EE.run({var}, {inputs});
  out->copyFrom(&result->getVariable()->getPayload());
}

void inferComplexNet1(Tensor *inputs1, Tensor *inputs2, Tensor *inputs3,
                      Tensor *inputs4, Tensor *out, BackendKind kind) {
  ExecutionEngine EE(kind);
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  auto *var1 = mod.createVariable(inputs1->getElementType(), inputs1->dims(),
                                  "inputs1", Variable::VisibilityKind::Public);
  auto *var2 = mod.createVariable(inputs2->getElementType(), inputs2->dims(),
                                  "inputs2", Variable::VisibilityKind::Public);
  auto *var3 = mod.createVariable(inputs3->getElementType(), inputs3->dims(),
                                  "inputs3", Variable::VisibilityKind::Public);
  auto *var4 = mod.createVariable(inputs4->getElementType(), inputs4->dims(),
                                  "inputs4", Variable::VisibilityKind::Public);
  auto *conv1 = F->createConv("conv1", var1, 6, 4, 1, 2);
  cast<Variable>(conv1->getFilter())->getHandle().clear(0.5);
  cast<Variable>(conv1->getBias())->getHandle().clear(0.7);
  auto *sigmoid1 = F->createSigmoid("sigmoid1", conv1);
  auto *fc1 = F->createFullyConnected("fc1", var2, 2352);
  cast<Variable>(fc1->getWeights())->getHandle().clear(0.6);
  auto *reshape1 = F->createReshape("reshape1", fc1, {8, 14, 28, 6});
  auto *relu1 = F->createRELU("relu1", reshape1);
  auto *pool1 = F->createPoolMax("pool1", relu1, 2, 2, 1);
  auto *add = F->createAdd("add", sigmoid1, pool1);
  auto *tanh = F->createTanh("tanh", add);
  auto *fc2 = F->createFullyConnected("fc2", var3, 720);
  cast<Variable>(fc2->getWeights())->getHandle().clear(1.1);
  auto *reshape2 = F->createReshape("reshape2", fc2, {8, 8, 15, 6});
  auto *mul = F->createMul("mul", tanh, reshape2);
  auto *sigmoid2 = F->createSigmoid("sigmoid2", mul);
  auto *conv2 = F->createConv("conv2", sigmoid2, 7, 3, 2, 1);
  cast<Variable>(conv2->getFilter())->getHandle().clear(0.3);
  cast<Variable>(conv2->getBias())->getHandle().clear(1.3);
  auto *reshape3 = F->createReshape("reshape3", conv2, {8, 8, 7, 4});
  auto *sub = F->createSub("sub", reshape3, var4);
  auto *relu2 = F->createRELU("relu2", sub);
  auto *pool2 = F->createPoolAvg("pool2", relu2, 3, 2, 1);
  auto *sigmoid3 = F->createSigmoid("sigmoid3", pool2);
  auto result = F->createSave("ret", sigmoid3);
  EE.compile(CompilationMode::Infer, F);
  EE.run({var1, var2, var3, var4}, {inputs1, inputs2, inputs3, inputs4});
  out->copyFrom(&result->getVariable()->getPayload());
}

} // namespace glow
