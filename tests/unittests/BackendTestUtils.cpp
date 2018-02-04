// Copyright 2017-2018 Facebook. All Rights Reserved.

#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Graph/Graph.h"
#include "glow/IR/IR.h"
#include "glow/IR/IRBuilder.h"
#include "glow/IR/Instrs.h"

using namespace glow;
using llvm::cast;

void inferBatchedAddNet(Tensor *inputs1, Tensor *inputs2, Tensor *out,
                        BackendKind kind) {
  ExecutionEngine EE(kind);
  auto &G = EE.getGraph();
  auto *var1 = G.createVariable(inputs1->getElementType(), inputs1->dims(),
                                "input1", Variable::VisibilityKind::Public);
  auto *var2 = G.createVariable(inputs2->getElementType(), inputs2->dims(),
                                "input2", Variable::VisibilityKind::Public);
  auto *batchedadd = G.createBatchedArithmetic(
      "batchedadd", BatchedArithmeticNode::Mode::Add, var1, var2);
  auto result = G.createSave("ret", batchedadd);
  EE.compile(CompilationMode::Infer);
  EE.run({var1, var2}, {inputs1, inputs2});
  out->copyFrom(&result->getVariable()->getPayload());
}

void inferBatchedReduceAddNet(Tensor *inputs1, Tensor *out, BackendKind kind) {
  ExecutionEngine EE(kind);
  auto &G = EE.getGraph();
  auto *var1 = G.createVariable(inputs1->getElementType(), inputs1->dims(),
                                "input1", Variable::VisibilityKind::Public);
  auto *batchedreduce = G.createBatchedReduce(
      "batchedreduce", BatchedReduceNode::Mode::Add, var1);
  auto result = G.createSave("ret", batchedreduce);
  EE.compile(CompilationMode::Infer);
  EE.run({var1}, {inputs1});
  out->copyFrom(&result->getVariable()->getPayload());
}

void inferMaxNet(Tensor *inputs1, Tensor *inputs2, Tensor *out,
                 BackendKind kind) {
  ExecutionEngine EE(kind);
  auto &G = EE.getGraph();
  auto *var1 = G.createVariable(inputs1->getElementType(), inputs1->dims(),
                                "input1", Variable::VisibilityKind::Public);
  auto *var2 = G.createVariable(inputs2->getElementType(), inputs2->dims(),
                                "input2", Variable::VisibilityKind::Public);
  auto *max = G.createArithmetic("max", var1, var2, ArithmeticNode::Mode::Max);
  auto result = G.createSave("ret", max);
  EE.compile(CompilationMode::Infer);
  EE.run({var1, var2}, {inputs1, inputs2});
  out->copyFrom(&result->getVariable()->getPayload());
}

void inferMinNet(Tensor *inputs1, Tensor *inputs2, Tensor *out,
                 BackendKind kind) {
  ExecutionEngine EE(kind);
  auto &G = EE.getGraph();
  auto *var1 = G.createVariable(inputs1->getElementType(), inputs1->dims(),
                                "input1", Variable::VisibilityKind::Public);
  auto *var2 = G.createVariable(inputs2->getElementType(), inputs2->dims(),
                                "input2", Variable::VisibilityKind::Public);
  auto *min = G.createArithmetic("min", var1, var2, ArithmeticNode::Mode::Min);
  auto result = G.createSave("ret", min);
  EE.compile(CompilationMode::Infer);
  EE.run({var1, var2}, {inputs1, inputs2});
  out->copyFrom(&result->getVariable()->getPayload());
}

void inferReluNet(Tensor *inputs, Tensor *out, BackendKind kind) {
  ExecutionEngine EE(kind);
  auto &G = EE.getGraph();
  auto *var = G.createVariable(inputs->getElementType(), inputs->dims(),
                               "input", Variable::VisibilityKind::Public);
  auto *relu = G.createRELU("relu", var);
  auto result = G.createSave("ret", relu);
  EE.compile(CompilationMode::Infer);
  EE.run({var}, {inputs});
  out->copyFrom(&result->getVariable()->getPayload());
}

void inferReshapeNet(Tensor *inputs, llvm::ArrayRef<size_t> shape, Tensor *out,
                     BackendKind kind) {
  ExecutionEngine EE(kind);
  auto &G = EE.getGraph();
  auto *var = G.createVariable(inputs->getElementType(), inputs->dims(),
                               "input", Variable::VisibilityKind::Public);
  auto *reshape = G.createReshape("reshape", var, shape);
  auto result = G.createSave("ret", reshape);
  EE.compile(CompilationMode::Infer);
  EE.run({var}, {inputs});
  out->copyFrom(&result->getVariable()->getPayload());
}

void inferSelectNet(Tensor *cond, Tensor *inputs1, Tensor *inputs2, Tensor *out,
                    BackendKind kind) {
  ExecutionEngine EE(kind);
  auto &G = EE.getGraph();
  auto *var1 = G.createVariable(cond->getElementType(), cond->dims(), "cond",
                                Variable::VisibilityKind::Public);
  auto *var2 = G.createVariable(inputs1->getElementType(), inputs1->dims(),
                                "input1", Variable::VisibilityKind::Public);
  auto *var3 = G.createVariable(inputs2->getElementType(), inputs2->dims(),
                                "input2", Variable::VisibilityKind::Public);
  auto *select = G.createSelect("cond", var1, var2, var3);
  auto result = G.createSave("ret", select);
  EE.compile(CompilationMode::Infer);
  EE.run({var1, var2, var3}, {cond, inputs1, inputs2});
  out->copyFrom(&result->getVariable()->getPayload());
}

void inferSigmoidNet(Tensor *inputs, Tensor *out, BackendKind kind) {
  ExecutionEngine EE(kind);
  auto &G = EE.getGraph();
  auto *var = G.createVariable(inputs->getElementType(), inputs->dims(),
                               "input", Variable::VisibilityKind::Public);
  auto *sigmoid = G.createSigmoid("sigmoid", var);
  auto result = G.createSave("ret", sigmoid);
  EE.compile(CompilationMode::Infer);
  EE.run({var}, {inputs});
  out->copyFrom(&result->getVariable()->getPayload());
}

void inferSoftMaxNet(Tensor *inputs, Tensor *selected, Tensor *out,
                     BackendKind kind) {
  ExecutionEngine EE(kind);
  auto &G = EE.getGraph();
  auto *var1 = G.createVariable(inputs->getElementType(), inputs->dims(),
                                "input", Variable::VisibilityKind::Public);
  auto *var2 = G.createVariable(selected->getElementType(), selected->dims(),
                                "selected", Variable::VisibilityKind::Public);
  auto *softmax = G.createSoftMax("softmax", var1, var2);
  auto result = G.createSave("ret", softmax);
  EE.compile(CompilationMode::Infer);
  EE.run({var1, var2}, {inputs, selected});
  out->copyFrom(&result->getVariable()->getPayload());
}

void trainSoftMaxNet(Tensor *inputs, Tensor *selected, Tensor *out,
                     BackendKind kind) {
  ExecutionEngine EE(kind);
  EE.getConfig().learningRate = 0.003;
  EE.getConfig().maxNumThreads = 1;
  EE.getConfig().momentum = 0.7;
  EE.getConfig().L2Decay = 0.001;
  auto &G = EE.getGraph();
  auto *var1 = G.createVariable(inputs->getElementType(), inputs->dims(),
                                "input", Variable::VisibilityKind::Public);
  auto *var2 = G.createVariable(selected->getElementType(), selected->dims(),
                                "selected", Variable::VisibilityKind::Public,
                                Variable::TrainKind::None);
  auto *softmax = G.createSoftMax("softmax", var1, var2);
  auto result = G.createSave("ret", softmax);
  EE.compile(CompilationMode::Train);
  EE.runBatch(50, {var1, var2}, {inputs, selected});
  EE.compile(CompilationMode::Infer);
  EE.run({var1}, {inputs});
  out->copyFrom(&result->getVariable()->getPayload());
}

void inferTanhNet(Tensor *inputs, Tensor *out, BackendKind kind) {
  ExecutionEngine EE(kind);
  auto &G = EE.getGraph();
  auto *var = G.createVariable(inputs->getElementType(), inputs->dims(),
                               "input", Variable::VisibilityKind::Public);
  auto *tanh = G.createTanh("tanh", var);
  auto result = G.createSave("ret", tanh);
  EE.compile(CompilationMode::Infer);
  EE.run({var}, {inputs});
  out->copyFrom(&result->getVariable()->getPayload());
}

void inferBasicConvNet(Tensor *inputs, Tensor *out, BackendKind kind) {
  ExecutionEngine EE(kind);
  auto &G = EE.getGraph();
  auto *var = G.createVariable(inputs->getElementType(), inputs->dims(),
                               "input", Variable::VisibilityKind::Public);
  auto *tr = G.createTranspose("tr", var, {0, 2, 3, 1});
  auto *conv = G.createConv("conv", tr, 4, 5, 2, 1);
  cast<Variable>(conv->getFilter())->getHandle().clear(2);
  cast<Variable>(conv->getBias())->getHandle().clear(2);
  auto *pool = G.createPool("pool", conv, PoolNode::Mode::Max, 2, 2, 0);
  auto result = G.createSave("ret", pool);
  EE.compile(CompilationMode::Infer);
  EE.run({var}, {inputs});
  out->copyFrom(&result->getVariable()->getPayload());
}

void inferBasicFCNet(Tensor *inputs, Tensor *out, BackendKind kind) {
  ExecutionEngine EE(kind);
  auto &G = EE.getGraph();
  auto *var = G.createVariable(inputs->getElementType(), inputs->dims(),
                               "input", Variable::VisibilityKind::Public);
  auto *tr = G.createTranspose("tr", var, {0, 2, 3, 1});
  auto *fc = G.createFullyConnected("fc", tr, 16);
  auto *rl0 = G.createRELU("relu", fc);
  auto *fc2 = G.createFullyConnected("fc2", rl0, 8);
  auto *rl1 = G.createRELU("relu", fc);
  cast<Variable>(fc->getFilter())->getHandle().clear(0.8);
  cast<Variable>(fc2->getFilter())->getHandle().clear(1.5);
  auto result = G.createSave("ret", rl1);
  EE.compile(CompilationMode::Infer);
  EE.run({var}, {inputs});
  out->copyFrom(&result->getVariable()->getPayload());
}

void inferMixedNet(Tensor *inputs, Tensor *out, BackendKind kind) {
  ExecutionEngine EE(kind);
  auto &G = EE.getGraph();
  auto *var = G.createVariable(inputs->getElementType(), inputs->dims(),
                               "input", Variable::VisibilityKind::Public);
  auto *selected = G.createVariable(ElemKind::IndexTy, {2, 1}, "selected");

  auto *tr = G.createTranspose("tr", var, {0, 2, 3, 1});
  auto *fc = G.createFullyConnected("fc", tr, 16);
  auto *th0 = G.createTanh("tanh", fc);
  auto *sg0 = G.createSigmoid("sig", fc);
  auto *A1 = G.createArithmetic("add", th0, sg0, ArithmeticNode::Mode::Add);
  auto *fc2 = G.createFullyConnected("fc2", A1, 16);

  auto *R = G.createRegression("reg", fc2, fc2);
  auto *SM = G.createSoftMax("SM", R, selected);
  auto result = G.createSave("ret", SM);

  cast<Variable>(fc->getFilter())->getHandle().clear(0.4);
  cast<Variable>(fc2->getFilter())->getHandle().clear(3.5);

  EE.compile(CompilationMode::Infer);
  EE.run({var}, {inputs});
  out->copyFrom(&result->getVariable()->getPayload());
}

void inferComplexNet1(Tensor *inputs1, Tensor *inputs2, Tensor *inputs3,
                      Tensor *inputs4, Tensor *out, BackendKind kind) {
  ExecutionEngine EE(kind);
  auto &G = EE.getGraph();
  auto *var1 = G.createVariable(inputs1->getElementType(), inputs1->dims(),
                                "inputs1", Variable::VisibilityKind::Public);
  auto *var2 = G.createVariable(inputs2->getElementType(), inputs2->dims(),
                                "inputs2", Variable::VisibilityKind::Public);
  auto *var3 = G.createVariable(inputs3->getElementType(), inputs3->dims(),
                                "inputs3", Variable::VisibilityKind::Public);
  auto *var4 = G.createVariable(inputs4->getElementType(), inputs4->dims(),
                                "inputs4", Variable::VisibilityKind::Public);
  auto *conv1 = G.createConv("conv1", var1, 6, 4, 1, 2);
  cast<Variable>(conv1->getFilter())->getHandle().clear(0.5);
  cast<Variable>(conv1->getBias())->getHandle().clear(0.7);
  auto *sigmoid1 = G.createSigmoid("sigmoid1", conv1);
  auto *fc1 = G.createFullyConnected("fc1", var2, 2352);
  cast<Variable>(fc1->getFilter())->getHandle().clear(0.6);
  auto *reshape1 = G.createReshape("reshape1", fc1, {8, 14, 28, 6});
  auto *relu1 = G.createRELU("relu1", reshape1);
  auto *pool1 = G.createPool("pool1", relu1, PoolNode::Mode::Max, 2, 2, 1);
  auto *add =
      G.createArithmetic("add", sigmoid1, pool1, ArithmeticNode::Mode::Add);
  auto *tanh = G.createTanh("tanh", add);
  auto *fc2 = G.createFullyConnected("fc2", var3, 720);
  cast<Variable>(fc2->getFilter())->getHandle().clear(1.1);
  auto *reshape2 = G.createReshape("reshape2", fc2, {8, 8, 15, 6});
  auto *mul =
      G.createArithmetic("mul", tanh, reshape2, ArithmeticNode::Mode::Mul);
  auto *sigmoid2 = G.createSigmoid("sigmoid2", mul);
  auto *conv2 = G.createConv("conv2", sigmoid2, 7, 3, 2, 1);
  cast<Variable>(conv2->getFilter())->getHandle().clear(0.3);
  cast<Variable>(conv2->getBias())->getHandle().clear(1.3);
  auto *reshape3 = G.createReshape("reshape3", conv2, {8, 8, 7, 4});
  auto *sub =
      G.createArithmetic("sub", reshape3, var4, ArithmeticNode::Mode::Sub);
  auto *relu2 = G.createRELU("relu2", sub);
  auto *pool2 = G.createPool("pool2", relu2, PoolNode::Mode::Avg, 3, 2, 1);
  auto *sigmoid3 = G.createSigmoid("sigmoid3", pool2);
  auto result = G.createSave("ret", sigmoid3);
  EE.compile(CompilationMode::Infer);
  EE.run({var1, var2, var3, var4}, {inputs1, inputs2, inputs3, inputs4});
  out->copyFrom(&result->getVariable()->getPayload());
}
