// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Graph/Graph.h"
#include "glow/IR/IR.h"
#include "glow/IR/IRBuilder.h"
#include "glow/IR/Instrs.h"

using namespace glow;
using llvm::cast;

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
