// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "glow/Base/Tensor.h"
#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Graph/Graph.h"
#include "glow/Graph/Nodes.h"
#include "glow/IR/IR.h"
#include "glow/IR/IRBuilder.h"
#include "glow/IR/Instrs.h"

#include "gtest/gtest.h"

using namespace glow;

/// Compute the regression loss for the tensor \p X with regard to Y.
float computeL2Loss(Tensor *X, Tensor *Y) {
  assert(X->dims() == Y->dims() && "Invalid input dims");
  auto xH = X->getHandle<>();
  auto yH = Y->getHandle<>();
  float loss = 0;

  for (size_t i = 0, e = X->size(); i < e; i++) {
    float dy = (xH.raw(i) - yH.raw(i));
    loss += 0.5 * dy * dy;
  }

  return loss;
}

/// \returns the error rate when comparing two grads.
float gradDiff(float G1, float G2) {
  return std::abs(G1 - G2) / std::abs(G1 + G2 + 1);
}

Variable *getGrad(Graph &G, Variable *V) { return G.getGradientVariable(V); }

void performGradCheck(ExecutionEngine &IP, SaveNode *result, Variable *inputVar,
                      Variable *expVar, Tensor *inputs, Tensor *outputs,
                      float delta, float allowedError) {
  auto inputsH = inputs->getHandle<>();

  // Train the network.
  IP.runBatch(300, {inputVar, expVar}, {inputs, outputs});
  Graph &G = IP.getGraph();

  // Remove the SGD nodes by compiling in Inference mode. This compilation will
  // keep the "grad" nodes.
  IP.compile(CompilationMode::Infer);

  // Clear the gradients of the first layer.
  auto gradVar = getGrad(G, inputVar);
  gradVar->getPayload().zero();

  // Train the network just once to calculate the grads.
  IP.runBatch(1, {inputVar, expVar}, {inputs, outputs});

  // Copy the gradient buffer. Future iterations will invalidate the buffer.
  Tensor gradCopy = gradVar->getPayload().clone();
  auto analyticalGradsH = gradCopy.getHandle();

  for (size_t i = 0; i < analyticalGradsH.size(); i++) {
    auto old = inputsH.raw(i);

    // Calculate f(x+e):
    inputsH.raw(i) = old + delta;
    IP.run({inputVar}, {inputs});
    Tensor &res = result->getVariable()->getPayload();
    auto plusLoss = computeL2Loss(outputs, &res);

    // Calculate f(x-e):
    inputsH.raw(i) = old - delta;
    IP.run({inputVar}, {inputs});
    auto minusLoss = computeL2Loss(outputs, &res);
    inputsH.raw(i) = old;

    auto numericGrad = (plusLoss - minusLoss) / (2 * delta);
    auto analyticalGrad = analyticalGradsH.raw(i);

    auto err = gradDiff(analyticalGrad, numericGrad);

    // Make sure that the analytical and numerical gradients agree.
    EXPECT_LE(err, allowedError);
  }
}

TEST(Network, gradientCheck_FC_Concat_RELU) {
  ExecutionEngine IP;
  IP.getConfig().maxNumThreads = 1;

  size_t numInputElem = 20;
  size_t numOutputElem = 10;

  auto &G = IP.getGraph();

  auto *A = G.createVariable(ElemKind::FloatTy, {1, numInputElem}, "A",
                             Variable::InitKind::Extern);
  auto *Exp = G.createVariable(ElemKind::FloatTy, {1, numOutputElem}, "exp",
                               Variable::InitKind::Extern);

  Node *FA = G.createFullyConnected("fc1", A, numOutputElem / 2);
  FA = G.createRELU("relu1", FA);

  auto *B = G.createVariable(ElemKind::FloatTy, {1, numInputElem}, "B");
  Node *FB = G.createFullyConnected("fc2", B, numOutputElem / 2);
  FB = G.createRELU("relu2", FB);

  Node *O = G.createConcat("concat", {FA, FB}, 1);
  O = G.createRegression("reg", O, Exp);
  auto *result = G.createSave("ret", O);

  IP.compile(CompilationMode::TrainDebug);

  Tensor inputs(ElemKind::FloatTy, {{1, numInputElem}});
  Tensor outputs(ElemKind::FloatTy, {{1, numOutputElem}});

  auto inputsH = inputs.getHandle<>();
  auto outputsH = outputs.getHandle<>();

  inputsH.randomize(100);
  outputsH.randomize(100);

  performGradCheck(IP, result, A, Exp, &inputs, &outputs, 0.001, 0.001);
}

TEST(Network, gradientCheck_Conv) {
  ExecutionEngine IP;
  IP.getConfig().maxNumThreads = 1;

  size_t numDim = 10;
  size_t numOutputElem = 10;

  auto &G = IP.getGraph();

  auto *A = G.createVariable(ElemKind::FloatTy, {1, numDim, numDim, 1}, "A",
                             Variable::InitKind::Extern);
  auto *Ex = G.createVariable(ElemKind::FloatTy, {1, numOutputElem}, "exp",
                              Variable::InitKind::Extern);

  Node *O = G.createConv("conv", A, 4, 5, 1, 2);
  O = G.createPool("pool", O, PoolNode::Mode::Max, 3, 3, 0);
  O = G.createFullyConnected("fc", O, numOutputElem);
  O = G.createRELU("relu", O);
  O = G.createRegression("reg", O, Ex);
  auto *result = G.createSave("ret", O);

  IP.compile(CompilationMode::TrainDebug);

  Tensor inputs(ElemKind::FloatTy, {1, numDim, numDim, 1});
  Tensor outputs(ElemKind::FloatTy, {1, numOutputElem});

  auto inputsH = inputs.getHandle<>();
  auto outputsH = outputs.getHandle<>();

  inputsH.randomize(1);
  outputsH.randomize(1);

  performGradCheck(IP, result, A, Ex, &inputs, &outputs, 0.001, 0.004);
}

TEST(Network, gradientCheck_AvgPool) {
  ExecutionEngine IP;
  IP.getConfig().maxNumThreads = 1;

  size_t numDim = 10;
  size_t numOutputElem = 10;

  auto &G = IP.getGraph();

  auto *A = G.createVariable(ElemKind::FloatTy, {1, numDim, numDim, 1}, "A",
                             Variable::InitKind::Extern);
  auto *Exp = G.createVariable(ElemKind::FloatTy, {1, numOutputElem}, "Exp",
                               Variable::InitKind::Extern);

  Node *O = G.createPool("pool", A, PoolNode::Mode::Avg, 3, 3, 0);
  O = G.createFullyConnected("fc", O, numOutputElem);
  O = G.createRegression("reg", O, Exp);
  auto *result = G.createSave("ret", O);

  IP.compile(CompilationMode::TrainDebug);

  Tensor inputs(ElemKind::FloatTy, {1, numDim, numDim, 1});
  Tensor outputs(ElemKind::FloatTy, {1, numOutputElem});

  auto inputsH = inputs.getHandle<>();
  auto outputsH = outputs.getHandle<>();

  inputsH.randomize(1);
  outputsH.randomize(1);

  performGradCheck(IP, result, A, Exp, &inputs, &outputs, 0.001, 0.004);
}

TEST(Network, gradientCheck_batchNorm) {
  ExecutionEngine IP;
  IP.getConfig().maxNumThreads = 1;

  size_t numDim = 5;
  size_t numOutputElem = numDim * numDim * 3;

  auto &G = IP.getGraph();

  auto *A = G.createVariable(ElemKind::FloatTy, {1, numDim, numDim, 3}, "A",
                             Variable::InitKind::Extern);
  auto *Ex = G.createVariable(ElemKind::FloatTy, {1, numOutputElem}, "exp",
                              Variable::InitKind::Extern);

  Node *O = G.createBatchNormalization("batch", A, 3, 0.0001, 0.9);
  O = G.createReshape("reshape", O, {1, numDim * numDim * 3});
  O = G.createRegression("reg", O, Ex);
  auto result = G.createSave("ret", O);

  IP.compile(CompilationMode::TrainDebug);

  Tensor inputs(ElemKind::FloatTy, {1, numDim, numDim, 3});
  Tensor outputs(ElemKind::FloatTy, {1, numOutputElem});

  auto inputsH = inputs.getHandle<>();
  auto outputsH = outputs.getHandle<>();

  inputsH.randomize(1);
  outputsH.randomize(1);

  for (int i = 0, e = inputsH.size(); i < e; i++) {
    inputsH.raw(i) *= 6;
    inputsH.raw(i) += 4;
  }

  performGradCheck(IP, result, A, Ex, &inputs, &outputs, 0.001, 0.004);
}

TEST(Network, gradientCheck_Arithmetic) {
  ExecutionEngine IP;
  IP.getConfig().maxNumThreads = 1;

  size_t numDim = 20;

  auto &G = IP.getGraph();

  auto *A = G.createVariable(ElemKind::FloatTy, {1, numDim}, "A",
                             Variable::InitKind::Extern);
  auto *B = G.createVariable(ElemKind::FloatTy, {1, numDim}, "B",
                             Variable::InitKind::Extern, 0.1);
  auto *C = G.createVariable(ElemKind::FloatTy, {1, numDim}, "C",
                             Variable::InitKind::Extern, 0.1);
  auto *D = G.createVariable(ElemKind::FloatTy, {1, numDim}, "D",
                             Variable::InitKind::Extern, 0.1);
  auto *E = G.createVariable(ElemKind::FloatTy, {1, numDim}, "E",
                             Variable::InitKind::Broadcast, 0.1);

  auto *Exp = G.createVariable(ElemKind::FloatTy, {1, numDim}, "exp",
                               Variable::InitKind::Extern);

  Node *O = G.createArithmetic("mul", A, B, ArithmeticNode::Mode::Mul);
  O = G.createArithmetic("add", O, C, ArithmeticNode::Mode::Add);
  O = G.createArithmetic("sub", D, O, ArithmeticNode::Mode::Sub);
  O = G.createArithmetic("div", O, E, ArithmeticNode::Mode::Div);
  O = G.createRegression("reg", O, Exp);
  auto *result = G.createSave("ret", O);

  IP.compile(CompilationMode::TrainDebug);

  Tensor inputs(ElemKind::FloatTy, {1, numDim});
  Tensor outputs(ElemKind::FloatTy, {1, numDim});

  auto inputsH = inputs.getHandle<>();
  auto outputsH = outputs.getHandle<>();

  inputsH.randomize(1);
  outputsH.randomize(1);

  performGradCheck(IP, result, A, Exp, &inputs, &outputs, 0.01, 0.004);
}

TEST(Network, gradientCheck_FC_Concat_Tanh) {
  // Using the same gradient check test setup as gradientCheck_FC_Concat_RELU
  ExecutionEngine IP;
  IP.getConfig().maxNumThreads = 1;

  size_t numInputElem = 20;
  size_t numOutputElem = 10;

  auto &G = IP.getGraph();
  auto *A = G.createVariable(ElemKind::FloatTy, {1, numInputElem}, "A",
                             Variable::InitKind::Extern);
  auto *Exp = G.createVariable(ElemKind::FloatTy, {1, numOutputElem}, "Exp",
                               Variable::InitKind::Extern);

  Node *FA = G.createFullyConnected("fc", A, numOutputElem);
  FA = G.createTanh("tanh", FA);
  FA = G.createRegression("reg", FA, Exp);
  auto *result = G.createSave("ret", FA);

  IP.compile(CompilationMode::TrainDebug);

  Tensor inputs(ElemKind::FloatTy, {{1, numInputElem}});
  Tensor outputs(ElemKind::FloatTy, {{1, numOutputElem}});

  auto inputsH = inputs.getHandle<>();
  auto outputsH = outputs.getHandle<>();

  inputsH.randomize(100);
  outputsH.randomize(100);

  performGradCheck(IP, result, A, Exp, &inputs, &outputs, 0.0001, 0.001);
}

TEST(Network, gradientCheck_Transpose) {
  // Using the same gradient check test setup as gradientCheck_FC_Concat_RELU
  ExecutionEngine IP;
  IP.getConfig().maxNumThreads = 1;
  size_t numOutputElem = 10;

  auto &G = IP.getGraph();

  auto *A = G.createVariable(ElemKind::FloatTy, {1, 5, 10, 5}, "input",
                             Variable::InitKind::Extern);
  auto *Exp = G.createVariable(ElemKind::FloatTy, {1, numOutputElem}, "exp",
                               Variable::InitKind::Extern);
  Node *TA = G.createTranspose("transpose", A, {0, 3, 1, 2});
  TA = G.createFullyConnected("fc", TA, numOutputElem);
  TA = G.createRegression("regress", TA, Exp);
  auto *result = G.createSave("ret", TA);

  IP.compile(CompilationMode::TrainDebug);

  Tensor inputs(ElemKind::FloatTy, {1, 5, 10, 5});
  Tensor outputs(ElemKind::FloatTy, {1, numOutputElem});

  auto inputsH = inputs.getHandle<>();
  auto outputsH = outputs.getHandle<>();

  inputsH.randomize(100);
  outputsH.randomize(100);

  performGradCheck(IP, result, A, Exp, &inputs, &outputs, 0.0001, 0.001);
}
