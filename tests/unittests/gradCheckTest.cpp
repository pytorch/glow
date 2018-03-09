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

/// \returns the regression loss for the tensor \p X with regard to \p Y.
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

/// \returns the error when comparing two grads: absolute or relative.
float gradDiff(float G1, float G2) {
  return std::min(std::abs(G1 - G2), std::abs(G1 - G2) / std::abs(G1 + G2 + 1));
}

Variable *getGrad(const VariableGradientsList &grads, Variable *V) {
  for (auto &p : grads) {
    if (p.first == V) {
      return p.second;
    }
  }
  return nullptr;
}

/// Performs gradient check by comparing analytical and numerical gradients.
/// Numeric grad is calculated based on: f(x-delta) and f(x+delta) values.
/// Analytical grad is based on the gradient output calculated during back
/// propagation.
///
/// \param IP Execution engine to compile/run network.
/// \param result Node that contains result of f(x).
/// \param inputVar Variable which gradient is assessed.
/// \param expVar Variable with expected value, only used during the training.
/// \param inputs Tensor for \p inputVar variable.
/// \param outputs Tensor for \p expVar variable.
/// \param allowedError allowed delta between analytical and numerical
///                     gradients.
void performGradCheck(ExecutionEngine &IP, SaveNode *result, Variable *inputVar,
                      Variable *expVar, Tensor *inputs, Tensor *outputs,
                      float delta, float allowedError) {
  auto &F = *IP.getModule().getFunction("main");

  // Create a function that trains the network.
  Function *TF = glow::differentiate(&F, IP.getConfig());
  IP.compile(CompilationMode::Train, TF);

  // The network might have variables, other than inputVar and expVar.
  // Train the network until other variables reach some stable local minimum.
  IP.runBatch(300, {inputVar, expVar}, {inputs, outputs});

  // Create a version of the network that records the gradients to some side
  // table instead of updating them.
  VariableGradientsList varGrads;
  Function *recordNet =
      glow::differentiate(&F, IP.getConfig(), "record", &varGrads);
  IP.compile(CompilationMode::Train, recordNet);

  // Clear the gradients of the first layer.
  auto gradVar = getGrad(varGrads, inputVar);
  gradVar->getPayload().zero();

  // Train the network just once to record the values of gradient for inputVar.
  IP.runBatch(1, {inputVar, expVar}, {inputs, outputs});

  // Compile the original network in inference mode.
  IP.compile(CompilationMode::Infer, &F);

  auto analyticalGradsH = gradVar->getPayload().getHandle();
  auto inputsH = inputs->getHandle<>();
  for (size_t i = 0; i < analyticalGradsH.size(); i++) {
    auto old = inputsH.raw(i);
    Tensor &res = result->getVariable()->getPayload();

    // Calculate f(x+e):
    inputsH.raw(i) = old + delta;
    IP.run({inputVar}, {inputs});
    auto plusLoss = computeL2Loss(outputs, &res);

    // Calculate f(x-e):
    inputsH.raw(i) = old - delta;
    IP.run({inputVar}, {inputs});
    auto minusLoss = computeL2Loss(outputs, &res);

    // Restore value back.
    inputsH.raw(i) = old;

    auto numericGrad = (plusLoss - minusLoss) / (2 * delta);
    auto analyticalGrad = analyticalGradsH.raw(i);

    auto err = gradDiff(analyticalGrad, numericGrad);

    // Make sure that the analytical and numerical gradients agree.
    EXPECT_LE(err, allowedError);
  }
}

TEST(Network, gradientCheckFCConcatRELU) {
  ExecutionEngine IP;

  size_t numInputElem = 20;
  size_t numOutputElem = 10;

  auto &mod = IP.getModule();
  Function *F = mod.createFunction("main");
  auto *A = mod.createVariable(ElemKind::FloatTy, {1, numInputElem}, "A",
                               Variable::VisibilityKind::Public,
                               Variable::TrainKind::None);
  auto *Exp = mod.createVariable(ElemKind::FloatTy, {1, numOutputElem}, "exp",
                                 Variable::VisibilityKind::Public,
                                 Variable::TrainKind::None);

  Node *FA = F->createFullyConnected("fc1", A, numOutputElem / 2);
  FA = F->createRELU("relu1", FA);

  auto *B = mod.createVariable(ElemKind::FloatTy, {1, numInputElem}, "B");
  Node *FB = F->createFullyConnected("fc2", B, numOutputElem / 2);
  FB = F->createRELU("relu2", FB);

  Node *O = F->createConcat("concat", {FA, FB}, 1);
  O = F->createRegression("reg", O, Exp);
  auto *result = F->createSave("ret", O);

  Tensor inputs(ElemKind::FloatTy, {{1, numInputElem}});
  Tensor outputs(ElemKind::FloatTy, {{1, numOutputElem}});

  auto inputsH = inputs.getHandle<>();
  auto outputsH = outputs.getHandle<>();

  inputsH.initXavier(1);
  outputsH.initXavier(1);

  performGradCheck(IP, result, A, Exp, &inputs, &outputs, 0.0001, 0.001);
}

TEST(Network, gradientCheckConv) {
  ExecutionEngine IP;

  size_t numDim = 10;
  size_t numOutputElem = 10;

  auto &mod = IP.getModule();
  Function *F = mod.createFunction("main");
  auto *A = mod.createVariable(ElemKind::FloatTy, {1, numDim, numDim, 1}, "A",
                               Variable::VisibilityKind::Public,
                               Variable::TrainKind::None);
  auto *Ex = mod.createVariable(ElemKind::FloatTy, {1, numOutputElem}, "exp",
                                Variable::VisibilityKind::Public,
                                Variable::TrainKind::None);

  Node *O = F->createConv("conv", A, 4, 5, 1, 2);
  O = F->createPoolMax("pool", O, 3, 3, 0);
  O = F->createFullyConnected("fc", O, numOutputElem);
  O = F->createRELU("relu", O);
  O = F->createRegression("reg", O, Ex);
  auto *result = F->createSave("ret", O);

  Tensor inputs(ElemKind::FloatTy, {1, numDim, numDim, 1});
  Tensor outputs(ElemKind::FloatTy, {1, numOutputElem});

  auto inputsH = inputs.getHandle<>();
  auto outputsH = outputs.getHandle<>();

  inputsH.initXavier(1);
  outputsH.initXavier(1);

  performGradCheck(IP, result, A, Ex, &inputs, &outputs, 0.001, 0.004);
}

TEST(Network, gradientCheckAvgPool) {
  ExecutionEngine IP;

  size_t numDim = 10;
  size_t numOutputElem = 10;

  auto &mod = IP.getModule();
  Function *F = mod.createFunction("main");
  auto *A = mod.createVariable(ElemKind::FloatTy, {1, numDim, numDim, 1}, "A",
                               Variable::VisibilityKind::Public,
                               Variable::TrainKind::None);
  auto *Exp = mod.createVariable(ElemKind::FloatTy, {1, numOutputElem}, "Exp",
                                 Variable::VisibilityKind::Public,
                                 Variable::TrainKind::None);

  Node *O = F->createPoolAvg("pool", A, 3, 3, 1);
  O = F->createTanh("tanh", O);
  O = F->createFullyConnected("fc", O, numOutputElem);
  O = F->createRegression("reg", O, Exp);
  auto *result = F->createSave("ret", O);

  Tensor inputs(ElemKind::FloatTy, {1, numDim, numDim, 1});
  Tensor outputs(ElemKind::FloatTy, {1, numOutputElem});

  auto inputsH = inputs.getHandle<>();
  auto outputsH = outputs.getHandle<>();

  inputsH.initXavier(1);
  outputsH.initXavier(1);

  performGradCheck(IP, result, A, Exp, &inputs, &outputs, 0.001, 0.004);
}

TEST(Network, gradientCheckBatchNorm) {
  ExecutionEngine IP;

  size_t numDim = 5;
  size_t numOutputElem = numDim * numDim * 3;

  auto &mod = IP.getModule();
  Function *F = mod.createFunction("main");
  auto *A = mod.createVariable(ElemKind::FloatTy, {1, numDim, numDim, 3}, "A",
                               Variable::VisibilityKind::Public,
                               Variable::TrainKind::None);
  auto *Ex = mod.createVariable(ElemKind::FloatTy, {1, numOutputElem}, "exp",
                                Variable::VisibilityKind::Public,
                                Variable::TrainKind::None);

  Node *O = F->createBatchNormalization("batch", A, 3, 0.0001, 0.9);
  O = F->createReshape("reshape", O, {1, numDim * numDim * 3});
  O = F->createRegression("reg", O, Ex);
  auto result = F->createSave("ret", O);

  Tensor inputs(ElemKind::FloatTy, {1, numDim, numDim, 3});
  Tensor outputs(ElemKind::FloatTy, {1, numOutputElem});

  auto inputsH = inputs.getHandle<>();
  auto outputsH = outputs.getHandle<>();

  inputsH.initXavier(1);
  outputsH.initXavier(1);

  for (int i = 0, e = inputsH.size(); i < e; i++) {
    inputsH.raw(i) *= 6;
    inputsH.raw(i) += 4;
  }

  performGradCheck(IP, result, A, Ex, &inputs, &outputs, 0.001, 0.004);
}

TEST(Network, gradientCheckArithmeticDiv) {
  // The test creates a net: A / B = Exp. Where A is trainable weight,
  // B and Exp are external data (initialized randomly once). SGD will find
  // correct value for A, and then gradient check will be performed.
  ExecutionEngine IP;
  size_t numDim = 10;

  auto &mod = IP.getModule();
  Function *F = mod.createFunction("main");
  auto *A = mod.createVariable(ElemKind::FloatTy, {1, numDim}, "A",
                               Variable::VisibilityKind::Public,
                               Variable::TrainKind::Xavier, 1);
  auto *B = mod.createVariable(ElemKind::FloatTy, {1, numDim}, "B",
                               Variable::VisibilityKind::Public,
                               Variable::TrainKind::None);
  auto *Exp = mod.createVariable(ElemKind::FloatTy, {1, numDim}, "exp",
                                 Variable::VisibilityKind::Public,
                                 Variable::TrainKind::None);
  Node *O = F->createDiv("div", A, B);
  O = F->createRegression("reg", O, Exp);
  auto *result = F->createSave("ret", O);

  Tensor BValues(ElemKind::FloatTy, {1, numDim});
  Tensor ExpValues(ElemKind::FloatTy, {1, numDim});
  // Random values are in the range, so that all intermediate computations are
  // not too small and not too large.
  BValues.getHandle().randomize(0.1, 1);
  ExpValues.getHandle().randomize(0.1, 1);

  performGradCheck(IP, result, B, Exp, &BValues, &ExpValues, 0.0001, 0.001);
}

TEST(Network, gradientCheckArithmetic) {
  ExecutionEngine IP;

  size_t numDim = 20;

  auto &mod = IP.getModule();
  Function *F = mod.createFunction("main");
  auto *A = mod.createVariable(ElemKind::FloatTy, {1, numDim}, "A",
                               Variable::VisibilityKind::Public,
                               Variable::TrainKind::None);
  auto *B = mod.createVariable(ElemKind::FloatTy, {1, numDim}, "B",
                               Variable::VisibilityKind::Public,
                               Variable::TrainKind::None, 0.1);
  auto *C = mod.createVariable(ElemKind::FloatTy, {1, numDim}, "C",
                               Variable::VisibilityKind::Public,
                               Variable::TrainKind::None, 0.1);
  auto *D = mod.createVariable(ElemKind::FloatTy, {1, numDim}, "D",
                               Variable::VisibilityKind::Public,
                               Variable::TrainKind::None, 0.1);
  auto *E = mod.createVariable(ElemKind::FloatTy, {1, numDim}, "E",
                               Variable::VisibilityKind::Public,
                               Variable::TrainKind::None);
  // Randomize E to avoid div by zero.
  E->getPayload().getHandle().initXavier(1);

  auto *Exp = mod.createVariable(ElemKind::FloatTy, {1, numDim}, "exp",
                                 Variable::VisibilityKind::Public,
                                 Variable::TrainKind::None);

  Node *O = F->createMul("mul", A, B);
  O = F->createAdd("add", O, C);
  O = F->createSub("sub", D, O);
  O = F->createDiv("div", O, E);
  O = F->createRegression("reg", O, Exp);
  auto *result = F->createSave("ret", O);

  Tensor inputs(ElemKind::FloatTy, {1, numDim});
  Tensor outputs(ElemKind::FloatTy, {1, numDim});

  auto inputsH = inputs.getHandle<>();
  auto outputsH = outputs.getHandle<>();

  inputsH.initXavier(1);
  outputsH.initXavier(1);

  performGradCheck(IP, result, A, Exp, &inputs, &outputs, 0.01, 0.004);
}

TEST(Network, gradientCheckFCConcatTanh) {
  // Using the same gradient check test setup as gradientCheck_FC_Concat_RELU
  ExecutionEngine IP;

  size_t numInputElem = 20;
  size_t numOutputElem = 10;

  auto &mod = IP.getModule();
  Function *F = mod.createFunction("main");
  auto *A = mod.createVariable(ElemKind::FloatTy, {1, numInputElem}, "A",
                               Variable::VisibilityKind::Public,
                               Variable::TrainKind::None);
  auto *Exp = mod.createVariable(ElemKind::FloatTy, {1, numOutputElem}, "Exp",
                                 Variable::VisibilityKind::Public,
                                 Variable::TrainKind::None);

  Node *FA = F->createFullyConnected("fc", A, numOutputElem);
  FA = F->createTanh("tanh", FA);
  FA = F->createRegression("reg", FA, Exp);
  auto *result = F->createSave("ret", FA);

  Tensor inputs(ElemKind::FloatTy, {{1, numInputElem}});
  Tensor outputs(ElemKind::FloatTy, {{1, numOutputElem}});

  auto inputsH = inputs.getHandle<>();
  auto outputsH = outputs.getHandle<>();

  inputsH.initXavier(1);
  outputsH.initXavier(1);

  performGradCheck(IP, result, A, Exp, &inputs, &outputs, 0.0001, 0.001);
}

TEST(Network, gradientCheckFC) {
  ExecutionEngine IP;

  size_t numInputElem = 20;
  size_t numOutputElem = 10;

  auto &mod = IP.getModule();
  Function *F = mod.createFunction("main");
  auto *A = mod.createVariable(ElemKind::FloatTy, {1, numInputElem}, "A",
                               Variable::VisibilityKind::Public,
                               Variable::TrainKind::None);
  auto *Exp = mod.createVariable(ElemKind::FloatTy, {1, numOutputElem}, "Exp",
                                 Variable::VisibilityKind::Public,
                                 Variable::TrainKind::None);

  Node *FA = F->createFullyConnected("fc", A, numOutputElem);
  FA = F->createRegression("reg", FA, Exp);
  auto *result = F->createSave("ret", FA);

  Tensor inputs(ElemKind::FloatTy, {{1, numInputElem}});
  Tensor outputs(ElemKind::FloatTy, {{1, numOutputElem}});

  auto inputsH = inputs.getHandle<>();
  auto outputsH = outputs.getHandle<>();

  inputsH.initXavier(1);
  outputsH.initXavier(1);

  performGradCheck(IP, result, A, Exp, &inputs, &outputs, 0.0001, 0.0001);
}

TEST(Network, gradientCheckSigmoid) {
  ExecutionEngine IP;

  size_t numInputElem = 20;
  size_t numOutputElem = 10;

  auto &mod = IP.getModule();
  Function *F = mod.createFunction("main");
  auto *A = mod.createVariable(ElemKind::FloatTy, {1, numInputElem}, "A",
                               Variable::VisibilityKind::Public,
                               Variable::TrainKind::None);
  auto *Exp = mod.createVariable(ElemKind::FloatTy, {1, numOutputElem}, "Exp",
                                 Variable::VisibilityKind::Public,
                                 Variable::TrainKind::None);
  F->createSave("ret", A);

  Node *FA = F->createSigmoid("sig", Exp);
  FA = F->createRegression("reg", FA, Exp);
  auto *result = F->createSave("ret", FA);

  Tensor inputs(ElemKind::FloatTy, {{1, numInputElem}});
  Tensor outputs(ElemKind::FloatTy, {{1, numOutputElem}});

  auto inputsH = inputs.getHandle<>();
  auto outputsH = outputs.getHandle<>();

  inputsH.initXavier(1);
  outputsH.initXavier(1);

  performGradCheck(IP, result, A, Exp, &inputs, &outputs, 0.0001, 0.001);
}

TEST(Network, gradientCheckRelu) {
  ExecutionEngine IP;

  size_t numInputElem = 20;
  size_t numOutputElem = 10;

  auto &mod = IP.getModule();
  Function *F = mod.createFunction("main");
  auto *A = mod.createVariable(ElemKind::FloatTy, {1, numInputElem}, "A",
                               Variable::VisibilityKind::Public,
                               Variable::TrainKind::None);
  auto *Exp = mod.createVariable(ElemKind::FloatTy, {1, numOutputElem}, "Exp",
                                 Variable::VisibilityKind::Public,
                                 Variable::TrainKind::None);
  F->createSave("ret", A);

  Node *FA = F->createRELU("relu", Exp);
  FA = F->createRegression("reg", FA, Exp);
  auto *result = F->createSave("ret", FA);

  Tensor inputs(ElemKind::FloatTy, {{1, numInputElem}});
  Tensor outputs(ElemKind::FloatTy, {{1, numOutputElem}});

  auto inputsH = inputs.getHandle<>();
  auto outputsH = outputs.getHandle<>();

  inputsH.initXavier(1);
  outputsH.initXavier(1);

  performGradCheck(IP, result, A, Exp, &inputs, &outputs, 0.0001, 0.001);
}

TEST(Network, gradientCheckTranspose) {
  // Using the same gradient check test setup as gradientCheck_FC_Concat_RELU
  ExecutionEngine IP;
  size_t numOutputElem = 10;

  auto &mod = IP.getModule();
  Function *F = mod.createFunction("main");
  auto *A = mod.createVariable(ElemKind::FloatTy, {1, 5, 10, 5}, "input",
                               Variable::VisibilityKind::Public,
                               Variable::TrainKind::None);
  auto *Exp = mod.createVariable(ElemKind::FloatTy, {1, numOutputElem}, "exp",
                                 Variable::VisibilityKind::Public,
                                 Variable::TrainKind::None);
  Node *TA = F->createTranspose("transpose", A, {0, 3, 1, 2});
  TA = F->createFullyConnected("fc", TA, numOutputElem);
  TA = F->createRegression("regress", TA, Exp);
  auto *result = F->createSave("ret", TA);

  Tensor inputs(ElemKind::FloatTy, {1, 5, 10, 5});
  Tensor outputs(ElemKind::FloatTy, {1, numOutputElem});

  auto inputsH = inputs.getHandle<>();
  auto outputsH = outputs.getHandle<>();

  inputsH.randomize(-1, 1);
  outputsH.randomize(-1, 1);

  performGradCheck(IP, result, A, Exp, &inputs, &outputs, 0.0001, 0.001);
}

TEST(Network, gradientCheckCrossEntropyLoss) {
  ExecutionEngine IP;
  const int batchSize = 6;
  const int testSamples = 5;
  const float stepSize = 1e-4;
  const float delta = 0.015;

  auto &mod = IP.getModule();
  Function *F = mod.createFunction("main");
  auto *P = mod.createVariable(ElemKind::FloatTy, {batchSize, 4}, "P",
                               Variable::VisibilityKind::Public,
                               Variable::TrainKind::None);
  auto *Y = mod.createVariable(ElemKind::IndexTy, {batchSize}, "Labels",
                               Variable::VisibilityKind::Public,
                               Variable::TrainKind::None);
  auto *L = mod.createVariable(ElemKind::FloatTy, {1}, "L",
                               Variable::VisibilityKind::Public,
                               Variable::TrainKind::None);
  Node *CE = F->createCrossEntropyLoss("celoss", P, Y);
  F->createSave("ret", CE, L);

  Tensor inputs(ElemKind::FloatTy, {batchSize, 4});
  Tensor outputs(ElemKind::IndexTy, {batchSize});

  auto inputsH = inputs.getHandle();
  auto outputsH = outputs.getHandle<size_t>();

  inputsH.randomize(0.0, 1.0);
  outputsH.at({0}) = 2;
  outputsH.at({1}) = 0;
  outputsH.at({2}) = 1;

  VariableGradientsList varGrads;
  Function *TF = glow::differentiate(F, IP.getConfig(), "record", &varGrads);
  IP.compile(CompilationMode::Train, TF);

  auto gradP = getGrad(varGrads, P)->getHandle();

  for (int i = 0; i < testSamples; ++i) {
    inputsH.randomize(0.0, 1.0);
    for (size_t j = 0; j < inputsH.size(); ++j) {
      IP.run({P, Y}, {&inputs, &outputs});
      L->getPayload().zero();
      auto x = inputsH.raw(j);
      auto g = gradP.raw(j);
      inputsH.raw(j) = x + stepSize;
      IP.run({P, Y}, {&inputs, &outputs});
      auto lp = L->getHandle().raw(0);
      inputsH.raw(j) = x - stepSize;
      L->getPayload().zero();
      IP.run({P, Y}, {&inputs, &outputs});
      auto lm = L->getHandle().raw(0);
      auto diff = (lp - lm) / (2 * stepSize);
      inputsH.raw(j) = x;
      IP.run({P, Y}, {&inputs, &outputs});
      EXPECT_NEAR(diff, g, delta);
    }
  }
}
