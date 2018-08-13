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

#include "glow/Base/Tensor.h"
#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Graph/Graph.h"
#include "glow/Graph/Nodes.h"
#include "glow/IR/IR.h"
#include "glow/IR/IRBuilder.h"
#include "glow/IR/Instrs.h"

#include "gtest/gtest.h"

using namespace glow;

class GradCheckBase : public ::testing::TestWithParam<BackendKind> {
public:
  ExecutionEngine EE_{GetParam()};
};

class InterpreterGrad : public GradCheckBase {};
class GradCheck : public GradCheckBase {};

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
/// \param EE Execution engine to compile/run network.
/// \param result Node that contains result of f(x).
/// \param inputVar Variable which gradient is assessed.
/// \param expVar Variable with expected value, only used during the training.
/// \param inputs Tensor for \p inputVar variable.
/// \param outputs Tensor for \p expVar variable.
/// \param allowedError allowed delta between analytical and numerical
///                     gradients.
void performGradCheck(ExecutionEngine &EE, SaveNode *result, Variable *inputVar,
                      Variable *expVar, Tensor *inputs, Tensor *outputs,
                      float delta, float allowedError) {
  TrainingConfig TC;
  auto &F = *EE.getModule().getFunction("main");

  // Create a function that trains the network.
  Function *TF = glow::differentiate(&F, TC);
  EE.compile(CompilationMode::Train, TF);

  // The network might have variables, other than inputVar and expVar.
  // Train the network until other variables reach some stable local minimum.
  EE.runBatch(300, {inputVar, expVar}, {inputs, outputs});

  // Create a version of the network that records the gradients to some side
  // table instead of updating them.
  VariableGradientsList varGrads;
  Function *recordNet = glow::differentiate(&F, TC, "record", &varGrads);
  EE.compile(CompilationMode::Train, recordNet);

  // Clear the gradients of the first layer.
  auto gradVar = getGrad(varGrads, inputVar);
  gradVar->getPayload().zero();

  // Train the network just once to record the values of gradient for inputVar.
  EE.runBatch(1, {inputVar, expVar}, {inputs, outputs});

  // Compile the original network in inference mode.
  EE.compile(CompilationMode::Infer, &F);

  auto analyticalGradsH = gradVar->getPayload().getHandle();
  auto inputsH = inputs->getHandle<>();
  for (size_t i = 0; i < analyticalGradsH.size(); i++) {
    auto old = inputsH.raw(i);
    Tensor &res = result->getVariable()->getPayload();

    // Calculate f(x+e):
    inputsH.raw(i) = old + delta;
    EE.run({inputVar}, {inputs});
    auto plusLoss = computeL2Loss(outputs, &res);

    // Calculate f(x-e):
    inputsH.raw(i) = old - delta;
    EE.run({inputVar}, {inputs});
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

TEST_P(InterpreterGrad, gradientCheckFCConcatRELU) {
  size_t numInputElem = 20;
  size_t numOutputElem = 10;

  auto &mod = EE_.getModule();
  Function *F = mod.createFunction("main");
  auto *A = mod.createVariable(ElemKind::FloatTy, {1, numInputElem}, "A",
                               VisibilityKind::Public, false);
  auto *Exp = mod.createVariable(ElemKind::FloatTy, {1, numOutputElem}, "exp",
                                 VisibilityKind::Public, false);

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

  inputsH.initXavier(1, mod.getPRNG());
  outputsH.initXavier(1, mod.getPRNG());

  performGradCheck(EE_, result, A, Exp, &inputs, &outputs, 0.0001, 0.001);
}

static void gradientCheckGroupConv(size_t depth, size_t group,
                                   ExecutionEngine &EE_) {
  size_t numDim = 10;

  auto &mod = EE_.getModule();
  Function *F = mod.createFunction("main");
  auto *A = mod.createVariable(ElemKind::FloatTy, {1, numDim, numDim, depth},
                               "A", VisibilityKind::Public, false);
  auto *Ex =
      mod.createVariable(ElemKind::FloatTy, {1, numDim + 1, numDim + 1, depth},
                         "exp", VisibilityKind::Public, false);

  Node *O = F->createConv("conv", A, depth, 2, 1, 1, group);
  O = F->createRegression("reg", O, Ex);
  auto *result = F->createSave("ret", O);

  Tensor inputs(ElemKind::FloatTy, {1, numDim, numDim, depth});
  Tensor outputs(ElemKind::FloatTy, {1, numDim + 1, numDim + 1, depth});

  auto inputsH = inputs.getHandle<>();
  auto outputsH = outputs.getHandle<>();

  inputsH.randomize(-1, 1, mod.getPRNG());
  outputsH.randomize(-1, 1, mod.getPRNG());

  performGradCheck(EE_, result, A, Ex, &inputs, &outputs, 0.001, 0.04);
}

TEST_P(GradCheck, gradientCheckConv) { gradientCheckGroupConv(1, 1, EE_); }

TEST_P(GradCheck, gradientCheckDepthwiseConv) {
  gradientCheckGroupConv(4, 4, EE_);
}

TEST_P(GradCheck, gradientCheckGroupConv) { gradientCheckGroupConv(4, 2, EE_); }

TEST_P(InterpreterGrad, gradientCheckAvgPool) {
  size_t numDim = 10;
  size_t numOutputElem = 10;

  auto &mod = EE_.getModule();
  Function *F = mod.createFunction("main");
  auto *A = mod.createVariable(ElemKind::FloatTy, {1, numDim, numDim, 1}, "A",
                               VisibilityKind::Public, false);
  auto *Exp = mod.createVariable(ElemKind::FloatTy, {1, numOutputElem}, "Exp",
                                 VisibilityKind::Public, false);

  Node *O = F->createAvgPool("pool", A, 3, 3, 1);
  O = F->createTanh("tanh", O);
  O = F->createFullyConnected("fc", O, numOutputElem);
  O = F->createRegression("reg", O, Exp);
  auto *result = F->createSave("ret", O);

  Tensor inputs(ElemKind::FloatTy, {1, numDim, numDim, 1});
  Tensor outputs(ElemKind::FloatTy, {1, numOutputElem});

  auto inputsH = inputs.getHandle<>();
  auto outputsH = outputs.getHandle<>();

  inputsH.initXavier(1, mod.getPRNG());
  outputsH.initXavier(1, mod.getPRNG());

  performGradCheck(EE_, result, A, Exp, &inputs, &outputs, 0.001, 0.004);
}

TEST_P(InterpreterGrad, gradientCheckBatchNorm) {
  size_t numDim = 5;
  size_t numOutputElem = numDim * numDim * 3;

  auto &mod = EE_.getModule();
  Function *F = mod.createFunction("main");
  auto *A = mod.createVariable(ElemKind::FloatTy, {1, numDim, numDim, 3}, "A",
                               VisibilityKind::Public, false);
  auto *Ex = mod.createVariable(ElemKind::FloatTy, {1, numOutputElem}, "exp",
                                VisibilityKind::Public, false);

  Node *O = F->createBatchNormalization("batch", A, 3, 0.0001, 0.9);
  O = F->createReshape("reshape", O, {1, numDim * numDim * 3});
  O = F->createRegression("reg", O, Ex);
  auto result = F->createSave("ret", O);

  Tensor inputs(ElemKind::FloatTy, {1, numDim, numDim, 3});
  Tensor outputs(ElemKind::FloatTy, {1, numOutputElem});

  auto inputsH = inputs.getHandle<>();
  auto outputsH = outputs.getHandle<>();

  inputsH.initXavier(1, mod.getPRNG());
  outputsH.initXavier(1, mod.getPRNG());

  for (int i = 0, e = inputsH.size(); i < e; i++) {
    inputsH.raw(i) *= 6;
    inputsH.raw(i) += 4;
  }

  performGradCheck(EE_, result, A, Ex, &inputs, &outputs, 0.001, 0.004);
}

TEST_P(InterpreterGrad, gradientCheckArithmeticDiv) {
  // The test creates a net: A / B = Exp. Where A is trainable weight,
  // B and Exp are external data (initialized randomly once). SGD will find
  // correct value for A, and then gradient check will be performed.
  size_t numDim = 10;

  auto &mod = EE_.getModule();
  Function *F = mod.createFunction("main");
  auto *A = mod.createVariable(ElemKind::FloatTy, {1, numDim}, "A",
                               VisibilityKind::Public, true);
  auto *B = mod.createVariable(ElemKind::FloatTy, {1, numDim}, "B",
                               VisibilityKind::Public, false);
  auto *Exp = mod.createVariable(ElemKind::FloatTy, {1, numDim}, "exp",
                                 VisibilityKind::Public, false);

  A->getPayload().init(Tensor::InitKind::Xavier, 1.0, mod.getPRNG());

  Node *O = F->createDiv("div", A, B);
  O = F->createRegression("reg", O, Exp);
  auto *result = F->createSave("ret", O);

  Tensor BValues(ElemKind::FloatTy, {1, numDim});
  Tensor ExpValues(ElemKind::FloatTy, {1, numDim});
  // Random values are in the range, so that all intermediate computations are
  // not too small and not too large.
  BValues.getHandle().randomize(0.1, 1, mod.getPRNG());
  ExpValues.getHandle().randomize(0.1, 1, mod.getPRNG());

  performGradCheck(EE_, result, B, Exp, &BValues, &ExpValues, 0.0001, 0.001);
}

TEST_P(InterpreterGrad, gradientCheckArithmetic) {
  size_t numDim = 20;

  auto &mod = EE_.getModule();
  Function *F = mod.createFunction("main");
  auto *A = mod.createVariable(ElemKind::FloatTy, {1, numDim}, "A",
                               VisibilityKind::Public, false);
  auto *B = mod.createVariable(ElemKind::FloatTy, {1, numDim}, "B",
                               VisibilityKind::Public, false);
  auto *C = mod.createVariable(ElemKind::FloatTy, {1, numDim}, "C",
                               VisibilityKind::Public, false);
  auto *D = mod.createVariable(ElemKind::FloatTy, {1, numDim}, "D",
                               VisibilityKind::Public, false);
  auto *E = mod.createVariable(ElemKind::FloatTy, {1, numDim}, "E",
                               VisibilityKind::Public, false);
  // Randomize E to avoid div by zero.
  E->getPayload().getHandle().initXavier(1, mod.getPRNG());

  auto *Exp = mod.createVariable(ElemKind::FloatTy, {1, numDim}, "exp",
                                 VisibilityKind::Public, false);

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

  inputsH.initXavier(1, mod.getPRNG());
  outputsH.initXavier(1, mod.getPRNG());

  performGradCheck(EE_, result, A, Exp, &inputs, &outputs, 0.01, 0.004);
}

TEST_P(InterpreterGrad, gradientCheckFCConcatTanh) {
  // Using the same gradient check test setup as gradientCheck_FC_Concat_RELU
  size_t numInputElem = 20;
  size_t numOutputElem = 10;

  auto &mod = EE_.getModule();
  Function *F = mod.createFunction("main");
  auto *A = mod.createVariable(ElemKind::FloatTy, {1, numInputElem}, "A",
                               VisibilityKind::Public, false);
  auto *Exp = mod.createVariable(ElemKind::FloatTy, {1, numOutputElem}, "Exp",
                                 VisibilityKind::Public, false);

  Node *FA = F->createFullyConnected("fc", A, numOutputElem);
  FA = F->createTanh("tanh", FA);
  FA = F->createRegression("reg", FA, Exp);
  auto *result = F->createSave("ret", FA);

  Tensor inputs(ElemKind::FloatTy, {{1, numInputElem}});
  Tensor outputs(ElemKind::FloatTy, {{1, numOutputElem}});

  auto inputsH = inputs.getHandle<>();
  auto outputsH = outputs.getHandle<>();

  inputsH.initXavier(1, mod.getPRNG());
  outputsH.initXavier(1, mod.getPRNG());

  performGradCheck(EE_, result, A, Exp, &inputs, &outputs, 0.0001, 0.001);
}

TEST_P(InterpreterGrad, gradientCheckFC) {
  size_t numInputElem = 20;
  size_t numOutputElem = 10;

  auto &mod = EE_.getModule();
  Function *F = mod.createFunction("main");
  auto *A = mod.createVariable(ElemKind::FloatTy, {1, numInputElem}, "A",
                               VisibilityKind::Public, false);
  auto *Exp = mod.createVariable(ElemKind::FloatTy, {1, numOutputElem}, "Exp",
                                 VisibilityKind::Public, false);

  Node *FA = F->createFullyConnected("fc", A, numOutputElem);
  FA = F->createRegression("reg", FA, Exp);
  auto *result = F->createSave("ret", FA);

  Tensor inputs(ElemKind::FloatTy, {{1, numInputElem}});
  Tensor outputs(ElemKind::FloatTy, {{1, numOutputElem}});

  auto inputsH = inputs.getHandle<>();
  auto outputsH = outputs.getHandle<>();

  inputsH.initXavier(1, mod.getPRNG());
  outputsH.initXavier(1, mod.getPRNG());

  performGradCheck(EE_, result, A, Exp, &inputs, &outputs, 0.0001, 0.0001);
}

TEST_P(InterpreterGrad, gradientCheckSigmoid) {
  size_t numInputElem = 20;
  size_t numOutputElem = 10;

  auto &mod = EE_.getModule();
  Function *F = mod.createFunction("main");
  auto *A = mod.createVariable(ElemKind::FloatTy, {1, numInputElem}, "A",
                               VisibilityKind::Public, false);
  auto *Exp = mod.createVariable(ElemKind::FloatTy, {1, numOutputElem}, "Exp",
                                 VisibilityKind::Public, false);
  F->createSave("ret", A);

  Node *FA = F->createSigmoid("sig", Exp);
  FA = F->createRegression("reg", FA, Exp);
  auto *result = F->createSave("ret", FA);

  Tensor inputs(ElemKind::FloatTy, {{1, numInputElem}});
  Tensor outputs(ElemKind::FloatTy, {{1, numOutputElem}});

  auto inputsH = inputs.getHandle<>();
  auto outputsH = outputs.getHandle<>();

  inputsH.initXavier(1, mod.getPRNG());
  outputsH.initXavier(1, mod.getPRNG());

  performGradCheck(EE_, result, A, Exp, &inputs, &outputs, 0.0001, 0.001);
}

TEST_P(InterpreterGrad, gradientCheckRelu) {
  size_t numInputElem = 20;
  size_t numOutputElem = 10;

  auto &mod = EE_.getModule();
  Function *F = mod.createFunction("main");
  auto *A = mod.createVariable(ElemKind::FloatTy, {1, numInputElem}, "A",
                               VisibilityKind::Public, false);
  auto *Exp = mod.createVariable(ElemKind::FloatTy, {1, numOutputElem}, "Exp",
                                 VisibilityKind::Public, false);
  F->createSave("ret", A);

  Node *FA = F->createRELU("relu", Exp);
  FA = F->createRegression("reg", FA, Exp);
  auto *result = F->createSave("ret", FA);

  Tensor inputs(ElemKind::FloatTy, {{1, numInputElem}});
  Tensor outputs(ElemKind::FloatTy, {{1, numOutputElem}});

  auto inputsH = inputs.getHandle<>();
  auto outputsH = outputs.getHandle<>();

  inputsH.initXavier(1, mod.getPRNG());
  outputsH.initXavier(1, mod.getPRNG());

  performGradCheck(EE_, result, A, Exp, &inputs, &outputs, 0.0001, 0.001);
}

TEST_P(InterpreterGrad, gradientCheckTranspose) {
  // Using the same gradient check test setup as gradientCheck_FC_Concat_RELU
  size_t numOutputElem = 10;

  auto &mod = EE_.getModule();
  Function *F = mod.createFunction("main");
  auto *A = mod.createVariable(ElemKind::FloatTy, {1, 5, 10, 5}, "input",
                               VisibilityKind::Public, false);
  auto *Exp = mod.createVariable(ElemKind::FloatTy, {1, numOutputElem}, "exp",
                                 VisibilityKind::Public, false);
  Node *TA = F->createTranspose("transpose", A, NHWC2NCHW);
  TA = F->createFullyConnected("fc", TA, numOutputElem);
  TA = F->createRegression("regress", TA, Exp);
  auto *result = F->createSave("ret", TA);

  Tensor inputs(ElemKind::FloatTy, {1, 5, 10, 5});
  Tensor outputs(ElemKind::FloatTy, {1, numOutputElem});

  auto inputsH = inputs.getHandle<>();
  auto outputsH = outputs.getHandle<>();

  inputsH.randomize(-1, 1, mod.getPRNG());
  outputsH.randomize(-1, 1, mod.getPRNG());

  performGradCheck(EE_, result, A, Exp, &inputs, &outputs, 0.0001, 0.001);
}

TEST_P(InterpreterGrad, gradientCheckCrossEntropyLoss) {
  const size_t batchSize = 6;
  const int testSamples = 5;
  const float stepSize = 1e-4;
  const float delta = 0.015;
  TrainingConfig TC;

  auto &mod = EE_.getModule();
  Function *F = mod.createFunction("main");
  auto *P = mod.createVariable(ElemKind::FloatTy, {batchSize, 4}, "P",
                               VisibilityKind::Public, false);
  auto *Y = mod.createVariable(ElemKind::IndexTy, {batchSize}, "Labels",
                               VisibilityKind::Public, false);
  auto *L = mod.createVariable(ElemKind::FloatTy, {1}, "L",
                               VisibilityKind::Public, false);
  Node *CE = F->createCrossEntropyLoss("celoss", P, Y);
  F->createSave("ret", CE, L);

  Tensor inputs(ElemKind::FloatTy, {batchSize, 4});
  Tensor outputs(ElemKind::IndexTy, {batchSize});

  auto inputsH = inputs.getHandle();
  auto outputsH = outputs.getHandle<size_t>();

  inputsH.randomize(0.0, 1.0, mod.getPRNG());
  outputsH.at({0}) = 2;
  outputsH.at({1}) = 0;
  outputsH.at({2}) = 1;

  VariableGradientsList varGrads;
  Function *TF = glow::differentiate(F, TC, "record", &varGrads);
  EE_.compile(CompilationMode::Train, TF);

  auto gradP = getGrad(varGrads, P)->getHandle();

  for (int i = 0; i < testSamples; ++i) {
    inputsH.randomize(0.0, 1.0, mod.getPRNG());
    for (size_t j = 0; j < inputsH.size(); ++j) {
      EE_.run({P, Y}, {&inputs, &outputs});
      L->getPayload().zero();
      auto x = inputsH.raw(j);
      auto g = gradP.raw(j);
      inputsH.raw(j) = x + stepSize;
      EE_.run({P, Y}, {&inputs, &outputs});
      auto lp = L->getHandle().raw(0);
      inputsH.raw(j) = x - stepSize;
      L->getPayload().zero();
      EE_.run({P, Y}, {&inputs, &outputs});
      auto lm = L->getHandle().raw(0);
      auto diff = (lp - lm) / (2 * stepSize);
      inputsH.raw(j) = x;
      EE_.run({P, Y}, {&inputs, &outputs});
      EXPECT_NEAR(diff, g, delta);
    }
  }
}

INSTANTIATE_TEST_CASE_P(Interpreter, InterpreterGrad,
                        ::testing::Values(BackendKind::Interpreter));

INSTANTIATE_TEST_CASE_P(Interpreter, GradCheck,
                        ::testing::Values(BackendKind::Interpreter));

#ifdef GLOW_WITH_CPU
INSTANTIATE_TEST_CASE_P(JIT, GradCheck, ::testing::Values(BackendKind::CPU));
#endif // GLOW_WITH_CPU
