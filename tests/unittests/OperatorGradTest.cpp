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

#include "BackendTestUtils.h"

#include "glow/Base/Tensor.h"
#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Graph/Graph.h"
#include "glow/Graph/Nodes.h"

#include "gtest/gtest.h"

using namespace glow;

class OperatorGradTest : public BackendTest {
protected:
  /// Compute gradients of placeholders in bindings_. Given
  /// outputNode, representing H(Vars), this function will train H(vars) to be
  /// 0. It is achieved by creating RegressionNode between outputNode and 0,
  /// which will minimize F, a total squared error of H divided by 2.
  ///
  /// Note that gradient value at the start of backpropagation is the same as
  /// outputNode's forward value, because RegressionNode's grad is outputNode-0.
  ///
  /// \param outputNode Node that contains result of H(Vars).
  VariableGradientsList computeVarGrads(Node *outputNode) {
    auto *Exp = mod_.createPlaceholder(ElemKind::FloatTy, outputNode->dims(0),
                                       "exp", false);
    bindings_.allocate(Exp)->zero();

    auto *reg = F_->createRegression("reg", outputNode, Exp);
    auto *result = F_->createSave("ret", reg);
    bindings_.allocate(result->getPlaceholder());

    // Create a version of the network that records the gradients to some side
    // table instead of updating them.
    VariableGradientsList varGrads;
    TrainingConfig TC;
    Function *recordNet = glow::differentiate(F_, TC, "record", &varGrads);
    allocateGrads(varGrads);
    EE_.compile(CompilationMode::Train, recordNet);

    // Train the network just once to record the values of gradient for
    // all variables.
    EE_.run(bindings_);

    return varGrads;
  }

  Tensor *getGradTensor(const VariableGradientsList &grads, Placeholder *V) {
    for (auto &p : grads) {
      if (p.first == V) {
        return bindings_.get(p.second);
      }
    }
    return nullptr;
  }

  void allocateGrads(const VariableGradientsList &grads) {
    for (auto &p : grads) {
      auto grad = p.second;
      bindings_.allocate(grad);
    }
  }

  PlaceholderBindings bindings_;
};

INSTANTIATE_TEST_CASE_P_FOR_BACKEND_TEST(OperatorGradTest, OperatorGradTest);

TEST_P(OperatorGradTest, concat) {
  ENABLED_BACKENDS(Interpreter);

  size_t numOutputElem = 4;

  auto *A = mod_.createPlaceholder(ElemKind::FloatTy, {1, numOutputElem / 2},
                                   "A", false);
  bindings_.allocate(A)->getHandle() = {1, 2};
  auto *B = mod_.createPlaceholder(ElemKind::FloatTy, {1, numOutputElem / 2},
                                   "B", false);
  bindings_.allocate(B)->getHandle() = {3, 4};

  Node *O = F_->createConcat("concat", {A, B}, 1);
  VariableGradientsList varGrads = computeVarGrads(O);

  Tensor expectedLeft(ElemKind::FloatTy, {1, numOutputElem / 2});
  expectedLeft.getHandle() = {1, 2};
  EXPECT_TRUE(expectedLeft.isEqual(*getGradTensor(varGrads, A)));

  Tensor expectedRight(ElemKind::FloatTy, {1, numOutputElem / 2});
  expectedRight.getHandle() = {3, 4};
  EXPECT_TRUE(expectedRight.isEqual(*getGradTensor(varGrads, B)));
}

TEST_P(OperatorGradTest, conv) {
  ENABLED_BACKENDS(Interpreter, CPU);

  auto *X = mod_.createPlaceholder(
      ElemKind::FloatTy, {1, 3 /* width */, 3 /* height */, 1 /* depth */}, "x",
      false);
  bindings_.allocate(X)->getHandle() = {0, 1, 2, 3, 4, 5, 6, 7, 8};

  ConvolutionNode *O =
      F_->createConv(bindings_, "conv", X, 1 /* outChannels */, 2 /* kernel */,
                     1 /* stride */, 0 /* pad */, 1 /* group */);

  // Note that filter is trainable, but probably it's okay since we run the
  // graph only once.
  bindings_.get(llvm::cast<Placeholder>(O->getFilter()))->getHandle() = {0, 1,
                                                                         2, 3};
  bindings_.get(llvm::cast<Placeholder>(O->getBias()))->getHandle() = {1};

  VariableGradientsList varGrads = computeVarGrads(O);

  // (Note that T stands for the filter).
  // O_0_0 = X_0_0*T_0_0 + X_0_1*T_0_1 + X_1_0*T_1_0 + X_1_1*T_1_1 + b = 20
  // O_0_1 = X_0_1*T_0_0 + X_0_2*T_0_1 + X_1_1*T_1_0 + X_1_2*T_1_1 + b = 26
  // O_1_0 = X_1_0*T_0_0 + X_1_1*T_0_1 + X_2_0*T_1_0 + X_2_1*T_1_1 + b = 38
  // O_1_1 = X_1_1*T_0_0 + X_1_2*T_0_1 + X_2_1*T_1_0 + X_2_2*T_1_1 + b = 44
  // (b is ignored).
  //
  // == Table for derivates ==
  //         d|X00|X01|X02|X10|X11|X12|X20|X21|X22
  // ----------------------------------------------
  // dO_0_0   |T00|T01| 0 |T10|T11| 0 | 0 | 0 | 0
  // dO_0_1   | 0 |T00|T01| 0 |T10|T11| 0 | 0 | 0
  // dO_1_0   | 0 | 0 | 0 |T00|T01| 0 |T10|T11| 0
  // dO_1_1   | 0 | 0 | 0 | 0 |T00|T01| 0 |T10|T11
  //
  // Based on the table above, dF/dX can be easily computed by
  // dF/dX_i_j = sum(dO_k_l/dX_i_j) for all k,l.

  Tensor expected(ElemKind::FloatTy, {1, 3, 3, 1});
  expected.getHandle() = {20 * 0, // row 0.
                          20 * 1 + 26 * 0,
                          26 * 1,
                          20 * 2 + 38 * 0, // row 1.
                          20 * 3 + 26 * 2 + 38 * 1 + 44 * 0,
                          26 * 3 + 44 * 1,
                          38 * 2,  // row 2.
                          38 * 3 + 44 * 2,
                          44 * 3};
  EXPECT_TRUE(expected.isEqual(*getGradTensor(varGrads, X)));
}

TEST_P(OperatorGradTest, fc) {
  ENABLED_BACKENDS(Interpreter);

  auto *x = mod_.createPlaceholder(ElemKind::FloatTy, {1, 2}, "x", false);
  bindings_.allocate(x)->getHandle() = {1, 2};
  auto *W = mod_.createPlaceholder(ElemKind::FloatTy, {2, 2}, "W", false);
  bindings_.allocate(W)->getHandle() = {1, 2, 3, 4};
  auto *b = mod_.createPlaceholder(ElemKind::FloatTy, {2}, "b", false);
  bindings_.allocate(b)->getHandle() = {1, 1};

  Node *O = F_->createFullyConnected("fc", x, W, b);
  VariableGradientsList varGrads = computeVarGrads(O);

  Tensor expected(ElemKind::FloatTy, {2, 2});
  // x = [x1, x2]
  // W = [[w1_1,w1_2],[w2_1,w2_2]]
  // O = [O1, O2] = [x1*w1_1 + x2*w2_1 + b2, x1*w1_2 + x2*w2_2 + b2] = [8, 11]
  // dO_k/dWi_j = (k == j ? 1 : 0) * x_i
  // dF/dW_i_j = sum_of_all_k (dF/dO_k * dO_k/dW_i_j).
  // dF/dO = [ 8, 11 ]
  // dF/dW = [ [O1*x1, O2*x1], [O1*x2, O2*x2] ]
  // dF/dW = [ [8*1,   11*1 ], [8*2,   11*2 ] ]
  expected.getHandle() = {8 * 1, 11 * 1, 8 * 2, 11 * 2};
  EXPECT_TRUE(expected.isEqual(*getGradTensor(varGrads, W)));
  // TODO: Add checks for grads of x and b.
}
