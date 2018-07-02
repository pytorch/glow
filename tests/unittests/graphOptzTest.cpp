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

#include "glow/Graph/Graph.h"
#include "glow/Graph/Node.h"
#include "glow/Graph/Nodes.h"
#include "glow/IR/IR.h"
#include "glow/Optimizer/Optimizer.h"

#include "gtest/gtest.h"

using namespace glow;

class GraphOptz : public ::testing::Test {
public:
  GraphOptz() { F_ = mod_.createFunction("main"); }

protected:
  Module mod_;
  Function *F_;
};

TEST_F(GraphOptz, DCE) {
  Node *K = mod_.createVariable(ElemKind::FloatTy, {4, 320, 200, 3}, "input");

  for (int i = 0; i < 40; i++) {
    K = F_->createRELU("relu", K);
    // Add a graph structure that diverges and converges, to catch algorithms
    // that perform a dump recursive scan.
    K = F_->createAdd("arith", K, K);
  }

  // Check that we know how many nodes we've created.
  EXPECT_EQ(F_->getNodes().size(), 80);

  // Optimize all of the dead code.
  ::glow::optimize(F_, CompilationMode::Infer);

  //  All of the nodes are gone.
  EXPECT_EQ(F_->getNodes().size(), 0);
  EXPECT_EQ(mod_.getVars().size(), 0);
}

TEST_F(GraphOptz, liveCodeNotEliminated) {
  Node *K = mod_.createVariable(ElemKind::FloatTy, {4, 320, 200, 3}, "input");
  auto *Ex = mod_.createVariable(ElemKind::IndexTy, {4, 1}, "Ex");

  for (int i = 0; i < 40; i++) {
    K = F_->createRELU("relu", K);
    K = F_->createAdd("arith", K, K);
  }
  K = F_->createSoftMax("Regression", K, Ex);
  F_->createSave("ret", K);

  // Check that we know how many nodes we've created.
  EXPECT_EQ(F_->getNodes().size(), 82);

  // This should not optimize code because none is dead.
  ::glow::optimize(F_, CompilationMode::Infer);

  //  Nothing got optimized.
  EXPECT_EQ(F_->getNodes().size(), 82);
  EXPECT_EQ(mod_.getVars().size(), 3);
}

TEST_F(GraphOptz, optimizeBatchNormAfterConv) {
  Node *A =
      mod_.createVariable(ElemKind::FloatTy, {1, 10, 20, 3}, "A",
                          VisibilityKind::Public, Variable::TrainKind::None);
  Node *CV = F_->createConv("conv", A, 16, 5, 1, 2, 1);
  Node *BN = F_->createBatchNormalization("batch", CV, 3, 0.0001, 0.9);
  F_->createSave("ret", BN);

  EXPECT_EQ(F_->getNodes().size(), 3);

  ::glow::optimize(F_, CompilationMode::Infer);
  EXPECT_EQ(F_->getNodes().size(), 2);
}

TEST_F(GraphOptz, optimizeBatchNormAfterConvButConvReused) {
  Node *A =
      mod_.createVariable(ElemKind::FloatTy, {1, 10, 20, 3}, "A",
                          VisibilityKind::Public, Variable::TrainKind::None);
  Node *CV = F_->createConv("conv", A, 16, 5, 1, 2, 1);
  Node *BN = F_->createBatchNormalization("batch", CV, 3, 0.0001, 0.9);
  SaveNode *ret = F_->createSave("ret", BN);
  SaveNode *convSave = F_->createSave("convSave", CV);

  EXPECT_EQ(F_->getNodes().size(), 4);

  ::glow::optimize(F_, CompilationMode::Infer);
  // Make sure the structure of the graph did not change, since the convolution
  // node is used more than once.
  EXPECT_EQ(F_->getNodes().size(), 4);
  EXPECT_TRUE(llvm::isa<ConvolutionNode>(convSave->getInput()));
  ConvolutionNode *conv = llvm::dyn_cast<ConvolutionNode>(convSave->getInput());
  EXPECT_EQ(conv, CV);
  EXPECT_TRUE(llvm::isa<BatchNormalizationNode>(ret->getInput()));
  BatchNormalizationNode *batchNorm =
      llvm::dyn_cast<BatchNormalizationNode>(ret->getInput());
  EXPECT_EQ(batchNorm, BN);
  EXPECT_EQ(batchNorm->getInput().getNode(), CV);
  EXPECT_EQ(conv->getInput().getNode(), A);
}

TEST_F(GraphOptz, optimizeBatchNormAfterConvButVarReused) {
  Node *A =
      mod_.createVariable(ElemKind::FloatTy, {1, 10, 20, 3}, "A",
                          VisibilityKind::Public, Variable::TrainKind::None);
  Variable *filter = mod_.createVariable(ElemKind::FloatTy, {16, 5, 5, 3},
                                         "filter", VisibilityKind::Private,
                                         Variable::TrainKind::Broadcast, 75);
  Variable *bias = mod_.createVariable(ElemKind::FloatTy, {16}, "bias",
                                       VisibilityKind::Private,
                                       Variable::TrainKind::Broadcast, 1.0);
  ConvolutionNode *CV = F_->createConv(
      "conv", A, filter, bias,
      mod_.uniqueType(ElemKind::FloatTy, {1, 10, 20, 16}), 5, 1, 2, 1);
  float filterValue = llvm::cast<Variable>(CV->getFilter())->getValue();
  Variable *beta = mod_.createVariable(ElemKind::FloatTy, {16}, "beta",
                                       VisibilityKind::Private,
                                       Variable::TrainKind::Broadcast, 0.0);
  Variable *gamma = mod_.createVariable(ElemKind::FloatTy, {16}, "gamma",
                                        VisibilityKind::Private,
                                        Variable::TrainKind::Broadcast, 2.0);
  Variable *mean = mod_.createVariable(ElemKind::FloatTy, {16}, "mean");
  Variable *var = mod_.createVariable(ElemKind::FloatTy, {16}, "var");

  Node *BN = F_->createBatchNormalization("batch", CV, beta, gamma, mean, var,
                                          3, 0.0001, 0.9);
  SaveNode *ret = F_->createSave("ret", BN);
  SaveNode *filterSave = F_->createSave("filterSave", CV->getFilter());

  EXPECT_EQ(F_->getNodes().size(), 4);

  ::glow::optimize(F_, CompilationMode::Infer);
  // Make sure the structure of the graph did not change.
  EXPECT_EQ(F_->getNodes().size(), 4);
  EXPECT_TRUE(llvm::isa<VariableNode>(filterSave->getInput()));
  VariableNode *varFilter =
      llvm::dyn_cast<VariableNode>(filterSave->getInput());
  EXPECT_EQ(varFilter, CV->getFilter());
  EXPECT_TRUE(llvm::isa<BatchNormalizationNode>(ret->getInput()));
  BatchNormalizationNode *batchNorm =
      llvm::dyn_cast<BatchNormalizationNode>(ret->getInput());
  EXPECT_EQ(batchNorm, BN);
  EXPECT_TRUE(batchNorm && batchNorm->getInput() &&
              batchNorm->getInput().getNode() == CV);

  // Make sure that we didn't temper the values in the filter
  // given we had more than one use for it.
  auto filterH = filter->getHandle<>();
  for (size_t i = 0, e = filterH.size(); i < e; ++i) {
    EXPECT_EQ(filterH.raw(i), filterValue);
  }
}

TEST_F(GraphOptz, transposePrivateVariable) {
  Node *A =
      mod_.createVariable(ElemKind::FloatTy, {1, 10, 20, 3}, "A",
                          VisibilityKind::Private, Variable::TrainKind::None);
  Node *T = F_->createTranspose("transpose", A, NHWC2NCHW);
  F_->createSave("ret", T);
  EXPECT_EQ(F_->getNodes().size(), 2);

  ::glow::optimize(F_, CompilationMode::Infer);
  EXPECT_EQ(F_->getNodes().size(), 1);
}

TEST_F(GraphOptz, BatchNormAfterConvNotOptimizeForTrain) {
  Node *A =
      mod_.createVariable(ElemKind::FloatTy, {1, 10, 20, 3}, "A",
                          VisibilityKind::Public, Variable::TrainKind::None);
  Node *CV = F_->createConv("conv", A, 16, 5, 1, 2, 1);
  Node *BN = F_->createBatchNormalization("batch", CV, 3, 0.0001, 0.9);
  F_->createSave("ret", BN);

  EXPECT_EQ(F_->getNodes().size(), 3);

  ::glow::optimize(F_, CompilationMode::Train);
  EXPECT_EQ(F_->getNodes().size(), 3);
}

TEST_F(GraphOptz, batchNormAfterConvNotOptimizeWhenMoreThanOneUseOfConv) {
  Node *A =
      mod_.createVariable(ElemKind::FloatTy, {1, 10, 20, 3}, "A",
                          VisibilityKind::Public, Variable::TrainKind::None);

  Node *CV = F_->createConv("conv", A, 16, 5, 1, 2, 1);
  Node *BN = F_->createBatchNormalization("batch", CV, 3, 0.0001, 0.9);
  F_->createSave("ret", BN);
  F_->createSave("ret", CV);

  EXPECT_EQ(F_->getNodes().size(), 4);

  ::glow::optimize(F_, CompilationMode::Infer);
  EXPECT_EQ(F_->getNodes().size(), 4);
}

TEST_F(GraphOptz, sinkTransposeBelowOptimizeBatchNorm) {
  Node *A =
      mod_.createVariable(ElemKind::FloatTy, {1, 5, 10, 15}, "input",
                          VisibilityKind::Public, Variable::TrainKind::None);
  Node *T = F_->createTranspose("transpose", A, NHWC2NCHW);
  Node *BN = F_->createBatchNormalization("batch", T, 3, 0.0001, 0.9);
  Node *O = F_->createSave("ret", BN);

  EXPECT_EQ(F_->getNodes().size(), 3);

  ::glow::optimize(F_, CompilationMode::Infer);

  // Expecting Transpose->Output rather than BN->Output.
  EXPECT_TRUE(llvm::isa<SaveNode>(O));
  EXPECT_TRUE(
      llvm::isa<TransposeNode>(llvm::dyn_cast<SaveNode>(O)->getInput()));

  EXPECT_EQ(F_->getNodes().size(), 3);
}

TEST_F(GraphOptz, sinkTransposeBelowRELU) {
  Node *A =
      mod_.createVariable(ElemKind::FloatTy, {1, 5, 10, 15}, "input",
                          VisibilityKind::Public, Variable::TrainKind::None);
  Node *T = F_->createTranspose("transpose", A, NHWC2NCHW);
  Node *K = F_->createRELU("relu", T);
  Node *O = F_->createSave("ret", K);

  EXPECT_EQ(F_->getNodes().size(), 3);

  ::glow::optimize(F_, CompilationMode::Infer);

  // Expecting Transpose->Output rather than RELU->Output.
  EXPECT_TRUE(llvm::isa<SaveNode>(O));
  EXPECT_TRUE(
      llvm::isa<TransposeNode>(llvm::dyn_cast<SaveNode>(O)->getInput()));

  EXPECT_EQ(F_->getNodes().size(), 3);
}

TEST_F(GraphOptz, sinkTransposeBelowSigmoid) {
  Node *A =
      mod_.createVariable(ElemKind::FloatTy, {1, 5, 10, 15}, "input",
                          VisibilityKind::Public, Variable::TrainKind::None);
  Node *T = F_->createTranspose("transpose", A, NHWC2NCHW);
  Node *SI = F_->createSigmoid("sigmoid", T);
  Node *O = F_->createSave("ret", SI);

  EXPECT_EQ(F_->getNodes().size(), 3);

  ::glow::optimize(F_, CompilationMode::Infer);

  // Expecting Transpose->Output rather than Sigmoid->Output.
  EXPECT_TRUE(llvm::isa<SaveNode>(O));
  EXPECT_TRUE(
      llvm::isa<TransposeNode>(llvm::dyn_cast<SaveNode>(O)->getInput()));

  EXPECT_EQ(F_->getNodes().size(), 3);
}

TEST_F(GraphOptz, sinkTransposeBelowTanh) {
  Node *A =
      mod_.createVariable(ElemKind::FloatTy, {1, 5, 10, 15}, "input",
                          VisibilityKind::Public, Variable::TrainKind::None);
  Node *T = F_->createTranspose("transpose", A, NHWC2NCHW);
  Node *TN = F_->createTanh("tanh", T);
  Node *O = F_->createSave("ret", TN);

  EXPECT_EQ(F_->getNodes().size(), 3);

  ::glow::optimize(F_, CompilationMode::Infer);

  // Expecting Transpose->Output rather than Sigmoid->Output.
  EXPECT_TRUE(llvm::isa<SaveNode>(O));
  EXPECT_TRUE(
      llvm::isa<TransposeNode>(llvm::dyn_cast<SaveNode>(O)->getInput()));

  EXPECT_EQ(F_->getNodes().size(), 3);
}

TEST_F(GraphOptz, cancelTwoTransposes) {
  Node *A =
      mod_.createVariable(ElemKind::FloatTy, {1, 5, 10, 15}, "input",
                          VisibilityKind::Public, Variable::TrainKind::None);
  Node *T1 = F_->createTranspose("transpose", A, NCHW2NHWC);
  Node *T2 = F_->createTranspose("transpose", T1, NHWC2NCHW);
  Node *K = F_->createRELU("relu", T2);
  F_->createSave("ret", K);

  EXPECT_EQ(F_->getNodes().size(), 4);

  ::glow::optimize(F_, CompilationMode::Infer);

  EXPECT_EQ(F_->getNodes().size(), 2);
}

TEST_F(GraphOptz, dontCancelTwoTransposesIfNotMatching) {
  Node *A =
      mod_.createVariable(ElemKind::FloatTy, {1, 5, 10, 15}, "input",
                          VisibilityKind::Public, Variable::TrainKind::None);
  Node *T1 = F_->createTranspose("transpose", A, NCHW2NHWC);
  Node *T2 = F_->createTranspose("transpose", T1, {0, 1, 2, 3});
  Node *K = F_->createRELU("relu", T2);
  F_->createSave("ret", K);

  EXPECT_EQ(F_->getNodes().size(), 4);

  ::glow::optimize(F_, CompilationMode::Infer);

  EXPECT_EQ(F_->getNodes().size(), 4);
}

TEST_F(GraphOptz, sinkTransposeBelowArithmeticNodes) {
  Node *A1 =
      mod_.createVariable(ElemKind::FloatTy, {1, 5, 10, 15}, "input1",
                          VisibilityKind::Public, Variable::TrainKind::None);
  Node *A2 =
      mod_.createVariable(ElemKind::FloatTy, {1, 5, 10, 15}, "input2",
                          VisibilityKind::Public, Variable::TrainKind::None);
  Node *T1 = F_->createTranspose("transpose1", A1, NHWC2NCHW);
  Node *T2 = F_->createTranspose("transpose2", A2, NHWC2NCHW);
  Node *K = F_->createAdd("arith", T1, T2);
  Node *O = F_->createSave("ret", K);

  EXPECT_EQ(F_->getNodes().size(), 4);

  ::glow::optimize(F_, CompilationMode::Infer);

  // Expecting Transpose->Output rather than Add->Output.
  EXPECT_TRUE(llvm::isa<SaveNode>(O));
  EXPECT_TRUE(
      llvm::isa<TransposeNode>(llvm::dyn_cast<SaveNode>(O)->getInput()));

  EXPECT_EQ(F_->getNodes().size(), 3);
}

TEST_F(GraphOptz, sinkReluBelowConcatNodes) {
  Node *A1 =
      mod_.createVariable(ElemKind::FloatTy, {1, 5, 10, 15}, "input1",
                          VisibilityKind::Public, Variable::TrainKind::None);
  Node *A2 =
      mod_.createVariable(ElemKind::FloatTy, {1, 5, 10, 15}, "input2",
                          VisibilityKind::Public, Variable::TrainKind::None);
  Node *R1 = F_->createRELU("relu1", A1);
  Node *R2 = F_->createRELU("relu2", A2);
  Node *CN = F_->createConcat("concat", {R1, R2}, 1);
  Node *O = F_->createSave("ret", CN);

  EXPECT_EQ(F_->getNodes().size(), 4);

  ::glow::optimize(F_, CompilationMode::Infer);

  // Expecting RELU->Output rather than Concat->Output.
  EXPECT_TRUE(llvm::isa<SaveNode>(O));
  EXPECT_TRUE(llvm::isa<ReluNode>(llvm::dyn_cast<SaveNode>(O)->getInput()));

  EXPECT_EQ(F_->getNodes().size(), 3);
}

TEST_F(GraphOptz, sinkTransposeBelowConcatNodes) {
  Node *A1 =
      mod_.createVariable(ElemKind::FloatTy, {1, 5, 10, 15}, "input1",
                          VisibilityKind::Public, Variable::TrainKind::None);
  Node *A2 =
      mod_.createVariable(ElemKind::FloatTy, {1, 5, 10, 15}, "input2",
                          VisibilityKind::Public, Variable::TrainKind::None);
  Node *T1 = F_->createTranspose("transpose", A1, NCHW2NHWC);
  Node *T2 = F_->createTranspose("transpose", A2, NCHW2NHWC);
  Node *CN = F_->createConcat("concat", {T1, T2}, 1);
  Node *O = F_->createSave("ret", CN);

  EXPECT_EQ(F_->getNodes().size(), 4);

  ::glow::optimize(F_, CompilationMode::Infer);

  // Expecting Transpose->Output rather than Concat->Output.
  EXPECT_TRUE(llvm::isa<SaveNode>(O));
  EXPECT_TRUE(
      llvm::isa<TransposeNode>(llvm::dyn_cast<SaveNode>(O)->getInput()));

  EXPECT_EQ(F_->getNodes().size(), 3);
}

TEST_F(GraphOptz, poolBelowReluSwapped) {
  Node *A =
      mod_.createVariable(ElemKind::FloatTy, {1, 5, 10, 15}, "input",
                          VisibilityKind::Public, Variable::TrainKind::None);
  Node *R = F_->createRELU("relu", A);
  Node *PL = F_->createPoolMax("pool", R, 1, 10, 20);
  Node *O = F_->createSave("ret", PL);

  EXPECT_EQ(F_->getNodes().size(), 3);

  ::glow::optimize(F_, CompilationMode::Infer);

  // Expecting RELU->Output rather than Pool->Output.
  EXPECT_TRUE(llvm::isa<SaveNode>(O));
  EXPECT_TRUE(llvm::isa<ReluNode>(llvm::dyn_cast<SaveNode>(O)->getInput()));

  EXPECT_EQ(F_->getNodes().size(), 3);
}

TEST_F(GraphOptz, poolBelowReluNotSwappedIfModeNotMax) {
  Node *A =
      mod_.createVariable(ElemKind::FloatTy, {1, 5, 10, 15}, "input",
                          VisibilityKind::Public, Variable::TrainKind::None);
  Node *R = F_->createRELU("relu", A);
  Node *PL = F_->createPoolAvg("pool", R, 1, 10, 20);
  Node *O = F_->createSave("ret", PL);

  EXPECT_EQ(F_->getNodes().size(), 3);

  ::glow::optimize(F_, CompilationMode::Infer);

  // Expecting Pool->Output (no swap).
  EXPECT_TRUE(llvm::isa<SaveNode>(O));
  EXPECT_TRUE(llvm::isa<PoolAvgNode>(llvm::dyn_cast<SaveNode>(O)->getInput()));

  EXPECT_EQ(F_->getNodes().size(), 3);
}

TEST_F(GraphOptz, poolBelowReluNotSwappedIfNotSingleUse) {
  Node *A =
      mod_.createVariable(ElemKind::FloatTy, {1, 5, 10, 15}, "input",
                          VisibilityKind::Public, Variable::TrainKind::None);
  Node *R = F_->createRELU("relu", A);
  Node *PL = F_->createPoolMax("pool", R, 1, 10, 20);
  Node *O = F_->createSave("ret", PL);
  F_->createSave("ret", R);

  EXPECT_EQ(F_->getNodes().size(), 4);

  ::glow::optimize(F_, CompilationMode::Infer);

  // Expecting Pool->Output (no swap).
  EXPECT_TRUE(llvm::isa<SaveNode>(O));
  EXPECT_TRUE(llvm::isa<PoolMaxNode>(llvm::dyn_cast<SaveNode>(O)->getInput()));

  EXPECT_EQ(F_->getNodes().size(), 4);
}

/// A helper predicate to check if the provided node has the same address as a
/// pre-defined address provided in constructor. This is useful if you need to
/// check that a given node is still in the graph. In general, it is not safe to
/// use the std::find(begin_it, end_it, value) and compare the nodes by value,
/// because the node provided as the last parameter of std::find (i.e. the value
/// reference) may have been removed by some optimizations and cannot be
/// dereferenced anymore. But comparing the addresses of the nodes should be
/// fine. Thus, one can use the following form instead:
/// std::find_if(begin_it, end_it, IsSameNodeAddress(node_address))
struct IsSameNodeAddress {
  Node *nodeAddress_;
  IsSameNodeAddress(Node *nodeAddress) : nodeAddress_(nodeAddress) {}
  bool operator()(const Node &n) const { return &n == nodeAddress_; }
};

TEST_F(GraphOptz, mergeConcatNodes) {
  Node *A1 =
      mod_.createVariable(ElemKind::FloatTy, {1, 5, 10, 15}, "input1",
                          VisibilityKind::Public, Variable::TrainKind::None);
  Node *A2 =
      mod_.createVariable(ElemKind::FloatTy, {1, 5, 10, 15}, "input2",
                          VisibilityKind::Public, Variable::TrainKind::None);
  Node *A3 =
      mod_.createVariable(ElemKind::FloatTy, {1, 5, 10, 15}, "input3",
                          VisibilityKind::Public, Variable::TrainKind::None);
  Node *A4 =
      mod_.createVariable(ElemKind::FloatTy, {1, 1, 10, 15}, "input4",
                          VisibilityKind::Public, Variable::TrainKind::None);

  Node *CN1 = F_->createConcat("concat1", {A1, A2}, 1);
  Node *CN2 = F_->createConcat("concat2", {A1, CN1}, 1);
  Node *CN3 = F_->createConcat("concat3", {A4}, 2);
  Node *CN4 = F_->createConcat("concat4", {A3, CN2, CN3}, 1);
  Node *O = F_->createSave("ret", CN4);

  EXPECT_EQ(F_->getNodes().size(), 5);

  ::glow::optimize(F_, CompilationMode::Train);

  // It is expected that the optimization transforms
  // concat4(1, A3, concat2(1, A1, concat1(1, A1, A2)), concat3(2, A4))
  // into
  // concat4(1, A3, A1, A1, A2, concat3(2, A4))

  EXPECT_TRUE(llvm::isa<SaveNode>(O));

  auto *CN =
      llvm::dyn_cast<ConcatNode>(llvm::dyn_cast<SaveNode>(O)->getInput());
  EXPECT_TRUE(CN);

  // The merged ConcatNode should have 5 inputs.
  EXPECT_EQ(CN->getInputs().size(), 5);

  // CN1 should be merged into a new CN2 and later into a new CN4 and removed by
  // the optimizations.
  EXPECT_TRUE(std::find_if(F_->getNodes().begin(), F_->getNodes().end(),
                           IsSameNodeAddress(CN1)) == F_->getNodes().end());

  // CN2 should be merged into a new CN4 and removed by the optimizations.
  EXPECT_TRUE(std::find_if(F_->getNodes().begin(), F_->getNodes().end(),
                           IsSameNodeAddress(CN2)) == F_->getNodes().end());

  // CN3 should not be merged into CN4 and should not be removed,
  // because CN4 and CN3 have a different dimension parameter.
  EXPECT_TRUE(std::find_if(F_->getNodes().begin(), F_->getNodes().end(),
                           IsSameNodeAddress(CN3)) != F_->getNodes().end());

  // The CN4 concat node should be replaced by a merged concat node.
  EXPECT_TRUE(std::find_if(F_->getNodes().begin(), F_->getNodes().end(),
                           IsSameNodeAddress(CN4)) == F_->getNodes().end());

  EXPECT_EQ(F_->getNodes().size(), 3);
}

TEST_F(GraphOptz, CSE) {
  Node *A1 =
      mod_.createVariable(ElemKind::FloatTy, {1, 5, 10, 15}, "input1",
                          VisibilityKind::Public, Variable::TrainKind::None);
  Node *A2 =
      mod_.createVariable(ElemKind::FloatTy, {1, 5, 10, 15}, "input2",
                          VisibilityKind::Public, Variable::TrainKind::None);

  Node *CN1 = F_->createConcat("concat1", {A1, A2}, 1);
  Node *CN2 = F_->createConcat("concat2", {A1, A2}, 1);
  Node *CN3 = F_->createConcat("concat3", {CN1, CN2}, 2);
  Node *O = F_->createSave("ret", CN3);

  EXPECT_EQ(F_->getNodes().size(), 4);

  ::glow::optimize(F_, CompilationMode::Train);

  EXPECT_TRUE(llvm::isa<SaveNode>(O));

  auto *CN =
      llvm::dyn_cast<ConcatNode>(llvm::dyn_cast<SaveNode>(O)->getInput());
  EXPECT_TRUE(CN);

  // The merged ConcatNode should have 2 inputs.
  EXPECT_EQ(CN->getInputs().size(), 2);

  // CN1 should not be removed.
  EXPECT_TRUE(std::find_if(F_->getNodes().begin(), F_->getNodes().end(),
                           IsSameNodeAddress(CN1)) != F_->getNodes().end());

  // CSE should replace CN2 by CN1 and remove CN2.
  EXPECT_TRUE(std::find_if(F_->getNodes().begin(), F_->getNodes().end(),
                           IsSameNodeAddress(CN2)) == F_->getNodes().end());

  EXPECT_EQ(F_->getNodes().size(), 3);
}

TEST_F(GraphOptz, SliceOfSplatNode) {
  Type t(ElemKind::FloatTy, {1000, 1000, 1000});
  Node *Z = F_->createSplat("zero", &t, 0.);
  Node *S = F_->createSlice("slice", Z, {5, 15, 42}, {99, 88, 77});
  Node *O = F_->createSave("ret", S);

  EXPECT_EQ(F_->getNodes().size(), 3);

  ::glow::optimize(F_, CompilationMode::Train);

  EXPECT_EQ(F_->getNodes().size(), 2);

  EXPECT_TRUE(llvm::isa<SaveNode>(O));

  auto *CN = llvm::dyn_cast<SplatNode>(llvm::dyn_cast<SaveNode>(O)->getInput());
  EXPECT_TRUE(CN);

  EXPECT_TRUE(CN->getType()->dims().equals({94, 73, 35}));
}

TEST_F(GraphOptz, ZeroArithmetic) {
  // Tests the identities: [0 + X = X] [0 * X = 0] [0 / X = 0] [ X - 0 = X]

  Node *input = mod_.createVariable(ElemKind::FloatTy, {4, 10}, "input",
                                    VisibilityKind::Public);

  // This builds the expression: ((0 / I) + (0 + I) + (0 * I)) - 0

  auto *zero = F_->createSplat("zero", input->getType(), 0.);

  auto *div = F_->createDiv("div", zero, input); // -> zero

  auto *add = F_->createAdd("add", zero, input); // -> input

  auto *mul = F_->createMul("mul", zero, input); // -> zero

  auto *add3 = F_->createAdd("add", div, add);

  add3 = F_->createAdd("add", add3, mul);

  auto *sub = F_->createSub("sub", add3, zero); // -> input

  SaveNode *O = F_->createSave("ret", sub);

  // The expression evaluates to "I".

  EXPECT_EQ(F_->getNodes().size(), 8);

  ::glow::optimize(F_, CompilationMode::Infer);

  EXPECT_EQ(F_->getNodes().size(), 1);

  EXPECT_EQ(O->getInput().getNode(), input);
}

/// Reverse the intrusive list of nodes. This custom implementation is required,
/// because std::reverse cannot be used with LLVM's intrusive lists.
static void reverse(NodesList &L) {
  if (L.empty())
    return;
  // Last element of the list before reversal.
  auto &last = L.back();
  // Take element from the beginning and move it right after the old last
  // element. Do it until the old last element becomes the first element.
  while (true) {
    auto &first = L.front();
    // Finish when the old last element becomes the new front element.
    if (&first == &last) {
      break;
    }
    L.remove(first);
    L.insert(++last.getIterator(), &first);
  }
}

TEST(GraphOptzTest, SliceOfSplatNodeChain) {
  for (int shouldReverse = 0; shouldReverse <= 1; shouldReverse++) {
    Module mod;
    Function *F = mod.createFunction("foo");

    Type t(ElemKind::FloatTy, {1000, 1000, 1000});
    Node *Z = F->createSplat("zero", &t, 0.);
    Node *S1 = F->createSlice("slice1", Z, {5, 15, 42}, {99, 88, 77});
    Node *S2 = F->createSlice("slice2", S1, {1, 1, 1}, {2, 3, 4});
    F->createSave("ret", S2);

    if (shouldReverse) {
      auto &nodes = F->getNodes();
      reverse(nodes);
    }

    EXPECT_EQ(F->getNodes().size(), 4);

    ::glow::optimize(F, CompilationMode::Train);

    // This test illustrates some inconsistency in the optimization.
    // Chain splats are not guaranteed to be optimized.
    EXPECT_EQ(F->getNodes().size(), shouldReverse ? 3 : 2);
  }
}

TEST_F(GraphOptz, ReshapeNoop) {
  const size_t shape[] = {10, 20, 30};
  Type t(ElemKind::FloatTy, shape);
  auto *Z = F_->createSplat("zero", &t, 0.);
  auto *R = F_->createReshape("reshape", Z, shape);
  auto *O = F_->createSave("ret", R);

  EXPECT_EQ(F_->getNodes().size(), 3);

  ::glow::optimize(F_, CompilationMode::Train);

  EXPECT_EQ(F_->getNodes().size(), 2);

  auto *SN = llvm::dyn_cast<SplatNode>(llvm::dyn_cast<SaveNode>(O)->getInput());
  EXPECT_TRUE(SN);

  EXPECT_TRUE(SN->getType()->dims().equals(shape));
}

/// Test the Reshape(Splat(args)) -> Splat(args') transformation.
/// Including a positive and a negative test case. In the positive case,
/// the optimization will take place for the splat node (Z2) that has only one
/// use. In the negative case, the optimization will not happen as the splat
/// node (Z1) has more than one use.
TEST_F(GraphOptz, ReshapeAfterSplat) {
  const size_t shape[] = {10, 20, 30};
  const size_t reshape[] = {1, 6000};
  Type t1(ElemKind::FloatTy, shape);
  Type t2(ElemKind::FloatTy, reshape);
  Node *input =
      F_->getParent()->createVariable(ElemKind::FloatTy, shape, "input");
  auto *Z1 = F_->createSplat("zero1", &t1, 1.5);
  auto *A1 = F_->createAdd("add1", Z1->getType(), input, Z1);
  auto *R1 = F_->createReshape("reshape1", Z1, reshape);
  // Z1 is used by R1 and A1.
  // The reshape optimization will thus NOT be able to remove this reshape node
  // (R1).
  auto *R2 = F_->createReshape("reshape2", A1, reshape);
  auto *A2 = F_->createAdd("add", R1->getType(), R1, R2);
  auto *Z2 = F_->createSplat("zero2", &t1, 2.5);
  auto *R3 = F_->createReshape("reshape3", Z2, reshape);
  // Z2 is only used by R3.
  // The Z2,R3 nodes will be replaced by a new splat node with the shape of R3.
  auto *A3 = F_->createAdd("add", A2->getType(), A2, R3);
  auto *O = F_->createSave("ret", A3);

  // Before optimization, we have 9 nodes in the graph.
  EXPECT_EQ(F_->getNodes().size(), 9);

  ::glow::optimize(F_, CompilationMode::Infer);

  // After optimization, we expect to see only 8 nodes, as Z2,R2 would be
  // replace by a new splat node.
  EXPECT_EQ(F_->getNodes().size(), 8);

  // The second input of A3 shoule be a splat node with a shape of R3.
  auto *SN = llvm::dyn_cast<SplatNode>(
      llvm::dyn_cast<SaveNode>(O)->getInput()->getNthInput(1));
  EXPECT_TRUE(SN);
  EXPECT_TRUE(SN->getType()->dims().equals(reshape));

  // R1 should still be in the graph.
  EXPECT_TRUE(std::find_if(F_->getNodes().begin(), F_->getNodes().end(),
                           IsSameNodeAddress(R1)) != F_->getNodes().end());

  // R3 and Z2 should not be in the graph any more.
  EXPECT_TRUE(std::find_if(F_->getNodes().begin(), F_->getNodes().end(),
                           IsSameNodeAddress(R3)) == F_->getNodes().end());
  EXPECT_TRUE(std::find_if(F_->getNodes().begin(), F_->getNodes().end(),
                           IsSameNodeAddress(Z2)) == F_->getNodes().end());
}

TEST_F(GraphOptz, DCEPublicVars) {
  mod_.createVariable(ElemKind::FloatTy, {4, 320, 200, 3}, "input",
                      VisibilityKind::Public);

  EXPECT_EQ(mod_.getVars().size(), 1);

  // Optimize all of the dead code.
  ::glow::optimize(F_, CompilationMode::Infer);

  //  Public nodes should not be deleted.
  EXPECT_EQ(mod_.getVars().size(), 1);
}

TEST_F(GraphOptz, foldQuantizeIntoVar) {
  auto input = mod_.createVariable(ElemKind::FloatTy, {4}, "input",
                                   VisibilityKind::Private);
  input->getPayload() = {10, 10, 10, 10};
  auto qType = mod_.uniqueType(ElemKind::Int8QTy, {4}, 2, 0);

  auto Q = F_->createQuantize("quantize", input, qType);
  auto S = F_->createSave("save", Q);

  EXPECT_EQ(2, F_->getNodes().size());
  ::glow::optimize(F_, CompilationMode::Infer);
  // Quantization node was merged into input var.
  EXPECT_EQ(1, F_->getNodes().size());

  auto quantizedInput = llvm::cast<Variable>(S->getInput());
  auto quantizedValues = quantizedInput->getHandle<int8_t>();
  for (unsigned i = 0; i < 4; ++i) {
    EXPECT_EQ(5, quantizedValues.raw(i));
  }
}

TEST_F(GraphOptz, foldQuantizeIntoVarMultipleUsages) {
  auto input = mod_.createVariable(ElemKind::FloatTy, {4}, "input",
                                   VisibilityKind::Private);
  input->getPayload() = {10, 10, 10, 10};
  auto qType = mod_.uniqueType(ElemKind::Int8QTy, {4}, 2, 0);

  auto Q = F_->createQuantize("quantize", input, qType);
  F_->createSave("save", Q);
  auto clonedF = F_->clone("cloned");

  EXPECT_EQ(2, clonedF->getNodes().size());
  ::glow::optimize(clonedF, CompilationMode::Infer);
  // F_ function should not be affected.
  EXPECT_EQ(2, F_->getNodes().size());

  // Check original var.
  for (unsigned i = 0; i < 4; ++i) {
    EXPECT_EQ(10, input->getHandle().raw(i));
  }

  // Quantization node was merged into input var.
  EXPECT_EQ(1, clonedF->getNodes().size());
  auto quantizedInput =
      llvm::cast<Variable>(clonedF->getNodes().front().getNthInput(0));
  auto quantizedValues = quantizedInput->getHandle<int8_t>();
  for (unsigned i = 0; i < 4; ++i) {
    EXPECT_EQ(5, quantizedValues.raw(i));
  }
}

TEST_F(GraphOptz, quantizeToRescale) {
  // Check that we are combining quantization-dequantization pairs.
  Node *input = mod_.createVariable(ElemKind::Int8QTy, {4, 10}, 0.5, 11,
                                    "input", VisibilityKind::Public,
                                    Variable::TrainKind::Broadcast, 15);
  auto *D = F_->createDequantize("dequantize", input);

  auto qType = mod_.uniqueType(ElemKind::Int8QTy, {4, 10}, 0.03, 5);
  auto *Q = F_->createQuantize("quantize", D, qType);

  F_->createSave("ret", Q);

  EXPECT_EQ(F_->getNodes().size(), 3);

  ::glow::optimize(F_, CompilationMode::Infer);
  EXPECT_EQ(F_->getNodes().size(), 2);
}

TEST_F(GraphOptz, MaxOfQuantizedSplat) {
  const size_t size = 5;
  const float scale = 1;
  // offset == -128 guarantees that fp range has values which are not less than
  // 0.
  const int32_t offset = -128;

  auto splatTy = mod_.uniqueType(ElemKind::Int8QTy, {size}, scale, offset);
  auto *splat = F_->createSplat("splat", splatTy, 0.0);

  Node *input = mod_.createVariable(ElemKind::Int8QTy, {size}, scale, offset,
                                    "input", VisibilityKind::Public,
                                    Variable::TrainKind::Broadcast, 4);
  auto *max = F_->createMax("max", splat, input);
  F_->createSave("save", max);
  EXPECT_EQ(F_->getNodes().size(), 3);

  ::glow::optimize(F_, CompilationMode::Infer);
  // Splat and Max should be gone.
  EXPECT_EQ(F_->getNodes().size(), 1);
}

TEST_F(GraphOptz, FuseRescaleIntoArithmetic) {
  // This test ensures the fact that fusing of rescale is done.
  auto opOutTy = mod_.uniqueType(ElemKind::Int8QTy, {10}, 1, 0);
  auto rescaleOutTy = mod_.uniqueType(ElemKind::Int8QTy, {10}, 2, 1);

  Node *LHS = mod_.createVariable(ElemKind::Int8QTy, {10}, 0.4, 0, "LHS",
                                  VisibilityKind::Public);
  Node *RHS = mod_.createVariable(ElemKind::Int8QTy, {10}, 0.3, 0, "RHS",
                                  VisibilityKind::Public);

  Node *add = F_->createAdd("qAdd", opOutTy, LHS, RHS);
  add = F_->createRescaleQuantized("rsAdd", add, rescaleOutTy);
  add = F_->createSave("saveAdd", add);

  Node *sub = F_->createSub("qSub", opOutTy, LHS, RHS);
  sub = F_->createRescaleQuantized("rsSub", sub, rescaleOutTy);
  sub = F_->createSave("saveSub", sub);

  Node *div = F_->createDiv("qDiv", opOutTy, LHS, RHS);
  div = F_->createRescaleQuantized("rsDiv", div, rescaleOutTy);
  div = F_->createSave("saveDiv", div);

  Node *mul = F_->createMul("qMul", opOutTy, LHS, RHS);
  mul = F_->createRescaleQuantized("rsMul", mul, rescaleOutTy);
  mul = F_->createSave("saveMul", mul);

  Node *min = F_->createMin("qMin", opOutTy, LHS, RHS);
  min = F_->createRescaleQuantized("rsMin", min, rescaleOutTy);
  min = F_->createSave("saveMin", min);

  Node *max = F_->createMax("qMax", opOutTy, LHS, RHS);
  max = F_->createRescaleQuantized("rsMax", max, rescaleOutTy);
  max = F_->createSave("saveMax", max);

  // All rescales must be fused into arithmetic operations above.
  ::glow::optimize(F_, CompilationMode::Infer);

  EXPECT_EQ(F_->getNodes().size(), 12);

  EXPECT_EQ(add->getNthInput(0).getType(), rescaleOutTy);
  EXPECT_EQ(sub->getNthInput(0).getType(), rescaleOutTy);
  EXPECT_EQ(mul->getNthInput(0).getType(), rescaleOutTy);
  EXPECT_EQ(div->getNthInput(0).getType(), rescaleOutTy);
  EXPECT_EQ(min->getNthInput(0).getType(), rescaleOutTy);
  EXPECT_EQ(max->getNthInput(0).getType(), rescaleOutTy);
}

TEST_F(GraphOptz, sinkRescaledQuantizedNode) {
  // Check that we eliminate rescale nodes by sinking them into other operators.
  Node *input = mod_.createVariable(ElemKind::Int8QTy, {4, 10}, 0.5, 11,
                                    "input", VisibilityKind::Public,
                                    Variable::TrainKind::Broadcast, 15);
  // slice -> rescale -> reshape -> rescale -> transpose -> save.
  auto *slice = F_->createSlice("slice", input, {0, 0}, {3, 3});
  auto *rescale = F_->createRescaleQuantized(
      "rescale", slice, mod_.uniqueType(ElemKind::Int8QTy, {3, 3}, 0.4, 10));
  auto *reshape = F_->createReshape("reshape", rescale, {1, 9});
  auto *rescale2 = F_->createRescaleQuantized(
      "rescale", reshape, mod_.uniqueType(ElemKind::Int8QTy, {1, 9}, 0.3, 9));
  auto *transpose = F_->createTranspose("transpose", rescale2, {1, 0});
  F_->createSave("ret", transpose);

  EXPECT_EQ(F_->getNodes().size(), 6);
  ::glow::optimize(F_, CompilationMode::Infer);
  EXPECT_EQ(F_->getNodes().size(), 5);
}

TEST_F(GraphOptz, mergeRescaleWithArithmeticNode) {
  // Check that Arithmetic operations can be merged with the Rescale.
  Node *input = mod_.createVariable(ElemKind::Int8QTy, {4, 10}, 0.5, 11,
                                    "input", VisibilityKind::Public,
                                    Variable::TrainKind::Broadcast, 15);
  auto *rescale1 = F_->createRescaleQuantized(
      "rescale", input, mod_.uniqueType(ElemKind::Int8QTy, {4, 10}, 0.4, 11));
  auto *add = F_->createAdd("add", rescale1, rescale1);
  auto *rescale2 = F_->createRescaleQuantized(
      "rescale", add, mod_.uniqueType(ElemKind::Int8QTy, {4, 10}, 0.3, 11));
  auto *sub = F_->createSub("sub", rescale2, rescale2);
  auto *rescale3 = F_->createRescaleQuantized(
      "rescale", sub, mod_.uniqueType(ElemKind::Int8QTy, {4, 10}, 0.2, 11));
  auto *mul = F_->createMul("mul", rescale3, rescale3);
  auto *rescale4 = F_->createRescaleQuantized(
      "rescale", mul, mod_.uniqueType(ElemKind::Int8QTy, {4, 10}, 0.1, 11));
  auto *div = F_->createDiv("div", rescale4, rescale4);
  F_->createSave("save", div);

  EXPECT_EQ(F_->getNodes().size(), 9);
  ::glow::optimize(F_, CompilationMode::Infer);
  EXPECT_EQ(F_->getNodes().size(), 5);
}

/// \returns the number of nodes in \p F of kind \p kind.
static unsigned countNodeKind(Function *F, Kinded::Kind kind) {
  unsigned count = 0;
  for (auto &n : F->getNodes()) {
    if (n.getKind() == kind) {
      count++;
    }
  }

  return count;
}

// Check that we are able to merge some small matmuls into a larger one.
TEST_F(GraphOptz, mergeMatMulNodes) {
  Node *input = mod_.createVariable(ElemKind::FloatTy, {10, 10, 10}, "input");
  Node *weight = mod_.createVariable(ElemKind::FloatTy, {10, 10}, "weight");

  // Split the input to a bunch of small slices.
  std::vector<NodeValue> inputs;
  for (size_t i = 0; i < 10; i++) {
    auto *K = F_->createSlice("extract", input, {i, 0, 0}, {i + 1, 10, 10});
    auto *R = F_->createReshape("reshape", K, {10, 10});
    auto *MM = F_->createMatMul("mm", R, weight);
    inputs.push_back(MM);
  }

  auto *cc = F_->createConcat("merge", inputs, 0);
  F_->createSave("save", cc);

  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::MatMulNodeKind), 10);
  ::glow::optimize(F_, CompilationMode::Infer);

  // Check that all of the matmuls are merged into a single matmul node.
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::MatMulNodeKind), 1);
}

// Check that we are able to merge batched adds.
TEST_F(GraphOptz, mergeBANodes) {
  Node *input = mod_.createVariable(ElemKind::FloatTy, {10, 10, 10}, "input");
  Node *slice = mod_.createVariable(ElemKind::FloatTy, {10, 10}, "weight");

  // Split the input to a bunch of small slices.
  std::vector<NodeValue> inputs;
  for (size_t i = 0; i < 10; i++) {
    auto *K = F_->createSlice("extract", input, {i, 0, 0}, {i + 1, 10, 10});
    auto *MM = F_->createBatchedAdd("BA", K, slice);
    inputs.push_back(MM);
  }

  auto *cc = F_->createConcat("merge", inputs, 0);
  F_->createSave("save", cc);

  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::BatchedAddNodeKind), 10);
  ::glow::optimize(F_, CompilationMode::Infer);

  // Check that all of the batched-adds are merged into a single node.
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::BatchedAddNodeKind), 1);
}

// Check that we are able to eliminate concat nodes.
TEST_F(GraphOptz, concatElim) {
  Node *input = mod_.createVariable(ElemKind::FloatTy, {10, 10, 10}, "input");

  // Split the input to a bunch of small slices.
  std::vector<NodeValue> inputs;
  for (size_t i = 0; i < 10; i++) {
    auto *K = F_->createSlice("extract", input, {i, 0, 0}, {i + 1, 10, 10});
    // Insert the nodes in reverse order to make sure that we can catch
    // non-consecutive graph-order slices.
    inputs.insert(inputs.begin(), K);
  }

  auto *cc = F_->createConcat("merge", inputs, 0);
  F_->createSave("save", cc);

  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::SliceNodeKind), 10);
  ::glow::optimize(F_, CompilationMode::Infer);

  // Check that the concat node is gone.
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::ConcatNodeKind), 0);
}

// Check the transformation Concat(Reshape(x) * N) -> Reshape(Concat(x * N)).
TEST_F(GraphOptz, concatReshapes) {
  const size_t shape1[] = {10, 20};
  const size_t shape2[] = {10, 1, 20};
  const size_t shape3[] = {5, 40};
  llvm::SmallVector<NodeValue, 10> inputs1;
  llvm::SmallVector<NodeValue, 10> inputs2;
  for (size_t i = 0; i < 10; i++) {
    // 10 reshape nodes that transform from {10,20} to {10, 1, 20}.
    // And a ConcatNode concatenates the outputs of reshape at 2nd dim.
    // The optimization would kick in, as the size of sub-tensor  of original
    // ConcatNode (before opt) is  20, which can be obtained from the
    // dims of {10,20}.
    Node *var = F_->getParent()->createVariable(ElemKind::FloatTy, shape1,
                                                "input" + std::to_string(i));
    auto *RN = F_->createReshape("reshape" + std::to_string(i), var, shape2);
    inputs1.push_back(RN);
  }
  auto *concatNode1 = F_->createConcat("concat", inputs1, 1);
  for (size_t i = 0; i < 10; i++) {
    // 10 reshape nodes that transform from {5,40} to {10, 1, 20}.
    // And a ConcatNode concatenates the outputs of reshape at 2nd dim.
    // The optimization would NOT kick in, as the size of sub-tensor of original
    // ConcatNode (before opt) is 20, which can not be obtained from the
    // dims of {5,40}.
    Node *var = F_->getParent()->createVariable(ElemKind::FloatTy, shape3,
                                                "input" + std::to_string(i));
    auto *RN = F_->createReshape("reshape" + std::to_string(i), var, shape2);
    inputs2.push_back(RN);
  }
  auto *concatNode2 = F_->createConcat("concat", inputs2, 1);
  auto outputShape = concatNode1->getResult().dims();
  // Need to dereference the RN vectors, otherwise the user number of those
  // nodes would always be positive, making them unable to be removed by DCE.
  inputs1.clear();
  inputs2.clear();

  auto *addNode = F_->createAdd("add", concatNode1, concatNode2);
  auto *O = F_->createSave("ret", addNode);

  EXPECT_EQ(F_->getNodes().size(), 24);

  ::glow::optimize(F_, CompilationMode::Infer);

  // After optimization, we expect to see only 15 nodes. All 10 of the reshapes
  // that were the inputs to the first original concat node (concatNode1) are
  // removed, and a single new reshape is added after the new concat.
  EXPECT_EQ(F_->getNodes().size(), 15);

  // concatNode1 should not exist any more.
  EXPECT_TRUE(std::find_if(F_->getNodes().begin(), F_->getNodes().end(),
                           IsSameNodeAddress(concatNode1)) ==
              F_->getNodes().end());
  // concatNode2 should still exist.
  EXPECT_TRUE(std::find_if(F_->getNodes().begin(), F_->getNodes().end(),
                           IsSameNodeAddress(concatNode2)) !=
              F_->getNodes().end());

  // The first input of addNode should be a Reshape node now, with the same
  // result shape of concatNode1.
  auto *newRN = llvm::dyn_cast<ReshapeNode>(O->getInput()->getNthInput(0));
  ASSERT_TRUE(newRN);
  EXPECT_TRUE(newRN->getType()->dims().equals(outputShape));

  // The input of newRN should be a ConcatNode now.
  auto *newCN =
      llvm::dyn_cast<ConcatNode>(O->getInput()->getNthInput(0)->getNthInput(0));
  ASSERT_TRUE(newCN);
}
