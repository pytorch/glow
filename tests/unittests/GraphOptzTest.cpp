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
#include "glow/Graph/PlaceholderBindings.h"
#include "glow/IR/IR.h"
#include "glow/Optimizer/GraphOptimizer/GraphOptimizer.h"

#include "gtest/gtest.h"

using namespace glow;

class GraphOptz : public ::testing::Test {
public:
  GraphOptz() { F_ = mod_.createFunction("main"); }

protected:
  Module mod_;
  Function *F_;
  PlaceholderBindings bindings_;
};

class GraphFold : public GraphOptz {};

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

TEST_F(GraphOptz, DCE) {
  Node *K = mod_.createPlaceholder(ElemKind::FloatTy, {4, 320, 200, 3}, "input",
                                   false);

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
  EXPECT_EQ(mod_.getConstants().size(), 0);
}

/// Check that predicated instructions are DCE'ed like
/// regular instructions.
TEST_F(GraphOptz, DCEwithPredicate) {
  Node *K = mod_.createPlaceholder(ElemKind::FloatTy, {4, 320, 200, 3}, "input",
                                   false);
  Node *predicatedBatch =
      mod_.createPlaceholder(ElemKind::FloatTy, {4}, "predicate", true);
  for (int i = 0; i < 40; i++) {
    K = F_->createRELU("relu", K);
    K->setPredicate(predicatedBatch);
    // Add a graph structure that diverges and converges, to catch algorithms
    // that perform a dump recursive scan.
    K = F_->createAdd("arith", K, K);
    K->setPredicate(predicatedBatch);
  }

  // Check that we know how many nodes we've created.
  EXPECT_EQ(F_->getNodes().size(), 80);

  // Optimize all of the dead code.
  ::glow::optimize(F_, CompilationMode::Infer);

  //  All of the nodes are gone.
  EXPECT_EQ(F_->getNodes().size(), 0);
  EXPECT_EQ(mod_.getConstants().size(), 0);
}

TEST_F(GraphOptz, liveCodeNotEliminated) {
  Node *K = mod_.createPlaceholder(ElemKind::FloatTy, {4, 320, 200, 3}, "input",
                                   false);
  auto *Ex = mod_.createPlaceholder(ElemKind::Int64ITy, {4, 1}, "Ex", false);

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
  EXPECT_EQ(mod_.getPlaceholders().size(), 3);
}

TEST_F(GraphOptz, optimizeBatchNormAfterConv) {
  Node *A =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 10, 20, 3}, "A", false);
  Node *CV = F_->createConv(bindings_, "conv", A, 16, 5, 1, 2, 1);
  Node *BN =
      F_->createBatchNormalization(bindings_, "batch", CV, 3, 0.0001, 0.9);
  F_->createSave("ret", BN);

  EXPECT_EQ(F_->getNodes().size(), 3);

  ::glow::convertPlaceholdersToConstants(F_, bindings_, {});
  ::glow::optimize(F_, CompilationMode::Infer);
  EXPECT_EQ(F_->getNodes().size(), 2);

  ASSERT_EQ(A->getNumUsers(), 1);
  Node *newCV = A->getUsers().begin()->getUser();
  EXPECT_TRUE(llvm::isa<ConvolutionNode>(newCV));
  ASSERT_EQ(newCV->getNumUsers(), 1);
  Node *save = newCV->getUsers().begin()->getUser();
  EXPECT_TRUE(llvm::isa<SaveNode>(save));
}

/// Verify that the Conv-BatchNorm merging optimization is not impacted by
/// multiple users on the filter/bias.
TEST_F(GraphOptz, optimizeBatchNormAfterConvMultiple) {
  Placeholder *A =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 10, 20, 3}, "A", false);
  ConvolutionNode *CV = F_->createConv(bindings_, "conv", A, 16, 5, 1, 2, 1);
  BatchNormalizationNode *BN =
      F_->createBatchNormalization(bindings_, "batch", CV, 3, 0.0001, 0.9);
  F_->createSave("ret", BN);

  // Adding these saves means the filter and bias have multiple uses. This
  // should not impact the Conv-BatchNorm merging optimization.
  F_->createSave("saveFilter", CV->getFilter());
  F_->createSave("saveBias", CV->getBias());

  // Three Saves, one Conv, and one BatchNorm.
  EXPECT_EQ(F_->getNodes().size(), 5);

  ::glow::convertPlaceholdersToConstants(F_, bindings_, {});

  // Conv's Filter and Bias, plus BN's Scale, Bias, Mean, and Var.
  EXPECT_EQ(mod_.getConstants().size(), 6);

  ::glow::optimize(F_, CompilationMode::Infer);

  // BatchNorm should have been merged into the Conv.
  EXPECT_EQ(F_->getNodes().size(), 4);

  // Filter and Bias should have been duplicated so that the Conv-BN
  // optimization does not modify the filter/bias being saved, equaling 4
  // Constants. Additionally, the BN's Scale, Bias, Mean, and Var should be
  // eliminated due to the opti.
  EXPECT_EQ(mod_.getConstants().size(), 4);

  ASSERT_EQ(A->getNumUsers(), 1);
  Node *newCV = A->getUsers().begin()->getUser();
  EXPECT_TRUE(llvm::isa<ConvolutionNode>(newCV));
  ASSERT_EQ(newCV->getNumUsers(), 1);
  Node *save = newCV->getUsers().begin()->getUser();
  EXPECT_TRUE(llvm::isa<SaveNode>(save));

  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::BatchNormalizationNodeKind), 0);
}

TEST_F(GraphOptz, optimizeBatchNormAfterConvFP16) {
  Node *A =
      mod_.createPlaceholder(ElemKind::Float16Ty, {1, 10, 20, 3}, "A", false);
  Node *CV = F_->createConv(bindings_, "conv", A, 16, 5, 1, 2, 1);
  Node *BN =
      F_->createBatchNormalization(bindings_, "batch", CV, 3, 0.0001, 0.9);
  F_->createSave("ret", BN);

  EXPECT_EQ(F_->getNodes().size(), 3);

  ::glow::convertPlaceholdersToConstants(F_, bindings_, {});
  ::glow::optimize(F_, CompilationMode::Infer);
  EXPECT_EQ(F_->getNodes().size(), 2);

  ASSERT_EQ(A->getNumUsers(), 1);
  Node *newCV = A->getUsers().begin()->getUser();
  EXPECT_TRUE(llvm::isa<ConvolutionNode>(newCV));
  ASSERT_EQ(newCV->getNumUsers(), 1);
  Node *save = newCV->getUsers().begin()->getUser();
  EXPECT_TRUE(llvm::isa<SaveNode>(save));
}

/// Check that transpose constant folding is done before BatchNorm optimization,
/// which allows to merge BatchNorm into Convolution with transposed weights.
TEST_F(GraphOptz, optimizeBatchNormAfterConvWithTransposedWeights) {
  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 10, 20, 3}, "input", false);
  auto *filter =
      mod_.createPlaceholder(ElemKind::FloatTy, {16, 3, 5, 5}, "filter", false);
  auto *bias = mod_.createPlaceholder(ElemKind::FloatTy, {16}, "bias", false);

  auto *TN = F_->createTranspose("transpose", filter, NCHW2NHWC);
  auto *CV = F_->createConv("conv", input, TN, bias,
                            mod_.uniqueType(ElemKind::FloatTy, {1, 10, 20, 16}),
                            5, 1, 2, 1);
  auto *BN =
      F_->createBatchNormalization(bindings_, "batch", CV, 3, 0.0001, 0.9);
  F_->createSave("ret", BN);

  // Initialize to ensure that constant tensors are not optimized out.
  bindings_.allocate(filter)->getHandle().randomize(-1.0, 1.0, mod_.getPRNG());
  bindings_.allocate(bias)->getHandle().randomize(-1.0, 1.0, mod_.getPRNG());
  ::glow::convertPlaceholdersToConstants(F_, bindings_, {input});

  EXPECT_EQ(F_->getNodes().size(), 4);
  ::glow::optimize(F_, CompilationMode::Infer);
  EXPECT_EQ(F_->getNodes().size(), 2);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::BatchNormalizationNodeKind), 0);
}

/// Check that reshape constant folding is done before BatchNorm optimization,
/// where Reshape is a result of Transpose 2 Reshape optimization,
/// which allows to merge BatchNorm into Convolution with transposed weights.
TEST_F(GraphOptz, optimizeBatchNormAfterConvWithReshapeConst) {
  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 10, 20, 3}, "input", false);
  auto *filter =
      mod_.createPlaceholder(ElemKind::FloatTy, {5, 5, 3, 1}, "filter", false);
  auto *bias = mod_.createPlaceholder(ElemKind::FloatTy, {1}, "bias", false);

  auto *TN = F_->createTranspose("transpose", filter, {3, 0, 1, 2});
  auto *CV = F_->createConv("conv", input, TN, bias,
                            mod_.uniqueType(ElemKind::FloatTy, {1, 10, 20, 1}),
                            5, 1, 2, 1);
  auto *BN =
      F_->createBatchNormalization(bindings_, "batch", CV, 3, 0.0001, 0.9);
  F_->createSave("ret", BN);

  // Initialize to ensure that constant tensors are not optimized out.
  bindings_.allocate(filter)->getHandle().randomize(-1.0, 1.0, mod_.getPRNG());
  bindings_.allocate(bias)->getHandle().randomize(-1.0, 1.0, mod_.getPRNG());
  ::glow::convertPlaceholdersToConstants(F_, bindings_, {input});

  EXPECT_EQ(F_->getNodes().size(), 4);
  ::glow::optimize(F_, CompilationMode::Infer);
  EXPECT_EQ(F_->getNodes().size(), 2);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::BatchNormalizationNodeKind), 0);
}

/// Check that the batch normalization optimization is
/// not blocked by predicates and that it preserves them.
TEST_F(GraphOptz, optimizeBatchNormAfterConvWithPred) {
  Node *A =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 10, 20, 3}, "A", false);
  Node *pred1 =
      mod_.createPlaceholder(ElemKind::FloatTy, {1}, "predicate", false);
  Node *pred2 =
      mod_.createPlaceholder(ElemKind::FloatTy, {1}, "predicate", false);
  Node *CV = F_->createConv(bindings_, "conv", A, 16, 5, 1, 2, 1);
  CV->setPredicate(pred1);
  Node *BN =
      F_->createBatchNormalization(bindings_, "batch", CV, 3, 0.0001, 0.9);
  BN->setPredicate(pred2);
  F_->createSave("ret", BN);

  EXPECT_EQ(F_->getNodes().size(), 3);

  ::glow::convertPlaceholdersToConstants(F_, bindings_, {});
  ::glow::optimize(F_, CompilationMode::Infer);
  EXPECT_EQ(F_->getNodes().size(), 2);

  ASSERT_EQ(A->getNumUsers(), 1);
  Node *newCV = A->getUsers().begin()->getUser();
  EXPECT_TRUE(llvm::isa<ConvolutionNode>(newCV));
  ASSERT_TRUE(newCV->hasPredicate());
  EXPECT_EQ(newCV->getPredicate().getNode(), pred2);
  ASSERT_EQ(newCV->getNumUsers(), 1);
  Node *save = newCV->getUsers().begin()->getUser();
  EXPECT_TRUE(llvm::isa<SaveNode>(save));
}

/// Check CSE will not merge two nodes that have all the same inputs but
/// different predicates.
TEST_F(GraphOptz, cseRespectsPredicates) {
  Node *in = mod_.createPlaceholder(ElemKind::FloatTy, {5}, "in", false);
  Node *pred1 = mod_.createPlaceholder(ElemKind::FloatTy, {1}, "pred", false);
  Node *pred2 = mod_.createPlaceholder(ElemKind::FloatTy, {1}, "pred", false);

  Node *RN1 = F_->createRELU("relu1", in);
  RN1->setPredicate(pred1);
  SaveNode *save1 = F_->createSave("save1", RN1);
  save1->setPredicate(pred1);

  Node *RN2 = F_->createRELU("relu2", in);
  RN2->setPredicate(pred2);
  SaveNode *save2 = F_->createSave("save2", RN2);
  save2->setPredicate(pred2);

  // Two RELUS and two Saves.
  EXPECT_EQ(F_->getNodes().size(), 4);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::ReluNodeKind), 2);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::SaveNodeKind), 2);

  ::glow::convertPlaceholdersToConstants(F_, bindings_, {});
  ::glow::optimize(F_, CompilationMode::Infer);

  // Two RELUS and two Saves should still be there.
  EXPECT_EQ(F_->getNodes().size(), 4);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::ReluNodeKind), 2);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::SaveNodeKind), 2);
}

TEST_F(GraphOptz, optimizeBatchNormAfterConvButConvReused) {
  Node *A =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 10, 20, 3}, "A", false);
  Node *CV = F_->createConv(bindings_, "conv", A, 16, 5, 1, 2, 1);
  Node *BN =
      F_->createBatchNormalization(bindings_, "batch", CV, 3, 0.0001, 0.9);
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
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 10, 20, 3}, "A", false);
  auto *filter =
      mod_.createPlaceholder(ElemKind::FloatTy, {16, 5, 5, 3}, "filter", true);
  auto *bias = mod_.createPlaceholder(ElemKind::FloatTy, {16}, "bias", true);

  ConvolutionNode *CV = F_->createConv(
      "conv", A, filter, bias,
      mod_.uniqueType(ElemKind::FloatTy, {1, 10, 20, 16}), 5, 1, 2, 1);
  auto *beta = mod_.createPlaceholder(ElemKind::FloatTy, {16}, "beta", true);
  auto *gamma = mod_.createPlaceholder(ElemKind::FloatTy, {16}, "gamma", true);

  auto *mean = mod_.createPlaceholder(ElemKind::FloatTy, {16}, "mean", false);
  auto *var = mod_.createPlaceholder(ElemKind::FloatTy, {16}, "var", false);

  Node *BN = F_->createBatchNormalization("batch", CV, beta, gamma, mean, var,
                                          3, 0.0001, 0.9);
  SaveNode *ret = F_->createSave("ret", BN);
  SaveNode *filterSave = F_->createSave("filterSave", CV->getFilter());

  EXPECT_EQ(F_->getNodes().size(), 4);

  ::glow::optimize(F_, CompilationMode::Infer);
  // Make sure the structure of the graph did not change.
  EXPECT_EQ(F_->getNodes().size(), 4);
  EXPECT_TRUE(llvm::isa<Placeholder>(filterSave->getInput()));
  auto *varFilter = llvm::dyn_cast<Placeholder>(filterSave->getInput());
  EXPECT_EQ(varFilter, CV->getFilter());
  EXPECT_TRUE(llvm::isa<BatchNormalizationNode>(ret->getInput()));
  BatchNormalizationNode *batchNorm =
      llvm::dyn_cast<BatchNormalizationNode>(ret->getInput());
  EXPECT_EQ(batchNorm, BN);
  EXPECT_TRUE(batchNorm && batchNorm->getInput() &&
              batchNorm->getInput().getNode() == CV);
}

TEST_F(GraphOptz, transposeConstant) {
  auto *A =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 10, 20, 3}, "A", false);
  bindings_.allocate(A)->getHandle().randomize(-7.0, 12.0, mod_.getPRNG());
  Tensor transposedA;
  bindings_.get(A)->transpose(&transposedA, {0, 3, 1, 2});
  Node *T = F_->createTranspose("transpose", A, NHWC2NCHW);
  SaveNode *save = F_->createSave("ret", T);
  EXPECT_EQ(F_->getNodes().size(), 2);

  ::glow::convertPlaceholdersToConstants(F_, bindings_, {});
  ::glow::optimize(F_, CompilationMode::Infer);
  ASSERT_EQ(F_->getNodes().size(), 1);
  EXPECT_EQ(&*F_->getNodes().begin(), save);
  Constant *optimizedA = llvm::dyn_cast<Constant>(save->getInput().getNode());
  ASSERT_NE(optimizedA, nullptr);
  // Check that A has been properly transposed.
  EXPECT_TRUE(optimizedA->getPayload().isEqual(transposedA));
}

/// Check that the removing of transposes still happens when
/// predicates are involved.
TEST_F(GraphOptz, transposeConstantWithPredicate) {
  auto *A =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 10, 20, 3}, "A", false);
  auto *pred = mod_.createPlaceholder(ElemKind::FloatTy, {1}, "pred", false);
  bindings_.allocate(A)->getHandle().randomize(-7.0, 12.0, mod_.getPRNG());
  Tensor transposedA;
  bindings_.get(A)->transpose(&transposedA, {0, 3, 1, 2});
  // Arguably, if the transpose doesn't happen because the predicate is false
  // the value of A should be unchanged. However, the semantic of our
  // predicate is that they can be ignored and the program would still
  // be correct, thus this optimization is still legal.
  Node *T = F_->createTranspose("transpose", A, NHWC2NCHW);
  T->setPredicate(pred);
  SaveNode *save = F_->createSave("ret", T);
  save->setPredicate(pred);
  EXPECT_EQ(F_->getNodes().size(), 2);

  ::glow::convertPlaceholdersToConstants(F_, bindings_, {});
  ::glow::optimize(F_, CompilationMode::Infer);
  ASSERT_EQ(F_->getNodes().size(), 1);
  EXPECT_EQ(&*F_->getNodes().begin(), save);
  // We should have kept the predicate on the save node.
  ASSERT_EQ(pred->getNumUsers(), 1);
  EXPECT_EQ(pred->getUsers().begin()->getUser(), save);
  Constant *optimizedA = llvm::dyn_cast<Constant>(save->getInput().getNode());
  ASSERT_NE(optimizedA, nullptr);
  // Check that A has been properly transposed.
  EXPECT_TRUE(optimizedA->getPayload().isEqual(transposedA));
}

TEST_F(GraphOptz, BatchNormAfterConvNotOptimizeForTrain) {
  Node *A =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 10, 20, 3}, "A", false);
  Node *CV = F_->createConv(bindings_, "conv", A, 16, 5, 1, 2, 1);
  Node *BN =
      F_->createBatchNormalization(bindings_, "batch", CV, 3, 0.0001, 0.9);
  F_->createSave("ret", BN);

  EXPECT_EQ(F_->getNodes().size(), 3);

  ::glow::optimize(F_, CompilationMode::Train);
  EXPECT_EQ(F_->getNodes().size(), 3);

  ASSERT_EQ(A->getNumUsers(), 1);
  Node *curCV = A->getUsers().begin()->getUser();
  EXPECT_EQ(curCV, CV);
  ASSERT_EQ(curCV->getNumUsers(), 1);
  Node *curBN = curCV->getUsers().begin()->getUser();
  EXPECT_EQ(curBN, BN);
  ASSERT_EQ(curBN->getNumUsers(), 1);
  Node *save = curBN->getUsers().begin()->getUser();
  EXPECT_TRUE(llvm::isa<SaveNode>(save));
}

TEST_F(GraphOptz, batchNormAfterConvNotOptimizeWhenMoreThanOneUseOfConv) {
  Node *A =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 10, 20, 3}, "A", false);

  Node *CV = F_->createConv(bindings_, "conv", A, 16, 5, 1, 2, 1);
  Node *BN =
      F_->createBatchNormalization(bindings_, "batch", CV, 3, 0.0001, 0.9);
  SaveNode *convSave = F_->createSave("ret", CV);
  SaveNode *ret = F_->createSave("ret", BN);

  EXPECT_EQ(F_->getNodes().size(), 4);

  ::glow::optimize(F_, CompilationMode::Infer);
  // Make sure the structure of the graph did not change, since the convolution
  // node is used more than once.
  EXPECT_EQ(F_->getNodes().size(), 4);
  ASSERT_TRUE(llvm::isa<ConvolutionNode>(convSave->getInput()));
  ConvolutionNode *conv = llvm::dyn_cast<ConvolutionNode>(convSave->getInput());
  EXPECT_EQ(conv, CV);
  EXPECT_TRUE(llvm::isa<BatchNormalizationNode>(ret->getInput()));
  BatchNormalizationNode *batchNorm =
      llvm::dyn_cast<BatchNormalizationNode>(ret->getInput());
  EXPECT_EQ(batchNorm, BN);
  EXPECT_EQ(batchNorm->getInput().getNode(), CV);
  EXPECT_EQ(conv->getInput().getNode(), A);
}

TEST_F(GraphOptz, sinkTransposeBelowOptimizeBatchNorm) {
  const size_t origDims[] = {1, 5, 10, 15};
  const size_t transposedDims[] = {1, 15, 5, 10};
  Node *A = mod_.createPlaceholder(ElemKind::FloatTy, origDims, "input", false);
  Node *T = F_->createTranspose("transpose", A, NHWC2NCHW);
  BatchNormalizationNode *BN =
      F_->createBatchNormalization(bindings_, "batch", T, 3, 0.0001, 0.9);
  SaveNode *O = F_->createSave("ret", BN);

  EXPECT_EQ(F_->getNodes().size(), 3);
  EXPECT_EQ(BN->getResult().dims(), llvm::makeArrayRef(transposedDims));

  ::glow::optimize(F_, CompilationMode::Infer);

  // Expecting Transpose->Output rather than BN->Output.
  auto *transpose = llvm::dyn_cast<TransposeNode>(O->getInput());
  ASSERT_NE(transpose, nullptr);
  auto *bn = llvm::dyn_cast<BatchNormalizationNode>(transpose->getInput());
  ASSERT_TRUE(bn);
  // Check that the dimensions of the input and output have been
  // updated to compensate the absence of transpose.
  EXPECT_EQ(bn->getResult().dims(), llvm::makeArrayRef(origDims));
  EXPECT_EQ(bn->getInput().dims(), llvm::makeArrayRef(origDims));
  EXPECT_EQ(F_->getNodes().size(), 3);
}

/// Check that the predicates are preserved while sinking transposes
/// through batch normalization.
TEST_F(GraphOptz, sinkTransposeBelowOptimizeBatchNormWithPredicate) {
  const size_t origDims[] = {1, 5, 10, 15};
  const size_t transposedDims[] = {1, 15, 5, 10};
  Node *A = mod_.createPlaceholder(ElemKind::FloatTy, origDims, "input", false);
  Node *pred1 = mod_.createPlaceholder(ElemKind::FloatTy, {1}, "pred", false);
  Node *pred2 = mod_.createPlaceholder(ElemKind::FloatTy, {1}, "pred", false);
  Node *pred3 = mod_.createPlaceholder(ElemKind::FloatTy, {1}, "pred", false);
  Node *T = F_->createTranspose("transpose", A, NHWC2NCHW);
  T->setPredicate(pred1);
  BatchNormalizationNode *BN =
      F_->createBatchNormalization(bindings_, "batch", T, 3, 0.0001, 0.9);
  BN->setPredicate(pred2);
  SaveNode *O = F_->createSave("ret", BN);
  O->setPredicate(pred3);

  EXPECT_EQ(F_->getNodes().size(), 3);
  EXPECT_EQ(BN->getResult().dims(), llvm::makeArrayRef(transposedDims));

  ::glow::optimize(F_, CompilationMode::Infer);

  EXPECT_EQ(O->getPredicate().getNode(), pred3);
  // Expecting Transpose->Output rather than BN->Output.
  auto *transpose = llvm::dyn_cast<TransposeNode>(O->getInput());
  ASSERT_NE(transpose, nullptr);
  EXPECT_EQ(transpose->getPredicate().getNode(), pred2);
  auto *bn = llvm::dyn_cast<BatchNormalizationNode>(transpose->getInput());
  ASSERT_TRUE(bn);
  EXPECT_EQ(bn->getPredicate().getNode(), pred2);
  // Check that the dimensions of the input and output have been
  // updated to compensate the absence of transpose.
  EXPECT_EQ(bn->getResult().dims(), llvm::makeArrayRef(origDims));
  EXPECT_EQ(bn->getInput().dims(), llvm::makeArrayRef(origDims));
  EXPECT_EQ(F_->getNodes().size(), 3);
}

/// Sinks Transpose below Rescale potentially exposing futher optimizations.
/// For example folding Rescale in to Convolution.
TEST_F(GraphOptz, sinkTransposeBelowRescale) {
  // Inputs.
  const size_t origDims[] = {1, 5, 10, 15};
  const size_t transposedDims[] = {1, 15, 5, 10};
  auto *input = mod_.createPlaceholder(ElemKind::Int8QTy, origDims, 0.1, 0,
                                       "input", false);
  auto *filter = mod_.createPlaceholder(ElemKind::Int8QTy, {15, 1, 1, 15}, 0.1,
                                        0, "filter", false);
  auto *bias =
      mod_.createPlaceholder(ElemKind::Int32QTy, {15}, 0.01, 0, "bias", false);

  // Graph.
  ConvolutionNode *conv =
      F_->createConv("conv", input, filter, bias, input->getType(), {1, 1},
                     {1, 1}, {0, 0, 0, 0}, 1);

  auto *T = F_->createTranspose("transpose", conv, NHWC2NCHW);
  auto *RT = mod_.uniqueType(ElemKind::Int8QTy, T->getResult().dims(), 0.2, 0);
  auto *R = F_->createRescaleQuantized("rescale", T, RT);
  SaveNode *O = F_->createSave("ret", R);

  EXPECT_EQ(F_->getNodes().size(), 4);
  EXPECT_EQ(RT->dims(), llvm::makeArrayRef(transposedDims));

  ::glow::optimize(F_, CompilationMode::Infer);

  // Expecting Transpose->Output rather than Rescale->Output.
  auto *transpose = llvm::dyn_cast<TransposeNode>(O->getInput());
  ASSERT_NE(transpose, nullptr);
  ASSERT_TRUE(llvm::isa<ConvolutionNode>(transpose->getInput()));
  auto &convTRInput = transpose->getInput();
  // Check that the dimensions of the input and output have been
  // updated to compensate the absence of transpose.
  EXPECT_EQ(convTRInput.dims(), llvm::makeArrayRef(origDims));
  EXPECT_EQ(convTRInput.getNode()->getNthInput(0).dims(),
            llvm::makeArrayRef(origDims));
  EXPECT_EQ(F_->getNodes().size(), 3);
}

TEST_F(GraphOptz, sinkTransposeBelowRELU) {
  const size_t origDims[] = {1, 5, 10, 15};
  const size_t transposedDims[] = {1, 15, 5, 10};
  Node *A = mod_.createPlaceholder(ElemKind::FloatTy, origDims, "input", false);
  Node *T = F_->createTranspose("transpose", A, NHWC2NCHW);
  ReluNode *K = F_->createRELU("relu", T);
  SaveNode *O = F_->createSave("ret", K);

  EXPECT_EQ(F_->getNodes().size(), 3);
  EXPECT_EQ(K->getResult().dims(), llvm::makeArrayRef(transposedDims));

  ::glow::optimize(F_, CompilationMode::Infer);

  // Expecting Transpose->Output rather than RELU->Output.
  auto *transpose = llvm::dyn_cast<TransposeNode>(O->getInput());
  ASSERT_NE(transpose, nullptr);
  auto *relu = llvm::dyn_cast<ReluNode>(transpose->getInput());
  ASSERT_TRUE(relu);
  // Check that the dimensions of the input and output have been
  // updated to compensate the absence of transpose.
  EXPECT_EQ(relu->getResult().dims(), llvm::makeArrayRef(origDims));
  EXPECT_EQ(relu->getInput().dims(), llvm::makeArrayRef(origDims));
  EXPECT_EQ(F_->getNodes().size(), 3);
}

/// Check that the predicates are preserved while sinking transposes
/// through ReLU.
TEST_F(GraphOptz, sinkTransposeBelowRELUWithPredicate) {
  const size_t origDims[] = {1, 5, 10, 15};
  const size_t transposedDims[] = {1, 15, 5, 10};
  Node *A = mod_.createPlaceholder(ElemKind::FloatTy, origDims, "input", false);
  Node *pred1 = mod_.createPlaceholder(ElemKind::FloatTy, {1}, "pred", false);
  Node *pred2 = mod_.createPlaceholder(ElemKind::FloatTy, {1}, "pred", false);
  Node *pred3 = mod_.createPlaceholder(ElemKind::FloatTy, {1}, "pred", false);
  Node *T = F_->createTranspose("transpose", A, NHWC2NCHW);
  T->setPredicate(pred1);
  ReluNode *K = F_->createRELU("relu", T);
  K->setPredicate(pred2);
  SaveNode *O = F_->createSave("ret", K);
  O->setPredicate(pred3);

  EXPECT_EQ(F_->getNodes().size(), 3);
  EXPECT_EQ(K->getResult().dims(), llvm::makeArrayRef(transposedDims));

  ::glow::optimize(F_, CompilationMode::Infer);

  EXPECT_EQ(O->getPredicate().getNode(), pred3);
  // Expecting Transpose->Output rather than RELU->Output.
  auto *transpose = llvm::dyn_cast<TransposeNode>(O->getInput());
  ASSERT_NE(transpose, nullptr);
  EXPECT_EQ(transpose->getPredicate().getNode(), pred2);
  auto *relu = llvm::dyn_cast<ReluNode>(transpose->getInput());
  ASSERT_TRUE(relu);
  EXPECT_EQ(relu->getPredicate().getNode(), pred2);
  // Check that the dimensions of the input and output have been
  // updated to compensate the absence of transpose.
  EXPECT_EQ(relu->getResult().dims(), llvm::makeArrayRef(origDims));
  EXPECT_EQ(relu->getInput().dims(), llvm::makeArrayRef(origDims));
  EXPECT_EQ(F_->getNodes().size(), 3);
}

TEST_F(GraphOptz, sinkTransposeBelowSigmoid) {
  const size_t origDims[] = {1, 5, 10, 15};
  const size_t transposedDims[] = {1, 15, 5, 10};
  Node *A = mod_.createPlaceholder(ElemKind::FloatTy, origDims, "input", false);
  Node *T = F_->createTranspose("transpose", A, NHWC2NCHW);
  SigmoidNode *SI = F_->createSigmoid("sigmoid", T);
  SaveNode *O = F_->createSave("ret", SI);

  EXPECT_EQ(F_->getNodes().size(), 3);
  EXPECT_EQ(SI->getResult().dims(), llvm::makeArrayRef(transposedDims));

  ::glow::optimize(F_, CompilationMode::Infer);

  // Expecting Transpose->Output rather than Sigmoid->Output.
  auto *transpose = llvm::dyn_cast<TransposeNode>(O->getInput());
  ASSERT_NE(transpose, nullptr);
  auto *si = llvm::dyn_cast<SigmoidNode>(transpose->getInput());
  ASSERT_TRUE(si);
  // Check that the dimensions of the input and output have been
  // updated to compensate the absence of transpose.
  EXPECT_EQ(si->getResult().dims(), llvm::makeArrayRef(origDims));
  EXPECT_EQ(si->getInput().dims(), llvm::makeArrayRef(origDims));
  EXPECT_EQ(F_->getNodes().size(), 3);
}

/// Check that the predicates are preserved while sinking transposes
/// through Sigmoid.
TEST_F(GraphOptz, sinkTransposeBelowSigmoidWithPredicate) {
  const size_t origDims[] = {1, 5, 10, 15};
  const size_t transposedDims[] = {1, 15, 5, 10};
  Node *A = mod_.createPlaceholder(ElemKind::FloatTy, origDims, "input", false);
  Node *pred1 = mod_.createPlaceholder(ElemKind::FloatTy, {1}, "pred", false);
  Node *pred2 = mod_.createPlaceholder(ElemKind::FloatTy, {1}, "pred", false);
  Node *pred3 = mod_.createPlaceholder(ElemKind::FloatTy, {1}, "pred", false);
  Node *T = F_->createTranspose("transpose", A, NHWC2NCHW);
  T->setPredicate(pred1);
  SigmoidNode *SI = F_->createSigmoid("sigmoid", T);
  SI->setPredicate(pred2);
  SaveNode *O = F_->createSave("ret", SI);
  O->setPredicate(pred3);

  EXPECT_EQ(F_->getNodes().size(), 3);
  EXPECT_EQ(SI->getResult().dims(), llvm::makeArrayRef(transposedDims));

  ::glow::optimize(F_, CompilationMode::Infer);

  EXPECT_EQ(O->getPredicate().getNode(), pred3);
  // Expecting Transpose->Output rather than Sigmoid->Output.
  auto *transpose = llvm::dyn_cast<TransposeNode>(O->getInput());
  ASSERT_NE(transpose, nullptr);
  EXPECT_EQ(transpose->getPredicate().getNode(), pred2);
  auto *si = llvm::dyn_cast<SigmoidNode>(transpose->getInput());
  ASSERT_TRUE(si);
  EXPECT_EQ(si->getPredicate().getNode(), pred2);
  // Check that the dimensions of the input and output have been
  // updated to compensate the absence of transpose.
  EXPECT_EQ(si->getResult().dims(), llvm::makeArrayRef(origDims));
  EXPECT_EQ(si->getInput().dims(), llvm::makeArrayRef(origDims));
  EXPECT_EQ(F_->getNodes().size(), 3);
}

TEST_F(GraphOptz, sinkTransposeBelowTanh) {
  const size_t origDims[] = {1, 5, 10, 15};
  const size_t transposedDims[] = {1, 15, 5, 10};
  Node *A = mod_.createPlaceholder(ElemKind::FloatTy, origDims, "input", false);
  Node *T = F_->createTranspose("transpose", A, NHWC2NCHW);
  TanhNode *TN = F_->createTanh("tanh", T);
  SaveNode *O = F_->createSave("ret", TN);

  EXPECT_EQ(F_->getNodes().size(), 3);
  EXPECT_EQ(TN->getResult().dims(), llvm::makeArrayRef(transposedDims));

  ::glow::optimize(F_, CompilationMode::Infer);

  // Expecting Transpose->Output rather than Tanh->Output.
  auto *transpose = llvm::dyn_cast<TransposeNode>(O->getInput());
  ASSERT_NE(transpose, nullptr);
  auto *tn = llvm::dyn_cast<TanhNode>(transpose->getInput());
  ASSERT_TRUE(tn);
  // Check that the dimensions of the input and output have been
  // updated to compensate the absence of transpose.
  EXPECT_EQ(tn->getResult().dims(), llvm::makeArrayRef(origDims));
  EXPECT_EQ(tn->getInput().dims(), llvm::makeArrayRef(origDims));
  EXPECT_EQ(F_->getNodes().size(), 3);
}

TEST_F(GraphOptz, sinkTransposeBelowTanhWithPredicate) {
  const size_t origDims[] = {1, 5, 10, 15};
  const size_t transposedDims[] = {1, 15, 5, 10};
  Node *A = mod_.createPlaceholder(ElemKind::FloatTy, origDims, "input", false);
  Node *pred1 = mod_.createPlaceholder(ElemKind::FloatTy, {1}, "pred", false);
  Node *pred2 = mod_.createPlaceholder(ElemKind::FloatTy, {1}, "pred", false);
  Node *pred3 = mod_.createPlaceholder(ElemKind::FloatTy, {1}, "pred", false);
  Node *T = F_->createTranspose("transpose", A, NHWC2NCHW);
  T->setPredicate(pred1);
  TanhNode *TN = F_->createTanh("tanh", T);
  TN->setPredicate(pred2);
  SaveNode *O = F_->createSave("ret", TN);
  O->setPredicate(pred3);

  EXPECT_EQ(F_->getNodes().size(), 3);
  EXPECT_EQ(TN->getResult().dims(), llvm::makeArrayRef(transposedDims));

  ::glow::optimize(F_, CompilationMode::Infer);

  EXPECT_EQ(O->getPredicate().getNode(), pred3);
  // Expecting Transpose->Output rather than Tanh->Output.
  auto *transpose = llvm::dyn_cast<TransposeNode>(O->getInput());
  ASSERT_NE(transpose, nullptr);
  EXPECT_EQ(transpose->getPredicate().getNode(), pred2);
  auto *tn = llvm::dyn_cast<TanhNode>(transpose->getInput());
  ASSERT_TRUE(tn);
  EXPECT_EQ(tn->getPredicate().getNode(), pred2);
  // Check that the dimensions of the input and output have been
  // updated to compensate the absence of transpose.
  EXPECT_EQ(tn->getResult().dims(), llvm::makeArrayRef(origDims));
  EXPECT_EQ(tn->getInput().dims(), llvm::makeArrayRef(origDims));
  EXPECT_EQ(F_->getNodes().size(), 3);
}

TEST_F(GraphOptz, cancelTwoTransposes) {
  const size_t origDims[] = {1, 5, 10, 15};
  Node *A = mod_.createPlaceholder(ElemKind::FloatTy, origDims, "input", false);
  Node *T1 = F_->createTranspose("transpose", A, NCHW2NHWC);
  Node *T2 = F_->createTranspose("transpose", T1, NHWC2NCHW);
  ReluNode *K = F_->createRELU("relu", T2);
  SaveNode *save = F_->createSave("ret", K);

  EXPECT_EQ(K->getInput().dims(), llvm::makeArrayRef(origDims));
  EXPECT_EQ(F_->getNodes().size(), 4);

  ::glow::optimize(F_, CompilationMode::Infer);

  EXPECT_EQ(F_->getNodes().size(), 2);
  ReluNode *relu = llvm::dyn_cast<ReluNode>(save->getInput());
  ASSERT_TRUE(relu);
  EXPECT_EQ(relu->getResult().dims(), llvm::makeArrayRef(origDims));
  EXPECT_EQ(relu->getInput().getNode(), A);
}

/// Make sure the predicates don't get in the way of the
/// transpose(transpose) => identity and that they are
/// preserved.
TEST_F(GraphOptz, cancelTwoTransposesWithPredicate) {
  const size_t origDims[] = {1, 5, 10, 15};
  Node *A = mod_.createPlaceholder(ElemKind::FloatTy, origDims, "input", false);
  Node *pred1 = mod_.createPlaceholder(ElemKind::FloatTy, {1}, "pred", false);
  Node *pred2 = mod_.createPlaceholder(ElemKind::FloatTy, {1}, "pred", false);
  Node *pred3 = mod_.createPlaceholder(ElemKind::FloatTy, {1}, "pred", false);
  Node *pred4 = mod_.createPlaceholder(ElemKind::FloatTy, {1}, "pred", false);
  Node *T1 = F_->createTranspose("transpose", A, NCHW2NHWC);
  T1->setPredicate(pred1);
  Node *T2 = F_->createTranspose("transpose", T1, NHWC2NCHW);
  T2->setPredicate(pred2);
  ReluNode *K = F_->createRELU("relu", T2);
  K->setPredicate(pred3);
  SaveNode *save = F_->createSave("ret", K);
  save->setPredicate(pred4);

  EXPECT_EQ(K->getInput().dims(), llvm::makeArrayRef(origDims));
  EXPECT_EQ(F_->getNodes().size(), 4);

  ::glow::optimize(F_, CompilationMode::Infer);

  EXPECT_EQ(F_->getNodes().size(), 2);
  EXPECT_EQ(save->getPredicate().getNode(), pred4);
  ReluNode *relu = llvm::dyn_cast<ReluNode>(save->getInput());
  ASSERT_TRUE(relu);
  EXPECT_EQ(relu->getPredicate().getNode(), pred3);
  EXPECT_EQ(relu->getResult().dims(), llvm::makeArrayRef(origDims));
  EXPECT_EQ(relu->getInput().getNode(), A);
}

TEST_F(GraphOptz, removeIdentityTranspose) {
  const size_t origDims[] = {1, 5, 10, 15};
  Placeholder *A =
      mod_.createPlaceholder(ElemKind::FloatTy, origDims, "input", false);
  TransposeNode *T = F_->createTranspose("transpose", A, {0, 1, 2, 3});
  ReluNode *K = F_->createRELU("relu", T);
  F_->createSave("ret", K);

  EXPECT_EQ(F_->getNodes().size(), 3);
  EXPECT_EQ(K->getInput().getNode(), T);

  ::glow::optimize(F_, CompilationMode::Infer);

  EXPECT_EQ(F_->getNodes().size(), 2);
  EXPECT_EQ(K->getInput().getNode(), A);
  // Make sure we didn't mess up with the dimensions of the
  // variable while eliminating the transpose.
  EXPECT_EQ(A->dims(), llvm::makeArrayRef(origDims));
}

/// Check that the predicates don't get in the way of
/// the identity transpose removal, while still being
/// preserved.
TEST_F(GraphOptz, removeIdentityTransposeWithPredicate) {
  const size_t origDims[] = {1, 5, 10, 15};
  Placeholder *A =
      mod_.createPlaceholder(ElemKind::FloatTy, origDims, "input", false);
  Placeholder *pred1 =
      mod_.createPlaceholder(ElemKind::FloatTy, {1}, "pred", false);
  Placeholder *pred2 =
      mod_.createPlaceholder(ElemKind::FloatTy, {1}, "pred", false);
  Placeholder *pred3 =
      mod_.createPlaceholder(ElemKind::FloatTy, {1}, "pred", false);
  TransposeNode *T = F_->createTranspose("transpose", A, {0, 1, 2, 3});
  T->setPredicate(pred1);
  ReluNode *K = F_->createRELU("relu", T);
  K->setPredicate(pred2);
  SaveNode *save = F_->createSave("ret", K);
  save->setPredicate(pred3);

  EXPECT_EQ(F_->getNodes().size(), 3);
  EXPECT_EQ(K->getInput().getNode(), T);

  ::glow::optimize(F_, CompilationMode::Infer);
  EXPECT_EQ(F_->getNodes().size(), 2);
  EXPECT_EQ(save->getPredicate().getNode(), pred3);
  EXPECT_EQ(save->getInput().getNode(), K);
  EXPECT_EQ(K->getInput().getNode(), A);
  EXPECT_EQ(K->getPredicate().getNode(), pred2);
  // Make sure we didn't mess up with the dimensions of the
  // variable while eliminating the transpose.
  EXPECT_EQ(A->dims(), llvm::makeArrayRef(origDims));
}

/// Check that consecutive non-inverse transposes are merged
/// into an equivalent single transpose node.
TEST_F(GraphOptz, mergeNonInverseTransposes) {
  const size_t origDims[] = {1, 5, 10, 15};
  const size_t finalDims[] = {5, 1, 15, 10};

  Node *A = mod_.createPlaceholder(ElemKind::FloatTy, origDims, "input", false);
  TransposeNode *T1 = F_->createTranspose("transpose", A, {0, 3, 2, 1});
  TransposeNode *T2 = F_->createTranspose("transpose", T1, {0, 2, 3, 1});
  TransposeNode *T3 = F_->createTranspose("transpose", T2, {1, 0, 3, 2});
  TransposeNode *T4 = F_->createTranspose("transpose", T3, {3, 1, 2, 0});

  // Intermediate dims after each tranpose
  // Initial : {1, 5, 10, 15}
  // After T1: {1, 15, 10, 5}
  // After T2: {1, 10, 5, 15}
  // After T3: {10, 1, 15, 5}
  // After T4: {5, 1, 15, 10}

  SaveNode *save = F_->createSave("ret", T4);

  EXPECT_EQ(F_->getNodes().size(), 5);

  ::glow::optimize(F_, CompilationMode::Infer);

  auto *TR = llvm::dyn_cast<TransposeNode>(save->getInput());
  ASSERT_NE(TR, nullptr);

  EXPECT_EQ(F_->getNodes().size(), 2);
  EXPECT_EQ(TR->getResult().dims(), llvm::makeArrayRef(finalDims));
  EXPECT_EQ(A->getNthResult(0).dims(), llvm::makeArrayRef(origDims));
  EXPECT_EQ(TR->getInput().getNode(), A);
}

TEST_F(GraphOptz, sinkTransposeBelowArithmeticNodes) {
  const size_t origDims[] = {1, 5, 10, 15};
  Node *A1 =
      mod_.createPlaceholder(ElemKind::FloatTy, origDims, "input1", false);
  Node *A2 =
      mod_.createPlaceholder(ElemKind::FloatTy, origDims, "input2", false);
  Node *T1 = F_->createTranspose("transpose1", A1, NHWC2NCHW);
  Node *T2 = F_->createTranspose("transpose2", A2, NHWC2NCHW);
  Node *K = F_->createAdd("arith", T1, T2);
  SaveNode *O = F_->createSave("ret", K);

  EXPECT_EQ(F_->getNodes().size(), 4);

  ::glow::optimize(F_, CompilationMode::Infer);

  // Expecting Transpose->Output rather than Add->Output.
  auto *transpose = llvm::dyn_cast<TransposeNode>(O->getInput());
  ASSERT_NE(transpose, nullptr);
  auto *add = llvm::dyn_cast<AddNode>(transpose->getInput());
  ASSERT_TRUE(add);
  // Check that the dimensions of the input and output have been
  // updated to compensate the absence of transpose.
  EXPECT_EQ(add->getResult().dims(), llvm::makeArrayRef(origDims));
  EXPECT_EQ(add->getRHS().dims(), llvm::makeArrayRef(origDims));
  EXPECT_EQ(add->getLHS().dims(), llvm::makeArrayRef(origDims));
  EXPECT_EQ(add->getLHS().getNode(), A1);
  EXPECT_EQ(add->getRHS().getNode(), A2);

  EXPECT_EQ(F_->getNodes().size(), 3);
}

/// Check that the predicates are properly preserved while doing
/// the add(transpose, transpose) => transpose(add).
TEST_F(GraphOptz, sinkTransposeBelowArithmeticNodesWithPredicate) {
  const size_t origDims[] = {1, 5, 10, 15};
  Node *A1 =
      mod_.createPlaceholder(ElemKind::FloatTy, origDims, "input1", false);
  Node *A2 =
      mod_.createPlaceholder(ElemKind::FloatTy, origDims, "input2", false);
  Node *pred1 = mod_.createPlaceholder(ElemKind::FloatTy, {1}, "pred", false);
  Node *pred2 = mod_.createPlaceholder(ElemKind::FloatTy, {1}, "pred", false);
  Node *pred3 = mod_.createPlaceholder(ElemKind::FloatTy, {1}, "pred", false);
  Node *pred4 = mod_.createPlaceholder(ElemKind::FloatTy, {1}, "pred", false);
  Node *T1 = F_->createTranspose("transpose1", A1, NHWC2NCHW);
  T1->setPredicate(pred1);
  Node *T2 = F_->createTranspose("transpose2", A2, NHWC2NCHW);
  T2->setPredicate(pred2);
  Node *K = F_->createAdd("arith", T1, T2);
  K->setPredicate(pred3);
  SaveNode *O = F_->createSave("ret", K);
  O->setPredicate(pred4);

  EXPECT_EQ(F_->getNodes().size(), 4);

  ::glow::optimize(F_, CompilationMode::Infer);

  EXPECT_EQ(O->getPredicate().getNode(), pred4);
  // Expecting Transpose->Output rather than Add->Output.
  auto *transpose = llvm::dyn_cast<TransposeNode>(O->getInput());
  ASSERT_NE(transpose, nullptr);
  EXPECT_EQ(transpose->getPredicate().getNode(), pred3);
  auto *add = llvm::dyn_cast<AddNode>(transpose->getInput());
  ASSERT_TRUE(add);
  EXPECT_EQ(add->getPredicate().getNode(), pred3);
  // Check that the dimensions of the input and output have been
  // updated to compensate the absence of transpose.
  EXPECT_EQ(add->getResult().dims(), llvm::makeArrayRef(origDims));
  EXPECT_EQ(add->getRHS().dims(), llvm::makeArrayRef(origDims));
  EXPECT_EQ(add->getLHS().dims(), llvm::makeArrayRef(origDims));
  EXPECT_EQ(add->getLHS().getNode(), A1);
  EXPECT_EQ(add->getRHS().getNode(), A2);

  EXPECT_EQ(F_->getNodes().size(), 3);
}

TEST_F(GraphOptz, sinkReluBelowConcatNodes) {
  const size_t origDims[] = {1, 5, 10, 15};
  const size_t origDimsConcat[] = {1, 10, 10, 15};
  Node *A1 =
      mod_.createPlaceholder(ElemKind::FloatTy, origDims, "input1", false);
  Node *A2 =
      mod_.createPlaceholder(ElemKind::FloatTy, origDims, "input2", false);
  Node *R1 = F_->createRELU("relu1", A1);
  Node *R2 = F_->createRELU("relu2", A2);
  Node *CN = F_->createConcat("concat", {R1, R2}, 1);
  SaveNode *O = F_->createSave("ret", CN);

  EXPECT_EQ(F_->getNodes().size(), 4);

  ::glow::optimize(F_, CompilationMode::Infer);

  // Expecting RELU->Output rather than Concat->Output.
  auto *relu = llvm::dyn_cast<ReluNode>(O->getInput());
  ASSERT_NE(relu, nullptr);
  auto *concat = llvm::dyn_cast<ConcatNode>(relu->getInput());
  ASSERT_TRUE(concat);
  // Check that the dimensions of the input and output have been
  // updated to compensate the absence of transpose.
  EXPECT_EQ(concat->getResult().dims(), llvm::makeArrayRef(origDimsConcat));
  EXPECT_EQ(concat->getInputs()[0].dims(), llvm::makeArrayRef(origDims));
  EXPECT_EQ(concat->getInputs()[1].dims(), llvm::makeArrayRef(origDims));
  EXPECT_EQ(concat->getInputs()[0].getNode(), A1);
  EXPECT_EQ(concat->getInputs()[1].getNode(), A2);

  EXPECT_EQ(F_->getNodes().size(), 3);
}

/// Check that the predicates are properly preserved while doing
/// the sinking of relu nodes.
TEST_F(GraphOptz, sinkReluBelowConcatNodesWithPredicate) {
  const size_t origDims[] = {1, 5, 10, 15};
  const size_t origDimsConcat[] = {1, 10, 10, 15};
  Node *A1 =
      mod_.createPlaceholder(ElemKind::FloatTy, origDims, "input1", false);
  Node *A2 =
      mod_.createPlaceholder(ElemKind::FloatTy, origDims, "input2", false);
  Node *pred1 = mod_.createPlaceholder(ElemKind::FloatTy, {1}, "pred", false);
  Node *pred2 = mod_.createPlaceholder(ElemKind::FloatTy, {1}, "pred", false);
  Node *pred3 = mod_.createPlaceholder(ElemKind::FloatTy, {1}, "pred", false);
  Node *pred4 = mod_.createPlaceholder(ElemKind::FloatTy, {1}, "pred", false);
  Node *R1 = F_->createRELU("relu1", A1);
  R1->setPredicate(pred1);
  Node *R2 = F_->createRELU("relu2", A2);
  R2->setPredicate(pred2);
  Node *CN = F_->createConcat("concat", {R1, R2}, 1);
  CN->setPredicate(pred3);
  SaveNode *O = F_->createSave("ret", CN);
  O->setPredicate(pred4);

  EXPECT_EQ(F_->getNodes().size(), 4);

  ::glow::optimize(F_, CompilationMode::Infer);

  // Expecting RELU->Output rather than Concat->Output.
  EXPECT_EQ(O->getPredicate().getNode(), pred4);
  auto *relu = llvm::dyn_cast<ReluNode>(O->getInput());
  ASSERT_NE(relu, nullptr);
  EXPECT_EQ(relu->getPredicate().getNode(), pred3);
  auto *concat = llvm::dyn_cast<ConcatNode>(relu->getInput());
  ASSERT_TRUE(concat);
  EXPECT_EQ(concat->getPredicate().getNode(), pred3);
  // Check that the dimensions of the input and output have been
  // updated to compensate the absence of transpose.
  EXPECT_EQ(concat->getResult().dims(), llvm::makeArrayRef(origDimsConcat));
  EXPECT_EQ(concat->getInputs()[0].dims(), llvm::makeArrayRef(origDims));
  EXPECT_EQ(concat->getInputs()[1].dims(), llvm::makeArrayRef(origDims));
  EXPECT_EQ(concat->getInputs()[0].getNode(), A1);
  EXPECT_EQ(concat->getInputs()[1].getNode(), A2);

  EXPECT_EQ(F_->getNodes().size(), 3);
}

TEST_F(GraphOptz, sinkTransposeBelowConcatNodes) {
  const size_t origDims[] = {1, 5, 10, 15};
  const size_t origDimsConcat[] = {1, 5, 20, 15};
  Node *A1 =
      mod_.createPlaceholder(ElemKind::FloatTy, origDims, "input1", false);
  Node *A2 =
      mod_.createPlaceholder(ElemKind::FloatTy, origDims, "input2", false);
  Node *T1 = F_->createTranspose("transpose", A1, NCHW2NHWC);
  Node *T2 = F_->createTranspose("transpose", A2, NCHW2NHWC);
  Node *CN = F_->createConcat("concat", {T1, T2}, 1);
  SaveNode *O = F_->createSave("ret", CN);

  EXPECT_EQ(F_->getNodes().size(), 4);

  ::glow::optimize(F_, CompilationMode::Infer);

  // Expecting Transpose->Output rather than Add->Output.
  auto *transpose = llvm::dyn_cast<TransposeNode>(O->getInput());
  ASSERT_NE(transpose, nullptr);
  auto *concat = llvm::dyn_cast<ConcatNode>(transpose->getInput());
  ASSERT_TRUE(concat);
  // Check that the dimensions of the input and output have been
  // updated to compensate the absence of transpose.
  EXPECT_EQ(concat->getResult().dims(), llvm::makeArrayRef(origDimsConcat));
  EXPECT_EQ(concat->getInputs()[0].dims(), llvm::makeArrayRef(origDims));
  EXPECT_EQ(concat->getInputs()[1].dims(), llvm::makeArrayRef(origDims));
  EXPECT_EQ(concat->getInputs()[0].getNode(), A1);
  EXPECT_EQ(concat->getInputs()[1].getNode(), A2);

  EXPECT_EQ(F_->getNodes().size(), 3);
}

/// Check that the predicates are properly preserved while doing
/// the concat(transpose, transpose) => transpose(add).
TEST_F(GraphOptz, sinkTransposeBelowConcatWithPredicate) {
  const size_t origDims[] = {1, 5, 10, 15};
  const size_t origDimsConcat[] = {1, 5, 20, 15};
  Node *A1 =
      mod_.createPlaceholder(ElemKind::FloatTy, origDims, "input1", false);
  Node *A2 =
      mod_.createPlaceholder(ElemKind::FloatTy, origDims, "input2", false);
  Node *pred1 = mod_.createPlaceholder(ElemKind::FloatTy, {1}, "pred", false);
  Node *pred2 = mod_.createPlaceholder(ElemKind::FloatTy, {1}, "pred", false);
  Node *pred3 = mod_.createPlaceholder(ElemKind::FloatTy, {1}, "pred", false);
  Node *pred4 = mod_.createPlaceholder(ElemKind::FloatTy, {1}, "pred", false);
  Node *T1 = F_->createTranspose("transpose", A1, NCHW2NHWC);
  T1->setPredicate(pred1);
  Node *T2 = F_->createTranspose("transpose", A2, NCHW2NHWC);
  T2->setPredicate(pred2);
  Node *CN = F_->createConcat("concat", {T1, T2}, 1);
  CN->setPredicate(pred3);
  SaveNode *O = F_->createSave("ret", CN);
  O->setPredicate(pred4);

  EXPECT_EQ(F_->getNodes().size(), 4);

  ::glow::optimize(F_, CompilationMode::Infer);

  EXPECT_EQ(O->getPredicate().getNode(), pred4);
  // Expecting Transpose->Output rather than Add->Output.
  auto *transpose = llvm::dyn_cast<TransposeNode>(O->getInput());
  ASSERT_NE(transpose, nullptr);
  EXPECT_EQ(transpose->getPredicate().getNode(), pred3);
  auto *concat = llvm::dyn_cast<ConcatNode>(transpose->getInput());
  ASSERT_TRUE(concat);
  EXPECT_EQ(concat->getPredicate().getNode(), pred3);
  // Check that the dimensions of the input and output have been
  // updated to compensate the absence of transpose.
  EXPECT_EQ(concat->getResult().dims(), llvm::makeArrayRef(origDimsConcat));
  EXPECT_EQ(concat->getInputs()[0].dims(), llvm::makeArrayRef(origDims));
  EXPECT_EQ(concat->getInputs()[1].dims(), llvm::makeArrayRef(origDims));
  EXPECT_EQ(concat->getInputs()[0].getNode(), A1);
  EXPECT_EQ(concat->getInputs()[1].getNode(), A2);

  EXPECT_EQ(F_->getNodes().size(), 3);
}

TEST_F(GraphOptz, sinkTransposeBelowPad) {
  // The shape of the graph before the optimization.
  const size_t inputDims[] = {1, 5, 10, 15};
  const size_t outTransposeDims[] = {1, 10, 15, 5};
  const size_t outPadDims[] = {5, 18, 25, 11};
  // Padding before the optimization.
  int pads[] = {0, 2, 3, 1, 4, 6, 7, 5};

  // The shape of the graph after the optimization.
  const size_t outPadDimsAfterOptim[] = {5, 11, 18, 25};
  const size_t outTransposeDimsAfterOptims[] = {5, 18, 25, 11};
  // Padding after the optimization.
  int padsAfterOptim[] = {0, 1, 2, 3, 4, 5, 6, 7};

  // Create the initial graph.
  Node *A =
      mod_.createPlaceholder(ElemKind::FloatTy, inputDims, "input", false);
  auto outTy = mod_.uniqueType(ElemKind::FloatTy, outPadDims);
  TransposeNode *T = F_->createTranspose("transpose", A, NCHW2NHWC);
  Node *P = F_->createPad("pad", T, outTy, PaddingMode::CONSTANT, pads, 23.f);
  EXPECT_EQ(T->getResult().dims(), llvm::makeArrayRef(outTransposeDims));
  SaveNode *O = F_->createSave("ret", P);

  EXPECT_EQ(F_->getNodes().size(), 3);

  ::glow::optimize(F_, CompilationMode::Infer);

  // Check the graph structure and additional properties after optimization.
  auto *trans = llvm::dyn_cast<TransposeNode>(O->getInput());
  ASSERT_NE(trans, nullptr);
  EXPECT_EQ(trans->getResult().dims(),
            llvm::makeArrayRef(outTransposeDimsAfterOptims));
  auto *pad = llvm::dyn_cast<PadNode>(trans->getInput().getNode());
  ASSERT_NE(pad, nullptr);

  EXPECT_EQ(pad->getPads(), llvm::makeArrayRef(padsAfterOptim));
  EXPECT_EQ(pad->getResult().dims(), llvm::makeArrayRef(outPadDimsAfterOptim));

  EXPECT_EQ(F_->getNodes().size(), 3);
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
  Node *A1 = mod_.createPlaceholder(ElemKind::FloatTy, {1, 5, 10, 15}, "input1",
                                    false);
  Node *A2 = mod_.createPlaceholder(ElemKind::FloatTy, {1, 5, 10, 15}, "input2",
                                    false);
  Node *A3 = mod_.createPlaceholder(ElemKind::FloatTy, {1, 5, 10, 15}, "input3",
                                    false);
  Node *A4 =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 1, 5, 15}, "input4", false);
  Node *A5 =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 1, 5, 15}, "input5", false);

  Node *CN1 = F_->createConcat("concat1", {A1, A2}, 1);
  Node *CN2 = F_->createConcat("concat2", {A1, CN1}, 1);
  Node *CN3 = F_->createConcat("concat3", {A4, A5}, 2);
  Node *CN4 = F_->createConcat("concat4", {A3, CN2, CN3}, 1);
  Node *O = F_->createSave("ret", CN4);

  EXPECT_EQ(F_->getNodes().size(), 5);

  ::glow::optimize(F_, CompilationMode::Train);

  // It is expected that the optimization transforms
  // concat4(1, A3, concat2(1, A1, concat1(1, A1, A2)), concat3(2, A4, A5))
  // into
  // concat4(1, A3, A1, A1, A2, concat3(2, A4, A5))

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
  Node *A1 = mod_.createPlaceholder(ElemKind::FloatTy, {1, 5, 10, 15}, "input1",
                                    false);
  Node *A2 = mod_.createPlaceholder(ElemKind::FloatTy, {1, 5, 10, 15}, "input2",
                                    false);

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

  EXPECT_TRUE(CN->getResult().getType()->dims().equals({94, 73, 35}));
}

TEST_F(GraphOptz, ZeroArithmetic) {
  // Tests the identities: [0 + X = X] [0 * X = 0] [0 / X = 0] [ X - 0 = X]

  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {4, 10}, "input", true);

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

/// A test that verifies that arithmetic simplification works correctly when
/// the parents need to be simplified prior to the node itself.
TEST_F(GraphOptz, ZeroArithmeticParentsMustBeSimplifiedFirst) {
  auto *input1 =
      mod_.createPlaceholder(ElemKind::FloatTy, {4, 10}, "input1", true);
  auto *input2 =
      mod_.createPlaceholder(ElemKind::FloatTy, {4, 10}, "input2", true);

  // This builds the expression: ((0 * I1) * (0 * I2)) = 0
  // It should be simplified to simply the splat zero node being saved.

  SplatNode *zero = F_->createSplat("zero", input1->getType(), 0.);

  MulNode *mul1 = F_->createMul("mul1", zero, input1); // -> 0
  MulNode *mul2 = F_->createMul("mul2", zero, input2); // -> 0

  MulNode *mul3 = F_->createMul("mul3", mul1, mul2); // -> 0

  SaveNode *O = F_->createSave("ret", mul3);

  // Expect 1 splat, 3 muls, 1 save.
  EXPECT_EQ(F_->getNodes().size(), 5);

  ::glow::optimize(F_, CompilationMode::Infer);

  // Expect all muls to be optimized away, with 1 splat and 1 save left.
  EXPECT_EQ(F_->getNodes().size(), 2);
  EXPECT_TRUE(std::find_if(F_->getNodes().begin(), F_->getNodes().end(),
                           IsSameNodeAddress(O)) != F_->getNodes().end());
  EXPECT_TRUE(std::find_if(F_->getNodes().begin(), F_->getNodes().end(),
                           IsSameNodeAddress(zero)) != F_->getNodes().end());
  EXPECT_EQ(O->getInput().getNode(), zero);
}

/// Tests opts for the identities: [1 * X = X] [X / 1 = X]
TEST_F(GraphOptz, ArithmeticIdentitiesOne) {
  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {4, 10}, "input", true);

  // This builds the expression: (I / 1) * 1:
  SplatNode *one = F_->createSplat("one", input->getType(), 1.);
  DivNode *div = F_->createDiv("div", input, one);
  MulNode *mul = F_->createMul("mul", div, one);
  SaveNode *SN = F_->createSave("ret", mul);

  // Splat, Div, Mul, Save.
  EXPECT_EQ(F_->getNodes().size(), 4);

  ::glow::optimize(F_, CompilationMode::Infer);

  // The expression evaluates to "I", so Save is only node left.
  EXPECT_EQ(F_->getNodes().size(), 1);
  ASSERT_TRUE(std::find_if(F_->getNodes().begin(), F_->getNodes().end(),
                           IsSameNodeAddress(SN)) != F_->getNodes().end());

  // Save node should just save the input.
  EXPECT_TRUE(SN->getInput().getNode() == input);
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

  EXPECT_TRUE(SN->getResult().getType()->dims().equals(shape));
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
  Node *input = F_->getParent()->createPlaceholder(ElemKind::FloatTy, shape,
                                                   "input", true);
  auto *Z1 = F_->createSplat("zero1", &t1, 1.5);
  auto *A1 = F_->createAdd("add1", Z1->getResult().getType(), input, Z1);
  auto *R1 = F_->createReshape("reshape1", Z1, reshape);
  // Z1 is used by R1 and A1.
  // The reshape optimization will thus NOT be able to remove this reshape node
  // (R1).
  auto *R2 = F_->createReshape("reshape2", A1, reshape);
  auto *A2 = F_->createAdd("add", R1->getResult().getType(), R1, R2);
  auto *Z2 = F_->createSplat("zero2", &t1, 2.5);
  auto *R3 = F_->createReshape("reshape3", Z2, reshape);
  // Z2 is only used by R3.
  // The Z2,R3 nodes will be replaced by a new splat node with the shape of R3.
  auto *A3 = F_->createAdd("add", A2->getResult().getType(), A2, R3);
  auto *O = F_->createSave("ret", A3);

  // Before optimization, we have 9 nodes in the graph.
  EXPECT_EQ(F_->getNodes().size(), 9);

  ::glow::optimize(F_, CompilationMode::Infer);

  // After optimization, we expect to see only 8 nodes, as Z2,R2 would be
  // replace by a new splat node.
  EXPECT_EQ(F_->getNodes().size(), 8);

  // The second input of A3 shoule be a splat node with a shape of R3.
  auto *newA3 = llvm::dyn_cast<AddNode>(O->getInput());
  ASSERT_TRUE(newA3);
  auto *SN = llvm::dyn_cast<SplatNode>(newA3->getRHS());
  EXPECT_TRUE(SN);
  EXPECT_TRUE(SN->getResult().getType()->dims().equals(reshape));

  // R1 should still be in the graph.
  EXPECT_TRUE(std::find_if(F_->getNodes().begin(), F_->getNodes().end(),
                           IsSameNodeAddress(R1)) != F_->getNodes().end());

  // R3 and Z2 should not be in the graph any more.
  EXPECT_TRUE(std::find_if(F_->getNodes().begin(), F_->getNodes().end(),
                           IsSameNodeAddress(R3)) == F_->getNodes().end());
  EXPECT_TRUE(std::find_if(F_->getNodes().begin(), F_->getNodes().end(),
                           IsSameNodeAddress(Z2)) == F_->getNodes().end());
}

/// Test the Reshape(Reshape(x)) -> Reshape(x) transformation.
TEST_F(GraphOptz, ReshapeReshapeOpt) {
  const size_t shape[] = {10, 20};
  const size_t reshape1[] = {200, 1};
  const size_t reshape2[] = {200};
  Node *input = F_->getParent()->createPlaceholder(ElemKind::FloatTy, shape,
                                                   "input", true);
  auto *R1 = F_->createReshape("reshape1", input, reshape1);
  auto *R2 = F_->createReshape("reshape2", R1, reshape2);
  auto *O = F_->createSave("ret", R2);

  // Before optimization, we have 2 Reshapes and a Save.
  EXPECT_EQ(F_->getNodes().size(), 3);

  ::glow::optimize(F_, CompilationMode::Infer);

  // After optimization, we expect to see only 1 Reshape and a Save.
  EXPECT_EQ(F_->getNodes().size(), 2);

  // Save should have the new Reshape as input.
  auto *RN = llvm::dyn_cast<ReshapeNode>(O->getInput());
  ASSERT_TRUE(RN);
  // The new Reshape should have the same shape as the original second Reshape.
  EXPECT_TRUE(RN->getResult().getType()->dims().equals(reshape2));

  // R1 and R2 should not be in the graph any more; they were replaced by a
  // single new reshape.
  EXPECT_TRUE(std::find_if(F_->getNodes().begin(), F_->getNodes().end(),
                           IsSameNodeAddress(R1)) == F_->getNodes().end());
  EXPECT_TRUE(std::find_if(F_->getNodes().begin(), F_->getNodes().end(),
                           IsSameNodeAddress(R2)) == F_->getNodes().end());
}

TEST_F(GraphOptz, DCEPublicVars) {
  mod_.createPlaceholder(ElemKind::FloatTy, {4, 320, 200, 3}, "input", true);

  EXPECT_EQ(mod_.getPlaceholders().size(), 1);

  // Optimize all of the dead code.
  ::glow::optimize(F_, CompilationMode::Infer);

  //  Public nodes should not be deleted.
  EXPECT_EQ(mod_.getPlaceholders().size(), 1);
}

TEST_F(GraphOptz, foldQuantizeIntoVar) {
  auto *input = mod_.createPlaceholder(ElemKind::FloatTy, {4}, "input", true);
  *bindings_.allocate(input) = {10, 10, 10, 10};
  auto qType = mod_.uniqueType(ElemKind::Int8QTy, {4}, 2, 0);

  auto *Q = F_->createQuantize("quantize", input, qType);
  auto *S = F_->createSave("save", Q);

  EXPECT_EQ(2, F_->getNodes().size());
  ::glow::convertPlaceholdersToConstants(F_, bindings_, {S->getPlaceholder()});
  ::glow::optimize(F_, CompilationMode::Infer);
  // Quantization node was merged into input var.
  EXPECT_EQ(1, F_->getNodes().size());

  auto quantizedInput = llvm::cast<Constant>(S->getInput());
  auto quantizedValues = quantizedInput->getHandle<int8_t>();
  for (unsigned i = 0; i < 4; ++i) {
    EXPECT_EQ(5, quantizedValues.raw(i));
  }
}

TEST_F(GraphOptz, foldQuantizeIntoVarMultipleUsages) {
  auto *input = mod_.createPlaceholder(ElemKind::FloatTy, {4}, "input", true);
  *bindings_.allocate(input) = {10, 10, 10, 10};
  auto qType = mod_.uniqueType(ElemKind::Int8QTy, {4}, 2, 0);

  auto *Q = F_->createQuantize("quantize", input, qType);
  F_->createSave("save", Q);
  auto clonedF = F_->clone("cloned");

  EXPECT_EQ(2, clonedF->getNodes().size());
  ::glow::convertPlaceholdersToConstants(clonedF, bindings_, {});
  ::glow::optimize(clonedF, CompilationMode::Infer);
  // F_ function should not be affected.
  EXPECT_EQ(2, F_->getNodes().size());

  // Check original var.
  for (unsigned i = 0; i < 4; ++i) {
    EXPECT_EQ(10, bindings_.get(input)->getHandle().raw(i));
  }

  // Quantization node was merged into input var.
  EXPECT_EQ(1, clonedF->getNodes().size());
  auto *save = llvm::dyn_cast<SaveNode>(&clonedF->getNodes().front());
  ASSERT_TRUE(save);
  auto quantizedInput = llvm::cast<Constant>(save->getInput());
  auto quantizedValues = quantizedInput->getHandle<int8_t>();
  for (unsigned i = 0; i < 4; ++i) {
    EXPECT_EQ(5, quantizedValues.raw(i));
  }
}

/// Check that the Quantize(Splat) -> Splat' optimization works.
TEST_F(GraphOptz, foldQuantizeIntoSplat) {
  TypeRef fType = mod_.uniqueType(ElemKind::FloatTy, {4});
  TypeRef qType = mod_.uniqueType(ElemKind::Int8QTy, {4}, 2, 0);

  const float splatVal = 6.0;
  SplatNode *SN = F_->createSplat("splat", fType, splatVal);

  QuantizeNode *Q = F_->createQuantize("quantize", SN, qType);
  SaveNode *S = F_->createSave("save", Q);

  // Splat, quantize, save.
  EXPECT_EQ(3, F_->getNodes().size());

  ::glow::optimize(F_, CompilationMode::Infer);

  // Quantization node was merged into input splat.
  EXPECT_EQ(2, F_->getNodes().size());

  // New quantized splat should exist with same value.
  SplatNode *newSN = llvm::dyn_cast<SplatNode>(S->getInput());
  ASSERT_TRUE(newSN);
  EXPECT_EQ(splatVal, newSN->getValue());
  EXPECT_EQ(qType, newSN->getResult().getType());
}

/// Check that the Dequantize(Splat) -> Splat' optimization works.
TEST_F(GraphOptz, foldDequantizeIntoSplat) {
  TypeRef fType = mod_.uniqueType(ElemKind::FloatTy, {4});
  TypeRef qType = mod_.uniqueType(ElemKind::Int8QTy, {4}, 2, 0);

  const float splatVal = 6.0;
  SplatNode *SN = F_->createSplat("splat", qType, splatVal);

  DequantizeNode *Q = F_->createDequantize("dequantize", SN);
  SaveNode *S = F_->createSave("save", Q);

  // Splat, dequantize, save.
  EXPECT_EQ(3, F_->getNodes().size());

  ::glow::optimize(F_, CompilationMode::Infer);

  // Dequantization node was merged into input splat.
  EXPECT_EQ(2, F_->getNodes().size());

  // New quantized splat should exist with same value.
  SplatNode *newSN = llvm::dyn_cast<SplatNode>(S->getInput());
  ASSERT_TRUE(newSN);
  EXPECT_EQ(splatVal, newSN->getValue());
  EXPECT_EQ(fType, newSN->getResult().getType());
}

/// Check that the Quantize(Splat) -> Splat' optimization works when the Splat
/// has multiple users.
TEST_F(GraphOptz, foldQuantizeIntoSplatMultipleUsers) {
  TypeRef fType = mod_.uniqueType(ElemKind::FloatTy, {4});
  TypeRef qType = mod_.uniqueType(ElemKind::Int8QTy, {4}, 2, 0);

  SplatNode *SN = F_->createSplat("splat", fType, 6.0);

  QuantizeNode *Q = F_->createQuantize("quantize", SN, qType);
  SaveNode *SQ = F_->createSave("saveQ", Q);
  SaveNode *SF = F_->createSave("saveF", SN);

  // Splat, quantize, 2 saves.
  EXPECT_EQ(4, F_->getNodes().size());

  ::glow::optimize(F_, CompilationMode::Infer);

  // Quantization node was merged into input splat creating a new quantized
  // splat, but the original float splat still exists.
  EXPECT_EQ(4, F_->getNodes().size());

  // New quantized splat should exist with same value.
  SplatNode *newSN = llvm::dyn_cast<SplatNode>(SQ->getInput());
  ASSERT_TRUE(newSN);
  EXPECT_EQ(SN->getValue(), newSN->getValue());
  EXPECT_EQ(qType, newSN->getResult().getType());

  // Original float splat should still exist.
  EXPECT_EQ(llvm::dyn_cast<SplatNode>(SF->getInput()), SN);
}

/// Check that an unnecessary rescale gets removed.
TEST_F(GraphOptz, removeUnnecessaryRescale) {
  TypeRef qType = mod_.uniqueType(ElemKind::Int8QTy, {4, 10}, 0.03f, 5);
  Placeholder *input =
      mod_.createPlaceholder(qType, "input", /* isTrainable */ true);
  RescaleQuantizedNode *RQ =
      F_->createRescaleQuantized("rescale", input, qType);
  SaveNode *save = F_->createSave("ret", RQ);

  // RescaleQuantized and Save.
  EXPECT_EQ(F_->getNodes().size(), 2);

  ::glow::optimize(F_, CompilationMode::Infer);

  // Only Save should be left, which saves the Placeholder directly with
  // unchanged quantization parameters.
  EXPECT_EQ(F_->getNodes().size(), 1);
  EXPECT_EQ(save->getInput().getNode(), input);
  EXPECT_EQ(save->getInput().getType(), qType);
}

/// Check that rescale gets correctly merged into a following dequantize node
TEST_F(GraphOptz, mergeRescaleIntoDequantize) {
  // Check that we are combining quantization-dequantization pairs.
  auto *input = mod_.createPlaceholder(ElemKind::Int8QTy, {4, 10}, 0.5, 11,
                                       "input", true);
  auto *qType = mod_.uniqueType(ElemKind::Int8QTy, {4, 10}, 0.03f, 5);
  auto *R = F_->createRescaleQuantized("rescale", input, qType);
  auto *D = F_->createDequantize("dequantize", R);
  F_->createSave("ret", D);

  EXPECT_EQ(F_->getNodes().size(), 3);
  ::glow::optimize(F_, CompilationMode::Infer);

  // Only 2 nodes should remain (Dequantize -> Save)
  EXPECT_EQ(F_->getNodes().size(), 2);
  // Check the graph structure
  auto *SN = F_->getNodeByName("ret");
  EXPECT_NE(nullptr, SN);
  auto *S = llvm::dyn_cast<SaveNode>(SN);
  EXPECT_NE(nullptr, S);
  auto *newDN = S->getInput().getNode();
  EXPECT_NE(nullptr, newDN);
  EXPECT_NE(nullptr, llvm::dyn_cast<DequantizeNode>(newDN));
}

TEST_F(GraphOptz, quantizeToRescale) {
  // Check that we are combining quantization-dequantization pairs.
  auto *input = mod_.createPlaceholder(ElemKind::Int8QTy, {4, 10}, 0.5, 11,
                                       "input", true);

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

  auto *input = mod_.createPlaceholder(ElemKind::Int8QTy, {size}, scale, offset,
                                       "input", true);

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

  Placeholder *LHS =
      mod_.createPlaceholder(ElemKind::Int8QTy, {10}, 0.4, 0, "LHS", true);
  Placeholder *RHS =
      mod_.createPlaceholder(ElemKind::Int8QTy, {10}, 0.3, 0, "RHS", true);

  AddNode *add = F_->createAdd("qAdd", opOutTy, LHS, RHS);
  RescaleQuantizedNode *rescaleAdd =
      F_->createRescaleQuantized("rsAdd", add, rescaleOutTy);
  SaveNode *addSave = F_->createSave("saveAdd", rescaleAdd);

  SubNode *sub = F_->createSub("qSub", opOutTy, LHS, RHS);
  RescaleQuantizedNode *rescaleSub =
      F_->createRescaleQuantized("rsSub", sub, rescaleOutTy);
  SaveNode *subSave = F_->createSave("saveSub", rescaleSub);

  DivNode *div = F_->createDiv("qDiv", opOutTy, LHS, RHS);
  RescaleQuantizedNode *rescaleDiv =
      F_->createRescaleQuantized("rsDiv", div, rescaleOutTy);
  SaveNode *divSave = F_->createSave("saveDiv", rescaleDiv);

  MulNode *mul = F_->createMul("qMul", opOutTy, LHS, RHS);
  RescaleQuantizedNode *rescaleMul =
      F_->createRescaleQuantized("rsMul", mul, rescaleOutTy);
  SaveNode *mulSave = F_->createSave("saveMul", rescaleMul);

  MinNode *min = F_->createMin("qMin", opOutTy, LHS, RHS);
  RescaleQuantizedNode *rescaleMin =
      F_->createRescaleQuantized("rsMin", min, rescaleOutTy);
  SaveNode *minSave = F_->createSave("saveMin", rescaleMin);

  MaxNode *max = F_->createMax("qMax", opOutTy, LHS, RHS);
  RescaleQuantizedNode *rescaleMax =
      F_->createRescaleQuantized("rsMax", max, rescaleOutTy);
  SaveNode *maxSave = F_->createSave("saveMax", rescaleMax);

  // All rescales must be fused into arithmetic operations above.
  ::glow::optimize(F_, CompilationMode::Infer);

  EXPECT_EQ(F_->getNodes().size(), 12);

  EXPECT_EQ(addSave->getInput().getType(), rescaleOutTy);
  EXPECT_EQ(subSave->getInput().getType(), rescaleOutTy);
  EXPECT_EQ(mulSave->getInput().getType(), rescaleOutTy);
  EXPECT_EQ(divSave->getInput().getType(), rescaleOutTy);
  EXPECT_EQ(minSave->getInput().getType(), rescaleOutTy);
  EXPECT_EQ(maxSave->getInput().getType(), rescaleOutTy);
}

/// Check that the Rescale(MatMul) -> MatMul' optimization works correctly.
TEST_F(GraphOptz, FuseRescaleUpIntoMatMul) {
  // This test ensures the fact that fusing of rescale is done.
  auto opOutTy = mod_.uniqueType(ElemKind::Int8QTy, {10}, 1, 0);
  auto rescaleOutTy = mod_.uniqueType(ElemKind::Int8QTy, {10}, 2, 1);

  Placeholder *LHS = mod_.createPlaceholder(ElemKind::Int8QTy, {10}, 0.4, 0,
                                            "LHS", /* isTrainable */ false);
  Placeholder *RHS = mod_.createPlaceholder(ElemKind::Int8QTy, {10}, 0.3, 0,
                                            "RHS", /* isTrainable */ false);

  MatMulNode *MMN = F_->createMatMul("matmul", opOutTy, LHS, RHS);
  RescaleQuantizedNode *rescaleMMN =
      F_->createRescaleQuantized("rsMMN", MMN, rescaleOutTy);
  SaveNode *saveMMN = F_->createSave("saveMMN", rescaleMMN);

  // MatMul, Rescale, Save.
  EXPECT_EQ(F_->getNodes().size(), 3);

  // All rescales must be fused into arithmetic operations above.
  ::glow::optimize(F_, CompilationMode::Infer);

  // Rescale merged up into the MatMul.
  EXPECT_EQ(F_->getNodes().size(), 2);

  MatMulNode *newMMN = llvm::dyn_cast<MatMulNode>(saveMMN->getInput());
  ASSERT_TRUE(newMMN);
  EXPECT_EQ(newMMN->getResult().getType(), rescaleOutTy);
}

/// Check that the Rescale(SparseLengthsWeightedSum) ->
/// SparseLengthsWeightedSum' optimization works correctly.
TEST_F(GraphOptz, FuseRescaleUpIntoSparseLengthsWeightedSum) {
  // This test ensures the fact that fusing of rescale is done.
  TypeRef rescaleOutTy = mod_.uniqueType(ElemKind::Int8QTy, {4}, 2, 1);

  Placeholder *data =
      mod_.createPlaceholder(ElemKind::Int8QTy, {3}, 0.5, 0, "data",
                             /* isTrainable */ false);
  Placeholder *weights = mod_.createPlaceholder(
      ElemKind::Int8QTy, {8}, 0.5, 0, "weights", /* isTrainable */ false);
  Placeholder *indices =
      mod_.createPlaceholder(ElemKind::Int64ITy, {8}, "indices",
                             /* isTrainable */ false);
  Placeholder *lengths =
      mod_.createPlaceholder(ElemKind::Int32ITy, {4}, "lengths",
                             /* isTrainable */ false);

  SparseLengthsWeightedSumNode *SLWS = F_->createSparseLengthsWeightedSum(
      "SLWS", data, weights, indices, lengths);
  RescaleQuantizedNode *rescaleSLWS =
      F_->createRescaleQuantized("rsSLWS", SLWS, rescaleOutTy);
  SaveNode *saveSLWS = F_->createSave("saveSLWS", rescaleSLWS);

  // SparseLengthsWeightedSum, Rescale, Save.
  EXPECT_EQ(F_->getNodes().size(), 3);

  // All rescales must be fused into arithmetic operations above.
  ::glow::optimize(F_, CompilationMode::Infer);

  // Rescale merged up into the SparseLengthsWeightedSum.
  EXPECT_EQ(F_->getNodes().size(), 2);

  SparseLengthsWeightedSumNode *newSLWS =
      llvm::dyn_cast<SparseLengthsWeightedSumNode>(saveSLWS->getInput());
  ASSERT_TRUE(newSLWS);
  EXPECT_EQ(newSLWS->getResult().getType(), rescaleOutTy);
}

TEST_F(GraphOptz, fuseRescaleIntoConv) {
  // This test ensures the fact that fusing of rescale is done.
  auto *input = mod_.createPlaceholder(ElemKind::Int8QTy, {1, 10, 20, 3}, 0.5,
                                       10, "input", true);
  auto *filter = mod_.createPlaceholder(ElemKind::Int8QTy, {16, 5, 5, 3}, 0.5,
                                        10, "filter", true);
  auto *bias =
      mod_.createPlaceholder(ElemKind::Int8QTy, {16}, 0.5, 10, "bias", true);

  auto *rInput = F_->createRescaleQuantized(
      "rescale", input,
      mod_.uniqueType(ElemKind::Int8QTy, {1, 10, 20, 3}, 0.1, -25));
  auto *rFilter = F_->createRescaleQuantized(
      "rescale", filter,
      mod_.uniqueType(ElemKind::Int8QTy, {16, 5, 5, 3}, 0.2, 0));
  auto *rBias = F_->createRescaleQuantized(
      "rescale", bias, mod_.uniqueType(ElemKind::Int8QTy, {16}, 0.3, 25));
  auto *CV = F_->createConv(
      "conv", rInput, rFilter, rBias,
      mod_.uniqueType(ElemKind::Int8QTy, {1, 10, 20, 16}, 0.7, -3), 5, 1, 2, 1);
  auto *rCV = F_->createRescaleQuantized(
      "rescale", CV,
      mod_.uniqueType(ElemKind::Int8QTy, {1, 10, 20, 16}, 0.4, 37));
  F_->createSave("save", rCV);

  // All rescales must be fused into convolution.
  EXPECT_EQ(F_->getNodes().size(), 6);
  ::glow::optimize(F_, CompilationMode::Infer);
  EXPECT_EQ(F_->getNodes().size(), 2);
}

/// This test ensures that if there is a Pad node as input of a Convolution
/// node, Pad gets merges into Convolution.
/// Note that Pads is merged into convolution only when it is compatible with
/// the convolution padding:
/// - Resulting padding after merge is positive
/// - Padding only concerns spatial dimensions
/// - Padding has mode 'constant' with value 0.f
void fusePadIntoConvTest(glow::Module &mod_, glow::Function *F_,
                         llvm::ArrayRef<size_t> inputDims,
                         llvm::ArrayRef<int> pads, unsigned_t convKernelSize,
                         llvm::ArrayRef<unsigned_t> convPads,
                         unsigned_t convStride, unsigned_t convNumKernels) {
  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, inputDims, "input", true);

  // Pad
  size_t outPadDims[4];
  for (int i = 0; i < 4; i++) {
    outPadDims[i] = size_t(ssize_t(inputDims[i]) + pads[i] + pads[4 + i]);
  }
  auto outTy = mod_.uniqueType(ElemKind::FloatTy, outPadDims);
  Node *P =
      F_->createPad("pad", input, outTy, PaddingMode::CONSTANT, pads, 0.f);

  // Convolution
  size_t filterDims[] = {convNumKernels, convKernelSize, convKernelSize,
                         inputDims[3]};
  auto *F =
      mod_.createPlaceholder(ElemKind::FloatTy, filterDims, "filter", true);
  auto *B =
      mod_.createPlaceholder(ElemKind::FloatTy, {convNumKernels}, "bias", true);
  auto *CV = F_->createConv(
      "conv", P, F, B,
      mod_.uniqueType(ElemKind::FloatTy, {outPadDims[0], outPadDims[1],
                                          outPadDims[2], convNumKernels}),
      {convKernelSize, convKernelSize}, {convStride, convStride}, convPads, 1);

  SaveNode *O = F_->createSave("save", CV);

  // The pad node must be merged into convolution.
  EXPECT_EQ(F_->getNodes().size(), 3);
  ::glow::optimize(F_, CompilationMode::Infer);
  EXPECT_EQ(F_->getNodes().size(), 2);

  // Check the graph structure and additional properties after optimization.
  auto *conv = llvm::dyn_cast<ConvolutionNode>(O->getInput());
  ASSERT_NE(conv, nullptr);
  EXPECT_EQ(conv->getResult().dims(),
            llvm::ArrayRef<size_t>(
                {outPadDims[0], outPadDims[1], outPadDims[2], filterDims[0]}));
  unsigned_t expectedPads[4];
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      expectedPads[2 * i + j] =
          unsigned_t(int(convPads[2 * i + j]) + pads[4 * i + (1 + j)]);
    }
  }
  EXPECT_EQ(conv->getPads(), llvm::makeArrayRef(expectedPads));
}

TEST_F(GraphOptz, fusePadIntoConv) {
  fusePadIntoConvTest(mod_, F_, {1, 6, 14, 3} /* inputDims */,
                      {0, 1, 2, 0, 0, 3, 4, 0} /* pads */,
                      5 /* convKernelSize */, {0, 0, 0, 0} /* convPads */,
                      1 /* convStride */, 16 /* convNumKernels */);
}

TEST_F(GraphOptz, fusePadIntoConvNeg1) {
  fusePadIntoConvTest(mod_, F_, {1, 6, 14, 3} /* inputDims */,
                      {0, -1, 2, 0, 0, 3, -2, 0} /* pads */,
                      5 /* convKernelSize */, {3, 0, 2, 5} /* convPads */,
                      1 /* convStride */, 16 /* convNumKernels */);
}

TEST_F(GraphOptz, fusePadIntoConvNeg2) {
  fusePadIntoConvTest(mod_, F_, {1, 6, 14, 3} /* inputDims */,
                      {0, 1, -2, 0, 0, -3, 4, 0} /* pads */,
                      5 /* convKernelSize */, {0, 2, 5, 7} /* convPads */,
                      1 /* convStride */, 16 /* convNumKernels */);
}

/// This test checks that a lowered LeakyRelu is corrected folded:
/// Max(A, Mult(A, Splat)) -> PRelu(Splat)
TEST_F(GraphFold, foldLeakyReluFromSplat) {
  std::vector<size_t> dims = {5, 2};

  auto *input = mod_.createPlaceholder(ElemKind::FloatTy, dims, "input", true);

  const float leakyAlpha = 0.05f;
  auto OutTy = mod_.uniqueType(ElemKind::FloatTy, dims);
  SplatNode *splatNode = F_->createSplat("splat", OutTy, leakyAlpha);
  MulNode *mulNode = F_->createMul("mul", input, splatNode);
  MaxNode *maxNode = F_->createMax("max", input, mulNode);
  SaveNode *output = F_->createSave("save", maxNode);

  EXPECT_EQ(4, F_->getNodes().size());

  ::glow::fold(F_, CompilationMode::Infer);

  // Check the resulting graph after folding.
  EXPECT_EQ(3, F_->getNodes().size());
  auto *newPReluNode = llvm::dyn_cast<PReluNode>(output->getInput());
  ASSERT_TRUE(newPReluNode);
  auto *newSplatNode = llvm::dyn_cast<SplatNode>(newPReluNode->getSlope());
  ASSERT_TRUE(newSplatNode);
  EXPECT_EQ(leakyAlpha, newSplatNode->getValue());
  EXPECT_EQ(input, newPReluNode->getInput());
}

/// This test checks that a lowered LeakyRelu is corrected folded:
/// Max(A, Mult(A, broadcasted Const)) -> PRelu(Splat)
TEST_F(GraphFold, foldLeakyReluFromConst) {
  std::vector<size_t> dims = {5, 2};
  auto *input = mod_.createPlaceholder(ElemKind::FloatTy, dims, "input", true);

  const float leakyAlpha = 0.99f;
  auto *alphaConst = mod_.createConstant(ElemKind::FloatTy, {1}, "alphaConst");
  alphaConst->getHandle() = {leakyAlpha};
  ReshapeNode *reshapeNode = F_->createReshape("reshape", alphaConst, {1, 1});
  TileNode *tileNode1 = F_->createTile("tile1", reshapeNode, 2, 1);
  TileNode *tileNode2 = F_->createTile("tile2", tileNode1, 5, 0);
  MulNode *mulNode = F_->createMul("mul", input, tileNode2);
  MaxNode *maxNode = F_->createMax("max", input, mulNode);
  SaveNode *output = F_->createSave("save", maxNode);

  EXPECT_EQ(6, F_->getNodes().size());

  ::glow::fold(F_, CompilationMode::Infer);

  // Check the resulting graph after folding. Reshape must have been merged into
  // the constant and LeakyRelu must have been folded.
  EXPECT_EQ(3, F_->getNodes().size());
  auto *newPReluNode = llvm::dyn_cast<PReluNode>(output->getInput());
  ASSERT_TRUE(newPReluNode);
  auto *newSplatNode = llvm::dyn_cast<SplatNode>(newPReluNode->getSlope());
  ASSERT_TRUE(newSplatNode);
  EXPECT_EQ(leakyAlpha, newSplatNode->getValue());
  EXPECT_EQ(input, newPReluNode->getInput());
}

/// Testing folding of Reshape->Transpose->Reshape into ChannelShuffle.
TEST_F(GraphFold, foldChannelShuffle) {
  // FIXME: foldChannelShuffle is disabled; see GraphOptimizer.cpp.
  return;

  const size_t inputDims[] = {3, 136, 28, 28};

  Node *K =
      mod_.createPlaceholder(ElemKind::FloatTy, inputDims, "input", false);
  K = F_->createReshape("CS_reshape1", K, {3, 4, 34, 28, 28});
  K = F_->createTranspose("CS_transpose", K, {0, 2, 1, 3, 4});
  K = F_->createReshape("CS_reshape2", K, {3, 136, 28, 28});
  auto *save = F_->createSave("ret", K);

  EXPECT_EQ(F_->getNodes().size(), 4);

  // Fold RN->TR->RN into ChannelShuffle
  ::glow::fold(F_, CompilationMode::Infer);

  ASSERT_EQ(F_->getNodes().size(), 2);

  // Check for ChannelShuffle node.
  auto *CS = llvm::dyn_cast<ChannelShuffleNode>(save->getInput().getNode());
  ASSERT_NE(nullptr, CS);

  // Ensure ChannelShuffle node has the same dimensions as the input.
  EXPECT_EQ(CS->getResult().dims(), llvm::makeArrayRef(inputDims));

  // Ensure Group and Kernel are as expected.
  EXPECT_EQ(CS->getGroup(), 4);
  EXPECT_EQ(CS->getKernel(), 1);
}

TEST_F(GraphFold, NoFoldChannelShuffle) {
  auto Float = ElemKind::FloatTy;
  auto *P = mod_.createPlaceholder(Float, {10, 8928}, "P", false);
  auto *R1 = F_->createReshape("R1", P, {10, 186, 48});
  auto *TR = F_->createTranspose("TR", R1, {0, 2, 1});
  auto *R2 = F_->createReshape("R2", TR, {480, 186});
  auto *save = F_->createSave("save", R2);

  EXPECT_EQ(F_->getNodes().size(), 4);

  ::glow::fold(F_, CompilationMode::Infer);

  EXPECT_EQ(F_->getNodes().size(), 4);
  EXPECT_FALSE(llvm::isa<ChannelShuffleNode>(save->getInput()));
}

/// This test ensures that if there is a RescaleNode whose input has multiple
/// users that the input is not cloned, as this duplicates the node.
TEST_F(GraphOptz, MultipleUsersRescaleCombineNoOpt) {
  auto opOutTy = mod_.uniqueType(ElemKind::Int8QTy, {10}, 1, 0);
  auto rescaleOutTy = mod_.uniqueType(ElemKind::Int8QTy, {10}, 2, 1);

  Node *LHS =
      mod_.createPlaceholder(ElemKind::Int8QTy, {10}, 0.4, 0, "LHS", true);
  Node *RHS =
      mod_.createPlaceholder(ElemKind::Int8QTy, {10}, 0.3, 0, "RHS", true);

  AddNode *AN = F_->createAdd("qAdd", opOutTy, LHS, RHS);
  RescaleQuantizedNode *RQN =
      F_->createRescaleQuantized("rsAdd", AN, rescaleOutTy);
  SaveNode *saveRQN = F_->createSave("saveRQN", RQN);
  SaveNode *saveAN = F_->createSave("saveAN", AN);

  EXPECT_EQ(F_->getNodes().size(), 4);

  ::glow::optimize(F_, CompilationMode::Infer);

  // The graph should be unchanged.
  EXPECT_EQ(F_->getNodes().size(), 4);
  EXPECT_EQ(saveRQN->getInput().getNode(), RQN);
  EXPECT_EQ(RQN->getInput().getNode(), AN);
  EXPECT_EQ(saveAN->getInput().getNode(), AN);
  EXPECT_EQ(AN->getLHS().getNode(), LHS);
  EXPECT_EQ(AN->getRHS().getNode(), RHS);
}

/// This test ensures that fusing of rescale into MatMul is done.
TEST_F(GraphOptz, FuseRescaleIntoMatMul) {
  auto opOutTy = mod_.uniqueType(ElemKind::Int8QTy, {10}, 1, 0);
  auto rescaleOutTy = mod_.uniqueType(ElemKind::Int8QTy, {10}, 2, 1);

  Placeholder *LHS =
      mod_.createPlaceholder(ElemKind::Int8QTy, {10}, 0.4, 0, "LHS", true);
  Placeholder *RHS =
      mod_.createPlaceholder(ElemKind::Int8QTy, {10}, 0.3, 0, "RHS", true);

  RescaleQuantizedNode *LHSR =
      F_->createRescaleQuantized("rs1", LHS, rescaleOutTy);
  RescaleQuantizedNode *RHSR =
      F_->createRescaleQuantized("rs2", RHS, rescaleOutTy);
  MatMulNode *MN = F_->createMatMul("qMatMul", opOutTy, LHSR, RHSR);
  SaveNode *SN = F_->createSave("save", MN);

  // All rescales must be fused into arithmetic operations above.
  ::glow::optimize(F_, CompilationMode::Infer);

  // Only the MatMul and Save should be left.
  EXPECT_EQ(F_->getNodes().size(), 2);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::RescaleQuantizedNodeKind), 0);

  MatMulNode *newMN = llvm::dyn_cast<MatMulNode>(SN->getInput());
  ASSERT_TRUE(newMN);
  Placeholder *LPH = llvm::dyn_cast<Placeholder>(newMN->getLHS());
  EXPECT_EQ(LPH, LHS);
  Placeholder *RPH = llvm::dyn_cast<Placeholder>(newMN->getRHS());
  EXPECT_EQ(RPH, RHS);
}

TEST_F(GraphOptz, sinkRescaledQuantizedNode) {
  // Check that we eliminate rescale nodes by sinking them into other
  // operators.
  auto *input = mod_.createPlaceholder(ElemKind::Int8QTy, {4, 10}, 0.5, 11,
                                       "input", true);

  // slice -> rescale -> reshape -> rescale -> transpose -> maxpool -> save.
  auto *slice = F_->createSlice("slice", input, {0, 0}, {2, 4});
  auto *rescale = F_->createRescaleQuantized(
      "rescale", slice, mod_.uniqueType(ElemKind::Int8QTy, {2, 4}, 0.4, 10));
  auto *reshape = F_->createReshape("reshape", rescale, {1, 2, 2, 2});
  auto *rescale2 = F_->createRescaleQuantized(
      "rescale", reshape,
      mod_.uniqueType(ElemKind::Int8QTy, {1, 2, 2, 2}, 0.3, 9));
  auto *transpose = F_->createTranspose("transpose", rescale2, {0, 2, 3, 1});
  auto *maxpool =
      F_->createMaxPool("maxpool", transpose, {2, 2}, {1, 1}, {0, 0, 0, 0});
  auto *save = F_->createSave("ret", maxpool);

  EXPECT_EQ(F_->getNodes().size(), 7);
  ::glow::optimize(F_, CompilationMode::Infer);
  EXPECT_EQ(F_->getNodes().size(), 6);
  // Check that rescale sank all the way down to the save node.
  EXPECT_TRUE(llvm::dyn_cast<RescaleQuantizedNode>(save->getInput()));
}

TEST_F(GraphOptz, mergeRescaleWithArithmeticNode) {
  // Check that Arithmetic operations can be merged with the Rescale.
  auto *input = mod_.createPlaceholder(ElemKind::Int8QTy, {4, 10}, 0.5, 11,
                                       "input", true);

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

/// Check that Relu can be merged with Rescale.
TEST_F(GraphOptz, mergeRescaleWithRelu) {
  auto *input = mod_.createPlaceholder(ElemKind::Int8QTy, {4, 10}, 0.5, 11,
                                       "input", false);

  auto *rescale1 = F_->createRescaleQuantized(
      "rescale", input, mod_.uniqueType(ElemKind::Int8QTy, {4, 10}, 0.4, 11));
  auto *relu = F_->createRELU("relu", rescale1);
  F_->createSave("save", relu);

  // Rescale, RELU, Save nodes.
  EXPECT_EQ(F_->getNodes().size(), 3);

  ::glow::optimize(F_, CompilationMode::Infer);

  // RELU, Save nodes left; Rescale merged into RELU.
  EXPECT_EQ(F_->getNodes().size(), 2);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::RescaleQuantizedNodeKind), 0);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::ReluNodeKind), 1);
}

// Check that we are able to merge some small matmuls into a larger one.
TEST_F(GraphOptz, mergeMatMulNodes) {
  Node *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {10, 10, 10}, "input", true);
  Node *weight =
      mod_.createPlaceholder(ElemKind::FloatTy, {10, 10}, "weight", true);

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
  Node *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {10, 10, 10}, "input", true);
  Node *slice =
      mod_.createPlaceholder(ElemKind::FloatTy, {10, 10}, "weight", true);

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
  Node *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {10, 10, 10}, "input", true);

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
  const size_t shape1[] = {2, 5, 2, 1, 20};
  const size_t shape2[] = {10, 2, 2, 10};
  const size_t shape3[] = {5, 80};
  llvm::SmallVector<NodeValue, 10> inputs1;
  llvm::SmallVector<NodeValue, 10> inputs2;
  for (size_t i = 0; i < 10; i++) {
    // 10 reshape nodes that transform from {2,5,2,1,20} to {10,2,2,10}.
    // And a ConcatNode concatenates the outputs of reshape at 2nd dim.
    // The optimization would kick in, as the size of trailing dimensions of
    // original ConcatNode (before opt) is 20, and the size of leading
    // dimensions of original ConcatNode (before opt) is 10.
    Node *var = F_->getParent()->createPlaceholder(
        ElemKind::FloatTy, shape1, "input" + std::to_string(i), true);
    auto *RN = F_->createReshape("reshape" + std::to_string(i), var, shape2);
    inputs1.push_back(RN);
  }
  auto *concatNode1 = F_->createConcat("concat", inputs1, 1);
  for (size_t i = 0; i < 10; i++) {
    // 10 reshape nodes that transform from {5,80} to {10,1,2,10}.
    // And a ConcatNode concatenates the outputs of reshape at 2nd dim.
    // The optimization would NOT kick in, as we cannot find the dim that
    // makes the leading/trailing dims same as in the case of the original
    // concat node.
    Node *var = F_->getParent()->createPlaceholder(
        ElemKind::FloatTy, shape3, "input" + std::to_string(i), true);
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

  // After optimization, we expect to see only 15 nodes. All 10 of the
  // reshapes that were the inputs to the first original concat node
  // (concatNode1) are removed, and a single new reshape is added after the
  // new concat.
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
  auto *newAddNode = llvm::dyn_cast<AddNode>(O->getInput());
  ASSERT_TRUE(newAddNode);
  auto *newRN = llvm::dyn_cast<ReshapeNode>(newAddNode->getLHS());
  ASSERT_TRUE(newRN);
  EXPECT_TRUE(newRN->getResult().getType()->dims().equals(outputShape));

  // The input of newRN should be a ConcatNode now.
  auto *newCN = llvm::dyn_cast<ConcatNode>(newRN->getInput());
  ASSERT_TRUE(newCN);
}

/// Check that Variable CSE works correctly, combining small Variables that
/// have the same data.
TEST_F(GraphOptz, VarsCSE) {
  // Create three variables that are Private, are not trainable, and have no
  // writers. The first two variables have the same data, and so should be
  // combined via variable CSE. The third variable differs by the last value,
  // and so should not be combined.
  auto *input1 = mod_.createConstant(ElemKind::FloatTy, {10}, "input1");
  auto *input2 = mod_.createConstant(ElemKind::FloatTy, {10}, "input2");
  auto *input3 = mod_.createConstant(ElemKind::FloatTy, {10}, "input3");
  input1->getHandle() = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  input2->getHandle() = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  input3->getHandle() = {0, 1, 2, 3, 4, 5, 6, 7, 8, -1};

  // Input them each to different nodes, so node CSE does not change them.
  auto *TN = F_->createTanh("tanh", input1);
  auto *SN = F_->createSigmoid("sigmoid", input2);
  auto *RN = F_->createRELU("relu", input3);
  auto *CN = F_->createConcat("concat", {TN, SN, RN}, /* axis */ 0);
  F_->createSave("ret", CN);

  // Initially there are three variables: inputs 1, 2, and 3 (the save uses a
  // placeholder).
  EXPECT_EQ(mod_.getConstants().size(), 3);

  ::glow::optimize(F_, CompilationMode::Infer);

  // Now only two variables are left; input1 and input2 have been combined,
  // but input3 has not.
  EXPECT_EQ(mod_.getConstants().size(), 2);

  // Verify that only one of input1 and input2 exists, and that input3 still
  // exists.
  Constant *varOneOrTwo = nullptr;
  bool foundVarThree = false;
  for (auto *V : mod_.getConstants()) {
    if (V == input1 || V == input2) {
      EXPECT_TRUE(varOneOrTwo == nullptr);
      varOneOrTwo = V;
    } else if (V == input3) {
      foundVarThree = true;
    }
  }
  EXPECT_TRUE(varOneOrTwo != nullptr);
  EXPECT_TRUE(foundVarThree);

  // Verify that the users of the inputs are updated correctly.
  EXPECT_TRUE(TN->getInput().getNode() == varOneOrTwo);
  EXPECT_TRUE(SN->getInput().getNode() == varOneOrTwo);
  EXPECT_TRUE(RN->getInput().getNode() == input3);

  // Verify that whichever input1/input2 is left over has two users TN and SN.
  EXPECT_TRUE(varOneOrTwo->getUsers().size() == 2);
  for (auto &U : varOneOrTwo->getUsers()) {
    auto *N = U.getUser();
    EXPECT_TRUE(N == TN || N == SN);
  }

  // Verify that input3 only has a single user RN.
  ASSERT_TRUE(input3->getUsers().size() == 1);
  EXPECT_TRUE(input3->getUsers().begin()->getUser() == RN);
}

// Verify that constant input canonicalization works correctly when the
// arithmetic nodes have multiple users.
TEST_F(GraphOptz, simplifyArithmeticMultipleUsers) {
  Node *I1 =
      mod_.createPlaceholder(ElemKind::FloatTy, {10, 10, 10}, "input1", false);

  Type t(ElemKind::FloatTy, {10, 10, 10});
  Node *SN = F_->createSplat("one", &t, 1.0);

  // The splat is a constant input to add1 and add2, and is their LHS input.
  // We expect canonicalization to occur during optimization, moving the splat
  // to the RHS for both. Note that add1 has multiple users: add2 and save1.
  Node *AN1 = F_->createAdd("add1", SN, I1);
  Node *AN2 = F_->createAdd("add2", SN, AN1);
  SaveNode *SN1 = F_->createSave("save1", AN1);
  SaveNode *SN2 = F_->createSave("save2", AN2);

  // Five nodes in total: one splat, two adds, and two saves.
  EXPECT_EQ(F_->getNodes().size(), 5);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::SplatNodeKind), 1);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::AddNodeKind), 2);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::SaveNodeKind), 2);

  // input1 has a single user before optimization.
  EXPECT_EQ(I1->getUsers().size(), 1);

  // Simplify nodes will canonicalize add1 and add2, and should replace all
  // their users, without otherwise adding new nodes to the graph/changing the
  // overall structure.
  ::glow::optimize(F_, CompilationMode::Infer);

  // We should have the same five nodes: one splat, two adds, and two saves.
  EXPECT_EQ(F_->getNodes().size(), 5);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::SplatNodeKind), 1);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::AddNodeKind), 2);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::SaveNodeKind), 2);

  // Verify that both add nodes were canonicalized, and that the graph's shape
  // is the same as prior to optimization other than canonicalization.
  AddNode *newAN1 = llvm::dyn_cast<AddNode>(SN1->getInput().getNode());
  ASSERT_TRUE(newAN1 != nullptr);
  EXPECT_TRUE(llvm::isa<Placeholder>(newAN1->getLHS()));
  EXPECT_TRUE(llvm::isa<SplatNode>(newAN1->getRHS()));

  AddNode *newAN2 = llvm::dyn_cast<AddNode>(SN2->getInput().getNode());
  ASSERT_TRUE(newAN2 != nullptr);
  EXPECT_TRUE(llvm::isa<AddNode>(newAN2->getLHS()));
  EXPECT_TRUE(llvm::isa<SplatNode>(newAN2->getRHS()));

  EXPECT_EQ(newAN1, newAN2->getLHS());

  // input1 should still have a single user after optimization.
  EXPECT_EQ(I1->getUsers().size(), 1);
}

/// Test that a concat with a single input is replaced by the input.
TEST_F(GraphOptz, eliminateSingleConcat) {
  Node *input = mod_.createPlaceholder(ElemKind::FloatTy, {10}, "input", false);

  ConcatNode *CN = F_->createConcat("concat1", {input}, 0);
  SaveNode *SN = F_->createSave("ret", CN);

  // The ConcatNode and SaveNode.
  EXPECT_EQ(F_->getNodes().size(), 2);

  ::glow::optimize(F_, CompilationMode::Infer);

  // Just the SaveNode should be left.
  EXPECT_EQ(F_->getNodes().size(), 1);
  ASSERT_TRUE(std::find_if(F_->getNodes().begin(), F_->getNodes().end(),
                           IsSameNodeAddress(SN)) != F_->getNodes().end());

  // Save node should just save the input.
  EXPECT_TRUE(SN->getInput().getNode() == input);
}

/// Test that a reshape of a private variable with one use has the reshape
/// merged into the variable.
TEST_F(GraphOptz, ReshapeConstantOneUse) {
  const size_t shape[] = {10, 20};
  const size_t reshape1[] = {200, 1};
  const size_t reshape2[] = {200};
  auto *input = F_->getParent()->createPlaceholder(ElemKind::FloatTy, shape,
                                                   "input", true);
  bindings_.allocate(input);
  auto *R1 = F_->createReshape("reshape1", input, reshape1);
  auto *R2 = F_->createReshape("reshape2", R1, reshape2);
  auto *O = F_->createSave("ret", R2);

  // Before optimization, we have 2 Reshapes and a Save.
  EXPECT_EQ(F_->getNodes().size(), 3);

  ::glow::convertPlaceholdersToConstants(F_, bindings_, {});
  ::glow::optimize(F_, CompilationMode::Infer);

  // After optimization, we expect to see just a Save.
  EXPECT_EQ(F_->getNodes().size(), 1);

  // Save should have the new Variable as input.
  auto *V = llvm::dyn_cast<Constant>(O->getInput());
  ASSERT_TRUE(V);
  // The new Variable should have the same shape as the original second
  // Reshape.
  EXPECT_TRUE(V->getType()->dims().equals(reshape2));
}

/// Test that Transpose is optimized into Reshape when it moves no data.
TEST_F(GraphOptz, transposeIntoReshapeOptim) {
  auto *batch =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 3, 2, 4}, "batch", false);
  Node *T = F_->createTranspose("transpose", batch, {1, 2, 0, 3});
  SaveNode *O = F_->createSave("ret", T);

  EXPECT_EQ(F_->getNodes().size(), 2);

  ::glow::optimize(F_, CompilationMode::Infer);
  EXPECT_EQ(F_->getNodes().size(), 2);

  // TransposeNode is Optimized into ReshapeNode.
  auto *reshape = llvm::dyn_cast<ReshapeNode>(O->getInput().getNode());
  ASSERT_NE(reshape, nullptr);
}

/// Test that transpose is merged into matmul.
TEST_F(GraphOptz, mergeTransposeIntoMatMul) {
  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 1, 2, 3}, "input", false);
  auto *weights =
      F_->getParent()->createConstant(ElemKind::FloatTy, {6, 1}, "weights");

  weights->getHandle() = {0, 1, 2, 3, 4, 5};
  float newWeightsRef[] = {0, 2, 4, 1, 3, 5};

  auto *TN = F_->createTranspose("transpose", input, NHWC2NCHW);
  auto *RN = F_->createReshape("reshape", TN, {1, 6});
  auto *MMN = F_->createMatMul("matmul", RN, weights);
  auto *SN = F_->createSave("ret", MMN);

  // Transpose + Reshape + MatMul + Save.
  EXPECT_EQ(F_->getNodes().size(), 4);

  ::glow::optimize(F_, CompilationMode::Infer);

  // Reshape + MatMul + Save.
  EXPECT_EQ(F_->getNodes().size(), 3);

  // Check reordered weights.
  auto *newMMN = llvm::dyn_cast<MatMulNode>(SN->getInput());
  ASSERT_TRUE(newMMN != nullptr);
  auto *newW = llvm::dyn_cast<Constant>(newMMN->getRHS());
  ASSERT_TRUE(newW != nullptr);
  for (unsigned i = 0; i < 6; ++i) {
    EXPECT_EQ(newWeightsRef[i], newW->getHandle().raw(i));
  }
}

/// Test that transpose is merged into FullyConnected.
TEST_F(GraphOptz, mergeTransposeIntoFC) {
  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 1, 2, 3}, "input", false);
  auto *weights =
      F_->getParent()->createConstant(ElemKind::FloatTy, {6, 1}, "weights");
  auto *bias = F_->getParent()->createConstant(ElemKind::FloatTy, {1}, "bias");

  weights->getHandle() = {0, 1, 2, 3, 4, 5};
  float newWeightsRef[] = {0, 2, 4, 1, 3, 5};

  auto *TN = F_->createTranspose("transpose", input, NHWC2NCHW);
  auto *RN = F_->createReshape("reshape", TN, {1, 6});
  auto *FCN = F_->createFullyConnected("fc", RN, weights, bias);
  auto *SN = F_->createSave("ret", FCN);

  // Transpose + Reshape + FC + Save.
  EXPECT_EQ(F_->getNodes().size(), 4);

  ::glow::optimize(F_, CompilationMode::Infer);

  // Reshape + FC + Save.
  EXPECT_EQ(F_->getNodes().size(), 3);

  // Check reordered weights.
  auto *newFCN = llvm::dyn_cast<FullyConnectedNode>(SN->getInput());
  ASSERT_TRUE(newFCN != nullptr);
  auto *newW = llvm::dyn_cast<Constant>(newFCN->getWeights());
  ASSERT_TRUE(newW != nullptr);
  for (unsigned i = 0; i < 6; ++i) {
    EXPECT_EQ(newWeightsRef[i], newW->getHandle().raw(i));
  }
}

TEST_F(GraphOptz, ConvertPlaceholdersToConstants) {
  auto *input1 = mod_.createPlaceholder(ElemKind::FloatTy, {1}, "input1", true);
  auto *input2 = mod_.createPlaceholder(ElemKind::FloatTy, {1}, "input2", true);
  auto *input3 = mod_.createPlaceholder(ElemKind::FloatTy, {1}, "input3", true);
  auto *save1 = F_->createSave("save1", input1);
  auto *save2 = F_->createSave("save2", input2);
  auto *save3 = F_->createSave("save3", input3);

  // No variables, six PHs (3 inputs, 3 saves).
  EXPECT_EQ(mod_.getConstants().size(), 0);
  EXPECT_EQ(mod_.getPlaceholders().size(), 6);

  // Allocate two of the three inputs, but mark input2 of them as
  // non-constant.
  bindings_.allocate(input1);
  bindings_.allocate(input2);
  // Don't allocate input3; keep it as a placeholder instead.
  ::glow::convertPlaceholdersToConstants(F_, bindings_, {input2});

  // input1 becomes a variable.
  EXPECT_EQ(mod_.getConstants().size(), 1);
  EXPECT_EQ(mod_.getPlaceholders().size(), 6);

  EXPECT_TRUE(llvm::isa<Constant>(save1->getInput()));
  EXPECT_TRUE(llvm::isa<Placeholder>(save2->getInput()));
  EXPECT_TRUE(llvm::isa<Placeholder>(save3->getInput()));
}

TEST_F(GraphOptz, optimizeSameTypeConversions) {
  auto *input1 = mod_.createPlaceholder(ElemKind::FloatTy, {1}, "input1", true);
  auto *input2 = mod_.createPlaceholder(ElemKind::FloatTy, {1}, "input2", true);
  auto *conv1 = F_->createConvertTo("cast1", input1, input1->getType());
  auto *conv2 = F_->createConvertTo(
      "cast2", input2, mod_.uniqueType(ElemKind::Float16Ty, input2->dims()));
  auto *save1 = F_->createSave("save1", conv1);
  auto *save2 = F_->createSave("save1", conv2);

  // convert_to1 + save1 + convert_to2 + save2 nodes.
  EXPECT_EQ(F_->getNodes().size(), 4);
  EXPECT_TRUE(llvm::isa<ConvertToNode>(save1->getInput()));

  ::glow::optimize(F_, CompilationMode::Infer);

  // save1 + convert_to2 + save2 nodes.
  EXPECT_EQ(F_->getNodes().size(), 3);
  // convert_to1 node should be eliminated, because it converts the node into
  // the same type.
  EXPECT_TRUE(llvm::isa<Placeholder>(save1->getInput()));
  // convert_to1 node should not be eliminated, because it converts the node
  // into a different type.
  EXPECT_TRUE(llvm::isa<ConvertToNode>(save2->getInput()));
  EXPECT_EQ(save2->getInput(), NodeValue(conv2));
}

TEST_F(GraphOptz, dceBeforeOptimizeTranpose) {
  auto *input1 = mod_.createConstant(ElemKind::FloatTy, {5, 10}, "input1");
  // Create an unused node.
  F_->createAdd("add", input1, input1);
  auto *transposedInput1 = F_->createTranspose("transpose", input1, {1, 0});
  auto *save1 = F_->createSave("save1", transposedInput1);

  // add + transpose + save.
  EXPECT_EQ(F_->getNodes().size(), 3);

  ::glow::optimize(F_, CompilationMode::Infer);

  // A single node: save.
  EXPECT_EQ(F_->getNodes().size(), 1);
  // transpose should be eliminated and replaced by the transposed constant.
  EXPECT_TRUE(llvm::isa<Constant>(save1->getInput()));
}

/// Test that Transpose is sunk below ChannelShuffle and cancels with an
/// inverse transpose below the ChannelShuffle. This test models a pattern
/// that has has been observed in shufflenet during graph optimization.
TEST_F(GraphOptz, sinkTransposeBelowChannelShuffleNodesAndEliminate) {
  const size_t inputDims[] = {3, 28, 28, 136};

  Node *K =
      mod_.createPlaceholder(ElemKind::FloatTy, inputDims, "input", false);
  K = F_->createTranspose("unnecessary_transpose_1", K, {0, 3, 1, 2});
  K = F_->createChannelShuffle("channel_shuffle", K, 4, 1);
  K = F_->createTranspose("unnecessary_transpose_2", K, {0, 2, 3, 1});
  auto *save = F_->createSave("ret", K);

  EXPECT_EQ(F_->getNodes().size(), 4);

  // Optimize away the unnecessary transposes.
  optimize(F_, CompilationMode::Infer);

  // Ensure the two unnecessary transposes are gone.
  ASSERT_EQ(F_->getNodes().size(), 2);

  // Check that the channel shuffle node is still there.
  auto *CSN = llvm::dyn_cast<ChannelShuffleNode>(save->getInput().getNode());
  ASSERT_NE(nullptr, CSN);

  // Ensure ChannelShuffle node has the same dimensions as the input.
  EXPECT_EQ(CSN->getResult().dims(), llvm::makeArrayRef(inputDims));

  // Ensure Group and Kernel are as expected.
  EXPECT_EQ(CSN->getGroup(), 4);
  EXPECT_EQ(CSN->getKernel(), 3);
}

/// Test that convertPlaceholdersToConstants works properly with quantized
/// types.
TEST_F(GraphOptz, QuantizedFC) {
  auto *input = mod_.createPlaceholder(ElemKind::Int8QTy, {2, 32}, 1.0, 0,
                                       "input", false);
  auto *weights = mod_.createPlaceholder(ElemKind::Int8QTy, {32, 32}, 1.0, 0,
                                         "weights", false);
  auto *bias =
      mod_.createPlaceholder(ElemKind::Int32QTy, {32}, 1.0, 0, "bias", false);
  auto *output = mod_.createPlaceholder(ElemKind::Int8QTy, {2, 32}, 1.0, 0,
                                        "output", false);

  auto *fc = F_->createFullyConnected("fc", input, weights, bias);
  F_->createSave("save", fc, output);

  bindings_.allocate(input);
  bindings_.allocate(weights);
  bindings_.allocate(bias);
  bindings_.allocate(output);

  glow::convertPlaceholdersToConstants(F_, bindings_, {input, output});
  // Two constants: weight and bias
  EXPECT_EQ(mod_.getConstants().size(), 2);
  // All four placeholders still exist in the module.  The old weight and bias
  // placeholders just aren't hooked up the the Graph F_.
  EXPECT_EQ(mod_.getPlaceholders().size(), 4);
}

/// Test batchedReduceMean optimization using AvgPool.
TEST_F(GraphOptz, convertReduceMean2AvgPool) {
  const size_t dims[] = {2, 2, 2, 2};

  Node *A = mod_.createPlaceholder(ElemKind::FloatTy, dims, "input", false);
  Node *R = F_->createBatchedReduceMean("reduce.mean", A, {2, 3});

  SaveNode *O = F_->createSave("ret", R);

  EXPECT_EQ(F_->getNodes().size(), 2);

  ::glow::optimize(F_, CompilationMode::Infer);

  // Optimization adds 2 transpose nodes and one reshape node.
  EXPECT_EQ(F_->getNodes().size(), 5);

  // Expecting reshape output rather than ReduceMean.
  auto *RN = llvm::dyn_cast<ReshapeNode>(O->getInput());
  ASSERT_NE(RN, nullptr);

  // Expecting Transpose node before Reshape node.
  auto *TN = llvm::dyn_cast<TransposeNode>(RN->getInput());
  ASSERT_NE(TN, nullptr);

  // Expecting AvgPool node before Transpose node.
  auto *APN = llvm::dyn_cast<AvgPoolNode>(TN->getInput());
  ASSERT_NE(APN, nullptr);
}

/// Test Broadcasted RHS BatchMatMul is converted correctly to a single MatMul.
TEST_F(GraphOptz, convertBroadcastedBatchMatMulToMatMul) {
  auto *lhs =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 3, 2}, "lhs", false);
  auto *rhs = mod_.createPlaceholder(ElemKind::FloatTy, {2, 1}, "rhs", false);
  auto *BMMN = F_->createBatchMatMul("BMM", lhs, rhs);
  F_->createSave("save", BMMN);

  // Start with a BatchMatMul, not a MatMul.
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::BatchMatMulNodeKind), 1);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::MatMulNodeKind), 0);

  ::glow::optimize(F_, CompilationMode::Infer);

  // Optimization should replace the BatchMatMul with a single MatMul.
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::MatMulNodeKind), 1);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::BatchMatMulNodeKind), 0);
}

TEST_F(GraphOptz, dceQuantization) {
  auto *lhs =
      mod_.createPlaceholder(ElemKind::Int8QTy, {3, 5}, 0.3, 15, "lhs", false);
  auto *weights =
      mod_.createConstant(ElemKind::Int8QTy, {3, 5}, 0.3, 15, "weights");

  auto *add = F_->createAdd("add", lhs, weights);
  auto *t1 = mod_.uniqueType(ElemKind::Int8QTy, {3, 5}, 0.2, 0);
  auto *rs1 = F_->createRescaleQuantized("rs1", add, t1);
  auto *t2 = mod_.uniqueType(ElemKind::Int8QTy, {3, 5}, 0.1, 1);
  auto *rs2 = F_->createRescaleQuantized("rs2", rs1, t2);
  F_->createSave("save", rs2);

  ::glow::optimize(F_, CompilationMode::Infer);

  EXPECT_EQ(F_->getNodes().size(), 2);
}

TEST_F(GraphOptz, nopRelu) {
  auto *in = mod_.createPlaceholder(ElemKind::Int8QTy, {3, 5}, 0.3, -128, "lhs",
                                    false);

  auto *relu = F_->createRELU("relu", in);
  F_->createSave("save", relu);

  ::glow::optimize(F_, CompilationMode::Infer);

  EXPECT_EQ(F_->getNodes().size(), 1);
}
