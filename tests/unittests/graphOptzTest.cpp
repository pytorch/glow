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

#include "glow/Graph/Context.h"
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
  Context ctx_;
};

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
  EXPECT_EQ(mod_.getVars().size(), 0);
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
  EXPECT_EQ(mod_.getVars().size(), 0);
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
  F_->createSave(ctx_, "ret", K);

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
  Node *CV = F_->createConv(ctx_, "conv", A, 16, 5, 1, 2, 1);
  Node *BN = F_->createBatchNormalization("batch", CV, 3, 0.0001, 0.9);
  F_->createSave(ctx_, "ret", BN);

  EXPECT_EQ(F_->getNodes().size(), 3);

  ::glow::convertPlaceholdersToConstants(F_, ctx_, {});
  ::glow::optimize(F_, CompilationMode::Infer);
  EXPECT_EQ(F_->getNodes().size(), 2);

  ASSERT_EQ(A->getNumUsers(), 1);
  Node *newCV = A->getUsers().begin()->getUser();
  EXPECT_TRUE(llvm::isa<ConvolutionNode>(newCV));
  ASSERT_EQ(newCV->getNumUsers(), 1);
  Node *save = newCV->getUsers().begin()->getUser();
  EXPECT_TRUE(llvm::isa<SaveNode>(save));
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
  Node *CV = F_->createConv(ctx_, "conv", A, 16, 5, 1, 2, 1);
  CV->setPredicate(pred1);
  Node *BN = F_->createBatchNormalization("batch", CV, 3, 0.0001, 0.9);
  BN->setPredicate(pred2);
  F_->createSave(ctx_, "ret", BN);

  EXPECT_EQ(F_->getNodes().size(), 3);

  ::glow::convertPlaceholdersToConstants(F_, ctx_, {});
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

TEST_F(GraphOptz, optimizeBatchNormAfterConvButConvReused) {
  Node *A =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 10, 20, 3}, "A", false);
  Node *CV = F_->createConv(ctx_, "conv", A, 16, 5, 1, 2, 1);
  Node *BN = F_->createBatchNormalization(ctx_, "batch", CV, 3, 0.0001, 0.9);
  SaveNode *ret = F_->createSave(ctx_, "ret", BN);
  SaveNode *convSave = F_->createSave(ctx_, "convSave", CV);

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
  SaveNode *ret = F_->createSave(ctx_, "ret", BN);
  SaveNode *filterSave = F_->createSave(ctx_, "filterSave", CV->getFilter());

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

TEST_F(GraphOptz, transposePrivateVariable) {
  auto *A =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 10, 20, 3}, "A", false);
  ctx_.allocate(A)->getHandle().randomize(-7.0, 12.0, mod_.getPRNG());
  Tensor transposedA;
  ctx_.get(A)->transpose(&transposedA, {0, 3, 1, 2});
  Node *T = F_->createTranspose("transpose", A, NHWC2NCHW);
  SaveNode *save = F_->createSave(ctx_, "ret", T);
  EXPECT_EQ(F_->getNodes().size(), 2);

  ::glow::convertPlaceholdersToConstants(F_, ctx_, {});
  ::glow::optimize(F_, CompilationMode::Infer);
  ASSERT_EQ(F_->getNodes().size(), 1);
  EXPECT_EQ(&*F_->getNodes().begin(), save);
  Variable *optimizedA = llvm::dyn_cast<Variable>(save->getInput().getNode());
  ASSERT_NE(optimizedA, nullptr);
  // Check that A has been properly transposed.
  EXPECT_TRUE(optimizedA->getPayload().isEqual(transposedA));
}

/// Check that the removing of transposes still happens when
/// predicates are involved.
TEST_F(GraphOptz, transposePrivateVariableWithPredicate) {
  auto *A =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 10, 20, 3}, "A", false);
  auto *pred = mod_.createPlaceholder(ElemKind::FloatTy, {1}, "pred", false);
  ctx_.allocate(A)->getHandle().randomize(-7.0, 12.0, mod_.getPRNG());
  Tensor transposedA;
  ctx_.get(A)->transpose(&transposedA, {0, 3, 1, 2});
  // Arguably, if the transpose doesn't happen because the predicate is false
  // the value of A should be unchanged. However, the semantic of our
  // predicate is that they can be ignored and the program would still
  // be correct, thus this optimization is still legal.
  Node *T = F_->createTranspose("transpose", A, NHWC2NCHW);
  T->setPredicate(pred);
  SaveNode *save = F_->createSave(ctx_, "ret", T);
  save->setPredicate(pred);
  EXPECT_EQ(F_->getNodes().size(), 2);

  ::glow::convertPlaceholdersToConstants(F_, ctx_, {});
  ::glow::optimize(F_, CompilationMode::Infer);
  ASSERT_EQ(F_->getNodes().size(), 1);
  EXPECT_EQ(&*F_->getNodes().begin(), save);
  // We should have kept the predicate on the save node.
  ASSERT_EQ(pred->getNumUsers(), 1);
  EXPECT_EQ(pred->getUsers().begin()->getUser(), save);
  Variable *optimizedA = llvm::dyn_cast<Variable>(save->getInput().getNode());
  ASSERT_NE(optimizedA, nullptr);
  // Check that A has been properly transposed.
  EXPECT_TRUE(optimizedA->getPayload().isEqual(transposedA));
}

TEST_F(GraphOptz, BatchNormAfterConvNotOptimizeForTrain) {
  Node *A =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 10, 20, 3}, "A", false);
  Node *CV = F_->createConv(ctx_, "conv", A, 16, 5, 1, 2, 1);
  Node *BN = F_->createBatchNormalization(ctx_, "batch", CV, 3, 0.0001, 0.9);
  F_->createSave(ctx_, "ret", BN);

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

  Node *CV = F_->createConv(ctx_, "conv", A, 16, 5, 1, 2, 1);
  Node *BN = F_->createBatchNormalization(ctx_, "batch", CV, 3, 0.0001, 0.9);
  SaveNode *convSave = F_->createSave(ctx_, "ret", CV);
  SaveNode *ret = F_->createSave(ctx_, "ret", BN);

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
  Node *BN = F_->createBatchNormalization(ctx_, "batch", T, 3, 0.0001, 0.9);
  SaveNode *O = F_->createSave(ctx_, "ret", BN);

  EXPECT_EQ(F_->getNodes().size(), 3);
  EXPECT_EQ(BN->dims(0), llvm::makeArrayRef(transposedDims));

  ::glow::optimize(F_, CompilationMode::Infer);

  // Expecting Transpose->Output rather than BN->Output.
  auto *transpose = llvm::dyn_cast<TransposeNode>(O->getInput());
  ASSERT_NE(transpose, nullptr);
  ASSERT_TRUE(llvm::isa<BatchNormalizationNode>(transpose->getInput()));
  auto *bn = llvm::cast<BatchNormalizationNode>(transpose->getInput());
  // Check that the dimensions of the input and output have been
  // updated to compensate the absence of transpose.
  EXPECT_EQ(bn->dims(0), llvm::makeArrayRef(origDims));
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
  Node *BN = F_->createBatchNormalization(ctx_, "batch", T, 3, 0.0001, 0.9);
  BN->setPredicate(pred2);
  SaveNode *O = F_->createSave(ctx_, "ret", BN);
  O->setPredicate(pred3);

  EXPECT_EQ(F_->getNodes().size(), 3);
  EXPECT_EQ(BN->dims(0), llvm::makeArrayRef(transposedDims));

  ::glow::optimize(F_, CompilationMode::Infer);

  EXPECT_EQ(O->getPredicate().getNode(), pred3);
  // Expecting Transpose->Output rather than BN->Output.
  auto *transpose = llvm::dyn_cast<TransposeNode>(O->getInput());
  ASSERT_NE(transpose, nullptr);
  EXPECT_EQ(transpose->getPredicate().getNode(), pred2);
  ASSERT_TRUE(llvm::isa<BatchNormalizationNode>(transpose->getInput()));
  auto *bn = llvm::cast<BatchNormalizationNode>(transpose->getInput());
  EXPECT_EQ(bn->getPredicate().getNode(), pred2);
  // Check that the dimensions of the input and output have been
  // updated to compensate the absence of transpose.
  EXPECT_EQ(bn->dims(0), llvm::makeArrayRef(origDims));
  EXPECT_EQ(bn->getInput().dims(), llvm::makeArrayRef(origDims));
  EXPECT_EQ(F_->getNodes().size(), 3);
}

TEST_F(GraphOptz, sinkTransposeBelowRELU) {
  const size_t origDims[] = {1, 5, 10, 15};
  const size_t transposedDims[] = {1, 15, 5, 10};
  Node *A = mod_.createPlaceholder(ElemKind::FloatTy, origDims, "input", false);
  Node *T = F_->createTranspose("transpose", A, NHWC2NCHW);
  Node *K = F_->createRELU("relu", T);
  SaveNode *O = F_->createSave(ctx_, "ret", K);

  EXPECT_EQ(F_->getNodes().size(), 3);
  EXPECT_EQ(K->dims(0), llvm::makeArrayRef(transposedDims));

  ::glow::optimize(F_, CompilationMode::Infer);

  // Expecting Transpose->Output rather than RELU->Output.
  auto *transpose = llvm::dyn_cast<TransposeNode>(O->getInput());
  ASSERT_NE(transpose, nullptr);
  ASSERT_TRUE(llvm::isa<ReluNode>(transpose->getInput()));
  auto *relu = llvm::cast<ReluNode>(transpose->getInput());
  // Check that the dimensions of the input and output have been
  // updated to compensate the absence of transpose.
  EXPECT_EQ(relu->dims(0), llvm::makeArrayRef(origDims));
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
  Node *K = F_->createRELU("relu", T);
  K->setPredicate(pred2);
  SaveNode *O = F_->createSave(ctx_, "ret", K);
  O->setPredicate(pred3);

  EXPECT_EQ(F_->getNodes().size(), 3);
  EXPECT_EQ(K->dims(0), llvm::makeArrayRef(transposedDims));

  ::glow::optimize(F_, CompilationMode::Infer);

  EXPECT_EQ(O->getPredicate().getNode(), pred3);
  // Expecting Transpose->Output rather than RELU->Output.
  auto *transpose = llvm::dyn_cast<TransposeNode>(O->getInput());
  ASSERT_NE(transpose, nullptr);
  EXPECT_EQ(transpose->getPredicate().getNode(), pred2);
  ASSERT_TRUE(llvm::isa<ReluNode>(transpose->getInput()));
  auto *relu = llvm::cast<ReluNode>(transpose->getInput());
  EXPECT_EQ(relu->getPredicate().getNode(), pred2);
  // Check that the dimensions of the input and output have been
  // updated to compensate the absence of transpose.
  EXPECT_EQ(relu->dims(0), llvm::makeArrayRef(origDims));
  EXPECT_EQ(relu->getInput().dims(), llvm::makeArrayRef(origDims));
  EXPECT_EQ(F_->getNodes().size(), 3);
}

TEST_F(GraphOptz, sinkTransposeBelowSigmoid) {
  const size_t origDims[] = {1, 5, 10, 15};
  const size_t transposedDims[] = {1, 15, 5, 10};
  Node *A = mod_.createPlaceholder(ElemKind::FloatTy, origDims, "input", false);
  Node *T = F_->createTranspose("transpose", A, NHWC2NCHW);
  Node *SI = F_->createSigmoid("sigmoid", T);
  SaveNode *O = F_->createSave(ctx_, "ret", SI);

  EXPECT_EQ(F_->getNodes().size(), 3);
  EXPECT_EQ(SI->dims(0), llvm::makeArrayRef(transposedDims));

  ::glow::optimize(F_, CompilationMode::Infer);

  // Expecting Transpose->Output rather than Sigmoid->Output.
  auto *transpose = llvm::dyn_cast<TransposeNode>(O->getInput());
  ASSERT_NE(transpose, nullptr);
  ASSERT_TRUE(llvm::isa<SigmoidNode>(transpose->getInput()));
  auto *si = llvm::cast<SigmoidNode>(transpose->getInput());
  // Check that the dimensions of the input and output have been
  // updated to compensate the absence of transpose.
  EXPECT_EQ(si->dims(0), llvm::makeArrayRef(origDims));
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
  Node *SI = F_->createSigmoid("sigmoid", T);
  SI->setPredicate(pred2);
  SaveNode *O = F_->createSave(ctx_, "ret", SI);
  O->setPredicate(pred3);

  EXPECT_EQ(F_->getNodes().size(), 3);
  EXPECT_EQ(SI->dims(0), llvm::makeArrayRef(transposedDims));

  ::glow::optimize(F_, CompilationMode::Infer);

  EXPECT_EQ(O->getPredicate().getNode(), pred3);
  // Expecting Transpose->Output rather than Sigmoid->Output.
  auto *transpose = llvm::dyn_cast<TransposeNode>(O->getInput());
  ASSERT_NE(transpose, nullptr);
  EXPECT_EQ(transpose->getPredicate().getNode(), pred2);
  ASSERT_TRUE(llvm::isa<SigmoidNode>(transpose->getInput()));
  auto *si = llvm::cast<SigmoidNode>(transpose->getInput());
  EXPECT_EQ(si->getPredicate().getNode(), pred2);
  // Check that the dimensions of the input and output have been
  // updated to compensate the absence of transpose.
  EXPECT_EQ(si->dims(0), llvm::makeArrayRef(origDims));
  EXPECT_EQ(si->getInput().dims(), llvm::makeArrayRef(origDims));
  EXPECT_EQ(F_->getNodes().size(), 3);
}

TEST_F(GraphOptz, sinkTransposeBelowTanh) {
  const size_t origDims[] = {1, 5, 10, 15};
  const size_t transposedDims[] = {1, 15, 5, 10};
  Node *A = mod_.createPlaceholder(ElemKind::FloatTy, origDims, "input", false);
  Node *T = F_->createTranspose("transpose", A, NHWC2NCHW);
  Node *TN = F_->createTanh("tanh", T);
  SaveNode *O = F_->createSave(ctx_, "ret", TN);

  EXPECT_EQ(F_->getNodes().size(), 3);
  EXPECT_EQ(TN->dims(0), llvm::makeArrayRef(transposedDims));

  ::glow::optimize(F_, CompilationMode::Infer);

  // Expecting Transpose->Output rather than Tanh->Output.
  auto *transpose = llvm::dyn_cast<TransposeNode>(O->getInput());
  ASSERT_NE(transpose, nullptr);
  ASSERT_TRUE(llvm::isa<TanhNode>(transpose->getInput()));
  auto *tn = llvm::cast<TanhNode>(transpose->getInput());
  // Check that the dimensions of the input and output have been
  // updated to compensate the absence of transpose.
  EXPECT_EQ(tn->dims(0), llvm::makeArrayRef(origDims));
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
  Node *TN = F_->createTanh("tanh", T);
  TN->setPredicate(pred2);
  SaveNode *O = F_->createSave(ctx_, "ret", TN);
  O->setPredicate(pred3);

  EXPECT_EQ(F_->getNodes().size(), 3);
  EXPECT_EQ(TN->dims(0), llvm::makeArrayRef(transposedDims));

  ::glow::optimize(F_, CompilationMode::Infer);

  EXPECT_EQ(O->getPredicate().getNode(), pred3);
  // Expecting Transpose->Output rather than Tanh->Output.
  auto *transpose = llvm::dyn_cast<TransposeNode>(O->getInput());
  ASSERT_NE(transpose, nullptr);
  EXPECT_EQ(transpose->getPredicate().getNode(), pred2);
  ASSERT_TRUE(llvm::isa<TanhNode>(transpose->getInput()));
  auto *tn = llvm::cast<TanhNode>(transpose->getInput());
  EXPECT_EQ(tn->getPredicate().getNode(), pred2);
  // Check that the dimensions of the input and output have been
  // updated to compensate the absence of transpose.
  EXPECT_EQ(tn->dims(0), llvm::makeArrayRef(origDims));
  EXPECT_EQ(tn->getInput().dims(), llvm::makeArrayRef(origDims));
  EXPECT_EQ(F_->getNodes().size(), 3);
}

TEST_F(GraphOptz, cancelTwoTransposes) {
  const size_t origDims[] = {1, 5, 10, 15};
  Node *A = mod_.createPlaceholder(ElemKind::FloatTy, origDims, "input", false);
  Node *T1 = F_->createTranspose("transpose", A, NCHW2NHWC);
  Node *T2 = F_->createTranspose("transpose", T1, NHWC2NCHW);
  ReluNode *K = F_->createRELU("relu", T2);
  SaveNode *save = F_->createSave(ctx_, "ret", K);

  EXPECT_EQ(K->getInput().dims(), llvm::makeArrayRef(origDims));
  EXPECT_EQ(F_->getNodes().size(), 4);

  ::glow::optimize(F_, CompilationMode::Infer);

  EXPECT_EQ(F_->getNodes().size(), 2);
  Node *relu = save->getInput();
  EXPECT_EQ(relu->dims(0), llvm::makeArrayRef(origDims));
  ASSERT_TRUE(llvm::isa<ReluNode>(relu));
  EXPECT_EQ(llvm::cast<ReluNode>(relu)->getInput().getNode(), A);
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
  SaveNode *save = F_->createSave(ctx_, "ret", K);
  save->setPredicate(pred4);

  EXPECT_EQ(K->getInput().dims(), llvm::makeArrayRef(origDims));
  EXPECT_EQ(F_->getNodes().size(), 4);

  ::glow::optimize(F_, CompilationMode::Infer);

  EXPECT_EQ(F_->getNodes().size(), 2);
  EXPECT_EQ(save->getPredicate().getNode(), pred4);
  Node *relu = save->getInput();
  EXPECT_EQ(relu->getPredicate().getNode(), pred3);
  EXPECT_EQ(relu->dims(0), llvm::makeArrayRef(origDims));
  ASSERT_TRUE(llvm::isa<ReluNode>(relu));
  EXPECT_EQ(llvm::cast<ReluNode>(relu)->getInput().getNode(), A);
}

TEST_F(GraphOptz, removeIdentityTranspose) {
  const size_t origDims[] = {1, 5, 10, 15};
  Node *A = mod_.createPlaceholder(ElemKind::FloatTy, origDims, "input", false);
  Node *T = F_->createTranspose("transpose", A, {0, 1, 2, 3});
  Node *K = F_->createRELU("relu", T);
  F_->createSave(ctx_, "ret", K);

  EXPECT_EQ(F_->getNodes().size(), 3);
  EXPECT_EQ(K->getNthInput(0).getNode(), T);

  ::glow::optimize(F_, CompilationMode::Infer);

  EXPECT_EQ(F_->getNodes().size(), 2);
  EXPECT_EQ(K->getNthInput(0).getNode(), A);
  // Make sure we didn't mess up with the dimensions of the
  // variable while eliminating the transpose.
  EXPECT_EQ(A->dims(0), llvm::makeArrayRef(origDims));
}

/// Check that the predicates don't get in the way of
/// the identity transpose removal, while still being
/// preserved.
TEST_F(GraphOptz, removeIdentityTransposeWithPredicate) {
  const size_t origDims[] = {1, 5, 10, 15};
  Node *A = mod_.createPlaceholder(ElemKind::FloatTy, origDims, "input", false);
  Node *pred1 = mod_.createPlaceholder(ElemKind::FloatTy, {1}, "pred", false);
  Node *pred2 = mod_.createPlaceholder(ElemKind::FloatTy, {1}, "pred", false);
  Node *pred3 = mod_.createPlaceholder(ElemKind::FloatTy, {1}, "pred", false);
  Node *T = F_->createTranspose("transpose", A, {0, 1, 2, 3});
  T->setPredicate(pred1);
  Node *K = F_->createRELU("relu", T);
  K->setPredicate(pred2);
  SaveNode *save = F_->createSave(ctx_, "ret", K);
  save->setPredicate(pred3);

  EXPECT_EQ(F_->getNodes().size(), 3);
  EXPECT_EQ(K->getNthInput(0).getNode(), T);

  ::glow::optimize(F_, CompilationMode::Infer);
  EXPECT_EQ(F_->getNodes().size(), 2);
  EXPECT_EQ(save->getPredicate().getNode(), pred3);
  EXPECT_EQ(save->getInput().getNode(), K);
  EXPECT_EQ(K->getNthInput(0).getNode(), A);
  EXPECT_EQ(K->getPredicate().getNode(), pred2);
  // Make sure we didn't mess up with the dimensions of the
  // variable while eliminating the transpose.
  EXPECT_EQ(A->dims(0), llvm::makeArrayRef(origDims));
}

TEST_F(GraphOptz, dontCancelTwoTransposesIfNotMatching) {
  const size_t origDims[] = {1, 5, 10, 15};
  const size_t afterFirstTransposeDims[] = {1, 10, 15, 5};
  const size_t afterSecondTransposeDims[] = {1, 15, 5, 10};
  Node *A = mod_.createPlaceholder(ElemKind::FloatTy, origDims, "input", false);
  TransposeNode *T1 = F_->createTranspose("transpose", A, NCHW2NHWC);
  TransposeNode *T2 = F_->createTranspose("transpose", T1, NCHW2NHWC);
  SaveNode *save = F_->createSave(ctx_, "ret", T2);

  EXPECT_EQ(F_->getNodes().size(), 3);

  ::glow::optimize(F_, CompilationMode::Infer);

  EXPECT_EQ(F_->getNodes().size(), 3);
  // Make sure the structure of the graph did not change.
  Node *secondTranspose = save->getInput();
  EXPECT_EQ(secondTranspose->dims(0),
            llvm::makeArrayRef(afterSecondTransposeDims));
  EXPECT_EQ(secondTranspose, T2);
  Node *firstTranspose = T2->getInput();
  EXPECT_EQ(firstTranspose, T1);
  EXPECT_EQ(T1->dims(0), llvm::makeArrayRef(afterFirstTransposeDims));
  EXPECT_EQ(T1->getInput().getNode(), A);
  EXPECT_EQ(A->dims(0), llvm::makeArrayRef(origDims));
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
  SaveNode *O = F_->createSave(ctx_, "ret", K);

  EXPECT_EQ(F_->getNodes().size(), 4);

  ::glow::optimize(F_, CompilationMode::Infer);

  // Expecting Transpose->Output rather than Add->Output.
  auto *transpose = llvm::dyn_cast<TransposeNode>(O->getInput());
  ASSERT_NE(transpose, nullptr);
  ASSERT_TRUE(llvm::isa<AddNode>(transpose->getInput()));
  auto *add = llvm::cast<AddNode>(transpose->getInput());
  // Check that the dimensions of the input and output have been
  // updated to compensate the absence of transpose.
  EXPECT_EQ(add->dims(0), llvm::makeArrayRef(origDims));
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
  SaveNode *O = F_->createSave(ctx_, "ret", K);
  O->setPredicate(pred4);

  EXPECT_EQ(F_->getNodes().size(), 4);

  ::glow::optimize(F_, CompilationMode::Infer);

  EXPECT_EQ(O->getPredicate().getNode(), pred4);
  // Expecting Transpose->Output rather than Add->Output.
  auto *transpose = llvm::dyn_cast<TransposeNode>(O->getInput());
  ASSERT_NE(transpose, nullptr);
  EXPECT_EQ(transpose->getPredicate().getNode(), pred3);
  ASSERT_TRUE(llvm::isa<AddNode>(transpose->getInput()));
  auto *add = llvm::cast<AddNode>(transpose->getInput());
  EXPECT_EQ(add->getPredicate().getNode(), pred3);
  // Check that the dimensions of the input and output have been
  // updated to compensate the absence of transpose.
  EXPECT_EQ(add->dims(0), llvm::makeArrayRef(origDims));
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
  SaveNode *O = F_->createSave(ctx_, "ret", CN);

  EXPECT_EQ(F_->getNodes().size(), 4);

  ::glow::optimize(F_, CompilationMode::Infer);

  // Expecting RELU->Output rather than Concat->Output.
  auto *relu = llvm::dyn_cast<ReluNode>(O->getInput());
  ASSERT_NE(relu, nullptr);
  ASSERT_TRUE(llvm::isa<ConcatNode>(relu->getInput()));
  auto *concat = llvm::cast<ConcatNode>(relu->getInput());
  // Check that the dimensions of the input and output have been
  // updated to compensate the absence of transpose.
  EXPECT_EQ(concat->dims(0), llvm::makeArrayRef(origDimsConcat));
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
  SaveNode *O = F_->createSave(ctx_, "ret", CN);
  O->setPredicate(pred4);

  EXPECT_EQ(F_->getNodes().size(), 4);

  ::glow::optimize(F_, CompilationMode::Infer);

  // Expecting RELU->Output rather than Concat->Output.
  EXPECT_EQ(O->getPredicate().getNode(), pred4);
  auto *relu = llvm::dyn_cast<ReluNode>(O->getInput());
  ASSERT_NE(relu, nullptr);
  EXPECT_EQ(relu->getPredicate().getNode(), pred3);
  ASSERT_TRUE(llvm::isa<ConcatNode>(relu->getInput()));
  auto *concat = llvm::cast<ConcatNode>(relu->getInput());
  EXPECT_EQ(concat->getPredicate().getNode(), pred3);
  // Check that the dimensions of the input and output have been
  // updated to compensate the absence of transpose.
  EXPECT_EQ(concat->dims(0), llvm::makeArrayRef(origDimsConcat));
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
  SaveNode *O = F_->createSave(ctx_, "ret", CN);

  EXPECT_EQ(F_->getNodes().size(), 4);

  ::glow::optimize(F_, CompilationMode::Infer);

  // Expecting Transpose->Output rather than Add->Output.
  auto *transpose = llvm::dyn_cast<TransposeNode>(O->getInput());
  ASSERT_NE(transpose, nullptr);
  ASSERT_TRUE(llvm::isa<ConcatNode>(transpose->getInput()));
  auto *concat = llvm::cast<ConcatNode>(transpose->getInput());
  // Check that the dimensions of the input and output have been
  // updated to compensate the absence of transpose.
  EXPECT_EQ(concat->dims(0), llvm::makeArrayRef(origDimsConcat));
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
  SaveNode *O = F_->createSave(ctx_, "ret", CN);
  O->setPredicate(pred4);

  EXPECT_EQ(F_->getNodes().size(), 4);

  ::glow::optimize(F_, CompilationMode::Infer);

  EXPECT_EQ(O->getPredicate().getNode(), pred4);
  // Expecting Transpose->Output rather than Add->Output.
  auto *transpose = llvm::dyn_cast<TransposeNode>(O->getInput());
  ASSERT_NE(transpose, nullptr);
  EXPECT_EQ(transpose->getPredicate().getNode(), pred3);
  ASSERT_TRUE(llvm::isa<ConcatNode>(transpose->getInput()));
  auto *concat = llvm::cast<ConcatNode>(transpose->getInput());
  EXPECT_EQ(concat->getPredicate().getNode(), pred3);
  // Check that the dimensions of the input and output have been
  // updated to compensate the absence of transpose.
  EXPECT_EQ(concat->dims(0), llvm::makeArrayRef(origDimsConcat));
  EXPECT_EQ(concat->getInputs()[0].dims(), llvm::makeArrayRef(origDims));
  EXPECT_EQ(concat->getInputs()[1].dims(), llvm::makeArrayRef(origDims));
  EXPECT_EQ(concat->getInputs()[0].getNode(), A1);
  EXPECT_EQ(concat->getInputs()[1].getNode(), A2);

  EXPECT_EQ(F_->getNodes().size(), 3);
}

TEST_F(GraphOptz, poolBelowReluSwapped) {
  Node *A =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 5, 10, 15}, "input", false);
  Node *R = F_->createRELU("relu", A);
  Node *PL = F_->createMaxPool("pool", R, 1, 10, 20);
  Node *O = F_->createSave(ctx_, "ret", PL);

  EXPECT_EQ(F_->getNodes().size(), 3);

  ::glow::optimize(F_, CompilationMode::Infer);

  // Expecting RELU->Output rather than Pool->Output.
  EXPECT_TRUE(llvm::isa<SaveNode>(O));
  EXPECT_TRUE(llvm::isa<ReluNode>(llvm::dyn_cast<SaveNode>(O)->getInput()));

  EXPECT_EQ(F_->getNodes().size(), 3);
}

TEST_F(GraphOptz, poolBelowReluNotSwappedIfModeNotMax) {
  Node *A =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 5, 10, 15}, "input", false);
  Node *R = F_->createRELU("relu", A);
  Node *PL = F_->createAvgPool("pool", R, 1, 10, 20);
  Node *O = F_->createSave(ctx_, "ret", PL);

  EXPECT_EQ(F_->getNodes().size(), 3);

  ::glow::optimize(F_, CompilationMode::Infer);

  // Expecting Pool->Output (no swap).
  EXPECT_TRUE(llvm::isa<SaveNode>(O));
  EXPECT_TRUE(llvm::isa<AvgPoolNode>(llvm::dyn_cast<SaveNode>(O)->getInput()));

  EXPECT_EQ(F_->getNodes().size(), 3);
}

TEST_F(GraphOptz, poolBelowReluNotSwappedIfNotSingleUse) {
  Node *A =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 5, 10, 15}, "input", false);
  Node *R = F_->createRELU("relu", A);
  Node *PL = F_->createMaxPool("pool", R, 1, 10, 20);
  Node *O = F_->createSave(ctx_, "ret", PL);
  F_->createSave(ctx_, "ret", R);

  EXPECT_EQ(F_->getNodes().size(), 4);

  ::glow::optimize(F_, CompilationMode::Infer);

  // Expecting Pool->Output (no swap).
  EXPECT_TRUE(llvm::isa<SaveNode>(O));
  EXPECT_TRUE(llvm::isa<MaxPoolNode>(llvm::dyn_cast<SaveNode>(O)->getInput()));

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
  Node *O = F_->createSave(ctx_, "ret", CN4);

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
  Node *O = F_->createSave(ctx_, "ret", CN3);

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
  Node *O = F_->createSave(ctx_, "ret", S);

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

  SaveNode *O = F_->createSave(ctx_, "ret", sub);

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

  SaveNode *O = F_->createSave(ctx_, "ret", mul3);

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
  SaveNode *SN = F_->createSave(ctx_, "ret", mul);

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
    Context ctx;

    Type t(ElemKind::FloatTy, {1000, 1000, 1000});
    Node *Z = F->createSplat("zero", &t, 0.);
    Node *S1 = F->createSlice("slice1", Z, {5, 15, 42}, {99, 88, 77});
    Node *S2 = F->createSlice("slice2", S1, {1, 1, 1}, {2, 3, 4});
    F->createSave(ctx, "ret", S2);

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
  auto *O = F_->createSave(ctx_, "ret", R);

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
  auto *O = F_->createSave(ctx_, "ret", A3);

  // Before optimization, we have 9 nodes in the graph.
  EXPECT_EQ(F_->getNodes().size(), 9);

  ::glow::optimize(F_, CompilationMode::Infer);

  // After optimization, we expect to see only 8 nodes, as Z2,R2 would be
  // replace by a new splat node.
  EXPECT_EQ(F_->getNodes().size(), 8);

  // The second input of A3 shoule be a splat node with a shape of R3.
  auto *SN = llvm::dyn_cast<SplatNode>(
      llvm::dyn_cast<SaveNode>(O)->getInput().getNode()->getNthInput(1));
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
  auto *O = F_->createSave(ctx_, "ret", R2);

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
  *ctx_.allocate(input) = {10, 10, 10, 10};
  auto qType = mod_.uniqueType(ElemKind::Int8QTy, {4}, 2, 0);

  auto Q = F_->createQuantize("quantize", input, qType);
  auto S = F_->createSave(ctx_, "save", Q);

  EXPECT_EQ(2, F_->getNodes().size());
  ::glow::convertPlaceholdersToConstants(F_, ctx_, {S->getPlaceholder()});
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
  auto *input = mod_.createPlaceholder(ElemKind::FloatTy, {4}, "input", true);
  *ctx_.allocate(input) = {10, 10, 10, 10};
  auto qType = mod_.uniqueType(ElemKind::Int8QTy, {4}, 2, 0);

  auto Q = F_->createQuantize("quantize", input, qType);
  F_->createSave(ctx_, "save", Q);
  auto clonedF = F_->clone("cloned");

  EXPECT_EQ(2, clonedF->getNodes().size());
  ::glow::convertPlaceholdersToConstants(clonedF, ctx_, {});
  ::glow::optimize(clonedF, CompilationMode::Infer);
  // F_ function should not be affected.
  EXPECT_EQ(2, F_->getNodes().size());

  // Check original var.
  for (unsigned i = 0; i < 4; ++i) {
    EXPECT_EQ(10, ctx_.get(input)->getHandle().raw(i));
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
  auto *input = mod_.createPlaceholder(ElemKind::Int8QTy, {4, 10}, 0.5, 11,
                                       "input", true);

  auto *D = F_->createDequantize("dequantize", input);

  auto qType = mod_.uniqueType(ElemKind::Int8QTy, {4, 10}, 0.03, 5);
  auto *Q = F_->createQuantize("quantize", D, qType);

  F_->createSave(ctx_, "ret", Q);

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
  F_->createSave(ctx_, "save", max);
  EXPECT_EQ(F_->getNodes().size(), 3);

  ::glow::optimize(F_, CompilationMode::Infer);
  // Splat and Max should be gone.
  EXPECT_EQ(F_->getNodes().size(), 1);
}

TEST_F(GraphOptz, FuseRescaleIntoArithmetic) {
  // This test ensures the fact that fusing of rescale is done.
  auto opOutTy = mod_.uniqueType(ElemKind::Int8QTy, {10}, 1, 0);
  auto rescaleOutTy = mod_.uniqueType(ElemKind::Int8QTy, {10}, 2, 1);

  Node *LHS =
      mod_.createPlaceholder(ElemKind::Int8QTy, {10}, 0.4, 0, "LHS", true);
  Node *RHS =
      mod_.createPlaceholder(ElemKind::Int8QTy, {10}, 0.3, 0, "RHS", true);

  Node *add = F_->createAdd("qAdd", opOutTy, LHS, RHS);
  add = F_->createRescaleQuantized("rsAdd", add, rescaleOutTy);
  add = F_->createSave(ctx_, "saveAdd", add);

  Node *sub = F_->createSub("qSub", opOutTy, LHS, RHS);
  sub = F_->createRescaleQuantized("rsSub", sub, rescaleOutTy);
  sub = F_->createSave(ctx_, "saveSub", sub);

  Node *div = F_->createDiv("qDiv", opOutTy, LHS, RHS);
  div = F_->createRescaleQuantized("rsDiv", div, rescaleOutTy);
  div = F_->createSave(ctx_, "saveDiv", div);

  Node *mul = F_->createMul("qMul", opOutTy, LHS, RHS);
  mul = F_->createRescaleQuantized("rsMul", mul, rescaleOutTy);
  mul = F_->createSave(ctx_, "saveMul", mul);

  Node *min = F_->createMin("qMin", opOutTy, LHS, RHS);
  min = F_->createRescaleQuantized("rsMin", min, rescaleOutTy);
  min = F_->createSave(ctx_, "saveMin", min);

  Node *max = F_->createMax("qMax", opOutTy, LHS, RHS);
  max = F_->createRescaleQuantized("rsMax", max, rescaleOutTy);
  max = F_->createSave(ctx_, "saveMax", max);

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
  F_->createSave(ctx_, "save", rCV);

  // All rescales must be fused into convolution.
  EXPECT_EQ(F_->getNodes().size(), 6);
  ::glow::optimize(F_, CompilationMode::Infer);
  EXPECT_EQ(F_->getNodes().size(), 2);
}

TEST_F(GraphOptz, sinkRescaledQuantizedNode) {
  // Check that we eliminate rescale nodes by sinking them into other operators.
  auto *input = mod_.createPlaceholder(ElemKind::Int8QTy, {4, 10}, 0.5, 11,
                                       "input", true);

  // slice -> rescale -> reshape -> rescale -> transpose -> maxpool -> save.
  auto *slice = F_->createSlice("slice", input, {0, 0}, {3, 3});
  auto *rescale = F_->createRescaleQuantized(
      "rescale", slice, mod_.uniqueType(ElemKind::Int8QTy, {3, 3}, 0.4, 10));
  auto *reshape = F_->createReshape("reshape", rescale, {1, 1, 3, 3});
  auto *rescale2 = F_->createRescaleQuantized(
      "rescale", reshape,
      mod_.uniqueType(ElemKind::Int8QTy, {1, 1, 3, 3}, 0.3, 9));
  auto *transpose = F_->createTranspose("transpose", rescale2, {0, 2, 3, 1});
  auto *maxpool =
      F_->createMaxPool("maxpool", transpose, {3, 3}, {1, 1}, {0, 0, 0, 0});
  auto *save = F_->createSave(ctx_, "ret", maxpool);

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
  F_->createSave(ctx_, "save", div);

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
  F_->createSave(ctx_, "save", relu);

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
  F_->createSave(ctx_, "save", cc);

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
  F_->createSave(ctx_, "save", cc);

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
  F_->createSave(ctx_, "save", cc);

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
  auto *O = F_->createSave(ctx_, "ret", addNode);

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
  auto *newRN =
      llvm::dyn_cast<ReshapeNode>(O->getInput().getNode()->getNthInput(0));
  ASSERT_TRUE(newRN);
  EXPECT_TRUE(newRN->getResult().getType()->dims().equals(outputShape));

  // The input of newRN should be a ConcatNode now.
  auto *newCN = llvm::dyn_cast<ConcatNode>(
      O->getInput().getNode()->getNthInput(0).getNode()->getNthInput(0));
  ASSERT_TRUE(newCN);
}

/// Check that Variable CSE works correctly, combining small Variables that have
/// the same data.
TEST_F(GraphOptz, VarsCSE) {
  // Create three variables that are Private, are not trainable, and have no
  // writers. The first two variables have the same data, and so should be
  // combined via variable CSE. The third variable differs by the last value,
  // and so should not be combined.
  auto *input1 = mod_.createVariable(ElemKind::FloatTy, {10}, "input1",
                                     VisibilityKind::Private, false);
  auto *input2 = mod_.createVariable(ElemKind::FloatTy, {10}, "input2",
                                     VisibilityKind::Private, false);
  auto *input3 = mod_.createVariable(ElemKind::FloatTy, {10}, "input3",
                                     VisibilityKind::Private, false);
  input1->getHandle() = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  input2->getHandle() = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  input3->getHandle() = {0, 1, 2, 3, 4, 5, 6, 7, 8, -1};

  // Input them each to different nodes, so node CSE does not change them.
  auto *TN = F_->createTanh("tanh", input1);
  auto *SN = F_->createSigmoid("sigmoid", input2);
  auto *RN = F_->createRELU("relu", input3);
  auto *CN = F_->createConcat("concat", {TN, SN, RN}, /* axis */ 0);
  F_->createSave(ctx_, "ret", CN);

  // Initially there are three variables: inputs 1, 2, and 3 (the save uses a
  // placeholder).
  EXPECT_EQ(mod_.getVars().size(), 3);

  ::glow::optimize(F_, CompilationMode::Infer);

  // Now only two variables are left; input1 and input2 have been combined,
  // but input3 has not.
  EXPECT_EQ(mod_.getVars().size(), 2);

  // Verify that only one of input1 and input2 exists, and that input3 still
  // exists.
  Variable *varOneOrTwo = nullptr;
  bool foundVarThree = false;
  for (auto *V : mod_.getVars()) {
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

  // The splat is a constant input to add1 and add2, and is their LHS input. We
  // expect canonicalization to occur during optimization, moving the splat
  // to the RHS for both. Note that add1 has multiple users: add2 and save1.
  Node *AN1 = F_->createAdd("add1", SN, I1);
  Node *AN2 = F_->createAdd("add2", SN, AN1);
  SaveNode *SN1 = F_->createSave(ctx_, "save1", AN1);
  SaveNode *SN2 = F_->createSave(ctx_, "save2", AN2);

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
  SaveNode *SN = F_->createSave(ctx_, "ret", CN);

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
TEST_F(GraphOptz, ReshapePrivateVarOneUse) {
  const size_t shape[] = {10, 20};
  const size_t reshape1[] = {200, 1};
  const size_t reshape2[] = {200};
  auto *input = F_->getParent()->createPlaceholder(ElemKind::FloatTy, shape,
                                                   "input", true);
  ctx_.allocate(input);
  auto *R1 = F_->createReshape("reshape1", input, reshape1);
  auto *R2 = F_->createReshape("reshape2", R1, reshape2);
  auto *O = F_->createSave(ctx_, "ret", R2);

  // Before optimization, we have 2 Reshapes and a Save.
  EXPECT_EQ(F_->getNodes().size(), 3);

  ::glow::convertPlaceholdersToConstants(F_, ctx_, {});
  ::glow::optimize(F_, CompilationMode::Infer);

  // After optimization, we expect to see just a Save.
  EXPECT_EQ(F_->getNodes().size(), 1);

  // Save should have the new Variable as input.
  auto *V = llvm::dyn_cast<Variable>(O->getInput());
  ASSERT_TRUE(V);
  // The new Variable should have the same shape as the original second Reshape.
  EXPECT_TRUE(V->getType()->dims().equals(reshape2));
}

/// Test that transpose is merged into matmul.
TEST_F(GraphOptz, mergeTransposeIntoMatMul) {
  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 1, 2, 3}, "input", false);
  auto *weights = F_->getParent()->createVariable(
      ElemKind::FloatTy, {6, 1}, "weights", VisibilityKind::Private);

  weights->getHandle() = {0, 1, 2, 3, 4, 5};
  float newWeightsRef[] = {0, 2, 4, 1, 3, 5};

  auto *TN = F_->createTranspose("transpose", input, NHWC2NCHW);
  auto *RN = F_->createReshape("reshape", TN, {1, 6});
  auto *MMN = F_->createMatMul("matmul", RN, weights);
  auto *SN = F_->createSave(ctx_, "ret", MMN);

  // Transpose + Reshape + MatMul + Save.
  EXPECT_EQ(F_->getNodes().size(), 4);

  ::glow::optimize(F_, CompilationMode::Infer);

  // Reshape + MatMul + Save.
  EXPECT_EQ(F_->getNodes().size(), 3);

  // Check reordered weights.
  auto *newMMN = llvm::dyn_cast<MatMulNode>(SN->getInput());
  ASSERT_TRUE(newMMN != nullptr);
  auto *newW = llvm::dyn_cast<Variable>(newMMN->getRHS());
  ASSERT_TRUE(newW != nullptr);
  for (unsigned i = 0; i < 6; ++i) {
    EXPECT_EQ(newWeightsRef[i], newW->getHandle().raw(i));
  }
}

TEST_F(GraphOptz, ConvertPlaceholdersToConstants) {
  auto *input1 = mod_.createPlaceholder(ElemKind::FloatTy, {1}, "input1", true);
  auto *input2 = mod_.createPlaceholder(ElemKind::FloatTy, {1}, "input2", true);
  auto *input3 = mod_.createPlaceholder(ElemKind::FloatTy, {1}, "input3", true);
  auto *save1 = F_->createSave(ctx_, "save1", input1);
  auto *save2 = F_->createSave(ctx_, "save2", input2);
  auto *save3 = F_->createSave(ctx_, "save3", input3);

  // No variables, six PHs (3 inputs, 3 saves).
  EXPECT_EQ(mod_.getVars().size(), 0);
  EXPECT_EQ(mod_.getPlaceholders().size(), 6);

  // Allocate two of the three inputs, but mark input2 of them as non-constant.
  ctx_.allocate(input1);
  ctx_.allocate(input2);
  // Don't allocate input3; keep it as a placeholder instead.
  ::glow::convertPlaceholdersToConstants(F_, ctx_, {input2});

  // input1 becomes a variable.
  EXPECT_EQ(mod_.getVars().size(), 1);
  EXPECT_EQ(mod_.getPlaceholders().size(), 6);

  EXPECT_TRUE(llvm::isa<Variable>(save1->getInput()));
  EXPECT_TRUE(llvm::isa<Placeholder>(save2->getInput()));
  EXPECT_TRUE(llvm::isa<Placeholder>(save3->getInput()));
}
