/**
 * Copyright (c) Glow Contributors. See CONTRIBUTORS file.
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

#include "glow/Base/Type.h"
#include "glow/Graph/Graph.h"
#include "glow/Graph/Node.h"
#include "glow/Graph/Nodes.h"
#include "glow/Graph/PlaceholderBindings.h"
#include "glow/IR/IR.h"
#include "glow/Optimizer/GraphOptimizer/FunctionPassPipeline.h"
#include "glow/Optimizer/GraphOptimizer/GraphOptimizer.h"
#include "glow/Optimizer/Lower/Lower.h"

#include "gtest/gtest.h"

using namespace glow;

class GraphFold : public GraphOptz {};

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
  const Node *nodeAddress_;
  IsSameNodeAddress(const Node *nodeAddress) : nodeAddress_(nodeAddress) {}
  bool operator()(const Node &n) const { return &n == nodeAddress_; }
};

/// \returns true if the Function \p F contains the Node \p N.
static bool functionContainsNode(const Function *F, const Node *N) {
  return std::find_if(F->getNodes().begin(), F->getNodes().end(),
                      IsSameNodeAddress(N)) != F->getNodes().end();
}

/// Optimize the function \p F with \p cctx. \returns the optimized function. If
/// \p pass is empty then the whole default optimization pipeline is run.
/// Otherwise only \p pipeline is used.
static Function *
optimizeFunctionForTest(Function *F,
                        std::initializer_list<FunctionPassConfig> configs = {},
                        const CompilationContext &cctx = CompilationContext()) {
  auto *G = F->clone(F->getName().str() + "_optimized");
  if (configs.size() == 0) {
    ::glow::optimize(G, cctx);
    return G;
  }
  FunctionPassManager FPM("TestFPM", configs);
  FPM.run(G, cctx);
  return G;
}

/// \returns the first node in a function which has the specificied name.
template <typename NodeT = Node>
static const NodeT *findFunctionNodeByName(const Function *F,
                                           const llvm::StringRef name) {
  return llvm::dyn_cast<NodeT>(
      std::find_if(F->getNodes().begin(), F->getNodes().end(),
                   [=](auto &N) { return N.getName() == name; }));
}

TEST_F(GraphOptz, OptimizeClipFunnel) {
  auto *A =
      mod_.createPlaceholder(ElemKind::FloatTy, {100, 16}, "input", false);
  Node *K = A;
  float min = 0.0;
  float max = 1000.0;
  for (int i = 0; i < 10; ++i) {
    min += 1.0;
    max -= 1.0;
    K = F_->createClip("clip", K, min, max);
  }
  F_->createSave("ret", K);

  EXPECT_EQ(F_->getNodes().size(), 11);

  optimizedF_ = optimizeFunctionForTest(F_);
  EXPECT_EQ(optimizedF_->getNodes().size(), 2);

  // Find clip node in the optimized graph.
  Node *newClip = A;
  for (auto &N : optimizedF_->getNodes()) {
    if (N.getKind() == Kinded::Kind::ClipNodeKind) {
      newClip = llvm::dyn_cast<ClipNode>(&N);
    }
  }
  EXPECT_TRUE(llvm::isa<ClipNode>(newClip));
  ClipNode *c = llvm::dyn_cast<ClipNode>(newClip);
  EXPECT_EQ(min, c->getMin());
  EXPECT_EQ(max, c->getMax());

  bindings_.allocate(mod_.getPlaceholders());
  bindings_.get(A)->getHandle().randomize(-1000, 1000, mod_.getPRNG());
  bindings_.get(A)->getHandle().raw(0) = -1000;
  checkNumericalEquivalence();
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

/// Skip Reshape sinking below BatchNorm when inapplicable.
TEST_F(GraphOptz, SkipReshapeSinkBatchNorm) {
  auto *A = mod_.createPlaceholder(ElemKind::FloatTy, {32, 64}, "A", false);
  Node *RS = F_->createReshape("reshape", A, {32, 64, 1});
  Node *BN =
      F_->createBatchNormalization(bindings_, "batch", RS, 1, 0.0001, 0.9);
  F_->createSave("ret", BN);

  optimizedF_ = optimizeFunctionForTest(F_);
  EXPECT_EQ(F_->toString(/* skipUsersForStorage */ false, /* skipName */ true),
            optimizedF_->toString(/* skipUsersForStorage */ false,
                                  /* skipName */ true));
}

// Conv->Reshape->BatchNorm is optimized to Conv->Reshape after sinking Reshape
// below BatchNorm. Reshape transforms [N][H][W][C] to [N][W][H][C].
TEST_F(GraphOptz, optimizeBatchNormAfterConvAndReshapeNHWC) {
  auto *A =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 10, 20, 3}, "A", false);
  Node *CV = F_->createConv(bindings_, "conv", A, 16, 5, 1, 2, 1);
  Node *RS = F_->createReshape("reshape", CV, {1, 20, 10, 16});
  Node *BN =
      F_->createBatchNormalization(bindings_, "batch", RS, 3, 0.0001, 0.9);
  F_->createSave("ret", BN);

  EXPECT_EQ(F_->getNodes().size(), 4);
  ::glow::convertPlaceholdersToConstants(F_, bindings_, {});
  optimizedF_ = optimizeFunctionForTest(F_);
  EXPECT_EQ(optimizedF_->getNodes().size(), 3);

  ASSERT_EQ(A->getNumUsers(), 2);
  Node *newCV = std::find_if_not(A->getUsers().begin(), A->getUsers().end(),
                                 [CV](auto &it) { return it.getUser() == CV; })
                    ->getUser();
  EXPECT_TRUE(llvm::isa<ConvolutionNode>(newCV));
  ASSERT_EQ(newCV->getNumUsers(), 1);
  Node *reshape = newCV->getUsers().begin()->getUser();
  EXPECT_TRUE(llvm::isa<ReshapeNode>(reshape));

  bindings_.allocate(mod_.getPlaceholders());
  bindings_.get(A)->getHandle().randomize(-1.0, 1.0, mod_.getPRNG());
  checkNumericalEquivalence();
}

// Conv->Reshape->BatchNorm is optimized to Conv->Reshape after sinking Reshape
// below BatchNorm. Reshape flattens [N][H][W][C] to [N][HxW][C].
TEST_F(GraphOptz, optimizeBatchNormAfterConvAndReshapeNHWC2) {
  auto *A =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 10, 20, 3}, "A", false);
  Node *CV = F_->createConv(bindings_, "conv", A, 16, 5, 1, 2, 1);
  Node *RS = F_->createReshape("reshape", CV, {1, 200, 16});
  Node *BN =
      F_->createBatchNormalization(bindings_, "batch", RS, 2, 0.0001, 0.9);
  F_->createSave("ret", BN);

  EXPECT_EQ(F_->getNodes().size(), 4);
  ::glow::convertPlaceholdersToConstants(F_, bindings_, {});
  optimizedF_ = optimizeFunctionForTest(F_);
  EXPECT_EQ(optimizedF_->getNodes().size(), 3);

  ASSERT_EQ(A->getNumUsers(), 2);
  Node *newCV = std::find_if_not(A->getUsers().begin(), A->getUsers().end(),
                                 [CV](auto &it) { return it.getUser() == CV; })
                    ->getUser();
  EXPECT_TRUE(llvm::isa<ConvolutionNode>(newCV));
  ASSERT_EQ(newCV->getNumUsers(), 1);
  Node *reshape = newCV->getUsers().begin()->getUser();
  EXPECT_TRUE(llvm::isa<ReshapeNode>(reshape));

  bindings_.allocate(mod_.getPlaceholders());
  bindings_.get(A)->getHandle().randomize(-1.0, 1.0, mod_.getPRNG());
  checkNumericalEquivalence();
}

// BatchNorm is not folded into Conv. Reshape changes Channel Index dimensions
// and it prevents optimization. Reshape transforms [N][H][W][C] to
// [N][H][W/2][C*2].
TEST_F(GraphOptz, optimizeBatchNormAfterConvAndReshapeNHWCneg) {
  auto *A =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 10, 20, 3}, "A", false);
  Node *CV = F_->createConv(bindings_, "conv", A, 16, 5, 1, 2, 1);
  Node *RS = F_->createReshape("reshape", CV, {1, 10, 10, 32});
  Node *BN =
      F_->createBatchNormalization(bindings_, "batch", RS, 3, 0.0001, 0.9);
  F_->createSave("ret", BN);

  EXPECT_EQ(F_->getNodes().size(), 4);
  ::glow::convertPlaceholdersToConstants(F_, bindings_, {});
  optimizedF_ = optimizeFunctionForTest(F_);
  EXPECT_EQ(optimizedF_->getNodes().size(), 4);

  ASSERT_EQ(A->getNumUsers(), 2);
  Node *newCV = std::find_if_not(A->getUsers().begin(), A->getUsers().end(),
                                 [CV](auto &it) { return it.getUser() == CV; })
                    ->getUser();
  EXPECT_TRUE(llvm::isa<ConvolutionNode>(newCV));
  ASSERT_EQ(newCV->getNumUsers(), 1);
  Node *reshape = newCV->getUsers().begin()->getUser();
  EXPECT_TRUE(llvm::isa<ReshapeNode>(reshape));
  Node *bn = reshape->getUsers().begin()->getUser();
  EXPECT_TRUE(llvm::isa<BatchNormalizationNode>(bn));

  bindings_.allocate(mod_.getPlaceholders());
  bindings_.get(A)->getHandle().randomize(-1.0, 1.0, mod_.getPRNG());
  checkNumericalEquivalence();
}

// Sink Reshape below BatchNorm: multi-user testcase.
TEST_F(GraphOptz, sinkReshapeBelowBatchNormMultiUser) {
  auto *in =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 10, 40, 8}, "input", false);
  auto *RS = F_->createReshape("reshape", in, {1, 20, 20, 8});
  auto *BN =
      F_->createBatchNormalization(bindings_, "batch", RS, 3, 0.0001, 0.9);
  auto *save = F_->createSave("ret", BN);
  F_->createSave("extra_user", RS);

  optimizedF_ = optimizeFunctionForTest(F_);
  EXPECT_EQ(F_->getNodes().size(), 4);
  EXPECT_EQ(optimizedF_->getNodes().size(), 5);

  auto *saveOpt =
      findFunctionNodeByName<SaveNode>(optimizedF_, save->getName());
  auto *reshapeOpt = llvm::dyn_cast<ReshapeNode>(saveOpt->getInput());
  ASSERT_TRUE(reshapeOpt);
  ASSERT_TRUE(llvm::isa<BatchNormalizationNode>(reshapeOpt->getInput()));

  bindings_.allocate(mod_.getPlaceholders());
  bindings_.get(in)->getHandle().randomize(-1.0, 1.0, mod_.getPRNG());
  checkNumericalEquivalence();
}

// Sink Reshape below BatchNorm: quantized testcase.
TEST_F(GraphOptz, sinkReshapeBelowBatchNormQuantized) {
  auto *in = mod_.createPlaceholder(ElemKind::Int8QTy, {1, 10, 40, 3}, 1.5f, 0,
                                    "input", false);
  auto *params =
      mod_.createPlaceholder(ElemKind::Int8QTy, {3}, 1.0f, 0, "params", false);
  auto *RS = F_->createReshape("reshape", in, {1, 20, 20, 3});
  auto *bnOutTy = mod_.uniqueType(ElemKind::Int8QTy, {1, 20, 20, 3}, 2.7f, 0);
  auto *BN = F_->createBatchNormalization("batch", bnOutTy, RS, params, params,
                                          params, params, 3);
  auto *save = F_->createSave("ret", BN);

  optimizedF_ = optimizeFunctionForTest(F_);
  EXPECT_EQ(F_->getNodes().size(), 3);
  EXPECT_EQ(optimizedF_->getNodes().size(), 3);

  auto *saveOpt =
      findFunctionNodeByName<SaveNode>(optimizedF_, save->getName());
  auto *reshapeOpt = llvm::dyn_cast<ReshapeNode>(saveOpt->getInput());
  ASSERT_TRUE(reshapeOpt);
  auto *bnOpt = llvm::dyn_cast<BatchNormalizationNode>(reshapeOpt->getInput());
  ASSERT_TRUE(bnOpt);
  EXPECT_TRUE(
      BN->getInput().getType()->isEqual(*bnOpt->getInput().getType(), true));
  EXPECT_TRUE(
      BN->getResult().getType()->isEqual(*bnOpt->getResult().getType(), true));
}

// Conv->Reshape->BatchNorm. Sink Reshape below BatchNorm. Check that BatchNorm
// does not fold in to Conv.
TEST_F(GraphOptz, sinkReshapeBelowBatchNormAndDoNotFuseConvBatchNorm) {
  // Skip this test for now since Glow doesn't fully support
  // Convolution of NCHW layout
  GTEST_SKIP();

  auto *A =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 3, 10, 20}, "A", false);
  Node *CV = F_->createConv(bindings_, "conv", A, /* outChannels */ 16,
                            /* kernel */ 5, /* stride */ 1, /* pad */ 2,
                            /* group */ 1, /* dilation */ {1, 1},
                            /* layout */ ConvolutionLayout::NCHW);
  Node *RS = F_->createReshape("reshape", CV, {1, 10, 16, 20});
  Node *BN =
      F_->createBatchNormalization(bindings_, "batch", RS, 1, 0.0001, 0.9);
  F_->createSave("ret", BN);

  EXPECT_EQ(F_->getNodes().size(), 4);
  ::glow::convertPlaceholdersToConstants(F_, bindings_, {});
  optimizedF_ = optimizeFunctionForTest(F_);
  EXPECT_EQ(optimizedF_->getNodes().size(), 4);

  ASSERT_EQ(A->getNumUsers(), 2);
  Node *newCV = std::find_if_not(A->getUsers().begin(), A->getUsers().end(),
                                 [CV](auto &it) { return it.getUser() == CV; })
                    ->getUser();

  EXPECT_TRUE(llvm::isa<ConvolutionNode>(newCV));
  ASSERT_EQ(newCV->getNumUsers(), 1);
  Node *bn = newCV->getUsers().begin()->getUser();
  EXPECT_TRUE(llvm::isa<BatchNormalizationNode>(bn));
  Node *reshape = bn->getUsers().begin()->getUser();
  EXPECT_TRUE(llvm::isa<ReshapeNode>(reshape));

  bindings_.allocate(mod_.getPlaceholders());
  bindings_.get(A)->getHandle().randomize(-1.0, 1.0, mod_.getPRNG());
  checkNumericalEquivalence();
}

TEST_F(GraphOptz, optimizeBatchNormAfterConv) {
  auto *A =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 10, 20, 3}, "A", false);
  Node *CV = F_->createConv(bindings_, "conv", A, 16, 5, 1, 2, 1);
  Node *BN =
      F_->createBatchNormalization(bindings_, "batch", CV, 3, 0.0001, 0.9);
  F_->createSave("ret", BN);

  EXPECT_EQ(F_->getNodes().size(), 3);
  ::glow::convertPlaceholdersToConstants(F_, bindings_, {});
  optimizedF_ = optimizeFunctionForTest(F_);
  EXPECT_EQ(optimizedF_->getNodes().size(), 2);

  ASSERT_EQ(A->getNumUsers(), 2);
  Node *newCV = std::find_if_not(A->getUsers().begin(), A->getUsers().end(),
                                 [CV](auto &it) { return it.getUser() == CV; })
                    ->getUser();
  EXPECT_TRUE(llvm::isa<ConvolutionNode>(newCV));
  ASSERT_EQ(newCV->getNumUsers(), 1);
  Node *save = newCV->getUsers().begin()->getUser();
  EXPECT_TRUE(llvm::isa<SaveNode>(save));

  bindings_.allocate(mod_.getPlaceholders());
  bindings_.get(A)->getHandle().randomize(-1.0, 1.0, mod_.getPRNG());
  checkNumericalEquivalence();
}

void optimizeRedundantBatchNormTest(
    glow::Module &mod_, glow::Function *F_, glow::Function *&optimizedF_,
    glow::PlaceholderBindings &bindings_, llvm::ArrayRef<float> varV,
    llvm::ArrayRef<float> meanV, llvm::ArrayRef<float> gammaV,
    llvm::ArrayRef<float> betaV, const float eps) {
  auto *A =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 10, 20, 3}, "A", false);

  auto *var = mod_.createConstant(ElemKind::FloatTy, {3}, "var");
  auto *mean = mod_.createConstant(ElemKind::FloatTy, {3}, "mean");
  auto *beta = mod_.createConstant(ElemKind::FloatTy, {3}, "beta");
  auto *gamma = mod_.createConstant(ElemKind::FloatTy, {3}, "gamma");

  // (X - mean) * (1.0 / sqrt(var + eps)) * gamma + beta
  var->getPayloadMutable().getHandle<float>() = varV;
  mean->getPayloadMutable().getHandle<float>() = meanV;
  beta->getPayloadMutable().getHandle<float>() = betaV;
  gamma->getPayloadMutable().getHandle<float>() = gammaV;
  Node *BN = F_->createBatchNormalization("batch", A->getType(), A, beta, gamma,
                                          mean, var, 3, eps);
  Node *LRN = F_->createLocalResponseNormalization("LRN", BN);
  F_->createSave("ret", LRN);

  EXPECT_EQ(F_->getNodes().size(), 3);
  ::glow::convertPlaceholdersToConstants(F_, bindings_, {});
  optimizedF_ = optimizeFunctionForTest(F_);
  EXPECT_EQ(optimizedF_->getNodes().size(), 2);

  ASSERT_EQ(A->getNumUsers(), 2);
  Node *LRN1 = std::find_if_not(A->getUsers().begin(), A->getUsers().end(),
                                [BN](auto &it) { return it.getUser() == BN; })
                   ->getUser();
  ASSERT_TRUE(llvm::isa<LocalResponseNormalizationNode>(LRN1));
  ASSERT_EQ(LRN1->getNumUsers(), 1);
  Node *save = LRN1->getUsers().begin()->getUser();
  EXPECT_TRUE(llvm::isa<SaveNode>(save));

  bindings_.allocate(mod_.getPlaceholders());
  bindings_.get(A)->getHandle().randomize(-1.0, 1.0, mod_.getPRNG());
}

TEST_F(GraphOptz, optimizeRedundantBatchNorm1) {
  optimizeRedundantBatchNormTest(mod_, F_, optimizedF_, bindings_, {1., 1., 1.},
                                 {0., 0., 0.}, {1., 1., 1.}, {0., 0., 0.}, 0.0);
  checkNumericalEquivalence();
}

TEST_F(GraphOptz, optimizeRedundantBatchNorm2) {
  optimizeRedundantBatchNormTest(mod_, F_, optimizedF_, bindings_, {1., 1., 1.},
                                 {33., 33., 33.}, {1., 1., 1.}, {33., 33., 33.},
                                 0.0);
  checkNumericalEquivalence();
}

TEST_F(GraphOptz, optimizeRedundantBatchNorm3) {
  const float eps = 0.000001;
  optimizeRedundantBatchNormTest(
      mod_, F_, optimizedF_, bindings_, {1.0f - eps, 1.0f - eps, 1.0f - eps},
      {33., 33., 33.}, {1., 1., 1.}, {33., 33., 33.}, eps);
  checkNumericalEquivalence();
}
TEST_F(GraphOptz, optimizeRedundantBatchNorm4) {
  optimizeRedundantBatchNormTest(mod_, F_, optimizedF_, bindings_,
                                 {225., 225., 225.}, {-3., -3., -3.},
                                 {15., 15., 15.}, {-3., -3., -3.}, 0.0);
  checkNumericalEquivalence();
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

  optimizedF_ = optimizeFunctionForTest(F_);

  // BatchNorm should have been merged into the Conv.
  EXPECT_EQ(optimizedF_->getNodes().size(), 4);

  // Filter and Bias should have been duplicated so that the Conv-BN
  // optimization does not modify the filter/bias being saved, equaling 4
  // Constants. Additionally, the BN's Scale, Bias, Mean, and Var should be
  // eliminated due to the opti.
  EXPECT_EQ(mod_.getConstants().size(), 8);

  ASSERT_EQ(A->getNumUsers(), 2);
  Node *newCV = A->getUsers().back().getUser();
  EXPECT_TRUE(llvm::isa<ConvolutionNode>(newCV));
  ASSERT_EQ(newCV->getNumUsers(), 1);
  Node *save = newCV->getUsers().begin()->getUser();
  EXPECT_TRUE(llvm::isa<SaveNode>(save));

  EXPECT_EQ(
      countNodeKind(optimizedF_, Kinded::Kind::BatchNormalizationNodeKind), 0);

  bindings_.allocate(mod_.getPlaceholders());
  bindings_.get(A)->getHandle().randomize(-1.0, 1.0, mod_.getPRNG());
  checkNumericalEquivalence();
}

TEST_F(GraphOptz, optimizeBatchNormAfterConvFP16) {
  auto *A =
      mod_.createPlaceholder(ElemKind::Float16Ty, {1, 10, 20, 3}, "A", false);
  Node *CV = F_->createConv(bindings_, "conv", A, 16, 5, 1, 2, 1);
  Node *BN =
      F_->createBatchNormalization(bindings_, "batch", CV, 3, 0.0001, 0.9);
  F_->createSave("ret", BN);

  EXPECT_EQ(F_->getNodes().size(), 3);

  ::glow::convertPlaceholdersToConstants(F_, bindings_, {});
  optimizedF_ = optimizeFunctionForTest(F_);

  EXPECT_EQ(optimizedF_->getNodes().size(), 2);

  ASSERT_EQ(A->getNumUsers(), 2);

  bool optimizedPathExists{false};
  for (const auto &path : A->getUsers()) {
    auto cv = path.getUser();
    EXPECT_TRUE(llvm::isa<ConvolutionNode>(cv));
    ASSERT_EQ(cv->getNumUsers(), 1);
    auto next = cv->getUsers().begin()->getUser();
    optimizedPathExists |= llvm::isa<SaveNode>(next);
  }

  EXPECT_TRUE(optimizedPathExists);

  bindings_.allocate(A)->getHandle<float16_t>().randomize(-1.0, 1.0,
                                                          mod_.getPRNG());

  checkNumericalEquivalence();
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

  EXPECT_EQ(F_->getNodes().size(), 4);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::BatchNormalizationNodeKind), 1);

  ::glow::convertPlaceholdersToConstants(F_, bindings_, {input});
  optimizedF_ = optimizeFunctionForTest(F_);

  EXPECT_EQ(optimizedF_->getNodes().size(), 2);
  EXPECT_EQ(
      countNodeKind(optimizedF_, Kinded::Kind::BatchNormalizationNodeKind), 0);

  bindings_.allocate(input)->getHandle().randomize(-1.0, 1.0, mod_.getPRNG());
  checkNumericalEquivalence();
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

  auto *TN = F_->createTranspose("transpose", filter, HWCN2NHWC);
  auto *CV = F_->createConv("conv", input, TN, bias,
                            mod_.uniqueType(ElemKind::FloatTy, {1, 10, 20, 1}),
                            5, 1, 2, 1);
  auto *BN =
      F_->createBatchNormalization(bindings_, "batch", CV, 3, 0.0001, 0.9);
  F_->createSave("ret", BN);

  // Initialize to ensure that constant tensors are not optimized out.
  bindings_.allocate(filter)->getHandle().randomize(-1.0, 1.0, mod_.getPRNG());
  bindings_.allocate(bias)->getHandle().randomize(-1.0, 1.0, mod_.getPRNG());

  EXPECT_EQ(F_->getNodes().size(), 4);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::BatchNormalizationNodeKind), 1);

  ::glow::convertPlaceholdersToConstants(F_, bindings_, {input});
  optimizedF_ = optimizeFunctionForTest(F_);

  EXPECT_EQ(optimizedF_->getNodes().size(), 2);
  EXPECT_EQ(
      countNodeKind(optimizedF_, Kinded::Kind::BatchNormalizationNodeKind), 0);

  bindings_.allocate(input)->getHandle().randomize(-1.0, 1.0, mod_.getPRNG());
  checkNumericalEquivalence();
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

/// Testing merge of single-user arithmetic operation chain (Sub, Mul, Add)
/// into a BatchNorm.
TEST_F(GraphOptz, MergeBatchNormalizationWithArithmeticChainTest) {
  // Inputs.
  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {3, 2, 2, 4}, "input", false);
  auto *var = mod_.createConstant(ElemKind::FloatTy, {4}, "var");
  auto *mean = mod_.createConstant(ElemKind::FloatTy, {4}, "mean");
  auto *beta = mod_.createConstant(ElemKind::FloatTy, {4}, "beta");
  auto *gamma = mod_.createConstant(ElemKind::FloatTy, {4}, "gamma");

  Node *subC = mod_.createConstant(ElemKind::FloatTy, {3, 2, 2, 4}, "subC");
  Node *mulC = mod_.createConstant(ElemKind::FloatTy, {3, 2, 2, 4}, "mulC");
  Node *addC = mod_.createConstant(ElemKind::FloatTy, {3, 2, 2, 4}, "addC");
  Node *divC = mod_.createConstant(ElemKind::FloatTy, {3, 2, 2, 4}, "divC");

  // Fill tensors to check boundary values after the transformation.
  std::vector<float> betaV = {1., 2., 3., 7.};
  std::vector<float> gammaV = {4., 5., 6., 7.};

  var->getPayloadMutable().getHandle<float>() = {1., 1., 1., 1.};
  mean->getPayloadMutable().getHandle<float>() = {0., 0., 0., 0.};
  beta->getPayloadMutable().getHandle<float>() = betaV;
  gamma->getPayloadMutable().getHandle<float>() = gammaV;

  // For at least one node (sub) make values within channel different, to test
  // folding better.
  const std::vector<float> subV = {1, 2., 3., 4.};
  const float mulV = 4., addV = 3., divV = 2.;
  auto subH = llvm::cast<Constant>(subC)->getHandle<float>();
  subH = {1., 2., 3., 4., 1., 2., 3., 4., 1., 2., 3., 4., 1., 2., 3., 4.,
          1., 2., 3., 4., 1., 2., 3., 4., 1., 2., 3., 4., 1., 2., 3., 4.,
          1., 2., 3., 4., 1., 2., 3., 4., 1., 2., 3., 4., 1., 2., 3., 4.};

  llvm::cast<Constant>(mulC)->getHandle<float>().clear(mulV);
  llvm::cast<Constant>(addC)->getHandle<float>().clear(addV);
  llvm::cast<Constant>(divC)->getHandle<float>().clear(divV);

  BatchNormalizationNode *bn = F_->createBatchNormalization(
      "batch", input->getType(), input, beta, gamma, mean, var, 3);

  auto *sub = F_->createSub("sub", bn, subC);
  auto *mul = F_->createMul("mul", sub, mulC);
  auto *add = F_->createAdd("add", addC, mul);
  auto *div = F_->createDiv("div", add, divC);
  auto *res = F_->createSave("save", div);

  // Compile.
  EXPECT_EQ(F_->getNodes().size(), 6);
  ::glow::convertPlaceholdersToConstants(F_, bindings_, {input});
  optimizedF_ = optimizeFunctionForTest(F_);
  EXPECT_EQ(optimizedF_->getNodes().size(), 2);

  Constant *cs, *cb;

  auto *opt_res = findFunctionNodeByName<SaveNode>(optimizedF_, res->getName());

  auto *newBn = llvm::dyn_cast<BatchNormalizationNode>(opt_res->getInput());
  ASSERT_TRUE(newBn);

  cs = llvm::dyn_cast<Constant>(newBn->getScale());
  cb = llvm::dyn_cast<Constant>(newBn->getBias());
  ASSERT_TRUE(cs);
  ASSERT_TRUE(cb);
  ASSERT_TRUE(cs->getType()->isFPType());
  ASSERT_TRUE(cb->getType()->isFPType());

  auto hs = cs->getHandle<float>();
  auto hb = cb->getHandle<float>();

  // Verify that scale and offset are computed correctly.
  for (dim_t i = 0; i < 4; i++) {
    const float expScale = gammaV[i] * mulV / divV;
    const float expBias = ((betaV[i] - subV[i]) * mulV + addV) / divV;
    EXPECT_EQ(expScale, hs.raw(i));
    EXPECT_EQ(expBias, hb.raw(i));
  }

  bindings_.allocate(mod_.getPlaceholders());
  bindings_.get(input)->getHandle().randomize(-1.0, 1.0, mod_.getPRNG());
  checkNumericalEquivalence();
}

/// Testing merge of single-user arithmetic operation chain (Sub, Mul, Add)
/// into a BatchNorm.
TEST_F(GraphOptz, FoldArithmeticChainAfterConvIntoBatchNorm) {
  Node *subC = mod_.createConstant(ElemKind::FloatTy, {2, 3, 3, 3}, "subC");
  Node *mulC = mod_.createConstant(ElemKind::FloatTy, {2, 3, 3, 3}, "mulC");
  Node *addC = mod_.createConstant(ElemKind::FloatTy, {2, 3, 3, 3}, "addC");
  Node *divC = mod_.createConstant(ElemKind::FloatTy, {2, 3, 3, 3}, "divC");

  // Start with identity values.
  std::vector<float> betaV = {0., 0., 0.};
  std::vector<float> gammaV = {1., 1., 1.};

  // For at least one node make values within channel different, to test
  // the folding better (ideally all should have different values).
  const std::vector<float> subV = {1, 2., 3.};
  const float mulV = 4., addV = 3., divV = 2.;
  llvm::cast<Constant>(mulC)->getHandle<float>().clear(mulV);
  llvm::cast<Constant>(addC)->getHandle<float>().clear(addV);
  llvm::cast<Constant>(divC)->getHandle<float>().clear(divV);
  auto subH = llvm::cast<Constant>(subC)->getHandle<float>();
  subH = {1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3,
          1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3,
          1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3};

  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 2, 2, 3}, "input", false);
  auto filter =
      mod_.createPlaceholder(ElemKind::FloatTy, {3, 2, 2, 3}, "filter", false);
  auto *bias = mod_.createPlaceholder(ElemKind::FloatTy, {3}, "bias", false);
  bindings_.allocate(bias)->zero();

  ConvolutionNode *CV = F_->createConv(
      "Conv", input, filter, bias,
      mod_.uniqueType(ElemKind::FloatTy, {2, 3, 3, 3}), 2, 1, 1, 1);

  auto *sub = F_->createSub("sub", CV, subC);
  auto *mul = F_->createMul("mul", sub, mulC);
  auto *add = F_->createAdd("add", addC, mul);
  auto *div = F_->createDiv("div", add, divC);
  auto *res = F_->createSave("save", div);

  // Compile.
  EXPECT_EQ(F_->getNodes().size(), 6);
  ::glow::convertPlaceholdersToConstants(F_, bindings_, {});
  optimizedF_ = optimizeFunctionForTest(F_);
  EXPECT_EQ(optimizedF_->getNodes().size(), 3);

  auto *opt_res = findFunctionNodeByName<SaveNode>(optimizedF_, res->getName());

  Constant *cs, *cb;

  auto *bn = llvm::dyn_cast<BatchNormalizationNode>(opt_res->getInput());
  ASSERT_TRUE(bn);

  cs = llvm::dyn_cast<Constant>(bn->getScale());
  cb = llvm::dyn_cast<Constant>(bn->getBias());

  ASSERT_TRUE(cs);
  ASSERT_TRUE(cb);
  ASSERT_TRUE(cs->getType()->isFPType());
  ASSERT_TRUE(cb->getType()->isFPType());

  auto hs = cs->getHandle<float>();
  auto hb = cb->getHandle<float>();

  // Verify that scale and offset are computed correctly.
  for (dim_t i = 0; i < 3; i++) {
    const float expectedScale = gammaV[i] * (mulV / divV);
    const float expectedBias = ((betaV[i] - subV[i]) * mulV + addV) / divV;
    EXPECT_EQ(expectedScale, hs.raw(i));
    EXPECT_EQ(expectedBias, hb.raw(i));
  }
  bindings_.allocate(mod_.getPlaceholders());
  bindings_.get(input)->getHandle().randomize(-1.0, 1.0, mod_.getPRNG());
  bindings_.get(filter)->getHandle().randomize(-1.0, 1.0, mod_.getPRNG());
  bindings_.get(bias)->getHandle().randomize(-1.0, 1.0, mod_.getPRNG());
  checkNumericalEquivalence();
}

/// Check CSE will not merge two nodes that have all the same inputs but
/// different predicates.
TEST_F(GraphOptz, cseRespectsPredicates) {
  Placeholder *in = mod_.createPlaceholder(ElemKind::FloatTy, {5}, "in", false);
  Placeholder *pred1 =
      mod_.createPlaceholder(ElemKind::FloatTy, {1}, "pred", false);
  Placeholder *pred2 =
      mod_.createPlaceholder(ElemKind::FloatTy, {1}, "pred", false);

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
  optimizedF_ = optimizeFunctionForTest(F_);

  // Two RELUS and two Saves should still be there.
  EXPECT_EQ(F_->getNodes().size(), 4);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::ReluNodeKind), 2);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::SaveNodeKind), 2);

  bindings_.allocate(mod_.getPlaceholders());
  bindings_.get(in)->getHandle().randomize(-1.0, 1.0, mod_.getPRNG());
  checkNumericalEquivalence();
}

TEST_F(GraphOptz, optimizeBatchNormAfterConvButConvReused) {
  Placeholder *A =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 10, 20, 3}, "A", false);
  Node *CV = F_->createConv(bindings_, "conv", A, 16, 5, 1, 2, 1);
  Node *BN =
      F_->createBatchNormalization(bindings_, "batch", CV, 3, 0.0001, 0.9);
  F_->createSave("ret", BN);
  F_->createSave("convSave", CV);

  EXPECT_EQ(F_->getNodes().size(), 4);
  optimizedF_ = optimizeFunctionForTest(F_);
  // Make sure the structure of the graph did not change, since the convolution
  // node is used more than once.
  EXPECT_EQ(optimizedF_->getNodes().size(), 4);
  auto convIt =
      std::find_if(optimizedF_->getNodes().begin(),
                   optimizedF_->getNodes().end(), [](const Node &node) -> bool {
                     return llvm::isa<ConvolutionNode>(node);
                   });
  ASSERT_NE(convIt, optimizedF_->getNodes().end());
  auto batchNormIt =
      std::find_if(optimizedF_->getNodes().begin(),
                   optimizedF_->getNodes().end(), [](const Node &node) -> bool {
                     return (llvm::isa<BatchNormalizationNode>(node));
                   });
  ConvolutionNode *conv = llvm::dyn_cast<ConvolutionNode>(convIt);
  BatchNormalizationNode *batchNorm =
      llvm::dyn_cast<BatchNormalizationNode>(batchNormIt);

  EXPECT_EQ(*conv, *CV);
  EXPECT_EQ(batchNorm->getInput().getNode(), conv);
  EXPECT_EQ(conv->getInput().getNode(), A);

  bindings_.allocate(mod_.getPlaceholders());
  bindings_.get(A)->getHandle().randomize(-1.0, 1.0, mod_.getPRNG());
  checkNumericalEquivalence();
}

TEST_F(GraphOptz, optimizeBatchNormAfterConvButVarReused) {
  auto *A =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 10, 20, 3}, "A", false);

  ConvolutionNode *CV = F_->createConv(bindings_, "conv", A, 16, 5, 1, 2, 1);
  Node *BN =
      F_->createBatchNormalization(bindings_, "batch", CV, 3, 0.0001, 0.9);
  auto *retSaveNode = F_->createSave("ret", BN);
  auto *filterSaveNode = F_->createSave("filter", CV->getFilter());

  EXPECT_EQ(F_->getNodes().size(), 4);
  optimizedF_ = optimizeFunctionForTest(F_);
  ASSERT_EQ(A->getNumUsers(), 2);

  auto *optimizedF_ret =
      findFunctionNodeByName<SaveNode>(optimizedF_, retSaveNode->getName());
  auto *optimizedF_filterSave =
      findFunctionNodeByName<SaveNode>(optimizedF_, filterSaveNode->getName());

  // Make sure the structure of the graph did not change.
  EXPECT_EQ(optimizedF_->getNodes().size(), 4);
  EXPECT_TRUE(llvm::isa<Placeholder>(optimizedF_filterSave->getInput()));
  auto *varFilter =
      llvm::dyn_cast<Placeholder>(optimizedF_filterSave->getInput());
  EXPECT_EQ(varFilter, CV->getFilter());
  EXPECT_TRUE(llvm::isa<BatchNormalizationNode>(optimizedF_ret->getInput()));

  BatchNormalizationNode *batchNorm =
      llvm::dyn_cast<BatchNormalizationNode>(optimizedF_ret->getInput());
  ASSERT_TRUE(batchNorm);
  auto *newCVNode =
      llvm::dyn_cast<ConvolutionNode>(batchNorm->getInput().getNode());
  ASSERT_TRUE(newCVNode);
  EXPECT_EQ(newCVNode->getInput().getNode(), CV->getInput().getNode());
  EXPECT_EQ(newCVNode->getInput().getNode(), A);

  bindings_.allocate(mod_.getPlaceholders());
  bindings_.get(A)->getHandle().randomize(-1.0, 1.0, mod_.getPRNG());
  checkNumericalEquivalence();
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

/// Check that the Transpose is merged with Constant in a sequence
/// Transpose(Quantize(Constant)).
TEST_F(GraphOptz, transposeQuantizeConstant) {
  auto *qTy = mod_.uniqueType(ElemKind::Int8QTy, {1, 10, 20, 3}, 0.2, 0);
  auto *input = F_->getParent()->createConstant(ElemKind::FloatTy,
                                                {1, 10, 20, 3}, "input");
  auto *Q = F_->createQuantize("quantize", input, qTy);
  auto *T = F_->createTranspose("transpose", Q, NHWC2NCHW);
  auto *S = F_->createSave("save", T);

  // Skip ConstantFolding as it would have the same result as this opt.
  CompilationContext cctx;
  cctx.optimizationOpts.enableConstantFolding = false;

  EXPECT_EQ(F_->getNodes().size(), 3);
  ::glow::optimize(F_, cctx);
  EXPECT_EQ(F_->getNodes().size(), 2);

  // Constant and Quantize should have new shape.
  auto *newQ = llvm::dyn_cast<QuantizeNode>(S->getInput());
  ASSERT_TRUE(newQ);
  EXPECT_TRUE(newQ->getResult().dims().equals({1, 3, 10, 20}));
  auto *newC = llvm::dyn_cast<Constant>(newQ->getInput());
  ASSERT_TRUE(newC);
  EXPECT_TRUE(newC->getType()->dims().equals({1, 3, 10, 20}));
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
  Placeholder *A =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 10, 20, 3}, "A", false);
  Node *CV = F_->createConv(bindings_, "conv", A, 16, 5, 1, 2, 1);
  Node *BN =
      F_->createBatchNormalization(bindings_, "batch", CV, 3, 0.0001, 0.9);
  F_->createSave("ret", BN);

  EXPECT_EQ(F_->getNodes().size(), 3);

  optimizedF_ = F_->clone(F_->getName().str() + "_optimized");
  ::glow::optimize(optimizedF_, CompilationMode::Train);
  EXPECT_EQ(optimizedF_->getNodes().size(), 3);

  ASSERT_EQ(A->getNumUsers(), 2);
  Node *curCV = A->getUsers().begin()->getUser();
  EXPECT_EQ(curCV, CV);
  ASSERT_EQ(curCV->getNumUsers(), 1);
  Node *curBN = curCV->getUsers().begin()->getUser();
  EXPECT_EQ(curBN, BN);
  ASSERT_EQ(curBN->getNumUsers(), 1);
  Node *save = curBN->getUsers().begin()->getUser();
  EXPECT_TRUE(llvm::isa<SaveNode>(save));

  bindings_.allocate(mod_.getPlaceholders());
  bindings_.get(A)->getHandle().randomize(-1.0, 1.0, mod_.getPRNG());
  checkNumericalEquivalence();
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

enum class TestSinkTransposeNodesKind {
  BatchNormalization,
  Relu,
  LeakyRelu,
  Clip,
  Sigmoid,
  Tanh,
  Quantize,
};

class GraphOptzSinkTransposeBelowParametrized
    : public GraphOptz,
      public ::testing::WithParamInterface<TestSinkTransposeNodesKind> {
public:
  NodeValue getNodeFromInput(TestSinkTransposeNodesKind testNode, Node *T) {
    switch (testNode) {
    case TestSinkTransposeNodesKind::BatchNormalization: {
      return F_->createBatchNormalization(bindings_, "batch", T, 1, 0.0001, 0.9)
          ->getResult();
    }
    case TestSinkTransposeNodesKind::Relu: {
      return F_->createRELU("relu", T)->getResult();
    }
    case TestSinkTransposeNodesKind::LeakyRelu: {
      return F_->createLeakyRELU("leaky_relu", T, 0.1)->getResult();
    }
    case TestSinkTransposeNodesKind::Clip: {
      return F_->createClip("clip", T, 0.0, 6.0)->getResult();
    }
    case TestSinkTransposeNodesKind::Sigmoid: {
      return F_->createSigmoid("sigmoid", T)->getResult();
    }
    case TestSinkTransposeNodesKind::Tanh: {
      return F_->createTanh("tanh", T)->getResult();
    }
    case TestSinkTransposeNodesKind::Quantize: {
      return F_
          ->createQuantize(
              "quantize", T,
              mod_.uniqueType(ElemKind::Int8QTy, T->dims(0), 0.03, 5))
          ->getResult();
    }
    }
    LOG(DFATAL) << "Cannot reach here.";
    return NodeValue(); // Prevents a compilation warning.
  }
};

TEST_P(GraphOptzSinkTransposeBelowParametrized,
       TestSinkTransposeForDifferentCases) {
  const dim_t origDims[] = {1, 5, 10, 15};
  const dim_t transposedDims[] = {1, 15, 5, 10};
  auto *A = mod_.createPlaceholder(ElemKind::FloatTy, origDims, "input", false);
  Node *T = F_->createTranspose("transpose", A, NHWC2NCHW);
  auto IN = getNodeFromInput(GetParam(), T);
  SaveNode *O = F_->createSave("ret", IN);

  EXPECT_EQ(F_->getNodes().size(), 3);
  EXPECT_EQ(IN.dims(), llvm::makeArrayRef(transposedDims));

  optimizedF_ = optimizeFunctionForTest(F_);
  O = llvm::dyn_cast<SaveNode>(std::find_if(
      optimizedF_->getNodes().begin(), optimizedF_->getNodes().end(),
      [](const auto &N) { return N.getKind() == Kinded::Kind::SaveNodeKind; }));

  // Expecting Transpose->Output rather than N->Output.
  auto *transpose = llvm::dyn_cast<TransposeNode>(O->getInput());
  ASSERT_NE(transpose, nullptr);
  Node *N = transpose->getInput();
  ASSERT_TRUE(N);
  // Test correct input.
  if (GetParam() == TestSinkTransposeNodesKind::BatchNormalization) {
    ASSERT_EQ(BatchNormalizationNode::InputIdx, 0);
  } else {
    ASSERT_EQ(N->getNumInputs(), 1);
  }
  // Check that the dimensions of the input and output have been
  // updated to compensate the absence of transpose.
  EXPECT_EQ(transpose->getInput().dims(), llvm::makeArrayRef(origDims));
  EXPECT_EQ(N->getNthInput(0).dims(), llvm::makeArrayRef(origDims));
  EXPECT_EQ(F_->getNodes().size(), 3);

  bindings_.allocate(mod_.getPlaceholders());
  bindings_.get(A)->getHandle().randomize(-1.0, 1.0, mod_.getPRNG());
  checkNumericalEquivalence();
}

TEST_P(GraphOptzSinkTransposeBelowParametrized,
       TestSinkTransposeWithPredicateForDifferentCases) {
  if (GetParam() == TestSinkTransposeNodesKind::Quantize) {
    // Quantize does not work with generic test for predicates.
    return;
  }
  const dim_t origDims[] = {1, 5, 10, 15};
  const dim_t transposedDims[] = {1, 15, 5, 10};
  Node *A = mod_.createPlaceholder(ElemKind::FloatTy, origDims, "input", false);
  Node *pred1 = mod_.createPlaceholder(ElemKind::FloatTy, {1}, "pred", false);
  Node *pred2 = mod_.createPlaceholder(ElemKind::FloatTy, {1}, "pred", false);
  Node *pred3 = mod_.createPlaceholder(ElemKind::FloatTy, {1}, "pred", false);
  Node *T = F_->createTranspose("transpose", A, NHWC2NCHW);
  T->setPredicate(pred1);
  Node *IN = getNodeFromInput(GetParam(), T);
  IN->setPredicate(pred2);
  SaveNode *O = F_->createSave("ret", IN);
  O->setPredicate(pred3);

  EXPECT_EQ(F_->getNodes().size(), 3);
  EXPECT_EQ(IN->getNthResult(0).dims(), llvm::makeArrayRef(transposedDims));

  ::glow::optimize(F_, CompilationMode::Infer);

  EXPECT_EQ(O->getPredicate().getNode(), pred3);
  // Expecting Transpose->Output rather than N->Output.
  auto *transpose = llvm::dyn_cast<TransposeNode>(O->getInput());
  ASSERT_NE(transpose, nullptr);
  EXPECT_EQ(transpose->getPredicate().getNode(), pred2);
  Node *N = transpose->getInput();
  ASSERT_TRUE(N);
  EXPECT_EQ(N->getPredicate().getNode(), pred2);

  // Test correct input.
  if (GetParam() == TestSinkTransposeNodesKind::BatchNormalization) {
    ASSERT_EQ(BatchNormalizationNode::InputIdx, 0);
  } else {
    ASSERT_EQ(N->getNumInputs(), 1);
  }

  // Check that the dimensions of the input and output have been
  // updated to compensate the absence of transpose.
  EXPECT_EQ(transpose->getInput().dims(), llvm::makeArrayRef(origDims));
  EXPECT_EQ(N->getNthInput(0).dims(), llvm::makeArrayRef(origDims));
  EXPECT_EQ(F_->getNodes().size(), 3);
}

GLOW_INSTANTIATE_TEST_SUITE_P(
    TestSinkTranspose, GraphOptzSinkTransposeBelowParametrized,
    ::testing::Values(TestSinkTransposeNodesKind::BatchNormalization,
                      TestSinkTransposeNodesKind::Relu,
                      TestSinkTransposeNodesKind::LeakyRelu,
                      TestSinkTransposeNodesKind::Clip,
                      TestSinkTransposeNodesKind::Sigmoid,
                      TestSinkTransposeNodesKind::Tanh,
                      TestSinkTransposeNodesKind::Quantize));

TEST_F(GraphOptz, SinkTransposeBelowDequantize) {
  auto *in =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 5, 10, 15}, "input", false);
  auto *quantize = F_->createQuantize(
      "quantize", in, mod_.uniqueType(ElemKind::Int8QTy, in->dims(), 0.01, 2));
  auto *tile = F_->createTile("tile", quantize, 3, 0);
  auto *transpose = F_->createTranspose("transpose", tile, NHWC2NCHW);
  auto *deq = F_->createDequantize("dequantize", transpose, ElemKind::FloatTy);
  SaveNode *O = F_->createSave("out", deq);

  optimizedF_ = optimizeFunctionForTest(F_);

  EXPECT_EQ(F_->getNodes().size(), 5);
  EXPECT_EQ(optimizedF_->getNodes().size(), 5);

  auto *optOut = findFunctionNodeByName<SaveNode>(optimizedF_, O->getName());
  EXPECT_TRUE(llvm::isa<TransposeNode>(optOut->getInput().getNode()));

  bindings_.allocate(mod_.getPlaceholders());
  bindings_.get(in)->getHandle().randomize(-1.0, 1.0, mod_.getPRNG());
  checkNumericalEquivalence();
}

TEST_F(GraphOptz, SinkTransposeBelowPRelu) {
  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 5, 10, 15}, "input", false);
  auto *slope =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 5, 10, 15}, "slope", false);
  auto *OT = mod_.uniqueType(ElemKind::FloatTy, {1, 5, 10, 15});
  auto *prelu = F_->createPRELU("prelu", input, slope, OT);
  auto *transpose = F_->createTranspose("transpose", prelu, NHWC2NCHW);
  SaveNode *O = F_->createSave("out", transpose);

  optimizedF_ = optimizeFunctionForTest(F_);

  EXPECT_EQ(F_->getNodes().size(), 3);
  EXPECT_EQ(optimizedF_->getNodes().size(), 3);

  auto *optOut = findFunctionNodeByName<SaveNode>(optimizedF_, O->getName());
  EXPECT_TRUE(llvm::isa<TransposeNode>(optOut->getInput().getNode()));

  bindings_.allocate(mod_.getPlaceholders());
  bindings_.get(input)->getHandle().randomize(-1.0, 1.0, mod_.getPRNG());
  bindings_.get(slope)->getHandle().randomize(-1.0, 1.0, mod_.getPRNG());
  checkNumericalEquivalence();
}

TEST_F(GraphOptz, SinkTransposeBelowTile) {
  auto *in =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 5, 10, 15}, "input", false);
  auto *transpose = F_->createTranspose("transpose", in, NHWC2NCHW);
  auto *tile = F_->createTile("tile", transpose, 4, 1);
  auto *save = F_->createSave("save", tile);

  optimizedF_ = optimizeFunctionForTest(
      F_, {FunctionPassID::SinkCode, getDCEPassConfig()});

  EXPECT_EQ(F_->getNodes().size(), 3);
  EXPECT_EQ(optimizedF_->getNodes().size(), 3);

  auto *saveOpt =
      findFunctionNodeByName<SaveNode>(optimizedF_, save->getName());
  auto *transposeOpt = llvm::dyn_cast<TransposeNode>(saveOpt->getInput());
  ASSERT_TRUE(transposeOpt);
  EXPECT_EQ(transposeOpt->getShuffle(), transpose->getShuffle());
  auto *tileOpt = llvm::dyn_cast<TileNode>(transposeOpt->getInput());
  ASSERT_TRUE(tileOpt);
  EXPECT_EQ(tileOpt->getAxis(), 3);
  EXPECT_EQ(tileOpt->getCount(), 4);

  bindings_.allocate(mod_.getPlaceholders());
  bindings_.get(in)->getHandle().randomize(-1.0, 1.0, mod_.getPRNG());
  checkNumericalEquivalence();
}

TEST_F(GraphOptz, HoistTransposeAboveTile) {
  auto *in =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 5, 10, 15}, "input", false);
  auto *tile = F_->createTile("tile", in, 4, 3);
  auto *transpose = F_->createTranspose("transpose", tile, NHWC2NCHW);
  auto *save = F_->createSave("save", transpose);

  optimizedF_ = optimizeFunctionForTest(F_);

  EXPECT_EQ(F_->getNodes().size(), 3);
  EXPECT_EQ(optimizedF_->getNodes().size(), 3);

  auto *saveOpt =
      findFunctionNodeByName<SaveNode>(optimizedF_, save->getName());
  auto *tileOpt = llvm::dyn_cast<TileNode>(saveOpt->getInput());
  ASSERT_TRUE(tileOpt);
  EXPECT_EQ(tileOpt->getAxis(), 1);
  EXPECT_EQ(tileOpt->getCount(), 4);
  auto *transposeOpt = llvm::dyn_cast<TransposeNode>(tileOpt->getInput());
  ASSERT_TRUE(transposeOpt);
  EXPECT_EQ(transposeOpt->getShuffle(), transpose->getShuffle());

  bindings_.allocate(mod_.getPlaceholders());
  bindings_.get(in)->getHandle().randomize(-1.0, 1.0, mod_.getPRNG());
  checkNumericalEquivalence();
}

/// For example folding Rescale in to Convolution.
TEST_F(GraphOptz, sinkTransposeBelowRescale) {
  // Inputs.
  const dim_t origDims[] = {1, 5, 10, 15};
  const dim_t transposedDims[] = {1, 15, 5, 10};
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

TEST_F(GraphOptz, cancelTwoTransposes) {
  const dim_t origDims[] = {1, 5, 10, 15};
  Placeholder *A =
      mod_.createPlaceholder(ElemKind::FloatTy, origDims, "input", false);
  Node *T1 = F_->createTranspose("transpose", A, NCHW2NHWC);
  Node *T2 = F_->createTranspose("transpose", T1, NHWC2NCHW);
  ReluNode *K = F_->createRELU("relu", T2);
  SaveNode *save = F_->createSave("ret", K);

  EXPECT_EQ(K->getInput().dims(), llvm::makeArrayRef(origDims));
  EXPECT_EQ(F_->getNodes().size(), 4);

  optimizedF_ = optimizeFunctionForTest(F_);

  EXPECT_EQ(optimizedF_->getNodes().size(), 2);

  for (auto &N : optimizedF_->getNodes()) {
    if (N.getKind() == Kinded::Kind::SaveNodeKind) {
      save = llvm::dyn_cast<SaveNode>(&N);
    }
  }

  ReluNode *relu = llvm::dyn_cast<ReluNode>(save->getInput());
  ASSERT_TRUE(relu);
  EXPECT_EQ(relu->getResult().dims(), llvm::makeArrayRef(origDims));
  EXPECT_EQ(relu->getInput().getNode(), A);

  bindings_.allocate(mod_.getPlaceholders());
  bindings_.get(A)->getHandle().randomize(-1.0, 1.0, mod_.getPRNG());

  checkNumericalEquivalence();
}

/// Make sure the predicates don't get in the way of the
/// transpose(transpose) => identity and that they are
/// preserved.
TEST_F(GraphOptz, cancelTwoTransposesWithPredicate) {
  const dim_t origDims[] = {1, 5, 10, 15};
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
  const dim_t origDims[] = {1, 5, 10, 15};
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
  const dim_t origDims[] = {1, 5, 10, 15};
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
  const dim_t origDims[] = {1, 5, 10, 15};
  const dim_t finalDims[] = {5, 1, 15, 10};

  Placeholder *A =
      mod_.createPlaceholder(ElemKind::FloatTy, origDims, "input", false);
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

  optimizedF_ = optimizeFunctionForTest(F_);
  // Find save node in the optimized graph.
  for (auto &N : optimizedF_->getNodes()) {
    if (N.getKind() == Kinded::Kind::SaveNodeKind) {
      save = llvm::dyn_cast<SaveNode>(&N);
    }
  }
  // Get the last transpose node in the optimized graph.
  auto *TR = llvm::dyn_cast<TransposeNode>(save->getInput());
  ASSERT_NE(TR, nullptr);

  EXPECT_EQ(optimizedF_->getNodes().size(), 2);
  EXPECT_EQ(TR->getResult().dims(), llvm::makeArrayRef(finalDims));
  EXPECT_EQ(A->getNthResult(0).dims(), llvm::makeArrayRef(origDims));
  EXPECT_EQ(TR->getInput().getNode(), A);

  bindings_.allocate(mod_.getPlaceholders());
  bindings_.get(A)->getHandle().randomize(-1.0, 1.0, mod_.getPRNG());
  checkNumericalEquivalence();
}

TEST_F(GraphOptz, sinkTransposeBelowArithmeticNodes) {
  const dim_t origDims[] = {1, 5, 10, 15};
  Node *A1 =
      mod_.createPlaceholder(ElemKind::FloatTy, origDims, "input1", false);
  Node *A2 =
      mod_.createPlaceholder(ElemKind::FloatTy, origDims, "input2", false);
  Node *T1 = F_->createTranspose("transpose1", A1, NHWC2NCHW);
  Node *T2 = F_->createTranspose("transpose2", A2, NHWC2NCHW);
  Node *K = F_->createAdd("arith", T1, T2);
  Node *P = F_->createPow("pow", K, T2);
  SaveNode *O = F_->createSave("ret", P);

  EXPECT_EQ(F_->getNodes().size(), 5);

  ::glow::optimize(F_, CompilationMode::Infer);

  // Expecting Transpose->Output rather than Add->Output.
  auto *transpose = llvm::dyn_cast<TransposeNode>(O->getInput());
  ASSERT_NE(transpose, nullptr);
  auto *pow = llvm::dyn_cast<PowNode>(transpose->getInput());
  ASSERT_TRUE(pow);
  auto *add = llvm::dyn_cast<AddNode>(pow->getLHS());
  ASSERT_TRUE(add);
  // Check that the dimensions of the input and output have been
  // updated to compensate the absence of transpose.
  EXPECT_EQ(add->getResult().dims(), llvm::makeArrayRef(origDims));
  EXPECT_EQ(add->getRHS().dims(), llvm::makeArrayRef(origDims));
  EXPECT_EQ(add->getLHS().dims(), llvm::makeArrayRef(origDims));
  EXPECT_EQ(add->getLHS().getNode(), A1);
  EXPECT_EQ(add->getRHS().getNode(), A2);

  EXPECT_EQ(F_->getNodes().size(), 4);
}

/// Check that Transpose node is sunk below arithmetic nodes when one of the
/// operands is a Constant.
TEST_F(GraphOptz, sinkTransposeBelowArithmeticNodesWithConstantOperand) {
  const dim_t origDims[] = {1, 5, 10, 15};
  const dim_t transposedDims[] = {1, 15, 5, 10};

  // Create one subgraph in which the Constant is the LHS operand of the Add.
  Constant *C1 = mod_.createConstant(ElemKind::FloatTy, transposedDims, "C1");
  // Initialize the payload before optimization so that it can be copied to the
  // new Constant that will be created by the GraphOptimizer.
  C1->getHandle().randomize(-1, 1, mod_.getPRNG());

  auto *P1 = mod_.createPlaceholder(ElemKind::FloatTy, origDims, "P1", false);
  auto *T1 = F_->createTranspose("T1", P1, NHWC2NCHW);
  auto *A1 = F_->createAdd("A1", C1, T1);
  SaveNode *S1 = F_->createSave("S1", A1);

  // Create one subgraph in which the Constnat is the RHS operand of the Add.
  Constant *C2 = mod_.createConstant(ElemKind::FloatTy, transposedDims, "C2");
  // Initialize the payload before optimization so that it can be copied to the
  // new Constant that will be created by the GraphOptimizer.
  C2->getHandle().randomize(-1, 1, mod_.getPRNG());

  auto *P2 = mod_.createPlaceholder(ElemKind::FloatTy, origDims, "P2", false);
  auto *T2 = F_->createTranspose("T2", P2, NHWC2NCHW);
  auto *A2 = F_->createAdd("A2", T2, C2);
  SaveNode *S2 = F_->createSave("S2", A2);

  EXPECT_EQ(F_->getNodes().size(), 6);

  optimizedF_ = optimizeFunctionForTest(F_);

  // Find the SaveNodes of the optimized graph.
  for (auto &N : optimizedF_->getNodes()) {
    if (N.getKind() == Kinded::Kind::SaveNodeKind) {
      if (N.getName() == S1->getName()) {
        S1 = llvm::dyn_cast<SaveNode>(&N);
      }

      if (N.getName() == S2->getName()) {
        S2 = llvm::dyn_cast<SaveNode>(&N);
      }
    }
  }

  // Expecting Transpose->Output rather than Add->Output.
  auto *transpose = llvm::dyn_cast<TransposeNode>(S1->getInput());
  ASSERT_NE(transpose, nullptr);
  auto *add = llvm::dyn_cast<AddNode>(transpose->getInput());
  ASSERT_TRUE(add);
  // Check that the dimensions of the input and output of the add have been
  // updated to compensate the absence of transpose.
  EXPECT_EQ(add->getResult().dims(), llvm::makeArrayRef(origDims));
  EXPECT_EQ(add->getRHS().dims(), llvm::makeArrayRef(origDims));
  EXPECT_EQ(add->getLHS().dims(), llvm::makeArrayRef(origDims));
  EXPECT_EQ(add->getRHS().getNode(), P1);

  // Repeat checks for other subgraph.
  transpose = llvm::dyn_cast<TransposeNode>(S2->getInput());
  ASSERT_NE(transpose, nullptr);
  add = llvm::dyn_cast<AddNode>(transpose->getInput());
  ASSERT_TRUE(add);
  EXPECT_EQ(add->getResult().dims(), llvm::makeArrayRef(origDims));
  EXPECT_EQ(add->getRHS().dims(), llvm::makeArrayRef(origDims));
  EXPECT_EQ(add->getLHS().dims(), llvm::makeArrayRef(origDims));
  EXPECT_EQ(add->getLHS().getNode(), P2);

  EXPECT_EQ(optimizedF_->getNodes().size(), 6);

  // Check that the original and optimized functions are numerically equivalent.
  // This indirectly checks that the Constant has been transposed properly.
  bindings_.allocate(mod_.getPlaceholders());
  bindings_.get(P1)->getHandle().randomize(-1, 1, mod_.getPRNG());
  bindings_.get(P2)->getHandle().randomize(-1, 1, mod_.getPRNG());

  checkNumericalEquivalence();
}

/// Test sink Transpose below Add of which operands has the same element type
/// and shape, but different scale and offset.
TEST_F(GraphOptz, sinkQuantTransposeBelowArithmeticNodesWithConstantOperand) {
  const dim_t origDims[] = {1, 5, 10, 15};
  const dim_t transposedDims[] = {1, 15, 5, 10};

  // Create graph where a Add take a Constant in LHS and Transpose in RHS.
  // LHS and RHS has different scale and offset.
  Constant *lhsC =
      mod_.createConstant(ElemKind::Int8QTy, transposedDims, 0.2, 0, "C1");
  lhsC->getHandle<int8_t>().randomize(-128, 127, mod_.getPRNG());

  auto *inputP =
      mod_.createPlaceholder(ElemKind::FloatTy, origDims, "Input", false);
  auto *qTy = mod_.uniqueType(ElemKind::Int8QTy, origDims, 0.3, 2);
  auto *quant = F_->createQuantize("Quant", inputP, qTy);
  auto *rhsT = F_->createTranspose("RHS", quant, NHWC2NCHW);
  auto *addQ = F_->createAdd("Add", lhsC, rhsT);
  SaveNode *save = F_->createSave("Save", addQ);

  EXPECT_EQ(F_->getNodes().size(), 4);

  optimizedF_ = optimizeFunctionForTest(F_);

  // Expecting Transpose->Output rather than Add->Output.
  const auto *saveOpt =
      findFunctionNodeByName<SaveNode>(optimizedF_, save->getName());
  auto *transpose = llvm::dyn_cast<TransposeNode>(saveOpt->getInput());
  ASSERT_NE(transpose, nullptr);
  auto *add = llvm::dyn_cast<AddNode>(transpose->getInput());
  ASSERT_TRUE(add);
  // Check that the dimensions of the input and output of the add have been
  // updated to compensate the absence of transpose.
  EXPECT_EQ(add->getResult().dims(), llvm::makeArrayRef(origDims));
  EXPECT_EQ(add->getLHS().dims(), llvm::makeArrayRef(origDims));
  EXPECT_EQ(add->getRHS().dims(), llvm::makeArrayRef(origDims));
  quant = llvm::dyn_cast<QuantizeNode>(add->getRHS().getNode());
  ASSERT_TRUE(quant);
  EXPECT_EQ(quant->getInput().getNode(), inputP);
  EXPECT_EQ(optimizedF_->getNodes().size(), 4);

  // Check that the original and optimized functions are numerically equivalent.
  // This indirectly checks that the Constant has been transposed properly.
  bindings_.allocate(mod_.getPlaceholders());
  bindings_.get(inputP)->getHandle().randomize(-128, 127, mod_.getPRNG());

  checkNumericalEquivalence();
}

/// Check that the predicates are properly preserved while doing
/// the add(transpose, transpose) => transpose(add).
TEST_F(GraphOptz, sinkTransposeBelowArithmeticNodesWithPredicate) {
  const dim_t origDims[] = {1, 5, 10, 15};
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
  const dim_t origDims[] = {1, 5, 10, 15};
  const dim_t origDimsConcat[] = {1, 10, 10, 15};
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
  const dim_t origDims[] = {1, 5, 10, 15};
  const dim_t origDimsConcat[] = {1, 10, 10, 15};
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
  const dim_t origDims[] = {1, 5, 10, 15};
  const dim_t origDimsConcat[] = {1, 5, 20, 15};
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
  const dim_t origDims[] = {1, 5, 10, 15};
  const dim_t origDimsConcat[] = {1, 5, 20, 15};
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
  const dim_t inputDims[] = {1, 5, 10, 15};
  const dim_t outTransposeDims[] = {1, 10, 15, 5};
  const dim_t outPadDims[] = {5, 18, 25, 11};
  // Padding before the optimization.
  int pads[] = {0, 2, 3, 1, 4, 6, 7, 5};

  // The shape of the graph after the optimization.
  const dim_t outPadDimsAfterOptim[] = {5, 11, 18, 25};
  const dim_t outTransposeDimsAfterOptims[] = {5, 18, 25, 11};
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

TEST_F(GraphOptz, sinkTransposeBelowRelu) {
  // Define a type with custom alignments.
  Type typeWithAlignments(ElemKind::FloatTy, {2, 3, 4, 5}, {1, 1, 32, 1});
  Type transposedTypeWithAlignments(ElemKind::FloatTy, {2, 4, 5, 3},
                                    {1, 1, 32, 1});
  auto modTyWithAlignments = mod_.uniqueType(typeWithAlignments);
  auto modTransposedTyWithAlignments =
      mod_.uniqueType(transposedTypeWithAlignments);
  auto *A1 = mod_.createPlaceholder(modTyWithAlignments, "input1", false);
  auto *T1 = F_->createTranspose("transpose", A1, NCHW2NHWC);
  T1->setType(0, modTransposedTyWithAlignments);
  auto *RN = F_->createRELU("relu", T1);
  SaveNode *O = F_->createSave("ret", RN);

  EXPECT_EQ(F_->getNodes().size(), 3);

  ::glow::optimize(F_, CompilationMode::Infer);

  // Expecting Transpose->Output rather than Relu->Output, because Transpose was
  // sinked.
  auto *transpose = llvm::dyn_cast<TransposeNode>(O->getInput());
  ASSERT_NE(transpose, nullptr);
  auto *relu = llvm::dyn_cast<ReluNode>(transpose->getInput());
  ASSERT_TRUE(relu);
  // Check that alignments are preserved by optimizations.
  ASSERT_TRUE(relu->getInput().getType()->isEqual(modTyWithAlignments));
  ASSERT_TRUE(transpose->getInput().getType()->isEqual(modTyWithAlignments));
  ASSERT_TRUE(
      transpose->getResult().getType()->isEqual(modTransposedTyWithAlignments));

  EXPECT_EQ(F_->getNodes().size(), 3);
  ASSERT_TRUE(F_->verify());
}

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
  EXPECT_FALSE(functionContainsNode(F_, CN1));

  // CN2 should be merged into a new CN4 and removed by the optimizations.
  EXPECT_FALSE(functionContainsNode(F_, CN2));

  // CN3 should not be merged into CN4 and should not be removed,
  // because CN4 and CN3 have a different dimension parameter.
  EXPECT_TRUE(functionContainsNode(F_, CN3));

  // The CN4 concat node should be replaced by a merged concat node.
  EXPECT_FALSE(functionContainsNode(F_, CN4));

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
  EXPECT_TRUE(functionContainsNode(F_, CN1));

  // CSE should replace CN2 by CN1 and remove CN2.
  EXPECT_FALSE(functionContainsNode(F_, CN2));

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
  ASSERT_TRUE(CN);

  EXPECT_TRUE(CN->getResult().getType()->dims().equals({94, 73, 35}));
}

/// Test Clip(Splat(args)) -> Splat(args').
TEST_F(GraphOptz, ClipOfSplatNode) {
  Type T(ElemKind::FloatTy, {10, 10});
  SplatNode *splat = F_->createSplat("zero", &T, 5);
  ClipNode *clipMin = F_->createClip("clip", splat, 10, 15);
  ClipNode *clipMax = F_->createClip("clip", splat, 0, 2);
  ClipNode *clipSame = F_->createClip("clip", splat, 0, 10);
  SaveNode *saveMin = F_->createSave("saveMin", clipMin);
  SaveNode *saveMax = F_->createSave("saveMax", clipMax);
  SaveNode *saveSame = F_->createSave("saveSame", clipSame);

  // Start with one splat, three clips, three saves.
  EXPECT_EQ(F_->getNodes().size(), 7);

  ::glow::optimize(F_, CompilationMode::Infer);

  // We will end up with three Splats and three saves.
  EXPECT_EQ(F_->getNodes().size(), 6);

  SplatNode *splatMin = llvm::dyn_cast<SplatNode>(saveMin->getInput());
  ASSERT_TRUE(splatMin);
  EXPECT_EQ(splatMin->getValue(), 10);

  SplatNode *splatMax = llvm::dyn_cast<SplatNode>(saveMax->getInput());
  ASSERT_TRUE(splatMax);
  EXPECT_EQ(splatMax->getValue(), 2);

  ASSERT_EQ(saveSame->getInput().getNode(), splat);
  EXPECT_EQ(splat->getValue(), 5);
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

  optimizedF_ = optimizeFunctionForTest(F_);

  bindings_.allocate(mod_.getPlaceholders());
  bindings_.get(input)->getHandle().randomize(-1.0, 1.0, mod_.getPRNG());

  checkNumericalEquivalence();
}

// Similar to ZeroArithmetic, but tests that nodes with multiple results are
// correctly handled (i.e. that the correct output is selected after optimising
// away an arithmetic identity).
TEST_F(GraphOptz, ZeroArithmeticMultiResNode) {
  auto *input = mod_.createPlaceholder(ElemKind::FloatTy, {10}, "input", true);
  auto *topK = F_->createTopK("topK", input, /*k=*/5);
  auto *zero = F_->createSplat("zero", topK->getValues().getType(), 0.);
  auto *add = F_->createAdd("add", topK->getValues(), zero);
  auto *sub = F_->createSub("sub", topK->getValues(), zero);

  SaveNode *AS = F_->createSave("ret", add);
  SaveNode *SS = F_->createSave("ret", sub);

  // There should be 6 nodes: 2 Saves, Add, Sub, Splat and TopK.
  EXPECT_EQ(F_->getNodes().size(), 6);

  optimizedF_ = optimizeFunctionForTest(F_);

  // Now there should only be 3 nodes: TopK and 2 Saves.
  EXPECT_EQ(optimizedF_->getNodes().size(), 3);

  auto *OAS = findFunctionNodeByName<SaveNode>(optimizedF_, AS->getName());
  auto *OSS = findFunctionNodeByName<SaveNode>(optimizedF_, SS->getName());
  auto *OTopK = findFunctionNodeByName<TopKNode>(optimizedF_, topK->getName());

  // Since the operations reprsented by the arithmetic nodes are no-ops,
  // the input to both SaveNodes should be the Values result of TopKNode.
  EXPECT_EQ(OAS->getInput(), OTopK->getValues());
  EXPECT_EQ(OSS->getInput(), OTopK->getValues());

  // Check numerical equivalence.
  bindings_.allocate(mod_.getPlaceholders());
  bindings_.get(input)->getHandle().randomize(-1.0, 1.0, mod_.getPRNG());

  checkNumericalEquivalence();
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
  EXPECT_TRUE(functionContainsNode(F_, O));
  EXPECT_TRUE(functionContainsNode(F_, zero));
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
  SaveNode *save = F_->createSave("ret", mul);

  // Splat, Div, Mul, Save.
  EXPECT_EQ(F_->getNodes().size(), 4);
  // Save optimized function for future comparision
  optimizedF_ = optimizeFunctionForTest(F_);

  // The expression evaluates to "I", so Save is only node left.
  EXPECT_EQ(optimizedF_->getNodes().size(), 1);
  SaveNode *SN =
      llvm::dyn_cast<SaveNode>(optimizedF_->getNodeByName(save->getName()));
  ASSERT_TRUE(functionContainsNode(optimizedF_, SN));
  ASSERT_NE(SN, nullptr);

  // Save node should just save the input.
  EXPECT_TRUE(SN->getInput().getNode() == input);

  bindings_.allocate(mod_.getPlaceholders());
  bindings_.get(input)->getHandle().randomize(-1.0, 1.0, mod_.getPRNG());

  checkNumericalEquivalence();
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

    CompilationContext cctx;
    cctx.compMode = CompilationMode::Train;
    // Do not perform any compile-time constant folding.
    cctx.optimizationOpts.enableConstantFolding = false;
    ::glow::optimize(F, cctx);

    // This test illustrates some inconsistency in the optimization.
    // Chain splats are not guaranteed to be optimized.
    EXPECT_EQ(F->getNodes().size(), shouldReverse ? 3 : 2);
  }
}

TEST_F(GraphOptz, ReshapeNoop) {
  const dim_t shape[] = {10, 20, 30};
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
  const dim_t shape[] = {10, 20, 30};
  const dim_t reshape[] = {1, 6000};
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

  cctx_.compMode = CompilationMode::Infer;
  // Do not perform any compile-time constant folding.
  cctx_.optimizationOpts.enableConstantFolding = false;
  ::glow::optimize(F_, cctx_);

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
  EXPECT_TRUE(functionContainsNode(F_, R1));

  // R3 and Z2 should not be in the graph any more.
  EXPECT_FALSE(functionContainsNode(F_, R3));
  EXPECT_FALSE(functionContainsNode(F_, Z2));
}

/// Test the Reshape(Reshape(x)) -> Reshape(x) transformation.
TEST_F(GraphOptz, ReshapeReshapeOpt) {
  const dim_t shape[] = {10, 20};
  const dim_t reshape1[] = {200, 1};
  const dim_t reshape2[] = {200};
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
  EXPECT_FALSE(functionContainsNode(F_, R1));
  EXPECT_FALSE(functionContainsNode(F_, R2));
}

TEST_F(GraphOptz, DCEPublicVars) {
  mod_.createPlaceholder(ElemKind::FloatTy, {4, 320, 200, 3}, "input", true);

  EXPECT_EQ(mod_.getPlaceholders().size(), 1);

  // Optimize all of the dead code.
  ::glow::optimize(F_, CompilationMode::Infer);

  //  Public nodes should not be deleted.
  EXPECT_EQ(mod_.getPlaceholders().size(), 1);
}

TEST_F(GraphOptz, foldQuantizeIntoConstant) {
  auto *input = mod_.createPlaceholder(ElemKind::FloatTy, {4}, "input", true);
  *bindings_.allocate(input) = {10, 10, 10, 10};
  auto qType = mod_.uniqueType(ElemKind::Int8QTy, {4}, 2, 0);

  auto *Q = F_->createQuantize("quantize", input, qType);
  auto *S = F_->createSave("save", Q);

  EXPECT_EQ(2, F_->getNodes().size());
  ::glow::convertPlaceholdersToConstants(F_, bindings_, {S->getPlaceholder()});

  // 'optimize' doesn't merge quantize nodes into Constant.
  ::glow::optimize(F_, CompilationMode::Infer);
  EXPECT_EQ(2, F_->getNodes().size());

  // 'convertQuantizedConstants' merges quantize nodes into Constant
  CompilationContext cctx;
  ::glow::convertQuantizedConstants(F_, cctx);
  EXPECT_EQ(1, F_->getNodes().size());

  auto quantizedInput = llvm::cast<Constant>(S->getInput());
  auto quantizedValues = quantizedInput->getHandle<int8_t>();
  for (unsigned i = 0; i < 4; ++i) {
    EXPECT_EQ(5, quantizedValues.raw(i));
  }
}

TEST_F(GraphOptz, foldQuantizeIntoConstantMultipleUsages) {
  auto *input = mod_.createPlaceholder(ElemKind::FloatTy, {4}, "input", true);
  *bindings_.allocate(input) = {10, 10, 10, 10};
  auto qType = mod_.uniqueType(ElemKind::Int8QTy, {4}, 2, 0);

  auto *Q = F_->createQuantize("quantize", input, qType);
  F_->createSave("save", Q);
  auto clonedF = F_->clone("cloned");

  EXPECT_EQ(2, clonedF->getNodes().size());
  ::glow::convertPlaceholdersToConstants(clonedF, bindings_, {});
  CompilationContext cctx;
  ::glow::convertQuantizedConstants(clonedF, cctx);

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

/// Search for a unique Save node in input graph \p F and return it.
/// Fails in case there is no Save node or more than one detected.
static SaveNode *getUniqueSaveNode(Function *F) {
  SaveNode *foundSaveNode = nullptr;
  for (auto &node : F->getNodes()) {
    if (auto *s = llvm::dyn_cast<SaveNode>(&node)) {
      EXPECT_EQ(foundSaveNode, nullptr);
      foundSaveNode = s;
    }
  }
  EXPECT_NE(foundSaveNode, nullptr);
  return foundSaveNode;
}

/// Mock backend that requests the pre-quantization of constants.
class MockBackendPrequantizeConst : public MockBackend {
  bool shouldPreQuantizeConstants() const override { return true; }
  bool isOpSupported(const NodeInfo &) const override { return true; }
  Expected<bool>
  transformPostLowering(Function *F, CompilationContext &,
                        const glow::runtime::DeviceInfo *) const override {
    // Check the IR.
    EXPECT_EQ(F->getNodes().size(), 1);
    auto *save = getUniqueSaveNode(F);
    EXPECT_TRUE(llvm::isa<Constant>(save->getInput()));

    return false;
  }
};
/// Mock backend that requests the non pre-quantization of constants.
class MockBackendNotPrequantizeConst : public MockBackend {
  bool shouldPreQuantizeConstants() const override { return false; }
  bool isOpSupported(const NodeInfo &) const override { return true; }
  Expected<bool>
  transformPostLowering(Function *F, CompilationContext &,
                        const glow::runtime::DeviceInfo *) const override {
    // Check the IR.
    EXPECT_EQ(F->getNodes().size(), 2);
    auto *save = getUniqueSaveNode(F);
    auto *quant = llvm::dyn_cast<QuantizeNode>(save->getInput());
    EXPECT_TRUE(quant);
    EXPECT_TRUE(llvm::isa<Constant>(quant->getInput()));

    return false;
  }
};

/// Test the actual constant quantization for backends.
template <typename Backend>
void testFoldQuantizeIntoConstant(Module &mod_, Function *F_) {
  auto *input = mod_.createConstant(ElemKind::FloatTy, {4}, "input");
  input->getHandle<float>() = {10, 10, 10, 10};
  auto qType = mod_.uniqueType(ElemKind::Int8QTy, {4}, 2, 0);
  auto *Q = F_->createQuantize("quantize", input, qType);
  auto *save = F_->createSave("save", Q);

  CompilationContext cctx;
  auto B = Backend();
  // Note: the check that Quantize is or not folded into Constant before
  // post-lowering is done in <backend>::transformPostLowering()
  EXIT_ON_ERR(::glow::optimizeFunction(F_, B, cctx));

  // Check the IR (the constant must have been quantized).
  EXPECT_EQ(F_->getNodes().size(), 1);
  EXPECT_TRUE(llvm::isa<Constant>(save->getInput()));
}

/// Check the backend actual constant quantization is done before post-lowering.
TEST_F(GraphOptz, foldQuantizeIntoConstantBeforePostLowering) {
  testFoldQuantizeIntoConstant<MockBackendPrequantizeConst>(mod_, F_);
}

/// Check the backend actual constant quantization is done after post-lowering.
TEST_F(GraphOptz, foldQuantizeIntoConstantAfterPostLowering) {
  testFoldQuantizeIntoConstant<MockBackendNotPrequantizeConst>(mod_, F_);
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

  DequantizeNode *Q = F_->createDequantize("dequantize", SN, ElemKind::FloatTy);
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
  auto *D = F_->createDequantize("dequantize", R, ElemKind::FloatTy);
  F_->createSave("ret", D);

  EXPECT_EQ(F_->getNodes().size(), 3);
  ::glow::optimize(F_, CompilationMode::Infer);

  // Only 2 nodes should remain (Dequantize -> Save)
  EXPECT_EQ(F_->getNodes().size(), 2);

  // Check the graph structure
  auto *SN = F_->getNodeByName("ret_save");
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

  auto *D = F_->createDequantize("dequantize", input, ElemKind::FloatTy);

  auto qType = mod_.uniqueType(ElemKind::Int8QTy, {4, 10}, 0.03, 5);
  auto *Q = F_->createQuantize("quantize", D, qType);

  F_->createSave("ret", Q);

  EXPECT_EQ(F_->getNodes().size(), 3);

  ::glow::optimize(F_, CompilationMode::Infer);
  EXPECT_EQ(F_->getNodes().size(), 2);
}

TEST_F(GraphOptz, MaxOfQuantizedSplat) {
  const dim_t size = 5;
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
  auto opOutTy = mod_.uniqueType(ElemKind::Int8QTy, {10, 10}, 1, 0);
  auto rescaleOutTy = mod_.uniqueType(ElemKind::Int8QTy, {10, 10}, 2, 1);

  Placeholder *LHS = mod_.createPlaceholder(ElemKind::Int8QTy, {10, 10}, 0.4, 0,
                                            "LHS", /* isTrainable */ false);
  Placeholder *RHS = mod_.createPlaceholder(ElemKind::Int8QTy, {10, 10}, 0.3, 0,
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
                         llvm::ArrayRef<dim_t> inputDims,
                         llvm::ArrayRef<int> pads, unsigned_t convKernelSize,
                         llvm::ArrayRef<unsigned_t> convPads,
                         unsigned_t convStride, unsigned_t convNumKernels) {
  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, inputDims, "input", true);

  // Pad
  dim_t inputWithPadDims[4];
  for (int i = 0; i < 4; i++) {
    inputWithPadDims[i] = dim_t(ssize_t(inputDims[i]) + pads[i] + pads[4 + i]);
  }
  dim_t outputConvDims[4] = {
      inputWithPadDims[0],
      inputWithPadDims[1] + convPads[0] + convPads[2] - (convKernelSize - 1),
      inputWithPadDims[2] + convPads[1] + convPads[3] - (convKernelSize - 1),
      convNumKernels};

  auto outTy = mod_.uniqueType(ElemKind::FloatTy, inputWithPadDims);
  Node *P =
      F_->createPad("pad", input, outTy, PaddingMode::CONSTANT, pads, 0.f);

  // Convolution
  dim_t filterDims[] = {convNumKernels, convKernelSize, convKernelSize,
                        inputDims[3]};
  auto *F =
      mod_.createPlaceholder(ElemKind::FloatTy, filterDims, "filter", true);
  auto *B =
      mod_.createPlaceholder(ElemKind::FloatTy, {convNumKernels}, "bias", true);
  auto *CV = F_->createConv(
      "conv", P, F, B, mod_.uniqueType(ElemKind::FloatTy, outputConvDims),
      {convKernelSize, convKernelSize}, {convStride, convStride}, convPads, 1);

  SaveNode *O = F_->createSave("save", CV);

  // The pad node must be merged into convolution.
  EXPECT_EQ(F_->getNodes().size(), 3);
  ::glow::optimize(F_, CompilationMode::Infer);
  EXPECT_EQ(F_->getNodes().size(), 2);

  // Check the graph structure and additional properties after optimization.
  auto *conv = llvm::dyn_cast<ConvolutionNode>(O->getInput());
  ASSERT_NE(conv, nullptr);
  EXPECT_EQ(conv->getResult().dims(), llvm::ArrayRef<dim_t>(outputConvDims));
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
  std::vector<dim_t> dims = {5, 2};

  auto *input = mod_.createPlaceholder(ElemKind::FloatTy, dims, "input", true);

  const float leakyAlpha = 0.05f;
  auto OutTy = mod_.uniqueType(ElemKind::FloatTy, dims);
  SplatNode *splatNode = F_->createSplat("splat", OutTy, leakyAlpha);
  MulNode *mulNode = F_->createMul("mul", input, splatNode);
  MaxNode *maxNode = F_->createMax("max", input, mulNode);
  SaveNode *output = F_->createSave("save", maxNode);

  EXPECT_EQ(4, F_->getNodes().size());

  CompilationContext cctx;
  ::glow::fold(F_, cctx);

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
  std::vector<dim_t> dims = {5, 2};
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

  CompilationContext cctx;
  ::glow::fold(F_, cctx);

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

/// Test optimization of  Convolution nodes with small input tensors by reducing
/// filters and removing redundant padding.
TEST_F(GraphFold, optimizeSmallConv) {
  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 2, 2, 16}, "input", true);
  auto filter =
      mod_.createConstant(ElemKind::FloatTy, {16, 5, 5, 16}, "filter");
  auto bias = mod_.createConstant(ElemKind::FloatTy, {16}, "bias");

  filter->getHandle().randomize(-1.0, 1.0, mod_.getPRNG());
  bias->getHandle().randomize(-1.0, 1.0, mod_.getPRNG());

  auto *outTy = mod_.uniqueType(ElemKind::FloatTy, {1, 1, 1, 16});
  auto *CN = F_->createConv("conv", input, filter, bias, outTy, {5, 5}, {2, 2},
                            {2, 1, 1, 2}, 1);
  auto *save = F_->createSave("save", CN);

  EXPECT_EQ(2, F_->getNodes().size());
  optimizedF_ = optimizeFunctionForTest(F_);
  EXPECT_EQ(2, optimizedF_->getNodes().size());

  const auto *optSave =
      findFunctionNodeByName<SaveNode>(optimizedF_, save->getName());

  auto *newCN = llvm::dyn_cast<ConvolutionNode>(optSave->getInput());
  ASSERT_TRUE(newCN);
  // Kernel should be reduced.
  EXPECT_TRUE(isUniformArray(newCN->getKernels(), 2u));
  // Padding should be removed.
  EXPECT_TRUE(isUniformArray(newCN->getPads(), 0u));
  // Stride should be canonicalized to 1.
  EXPECT_TRUE(isUniformArray(newCN->getStrides(), 1u));

  bindings_.allocate(mod_.getPlaceholders());
  bindings_.get(input)->getHandle().randomize(-1.0, 1.0, mod_.getPRNG());
  checkNumericalEquivalence();
}

TEST_F(GraphOptz, GatherToSliceOpt) {
  auto *LHS = mod_.createPlaceholder(ElemKind::Int32ITy, {16, 3}, "LHS", false);
  auto *RHS = mod_.createConstant(ElemKind::Int32ITy, {}, "RHS");
  RHS->getPayloadMutable().getHandle<int32_t>() = {1};

  auto *gather = F_->createGather("gather", LHS, RHS, 1);
  auto *save = F_->createSave("save", gather);

  optimizedF_ = optimizeFunctionForTest(F_);

  auto *saveOpt =
      llvm::dyn_cast<SaveNode>(optimizedF_->getNodeByName(save->getName()));
  ASSERT_TRUE(saveOpt);
  auto *reshapeN = llvm::dyn_cast<ReshapeNode>(saveOpt->getInput());
  ASSERT_TRUE(reshapeN);
  EXPECT_EQ(reshapeN->getResult().dims().size(), 1);
  EXPECT_EQ(reshapeN->getResult().dims()[0], 16);

  bindings_.allocate(LHS)->getHandle<int32_t>().randomize(-128, 127,
                                                          mod_.getPRNG());
  checkNumericalEquivalence();
}

/// Fold a Convolution dilated manually using Transpose, SpaceToDepth and
/// DepthToSpace nodes into a single Convolution node. Pattern:
/// NHWC2CHWN -> S2D -> CHWN2NHWC -> Conv -> NHWC2CHWN -> D2S -> CHWN2NHWC
TEST_F(GraphFold, foldDilatedConv) {
  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 10, 10, 16}, "input", true);

  auto *T1 = F_->createTranspose("t1", input, NHWC2CHWN, "NHWC");
  auto *S2D = F_->createSpaceToDepth("s2d", T1, 2);
  auto *T2 = F_->createTranspose("t2", S2D, CHWN2NHWC, "NHWC");
  auto *CN = F_->createConv(bindings_, "conv", T2, 16, 3, 1, 0, 16, {1, 1});
  auto *T3 = F_->createTranspose("t3", CN, NHWC2CHWN, "NHWC");
  auto *D2S = F_->createDepthToSpace("d2s", T3, 2);
  auto *T4 = F_->createTranspose("t4", D2S, CHWN2NHWC, "NHWC");
  auto *save = F_->createSave("save", T4);

  // To spice things up, add additional users for some nodes. The pattern should
  // still be recognized.
  F_->createSave("save_t1", T1);
  F_->createSave("save_s2d", S2D);
  F_->createSave("save_t2", T2);

  EXPECT_EQ(13, F_->getNodes().size());
  optimizedF_ = optimizeFunctionForTest(F_);
  EXPECT_EQ(8, optimizedF_->getNodes().size());

  const auto *optSave =
      findFunctionNodeByName<SaveNode>(optimizedF_, save->getName());

  auto *newCN = llvm::dyn_cast<ConvolutionNode>(optSave->getInput());
  ASSERT_TRUE(newCN);
  EXPECT_TRUE(isUniformArray(newCN->getDilation(), 2u));

  bindings_.allocate(mod_.getPlaceholders());
  bindings_.get(input)->getHandle().randomize(-1.0, 1.0, mod_.getPRNG());
  checkNumericalEquivalence();
}

/// Fold a Convolution dilated manually using Transpose, SpaceToDepth and
/// DepthToSpace nodes into a single Convolution node. Pattern:
/// NHWC2CHWN -> S2D -> CHWN2NHWC -> Conv -> NHWC2CHWN -> D2S -> CHWN2NHWC
/// Test for ChannelwiseQuantizedConvolution.
TEST_F(GraphFold, foldDilatedConv_ChannelwiseQuantized) {
  auto *input = mod_.createPlaceholder(ElemKind::Int8QTy, {1, 10, 10, 16}, 1.f,
                                       0, "input", true);

  auto *filterF =
      mod_.createConstant(ElemKind::FloatTy, {16, 3, 3, 16}, "filterF");
  filterF->getPayloadMutable().getHandle<float>().randomize(-1.0, 1.0,
                                                            mod_.getPRNG());
  auto *biasF = mod_.createConstant(ElemKind::FloatTy, {16}, "biasF");
  biasF->getPayloadMutable().getHandle<float>().randomize(-1.0, 1.0,
                                                          mod_.getPRNG());

  auto *T1 = F_->createTranspose("t1", input, NHWC2CHWN, "NHWC");
  auto *S2D = F_->createSpaceToDepth("s2d", T1, 2);
  auto *T2 = F_->createTranspose("t2", S2D, CHWN2NHWC, "NHWC");
  auto outTy = mod_.uniqueType(ElemKind::Int8QTy, {4, 3, 3, 16}, 1.f, 0);
  auto *CN = F_->createChannelwiseQuantizedConv(
      "conv", T2, filterF, biasF, nullptr, nullptr, nullptr, nullptr, outTy,
      {3, 3}, {1, 1}, {0, 0, 0, 0}, 1, {1, 1}, true, true,
      quantization::Schema::Asymmetric, ElemKind::Int8QTy, ElemKind::Int32QTy);
  auto *T3 = F_->createTranspose("t3", CN, NHWC2CHWN, "NHWC");
  auto *D2S = F_->createDepthToSpace("d2s", T3, 2);
  auto *T4 = F_->createTranspose("t4", D2S, CHWN2NHWC, "NHWC");
  auto *save = F_->createSave("save", T4);

  EXPECT_EQ(10, F_->getNodes().size());
  optimizedF_ = optimizeFunctionForTest(F_);
  EXPECT_EQ(2, optimizedF_->getNodes().size());

  const auto *optSave =
      findFunctionNodeByName<SaveNode>(optimizedF_, save->getName());

  auto *newCN =
      llvm::dyn_cast<ChannelwiseQuantizedConvolutionNode>(optSave->getInput());
  ASSERT_TRUE(newCN);
  EXPECT_TRUE(isUniformArray(newCN->getDilation(), 2u));

  bindings_.allocate(mod_.getPlaceholders());
  bindings_.get(input)->getHandle<int8_t>().randomize(-128, 127,
                                                      mod_.getPRNG());
  checkNumericalEquivalence();
}

/// Testing folding of Reshape->Transpose->Reshape into ChannelShuffle.
TEST_F(GraphFold, foldChannelShuffle) {
  const dim_t inputDims[] = {3, 136, 28, 28};

  Node *K =
      mod_.createPlaceholder(ElemKind::FloatTy, inputDims, "input", false);
  K = F_->createReshape("CS_reshape1", K, {3, 4, 34, 28, 28});
  K = F_->createTranspose("CS_transpose", K, {0, 2, 1, 3, 4});
  K = F_->createReshape("CS_reshape2", K, {3, 136, 28, 28});
  auto *save = F_->createSave("ret", K);

  EXPECT_EQ(F_->getNodes().size(), 4);

  // Fold RN->TR->RN into ChannelShuffle
  CompilationContext cctx;
  ::glow::fold(F_, cctx);

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

  CompilationContext cctx;
  ::glow::fold(F_, cctx);

  EXPECT_EQ(F_->getNodes().size(), 4);
  EXPECT_FALSE(llvm::isa<ChannelShuffleNode>(save->getInput()));
}

class MockBackendWithFusion : public MockBackend {
  bool supportsFusedActivation(Node *parent, Node *activation) const override {
    switch (parent->getKind()) {
    case Kinded::Kind::ConvolutionNodeKind:
      switch (activation->getKind()) {
      case Kinded::Kind::ReluNodeKind:
      case Kinded::Kind::ClipNodeKind:
      case Kinded::Kind::SigmoidNodeKind:
      case Kinded::Kind::TanhNodeKind:
      case Kinded::Kind::LeakyReluNodeKind:
        return true;
      default:
        return false;
      }
    default:
      return false;
    }
  }
};

#define CONV_ACTIVATION_TEST(ACTIVATION_, CREATOR_, ...)                       \
  TEST_F(GraphFold, FoldConv##ACTIVATION_##Activation) {                       \
    auto *A =                                                                  \
        mod_.createPlaceholder(ElemKind::FloatTy, {1, 10, 20, 3}, "A", false); \
    ConvolutionNode *CV =                                                      \
        F_->createConv(bindings_, "conv", A, 16, 5, 1, 2, 1);                  \
    auto *AN = F_->CREATOR_(__VA_ARGS__);                                      \
    SaveNode *SN = F_->createSave("ret", AN);                                  \
                                                                               \
    EXPECT_EQ(F_->getNodes().size(), 3);                                       \
                                                                               \
    CompilationContext cctx;                                                   \
    auto B = MockBackendWithFusion();                                          \
    ::glow::fold(F_, cctx, &B);                                                \
                                                                               \
    ConvolutionNode *fusedCV =                                                 \
        llvm::dyn_cast<ConvolutionNode>(SN->getInput());                       \
    ASSERT_TRUE(fusedCV);                                                      \
    EXPECT_EQ(fusedCV->getFusedActivation(), FusedActivation::ACTIVATION_);    \
  }

CONV_ACTIVATION_TEST(RELU, createRELU, "Relu", CV);
CONV_ACTIVATION_TEST(CLIP, createClip, "Clip", CV, 0.0, 1.0);
CONV_ACTIVATION_TEST(SIGMOID, createSigmoid, "Sigmoid", CV);
CONV_ACTIVATION_TEST(TANH, createTanh, "Tanh", CV);
CONV_ACTIVATION_TEST(LEAKY_RELU, createLeakyRELU, "LeakyRelu", CV, 1.0);

#undef CONV_ACTIVATION_TEST

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
  auto opOutTy = mod_.uniqueType(ElemKind::Int8QTy, {10, 10}, 1, 0);
  auto rescaleOutTy = mod_.uniqueType(ElemKind::Int8QTy, {10, 10}, 2, 1);

  Placeholder *LHS =
      mod_.createPlaceholder(ElemKind::Int8QTy, {10, 10}, 0.4, 0, "LHS", true);
  Placeholder *RHS =
      mod_.createPlaceholder(ElemKind::Int8QTy, {10, 10}, 0.3, 0, "RHS", true);

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
  auto *save = F_->createSave("ret", maxpool->getResult());

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
  for (dim_t i = 0; i < 10; i++) {
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
  for (dim_t i = 0; i < 10; i++) {
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

/// Check that EliminateNoop optimization pass removes nodes which don't do
/// anything useful.
TEST_F(GraphOptz, eliminateNoop) {
  std::vector<dim_t> shape = {1, 2, 2, 3};
  Placeholder *input1 = mod_.createPlaceholder(ElemKind::Int8QTy, shape, 0.004,
                                               0, "input", false);
  Placeholder *input2 = mod_.createPlaceholder(ElemKind::Int8QTy, shape, 0.004,
                                               0, "input", false);
  auto *cond = mod_.createConstant(ElemKind::BoolTy, shape, "input1");
  cond->getHandle<bool>() = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

  auto *select = F_->createSelect("select", cond, input1, input2);
  auto *slice = F_->createSlice("slice", select, {0, 0, 0, 0}, shape);
  auto *tile = F_->createTile("tile", slice, 1, 1);
  auto *pad = F_->createPad("pad", tile, tile->getResult().getType(), 0,
                            {0, 0, 0, 0, 0, 0, 0, 0}, 0);
  auto *avgPool = F_->createAvgPool("avgpool", pad, 1, 1, 0);
  auto *maxPool = F_->createMaxPool("maxpool", avgPool, 1, 1, 0);

  F_->createSave("save", maxPool->getResult());

  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::SelectNodeKind), 1);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::SliceNodeKind), 1);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::TileNodeKind), 1);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::PadNodeKind), 1);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::AvgPoolNodeKind), 1);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::MaxPoolNodeKind), 1);

  optimizedF_ = optimizeFunctionForTest(F_);

  // Check that all nodes except for Save are eliminated.
  EXPECT_EQ(optimizedF_->getNodes().size(), 1);

  bindings_.allocate(mod_.getPlaceholders());
  bindings_.get(input1)->getHandle<int8_t>().randomize(-1.0, 1.0,
                                                       mod_.getPRNG());
  bindings_.get(input2)->getHandle<int8_t>().randomize(-1.0, 1.0,
                                                       mod_.getPRNG());

  checkNumericalEquivalence();
}

// Check that we are able to replace
// Add(I, tile(B)) with -> BatchedAdd(I, B).
TEST_F(GraphOptz, FoldTileAddIntoBatchedAdd) {
  auto *batch =
      mod_.createPlaceholder(ElemKind::FloatTy, {3, 1, 2}, "batch", false);
  auto *added = mod_.createConstant(ElemKind::FloatTy, {1, 1, 2}, "added");
  auto *addedTiled = F_->createTile("addedTiled", added, 3, 0);
  auto *add = F_->createAdd("add", batch, addedTiled);
  auto *save = F_->createSave("save", add);
  auto *output = save->getPlaceholder();

  bindings_.allocate(batch)->getHandle() = {2, 2, 3, 3, 4, 4};
  added->getPayloadMutable().getHandle() = {1, 1};
  bindings_.allocate(output);

  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::TileNodeKind), 1);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::AddNodeKind), 1);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::BatchedAddNodeKind), 0);

  ASSERT_TRUE(F_->verify());

  // Currently the FoldTileAddIntoBatchedAdd opt which we're testing here is not
  // part of the default optimization pipeline. Create a local version of the
  // pipeline with that pass included.
  auto p = createDefaultGraphOptimizationPassPipeline();
  p->pushFront({FunctionPassID::FoldTileAddIntoBatchedAdd});
  FunctionPassManager FPM("opt", std::move(p));
  FPM.run(F_, CompilationContext());
  ASSERT_TRUE(F_->verify());

  // Check that the Tile node and the Add node is replaced by
  // a BatchedAdd node.
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::TileNodeKind), 0);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::AddNodeKind), 0);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::BatchedAddNodeKind), 1);

  // Verify the correctness of the input to BatchedAdd operator.
  // The correctness of BatchedAdd operator itself is verified
  // by operator's unit tests.
  Tensor expectedBatch(ElemKind::FloatTy, {3, 1, 2});
  expectedBatch.getHandle() = {2, 2, 3, 3, 4, 4};
  Tensor expectedSlice(ElemKind::FloatTy, {1, 2});
  expectedSlice.getHandle() = {1, 1};
  for (auto &node : F_->getNodes()) {
    auto *recvdBANode = llvm::dyn_cast<BatchedAddNode>(&node);
    if (!recvdBANode) {
      continue;
    }
    auto *recvdBatch = llvm::dyn_cast<Placeholder>(recvdBANode->getBatch());
    ASSERT_TRUE(recvdBatch);
    auto *recvdSlice = llvm::dyn_cast<Constant>(recvdBANode->getSlice());
    ASSERT_TRUE(recvdSlice);
    EXPECT_TRUE(recvdBatch->dims().equals({3, 1, 2}));
    EXPECT_TRUE(recvdSlice->dims().equals({1, 2}));
    EXPECT_TRUE(bindings_.get(recvdBatch)->isEqual(expectedBatch));
    EXPECT_TRUE(recvdSlice->getPayload().isEqual(expectedSlice));
    break;
  }
}

/// Test Concat(Slice, ..., Slice) opt works correctly. If \p reverseOrder then
/// the optimization is inapplicable and should not occur.
static void testConcatElim(Module &mod, Function *F, Function *&optimizedF,
                           PlaceholderBindings &bindings, bool reverseOrder) {
  Placeholder *input =
      mod.createPlaceholder(ElemKind::FloatTy, {10, 10, 10}, "input", true);
  bindings.allocate(input)->getHandle().randomize(-1.0, 1.0, mod.getPRNG());

  // Split the input to a bunch of small slices.
  std::array<NodeValue, 10> inputs;
  for (dim_t i = 0; i < 10; i++) {
    dim_t idx = reverseOrder ? 9 - i : i;
    inputs[i] =
        F->createSlice("extract", input, {idx, 0, 0}, {idx + 1, 10, 10});
  }

  auto *cc = F->createConcat("merge", inputs, 0);
  F->createSave("save", cc);

  EXPECT_EQ(countNodeKind(F, Kinded::Kind::SliceNodeKind), 10);

  optimizedF = optimizeFunctionForTest(F);

  // Check that either the concat and slices are gone if the optimization was
  // applicable, or otherwise that they're still there.
  EXPECT_EQ(countNodeKind(optimizedF, Kinded::Kind::ConcatNodeKind),
            reverseOrder ? 1 : 0);
  EXPECT_EQ(countNodeKind(optimizedF, Kinded::Kind::SliceNodeKind),
            reverseOrder ? 10 : 0);
}

// Check that we are able to eliminate concat nodes.
TEST_F(GraphOptz, concatElim) {
  testConcatElim(mod_, F_, optimizedF_, bindings_, /* reverseOrder */ false);
  checkNumericalEquivalence(0.0f);
}

// Check that when the order of the Slices is reversed no optimization kicks in.
TEST_F(GraphOptz, concatElimReverseOrder) {
  testConcatElim(mod_, F_, optimizedF_, bindings_, /* reverseOrder */ true);
  checkNumericalEquivalence(0.0f);
}

/// Check that we are able to eliminate concat nodes with redundant arithmetic
/// ops in way.
TEST_F(GraphOptz, concatArithElim) {
  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {10, 10, 10}, "input", true);
  bindings_.allocate(input)->getHandle().randomize(-1.0, 1.0, mod_.getPRNG());

  Type t(ElemKind::FloatTy, {1, 10, 10});
  Node *one = F_->createSplat("one", &t, 1.0);
  Node *zero = F_->createSplat("zero", &t, 0.0);

  // Split the input to a bunch of small slices.
  std::vector<NodeValue> inputs;
  for (dim_t i = 0; i < 10; i++) {
    auto *K = F_->createSlice("extract", input, {i, 0, 0}, {i + 1, 10, 10});
    // Insert the nodes in reverse order to make sure that we can catch
    // non-consecutive graph-order slices.
    Node *N = K;
    switch (i) {
    case 0:
      N = F_->createAdd("add0", K, zero);
      break;
    case 1:
      N = F_->createSub("sub0", K, zero);
      break;
    case 2:
      N = F_->createAdd("add_0", zero, K);
      break;
    case 3:
      N = F_->createMul("mul1", K, one);
      break;
    case 4:
      N = F_->createDiv("div1", K, one);
      break;
    case 5:
      N = F_->createMul("mul_1", one, K);
      break;
    default:
      break;
    }
    inputs.push_back(N);
  }

  auto *cc = F_->createConcat("merge", inputs, 0);
  F_->createSave("save", cc);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::SliceNodeKind), 10);
  optimizedF_ = optimizeFunctionForTest(F_);

  // Check that the concat node is gone.
  EXPECT_EQ(countNodeKind(optimizedF_, Kinded::Kind::ConcatNodeKind), 0);
  checkNumericalEquivalence(0.0f);
}

/// Check that we are able to eliminate concat followed by slices on axis
/// \p dim under certain conditions.
static void testConcatSliceElim(Module &mod, Function *F, Function *&optimizedF,
                                PlaceholderBindings &bindings, size_t dim) {
  constexpr size_t N = 5;
  std::array<NodeValue, N> inputs;
  std::vector<dim_t> inShape = {10, 20};
  inShape.insert(inShape.begin() + dim, 0);
  for (dim_t i = 0; i < N; i++) {
    inShape[dim] = 1 + i;
    auto *P = mod.createPlaceholder(ElemKind::FloatTy, inShape, "in", true);
    bindings.allocate(P)->getHandle().randomize(-1.0, 1.0, mod.getPRNG());
    inputs[i] = P;
  }
  auto *CN = F->createConcat("merge", inputs, dim);

  // Split the concat to a bunch of slices of the same shape as the concat
  // inputs and on the same axis.
  std::vector<dim_t> startShape = {0, 0, 0};
  std::vector<dim_t> endShape = {10, 20};
  endShape.insert(endShape.begin() + dim, 0);
  for (dim_t i = 0; i < N; i++) {
    startShape[dim] = (i * (i + 1)) / 2;
    endShape[dim] = ((i + 1) * (i + 2)) / 2;
    auto *SN = F->createSlice("extract", CN, startShape, endShape);
    F->createSave("save", SN);
  }

  // We created a concat followed by N slices of its results.
  EXPECT_EQ(countNodeKind(F, Kinded::Kind::SliceNodeKind), N);
  EXPECT_EQ(countNodeKind(F, Kinded::Kind::ConcatNodeKind), 1);

  optimizedF = optimizeFunctionForTest(F);

  // Check that the concat and slices are gone.
  EXPECT_EQ(countNodeKind(optimizedF, Kinded::Kind::ConcatNodeKind), 0);
  EXPECT_EQ(countNodeKind(optimizedF, Kinded::Kind::SliceNodeKind), 0);
}

TEST_F(GraphOptz, concatSliceElimInnerDim) {
  testConcatSliceElim(mod_, F_, optimizedF_, bindings_, 0);
  checkNumericalEquivalence(0.0f);
}

TEST_F(GraphOptz, concatSliceElimMiddleDim) {
  testConcatSliceElim(mod_, F_, optimizedF_, bindings_, 1);
  checkNumericalEquivalence(0.0f);
}

TEST_F(GraphOptz, concatSliceElimOuterDim) {
  testConcatSliceElim(mod_, F_, optimizedF_, bindings_, 2);
  checkNumericalEquivalence(0.0f);
}

/// Check the interaction between Sices(Concat) and Concat(Slices) optimizations
/// to make sure they work nicely together. Builds Concat(Slices(Concat)) and
/// expected a single Concat after optimizations.
TEST_F(GraphOptz, concatSliceElimMultiConcat) {
  std::array<NodeValue, 4> inputs;
  for (size_t i = 0; i < 4; i++) {
    auto *P = mod_.createPlaceholder(ElemKind::FloatTy, {2, 4},
                                     "in_" + std::to_string(i), false);
    bindings_.allocate(P)->getHandle().randomize(-1.0, 1.0, mod_.getPRNG());
    inputs[i] = P;
  }
  auto *CN0 = F_->createConcat("merge0", inputs, /* axis */ 1);

  auto *SN0 = F_->createSlice("slice0", CN0, {0, 0}, {2, 4});
  auto *SN1 = F_->createSlice("slice1", CN0, {0, 4}, {2, 8});
  auto *SN2 = F_->createSlice("slice2", CN0, {0, 8}, {2, 12});
  auto *SN3 = F_->createSlice("slice3", CN0, {0, 12}, {2, 16});

  auto *CN1 = F_->createConcat("merge1", {SN1, SN0, SN3, SN2}, /* axis */ 1);
  F_->createSave("save", CN1);

  // We created a concat followed by 4 slices of its results followed by another
  // concat.
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::ConcatNodeKind), 2);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::SliceNodeKind), 4);

  optimizedF_ = optimizeFunctionForTest(F_);

  // Check that one concat and slices are gone.
  EXPECT_EQ(countNodeKind(optimizedF_, Kinded::Kind::ConcatNodeKind), 1);
  EXPECT_EQ(countNodeKind(optimizedF_, Kinded::Kind::SliceNodeKind), 0);

  checkNumericalEquivalence(0.0f);
}

// Check the transformation Concat(Reshape(x) * N) -> Reshape(Concat(x * N)).
TEST_F(GraphOptz, concatReshapes) {
  const dim_t shape1[] = {2, 5, 2, 1, 20};
  const dim_t shape2[] = {10, 2, 2, 10};
  const dim_t shape3[] = {5, 80};
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
  EXPECT_FALSE(functionContainsNode(F_, concatNode1));
  // concatNode2 should still exist.
  EXPECT_TRUE(functionContainsNode(F_, concatNode2));

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

// Making sure we do not try to to optimize concat2(dim1, concat1(dim2, X, Y),
// Z)
// -> concat(dim1, X, Y, Z) when concat1 has multiple users.
TEST_F(GraphOptz, ConcatSimplificationNegative) {
  const dim_t dim1[] = {1, 4, 4, 4};
  const dim_t dim2[] = {1, 4, 4, 8};
  auto *in1 = mod_.createPlaceholder(ElemKind::FloatTy, dim1, "in1", false);
  auto *in2 = mod_.createPlaceholder(ElemKind::FloatTy, dim1, "in2", false);
  auto *in3 = mod_.createPlaceholder(ElemKind::FloatTy, dim2, "in3", false);

  auto *cnc1 = F_->createConcat("cnc1", {in1, in2}, 3);
  auto *add1 = F_->createAdd("add1", in3, cnc1);
  auto *cnc2 = F_->createConcat("cnc2", {add1, cnc1}, 3);
  F_->createSave("ret", cnc2);
  EXPECT_EQ(F_->getNodes().size(), 4);
  ::glow::optimize(F_, CompilationMode::Infer);
  EXPECT_EQ(F_->getNodes().size(), 4);
  for (auto &n : F_->getNodes()) {
    if (auto *tcnc = llvm::dyn_cast<ConcatNode>(&n)) {
      EXPECT_EQ(tcnc->getNumInputs(), 2);
    }
  }
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

  cctx_.compMode = CompilationMode::Infer;
  // Do not perform any compile-time constant folding.
  cctx_.optimizationOpts.enableConstantFolding = false;
  ::glow::optimize(F_, cctx_);

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

TEST_F(GraphOptz, VarsCSENaN) {
  // Create two variables that are Private, are not trainable, have no writers
  // and include NaNs. The first two variables have the same data, and so should
  // be combined via variable CSE.  In particular, the NaN constants should not
  // prevent the variables from being combine.
  auto *input1 = mod_.createConstant(ElemKind::FloatTy, {5}, "input1");
  auto *input2 = mod_.createConstant(ElemKind::FloatTy, {5}, "input2");
  input1->getHandle() = {0, NAN, 2, NAN, 4};
  input2->getHandle() = {0, NAN, 2, NAN, 4};

  // Input them each to different nodes, so node CSE does not change them.
  auto *TN = F_->createTanh("tanh", input1);
  auto *SN = F_->createSigmoid("sigmoid", input2);
  auto *CN = F_->createConcat("concat", {TN, SN}, /* axis */ 0);
  F_->createSave("ret", CN);

  // Initially there are two variables: inputs 1 and 2 (the save uses a
  // placeholder).
  EXPECT_EQ(mod_.getConstants().size(), 2);

  cctx_.compMode = CompilationMode::Infer;
  // Do not perform any compile-time constant folding.
  cctx_.optimizationOpts.enableConstantFolding = false;
  ::glow::optimize(F_, cctx_);

  // Now only one variables is left; input1 and input2 have been combined.
  EXPECT_EQ(mod_.getConstants().size(), 1);

  // Verify that only one of input1 and input2 exists.
  Constant *varOneOrTwo = nullptr;
  for (auto *V : mod_.getConstants()) {
    if (V == input1 || V == input2) {
      EXPECT_TRUE(varOneOrTwo == nullptr);
      varOneOrTwo = V;
    }
  }
  EXPECT_TRUE(varOneOrTwo != nullptr);

  // Verify that the users of the inputs are updated correctly.
  EXPECT_TRUE(TN->getInput().getNode() == varOneOrTwo);
  EXPECT_TRUE(SN->getInput().getNode() == varOneOrTwo);

  // Verify that whichever input1/input2 is left over has two users TN and SN.
  EXPECT_TRUE(varOneOrTwo->getUsers().size() == 2);
  for (auto &U : varOneOrTwo->getUsers()) {
    auto *N = U.getUser();
    EXPECT_TRUE(N == TN || N == SN);
  }
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
  ASSERT_TRUE(functionContainsNode(F_, SN));

  // Save node should just save the input.
  EXPECT_TRUE(SN->getInput().getNode() == input);
}

/// Test that a reshape of a private variable with one use has the reshape
/// merged into the variable.
TEST_F(GraphOptz, ReshapeConstantOneUse) {
  const dim_t shape[] = {10, 20};
  const dim_t reshape1[] = {200, 1};
  const dim_t reshape2[] = {200};
  Constant *input =
      F_->getParent()->createConstant(ElemKind::FloatTy, shape, "input");
  input->getHandle().randomize(-1.0, 1.0, mod_.getPRNG());

  auto *R1 = F_->createReshape("reshape1", input, reshape1);
  auto *R2 = F_->createReshape("reshape2", R1, reshape2);
  auto *O = F_->createSave("ret", R2);

  // Before optimization, we have 2 Reshapes and a Save.
  EXPECT_EQ(F_->getNodes().size(), 3);

  // Skip ConstantFolding as it would have the same result as this opt.
  cctx_.optimizationOpts.enableConstantFolding = false;
  ::glow::optimize(F_, cctx_);

  // After optimization, we expect to see just a Save.
  EXPECT_EQ(F_->getNodes().size(), 1);

  // Save should have the new Variable as input.
  auto *V = llvm::dyn_cast<Constant>(O->getInput());
  ASSERT_TRUE(V);
  // The new Variable should have the same shape as the original second
  // Reshape.
  EXPECT_TRUE(V->getType()->dims().equals(reshape2));
}

/// Test that reshape node is merged into Constant in a sequence
/// Reshape(Quantize(Constant)).
TEST_F(GraphOptz, ReshapeQuantizeConstant) {
  const dim_t shape[] = {10, 20};
  const dim_t newShape[] = {200, 1};

  auto *qTy = mod_.uniqueType(ElemKind::Int8QTy, shape, 0.2, 0);

  auto *input =
      F_->getParent()->createConstant(ElemKind::FloatTy, shape, "input");
  auto *Q = F_->createQuantize("quantize", input, qTy);
  auto *R = F_->createReshape("reshape", Q, newShape);
  auto *S = F_->createSave("ret", R);

  // Skip ConstantFolding as it would have the same result as this opt.
  CompilationContext cctx;
  cctx.optimizationOpts.enableConstantFolding = false;

  EXPECT_EQ(F_->getNodes().size(), 3);
  ::glow::optimize(F_, cctx);
  EXPECT_EQ(F_->getNodes().size(), 2);

  // Constant and Quantize should have new shape.
  auto *newQ = llvm::dyn_cast<QuantizeNode>(S->getInput());
  ASSERT_TRUE(newQ);
  EXPECT_TRUE(newQ->getResult().dims().equals(newShape));
  auto *newC = llvm::dyn_cast<Constant>(newQ->getInput());
  ASSERT_TRUE(newC);
  EXPECT_TRUE(newC->getType()->dims().equals(newShape));
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

TEST_F(GraphOptz, optimizeConversion_i32_i64_i32) {
  auto *i32 = mod_.uniqueType(ElemKind::Int32ITy, {1});
  auto *i64 = mod_.uniqueType(ElemKind::Int64ITy, {1});

  auto *A = mod_.createPlaceholder(i32, "A", false);
  auto *B = F_->createConvertTo("B", A, i64);
  auto *C = F_->createConvertTo("C", B, i32);
  auto *S = F_->createSave("S", C);

  ::glow::optimize(F_, CompilationMode::Infer);

  // All casting is optimized away, only left with Save of Placeholder.
  EXPECT_EQ(F_->getNodes().size(), 1);
  EXPECT_TRUE(llvm::isa<Placeholder>(S->getInput()));
}

TEST_F(GraphOptz, optimizeSameTypeConversions) {
  auto *input1 = mod_.createPlaceholder(ElemKind::FloatTy, {1}, "input1", true);
  auto *input2 = mod_.createPlaceholder(ElemKind::FloatTy, {1}, "input2", true);
  auto *conv1 = F_->createConvertTo("cast1", input1, ElemKind::FloatTy);
  auto *conv2 = F_->createConvertTo("cast2", input2, ElemKind::Float16Ty);
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

TEST_F(GraphOptz, optimizeConvertingBetweenFused) {
  // Call with dims {5, 2}, which will actually create a constant with {5, 10}
  // for scale/offset per row.
  Constant *C = createRandomFusedRowwiseQuantizedConstant(
      mod_, {5, 2}, "fused", /* useFusedFP16 */ false);
  // Converting to fused FP16 means we have 4 total less bytes for scale/offset,
  // so we move to {5, 10} from {5, 6}.
  auto newOT = mod_.uniqueType(ElemKind::UInt8FusedFP16QTy, {5, 6}, 1.0, 0);
  auto *CN = F_->createConvertTo("convert", C, newOT);
  auto *SN = F_->createSave("save", CN);

  ::glow::optimize(F_, CompilationMode::Infer);

  // Convert should be eliminated and just the save of the Constant left.
  EXPECT_EQ(F_->getNodes().size(), 1);
  Constant *convertedC = llvm::dyn_cast<Constant>(SN->getInput());
  ASSERT_TRUE(convertedC);
  EXPECT_EQ(convertedC->getElementType(), ElemKind::UInt8FusedFP16QTy);
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
  const dim_t inputDims[] = {3, 28, 28, 136};

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

/// Test BatchNorm sinking below Slice.
TEST_F(GraphOptz, sinkBatchNormBelowSlice) {
  auto *inputTy = mod_.uniqueType(ElemKind::FloatTy, {1, 10, 10, 3});
  auto *slicedTy1 = mod_.uniqueType(ElemKind::FloatTy, {1, 8, 8, 3});
  auto *slicedTy2 = mod_.uniqueType(ElemKind::FloatTy, {1, 6, 6, 1});

  auto *input = mod_.createPlaceholder(inputTy, "input", false);
  auto *BN = F_->createBatchNormalization(bindings_, "batchnorm", input, 3,
                                          0.0001, 0.9);
  auto *SN1 = F_->createSlice("slice1", BN, {0, 1, 1, 0}, slicedTy1);
  auto *SN2 = F_->createSlice("slice2", SN1, {0, 1, 1, 1}, slicedTy2);
  auto *save = F_->createSave("save", SN2);

  EXPECT_EQ(F_->getNodes().size(), 4);
  ::glow::convertPlaceholdersToConstants(F_, bindings_, {});
  optimizedF_ = optimizeFunctionForTest(F_);
  EXPECT_EQ(optimizedF_->getNodes().size(), 4);

  // BatchNorm should have sunk below the first Slice, but not the second one,
  // as it changes channel dimmension.
  auto *newSave =
      findFunctionNodeByName<SaveNode>(optimizedF_, save->getName());
  ASSERT_TRUE(newSave);
  auto *newSN2 = llvm::dyn_cast<SliceNode>(newSave->getInput());
  ASSERT_TRUE(newSN2);
  auto *newBN = llvm::dyn_cast<BatchNormalizationNode>(newSN2->getInput());
  ASSERT_TRUE(newBN);
  ASSERT_EQ(newBN->getResult().dims(), slicedTy1->dims());
  ASSERT_TRUE(llvm::isa<SliceNode>(newBN->getInput()));

  bindings_.allocate(mod_.getPlaceholders());
  bindings_.get(input)->getHandle().randomize(-1.0, 1.0, mod_.getPRNG());
  checkNumericalEquivalence();
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
  const dim_t dims[] = {2, 2, 2, 2};

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
      mod_.createPlaceholder(ElemKind::FloatTy, {6, 10, 4}, "lhs", false);
  auto *rhs = mod_.createConstant(ElemKind::FloatTy, {4, 8}, "rhs");
  rhs->getPayloadMutable().getHandle().randomize(-10, 10, mod_.getPRNG());
  auto *BMMN = F_->createBatchMatMul("BMM", lhs, rhs);
  F_->createSave("save", BMMN);

  // Start with a BatchMatMul, not a MatMul.
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::BatchMatMulNodeKind), 1);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::MatMulNodeKind), 0);

  optimizedF_ = optimizeFunctionForTest(F_);

  // Optimization should replace the BatchMatMul with a single MatMul.
  EXPECT_EQ(countNodeKind(optimizedF_, Kinded::Kind::MatMulNodeKind), 1);
  EXPECT_EQ(countNodeKind(optimizedF_, Kinded::Kind::BatchMatMulNodeKind), 0);

  bindings_.allocate(lhs)->getHandle().randomize(-10, 10, mod_.getPRNG());

  checkNumericalEquivalence(0.f);
}

/// Test Broadcasted RHS BatchMatMul is converted correctly to a single MatMul,
/// where RHS is broadcasted in multiple dimensions.
TEST_F(GraphOptz, convertMultiBroadcastedBatchMatMulToMatMul) {
  auto *lhs =
      mod_.createPlaceholder(ElemKind::FloatTy, {5, 10, 4}, "lhs", false);
  auto *rhs = mod_.createConstant(ElemKind::FloatTy, {1, 1, 6}, "rhs");
  rhs->getPayloadMutable().getHandle().randomize(-10, 10, mod_.getPRNG());
  auto *BN = F_->createBroadcast("broadcast", rhs, {5, 4, 6}, /* axis */ 0);
  auto *BMMN = F_->createBatchMatMul("BMM", lhs, BN);
  F_->createSave("save", BMMN);

  // Start with a BatchMatMul, not a MatMul, as well as a broadcast.
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::BatchMatMulNodeKind), 1);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::MatMulNodeKind), 0);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::BroadcastNodeKind), 1);

  optimizedF_ = optimizeFunctionForTest(
      F_, {FunctionPassID::ConvertBroadcastedBatchMatMul, getDCEPassConfig()});

  // Optimization should replace the BatchMatMul with a single MatMul, as well
  // as include a broadcast leftover.
  EXPECT_EQ(countNodeKind(optimizedF_, Kinded::Kind::MatMulNodeKind), 1);
  EXPECT_EQ(countNodeKind(optimizedF_, Kinded::Kind::BatchMatMulNodeKind), 0);
  EXPECT_EQ(countNodeKind(optimizedF_, Kinded::Kind::BroadcastNodeKind), 1);

  bindings_.allocate(lhs)->getHandle().randomize(-10, 10, mod_.getPRNG());

  checkNumericalEquivalence(0.f);
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

  optimizedF_ = optimizeFunctionForTest(F_);

  EXPECT_EQ(optimizedF_->getNodes().size(), 1);

  bindings_.allocate(mod_.getPlaceholders());
  bindings_.get(in)->getHandle<int8_t>().randomize(-4, 4, mod_.getPRNG());

  checkNumericalEquivalence();
}

template <typename ElemTy>
static void setConstValue(Constant *C, ElemTy value) {
  Handle<ElemTy> TH = C->getPayload().getHandle<ElemTy>();
  TH.clear(value);
}

TEST_F(GraphOptz, constantFoldSingleNode) {
  auto *const1 = mod_.createConstant(ElemKind::FloatTy, {2, 2}, "const1");
  auto *const2 = mod_.createConstant(ElemKind::FloatTy, {2, 2}, "const2");
  auto *ph1 = mod_.createPlaceholder(ElemKind::FloatTy, {2, 2}, "input1",
                                     /* isTrainable */ false);
  setConstValue(const1, 1.0f);
  setConstValue(const2, 2.0f);
  auto *splat2 = F_->createSplat(
      "splat2", mod_.uniqueType(ElemKind::FloatTy, {2, 2}), 2.0f);
  auto *splat3 = F_->createSplat(
      "splat3", mod_.uniqueType(ElemKind::FloatTy, {2, 2}), 3.0f);

  auto *add1 = F_->createAdd("add", const1, const2);
  auto *mul1 = F_->createMul("mul1", add1, splat2);
  auto *mul2 = F_->createMul("mul2", mul1, splat3);
  auto *SN1 = F_->createSave("save", mul2);
  auto *add3 = F_->createAdd("add", const1, ph1);
  auto *SN2 = F_->createSave("save", add3);

  // Perform constant folding for a specific node.
  std::vector<Constant *> constResults =
      constantFold(SN1->getInput().getNode());

  ASSERT_EQ(constResults.size(), 1);
  SN1->getInput().replaceAllUsesOfWith(constResults[0]);
  // Second save should be unaffected.
  EXPECT_FALSE(llvm::isa<Constant>(SN2->getInput()));
  // First save should have been constant folded.
  EXPECT_TRUE(llvm::isa<Constant>(SN1->getInput()));
  Constant *C = llvm::dyn_cast<Constant>(SN1->getInput());
  auto CH = C->getHandle();
  // The expected result should be: (((1+2) * 2 * 3) = 18
  EXPECT_EQ(CH.at({0, 0}), 18.0f);
  EXPECT_EQ(CH.at({0, 1}), 18.0f);
  EXPECT_EQ(CH.at({1, 0}), 18.0f);
  EXPECT_EQ(CH.at({1, 1}), 18.0f);
}

/// Verify that we can specify what splats should be materialized to constants
/// based on their users via optimizationOpts.materializeSplatsUsedBySet.
TEST_F(GraphOptz, constantFoldSpecificSplat) {
  Placeholder *PH = mod_.createPlaceholder(ElemKind::FloatTy, {1, 1}, "input",
                                           /* isTrainable */ false);
  SplatNode *splat1 = F_->createSplat(
      "splat1", mod_.uniqueType(ElemKind::FloatTy, {1, 1}), 1.0f);
  AddNode *add = F_->createAdd("add", PH, splat1);
  SplatNode *splat2 = F_->createSplat(
      "splat2", mod_.uniqueType(ElemKind::FloatTy, {1, 1}), 2.0f);
  MulNode *mul = F_->createMul("mul", add, splat2);
  SaveNode *save = F_->createSave("save", mul);

  // Signal to materialize the splat used by Add, but not by Mul.
  cctx_.optimizationOpts.materializeSplatsUsedBySet.insert(
      Kinded::Kind::AddNodeKind);

  optimizedF_ = F_->clone(F_->getName().str() + "_optimized");

  ConstantFoldingRecordMap record = constantFoldAndRecord(optimizedF_, cctx_);
  runDCEPass(optimizedF_, cctx_);

  ASSERT_EQ(record.size(), 1);
  SaveNode *SN = record.begin()->second;
  SplatNode *foldSplat1 = llvm::dyn_cast<SplatNode>(SN->getInput());
  ASSERT_TRUE(foldSplat1);
  EXPECT_EQ(foldSplat1->getValue(), 1.0f);

  // Verify one splat left in the optimized Function, and a new Constant.
  EXPECT_EQ(1, countNodeKind(optimizedF_, Kinded::Kind::SplatNodeKind));
  const SaveNode *optSave =
      findFunctionNodeByName<SaveNode>(optimizedF_, save->getName());
  MulNode *optMul = llvm::dyn_cast<MulNode>(optSave->getInput());
  ASSERT_TRUE(optMul);
  SplatNode *optSplat2 = llvm::dyn_cast<SplatNode>(optMul->getRHS());
  ASSERT_TRUE(optSplat2);
  EXPECT_EQ(optSplat2->getValue(), 2.0f);
  AddNode *optAdd = llvm::dyn_cast<AddNode>(optMul->getLHS());
  ASSERT_TRUE(optAdd);
  EXPECT_EQ(optAdd->getLHS().getNode(), PH);
  Constant *optSplatConst1 = llvm::dyn_cast<Constant>(optAdd->getRHS());
  ASSERT_TRUE(optSplatConst1);
  EXPECT_EQ(optSplatConst1->getPayload().getHandle().at({0, 0}), 1.0f);
}

/// Test that we correctly record a single constant folding subgraph that has a
/// single output.
TEST_F(GraphOptz, constantFoldWithRecordSingleChain) {
  Placeholder *I =
      mod_.createPlaceholder(ElemKind::Float16Ty, {2, 100}, "input",
                             /* isTrainable */ false);
  Constant *W = mod_.createConstant(ElemKind::FloatTy, {10, 100}, "weight");
  ClipNode *clipW = F_->createClip("clip", W, -5.f, 5.f);
  ConvertToNode *convertW =
      F_->createConvertTo("conv", clipW, ElemKind::Float16Ty);
  TransposeNode *transposeW =
      F_->createTranspose("transpose", convertW, {1, 0});
  MatMulNode *MM = F_->createMatMul("matmul", I, transposeW);
  SaveNode *save = F_->createSave("save", MM);
  Placeholder *O = save->getPlaceholder();
  bindings_.allocate(O);

  ASSERT_TRUE(F_->verify());

  Tensor *IT = bindings_.allocate(I);
  IT->getHandle<float16_t>().randomize(-10, 10, mod_.getPRNG());
  W->getPayloadMutable().getHandle<float>().randomize(-10, 10, mod_.getPRNG());

  optimizedF_ = F_->clone(F_->getName().str() + "_optimized");

  ConstantFoldingRecordMap record = constantFoldAndRecord(optimizedF_, cctx_);

  runDCEPass(optimizedF_, cctx_);

  ASSERT_EQ(record.size(), 1);
  SaveNode *SN = record.begin()->second;
  Function *constFoldF = SN->getParent();

  // Expect to find a chain of Nodes based on Nodes above. Note that the clip is
  // lowered for the Interpreter backend which performs constant folding.
  EXPECT_EQ(2, countNodeKind(constFoldF, Kinded::Kind::SplatNodeKind));
  EXPECT_EQ(1, countNodeKind(constFoldF, Kinded::Kind::MaxNodeKind));
  EXPECT_EQ(1, countNodeKind(constFoldF, Kinded::Kind::MinNodeKind));
  EXPECT_EQ(1, countNodeKind(constFoldF, Kinded::Kind::ConvertToNodeKind));
  EXPECT_EQ(1, countNodeKind(constFoldF, Kinded::Kind::TransposeNodeKind));

  // Skip optimizations -- we just want to run them as is (otherwise we'll
  // constant fold them inside the optimization pipeline).
  cctx_.optimizationOpts.onlyLowerFuns.insert(constFoldF);
  cctx_.optimizationOpts.onlyLowerFuns.insert(F_);
  cctx_.optimizationOpts.onlyLowerFuns.insert(optimizedF_);

  // Don't strip the module as we want to compare the Constant values below.
  EE_.setSkipModuleStrip(true);

  EE_.compile(cctx_);
  alreadyCompiled_ = true;

  bindings_.allocate(mod_.getPlaceholders());

  // Run the constant folding chain to check that we have the same constant used
  // by the optimized Function.
  EE_.run(bindings_, constFoldF->getName());
  Tensor *rerunT = bindings_.get(SN->getPlaceholder());
  ASSERT_TRUE(rerunT);
  auto optimizedConstants = optimizedF_->findConstants();
  ASSERT_EQ(optimizedConstants.size(), 1);
  EXPECT_TRUE(
      (*optimizedConstants.begin())->getPayload().isEqual(*rerunT, 0.f));

  // Remove the temporary constant folding Functions and their Placeholders.
  cleanupConstantFolding(mod_, record, &bindings_);

  // Now compile/run/compare F_ and optimizedF_.
  checkNumericalEquivalence(0.f);
}

/// Test that we correctly record two constant folding subgraphs, with each with
/// a single output.
TEST_F(GraphOptz, constantFoldWithRecordMultiChain) {
  Placeholder *I =
      mod_.createPlaceholder(ElemKind::Float16Ty, {2, 100}, "input",
                             /* isTrainable */ false);
  Constant *W = mod_.createConstant(ElemKind::FloatTy, {10, 100}, "weight");
  ClipNode *clipW = F_->createClip("clip", W, -5.f, 5.f);
  ConvertToNode *convertW =
      F_->createConvertTo("conv", clipW, ElemKind::Float16Ty);
  TransposeNode *transposeW =
      F_->createTranspose("transpose", convertW, {1, 0});
  MatMulNode *MM = F_->createMatMul("matmul", I, transposeW);
  SaveNode *saveMM = F_->createSave("save_mm", MM);
  Placeholder *MMP = saveMM->getPlaceholder();
  bindings_.allocate(MMP);

  SigmoidNode *sigmoidW = F_->createSigmoid("sig", convertW);
  SaveNode *saveSig = F_->createSave("save_sig", sigmoidW);
  Placeholder *sigP = saveSig->getPlaceholder();
  bindings_.allocate(sigP);

  ASSERT_TRUE(F_->verify());

  Tensor *IT = bindings_.allocate(I);
  IT->getHandle<float16_t>().randomize(-10, 10, mod_.getPRNG());
  W->getPayloadMutable().getHandle<float>().randomize(-10, 10, mod_.getPRNG());

  optimizedF_ = F_->clone(F_->getName().str() + "_optimized");

  ConstantFoldingRecordMap record = constantFoldAndRecord(optimizedF_, cctx_);

  runDCEPass(optimizedF_, cctx_);

  ASSERT_EQ(record.size(), 2);
  SaveNode *sigSN = record.begin()->second;
  SaveNode *transSN = std::next(record.begin())->second;
  if (llvm::isa<SigmoidNode>(transSN->getInput())) {
    std::swap(sigSN, transSN);
  }

  Function *constFoldSig = sigSN->getParent();
  Function *constFoldTrans = transSN->getParent();

  // Expect to find a chain of Nodes based on Nodes above. Note that the clip is
  // lowered for the Interpreter backend which performs constant folding.
  EXPECT_EQ(2, countNodeKind(constFoldTrans, Kinded::Kind::SplatNodeKind));
  EXPECT_EQ(1, countNodeKind(constFoldTrans, Kinded::Kind::MaxNodeKind));
  EXPECT_EQ(1, countNodeKind(constFoldTrans, Kinded::Kind::MinNodeKind));
  EXPECT_EQ(1, countNodeKind(constFoldTrans, Kinded::Kind::ConvertToNodeKind));
  EXPECT_EQ(1, countNodeKind(constFoldTrans, Kinded::Kind::TransposeNodeKind));

  EXPECT_EQ(2, countNodeKind(constFoldSig, Kinded::Kind::SplatNodeKind));
  EXPECT_EQ(1, countNodeKind(constFoldSig, Kinded::Kind::MaxNodeKind));
  EXPECT_EQ(1, countNodeKind(constFoldSig, Kinded::Kind::MinNodeKind));
  EXPECT_EQ(1, countNodeKind(constFoldSig, Kinded::Kind::ConvertToNodeKind));
  EXPECT_EQ(1, countNodeKind(constFoldSig, Kinded::Kind::SigmoidNodeKind));

  // Skip optimizations -- we just want to run them as is (otherwise we'll
  // constant fold them inside the optimization pipeline).
  cctx_.optimizationOpts.onlyLowerFuns.insert(constFoldTrans);
  cctx_.optimizationOpts.onlyLowerFuns.insert(constFoldSig);
  cctx_.optimizationOpts.onlyLowerFuns.insert(F_);
  cctx_.optimizationOpts.onlyLowerFuns.insert(optimizedF_);

  // Don't strip the module as we want to compare the Constant values below.
  EE_.setSkipModuleStrip(true);

  EE_.compile(cctx_);
  alreadyCompiled_ = true;

  bindings_.allocate(mod_.getPlaceholders());

  // Run the constant folding chain to check that we have the same constant used
  // by the optimized Function.
  EE_.run(bindings_, constFoldTrans->getName());
  EE_.run(bindings_, constFoldSig->getName());

  // Find the correct PHs for each of the constant folding we do.
  Tensor *rerunTransT = bindings_.get(transSN->getPlaceholder());
  Tensor *rerunSigT = bindings_.get(sigSN->getPlaceholder());
  ASSERT_TRUE(rerunTransT);
  ASSERT_TRUE(rerunSigT);

  auto optimizedConstants = optimizedF_->findConstants();
  ASSERT_EQ(optimizedConstants.size(), 2);
  Constant *transC = *optimizedConstants.begin();
  Constant *sigC = *std::next(optimizedConstants.begin());
  // If we have the constants backwards then swap them. Note that we know
  // sigC must be directly saved, while transC is input to a MatMulNode.
  ASSERT_EQ(transC->getNumUsers(), 1);
  if (llvm::isa<SaveNode>(transC->getUsers().begin()->getUser())) {
    std::swap(transC, sigC);
  }
  EXPECT_TRUE(transC->getPayload().isEqual(*rerunTransT, 0.f));
  EXPECT_TRUE(sigC->getPayload().isEqual(*rerunSigT, 0.f));

  // Remove the temporary constant folding Functions and their Placeholders.
  cleanupConstantFolding(mod_, record, &bindings_);

  // Now compile/run/compare F_ and optimizedF_.
  checkNumericalEquivalence(0.f);
}

/// Test that we correctly record a single constant folding subgraph that has
/// two outputs.
TEST_F(GraphOptz, constantFoldWithRecordSingleChainMultiOutput) {
  Constant *W = mod_.createConstant(ElemKind::FloatTy, {100}, "weight");
  SigmoidNode *sigmoidW = F_->createSigmoid("sig", W);
  ConvertToNode *convertW =
      F_->createConvertTo("conv", sigmoidW, ElemKind::Float16Ty);
  TopKNode *TK = F_->createTopK("topk", convertW, 5);

  SaveNode *indicesSave = F_->createSave("save_indices", TK->getIndices());
  Placeholder *indicesP = indicesSave->getPlaceholder();
  bindings_.allocate(indicesP);

  Placeholder *I = mod_.createPlaceholder(ElemKind::Float16Ty, {5}, "input",
                                          /* isTrainable */ false);
  AddNode *add = F_->createAdd("add", I, TK->getValues());
  SaveNode *addSave = F_->createSave("save_add", add);
  Placeholder *addP = addSave->getPlaceholder();
  bindings_.allocate(addP);

  ASSERT_TRUE(F_->verify());

  Tensor *IT = bindings_.allocate(I);
  IT->getHandle<float16_t>().randomize(-10, 10, mod_.getPRNG());
  W->getPayloadMutable().getHandle<float>().randomize(-10, 10, mod_.getPRNG());

  optimizedF_ = F_->clone(F_->getName().str() + "_optimized");

  ConstantFoldingRecordMap record = constantFoldAndRecord(optimizedF_, cctx_);

  runDCEPass(optimizedF_, cctx_);

  ASSERT_EQ(record.size(), 2);
  SaveNode *indicesSN = record.begin()->second;
  SaveNode *addSN = std::next(record.begin())->second;

  // Find the correct PHs for each of the constant folding we do.
  if (indicesSN->getInput().getResNo() != TopKNode::IndicesIdx) {
    std::swap(indicesSN, addSN);
  }

  // Expect that the two constants that we folded are from the same Function,
  // and that the two saves use the two different outputs from a topk.
  EXPECT_EQ(indicesSN->getParent(), addSN->getParent());
  ASSERT_TRUE(llvm::isa<TopKNode>(addSN->getInput()));
  ASSERT_TRUE(llvm::isa<TopKNode>(indicesSN->getInput()));
  EXPECT_EQ(addSN->getInput().getNode(), indicesSN->getInput().getNode());

  Function *constFoldF = addSN->getParent();

  // Expect to find a chain of Nodes based on Nodes above.
  EXPECT_EQ(1, countNodeKind(constFoldF, Kinded::Kind::TopKNodeKind));
  EXPECT_EQ(1, countNodeKind(constFoldF, Kinded::Kind::SigmoidNodeKind));
  EXPECT_EQ(1, countNodeKind(constFoldF, Kinded::Kind::ConvertToNodeKind));

  // Skip optimizations -- we just want to run them as is (otherwise we'll
  // constant fold them inside the optimization pipeline).
  cctx_.optimizationOpts.onlyLowerFuns.insert(constFoldF);
  cctx_.optimizationOpts.onlyLowerFuns.insert(F_);
  cctx_.optimizationOpts.onlyLowerFuns.insert(optimizedF_);

  // Don't strip the module as we want to compare the Constant values below.
  EE_.setSkipModuleStrip(true);

  EE_.compile(cctx_);
  alreadyCompiled_ = true;

  bindings_.allocate(mod_.getPlaceholders());

  // Run the constant folding chain to check that we have the same constant used
  // by the optimized Function.
  EE_.run(bindings_, constFoldF->getName());

  Tensor *rerunAddT = bindings_.get(addSN->getPlaceholder());
  Tensor *rerunIndicesT = bindings_.get(indicesSN->getPlaceholder());
  ASSERT_TRUE(rerunAddT);
  ASSERT_TRUE(rerunIndicesT);

  auto optimizedConstants = optimizedF_->findConstants();
  ASSERT_EQ(optimizedConstants.size(), 2);
  Constant *addC = *optimizedConstants.begin();
  Constant *indicesC = *std::next(optimizedConstants.begin());

  // If we have the constants backwards then swap them. Note that we know
  // indicesC must be directly saved, while addC is input to an AddNode.
  ASSERT_EQ(addC->getNumUsers(), 1);
  if (llvm::isa<SaveNode>(addC->getUsers().begin()->getUser())) {
    std::swap(addC, indicesC);
  }
  EXPECT_TRUE(addC->getPayload().isEqual(*rerunAddT, 0.f));
  EXPECT_TRUE(indicesC->getPayload().isEqual(*rerunIndicesT, 0.f));

  // Remove the temporary constant folding Functions and their Placeholders.
  cleanupConstantFolding(mod_, record, &bindings_);

  // Now compile/run/compare F_ and optimizedF_.
  checkNumericalEquivalence(0.f);
}

/// Test that the constant folding record Function includes all ops,
/// i.e. they're not optimized away during optimizations when the constant
/// folding function is optimized.
TEST_F(GraphOptz, constantFoldOnlyLower) {
  Constant *W = mod_.createConstant(ElemKind::FloatTy, {10, 100}, "weight");
  ConvertToNode *convertW = F_->createConvertTo("conv", W, ElemKind::Float16Ty);
  SaveNode *save = F_->createSave("save", convertW);
  Placeholder *O = save->getPlaceholder();
  bindings_.allocate(O);

  ASSERT_TRUE(F_->verify());

  W->getPayloadMutable().getHandle<float>().randomize(-10, 10, mod_.getPRNG());

  optimizedF_ = F_->clone(F_->getName().str() + "_optimized");

  ConstantFoldingRecordMap record = constantFoldAndRecord(optimizedF_, cctx_);

  ASSERT_EQ(record.size(), 1);
  SaveNode *SN = record.begin()->second;
  Function *constFoldF = SN->getParent();

  // Expect to find a Save and the ConvertTo still, i.e. it shouldn't have been
  // folded into the Constant as part of the OptimizeConversions pass.
  EXPECT_EQ(2, constFoldF->getNodes().size());
  EXPECT_EQ(1, countNodeKind(constFoldF, Kinded::Kind::ConvertToNodeKind));
  EXPECT_EQ(1, countNodeKind(constFoldF, Kinded::Kind::SaveNodeKind));
}

TEST_F(GraphOptz, constantFoldWholeFunction) {
  auto *const1 = mod_.createConstant(ElemKind::FloatTy, {2, 2}, "const1");
  auto *const2 = mod_.createConstant(ElemKind::FloatTy, {2, 2}, "const2");
  auto *const3 = mod_.createConstant(ElemKind::FloatTy, {2, 2}, "const3");
  auto *const4 = mod_.createConstant(ElemKind::FloatTy, {2, 2}, "const4");
  auto *ph1 = mod_.createPlaceholder(ElemKind::FloatTy, {2, 2}, "input1",
                                     /* isTrainable */ false);
  setConstValue(const1, 1.0f);
  setConstValue(const2, 2.0f);
  setConstValue(const3, 3.0f);
  setConstValue(const4, 4.0f);
  auto *splat2 = F_->createSplat(
      "splat2", mod_.uniqueType(ElemKind::FloatTy, {2, 2}), 2.0f);
  auto *splat3 = F_->createSplat(
      "splat2", mod_.uniqueType(ElemKind::FloatTy, {2, 2}), 3.0f);
  auto *splat4 = F_->createSplat(
      "splat2", mod_.uniqueType(ElemKind::FloatTy, {2, 2}), 4.0f);

  auto *add1 = F_->createAdd("add", const1, const2);
  auto *mul1 = F_->createMul("mul1", add1, splat2);
  auto *mul2 = F_->createMul("mul2", mul1, splat3);
  auto *sub = F_->createSub("sub", mul2, const3);
  auto *add2 = F_->createAdd("add2", sub, const4);
  auto *mul3 = F_->createMul("mul3", add2, splat4);
  // Check compile-time constant folding for nodes with multiple results.
  auto *topK = F_->createTopK("topK", mul3, 2);
  auto *SN1_0 = F_->createSave("save", topK->getValues());
  auto *SN1_1 = F_->createSave("save", topK->getIndices());
  auto *add3 = F_->createAdd("add", const1, ph1);
  auto *SN2 = F_->createSave("save", add3);

  // Perform constant folding for a whole function.
  ::glow::optimize(F_, CompilationMode::Infer);

  EXPECT_EQ(F_->getNodes().size(), 4);
  // Second save should be unaffected, as its value is not a constant operation.
  EXPECT_FALSE(llvm::isa<Constant>(SN2->getInput()));
  // First save should have been constant folded.
  EXPECT_TRUE(llvm::isa<Constant>(SN1_0->getInput()));
  EXPECT_TRUE(llvm::isa<Constant>(SN1_1->getInput()));
  Constant *C = llvm::dyn_cast<Constant>(SN1_0->getInput());
  auto CH = C->getHandle();
  // The expected result should be: (((1+2) * 2 * 3 - 3) + 4) * 4 = 76
  EXPECT_EQ(CH.at({0, 0}), 76.0f);
  EXPECT_EQ(CH.at({0, 1}), 76.0f);
  EXPECT_EQ(CH.at({1, 0}), 76.0f);
  EXPECT_EQ(CH.at({1, 1}), 76.0f);
}

/// Test constant folding for operators which are lowered in Interpreter
/// backend.
TEST_F(GraphOptz, constantFoldWithLowering) {
  auto *input = mod_.createConstant(ElemKind::FloatTy, {1, 6}, "input");
  input->getHandle() = {5, 4, 3, 2, 1, 0};
  auto *TN = F_->createTile("tile", input, 5, 0);
  auto *SN = F_->createSave("ret", TN);

  // Perform constant folding.
  EXPECT_EQ(F_->getNodes().size(), 2);
  ::glow::optimize(F_, CompilationMode::Infer);

  // Tile with its input should be folded into a single Constant node.
  EXPECT_EQ(F_->getNodes().size(), 1);
  ASSERT_TRUE(llvm::isa<Constant>(SN->getInput()));
}

/// Test Splitting FC into multiple FCs.
TEST_F(GraphOptz, SplitFCIntoMultipleOps) {
  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 32}, "input", false);
  bindings_.allocate(input)->getHandle<float>().randomize(-1.0, 1.0,
                                                          mod_.getPRNG());
  auto *weights = mod_.createConstant(ElemKind::FloatTy, {32, 850}, "weights");
  weights->getHandle().randomize(-1.0, 1.0, mod_.getPRNG());
  auto *bias = mod_.createConstant(ElemKind::FloatTy, {850}, "bias");
  bias->getHandle().randomize(0.0, 0.5, mod_.getPRNG());
  auto *output =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 850}, "output", false);
  bindings_.allocate(output);

  auto *fc = F_->createFullyConnected("fc", input, weights, bias);
  auto *save = F_->createSave("save", fc, output);

  ::glow::optimize(F_, CompilationMode::Infer);

  // This is F_ but without the parallel transformation below.
  optimizedF_ = F_->clone(F_->getName().str() + "_optimized");

  EXPECT_TRUE(::glow::executeVerticalFCWeightsSplit(F_,
                                                    /*numOfChunks*/ 12,
                                                    /*minKToSplit*/ 800));
  runDCEPass(F_, cctx_);

  // 24 Slices: 12 from bias and 12 from weights.
  EXPECT_EQ(24, countNodeKind(F_, Kinded::Kind::SliceNodeKind));

  EXPECT_EQ(1, countNodeKind(F_, Kinded::Kind::ConcatNodeKind));

  // 12 newly created FCs.
  EXPECT_EQ(12, countNodeKind(F_, Kinded::Kind::FullyConnectedNodeKind));

  auto *concatNode = llvm::dyn_cast<ConcatNode>(save->getInput());
  ASSERT_TRUE(concatNode);
  // 12 FCs are connected to the concat node.
  EXPECT_EQ(12, concatNode->getInputs().size());

  // Check all splitted FCs.
  for (unsigned i = 0; i < 12; ++i) {
    auto *fc = llvm::dyn_cast<FullyConnectedNode>(concatNode->getNthInput(i));
    ASSERT_TRUE(fc);
    // 2 * 71 for first 11 FCs and last 2 * 69
    if (i == 11) {
      EXPECT_TRUE(fc->getResult().dims().equals({2, 69}));
      EXPECT_TRUE(fc->getBias().dims().equals({69}));
      EXPECT_TRUE(fc->getWeights().dims().equals({32, 69}));
    } else {
      EXPECT_TRUE(fc->getResult().dims().equals({2, 71}));
      EXPECT_TRUE(fc->getBias().dims().equals({71}));
      EXPECT_TRUE(fc->getWeights().dims().equals({32, 71}));
    }
  }

  checkNumericalEquivalence();
}

/// Test Splitting FC into multiple FCs.
TEST_F(GraphOptz, ParallelizeGraph_FC_ModelParallel) {
  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {8, 3}, "input", false);
  bindings_.allocate(input)->getHandle<float>().randomize(-1.0, 1.0,
                                                          mod_.getPRNG());
  auto *weights1 = mod_.createConstant(ElemKind::FloatTy, {3, 150}, "weights");
  weights1->getHandle().randomize(-1.0, 1.0, mod_.getPRNG());
  auto *bias1 = mod_.createConstant(ElemKind::FloatTy, {150}, "bias");
  bias1->getHandle().randomize(0.0, 0.5, mod_.getPRNG());
  auto *weights2 =
      mod_.createConstant(ElemKind::FloatTy, {150, 150}, "weights");
  weights2->getHandle().randomize(-1.0, 1.0, mod_.getPRNG());
  auto *bias2 = mod_.createConstant(ElemKind::FloatTy, {150}, "bias");
  bias2->getHandle().randomize(0.0, 0.5, mod_.getPRNG());
  auto *output =
      mod_.createPlaceholder(ElemKind::FloatTy, {8, 150}, "output", false);
  bindings_.allocate(output);

  auto *fc1 = F_->createFullyConnected("fc1", input, weights1, bias1);
  auto *relu1 = F_->createRELU("relu1", fc1);

  auto *fc2 = F_->createFullyConnected("fc2", relu1, weights2, bias2);
  auto *relu2 = F_->createRELU("relu2", fc2);
  F_->createSave("save", relu2, output);

  ::glow::optimize(F_, CompilationMode::Infer);

  // This is F_ but without the parallel transformation below.
  optimizedF_ = F_->clone(F_->getName().str() + "_optimized");

  // Perform parallel transformation on F_.
  llvm::DenseMap<Node *, size_t> numChunks;
  llvm::DenseMap<Node *, ParallelTransformKind> parOpts;
  numChunks[fc1] = 2;
  numChunks[relu1] = 2;
  numChunks[fc2] = 2;
  numChunks[relu2] = 2;
  parOpts[fc1] = ParallelTransformKind::Model;
  parOpts[relu1] = ParallelTransformKind::Model;
  parOpts[fc2] = ParallelTransformKind::Model;
  parOpts[relu2] = ParallelTransformKind::Model;
  std::unordered_map<Node *, ConcatNode *> replacedMap;
  ASSIGN_VALUE_OR_FAIL_TEST(replacedMap,
                            ::glow::parallelizeOps(F_, numChunks, parOpts));
  EXPECT_EQ(replacedMap.size(), parOpts.size());

  runDCEPass(F_, cctx_);

  EXPECT_EQ(4, countNodeKind(F_, Kinded::Kind::FullyConnectedNodeKind));
  EXPECT_EQ(4, countNodeKind(F_, Kinded::Kind::ReluNodeKind));

  checkNumericalEquivalence();
}

/// Test Splitting FC into multiple FCs, special case for 866 by 8 with an
/// alignment of 64, which is a corner case for alignment and should only
/// produce 7 splits
TEST_F(GraphOptz, ParallelizeGraph_FC_ModelParallel_Split866by8) {
  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {8, 3}, "input", false);
  bindings_.allocate(input)->getHandle<float>().randomize(-1.0, 1.0,
                                                          mod_.getPRNG());
  auto *weights1 = mod_.createConstant(ElemKind::FloatTy, {3, 866}, "weights");
  weights1->getHandle().randomize(-1.0, 1.0, mod_.getPRNG());
  auto *bias1 = mod_.createConstant(ElemKind::FloatTy, {866}, "bias");
  bias1->getHandle().randomize(0.0, 0.5, mod_.getPRNG());
  auto *output =
      mod_.createPlaceholder(ElemKind::FloatTy, {8, 866}, "output", false);
  bindings_.allocate(output);

  auto *fc1 = F_->createFullyConnected("fc1", input, weights1, bias1);
  auto *relu1 = F_->createRELU("relu1", fc1);

  F_->createSave("save", relu1, output);

  ::glow::optimize(F_, CompilationMode::Infer);

  // This is F_ but without the parallel transformation below.
  optimizedF_ = F_->clone(F_->getName().str() + "_optimized");

  // Perform parallel transformation on F_.
  llvm::DenseMap<Node *, size_t> numChunks;
  llvm::DenseMap<Node *, ParallelTransformKind> parOpts;
  numChunks[fc1] = 8;
  numChunks[relu1] = 8;
  parOpts[fc1] = ParallelTransformKind::Model;
  parOpts[relu1] = ParallelTransformKind::Model;
  std::unordered_map<Node *, ConcatNode *> replacedMap;
  ASSIGN_VALUE_OR_FAIL_TEST(
      replacedMap,
      ::glow::parallelizeOps(F_, numChunks, parOpts, /*numOfChunks*/ 1,
                             /*modelParallelSplitAlignment*/ 64));
  EXPECT_EQ(replacedMap.size(), parOpts.size());

  runDCEPass(F_, cctx_);

  EXPECT_EQ(7, countNodeKind(F_, Kinded::Kind::FullyConnectedNodeKind));
  EXPECT_EQ(7, countNodeKind(F_, Kinded::Kind::ReluNodeKind));

  // Check all splitted FCs.
  auto *concatNode = replacedMap[fc1];
  for (unsigned i = 0; i < 7; ++i) {
    auto *fc = llvm::dyn_cast<FullyConnectedNode>(concatNode->getNthInput(i));
    ASSERT_TRUE(fc);
    // 8 x 128 for first 6 FCs and last 8 x 30
    if (i == 6) {
      EXPECT_TRUE(fc->getResult().dims().equals({8, 98}));
      EXPECT_TRUE(fc->getBias().dims().equals({98}));
      EXPECT_TRUE(fc->getWeights().dims().equals({3, 98}));
    } else {
      EXPECT_TRUE(fc->getResult().dims().equals({8, 128}));
      EXPECT_TRUE(fc->getBias().dims().equals({128}));
      EXPECT_TRUE(fc->getWeights().dims().equals({3, 128}));
    }
  }

  checkNumericalEquivalence();
}

/// Test Splitting FC into multiple FCs, special case for 140 by 3 with an
/// alignment of 64. Should split 64, 64, 12
TEST_F(GraphOptz, ParallelizeGraph_FC_ModelParallel_Split140by3) {
  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {8, 3}, "input", false);
  bindings_.allocate(input)->getHandle<float>().randomize(-1.0, 1.0,
                                                          mod_.getPRNG());
  auto *weights1 = mod_.createConstant(ElemKind::FloatTy, {3, 140}, "weights");
  weights1->getHandle().randomize(-1.0, 1.0, mod_.getPRNG());
  auto *bias1 = mod_.createConstant(ElemKind::FloatTy, {140}, "bias");
  bias1->getHandle().randomize(0.0, 0.5, mod_.getPRNG());
  auto *output =
      mod_.createPlaceholder(ElemKind::FloatTy, {8, 140}, "output", false);
  bindings_.allocate(output);

  auto *fc1 = F_->createFullyConnected("fc1", input, weights1, bias1);
  auto *relu1 = F_->createRELU("relu1", fc1);

  F_->createSave("save", relu1, output);

  ::glow::optimize(F_, CompilationMode::Infer);

  // This is F_ but without the parallel transformation below.
  optimizedF_ = F_->clone(F_->getName().str() + "_optimized");

  // Perform parallel transformation on F_.
  llvm::DenseMap<Node *, size_t> numChunks;
  llvm::DenseMap<Node *, ParallelTransformKind> parOpts;
  numChunks[fc1] = 3;
  parOpts[fc1] = ParallelTransformKind::Model;
  std::unordered_map<Node *, ConcatNode *> replacedMap;
  ASSIGN_VALUE_OR_FAIL_TEST(
      replacedMap,
      ::glow::parallelizeOps(F_, numChunks, parOpts, /*numOfChunks*/ 1,
                             /*modelParallelSplitAlignment*/ 64));
  EXPECT_EQ(replacedMap.size(), parOpts.size());
  runDCEPass(F_, cctx_);
  EXPECT_EQ(3, countNodeKind(F_, Kinded::Kind::FullyConnectedNodeKind));

  // Check all splitted FCs.
  auto *concatNode = replacedMap[fc1];
  auto *fc_split0 =
      llvm::dyn_cast<FullyConnectedNode>(concatNode->getNthInput(0));
  auto *fc_split1 =
      llvm::dyn_cast<FullyConnectedNode>(concatNode->getNthInput(1));
  auto *fc_split2 =
      llvm::dyn_cast<FullyConnectedNode>(concatNode->getNthInput(2));
  ASSERT_TRUE(fc_split0);
  ASSERT_TRUE(fc_split1);
  ASSERT_TRUE(fc_split2);
  EXPECT_TRUE(fc_split0->getResult().dims().equals({8, 64}));
  EXPECT_TRUE(fc_split0->getBias().dims().equals({64}));
  EXPECT_TRUE(fc_split0->getWeights().dims().equals({3, 64}));
  EXPECT_TRUE(fc_split1->getResult().dims().equals({8, 64}));
  EXPECT_TRUE(fc_split1->getBias().dims().equals({64}));
  EXPECT_TRUE(fc_split1->getWeights().dims().equals({3, 64}));
  EXPECT_TRUE(fc_split2->getResult().dims().equals({8, 12}));
  EXPECT_TRUE(fc_split2->getBias().dims().equals({12}));
  EXPECT_TRUE(fc_split2->getWeights().dims().equals({3, 12}));

  checkNumericalEquivalence();
}

/// Test Splitting MatMul into multiple MatMuls
TEST_F(GraphOptz, SplitMatMulIntoMultipleOps_Data) {
  auto *input1 =
      mod_.createPlaceholder(ElemKind::FloatTy, {12, 32}, "input1", false);
  bindings_.allocate(input1)->getHandle<float>().randomize(-1.0, 1.0,
                                                           mod_.getPRNG());
  auto *input2 =
      mod_.createPlaceholder(ElemKind::FloatTy, {32, 32}, "input2", false);
  bindings_.allocate(input2)->getHandle<float>().randomize(-1.0, 1.0,
                                                           mod_.getPRNG());
  auto *output =
      mod_.createPlaceholder(ElemKind::FloatTy, {12, 32}, "output", false);
  bindings_.allocate(output);

  auto *mm = F_->createMatMul("mm", input1, input2);
  auto *save = F_->createSave("save", mm, output);

  ::glow::optimize(F_, CompilationMode::Infer);

  // This is F_ but without the parallel transformation below.
  optimizedF_ = F_->clone(F_->getName().str() + "_optimized");

  llvm::DenseMap<Node *, ParallelTransformKind> parOpts;
  parOpts[mm] = ParallelTransformKind::Data;

  std::unordered_map<Node *, ConcatNode *> replacedMap;
  ASSIGN_VALUE_OR_FAIL_TEST(
      replacedMap, ::glow::parallelizeOps(F_, llvm::DenseMap<Node *, size_t>(),
                                          parOpts, 12));
  EXPECT_EQ(replacedMap.size(), parOpts.size());
  runDCEPass(F_, cctx_);

  // 12 Slices from LHS
  EXPECT_EQ(12, countNodeKind(F_, Kinded::Kind::SliceNodeKind));

  EXPECT_EQ(1, countNodeKind(F_, Kinded::Kind::ConcatNodeKind));

  // 12 newly created MatMuls.
  EXPECT_EQ(12, countNodeKind(F_, Kinded::Kind::MatMulNodeKind));

  auto *concatNode = llvm::dyn_cast<ConcatNode>(save->getInput());
  ASSERT_TRUE(concatNode);
  // 12 FCs are connected to the concat node.
  EXPECT_EQ(12, concatNode->getInputs().size());

  for (unsigned i = 0; i < 12; ++i) {
    auto *mmInput = llvm::dyn_cast<MatMulNode>(concatNode->getNthInput(i));
    ASSERT_TRUE(mmInput);
    EXPECT_TRUE(mmInput->getResult().dims().equals({1, 32}));
  }

  checkNumericalEquivalence();
}

/// Test Splitting MatMul into multiple MatMuls
TEST_F(GraphOptz, SplitMatMulIntoMultipleOps_Model) {
  auto *input1 =
      mod_.createPlaceholder(ElemKind::FloatTy, {12, 48}, "input1", false);
  bindings_.allocate(input1)->getHandle<float>().randomize(-1.0, 1.0,
                                                           mod_.getPRNG());
  auto *input2 =
      mod_.createPlaceholder(ElemKind::FloatTy, {48, 48}, "input2", false);
  bindings_.allocate(input2)->getHandle<float>().randomize(-1.0, 1.0,
                                                           mod_.getPRNG());
  auto *output =
      mod_.createPlaceholder(ElemKind::FloatTy, {12, 48}, "output", false);
  bindings_.allocate(output);

  auto *mm = F_->createMatMul("mm", input1, input2);
  auto *save = F_->createSave("save", mm, output);

  ::glow::optimize(F_, CompilationMode::Infer);

  // This is F_ but without the parallel transformation below.
  optimizedF_ = F_->clone(F_->getName().str() + "_optimized");

  llvm::DenseMap<Node *, ParallelTransformKind> parOpts;
  parOpts[mm] = ParallelTransformKind::Model;

  std::unordered_map<Node *, ConcatNode *> replacedMap;
  ASSIGN_VALUE_OR_FAIL_TEST(
      replacedMap, ::glow::parallelizeOps(F_, llvm::DenseMap<Node *, size_t>(),
                                          parOpts, 12));
  EXPECT_EQ(replacedMap.size(), parOpts.size());
  runDCEPass(F_, cctx_);

  // 12 Slices from RHS
  EXPECT_EQ(12, countNodeKind(F_, Kinded::Kind::SliceNodeKind));

  EXPECT_EQ(1, countNodeKind(F_, Kinded::Kind::ConcatNodeKind));

  // 12 newly created MatMuls.
  EXPECT_EQ(12, countNodeKind(F_, Kinded::Kind::MatMulNodeKind));

  auto *concatNode = llvm::dyn_cast<ConcatNode>(save->getInput());
  ASSERT_TRUE(concatNode);
  // 12 FCs are connected to the concat node.
  EXPECT_EQ(12, concatNode->getInputs().size());

  for (unsigned i = 0; i < 12; ++i) {
    auto *mmInput = llvm::dyn_cast<MatMulNode>(concatNode->getNthInput(i));
    ASSERT_TRUE(mmInput);
    EXPECT_TRUE(mmInput->getResult().dims().equals({12, 4}));
  }

  checkNumericalEquivalence();
}

/// Test Splitting Add into multiple Adds.
TEST_F(GraphOptz, ParallelizeGraph_Add) {
  auto *input1 =
      mod_.createPlaceholder(ElemKind::FloatTy, {32, 2048}, "input1", false);
  bindings_.allocate(input1)->getHandle<float>().randomize(-1.0, 1.0,
                                                           mod_.getPRNG());
  auto *input2 =
      mod_.createPlaceholder(ElemKind::FloatTy, {32, 2048}, "input2", false);
  bindings_.allocate(input2)->getHandle<float>().randomize(-1.0, 1.0,
                                                           mod_.getPRNG());
  auto *output =
      mod_.createPlaceholder(ElemKind::FloatTy, {32, 2048}, "output", false);
  bindings_.allocate(output);

  auto *add1 = F_->createAdd("add1", input1, input2);
  auto *add2 = F_->createAdd("add2", add1, add1);
  F_->createSave("save", add2, output);

  ::glow::optimize(F_, CompilationMode::Infer);

  // This is F_ but without the parallel transformation below.
  optimizedF_ = F_->clone(F_->getName().str() + "_optimized");

  llvm::DenseMap<Node *, ParallelTransformKind> parOpts;
  parOpts[add1] = ParallelTransformKind::Data;

  std::unordered_map<Node *, ConcatNode *> replacedMap;
  ASSIGN_VALUE_OR_FAIL_TEST(
      replacedMap, ::glow::parallelizeOps(F_, llvm::DenseMap<Node *, size_t>(),
                                          parOpts, 12));
  EXPECT_EQ(replacedMap.size(), parOpts.size());
  runDCEPass(F_, cctx_);

  // We now have 12 Adds from add1, as well as the original add2 which is
  // unchanged.
  EXPECT_EQ(13, countNodeKind(F_, Kinded::Kind::AddNodeKind));

  // Each input of the 12 Adds are sliced.
  EXPECT_EQ(24, countNodeKind(F_, Kinded::Kind::SliceNodeKind));

  // One concat to bring all of the parallelized sliced Adds together.
  EXPECT_EQ(1, countNodeKind(F_, Kinded::Kind::ConcatNodeKind));

  checkNumericalEquivalence();
}

/// Test Splitting Add into multiple Adds along different axes.
static void testParallelizeGraphAddModel(PlaceholderBindings &bindings,
                                         Module &mod, Function *F,
                                         Function *&optF,
                                         CompilationContext &cctx,
                                         ParallelTransformKind parKind) {
  auto *input1 = mod.createPlaceholder(ElemKind::FloatTy, {16, 17, 18, 19, 20},
                                       "input1", false);
  bindings.allocate(input1)->getHandle<float>().randomize(-1.0, 1.0,
                                                          mod.getPRNG());
  auto *input2 = mod.createPlaceholder(ElemKind::FloatTy, {16, 17, 18, 19, 20},
                                       "input2", false);
  bindings.allocate(input2)->getHandle<float>().randomize(-1.0, 1.0,
                                                          mod.getPRNG());
  auto *output = mod.createPlaceholder(ElemKind::FloatTy, {16, 17, 18, 19, 20},
                                       "output", false);
  bindings.allocate(output);

  auto *add1 = F->createAdd("add1", input1, input2);
  auto *add2 = F->createAdd("add2", add1, add1);
  F->createSave("save", add2, output);

  ::glow::optimize(F, CompilationMode::Infer);

  // This is F_ but without the parallel transformation below.
  optF = F->clone(F->getName().str() + "_optimized");

  llvm::DenseMap<Node *, ParallelTransformKind> parOpts;
  parOpts[add1] = parKind;

  std::unordered_map<Node *, ConcatNode *> replacedMap;
  ASSIGN_VALUE_OR_FAIL_TEST(
      replacedMap,
      ::glow::parallelizeOps(F, llvm::DenseMap<Node *, size_t>(), parOpts, 12));
  EXPECT_EQ(replacedMap.size(), parOpts.size());
  runDCEPass(F, cctx);

  // We now have 12 Adds from add1, as well as the original add2 which is
  // unchanged.
  EXPECT_EQ(13, countNodeKind(F, Kinded::Kind::AddNodeKind));

  // Each input of the 12 Adds are sliced.
  EXPECT_EQ(24, countNodeKind(F, Kinded::Kind::SliceNodeKind));

  // One concat to bring all of the parallelized sliced Adds together.
  EXPECT_EQ(1, countNodeKind(F, Kinded::Kind::ConcatNodeKind));
}

TEST_F(GraphOptz, ParallelizeGraph_Add_Model_Axis1) {
  testParallelizeGraphAddModel(bindings_, mod_, F_, optimizedF_, cctx_,
                               ParallelTransformKind::Model_Axis1);
  checkNumericalEquivalence(0.f);
}

TEST_F(GraphOptz, ParallelizeGraph_Add_Model_Axis3) {
  testParallelizeGraphAddModel(bindings_, mod_, F_, optimizedF_, cctx_,
                               ParallelTransformKind::Model_Axis3);
  checkNumericalEquivalence(0.f);
}

TEST_F(GraphOptz, ParallelizeGraph_Add_Model_Axis4) {
  testParallelizeGraphAddModel(bindings_, mod_, F_, optimizedF_, cctx_,
                               ParallelTransformKind::Model_Axis4);
  checkNumericalEquivalence(0.f);
}

/// Test Splitting Sub into multiple Subs.
TEST_F(GraphOptz, ParallelizeGraph_Sub) {
  auto *input1 =
      mod_.createPlaceholder(ElemKind::FloatTy, {32, 2048}, "input1", false);
  bindings_.allocate(input1)->getHandle<float>().randomize(-1.0, 1.0,
                                                           mod_.getPRNG());
  auto *input2 =
      mod_.createPlaceholder(ElemKind::FloatTy, {32, 2048}, "input2", false);
  bindings_.allocate(input2)->getHandle<float>().randomize(-1.0, 1.0,
                                                           mod_.getPRNG());
  auto *output =
      mod_.createPlaceholder(ElemKind::FloatTy, {32, 2048}, "output", false);
  bindings_.allocate(output);

  auto *sub1 = F_->createSub("sub1", input1, input2);
  auto *sub2 = F_->createSub("sub2", sub1, sub1);
  F_->createSave("save", sub2, output);

  ::glow::optimize(F_, CompilationMode::Infer);

  // This is F_ but without the parallel transformation below.
  optimizedF_ = F_->clone(F_->getName().str() + "_optimized");

  llvm::DenseMap<Node *, ParallelTransformKind> parOpts;
  parOpts[sub1] = ParallelTransformKind::Data;

  std::unordered_map<Node *, ConcatNode *> replacedMap;
  ASSIGN_VALUE_OR_FAIL_TEST(
      replacedMap, ::glow::parallelizeOps(F_, llvm::DenseMap<Node *, size_t>(),
                                          parOpts, 12));
  EXPECT_EQ(replacedMap.size(), parOpts.size());
  runDCEPass(F_, cctx_);

  // We now have 12 Subs from sub1, as well as the original sub2 which is
  // unchanged.
  EXPECT_EQ(13, countNodeKind(F_, Kinded::Kind::SubNodeKind));

  // Each input of the 12 Subs are sliced.
  EXPECT_EQ(24, countNodeKind(F_, Kinded::Kind::SliceNodeKind));

  // One concat to bring all of the parallelized sliced Subs together.
  EXPECT_EQ(1, countNodeKind(F_, Kinded::Kind::ConcatNodeKind));

  checkNumericalEquivalence();
}

/// Test Splitting Pow into multiple Pows.
TEST_F(GraphOptz, ParallelizeGraph_Pow) {
  auto *input1 =
      mod_.createPlaceholder(ElemKind::FloatTy, {32, 2048}, "input1", false);
  bindings_.allocate(input1)->getHandle<float>().randomize(1.0, 2.0,
                                                           mod_.getPRNG());
  auto *input2 =
      mod_.createPlaceholder(ElemKind::FloatTy, {32, 2048}, "input2", false);
  bindings_.allocate(input2)->getHandle<float>().randomize(0.0, 5.0,
                                                           mod_.getPRNG());
  auto *output =
      mod_.createPlaceholder(ElemKind::FloatTy, {32, 2048}, "output", false);
  bindings_.allocate(output);

  auto *Pow1 = F_->createPow("Pow1", input1, input2);
  F_->createSave("save", Pow1, output);

  ::glow::optimize(F_, CompilationMode::Infer);

  // This is F_ but without the parallel transformation below.
  optimizedF_ = F_->clone(F_->getName().str() + "_optimized");

  llvm::DenseMap<Node *, ParallelTransformKind> parOpts;
  parOpts[Pow1] = ParallelTransformKind::Data;

  std::unordered_map<Node *, ConcatNode *> replacedMap;
  ASSIGN_VALUE_OR_FAIL_TEST(
      replacedMap, ::glow::parallelizeOps(F_, llvm::DenseMap<Node *, size_t>(),
                                          parOpts, 12));
  EXPECT_EQ(replacedMap.size(), parOpts.size());
  runDCEPass(F_, cctx_);

  // We now have 12 Pows from Pow1
  EXPECT_EQ(12, countNodeKind(F_, Kinded::Kind::PowNodeKind));

  // Each input of the 12 Pows are sliced.
  EXPECT_EQ(24, countNodeKind(F_, Kinded::Kind::SliceNodeKind));

  // One concat to bring all of the parallelized sliced Pows together.
  EXPECT_EQ(1, countNodeKind(F_, Kinded::Kind::ConcatNodeKind));

  checkNumericalEquivalence();
}

/// Test Splitting Max into multiple Maxs.
TEST_F(GraphOptz, ParallelizeGraph_Max) {
  auto *input1 =
      mod_.createPlaceholder(ElemKind::FloatTy, {32, 2048}, "input1", false);
  bindings_.allocate(input1)->getHandle<float>().randomize(-1.0, 1.0,
                                                           mod_.getPRNG());
  auto *input2 =
      mod_.createPlaceholder(ElemKind::FloatTy, {32, 2048}, "input2", false);
  bindings_.allocate(input2)->getHandle<float>().randomize(-1.0, 1.0,
                                                           mod_.getPRNG());
  auto *output =
      mod_.createPlaceholder(ElemKind::FloatTy, {32, 2048}, "output", false);
  bindings_.allocate(output);

  auto *Max1 = F_->createMax("Max1", input1, input2);
  F_->createSave("save", Max1, output);

  ::glow::optimize(F_, CompilationMode::Infer);

  // This is F_ but without the parallel transformation below.
  optimizedF_ = F_->clone(F_->getName().str() + "_optimized");

  llvm::DenseMap<Node *, ParallelTransformKind> parOpts;
  parOpts[Max1] = ParallelTransformKind::Data;

  std::unordered_map<Node *, ConcatNode *> replacedMap;
  ASSIGN_VALUE_OR_FAIL_TEST(
      replacedMap, ::glow::parallelizeOps(F_, llvm::DenseMap<Node *, size_t>(),
                                          parOpts, 12));
  EXPECT_EQ(replacedMap.size(), parOpts.size());
  runDCEPass(F_, cctx_);

  // We now have 12 Maxs from Max1
  EXPECT_EQ(12, countNodeKind(F_, Kinded::Kind::MaxNodeKind));

  // Each input of the 12 Maxs are sliced.
  EXPECT_EQ(24, countNodeKind(F_, Kinded::Kind::SliceNodeKind));

  // One concat to bring all of the parallelized sliced Maxs together.
  EXPECT_EQ(1, countNodeKind(F_, Kinded::Kind::ConcatNodeKind));

  checkNumericalEquivalence();
}

/// Test Splitting Min into multiple Mins.
TEST_F(GraphOptz, ParallelizeGraph_Min) {
  auto *input1 =
      mod_.createPlaceholder(ElemKind::FloatTy, {32, 2048}, "input1", false);
  bindings_.allocate(input1)->getHandle<float>().randomize(-1.0, 1.0,
                                                           mod_.getPRNG());
  auto *input2 =
      mod_.createPlaceholder(ElemKind::FloatTy, {32, 2048}, "input2", false);
  bindings_.allocate(input2)->getHandle<float>().randomize(-1.0, 1.0,
                                                           mod_.getPRNG());
  auto *output =
      mod_.createPlaceholder(ElemKind::FloatTy, {32, 2048}, "output", false);
  bindings_.allocate(output);

  auto *Min1 = F_->createMin("Min1", input1, input2);
  F_->createSave("save", Min1, output);

  ::glow::optimize(F_, CompilationMode::Infer);

  // This is F_ but without the parallel transformation below.
  optimizedF_ = F_->clone(F_->getName().str() + "_optimized");

  llvm::DenseMap<Node *, ParallelTransformKind> parOpts;
  parOpts[Min1] = ParallelTransformKind::Data;

  std::unordered_map<Node *, ConcatNode *> replacedMap;
  ASSIGN_VALUE_OR_FAIL_TEST(
      replacedMap, ::glow::parallelizeOps(F_, llvm::DenseMap<Node *, size_t>(),
                                          parOpts, 12));
  EXPECT_EQ(replacedMap.size(), parOpts.size());
  runDCEPass(F_, cctx_);

  // We now have 12 Mins from Min1
  EXPECT_EQ(12, countNodeKind(F_, Kinded::Kind::MinNodeKind));

  // Each input of the 12 Mins are sliced.
  EXPECT_EQ(24, countNodeKind(F_, Kinded::Kind::SliceNodeKind));

  // One concat to bring all of the parallelized sliced Mins together.
  EXPECT_EQ(1, countNodeKind(F_, Kinded::Kind::ConcatNodeKind));

  checkNumericalEquivalence();
}

/// Test Splitting BatchedReduceMean into multiple BatchedReduceMeans.
TEST_F(GraphOptz, ParallelizeGraph_BatchedReduceMean) {
  auto *input1 = mod_.createPlaceholder(ElemKind::FloatTy, {32, 16, 2048},
                                        "input1", false);
  bindings_.allocate(input1)->getHandle<float>().randomize(-1.0, 1.0,
                                                           mod_.getPRNG());
  auto *output =
      mod_.createPlaceholder(ElemKind::FloatTy, {32, 2048}, "output", false);
  bindings_.allocate(output);

  auto *BatchedReduceMean1 =
      F_->createBatchedReduceMean("BatchedReduceMean1", input1, {1});
  F_->createSave("save", BatchedReduceMean1, output);

  ::glow::optimize(F_, CompilationMode::Infer);

  // This is F_ but without the parallel transformation below.
  optimizedF_ = F_->clone(F_->getName().str() + "_optimized");

  llvm::DenseMap<Node *, ParallelTransformKind> parOpts;
  parOpts[BatchedReduceMean1] = ParallelTransformKind::Data;

  std::unordered_map<Node *, ConcatNode *> replacedMap;
  ASSIGN_VALUE_OR_FAIL_TEST(
      replacedMap, ::glow::parallelizeOps(F_, llvm::DenseMap<Node *, size_t>(),
                                          parOpts, 12));
  EXPECT_EQ(replacedMap.size(), parOpts.size());
  runDCEPass(F_, cctx_);

  // We now have 12 BatchedReduceMeans from BatchedReduceMean1
  EXPECT_EQ(12, countNodeKind(F_, Kinded::Kind::BatchedReduceMeanNodeKind));

  // Each input of the 12 BatchedReduceMeans are sliced.
  EXPECT_EQ(12, countNodeKind(F_, Kinded::Kind::SliceNodeKind));

  // One concat to bring all of the parallelized sliced BatchedReduceMeans
  // together.
  EXPECT_EQ(1, countNodeKind(F_, Kinded::Kind::ConcatNodeKind));

  checkNumericalEquivalence();
}

/// Test Splitting BatchedReduceMean into multiple BatchedReduceMeans.
/// Failure case with first dimension in reduction
TEST_F(GraphOptz, ParallelizeGraph_BatchedReduceMean_failure) {
  auto *input1 = mod_.createPlaceholder(ElemKind::FloatTy, {32, 16, 2048},
                                        "input1", false);
  bindings_.allocate(input1)->getHandle<float>().randomize(-1.0, 1.0,
                                                           mod_.getPRNG());
  auto *output =
      mod_.createPlaceholder(ElemKind::FloatTy, {16, 2048}, "output", false);
  bindings_.allocate(output);

  auto *BatchedReduceMean1 =
      F_->createBatchedReduceMean("BatchedReduceMean1", input1, {0});
  F_->createSave("save", BatchedReduceMean1, output);

  ::glow::optimize(F_, CompilationMode::Infer);

  // This is F_ but without the parallel transformation below.
  optimizedF_ = F_->clone(F_->getName().str() + "_optimized");

  llvm::DenseMap<Node *, ParallelTransformKind> parOpts;
  parOpts[BatchedReduceMean1] = ParallelTransformKind::Data;

  std::unordered_map<Node *, ConcatNode *> replacedMap;
  ASSIGN_VALUE_OR_FAIL_TEST(
      replacedMap, ::glow::parallelizeOps(F_, llvm::DenseMap<Node *, size_t>(),
                                          parOpts, 12));
  EXPECT_EQ(replacedMap.size(), 0); // Nothing changes
  runDCEPass(F_, cctx_);

  // We now have only 1 BatchedReduceMean since parallelization is disabled
  EXPECT_EQ(1, countNodeKind(F_, Kinded::Kind::BatchedReduceMeanNodeKind));

  // No concats
  EXPECT_EQ(0, countNodeKind(F_, Kinded::Kind::ConcatNodeKind));

  checkNumericalEquivalence();
}

/// Test Splitting Transpose into multiple Transposes.
TEST_F(GraphOptz, ParallelizeGraph_Transpose) {
  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {32, 151, 64}, "input", false);
  bindings_.allocate(input)->getHandle<float>().randomize(-1.0, 1.0,
                                                          mod_.getPRNG());
  auto *output =
      mod_.createPlaceholder(ElemKind::FloatTy, {32, 64, 151}, "output", false);
  bindings_.allocate(output);

  auto *trans1 = F_->createTranspose("trans1", input, {0, 2, 1});
  F_->createSave("save", trans1, output);

  ::glow::optimize(F_, CompilationMode::Infer);

  // This is F_ but without the parallel transformation below.
  optimizedF_ = F_->clone(F_->getName().str() + "_optimized");

  llvm::DenseMap<Node *, size_t> numChunks;
  llvm::DenseMap<Node *, ParallelTransformKind> parOpts;
  numChunks[trans1] = 2;
  parOpts[trans1] = ParallelTransformKind::Data;
  std::unordered_map<Node *, ConcatNode *> replacedMap;
  ASSIGN_VALUE_OR_FAIL_TEST(replacedMap,
                            ::glow::parallelizeOps(F_, numChunks, parOpts));
  EXPECT_EQ(replacedMap.size(), parOpts.size());

  runDCEPass(F_, cctx_);

  EXPECT_EQ(2, countNodeKind(F_, Kinded::Kind::TransposeNodeKind));

  checkNumericalEquivalence();
}

/// Test Splitting Transpose into multiple Transposes.
TEST_F(GraphOptz, ParallelizeGraph_Transpose3D_210) {
  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {4, 15, 23}, "input", false);
  bindings_.allocate(input)->getHandle<float>().randomize(-1.0, 1.0,
                                                          mod_.getPRNG());
  auto *output =
      mod_.createPlaceholder(ElemKind::FloatTy, {23, 15, 4}, "output", false);
  bindings_.allocate(output);

  auto *trans1 = F_->createTranspose("trans1", input, {2, 1, 0});
  F_->createSave("save", trans1, output);

  ::glow::optimize(F_, CompilationMode::Infer);

  // This is F_ but without the parallel transformation below.
  optimizedF_ = F_->clone(F_->getName().str() + "_optimized");

  llvm::DenseMap<Node *, size_t> numChunks;
  llvm::DenseMap<Node *, ParallelTransformKind> parOpts;
  numChunks[trans1] = 8;
  parOpts[trans1] = ParallelTransformKind::Data;
  std::unordered_map<Node *, ConcatNode *> replacedMap;
  ASSIGN_VALUE_OR_FAIL_TEST(replacedMap,
                            ::glow::parallelizeOps(F_, numChunks, parOpts));
  EXPECT_EQ(replacedMap.size(), parOpts.size());

  runDCEPass(F_, cctx_);

  EXPECT_EQ(8, countNodeKind(F_, Kinded::Kind::TransposeNodeKind));

  checkNumericalEquivalence();
}

/// Test Splitting Transpose into multiple Transposes.
TEST_F(GraphOptz, ParallelizeGraph_Transpose3D_120) {
  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {15, 8, 23}, "input", false);
  bindings_.allocate(input)->getHandle<float>().randomize(-1.0, 1.0,
                                                          mod_.getPRNG());
  auto *output =
      mod_.createPlaceholder(ElemKind::FloatTy, {8, 23, 15}, "output", false);
  bindings_.allocate(output);

  auto *trans1 = F_->createTranspose("trans1", input, {1, 2, 0});
  F_->createSave("save", trans1, output);

  ::glow::optimize(F_, CompilationMode::Infer);

  // This is F_ but without the parallel transformation below.
  optimizedF_ = F_->clone(F_->getName().str() + "_optimized");

  llvm::DenseMap<Node *, size_t> numChunks;
  llvm::DenseMap<Node *, ParallelTransformKind> parOpts;
  numChunks[trans1] = 8;
  parOpts[trans1] = ParallelTransformKind::Data;
  std::unordered_map<Node *, ConcatNode *> replacedMap;
  ASSIGN_VALUE_OR_FAIL_TEST(replacedMap,
                            ::glow::parallelizeOps(F_, numChunks, parOpts));
  EXPECT_EQ(replacedMap.size(), parOpts.size());

  runDCEPass(F_, cctx_);

  EXPECT_EQ(8, countNodeKind(F_, Kinded::Kind::TransposeNodeKind));

  checkNumericalEquivalence();
}

/// Test Splitting Select into multiple Selects.
TEST_F(GraphOptz, ParallelizeGraphData_Select) {
  auto *sel1_lhs =
      mod_.createPlaceholder(ElemKind::FloatTy, {32, 2048}, "sel1_lhs", false);
  bindings_.allocate(sel1_lhs)->getHandle<float>().randomize(-1.0, 1.0,
                                                             mod_.getPRNG());
  auto *sel1_rhs =
      mod_.createPlaceholder(ElemKind::FloatTy, {32, 2048}, "sel1_rhs", false);
  bindings_.allocate(sel1_rhs)->getHandle<float>().randomize(-1.0, 1.0,
                                                             mod_.getPRNG());
  auto *sel1_cond =
      mod_.createPlaceholder(ElemKind::BoolTy, {32, 2048}, "sel1_cond", false);
  bindings_.allocate(sel1_cond)->getHandle<bool>().randomize(0, 1,
                                                             mod_.getPRNG());
  auto *sel2_rhs =
      mod_.createPlaceholder(ElemKind::FloatTy, {32, 2048}, "sel2_rhs", false);
  bindings_.allocate(sel2_rhs)->getHandle<float>().randomize(-1.0, 1.0,
                                                             mod_.getPRNG());
  auto *sel2_cond =
      mod_.createPlaceholder(ElemKind::BoolTy, {32, 2048}, "sel2_cond", false);
  bindings_.allocate(sel2_cond)->getHandle<bool>().randomize(0, 1,
                                                             mod_.getPRNG());
  auto *output =
      mod_.createPlaceholder(ElemKind::FloatTy, {32, 2048}, "output", false);
  bindings_.allocate(output);

  auto *sel1 = F_->createSelect("sel1", sel1_cond, sel1_lhs, sel1_rhs);
  auto *sel2 = F_->createSelect("sel2", sel2_cond, sel1, sel2_rhs);
  F_->createSave("save", sel2, output);

  ::glow::optimize(F_, CompilationMode::Infer);

  // This is F_ but without the parallel transformation below.
  optimizedF_ = F_->clone(F_->getName().str() + "_optimized");

  llvm::DenseMap<Node *, ParallelTransformKind> parOpts;
  parOpts[sel1] = ParallelTransformKind::Data;

  std::unordered_map<Node *, ConcatNode *> replacedMap;
  ASSIGN_VALUE_OR_FAIL_TEST(
      replacedMap, ::glow::parallelizeOps(F_, llvm::DenseMap<Node *, size_t>(),
                                          parOpts, 12));
  EXPECT_EQ(replacedMap.size(), parOpts.size());
  runDCEPass(F_, cctx_);

  // We now have 12 Selects from sel1, as well as the original sel2 which is
  // unchanged.
  EXPECT_EQ(13, countNodeKind(F_, Kinded::Kind::SelectNodeKind));

  // Each input (3 total inputs) of the 12 Selects are sliced.
  EXPECT_EQ(36, countNodeKind(F_, Kinded::Kind::SliceNodeKind));

  // One concat to bring all of the parallelized sliced Select together.
  EXPECT_EQ(1, countNodeKind(F_, Kinded::Kind::ConcatNodeKind));

  checkNumericalEquivalence();
}

/// Test Splitting Select into multiple Selects.
TEST_F(GraphOptz, ParallelizeGraphModel_Select) {
  auto *sel1_lhs =
      mod_.createPlaceholder(ElemKind::FloatTy, {32, 2048}, "sel1_lhs", false);
  bindings_.allocate(sel1_lhs)->getHandle<float>().randomize(-1.0, 1.0,
                                                             mod_.getPRNG());
  auto *sel1_rhs =
      mod_.createPlaceholder(ElemKind::FloatTy, {32, 2048}, "sel1_rhs", false);
  bindings_.allocate(sel1_rhs)->getHandle<float>().randomize(-1.0, 1.0,
                                                             mod_.getPRNG());
  auto *sel1_cond =
      mod_.createPlaceholder(ElemKind::BoolTy, {32, 2048}, "sel1_cond", false);
  bindings_.allocate(sel1_cond)->getHandle<bool>().randomize(0, 1,
                                                             mod_.getPRNG());
  auto *sel2_rhs =
      mod_.createPlaceholder(ElemKind::FloatTy, {32, 2048}, "sel2_rhs", false);
  bindings_.allocate(sel2_rhs)->getHandle<float>().randomize(-1.0, 1.0,
                                                             mod_.getPRNG());
  auto *sel2_cond =
      mod_.createPlaceholder(ElemKind::BoolTy, {32, 2048}, "sel2_cond", false);
  bindings_.allocate(sel2_cond)->getHandle<bool>().randomize(0, 1,
                                                             mod_.getPRNG());
  auto *output =
      mod_.createPlaceholder(ElemKind::FloatTy, {32, 2048}, "output", false);
  bindings_.allocate(output);

  auto *sel1 = F_->createSelect("sel1", sel1_cond, sel1_lhs, sel1_rhs);
  auto *sel2 = F_->createSelect("sel2", sel2_cond, sel1, sel2_rhs);
  F_->createSave("save", sel2, output);

  ::glow::optimize(F_, CompilationMode::Infer);

  // This is F_ but without the parallel transformation below.
  optimizedF_ = F_->clone(F_->getName().str() + "_optimized");

  llvm::DenseMap<Node *, ParallelTransformKind> parOpts;
  parOpts[sel1] = ParallelTransformKind::Model;

  std::unordered_map<Node *, ConcatNode *> replacedMap;
  ASSIGN_VALUE_OR_FAIL_TEST(
      replacedMap, ::glow::parallelizeOps(F_, llvm::DenseMap<Node *, size_t>(),
                                          parOpts, 12));
  EXPECT_EQ(replacedMap.size(), parOpts.size());
  runDCEPass(F_, cctx_);

  // We now have 12 Selects from sel1, as well as the original sel2 which is
  // unchanged.
  EXPECT_EQ(13, countNodeKind(F_, Kinded::Kind::SelectNodeKind));

  // Each input (3 total inputs) of the 12 Selects are sliced.
  EXPECT_EQ(36, countNodeKind(F_, Kinded::Kind::SliceNodeKind));

  // One concat to bring all of the parallelized sliced Select together.
  EXPECT_EQ(1, countNodeKind(F_, Kinded::Kind::ConcatNodeKind));

  checkNumericalEquivalence();
}

/// Test Splitting Reshape into multiple Reshapes.
TEST_F(GraphOptz, ParallelizeData_Reshape) {
  auto *input1 =
      mod_.createPlaceholder(ElemKind::FloatTy, {3, 64}, "input1", false);
  bindings_.allocate(input1)->getHandle<float>().randomize(-1.0, 1.0,
                                                           mod_.getPRNG());
  auto *output =
      mod_.createPlaceholder(ElemKind::FloatTy, {3, 8, 8}, "output", false);
  bindings_.allocate(output);

  auto *rs = F_->createReshape("reshape1", input1, {3, 8, 8});
  F_->createSave("save", rs, output);

  ::glow::optimize(F_, CompilationMode::Infer);

  // This is F_ but without the parallel transformation below.
  optimizedF_ = F_->clone(F_->getName().str() + "_optimized");

  llvm::DenseMap<Node *, ParallelTransformKind> parOpts;
  parOpts[rs] = ParallelTransformKind::Data;

  std::unordered_map<Node *, ConcatNode *> replacedMap;
  ASSIGN_VALUE_OR_FAIL_TEST(
      replacedMap,
      ::glow::parallelizeOps(F_, llvm::DenseMap<Node *, size_t>(), parOpts, 3));
  EXPECT_EQ(replacedMap.size(), parOpts.size());
  runDCEPass(F_, cctx_);

  // We now have 3 Reshapes
  EXPECT_EQ(3, countNodeKind(F_, Kinded::Kind::ReshapeNodeKind));

  // One concat to bring all of the parallelized sliced Reshapes together.
  EXPECT_EQ(1, countNodeKind(F_, Kinded::Kind::ConcatNodeKind));

  checkNumericalEquivalence();
}

/// Test Splitting Reshape into multiple Reshapes when the batch
/// dimension changes. This is not allowed when the input or output batch size
/// dim cannot be divided by the # of the parallel chunks.
TEST_F(GraphOptz, ParallelizeData_Reshape_badcase) {
  auto *input1 =
      mod_.createPlaceholder(ElemKind::FloatTy, {4, 48}, "input1", false);
  bindings_.allocate(input1)->getHandle<float>().randomize(-1.0, 1.0,
                                                           mod_.getPRNG());
  auto *output =
      mod_.createPlaceholder(ElemKind::FloatTy, {24, 8}, "output", false);
  bindings_.allocate(output);

  auto *rs = F_->createReshape("reshape1", input1, {24, 8});
  F_->createSave("save", rs, output);

  ::glow::optimize(F_, CompilationMode::Infer);

  // This is F_ but without the parallel transformation below.
  optimizedF_ = F_->clone(F_->getName().str() + "_optimized");

  llvm::DenseMap<Node *, ParallelTransformKind> parOpts;
  parOpts[rs] = ParallelTransformKind::Data;

  std::unordered_map<Node *, ConcatNode *> replacedMap;
  ASSIGN_VALUE_OR_FAIL_TEST(
      replacedMap,
      ::glow::parallelizeOps(F_, llvm::DenseMap<Node *, size_t>(), parOpts, 3));
  EXPECT_EQ(replacedMap.size(), 0); // Nothing gets replaced
  runDCEPass(F_, cctx_);

  // We now have only 1 Reshape as nothing should have split
  EXPECT_EQ(1, countNodeKind(F_, Kinded::Kind::ReshapeNodeKind));

  checkNumericalEquivalence();
}

/// Test Splitting AdaptiveAvgPool into multiple AdaptiveAvgPools.
TEST_F(GraphOptz, ParallelizeData_AdaptiveAvgPool) {
  auto *input1 =
      mod_.createPlaceholder(ElemKind::FloatTy, {3, 5, 5, 8}, "input1", false);
  bindings_.allocate(input1)->getHandle<float>().randomize(-1.0, 1.0,
                                                           mod_.getPRNG());
  auto *output =
      mod_.createPlaceholder(ElemKind::FloatTy, {3, 1, 1, 8}, "output", false);
  bindings_.allocate(output);

  auto outTy = mod_.uniqueType(ElemKind::FloatTy, {3, 1, 1, 8});

  auto *aap = F_->createAdaptiveAvgPool("AdaptiveAvgPool1", input1, outTy);
  F_->createSave("save", aap, output);

  ::glow::optimize(F_, CompilationMode::Infer);

  // This is F_ but without the parallel transformation below.
  optimizedF_ = F_->clone(F_->getName().str() + "_optimized");

  llvm::DenseMap<Node *, ParallelTransformKind> parOpts;
  parOpts[aap] = ParallelTransformKind::Data;

  std::unordered_map<Node *, ConcatNode *> replacedMap;
  ASSIGN_VALUE_OR_FAIL_TEST(
      replacedMap,
      ::glow::parallelizeOps(F_, llvm::DenseMap<Node *, size_t>(), parOpts, 3));
  EXPECT_EQ(replacedMap.size(), parOpts.size());
  runDCEPass(F_, cctx_);

  // We now have 3 AdaptiveAvgPools
  EXPECT_EQ(3, countNodeKind(F_, Kinded::Kind::AdaptiveAvgPoolNodeKind));

  // One concat to bring all of the parallelized sliced AdaptiveAvgPools
  // together.
  EXPECT_EQ(1, countNodeKind(F_, Kinded::Kind::ConcatNodeKind));

  checkNumericalEquivalence();
}

/// Test Splitting RoIAlign into multiple RoIAligns.
TEST_F(GraphOptz, ParallelizeData_RoIAlign) {
  auto *input1 =
      mod_.createPlaceholder(ElemKind::FloatTy, {4, 5, 5, 8}, "input1", false);
  bindings_.allocate(input1)->getHandle<float>().randomize(-1.0, 1.0,
                                                           mod_.getPRNG());
  auto *boxes = mod_.createPlaceholder(ElemKind::FloatTy, {6, 4}, "roi", false);
  bindings_.allocate(boxes)->getHandle<float>() = {
      0, 0, 3, 3, 0, 0, 3, 3, 0, 0, 3, 3, 0, 0, 3, 3, 0, 0, 3, 3, 0, 0, 3, 3};
  auto *batchIndices =
      mod_.createPlaceholder(ElemKind::Int64ITy, {6}, "roi", false);
  bindings_.allocate(batchIndices)
      ->getHandle<int64_t>()
      .randomize(0, 3, mod_.getPRNG());

  auto *output =
      mod_.createPlaceholder(ElemKind::FloatTy, {6, 1, 1, 8}, "output", false);
  bindings_.allocate(output);

  auto *aap = F_->createROIAlign("ROIAlign", input1, boxes, batchIndices, 1, 1,
                                 0, 1, false);
  F_->createSave("save", aap, output);

  ::glow::optimize(F_, CompilationMode::Infer);

  // This is F_ but without the parallel transformation below.
  optimizedF_ = F_->clone(F_->getName().str() + "_optimized");

  llvm::DenseMap<Node *, ParallelTransformKind> parOpts;
  parOpts[aap] = ParallelTransformKind::Data;

  std::unordered_map<Node *, ConcatNode *> replacedMap;
  ASSIGN_VALUE_OR_FAIL_TEST(
      replacedMap,
      ::glow::parallelizeOps(F_, llvm::DenseMap<Node *, size_t>(), parOpts, 3));
  EXPECT_EQ(replacedMap.size(), parOpts.size());
  runDCEPass(F_, cctx_);

  // We now have 3 RoIAligns
  EXPECT_EQ(3, countNodeKind(F_, Kinded::Kind::ROIAlignNodeKind));

  // One concat to bring all of the parallelized sliced RoIAligns
  // together.
  EXPECT_EQ(1, countNodeKind(F_, Kinded::Kind::ConcatNodeKind));

  checkNumericalEquivalence();
}

/// Test Splitting MaxPool into multiple MaxPools.
TEST_F(GraphOptz, ParallelizeData_MaxPool) {
  auto *input1 = mod_.createPlaceholder(ElemKind::Int8QTy, {3, 5, 5, 8}, 1.0, 0,
                                        "input1", false);
  bindings_.allocate(input1)->getHandle<int8_t>().randomize(-1.0, 1.0,
                                                            mod_.getPRNG());

  auto *maxp = F_->createMaxPool("MaxPool1", input1, 5, 1, 0);
  F_->createSave("save", maxp->getResult());

  ::glow::optimize(F_, CompilationMode::Infer);

  // This is F_ but without the parallel transformation below.
  optimizedF_ = F_->clone(F_->getName().str() + "_optimized");

  llvm::DenseMap<Node *, ParallelTransformKind> parOpts;
  parOpts[maxp] = ParallelTransformKind::Data;

  std::unordered_map<Node *, ConcatNode *> replacedMap;
  ASSIGN_VALUE_OR_FAIL_TEST(
      replacedMap,
      ::glow::parallelizeOps(F_, llvm::DenseMap<Node *, size_t>(), parOpts, 3));
  EXPECT_EQ(replacedMap.size(), parOpts.size());
  runDCEPass(F_, cctx_);

  // We now have 3 MaxPools
  EXPECT_EQ(3, countNodeKind(F_, Kinded::Kind::MaxPoolNodeKind));

  // One concat to bring all of the parallelized sliced MaxPools
  // together.
  EXPECT_EQ(1, countNodeKind(F_, Kinded::Kind::ConcatNodeKind));

  checkNumericalEquivalence();
}

/// Test Splitting ChannelwiseQuantizedConvolution into multiple
/// ChannelwiseQuantizedConvolutions.
TEST_F(GraphOptz, ParallelizeData_ChannelwiseQuantizedConvolution) {
  auto *input1 = mod_.createPlaceholder(ElemKind::Int8QTy, {3, 5, 5, 8}, 1.0, 0,
                                        "input1", false);
  bindings_.allocate(input1)->getHandle<int8_t>().randomize(-4, 4,
                                                            mod_.getPRNG());
  auto *filter =
      mod_.createConstant(ElemKind::FloatTy, {12, 1, 1, 8}, "weights");
  filter->getPayloadMutable().getHandle().randomize(-10, 10, mod_.getPRNG());
  auto *bias = mod_.createConstant(ElemKind::FloatTy, {12}, "bias");
  bias->getPayloadMutable().getHandle().randomize(-1, 1, mod_.getPRNG());
  auto *output = mod_.createPlaceholder(ElemKind::Int8QTy, {3, 5, 5, 12}, 1.0,
                                        0, "output", false);
  bindings_.allocate(output);
  auto outTy = mod_.uniqueType(ElemKind::Int8QTy, {3, 5, 5, 12}, 1.0, 0);

  auto *cqc = F_->createChannelwiseQuantizedConv(
      "ChannelwiseQuantizedConvolution1", input1, filter, bias, nullptr,
      nullptr, nullptr, nullptr, outTy, {1, 1}, {1, 1}, {0, 0, 0, 0}, 1);
  F_->createSave("save", cqc, output);

  ::glow::optimize(F_, CompilationMode::Infer);

  // This is F_ but without the parallel transformation below.
  optimizedF_ = F_->clone(F_->getName().str() + "_optimized");

  llvm::DenseMap<Node *, ParallelTransformKind> parOpts;
  parOpts[cqc] = ParallelTransformKind::Data;

  std::unordered_map<Node *, ConcatNode *> replacedMap;
  ASSIGN_VALUE_OR_FAIL_TEST(
      replacedMap,
      ::glow::parallelizeOps(F_, llvm::DenseMap<Node *, size_t>(), parOpts, 3));
  EXPECT_EQ(replacedMap.size(), parOpts.size());
  runDCEPass(F_, cctx_);

  // We now have 3 ChannelwiseQuantizedConvolutions
  EXPECT_EQ(3, countNodeKind(
                   F_, Kinded::Kind::ChannelwiseQuantizedConvolutionNodeKind));

  // One concat to bring all of the parallelized sliced
  // ChannelwiseQuantizedConvolutions together.
  EXPECT_EQ(1, countNodeKind(F_, Kinded::Kind::ConcatNodeKind));

  checkNumericalEquivalence();
}

/// Test Splitting Convolution into multiple Convolutions.
TEST_F(GraphOptz, ParallelizeData_Convolution) {
  auto *input1 =
      mod_.createPlaceholder(ElemKind::FloatTy, {3, 5, 5, 4}, "input1", false);
  bindings_.allocate(input1)->getHandle<float>().randomize(-1, 1,
                                                           mod_.getPRNG());
  auto *filter =
      mod_.createConstant(ElemKind::FloatTy, {6, 1, 1, 2}, "weights");
  filter->getPayloadMutable().getHandle().randomize(-1, 1, mod_.getPRNG());
  auto *bias = mod_.createConstant(ElemKind::FloatTy, {6}, "bias");
  bias->getPayloadMutable().getHandle().randomize(-.1, .1, mod_.getPRNG());
  auto *output =
      mod_.createPlaceholder(ElemKind::FloatTy, {3, 5, 5, 6}, "output", false);
  bindings_.allocate(output);
  auto outTy = mod_.uniqueType(ElemKind::FloatTy, {3, 5, 5, 6});

  auto *conv =
      F_->createConv("Convolution1", input1, filter, bias, outTy, 1, 1, 0, 2);
  F_->createSave("save", conv, output);

  ::glow::optimize(F_, CompilationMode::Infer);

  // This is F_ but without the parallel transformation below.
  optimizedF_ = F_->clone(F_->getName().str() + "_optimized");

  llvm::DenseMap<Node *, ParallelTransformKind> parOpts;
  parOpts[conv] = ParallelTransformKind::Data;

  std::unordered_map<Node *, ConcatNode *> replacedMap;
  ASSIGN_VALUE_OR_FAIL_TEST(
      replacedMap,
      ::glow::parallelizeOps(F_, llvm::DenseMap<Node *, size_t>(), parOpts, 3));
  EXPECT_EQ(replacedMap.size(), parOpts.size());
  runDCEPass(F_, cctx_);

  // We now have 3 Convolutions
  EXPECT_EQ(3, countNodeKind(F_, Kinded::Kind::ConvolutionNodeKind));

  // One concat to bring all of the parallelized sliced
  // Convolutions together.
  EXPECT_EQ(1, countNodeKind(F_, Kinded::Kind::ConcatNodeKind));

  checkNumericalEquivalence();
}

/// Test Splitting RowwiseQuantizedFullyConnected into multiple
/// RowwiseQuantizedFullyConnected nodes.
TEST_F(GraphOptz, ParallelizeData_RowwiseQuantizedFullyConnected) {
  auto *input1 = mod_.createPlaceholder(ElemKind::Int8QTy, {3, 8}, 1.0, 0,
                                        "input1", false);
  bindings_.allocate(input1)->getHandle<int8_t>().randomize(-4, 4,
                                                            mod_.getPRNG());
  auto *weights =
      mod_.createConstant(ElemKind::Int8QTy, {12, 8}, 1.0, 0, "weights");
  weights->getPayloadMutable().getHandle<int8_t>().randomize(-128, 127,
                                                             mod_.getPRNG());
  auto *scales = mod_.createConstant(ElemKind::FloatTy, {12}, "scales");
  scales->getPayloadMutable().getHandle().randomize(0.01, 0.1, mod_.getPRNG());
  auto *offsets = mod_.createConstant(ElemKind::Int32ITy, {12}, "offsets");
  offsets->getPayloadMutable().getHandle<int32_t>().randomize(0, 10,
                                                              mod_.getPRNG());

  auto *bias = mod_.createConstant(ElemKind::Int8QTy, {12}, 1.0, 0, "bias");
  bias->getPayloadMutable().getHandle<int8_t>().randomize(-128, 127,
                                                          mod_.getPRNG());
  auto *output = mod_.createPlaceholder(ElemKind::Int8QTy, {3, 12}, 1.0, 0,
                                        "output", false);
  bindings_.allocate(output);
  auto outTy = mod_.uniqueType(ElemKind::Int8QTy, {3, 12}, 1.0, 0);

  auto *rqfc = F_->createRowwiseQuantizedFullyConnected(
      "RowwiseQuantizedFullyConnected1", input1, weights, scales, offsets, bias,
      outTy);
  F_->createSave("save", rqfc, output);

  ::glow::optimize(F_, CompilationMode::Infer);

  llvm::DenseMap<Node *, ParallelTransformKind> parOpts;
  parOpts[rqfc] = ParallelTransformKind::Data;

  std::unordered_map<Node *, ConcatNode *> replacedMap;
  ASSIGN_VALUE_OR_FAIL_TEST(
      replacedMap,
      ::glow::parallelizeOps(F_, llvm::DenseMap<Node *, size_t>(), parOpts, 3));
  EXPECT_EQ(replacedMap.size(), parOpts.size());
  runDCEPass(F_, cctx_);

  // We now have 3 RowwiseQuantizedFullyConnecteds
  EXPECT_EQ(3, countNodeKind(
                   F_, Kinded::Kind::RowwiseQuantizedFullyConnectedNodeKind));

  // One concat to bring all of the parallelized sliced
  // RowwiseQuantizedFullyConnecteds together.
  EXPECT_EQ(1, countNodeKind(F_, Kinded::Kind::ConcatNodeKind));
}

/// Test Splitting Convolution into multiple Convolutions.
TEST_F(GraphOptz, ParallelizeGraph_Convolution_Model_Axis3) {
  auto *input1 = mod_.createPlaceholder(ElemKind::Int8QTy, {3, 5, 5, 8}, 1.0, 0,
                                        "input1", false);
  bindings_.allocate(input1)->getHandle<int8_t>().randomize(-4, 4,
                                                            mod_.getPRNG());
  auto *filter = mod_.createPlaceholder(ElemKind::Int8QTy, {12, 1, 1, 8}, 0.1,
                                        0, "weights", false);
  auto *bias =
      mod_.createPlaceholder(ElemKind::Int32QTy, {12}, 0.01, 0, "bias", false);

  auto *output = mod_.createPlaceholder(ElemKind::Int8QTy, {3, 5, 5, 12}, 1.0,
                                        0, "output", false);
  bindings_.allocate(output);
  auto outTy = mod_.uniqueType(ElemKind::Int8QTy, {3, 5, 5, 12}, 1.0, 0);

  auto *c = F_->createConv("Convolution1", input1, filter, bias, outTy, {1, 1},
                           {1, 1}, {0, 0, 0, 0}, 1);
  F_->createSave("save", c, output);

  ::glow::optimize(F_, CompilationMode::Infer);

  // This is F_ but without the parallel transformation below.
  optimizedF_ = F_->clone(F_->getName().str() + "_optimized");

  llvm::DenseMap<Node *, ParallelTransformKind> parOpts;
  parOpts[c] = ParallelTransformKind::Model_Axis3;

  std::unordered_map<Node *, ConcatNode *> replacedMap;
  ASSIGN_VALUE_OR_FAIL_TEST(
      replacedMap, ::glow::parallelizeOps(F_, llvm::DenseMap<Node *, size_t>(),
                                          parOpts, 12));
  EXPECT_EQ(replacedMap.size(), parOpts.size());
  runDCEPass(F_, cctx_);

  // We now have 12 Convolutions
  EXPECT_EQ(12, countNodeKind(F_, Kinded::Kind::ConvolutionNodeKind));

  // One concat to bring all of the parallelized sliced
  // ChannelwiseQuantizedConvolutions together.
  EXPECT_EQ(1, countNodeKind(F_, Kinded::Kind::ConcatNodeKind));

  checkNumericalEquivalence(0.f);
}

/// Test Splitting Convolution3D into multiple Convolution3Ds.
TEST_F(GraphOptz, ParallelizeGraph_Convolution3D_Model_Axis4) {
  auto *input1 = mod_.createPlaceholder(ElemKind::Int8QTy, {3, 5, 5, 5, 8}, 1.0,
                                        0, "input1", false);
  bindings_.allocate(input1)->getHandle<int8_t>().randomize(-4, 4,
                                                            mod_.getPRNG());
  auto *filter = mod_.createPlaceholder(ElemKind::Int8QTy, {12, 1, 1, 1, 8},
                                        0.1, 0, "weights", false);
  auto *bias =
      mod_.createPlaceholder(ElemKind::Int32QTy, {12}, 0.01, 0, "bias", false);

  auto *output = mod_.createPlaceholder(ElemKind::Int8QTy, {3, 5, 5, 5, 12},
                                        1.0, 0, "output", false);
  bindings_.allocate(output);
  auto outTy = mod_.uniqueType(ElemKind::Int8QTy, {3, 5, 5, 5, 12}, 1.0, 0);

  auto *c3d = F_->createConv3D("Convolution3D1", input1, filter, bias, outTy,
                               {1, 1, 1}, {1, 1, 1}, {0, 0, 0, 0, 0, 0}, 1);
  F_->createSave("save", c3d, output);

  ::glow::optimize(F_, CompilationMode::Infer);

  // This is F_ but without the parallel transformation below.
  optimizedF_ = F_->clone(F_->getName().str() + "_optimized");

  llvm::DenseMap<Node *, ParallelTransformKind> parOpts;
  parOpts[c3d] = ParallelTransformKind::Model_Axis4;

  std::unordered_map<Node *, ConcatNode *> replacedMap;
  ASSIGN_VALUE_OR_FAIL_TEST(
      replacedMap, ::glow::parallelizeOps(F_, llvm::DenseMap<Node *, size_t>(),
                                          parOpts, 12));
  EXPECT_EQ(replacedMap.size(), parOpts.size());
  runDCEPass(F_, cctx_);

  // We now have 12 Convolution3Ds
  EXPECT_EQ(12, countNodeKind(F_, Kinded::Kind::Convolution3DNodeKind));

  // One concat to bring all of the parallelized sliced
  // ChannelwiseQuantizedConvolutions together.
  EXPECT_EQ(1, countNodeKind(F_, Kinded::Kind::ConcatNodeKind));

  checkNumericalEquivalence(0.f);
}

/// Test Splitting AvgPool into multiple AvgPools.
TEST_F(GraphOptz, ParallelizeGraph_AvgPool_Model_Axis4) {
  auto *input1 = mod_.createPlaceholder(ElemKind::FloatTy, {3, 5, 5, 5, 8},
                                        "input1", false);
  bindings_.allocate(input1)->getHandle<float>().randomize(-1.0, 1.0,
                                                           mod_.getPRNG());
  auto *output = mod_.createPlaceholder(ElemKind::FloatTy, {3, 1, 1, 1, 8},
                                        "output", false);
  bindings_.allocate(output);

  auto *ap = F_->createAvgPool("AvgPool1", input1, {5, 5, 5}, {1, 1, 1},
                               {0, 0, 0, 0, 0, 0}, ConvolutionLayout::NTHWC);
  F_->createSave("save", ap, output);

  ::glow::optimize(F_, CompilationMode::Infer);

  // This is F_ but without the parallel transformation below.
  optimizedF_ = F_->clone(F_->getName().str() + "_optimized");

  llvm::DenseMap<Node *, ParallelTransformKind> parOpts;
  parOpts[ap] = ParallelTransformKind::Model_Axis4;

  std::unordered_map<Node *, ConcatNode *> replacedMap;
  ASSIGN_VALUE_OR_FAIL_TEST(
      replacedMap,
      ::glow::parallelizeOps(F_, llvm::DenseMap<Node *, size_t>(), parOpts, 8));
  EXPECT_EQ(replacedMap.size(), parOpts.size());
  runDCEPass(F_, cctx_);

  // We now have 8 AvgPools
  EXPECT_EQ(8, countNodeKind(F_, Kinded::Kind::AvgPoolNodeKind));

  // One concat to bring all of the parallelized sliced AvgPools
  // together.
  EXPECT_EQ(1, countNodeKind(F_, Kinded::Kind::ConcatNodeKind));

  checkNumericalEquivalence(0.f);
}

/// Test that Add after ConvTranspose is folded into Bias add when the actual
/// Add is is a broadcast of the bias. Test \p RnL (right of left) side add.
static void foldConvTransposeAddIntoBiasAdd(PlaceholderBindings &bindings,
                                            Module &mod, Function *F,
                                            Function *&optF, bool RnL) {
  dim_t batch = 2;
  dim_t inC = 2;
  dim_t outC = 5;
  dim_t inH = 3;
  dim_t inW = 3;
  unsigned_t kernel = 3;
  std::vector<uint32_t> pads = {0, 0, 0, 0};
  std::vector<uint32_t> stride = {1, 1};

  auto *input = mod.createPlaceholder(ElemKind::FloatTy, {2, inH, inW, inC},
                                      "input", false);
  auto *filter = mod.createPlaceholder(
      ElemKind::FloatTy, {outC, kernel, kernel, inC}, "filter", false);

  auto *bias = mod.createConstant(ElemKind::FloatTy, {outC}, "bias");
  bias->getPayloadMutable().getHandle<float>() = {1, 3, 5, 7, 9};

  std::pair<dim_t, dim_t> outHW = calculateConvTransposeOutputDims(
      inH, inW, {kernel, kernel}, stride, pads);
  auto outTy = mod.uniqueType(ElemKind::FloatTy,
                              {batch, outHW.first, outHW.second, outC});

  ConvTransposeNode *CTN =
      F->createConvTranspose("ConvTranspose", input, filter, bias, outTy,
                             {kernel, kernel}, stride, {0, 0, 0, 0}, 1);

  auto *CN = mod.createConstant(ElemKind::FloatTy,
                                {batch, outHW.first, outHW.second, outC}, "c1");
  auto *AN = RnL ? F->createAdd("add", CN, CTN) : F->createAdd("add", CTN, CN);

  CN->getPayloadMutable().getHandle<float>() = {
      1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3,
      4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1,
      2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4,
      5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2,
      3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
      1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3,
      4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1,
      2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4,
      5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2,
      3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
      1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5};

  SaveNode *save = F->createSave("save", AN);
  bindings.allocate(save->getPlaceholder());

  EXPECT_EQ(F->getNodes().size(), 3);
  optF = optimizeFunctionForTest(F);
  EXPECT_EQ(optF->getNodes().size(), 2);

  const SaveNode *optSave =
      findFunctionNodeByName<SaveNode>(optF, save->getName());

  ConvTransposeNode *optCN =
      llvm::dyn_cast<ConvTransposeNode>(optSave->getInput());
  EXPECT_TRUE(optCN);

  Constant *optBias = llvm::dyn_cast<Constant>(optCN->getBias());
  EXPECT_TRUE(optBias);

  auto BH = optBias->getPayload().getHandle();
  EXPECT_EQ(BH.raw(0), 1 + 1);
  EXPECT_EQ(BH.raw(1), 2 + 3);
  EXPECT_EQ(BH.raw(2), 3 + 5);
  EXPECT_EQ(BH.raw(3), 4 + 7);
  EXPECT_EQ(BH.raw(4), 5 + 9);

  bindings.allocate(mod.getPlaceholders());
  bindings.get(input)->getHandle().randomize(-1.0, 1.0, mod.getPRNG());
  bindings.get(filter)->getHandle().randomize(-1.0, 1.0, mod.getPRNG());
}

/// Test that Add after ConvTranspose is folded into Bias add when the actual
/// Add is is a broadcast of the bias.
TEST_F(GraphOptz, FoldConvTransposeAddIntoBiasAddRHS) {
  foldConvTransposeAddIntoBiasAdd(bindings_, mod_, F_, optimizedF_, false);
  checkNumericalEquivalence();
}
TEST_F(GraphOptz, FoldConvTransposeAddIntoBiasAddLHS) {
  foldConvTransposeAddIntoBiasAdd(bindings_, mod_, F_, optimizedF_, true);
  checkNumericalEquivalence();
}

/// Test that MatMul + Add is folded into FullyConnected.
TEST_F(GraphOptz, FoldMatMulAddIntoFullyConnected) {

  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 3}, "input", false);
  auto *weights =
      mod_.createPlaceholder(ElemKind::FloatTy, {3, 5}, "weights", false);
  auto *bias = mod_.createPlaceholder(ElemKind::FloatTy, {1, 5}, "bias", false);

  MatMulNode *matmul = F_->createMatMul("matmul", input, weights);
  AddNode *add = F_->createAdd("add", matmul, bias);
  F_->createSave("save", add);
  EXPECT_EQ(3, F_->getNodes().size());

  // The folding should replace the MatMul + Add into a FullyConnected and a
  // Reshape to 1D for the Bias.
  CompilationContext cctx;
  ::glow::fold(F_, cctx);
  EXPECT_EQ(3, F_->getNodes().size());
  EXPECT_EQ(0, countNodeKind(F_, Kinded::Kind::AddNodeKind));
  EXPECT_EQ(0, countNodeKind(F_, Kinded::Kind::MatMulNodeKind));
  EXPECT_EQ(1, countNodeKind(F_, Kinded::Kind::FullyConnectedNodeKind));
  EXPECT_EQ(1, countNodeKind(F_, Kinded::Kind::ReshapeNodeKind));
}

/// Test that batched MatMul + Add is folded into batched FullyConnected.
/// This optimization takes place only if the Bias is constant and the
/// bias data repeats for all the batches.
TEST_F(GraphOptz, FoldMatMulAddIntoFullyConnectedBatched) {

  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 3}, "input", false);
  auto *weights =
      mod_.createPlaceholder(ElemKind::FloatTy, {3, 5}, "weights", false);
  auto *bias = mod_.createConstant(ElemKind::FloatTy, {2, 5}, "bias");
  auto biasH = bias->getPayloadMutable().getHandle<float>();
  biasH = {1, 2, 3, 4, 5, 1, 2, 3, 4, 5};

  MatMulNode *matmul = F_->createMatMul("matmul", input, weights);
  AddNode *add = F_->createAdd("add", matmul, bias);
  F_->createSave("save", add);
  EXPECT_EQ(3, F_->getNodes().size());

  // The folding should replace the MatMul + Add into a FullyConnected and a
  // Reshape to 1D for the Bias.
  CompilationContext cctx;
  ::glow::fold(F_, cctx);
  EXPECT_EQ(4, F_->getNodes().size());
  EXPECT_EQ(0, countNodeKind(F_, Kinded::Kind::AddNodeKind));
  EXPECT_EQ(0, countNodeKind(F_, Kinded::Kind::MatMulNodeKind));
  EXPECT_EQ(1, countNodeKind(F_, Kinded::Kind::FullyConnectedNodeKind));
  EXPECT_EQ(1, countNodeKind(F_, Kinded::Kind::SliceNodeKind));
  EXPECT_EQ(1, countNodeKind(F_, Kinded::Kind::ReshapeNodeKind));
}

/// Test that MatMul is converted to FullyConnected for Int8QTy.
TEST_F(GraphOptz, ConvertMatMulToFullyConnected_Int8QTy) {

  auto *input = mod_.createPlaceholder(ElemKind::Int8QTy, {1, 3}, 0.1f, -13,
                                       "input", false);
  auto *weights = mod_.createPlaceholder(ElemKind::Int8QTy, {3, 5}, 0.2f, 15,
                                         "weights", false);
  MatMulNode *matmul = F_->createMatMul("matmul", input, weights);
  F_->createSave("save", matmul);
  EXPECT_EQ(2, F_->getNodes().size());

  optimizedF_ = optimizeFunctionForTest(
      F_, {FunctionPassID::ConvertMatMulToFullyConnected, getDCEPassConfig()});

  EXPECT_EQ(2, optimizedF_->getNodes().size());
  EXPECT_EQ(1,
            countNodeKind(optimizedF_, Kinded::Kind::FullyConnectedNodeKind));
}

/// Test that MatMul is converted to FullyConnected for FloatTy.
TEST_F(GraphOptz, ConvertMatMulToFullyConnected_FloatTy) {

  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 3}, "input", false);
  auto *weights =
      mod_.createPlaceholder(ElemKind::FloatTy, {3, 5}, "weights", false);
  MatMulNode *matmul = F_->createMatMul("matmul", input, weights);
  F_->createSave("save", matmul);
  EXPECT_EQ(2, F_->getNodes().size());

  optimizedF_ = optimizeFunctionForTest(
      F_, {FunctionPassID::ConvertMatMulToFullyConnected, getDCEPassConfig()});

  EXPECT_EQ(2, optimizedF_->getNodes().size());
  EXPECT_EQ(1,
            countNodeKind(optimizedF_, Kinded::Kind::FullyConnectedNodeKind));
}

/// Test that FoldSlicesIntoConstants pass works as expected.
TEST_F(GraphOptz, FoldSlicesIntoConstantsTest) {
  Constant *C = mod_.createConstant(ElemKind::FloatTy, {3, 4}, "C");
  auto CH = C->getPayloadMutable().getHandle<float>();
  CH = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

  SliceNode *S1 = F_->createSlice("s1", C, {0, 0}, {3, 2});
  SliceNode *S2 = F_->createSlice("s2", C, {0, 2}, {3, 4});
  SaveNode *SN1 = F_->createSave("save1", S1);
  SaveNode *SN2 = F_->createSave("save2", S2);

  optimizedF_ = optimizeFunctionForTest(
      F_, {FunctionPassID::FoldSlicesIntoConstants, getDCEPassConfig()});

  SaveNode *optSN1 =
      llvm::dyn_cast<SaveNode>(optimizedF_->getNodeByName(SN1->getName()));
  SaveNode *optSN2 =
      llvm::dyn_cast<SaveNode>(optimizedF_->getNodeByName(SN2->getName()));
  ASSERT_TRUE(optSN1);
  ASSERT_TRUE(optSN2);

  Constant *C1 = llvm::dyn_cast<Constant>(optSN1->getInput());
  ASSERT_TRUE(C1);
  auto H1 = C1->getPayloadMutable().getHandle();
  Constant *C2 = llvm::dyn_cast<Constant>(optSN2->getInput());
  ASSERT_TRUE(C2);
  auto H2 = C2->getPayloadMutable().getHandle();
  for (dim_t i = 0, e = 3; i < e; i++) {
    for (dim_t j = 0, e = 2; j < e; j++) {
      EXPECT_EQ(H1.at({i, j}), CH.at({i, j}));
      EXPECT_EQ(H2.at({i, j}), CH.at({i, j + 2}));
    }
  }
}

/// Test that RaiseClipsAboveShapeNodes pass works as expected.
TEST_F(GraphOptz, RaiseClipsAboveShapeNodesTest) {
  Placeholder *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {256, 64}, "input", false);

  ReshapeNode *RN1 = F_->createReshape("reshape1", input, {4, 128, 32});
  ReshapeNode *RN2 = F_->createReshape("reshape2", RN1, {64, 256});
  TransposeNode *TN = F_->createTranspose("transpose", RN2, {1, 0});
  SliceNode *SN = F_->createSlice("slice", TN, {64, 0}, {256, 64});
  TileNode *TiN = F_->createTile("tile", SN, 2, 0);
  ClipNode *CN = F_->createClip("clip", TiN, -0.1, 0.1);
  SaveNode *save1 = F_->createSave("save1", RN1);
  SaveNode *save2 = F_->createSave("save2", CN);

  optimizedF_ =
      optimizeFunctionForTest(F_, {FunctionPassID::RaiseClipsAboveShapeNodes});

  auto *optSave1 =
      llvm::dyn_cast<SaveNode>(optimizedF_->getNodeByName(save1->getName()));
  ASSERT_TRUE(optSave1);
  auto *optSave2 =
      llvm::dyn_cast<SaveNode>(optimizedF_->getNodeByName(save2->getName()));
  ASSERT_TRUE(optSave2);

  // save1 should only have a single untouched Reshape RN1 input which has input
  // input into it, because RN1 has multiple users.
  auto *optRN1 = llvm::dyn_cast<ReshapeNode>(optSave1->getInput().getNode());
  ASSERT_TRUE(optRN1);
  EXPECT_EQ(input, optRN1->getInput().getNode());

  // save2 should have CN it originally saved pushed up above SN, TiN, TN, and
  // RN2.
  TileNode *newTiN = llvm::dyn_cast<TileNode>(optSave2->getInput());
  ASSERT_TRUE(newTiN);
  EXPECT_EQ(newTiN->getCount(), TiN->getCount());
  SliceNode *newSN = llvm::dyn_cast<SliceNode>(newTiN->getInput());
  ASSERT_TRUE(newSN);
  EXPECT_EQ(newSN->getStart(), SN->getStart());
  TransposeNode *newTN = llvm::dyn_cast<TransposeNode>(newSN->getInput());
  ASSERT_TRUE(newTN);
  EXPECT_EQ(newTN->getShuffle(), TN->getShuffle());
  ReshapeNode *newRN2 = llvm::dyn_cast<ReshapeNode>(newTN->getInput());
  ASSERT_TRUE(newRN2);
  ClipNode *newCN = llvm::dyn_cast<ClipNode>(newRN2->getInput());
  ASSERT_TRUE(newCN);
  EXPECT_EQ(newCN->getMin(), CN->getMin());
  EXPECT_EQ(newCN->getMax(), CN->getMax());

  bindings_.allocate(mod_.getPlaceholders());
  bindings_.get(input)->getHandle().randomize(-1.0, 1.0, mod_.getPRNG());
  checkNumericalEquivalence();
}

static void testOptimizeDequantizeClip(PlaceholderBindings &bindings,
                                       Module &mod, Function *F,
                                       Function *&optF,
                                       bool enableQuantParamChanges) {
  Placeholder *input =
      mod.createPlaceholder(ElemKind::FloatTy, {20, 20}, "input", false);

  const auto qParams = quantization::chooseQuantizationParams({-0.1, 0.1});

  QuantizeNode *QN =
      F->createQuantize("quantize", input,
                        mod.uniqueType(ElemKind::Int8QTy, {20, 20},
                                       qParams.scale, qParams.offset));
  DequantizeNode *DN = F->createDequantize("dequantize", QN, ElemKind::FloatTy);
  ClipNode *CN =
      F->createClip("clip", DN, enableQuantParamChanges ? 0 : -100, 100);
  SaveNode *SN = F->createSave("save", CN);

  CompilationContext cctx;
  cctx.optimizationOpts.enableQuantParamChanges = true;
  optF = optimizeFunctionForTest(
      F, {FunctionPassID::OptimizeQuantizeClip, getDCEPassConfig()}, cctx);

  EXPECT_EQ(countNodeKind(optF, Kinded::Kind::ClipNodeKind), 0);

  SaveNode *optSN =
      llvm::dyn_cast<SaveNode>(optF->getNodeByName(SN->getName()));
  ASSERT_TRUE(optSN);

  // Now check that the quantization params have been correctly updated for QN,
  // and that CN has been eliminated.
  DequantizeNode *optDN =
      llvm::dyn_cast<DequantizeNode>(optSN->getInput().getNode());
  ASSERT_TRUE(optDN);
  const auto qMinMax = optDN->getInput().getType()->getQuantizedValueRange();
  // Min is either from Clip or Quant range depending on enableQuantParamChanges
  EXPECT_NEAR(qMinMax.first, enableQuantParamChanges ? 0 : -0.1, 1E-3);
  EXPECT_NEAR(qMinMax.second, 0.1, 1E-3); // Max from Quant range

  bindings.allocate(mod.getPlaceholders());
  bindings.get(input)->getHandle().randomize(-1.0, 1.0, mod.getPRNG());
}

/// Test that OptimizeQuantizeClip pass works as expected for Clip(Dequantize)
/// when the quantization parameters are allowed to change.
TEST_F(GraphOptz, OptimizeDequantizeClipTest_QuantParamChanges) {
  testOptimizeDequantizeClip(bindings_, mod_, F_, optimizedF_,
                             /* enableQuantParamChanges */ true);
  checkNumericalEquivalence(0.0005);
}

/// Test that OptimizeQuantizeClip pass works as expected for Clip(Dequantize)
/// when the quantization parameters are not allowed to change.
TEST_F(GraphOptz, OptimizeDequantizeClipTest_NoQuantParamChanges) {
  testOptimizeDequantizeClip(bindings_, mod_, F_, optimizedF_,
                             /* enableQuantParamChanges */ false);
  checkNumericalEquivalence();
}

static void testOptimizeClipQuantize(PlaceholderBindings &bindings, Module &mod,
                                     Function *F, Function *&optF,
                                     bool enableQuantParamChanges) {
  Placeholder *input =
      mod.createPlaceholder(ElemKind::FloatTy, {20, 20}, "input", false);

  const auto qParams = quantization::chooseQuantizationParams({-0.1, 0.1});

  ClipNode *CN =
      F->createClip("clip", input, enableQuantParamChanges ? 0 : -100, 100);
  QuantizeNode *QN =
      F->createQuantize("quantize", CN,
                        mod.uniqueType(ElemKind::Int8QTy, {20, 20},
                                       qParams.scale, qParams.offset));
  DequantizeNode *DN = F->createDequantize("dequantize", QN, ElemKind::FloatTy);
  SaveNode *SN = F->createSave("save", DN);

  CompilationContext cctx;
  cctx.optimizationOpts.enableQuantParamChanges = enableQuantParamChanges;
  optF = optimizeFunctionForTest(
      F, {FunctionPassID::OptimizeQuantizeClip, getDCEPassConfig()}, cctx);

  EXPECT_EQ(countNodeKind(optF, Kinded::Kind::ClipNodeKind), 0);

  SaveNode *optSN =
      llvm::dyn_cast<SaveNode>(optF->getNodeByName(SN->getName()));
  ASSERT_TRUE(optSN);

  // Now check that the quantization params have been correctly updated for QN,
  // and that CN has been eliminated.
  DequantizeNode *optDN =
      llvm::dyn_cast<DequantizeNode>(optSN->getInput().getNode());
  ASSERT_TRUE(optDN);
  const auto qMinMax = optDN->getInput().getType()->getQuantizedValueRange();
  // Min is either from Clip or Quant range depending on enableQuantParamChanges
  EXPECT_NEAR(qMinMax.first, enableQuantParamChanges ? 0 : -0.1, 1E-3);
  EXPECT_NEAR(qMinMax.second, 0.1, 1E-3); // Max always from Quant range

  bindings.allocate(mod.getPlaceholders());
  bindings.get(input)->getHandle().randomize(-1.0, 1.0, mod.getPRNG());
}

/// Test that OptimizeQuantizeClip pass works as expected for Clip(Quantize)
/// when the quantization parameters are allowed to change.
TEST_F(GraphOptz, OptimizeClipQuantizeTest_QuantParamChanges) {
  testOptimizeClipQuantize(bindings_, mod_, F_, optimizedF_,
                           /* enableQuantParamChanges */ true);
  checkNumericalEquivalence(0.0005);
}

/// Test that OptimizeQuantizeClip pass works as expected for Clip(Quantize)
/// when the quantization parameters are not allowed to change.
TEST_F(GraphOptz, OptimizeClipQuantizeTest_NoQuantParamChanges) {
  testOptimizeClipQuantize(bindings_, mod_, F_, optimizedF_,
                           /* enableQuantParamChanges */ false);
  checkNumericalEquivalence();
}

/// Test Quantize(ConvertTo(Node)) -> Quantize(Node), where Quantize is int8.
TEST_F(GraphOptz, OptimizeOutIntermediateConversionsTest) {
  Placeholder *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {20, 20}, "input", false);

  const auto qParams = quantization::chooseQuantizationParams({-0.1, 0.1});

  ConvertToNode *CN = F_->createConvertTo("conv", input, ElemKind::Float16Ty);
  QuantizeNode *QN =
      F_->createQuantize("quantize", CN,
                         mod_.uniqueType(ElemKind::Int8QTy, {20, 20},
                                         qParams.scale, qParams.offset));
  DequantizeNode *DN =
      F_->createDequantize("dequantize", QN, ElemKind::FloatTy);
  F_->createSave("save", DN);

  optimizedF_ = optimizeFunctionForTest(
      F_,
      {FunctionPassID::OptimizeOutIntermediateConversions, getDCEPassConfig()});

  // Now check that the ConvertToNode has been eliminated.
  EXPECT_EQ(countNodeKind(optimizedF_, Kinded::Kind::ConvertToNodeKind), 0);

  bindings_.allocate(mod_.getPlaceholders());
  bindings_.get(input)->getHandle().randomize(-1.0, 1.0, mod_.getPRNG());
  checkNumericalEquivalence();
}

/// Test Clip(Relu(Clip)) -> Clip'.
TEST_F(GraphOptz, ClipReluClipElimTest) {
  Placeholder *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {64, 64}, "input", false);
  ClipNode *CN1 = F_->createClip("CN1", input, -10, 30);
  ReluNode *RN = F_->createRELU("RN", CN1);
  ClipNode *CN2 = F_->createClip("CN2", RN, -5, 20);
  SaveNode *SN = F_->createSave("save", CN2);

  // Start with 2 clips, a relu, and a save.
  EXPECT_EQ(F_->getNodes().size(), 4);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::ClipNodeKind), 2);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::ReluNodeKind), 1);

  optimizedF_ = optimizeFunctionForTest(F_);

  // Remove one of the clips and the relu.
  EXPECT_EQ(optimizedF_->getNodes().size(), 2);
  EXPECT_EQ(countNodeKind(optimizedF_, Kinded::Kind::ClipNodeKind), 1);
  EXPECT_EQ(countNodeKind(optimizedF_, Kinded::Kind::ReluNodeKind), 0);

  SaveNode *optSN =
      llvm::dyn_cast<SaveNode>(optimizedF_->getNodeByName(SN->getName()));
  ASSERT_TRUE(optSN);

  // We combined all of the ranges into the single Clip.
  ClipNode *optCN = llvm::dyn_cast<ClipNode>(optSN->getInput());
  ASSERT_TRUE(optCN);
  EXPECT_EQ(optCN->getMin(), 0);
  EXPECT_EQ(optCN->getMax(), 20);

  bindings_.allocate(input)->getHandle().randomize(-50.0, 5.0, mod_.getPRNG());
  checkNumericalEquivalence();
}

/// Test that we can find a non-quantized relu and fuse it up into a quant FC.
TEST_F(GraphOptz, OptimizeQuantFCFloatReluTest) {
  auto *input = mod_.createPlaceholder(ElemKind::Int8QTy, {2, 32}, 1.0, 0,
                                       "input", false);
  auto *weights =
      mod_.createConstant(ElemKind::Int8QTy, {32, 32}, 1.0, 0, "weights");
  auto *bias = mod_.createConstant(ElemKind::Int32QTy, {32}, 1.0, 0, "bias");

  auto *FC = F_->createFullyConnected("fc", input, weights, bias);
  auto *DN = F_->createDequantize("dq", FC, ElemKind::FloatTy);
  auto *RN = F_->createRELU("relu", DN);
  auto *SN = F_->createSave("save", RN);

  optimizedF_ = optimizeFunctionForTest(
      F_, {FunctionPassID::OptimizeQuantFCFloatRelu, getDCEPassConfig()});

  SaveNode *optSN =
      llvm::dyn_cast<SaveNode>(optimizedF_->getNodeByName(SN->getName()));
  ASSERT_TRUE(optSN);

  DequantizeNode *optDN = llvm::dyn_cast<DequantizeNode>(optSN->getInput());
  ASSERT_TRUE(optDN);
  ReluNode *optRN = llvm::dyn_cast<ReluNode>(optDN->getInput());
  ASSERT_TRUE(optRN);
  auto rangeRN = optRN->getResult().getType()->getQuantizedValueRange();
  EXPECT_EQ(rangeRN.first, 0.0f);
  FullyConnectedNode *optFC =
      llvm::dyn_cast<FullyConnectedNode>(optRN->getInput());
  ASSERT_TRUE(optFC);
  auto rangeFC = optFC->getResult().getType()->getQuantizedValueRange();
  EXPECT_EQ(rangeRN.second, rangeFC.second);

  bindings_.allocate(input)->getHandle<int8_t>().randomize(-128, 127,
                                                           mod_.getPRNG());
  weights->getPayloadMutable().getHandle<int8_t>().randomize(-128, 127,
                                                             mod_.getPRNG());
  bias->getPayloadMutable().getHandle<int32_t>().randomize(-128, 127,
                                                           mod_.getPRNG());
  checkNumericalEquivalence();
}

/// Test that we can find a non-quantized relu and fuse it up into a quant FC
/// even when setting dummy qparams to true.
TEST_F(GraphOptz, OptimizeDummyQuantFCFloatReluTest) {
  auto *input = mod_.createPlaceholder(ElemKind::Int8QTy, {2, 32}, 1.0, 0,
                                       "input", false);
  auto *weights =
      mod_.createConstant(ElemKind::Int8QTy, {32, 32}, 1.0, 0, "weights");
  auto *bias = mod_.createConstant(ElemKind::Int32QTy, {32}, 1.0, 0, "bias");
  auto *addW =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 32}, "addw", false);
  auto *FC = F_->createFullyConnected("fc", input, weights, bias);
  auto *DN = F_->createDequantize("dq", FC, ElemKind::FloatTy);
  auto *RN = F_->createRELU("relu", DN);
  auto *AN = F_->createAdd("add", RN, addW);
  auto *SN = F_->createSave("save", AN);

  CompilationContext cctx;
  cctx.precisionConfig.loadUniquedDummyQParams = true;
  optimizedF_ = optimizeFunctionForTest(
      F_, {FunctionPassID::OptimizeQuantFCFloatRelu, getDCEPassConfig()}, cctx);

  SaveNode *optSN =
      llvm::dyn_cast<SaveNode>(optimizedF_->getNodeByName(SN->getName()));
  ASSERT_TRUE(optSN);

  AddNode *optAN = llvm::dyn_cast<AddNode>(optSN->getInput());
  ASSERT_TRUE(optAN);
  DequantizeNode *optDN = llvm::dyn_cast<DequantizeNode>(optAN->getLHS());
  ASSERT_TRUE(optDN);
  ReluNode *optRN = llvm::dyn_cast<ReluNode>(optDN->getInput());
  ASSERT_TRUE(optRN);
  auto rangeRN = optRN->getResult().getType()->getQuantizedValueRange();
  FullyConnectedNode *optFC =
      llvm::dyn_cast<FullyConnectedNode>(optRN->getInput());
  ASSERT_TRUE(optFC);
  auto rangeFC = optFC->getResult().getType()->getQuantizedValueRange();
  EXPECT_EQ(rangeRN.first, rangeFC.first);
  EXPECT_EQ(rangeRN.second, rangeFC.second);

  bindings_.allocate(input)->getHandle<int8_t>().randomize(-128, 127,
                                                           mod_.getPRNG());
  bindings_.allocate(addW)->getHandle<float>().randomize(-128, 127,
                                                         mod_.getPRNG());
  weights->getPayloadMutable().getHandle<int8_t>().randomize(-128, 127,
                                                             mod_.getPRNG());
  bias->getPayloadMutable().getHandle<int32_t>().randomize(-128, 127,
                                                           mod_.getPRNG());
  checkNumericalEquivalence();
}

/// Test that we can find a non-quantized relu and fuse it up into a series of
/// concatenated quant FCs.
TEST_F(GraphOptz, OptimizeConcatQuantFCFloatReluTest) {
  std::array<NodeValue, 5> DQs;
  for (size_t i = 0; i < 5; i++) {
    auto *input = mod_.createPlaceholder(ElemKind::Int8QTy, {2, 32},
                                         1.0 / (i + 1), 0, "input", false);
    auto *weights =
        mod_.createConstant(ElemKind::Int8QTy, {32, 32}, 1.0, 0, "weights");
    auto *bias = mod_.createConstant(ElemKind::Int32QTy, {32}, 1.0, 0, "bias");

    auto *FC = F_->createFullyConnected("fc", input, weights, bias);
    DQs[i] = F_->createDequantize("dq", FC, ElemKind::FloatTy)->getResult();

    bindings_.allocate(input)->getHandle<int8_t>().randomize(-128, 127,
                                                             mod_.getPRNG());
    weights->getPayloadMutable().getHandle<int8_t>().randomize(-128, 127,
                                                               mod_.getPRNG());
    bias->getPayloadMutable().getHandle<int32_t>().randomize(-128, 127,
                                                             mod_.getPRNG());
  }

  auto *CN = F_->createConcat("concat", DQs, 0);
  auto *RN = F_->createRELU("relu", CN);
  auto *SN = F_->createSave("save", RN);

  optimizedF_ = optimizeFunctionForTest(
      F_, {FunctionPassID::OptimizeQuantFCFloatRelu, getDCEPassConfig()});

  SaveNode *optSN =
      llvm::dyn_cast<SaveNode>(optimizedF_->getNodeByName(SN->getName()));
  ASSERT_TRUE(optSN);
  ConcatNode *optCN = llvm::dyn_cast<ConcatNode>(optSN->getInput());
  ASSERT_TRUE(optCN);
  EXPECT_EQ(optCN->getInputs().size(), 5);

  for (const NodeValue &NV : optCN->getInputs()) {
    DequantizeNode *optDN = llvm::dyn_cast<DequantizeNode>(NV);
    ASSERT_TRUE(optDN);
    ReluNode *optRN = llvm::dyn_cast<ReluNode>(optDN->getInput());
    ASSERT_TRUE(optRN);
    auto rangeRN = optRN->getResult().getType()->getQuantizedValueRange();
    EXPECT_EQ(rangeRN.first, 0.0f);
    FullyConnectedNode *optFC =
        llvm::dyn_cast<FullyConnectedNode>(optRN->getInput());
    ASSERT_TRUE(optFC);
    auto rangeFC = optFC->getResult().getType()->getQuantizedValueRange();
    EXPECT_EQ(rangeRN.second, rangeFC.second);
  }

  checkNumericalEquivalence();
}

/// Test that we can find a concat with all dequantize inputs and a quantize at
/// its output, and then replace quant/dequants with rescales.
TEST_F(GraphOptz, OptimizeDequantConcatQuant) {
  std::array<NodeValue, 5> DQs;
  std::array<Placeholder *, 5> inputs;
  for (size_t i = 0; i < 5; i++) {
    inputs[i] = mod_.createPlaceholder(ElemKind::Int8QTy, {2, 32},
                                       0.3 / (i + 1), 5, "input", false);
    DQs[i] =
        F_->createDequantize("dq", inputs[i], ElemKind::FloatTy)->getResult();

    bindings_.allocate(inputs[i])->getHandle<int8_t>().randomize(
        -128, 127, mod_.getPRNG());
  }

  auto *CN = F_->createConcat("concat", DQs, 0);
  constexpr float scale = 0.3;
  constexpr int32_t offset = 5;
  auto *RN = F_->createQuantize("quantize", CN,
                                mod_.uniqueType(ElemKind::Int8QTy,
                                                CN->getResult().dims(), scale,
                                                offset));
  auto *SN = F_->createSave("save", RN);

  optimizedF_ = optimizeFunctionForTest(
      F_, {FunctionPassID::OptimizeConcatQuantization, getDCEPassConfig()});

  SaveNode *optSN =
      llvm::dyn_cast<SaveNode>(optimizedF_->getNodeByName(SN->getName()));
  ASSERT_TRUE(optSN);
  ConcatNode *optCN = llvm::dyn_cast<ConcatNode>(optSN->getInput());
  ASSERT_TRUE(optCN);
  EXPECT_EQ(optCN->getInputs().size(), 5);

  for (size_t i = 0, e = optCN->getInputs().size(); i < e; i++) {
    const NodeValue NV = optCN->getInputs()[i];
    if (i == 0) {
      EXPECT_EQ(inputs[i], NV.getNode());
      EXPECT_EQ(inputs[i]->getOutput().getType()->getScale(), scale);
      EXPECT_EQ(inputs[i]->getOutput().getType()->getOffset(), offset);
    } else {
      RescaleQuantizedNode *optRN = llvm::dyn_cast<RescaleQuantizedNode>(NV);
      ASSERT_TRUE(optRN);
      EXPECT_EQ(optRN->getResult().getType()->getScale(), scale);
      EXPECT_EQ(optRN->getResult().getType()->getOffset(), offset);
      EXPECT_EQ(inputs[i], optRN->getInput().getNode());
    }
  }
  checkNumericalEquivalence();
}

/// Test that if we have a Concat with all Dequantize inputs with the same
/// scale/offset/kind that we can sink the Dequantizes below the Concat.
TEST_F(GraphOptz, SinkDequantizeBelowConcatTest) {
  const float scale = 0.06;
  const int32_t offset = -15;
  std::array<NodeValue, 5> inputs;
  for (dim_t i = 0; i < 5; i++) {
    Placeholder *input = mod_.createPlaceholder(ElemKind::Int8QTy, {i + 1, 100},
                                                scale, offset, "input", false);
    bindings_.allocate(input)->getHandle<int8_t>().randomize(-100, 100,
                                                             mod_.getPRNG());
    DequantizeNode *dequantize =
        F_->createDequantize("dequantize", input, ElemKind::Float16Ty);
    inputs[i] = dequantize->getResult();
  }
  ConcatNode *concat = F_->createConcat("concat", inputs, 0);
  SaveNode *SN = F_->createSave("ret", concat);

  optimizedF_ = optimizeFunctionForTest(
      F_, {FunctionPassID::SinkConversions, getDCEPassConfig()});

  // Concat, dequantize, save.
  EXPECT_EQ(optimizedF_->getNodes().size(), 3);
  EXPECT_EQ(countNodeKind(optimizedF_, Kinded::Kind::DequantizeNodeKind), 1);
  EXPECT_EQ(countNodeKind(optimizedF_, Kinded::Kind::ConcatNodeKind), 1);
  EXPECT_EQ(countNodeKind(optimizedF_, Kinded::Kind::SaveNodeKind), 1);

  SaveNode *optSN =
      llvm::dyn_cast<SaveNode>(optimizedF_->getNodeByName(SN->getName()));
  ASSERT_TRUE(optSN);
  DequantizeNode *optDequantize =
      llvm::dyn_cast<DequantizeNode>(optSN->getInput());
  ASSERT_TRUE(optDequantize);
  NodeValue input = optDequantize->getInput();
  EXPECT_EQ(scale, input.getType()->getScale());
  EXPECT_EQ(offset, input.getType()->getOffset());
  EXPECT_EQ(ElemKind::Int8QTy, input.getType()->getElementType());

  // Find dequantize node in the optimized graph.
  checkNumericalEquivalence();
}

/// Test that if we have a Concat with all Quantize inputs with the same
/// scale/offset/kind that we can sink the Dequantizes below the Concat.
TEST_F(GraphOptz, SinkQuantizeBelowConcatTest) {
  const float scale = 0.06;
  const int32_t offset = -15;
  std::array<NodeValue, 5> inputs;
  for (dim_t i = 0; i < 5; i++) {
    Placeholder *input = mod_.createPlaceholder(ElemKind::Float16Ty,
                                                {i + 1, 100}, "input", false);
    bindings_.allocate(input)->getHandle<float16_t>().randomize(-100, 100,
                                                                mod_.getPRNG());
    const TypeRef QTy = mod_.uniqueType(
        ElemKind::Int8QTy, input->getOutput().dims(), scale, offset);
    QuantizeNode *quantize = F_->createQuantize("quantize", input, QTy);
    inputs[i] = quantize->getResult();
  }
  ConcatNode *concat = F_->createConcat("concat", inputs, 0);
  SaveNode *SN = F_->createSave("ret", concat);

  optimizedF_ = optimizeFunctionForTest(
      F_, {FunctionPassID::SinkConversions, getDCEPassConfig()});

  // Concat, quantize, save.
  EXPECT_EQ(optimizedF_->getNodes().size(), 3);
  EXPECT_EQ(countNodeKind(optimizedF_, Kinded::Kind::QuantizeNodeKind), 1);
  EXPECT_EQ(countNodeKind(optimizedF_, Kinded::Kind::ConcatNodeKind), 1);
  EXPECT_EQ(countNodeKind(optimizedF_, Kinded::Kind::SaveNodeKind), 1);

  SaveNode *optSN =
      llvm::dyn_cast<SaveNode>(optimizedF_->getNodeByName(SN->getName()));
  ASSERT_TRUE(optSN);
  QuantizeNode *optQuantize = llvm::dyn_cast<QuantizeNode>(optSN->getInput());
  ASSERT_TRUE(optQuantize);
  EXPECT_EQ(scale, optQuantize->getResult().getType()->getScale());
  EXPECT_EQ(offset, optQuantize->getResult().getType()->getOffset());
  EXPECT_EQ(ElemKind::Int8QTy,
            optQuantize->getResult().getType()->getElementType());

  // Find quantize node in the optimized graph.
  checkNumericalEquivalence();
}

/// Test that if we have a Concat with all Tanh inputs,
/// we can sink the Tanh's below the Concat.
TEST_F(GraphOptz, SinkTanhBelowConcatTest) {
  std::array<NodeValue, 5> inputs;
  for (dim_t i = 0; i < 5; i++) {
    Placeholder *input = mod_.createPlaceholder(ElemKind::Float16Ty,
                                                {i + 1, 100}, "input", false);
    bindings_.allocate(input)->getHandle<float16_t>().randomize(-100, 100,
                                                                mod_.getPRNG());
    TanhNode *tanh = F_->createTanh("tanh", input);
    inputs[i] = tanh->getResult();
  }
  ConcatNode *concat = F_->createConcat("concat", inputs, 0);
  SaveNode *SN = F_->createSave("ret", concat);
  EXPECT_EQ(F_->getNodes().size(), 7);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::TanhNodeKind), 5);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::ConcatNodeKind), 1);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::SaveNodeKind), 1);

  CompilationContext cctx;
  cctx.optimizationOpts.sinkTanhBelowConcat = true;

  optimizedF_ = optimizeFunctionForTest(
      F_, {FunctionPassID::SinkConversions, getDCEPassConfig()}, cctx);

  // Concat, dequantize, save.
  EXPECT_EQ(optimizedF_->getNodes().size(), 3);
  EXPECT_EQ(countNodeKind(optimizedF_, Kinded::Kind::TanhNodeKind), 1);
  EXPECT_EQ(countNodeKind(optimizedF_, Kinded::Kind::ConcatNodeKind), 1);
  EXPECT_EQ(countNodeKind(optimizedF_, Kinded::Kind::SaveNodeKind), 1);

  SaveNode *optSN =
      llvm::dyn_cast<SaveNode>(optimizedF_->getNodeByName(SN->getName()));
  ASSERT_TRUE(optSN);
  TanhNode *optTanh = llvm::dyn_cast<TanhNode>(optSN->getInput());
  ASSERT_TRUE(optTanh);
  NodeValue input = optTanh->getInput();
  EXPECT_EQ(ElemKind::Float16Ty, input.getType()->getElementType());

  checkNumericalEquivalence();
}

/// Test that if we have a Concat with all ConvertTo inputs,
/// we can sink the ConvertTo's below the Concat.
TEST_F(GraphOptz, SinkConvertToBelowConcatTest) {
  std::array<NodeValue, 5> inputs;
  for (dim_t i = 0; i < 5; i++) {
    Placeholder *input = mod_.createPlaceholder(ElemKind::Float16Ty,
                                                {i + 1, 100}, "input", false);
    bindings_.allocate(input)->getHandle<float16_t>().randomize(-100, 100,
                                                                mod_.getPRNG());
    ConvertToNode *convertTo =
        F_->createConvertTo("convertToFP32", input, ElemKind::FloatTy);
    inputs[i] = convertTo->getResult();
  }
  ConcatNode *concat = F_->createConcat("concat", inputs, 0);
  SaveNode *SN = F_->createSave("ret", concat);
  EXPECT_EQ(F_->getNodes().size(), 7);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::ConvertToNodeKind), 5);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::ConcatNodeKind), 1);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::SaveNodeKind), 1);

  CompilationContext cctx;

  optimizedF_ = optimizeFunctionForTest(
      F_, {FunctionPassID::SinkConversions, getDCEPassConfig()}, cctx);

  // Concat, converTo, save.
  EXPECT_EQ(optimizedF_->getNodes().size(), 3);
  EXPECT_EQ(countNodeKind(optimizedF_, Kinded::Kind::ConvertToNodeKind), 1);
  EXPECT_EQ(countNodeKind(optimizedF_, Kinded::Kind::ConcatNodeKind), 1);
  EXPECT_EQ(countNodeKind(optimizedF_, Kinded::Kind::SaveNodeKind), 1);

  SaveNode *optSN =
      llvm::dyn_cast<SaveNode>(optimizedF_->getNodeByName(SN->getName()));
  ASSERT_TRUE(optSN);
  ConvertToNode *optConvertTo =
      llvm::dyn_cast<ConvertToNode>(optSN->getInput());
  ASSERT_TRUE(optConvertTo);
  NodeValue input = optConvertTo->getInput();
  EXPECT_EQ(ElemKind::Float16Ty, input.getType()->getElementType());

  checkNumericalEquivalence();
}

/// Test Clip(Relu) -> Clip'.
TEST_F(GraphOptz, ClipReluTest) {
  Placeholder *input =
      mod_.createPlaceholder(ElemKind::Float16Ty, {64, 64}, "input", false);
  ReluNode *RN = F_->createRELU("RN", input);
  ClipNode *CN = F_->createClip("CN", RN, -5, 20);
  SaveNode *SN = F_->createSave("save", CN);

  // Start with a clip, a relu, and a save.
  EXPECT_EQ(F_->getNodes().size(), 3);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::ClipNodeKind), 1);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::ReluNodeKind), 1);

  optimizedF_ = optimizeFunctionForTest(F_);

  // Removed the relu
  EXPECT_EQ(optimizedF_->getNodes().size(), 2);
  EXPECT_EQ(countNodeKind(optimizedF_, Kinded::Kind::ClipNodeKind), 1);
  EXPECT_EQ(countNodeKind(optimizedF_, Kinded::Kind::ReluNodeKind), 0);

  SaveNode *optSN =
      llvm::dyn_cast<SaveNode>(optimizedF_->getNodeByName(SN->getName()));
  ASSERT_TRUE(optSN);

  // We have the same max for clip as before, but 0 for min due to the Relu.
  ClipNode *optCN = llvm::dyn_cast<ClipNode>(optSN->getInput());
  ASSERT_TRUE(optCN);
  EXPECT_EQ(optCN->getMin(), 0);
  EXPECT_EQ(optCN->getMax(), 20);

  bindings_.allocate(input)->getHandle<float16_t>().randomize(-50.0, 5.0,
                                                              mod_.getPRNG());
  checkNumericalEquivalence();
}

/// Test that if we have a concat with some dequantize inputs that are
/// concatenated together, and then a quantize after the concat, that we can
/// move the quantize above the concat and eliminate the dequantizes.
TEST_F(GraphOptz, SinkConcatBelowQuantize) {
  const float scale = 0.06;
  const int32_t offset = -15;
  std::array<NodeValue, 3> inputs;

  // Concat input 0: Dequant(PH)
  const TypeRef in0QTy =
      mod_.uniqueType(ElemKind::Int8QTy, {1, 3}, scale, offset);
  Placeholder *input0 = mod_.createPlaceholder(in0QTy, "input", false);
  inputs[0] =
      F_->createDequantize("deq", input0, ElemKind::Float16Ty)->getResult();

  // Concat input 1: Dequant(Add(PH, PH))
  const TypeRef in1QTy =
      mod_.uniqueType(ElemKind::Int8QTy, {5, 3}, scale, offset + 1);
  Placeholder *input1 = mod_.createPlaceholder(in1QTy, "input", false);
  AddNode *add = F_->createAdd("add", input1, input1);
  inputs[1] =
      F_->createDequantize("deq", add, ElemKind::Float16Ty)->getResult();

  // Concat input 2: PH
  Placeholder *input2 =
      mod_.createPlaceholder(ElemKind::Float16Ty, {10, 3}, "input_fp", false);
  inputs[2] = input2->getOutput();

  // Concat all 3 together, all FP16.
  ConcatNode *concat = F_->createConcat("concat", inputs, 0);

  // Now quantize the result of the concat.
  const TypeRef QTy = mod_.uniqueType(
      ElemKind::Int8QTy, concat->getResult().dims(), scale, offset);
  QuantizeNode *QN = F_->createQuantize("quantize", concat, QTy);
  SaveNode *SN = F_->createSave("ret", QN);

  optimizedF_ = optimizeFunctionForTest(
      F_,
      {FunctionPassID::SinkConcatBelowQuantize,
       {FunctionPassID::OptimizeQuantization, ConvergenceMode::UntilFixedPoint},
       getDCEPassConfig()});

  EXPECT_EQ(optimizedF_->getNodes().size(), 4);
  EXPECT_EQ(countNodeKind(optimizedF_, Kinded::Kind::ConcatNodeKind), 1);
  EXPECT_EQ(countNodeKind(optimizedF_, Kinded::Kind::AddNodeKind), 1);
  EXPECT_EQ(countNodeKind(optimizedF_, Kinded::Kind::QuantizeNodeKind), 1);
  EXPECT_EQ(countNodeKind(optimizedF_, Kinded::Kind::SaveNodeKind), 1);

  SaveNode *optSN =
      llvm::dyn_cast<SaveNode>(optimizedF_->getNodeByName(SN->getName()));
  ASSERT_TRUE(optSN);

  // Concat should be directly connected to save, with same quantization
  // parameters as the quantize which used to follow it.
  ConcatNode *optCN = llvm::dyn_cast<ConcatNode>(optSN->getInput());
  ASSERT_TRUE(optCN);
  ASSERT_EQ(ElemKind::Int8QTy, optCN->getResult().getType()->getElementType());
  EXPECT_EQ(scale, optCN->getResult().getType()->getScale());
  EXPECT_EQ(offset, optCN->getResult().getType()->getOffset());

  ASSERT_EQ(optCN->getInputs().size(), 3);

  // No rescale here for the PH since its scale/offset match the PH and so
  // are optimized away.
  EXPECT_EQ(optCN->getInputs()[0], input0->getOutput());

  // No rescale here because it should be fused into optAN. Check the
  // scale/offset use that scale/offset.
  AddNode *optAN = llvm::dyn_cast<AddNode>(optCN->getInputs()[1]);
  ASSERT_TRUE(optAN);
  ASSERT_EQ(ElemKind::Int8QTy, optAN->getResult().getType()->getElementType());
  EXPECT_EQ(scale, optAN->getResult().getType()->getScale());
  EXPECT_EQ(offset, optAN->getResult().getType()->getOffset());
  EXPECT_EQ(optAN->getLHS(), input1->getOutput());
  EXPECT_EQ(optAN->getRHS(), input1->getOutput());

  // Must quantize this input since the PH is float16.
  QuantizeNode *optQN = llvm::dyn_cast<QuantizeNode>(optCN->getInputs()[2]);
  ASSERT_TRUE(optQN);
  ASSERT_EQ(ElemKind::Int8QTy, optQN->getResult().getType()->getElementType());
  EXPECT_EQ(scale, optQN->getResult().getType()->getScale());
  EXPECT_EQ(offset, optQN->getResult().getType()->getOffset());
  EXPECT_EQ(optQN->getInput(), input2->getOutput());

  bindings_.allocate(input0)->getHandle<int8_t>().randomize(-50, 50,
                                                            mod_.getPRNG());
  bindings_.allocate(input1)->getHandle<int8_t>().randomize(-50, 50,
                                                            mod_.getPRNG());
  bindings_.allocate(input2)->getHandle<float16_t>().randomize(-10, 10,
                                                               mod_.getPRNG());
}

TEST_F(GraphOptz, EliminateSliceConcatTest) {
  auto *src1 =
      mod_.createPlaceholder(ElemKind::FloatTy, {10, 70}, "src1", false);
  auto *src2 =
      mod_.createPlaceholder(ElemKind::FloatTy, {10, 80}, "src2", false);
  auto *A = F_->createSlice("A", src1, {0, 0}, {10, 10});
  auto *B = F_->createSlice("B", src1, {0, 10}, {10, 20});
  auto *C = F_->createSlice("C", src1, {0, 20}, {10, 30});
  // interleaved Slices with different sources shouldn't merge
  auto *E = F_->createSlice("E", src1, {0, 30}, {10, 40});
  auto *F = F_->createSlice("F", src2, {0, 30}, {10, 40});
  auto *G = F_->createSlice("G", src1, {0, 40}, {10, 50});
  auto *H = F_->createSlice("H", src2, {0, 40}, {10, 50});

  auto *D = mod_.createPlaceholder(ElemKind::FloatTy, {10, 50}, "D", false);
  auto *R = F_->createRELU("Relu", C);
  auto *CN = F_->createConcat("Concat", {A, B, D, E, F, G, H}, 1);
  F_->createSave("save1", CN);
  F_->createSave("save2", R);

  EXPECT_EQ(F_->getNodes().size(), 11);

  optimizedF_ = optimizeFunctionForTest(
      F_, {FunctionPassID::EliminateSliceConcat, getDCEPassConfig()});

  EXPECT_EQ(optimizedF_->getNodes().size(), 10);

  int numSlicesToConcat = 0;
  for (const auto &node : optimizedF_->getNodes()) {
    auto *newCN = llvm::dyn_cast<ConcatNode>(&node);
    if (!newCN) {
      continue;
    }
    EXPECT_EQ(newCN->getInputs().size(), 6);
    for (const auto &concatInput : newCN->getInputs()) {
      auto *SN = llvm::dyn_cast<SliceNode>(concatInput.getNode());
      if (SN) {
        numSlicesToConcat++;
      }
    }
  }
  EXPECT_EQ(numSlicesToConcat, 5);

  bindings_.allocate(src1)->getHandle<float>().randomize(-10.0, 10.0,
                                                         mod_.getPRNG());
  bindings_.allocate(src2)->getHandle<float>().randomize(-10.0, 10.0,
                                                         mod_.getPRNG());
  bindings_.allocate(D)->getHandle<float>().randomize(-10.0, 10.0,
                                                      mod_.getPRNG());
  checkNumericalEquivalence();
}

TEST_F(GraphOptz, EliminateSliceConcatWithReshapeTest) {
  auto *src =
      mod_.createPlaceholder(ElemKind::FloatTy, {4, 5, 4}, "src", false);
  auto *A = F_->createSlice("A", src, {0, 0, 0}, {1, 5, 4});
  auto *B = F_->createSlice("B", src, {1, 0, 0}, {2, 5, 4});
  auto *C = F_->createSlice("C", src, {2, 0, 0}, {3, 5, 4});
  auto *CN1 = F_->createConcat("Concat1", {A, B, C}, 1);

  auto *E = F_->createSlice("E", src, {0, 0, 0}, {4, 5, 1});
  auto *F = F_->createSlice("F", src, {0, 0, 1}, {4, 5, 2});
  auto *G = F_->createSlice("G", src, {0, 0, 2}, {4, 5, 3});
  auto *H = F_->createSlice("H", src, {0, 0, 3}, {4, 5, 4});
  auto *CN2 = F_->createConcat("Concat2", {E, F, G, H}, 1);

  F_->createSave("save1", CN1);
  F_->createSave("save2", CN2);

  EXPECT_EQ(F_->getNodes().size(), 11);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::SliceNodeKind), 7);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::ConcatNodeKind), 2);

  optimizedF_ = optimizeFunctionForTest(
      F_, {FunctionPassID::EliminateSliceConcat, getDCEPassConfig()});

  EXPECT_EQ(optimizedF_->getNodes().size(), 9);
  EXPECT_EQ(countNodeKind(optimizedF_, Kinded::Kind::SliceNodeKind), 2);
  EXPECT_EQ(countNodeKind(optimizedF_, Kinded::Kind::ConcatNodeKind), 2);
  EXPECT_EQ(countNodeKind(optimizedF_, Kinded::Kind::ReshapeNodeKind), 2);
  EXPECT_EQ(countNodeKind(optimizedF_, Kinded::Kind::TransposeNodeKind), 1);

  bindings_.allocate(src)->getHandle<float>().randomize(-10.0, 10.0,
                                                        mod_.getPRNG());
  checkNumericalEquivalence(0.f);
}

// Check the merging of Sub(const, BN(x, scale, bias)) into BN.
TEST_F(GraphOptz, FoldArithmeticChainIntoBatchNormQuant) {
  auto *subC = mod_.createConstant(ElemKind::FloatTy, {1, 1, 1, 1}, "subC");
  auto *var = mod_.createConstant(ElemKind::FloatTy, {1}, "var");
  auto *mean = mod_.createConstant(ElemKind::FloatTy, {1}, "mean");
  auto *beta = mod_.createConstant(ElemKind::FloatTy, {1}, "beta");
  auto *gamma = mod_.createConstant(ElemKind::FloatTy, {1}, "gamma");
  float v = 0.3f, m = 0.4f, b = 0.7f, g = -0.5f, c = 0.1;
  // (X - mean) * (1.0 / sqrt(var + eps)) * gamma + beta
  var->getPayloadMutable().getHandle<float>() = {v};
  mean->getPayloadMutable().getHandle<float>() = {m};
  beta->getPayloadMutable().getHandle<float>() = {b};
  gamma->getPayloadMutable().getHandle<float>() = {g};
  subC->getPayloadMutable().getHandle<float>() = {c};
  auto *input = mod_.createPlaceholder(ElemKind::FloatTy, {1, 1, 1, 1}, "input",
                                       false, "NHWC");

  auto *BN = F_->createBatchNormalization("batch", input->getType(), input,
                                          beta, gamma, mean, var);
  auto *sub = F_->createSub("sub", subC, BN);
  auto *res = F_->createSave("save", sub);
  // Compile.
  EXPECT_EQ(F_->getNodes().size(), 3);
  ::glow::convertPlaceholdersToConstants(F_, bindings_, {});

  optimizedF_ = optimizeFunctionForTest(F_, {}, cctx_);
  EXPECT_EQ(optimizedF_->getNodes().size(), 2);

  auto *opt_res = findFunctionNodeByName<SaveNode>(optimizedF_, res->getName());
  auto *opt_bn = llvm::dyn_cast<BatchNormalizationNode>(opt_res->getInput());
  ASSERT_TRUE(opt_bn);
  // Verify that scale and offset are computed correctly.
  Constant *bnScale = llvm::dyn_cast<Constant>(opt_bn->getScale().getNode());
  Constant *bnBias = llvm::dyn_cast<Constant>(opt_bn->getBias().getNode());
  auto bnBiasVals = bnBias->getHandle<float>().raw(0);
  auto bnScaleVals = bnScale->getHandle<float>().raw(0);
  EXPECT_EQ(bnBiasVals, c - b);
  EXPECT_EQ(bnScaleVals, -g);
}

/// Test that EliminateSliceConcat makes no optimization when the axis of
/// concatenation and slicing are not adjacent.
TEST_F(GraphOptz, EliminateSliceConcatWithReshapeTestNoChange) {
  auto *src =
      mod_.createPlaceholder(ElemKind::FloatTy, {4, 5, 4}, "src", false);
  auto *A = F_->createSlice("A", src, {0, 0, 0}, {1, 5, 4});
  auto *B = F_->createSlice("B", src, {1, 0, 0}, {2, 5, 4});
  auto *C = F_->createSlice("C", src, {2, 0, 0}, {3, 5, 4});
  auto *CN = F_->createConcat("Concat", {A, B, C}, 2);

  F_->createSave("save", CN);

  EXPECT_EQ(F_->getNodes().size(), 5);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::SliceNodeKind), 3);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::ConcatNodeKind), 1);

  optimizedF_ = optimizeFunctionForTest(
      F_, {FunctionPassID::EliminateSliceConcat, getDCEPassConfig()});

  EXPECT_EQ(optimizedF_->getNodes().size(), 5);
  EXPECT_EQ(countNodeKind(optimizedF_, Kinded::Kind::SliceNodeKind), 3);
  EXPECT_EQ(countNodeKind(optimizedF_, Kinded::Kind::ConcatNodeKind), 1);
  EXPECT_EQ(F_->toString(/* skipUsersForStorage */ false,
                         /* skipName */ true),
            optimizedF_->toString(/* skipUsersForStorage */ false,
                                  /* skipName */ true));
}

/// Verify that when we want to prevent constant folding it doesn't occur.
TEST_F(GraphOptz, constantFoldPreventedNoop) {
  auto *const1 = mod_.createConstant(ElemKind::FloatTy, {2, 2}, "const1");
  auto *const2 = mod_.createConstant(ElemKind::FloatTy, {2, 2}, "const2");
  auto *ph1 = mod_.createPlaceholder(ElemKind::FloatTy, {2, 2}, "input1",
                                     /* isTrainable */ false);
  setConstValue(const1, 1.0f);
  setConstValue(const2, 2.0f);
  auto *splat2 = F_->createSplat(
      "splat2", mod_.uniqueType(ElemKind::FloatTy, {2, 2}), 2.0f);
  auto *splat3 = F_->createSplat(
      "splat3", mod_.uniqueType(ElemKind::FloatTy, {2, 2}), 3.0f);

  auto *add1 = F_->createAdd("add", const1, const2);
  auto *mul1 = F_->createMul("mul1", add1, splat2);
  auto *mul2 = F_->createMul("mul2", mul1, splat3);
  F_->createSave("save", mul2);
  auto *add3 = F_->createAdd("add", const1, ph1);
  F_->createSave("save", add3);

  ConstantModificationPreventer constModPreventer(mod_, cctx_);
  constModPreventer.activate();
  EXPECT_FALSE(cctx_.optimizationOpts.enableConstantFolding);

  // Check that both Constants are protected and no change is made to the
  // Function during optimization.
  EXPECT_EQ(constModPreventer.getMapping().size(), 2);
  optimizedF_ = optimizeFunctionForTest(F_);
  EXPECT_EQ(F_->toString(/* skipUsersForStorage */ false,
                         /* skipName */ true),
            optimizedF_->toString(/* skipUsersForStorage */ false,
                                  /* skipName */ true));

  // Now deactivate the constModPreventer and check we can const fold still.
  constModPreventer.deactivateAndCleanup();
  EXPECT_TRUE(cctx_.optimizationOpts.enableConstantFolding);
  mod_.eraseFunction(optimizedF_);
  optimizedF_ = optimizeFunctionForTest(F_);

  // After constant folding, left with just two Saves, one Add.
  EXPECT_EQ(optimizedF_->getNodes().size(), 3);
  EXPECT_EQ(countNodeKind(optimizedF_, Kinded::Kind::AddNodeKind), 1);
  EXPECT_EQ(countNodeKind(optimizedF_, Kinded::Kind::SaveNodeKind), 2);

  bindings_.allocate(ph1)->getHandle<float>().randomize(-10.0, 10.0,
                                                        mod_.getPRNG());
  checkNumericalEquivalence();
}

/// Test that a Conv2D is correctly lowered to FC for single batch.
TEST_F(GraphOptz, lowerConv2DToFCSingleBatch) {
  Placeholder *input = mod_.createPlaceholder(ElemKind::FloatTy, {1, 2, 3, 4},
                                              "input", /* isTrainable */ false);
  bindings_.allocate(input)->getHandle<float>().randomize(-10, 10,
                                                          mod_.getPRNG());

  Constant *filter =
      mod_.createConstant(ElemKind::FloatTy, {8, 1, 1, 4}, "filter");
  filter->getPayloadMutable().getHandle<float>().randomize(-10, 10,
                                                           mod_.getPRNG());

  Constant *bias = mod_.createConstant(ElemKind::FloatTy, {8}, "bias");
  bias->getPayloadMutable().getHandle<float>().randomize(-10, 10,
                                                         mod_.getPRNG());

  auto outTy = mod_.uniqueType(ElemKind::FloatTy, {1, 2, 3, 8});
  auto *conv = F_->createConv("conv", input, filter, bias, outTy, {1, 1},
                              {1, 1}, {0, 0, 0, 0}, 1);
  SaveNode *save = F_->createSave("save", conv);
  bindings_.allocate(save->getPlaceholder());

  // Backup function in optimizedF_.
  optimizedF_ = F_->clone(F_->getName().str() + "_optimized");

  // Lower Convolution.
  EXPECT_TRUE(isConvolutionSameAsFullyConnected(conv));
  EXPECT_TRUE(glow::lowerNode(F_, conv, cctx_));
  runDCEPass(F_, cctx_);
  EXPECT_EQ(0, countNodeKind(F_, Kinded::Kind::ConvolutionNodeKind));
  EXPECT_EQ(1, countNodeKind(F_, Kinded::Kind::FullyConnectedNodeKind));

  // Now compile/run/compare F_ and optimizedF_.
  checkNumericalEquivalence(1e-6);
}

/// Test that a Conv2D is correctly lowered to FC for multi batch.
TEST_F(GraphOptz, lowerConv2DToFCMultiBatch) {
  Placeholder *input = mod_.createPlaceholder(ElemKind::FloatTy, {2, 2, 3, 4},
                                              "input", /* isTrainable */ false);
  bindings_.allocate(input)->getHandle<float>().randomize(-10, 10,
                                                          mod_.getPRNG());

  Constant *filter =
      mod_.createConstant(ElemKind::FloatTy, {8, 1, 1, 4}, "filter");
  filter->getPayloadMutable().getHandle<float>().randomize(-10, 10,
                                                           mod_.getPRNG());

  Constant *bias = mod_.createConstant(ElemKind::FloatTy, {8}, "bias");
  bias->getPayloadMutable().getHandle<float>().randomize(-10, 10,
                                                         mod_.getPRNG());

  auto outTy = mod_.uniqueType(ElemKind::FloatTy, {2, 2, 3, 8});
  auto *conv = F_->createConv("conv", input, filter, bias, outTy, {1, 1},
                              {1, 1}, {0, 0, 0, 0}, 1);
  SaveNode *save = F_->createSave("save", conv);
  bindings_.allocate(save->getPlaceholder());

  // Backup function in optimizedF_.
  optimizedF_ = F_->clone(F_->getName().str() + "_optimized");

  // Lower Convolution.
  EXPECT_TRUE(isConvolutionSameAsFullyConnected(conv));
  EXPECT_TRUE(glow::lowerNode(F_, conv, cctx_));
  runDCEPass(F_, cctx_);
  EXPECT_EQ(0, countNodeKind(F_, Kinded::Kind::ConvolutionNodeKind));
  EXPECT_EQ(1, countNodeKind(F_, Kinded::Kind::FullyConnectedNodeKind));

  // Now compile/run/compare F_ and optimizedF_.
  checkNumericalEquivalence(1e-6);
}

/// Test that Mul and Add can be folded into LayerNorm.
TEST_F(GraphOptz, foldMulAddIntoLayerNorm) {
  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 4, 10, 20}, "in", false);

  Tensor scaleT(ElemKind::FloatTy, {10, 20});
  scaleT.getHandle().randomize(0.0f, 1.0f, mod_.getPRNG());
  Constant *scaleC = mod_.createConstant("scale", std::move(scaleT));
  SplatNode *biasS = F_->createSplat("bias", scaleC->getType(), 1.5f);

  auto *LN = F_->createLayerNormalization("LN", input->getType(), input, scaleC,
                                          biasS, 1e-5);

  SplatNode *splat = F_->createSplat("splat", scaleC->getType(), 0.5f);
  MulNode *MN =
      F_->createNodeWithBroadcast<MulNode>("mul", /* axis */ -1, LN, splat);

  Tensor addT(ElemKind::FloatTy, {1, 1, 10, 20});
  addT.getHandle().randomize(-1.0f, 1.0f, mod_.getPRNG());
  Constant *addC = mod_.createConstant("addC", std::move(addT));
  AddNode *AN =
      F_->createNodeWithBroadcast<AddNode>("add", /* axis */ -1, MN, addC);

  // This MulNode has a Placeholder as RHS and shouldn't be fused into LayerNorm
  Tensor mulT(ElemKind::FloatTy, {1, 1, 10, 20});
  mulT.getHandle().randomize(0.0f, 1.0f, mod_.getPRNG());
  Constant *mulC = mod_.createConstant("mulC", std::move(mulT));
  MN = F_->createNodeWithBroadcast<MulNode>("mul_not_fuse", /* axis */ -1, AN,
                                            mulC);
  F_->createSave("save", MN);

  ConstantModificationPreventer constModPreventer(mod_, cctx_);
  constModPreventer.activate();
  optimizedF_ = optimizeFunctionForTest(F_, {}, cctx_);
  // Now do const folding with constants swapped back in.
  constModPreventer.deactivateAndCleanup();
  ConstantFoldingRecordMap record = constantFoldAndRecord(optimizedF_, cctx_);
  runDCEPass(optimizedF_, cctx_);

  // Because Muls and Add are folded in, they should not exist anymore, nor
  // should Broadcasts that expand them to match the output of LN.
  EXPECT_EQ(0, countNodeKind(optimizedF_, Kinded::Kind::MulNodeKind));
  EXPECT_EQ(0, countNodeKind(optimizedF_, Kinded::Kind::AddNodeKind));
  EXPECT_EQ(0, countNodeKind(optimizedF_, Kinded::Kind::BroadcastNodeKind));

  // Remove the temporary constant folding Functions and their Placeholders
  // so that they don't participate in 'checkNumericalEquivalence'.
  cleanupConstantFolding(mod_, record, &bindings_);

  // Now compile/run/compare F_ and optimizedF_.
  bindings_.allocate(input)->getHandle().randomize(0.0f, 1.0f, mod_.getPRNG());
  checkNumericalEquivalence(1.2e-7);
}

/// Test that Mul and Add can be folded into LayerNorm when the leading dims are
/// all one.
TEST_F(GraphOptz, foldMulAddIntoLayerNormNoBatch) {
  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 1, 10, 20}, "in", false);

  Tensor scaleT(ElemKind::FloatTy, {10, 20});
  scaleT.getHandle().randomize(0.0f, 1.0f, mod_.getPRNG());
  Constant *scaleC = mod_.createConstant("scale", std::move(scaleT));
  SplatNode *biasS = F_->createSplat("bias", scaleC->getType(), 1.5f);

  auto *LN = F_->createLayerNormalization("LN", input->getType(), input, scaleC,
                                          biasS, 1e-5);

  SplatNode *splat = F_->createSplat("splat", scaleC->getType(), 0.5f);
  MulNode *MN =
      F_->createNodeWithBroadcast<MulNode>("mul", /* axis */ -1, LN, splat);

  Tensor addT(ElemKind::FloatTy, {1, 1, 10, 20});
  addT.getHandle().randomize(-1.0f, 1.0f, mod_.getPRNG());
  Constant *addC = mod_.createConstant("addC", std::move(addT));
  AddNode *AN =
      F_->createNodeWithBroadcast<AddNode>("add", /* axis */ -1, MN, addC);
  F_->createSave("save", AN);

  optimizedF_ = optimizeFunctionForTest(F_);

  // Because Mul and Add are folded in, they should not exist anymore, nor
  // should tiles that expand them to match the output of LN.
  EXPECT_EQ(0, countNodeKind(optimizedF_, Kinded::Kind::MulNodeKind));
  EXPECT_EQ(0, countNodeKind(optimizedF_, Kinded::Kind::AddNodeKind));
  EXPECT_EQ(0, countNodeKind(optimizedF_, Kinded::Kind::TileNodeKind));

  // Now compile/run/compare F_ and optimizedF_.
  bindings_.allocate(input)->getHandle().randomize(0.0f, 1.0f, mod_.getPRNG());
  checkNumericalEquivalence(1e-6);
}

TEST_F(GraphOptz, transposeQuantizeConstantWithAlignment) {
  // Define a type with custom alignments.
  Type typeWithAlignments(ElemKind::FloatTy, {2, 3, 4, 5}, {1, 1, 32, 1});
  Type quantTypeWithAlignments(ElemKind::Int8QTy, {2, 3, 4, 5}, {1, 1, 32, 1},
                               1.0, 0);
  Type transposedQuantTypeWithAlignments(ElemKind::Int8QTy, {2, 4, 5, 3},
                                         {1, 1, 32, 1}, 1.0, 0);
  auto modTyWithAlignments = mod_.uniqueType(typeWithAlignments);
  auto modQuantTransposedTyWithAlignments =
      mod_.uniqueType(transposedQuantTypeWithAlignments);
  auto modQuantTyWithAlignments = mod_.uniqueType(quantTypeWithAlignments);
  auto *I = mod_.createConstant(modTyWithAlignments, "input1");
  auto *Q = F_->createQuantize("quantize", I, modQuantTyWithAlignments);
  auto *T = F_->createTranspose("transpose", Q, NCHW2NHWC);
  T->setType(TransposeNode::ResultIdx, modQuantTransposedTyWithAlignments);
  SaveNode *S = F_->createSave("ret", T);

  // Skip ConstantFolding as it would have the same result as this opt.
  CompilationContext cctx;
  cctx.optimizationOpts.enableConstantFolding = false;

  EXPECT_EQ(F_->getNodes().size(), 3);
  ::glow::optimize(F_, cctx);
  EXPECT_EQ(F_->getNodes().size(), 2);

  // Constant and Quantize should have new shape.
  auto *newQ = llvm::dyn_cast<QuantizeNode>(S->getInput());
  ASSERT_TRUE(newQ);
  EXPECT_TRUE(newQ->getResult().dims().equals({2, 4, 5, 3}));
  auto *newC = llvm::dyn_cast<Constant>(newQ->getInput());
  ASSERT_TRUE(newC);
  EXPECT_TRUE(newC->getType()->dims().equals({2, 4, 5, 3}));

  // Check that alignments are preserved by optimizations.
  auto expectedNewTy = mod_.uniqueTypeWithNewShape(
      modTyWithAlignments, modQuantTransposedTyWithAlignments);
  EXPECT_TRUE(newQ->getInput().getType()->isEqual(expectedNewTy));

  EXPECT_TRUE(F_->verify());
}

TEST_F(GraphOptz, DequantSwishQuantOpt) {
  const dim_t origDims[] = {1, 5, 10, 15};
  Placeholder *A = mod_.createPlaceholder(ElemKind::Int8QTy, origDims, 0.039, 0,
                                          "input", false);
  DequantizeNode *DN = F_->createDequantize("deq", A, ElemKind::Float16Ty);
  SwishNode *swish = F_->createSwish("swish", DN);
  QuantizeNode *QN =
      F_->createQuantize("quant", swish, ElemKind::Int8QTy, 0.0204, -114);
  DequantizeNode *finalDN =
      F_->createDequantize("deq_final", QN, ElemKind::Float16Ty);
  F_->createSave("ret", finalDN);

  optimizedF_ = optimizeFunctionForTest(
      F_, {FunctionPassID::QuantizeSwish, getDCEPassConfig()});

  // Swish, Dequant, Save
  EXPECT_EQ(optimizedF_->getNodes().size(), 3);

  SaveNode *save = nullptr;
  for (auto &N : optimizedF_->getNodes()) {
    if (N.getKind() == Kinded::Kind::SaveNodeKind) {
      save = llvm::dyn_cast<SaveNode>(&N);
      break;
    }
  }
  ASSERT_TRUE(save);

  DequantizeNode *dequantizeOpt =
      llvm::dyn_cast<DequantizeNode>(save->getInput());
  ASSERT_TRUE(dequantizeOpt);

  SwishNode *swishOpt = llvm::dyn_cast<SwishNode>(dequantizeOpt->getInput());
  ASSERT_TRUE(swishOpt);
  EXPECT_EQ(swishOpt->getInput(), A->getOutput());
  EXPECT_EQ(swishOpt->getResult().getType(), QN->getResult().getType());

  bindings_.allocate(mod_.getPlaceholders());
  bindings_.get(A)->getHandle<int8_t>().randomize(-128, 127, mod_.getPRNG());

  checkNumericalEquivalence(0.025f);
}

/// Test the conversion of FullyConnected to 1x1 Convolution.
TEST_F(GraphOptz, ConvertFullyConnectedToConvolutionOpt) {

  const std::vector<dim_t> inpDims = {3, 5};
  const std::vector<dim_t> weightsDims = {5, 7};
  const std::vector<dim_t> biasDims = {7};

  // Create graph.
  Placeholder *input =
      mod_.createPlaceholder(ElemKind::FloatTy, inpDims, "input", false);
  Placeholder *weights =
      mod_.createPlaceholder(ElemKind::FloatTy, weightsDims, "weights", false);
  Placeholder *bias =
      mod_.createPlaceholder(ElemKind::FloatTy, biasDims, "bias", false);
  FullyConnectedNode *FCN =
      F_->createFullyConnected("fc", input, weights, bias);
  F_->createSave("save", FCN);

  // Optimize graph.
  optimizedF_ = optimizeFunctionForTest(
      F_,
      {FunctionPassID::ConvertFullyConnectedToConvolution, getDCEPassConfig()});

  // Check optimized graph.
  EXPECT_EQ(optimizedF_->getNodes().size(), 6);
  SaveNode *save = nullptr;
  for (auto &N : optimizedF_->getNodes()) {
    if (N.getKind() == Kinded::Kind::SaveNodeKind) {
      save = llvm::dyn_cast<SaveNode>(&N);
      break;
    }
  }
  ASSERT_TRUE(save);
  ReshapeNode *reshapeOut = llvm::dyn_cast<ReshapeNode>(save->getInput());
  ASSERT_TRUE(reshapeOut);
  ConvolutionNode *conv =
      llvm::dyn_cast<ConvolutionNode>(reshapeOut->getInput());
  ASSERT_TRUE(conv);
  ReshapeNode *reshapeFilter = llvm::dyn_cast<ReshapeNode>(conv->getFilter());
  ASSERT_TRUE(reshapeFilter);
  TransposeNode *transpFilter =
      llvm::dyn_cast<TransposeNode>(reshapeFilter->getInput());
  ASSERT_TRUE(transpFilter);
  ReshapeNode *reshapeInput = llvm::dyn_cast<ReshapeNode>(conv->getInput());
  ASSERT_TRUE(reshapeInput);

  // Check numerical equivalence.
  bindings_.allocate(mod_.getPlaceholders());
  bindings_.get(input)->getHandle<float>().randomize(-1, 1, mod_.getPRNG());
  bindings_.get(weights)->getHandle<float>().randomize(-1, 1, mod_.getPRNG());
  bindings_.get(bias)->getHandle<float>().randomize(-1, 1, mod_.getPRNG());
  checkNumericalEquivalence(1e-8);
}

/// Test that when we have Concat({X, Quantize(Clip)}), that we don't optimize
/// to Concat({X, Quantize'}), since Quantize' will have different quantization
/// parameters and therefore won't have the same quantization parameters as X.
TEST_F(GraphOptz, DisallowChangeQuantParamWithConcatInput) {
  Placeholder *PH1 = mod_.createPlaceholder(ElemKind::Int8QTy, {2, 32}, 0.3, 5,
                                            "input", false);
  bindings_.allocate(PH1)->getHandle<int8_t>().randomize(-128, 127,
                                                         mod_.getPRNG());
  Placeholder *PH2 =
      mod_.createPlaceholder(ElemKind::Float16Ty, {1, 32}, "input", false);
  bindings_.allocate(PH2)->getHandle<float16_t>().randomize(-40.f, 40.f,
                                                            mod_.getPRNG());

  ClipNode *clip = F_->createClip("clip", PH2, 0.f, 1000.f);
  QuantizeNode *quant = F_->createQuantize(
      "quantize", clip, mod_.uniqueType(ElemKind::Int8QTy, {1, 32}, 0.3, 5));

  ConcatNode *CN = F_->createConcat("concat", {PH1, quant}, 0);
  F_->createSave("save", CN);

  optimizedF_ = optimizeFunctionForTest(F_);

  // Expect the graph didn't change at all, since we disallowed it due to the
  // fact that we disallowed Quantize(Clip) to be merged into Quantize', ssince
  // the Quantize is consumed by a Concat which requires the quantization
  // parameters to stay the same across all inputs.
  EXPECT_EQ(F_->toString(/* skipUsersForStorage */ false,
                         /* skipName */ true),
            optimizedF_->toString(/* skipUsersForStorage */ false,
                                  /* skipName */ true));

  checkNumericalEquivalence();
}

/// Test that a AdaptiveAvgPool with 1x1 OFM is correctly lowered to AvgPool.
TEST_F(GraphOptz, lower1x1AdaptiveAvgPoolToAvgPool) {
  Placeholder *input = mod_.createPlaceholder(ElemKind::FloatTy, {2, 2, 3, 4},
                                              "input", /* isTrainable */ false);
  bindings_.allocate(input)->getHandle<float>().randomize(-10, 10,
                                                          mod_.getPRNG());

  auto outTy = mod_.uniqueType(ElemKind::FloatTy, {2, 1, 1, 4});
  auto *pool = F_->createAdaptiveAvgPool("avg", input, outTy);
  SaveNode *save = F_->createSave("save", pool);
  bindings_.allocate(save->getPlaceholder());

  // Backup function in optimizedF_.
  optimizedF_ = F_->clone(F_->getName().str() + "_optimized");

  // Lower
  EXPECT_TRUE(glow::lowerNode(F_, pool, cctx_));
  runDCEPass(F_, cctx_);
  EXPECT_EQ(0, countNodeKind(F_, Kinded::Kind::AdaptiveAvgPoolNodeKind));
  EXPECT_EQ(1, countNodeKind(F_, Kinded::Kind::AvgPoolNodeKind));

  // Now compile/run/compare F_ and optimizedF_.
  checkNumericalEquivalence(1e-6);
}

/// Skip Clip-Quantize optimization when loadUniquedDummyQParams.
TEST_F(GraphOptz, SkipDummyQParamOpts) {
  Placeholder *A = mod_.createPlaceholder(ElemKind::FloatTy, {5}, "A", false);
  ClipNode *CN = F_->createClip("clip", A, -1000.f, 1000.f);
  QuantizeNode *QN = F_->createQuantize(
      "quantize", CN, mod_.uniqueType(ElemKind::Int8QTy, {5}, 0.3, 5));
  F_->createSave("ret", QN);

  CompilationContext cctx;
  cctx.precisionConfig.loadUniquedDummyQParams = true;

  optimizedF_ = optimizeFunctionForTest(F_, {}, cctx);
  EXPECT_EQ(F_->toString(/* skipUsersForStorage */ false, /* skipName */ true),
            optimizedF_->toString(/* skipUsersForStorage */ false,
                                  /* skipName */ true));
}

/// Test that Min -> Max is correctly folded into Clip
TEST_F(GraphOptz, foldMinMaxToClipTest) {
  Placeholder *input = mod_.createPlaceholder(ElemKind::FloatTy, {1, 5, 5},
                                              "input", /* isTrainable */ false);
  bindings_.allocate(input)->getHandle<float>().randomize(-10, 10,
                                                          mod_.getPRNG());

  auto *minFirstSplat = F_->createSplat("min_first_splat", input->getType(), 5);
  auto *maxFirstSplat =
      F_->createSplat("max_first_splat", input->getType(), -2);
  auto *minFirst = F_->createMin("min_first", input, minFirstSplat);
  auto *maxFirst = F_->createMax("max_first", maxFirstSplat, minFirst);

  auto *minSecondSplat = F_->createSplat(
      "min_second_splat",
      F_->getParent()->uniqueTypeWithNewShape(input->getType(), {3, 1, 1}), 3);
  auto *maxSecondSplat =
      F_->createSplat("max_second_splat", input->getType(), 1);
  auto *maxSecond = F_->createMax("max_second", maxFirst, maxSecondSplat);
  auto *minSecond = F_->createNodeWithBroadcast<MinNode>(
      "min_second", /* axis */ -1, maxSecond, minSecondSplat);
  SaveNode *save = F_->createSave("save", minSecond);
  bindings_.allocate(save->getPlaceholder());

  // Need to run OptimizeArithmeticNodes first to move constant operators in
  // communative nodes to RHS.
  optimizedF_ = optimizeFunctionForTest(
      F_, {FunctionPassID::OptimizeArithmeticNodes,
           FunctionPassID::FoldMinMaxToClip, getDCEPassConfig()});

  EXPECT_EQ(4, optimizedF_->getNodes().size());
  EXPECT_EQ(0, countNodeKind(optimizedF_, Kinded::Kind::MinNodeKind));
  EXPECT_EQ(0, countNodeKind(optimizedF_, Kinded::Kind::MaxNodeKind));

  // Get SaveNode in optimizedF_
  save = llvm::dyn_cast<SaveNode>(optimizedF_->getNodeByName("save_save"));
  // Check min and max of the second ClipNode
  ClipNode *CN = llvm::dyn_cast<ClipNode>(save->getInput().getNode());
  EXPECT_EQ(1, CN->getMin());
  EXPECT_EQ(3, CN->getMax());

  // There's a BroadcastNode in between the first and the second ClipNode
  BroadcastNode *BN = llvm::dyn_cast<BroadcastNode>(CN->getInput().getNode());
  // Check min and max of the first ClipNode
  CN = llvm::dyn_cast<ClipNode>(BN->getInput().getNode());
  EXPECT_EQ(-2, CN->getMin());
  EXPECT_EQ(5, CN->getMax());

  checkNumericalEquivalence();
}

/// Test that Min -> Max Fold pass does not break with a reshape LHS input.
TEST_F(GraphOptz, foldMinMaxToClipReshapeNoBroadcastTest) {
  Placeholder *input = mod_.createPlaceholder(ElemKind::FloatTy, {1, 100},
                                              "input", /* isTrainable */ false);
  bindings_.allocate(input)->getHandle<float>().randomize(-10, 10,
                                                          mod_.getPRNG());

  auto *reshape = F_->createReshape("reshape", input, {100, 1});
  const TypeRef T = reshape->getResult().getType();

  auto *maxSplat = F_->createSplat("max_splat", T, -2);
  auto *minSplat = F_->createSplat("min_splat", T, 5);
  auto *max = F_->createMax("max", reshape, maxSplat);
  auto *min = F_->createMin("min", max, minSplat);
  SaveNode *save = F_->createSave("save", min);
  bindings_.allocate(save->getPlaceholder());

  optimizedF_ = optimizeFunctionForTest(
      F_, {FunctionPassID::FoldMinMaxToClip, getDCEPassConfig()});

  EXPECT_EQ(3, optimizedF_->getNodes().size());
  EXPECT_EQ(0, countNodeKind(optimizedF_, Kinded::Kind::MinNodeKind));
  EXPECT_EQ(0, countNodeKind(optimizedF_, Kinded::Kind::MaxNodeKind));

  save = llvm::dyn_cast<SaveNode>(optimizedF_->getNodeByName("save_save"));
  ASSERT_TRUE(save);
  auto *CN = llvm::dyn_cast<ClipNode>(save->getInput().getNode());
  ASSERT_TRUE(CN);
  EXPECT_EQ(-2, CN->getMin());
  EXPECT_EQ(5, CN->getMax());
  auto *RN = llvm::dyn_cast<ReshapeNode>(CN->getInput());
  ASSERT_TRUE(RN);
  EXPECT_TRUE(RN->getResult().getType()->isEqual(T));
  EXPECT_EQ(RN->getInput(), input->getOutput());

  checkNumericalEquivalence();
}

/// Check that we replace a Node with 0.f scale in fp16 with a splat correctly.
TEST_F(GraphOptz, ReplaceZeroScaleFP16QuantOpt) {
  auto *LHS = mod_.createPlaceholder(ElemKind::FloatTy, {20, 30}, "LHS", false);
  auto *RHSQ = mod_.createPlaceholder(ElemKind::Int8QTy, {20, 30}, 0.1f, 10,
                                      "LHS", false);

  // scale = 1e-9 underflows fp16 and so this opt applies.
  auto *LHSQTy = mod_.uniqueType(ElemKind::Int8QTy, {20, 30}, 1e-9, 10);
  auto *LHSQ = F_->createQuantize("LHSQ", LHS, LHSQTy);

  auto *A = F_->createAdd("add", RHSQ->getOutput().getType(), LHSQ, RHSQ);
  auto *Q = F_->createDequantize("deq", A, ElemKind::FloatTy);
  F_->createSave("save", Q);

  optimizedF_ = optimizeFunctionForTest(
      F_, {FunctionPassID::ReplaceZeroScaleFP16QuantNodes, getDCEPassConfig()});

  SaveNode *save = nullptr;
  for (auto &N : optimizedF_->getNodes()) {
    if (N.getKind() == Kinded::Kind::SaveNodeKind) {
      save = llvm::dyn_cast<SaveNode>(&N);
      break;
    }
  }
  ASSERT_TRUE(save);

  DequantizeNode *optQ = llvm::dyn_cast<DequantizeNode>(save->getInput());
  ASSERT_TRUE(optQ);
  AddNode *optA = llvm::dyn_cast<AddNode>(optQ->getInput());
  ASSERT_TRUE(A);

  SplatNode *splat = llvm::dyn_cast<SplatNode>(optA->getLHS());
  ASSERT_TRUE(splat);
  EXPECT_EQ(splat->getValue(), 0.f);
  const TypeRef optLHSQTy = splat->getResult().getType();
  EXPECT_EQ(optLHSQTy->getScale(), 1.f);
  EXPECT_EQ(optLHSQTy->getOffset(), 0);
  EXPECT_EQ(optLHSQTy->getElementType(), LHSQTy->getElementType());
  EXPECT_EQ(optLHSQTy->dims(), LHSQTy->dims());

  bindings_.allocate(LHS)->getHandle<float>().randomize(-10.f, 10.f,
                                                        mod_.getPRNG());
  bindings_.allocate(RHSQ)->getHandle<int8_t>().randomize(-128, 127,
                                                          mod_.getPRNG());

  checkNumericalEquivalence(0.f);
}

/// Same as GraphOptz, but when running numerical equivalence use the CPU
/// backend instead of Interpreter.
class GraphOptzOnCPU : public GraphOptz {
public:
  GraphOptzOnCPU() : GraphOptz("CPU") {}
#ifndef GLOW_WITH_CPU
  virtual void checkNumericalEquivalence(float allowedError = 0.0001) override {
    LOG(INFO) << "Skipping numerical equivalence check as the CPU backend is "
                 "not built.";
  }
#endif /* GLOW_WITH_CPU */
};

/// Check that we replace a Node with 0.f scale in fp16 with a splat correctly.
TEST_F(GraphOptzOnCPU, ReplaceZeroScaleFP16QuantConstOpt) {
  auto *input =
      mod_.createPlaceholder(ElemKind::Int8QTy, {1, 1}, 1.0, 0, "input", false);
  // scale = 1e-9 underflows fp16 and so this opt applies.
  auto *weights =
      mod_.createConstant(ElemKind::Int8QTy, {1, 1}, 1e-9, 0, "weights");
  weights->getPayloadMutable().getHandle<int8_t>().randomize(-128, 127,
                                                             mod_.getPRNG());
  auto *MM = F_->createMatMul("matmul", input, weights);
  auto *DQ = F_->createDequantize("dq", MM, ElemKind::FloatTy);
  F_->createSave("save", DQ);

  optimizedF_ = optimizeFunctionForTest(
      F_, {FunctionPassID::ReplaceZeroScaleFP16QuantNodes, getDCEPassConfig()});

  SaveNode *save = nullptr;
  for (auto &N : optimizedF_->getNodes()) {
    if (N.getKind() == Kinded::Kind::SaveNodeKind) {
      save = llvm::dyn_cast<SaveNode>(&N);
      break;
    }
  }
  ASSERT_TRUE(save);

  auto *optDQ = llvm::dyn_cast<DequantizeNode>(save->getInput());
  ASSERT_TRUE(optDQ);
  auto *optMM = llvm::dyn_cast<MatMulNode>(optDQ->getInput());
  ASSERT_TRUE(optMM);

  SplatNode *splat = llvm::dyn_cast<SplatNode>(optMM->getRHS());
  ASSERT_TRUE(splat);
  EXPECT_EQ(splat->getValue(), 0.f);
  const TypeRef splatQTy = splat->getResult().getType();
  EXPECT_EQ(splatQTy->getScale(), 1.f);
  EXPECT_EQ(splatQTy->getOffset(), 0);
  EXPECT_EQ(splatQTy->getElementType(), weights->getOutput().getElementType());
  EXPECT_EQ(splatQTy->dims(), weights->getOutput().dims());

  bindings_.allocate(input)->getHandle<int8_t>().randomize(-128, 127,
                                                           mod_.getPRNG());
  checkNumericalEquivalence(0.f);
}

TEST_F(GraphOptz, TestEliminateClipsOutsideFP16Range) {
  Placeholder *A = mod_.createPlaceholder(ElemKind::Float16Ty, {5}, "A", false);
  ClipNode *CN1 = F_->createClipMinMaxFP16("clip1", A);
  ClipNode *CN2 = F_->createClip("clip2", A, kMinFP16, kMaxFP16 - 1.f);
  QuantizeNode *QN1 = F_->createQuantize(
      "q1", CN1, mod_.uniqueType(ElemKind::Int8QTy, {5}, 0.3, 5));
  QuantizeNode *QN2 = F_->createQuantize(
      "q2", CN2, mod_.uniqueType(ElemKind::Int8QTy, {5}, 0.3, 5));
  AddNode *AN = F_->createAdd("add", QN1, QN2);
  DequantizeNode *DN = F_->createDequantize("dq", AN, ElemKind::Float16Ty);
  ClipNode *CN3 = F_->createClipMinMaxFP16("clip3", DN);
  F_->createSave("ret", CN3);

  CompilationContext cctx;
  cctx.precisionConfig.clipQuantRangeToFP16 = true;

  optimizedF_ = optimizeFunctionForTest(
      F_, {FunctionPassID::EliminateClipsOutsideFP16Range, getDCEPassConfig()},
      cctx);

  SaveNode *save = nullptr;
  for (auto &N : optimizedF_->getNodes()) {
    if (N.getKind() == Kinded::Kind::SaveNodeKind) {
      save = llvm::dyn_cast<SaveNode>(&N);
      break;
    }
  }
  ASSERT_TRUE(save);

  auto *optDQ = llvm::dyn_cast<DequantizeNode>(save->getInput());
  ASSERT_TRUE(optDQ);
  auto *optAN = llvm::dyn_cast<AddNode>(optDQ->getInput());
  ASSERT_TRUE(optAN);

  auto *optQN1 = llvm::dyn_cast<QuantizeNode>(optAN->getLHS());
  ASSERT_TRUE(optQN1);
  EXPECT_EQ(optQN1->getInput(), A->getOutput());

  auto *optQN2 = llvm::dyn_cast<QuantizeNode>(optAN->getRHS());
  ASSERT_TRUE(optQN2);
  auto *optCN2 = llvm::dyn_cast<ClipNode>(optQN2->getInput());
  ASSERT_TRUE(optCN2);
  EXPECT_EQ(optCN2->getMin(), CN2->getMin());
  EXPECT_EQ(optCN2->getMax(), CN2->getMax());
  EXPECT_EQ(optCN2->getInput(), A->getOutput());

  bindings_.allocate(A)->getHandle<float16_t>().randomize(-128, 127,
                                                          mod_.getPRNG());
  checkNumericalEquivalence(0.f);
}

TEST_F(GraphOptz, TestUpdateQuantReluTypes) {
  auto *input = mod_.createPlaceholder(ElemKind::Int8QTy, {2, 32}, 0.11, -1,
                                       "input", false);
  auto *weights = mod_.createPlaceholder(ElemKind::Int8QTy, {32, 32}, 0.2, 3,
                                         "weights", false);
  auto *bias =
      mod_.createPlaceholder(ElemKind::Int32QTy, {32}, 0.01, 2, "bias", false);
  auto *addW = mod_.createPlaceholder(ElemKind::Int8QTy, {2, 32}, 0.3, -4,
                                      "addw", false);

  auto *fc = F_->createFullyConnected("fc", input, weights, bias);
  auto *qRelu = F_->createRELU("relu", fc->getResult());
  auto *qAdd = F_->createAdd("add", qRelu, addW);
  F_->createSave("save", qAdd);

  updateQuantReluTypes(F_);

  const auto fcRange = fc->getResult().getType()->getQuantizedValueRange();
  const auto reluRange = qRelu->getResult().getType()->getQuantizedValueRange();
  EXPECT_NE(reluRange.first, fcRange.first);
  EXPECT_EQ(reluRange.first, 0);
  EXPECT_EQ(reluRange.second, fcRange.second);
}

TEST_F(GraphOptz, TestUpdateQuantReluTypesChained) {
  auto *input = mod_.createPlaceholder(ElemKind::Int8QTy, {2, 32}, 0.11, -1,
                                       "input", false);
  auto *weights = mod_.createPlaceholder(ElemKind::Int8QTy, {32, 32}, 0.2, 3,
                                         "weights", false);
  auto *bias =
      mod_.createPlaceholder(ElemKind::Int32QTy, {32}, 0.01, 2, "bias", false);
  auto *addW =
      mod_.createPlaceholder(ElemKind::Int8QTy, {128}, 0.3, -4, "addw", false);

  auto *fc = F_->createFullyConnected("fc", input, weights, bias);
  auto *qRelu = F_->createRELU("relu", fc->getResult());
  auto *qConcat = F_->createConcat("concat", {qRelu, qRelu}, 0);
  auto *qReshape = F_->createReshape("reshape", qConcat, {128});
  auto *qAdd = F_->createAdd("add", qReshape, addW);
  F_->createSave("save", qAdd);

  updateQuantReluTypes(F_);

  const auto fcRange = fc->getResult().getType()->getQuantizedValueRange();
  const auto reluRange = qRelu->getResult().getType()->getQuantizedValueRange();
  EXPECT_NE(reluRange.first, fcRange.first);
  EXPECT_EQ(reluRange.first, 0);
  EXPECT_EQ(reluRange.second, fcRange.second);

  // Check that the relu's type now also matches that of the chain of shape
  // users after it.
  const TypeRef qReluTy = qRelu->getResult().getType();
  EXPECT_EQ(qReluTy->getScale(), qConcat->getResult().getType()->getScale());
  EXPECT_EQ(qReluTy->getOffset(), qConcat->getResult().getType()->getOffset());
  EXPECT_EQ(qReluTy->getScale(), qReshape->getResult().getType()->getScale());
  EXPECT_EQ(qReluTy->getOffset(), qReshape->getResult().getType()->getOffset());
}

TEST_F(GraphOptz, SinkReshapeBelowQuantize) {
  auto *I = mod_.createPlaceholder(ElemKind::FloatTy, {32, 64}, "A", false);
  auto *RN = F_->createReshape("reshape", I, {32, 64, 1});
  auto *QN = F_->createQuantize("quantize", RN, ElemKind::Int8QTy, 0.2f, 1);
  auto *SN = F_->createSave("ret", QN);

  optimizedF_ = optimizeFunctionForTest(
      F_, {FunctionPassID::SinkCode, getDCEPassConfig()});

  auto *optSN =
      llvm::dyn_cast<SaveNode>(optimizedF_->getNodeByName(SN->getName()));
  ASSERT_TRUE(optSN);
  auto *optRN = llvm::dyn_cast<ReshapeNode>(optSN->getInput());
  ASSERT_TRUE(optRN);
  EXPECT_EQ(optRN->getResult().getElementType(), ElemKind::Int8QTy);
  EXPECT_EQ(optRN->getResult().getScale(), 0.2f);
  EXPECT_EQ(optRN->getResult().getOffset(), 1);
  EXPECT_EQ(optRN->getResult().dims(), RN->getResult().dims());
  auto *optQN = llvm::dyn_cast<QuantizeNode>(optRN->getInput());
  ASSERT_TRUE(optQN);
  EXPECT_EQ(optQN->getResult().getElementType(), ElemKind::Int8QTy);
  EXPECT_EQ(optQN->getResult().getScale(), 0.2f);
  EXPECT_EQ(optQN->getResult().getOffset(), 1);
  EXPECT_EQ(optQN->getInput().getNode(), I);

  bindings_.allocate(I)->getHandle<float>().randomize(-30, 30, mod_.getPRNG());
  checkNumericalEquivalence(0.f);
}

TEST_F(GraphOptz, SinkReshapeBelowConvertTo) {
  auto *I = mod_.createPlaceholder(ElemKind::FloatTy, {32, 64}, "A", false);
  auto *RN = F_->createReshape("reshape", I, {32, 64, 1});
  auto *CN = F_->createConvertTo("convert", RN, ElemKind::Float16Ty);
  auto *SN = F_->createSave("ret", CN);

  optimizedF_ = optimizeFunctionForTest(
      F_, {FunctionPassID::SinkCode, getDCEPassConfig()});

  auto *optSN =
      llvm::dyn_cast<SaveNode>(optimizedF_->getNodeByName(SN->getName()));
  ASSERT_TRUE(optSN);
  auto *optRN = llvm::dyn_cast<ReshapeNode>(optSN->getInput());
  ASSERT_TRUE(optRN);
  EXPECT_EQ(optRN->getResult().getElementType(), ElemKind::Float16Ty);
  EXPECT_EQ(optRN->getResult().dims(), RN->getResult().dims());
  auto *optCN = llvm::dyn_cast<ConvertToNode>(optRN->getInput());
  ASSERT_TRUE(optCN);
  EXPECT_EQ(optCN->getResult().getElementType(), ElemKind::Float16Ty);
  EXPECT_EQ(optCN->getInput().getNode(), I);

  bindings_.allocate(I)->getHandle<float>().randomize(-30, 30, mod_.getPRNG());
  checkNumericalEquivalence(0.f);
}

TEST_F(GraphOptz, SinkReshapeBelowUnaryEltwiseOps) {
  const dim_t dimsIn[] = {10, 10};
  const dim_t dimsOut[] = {5, 5, 4};

  auto *in = mod_.createPlaceholder(ElemKind::FloatTy, dimsIn, "in", false);
  auto *RN = F_->createReshape("reshape", in, dimsOut);
  auto *AN = F_->createAbs("abs", RN);
  auto *SN = F_->createSin("sin", AN);
  auto *CN = F_->createClip("clip", SN, -4.f, 5.f);
  auto *TN = F_->createTanh("tanh", CN);
  auto *save = F_->createSave("ret", TN);

  optimizedF_ = optimizeFunctionForTest(F_);

  auto *optSave =
      llvm::dyn_cast<SaveNode>(optimizedF_->getNodeByName(save->getName()));
  ASSERT_TRUE(optSave);
  auto *optRN = llvm::dyn_cast<ReshapeNode>(optSave->getInput());
  ASSERT_TRUE(optRN);
  EXPECT_EQ(optRN->getResult().dims(), llvm::makeArrayRef(dimsOut));
  auto *optTN = llvm::dyn_cast<TanhNode>(optRN->getInput());
  ASSERT_TRUE(optTN);
  EXPECT_EQ(optTN->getResult().dims(), llvm::makeArrayRef(dimsIn));
  auto *optCN = llvm::dyn_cast<ClipNode>(optTN->getInput());
  ASSERT_TRUE(optCN);
  EXPECT_FLOAT_EQ(optCN->getMin(), CN->getMin());
  EXPECT_FLOAT_EQ(optCN->getMax(), CN->getMax());
  EXPECT_EQ(optCN->getResult().dims(), llvm::makeArrayRef(dimsIn));
  auto *optSN = llvm::dyn_cast<SinNode>(optCN->getInput());
  ASSERT_TRUE(optSN);
  EXPECT_EQ(optSN->getResult().dims(), llvm::makeArrayRef(dimsIn));
  auto *optAN = llvm::dyn_cast<AbsNode>(optSN->getInput());
  ASSERT_TRUE(optAN);
  EXPECT_EQ(optAN->getResult().dims(), llvm::makeArrayRef(dimsIn));

  bindings_.allocate(in)->getHandle<float>().randomize(-30.f, 30.f,
                                                       mod_.getPRNG());
  checkNumericalEquivalence(0.f);
}

TEST_F(GraphOptz, OptConvertToDequantize) {
  auto *I =
      mod_.createPlaceholder(ElemKind::Int8QTy, {32, 64}, 0.2f, 1, "A", false);
  auto *DN = F_->createDequantize("deq", I, ElemKind::Float16Ty);
  auto *CN = F_->createConvertTo("convert", DN, ElemKind::FloatTy);
  auto *SN = F_->createSave("ret", CN);

  optimizedF_ = optimizeFunctionForTest(
      F_,
      {FunctionPassID::OptimizeOutIntermediateConversions, getDCEPassConfig()});

  auto *optSN =
      llvm::dyn_cast<SaveNode>(optimizedF_->getNodeByName(SN->getName()));
  ASSERT_TRUE(optSN);
  auto *optDN = llvm::dyn_cast<DequantizeNode>(optSN->getInput());
  ASSERT_TRUE(optDN);
  EXPECT_EQ(optDN->getResult().getElementType(), ElemKind::FloatTy);
  EXPECT_EQ(optDN->getResult().dims(), DN->getResult().dims());
  EXPECT_EQ(optDN->getInput().getNode(), I);

  bindings_.allocate(I)->getHandle<int8_t>().randomize(-128, 127,
                                                       mod_.getPRNG());
  checkNumericalEquivalence(0.007f);
}

/// Test that Exp+ReduceSum+Div is replaced with SoftMax.
TEST_F(GraphOptz, FoldExpSumDivIntoSoftmax) {
  Placeholder *input = mod_.createPlaceholder(ElemKind::FloatTy, {1, 10},
                                              "input", /* isTrainable */ false);
  bindings_.allocate(input)->getHandle<float>().randomize(-10, 10,
                                                          mod_.getPRNG());
  auto *exp = F_->createExp("exp", input);
  auto *reduceSum = F_->createBatchedReduceAdd("reduce_sum", exp, {1});
  auto *div = F_->createNodeWithBroadcast<DivNode>("div", 1, exp, reduceSum);
  F_->createSave("save", div);

  EXPECT_EQ(5, F_->getNodes().size());

  optimizedF_ = optimizeFunctionForTest(
      F_, {FunctionPassID::FoldExpSumDivIntoSoftmax, getDCEPassConfig()});

  EXPECT_EQ(2, optimizedF_->getNodes().size());

  EXPECT_EQ(0, countNodeKind(optimizedF_, Kinded::Kind::ExpNodeKind));
  EXPECT_EQ(0, countNodeKind(optimizedF_, Kinded::Kind::DivNodeKind));
  EXPECT_EQ(0,
            countNodeKind(optimizedF_, Kinded::Kind::BatchedReduceAddNodeKind));
  EXPECT_EQ(1, countNodeKind(optimizedF_, Kinded::Kind::SoftMaxNodeKind));

  checkNumericalEquivalence(1e-7f);
}

/// Test that identity Relu is removed.
TEST_F(GraphOptz, RemoveIdentityRelu) {

  Placeholder *input = mod_.createPlaceholder(
      ElemKind::Int8QTy, {20}, 0.123f, -128, "input", /* isTrainable */ false);
  bindings_.allocate(input)->getHandle<int8_t>().randomize(-128, 127,
                                                           mod_.getPRNG());
  auto *relu = F_->createRELU("exp", input);
  F_->createSave("save", relu);

  EXPECT_EQ(2, F_->getNodes().size());
  EXPECT_EQ(1, countNodeKind(F_, Kinded::Kind::ReluNodeKind));
  EXPECT_EQ(1, countNodeKind(F_, Kinded::Kind::SaveNodeKind));

  optimizedF_ = optimizeFunctionForTest(
      F_, {FunctionPassID::RemoveIdentityRelu, getDCEPassConfig()});

  EXPECT_EQ(1, optimizedF_->getNodes().size());
  EXPECT_EQ(0, countNodeKind(optimizedF_, Kinded::Kind::ReluNodeKind));
  EXPECT_EQ(1, countNodeKind(optimizedF_, Kinded::Kind::SaveNodeKind));

  checkNumericalEquivalence(0);
}

/// Test that identity Clip is removed.
TEST_F(GraphOptz, RemoveIdentityClip) {

  Placeholder *input =
      mod_.createPlaceholder(ElemKind::Int8QTy, {20}, 0.023529412f, -128,
                             "input", /* isTrainable */ false);
  bindings_.allocate(input)->getHandle<int8_t>().randomize(-128, 127,
                                                           mod_.getPRNG());
  auto *clip = F_->createClip("exp", input, 0.0f, 6.0f);
  F_->createSave("save", clip);

  EXPECT_EQ(2, F_->getNodes().size());
  EXPECT_EQ(1, countNodeKind(F_, Kinded::Kind::ClipNodeKind));
  EXPECT_EQ(1, countNodeKind(F_, Kinded::Kind::SaveNodeKind));

  optimizedF_ = optimizeFunctionForTest(
      F_, {FunctionPassID::RemoveIdentityClip, getDCEPassConfig()});

  EXPECT_EQ(1, optimizedF_->getNodes().size());
  EXPECT_EQ(0, countNodeKind(optimizedF_, Kinded::Kind::ClipNodeKind));
  EXPECT_EQ(1, countNodeKind(optimizedF_, Kinded::Kind::SaveNodeKind));

  checkNumericalEquivalence(0);
}

/// Test that an identity ResizeNearest is removed.
TEST_F(GraphOptz, OptimizeIdentityResizeNearest) {
  Placeholder *input = mod_.createPlaceholder(ElemKind::FloatTy, {1, 33, 33, 1},
                                              "input", /* isTrainable */ false);
  bindings_.allocate(input)->getHandle<float>().randomize(-10, 10,
                                                          mod_.getPRNG());
  auto *resize = F_->createResizeNearest("resize", input, {1, 1, 1, 1});
  F_->createSave("save", resize);
  EXPECT_EQ(2, F_->getNodes().size());
  optimizedF_ = optimizeFunctionForTest(
      F_, {FunctionPassID::OptimizeResize, getDCEPassConfig()});
  EXPECT_EQ(1, optimizedF_->getNodes().size());
  EXPECT_EQ(1, countNodeKind(optimizedF_, Kinded::Kind::SaveNodeKind));
  checkNumericalEquivalence(1e-7f);
}

/// Test that a ResizeNearest with integer scales is transformed to Tile.
TEST_F(GraphOptz, OptimizeResizeNearest) {
  Placeholder *input = mod_.createPlaceholder(ElemKind::FloatTy, {1, 1, 33, 1},
                                              "input", /* isTrainable */ false);
  bindings_.allocate(input)->getHandle<float>().randomize(-10, 10,
                                                          mod_.getPRNG());
  auto *resize = F_->createResizeNearest("resize", input, {1, 2, 7.787879, 1});
  F_->createSave("save", resize);
  EXPECT_EQ(2, F_->getNodes().size());
  optimizedF_ = optimizeFunctionForTest(
      F_, {FunctionPassID::OptimizeResize, getDCEPassConfig()});
  EXPECT_EQ(3, optimizedF_->getNodes().size());
  EXPECT_EQ(1, countNodeKind(optimizedF_, Kinded::Kind::TileNodeKind));
  EXPECT_EQ(1, countNodeKind(optimizedF_, Kinded::Kind::ResizeNearestNodeKind));
  EXPECT_EQ(1, countNodeKind(optimizedF_, Kinded::Kind::SaveNodeKind));
  checkNumericalEquivalence(1e-7f);
}

/// Test that an identity ResizeBilinear is removed.
TEST_F(GraphOptz, OptimizeIdentityResizeBilinear) {
  Placeholder *input = mod_.createPlaceholder(ElemKind::FloatTy, {1, 33, 33, 1},
                                              "input", /* isTrainable */ false);
  bindings_.allocate(input)->getHandle<float>().randomize(-10, 10,
                                                          mod_.getPRNG());
  auto *resize = F_->createResizeBilinear("resize", input, {1, 1, 1, 1});
  F_->createSave("save", resize);
  EXPECT_EQ(2, F_->getNodes().size());
  optimizedF_ = optimizeFunctionForTest(
      F_, {FunctionPassID::OptimizeResize, getDCEPassConfig()});
  EXPECT_EQ(1, optimizedF_->getNodes().size());
  EXPECT_EQ(1, countNodeKind(optimizedF_, Kinded::Kind::SaveNodeKind));
  checkNumericalEquivalence(1e-7f);
}

/// Test that a ResizeBilinear with integer scales is transformed to Tile.
TEST_F(GraphOptz, OptimizeResizeBilinear) {
  Placeholder *input = mod_.createPlaceholder(ElemKind::FloatTy, {1, 1, 33, 1},
                                              "input", /* isTrainable */ false);
  bindings_.allocate(input)->getHandle<float>().randomize(-10, 10,
                                                          mod_.getPRNG());
  auto *resize = F_->createResizeBilinear("resize", input, {1, 2, 7.787879, 1});
  F_->createSave("save", resize);
  EXPECT_EQ(2, F_->getNodes().size());
  optimizedF_ = optimizeFunctionForTest(
      F_, {FunctionPassID::OptimizeResize, getDCEPassConfig()});
  EXPECT_EQ(3, optimizedF_->getNodes().size());
  EXPECT_EQ(1, countNodeKind(optimizedF_, Kinded::Kind::TileNodeKind));
  EXPECT_EQ(1,
            countNodeKind(optimizedF_, Kinded::Kind::ResizeBilinearNodeKind));
  EXPECT_EQ(1, countNodeKind(optimizedF_, Kinded::Kind::SaveNodeKind));
  checkNumericalEquivalence(1e-7f);
}

/// Test that a InsertTensor which has the Big operand a Splat is replaced
/// with a Touch node when the Small operand fills it entirely.
TEST_F(GraphOptz, OptimizeInsertTensorBigSplat) {
  Type bigTy(ElemKind::FloatTy, {10});
  SplatNode *big = F_->createSplat("splat", &bigTy, 0);
  Placeholder *small = mod_.createPlaceholder(ElemKind::FloatTy, {1}, "input",
                                              /* isTrainable */ false);
  bindings_.allocate(small)->getHandle<float>().randomize(-10, 10,
                                                          mod_.getPRNG());
  auto *insert = F_->createInsertTensor("insert", big, small,
                                        /* start */ {0},
                                        /* count */ 10,
                                        /* axis */ 0);
  F_->createSave("save", insert);
  EXPECT_EQ(3, F_->getNodes().size());
  optimizedF_ = optimizeFunctionForTest(
      F_, {FunctionPassID::OptimizeInsert, getDCEPassConfig()});
  EXPECT_EQ(3, optimizedF_->getNodes().size());
  EXPECT_EQ(1, countNodeKind(optimizedF_, Kinded::Kind::TouchNodeKind));
  EXPECT_EQ(1, countNodeKind(optimizedF_, Kinded::Kind::InsertTensorNodeKind));
  EXPECT_EQ(1, countNodeKind(optimizedF_, Kinded::Kind::SaveNodeKind));
  checkNumericalEquivalence(1e-7f);
}

TEST_F(GraphOptz, sinkQuantizeTransposeMultiUser) {
  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 10, 20, 3}, "input",
                             /* isTrainable */ false);
  auto *T = F_->createTranspose("transpose", input, NHWC2NCHW);
  auto *Q1 = F_->createQuantize("q1", T, ElemKind::Int8QTy, 0.11, 1);
  auto *Q2 = F_->createQuantize("q2", T, ElemKind::Int8QTy, 0.12, 2);
  auto *S1 = F_->createSave("save1", Q1);
  auto *S2 = F_->createSave("save2", Q2);

  optimizedF_ = optimizeFunctionForTest(F_);

  auto *optS1 = findFunctionNodeByName<SaveNode>(optimizedF_, S1->getName());
  auto *optS2 = findFunctionNodeByName<SaveNode>(optimizedF_, S2->getName());

  // Check that transpose has been sunk below quantize now for both.
  EXPECT_TRUE(llvm::isa<TransposeNode>(optS1->getInput()));
  EXPECT_TRUE(llvm::isa<TransposeNode>(optS2->getInput()));

  bindings_.allocate(input)->getHandle<float>().randomize(-10, 10,
                                                          mod_.getPRNG());
  checkNumericalEquivalence(0.f);
}

TEST_F(GraphOptz, skipSinkQuantizeTransposeMultiUser) {
  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 10, 20, 3}, "input",
                             /* isTrainable */ false);
  auto *T = F_->createTranspose("transpose", input, NHWC2NCHW);
  auto *Q = F_->createQuantize("quant", T, ElemKind::Int8QTy, 0.11, 1);
  F_->createSave("save1", Q);
  F_->createSave("save2", T);

  optimizedF_ = optimizeFunctionForTest(F_);

  // Verify the graph hasn't changed.
  EXPECT_EQ(F_->toString(/* skipUsersForStorage */ false, /* skipName */ true),
            optimizedF_->toString(/* skipUsersForStorage */ false,
                                  /* skipName */ true));
}

TEST_F(GraphOptz, MergeMatMulsOnLHSWhenSkippingOne) {
  Placeholder *LHS1 =
      mod_.createPlaceholder(ElemKind::FloatTy, {10, 10}, "LHS1", false);
  Placeholder *LHS2 =
      mod_.createPlaceholder(ElemKind::FloatTy, {30, 10}, "LHS2", false);
  Placeholder *LHS3 =
      mod_.createPlaceholder(ElemKind::FloatTy, {20, 10}, "LHS3", false);
  Placeholder *RHS =
      mod_.createPlaceholder(ElemKind::FloatTy, {10, 15}, "RHS", false);
  bindings_.allocate(LHS1)->getHandle().randomize(-1.f, 1.f, mod_.getPRNG());
  bindings_.allocate(LHS2)->getHandle().randomize(-1.f, 1.f, mod_.getPRNG());
  bindings_.allocate(LHS3)->getHandle().randomize(-1.f, 1.f, mod_.getPRNG());
  bindings_.allocate(RHS)->getHandle().randomize(-1.f, 1.f, mod_.getPRNG());

  // Chain a bunch of nodes together for LHS2 to prevent dependency analysis
  // from allowing merging for MM2 below.
  Node *sigLHS2 = LHS2;
  for (size_t i = 0, e = 7; i < e; i++) {
    sigLHS2 = F_->createSigmoid("s_lhs2", sigLHS2);
  }

  Node *MM1 = F_->createMatMul("mm1", LHS1, RHS);
  Node *MM2 = F_->createMatMul("mm2", sigLHS2, RHS);
  Node *MM3 = F_->createMatMul("mm3", LHS3, RHS);

  F_->createSave("save1", MM1);
  F_->createSave("save2", MM2);
  F_->createSave("save3", MM3);
  ASSERT_TRUE(F_->verify());

  optimizedF_ = optimizeFunctionForTest(
      F_, {FunctionPassID::MergeMatMulOnLHS, getDCEPassConfig()});
  ASSERT_TRUE(optimizedF_->verify());

  // Expect three matmuls -> two matmuls, because mm1 and mm3 were merged.
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::MatMulNodeKind), 3);
  EXPECT_EQ(countNodeKind(optimizedF_, Kinded::Kind::MatMulNodeKind), 2);

  checkNumericalEquivalence(0.f);
}

TEST_F(GraphOptz, MergeMatMulsOnRHSWhenSkippingOne) {
  Placeholder *LHS =
      mod_.createPlaceholder(ElemKind::FloatTy, {40, 10}, "LHS", false);
  Placeholder *RHS1 =
      mod_.createPlaceholder(ElemKind::FloatTy, {10, 15}, "RHS1", false);
  Placeholder *RHS2 =
      mod_.createPlaceholder(ElemKind::FloatTy, {10, 20}, "RHS2", false);
  Placeholder *RHS3 =
      mod_.createPlaceholder(ElemKind::FloatTy, {10, 30}, "RHS3", false);
  bindings_.allocate(LHS)->getHandle().randomize(-1.f, 1.f, mod_.getPRNG());
  bindings_.allocate(RHS1)->getHandle().randomize(-1.f, 1.f, mod_.getPRNG());
  bindings_.allocate(RHS2)->getHandle().randomize(-1.f, 1.f, mod_.getPRNG());
  bindings_.allocate(RHS3)->getHandle().randomize(-1.f, 1.f, mod_.getPRNG());

  // Chain a bunch of nodes together for RHS2 to prevent dependency analysis
  // from allowing merging for MM2 below.
  Node *sigRHS2 = RHS2;
  for (size_t i = 0, e = 7; i < e; i++) {
    sigRHS2 = F_->createSigmoid("s_rhs2", sigRHS2);
  }

  Node *MM1 = F_->createMatMul("mm1", LHS, RHS1);
  Node *MM2 = F_->createMatMul("mm2", LHS, sigRHS2);
  Node *MM3 = F_->createMatMul("mm3", LHS, RHS3);

  F_->createSave("save1", MM1);
  F_->createSave("save2", MM2);
  F_->createSave("save3", MM3);
  ASSERT_TRUE(F_->verify());

  optimizedF_ = optimizeFunctionForTest(
      F_, {FunctionPassID::MergeMatMulOnRHS, getDCEPassConfig()});
  ASSERT_TRUE(optimizedF_->verify());

  // Expect three matmuls -> two matmuls, because mm1 and mm3 were merged.
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::MatMulNodeKind), 3);
  EXPECT_EQ(countNodeKind(optimizedF_, Kinded::Kind::MatMulNodeKind), 2);

  checkNumericalEquivalence(0.f);
}
