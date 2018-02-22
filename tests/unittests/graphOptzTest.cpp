// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "glow/Graph/Graph.h"
#include "glow/Graph/Node.h"
#include "glow/Graph/Nodes.h"
#include "glow/IR/IR.h"
#include "glow/Optimizer/Optimizer.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>

using namespace glow;

TEST(GraphOptz, DCE) {
  Module mod;
  Function &G = *mod.createFunction("foo");
  IRFunction M(&G);
  Node *K = mod.createVariable(ElemKind::FloatTy, {4, 320, 200, 3}, "input");

  for (int i = 0; i < 40; i++) {
    K = G.createRELU("relu", K);
    // Add a graph structure that diverges and converges, to catch algorithms
    // that perform a dump recursive scan.
    K = G.createArithmetic("arith", K, K, ArithmeticNode::Mode::Add);
  }

  // Check that we know how many nodes we've created.
  EXPECT_EQ(G.getNodes().size(), 80);

  // Optimize all of the dead code.
  ::glow::optimize(&G, CompilationMode::Infer);

  //  All of the nodes are gone.
  EXPECT_EQ(G.getNodes().size(), 0);
  EXPECT_EQ(mod.getVars().size(), 0);
}

TEST(GraphOptz, liveCodeNotEliminated) {
  Module mod;
  Function &G = *mod.createFunction("foo");

  IRFunction M(&G);
  Node *K = mod.createVariable(ElemKind::FloatTy, {4, 320, 200, 3}, "input");
  auto *Ex = mod.createVariable(ElemKind::IndexTy, {4, 1}, "Ex");

  for (int i = 0; i < 40; i++) {
    K = G.createRELU("relu", K);
    K = G.createArithmetic("arith", K, K, ArithmeticNode::Mode::Add);
  }
  K = G.createSoftMax("Regression", K, Ex);
  G.createSave("ret", K);

  // Check that we know how many nodes we've created.
  EXPECT_EQ(G.getNodes().size(), 82);

  // This should not optimize code because none is dead.
  ::glow::optimize(&G, CompilationMode::Infer);

  //  Nothing got optimized.
  EXPECT_EQ(G.getNodes().size(), 82);
  EXPECT_EQ(mod.getVars().size(), 3);
}

TEST(GraphOptz, optimizeBatchNormAfterConv) {
  Module mod;
  Function &G = *mod.createFunction("foo");

  IRFunction M(&G);
  Node *A = mod.createVariable(ElemKind::FloatTy, {1, 10, 20, 3}, "A",
                               Variable::VisibilityKind::Public,
                               Variable::TrainKind::None);
  Node *CV = G.createConv("conv", A, 16, 5, 1, 2);
  Node *BN = G.createBatchNormalization("batch", CV, 3, 0.0001, 0.9);
  G.createSave("ret", BN);

  EXPECT_EQ(G.getNodes().size(), 3);

  ::glow::optimize(&G, CompilationMode::Infer);
  EXPECT_EQ(G.getNodes().size(), 2);
}

TEST(GraphOptz, transposePrivateVariable) {
  Module mod;
  Function *F = mod.createFunction("foo");

  Node *A = mod.createVariable(ElemKind::FloatTy, {1, 10, 20, 3}, "A",
                               Variable::VisibilityKind::Private,
                               Variable::TrainKind::None);
  Node *T = F->createTranspose("transpose", A, {0, 3, 1, 2});
  F->createSave("ret", T);
  EXPECT_EQ(F->getNodes().size(), 2);

  ::glow::optimize(F, CompilationMode::Infer);
  EXPECT_EQ(F->getNodes().size(), 1);
}

TEST(GraphOptz, BatchNormAfterConvNotOptimizeForTrain) {
  Module mod;
  Function &G = *mod.createFunction("foo");

  IRFunction M(&G);
  Node *A = mod.createVariable(ElemKind::FloatTy, {1, 10, 20, 3}, "A",
                               Variable::VisibilityKind::Public,
                               Variable::TrainKind::None);
  Node *CV = G.createConv("conv", A, 16, 5, 1, 2);
  Node *BN = G.createBatchNormalization("batch", CV, 3, 0.0001, 0.9);
  G.createSave("ret", BN);

  EXPECT_EQ(G.getNodes().size(), 3);

  ::glow::optimize(&G, CompilationMode::Train);
  EXPECT_EQ(G.getNodes().size(), 3);
}

TEST(GraphOptz, batchNormAfterConvNotOptimizeWhenMoreThanOneUseOfConv) {
  Module mod;
  Function &G = *mod.createFunction("foo");

  IRFunction M(&G);
  Node *A = mod.createVariable(ElemKind::FloatTy, {1, 10, 20, 3}, "A",
                               Variable::VisibilityKind::Public,
                               Variable::TrainKind::None);

  Node *CV = G.createConv("conv", A, 16, 5, 1, 2);
  Node *BN = G.createBatchNormalization("batch", CV, 3, 0.0001, 0.9);
  G.createSave("ret", BN);
  G.createSave("ret", CV);

  EXPECT_EQ(G.getNodes().size(), 4);

  ::glow::optimize(&G, CompilationMode::Infer);
  EXPECT_EQ(G.getNodes().size(), 4);
}

TEST(GraphOptz, sinkTransposeBelowOptimizeBatchNorm) {
  Module mod;
  Function &G = *mod.createFunction("foo");

  IRFunction M(&G);
  Node *A = mod.createVariable(ElemKind::FloatTy, {1, 5, 10, 15}, "input",
                               Variable::VisibilityKind::Public,
                               Variable::TrainKind::None);
  Node *T = G.createTranspose("transpose", A, {0, 3, 1, 2});
  Node *BN = G.createBatchNormalization("batch", T, 3, 0.0001, 0.9);
  Node *O = G.createSave("ret", BN);

  EXPECT_EQ(G.getNodes().size(), 3);

  ::glow::optimize(&G, CompilationMode::Infer);

  // Expecting Transpose->Output rather than BN->Output.
  EXPECT_TRUE(llvm::isa<SaveNode>(O));
  EXPECT_TRUE(
      llvm::isa<TransposeNode>(llvm::dyn_cast<SaveNode>(O)->getInput()));

  EXPECT_EQ(G.getNodes().size(), 3);
}

TEST(GraphOptz, sinkTransposeBelowRELU) {
  Module mod;
  Function &G = *mod.createFunction("foo");

  IRFunction M(&G);
  Node *A = mod.createVariable(ElemKind::FloatTy, {1, 5, 10, 15}, "input",
                               Variable::VisibilityKind::Public,
                               Variable::TrainKind::None);
  Node *T = G.createTranspose("transpose", A, {0, 3, 1, 2});
  Node *K = G.createRELU("relu", T);
  Node *O = G.createSave("ret", K);

  EXPECT_EQ(G.getNodes().size(), 3);

  ::glow::optimize(&G, CompilationMode::Infer);

  // Expecting Transpose->Output rather than RELU->Output.
  EXPECT_TRUE(llvm::isa<SaveNode>(O));
  EXPECT_TRUE(
      llvm::isa<TransposeNode>(llvm::dyn_cast<SaveNode>(O)->getInput()));

  EXPECT_EQ(G.getNodes().size(), 3);
}

TEST(GraphOptz, sinkTransposeBelowSigmoid) {
  Module mod;
  Function &G = *mod.createFunction("foo");

  IRFunction M(&G);
  Node *A = mod.createVariable(ElemKind::FloatTy, {1, 5, 10, 15}, "input",
                               Variable::VisibilityKind::Public,
                               Variable::TrainKind::None);
  Node *T = G.createTranspose("transpose", A, {0, 3, 1, 2});
  Node *SI = G.createSigmoid("sigmoid", T);
  Node *O = G.createSave("ret", SI);

  EXPECT_EQ(G.getNodes().size(), 3);

  ::glow::optimize(&G, CompilationMode::Infer);

  // Expecting Transpose->Output rather than Sigmoid->Output.
  EXPECT_TRUE(llvm::isa<SaveNode>(O));
  EXPECT_TRUE(
      llvm::isa<TransposeNode>(llvm::dyn_cast<SaveNode>(O)->getInput()));

  EXPECT_EQ(G.getNodes().size(), 3);
}

TEST(GraphOptz, sinkTransposeBelowTanh) {
  Module mod;
  Function &G = *mod.createFunction("foo");

  IRFunction M(&G);
  Node *A = mod.createVariable(ElemKind::FloatTy, {1, 5, 10, 15}, "input",
                               Variable::VisibilityKind::Public,
                               Variable::TrainKind::None);
  Node *T = G.createTranspose("transpose", A, {0, 3, 1, 2});
  Node *TN = G.createTanh("tanh", T);
  Node *O = G.createSave("ret", TN);

  EXPECT_EQ(G.getNodes().size(), 3);

  ::glow::optimize(&G, CompilationMode::Infer);

  // Expecting Transpose->Output rather than Sigmoid->Output.
  EXPECT_TRUE(llvm::isa<SaveNode>(O));
  EXPECT_TRUE(
      llvm::isa<TransposeNode>(llvm::dyn_cast<SaveNode>(O)->getInput()));

  EXPECT_EQ(G.getNodes().size(), 3);
}

TEST(GraphOptz, cancelTwoTransposes) {
  Module mod;
  Function &G = *mod.createFunction("foo");

  IRFunction M(&G);
  Node *A = mod.createVariable(ElemKind::FloatTy, {1, 5, 10, 15}, "input",
                               Variable::VisibilityKind::Public,
                               Variable::TrainKind::None);
  Node *T1 = G.createTranspose("transpose", A, {0, 2, 3, 1});
  Node *T2 = G.createTranspose("transpose", T1, {0, 3, 1, 2});
  Node *K = G.createRELU("relu", T2);
  G.createSave("ret", K);

  EXPECT_EQ(G.getNodes().size(), 4);

  ::glow::optimize(&G, CompilationMode::Infer);

  EXPECT_EQ(G.getNodes().size(), 2);
}

TEST(GraphOptz, dontCancelTwoTransposesIfNotMatching) {
  Module mod;
  Function &G = *mod.createFunction("foo");

  IRFunction M(&G);
  Node *A = mod.createVariable(ElemKind::FloatTy, {1, 5, 10, 15}, "input",
                               Variable::VisibilityKind::Public,
                               Variable::TrainKind::None);
  Node *T1 = G.createTranspose("transpose", A, {0, 2, 3, 1});
  Node *T2 = G.createTranspose("transpose", T1, {0, 1, 2, 3});
  Node *K = G.createRELU("relu", T2);
  G.createSave("ret", K);

  EXPECT_EQ(G.getNodes().size(), 4);

  ::glow::optimize(&G, CompilationMode::Infer);

  EXPECT_EQ(G.getNodes().size(), 4);
}

TEST(GraphOptz, sinkTransposeBelowArithmeticNodes) {
  Module mod;
  Function &G = *mod.createFunction("foo");

  IRFunction M(&G);
  Node *A1 = mod.createVariable(ElemKind::FloatTy, {1, 5, 10, 15}, "input1",
                                Variable::VisibilityKind::Public,
                                Variable::TrainKind::None);
  Node *A2 = mod.createVariable(ElemKind::FloatTy, {1, 5, 10, 15}, "input2",
                                Variable::VisibilityKind::Public,
                                Variable::TrainKind::None);
  Node *T1 = G.createTranspose("transpose1", A1, {0, 3, 1, 2});
  Node *T2 = G.createTranspose("transpose2", A2, {0, 3, 1, 2});
  Node *K = G.createArithmetic("arith", T1, T2, ArithmeticNode::Mode::Add);
  Node *O = G.createSave("ret", K);

  EXPECT_EQ(G.getNodes().size(), 4);

  ::glow::optimize(&G, CompilationMode::Infer);

  // Expecting Transpose->Output rather than Add->Output.
  EXPECT_TRUE(llvm::isa<SaveNode>(O));
  EXPECT_TRUE(
      llvm::isa<TransposeNode>(llvm::dyn_cast<SaveNode>(O)->getInput()));

  EXPECT_EQ(G.getNodes().size(), 3);
}

TEST(GraphOptz, sinkReluBelowConcatNodes) {
  Module mod;
  Function &G = *mod.createFunction("foo");

  IRFunction M(&G);
  Node *A1 = mod.createVariable(ElemKind::FloatTy, {1, 5, 10, 15}, "input1",
                                Variable::VisibilityKind::Public,
                                Variable::TrainKind::None);
  Node *A2 = mod.createVariable(ElemKind::FloatTy, {1, 5, 10, 15}, "input2",
                                Variable::VisibilityKind::Public,
                                Variable::TrainKind::None);
  Node *R1 = G.createRELU("relu1", A1);
  Node *R2 = G.createRELU("relu2", A2);
  Node *CN = G.createConcat("concat", {R1, R2}, 1);
  Node *O = G.createSave("ret", CN);

  EXPECT_EQ(G.getNodes().size(), 4);

  ::glow::optimize(&G, CompilationMode::Infer);

  // Expecting RELU->Output rather than Concat->Output.
  EXPECT_TRUE(llvm::isa<SaveNode>(O));
  EXPECT_TRUE(llvm::isa<ReluNode>(llvm::dyn_cast<SaveNode>(O)->getInput()));

  EXPECT_EQ(G.getNodes().size(), 3);
}

TEST(GraphOptz, sinkTransposeBelowConcatNodes) {
  Module mod;
  Function &G = *mod.createFunction("foo");

  IRFunction M(&G);
  Node *A1 = mod.createVariable(ElemKind::FloatTy, {1, 5, 10, 15}, "input1",
                                Variable::VisibilityKind::Public,
                                Variable::TrainKind::None);
  Node *A2 = mod.createVariable(ElemKind::FloatTy, {1, 5, 10, 15}, "input2",
                                Variable::VisibilityKind::Public,
                                Variable::TrainKind::None);
  Node *T1 = G.createTranspose("transpose", A1, {0, 2, 3, 1});
  Node *T2 = G.createTranspose("transpose", A2, {0, 2, 3, 1});
  Node *CN = G.createConcat("concat", {T1, T2}, 1);
  Node *O = G.createSave("ret", CN);

  EXPECT_EQ(G.getNodes().size(), 4);

  ::glow::optimize(&G, CompilationMode::Infer);

  // Expecting Transpose->Output rather than Concat->Output.
  EXPECT_TRUE(llvm::isa<SaveNode>(O));
  EXPECT_TRUE(
      llvm::isa<TransposeNode>(llvm::dyn_cast<SaveNode>(O)->getInput()));

  EXPECT_EQ(G.getNodes().size(), 3);
}

TEST(GraphOptz, poolBelowReluSwapped) {
  Module mod;
  Function &G = *mod.createFunction("foo");

  IRFunction M(&G);
  Node *A = mod.createVariable(ElemKind::FloatTy, {1, 5, 10, 15}, "input",
                               Variable::VisibilityKind::Public,
                               Variable::TrainKind::None);
  Node *R = G.createRELU("relu", A);
  Node *PL = G.createPool("pool", R, PoolNode::Mode::Max, 1, 10, 20);
  Node *O = G.createSave("ret", PL);

  EXPECT_EQ(G.getNodes().size(), 3);

  ::glow::optimize(&G, CompilationMode::Infer);

  // Expecting RELU->Output rather than Pool->Output.
  EXPECT_TRUE(llvm::isa<SaveNode>(O));
  EXPECT_TRUE(llvm::isa<ReluNode>(llvm::dyn_cast<SaveNode>(O)->getInput()));

  EXPECT_EQ(G.getNodes().size(), 3);
}

TEST(GraphOptz, poolBelowReluNotSwappedIfModeNotMax) {
  Module mod;
  Function &G = *mod.createFunction("foo");

  IRFunction M(&G);
  Node *A = mod.createVariable(ElemKind::FloatTy, {1, 5, 10, 15}, "input",
                               Variable::VisibilityKind::Public,
                               Variable::TrainKind::None);
  Node *R = G.createRELU("relu", A);
  Node *PL = G.createPool("pool", R, PoolNode::Mode::Avg, 1, 10, 20);
  Node *O = G.createSave("ret", PL);

  EXPECT_EQ(G.getNodes().size(), 3);

  ::glow::optimize(&G, CompilationMode::Infer);

  // Expecting Pool->Output (no swap).
  EXPECT_TRUE(llvm::isa<SaveNode>(O));
  EXPECT_TRUE(llvm::isa<PoolNode>(llvm::dyn_cast<SaveNode>(O)->getInput()));

  EXPECT_EQ(G.getNodes().size(), 3);
}

TEST(GraphOptz, poolBelowReluNotSwappedIfNotSingleUse) {
  Module mod;
  Function &G = *mod.createFunction("foo");

  IRFunction M(&G);
  Node *A = mod.createVariable(ElemKind::FloatTy, {1, 5, 10, 15}, "input",
                               Variable::VisibilityKind::Public,
                               Variable::TrainKind::None);
  Node *R = G.createRELU("relu", A);
  Node *PL = G.createPool("pool", R, PoolNode::Mode::Max, 1, 10, 20);
  Node *O = G.createSave("ret", PL);
  G.createSave("ret", R);

  EXPECT_EQ(G.getNodes().size(), 4);

  ::glow::optimize(&G, CompilationMode::Infer);

  // Expecting Pool->Output (no swap).
  EXPECT_TRUE(llvm::isa<SaveNode>(O));
  EXPECT_TRUE(llvm::isa<PoolNode>(llvm::dyn_cast<SaveNode>(O)->getInput()));

  EXPECT_EQ(G.getNodes().size(), 4);
}

TEST(GraphOptz, mergeConcatNodes) {
  Module mod;
  Function &G = *mod.createFunction("foo");

  IRFunction M(&G);
  Node *A1 = mod.createVariable(ElemKind::FloatTy, {1, 5, 10, 15}, "input1",
                                Variable::VisibilityKind::Public,
                                Variable::TrainKind::None);
  Node *A2 = mod.createVariable(ElemKind::FloatTy, {1, 5, 10, 15}, "input2",
                                Variable::VisibilityKind::Public,
                                Variable::TrainKind::None);
  Node *A3 = mod.createVariable(ElemKind::FloatTy, {1, 5, 10, 15}, "input3",
                                Variable::VisibilityKind::Public,
                                Variable::TrainKind::None);
  Node *A4 = mod.createVariable(ElemKind::FloatTy, {1, 1, 10, 15}, "input4",
                                Variable::VisibilityKind::Public,
                                Variable::TrainKind::None);

  Node *CN1 = G.createConcat("concat1", {A1, A2}, 1);
  Node *CN2 = G.createConcat("concat2", {A1, CN1}, 1);
  Node *CN3 = G.createConcat("concat3", {A4}, 2);
  Node *CN4 = G.createConcat("concat4", {A3, CN2, CN3}, 1);
  Node *O = G.createSave("ret", CN4);

  EXPECT_EQ(G.getNodes().size(), 5);

  ::glow::optimize(&G, CompilationMode::Train);

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
  EXPECT_TRUE(std::find(G.getNodes().begin(), G.getNodes().end(), CN1) ==
              G.getNodes().end());

  // CN2 should be merged into a new CN4 and removed by the optimizations.
  EXPECT_TRUE(std::find(G.getNodes().begin(), G.getNodes().end(), CN2) ==
              G.getNodes().end());

  // CN3 should not be merged into CN4 and should not be removed,
  // because CN4 and CN3 have a different dimension parameter.
  EXPECT_TRUE(std::find(G.getNodes().begin(), G.getNodes().end(), CN3) !=
              G.getNodes().end());

  // The CN4 concat node should be replaced by a merged concat node.
  EXPECT_TRUE(std::find(G.getNodes().begin(), G.getNodes().end(), CN4) ==
              G.getNodes().end());

  EXPECT_EQ(G.getNodes().size(), 3);
}

TEST(GraphOptz, CSE) {
  Module mod;
  Function &G = *mod.createFunction("foo");

  IRFunction M(&G);
  Node *A1 = mod.createVariable(ElemKind::FloatTy, {1, 5, 10, 15}, "input1",
                                Variable::VisibilityKind::Public,
                                Variable::TrainKind::None);
  Node *A2 = mod.createVariable(ElemKind::FloatTy, {1, 5, 10, 15}, "input2",
                                Variable::VisibilityKind::Public,
                                Variable::TrainKind::None);

  Node *CN1 = G.createConcat("concat1", {A1, A2}, 1);
  Node *CN2 = G.createConcat("concat2", {A1, A2}, 1);
  Node *CN3 = G.createConcat("concat3", {CN1, CN2}, 2);
  Node *O = G.createSave("ret", CN3);

  EXPECT_EQ(G.getNodes().size(), 4);

  ::glow::optimize(&G, CompilationMode::Train);

  EXPECT_TRUE(llvm::isa<SaveNode>(O));

  auto *CN =
      llvm::dyn_cast<ConcatNode>(llvm::dyn_cast<SaveNode>(O)->getInput());
  EXPECT_TRUE(CN);

  // The merged ConcatNode should have 2 inputs.
  EXPECT_EQ(CN->getInputs().size(), 2);

  // CN1 should not be removed.
  EXPECT_TRUE(std::find(G.getNodes().begin(), G.getNodes().end(), CN1) !=
              G.getNodes().end());

  // CSE should replace CN2 by CN1 and remove CN2.
  EXPECT_TRUE(std::find(G.getNodes().begin(), G.getNodes().end(), CN2) ==
              G.getNodes().end());

  EXPECT_EQ(G.getNodes().size(), 3);
}

TEST(GraphOptz, SliceOfSplatNode) {
  Module mod;
  Function &G = *mod.createFunction("foo");

  IRFunction M(&G);
  Type t(ElemKind::FloatTy, {1000, 1000, 1000});
  Node *Z = G.createSplat("zero", &t, 0.);
  Node *S = G.createSlice("slice", Z, {5, 15, 42}, {99, 88, 77});
  Node *O = G.createSave("ret", S);

  EXPECT_EQ(G.getNodes().size(), 3);

  ::glow::optimize(&G, CompilationMode::Train);

  EXPECT_EQ(G.getNodes().size(), 2);

  EXPECT_TRUE(llvm::isa<SaveNode>(O));

  auto *CN = llvm::dyn_cast<SplatNode>(llvm::dyn_cast<SaveNode>(O)->getInput());
  EXPECT_TRUE(CN);

  EXPECT_TRUE(CN->getType()->dims().equals({94, 73, 35}));
}

TEST(GraphOptz, SliceOfSplatNodeChain) {
  for (int shouldReverse = 0; shouldReverse <= 1; shouldReverse++) {
    Module mod;
    Function &G = *mod.createFunction("foo");

    IRFunction M(&G);
    Type t(ElemKind::FloatTy, {1000, 1000, 1000});
    Node *Z = G.createSplat("zero", &t, 0.);
    Node *S1 = G.createSlice("slice1", Z, {5, 15, 42}, {99, 88, 77});
    Node *S2 = G.createSlice("slice2", S1, {1, 1, 1}, {2, 3, 4});
    G.createSave("ret", S2);

    if (shouldReverse) {
      auto &nodes = G.getNodes();
      reverse(nodes.begin(), nodes.end());
    }

    EXPECT_EQ(G.getNodes().size(), 4);

    ::glow::optimize(&G, CompilationMode::Train);

    // This test illustrates some inconsistency in the optimization.
    // Chain splats are not guaranteed to be optimized.
    EXPECT_EQ(G.getNodes().size(), shouldReverse ? 3 : 2);
  }
}

TEST(GraphOptz, DCEPublicVars) {
  Module mod;
  Function *F = mod.createFunction("main");
  IRFunction M(F);
  mod.createVariable(ElemKind::FloatTy, {4, 320, 200, 3}, "input",
                     Variable::VisibilityKind::Public);

  EXPECT_EQ(mod.getVars().size(), 1);

  // Optimize all of the dead code.
  ::glow::optimize(F, CompilationMode::Infer);

  //  Public nodes should not be deleted.
  EXPECT_EQ(mod.getVars().size(), 1);
}
