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

TEST_F(GraphOptz, transposePrivateVariable) {
  Node *A =
      mod_.createVariable(ElemKind::FloatTy, {1, 10, 20, 3}, "A",
                          VisibilityKind::Private, Variable::TrainKind::None);
  Node *T = F_->createTranspose("transpose", A, {0, 3, 1, 2});
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
  Node *T = F_->createTranspose("transpose", A, {0, 3, 1, 2});
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
  Node *T = F_->createTranspose("transpose", A, {0, 3, 1, 2});
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
  Node *T = F_->createTranspose("transpose", A, {0, 3, 1, 2});
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
  Node *T = F_->createTranspose("transpose", A, {0, 3, 1, 2});
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
  Node *T1 = F_->createTranspose("transpose", A, {0, 2, 3, 1});
  Node *T2 = F_->createTranspose("transpose", T1, {0, 3, 1, 2});
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
  Node *T1 = F_->createTranspose("transpose", A, {0, 2, 3, 1});
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
  Node *T1 = F_->createTranspose("transpose1", A1, {0, 3, 1, 2});
  Node *T2 = F_->createTranspose("transpose2", A2, {0, 3, 1, 2});
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
  Node *T1 = F_->createTranspose("transpose", A1, {0, 2, 3, 1});
  Node *T2 = F_->createTranspose("transpose", A2, {0, 2, 3, 1});
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
  EXPECT_TRUE(std::find(F_->getNodes().begin(), F_->getNodes().end(), CN1) ==
              F_->getNodes().end());

  // CN2 should be merged into a new CN4 and removed by the optimizations.
  EXPECT_TRUE(std::find(F_->getNodes().begin(), F_->getNodes().end(), CN2) ==
              F_->getNodes().end());

  // CN3 should not be merged into CN4 and should not be removed,
  // because CN4 and CN3 have a different dimension parameter.
  EXPECT_TRUE(std::find(F_->getNodes().begin(), F_->getNodes().end(), CN3) !=
              F_->getNodes().end());

  // The CN4 concat node should be replaced by a merged concat node.
  EXPECT_TRUE(std::find(F_->getNodes().begin(), F_->getNodes().end(), CN4) ==
              F_->getNodes().end());

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
  EXPECT_TRUE(std::find(F_->getNodes().begin(), F_->getNodes().end(), CN1) !=
              F_->getNodes().end());

  // CSE should replace CN2 by CN1 and remove CN2.
  EXPECT_TRUE(std::find(F_->getNodes().begin(), F_->getNodes().end(), CN2) ==
              F_->getNodes().end());

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
      reverse(nodes.begin(), nodes.end());
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

TEST_F(GraphOptz, DCEPublicVars) {
  mod_.createVariable(ElemKind::FloatTy, {4, 320, 200, 3}, "input",
                      VisibilityKind::Public);

  EXPECT_EQ(mod_.getVars().size(), 1);

  // Optimize all of the dead code.
  ::glow::optimize(F_, CompilationMode::Infer);

  //  Public nodes should not be deleted.
  EXPECT_EQ(mod_.getVars().size(), 1);
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

  // All rescales must be fused into arithmetic operations above.
  ::glow::optimize(F_, CompilationMode::Infer);

  EXPECT_EQ(F_->getNodes().size(), 8);

  EXPECT_EQ(add->getNthInput(0).getType(), rescaleOutTy);
  EXPECT_EQ(sub->getNthInput(0).getType(), rescaleOutTy);
  EXPECT_EQ(mul->getNthInput(0).getType(), rescaleOutTy);
  EXPECT_EQ(div->getNthInput(0).getType(), rescaleOutTy);
}
