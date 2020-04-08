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

#include "glow/Graph/Graph.h"
#include "glow/Graph/Node.h"
#include "glow/Graph/Nodes.h"
#include "glow/Graph/PlaceholderBindings.h"
#include "glow/Optimizer/GraphOptimizer/GraphOptimizer.h"

#include "gtest/gtest.h"

using namespace glow;

/// Tests using this harness generally behave as following:
/// 1. Create some Function which we want to compile/verify is optimized
///    correctly.
/// 2. Compile that Function both with and without optimizations.
/// 3. Verify the structure of the optimized Function is as expected.
/// 4. Run both the optimized and unoptimized versions on the NNPI backend and
///    verify they have the same-ish results.
class NNPIOptPipelineTest : public GraphOptz {
public:
  NNPIOptPipelineTest() : GraphOptz("NNPI") {}

protected:
  /// Clones the current \ref F_ to \ref optimizedF_, and then compiles the
  /// Module containing both Functions, skipping optimizations for \ref
  /// F_. Currently checks that no partitioning occurred. Sets \ref
  /// alreadyCompiled_ so that \ref checkNumericalEquivalence() can skip
  /// compilation when comparing execution of the optimized and unoptimized
  /// Function.
  void cloneAndCompile() {
    optimizedF_ = F_->clone(F_->getName().str() + "_opt");
    cctx_.optimizationOpts.onlyLowerFuns.insert(F_);
    EE_.compile(cctx_);
    // Expect that partitioning did not occur, and so there should be two
    // functions: the original one, and the now-optimized one.
    ASSERT_EQ(mod_.getFunctions().size(), 2);
    alreadyCompiled_ = true;
  }
};

/// Note: This differs from NNPIOptPipelineTest only in that cloneAndCompile()
/// will clone into unoptimizedF_ instead of optimizedF_. This means that we can
/// correctly specify per-node opts in backendSpecificNodeInfo based on the
/// original F_ that is constructed (otherwise it didn't apply because the nodes
/// were cloned into optimizedF_).
class NNPIOptPipelineTestNodeOpts : public GraphOptz {
public:
  NNPIOptPipelineTestNodeOpts() : GraphOptz("NNPI") {}

protected:
  /// Same as cloneAndCompile() in NNPIOptPipelineTest, but use F_ as the
  /// optimize Function, and unoptimizedF_ as the unoptimized.
  void cloneAndCompile() {
    // Only set optimizedF_ so that checkNumericalEquivalence from GraphOptz
    // works later.
    optimizedF_ = F_;
    unoptimizedF_ = F_->clone(F_->getName().str() + "_unopt");
    cctx_.optimizationOpts.onlyLowerFuns.insert(unoptimizedF_);
    EE_.compile(cctx_);
    // Expect that partitioning did not occur, and so there should be two
    // functions: the original one, and the now-optimized one.
    ASSERT_EQ(mod_.getFunctions().size(), 2);
    alreadyCompiled_ = true;
  }

  Function *unoptimizedF_;
};

/// Test that the backend correctly removed Clips that are in between FCs and
/// activations, and therefore block fusion from occurring.
TEST_F(NNPIOptPipelineTest, RemoveClipBlockingFCReluFusion) {
  auto *input =
      mod_.createPlaceholder(ElemKind::Float16Ty, {10, 6}, "input", false);
  auto *weights =
      F_->getParent()->createConstant(ElemKind::Float16Ty, {6, 20}, "weights");
  auto *bias =
      F_->getParent()->createConstant(ElemKind::Float16Ty, {20}, "bias");
  weights->getPayloadMutable().getHandle<float16_t>().randomize(-1.0, 1.0,
                                                                mod_.getPRNG());
  bias->getPayloadMutable().getHandle<float16_t>().randomize(-1.0, 1.0,
                                                             mod_.getPRNG());

  auto *FC = F_->createFullyConnected("fc", input, weights, bias);
  auto *clipFC = F_->createClipMinMaxFP16("clipFC", FC);
  auto *RN = F_->createRELU("relu", clipFC);
  auto *clipRelu = F_->createClipMinMaxFP16("clipRelu", RN);
  F_->createSave("ret", clipRelu);
  const float float16Max = clipFC->getMax();

  EXPECT_EQ(F_->getNodes().size(), 5);

  cloneAndCompile();

  EXPECT_EQ(optimizedF_->getNodes().size(), 4);

  SaveNode *optSave = nullptr;
  for (auto &N : optimizedF_->getNodes()) {
    if ((optSave = llvm::dyn_cast<SaveNode>(&N))) {
      break;
    }
  }
  ASSERT_TRUE(optSave);

  // Check that we got rid of the Clip between the FC and Relu.
  {
    ClipNode *clipRelu = llvm::dyn_cast<ClipNode>(optSave->getInput());
    ASSERT_TRUE(clipRelu);
    // Note: Min here is 0, because relu changed the Clip's min range.
    EXPECT_EQ(clipRelu->getMin(), 0);
    EXPECT_EQ(clipRelu->getMax(), float16Max);
    ReluNode *RN = llvm::dyn_cast<ReluNode>(clipRelu->getInput());
    ASSERT_TRUE(RN);
    FullyConnectedNode *FC = llvm::dyn_cast<FullyConnectedNode>(RN->getInput());
    ASSERT_TRUE(FC);
  }

  bindings_.allocate(input)->getHandle<float16_t>().randomize(-1.0, 1.0,
                                                              mod_.getPRNG());

  checkNumericalEquivalence(/* allowedError */ 0.f);
}

/// Test data parallel and model parallel splitting inside
/// of NNPIPrivateTransforms.cpp for FC/RELU
TEST_F(NNPIOptPipelineTest, SplitParallelizationTestFCReluNNPI) {
  auto *input =
      mod_.createPlaceholder(ElemKind::Float16Ty, {32, 512}, "input", false);
  auto *weights = F_->getParent()->createConstant(ElemKind::Float16Ty,
                                                  {512, 512}, "weights");
  auto *bias =
      F_->getParent()->createConstant(ElemKind::Float16Ty, {512}, "bias");
  weights->getPayloadMutable().getHandle<float16_t>().randomize(-1.0, 1.0,
                                                                mod_.getPRNG());
  bias->getPayloadMutable().getHandle<float16_t>().randomize(-1.0, 1.0,
                                                             mod_.getPRNG());

  auto *FC = F_->createFullyConnected("fc", input, weights, bias);
  auto *RN = F_->createRELU("relu", FC);
  F_->createSave("ret", RN);

  cctx_.backendOpts.backendSpecificOpts["NNPINumParallelChunks"] =
      std::to_string(8);
  cloneAndCompile();

  EXPECT_LT(F_->getNodes().size(), optimizedF_->getNodes().size());
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::FullyConnectedNodeKind), 1);
  EXPECT_EQ(countNodeKind(optimizedF_, Kinded::Kind::FullyConnectedNodeKind),
            8);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::ReluNodeKind), 1);
  EXPECT_EQ(countNodeKind(optimizedF_, Kinded::Kind::ReluNodeKind), 8);

  SaveNode *optSave = nullptr;
  for (auto &N : optimizedF_->getNodes()) {
    if ((optSave = llvm::dyn_cast<SaveNode>(&N))) {
      break;
    }
  }
  ASSERT_TRUE(optSave);

  bindings_.allocate(input)->getHandle<float16_t>().randomize(-1.0, 1.0,
                                                              mod_.getPRNG());

  checkNumericalEquivalence(/* allowedError */ 0.f);
}

/// Test data parallel and model parallel splitting inside
/// of NNPIPrivateTransforms.cpp for FC/RELU/Clip
TEST_F(NNPIOptPipelineTest, SplitParallelizationTestFCReluClipNNPI) {
  auto *input =
      mod_.createPlaceholder(ElemKind::Float16Ty, {32, 512}, "input", false);
  auto *weights = F_->getParent()->createConstant(ElemKind::Float16Ty,
                                                  {512, 512}, "weights");
  auto *bias =
      F_->getParent()->createConstant(ElemKind::Float16Ty, {512}, "bias");
  weights->getPayloadMutable().getHandle<float16_t>().randomize(-1.0, 1.0,
                                                                mod_.getPRNG());
  bias->getPayloadMutable().getHandle<float16_t>().randomize(-1.0, 1.0,
                                                             mod_.getPRNG());

  auto *FC = F_->createFullyConnected("fc", input, weights, bias);
  auto *RN = F_->createRELU("relu", FC);
  auto *CLP = F_->createClip("clip", RN, 0.0f, 5.0f);
  F_->createSave("ret", CLP);

  cctx_.backendOpts.backendSpecificOpts["NNPINumParallelChunks"] =
      std::to_string(8);
  cloneAndCompile();

  EXPECT_LT(F_->getNodes().size(), optimizedF_->getNodes().size());
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::FullyConnectedNodeKind), 1);
  EXPECT_EQ(countNodeKind(optimizedF_, Kinded::Kind::FullyConnectedNodeKind),
            8);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::ReluNodeKind), 1);
  EXPECT_EQ(countNodeKind(optimizedF_, Kinded::Kind::ReluNodeKind), 8);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::ClipNodeKind), 1);
  EXPECT_EQ(countNodeKind(optimizedF_, Kinded::Kind::ClipNodeKind), 8);

  SaveNode *optSave = nullptr;
  for (auto &N : optimizedF_->getNodes()) {
    if ((optSave = llvm::dyn_cast<SaveNode>(&N))) {
      break;
    }
  }
  ASSERT_TRUE(optSave);

  bindings_.allocate(input)->getHandle<float16_t>().randomize(-1.0, 1.0,
                                                              mod_.getPRNG());

  checkNumericalEquivalence(/* allowedError */ 0.f);
}

/// Make sure no nodes are split when the parameter is 1
TEST_F(NNPIOptPipelineTest, NoSplitTestFCReluNNPI) {
  auto *input =
      mod_.createPlaceholder(ElemKind::Float16Ty, {32, 512}, "input", false);
  auto *weights = F_->getParent()->createConstant(ElemKind::Float16Ty,
                                                  {512, 512}, "weights");
  auto *bias =
      F_->getParent()->createConstant(ElemKind::Float16Ty, {512}, "bias");
  weights->getPayloadMutable().getHandle<float16_t>().randomize(-1.0, 1.0,
                                                                mod_.getPRNG());
  bias->getPayloadMutable().getHandle<float16_t>().randomize(-1.0, 1.0,
                                                             mod_.getPRNG());

  auto *FC = F_->createFullyConnected("fc", input, weights, bias);
  auto *RN = F_->createRELU("relu", FC);
  F_->createSave("ret", RN);

  cctx_.backendOpts.backendSpecificOpts["NNPINumParallelChunks"] =
      std::to_string(1);
  cloneAndCompile();

  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::FullyConnectedNodeKind), 1);
  EXPECT_EQ(countNodeKind(optimizedF_, Kinded::Kind::FullyConnectedNodeKind),
            1);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::ReluNodeKind), 1);
  EXPECT_EQ(countNodeKind(optimizedF_, Kinded::Kind::ReluNodeKind), 1);

  SaveNode *optSave = nullptr;
  for (auto &N : optimizedF_->getNodes()) {
    if ((optSave = llvm::dyn_cast<SaveNode>(&N))) {
      break;
    }
  }
  ASSERT_TRUE(optSave);

  bindings_.allocate(input)->getHandle<float16_t>().randomize(-1.0, 1.0,
                                                              mod_.getPRNG());

  checkNumericalEquivalence(/* allowedError */ 0.f);
}

/// Test data parallel and model parallel splitting inside
/// of NNPIPrivateTransforms.cpp for Transpose.
TEST_F(NNPIOptPipelineTest, SplitParallelizationTestTransposeNNPI) {
  auto *input = mod_.createPlaceholder(ElemKind::Float16Ty, {32, 128, 128},
                                       "input", false);

  auto *TP = F_->createTranspose("tp", input, {0, 2, 1});
  F_->createSave("ret", TP);

  cctx_.backendOpts.backendSpecificOpts["NNPINumParallelChunks"] =
      std::to_string(3);
  cloneAndCompile();

  EXPECT_LT(F_->getNodes().size(), optimizedF_->getNodes().size());
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::TransposeNodeKind), 1);
  EXPECT_EQ(countNodeKind(optimizedF_, Kinded::Kind::TransposeNodeKind), 3);

  SaveNode *optSave = nullptr;
  for (auto &N : optimizedF_->getNodes()) {
    if ((optSave = llvm::dyn_cast<SaveNode>(&N))) {
      break;
    }
  }
  ASSERT_TRUE(optSave);

  bindings_.allocate(input)->getHandle<float16_t>().randomize(-1.0, 1.0,
                                                              mod_.getPRNG());

  checkNumericalEquivalence(/* allowedError */ 0.f);
}

/// Test data parallel and model parallel splitting inside
/// of NNPIPrivateTransforms.cpp for Transpose and Relu
TEST_F(NNPIOptPipelineTest, SplitParallelizationTestBatchMatMulReluNNPI) {
  auto *input1 =
      mod_.createPlaceholder(ElemKind::Float16Ty, {8, 64, 64}, "input", false);
  auto *input2 =
      mod_.createPlaceholder(ElemKind::Float16Ty, {8, 64, 64}, "input", false);

  auto *BMM = F_->createBatchMatMul("bmm", input1, input2);
  auto *BMM_relu = F_->createRELU("relu", BMM);
  F_->createSave("ret", BMM_relu);

  cctx_.backendOpts.backendSpecificOpts["NNPINumParallelChunks"] =
      std::to_string(3);
  cloneAndCompile();

  EXPECT_LT(F_->getNodes().size(), optimizedF_->getNodes().size());
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::BatchMatMulNodeKind), 1);
  EXPECT_EQ(countNodeKind(optimizedF_, Kinded::Kind::BatchMatMulNodeKind), 3);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::ReluNodeKind), 1);
  EXPECT_EQ(countNodeKind(optimizedF_, Kinded::Kind::ReluNodeKind), 3);

  bindings_.allocate(input1)->getHandle<float16_t>().randomize(-1.0, 1.0,
                                                               mod_.getPRNG());
  bindings_.allocate(input2)->getHandle<float16_t>().randomize(-1.0, 1.0,
                                                               mod_.getPRNG());

  checkNumericalEquivalence(/* allowedError */ 0.f);
}

/// Test data parallel splitting for Mul and Relu
TEST_F(NNPIOptPipelineTest, SplitParallelizationTestMulReluNNPI) {
  auto *input1 =
      mod_.createPlaceholder(ElemKind::Float16Ty, {8, 4096}, "input", false);
  auto *input2 =
      mod_.createPlaceholder(ElemKind::Float16Ty, {8, 4096}, "input", false);

  auto *M = F_->createMul("mul", input1, input2);
  auto *M_relu = F_->createRELU("relu", M);
  F_->createSave("ret", M_relu);

  cctx_.backendOpts.backendSpecificOpts["NNPINumParallelChunks"] =
      std::to_string(3);
  cloneAndCompile();

  EXPECT_LT(F_->getNodes().size(), optimizedF_->getNodes().size());
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::MulNodeKind), 1);
  EXPECT_EQ(countNodeKind(optimizedF_, Kinded::Kind::MulNodeKind), 3);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::ReluNodeKind), 1);
  EXPECT_EQ(countNodeKind(optimizedF_, Kinded::Kind::ReluNodeKind), 3);

  bindings_.allocate(input1)->getHandle<float16_t>().randomize(-1.0, 1.0,
                                                               mod_.getPRNG());
  bindings_.allocate(input2)->getHandle<float16_t>().randomize(-1.0, 1.0,
                                                               mod_.getPRNG());

  checkNumericalEquivalence(/* allowedError */ 0.f);
}

/// Test data parallel splitting for Tanh and Relu
TEST_F(NNPIOptPipelineTest, SplitParallelizationTestTanhReluNNPI) {
  auto *input1 =
      mod_.createPlaceholder(ElemKind::Float16Ty, {8, 4096}, "input", false);

  auto *TH = F_->createTanh("tanh", input1);
  auto *TH_relu = F_->createRELU("relu", TH);
  F_->createSave("ret", TH_relu);

  cctx_.backendOpts.backendSpecificOpts["NNPINumParallelChunks"] =
      std::to_string(3);
  cloneAndCompile();

  EXPECT_LT(F_->getNodes().size(), optimizedF_->getNodes().size());
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::TanhNodeKind), 1);
  EXPECT_EQ(countNodeKind(optimizedF_, Kinded::Kind::TanhNodeKind), 3);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::ReluNodeKind), 1);
  EXPECT_EQ(countNodeKind(optimizedF_, Kinded::Kind::ReluNodeKind), 3);

  bindings_.allocate(input1)->getHandle<float16_t>().randomize(-1.0, 1.0,
                                                               mod_.getPRNG());

  checkNumericalEquivalence(/* allowedError */ 0.f);
}

static void setupSplitParallelizationTestFCReluNNPI(
    Module &mod_, Function *F_, CompilationContext &cctx_,
    PlaceholderBindings &bindings_, const std::string &parKind) {
  auto *input1 =
      mod_.createPlaceholder(ElemKind::Float16Ty, {512, 32}, "input1", false);
  auto *input2 =
      mod_.createPlaceholder(ElemKind::Float16Ty, {512, 32}, "input2", false);
  auto *weights = F_->getParent()->createConstant(ElemKind::Float16Ty,
                                                  {512, 512}, "weights");
  auto *bias =
      F_->getParent()->createConstant(ElemKind::Float16Ty, {512}, "bias");
  weights->getPayloadMutable().getHandle<float16_t>().randomize(-1.0, 1.0,
                                                                mod_.getPRNG());
  bias->getPayloadMutable().getHandle<float16_t>().randomize(-1.0, 1.0,
                                                             mod_.getPRNG());

  auto *TN = F_->createTranspose("transpose", input1, {1, 0});
  auto *FC = F_->createFullyConnected("fc", TN, weights, bias);
  auto *RN = F_->createRELU("relu", FC);
  auto *SN = F_->createSigmoid("sigmoid", RN);
  F_->createSave("ret", SN);

  auto *AN = F_->createAdd("add", input1, input2);
  F_->createSave("add_save", AN);

  bindings_.allocate(input1)->getHandle<float16_t>().randomize(-1.0, 1.0,
                                                               mod_.getPRNG());
  bindings_.allocate(input2)->getHandle<float16_t>().randomize(-1.0, 1.0,
                                                               mod_.getPRNG());

  auto &nodeInfo = cctx_.backendOpts.backendSpecificNodeInfo[F_];
  // Setup some parallelization.
  nodeInfo[FC]["NNPI_numParallelChunks"].push_back("8");
  nodeInfo[FC]["NNPI_parallelTransformKind"].push_back(parKind);
  nodeInfo[RN]["NNPI_numParallelChunks"].push_back("8");
  nodeInfo[RN]["NNPI_parallelTransformKind"].push_back(parKind);
  // Add some extra edges.
  nodeInfo[TN]["NNPI_extraEdges"].push_back(FC->getName().str() + "@3");
  nodeInfo[TN]["NNPI_extraEdges"].push_back(FC->getName().str() + "@7");
  nodeInfo[RN]["NNPI_extraEdges"].push_back(SN->getName().str());
  nodeInfo[TN]["NNPI_extraEdges"].push_back(SN->getName().str());
  nodeInfo[AN]["NNPI_extraEdges"].push_back(TN->getName().str());
  // Assign some ops to cores.
  nodeInfo[TN]["NNPI_coreAssignments"].push_back("3");
  nodeInfo[FC]["NNPI_coreAssignments"].push_back("2");
  nodeInfo[FC]["NNPI_coreAssignments"].push_back("0");
  nodeInfo[FC]["NNPI_coreAssignments"].push_back("10");
  nodeInfo[FC]["NNPI_coreAssignments"].push_back("5");
  nodeInfo[FC]["NNPI_coreAssignments"].push_back("1");
  nodeInfo[FC]["NNPI_coreAssignments"].push_back("7");
  nodeInfo[FC]["NNPI_coreAssignments"].push_back("3");
  nodeInfo[FC]["NNPI_coreAssignments"].push_back("0");
}

/// Test model parallel splitting inside of NNPIPrivateTransforms.cpp for
/// FC/RELU
TEST_F(NNPIOptPipelineTestNodeOpts, ModelSplitParallelizationTestFCReluNNPI) {
  setupSplitParallelizationTestFCReluNNPI(mod_, F_, cctx_, bindings_, "Model");
  cloneAndCompile();

  EXPECT_LT(unoptimizedF_->getNodes().size(), optimizedF_->getNodes().size());
  EXPECT_EQ(countNodeKind(unoptimizedF_, Kinded::Kind::FullyConnectedNodeKind),
            1);
  EXPECT_EQ(countNodeKind(optimizedF_, Kinded::Kind::FullyConnectedNodeKind),
            8);
  EXPECT_EQ(countNodeKind(unoptimizedF_, Kinded::Kind::ReluNodeKind), 1);
  EXPECT_EQ(countNodeKind(optimizedF_, Kinded::Kind::ReluNodeKind), 8);
  EXPECT_EQ(countNodeKind(optimizedF_, Kinded::Kind::ConcatNodeKind), 1);

  SaveNode *optSave = nullptr;
  for (auto &N : optimizedF_->getNodes()) {
    if ((optSave = llvm::dyn_cast<SaveNode>(&N))) {
      break;
    }
  }
  ASSERT_TRUE(optSave);

  // Data parallel split concats results on 1st dim.
  SigmoidNode *SN = llvm::dyn_cast<SigmoidNode>(optSave->getInput());
  ASSERT_TRUE(SN);
  ConcatNode *CN = llvm::dyn_cast<ConcatNode>(SN->getInput());
  ASSERT_TRUE(CN);
  EXPECT_EQ(CN->getDim(), 1);

  checkNumericalEquivalence(/* allowedError */ 0.f);
}

/// Test data parallel splitting inside of NNPIPrivateTransforms.cpp for FC/RELU
TEST_F(NNPIOptPipelineTestNodeOpts, DataSplitParallelizationTestFCReluNNPI) {
  setupSplitParallelizationTestFCReluNNPI(mod_, F_, cctx_, bindings_, "Data");
  cloneAndCompile();

  EXPECT_LT(unoptimizedF_->getNodes().size(), optimizedF_->getNodes().size());
  EXPECT_EQ(countNodeKind(unoptimizedF_, Kinded::Kind::FullyConnectedNodeKind),
            1);
  EXPECT_EQ(countNodeKind(optimizedF_, Kinded::Kind::FullyConnectedNodeKind),
            8);
  EXPECT_EQ(countNodeKind(unoptimizedF_, Kinded::Kind::ReluNodeKind), 1);
  EXPECT_EQ(countNodeKind(optimizedF_, Kinded::Kind::ReluNodeKind), 8);
  EXPECT_EQ(countNodeKind(optimizedF_, Kinded::Kind::ConcatNodeKind), 1);

  SaveNode *optSave = nullptr;
  for (auto &N : optimizedF_->getNodes()) {
    if ((optSave = llvm::dyn_cast<SaveNode>(&N))) {
      break;
    }
  }
  ASSERT_TRUE(optSave);

  // Data parallel split concats results on 0th dim.
  SigmoidNode *SN = llvm::dyn_cast<SigmoidNode>(optSave->getInput());
  ASSERT_TRUE(SN);
  ConcatNode *CN = llvm::dyn_cast<ConcatNode>(SN->getInput());
  ASSERT_TRUE(CN);
  EXPECT_EQ(CN->getDim(), 0);

  checkNumericalEquivalence(/* allowedError */ 0.f);
}

/// Test Relu is not parallelized when following an op that was not
/// parallelized.
TEST_F(NNPIOptPipelineTest, NoParallelizationTestAddReluNNPI) {
  auto *input =
      mod_.createPlaceholder(ElemKind::Float16Ty, {32, 1024}, "input", false);

  auto *AN = F_->createAdd("add", input, input);
  auto *RN = F_->createRELU("relu", AN);
  F_->createSave("ret", RN);

  // Set 8, but Add won't be parallelized, and so Relu shouldn't be either.
  cctx_.backendOpts.backendSpecificOpts["NNPINumParallelChunks"] =
      std::to_string(8);
  cloneAndCompile();

  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::AddNodeKind), 1);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::ReluNodeKind), 1);
  EXPECT_EQ(countNodeKind(optimizedF_, Kinded::Kind::AddNodeKind), 1);
  EXPECT_EQ(countNodeKind(optimizedF_, Kinded::Kind::ReluNodeKind), 1);
}
