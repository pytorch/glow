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

/// Test that when we rely on the optimization pipeline to do lowering based on
/// precision that the resulting ops are using the precision we expect. Note
/// that this test is not intended to run, just to compile to examine the output
/// Function, and thus \ref checkNumericalEquivalence() will fail if called.
class NNPIOptPipelineTestLowering : public GraphOptz {
public:
  NNPIOptPipelineTestLowering() : GraphOptz("NNPI") {}

protected:
  /// Disabled for this harness.
  void checkNumericalEquivalence(float allowedError = 0.0001) override {
    FAIL() << "checkNumericalEquivalence not supported for tests using "
              "NNPIOptPipelineTestLowering";
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

/// Find and \returns a Save found in \p F by name \p saveName. \returns nullptr
/// if not found.
SaveNode *getSaveByName(Function *F, llvm::StringRef saveName) {
  for (auto &N : F->getNodes()) {
    if (SaveNode *SN = llvm::dyn_cast<SaveNode>(&N)) {
      if (SN->getName() == saveName) {
        return SN;
      }
    }
  }
  return nullptr;
}

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
  auto *save = F_->createSave("ret", clipRelu);
  const float float16Max = clipFC->getMax();

  EXPECT_EQ(F_->getNodes().size(), 5);

  cloneAndCompile();

  // (Clip -> Relu -> Clip) -> Clip
  EXPECT_EQ(optimizedF_->getNodes().size(), 3);

  SaveNode *optSave = getSaveByName(optimizedF_, save->getName());
  ASSERT_TRUE(optSave);

  // Check that we got rid of the Clip between the FC and Relu.
  {
    ClipNode *clipRelu = llvm::dyn_cast<ClipNode>(optSave->getInput());
    ASSERT_TRUE(clipRelu);
    // Note: Min here is 0, because relu changed the Clip's min range.
    EXPECT_EQ(clipRelu->getMin(), 0);
    EXPECT_EQ(clipRelu->getMax(), float16Max);
    FullyConnectedNode *FC =
        llvm::dyn_cast<FullyConnectedNode>(clipRelu->getInput());
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
  auto *save = F_->createSave("ret", RN);

  cctx_.backendOpts.backendSpecificOpts["NNPINumParallelChunks"] =
      std::to_string(8);
  cloneAndCompile();

  EXPECT_LT(F_->getNodes().size(), optimizedF_->getNodes().size());
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::FullyConnectedNodeKind), 1);
  EXPECT_EQ(countNodeKind(optimizedF_, Kinded::Kind::FullyConnectedNodeKind),
            8);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::ReluNodeKind), 1);
  EXPECT_EQ(countNodeKind(optimizedF_, Kinded::Kind::ReluNodeKind), 8);

  SaveNode *optSave = getSaveByName(optimizedF_, save->getName());
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
  auto *save = F_->createSave("ret", CLP);

  cctx_.backendOpts.backendSpecificOpts["NNPINumParallelChunks"] =
      std::to_string(8);
  cloneAndCompile();

  EXPECT_LT(F_->getNodes().size(), optimizedF_->getNodes().size());
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::FullyConnectedNodeKind), 1);
  EXPECT_EQ(countNodeKind(optimizedF_, Kinded::Kind::FullyConnectedNodeKind),
            8);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::ReluNodeKind), 1);
  // Note: FC->Relu->Clip is optimized to FC->Clip, so no Relu is left.
  EXPECT_EQ(countNodeKind(optimizedF_, Kinded::Kind::ReluNodeKind), 0);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::ClipNodeKind), 1);
  EXPECT_EQ(countNodeKind(optimizedF_, Kinded::Kind::ClipNodeKind), 8);

  SaveNode *optSave = getSaveByName(optimizedF_, save->getName());
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
  auto *save = F_->createSave("ret", RN);

  cctx_.backendOpts.backendSpecificOpts["NNPINumParallelChunks"] =
      std::to_string(1);
  cloneAndCompile();

  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::FullyConnectedNodeKind), 1);
  EXPECT_EQ(countNodeKind(optimizedF_, Kinded::Kind::FullyConnectedNodeKind),
            1);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::ReluNodeKind), 1);
  EXPECT_EQ(countNodeKind(optimizedF_, Kinded::Kind::ReluNodeKind), 1);

  SaveNode *optSave = getSaveByName(optimizedF_, save->getName());
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
  auto *save = F_->createSave("ret", TP);

  cctx_.backendOpts.backendSpecificOpts["NNPINumParallelChunks"] =
      std::to_string(3);
  cloneAndCompile();

  EXPECT_LT(F_->getNodes().size(), optimizedF_->getNodes().size());
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::TransposeNodeKind), 1);
  EXPECT_EQ(countNodeKind(optimizedF_, Kinded::Kind::TransposeNodeKind), 3);

  SaveNode *optSave = getSaveByName(optimizedF_, save->getName());
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

/// Test model parallelism for MatMul
TEST_F(NNPIOptPipelineTest, SplitParallelizationTestMatMul) {
  auto *input1 =
      mod_.createPlaceholder(ElemKind::Float16Ty, {8, 64}, "input", false);
  auto *input2 =
      mod_.createPlaceholder(ElemKind::Float16Ty, {64, 16}, "input", false);

  auto *MM = F_->createMatMul("mm", input1, input2);
  F_->createSave("ret", MM);

  cctx_.backendOpts.backendSpecificOpts["NNPINumParallelChunks"] =
      std::to_string(3);
  cloneAndCompile();

  EXPECT_LT(F_->getNodes().size(), optimizedF_->getNodes().size());
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::MatMulNodeKind), 1);
  EXPECT_EQ(countNodeKind(optimizedF_, Kinded::Kind::MatMulNodeKind), 3);

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
TEST_F(NNPIOptPipelineTest, SplitParallelizationTestTanhReluGeluNNPI) {
  auto *input1 =
      mod_.createPlaceholder(ElemKind::Float16Ty, {8, 4096}, "input", false);

  auto *TH = F_->createTanh("tanh", input1);
  auto *TH_relu = F_->createRELU("relu", TH);
  auto *TH_relu_gelu = F_->createGELU("gelu", TH_relu);
  F_->createSave("ret", TH_relu_gelu);

  cctx_.backendOpts.backendSpecificOpts["NNPINumParallelChunks"] =
      std::to_string(3);
  cloneAndCompile();

  EXPECT_LT(F_->getNodes().size(), optimizedF_->getNodes().size());
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::TanhNodeKind), 1);
  EXPECT_EQ(countNodeKind(optimizedF_, Kinded::Kind::TanhNodeKind), 3);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::ReluNodeKind), 1);
  EXPECT_EQ(countNodeKind(optimizedF_, Kinded::Kind::ReluNodeKind), 3);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::GeluNodeKind), 1);
  EXPECT_EQ(countNodeKind(optimizedF_, Kinded::Kind::GeluNodeKind), 3);

  bindings_.allocate(input1)->getHandle<float16_t>().randomize(-1.0, 1.0,
                                                               mod_.getPRNG());

  checkNumericalEquivalence(/* allowedError */ 0.f);
}

static void setupSplitParallelizationTestFCReluNNPI(
    Module &mod_, Function *F_, CompilationContext &cctx_,
    PlaceholderBindings &bindings_, const std::string &parKind,
    const bool addParallelizationHints, const bool addPlacementHints) {
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

  auto *CI = F_->createConcat("concat", {input1, input2}, 1);
  auto *TN = F_->createTranspose("transpose", CI, {1, 0});
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
  // Placement and parallelization cannot happen at the same time
  ASSERT_FALSE(addParallelizationHints && addPlacementHints);

  if (addParallelizationHints) {
    // Setup some parallelization.
    nodeInfo[FC]["NNPI_numParallelChunks"].push_back("8");
    nodeInfo[FC]["NNPI_parallelTransformKind"].push_back(parKind);
    nodeInfo[RN]["NNPI_numParallelChunks"].push_back("8");
    nodeInfo[RN]["NNPI_parallelTransformKind"].push_back(parKind);
  }

  if (addPlacementHints) {
    // Add some extra edges.
    nodeInfo[AN]["NNPI_extraEdgesTargetName"].push_back(CI->getName().str());
    nodeInfo[AN]["NNPI_extraEdgesTargetSuffix"].push_back("@copy_1");
    nodeInfo[CI]["NNPI_extraEdgesTargetName"].push_back(TN->getName().str());
    nodeInfo[CI]["NNPI_extraEdgesSourceSuffix"].push_back("@copy_1");
    nodeInfo[CI]["NNPI_extraEdgesTargetName"].push_back(FC->getName().str());
    nodeInfo[CI]["NNPI_extraEdgesSourceSuffix"].push_back("@copy_1");
    nodeInfo[AN]["NNPI_extraEdgesTargetName"].push_back(CI->getName().str());
    nodeInfo[AN]["NNPI_extraEdgesTargetSuffix"].push_back("@copy_1");
    nodeInfo[TN]["NNPI_extraEdgesTargetName"].push_back(FC->getName().str());
    nodeInfo[TN]["NNPI_extraEdgesTargetName"].push_back(SN->getName().str());
    nodeInfo[FC]["NNPI_extraEdgesTargetName"].push_back(RN->getName().str());

    // Assign some ops to cores.
    nodeInfo[TN]["NNPI_coreAssignments"].push_back("3");
    nodeInfo[FC]["NNPI_coreAssignments"].push_back("2");
    nodeInfo[CI]["NNPI_coreAssignments"].push_back("1");
    nodeInfo[CI]["NNPI_coreAssignmentsSuffix"].push_back("@copy_0");
    nodeInfo[CI]["NNPI_coreAssignments"].push_back("3");
    nodeInfo[CI]["NNPI_coreAssignmentsSuffix"].push_back("@copy_1");

    // Assign some tensors to memory levels
    nodeInfo[TN]["NNPI_tensorAssignmentNames"].push_back("tensor1");
    nodeInfo[TN]["NNPI_tensorAssignmentValues"].push_back("LLC");
    nodeInfo[TN]["NNPI_tensorAssignmentNames"].push_back("tensor2");
    nodeInfo[TN]["NNPI_tensorAssignmentValues"].push_back("SRAM");
    nodeInfo[TN]["NNPI_tensorAssignmentNames"].push_back("tensor3");
    nodeInfo[TN]["NNPI_tensorAssignmentValues"].push_back("DRAM");
  }
}

/// Test model parallel splitting inside of NNPIPrivateTransforms.cpp for
/// FC/RELU
TEST_F(NNPIOptPipelineTestNodeOpts, ModelSplitParallelizationTestFCReluNNPI) {
  setupSplitParallelizationTestFCReluNNPI(mod_, F_, cctx_, bindings_, "Model",
                                          /* addParallelizationHints */ true,
                                          /* addPlacementHints */ false);
  cloneAndCompile();

  EXPECT_LT(unoptimizedF_->getNodes().size(), optimizedF_->getNodes().size());
  EXPECT_EQ(countNodeKind(unoptimizedF_, Kinded::Kind::FullyConnectedNodeKind),
            1);
  EXPECT_EQ(countNodeKind(optimizedF_, Kinded::Kind::FullyConnectedNodeKind),
            8);
  EXPECT_EQ(countNodeKind(unoptimizedF_, Kinded::Kind::ReluNodeKind), 1);
  EXPECT_EQ(countNodeKind(optimizedF_, Kinded::Kind::ReluNodeKind), 8);
  EXPECT_EQ(countNodeKind(optimizedF_, Kinded::Kind::ConcatNodeKind), 2);

  SaveNode *optSave = getSaveByName(optimizedF_, "ret_save");
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
  setupSplitParallelizationTestFCReluNNPI(mod_, F_, cctx_, bindings_, "Data",
                                          /* addParallelizationHints */ true,
                                          /* addPlacementHints */ false);
  cloneAndCompile();

  EXPECT_LT(unoptimizedF_->getNodes().size(), optimizedF_->getNodes().size());
  EXPECT_EQ(countNodeKind(unoptimizedF_, Kinded::Kind::FullyConnectedNodeKind),
            1);
  EXPECT_EQ(countNodeKind(optimizedF_, Kinded::Kind::FullyConnectedNodeKind),
            8);
  EXPECT_EQ(countNodeKind(unoptimizedF_, Kinded::Kind::ReluNodeKind), 1);
  EXPECT_EQ(countNodeKind(optimizedF_, Kinded::Kind::ReluNodeKind), 8);
  EXPECT_EQ(countNodeKind(optimizedF_, Kinded::Kind::ConcatNodeKind), 2);

  SaveNode *optSave = getSaveByName(optimizedF_, "ret_save");
  ASSERT_TRUE(optSave);

  // Data parallel split concats results on 0th dim.
  SigmoidNode *SN = llvm::dyn_cast<SigmoidNode>(optSave->getInput());
  ASSERT_TRUE(SN);
  ConcatNode *CN = llvm::dyn_cast<ConcatNode>(SN->getInput());
  ASSERT_TRUE(CN);
  EXPECT_EQ(CN->getDim(), 0);

  checkNumericalEquivalence(/* allowedError */ 0.f);
}

/// Test placement hints
TEST_F(NNPIOptPipelineTestNodeOpts, PlacementTestFCReluNNPI) {
  setupSplitParallelizationTestFCReluNNPI(mod_, F_, cctx_, bindings_, "Data",
                                          /* addParallelizationHints */ false,
                                          /* addPlacementHints */ true);
  cloneAndCompile();

  EXPECT_EQ(unoptimizedF_->getNodes().size(), optimizedF_->getNodes().size());
  EXPECT_EQ(countNodeKind(unoptimizedF_, Kinded::Kind::FullyConnectedNodeKind),
            1);
  EXPECT_EQ(countNodeKind(optimizedF_, Kinded::Kind::FullyConnectedNodeKind),
            1);
  EXPECT_EQ(countNodeKind(unoptimizedF_, Kinded::Kind::ReluNodeKind), 1);
  EXPECT_EQ(countNodeKind(optimizedF_, Kinded::Kind::ReluNodeKind), 1);
  EXPECT_EQ(countNodeKind(optimizedF_, Kinded::Kind::ConcatNodeKind), 1);

  SaveNode *optSave = getSaveByName(optimizedF_, "ret_save");
  ASSERT_TRUE(optSave);

  // Data parallel split concats results on 0th dim.
  SigmoidNode *SN = llvm::dyn_cast<SigmoidNode>(optSave->getInput());
  ASSERT_TRUE(SN);

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

/// Test FRWQ-SLS is not lowered when we rely on FP16 conversion in the
/// optimization pipeline.
TEST_F(NNPIOptPipelineTestLowering, NoLowerSLSFP16) {
  Tensor data(ElemKind::FloatTy, {3, 2});
  data.getHandle() = {1.0f, 1.2f, 2.3f, 3.4f, 4.5f, 5.7f};

  Placeholder *indices =
      mod_.createPlaceholder(ElemKind::Int64ITy, {8}, "indices",
                             /* isTrainable */ false);
  Placeholder *lengths = mod_.createPlaceholder(
      ElemKind::Int32ITy, {5}, "lengths", /* isTrainable */ false);

  auto *R = F_->createFusedRowwiseQuantizedSparseLengthsSum(
      "FRQSLWS", data, indices, lengths, ElemKind::UInt8FusedQTy);
  auto *save = F_->createSave("save", R);

  cctx_.precisionConfig.convertToFP16 = true;
  cctx_.precisionConfig.convertFusedToFP16 = true;

  EE_.compile(cctx_);

  // Expect one FP16 SLS after optimization (converted back to Float for Save).
  auto *CN = llvm::dyn_cast<ConvertToNode>(save->getInput());
  ASSERT_TRUE(CN);
  auto *SLS =
      llvm::dyn_cast<FusedRowwiseQuantizedSparseLengthsSumNode>(CN->getInput());
  ASSERT_TRUE(SLS);
  EXPECT_EQ(SLS->getResult().getElementType(), ElemKind::Float16Ty);
  EXPECT_EQ(SLS->getData().getElementType(), ElemKind::UInt8FusedFP16QTy);
}

/// Test Logit is not lowered when we rely on FP16 conversion in the
/// optimization pipeline.
TEST_F(NNPIOptPipelineTestLowering, NoLowerLogit) {
  auto *input = mod_.createPlaceholder(ElemKind::FloatTy, {10}, "input", false);
  auto *tanh = F_->createLogit("logit", input, 1E-6f);
  auto *save = F_->createSave("Save", tanh);

  cctx_.precisionConfig.convertToFP16 = true;
  cctx_.precisionConfig.convertFusedToFP16 = true;

  EE_.compile(cctx_);

  // Expect one FP16 SLS after optimization (converted back to Float for Save).
  auto *CN = llvm::dyn_cast<ConvertToNode>(save->getInput());
  ASSERT_TRUE(CN);
  auto *LN = llvm::dyn_cast<LogitNode>(CN->getInput());
  ASSERT_TRUE(LN);
  EXPECT_EQ(LN->getResult().getElementType(), ElemKind::Float16Ty);
  EXPECT_EQ(LN->getInput().getElementType(), ElemKind::Float16Ty);
}

// Tile->LayerNorm
TEST_F(NNPIOptPipelineTest, DataParallelLNClip) {
  auto *input =
      mod_.createPlaceholder(ElemKind::Float16Ty, {1, 1024}, "input", false);
  bindings_.allocate(input)->getHandle<float16_t>().randomize(-1.0, 1.0,
                                                              mod_.getPRNG());
  auto *tiled = F_->createTile("tile", input, 32, 0);
  Tensor scaleT(ElemKind::Float16Ty, {1024});
  scaleT.getHandle<float16_t>().randomize(0.0f, 1.0f, mod_.getPRNG());
  Constant *scaleC = mod_.createConstant("scale", std::move(scaleT));
  SplatNode *biasS = F_->createSplat("bias", scaleC->getType(), 1.5f);
  auto *ln =
      F_->createLayerNormalization("layernorm", tiled, scaleC, biasS, 1e-4);
  auto *clipped = F_->createClip("clip", ln, -128.0f, 128.0f);
  F_->createSave("ret", clipped);

  // Set 8, but Add won't be parallelized, and so Relu shouldn't be either.
  cctx_.backendOpts.backendSpecificOpts["NNPINumParallelChunks"] =
      std::to_string(8);
  cloneAndCompile();

  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::TileNodeKind), 1);
  EXPECT_EQ(countNodeKind(optimizedF_, Kinded::Kind::TileNodeKind), 8);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::LayerNormalizationNodeKind), 1);
  EXPECT_EQ(
      countNodeKind(optimizedF_, Kinded::Kind::LayerNormalizationNodeKind), 8);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::ClipNodeKind), 1);
  EXPECT_EQ(countNodeKind(optimizedF_, Kinded::Kind::ClipNodeKind), 8);
  checkNumericalEquivalence(/* allowedError */ 0.00f);
}

// BatchedReduceAdd Data Parallel, reducing dim 0
TEST_F(NNPIOptPipelineTest, ParallelBatchedReduceAddReduceDim0) {
  auto *input =
      mod_.createPlaceholder(ElemKind::Float16Ty, {20, 64, 6}, "input", false);
  bindings_.allocate(input)->getHandle<float16_t>().randomize(-1.0, 1.0,
                                                              mod_.getPRNG());

  auto *reduced = F_->createBatchedReduceAdd("BR", input, {0});
  auto *clipped = F_->createClip("clip", reduced, -10.0f, 10.0f);
  F_->createSave("ret", clipped);

  // Should split BatchedReduceAdd by 8
  cctx_.backendOpts.backendSpecificOpts["NNPINumParallelChunks"] =
      std::to_string(8);
  cloneAndCompile();

  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::BatchedReduceAddNodeKind), 1);
  EXPECT_EQ(countNodeKind(optimizedF_, Kinded::Kind::BatchedReduceAddNodeKind),
            8);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::ClipNodeKind), 1);
  EXPECT_EQ(countNodeKind(optimizedF_, Kinded::Kind::ClipNodeKind), 8);
  checkNumericalEquivalence(/* allowedError */ 0.00f);
}

// BatchedReduceAdd Data Parallel reducing dim 1
TEST_F(NNPIOptPipelineTest, ParallelBatchedReduceAddReduceDim1) {
  auto *input =
      mod_.createPlaceholder(ElemKind::Float16Ty, {64, 20, 6}, "input", false);
  bindings_.allocate(input)->getHandle<float16_t>().randomize(-1.0, 1.0,
                                                              mod_.getPRNG());

  auto *reduced = F_->createBatchedReduceAdd("BR", input, {1});
  auto *clipped = F_->createClip("clip", reduced, -10.0f, 10.0f);
  F_->createSave("ret", clipped);

  // Should split BatchedReduceAdd by 8
  cctx_.backendOpts.backendSpecificOpts["NNPINumParallelChunks"] =
      std::to_string(8);
  cloneAndCompile();

  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::BatchedReduceAddNodeKind), 1);
  EXPECT_EQ(countNodeKind(optimizedF_, Kinded::Kind::BatchedReduceAddNodeKind),
            8);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::ClipNodeKind), 1);
  EXPECT_EQ(countNodeKind(optimizedF_, Kinded::Kind::ClipNodeKind), 8);
  checkNumericalEquivalence(/* allowedError */ 0.00f);
}

// Quantize -> FC -> DQ
TEST_F(NNPIOptPipelineTest, QuantizeFCDequantize) {
  auto *input =
      mod_.createPlaceholder(ElemKind::Float16Ty, {32, 1024}, "input", false);
  auto *weights = F_->getParent()->createConstant(
      ElemKind::Int8QTy, {1024, 1024}, 0.2, 0, "weights");
  auto *bias = F_->getParent()->createConstant(ElemKind::Int32QTy, {1024}, 0.2,
                                               0, "bias");

  auto outTy = mod_.uniqueType(ElemKind::Int8QTy, {32, 1024}, 0.2, 0);
  auto *quantized = F_->createQuantize("quantize", input, outTy);
  auto *FC = F_->createFullyConnected("fc", quantized, weights, bias);
  auto *dequantized =
      F_->createDequantize("dequantize", FC, ElemKind::Float16Ty);
  F_->createSave("ret", dequantized);

  // Set 8, but Add won't be parallelized, and so Relu shouldn't be either.
  cctx_.backendOpts.backendSpecificOpts["NNPINumParallelChunks"] =
      std::to_string(8);
  cloneAndCompile();

  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::QuantizeNodeKind), 1);
  EXPECT_EQ(countNodeKind(optimizedF_, Kinded::Kind::QuantizeNodeKind), 8);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::DequantizeNodeKind), 1);
  EXPECT_EQ(countNodeKind(optimizedF_, Kinded::Kind::DequantizeNodeKind), 8);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::FullyConnectedNodeKind), 1);
  EXPECT_EQ(countNodeKind(optimizedF_, Kinded::Kind::FullyConnectedNodeKind),
            8);
}

// BMM->clip
TEST_F(NNPIOptPipelineTest, BMMClip) {
  auto *input0 =
      mod_.createPlaceholder(ElemKind::Float16Ty, {32, 32, 32}, "input", false);

  auto *input1 =
      mod_.createPlaceholder(ElemKind::Float16Ty, {32, 32, 32}, "input", false);

  auto *BMM = F_->createBatchMatMul("bmm", input0, input1);
  auto *clipped = F_->createClip("clip", BMM, -128.0f, 128.0f);
  F_->createSave("ret", clipped);

  // Set 8, but Add won't be parallelized, and so Relu shouldn't be either.
  cctx_.backendOpts.backendSpecificOpts["NNPINumParallelChunks"] =
      std::to_string(8);
  cloneAndCompile();

  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::ClipNodeKind), 1);
  EXPECT_EQ(countNodeKind(optimizedF_, Kinded::Kind::ClipNodeKind), 8);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::BatchMatMulNodeKind), 1);
  EXPECT_EQ(countNodeKind(optimizedF_, Kinded::Kind::BatchMatMulNodeKind), 8);
}

// Swish
TEST_F(NNPIOptPipelineTest, Swish) {
  auto *input0 =
      mod_.createPlaceholder(ElemKind::Float16Ty, {32, 2048}, "input", false);

  auto *S = F_->createSwish("swish", input0);
  F_->createSave("ret", S);

  cctx_.backendOpts.backendSpecificOpts["NNPINumParallelChunks"] =
      std::to_string(8);
  cloneAndCompile();

  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::SwishNodeKind), 1);
  EXPECT_EQ(countNodeKind(optimizedF_, Kinded::Kind::SwishNodeKind), 8);
}

// Swish with a small batch. When we try to parallelize beyond the size
// of the batch, it should fall back to fully split the batch dim
TEST_F(NNPIOptPipelineTest, SwishSmallBatch) {
  auto *input0 =
      mod_.createPlaceholder(ElemKind::Float16Ty, {4, 2048}, "input", false);

  auto *S = F_->createSwish("swish", input0);
  F_->createSave("ret", S);

  cctx_.backendOpts.backendSpecificOpts["NNPINumParallelChunks"] =
      std::to_string(8);
  cloneAndCompile();

  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::SwishNodeKind), 1);
  EXPECT_EQ(countNodeKind(optimizedF_, Kinded::Kind::SwishNodeKind), 4);
}

// SoftMax
TEST_F(NNPIOptPipelineTest, SoftMax) {
  auto *input0 =
      mod_.createPlaceholder(ElemKind::Float16Ty, {512, 2048}, "input", false);
  auto selected = mod_.createConstant(ElemKind::Int64ITy, {512, 1}, "selected");
  auto *SFMX = F_->createSoftMax("softmax", input0, selected);
  F_->createSave("ret", SFMX);

  cctx_.backendOpts.backendSpecificOpts["NNPINumParallelChunks"] =
      std::to_string(8);
  cloneAndCompile();

  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::SoftMaxNodeKind), 1);
  EXPECT_EQ(countNodeKind(optimizedF_, Kinded::Kind::SoftMaxNodeKind), 8);
}
