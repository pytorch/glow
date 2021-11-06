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
#include "glow/Optimizer/GraphOptimizer/FunctionPassPipeline.h"
#include "glow/Optimizer/GraphOptimizer/GraphOptimizer.h"
#include "glow/Optimizer/GraphOptimizer/NodeSplitting.h"

#include "gtest/gtest.h"

using namespace glow;

class NodeSplitting : public GraphOptz {};

bool operator==(const std::vector<dim_t> &lhs, const std::vector<dim_t> &rhs) {
  return std::equal(lhs.begin(), lhs.end(), rhs.begin());
}

/// Test for checking the LLVM style RTTI used for split options.
TEST(TestSplitNodeOption, CheckLLVMStyleRTTI) {
  // Check orthogonal options.
  auto opt1 = SplitNodeByNumChunks({0}, {1});
  auto opt2 = SplitNodeByChunkSize({0}, {1});
  auto opt3 = SplitNodeByChunkSizes({0}, {{1}});
  auto opt4 = SplitNodeByChunkWeights({0}, {{1}});
  EXPECT_EQ(opt1.getKind(),
            SplitNodeOption::SplitNodeKind::SplitNodeByNumChunks);
  EXPECT_EQ(opt2.getKind(),
            SplitNodeOption::SplitNodeKind::SplitNodeByChunkSize);
  EXPECT_EQ(opt3.getKind(),
            SplitNodeOption::SplitNodeKind::SplitNodeByChunkSizes);
  EXPECT_EQ(opt4.getKind(),
            SplitNodeOption::SplitNodeKind::SplitNodeByChunkWeights);
  std::vector<SplitNodeOption *> orthogonalOpts = {&opt1, &opt2, &opt3, &opt4};
  for (auto opt : orthogonalOpts) {
    EXPECT_NE(nullptr, dyn_cast<SplitNodeOptionOrthogonal>(opt));
    EXPECT_EQ(nullptr, dyn_cast<SplitNodeBySliceRanges>(opt));
  }
  // Check non-orthogonal options.
  std::vector<SliceRange> sliceRanges = {SliceRange({{0, 1}})};
  auto opt5 = SplitNodeBySliceRanges(sliceRanges);
  EXPECT_EQ(opt5.getKind(),
            SplitNodeOption::SplitNodeKind::SplitNodeBySliceRanges);
  SplitNodeOption *nonOrthogonalOpt = &opt5;
  EXPECT_EQ(nullptr, dyn_cast<SplitNodeOptionOrthogonal>(nonOrthogonalOpt));
  EXPECT_NE(nullptr, dyn_cast<SplitNodeBySliceRanges>(nonOrthogonalOpt));
}

/// Test for SplitNodeByNumChunks option.
TEST(TestSplitNodeOption, SplitNodeByNumChunksOptionTest) {
  auto opt1 = SplitNodeByNumChunks({0, 1, 2, 3}, {1, 2, 3, 4},
                                   /* bigChunksFirst */ false);
  EXPECT_EQ(opt1.splitAlongDim(0, 10), std::vector<dim_t>({10}));
  EXPECT_EQ(opt1.splitAlongDim(1, 10), std::vector<dim_t>({5, 5}));
  EXPECT_EQ(opt1.splitAlongDim(2, 10), std::vector<dim_t>({3, 3, 4}));
  EXPECT_EQ(opt1.splitAlongDim(3, 10), std::vector<dim_t>({2, 2, 3, 3}));
  EXPECT_EQ(opt1.splitAlongDim(3, 12), std::vector<dim_t>({3, 3, 3, 3}));

  auto opt2 = SplitNodeByNumChunks({0, 1, 2, 3}, {1, 2, 3, 4},
                                   /* bigChunksFirst */ true);
  EXPECT_EQ(opt2.splitAlongDim(0, 10), std::vector<dim_t>({10}));
  EXPECT_EQ(opt2.splitAlongDim(1, 10), std::vector<dim_t>({5, 5}));
  EXPECT_EQ(opt2.splitAlongDim(2, 10), std::vector<dim_t>({4, 3, 3}));
  EXPECT_EQ(opt2.splitAlongDim(3, 10), std::vector<dim_t>({3, 3, 2, 2}));
  EXPECT_EQ(opt2.splitAlongDim(3, 12), std::vector<dim_t>({3, 3, 3, 3}));
}

/// Test for SplitNodeByChunkSize option.
TEST(TestSplitNodeOption, SplitNodeByChunkSizeOptionTest) {
  auto opt1 = SplitNodeByChunkSize({0, 1, 2, 3}, {3, 4, 5, 6},
                                   /* bigChunksFirst */ false);
  EXPECT_EQ(opt1.splitAlongDim(0, 10), std::vector<dim_t>({1, 3, 3, 3}));
  EXPECT_EQ(opt1.splitAlongDim(1, 10), std::vector<dim_t>({2, 4, 4}));
  EXPECT_EQ(opt1.splitAlongDim(2, 10), std::vector<dim_t>({5, 5}));
  EXPECT_EQ(opt1.splitAlongDim(3, 10), std::vector<dim_t>({4, 6}));
  EXPECT_EQ(opt1.splitAlongDim(3, 18), std::vector<dim_t>({6, 6, 6}));

  auto opt2 = SplitNodeByChunkSize({0, 1, 2, 3}, {3, 4, 5, 6},
                                   /* bigChunksFirst */ true);
  EXPECT_EQ(opt2.splitAlongDim(0, 10), std::vector<dim_t>({3, 3, 3, 1}));
  EXPECT_EQ(opt2.splitAlongDim(1, 10), std::vector<dim_t>({4, 4, 2}));
  EXPECT_EQ(opt2.splitAlongDim(2, 10), std::vector<dim_t>({5, 5}));
  EXPECT_EQ(opt2.splitAlongDim(3, 10), std::vector<dim_t>({6, 4}));
  EXPECT_EQ(opt2.splitAlongDim(3, 18), std::vector<dim_t>({6, 6, 6}));
}

/// Test for SplitNodeByChunkSizes option.
TEST(TestSplitNodeOption, SplitNodeByChunkSizesOptionTest) {
  auto opt = SplitNodeByChunkSizes({0, 1, 2, 3},
                                   {{1, 3, 3, 3}, {2, 4, 4}, {5, 5}, {4, 6}});
  EXPECT_EQ(opt.splitAlongDim(0, 10), std::vector<dim_t>({1, 3, 3, 3}));
  EXPECT_EQ(opt.splitAlongDim(1, 10), std::vector<dim_t>({2, 4, 4}));
  EXPECT_EQ(opt.splitAlongDim(2, 10), std::vector<dim_t>({5, 5}));
  EXPECT_EQ(opt.splitAlongDim(3, 10), std::vector<dim_t>({4, 6}));
}

/// Test for SplitNodeByChunkWeights option.
TEST(TestSplitNodeOption, SplitNodeByChunkWeightsOptionTest) {
  auto opt1 = SplitNodeByChunkWeights(
      {0, 1, 2, 3}, {{1, 3, 3, 3}, {2, 4, 4}, {5, 5}, {4, 6}});
  EXPECT_EQ(opt1.splitAlongDim(0, 20), std::vector<dim_t>({2, 6, 6, 6}));
  EXPECT_EQ(opt1.splitAlongDim(1, 20), std::vector<dim_t>({4, 8, 8}));
  EXPECT_EQ(opt1.splitAlongDim(2, 20), std::vector<dim_t>({10, 10}));
  EXPECT_EQ(opt1.splitAlongDim(3, 20), std::vector<dim_t>({8, 12}));

  auto opt2 = SplitNodeByChunkWeights({0}, {{0.15, 0.15, 0.2, 0.5}});
  EXPECT_EQ(opt2.splitAlongDim(0, 100), std::vector<dim_t>({15, 15, 20, 50}));

  auto opt3 = SplitNodeByChunkWeights({0}, {{0.00000001, 33, 66}});
  EXPECT_EQ(opt3.splitAlongDim(0, 100), std::vector<dim_t>({1, 33, 66}));
}

///===---------------------------------------------------------------------===//
///                                   Conv2D
///===---------------------------------------------------------------------===//
/// Utility function to create a simple network with a single Conv2D node using
/// the function \p F and the bindings \p bindings.
static Node *createConv2D(
    Function *F, PlaceholderBindings &bindings, llvm::ArrayRef<dim_t> inputDims,
    llvm::ArrayRef<dim_t> filterDims, llvm::ArrayRef<dim_t> biasDims,
    llvm::ArrayRef<dim_t> outputDims, llvm::ArrayRef<unsigned_t> kernels,
    llvm::ArrayRef<unsigned_t> strides, llvm::ArrayRef<unsigned_t> pads,
    dim_t group, llvm::ArrayRef<unsigned_t> dilation) {
  // Create input placeholder.
  auto &mod = *(F->getParent());
  auto *input =
      mod.createPlaceholder(ElemKind::FloatTy, inputDims, "input", false);
  bindings.allocate(input)->getHandle<float>().randomize(-1.0, 1.0,
                                                         mod.getPRNG());
  // Create filter constant.
  auto *filter = mod.createConstant(ElemKind::FloatTy, filterDims, "filter");
  filter->getPayloadMutable().getHandle<float>().randomize(-1.0, 1.0,
                                                           mod.getPRNG());
  // Create bias constant.
  auto *bias = mod.createConstant(ElemKind::FloatTy, biasDims, "bias");
  bias->getPayloadMutable().getHandle<float>().randomize(-1.0, 1.0,
                                                         mod.getPRNG());
  // Create Conv2D.
  auto *outTy = mod.uniqueType(ElemKind::FloatTy, outputDims);
  ConvolutionNode *conv =
      F->createConv("conv", input, filter, bias, outTy, kernels, strides, pads,
                    group, dilation);
  SaveNode *save = F->createSave("save", conv);
  bindings.allocate(save->getPlaceholder());
  return conv;
}

/// Utility function to test splitting a basic Conv2D node along the dimensions
/// \p splitDims in the given number chunks \p numChunks. The split is done
/// implicitly relative to the Conv2D output operand.
static void splitConv2DBasic(Function *F, Function *&optF,
                             PlaceholderBindings &bindings,
                             CompilationContext &cctx,
                             llvm::ArrayRef<size_t> splitDims,
                             llvm::ArrayRef<dim_t> numChunks) {
  Node *node = createConv2D(F, bindings,
                            /* inputDims */ {5, 7, 8, 2},
                            /* filterDims */ {8, 2, 2, 1},
                            /* biasDims */ {8},
                            /* outputDims */ {5, 6, 7, 8},
                            /* kernels */ {2, 2},
                            /* strides */ {1, 1},
                            /* pads */ {0, 0, 0, 0},
                            /* group */ 2,
                            /* dilation */ {1, 1});

  // Save current function state as reference.
  optF = F->clone(F->getName().str() + "_optimized");

  // Split node.
  auto splitOption = SplitNodeByNumChunks(splitDims, numChunks);
  std::vector<Node *> splitNodes;
  ASSIGN_VALUE_OR_FAIL_TEST(splitNodes, ::glow::splitNode(node, splitOption));

  // Compute total number of chunks.
  dim_t totNumChunks = 1;
  for (auto numChunk : numChunks) {
    totNumChunks *= numChunk;
  }

  // Check node count.
  EXPECT_EQ(splitNodes.size(), totNumChunks);
  EXPECT_EQ(countNodeKind(F, Kinded::Kind::ConvolutionNodeKind), totNumChunks);
  EXPECT_EQ(countNodeKind(F, Kinded::Kind::InsertTensorNodeKind), totNumChunks);
  EXPECT_EQ(countNodeKind(F, Kinded::Kind::TouchNodeKind), 1);
}

/// Test splitting a Conv2D along dimension N, H, W or C.
/// Not all the combinations are allowed when splitting along C.
#define TEST_CONV2D_BASIC_SPLIT(splitDim, numChunks)                           \
  TEST_F(NodeSplitting, Conv2D_Basic_Dim##splitDim##_Chunks##numChunks) {      \
    splitConv2DBasic(F_, optimizedF_, bindings_, cctx_,                        \
                     {ShapeNHWC::Dim##splitDim}, {numChunks});                 \
    checkNumericalEquivalence(0);                                              \
  }
TEST_CONV2D_BASIC_SPLIT(N, 2)
TEST_CONV2D_BASIC_SPLIT(N, 3)
TEST_CONV2D_BASIC_SPLIT(N, 4)
TEST_CONV2D_BASIC_SPLIT(N, 5)
TEST_CONV2D_BASIC_SPLIT(H, 2)
TEST_CONV2D_BASIC_SPLIT(H, 3)
TEST_CONV2D_BASIC_SPLIT(H, 4)
TEST_CONV2D_BASIC_SPLIT(H, 5)
TEST_CONV2D_BASIC_SPLIT(H, 6)
TEST_CONV2D_BASIC_SPLIT(W, 2)
TEST_CONV2D_BASIC_SPLIT(W, 3)
TEST_CONV2D_BASIC_SPLIT(W, 4)
TEST_CONV2D_BASIC_SPLIT(W, 5)
TEST_CONV2D_BASIC_SPLIT(W, 6)
TEST_CONV2D_BASIC_SPLIT(W, 7)
TEST_CONV2D_BASIC_SPLIT(C, 2)
TEST_CONV2D_BASIC_SPLIT(C, 4)
TEST_CONV2D_BASIC_SPLIT(C, 8)
#undef TEST_CONV2D_BASIC_SPLIT

/// Test splitting a Conv2D along dimensions N, H.
TEST_F(NodeSplitting, Conv2D_Basic_DimNH_Chunks4) {
  splitConv2DBasic(F_, optimizedF_, bindings_, cctx_,
                   {ShapeNHWC::DimN, ShapeNHWC::DimH}, {2, 2});
  checkNumericalEquivalence(0);
}

/// Test splitting a Conv2D along dimensions N, H, W.
TEST_F(NodeSplitting, Conv2D_Basic_DimNHW_Chunks8) {
  splitConv2DBasic(F_, optimizedF_, bindings_, cctx_,
                   {ShapeNHWC::DimN, ShapeNHWC::DimH, ShapeNHWC::DimW},
                   {2, 2, 2});
  checkNumericalEquivalence(0);
}

/// Test splitting a Conv2D along dimensions N, H, W, C.
TEST_F(NodeSplitting, Conv2D_Basic_DimNHWC_Chunks16) {
  splitConv2DBasic(
      F_, optimizedF_, bindings_, cctx_,
      {ShapeNHWC::DimN, ShapeNHWC::DimH, ShapeNHWC::DimW, ShapeNHWC::DimC},
      {2, 2, 2, 2});
  checkNumericalEquivalence(0);
}

/// Utility function to test splitting a Conv2D node with non-zero padding
/// along the dimensions \p splitDims in the given number chunks \p numChunks.
/// The split is done implicitly relative to the Conv2D output operand.
static void splitConv2DNonZeroPad(Function *F, Function *&optF,
                                  PlaceholderBindings &bindings,
                                  CompilationContext &cctx,
                                  llvm::ArrayRef<size_t> splitDims,
                                  llvm::ArrayRef<dim_t> numChunks) {
  Node *node = createConv2D(F, bindings,
                            /* inputDims */ {1, 8, 9, 1},
                            /* filterDims */ {1, 2, 3, 1},
                            /* biasDims */ {1},
                            /* outputDims */ {1, 11, 10, 1},
                            /* kernels */ {2, 3},
                            /* strides */ {1, 1},
                            /* pads */ {2, 1, 3, 4},
                            /* group */ 1,
                            /* dilation */ {2, 2});

  // Save current function state as reference.
  optF = F->clone(F->getName().str() + "_optimized");

  // Split node.
  auto splitOption = SplitNodeByNumChunks(splitDims, numChunks);
  std::vector<Node *> splitNodes;
  ASSIGN_VALUE_OR_FAIL_TEST(splitNodes, ::glow::splitNode(node, splitOption));

  // Compute total number of chunks.
  dim_t totNumChunks = 1;
  for (auto numChunk : numChunks) {
    totNumChunks *= numChunk;
  }

  // Check node count.
  EXPECT_EQ(splitNodes.size(), totNumChunks);
  EXPECT_EQ(countNodeKind(F, Kinded::Kind::ConvolutionNodeKind), totNumChunks);
  EXPECT_EQ(countNodeKind(F, Kinded::Kind::InsertTensorNodeKind), totNumChunks);
  EXPECT_EQ(countNodeKind(F, Kinded::Kind::TouchNodeKind), 1);
}

/// Test splitting a Conv2D with padding along dimension N, H, W or C.
#define TEST_CONV2D_NONZEROPAD_SPLIT(splitDim, numChunks)                      \
  TEST_F(NodeSplitting, Conv2D_NonZeroPad_Dim##splitDim##_Chunks##numChunks) { \
    splitConv2DNonZeroPad(F_, optimizedF_, bindings_, cctx_,                   \
                          {ShapeNHWC::Dim##splitDim}, {numChunks});            \
    checkNumericalEquivalence(0);                                              \
  }
TEST_CONV2D_NONZEROPAD_SPLIT(H, 2)
TEST_CONV2D_NONZEROPAD_SPLIT(H, 3)
TEST_CONV2D_NONZEROPAD_SPLIT(W, 2)
TEST_CONV2D_NONZEROPAD_SPLIT(W, 3)
#undef TEST_CONV2D_NONZEROPAD_SPLIT

/// Test splitting a Conv2D with padding along dimensions H, W.
TEST_F(NodeSplitting, Conv2D_NonZeroPad_DimHW_Chunks9) {
  splitConv2DNonZeroPad(F_, optimizedF_, bindings_, cctx_,
                        {ShapeNHWC::DimH, ShapeNHWC::DimW}, {3, 3});
  checkNumericalEquivalence(0);
}

/// Utility function to test splitting a group Conv2D node along dimension C in
/// \p numChunks having the given number of \p inputChannels, \p outputChannels
/// and the given \p group. The split is done implicitly relative to the Conv2D
/// output operand.
static void splitConv2DGrouped(Function *F, Function *&optF,
                               PlaceholderBindings &bindings,
                               CompilationContext &cctx, dim_t inputChannels,
                               dim_t outputChannels, dim_t group,
                               dim_t numChunks) {
  dim_t filterChannels = inputChannels / group;
  dim_t filterNum = outputChannels;
  Node *node = createConv2D(F, bindings,
                            /* inputDims */ {1, 2, 2, inputChannels},
                            /* filterDims */ {filterNum, 2, 2, filterChannels},
                            /* biasDims */ {outputChannels},
                            /* outputDims */ {1, 1, 1, outputChannels},
                            /* kernels */ {2, 2},
                            /* strides */ {1, 1},
                            /* pads */ {0, 0, 0, 0},
                            /* group */ group,
                            /* dilation */ {1, 1});

  // Save current function state as reference.
  optF = F->clone(F->getName().str() + "_optimized");

  // Split node.
  auto splitOption = SplitNodeByNumChunks({ShapeNHWC::DimC}, {numChunks});
  std::vector<Node *> splitNodes;
  ASSIGN_VALUE_OR_FAIL_TEST(splitNodes, ::glow::splitNode(node, splitOption));

  // Check node count.
  EXPECT_EQ(splitNodes.size(), numChunks);
  EXPECT_EQ(countNodeKind(F, Kinded::Kind::ConvolutionNodeKind), numChunks);
  EXPECT_EQ(countNodeKind(F, Kinded::Kind::InsertTensorNodeKind), numChunks);
  EXPECT_EQ(countNodeKind(F, Kinded::Kind::TouchNodeKind), 1);
}

/// Test splitting a grouped Conv2D along dimension C.
#define TEST_CONV2D_GROUP_SPLIT(IC, OC, G, chunks)                             \
  TEST_F(NodeSplitting,                                                        \
         Conv2D_Group_DimC_InpC##IC##_OutC##OC##_Group##G##_Chunks##chunks) {  \
    splitConv2DGrouped(F_, optimizedF_, bindings_, cctx_, IC, OC, G, chunks);  \
    checkNumericalEquivalence(0);                                              \
  }
TEST_CONV2D_GROUP_SPLIT(8, 8, 2, 2)
TEST_CONV2D_GROUP_SPLIT(8, 8, 2, 4)
TEST_CONV2D_GROUP_SPLIT(8, 8, 2, 8)
TEST_CONV2D_GROUP_SPLIT(8, 8, 4, 2)
TEST_CONV2D_GROUP_SPLIT(8, 8, 4, 4)
TEST_CONV2D_GROUP_SPLIT(8, 8, 4, 8)
TEST_CONV2D_GROUP_SPLIT(8, 8, 8, 2)
TEST_CONV2D_GROUP_SPLIT(8, 8, 8, 4)
TEST_CONV2D_GROUP_SPLIT(8, 8, 8, 8)
TEST_CONV2D_GROUP_SPLIT(8, 16, 2, 2)
TEST_CONV2D_GROUP_SPLIT(8, 16, 2, 4)
TEST_CONV2D_GROUP_SPLIT(8, 16, 2, 8)
TEST_CONV2D_GROUP_SPLIT(8, 16, 4, 2)
TEST_CONV2D_GROUP_SPLIT(8, 16, 4, 4)
TEST_CONV2D_GROUP_SPLIT(8, 16, 4, 8)
TEST_CONV2D_GROUP_SPLIT(8, 16, 8, 2)
TEST_CONV2D_GROUP_SPLIT(8, 16, 8, 4)
TEST_CONV2D_GROUP_SPLIT(8, 16, 8, 8)
#undef TEST_CONV2D_GROUP_SPLIT

/// Test splitting an "ill-defined" Conv2D for which not all the input
/// (including padding) is referenced by the output tensor. This happens
/// when using a stride larger than 1. This verifies that the node
/// splitting infrastructure uses a weaker verification of the mapping
/// between input and output for Conv2D.
TEST_F(NodeSplitting, Conv2D_IllDefined_DimHW) {
  std::vector<size_t> splitDims = {ShapeNHWC::DimH, ShapeNHWC::DimW};
  std::vector<dim_t> numChunks = {3, 3};
  Node *node = createConv2D(F_, bindings_,
                            /* inputDims */ {1, 16, 18, 1},
                            /* filterDims */ {1, 2, 2, 1},
                            /* biasDims */ {1},
                            /* outputDims */ {1, 8, 9, 1},
                            /* kernels */ {2, 2},
                            /* strides */ {2, 2},
                            /* pads */ {1, 1, 0, 0},
                            /* group */ 1,
                            /* dilation */ {1, 1});

  // Save current function state as reference.
  optimizedF_ = F_->clone(F_->getName().str() + "_optimized");

  // Split node.
  auto splitOption = SplitNodeByNumChunks(splitDims, numChunks);
  std::vector<Node *> splitNodes;
  ASSIGN_VALUE_OR_FAIL_TEST(splitNodes, ::glow::splitNode(node, splitOption));

  // Check node count.
  dim_t totNumChunks = numChunks[0] * numChunks[1];
  EXPECT_EQ(splitNodes.size(), totNumChunks);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::ConvolutionNodeKind), totNumChunks);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::InsertTensorNodeKind),
            totNumChunks);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::TouchNodeKind), 1);
  checkNumericalEquivalence(0);
}

/// Test splitting a Conv2D based on memory constraint.
TEST_F(NodeSplitting, Conv2D_MaxMem) {
  Node *node = createConv2D(F_, bindings_,
                            /* inputDims */ {5, 7, 8, 2},
                            /* filterDims */ {8, 2, 2, 1},
                            /* biasDims */ {8},
                            /* outputDims */ {5, 6, 7, 8},
                            /* kernels */ {2, 2},
                            /* strides */ {1, 1},
                            /* pads */ {0, 0, 0, 0},
                            /* group */ 2,
                            /* dilation */ {1, 1});

  // Save current function state as reference.
  optimizedF_ = F_->clone(F_->getName().str() + "_optimized");

  // Split node by memory size.
  auto origMemSize = node->getTotMemSize();
  auto splitMaxMemSize = origMemSize / 2;
  std::vector<Node *> splitNodes;
  ASSIGN_VALUE_OR_FAIL_TEST(
      splitNodes,
      ::glow::splitNode(node, SplitNodeMaxMemConstraint(splitMaxMemSize)));

  // Check node count.
  auto totNumChunks = countNodeKind(F_, Kinded::Kind::ConvolutionNodeKind);
  EXPECT_TRUE(totNumChunks > 1);
  EXPECT_EQ(splitNodes.size(), totNumChunks);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::InsertTensorNodeKind),
            totNumChunks);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::TouchNodeKind), 1);

  // Check split nodes memory sizes.
  for (auto *splitNode : splitNodes) {
    EXPECT_TRUE(splitNode->getTotMemSize() <= splitMaxMemSize);
  }
  checkNumericalEquivalence(0);
}

/// Test splitting a Conv2D based on an impossible constraint forcing the split
/// procedure to go through all the split configurations while verifying them.
/// In the end no split should be performed.
TEST_F(NodeSplitting, Conv2D_NoSplit) {
  Node *node = createConv2D(F_, bindings_,
                            /* inputDims */ {5, 7, 8, 2},
                            /* filterDims */ {8, 2, 2, 1},
                            /* biasDims */ {8},
                            /* outputDims */ {5, 6, 7, 8},
                            /* kernels */ {2, 2},
                            /* strides */ {1, 1},
                            /* pads */ {0, 0, 0, 0},
                            /* group */ 2,
                            /* dilation */ {1, 1});

  // Save current function state as reference.
  optimizedF_ = F_->clone(F_->getName().str() + "_optimized");

  // Split node by memory size 0.
  std::vector<Node *> splitNodes;
  ASSIGN_VALUE_OR_FAIL_TEST(
      splitNodes, ::glow::splitNode(node, SplitNodeMaxMemConstraint(0)));

  // Check node count.
  EXPECT_EQ(splitNodes.size(), 0);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::SliceNodeKind), 0);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::ConvolutionNodeKind), 1);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::InsertTensorNodeKind), 0);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::TouchNodeKind), 0);
  checkNumericalEquivalence(0);
}

///===---------------------------------------------------------------------===//
///                         ChannelwiseQuantizedConv2D
///===---------------------------------------------------------------------===//
/// Utility function to create a simple network with a CWQConv2D node using
/// the function \p F and the bindings \p bindings.
static Node *createCWQConv2D(
    Function *F, PlaceholderBindings &bindings, llvm::ArrayRef<dim_t> inputDims,
    llvm::ArrayRef<dim_t> filterDims, llvm::ArrayRef<dim_t> biasDims,
    llvm::ArrayRef<dim_t> outputDims, llvm::ArrayRef<unsigned_t> kernels,
    llvm::ArrayRef<unsigned_t> strides, llvm::ArrayRef<unsigned_t> pads,
    dim_t group, llvm::ArrayRef<unsigned_t> dilation) {
  // Create quantized input placeholder.
  auto &mod = *(F->getParent());
  auto *inputQ = mod.createPlaceholder(ElemKind::Int8QTy, inputDims, 1.0, 0,
                                       "inputQ", false);
  bindings.allocate(inputQ)->getHandle<int8_t>().randomize(-128, 127,
                                                           mod.getPRNG());
  // Create float filter constant.
  auto *filterF = mod.createConstant(ElemKind::FloatTy, filterDims, "filterF");
  filterF->getPayloadMutable().getHandle<float>().randomize(-1.0, 1.0,
                                                            mod.getPRNG());
  // Create float bias constant.
  auto *biasF = mod.createConstant(ElemKind::FloatTy, biasDims, "biasF");
  biasF->getPayloadMutable().getHandle<float>().randomize(-1.0, 1.0,
                                                          mod.getPRNG());
  // Create ChannelwiseQuantizedConv2D.
  auto *outTy = mod.uniqueType(ElemKind::Int8QTy, outputDims, 1.0, 0);
  auto *conv = F->createChannelwiseQuantizedConv(
      "cwqconv", inputQ, filterF, biasF,
      /* filterScales */ nullptr, /* filterOffsets */ nullptr,
      /* biasScales */ nullptr, /* biasOffsets */ nullptr, outTy, kernels,
      strides, pads, group, dilation,
      /* quantizeFilter */ true, /* quantizeBias */ true);
  SaveNode *save = F->createSave("save", conv);
  bindings.allocate(save->getPlaceholder());
  return conv;
}

/// Utility function to test splitting a basic CWQConv2D node along the
/// dimensions \p splitDims in the given number chunks \p numChunks. The split
/// is done implicitly relative to the Conv2D output operand.
static void splitCWQConv2DBasic(Function *F, Function *&optF,
                                PlaceholderBindings &bindings,
                                CompilationContext &cctx,
                                llvm::ArrayRef<size_t> splitDims,
                                llvm::ArrayRef<dim_t> numChunks) {
  Node *node = createCWQConv2D(F, bindings,
                               /* inputDims */ {5, 7, 8, 2},
                               /* filterDims */ {8, 2, 2, 1},
                               /* biasDims */ {8},
                               /* outputDims */ {5, 6, 7, 8},
                               /* kernels */ {2, 2},
                               /* strides */ {1, 1},
                               /* pads */ {0, 0, 0, 0},
                               /* group */ 2,
                               /* dilation */ {1, 1});

  // Save current function state as reference.
  optF = F->clone(F->getName().str() + "_optimized");

  // Split node.
  auto splitOption = SplitNodeByNumChunks(splitDims, numChunks);
  std::vector<Node *> splitNodes;
  ASSIGN_VALUE_OR_FAIL_TEST(splitNodes, ::glow::splitNode(node, splitOption));

  // Compute total number of chunks.
  dim_t totNumChunks = 1;
  for (auto numChunk : numChunks) {
    totNumChunks *= numChunk;
  }

  // Check node count.
  EXPECT_EQ(splitNodes.size(), totNumChunks);
  EXPECT_EQ(
      countNodeKind(F, Kinded::Kind::ChannelwiseQuantizedConvolutionNodeKind),
      totNumChunks);
  EXPECT_EQ(countNodeKind(F, Kinded::Kind::InsertTensorNodeKind), totNumChunks);
  EXPECT_EQ(countNodeKind(F, Kinded::Kind::TouchNodeKind), 1);
}

/// Test splitting a CWQConv2D along dimension N, H, W or C.
/// Not all the combinations are allowed when splitting along C.
#define TEST_CWQCONV2D_BASIC_SPLIT(splitDim, numChunks)                        \
  TEST_F(NodeSplitting, CWQConv2D_Basic_Dim##splitDim##_Chunks##numChunks) {   \
    splitCWQConv2DBasic(F_, optimizedF_, bindings_, cctx_,                     \
                        {ShapeNHWC::Dim##splitDim}, {numChunks});              \
    checkNumericalEquivalence(0);                                              \
  }
TEST_CWQCONV2D_BASIC_SPLIT(N, 2)
TEST_CWQCONV2D_BASIC_SPLIT(N, 3)
TEST_CWQCONV2D_BASIC_SPLIT(N, 4)
TEST_CWQCONV2D_BASIC_SPLIT(N, 5)
TEST_CWQCONV2D_BASIC_SPLIT(H, 2)
TEST_CWQCONV2D_BASIC_SPLIT(H, 3)
TEST_CWQCONV2D_BASIC_SPLIT(H, 4)
TEST_CWQCONV2D_BASIC_SPLIT(H, 5)
TEST_CWQCONV2D_BASIC_SPLIT(H, 6)
TEST_CWQCONV2D_BASIC_SPLIT(W, 2)
TEST_CWQCONV2D_BASIC_SPLIT(W, 3)
TEST_CWQCONV2D_BASIC_SPLIT(W, 4)
TEST_CWQCONV2D_BASIC_SPLIT(W, 5)
TEST_CWQCONV2D_BASIC_SPLIT(W, 6)
TEST_CWQCONV2D_BASIC_SPLIT(W, 7)
TEST_CWQCONV2D_BASIC_SPLIT(C, 2)
TEST_CWQCONV2D_BASIC_SPLIT(C, 4)
TEST_CWQCONV2D_BASIC_SPLIT(C, 8)
#undef TEST_CWQCONV2D_BASIC_SPLIT

/// Test splitting a CWQConv2D along dimensions N, H.
TEST_F(NodeSplitting, CWQConv2D_Basic_DimNH_Chunks4) {
  splitCWQConv2DBasic(F_, optimizedF_, bindings_, cctx_,
                      {ShapeNHWC::DimN, ShapeNHWC::DimH}, {2, 2});
  checkNumericalEquivalence(0);
}

/// Test splitting a CWQConv2D along dimensions N, H, W.
TEST_F(NodeSplitting, CWQConv2D_Basic_DimNHW_Chunks8) {
  splitCWQConv2DBasic(F_, optimizedF_, bindings_, cctx_,
                      {ShapeNHWC::DimN, ShapeNHWC::DimH, ShapeNHWC::DimW},
                      {2, 2, 2});
  checkNumericalEquivalence(0);
}

/// Test splitting a CWQConv2D along dimensions N, H, W, C.
TEST_F(NodeSplitting, CWQConv2D_Basic_DimNHWC_Chunks16) {
  splitCWQConv2DBasic(
      F_, optimizedF_, bindings_, cctx_,
      {ShapeNHWC::DimN, ShapeNHWC::DimH, ShapeNHWC::DimW, ShapeNHWC::DimC},
      {2, 2, 2, 2});
  checkNumericalEquivalence(0);
}

/// Utility function to test splitting a CWQConv2D node with non-zero padding
/// along the dimensions \p splitDims in the given number chunks \p numChunks.
/// The split is done implicitly relative to the Conv2D output operand.
static void splitCWQConv2DNonZeroPad(Function *F, Function *&optF,
                                     PlaceholderBindings &bindings,
                                     CompilationContext &cctx,
                                     llvm::ArrayRef<size_t> splitDims,
                                     llvm::ArrayRef<dim_t> numChunks) {
  Node *node = createCWQConv2D(F, bindings,
                               /* inputDims */ {1, 8, 9, 1},
                               /* filterDims */ {1, 2, 3, 1},
                               /* biasDims */ {1},
                               /* outputDims */ {1, 11, 10, 1},
                               /* kernels */ {2, 3},
                               /* strides */ {1, 1},
                               /* pads */ {2, 1, 3, 4},
                               /* group */ 1,
                               /* dilation */ {2, 2});

  // Save current function state as reference.
  optF = F->clone(F->getName().str() + "_optimized");

  // Split node.
  auto splitOption = SplitNodeByNumChunks(splitDims, numChunks);
  std::vector<Node *> splitNodes;
  ASSIGN_VALUE_OR_FAIL_TEST(splitNodes, ::glow::splitNode(node, splitOption));

  // Compute total number of chunks.
  dim_t totNumChunks = 1;
  for (auto numChunk : numChunks) {
    totNumChunks *= numChunk;
  }

  // Check node count.
  EXPECT_EQ(splitNodes.size(), totNumChunks);
  EXPECT_EQ(
      countNodeKind(F, Kinded::Kind::ChannelwiseQuantizedConvolutionNodeKind),
      totNumChunks);
  EXPECT_EQ(countNodeKind(F, Kinded::Kind::InsertTensorNodeKind), totNumChunks);
  EXPECT_EQ(countNodeKind(F, Kinded::Kind::TouchNodeKind), 1);
}

/// Test splitting a CWQConv2D with padding along dimension N, H, W or C.
#define TEST_CWQCONV2D_NONZEROPAD_SPLIT(splitDim, numChunks)                   \
  TEST_F(NodeSplitting,                                                        \
         CWQConv2D_NonZeroPad_Dim##splitDim##_Chunks##numChunks) {             \
    splitCWQConv2DNonZeroPad(F_, optimizedF_, bindings_, cctx_,                \
                             {ShapeNHWC::Dim##splitDim}, {numChunks});         \
    checkNumericalEquivalence(0);                                              \
  }
TEST_CWQCONV2D_NONZEROPAD_SPLIT(H, 2)
TEST_CWQCONV2D_NONZEROPAD_SPLIT(H, 3)
TEST_CWQCONV2D_NONZEROPAD_SPLIT(W, 2)
TEST_CWQCONV2D_NONZEROPAD_SPLIT(W, 3)
#undef TEST_CWQCONV2D_NONZEROPAD_SPLIT

/// Test splitting a CWQConv2D with padding along dimensions H, W.
TEST_F(NodeSplitting, CWQConv2D_NonZeroPad_DimHW_Chunks9) {
  splitCWQConv2DNonZeroPad(F_, optimizedF_, bindings_, cctx_,
                           {ShapeNHWC::DimH, ShapeNHWC::DimW}, {3, 3});
  checkNumericalEquivalence(0);
}

/// Utility function to test splitting a group CWQConv2D node along dimension C
/// in \p numChunks having the given number of \p inputChannels,
/// \p outputChannels and the given \p group. The split is done implicitly
/// relative to the Conv2D output operand.
static void splitCWQConv2DGrouped(Function *F, Function *&optF,
                                  PlaceholderBindings &bindings,
                                  CompilationContext &cctx, dim_t inputChannels,
                                  dim_t outputChannels, dim_t group,
                                  dim_t numChunks) {
  dim_t filterChannels = inputChannels / group;
  dim_t filterNum = outputChannels;
  Node *node =
      createCWQConv2D(F, bindings,
                      /* inputDims */ {1, 2, 2, inputChannels},
                      /* filterDims */ {filterNum, 2, 2, filterChannels},
                      /* biasDims */ {outputChannels},
                      /* outputDims */ {1, 1, 1, outputChannels},
                      /* kernels */ {2, 2},
                      /* strides */ {1, 1},
                      /* pads */ {0, 0, 0, 0},
                      /* group */ group,
                      /* dilation */ {1, 1});

  // Save current function state as reference.
  optF = F->clone(F->getName().str() + "_optimized");

  // Split node.
  auto splitOption = SplitNodeByNumChunks({ShapeNHWC::DimC}, {numChunks});
  std::vector<Node *> splitNodes;
  ASSIGN_VALUE_OR_FAIL_TEST(splitNodes, ::glow::splitNode(node, splitOption));

  // Check node count.
  EXPECT_EQ(splitNodes.size(), numChunks);
  EXPECT_EQ(
      countNodeKind(F, Kinded::Kind::ChannelwiseQuantizedConvolutionNodeKind),
      numChunks);
  EXPECT_EQ(countNodeKind(F, Kinded::Kind::InsertTensorNodeKind), numChunks);
  EXPECT_EQ(countNodeKind(F, Kinded::Kind::TouchNodeKind), 1);
}

/// Test splitting a grouped Conv2D along dimension C.
#define TEST_CWQCONV2D_GROUP_SPLIT(IC, OC, G, chunks)                          \
  TEST_F(                                                                      \
      NodeSplitting,                                                           \
      CWQConv2D_Group_DimC_InpC##IC##_OutC##OC##_Group##G##_Chunks##chunks) {  \
    splitCWQConv2DGrouped(F_, optimizedF_, bindings_, cctx_, IC, OC, G,        \
                          chunks);                                             \
    checkNumericalEquivalence(0);                                              \
  }
TEST_CWQCONV2D_GROUP_SPLIT(8, 8, 2, 2)
TEST_CWQCONV2D_GROUP_SPLIT(8, 8, 2, 4)
TEST_CWQCONV2D_GROUP_SPLIT(8, 8, 2, 8)
TEST_CWQCONV2D_GROUP_SPLIT(8, 8, 4, 2)
TEST_CWQCONV2D_GROUP_SPLIT(8, 8, 4, 4)
TEST_CWQCONV2D_GROUP_SPLIT(8, 8, 4, 8)
TEST_CWQCONV2D_GROUP_SPLIT(8, 8, 8, 2)
TEST_CWQCONV2D_GROUP_SPLIT(8, 8, 8, 4)
TEST_CWQCONV2D_GROUP_SPLIT(8, 8, 8, 8)
TEST_CWQCONV2D_GROUP_SPLIT(8, 16, 2, 2)
TEST_CWQCONV2D_GROUP_SPLIT(8, 16, 2, 4)
TEST_CWQCONV2D_GROUP_SPLIT(8, 16, 2, 8)
TEST_CWQCONV2D_GROUP_SPLIT(8, 16, 4, 2)
TEST_CWQCONV2D_GROUP_SPLIT(8, 16, 4, 4)
TEST_CWQCONV2D_GROUP_SPLIT(8, 16, 4, 8)
TEST_CWQCONV2D_GROUP_SPLIT(8, 16, 8, 2)
TEST_CWQCONV2D_GROUP_SPLIT(8, 16, 8, 4)
TEST_CWQCONV2D_GROUP_SPLIT(8, 16, 8, 8)
#undef TEST_CWQCONV2D_GROUP_SPLIT

///===---------------------------------------------------------------------===//
///                                   MaxPool
///===---------------------------------------------------------------------===//
/// Utility function to create a simple network with a single MaxPool node using
/// the function \p F and the bindings \p bindings.
static Node *createMaxPool(Function *F, PlaceholderBindings &bindings,
                           llvm::ArrayRef<dim_t> inputDims,
                           llvm::ArrayRef<dim_t> outputDims,
                           llvm::ArrayRef<unsigned_t> kernels,
                           llvm::ArrayRef<unsigned_t> strides,
                           llvm::ArrayRef<unsigned_t> pads) {
  // Create input placeholder.
  auto &mod = *(F->getParent());
  auto *input =
      mod.createPlaceholder(ElemKind::FloatTy, inputDims, "input", false);
  bindings.allocate(input)->getHandle<float>().randomize(-1.0, 1.0,
                                                         mod.getPRNG());
  // Create MaxPool.
  MaxPoolNode *maxpool =
      F->createMaxPool("maxpool", input, kernels, strides, pads);
  SaveNode *save = F->createSave("save", maxpool->getResult());
  bindings.allocate(save->getPlaceholder());
  EXPECT_EQ(maxpool->getResult().getType()->dims(), outputDims);
  return maxpool;
}

/// Utility function to test splitting a basic MaxPool node along the dimensions
/// \p splitDims in the given number chunks \p numChunks. The split is done
/// implicitly relative to the MaxPool output operand.
static void splitMaxPoolBasic(Function *F, Function *&optF,
                              PlaceholderBindings &bindings,
                              CompilationContext &cctx,
                              llvm::ArrayRef<size_t> splitDims,
                              llvm::ArrayRef<dim_t> numChunks) {
  Node *node = createMaxPool(F, bindings,
                             /* inputDims */ {3, 7, 8, 4},
                             /* outputDims */ {3, 6, 7, 4},
                             /* kernels */ {2, 2},
                             /* strides */ {1, 1},
                             /* pads */ {0, 0, 0, 0});

  // Save current function state as reference.
  optF = F->clone(F->getName().str() + "_optimized");

  // Split node.
  auto splitOption = SplitNodeByNumChunks(splitDims, numChunks);
  std::vector<Node *> splitNodes;
  ASSIGN_VALUE_OR_FAIL_TEST(splitNodes, ::glow::splitNode(node, splitOption));

  // Compute total number of chunks.
  dim_t totNumChunks = 1;
  for (auto numChunk : numChunks) {
    totNumChunks *= numChunk;
  }

  // Check node count.
  EXPECT_EQ(splitNodes.size(), totNumChunks);
  EXPECT_EQ(countNodeKind(F, Kinded::Kind::MaxPoolNodeKind), totNumChunks);
  EXPECT_EQ(countNodeKind(F, Kinded::Kind::InsertTensorNodeKind), totNumChunks);
  EXPECT_EQ(countNodeKind(F, Kinded::Kind::TouchNodeKind), 1);
}

/// Test splitting a MaxPool along dimension N, H, W or C.
#define TEST_MAXPOOL_BASIC_SPLIT(splitDim, numChunks)                          \
  TEST_F(NodeSplitting, MaxPool_Basic_Dim##splitDim##_Chunks##numChunks) {     \
    splitMaxPoolBasic(F_, optimizedF_, bindings_, cctx_,                       \
                      {ShapeNHWC::Dim##splitDim}, {numChunks});                \
    checkNumericalEquivalence(0);                                              \
  }
TEST_MAXPOOL_BASIC_SPLIT(N, 2)
TEST_MAXPOOL_BASIC_SPLIT(N, 3)
TEST_MAXPOOL_BASIC_SPLIT(H, 2)
TEST_MAXPOOL_BASIC_SPLIT(H, 3)
TEST_MAXPOOL_BASIC_SPLIT(H, 4)
TEST_MAXPOOL_BASIC_SPLIT(H, 5)
TEST_MAXPOOL_BASIC_SPLIT(H, 6)
TEST_MAXPOOL_BASIC_SPLIT(W, 2)
TEST_MAXPOOL_BASIC_SPLIT(W, 3)
TEST_MAXPOOL_BASIC_SPLIT(W, 4)
TEST_MAXPOOL_BASIC_SPLIT(W, 5)
TEST_MAXPOOL_BASIC_SPLIT(W, 6)
TEST_MAXPOOL_BASIC_SPLIT(W, 7)
TEST_MAXPOOL_BASIC_SPLIT(C, 2)
TEST_MAXPOOL_BASIC_SPLIT(C, 3)
TEST_MAXPOOL_BASIC_SPLIT(C, 4)
#undef TEST_MAXPOOL_BASIC_SPLIT

/// Utility function to test splitting a MaxPool node with non-zero padding
/// along the dimensions \p splitDims in the given number chunks \p numChunks.
/// The split is done implicitly relative to the MaxPool output operand.
static void splitMaxPoolNonZeroPad(Function *F, Function *&optF,
                                   PlaceholderBindings &bindings,
                                   CompilationContext &cctx,
                                   llvm::ArrayRef<size_t> splitDims,
                                   llvm::ArrayRef<dim_t> numChunks) {
  Node *node = createMaxPool(F, bindings,
                             /* inputDims */ {1, 4, 4, 1},
                             /* outputDims */ {1, 4, 8, 1},
                             /* kernels */ {2, 2},
                             /* strides */ {1, 1},
                             /* pads */ {0, 2, 1, 3});

  // Save current function state as reference.
  optF = F->clone(F->getName().str() + "_optimized");

  // Split node.
  auto splitOption = SplitNodeByNumChunks(splitDims, numChunks);
  std::vector<Node *> splitNodes;
  ASSIGN_VALUE_OR_FAIL_TEST(splitNodes, ::glow::splitNode(node, splitOption));

  // Compute total number of chunks.
  dim_t totNumChunks = 1;
  for (auto numChunk : numChunks) {
    totNumChunks *= numChunk;
  }

  // Check node count.
  EXPECT_EQ(splitNodes.size(), totNumChunks);
  EXPECT_EQ(countNodeKind(F, Kinded::Kind::MaxPoolNodeKind), totNumChunks);
  EXPECT_EQ(countNodeKind(F, Kinded::Kind::InsertTensorNodeKind), totNumChunks);
  EXPECT_EQ(countNodeKind(F, Kinded::Kind::TouchNodeKind), 1);
}

/// Test splitting a MaxPool with padding along dimension N, H, W or C.
#define TEST_MAXPOOL_NONZEROPAD_SPLIT(splitDim, numChunks)                     \
  TEST_F(NodeSplitting,                                                        \
         MaxPool_NonZeroPad_Dim##splitDim##_Chunks##numChunks) {               \
    splitMaxPoolNonZeroPad(F_, optimizedF_, bindings_, cctx_,                  \
                           {ShapeNHWC::Dim##splitDim}, {numChunks});           \
    checkNumericalEquivalence(0);                                              \
  }
TEST_MAXPOOL_NONZEROPAD_SPLIT(H, 2)
TEST_MAXPOOL_NONZEROPAD_SPLIT(W, 2)
#undef TEST_MAXPOOL_NONZEROPAD_SPLIT

/// Test splitting a MaxPool with padding along dimensions H, W.
TEST_F(NodeSplitting, MaxPool_NonZeroPad_DimHW_Chunks4) {
  splitMaxPoolNonZeroPad(F_, optimizedF_, bindings_, cctx_,
                         {ShapeNHWC::DimH, ShapeNHWC::DimW}, {2, 2});
  checkNumericalEquivalence(0);
}

/// Test splitting an "ill-defined" MaxPool for which not all the input
/// (including padding) is referenced by the output tensor. This happens
/// when using a stride larger than 1. This verifies that the node
/// splitting infrastructure uses a weaker verification of the mapping
/// between input and output for MaxPool.
TEST_F(NodeSplitting, MaxPool_IllDefined_DimHW) {
  std::vector<size_t> splitDims = {ShapeNHWC::DimH, ShapeNHWC::DimW};
  std::vector<dim_t> numChunks = {3, 3};
  Node *node = createMaxPool(F_, bindings_,
                             /* inputDims */ {1, 16, 18, 1},
                             /* outputDims */ {1, 8, 9, 1},
                             /* kernels */ {2, 2},
                             /* strides */ {2, 2},
                             /* pads */ {1, 1, 0, 0});

  // Save current function state as reference.
  optimizedF_ = F_->clone(F_->getName().str() + "_optimized");

  // Split node.
  auto splitOption = SplitNodeByNumChunks(splitDims, numChunks);
  std::vector<Node *> splitNodes;
  ASSIGN_VALUE_OR_FAIL_TEST(splitNodes, ::glow::splitNode(node, splitOption));

  // Check node count.
  dim_t totNumChunks = numChunks[0] * numChunks[1];
  EXPECT_EQ(splitNodes.size(), totNumChunks);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::MaxPoolNodeKind), totNumChunks);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::InsertTensorNodeKind),
            totNumChunks);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::TouchNodeKind), 1);
  checkNumericalEquivalence(0);
}

/// Test splitting a MaxPool based on memory constraint.
TEST_F(NodeSplitting, MaxPool_MaxMem) {
  Node *node = createMaxPool(F_, bindings_,
                             /* inputDims */ {3, 7, 8, 4},
                             /* outputDims */ {3, 6, 7, 4},
                             /* kernels */ {2, 2},
                             /* strides */ {1, 1},
                             /* pads */ {0, 0, 0, 0});

  // Save current function state as reference.
  optimizedF_ = F_->clone(F_->getName().str() + "_optimized");

  // Split node by memory size.
  auto origMemSize = node->getTotMemSize();
  auto splitMaxMemSize = origMemSize / 2;
  std::vector<Node *> splitNodes;
  ASSIGN_VALUE_OR_FAIL_TEST(
      splitNodes,
      ::glow::splitNode(node, SplitNodeMaxMemConstraint(splitMaxMemSize)));

  // Check node count.
  auto totNumChunks = countNodeKind(F_, Kinded::Kind::MaxPoolNodeKind);
  EXPECT_EQ(splitNodes.size(), totNumChunks);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::InsertTensorNodeKind),
            totNumChunks);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::TouchNodeKind), 1);

  // Check split nodes memory sizes.
  for (auto *splitNode : splitNodes) {
    EXPECT_TRUE(splitNode->getTotMemSize() <= splitMaxMemSize);
  }
  checkNumericalEquivalence(0);
}

/// Test that a MaxPool node is not split when the second output operand
/// Argmax has users.
TEST_F(NodeSplitting, MaxPool_Argmax_NoSplit) {
  std::vector<dim_t> inputDims = {1, 16, 18, 1};
  std::vector<dim_t> outputDims = {1, 8, 9, 1};
  std::vector<unsigned_t> kernels = {2, 2};
  std::vector<unsigned_t> strides = {2, 2};
  std::vector<unsigned_t> pads = {1, 1, 0, 0};
  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, inputDims, "input", false);
  bindings_.allocate(input)->getHandle<float>().randomize(-1.0, 1.0,
                                                          mod_.getPRNG());
  MaxPoolNode *maxpool =
      F_->createMaxPool("maxpool", input, kernels, strides, pads);
  SaveNode *saveResult = F_->createSave("saveResult", maxpool->getResult());
  bindings_.allocate(saveResult->getPlaceholder());
  SaveNode *saveArgmax = F_->createSave("saveArgmax", maxpool->getArgmax());
  bindings_.allocate(saveArgmax->getPlaceholder());
  std::vector<dim_t> actualOutputDims = maxpool->getResult().getType()->dims();
  EXPECT_EQ(actualOutputDims, outputDims);

  // Save current function state as reference.
  optimizedF_ = F_->clone(F_->getName().str() + "_optimized");

  // Split node.
  auto splitOption = SplitNodeByNumChunks({ShapeNHWC::DimH}, {3});
  std::vector<Node *> splitNodes;
  ASSIGN_VALUE_OR_FAIL_TEST(splitNodes,
                            ::glow::splitNode(maxpool, splitOption));

  // Check node count.
  EXPECT_EQ(splitNodes.size(), 0);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::SliceNodeKind), 0);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::MaxPoolNodeKind), 1);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::InsertTensorNodeKind), 0);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::TouchNodeKind), 0);
  checkNumericalEquivalence(0);
}

/// Test splitting a MaxPool based on an impossible constraint forcing the
/// split procedure to go through all the split configurations while verifying
/// them. In the end no split should be performed.
TEST_F(NodeSplitting, MaxPool_NoSplit) {
  Node *node = createMaxPool(F_, bindings_,
                             /* inputDims */ {3, 7, 8, 4},
                             /* outputDims */ {3, 6, 7, 4},
                             /* kernels */ {2, 2},
                             /* strides */ {1, 1},
                             /* pads */ {0, 0, 0, 0});

  // Save current function state as reference.
  optimizedF_ = F_->clone(F_->getName().str() + "_optimized");

  // Split node by memory size 0.
  std::vector<Node *> splitNodes;
  ASSIGN_VALUE_OR_FAIL_TEST(
      splitNodes, ::glow::splitNode(node, SplitNodeMaxMemConstraint(0)));

  // Check node count.
  EXPECT_EQ(splitNodes.size(), 0);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::SliceNodeKind), 0);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::MaxPoolNodeKind), 1);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::InsertTensorNodeKind), 0);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::TouchNodeKind), 0);
  checkNumericalEquivalence(0);
}

///===---------------------------------------------------------------------===//
///                                   AvgPool
///===---------------------------------------------------------------------===//
/// Utility function to create a simple network with a single AvgPool node using
/// the function \p F and the bindings \p bindings.
static Node *createAvgPool(Function *F, PlaceholderBindings &bindings,
                           llvm::ArrayRef<dim_t> inputDims,
                           llvm::ArrayRef<dim_t> outputDims,
                           llvm::ArrayRef<unsigned_t> kernels,
                           llvm::ArrayRef<unsigned_t> strides,
                           llvm::ArrayRef<unsigned_t> pads) {
  // Create input placeholder.
  auto &mod = *(F->getParent());
  auto *input =
      mod.createPlaceholder(ElemKind::FloatTy, inputDims, "input", false);
  bindings.allocate(input)->getHandle<float>().randomize(-1.0, 1.0,
                                                         mod.getPRNG());
  // Create AvgPool.
  AvgPoolNode *avgpool =
      F->createAvgPool("avgpool", input, kernels, strides, pads);
  SaveNode *save = F->createSave("save", avgpool->getResult());
  bindings.allocate(save->getPlaceholder());
  EXPECT_EQ(avgpool->getResult().getType()->dims(), outputDims);
  return avgpool;
}

/// Utility function to test splitting a basic AvgPool node along the dimensions
/// \p splitDims in the given number chunks \p numChunks. The split is done
/// implicitly relative to the AvgPool output operand.
static void splitAvgPoolBasic(Function *F, Function *&optF,
                              PlaceholderBindings &bindings,
                              CompilationContext &cctx,
                              llvm::ArrayRef<size_t> splitDims,
                              llvm::ArrayRef<dim_t> numChunks) {
  Node *node = createAvgPool(F, bindings,
                             /* inputDims */ {3, 7, 8, 4},
                             /* outputDims */ {3, 6, 7, 4},
                             /* kernels */ {2, 2},
                             /* strides */ {1, 1},
                             /* pads */ {0, 0, 0, 0});

  // Save current function state as reference.
  optF = F->clone(F->getName().str() + "_optimized");

  // Split node.
  auto splitOption = SplitNodeByNumChunks(splitDims, numChunks);
  std::vector<Node *> splitNodes;
  ASSIGN_VALUE_OR_FAIL_TEST(splitNodes, ::glow::splitNode(node, splitOption));

  // Compute total number of chunks.
  dim_t totNumChunks = 1;
  for (auto numChunk : numChunks) {
    totNumChunks *= numChunk;
  }

  // Check node count.
  EXPECT_EQ(splitNodes.size(), totNumChunks);
  EXPECT_EQ(countNodeKind(F, Kinded::Kind::AvgPoolNodeKind), totNumChunks);
  EXPECT_EQ(countNodeKind(F, Kinded::Kind::InsertTensorNodeKind), totNumChunks);
  EXPECT_EQ(countNodeKind(F, Kinded::Kind::TouchNodeKind), 1);
}

/// Test splitting a AvgPool along dimension N, H, W or C.
#define TEST_AVGPOOL_BASIC_SPLIT(splitDim, numChunks)                          \
  TEST_F(NodeSplitting, AvgPool_Basic_Dim##splitDim##_Chunks##numChunks) {     \
    splitAvgPoolBasic(F_, optimizedF_, bindings_, cctx_,                       \
                      {ShapeNHWC::Dim##splitDim}, {numChunks});                \
    checkNumericalEquivalence(0);                                              \
  }
TEST_AVGPOOL_BASIC_SPLIT(N, 2)
TEST_AVGPOOL_BASIC_SPLIT(N, 3)
TEST_AVGPOOL_BASIC_SPLIT(H, 2)
TEST_AVGPOOL_BASIC_SPLIT(H, 3)
TEST_AVGPOOL_BASIC_SPLIT(H, 4)
TEST_AVGPOOL_BASIC_SPLIT(H, 5)
TEST_AVGPOOL_BASIC_SPLIT(H, 6)
TEST_AVGPOOL_BASIC_SPLIT(W, 2)
TEST_AVGPOOL_BASIC_SPLIT(W, 3)
TEST_AVGPOOL_BASIC_SPLIT(W, 4)
TEST_AVGPOOL_BASIC_SPLIT(W, 5)
TEST_AVGPOOL_BASIC_SPLIT(W, 6)
TEST_AVGPOOL_BASIC_SPLIT(W, 7)
TEST_AVGPOOL_BASIC_SPLIT(C, 2)
TEST_AVGPOOL_BASIC_SPLIT(C, 3)
TEST_AVGPOOL_BASIC_SPLIT(C, 4)
#undef TEST_AVGPOOL_BASIC_SPLIT

/// Utility function to test splitting a AvgPool node with non-zero padding
/// along the dimensions \p splitDims in the given number chunks \p numChunks.
/// The split is done implicitly relative to the AvgPool output operand.
static void splitAvgPoolNonZeroPad(Function *F, Function *&optF,
                                   PlaceholderBindings &bindings,
                                   CompilationContext &cctx,
                                   llvm::ArrayRef<size_t> splitDims,
                                   llvm::ArrayRef<dim_t> numChunks) {
  Node *node = createAvgPool(F, bindings,
                             /* inputDims */ {1, 4, 4, 1},
                             /* outputDims */ {1, 4, 8, 1},
                             /* kernels */ {2, 2},
                             /* strides */ {1, 1},
                             /* pads */ {0, 2, 1, 3});

  // Save current function state as reference.
  optF = F->clone(F->getName().str() + "_optimized");

  // Split node.
  auto splitOption = SplitNodeByNumChunks(splitDims, numChunks);
  std::vector<Node *> splitNodes;
  ASSIGN_VALUE_OR_FAIL_TEST(splitNodes, ::glow::splitNode(node, splitOption));

  // Compute total number of chunks.
  dim_t totNumChunks = 1;
  for (auto numChunk : numChunks) {
    totNumChunks *= numChunk;
  }

  // Check node count.
  EXPECT_EQ(splitNodes.size(), totNumChunks);
  EXPECT_EQ(countNodeKind(F, Kinded::Kind::AvgPoolNodeKind), totNumChunks);
  EXPECT_EQ(countNodeKind(F, Kinded::Kind::InsertTensorNodeKind), totNumChunks);
  EXPECT_EQ(countNodeKind(F, Kinded::Kind::TouchNodeKind), 1);
}

/// Test splitting a AvgPool with padding along dimension N, H, W or C.
#define TEST_AVGPOOL_NONZEROPAD_SPLIT(splitDim, numChunks)                     \
  TEST_F(NodeSplitting,                                                        \
         AvgPool_NonZeroPad_Dim##splitDim##_Chunks##numChunks) {               \
    splitAvgPoolNonZeroPad(F_, optimizedF_, bindings_, cctx_,                  \
                           {ShapeNHWC::Dim##splitDim}, {numChunks});           \
    checkNumericalEquivalence(0);                                              \
  }
TEST_AVGPOOL_NONZEROPAD_SPLIT(H, 2)
TEST_AVGPOOL_NONZEROPAD_SPLIT(W, 2)
#undef TEST_AVGPOOL_NONZEROPAD_SPLIT

/// Test splitting a AvgPool with padding along dimensions H, W.
TEST_F(NodeSplitting, AvgPool_NonZeroPad_DimHW_Chunks4) {
  splitAvgPoolNonZeroPad(F_, optimizedF_, bindings_, cctx_,
                         {ShapeNHWC::DimH, ShapeNHWC::DimW}, {2, 2});
  checkNumericalEquivalence(0);
}

/// Test splitting an "ill-defined" AvgPool for which not all the input
/// (including padding) is referenced by the output tensor. This happens
/// when using a stride larger than 1. This verifies that the node
/// splitting infrastructure uses a weaker verification of the mapping
/// between input and output for AvgPool.
TEST_F(NodeSplitting, AvgPool_IllDefined_DimHW) {
  std::vector<size_t> splitDims = {ShapeNHWC::DimH, ShapeNHWC::DimW};
  std::vector<dim_t> numChunks = {3, 3};
  Node *node = createAvgPool(F_, bindings_,
                             /* inputDims */ {1, 16, 18, 1},
                             /* outputDims */ {1, 8, 9, 1},
                             /* kernels */ {2, 2},
                             /* strides */ {2, 2},
                             /* pads */ {1, 1, 0, 0});

  // Save current function state as reference.
  optimizedF_ = F_->clone(F_->getName().str() + "_optimized");

  // Split node.
  auto splitOption = SplitNodeByNumChunks(splitDims, numChunks);
  std::vector<Node *> splitNodes;
  ASSIGN_VALUE_OR_FAIL_TEST(splitNodes, ::glow::splitNode(node, splitOption));

  // Check node count.
  dim_t totNumChunks = numChunks[0] * numChunks[1];
  EXPECT_EQ(splitNodes.size(), totNumChunks);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::AvgPoolNodeKind), totNumChunks);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::InsertTensorNodeKind),
            totNumChunks);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::TouchNodeKind), 1);
  checkNumericalEquivalence(0);
}

/// Test splitting a AvgPool based on memory constraint.
TEST_F(NodeSplitting, AvgPool_MaxMem) {
  Node *node = createAvgPool(F_, bindings_,
                             /* inputDims */ {3, 7, 8, 4},
                             /* outputDims */ {3, 6, 7, 4},
                             /* kernels */ {2, 2},
                             /* strides */ {1, 1},
                             /* pads */ {0, 0, 0, 0});

  // Save current function state as reference.
  optimizedF_ = F_->clone(F_->getName().str() + "_optimized");

  // Split node by memory size.
  auto origMemSize = node->getTotMemSize();
  auto splitMaxMemSize = origMemSize / 2;
  std::vector<Node *> splitNodes;
  ASSIGN_VALUE_OR_FAIL_TEST(
      splitNodes,
      ::glow::splitNode(node, SplitNodeMaxMemConstraint(splitMaxMemSize)));

  // Check node count.
  auto totNumChunks = countNodeKind(F_, Kinded::Kind::AvgPoolNodeKind);
  EXPECT_EQ(splitNodes.size(), totNumChunks);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::InsertTensorNodeKind),
            totNumChunks);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::TouchNodeKind), 1);

  // Check split nodes memory sizes.
  for (auto *splitNode : splitNodes) {
    EXPECT_TRUE(splitNode->getTotMemSize() <= splitMaxMemSize);
  }
  checkNumericalEquivalence(0);
}

/// Test splitting a AvgPool based on an impossible constraint forcing the
/// split procedure to go through all the split configurations while verifying
/// them. In the end no split should be performed.
TEST_F(NodeSplitting, AvgPool_NoSplit) {
  Node *node = createAvgPool(F_, bindings_,
                             /* inputDims */ {3, 7, 8, 4},
                             /* outputDims */ {3, 6, 7, 4},
                             /* kernels */ {2, 2},
                             /* strides */ {1, 1},
                             /* pads */ {0, 0, 0, 0});

  // Save current function state as reference.
  optimizedF_ = F_->clone(F_->getName().str() + "_optimized");

  // Split node by memory size 0.
  std::vector<Node *> splitNodes;
  ASSIGN_VALUE_OR_FAIL_TEST(
      splitNodes, ::glow::splitNode(node, SplitNodeMaxMemConstraint(0)));

  // Check node count.
  EXPECT_EQ(splitNodes.size(), 0);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::SliceNodeKind), 0);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::AvgPoolNodeKind), 1);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::InsertTensorNodeKind), 0);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::TouchNodeKind), 0);
  checkNumericalEquivalence(0);
}

///===---------------------------------------------------------------------===//
///                               FullyConnected
///===---------------------------------------------------------------------===//
/// Utility function to test splitting a FullyConnected node.
static void splitFullyConnected(Function *F, Function *&optF,
                                PlaceholderBindings &bindings,
                                CompilationContext &cctx,
                                llvm::ArrayRef<size_t> splitDims,
                                llvm::ArrayRef<dim_t> numChunks) {
  std::vector<dim_t> inputDims = {10, 13};
  std::vector<dim_t> weightsDims = {13, 20};
  std::vector<dim_t> biasDims = {20};
  auto &mod = *(F->getParent());
  auto *input =
      mod.createPlaceholder(ElemKind::FloatTy, inputDims, "input", false);
  bindings.allocate(input)->getHandle<float>().randomize(-10.0, 10.0,
                                                         mod.getPRNG());
  auto *weights =
      mod.createPlaceholder(ElemKind::FloatTy, weightsDims, "weights", false);
  bindings.allocate(weights)->getHandle<float>().randomize(-10.0, 10.0,
                                                           mod.getPRNG());
  auto *bias =
      mod.createPlaceholder(ElemKind::FloatTy, biasDims, "bias", false);
  bindings.allocate(bias)->getHandle<float>().randomize(-10.0, 10.0,
                                                        mod.getPRNG());
  Node *node = F->createFullyConnected("fc", input, weights, bias);
  SaveNode *output = F->createSave("output", node);
  bindings.allocate(output->getPlaceholder());

  // Save current function state as reference.
  optF = F->clone(F->getName().str() + "_optimized");

  // Split node.
  auto splitOption = SplitNodeByNumChunks(splitDims, numChunks);
  std::vector<Node *> splitNodes;
  ASSIGN_VALUE_OR_FAIL_TEST(splitNodes, ::glow::splitNode(node, splitOption));

  // Compute total number of chunks.
  dim_t totNumChunks = 1;
  for (auto numChunk : numChunks) {
    totNumChunks *= numChunk;
  }

  // Check node count.
  EXPECT_EQ(splitNodes.size(), totNumChunks);
  EXPECT_EQ(countNodeKind(F, Kinded::Kind::FullyConnectedNodeKind),
            totNumChunks);
  EXPECT_EQ(countNodeKind(F, Kinded::Kind::InsertTensorNodeKind), totNumChunks);
  EXPECT_EQ(countNodeKind(F, Kinded::Kind::TouchNodeKind), 1);
}

/// Test splitting FullyConnected along dimension H.
TEST_F(NodeSplitting, FullyConnected_DimH_Chunks2) {
  splitFullyConnected(F_, optimizedF_, bindings_, cctx_, {ShapeHW::DimH}, {2});
  checkNumericalEquivalence(0);
}

/// Test splitting FullyConnected along dimension W.
TEST_F(NodeSplitting, FullyConnected_DimW_Chunks2) {
  splitFullyConnected(F_, optimizedF_, bindings_, cctx_, {ShapeHW::DimW}, {2});
  checkNumericalEquivalence(0);
}

/// Test splitting FullyConnected along dimension H and W.
TEST_F(NodeSplitting, FullyConnected_DimHW_Chunks4) {
  splitFullyConnected(F_, optimizedF_, bindings_, cctx_,
                      {ShapeHW::DimH, ShapeHW::DimW}, {2, 2});
  checkNumericalEquivalence(0);
}

///===---------------------------------------------------------------------===//
///                                   MatMul
///===---------------------------------------------------------------------===//
/// Utility function to test splitting a MatMul node.
static void splitMatMul(Function *F, Function *&optF,
                        PlaceholderBindings &bindings, CompilationContext &cctx,
                        llvm::ArrayRef<size_t> splitDims,
                        llvm::ArrayRef<dim_t> numChunks) {
  std::vector<dim_t> dimsLHS = {10, 13};
  std::vector<dim_t> dimsRHS = {13, 20};
  auto &mod = *(F->getParent());
  auto *LHS = mod.createPlaceholder(ElemKind::FloatTy, dimsLHS, "LHS", false);
  bindings.allocate(LHS)->getHandle<float>().randomize(-10.0, 10.0,
                                                       mod.getPRNG());
  auto *RHS = mod.createPlaceholder(ElemKind::FloatTy, dimsRHS, "RHS", false);
  bindings.allocate(RHS)->getHandle<float>().randomize(-10.0, 10.0,
                                                       mod.getPRNG());
  Node *node = F->createMatMul("matmul", LHS, RHS);
  SaveNode *output = F->createSave("output", node);
  bindings.allocate(output->getPlaceholder());

  // Save current function state as reference.
  optF = F->clone(F->getName().str() + "_optimized");

  // Split node.
  auto splitOption = SplitNodeByNumChunks(splitDims, numChunks);
  std::vector<Node *> splitNodes;
  ASSIGN_VALUE_OR_FAIL_TEST(splitNodes, ::glow::splitNode(node, splitOption));

  // Compute total number of chunks.
  dim_t totNumChunks = 1;
  for (auto numChunk : numChunks) {
    totNumChunks *= numChunk;
  }

  // Check node count.
  EXPECT_EQ(splitNodes.size(), totNumChunks);
  EXPECT_EQ(countNodeKind(F, Kinded::Kind::MatMulNodeKind), totNumChunks);
  EXPECT_EQ(countNodeKind(F, Kinded::Kind::InsertTensorNodeKind), totNumChunks);
  EXPECT_EQ(countNodeKind(F, Kinded::Kind::TouchNodeKind), 1);
}

/// Test splitting MatMul along dimension H.
TEST_F(NodeSplitting, MatMul_DimH_Chunks2) {
  splitMatMul(F_, optimizedF_, bindings_, cctx_, {ShapeHW::DimH}, {2});
  checkNumericalEquivalence(0);
}

/// Test splitting MatMul along dimension W.
TEST_F(NodeSplitting, MatMul_DimW_Chunks2) {
  splitMatMul(F_, optimizedF_, bindings_, cctx_, {ShapeHW::DimW}, {2});
  checkNumericalEquivalence(0);
}

/// Test splitting MatMul along dimension H and W.
TEST_F(NodeSplitting, MatMul_DimHW_Chunks4) {
  splitMatMul(F_, optimizedF_, bindings_, cctx_, {ShapeHW::DimH, ShapeHW::DimW},
              {2, 2});
  checkNumericalEquivalence(0);
}

///===---------------------------------------------------------------------===//
///                                 BatchMatMul
///===---------------------------------------------------------------------===//
/// Utility function to test splitting a BatchMatMul node.
static void splitBatchMatMul(Function *F, Function *&optF,
                             PlaceholderBindings &bindings,
                             CompilationContext &cctx,
                             llvm::ArrayRef<size_t> splitDims,
                             llvm::ArrayRef<dim_t> numChunks) {
  std::vector<dim_t> dimsLHS = {2, 10, 13};
  std::vector<dim_t> dimsRHS = {2, 13, 20};
  auto &mod = *(F->getParent());
  auto *LHS = mod.createPlaceholder(ElemKind::FloatTy, dimsLHS, "LHS", false);
  bindings.allocate(LHS)->getHandle<float>().randomize(-10.0, 10.0,
                                                       mod.getPRNG());
  auto *RHS = mod.createPlaceholder(ElemKind::FloatTy, dimsRHS, "RHS", false);
  bindings.allocate(RHS)->getHandle<float>().randomize(-10.0, 10.0,
                                                       mod.getPRNG());
  Node *node = F->createBatchMatMul("batchmatmul", LHS, RHS);
  SaveNode *output = F->createSave("output", node);
  bindings.allocate(output->getPlaceholder());

  // Save current function state as reference.
  optF = F->clone(F->getName().str() + "_optimized");

  // Split node.
  auto splitOption = SplitNodeByNumChunks(splitDims, numChunks);
  std::vector<Node *> splitNodes;
  ASSIGN_VALUE_OR_FAIL_TEST(splitNodes, ::glow::splitNode(node, splitOption));

  // Compute total number of chunks.
  dim_t totNumChunks = 1;
  for (auto numChunk : numChunks) {
    totNumChunks *= numChunk;
  }

  // Check node count.
  EXPECT_EQ(splitNodes.size(), totNumChunks);
  EXPECT_EQ(countNodeKind(F, Kinded::Kind::BatchMatMulNodeKind), totNumChunks);
  EXPECT_EQ(countNodeKind(F, Kinded::Kind::InsertTensorNodeKind), totNumChunks);
  EXPECT_EQ(countNodeKind(F, Kinded::Kind::TouchNodeKind), 1);
}

/// Test splitting BatchMatMul along dimension N.
TEST_F(NodeSplitting, BatchMatMul_DimN_Chunks2) {
  splitBatchMatMul(F_, optimizedF_, bindings_, cctx_, {ShapeNHW::DimN}, {2});
  checkNumericalEquivalence(0);
}

/// Test splitting BatchMatMul along dimension H.
TEST_F(NodeSplitting, BatchMatMul_DimH_Chunks2) {
  splitBatchMatMul(F_, optimizedF_, bindings_, cctx_, {ShapeNHW::DimH}, {2});
  checkNumericalEquivalence(0);
}

/// Test splitting BatchMatMul along dimension W.
TEST_F(NodeSplitting, BatchMatMul_DimW_Chunks2) {
  splitBatchMatMul(F_, optimizedF_, bindings_, cctx_, {ShapeNHW::DimW}, {2});
  checkNumericalEquivalence(0);
}

/// Test splitting BatchMatMul along dimension N and H.
TEST_F(NodeSplitting, BatchMatMul_DimNH_Chunks4) {
  splitBatchMatMul(F_, optimizedF_, bindings_, cctx_,
                   {ShapeNHW::DimN, ShapeNHW::DimH}, {2, 2});
  checkNumericalEquivalence(0);
}

/// Test splitting BatchMatMul along dimension N, H and W.
TEST_F(NodeSplitting, BatchMatMul_DimNHW_Chunks8) {
  splitBatchMatMul(F_, optimizedF_, bindings_, cctx_,
                   {ShapeNHW::DimN, ShapeNHW::DimH, ShapeNHW::DimW}, {2, 2, 2});
  checkNumericalEquivalence(0);
}

///===---------------------------------------------------------------------===//
///                                 BatchedAdd
///===---------------------------------------------------------------------===//
/// Utility function to test splitting a BatchedAdd node.
static void splitBatchedAdd(Function *F, Function *&optF,
                            PlaceholderBindings &bindings,
                            CompilationContext &cctx,
                            llvm::ArrayRef<size_t> splitDims,
                            llvm::ArrayRef<dim_t> numChunks) {
  std::vector<dim_t> batchDims = {2, 10, 13};
  std::vector<dim_t> sliceDims = {10, 13};
  auto &mod = *(F->getParent());
  auto *batch =
      mod.createPlaceholder(ElemKind::FloatTy, batchDims, "batch", false);
  bindings.allocate(batch)->getHandle<float>().randomize(-10.0, 10.0,
                                                         mod.getPRNG());
  auto *slice =
      mod.createPlaceholder(ElemKind::FloatTy, sliceDims, "slice", false);
  bindings.allocate(slice)->getHandle<float>().randomize(-10.0, 10.0,
                                                         mod.getPRNG());
  Node *node = F->createBatchedAdd("batchedadd", batch, slice);
  SaveNode *output = F->createSave("output", node);
  bindings.allocate(output->getPlaceholder());

  // Save current function state as reference.
  optF = F->clone(F->getName().str() + "_optimized");

  // Split node.
  auto splitOption = SplitNodeByNumChunks(splitDims, numChunks);
  std::vector<Node *> splitNodes;
  ASSIGN_VALUE_OR_FAIL_TEST(splitNodes, ::glow::splitNode(node, splitOption));

  // Compute total number of chunks.
  dim_t totNumChunks = 1;
  for (auto numChunk : numChunks) {
    totNumChunks *= numChunk;
  }

  // Check node count.
  EXPECT_EQ(splitNodes.size(), totNumChunks);
  EXPECT_EQ(countNodeKind(F, Kinded::Kind::BatchedAddNodeKind), totNumChunks);
  EXPECT_EQ(countNodeKind(F, Kinded::Kind::InsertTensorNodeKind), totNumChunks);
  EXPECT_EQ(countNodeKind(F, Kinded::Kind::TouchNodeKind), 1);
}

/// Test splitting BatchedAdd along dimension 0.
TEST_F(NodeSplitting, BatchedAdd_Dim0_Chunks2) {
  splitBatchedAdd(F_, optimizedF_, bindings_, cctx_, {0}, {2});
  checkNumericalEquivalence(0);
}

/// Test splitting BatchedAdd along dimension 1.
TEST_F(NodeSplitting, BatchedAdd_Dim1_Chunks2) {
  splitBatchedAdd(F_, optimizedF_, bindings_, cctx_, {1}, {2});
  checkNumericalEquivalence(0);
}

/// Test splitting BatchedAdd along dimension 2.
TEST_F(NodeSplitting, BatchedAdd_Dim2_Chunks2) {
  splitBatchedAdd(F_, optimizedF_, bindings_, cctx_, {2}, {2});
  checkNumericalEquivalence(0);
}

/// Test splitting BatchedAdd along dimension 0 and 1.
TEST_F(NodeSplitting, BatchedAdd_Dim01_Chunks4) {
  splitBatchedAdd(F_, optimizedF_, bindings_, cctx_, {0, 1}, {2, 2});
  checkNumericalEquivalence(0);
}

/// Test splitting BatchedAdd along dimension 0, 1 and 2.
TEST_F(NodeSplitting, BatchedAdd_Dim012_Chunks8) {
  splitBatchedAdd(F_, optimizedF_, bindings_, cctx_, {0, 1, 2}, {2, 2, 2});
  checkNumericalEquivalence(0);
}

///===---------------------------------------------------------------------===//
///                                  Transpose
///===---------------------------------------------------------------------===//
/// Utility function to test splitting a Transpose node.
static void splitTranspose(Function *F, Function *&optF,
                           PlaceholderBindings &bindings,
                           CompilationContext &cctx,
                           llvm::ArrayRef<size_t> splitDims,
                           llvm::ArrayRef<dim_t> numChunks) {
  std::vector<dim_t> inputDims = {3, 5, 7};
  std::vector<unsigned_t> shuffle = {2, 0, 1};
  auto &mod = *(F->getParent());
  auto *input =
      mod.createPlaceholder(ElemKind::FloatTy, inputDims, "input", false);
  bindings.allocate(input)->getHandle<float>().randomize(-10.0, 10.0,
                                                         mod.getPRNG());
  Node *node = F->createTranspose("transpose", input, shuffle);
  SaveNode *output = F->createSave("output", node);
  bindings.allocate(output->getPlaceholder());

  // Save current function state as reference.
  optF = F->clone(F->getName().str() + "_optimized");

  // Split node.
  auto splitOption = SplitNodeByNumChunks(splitDims, numChunks);
  std::vector<Node *> splitNodes;
  ASSIGN_VALUE_OR_FAIL_TEST(splitNodes, ::glow::splitNode(node, splitOption));

  // Compute total number of chunks.
  dim_t totNumChunks = 1;
  for (auto numChunk : numChunks) {
    totNumChunks *= numChunk;
  }

  // Check node count.
  EXPECT_EQ(splitNodes.size(), totNumChunks);
  EXPECT_EQ(countNodeKind(F, Kinded::Kind::TransposeNodeKind), totNumChunks);
  EXPECT_EQ(countNodeKind(F, Kinded::Kind::InsertTensorNodeKind), totNumChunks);
  EXPECT_EQ(countNodeKind(F, Kinded::Kind::TouchNodeKind), 1);
}

/// Test splitting Transpose along dimension 0.
TEST_F(NodeSplitting, Transpose_Dim0_Chunks2) {
  splitTranspose(F_, optimizedF_, bindings_, cctx_, {0}, {2});
  checkNumericalEquivalence(0);
}

/// Test splitting Transpose along dimension 1.
TEST_F(NodeSplitting, Transpose_Dim1_Chunks2) {
  splitTranspose(F_, optimizedF_, bindings_, cctx_, {1}, {2});
  checkNumericalEquivalence(0);
}

/// Test splitting Transpose along dimension 2.
TEST_F(NodeSplitting, Transpose_Dim2_Chunks2) {
  splitTranspose(F_, optimizedF_, bindings_, cctx_, {2}, {2});
  checkNumericalEquivalence(0);
}

/// Test splitting Transpose along dimension 0 and 1.
TEST_F(NodeSplitting, Transpose_Dim01_Chunks4) {
  splitTranspose(F_, optimizedF_, bindings_, cctx_, {0, 1}, {2, 2});
  checkNumericalEquivalence(0);
}

/// Test splitting Transpose along dimension 0, 1 and 2.
TEST_F(NodeSplitting, Transpose_Dim012_Chunks8) {
  splitTranspose(F_, optimizedF_, bindings_, cctx_, {0, 1, 2}, {2, 2, 2});
  checkNumericalEquivalence(0);
}

///===---------------------------------------------------------------------===//
///                              Binary Operators
///===---------------------------------------------------------------------===//
/// Test splitting binary operators.
TEST_F(NodeSplitting, BinaryOps) {
  // Create network with parallel binary operators.
  std::vector<dim_t> dims = {10, 10};
  auto *inputLHS =
      mod_.createPlaceholder(ElemKind::FloatTy, dims, "inputLHS", false);
  auto *inputRHS =
      mod_.createPlaceholder(ElemKind::FloatTy, dims, "inputRHS", false);
  bindings_.allocate(inputLHS)->getHandle<float>().randomize(1.0, 2.0,
                                                             mod_.getPRNG());
  bindings_.allocate(inputRHS)->getHandle<float>().randomize(1.0, 2.0,
                                                             mod_.getPRNG());
  Node *add = F_->createAdd("add", inputLHS, inputRHS);
  Node *mul = F_->createMul("mul", inputLHS, inputRHS);
  Node *sub = F_->createSub("sub", inputLHS, inputRHS);
  Node *div = F_->createDiv("div", inputLHS, inputRHS);
  Node *max = F_->createMax("max", inputLHS, inputRHS);
  Node *min = F_->createMin("min", inputLHS, inputRHS);
  Node *cmpLTE = F_->createCmpLTE("cmpLTE", inputLHS, inputRHS);
  Node *cmpLT = F_->createCmpLT("cmpLT", inputLHS, inputRHS);
  Node *cmpEQ = F_->createCmpEQ("cmpEQ", inputLHS, inputRHS);
  Node *pow = F_->createPow("pow", inputLHS, inputRHS);
  SaveNode *addSave = F_->createSave("addSave", add);
  SaveNode *mulSave = F_->createSave("mulSave", mul);
  SaveNode *subSave = F_->createSave("subSave", sub);
  SaveNode *divSave = F_->createSave("divSave", div);
  SaveNode *maxSave = F_->createSave("maxSave", max);
  SaveNode *minSave = F_->createSave("minSave", min);
  SaveNode *cmpLTESave = F_->createSave("cmpLTESave", cmpLTE);
  SaveNode *cmpLTSave = F_->createSave("cmpLTSave", cmpLT);
  SaveNode *cmpEQSave = F_->createSave("cmpEQSave", cmpEQ);
  SaveNode *powSave = F_->createSave("powSave", pow);
  bindings_.allocate(addSave->getPlaceholder());
  bindings_.allocate(mulSave->getPlaceholder());
  bindings_.allocate(subSave->getPlaceholder());
  bindings_.allocate(divSave->getPlaceholder());
  bindings_.allocate(maxSave->getPlaceholder());
  bindings_.allocate(minSave->getPlaceholder());
  bindings_.allocate(cmpLTESave->getPlaceholder());
  bindings_.allocate(cmpLTSave->getPlaceholder());
  bindings_.allocate(cmpEQSave->getPlaceholder());
  bindings_.allocate(powSave->getPlaceholder());

  // Save current function state as reference.
  optimizedF_ = F_->clone(F_->getName().str() + "_optimized");

  // Split nodes.
  auto splitOption = SplitNodeByNumChunks({0}, {2});
  SplitNodeMap splitMap;
  ASSIGN_VALUE_OR_FAIL_TEST(splitMap, ::glow::splitNodes(F_, splitOption));

  // Check node count.
  EXPECT_EQ(2, splitMap[add].size());
  EXPECT_EQ(2, splitMap[mul].size());
  EXPECT_EQ(2, splitMap[sub].size());
  EXPECT_EQ(2, splitMap[div].size());
  EXPECT_EQ(2, splitMap[max].size());
  EXPECT_EQ(2, splitMap[min].size());
  EXPECT_EQ(2, splitMap[cmpLTE].size());
  EXPECT_EQ(2, splitMap[cmpLT].size());
  EXPECT_EQ(2, splitMap[cmpEQ].size());
  EXPECT_EQ(2, splitMap[pow].size());
  EXPECT_EQ(2, countNodeKind(F_, Kinded::Kind::AddNodeKind));
  EXPECT_EQ(2, countNodeKind(F_, Kinded::Kind::MulNodeKind));
  EXPECT_EQ(2, countNodeKind(F_, Kinded::Kind::SubNodeKind));
  EXPECT_EQ(2, countNodeKind(F_, Kinded::Kind::DivNodeKind));
  EXPECT_EQ(2, countNodeKind(F_, Kinded::Kind::MaxNodeKind));
  EXPECT_EQ(2, countNodeKind(F_, Kinded::Kind::MinNodeKind));
  EXPECT_EQ(2, countNodeKind(F_, Kinded::Kind::CmpLTENodeKind));
  EXPECT_EQ(2, countNodeKind(F_, Kinded::Kind::CmpLTNodeKind));
  EXPECT_EQ(2, countNodeKind(F_, Kinded::Kind::CmpEQNodeKind));
  EXPECT_EQ(2, countNodeKind(F_, Kinded::Kind::PowNodeKind));
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::SliceNodeKind), 10 * 2 * 2);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::InsertTensorNodeKind), 10 * 2);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::TouchNodeKind), 10);
  checkNumericalEquivalence(0);
}

///===---------------------------------------------------------------------===//
///                               Unary Operators
///===---------------------------------------------------------------------===//
/// Test splitting unary operators.
TEST_F(NodeSplitting, UnaryOps) {
  std::vector<dim_t> dims = {10, 10};
  auto quantizeTy = mod_.uniqueType(ElemKind::Int8QTy, dims, 1.0, 0);
  auto requantizeTy = mod_.uniqueType(ElemKind::Int8QTy, dims, 0.5, 0);
  auto convertTy = mod_.uniqueType(ElemKind::Float16Ty, dims);

  // Create network with chained unary operators.
  auto *input = mod_.createPlaceholder(ElemKind::FloatTy, dims, "input", false);
  bindings_.allocate(input)->getHandle<float>().randomize(-10.0, 10.0,
                                                          mod_.getPRNG());
  Node *relu = F_->createRELU("relu", input);
  Node *leakyRelu = F_->createLeakyRELU("leakyrelu", relu, 0.1);
  Node *clip = F_->createClip("clip", leakyRelu, 1.0, 10.0);
  Node *tanh = F_->createTanh("tanh", clip);
  Node *sigmoid = F_->createSigmoid("sigmoid", tanh);
  Node *log = F_->createLog("log", sigmoid);
  Node *exp = F_->createExp("exp", log);
  Node *quantize = F_->createQuantize("quantize", exp, quantizeTy);
  Node *requantize =
      F_->createRescaleQuantized("requantize", quantize, requantizeTy);
  Node *dequantize =
      F_->createDequantize("dequantize", requantize, ElemKind::FloatTy);
  Node *convert = F_->createConvertTo("convert", dequantize, convertTy);
  SaveNode *output = F_->createSave("output", convert);
  bindings_.allocate(output->getPlaceholder());

  // Save current function state as reference.
  optimizedF_ = F_->clone(F_->getName().str() + "_optimized");

  // Split nodes.
  auto splitOption = SplitNodeByNumChunks({0}, {2});
  SplitNodeMap splitMap;
  ASSIGN_VALUE_OR_FAIL_TEST(splitMap, ::glow::splitNodes(F_, splitOption));

  // Check node count.
  EXPECT_EQ(2, splitMap[relu].size());
  EXPECT_EQ(2, splitMap[leakyRelu].size());
  EXPECT_EQ(2, splitMap[clip].size());
  EXPECT_EQ(2, splitMap[tanh].size());
  EXPECT_EQ(2, splitMap[sigmoid].size());
  EXPECT_EQ(2, splitMap[log].size());
  EXPECT_EQ(2, splitMap[exp].size());
  EXPECT_EQ(2, splitMap[quantize].size());
  EXPECT_EQ(2, splitMap[requantize].size());
  EXPECT_EQ(2, splitMap[dequantize].size());
  EXPECT_EQ(2, splitMap[convert].size());
  EXPECT_EQ(2, countNodeKind(F_, Kinded::Kind::ReluNodeKind));
  EXPECT_EQ(2, countNodeKind(F_, Kinded::Kind::LeakyReluNodeKind));
  EXPECT_EQ(2, countNodeKind(F_, Kinded::Kind::ClipNodeKind));
  EXPECT_EQ(2, countNodeKind(F_, Kinded::Kind::TanhNodeKind));
  EXPECT_EQ(2, countNodeKind(F_, Kinded::Kind::SigmoidNodeKind));
  EXPECT_EQ(2, countNodeKind(F_, Kinded::Kind::LogNodeKind));
  EXPECT_EQ(2, countNodeKind(F_, Kinded::Kind::ExpNodeKind));
  EXPECT_EQ(2, countNodeKind(F_, Kinded::Kind::QuantizeNodeKind));
  EXPECT_EQ(2, countNodeKind(F_, Kinded::Kind::RescaleQuantizedNodeKind));
  EXPECT_EQ(2, countNodeKind(F_, Kinded::Kind::DequantizeNodeKind));
  EXPECT_EQ(2, countNodeKind(F_, Kinded::Kind::ConvertToNodeKind));
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::SliceNodeKind), 11 * 2);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::InsertTensorNodeKind), 11 * 2);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::TouchNodeKind), 11);
  checkNumericalEquivalence(0);
}

///===---------------------------------------------------------------------===//
///                            Non-Orthogonal Split
///===---------------------------------------------------------------------===//
/// Utility function to test splitting a Conv2D non-orthogonally based on given
/// raw \p sliceRanges.
static void splitConv2DNonOrthogonal(Function *F, Function *&optF,
                                     PlaceholderBindings &bindings,
                                     CompilationContext &cctx,
                                     llvm::ArrayRef<SliceRange> sliceRanges) {
  Node *node = createConv2D(F, bindings,
                            /* inputDims */ {5, 4, 4, 7},
                            /* filterDims */ {8, 3, 3, 7},
                            /* biasDims */ {8},
                            /* outputDims */ {5, 4, 4, 8},
                            /* kernels */ {3, 3},
                            /* strides */ {1, 1},
                            /* pads */ {1, 1, 1, 1},
                            /* group */ 1,
                            /* dilation */ {1, 1});

  // Save current function state as reference.
  optF = F->clone(F->getName().str() + "_optimized");

  // Split node.
  auto splitOption = SplitNodeBySliceRanges(sliceRanges);
  std::vector<Node *> splitNodes;
  ASSIGN_VALUE_OR_FAIL_TEST(splitNodes, ::glow::splitNode(node, splitOption));

  // Check node count.
  dim_t totNumChunks = sliceRanges.size();
  EXPECT_EQ(splitNodes.size(), totNumChunks);
  EXPECT_EQ(countNodeKind(F, Kinded::Kind::ConvolutionNodeKind), totNumChunks);
  EXPECT_EQ(countNodeKind(F, Kinded::Kind::InsertTensorNodeKind), totNumChunks);
  EXPECT_EQ(countNodeKind(F, Kinded::Kind::TouchNodeKind), 1);
}

/// Test splitting a Conv2D non-orthogonally along N dimension.
TEST_F(NodeSplitting, Conv2D_NonOrthogonal_DimN) {
  std::vector<SliceRange> sliceRanges = {
      SliceRange({{0, 4}, {0, 4}, {0, 4}, {0, 8}}),
      SliceRange({{2, 5}, {0, 4}, {0, 4}, {0, 8}})};
  splitConv2DNonOrthogonal(F_, optimizedF_, bindings_, cctx_, sliceRanges);
  checkNumericalEquivalence(0);
}

/// Test splitting a Conv2D non-orthogonally along H dimension.
TEST_F(NodeSplitting, Conv2D_NonOrthogonal_DimH) {
  std::vector<SliceRange> sliceRanges = {
      SliceRange({{0, 5}, {0, 3}, {0, 4}, {0, 8}}),
      SliceRange({{0, 5}, {2, 4}, {0, 4}, {0, 8}})};
  splitConv2DNonOrthogonal(F_, optimizedF_, bindings_, cctx_, sliceRanges);
  checkNumericalEquivalence(0);
}

/// Test splitting a Conv2D non-orthogonally along W dimension.
TEST_F(NodeSplitting, Conv2D_NonOrthogonal_DimW) {
  std::vector<SliceRange> sliceRanges = {
      SliceRange({{0, 5}, {0, 4}, {0, 3}, {0, 8}}),
      SliceRange({{0, 5}, {0, 4}, {2, 4}, {0, 8}})};
  splitConv2DNonOrthogonal(F_, optimizedF_, bindings_, cctx_, sliceRanges);
  checkNumericalEquivalence(0);
}

/// Test splitting a Conv2D non-orthogonally along C dimension.
TEST_F(NodeSplitting, Conv2D_NonOrthogonal_DimC) {
  std::vector<SliceRange> sliceRanges = {
      SliceRange({{0, 5}, {0, 4}, {0, 4}, {0, 6}}),
      SliceRange({{0, 5}, {0, 4}, {0, 4}, {2, 8}})};
  splitConv2DNonOrthogonal(F_, optimizedF_, bindings_, cctx_, sliceRanges);
  checkNumericalEquivalence(0);
}

/// Test splitting a Conv2D non-orthogonally along H and W dimensions.
TEST_F(NodeSplitting, Conv2D_NonOrthogonal_DimHW) {
  std::vector<SliceRange> sliceRanges = {
      SliceRange({{0, 5}, {0, 3}, {0, 3}, {0, 8}}),
      SliceRange({{0, 5}, {0, 3}, {1, 4}, {0, 8}}),
      SliceRange({{0, 5}, {1, 4}, {0, 3}, {0, 8}}),
      SliceRange({{0, 5}, {1, 4}, {1, 4}, {0, 8}})};
  splitConv2DNonOrthogonal(F_, optimizedF_, bindings_, cctx_, sliceRanges);
  checkNumericalEquivalence(0);
}

/// Test splitting a Conv2D non-orthogonally along H, W and C dimensions.
TEST_F(NodeSplitting, Conv2D_NonOrthogonal_DimHWC) {
  std::vector<SliceRange> sliceRanges = {
      SliceRange({{0, 5}, {0, 3}, {0, 3}, {0, 6}}),
      SliceRange({{0, 5}, {0, 3}, {1, 4}, {0, 6}}),
      SliceRange({{0, 5}, {1, 4}, {0, 3}, {0, 6}}),
      SliceRange({{0, 5}, {1, 4}, {1, 4}, {0, 6}}),
      SliceRange({{0, 5}, {0, 3}, {0, 3}, {2, 8}}),
      SliceRange({{0, 5}, {0, 3}, {1, 4}, {2, 8}}),
      SliceRange({{0, 5}, {1, 4}, {0, 3}, {2, 8}}),
      SliceRange({{0, 5}, {1, 4}, {1, 4}, {2, 8}})};
  splitConv2DNonOrthogonal(F_, optimizedF_, bindings_, cctx_, sliceRanges);
  checkNumericalEquivalence(0);
}

/// Test splitting a Conv2D non-orthogonally along N, H, W and C dimensions.
TEST_F(NodeSplitting, Conv2D_NonOrthogonal_DimNHWC) {
  std::vector<SliceRange> sliceRanges = {
      SliceRange({{0, 4}, {0, 3}, {0, 3}, {0, 6}}),
      SliceRange({{0, 4}, {0, 3}, {1, 4}, {0, 6}}),
      SliceRange({{0, 4}, {1, 4}, {0, 3}, {0, 6}}),
      SliceRange({{0, 4}, {1, 4}, {1, 4}, {0, 6}}),
      SliceRange({{0, 4}, {0, 3}, {0, 3}, {2, 8}}),
      SliceRange({{0, 4}, {0, 3}, {1, 4}, {2, 8}}),
      SliceRange({{0, 4}, {1, 4}, {0, 3}, {2, 8}}),
      SliceRange({{0, 4}, {1, 4}, {1, 4}, {2, 8}}),
      SliceRange({{2, 5}, {0, 3}, {0, 3}, {0, 6}}),
      SliceRange({{2, 5}, {0, 3}, {1, 4}, {0, 6}}),
      SliceRange({{2, 5}, {1, 4}, {0, 3}, {0, 6}}),
      SliceRange({{2, 5}, {1, 4}, {1, 4}, {0, 6}}),
      SliceRange({{2, 5}, {0, 3}, {0, 3}, {2, 8}}),
      SliceRange({{2, 5}, {0, 3}, {1, 4}, {2, 8}}),
      SliceRange({{2, 5}, {1, 4}, {0, 3}, {2, 8}}),
      SliceRange({{2, 5}, {1, 4}, {1, 4}, {2, 8}})};
  splitConv2DNonOrthogonal(F_, optimizedF_, bindings_, cctx_, sliceRanges);
  checkNumericalEquivalence(0);
}

///===---------------------------------------------------------------------===//
///                            Recursive Split
///===---------------------------------------------------------------------===//
/// Utility function to split Conv2D with Relu recursively.
static void splitConv2DReluRecursive(Function *F, Function *&optF,
                                     PlaceholderBindings &bindings,
                                     CompilationContext &cctx,
                                     SplitNodeOption &splitOption) {
  // Conv2D params.
  std::vector<dim_t> inputDims = {5, 8, 8, 4};
  std::vector<dim_t> filterDims = {16, 3, 3, 4};
  std::vector<dim_t> biasDims = {16};
  std::vector<dim_t> outputDims = {5, 4, 4, 16};
  std::vector<unsigned_t> kernels = {3, 3};
  std::vector<unsigned_t> strides = {2, 2};
  std::vector<unsigned_t> pads = {1, 1, 0, 0};
  dim_t group = 1;
  std::vector<unsigned_t> dilation = {1, 1};

  // Create input placeholder.
  auto &mod = *(F->getParent());
  auto *input =
      mod.createPlaceholder(ElemKind::FloatTy, inputDims, "input", false);
  bindings.allocate(input)->getHandle<float>().randomize(-1.0, 1.0,
                                                         mod.getPRNG());
  // Create filter constant.
  auto *filter = mod.createConstant(ElemKind::FloatTy, filterDims, "filter");
  filter->getPayloadMutable().getHandle<float>().randomize(-1.0, 1.0,
                                                           mod.getPRNG());
  // Create bias constant.
  auto *bias = mod.createConstant(ElemKind::FloatTy, biasDims, "bias");
  bias->getPayloadMutable().getHandle<float>().randomize(-1.0, 1.0,
                                                         mod.getPRNG());
  // Create Conv2D.
  auto *outTy = mod.uniqueType(ElemKind::FloatTy, outputDims);
  auto *conv = F->createConv("conv", input, filter, bias, outTy, kernels,
                             strides, pads, group, dilation);
  // Create Relu.
  auto *relu = F->createRELU("relu", conv);
  // Create Save.
  SaveNode *save = F->createSave("save", relu);
  bindings.allocate(save->getPlaceholder());

  // Save current function state as reference.
  optF = F->clone(F->getName().str() + "_optimized");

  // Split node recursively.
  SplitNodeMap splitMap;
  ASSIGN_VALUE_OR_FAIL_TEST(
      splitMap,
      ::glow::splitNodeRecursively(relu, splitOption, /* maxDepth */ 10));

  EXPECT_EQ(splitMap.size(), 2);
  EXPECT_TRUE(splitMap.count(conv));
  EXPECT_TRUE(splitMap.count(relu));
}

/// Test splitting Conv2D with Relu along N dimension.
TEST_F(NodeSplitting, Conv2D_Relu_Recursive_DimN) {
  auto splitOption = SplitNodeByNumChunks({ShapeNHWC::DimN}, {2});
  splitConv2DReluRecursive(F_, optimizedF_, bindings_, cctx_, splitOption);
  checkNumericalEquivalence(0);
}

/// Test splitting Conv2D with Relu along H dimension.
TEST_F(NodeSplitting, Conv2D_Relu_Recursive_DimH) {
  auto splitOption = SplitNodeByNumChunks({ShapeNHWC::DimH}, {2});
  splitConv2DReluRecursive(F_, optimizedF_, bindings_, cctx_, splitOption);
  checkNumericalEquivalence(0);
}

/// Test splitting Conv2D with Relu along W dimension.
TEST_F(NodeSplitting, Conv2D_Relu_Recursive_DimW) {
  auto splitOption = SplitNodeByNumChunks({ShapeNHWC::DimW}, {2});
  splitConv2DReluRecursive(F_, optimizedF_, bindings_, cctx_, splitOption);
  checkNumericalEquivalence(0);
}

/// Test splitting Conv2D with Relu along C dimension.
TEST_F(NodeSplitting, Conv2D_Relu_Recursive_DimC) {
  auto splitOption = SplitNodeByNumChunks({ShapeNHWC::DimC}, {2});
  splitConv2DReluRecursive(F_, optimizedF_, bindings_, cctx_, splitOption);
  checkNumericalEquivalence(0);
}

/// Test splitting Conv2D with Relu along H and W dimensions.
TEST_F(NodeSplitting, Conv2D_Relu_Recursive_DimHW) {
  auto splitOption =
      SplitNodeByNumChunks({ShapeNHWC::DimH, ShapeNHWC::DimW}, {2, 2});
  splitConv2DReluRecursive(F_, optimizedF_, bindings_, cctx_, splitOption);
  checkNumericalEquivalence(0);
}

/// Test splitting Conv2D with Relu along H, W and C dimensions.
TEST_F(NodeSplitting, Conv2D_Relu_Recursive_DimHWC) {
  auto splitOption = SplitNodeByNumChunks(
      {ShapeNHWC::DimH, ShapeNHWC::DimW, ShapeNHWC::DimC}, {2, 2, 2});
  splitConv2DReluRecursive(F_, optimizedF_, bindings_, cctx_, splitOption);
  checkNumericalEquivalence(0);
}

/// Test splitting Conv2D with Relu along N, H, W and C dimensions.
TEST_F(NodeSplitting, Conv2D_Relu_Recursive_DimNHWC) {
  auto splitOption = SplitNodeByNumChunks(
      {ShapeNHWC::DimN, ShapeNHWC::DimH, ShapeNHWC::DimW, ShapeNHWC::DimC},
      {2, 2, 2, 2});
  splitConv2DReluRecursive(F_, optimizedF_, bindings_, cctx_, splitOption);
  checkNumericalEquivalence(0);
}

/// Utility function to split Conv2D with Relu and MaxPool recursively.
static void splitConv2DReluMaxPoolRecursive(Function *F, Function *&optF,
                                            PlaceholderBindings &bindings,
                                            CompilationContext &cctx,
                                            SplitNodeOption &splitOption) {
  // Conv2D params.
  std::vector<dim_t> inputDims = {5, 16, 16, 4};
  std::vector<dim_t> filterDims = {16, 3, 3, 4};
  std::vector<dim_t> biasDims = {16};
  std::vector<dim_t> outputDims = {5, 8, 8, 16};
  std::vector<unsigned_t> kernels = {3, 3};
  std::vector<unsigned_t> strides = {2, 2};
  std::vector<unsigned_t> pads = {1, 1, 0, 0};
  dim_t group = 1;
  std::vector<unsigned_t> dilation = {1, 1};

  // MaxPool params.
  std::vector<unsigned_t> poolKernels = {3, 3};
  std::vector<unsigned_t> poolStrides = {2, 2};
  std::vector<unsigned_t> poolPads = {1, 1, 1, 1};

  // Create input placeholder.
  auto &mod = *(F->getParent());
  auto *input =
      mod.createPlaceholder(ElemKind::FloatTy, inputDims, "input", false);
  bindings.allocate(input)->getHandle<float>().randomize(-1.0, 1.0,
                                                         mod.getPRNG());
  // Create filter constant.
  auto *filter = mod.createConstant(ElemKind::FloatTy, filterDims, "filter");
  filter->getPayloadMutable().getHandle<float>().randomize(-1.0, 1.0,
                                                           mod.getPRNG());
  // Create bias constant.
  auto *bias = mod.createConstant(ElemKind::FloatTy, biasDims, "bias");
  bias->getPayloadMutable().getHandle<float>().randomize(-1.0, 1.0,
                                                         mod.getPRNG());
  // Create Conv2D.
  auto *outTy = mod.uniqueType(ElemKind::FloatTy, outputDims);
  auto *conv = F->createConv("conv", input, filter, bias, outTy, kernels,
                             strides, pads, group, dilation);
  // Create Relu.
  auto *relu = F->createRELU("relu", conv);
  // Create MaxPool.
  auto *pool =
      F->createMaxPool("pool", relu, poolKernels, poolStrides, poolPads);
  // Create Save.
  SaveNode *save = F->createSave("save", pool->getResult());
  bindings.allocate(save->getPlaceholder());

  // Save current function state as reference.
  optF = F->clone(F->getName().str() + "_optimized");

  // Split node recursively.
  SplitNodeMap splitMap;
  ASSIGN_VALUE_OR_FAIL_TEST(
      splitMap,
      ::glow::splitNodeRecursively(pool, splitOption, /* maxDepth */ 10));

  EXPECT_EQ(splitMap.size(), 3);
  EXPECT_TRUE(splitMap.count(conv));
  EXPECT_TRUE(splitMap.count(relu));
  EXPECT_TRUE(splitMap.count(pool));
}

/// Test splitting Conv2D with Relu and MaxPool along N dimension.
TEST_F(NodeSplitting, Conv2D_Relu_MaxPool_Recursive_DimN) {
  auto splitOption = SplitNodeByNumChunks({ShapeNHWC::DimN}, {2});
  splitConv2DReluMaxPoolRecursive(F_, optimizedF_, bindings_, cctx_,
                                  splitOption);
  checkNumericalEquivalence(0);
}

/// Test splittingConv2D with Relu and MaxPool along H dimension.
TEST_F(NodeSplitting, Conv2D_Relu_MaxPool_Recursive_DimH) {
  auto splitOption = SplitNodeByNumChunks({ShapeNHWC::DimH}, {2});
  splitConv2DReluMaxPoolRecursive(F_, optimizedF_, bindings_, cctx_,
                                  splitOption);
  checkNumericalEquivalence(0);
}

/// Test splitting Conv2D with Relu and MaxPool along W dimension.
TEST_F(NodeSplitting, Conv2D_Relu_MaxPool_Recursive_DimW) {
  auto splitOption = SplitNodeByNumChunks({ShapeNHWC::DimW}, {2});
  splitConv2DReluMaxPoolRecursive(F_, optimizedF_, bindings_, cctx_,
                                  splitOption);
  checkNumericalEquivalence(0);
}

/// Test splitting Conv2D with Relu and MaxPool along C dimension.
TEST_F(NodeSplitting, Conv2D_Relu_MaxPool_Recursive_DimC) {
  auto splitOption = SplitNodeByNumChunks({ShapeNHWC::DimC}, {2});
  splitConv2DReluMaxPoolRecursive(F_, optimizedF_, bindings_, cctx_,
                                  splitOption);
  checkNumericalEquivalence(0);
}

/// Test splitting Conv2D with Relu and MaxPool along H and W dimensions.
TEST_F(NodeSplitting, Conv2D_Relu_MaxPool_Recursive_DimHW) {
  auto splitOption =
      SplitNodeByNumChunks({ShapeNHWC::DimH, ShapeNHWC::DimW}, {2, 2});
  splitConv2DReluMaxPoolRecursive(F_, optimizedF_, bindings_, cctx_,
                                  splitOption);
  checkNumericalEquivalence(0);
}

/// Test splitting Conv2D with Relu and MaxPool along H, W and C dimensions.
TEST_F(NodeSplitting, Conv2D_Relu_MaxPool_Recursive_DimHWC) {
  auto splitOption = SplitNodeByNumChunks(
      {ShapeNHWC::DimH, ShapeNHWC::DimW, ShapeNHWC::DimC}, {2, 2, 2});
  splitConv2DReluMaxPoolRecursive(F_, optimizedF_, bindings_, cctx_,
                                  splitOption);
  checkNumericalEquivalence(0);
}

/// Test splitting Conv2D with Relu and MaxPool along N, H, W and C dimensions.
TEST_F(NodeSplitting, Conv2D_Relu_MaxPool_Recursive_DimNHWC) {
  auto splitOption = SplitNodeByNumChunks(
      {ShapeNHWC::DimN, ShapeNHWC::DimH, ShapeNHWC::DimW, ShapeNHWC::DimC},
      {2, 2, 2, 2});
  splitConv2DReluMaxPoolRecursive(F_, optimizedF_, bindings_, cctx_,
                                  splitOption);
  checkNumericalEquivalence(0);
}

/// Verify that the recursive splitting max depth parameter is honored.
TEST_F(NodeSplitting, Recursive_MaxDepth) {
  std::vector<dim_t> dims = {10, 10};

  // Create network with chained unary operators.
  auto *input = mod_.createPlaceholder(ElemKind::FloatTy, dims, "input", false);
  bindings_.allocate(input)->getHandle<float>().randomize(-10.0, 10.0,
                                                          mod_.getPRNG());
  Node *relu = F_->createRELU("relu", input);
  Node *clip = F_->createClip("clip", relu, 1.0, 10.0);
  Node *tanh = F_->createTanh("tanh", clip);
  Node *sigmoid = F_->createSigmoid("sigmoid", tanh);
  SaveNode *output = F_->createSave("output", sigmoid);
  bindings_.allocate(output->getPlaceholder());

  // Save current function state as reference.
  optimizedF_ = F_->clone(F_->getName().str() + "_optimized");

  // Split nodes.
  unsigned maxDepth = 2;
  auto splitOption = SplitNodeByNumChunks({0}, {2});
  SplitNodeMap splitMap;
  ASSIGN_VALUE_OR_FAIL_TEST(
      splitMap, ::glow::splitNodeRecursively(sigmoid, splitOption, maxDepth));

  // Check node count.
  EXPECT_EQ(splitMap.size(), 2);
  EXPECT_EQ(0, splitMap.count(relu));
  EXPECT_EQ(0, splitMap.count(clip));
  EXPECT_EQ(1, splitMap.count(tanh));
  EXPECT_EQ(1, splitMap.count(sigmoid));
  EXPECT_EQ(2, splitMap[tanh].size());
  EXPECT_EQ(2, splitMap[sigmoid].size());
  EXPECT_EQ(1, countNodeKind(F_, Kinded::Kind::ReluNodeKind));
  EXPECT_EQ(1, countNodeKind(F_, Kinded::Kind::ClipNodeKind));
  EXPECT_EQ(2, countNodeKind(F_, Kinded::Kind::TanhNodeKind));
  EXPECT_EQ(2, countNodeKind(F_, Kinded::Kind::SigmoidNodeKind));
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::SliceNodeKind), 2);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::InsertTensorNodeKind), 2);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::TouchNodeKind), 1);
  checkNumericalEquivalence(0);
}

/// Verify recursive splitting for nodes with single output uses.
TEST_F(NodeSplitting, Recursive_SingleOutputUse) {
  std::vector<dim_t> dims = {10, 10};

  // Create network with chained unary operators.
  auto *input = mod_.createPlaceholder(ElemKind::FloatTy, dims, "input", false);
  bindings_.allocate(input)->getHandle<float>().randomize(-10.0, 10.0,
                                                          mod_.getPRNG());
  Node *relu = F_->createRELU("relu", input);
  Node *clip = F_->createClip("clip", relu, 1.0, 10.0);
  Node *tanh = F_->createTanh("tanh", clip);
  Node *sigmoid = F_->createSigmoid("sigmoid", tanh);
  SaveNode *output1 = F_->createSave("output1", clip);
  SaveNode *output2 = F_->createSave("output2", sigmoid);
  bindings_.allocate(output1->getPlaceholder());
  bindings_.allocate(output2->getPlaceholder());

  // Save current function state as reference.
  optimizedF_ = F_->clone(F_->getName().str() + "_optimized");

  // Split nodes.
  bool singleUseOnly = true;
  unsigned maxDepth = 10;
  auto splitOption = SplitNodeByNumChunks({0}, {2});
  SplitNodeMap splitMap;
  ASSIGN_VALUE_OR_FAIL_TEST(
      splitMap, ::glow::splitNodeRecursively(sigmoid, splitOption, maxDepth,
                                             singleUseOnly));

  // Check node count.
  EXPECT_EQ(splitMap.size(), 2);
  EXPECT_EQ(0, splitMap.count(relu));
  EXPECT_EQ(0, splitMap.count(clip));
  EXPECT_EQ(1, splitMap.count(tanh));
  EXPECT_EQ(1, splitMap.count(sigmoid));
  EXPECT_EQ(2, splitMap[tanh].size());
  EXPECT_EQ(2, splitMap[sigmoid].size());
  EXPECT_EQ(1, countNodeKind(F_, Kinded::Kind::ReluNodeKind));
  EXPECT_EQ(1, countNodeKind(F_, Kinded::Kind::ClipNodeKind));
  EXPECT_EQ(2, countNodeKind(F_, Kinded::Kind::TanhNodeKind));
  EXPECT_EQ(2, countNodeKind(F_, Kinded::Kind::SigmoidNodeKind));
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::SliceNodeKind), 2);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::InsertTensorNodeKind), 2);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::TouchNodeKind), 1);
  checkNumericalEquivalence(0);
}

/// Verify recursive splitting for nodes with multiple output uses.
TEST_F(NodeSplitting, Recursive_MultipleOutputUses) {
  std::vector<dim_t> dims = {10, 10};

  // Create network with chained unary operators.
  auto *input = mod_.createPlaceholder(ElemKind::FloatTy, dims, "input", false);
  bindings_.allocate(input)->getHandle<float>().randomize(-10.0, 10.0,
                                                          mod_.getPRNG());
  Node *relu = F_->createRELU("relu", input);
  Node *clip = F_->createClip("clip", relu, 1.0, 10.0);
  Node *tanh = F_->createTanh("tanh", clip);
  Node *sigmoid = F_->createSigmoid("sigmoid", tanh);
  SaveNode *output1 = F_->createSave("output1", clip);
  SaveNode *output2 = F_->createSave("output2", sigmoid);
  bindings_.allocate(output1->getPlaceholder());
  bindings_.allocate(output2->getPlaceholder());

  // Save current function state as reference.
  optimizedF_ = F_->clone(F_->getName().str() + "_optimized");

  // Split nodes.
  bool singleUseOnly = false;
  unsigned maxDepth = 10;
  auto splitOption = SplitNodeByNumChunks({0}, {2});
  SplitNodeMap splitMap;
  ASSIGN_VALUE_OR_FAIL_TEST(
      splitMap, ::glow::splitNodeRecursively(sigmoid, splitOption, maxDepth,
                                             singleUseOnly));

  // Check node count.
  EXPECT_EQ(splitMap.size(), 4);
  EXPECT_EQ(1, splitMap.count(relu));
  EXPECT_EQ(1, splitMap.count(clip));
  EXPECT_EQ(1, splitMap.count(tanh));
  EXPECT_EQ(1, splitMap.count(sigmoid));
  EXPECT_EQ(2, splitMap[relu].size());
  EXPECT_EQ(2, splitMap[clip].size());
  EXPECT_EQ(2, splitMap[tanh].size());
  EXPECT_EQ(2, splitMap[sigmoid].size());
  EXPECT_EQ(2, countNodeKind(F_, Kinded::Kind::ReluNodeKind));
  EXPECT_EQ(2, countNodeKind(F_, Kinded::Kind::ClipNodeKind));
  EXPECT_EQ(2, countNodeKind(F_, Kinded::Kind::TanhNodeKind));
  EXPECT_EQ(2, countNodeKind(F_, Kinded::Kind::SigmoidNodeKind));
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::SliceNodeKind), 2);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::InsertTensorNodeKind), 4);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::TouchNodeKind), 2);
  checkNumericalEquivalence(0);
}

/// Verify that the recursive splitting stops based on constraint.
TEST_F(NodeSplitting, Recursive_StopConstraint) {
  std::vector<dim_t> dims = {10, 10};

  // Create network with chained unary operators.
  auto *input = mod_.createPlaceholder(ElemKind::FloatTy, dims, "input", false);
  bindings_.allocate(input)->getHandle<float>().randomize(-10.0, 10.0,
                                                          mod_.getPRNG());
  Node *relu = F_->createRELU("relu", input);
  Node *clip = F_->createClip("clip", relu, 1.0, 10.0);
  Node *tanh = F_->createTanh("tanh", clip);
  Node *sigmoid = F_->createSigmoid("sigmoid", tanh);
  SaveNode *output1 = F_->createSave("output1", clip);
  SaveNode *output2 = F_->createSave("output2", sigmoid);
  bindings_.allocate(output1->getPlaceholder());
  bindings_.allocate(output2->getPlaceholder());

  // Save current function state as reference.
  optimizedF_ = F_->clone(F_->getName().str() + "_optimized");

  // Split nodes.
  unsigned maxDepth = 10;
  auto splitOption = SplitNodeByNumChunks({0}, {2});
  SplitNodeConstraint splitConstraint =
      [=](const Node *origNode, const std::vector<Node *> &splitNodes) -> bool {
    return (origNode->getKind() != Kinded::Kind::ClipNodeKind);
  };
  SplitNodeMap splitMap;
  ASSIGN_VALUE_OR_FAIL_TEST(
      splitMap, ::glow::splitNodeRecursively(sigmoid, &splitOption,
                                             &splitConstraint, maxDepth));

  // Check node count.
  EXPECT_EQ(splitMap.size(), 2);
  EXPECT_EQ(0, splitMap.count(relu));
  EXPECT_EQ(0, splitMap.count(clip));
  EXPECT_EQ(1, splitMap.count(tanh));
  EXPECT_EQ(1, splitMap.count(sigmoid));
  EXPECT_EQ(2, splitMap[tanh].size());
  EXPECT_EQ(2, splitMap[sigmoid].size());
  EXPECT_EQ(1, countNodeKind(F_, Kinded::Kind::ReluNodeKind));
  EXPECT_EQ(1, countNodeKind(F_, Kinded::Kind::ClipNodeKind));
  EXPECT_EQ(2, countNodeKind(F_, Kinded::Kind::TanhNodeKind));
  EXPECT_EQ(2, countNodeKind(F_, Kinded::Kind::SigmoidNodeKind));
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::SliceNodeKind), 2);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::InsertTensorNodeKind), 2);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::TouchNodeKind), 1);
  checkNumericalEquivalence(0);
}

/// Verify that the recursive splitting stops when reaching unsupported node.
TEST_F(NodeSplitting, Recursive_StopUnsupportedNode) {
  std::vector<dim_t> inputDims = {1, 5, 5, 4};
  std::vector<dim_t> outputDims = {1, 10, 10, 4};
  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, inputDims, "input", false);
  bindings_.allocate(input)->getHandle<float>().randomize(-10.0, 10.0,
                                                          mod_.getPRNG());
  auto *outTy = mod_.uniqueType(ElemKind::FloatTy, outputDims);
  auto *resize = F_->createResizeBilinear("resize", input, outTy);
  Node *relu = F_->createRELU("relu", resize);
  SaveNode *output = F_->createSave("output", relu);
  bindings_.allocate(output->getPlaceholder());

  // Save current function state as reference.
  optimizedF_ = F_->clone(F_->getName().str() + "_optimized");

  // Split nodes.
  unsigned maxDepth = 10;
  auto splitOption = SplitNodeByNumChunks({1}, {2});
  SplitNodeMap splitMap;
  ASSIGN_VALUE_OR_FAIL_TEST(
      splitMap, ::glow::splitNodeRecursively(relu, splitOption, maxDepth));

  // Check node count.
  EXPECT_EQ(splitMap.size(), 1);
  EXPECT_EQ(1, splitMap.count(relu));
  EXPECT_EQ(0, splitMap.count(resize));
  EXPECT_EQ(2, splitMap[relu].size());
  EXPECT_EQ(2, countNodeKind(F_, Kinded::Kind::ReluNodeKind));
  EXPECT_EQ(1, countNodeKind(F_, Kinded::Kind::ResizeBilinearNodeKind));
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::SliceNodeKind), 2);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::InsertTensorNodeKind), 2);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::TouchNodeKind), 1);
  checkNumericalEquivalence(0);
}

/// Verify that the recursive splitting only attempts to split the first output
/// operand. In this example we verify that the recursive splitting does not
/// attempt to split the ArgMax operand of MaxPool.
TEST_F(NodeSplitting, Recursive_OnlyFirstOutputOperand) {
  std::vector<dim_t> inputDims = {2, 8, 8, 4};
  std::vector<unsigned_t> kernels = {3, 3};
  std::vector<unsigned_t> strides = {1, 1};
  std::vector<unsigned_t> pads = {1, 1, 1, 1};

  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, inputDims, "input", false);
  bindings_.allocate(input)->getHandle<float>().randomize(-10.0, 10.0,
                                                          mod_.getPRNG());
  auto *pool = F_->createMaxPool("pool", input, kernels, strides, pads);
  Node *relu = F_->createRELU("relu", pool->getArgmax());
  SaveNode *result = F_->createSave("result", pool->getResult());
  SaveNode *argmax = F_->createSave("argmax", relu);
  bindings_.allocate(result->getPlaceholder());
  bindings_.allocate(argmax->getPlaceholder());

  // Save current function state as reference.
  optimizedF_ = F_->clone(F_->getName().str() + "_optimized");

  // Split nodes.
  unsigned maxDepth = 10;
  auto splitOption = SplitNodeByNumChunks({0}, {2});
  SplitNodeMap splitMap;
  ASSIGN_VALUE_OR_FAIL_TEST(
      splitMap, ::glow::splitNodeRecursively(relu, splitOption, maxDepth));

  // Check node count.
  EXPECT_EQ(splitMap.size(), 1);
  EXPECT_EQ(1, splitMap.count(relu));
  EXPECT_EQ(0, splitMap.count(pool));
  EXPECT_EQ(2, splitMap[relu].size());
  EXPECT_EQ(2, countNodeKind(F_, Kinded::Kind::ReluNodeKind));
  EXPECT_EQ(1, countNodeKind(F_, Kinded::Kind::MaxPoolNodeKind));
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::SliceNodeKind), 2);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::InsertTensorNodeKind), 2);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::TouchNodeKind), 1);
  // Note: We do not run the inference since we do not support Relu for INT64.
}

/// Verify that identical Touch nodes to not get reused. The Touch node should
/// be excepted from CSE because it is not safe to assume that identical Touch
/// nodes produce same output because the output is not initialized. The
/// presence of the Touch node in CSE tends to increase the lifetime of some
/// buffers resulting in extra copy instructions (runtime overhead) and
/// increased buffer lifetimes (memory overhead).
TEST_F(NodeSplitting, DoNotShareTouchNodes) {
  std::vector<dim_t> inputDims = {2, 2};
  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, inputDims, "input", false);
  bindings_.allocate(input)->getHandle<float>().randomize(-10.0, 10.0,
                                                          mod_.getPRNG());
  Node *relu1 = F_->createRELU("relu1", input);
  Node *relu2 = F_->createRELU("relu2", relu1);
  SaveNode *result = F_->createSave("result", relu2);
  bindings_.allocate(result->getPlaceholder());

  // Save current function state as reference.
  optimizedF_ = F_->clone(F_->getName().str() + "_optimized");

  // Split nodes.
  auto splitOption = SplitNodeByNumChunks({0}, {2});
  std::vector<Node *> splitNodes1;
  ASSIGN_VALUE_OR_FAIL_TEST(splitNodes1, ::glow::splitNode(relu1, splitOption));
  std::vector<Node *> splitNodes2;
  ASSIGN_VALUE_OR_FAIL_TEST(splitNodes2, ::glow::splitNode(relu2, splitOption));

  // Optimize function (this includes CSE).
  ::glow::optimize(F_, cctx_);

  // Check node count.
  EXPECT_EQ(splitNodes1.size(), 2);
  EXPECT_EQ(splitNodes2.size(), 2);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::ReluNodeKind), 4);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::SliceNodeKind), 4);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::InsertTensorNodeKind), 4);

  // We should have 2 Touch nodes and not 1 because we should not reuse them.
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::TouchNodeKind), 2);
}
