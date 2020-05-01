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

/// Test for SplitNodeByNumChunks option.
TEST_F(NodeSplitting, SplitNodeByNumChunksOptionTest) {
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
TEST_F(NodeSplitting, SplitNodeByChunkSizeOptionTest) {
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
TEST_F(NodeSplitting, SplitNodeByChunkSizesOptionTest) {
  auto opt = SplitNodeByChunkSizes({0, 1, 2, 3},
                                   {{1, 3, 3, 3}, {2, 4, 4}, {5, 5}, {4, 6}});
  EXPECT_EQ(opt.splitAlongDim(0, 10), std::vector<dim_t>({1, 3, 3, 3}));
  EXPECT_EQ(opt.splitAlongDim(1, 10), std::vector<dim_t>({2, 4, 4}));
  EXPECT_EQ(opt.splitAlongDim(2, 10), std::vector<dim_t>({5, 5}));
  EXPECT_EQ(opt.splitAlongDim(3, 10), std::vector<dim_t>({4, 6}));
}

/// Test for SplitNodeByChunkWeights option.
TEST_F(NodeSplitting, SplitNodeByChunkWeightsOptionTest) {
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
static Node *
createConv2D(Function *F, PlaceholderBindings &bindings,
             std::vector<dim_t> inputDims, std::vector<dim_t> filterDims,
             std::vector<dim_t> biasDims, std::vector<dim_t> outputDims,
             std::vector<unsigned_t> kernels, std::vector<unsigned_t> strides,
             std::vector<unsigned_t> pads, dim_t group, dim_t dilation) {
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
                             std::vector<size_t> splitDims,
                             std::vector<dim_t> numChunks) {
  // Create basic Conv2D.
  Node *conv = createConv2D(F, bindings,
                            /* inputDims */ {5, 7, 8, 2},
                            /* filterDims */ {8, 2, 2, 1},
                            /* biasDims */ {8},
                            /* outputDims */ {5, 6, 7, 8},
                            /* kernels */ {2, 2},
                            /* strides */ {1, 1},
                            /* pads */ {0, 0, 0, 0},
                            /* group */ 2,
                            /* dilation */ 1);

  // Optimize current function and save.
  ::glow::optimize(F, CompilationMode::Infer);
  optF = F->clone(F->getName().str() + "_optimized");

  // Split node.
  auto splitOption = SplitNodeByNumChunks(splitDims, numChunks);
  std::vector<Node *> splitNodes;
  ASSIGN_VALUE_OR_FAIL_TEST(
      splitNodes, ::glow::splitNodeWithConstraints(conv, &splitOption, {}));
  runDCEPass(F, cctx);

  // Compute total number of chunks.
  dim_t totNumChunks = 1;
  for (auto numChunk : numChunks) {
    totNumChunks *= numChunk;
  }

  // Check node count.
  EXPECT_EQ(splitNodes.size(), totNumChunks);
  EXPECT_EQ(countNodeKind(F, Kinded::Kind::SliceNodeKind), 3 * totNumChunks);
  EXPECT_EQ(countNodeKind(F, Kinded::Kind::ConvolutionNodeKind), totNumChunks);
  EXPECT_EQ(countNodeKind(F, Kinded::Kind::InsertTensorNodeKind), totNumChunks);
  EXPECT_EQ(countNodeKind(F, Kinded::Kind::TouchNodeKind), 1);
}

/// Test splitting a Conv2D along dimension N, H, W or C.
/// Not all the combinations are allowed when splitting along C.
#define TEST_CONV2D_BASIC_SPLIT(splitDim, numChunks)                           \
  TEST_F(NodeSplitting, Conv2D_Basic_Dim##splitDim##_Chunks##numChunks) {      \
    splitConv2DBasic(F_, optimizedF_, bindings_, cctx_,                        \
                     {ShapeNHWC::dim##splitDim}, {numChunks});                 \
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
                   {ShapeNHWC::dimN, ShapeNHWC::dimH}, {2, 2});
  checkNumericalEquivalence(0);
}

/// Test splitting a Conv2D along dimensions N, H, W.
TEST_F(NodeSplitting, Conv2D_Basic_DimNHW_Chunks8) {
  splitConv2DBasic(F_, optimizedF_, bindings_, cctx_,
                   {ShapeNHWC::dimN, ShapeNHWC::dimH, ShapeNHWC::dimW},
                   {2, 2, 2});
  checkNumericalEquivalence(0);
}

/// Test splitting a Conv2D along dimensions N, H, W, C.
TEST_F(NodeSplitting, Conv2D_Basic_DimNHWC_Chunks16) {
  splitConv2DBasic(
      F_, optimizedF_, bindings_, cctx_,
      {ShapeNHWC::dimN, ShapeNHWC::dimH, ShapeNHWC::dimW, ShapeNHWC::dimC},
      {2, 2, 2, 2});
  checkNumericalEquivalence(0);
}

/// Utility function to test splitting a Conv2D node with non-zero padding
/// along the dimensions \p splitDims in the given number chunks \p numChunks.
/// The split is done implicitly relative to the Conv2D output operand.
static void splitConv2DNonZeroPad(Function *F, Function *&optF,
                                  PlaceholderBindings &bindings,
                                  CompilationContext &cctx,
                                  std::vector<size_t> splitDims,
                                  std::vector<dim_t> numChunks) {
  // Create Conv2D with non-zero padding.
  Node *conv = createConv2D(F, bindings,
                            /* inputDims */ {1, 4, 4, 1},
                            /* filterDims */ {2, 2, 2, 1},
                            /* biasDims */ {2},
                            /* outputDims */ {1, 4, 8, 2},
                            /* kernels */ {2, 2},
                            /* strides */ {1, 1},
                            /* pads */ {0, 2, 1, 3},
                            /* group */ 1,
                            /* dilation */ 1);

  // Optimize current function and save.
  ::glow::optimize(F, CompilationMode::Infer);
  optF = F->clone(F->getName().str() + "_optimized");

  // Split node.
  auto splitOption = SplitNodeByNumChunks(splitDims, numChunks);
  std::vector<Node *> splitNodes;
  ASSIGN_VALUE_OR_FAIL_TEST(
      splitNodes, ::glow::splitNodeWithConstraints(conv, &splitOption, {}));
  runDCEPass(F, cctx);

  // Compute total number of chunks.
  dim_t totNumChunks = 1;
  for (auto numChunk : numChunks) {
    totNumChunks *= numChunk;
  }

  // Check node count.
  EXPECT_EQ(splitNodes.size(), totNumChunks);
  EXPECT_EQ(countNodeKind(F, Kinded::Kind::SliceNodeKind), 3 * totNumChunks);
  EXPECT_EQ(countNodeKind(F, Kinded::Kind::ConvolutionNodeKind), totNumChunks);
  EXPECT_EQ(countNodeKind(F, Kinded::Kind::InsertTensorNodeKind), totNumChunks);
  EXPECT_EQ(countNodeKind(F, Kinded::Kind::TouchNodeKind), 1);
}

/// Test splitting a Conv2D with padding along dimension N, H, W or C.
#define TEST_CONV2D_NONZEROPAD_SPLIT(splitDim, numChunks)                      \
  TEST_F(NodeSplitting, Conv2D_NonZeroPad_Dim##splitDim##_Chunks##numChunks) { \
    splitConv2DNonZeroPad(F_, optimizedF_, bindings_, cctx_,                   \
                          {ShapeNHWC::dim##splitDim}, {numChunks});            \
    checkNumericalEquivalence(0);                                              \
  }
TEST_CONV2D_NONZEROPAD_SPLIT(H, 2)
TEST_CONV2D_NONZEROPAD_SPLIT(W, 2)
#undef TEST_CONV2D_NONZEROPAD_SPLIT

/// Test splitting a Conv2D with padding along dimensions H, W.
TEST_F(NodeSplitting, Conv2D_NonZeroPad_DimHW_Chunks4) {
  splitConv2DNonZeroPad(F_, optimizedF_, bindings_, cctx_,
                        {ShapeNHWC::dimH, ShapeNHWC::dimW}, {2, 2});
  checkNumericalEquivalence(0);
}

/// Test splitting an "ill-defined" Conv2D for which not all the input
/// (including padding) is referenced by the output tensor. This happens
/// when using a stride larger than 1. This verifies that the node
/// splitting infrastructure uses a weaker verification of the mapping
/// between input and output for Conv2D.
TEST_F(NodeSplitting, Conv2D_IllDefined_DimHW) {
  std::vector<size_t> splitDims = {ShapeNHWC::dimH, ShapeNHWC::dimW};
  std::vector<dim_t> numChunks = {3, 3};
  Node *conv = createConv2D(F_, bindings_,
                            /* inputDims */ {1, 16, 18, 1},
                            /* filterDims */ {1, 2, 2, 1},
                            /* biasDims */ {1},
                            /* outputDims */ {1, 8, 9, 1},
                            /* kernels */ {2, 2},
                            /* strides */ {2, 2},
                            /* pads */ {1, 1, 0, 0},
                            /* group */ 1,
                            /* dilation */ 1);

  // Optimize current function and save.
  ::glow::optimize(F_, CompilationMode::Infer);
  optimizedF_ = F_->clone(F_->getName().str() + "_optimized");

  // Split node.
  auto splitOption = SplitNodeByNumChunks(splitDims, numChunks);
  std::vector<Node *> splitNodes;
  ASSIGN_VALUE_OR_FAIL_TEST(
      splitNodes, ::glow::splitNodeWithConstraints(conv, &splitOption, {}));
  runDCEPass(F_, cctx_);

  // Check node count.
  dim_t totNumChunks = numChunks[0] * numChunks[1];
  EXPECT_EQ(splitNodes.size(), totNumChunks);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::SliceNodeKind), 3 * totNumChunks);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::ConvolutionNodeKind), totNumChunks);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::InsertTensorNodeKind),
            totNumChunks);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::TouchNodeKind), 1);
}

///===---------------------------------------------------------------------===//
///                                   MaxPool
///===---------------------------------------------------------------------===//
/// Utility function to create a simple network with a single MaxPool node using
/// the function \p F and the bindings \p bindings.
static Node *createMaxPool(Function *F, PlaceholderBindings &bindings,
                           std::vector<dim_t> inputDims,
                           std::vector<dim_t> outputDims,
                           std::vector<unsigned_t> kernels,
                           std::vector<unsigned_t> strides,
                           std::vector<unsigned_t> pads) {

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
  std::vector<dim_t> actualOutputDims = maxpool->getResult().getType()->dims();
  EXPECT_EQ(actualOutputDims, outputDims);
  return maxpool;
}

/// Utility function to test splitting a basic MaxPool node along the dimensions
/// \p splitDims in the given number chunks \p numChunks. The split is done
/// implicitly relative to the MaxPool output operand.
static void splitMaxPoolBasic(Function *F, Function *&optF,
                              PlaceholderBindings &bindings,
                              CompilationContext &cctx,
                              std::vector<size_t> splitDims,
                              std::vector<dim_t> numChunks) {
  // Create basic MaxPool.
  Node *maxpool = createMaxPool(F, bindings,
                                /* inputDims */ {3, 7, 8, 4},
                                /* outputDims */ {3, 6, 7, 4},
                                /* kernels */ {2, 2},
                                /* strides */ {1, 1},
                                /* pads */ {0, 0, 0, 0});

  // Optimize current function and save.
  ::glow::optimize(F, CompilationMode::Infer);
  optF = F->clone(F->getName().str() + "_optimized");

  // Split node.
  auto splitOption = SplitNodeByNumChunks(splitDims, numChunks);
  std::vector<Node *> splitNodes;
  ASSIGN_VALUE_OR_FAIL_TEST(
      splitNodes, ::glow::splitNodeWithConstraints(maxpool, &splitOption, {}));
  runDCEPass(F, cctx);

  // Compute total number of chunks.
  dim_t totNumChunks = 1;
  for (auto numChunk : numChunks) {
    totNumChunks *= numChunk;
  }

  // Check node count.
  EXPECT_EQ(splitNodes.size(), totNumChunks);
  EXPECT_EQ(countNodeKind(F, Kinded::Kind::SliceNodeKind), totNumChunks);
  EXPECT_EQ(countNodeKind(F, Kinded::Kind::MaxPoolNodeKind), totNumChunks);
  EXPECT_EQ(countNodeKind(F, Kinded::Kind::InsertTensorNodeKind), totNumChunks);
  EXPECT_EQ(countNodeKind(F, Kinded::Kind::TouchNodeKind), 1);
}

/// Test splitting a MaxPool along dimension N, H, W or C.
#define TEST_MAXPOOL_BASIC_SPLIT(splitDim, numChunks)                          \
  TEST_F(NodeSplitting, MaxPool_Basic_Dim##splitDim##_Chunks##numChunks) {     \
    splitMaxPoolBasic(F_, optimizedF_, bindings_, cctx_,                       \
                      {ShapeNHWC::dim##splitDim}, {numChunks});                \
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
                                   std::vector<size_t> splitDims,
                                   std::vector<dim_t> numChunks) {
  // Create MaxPool with non-zero padding.
  Node *maxpool = createMaxPool(F, bindings,
                                /* inputDims */ {1, 4, 4, 1},
                                /* outputDims */ {1, 4, 8, 1},
                                /* kernels */ {2, 2},
                                /* strides */ {1, 1},
                                /* pads */ {0, 2, 1, 3});

  // Optimize current function and save.
  ::glow::optimize(F, CompilationMode::Infer);
  optF = F->clone(F->getName().str() + "_optimized");

  // Split node.
  auto splitOption = SplitNodeByNumChunks(splitDims, numChunks);
  std::vector<Node *> splitNodes;
  ASSIGN_VALUE_OR_FAIL_TEST(
      splitNodes, ::glow::splitNodeWithConstraints(maxpool, &splitOption, {}));
  runDCEPass(F, cctx);

  // Compute total number of chunks.
  dim_t totNumChunks = 1;
  for (auto numChunk : numChunks) {
    totNumChunks *= numChunk;
  }

  // Check node count.
  EXPECT_EQ(splitNodes.size(), totNumChunks);
  EXPECT_EQ(countNodeKind(F, Kinded::Kind::SliceNodeKind), totNumChunks);
  EXPECT_EQ(countNodeKind(F, Kinded::Kind::MaxPoolNodeKind), totNumChunks);
  EXPECT_EQ(countNodeKind(F, Kinded::Kind::InsertTensorNodeKind), totNumChunks);
  EXPECT_EQ(countNodeKind(F, Kinded::Kind::TouchNodeKind), 1);
}

/// Test splitting a MaxPool with padding along dimension N, H, W or C.
#define TEST_MAXPOOL_NONZEROPAD_SPLIT(splitDim, numChunks)                     \
  TEST_F(NodeSplitting,                                                        \
         MaxPool_NonZeroPad_Dim##splitDim##_Chunks##numChunks) {               \
    splitMaxPoolNonZeroPad(F_, optimizedF_, bindings_, cctx_,                  \
                           {ShapeNHWC::dim##splitDim}, {numChunks});           \
    checkNumericalEquivalence(0);                                              \
  }
TEST_MAXPOOL_NONZEROPAD_SPLIT(H, 2)
TEST_MAXPOOL_NONZEROPAD_SPLIT(W, 2)
#undef TEST_MAXPOOL_NONZEROPAD_SPLIT

/// Test splitting a MaxPool with padding along dimensions H, W.
TEST_F(NodeSplitting, MaxPool_NonZeroPad_DimHW_Chunks4) {
  splitMaxPoolNonZeroPad(F_, optimizedF_, bindings_, cctx_,
                         {ShapeNHWC::dimH, ShapeNHWC::dimW}, {2, 2});
  checkNumericalEquivalence(0);
}

/// Test splitting an "ill-defined" MaxPool for which not all the input
/// (including padding) is referenced by the output tensor. This happens
/// when using a stride larger than 1. This verifies that the node
/// splitting infrastructure uses a weaker verification of the mapping
/// between input and output for MaxPool.
TEST_F(NodeSplitting, MaxPool_IllDefined_DimHW) {
  std::vector<size_t> splitDims = {ShapeNHWC::dimH, ShapeNHWC::dimW};
  std::vector<dim_t> numChunks = {3, 3};
  Node *maxpool = createMaxPool(F_, bindings_,
                                /* inputDims */ {1, 16, 18, 1},
                                /* outputDims */ {1, 8, 9, 1},
                                /* kernels */ {2, 2},
                                /* strides */ {2, 2},
                                /* pads */ {1, 1, 0, 0});

  // Optimize current function and save.
  ::glow::optimize(F_, CompilationMode::Infer);
  optimizedF_ = F_->clone(F_->getName().str() + "_optimized");

  // Split node.
  auto splitOption = SplitNodeByNumChunks(splitDims, numChunks);
  std::vector<Node *> splitNodes;
  ASSIGN_VALUE_OR_FAIL_TEST(
      splitNodes, ::glow::splitNodeWithConstraints(maxpool, &splitOption, {}));
  runDCEPass(F_, cctx_);

  // Check node count.
  dim_t totNumChunks = numChunks[0] * numChunks[1];
  EXPECT_EQ(splitNodes.size(), totNumChunks);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::SliceNodeKind), totNumChunks);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::MaxPoolNodeKind), totNumChunks);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::InsertTensorNodeKind),
            totNumChunks);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::TouchNodeKind), 1);
}

/// Test that a MaxPool node is not split when the second output operand
/// Argmax has users.
TEST_F(NodeSplitting, MaxPool_Argmax_NoSplit) {
  std::vector<dim_t> inputDims = {1, 16, 18, 1};
  std::vector<dim_t> outputDims = {1, 8, 9, 1};
  std::vector<unsigned_t> kernels = {2, 2};
  std::vector<unsigned_t> strides = {2, 2};
  std::vector<unsigned_t> pads = {1, 1, 0, 0};

  // Create MaxPool with Argmax.
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

  // Optimize current function and save.
  ::glow::optimize(F_, CompilationMode::Infer);
  optimizedF_ = F_->clone(F_->getName().str() + "_optimized");

  // Split node.
  auto splitOption = SplitNodeByNumChunks({ShapeNHWC::dimH}, {3});
  std::vector<Node *> splitNodes;
  ASSIGN_VALUE_OR_FAIL_TEST(
      splitNodes, ::glow::splitNodeWithConstraints(maxpool, &splitOption, {}));
  runDCEPass(F_, cctx_);

  // Check node count.
  EXPECT_EQ(splitNodes.size(), 0);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::SliceNodeKind), 0);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::MaxPoolNodeKind), 1);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::InsertTensorNodeKind), 0);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::TouchNodeKind), 0);
}

///===---------------------------------------------------------------------===//
///                                   AvgPool
///===---------------------------------------------------------------------===//
/// Utility function to create a simple network with a single AvgPool node using
/// the function \p F and the bindings \p bindings.
static Node *createAvgPool(Function *F, PlaceholderBindings &bindings,
                           std::vector<dim_t> inputDims,
                           std::vector<dim_t> outputDims,
                           std::vector<unsigned_t> kernels,
                           std::vector<unsigned_t> strides,
                           std::vector<unsigned_t> pads) {

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
  std::vector<dim_t> actualOutputDims = avgpool->getResult().getType()->dims();
  EXPECT_EQ(actualOutputDims, outputDims);
  return avgpool;
}

/// Utility function to test splitting a basic AvgPool node along the dimensions
/// \p splitDims in the given number chunks \p numChunks. The split is done
/// implicitly relative to the AvgPool output operand.
static void splitAvgPoolBasic(Function *F, Function *&optF,
                              PlaceholderBindings &bindings,
                              CompilationContext &cctx,
                              std::vector<size_t> splitDims,
                              std::vector<dim_t> numChunks) {
  // Create basic AvgPool.
  Node *avgpool = createAvgPool(F, bindings,
                                /* inputDims */ {3, 7, 8, 4},
                                /* outputDims */ {3, 6, 7, 4},
                                /* kernels */ {2, 2},
                                /* strides */ {1, 1},
                                /* pads */ {0, 0, 0, 0});

  // Optimize current function and save.
  ::glow::optimize(F, CompilationMode::Infer);
  optF = F->clone(F->getName().str() + "_optimized");

  // Split node.
  auto splitOption = SplitNodeByNumChunks(splitDims, numChunks);
  std::vector<Node *> splitNodes;
  ASSIGN_VALUE_OR_FAIL_TEST(
      splitNodes, ::glow::splitNodeWithConstraints(avgpool, &splitOption, {}));
  runDCEPass(F, cctx);

  // Compute total number of chunks.
  dim_t totNumChunks = 1;
  for (auto numChunk : numChunks) {
    totNumChunks *= numChunk;
  }

  // Check node count.
  EXPECT_EQ(splitNodes.size(), totNumChunks);
  EXPECT_EQ(countNodeKind(F, Kinded::Kind::SliceNodeKind), totNumChunks);
  EXPECT_EQ(countNodeKind(F, Kinded::Kind::AvgPoolNodeKind), totNumChunks);
  EXPECT_EQ(countNodeKind(F, Kinded::Kind::InsertTensorNodeKind), totNumChunks);
  EXPECT_EQ(countNodeKind(F, Kinded::Kind::TouchNodeKind), 1);
}

/// Test splitting a AvgPool along dimension N, H, W or C.
#define TEST_AVGPOOL_BASIC_SPLIT(splitDim, numChunks)                          \
  TEST_F(NodeSplitting, AvgPool_Basic_Dim##splitDim##_Chunks##numChunks) {     \
    splitAvgPoolBasic(F_, optimizedF_, bindings_, cctx_,                       \
                      {ShapeNHWC::dim##splitDim}, {numChunks});                \
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
                                   std::vector<size_t> splitDims,
                                   std::vector<dim_t> numChunks) {
  // Create AvgPool with non-zero padding.
  Node *avgpool = createAvgPool(F, bindings,
                                /* inputDims */ {1, 4, 4, 1},
                                /* outputDims */ {1, 4, 8, 1},
                                /* kernels */ {2, 2},
                                /* strides */ {1, 1},
                                /* pads */ {0, 2, 1, 3});

  // Optimize current function and save.
  ::glow::optimize(F, CompilationMode::Infer);
  optF = F->clone(F->getName().str() + "_optimized");

  // Split node.
  auto splitOption = SplitNodeByNumChunks(splitDims, numChunks);
  std::vector<Node *> splitNodes;
  ASSIGN_VALUE_OR_FAIL_TEST(
      splitNodes, ::glow::splitNodeWithConstraints(avgpool, &splitOption, {}));
  runDCEPass(F, cctx);

  // Compute total number of chunks.
  dim_t totNumChunks = 1;
  for (auto numChunk : numChunks) {
    totNumChunks *= numChunk;
  }

  // Check node count.
  EXPECT_EQ(splitNodes.size(), totNumChunks);
  EXPECT_EQ(countNodeKind(F, Kinded::Kind::SliceNodeKind), totNumChunks);
  EXPECT_EQ(countNodeKind(F, Kinded::Kind::AvgPoolNodeKind), totNumChunks);
  EXPECT_EQ(countNodeKind(F, Kinded::Kind::InsertTensorNodeKind), totNumChunks);
  EXPECT_EQ(countNodeKind(F, Kinded::Kind::TouchNodeKind), 1);
}

/// Test splitting a AvgPool with padding along dimension N, H, W or C.
#define TEST_AVGPOOL_NONZEROPAD_SPLIT(splitDim, numChunks)                     \
  TEST_F(NodeSplitting,                                                        \
         AvgPool_NonZeroPad_Dim##splitDim##_Chunks##numChunks) {               \
    splitAvgPoolNonZeroPad(F_, optimizedF_, bindings_, cctx_,                  \
                           {ShapeNHWC::dim##splitDim}, {numChunks});           \
    checkNumericalEquivalence(0);                                              \
  }
TEST_AVGPOOL_NONZEROPAD_SPLIT(H, 2)
TEST_AVGPOOL_NONZEROPAD_SPLIT(W, 2)
#undef TEST_AVGPOOL_NONZEROPAD_SPLIT

/// Test splitting a AvgPool with padding along dimensions H, W.
TEST_F(NodeSplitting, AvgPool_NonZeroPad_DimHW_Chunks4) {
  splitAvgPoolNonZeroPad(F_, optimizedF_, bindings_, cctx_,
                         {ShapeNHWC::dimH, ShapeNHWC::dimW}, {2, 2});
  checkNumericalEquivalence(0);
}

/// Test splitting an "ill-defined" AvgPool for which not all the input
/// (including padding) is referenced by the output tensor. This happens
/// when using a stride larger than 1. This verifies that the node
/// splitting infrastructure uses a weaker verification of the mapping
/// between input and output for AvgPool.
TEST_F(NodeSplitting, AvgPool_IllDefined_DimHW) {
  std::vector<size_t> splitDims = {ShapeNHWC::dimH, ShapeNHWC::dimW};
  std::vector<dim_t> numChunks = {3, 3};
  Node *avgpool = createAvgPool(F_, bindings_,
                                /* inputDims */ {1, 16, 18, 1},
                                /* outputDims */ {1, 8, 9, 1},
                                /* kernels */ {2, 2},
                                /* strides */ {2, 2},
                                /* pads */ {1, 1, 0, 0});

  // Optimize current function and save.
  ::glow::optimize(F_, CompilationMode::Infer);
  optimizedF_ = F_->clone(F_->getName().str() + "_optimized");

  // Split node.
  auto splitOption = SplitNodeByNumChunks(splitDims, numChunks);
  std::vector<Node *> splitNodes;
  ASSIGN_VALUE_OR_FAIL_TEST(
      splitNodes, ::glow::splitNodeWithConstraints(avgpool, &splitOption, {}));
  runDCEPass(F_, cctx_);

  // Check node count.
  dim_t totNumChunks = numChunks[0] * numChunks[1];
  EXPECT_EQ(splitNodes.size(), totNumChunks);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::SliceNodeKind), totNumChunks);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::AvgPoolNodeKind), totNumChunks);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::InsertTensorNodeKind),
            totNumChunks);
  EXPECT_EQ(countNodeKind(F_, Kinded::Kind::TouchNodeKind), 1);
}
