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

#include "../../lib/Backends/Habana/Habana.h"
#include "../../lib/Backends/Habana/HabanaDeviceManager.h"
#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Graph/Graph.h"
#include "glow/Graph/PlaceholderBindings.h"
#include "glow/Optimizer/GraphOptimizer/GraphOptimizer.h"

#include "gtest/gtest.h"

#include <future>
#include <thread>
#include <vector>

using namespace glow;

class HabanaBackendTest : public ::testing::Test {
protected:
  HabanaBackendTest()
      : EE_("Habana"), mod_(EE_.getModule()), F_(mod_.createFunction("main")) {}
  ~HabanaBackendTest() = default;

  template <ElemKind kind, typename ElemTy> void testFCHelper();

  ExecutionEngine EE_;
  Module &mod_;
  Function *F_;
  PlaceholderBindings ctx_;
};

TEST_F(HabanaBackendTest, SurroundTile) {
  HabanaBackend backend;

  // Create a graph that looks like this:
  //   Placeholder
  //       |
  //       v
  //     Tile
  //       |
  //       v
  //     Save
  Placeholder *A = mod_.createPlaceholder(ElemKind::FloatTy, {1}, "A", false);

  TileNode *TN = F_->createTile("tile", A, /*tiles=*/5, /*axis=*/0);
  SaveNode *SN = F_->createSave("save", TN);

  // Invoke Habana backend specific graph optimisations.
  CompilationContext cctx;
  bool changed;
  ASSIGN_VALUE_OR_FAIL_TEST(changed, backend.transformPostLowering(F_, cctx));
  EXPECT_TRUE(changed);

  // Invoke dead code elimination.
  ::glow::optimize(F_, CompilationMode::Infer);

  // Now, the graph should look like this:
  //   Placeholder
  //       |
  //       v
  //   Reshape
  //       |
  //       v
  //     Tile
  //       |
  //       v
  //   Reshape
  //       |
  //       v
  //      Save

  auto *RO = llvm::dyn_cast<ReshapeNode>(SN->getInput());
  ASSERT_TRUE(RO);
  TN = llvm::dyn_cast<TileNode>(RO->getInput());
  ASSERT_TRUE(TN);
  auto *RI = llvm::dyn_cast<ReshapeNode>(TN->getInput());
  ASSERT_TRUE(RI);
  EXPECT_EQ(llvm::dyn_cast<Placeholder>(RI->getInput()), A);

  // Check that there four nodes.
  EXPECT_EQ(F_->getNodes().size(), 4);
}

TEST_F(HabanaBackendTest, DoNotSurroundTile) {
  HabanaBackend backend;

  // Create a graph that looks like this:
  //   Placeholder
  //       |
  //       v
  //     Tile
  //       |
  //       v
  //     Save
  Placeholder *A =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 1, 1, 1}, "A", false);

  TileNode *TN = F_->createTile("tile", A, /*tiles=*/5, /*axis=*/0);
  F_->createSave("save", TN);

  // Invoke Habana backend specific graph optimisations.
  CompilationContext cctx;
  bool changed;
  ASSIGN_VALUE_OR_FAIL_TEST(changed, backend.transformPostLowering(F_, cctx));

  // Graph should not change since input to Tile is already 4D.
  EXPECT_FALSE(changed);
}

TEST_F(HabanaBackendTest, FuseConvRelu) {
  HabanaBackend backend;

  // Create a graph that looks like this:
  //   Placeholder
  //       |
  //       v
  //     Conv
  //       |
  //       v
  //     Relu
  //       |
  //       v
  //     Save
  Placeholder *A =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 10, 20, 3}, "A", false);

  ConvolutionNode *CVN = F_->createConv(ctx_, "conv", A, 3, 5, 1, 2, 1);
  ReluNode *RN = F_->createRELU("relu", CVN);
  SaveNode *SN = F_->createSave("save", RN);

  // Invoke Habana backend specific graph optimisations.
  CompilationContext cctx;
  bool changed;
  ASSIGN_VALUE_OR_FAIL_TEST(changed, backend.transformPostLowering(F_, cctx));
  EXPECT_TRUE(changed);

  // Now, the graph should look like this:
  //   Placeholder ------
  //       |            |
  //       v            v
  //   HabanaConv       Conv
  //       |            |
  //       v            v
  //     Save         Relu

  // Check that HabanaConv feeds into the Save.
  auto *HCA = llvm::dyn_cast<HabanaConvolutionNode>(SN->getInput());
  ASSERT_TRUE(HCA);

  // Check that the inputs to the HabanaConvAdd are the same as the Conv
  // that it replaced (except filter, which feeds into a transpose).
  EXPECT_EQ(HCA->getInput(), CVN->getInput());
  EXPECT_EQ(HCA->getBias(), CVN->getBias());
  EXPECT_EQ(HCA->getKernels(), CVN->getKernels());
  EXPECT_EQ(HCA->getStrides(), CVN->getStrides());
  EXPECT_EQ(HCA->getPads(), CVN->getPads());
  EXPECT_EQ(HCA->getGroup(), CVN->getGroup());

  // Check that the doRelu parameter of the HabanaConvAdd is true.
  EXPECT_TRUE(HCA->getDoRelu());

  // Dead code elimination.
  ::glow::optimize(F_, CompilationMode::Infer);

  // Now, the graph should look like this:
  //   Placeholder
  //       |
  //       v
  //   HabanaConv
  //       |
  //       v
  //     Save

  // Check that the Placeholder that used to feed the removed Conv
  // now feeds into HabanaConvAdd.
  EXPECT_EQ(llvm::dyn_cast<Placeholder>(HCA->getInput()), A);

  // Check that there three nodes (there is a Transpose inserted during pre
  // lowering to arrange the filter values in the right order).
  EXPECT_EQ(F_->getNodes().size(), 3);
}

TEST_F(HabanaBackendTest, FuseConvAdd) {
  HabanaBackend backend;

  // Create a graph that looks like this:
  //   Placeholder       Placeholder
  //       |                 |
  //       v                 v
  //     Conv              Pool
  //       |                 |
  //       -----> Add <-------
  //               |
  //               v
  //             Save
  Placeholder *A =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 10, 20, 3}, "A", false);
  Placeholder *B =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 10, 20, 3}, "B", false);

  ConvolutionNode *CVN = F_->createConv(ctx_, "conv", A, 3, 5, 1, 2, 1);
  MaxPoolNode *PN = F_->createMaxPool("pool", B, 5, 1, 2);
  AddNode *AN = F_->createAdd("add", CVN, PN->getResult());
  SaveNode *SN = F_->createSave("save", AN);

  // Invoke Habana backend specific graph optimisations.
  CompilationContext cctx;
  bool changed;
  ASSIGN_VALUE_OR_FAIL_TEST(changed, backend.transformPostLowering(F_, cctx));
  EXPECT_TRUE(changed);

  // Now, the graph should look like this:
  //   Placeholder       Placeholder
  //   |   |                 |
  //   |   v                 v
  //   | Conv              Pool
  //   |   |                 |
  //   |   -----> Add <-------
  //   |                     |
  //   |                     |
  //   ---> HabanaConvAdd <-----
  //            |
  //            v
  //           Save

  // Check that HabanaConvAdd feeds into the Save.
  auto *HCA = llvm::dyn_cast<HabanaConvolutionAddNode>(SN->getInput());
  ASSERT_TRUE(HCA);

  // Check that the inputs to the HabanaConvAdd are the same as the Conv
  // that it replaced (except filter, which feeds into a transpose).
  EXPECT_EQ(HCA->getInput(), CVN->getInput());
  EXPECT_EQ(HCA->getBias(), CVN->getBias());
  EXPECT_EQ(HCA->getKernels(), CVN->getKernels());
  EXPECT_EQ(HCA->getStrides(), CVN->getStrides());
  EXPECT_EQ(HCA->getPads(), CVN->getPads());
  EXPECT_EQ(HCA->getGroup(), CVN->getGroup());

  // Check that the doRelu parameter of the HabanaConvAdd is false.
  EXPECT_FALSE(HCA->getDoRelu());

  // Check that the addend to the HabanaConvAdd is the MaxPoolNode.
  EXPECT_EQ(llvm::dyn_cast<MaxPoolNode>(HCA->getAddend()), PN);

  // Dead code elimination.
  ::glow::optimize(F_, CompilationMode::Infer);

  // Now, the graph should look like this:
  //   Placeholder       Placeholder
  //        |                 |
  //        |                 v
  //        |               Pool
  //        |                 |
  //        |                 |
  //        |                 |
  //        |                 |
  //        -> HabanaConvAdd <-
  //                |
  //                v
  //               Save

  // Check that the Placeholder that used to feed the removed Conv
  // now feeds into HabanaConvAdd.
  EXPECT_EQ(llvm::dyn_cast<Placeholder>(HCA->getInput()), A);

  // Check that there four nodes (there is a Transpose inserted during pre
  // lowering to arrange the filter values in the right order).
  EXPECT_EQ(F_->getNodes().size(), 4);
}

TEST_F(HabanaBackendTest, FuseConvAddRelu) {
  HabanaBackend backend;

  // Create a graph that looks like this:
  //   Placeholder       Placeholder
  //       |                 |
  //       v                 v
  //     Conv              Pool
  //       |                 |
  //       -----> Add <-------
  //               |
  //               v
  //             Relu
  //               |
  //               v
  //             Save
  Placeholder *A =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 10, 20, 3}, "A", false);
  Placeholder *B =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 10, 20, 3}, "B", false);

  ConvolutionNode *CVN = F_->createConv(ctx_, "conv", A, 3, 5, 1, 2, 1);
  MaxPoolNode *PN = F_->createMaxPool("pool", B, 5, 1, 2);
  AddNode *AN = F_->createAdd("add", CVN, PN->getResult());
  ReluNode *RN = F_->createRELU("relu", AN);
  SaveNode *SN = F_->createSave("save", RN);

  // Invoke Habana backend specific graph optimisations.
  CompilationContext cctx;
  bool changed;
  ASSIGN_VALUE_OR_FAIL_TEST(changed, backend.transformPostLowering(F_, cctx));
  EXPECT_TRUE(changed);

  // Now, the graph should look like this:
  //   Placeholder       Placeholder
  //   |   |                 |
  //   |   v                 v
  //   | Conv              Pool
  //   |   |                 |
  //   |   -----> Add <-------
  //   |                     |
  //   |                     |
  //   ---> HabanaConvAdd <---
  //   |         |           |
  //   |         v           |
  //   |        Relu         |
  //   |                     |
  //   ---> HabanaConvAdd <---
  //              |
  //              v
  //            Save
  //
  // Check that HabanaConvAdd feeds into the Save.
  auto *HCA = llvm::dyn_cast<HabanaConvolutionAddNode>(SN->getInput());
  ASSERT_TRUE(HCA);

  // Check that the inputs to the HabanaConvAdd are the same as the Conv
  // that it replaced (except filter, which feeds into a transpose).
  EXPECT_EQ(HCA->getInput(), CVN->getInput());
  EXPECT_EQ(HCA->getBias(), CVN->getBias());
  EXPECT_EQ(HCA->getKernels(), CVN->getKernels());
  EXPECT_EQ(HCA->getStrides(), CVN->getStrides());
  EXPECT_EQ(HCA->getPads(), CVN->getPads());
  EXPECT_EQ(HCA->getGroup(), CVN->getGroup());

  // Check that the doRelu parameter of the HabanaConvAdd is true.
  EXPECT_TRUE(HCA->getDoRelu());

  // Check that the addend to the HabanaConvAdd is the MaxPoolNode.
  EXPECT_EQ(llvm::dyn_cast<MaxPoolNode>(HCA->getAddend()), PN);

  // Dead code elimination.
  ::glow::optimize(F_, CompilationMode::Infer);

  // Now, the graph should look like this:
  //   Placeholder       Placeholder
  //        |                 |
  //        |                 v
  //        |               Pool
  //        |                 |
  //        |                 |
  //        |                 |
  //        |                 |
  //        -> HabanaConvAdd <-
  //                |
  //                v
  //               Save

  // Check that the Placeholder that used to feed the removed Conv
  // now feeds into HabanaConvAdd.
  EXPECT_EQ(llvm::dyn_cast<Placeholder>(HCA->getInput()), A);

  // Check that there four nodes (there is a Transpose inserted during pre
  // lowering to arrange the filter values in the right order).
  EXPECT_EQ(F_->getNodes().size(), 4);
}

TEST_F(HabanaBackendTest, SetDeviceMemory) {
  uint64_t defaultMemory = (7 << 20);
  auto configEmpty = glow::runtime::DeviceConfig("Habana");
  auto configFull = glow::runtime::DeviceConfig("Habana");
  configFull.setDeviceMemory(32768);
  // With no commandline or deviceConfig, the memory should be default 7 <<20.
  glow::runtime::HabanaDeviceManager device1(configEmpty, 1, 1);
  Error err1 = device1.init();
  EXPECT_EQ(defaultMemory * 1024, device1.getMaximumMemory());
  // With only deviceConfig, the memory should be set by deviceConfig.
  glow::runtime::HabanaDeviceManager device2(configFull, 1, 1);
  Error err2 = device2.init();
  EXPECT_EQ(32768, device2.getMaximumMemory());
}

TEST_F(HabanaBackendTest, ConvertFC) {
  HabanaBackend backend;
  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 16}, "input", false);
  auto *weight = mod_.createConstant(ElemKind::FloatTy, {16, 16}, "weight");
  auto *bias = mod_.createConstant(ElemKind::FloatTy, {16}, "bias");
  auto *FC = F_->createFullyConnected("fc", input, weight, bias);
  auto *save = F_->createSave("save", FC);
  CompilationContext cctx;
  bool changed;
  ASSIGN_VALUE_OR_FAIL_TEST(changed, backend.transformPostLowering(F_, cctx));
  EXPECT_TRUE(changed);
  ASSERT_TRUE(save);
  ASSERT_TRUE(llvm::isa<HabanaFullyConnectedNode>(save->getInput()));
}

TEST_F(HabanaBackendTest, ConvertConv) {
  HabanaBackend backend;

  Placeholder *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 10, 20, 3}, "input", false);
  ConvolutionNode *conv = F_->createConv(ctx_, "conv", input, 3, 5, 1, 2, 1);
  SaveNode *save = F_->createSave("save", conv);

  CompilationContext cctx;
  bool changed;
  ASSIGN_VALUE_OR_FAIL_TEST(changed, backend.transformPostLowering(F_, cctx));
  EXPECT_TRUE(changed);
  ASSERT_TRUE(save);
  ASSERT_TRUE(llvm::isa<HabanaConvolutionNode>(save->getInput()));
}

template <ElemKind kind, typename ElemTy>
void HabanaBackendTest::testFCHelper() {
  auto *input = mod_.createPlaceholder(kind, {2, 32}, 1.0, 0, "input", false);
  auto *weights =
      mod_.createPlaceholder(kind, {32, 32}, 1.0, 0, "weights", false);
  auto *bias =
      mod_.createPlaceholder(ElemKind::Int32QTy, {32}, 1.0, 0, "bias", false);
  auto *output = mod_.createPlaceholder(kind, {2, 32}, 1.0, 0, "output", false);

  auto *fc = F_->createFullyConnected("fc", input, weights, bias);
  F_->createSave("save", fc, output);

  ctx_.allocate(input)->getHandle<ElemTy>().clear(1);
  ctx_.allocate(weights)->getHandle<ElemTy>().clear(1);
  ctx_.allocate(bias)->getHandle<int32_t>().clear(0);
  Tensor *out = ctx_.allocate(output);

  glow::convertPlaceholdersToConstants(F_, ctx_, {input, output});
  EE_.compile(CompilationMode::Infer);
  EE_.run(ctx_);
  auto H = out->getHandle<ElemTy>();
  for (size_t i = 0; i < H.size(); i++) {
    EXPECT_EQ(H.raw(i), 32);
  }
}

TEST_F(HabanaBackendTest, FC) { testFCHelper<ElemKind::Int8QTy, int8_t>(); }

TEST_F(HabanaBackendTest, FC16) { testFCHelper<ElemKind::Int16QTy, int16_t>(); }

TEST_F(HabanaBackendTest, FCFP32) {
  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 32}, "input", false);
  auto *weights =
      mod_.createPlaceholder(ElemKind::FloatTy, {32, 33}, "weights", false);
  auto *bias = mod_.createPlaceholder(ElemKind::FloatTy, {33}, "bias", false);
  auto *output =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 33}, "output", false);

  auto *fc = F_->createFullyConnected("fc", input, weights, bias);
  F_->createSave("save", fc, output);

  ctx_.allocate(input)->getHandle<float>().clear(1.0);
  ctx_.allocate(weights)->getHandle<float>().clear(1.0);
  ctx_.allocate(bias)->getHandle<float>().clear(0.0);
  Tensor *out = ctx_.allocate(output);

  // Let weight and bias to be constants.
  glow::convertPlaceholdersToConstants(F_, ctx_, {input, output});
  EE_.compile(CompilationMode::Infer);
  EE_.run(ctx_);
  auto H = out->getHandle<float>();
  for (size_t i = 0; i < H.size(); i++) {
    EXPECT_EQ(H.raw(i), 32);
  }
}

TEST_F(HabanaBackendTest, Conv) {
  auto *input = mod_.createPlaceholder(ElemKind::Int8QTy, {1, 8, 8, 4}, 1.0, 0,
                                       "input", false);

  auto *filters = mod_.createPlaceholder(ElemKind::Int8QTy, {2, 2, 2, 4}, 1.0,
                                         0, "filters", false);
  auto *bias =
      mod_.createPlaceholder(ElemKind::Int32QTy, {2}, 1.0, 0, "bias", false);
  auto *output = mod_.createPlaceholder(ElemKind::Int8QTy, {1, 3, 3, 2}, 1.0, 0,
                                        "output", false);

  auto *conv = F_->createConv("conv", input, filters, bias, output->getType(),
                              {2, 2}, {3, 3}, {0, 0, 0, 0}, 1);
  F_->createSave("save", conv, output);

  ctx_.allocate(input)->getHandle<int8_t>().clear(1);
  ctx_.allocate(filters)->getHandle<int8_t>().clear(1);
  ctx_.allocate(bias)->getHandle<int32_t>().clear(4);
  Tensor *out = ctx_.allocate(output);

  glow::convertPlaceholdersToConstants(F_, ctx_, {input, output});
  EE_.compile(CompilationMode::Infer);
  EE_.run(ctx_);

  auto H = out->getHandle<int8_t>();
  for (size_t i = 0; i < H.size(); i++) {
    // {4,2,2} filters filled with 1 plus bias of 4 = 1*4*2*2 + 4 = 16 + 4 = 20
    EXPECT_EQ(H.raw(i), 20);
  }
}

TEST_F(HabanaBackendTest, MaxPool) {
  auto *input = mod_.createPlaceholder(ElemKind::Int8QTy, {1, 3, 3, 1}, 1.0, 0,
                                       "input", false);
  auto *output = mod_.createPlaceholder(ElemKind::Int8QTy, {1, 2, 2, 1}, 1.0, 0,
                                        "output", false);

  auto *pool = F_->createMaxPool("pool", input, {2, 2}, {1, 1}, {0, 0, 0, 0});
  F_->createSave("save", pool->getResult(), output);

  Tensor *in = ctx_.allocate(input);
  auto IH = in->getHandle<int8_t>();

  for (size_t i = 0; i < IH.size(); ++i) {
    IH.raw(i) = i;
  }

  Tensor *out = ctx_.allocate(output);
  auto OH = out->getHandle<int8_t>();

  glow::convertPlaceholdersToConstants(F_, ctx_, {input, output});
  EE_.compile(CompilationMode::Infer);
  EE_.run(ctx_);

  // The input looks like this:
  // -------------
  // | 0 | 1 | 2 |
  // -------------
  // | 3 | 4 | 5 |
  // -------------
  // | 6 | 7 | 8 |
  // -------------
  //
  // so with a {2,2} kernel, {1,1} strides, and no padding,
  // the output should be:
  // ---------
  // | 4 | 5 |
  // ---------
  // | 7 | 8 |
  // ---------
  EXPECT_EQ(OH.raw(0), 4);
  EXPECT_EQ(OH.raw(1), 5);
  EXPECT_EQ(OH.raw(2), 7);
  EXPECT_EQ(OH.raw(3), 8);
}

// The input and output are the same as the MaxPool test case above. The
// main purpose of this test case is to make sure that pooling parameters
// are passed correctly to the Synapse API when creating a pooling operator.
// The addition of a ReLU at the output is functionally a no-op, but tests
// that pooling parameters are not allocated on the stack and destroyed at
// scope exit when a pooling operator is added by attempting to clobber
// those pooling parameters on the stack with other local variables used
// while adding a ReLU operator using the Synapse API.
TEST_F(HabanaBackendTest, MaxPoolRelu) {
  auto *input = mod_.createPlaceholder(ElemKind::Int8QTy, {1, 3, 3, 1}, 1.0, 0,
                                       "input", false);
  auto *output = mod_.createPlaceholder(ElemKind::Int8QTy, {1, 2, 2, 1}, 1.0, 0,
                                        "output", false);

  auto *pool = F_->createMaxPool("pool", input, {2, 2}, {1, 1}, {0, 0, 0, 0});
  auto *relu = F_->createRELU("relu", pool->getResult());
  F_->createSave("save", relu, output);

  Tensor *in = ctx_.allocate(input);
  auto IH = in->getHandle<int8_t>();

  for (size_t i = 0; i < IH.size(); ++i) {
    IH.raw(i) = i;
  }

  Tensor *out = ctx_.allocate(output);
  auto OH = out->getHandle<int8_t>();

  glow::convertPlaceholdersToConstants(F_, ctx_, {input, output});
  EE_.compile(CompilationMode::Infer);
  EE_.run(ctx_);

  EXPECT_EQ(OH.raw(0), 4);
  EXPECT_EQ(OH.raw(1), 5);
  EXPECT_EQ(OH.raw(2), 7);
  EXPECT_EQ(OH.raw(3), 8);
}

TEST_F(HabanaBackendTest, AvgPool) {
  auto *input = mod_.createPlaceholder(ElemKind::Int8QTy, {1, 3, 3, 1}, 1.0, 0,
                                       "input", false);
  auto *output = mod_.createPlaceholder(ElemKind::Int8QTy, {1, 2, 2, 1}, 1.0, 0,
                                        "output", false);

  auto *pool = F_->createAvgPool("pool", input, {2, 2}, {1, 1}, {0, 0, 0, 0});
  F_->createSave("save", pool, output);

  Tensor *in = ctx_.allocate(input);
  auto IH = in->getHandle<int8_t>();

  for (size_t i = 0; i < IH.size(); ++i) {
    IH.raw(i) = i;
  }

  Tensor *out = ctx_.allocate(output);
  auto OH = out->getHandle<int8_t>();

  glow::convertPlaceholdersToConstants(F_, ctx_, {input, output});
  EE_.compile(CompilationMode::Infer);
  EE_.run(ctx_);

  // The input looks like this:
  // -------------
  // | 0 | 1 | 2 |
  // -------------
  // | 3 | 4 | 5 |
  // -------------
  // | 6 | 7 | 8 |
  // -------------
  //
  // so with a {2,2} kernel, {1,1} strides, and no padding,
  // the output should be:
  // ---------
  // | 2 | 3 |
  // ---------
  // | 7 | 8 |
  // ---------
  EXPECT_EQ(OH.raw(0), 2);
  EXPECT_EQ(OH.raw(1), 3);
  EXPECT_EQ(OH.raw(2), 5);
  EXPECT_EQ(OH.raw(3), 6);
}

// The input and output are the same as the AvgPool test case above. The
// main purpose of this test case is to make sure that pooling parameters
// are passed correctly to the Synapse API when creating a pooling operator.
// The addition of a ReLU at the output is functionally a no-op, but tests
// that pooling parameters are not allocated on the stack and destroyed at
// scope exit when a pooling operator is added by attempting to clobber
// those pooling parameters on the stack with other local variables used
// while adding a ReLU operator using the Synapse API.
TEST_F(HabanaBackendTest, AvgPoolRelu) {
  auto *input = mod_.createPlaceholder(ElemKind::Int8QTy, {1, 3, 3, 1}, 1.0, 0,
                                       "input", false);
  auto *output = mod_.createPlaceholder(ElemKind::Int8QTy, {1, 2, 2, 1}, 1.0, 0,
                                        "output", false);

  auto *pool = F_->createAvgPool("pool", input, {2, 2}, {1, 1}, {0, 0, 0, 0});
  auto *relu = F_->createRELU("relu", pool);
  F_->createSave("save", relu, output);

  Tensor *in = ctx_.allocate(input);
  auto IH = in->getHandle<int8_t>();

  for (size_t i = 0; i < IH.size(); ++i) {
    IH.raw(i) = i;
  }

  Tensor *out = ctx_.allocate(output);
  auto OH = out->getHandle<int8_t>();

  glow::convertPlaceholdersToConstants(F_, ctx_, {input, output});
  EE_.compile(CompilationMode::Infer);
  EE_.run(ctx_);

  EXPECT_EQ(OH.raw(0), 2);
  EXPECT_EQ(OH.raw(1), 3);
  EXPECT_EQ(OH.raw(2), 5);
  EXPECT_EQ(OH.raw(3), 6);
}

TEST_F(HabanaBackendTest, Transpose) {
  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 2, 4, 8}, "input", false);
  auto *output =
      mod_.createPlaceholder(ElemKind::FloatTy, {8, 1, 4, 2}, "output", false);

  auto *pool = F_->createTranspose("transpose", input, {3, 0, 2, 1});
  F_->createSave("save", pool, output);

  Tensor *in = ctx_.allocate(input);
  auto IH = in->getHandle<>();

  for (size_t i = 0; i < IH.size(); ++i) {
    IH.raw(i) = i;
  }

  Tensor *out = ctx_.allocate(output);
  auto OH = out->getHandle<>();

  glow::convertPlaceholdersToConstants(F_, ctx_, {input, output});
  EE_.compile(CompilationMode::Infer);
  EE_.run(ctx_);

  // Since the shuffle for the transpose is {3, 0, 2 ,1}, it should be true that
  // input(i, j, k, l) = output(l, i, k, j) for all i, j, k, l.
  for (size_t i = 0, ie = IH.dims()[0]; i != ie; ++i) {
    for (size_t j = 0, je = IH.dims()[1]; j != je; ++j) {
      for (size_t k = 0, ke = IH.dims()[2]; k != ke; ++k) {
        for (size_t l = 0, le = IH.dims()[3]; l != le; ++l) {
          EXPECT_EQ(IH.at({i, j, k, l}), OH.at({l, i, k, j}));
        }
      }
    }
  }
}

// The input and output are the same as the Transpose test case above. The
// main purpose of this test case is to make sure that transpose parameters
// are passed correctly to the Synapse API when creating a transpose operator.
// The addition of a ReLU at the output is functionally a no-op, but tests
// that transpose parameters are not allocated on the stack and destroyed at
// scope exit when a transpose operator is added by attempting to clobber
// those transpose parameters on the stack with other local variables used
// while adding a ReLU operator using the Synapse API.
TEST_F(HabanaBackendTest, TransposeRelu) {
  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 2, 4, 8}, "input", false);
  auto *output =
      mod_.createPlaceholder(ElemKind::FloatTy, {8, 1, 4, 2}, "output", false);

  auto *pool = F_->createTranspose("transpose", input, {3, 0, 2, 1});
  auto *relu = F_->createRELU("relu", pool);
  F_->createSave("save", relu, output);

  Tensor *in = ctx_.allocate(input);
  auto IH = in->getHandle<>();

  for (size_t i = 0; i < IH.size(); ++i) {
    IH.raw(i) = i;
  }

  Tensor *out = ctx_.allocate(output);
  auto OH = out->getHandle<>();

  glow::convertPlaceholdersToConstants(F_, ctx_, {input, output});
  EE_.compile(CompilationMode::Infer);
  EE_.run(ctx_);

  // Since the shuffle for the transpose is {3, 0, 2 ,1}, it should be true that
  // input(i, j, k, l) = output(l, i, k, j) for all i, j, k, l.
  for (size_t i = 0, ie = IH.dims()[0]; i != ie; ++i) {
    for (size_t j = 0, je = IH.dims()[1]; j != je; ++j) {
      for (size_t k = 0, ke = IH.dims()[2]; k != ke; ++k) {
        for (size_t l = 0, le = IH.dims()[3]; l != le; ++l) {
          EXPECT_EQ(IH.at({i, j, k, l}), OH.at({l, i, k, j}));
        }
      }
    }
  }
}

TEST_F(HabanaBackendTest, QuantizedFC) {
  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 32}, "input", false);
  auto *weights =
      mod_.createPlaceholder(ElemKind::FloatTy, {32, 32}, "weights", false);
  auto *bias = mod_.createPlaceholder(ElemKind::FloatTy, {32}, "bias", false);
  auto *output =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 32}, "output", false);

  auto *qInput = F_->createQuantize(
      "qInput", input, mod_.uniqueType(ElemKind::Int8QTy, {2, 32}, 1.0, 0));
  auto *qWeights =
      F_->createQuantize("qWeights", weights,
                         mod_.uniqueType(ElemKind::Int8QTy, {32, 32}, 1.0, 0));
  auto *qBias = F_->createQuantize(
      "qBias", bias, mod_.uniqueType(ElemKind::Int32QTy, {32}, 1.0, 0));
  auto *fc = F_->createFullyConnected(
      "fc", qInput, qWeights, qBias,
      mod_.uniqueType(ElemKind::Int8QTy, {2, 32}, 1.0, 0));
  auto *dq = F_->createDequantize("dq", fc, ElemKind::FloatTy);
  F_->createSave("save", dq, output);

  ctx_.allocate(input)->getHandle<float>().clear(1);
  ctx_.allocate(weights)->getHandle<float>().clear(1);
  ctx_.allocate(bias)->getHandle<float>().clear(0);
  Tensor *out = ctx_.allocate(output);

  glow::convertPlaceholdersToConstants(F_, ctx_, {input, output});
  EE_.compile(CompilationMode::Infer);
  EE_.run(ctx_);

  auto H = out->getHandle<float>();
  for (size_t i = 0; i < H.size(); i++) {
    EXPECT_EQ(H.raw(i), 32);
  }
}

TEST_F(HabanaBackendTest, QuantizedNonZeroOffset) {
  TypeRef resTy = mod_.uniqueType(ElemKind::Int8QTy, {2, 2}, 0.08, 4);
  TypeRef lhsTy = mod_.uniqueType(ElemKind::Int8QTy, {2, 2}, 0.075, -5);
  TypeRef rhsTy = mod_.uniqueType(ElemKind::Int8QTy, {2, 2}, 0.075, -5);

  auto *lhs = mod_.createPlaceholder(ElemKind::FloatTy, {2, 2}, "lhs", false);
  auto *rhs = mod_.createPlaceholder(ElemKind::FloatTy, {2, 2}, "rhs", false);

  ctx_.allocate(lhs)->getHandle() = {1.0f, 2.0f, 3.0f, 4.0f};

  ctx_.allocate(rhs)->getHandle() = {0.1f, -0.2f, 0.3f, 9.0f};

  auto *lhsq = F_->createQuantize("lhs.q", lhs, lhsTy);
  auto *rhsq = F_->createQuantize("rhs.q", rhs, rhsTy);

  auto *matmulq = F_->createSub("sub.q", resTy, lhsq, rhsq);

  auto *rq = F_->createDequantize("dequant", matmulq, ElemKind::FloatTy);

  auto *result = F_->createSave("save", rq);
  ctx_.allocate(result->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(ctx_);

  auto H = ctx_.get(result->getPlaceholder())->getHandle();
  EXPECT_NEAR(H.at({0, 0}), 0.9f, 0.1);
  EXPECT_NEAR(H.at({0, 1}), 2.2f, 0.1);
  EXPECT_NEAR(H.at({1, 0}), 2.7f, 0.1);
  EXPECT_NEAR(H.at({1, 1}), -5.0f, 0.1);
}

TEST_F(HabanaBackendTest, FC2) {
  constexpr unsigned batch = 2;
  constexpr unsigned inputs = 32;
  constexpr unsigned outputs = 32;
  auto *input = mod_.createPlaceholder(ElemKind::Int8QTy, {batch, inputs}, 0.1,
                                       0, "input", false);
  auto *weights = mod_.createPlaceholder(ElemKind::Int8QTy, {inputs, outputs},
                                         0.1, 0, "weights", false);
  auto *bias = mod_.createPlaceholder(ElemKind::Int32QTy, {outputs}, 0.01, 0,
                                      "bias", false);
  auto *output = mod_.createPlaceholder(ElemKind::Int8QTy, {batch, outputs},
                                        0.01, 0, "output", false);

  auto *fc =
      F_->createFullyConnected("fc", input, weights, bias, output->getType());
  F_->createSave("save", fc, output);

  ctx_.allocate(input)->getHandle<int8_t>().clear(1);
  ctx_.allocate(weights)->getHandle<int8_t>().clear(1);
  ctx_.allocate(bias)->getHandle<int32_t>().clear(1);
  Tensor *out = ctx_.allocate(output);

  glow::convertPlaceholdersToConstants(F_, ctx_, /*except=*/{input, output});
  EE_.compile(CompilationMode::Infer);
  EE_.run(ctx_);

  auto H = out->getHandle<int8_t>();
  for (size_t i = 0; i < H.size(); i++) {
    EXPECT_EQ(H.raw(i), 33);
  }
}

TEST_F(HabanaBackendTest, MLP) {
  constexpr unsigned B = 2;
  constexpr unsigned K = 32;

  auto *input =
      mod_.createPlaceholder(ElemKind::Int8QTy, {B, K}, 0.5, 0, "input", false);
  auto *output = mod_.createPlaceholder(ElemKind::Int8QTy, {B, K}, 0.25, 0,
                                        "output", false);

  auto *w1 =
      mod_.createPlaceholder(ElemKind::Int8QTy, {K, K}, 0.5, 0, "w1", false);
  auto *b1 =
      mod_.createPlaceholder(ElemKind::Int32QTy, {K}, 0.25, 0, "b1", false);
  auto *fc1 = F_->createFullyConnected(
      "fc1", input, w1, b1,
      mod_.uniqueType(ElemKind::Int8QTy, {B, K}, 0.25, 0));
  auto *relu = F_->createRELU("relu", fc1);
  auto *w2 =
      mod_.createPlaceholder(ElemKind::Int8QTy, {K, K}, 0.25, 0, "w2", false);
  auto *b2 =
      mod_.createPlaceholder(ElemKind::Int32QTy, {K}, 0.0625, 0, "b2", false);
  auto *fc2 = F_->createFullyConnected("fc2", relu, w2, b2, output->getType());

  F_->createSave("save", fc2, output);

  ctx_.allocate(input)->getHandle<int8_t>().clear(1);
  ctx_.allocate(w1)->getHandle<int8_t>().clear(1);
  ctx_.allocate(b1)->getHandle<int32_t>().clear(-31);
  ctx_.allocate(w2)->getHandle<int8_t>().clear(1);
  ctx_.allocate(b2)->getHandle<int32_t>().clear(0);
  Tensor *out = ctx_.allocate(output);

  glow::convertPlaceholdersToConstants(F_, ctx_, {input, output});
  EE_.compile(CompilationMode::Infer);
  EE_.run(ctx_);

  auto H = out->getHandle<int8_t>();
  for (size_t i = 0; i < H.size(); i++) {
    EXPECT_EQ(H.raw(i), 8);
  }
}

TEST_F(HabanaBackendTest, MatMul) {
  constexpr unsigned B = 4;
  constexpr unsigned K = 4;
  auto *input =
      mod_.createPlaceholder(ElemKind::Int8QTy, {B, K}, 1.0, 0, "input", false);
  auto *weights = mod_.createPlaceholder(ElemKind::Int8QTy, {K, K}, 1.0, 0,
                                         "weights", false);
  auto *output = mod_.createPlaceholder(ElemKind::Int8QTy, {B, K}, 1.0, 0,
                                        "output", false);
  auto *mm = F_->createMatMul("mm", input, weights);
  F_->createSave("save", mm, output);
  ctx_.allocate(input)->getHandle<int8_t>() = {
      0, 1, 2, 3, 1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6,
  };
  ctx_.allocate(weights)->getHandle<int8_t>() = {
      6, 5, 4, 3, 5, 4, 3, 2, 4, 3, 2, 1, 3, 2, 1, 0,
  };
  Tensor *out = ctx_.allocate(output);
  glow::convertPlaceholdersToConstants(F_, ctx_, /*except=*/{input, output});
  EE_.compile(CompilationMode::Infer);
  EE_.run(ctx_);
  Tensor expected(ElemKind::Int8QTy, {B, K}, 1.0, 0);
  expected.getHandle<int8_t>() = {
      22, 16, 10, 4, 40, 30, 20, 10, 58, 44, 30, 16, 76, 58, 40, 22,
  };
  EXPECT_TRUE(out->isEqual(expected));
}

TEST_F(HabanaBackendTest, MatMulFp32) {
  constexpr unsigned B = 2;
  constexpr unsigned K = 2;
  auto *lhs = mod_.createPlaceholder(ElemKind::FloatTy, {B, K}, "lhs", false);
  auto *rhs = mod_.createPlaceholder(ElemKind::FloatTy, {K, K}, "rhs", false);
  auto *output =
      mod_.createPlaceholder(ElemKind::FloatTy, {B, K}, "output", false);
  auto *mm = F_->createMatMul("mm", lhs, rhs);
  F_->createSave("save", mm, output);
  ctx_.allocate(lhs)->getHandle<float>() = {1, 2, 3, 4};
  ctx_.allocate(rhs)->getHandle<float>() = {1, 2, 3, 4};
  Tensor *out = ctx_.allocate(output);
  glow::convertPlaceholdersToConstants(F_, ctx_, /*except=*/{lhs, output});
  EE_.compile(CompilationMode::Infer);
  EE_.run(ctx_);
  Tensor expected(ElemKind::FloatTy, {B, K});
  expected.getHandle<float>() = {7, 10, 15, 22};
  EXPECT_TRUE(out->isEqual(expected));
}

TEST_F(HabanaBackendTest, MatMulFp32NonStaticB) {
  constexpr unsigned B = 2;
  constexpr unsigned K = 2;
  auto *lhs = mod_.createPlaceholder(ElemKind::FloatTy, {B, K}, "lhs", false);
  auto *rhs = mod_.createPlaceholder(ElemKind::FloatTy, {K, K}, "rhs", false);
  auto *output =
      mod_.createPlaceholder(ElemKind::FloatTy, {B, K}, "output", false);
  auto *mm = F_->createMatMul("mm", lhs, rhs);
  F_->createSave("save", mm, output);
  ctx_.allocate(lhs)->getHandle<float>() = {1, 2, 3, 4};
  ctx_.allocate(rhs)->getHandle<float>() = {1, 2, 3, 4};
  Tensor *out = ctx_.allocate(output);
  EE_.compile(CompilationMode::Infer);
  EE_.run(ctx_);
  Tensor expected(ElemKind::FloatTy, {B, K});
  expected.getHandle<float>() = {7, 10, 15, 22};
  EXPECT_TRUE(out->isEqual(expected));
}

TEST_F(HabanaBackendTest, FCx2) {
  auto *i1 =
      mod_.createPlaceholder(ElemKind::Int8QTy, {2, 2}, 1.0, 0, "i1", false);
  auto *w1 =
      mod_.createPlaceholder(ElemKind::Int8QTy, {2, 2}, 1.0, 0, "w1", false);
  auto *b1 =
      mod_.createPlaceholder(ElemKind::Int32QTy, {2}, 1.0, 0, "b1", false);
  auto *o1 =
      mod_.createPlaceholder(ElemKind::Int8QTy, {2, 2}, 1.0, 0, "o1", false);

  auto *i2 =
      mod_.createPlaceholder(ElemKind::Int8QTy, {2, 2}, 1.0, 0, "i2", false);
  auto *w2 =
      mod_.createPlaceholder(ElemKind::Int8QTy, {2, 2}, 1.0, 0, "w2", false);
  auto *b2 =
      mod_.createPlaceholder(ElemKind::Int32QTy, {2}, 1.0, 0, "b2", false);
  auto *o2 =
      mod_.createPlaceholder(ElemKind::Int8QTy, {2, 2}, 1.0, 0, "o2", false);

  auto *fc1 = F_->createFullyConnected("fc1", i1, w1, b1, o1->getType());
  F_->createSave("s1", fc1, o1);

  auto *fc2 = F_->createFullyConnected("fc2", i2, w2, b2, o2->getType());
  F_->createSave("s2", fc2, o2);

  ctx_.allocate(w1)->getHandle<int8_t>() = {1, 2, 3, 4};
  ctx_.allocate(w2)->getHandle<int8_t>() = {5, 6, 7, 8};
  ctx_.allocate(b1)->getHandle<int32_t>().clear(0);
  ctx_.allocate(b2)->getHandle<int32_t>().clear(0);
  glow::convertPlaceholdersToConstants(F_, ctx_, {});

  ctx_.allocate(i1)->getHandle<int8_t>() = {8, 7, 6, 5};
  ctx_.allocate(i2)->getHandle<int8_t>() = {4, 3, 2, 1};
  auto *ot1 = ctx_.allocate(o1);
  auto *ot2 = ctx_.allocate(o2);

  EE_.compile(CompilationMode::Infer);
  EE_.run(ctx_);

  Tensor e1(ElemKind::Int8QTy, {2, 2}, 1.0, 0);
  Tensor e2(ElemKind::Int8QTy, {2, 2}, 1.0, 0);
  e1.getHandle<int8_t>() = {29, 44, 21, 32};
  e2.getHandle<int8_t>() = {41, 48, 17, 20};

  EXPECT_TRUE(ot1->isEqual(e1));
  EXPECT_TRUE(ot2->isEqual(e2));
}

TEST_F(HabanaBackendTest, Sigmoid) {
  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 2}, "input", false);
  auto *output =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 2}, "output", false);

  auto inputHandle = ctx_.allocate(input)->getHandle();
  auto outputHandle = ctx_.allocate(output)->getHandle();

  auto *sigmoid = F_->createSigmoid("sigmoid", input);
  F_->createSave("save", sigmoid, output);

  inputHandle.randomize(-1.0f, 1.0f, mod_.getPRNG());

  EE_.compile(CompilationMode::Infer);
  EE_.run(ctx_);

  for (size_t i = 0; i < inputHandle.size(); ++i) {
    float expected = 1.0 / (1.0 + std::exp(-inputHandle.raw(i)));
    EXPECT_EQ(expected, outputHandle.raw(i));
  }
}

TEST_F(HabanaBackendTest, Reshape) {
  auto *input = mod_.createPlaceholder(ElemKind::Int8QTy, {2, 3, 4}, 1.0, 0,
                                       "input", false);
  auto *output = mod_.createPlaceholder(ElemKind::Int8QTy, {1, 2, 12}, 1.0, 0,
                                        "output", false);

  auto inputHandle = ctx_.allocate(input)->getHandle<int8_t>();
  auto outputHandle = ctx_.allocate(output)->getHandle<int8_t>();

  auto reshape = F_->createReshape("reshape", input, output->dims());
  F_->createSave("save", reshape, output);

  for (size_t i = 0; i < inputHandle.size(); ++i) {
    inputHandle.raw(i) = i;
  }

  EE_.compile(CompilationMode::Infer);
  EE_.run(ctx_);

  for (size_t i = 0; i < inputHandle.size(); ++i) {
    EXPECT_EQ(outputHandle.raw(i), i);
  }
}

TEST_F(HabanaBackendTest, Tanh) {
  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 3, 4}, "input", false);
  auto *output =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 3, 4}, "output", false);

  auto inputHandle = ctx_.allocate(input)->getHandle();
  auto outputHandle = ctx_.allocate(output)->getHandle();

  auto tanh = F_->createTanh("tanh", input);
  F_->createSave("save", tanh, output);

  inputHandle.randomize(-1.0f, 1.0f, mod_.getPRNG());

  EE_.compile(CompilationMode::Infer);
  EE_.run(ctx_);

  for (size_t i = 0; i < inputHandle.size(); ++i) {
    EXPECT_FLOAT_EQ(outputHandle.raw(i), std::tanh(inputHandle.raw(i)));
  }
}

TEST_F(HabanaBackendTest, Logit) {
  constexpr float eps = 0.3;
  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 3, 4}, "input", false);
  auto *output =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 3, 4}, "output", false);

  auto inputHandle = ctx_.allocate(input)->getHandle();
  auto outputHandle = ctx_.allocate(output)->getHandle();

  auto logit = F_->createLogit("logit", input, eps);
  F_->createSave("save", logit, output);

  inputHandle.randomize(-1.0f, 1.0f, mod_.getPRNG());

  EE_.compile(CompilationMode::Infer);
  EE_.run(ctx_);

  for (size_t i = 0; i < inputHandle.size(); ++i) {
    float input = inputHandle.raw(i);
    float clampedInput = std::min(std::max(input, eps), 1 - eps);
    float expOutput = std::log(clampedInput / (1 - clampedInput));

    EXPECT_FLOAT_EQ(outputHandle.raw(i), expOutput);
  }
}

TEST_F(HabanaBackendTest, BatchedAdd) {
  auto *batch =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 3, 3}, "batch", false);
  auto *added =
      mod_.createPlaceholder(ElemKind::FloatTy, {3, 3}, "added", false);

  ctx_.allocate(batch)->getHandle() = {9, 8, 7, 6, 5,  4,  3,  4,  5,
                                       6, 7, 8, 9, 10, 11, 12, 13, 14};
  ctx_.allocate(added)->getHandle().clear(1.0);

  auto *R = F_->createBatchedAdd("batch.add", batch, added);
  auto *save = F_->createSave("save", R);
  auto *result = ctx_.allocate(save->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(ctx_);

  auto BH = ctx_.get(batch)->getHandle();
  auto RH = result->getHandle();
  for (size_t i = 0; i < 2; i++) {
    for (size_t j = 0; j < 3; j++) {
      for (size_t k = 0; k < 3; k++) {
        EXPECT_NEAR(RH.at({i, j, k}), BH.at({i, j, k}) + 1.0, 0.001);
      }
    }
  }
}

TEST_F(HabanaBackendTest, Slice) {
  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 3, 4}, "input", false);
  auto *output =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 1, 1}, "output", false);

  auto inputHandle = ctx_.allocate(input)->getHandle();
  auto outputHandle = ctx_.allocate(output)->getHandle();

  auto slice = F_->createSlice("slice", input, {0, 1, 2}, {1, 2, 3});
  F_->createSave("save", slice, output);

  inputHandle.randomize(-1.0f, 1.0f, mod_.getPRNG());

  EE_.compile(CompilationMode::Infer);
  EE_.run(ctx_);

  EXPECT_FLOAT_EQ(outputHandle.at({0, 0, 0}), inputHandle.at({0, 1, 2}));
}

TEST_F(HabanaBackendTest, Flatten) {
  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 3, 4}, "input", false);
  auto *output =
      mod_.createPlaceholder(ElemKind::FloatTy, {6, 4}, "output", false);

  auto inputHandle = ctx_.allocate(input)->getHandle();
  auto outputHandle = ctx_.allocate(output)->getHandle();

  auto flatten = F_->createFlatten("flatten", input, /*axis=*/2);
  F_->createSave("save", flatten, output);

  inputHandle.randomize(-1.0f, 1.0f, mod_.getPRNG());

  EE_.compile(CompilationMode::Infer);
  EE_.run(ctx_);

  for (size_t i = 0; i < inputHandle.size(); ++i) {
    EXPECT_FLOAT_EQ(outputHandle.raw(i), inputHandle.raw(i));
  }
}

TEST_F(HabanaBackendTest, Broadcast) {
  auto *input =
      mod_.createPlaceholder(ElemKind::Int8QTy, {1}, 1.0, 0, "input", false);
  auto *output =
      mod_.createPlaceholder(ElemKind::Int8QTy, {5}, 1.0, 0, "output", false);

  auto inputHandle = ctx_.allocate(input)->getHandle<int8_t>();
  auto outputHandle = ctx_.allocate(output)->getHandle<int8_t>();

  auto broadcast = F_->createBroadcast("broadcast", input, {5}, 0);
  F_->createSave("save", broadcast, output);

  inputHandle.clear(42);

  EE_.compile(CompilationMode::Infer);
  EE_.run(ctx_);

  for (size_t i = 0; i < outputHandle.size(); ++i) {
    EXPECT_FLOAT_EQ(outputHandle.raw(i), 42) << " for index " << i << "\n";
  }
}

TEST_F(HabanaBackendTest, Add) {
  auto *lhs =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 3, 4}, "lhs", false);
  auto *rhs =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 3, 4}, "rhs", false);

  auto *output =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 3, 4}, "output", false);

  auto lhsHandle = ctx_.allocate(lhs)->getHandle();
  auto rhsHandle = ctx_.allocate(rhs)->getHandle();
  auto outHandle = ctx_.allocate(output)->getHandle();

  auto add = F_->createAdd("add", rhs, lhs);
  F_->createSave("save", add, output);

  lhsHandle.randomize(-3.0f, 3.0f, mod_.getPRNG());
  rhsHandle.randomize(-3.0f, 3.0f, mod_.getPRNG());

  EE_.compile(CompilationMode::Infer);
  EE_.run(ctx_);

  for (size_t i = 0; i < lhsHandle.size(); ++i) {
    EXPECT_FLOAT_EQ(outHandle.raw(i), lhsHandle.raw(i) + rhsHandle.raw(i));
  }
}

TEST_F(HabanaBackendTest, BroadcastAdd) {
  auto *lhs =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 3, 1}, "lhs", false);
  auto *rhs = mod_.createPlaceholder(ElemKind::FloatTy, {1}, "rhs", false);

  auto *output =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 3, 1}, "output", false);

  auto lhsHandle = ctx_.allocate(lhs)->getHandle();
  auto rhsHandle = ctx_.allocate(rhs)->getHandle();
  auto outHandle = ctx_.allocate(output)->getHandle();

  auto broadcast = F_->createBroadcast("broadcast", rhs, {2, 3, 1}, 0);
  auto add = F_->createAdd("add", broadcast, lhs);
  F_->createSave("save", add, output);

  lhsHandle.randomize(-3.0f, 3.0f, mod_.getPRNG());
  rhsHandle.randomize(-3.0f, 3.0f, mod_.getPRNG());

  EE_.compile(CompilationMode::Infer);
  EE_.run(ctx_);

  float rhsValue = rhsHandle.raw(0);
  for (size_t i = 0; i < lhsHandle.size(); ++i) {
    EXPECT_FLOAT_EQ(outHandle.raw(i), lhsHandle.raw(i) + rhsValue);
  }
}

TEST_F(HabanaBackendTest, Clip) {
  constexpr float kMin = -1.0f;
  constexpr float kMax = 1.0f;

  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 3, 4}, "input", false);
  auto *output =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 3, 4}, "output", false);

  auto inputHandle = ctx_.allocate(input)->getHandle();
  auto outputHandle = ctx_.allocate(output)->getHandle();

  auto clip = F_->createClip("clip", input, kMin, kMax);
  F_->createSave("save", clip, output);

  inputHandle.randomize(-3.0f, 3.0f, mod_.getPRNG());

  EE_.compile(CompilationMode::Infer);
  EE_.run(ctx_);

  for (size_t i = 0; i < inputHandle.size(); ++i) {
    float in = inputHandle.raw(i);
    float out = outputHandle.raw(i);
    float expOut = std::min(std::max(in, kMin), kMax);

    EXPECT_EQ(out, expOut);
  }
}

static void fill(Tensor &T, int val) {
  auto H = T.getHandle();
  std::iota(H.begin(), H.end(), val);
}

TEST_F(HabanaBackendTest, Copy) {
  Tensor cref(ElemKind::FloatTy, {20});
  Tensor c2ref(ElemKind::FloatTy, {20});
  fill(cref, 1);
  fill(c2ref, 21);

  auto *c = mod_.createConstant(ElemKind::FloatTy, {20}, "c");
  auto *p = mod_.createPlaceholder(ElemKind::FloatTy, {20}, "p", false);
  F_->createSave("s", c, p);
  auto &ct = c->getPayloadMutable();
  auto *pt = ctx_.allocate(p);
  ct.assign(&cref);

  auto *c2 = mod_.createConstant(ElemKind::FloatTy, {20}, "c2");
  auto *p2 = mod_.createPlaceholder(ElemKind::FloatTy, {20}, "p2", false);
  F_->createSave("s2", c2, p2);
  auto &c2t = c2->getPayloadMutable();
  auto *p2t = ctx_.allocate(p2);
  c2t.assign(&c2ref);

  EE_.compile(CompilationMode::Infer);
  EE_.run(ctx_);

  ASSERT_TRUE(cref.isEqual(*pt));
  ASSERT_TRUE(c2ref.isEqual(*p2t));
}

TEST_F(HabanaBackendTest, CopyPlaceholder) {
  auto *in = mod_.createPlaceholder(ElemKind::FloatTy, {64, 1}, "in", false);
  auto *save = F_->createSave("save", in);
  EE_.compile(CompilationMode::Infer);
  ctx_.allocate(in)->getHandle().clear(3.0);
  Tensor *out = ctx_.allocate(save->getPlaceholder());
  EE_.run(ctx_);
  for (auto e : out->getHandle()) {
    ASSERT_EQ(e, 3.0);
  }
}

TEST_F(HabanaBackendTest, ReluQuantized) {
  constexpr size_t size = 16;
  auto *input =
      mod_.createPlaceholder(ElemKind::Int8QTy, {size}, 1.0, 0, "input", false);
  auto *output = mod_.createPlaceholder(ElemKind::Int8QTy, {size}, 1.0, 0,
                                        "output", false);
  auto *relu = F_->createRELU("relu", input);
  F_->createSave("save", relu, output);
  auto IH = ctx_.allocate(input)->getHandle<int8_t>();
  auto OH = ctx_.allocate(output)->getHandle<int8_t>();
  for (size_t i = 0; i < size; i++) {
    IH.at({i}) = i - size / 2;
  }
  EE_.compile(CompilationMode::Infer);
  EE_.run(ctx_);
  for (size_t i = 0; i < size; i++) {
    EXPECT_EQ(OH.at({i}), std::max<int8_t>(0, i - size / 2));
  }
}

TEST_F(HabanaBackendTest, Concat) {
  auto *i1 = mod_.createPlaceholder(ElemKind::FloatTy, {8, 2}, "i1", false);
  auto *i2 = mod_.createPlaceholder(ElemKind::FloatTy, {8, 2}, "i2", false);
  auto *concat = F_->createConcat("concat", {i1, i2}, 1);
  auto *save = F_->createSave("save", concat);
  EE_.compile(CompilationMode::Infer);

  auto i1h = ctx_.allocate(i1)->getHandle<float>();
  auto i2h = ctx_.allocate(i2)->getHandle<float>();
  i1h = {
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  };
  i2h = {
      2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
  };
  auto OT = ctx_.allocate(save->getPlaceholder()); //->getHandle<float>();
  EE_.run(ctx_);

  Tensor expected(ElemKind::FloatTy, {8, 4});
  expected.getHandle() = {
      1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2,
      1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2,
  };
  EXPECT_TRUE(expected.isEqual(*OT));
}

TEST_F(HabanaBackendTest, IntermediateReshapeMLP) {
  auto *dense =
      mod_.createPlaceholder(ElemKind::FloatTy, {1000, 128}, "dense", false);
  auto *weights1 =
      mod_.createConstant(ElemKind::FloatTy, {128, 584}, "weights1");
  auto *biases1 = mod_.createConstant(ElemKind::FloatTy, {584}, "biases1");
  auto *fc2 = F_->createFullyConnected("fc2", dense, weights1, biases1);
  auto *reshape = F_->createReshape("reshape", fc2, {1000, 73, 8});
  F_->createSave("save", reshape);

  auto *weights2 =
      mod_.createConstant(ElemKind::FloatTy, {584, 128}, "weights2");
  auto *bias2 = mod_.createConstant(ElemKind::FloatTy, {128}, "bias2");
  auto *fc3 = F_->createFullyConnected("fc3", fc2, weights2, bias2);
  auto *relu2 = F_->createRELU("relu2", fc3);
  F_->createSave("save4", relu2);

  EE_.compile(CompilationMode::Infer);
  ctx_.allocate(mod_.getPlaceholders());
  EE_.run(ctx_);
}

TEST_F(HabanaBackendTest, BatchedReduceAdd) {
  auto *batch =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 4}, "batch", false);
  ctx_.allocate(batch)->getHandle() = {10, 20, 30, 40, 1, 2, 3, 4};

  auto *R = F_->createBatchedReduceAdd("reduce.add", batch, /* axis */ 0);

  auto *save = F_->createSave("save", R);
  auto *result = ctx_.allocate(save->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(ctx_);

  auto H = result->getHandle();
  EXPECT_NEAR(H.at({0}), 11, 0.001);
  EXPECT_NEAR(H.at({1}), 22, 0.001);
  EXPECT_NEAR(H.at({2}), 33, 0.001);
  EXPECT_NEAR(H.at({3}), 44, 0.001);
}

TEST_F(HabanaBackendTest, BigPseudoBatchedAdd) {
  constexpr size_t batch = 1000;

  auto *i1 =
      mod_.createPlaceholder(ElemKind::FloatTy, {batch, 2, 8}, "i1", false);
  auto *i2 =
      mod_.createPlaceholder(ElemKind::FloatTy, {batch, 2, 8}, "i2", false);

  std::vector<NodeValue> adds;
  for (size_t idx = 0; idx < batch; idx++) {
    auto *sl1 = F_->createSlice("sl1", i1, {idx, 0, 0}, {idx + 1, 2, 8});
    auto *r1 = F_->createReshape("r1", sl1, {2, 8});
    auto *sl2 = F_->createSlice("sl2", i2, {idx, 0, 0}, {idx + 1, 2, 8});
    auto *r2 = F_->createReshape("r2", sl2, {2, 8});
    auto *add = F_->createAdd("add", r1, r2);
    adds.push_back(add);
  }
  auto *concat = F_->createConcat("concat", adds, 0);
  auto *reshape = F_->createReshape("reshape", concat, {batch, 2, 8});
  auto *save = F_->createSave("save", reshape);
  auto *o = save->getPlaceholder();

  auto i1h = ctx_.allocate(i1)->getHandle();
  std::iota(i1h.begin(), i1h.end(), 0);
  auto i2h = ctx_.allocate(i2)->getHandle();
  std::iota(i2h.begin(), i2h.end(), 1);
  auto oh = ctx_.allocate(o)->getHandle();

  EE_.compile(CompilationMode::Infer);
  EE_.run(ctx_);

  int expected = 1;
  for (auto &out : oh) {
    ASSERT_EQ(out, expected);
    expected += 2;
  }
}

TEST_F(HabanaBackendTest, Mul) {
  auto *i1 = mod_.createPlaceholder(ElemKind::FloatTy, {8}, "i1", false);
  auto *i2 = mod_.createPlaceholder(ElemKind::FloatTy, {8}, "i2", false);
  auto *mul = F_->createMul("mul", i1, i2);
  auto *save = F_->createSave("save", mul);
  ctx_.allocate(i1)->getHandle() = {1, 2, 3, 4, 5, 6, 7, 8};
  ctx_.allocate(i2)->getHandle() = {8, 7, 6, 5, 4, 3, 2, 1};
  auto *out = ctx_.allocate(save->getPlaceholder());
  EE_.compile(CompilationMode::Infer);
  EE_.run(ctx_);
  Tensor expected(ElemKind::FloatTy, {8});
  expected.getHandle() = {8, 14, 18, 20, 20, 18, 14, 8};
  ASSERT_TRUE(out->isEqual(expected));
}

TEST_F(HabanaBackendTest, SingleFunctionMultiThreadMultiDevice) {
  // Test constants.
  constexpr unsigned maxDeviceManagers = 6;
  constexpr unsigned threadsPerDeviceManager = 6;
  constexpr unsigned iterationsPerThread = 50;

  using DeviceManager = glow::runtime::DeviceManager;
  using RunIdentifierTy = glow::runtime::RunIdentifierTy;

  // Create device managers.
  std::vector<std::unique_ptr<DeviceManager>> deviceManagers;

  for (unsigned i = 0; i < maxDeviceManagers; ++i) {
    DeviceManager *deviceManager =
        new glow::runtime::HabanaDeviceManager(runtime::DeviceConfig("Habana"));

    if (deviceManager->init()) {
      delete deviceManager;
    } else {
      deviceManagers.emplace_back(deviceManager);
    }
  }

  // At least one DeviceManager is needed to continue.
  ASSERT_GE(deviceManagers.size(), 0);

  // Create function.
  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {16, 32}, "input", false);
  auto *weights1 =
      mod_.createPlaceholder(ElemKind::FloatTy, {32, 32}, "weights1", false);
  // auto *weights2 =
  //     mod_.createPlaceholder(ElemKind::FloatTy, {32, 32}, "weights2", false);
  auto *bias = mod_.createPlaceholder(ElemKind::FloatTy, {32}, "bias", false);
  auto *output =
      mod_.createPlaceholder(ElemKind::FloatTy, {16, 32}, "output", false);

  auto *fc1 = F_->createFullyConnected("fc1", input, weights1, bias);
  // auto *fc2 = F_->createFullyConnected("fc2", fc1, weights2, bias);

  F_->createSave("save", fc1, output);
  ctx_.allocate(input)->getHandle().clear(1);
  ctx_.allocate(weights1)->getHandle().clear(0);
  // ctx_.allocate(weights2)->getHandle().clear(0);
  ctx_.allocate(bias)->getHandle().clear(32);
  ctx_.allocate(output);

  glow::convertPlaceholdersToConstants(F_, ctx_, {input, output});

  // Compile function.
  glow::runtime::FunctionMapTy functions;
  auto backend = std::unique_ptr<Backend>(createBackend("Habana"));
  CompilationContext cctx;
  cctx.compMode = CompilationMode::Infer;
  EXIT_ON_ERR(::glow::optimizeFunction(F_, *backend, cctx));
  auto compiledFunction = EXIT_ON_ERR(backend->compile(F_, cctx.backendOpts));
  functions.emplace(F_->getName(), compiledFunction.get());

  // Add the function to each device.
  std::vector<std::future<bool>> addNetworkFutures;

  for (auto &deviceManager : deviceManagers) {
    auto addNetworkPromise = std::make_shared<std::promise<bool>>();
    addNetworkFutures.emplace_back(addNetworkPromise->get_future());
    deviceManager->addNetwork(
        &mod_, functions,
        [promise = addNetworkPromise](const Module * /*module*/,
                                      Error err) mutable {
          promise->set_value(ERR_TO_BOOL(std::move(err)));
        });
  }

  for (auto &future : addNetworkFutures) {
    ASSERT_FALSE(future.get());
  }

  // Run function.
  std::vector<std::thread> threads;

  for (auto &deviceManager : deviceManagers) {
    for (size_t i = 0, t = threadsPerDeviceManager; i < t; ++i) {
      threads.emplace_back([deviceManager = deviceManager.get(),
                            functionName = F_->getName(), inputP = input,
                            outputP = output]() mutable {
        std::atomic<unsigned> completeIterations{1};
        std::promise<void> threadDonePromise;
        std::future<void> threadDoneFuture = threadDonePromise.get_future();

        std::vector<std::unique_ptr<ExecutionContext>> inputExecutionContexts;
        std::vector<std::shared_ptr<PlaceholderBindings>> outputBindings;

        // Compute inputs and outputs for all iterations to increase congestion
        // in runFunction (hopefully).
        for (unsigned j = 0, e = iterationsPerThread; j < e; ++j) {
          // Set inputs.
          auto iBindings = glow::make_unique<PlaceholderBindings>();
          auto inputHandle = iBindings->allocate(inputP)->getHandle();
          inputHandle.clear(1);

          iBindings->allocate(outputP);

          // Set expected outputs.
          auto oBindings =
              std::make_shared<PlaceholderBindings>(iBindings->clone());
          auto outputHandle = oBindings->get(outputP)->getHandle();
          outputHandle.clear(32);

          inputExecutionContexts.emplace_back(
              glow::make_unique<ExecutionContext>(std::move(iBindings)));
          outputBindings.emplace_back(std::move(oBindings));
        }

        for (unsigned j = 0, e = iterationsPerThread; j < e; ++j) {
          // Run function.
          deviceManager->runFunction(
              functionName.str(), std::move(inputExecutionContexts[j]),
              [&threadDonePromise, &completeIterations,
               expectedResultBindings = outputBindings[j]](
                  RunIdentifierTy runId, Error err,
                  std::unique_ptr<ExecutionContext> resultContext) {
                EXPECT_FALSE(ERR_TO_BOOL(std::move(err)));
                EXPECT_TRUE(PlaceholderBindings::compare(
                    resultContext->getPlaceholderBindings(),
                    expectedResultBindings.get()));

                // If all iterations are done (i.e. executed AND this callback
                // has been called), fulfill the promise.
                if (completeIterations++ == iterationsPerThread) {
                  threadDonePromise.set_value();
                }
              });
        }

        // Wait for all callbacks given to runFunction to be called.
        threadDoneFuture.wait();
      });
    }
  }

  // Join all threads.
  for (auto &thread : threads) {
    thread.join();
  }

  // Stop all devices.
  for (auto &deviceManager : deviceManagers) {
    EXPECT_FALSE(ERR_TO_BOOL(deviceManager->stop()));
  }
}

TEST_F(HabanaBackendTest, FCPerf) {
  const int fcSize = 1000;
  auto *data =
      mod_.createPlaceholder(ElemKind::FloatTy, {16, fcSize}, "data", false);
  auto *weights = mod_.createPlaceholder(ElemKind::FloatTy, {fcSize, fcSize},
                                         "weights", false);
  auto *bias =
      mod_.createPlaceholder(ElemKind::FloatTy, {fcSize}, "bias", false);

  auto *fc = F_->createFullyConnected("fc", data, weights, bias);
  auto *output =
      mod_.createPlaceholder(ElemKind::FloatTy, {16, fcSize}, "output", false);
  F_->createSave("save", fc, output);

  ctx_.allocate(data)->getHandle().clear(1);
  ctx_.allocate(weights)->getHandle().clear(0);
  ctx_.allocate(bias)->getHandle().clear(32);
  ctx_.allocate(output);

  glow::convertPlaceholdersToConstants(F_, ctx_, {data, output});

  EE_.compile(CompilationMode::Infer);
  EE_.run(ctx_);
  struct timespec begin;
  clock_gettime(CLOCK_MONOTONIC_RAW, &begin);
  for (int i = 0; i < 10; i++)
    EE_.run(ctx_);
  struct timespec end;
  clock_gettime(CLOCK_MONOTONIC_RAW, &end);
  double elapsedSecs = (end.tv_nsec - begin.tv_nsec) / 1000000000.0 +
                       (end.tv_sec - begin.tv_sec);
  printf("Time: %lf\n", elapsedSecs);
  uint64_t flops = 10 * fcSize * (uint64_t)fcSize * 16 * 2;
  printf("Tflops: %lf\n", (flops) / elapsedSecs * 1e-12);
}

// Test performance of Gather.
#if 0
// Disable Gather tests since Gather appears to be broken.
TEST_F(HabanaBackendTest, GatherPerf) {
  // Create function.
  auto *data = mod_.createPlaceholder(ElemKind::FloatTy, {50}, "data", false);
  auto *indices =
      mod_.createPlaceholder(ElemKind::Int32ITy, {3000}, "indices", false);

  auto H = ctx_.allocate(data)->getHandle();
  H.randomize(-1.0f, 1.0f, mod_.getPRNG());
  auto H2 = ctx_.allocate(indices)->getHandle<int32_t>();
  for (int i = 0; i < 50; i++) {
    H2.raw(i) = rand() % 50;
  }
  for (int i = 50; i < 3000; i++) {
    H2.raw(i) = 0;
  }

  // Create a gather (a single batch dimension).
  auto *R = F_->createGather("gather", data, indices, 0);

  auto *result = F_->createSave("save", R);
  ctx_.allocate(result->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(ctx_);
  struct timespec begin;
  clock_gettime(CLOCK_MONOTONIC_RAW, &begin);
  for (int i = 0; i < 1000; i++)
    EE_.run(ctx_);
  struct timespec end;
  clock_gettime(CLOCK_MONOTONIC_RAW, &end);
  double elapsedSecs = (end.tv_nsec - begin.tv_nsec) / 1000000000.0 +
                       (end.tv_sec - begin.tv_sec);
  printf("Time: %lf\n", elapsedSecs);
  printf("GBps: %lf\n", 10 * 3000 * 4.0 / elapsedSecs / 1000 / 1000 / 1000);
}

TEST_F(HabanaBackendTest, BatchedGather) {
  /*
   DATA  = [
    [1.0, 1.2, 2.4, 4.5],
    [2.3, 3.4, 3.6, 2.3],
    [4.5, 5.7, 1.2, 4.5],
   ]

   INDICES = [0, 2],

   OUTPUT = [
    [1.0, 2.4],
    [2.3, 3.6],
    [4.5, 1.2],
   ]
   */
  auto *data = mod_.createPlaceholder(ElemKind::FloatTy, {3, 4}, "data", false);
  auto *indices =
      mod_.createPlaceholder(ElemKind::Int32ITy, {2}, "indices", false);

  ctx_.allocate(data)->getHandle() = {
      1.0f, 1.2f, 2.4f, 4.5f, 2.3f, 3.4f, 3.6f, 2.3f, 4.5f, 5.7f, 1.2f, 4.5f,
  };
  ctx_.allocate(indices)->getHandle<int32_t>() = {
      0,
      2,
  };

  // Create a batched gather (a single batch dimension).
  auto *R = F_->createGather("gather", data, indices, 1);

  auto *result = F_->createSave("save", R);
  ctx_.allocate(result->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(ctx_);

  auto H = ctx_.get(result->getPlaceholder())->getHandle();
  EXPECT_FLOAT_EQ(H.at({0, 0}), 1.0);
  EXPECT_FLOAT_EQ(H.at({0, 1}), 2.4);
  EXPECT_FLOAT_EQ(H.at({1, 0}), 2.3);
  EXPECT_FLOAT_EQ(H.at({1, 1}), 3.6);
  EXPECT_FLOAT_EQ(H.at({2, 0}), 4.5);
  EXPECT_FLOAT_EQ(H.at({2, 1}), 1.2);
}

TEST_F(HabanaBackendTest, BatchedGatherMultipleRuns) {
  const unsigned M = 1000;
  const unsigned N = 1;

  // Fill out the array with random data
  std::vector<float> inputData;
  inputData.resize(M);
  for (unsigned int i = 0; i < M; i++) {
    inputData[i] = float(rand() % 1000) / 100;
  }

  // ID list, to be filled up
  unsigned idLen = 10000;
  std::vector<int> inputIds;
  inputIds.resize(idLen);

  // Create placeholder for data
  auto *data = mod_.createPlaceholder(ElemKind::FloatTy, {N, M}, "data", false);
  ctx_.allocate(data)->getHandle() = inputData;

  auto *indices =
      mod_.createPlaceholder(ElemKind::Int32ITy, {idLen}, "indices", false);
  auto indicesH = ctx_.allocate(indices)->getHandle<int32_t>();
  indicesH = inputIds;

  // create the net
  auto *R = F_->createGather("gather", data, indices, 1);
  auto *result = F_->createSave("save", R);
  ctx_.allocate(result->getPlaceholder());

  EE_.compile(CompilationMode::Infer);

  // run this multiple times
  for (auto ntimes = 0; ntimes < 10; ntimes++) {
    // fill up the ID list with random data
    for (unsigned int i = 0; i < inputIds.size(); i++) {
      inputIds[i] = rand() % M;
    }
    indicesH = inputIds;

    EE_.run(ctx_);

    auto H = ctx_.get(result->getPlaceholder())->getHandle();
    for (unsigned i = 0; i < idLen; i++) {
      EXPECT_FLOAT_EQ(inputData[inputIds[i]], H.at({0, i}));
    }
  }
}
#endif

TEST_F(HabanaBackendTest, MergeFCRelu) {
  auto *FCi = mod_.createPlaceholder(ElemKind::FloatTy, {2, 2}, "input", false);
  auto *FCw = mod_.createConstant(ElemKind::FloatTy, {2, 2}, "weight");
  auto *FCb = mod_.createConstant(ElemKind::FloatTy, {2}, "bias");
  auto *fcNode = F_->createFullyConnected("fc", FCi, FCw, FCb);
  auto *relu = F_->createRELU("relu", fcNode);
  F_->createSave("save", relu);

  // Should have three nodes FC, Relu, save
  ASSERT_EQ(F_->getNodes().size(), 3);

  EE_.compile(CompilationMode::Infer);

  // Should have two nodes FC, save
  ASSERT_EQ(F_->getNodes().size(), 2);
}
