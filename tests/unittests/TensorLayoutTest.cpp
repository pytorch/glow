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

#include "glow/Backend/Backend.h"
#include "glow/Graph/Graph.h"
#include "glow/Graph/TensorLayout.h"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

#include <sstream>

using namespace glow;

class TensorLayoutTest : public BackendTest {
protected:
  PlaceholderBindings bindings_;
};

// Check CanonicalTensorLayout for conv works default values:
TEST_P(TensorLayoutTest, convDefault) {
  CHECK_IF_ENABLED();

  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 3, 3, 1}, "input", false);
  auto IH = bindings_.allocate(input)->getHandle();
  IH = {1, 1, 1, 1, 1, 1, 1, 1, 1};

  auto filter =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 3, 3, 1}, "filter", false);
  auto FH = bindings_.allocate(filter)->getHandle();
  FH = {0, 0, 0, 1, 1, 1, 0, 0, 0};

  auto *zeroBias =
      mod_.createPlaceholder(ElemKind::FloatTy, {1}, "bias", false);
  bindings_.allocate(zeroBias)->zero();

  auto outTy = mod_.uniqueType(ElemKind::FloatTy, {1, 3, 3, 1});

  ConvolutionNode *CN =
      F_->createConv("Conv", input, filter, zeroBias, outTy, 3, 1, 1, 1);
  SaveNode *S = F_->createSave("save", CN);
  bindings_.allocate(S->getPlaceholder());

  EXPECT_TRUE(verifyLayouts(*F_, CanonicalTensorLayout::getInstance()));
}

static void buildBadConv(PlaceholderBindings &bindings, Module &mod,
                         Function *F) {
  auto *input = mod.createPlaceholder(ElemKind::FloatTy, {1, 3, 3, 1}, "input",
                                      false, "NWCH");
  auto IH = bindings.allocate(input)->getHandle();
  IH = {1, 1, 1, 1, 1, 1, 1, 1, 1};

  auto filter = mod.createPlaceholder(ElemKind::FloatTy, {1, 3, 3, 1}, "filter",
                                      false, "NWCH");
  auto FH = bindings.allocate(filter)->getHandle();
  FH = {0, 0, 0, 1, 1, 1, 0, 0, 0};

  auto *zeroBias = mod.createPlaceholder(ElemKind::FloatTy, {1}, "bias", false);
  bindings.allocate(zeroBias)->zero();

  auto outTy = mod.uniqueType(ElemKind::FloatTy, {1, 3, 3, 1});

  ConvolutionNode *CN =
      F->createConv("Conv", input, filter, zeroBias, outTy, 3, 1, 1, 1);
  SaveNode *S = F->createSave("save", CN);
  bindings.allocate(S->getPlaceholder());
}

// Check CanonicalTensorLayout for conv fails verification with bad layout:
TEST_P(TensorLayoutTest, convBadLayout) {
  CHECK_IF_ENABLED();

  buildBadConv(bindings_, mod_, F_);

  EXPECT_FALSE(verifyLayouts(*F_, CanonicalTensorLayout::getInstance(), false));
}

// Check TensorLayoutDescription's parser with simple input.
TEST_P(TensorLayoutTest, parseTestSimple) {
  CHECK_IF_ENABLED();

  TensorLayoutDescription simple("NHWC");
  EXPECT_FALSE(simple.isAnyLayout());
  EXPECT_EQ(simple.getNumDims(), 4);
  EXPECT_EQ(simple.getDims()[0], "N");
  EXPECT_EQ(simple.getDims()[1], "H");
  EXPECT_EQ(simple.getDims()[2], "W");
  EXPECT_EQ(simple.getDims()[3], "C");
  for (size_t i = 0; i < simple.getNumDims(); ++i) {
    EXPECT_EQ(simple.getAlignment(i), 1);
  }
}

// Check TensorLayoutDescription's parser with alignment.
TEST_P(TensorLayoutTest, parseTestAlignment) {
  CHECK_IF_ENABLED();

  TensorLayoutDescription alignment("N[a=32]HW[a=64]C");
  EXPECT_FALSE(alignment.isAnyLayout());
  EXPECT_EQ(alignment.getNumDims(), 4);
  EXPECT_EQ(alignment.getDims()[0], "N[a=32]");
  EXPECT_EQ(alignment.getDims()[1], "H");
  EXPECT_EQ(alignment.getDims()[2], "W[a=64]");
  EXPECT_EQ(alignment.getDims()[3], "C");
  EXPECT_EQ(alignment.getAlignment(0), 32);
  EXPECT_EQ(alignment.getAlignment(1), 1);
  EXPECT_EQ(alignment.getAlignment(2), 64);
  EXPECT_EQ(alignment.getAlignment(3), 1);
}

// Check TensorLayoutDescription's parser with custom extensions.
TEST_P(TensorLayoutTest, parseTestCustom) {
  CHECK_IF_ENABLED();

  TensorLayoutDescription custom("N[a=32][after:align]C[mal:reynolds][answer:"
                                 "42]HW[before:alignment][a=64]");
  EXPECT_FALSE(custom.isAnyLayout());
  EXPECT_EQ(custom.getNumDims(), 4);
  EXPECT_EQ(custom.getDims()[0], "N[a=32][after:align]");
  EXPECT_EQ(custom.getDims()[1], "C[mal:reynolds][answer:42]");
  EXPECT_EQ(custom.getDims()[2], "H");
  EXPECT_EQ(custom.getDims()[3], "W[before:alignment][a=64]");
  EXPECT_EQ(custom.getAlignment(0), 32);
  EXPECT_EQ(custom.getAlignment(1), 1);
  EXPECT_EQ(custom.getAlignment(2), 1);
  EXPECT_EQ(custom.getAlignment(3), 64);
}

// Check TensorLayoutDescription's parser with star dims.
TEST_P(TensorLayoutTest, parseTestStar) {
  CHECK_IF_ENABLED();

  TensorLayoutDescription custom("N[a=32]*H*[a=64]");
  EXPECT_FALSE(custom.isAnyLayout());
  EXPECT_EQ(custom.getNumDims(), 4);
  EXPECT_EQ(custom.getDims()[0], "N[a=32]");
  EXPECT_EQ(custom.getDims()[1], "*");
  EXPECT_EQ(custom.getDims()[2], "H");
  EXPECT_EQ(custom.getDims()[3], "*[a=64]");
  EXPECT_EQ(custom.getAlignment(0), 32);
  EXPECT_EQ(custom.getAlignment(1), 1);
  EXPECT_EQ(custom.getAlignment(2), 1);
  EXPECT_EQ(custom.getAlignment(3), 64);
}

INSTANTIATE_BACKEND_TEST(TensorLayoutTest);
