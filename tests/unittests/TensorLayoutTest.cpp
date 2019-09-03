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

// Check CanonicalTensorLayout for conv fails verification with bad layout:
TEST_P(TensorLayoutTest, convBadLayout) {
  CHECK_IF_ENABLED();

  auto *input = mod_.createPlaceholder(ElemKind::FloatTy, {1, 3, 3, 1}, "input",
                                       false, "NWCH");
  auto IH = bindings_.allocate(input)->getHandle();
  IH = {1, 1, 1, 1, 1, 1, 1, 1, 1};

  auto filter = mod_.createPlaceholder(ElemKind::FloatTy, {1, 3, 3, 1},
                                       "filter", false, "NWCH");
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

  EXPECT_FALSE(verifyLayouts(*F_, CanonicalTensorLayout::getInstance(), false));
}

INSTANTIATE_BACKEND_TEST(TensorLayoutTest);
