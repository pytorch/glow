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

#include "glow/glow/torch_glow/src/InputMeta.h"
#include <gtest/gtest.h>

using namespace glow;

TEST(GlowIValueTests, TestOptimizedHashValidInputs) {
  InputMetaStack metaStack;
  int batchSize = 2;
  metaStack.inputMetas.emplace_back(
      InputMeta(c10::ScalarType::Float, c10::IntArrayRef({batchSize, 2, 3})));
  metaStack.inputMetas.emplace_back(
      InputMeta(c10::ScalarType::Float, c10::IntArrayRef({batchSize, 4})));
  metaStack.inputMetas.emplace_back(
      InputMeta(c10::ScalarType::Float, c10::IntArrayRef({6})));
  EXPECT_EQ(metaStack.optimizedHash(0), batchSize);
}

TEST(GlowIValueTests, TestOptimizedHashInvalidNoninalBatchIdx) {
  /// Test inputs are not valid(nomialBatchIdx is out of boundary).
  /// Fallback to hash function
  InputMetaStack metaStack;
  metaStack.inputMetas.emplace_back(
      InputMeta(c10::ScalarType::Float, c10::IntArrayRef({2, 2, 3})));
  metaStack.inputMetas.emplace_back(
      InputMeta(c10::ScalarType::Float, c10::IntArrayRef({6})));
  EXPECT_EQ(metaStack.optimizedHash(-1), metaStack.hash());
  EXPECT_EQ(metaStack.optimizedHash(2), metaStack.hash());
}

TEST(GlowIValueTests, TestOptimizedHashEmptyInputMeta) {
  /// Test inputs are not valid(nomial input has empty shape).
  InputMetaStack metaStack;
  metaStack.inputMetas.emplace_back(
      InputMeta(c10::ScalarType::Float, c10::IntArrayRef({})));
  metaStack.inputMetas.emplace_back(
      InputMeta(c10::ScalarType::Float, c10::IntArrayRef({3, 6})));
  EXPECT_EQ(metaStack.optimizedHash(0), metaStack.hash());
}
