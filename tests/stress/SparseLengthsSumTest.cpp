/**
 * Copyright (c) 2017-present, Facebook, Inc.
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

#include "glow/ExecutionEngine/ExecutionEngine.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

using namespace glow;

class SparseLengthsSum : public BackendTest {};

TEST_P(SparseLengthsSum, Big) {
  ENABLED_BACKENDS(CPU, Habana);

  std::array<size_t, 13> dataRows = {
      5000000, 5000000, 6000000, 8000000, 8000000, 8000000, 3000000,
      3000000, 1000000, 5000000, 8000000, 5000000, 1000000,
  };

  std::vector<Constant *> data;
  std::vector<Placeholder *> indices;
  std::vector<Placeholder *> lengths;
  std::vector<Placeholder *> weights;
  std::vector<Placeholder *> results;
  for (int i = 0; i < 13; i++) {
    LOG(INFO) << "Creating data " << i;
    Tensor fData(ElemKind::FloatTy, {dataRows[i], 72});
    fData.getHandle<float>().randomize(-1.0, 1.0, mod_.getPRNG());
    auto *C = mod_.createConstant(ElemKind::UInt8FusedQTy, {dataRows[i], 80},
                                  0.0, 0, "data");
    quantization::tensorFusedRowwiseQuantization(fData, C->getPayloadMutable());
    data.push_back(C);
  }
  for (int i = 0; i < 13; i++) {
    LOG(INFO) << "Creating indices " << i;
    indices.push_back(
        mod_.createPlaceholder(ElemKind::Int64ITy, {3000}, "indices", false));
  }
  for (int i = 0; i < 13; i++) {
    LOG(INFO) << "Creating lengths " << i;
    lengths.push_back(
        mod_.createPlaceholder(ElemKind::Int32ITy, {10}, "lengths", false));
  }
  for (int i = 0; i < 13; i++) {
    LOG(INFO) << "Creating weights " << i;
    weights.push_back(
        mod_.createPlaceholder(ElemKind::FloatTy, {3000}, "weights", false));
  }
  for (int i = 0; i < 10; i++) {
    auto *sls = F_->createFusedRowwiseQuantizedSparseLengthsWeightedSum(
        "sls", data[i], weights[i], indices[i], lengths[i]);
    auto *save = F_->createSave("save", sls);
    results.push_back(save->getPlaceholder());
  }
  for (int i = 10; i < 11; i++) {
    auto *sls = F_->createFusedRowwiseQuantizedSparseLengthsWeightedSum(
        "sls", data[i], weights[i], indices[3], lengths[3]);
    auto *save = F_->createSave("save", sls);
    results.push_back(save->getPlaceholder());
  }
  for (int i = 11; i < 13; i++) {
    auto *sls = F_->createFusedRowwiseQuantizedSparseLengthsSum(
        "sls", data[i], indices[i], lengths[i]);
    auto *save = F_->createSave("save", sls);
    results.push_back(save->getPlaceholder());
  }

  PlaceholderBindings bindings;
  for (size_t i = 0; i < indices.size(); i++) {
    auto *index = indices[i];
    auto *I = bindings.allocate(index);
    I->getHandle<int64_t>().randomize(0, dataRows[i] - 1, mod_.getPRNG());
  }
  for (auto *length : lengths) {
    auto *L = bindings.allocate(length);
    L->getHandle<int32_t>().randomize(0, 100, mod_.getPRNG());
  }
  for (auto *weight : weights) {
    auto *W = bindings.allocate(weight);
    W->getHandle<float>().randomize(-1.0, 1.0, mod_.getPRNG());
  }
  for (auto *result : results) {
    bindings.allocate(result);
  }

  EE_.compile(CompilationMode::Infer, F_);
  EE_.run(bindings);
  std::vector<Tensor> test;
  for (auto *result : results) {
    auto *T = bindings.get(result);
    test.push_back(T->clone());
    T->getHandle<float>().clear(0);
  }

  // TODO: We should really clone the function, since compiling for the test
  // backend might mutate the graph such that it is invalid for the
  // interpreter.  Sadly, cloning the graph currently creates problems for some
  // backends (e.g., Habana).
  ExecutionEngine interp{};
  interp.compile(CompilationMode::Infer, F_);
  interp.run(bindings);
  std::vector<Tensor *> base;
  for (auto *result : results) {
    auto *T = bindings.get(result);
    base.push_back(T);
  }

  for (size_t i = 0; i < base.size(); i++) {
    ASSERT_TRUE(base[i]->isEqual(test[i]));
  }
}

INSTANTIATE_TEST_CASE_P_FOR_BACKEND_TEST(SparseLengthsSum, SparseLengthsSum);

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
