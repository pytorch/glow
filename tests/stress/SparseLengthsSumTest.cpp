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

#include "glow/ExecutionEngine/ExecutionEngine.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

using namespace glow;

class SparseLengthsSum : public BackendTest {};

TEST_P(SparseLengthsSum, Big) {
  ENABLED_BACKENDS("CPU", "Habana", "NNPI");
  ExecutionEngine interp{};
  interp.setDeviceMemory(10000000000);
  EE_.setDeviceMemory(10000000000);
  auto *mod = &EE_.getModule();
  F_ = mod->createFunction("main");
  auto *interpMod = &interp.getModule();
  auto *G = interp.getModule().createFunction("main");
  std::array<dim_t, 13> dataRows = {{
      5000000,
      5000000,
      6000000,
      8000000,
      8000000,
      8000000,
      3000000,
      3000000,
      1000000,
      5000000,
      8000000,
      5000000,
      1000000,
  }};

  std::vector<Constant *> data;
  std::vector<Constant *> dataI;
  std::vector<Placeholder *> indices;
  std::vector<Placeholder *> indicesI;
  std::vector<Placeholder *> lengths;
  std::vector<Placeholder *> lengthsI;
  std::vector<Placeholder *> weights;
  std::vector<Placeholder *> weightsI;
  std::vector<Placeholder *> results;
  std::vector<Placeholder *> resultsI;
  for (int i = 0; i < 13; i++) {
    LOG(INFO) << "Creating data " << i;
    Tensor fData(ElemKind::FloatTy, {dataRows[i], 72});
    fData.getHandle<float>().randomize(-1.0, 1.0, mod->getPRNG());
    auto *C = mod->createConstant(ElemKind::UInt8FusedQTy, {dataRows[i], 80},
                                  0.0, 0, "data");
    auto *CI = interpMod->createConstant(ElemKind::UInt8FusedQTy,
                                         {dataRows[i], 80}, 0.0, 0, "data");
    quantization::tensorFusedRowwiseQuantization<float>(fData,
                                                        C->getPayloadMutable());
    quantization::tensorFusedRowwiseQuantization<float>(
        fData, CI->getPayloadMutable());
    data.push_back(C);
    dataI.push_back(CI);
  }
  for (int i = 0; i < 13; i++) {
    LOG(INFO) << "Creating indices " << i;
    indices.push_back(
        mod->createPlaceholder(ElemKind::Int64ITy, {3000}, "indices", false));
    indicesI.push_back(interpMod->createPlaceholder(ElemKind::Int64ITy, {3000},
                                                    "indices", false));
  }
  for (int i = 0; i < 13; i++) {
    LOG(INFO) << "Creating lengths " << i;
    lengths.push_back(
        mod->createPlaceholder(ElemKind::Int32ITy, {10}, "lengths", false));
    lengthsI.push_back(interpMod->createPlaceholder(ElemKind::Int32ITy, {10},
                                                    "lengths", false));
  }
  for (int i = 0; i < 13; i++) {
    LOG(INFO) << "Creating weights " << i;
    weights.push_back(
        mod->createPlaceholder(ElemKind::FloatTy, {3000}, "weights", false));
    weightsI.push_back(interpMod->createPlaceholder(ElemKind::FloatTy, {3000},
                                                    "weights", false));
  }
  for (int i = 0; i < 10; i++) {
    auto *sls = F_->createFusedRowwiseQuantizedSparseLengthsWeightedSum(
        "sls", data[i], weights[i], indices[i], lengths[i]);
    auto *save = F_->createSave("save", sls);
    results.push_back(save->getPlaceholder());

    auto *slsI = G->createFusedRowwiseQuantizedSparseLengthsWeightedSum(
        "sls", dataI[i], weightsI[i], indicesI[i], lengthsI[i]);
    auto *saveI = G->createSave("save", slsI);
    results.push_back(saveI->getPlaceholder());
  }
  for (int i = 10; i < 11; i++) {
    auto *sls = F_->createFusedRowwiseQuantizedSparseLengthsWeightedSum(
        "sls", data[i], weights[i], indices[3], lengths[3]);
    auto *save = F_->createSave("save", sls);
    results.push_back(save->getPlaceholder());

    auto *slsI = G->createFusedRowwiseQuantizedSparseLengthsWeightedSum(
        "sls", dataI[i], weightsI[i], indicesI[3], lengthsI[3]);
    auto *saveI = G->createSave("save", slsI);
    results.push_back(saveI->getPlaceholder());
  }
  for (int i = 11; i < 13; i++) {
    auto *sls = F_->createFusedRowwiseQuantizedSparseLengthsSum(
        "sls", data[i], indices[i], lengths[i]);
    auto *save = F_->createSave("save", sls);
    results.push_back(save->getPlaceholder());

    auto *slsI = G->createFusedRowwiseQuantizedSparseLengthsSum(
        "sls", dataI[i], indicesI[i], lengthsI[i]);
    auto *saveI = G->createSave("save", slsI);
    results.push_back(saveI->getPlaceholder());
  }

  PlaceholderBindings bindings, interpBindings;
  for (size_t i = 0; i < indices.size(); i++) {
    auto *index = indices[i];
    auto *I = bindings.allocate(index);
    I->getHandle<int64_t>().randomize(0, dataRows[i] - 1, mod->getPRNG());

    auto *indexI = indicesI[i];
    auto *II = interpBindings.allocate(indexI);
    II->assign(I);
  }
  for (auto *length : lengths) {
    auto *L = bindings.allocate(length);
    L->getHandle<int32_t>().randomize(0, 100, mod->getPRNG());

    auto *lengthI = interpMod->getPlaceholderByName(length->getName());
    auto *LI = interpBindings.allocate(lengthI);
    LI->assign(L);
  }
  for (auto *weight : weights) {
    auto *W = bindings.allocate(weight);
    W->getHandle<float>().randomize(-1.0, 1.0, mod->getPRNG());

    auto *weightI = interpMod->getPlaceholderByName(weight->getName());
    auto *WI = interpBindings.allocate(weightI);
    WI->assign(W);
  }
  for (auto *result : results) {
    bindings.allocate(result);
  }
  for (auto *result : resultsI) {
    interpBindings.allocate(result);
  }

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings);
  std::vector<Tensor> test;
  for (auto *result : results) {
    auto *T = bindings.get(result);
    test.push_back(T->clone());
    T->getHandle<float>().clear(0);
  }

  interp.compile(CompilationMode::Infer);
  interp.run(interpBindings);
  std::vector<Tensor *> base;
  for (auto *result : resultsI) {
    auto *T = interpBindings.get(result);
    base.push_back(T);
  }

  for (size_t i = 0; i < base.size(); i++) {
    ASSERT_TRUE(base[i]->isEqual(test[i]));
  }
}

GLOW_INSTANTIATE_TEST_SUITE_P_FOR_BACKEND_TEST(SparseLengthsSum,
                                               SparseLengthsSum);

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
