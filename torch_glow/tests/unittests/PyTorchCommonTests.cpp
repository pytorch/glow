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

#include <gtest/gtest.h>
#include <torch/csrc/jit/api/module.h>

#include "glow/glow/torch_glow/src/InputMeta.h"
#include "glow/glow/torch_glow/src/PyTorchCommon.h"

TEST(PyTorchCommonTests, PostFusionProcessTest) {
  const auto moduleScript = R"JIT(
    def forward(self, input_0: Tensor, input_1: Tensor):
        res = input_0 + input_1
        return res.clone()
  )JIT";

  torch::jit::script::Module module{"testModule"};
  module.define(moduleScript);

  const std::vector<glow::sdim_t> dims{3, 4};

  glow::InputMetaStack inputMeta;
  inputMeta.inputMetas.emplace_back(c10::ScalarType::Float, dims);
  inputMeta.inputMetas.emplace_back(c10::ScalarType::Float, dims);

  auto settings = glow::getGlobalPyTorchLoaderSettingsSnapshot();

  std::shared_ptr<int> testPtr = std::make_shared<int>(0);
  ASSERT_EQ(0, *testPtr);

  glow::glowAOTFusionWithShapeInference(module, inputMeta, nullptr, settings,
                                        "forward", {}, nullptr, nullptr, "", "",
                                        [testPtr](auto &&) { *testPtr = 8; });

  ASSERT_EQ(8, *testPtr) << "Ensuring post fusion callback is invoked";
}

TEST(PyTorchCommonTests, IsSupportedFnTest) {
  const auto moduleScript = R"JIT(
    def forward(self, input_0: Tensor, input_1: Tensor):
        res = input_0 + input_1
        return res
  )JIT";

  glow::InputMetaStack inputMeta;
  const std::vector<glow::sdim_t> dims{3, 4};
  inputMeta.inputMetas.emplace_back(c10::ScalarType::Float, dims);
  inputMeta.inputMetas.emplace_back(c10::ScalarType::Float, dims);

  auto settings = glow::getGlobalPyTorchLoaderSettingsSnapshot();
  settings.minFusionGroupSize = 0;

  auto fusionNodeSymbol = glow::getGlowSymbol(nullptr);

  torch::jit::script::Module module{"ModuleDefaultSupportFn"};
  module.define(moduleScript);
  glow::glowAOTFusionWithShapeInference(module, inputMeta, nullptr, settings,
                                        "forward");
  int expectedFusionNodeCnt = 1;
  int fusionNodeCnt = 0;
  for (auto n : toGraphFunction(module.get_method("forward").function())
                    .graph()
                    ->nodes()) {
    auto kind = n->kind();
    if (fusionNodeSymbol == kind) {
      fusionNodeCnt++;
    }
  }
  ASSERT_EQ(expectedFusionNodeCnt, fusionNodeCnt)
      << "Check that with the default isNodeSupported function, a fusion "
         "node is created";

  torch::jit::script::Module moduleCustomSupportFn{"ModuleCustomSupportFn"};
  moduleCustomSupportFn.define(moduleScript);
  glow::glowAOTFusionWithShapeInference(
      moduleCustomSupportFn, inputMeta, nullptr, settings, "forward", {},
      nullptr, nullptr, "", "", c10::nullopt, c10::nullopt,
      [](auto &&) { return false; });
  expectedFusionNodeCnt = 0;
  fusionNodeCnt = 0;
  for (auto n :
       toGraphFunction(moduleCustomSupportFn.get_method("forward").function())
           .graph()
           ->nodes()) {
    auto kind = n->kind();
    if (fusionNodeSymbol == kind) {
      fusionNodeCnt++;
    }
  }
  ASSERT_EQ(expectedFusionNodeCnt, fusionNodeCnt)
      << "Check that with the custom isNodeSupported function, no fusion "
         "nodes are created.";
}
