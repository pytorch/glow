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
#include "ImporterTestUtils.h"
#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Graph/Graph.h"
#include "glow/Importer/ONNX.h"
#include "gtest/gtest.h"

using namespace glow;

/// Test loading conv op from a ONNX model.
/// The input is N*C*H*W (1*1*3*3), the kernels is {2, 2},
/// strides is {1, 1}, pads is {1, 1, 1, 1}, group is 1.
TEST(onnx, importConv) {
  ExecutionEngine EE{BackendKind::Interpreter};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::string NetFilename("tests/models/onnxModels/simpleConv.onnxtxt");

  SaveNode *graphOutputNode;
  // Destroy the loader after the graph is loaded since the following execution
  // will not depend on anyting from the loader.
  {
    Tensor data;
    getNCHWData(&data, 1, 1, 3, 3);
    ONNXModelLoader onnxLD(NetFilename, {"data"}, {&data}, *F);
    graphOutputNode = onnxLD.getSingleOutput();
  }

  // ONNX importer loads a conv node and converts it to 4 ops:
  // Transpose (input)   -> Conv -> Transpose
  // Transpose (filter) ->
  // A save node is added in the network as well. Therefore there are 5 nodes:
  // Transpose (input)   -> Conv -> Transpose -> Save
  // Transpose (filter) ->
  // Note that in case the convolution filter is a constant tensor, the filter
  // transpose node will be later optimized out by the optimizer.
  EXPECT_EQ(F->getNodes().size(), 5);
  EXPECT_EQ(mod.getVars().size(), 4);

  auto *node = graphOutputNode->getInput().getNode();

  EXPECT_TRUE(node->getKind() == Kinded::Kind::TransposeNodeKind);
  auto *convNode = llvm::dyn_cast<TransposeNode>(node)->getInput().getNode();

  EXPECT_TRUE(convNode->getKind() == Kinded::Kind::ConvolutionNodeKind);
  auto *tInNode =
      llvm::dyn_cast<ConvolutionNode>(convNode)->getInput().getNode();
  auto *tFilterNode =
      llvm::dyn_cast<ConvolutionNode>(convNode)->getFilter().getNode();
  EXPECT_TRUE(tInNode->getKind() == Kinded::Kind::TransposeNodeKind);
  EXPECT_TRUE(tFilterNode->getKind() == Kinded::Kind::TransposeNodeKind);

  EE.compile(CompilationMode::Infer, F);
  EE.run({}, {});
  auto result = graphOutputNode->getVariable()->getHandle();
  std::vector<size_t> expectedDims = {1, 1, 4, 4};
  std::vector<float> expectedValues = {2,  3,  5,  4,  5, 10, 14, 9,
                                       11, 22, 26, 15, 8, 15, 17, 10};
  EXPECT_TRUE(result.dims().vec() == expectedDims);
  for (size_t i = 0; i < 4 * 4; i++)
    EXPECT_FLOAT_EQ(result.raw(i), expectedValues[i]);
}
