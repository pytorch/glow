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

#include "glow/Converter/Float16Converter.h"
#include "glow/Converter/TypeAToTypeBFunctionConverter.h"

#include "glow/Backend/Backend.h"
#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Optimizer/GraphOptimizer/GraphOptimizer.h"

#include "llvm/Support/Casting.h"

#include "gtest/gtest.h"

using namespace glow;

/// Check that a simple graph is converted properly.
/// Namely, check that:
/// \verbatim
/// Input: Placeholder(float)
///    |
///    V
///   FC(float)  Output: Placeholder(float)
///    |            |
///    |    +-------+
///    |   /
///    V  V
///   Save
/// \endverbatim
///
/// Gets converted into:
/// \verbatim
/// Input: Placeholder(float)
///    |
///    V
/// ConvertTo(float16)
///    |
///    V
///   FC(float16)  Output: Placeholder(float)
///    |              |
///    V              |
/// ConvertTo(float)  |
///    |    +---------+
///    |   /
///    V  V
///   Save
/// \endverbatim
///
/// In particular, the input and output of the network shouldn't be modified.
TEST(TypeAToTypeBFunctionConverter, SimpleOneUseConversionFloatToFloat16) {
  Module mod;
  Function *F = mod.createFunction("test");
  PlaceholderBindings bindings;

  auto *input =
      mod.createPlaceholder(ElemKind::FloatTy, {20, 13}, "Input", false);
  auto *output =
      mod.createPlaceholder(ElemKind::FloatTy, {20, 10}, "Output", false);

  auto *FC = F->createFullyConnected(bindings, "FC", input, 10);
  auto *result = F->createSave("save", FC, output);

  size_t origGraphSize = F->getNodes().size();

  PrecisionConfiguration precConfig;
  TypeAToTypeBFunctionConverter converter(*F, ElemKind::FloatTy,
                                          ElemKind::Float16Ty, precConfig);
  converter.convert();

  // We should have 4 more nodes:
  // 1 conversion float to float16 for each input of FC (3)
  // and 1 conversion float16 to float for the result of FC.
  EXPECT_EQ(F->getNodes().size(), origGraphSize + 4);
  // Make sure the save node is still in the function and is unchanged.
  EXPECT_TRUE(std::find(F->getNodes().begin(), F->getNodes().end(), *result) !=
              F->getNodes().end());
  EXPECT_EQ(result->getOutput(), output->getOutput());
  // Check that the save is fed from a conversion from float16 to float.
  auto *convertedBackFCRes = llvm::dyn_cast<ConvertToNode>(result->getInput());
  ASSERT_NE(convertedBackFCRes, nullptr);
  EXPECT_EQ(convertedBackFCRes->getElementType(ConvertToNode::ResultIdx),
            ElemKind::FloatTy);
  auto *convertedFC =
      llvm::dyn_cast<FullyConnectedNode>(convertedBackFCRes->getInput());
  ASSERT_NE(convertedFC, nullptr);
  EXPECT_EQ(convertedFC->getElementType(FullyConnectedNode::ResultIdx),
            ElemKind::Float16Ty);
  // Check that all the input of FC are convertTo node with from float to
  // Float16Ty.
  for (unsigned idx = 0, end = convertedFC->getNumInputs(); idx != end; ++idx) {
    auto *convertedFCInput =
        llvm::dyn_cast<ConvertToNode>(convertedFC->getNthInput(idx));
    ASSERT_NE(convertedFCInput, nullptr);
    EXPECT_EQ(convertedFCInput->getElementType(ConvertToNode::ResultIdx),
              ElemKind::Float16Ty);
    EXPECT_TRUE(llvm::isa<Placeholder>(convertedFCInput->getInput()));
    EXPECT_EQ(convertedFCInput->getInput().getElementType(), ElemKind::FloatTy);
  }
  // At this point we know the input of FC is convertTo(placeholder).
  // Check that this placeholder is the expected input.
  EXPECT_EQ(convertedFC->getInput()
                .getNode()
                ->getNthInput(ConvertToNode::InputIdx)
                .getNode(),
            input);
}

/// Check that a graph with a simple chain of computation is converted
/// properly. In particular, check that the intermediate conversion
/// steps are not eliminated by default.
/// Namely, check that:
/// \verbatim
/// Input: Placeholder(float)
///    |
///    V
///   FC(float)
///    |
///    V
///   ReLU(float)  Output: Placeholder(float)
///    |            |
///    |    +-------+
///    |   /
///    V  V
///   Save
/// \endverbatim
///
/// Gets converted into:
/// \verbatim
/// Input: Placeholder(float)
///    |
///    V
/// ConvertTo(float16)
///    |
///    V
///   FC(float16)
///    |
///    V
/// ConvertTo(float)
///    |
///    V
/// ConvertTo(float16)
///    |
///    V
///   ReLU(float16)  Output: Placeholder(float)
///    |              |
///    V              |
/// ConvertTo(float)  |
///    |    +---------+
///    |   /
///    V  V
///   Save
/// \endverbatim
///
/// In particular, the input and output of the network shouldn't be modified.
TEST(TypeAToTypeBFunctionConverter,
     SimpleChainOfComputationConversionFloatToFloat16) {
  Module mod;
  Function *F = mod.createFunction("test");
  PlaceholderBindings bindings;

  auto *input =
      mod.createPlaceholder(ElemKind::FloatTy, {20, 13}, "Input", false);
  auto *output =
      mod.createPlaceholder(ElemKind::FloatTy, {20, 10}, "Output", false);

  auto *FC = F->createFullyConnected(bindings, "FC", input, 10);
  auto *ReLU =
      F->createRELU("ReLU", FC, FC->getType(FullyConnectedNode::ResultIdx));
  auto *result = F->createSave("save", ReLU, output);

  size_t origGraphSize = F->getNodes().size();

  PrecisionConfiguration precConfig;
  TypeAToTypeBFunctionConverter converter(*F, ElemKind::FloatTy,
                                          ElemKind::Float16Ty, precConfig);
  converter.convert();

  // We should have 6 more nodes:
  // 1 conversion float to float16 for each input of FC (3)
  // 1 conversion float to float16 for the input of ReLU
  // 1 conversion float16 to float for the result of FC.
  // 1 conversion float16 to float for the result of ReLU.
  EXPECT_EQ(F->getNodes().size(), origGraphSize + 6);
  // Make sure the save node is still in the function and is unchanged.
  EXPECT_TRUE(std::find(F->getNodes().begin(), F->getNodes().end(), *result) !=
              F->getNodes().end());
  EXPECT_EQ(result->getOutput(), output->getOutput());
  // Check that the save is fed from a conversion from float16 to float.
  auto *convertedBackReLURes =
      llvm::dyn_cast<ConvertToNode>(result->getInput());
  ASSERT_NE(convertedBackReLURes, nullptr);
  EXPECT_EQ(convertedBackReLURes->getElementType(ConvertToNode::ResultIdx),
            ElemKind::FloatTy);
  auto *convertedReLU =
      llvm::dyn_cast<ReluNode>(convertedBackReLURes->getInput());
  ASSERT_NE(convertedReLU, nullptr);
  EXPECT_EQ(convertedReLU->getElementType(ReluNode::ResultIdx),
            ElemKind::Float16Ty);

  // Check that the ReLU is fed from a conversion from float to float16.
  auto *convertedToReLUInput =
      llvm::dyn_cast<ConvertToNode>(convertedReLU->getInput());
  ASSERT_NE(convertedToReLUInput, nullptr);
  EXPECT_EQ(convertedToReLUInput->getElementType(ConvertToNode::ResultIdx),
            ElemKind::Float16Ty);

  // Check that this conversion is fed from a conversion from float16 to float.
  auto *convertedBackFCRes =
      llvm::dyn_cast<ConvertToNode>(convertedToReLUInput->getInput());
  ASSERT_NE(convertedBackFCRes, nullptr);
  EXPECT_EQ(convertedBackFCRes->getElementType(ConvertToNode::ResultIdx),
            ElemKind::FloatTy);
  // Check that this conversion comes from the float16 FC node.
  auto *convertedFC =
      llvm::dyn_cast<FullyConnectedNode>(convertedBackFCRes->getInput());
  ASSERT_NE(convertedFC, nullptr);
  EXPECT_EQ(convertedFC->getElementType(FullyConnectedNode::ResultIdx),
            ElemKind::Float16Ty);
  // Check that all the input of FC are convertTo node with from float to
  // Float16Ty.
  for (unsigned idx = 0, end = convertedFC->getNumInputs(); idx != end; ++idx) {
    auto *convertedFCInput =
        llvm::dyn_cast<ConvertToNode>(convertedFC->getNthInput(idx));
    ASSERT_NE(convertedFCInput, nullptr);
    EXPECT_EQ(convertedFCInput->getElementType(ConvertToNode::ResultIdx),
              ElemKind::Float16Ty);
    EXPECT_TRUE(llvm::isa<Placeholder>(convertedFCInput->getInput()));
    EXPECT_EQ(convertedFCInput->getInput().getElementType(), ElemKind::FloatTy);
  }
  // At this point we know the input of FC is convertTo(placeholder).
  // Check that this placeholder is the expected input.
  EXPECT_EQ(convertedFC->getInput()
                .getNode()
                ->getNthInput(ConvertToNode::InputIdx)
                .getNode(),
            input);
}

/// Check that the conversion honor the precision configuration for blacklisting
/// a node kind (Relu here) for a graph with a simple chain of computation.
/// Namely, check that:
/// \verbatim
/// Input: Placeholder(float)
///    |
///    V
///   FC(float)
///    |
///    V
///   ReLU(float)  Output: Placeholder(float)
///    |            |
///    |    +-------+
///    |   /
///    V  V
///   Save
/// \endverbatim
///
/// Gets converted into:
/// \verbatim
/// Input: Placeholder(float)
///    |
///    V
/// ConvertTo(float16)
///    |
///    V
///   FC(float16)
///    |
///    V
/// ConvertTo(float)
///    |
///    V
///   ReLU(float)  Output: Placeholder(float)
///    |              |
///    |    +---------+
///    |   /
///    V  V
///   Save
/// \endverbatim
///
/// In particular, the input and output of the network shouldn't be modified.
TEST(TypeAToTypeBFunctionConverter, DoNotConvertReLUConversionFloatToFloat16) {
  Module mod;
  Function *F = mod.createFunction("test");
  PlaceholderBindings bindings;

  auto *input =
      mod.createPlaceholder(ElemKind::FloatTy, {20, 13}, "Input", false);
  auto *output =
      mod.createPlaceholder(ElemKind::FloatTy, {20, 10}, "Output", false);

  auto *FC = F->createFullyConnected(bindings, "FC", input, 10);
  auto *ReLU =
      F->createRELU("ReLU", FC, FC->getType(FullyConnectedNode::ResultIdx));
  auto *result = F->createSave("save", ReLU, output);

  size_t origGraphSize = F->getNodes().size();

  PrecisionConfiguration precConfig;
  precConfig.precisionModeKindSet.insert(Kinded::Kind::ReluNodeKind);
  TypeAToTypeBFunctionConverter converter(*F, ElemKind::FloatTy,
                                          ElemKind::Float16Ty, precConfig);
  converter.convert();

  // We should have 4 more nodes:
  // 1 conversion float to float16 for each input of FC (3)
  // 1 conversion float16 to float for the result of FC.
  EXPECT_EQ(F->getNodes().size(), origGraphSize + 4);
  // Make sure the save node is still in the function and is unchanged.
  EXPECT_TRUE(std::find(F->getNodes().begin(), F->getNodes().end(), *result) !=
              F->getNodes().end());
  EXPECT_EQ(result->getOutput(), output->getOutput());
  // Check that the save is fed from a conversion from float16 to float.
  auto *resultInput = llvm::dyn_cast<ReluNode>(result->getInput());
  ASSERT_NE(resultInput, nullptr);
  EXPECT_EQ(resultInput->getElementType(ReluNode::ResultIdx),
            ElemKind::FloatTy);
  EXPECT_EQ(resultInput, ReLU);

  // Check that the ReLU is fed from a conversion from float16 to float.
  auto *convertedToReLUInput = llvm::dyn_cast<ConvertToNode>(ReLU->getInput());
  ASSERT_NE(convertedToReLUInput, nullptr);
  EXPECT_EQ(convertedToReLUInput->getElementType(ConvertToNode::ResultIdx),
            ElemKind::FloatTy);

  // Check that this conversion comes from the float16 FC node.
  auto *convertedFC =
      llvm::dyn_cast<FullyConnectedNode>(convertedToReLUInput->getInput());
  ASSERT_NE(convertedFC, nullptr);
  EXPECT_EQ(convertedFC->getElementType(FullyConnectedNode::ResultIdx),
            ElemKind::Float16Ty);
  // Check that all the input of FC are convertTo node with from float to
  // Float16Ty.
  for (unsigned idx = 0, end = convertedFC->getNumInputs(); idx != end; ++idx) {
    auto *convertedFCInput =
        llvm::dyn_cast<ConvertToNode>(convertedFC->getNthInput(idx));
    ASSERT_NE(convertedFCInput, nullptr);
    EXPECT_EQ(convertedFCInput->getElementType(ConvertToNode::ResultIdx),
              ElemKind::Float16Ty);
    EXPECT_TRUE(llvm::isa<Placeholder>(convertedFCInput->getInput()));
    EXPECT_EQ(convertedFCInput->getInput().getElementType(), ElemKind::FloatTy);
  }
  // At this point we know the input of FC is convertTo(placeholder).
  // Check that this placeholder is the expected input.
  EXPECT_EQ(convertedFC->getInput()
                .getNode()
                ->getNthInput(ConvertToNode::InputIdx)
                .getNode(),
            input);
}

/// Check that the conversion honor the precision configuration for whitelisting
/// a node kind (Relu here) for a graph with a simple chain of computation.
/// Namely, check that:
/// \verbatim
/// Input: Placeholder(float)
///    |
///    V
///   FC(float)
///    |
///    V
///   ReLU(float)  Output: Placeholder(float)
///    |            |
///    |    +-------+
///    |   /
///    V  V
///   Save
/// \endverbatim
///
/// Gets converted into:
/// \verbatim
/// Input: Placeholder(float)
///    |
///    V
///   FC(float)
///    |
///    V
/// ConvertTo(float16)
///    |
///    V
///   ReLU(float16)
///    |
///    V
/// ConvertTo(float)   Output: Placeholder(float)
///    |                    |
///    |    +---------------+
///    |   /
///    V  V
///   Save
/// \endverbatim
///
/// In particular, the input and output of the network shouldn't be modified.
TEST(TypeAToTypeBFunctionConverter, OnlyReluConversionFloatToFloat16) {
  Module mod;
  Function *F = mod.createFunction("test");
  PlaceholderBindings bindings;

  auto *input =
      mod.createPlaceholder(ElemKind::FloatTy, {20, 13}, "Input", false);
  auto *output =
      mod.createPlaceholder(ElemKind::FloatTy, {20, 10}, "Output", false);

  auto *FC = F->createFullyConnected(bindings, "FC", input, 10);
  auto *RN =
      F->createRELU("Relu", FC, FC->getType(FullyConnectedNode::ResultIdx));
  auto *result = F->createSave("save", RN, output);

  size_t origGraphSize = F->getNodes().size();

  PrecisionConfiguration precConfig;
  precConfig.precisionModeKindSet.insert(Kinded::Kind::ReluNodeKind);
  precConfig.useSetAsWhitelist = true;
  TypeAToTypeBFunctionConverter converter(*F, ElemKind::FloatTy,
                                          ElemKind::Float16Ty, precConfig);
  converter.convert();

  // We should have 4 more nodes:
  // 1 conversion float to float16 for the input of Relu.
  // 1 conversion float16 to float for the result of Relu.
  EXPECT_EQ(F->getNodes().size(), origGraphSize + 2);
  // Make sure the save node is still in the function and is unchanged.
  EXPECT_TRUE(std::find(F->getNodes().begin(), F->getNodes().end(), *result) !=
              F->getNodes().end());
  EXPECT_EQ(result->getOutput(), output->getOutput());
  // Check that the save is fed from a conversion from float16 to float.
  auto *resultInput = llvm::dyn_cast<ConvertToNode>(result->getInput());
  ASSERT_NE(resultInput, nullptr);
  EXPECT_EQ(resultInput->getInput().getElementType(), ElemKind::Float16Ty);
  EXPECT_EQ(resultInput->getResult().getElementType(), ElemKind::FloatTy);

  // Check the Relu has FP16 inputs and outputs.
  auto *convertedRelu = llvm::dyn_cast<ReluNode>(resultInput->getInput());
  ASSERT_NE(convertedRelu, nullptr);
  EXPECT_EQ(convertedRelu->getInput().getElementType(), ElemKind::Float16Ty);
  EXPECT_EQ(convertedRelu->getResult().getElementType(), ElemKind::Float16Ty);

  // Check that the Relu is fed from a conversion from float to float16.
  auto *convertedToReluInput = llvm::dyn_cast<ConvertToNode>(RN->getInput());
  ASSERT_NE(convertedToReluInput, nullptr);
  EXPECT_EQ(convertedToReluInput->getInput().getElementType(),
            ElemKind::FloatTy);
  EXPECT_EQ(convertedToReluInput->getResult().getElementType(),
            ElemKind::Float16Ty);

  // Check that this conversion comes from the original float FC node.
  EXPECT_EQ(convertedToReluInput->getInput().getNode(), FC);
  EXPECT_EQ(FC->getResult().getElementType(), ElemKind::FloatTy);
  // Check that all the input of FC are float.
  for (unsigned idx = 0, end = FC->getNumInputs(); idx != end; ++idx) {
    EXPECT_EQ(FC->getNthInput(idx).getElementType(), ElemKind::FloatTy);
  }
  // Check that the original placeholder is still the input to the FC and float.
  EXPECT_EQ(FC->getInput().getNode(), input);
  EXPECT_EQ(input->getOutput().getElementType(), ElemKind::FloatTy);
}

/// Check that don't convert types we didn't asked for.
/// Namely, check that:
/// \verbatim
/// Input: Placeholder(float)
///    |
///    V
///   TopK(float, Int64I)
///    |     |
///    |     |  Output: Placeholder(Int64I)
///    |     |    /
///    |     V   V
///    |     Save   Output: Placeholder(float)
///    |            |
///    |    +-------+
///    |   /
///    V  V
///   Save
/// \endverbatim
///
/// Gets converted into:
/// \verbatim
/// Input: Placeholder(float)
///    |
///    V
/// ConvertTo(float16)
///    |
///    V
///   TopK(float, Int64I)
///    |     |
///    |     |  Output: Placeholder(Int64I)
///    |     |    /
///    |     V   V
///    |     Save   Output: Placeholder(float)
///    V               |
/// ConvertTo(float)   |
///    |               |
///    |    +----------+
///    |   /
///    V  V
/// \endverbatim
///
/// In particular, the input and outputs of the network shouldn't be modified
/// as well as the Int64I result.
TEST(TypeAToTypeBFunctionConverter, int64IConversionFloatToFloat16) {
  Module mod;
  Function *F = mod.createFunction("test");
  PlaceholderBindings bindings;

  auto *input =
      mod.createPlaceholder(ElemKind::FloatTy, {20, 13}, "Input", false);
  auto *output =
      mod.createPlaceholder(ElemKind::FloatTy, {20, 3}, "Output", false);
  auto *outputIdx =
      mod.createPlaceholder(ElemKind::Int64ITy, {20, 3}, "Output", false);

  auto *topK = F->createTopK("topK", input, 3);
  auto *result = F->createSave("save", topK->getValues(), output);
  auto *resultIndices = F->createSave("saveIdx", topK->getIndices(), outputIdx);

  size_t origGraphSize = F->getNodes().size();

  PrecisionConfiguration precConfig;
  TypeAToTypeBFunctionConverter converter(*F, ElemKind::FloatTy,
                                          ElemKind::Float16Ty, precConfig);
  converter.convert();

  // We should have 2 more nodes:
  // 1 conversion float to float16 the input of TopK
  // and 1 conversion float16 to float for the result of TopK.
  EXPECT_EQ(F->getNodes().size(), origGraphSize + 2);
  // Make sure the save node is still in the function and is unchanged.
  EXPECT_TRUE(std::find(F->getNodes().begin(), F->getNodes().end(), *result) !=
              F->getNodes().end());
  EXPECT_EQ(result->getOutput(), output->getOutput());
  // Check that the save is fed from a conversion from float16 to float.
  auto *convertedBackTopKRes =
      llvm::dyn_cast<ConvertToNode>(result->getInput());
  ASSERT_NE(convertedBackTopKRes, nullptr);
  EXPECT_EQ(convertedBackTopKRes->getElementType(ConvertToNode::ResultIdx),
            ElemKind::FloatTy);
  auto *convertedTopK =
      llvm::dyn_cast<TopKNode>(convertedBackTopKRes->getInput());
  ASSERT_NE(convertedTopK, nullptr);
  EXPECT_EQ(convertedTopK->getElementType(TopKNode::ValuesIdx),
            ElemKind::Float16Ty);
  EXPECT_EQ(convertedTopK->getElementType(TopKNode::IndicesIdx),
            ElemKind::Int64ITy);
  // Check that the input of TopK is a convertTo node from float to
  // Float16Ty.
  auto *convertedTopKInput =
      llvm::dyn_cast<ConvertToNode>(convertedTopK->getInput());
  ASSERT_NE(convertedTopKInput, nullptr);
  EXPECT_EQ(convertedTopKInput->getElementType(ConvertToNode::ResultIdx),
            ElemKind::Float16Ty);
  EXPECT_TRUE(llvm::isa<Placeholder>(convertedTopKInput->getInput()));
  EXPECT_EQ(convertedTopKInput->getInput().getElementType(), ElemKind::FloatTy);
  // At this point we know the input of TopK is convertTo(placeholder).
  // Check that this placeholder is the expected input.
  EXPECT_EQ(convertedTopK->getInput()
                .getNode()
                ->getNthInput(ConvertToNode::InputIdx)
                .getNode(),
            input);

  // Now check the Int64ITy part of the graph.
  // Make sure the save node for the indices is still in the function and is
  // unchanged.
  EXPECT_TRUE(std::find(F->getNodes().begin(), F->getNodes().end(),
                        *resultIndices) != F->getNodes().end());
  EXPECT_EQ(resultIndices->getOutput(), outputIdx->getOutput());
  EXPECT_EQ(resultIndices->getInput(),
            convertedTopK->getNthResult(TopKNode::IndicesIdx));
  EXPECT_EQ(resultIndices->getInput().getElementType(), ElemKind::Int64ITy);
}

/// Check that the conversion optimization can get rid of conversion of
/// constants and intermediate conversions.
/// Namely, check that:
/// \verbatim
/// Input: Placeholder(float)
///    |
///    |  Weight: Constant(float)
///    |   |   Bias: Constant(float)
///    |   |   /
///    V   V  V
///   FC(float)
///    |
///    V
///   ReLU(float)  Output: Placeholder(float)
///    |            |
///    |    +-------+
///    |   /
///    V  V
///   Save
/// \endverbatim
///
/// Gets converted into:
/// \verbatim
/// Input: Placeholder(float)
///    |
///    V
/// ConvertTo(float16)
///    |
///    |  Weight: Constant(float16)
///    |   |   Bias: Constant(float16)
///    |   |   /
///    V   V  V
///   FC(float16)
///    |
///    V
///   ReLU(float16)  Output: Placeholder(float)
///    |              |
///    V              |
/// ConvertTo(float)  |
///    |    +---------+
///    |   /
///    V  V
///   Save
/// \endverbatim
///
/// In particular, the input and output of the network shouldn't be modified.
TEST(TypeAToTypeBFunctionConverter, OptimizeMiddleConversionsFloatToFloat16) {
  Module mod;
  Function *F = mod.createFunction("test");
  PlaceholderBindings bindings;

  auto *input =
      mod.createPlaceholder(ElemKind::FloatTy, {20, 13}, "Input", false);
  auto *output =
      mod.createPlaceholder(ElemKind::FloatTy, {20, 10}, "Output", false);

  auto *weights = mod.createConstant(
      mod.uniqueType(ElemKind::FloatTy, {13, 10}), "weights");
  weights->getPayloadMutable().getHandle().randomize(-5.0, 5.0, mod.getPRNG());
  Tensor origWeights;
  origWeights.assign(&weights->getPayload());
  auto *bias =
      mod.createConstant(mod.uniqueType(ElemKind::FloatTy, {10}), "bias");
  bias->getPayloadMutable().getHandle().randomize(-5.0, 5.0, mod.getPRNG());
  Tensor origBias;
  origBias.assign(&bias->getPayload());

  // This save is just to test that we do the right thing for constants with
  // more than one use.
  auto *saveBias = F->createSave("saveBias", bias);
  TypeRef FCTy = mod.uniqueType(ElemKind::FloatTy, {20, 10});
  auto *FC = F->createFullyConnected("FC", input, weights, bias, FCTy);
  auto *ReLU =
      F->createRELU("ReLU", FC, FC->getType(FullyConnectedNode::ResultIdx));
  auto *result = F->createSave("save", ReLU, output);

  size_t origGraphSize = F->getNodes().size();

  PrecisionConfiguration precConfig;
  TypeAToTypeBFunctionConverter converter(*F, ElemKind::FloatTy,
                                          ElemKind::Float16Ty, precConfig);
  converter.convert();

  optimize(F, CompilationMode::Infer);

  // We should have 2 more nodes:
  // 1 conversion float to float16 for the input of FC
  // 1 conversion float16 to float for the result of ReLU.
  EXPECT_EQ(F->getNodes().size(), origGraphSize + 2);
  // Make sure the save node is still in the function and is unchanged.
  EXPECT_TRUE(std::find(F->getNodes().begin(), F->getNodes().end(), *result) !=
              F->getNodes().end());
  EXPECT_EQ(result->getOutput(), output->getOutput());
  // Check that the save is fed from a conversion from float16 to float.
  auto *convertedBackReLURes =
      llvm::dyn_cast<ConvertToNode>(result->getInput());
  ASSERT_NE(convertedBackReLURes, nullptr);
  EXPECT_EQ(convertedBackReLURes->getElementType(ConvertToNode::ResultIdx),
            ElemKind::FloatTy);
  auto *convertedReLU =
      llvm::dyn_cast<ReluNode>(convertedBackReLURes->getInput());
  ASSERT_NE(convertedReLU, nullptr);
  EXPECT_EQ(convertedReLU->getElementType(ReluNode::ResultIdx),
            ElemKind::Float16Ty);

  // Check that the ReLU is fed directly by FC float16.
  auto *convertedFC =
      llvm::dyn_cast<FullyConnectedNode>(convertedReLU->getInput());
  ASSERT_NE(convertedFC, nullptr);
  EXPECT_EQ(convertedFC->getElementType(FullyConnectedNode::ResultIdx),
            ElemKind::Float16Ty);
  // Check that the input of FC is a convertTo node from "input" from float to
  // Float16Ty.
  auto *convertedFCInput =
      llvm::dyn_cast<ConvertToNode>(convertedFC->getInput());
  ASSERT_NE(convertedFCInput, nullptr);
  EXPECT_EQ(convertedFCInput->getElementType(ConvertToNode::ResultIdx),
            ElemKind::Float16Ty);
  EXPECT_TRUE(llvm::isa<Placeholder>(convertedFCInput->getInput()));
  EXPECT_EQ(convertedFCInput->getInput().getElementType(), ElemKind::FloatTy);
  EXPECT_EQ(convertedFCInput->getInput().getNode(), input);

  // Check that the weights have been updated to float16.
  auto *convertedFCWeights =
      llvm::dyn_cast<Constant>(convertedFC->getWeights());
  ASSERT_NE(convertedFCWeights, nullptr);
  EXPECT_EQ(convertedFCWeights->getElementType(), ElemKind::Float16Ty);
  EXPECT_EQ(convertedFCWeights, weights);
  origWeights.convertToType(ElemKind::Float16Ty);
  EXPECT_TRUE(origWeights.isEqual(weights->getPayload()));

  // Check that the bias has been duplicated and converted.
  auto *convertedFCBias = llvm::dyn_cast<Constant>(convertedFC->getBias());
  ASSERT_NE(convertedFCBias, nullptr);
  EXPECT_EQ(convertedFCBias->getElementType(), ElemKind::Float16Ty);
  EXPECT_NE(convertedFCBias, bias);
  origBias.convertToType(ElemKind::Float16Ty);
  EXPECT_TRUE(origBias.isEqual(convertedFCBias->getPayload()));

  // Check that the original bias hasn't been altered.
  EXPECT_EQ(bias->getElementType(), ElemKind::FloatTy);
  EXPECT_EQ(saveBias->getInput().getNode(), bias);
}

/// Check that the conversion of placeholder inserts conversion
/// at the right places, and in all the functions.
/// Namely, check that:
/// \verbatim
///    #### F ####
/// Input: Placeholder(float)
/// |   |
/// |   |  Weight: Constant(float)
/// |   |   |   Bias: Constant(float)
/// |   |   |   /
/// |   V   V  V
/// |  FC(float)
/// |   |
/// |   V
/// |  ReLU(float)  Output: Placeholder(float)
/// |   |            |
/// |   |    +-------+
/// |   |   /
/// |   V  V
/// |  Save
/// |
/// |  #### F2 ####
/// |  Output2: Placeholder(float)
/// +-+   /
/// | |  |
/// | V  V
/// | Save
/// |
/// |  #### F3 ####
/// |  Output3: Placeholder(float)
/// |  /
/// V V
/// Save
/// \endverbatim
///
/// Gets converted into:
/// \verbatim
///    #### F ####
/// Input: Placeholder(float16)
/// |   |
/// |   V
/// |ConvertTo(float)
/// |   |
/// |   |  Weight: Constant(float)
/// |   |   |   Bias: Constant(float)
/// |   |   |   /
/// |   V   V  V
/// |  FC(float)
/// |   |
/// |   V
/// |  ReLU(float)  Output: Placeholder(float16)
/// |   |              |
/// |   V              |
/// |ConvertTo(float16)|
/// |   |    +---------+
/// |   |   /
/// |   V  V
/// |  Save
/// |  #### F2 ####
/// +-+
/// | |
/// | V
/// | ConvertTo(float)
/// | |
/// | | Output2: Placeholder(float)
/// | |  |
/// | V  V
/// | Save
/// |
/// |  #### F3 ####
/// V
/// ConvertTo(float)
/// |
/// V
/// ConvertTo(float16)
/// |
/// |  Output3: Placeholder(float16)
/// |  /
/// V V
/// Save
/// \endverbatim
///
/// In particular, the input and output of the network should be modified
/// and the input of the last save node should be converted to the expected
/// output type.
TEST(TypeAToTypeBFunctionConverter, convertPlaceholderFloatToFloat16) {
  Module mod;
  Function *F = mod.createFunction("test");
  Function *F2 = mod.createFunction("test2");
  Function *F3 = mod.createFunction("test3");
  PlaceholderBindings bindings;

  auto *input =
      mod.createPlaceholder(ElemKind::FloatTy, {20, 13}, "Input", false);
  Tensor *inputTensor = bindings.allocate(input);
  inputTensor->getHandle().randomize(-6.0, 6.0, mod.getPRNG());
  Tensor origInput;
  origInput.assign(inputTensor);

  auto *output =
      mod.createPlaceholder(ElemKind::FloatTy, {20, 10}, "Output", false);
  auto *output2 =
      mod.createPlaceholder(ElemKind::FloatTy, {20, 13}, "Output2", false);

  auto *saveOutput2 = F2->createSave("saveOutput2", input, output2);

  auto *output3 =
      mod.createPlaceholder(ElemKind::FloatTy, {20, 13}, "Output2", false);
  auto *saveOutput3 = F3->createSave("saveOutput3", input, output3);

  auto *weights = mod.createConstant(
      mod.uniqueType(ElemKind::FloatTy, {13, 10}), "weights");
  weights->getPayloadMutable().getHandle().randomize(-5.0, 5.0, mod.getPRNG());
  Tensor origWeights;
  origWeights.assign(&weights->getPayload());
  auto *bias =
      mod.createConstant(mod.uniqueType(ElemKind::FloatTy, {10, 20}), "bias");
  bias->getPayloadMutable().getHandle().randomize(-5.0, 5.0, mod.getPRNG());

  TypeRef FCTy = mod.uniqueType(ElemKind::FloatTy, {20, 10});
  auto *FC = F->createFullyConnected("FC", input, weights, bias, FCTy);
  auto *ReLU =
      F->createRELU("ReLU", FC, FC->getType(FullyConnectedNode::ResultIdx));
  auto *result = F->createSave("save", ReLU, output);

  size_t origGraphSize = F->getNodes().size();
  size_t f2OrigGraphSize = F2->getNodes().size();
  size_t f3OrigGraphSize = F3->getNodes().size();

  PrecisionConfiguration precConfig;
  TypeAToTypeBFunctionConverter converter(*F, ElemKind::FloatTy,
                                          ElemKind::Float16Ty, precConfig);
  for (auto *placeholder : mod.getPlaceholders()) {
    if (output2 == placeholder) {
      continue;
    }
    converter.convertPlaceholder(*placeholder, &bindings);
  }

  // We should have 2 more nodes in F:
  // 1 conversion for each conversion for the input of the save node of output
  // 1 conversion from the input to the FC
  EXPECT_EQ(F->getNodes().size(), origGraphSize + 2);
  // Make the save node of F is still in the function and is unchanged.
  EXPECT_TRUE(std::find(F->getNodes().begin(), F->getNodes().end(), *result) !=
              F->getNodes().end());
  EXPECT_EQ(result->getOutput(), output->getOutput());
  // Check that the save is fed from a conversion from float16 to float.
  auto *convertedBackReLURes =
      llvm::dyn_cast<ConvertToNode>(result->getInput());
  ASSERT_NE(convertedBackReLURes, nullptr);
  EXPECT_EQ(convertedBackReLURes->getElementType(ConvertToNode::ResultIdx),
            ElemKind::Float16Ty);
  auto *convertedReLU =
      llvm::dyn_cast<ReluNode>(convertedBackReLURes->getInput());
  ASSERT_NE(convertedReLU, nullptr);
  EXPECT_EQ(convertedReLU->getElementType(ReluNode::ResultIdx),
            ElemKind::FloatTy);

  // Check that the ReLU is fed directly by FC float.
  auto *convertedFC =
      llvm::dyn_cast<FullyConnectedNode>(convertedReLU->getInput());
  ASSERT_NE(convertedFC, nullptr);
  EXPECT_EQ(convertedFC->getElementType(FullyConnectedNode::ResultIdx),
            ElemKind::FloatTy);
  // Check that the input of FC is a convertTo node from "input" from float to
  // Float16Ty.
  auto *convertedFCInput =
      llvm::dyn_cast<ConvertToNode>(convertedFC->getInput());
  ASSERT_NE(convertedFCInput, nullptr);
  EXPECT_EQ(convertedFCInput->getElementType(ConvertToNode::ResultIdx),
            ElemKind::FloatTy);
  EXPECT_TRUE(llvm::isa<Placeholder>(convertedFCInput->getInput()));
  EXPECT_EQ(convertedFCInput->getInput().getElementType(), ElemKind::Float16Ty);
  EXPECT_EQ(convertedFCInput->getInput().getNode(), input);

  // Checks for F2.

  // We should have 1 more node in F2:
  // 1 conversion from the input to the input of the save node

  // Make the save node of F2 is still in the function and is unchanged.
  EXPECT_EQ(F2->getNodes().size(), f2OrigGraphSize + 1);
  EXPECT_TRUE(std::find(F2->getNodes().begin(), F2->getNodes().end(),
                        *saveOutput2) != F2->getNodes().end());
  EXPECT_EQ(saveOutput2->getOutput(), output2->getOutput());

  // Check that the save is fed from a conversion from float16 to float.
  auto *inputToFloat = llvm::dyn_cast<ConvertToNode>(saveOutput2->getInput());
  ASSERT_NE(inputToFloat, nullptr);
  EXPECT_EQ(inputToFloat->getElementType(ConvertToNode::ResultIdx),
            ElemKind::FloatTy);
  // Check that this input is "input".
  auto *inputOfF2 = llvm::dyn_cast<Placeholder>(inputToFloat->getInput());
  ASSERT_NE(inputOfF2, nullptr);
  EXPECT_EQ(inputOfF2, input);

  // Checks for F3.

  // We should have 2 more nodes in F3:
  // 1 conversion from the input to the input of the save node (coming
  //   from the input)
  // 1 conversion from the input to the input of the save node (coming
  //   from the requirement for the output)

  // Make the save node of F3 is still in the function and is unchanged.
  EXPECT_EQ(F3->getNodes().size(), f3OrigGraphSize + 2);
  EXPECT_TRUE(std::find(F3->getNodes().begin(), F3->getNodes().end(),
                        *saveOutput3) != F3->getNodes().end());
  EXPECT_EQ(saveOutput3->getOutput(), output3->getOutput());
  EXPECT_EQ(output3->getElementType(), ElemKind::Float16Ty);

  // Check that the save is fed from a conversion from float16 to float.
  auto *convertOutput3 = llvm::dyn_cast<ConvertToNode>(saveOutput3->getInput());
  ASSERT_NE(convertOutput3, nullptr);
  EXPECT_EQ(convertOutput3->getElementType(ConvertToNode::ResultIdx),
            ElemKind::Float16Ty);

  auto *convertInputFor3 =
      llvm::dyn_cast<ConvertToNode>(convertOutput3->getInput());
  ASSERT_NE(convertInputFor3, nullptr);
  EXPECT_EQ(convertInputFor3->getElementType(ConvertToNode::ResultIdx),
            ElemKind::FloatTy);
  // Check that this input is "input".
  auto *inputOfF3 = llvm::dyn_cast<Placeholder>(convertInputFor3->getInput());
  ASSERT_NE(inputOfF3, nullptr);
  EXPECT_EQ(inputOfF3, input);

  origInput.convertToType(ElemKind::Float16Ty);
  EXPECT_TRUE(origInput.isEqual(*inputTensor));
}

/// Check that the verify doesn't complain when there are
/// noop conversion. This may happen on unoptimized network.
/// E.g.,
/// Input: Placeholder(float)
/// |
/// V
/// OrigConvert: ConvertTo(float16)
/// |
/// V
/// Save
///
/// Now converting the network to float16 will yield:
/// Input: Placeholder(float)
/// |
/// V
/// ConvertTo(float16); convert the input to fp16
/// |
/// V
/// OrigConvert: ConvertTo(float16); <-- now this is a noop conversion.
/// |
/// V
/// Save
TEST(TypeAToTypeBFunctionConverter, convertExistingConversionToNoop) {
  Module mod;
  Function *F = mod.createFunction("test");
  auto *placeholder =
      mod.createPlaceholder(ElemKind::FloatTy, {20, 13}, "Input", false);

  auto *convert =
      F->createConvertTo("convert", placeholder, ElemKind::Float16Ty);
  auto *save = F->createSave("save", convert);

  size_t origSize = F->getNodes().size();

  PrecisionConfiguration precConfig;
  TypeAToTypeBFunctionConverter converter(*F, ElemKind::FloatTy,
                                          ElemKind::Float16Ty, precConfig);
  converter.convert();

  EXPECT_EQ(F->getNodes().size(), origSize + 1);

  auto *convertToSave = llvm::dyn_cast<ConvertToNode>(save->getInput());
  EXPECT_EQ(convertToSave, convert);
  EXPECT_EQ(convert->getElementType(ConvertToNode::ResultIdx),
            ElemKind::Float16Ty);

  auto *addedConversion = llvm::dyn_cast<ConvertToNode>(convert->getInput());
  ASSERT_NE(addedConversion, nullptr);
  // At this point both the input and output of convert are FP16.
  EXPECT_EQ(addedConversion->getElementType(ConvertToNode::ResultIdx),
            ElemKind::Float16Ty);

  EXPECT_EQ(addedConversion->getInput().getNode(), placeholder);
  EXPECT_EQ(placeholder->getElementType(), ElemKind::FloatTy);

  EXPECT_TRUE(F->verify());
}

/// Helper for testing FRWQSLWS FP16 conversion, with and without FP16
/// accumulation based on \p forceFP16AccumSLS.
static void testConvertFRWQSLWS(bool forceFP16AccumSLS) {
  Module mod;
  Function *F = mod.createFunction("test");
  Tensor data(ElemKind::FloatTy, {3, 1});
  data.getHandle() = {
      2.0,
      -0.5,
      13,
  };

  Constant *weights = mod.createConstant(ElemKind::FloatTy, {8}, "weights");

  Placeholder *indices =
      mod.createPlaceholder(ElemKind::Int64ITy, {8}, "indices",
                            /* isTrainable */ false);
  Placeholder *lengths =
      mod.createPlaceholder(ElemKind::Int32ITy, {4}, "lengths",
                            /* isTrainable */ false);
  auto *R = F->createFusedRowwiseQuantizedSparseLengthsWeightedSum(
      "RQSLWS", data, weights, indices, lengths, ElemKind::UInt8FusedQTy);
  SaveNode *S = F->createSave("save", R);

  size_t origSize = F->getNodes().size();

  CompilationContext cctx;
  PrecisionConfiguration &precConfig = cctx.precisionConfig;
  precConfig.convertToFP16 = true;
  precConfig.convertFusedToFP16 = true;
  precConfig.forceFP16AccumSLS = forceFP16AccumSLS;
  transformForPrecisionMode(MockBackend(), F, cctx);

  // Should have added convert nodes for the Data, Weights, and Result. Data
  // and Weights ConvertTo nodes should have been merged in, while Result stil
  // has a ConvertTo.
  EXPECT_EQ(F->getNodes().size(), origSize + 1);

  auto *convertResult = llvm::dyn_cast<ConvertToNode>(S->getInput());
  ASSERT_NE(convertResult, nullptr);
  EXPECT_EQ(convertResult->getResult().getElementType(), ElemKind::FloatTy);

  auto *SLWS =
      llvm::dyn_cast<FusedRowwiseQuantizedSparseLengthsWeightedSumNode>(
          convertResult->getInput());
  ASSERT_NE(SLWS, nullptr);
  EXPECT_EQ(SLWS->getResult().getElementType(), ElemKind::Float16Ty);
  EXPECT_EQ(SLWS->getUseFP16Accumulation(), forceFP16AccumSLS);

  EXPECT_EQ(SLWS->getData().getElementType(), ElemKind::UInt8FusedFP16QTy);
  EXPECT_EQ(SLWS->getWeights().getElementType(), ElemKind::Float16Ty);

  EXPECT_TRUE(F->verify());
}

/// Test conversion of a FusedRowwiseQuantizedSparseLengthsWeightedSumNode to
/// FP16, instead of creating it directly. Use FP16 accumulation.
TEST(TypeAToTypeBFunctionConverter, convertFRWQSLWS_FP16Accum) {
  testConvertFRWQSLWS(/* forceFP16AccumSLS */ true);
}

/// Test conversion of a FusedRowwiseQuantizedSparseLengthsWeightedSumNode to
/// FP16, instead of creating it directly. Do not use FP16 accumulation; note
/// that conversion by default uses FP32 accumulation.
TEST(TypeAToTypeBFunctionConverter, convertFRWQSLWS_FP32Accum) {
  testConvertFRWQSLWS(/* forceFP16AccumSLS */ false);
}

/// Test skipping conversion of a
/// FusedRowwiseQuantizedSparseLengthsWeightedSumNode to FP16.
TEST(TypeAToTypeBFunctionConverter, skipConvertingFRWQSLWS) {
  Module mod;
  Function *F = mod.createFunction("test");
  Tensor data(ElemKind::FloatTy, {3, 1});
  data.getHandle() = {
      2.0,
      -0.5,
      13,
  };

  Constant *weights = mod.createConstant(ElemKind::FloatTy, {8}, "weights");

  Placeholder *indices =
      mod.createPlaceholder(ElemKind::Int64ITy, {8}, "indices",
                            /* isTrainable */ false);
  Placeholder *lengths =
      mod.createPlaceholder(ElemKind::Int32ITy, {4}, "lengths",
                            /* isTrainable */ false);
  auto *R = F->createFusedRowwiseQuantizedSparseLengthsWeightedSum(
      "RQSLWS", data, weights, indices, lengths, ElemKind::UInt8FusedQTy);
  SaveNode *S = F->createSave("save", R);

  size_t origSize = F->getNodes().size();

  PrecisionConfiguration precConfig;
  precConfig.convertToFP16 = true;
  precConfig.convertFusedToFP16 = true;
  precConfig.precisionModeKindSet.insert(
      Kinded::Kind::FusedRowwiseQuantizedSparseLengthsWeightedSumNodeKind);
  convertFunctionToFloat16(F, precConfig);

  // Should have done nothing since we skipped its conversion. Check the
  // Function is the same as before.
  EXPECT_EQ(F->getNodes().size(), origSize);

  auto *SLWS =
      llvm::dyn_cast<FusedRowwiseQuantizedSparseLengthsWeightedSumNode>(
          S->getInput());
  ASSERT_EQ(SLWS, R);

  auto *origData = llvm::dyn_cast<Constant>(SLWS->getData());
  ASSERT_EQ(origData, R->getData().getNode());
  EXPECT_EQ(origData->getOutput().getElementType(), ElemKind::UInt8FusedQTy);

  auto *origWeights = llvm::dyn_cast<Constant>(SLWS->getWeights());
  ASSERT_EQ(origWeights, weights);
  EXPECT_EQ(origWeights->getOutput().getElementType(), ElemKind::FloatTy);

  EXPECT_TRUE(F->verify());
}

static void
convertOnlyFloat16Ty(PrecisionConfiguration::Float16Format float16Format) {
  Module mod;
  Function *F = mod.createFunction("test");
  Tensor data(ElemKind::FloatTy, {3, 1});
  data.getHandle() = {
      2.0,
      -0.5,
      13,
  };

  Constant *weights = mod.createConstant(ElemKind::FloatTy, {8}, "weights");

  Placeholder *indices =
      mod.createPlaceholder(ElemKind::Int64ITy, {8}, "indices",
                            /* isTrainable */ false);
  Placeholder *lengths =
      mod.createPlaceholder(ElemKind::Int32ITy, {4}, "lengths",
                            /* isTrainable */ false);
  auto *R = F->createFusedRowwiseQuantizedSparseLengthsWeightedSum(
      "RQSLWS", data, weights, indices, lengths, ElemKind::UInt8FusedQTy);
  SaveNode *S = F->createSave("save", R);

  size_t origSize = F->getNodes().size();

  PrecisionConfiguration precConfig;
  precConfig.convertToFP16 = true;
  precConfig.convertFusedToFP16 = false;
  precConfig.float16Format = float16Format;
  convertFunctionToFloat16(F, precConfig);

  ElemKind convertedElementType =
      PrecisionConfiguration::getElementType(float16Format);

  // Should have added convert nodes for the weights and results.
  EXPECT_EQ(F->getNodes().size(), origSize + 2);

  auto *convertResult = llvm::dyn_cast<ConvertToNode>(S->getInput());
  ASSERT_NE(convertResult, nullptr);
  EXPECT_EQ(convertResult->getResult().getElementType(), ElemKind::FloatTy);

  auto *SLWS =
      llvm::dyn_cast<FusedRowwiseQuantizedSparseLengthsWeightedSumNode>(
          convertResult->getInput());
  ASSERT_NE(SLWS, nullptr);
  EXPECT_EQ(SLWS->getResult().getElementType(), convertedElementType);

  auto *origData = llvm::dyn_cast<Constant>(SLWS->getData());
  ASSERT_NE(origData, nullptr);
  EXPECT_EQ(origData->getOutput().getElementType(), ElemKind::UInt8FusedQTy);

  auto *convertWeights = llvm::dyn_cast<ConvertToNode>(SLWS->getWeights());
  ASSERT_NE(convertWeights, nullptr);
  EXPECT_EQ(convertWeights->getResult().getElementType(), convertedElementType);

  auto *origWeights = llvm::dyn_cast<Constant>(convertWeights->getInput());
  ASSERT_NE(origWeights, nullptr);
  EXPECT_EQ(origWeights->getOutput().getElementType(), ElemKind::FloatTy);
  EXPECT_EQ(weights, origWeights);

  EXPECT_TRUE(F->verify());
}

/// Test conversion of only FP16 inputs of Node and not UInt8FusedQTy.
TEST(TypeAToTypeBFunctionConverter, convertOnlyFP16Ty) {
  convertOnlyFloat16Ty(PrecisionConfiguration::Float16Format::FP16);
}

/// Test conversion of only BFloat16 inputs of Node and not UInt8FusedQTy.
TEST(TypeAToTypeBFunctionConverter, convertOnlyBFloat16Ty) {
  convertOnlyFloat16Ty(PrecisionConfiguration::Float16Format::BFloat16);
}

/// Test conversion of only UInt8FusedQTy inputs of Node and not Float16Ty.
TEST(TypeAToTypeBFunctionConverter, convertOnlyUInt8FusedQTy) {
  Module mod;
  Function *F = mod.createFunction("test");
  Tensor data(ElemKind::FloatTy, {3, 1});
  data.getHandle() = {
      2.0,
      -0.5,
      13,
  };

  Constant *weights = mod.createConstant(ElemKind::FloatTy, {8}, "weights");

  Placeholder *indices =
      mod.createPlaceholder(ElemKind::Int64ITy, {8}, "indices",
                            /* isTrainable */ false);
  Placeholder *lengths =
      mod.createPlaceholder(ElemKind::Int32ITy, {4}, "lengths",
                            /* isTrainable */ false);
  auto *R = F->createFusedRowwiseQuantizedSparseLengthsWeightedSum(
      "RQSLWS", data, weights, indices, lengths, ElemKind::UInt8FusedQTy);
  SaveNode *S = F->createSave("save", R);

  size_t origSize = F->getNodes().size();

  PrecisionConfiguration precConfig;
  precConfig.convertToFP16 = false;
  precConfig.convertFusedToFP16 = true;
  convertFunctionToFloat16(F, precConfig);

  // Should have added a convert nodes for the data.
  EXPECT_EQ(F->getNodes().size(), origSize + 1);

  auto *SLWS =
      llvm::dyn_cast<FusedRowwiseQuantizedSparseLengthsWeightedSumNode>(
          S->getInput());
  ASSERT_EQ(SLWS, R);

  auto *convertData = llvm::dyn_cast<ConvertToNode>(SLWS->getData());
  ASSERT_NE(convertData, nullptr);
  EXPECT_EQ(convertData->getResult().getElementType(),
            ElemKind::UInt8FusedFP16QTy);

  auto *origData = llvm::dyn_cast<Constant>(convertData->getInput());
  ASSERT_NE(origData, nullptr);
  EXPECT_EQ(origData->getOutput().getElementType(), ElemKind::UInt8FusedQTy);

  auto *origWeights = llvm::dyn_cast<Constant>(SLWS->getWeights());
  ASSERT_EQ(origWeights, weights);
  EXPECT_EQ(origWeights->getOutput().getElementType(), ElemKind::FloatTy);

  EXPECT_TRUE(F->verify());
}

static void convertWithoutClipAroundNonNumericNodes(
    PrecisionConfiguration::Float16Format float16Format) {
  Module mod;
  Function *F = mod.createFunction("test");
  const dim_t dims[] = {1, 5, 10, 15};
  const dim_t dimsReshape[] = {10, 10, 15};
  Node *I0 = mod.createPlaceholder(ElemKind::FloatTy, dims, "i0", false);
  Node *I1 = mod.createPlaceholder(ElemKind::FloatTy, dims, "i1", false);
  Node *I2 = mod.createPlaceholder(ElemKind::Int32ITy, {2, 2, 2}, "i2", false);
  Node *CN = F->createConcat("concat", {I0, I1}, 1);
  Node *R = F->createReshape("reshape", CN, dimsReshape);
  Node *S = F->createSlice("slice", R, {0, 0, 0}, {5, 5, 5});
  Node *G = F->createGather("gather", S, I2);
  F->createSave("ret", G);

  PrecisionConfiguration precConfig;
  precConfig.convertToFP16 = true;
  precConfig.clipFP16 = true;
  precConfig.float16Format = float16Format;
  convertFunctionToFloat16(F, precConfig);

  int numClips = 0;
  int numConvertTos = 0;
  for (auto &n : F->getNodes()) {
    if (n.getKind() == Kinded::Kind::ClipNodeKind) {
      ++numClips;
    } else if (n.getKind() == Kinded::Kind::ConvertToNodeKind) {
      ++numConvertTos;
    }
  }

  EXPECT_EQ(9, numConvertTos);
  EXPECT_EQ(0, numClips);

  EXPECT_TRUE(F->verify());
}

// Test that we don't insert Clips around non-numeric nodes.
TEST(TypeAToTypeBFunctionConverter,
     convertWithFP16WithoutClipAroundNonNumericNodes) {
  convertWithoutClipAroundNonNumericNodes(
      PrecisionConfiguration::Float16Format::FP16);
}

// Test that we don't insert Clips around non-numeric nodes.
TEST(TypeAToTypeBFunctionConverter,
     convertWithBFloat16WithoutClipAroundNonNumericNodes) {
  convertWithoutClipAroundNonNumericNodes(
      PrecisionConfiguration::Float16Format::BFloat16);
}

static void convertWithoutClipAfterTanhOrSigmoid(
    PrecisionConfiguration::Float16Format float16Format) {
  Module mod;
  Function *F = mod.createFunction("test");
  const dim_t dims[] = {10, 20};
  const dim_t dims2[] = {10, 30};
  Node *I0 = mod.createPlaceholder(ElemKind::FloatTy, dims, "i0", false);
  Node *I1 = mod.createPlaceholder(ElemKind::FloatTy, dims2, "i1", false);
  Node *T = F->createTanh("tanh", {I0});
  Node *S = F->createSigmoid("sigmoid", {I1});
  Node *CN = F->createConcat("concat", {T, S}, 1);
  F->createSave("ret", CN);

  PrecisionConfiguration precConfig;
  precConfig.convertToFP16 = true;
  precConfig.clipFP16 = true;
  precConfig.float16Format = float16Format;
  convertFunctionToFloat16(F, precConfig);

  int numClips = 0;
  int numConvertTos = 0;
  for (auto &n : F->getNodes()) {
    if (n.getKind() == Kinded::Kind::ClipNodeKind) {
      ++numClips;
    } else if (n.getKind() == Kinded::Kind::ConvertToNodeKind) {
      ++numConvertTos;
    }
  }

  EXPECT_EQ(7, numConvertTos);
  EXPECT_EQ(2, numClips);

  EXPECT_TRUE(F->verify());
}

// Test that we don't insert Clips at the output of Tanh or Sigmoid
TEST(TypeAToTypeBFunctionConverter,
     convertWithFP16WithoutClipAfterTanhOrSigmoid) {
  convertWithoutClipAfterTanhOrSigmoid(
      PrecisionConfiguration::Float16Format::FP16);
}

// Test that we don't insert Clips at the output of Tanh or Sigmoid
TEST(TypeAToTypeBFunctionConverter,
     convertWithBFloat16WithoutClipAfterTanhOrSigmoid) {
  convertWithoutClipAfterTanhOrSigmoid(
      PrecisionConfiguration::Float16Format::BFloat16);
}

static void convertWithoutClipAfterFp16ConvertTo(
    PrecisionConfiguration::Float16Format float16Format) {
  Module mod;
  Function *F = mod.createFunction("test");
  const dim_t dims[] = {10, 20};
  const dim_t dims2[] = {10, 30};
  Node *I0 = mod.createPlaceholder(ElemKind::Float16Ty, dims, "i0", false);
  Node *I1 = mod.createPlaceholder(ElemKind::Float16Ty, dims2, "i1", false);
  Node *T = F->createConvertTo("c1", {I0}, ElemKind::FloatTy);
  Node *S = F->createConvertTo("c2", {I1}, ElemKind::FloatTy);
  Node *CN = F->createConcat("concat", {T, S}, 1);
  F->createSave("ret", CN);

  PrecisionConfiguration precConfig;
  precConfig.convertToFP16 = true;
  precConfig.clipFP16 = true;
  precConfig.float16Format = float16Format;
  convertFunctionToFloat16(F, precConfig);

  int numClips = 0;
  for (auto &n : F->getNodes()) {
    if (n.getKind() == Kinded::Kind::ClipNodeKind) {
      ++numClips;
    }
  }

  EXPECT_EQ(0, numClips);

  EXPECT_TRUE(F->verify());
}

// Test that we don't insert Clips at the output of ConvertTo if its input is
// fp16.
TEST(TypeAToTypeBFunctionConverter,
     convertWithFP16WithoutClipAfterFp16ConvertTo) {
  convertWithoutClipAfterFp16ConvertTo(
      PrecisionConfiguration::Float16Format::FP16);
}

// Test that we don't insert Clips at the output of ConvertTo if its input is
// bfloat16.
TEST(TypeAToTypeBFunctionConverter,
     convertWithBFloat16WithoutClipAfterFp16ConvertTo) {
  convertWithoutClipAfterFp16ConvertTo(
      PrecisionConfiguration::Float16Format::BFloat16);
}

// Test that we only insert clips for outputs.
static void
checkConvertOnlyOutputs(PrecisionConfiguration::Float16Format float16Format) {
  Module mod;
  Function *F = mod.createFunction("test");
  Node *I = mod.createPlaceholder(ElemKind::FloatTy, {10}, "i", false);
  ReluNode *RN = F->createRELU("relu", I);
  SaveNode *SN = F->createSave("ret", RN);

  PrecisionConfiguration precConfig;
  precConfig.convertToFP16 = true;
  precConfig.clipFP16 = true;
  precConfig.clipFP16SkipInputs = true;
  precConfig.convertPlaceholdersToFP16 = true;
  precConfig.convertConstantsToFP16 = true;
  precConfig.float16Format = float16Format;
  convertFunctionToFloat16(F, precConfig);

  ElemKind convertedElementType =
      PrecisionConfiguration::getElementType(float16Format);

  // PH -> ConvertToFP16 -> ConvertToFP32 -> ConvertToFP16 -> Relu ->
  // Clip -> ConvertToFP32 -> Save

  ConvertToNode *convertRN = llvm::dyn_cast<ConvertToNode>(SN->getInput());
  ASSERT_TRUE(convertRN);
  EXPECT_EQ(convertRN->getResult().getType()->getElementType(),
            ElemKind::FloatTy);
  ClipNode *clipRN = llvm::dyn_cast<ClipNode>(convertRN->getInput());
  convertRN->getInput().getNode()->dump();
  ASSERT_TRUE(clipRN);
  ASSERT_TRUE(clipRN->getInput() == RN->getResult());
  ConvertToNode *convert32To16 = llvm::dyn_cast<ConvertToNode>(RN->getInput());
  ASSERT_TRUE(convert32To16);
  EXPECT_EQ(convert32To16->getResult().getType()->getElementType(),
            convertedElementType);
  ConvertToNode *convert16To32 =
      llvm::dyn_cast<ConvertToNode>(convert32To16->getInput());
  ASSERT_TRUE(convert16To32);
  EXPECT_EQ(convert16To32->getResult().getType()->getElementType(),
            ElemKind::FloatTy);
  ConvertToNode *convertPH =
      llvm::dyn_cast<ConvertToNode>(convert16To32->getInput());
  ASSERT_TRUE(convertPH);
  EXPECT_EQ(convertPH->getResult().getType()->getElementType(),
            convertedElementType);

  EXPECT_TRUE(F->verify());
}

// Test that we only insert clips for outputs.
TEST(TypeAToTypeBFunctionConverter, checkWithFP16ConvertOnlyOutputs) {
  checkConvertOnlyOutputs(PrecisionConfiguration::Float16Format::FP16);
}

// Test that we only insert clips for outputs.
TEST(TypeAToTypeBFunctionConverter, checkWithBFloat16ConvertOnlyOutputs) {
  checkConvertOnlyOutputs(PrecisionConfiguration::Float16Format::BFloat16);
}

// Test that we only insert clips for outputs.
static void
checkConvertClipStorage(PrecisionConfiguration::Float16Format float16Format) {
  Module mod;
  Function *F = mod.createFunction("test");
  Node *PH = mod.createPlaceholder(ElemKind::FloatTy, {10}, "ph", false);
  Node *C = mod.createConstant(ElemKind::FloatTy, {10, 1}, "c");
  SaveNode *SPH = F->createSave("ret", PH);
  SaveNode *SC = F->createSave("ret", C);

  PrecisionConfiguration precConfig;
  precConfig.convertToFP16 = true;
  precConfig.clipFP16 = true;
  precConfig.clipFP16SkipInputs = true;
  precConfig.convertPlaceholdersToFP16 = true;
  precConfig.convertConstantsToFP16 = true;
  precConfig.float16Format = float16Format;
  convertFunctionToFloat16(F, precConfig);

  ConvertToNode *convertFP32PH = llvm::dyn_cast<ConvertToNode>(SPH->getInput());
  ASSERT_TRUE(convertFP32PH);
  ConvertToNode *convertFP16PH =
      llvm::dyn_cast<ConvertToNode>(convertFP32PH->getInput());
  ASSERT_TRUE(convertFP16PH);

  ConvertToNode *convertFP32C = llvm::dyn_cast<ConvertToNode>(SC->getInput());
  ASSERT_TRUE(convertFP32C);
  ClipNode *clipC = llvm::dyn_cast<ClipNode>(convertFP32C->getInput());
  ASSERT_TRUE(clipC);
  ConvertToNode *convertFP16C =
      llvm::dyn_cast<ConvertToNode>(clipC->getInput());
  ASSERT_TRUE(convertFP16C);

  EXPECT_TRUE(F->verify());
}

// Test that we only insert clips for outputs.
TEST(TypeAToTypeBFunctionConverter, checkWithFP16ConvertClipStorage) {
  checkConvertClipStorage(PrecisionConfiguration::Float16Format::FP16);
}

// Test that we only insert clips for outputs.
TEST(TypeAToTypeBFunctionConverter, checkWithBFloat16ConvertClipStorage) {
  checkConvertClipStorage(PrecisionConfiguration::Float16Format::BFloat16);
}

/// Check that quantized FC with FP32 bias doesn't have bias converted to FP16.
TEST(TypeAToTypeBFunctionConverter, DoNotConvertFloatBiasWithIntInput) {
  Module mod;
  Function *F = mod.createFunction("test");
  PlaceholderBindings bindings;

  auto *input = mod.createPlaceholder(ElemKind::Int8QTy, {3, 8}, 0.05, -2,
                                      "input", false);
  auto *weight = mod.createConstant(ElemKind::Int8QTy, {8, 10}, 0.02, 3, "w");
  auto *bias = mod.createConstant(ElemKind::FloatTy, {10}, "w");

  auto *FC = F->createFullyConnected("FC", input, weight, bias);
  F->createSave("save", FC);

  std::string origGraph = F->toString();

  PrecisionConfiguration precConfig;
  TypeAToTypeBFunctionConverter converter(*F, ElemKind::FloatTy,
                                          ElemKind::Float16Ty, precConfig);
  converter.convert();

  EXPECT_EQ(origGraph, F->toString());
}

/// Create a FRWQSLWS node with data type \p fusedKind and shape \p row and \p
/// col, check if its data can be properly converted from UInt8FP16QTy to
/// UInt8FusedQTy, or from UInt4FP16QTy to UInt4FusedQTy, and its indices can be
/// properly coverted to Int64.
static void testFRWQSLWSDataIndicesConvert(ElemKind fusedKind, dim_t row,
                                           dim_t col) {
  EXPECT_LT(row, 100);
  EXPECT_LT(col, 100);

  Module mod;
  Function *F = mod.createFunction("test");
  Tensor data(ElemKind::FloatTy, {row, col});
  auto dataH = data.getHandle();
  for (dim_t i = 0; i < row; i++) {
    for (dim_t j = 0; j < col; j++) {
      dataH.at({i, j}) = 2.0 * i + 1.0 * j;
    }
  }

  Constant *weights = mod.createConstant(ElemKind::FloatTy, {8}, "weights");
  Placeholder *indices =
      mod.createPlaceholder(ElemKind::Int32ITy, {8}, "indices",
                            /* isTrainable */ false);
  Placeholder *lengths =
      mod.createPlaceholder(ElemKind::Int32ITy, {4}, "lengths",
                            /* isTrainable */ false);
  auto *R = F->createFusedRowwiseQuantizedSparseLengthsWeightedSum(
      "RQSLWS", data, weights, indices, lengths, fusedKind);
  SaveNode *S = F->createSave("save", R);

  size_t origSize = F->getNodes().size();
  CompilationContext cctx;
  PrecisionConfiguration &precConfig = cctx.precisionConfig;
  precConfig.convert4BitFusedToFP32 = true;
  precConfig.convert8BitFusedToFP32 = true;
  precConfig.convertIndicesToInt64 = true;
  precConfig.forceFP16AccumSLS = false;

  transformForPrecisionMode(MockBackend(), F, cctx);
  // Should have added ConvertTo nodes for the Data and indices.
  EXPECT_EQ(F->getNodes().size(), origSize + 2);

  optimize(F, CompilationMode::Infer);
  // Since data is a constant, after optimization, const folding should be
  // applied and a new data is created. Therefore, only 1 ConverTo node is left
  // for indices.
  EXPECT_EQ(F->getNodes().size(), origSize + 1);

  auto *SLWS =
      llvm::dyn_cast<FusedRowwiseQuantizedSparseLengthsWeightedSumNode>(
          S->getInput());
  ASSERT_NE(SLWS, nullptr);
  if (fusedKind == ElemKind::UInt8FusedFP16QTy) {
    EXPECT_EQ(SLWS->getData().getElementType(), ElemKind::UInt8FusedQTy);
  } else {
    EXPECT_EQ(SLWS->getData().getElementType(), ElemKind::UInt4FusedQTy);
  }
  EXPECT_EQ(SLWS->getIndices().getElementType(), ElemKind::Int64ITy);

  EXPECT_TRUE(F->verify());
}

/// Testing converting UInt8FusedFP16QTy to UInt8FusedQTy.
TEST(TypeAToTypeBFunctionConverter, FRWLWSConvert8Bit) {
  testFRWQSLWSDataIndicesConvert(ElemKind::UInt8FusedFP16QTy, 10, 10);
}

/// Testing converting UInt4FusedFP16QTy to UInt4FusedQTy.
TEST(TypeAToTypeBFunctionConverter, FRWLWSConvert4Bit) {
  testFRWQSLWSDataIndicesConvert(ElemKind::UInt4FusedFP16QTy, 10, 10);
}
