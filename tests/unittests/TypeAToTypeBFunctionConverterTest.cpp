/**
 * Copyright (c) 2018-present, Facebook, Inc.
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

#include "glow/Converter/TypeAToTypeBFunctionConverter.h"

#include "glow/Backend/Backend.h"
#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Optimizer/GraphOptimizer/GraphOptimizer.h"

#include "llvm/Support/Casting.h"

#include "gtest/gtest.h"

using namespace glow;

struct AllBackends : public ::testing::TestWithParam<std::string> {
protected:
  ExecutionEngine EE_{GetParam()};
};

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
TEST_P(AllBackends, SimpleOneUseConversionFloatToFloat16) {
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

  TypeAToTypeBFunctionConverter converter(*F, ElemKind::FloatTy,
                                          ElemKind::Float16Ty);
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
TEST_P(AllBackends, SimpleChainOfComputationConversionFloatToFloat16) {
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

  TypeAToTypeBFunctionConverter converter(*F, ElemKind::FloatTy,
                                          ElemKind::Float16Ty);
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

/// Check that the conversion honor the doNotConvertKinds set (here ReLU)
/// for a graph with a simple chain of computation.
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
TEST_P(AllBackends, DoNotConvertReLUConversionFloatToFloat16) {
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

  KindSet doNotConvertKinds;
  doNotConvertKinds.insert(Kinded::Kind::ReluNodeKind);
  TypeAToTypeBFunctionConverter converter(
      *F, ElemKind::FloatTy, ElemKind::Float16Ty, &doNotConvertKinds);
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
TEST_P(AllBackends, int64IConversionFloatToFloat16) {
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

  TypeAToTypeBFunctionConverter converter(*F, ElemKind::FloatTy,
                                          ElemKind::Float16Ty);
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
TEST_P(AllBackends, OptimizeMiddleConversionsFloatToFloat16) {
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
      mod.createConstant(mod.uniqueType(ElemKind::FloatTy, {10, 20}), "bias");
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

  TypeAToTypeBFunctionConverter converter(*F, ElemKind::FloatTy,
                                          ElemKind::Float16Ty);
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
TEST_P(AllBackends, convertPlaceholderFloatToFloat16) {
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

  TypeAToTypeBFunctionConverter converter(*F, ElemKind::FloatTy,
                                          ElemKind::Float16Ty);
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
TEST_P(AllBackends, convertExistingConversionToNoop) {
  Module mod;
  Function *F = mod.createFunction("test");
  auto *placeholder =
      mod.createPlaceholder(ElemKind::FloatTy, {20, 13}, "Input", false);

  TypeRef outTy = mod.uniqueType(ElemKind::Float16Ty, placeholder->dims());
  auto *convert = F->createConvertTo("convert", placeholder, outTy);
  auto *save = F->createSave("save", convert);

  size_t origSize = F->getNodes().size();

  TypeAToTypeBFunctionConverter converter(*F, ElemKind::FloatTy,
                                          ElemKind::Float16Ty);
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

INSTANTIATE_TEST_CASE_P(Interpreter, AllBackends,
                        ::testing::Values("Interpreter"));

#ifdef GLOW_WITH_CPU
INSTANTIATE_TEST_CASE_P(CPU, AllBackends, ::testing::Values("CPU"));
#endif // GLOW_WITH_CPU

#ifdef GLOW_WITH_OPENCL
INSTANTIATE_TEST_CASE_P(OpenCL, AllBackends, ::testing::Values("OpenCL"));
#endif // GLOW_WITH_OPENCL
