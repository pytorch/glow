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

#include "glow/Backends/Backend.h"
#include "glow/ExecutionEngine/ExecutionEngine.h"

#include "llvm/Support/Casting.h"

#include "gtest/gtest.h"

using namespace glow;

struct AllBackends : public ::testing::TestWithParam<BackendKind> {
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
  Context ctx;

  auto *input =
      mod.createPlaceholder(ElemKind::FloatTy, {20, 13}, "Input", false);
  auto *output =
      mod.createPlaceholder(ElemKind::FloatTy, {20, 10}, "Output", false);

  auto *FC = F->createFullyConnected(ctx, "FC", input, 10);
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
  EXPECT_EQ(result->getOutput(), NodeValue(output, 0));
  // Check that the save is fed from a conversion from float16 to float.
  auto *convertedBackFCRes = llvm::dyn_cast<ConvertToNode>(result->getInput());
  ASSERT_NE(convertedBackFCRes, nullptr);
  EXPECT_EQ(convertedBackFCRes->getElementType(0), ElemKind::FloatTy);
  auto *convertedFC =
      llvm::dyn_cast<FullyConnectedNode>(convertedBackFCRes->getInput());
  ASSERT_NE(convertedFC, nullptr);
  EXPECT_EQ(convertedFC->getElementType(0), ElemKind::Float16Ty);
  // Check that all the input of FC are convertTo node with from float to
  // Float16Ty.
  for (unsigned idx = 0, end = convertedFC->getNumInputs(); idx != end; ++idx) {
    auto *convertedFCInput =
        llvm::dyn_cast<ConvertToNode>(convertedFC->getNthInput(idx));
    ASSERT_NE(convertedFCInput, nullptr);
    EXPECT_EQ(convertedFCInput->getElementType(0), ElemKind::Float16Ty);
    EXPECT_TRUE(llvm::isa<Placeholder>(convertedFCInput->getInput()));
    EXPECT_EQ(convertedFCInput->getInput().getElementType(), ElemKind::FloatTy);
  }
  // At this point we know the input of FC is convertTo(placeholder).
  // Check that this placeholder is the expected input.
  EXPECT_EQ(convertedFC->getInput().getNode()->getNthInput(0).getNode(), input);
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
  Context ctx;

  auto *input =
      mod.createPlaceholder(ElemKind::FloatTy, {20, 13}, "Input", false);
  auto *output =
      mod.createPlaceholder(ElemKind::FloatTy, {20, 10}, "Output", false);

  auto *FC = F->createFullyConnected(ctx, "FC", input, 10);
  auto *ReLU = F->createRELU("ReLU", FC, FC->getType(0));
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
  EXPECT_EQ(result->getOutput(), NodeValue(output, 0));
  // Check that the save is fed from a conversion from float16 to float.
  auto *convertedBackReLURes =
      llvm::dyn_cast<ConvertToNode>(result->getInput());
  ASSERT_NE(convertedBackReLURes, nullptr);
  EXPECT_EQ(convertedBackReLURes->getElementType(0), ElemKind::FloatTy);
  auto *convertedReLU =
      llvm::dyn_cast<ReluNode>(convertedBackReLURes->getInput());
  ASSERT_NE(convertedReLU, nullptr);
  EXPECT_EQ(convertedReLU->getElementType(0), ElemKind::Float16Ty);

  // Check that the ReLU is fed from a conversion from float to float16.
  auto *convertedToReLUInput =
      llvm::dyn_cast<ConvertToNode>(convertedReLU->getInput());
  ASSERT_NE(convertedToReLUInput, nullptr);
  EXPECT_EQ(convertedToReLUInput->getElementType(0), ElemKind::Float16Ty);

  // Check that this conversion is fed from a conversion from float16 to float.
  auto *convertedBackFCRes =
      llvm::dyn_cast<ConvertToNode>(convertedToReLUInput->getInput());
  ASSERT_NE(convertedBackFCRes, nullptr);
  EXPECT_EQ(convertedBackFCRes->getElementType(0), ElemKind::FloatTy);
  // Check that this conversion comes from the float16 FC node.
  auto *convertedFC =
      llvm::dyn_cast<FullyConnectedNode>(convertedBackFCRes->getInput());
  ASSERT_NE(convertedFC, nullptr);
  EXPECT_EQ(convertedFC->getElementType(0), ElemKind::Float16Ty);
  // Check that all the input of FC are convertTo node with from float to
  // Float16Ty.
  for (unsigned idx = 0, end = convertedFC->getNumInputs(); idx != end; ++idx) {
    auto *convertedFCInput =
        llvm::dyn_cast<ConvertToNode>(convertedFC->getNthInput(idx));
    ASSERT_NE(convertedFCInput, nullptr);
    EXPECT_EQ(convertedFCInput->getElementType(0), ElemKind::Float16Ty);
    EXPECT_TRUE(llvm::isa<Placeholder>(convertedFCInput->getInput()));
    EXPECT_EQ(convertedFCInput->getInput().getElementType(), ElemKind::FloatTy);
  }
  // At this point we know the input of FC is convertTo(placeholder).
  // Check that this placeholder is the expected input.
  EXPECT_EQ(convertedFC->getInput().getNode()->getNthInput(0).getNode(), input);
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
  Context ctx;

  auto *input =
      mod.createPlaceholder(ElemKind::FloatTy, {20, 13}, "Input", false);
  auto *output =
      mod.createPlaceholder(ElemKind::FloatTy, {20, 10}, "Output", false);

  auto *FC = F->createFullyConnected(ctx, "FC", input, 10);
  auto *ReLU = F->createRELU("ReLU", FC, FC->getType(0));
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
  EXPECT_EQ(result->getOutput(), NodeValue(output, 0));
  // Check that the save is fed from a conversion from float16 to float.
  auto *resultInput = llvm::dyn_cast<ReluNode>(result->getInput());
  ASSERT_NE(resultInput, nullptr);
  EXPECT_EQ(resultInput->getElementType(0), ElemKind::FloatTy);
  EXPECT_EQ(resultInput, ReLU);

  // Check that the ReLU is fed from a conversion from float16 to float.
  auto *convertedToReLUInput = llvm::dyn_cast<ConvertToNode>(ReLU->getInput());
  ASSERT_NE(convertedToReLUInput, nullptr);
  EXPECT_EQ(convertedToReLUInput->getElementType(0), ElemKind::FloatTy);

  // Check that this conversion comes from the float16 FC node.
  auto *convertedFC =
      llvm::dyn_cast<FullyConnectedNode>(convertedToReLUInput->getInput());
  ASSERT_NE(convertedFC, nullptr);
  EXPECT_EQ(convertedFC->getElementType(0), ElemKind::Float16Ty);
  // Check that all the input of FC are convertTo node with from float to
  // Float16Ty.
  for (unsigned idx = 0, end = convertedFC->getNumInputs(); idx != end; ++idx) {
    auto *convertedFCInput =
        llvm::dyn_cast<ConvertToNode>(convertedFC->getNthInput(idx));
    ASSERT_NE(convertedFCInput, nullptr);
    EXPECT_EQ(convertedFCInput->getElementType(0), ElemKind::Float16Ty);
    EXPECT_TRUE(llvm::isa<Placeholder>(convertedFCInput->getInput()));
    EXPECT_EQ(convertedFCInput->getInput().getElementType(), ElemKind::FloatTy);
  }
  // At this point we know the input of FC is convertTo(placeholder).
  // Check that this placeholder is the expected input.
  EXPECT_EQ(convertedFC->getInput().getNode()->getNthInput(0).getNode(), input);
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
  Context ctx;

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
  EXPECT_EQ(result->getOutput(), NodeValue(output, 0));
  // Check that the save is fed from a conversion from float16 to float.
  auto *convertedBackTopKRes =
      llvm::dyn_cast<ConvertToNode>(result->getInput());
  ASSERT_NE(convertedBackTopKRes, nullptr);
  EXPECT_EQ(convertedBackTopKRes->getElementType(0), ElemKind::FloatTy);
  auto *convertedTopK =
      llvm::dyn_cast<TopKNode>(convertedBackTopKRes->getInput());
  ASSERT_NE(convertedTopK, nullptr);
  EXPECT_EQ(convertedTopK->getElementType(0), ElemKind::Float16Ty);
  EXPECT_EQ(convertedTopK->getElementType(1), ElemKind::Int64ITy);
  // Check that the input of TopK is a convertTo node from float to
  // Float16Ty.
  auto *convertedTopKInput =
      llvm::dyn_cast<ConvertToNode>(convertedTopK->getInput());
  ASSERT_NE(convertedTopKInput, nullptr);
  EXPECT_EQ(convertedTopKInput->getElementType(0), ElemKind::Float16Ty);
  EXPECT_TRUE(llvm::isa<Placeholder>(convertedTopKInput->getInput()));
  EXPECT_EQ(convertedTopKInput->getInput().getElementType(), ElemKind::FloatTy);
  // At this point we know the input of TopK is convertTo(placeholder).
  // Check that this placeholder is the expected input.
  EXPECT_EQ(convertedTopK->getInput().getNode()->getNthInput(0).getNode(),
            input);

  // Now check the Int64ITy part of the graph.
  // Make sure the save node for the indices is still in the function and is
  // unchanged.
  EXPECT_TRUE(std::find(F->getNodes().begin(), F->getNodes().end(),
                        *resultIndices) != F->getNodes().end());
  EXPECT_EQ(resultIndices->getOutput(), NodeValue(outputIdx, 0));
  EXPECT_EQ(resultIndices->getInput(), NodeValue(convertedTopK, 1));
  EXPECT_EQ(resultIndices->getInput().getElementType(), ElemKind::Int64ITy);
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
  Context ctx;

  auto *input =
      mod.createPlaceholder(ElemKind::FloatTy, {20, 13}, "Input", false);
  auto *output =
      mod.createPlaceholder(ElemKind::FloatTy, {20, 10}, "Output", false);

  auto *FC = F->createFullyConnected(ctx, "FC", input, 10);
  auto *ReLU = F->createRELU("ReLU", FC, FC->getType(0));
  auto *result = F->createSave("save", ReLU, output);

  size_t origGraphSize = F->getNodes().size();

  TypeAToTypeBFunctionConverter converter(*F, ElemKind::FloatTy,
                                          ElemKind::Float16Ty);
  converter.convert();
  bool changed = converter.optimizeConversions();
  EXPECT_TRUE(changed);

  // We should have 4 more nodes:
  // 1 conversion float to float16 for each input of FC (3)
  // 1 conversion float16 to float for the result of ReLU.
  EXPECT_EQ(F->getNodes().size(), origGraphSize + 4);
  // Make sure the save node is still in the function and is unchanged.
  EXPECT_TRUE(std::find(F->getNodes().begin(), F->getNodes().end(), *result) !=
              F->getNodes().end());
  EXPECT_EQ(result->getOutput(), NodeValue(output, 0));
  // Check that the save is fed from a conversion from float16 to float.
  auto *convertedBackReLURes =
      llvm::dyn_cast<ConvertToNode>(result->getInput());
  ASSERT_NE(convertedBackReLURes, nullptr);
  EXPECT_EQ(convertedBackReLURes->getElementType(0), ElemKind::FloatTy);
  auto *convertedReLU =
      llvm::dyn_cast<ReluNode>(convertedBackReLURes->getInput());
  ASSERT_NE(convertedReLU, nullptr);
  EXPECT_EQ(convertedReLU->getElementType(0), ElemKind::Float16Ty);

  // Check that the ReLU is fed directly by FC float16.
  auto *convertedFC =
      llvm::dyn_cast<FullyConnectedNode>(convertedReLU->getInput());
  ASSERT_NE(convertedFC, nullptr);
  EXPECT_EQ(convertedFC->getElementType(0), ElemKind::Float16Ty);
  // Check that all the input of FC are convertTo node with from float to
  // Float16Ty.
  for (unsigned idx = 0, end = convertedFC->getNumInputs(); idx != end; ++idx) {
    auto *convertedFCInput =
        llvm::dyn_cast<ConvertToNode>(convertedFC->getNthInput(idx));
    ASSERT_NE(convertedFCInput, nullptr);
    EXPECT_EQ(convertedFCInput->getElementType(0), ElemKind::Float16Ty);
    EXPECT_TRUE(llvm::isa<Placeholder>(convertedFCInput->getInput()));
    EXPECT_EQ(convertedFCInput->getInput().getElementType(), ElemKind::FloatTy);
  }
  // At this point we know the input of FC is convertTo(placeholder).
  // Check that this placeholder is the expected input.
  EXPECT_EQ(convertedFC->getInput().getNode()->getNthInput(0).getNode(), input);
}

INSTANTIATE_TEST_CASE_P(Interpreter, AllBackends,
                        ::testing::Values(BackendKind::Interpreter));

#ifdef GLOW_WITH_CPU
INSTANTIATE_TEST_CASE_P(CPU, AllBackends, ::testing::Values(BackendKind::CPU));
#endif // GLOW_WITH_CPU

#ifdef GLOW_WITH_OPENCL
INSTANTIATE_TEST_CASE_P(OpenCL, AllBackends,
                        ::testing::Values(BackendKind::OpenCL));
#endif // GLOW_WITH_OPENCL
