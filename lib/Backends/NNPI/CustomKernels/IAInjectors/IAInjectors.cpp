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

#include "IAInjectors.h"
#include "../GetNNPIKernels.h"
#include "glow/Graph/Graph.h"

namespace glow {

namespace {
template <typename GlowNode, ElemKind InputKind>
struct UnaryNodeIAKernelInjector : public CustomKernelInjector {
  Node *tryInject(Function *F, Node *node) const override {
    GlowNode *castedNode = llvm::dyn_cast<GlowNode>(node);
    if (!castedNode) {
      return nullptr;
    }

    if (castedNode->getInput().getElementType() != InputKind) {
      return nullptr;
    }

    auto kernelName = strFormat("%s_%s_IAKernel", getNodeName<GlowNode>(),
                                Type::getElementName(InputKind).str().c_str());
    Constant *kernelParams =
        F->getParent()->createConstant(ElemKind::Int32ITy, {}, "kernelParams");
    auto *iaNode = F->addNode(new NNPICustomIANode(
        strFormat("%s_as_%s", castedNode->getName().str().c_str(),
                  kernelName.c_str()),
        castedNode->getResult().getType(), kernelParams,
        {castedNode->getInput()}, kernelName,
        GetNNPIKernels::getCompiledIAKernelsFilePath(),
        /* IceRefCallback */ static_cast<int64_t>(0),
        /* PointerToLib */ static_cast<int64_t>(0), /* SizeOfLib */ 0));

    return iaNode;
  }
};

template <typename GlowNode, ElemKind Input1Kind, ElemKind Input2Kind>
struct BinaryNodeIAKernelInjector : public CustomKernelInjector {
  Node *tryInject(Function *F, Node *node) const override {
    GlowNode *castedNode = llvm::dyn_cast<GlowNode>(node);
    if (!castedNode) {
      return nullptr;
    }

    if (castedNode->getNthInput(0).getElementType() != Input1Kind) {
      return nullptr;
    }

    if (castedNode->getNthInput(1).getElementType() != Input2Kind) {
      return nullptr;
    }

    auto kernelName = strFormat("%s_%s_%s_IAKernel", getNodeName<GlowNode>(),
                                Type::getElementName(Input1Kind).str().c_str(),
                                Type::getElementName(Input2Kind).str().c_str());
    Constant *kernelParams =
        F->getParent()->createConstant(ElemKind::Int32ITy, {}, "kernelParams");
    auto *iaNode = F->addNode(new NNPICustomIANode(
        strFormat("%s_as_%s", castedNode->getName().str().c_str(),
                  kernelName.c_str()),
        castedNode->getResult().getType(), kernelParams,
        {castedNode->getNthInput(0), castedNode->getNthInput(1)}, kernelName,
        GetNNPIKernels::getCompiledIAKernelsFilePath(),
        /* IceRefCallback */ static_cast<int64_t>(0),
        /* PointerToLib */ static_cast<int64_t>(0), /* SizeOfLib */ 0));

    return iaNode;
  }
};

// For ops that take a dimension argument, convert that dimension to a second
// tensor input.
template <typename GlowNode, ElemKind Input1Kind>
struct DimensionedIAKernelInjector : public CustomKernelInjector {
  Node *tryInject(Function *F, Node *node) const override {
    GlowNode *castedNode = llvm::dyn_cast<GlowNode>(node);
    if (!castedNode) {
      return nullptr;
    }

    if (castedNode->getNthInput(0).getElementType() != Input1Kind) {
      return nullptr;
    }

    auto kernelName = strFormat("%s_%s_Dim_IAKernel", getNodeName<GlowNode>(),
                                Type::getElementName(Input1Kind).str().c_str());

    // Pass the dim argument as a constant tensor whose only value is dim.
    Constant *constNode = F->getParent()->createConstant(
        ElemKind::Int32ITy, {1}, castedNode->getName().str() + "_dim_constant");
    Handle<int32_t> tensor_handle(&constNode->getPayloadMutable());
    *tensor_handle.begin() = castedNode->getDim();

    Constant *kernelParams =
        F->getParent()->createConstant(ElemKind::Int32ITy, {}, "kernelParams");
    auto *iaNode = F->addNode(new NNPICustomIANode(
        strFormat("%s_as_%s", castedNode->getName().str().c_str(),
                  kernelName.c_str()),
        castedNode->getResult().getType(), kernelParams,
        {castedNode->getNthInput(0), constNode}, kernelName,
        GetNNPIKernels::getCompiledIAKernelsFilePath(),
        /* IceRefCallback */ static_cast<int64_t>(0)));

    return iaNode;
  }
};

template <typename GlowNode, ElemKind ToKind, ElemKind FromKind>
struct ConversionNodeIAKernelInjector : public CustomKernelInjector {
  Node *tryInject(Function *F, Node *node) const override {
    GlowNode *castedNode = llvm::dyn_cast<GlowNode>(node);
    if (!castedNode) {
      return nullptr;
    }

    if (castedNode->getResult().getElementType() != ToKind) {
      return nullptr;
    }

    if (castedNode->getInput().getElementType() != FromKind) {
      return nullptr;
    }

    auto kernelName = strFormat("%s_%s_%s_IAKernel", getNodeName<GlowNode>(),
                                Type::getElementName(ToKind).str().c_str(),
                                Type::getElementName(FromKind).str().c_str());

    Constant *kernelParams =
        F->getParent()->createConstant(ElemKind::Int32ITy, {}, "kernelParams");
    auto *iaNode = F->addNode(new NNPICustomIANode(
        strFormat("%s_as_%s", castedNode->getName().str().c_str(),
                  kernelName.c_str()),
        castedNode->getResult().getType(), kernelParams,
        {castedNode->getInput()}, kernelName,
        GetNNPIKernels::getCompiledIAKernelsFilePath(),
        /* IceRefCallback */ static_cast<int64_t>(0),
        /* PointerToLib */ static_cast<int64_t>(0), /* SizeOfLib */ 0));

    return iaNode;
  }
};
} // namespace

/// Build the list of CustomKernelInjectors to be called on each node in
/// graphs that are being compiled for NNPI. CustomKernelInjector are called
/// in order for each node so if there is a preference between kernels for the
/// same Glow Operator (for example, prefer DSP kernels), make sure those are
/// listed first.
std::vector<std::unique_ptr<CustomKernelInjector>> buildIAInjectors() {
  std::vector<std::unique_ptr<CustomKernelInjector>> injectors;

  // Add Int32 Int32
  injectors.emplace_back(
      std::make_unique<BinaryNodeIAKernelInjector<AddNode, ElemKind::Int32ITy,
                                                  ElemKind::Int32ITy>>());

  // Sub Int32 Int32
  injectors.emplace_back(
      std::make_unique<BinaryNodeIAKernelInjector<SubNode, ElemKind::Int32ITy,
                                                  ElemKind::Int32ITy>>());

  // Mul Int32 Int32
  injectors.emplace_back(
      std::make_unique<BinaryNodeIAKernelInjector<MulNode, ElemKind::Int32ITy,
                                                  ElemKind::Int32ITy>>());

  // Div Int32 Int32
  injectors.emplace_back(
      std::make_unique<BinaryNodeIAKernelInjector<DivNode, ElemKind::Int32ITy,
                                                  ElemKind::Int32ITy>>());

  // Max Int32 Int32
  injectors.emplace_back(
      std::make_unique<BinaryNodeIAKernelInjector<MaxNode, ElemKind::Int32ITy,
                                                  ElemKind::Int32ITy>>());

  // Min Int32 Int32
  injectors.emplace_back(
      std::make_unique<BinaryNodeIAKernelInjector<MinNode, ElemKind::Int32ITy,
                                                  ElemKind::Int32ITy>>());

  // Neg Int32
  injectors.emplace_back(
      std::make_unique<
          UnaryNodeIAKernelInjector<NegNode, ElemKind::Int32ITy>>());

#if NNPI_MAJOR_VERSION >= 1 && NNPI_MINOR_VERSION < 8
  // LTE Int32 Int32
  injectors.emplace_back(
      std::make_unique<BinaryNodeIAKernelInjector<
          CmpLTENode, ElemKind::Int32ITy, ElemKind::Int32ITy>>());

  // LT Int32 Int32
  injectors.emplace_back(
      std::make_unique<BinaryNodeIAKernelInjector<CmpLTNode, ElemKind::Int32ITy,
                                                  ElemKind::Int32ITy>>());

  // EQ Int32 Int32
  injectors.emplace_back(
      std::make_unique<BinaryNodeIAKernelInjector<CmpEQNode, ElemKind::Int32ITy,
                                                  ElemKind::Int32ITy>>());

  // NEQ Int32 Int32
  injectors.emplace_back(
      std::make_unique<BinaryNodeIAKernelInjector<
          CmpNEQNode, ElemKind::Int32ITy, ElemKind::Int32ITy>>());

#endif // NNPI < 1.8

#if NNPI_MAJOR_VERSION >= 1 && NNPI_MINOR_VERSION < 7
  // CumSum Int32
  // TODO: pass dim as a second tensor
  injectors.emplace_back(
      std::make_unique<
          DimensionedIAKernelInjector<CumSumNode, ElemKind::Int32ITy>>());
#endif // NNPI < 1.7

  // ConverTo Bool Int32
  injectors.emplace_back(
      std::make_unique<ConversionNodeIAKernelInjector<
          ConvertToNode, ElemKind::BoolTy, ElemKind::Int32ITy>>());

  // ConverTo Int32 Bool
  injectors.emplace_back(
      std::make_unique<ConversionNodeIAKernelInjector<
          ConvertToNode, ElemKind::Int32ITy, ElemKind::BoolTy>>());

#if NNPI_MAJOR_VERSION >= 1 && NNPI_MINOR_VERSION < 7
  // ConverTo Float16 Int32
  injectors.emplace_back(
      std::make_unique<ConversionNodeIAKernelInjector<
          ConvertToNode, ElemKind::Float16Ty, ElemKind::Int32ITy>>());

  // ConverTo Int32 Float16Ty
  injectors.emplace_back(
      std::make_unique<ConversionNodeIAKernelInjector<
          ConvertToNode, ElemKind::Int32ITy, ElemKind::Float16Ty>>());
#endif // NNPI < 1.7
  return injectors;
}
} // namespace glow
