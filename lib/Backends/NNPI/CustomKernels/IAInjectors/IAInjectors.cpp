// (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

#include "glow/lib/Backends/NNPI/CustomKernels/IAInjectors/IAInjectors.h"
#include "glow/Graph/Graph.h"
#include "glow/lib/Backends/NNPI/CustomKernels/GetNNPIKernels.h"

namespace glow {

namespace {
template <typename GlowNode, ElemKind InputKind>
struct UnaryNodeIAKernelInjector : public CustomKernelInjector {
  bool tryInject(Function *F, Node *node) const override {
    GlowNode *castedNode = llvm::dyn_cast<GlowNode>(node);
    if (!castedNode) {
      return false;
    }

    if (castedNode->getInput().getElementType() != InputKind) {
      return false;
    }

    auto kernelName = strFormat("%s_%s_IAKernel", getNodeName<GlowNode>(),
                                Type::getElementName(InputKind).str().c_str());

    auto *iaNode = F->addNode(new NNPICustomIANode(
        strFormat("%s_as_%s", castedNode->getName().str().c_str(),
                  kernelName.c_str()),
        castedNode->getResult().getType(), {castedNode->getInput()}, kernelName,
        GetNNPIKernels::getCompiledIAKernelsFilePath()));
    castedNode->getResult().replaceAllUsesOfWith(iaNode);

    return true;
  }
};

template <typename GlowNode, ElemKind Input1Kind, ElemKind Input2Kind>
struct BinaryNodeIAKernelInjector : public CustomKernelInjector {
  bool tryInject(Function *F, Node *node) const override {
    GlowNode *castedNode = llvm::dyn_cast<GlowNode>(node);
    if (!castedNode) {
      return false;
    }

    if (castedNode->getNthInput(0).getElementType() != Input1Kind) {
      return false;
    }

    if (castedNode->getNthInput(1).getElementType() != Input2Kind) {
      return false;
    }

    auto kernelName = strFormat("%s_%s_%s_IAKernel", getNodeName<GlowNode>(),
                                Type::getElementName(Input1Kind).str().c_str(),
                                Type::getElementName(Input2Kind).str().c_str());

    auto *iaNode = F->addNode(new NNPICustomIANode(
        strFormat("%s_as_%s", castedNode->getName().str().c_str(),
                  kernelName.c_str()),
        castedNode->getResult().getType(),
        {castedNode->getNthInput(0), castedNode->getNthInput(1)}, kernelName,
        GetNNPIKernels::getCompiledIAKernelsFilePath()));
    castedNode->getResult().replaceAllUsesOfWith(iaNode);

    return true;
  }
};

// For ops that take a dimension argument, convert that dimension to a second
// tensor input.
template <typename GlowNode, ElemKind Input1Kind>
struct DimensionedIAKernelInjector : public CustomKernelInjector {
  bool tryInject(Function *F, Node *node) const override {
    GlowNode *castedNode = llvm::dyn_cast<GlowNode>(node);
    if (!castedNode) {
      return false;
    }

    if (castedNode->getNthInput(0).getElementType() != Input1Kind) {
      return false;
    }

    auto kernelName = strFormat("%s_%s_Dim_IAKernel", getNodeName<GlowNode>(),
                                Type::getElementName(Input1Kind).str().c_str());

    // Pass the dim argument as a constant tensor whose only value is dim.
    Constant *constNode = F->getParent()->createConstant(
        ElemKind::Int32ITy, {1}, castedNode->getName().str() + "_dim_constant");
    Handle<int32_t> tensor_handle(&constNode->getPayloadMutable());
    *tensor_handle.begin() = castedNode->getDim();

    auto *iaNode = F->addNode(new NNPICustomIANode(
        strFormat("%s_as_%s", castedNode->getName().str().c_str(),
                  kernelName.c_str()),
        castedNode->getResult().getType(),
        {castedNode->getNthInput(0), constNode}, kernelName,
        GetNNPIKernels::getCompiledIAKernelsFilePath()));
    castedNode->getResult().replaceAllUsesOfWith(iaNode);

    return true;
  }
};

template <typename GlowNode, ElemKind ToKind, ElemKind FromKind>
struct ConversionNodeIAKernelInjector : public CustomKernelInjector {
  bool tryInject(Function *F, Node *node) const override {
    GlowNode *castedNode = llvm::dyn_cast<GlowNode>(node);
    if (!castedNode) {
      return false;
    }

    if (castedNode->getResult().getElementType() != ToKind) {
      return false;
    }

    if (castedNode->getInput().getElementType() != FromKind) {
      return false;
    }

    auto kernelName = strFormat("%s_%s_%s_IAKernel", getNodeName<GlowNode>(),
                                Type::getElementName(ToKind).str().c_str(),
                                Type::getElementName(FromKind).str().c_str());

    auto *iaNode = F->addNode(new NNPICustomIANode(
        strFormat("%s_as_%s", castedNode->getName().str().c_str(),
                  kernelName.c_str()),
        castedNode->getResult().getType(), {castedNode->getInput()}, kernelName,
        GetNNPIKernels::getCompiledIAKernelsFilePath()));
    castedNode->getResult().replaceAllUsesOfWith(iaNode);

    return true;
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

  // CumSum Int32
  // TODO: pass dim as a second tensor
  injectors.emplace_back(
      std::make_unique<
          DimensionedIAKernelInjector<CumSumNode, ElemKind::Int32ITy>>());

  // SparseToDense Int32, float
  injectors.emplace_back(
      std::make_unique<BinaryNodeIAKernelInjector<
          SparseToDenseNode, ElemKind::Int32ITy, ElemKind::FloatTy>>());

  // SparseToDense Int32, fp16
  injectors.emplace_back(
      std::make_unique<BinaryNodeIAKernelInjector<
          SparseToDenseNode, ElemKind::Int32ITy, ElemKind::Float16Ty>>());

  // ConverTo Bool Int32
  injectors.emplace_back(
      std::make_unique<ConversionNodeIAKernelInjector<
          ConvertToNode, ElemKind::BoolTy, ElemKind::Int32ITy>>());

  // ConverTo Int32 Bool
  injectors.emplace_back(
      std::make_unique<ConversionNodeIAKernelInjector<
          ConvertToNode, ElemKind::Int32ITy, ElemKind::BoolTy>>());

  // ConverTo Float16 Int32
  injectors.emplace_back(
      std::make_unique<ConversionNodeIAKernelInjector<
          ConvertToNode, ElemKind::Float16Ty, ElemKind::Int32ITy>>());

  // ConverTo Int32 Float16Ty
  injectors.emplace_back(
      std::make_unique<ConversionNodeIAKernelInjector<
          ConvertToNode, ElemKind::Int32ITy, ElemKind::Float16Ty>>());

  return injectors;
}
} // namespace glow
