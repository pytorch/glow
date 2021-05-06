// (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

#include "glow/lib/Backends/NNPI/CustomKernels/DSPInjectors/DSPInjectorUtils.h"
#include "glow/lib/Backends/NNPI/CustomKernels/DSPInjectors/DSPInjectors.h"

namespace glow {
namespace {
uint64_t IceRefCallback_CmpNEQ_int32(
    void **inputsData, NNPITensorDesc *inputsTensorDescs,
    uint32_t /* inputsNum */, void **outputsData,
    NNPITensorDesc * /* outputsTensorDescs */, uint32_t /* outputsNum */,
    const void * /* ParamBlob */, uint32_t /* ParamBlobSize */) {
  auto *in0 = reinterpret_cast<int32_t *>(inputsData[0]);
  auto *in1 = reinterpret_cast<int32_t *>(inputsData[1]);
  auto *output = reinterpret_cast<bool *>(outputsData[0]);

  int numElements = DSPInjectorUtils::GetNumElements(
      inputsTensorDescs->dims, (int)inputsTensorDescs->numDims);
  for (int i = 0; i < numElements; i++) {
    output[i] = (in0[i] != in1[i]);
  }
  return 0;
}

NNPICustomDSPNode *createCustomCmpNEQ_int32(Function *F_, llvm::StringRef name,
                                            NodeValue in0, NodeValue in1) {
  auto cb = reinterpret_cast<int64_t>(IceRefCallback_CmpNEQ_int32);
  return DSPInjectorUtils::createEltwiseInt32Compare(
      F_, name, std::string("CmpNEQ_int32"), {in0, in1}, cb);
}
} // namespace

bool CustomCmpNEQNodeDSPKernelInjector::tryInject(Function *F,
                                                  Node *node) const {
  auto *castedNode = llvm::dyn_cast<CmpNEQNode>(node);
  if (!castedNode) {
    return false;
  }

  if (castedNode->getLHS().getElementType() != ElemKind::Int32ITy) {
    return false;
  }

  if (castedNode->getRHS().getElementType() != ElemKind::Int32ITy) {
    return false;
  }

  NNPICustomDSPNode *dspNode = createCustomCmpNEQ_int32(
      F, "custom_cmpneq", castedNode->getLHS(), castedNode->getRHS());

  castedNode->getResult().replaceAllUsesOfWith(dspNode);

  return true;
}
} // namespace glow
