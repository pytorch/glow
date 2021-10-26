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

#include "DSPInjectorUtils.h"
#include "DSPInjectors.h"

namespace glow {
namespace {

struct CmpNEQ_int32_func {
  bool operator()(int32_t a, int32_t b) { return (a != b); }
};

struct CmpEQ_int32_func {
  bool operator()(int32_t a, int32_t b) { return (a == b); }
};

struct CmpLT_int32_func {
  bool operator()(int32_t a, int32_t b) { return (a < b); }
};

struct CmpLTE_int32_func {
  bool operator()(int32_t a, int32_t b) { return (a <= b); }
};

template <class CmpOp>
uint64_t IceRefCallback_Cmp_int32(void **inputsData,
                                  NNPITensorDesc *inputsTensorDescs,
                                  uint32_t /* inputsNum */, void **outputsData,
                                  NNPITensorDesc * /* outputsTensorDescs */,
                                  uint32_t /* outputsNum */,
                                  const void * /* ParamBlob */,
                                  uint32_t /* ParamBlobSize */) {
  auto *in0 = reinterpret_cast<int32_t *>(inputsData[0]);
  auto *in1 = reinterpret_cast<int32_t *>(inputsData[1]);
  auto *output = reinterpret_cast<bool *>(outputsData[0]);

  int numElements = DSPInjectorUtils::GetNumElements(
      inputsTensorDescs->dims, (int)inputsTensorDescs->numDims);
  CmpOp op;
  for (int i = 0; i < numElements; i++) {
    output[i] = op(in0[i], in1[i]);
  }
  return 0;
}

template <class CmpOp>
NNPICustomDSPNode *createCustomCmp_int32(Function *F_, llvm::StringRef name,
                                         NodeValue in0, NodeValue in1,
                                         std::string kernelName) {
  auto cb = reinterpret_cast<int64_t>(IceRefCallback_Cmp_int32<CmpOp>);
  return DSPInjectorUtils::createEltwiseInt32Compare(
      F_, name.str(), std::string(kernelName), {in0, in1}, cb);
}

template <class CmpOp, typename GlowNode>
Node *tryInject_Cmp_int32(Function *F, Node *node, std::string kernelName) {

  auto *castedNode = llvm::dyn_cast<GlowNode>(node);

  if (!castedNode) {
    return nullptr;
  }

  if (castedNode->getLHS().getElementType() != ElemKind::Int32ITy) {
    return nullptr;
  }

  if (castedNode->getRHS().getElementType() != ElemKind::Int32ITy) {
    return nullptr;
  }

  NNPICustomDSPNode *dspNode =
      createCustomCmp_int32<CmpOp>(F, "custom_cmpneq", castedNode->getLHS(),
                                   castedNode->getRHS(), kernelName);

  return dspNode;
}

} // namespace

Node *CustomCmpNEQNodeDSPKernelInjector::tryInject(Function *F,
                                                   Node *node) const {
  return tryInject_Cmp_int32<CmpNEQ_int32_func, CmpNEQNode>(F, node,
                                                            "CmpNEQ_int32");
}

Node *CustomCmpEQNodeDSPKernelInjector::tryInject(Function *F,
                                                  Node *node) const {
  return tryInject_Cmp_int32<CmpEQ_int32_func, CmpEQNode>(F, node,
                                                          "CmpEQ_int32");
}

Node *CustomCmpLTNodeDSPKernelInjector::tryInject(Function *F,
                                                  Node *node) const {
  return tryInject_Cmp_int32<CmpLT_int32_func, CmpLTNode>(F, node,
                                                          "CmpLT_int32");
}

Node *CustomCmpLTENodeDSPKernelInjector::tryInject(Function *F,
                                                   Node *node) const {
  return tryInject_Cmp_int32<CmpLTE_int32_func, CmpLTENode>(F, node,
                                                            "CmpLTE_int32");
}

} // namespace glow
