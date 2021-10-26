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
uint64_t IceRefCallback_ReluFP16(void **inputsData,
                                 NNPITensorDesc *inputsTensorDescs,
                                 uint32_t /* inputsNum */, void **outputsData,
                                 NNPITensorDesc * /* outputsTensorDescs */,
                                 uint32_t /* outputsNum */,
                                 const void * /* ParamBlob */,
                                 uint32_t /* ParamBlobSize */) {
  float16_t *in0 = (float16_t *)inputsData[0];
  float16_t *output = (float16_t *)outputsData[0];

  int numElements = DSPInjectorUtils::GetNumElements(
      inputsTensorDescs->dims, (int)inputsTensorDescs->numDims);
  for (int i = 0; i < numElements; i++) {
    output[i] = (in0[i] < (float16_t)0.0f) ? (float16_t)0.0f : in0[i];
  }
  return 0;
}

NNPICustomDSPNode *createCustomReluFP16(Function *F_, llvm::StringRef name,
                                        NodeValue in0) {
  int64_t cb = reinterpret_cast<int64_t>(IceRefCallback_ReluFP16);
  return DSPInjectorUtils::createEltwiseFP16(
      F_, name.str(), std::string("ReluFP16"), {in0}, cb);
}
} // namespace

Node *CustomReluNodeDSPKernelInjector::tryInject(Function *F,
                                                 Node *node) const {
  ReluNode *castedNode = llvm::dyn_cast<ReluNode>(node);
  if (!castedNode) {
    return nullptr;
  }

  if (castedNode->getInput().getElementType() != ElemKind::Float16Ty) {
    return nullptr;
  }

  NNPICustomDSPNode *dspNode =
      createCustomReluFP16(F, "custom_relu", castedNode->getInput());

  return dspNode;
}
} // namespace glow
