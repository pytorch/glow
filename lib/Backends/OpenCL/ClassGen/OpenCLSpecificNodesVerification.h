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

#ifdef GLOW_WITH_OPENCL

#include "glow/Graph/VerifierHelper.h"

bool OCLBatchedReduceAddNode::verify() const {
  Constant *destSliceSizes =
      llvm::dyn_cast<Constant>(getDestSliceSizes().getNode());
  Constant *srcSliceSizes =
      llvm::dyn_cast<Constant>(getSrcSliceSizes().getNode());

  // Both the destSliceSizes and srcSliceSizes should be Constants.
  if (!destSliceSizes || !srcSliceSizes) {
    return false;
  }

  // Check that the values of destSliceSizes and srcSliceSizes still match the
  // Types of the Input and Result. For more information, see
  // OCLBackend::transformPostLowering.
  bool ok = true;
  auto srcSliceSizesH = srcSliceSizes->getPayload().getHandle<int32_t>();
  auto srcDims = getInput().getType()->dims();

  if (!srcDims.empty()) {
    unsigned_t currentSliceSize = 1;
    unsigned j = srcSliceSizesH.size() - 1;
    for (ssize_t i = srcDims.size() - 1; i >= 0; --i) {
      if (i == getAxis()) {
        ok &= expectCompareTrue("axisSrcSlizeSize is incorrect",
                                getAxisSrcSliceSize(), currentSliceSize, this);
      } else {
        ok &=
            expectCompareTrue("srcSliceSize is incorrect",
                              static_cast<unsigned_t>(srcSliceSizesH.at({j--})),
                              currentSliceSize, this);
      }
      currentSliceSize *= srcDims[i];
    }
  } else {
    ok &= expectCompareTrue("axisSrcSlizeSize is incorrect",
                            getAxisSrcSliceSize(), static_cast<unsigned_t>(1),
                            this);
    ok &= expectCompareTrue("srcSliceSizes has the wrong shape",
                            srcSliceSizesH.size(), static_cast<dim_t>(1), this);
    ok &= expectCompareTrue("srcSliceSizes is incorrect",
                            srcSliceSizesH.at({0}), 1, this);
  }

  auto destDims = getResult().getType()->dims();
  std::vector<int32_t> destDimsVec(destDims.begin(), destDims.end());
  if (destDims.empty()) {
    destDimsVec.emplace_back(1);
  }
  auto destSliceSizesH = destSliceSizes->getPayload().getHandle<int32_t>();

  ok &= expectCompareTrue("destSliceSizes is incorrect",
                          destSliceSizesH.at({(dim_t)destDimsVec.size() - 1}),
                          1, this);

  for (ssize_t i = destDimsVec.size() - 2; i >= 0; --i) {
    ok &= expectCompareTrue("destSliceSizes is incorrect",
                            destSliceSizesH.at({static_cast<unsigned>(i)}),
                            destSliceSizesH.at({static_cast<unsigned>(i + 1)}) *
                                destDimsVec[i + 1],
                            this);
  }

  return ok;
}

#endif // GLOW_WITH_OPENCL
