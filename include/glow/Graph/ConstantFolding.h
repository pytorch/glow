/**
 * Copyright (c) 2019-present, Facebook, Inc.
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
#ifndef GLOW_GRAPH_CONSTANT_FOLDING_H
#define GLOW_GRAPH_CONSTANT_FOLDING_H

#include "glow/Base/Tensor.h"
#include "glow/Base/Type.h"

namespace glow {

template <typename ElemTy>
void constantFoldGather(Tensor *outT, const Tensor *dataT,
                        const Tensor *indicesT, const unsigned_t batchDims) {
  auto &dataTy = dataT->getType();

  size_t out_p = 0;
  unsigned elementSize = dataTy.getElementSize();
  // The size of the sample in the batch.
  size_t dataSampleSize = dataTy.getSliceSize(batchDims) * elementSize;
  // The size of the slices that we gather.
  size_t dataSliceSize = dataTy.getSliceSize(batchDims + 1) * elementSize;

  // Calculate the size of each sample in the batch.
  size_t numSamples = (dataT->size() * elementSize) / dataSampleSize;

  // Calculate number of samples in the batch.
  size_t batchSize = dataTy.dims()[batchDims];
  (void)batchSize;

  // For each sample in the batch:
  for (size_t sample = 0; sample < numSamples; sample++) {
    size_t sampleStart = sample * dataSampleSize;

    // For each slice (small fragment) that we copy from the source memory:
    for (size_t i = 0, end = indicesT->size(); i < end; i++) {
      size_t slice = indicesT->getHandle<ElemTy>().raw(i);
      assert(slice < batchSize && "Invalid index seen during Gather operation");
      std::copy(
          &dataT->getUnsafePtr()[sampleStart + dataSliceSize * slice],
          &dataT->getUnsafePtr()[sampleStart + dataSliceSize * (slice + 1)],
          &outT->getUnsafePtr()[out_p]);
      out_p += dataSliceSize;
    }
  }
}

} // namespace glow

#endif // GLOW_GRAPH_CONSTANT_FOLDING_H
