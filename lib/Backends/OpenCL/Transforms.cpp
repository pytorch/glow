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

#include "OpenCL.h"

#include "glow/Backends/LayoutConverter.h"
#include "glow/Graph/Nodes.h"
#include "glow/Runtime/RuntimeTypes.h"

#include "llvm/Support/Casting.h"

using llvm::dyn_cast;

using namespace glow;

/// Perform OpenCL specific post-lowering graph transformation.
Expected<bool> OCLBackend::transformPostLowering(
    Function *F, CompilationContext &cctx,
    const glow::runtime::DeviceInfo *devInfo) const {
  // NCHW transformation is not supported in training mode yet, because of some
  // issues with gradient nodes.

  LOG_SCOPE(F->getLogContext(), "OCLBackend::transformPostLowering")

  bool changed = false;
  for (auto &node : F->getNodes()) {
    if (auto *CN = dyn_cast<ConvolutionNode>(&node)) {
      if (cctx.compMode == CompilationMode::Train) {
        continue;
      }

      // TODO: OpenCL fast convolution kernel itself has some issue with group >
      // 1, which will be investigated later. So far, if the group > 1, we just
      // call the slow convolution kernel.
      if (CN->getGroup() > 1) {
        continue;
      }

      if (CN->getLayout() == NCHW) {
        continue;
      }

      // If there is no compiler controlled local memory on the device,
      // try to avoid kernels that use (additional) copies to local memory.
      if (devInfo != nullptr && devInfo->availableLocalMemory == 0) {
        continue;
      }

      auto *NR = convertConvToNCHWConv(CN, F);
      CN->getResult().replaceAllUsesOfWith(NR);
      changed = true;
      continue;
    }
    if (auto *PMN = dyn_cast<MaxPoolNode>(&node)) {
      if (cctx.compMode == CompilationMode::Train) {
        continue;
      }

      if (PMN->getLayout() == NCHW) {
        continue;
      }

      // We need to replace both of the MaxPool results with their NCHW
      // counterpart in order to get rid of the old node. There appears to be a
      // bug in the implementation of the result(1) portion of the optimized
      // kernel, if getArgmax got users - bail.
      if (PMN->getArgmax().getNumUsers() > 0) {
        continue;
      }

      auto results = convertMaxPoolToNCHWPool(PMN, F);
      PMN->getResult().replaceAllUsesOfWith(results.first);
      changed = true;
      continue;
    }
    if (auto *PAN = dyn_cast<AvgPoolNode>(&node)) {
      if (cctx.compMode == CompilationMode::Train) {
        continue;
      }
      if (PAN->getLayout() == NCHW) {
        continue;
      }

      auto *NR = convertAvgPoolToNCHWPool(PAN, F);
      PAN->getResult().replaceAllUsesOfWith(NR);
      changed = true;
      continue;
    }
    // The code below replaces a regular BatchedReduceAddNode with a
    // semantically identical OCLBatchedReduceAddNode that has two additional
    // inputs for the slice sizes of the input and output nodes. The OpenCL
    // implementation of the batchedreduceadd instruction needs this information
    // and storing it in graph Constants ensures that it will be copied to the
    // device with the rest of the Function's Constants. Consequently, it does
    // not need to be copied separately or at runtime (which would increase
    // execution latency).
    if (auto *BRA = dyn_cast<BatchedReduceAddNode>(&node)) {

      assert(BRA->getAxes().size() == 1 &&
             "ReduceAdd: supporting reduction of a single axis only.");
      auto axis = BRA->getAxes()[0];

      // Determine and store the slice sizes of each input dimension excluding
      // the reduce axis into batchSliceSizes. Determine also the slice size on
      // the reduce axis and store that separately. These are used by the kernel
      // to index correctly into the input buffer. If the input has one
      // dimension (that is also the reduce axis), store one slice of size 1
      // into batchSliceSizes.
      auto batchDims = BRA->getBatch().getType()->dims();
      auto numBatchDims = batchDims.size();
      auto batchSliceSizesLen = numBatchDims > 1 ? numBatchDims - 1 : 1;
      auto *batchSliceSizes = F->getParent()->createConstant(
          ElemKind::Int32ITy, {(dim_t)batchSliceSizesLen}, "batchSliceSizes");
      auto batchSliceSizesH =
          batchSliceSizes->getPayloadMutable().getHandle<int32_t>();
      batchSliceSizesH.clear(1);

      size_t currentSliceSize = 1, axisSliceSize = 1;
      unsigned j = batchSliceSizesLen - 1;
      for (ssize_t i = batchDims.size() - 1; i >= 0; --i) {
        // If i is the reduce axis, currentSliceSize is the slice size at the
        // reduce axis. Store it in axisSliceSize and not in batchSliceSizes. If
        // not, do the opposite.
        if (i == axis) {
          axisSliceSize = currentSliceSize;
        } else {
          batchSliceSizesH.at({j--}) = currentSliceSize;
        }
        // Compute the slice size for the next iteration.
        currentSliceSize *= batchDims[i];
      }

      // Determine and store the slice sizes of each output dimension excluding
      // the reduce axis into destSliceSizes. These are used by the kernel to
      // index correctly into the output buffer. If the output has zero
      // dimensions store one slice of size 1 into destSliceSizes.
      auto destDims = BRA->getResult().getType()->dims();
      std::vector<size_t> destDimsVec(destDims.begin(), destDims.end());
      if (destDims.empty()) {
        destDimsVec.emplace_back(1);
      }
      auto numDestDims = destDimsVec.size();
      auto destSliceSizesLen = numDestDims > 0 ? numDestDims : 1;
      auto *destSliceSizes = F->getParent()->createConstant(
          ElemKind::Int32ITy, {(dim_t)destSliceSizesLen}, "destSliceSizes");
      auto destSliceSizesH =
          destSliceSizes->getPayloadMutable().getHandle<int32_t>();
      destSliceSizesH.clear(1);

      // Start i at destDimsVec.size() - 2 because the last slice size is always
      // known to be 1.
      for (ssize_t i = destDimsVec.size() - 2; i >= 0; --i) {
        // The slice size of the current dimension is the slice size of the
        // previous dimension multiplied by the number of elements in that
        // dimension.
        destSliceSizesH.at({static_cast<unsigned>(i)}) =
            destSliceSizesH.at({static_cast<unsigned>(i + 1)}) *
            destDimsVec[i + 1];
      }

      auto *OCLBRA = F->addNode(new OCLBatchedReduceAddNode(
          BRA->getName(), BRA->getResult().getType(), BRA->getBatch(),
          destSliceSizes, batchSliceSizes, axis, axisSliceSize));
      BRA->getResult().replaceAllUsesOfWith(OCLBRA);
      changed = true;
      continue;
    }
  }
  return changed;
}
