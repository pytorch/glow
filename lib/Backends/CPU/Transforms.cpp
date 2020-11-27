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

#include "CPUBackend.h"

#include "glow/Graph/Graph.h"
#include "glow/Graph/Nodes.h"

using namespace glow;
using llvm::dyn_cast;
using llvm::isa;

/// Try to optimize the regular Convolution into a target-specific convolution
/// with a different filter memory layout. This optimization adds a new kind of
/// cpu-specific convolution that operates on filter weight data in a
/// non-standard format. The default format is DKKC, where D is the output
/// depth of the filter and C is the input channel, and K is the kernel size.
/// This optimization changes the data layout to [D/8, K, K, C, 8].  We
/// pre-swizzle the data in the weights to make the access pattern more
/// efficient.
static Node *optimizeCPUConv(ConvolutionNode *CN, Function *F) {
  auto depth = CN->getFilter().dims()[0];
  auto *M = F->getParent();
  auto group = CN->getGroup();

  // Make sure that the depth group is divisible by 64 to perform the
  // transformation. This transformation is currently only profitable on
  // low-channel convolutions.
  if (((depth / group) % 64) != 0) {
    return nullptr;
  }

  Constant *filter = dyn_cast<Constant>(CN->getFilter());
  if (!filter || filter->getNumUsers() != 1) {
    // Can't mutate the filter.
    return nullptr;
  }

  // We only support Floats for now.
  if (filter->getElementType() != ElemKind::FloatTy) {
    return nullptr;
  }

  // This optimization is not supported with Dilation currently.
  if (!std::all_of(CN->getDilation().begin(), CN->getDilation().end(),
                   [](unsigned_t i) { return i == 1; })) {
    return nullptr;
  }

  // Create a new constant filter with the layout [D/8, K, K, C, 8];
  TypeRef filterTy = filter->getType();
  auto dims = filterTy->dims();
  assert(dims.size() == 4 && "Invalid filter size");
  auto *filter8 = M->createConstant(filterTy->getElementType(),
                                    {dims[0] / 8, dims[1], dims[2], dims[3], 8},
                                    filter->getName());

  auto F8H = filter8->getHandle();
  auto FH = filter->getHandle();

  // Transpose the weights into the format [D/8, K, K, C, 8], where the depth
  // dimension is consecutive in memory.
  for (dim_t c0 = 0; c0 < dims[0]; c0++)
    for (dim_t c1 = 0; c1 < dims[1]; c1++)
      for (dim_t c2 = 0; c2 < dims[2]; c2++)
        for (dim_t c3 = 0; c3 < dims[3]; c3++) {
          F8H.at({c0 / 8, c1, c2, c3, c0 % 8}) = FH.at({c0, c1, c2, c3});
        }

  return F->addNode(new CPUConvDKKC8Node(
      CN->getName(), CN->getResult().getType(), CN->getInput(), filter8,
      CN->getBias(), CN->getKernels(), CN->getStrides(), CN->getPads(), group));
}

/// Merge Max and Splat nodes into target-specific CPUMaxSplat node.
/// For quantized network, sinkRescaleQuantizedNode transformation might have
/// merged Rescale into Max node. In this case we need to pull it out, since
/// CPUMaxSplat requires input and output to be quantized the same way.
static Node *optimizeCPUMaxSplat(MaxNode *MN, Function *F) {
  SplatNode *splat;
  NodeValue input;

  // One of the inputs must be Splat.
  if ((splat = dyn_cast<SplatNode>(MN->getLHS()))) {
    input = MN->getRHS();
  } else if ((splat = dyn_cast<SplatNode>(MN->getRHS()))) {
    input = MN->getLHS();
  } else {
    return nullptr;
  }

  // Pull out Rescale (for quantized types only).
  if (input.getType() != MN->getResult().getType()) {
    assert(input.getType()->isQuantizedType() &&
           MN->getResult().getType()->isQuantizedType() &&
           "Types should be quantized");
    auto *RS = F->createRescaleQuantized(MN->getName(), input,
                                         MN->getResult().getType());
    input = RS->getResult();
  }

  assert(input.dims() == splat->getResult().dims() &&
         input.getElementType() == splat->getResult().getElementType() &&
         "Element type and dimensions of the max inputs must match.");

  return F->addNode(
      new CPUMaxSplatNode(MN->getName(), input, splat->getValue()));
}

Expected<bool>
CPUBackend::transformPostLowering(Function *F, CompilationContext &,
                                  const glow::runtime::DeviceInfo *) const {
  LOG_SCOPE(F->getLogContext(), "CPUBackend::transformPostLowering")

  bool changed = false;
  for (auto &node : F->getNodes()) {
    // Try to replace generic convolution with cpu-optimized version.
    if (auto *CN = dyn_cast<ConvolutionNode>(&node)) {
      if (Node *NCN = optimizeCPUConv(CN, F)) {
        CN->getResult().replaceAllUsesOfWith(NCN);
        changed = true;
        continue;
      }
    }

    // Merge Max and Splat nodes into CPUMaxSplat.
    if (auto *MN = dyn_cast<MaxNode>(&node)) {
      if (Node *MSN = optimizeCPUMaxSplat(MN, F)) {
        MN->getResult().replaceAllUsesOfWith(MSN);
        changed = true;
        continue;
      }
    }
  }

  return changed;
}
