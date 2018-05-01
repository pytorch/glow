/**
 * Copyright (c) 2017-present, Facebook, Inc.
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

  // The depth dimension must be a multiple of 64 to perform the
  // transformation. This transformation is currently only profitable on
  // low-channel convolutions.
  if (depth < 64 || (depth % 64) != 0) {
    return nullptr;
  }

  Variable *filter = dyn_cast<Variable>(CN->getFilter());
  if (!filter || filter->getNumUsers() != 1 || !filter->isPrivate()) {
    // Can't mutate the filter.
    return nullptr;
  }

  // We only support Floats for now.
  if (filter->getElementType() != ElemKind::FloatTy) {
    return nullptr;
  }

  // Create a new variable filter with the layout [D/8, K, K, C, 8];
  TypeRef filterTy = filter->getType();
  auto dims = filterTy->dims();
  assert(dims.size() == 4 && "Invalid filter size");
  auto *filter8 = M->createVariable(
      filterTy->getElementType(), {dims[0] / 8, dims[1], dims[2], dims[3], 8},
      filter->getName(), VisibilityKind::Private, Variable::TrainKind::None);

  auto F8H = filter8->getHandle();
  auto FH = filter->getHandle();

  // Transpose the weights into the format [D/8, K, K, C, 8], where the depth
  // dimension is consecutive in memory.
  for (size_t c0 = 0; c0 < dims[0]; c0++)
    for (size_t c1 = 0; c1 < dims[1]; c1++)
      for (size_t c2 = 0; c2 < dims[2]; c2++)
        for (size_t c3 = 0; c3 < dims[3]; c3++) {
          F8H.at({c0 / 8, c1, c2, c3, c0 % 8}) = FH.at({c0, c1, c2, c3});
        }

  return F->addNode(new CPUConvDKKC8Node(
      CN->getName(), CN->getType(), CN->getInput(), filter8, CN->getBias(),
      CN->getKernel(), CN->getStride(), CN->getPad()));
}

bool CPUBackend::transformPostLowering(Function *F, CompilationMode mode) {
  bool changed = false;
  for (auto node : F->getNodes()) {

    if (auto *CN = dyn_cast<ConvolutionNode>(node)) {
      if (Node *NCN = optimizeCPUConv(CN, F)) {
        NodeValue(node, 0).replaceAllUsesOfWith(NCN);
        changed = true;
        continue;
      }
    }
    if (auto *MN = dyn_cast<MaxNode>(node)) {
      if (auto *splat = dyn_cast<SplatNode>(MN->getLHS())) {
        auto MSN = F->addNode(new CPUMaxSplatNode(MN->getName(), MN->getRHS(),
                                                  splat->getValue()));
        NodeValue(node, 0).replaceAllUsesOfWith(MSN);
        changed = true;
        continue;
      }
      if (auto *splat = dyn_cast<SplatNode>(MN->getRHS())) {
        auto MSN = F->addNode(new CPUMaxSplatNode(MN->getName(), MN->getLHS(),
                                                  splat->getValue()));
        NodeValue(node, 0).replaceAllUsesOfWith(MSN);
        changed = true;
        continue;
      }
    }
  }

  return changed;
}
