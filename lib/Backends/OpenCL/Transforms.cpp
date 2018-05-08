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

#include "OpenCL.h"

#include "glow/Graph/Graph.h"
#include "glow/Graph/Nodes.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace glow;
using llvm::dyn_cast;
using llvm::isa;

static auto NCHW2NHWC = {0u, 2u, 3u, 1u};
static auto NHWC2NCHW = {0u, 3u, 1u, 2u};

/// Optimize the regular Convolution into a target-specific convolution
/// with a different memory layout. Many GPU kernels prefer the NCHW memory
/// layout.
static Node *optimizeOCLConv(ConvolutionNode *CN, Function *F) {
  // Convert filter and input from NHWC (Glow's default) into NCHW.
  auto *NI = F->createTranspose("conv.input", CN->getInput(), NHWC2NCHW);
  auto *NF = F->createTranspose("conv.filter", CN->getFilter(), NHWC2NCHW);

  auto originalDims = CN->getType()->dims();
  size_t outDims[] = {originalDims[0], originalDims[3], originalDims[1],
                      originalDims[2]};
  auto outTy = F->getParent()->uniqueType(ElemKind::FloatTy, outDims);

  auto *NC = F->addNode(new OCLConvolutionNode(CN->getName(), outTy, NI, NF,
                                               CN->getBias(), CN->getKernel(),
                                               CN->getStride(), CN->getPad()));
  auto NR = F->createTranspose("conv.result", NC, NCHW2NHWC);
  return NR;
}

/// Optimize the regular pool average node into a target-specific node using the
/// NCHW memory layout.
static Node *optimizeOCLPoolAvg(PoolAvgNode *PAN, Function *F) {
  // Convert input from NHWC (Glow's default) into NCHW.
  auto *NI = F->createTranspose("conv.input", PAN->getInput(), NHWC2NCHW);

  auto originalDims = PAN->getType()->dims();
  size_t outDims[] = {originalDims[0], originalDims[3], originalDims[1],
                      originalDims[2]};
  auto outTy = F->getParent()->uniqueType(ElemKind::FloatTy, outDims);

  auto *NPAN =
      F->addNode(new OCLPoolAvgNode(PAN->getName(), outTy, NI, PAN->getKernel(),
                                    PAN->getStride(), PAN->getPad()));
  auto NR = F->createTranspose("poolavg.result", NPAN, NCHW2NHWC);
  return NR;
}

/// Optimize the regular pool max node into a target-specific node using the
/// NCHW memory layout.
static Node *optimizeOCLPoolMax(PoolMaxNode *PMN, Function *F) {
  // Convert input from NHWC (Glow's default) into NCHW.
  auto *NI = F->createTranspose("conv.input", PMN->getInput(), NHWC2NCHW);

  auto originalDims = PMN->getType()->dims();
  size_t outDims[] = {originalDims[0], originalDims[3], originalDims[1],
                      originalDims[2]};
  auto outTy = F->getParent()->uniqueType(ElemKind::FloatTy, outDims);

  auto *NPAN =
      F->addNode(new OCLPoolMaxNode(PMN->getName(), outTy, NI, PMN->getKernel(),
                                    PMN->getStride(), PMN->getPad()));
  auto NR = F->createTranspose("poolmax.result", NPAN, NCHW2NHWC);
  return NR;
}

/// Perform OpenCL specific post-lowering graph transformation.
bool OCLBackend::transformPostLowering(Function *F, CompilationMode mode) {
  bool changed = false;
  // Transformation is not supported in training mode yet, because of some
  // issues with gradient nodes.
  if (mode == CompilationMode::Train)
    return false;
  // Convert convolutions and pooling nodes into nodes using the NCHW format.
  for (auto node : F->getNodes()) {
    if (auto *CN = dyn_cast<ConvolutionNode>(node)) {
      if (Node *NCN = optimizeOCLConv(CN, F)) {
        NodeValue(node, 0).replaceAllUsesOfWith(NCN);
        changed = true;
        continue;
      }
    }
    if (auto *PAN = dyn_cast<PoolAvgNode>(node)) {
      if (Node *NPAN = optimizeOCLPoolAvg(PAN, F)) {
        NodeValue(node, 0).replaceAllUsesOfWith(NPAN);
        changed = true;
        continue;
      }
    }
    if (auto *PMN = dyn_cast<PoolMaxNode>(node)) {
      if (Node *NPMN = optimizeOCLPoolMax(PMN, F)) {
        NodeValue(node, 0).replaceAllUsesOfWith(NPMN);
        changed = true;
        continue;
      }
    }
  }
  return changed;
}
