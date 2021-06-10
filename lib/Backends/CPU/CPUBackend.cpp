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
#include "CPUFunction.h"
#include "CPULLVMIRGen.h"

#include "glow/Backend/BackendUtils.h"
#include "glow/Graph/Graph.h"
#include "glow/IR/Instrs.h"
#include "glow/LLVMIRCodeGen/LLVMIRGen.h"
#include "glow/Support/Debug.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"

#include <numeric>

using namespace glow;

CPUBackend::CPUBackend() {
  /// If target is not explicitly given we use the host attributes.
  auto &opts = getOptions();
  if (opts.getTarget().empty()) {
    opts.setTarget(LLVMBackend::getHostTarget());
    opts.setCPU(LLVMBackend::getHostCPU());
    opts.setTargetFeatures(LLVMBackend::getHostFeatures());
  }
}

/// We compile the standard library (libjit) to LLVM bitcode, and then convert
/// that binary data to an include file using an external utility (include-bin).
/// The resulting file is included here to compile the bitcode image into our
/// library.
static const unsigned char libjit_bc[] = {
#include "glow/libjit/libjit_cpu.inc"
};
static const size_t libjit_bc_size = sizeof(libjit_bc);

bool CPUBackend::isOpSupported(const NodeInfo &NI) const {
  switch (NI.getKind()) {

  case Kinded::Kind::CPUMaxSplatNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
        {ElemKind::FloatTy, ElemKind::Int8QTy});

  case Kinded::Kind::CPUConvDKKC8NodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind({ElemKind::FloatTy});

  // Delegate everything else to the LLVM backend.
  default:
    return LLVMBackend::isOpSupported(NI);
  }
}

bool CPUBackend::shouldLower(const Node *N) const {
  switch (N->getKind()) {
  case Kinded::Kind::ReluNodeKind:
  case Kinded::Kind::ClipNodeKind:
  case Kinded::Kind::LeakyReluNodeKind:
  case Kinded::Kind::FullyConnectedNodeKind:
  case Kinded::Kind::ConvolutionNodeKind:
  case Kinded::Kind::SparseLengthsSumNodeKind:
    return false;
  default:
    return true;
  }
}

bool CPUBackend::supportsFusedActivation(Node *parent, Node *activation) const {
  // CPU backend only supports fusing activations into Convolution and
  // ChannelwiseQuantizedConvolution.
  if (!llvm::isa<ConvolutionNode>(parent) &&
      !llvm::isa<ChannelwiseQuantizedConvolutionNode>(parent)) {
    return false;
  }

  // Only the following activations can be fused.
  // Additionally Tanh/Sigmoid are fused only for floating-point type. For
  // quantized type Lookup Tables should be used instead.
  switch (activation->getKind()) {
  case Kinded::Kind::ReluNodeKind:
  case Kinded::Kind::ClipNodeKind:
  case Kinded::Kind::LeakyReluNodeKind:
    return true;
  case Kinded::Kind::SigmoidNodeKind:
    return llvm::cast<SigmoidNode>(activation)
        ->getResult()
        .getType()
        ->isFPType();
  case Kinded::Kind::TanhNodeKind:
    return llvm::cast<TanhNode>(activation)->getResult().getType()->isFPType();
  default:
    return false;
  }
}

unsigned CPUBackend::numDevices() {
  return std::thread::hardware_concurrency();
}

std::vector<unsigned> CPUBackend::scanDeviceIDs() {
  std::vector<unsigned> deviceIDs(CPUBackend::numDevices());
  std::iota(std::begin(deviceIDs), std::end(deviceIDs), 0);
  return deviceIDs;
}

std::unique_ptr<CompiledFunction> CPUBackend::createCompiledFunction(
    std::unique_ptr<GlowJIT> JIT,
    runtime::RuntimeBundle &&runtimeBundle) const {
  return glow::make_unique<CPUFunction>(std::move(JIT),
                                        std::move(runtimeBundle));
}

std::unique_ptr<LLVMIRGen>
CPUBackend::createIRGen(const IRFunction *IR,
                        AllocationsInfo &allocationsInfo) const {
  CPULLVMIRGen *irgen = new CPULLVMIRGen(
      IR, allocationsInfo, "", getLibjitBitcode(), getObjectRegistry());
  return std::unique_ptr<CPULLVMIRGen>(irgen);
}

llvm::StringRef CPUBackend::getLibjitBitcode() const {
  return llvm::StringRef(reinterpret_cast<const char *>(libjit_bc),
                         libjit_bc_size);
}

/// \returns true if network supports Type Lowering from \p T1 to \p T2.
/// Populates PrecisionConfiguration with black list of operations that can't be
/// converted.
bool CPUBackend::canDoIndexTypeDemotion(
    ElemKind fromTy, ElemKind toTy, PrecisionConfiguration &precConfig) const {
  precConfig.precisionModeKindSet.insert(Kinded::Kind::EmbeddingBagNodeKind);
  precConfig.precisionModeKindSet.insert(
      Kinded::Kind::EmbeddingBagByteRowwiseOffsetsNodeKind);
  precConfig.precisionModeKindSet.insert(
      Kinded::Kind::FusedRowwiseQuantizedSparseLengthsSumNodeKind);
  precConfig.precisionModeKindSet.insert(
      Kinded::Kind::FusedRowwiseQuantizedSparseLengthsWeightedSumNodeKind);
  precConfig.precisionModeKindSet.insert(
      Kinded::Kind::SparseToDenseMaskNodeKind);
  return fromTy == ElemKind::Int64ITy && toTy == ElemKind::Int32ITy;
}

#if FACEBOOK_INTERNAL
llvm::ArrayRef<llvm::MemoryBufferRef> CPUBackend::getObjectRegistry() const {
  return llvm::ArrayRef<llvm::MemoryBufferRef>();
}
#else
#include "cpuObjectRegistry.h"
llvm::ArrayRef<llvm::MemoryBufferRef> CPUBackend::getObjectRegistry() const {
  return cpuObjectRegistry;
}
#endif
