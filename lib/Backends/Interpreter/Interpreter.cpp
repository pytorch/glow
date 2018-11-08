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

#include "Interpreter.h"
#include "InterpreterFunction.h"

#include "glow/Backends/BackendUtils.h"
#include "glow/CodeGen/MemoryAllocator.h"
#include "glow/Graph/Graph.h"
#include "glow/Graph/Nodes.h"
#include "glow/IR/IR.h"

using namespace glow;

runtime::RuntimeBundle generateInterpreterRuntimeBundle(const IRFunction *F) {
  std::unordered_map<std::string, runtime::RuntimeSymbolInfo> symbolTable;
  MemoryAllocator constantWeightsAllocator("ConstantWeights", 0);
  for (auto &v : F->getGraph()->getParent()->getConstants()) {
    auto *w = F->getWeightForNode(v);
    auto numBytes = w->getSizeInBytes();
    size_t addr = constantWeightsAllocator.allocate(numBytes, w);
    runtime::RuntimeSymbolInfo symbol{};
    symbol.offset = addr;
    symbol.size = numBytes;
    symbol.type = *w->getType();
    symbolTable.emplace(std::string(w->getName()), symbol);
  }
  auto constantWeightsMaxSize = constantWeightsAllocator.getMaxMemoryUsage();
  runtime::RuntimeBundle bundle(constantWeightsMaxSize, 0, 0);
  bundle.symbolTable = std::move(symbolTable);
  bundle.constants =
      collectConstants(F, constantWeightsMaxSize, bundle.symbolTable);
  return bundle;
}

std::unique_ptr<CompiledFunction>
Interpreter::compile(Function *F, const Context &ctx) const {
  auto IR = generateAndOptimizeIR(F, shouldShareBuffers());
  return compileIR(std::move(IR), ctx);
}

std::unique_ptr<CompiledFunction>
Interpreter::compileIR(std::unique_ptr<IRFunction> IR,
                       const Context &ctx) const {
  runtime::RuntimeBundle bundle = generateInterpreterRuntimeBundle(IR.get());
  return llvm::make_unique<InterpreterFunction>(std::move(IR), ctx, bundle);
}

bool Interpreter::isOpSupported(Kinded::Kind opKind, ElemKind elementTy) const {
  // Check quantization support.
  if (elementTy == ElemKind::Int8QTy) {
    switch (opKind) {
    case Kinded::Kind::AddNodeKind:
    case Kinded::Kind::BatchedAddNodeKind:
    case Kinded::Kind::BatchedReduceAddNodeKind:
    case Kinded::Kind::CmpLTENodeKind:
    case Kinded::Kind::ConcatNodeKind:
    case Kinded::Kind::ConvolutionNodeKind:
    case Kinded::Kind::DequantizeNodeKind:
    case Kinded::Kind::DivNodeKind:
    case Kinded::Kind::FullyConnectedNodeKind:
    case Kinded::Kind::GatherNodeKind:
    case Kinded::Kind::LogNodeKind:
    case Kinded::Kind::MatMulNodeKind:
    case Kinded::Kind::MaxNodeKind:
    case Kinded::Kind::MinNodeKind:
    case Kinded::Kind::MulNodeKind:
    case Kinded::Kind::AvgPoolNodeKind:
    case Kinded::Kind::MaxPoolNodeKind:
    case Kinded::Kind::QuantizeNodeKind:
    case Kinded::Kind::ReluNodeKind:
    case Kinded::Kind::RescaleQuantizedNodeKind:
    case Kinded::Kind::ReshapeNodeKind:
    case Kinded::Kind::SelectNodeKind:
    case Kinded::Kind::SigmoidNodeKind:
    case Kinded::Kind::SliceNodeKind:
    case Kinded::Kind::SparseLengthsWeightedSumNodeKind:
    case Kinded::Kind::SubNodeKind:
    case Kinded::Kind::TanhNodeKind:
    case Kinded::Kind::TileNodeKind:
    case Kinded::Kind::TopKNodeKind:
    case Kinded::Kind::TransposeNodeKind:
    case Kinded::Kind::RowwiseQuantizedFullyConnectedNodeKind:
      return true;
    default:
      return false;
    }
  }
  if (elementTy == ElemKind::Float16Ty) {
    return false;
  }

  return true;
}

bool Interpreter::shouldLower(const Node *N) const {
  if (N->getKind() == Kinded::Kind::ConvolutionNodeKind)
    return false;
  return true;
}

namespace glow {
Backend *createInterpreter() { return new Interpreter(); }
} // namespace glow
