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

#include "glow/Backends/Backend.h"
#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Graph/Graph.h"
#include "glow/IR/IR.h"

#include "llvm/Support/Casting.h"

#include "glow/Backends/Backend.h"
#include "glow/Graph/Node.h"
#include "glow/IR/IRBuilder.h"
#include "glow/IR/IRGen.h"

namespace glow {

/// MockBackend used only for unit testing.
class MockBackend : public Backend {
  class MockFunction : public CompiledFunction {
    void execute(Context *) override {}

    BackendKind getCompileBackendKind() const override {
      return BackendKind::Interpreter;
    }
  };

  BackendKind getBackendKind() const override {
    return BackendKind::Interpreter;
  }

  std::unique_ptr<CompiledFunction> compile(Function *F) const override {
    return llvm::make_unique<MockFunction>();
  }

  std::unique_ptr<CompiledFunction>
  compileWithoutConstants(Function *F) const override {
    return llvm::make_unique<MockFunction>();
  }

  bool isOpSupported(Kinded::Kind opKind, ElemKind elementTy) const override {
    return false;
  }

  bool generateInst(Node *N, IRGenVisitor &irgen) const override {
    return false;
  }
};

/// MockBackendCustomIRGen used only for unit testing to test custom lowering
/// from Node to Instruction IR.
class MockBackendCustomIRGen : public Backend {
  class MockFunction : public CompiledFunction {
    void execute(Context *) override {}

    BackendKind getCompileBackendKind() const override {
      return BackendKind::Interpreter;
    }
  };

  BackendKind getBackendKind() const override {
    return BackendKind::Interpreter;
  }

  std::unique_ptr<CompiledFunction> compile(Function *F) const override {
    return llvm::make_unique<MockFunction>();
  }

  std::unique_ptr<CompiledFunction>
  compileWithoutConstants(Function *F) const override {
    return llvm::make_unique<MockFunction>();
  }

  bool isOpSupported(Kinded::Kind opKind, ElemKind elementTy) const override {
    return false;
  }

  bool generateInst(Node *N, IRGenVisitor &irgen) const override {
    bool hasChanged = false;
    auto builder_ = irgen.getBuilder();
    switch (N->getKind()) {
    case glow::Kinded::Kind::ConvolutionNodeKind: {
      auto *CN__ = llvm::cast<ConvolutionNode>(N);
      auto *Src = irgen.valueForNode(CN__->getInput());
      auto *Filter = irgen.valueForNode(CN__->getFilter());
      auto *Bias = irgen.valueForNode(CN__->getBias());
      std::string allocName = std::string(N->getName()) + ".res";
      auto *Dest__ = builder_->createAllocActivationInst(
          allocName, CN__->getResult().getType());
      auto *V = builder_->createConvolutionInst(
          "CustomConvolutionInstruction", Dest__, Src, Filter, Bias,
          CN__->getKernels(), CN__->getStrides(), CN__->getPads(),
          CN__->getGroup());
      if (N->hasPredicate()) {
        V->setPredicate(irgen.valueForNode(N->getPredicate()));
      }
      irgen.registerIR(CN__->getResult(), V->getDest());
      irgen.setNodeToIR(N, V);
      hasChanged = true;
      break;
    }
    default:
      break;
    }
    return hasChanged;
  }
};

/// Pair representing a pointer to a Function with a single output, and the
/// allocated Tensor that backs the Placeholder of the single output.
using FunctionTensorPair = std::pair<Function *, Tensor *>;

void inferConvNet(Tensor *inputs, Tensor *filter, Tensor *bias, Tensor *out,
                  BackendKind kind);

void trainConvNet(Tensor *inputs, Tensor *kernel1, Tensor *bias1,
                  Tensor *kernel2, Tensor *bias2, Tensor *selected,
                  llvm::ArrayRef<size_t> shape1, llvm::ArrayRef<size_t> shape2,
                  Tensor *out, BackendKind kind);

void inferLocalResponseNormalizationNet(Tensor *inputs, Tensor *out,
                                        BackendKind kind);

void trainLocalResponseNormalizationNet(Tensor *inputs, Tensor *weights,
                                        Tensor *bias, Tensor *selected,
                                        llvm::ArrayRef<size_t> shape1,
                                        llvm::ArrayRef<size_t> shape2,
                                        Tensor *out, BackendKind kind);
void trainAvgPoolNet(Tensor *inputs, Tensor *weights, Tensor *bias,
                     Tensor *selected, llvm::ArrayRef<size_t> shape1,
                     llvm::ArrayRef<size_t> shape2, Tensor *out,
                     BackendKind kind);

void trainMaxPoolNet(Tensor *inputs, Tensor *weights, Tensor *bias,
                     Tensor *selected, llvm::ArrayRef<size_t> shape1,
                     llvm::ArrayRef<size_t> shape2, Tensor *out,
                     BackendKind kind);

void inferIntLookupTableNet(Tensor *input, Tensor *out,
                            llvm::ArrayRef<int8_t> table, BackendKind kind);

void inferGroupConv(Tensor *out, BackendKind kind);

void inferNonSquarePaddingConv(Tensor *out, BackendKind kind);

void inferNonSquareKernelConv(Tensor *out, BackendKind kind);

void inferNonSquareStrideConv(Tensor *out, BackendKind kind);

void inferConvDKKC8(Tensor *out, BackendKind kind);

void inferSmallConv(Tensor *inputs, Tensor *out, BackendKind kind);

void trainSoftMaxNet(Tensor *inputs, Tensor *weights, Tensor *bias,
                     Tensor *selected, Tensor *out, BackendKind kind);

void inferBasicConvNet(Tensor *inputs, Tensor *out, BackendKind kind,
                       size_t convDepth);

void inferTanhConcatNet(Tensor *input1, Tensor *input2, Tensor *input3,
                        Tensor *out, BackendKind kind);

FunctionTensorPair createAndInitBasicFCNet(Context &ctx, ExecutionEngine &EE);

void inferMixedNet(Tensor *inputs, Tensor *out, BackendKind kind);

void inferComplexNet1(Tensor *inputs1, Tensor *inputs2, Tensor *inputs3,
                      Tensor *inputs4, Tensor *out, BackendKind kind);

void inferTinyResnet(Tensor *input, Tensor *out, std::vector<Tensor> &weights,
                     BackendKind kind);

void inferExtract3D(Tensor *input, Tensor *out, BackendKind kind);

void inferMaxSplat(Tensor *input, Tensor *out, BackendKind kind);

} // namespace glow
