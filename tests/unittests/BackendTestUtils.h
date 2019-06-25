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

#include "glow/Backend/Backend.h"
#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Graph/Graph.h"
#include "glow/Graph/Node.h"
#include "glow/IR/IR.h"
#include "glow/IR/IRBuilder.h"
#include "glow/IR/IRGen.h"
#include "glow/Quantization/Base/Base.h"

#include "llvm/Support/Casting.h"

#include "gtest/gtest.h"

namespace glow {

extern unsigned parCloneCountOpt;

// A test harness to enable a test case for specific backends. A test suite
// should subclass this and instantiate it as follows:
//
// class OperationTest : public BackendTest {
//   ...
// };
//
// INSTANTIATE_TEST_CASE_P_FOR_BACKEND_TEST(Prefix, OperationTest);
//
// A test case is defined using TEST_P(), and ENABLED_BACKENDS() can be used
// to whitelist certain backends for the test. The absence of ENABLED_BACKENDS()
// enables the test for all available backends:
//
//
// TEST_P(OperationTest, replaceNaN) {
//   // Enable this test case only for Interpreter and CPU.
//   ENABLED_BACKENDS(Interpreter, CPU);
//   // Regular test code.
//   ...
// }
#define DECLARE_STATELESS_BACKEND_TEST(CLASS_NAME, CONFIG_NAME)                \
  class CLASS_NAME : public ::testing::TestWithParam<CONFIG_NAME> {            \
  protected:                                                                   \
    bool isEnabledBackend(const std::set<std::string> &enabledBackends) {      \
      return enabledBackends.find(getBackendName()) != enabledBackends.end();  \
    }                                                                          \
    const std::string Interpreter = "Interpreter";                             \
    const std::string CPU = "CPU";                                             \
    const std::string OpenCL = "OpenCL";                                       \
    const std::string Habana = "Habana";                                       \
                                                                               \
  public:                                                                      \
    std::string getBackendName() { return std::get<0>(GetParam()); }           \
  }

/// Note that we use std::tuple<std::string> here to match other tests which are
/// parameterized across many other values, e.g. those in ParameterSweepTest.
DECLARE_STATELESS_BACKEND_TEST(BackendStatelessTest, std::tuple<std::string>);

class BackendTest : public BackendStatelessTest {
public:
  BackendTest() : mod_(EE_.getModule()) { F_ = mod_.createFunction("main"); }

  ~BackendTest() override { mod_.clear(); }

protected:
  ExecutionEngine EE_{getBackendName()};
  Module &mod_;
  Function *F_;
};

static const auto all_backends = ::testing::Values(
#ifdef GLOW_WITH_CPU
    "CPU",
#endif // GLOW_WITH_CPU
#ifdef GLOW_WITH_OPENCL
    "OpenCL",
#endif // GLOW_WITH_OPENCL
#ifdef GLOW_WITH_HABANA
    "Habana",
#endif // GLOW_WITH_HABANA
    "Interpreter");

// Instantiate parameterized test suite with all available backends.
#define INSTANTIATE_TEST_CASE_P_FOR_BACKEND_TEST(prefix, test_case_name)       \
  INSTANTIATE_TEST_CASE_P(prefix, test_case_name, all_backends)

// Instantiate parameterized test suite with all available backends.
#define INSTANTIATE_TEST_CASE_P_FOR_BACKEND_COMBINED_TEST(                     \
    prefix, test_case_name, combine)                                           \
  INSTANTIATE_TEST_CASE_P(prefix, test_case_name,                              \
                          ::testing::Combine(all_backends, combine))

// TODO: Replace return for GTEST_SKIP() so that skipped tests are
// correctly reported once the macro gets available.
#define ENABLED_BACKENDS(...)                                                  \
  if (!isEnabledBackend({__VA_ARGS__}))                                        \
    return;

/// MockBackend used only for unit testing.
class MockBackend : public Backend {
  class MockFunction : public CompiledFunction {
  public:
    MockFunction(const runtime::RuntimeBundle &bundle)
        : CompiledFunction(bundle) {}

    llvm::Error execute(ExecutionContext *) override {
      return llvm::Error::success();
    }

    std::string getCompileBackendName() const override { return "Interpreter"; }
  };

  std::string getBackendName() const override { return "Interpreter"; }

  llvm::Expected<std::unique_ptr<CompiledFunction>>
  compile(Function *F, const BackendOptions &) const override {
    return llvm::make_unique<MockFunction>(runtime::RuntimeBundle::create(*F));
  }

  bool isOpSupported(const NodeInfo &NI) const override { return false; }

  bool generateInst(Node *N, IRGenVisitor &irgen) const override {
    return false;
  }
};

/// MockBackendCustomIRGen used only for unit testing to test custom lowering
/// from Node to Instruction IR.
class MockBackendCustomIRGen : public Backend {
  class MockFunction : public CompiledFunction {
  public:
    MockFunction(const runtime::RuntimeBundle &bundle)
        : CompiledFunction(bundle) {}

    llvm::Error execute(ExecutionContext *) override {
      return llvm::Error::success();
    }

    std::string getCompileBackendName() const override { return "Interpreter"; }
  };

  std::string getBackendName() const override { return "Interpreter"; }

  llvm::Expected<std::unique_ptr<CompiledFunction>>
  compile(Function *F, const BackendOptions &) const override {
    return llvm::make_unique<MockFunction>(runtime::RuntimeBundle::create(*F));
  }

  bool isOpSupported(const NodeInfo &NI) const override { return false; }

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
          CN__->getGroup(), CN__->getDilation());
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

/// Signature of functions used to create and init a Function. Returns a pair of
/// the Function created and the Placeholder of the output of the Function.
using CreateAndInitFunction =
    std::function<FunctionTensorPair(PlaceholderBindings &, ExecutionEngine &)>;

/// Given a method \p createAndInitFunction that creates and initializes a
/// FloatTy Function with a single output Tensor, \returns a bool representing
/// if the output Tensor of executing the Function on the Interpreter backend is
/// equal to executing it on a backend \p backendName. \p interpElemKind
/// and \p backendElemKind represent the desired ElemKinds for their respective
/// functions to use. If either require quantization then a profile will first
/// be gathered on the Interpreter, and then that profile will be used to
/// quantize one or both. Otherwise if either is Float16Ty then the respective
/// Function it will be converted using the Converter. If
/// \p enableRowwiseQuantization then rowwise quantization will be used for
/// nodes that support it. \p schema represents the quantization schema to use,
/// if applicable. \p parallelCount represents the number of times to clone the
/// Function inside itself, so that testing can be done on architectures that
/// have parallel compute engines.
void compareAgainstInterpreter(
    llvm::StringRef backendName, CreateAndInitFunction createAndInitFunction,
    ElemKind interpElemKind, ElemKind backendElemKind,
    float allowedError = 0.0001, unsigned parallelCount = 1,
    bool enableRowwiseQuantization = false,
    quantization::Schema schema = quantization::Schema::Asymmetric);

/// Given some \p FTP representing a Function with a single SaveNode and its
/// Tensor output, duplicate the Nodes in the Function and their Placeholder
/// inputs given \p bindings \p parallelCount times. \returns a set of Tensor
/// pointers for each output of the cloned Function.
std::unordered_set<Tensor *> cloneFunInsideFun(FunctionTensorPair FTP,
                                               PlaceholderBindings *bindings,
                                               unsigned parallelCount);

void inferConvNet(Tensor *inputs, Tensor *filter, Tensor *bias, Tensor *out,
                  llvm::StringRef kind);

void trainConvNet(Tensor *inputs, Tensor *kernel1, Tensor *bias1,
                  Tensor *kernel2, Tensor *bias2, Tensor *selected,
                  llvm::ArrayRef<size_t> shape1, llvm::ArrayRef<size_t> shape2,
                  Tensor *out, llvm::StringRef kind);

void inferLocalResponseNormalizationNet(Tensor *inputs, Tensor *out,
                                        llvm::StringRef kind);

void trainLocalResponseNormalizationNet(Tensor *inputs, Tensor *weights,
                                        Tensor *bias, Tensor *selected,
                                        llvm::ArrayRef<size_t> shape1,
                                        llvm::ArrayRef<size_t> shape2,
                                        Tensor *out, llvm::StringRef kind);
void trainAvgPoolNet(Tensor *inputs, Tensor *weights, Tensor *bias,
                     Tensor *selected, llvm::ArrayRef<size_t> shape1,
                     llvm::ArrayRef<size_t> shape2, Tensor *out,
                     llvm::StringRef kind);

void trainMaxPoolNet(Tensor *inputs, Tensor *weights, Tensor *bias,
                     Tensor *selected, llvm::ArrayRef<size_t> shape1,
                     llvm::ArrayRef<size_t> shape2, Tensor *out,
                     llvm::StringRef kind);

void inferIntLookupTableNet(Tensor *input, Tensor *out,
                            llvm::ArrayRef<int8_t> table, llvm::StringRef kind);

void inferGroupConv(Tensor *out, llvm::StringRef kind);

void inferNonSquarePaddingConv(Tensor *out, llvm::StringRef kind);

void inferNonSquareKernelConv(Tensor *out, llvm::StringRef kind);

void inferNonSquareStrideConv(Tensor *out, llvm::StringRef kind);

void inferConvDKKC8(Tensor *out, llvm::StringRef kind);

void inferSmallConv(Tensor *inputs, Tensor *out, llvm::StringRef kind);

void trainSoftMaxNet(Tensor *inputs, Tensor *weights, Tensor *bias,
                     Tensor *selected, Tensor *out, llvm::StringRef kind);

void inferBasicConvNet(Tensor *inputs, Tensor *out, llvm::StringRef kind,
                       size_t convDepth);

void inferTanhConcatNet(Tensor *input1, Tensor *input2, Tensor *input3,
                        Tensor *out, llvm::StringRef kind);

FunctionTensorPair createAndInitBasicFCNet(PlaceholderBindings &bindings,
                                           ExecutionEngine &EE);

void inferMixedNet(Tensor *inputs, Tensor *out, llvm::StringRef kind);

void inferComplexNet1(Tensor *inputs1, Tensor *inputs2, Tensor *inputs3,
                      Tensor *inputs4, Tensor *out, llvm::StringRef kind);

void inferTinyResnet(Tensor *input, Tensor *out, std::vector<Tensor> &weights,
                     llvm::StringRef kind);

void inferExtract3D(Tensor *input, Tensor *out, llvm::StringRef kind);

void inferMaxSplat(Tensor *input, Tensor *out, llvm::StringRef kind);

} // namespace glow
