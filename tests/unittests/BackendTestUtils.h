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
#ifndef GLOW_TESTS_BACKENDTESTUTILS_H
#define GLOW_TESTS_BACKENDTESTUTILS_H

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

// INSTANTIATE_TEST_CASE_P is deprecated in gtest v1.10.0. For now use it still
// internally.
#if FACEBOOK_INTERNAL
#define GLOW_INSTANTIATE_TEST_SUITE_P INSTANTIATE_TEST_CASE_P
#else
#define GLOW_INSTANTIATE_TEST_SUITE_P INSTANTIATE_TEST_SUITE_P
#endif /* FACEBOOK_INTERNAL */

// A test harness to enable a test case for specific backends. A test suite
// should subclass this and instantiate it as follows:
//
// class OperationTest : public BackendTest {
//   ...
// };
//
// GLOW_INSTANTIATE_TEST_SUITE_P_FOR_BACKEND_TEST(Prefix, OperationTest);
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
                                                                               \
  public:                                                                      \
    std::string getBackendName() { return std::get<0>(GetParam()); }           \
  }

/// Note that we use std::tuple<std::string> here to match other tests which are
/// parameterized across many other values, e.g. those in ParameterSweepTest.
DECLARE_STATELESS_BACKEND_TEST(BackendStatelessTest, std::tuple<std::string>);

/// Whether to ignore the blacklist and run disabled tests.
extern bool runDisabledTests;

class BackendTest : public BackendStatelessTest {
public:
  BackendTest(uint64_t deviceMemory = 0)
      : EE_(getBackendName(), deviceMemory), mod_(EE_.getModule()) {
    F_ = mod_.createFunction("main");
  }

protected:
  ExecutionEngine EE_{getBackendName()};
  Module &mod_;
  Function *F_;
};

/// Stringify a macro def.
#define BACKEND_TO_STR(X) #X

#ifdef GLOW_TEST_BACKEND
#define STRINGIZE(X) BACKEND_TO_STR(X)
#define ALL_BACKENDS ::testing::Values(STRINGIZE(GLOW_TEST_BACKEND))
#else
#define ALL_BACKENDS ::testing::ValuesIn(getAvailableBackends())
#endif

// Instantiate parameterized test suite with all available backends.
#define GLOW_INSTANTIATE_TEST_SUITE_P_FOR_BACKEND_TEST(prefix, test_case_name) \
  GLOW_INSTANTIATE_TEST_SUITE_P(prefix, test_case_name, ALL_BACKENDS)

// Instantiate parameterized test suite with all available backends.
#define GLOW_INSTANTIATE_TEST_SUITE_P_FOR_BACKEND_COMBINED_TEST(               \
    prefix, test_case_name, combine)                                           \
  GLOW_INSTANTIATE_TEST_SUITE_P(prefix, test_case_name,                        \
                                ::testing::Combine(ALL_BACKENDS, combine))

// TODO: Replace return for GTEST_SKIP() so that skipped tests are
// correctly reported once the macro gets available.
#define ENABLED_BACKENDS(...)                                                  \
  if (!runDisabledTests && !isEnabledBackend({__VA_ARGS__}))                   \
    GTEST_SKIP();

/// Blacklist of tests for the current backend under test.
extern std::set<std::string> backendTestBlacklist;

/// Bool for whether to use symmetric quantization for rowwise-quantized FCs.
extern bool useSymmetricRowwiseQuantFC;

/// Intermediate layer of macros to make expansion of defs work correctly.
#define INSTANTIATE_TEST_INTERNAL(B, T)                                        \
  GLOW_INSTANTIATE_TEST_SUITE_P(B, T, ::testing::Values(BACKEND_TO_STR(B)));

/// Instantate a test suite for the backend specified by GLOW_TEST_BACKEND.
/// Usually this macro will be defined by the build system, to avoid tightly
/// coupling the existing set of backends to the source.
#define INSTANTIATE_BACKEND_TEST(T)                                            \
  INSTANTIATE_TEST_INTERNAL(GLOW_TEST_BACKEND, T);

/// Helper macro to check the current test against the blacklist.
#define CHECK_IF_ENABLED()                                                     \
  if (!runDisabledTests &&                                                     \
      backendTestBlacklist.count(                                              \
          ::testing::UnitTest::GetInstance()->current_test_info()->name()))    \
    GTEST_SKIP();

class NumericsTest : public BackendTest {
protected:
  PlaceholderBindings bindings_;
};

class GraphOptz : public ::testing::Test {
public:
  GraphOptz(llvm::StringRef backendName = "Interpreter")
      : EE_(backendName), mod_(EE_.getModule()) {
    F_ = mod_.createFunction("main");
  }

protected:
  void checkNumericalEquivalence(float allowedError = 0.0001) {
    // Check that the function and its optimized complement exist.
    EXPECT_TRUE(F_);
    EXPECT_TRUE(optimizedF_);

    // Check that the bindings are not empty. If they are, the numerical
    // equivalence check can produce a false positive.
    EXPECT_GT(bindings_.getDataSize(), 0);

    // Clone bindings to use for original and optimized functions.
    PlaceholderBindings originalBindings = bindings_.clone();
    PlaceholderBindings optimizedBindings = bindings_.clone();

    // Compile and run functions. Only lower Functions; we do not want to
    // optimize the unoptimized Function, and the optimized Function has, well,
    // already been optimized.
    if (!alreadyCompiled_) {
      cctx_.optimizationOpts.onlyLowerFuns.insert(F_);
      cctx_.optimizationOpts.onlyLowerFuns.insert(optimizedF_);
      EE_.compile(cctx_);
    }
    EE_.run(originalBindings, F_->getName());
    EE_.run(optimizedBindings, optimizedF_->getName());

    // Compare outputs.
    EXPECT_TRUE(PlaceholderBindings::compare(&originalBindings,
                                             &optimizedBindings, allowedError));
  }

  /// ExecutionEngine instance for running functions to check numerical
  /// equivalence.
  ExecutionEngine EE_;
  /// A reference to the Module inside EE_.
  Module &mod_;
  /// The original Function for the test case.
  Function *F_{nullptr};
  /// The optimized Function for the test case.
  Function *optimizedF_{nullptr};
  /// The bindings used to check numerical equivalence for the test case.
  PlaceholderBindings bindings_;
  /// CompilationContext used for all Functions in \ref mod_.
  CompilationContext cctx_;
  /// Whether \ref mod_ has already been compiled.
  bool alreadyCompiled_{false};
};

/// MockBackend used only for unit testing.
class MockBackend : public Backend {
  class MockFunction : public CompiledFunction {
  public:
    MockFunction(runtime::RuntimeBundle &&bundle)
        : CompiledFunction(std::move(bundle)) {}

    Error execute(ExecutionContext *) override { return Error::success(); }

    std::string getCompileBackendName() const override { return "Interpreter"; }
  };

  std::string getBackendName() const override { return "Interpreter"; }

  Expected<std::unique_ptr<CompiledFunction>>
  compile(Function *F, const BackendOptions &) const override {
    return glow::make_unique<MockFunction>(runtime::RuntimeBundle::create(*F));
  }

  bool isOpSupported(const NodeInfo &NI) const override { return false; }

  bool generateInst(Node *N, IRGenVisitor &irgen) const override {
    return false;
  }

  runtime::DeviceManager *
  createDeviceManager(const runtime::DeviceConfig &deviceConfig) override {
    return nullptr;
  }
};

/// MockBackendCustomIRGen used only for unit testing to test custom lowering
/// from Node to Instruction IR.
class MockBackendCustomIRGen : public Backend {
  class MockFunction : public CompiledFunction {
  public:
    MockFunction(runtime::RuntimeBundle &&bundle)
        : CompiledFunction(std::move(bundle)) {}

    Error execute(ExecutionContext *) override { return Error::success(); }

    std::string getCompileBackendName() const override { return "Interpreter"; }
  };

  std::string getBackendName() const override { return "Interpreter"; }

  Expected<std::unique_ptr<CompiledFunction>>
  compile(Function *F, const BackendOptions &) const override {
    return glow::make_unique<MockFunction>(runtime::RuntimeBundle::create(*F));
  }

  bool isOpSupported(const NodeInfo &NI) const override { return false; }

  runtime::DeviceManager *
  createDeviceManager(const runtime::DeviceConfig &deviceConfig) override {
    return nullptr;
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
          CN__->getGroup(), CN__->getDilation(), CN__->getLayout(),
          CN__->getFusedActivation());
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
/// \p convertToRowwiseQuantization then nodes supporting rowwise quantization
/// will converted to use it. \p schema represents the quantization schema to
/// use, if applicable. \p parallelCount represents the number of times to clone
/// the Function inside itself, so that testing can be done on architectures
/// that have parallel compute engines. The bias is quantized using the
/// precision \p biasElemKind. \p forceFP16AccumSLS is propagated into the
/// precision config for compilation.
void compareAgainstInterpreter(
    llvm::StringRef backendName, CreateAndInitFunction createAndInitFunction,
    ElemKind interpElemKind, ElemKind backendElemKind,
    float allowedError = 0.0001, unsigned parallelCount = 1,
    bool convertToRowwiseQuantization = false,
    quantization::Schema schema = quantization::Schema::Asymmetric,
    ElemKind biasElemKind = ElemKind::Int32QTy, bool forceFP16AccumSLS = false);

/// Given some \p FTP representing a Function with a single SaveNode and its
/// Tensor output, duplicate the Nodes in the Function and their Placeholder
/// inputs given \p bindings \p parallelCount times. \returns a set of Tensor
/// pointers for each output of the cloned Function. If the quantization node
/// info found in \p cctx exists, then all of the node infos will be cloned
/// accordingly with the names of the newly cloned nodes added to the Function.
std::unordered_set<Tensor *> cloneFunInsideFun(FunctionTensorPair FTP,
                                               PlaceholderBindings *bindings,
                                               CompilationContext &cctx,
                                               unsigned parallelCount);

/// \returns the number of nodes in \p F of kind \p kind.
unsigned countNodeKind(Function *F, Kinded::Kind kind);

void inferConvNet(Tensor *inputs, Tensor *filter, Tensor *bias, Tensor *out,
                  llvm::StringRef kind);

void trainConvNet(Tensor *inputs, Tensor *kernel1, Tensor *bias1,
                  Tensor *kernel2, Tensor *bias2, Tensor *selected,
                  llvm::ArrayRef<dim_t> shape1, llvm::ArrayRef<dim_t> shape2,
                  Tensor *out, llvm::StringRef kind);

void inferLocalResponseNormalizationNet(Tensor *inputs, Tensor *out,
                                        llvm::StringRef kind);

void trainLocalResponseNormalizationNet(Tensor *inputs, Tensor *weights,
                                        Tensor *bias, Tensor *selected,
                                        llvm::ArrayRef<dim_t> shape1,
                                        llvm::ArrayRef<dim_t> shape2,
                                        Tensor *out, llvm::StringRef kind);
void trainAvgPoolNet(Tensor *inputs, Tensor *weights, Tensor *bias,
                     Tensor *selected, llvm::ArrayRef<dim_t> shape1,
                     llvm::ArrayRef<dim_t> shape2, Tensor *out,
                     llvm::StringRef kind);

void trainMaxPoolNet(Tensor *inputs, Tensor *weights, Tensor *bias,
                     Tensor *selected, llvm::ArrayRef<dim_t> shape1,
                     llvm::ArrayRef<dim_t> shape2, Tensor *out,
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

/// A helper method to insert a compiledFunction \p func into the deviceManager
/// \p device.
void insertCompiledFunction(llvm::StringRef name, CompiledFunction *func,
                            runtime::DeviceManager *device, Module *mod);

/// A helper method to run the specified function \p name with the provided
/// ExecutionContext \p context on the specified DeviceManager \p device.
void runOnDevice(ExecutionContext &context, llvm::StringRef name,
                 runtime::DeviceManager *device);

/// Returns a new Constant of type UInt8FusedQTy with fused rowwise
/// quantization scales and offsets (i.e. the last 8 bytes of each row
/// contains the scale and offset).
Constant *createRandomFusedRowwiseQuantizedConstant(Module &mod,
                                                    llvm::ArrayRef<dim_t> dims,
                                                    llvm::StringRef name,
                                                    bool useFusedFP16 = false);

/// Returns a new Constant, of the provided \p type and \p dims initialized
/// with random data. If using floating point, then it is initialized via
/// Xavier with filterSize equal to twice the number of elements in \p dims.
/// Otherwise integer types are initialzed via their min and max values.
Constant *createRandomizedConstant(Module &mod, TypeRef type,
                                   llvm::ArrayRef<dim_t> dims,
                                   llvm::StringRef name);

/// Helper method to wrap dispatching an inference on function \p fname request
/// to a \p Hostmanager as a synchronous interface. If \p concurrentRequestsOpt
/// is set it will duplicate the request to send multiple requests concurrently.
void dispatchInference(const std::string &fname,
                       runtime::HostManager *hostManager,
                       ExecutionContext &context,
                       unsigned concurrentRequestsOpt);
} // namespace glow

#endif // GLOW_TESTS_BACKENDTESTUTILS_H
