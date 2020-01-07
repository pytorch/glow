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

#include "BackendTestUtils.h"

#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Graph/Graph.h"
#include "glow/Graph/PlaceholderBindings.h"
#include "glow/Optimizer/GraphOptimizer/GraphOptimizer.h"
#include "glow/Support/Random.h"

#include "gtest/gtest.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Signals.h"

#include <functional>

using namespace glow;
using llvm::cast;

/// This matches the signature that is used for the parameterized tests here,
/// i.e. those passing three parameters via a single ::testing::Combine() into
/// GLOW_INSTANTIATE_TEST_SUITE_P_FOR_BACKEND_COMBINED_TEST().
using ThreeIntTupleConfig = std::tuple<std::string, std::tuple<int, int, int>>;
using FourIntTupleConfig =
    std::tuple<std::string, std::tuple<int, int, int, int>>;

#define SET_BACKEND_KIND_AND_THREE_INT_PARAMS(CONFIG, BACKEND_NAME, PARAM1,    \
                                              PARAM2, PARAM3)                  \
  std::tuple<int, int, int> threeIntTupleParams;                               \
  std::tie(BACKEND_NAME, threeIntTupleParams) = CONFIG;                        \
  std::tie(PARAM1, PARAM2, PARAM3) = threeIntTupleParams;

#define SET_BACKEND_KIND_AND_FOUR_INT_PARAMS(CONFIG, BACKEND_KIND, PARAM1,     \
                                             PARAM2, PARAM3, PARAM4)           \
  std::tuple<int, int, int, int> fourIntTupleParams;                           \
  std::tie(BACKEND_KIND, fourIntTupleParams) = CONFIG;                         \
  std::tie(PARAM1, PARAM2, PARAM3, PARAM4) = fourIntTupleParams;

//===--------------------------------------------------------------------===//
//                   Convolution Parameter Sweep Tests
//===--------------------------------------------------------------------===//

/// Create a simple network that has a single fp convolution.
static FunctionTensorPair
createAndInitConvNet(glow::PlaceholderBindings &bindings,
                     glow::ExecutionEngine &EE, dim_t size, dim_t convDepth,
                     dim_t kernel, dim_t stride, dim_t pad) {
  PseudoRNG PRNG;
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  auto *var = mod.createPlaceholder(ElemKind::FloatTy,
                                    {1, size, size, convDepth}, "var", false);
  bindings.allocate(var)->getHandle().initXavier(1, PRNG);

  auto *conv =
      F->createConv(bindings, "conv", var, convDepth, kernel, stride, pad, 1);
  bindings.get(cast<Placeholder>(conv->getFilter()))->getHandle().clear(0.1);
  bindings.get(cast<Placeholder>(conv->getBias()))->getHandle().clear(0.1);
  auto *result = F->createSave("ret", conv);
  auto *resultTensor = bindings.allocate(result->getPlaceholder());
  convertPlaceholdersToConstants(F, bindings, {var, result->getPlaceholder()});

  return std::make_pair(F, resultTensor);
}

/// Helper to test sweeping across a variety of configurations of a convolution
/// by comparing the results to the Interpreter given some \p allowedError.
/// \p config contains the backend to compare the Interpreter against, plus the
/// specific configuration to run for this test. \p interpElemKind and \p
/// backendElemKind are the element kinds to use for the Interpreter and
/// backend, respectively.
static void testParamSweepConv(ThreeIntTupleConfig config,
                               ElemKind interpElemKind,
                               ElemKind backendElemKind, float allowedError) {
  std::string backend;
  size_t size, depth, kernel;
  SET_BACKEND_KIND_AND_THREE_INT_PARAMS(config, backend, size, depth, kernel)

  LOG(INFO) << "Testing Conv with size: " << size << "; depth: " << depth
            << "; kernel: " << kernel << "\n";

  auto boundF = std::bind(createAndInitConvNet, std::placeholders::_1,
                          std::placeholders::_2, size, depth, kernel,
                          /* stride */ 1, /* pad */ 0);
  compareAgainstInterpreter(backend, boundF, interpElemKind, backendElemKind,
                            allowedError, parCloneCountOpt);
}

DECLARE_STATELESS_BACKEND_TEST(ConvSweepTest, ThreeIntTupleConfig);

GLOW_INSTANTIATE_TEST_SUITE_P_FOR_BACKEND_COMBINED_TEST(
    SweepTest, ConvSweepTest,
    ::testing::Combine(/* size */ ::testing::Values(5, 7, 15),
                       /* depth */ ::testing::Values(8, 64),
                       /* kernel */ ::testing::Values(1, 3)));

/// Compare backend against the interpreter in Float.
TEST_P(ConvSweepTest, ConvTest_Float) {
  CHECK_IF_ENABLED();
  testParamSweepConv(GetParam(), ElemKind::FloatTy, ElemKind::FloatTy, 0.0001f);
}

/// Compare backend against the interpreter in Int8.
TEST_P(ConvSweepTest, ConvTest_Int8) {
  CHECK_IF_ENABLED();
  testParamSweepConv(GetParam(), ElemKind::FloatTy, ElemKind::Int8QTy, 0.045f);
}

/// Compare backend against the interpreter in FP16.
TEST_P(ConvSweepTest, ConvTest_Float16) {
  CHECK_IF_ENABLED();
  testParamSweepConv(GetParam(), ElemKind::FloatTy, ElemKind::Float16Ty,
                     0.005f);
}

//===--------------------------------------------------------------------===//
//                   BatchMatMul Parameter Sweep Tests
//===--------------------------------------------------------------------===//

/// Create a simple network that has a single fp batch mat mul.
static FunctionTensorPair
createAndInitBatchMatMulNet(glow::PlaceholderBindings &bindings,
                            glow::ExecutionEngine &EE, dim_t N, dim_t A,
                            dim_t Z, dim_t B) {
  PseudoRNG PRNG;
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  auto *LHS = mod.createPlaceholder(ElemKind::FloatTy, {N, A, Z}, "LHS", false);
  auto *RHS = mod.createPlaceholder(ElemKind::FloatTy, {N, Z, B}, "RHS", false);
  bindings.allocate(LHS)->getHandle().initXavier(10, PRNG);
  bindings.allocate(RHS)->getHandle().initXavier(10, PRNG);

  auto *R = F->createBatchMatMul("BMM", LHS, RHS);

  auto *save = F->createSave("save", R);
  auto *resultTensor = bindings.allocate(save->getPlaceholder());

  return std::make_pair(F, resultTensor);
}

/// Helper to test sweeping across a variety of configurations of a BatchMatMul
/// by comparing the results to the Interpreter given some \p allowedError.
/// \p config contains the backend to compare the Interpreter against, plus the
/// specific configuration to run for this test. \p interpElemKind and \p
/// backendElemKind are the element kinds to use for the Interpreter and
/// backend, respectively.
static void testParamSweepBatchMatMul(ThreeIntTupleConfig config,
                                      ElemKind interpElemKind,
                                      ElemKind backendElemKind,
                                      float allowedError) {
  std::string backend;
  size_t N, A, Z;
  SET_BACKEND_KIND_AND_THREE_INT_PARAMS(config, backend, N, A, Z);
  size_t B = A;

  LOG(INFO) << "\n\tTesting BatchMatMul with N: " << N << "; A: " << A
            << "; Z: " << Z << "; B: " << B << "\n";

  // Multiplying LHS {N, A, Z} by RHS {N, Z, B} to get result {N, A, B}.
  auto boundF = std::bind(createAndInitBatchMatMulNet, std::placeholders::_1,
                          std::placeholders::_2, N, A, Z, B);
  compareAgainstInterpreter(backend, boundF, interpElemKind, backendElemKind,
                            allowedError, parCloneCountOpt);
}

DECLARE_STATELESS_BACKEND_TEST(BatchMatMulSweepTest, ThreeIntTupleConfig);

GLOW_INSTANTIATE_TEST_SUITE_P_FOR_BACKEND_COMBINED_TEST(
    SweepTest, BatchMatMulSweepTest,
    ::testing::Combine(/* N */ ::testing::Values(1, 4, 16, 24),
                       /* A */ ::testing::Range(10, 16),
                       /* Z */ ::testing::Values(32, 64, 128, 256)));

/// Compare backend against the interpreter in Float.
TEST_P(BatchMatMulSweepTest, BatchMatMulTest_Float) {
  CHECK_IF_ENABLED();
  testParamSweepBatchMatMul(GetParam(), ElemKind::FloatTy, ElemKind::FloatTy,
                            0.0001f);
}

/// Compare backend against the interpreter in Int8.
TEST_P(BatchMatMulSweepTest, BatchMatMulTest_Int8) {
  CHECK_IF_ENABLED();
  testParamSweepBatchMatMul(GetParam(), ElemKind::FloatTy, ElemKind::Int8QTy,
                            0.06f);
}

/// Compare backend against the interpreter in FP16.
TEST_P(BatchMatMulSweepTest, BatchMatMulTest_Float16) {
  CHECK_IF_ENABLED();
  testParamSweepBatchMatMul(GetParam(), ElemKind::FloatTy, ElemKind::Float16Ty,
                            0.005f);
}

//===--------------------------------------------------------------------===//
//                   FullyConnected Parameter Sweep Tests
//===--------------------------------------------------------------------===//

/// Create a simple network that has a single fp FC.
static FunctionTensorPair
createAndInitFCNet(glow::PlaceholderBindings &bindings,
                   glow::ExecutionEngine &EE, dim_t A, dim_t Z, dim_t B) {
  PseudoRNG PRNG;
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  auto *IP = mod.createPlaceholder(ElemKind::FloatTy, {A, Z}, "input", false);
  auto *WC = mod.createConstant(ElemKind::FloatTy, {Z, B}, "weights");
  auto *BC = mod.createConstant(ElemKind::FloatTy, {B}, "bias");
  bindings.allocate(IP)->getHandle().randomize(-0.2, 0.2, mod.getPRNG());
  BC->getPayloadMutable().getHandle().randomize(0, 0.000005, mod.getPRNG());
  WC->getPayloadMutable().getHandle().randomize(-0.4, 0.4, mod.getPRNG());

  auto *FC = F->createFullyConnected("FC", IP, WC, BC);
  auto *save = F->createSave("save", FC);
  auto *resultTensor = bindings.allocate(save->getPlaceholder());

  return std::make_pair(F, resultTensor);
}

/// Helper to test sweeping across a variety of configurations of a FC by
/// comparing the results to the Interpreter given some \p allowedError.
/// \p config contains the backend to compare the Interpreter against, plus the
/// specific configuration to run for this test. \p interpElemKind and \p
/// backendElemKind are the element kinds to use for the Interpreter and
/// backend, respectively.
static void testParamSweepFC(ThreeIntTupleConfig config,
                             ElemKind interpElemKind, ElemKind backendElemKind,
                             float allowedError) {
  std::string backend;
  size_t A, Z, B;
  SET_BACKEND_KIND_AND_THREE_INT_PARAMS(config, backend, A, Z, B);

  LOG(INFO) << "\n\tTesting FC with A: " << A << "; Z: " << Z << "; B: " << B
            << "\n";

  auto boundF = std::bind(createAndInitFCNet, std::placeholders::_1,
                          std::placeholders::_2, A, Z, B);
  compareAgainstInterpreter(backend, boundF, interpElemKind, backendElemKind,
                            allowedError, parCloneCountOpt);
}

DECLARE_STATELESS_BACKEND_TEST(FCSweepTest, ThreeIntTupleConfig);

GLOW_INSTANTIATE_TEST_SUITE_P_FOR_BACKEND_COMBINED_TEST(
    SweepTest, FCSweepTest,
    ::testing::Combine(
        /* A */ ::testing::Values(1, 4, 16, 64),
        /* Z */ ::testing::Values(16, 128, 256, 512, 1024, 2048, 4096),
        /* B */ ::testing::Values(1, 48, 64, 256, 1024)));

/// Compare backend against the interpreter in Float.
TEST_P(FCSweepTest, FCTest_Float) {
  CHECK_IF_ENABLED();
  testParamSweepFC(GetParam(), ElemKind::FloatTy, ElemKind::FloatTy, 0.0001f);
}

/// Compare backend against the interpreter in Int8.
TEST_P(FCSweepTest, FCTest_Int8) {
  CHECK_IF_ENABLED();
  testParamSweepFC(GetParam(), ElemKind::FloatTy, ElemKind::Int8QTy, 0.065f);
}

/// Compare backend against the interpreter in FP16.
TEST_P(FCSweepTest, FCTest_Float16) {
  CHECK_IF_ENABLED();
  testParamSweepFC(GetParam(), ElemKind::FloatTy, ElemKind::Float16Ty, 0.005f);
}

//===--------------------------------------------------------------------===//
//                   Concat Parameter Sweep Tests
//===--------------------------------------------------------------------===//

/// Create a simple network that has a single fp Concat.
static FunctionTensorPair
createAndInitConcatNet(glow::PlaceholderBindings &bindings,
                       glow::ExecutionEngine &EE, size_t numInputs,
                       size_t numDims, size_t maxLength, size_t axis) {
  PseudoRNG PRNG;
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  // Make leading dimensions smaller than trailing. Reduces size of tests and is
  // also in line with typical tests.
  std::vector<dim_t> dims(numDims, maxLength);
  for (size_t i = 0; i < numDims; i++) {
    dims[numDims - 1 - i] /= std::pow(2, i);
  }

  std::vector<NodeValue> inputs(numInputs);
  for (size_t i = 0; i < numInputs; i++) {
    auto *IP = mod.createPlaceholder(ElemKind::FloatTy, dims, "input", false);
    bindings.allocate(IP)->getHandle().randomize(-0.2, 0.2, mod.getPRNG());
    assert(IP);
    inputs[i] = IP->getOutput();
  }

  auto *concat = F->createConcat("concat", inputs, axis);
  auto *save = F->createSave("save", concat);
  auto *resultTensor = bindings.allocate(save->getPlaceholder());

  return std::make_pair(F, resultTensor);
}

/// Helper to test sweeping across a variety of configurations of a Concat by
/// comparing the results to the Interpreter given some \p allowedError.
/// \p config contains the backend to compare the Interpreter against, plus the
/// specific configuration to run for this test. \p interpElemKind and \p
/// backendElemKind are the element kinds to use for the Interpreter and
/// backend, respectively.
static void testParamSweepConcat(FourIntTupleConfig config,
                                 ElemKind interpElemKind,
                                 ElemKind backendElemKind, float allowedError) {
  std::string backend;
  size_t numInputs, numDims, maxLength, axis;
  SET_BACKEND_KIND_AND_FOUR_INT_PARAMS(config, backend, numInputs, numDims,
                                       maxLength, axis);
  // Exit if axis outside of numDims.
  if (axis >= numDims) {
    return;
  }

  LOG(INFO) << "\n\tTesting Concat with numInputs: " << numInputs
            << "; numDims: " << numDims << "; maxLength: " << maxLength
            << "; axis: " << axis << "\n";

  auto boundF =
      std::bind(createAndInitConcatNet, std::placeholders::_1,
                std::placeholders::_2, numInputs, numDims, maxLength, axis);
  compareAgainstInterpreter(backend, boundF, interpElemKind, backendElemKind,
                            allowedError, parCloneCountOpt);
}

DECLARE_STATELESS_BACKEND_TEST(ConcatSweepTest, FourIntTupleConfig);

GLOW_INSTANTIATE_TEST_SUITE_P_FOR_BACKEND_COMBINED_TEST(
    SweepTest, ConcatSweepTest,
    ::testing::Combine(/* numInputs */ ::testing::Values(1, 2, 4, 8, 16, 32, 64,
                                                         128, 192, 256),
                       /* numDims */ ::testing::Range(1, 4),
                       /* maxLength */ ::testing::Values(16, 32, 64, 128),
                       /* axis */ ::testing::Range(0, 3)));

/// Compare backend against the interpreter in Float.
TEST_P(ConcatSweepTest, ConcatTest_Float) {
  CHECK_IF_ENABLED();
  testParamSweepConcat(GetParam(), ElemKind::FloatTy, ElemKind::FloatTy, 0.0f);
}

/// Compare backend against the interpreter in Int8. Note that we do not use the
/// same ElemKind for the Interpreter; this is because the backend will
/// quantize/dequantize the input/result anyway, so the comparison wouldn't be
/// purely on data movement.
TEST_P(ConcatSweepTest, ConcatTest_Int8) {
  CHECK_IF_ENABLED();
  testParamSweepConcat(GetParam(), ElemKind::FloatTy, ElemKind::Int8QTy,
                       0.002f);
}

/// Compare backend against the interpreter in Float16. Note that we do not use
/// the same ElemKind for the Interpreter; this is because the backend will
/// down/up convert the input/result anyway, so the comparison wouldn't be
/// purely on data movement.
TEST_P(ConcatSweepTest, ConcatTest_Float16) {
  CHECK_IF_ENABLED();
  testParamSweepConcat(GetParam(), ElemKind::FloatTy, ElemKind::Float16Ty,
                       0.0001f);
}

//===--------------------------------------------------------------------===//
//                   SLWS Parameter Sweep Tests
//===--------------------------------------------------------------------===//

/// Create a simple network that has a single fp SLWS.
static FunctionTensorPair
createAndInitSLWSNet(glow::PlaceholderBindings &bindings,
                     glow::ExecutionEngine &EE, dim_t embeddingRows,
                     dim_t embeddingDim, dim_t numLengths, bool rowwiseQuantize,
                     bool fused, bool FP16, bool accumFP16) {
  PseudoRNG PRNG;
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  // Initialize lengths according to the number provided by the test. Note that
  // we arbitrarily set them between [80,120].
  auto *lengths =
      mod.createPlaceholder(ElemKind::Int32ITy, {numLengths}, "lengths", false);
  auto LH = bindings.allocate(lengths)->getHandle<int32_t>();
  LH.randomize(80, 120, mod.getPRNG());

  // Get the sum of the lengths to then use as the size for indices and weights.
  dim_t sumOfLengths = 0;
  for (const int32_t &e : LH) {
    sumOfLengths += e;
  }

  // Initialize indices to size of sum of lengths. Randomly set them to point
  // somewhere inside the embedding.
  auto *indices =
      mod.createPlaceholder(IndexElemKind, {sumOfLengths}, "indices", false);
  bindings.allocate(indices)->getHandle<sdim_t>().randomize(
      0, embeddingRows - 1, mod.getPRNG());

  // Xavier initialize the weights with the correct data type.
  Constant *weights;
  if (FP16) {
    weights =
        mod.createConstant(ElemKind::Float16Ty, {sumOfLengths}, "weights");
    weights->getPayloadMutable().getHandle<float16_t>().initXavier(
        weights->getType()->size() * 2, mod.getPRNG());
  } else {
    weights = mod.createConstant(ElemKind::FloatTy, {sumOfLengths}, "weights");
    weights->getPayloadMutable().getHandle<float>().initXavier(
        weights->getType()->size() * 2, mod.getPRNG());
  }

  // Create the embedding; non-RWQ versions will simply create a Constant with
  // it, while RWQ versions will use its data to create a RWQ Constant
  // internally in the Node constructor.
  Tensor embeddingT(ElemKind::FloatTy, {embeddingRows, embeddingDim});
  embeddingT.getHandle().initXavier(embeddingT.size() * 2, mod.getPRNG());

  // Create the SLWS based on provided options.
  Node *SLWS;
  if (!rowwiseQuantize) {
    auto *embeddingC = mod.createConstant("embedding", std::move(embeddingT));
    SLWS = F->createSparseLengthsWeightedSum("SLWS", embeddingC, weights,
                                             indices, lengths);
  } else {
    if (fused) {
      const ElemKind precision =
          FP16 ? ElemKind::UInt8FusedFP16QTy : ElemKind::UInt8FusedQTy;
      SLWS = F->createFusedRowwiseQuantizedSparseLengthsWeightedSum(
          "FRQSLWS", embeddingT, weights, indices, lengths, precision,
          accumFP16);
    } else {
      const ElemKind precision = FP16 ? ElemKind::Float16Ty : ElemKind::FloatTy;
      SLWS = F->createRowwiseQuantizedSparseLengthsWeightedSum(
          "RQSLWS", embeddingT, weights, indices, lengths,
          quantization::Schema::Asymmetric, precision, accumFP16);
    }
  }
  auto *save = F->createSave("save", SLWS);
  auto *resultTensor = bindings.allocate(save->getPlaceholder());

  return std::make_pair(F, resultTensor);
}

/// Helper to test sweeping across a variety of configurations of a SLWS by
/// comparing the results to the Interpreter given some \p allowedError.
/// \p config contains the backend to compare the Interpreter against, plus the
/// specific configuration to run for this test. \p interpElemKind and \p
/// backendElemKind are the element kinds to use for the Interpreter and
/// backend, respectively. Pass in options for the test \p rowwiseQuantize,
/// \p fused, \p FP16, and \p accumFP16.
static void testParamSweepSLWS(ThreeIntTupleConfig config,
                               ElemKind interpElemKind,
                               ElemKind backendElemKind, float allowedError,
                               bool rowwiseQuantize, bool fused, bool FP16,
                               bool accumFP16) {
  std::string backend;
  size_t embeddingRows, embeddingDim, numLengths;
  SET_BACKEND_KIND_AND_THREE_INT_PARAMS(config, backend, embeddingRows,
                                        embeddingDim, numLengths);

  LOG(INFO) << "\n\tTesting SLWS with embeddingRows: " << embeddingRows
            << "; embeddingDim: " << embeddingDim
            << "; numLengths: " << numLengths << "\n";

  auto boundF = std::bind(createAndInitSLWSNet, std::placeholders::_1,
                          std::placeholders::_2, embeddingRows, embeddingDim,
                          numLengths, rowwiseQuantize, fused, FP16, accumFP16);
  compareAgainstInterpreter(backend, boundF, interpElemKind, backendElemKind,
                            allowedError, parCloneCountOpt);
}

DECLARE_STATELESS_BACKEND_TEST(SLWSSweepTest, ThreeIntTupleConfig);

GLOW_INSTANTIATE_TEST_SUITE_P_FOR_BACKEND_COMBINED_TEST(
    SweepTest, SLWSSweepTest,
    ::testing::Combine(
        /* embeddingRows */ ::testing::Values(100, 1000, 10000, 100000),
        /* embeddingDim */ ::testing::Values(32, 64, 96, 128),
        /* numLengths */ ::testing::Values(16, 32, 64, 128, 256)));

/// Compare backend against the interpreter.
TEST_P(SLWSSweepTest, SLWS_Float) {
  CHECK_IF_ENABLED();
  testParamSweepSLWS(GetParam(), ElemKind::FloatTy, ElemKind::FloatTy,
                     0.000001f,
                     /* rowwiseQuantize */ false,
                     /* fused */ false, /* FP16 */ false,
                     /* accumFP16 */ false);
}

/// Compare backend against the interpreter in Float.
TEST_P(SLWSSweepTest, RWQSLWS_Float) {
  CHECK_IF_ENABLED();
  testParamSweepSLWS(GetParam(), ElemKind::FloatTy, ElemKind::FloatTy,
                     0.000001f,
                     /* rowwiseQuantize */ true,
                     /* fused */ false, /* FP16 */ false,
                     /* accumFP16 */ false);
}

/// Compare backend against the interpreter in Float.
TEST_P(SLWSSweepTest, FRWQSLWS_Float) {
  CHECK_IF_ENABLED();
  testParamSweepSLWS(GetParam(), ElemKind::FloatTy, ElemKind::FloatTy,
                     0.000001f,
                     /* rowwiseQuantize */ true,
                     /* fused */ true, /* FP16 */ false,
                     /* accumFP16 */ false);
}

/// Compare backend against the interpreter in Float.
TEST_P(SLWSSweepTest, RWQSLWS_Float16) {
  // Note: not currently enabled for any open-source backends, as only the
  // Interpreter supports this.
  CHECK_IF_ENABLED();
  testParamSweepSLWS(GetParam(), ElemKind::FloatTy, ElemKind::FloatTy,
                     0.000001f,
                     /* rowwiseQuantize */ true,
                     /* fused */ false, /* FP16 */ true,
                     /* accumFP16 */ false);
}

/// Compare backend against the interpreter in Float.
TEST_P(SLWSSweepTest, FRWQSLWS_Float16) {
  // Note: not currently enabled for any open-source backends, as only the
  // Interpreter supports this.
  CHECK_IF_ENABLED();
  testParamSweepSLWS(GetParam(), ElemKind::FloatTy, ElemKind::FloatTy,
                     0.000001f,
                     /* rowwiseQuantize */ true,
                     /* fused */ true, /* FP16 */ true,
                     /* accumFP16 */ false);
}

/// Compare backend against the interpreter in Float.
TEST_P(SLWSSweepTest, RWQSLWS_Float16_AccumFloat16) {
  // Note: not currently enabled for any open-source backends, as only the
  // Interpreter supports this.
  CHECK_IF_ENABLED();
  testParamSweepSLWS(GetParam(), ElemKind::FloatTy, ElemKind::FloatTy,
                     0.000001f,
                     /* rowwiseQuantize */ true,
                     /* fused */ false, /* FP16 */ true,
                     /* accumFP16 */ true);
}

/// Compare backend against the interpreter in Float.
TEST_P(SLWSSweepTest, FRWQSLWS_Float16_AccumFloat16) {
  // Note: not currently enabled for any open-source backends, as only the
  // Interpreter supports this.
  CHECK_IF_ENABLED();
  testParamSweepSLWS(GetParam(), ElemKind::FloatTy, ElemKind::FloatTy,
                     0.000001f,
                     /* rowwiseQuantize */ true,
                     /* fused */ true, /* FP16 */ true,
                     /* accumFP16 */ true);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);
  llvm::cl::ParseCommandLineOptions(argc, argv);
  return RUN_ALL_TESTS();
}
