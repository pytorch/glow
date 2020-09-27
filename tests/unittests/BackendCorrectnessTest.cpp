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
#include "glow/IR/IR.h"
#include "glow/IR/IRBuilder.h"
#include "glow/IR/Instrs.h"
#include "glow/Support/Random.h"

#include "gtest/gtest.h"

#include "llvm/ADT/STLExtras.h"

using namespace glow;
using llvm::cast;

class BackendCorrectnessTest : public ::testing::TestWithParam<std::string> {
protected:
  std::string backendName_{GetParam()};
};

TEST_P(BackendCorrectnessTest, convTest) {
  CHECK_IF_ENABLED();
  PseudoRNG PRNG;
  Tensor inputs(ElemKind::FloatTy, {20, 41, 32, 6});
  Tensor kernel(ElemKind::FloatTy, {10, 5, 5, 6});
  Tensor bias(ElemKind::FloatTy, {10});
  inputs.getHandle().initXavier(1, PRNG);
  kernel.getHandle().randomize(-3.0, 3.0, PRNG);
  bias.getHandle().randomize(-0.5, 0.5, PRNG);
  std::array<dim_t, 4> S{{20, 15, 12, 10}};
  llvm::ArrayRef<dim_t> shape(S);
  Tensor out1(ElemKind::FloatTy, shape);
  Tensor out2(ElemKind::FloatTy, shape);

  inferConvNet(&inputs, &kernel, &bias, &out1, backendName_);
  inferConvNet(&inputs, &kernel, &bias, &out2, "Interpreter");

  EXPECT_TRUE(out1.isEqual(out2));
}

TEST_P(BackendCorrectnessTest, extract3Dtest) {
  CHECK_IF_ENABLED();
  PseudoRNG PRNG;
  Tensor inputs(ElemKind::FloatTy, {5, 100, 100});
  inputs.getHandle().initXavier(1, PRNG);
  Tensor out1;
  Tensor out2;

  inferExtract3D(&inputs, &out1, backendName_);
  inferExtract3D(&inputs, &out2, "Interpreter");

  EXPECT_TRUE(out1.isEqual(out2));
}

TEST_P(BackendCorrectnessTest, quantizedConvTest) {
  CHECK_IF_ENABLED();
  PseudoRNG PRNG;
  Tensor inputs(ElemKind::Int8QTy, {20, 41, 32, 6}, 0.025, -7);
  Tensor kernel(ElemKind::Int8QTy, {10, 5, 5, 6}, 0.003, 3);
  Tensor bias(ElemKind::Int32QTy, {10}, 0.5, -4);
  inputs.getHandle<int8_t>().randomize(-128, 127, PRNG);
  kernel.getHandle<int8_t>().randomize(-128, 127, PRNG);
  bias.getHandle<int32_t>().randomize(-11, 8, PRNG);
  std::array<dim_t, 4> S{{20, 15, 12, 10}};
  llvm::ArrayRef<dim_t> shape(S);
  Tensor out1(ElemKind::Int8QTy, shape, 0.05, -17);
  Tensor out2(ElemKind::Int8QTy, shape, 0.05, -17);

  inferConvNet(&inputs, &kernel, &bias, &out1, backendName_);
  inferConvNet(&inputs, &kernel, &bias, &out2, "Interpreter");

  EXPECT_TRUE(out1.isEqual(out2, 1.0));
}

TEST_P(BackendCorrectnessTest, convGradTest) {
  CHECK_IF_ENABLED();
  PseudoRNG PRNG;
  Tensor inputs(ElemKind::FloatTy, {9, 8, 5, 4});
  Tensor kernel1(ElemKind::FloatTy, {3, 5, 3, 4});
  Tensor bias1(ElemKind::FloatTy, {3});
  Tensor kernel2(ElemKind::FloatTy, {2, 2, 2, 1});
  Tensor bias2(ElemKind::FloatTy, {2});
  Tensor selected(ElemKind::Int64ITy, {9, 1});
  inputs.getHandle().initXavier(1, PRNG);
  kernel1.getHandle().randomize(-1.0, 1.4, PRNG);
  bias1.getHandle().randomize(-0.2, 0.5, PRNG);
  kernel2.getHandle().randomize(-1.8, 2.3, PRNG);
  bias2.getHandle().randomize(-0.5, 1.0, PRNG);
  auto selectedH = selected.getHandle<int64_t>();
  for (size_t i = 0; i < 9; i++) {
    selectedH.raw(i) = PRNG.nextRandInt(0, 29);
  }
  std::array<dim_t, 4> S1{{9, 6, 10, 1}};
  llvm::ArrayRef<dim_t> shape1(S1);
  std::array<dim_t, 2> S2{{9, 30}};
  llvm::ArrayRef<dim_t> shape2(S2);
  Tensor out1;
  Tensor out2;

  trainConvNet(&inputs, &kernel1, &bias1, &kernel2, &bias2, &selected, shape1,
               shape2, &out1, backendName_);
  trainConvNet(&inputs, &kernel1, &bias1, &kernel2, &bias2, &selected, shape1,
               shape2, &out2, "Interpreter");

  EXPECT_TRUE(out1.isEqual(out2));
}

TEST_P(BackendCorrectnessTest, localResponseNormalizationTest) {
  CHECK_IF_ENABLED();
  PseudoRNG PRNG;
  Tensor inputs(ElemKind::FloatTy, {8, 15, 13, 30});
  inputs.getHandle().initXavier(1, PRNG);
  Tensor out1;
  Tensor out2;

  inferLocalResponseNormalizationNet(&inputs, &out1, backendName_);
  inferLocalResponseNormalizationNet(&inputs, &out2, "Interpreter");

  EXPECT_TRUE(out1.isEqual(out2));
}

TEST_P(BackendCorrectnessTest, localResponseNormalizationGradTest) {
  CHECK_IF_ENABLED();
  PseudoRNG PRNG;
  Tensor inputs(ElemKind::FloatTy, {5, 4, 7, 3});
  Tensor weights(ElemKind::FloatTy, {84, 180});
  Tensor bias(ElemKind::FloatTy, {180});
  Tensor selected(ElemKind::Int64ITy, {5, 1});
  inputs.getHandle().initXavier(1, PRNG);
  weights.getHandle().randomize(-2.0, 3.0, PRNG);
  bias.getHandle().randomize(-1.0, 1.3, PRNG);
  auto selectedH = selected.getHandle<int64_t>();
  for (size_t i = 0; i < 5; i++) {
    selectedH.raw(i) = PRNG.nextRandInt(0, 179);
  }
  std::array<dim_t, 4> S1{{5, 2, 2, 45}};
  llvm::ArrayRef<dim_t> shape1(S1);
  std::array<dim_t, 2> S2{{5, 180}};
  llvm::ArrayRef<dim_t> shape2(S2);
  Tensor out1(ElemKind::FloatTy, shape2);
  Tensor out2(ElemKind::FloatTy, shape1);

  trainLocalResponseNormalizationNet(&inputs, &weights, &bias, &selected,
                                     shape1, shape2, &out1, backendName_);
  trainLocalResponseNormalizationNet(&inputs, &weights, &bias, &selected,
                                     shape1, shape2, &out2, "Interpreter");

  EXPECT_TRUE(out1.isEqual(out2));
}

/// This is a mock backend wrapping the CPU backend. It is used only for unit
/// testing.
class MockCPUBackend : public BackendUsingGlowIR {
  // The actual backend being wrapped.
  std::unique_ptr<BackendUsingGlowIR> backend_;

public:
  MockCPUBackend() {
    backend_.reset(static_cast<BackendUsingGlowIR *>(createBackend("CPU")));
  }

  std::string getBackendName() const override { return "CPU"; }

  Expected<std::unique_ptr<CompiledFunction>>
  compile(Function *F, const BackendOptions &opts) const override {
    return backend_->compile(F, opts);
  }

  std::unique_ptr<CompiledFunction>
  compileIR(std::unique_ptr<IRFunction> IR) const override {
    return backend_->compileIR(std::move(IR));
  }
  bool isOpSupported(const NodeInfo &NI) const override { return true; }

  runtime::DeviceManager *
  createDeviceManager(const runtime::DeviceConfig &deviceConfig) override {
    return nullptr;
  }
};

TEST_P(BackendCorrectnessTest, dataParallelStackingTest) {
  CHECK_IF_ENABLED();
  // Create an activation of size 3 and create two overlapping tensorviews of
  // this activation. Perform data-parallel instructions involving those
  // tensorviews. The backend's logic for the creation of stacked kernels should
  // handle them correctly and avoid putting all those data-parallel
  // instructuins into the same kernel even though they are all
  // shape-compatible, because some of these instructions would mutate the same
  // buffer that is already used by other instructions in the stacked kernel.
  Module mod;
  Function *F = mod.createFunction("DataParallelStacking");
  auto M = glow::make_unique<IRFunction>(F);

  auto *var =
      mod.createPlaceholder(glow::ElemKind::FloatTy, {2}, "output", false);
  auto ctx = glow::make_unique<ExecutionContext>();
  auto *outputTensor = ctx->getPlaceholderBindings()->allocate(var);
  {
    // Scope the IRBuilder so the active allocations are properly deallocated at
    // destruction.
    IRBuilder bb(M.get());

    auto *output = bb.createWeightVar(glow::ElemKind::FloatTy, {2}, "output1",
                                      WeightVar::MutabilityKind::Mutable);

    M->getVariableMap()[var] = output;

    auto *act = bb.createAllocActivationInst(
        "act1", mod.uniqueType(glow::ElemKind::FloatTy, {3}));
    bb.createSplatInst("zero", act, 0.0);
    auto tv1 = bb.createTensorViewInst(
        "tv1", act, mod.uniqueType(glow::ElemKind::FloatTy, {2}), {0});
    auto tv2 = bb.createTensorViewInst(
        "tv2", act, mod.uniqueType(glow::ElemKind::FloatTy, {2}), {1});

    auto *one = bb.createAllocActivationInst(
        "act2", mod.uniqueType(glow::ElemKind::FloatTy, {2}));
    bb.createSplatInst("one", one, 1.0);
    // after this instruction:
    // act will be: [1, 1, 0]
    // tv1 will be: [1, 1]
    // tv2 will be: [1, 0]
    bb.createElementAddInst("elem_add1", tv1, tv1, one);
    // The next instruction should not be put into the same stacking kernel,
    // because tv2 overlaps with tv1.
    // after this instruction:
    // act will be: [1, 2, 2]
    // tv1 will be: [1, 2]
    // tv2 will be: [2, 2]
    bb.createElementAddInst("elem_add2", tv2, tv2, tv1);
    // after this instruction:
    // output will be: [3, 4]
    // tv1 will be: [1, 2]
    // tv2 will be: [2, 2]
    // If stacking would put elem_add1 and elem_add2 into the same stacking
    // kernel, the output would be: [2, 4], which is wrong.
    bb.createElementAddInst("elem_add3", output, tv2, tv1);
    bb.createDeallocActivationInst("dealloc", act);
  }

  MockCPUBackend backend;
  auto function = backend.compileIR(std::move(M));
  ASSERT_FALSE(ERR_TO_BOOL(function->execute(ctx.get())));
  auto H = outputTensor->getHandle();
  EXPECT_EQ(H.at(0), 3);
  EXPECT_EQ(H.at(1), 4);
}

TEST_P(BackendCorrectnessTest, AvgPoolGradTest) {
  CHECK_IF_ENABLED();
  PseudoRNG PRNG;
  Tensor inputs(ElemKind::FloatTy, {5, 7, 6, 3});
  Tensor weights(ElemKind::FloatTy, {126, 72});
  Tensor bias(ElemKind::FloatTy, {72});
  Tensor selected(ElemKind::Int64ITy, {5, 1});
  inputs.getHandle().initXavier(1, PRNG);
  weights.getHandle().randomize(-0.3, 0.6, PRNG);
  bias.getHandle().randomize(-0.2, 0.1, PRNG);
  auto selectedH = selected.getHandle<int64_t>();
  for (size_t i = 0; i < 5; i++) {
    selectedH.raw(i) = PRNG.nextRandInt(0, 17);
  }
  std::array<dim_t, 4> S1{{5, 6, 4, 3}};
  llvm::ArrayRef<dim_t> shape1(S1);
  std::array<dim_t, 2> S2{{5, 18}};
  llvm::ArrayRef<dim_t> shape2(S2);
  Tensor out1;
  Tensor out2;

  trainAvgPoolNet(&inputs, &weights, &bias, &selected, shape1, shape2, &out1,
                  backendName_);
  trainAvgPoolNet(&inputs, &weights, &bias, &selected, shape1, shape2, &out2,
                  "Interpreter");

  EXPECT_TRUE(out1.isEqual(out2));
}

TEST_P(BackendCorrectnessTest, MaxPoolGradTest) {
  CHECK_IF_ENABLED();
  PseudoRNG PRNG;
  Tensor inputs(ElemKind::FloatTy, {4, 8, 7, 2});
  Tensor weights(ElemKind::FloatTy, {112, 84});
  Tensor bias(ElemKind::FloatTy, {84});
  Tensor selected(ElemKind::Int64ITy, {4, 1});
  inputs.getHandle().initXavier(1, PRNG);
  weights.getHandle().randomize(-0.1, 0.7, PRNG);
  bias.getHandle().randomize(-0.3, 0.1, PRNG);
  auto selectedH = selected.getHandle<int64_t>();
  for (size_t i = 0; i < 4; i++) {
    selectedH.raw(i) = PRNG.nextRandInt(0, 31);
  }
  std::array<dim_t, 4> S1{{4, 6, 7, 2}};
  llvm::ArrayRef<dim_t> shape1(S1);
  std::array<dim_t, 2> S2{{4, 32}};
  llvm::ArrayRef<dim_t> shape2(S2);
  Tensor out1;
  Tensor out2;

  trainMaxPoolNet(&inputs, &weights, &bias, &selected, shape1, shape2, &out1,
                  backendName_);
  trainMaxPoolNet(&inputs, &weights, &bias, &selected, shape1, shape2, &out2,
                  "Interpreter");

  EXPECT_TRUE(out1.isEqual(out2));
}

TEST_P(BackendCorrectnessTest, intLookupTable) {
  CHECK_IF_ENABLED();
  PseudoRNG PRNG;
  constexpr dim_t inputSize = 100;
  Tensor inputs(ElemKind::Int8QTy, {inputSize}, 0.8, 4);
  inputs.getHandle<int8_t>().randomize(-128, 127, PRNG);
  Tensor out1, out2;

  // Mapping i -> i.
  std::vector<int8_t> initValues(256);
  for (dim_t i = 0; i < 256; ++i) {
    initValues[i] = i - 128;
  }

  inferIntLookupTableNet(&inputs, &out1, initValues, backendName_);
  inferIntLookupTableNet(&inputs, &out2, initValues, "Interpreter");

  EXPECT_TRUE(out1.isEqual(out2));
}

TEST_P(BackendCorrectnessTest, smallConv) {
  CHECK_IF_ENABLED();
  Tensor input(ElemKind::FloatTy, {1, 3, 3, 32});
  input.getHandle().clear(0.2);
  Tensor out1;
  Tensor out2;

  inferSmallConv(&input, &out1, backendName_);
  inferSmallConv(&input, &out2, "Interpreter");

  EXPECT_TRUE(out1.isEqual(out2));
}

/// This test targets the DKKC8 optimization.
TEST_P(BackendCorrectnessTest, groupConvTest) {
  CHECK_IF_ENABLED();
  std::array<dim_t, 4> S{{1, 2, 1, 128}};
  llvm::ArrayRef<dim_t> shape(S);
  Tensor out1(ElemKind::FloatTy, shape);
  Tensor out2(ElemKind::FloatTy, shape);
  inferGroupConv(&out1, backendName_);
  inferGroupConv(&out2, "Interpreter");

  EXPECT_TRUE(out1.isEqual(out2));
}

/// This test targets the DKKC8 optimization.
TEST_P(BackendCorrectnessTest, nonSquarePaddingConvTest) {
  CHECK_IF_ENABLED();
  Tensor out1;
  Tensor out2;

  inferNonSquarePaddingConv(&out1, backendName_);
  inferNonSquarePaddingConv(&out2, "Interpreter");

  EXPECT_TRUE(out1.isEqual(out2));
}

/// This non-square kernel test targets the DKKC8 optimization.
TEST_P(BackendCorrectnessTest, nonSquareKernelConvTest) {
  CHECK_IF_ENABLED();

  Tensor out1;
  Tensor out2;

  inferNonSquareKernelConv(&out1, backendName_);
  inferNonSquareKernelConv(&out2, "Interpreter");

  EXPECT_TRUE(out1.isEqual(out2));
}

/// This non-square stride test targets the DKKC8 optimization.
TEST_P(BackendCorrectnessTest, nonSquareStrideConvTest) {
  CHECK_IF_ENABLED();
  Tensor out1;
  Tensor out2;
  inferNonSquareStrideConv(&out1, backendName_);
  inferNonSquareStrideConv(&out2, "Interpreter");

  EXPECT_TRUE(out1.isEqual(out2));
}

/// This test targets the DKKC8 opt correctionimization.
TEST_P(BackendCorrectnessTest, convDKKC8Test) {
  CHECK_IF_ENABLED();
  Tensor out1;
  Tensor out2;
  inferConvDKKC8(&out1, backendName_);
  inferConvDKKC8(&out2, "Interpreter");
  EXPECT_TRUE(out1.isEqual(out2, 0.00013));
}

TEST_P(BackendCorrectnessTest, softmaxGradTest) {
  CHECK_IF_ENABLED();
  PseudoRNG PRNG;
  std::array<dim_t, 2> S{{8, 23}};
  llvm::ArrayRef<dim_t> shape(S);
  Tensor inputs(ElemKind::FloatTy, shape);
  Tensor weights(ElemKind::FloatTy, {23, 23});
  Tensor bias(ElemKind::FloatTy, {23});
  Tensor selected(ElemKind::Int64ITy, {8, 1});
  inputs.getHandle().initXavier(1, PRNG);
  weights.getHandle().randomize(0.0, 0.5, PRNG);
  bias.getHandle().randomize(-0.2, 0.0, PRNG);
  auto selectedH = selected.getHandle<int64_t>();
  for (size_t i = 0; i < 8; i++) {
    selectedH.raw(i) = PRNG.nextRandInt(0, 22);
  }
  Tensor out1;
  Tensor out2;

  trainSoftMaxNet(&inputs, &weights, &bias, &selected, &out1, backendName_);
  trainSoftMaxNet(&inputs, &weights, &bias, &selected, &out2, "Interpreter");

  EXPECT_TRUE(out1.isEqual(out2));
}

TEST_P(BackendCorrectnessTest, convOps) {
  CHECK_IF_ENABLED();
  PseudoRNG PRNG;
  // Construct networks with a different convolution depth.
  for (auto depth : {4, 64, 128}) {
    Tensor inputs(ElemKind::FloatTy, {2, 32, 16, 16});
    inputs.getHandle().initXavier(1, PRNG);
    Tensor out1;
    Tensor out2;

    inferBasicConvNet(&inputs, &out1, backendName_, depth);
    inferBasicConvNet(&inputs, &out2, "Interpreter", depth);

    EXPECT_TRUE(out1.isEqual(out2));
  }
}

TEST_P(BackendCorrectnessTest, basicFCNet) {
  CHECK_IF_ENABLED();
  compareAgainstInterpreter(GetParam(), createAndInitBasicFCNet,
                            ElemKind::FloatTy, ElemKind::FloatTy, 0.0004f,
                            parCloneCountOpt);
}

TEST_P(BackendCorrectnessTest, basicFCNetQuantized) {
  CHECK_IF_ENABLED();
  compareAgainstInterpreter(GetParam(), createAndInitBasicFCNet,
                            ElemKind::Int8QTy, ElemKind::Int8QTy, 0.f,
                            parCloneCountOpt);
}

TEST_P(BackendCorrectnessTest, complexNet1) {
  CHECK_IF_ENABLED();
  PseudoRNG PRNG;
  std::array<dim_t, 4> S{{8, 7, 14, 11}};
  llvm::ArrayRef<dim_t> shape(S);
  Tensor inputs1(ElemKind::FloatTy, shape);
  Tensor inputs2(ElemKind::FloatTy, {8, 4, 7, 9});
  Tensor inputs3(ElemKind::FloatTy, shape);
  Tensor inputs4(ElemKind::FloatTy, {8, 8, 7, 4});
  inputs1.getHandle().initXavier(1, PRNG);
  inputs2.getHandle().initXavier(1, PRNG);
  inputs3.getHandle().initXavier(1, PRNG);
  inputs4.getHandle().initXavier(1, PRNG);
  Tensor out1;
  Tensor out2;

  inferComplexNet1(&inputs1, &inputs2, &inputs3, &inputs4, &out1, backendName_);
  inferComplexNet1(&inputs1, &inputs2, &inputs3, &inputs4, &out2,
                   "Interpreter");

  EXPECT_TRUE(out1.isEqual(out2));
}

TEST_P(BackendCorrectnessTest, tinyResnet) {
  CHECK_IF_ENABLED();
  PseudoRNG PRNG;
  Tensor input(ElemKind::FloatTy, {1, 7, 7, 64});
  input.getHandle().randomize(0, 1.0, PRNG);

  std::vector<Tensor> weights;
  using Dims = llvm::ArrayRef<dim_t>;
  weights.emplace_back(ElemKind::FloatTy, Dims{256, 1, 1, 64});
  weights.emplace_back(ElemKind::FloatTy, Dims{256});
  weights.emplace_back(ElemKind::FloatTy, Dims{64, 1, 1, 256});
  weights.emplace_back(ElemKind::FloatTy, Dims{64});
  weights.emplace_back(ElemKind::FloatTy, Dims{64, 3, 3, 64});
  weights.emplace_back(ElemKind::FloatTy, Dims{64});
  weights.emplace_back(ElemKind::FloatTy, Dims{256, 1, 1, 64});
  weights.emplace_back(ElemKind::FloatTy, Dims{256});
  for (auto &T : weights) {
    T.getHandle().initXavier(10.0, PRNG);
  }

  Tensor out1;
  Tensor out2;
  inferTinyResnet(&input, &out1, weights, "Interpreter");
  inferTinyResnet(&input, &out2, weights, backendName_);

  EXPECT_TRUE(out1.isEqual(out2, 0.001));
}

// Test MaxSplat transformation in CPU backend.
TEST_P(BackendCorrectnessTest, maxSplatTest) {
  CHECK_IF_ENABLED();
  PseudoRNG PRNG;
  Tensor input(ElemKind::Int8QTy, {5, 5}, 0.001, -10);
  input.getHandle<int8_t>().randomize(-128, 127, PRNG);
  Tensor out1, out2;

  inferMaxSplat(&input, &out1, backendName_);
  inferMaxSplat(&input, &out2, "Interpreter");

  EXPECT_TRUE(out1.isEqual(out2));
}

void QuantizedConvReluFusionTest(quantization::Schema schema,
                                 std::string backendName_, int expectedFusion) {
  PseudoRNG PRNG;
  Tensor inputs(ElemKind::Int8QTy, {1, 6, 6, 16}, 1, 2);
  Tensor kernel(ElemKind::Int8QTy, {1, 3, 3, 16}, 1, 2);
  Tensor bias(ElemKind::Int32QTy, {1}, 1, 2);
  inputs.getHandle<int8_t>().randomize(0, 60, PRNG);
  kernel.getHandle<int8_t>().randomize(-1, 1, PRNG);
  bias.getHandle<int32_t>().randomize(0, 32768, PRNG);

  TensorQuantizationParams quantParams =
      chooseQuantizationParams({0.0, 6.0}, schema, ElemKind::Int8QTy);

  Tensor out(ElemKind::Int8QTy, {1, 6, 6, 1}, quantParams.scale,
             quantParams.offset);

  int resFusion =
      inferConvReluNet(&inputs, &kernel, &bias, &out, 3, 1, 1, backendName_);
  // In asymmetric case, Conv and Relu are not fused.
  EXPECT_EQ(resFusion, expectedFusion);
}

/// Symmetric Quantized Conv+Relu fusion testing for only OpenCL.
TEST_P(BackendCorrectnessTest, SymmetricQuantizedConvReluFusionTest) {
  CHECK_IF_ENABLED()
  QuantizedConvReluFusionTest(quantization::Schema::Symmetric, backendName_,
                              FusedActivation::RELU);
}

/// Asymmetric Quantized Conv+Relu fusion testing for only OpenCL.
TEST_P(BackendCorrectnessTest, AsymmetricQuantizedConvReluFusionTest) {
  CHECK_IF_ENABLED()
  QuantizedConvReluFusionTest(quantization::Schema::Asymmetric, backendName_,
                              FusedActivation::NONE);
}

INSTANTIATE_BACKEND_TEST(BackendCorrectnessTest);
