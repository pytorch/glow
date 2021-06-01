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
#include "glow/IR/IR.h"
#include "glow/Optimizer/GraphOptimizer/GraphOptimizer.h"
#include "glow/Quantization/Base/Base.h"
#include "glow/Quantization/Base/Calibration.h"
#include "glow/Quantization/Base/Profile.h"
#include "glow/Quantization/Quantization.h"
#include "glow/Quantization/Serialization.h"

#include "gtest/gtest.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FileSystem.h"

namespace glow {

using llvm::cast;

class Quantization : public ::testing::Test {};

class Operator
    : public ::testing::TestWithParam<::std::tuple<std::string, std::string>> {
protected:
  ExecutionEngine profileEE{};
  ExecutionEngine backendSpecificEE{};

  void SetUp() override {
    std::string backend1;
    std::string backend2;
    std::tie(backend1, backend2) = GetParam();
    profileEE.setBackendName(backend1);
    backendSpecificEE.setBackendName(backend2);
  }
};

bool operator==(const std::vector<float> &lhs, const std::vector<float> &rhs) {
  return std::equal(lhs.begin(), lhs.end(), rhs.begin());
}

bool operator==(const NodeProfilingInfo &lhs, const NodeProfilingInfo &rhs) {
  return lhs.min() == rhs.min() && lhs.max() == rhs.max() &&
         lhs.nodeOutputName_ == rhs.nodeOutputName_ &&
         lhs.histogram() == rhs.histogram();
}

bool operator==(const NodeQuantizationInfo &lhs,
                const NodeQuantizationInfo &rhs) {
  return lhs.scale() == rhs.scale() && lhs.offset() == rhs.offset() &&
         lhs.nodeOutputName_ == rhs.nodeOutputName_;
}

/// This is a mock backend which extended support of quantized operators.
class MockQuantBackend : public Backend {
  // The actual backend being wrapped.
  std::unique_ptr<Backend> backend_;

public:
  MockQuantBackend() { backend_.reset(createBackend("Interpreter")); }

  std::string getBackendName() const override { return "Interpreter"; }

  Expected<std::unique_ptr<CompiledFunction>>
  compile(Function *F, const BackendOptions &opts) const override {
    return backend_->compile(F, opts);
  }

  runtime::DeviceManager *
  createDeviceManager(const runtime::DeviceConfig &deviceConfig) override {
    return nullptr;
  }

  bool isOpSupported(const NodeInfo &NI) const override {
    if (NI.getKind() == Kinded::Kind::SoftMaxNodeKind ||
        NI.getKind() == Kinded::Kind::LocalResponseNormalizationNodeKind ||
        NI.getKind() == Kinded::Kind::SaveNodeKind ||
        NI.getKind() == Kinded::Kind::ReluNodeKind ||
        NI.getKind() == Kinded::Kind::SelectNodeKind ||
        NI.getKind() == Kinded::Kind::LogNodeKind ||
        NI.getKind() == Kinded::Kind::SigmoidNodeKind ||
        NI.getKind() == Kinded::Kind::TanhNodeKind) {
      return true;
    }
    return backend_->isOpSupported(NI);
  }
};

/// Simple tests to verify the histogram rescale.
TEST(Quantization, rescaleHistogramTest) {
  EXPECT_EQ(quantization::rescaleHistogram({}, 0.0f, 1.0f, 0.0f, 2.0).size(),
            0);
  EXPECT_EQ(
      quantization::rescaleHistogram({1, 2, 3, 4}, 0.0f, 1.0f, -1.0f, 1.0),
      std::vector<float>({0, 0, 3, 7}));
  EXPECT_EQ(
      quantization::rescaleHistogram({2, 4, 6, 8}, -1.0f, 1.0f, 0.0f, 1.0),
      std::vector<float>({3, 3, 4, 4}));
}

/// Simple tests to verify the KL optimization.
TEST(Quantization, optimizeKLTest) {
  // Test that an all-zero histogram does not raise exceptions.
  std::vector<float> histAllZero(1000, 0);
  quantization::FloatRange rangeAllZero =
      quantization::optimizeKL(histAllZero, 0.f, 1.0f, 255);
  EXPECT_EQ(rangeAllZero.first, 0.f);
  EXPECT_EQ(rangeAllZero.second, 1.0f);

  // Test that an empty histogram does not raise exceptions.
  std::vector<float> histEmpty;
  quantization::FloatRange rangeEmpty =
      quantization::optimizeKL(histEmpty, 0.f, 1.0f, 255);
  EXPECT_EQ(rangeEmpty.first, 0.f);
  EXPECT_EQ(rangeEmpty.second, 1.0f);
}

void testProfilingInfosSerialization(std::vector<NodeProfilingInfo> &expected) {
  llvm::SmallVector<char, 10> resultPath;
  llvm::sys::fs::createTemporaryFile("prefix", "suffix", resultPath);
  std::string filePath(resultPath.begin(), resultPath.end());
  llvm::hash_code hash = 13;
  serializeProfilingInfosToYaml(filePath, hash, expected);
  std::vector<NodeProfilingInfo> deserialized;
  llvm::hash_code hashDeserialized;
  auto fileExists = deserializeProfilingInfosFromYaml(
      filePath, hashDeserialized, deserialized);
  llvm::sys::fs::remove(filePath);
  EXPECT_TRUE(fileExists);
  EXPECT_EQ(static_cast<size_t>(hash), static_cast<size_t>(hashDeserialized));
  EXPECT_EQ(expected, deserialized);
}

TEST(Quantization, DeserializeNonExistingFile) {
  std::string fakeFilePath = "/fake";
  std::vector<NodeProfilingInfo> deserialized;
  llvm::hash_code hashDeserialized;
  auto fileExists = deserializeProfilingInfosFromYaml(
      fakeFilePath, hashDeserialized, deserialized);
  EXPECT_FALSE(fileExists);
}

TEST(Quantization, ProfilingSerialize) {
  std::vector<float> histEmpty;
  std::vector<float> hist = {0, 1, 2, 3, 4};
  std::vector<NodeProfilingInfo> expected{{"first", {1.0, 10.0, histEmpty}},
                                          {"second", {-1.0, 3.0, hist}},
                                          {"third", {-10.0, 30.0, hist}},
                                          {"fourth", {0.1, 10.0, hist}},
                                          {"fifth", {0.123, 30.0, hist}}};
  testProfilingInfosSerialization(expected);
}

TEST(Quantization, ProfilingSerializePower2Range) {
  std::vector<NodeProfilingInfo> expected{
      {"pwr_0", {1.0000000000f, 1.0f}},   {"pwr_1", {0.5000000000f, 2.0f}},
      {"pwr_2", {0.2500000000f, 4.0f}},   {"pwr_3", {0.1250000000f, 8.0f}},
      {"pwr_4", {0.0625000000f, 16.0f}},  {"pwr_5", {0.0312500000f, 32.0f}},
      {"pwr_6", {0.0156250000f, 64.0f}},  {"pwr_7", {0.0078125000f, 128.0f}},
      {"pwr_8", {0.0039062500f, 256.0f}}, {"pwr_9", {0.0019531250f, 512.0f}}};
  testProfilingInfosSerialization(expected);
}

#if LLVM_VERSION_MAJOR < 8
TEST(Quantization, ProfilingSerializeEmpty) {
  std::vector<NodeProfilingInfo> expected;
  testProfilingInfosSerialization(expected);
}
#endif

TEST(Quantization, tensorAverageValue) {
  {
    float min = -10.0;
    float max = 10.0;
    std::vector<float> hist = {64, 64};
    TensorProfilingParams profParams(min, max, hist);
    float avgVal = quantization::getTensorAverageValue(profParams);
    EXPECT_FLOAT_EQ(avgVal, 0.0);
  }
  {
    float min = -10.0;
    float max = 10.0;
    std::vector<float> hist = {0, 64};
    TensorProfilingParams profParams(min, max, hist);
    float avgVal = quantization::getTensorAverageValue(profParams);
    EXPECT_FLOAT_EQ(avgVal, 5.0);
  }
  {
    float min = 0.0;
    float max = 10.0;
    std::vector<float> hist = {64, 0};
    TensorProfilingParams profParams(min, max, hist);
    float avgVal = quantization::getTensorAverageValue(profParams);
    EXPECT_FLOAT_EQ(avgVal, 2.5);
  }
}

template <typename From, typename To> static To clip(From in) {
  static_assert(sizeof(From) >= sizeof(To),
                "Clip should reduce the variable size");
  auto mx = std::numeric_limits<To>::max();
  auto mn = std::numeric_limits<To>::min();
  return std::max<From>(mn, std::min<From>(mx, in));
}

TEST(Quantization, quantScaleOffset) {
  // Test different scale values from 1<<-23 to 1>>1.
  float scales[] = {
      0.0000001596f, 0.00000025f, 0.000000995f, 0.0000035f, 0.00000952f,
      0.00000113f,   0.000721f,   0.0000721f,   0.0000172f, 0.0000951f,
      0.0000721f,    0.0000341f,  0.0000222f,   0.0000172f, 0.000752f,
      0.000371f,     0.000321f,   0.000223f,    0.000112f,  0.00852f,
      0.00671f,      0.00592f,    0.00200f,     0.00107f,   0.0931f,
      0.0721f,       0.031f,      0.014f,       0.0132f,    0.712f,
      0.613f,        0.412f,      0.223f,       0.134f,     1.0f,
      1.13f,         1.612f,      1.523f,       2.0f};

  // Try all scale factors:
  for (float scale : scales) {
    // Try all legal integers within the range:
    for (int8_t input = -128; input < 127; input++) {
      int32_t sum32num = round(input / scale);

      auto TR = quantization::quantizeScaleOffset32To8(scale, 0);
      int32_t computed = TR.transform(sum32num);

      EXPECT_NEAR(input, computed, 1);
    }
  }
}

TEST(Quantization, quantScaleOffsetPower2Scale) {
  // Test different power of 2 scale values (from 2^-10 to 2^1).
  float scales[] = {0.0009765625f, 0.0019531250f, 0.0039062500f, 0.0078125000f,
                    0.0156250000f, 0.0312500000f, 0.0625000000f, 0.1250000000f,
                    0.2500000000f, 0.5000000000f, 1.0000000000f, 2.0000000000f};

  // Try all scale factors:
  for (float scale : scales) {
    // Try all legal integers within the range:
    for (int8_t input = -128; input < 127; input++) {
      int32_t sum32num = round(input / scale);
      auto TR = quantization::quantizeScaleOffset32To8(scale, 0);
      EXPECT_EQ(quantization::isFloatPowerOf2(scale), true);
      EXPECT_EQ(TR.pre, 0);
      int exp = quantization::getFloat2Exp(scale);
      if (exp > 0) {
        EXPECT_EQ(TR.scale, (int)scale);
        EXPECT_EQ(TR.post, 0);
      } else {
        EXPECT_EQ(TR.scale, 1);
        EXPECT_EQ(TR.post, -exp);
      }
      int32_t computed = TR.transform(sum32num);
      EXPECT_NEAR(input, computed, 1);
    }
  }
}

template <class qtype>
void quantizeTensorTest(
    ElemKind qTy, quantization::Schema schema,
    quantization::Calibration calibration = quantization::Calibration::None) {
  // optimizeKL required histogram bins size to be atleast 255 so N is set to
  // 256
  dim_t N = 256;
  float maxValue = 255.0;
  if (qTy == ElemKind::Int8QTy) {
    N = 6;
    maxValue = 5.0;
    calibration = quantization::Calibration::None;
  }
  // Map float [0.0; maxValue] to a quantized type using its entire value range.
  std::vector<float> hist(N, 1);
  TensorQuantizationParams quantParams =
      chooseQuantizationParams({0.0, maxValue, hist}, schema, qTy, calibration);

  // Create an FP32 tensor with N elements and initialize it with numbers from 0
  // to maxValue.
  Tensor inputFP32(ElemKind::FloatTy, {N});
  Handle<float> THFP32 = inputFP32.getHandle<float>();
  for (unsigned i = 0; i < N; ++i) {
    THFP32.at({i}) = i * 1.0f;
  }

  // Quantize the tensor.
  auto quantizedFP32 =
      quantization::quantizeTensor(inputFP32, quantParams, qTy);
  // Check that the dequantized result is close to the original values before
  // the quantization.
  Handle<qtype> THquantizedFP32 = quantizedFP32.getHandle<qtype>();
  for (unsigned i = 0; i < N; ++i) {
    EXPECT_NEAR(THFP32.at({i}),
                quantization::dequantize(THquantizedFP32.at({i}), quantParams),
                0.05f);
  }

  // Create an FP16 tensor with N elements and initialize it with numbers from 0
  // to maxValue.
  Tensor inputFP16(ElemKind::Float16Ty, {N});
  Handle<float16> THFP16 = inputFP16.getHandle<float16>();
  for (unsigned i = 0; i < N; ++i) {
    THFP16.at({i}) = i * 1.0f;
  }

  // Quantize the tensor.
  auto quantizedFP16 =
      quantization::quantizeTensor(inputFP16, quantParams, qTy);
  // Check that the dequantized result is close to the original values before
  // the quantization.
  Handle<qtype> THquantizedFP16 = quantizedFP16.getHandle<qtype>();
  for (unsigned i = 0; i < N; ++i) {
    EXPECT_NEAR(THFP16.at({i}),
                quantization::dequantize(THquantizedFP16.at({i}), quantParams),
                0.05f);
  }
}

TEST(Quantization, quantizeTensorAsymmetricInt8) {
  quantizeTensorTest<int8_t>(ElemKind::Int8QTy,
                             quantization::Schema::Asymmetric);
}
TEST(Quantization, quantizeTensorAsymmetricInt8KLMinimization) {
  quantizeTensorTest<int8_t>(ElemKind::Int8QTy,
                             quantization::Schema::Asymmetric,
                             quantization::Calibration::KLMinimization);
}
TEST(Quantization, quantizeTensorAsymmetricInt16) {
  quantizeTensorTest<int16_t>(ElemKind::Int16QTy,
                              quantization::Schema::Asymmetric);
}
TEST(Quantization, quantizeTensorAsymmetricInt16KLMinimization) {
  quantizeTensorTest<int16_t>(ElemKind::Int16QTy,
                              quantization::Schema::Asymmetric,
                              quantization::Calibration::KLMinimization);
}
TEST(Quantization, quantizeTensorAsymmetricInt32) {
  quantizeTensorTest<int32_t>(ElemKind::Int32QTy,
                              quantization::Schema::Asymmetric);
}
TEST(Quantization, quantizeTensorAsymmetricInt32KLMinimization) {
  quantizeTensorTest<int32_t>(ElemKind::Int32QTy,
                              quantization::Schema::Asymmetric,
                              quantization::Calibration::KLMinimization);
}
TEST(Quantization, quantizeTensorSymmetricInt8) {
  quantizeTensorTest<int8_t>(ElemKind::Int8QTy,
                             quantization::Schema::Symmetric);
}
TEST(Quantization, quantizeTensorSymmetricInt8KLMinimization) {
  quantizeTensorTest<int8_t>(ElemKind::Int8QTy, quantization::Schema::Symmetric,
                             quantization::Calibration::KLMinimization);
}
TEST(Quantization, quantizeTensorSymmetricInt16) {
  quantizeTensorTest<int16_t>(ElemKind::Int16QTy,
                              quantization::Schema::Symmetric);
}
TEST(Quantization, quantizeTensorSymmetricInt16KLMinimization) {
  quantizeTensorTest<int16_t>(ElemKind::Int16QTy,
                              quantization::Schema::Symmetric,
                              quantization::Calibration::KLMinimization);
}
TEST(Quantization, quantizeTensorSymmetricInt32) {
  quantizeTensorTest<int32_t>(ElemKind::Int32QTy,
                              quantization::Schema::Symmetric);
}
TEST(Quantization, quantizeTensorSymmetricInt32KLMinimization) {
  quantizeTensorTest<int32_t>(ElemKind::Int32QTy,
                              quantization::Schema::Symmetric,
                              quantization::Calibration::KLMinimization);
}
TEST(Quantization, quantizeTensorSymmetricUInt8) {
  quantizeTensorTest<int8_t>(ElemKind::Int8QTy,
                             quantization::Schema::SymmetricWithUnsigned);
}
TEST(Quantization, quantizeTensorSymmetricUInt8KLMinimization) {
  quantizeTensorTest<int8_t>(ElemKind::Int8QTy,
                             quantization::Schema::SymmetricWithUnsigned,
                             quantization::Calibration::KLMinimization);
}
TEST(Quantization, quantizeTensorSymmetricUInt16) {
  quantizeTensorTest<int16_t>(ElemKind::Int16QTy,
                              quantization::Schema::SymmetricWithUnsigned);
}
TEST(Quantization, quantizeTensorSymmetricUInt16KLMinimization) {
  quantizeTensorTest<int16_t>(ElemKind::Int16QTy,
                              quantization::Schema::SymmetricWithUnsigned,
                              quantization::Calibration::KLMinimization);
}
TEST(Quantization, quantizeTensorSymmetricUInt32) {
  quantizeTensorTest<int32_t>(ElemKind::Int32QTy,
                              quantization::Schema::SymmetricWithUnsigned);
}
TEST(Quantization, quantizeTensorSymmetricUInt32KLMinimization) {
  quantizeTensorTest<int32_t>(ElemKind::Int32QTy,
                              quantization::Schema::SymmetricWithUnsigned,
                              quantization::Calibration::KLMinimization);
}
TEST(Quantization, quantizeTensorSymmetricPwr2Int8) {
  quantizeTensorTest<int8_t>(ElemKind::Int8QTy,
                             quantization::Schema::SymmetricWithPower2Scale);
}
TEST(Quantization, quantizeTensorSymmetricPwr2Int8KLMinimization) {
  quantizeTensorTest<int8_t>(ElemKind::Int8QTy,
                             quantization::Schema::SymmetricWithPower2Scale,
                             quantization::Calibration::KLMinimization);
}
TEST(Quantization, quantizeTensorSymmetricPwr2Int16) {
  quantizeTensorTest<int16_t>(ElemKind::Int16QTy,
                              quantization::Schema::SymmetricWithPower2Scale);
}
TEST(Quantization, quantizeTensorSymmetricPwr2Int16KLMinimization) {
  quantizeTensorTest<int16_t>(ElemKind::Int16QTy,
                              quantization::Schema::SymmetricWithPower2Scale,
                              quantization::Calibration::KLMinimization);
}
TEST(Quantization, quantizeTensorSymmetricPwr2Int32) {
  quantizeTensorTest<int32_t>(ElemKind::Int32QTy,
                              quantization::Schema::SymmetricWithPower2Scale);
}
TEST(Quantization, quantizeTensorSymmetricPwr2Int32KLMinimization) {
  quantizeTensorTest<int32_t>(ElemKind::Int32QTy,
                              quantization::Schema::SymmetricWithPower2Scale,
                              quantization::Calibration::KLMinimization);
}

/// Test 4-bit fused rowwise quantization.
template <typename T> void fused4BitRowwiseQuantizationTest(ElemKind qTy) {
  // Create an FP32 tensor with 12 elements and initialize it
  // with numbers from the following test inputs here.
  // 1. Input that contains at least one +ve, one -ve and zero.
  // 2. Input that contains at least one +ve and zero.
  // 3. Input that contains at least one -ve and zero.
  // 4. Input that contains at least only (+ve) numbers.
  // 5. Input that contains at least only (-ve) numbers.
  // 'deltas' is used to create the above 5 test cases hermetically.
  auto deltas = {-3, 0, 3, -7, 7};
  for (const auto &delta : deltas) {
    Tensor inputFP32(ElemKind::FloatTy, {2, 6});
    Tensor dequantized(ElemKind::FloatTy, {2, 6});
    dim_t col = inputFP32.dims()[1] / 2 + 2 * sizeof(T);
    Tensor quantized(qTy, {2, col}, /* dummy scale */ 1.0,
                     /* dummy offset */ 0);
    Handle<float> inputH = inputFP32.getHandle<float>();
    for (dim_t i = 0; i < 2; i++) {
      for (dim_t j = 0; j < 6; j++) {
        inputH.at({i, j}) = (i + j) * 1.0f + delta;
      }
    }

    quantization::tensorFusedRowwiseQuantization<T>(inputFP32, quantized);
    dequantized =
        quantization::tensor4BitsFusedRowwiseDequantization(quantized);

    Handle<float> dequantizedH = dequantized.getHandle<float>();
    for (dim_t i = 0; i < 2; i++) {
      for (dim_t j = 0; j < 6; j++) {
        EXPECT_NEAR(inputH.at({i, j}), dequantizedH.at({i, j}), 0.02f);
      }
    }
  }
}

/// Test 4-bit fused rowwise fp32 scale/offset quantization.
TEST(Quantization, fused4BitsFP32RowwiseQuantizeTensor) {
  fused4BitRowwiseQuantizationTest<float>(ElemKind::UInt4FusedQTy);
}

/// Test 4-bit fused rowwise fp16 quantization.
TEST(Quantization, fused4BitsFP16RowwiseQuantizeTensor) {
  fused4BitRowwiseQuantizationTest<float16_t>(ElemKind::UInt4FusedFP16QTy);
}

/// When quantizing a scalar the quantization should not lose precision: the
/// quantize->dequantize pair applied to a float scalar should preserve the
/// value (up to the precision lost by dividing/multiplying with the scale).
void quantizeScalarTest(float val, ElemKind qTy, quantization::Schema schema) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  PlaceholderBindings bindings;

  // Choose quantization parameters
  auto TQP = quantization::chooseQuantizationParams({val, val}, schema, qTy);

  // Create quantize/dequantize network for a single float value
  auto *input = mod.createPlaceholder(ElemKind::FloatTy, {1}, "val", false);
  auto inputQTy = mod.uniqueType(qTy, {1}, TQP.scale, TQP.offset);
  QuantizeNode *quant = F->createQuantize("quant", input, inputQTy);
  DequantizeNode *dequant =
      F->createDequantize("dequant", quant, ElemKind::FloatTy);
  SaveNode *save = F->createSave("save", dequant);

  // Allocate placeholders, set input, run, get output
  auto inpH = bindings.allocate(input)->getHandle();
  auto outH = bindings.allocate(save->getPlaceholder())->getHandle();
  inpH.at({0}) = val;
  EE.compile(CompilationMode::Infer);
  EE.run(bindings);
  float outVal = outH.raw(0);
  EXPECT_NEAR(val, outVal, 0.0000000001);
}

TEST(Quantization, quantizeScalarTestInt8) {
  quantizeScalarTest(0.0, ElemKind::Int8QTy, quantization::Schema::Asymmetric);
  quantizeScalarTest(0.0, ElemKind::Int8QTy, quantization::Schema::Symmetric);
  quantizeScalarTest(0.0, ElemKind::Int8QTy,
                     quantization::Schema::SymmetricWithUnsigned);
  quantizeScalarTest(1.3, ElemKind::Int8QTy, quantization::Schema::Asymmetric);
  quantizeScalarTest(1.3, ElemKind::Int8QTy, quantization::Schema::Symmetric);
  quantizeScalarTest(1.3, ElemKind::Int8QTy,
                     quantization::Schema::SymmetricWithUnsigned);
  quantizeScalarTest(-1.3, ElemKind::Int8QTy, quantization::Schema::Asymmetric);
  quantizeScalarTest(-1.3, ElemKind::Int8QTy, quantization::Schema::Symmetric);
  quantizeScalarTest(-1.3, ElemKind::Int8QTy,
                     quantization::Schema::SymmetricWithUnsigned);
}

TEST(Quantization, quantizeScalarTestInt16) {
  quantizeScalarTest(0.0, ElemKind::Int16QTy, quantization::Schema::Asymmetric);
  quantizeScalarTest(0.0, ElemKind::Int16QTy, quantization::Schema::Symmetric);
  quantizeScalarTest(0.0, ElemKind::Int16QTy,
                     quantization::Schema::SymmetricWithUnsigned);
  quantizeScalarTest(1.3, ElemKind::Int16QTy, quantization::Schema::Asymmetric);
  quantizeScalarTest(1.3, ElemKind::Int16QTy, quantization::Schema::Symmetric);
  quantizeScalarTest(1.3, ElemKind::Int16QTy,
                     quantization::Schema::SymmetricWithUnsigned);
  quantizeScalarTest(-1.3, ElemKind::Int16QTy,
                     quantization::Schema::Asymmetric);
  quantizeScalarTest(-1.3, ElemKind::Int16QTy, quantization::Schema::Symmetric);
  quantizeScalarTest(-1.3, ElemKind::Int16QTy,
                     quantization::Schema::SymmetricWithUnsigned);
}

TEST(Quantization, quantizeScalarTestInt32) {
  quantizeScalarTest(0.0, ElemKind::Int32QTy, quantization::Schema::Asymmetric);
  quantizeScalarTest(0.0, ElemKind::Int32QTy, quantization::Schema::Symmetric);
  quantizeScalarTest(0.0, ElemKind::Int32QTy,
                     quantization::Schema::SymmetricWithUnsigned);
  quantizeScalarTest(1.3, ElemKind::Int32QTy, quantization::Schema::Asymmetric);
  quantizeScalarTest(1.3, ElemKind::Int32QTy, quantization::Schema::Symmetric);
  quantizeScalarTest(1.3, ElemKind::Int32QTy,
                     quantization::Schema::SymmetricWithUnsigned);
  quantizeScalarTest(-1.3, ElemKind::Int32QTy,
                     quantization::Schema::Asymmetric);
  quantizeScalarTest(-1.3, ElemKind::Int32QTy, quantization::Schema::Symmetric);
  quantizeScalarTest(-1.3, ElemKind::Int32QTy,
                     quantization::Schema::SymmetricWithUnsigned);
}

/// Check corner case when bias is quantized as int32 with unconstrained
/// scale and offset parameters and used within a subtraction bias - biasOffset
/// which is expected to be within int32 limits.
static void quantizeBiasInt32CornerCaseTest(float val) {
  // Choose bias quantization parameters
  float biasF = val;
  auto biasTQP = quantization::chooseQuantizationParams(
      {biasF, biasF}, quantization::Schema::Asymmetric, ElemKind::Int32QTy);

  // Quantize the tensor.
  Tensor biasTF(ElemKind::FloatTy, {1});
  biasTF.getHandle<float>().at({0}) = biasF;
  auto biasTQ =
      quantization::quantizeTensor(biasTF, biasTQP, ElemKind::Int32QTy);
  int32_t biasQ = biasTQ.getHandle<int32_t>().at({0});
  int32_t biasOffset = biasTQP.offset;

  // Compute difference and check against int32 limits.
  int64_t diff = ((int64_t)biasQ) - ((int64_t)biasOffset);
  EXPECT_TRUE(std::numeric_limits<int32_t>::min() <= diff);
  EXPECT_TRUE(diff <= std::numeric_limits<int32_t>::max());
}

TEST(Quantization, quantizeBiasInt32CornerCaseTests) {
  quantizeBiasInt32CornerCaseTest(0.0);
  quantizeBiasInt32CornerCaseTest(0.3);
  quantizeBiasInt32CornerCaseTest(-0.3);
  quantizeBiasInt32CornerCaseTest(0.0000003);
  quantizeBiasInt32CornerCaseTest(-0.0000003);
  quantizeBiasInt32CornerCaseTest(30000000.0);
  quantizeBiasInt32CornerCaseTest(-30000000.0);
}

/// Verify the quantization utility function which performs finer grained
/// quantization along a given dimension for given \p qSchema and \p qTy.
template <class eTy>
static void quantizeTensorRowwise(quantization::Schema qSchema, ElemKind qTy) {
  dim_t numCols = 20;
  dim_t qDim = 0;
  dim_t qStep = 1;

  // Initialize tensors.
  Tensor tensor(ElemKind::FloatTy, {2, numCols});
  Tensor row1(ElemKind::FloatTy, {numCols});
  Tensor row2(ElemKind::FloatTy, {numCols});
  auto tensorH = tensor.getHandle<float>();
  auto row1H = row1.getHandle<float>();
  auto row2H = row2.getHandle<float>();
  for (dim_t idx = 0; idx < numCols; idx++) {
    tensorH.at({0, idx}) = float(idx);
    tensorH.at({1, idx}) = float(idx) - 128.0;
    row1H.raw(idx) = float(idx);
    row2H.raw(idx) = float(idx) - 128.0;
  }

  // Quantize rowwise using specialized function.
  Tensor scales(ElemKind::FloatTy, {2});
  Tensor offsets(ElemKind::Int32ITy, {2});
  getTensorQuantizationParams(tensor, scales, offsets, qSchema, qTy, qDim,
                              qStep);
  Tensor tensorQ =
      quantization::quantizeTensor(tensor, scales, offsets, qTy, qDim, qStep);
  auto tensorQH = tensorQ.getHandle<eTy>();
  auto scalesH = scales.getHandle<float>();
  auto offsetsH = offsets.getHandle<int32_t>();

  // Quantize rowwise using per-tensor functions.
  float row1Min = tensorH.at({0, 0});
  float row1Max = tensorH.at({0, numCols - 1});
  float row2Min = tensorH.at({1, 0});
  float row2Max = tensorH.at({1, numCols - 1});
  auto TQP1 =
      quantization::chooseQuantizationParams({row1Min, row1Max}, qSchema, qTy);
  auto TQP2 =
      quantization::chooseQuantizationParams({row2Min, row2Max}, qSchema, qTy);
  Tensor row1Q = quantization::quantizeTensor(row1, TQP1, qTy);
  Tensor row2Q = quantization::quantizeTensor(row2, TQP2, qTy);
  auto row1QH = row1Q.getHandle<eTy>();
  auto row2QH = row2Q.getHandle<eTy>();

  // Check.
  EXPECT_EQ(TQP1.scale, scalesH.raw(0));
  EXPECT_EQ(TQP2.scale, scalesH.raw(1));
  EXPECT_EQ(TQP1.offset, offsetsH.raw(0));
  EXPECT_EQ(TQP2.offset, offsetsH.raw(1));
  for (dim_t idx = 0; idx < 3; idx++) {
    EXPECT_EQ(tensorQH.at({0, idx}), row1QH.raw(idx));
    EXPECT_EQ(tensorQH.at({1, idx}), row2QH.raw(idx));
  }
}

TEST(Quantization, QuantizeTensorRowwiseTest) {
  quantizeTensorRowwise<int8_t>(quantization::Schema::Asymmetric,
                                ElemKind::Int8QTy);
  quantizeTensorRowwise<int16_t>(quantization::Schema::Asymmetric,
                                 ElemKind::Int16QTy);
  quantizeTensorRowwise<int32_t>(quantization::Schema::Asymmetric,
                                 ElemKind::Int32QTy);
  quantizeTensorRowwise<int8_t>(quantization::Schema::Symmetric,
                                ElemKind::Int8QTy);
  quantizeTensorRowwise<int16_t>(quantization::Schema::Symmetric,
                                 ElemKind::Int16QTy);
  quantizeTensorRowwise<int32_t>(quantization::Schema::Symmetric,
                                 ElemKind::Int32QTy);
  quantizeTensorRowwise<int8_t>(quantization::Schema::SymmetricWithUnsigned,
                                ElemKind::Int8QTy);
  quantizeTensorRowwise<int16_t>(quantization::Schema::SymmetricWithUnsigned,
                                 ElemKind::Int16QTy);
  quantizeTensorRowwise<int32_t>(quantization::Schema::SymmetricWithUnsigned,
                                 ElemKind::Int32QTy);
  quantizeTensorRowwise<int8_t>(quantization::Schema::SymmetricWithPower2Scale,
                                ElemKind::Int8QTy);
  quantizeTensorRowwise<int16_t>(quantization::Schema::SymmetricWithPower2Scale,
                                 ElemKind::Int16QTy);
  quantizeTensorRowwise<int32_t>(quantization::Schema::SymmetricWithPower2Scale,
                                 ElemKind::Int32QTy);
}

/// Helper for quantizing a simple Conv with precision \p quantizationPrecision
/// while the bias is quantized using \p quantizationPrecisionBias.
static void quantizeSimpleConvGraph(ElemKind quantizationPrecision,
                                    ElemKind quantizationPrecisionBias) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  auto *input =
      mod.createPlaceholder(ElemKind::FloatTy, {1, 4, 4, 1}, "input", false);
  auto *filter = mod.createConstant(ElemKind::FloatTy, {2, 2, 2, 1}, "filter");
  auto *bias = mod.createConstant(ElemKind::FloatTy, {2}, "bias");
  auto outTy = mod.uniqueType(ElemKind::FloatTy, {1, 4, 8, 2});
  PlaceholderBindings bindings;
  bindings.allocate(input)->getHandle().randomize(0.f, 2.f, mod.getPRNG());
  filter->getHandle().randomize(-1.0, 1.0, mod.getPRNG());
  bias->getHandle().randomize(-1.0, 1.0, mod.getPRNG());

  auto *CN = F->createConv("Conv", input, filter, bias, outTy, {2, 2}, {1, 1},
                           {0, 2, 1, 3}, 1);
  auto *S = F->createSave("ret", CN);
  bindings.allocate(S->getPlaceholder());

  quantization::QuantizationConfiguration quantConfig{{
      {input->getOutput().generateNodeOutputName(), {0.0f, 2.0f}},
      {filter->getOutput().generateNodeOutputName(), {0.0f, 3.0f}},
      {bias->getOutput().generateNodeOutputName(), {0.0f, 4.0f}},
      {CN->getResult().generateNodeOutputName(), {0.0f, 6.0f}},
  }};

  quantConfig.precision = quantizationPrecision;
  quantConfig.precisionBias = quantizationPrecisionBias;
  quantConfig.assertAllNodesQuantized = true;
  std::unique_ptr<Backend> backend(createBackend(EE.getBackendName()));
  quantization::quantizeFunction(F, quantConfig, *backend);

  // Make sure that graph can be compiled and run.
  EE.compile(CompilationMode::Infer);
  EE.run(bindings);
}

/// Test that a simple Conv graph can be quantized in Int8QTy and Int8QTy bias.
TEST(Quantization, QuantizeGraph_Int8_BiasInt8) {
  quantizeSimpleConvGraph(ElemKind::Int8QTy, ElemKind::Int8QTy);
}

/// Test that a simple Conv graph can be quantized in Int8QTy and Int32QTy bias.
TEST(Quantization, QuantizeGraph_Int8_BiasInt32) {
  quantizeSimpleConvGraph(ElemKind::Int8QTy, ElemKind::Int32QTy);
}

/// Test that a simple Conv graph can be quantized in Int16QTy and Int16QTy
/// bias.
TEST(Quantization, QuantizeGraph_Int16_BiasInt16) {
  quantizeSimpleConvGraph(ElemKind::Int16QTy, ElemKind::Int16QTy);
}

/// Test that a simple Conv graph can be quantized in Int16QTy and Int32QTy
/// bias.
TEST(Quantization, QuantizeGraph_Int16_BiasInt32) {
  quantizeSimpleConvGraph(ElemKind::Int16QTy, ElemKind::Int32QTy);
}

/// Test that when a node is quantized before its users are quantized then the
/// users correctly find the quantization parameters. This tests that updating
/// the nodeToTQP_ map in FunctionQuantizer::postProcessing() works correctly.
TEST(Quantization, TestQuantizedInputBeforeQuantizedNode) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  auto *input = mod.createPlaceholder(ElemKind::FloatTy, {3}, "input", true);
  PlaceholderBindings bindings;
  bindings.allocate(input)->getHandle().randomize(-1.0, 1.0, mod.getPRNG());

  // Note: Intentionally add successive reshapes so the GraphOptimizer merges
  // them and creates a new one. This way the newly created Reshape will be
  // placed at the end of the list of nodes in F, and then it will be quantized
  // before SN. I think this is the most straightforward way to cover the logic
  // path inside FunctionQuantizer::postProcessing() that updates nodeToTQP_.
  auto *reshape1 = F->createReshape("reshape1", input, {3, 1});
  auto *reshape2 = F->createReshape("reshape2", reshape1, {1, 3});
  auto *SN = F->createSlice("slice", reshape2, {0, 1}, {1, 2});
  auto *S = F->createSave("ret", SN);
  bindings.allocate(S->getPlaceholder());

  // We need to optimize here first so that the two reshapes are merged.
  optimize(F, CompilationMode::Infer);

  ReshapeNode *newReshape = llvm::dyn_cast<ReshapeNode>(SN->getInput());
  ASSERT_TRUE(newReshape);

  quantization::QuantizationConfiguration quantConfig{{
      {input->getOutput().generateNodeOutputName(), {-1.0, 1.0}},
      {newReshape->getResult().generateNodeOutputName(), {-1.0, 1.0}},
      {NodeValue::generateNodeOutputName(SN->getName().str()), {-1.0, 1.0}},
  }};

  quantConfig.assertAllNodesQuantized = true;
  std::unique_ptr<Backend> backend(createBackend(EE.getBackendName()));
  quantization::quantizeFunction(F, quantConfig, *backend);

  // Remove unnecessary conversions.
  optimize(F, CompilationMode::Infer);

  // Now we verify that the SliceNode was in fact quantized.
  {
    auto *saveNode = llvm::dyn_cast<SaveNode>(F->getNodeByName(S->getName()));
    ASSERT_TRUE(saveNode);
    auto *deqNode =
        llvm::dyn_cast<DequantizeNode>(saveNode->getInput().getNode());
    ASSERT_TRUE(deqNode);
    auto *sliceNode = llvm::dyn_cast<SliceNode>(deqNode->getInput().getNode());
    ASSERT_TRUE(sliceNode);
    EXPECT_TRUE(sliceNode->getResult().getType()->isQuantizedType());
  }
}

/// Test enabling RowwiseQuantizedFullyConnected in Glow quantization
/// procedure. A FC can be quantized and converted to a
/// RowwiseQuantizedFullyConnected if:
/// 1. The weights of FC is constant;
/// 2. Use -enable-rowwise option or set enableRowwise param in
/// quantization::quantizeFunction to true. In unittest, the later one is used.
static void
enableRowwiseQuantizedFullyConnected(ElemKind quantizationPrecision,
                                     ElemKind quantizationPrecisionBias) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  auto *input = mod.createPlaceholder(ElemKind::FloatTy, {1, 3}, "input", true);
  auto *W = mod.createPlaceholder(ElemKind::FloatTy, {3, 2}, "weights", true);
  auto *B = mod.createPlaceholder(ElemKind::FloatTy, {2}, "bias", true);
  PlaceholderBindings bindings;
  bindings.allocate(input)->getHandle().randomize(0.2f, 2.f, mod.getPRNG());
  bindings.allocate(W)->init(Tensor::InitKind::Xavier, 3, mod.getPRNG());
  bindings.allocate(B)->init(Tensor::InitKind::Broadcast, 0.1, mod.getPRNG());

  auto *WC = mod.createConstant(ElemKind::FloatTy, W->dims(), "wc");
  auto *FC = F->createFullyConnected("FC", input, WC, B);
  auto *S = F->createSave("ret", FC);
  bindings.allocate(S->getPlaceholder());

  LoweredInfoMap loweredMapForQuant;
  CompilationContext cctx(/* bindings */ nullptr, &loweredMapForQuant);
  ::glow::lower(F, cctx);

  // Get the MatMul node and the Batched_Add node.
  MatMulNode *matMul;
  BatchedAddNode *batchedAdd;
  for (Node &N : F->getNodes()) {
    if (N.getKind() == Kinded::Kind::MatMulNodeKind) {
      matMul = llvm::cast<MatMulNode>(&N);
    }
    if (N.getKind() == Kinded::Kind::BatchedAddNodeKind) {
      batchedAdd = llvm::cast<BatchedAddNode>(&N);
    }
  }
  ASSERT_TRUE(matMul);
  ASSERT_TRUE(batchedAdd);

  quantization::QuantizationConfiguration quantConfig{{
      {input->getOutput().generateNodeOutputName(), {0.2f, 2.0f}},
      {WC->getOutput().generateNodeOutputName(), {0.3f, 3.0f}},
      {B->getOutput().generateNodeOutputName(), {0.4f, 4.0f}},
      {matMul->getResult().generateNodeOutputName(), {0.6f, 6.0f}},
      {batchedAdd->getResult().generateNodeOutputName(), {0.6f, 6.0f}},
  }};

  quantConfig.precision = quantizationPrecision;
  quantConfig.precisionBias = quantizationPrecisionBias;
  quantConfig.enableRowwise = true;
  quantConfig.assertAllNodesQuantized = true;
  std::unique_ptr<Backend> backend(createBackend(EE.getBackendName()));
  quantization::quantizeFunction(F, quantConfig, *backend, loweredMapForQuant);

  // Check the graph structure after quantization.
  auto *saveNode = llvm::dyn_cast<SaveNode>(F->getNodeByName(S->getName()));
  ASSERT_TRUE(saveNode);
  auto *deqNode =
      llvm::dyn_cast<DequantizeNode>(saveNode->getInput().getNode());
  ASSERT_TRUE(deqNode);
  auto *rwNode = llvm::dyn_cast<RowwiseQuantizedFullyConnectedNode>(
      deqNode->getInput().getNode());
  ASSERT_TRUE(rwNode);
  auto *inNode = llvm::dyn_cast<QuantizeNode>(rwNode->getInput().getNode());
  ASSERT_TRUE(inNode);
  auto *biasNode = llvm::dyn_cast<QuantizeNode>(rwNode->getBias().getNode());
  ASSERT_TRUE(biasNode);
  auto *weightsNode = llvm::dyn_cast<Constant>(rwNode->getWeights().getNode());
  ASSERT_TRUE(weightsNode);
  auto *scalesNode = llvm::dyn_cast<Constant>(rwNode->getScales().getNode());
  ASSERT_TRUE(scalesNode);
  auto *offsetsNode = llvm::dyn_cast<Constant>(rwNode->getOffsets().getNode());
  ASSERT_TRUE(offsetsNode);

  // Make sure that graph can be compiled and run. We check the correctness of
  // RowwiseQuantizedFullyConnected in operatorTests.cpp.
  EE.compile(CompilationMode::Infer);

  EE.run(bindings);
}

TEST(Quantization, enableRowwiseQuantizedFullyConnected_Int8_BiasInt8) {
  enableRowwiseQuantizedFullyConnected(ElemKind::Int8QTy, ElemKind::Int8QTy);
}

TEST(Quantization, enableRowwiseQuantizedFullyConnected_Int8_BiasInt32) {
  enableRowwiseQuantizedFullyConnected(ElemKind::Int8QTy, ElemKind::Int32QTy);
}

/// Test enabling RowwiseQuantizedFullyConnected with Symmetric quantization.
TEST(Quantization, enableRowwiseQuantizedFullyConnectedSymmetric) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  PlaceholderBindings bindings;
  Function *F = mod.createFunction("main");

  auto *input = mod.createPlaceholder(ElemKind::FloatTy, {10, 80}, "in", false);
  auto *FC = F->createFullyConnected(bindings, "FC", input, 100);
  auto *res = F->createSave("save", FC);
  bindings.allocate(res->getPlaceholder());
  bindings.allocate(input)->getHandle().randomize(-1.f, 6.f, mod.getPRNG());

  ::glow::convertPlaceholdersToConstants(F, bindings,
                                         {input, res->getPlaceholder()});

  // Note that we generate values for the Weights because they will be used
  // during rowwise-quantization to select each row's scale/offset.
  auto *WC = llvm::cast<Constant>(FC->getWeights());
  WC->getPayloadMutable().getHandle().randomize(-0.7, 1.1, mod.getPRNG());
  auto *BC = llvm::cast<Constant>(FC->getBias());

  TensorProfilingParams inputTPP = {-1.0, 6.0};
  TensorProfilingParams matmulTPP = {0.0, 10.0};
  TensorProfilingParams batchedaddTPP = {0.0, 10.0};
  TensorProfilingParams biasTPP = {0, 20};

  TensorQuantizationParams inputTQP = chooseQuantizationParams(
      inputTPP, quantization::Schema::Symmetric, ElemKind::Int8QTy);
  TensorQuantizationParams matmulTQP = chooseQuantizationParams(
      matmulTPP, quantization::Schema::Symmetric, ElemKind::Int8QTy);
  TensorQuantizationParams batchedaddTQP = chooseQuantizationParams(
      batchedaddTPP, quantization::Schema::Symmetric, ElemKind::Int8QTy);
  TensorQuantizationParams biasTQP = chooseQuantizationParams(
      biasTPP, quantization::Schema::Symmetric, ElemKind::Int8QTy);

  EXPECT_EQ(inputTQP.offset, 0);
  EXPECT_EQ(matmulTQP.offset, 0);
  EXPECT_EQ(batchedaddTQP.offset, 0);
  EXPECT_EQ(biasTQP.offset, 0);

  LoweredInfoMap loweredMapForQuant;
  CompilationContext cctx(/* bindings */ nullptr, &loweredMapForQuant);
  ::glow::lower(F, cctx);

  // Get the MatMul node and the Batched_Add node.
  MatMulNode *matMul;
  BatchedAddNode *batchedAdd;
  for (Node &N : F->getNodes()) {
    if (N.getKind() == Kinded::Kind::MatMulNodeKind) {
      matMul = llvm::cast<MatMulNode>(&N);
    }
    if (N.getKind() == Kinded::Kind::BatchedAddNodeKind) {
      batchedAdd = llvm::cast<BatchedAddNode>(&N);
    }
  }
  ASSERT_TRUE(matMul);
  ASSERT_TRUE(batchedAdd);

  // Note: Using dummy offset for the weights, as it should be
  // rowwise-quantized.
  quantization::QuantizationConfiguration quantConfig{{
      {input->getOutput().generateNodeOutputName(), inputTPP},
      {WC->getOutput().generateNodeOutputName(), {-0.7, 1.1}},
      {BC->getOutput().generateNodeOutputName(), biasTPP},
      {matMul->getResult().generateNodeOutputName(), matmulTPP},
      {batchedAdd->getResult().generateNodeOutputName(), batchedaddTPP},
  }};

  quantConfig.schema = quantization::Schema::Symmetric;
  quantConfig.enableRowwise = true;
  quantConfig.assertAllNodesQuantized = true;
  std::unique_ptr<Backend> backend(createBackend(EE.getBackendName()));
  quantization::quantizeFunction(F, quantConfig, *backend, loweredMapForQuant);

  // Check the graph structure after quantization.
  auto *saveNode = llvm::dyn_cast<SaveNode>(F->getNodeByName(res->getName()));
  ASSERT_TRUE(saveNode);
  auto *deqNode =
      llvm::dyn_cast<DequantizeNode>(saveNode->getInput().getNode());
  ASSERT_TRUE(deqNode);
  auto *rwNode = llvm::dyn_cast<RowwiseQuantizedFullyConnectedNode>(
      deqNode->getInput().getNode());
  ASSERT_TRUE(rwNode);
  auto *inNode = llvm::dyn_cast<QuantizeNode>(rwNode->getInput().getNode());
  ASSERT_TRUE(inNode);
  auto *biasNode = llvm::dyn_cast<QuantizeNode>(rwNode->getBias().getNode());
  ASSERT_TRUE(biasNode);
  auto *weightsNode = llvm::dyn_cast<Constant>(rwNode->getWeights().getNode());
  ASSERT_TRUE(weightsNode);
  auto *scalesNode = llvm::dyn_cast<Constant>(rwNode->getScales().getNode());
  ASSERT_TRUE(scalesNode);
  auto *offsetsNode = llvm::dyn_cast<Constant>(rwNode->getOffsets().getNode());
  ASSERT_TRUE(offsetsNode);

  // Because we're using symmetric quantization, the offsets should all be zero.
  const auto offsetsH = offsetsNode->getPayload().getHandle<int32_t>();
  EXPECT_TRUE(offsetsH.isZero());

  // Make sure that graph can be compiled and run. We check the correctness of
  // RowwiseQuantizedFullyConnected in operatorTests.cpp.
  EE.compile(CompilationMode::Infer);

  EE.run(bindings);
}

/// Test enabling ChannelwiseQuantizedConv2D in the quantization procedure.
/// A standard Convolution node can be quantized and converted to a
/// ChannelwiseQuantizedConvolution if:
/// 1. The filter and bias are constants.
/// 2. Use -enable-channelwise option or set enableChannelwise param in
/// quantization::quantizeFunction to true.
static void enableChannelwiseQuantizedConv2D(ElemKind qPrec, ElemKind qPrecBias,
                                             quantization::Schema schema) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  PlaceholderBindings bindings;

  // Convolution parameters.
  std::vector<dim_t> inputDims = {5, 3, 3, 2};
  std::vector<dim_t> filterDims = {4, 2, 2, 1};
  std::vector<dim_t> biasDims = {4};
  std::vector<dim_t> outputDims = {5, 2, 2, 4};
  std::vector<unsigned_t> kernels = {2, 2};
  std::vector<unsigned_t> strides = {1, 1};
  std::vector<unsigned_t> pads = {0, 0, 0, 0};
  dim_t group = 2;
  std::vector<unsigned_t> dilation = {1, 1};

  // Create input placeholder.
  auto *input =
      mod.createPlaceholder(ElemKind::FloatTy, inputDims, "input", false);
  bindings.allocate(input)->getHandle<float>().randomize(-1.0, 1.0,
                                                         mod.getPRNG());

  // Create filter constant.
  auto *filterC = mod.createConstant(ElemKind::FloatTy, filterDims, "filterC");
  filterC->getPayloadMutable().getHandle<float>().randomize(-1.0, 1.0,
                                                            mod.getPRNG());

  // Create bias constant.
  auto *biasC = mod.createConstant(ElemKind::FloatTy, biasDims, "biasC");
  biasC->getPayloadMutable().getHandle<float>().randomize(-1.0, 1.0,
                                                          mod.getPRNG());

  // Create Convolution.
  auto *outTy = mod.uniqueType(ElemKind::FloatTy, outputDims);
  ConvolutionNode *conv =
      F->createConv("Conv", input, filterC, biasC, outTy, kernels, strides,
                    pads, group, dilation);
  SaveNode *save = F->createSave("save", conv);
  bindings.allocate(save->getPlaceholder());

  // Quantize function. Choose asymmetric ranges to test quantization params.
  quantization::QuantizationConfiguration quantConfig{{
      {input->getOutput().generateNodeOutputName(), {-2.0, 1.0}},
      {filterC->getOutput().generateNodeOutputName(), {-1.0, 2.0}},
      {biasC->getOutput().generateNodeOutputName(), {0.0, 3.0}},
      {conv->getResult().generateNodeOutputName(), {-3.0, 0.0}},
  }};
  quantConfig.schema = schema;
  quantConfig.precision = qPrec;
  quantConfig.precisionBias = qPrecBias;
  quantConfig.enableChannelwise = true;
  quantConfig.assertAllNodesQuantized = true;
  std::unique_ptr<Backend> backend(createBackend(EE.getBackendName()));
  quantization::quantizeFunction(F, quantConfig, *backend);

  // Check the graph structure after quantization.
  auto *saveNode = llvm::dyn_cast<SaveNode>(F->getNodeByName(save->getName()));
  ASSERT_TRUE(saveNode);
  auto *deqNode =
      llvm::dyn_cast<DequantizeNode>(saveNode->getInput().getNode());
  ASSERT_TRUE(deqNode);
  auto *cwqConvNode = llvm::dyn_cast<ChannelwiseQuantizedConvolutionNode>(
      deqNode->getInput().getNode());
  ASSERT_TRUE(cwqConvNode);
  auto *inputNode =
      llvm::dyn_cast<QuantizeNode>(cwqConvNode->getInput().getNode());
  ASSERT_TRUE(inputNode);
  auto *filterNode =
      llvm::dyn_cast<Constant>(cwqConvNode->getFilter().getNode());
  ASSERT_TRUE(filterNode);
  auto *biasNode = llvm::dyn_cast<Constant>(cwqConvNode->getBias().getNode());
  ASSERT_TRUE(biasNode);
  auto *filterScalesNode =
      llvm::dyn_cast<Constant>(cwqConvNode->getFilterScales().getNode());
  ASSERT_TRUE(filterScalesNode);
  auto *filterOffsetsNode =
      llvm::dyn_cast<Constant>(cwqConvNode->getFilterOffsets().getNode());
  ASSERT_TRUE(filterOffsetsNode);
  auto *biasScalesNode =
      llvm::dyn_cast<Constant>(cwqConvNode->getBiasScales().getNode());
  ASSERT_TRUE(biasScalesNode);
  auto *biasOffsetsNode =
      llvm::dyn_cast<Constant>(cwqConvNode->getBiasOffsets().getNode());
  ASSERT_TRUE(biasOffsetsNode);

  // Check precisions.
  ASSERT_EQ(inputNode->getResult().getElementType(), qPrec);
  ASSERT_EQ(filterNode->getOutput().getElementType(), qPrec);
  ASSERT_EQ(biasNode->getOutput().getElementType(), qPrecBias);
  ASSERT_EQ(filterScalesNode->getOutput().getElementType(), ElemKind::FloatTy);
  ASSERT_EQ(filterOffsetsNode->getOutput().getElementType(),
            ElemKind::Int32ITy);
  ASSERT_EQ(biasScalesNode->getOutput().getElementType(), ElemKind::FloatTy);
  ASSERT_EQ(biasOffsetsNode->getOutput().getElementType(), ElemKind::Int32ITy);
  ASSERT_EQ(cwqConvNode->getResult().getElementType(), qPrec);

  // Check quantization parameters.
  validateQuantizationParams({inputNode->getResult().getType()->getScale(),
                              inputNode->getResult().getType()->getOffset()},
                             schema, qPrec);
  validateQuantizationParams({cwqConvNode->getResult().getType()->getScale(),
                              cwqConvNode->getResult().getType()->getOffset()},
                             schema, qPrec);
  for (dim_t idx = 0; idx < outputDims[3]; idx++) {
    auto filterScalesH = filterScalesNode->getPayload().getHandle<float>();
    auto filterOffsetsH = filterOffsetsNode->getPayload().getHandle<int32_t>();
    auto biasScalesH = biasScalesNode->getPayload().getHandle<float>();
    auto biasOffsetsH = biasOffsetsNode->getPayload().getHandle<int32_t>();
    validateQuantizationParams(
        {filterScalesH.raw(idx), filterOffsetsH.raw(idx)}, schema, qPrec);
    validateQuantizationParams({biasScalesH.raw(idx), biasOffsetsH.raw(idx)},
                               schema, qPrecBias);
  }

  // Make sure that graph can be compiled and run. We check the correctness of
  // ChannelwiseQuantizedConvolution in OperatorTest.cpp.
  EE.compile(CompilationMode::Infer);
  EE.run(bindings);
}

TEST(Quantization, enableChannelwiseQuantizedConv2D_Int8_BiasInt8) {
  enableChannelwiseQuantizedConv2D(ElemKind::Int8QTy, ElemKind::Int8QTy,
                                   quantization::Schema::Asymmetric);
  enableChannelwiseQuantizedConv2D(ElemKind::Int8QTy, ElemKind::Int8QTy,
                                   quantization::Schema::Symmetric);
  enableChannelwiseQuantizedConv2D(ElemKind::Int8QTy, ElemKind::Int8QTy,
                                   quantization::Schema::SymmetricWithUnsigned);
  enableChannelwiseQuantizedConv2D(
      ElemKind::Int8QTy, ElemKind::Int8QTy,
      quantization::Schema::SymmetricWithPower2Scale);
}

TEST(Quantization, enableChannelwiseQuantizedConv2D_Int8_BiasInt32) {
  enableChannelwiseQuantizedConv2D(ElemKind::Int8QTy, ElemKind::Int32QTy,
                                   quantization::Schema::Asymmetric);
  enableChannelwiseQuantizedConv2D(ElemKind::Int8QTy, ElemKind::Int32QTy,
                                   quantization::Schema::Symmetric);
  enableChannelwiseQuantizedConv2D(ElemKind::Int8QTy, ElemKind::Int32QTy,
                                   quantization::Schema::SymmetricWithUnsigned);
  enableChannelwiseQuantizedConv2D(
      ElemKind::Int8QTy, ElemKind::Int32QTy,
      quantization::Schema::SymmetricWithPower2Scale);
}

/// Check that SLWS is correctly fused rowwise-quantized by the quantizer.
TEST(Quantization, enableRowwiseQuantizedSLWS) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  PlaceholderBindings bindings;

  auto *data = mod.createPlaceholder(ElemKind::FloatTy, {3, 1}, "data", false);
  auto *weights =
      mod.createPlaceholder(ElemKind::FloatTy, {8}, "weights", false);
  auto *indices =
      mod.createPlaceholder(ElemKind::Int64ITy, {8}, "indices", false);
  auto *lengths =
      mod.createPlaceholder(ElemKind::Int32ITy, {4}, "lengths", false);

  // Don't worry about allocating them as we are not going to run anyway.
  bindings.allocate(data);
  bindings.allocate(weights);
  bindings.allocate(indices);
  bindings.allocate(lengths);

  auto *SLWS = F->createSparseLengthsWeightedSum("SLWS", data, weights, indices,
                                                 lengths);
  auto *res = F->createSave("save", SLWS);
  ::glow::convertPlaceholdersToConstants(
      F, bindings, {indices, lengths, res->getPlaceholder()});
  bindings.allocate(res->getPlaceholder());

  quantization::QuantizationConfiguration quantConfig{{
      {SLWS->getData().generateNodeOutputName(), {0.2f, 2.0f}},
      {SLWS->getWeights().generateNodeOutputName(), {0.3f, 3.0f}},
      {SLWS->getResult().generateNodeOutputName(), {0.4f, 4.0f}},
  }};

  quantConfig.enableRowwise = true;
  quantConfig.assertAllNodesQuantized = true;
  std::unique_ptr<Backend> backend(createBackend(EE.getBackendName()));
  quantization::quantizeFunction(F, quantConfig, *backend);
  std::string saveName = std::string(res->getName());
  EE.compile(CompilationMode::Infer);

  // Check the graph structure after quantization.
  F = EE.getModule().getFunctions().front();
  auto *saveNode = llvm::dyn_cast<SaveNode>(F->getNodeByName(saveName));
  ASSERT_TRUE(saveNode);
  auto *FRWQSLWS =
      llvm::dyn_cast<FusedRowwiseQuantizedSparseLengthsWeightedSumNode>(
          saveNode->getInput().getNode());
  ASSERT_TRUE(FRWQSLWS);
}

/// Quantize ReLU node and make sure that quantized version
/// has quantization parameters mapping to non-negative floating
/// point range.
TEST(Quantization, quantizeReLU) {
  ExecutionEngine EE{};
  std::unique_ptr<Backend> backend(new MockQuantBackend);
  EE.setBackendName("Interpreter");
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  auto *input = mod.createPlaceholder(ElemKind::FloatTy, {1, 3}, "input", true);
  auto *relu = F->createRELU("ReLU", input);
  PlaceholderBindings bindings;
  auto *ret = F->createSave("ret", relu);
  std::string retName = std::string(ret->getName());
  // Make sure that offset quantization parameter of ReLU is set
  // such that it produces non-negative floating point range.
  quantization::QuantizationConfiguration quantConfig{
      {{input->getOutput().generateNodeOutputName(), {0.2f, 2.0f}},
       {relu->getResult().generateNodeOutputName(), {0.0f, 3.0f}}}};
  quantConfig.assertAllNodesQuantized = true;
  quantization::quantizeFunction(F, quantConfig, *backend);
  EE.compile(CompilationMode::Infer);

  // Compute tensor quantization parameters for verification.
  auto reluTQP = chooseQuantizationParams({0.0f, 3.0f}, quantConfig.schema,
                                          quantConfig.precision);

  F = EE.getModule().getFunctions().front();
  auto *save = llvm::cast<SaveNode>(F->getNodeByName(retName));
  ASSERT_TRUE(llvm::isa<DequantizeNode>(save->getInput().getNode()));
  auto *dequantize = llvm::cast<DequantizeNode>(save->getInput().getNode());
  ASSERT_TRUE(llvm::isa<MaxNode>(dequantize->getInput().getNode()));

  MaxNode *max = llvm::cast<MaxNode>(dequantize->getInput().getNode());
  ASSERT_TRUE(max->getResult().getType()->isQuantizedType());
  EXPECT_EQ(max->getResult().getType()->getOffset(), reluTQP.offset);
  EXPECT_EQ(max->getResult().getType()->getScale(), reluTQP.scale);
}

/// Quantize Log, Sigmoid, and Tanh nodes and make sure that quantized versions
/// are implemented as IntLookupTables, because the Interpreter only supports
/// them as such.
TEST(Quantization, quantizeLookupTables) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  auto *input = mod.createPlaceholder(ElemKind::FloatTy, {1, 3}, "input", true);
  auto *LN = F->createLog("log", input);
  auto *SN = F->createSigmoid("sigmoid", LN);
  auto *TN = F->createTanh("tanh", SN);
  auto *ret = F->createSave("ret", TN);

  quantization::QuantizationConfiguration quantConfig{
      {{input->getOutput().generateNodeOutputName(), {0.2f, 2.0f}},
       {LN->getResult().generateNodeOutputName(LN->getName().str()),
        {0.3f, 3.0f}},
       {SN->getResult().generateNodeOutputName(), {0.4f, 4.0f}},
       {TN->getResult().generateNodeOutputName(), {0.5f, 5.0f}}}};
  quantConfig.assertAllNodesQuantized = true;
  std::unique_ptr<Backend> backend(createBackend(EE.getBackendName()));
  quantization::quantizeFunction(F, quantConfig, *backend);
  optimize(F, CompilationMode::Infer);

  // Compute the quantization parameters based on the requirements of the
  // Sigmoid/Tanh or on the input/output values for Log.
  auto logInpTQP = chooseQuantizationParams({0.2, 2.0}, quantConfig.schema,
                                            quantConfig.precision);
  auto logOutTQP = chooseQuantizationParams({0.3, 3.0}, quantConfig.schema,
                                            quantConfig.precision);
  auto sigmoidInpTQP = chooseQuantizationParams({-6.0, 6.0}, quantConfig.schema,
                                                quantConfig.precision);
  auto sigmoidOutTQP = chooseQuantizationParams({0.0, 1.0}, quantConfig.schema,
                                                quantConfig.precision);
  auto tanhInpTQP = chooseQuantizationParams({-3.0, 3.0}, quantConfig.schema,
                                             quantConfig.precision);
  auto tanhOutTQP = chooseQuantizationParams({-1.0, 1.0}, quantConfig.schema,
                                             quantConfig.precision);

  auto *save = llvm::cast<SaveNode>(F->getNodeByName(ret->getName()));
  auto *dequantizeTanh =
      llvm::dyn_cast<DequantizeNode>(save->getInput().getNode());
  ASSERT_TRUE(dequantizeTanh);
  auto *tanhILT =
      llvm::dyn_cast<IntLookupTableNode>(dequantizeTanh->getInput().getNode());
  ASSERT_TRUE(tanhILT);
  EXPECT_FLOAT_EQ(tanhILT->getResult().getType()->getScale(), tanhOutTQP.scale);
  EXPECT_EQ(tanhILT->getResult().getType()->getOffset(), tanhOutTQP.offset);
  EXPECT_FLOAT_EQ(tanhILT->getInput().getType()->getScale(), tanhInpTQP.scale);
  EXPECT_EQ(tanhILT->getInput().getType()->getOffset(), tanhInpTQP.offset);

  auto *rescaleSigmoid =
      llvm::dyn_cast<RescaleQuantizedNode>(tanhILT->getInput().getNode());
  ASSERT_TRUE(rescaleSigmoid);
  auto *sigmoidILT =
      llvm::dyn_cast<IntLookupTableNode>(rescaleSigmoid->getInput().getNode());
  ASSERT_TRUE(sigmoidILT);
  EXPECT_FLOAT_EQ(sigmoidILT->getResult().getType()->getScale(),
                  sigmoidOutTQP.scale);
  EXPECT_EQ(sigmoidILT->getResult().getType()->getOffset(),
            sigmoidOutTQP.offset);
  EXPECT_FLOAT_EQ(sigmoidILT->getInput().getType()->getScale(),
                  sigmoidInpTQP.scale);
  EXPECT_EQ(sigmoidILT->getInput().getType()->getOffset(),
            sigmoidInpTQP.offset);

  auto *rescaleLog =
      llvm::dyn_cast<RescaleQuantizedNode>(sigmoidILT->getInput().getNode());
  ASSERT_TRUE(rescaleLog);
  auto *logILT =
      llvm::dyn_cast<IntLookupTableNode>(rescaleLog->getInput().getNode());
  ASSERT_TRUE(logILT);
  EXPECT_FLOAT_EQ(logILT->getResult().getType()->getScale(), logOutTQP.scale);
  EXPECT_EQ(logILT->getResult().getType()->getOffset(), logOutTQP.offset);
  EXPECT_FLOAT_EQ(logILT->getInput().getType()->getScale(), logInpTQP.scale);
  EXPECT_EQ(logILT->getInput().getType()->getOffset(), logInpTQP.offset);
}

/// Quantize Log, Sigmoid, and Tanh nodes and make sure that they are not
/// replaced by LookupTables because the backend supports them directly.
TEST(Quantization, quantizeWithoutLookupTables) {
  ExecutionEngine EE{};
  std::unique_ptr<Backend> backend(new MockQuantBackend);
  EE.setBackendName("Interpreter");
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  auto *input = mod.createPlaceholder(ElemKind::FloatTy, {1, 3}, "input", true);
  auto *LN = F->createLog("log", input);
  auto *SN = F->createSigmoid("sigmoid", LN);
  auto *TN = F->createTanh("tanh", SN);
  auto *ret = F->createSave("ret", TN);

  quantization::QuantizationConfiguration quantConfig{
      {{input->getOutput().generateNodeOutputName(), {0.2f, 2.0f}},
       {LN->getResult().generateNodeOutputName(), {0.3f, 3.0f}},
       {SN->getResult().generateNodeOutputName(), {0.4f, 4.0f}},
       {TN->getResult().generateNodeOutputName(), {0.5f, 5.0f}}}};
  quantConfig.assertAllNodesQuantized = true;
  quantization::quantizeFunction(F, quantConfig, *backend);
  optimize(F, CompilationMode::Infer);

  // Compute the quantization parameters for validation.
  auto logInpTQP = chooseQuantizationParams({0.2, 2.0}, quantConfig.schema,
                                            quantConfig.precision);
  auto logOutTQP = chooseQuantizationParams({0.3, 3.0}, quantConfig.schema,
                                            quantConfig.precision);
  auto sigmoidInpTQP = chooseQuantizationParams({0.3, 3.0}, quantConfig.schema,
                                                quantConfig.precision);
  auto sigmoidOutTQP = chooseQuantizationParams(
      {0.4f, 4.0f}, quantConfig.schema, quantConfig.precision);
  auto tanhInpTQP = chooseQuantizationParams({0.4f, 4.0f}, quantConfig.schema,
                                             quantConfig.precision);
  auto tanhOutTQP = chooseQuantizationParams({0.5f, 5.0f}, quantConfig.schema,
                                             quantConfig.precision);

  auto *save = llvm::cast<SaveNode>(F->getNodeByName(ret->getName()));
  auto *dequantize = llvm::dyn_cast<DequantizeNode>(save->getInput().getNode());
  ASSERT_TRUE(dequantize);
  auto *tanh = llvm::dyn_cast<TanhNode>(dequantize->getInput());
  ASSERT_TRUE(tanh);
  EXPECT_FLOAT_EQ(tanh->getResult().getType()->getScale(), tanhOutTQP.scale);
  EXPECT_EQ(tanh->getResult().getType()->getOffset(), tanhOutTQP.offset);
  EXPECT_FLOAT_EQ(tanh->getInput().getType()->getScale(), tanhInpTQP.scale);
  EXPECT_EQ(tanh->getInput().getType()->getOffset(), tanhInpTQP.offset);

  auto *sigmoid = llvm::dyn_cast<SigmoidNode>(tanh->getInput());
  ASSERT_TRUE(sigmoid);
  EXPECT_FLOAT_EQ(sigmoid->getResult().getType()->getScale(),
                  sigmoidOutTQP.scale);
  EXPECT_EQ(sigmoid->getResult().getType()->getOffset(), sigmoidOutTQP.offset);
  EXPECT_FLOAT_EQ(sigmoid->getInput().getType()->getScale(),
                  sigmoidInpTQP.scale);
  EXPECT_EQ(sigmoid->getInput().getType()->getOffset(), sigmoidInpTQP.offset);

  auto *log = llvm::dyn_cast<LogNode>(sigmoid->getInput());
  ASSERT_TRUE(log);
  EXPECT_FLOAT_EQ(log->getResult().getType()->getScale(), logOutTQP.scale);
  EXPECT_EQ(log->getResult().getType()->getOffset(), logOutTQP.offset);
  EXPECT_FLOAT_EQ(log->getInput().getType()->getScale(), logInpTQP.scale);
  EXPECT_EQ(log->getInput().getType()->getOffset(), logInpTQP.offset);
}

/// Fills the tensor \p H with some stable random data with the seed \p seed
/// and the range [-scale .. scale].
static void fillStableRandomData(Handle<float> H, size_t seed,
                                 float scale = 1) {
  for (size_t i = 0, e = H.size(); i < e; i++) {
    H.raw(i) = scale * (float((int(i * 1921 + seed) % 100) - 50) / 50);
  }
}

/// Builds a simple graph, returns back input var and save node through refs.
static Function *createSimpleGraphForQuantization(Module *M,
                                                  PlaceholderBindings &bindings,
                                                  Placeholder *A,
                                                  Placeholder *B,
                                                  llvm::StringRef funcName) {
  Function *F = M->createFunction(funcName);

  fillStableRandomData(bindings.allocate(A)->getHandle(), 1100, 1);

  fillStableRandomData(bindings.allocate(B)->getHandle(), 2001, 1);

  ConvolutionNode *CV = F->createConv(bindings, "conv", A, 16, 5, 1, 2, 2);
  auto *bias = cast<Placeholder>(CV->getBias());
  auto *filter = cast<Placeholder>(CV->getFilter());
  fillStableRandomData(bindings.get(bias)->getHandle(), 2001, 1);
  fillStableRandomData(bindings.get(filter)->getHandle(), 1000, 1);

  auto *RL = F->createRELU("relu", CV);
  auto *MP = F->createMaxPool("maxPool", RL, 2, 2, 1);
  // Just add noop transpose.
  auto *T = F->createTranspose("transpose", MP->getResult(), {0, 1, 2, 3});
  // Noop reshape, make sure conversion quantization procedure works well.
  auto *R = F->createReshape("reshape", T, T->getResult().dims());
  auto *AP = F->createAvgPool("avgPool", R, 2, 2, 1);

  FullyConnectedNode *FC = F->createFullyConnected(bindings, "fc", AP, 10);

  // Noop slice, make sure conversion quantization procedure works well.
  auto *S =
      F->createSlice("slice", FC, {0, 1},
                     {FC->getResult().dims()[0], FC->getResult().dims()[1]});
  auto *bias2 = cast<Placeholder>(FC->getBias());
  auto *filter2 = cast<Placeholder>(FC->getWeights());

  fillStableRandomData(bindings.get(bias2)->getHandle(), 3001, 1);
  fillStableRandomData(bindings.get(filter2)->getHandle(), 4000, 1);

  auto *CN = F->createConcat("concat", {S, B}, 0);
  auto *SP = F->createSplat("splat", B->getType(), 10.0);
  auto *O = F->createConcat("concat", {CN, SP}, 0);
  auto *TN = F->createTranspose("transpose", O, {1, 0});
  auto *BRAN = F->createBatchedReduceAdd("batchedreduceadd", TN, 0);
  auto *TLN = F->createTile("tile", BRAN, 2, 0);
  auto *SN = F->createSplat("splat", TLN->getResult().getType(), 100.0);
  auto *MN = F->createMax("max", SN, TLN);
  auto *CLTE = F->createCmpLTE("cmplte", MN, SN);
  auto *SLN = F->createSelect("select", CLTE, SN, MN);
  auto *save = F->createSave("save", SLN);
  bindings.allocate(save->getPlaceholder());
  return F;
}

/// Helper for an end to end test profiling a model on \p profileEE, then
/// quantizing and running it on \p backendSpecificEE, quantizing with precision
/// \p quantizationPrecision and disabling quantization for all Kinds in
/// \p keepOriginalPrecisionForNodes. Results are compared from the profiling
/// run and quantization run.
static void
testQuantizationEnd2End(ExecutionEngine &profileEE,
                        ExecutionEngine &backendSpecificEE,
                        ElemKind quantizationPrecision,
                        const KindSet &keepOriginalPrecisionForNodes = {}) {
  auto *mod = &profileEE.getModule();
  auto *modBackend = &backendSpecificEE.getModule();
  PlaceholderBindings bindings;
  PlaceholderBindings bindingsBackend;

  auto *A =
      mod->createPlaceholder(ElemKind::FloatTy, {1, 32, 32, 2}, "A", false);
  auto *B = mod->createPlaceholder(ElemKind::FloatTy, {10, 9}, "B", false);
  auto *AB = modBackend->createPlaceholder(ElemKind::FloatTy, {1, 32, 32, 2},
                                           "A", false);
  auto *BB =
      modBackend->createPlaceholder(ElemKind::FloatTy, {10, 9}, "B", false);

  // STEP1 - Generate the first network to record the quantization parameters.
  createSimpleGraphForQuantization(mod, bindings, A, B, "main");
  createSimpleGraphForQuantization(modBackend, bindingsBackend, AB, BB, "main");

  LoweredInfoMap loweredMapForProf;
  CompilationContext cctxProf{&bindings, &loweredMapForProf};
  cctxProf.precisionConfig.quantMode = QuantizationMode::Profile;
  profileEE.compile(cctxProf);
  bindings.allocate(mod->getPlaceholders());

  // Run graph to capture profile.
  profileEE.run(bindings, "main");

  // STEP2 - Use the profile to quantize a network.
  LoweredInfoMap loweredMapForQuant;
  CompilationContext cctxQuant{&bindings, &loweredMapForQuant};

  // Get quantization infos and build new quantized graph.
  PrecisionConfiguration &precConfig = cctxQuant.precisionConfig;
  precConfig.quantMode = QuantizationMode::Quantize;
  precConfig.quantConfig.infos = quantization::generateNodeProfilingInfos(
      bindings, mod->getFunctions().front(), loweredMapForProf);
  precConfig.quantConfig.precision = quantizationPrecision;
  precConfig.quantConfig.assertAllNodesQuantized = true;
  precConfig.precisionModeKindSet = keepOriginalPrecisionForNodes;

  backendSpecificEE.compile(cctxQuant);
  bindingsBackend.allocate(modBackend->getPlaceholders());
  backendSpecificEE.run(bindingsBackend);

  // STEP3 - Compare the results of the original and quantized functions.
  auto result1Handle =
      bindings.get(bindings.getPlaceholderByNameSlow("save"))->getHandle();
  auto result2Handle =
      bindingsBackend.get(bindingsBackend.getPlaceholderByNameSlow("save"))
          ->getHandle();

  EXPECT_EQ(result1Handle.size(), result2Handle.size());

  for (int i = 0, e = result1Handle.size(); i < e; ++i) {
    float mx = result2Handle.raw(result2Handle.minMaxArg().second);
    double diff = std::fabs(result2Handle.raw(i) - result1Handle.raw(i)) / mx;

    // Allow 3% difference.
    EXPECT_NEAR(diff, 0, 0.03);
  }
}

/// End to end quantization test for Int8 quantization.
TEST_P(Operator, end2endInt8) {
  // The OpenCL backend does not support some of the nodes in the test;
  // explicitly whitelist them here as staying in float, so that the quantizer
  // does not complain.
  KindSet keepOriginalPrecisionForNodes;
  if (backendSpecificEE.getBackendName() == "OpenCL") {
    keepOriginalPrecisionForNodes.insert(Kinded::Kind::SelectNodeKind);
    keepOriginalPrecisionForNodes.insert(Kinded::Kind::CmpLTENodeKind);
    keepOriginalPrecisionForNodes.insert(
        Kinded::Kind::BatchedReduceAddNodeKind);
  }

  testQuantizationEnd2End(profileEE, backendSpecificEE, ElemKind::Int8QTy,
                          keepOriginalPrecisionForNodes);
}

/// Fills the tensor \p H with some stable random integers with the seed \p seed
/// and the range [0, scale).
static void fillStableRandomIndex(Handle<int64_t> H, size_t seed,
                                  size_t scale = 10) {
  for (size_t i = 0, e = H.size(); i < e; i++) {
    H.raw(i) = int(i * 1921 + seed) % scale;
  }
}

/// Builds a graph with two GRUs and saves output from last hidden node.
static Function *createGRUForQuantization(Module *M,
                                          PlaceholderBindings &bindings,
                                          llvm::StringRef funcName) {
  Function *F = M->createFunction(funcName);

  constexpr unsigned sequenceSize = 2;
  constexpr unsigned embeddingSize = 10;
  constexpr unsigned languageSize = 10;
  constexpr unsigned batchSize = 5;
  constexpr unsigned hiddenSize = 3 * embeddingSize;

  // STEP1 - Initialize inputs into GRU
  auto *emb = F->getParent()->createPlaceholder(
      ElemKind::FloatTy, {languageSize, embeddingSize}, "embedding", false);
  fillStableRandomData(bindings.allocate(emb)->getHandle(), 4565, 1);

  auto *input = F->getParent()->createPlaceholder(
      ElemKind::Int64ITy, {batchSize, sequenceSize}, "input", false);
  fillStableRandomIndex(bindings.allocate(input)->getHandle<int64_t>(), 7227,
                        10);

  auto *hiddenInit = F->getParent()->createPlaceholder(
      ElemKind::FloatTy, {batchSize, embeddingSize}, "hiddenInit", false);
  bindings.allocate(hiddenInit)->zero();
  Node *hidden = hiddenInit;

  for (unsigned step = 0; step < sequenceSize; step++) {
    // STEP2 - Gather a single set of embeddings for the GRU
    Node *inputEmbedded = F->createGather("gru.embedding", emb, input);
    Node *inputSlice =
        F->createSlice("gru.inputSlice", inputEmbedded, {0, step, 0},
                       {batchSize, step + 1, embeddingSize});
    Node *reshape =
        F->createReshape("gru.reshape", inputSlice, {batchSize, embeddingSize});

    // STEP3 - Generate a GRU
    // reference implementation:
    // https://github.com/pytorch/pytorch/blob/dd5c195646b941d3e20a72847ac48c41e272b8b2/torch/nn/_functions/rnn.py#L46
    // similar to /examples/fr2en.cpp

    auto *FCi =
        F->createFullyConnected(bindings, "gru.fci", reshape, hiddenSize);
    auto *biasI = cast<Placeholder>(FCi->getBias());
    auto *filterI = cast<Placeholder>(FCi->getWeights());
    fillStableRandomData(bindings.get(biasI)->getHandle(), 8877, 1);
    fillStableRandomData(bindings.get(filterI)->getHandle(), 1441, 1);

    auto *FCh =
        F->createFullyConnected(bindings, "gru.fch", hidden, hiddenSize);
    auto *biasH = cast<Placeholder>(FCh->getBias());
    auto *filterH = cast<Placeholder>(FCh->getWeights());
    fillStableRandomData(bindings.get(biasH)->getHandle(), 9009, 1);
    fillStableRandomData(bindings.get(filterH)->getHandle(), 1001, 1);

    Node *i_r =
        F->createSlice("gru.i_r", FCi, {0, 0}, {batchSize, embeddingSize});
    Node *i_i = F->createSlice("gru.i_i", FCi, {0, embeddingSize},
                               {batchSize, 2 * embeddingSize});
    Node *i_n = F->createSlice("gru.i_n", FCi, {0, 2 * embeddingSize},
                               {batchSize, 3 * embeddingSize});

    Node *h_r =
        F->createSlice("gru.h_r", FCh, {0, 0}, {batchSize, embeddingSize});
    Node *h_i = F->createSlice("gru.h_i", FCh, {0, embeddingSize},
                               {batchSize, 2 * embeddingSize});
    Node *h_n = F->createSlice("gru.h_n", FCh, {0, 2 * embeddingSize},
                               {batchSize, 3 * embeddingSize});

    Node *resetgate = F->createSigmoid("gru.resetgate",
                                       F->createAdd("i_r_plus_h_r", i_r, h_r));
    Node *inputgate = F->createSigmoid("gru.inputgate",
                                       F->createAdd("i_i_plus_h_i", i_i, h_i));
    Node *newgate = F->createTanh(
        "gru.newgate",
        F->createAdd("i_n_plus_rg_mult_h_n", i_n,
                     F->createMul("rg_mult_h_n", resetgate, h_n)));
    hidden = F->createAdd(
        "gru.newhidden", newgate,
        F->createMul("ig_mult_hmng", inputgate,
                     F->createSub("hidden_minus_newgate", hidden, newgate)));
  }
  // No-op TopK selection to test quantization
  Node *downsample = F->createTopK("gru.downsample", hidden, embeddingSize / 2);

  auto *save = F->createSave("save", {downsample, 0});
  bindings.allocate(save->getPlaceholder());
  return F;
}

TEST_P(Operator, end2endGRU) {
  // STEP1 - Generate the first network to record the quantization parameters.
  auto *mod = &profileEE.getModule();
  auto *modBackend = &backendSpecificEE.getModule();
  PlaceholderBindings bindings;
  PlaceholderBindings bindingsBackend;
  createGRUForQuantization(mod, bindings, "main");
  createGRUForQuantization(modBackend, bindingsBackend, "main");

  LoweredInfoMap loweredMapForProf;
  CompilationContext cctxProf{&bindings, &loweredMapForProf};
  cctxProf.precisionConfig.quantMode = QuantizationMode::Profile;
  profileEE.compile(cctxProf);

  // Run graph to capture profile.
  profileEE.run(bindings);

  LoweredInfoMap loweredMapForQuant;
  CompilationContext cctxQuant{&bindings, &loweredMapForQuant};
  cctxQuant.precisionConfig.quantMode = QuantizationMode::Quantize;
  PrecisionConfiguration &precConfig = cctxQuant.precisionConfig;
  precConfig.quantConfig.infos = quantization::generateNodeProfilingInfos(
      bindings, mod->getFunctions().front(), loweredMapForProf);

  // The OpenCL backend does not support some of the nodes in the test;
  // explicitly whitelist them here as staying in float, so that the quantizer
  // does not complain.
  KindSet doNotQuantizeKinds;
  if (backendSpecificEE.getBackendName() == "OpenCL") {
    precConfig.precisionModeKindSet.insert(Kinded::Kind::TanhNodeKind);
    precConfig.precisionModeKindSet.insert(Kinded::Kind::SigmoidNodeKind);
    precConfig.precisionModeKindSet.insert(Kinded::Kind::GatherNodeKind);
  }

  // STEP2 - Use the profile to quantize a network.

  backendSpecificEE.compile(cctxQuant);
  backendSpecificEE.run(bindingsBackend);

  // STEP3 - Compare the results of the original and quantized functions.
  auto result1Handle =
      bindings.get(bindings.getPlaceholderByNameSlow("save"))->getHandle();
  auto result2Handle =
      bindingsBackend.get(bindingsBackend.getPlaceholderByNameSlow("save"))
          ->getHandle();

  EXPECT_EQ(result1Handle.size(), result2Handle.size());

  for (int i = 0, e = result1Handle.size(); i < e; ++i) {
    float mx = result2Handle.raw(result2Handle.minMaxArg().second);
    double diff = std::fabs(result2Handle.raw(i) - result1Handle.raw(i)) / mx;

    // Allow 3% difference.
    EXPECT_NEAR(diff, 0, 0.03);
  }
}

TEST(Quantization, rescaleSameType) {
  ExecutionEngine EE{};
  PlaceholderBindings bindings;
  auto &mod = EE.getModule();
  auto *F = mod.createFunction("foo");
  auto *input =
      mod.createPlaceholder(ElemKind::Int8QTy, {1, 1}, 0.5, 11, "input", true);
  bindings.allocate(input)->init(Tensor::InitKind::Broadcast, 21,
                                 mod.getPRNG());

  auto *Q = F->createRescaleQuantized(
      "rescale", input, mod.uniqueType(ElemKind::Int8QTy, {1, 1}, 0.5, 11));
  auto *D = F->createDequantize("dequantize", Q, ElemKind::FloatTy);
  auto *save = F->createSave("ret", D);
  auto *result = bindings.allocate(save->getPlaceholder());

  EXPECT_EQ(F->getNodes().size(), 3);
  EE.compile(CompilationMode::Infer);

  EE.run(bindings);
  F = EE.getModule().getFunctions().front();
  EXPECT_EQ(F->getNodes().size(), 2);

  auto RH = result->getHandle();
  EXPECT_NEAR(RH.at({0, 0}), 5.0, 0.001);
}

TEST(Quantization, optimizeRescaleQuantize) {
  ExecutionEngine EE{};
  PlaceholderBindings bindings;
  auto &mod = EE.getModule();
  auto *F = mod.createFunction("foo");
  auto *input = mod.createPlaceholder(ElemKind::FloatTy, {1, 1}, "input", true);
  bindings.allocate(input)->init(Tensor::InitKind::Broadcast, 21,
                                 mod.getPRNG());

  auto *Q = F->createQuantize(
      "quant", input, mod.uniqueType(ElemKind::Int8QTy, {1, 1}, 0.25, 4));
  auto *RS = F->createRescaleQuantized(
      "rescale", Q, mod.uniqueType(ElemKind::Int8QTy, {1, 1}, 0.5, 11));
  auto *D = F->createDequantize("dequantize", RS, ElemKind::FloatTy);
  auto *save = F->createSave("ret", D);
  auto *result = bindings.allocate(save->getPlaceholder());

  EXPECT_EQ(F->getNodes().size(), 4);
  EE.compile(CompilationMode::Infer);

  EE.run(bindings);

  EXPECT_EQ(EE.getModule().getFunctions().front()->getNodes().size(), 1);

  auto RH = result->getHandle();
  EXPECT_NEAR(RH.at({0, 0}), 21.0, 0.001);
}

/// Check that our asymmetric quantization schema produces
/// the expected scales and offsets for various ranges for Int8.
TEST(Quantization, chooseQuantizationAsymmetricInt8) {
  // Map float [0.0; 6.0] to int [-128; 127].
  TensorQuantizationParams asymmetricParams = chooseQuantizationParams(
      {0.0, 6.0}, quantization::Schema::Asymmetric, ElemKind::Int8QTy);
  // Dequantization formula is scale(X - offset).
  // So
  // 1. scale(-128 - offset) == 0.0
  // 2. scale(127 - offset) == 6.0
  // Given scale != 0, #1 gives -128 == offset
  // Then #2, gives scale == 6.0 / (127 - (-128)).
  EXPECT_EQ(asymmetricParams.offset, -128);
  EXPECT_NEAR(asymmetricParams.scale, 6.0 / 255, 0.001);

  // Map float [-3.0; 3.0] to int [-128; 127].
  asymmetricParams = chooseQuantizationParams(
      {-3.0, 3.0}, quantization::Schema::Asymmetric, ElemKind::Int8QTy);
  // Dequantization formula is scale(X - offset).
  // So in theory, we should get
  // 1. scale(-128 - offset) == -3.0
  // 2. scale(127 - offset) == 3.0
  // Given scale != 0, #1 + #2 gives scale(-128 + 127 - 2*offset) == 0.0
  // offset == -1 / -2 == 0.5
  // Then #2 or #1, gives scale == 3.0 / 127.5.
  // However, when we get symmetric ranges (i.e., [-X; X]),
  // we actually force the zero point to map to 0.
  // In other words, scale(0 - offset) == 0.0, so our offset is 0.
  // Then our scale is simply: (inputMax - inputMin) / (outputMax - outputMin).
  // (3.0 - (-3.0)) / (127 - (-128)) == 6.0 / 255.
  EXPECT_EQ(asymmetricParams.offset, 0);
  EXPECT_NEAR(asymmetricParams.scale, 6.0 / 255, 0.001);

  // Map float [-2.0; 5.0] to int [-128; 127].
  asymmetricParams = chooseQuantizationParams(
      {-2.0, 5.0}, quantization::Schema::Asymmetric, ElemKind::Int8QTy);
  // Scale: (5.0 - (-2.0)) / (127 - (-128)) == 7.0 / 255.0
  // Offset from min: scale(-128 - offset) == -2.0
  //                  7.0 / 255.0 * (-128 - offset) == -2.0
  //                  -128 - offset == -2.0 * 255.0 / 7.0
  //                  offset == 2.0 * 255.0 / 7.0 - 128
  //                  offset == ~-55
  EXPECT_EQ(asymmetricParams.offset, (int32_t)(2.0 * 255 / 7.0 - 128));
  EXPECT_NEAR(asymmetricParams.scale, 7.0 / 255, 0.001);

  // Map float [2.0; 5.0] to int [-128; 127].
  // Make sure we extend the range to include 0.0, i.e.,
  // we really map [0.0; 5.0] to int [-128; 127].
  asymmetricParams = chooseQuantizationParams(
      {2.0, 5.0}, quantization::Schema::Asymmetric, ElemKind::Int8QTy);
  // Scale: (5.0 - (0.0)) / (127 - (-128)) == 5.0 / 255.0
  // Offset from min: scale(-128 - offset) == 0.0
  EXPECT_EQ(asymmetricParams.offset, -128);
  EXPECT_NEAR(asymmetricParams.scale, 5.0 / 255, 0.001);

  // Map float [-8.0; -2.0] to int [-128; 127].
  // Make sure we extend the range to include 0.0, i.e.,
  // we really map [-8.0; 0.0] to int [-128; 127].
  asymmetricParams = chooseQuantizationParams(
      {-8.0, -2.0}, quantization::Schema::Asymmetric, ElemKind::Int8QTy);
  // Scale: (0.0 - (-8.0)) / (127 - (-128)) == 8.0 / 255.0
  // Offset from min: scale(127 - offset) == 0.0
  EXPECT_EQ(asymmetricParams.offset, 127);
  EXPECT_NEAR(asymmetricParams.scale, 8.0 / 255, 0.001);
}

/// Check that our symmetric quantization schema produces
/// the expected scales and offsets for various ranges for Int8.
TEST(Quantization, chooseQuantizationSymmetricInt8) {
  // Map float [0.0; 6.0] to int [-128; 127].
  // With symmetric mapping, we basically map [-6.0; 6.0]
  TensorQuantizationParams symmetricParams = chooseQuantizationParams(
      {0.0, 6.0}, quantization::Schema::Symmetric, ElemKind::Int8QTy);
  // With symmetric mapping offset should always be zero.
  EXPECT_EQ(symmetricParams.offset, 0);
  EXPECT_NEAR(symmetricParams.scale, 12.0 / 255, 0.001);

  // Map float [-3.0; 3.0] to int [-128; 127].
  symmetricParams = chooseQuantizationParams(
      {-3.0, 3.0}, quantization::Schema::Symmetric, ElemKind::Int8QTy);
  EXPECT_EQ(symmetricParams.offset, 0);
  EXPECT_NEAR(symmetricParams.scale, 6.0 / 255, 0.001);

  // Map float [-2.0; 5.0] to int [-128; 127].
  // => [-5.0; 5.0] range for symmetric mode.
  symmetricParams = chooseQuantizationParams(
      {-2.0, 5.0}, quantization::Schema::Symmetric, ElemKind::Int8QTy);
  EXPECT_EQ(symmetricParams.offset, 0);
  EXPECT_NEAR(symmetricParams.scale, 10.0 / 255, 0.001);

  // Map float [2.0; 5.0] to int [-128; 127].
  // Ranges are extended to include 0.
  // => [0.0; 5.0] range for symmetric mode.
  symmetricParams = chooseQuantizationParams(
      {2.0, 5.0}, quantization::Schema::Symmetric, ElemKind::Int8QTy);
  // Scale: (5.0 - (0.0)) / (127 - (-128)) == 5.0 / 255.0
  // Offset from min: scale(-128 - offset) == 0.0
  EXPECT_EQ(symmetricParams.offset, 0);
  EXPECT_NEAR(symmetricParams.scale, 10.0 / 255, 0.001);

  // Map float [-8.0; -2.0] to int [-128; 127].
  // => [-8.0; 8.0] range for symmetric mode.
  symmetricParams = chooseQuantizationParams(
      {-8.0, -2.0}, quantization::Schema::Symmetric, ElemKind::Int8QTy);
  EXPECT_EQ(symmetricParams.offset, 0);
  EXPECT_NEAR(symmetricParams.scale, 16.0 / 255, 0.001);
}

/// Check that our asymmetric quantization schema produces
/// the expected scales and offsets for various ranges for Int16.
TEST(Quantization, chooseQuantizationAsymmetricInt16) {
  // Map float [0.0; 6.0] to int [-32768; 32767].
  TensorQuantizationParams asymmetricParams = chooseQuantizationParams(
      {0.0, 6.0}, quantization::Schema::Asymmetric, ElemKind::Int16QTy);
  // Dequantization formula is scale(X - offset).
  // So
  // 1. scale(-32768 - offset) == 0.0
  // 2. scale(32767 - offset) == 6.0
  // Given scale != 0, #1 gives -32768 == offset
  // Then #2, gives scale == 6.0 / (32767 - (-32768)).
  EXPECT_EQ(asymmetricParams.offset, -32768);
  EXPECT_NEAR(asymmetricParams.scale, 6.0 / 65535, 0.00009);

  // Map float [-3.0; 3.0] to int [-32768; 32767].
  asymmetricParams = chooseQuantizationParams(
      {-3.0, 3.0}, quantization::Schema::Asymmetric, ElemKind::Int16QTy);
  // Dequantization formula is scale(X - offset).
  // So in theory, we should get
  // 1. scale(-32768 - offset) == -3.0
  // 2. scale(32767 - offset) == 3.0
  // Given scale != 0, #1 + #2 gives scale(-32768 + 32767 - 2*offset) == 0.0
  // offset == -1 / -2 == 0.5
  // Then #2 or #1, gives scale == 3.0 / 32767.5.
  // However, when we get symmetric ranges (i.e., [-X; X]),
  // we actually force the zero point to map to 0.
  // In other words, scale(0 - offset) == 0.0, so our offset is 0.
  // Then our scale is simply: (inputMax - inputMin) / (outputMax - outputMin).
  // (3.0 - (-3.0)) / (32767 - (-32768)) == 6.0 / 255.
  EXPECT_EQ(asymmetricParams.offset, 0);
  EXPECT_NEAR(asymmetricParams.scale, 6.0 / 65535, 0.00009);

  // Map float [-2.0; 5.0] to int [-32768; 32767].
  asymmetricParams = chooseQuantizationParams(
      {-2.0, 5.0}, quantization::Schema::Asymmetric, ElemKind::Int16QTy);
  // Scale: (5.0 - (-2.0)) / (32767 - (-32768)) == 7.0 / 255.0
  // Offset from min: scale(-32768 - offset) == -2.0
  //                  7.0 / 255.0 * (-32768 - offset) == -2.0
  //                  -32768 - offset == -2.0 * 255.0 / 7.0
  //                  offset == 2.0 * 255.0 / 7.0 - 32768
  //                  offset == ~-55
  EXPECT_EQ(asymmetricParams.offset, std::round(2.0 * 65535 / 7.0 - 32768));
  EXPECT_NEAR(asymmetricParams.scale, 7.0 / 65535, 0.00009);

  // Map float [2.0; 5.0] to int [-32768; 32767].
  // Make sure we extend the range to include 0.0, i.e.,
  // we really map [0.0; 5.0] to int [-32768; 32767].
  asymmetricParams = chooseQuantizationParams(
      {2.0, 5.0}, quantization::Schema::Asymmetric, ElemKind::Int16QTy);
  // Scale: (5.0 - (0.0)) / (32767 - (-32768)) == 5.0 / 255.0
  // Offset from min: scale(-32768 - offset) == 0.0
  EXPECT_EQ(asymmetricParams.offset, -32768);
  EXPECT_NEAR(asymmetricParams.scale, 5.0 / 65535, 0.00009);

  // Map float [-8.0; -2.0] to int [-32768; 32767].
  // Make sure we extend the range to include 0.0, i.e.,
  // we really map [-8.0; 0.0] to int [-32768; 32767].
  asymmetricParams = chooseQuantizationParams(
      {-8.0, -2.0}, quantization::Schema::Asymmetric, ElemKind::Int16QTy);
  // Scale: (0.0 - (-8.0)) / (32767 - (-32768)) == 8.0 / 255.0
  // Offset from min: scale(32767 - offset) == 0.0
  EXPECT_EQ(asymmetricParams.offset, 32767);
  EXPECT_NEAR(asymmetricParams.scale, 8.0 / 65535, 0.00009);
}

/// Check that our symmetric quantization schema produces
/// the expected scales and offsets for various ranges for Int16.
TEST(Quantization, chooseQuantizationSymmetricInt16) {
  // Map float [0.0; 6.0] to int [-32768; 32767].
  // With symmetric mapping, we basically map [-6.0; 6.0]
  TensorQuantizationParams symmetricParams = chooseQuantizationParams(
      {0.0, 6.0}, quantization::Schema::Symmetric, ElemKind::Int16QTy);
  // With symmetric mapping offset should always be zero.
  EXPECT_EQ(symmetricParams.offset, 0);
  EXPECT_NEAR(symmetricParams.scale, 12.0 / 65535, 0.00009);

  // Map float [-3.0; 3.0] to int [-32768; 32767].
  symmetricParams = chooseQuantizationParams(
      {-3.0, 3.0}, quantization::Schema::Symmetric, ElemKind::Int16QTy);
  EXPECT_EQ(symmetricParams.offset, 0);
  EXPECT_NEAR(symmetricParams.scale, 6.0 / 65535, 0.00009);

  // Map float [-2.0; 5.0] to int [-32768; 32767].
  // => [-5.0; 5.0] range for symmetric mode.
  symmetricParams = chooseQuantizationParams(
      {-2.0, 5.0}, quantization::Schema::Symmetric, ElemKind::Int16QTy);
  EXPECT_EQ(symmetricParams.offset, 0);
  EXPECT_NEAR(symmetricParams.scale, 10.0 / 65535, 0.00009);

  // Map float [2.0; 5.0] to int [-32768; 32767].
  // Ranges are extended to include 0.
  // => [0.0; 5.0] range for symmetric mode.
  symmetricParams = chooseQuantizationParams(
      {2.0, 5.0}, quantization::Schema::Symmetric, ElemKind::Int16QTy);
  // Scale: (5.0 - (0.0)) / (32767 - (-32768)) == 5.0 / 65535.0
  // Offset from min: scale(-32768 - offset) == 0.0
  EXPECT_EQ(symmetricParams.offset, 0);
  EXPECT_NEAR(symmetricParams.scale, 10.0 / 65535, 0.00009);

  // Map float [-8.0; -2.0] to int [-32768; 32767].
  // => [-8.0; 8.0] range for symmetric mode.
  symmetricParams = chooseQuantizationParams(
      {-8.0, -2.0}, quantization::Schema::Symmetric, ElemKind::Int16QTy);
  EXPECT_EQ(symmetricParams.offset, 0);
  EXPECT_NEAR(symmetricParams.scale, 16.0 / 65535, 0.00009);
}

/// Check quantization symmetry in presence of infinities.
TEST(Quantization, chooseQuantizationSymmetricInf) {
  auto sym = quantization::Schema::Symmetric;
  // Check for Int8 precision.
  EXPECT_EQ(
      chooseQuantizationParams({-INFINITY, INFINITY}, sym, ElemKind::Int8QTy)
          .offset,
      0);
  EXPECT_EQ(
      chooseQuantizationParams({INFINITY, INFINITY}, sym, ElemKind::Int8QTy)
          .offset,
      0);
  EXPECT_EQ(
      chooseQuantizationParams({-INFINITY, -INFINITY}, sym, ElemKind::Int8QTy)
          .offset,
      0);
  EXPECT_EQ(chooseQuantizationParams({-INFINITY, 1.0f}, sym, ElemKind::Int8QTy)
                .offset,
            0);
  EXPECT_EQ(chooseQuantizationParams({-INFINITY, -1.0f}, sym, ElemKind::Int8QTy)
                .offset,
            0);
  EXPECT_EQ(chooseQuantizationParams({-1.0f, INFINITY}, sym, ElemKind::Int8QTy)
                .offset,
            0);
  EXPECT_EQ(
      chooseQuantizationParams({1.0f, INFINITY}, sym, ElemKind::Int8QTy).offset,
      0);
  // Check for Int16 precision.
  EXPECT_EQ(
      chooseQuantizationParams({-INFINITY, INFINITY}, sym, ElemKind::Int16QTy)
          .offset,
      0);
  EXPECT_EQ(
      chooseQuantizationParams({INFINITY, INFINITY}, sym, ElemKind::Int16QTy)
          .offset,
      0);
  EXPECT_EQ(
      chooseQuantizationParams({-INFINITY, -INFINITY}, sym, ElemKind::Int16QTy)
          .offset,
      0);
  EXPECT_EQ(chooseQuantizationParams({-INFINITY, 1.0f}, sym, ElemKind::Int16QTy)
                .offset,
            0);
  EXPECT_EQ(
      chooseQuantizationParams({-INFINITY, -1.0f}, sym, ElemKind::Int16QTy)
          .offset,
      0);
  EXPECT_EQ(chooseQuantizationParams({-1.0f, INFINITY}, sym, ElemKind::Int16QTy)
                .offset,
            0);
  EXPECT_EQ(chooseQuantizationParams({1.0f, INFINITY}, sym, ElemKind::Int16QTy)
                .offset,
            0);
}

/// Check that Relu can use our symmetric quantization schema.
TEST(Quantization, reluCanUseSymmetricSchema) {
  PlaceholderBindings bindings;
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  Placeholder *input =
      mod.createPlaceholder(ElemKind::FloatTy, {10}, "input", false);
  auto *inputTensor = bindings.allocate(input);
  auto IH = inputTensor->getHandle<float>();
  for (dim_t i = 0; i < 10; i++) {
    IH.at({i}) = (i % 2 == 0) ? 5 : -5;
  }

  // Create symmetric params that will be used for Relu.
  TensorQuantizationParams reluParams =
      chooseQuantizationParams({0.0, 10.0}, quantization::Schema::Symmetric);
  TypeRef reluTy = mod.uniqueType(ElemKind::Int8QTy, {10}, reluParams.scale,
                                  reluParams.offset);
  TensorQuantizationParams inputParams =
      chooseQuantizationParams({-10.0, 10.0}, quantization::Schema::Symmetric);

  QuantizeNode *QN =
      F->createQuantize("quant", input,
                        mod.uniqueType(ElemKind::Int8QTy, {10},
                                       inputParams.scale, inputParams.offset));
  ReluNode *RN = F->createRELU("relu", QN, reluTy);
  DequantizeNode *DN = F->createDequantize("dequantize", RN, ElemKind::FloatTy);
  SaveNode *SN = F->createSave("save", DN);
  auto *res = bindings.allocate(SN->getPlaceholder());

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  // Verify all negative values were correctly set to zero.
  auto RH = res->getHandle();
  for (dim_t i = 0; i < 10; i++) {
    if (i % 2 == 0) {
      EXPECT_NEAR(RH.at({i}), 5, 0.05);
    } else {
      EXPECT_EQ(RH.at({i}), 0);
    }
  }
}

/// Check that our symmetric with uint8 quantization schema produces
/// the expected scales and offsets for various ranges.
TEST(Quantization, chooseQuantizationSymmetricWithUInt8) {
  // Map float [0.0; 6.0] to int [-128; 127].
  // With symmetric with uint8 mapping, we basically map [0.0; 6.0]
  TensorQuantizationParams symmetricParams = chooseQuantizationParams(
      {0.0, 6.0}, quantization::Schema::SymmetricWithUnsigned);
  // Given this is a purely positive range, we should use uint8,
  // thus int8 - (-128).
  EXPECT_EQ(symmetricParams.offset, -128);
  EXPECT_NEAR(symmetricParams.scale, 6.0 / 255, 0.001);

  // Map float [-3.0; 3.0] to int [-128; 127].
  symmetricParams = chooseQuantizationParams(
      {-3.0, 3.0}, quantization::Schema::SymmetricWithUnsigned);
  EXPECT_EQ(symmetricParams.offset, 0);
  EXPECT_NEAR(symmetricParams.scale, 6.0 / 255, 0.001);

  // Map float [-2.0; 5.0] to int [-128; 127].
  // This has negative value, thus we fall back to purely symmetric.
  // => [-5.0; 5.0] range for symmetric mode.
  symmetricParams = chooseQuantizationParams(
      {-2.0, 5.0}, quantization::Schema::SymmetricWithUnsigned);
  EXPECT_EQ(symmetricParams.offset, 0);
  EXPECT_NEAR(symmetricParams.scale, 10.0 / 255, 0.001);

  // Map float [0; 0] to int [-128; 127].
  symmetricParams = chooseQuantizationParams(
      {0.0, 0.0}, quantization::Schema::SymmetricWithUnsigned);
  EXPECT_EQ(symmetricParams.offset, 0);
  EXPECT_NEAR(symmetricParams.scale, 0.1, 0.001);

  // Map float [2.0; 5.0] to int [-128; 127].
  // All positive, using uint8.
  // However, our quantization schemas always include zero.
  // => [0.0; 5.0] range for uint8 mode.
  symmetricParams = chooseQuantizationParams(
      {2.0, 5.0}, quantization::Schema::SymmetricWithUnsigned);
  // Scale: (5.0 - (0.0)) / (127 - (-128)) == 5.0 / 255.0
  // Offset from min: scale(-128 - offset) == 0.0
  EXPECT_EQ(symmetricParams.offset, -128);
  EXPECT_NEAR(symmetricParams.scale, 5.0 / 255, 0.001);

  // Map float [-8.0; -2.0] to int [-128; 127].
  // => [-8.0; 8.0] range for symmetric mode.
  symmetricParams = chooseQuantizationParams(
      {-8.0, -2.0}, quantization::Schema::SymmetricWithUnsigned);
  EXPECT_EQ(symmetricParams.offset, 0);
  EXPECT_NEAR(symmetricParams.scale, 16.0 / 255, 0.001);
}

/// Verify the SymmetricWithPower2Scale quantization schema.
static void chooseQuantParamsPower2Scale(float min, float max, ElemKind qTy) {
  auto quantParams = quantization::chooseQuantizationParams(
      {min, max}, quantization::Schema::SymmetricWithPower2Scale, qTy);
  EXPECT_EQ(quantParams.offset, 0);
  EXPECT_TRUE(quantization::isFloatPowerOf2(quantParams.scale));
}

TEST(Quantization, chooseQuantizationSymmetricWithPower2Scale) {
  chooseQuantParamsPower2Scale(-3.0, 6.0, ElemKind::Int8QTy);
  chooseQuantParamsPower2Scale(3.0, 6.0, ElemKind::Int16QTy);
  chooseQuantParamsPower2Scale(-6.0, 0.0, ElemKind::Int32QTy);
}

/// Check that LRN and Softmax are quantized.
TEST(Quantization, quantizeSoftmaxAndLRN) {
  ExecutionEngine EE{};
  PlaceholderBindings bindings;
  std::unique_ptr<Backend> backend(new MockQuantBackend);
  EE.setBackendName("Interpreter");

  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  auto *input =
      mod.createPlaceholder(ElemKind::FloatTy, {1, 10}, "input", true);
  auto *selected =
      mod.createPlaceholder(ElemKind::Int64ITy, {1, 10}, "selected", true);
  auto *LRN =
      F->createLocalResponseNormalization("LRN", input, 2, 1.0, 0.0001, 0.75);
  auto *SM = F->createSoftMax("softmax", LRN, selected);
  auto *SN = F->createSave("ret", SM);

  quantization::QuantizationConfiguration quantConfig{
      {{input->getOutput().generateNodeOutputName(), {0.2f, 2.0f}},
       {LRN->getResult().generateNodeOutputName(LRN->getName().str()),
        {0.3f, 3.0f}},
       {SM->getResult().generateNodeOutputName(SM->getName().str()),
        {0.4f, 4.0f}},
       {NodeValue::generateNodeOutputName(SN->getName().str()), {0.4f, 4.0f}}}};

  quantConfig.assertAllNodesQuantized = true;
  quantization::quantizeFunction(F, quantConfig, *backend);

  auto qLRNIt = std::find_if(
      F->getNodes().begin(), F->getNodes().end(), [](const Node &node) -> bool {
        return llvm::isa<LocalResponseNormalizationNode>(&node) &&
               node.getNthResult(LocalResponseNormalizationNode::ResultIdx)
                   .getType()
                   ->isQuantizedType();
      });
  ASSERT_NE(qLRNIt, F->getNodes().end());
  auto qSMIt = std::find_if(F->getNodes().begin(), F->getNodes().end(),
                            [](const Node &node) -> bool {
                              return llvm::isa<SoftMaxNode>(&node) &&
                                     node.getNthResult(SoftMaxNode::ResultIdx)
                                         .getType()
                                         ->isQuantizedType();
                            });
  ASSERT_NE(qSMIt, F->getNodes().end());

  // Make sure that SaveNode is not quantized.
  for (const auto &node : F->getNodes()) {
    if (auto *saveNode = llvm::dyn_cast<SaveNode>(&node)) {
      EXPECT_FALSE(saveNode->getInput().getType()->isQuantizedType());
    }
  }
}

/// Check that Select is quantized.
TEST(Quantization, quantizeSelect) {
  ExecutionEngine EE{};
  PlaceholderBindings bindings;
  std::unique_ptr<Backend> backend(new MockQuantBackend);
  EE.setBackendName("Interpreter");

  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  auto *LHS = mod.createPlaceholder(ElemKind::FloatTy, {1, 10}, "LHS", false);
  auto *RHS = mod.createPlaceholder(ElemKind::FloatTy, {1, 10}, "RHS", false);
  auto *cond = mod.createPlaceholder(ElemKind::BoolTy, {1, 10}, "cond", false);
  auto *select = F->createSelect("select", cond, LHS, RHS);
  F->createSave("save", select);

  TensorProfilingParams LHSPP = {0.0, 1.0};
  TensorProfilingParams RHSPP = {-1.3, 2.7};
  TensorProfilingParams selectPP = {-2, 3.1};

  quantization::QuantizationConfiguration quantConfig{
      {{LHS->getOutput().generateNodeOutputName(), LHSPP},
       {RHS->getOutput().generateNodeOutputName(), RHSPP},
       {select->getResult().generateNodeOutputName(), selectPP}}};

  quantConfig.assertAllNodesQuantized = true;
  quantization::quantizeFunction(F, quantConfig, *backend);

  // Get quantization parameters for verification.
  TensorQuantizationParams LHSQP = chooseQuantizationParams(
      LHSPP, quantConfig.schema, quantConfig.precision);
  TensorQuantizationParams RHSQP = chooseQuantizationParams(
      RHSPP, quantConfig.schema, quantConfig.precision);
  TensorQuantizationParams selectQP = chooseQuantizationParams(
      selectPP, quantConfig.schema, quantConfig.precision);

  auto it = std::find_if(
      F->getNodes().begin(), F->getNodes().end(),
      [](const Node &node) -> bool { return llvm::isa<SelectNode>(&node); });
  ASSERT_NE(it, F->getNodes().end());

  SelectNode *qSelect = llvm::cast<SelectNode>(&(*it));
  TypeRef qSelectTy = qSelect->getResult().getType();
  TypeRef qLHSTy = qSelect->getLHS().getType();
  TypeRef qRHSTy = qSelect->getRHS().getType();

  ASSERT_TRUE(qSelectTy->isQuantizedType());
  EXPECT_EQ(qSelectTy->getScale(), selectQP.scale);
  EXPECT_EQ(qSelectTy->getOffset(), selectQP.offset);
  EXPECT_EQ(qLHSTy->getScale(), LHSQP.scale);
  EXPECT_EQ(qLHSTy->getOffset(), LHSQP.offset);
  EXPECT_EQ(qRHSTy->getScale(), RHSQP.scale);
  EXPECT_EQ(qRHSTy->getOffset(), RHSQP.offset);
}

/// Check that AvgPool is quantized, and its input and output have different
/// scale and offset.
TEST(Quantization, quantizeAvgPool) {
  ExecutionEngine EE{};
  PlaceholderBindings bindings;
  std::unique_ptr<Backend> backend(new MockQuantBackend);
  EE.setBackendName("Interpreter");

  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  auto *input =
      mod.createPlaceholder(ElemKind::FloatTy, {1, 3, 3, 1}, "input", true);
  auto *pool = F->createAvgPool("pool", input, {2, 2}, {1, 1}, {0, 0, 0, 0});
  auto *s = F->createSave("save", pool);

  quantization::QuantizationConfiguration quantConfig{{
      {input->getOutput().generateNodeOutputName(), {-2.0f, 2.0f}},
      {pool->getResult().generateNodeOutputName(), {0.3f, 3.0f}},
      {NodeValue::generateNodeOutputName(s->getName().str()), {0.4f, 4.0f}},
  }};

  quantConfig.assertAllNodesQuantized = true;
  quantization::quantizeFunction(F, quantConfig, *backend);

  auto qPool = std::find_if(F->getNodes().begin(), F->getNodes().end(),
                            [](const Node &node) -> bool {
                              return llvm::isa<AvgPoolNode>(&node) &&
                                     node.getNthResult(AvgPoolNode::ResultIdx)
                                         .getType()
                                         ->isQuantizedType();
                            });
  ASSERT_NE(qPool, F->getNodes().end());
  auto *avgPool = llvm::cast<AvgPoolNode>(qPool);
  ASSERT_NE(avgPool->getInput().getType()->getScale(),
            avgPool->getResult().getType()->getScale());
  ASSERT_NE(avgPool->getInput().getType()->getOffset(),
            avgPool->getResult().getType()->getOffset());
}

/// Test option to disable quantization of specific node kinds in the graph.
TEST(Quantization, quantizeGraphPartially) {
  ExecutionEngine EE{};
  PlaceholderBindings bindings;
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  auto *LHS = mod.createPlaceholder(ElemKind::FloatTy, {3, 3}, "lhs", true);
  auto *RHS = mod.createPlaceholder(ElemKind::FloatTy, {3, 3}, "rhs", true);
  bindings.allocate(LHS)->init(Tensor::InitKind::Xavier, 3, mod.getPRNG());
  bindings.allocate(RHS)->init(Tensor::InitKind::Xavier, 3, mod.getPRNG());

  auto *MMN = F->createMatMul("matmul", LHS, RHS);
  auto *TN = F->createTanh("tanh", MMN);
  auto *save = F->createSave("ret", TN);
  auto *result = save->getPlaceholder();
  bindings.allocate(result);

  // Note that we are creating profiling info even for nodes that will not be
  // quantized. This is how we expect quantizeFunction() to behave, as
  // quantization profiling will still get a profile for these nodes.
  quantization::QuantizationConfiguration quantConfig{{
      {LHS->getOutput().generateNodeOutputName(), {0.3f, 3.0f}},
      {RHS->getOutput().generateNodeOutputName(), {0.4f, 4.0f}},
      {MMN->getResult().generateNodeOutputName(), {0.6f, 6.0f}},
      {TN->getResult().generateNodeOutputName(), {0.5f, 5.0f}},
  }};

  // Do not quantize any tanh nodes.
  KindSet doNotQuantizeKinds;
  doNotQuantizeKinds.insert(Kinded::Kind::TanhNodeKind);

  quantConfig.assertAllNodesQuantized = true;
  std::unique_ptr<Backend> backend(createBackend(EE.getBackendName()));
  quantization::quantizeFunction(F, quantConfig, *backend,
                                 /* loweredMap */ {}, doNotQuantizeKinds);

  // Make sure that graph can be compiled and run.
  ::glow::convertPlaceholdersToConstants(F, bindings, {result});

  CompilationContext cctx;
  cctx.compMode = CompilationMode::Infer;
  // Do not perform any compile-time constant folding.
  cctx.optimizationOpts.enableConstantFolding = false;
  EE.compile(cctx);

  EE.run(bindings);

  {
    // Verify that the output variable is not quantized, and that it has a
    // single save node writer, which is also not quantized.
    EXPECT_TRUE(!result->getType()->isQuantizedType());
    ASSERT_EQ(result->getUsers().size(), 1);
    auto *SN = llvm::dyn_cast<SaveNode>(result->getUsers().begin()->getUser());
    ASSERT_TRUE(SN);
    EXPECT_TRUE(!SN->getOutput().getType()->isQuantizedType());

    // Verify that the tanh is not quantized.
    auto *TN = llvm::dyn_cast<TanhNode>(SN->getInput());
    ASSERT_TRUE(TN);
    EXPECT_TRUE(!TN->getResult().getType()->isQuantizedType());

    // Verify that the input to the tanh is a dequantize node.
    auto *DN = llvm::dyn_cast<DequantizeNode>(TN->getInput());
    ASSERT_TRUE(DN);

    // Verify that the matmul is quantized.
    auto *MMN = llvm::dyn_cast<MatMulNode>(DN->getInput());
    ASSERT_TRUE(MMN);
    EXPECT_TRUE(MMN->getResult().getType()->isQuantizedType());

    // Verify that the variable inputs to the matmul are quantized.
    auto *LHS = llvm::dyn_cast<Constant>(MMN->getLHS());
    ASSERT_TRUE(LHS);
    EXPECT_TRUE(LHS->getType()->isQuantizedType());

    auto *RHS = llvm::dyn_cast<Constant>(MMN->getRHS());
    ASSERT_TRUE(RHS);
    EXPECT_TRUE(RHS->getType()->isQuantizedType());
  }
}

/// Test option to disable quantization of specific node kinds in the graph,
/// where there are multiple of that node kind.
TEST(Quantization, quantizeGraphPartiallyMultipleNodes) {
  ExecutionEngine EE{};
  PlaceholderBindings bindings;
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  auto *LHS = mod.createPlaceholder(ElemKind::FloatTy, {3, 3}, "lhs", true);
  auto *RHS = mod.createPlaceholder(ElemKind::FloatTy, {3, 3}, "rhs", true);
  bindings.allocate(LHS)->init(Tensor::InitKind::Xavier, 3, mod.getPRNG());
  bindings.allocate(RHS)->init(Tensor::InitKind::Xavier, 3, mod.getPRNG());

  auto *TNLHS = F->createTanh("tanh", LHS);
  auto *MMN = F->createMatMul("matmul", TNLHS, RHS);
  auto *TN = F->createTanh("tanh", MMN);
  auto *save = F->createSave("ret", TN);
  auto *result = save->getPlaceholder();
  bindings.allocate(result);

  // Note that we are creating profiling info even for nodes that will not be
  // quantized. This is how we expect quantizeFunction() to behave, as
  // quantization profiling will still get a profile for these nodes.
  quantization::QuantizationConfiguration quantConfig{{
      {LHS->getOutput().generateNodeOutputName(), {0.3f, 3.0f}},
      {TNLHS->getResult().generateNodeOutputName(), {0.4f, 4.0f}},
      {RHS->getOutput().generateNodeOutputName(), {0.4f, 4.0f}},
      {MMN->getResult().generateNodeOutputName(), {0.6f, 6.0f}},
      {TN->getResult().generateNodeOutputName(), {0.5f, 5.0f}},
  }};

  // Do not quantize any tanh nodes.
  KindSet doNotQuantizeKinds;
  doNotQuantizeKinds.insert(Kinded::Kind::TanhNodeKind);

  quantConfig.assertAllNodesQuantized = true;
  std::unique_ptr<Backend> backend(createBackend(EE.getBackendName()));
  quantization::quantizeFunction(F, quantConfig, *backend,
                                 /* loweredMap */ {}, doNotQuantizeKinds);

  // Make sure that graph can be compiled and run.
  ::glow::convertPlaceholdersToConstants(F, bindings, {result});

  CompilationContext cctx;
  cctx.compMode = CompilationMode::Infer;
  // Do not perform any compile-time constant folding.
  cctx.optimizationOpts.enableConstantFolding = false;
  EE.compile(cctx);

  EE.run(bindings);

  {
    // Verify that the output variable is not quantized, and that it has a
    // single save node writer, which is also not quantized.
    EXPECT_TRUE(!result->getType()->isQuantizedType());
    ASSERT_EQ(result->getUsers().size(), 1);
    auto *SN = llvm::dyn_cast<SaveNode>(result->getUsers().begin()->getUser());
    ASSERT_TRUE(SN);
    EXPECT_TRUE(!SN->getOutput().getType()->isQuantizedType());

    // Verify that the tanh is not quantized.
    auto *TN1 = llvm::dyn_cast<TanhNode>(SN->getInput());
    ASSERT_TRUE(TN1);
    EXPECT_TRUE(!TN1->getResult().getType()->isQuantizedType());

    // Verify that the input to the tanh is a dequantize node.
    auto *DN = llvm::dyn_cast<DequantizeNode>(TN1->getInput());
    ASSERT_TRUE(DN);

    // Verify that the matmul is quantized.
    auto *MMN = llvm::dyn_cast<MatMulNode>(DN->getInput());
    ASSERT_TRUE(MMN);
    EXPECT_TRUE(MMN->getResult().getType()->isQuantizedType());

    // Verify that the LHS input is a quantize node.
    auto *QN = llvm::dyn_cast<QuantizeNode>(MMN->getLHS());
    ASSERT_TRUE(QN);

    // Verify that the second tanh node is also not quantized.
    auto *TN2 = llvm::dyn_cast<TanhNode>(QN->getInput());
    ASSERT_TRUE(TN2);
    EXPECT_TRUE(!TN2->getResult().getType()->isQuantizedType());

    // Verify that the input variable to the tanh is not quantized.
    auto *varTN2 = llvm::dyn_cast<Constant>(TN2->getInput());
    ASSERT_TRUE(varTN2);
    EXPECT_TRUE(!varTN2->getType()->isQuantizedType());

    // Verify that the RHS input to the matmul is a quantized variable.
    auto *RHS = llvm::dyn_cast<Constant>(MMN->getRHS());
    ASSERT_TRUE(RHS);
    EXPECT_TRUE(RHS->getType()->isQuantizedType());
  }
}

/// Test option to disable quantization of multiple specific node kinds in the
/// graph.
TEST(Quantization, quantizeGraphPartiallyMultipleKinds) {
  ExecutionEngine EE{};
  PlaceholderBindings bindings;
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  auto *LHS = mod.createPlaceholder(ElemKind::FloatTy, {3, 3}, "lhs", true);
  auto *RHS = mod.createPlaceholder(ElemKind::FloatTy, {3, 3}, "rhs", true);
  bindings.allocate(LHS)->init(Tensor::InitKind::Xavier, 3, mod.getPRNG());
  bindings.allocate(RHS)->init(Tensor::InitKind::Xavier, 3, mod.getPRNG());

  auto *MMN = F->createMatMul("matmul", LHS, RHS);
  auto *CN = F->createAdd("concat", LHS, MMN);
  auto *TN = F->createTanh("tanh", CN);
  auto *save = F->createSave("ret", TN);
  auto *result = save->getPlaceholder();
  bindings.allocate(result);

  // Note that we are creating profiling info even for nodes that will not be
  // quantized. This is how we expect quantizeFunction() to behave, as
  // quantization profiling will still get a profile for these nodes.
  quantization::QuantizationConfiguration quantConfig{{
      {LHS->getOutput().generateNodeOutputName(), {0.3f, 3.0f}},
      {RHS->getOutput().generateNodeOutputName(), {0.4f, 4.0f}},
      {MMN->getResult().generateNodeOutputName(), {0.6f, 6.0f}},
      {CN->getResult().generateNodeOutputName(), {0.6f, 6.0f}},
      {TN->getResult().generateNodeOutputName(), {0.5f, 5.0f}},
  }};

  // Do not quantize any tanh or add nodes.
  KindSet doNotQuantizeKinds;
  doNotQuantizeKinds.insert(Kinded::Kind::TanhNodeKind);
  doNotQuantizeKinds.insert(Kinded::Kind::AddNodeKind);

  quantConfig.assertAllNodesQuantized = true;
  std::unique_ptr<Backend> backend(createBackend(EE.getBackendName()));
  quantization::quantizeFunction(F, quantConfig, *backend,
                                 /* loweredMap */ {}, doNotQuantizeKinds);

  // Make sure that graph can be compiled and run.
  ::glow::convertPlaceholdersToConstants(F, bindings, {result});

  CompilationContext cctx;
  cctx.compMode = CompilationMode::Infer;
  // Do not perform any compile-time constant folding.
  cctx.optimizationOpts.enableConstantFolding = false;
  EE.compile(cctx);

  EE.run(bindings);

  {
    // Verify that the output variable is not quantized, and that it has a
    // single save node writer, which is also not quantized.
    EXPECT_TRUE(!result->getType()->isQuantizedType());
    ASSERT_EQ(result->getUsers().size(), 1);
    auto *SN = llvm::dyn_cast<SaveNode>(result->getUsers().begin()->getUser());
    ASSERT_TRUE(SN);
    EXPECT_TRUE(!SN->getOutput().getType()->isQuantizedType());

    // Verify that the tanh is not quantized.
    auto *TN = llvm::dyn_cast<TanhNode>(SN->getInput());
    ASSERT_TRUE(TN);
    EXPECT_TRUE(!TN->getResult().getType()->isQuantizedType());

    // Verify that the input to the tanh is a non-quantized add node.
    auto *AN = llvm::dyn_cast<AddNode>(TN->getInput());
    ASSERT_TRUE(AN);
    EXPECT_TRUE(!TN->getResult().getType()->isQuantizedType());

    // Verify that the LHS input to the AddNode is an unquantized variable.
    auto varANLHS = llvm::dyn_cast<Constant>(AN->getLHS());
    ASSERT_TRUE(varANLHS);
    EXPECT_TRUE(!varANLHS->getType()->isQuantizedType());

    // Verify that the RHS input to the AddNode is a dequantize node.
    auto *DN = llvm::dyn_cast<DequantizeNode>(AN->getRHS());
    ASSERT_TRUE(DN);

    // Verify that the matmul is quantized.
    auto *MMN = llvm::dyn_cast<MatMulNode>(DN->getInput());
    ASSERT_TRUE(MMN);
    EXPECT_TRUE(MMN->getResult().getType()->isQuantizedType());

    // Verify that the variable inputs to the matmul are quantized.
    auto *LHS = llvm::dyn_cast<Constant>(MMN->getLHS());
    ASSERT_TRUE(LHS);
    EXPECT_TRUE(LHS->getType()->isQuantizedType());

    auto *RHS = llvm::dyn_cast<Constant>(MMN->getRHS());
    ASSERT_TRUE(RHS);
    EXPECT_TRUE(RHS->getType()->isQuantizedType());
  }
}

/// Check that quantizeFunction directly converts the constants
/// instead of leaving quantize node around.
TEST(Quantization, quantizeFunctionConvertConstant) {
  ExecutionEngine EE{};
  PlaceholderBindings bindings;
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  auto *LHS = mod.createPlaceholder(ElemKind::FloatTy, {3, 3}, "lhs", true);
  auto *RHS = mod.createConstant(ElemKind::FloatTy, {3, 3}, "rhs");
  bindings.allocate(LHS)->init(Tensor::InitKind::Xavier, 3, mod.getPRNG());
  RHS->getPayloadMutable().init(Tensor::InitKind::Xavier, 3, mod.getPRNG());

  auto *MMN = F->createMatMul("matmul", LHS, RHS);
  auto *save = F->createSave("ret", MMN);
  auto *result = save->getPlaceholder();
  bindings.allocate(result);

  // Note that we are creating profiling info even for nodes that will not be
  // quantized. This is how we expect quantizeFunction() to behave, as
  // quantization profiling will still get a profile for these nodes.
  quantization::QuantizationConfiguration quantConfig{{
      {LHS->getOutput().generateNodeOutputName(), {0.3f, 3.0f}},
      {RHS->getOutput().generateNodeOutputName(), {0.4f, 4.0f}},
      {MMN->getResult().generateNodeOutputName(), {0.6f, 6.0f}},
  }};

  quantConfig.assertAllNodesQuantized = true;
  std::unique_ptr<Backend> backend(createBackend(EE.getBackendName()));
  quantization::quantizeFunction(F, quantConfig, *backend);

  optimize(F, CompilationMode::Infer);
  CompilationContext cctx;
  convertQuantizedConstants(F, cctx);

  {
    // Verify that the output variable is not quantized, and that it has a
    // single save node writer, which is also not quantized.
    EXPECT_TRUE(!result->getType()->isQuantizedType());
    ASSERT_EQ(result->getUsers().size(), 1);
    auto *SN = llvm::dyn_cast<SaveNode>(result->getUsers().begin()->getUser());
    ASSERT_TRUE(SN);
    EXPECT_TRUE(!SN->getOutput().getType()->isQuantizedType());

    // Verify that the input to save is a dequantize node.
    auto *DN = llvm::dyn_cast<DequantizeNode>(SN->getInput());
    ASSERT_TRUE(DN);

    // Verify that the matmul is quantized.
    auto *MMN = llvm::dyn_cast<MatMulNode>(DN->getInput());
    ASSERT_TRUE(MMN);
    EXPECT_TRUE(MMN->getResult().getType()->isQuantizedType());

    // Verify that the variable inputs to the matmul are quantized.
    auto *LHSQuantize = llvm::dyn_cast<QuantizeNode>(MMN->getLHS());
    ASSERT_TRUE(LHSQuantize);
    EXPECT_EQ(LHSQuantize->getInput().getNode(), LHS);

    auto *RHS = llvm::dyn_cast<Constant>(MMN->getRHS());
    ASSERT_TRUE(RHS);
    EXPECT_TRUE(RHS->getType()->isQuantizedType());
  }

  // Make sure that graph can be compiled and run.
  EE.compile(CompilationMode::Infer);

  EE.run(bindings);
}

/// Check that the slice node doesn't change the quantization parameters between
/// its input and output.
TEST(Quantization, quantizeSlice) {
  ExecutionEngine EE{};
  PlaceholderBindings bindings;
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  auto *input = mod.createPlaceholder(ElemKind::FloatTy, {4}, "input", true);
  bindings.allocate(input)->init(Tensor::InitKind::Xavier, 3, mod.getPRNG());

  auto *slice = F->createSlice("slice", input, {2}, {3});
  auto *save = F->createSave("ret", slice);
  auto *result = save->getPlaceholder();
  bindings.allocate(result);

  quantization::QuantizationConfiguration quantConfig{{
      {slice->getResult().generateNodeOutputName(), {0.2f, 2.0f}},
      {input->getOutput().generateNodeOutputName(), {0.4f, 4.0f}},
  }};

  // Compute quantization parameters for verification.
  auto sliceInpTQP = chooseQuantizationParams({0.4, 4.0}, quantConfig.schema,
                                              quantConfig.precision);

  quantConfig.assertAllNodesQuantized = true;
  std::unique_ptr<Backend> backend(createBackend(EE.getBackendName()));
  quantization::quantizeFunction(F, quantConfig, *backend);

  optimize(F, CompilationMode::Infer);

  {
    // Verify that the output variable is not quantized, and that it has a
    // single save node writer, which is also not quantized.
    EXPECT_TRUE(!result->getType()->isQuantizedType());
    ASSERT_EQ(result->getUsers().size(), 1);
    auto *SN = llvm::dyn_cast<SaveNode>(result->getUsers().begin()->getUser());
    ASSERT_TRUE(SN);
    EXPECT_TRUE(!SN->getOutput().getType()->isQuantizedType());

    // Verify that the input to save is a dequantize node.
    auto *DN = llvm::dyn_cast<DequantizeNode>(SN->getInput());
    ASSERT_TRUE(DN);

    // Verify that the slice is rescaled after being quantized.
    // The reason we need a rescale is because slicing doesn't perform rescaling
    // by itself.
    // Note: after optimization, the RescaleQuantized node created for the Slice
    // gets merged with the dequantize node.
    auto *qslice = llvm::dyn_cast<SliceNode>(DN->getInput());
    ASSERT_TRUE(qslice);
    ASSERT_TRUE(qslice->getResult().getType()->isQuantizedType());
    EXPECT_EQ(qslice->getResult().getType()->getOffset(), sliceInpTQP.offset);
    EXPECT_EQ(qslice->getResult().getType()->getScale(), sliceInpTQP.scale);

    // Verify that the variable inputs to the matmul are quantized.
    auto *qinput = llvm::dyn_cast<QuantizeNode>(qslice->getInput());
    ASSERT_TRUE(qinput);
    EXPECT_EQ(qinput->getResult().getType()->getOffset(),
              qslice->getResult().getType()->getOffset());
    EXPECT_EQ(qinput->getResult().getType()->getScale(),
              qslice->getResult().getType()->getScale());
    EXPECT_EQ(qinput->getInput().getNode(), input);
  }

  // Make sure that graph can be compiled and run.
  EE.compile(CompilationMode::Infer);

  EE.run(bindings);
}

/// Check that the reshape node doesn't change the quantization parameters
/// between its input and output.
TEST(Quantization, quantizeReshape) {
  ExecutionEngine EE{};
  PlaceholderBindings bindings;
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  auto *input = mod.createPlaceholder(ElemKind::FloatTy, {3, 3}, "input", true);
  bindings.allocate(input)->init(Tensor::InitKind::Xavier, 3, mod.getPRNG());

  auto *reshape = F->createReshape("reshape", input, {9});
  auto *save = F->createSave("ret", reshape);
  auto *result = save->getPlaceholder();
  bindings.allocate(result);

  quantization::QuantizationConfiguration quantConfig{{
      {reshape->getResult().generateNodeOutputName(), {0.2f, 2.0f}},
      {input->getOutput().generateNodeOutputName(), {0.4f, 4.0f}},
  }};

  // Compute quantization parameters for verification.
  auto reshapeInpTQP = chooseQuantizationParams({0.4, 4.0}, quantConfig.schema,
                                                quantConfig.precision);

  quantConfig.assertAllNodesQuantized = true;
  std::unique_ptr<Backend> backend(createBackend(EE.getBackendName()));
  quantization::quantizeFunction(F, quantConfig, *backend);

  {
    // Verify that the output variable is not quantized, and that it has a
    // single save node writer, which is also not quantized.
    EXPECT_TRUE(!result->getType()->isQuantizedType());
    ASSERT_EQ(result->getUsers().size(), 1);
    auto *SN = llvm::dyn_cast<SaveNode>(result->getUsers().begin()->getUser());
    ASSERT_TRUE(SN);
    EXPECT_TRUE(!SN->getOutput().getType()->isQuantizedType());

    // Verify that the input to save is a dequantize node.
    auto *DN = llvm::dyn_cast<DequantizeNode>(SN->getInput());
    ASSERT_TRUE(DN);

    // Verify that the reshape is rescaled after being quantized.
    // The reason we need a rescale is because reshaping doesn't perform
    // rescaling by itself.
    auto *RQ = llvm::dyn_cast<RescaleQuantizedNode>(DN->getInput());
    ASSERT_TRUE(RQ);
    auto *qreshape = llvm::dyn_cast<ReshapeNode>(RQ->getInput());
    ASSERT_TRUE(qreshape);
    ASSERT_TRUE(qreshape->getResult().getType()->isQuantizedType());
    EXPECT_EQ(qreshape->getResult().getType()->getOffset(),
              reshapeInpTQP.offset);
    EXPECT_EQ(qreshape->getResult().getType()->getScale(), reshapeInpTQP.scale);

    // Verify that the input to the reshape is quantized.
    auto *qinput = llvm::dyn_cast<QuantizeNode>(qreshape->getInput());
    ASSERT_TRUE(qinput);
    EXPECT_EQ(qinput->getResult().getType()->getOffset(),
              qreshape->getResult().getType()->getOffset());
    EXPECT_EQ(qinput->getResult().getType()->getScale(),
              qreshape->getResult().getType()->getScale());
    EXPECT_EQ(qinput->getInput().getNode(), input);
  }

  // Make sure that graph can be compiled and run.
  EE.compile(CompilationMode::Infer);

  EE.run(bindings);
}

/// Mock backend that does not lower FC nodes.
class MockBackendUnloweredFC : public MockBackend {
  bool shouldLower(const Node *N) const override {
    if (N->getKind() == Kinded::Kind::FullyConnectedNodeKind) {
      return false;
    }
    return true;
  }
  bool isOpSupported(const NodeInfo &NI) const override { return true; }
};

/// Mock backend that does lower FC nodes.
class MockBackendLoweredFC : public MockBackend {
  bool shouldLower(const Node *N) const override { return true; }
  bool isOpSupported(const NodeInfo &NI) const override { return true; }
};

/// Create a simple network with an FC given \p bindings, \p EE, and \p F.
/// \returns the FC node.
static FullyConnectedNode *createSimpleFCNet(PlaceholderBindings &bindings,
                                             ExecutionEngine &EE, Function &F) {
  auto &mod = EE.getModule();
  auto *input = mod.createPlaceholder(ElemKind::FloatTy, {1, 3}, "input", true);
  auto *W = mod.createPlaceholder(ElemKind::FloatTy, {3, 3}, "weights", true);
  auto *B = mod.createPlaceholder(ElemKind::FloatTy, {3}, "bias", true);

  bindings.allocate(input)->getHandle().randomize(-1.0, 1.0, mod.getPRNG());
  bindings.allocate(W)->init(Tensor::InitKind::Xavier, 3, mod.getPRNG());
  bindings.allocate(B)->init(Tensor::InitKind::Broadcast, 0.1, mod.getPRNG());

  auto *FC = F.createFullyConnected("FC", input, W, B);
  auto *S = F.createSave("ret", FC);
  ::glow::convertPlaceholdersToConstants(&F, bindings,
                                         {input, S->getPlaceholder()});
  bindings.allocate(S->getPlaceholder());

  return FC;
}

/// Helper to look for a node with kind \p NodeClass in \p F. If found, \returns
/// a pointer to the node. Otherwise \returns a nullptr.
template <class NodeClass>
static NodeClass *findNodeKindOrReturnNull(Function *F) {
  auto it = std::find_if(
      F->getNodes().begin(), F->getNodes().end(),
      [](const Node &node) -> bool { return llvm::isa<NodeClass>(&node); });
  if (it == F->getNodes().end()) {
    return nullptr;
  }
  return &llvm::cast<NodeClass>(*it);
}

/// Profile and quantize a graph with an FC, and make sure that we find the
/// correct quantization parameters, whether the \p BackendClass does or does
/// not lower the FC given \p expectLoweredFC. Note that in this test we
/// replicate the logic from optimizeFunction(), wherein we lower and then call
/// profileQuantization(), in order to ensure each stage of the compilation
/// pipeline for profiling/quantization is correct.
template <class BackendClass>
static void testProfileQuantizationOfFC(bool expectLoweredFC,
                                        bool rowwiseQuantizeFC) {
  ExecutionEngine profileEE{};
  Function *profileF = profileEE.getModule().createFunction("profile");
  PlaceholderBindings profilebindings;
  FullyConnectedNode *FC =
      createSimpleFCNet(profilebindings, profileEE, *profileF);
  auto outputNameFC = FC->getResult().generateNodeOutputName();
  auto weightsNameFC = FC->getWeights().generateNodeOutputName();
  auto biasNameFC = FC->getBias().generateNodeOutputName();
  auto inputNameFC = FC->getInput().generateNodeOutputName();

  // Lower everything and keep track of the lowered components source nodes via
  // the loweredMap.
  LoweredInfoMap loweredMapForProf;
  CompilationContext cctx(/* bindings */ nullptr, &loweredMapForProf);
  lower(profileF, cctx);

  // Check that the lowered graph only contains the lowered components of the
  // FC (MM and BA) and not the FC itself.
  auto *loweredFC = findNodeKindOrReturnNull<FullyConnectedNode>(profileF);
  auto *loweredMM = findNodeKindOrReturnNull<MatMulNode>(profileF);
  auto *loweredBA = findNodeKindOrReturnNull<BatchedAddNode>(profileF);
  ASSERT_FALSE(loweredFC);
  ASSERT_TRUE(loweredMM);
  ASSERT_TRUE(loweredBA);
  auto outputNameMM = loweredMM->getResult().generateNodeOutputName();
  auto outputNameBA = loweredBA->getResult().generateNodeOutputName();

  glow::profileQuantization(profilebindings, profileF,
                            cctx.precisionConfig.profConfig);

  // Compile/run to capture profile.
  profileEE.compile(CompilationMode::Infer);
  profileEE.run(profilebindings);

  // Get profiling infos and build new quantized graph, passing in the
  // loweredMapForProf to include the unlowered components in QI.
  profileF = profileEE.getModule().getFunctions().front();
  quantization::QuantizationConfiguration quantConfig{
      quantization::generateNodeProfilingInfos(profilebindings, profileF,
                                               loweredMapForProf)};

  // Verify that we have node profiling infos for the FC and the lowered
  // components of the FC (MM and BA).
  NodeProfilingInfo *FCPI = nullptr, *MMPI = nullptr, *BAPI = nullptr,
                    *FCWPI = nullptr, *FCBPI = nullptr, *FCIPI = nullptr;
  for (NodeProfilingInfo &NPI : quantConfig.infos) {
    if (NPI.nodeOutputName_ == outputNameFC) {
      FCPI = &NPI;
    } else if (NPI.nodeOutputName_ == outputNameMM) {
      MMPI = &NPI;
    } else if (NPI.nodeOutputName_ == outputNameBA) {
      BAPI = &NPI;
    } else if (NPI.nodeOutputName_ == weightsNameFC) {
      FCWPI = &NPI;
    } else if (NPI.nodeOutputName_ == biasNameFC) {
      FCBPI = &NPI;
    } else if (NPI.nodeOutputName_ == inputNameFC) {
      FCIPI = &NPI;
    }
  }
  ASSERT_TRUE(FCPI);
  ASSERT_TRUE(MMPI);
  ASSERT_TRUE(BAPI);
  ASSERT_TRUE(FCWPI);
  ASSERT_TRUE(FCBPI);
  ASSERT_TRUE(FCIPI);

  // Compute quantization parameters for verification.
  auto FCTQP = chooseQuantizationParams(
      FCPI->tensorProfilingParams_, quantConfig.schema, quantConfig.precision);
  auto MMTQP = chooseQuantizationParams(
      MMPI->tensorProfilingParams_, quantConfig.schema, quantConfig.precision);
  auto BATQP = chooseQuantizationParams(
      BAPI->tensorProfilingParams_, quantConfig.schema, quantConfig.precision);
  auto FCWTQP = chooseQuantizationParams(
      FCWPI->tensorProfilingParams_, quantConfig.schema, quantConfig.precision);
  auto FCBTQP =
      chooseQuantizationParams(FCBPI->tensorProfilingParams_,
                               quantConfig.schema, quantConfig.precisionBias);
  auto FCITQP = chooseQuantizationParams(
      FCIPI->tensorProfilingParams_, quantConfig.schema, quantConfig.precision);

  // Now create the same original function in the backend we're testing.
  ExecutionEngine backendEE;
  BackendClass backend;
  Backend *backendPtr = &backend;
  // backendEE.setBackend(&backend, /* ownsBackend */ false);
  Function *backendF = backendEE.getModule().createFunction("quantized");
  PlaceholderBindings backendbindings;
  createSimpleFCNet(backendbindings, backendEE, *backendF);

  // Lower the function given the backend's preferences for lowering.
  LoweredInfoMap loweredMapForQuant;
  CompilationContext cctx2(/* bindings */ nullptr, &loweredMapForQuant);
  lower(backendF, cctx2, backendPtr);

  // Check that the backend lowered the function as expected.
  auto *floatFC = findNodeKindOrReturnNull<FullyConnectedNode>(backendF);
  auto *floatMM = findNodeKindOrReturnNull<MatMulNode>(backendF);
  auto *floatBA = findNodeKindOrReturnNull<BatchedAddNode>(backendF);
  if (expectLoweredFC) {
    ASSERT_FALSE(floatFC);
    ASSERT_TRUE(floatMM);
    ASSERT_TRUE(floatBA);
  } else {
    ASSERT_TRUE(floatFC);
    ASSERT_FALSE(floatMM);
    ASSERT_FALSE(floatBA);
  }

  // Quantize the function given the current backend we're testing along with
  // the quantization infos gathered.
  quantConfig.enableRowwise = rowwiseQuantizeFC;
  quantConfig.assertAllNodesQuantized = true;
  quantization::quantizeFunction(backendF, quantConfig, *backendPtr,
                                 loweredMapForQuant);

  // Optimize the graph to remove dead code and optimize away unnecessary
  // quantize nodes. Note that we do not do a full compile call here, as we have
  // already lowered.
  ::glow::optimize(backendF, CompilationMode::Infer);

  // Check that the graph is still structured as expected, and that the
  // scales/offsets are set as found in TQP.
  auto *quantFC = findNodeKindOrReturnNull<FullyConnectedNode>(backendF);
  auto *quantMM = findNodeKindOrReturnNull<MatMulNode>(backendF);
  auto *quantBA = findNodeKindOrReturnNull<BatchedAddNode>(backendF);
  auto *quantRowwiseFC =
      findNodeKindOrReturnNull<RowwiseQuantizedFullyConnectedNode>(backendF);

  if (rowwiseQuantizeFC) {
    EXPECT_FALSE(quantMM);
    EXPECT_FALSE(quantBA);
    EXPECT_FALSE(quantFC);

    ASSERT_TRUE(quantRowwiseFC);
    EXPECT_EQ(quantRowwiseFC->getResult().getType()->getScale(), FCTQP.scale);
    EXPECT_EQ(quantRowwiseFC->getResult().getType()->getOffset(), FCTQP.offset);

    EXPECT_EQ(quantRowwiseFC->getBias().getElementType(), ElemKind::Int32QTy);
    // Bias scale was changed with the product inputScale * weightsScale only
    // if the product was larger.
    if (FCWTQP.scale * FCITQP.scale > FCBTQP.scale) {
      EXPECT_EQ(quantRowwiseFC->getBias().getType()->getScale(),
                FCWTQP.scale * FCITQP.scale);
      EXPECT_EQ(quantRowwiseFC->getBias().getType()->getOffset(), 0);
    } else {
      EXPECT_EQ(quantRowwiseFC->getBias().getType()->getScale(), FCBTQP.scale);
      EXPECT_EQ(quantRowwiseFC->getBias().getType()->getOffset(), 0);
    }
  } else if (expectLoweredFC) {
    ASSERT_FALSE(quantFC);
    ASSERT_FALSE(quantRowwiseFC);

    ASSERT_TRUE(quantMM);
    EXPECT_EQ(quantMM->getResult().getType()->getScale(), MMTQP.scale);
    EXPECT_EQ(quantMM->getResult().getType()->getOffset(), MMTQP.offset);

    ASSERT_TRUE(quantBA);
    EXPECT_EQ(quantBA->getResult().getType()->getScale(), BATQP.scale);
    EXPECT_EQ(quantBA->getResult().getType()->getOffset(), BATQP.offset);

    EXPECT_EQ(quantBA->getSlice().getElementType(), ElemKind::Int32QTy);
    // Bias scale was changed with the product inputScale * weightsScale only
    // if the product was larger.
    if (FCWTQP.scale * FCITQP.scale > FCBTQP.scale) {
      EXPECT_EQ(quantBA->getSlice().getType()->getScale(),
                FCWTQP.scale * FCITQP.scale);
      EXPECT_EQ(quantBA->getSlice().getType()->getOffset(), 0);
    } else {
      EXPECT_EQ(quantBA->getSlice().getType()->getScale(), FCBTQP.scale);
      EXPECT_EQ(quantBA->getSlice().getType()->getOffset(), 0);
    }
  } else {
    ASSERT_FALSE(quantRowwiseFC);

    ASSERT_TRUE(quantFC);
    EXPECT_EQ(quantFC->getResult().getType()->getScale(), FCTQP.scale);
    EXPECT_EQ(quantFC->getResult().getType()->getOffset(), FCTQP.offset);

    ASSERT_FALSE(quantMM);
    ASSERT_FALSE(quantBA);

    EXPECT_EQ(quantFC->getBias().getElementType(), ElemKind::Int32QTy);
    // Bias scale was changed with the product inputScale * weightsScale only
    // if the product was larger.
    if (FCWTQP.scale * FCITQP.scale > FCBTQP.scale) {
      EXPECT_EQ(quantFC->getBias().getType()->getScale(),
                FCWTQP.scale * FCITQP.scale);
      EXPECT_EQ(quantFC->getBias().getType()->getOffset(), 0);
    } else {
      EXPECT_EQ(quantFC->getBias().getType()->getScale(), FCBTQP.scale);
      EXPECT_EQ(quantFC->getBias().getType()->getOffset(), 0);
    }
  }
}

/// Test that backends that do not lower FCs can find the quantization
/// parameters of their nodes.
TEST(Quantization, TestProfileQuantizationOfUnloweredFC) {
  testProfileQuantizationOfFC<MockBackendUnloweredFC>(
      /* expectLoweredFC */ false, /* rowwiseQuantizeFC */ false);
}

/// Test that backends that do lower FCs can find the quantization parameters of
/// their nodes.
TEST(Quantization, TestProfileQuantizationOfLoweredFC) {
  testProfileQuantizationOfFC<MockBackendLoweredFC>(
      /* expectLoweredFC */ true, /* rowwiseQuantizeFC */ false);
}

/// Test that backends that do not lower FCs can find the quantization
/// parameters of their nodes and correctly rowwise quantize.
TEST(Quantization, TestProfileQuantizationOfUnloweredFCRowwise) {
  testProfileQuantizationOfFC<MockBackendUnloweredFC>(
      /* expectLoweredFC */ false, /* rowwiseQuantizeFC */ true);
}

/// Test that backends that do lower FCs can find the quantization parameters of
/// their nodes and correctly rowwise quantize even when lowering the FC.
TEST(Quantization, TestProfileQuantizationOfLoweredFCRowwise) {
  testProfileQuantizationOfFC<MockBackendLoweredFC>(
      /* expectLoweredFC */ true, /* rowwiseQuantizeFC */ true);
}

/// Check that asserting quantization for the quantizer works as expected.
TEST(Quantization, CheckAssertQuantization) {
  ExecutionEngine EE{};
  std::unique_ptr<Backend> backend(createBackend(EE.getBackendName()));
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  auto *input = mod.createPlaceholder(ElemKind::FloatTy, {1, 3}, "input", true);
  auto *relu = F->createRELU("ReLU", input);
  PlaceholderBindings bindings;
  auto *save = F->createSave("ret", relu);
  bindings.allocate(save->getPlaceholder());

  quantization::QuantizationConfiguration quantConfig{
      {{input->getOutput().generateNodeOutputName(), {0.2f, 2.0f}},
       {relu->getResult().generateNodeOutputName(), {0.2f, 3.0f}}}};
  quantConfig.precision = ElemKind::Int16QTy;
  quantConfig.assertAllNodesQuantized = true;

  // Expect this to die because quantizeFunction() is passed with
  // assertAllNodesQuantized true, and the Interpreter backend does not support
  // Int16QTy ReLU.
  Function *QF = F->clone("quant_clone1");
  EXPECT_DEATH(quantization::quantizeFunction(QF, quantConfig, *backend), "");

  {
    Function *QF = F->clone("quant_clone2");
    quantConfig.assertAllNodesQuantized = false;

    // This works fine because quantizeFunction() is passed with
    // assertAllNodesQuantized false, and so the ReLU will not be quantized as
    // the Interpreter does not support Int16QTy ReLU.
    quantization::quantizeFunction(QF, quantConfig, *backend);

    auto *saveNode =
        llvm::dyn_cast<SaveNode>(QF->getNodeByName(save->getName()));
    ASSERT_TRUE(saveNode);
    auto *reluNode = llvm::dyn_cast<ReluNode>(saveNode->getInput().getNode());
    ASSERT_TRUE(reluNode);
    EXPECT_TRUE(!reluNode->getResult().getType()->isQuantizedType());
  }

  {
    Function *QF = F->clone("quant_clone3");
    quantConfig.assertAllNodesQuantized = true;
    KindSet doNotQuantizeKinds;
    doNotQuantizeKinds.insert(Kinded::Kind::ReluNodeKind);

    // This works fine because quantizeFunction() is passed with
    // assertAllNodesQuantized true, but we explicitly tell the quantizer to
    // keep ReLU in its original precision.
    quantization::quantizeFunction(QF, quantConfig, *backend,
                                   /* loweredMap */ {}, doNotQuantizeKinds);

    auto *saveNode =
        llvm::dyn_cast<SaveNode>(QF->getNodeByName(save->getName()));
    ASSERT_TRUE(saveNode);
    auto *reluNode = llvm::dyn_cast<ReluNode>(saveNode->getInput().getNode());
    ASSERT_TRUE(reluNode);
    EXPECT_TRUE(!reluNode->getResult().getType()->isQuantizedType());
  }
}

/// Check that we can quantize nodes that have some quantized outputs as unused,
/// e.g. a TopK node where values is unused but indices is.
TEST(Quantization, QuantizationZeroUsersResult) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  PlaceholderBindings bindings;
  Function *F = mod.createFunction("main");
  auto *input =
      mod.createPlaceholder(ElemKind::FloatTy, {3, 1, 5}, "input", false);

  bindings.allocate(input)->getHandle() = {
      28, 4, 411, 19, 42, 0.4f, 0.4f, 0.4f, -0.4f, 0.45f, 7, 5, 9, 8, 100,
  };

  // Note we intentionally do not save the topk's values result.
  auto *TK = F->createTopK("TopK", input, 3);
  auto *SN = F->createSave("save_indices", TK->getIndices());
  bindings.allocate(SN->getPlaceholder());

  quantization::QuantizationConfiguration quantConfig{
      {{input->getOutput().generateNodeOutputName(), {0.2f, 2.0f}},
       {TK->getValues().generateNodeOutputName(), {0.2f, 3.0f}}}};
  quantConfig.assertAllNodesQuantized = true;

  std::unique_ptr<Backend> backend(createBackend(EE.getBackendName()));
  quantization::quantizeFunction(F, quantConfig, *backend);

  auto *qSN = llvm::dyn_cast<SaveNode>(F->getNodeByName(SN->getName()));
  ASSERT_TRUE(qSN);
  auto *qTK = llvm::dyn_cast<TopKNode>(qSN->getInput().getNode());
  ASSERT_TRUE(qTK);
  EXPECT_TRUE(qTK->getValues().getType()->isQuantizedType());
}

#ifdef GLOW_WITH_CPU

GLOW_INSTANTIATE_TEST_SUITE_P(
    InterpAndCPUProfAndQuant, Operator,
    ::testing::Combine(::testing::Values("Interpreter", "CPU"),
                       ::testing::Values("Interpreter", "CPU")));

#else
GLOW_INSTANTIATE_TEST_SUITE_P(
    InterpreterProfAndQuant, Operator,
    ::testing::Combine(::testing::Values("Interpreter"),
                       ::testing::Values("Interpreter")));

#endif // GLOW_WITH_CPU

#ifdef GLOW_WITH_OPENCL
GLOW_INSTANTIATE_TEST_SUITE_P(
    InterpProfOpenCLQuant, Operator,
    ::testing::Combine(::testing::Values("Interpreter"),
                       ::testing::Values("OpenCL")));
#endif // GLOW_WITH_OPENCL

} // namespace glow
