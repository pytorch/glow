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
#include "ImporterTestUtils.h"
#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Graph/Graph.h"
#include "glow/Graph/Nodes.h"
#include "glow/Graph/PlaceholderBindings.h"
#include "glow/Importer/TFLiteModelLoader.h"
#include "gtest/gtest.h"

#include "llvm/Support/CommandLine.h"

#include <fstream>

namespace {

llvm::cl::OptionCategory tfliteModelTestCat("TFLITE Test Options");

llvm::cl::opt<bool> tflitePrintTestTensorsOpt(
    "tflite-dump-test-tensors", llvm::cl::init(false), llvm::cl::Optional,
    llvm::cl::desc(
        "Print input/expected tensors from test files. Default is false."),
    llvm::cl::cat(tfliteModelTestCat));

} // namespace

using namespace glow;

class TFLiteImporterTest : public ::testing::Test {};

/// \p returns the full path of the TensorFlowLite model \p name.
static std::string getModelPath(std::string name) {
  return "tests/models/tfliteModels/" + name;
}

/// Utility function to load a binary file from \p fileName into \p tensor.
/// The binary files have a special format with an extra byte of '0' at the
/// start of the file followed by the actual tensor binary content. The extra
/// '0' leading byte was required in order for the GIT system to correctly
/// recognize the files as being binary files.
static void loadTensor(Tensor *tensor, const std::string &fileName) {
  std::ifstream file;
  file.open(fileName, std::ios::binary);
  assert(file.is_open() && "Error opening tensor file!");
  file.seekg(1);
  file.read(tensor->getUnsafePtr(), tensor->getSizeInBytes());
  file.close();
}

/// Utility function to load and run TensorFlowLite model named \p modelName.
/// The model with the name <name>.tflite is also associated with binary files
/// used to validate numerically the model. The binary files have the following
/// naming convention: <name>.inp0, <name>.inp1, etc for the model inputs and
/// <name>.out0, <name>.out1, etc for the model reference outputs. When testing
/// the output of the model a maximum error of \p maxError is allowed.
static void loadAndRunModel(std::string modelName, float maxError = 1e-6) {
  ExecutionEngine EE;
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  // Load TensorFlowLite model.
  std::string modelPath = getModelPath(modelName);
  { TFLiteModelLoader(modelPath, F); }

  // Allocate tensors for all placeholders.
  PlaceholderBindings bindings;
  bindings.allocate(mod.getPlaceholders());

  // Get model input/output placeholders.
  PlaceholderList inputPH;
  PlaceholderList outputPH;
  for (const auto &ph : mod.getPlaceholders()) {
    if (isInput(ph, *F)) {
      inputPH.push_back(ph);
    } else {
      outputPH.push_back(ph);
    }
  }

  // Load data into the input placeholders.
  size_t dotPos = llvm::StringRef(modelPath).find_first_of('.');
  std::string dataBasename = std::string(modelPath).substr(0, dotPos);
  size_t inpIdx = 0;
  for (const auto &inpPH : inputPH) {
    std::string inpFilename = dataBasename + ".inp" + std::to_string(inpIdx++);
    Tensor *inpT = bindings.get(inpPH);
    loadTensor(inpT, inpFilename);
    if (tflitePrintTestTensorsOpt) {
      llvm::outs() << "Input Placeholder: " << inpPH->getName() << "\n";
      inpT->dump();
    }
  }

  // Run model.
  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  // Compare output data versus reference.
  size_t outIdx = 0;
  for (const auto &outPH : outputPH) {
    std::string refFilename = dataBasename + ".out" + std::to_string(outIdx++);

    // Get output tensor.
    Tensor *outT = bindings.get(outPH);

    // Load reference tensor.
    Tensor refT(outT->getType());
    loadTensor(&refT, refFilename);
    if (tflitePrintTestTensorsOpt) {
      llvm::outs() << "Reference Tensor:\n";
      refT.dump();
      llvm::outs() << "Output Placeholder: " << outPH->getName() << "\n";
      outT->dump();
    }

    // Compare.
    ASSERT_TRUE(outT->isEqual(refT, maxError, /* verbose */ true));
  }
}

#define TFLITE_UNIT_TEST(name, model)                                          \
  TEST(TFLiteImporterTest, name) { loadAndRunModel(model); }

TFLITE_UNIT_TEST(Add, "add.tflite")

TFLITE_UNIT_TEST(AvgPool2D_PaddingSame, "avgpool2d_same.tflite")
TFLITE_UNIT_TEST(AvgPool2D_PaddingValid, "avgpool2d_valid.tflite")

TFLITE_UNIT_TEST(Concat, "concat.tflite")
TFLITE_UNIT_TEST(ConcatNegAxis, "concat_neg_axis.tflite")

TFLITE_UNIT_TEST(Conv2D_PaddingSame, "conv2d_same.tflite")
TFLITE_UNIT_TEST(Conv2D_PaddingValid, "conv2d_valid.tflite")
TFLITE_UNIT_TEST(Conv2D_FusedRelu, "conv2d_relu.tflite")

TFLITE_UNIT_TEST(DepthwiseConv2D_Ch1Mult1, "depthwise_conv2d_c1_m1.tflite")
TFLITE_UNIT_TEST(DepthwiseConv2D_Ch1Mult2, "depthwise_conv2d_c1_m2.tflite")
TFLITE_UNIT_TEST(DepthwiseConv2D_Ch2Mult1, "depthwise_conv2d_c2_m1.tflite")
TFLITE_UNIT_TEST(DepthwiseConv2D_Ch2Mult2, "depthwise_conv2d_c2_m2.tflite")

TFLITE_UNIT_TEST(HardSwish, "hardSwish.tflite")

TFLITE_UNIT_TEST(Floor, "floor.tflite")

TFLITE_UNIT_TEST(FullyConnected, "fully_connected.tflite")

TFLITE_UNIT_TEST(Sigmoid, "sigmoid.tflite")

TFLITE_UNIT_TEST(MaxPool2D_PaddingSame, "maxpool2d_same.tflite")
TFLITE_UNIT_TEST(MaxPool2D_PaddingValid, "maxpool2d_valid.tflite")

TFLITE_UNIT_TEST(Mul, "mul.tflite")

TFLITE_UNIT_TEST(Relu, "relu.tflite")

TFLITE_UNIT_TEST(ReluN1To1, "relu_n1to1.tflite")

TFLITE_UNIT_TEST(Relu6, "relu6.tflite")

TFLITE_UNIT_TEST(Reshape, "reshape.tflite")
TFLITE_UNIT_TEST(ReshapeNegShape, "reshape_neg_shape.tflite")

TFLITE_UNIT_TEST(Softmax, "softmax.tflite")

TFLITE_UNIT_TEST(Tanh, "tanh.tflite")

TFLITE_UNIT_TEST(Pad, "pad.tflite")

TFLITE_UNIT_TEST(Transpose, "transpose.tflite")

TFLITE_UNIT_TEST(MeanKeepDims, "mean_keep_dims.tflite")
TFLITE_UNIT_TEST(MeanNoKeepDims, "mean_no_keep_dims.tflite")
TFLITE_UNIT_TEST(MeanMultipleAxisKeepDims,
                 "mean_multiple_axis_keep_dims.tflite")
TFLITE_UNIT_TEST(MeanMultipleAxisNoKeepDims,
                 "mean_multiple_axis_no_keep_dims.tflite")

TFLITE_UNIT_TEST(Sub, "sub.tflite")

TFLITE_UNIT_TEST(Div, "div.tflite")

TFLITE_UNIT_TEST(Exp, "exp.tflite")

TFLITE_UNIT_TEST(Split, "split.tflite")

TFLITE_UNIT_TEST(PRelu, "prelu.tflite")

TFLITE_UNIT_TEST(Maximum, "max.tflite")

TFLITE_UNIT_TEST(ArgMax, "arg_max.tflite")

TFLITE_UNIT_TEST(Minimum, "min.tflite")

TFLITE_UNIT_TEST(Less, "less.tflite")

TFLITE_UNIT_TEST(Neg, "neg.tflite")

TFLITE_UNIT_TEST(Greater, "greater.tflite")

TFLITE_UNIT_TEST(GreaterEqual, "greater_equal.tflite")

TFLITE_UNIT_TEST(LessEqual, "less_equal.tflite")

TFLITE_UNIT_TEST(Slice, "slice.tflite")
TFLITE_UNIT_TEST(SliceNegSize, "slice_neg_size.tflite")

TFLITE_UNIT_TEST(StridedSliceTest0, "strided_slice_test0.tflite")
TFLITE_UNIT_TEST(StridedSliceTest1, "strided_slice_test1.tflite")
TFLITE_UNIT_TEST(StridedSliceTest2, "strided_slice_test2.tflite")
TFLITE_UNIT_TEST(StridedSliceTest3, "strided_slice_test3.tflite")
TFLITE_UNIT_TEST(StridedSliceTest4, "strided_slice_test4.tflite")
TFLITE_UNIT_TEST(StridedSliceTest5, "strided_slice_test5.tflite")
TFLITE_UNIT_TEST(StridedSliceTest6, "strided_slice_test6.tflite")

TFLITE_UNIT_TEST(Sin, "sin.tflite")

TFLITE_UNIT_TEST(Tile, "tile.tflite")

TFLITE_UNIT_TEST(ResizeBilinear, "resize_bilinear.tflite")

TFLITE_UNIT_TEST(ResizeNearest, "resize_nearest.tflite")

TFLITE_UNIT_TEST(SpaceToDepth, "space_to_depth.tflite")

TFLITE_UNIT_TEST(DepthToSpace, "depth_to_space.tflite")

TFLITE_UNIT_TEST(CastF32ToInt32, "cast_f32_to_int32.tflite")

TFLITE_UNIT_TEST(GatherAxis0, "gather_axis0.tflite")
TFLITE_UNIT_TEST(GatherAxis1, "gather_axis1.tflite")

TFLITE_UNIT_TEST(GatherND, "gather_nd.tflite")

TFLITE_UNIT_TEST(LogSoftmax, "log_softmax.tflite")

TFLITE_UNIT_TEST(Select, "select.tflite")

TFLITE_UNIT_TEST(SpaceToBatchNd, "spaceToBatchNd.tflite")
TFLITE_UNIT_TEST(BatchToSpaceNd, "batchToSpaceNd.tflite")

TFLITE_UNIT_TEST(Equal, "equal.tflite")

TFLITE_UNIT_TEST(NotEqual, "not_equal.tflite")

TFLITE_UNIT_TEST(Log, "log.tflite")

TFLITE_UNIT_TEST(Sqrt, "sqrt.tflite")

TFLITE_UNIT_TEST(Rsqrt, "rsqrt.tflite")

TFLITE_UNIT_TEST(Pow, "pow.tflite")

TFLITE_UNIT_TEST(ArgMin, "arg_min.tflite")

TFLITE_UNIT_TEST(Pack, "pack.tflite")

TFLITE_UNIT_TEST(LogicalOr, "logical_or.tflite")

TFLITE_UNIT_TEST(LogicalAnd, "logical_and.tflite")

TFLITE_UNIT_TEST(LogicalNot, "logical_not.tflite")

TFLITE_UNIT_TEST(Unpack, "unpack.tflite")

TFLITE_UNIT_TEST(Square, "square.tflite")

TFLITE_UNIT_TEST(LeakyRelu, "leaky_relu.tflite")

TFLITE_UNIT_TEST(Abs, "abs.tflite")

TFLITE_UNIT_TEST(Ceil, "ceil.tflite")

TFLITE_UNIT_TEST(Cos, "cos.tflite")

TFLITE_UNIT_TEST(Round, "round.tflite")

TFLITE_UNIT_TEST(Add_broadcast, "add_broadcast.tflite")
TFLITE_UNIT_TEST(Sub_broadcast, "sub_broadcast.tflite")
TFLITE_UNIT_TEST(Div_broadcast, "div_broadcast.tflite")
TFLITE_UNIT_TEST(Mul_broadcast, "mul_broadcast.tflite")
TFLITE_UNIT_TEST(Min_broadcast, "min_broadcast.tflite")
TFLITE_UNIT_TEST(Max_broadcast, "max_broadcast.tflite")

#undef TFLITE_UNIT_TEST

/// Test Regular TFLiteDetectionPostProcess node.
TEST(TFLiteImporterTest, TFLiteDetectionPostProcessRegular) {
  ExecutionEngine EE;
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  // Load TensorFlowLite model.
  std::string modelPath =
      getModelPath("tflite_detection_post_processing_regular.tflite");
  { TFLiteModelLoader(modelPath, F); }

  // Allocate tensors for all placeholders.
  PlaceholderBindings bindings;
  bindings.allocate(mod.getPlaceholders());

  // Get model input/output placeholders.
  std::vector<Placeholder *> inputPH;
  std::vector<Placeholder *> outputPH;
  for (const auto &ph : mod.getPlaceholders()) {
    if (isInput(ph, *F)) {
      inputPH.push_back(ph);
    } else {
      outputPH.push_back(ph);
    }
  }

  // Load data into the input placeholders.
  loadTensor(bindings.get(inputPH[0]),
             getModelPath("tflite_detection_post_processing_boxes.bin"));
  loadTensor(bindings.get(inputPH[1]),
             getModelPath("tflite_detection_post_processing_scores.bin"));

  // Run model.
  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  // Compare output data versus reference.
  std::vector<float> detectionBoxesRef = {
      0.270546197891235, 0.036445915699005, 0.625426292419434,
      0.715417265892029, 0.008843034505844, 0.453001916408539,
      0.434335529804230, 1.007383584976196, 0.264277368783951,
      0.225462928414345, 0.431514173746109, 0.499467015266418,
      0.012970104813576, 0.489649474620819, 0.433307945728302,
      1.010598421096802, 0.208248645067215, 0.414025753736496,
      0.256930917501450, 0.457198470830917, 0.259306669235229,
      0.276896983385086, 0.413792371749878, 0.558155655860901,
      0.296046763658524, 0.024428725242615, 0.620571494102478,
      0.726388156414032, 0.100624501705170, 0.478332787752151,
      0.341053903102875, 0.616274893283844, 0.195692524313927,
      0.446290910243988, 0.264245152473450, 0.527587413787842,
      0.232087373733521, 0.244561776518822, 0.373351573944092,
      0.512895405292511,
  };
  std::vector<int32_t> detectionClassesRef = {
      2, 7, 2, 5, 2, 2, 32, 7, 2, 2,
  };
  std::vector<float> detectionScoresRef = {
      0.709131240844727, 0.694569468498230, 0.563223838806152,
      0.540955007076263, 0.452089250087738, 0.439201682806015,
      0.433123916387558, 0.432144701480865, 0.416427463293076,
      0.408173263072968,
  };
  int32_t numDetectionsRef = 10;
  auto detectionBoxesH = bindings.get(outputPH[0])->getHandle<float>();
  auto detectionClassesH = bindings.get(outputPH[1])->getHandle<int32_t>();
  auto detectionScoresH = bindings.get(outputPH[2])->getHandle<float>();
  auto numDetectionsH = bindings.get(outputPH[3])->getHandle<int32_t>();
  for (size_t idx = 0; idx < 4 * numDetectionsRef; ++idx) {
    EXPECT_FLOAT_EQ(detectionBoxesH.raw(idx), detectionBoxesRef[idx]);
  }
  for (size_t idx = 0; idx < numDetectionsRef; ++idx) {
    EXPECT_EQ(detectionClassesH.raw(idx), detectionClassesRef[idx]);
    EXPECT_EQ(detectionScoresH.raw(idx), detectionScoresRef[idx]);
  }
  EXPECT_EQ(numDetectionsH.raw(0), numDetectionsRef);
}

/// Test Fast TFLiteDetectionPostProcess node.
TEST(TFLiteImporterTest, TFLiteDetectionPostProcessFast) {
  ExecutionEngine EE;
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  // Load TensorFlowLite model.
  std::string modelPath =
      getModelPath("tflite_detection_post_processing_fast.tflite");
  { TFLiteModelLoader(modelPath, F); }

  // Allocate tensors for all placeholders.
  PlaceholderBindings bindings;
  bindings.allocate(mod.getPlaceholders());

  // Get model input/output placeholders.
  std::vector<Placeholder *> inputPH;
  std::vector<Placeholder *> outputPH;
  for (const auto &ph : mod.getPlaceholders()) {
    if (isInput(ph, *F)) {
      inputPH.push_back(ph);
    } else {
      outputPH.push_back(ph);
    }
  }

  // Load data into the input placeholders.
  loadTensor(bindings.get(inputPH[0]),
             getModelPath("tflite_detection_post_processing_boxes.bin"));
  loadTensor(bindings.get(inputPH[1]),
             getModelPath("tflite_detection_post_processing_scores.bin"));

  // Run model.
  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  // Compare output data versus reference.
  std::vector<float> detectionBoxesRef = {
      0.270546197891235, 0.036445915699005, 0.625426292419434,
      0.715417265892029, 0.008843034505844, 0.453001916408539,
      0.434335529804230, 1.007383584976196, 0.264277368783951,
      0.225462928414345, 0.431514173746109, 0.499467015266418,
      0.208248645067215, 0.414025753736496, 0.256930917501450,
      0.457198470830917, 0.259306669235229, 0.276896983385086,
      0.413792371749878, 0.558155655860901, 0.100624501705170,
      0.478332787752151, 0.341053903102875, 0.616274893283844,
      0.195692524313927, 0.446290910243988, 0.264245152473450,
      0.527587413787842, 0.232087373733521, 0.244561776518822,
      0.373351573944092, 0.512895405292511, 0.275883287191391,
      0.037467807531357, 0.595628619194031, 0.463419944047928,
      0.203831464052200, 0.354441434144974, 0.266103237867355,
      0.427350491285324,
  };
  std::vector<int32_t> detectionClassesRef = {
      2, 7, 2, 2, 2, 7, 2, 2, 2, 2,
  };
  std::vector<float> detectionScoresRef = {
      0.709131240844727, 0.694569468498230, 0.563223838806152,
      0.452089250087738, 0.439201682806015, 0.432144701480865,
      0.416427463293076, 0.408173263072968, 0.405113369226456,
      0.398936122655869,
  };
  int32_t numDetectionsRef = 10;
  auto detectionBoxesH = bindings.get(outputPH[0])->getHandle<float>();
  auto detectionClassesH = bindings.get(outputPH[1])->getHandle<int32_t>();
  auto detectionScoresH = bindings.get(outputPH[2])->getHandle<float>();
  auto numDetectionsH = bindings.get(outputPH[3])->getHandle<int32_t>();

  for (size_t idx = 0; idx < 4 * numDetectionsRef; ++idx) {
    EXPECT_FLOAT_EQ(detectionBoxesH.raw(idx), detectionBoxesRef[idx]);
  }
  for (size_t idx = 0; idx < numDetectionsRef; ++idx) {
    EXPECT_EQ(detectionClassesH.raw(idx), detectionClassesRef[idx]);
    EXPECT_EQ(detectionScoresH.raw(idx), detectionScoresRef[idx]);
  }
  EXPECT_EQ(numDetectionsH.raw(0), numDetectionsRef);
}
