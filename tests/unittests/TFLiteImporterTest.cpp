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

#include <fstream>

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
  std::string dataBasename = llvm::StringRef(modelPath).substr(0, dotPos);
  size_t inpIdx = 0;
  for (const auto &inpPH : inputPH) {
    std::string inpFilename = dataBasename + ".inp" + std::to_string(inpIdx++);
    Tensor *inpT = bindings.get(inpPH);
    loadTensor(inpT, inpFilename);
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

TFLITE_UNIT_TEST(Sin, "sin.tflite")

TFLITE_UNIT_TEST(Tile, "tile.tflite")

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

#undef TFLITE_UNIT_TEST
