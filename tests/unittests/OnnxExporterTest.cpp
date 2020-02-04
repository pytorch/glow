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
#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Exporter/ONNXModelWriter.h"
#include "glow/Graph/Graph.h"
#include "glow/Graph/Nodes.h"
#include "glow/Graph/PlaceholderBindings.h"
#include "glow/Importer/ONNXModelLoader.h"
#include "gtest/gtest.h"

#include "llvm/Support/FileSystem.h"

#ifndef GLOW_DATA_PATH
#define GLOW_DATA_PATH
#endif

using namespace glow;

namespace {
/// Loads model from ONNX format file \p name into glow Function.
/// On success exports glow graph to the output file in "extended" ONNX format,
/// i.e. some glow operators don't have presentation in vanilla ONNX standard.
void testLoadAndSaveONNXModel(const std::string &name, bool zipMode) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  llvm::errs() << "loading model " << name << "\n";

  size_t irVer = 0, opsetVer = 0;

  // Load model from file.
  {
    Error err = Error::empty();
    ONNXModelLoader onnxLD(name, {}, {}, *F, &err);
    irVer = onnxLD.getIrVersion();
    opsetVer = onnxLD.getOpSetVersion();

    if (err) {
      llvm::errs() << "ONNXModelLoader failed to load model: " << name << "\n";
    }
    FAIL_TEST_IF_ERR(std::move(err));
  }

  llvm::SmallString<64> path;
  auto tempFileRes = llvm::sys::fs::createTemporaryFile(
      "exporter", zipMode ? ".output.zip" : ".output.onnxtxt", path);

  EXPECT_EQ(tempFileRes.value(), 0);

  std::string outputFilename(path.c_str());

  // Write model to file.
  {
    Error err = Error::empty();
    ONNXModelWriter onnxWR(outputFilename, *F, irVer, opsetVer, &err, !zipMode,
                           zipMode);

    if (err) {
      llvm::errs() << "ONNXModelWriter failed to write model: " << name
                   << ".\n";
      llvm::sys::fs::remove(outputFilename);
    }
    FAIL_TEST_IF_ERR(std::move(err));
  }

  Function *R = mod.createFunction("reload");

  // Reload model from file.
  {
    Error err = Error::empty();
    ONNXModelLoader onnxLD(outputFilename, {}, {}, *R, &err, zipMode);

    if (err) {
      llvm::errs() << "ONNXModelLoader failed to reload model: "
                   << outputFilename << "\n";
    }
    FAIL_TEST_IF_ERR(std::move(err));
  }
}

bool endsWith(const std::string &full, const std::string &ending) {
  if (full.length() >= ending.length()) {
    return (0 == full.compare(full.length() - ending.length(), ending.length(),
                              ending));
  } else {
    return false;
  }
}
} // namespace

TEST(exporter, onnxModels) {
  std::string inputDirectory(GLOW_DATA_PATH "tests/models/onnxModels");
  std::error_code code;
  for (llvm::sys::fs::directory_iterator dirIt(inputDirectory, code);
       !code && dirIt != llvm::sys::fs::directory_iterator();
       dirIt.increment(code)) {
    auto name = dirIt->path();
    if (!endsWith(name, ".onnxtxt")) {
      llvm::outs() << "Ignore non-onnxtxt input: " << name << "\n";
      continue;
    }
    if (name.find("preluInvalidBroadcastSlope.onnxtxt") != std::string::npos ||
        name.find("padReflect.onnxtxt") != std::string::npos ||
        name.find("gatherConstantFolding.onnxtxt") != std::string::npos ||
        name.find("averagePool3D.onnxtxt") != std::string::npos ||
        name.find("sparseLengthsSum.onnxtxt") != std::string::npos ||
        name.find("constantOfShapeInt32Fail.onnxtxt") != std::string::npos ||
        name.find("padEdge.onnxtxt") != std::string::npos ||
        name.find("castToFloat.onnxtxt") != std::string::npos ||
        name.find("castToFloat16.onnxtxt") != std::string::npos ||
        name.find("castToInt64.onnxtxt") != std::string::npos ||
        name.find("castToInt32.onnxtxt") != std::string::npos ||
        name.find("simpleConvBiasFail.onnxtxt") != std::string::npos ||
        name.find("Where.onnxtxt") != std::string::npos ||
        name.find("constantOfShapeInt64Fail.onnxtxt") != std::string::npos ||
        name.find("ArgMaxDefault.onnxtxt") != std::string::npos ||
        name.find("ArgMaxKeepDim.onnxtxt") != std::string::npos ||
        name.find("ArgMaxNoKeepDim.onnxtxt") != std::string::npos ||
        name.find("NonMaxSuppressionSSD_ONNX.onnxtxt") != std::string::npos ||
        name.find("NonMaxSuppression.onnxtxt") != std::string::npos ||
        name.find("NonMaxSuppressionSSD.onnxtxt") != std::string::npos ||
        name.find("Less.onnxtxt") != std::string::npos) {
      // Ignore invalid ONNX files and graphs without nodes.
      llvm::outs() << "Ignore invalid input files: " << name << "\n";
      continue;
    }
    if (name.find("constant.onnxtxt") != std::string::npos ||
        name.find("shape.onnxtxt") != std::string::npos ||
        name.find("sum1.onnxtxt") != std::string::npos) {
      // Ignore invalid ONNX files and graphs without nodes.
      llvm::outs() << "Ignore empty graph file: " << name << "\n";
      continue;
    }
    if (name.find(".output.onnxtxt") != std::string::npos) {
      // Ignore output files - debugging mode only.
      llvm::outs() << "Ignore output file: " << name << "\n";
      continue;
    }
    // TODO: Debug why these RNN models don`t work!
    if (name.find("rnn") != std::string::npos) {
      // Ignore RNN files.
      llvm::outs() << "Ignore RNN model file: " << name << "\n";
      continue;
    }
    if (name.find("gru") != std::string::npos) {
      // Ignore GRU files.
      llvm::outs() << "Ignore GRU model file: " << name << "\n";
      continue;
    }
    if (name.find("lstm") != std::string::npos) {
      // Ignore LSTM files.
      llvm::outs() << "Ignore LSTM model file: " << name << "\n";
      continue;
    }
    const bool customOnnxDefineSymbol =
        name.find("dimParam.onnxtxt") != std::string::npos;
    if (customOnnxDefineSymbol) {
      setOnnxDefineSymbol({"ONNXUndefinedSymbol,1"});
    }

    // Disable constant folding for these tests.
    setConstantFoldLoaderOpsFlag(false);

    testLoadAndSaveONNXModel(dirIt->path(), /* zipMode */ true);
    testLoadAndSaveONNXModel(dirIt->path(), /* zipMode */ false);

    // Reset the custom symbol used.
    if (customOnnxDefineSymbol) {
      setOnnxDefineSymbol({});
    }
  }
}

TEST(exporter, ChannelwiseQuantizedConvolution) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  auto *F = mod.createFunction("F");

  unsigned_t inChannels = 8;
  unsigned_t inSide = 6;
  unsigned_t batchSize = 8;
  unsigned_t outChannels = 12;
  unsigned_t filterSide = 3;
  unsigned_t groups = 4;

  Placeholder *input = mod.createPlaceholder(
      ElemKind::Int8QTy, {batchSize, inSide, inSide, inChannels}, 1.2, 3,
      "input", /* isTrainable */ false);

  Placeholder *weights = mod.createPlaceholder(
      ElemKind::Int8QTy,
      {outChannels, filterSide, filterSide, inChannels / groups}, 2.5, 1,
      "weights",
      /* isTrainable */ false);

  Placeholder *bias =
      mod.createPlaceholder(ElemKind::FloatTy, {outChannels}, "bias",
                            /* isTrainable */ false);

  Placeholder *scales =
      mod.createPlaceholder(ElemKind::FloatTy, {outChannels}, "scales",
                            /* isTrainable */ false);

  Placeholder *offsets =
      mod.createPlaceholder(ElemKind::Int32ITy, {outChannels}, "offsets",
                            /* isTrainable */ false);

  std::vector<unsigned_t> kernels = {filterSide, filterSide};
  std::vector<unsigned_t> strides = {1, 1};
  std::vector<unsigned_t> pads = {1, 1, 1, 1};

  auto outSize =
      calculateConvPoolOutputDims(inSide, inSide, kernels, strides, pads);
  auto *outTy = mod.uniqueType(
      ElemKind::Int8QTy,
      {batchSize, outSize.first, outSize.second, outChannels}, 3.8, 4);

  auto *cqConv = F->createChannelwiseQuantizedConv(
      "cqconv", input, weights, bias, scales, offsets, outTy, kernels, strides,
      pads, groups);

  auto *save = F->createSave("save_out", cqConv);

  Placeholder *output = save->getPlaceholder();

  ASSERT_TRUE(F->verify());

  PlaceholderBindings bindings;
  bindings.allocate({weights, bias, scales, offsets, input, output});
  convertPlaceholdersToConstants(F, bindings, {input, output});

  llvm::SmallString<64> path;
  auto tempFileRes = llvm::sys::fs::createTemporaryFile(
      "exportChannelwiseQuantizedConvolution", "output.onnxtxt", path);

  EXPECT_EQ(tempFileRes.value(), 0);

  std::string outputFilename(path.c_str());

  // Write model to file.
  {
    size_t irVer = 5, opsetVer = 10;
    Error err = Error::empty();
    ONNXModelWriter onnxWR(outputFilename, *F, irVer, opsetVer, &err,
                           /*textMode*/ true,
                           /*zipMode*/ false);

    if (err) {
      llvm::sys::fs::remove(outputFilename);
    }
    FAIL_TEST_IF_ERR(std::move(err));
  }

  // Load model from file.
  Function *R = mod.createFunction("R");
  {
    Error err = Error::empty();
    ONNXModelLoader onnxLD(outputFilename, {"input"}, {input->getType()}, *R,
                           &err,
                           /*zipMode*/ false);

    if (err) {
      llvm::errs() << "ONNXModelLoader failed to reload model: "
                   << outputFilename << "\n";
    }

    FAIL_TEST_IF_ERR(std::move(err));
  }

  // Verify reloaded function matches the original.
  ASSERT_TRUE(R->verify());

  auto *cqConvReloaded = llvm::dyn_cast<ChannelwiseQuantizedConvolutionNode>(
      R->getNodeByName("cqconv"));

  ASSERT_TRUE(cqConvReloaded != nullptr);

  EXPECT_EQ(cqConvReloaded->getInput().getType(), cqConv->getInput().getType());
  EXPECT_EQ(cqConvReloaded->getFilter().getType(),
            cqConv->getFilter().getType());
  EXPECT_EQ(cqConvReloaded->getBias().getType(), cqConv->getBias().getType());
  EXPECT_EQ(cqConvReloaded->getScales().getType(),
            cqConv->getScales().getType());
  EXPECT_EQ(cqConvReloaded->getOffsets().getType(),
            cqConv->getOffsets().getType());

  EXPECT_EQ(cqConvReloaded->getKernels(), cqConv->getKernels());
  EXPECT_EQ(cqConvReloaded->getStrides(), cqConv->getStrides());
  EXPECT_EQ(cqConvReloaded->getPads(), cqConv->getPads());
  EXPECT_EQ(cqConvReloaded->getGroup(), cqConv->getGroup());
}
