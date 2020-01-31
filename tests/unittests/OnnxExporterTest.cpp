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
  Error err = Error::empty();
  {
    ONNXModelLoader onnxLD(name, {}, {}, *F, &err);
    irVer = onnxLD.getIrVersion();
    opsetVer = onnxLD.getOpSetVersion();
  }

  if (ERR_TO_BOOL(std::move(err))) {
    llvm::errs() << "ONNXModelLoader failed to load model: " << name << ": ";
    FAIL();
  }

  llvm::SmallString<64> path;
  auto tempFileRes = llvm::sys::fs::createTemporaryFile(
      "exporter", zipMode ? ".output.zip" : ".output.onnxtxt", path);

  EXPECT_EQ(tempFileRes.value(), 0);

  std::string outputFilename(path.c_str());
  err = Error::empty();
  {
    ONNXModelWriter onnxWR(outputFilename, *F, irVer, opsetVer, &err, !zipMode,
                           zipMode);
  }

  if (ERR_TO_BOOL(std::move(err))) {
    llvm::errs() << "ONNXModelWriter failed to write model: " << name << ".\n";
    llvm::sys::fs::remove(outputFilename);
    FAIL();
  }

  Function *R = mod.createFunction("reload");
  err = Error::empty();
  { ONNXModelLoader onnxLD(outputFilename, {}, {}, *R, &err, zipMode); }
  // llvm::sys::fs::remove(outputFilename);
  EXPECT_FALSE(ERR_TO_BOOL(std::move(err)))
      << "ONNXModelLoader failed to reload model: " << outputFilename;
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
