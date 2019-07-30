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
#include "glow/ExecutionEngine/ExecutionEngine2.h"
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
void testLoadAndSaveONNXModel(const std::string &name) {
  ExecutionEngine2 EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  llvm::Error err = llvm::Error::success();
  ONNXModelLoader onnxLD(name, {}, {}, *F, &err);

  ASSERT_FALSE(handleErrors(std::move(err), [&name](const GlowErr &GE) {
    llvm::errs() << "ONNXModelLoader failed to load model: " << name << ": ";
    GE.log(llvm::errs());
    llvm::errs() << "\n";
  }));

  std::string outputFilename(name + ".output.onnxtxt");
  { ONNXModelWriter onnxWR(outputFilename, *F, 1, 1, &err, true); }
  llvm::sys::fs::remove(outputFilename);
  EXPECT_FALSE(err) << "file name: " << name;
}
} // namespace

TEST(exporter, onnxModels) {
  std::string inputDirectory(GLOW_DATA_PATH "tests/models/onnxModels");
  std::error_code code;
  llvm::sys::fs::directory_iterator dirIt(inputDirectory, code);
  for (llvm::sys::fs::directory_iterator dirIt(inputDirectory, code);
       !code && dirIt != llvm::sys::fs::directory_iterator();
       dirIt.increment(code)) {
    auto name = dirIt->path();
    if (name.find("preluInvalidBroadcastSlope.onnxtxt") != std::string::npos) {
      continue;
    }
    testLoadAndSaveONNXModel(dirIt->path());
  }
}
