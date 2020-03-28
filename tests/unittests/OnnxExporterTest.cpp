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
/// Given a Function \p F and input names \p inpuTensorNames and input types \p
/// inputTensorTypes, writes the function to file and reads it back using the
/// ONNXModelWriter and ONNXModelReader respectively then \returns the
/// reloaded function. \p useGlowCustomOps is used for determining the format
/// for ONNXModelWriter to write with.
Expected<Function *>
saveAndReloadFunction(Function *F, llvm::ArrayRef<const char *> inpuTensorNames,
                      llvm::ArrayRef<TypeRef> inputTensorTypes,
                      size_t irVer = 5, size_t opsetVer = 10,
                      bool zipMode = false, bool useGlowCustomOps = false) {
  auto &mod = *F->getParent();

  llvm::SmallString<64> path;
  auto tempFileRes = llvm::sys::fs::createTemporaryFile(
      "exporter", zipMode ? "output.zip" : "output.onnxtxt", path);

  RETURN_ERR_IF_NOT(tempFileRes.value() == 0,
                    "Failed to create temp file to write into.");

  std::string outputFilename(path.c_str());

  // Write model to file.
  {
    Error err = Error::empty();
    ONNXModelWriter onnxWR(outputFilename, *F, irVer, opsetVer, &err, !zipMode,
                           zipMode, useGlowCustomOps);

    if (err) {
      llvm::sys::fs::remove(outputFilename);
    }

    RETURN_IF_ERR(std::move(err));
  }

  // Load model from file.
  Function *R = mod.createFunction("R");
  {
    Error err = Error::empty();
    ONNXModelLoader onnxLD(outputFilename, inpuTensorNames, inputTensorTypes,
                           *R, &err, zipMode);

    if (err) {
      llvm::errs() << "ONNXModelLoader failed to reload model: "
                   << outputFilename << "\n";
    }

    RETURN_IF_ERR(std::move(err));
  }

  // Verify reloaded function is valid.
  RETURN_ERR_IF_NOT(R->verify(), "Reloaded function is not valid.");

  return R;
}

/// Loads model from ONNX format file \p name into glow Function.
/// On success exports glow graph to the output file in "extended" ONNX format,
/// i.e. some glow operators don't have presentation in vanilla ONNX standard.
void testLoadAndSaveONNXModel(const std::string &name, bool zipMode) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  llvm::errs() << "loading model " << name << "\n";

  size_t irVer = 0, opsetVer = 0;

  bool useGlowCustomOps = false;

  // Load model from file.
  {
    Error err = Error::empty();
    ONNXModelLoader onnxLD(name, {}, {}, *F, &err);
    irVer = onnxLD.getIrVersion();
    opsetVer = onnxLD.getOpSetVersion();
    useGlowCustomOps = onnxLD.usingGlowCustomOps();

    if (err) {
      llvm::errs() << "ONNXModelLoader failed to load model: " << name << "\n";
    }
    FAIL_TEST_IF_ERR(std::move(err));
  }

  FAIL_TEST_IF_ERR(saveAndReloadFunction(F, {}, {}, irVer, opsetVer, zipMode,
                                         useGlowCustomOps)
                       .takeError());
}

bool endsWith(const std::string &full, const std::string &ending) {
  if (full.length() >= ending.length()) {
    return (0 == full.compare(full.length() - ending.length(), ending.length(),
                              ending));
  } else {
    return false;
  }
}

/// Given a Function \p F, \returns a list of nodes with the Kind \p kind.
std::vector<Node *> getNodesByType(Function *F, Kinded::Kind kind) {
  std::vector<Node *> found;
  for (auto &N : F->getNodes()) {
    if (N.getKind() == kind) {
      found.push_back(&N);
    }
  }
  return found;
}

/// Given a function \p F and a Kind \p type, returns a casted pointer to the
/// single node in F with that kind or an Error if one occurs.
template <typename T>
Expected<T *> getSingleNodeWithKind(Function *F, Kinded::Kind type) {
  auto nodesWithKind = getNodesByType(F, type);

  RETURN_ERR_IF_NOT(nodesWithKind.size() == 1,
                    strFormat("Expected one node with kind %s but found %lu",
                              Kinded::getKindName(type), nodesWithKind.size()));

  T *node = llvm::dyn_cast<T>(nodesWithKind[0]);

  RETURN_ERR_IF_NOT(node != nullptr, "Node is not of expected types");

  return node;
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
        name.find("upsampleOpset7.onnxtxt") != std::string::npos ||
        name.find("upsampleOpset9.onnxtxt") != std::string::npos ||
        name.find("NonMaxSuppressionSSD_ONNX.onnxtxt") != std::string::npos ||
        name.find("NonMaxSuppression.onnxtxt") != std::string::npos ||
        name.find("NonMaxSuppressionSSD.onnxtxt") != std::string::npos ||
        name.find("Less.onnxtxt") != std::string::npos ||
        name.find("simpleConvTranspose.onnxtxt") != std::string::npos ||
        name.find("simpleConvTransposeOutShape.onnxtxt") != std::string::npos ||
        name.find("simpleConvTransposeOutShapeDilation.onnxtxt") !=
            std::string::npos ||
        name.find("simpleConvTransposePads.onnxtxt") != std::string::npos ||
        name.find("simpleConvTransposeAutoPadValid.onnxtxt") !=
            std::string::npos ||
        name.find("simpleConvTransposeOutShapeSameUpper.onnxtxt") !=
            std::string::npos ||
        name.find("simpleConvTransposeAutoPadSameLower.onnxtxt") !=
            std::string::npos ||
        name.find("simpleConvTransposeAutoPadSameUpper.onnxtxt") !=
            std::string::npos) {
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

  Constant *biasConstant =
      mod.createConstant(ElemKind::FloatTy, {outChannels}, "bias");

  Constant *scalesConstant =
      mod.createConstant(ElemKind::FloatTy, {outChannels}, "scales");

  Constant *offsetsConstant =
      mod.createConstant(ElemKind::Int32ITy, {outChannels}, "offsets");

  Constant *weightsConstant = mod.createConstant(
      ElemKind::Int8QTy,
      {outChannels, filterSide, filterSide, inChannels / groups}, 2.5, 1,
      "offsets");

  std::vector<unsigned_t> kernels = {filterSide, filterSide};
  std::vector<unsigned_t> strides = {1, 1};
  std::vector<unsigned_t> pads = {1, 1, 1, 1};

  auto outSize =
      calculateConvPoolOutputDims(inSide, inSide, kernels, strides, pads);
  auto *outTy = mod.uniqueType(
      ElemKind::Int8QTy,
      {batchSize, outSize.first, outSize.second, outChannels}, 3.8, 4);

  auto *cqConv = F->createChannelwiseQuantizedConv(
      "cqconv", input, weightsConstant, biasConstant, scalesConstant,
      offsetsConstant, outTy, kernels, strides, pads, groups);

  auto *save = F->createSave("save_out", cqConv);

  Placeholder *output = save->getPlaceholder();

  ASSERT_TRUE(F->verify());

  PlaceholderBindings bindings;
  bindings.allocate({input, output});

  // Save and reload F.
  Function *R;
  ASSIGN_VALUE_OR_FAIL_TEST(
      R, saveAndReloadFunction(F, {"input"}, {input->getType()}));

  ChannelwiseQuantizedConvolutionNode *cqConvReloaded;
  ASSIGN_VALUE_OR_FAIL_TEST(
      cqConvReloaded,
      getSingleNodeWithKind<ChannelwiseQuantizedConvolutionNode>(
          R, Kinded::Kind::ChannelwiseQuantizedConvolutionNodeKind));

  EXPECT_EQ(cqConvReloaded->getInput().getType(), cqConv->getInput().getType());
  EXPECT_EQ(cqConvReloaded->getResult().getType(),
            cqConv->getResult().getType());

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

TEST(exporter, QuantizedConvolution) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  auto *F = mod.createFunction("F");

  unsigned_t inChannels = 8;
  unsigned_t inSide = 6;
  unsigned_t batchSize = 8;
  unsigned_t outChannels = 12;
  unsigned_t filterSide = 3;
  unsigned_t groups = 4;
  unsigned_t dilation = 1;

  Placeholder *input = mod.createPlaceholder(
      ElemKind::Int8QTy, {batchSize, inSide, inSide, inChannels}, 1.2, 3,
      "input", /* isTrainable */ false);

  Placeholder *weights = mod.createPlaceholder(
      ElemKind::Int8QTy,
      {outChannels, filterSide, filterSide, inChannels / groups}, 2.5, 1,
      "weights",
      /* isTrainable */ false);

  Placeholder *bias =
      mod.createPlaceholder(ElemKind::Int32QTy, {outChannels}, 0.25, 2, "bias",
                            /* isTrainable */ false);

  std::vector<unsigned_t> kernels = {filterSide, filterSide};
  std::vector<unsigned_t> strides = {1, 1};
  std::vector<unsigned_t> pads = {1, 1, 1, 1};

  auto outSize =
      calculateConvPoolOutputDims(inSide, inSide, kernels, strides, pads);
  auto *outTy = mod.uniqueType(
      ElemKind::Int8QTy,
      {batchSize, outSize.first, outSize.second, outChannels}, 3.8, 4);

  auto *qConv = F->createConv("qconv", input, weights, bias, outTy, kernels,
                              strides, pads, groups, dilation);

  auto *save = F->createSave("save_out", qConv);

  Placeholder *output = save->getPlaceholder();

  ASSERT_TRUE(F->verify());

  PlaceholderBindings bindings;
  bindings.allocate({weights, bias, input, output});
  convertPlaceholdersToConstants(F, bindings, {input, output});

  // Save and reload F.
  Function *R;
  ASSIGN_VALUE_OR_FAIL_TEST(
      R, saveAndReloadFunction(F, {"input"}, {input->getType()}));

  // Verify reloaded function matches the original.
  ConvolutionNode *qConvReloaded;
  ASSIGN_VALUE_OR_FAIL_TEST(qConvReloaded,
                            getSingleNodeWithKind<ConvolutionNode>(
                                R, Kinded::Kind::ConvolutionNodeKind));

  EXPECT_EQ(qConvReloaded->getInput().getType(), qConv->getInput().getType());
  EXPECT_EQ(qConvReloaded->getResult().getType(), qConv->getResult().getType());

  EXPECT_EQ(qConvReloaded->getFilter().getType(), qConv->getFilter().getType());
  EXPECT_EQ(qConvReloaded->getBias().getType(), qConv->getBias().getType());

  EXPECT_EQ(qConvReloaded->getKernels(), qConv->getKernels());
  EXPECT_EQ(qConvReloaded->getStrides(), qConv->getStrides());
  EXPECT_EQ(qConvReloaded->getPads(), qConv->getPads());
  EXPECT_EQ(qConvReloaded->getGroup(), qConv->getGroup());
  EXPECT_EQ(qConvReloaded->getDilation(), qConv->getDilation());
}

TEST(exporter, QuantizedMaxPool) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  auto *F = mod.createFunction("F");

  unsigned_t inChannels = 8;
  unsigned_t inSide = 6;
  unsigned_t batchSize = 8;
  unsigned_t filterSide = 3;

  Placeholder *input = mod.createPlaceholder(
      ElemKind::Int8QTy, {batchSize, inSide, inSide, inChannels}, 1.2, 3,
      "input", /* isTrainable */ false);

  std::vector<unsigned_t> kernels = {filterSide, filterSide};
  std::vector<unsigned_t> strides = {1, 1};
  std::vector<unsigned_t> pads = {1, 1, 1, 1};

  auto *maxPool = F->createMaxPool("maxpool", input, kernels, strides, pads);

  auto *save = F->createSave("save_out", maxPool->getNthResult(0));

  Placeholder *output = save->getPlaceholder();

  ASSERT_TRUE(F->verify());

  PlaceholderBindings bindings;
  bindings.allocate({input, output});
  convertPlaceholdersToConstants(F, bindings, {input, output});

  // Save and reload F.
  Function *R;
  ASSIGN_VALUE_OR_FAIL_TEST(
      R, saveAndReloadFunction(F, {"input"}, {input->getType()}));

  // Verify reloaded function matches the original.
  MaxPoolNode *maxPoolReloaded;
  ASSIGN_VALUE_OR_FAIL_TEST(
      maxPoolReloaded,
      getSingleNodeWithKind<MaxPoolNode>(R, Kinded::Kind::MaxPoolNodeKind));

  EXPECT_EQ(maxPoolReloaded->getInput().getType(),
            maxPool->getInput().getType());
  EXPECT_EQ(maxPoolReloaded->getResult().getType(),
            maxPool->getResult().getType());

  EXPECT_EQ(maxPoolReloaded->getKernels(), maxPool->getKernels());
  EXPECT_EQ(maxPoolReloaded->getStrides(), maxPool->getStrides());
  EXPECT_EQ(maxPoolReloaded->getPads(), maxPool->getPads());
}

TEST(exporter, QuantizedAvgPool) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  auto *F = mod.createFunction("F");

  unsigned_t inChannels = 8;
  unsigned_t inSide = 6;
  unsigned_t batchSize = 8;
  unsigned_t filterSide = 3;

  Placeholder *input = mod.createPlaceholder(
      ElemKind::Int8QTy, {batchSize, inSide, inSide, inChannels}, 1.2, 3,
      "input", /* isTrainable */ false);

  std::vector<unsigned_t> kernels = {filterSide, filterSide};
  std::vector<unsigned_t> strides = {1, 1};
  std::vector<unsigned_t> pads = {1, 1, 1, 1};

  auto *avgPool = F->createAvgPool("avgpool", input, kernels, strides, pads);

  auto *save = F->createSave("save_out", avgPool->getNthResult(0));

  Placeholder *output = save->getPlaceholder();

  ASSERT_TRUE(F->verify());

  PlaceholderBindings bindings;
  bindings.allocate({input, output});
  convertPlaceholdersToConstants(F, bindings, {input, output});

  // Save and reload F.
  Function *R;
  ASSIGN_VALUE_OR_FAIL_TEST(
      R, saveAndReloadFunction(F, {"input"}, {input->getType()}));

  // Verify reloaded function matches the original.
  AvgPoolNode *avgPoolReloaded;
  ASSIGN_VALUE_OR_FAIL_TEST(
      avgPoolReloaded,
      getSingleNodeWithKind<AvgPoolNode>(R, Kinded::Kind::AvgPoolNodeKind));

  EXPECT_EQ(avgPoolReloaded->getInput().getType(),
            avgPool->getInput().getType());
  EXPECT_EQ(avgPoolReloaded->getResult().getType(),
            avgPool->getResult().getType());

  EXPECT_EQ(avgPoolReloaded->getKernels(), avgPool->getKernels());
  EXPECT_EQ(avgPoolReloaded->getStrides(), avgPool->getStrides());
  EXPECT_EQ(avgPoolReloaded->getPads(), avgPool->getPads());
}

TEST(exporter, QuantizedAdaptiveAvgPool) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  auto *F = mod.createFunction("F");

  unsigned_t inChannels = 8;
  unsigned_t inSide = 6;
  unsigned_t batchSize = 8;

  Placeholder *input = mod.createPlaceholder(
      ElemKind::Int8QTy, {batchSize, inSide, inSide, inChannels}, 1.2, 3,
      "input", /* isTrainable */ false);

  auto *outTy = mod.uniqueTypeWithNewShape(input->getType(),
                                           {batchSize, 3, 3, inChannels});

  auto *adaptiveAvgPool =
      F->createAdaptiveAvgPool("adaptive_avgpool", input, outTy);

  auto *save = F->createSave("save_out", adaptiveAvgPool->getNthResult(0));

  Placeholder *output = save->getPlaceholder();

  ASSERT_TRUE(F->verify());

  PlaceholderBindings bindings;
  bindings.allocate({input, output});
  convertPlaceholdersToConstants(F, bindings, {input, output});

  // Save and reload F.
  Function *R;
  ASSIGN_VALUE_OR_FAIL_TEST(
      R, saveAndReloadFunction(F, {"input"}, {input->getType()}));

  // Verify reloaded function matches the original.
  AdaptiveAvgPoolNode *adaptiveAvgPoolReloaded;
  ASSIGN_VALUE_OR_FAIL_TEST(adaptiveAvgPoolReloaded,
                            getSingleNodeWithKind<AdaptiveAvgPoolNode>(
                                R, Kinded::Kind::AdaptiveAvgPoolNodeKind));

  EXPECT_EQ(adaptiveAvgPoolReloaded->getInput().getType(),
            adaptiveAvgPool->getInput().getType());
  EXPECT_EQ(adaptiveAvgPoolReloaded->getResult().getType(),
            adaptiveAvgPool->getResult().getType());
}

TEST(exporter, RowwiseQuantizedFullyConnected) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  auto *F = mod.createFunction("F");

  Placeholder *input = mod.createPlaceholder(
      ElemKind::Int8QTy, {2, 100}, 1.2, 3, "input", /* isTrainable */ false);

  Constant *weightsConstant =
      mod.createConstant(ElemKind::Int8QTy, {10, 100}, 1.0, 0, "weights");

  Constant *biasConstant =
      mod.createConstant(ElemKind::Int32QTy, {10}, 1.0, 0, "bias");

  Constant *scalesConstant =
      mod.createConstant(ElemKind::FloatTy, {10}, "scales");

  Constant *offsetsConstant =
      mod.createConstant(ElemKind::Int32ITy, {10}, "offsets");

  auto *outTy = mod.uniqueType(ElemKind::Int8QTy, {2, 10}, 3.8, 4);

  auto *rwqFC = F->createRowwiseQuantizedFullyConnected(
      "rwqFC", input, weightsConstant, scalesConstant, offsetsConstant,
      biasConstant, outTy);

  auto *save = F->createSave("save_out", rwqFC);

  Placeholder *output = save->getPlaceholder();

  ASSERT_TRUE(F->verify());

  PlaceholderBindings bindings;
  bindings.allocate({input, output});

  // Save and reload F.
  Function *R;
  ASSIGN_VALUE_OR_FAIL_TEST(
      R, saveAndReloadFunction(F, {"input"}, {input->getType()}));

  RowwiseQuantizedFullyConnectedNode *rwqFCReloaded;
  ASSIGN_VALUE_OR_FAIL_TEST(
      rwqFCReloaded,
      getSingleNodeWithKind<RowwiseQuantizedFullyConnectedNode>(
          R, Kinded::Kind::RowwiseQuantizedFullyConnectedNodeKind));

  EXPECT_EQ(rwqFCReloaded->getInput().getType(), rwqFC->getInput().getType());
  EXPECT_EQ(rwqFCReloaded->getResult().getType(), rwqFC->getResult().getType());

  EXPECT_EQ(rwqFCReloaded->getWeights().getType(),
            rwqFC->getWeights().getType());
  EXPECT_EQ(rwqFCReloaded->getBias().getType(), rwqFC->getBias().getType());
  EXPECT_EQ(rwqFCReloaded->getScales().getType(), rwqFC->getScales().getType());
  EXPECT_EQ(rwqFCReloaded->getOffsets().getType(),
            rwqFC->getOffsets().getType());
}
