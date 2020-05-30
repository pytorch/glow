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
/// Given a Function \p F and input names \p inputTensorNames and input types \p
/// inputTensorTypes, writes the function to file and reads it back using the
/// ONNXModelWriter and ONNXModelReader respectively then \returns the
/// reloaded function. \p useGlowCustomOps is used for determining the format
/// for ONNXModelWriter to write with.
Expected<Function *>
saveAndReloadFunction(Module &reloadMod, Function *F,
                      llvm::ArrayRef<const char *> inputTensorNames,
                      llvm::ArrayRef<TypeRef> inputTensorTypes,
                      size_t irVer = 5, size_t opsetVer = 10,
                      bool zipMode = false, bool useGlowCustomOps = false,
                      bool includeConstantData = true,
                      ConstantFoldingRecordMap *constFoldRecord = nullptr) {
  llvm::SmallString<64> path;
  auto tempFileRes = llvm::sys::fs::createTemporaryFile(
      "exporter", zipMode ? "output.zip" : "output.onnxtxt", path);

  RETURN_ERR_IF_NOT(tempFileRes.value() == 0,
                    "Failed to create temp file to write into.");

  std::string outputFilename(path.c_str());
  ScopeGuard cleanup([&]() { llvm::sys::fs::remove(outputFilename); });

  // Write model to file.
  {
    Error err = Error::empty();
    ONNXModelWriter onnxWR(outputFilename, *F, irVer, opsetVer, &err, !zipMode,
                           zipMode, useGlowCustomOps, includeConstantData,
                           constFoldRecord ? *constFoldRecord
                                           : ConstantFoldingRecordMap());

    if (err) {
      llvm::sys::fs::remove(outputFilename);
    }

    RETURN_IF_ERR(std::move(err));
  }

  Function *R = nullptr;
  Module &origMod = *F->getParent();
  if (!includeConstantData) {
    R = reloadMod.getFunction(F->getName());
    RETURN_ERR_IF_NOT(R, "Did not find Function to reload into.");
    R->clear();
    if (constFoldRecord) {
      // Additionally remove the original Constants that we first folded, so
      // that when we reload below we can recreate them.
      std::unordered_set<Function *> funsToDelete;
      for (auto &pair : *constFoldRecord) {
        Function *origF = pair.second->getParent();
        funsToDelete.insert(origF);
        Constant *C = reloadMod.getConstantByName(pair.first->getName());
        RETURN_ERR_IF_NOT(C, "Did not find constant that was initially folded");
        reloadMod.eraseConstant(C);
      }
      for (Function *origF : funsToDelete) {
        Function *reloadConstFoldF = reloadMod.getFunction(origF->getName());
        RETURN_ERR_IF_NOT(reloadConstFoldF,
                          "Did not find const folding function reloaded");
        reloadMod.eraseFunction(reloadConstFoldF);
        origMod.eraseFunction(origF);
      }
    }
  } else {
    R = reloadMod.createFunction("R");
  }

  // Load model from file.
  {
    Error err = Error::empty();
    ONNXModelLoader onnxLD(outputFilename, inputTensorNames, inputTensorTypes,
                           *R, &err, zipMode, /* perNodeOpts */ nullptr,
                           /* disableConstFoldInLoader */ true,
                           /* loadIntoExistingModule */ !includeConstantData);

    if (err) {
      llvm::errs() << "ONNXModelLoader failed to reload model: "
                   << outputFilename << "\n";
    }

    RETURN_IF_ERR(std::move(err));
  }

  // Verify reloaded function is valid.
  RETURN_ERR_IF_NOT(R->verify(), "Reloaded function is not valid.");

  // Verify that the Constants from the original Module have the same data as
  // those in the reloaded module.
  if (constFoldRecord) {
    deleteUnusedConstants(reloadMod);
    deleteUnusedConstants(origMod);
    for (Constant *newC : reloadMod.getConstants()) {
      Constant *origC = R->getParent()->getConstantByName(newC->getName());
      RETURN_ERR_IF_NOT(origC,
                        strFormat("Expected original Constant by name %s",
                                  newC->getName().data()));
      RETURN_ERR_IF_NOT(newC->getPayload().isBitwiseEqual(origC->getPayload()),
                        strFormat("Mismatch on Constants of name %s",
                                  newC->getName().data()));
    }
  }

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

  Module reloadMod;
  FAIL_TEST_IF_ERR(saveAndReloadFunction(reloadMod, F, {}, {}, irVer, opsetVer,
                                         zipMode, useGlowCustomOps)
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

/// Use to test constant folding exporting and reloading tests, where some
/// constant folding is recorded and serialized in the custom Glow ONNX model.
class ConstFoldReloadTest : public ::testing::Test {
public:
  ConstFoldReloadTest() : EE_("Interpreter"), mod_(EE_.getModule()) {
    F_ = mod_.createFunction("main");
  }

protected:
  ExecutionEngine EE_;
  Module &mod_;
  Function *F_;
  PlaceholderBindings bindings_;
  CompilationContext cctx_;

  /// Constant folds \ref F_ and then serializes it. Then deserializes it and
  /// runs it and makes sure that running the original and the reloaded Function
  /// are bitwise equal. Verifies that \p numExpectedConstsFolded constant
  /// folding records are created during constant folding/recording.
  void serializeAndReloadAndCompareResults(unsigned numExpectedConstsFolded) {
    bindings_.allocate(mod_.getPlaceholders());

    // Perform constant folding, recording what occurs so we can serialize it.
    ConstantFoldingRecordMap record = constantFoldAndRecord(F_, cctx_);
    EXPECT_EQ(record.size(), numExpectedConstsFolded);
    runDCEPass(F_, cctx_);

    // Clone the original module into a new module used for reloading the model.
    ExecutionEngine reloadEE(EE_.getBackendName());
    Module &reloadMod = reloadEE.getModule();
    mod_.clone(&reloadMod);

    // Save and reload F.
    Function *reloadF_;
    ASSIGN_VALUE_OR_FAIL_TEST(
        reloadF_,
        saveAndReloadFunction(reloadMod, F_, {}, {}, 7, 9,
                              /* zipMode */ false,
                              /* useGlowCustomOps */ true,
                              /* includeConstantData */ false, &record));

    // Verify that the Function and its Module are the same before and after.
    EXPECT_EQ(reloadF_->toString(), F_->toString());
    EXPECT_EQ(reloadMod.toString(), mod_.toString());

    PlaceholderBindings reloadBindings =
        bindings_.clone(reloadMod.getPlaceholders());

    // Now run both to check they have bitwise equal results.
    EE_.compile(cctx_);
    EE_.run(bindings_);

    CompilationContext reloadCctx;
    reloadEE.compile(reloadCctx);
    reloadEE.run(reloadBindings);

    EXPECT_TRUE(
        PlaceholderBindings::compare(&bindings_, &reloadBindings, 0.0f));
  }
};

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
        name.find("resizeNearest.onnxtxt") != std::string::npos ||
        name.find("resizeNearestV11compat.onnxtxt") != std::string::npos ||
        name.find("resizeNearestV11compat_sizes.onnxtxt") !=
            std::string::npos ||
        name.find("resizeBilinear.onnxtxt") != std::string::npos ||
        name.find("resizeBilinearV11compat.onnxtxt") != std::string::npos ||
        name.find("resizeBilinearV11compat_sizes.onnxtxt") !=
            std::string::npos ||
        name.find("upsampleOpset9.onnxtxt") != std::string::npos ||
        name.find("NonMaxSuppressionSSD_ONNX.onnxtxt") != std::string::npos ||
        name.find("NonMaxSuppression.onnxtxt") != std::string::npos ||
        name.find("NonMaxSuppressionSSD.onnxtxt") != std::string::npos ||
        name.find("Less.onnxtxt") != std::string::npos ||
        name.find("simpleConvTranspose.onnxtxt") != std::string::npos ||
        name.find("simpleConvTransposeOutShape.onnxtxt") != std::string::npos ||
        name.find("simpleConvTransposeOutShapeDilation.onnxtxt") !=
            std::string::npos ||
        name.find("NonZero.onnxtxt") != std::string::npos ||
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
  unsigned_t dilation = 1;

  Placeholder *input = mod.createPlaceholder(
      ElemKind::Int8QTy, {batchSize, inSide, inSide, inChannels}, 1.2, 3,
      "input", /* isTrainable */ false);

  Constant *biasConstant =
      mod.createConstant(ElemKind::FloatTy, {outChannels}, "bias");
  biasConstant->getPayloadMutable().getHandle<float>().randomize(-0.1, 0.1,
                                                                 mod.getPRNG());

  Constant *filterScalesConstant =
      mod.createConstant(ElemKind::FloatTy, {outChannels}, "filter_scales");

  Constant *filterOffsetsConstant =
      mod.createConstant(ElemKind::Int32ITy, {outChannels}, "filter_offsets");

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
      "cqconv", input, weightsConstant, biasConstant, filterScalesConstant,
      filterOffsetsConstant, /* biasScales */ nullptr,
      /* biasOffsets */ nullptr, outTy, kernels, strides, pads, groups,
      dilation);

  auto *save = F->createSave("save_out", cqConv);

  Placeholder *output = save->getPlaceholder();

  ASSERT_TRUE(F->verify());

  PlaceholderBindings bindings;
  bindings.allocate({input, output});

  // Save and reload F.
  Function *R;
  Module reloadMod;
  ASSIGN_VALUE_OR_FAIL_TEST(
      R, saveAndReloadFunction(reloadMod, F, {"input"}, {input->getType()}));

  ChannelwiseQuantizedConvolutionNode *cqConvReloaded;
  ASSIGN_VALUE_OR_FAIL_TEST(
      cqConvReloaded,
      getSingleNodeWithKind<ChannelwiseQuantizedConvolutionNode>(
          R, Kinded::Kind::ChannelwiseQuantizedConvolutionNodeKind));

  EXPECT_TRUE(cqConvReloaded->getInput().getType()->isEqual(
      *cqConv->getInput().getType()));
  EXPECT_TRUE(cqConvReloaded->getResult().getType()->isEqual(
      *cqConv->getResult().getType()));

  EXPECT_TRUE(cqConvReloaded->getFilter().getType()->isEqual(
      *cqConv->getFilter().getType()));
  EXPECT_TRUE(cqConvReloaded->getBias().getType()->isEqual(
      *cqConv->getBias().getType()));
  EXPECT_TRUE(cqConvReloaded->getFilterScales().getType()->isEqual(
      *cqConv->getFilterScales().getType()));
  EXPECT_TRUE(cqConvReloaded->getFilterOffsets().getType()->isEqual(
      *cqConv->getFilterOffsets().getType()));
  EXPECT_TRUE(cqConvReloaded->getBiasScales().getType()->isEqual(
      *cqConv->getBiasScales().getType()));
  EXPECT_TRUE(cqConvReloaded->getBiasOffsets().getType()->isEqual(
      *cqConv->getBiasOffsets().getType()));

  EXPECT_EQ(cqConvReloaded->getKernels(), cqConv->getKernels());
  EXPECT_EQ(cqConvReloaded->getStrides(), cqConv->getStrides());
  EXPECT_EQ(cqConvReloaded->getPads(), cqConv->getPads());
  EXPECT_EQ(cqConvReloaded->getGroup(), cqConv->getGroup());
  EXPECT_EQ(cqConvReloaded->getDilation(), cqConv->getDilation());
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
  Module reloadMod;
  ASSIGN_VALUE_OR_FAIL_TEST(
      R, saveAndReloadFunction(reloadMod, F, {"input"}, {input->getType()}));

  // Verify reloaded function matches the original.
  ConvolutionNode *qConvReloaded;
  ASSIGN_VALUE_OR_FAIL_TEST(qConvReloaded,
                            getSingleNodeWithKind<ConvolutionNode>(
                                R, Kinded::Kind::ConvolutionNodeKind));

  EXPECT_TRUE(qConvReloaded->getInput().getType()->isEqual(
      *qConv->getInput().getType()));
  EXPECT_TRUE(qConvReloaded->getResult().getType()->isEqual(
      *qConv->getResult().getType()));

  EXPECT_TRUE(qConvReloaded->getFilter().getType()->isEqual(
      *qConv->getFilter().getType()));
  EXPECT_TRUE(
      qConvReloaded->getBias().getType()->isEqual(*qConv->getBias().getType()));

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
  Module reloadMod;
  ASSIGN_VALUE_OR_FAIL_TEST(
      R, saveAndReloadFunction(reloadMod, F, {"input"}, {input->getType()}));

  // Verify reloaded function matches the original.
  MaxPoolNode *maxPoolReloaded;
  ASSIGN_VALUE_OR_FAIL_TEST(
      maxPoolReloaded,
      getSingleNodeWithKind<MaxPoolNode>(R, Kinded::Kind::MaxPoolNodeKind));

  EXPECT_TRUE(maxPoolReloaded->getInput().getType()->isEqual(
      *maxPool->getInput().getType()));
  EXPECT_TRUE(maxPoolReloaded->getResult().getType()->isEqual(
      *maxPool->getResult().getType()));

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
  Module reloadMod;
  ASSIGN_VALUE_OR_FAIL_TEST(
      R, saveAndReloadFunction(reloadMod, F, {"input"}, {input->getType()}));

  // Verify reloaded function matches the original.
  AvgPoolNode *avgPoolReloaded;
  ASSIGN_VALUE_OR_FAIL_TEST(
      avgPoolReloaded,
      getSingleNodeWithKind<AvgPoolNode>(R, Kinded::Kind::AvgPoolNodeKind));

  EXPECT_TRUE(avgPoolReloaded->getInput().getType()->isEqual(
      *avgPool->getInput().getType()));
  EXPECT_TRUE(avgPoolReloaded->getResult().getType()->isEqual(
      *avgPool->getResult().getType()));

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
  Module reloadMod;
  ASSIGN_VALUE_OR_FAIL_TEST(
      R, saveAndReloadFunction(reloadMod, F, {"input"}, {input->getType()}));

  // Verify reloaded function matches the original.
  AdaptiveAvgPoolNode *adaptiveAvgPoolReloaded;
  ASSIGN_VALUE_OR_FAIL_TEST(adaptiveAvgPoolReloaded,
                            getSingleNodeWithKind<AdaptiveAvgPoolNode>(
                                R, Kinded::Kind::AdaptiveAvgPoolNodeKind));

  EXPECT_TRUE(adaptiveAvgPoolReloaded->getInput().getType()->isEqual(
      *adaptiveAvgPool->getInput().getType()));
  EXPECT_TRUE(adaptiveAvgPoolReloaded->getResult().getType()->isEqual(
      *adaptiveAvgPool->getResult().getType()));
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
  Module reloadMod;
  ASSIGN_VALUE_OR_FAIL_TEST(
      R, saveAndReloadFunction(reloadMod, F, {"input"}, {input->getType()}));

  RowwiseQuantizedFullyConnectedNode *rwqFCReloaded;
  ASSIGN_VALUE_OR_FAIL_TEST(
      rwqFCReloaded,
      getSingleNodeWithKind<RowwiseQuantizedFullyConnectedNode>(
          R, Kinded::Kind::RowwiseQuantizedFullyConnectedNodeKind));

  EXPECT_TRUE(rwqFCReloaded->getInput().getType()->isEqual(
      *rwqFC->getInput().getType()));
  EXPECT_TRUE(rwqFCReloaded->getResult().getType()->isEqual(
      *rwqFC->getResult().getType()));

  EXPECT_TRUE(rwqFCReloaded->getWeights().getType()->isEqual(
      *rwqFC->getWeights().getType()));
  EXPECT_TRUE(
      rwqFCReloaded->getBias().getType()->isEqual(*rwqFC->getBias().getType()));
  EXPECT_TRUE(rwqFCReloaded->getScales().getType()->isEqual(
      *rwqFC->getScales().getType()));
  EXPECT_TRUE(rwqFCReloaded->getOffsets().getType()->isEqual(
      *rwqFC->getOffsets().getType()));
}

TEST_F(ConstFoldReloadTest, exportGraphWithOneConstFoldingRecord) {
  Placeholder *I =
      mod_.createPlaceholder(ElemKind::Float16Ty, {2, 100}, "input",
                             /* isTrainable */ false);
  Constant *W = mod_.createConstant(ElemKind::FloatTy, {10, 100}, "weight");
  ClipNode *clipW = F_->createClip("clip", W, -5.f, 5.f);
  ConvertToNode *convertW =
      F_->createConvertTo("conv", clipW, ElemKind::Float16Ty);
  TransposeNode *transposeW =
      F_->createTranspose("transpose", convertW, {1, 0});
  MatMulNode *MM = F_->createMatMul("matmul", I, transposeW);
  F_->createSave("save", MM);

  bindings_.allocate(I)->getHandle<float16_t>().randomize(-10, 10,
                                                          mod_.getPRNG());
  W->getPayloadMutable().getHandle<float>().randomize(-10, 10, mod_.getPRNG());

  serializeAndReloadAndCompareResults(1);
}

TEST_F(ConstFoldReloadTest, exportGraphWithTwoConstFoldingRecords) {
  Placeholder *I =
      mod_.createPlaceholder(ElemKind::Float16Ty, {2, 100}, "input",
                             /* isTrainable */ false);
  Constant *W = mod_.createConstant(ElemKind::FloatTy, {10, 100}, "weight");
  ClipNode *clipW = F_->createClip("clip", W, -5.f, 5.f);
  ConvertToNode *convertW =
      F_->createConvertTo("conv", clipW, ElemKind::Float16Ty);
  TransposeNode *transposeW =
      F_->createTranspose("transpose", convertW, {1, 0});
  MatMulNode *MM = F_->createMatMul("matmul", I, transposeW);
  F_->createSave("save_mm", MM);

  Constant *W2 = mod_.createConstant(ElemKind::Float16Ty, {2, 100}, "weight2");
  TanhNode *tanhW = F_->createTanh("tanh", W2);
  AddNode *add = F_->createAdd("add", tanhW, I);
  F_->createSave("save_add", add);

  bindings_.allocate(I)->getHandle<float16_t>().randomize(-10, 10,
                                                          mod_.getPRNG());
  W->getPayloadMutable().getHandle<float>().randomize(-10, 10, mod_.getPRNG());
  W2->getPayloadMutable().getHandle<float16_t>().randomize(-10, 10,
                                                           mod_.getPRNG());

  serializeAndReloadAndCompareResults(2);
}

TEST_F(ConstFoldReloadTest, exportGraphWithTwoConstFoldingMultiOutputRecord) {
  Constant *W = mod_.createConstant(ElemKind::FloatTy, {100}, "weight");
  SigmoidNode *sigmoidW = F_->createSigmoid("sig", W);
  ConvertToNode *convertW =
      F_->createConvertTo("conv", sigmoidW, ElemKind::Float16Ty);
  TopKNode *TK = F_->createTopK("topk", convertW, 5);
  F_->createSave("save_indices", TK->getIndices());

  Placeholder *I = mod_.createPlaceholder(ElemKind::Float16Ty, {5}, "input",
                                          /* isTrainable */ false);
  AddNode *add = F_->createAdd("add", I, TK->getValues());
  F_->createSave("save_add", add);

  bindings_.allocate(I)->getHandle<float16_t>().randomize(-10, 10,
                                                          mod_.getPRNG());
  W->getPayloadMutable().getHandle<float>().randomize(-10, 10, mod_.getPRNG());

  serializeAndReloadAndCompareResults(2);
}
