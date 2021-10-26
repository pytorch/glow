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
/// for ONNXModelWriter to write with. \p useString is used to use strings
/// rather than files for reading and writing functions.
Expected<Function *> saveAndReloadFunction(
    Module &reloadMod, Function *F,
    llvm::ArrayRef<const char *> inputTensorNames,
    llvm::ArrayRef<TypeRef> inputTensorTypes, size_t irVer = 5,
    size_t opsetVer = 10, bool zipMode = false, bool useGlowCustomOps = false,
    bool useString = false, bool includeConstantData = true,
    ConstantFoldingRecordMap *constFoldRecord = nullptr,
    CompilationContext *reloadCctx = nullptr,
    const BackendSpecificNodeInfo &backendSpecificNodeInfo = {},
    const OriginNameToTQPMap &originNameToTQPMap = {}) {
  std::string outputString;
  std::string outputFilename = zipMode ? "output.zip" : "output.onnxtxt";

  if (!useString) {
    llvm::SmallString<64> path;

    auto tempFileRes =
        llvm::sys::fs::createTemporaryFile("exporter", outputFilename, path);

    RETURN_ERR_IF_NOT(tempFileRes.value() == 0,
                      "Failed to create temp file to write into.");

    outputFilename = path.c_str();
  }
  ScopeGuard cleanup([&]() { llvm::sys::fs::remove(outputFilename); });
  if (useString) {
    cleanup.dismiss();
  }
  // Write model to file or string.
  {
    Error err = Error::empty();
    llvm::StringMap<std::string> extraMetadataProps;
    RETURN_IF_ERR(ONNXModelWriter::insertLoaderNameUniqueOffsetMetadata(
        extraMetadataProps, originNameToTQPMap));
    ONNXModelWriter onnxWR(
        outputFilename, *F, irVer, opsetVer, &err, !zipMode, zipMode,
        useGlowCustomOps, includeConstantData, extraMetadataProps,
        constFoldRecord ? *constFoldRecord : ConstantFoldingRecordMap(),
        backendSpecificNodeInfo, (useString) ? &outputString : nullptr);
    if (err) {
      llvm::errs() << "Failed to write model\n";
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
    ONNXModelLoader onnxLD(
        outputFilename, inputTensorNames, inputTensorTypes, *R, &err, zipMode,
        reloadCctx ? &reloadCctx->backendOpts.backendSpecificNodeInfo : nullptr,
        /* disableConstFoldInLoader */ true,
        /* loadIntoExistingModule */ !includeConstantData,
        /* Backend */ nullptr,
        /* inputStringPtr */ (useString) ? &outputString : nullptr);

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
void testLoadAndSaveONNXModel(const std::string &name, bool zipMode,
                              bool useString) {
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
                                         zipMode, useGlowCustomOps, useString)
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

/// Helper that \returns whether two StringMaps \p LHS and \p RHS are equal.
template <typename T>
static bool isStrMapEqual(const llvm::StringMap<T> &LHS,
                          const llvm::StringMap<T> &RHS) {
  if (LHS.size() != RHS.size()) {
    return false;
  }
  for (const auto &keyValue : LHS) {
    auto findInRHS = RHS.find(keyValue.getKey());
    if (findInRHS == RHS.end()) {
      return false;
    }
    if (keyValue.getValue() != findInRHS->getValue()) {
      return false;
    }
  }
  return true;
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
  /// folding records are created during constant folding/recording. Any Nodes
  /// listed in \p nodesToPar will be Model parallelized in two.
  void serializeAndReloadAndCompareResults(
      unsigned numExpectedConstsFolded,
      const std::unordered_set<Node *> &nodesToPar = {}) {
    bindings_.allocate(mod_.getPlaceholders());

    // Perform constant folding, recording what occurs so we can serialize it.
    ConstantFoldingRecordMap record = constantFoldAndRecord(F_, cctx_);
    EXPECT_EQ(record.size(), numExpectedConstsFolded);
    runDCEPass(F_, cctx_);

    if (nodesToPar.size()) {
      llvm::DenseMap<Node *, size_t> numChunks;
      llvm::DenseMap<Node *, ParallelTransformKind> parOpts;
      for (Node *N : nodesToPar) {
        numChunks[N] = 2;
        parOpts[N] = ParallelTransformKind::Model;
      }

      std::unordered_map<Node *, ConcatNode *> replacedMap;
      ASSIGN_VALUE_OR_FAIL_TEST(replacedMap,
                                ::glow::parallelizeOps(F_, numChunks, parOpts));
      EXPECT_EQ(replacedMap.size(), parOpts.size());

      ConstantFoldingRecordMap parRecord = constantFoldAndRecord(F_, cctx_);
      record.insert(parRecord.begin(), parRecord.end());
      runDCEPass(F_, cctx_);
    }

    // Clone the original module into a new module used for reloading the model.
    ExecutionEngine reloadEE(EE_.getBackendName());
    Module &reloadMod = reloadEE.getModule();
    mod_.clone(&reloadMod);

    // Save and reload F.
    Function *reloadF;
    CompilationContext reloadCctx;
    ASSIGN_VALUE_OR_FAIL_TEST(
        reloadF, saveAndReloadFunction(
                     reloadMod, F_, {}, {}, 7, 9,
                     /* zipMode */ false,
                     /* useGlowCustomOps */ true,
                     /* useString */ false,
                     /* includeConstantData */ false, &record, &reloadCctx,
                     cctx_.backendOpts.backendSpecificNodeInfo));

    // Verify that the Function and its Module are the same before and after.
    EXPECT_EQ(reloadF->toString(), F_->toString());
    EXPECT_EQ(reloadMod.toString(), mod_.toString());

    PlaceholderBindings reloadBindings =
        bindings_.clone(reloadMod.getPlaceholders());

    // Now run both to check they have bitwise equal results.
    EE_.compile(cctx_);
    EE_.run(bindings_);

    reloadEE.compile(reloadCctx);
    reloadEE.run(reloadBindings);

    EXPECT_TRUE(
        PlaceholderBindings::compare(&bindings_, &reloadBindings, 0.0f));

    // Verify that backend-specific node info was serialized correctly.
    EXPECT_EQ(cctx_.backendOpts.backendSpecificNodeInfo.count(F_),
              reloadCctx.backendOpts.backendSpecificNodeInfo.count(reloadF));

    if (cctx_.backendOpts.backendSpecificNodeInfo.count(F_)) {
      auto &origNodeMap = cctx_.backendOpts.backendSpecificNodeInfo[F_];
      auto &reloadNodeMap =
          reloadCctx.backendOpts.backendSpecificNodeInfo[reloadF];

      for (const Node &origN : F_->getNodes()) {
        auto reloadNodeIt = std::find_if(
            reloadF->getNodes().begin(), reloadF->getNodes().end(),
            [&](const Node &N) { return N.getName() == origN.getName(); });
        ASSERT_NE(reloadNodeIt, reloadF->getNodes().end());
        EXPECT_TRUE(
            isStrMapEqual(reloadNodeMap[&*reloadNodeIt], origNodeMap[&origN]));
      }
    }
  }
};

TEST(exporter, onnxModels) {
  std::string inputDirectory(GLOW_DATA_PATH "tests/models/onnxModels");
  std::cout << "inputDirectory: " << inputDirectory << std::endl;
  std::error_code code;
  for (llvm::sys::fs::directory_iterator dirIt(inputDirectory, code);
       !code && dirIt != llvm::sys::fs::directory_iterator();
       dirIt.increment(code)) {
    auto name = dirIt->path();
    if (!endsWith(name, ".onnxtxt")) {
      llvm::outs() << "Ignore non-onnxtxt input: " << name << "\n";
      continue;
    }
    if (name.find("getInputsOnnxDefineSample.onnxtxt") != std::string::npos ||
        name.find("preluInvalidBroadcastSlope.onnxtxt") != std::string::npos ||
        name.find("padReflect.onnxtxt") != std::string::npos ||
        name.find("powMultiBroadcastOp7.onnxtxt") != std::string::npos ||
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
        name.find("ArgMinDefault.onnxtxt") != std::string::npos ||
        name.find("ArgMinKeepDim.onnxtxt") != std::string::npos ||
        name.find("ArgMinNoKeepDim.onnxtxt") != std::string::npos ||
        name.find("upsampleOpset7.onnxtxt") != std::string::npos ||
        name.find("upsampleOpset9.onnxtxt") != std::string::npos ||
        name.find("resizeNearestV11compat.onnxtxt") != std::string::npos ||
        name.find("resizeNearestV11compat_sizes.onnxtxt") !=
            std::string::npos ||
        name.find("resizeBilinearV11compat.onnxtxt") != std::string::npos ||
        name.find("resizeBilinearV11compat_sizes.onnxtxt") !=
            std::string::npos ||
        name.find("upsampleOpset9.onnxtxt") != std::string::npos ||
        name.find("NonMaxSuppressionSSD_ONNX.onnxtxt") != std::string::npos ||
        name.find("NonMaxSuppression.onnxtxt") != std::string::npos ||
        name.find("NonMaxSuppressionOptionalParams.onnxtxt") !=
            std::string::npos ||
        name.find("NonMaxSuppressionSSD.onnxtxt") != std::string::npos ||
        name.find("ROIAlign_onnx.onnxtxt") != std::string::npos ||
        name.find("MatMul4D.onnxtxt") != std::string::npos ||
        name.find("Less.onnxtxt") != std::string::npos ||
        name.find("Erf.onnxtxt") != std::string::npos ||
        name.find("Asin.onnxtxt") != std::string::npos ||
        name.find("Acos.onnxtxt") != std::string::npos ||
        name.find("Atan.onnxtxt") != std::string::npos ||
        name.find("Sin.onnxtxt") != std::string::npos ||
        name.find("Cos.onnxtxt") != std::string::npos ||
        name.find("abs.onnxtxt") != std::string::npos ||
        name.find("log.onnxtxt") != std::string::npos ||
        name.find("RangeInt32.onnxtxt") != std::string::npos ||
        name.find("RangeFloat.onnxtxt") != std::string::npos ||
        name.find("scatterND.onnxtxt") != std::string::npos ||
        name.find("mscatterND.onnxtxt") != std::string::npos ||
        name.find("loop_cond.onnxtxt") != std::string::npos ||
        name.find("loop_empty_tripcount.onnxtxt") != std::string::npos ||
        name.find("loop_emptycond.onnxtxt") != std::string::npos ||
        name.find("loop_no_iteration.onnxtxt") != std::string::npos ||
        name.find("loop_tripcount.onnxtxt") != std::string::npos ||
        name.find("loop_withoutN.onnxtxt") != std::string::npos ||
        name.find("sign.onnxtxt") != std::string::npos ||
        name.find("gatherND.onnxtxt") != std::string::npos ||
        name.find("softmax13.onnxtxt") != std::string::npos ||
        name.find("logsoftmax.onnxtxt") != std::string::npos ||
        name.find("hardsigmoid.onnxtxt") != std::string::npos ||
        name.find("simpleConvTranspose.onnxtxt") != std::string::npos ||
        name.find("simpleConvTransposeOutShape.onnxtxt") != std::string::npos ||
        name.find("simpleConvTransposeOutShapeDilation.onnxtxt") !=
            std::string::npos ||
        name.find("simpleConvTransposeOutShapeSameLower.onnxtxt") !=
            std::string::npos ||
        name.find("simpleConvTransposeOutShapeSameUpper.onnxtxt") !=
            std::string::npos ||
        name.find("simpleConvTransposeAutoPadSameLower.onnxtxt") !=
            std::string::npos ||
        name.find("simpleConvTransposeAutoPadSameUpper.onnxtxt") !=
            std::string::npos ||
        name.find("convTransposeAsymmetric.onnxtxt") != std::string::npos ||
        name.find("Mean.onnxtxt") != std::string::npos ||
        name.find("Mean_broadcast.onnxtxt") != std::string::npos ||
        name.find("NonZero.onnxtxt") != std::string::npos ||
        name.find("logicalAnd.onnxtxt") != std::string::npos ||
        name.find("logicalAndBcast.onnxtxt") != std::string::npos ||
        name.find("logicalOrBcast.onnxtxt") != std::string::npos ||
        name.find("logicalOr.onnxtxt") != std::string::npos ||
        name.find("logicalXorBcast.onnxtxt") != std::string::npos ||
        name.find("logicalXor.onnxtxt") != std::string::npos ||
        name.find("logicalNot.onnxtxt") != std::string::npos ||

        name.find("simpleConvTransposePads.onnxtxt") != std::string::npos ||
        name.find("simpleConvTransposeAutoPadValid.onnxtxt") !=
            std::string::npos ||
        name.find("simpleConvTransposeOutShapeSameUpper.onnxtxt") !=
            std::string::npos ||
        name.find("simpleConvTransposeAutoPadSameLower.onnxtxt") !=
            std::string::npos ||
        name.find("convTransposeGroup.onnxtxt") != std::string::npos ||
        name.find("pow_element_wise.onnxtxt") != std::string::npos ||
        name.find("pow_array_broadcast.onnxtxt") != std::string::npos ||
        name.find("pow_scalar_broadcast.onnxtxt") != std::string::npos ||
        name.find("simpleConvTransposeAutoPadSameUpper.onnxtxt") !=
            std::string::npos ||
        name.find("sliceInvalidAxes.onnxtxt") != std::string::npos ||
        name.find("sliceWithUnsupportedStep.onnxtxt") != std::string::npos ||
        name.find("simpleConv3DNonSquareDilation.onnxtxt") !=
            std::string::npos) {
      // Ignore invalid ONNX files and graphs without nodes.
      llvm::outs() << "Ignore invalid input files: " << name << "\n";
      continue;
    }
    if (name.find("constant.onnxtxt") != std::string::npos ||
        name.find("shape.onnxtxt") != std::string::npos ||
        name.find("bool_from_int.onnxtxt") != std::string::npos ||
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

    testLoadAndSaveONNXModel(dirIt->path(), /* zipMode */ true,
                             /* useString */ false);
    testLoadAndSaveONNXModel(dirIt->path(), /* zipMode */ false,
                             /* useString */ false);
    testLoadAndSaveONNXModel(dirIt->path(), /* zipMode */ false,
                             /* useString */ true);

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
      {dilation, dilation});

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
                              strides, pads, groups, {dilation, dilation});

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
  EXPECT_EQ(avgPoolReloaded->getCountIncludePads(),
            avgPool->getCountIncludePads());
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

/// Verify that exporting and reloading with placement hints retains the hints.
TEST_F(ConstFoldReloadTest, exportWithPlacementHints) {
  auto *input1 =
      mod_.createPlaceholder(ElemKind::Float16Ty, {16, 32}, "input1", false);
  auto *input2 =
      mod_.createPlaceholder(ElemKind::Float16Ty, {16, 32}, "input2", false);
  auto *weights =
      F_->getParent()->createConstant(ElemKind::Float16Ty, {16, 16}, "weights");
  auto *bias =
      F_->getParent()->createConstant(ElemKind::Float16Ty, {16}, "bias");
  weights->getPayloadMutable().getHandle<float16_t>().randomize(-1.0, 1.0,
                                                                mod_.getPRNG());
  bias->getPayloadMutable().getHandle<float16_t>().randomize(-1.0, 1.0,
                                                             mod_.getPRNG());

  auto *CI = F_->createConcat("concat", {input1, input2}, 1);
  auto *TN = F_->createTranspose("transpose", CI, {1, 0});
  auto *FC = F_->createFullyConnected("fc", TN, weights, bias);
  auto *THN = F_->createTanh("tanh", FC);
  auto *SN = F_->createSigmoid("sigmoid", THN);
  F_->createSave("ret", SN);

  auto *AN = F_->createAdd("add", input1, input2);
  F_->createSave("add_save", AN);

  bindings_.allocate(input1)->getHandle<float16_t>().randomize(-1.0, 1.0,
                                                               mod_.getPRNG());
  bindings_.allocate(input2)->getHandle<float16_t>().randomize(-1.0, 1.0,
                                                               mod_.getPRNG());

  auto &nodeInfo = cctx_.backendOpts.backendSpecificNodeInfo[F_];

  nodeInfo[AN]["Interpreter_Hint1"].push_back(CI->getName().str());
  nodeInfo[AN]["Interpreter_Hint2"].push_back("@1");
  nodeInfo[CI]["Interpreter_Hint1"].push_back(TN->getName().str());
  nodeInfo[CI]["Interpreter_Hint3"].push_back("@1");
  nodeInfo[CI]["Interpreter_Hint1"].push_back(FC->getName().str());
  nodeInfo[CI]["Interpreter_Hint3"].push_back("@1");
  nodeInfo[AN]["Interpreter_Hint1"].push_back(CI->getName().str());
  nodeInfo[AN]["Interpreter_Hint2"].push_back("@1");
  nodeInfo[TN]["Interpreter_Hint1"].push_back(FC->getName().str());
  nodeInfo[TN]["Interpreter_Hint1"].push_back(SN->getName().str());
  nodeInfo[FC]["Interpreter_Hint1"].push_back(THN->getName().str());

  nodeInfo[TN]["Interpreter_Hint4"].push_back("3");
  nodeInfo[FC]["Interpreter_Hint4"].push_back("2");
  nodeInfo[CI]["Interpreter_Hint4"].push_back("1");
  nodeInfo[CI]["Interpreter_Hint5"].push_back("@0");
  nodeInfo[CI]["Interpreter_Hint4"].push_back("3");
  nodeInfo[CI]["Interpreter_Hint5"].push_back("@1");

  serializeAndReloadAndCompareResults(0);
}

TEST_F(ConstFoldReloadTest, exportParallelizedGraphWithTwoConstFoldingRecords) {
  Placeholder *I =
      mod_.createPlaceholder(ElemKind::Float16Ty, {2, 100}, "input",
                             /* isTrainable */ false);
  Constant *W = mod_.createConstant(ElemKind::FloatTy, {10, 100}, "weights");
  Constant *B = mod_.createConstant(ElemKind::Float16Ty, {10}, "bias");

  ClipNode *clipW = F_->createClip("clip", W, -5.f, 5.f);
  ConvertToNode *convertW =
      F_->createConvertTo("conv", clipW, ElemKind::Float16Ty);
  TransposeNode *transposeW =
      F_->createTranspose("transpose", convertW, {1, 0});
  FullyConnectedNode *FC = F_->createFullyConnected("fc", I, transposeW, B);
  F_->createSave("save", FC);

  Constant *W2 = mod_.createConstant(ElemKind::Float16Ty, {2, 100}, "weight2");
  TanhNode *tanhW = F_->createTanh("tanh", W2);
  AddNode *add = F_->createAdd("add", tanhW, I);
  F_->createSave("save_add", add);

  bindings_.allocate(I)->getHandle<float16_t>().randomize(-10, 10,
                                                          mod_.getPRNG());
  W->getHandle().randomize(-1.0, 1.0, mod_.getPRNG());
  B->getHandle<float16_t>().randomize(0.0, 0.5, mod_.getPRNG());
  W2->getPayloadMutable().getHandle<float16_t>().randomize(-10, 10,
                                                           mod_.getPRNG());

  serializeAndReloadAndCompareResults(2, {FC});
}

TEST(exporter, VeryLongChain) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  auto *F = mod.createFunction("F");

  Placeholder *input =
      mod.createPlaceholder(ElemKind::Float16Ty, {1, 6}, "input", false);

  Node *cur = input;
  for (dim_t iter = 0; iter < 3000; iter++) {
    auto *mul = F->createMul("mul", cur, cur);
    auto *clip = F->createClip("clip", mul, 0.0, 128.0);
    if (iter == 0) {
      F->createSave("save_out0", clip);
    }
    cur = (Node *)clip;
  }
  auto *save = F->createSave("save_out", cur);

  Placeholder *output = save->getPlaceholder();

  ASSERT_TRUE(F->verify());

  PlaceholderBindings bindings;
  bindings.allocate({input, output});

  // Save and reload F.
  Function *R;
  Module reloadMod;
  ASSIGN_VALUE_OR_FAIL_TEST(
      R, saveAndReloadFunction(reloadMod, F, {"input"}, {input->getType()}));
  (void)R;
}

/// Tests that we can serialize and then reload a model with OriginNameToTQPMap
/// added to the model. Note that we don't do anything with the reloaded map.
TEST(exporter, TestUniqueOffsetMapSerialization) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  auto *F = mod.createFunction("F");

  Placeholder *I =
      mod.createPlaceholder(ElemKind::Float16Ty, {5, 3}, "input", false);
  Constant *W =
      mod.createConstant(ElemKind::Int8QTy, {3, 4}, 0.f, 0, "weights");
  Constant *B = mod.createConstant(ElemKind::Int8QTy, {4}, 0.f, 1, "bias");
  QuantizeNode *QI = F->createQuantize("quant", I, ElemKind::Int8QTy, 0.f, 2);
  FullyConnectedNode *FC = F->createFullyConnected("fc", QI, W, B);
  SaveNode *save = F->createSave("save_out", FC);

  Placeholder *output = save->getPlaceholder();

  ASSERT_TRUE(F->verify());

  PlaceholderBindings bindings;
  bindings.allocate({I, output});

#define GET_TQP(T_) TensorQuantizationParams{T_->getScale(), T_->getOffset()}
  OriginNameToTQPMap originNameToTQPMap;
  originNameToTQPMap.emplace(W->getName(), GET_TQP(W->getOutput().getType()));
  originNameToTQPMap.emplace(B->getName(), GET_TQP(B->getOutput().getType()));
  originNameToTQPMap.emplace(QI->getName(), GET_TQP(QI->getResult().getType()));
#undef GET_TQP

  // Save and reload F.
  Function *R;
  Module reloadMod;
  ASSIGN_VALUE_OR_FAIL_TEST(
      R, saveAndReloadFunction(reloadMod, F, {"input"}, {I->getType()}, 7, 9,
                               /* zipMode */ false,
                               /* useGlowCustomOps */ true,
                               /* useString */ false,
                               /* includeConstantData */ true,
                               /* record */ nullptr, /* reloadCctx */ nullptr,
                               /* backendSpecificNodeInfo */ {},
                               originNameToTQPMap));
  (void)R;
}

/// Test that we can serialize tensor strides.
TEST(exporter, TestStridesSerialization) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  auto *F = mod.createFunction("F");

  auto ty = mod.uniqueType(ElemKind::Float16Ty, {5, 3});
  // Create a type with non-standard strides.
  ty = mod.uniqueTypeWithNewStrides(ty, ty->dims(), {332, 1});
  Placeholder *I = mod.createPlaceholder(ty, "input", false);
  SaveNode *save = F->createSave("save_out", I);

  Placeholder *output = save->getPlaceholder();

  ASSERT_TRUE(F->verify());

  PlaceholderBindings bindings;
  bindings.allocate({I, output});

  // Save and reload F in text mode.
  {
    Function *R;
    Module reloadMod;
    ASSIGN_VALUE_OR_FAIL_TEST(
        R, saveAndReloadFunction(reloadMod, F, {"input"}, {I->getType()}, 7, 9,
                                 /* zipMode */ false,
                                 /* useGlowCustomOps */ true,
                                 /* useString */ false,
                                 /* includeConstantData */ true,
                                 /* record */ nullptr, /* reloadCctx */ nullptr,
                                 /* backendSpecificNodeInfo */ {}));
    (void)R;
  }
  // Save and reload F in zip mode.
  {
    Function *R;
    Module reloadMod;
    ASSIGN_VALUE_OR_FAIL_TEST(
        R, saveAndReloadFunction(reloadMod, F, {"input"}, {I->getType()}, 7, 9,
                                 /* zipMode */ true,
                                 /* useGlowCustomOps */ true,
                                 /* useString */ false,
                                 /* includeConstantData */ true,
                                 /* record */ nullptr, /* reloadCctx */ nullptr,
                                 /* backendSpecificNodeInfo */ {}));
    (void)R;
  }
}
