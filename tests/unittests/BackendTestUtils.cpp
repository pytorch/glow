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

#include "BackendTestUtils.h"

#include "glow/Converter/TypeAToTypeBFunctionConverter.h"
#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Graph/Graph.h"
#include "glow/IR/IR.h"
#include "glow/IR/IRBuilder.h"
#include "glow/IR/Instrs.h"
#include "glow/Optimizer/GraphOptimizer/GraphOptimizer.h"
#include "glow/Quantization/Quantization.h"

#include "gtest/gtest.h"

#include "llvm/Support/CommandLine.h"

namespace glow {

llvm::cl::OptionCategory operatorTestCat("OperatorTest Category");

unsigned parCloneCountOpt;
llvm::cl::opt<unsigned, /* ExternalStorage */ true> parCloneCountI(
    "parallel-clone-count",
    llvm::cl::desc(
        "Number of times to clone a graph in parallel. Intended to stress test "
        "different backends. This option is not used by all unit "
        "tests; for now you must check the test to see if so."),
    llvm::cl::location(parCloneCountOpt), llvm::cl::Optional, llvm::cl::init(1),
    llvm::cl::cat(operatorTestCat));

using llvm::cast;

namespace {
// Helpers for creating and intializing placeholders from tensors.
static Placeholder *createPlaceholder(Module &mod,
                                      PlaceholderBindings &bindings,
                                      Tensor *tensor, llvm::StringRef name) {
  auto *P = mod.createPlaceholder(tensor->getElementType(), tensor->dims(),
                                  name, false);
  auto *PTensor = bindings.allocate(P);
  PTensor->assign(tensor);

  return P;
}

static Placeholder *createQuantizedPlaceholder(Module &mod,
                                               PlaceholderBindings &bindings,
                                               Tensor *tensor, float scale,
                                               int32_t offset,
                                               llvm::StringRef name) {
  auto *P = mod.createPlaceholder(tensor->getElementType(), tensor->dims(),
                                  scale, offset, name, false);
  auto *PTensor = bindings.allocate(P);
  PTensor->assign(tensor);

  return P;
}

/// Clone, profile, and run \p origF given the \p bindings and \p EE. \returns
/// the quantization parameters from the profile given the specified \p schema.
static std::vector<NodeQuantizationInfo>
profileAndGetNodeQuantizationInfo(PlaceholderBindings &bindings,
                                  ExecutionEngine &EE, Function *origF,
                                  quantization::Schema schema) {
  LoweredInfoMap loweredMapForProf;
  CompilationContext cctx{&bindings, &loweredMapForProf};
  cctx.precisionConfig.quantMode = QuantizationMode::Profile;

  Function *profileF = origF->clone("profile");
  EE.compile(profileF, cctx);

  EE.run(bindings);

  return quantization::generateNodeQuantizationInfos(bindings, profileF,
                                                     loweredMapForProf, schema);
}

/// Helper that sets up and \returns a pair of configs for both interpreter and
/// backend being tested.
static std::pair<CompilationContext, CompilationContext>
setupInterpAndBackendConfigs(
    Function *IF, ExecutionEngine &IEE, PlaceholderBindings &Ibindings,
    LoweredInfoMap &ILIM, PlaceholderBindings &Bbindings, LoweredInfoMap &BLIM,
    ElemKind interpElemKind, ElemKind backendElemKind,
    quantization::Schema schema, bool enableRowwiseQuantization) {
  CompilationContext cctxI{&Ibindings, &ILIM};
  CompilationContext cctxB{&Bbindings, &BLIM};
  PrecisionConfiguration &precConfigI = cctxI.precisionConfig;
  PrecisionConfiguration &precConfigB = cctxB.precisionConfig;

  if (isQuantizedElemKind(interpElemKind) ||
      isQuantizedElemKind(backendElemKind)) {
    // If either interp or backend need to be quantized then we need to profile
    // and get quantization infos.
    auto NQIS = profileAndGetNodeQuantizationInfo(Ibindings, IEE, IF, schema);

    if (isQuantizedElemKind(interpElemKind)) {
      precConfigI.quantMode = QuantizationMode::Quantize;
      precConfigI.quantConfig.infos = NQIS;
      precConfigI.quantConfig.enableRowwise = enableRowwiseQuantization;
      precConfigI.quantConfig.schema = schema;
      precConfigI.quantConfig.precision = interpElemKind;
      precConfigI.quantConfig.assertAllNodesQuantized = true;
    }

    if (isQuantizedElemKind(backendElemKind)) {
      precConfigB.quantMode = QuantizationMode::Quantize;
      precConfigB.quantConfig.infos = NQIS;
      precConfigB.quantConfig.enableRowwise = enableRowwiseQuantization;
      precConfigB.quantConfig.schema = schema;
      precConfigB.quantConfig.precision = backendElemKind;
      precConfigB.quantConfig.assertAllNodesQuantized = true;
    }
  }

  precConfigI.convertToFP16 = interpElemKind == ElemKind::Float16Ty;
  precConfigB.convertToFP16 = backendElemKind == ElemKind::Float16Ty;

  return std::make_pair(cctxI, cctxB);
}
} // namespace

void compareAgainstInterpreter(llvm::StringRef backendName,
                               CreateAndInitFunction createAndInitFunction,
                               ElemKind interpElemKind,
                               ElemKind backendElemKind, float allowedError,
                               unsigned count, bool enableRowwiseQuantization,
                               quantization::Schema schema) {
  ExecutionEngine IEE{"Interpreter"};
  ExecutionEngine BEE{backendName};
  PlaceholderBindings Ibindings, Bbindings;

  LOG(INFO) << "Comparing Interpreter with precision "
            << Type::getElementName(interpElemKind).str() << " against "
            << backendName.str() << " with precision "
            << Type::getElementName(backendElemKind).str();

  // Create the same network on the interpreter and the backend being tested.
  FunctionTensorPair IFT = createAndInitFunction(Ibindings, IEE);
  FunctionTensorPair BFT = createAndInitFunction(Bbindings, BEE);

  // Clone the Function inside itself many times if desired.
  std::unordered_set<Tensor *> resultTensors =
      cloneFunInsideFun(BFT, &Bbindings, count);
  assert(resultTensors.size() == count &&
         "Should get the same number of Tensors back as count.");

  Function *IF = IFT.first;
  Function *BF = BFT.first;

  // Set up the configs for interpreter and backend. If one or both functions
  // will be quantized, then gather a profile the graph on the interpreter, and
  // then quantize the Functions as requested.
  LoweredInfoMap ILIM, BLIM;
  auto configs = setupInterpAndBackendConfigs(
      IF, IEE, Ibindings, ILIM, Bbindings, BLIM, interpElemKind,
      backendElemKind, schema, enableRowwiseQuantization);
  CompilationContext &cctxI = configs.first;
  CompilationContext &cctxB = configs.second;

  BEE.compile(BF, cctxB);
  BEE.run(Bbindings);

  // If we profiled inside of setupInterpAndBackendConfigs() (always in FloatTy)
  // then IF's result tensor is already set to the result of the original
  // (FloatTy) IF. So if we did not run profiling, or if we did profile but also
  // modified IF to be non-FloatTy, then need to run again to get IF's result.
  const bool alreadyProfiled = isQuantizedElemKind(interpElemKind) ||
                               isQuantizedElemKind(backendElemKind);
  if (!alreadyProfiled || interpElemKind != ElemKind::FloatTy) {
    IEE.compile(IF, cctxI);
    IEE.run(Ibindings);
  }

  // Compare each of our result tensors to the original.
  for (Tensor *T : resultTensors) {
    EXPECT_TRUE(IFT.second->isEqual(*T, allowedError));
  }
}

std::unordered_set<Tensor *> cloneFunInsideFun(FunctionTensorPair FTP,
                                               PlaceholderBindings *bindings,
                                               unsigned count) {
  Function *origF = FTP.first;

  // Always save the original Function's Tensor, which we will keep around.
  std::unordered_set<Tensor *> resultTensors;
  resultTensors.insert(FTP.second);

  // Nothing to do if we just want the one.
  if (count == 1) {
    return resultTensors;
  }

  Module *mod = origF->getParent();

  // Clone the original Function to repeatedly add it to the original.
  auto *cloneF = origF->clone("single_clone");

  // We keep the original Function, then clone/add count-1 more.
  for (size_t i = 1; i < count; i++) {
    // Clone the clone, and then add all the new nodes to the original function.
    auto *tmpF = cloneF->clone("tmp" + std::to_string(i));
    std::unordered_set<Node *> clonedNodes;
    bool foundSaveNode = false;
    for (auto &N : tmpF->getNodes()) {
      clonedNodes.insert(&N);

      // For every Node we add, check if it uses a Placeholder node, and if so
      // clone it in the Module so that CSE doesn't undo all our hard work.
      for (size_t j = 0, f = N.getNumInputs(); j < f; j++) {
        Placeholder *origPH = llvm::dyn_cast<Placeholder>(N.getNthInput(j));
        if (!origPH) {
          continue;
        }

        // Clone the Placeholder, allocate it in the bindings, and replace the
        // usage of the original node to point to the clone.
        Placeholder *clonePH = mod->createPlaceholder(
            origPH->getType(), origPH->getName(), origPH->isTraining());
        Tensor *oldT = bindings->get(origPH);
        assert(oldT);
        Tensor *newT = bindings->allocate(clonePH);
        newT->assign(oldT);
        N.setNthInput(j, clonePH);

        // Save the result Tensors to return so we can compare the results of
        // all of our clones.
        if (llvm::isa<SaveNode>(N)) {
          assert(!foundSaveNode &&
                 "Can only handle Functions with a single SaveNode.");
          foundSaveNode = true;
          resultTensors.insert(newT);
        }
      }
    }
    for (auto &N : clonedNodes) {
      origF->takeOwnershipOfNode(N);
    }
  }
  // Now erase the clone we used to copy in, as it's no longer needed.
  mod->eraseFunction(cloneF);

  return resultTensors;
}

void inferIntLookupTableNet(Tensor *input, Tensor *out,
                            llvm::ArrayRef<int8_t> table,
                            llvm::StringRef kind) {
  PlaceholderBindings bindings;
  ExecutionEngine EE(kind);
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  auto outTy = mod.uniqueType(ElemKind::Int8QTy, {input->size()}, 3, 3);
  auto var = createQuantizedPlaceholder(mod, bindings, input, outTy->getScale(),
                                        outTy->getOffset(), "var");
  auto *lookupTable = F->createIntLookupTable("lookuptable", var, table, outTy);
  auto *result = F->createSave("ret", lookupTable);
  auto *resultTensor = bindings.allocate(result->getPlaceholder());

  EE.compile(CompilationMode::Infer, F);

  updateInputPlaceholders(bindings, {var}, {input});
  EE.run(bindings);
  out->assign(resultTensor);
}

void inferConvNet(Tensor *inputs, Tensor *filter, Tensor *bias, Tensor *out,
                  llvm::StringRef kind) {
  PlaceholderBindings bindings;
  ExecutionEngine EE(kind);
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  Placeholder *inputP;
  Placeholder *filterP;
  Placeholder *biasP;
  Placeholder *outP;
  TypeRef OT;
  if (inputs->getType().isQuantizedType()) {
    auto &outType = out->getType();
    inputP =
        createQuantizedPlaceholder(mod, bindings, inputs, outType.getScale(),
                                   outType.getOffset(), "inputP");
    filterP =
        createQuantizedPlaceholder(mod, bindings, filter, outType.getScale(),
                                   outType.getOffset(), "filterP");
    biasP = createQuantizedPlaceholder(mod, bindings, bias, outType.getScale(),
                                       outType.getOffset(), "biasP");
    outP = createQuantizedPlaceholder(mod, bindings, out, outType.getScale(),
                                      outType.getOffset(), "outP");
    OT = F->getParent()->uniqueType(out->getElementType(), out->dims(),
                                    outType.getScale(), outType.getOffset());
  } else {
    inputP = createPlaceholder(mod, bindings, inputs, "inputP");
    filterP = createPlaceholder(mod, bindings, filter, "filterP");
    biasP = createPlaceholder(mod, bindings, bias, "biasP");
    outP = createPlaceholder(mod, bindings, out, "outP");
    OT = F->getParent()->uniqueType(out->getElementType(), out->dims());
  }
  auto *conv = F->createConv("conv", inputP, filterP, biasP, OT, 5, 3, 4, 1);
  auto *result = F->createSave("ret", conv, outP);
  auto *resultTensor = bindings.get(result->getPlaceholder());

  EE.compile(CompilationMode::Infer, F);

  updateInputPlaceholders(bindings, {inputP, filterP, biasP},
                          {inputs, filter, bias});
  EE.run(bindings);
  out->assign(resultTensor);
}

void trainConvNet(Tensor *inputs, Tensor *kernel1, Tensor *bias1,
                  Tensor *kernel2, Tensor *bias2, Tensor *selected,
                  llvm::ArrayRef<size_t> shape1, llvm::ArrayRef<size_t> shape2,
                  Tensor *out, llvm::StringRef kind) {
  ExecutionEngine EE(kind);
  TrainingConfig TC;
  PlaceholderBindings bindings;

  // This variable records the number of the next sample to be used for
  // training.
  size_t sampleCounter = 0;

  TC.learningRate = 0.03;
  TC.momentum = 0.3;
  TC.L2Decay = 0.01;
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  auto *var1 = createPlaceholder(mod, bindings, inputs, "var1");
  auto *var2 = createPlaceholder(mod, bindings, selected, "var2");
  auto *conv1 = F->createConv(bindings, "conv1", var1, 3, {5, 3}, {2, 1},
                              {2, 1, 2, 1}, 1);
  bindings.get(cast<Placeholder>(conv1->getFilter()))->assign(kernel1);
  bindings.get(cast<Placeholder>(conv1->getBias()))->assign(bias1);
  auto *reshape1 = F->createReshape("reshape1", conv1, shape1);
  auto *conv2 = F->createConv(bindings, "conv2", reshape1, 2, 2, 2, 0, 1);
  bindings.get(cast<Placeholder>(conv2->getFilter()))->assign(kernel2);
  bindings.get(cast<Placeholder>(conv2->getBias()))->assign(bias2);
  auto *reshape2 = F->createReshape("reshape2", conv2, shape2);
  auto *softmax = F->createSoftMax("softmax", reshape2, var2);
  auto *result = F->createSave("ret", softmax);
  auto *resultTensor = bindings.allocate(result->getPlaceholder());

  Function *TF = glow::differentiate(F, TC);
  EE.compile(CompilationMode::Train, TF);

  runBatch(EE, bindings, 8, sampleCounter, {var1, var2}, {inputs, selected});
  EE.compile(CompilationMode::Infer, F);
  updateInputPlaceholders(bindings, {var1, var2}, {inputs, selected});
  EE.run(bindings);
  out->assign(resultTensor);
}

void inferLocalResponseNormalizationNet(Tensor *inputs, Tensor *out,
                                        llvm::StringRef kind) {
  PlaceholderBindings bindings;
  ExecutionEngine EE(kind);
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  auto *var = createPlaceholder(mod, bindings, inputs, "var");
  auto *lrn = F->createLocalResponseNormalization("lrn", var, 5, 3.0, 0.5, 1.5);
  auto *result = F->createSave("ret", lrn);
  auto *resultTensor = bindings.allocate(result->getPlaceholder());

  EE.compile(CompilationMode::Infer, F);

  updateInputPlaceholders(bindings, {var}, {inputs});
  EE.run(bindings);
  out->assign(resultTensor);
}

void trainLocalResponseNormalizationNet(Tensor *inputs, Tensor *weights,
                                        Tensor *bias, Tensor *selected,
                                        llvm::ArrayRef<size_t> shape1,
                                        llvm::ArrayRef<size_t> shape2,
                                        Tensor *out, llvm::StringRef kind) {
  PlaceholderBindings bindings;
  ExecutionEngine EE(kind);
  TrainingConfig TC;

  // This variable records the number of the next sample to be used for
  // training.
  size_t sampleCounter = 0;

  TC.learningRate = 0.06;
  TC.momentum = 0.1;
  TC.L2Decay = 0.01;
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  auto *var1 = createPlaceholder(mod, bindings, inputs, "var1");
  auto *var2 = createPlaceholder(mod, bindings, selected, "var2");
  auto *fc = F->createFullyConnected(bindings, "fc", var1, bias->dims()[0]);
  bindings.get(cast<Placeholder>(fc->getWeights()))->assign(weights);
  bindings.get(cast<Placeholder>(fc->getBias()))->assign(bias);
  auto *reshape1 = F->createReshape("reshape1", fc, shape1);
  auto *lrn =
      F->createLocalResponseNormalization("lrn", reshape1, 2, 2.0, 0.5, 1.0);
  auto *reshape2 = F->createReshape("reshape2", lrn, shape2);
  auto *softmax = F->createSoftMax("softmax", reshape2, var2);
  auto *result = F->createSave("ret", softmax);
  auto *resultTensor = bindings.allocate(result->getPlaceholder());

  Function *TF = glow::differentiate(F, TC);
  EE.compile(CompilationMode::Train, TF);
  runBatch(EE, bindings, 8, sampleCounter, {var1, var2}, {inputs, selected});

  EE.compile(CompilationMode::Infer, F);

  runBatch(EE, bindings, 1, sampleCounter, {var1, var2}, {inputs, selected});
  out->assign(resultTensor);
}

void trainAvgPoolNet(Tensor *inputs, Tensor *weights, Tensor *bias,
                     Tensor *selected, llvm::ArrayRef<size_t> shape1,
                     llvm::ArrayRef<size_t> shape2, Tensor *out,
                     llvm::StringRef kind) {
  ExecutionEngine EE(kind);
  TrainingConfig TC;
  PlaceholderBindings bindings;

  // This variable records the number of the next sample to be used for
  // training.
  size_t sampleCounter = 0;

  TC.learningRate = 0.01;
  TC.momentum = 0.4;
  TC.L2Decay = 0.01;
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  auto *var1 = createPlaceholder(mod, bindings, inputs, "var1");
  auto *var2 = createPlaceholder(mod, bindings, selected, "var2");
  auto *fc = F->createFullyConnected(bindings, "fc", var1, bias->dims()[0]);
  bindings.get(cast<Placeholder>(fc->getWeights()))->assign(weights);
  bindings.get(cast<Placeholder>(fc->getBias()))->assign(bias);
  auto *reshape1 = F->createReshape("reshape1", fc, shape1);
  auto *pool = F->createAvgPool("pool", reshape1, 2, 2, 0);
  auto *reshape2 = F->createReshape("reshape2", pool, shape2);
  auto *softmax = F->createSoftMax("softmax", reshape2, var2);
  auto *result = F->createSave("ret", softmax);
  auto *resultTensor = bindings.allocate(result->getPlaceholder());

  Function *TF = glow::differentiate(F, TC);
  EE.compile(CompilationMode::Train, TF);

  runBatch(EE, bindings, 10, sampleCounter, {var1, var2}, {inputs, selected});
  EE.compile(CompilationMode::Infer, F);

  updateInputPlaceholders(bindings, {var1, var2}, {inputs, selected});
  EE.run(bindings);
  out->assign(resultTensor);
}

void trainMaxPoolNet(Tensor *inputs, Tensor *weights, Tensor *bias,
                     Tensor *selected, llvm::ArrayRef<size_t> shape1,
                     llvm::ArrayRef<size_t> shape2, Tensor *out,
                     llvm::StringRef kind) {
  ExecutionEngine EE(kind);
  TrainingConfig TC;
  PlaceholderBindings bindings;

  // This variable records the number of the next sample to be used for
  // training.
  size_t sampleCounter = 0;

  TC.learningRate = 0.03;
  TC.momentum = 0.3;
  TC.L2Decay = 0.003;
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  auto *var1 = createPlaceholder(mod, bindings, inputs, "var1");
  auto *var2 = createPlaceholder(mod, bindings, selected, "var2");
  auto *fc = F->createFullyConnected(bindings, "fc", var1, bias->dims()[0]);
  bindings.get(cast<Placeholder>(fc->getWeights()))->assign(weights);
  bindings.get(cast<Placeholder>(fc->getBias()))->assign(bias);
  auto *reshape1 = F->createReshape("reshape1", fc, shape1);
  auto *pool = F->createMaxPool("pool", reshape1, 5, 3, 4);
  auto *reshape2 = F->createReshape("reshape2", pool, shape2);
  auto *softmax = F->createSoftMax("softmax", reshape2, var2);
  auto *result = F->createSave("ret", softmax);
  auto *resultTensor = bindings.allocate(result->getPlaceholder());

  Function *TF = glow::differentiate(F, TC);
  EE.compile(CompilationMode::Train, TF);

  runBatch(EE, bindings, 7, sampleCounter, {var1, var2}, {inputs, selected});
  EE.compile(CompilationMode::Infer, F);

  runBatch(EE, bindings, 1, sampleCounter, {var1, var2}, {inputs, selected});
  out->assign(resultTensor);
}

void inferSmallConv(Tensor *inputs, Tensor *out, llvm::StringRef kind) {
  PlaceholderBindings bindings;
  ExecutionEngine EE(kind);
  auto &mod = EE.getModule();
  auto *F = mod.createFunction("main");
  auto *in = createPlaceholder(mod, bindings, inputs, "in");
  auto *C = F->createConv(bindings, "conv2a", in, 64, 1, 1, 0, 1);
  bindings.get(cast<Placeholder>(C->getFilter()))->getHandle().clear(0.3);
  bindings.get(cast<Placeholder>(C->getBias()))->getHandle().clear(0.4);
  auto *result = F->createSave("ret", C);
  auto *resultTensor = bindings.allocate(result->getPlaceholder());
  convertPlaceholdersToConstants(F, bindings, {in, result->getPlaceholder()});

  EE.compile(CompilationMode::Infer, F);

  updateInputPlaceholders(bindings, {in}, {inputs});
  EE.run(bindings);

  out->assign(resultTensor);
}

void inferGroupConv(Tensor *out, llvm::StringRef kind) {
  PlaceholderBindings bindings;
  ExecutionEngine EE(kind);
  auto &mod = EE.getModule();
  auto *F = mod.createFunction("main");

  auto *input =
      mod.createPlaceholder(ElemKind::FloatTy, {1, 2, 1, 32}, "input", false);
  auto *inputTensor = bindings.allocate(input);
  auto IH = inputTensor->getHandle();
  for (size_t i = 0; i < 2 * 32; i++) {
    IH.raw(i) = (i + 1) / 10.0;
  }

  auto *filter = mod.createPlaceholder(ElemKind::FloatTy, {128, 1, 1, 16},
                                       "filter", false);
  auto *filterTensor = bindings.allocate(filter);
  auto FH = filterTensor->getHandle();
  for (size_t i = 0; i < 128; i++)
    for (size_t j = 0; j < 16; j++) {
      FH.at({i, 0, 0, j}) = (i + j) / 100.0;
    }
  auto *zeroBias =
      mod.createPlaceholder(ElemKind::FloatTy, {128}, "bias", false);
  auto *zeroBiasTensor = bindings.allocate(zeroBias);
  zeroBiasTensor->zero();

  auto outTy = mod.uniqueType(ElemKind::FloatTy, {1, 2, 1, 128});

  ConvolutionNode *CN =
      F->createConv("Conv", input, filter, zeroBias, outTy, 1, 1, 0, 2);
  SaveNode *result = F->createSave("save", CN);
  auto *resultTensor = bindings.allocate(result->getPlaceholder());

  EE.compile(CompilationMode::Infer, F);

  EE.run(bindings);
  out->assign(resultTensor);
}

void inferNonSquarePaddingConv(Tensor *out, llvm::StringRef kind) {
  PlaceholderBindings bindings;
  ExecutionEngine EE(kind);
  auto &mod = EE.getModule();
  auto *F = mod.createFunction("main");

  auto *input =
      mod.createPlaceholder(ElemKind::FloatTy, {1, 2, 1, 32}, "input", false);
  auto *inputTensor = bindings.allocate(input);
  auto IH = inputTensor->getHandle();
  for (size_t i = 0; i < 2 * 32; i++) {
    IH.raw(i) = (i + 1) / 10.0;
  }

  auto *filter = mod.createPlaceholder(ElemKind::FloatTy, {128, 1, 1, 32},
                                       "filter", false);
  auto *filterTensor = bindings.allocate(filter);
  auto FH = filterTensor->getHandle();
  for (size_t i = 0; i < 128; i++)
    for (size_t j = 0; j < 32; j++) {
      FH.at({i, 0, 0, j}) = (i + j) / 100.0;
    }
  auto *zeroBias =
      mod.createPlaceholder(ElemKind::FloatTy, {128}, "bias", false);
  auto *zeroBiasTensor = bindings.allocate(zeroBias);
  zeroBiasTensor->zero();
  auto outTy = mod.uniqueType(ElemKind::FloatTy, {1, 4, 5, 128});

  ConvolutionNode *CN = F->createConv("Conv", input, filter, zeroBias, outTy,
                                      {1, 1}, {1, 1}, {0, 1, 2, 3}, 1);
  SaveNode *result = F->createSave("save", CN);
  auto *resultTensor = bindings.allocate(result->getPlaceholder());

  EE.compile(CompilationMode::Infer, F);

  EE.run(bindings);
  out->assign(resultTensor);
}

void inferNonSquareKernelConv(Tensor *out, llvm::StringRef kind) {
  PlaceholderBindings bindings;
  ExecutionEngine EE(kind);
  auto &mod = EE.getModule();
  auto *F = mod.createFunction("main");

  auto *input =
      mod.createPlaceholder(ElemKind::FloatTy, {1, 2, 1, 32}, "input", false);
  auto *inputTensor = bindings.allocate(input);
  auto IH = inputTensor->getHandle();
  for (size_t i = 0; i < 2 * 32; i++) {
    IH.raw(i) = (i + 1) / 10.0;
  }

  auto *filter = mod.createPlaceholder(ElemKind::FloatTy, {128, 2, 1, 32},
                                       "filter", false);
  auto *filterTensor = bindings.allocate(filter);
  auto FH = filterTensor->getHandle();
  for (size_t i = 0; i < 128; i++)
    for (size_t j = 0; j < 2; j++)
      for (size_t k = 0; k < 32; k++) {
        FH.at({i, j, 0, k}) = (i + j + k) / 100.0;
      }
  auto *zeroBias =
      mod.createPlaceholder(ElemKind::FloatTy, {128}, "bias", false);
  auto *zeroBiasTensor = bindings.allocate(zeroBias);
  zeroBiasTensor->zero();
  auto outTy = mod.uniqueType(ElemKind::FloatTy, {1, 3, 5, 128});

  ConvolutionNode *CN = F->createConv("Conv", input, filter, zeroBias, outTy,
                                      {2, 1}, {1, 1}, {0, 1, 2, 3}, 1);
  SaveNode *result = F->createSave("save", CN);
  auto *resultTensor = bindings.allocate(result->getPlaceholder());

  EE.compile(CompilationMode::Infer, F);

  EE.run(bindings);
  out->assign(resultTensor);
}

void inferNonSquareStrideConv(Tensor *out, llvm::StringRef kind) {
  PlaceholderBindings bindings;
  ExecutionEngine EE(kind);
  auto &mod = EE.getModule();
  auto *F = mod.createFunction("main");

  auto *input =
      mod.createPlaceholder(ElemKind::FloatTy, {1, 2, 1, 32}, "input", false);
  auto *inputTensor = bindings.allocate(input);
  auto IH = inputTensor->getHandle();
  for (size_t i = 0; i < 2 * 32; i++) {
    IH.raw(i) = (i + 1) / 10.0;
  }

  auto *filter = mod.createPlaceholder(ElemKind::FloatTy, {128, 2, 1, 32},
                                       "filter", false);
  auto *filterTensor = bindings.allocate(filter);
  auto FH = filterTensor->getHandle();
  for (size_t i = 0; i < 128; i++)
    for (size_t j = 0; j < 2; j++)
      for (size_t k = 0; k < 32; k++) {
        FH.at({i, j, 0, k}) = (i + j + k) / 100.0;
      }
  auto *zeroBias =
      mod.createPlaceholder(ElemKind::FloatTy, {128}, "bias", false);
  auto *zeroBiasTensor = bindings.allocate(zeroBias);
  zeroBiasTensor->zero();
  auto outTy = mod.uniqueType(ElemKind::FloatTy, {1, 2, 5, 128});

  ConvolutionNode *CN = F->createConv("Conv", input, filter, zeroBias, outTy,
                                      {2, 1}, {2, 1}, {0, 1, 2, 3}, 1);
  SaveNode *result = F->createSave("save", CN);
  auto *resultTensor = bindings.allocate(result->getPlaceholder());

  EE.compile(CompilationMode::Infer, F);

  EE.run(bindings);
  out->assign(resultTensor);
}

void inferConvDKKC8(Tensor *out, llvm::StringRef kind) {
  PlaceholderBindings bindings;
  ExecutionEngine EE(kind);
  auto &mod = EE.getModule();
  auto *F = mod.createFunction("main");

  auto *input =
      mod.createPlaceholder(ElemKind::FloatTy, {3, 3, 3, 32}, "input", false);
  auto *inputTensor = bindings.allocate(input);
  auto IH = inputTensor->getHandle();
  for (size_t i = 0; i < 3 * 3 * 3 * 32; i++) {
    IH.raw(i) = (i + 1) / 10.0;
  }

  auto *filter = mod.createPlaceholder(ElemKind::FloatTy, {192, 3, 3, 32},
                                       "filter", false);
  auto *filterTensor = bindings.allocate(filter);
  filterTensor->zero();
  auto FH = filterTensor->getHandle();
  for (size_t i = 0; i < 192; i++)
    for (size_t j = 0; j < 3; j++)
      for (size_t k = 0; k < 3; k++)
        for (size_t l = 0; l < 32; l++) {
          FH.at({i, j, k, k}) = (i + j + k + l) / 200.0;
        }
  auto *zeroBias =
      mod.createPlaceholder(ElemKind::FloatTy, {192}, "bias", false);
  auto *zeroBiasTensor = bindings.allocate(zeroBias);
  zeroBiasTensor->zero();
  auto outTy = mod.uniqueType(ElemKind::FloatTy, {3, 3, 3, 192});

  ConvolutionNode *CN = F->createConv("Conv", input, filter, zeroBias, outTy,
                                      {3, 3}, {1, 1}, {1, 1, 1, 1}, 1);
  SaveNode *result = F->createSave("save", CN);
  auto *resultTensor = bindings.allocate(result->getPlaceholder());

  EE.compile(CompilationMode::Infer, F);

  EE.run(bindings);
  out->assign(resultTensor);
}

void trainSoftMaxNet(Tensor *inputs, Tensor *weights, Tensor *bias,
                     Tensor *selected, Tensor *out, llvm::StringRef kind) {
  ExecutionEngine EE(kind);
  TrainingConfig TC;
  PlaceholderBindings bindings;

  // This variable records the number of the next sample to be used for
  // training.
  size_t sampleCounter = 0;

  TC.learningRate = 0.003;
  TC.momentum = 0.7;
  TC.L2Decay = 0.001;
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  auto *var1 = createPlaceholder(mod, bindings, inputs, "var1");
  auto *var2 = createPlaceholder(mod, bindings, selected, "var2");
  auto *fc = F->createFullyConnected(bindings, "fc", var1, bias->dims()[0]);
  bindings.get(cast<Placeholder>(fc->getWeights()))->assign(weights);
  bindings.get(cast<Placeholder>(fc->getBias()))->assign(bias);
  auto *softmax = F->createSoftMax("softmax", fc, var2);
  auto *result = F->createSave("ret", softmax);
  auto *resultTensor = bindings.allocate(result->getPlaceholder());

  Function *TF = glow::differentiate(F, TC);

  EE.compile(CompilationMode::Train, TF);

  runBatch(EE, bindings, 30, sampleCounter, {var1, var2}, {inputs, selected});
  EE.compile(CompilationMode::Infer, F);

  updateInputPlaceholders(bindings, {var1, var2}, {inputs, selected});
  EE.run(bindings);
  out->assign(resultTensor);
}

void inferTanhConcatNet(Tensor *input1, Tensor *input2, Tensor *input3,
                        Tensor *out, llvm::StringRef kind) {
  PlaceholderBindings bindings;
  ExecutionEngine EE(kind);
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  auto *var1 = createPlaceholder(mod, bindings, input1, "var1");
  auto *var2 = createPlaceholder(mod, bindings, input2, "var2");
  auto *var3 = createPlaceholder(mod, bindings, input3, "var3");
  auto *T1 = F->createTanh("tanh1", var1);
  auto *T2 = F->createTanh("tanh2", var2);
  auto *T3 = F->createTanh("tanh3", var3);
  Node *C1 = F->createConcat("concat", {T1, T2}, 0);
  Node *C2 = F->createConcat("concat", {T2, T3, C1, T2}, 0);
  auto *result = F->createSave("ret", C2);
  auto *resultTensor = bindings.allocate(result->getPlaceholder());

  EE.compile(CompilationMode::Infer, F);

  updateInputPlaceholders(bindings, {var1, var2, var3},
                          {input1, input2, input3});
  EE.run(bindings);
  out->assign(resultTensor);
}

void inferBasicConvNet(Tensor *inputs, Tensor *out, llvm::StringRef kind,
                       size_t convDepth) {
  PlaceholderBindings bindings;
  ExecutionEngine EE(kind);
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  auto *var = createPlaceholder(mod, bindings, inputs, "var");
  auto *tr = F->createTranspose("tr", var, NCHW2NHWC);
  auto *conv = F->createConv(bindings, "conv", tr, convDepth, {5, 5}, {2, 2},
                             {1, 1, 1, 1}, 1);
  bindings.get(cast<Placeholder>(conv->getFilter()))->getHandle().clear(0.1);
  bindings.get(cast<Placeholder>(conv->getBias()))->getHandle().clear(0.2);
  auto *pool = F->createMaxPool("pool", conv, 2, 2, 0);
  auto *result = F->createSave("ret", pool);
  auto *resultTensor = bindings.allocate(result->getPlaceholder());
  convertPlaceholdersToConstants(F, bindings, {var, result->getPlaceholder()});

  EE.compile(CompilationMode::Infer, F);

  updateInputPlaceholders(bindings, {var}, {inputs});
  EE.run(bindings);
  out->assign(resultTensor);
}

FunctionTensorPair createAndInitBasicFCNet(PlaceholderBindings &bindings,
                                           ExecutionEngine &EE) {
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  auto *var =
      mod.createPlaceholder(ElemKind::FloatTy, {2, 3, 16, 16}, "var", false);
  auto *tr = F->createTranspose("tr", var, NCHW2NHWC);
  auto *fc = F->createFullyConnected(bindings, "fc", tr, 16);
  auto *rl0 = F->createRELU("relu", fc);
  auto *fc2 = F->createFullyConnected(bindings, "fc2", rl0, 8);
  auto *rl1 = F->createRELU("relu", fc2);
  bindings.get(cast<Placeholder>(fc->getWeights()))->getHandle().clear(0.8);
  bindings.get(cast<Placeholder>(fc2->getWeights()))->getHandle().clear(1.5);
  auto *result = F->createSave("ret", rl1);
  auto *resultTensor = bindings.allocate(result->getPlaceholder());

  PseudoRNG PRNG;
  bindings.allocate(var)->getHandle().initXavier(1, PRNG);

  return std::make_pair(F, resultTensor);
}

void inferMixedNet(Tensor *inputs, Tensor *out, llvm::StringRef kind) {
  PlaceholderBindings bindings;
  ExecutionEngine EE(kind);
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  auto *var = createPlaceholder(mod, bindings, inputs, "var");
  auto *selected =
      mod.createPlaceholder(ElemKind::Int64ITy, {2, 1}, "selected", false);

  auto *tr = F->createTranspose("tr", var, NCHW2NHWC);
  auto *fc = F->createFullyConnected(bindings, "fc", tr, 16);
  auto *th0 = F->createTanh("tanh", fc);
  auto *sg0 = F->createSigmoid("sig", fc);
  auto *A1 = F->createAdd("add", th0, sg0);
  auto *fc2 = F->createFullyConnected(bindings, "fc2", A1, 16);

  auto *R = F->createRegression("reg", fc2, fc2);
  auto *SM = F->createSoftMax("SM", R, selected);
  auto *result = F->createSave("ret", SM);
  auto *resultTensor = bindings.allocate(result->getPlaceholder());

  bindings.get(cast<Placeholder>(fc->getWeights()))->getHandle().clear(0.4);
  bindings.get(cast<Placeholder>(fc2->getWeights()))->getHandle().clear(3.5);

  EE.compile(CompilationMode::Infer, F);

  updateInputPlaceholders(bindings, {var}, {inputs});
  EE.run(bindings);
  out->assign(resultTensor);
}

void inferComplexNet1(Tensor *inputs1, Tensor *inputs2, Tensor *inputs3,
                      Tensor *inputs4, Tensor *out, llvm::StringRef kind) {
  PlaceholderBindings bindings;
  ExecutionEngine EE(kind);
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  auto *var1 = createPlaceholder(mod, bindings, inputs1, "var1");
  auto *var2 = createPlaceholder(mod, bindings, inputs2, "var2");
  auto *var3 = createPlaceholder(mod, bindings, inputs3, "var3");
  auto *var4 = createPlaceholder(mod, bindings, inputs4, "var4");
  auto *conv1 = F->createConv(bindings, "conv1", var1, 6, 4, 1, 2, 1);
  bindings.get(cast<Placeholder>(conv1->getFilter()))->getHandle().clear(0.5);
  bindings.get(cast<Placeholder>(conv1->getBias()))->getHandle().clear(0.7);
  auto *sigmoid1 = F->createSigmoid("sigmoid1", conv1);
  auto *fc1 = F->createFullyConnected(bindings, "fc1", var2, 2352);
  bindings.get(cast<Placeholder>(fc1->getWeights()))->getHandle().clear(0.6);
  auto *reshape1 = F->createReshape("reshape1", fc1, {8, 14, 28, 6});
  auto *relu1 = F->createRELU("relu1", reshape1);
  auto *pool1 = F->createMaxPool("pool1", relu1, 2, 2, 1);
  auto *add = F->createAdd("add", sigmoid1, pool1);
  auto *tanh = F->createTanh("tanh", add);
  auto *fc2 = F->createFullyConnected(bindings, "fc2", var3, 720);
  bindings.get(cast<Placeholder>(fc2->getWeights()))->getHandle().clear(1.1);
  auto *reshape2 = F->createReshape("reshape2", fc2, {8, 8, 15, 6});
  auto *mul = F->createMul("mul", tanh, reshape2);
  auto *sigmoid2 = F->createSigmoid("sigmoid2", mul);
  auto *conv2 = F->createConv(bindings, "conv2", sigmoid2, 7, 3, 2, 1, 1);
  bindings.get(cast<Placeholder>(conv2->getFilter()))->getHandle().clear(0.3);
  bindings.get(cast<Placeholder>(conv2->getBias()))->getHandle().clear(1.3);
  auto *reshape3 = F->createReshape("reshape3", conv2, {8, 8, 7, 4});
  auto *sub = F->createSub("sub", reshape3, var4);
  auto *relu2 = F->createRELU("relu2", sub);
  auto *pool2 = F->createAvgPool("pool2", relu2, 3, 2, 1);
  auto *sigmoid3 = F->createSigmoid("sigmoid3", pool2);
  auto *result = F->createSave("ret", sigmoid3);
  auto *resultTensor = bindings.allocate(result->getPlaceholder());

  EE.compile(CompilationMode::Infer, F);

  updateInputPlaceholders(bindings, {var1, var2, var3, var4},
                          {inputs1, inputs2, inputs3, inputs4});
  EE.run(bindings);
  out->assign(resultTensor);
}

namespace {
// Helper for initializing conv node filter/bias from input tensors.
static void initConv(PlaceholderBindings &bindings, ConvolutionNode *C,
                     Tensor &filter, Tensor &bias) {
  bindings.get(cast<Placeholder>(C->getFilter()))->assign(&filter);
  bindings.get(cast<Placeholder>(C->getBias()))->assign(&bias);
}
} // namespace

void inferTinyResnet(Tensor *input, Tensor *out, std::vector<Tensor> &weights,
                     llvm::StringRef kind) {
  PlaceholderBindings bindings;
  ExecutionEngine EE(kind);
  auto &mod = EE.getModule();
  auto *F = mod.createFunction("main");

  auto *in = createPlaceholder(mod, bindings, input, "in");
  auto *conv1 = F->createConv(bindings, "conv1", in, 256, 1, 1, 0, 1);
  auto *conv2a = F->createConv(bindings, "conv2a", conv1, 64, 1, 1, 0, 1);
  auto *relu2a = F->createRELU("relu2a", conv2a);
  auto *conv2b = F->createConv(bindings, "conv2b", relu2a, 64, 3, 1, 1, 1);
  auto *relu2b = F->createRELU("relu2b", conv2b);
  auto *conv2c = F->createConv(bindings, "conv2c", relu2b, 256, 1, 1, 0, 1);
  auto *add = F->createAdd("add", conv2c, conv1);
  auto *relu = F->createRELU("res2a_relu", add);
  auto *result = F->createSave("ret", relu);
  auto *resultTensor = bindings.allocate(result->getPlaceholder());

  initConv(bindings, conv1, weights[0], weights[1]);
  initConv(bindings, conv2a, weights[2], weights[3]);
  initConv(bindings, conv2b, weights[4], weights[5]);
  initConv(bindings, conv2c, weights[6], weights[7]);
  convertPlaceholdersToConstants(F, bindings, {in, result->getPlaceholder()});

  EE.compile(CompilationMode::Infer, F);

  updateInputPlaceholders(bindings, {in}, {input});
  EE.run(bindings);
  out->assign(resultTensor);
}

void inferExtract3D(Tensor *input, Tensor *out, llvm::StringRef kind) {
  PlaceholderBindings bindings;
  ExecutionEngine EE(kind);
  auto &mod = EE.getModule();
  auto *F = mod.createFunction("main");

  auto *inputs = createPlaceholder(mod, bindings, input, "inputs");

  auto *x1 = F->createSlice("ex1", inputs, {0, 5, 0}, {1, 100, 100});
  auto *x2 = F->createSlice("ex2", inputs, {1, 5, 0}, {2, 100, 100});
  auto *x3 = F->createSlice("ex3", inputs, {2, 5, 0}, {3, 100, 100});
  auto *x4 = F->createSlice("ex4", inputs, {3, 5, 0}, {4, 100, 100});

  auto *x12 = F->createConcat("x12", {x1, x2}, 1);
  auto *x34 = F->createConcat("x34", {x3, x4}, 1);
  auto *x13 = F->createConcat("x34", {x1, x3}, 1);
  auto *x24 = F->createConcat("x34", {x2, x4}, 1);

  auto *add1 = F->createAdd("add1", x12, x34);
  auto *add2 = F->createAdd("add1", x13, x24);
  auto *add3 = F->createAdd("add1", add1, add2);

  auto *e = F->createSlice("slice", add3, {0, 55, 50}, {1, 150, 100});
  auto *result = F->createSave("ret", e);
  auto *resultTensor = bindings.allocate(result->getPlaceholder());

  EE.compile(CompilationMode::Infer, F);

  updateInputPlaceholders(bindings, {inputs}, {input});
  EE.run(bindings);
  out->assign(resultTensor);
}

void inferMaxSplat(Tensor *input, Tensor *out, llvm::StringRef kind) {
  PlaceholderBindings bindings;
  ExecutionEngine EE(kind);
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  auto T = mod.uniqueType(ElemKind::Int8QTy, input->getType().dims(),
                          2 * input->getType().getScale(),
                          -input->getType().getOffset());
  auto *var = createQuantizedPlaceholder(mod, bindings, input, T->getScale(),
                                         T->getOffset(), "var");
  auto *rescale = F->createRescaleQuantized("rescale", var, T);

  auto *splat1 = F->createSplat("splat1", T, 0.0);
  auto *splat2 = F->createSplat("splat2", T, 5.0);

  auto *max1 = F->createMax("max1", rescale, splat1);
  auto *max2 = F->createMax("max2", splat2, max1);

  auto *result = F->createSave("ret", max2);
  auto *resultTensor = bindings.allocate(result->getPlaceholder());

  EE.compile(CompilationMode::Infer, F);

  updateInputPlaceholders(bindings, {var}, {input});
  EE.run(bindings);
  out->assign(resultTensor);
}

} // namespace glow
