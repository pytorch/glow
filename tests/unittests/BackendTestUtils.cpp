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

#include <future>

namespace glow {

llvm::cl::OptionCategory backendTestUtilsCat("BackendTestUtils Category");

unsigned parCloneCountOpt;
llvm::cl::opt<unsigned, /* ExternalStorage */ true> parCloneCountI(
    "parallel-clone-count",
    llvm::cl::desc(
        "Number of times to clone a graph in parallel. Intended to stress test "
        "different backends. This option is not used by all unit "
        "tests; for now you must check the test to see if so."),
    llvm::cl::location(parCloneCountOpt), llvm::cl::Optional, llvm::cl::init(1),
    llvm::cl::cat(backendTestUtilsCat));

bool runDisabledTests;
llvm::cl::opt<bool, /* ExternalStorage */ true> runDisabledTestsI(
    "run-disabled-tests",
    llvm::cl::desc("If set, disabled tests will not be skipped."),
    llvm::cl::location(runDisabledTests), llvm::cl::Optional,
    llvm::cl::init(false), llvm::cl::cat(backendTestUtilsCat));

using llvm::cast;

namespace {

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

/// Create and initialize a function using the argument \p createAndInitFunction
/// then run the function in profiling mode to get the profiling parameters.
/// \p count is the number of times to clone the Function inside itself before
/// profiling. \returns the profiling parameters for all the function nodes.
static std::vector<NodeProfilingInfo>
profileAndGetNodeProfilingInfo(CreateAndInitFunction createAndInitFunction,
                               unsigned count) {
  LoweredInfoMap loweredMapForProf;
  PlaceholderBindings pBindings;
  // Note: deviceMemory = 0 is a signal to use the defaultMemory.
  ExecutionEngine PEE{"Interpreter", /* deviceMemory */ 0,
                      /* ignoreUserDeviceConfig */ true};
  auto FT = createAndInitFunction(pBindings, PEE);
  CompilationContext cctx{&pBindings, &loweredMapForProf};

  // Clone the number of times as requested to match the Function that will be
  // quantized.
  cloneFunInsideFun(FT, &pBindings, cctx, count);
  cctx.precisionConfig.quantMode = QuantizationMode::Profile;
  PEE.compile(cctx);
  PEE.run(pBindings);

  // We get the new function using front() because the original function was
  // deleted as part of the Partitioner quantization flow.
  return quantization::generateNodeProfilingInfos(
      pBindings, PEE.getModule().getFunctions().front(), loweredMapForProf);
}

/// Helper that sets up and \returns a pair of configs for both interpreter and
/// backend being tested.
static std::pair<CompilationContext, CompilationContext>
setupInterpAndBackendConfigs(
    Function *IF, ExecutionEngine &IEE, PlaceholderBindings &iBindings,
    LoweredInfoMap &ILIM, PlaceholderBindings &bBindings, LoweredInfoMap &BLIM,
    ElemKind interpElemKind, ElemKind backendElemKind,
    quantization::Schema schema, bool convertToRowwiseQuantization,
    CreateAndInitFunction createAndInitFunction, ElemKind biasElemKind,
    bool forceFP16AccumSLS, PrecisionConfiguration::Float16Format float16Format,
    unsigned count, bool convertToChannelwiseQuantization,
    bool skipQuantizeFCBias) {
  CompilationContext cctxI{&iBindings, &ILIM};
  CompilationContext cctxB{&bBindings, &BLIM};
  PrecisionConfiguration &precConfigI = cctxI.precisionConfig;
  PrecisionConfiguration &precConfigB = cctxB.precisionConfig;

  if (isQuantizedElemKind(interpElemKind) ||
      isQuantizedElemKind(backendElemKind)) {
    // If either interp or backend need to be quantized then we need to profile
    // and get quantization infos.
    if (isQuantizedElemKind(interpElemKind)) {
      // Note: We only do parallel cloning for the backend, so always use count
      // of 1 here.
      auto NQII =
          profileAndGetNodeProfilingInfo(createAndInitFunction, /* count */ 1);

      precConfigI.quantMode = QuantizationMode::Quantize;
      precConfigI.quantConfig.infos = NQII;
      precConfigI.quantConfig.enableRowwise = convertToRowwiseQuantization;
      precConfigI.quantConfig.enableChannelwise =
          convertToChannelwiseQuantization;
      precConfigI.quantConfig.schema = schema;
      precConfigI.quantConfig.precision = interpElemKind;
      precConfigI.quantConfig.assertAllNodesQuantized = true;
      precConfigI.quantConfig.precisionBias = biasElemKind;
      precConfigI.quantConfig.skipQuantizeFCBias = skipQuantizeFCBias;
    }

    if (isQuantizedElemKind(backendElemKind)) {
      // Always clone count times here. This matches the Function the backend
      // will quantize.
      auto NQIB = profileAndGetNodeProfilingInfo(createAndInitFunction, count);

      precConfigB.quantMode = QuantizationMode::Quantize;
      precConfigB.quantConfig.infos = NQIB;
      precConfigB.quantConfig.enableRowwise = convertToRowwiseQuantization;
      precConfigB.quantConfig.enableChannelwise =
          convertToChannelwiseQuantization;
      precConfigB.quantConfig.schema = schema;
      precConfigB.quantConfig.precision = backendElemKind;
      precConfigB.quantConfig.assertAllNodesQuantized = true;
      precConfigB.quantConfig.precisionBias = biasElemKind;
      precConfigB.quantConfig.skipQuantizeFCBias = skipQuantizeFCBias;
    }
  }

  // For now if the ElemKind is FP16 then we use Float16Ty, UInt8FusedFP16QTy.
  precConfigI.convertToFP16 = interpElemKind == ElemKind::Float16Ty;
  precConfigI.convertFusedToFP16 = interpElemKind == ElemKind::Float16Ty;
  precConfigI.forceFP16AccumSLS = forceFP16AccumSLS;
  precConfigB.convertToFP16 = backendElemKind == ElemKind::Float16Ty;
  precConfigB.convertFusedToFP16 = backendElemKind == ElemKind::Float16Ty;
  precConfigB.forceFP16AccumSLS = forceFP16AccumSLS;

  return std::make_pair(cctxI, cctxB);
}
} // namespace

void dispatchInference(const std::string &fname,
                       runtime::HostManager *hostManager,
                       ExecutionContext &context,
                       unsigned concurrentRequestsOpt,
                       bool useNewExecutionContext) {
  // If additional requests are desired, setup additional contexts.
  std::vector<std::unique_ptr<ExecutionContext>> contexts;
  std::unique_ptr<ExecutionContext> originalContextPtr(&context);
  contexts.push_back(std::move(originalContextPtr));
  if (concurrentRequestsOpt > 1) {
    // Clone the placeholder bindings into a new executionContext.
    for (unsigned i = 0, max = concurrentRequestsOpt - 1; i < max; i++) {
      std::unique_ptr<ExecutionContext> newContext =
          (useNewExecutionContext)
              ? glow::make_unique<ExecutionContext>()
              : glow::make_unique<ExecutionContext>(
                    glow::make_unique<PlaceholderBindings>(
                        context.getPlaceholderBindings()->clone()));
      contexts.push_back(std::move(newContext));
    }
  }
  std::vector<std::promise<void>> promises(concurrentRequestsOpt);
  std::vector<std::future<void>> futures;
  for (auto &promise : promises) {
    futures.push_back(promise.get_future());
  }
  for (unsigned i = 0; i < concurrentRequestsOpt; i++) {
    hostManager->runNetwork(fname, std::move(contexts[i]),
                            [&contexts, &promises,
                             i](runtime::RunIdentifierTy, Error err,
                                std::unique_ptr<ExecutionContext> contextPtr) {
                              contexts[i] = std::move(contextPtr);
                              // Expect no errors.
                              EXIT_ON_ERR(std::move(err));
                              promises[i].set_value();
                            });
  }

  for (auto &future : futures) {
    future.wait();
  }
  for (auto &c : contexts) {
    c->getPlaceholderBindings()->ensureOnHost();
  }
  // Release the original context passed in by reference so we don't free it.
  contexts[0].release();
}

/// Helper that iterates over all of the Placeholders from the function \p F
/// and converts the Tensors found in \p bindings to the same type as the
/// Placeholders if necessary.
static void convertBindingsToCorrectType(Function *F,
                                         PlaceholderBindings &bindings) {
  PlaceholderList PHs = F->findPlaceholders();
  for (Placeholder *PH : PHs) {
    Tensor *T = bindings.get(PH);
    TypeRef newTy = PH->getType();
    if (T->getType().isEqual(newTy)) {
      continue;
    }
    // For input placeholders convert tensor type and values.
    // For output placeholders convert only the tensor type.
    if (isInput(PH, *F)) {
      ElemKind newK = newTy->getElementType();
      if (isQuantizedElemKind(newK)) {
        Tensor QT = quantization::quantizeTensor(
            *T, {newTy->getScale(), newTy->getOffset()}, newK);
        T->assign(&QT);
      } else {
        T->convertToType(newK);
      }
    } else {
      T->reset(*newTy);
    }
  }
}

/// Helper to get a float copy of a Tensor \p T if needed.
static Tensor convertToFloatIfNecessary(Tensor &T) {
  const ElemKind srcK = T.getType().getElementType();
  if (srcK == ElemKind::FloatTy) {
    return T.clone();
  }
  if (isQuantizedElemKind(srcK)) {
    return quantization::dequantizeTensor(T, ElemKind::FloatTy);
  }
  return T.getCopyConvertedToType(ElemKind::FloatTy);
}

void compareAgainstInterpreter(
    llvm::StringRef backendName, CreateAndInitFunction createAndInitFunction,
    ElemKind interpElemKind, ElemKind backendElemKind, float allowedError,
    unsigned count, bool convertToRowwiseQuantization,
    quantization::Schema schema, ElemKind biasElemKind, bool forceFP16AccumSLS,
    PrecisionConfiguration::Float16Format float16Format,
    bool convertToChannelwiseQuantization, bool skipQuantizeFCBias) {
  // Note: deviceMemory = 0 is a signal to use the defaultMemory.
  ExecutionEngine IEE{"Interpreter", /* deviceMemory */ 0,
                      /* ignoreUserDeviceConfig */ true};
  ExecutionEngine BEE{backendName};
  PlaceholderBindings iBindings, bBindings;

  LOG(INFO) << "Comparing Interpreter with precision "
            << Type::getElementName(interpElemKind).str() << " against "
            << backendName.str() << " with precision "
            << Type::getElementName(backendElemKind).str() << " with Bias "
            << (skipQuantizeFCBias ? "unquantized"
                                   : Type::getElementName(biasElemKind).str())
            << " with FP16 AccumulationSLS " << forceFP16AccumSLS;

  // Create the same network on the interpreter and the backend being tested.
  FunctionTensorPair IFT = createAndInitFunction(iBindings, IEE);
  FunctionTensorPair BFT = createAndInitFunction(bBindings, BEE);

  Function *IF = IFT.first;

  // Set up the configs for interpreter and backend. If one or both functions
  // will be quantized, then gather a profile the graph on the interpreter, and
  // then quantize the Functions as requested.
  LoweredInfoMap ILIM, BLIM;
  auto configs = setupInterpAndBackendConfigs(
      IF, IEE, iBindings, ILIM, bBindings, BLIM, interpElemKind,
      backendElemKind, schema, convertToRowwiseQuantization,
      createAndInitFunction, biasElemKind, forceFP16AccumSLS, float16Format,
      count, convertToChannelwiseQuantization, skipQuantizeFCBias);
  CompilationContext &cctxI = configs.first;
  CompilationContext &cctxB = configs.second;

  // Skip conversion for rowwise quantized tests as they are a special case
  // which don't fit cleanly here -- e.g. RWQ-SLS has FloatTy outputs.
  if (!convertToRowwiseQuantization) {
    // We want to compare the ops themselves and not see differences in
    // conversion, so fold ElemKind conversion nodes into IO.
    cctxI.optimizationOpts.foldElemKindConversionIntoIO = true;
    cctxB.optimizationOpts.foldElemKindConversionIntoIO = true;
  }

  // Clone the Function inside itself many times if desired.
  std::unordered_set<Tensor *> resultTensors =
      cloneFunInsideFun(BFT, &bBindings, cctxB, count);
  assert(resultTensors.size() == count &&
         "Should get the same number of Tensors back as count.");

  IEE.compile(cctxI);
  BEE.compile(cctxB);

  // Again skip rowwise quantization as before.
  if (!convertToRowwiseQuantization) {
    // Now that we have compiled, precision transformation has occurred. Now
    // convert all mismatches for Placeholders given their original bindings.
    convertBindingsToCorrectType(IEE.getSingleFunctionFromModule(), iBindings);
    convertBindingsToCorrectType(BEE.getSingleFunctionFromModule(), bBindings);
  }

  IEE.run(iBindings);
  BEE.run(bBindings);

  // Compare each of our result tensors to the original. Always convert back to
  // float if necessary, as allowed error is expected to compare float.
  Tensor finalIT = convertToFloatIfNecessary(*IFT.second);
  for (Tensor *T : resultTensors) {
    Tensor finalBT = convertToFloatIfNecessary(*T);
    EXPECT_TRUE(finalIT.isEqual(finalBT, allowedError, /* verbose */ true));
  }

  // Additionally check that each of the results from the parallel cloned
  // Functions are bitwise equal.
  auto it = resultTensors.begin();
  Tensor *firstResult = *it;
  for (it++; it != resultTensors.end(); it++) {
    EXPECT_TRUE(firstResult->isBitwiseEqual(**it));
  }
}

std::unordered_set<Tensor *> cloneFunInsideFun(FunctionTensorPair FTP,
                                               PlaceholderBindings *bindings,
                                               CompilationContext &cctx,
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
    mod->eraseFunction(tmpF);
  }
  // Now erase the clone we used to copy in, as it's no longer needed.
  mod->eraseFunction(cloneF);

  // Finally, duplicate all of the node profiling infos with the new expected
  // clone's name so that the cloned copies will find the same profiling info
  // as the original node if being quantized.
  auto &origInfos = cctx.precisionConfig.quantConfig.infos;
  origInfos.reserve(count * origInfos.size());
  std::vector<NodeProfilingInfo> newInfos;
  newInfos.reserve((count - 1) * origInfos.size());
  for (const auto &PI : origInfos) {
    const size_t colonIdx = PI.nodeOutputName_.find(":");
    assert(colonIdx != std::string::npos && "Name should always contain ':'");
    for (size_t i = 1; i < count; i++) {
      std::string newName(PI.nodeOutputName_);
      // Cloned nodes end up with the original name plus the count number
      // appended to their name due to uniquing. Replicate the same thing.
      newName.insert(colonIdx, std::to_string(i));
      newInfos.emplace_back(newName, PI.tensorProfilingParams_);
    }
  }
  origInfos.insert(origInfos.end(), newInfos.begin(), newInfos.end());

  return resultTensors;
}

unsigned countNodeKind(Function *F, Kinded::Kind kind) {
  unsigned count = 0;
  for (auto &n : F->getNodes()) {
    if (n.getKind() == kind) {
      count++;
    }
  }
  return count;
}

void inferIntLookupTableNetInt8(Tensor *input, Tensor *out,
                                llvm::ArrayRef<int8_t> table,
                                llvm::StringRef kind) {
  PlaceholderBindings bindings;
  ExecutionEngine EE(kind);
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  auto outTy = mod.uniqueType(ElemKind::Int8QTy, {(dim_t)input->size()}, 3, 3);
  auto var = createQuantizedPlaceholder(mod, bindings, input,
                                        input->getType().getScale(),
                                        input->getType().getOffset(), "var");
  auto *lookupTable = F->createIntLookupTable("lookuptable", var, table, outTy);
  auto *result = F->createSave("ret", lookupTable);
  auto *resultTensor = bindings.allocate(result->getPlaceholder());

  EE.compile(CompilationMode::Infer);
  bindings.allocate(mod.getPlaceholders());

  updateInputPlaceholders(bindings, {var}, {input});
  EE.run(bindings);
  out->assign(resultTensor);
}

void inferIntLookupTableNetInt16(Tensor *input, Tensor *out,
                                 llvm::ArrayRef<int16_t> table,
                                 llvm::StringRef kind) {
  PlaceholderBindings bindings;
  ExecutionEngine EE(kind);
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  auto outTy = mod.uniqueType(ElemKind::Int16QTy, {(dim_t)input->size()}, 3, 3);
  auto var = createQuantizedPlaceholder(mod, bindings, input,
                                        input->getType().getScale(),
                                        input->getType().getOffset(), "var");
  auto *lookupTable = F->createIntLookupTable("lookuptable", var, table, outTy);
  auto *result = F->createSave("ret", lookupTable);
  auto *resultTensor = bindings.allocate(result->getPlaceholder());

  EE.compile(CompilationMode::Infer);
  bindings.allocate(mod.getPlaceholders());

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
    auto &inType = inputs->getType();
    auto &filterType = filter->getType();
    auto &biasType = bias->getType();
    inputP = createQuantizedPlaceholder(
        mod, bindings, inputs, inType.getScale(), inType.getOffset(), "inputP");
    filterP =
        createQuantizedPlaceholder(mod, bindings, filter, filterType.getScale(),
                                   filterType.getOffset(), "filterP");
    biasP = createQuantizedPlaceholder(mod, bindings, bias, biasType.getScale(),
                                       biasType.getOffset(), "biasP");
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

  EE.compile(CompilationMode::Infer);

  updateInputPlaceholders(bindings, {inputP, filterP, biasP},
                          {inputs, filter, bias});
  EE.run(bindings);
  out->assign(resultTensor);
}

int inferConvReluNet(Tensor *inputs, Tensor *filter, Tensor *bias, Tensor *out,
                     unsigned_t kernel, unsigned_t stride, unsigned_t pad,
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
    auto &inType = inputs->getType();
    auto &filterType = filter->getType();
    auto &biasType = bias->getType();
    inputP = createQuantizedPlaceholder(
        mod, bindings, inputs, inType.getScale(), inType.getOffset(), "inputP");
    filterP =
        createQuantizedPlaceholder(mod, bindings, filter, filterType.getScale(),
                                   filterType.getOffset(), "filterP");
    biasP = createQuantizedPlaceholder(mod, bindings, bias, biasType.getScale(),
                                       biasType.getOffset(), "biasP");
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
  auto *conv =
      F->createConv("conv", inputP, filterP, biasP, OT, kernel, stride, pad, 1);
  // Relu
  auto *relu = F->createRELU("relu", conv);
  auto *result = F->createSave("ret", relu, outP);
  auto *resultTensor = bindings.get(result->getPlaceholder());

  EE.compile(CompilationMode::Infer);

  // check fusion depending on build option.
  // EXPECT_EQ(conv->getFusedActivation(), FusedActivation::RELU);

  updateInputPlaceholders(bindings, {inputP, filterP, biasP},
                          {inputs, filter, bias});
  EE.run(bindings);
  out->assign(resultTensor);
  return conv->getFusedActivation();
}

void trainConvNet(Tensor *inputs, Tensor *kernel1, Tensor *bias1,
                  Tensor *kernel2, Tensor *bias2, Tensor *selected,
                  llvm::ArrayRef<dim_t> shape1, llvm::ArrayRef<dim_t> shape2,
                  Tensor *out, llvm::StringRef kind) {
  ExecutionEngine EET(kind);
  ExecutionEngine EEI(kind);
  std::vector<ExecutionEngine *> engines;
  engines.push_back(&EEI);
  engines.push_back(&EET);
  TrainingConfig TC;
  PlaceholderBindings bindings, inferBindings, trainingBindings;

  // This variable records the number of the next sample to be used for
  // training.
  size_t sampleCounter = 0;

  TC.learningRate = 0.03;
  TC.momentum = 0.3;
  TC.L2Decay = 0.01;
  Function *F;
  Placeholder *var1, *var2;
  for (auto *EE : engines) {
    auto &mod = EE->getModule();
    F = mod.createFunction("main");
    var1 = createPlaceholder(mod, bindings, inputs, "var1");
    var2 = createPlaceholder(mod, bindings, selected, "var2");
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
    F->createSave("ret", softmax);
  }

  auto *TF = glow::differentiate(F, TC);
  auto tfName = TF->getName();
  auto fName = F->getName();
  EET.compile(CompilationMode::Train);
  trainingBindings.allocate(EET.getModule().getPlaceholders());
  inferBindings.allocate(EEI.getModule().getPlaceholders());
  bindings.copyTrainableWeightsTo(trainingBindings);
  auto *res =
      inferBindings.get(EEI.getModule().getPlaceholderByNameSlow("ret"));

  runBatch(EET, trainingBindings, 8, sampleCounter, {var1, var2},
           {inputs, selected}, tfName);
  trainingBindings.copyTrainableWeightsTo(inferBindings);
  EEI.compile(CompilationMode::Infer);
  var1 = inferBindings.getPlaceholderByNameSlow("var1");
  var2 = inferBindings.getPlaceholderByNameSlow("var2");
  updateInputPlaceholders(inferBindings, {var1, var2}, {inputs, selected});
  EEI.run(inferBindings, fName);
  out->assign(res);
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

  EE.compile(CompilationMode::Infer);

  updateInputPlaceholders(bindings, {var}, {inputs});
  EE.run(bindings);
  out->assign(resultTensor);
}

void trainLocalResponseNormalizationNet(Tensor *inputs, Tensor *weights,
                                        Tensor *bias, Tensor *selected,
                                        llvm::ArrayRef<dim_t> shape1,
                                        llvm::ArrayRef<dim_t> shape2,
                                        Tensor *out, llvm::StringRef kind) {
  PlaceholderBindings bindings, trainingBindings;
  ExecutionEngine EET(kind);
  ExecutionEngine EEI(kind);
  std::vector<ExecutionEngine *> engines{&EEI, &EET};
  TrainingConfig TC;

  // This variable records the number of the next sample to be used for
  // training.
  size_t sampleCounter = 0;

  TC.learningRate = 0.06;
  TC.momentum = 0.1;
  TC.L2Decay = 0.01;
  Placeholder *var1, *var2;
  std::string fName;
  for (auto *EE : engines) {
    auto &mod = EE->getModule();
    Function *F = mod.createFunction("main");
    fName = F->getName().str();
    var1 = createPlaceholder(mod, bindings, inputs, "var1");
    var2 = createPlaceholder(mod, bindings, selected, "var2");
    auto *fc = F->createFullyConnected(bindings, "fc", var1, bias->dims()[0]);
    bindings.get(cast<Placeholder>(fc->getWeights()))->assign(weights);
    bindings.get(cast<Placeholder>(fc->getBias()))->assign(bias);
    auto *reshape1 = F->createReshape("reshape1", fc, shape1);
    auto *lrn =
        F->createLocalResponseNormalization("lrn", reshape1, 2, 2.0, 0.5, 1.0);
    auto *reshape2 = F->createReshape("reshape2", lrn, shape2);
    auto *softmax = F->createSoftMax("softmax", reshape2, var2);
    auto *result = F->createSave("ret", softmax);
    bindings.allocate(result->getPlaceholder());
  }
  auto *TF = glow::differentiate(EET.getModule().getFunction(fName), TC);
  auto tfName = TF->getName();
  EET.compile(CompilationMode::Train);
  trainingBindings.allocate(EET.getModule().getPlaceholders());
  bindings.copyTrainableWeightsTo(trainingBindings);
  bindings.clear();
  bindings.allocate(EEI.getModule().getPlaceholders());

  runBatch(EET, trainingBindings, 8, sampleCounter, {var1, var2},
           {inputs, selected}, tfName);
  trainingBindings.copyTrainableWeightsTo(bindings);
  var1 = bindings.getPlaceholderByNameSlow("var1");
  var2 = bindings.getPlaceholderByNameSlow("var2");
  EEI.compile(CompilationMode::Infer);

  runBatch(EEI, bindings, 1, sampleCounter, {var1, var2}, {inputs, selected});
  out->assign(bindings.get(bindings.getPlaceholderByNameSlow("ret")));
}

void trainAvgPoolNet(Tensor *inputs, Tensor *weights, Tensor *bias,
                     Tensor *selected, llvm::ArrayRef<dim_t> shape1,
                     llvm::ArrayRef<dim_t> shape2, Tensor *out,
                     llvm::StringRef kind) {
  ExecutionEngine EET(kind);
  ExecutionEngine EEI(kind);
  std::vector<ExecutionEngine *> engines{&EEI, &EET};
  TrainingConfig TC;
  PlaceholderBindings bindings, trainingBindings;

  // This variable records the number of the next sample to be used for
  // training.
  size_t sampleCounter = 0;

  TC.learningRate = 0.01;
  TC.momentum = 0.4;
  TC.L2Decay = 0.01;
  Placeholder *var1, *var2;
  std::string fName;
  for (auto *EE : engines) {
    auto &mod = EE->getModule();
    Function *F = mod.createFunction("main");
    fName = F->getName().str();
    var1 = createPlaceholder(mod, bindings, inputs, "var1");
    var2 = createPlaceholder(mod, bindings, selected, "var2");
    auto *fc = F->createFullyConnected(bindings, "fc", var1, bias->dims()[0]);
    bindings.get(cast<Placeholder>(fc->getWeights()))->assign(weights);
    bindings.get(cast<Placeholder>(fc->getBias()))->assign(bias);
    auto *reshape1 = F->createReshape("reshape1", fc, shape1);
    auto *pool = F->createAvgPool("pool", reshape1, 2, 2, 0);
    auto *reshape2 = F->createReshape("reshape2", pool, shape2);
    auto *softmax = F->createSoftMax("softmax", reshape2, var2);
    auto *result = F->createSave("ret", softmax);
    bindings.allocate(result->getPlaceholder());
  }
  auto *TF = glow::differentiate(EET.getModule().getFunction("main"), TC);
  auto tfName = TF->getName();
  EET.compile(CompilationMode::Train);
  trainingBindings.allocate(EET.getModule().getPlaceholders());
  bindings.copyTrainableWeightsTo(trainingBindings);
  bindings.clear();
  bindings.allocate(EEI.getModule().getPlaceholders());

  runBatch(EET, trainingBindings, 10, sampleCounter, {var1, var2},
           {inputs, selected}, tfName);
  trainingBindings.copyTrainableWeightsTo(bindings);
  var1 = bindings.getPlaceholderByNameSlow("var1");
  var2 = bindings.getPlaceholderByNameSlow("var2");
  EEI.compile(CompilationMode::Infer);

  updateInputPlaceholders(bindings, {var1, var2}, {inputs, selected});
  EEI.run(bindings);
  out->assign(bindings.get(bindings.getPlaceholderByNameSlow("ret")));
}

void trainMaxPoolNet(Tensor *inputs, Tensor *weights, Tensor *bias,
                     Tensor *selected, llvm::ArrayRef<dim_t> shape1,
                     llvm::ArrayRef<dim_t> shape2, Tensor *out,
                     llvm::StringRef kind) {
  ExecutionEngine EET(kind);
  ExecutionEngine EEI(kind);
  std::vector<ExecutionEngine *> engines;
  engines.push_back(&EEI);
  engines.push_back(&EET);
  TrainingConfig TC;
  PlaceholderBindings bindings, inferBindings, trainingBindings;

  // This variable records the number of the next sample to be used for
  // training.
  size_t sampleCounter = 0;

  TC.learningRate = 0.03;
  TC.momentum = 0.3;
  TC.L2Decay = 0.003;
  Function *F;
  Placeholder *var1, *var2;
  for (auto *EE : engines) {
    bindings.clear();
    auto &mod = EE->getModule();
    F = mod.createFunction("main");
    var1 = createPlaceholder(mod, bindings, inputs, "var1");
    var2 = createPlaceholder(mod, bindings, selected, "var2");
    auto *fc = F->createFullyConnected(bindings, "fc", var1, bias->dims()[0]);
    bindings.get(cast<Placeholder>(fc->getWeights()))->assign(weights);
    bindings.get(cast<Placeholder>(fc->getBias()))->assign(bias);
    auto *reshape1 = F->createReshape("reshape1", fc, shape1);
    auto *pool = F->createMaxPool("pool", reshape1, 5, 3, 4);
    auto *reshape2 = F->createReshape("reshape2", pool->getResult(), shape2);
    auto *softmax = F->createSoftMax("softmax", reshape2, var2);
    F->createSave("ret", softmax);
  }
  auto *TF = glow::differentiate(F, TC);
  auto fName = F->getName();
  auto tfName = TF->getName();
  EET.compile(CompilationMode::Train);
  trainingBindings.allocate(EET.getModule().getPlaceholders());
  inferBindings.allocate(EEI.getModule().getPlaceholders());
  bindings.copyTrainableWeightsTo(trainingBindings);
  auto *res =
      inferBindings.get(EEI.getModule().getPlaceholderByNameSlow("ret"));

  runBatch(EET, trainingBindings, 7, sampleCounter, {var1, var2},
           {inputs, selected}, tfName);
  trainingBindings.copyTrainableWeightsTo(inferBindings);
  EEI.compile(CompilationMode::Infer);
  var1 = inferBindings.getPlaceholderByNameSlow("var1");
  var2 = inferBindings.getPlaceholderByNameSlow("var2");
  runBatch(EEI, inferBindings, 1, sampleCounter, {var1, var2},
           {inputs, selected}, fName);
  out->assign(res);
}

void inferSmallConv(Tensor *inputs, Tensor *out, llvm::StringRef kind) {
  PlaceholderBindings bindings;
  ExecutionEngine EE(kind);
  auto &mod = EE.getModule();
  auto *F = mod.createFunction("main");
  auto *in = createPlaceholder(mod, bindings, inputs, "in", "NHWC");
  auto *C = F->createConv(bindings, "conv2a", in, 64, 1, 1, 0, 1);
  bindings.get(cast<Placeholder>(C->getFilter()))->getHandle().clear(0.3);
  bindings.get(cast<Placeholder>(C->getBias()))->getHandle().clear(0.4);
  auto *result = F->createSave("ret", C);
  auto *resultTensor = bindings.allocate(result->getPlaceholder());
  convertPlaceholdersToConstants(F, bindings, {in, result->getPlaceholder()});

  EE.compile(CompilationMode::Infer);

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
  for (dim_t i = 0; i < 128; i++)
    for (dim_t j = 0; j < 16; j++) {
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

  EE.compile(CompilationMode::Infer);

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
  for (dim_t i = 0; i < 128; i++)
    for (dim_t j = 0; j < 32; j++) {
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

  EE.compile(CompilationMode::Infer);

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
  for (dim_t i = 0; i < 128; i++)
    for (dim_t j = 0; j < 2; j++)
      for (dim_t k = 0; k < 32; k++) {
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

  EE.compile(CompilationMode::Infer);

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
  for (dim_t i = 0; i < 128; i++)
    for (dim_t j = 0; j < 2; j++)
      for (dim_t k = 0; k < 32; k++) {
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

  EE.compile(CompilationMode::Infer);

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
  for (dim_t i = 0; i < 192; i++)
    for (dim_t j = 0; j < 3; j++)
      for (dim_t k = 0; k < 3; k++)
        for (dim_t l = 0; l < 32; l++) {
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

  EE.compile(CompilationMode::Infer);

  EE.run(bindings);
  out->assign(resultTensor);
}

void trainSoftMaxNet(Tensor *inputs, Tensor *weights, Tensor *bias,
                     Tensor *selected, Tensor *out, llvm::StringRef kind) {
  ExecutionEngine EEI(kind);
  ExecutionEngine EET(kind);
  std::vector<ExecutionEngine *> engines;
  engines.push_back(&EEI);
  engines.push_back(&EET);
  TrainingConfig TC;
  PlaceholderBindings bindings, inferBindings, trainingBindings;

  // This variable records the number of the next sample to be used for
  // training.
  size_t sampleCounter = 0;

  TC.learningRate = 0.003;
  TC.momentum = 0.7;
  TC.L2Decay = 0.001;
  Function *F;
  Placeholder *var1, *var2;
  for (auto *EE : engines) {
    auto &mod = EE->getModule();
    F = mod.createFunction("main");
    var1 = createPlaceholder(mod, bindings, inputs, "var1");
    var2 = createPlaceholder(mod, bindings, selected, "var2");
    auto *fc = F->createFullyConnected(bindings, "fc", var1, bias->dims()[0]);
    bindings.get(cast<Placeholder>(fc->getWeights()))->assign(weights);
    bindings.get(cast<Placeholder>(fc->getBias()))->assign(bias);
    auto *softmax = F->createSoftMax("softmax", fc, var2);
    F->createSave("ret", softmax);
  }

  auto *TF = glow::differentiate(F, TC);
  auto tfName = TF->getName();
  auto fName = F->getName();

  EET.compile(CompilationMode::Train);
  trainingBindings.allocate(EET.getModule().getPlaceholders());
  bindings.copyTrainableWeightsTo(trainingBindings);
  runBatch(EET, trainingBindings, 30, sampleCounter, {var1, var2},
           {inputs, selected}, tfName);
  EEI.compile(CompilationMode::Infer);
  inferBindings.allocate(EEI.getModule().getPlaceholders());
  trainingBindings.copyTrainableWeightsTo(inferBindings);
  auto *res =
      inferBindings.get(EEI.getModule().getPlaceholderByNameSlow("ret"));
  var1 = inferBindings.getPlaceholderByNameSlow("var1");
  var2 = inferBindings.getPlaceholderByNameSlow("var2");
  updateInputPlaceholders(inferBindings, {var1, var2}, {inputs, selected});
  EEI.run(inferBindings, fName);
  out->assign(res);
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

  EE.compile(CompilationMode::Infer);

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
  auto *var = createPlaceholder(mod, bindings, inputs, "var", "NCHW");
  auto *tr = F->createTranspose("tr", var, NCHW2NHWC);
  auto *conv = F->createConv(bindings, "conv", tr, convDepth, {5, 5}, {2, 2},
                             {1, 1, 1, 1}, 1);
  bindings.get(cast<Placeholder>(conv->getFilter()))->getHandle().clear(0.1);
  bindings.get(cast<Placeholder>(conv->getBias()))->getHandle().clear(0.2);
  auto *pool = F->createMaxPool("pool", conv, 2, 2, 0);
  auto *result = F->createSave("ret", pool->getResult());
  auto *resultTensor = bindings.allocate(result->getPlaceholder());
  convertPlaceholdersToConstants(F, bindings, {var, result->getPlaceholder()});

  EE.compile(CompilationMode::Infer);

  updateInputPlaceholders(bindings, {var}, {inputs});
  EE.run(bindings);
  out->assign(resultTensor);
}

FunctionTensorPair createAndInitBasicFCNet(PlaceholderBindings &bindings,
                                           ExecutionEngine &EE) {
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  auto *var = mod.createPlaceholder(ElemKind::FloatTy, {2, 3, 16, 16}, "var",
                                    false, "NCHW");
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
  auto *var = createPlaceholder(mod, bindings, inputs, "var", "NCHW");
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

  EE.compile(CompilationMode::Infer);

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
  auto *reshape1 = F->createReshape("reshape1", fc1, {8, 14, 28, 6}, "NHWC");
  auto *relu1 = F->createRELU("relu1", reshape1);
  auto *pool1 = F->createMaxPool("pool1", relu1, 2, 2, 1);
  auto *add = F->createAdd("add", sigmoid1, pool1->getResult());
  auto *tanh = F->createTanh("tanh", add);
  auto *fc2 = F->createFullyConnected(bindings, "fc2", var3, 720);
  bindings.get(cast<Placeholder>(fc2->getWeights()))->getHandle().clear(1.1);
  auto *reshape2 = F->createReshape("reshape2", fc2, {8, 8, 15, 6}, "NHWC");
  auto *mul = F->createMul("mul", tanh, reshape2);
  auto *sigmoid2 = F->createSigmoid("sigmoid2", mul);
  auto *conv2 = F->createConv(bindings, "conv2", sigmoid2, 7, 3, 2, 1, 1);
  bindings.get(cast<Placeholder>(conv2->getFilter()))->getHandle().clear(0.3);
  bindings.get(cast<Placeholder>(conv2->getBias()))->getHandle().clear(1.3);
  auto *reshape3 = F->createReshape("reshape3", conv2, {8, 8, 7, 4}, "NHWC");
  auto *sub = F->createSub("sub", reshape3, var4);
  auto *relu2 = F->createRELU("relu2", sub);
  auto *pool2 = F->createAvgPool("pool2", relu2, 3, 2, 1);
  auto *sigmoid3 = F->createSigmoid("sigmoid3", pool2);
  auto *result = F->createSave("ret", sigmoid3);
  auto *resultTensor = bindings.allocate(result->getPlaceholder());

  EE.compile(CompilationMode::Infer);

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

  auto *in = createPlaceholder(mod, bindings, input, "in", "NHWC");
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

  EE.compile(CompilationMode::Infer);

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

  EE.compile(CompilationMode::Infer);

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
  auto *var = createQuantizedPlaceholder(mod, bindings, input,
                                         input->getType().getScale(),
                                         input->getType().getOffset(), "var");
  auto *rescale = F->createRescaleQuantized("rescale", var, T);

  auto *splat1 = F->createSplat("splat1", T, 0.0);
  auto *splat2 = F->createSplat("splat2", T, 5.0);

  auto *max1 = F->createMax("max1", rescale, splat1);
  auto *max2 = F->createMax("max2", splat2, max1);

  auto *result = F->createSave("ret", max2);
  auto *resultTensor = bindings.allocate(result->getPlaceholder());

  EE.compile(CompilationMode::Infer);

  updateInputPlaceholders(bindings, {var}, {input});
  EE.run(bindings);
  out->assign(resultTensor);
}

void insertCompiledFunction(llvm::StringRef name, CompiledFunction *func,
                            runtime::DeviceManager *device, Module *mod) {
  runtime::FunctionMapTy functionMap;
  functionMap[name.str()] = func;

  std::promise<void> addPromise;
  auto fut = addPromise.get_future();
  Error addErr = Error::empty();
  device->addNetwork(mod, std::move(functionMap),
                     [&addPromise, &addErr](const Module *, Error err) {
                       addErr = std::move(err);
                       addPromise.set_value();
                     });
  fut.wait();
  EXIT_ON_ERR(std::move(addErr));
}

void runOnDevice(ExecutionContext &context, llvm::StringRef name,
                 runtime::DeviceManager *device) {
  std::unique_ptr<ExecutionContext> contextPtr(&context);
  std::promise<void> runPromise;
  auto fut = runPromise.get_future();
  Error runErr = Error::empty();
  device->runFunction(
      name.str(), std::move(contextPtr),
      [&runPromise, &runErr](runtime::RunIdentifierTy, Error err,
                             std::unique_ptr<ExecutionContext> contextPtr) {
        // Don't delete context.
        contextPtr.release();
        runErr = std::move(err);
        runPromise.set_value();
      });
  fut.wait();
  EXIT_ON_ERR(std::move(runErr));
}

Constant *createRandomizedConstant(Module &mod, TypeRef type,
                                   llvm::ArrayRef<dim_t> dims,
                                   llvm::StringRef name) {
  auto *c = mod.createConstant(mod.uniqueTypeWithNewShape(type, dims), name);

  switch (type->getElementType()) {
  case ElemKind::FloatTy: {
    c->getHandle<float>().initXavier(c->getType()->size() * 2, mod.getPRNG());
    break;
  }
  case ElemKind::Float16Ty: {
    c->getHandle<float16_t>().initXavier(c->getType()->size() * 2,
                                         mod.getPRNG());
    break;
  }
  case ElemKind::BFloat16Ty: {
    c->getHandle<bfloat16_t>().initXavier(c->getType()->size() * 2,
                                          mod.getPRNG());
    break;
  }
  case ElemKind::Int32QTy: {
    c->getHandle<int32_t>().randomize(INT32_MIN, INT32_MAX, mod.getPRNG());
    break;
  }
  case ElemKind::Int8QTy: {
    c->getHandle<int8_t>().randomize(INT8_MIN, INT8_MAX, mod.getPRNG());
    break;
  }
  case ElemKind::UInt8FusedQTy:
  case ElemKind::UInt8FusedFP16QTy: {
    c->getHandle<uint8_t>().randomize(UINT8_MIN, UINT8_MAX, mod.getPRNG());
    break;
  }
  default:
    LOG(FATAL) << "Unsupported type: " << type->getElementName().str();
  }

  return c;
}

Constant *createRandomFusedRowwiseQuantizedConstant(Module &mod,
                                                    llvm::ArrayRef<dim_t> dims,
                                                    llvm::StringRef name,
                                                    bool useFusedFP16) {
  auto T = mod.uniqueType(
      (useFusedFP16 ? ElemKind::UInt8FusedFP16QTy : ElemKind::UInt8FusedQTy),
      {1}, 1, 0);
  const dim_t sizeScaleOffset =
      useFusedFP16 ? sizeof(float16_t) : sizeof(float);
  Constant *c = createRandomizedConstant(
      mod, T, {dims[0], dims[1] + 2 * sizeScaleOffset}, name);

  // Range (0, 255) -> (-0.1, 0.1)
  constexpr float scale = 1.0f / 1275;
  constexpr float offset = -0.1;
  auto cH = c->getPayload().getHandle<uint8_t>();
  for (unsigned i = 0, e = c->dims()[0]; i < e; i++) {
    if (useFusedFP16) {
      cH.setFusedScaleOffsetInRow<float16_t>(i, scale, offset);
    } else {
      cH.setFusedScaleOffsetInRow<float>(i, scale, offset);
    }
  }

  return c;
}

Placeholder *createFusedRowwiseQuantizedPlaceholder(Module &mod,
                                                    llvm::ArrayRef<dim_t> dims,
                                                    llvm::StringRef name,
                                                    bool useFusedFP16) {
  auto T = useFusedFP16 ? ElemKind::UInt8FusedFP16QTy : ElemKind::UInt8FusedQTy;
  const dim_t sizeScaleOffset =
      useFusedFP16 ? sizeof(float16_t) : sizeof(float);
  constexpr float scale = 1.0f / 1275;
  constexpr float offset = -0.1;
  Placeholder *ph = mod.createPlaceholder(
      T, {dims[0], dims[1] + 2 * sizeScaleOffset}, scale, offset, name, false);

  return ph;
}

// Helper for creating and intializing placeholders from tensors.
Placeholder *createPlaceholder(Module &mod, PlaceholderBindings &bindings,
                               Tensor *tensor, llvm::StringRef name,
                               const std::string &layout) {
  auto *P = mod.createPlaceholder(&tensor->getType(), name, false, layout);
  auto *PTensor = bindings.allocate(P);
  PTensor->assign(tensor);
  return P;
}

} // namespace glow
