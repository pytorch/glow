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

#include "NetworkComparator.h"

#include "glow/Graph/Graph.h"
#include "glow/Graph/Node.h"

#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Graph/Hook.h"
#include "glow/Graph/Utils.h"
#include "glow/Support/Debug.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/FormatVariadic.h"

#define DEBUG_TYPE "verifier"

using namespace glow;

void NetworkComparatorBase::dumpTensors(
    std::unordered_map<std::string, Tensor *> tensors,
    const std::string &layerName, const std::string &prefix) {
  // TODO: Need to add different flavours of dumping.
  auto it = tensors.begin();
  while (it != tensors.end()) {
    std::error_code EC;
    DCHECK(it->second) << "Tensor is Null\n";
    llvm::raw_fd_ostream fs(prefix + "_" + it->first + "_" + layerName, EC);
    it->second->dump(fs, std::numeric_limits<unsigned>::max());
    fs.close();
    it++;
  }
}
bool NetworkComparatorBase::checkTensors(
    std::unordered_map<std::string, Tensor *> &refOuts,
    std::unordered_map<std::string, Tensor *> &checkOuts) {
  if (refOuts.size() != checkOuts.size()) {
    LOG(ERROR) << "Backends produced different number of results\n";
    return false;
  }
  auto itRef = refOuts.begin();
  auto itCheck = checkOuts.begin();
  auto endRef = refOuts.end();
  auto endCheck = checkOuts.end();
  while (itRef != endRef && itCheck != endCheck) {
    if (!itRef->second->isEqual(*(itCheck->second), numericCmpThreshold_,
                                /* verbose */ true)) {
      LOG(INFO) << "Error: " << itRef->first << "\n";
      return false;
    }
    itRef++;
    itCheck++;
  }
  return true;
}

NetworkComparatorBase::NetworkComparatorBase(
    Module &mod, const std::string &referenceBackend,
    const std::string &testBackend, float numericCmpThreshold,
    bool dumpTensorsForBadLayer)
    : numericCmpThreshold_(numericCmpThreshold), inputModule_(&mod),
      dumpTensorsForBadLayer_(dumpTensorsForBadLayer) {
  DCHECK(mod.getFunctions().size() == 1)
      << "Module must have exactly one functions";
  EERefNet_.setBackendName(referenceBackend);
  mod.clone(&EERefNet_.getModule());
  EETestNet_.setBackendName(testBackend);
  mod.clone(&EETestNet_.getModule());
}

NetworkComparatorBase::InOutTensors RecursiveLayerComparator::hookAndRun(
    llvm::StringRef layerName, PlaceholderBindings *bindings, bool isRef) {
  ExecutionEngine execEngine;
  execEngine.setBackendName(isRef ? EERefNet_.getBackendName()
                                  : EETestNet_.getBackendName());
  inputModule_->clone(&execEngine.getModule());
  HookedFunction hook = hookNode(execEngine.getSingleFunctionFromModule(),
                                 layerName, /* hookInputs */ true);
  PlaceholderBindings &inferBindings =
      isRef ? inferBindingsRef_ : inferBindingsCheck_;
  inferBindings.allocate(execEngine.getModule().getPlaceholders());
  // Copy the tensors from the bindings to the inference bindings
  // we feed to the Temp network.
  for (auto &PH : bindings->pairs()) {
    auto iPH = inferBindings.getPlaceholderByNameSlow(PH.first->getName());
    inferBindings.get(iPH)->assign(&PH.second);
  }
  InOutTensors inOutTensors;
  for (const auto &P : hook.outputs) {
    Tensor *t = inferBindings.get(P);
    std::string str = P->getName().str();
    DCHECK(t) << "Placeholder not " << str << " found in the bindings\n";
    inOutTensors.outputs[str] = t;
  }
  for (const auto &P : hook.inputs) {
    Tensor *t = inferBindings.get(P);
    std::string str = P->getName().str();
    DCHECK(t) << "Placeholder " << str << " not found in the bindings\n";
    inOutTensors.inputs[str] = t;
  }
  auto fName = hook.function->getName();
  execEngine.compile(CompilationMode::Infer);
  execEngine.run(inferBindings, fName);
  return inOutTensors;
}

bool RecursiveLayerComparator::verify(PlaceholderBindings *bindings) {
  bool allPassed = true;
  // Sort the nodes in topological order to test nodes in that order.
  // Tensors flow through the network in topological order, testing the layers
  // in that order will allow us to see how errors propagate.
  GraphPostOrderVisitor visitor(*EERefNet_.getSingleFunctionFromModule());
  llvm::ArrayRef<Node *> order = visitor.getPostOrder();
  for (auto const *nodePtr : order) {
    auto &node = *nodePtr;
    DCHECK(nodePtr) << "Node is empty!";
    if (llvm::isa<SaveNode>(&node) || llvm::isa<Constant>(&node) ||
        llvm::isa<Placeholder>(&node)) {
      continue;
    }
    llvm::StringRef layerName = node.getName();
    llvm::StringRef kName = node.getKindName();
    LOG(INFO) << "Verifying layer: " << layerName.data()
              << "\tType: " << kName.data() << "\n";
    auto refOuts = hookAndRun(layerName, bindings, /*isRef*/ true);
    auto checkOuts = hookAndRun(layerName, bindings, /*isRef*/ false);
    if (!checkTensors(refOuts.outputs, checkOuts.outputs)) {
      LOG(ERROR) << "\tResults differ\n";
      LOG(ERROR) << "\tDumping tensors\n";
      dumpTensors(refOuts.outputs, layerName.data(), "ref_output");
      dumpTensors(refOuts.inputs, layerName.data(), "input");
      brokenLayers_.push_back(layerName.str());
      allPassed = false;
    }
    inferBindingsCheck_.clear();
    inferBindingsRef_.clear();
    LOG(INFO) << "DONE Verifying layer: " << layerName.data() << "\n";
  }
  return allPassed;
}
RecursiveLayerComparator::RecursiveLayerComparator(
    Module &mod, const std::string &referenceBackend,
    const std::string &testBackend, float numericCmpThreshold,
    bool dumpTensorsForBadLayer)
    : NetworkComparatorBase(mod, referenceBackend, testBackend,
                            numericCmpThreshold, dumpTensorsForBadLayer) {}

void IntermediateLayerComparator::hookSingleNodeInPlace(
    Node &node, std::list<SaveNode *> &saveNodes,
    std::list<Placeholder *> &hookPlaceholders) {
  std::string layerName = node.getName().str();
  for (unsigned i = 0; i < node.getNumResults(); ++i) {
    std::string saveName = node.getOutputName(i).str();
    saveName += "_" + layerName + "_hook";
    SaveNode *save =
        node.getParent()->createSave(saveName, node.getNthResult(i));
    saveNodes.emplace_back(save);
    hookPlaceholders.emplace_back(save->getPlaceholder());
  }
}
void IntermediateLayerComparator::hookNodesInPlace(
    Function *func, std::list<SaveNode *> &saveNodes,
    std::list<Placeholder *> &hookPlaceholders) {
  DEBUG_GLOW(LOG(INFO) << "Before hooking the function: " << func->dumpDAG());
  for (Node &node : func->getNodes()) {
    if (llvm::isa<SaveNode>(&node) || llvm::isa<Constant>(&node) ||
        llvm::isa<Placeholder>(&node)) {
      continue;
    }
    hookSingleNodeInPlace(node, saveNodes, hookPlaceholders);
  }
  DEBUG_GLOW(LOG(INFO) << "After hooking the function: " << func->dumpDAG());
}

void IntermediateLayerComparator::copyInputBindingsToHookedBindings(
    PlaceholderBindings &hookedBindigs, PlaceholderBindings &inputBindings) {
  // Copy tensors from input bindings to the bindings we use for Inference.
  for (auto &PH : inputBindings.pairs()) {
    auto iPH = hookedBindigs.getPlaceholderByNameSlow(PH.first->getName());
    hookedBindigs.get(iPH)->assign(&PH.second);
  }
}

void IntermediateLayerComparator::getIntermediateResults(
    ExecutionEngine &networkExecEngine, PlaceholderBindings *inputBindings,
    PlaceholderBindings &hookedBindigs) {
  std::list<SaveNode *> saveNodes;
  std::list<Placeholder *> hookPlaceholders;
  Function *func = networkExecEngine.getSingleFunctionFromModule();
  std::string newName = func->getName().str();
  Function *hookedFunction = func->clone(newName + "_hooked");
  // Instrument all the nodes in hookedFunction by inserts Save nodes.
  hookNodesInPlace(hookedFunction, saveNodes, hookPlaceholders);
  hookedBindigs.allocate(networkExecEngine.getModule().getPlaceholders());
  // Copy values from inputBindings to the allocated hookedBindigs.
  copyInputBindingsToHookedBindings(hookedBindigs, *inputBindings);
  networkExecEngine.compile(CompilationMode::Infer);
  networkExecEngine.run(hookedBindigs, hookedFunction->getName());
  DEBUG_GLOW(LOG(INFO) << "Network has " << hookPlaceholders.size()
                       << " hooks inserted and " << hookedBindigs.pairs().size()
                       << " total bindings");
  DEBUG_GLOW(LOG(INFO) << "Network after running and compiling the function: "
                       << hookedFunction->dumpDAG());
  hookedFunction->getParent()->eraseFunction(hookedFunction);
}

void IntermediateLayerComparator::fillSingleLayerInputs(
    const Node &originalNode, Node *singleLayerNode, Module &singleLayerMod,
    std::unordered_map<std::string, Tensor *> &singleLayerInputMap,
    PlaceholderBindings &singleLayerBindings) {
  // Copy the types used in the node to the singleLayerModule.
  for (unsigned idx = 0, e = singleLayerNode->getNumResults(); idx < e; ++idx) {
    singleLayerNode->setType(
        idx, singleLayerMod.uniqueType(*singleLayerNode->getType(idx)));
  }

  for (size_t idx = 0; idx < originalNode.getNumInputs(); idx++) {
    size_t resNo = originalNode.getNthInput(idx).getResNo();
    Node *inputNode = originalNode.getNthInput(idx).getNode();
    std::string hookedPlaceholderName = inputNode->getName().str();
    Node *inputToFeed = nullptr;
    DCHECK(!llvm::isa<SaveNode>(inputNode))
        << "SaveNode as an input was not hooked!";
    if (llvm::isa<Constant>(inputNode)) {
      // Constants live in the module, getting them from the reloaded
      // module. They get deleted after running.
      DEBUG_GLOW(LOG(INFO) << "\t\tInput name: " << hookedPlaceholderName
                           << " NodeType: " << inputNode->getKindName()
                           << "\n");
      Constant *constNode =
          inputModule_->getConstantByName(hookedPlaceholderName);
      DCHECK(constNode) << "Constant not found\n";
      Tensor &payLoad = constNode->getPayloadMutable();
      inputToFeed = singleLayerMod.createConstant(constNode->getName(), payLoad,
                                                  constNode->getLayout());
      singleLayerInputMap[originalNode.getInputName(idx)] = &payLoad;
    } else {
      if (!llvm::isa<Placeholder>(inputNode)) {
        // If the input is placeholder the name stays the
        // same since these don't get hooked.
        std::string outputName = inputNode->getOutputName(resNo).str();
        hookedPlaceholderName =
            outputName + "_" + hookedPlaceholderName + "_hook";
      }
      DEBUG_GLOW(LOG(INFO) << "\t\tInput name: " << hookedPlaceholderName
                           << " NodeType: " << inputNode->getKindName()
                           << "\n");
      Placeholder *PH =
          refHookedBindings_.getPlaceholderByNameSlow(hookedPlaceholderName);
      DCHECK(PH) << "Placeholder not found in the hooked bindings";
      Tensor *payloadTensor = refHookedBindings_.get(PH);
      Placeholder *singleLayerPH = singleLayerMod.createPlaceholder(
          PH->getType(), PH->getName(), PH->isTraining(), PH->getLayout());
      singleLayerBindings.allocate(singleLayerPH);
      singleLayerBindings.get(singleLayerPH)->assign(payloadTensor);
      inputToFeed = singleLayerPH;
      singleLayerInputMap[originalNode.getInputName(idx)] = payloadTensor;
    }
    singleLayerNode->setNthInput(idx, inputToFeed);
  }
}

void IntermediateLayerComparator::runAndGetoutputSingleLayer(
    ExecutionEngine &singleLayerExecEng,
    PlaceholderBindings &singleLayerBindings, Node *singleLayerNode,
    std::unordered_map<std::string, Tensor *> &singleLayerOutputs,
    std::unordered_map<std::string, Tensor *> &refOutputs) {
  std::list<SaveNode *> singleLayerSaveNodes;
  std::list<Placeholder *> singleLayerOutputPHs;
  hookSingleNodeInPlace(*singleLayerNode, singleLayerSaveNodes,
                        singleLayerOutputPHs);
  singleLayerBindings.allocate(singleLayerOutputPHs);
  DEBUG_GLOW(LOG(INFO) << "\t\tSingle layer network"
                       << singleLayerNode->getParent()->dumpDAG());
  singleLayerExecEng.compile(CompilationMode::Infer);
  singleLayerExecEng.run(singleLayerBindings);
  for (Placeholder *PH : singleLayerOutputPHs) {
    singleLayerOutputs[PH->getName().str()] = singleLayerBindings.get(PH);
    Placeholder *refPH =
        refHookedBindings_.getPlaceholderByNameSlow(PH->getName());
    refOutputs[PH->getName().str()] = refHookedBindings_.get(refPH);
  }
}

bool IntermediateLayerComparator::testSingleLayer(const Node *node) {
  bool pass = true;
  std::unordered_map<std::string, Tensor *> singleLayerInputMap;
  ExecutionEngine singleLayerExecEng(EETestNet_.getBackendName());
  PlaceholderBindings singleLayerBindings;
  Module &singleLayerMod = singleLayerExecEng.getModule();
  Function *singleLayerFunc = singleLayerMod.createFunction(node->getName());
  Node *singleLayerNode = node->clone();
  llvm::StringRef layerName = node->getName();
  llvm::StringRef kindName = node->getKindName();
  LOG(INFO) << "Verifying layer: " << layerName.data()
            << "\tType: " << kindName.data() << "\n";
  singleLayerFunc->addNode(singleLayerNode);
  // 1) Dynamically build a net made out of one layer and feed the
  // placeholders in.
  fillSingleLayerInputs(*node, singleLayerNode, singleLayerMod,
                        singleLayerInputMap, singleLayerBindings);
  // 2) Run the network and get outputs.
  std::unordered_map<std::string, Tensor *> singleLayerOutputs;
  std::unordered_map<std::string, Tensor *> refOutputs;
  runAndGetoutputSingleLayer(singleLayerExecEng, singleLayerBindings,
                             singleLayerNode, singleLayerOutputs, refOutputs);
  if (!checkTensors(refOutputs, singleLayerOutputs)) {
    LOG(ERROR) << "\tResults differ\n";
    LOG(ERROR) << "\tDumping tensors\n";
    dumpTensors(refOutputs, layerName.str(), "ref_output");
    dumpTensors(singleLayerInputMap, layerName.str(), "input");
    brokenLayers_.push_back(layerName.str());
    pass = false;
  }
  LOG(INFO) << "DONE Verifying layer: " << layerName.data() << "\n";
  return pass;
}

bool IntermediateLayerComparator::verify(PlaceholderBindings *bindings) {
  bool allPassed = true;
  // Instrument all the layers with Save nodes.
  getIntermediateResults(EERefNet_, bindings, refHookedBindings_);
  getIntermediateResults(EETestNet_, bindings, testHookedBindings_);
  if (refHookedBindings_.compare(&refHookedBindings_, &testHookedBindings_,
                                 numericCmpThreshold_)) {
    LOG(INFO) << "All intermediate results match.";
    return true;
  }
  // Sort the nodes in topological order to test nodes in that order.
  // Tensors flow through the network in topological order, testing the layers
  // in that order will allow us to see how errors propagate.
  Function *func = *inputModule_->getFunctions().begin();
  GraphPostOrderVisitor visitor(*func);
  llvm::ArrayRef<Node *> order = visitor.getPostOrder();
  for (auto const *nodePtr : order) {
    if (llvm::isa<SaveNode>(nodePtr) || llvm::isa<Constant>(nodePtr) ||
        llvm::isa<Placeholder>(nodePtr)) {
      continue;
    }
    allPassed &= testSingleLayer(nodePtr);
  }
  return allPassed;
}

IntermediateLayerComparator::IntermediateLayerComparator(
    Module &mod, const std::string &referenceBackend,
    const std::string &testBackend, float numericCmpThreshold,
    bool dumpTensorsForBadLayer)
    : NetworkComparatorBase(mod, referenceBackend, testBackend,
                            numericCmpThreshold, dumpTensorsForBadLayer) {}
