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
#ifndef GLOW_GRAPH_NETWORKCOMPARATOR_H
#define GLOW_GRAPH_NETWORKCOMPARATOR_H

#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Graph/Graph.h"
#include "glow/Graph/PlaceholderBindings.h"
#include "llvm/ADT/StringRef.h"
#include <list>
#include <string>
#include <unordered_map>
#include <vector>

namespace glow {

class Function;
class Node;
class Placeholder;
class SaveNode;
class Tensor;
class Module;

/// Base class for building a comparator that takes an input network \p
/// inputModule_ and run it on two backends; a test backend and a reference
/// backend and saves the list of layers that generate wrong results on the test
/// net in \p brokenLayers_ . This class has a pure virtual function (verify)
/// that must be implemented by a subclass.
class NetworkComparatorBase {
public:
  /// Struct to save the list of input and output tensors of a node\layer.
  struct InOutTensors {
    std::unordered_map<std::string, Tensor *> inputs;
    std::unordered_map<std::string, Tensor *> outputs;
  };

protected:
  /// Execution Engine to run the Network tested on the reference backend.
  ExecutionEngine EERefNet_;
  /// Execution Engine to run the Network tested on the test backend.
  ExecutionEngine EETestNet_;
  /// Stores layer names found to be broken on the test backend.
  std::vector<std::string> brokenLayers_;
  /// Threshold of numerical comparison for tensors.
  float numericCmpThreshold_;
  /// Pointer to the module being tested.
  Module *inputModule_;

  /// Dump input\output tensors of a bad layer.
  bool dumpTensorsForBadLayer_;
  /// Prints out the input or output \p tensors associated with a \p layerName
  /// the output file is a concatenation of the \p prefix and the \p layerName.
  virtual void dumpTensors(std::unordered_map<std::string, Tensor *> tensors,
                           const std::string &layerName,
                           const std::string &prefix);
  /// Compares tensors in \p refOuts with tensors in \p checkOuts using the
  /// isEqual Tensor method.
  bool checkTensors(std::unordered_map<std::string, Tensor *> &refOuts,
                    std::unordered_map<std::string, Tensor *> &checkOuts);

public:
  /// Constructor for the base class tester that tests the network passed in
  /// \p mod on the \p testBackend using the \p referenceBackend as a reference
  /// backend. \p numericCmpThreshold is accepted error threshold.
  /// Inputs\outputs of a detected bad layer are dumped if \p
  /// dumpTensorsForBadLayer is set.
  NetworkComparatorBase(Module &mod, const std::string &referenceBackend,
                        const std::string &testBackend,
                        float numericCmpThreshold, bool dumpTensorsForBadLayer);
  virtual ~NetworkComparatorBase() {}
  /// Test the network with the inputs passed in \p bindings. The function
  /// is pure virtual to be implemented by the underlying strategy in sub
  /// classes. \returns True if all checks passed.
  virtual bool verify(PlaceholderBindings *bindings) = 0;
};

/// A comparator class that tests layers by creating a subnet of the original
/// graph for every layer. The subnets are created by recursively visiting the
/// inputs of the layer. The results or running the subnet are compared between
/// the test and reference backend. The subnet might have more than one broken
/// layer.
class RecursiveLayerComparator : public NetworkComparatorBase {
private:
  PlaceholderBindings inferBindingsRef_;
  PlaceholderBindings inferBindingsCheck_;
  /// Takes a \p layerName, creates a subnet comprised of all the predecessor
  /// nodes till the inputs. The outputs of this layer are saved in the
  /// inferenceBindingsRef if \p isRef is true and in inferenceBindingsCheck if
  /// false. The inputs are passed in \p bindings. The run saves the inputs to
  /// the layer \p layerName.
  NetworkComparatorBase::InOutTensors hookAndRun(llvm::StringRef layerName,
                                                 PlaceholderBindings *bindings,
                                                 bool isRef);

public:
  /// Constructor for the RecursiveLayerComparator tester that tests the network
  /// passed in \p mod on the \p testBackend using the \p referenceBackend as a
  /// reference backend. \p numericCmpThreshold_. Inputs\outputs of
  /// a detected bad layer are dumped if \p dumpTensorsForBadLayer is set.
  RecursiveLayerComparator(Module &mod, const std::string &referenceBackend,
                           const std::string &testBackend,
                           float numericCmpThreshold_,
                           bool dumpTensorsForBadLayer);
  /// Takes the \p bindings as an input. For every layer in the Network creates
  /// a subnet comprised of all the predecessor nodes till the placeholders.
  /// This subnet is compiled and run on the reference and on the test backend
  /// and results are compared.
  bool verify(PlaceholderBindings *bindings);
};

/// A comparator class that first tests layer all at once to find suspicious
/// layers that are different between the reference and test backends runs. Then
/// all the layers are tested one layer at a time to find out which
/// of them are actually producing wrong results on their own and not as a
/// result of errors in any previous layers.
class IntermediateLayerComparator : public NetworkComparatorBase {
private:
  /// Save results (including intermediates) for running the network on the
  /// reference backend.
  PlaceholderBindings refHookedBindings_;
  /// Save results (including intermediates) for running the network on the test
  /// backend.
  PlaceholderBindings testHookedBindings_;
  /// Inserts Save Nodes for the outputs of \p node and saves them in \p
  /// saveNodes and the placeholders in \p hookPlaceholders.
  void hookSingleNodeInPlace(Node &node, std::list<SaveNode *> &saveNodes,
                             std::list<Placeholder *> &hookPlaceholders);
  /// Hook outputs of all the layers in \p func and saves the pointers to
  /// the inserted Save nodes and corresponding placeholders in \p saveNodes
  /// and \p hookPlaceholders.
  void hookNodesInPlace(Function *func, std::list<SaveNode *> &saveNodes,
                        std::list<Placeholder *> &hookPlaceholders);
  /// Populate tensors in \p  hookedBindigs with values from inputBindings.
  void copyInputBindingsToHookedBindings(PlaceholderBindings &hookedBindigs,
                                         PlaceholderBindings &inputBindings);
  /// Uses the \p networkExecEngine to get all the intermediate results of
  /// running the network and save the results in \p hookedBindigs. Uses \p
  /// inputBindings for inputs for the run.
  void getIntermediateResults(ExecutionEngine &networkExecEngine,
                              PlaceholderBindings *inputBindings,
                              PlaceholderBindings &hookedBindigs);
  /// Takes \p originalNode , that represents the layer being tested,
  /// and fills in the inputs of \p singleLayerNode with placeholders from
  /// running the hooked network on a ref backend or constants
  /// from the original module. Saves the inputs in \p singleLayerInputMap and
  /// the placeholders in \p singleLayerBindings.
  void fillSingleLayerInputs(
      const Node &originalNode, Node *singleLayerNode, Module &singleLayerMod,
      std::unordered_map<std::string, Tensor *> &singleLayerInputMap,
      PlaceholderBindings &singleLayerBindings);
  /// Runs the network made of a single layer \p singleLayerNode that is getting
  /// tested. The outputs are saved in \p singleLayerOutputs and bindings in \p
  /// singleLayerBindings \p refOutputs are populated fron the reference
  /// bindings run.
  void runAndGetoutputSingleLayer(
      ExecutionEngine &singleLayerExecEng,
      PlaceholderBindings &singleLayerBindings, Node *singleLayerNode,
      std::unordered_map<std::string, Tensor *> &singleLayerOutputs,
      std::unordered_map<std::string, Tensor *> &refOutputs);
  /// Tests a single layer node passed in \p node.
  bool testSingleLayer(const Node *node);

public:
  /// Constructor for the IntermediateLayerComparator tester that tests the
  /// network passed in \p mod on the \p testBackend using the \p
  /// referenceBackend as a reference backend. \p numericCmpThreshold_.
  /// Inputs\outputs of a detected bad layer are dumped if \p
  /// dumpTensorsForBadLayer is set.
  IntermediateLayerComparator(Module &mod, const std::string &referenceBackend,
                              const std::string &testBackend,
                              float numericCmpThreshold,
                              bool IntermediateLayerComparator);
  /// Takes the \p bindings as an input. For every layer in the Network creates
  /// a subnet comprised of all the predecessor nodes till the placeholders.
  /// This subnet is compiled and run on the reference and on the test backend
  /// and results are compared.
  virtual bool verify(PlaceholderBindings *bindings) override;
};

} // namespace glow

#endif // GLOW_GRAPH_NETWORKCOMPARATOR_H
