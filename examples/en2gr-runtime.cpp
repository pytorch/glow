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
#include "../../tools/loader/TextTranslator.h"
#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Graph/Graph.h"
#include "glow/Importer/Caffe2ModelLoader.h"
#include "glow/Partitioner/Partitioner.h"
#include "glow/Runtime/HostManager/HostManager.h"
#include "glow/Runtime/RuntimeTypes.h"
#include "glow/Support/Error.h"
#include "llvm/Support/FileSystem.h"

#include <chrono>
#include <future>

using namespace glow;
using namespace glow::runtime;

namespace {
llvm::cl::OptionCategory category("en2gr-runtime Options");
llvm::cl::opt<unsigned> numDevices("numDevices",
                                   llvm::cl::desc("Number of Devices to use"),
                                   llvm::cl::init(5), llvm::cl::value_desc("N"),
                                   llvm::cl::cat(category));
} // namespace

/// Load the en2gr model.
static void loadEn2grModel(Module *module, PlaceholderBindings &bindings) {
  Function *F = module->createFunction("en2gr");
  llvm::outs() << "Loading en2gr model.\n";
  std::string sentence("I love music .");

  // Encoded input sentence. Note that the batch size is 1 for inference models.
  Tensor encoderInputs(ElemKind::Int64ITy, {maxInputLenOpt, /* batchSize */ 1});
  encodeString(sentence, &encoderInputs);

  // Inputs other than tokenized input. These should all be initialized to zero
  // (which they are by default). Note, the init_net already defines these
  // tensors solely as placeholders (with incorrect shapes/elementtypes/data).
  // Glow uses these tensors in their place.
  Tensor attnWeights(ElemKind::FloatTy, {maxInputLenOpt});
  Tensor prevHyposIndices(ElemKind::Int64ITy, {beamSizeOpt});
  Tensor prevScores(ElemKind::FloatTy, {1});
  Tensor prevToken(ElemKind::Int64ITy, {1});

  constexpr char const *inputNames[5] = {"encoder_inputs", "attn_weights",
                                         "prev_hypos_indices", "prev_scores",
                                         "prev_token"};
  std::vector<TypeRef> inputTensors = {
      &encoderInputs.getType(), &attnWeights.getType(),
      &prevHyposIndices.getType(), &prevScores.getType(), &prevToken.getType()};

  Caffe2ModelLoader loader("en2gr/predict_net.pb", "en2gr/init_net.pb",
                           inputNames, inputTensors, *F);

  // Allocate tensors to back all inputs and outputs.
  bindings.allocate(module->getPlaceholders());

  Placeholder *encoderInputsVar = llvm::cast<Placeholder>(
      EXIT_ON_ERR(loader.getNodeValueByName("encoder_inputs")));

  updateInputPlaceholders(bindings, {encoderInputsVar}, {&encoderInputs});
  return;
}

/// Starts a run of En2gr
void dispatchEn2gr(HostManager *hostManager,
                   std::unique_ptr<ExecutionContext> context,
                   std::atomic<size_t> &returned,
                   std::promise<void> &finished) {
  auto runid = hostManager->runNetwork(
      "en2gr", std::move(context),
      [&finished](RunIdentifierTy, llvm::Error err,
                  std::unique_ptr<ExecutionContext> context) {
        EXIT_ON_ERR(std::move(err));
        auto *bindings = context->getPlaceholderBindings();
        Placeholder *outputTokenBeamList =
            bindings->getPlaceholderByName("save_output_token_beam_list");
        Placeholder *outputScoreBeamList =
            bindings->getPlaceholderByName("save_output_score_beam_list");
        Placeholder *outputPrevIndexBeamList =
            bindings->getPlaceholderByName("save_output_prev_index_beam_list");

        processAndPrintDecodedTranslation(
            bindings->get(outputTokenBeamList),
            bindings->get(outputScoreBeamList),
            bindings->get(outputPrevIndexBeamList));

        finished.set_value();
      });
  (void)runid;
}

/// Run en2gr model on the number CPU Devices provided by the user.
int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(
      argc, argv, "Run en2gr on a fixed number of CPU devices");

  llvm::outs() << "Initializing " << numDevices
               << " CPU Devices on HostManager.\n";

  std::vector<std::unique_ptr<DeviceConfig>> configs;
  for (unsigned int i = 0; i < numDevices; ++i) {
    auto config = llvm::make_unique<DeviceConfig>(BackendKind::CPU);
    configs.push_back(std::move(config));
  }

  std::unique_ptr<HostManager> hostManager =
      llvm::make_unique<HostManager>(std::move(configs));

  // Generate model, create a context, and add to HostManager.
  std::vector<std::unique_ptr<Module>> modules;

  std::unique_ptr<Module> module = llvm::make_unique<Module>();

  std::unique_ptr<ExecutionContext> context =
      llvm::make_unique<ExecutionContext>();

  // Generate the network.
  srcVocab.loadDictionaryFromFile("en2gr/src_dictionary.txt");
  dstVocab.loadDictionaryFromFile("en2gr/dst_dictionary.txt");

  loadEn2grModel(module.get(), *(context->getPlaceholderBindings()));

  llvm::outs() << "Adding to HostManager\n";

  EXIT_ON_ERR(hostManager->addNetwork(std::move(module)));

  std::promise<void> finished;
  std::atomic<size_t> returned{0};

  dispatchEn2gr(hostManager.get(), std::move(context), returned, finished);

  finished.get_future().wait();

  llvm::outs() << "Finished! \n";

  return 0;
}
