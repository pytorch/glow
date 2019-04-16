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

#include "TextTranslator.h"

using namespace glow;

int main(int argc, char **argv) {
  PlaceholderBindings bindings;

  // The loader verifies/initializes command line parameters, and initializes
  // the ExecutionEngine and Function.
  Loader loader(argc, argv);

  // Load the source and dest dictionaries.
  auto modelDir = loader.getModelOptPath();
  srcVocab.loadDictionaryFromFile(modelDir.str() + "/src_dictionary.txt");
  dstVocab.loadDictionaryFromFile(modelDir.str() + "/dst_dictionary.txt");

  // Encoded input sentence. Note that the batch size is 1 for inference models.
  Tensor encoderInputs(ElemKind::Int64ITy, {maxInputLenOpt, /* batchSize */ 1});

  // Inputs other than tokenized input. These should all be initialized to zero
  // (which they are by default). Note, the init_net already defines these
  // tensors solely as placeholders (with incorrect shapes/elementtypes/data).
  // Glow uses these tensors in their place.
  Tensor attnWeights(ElemKind::FloatTy, {maxInputLenOpt});
  Tensor prevHyposIndices(ElemKind::Int64ITy, {beamSizeOpt});
  Tensor prevScores(ElemKind::FloatTy, {1});
  Tensor prevToken(ElemKind::Int64ITy, {1});

  assert(!loader.getCaffe2NetDescFilename().empty() &&
         "Only supporting Caffe2 currently.");

  constexpr char const *inputNames[5] = {"encoder_inputs", "attn_weights",
                                         "prev_hypos_indices", "prev_scores",
                                         "prev_token"};
  std::vector<TypeRef> inputTensors = {
      &encoderInputs.getType(), &attnWeights.getType(),
      &prevHyposIndices.getType(), &prevScores.getType(), &prevToken.getType()};

  Caffe2ModelLoader LD(loader.getCaffe2NetDescFilename(),
                       loader.getCaffe2NetWeightFilename(), inputNames,
                       inputTensors, *loader.getFunction());

  // Allocate tensors to back all inputs and outputs.
  bindings.allocate(loader.getModule()->getPlaceholders());

  Placeholder *encoderInputsVar = llvm::cast<Placeholder>(
      EXIT_ON_ERR(LD.getNodeValueByName("encoder_inputs")));

  // Compile the model, and perform quantization/emit a bundle/dump debug info
  // if requested from command line.
  loader.compile(bindings);

  assert(!emittingBundle() && "Bundle mode has not been tested.");

  Placeholder *outputTokenBeamList =
      EXIT_ON_ERR(LD.getOutputByName("output_token_beam_list"));
  Placeholder *outputScoreBeamList =
      EXIT_ON_ERR(LD.getOutputByName("output_score_beam_list"));
  Placeholder *outputPrevIndexBeamList =
      EXIT_ON_ERR(LD.getOutputByName("output_prev_index_beam_list"));

  while (loadNextInputTranslationText(&encoderInputs)) {
    // Update the inputs.
    updateInputPlaceholders(bindings, {encoderInputsVar}, {&encoderInputs});

    // Run actual translation.
    loader.runInference(bindings);

    // Process the outputs to determine the highest likelihood sentence, and
    // print out the decoded translation using the dest dictionary.
    processAndPrintDecodedTranslation(bindings.get(outputTokenBeamList),
                                      bindings.get(outputScoreBeamList),
                                      bindings.get(outputPrevIndexBeamList));
  }

  // If profiling, generate and serialize the quantization infos now that we
  // have run inference to gather the profile.
  if (profilingGraph()) {
    loader.generateAndSerializeQuantizationInfos(bindings);
  }

  return 0;
}
