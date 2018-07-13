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

#include "Loader.h"

#include "glow/Importer/Caffe2.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

#include <fstream>
#include <iostream>
#include <sstream>

using namespace glow;

namespace {
llvm::cl::OptionCategory textTranslatorCat("Text Translator Options");

llvm::cl::opt<size_t>
    maxInputLenOpt("max_input_len",
                   llvm::cl::desc("Maximum allowed length of the input "
                                  "sentence. Specified by the input model."),
                   llvm::cl::Optional, llvm::cl::init(10),
                   llvm::cl::cat(textTranslatorCat));

llvm::cl::opt<size_t>
    maxOutputLenOpt("max_output_len",
                    llvm::cl::desc("Maximum allowed length of the output "
                                   "sentence. Specified by the input model."),
                    llvm::cl::Optional, llvm::cl::init(14),
                    llvm::cl::cat(textTranslatorCat));

llvm::cl::opt<size_t> beamSizeOpt(
    "beam_size", llvm::cl::desc("Beam size used by the input model."),
    llvm::cl::Optional, llvm::cl::init(6), llvm::cl::cat(textTranslatorCat));

llvm::cl::opt<double>
    lengthPenaltyOpt("length_penalty",
                     llvm::cl::desc("Length penalty to use when determining "
                                    "highest likelihood output sentence."),
                     llvm::cl::Optional, llvm::cl::init(0.0f),
                     llvm::cl::cat(textTranslatorCat));
} // namespace

/// These should be kept in sync with pytorch_translate/vocab_constants.py
constexpr size_t reservedOffset = 100;
constexpr size_t padIdx = 0;
constexpr size_t eosIdx = 2;
constexpr size_t unkIdx = 3;

/// Stores dictionary of a language. Contains mapping from word to index and
/// vice versa.
struct Dictionary {
private:
  std::vector<std::string> index2word_;
  std::unordered_map<std::string, size_t> word2index_;

public:
  /// Add a word from the \p line to the dictionary.
  void addWord(llvm::StringRef line) {
    // Lines generally should be formatted like "the 9876543", where the
    // trailing number is not relevant for inference.
    auto spaceIdx = line.find(" ");
    assert(spaceIdx != llvm::StringRef::npos &&
           "Unexpected format for dict file.");

    auto word = line.take_front(spaceIdx);
    assert(word != "" && "Did not find word correctly.");

    word2index_[word] = index2word_.size();
    index2word_.push_back(word);
  }

  Dictionary() = default;

  /// Load a dictionary from text file \p filename, adding each word from each
  /// line of the file.
  void loadDictionaryFromFile(llvm::StringRef filename) {
    std::ifstream file(filename);
    std::string word;
    while (getline(file, word)) {
      addWord(word);
    }
  }

  /// Get the index for the input \p word from the dictionary.
  size_t getIdxFromWord(llvm::StringRef word) {
    auto iter = word2index_.find(word);
    // If unknown word, return the index for unknown.
    if (iter == word2index_.end()) {
      return unkIdx;
    }
    return iter->second + reservedOffset;
  }

  /// Get the word for the input \p index from the dictionary.
  std::string getWordFromIdx(size_t idx) {
    if (idx < reservedOffset) {
      if (idx == eosIdx) {
        return "<EOS>";
      }
      if (idx == padIdx) {
        return "<PAD>";
      }
      return "<UNK>";
    }

    return index2word_[idx - reservedOffset];
  }
};

Dictionary srcVocab, dstVocab;

/// Break the input \p sentence up by spaces, and then encode the words as
/// indices from the input dictionary, placing them in \p encoderInputs. Note
/// that the model expects sentences to be in reverse order.
static void encodeString(const llvm::StringRef sentence,
                         Tensor *encoderInputs) {
  auto IH = encoderInputs->getHandle<int64_t>();

  std::vector<int64_t> encodedWords;
  encodedWords.reserve(maxInputLenOpt);

  // Get each word from the sentence and encode it.
  std::istringstream iss(sentence);
  std::string word;
  while (iss >> word) {
    auto idx = srcVocab.getIdxFromWord(word);
    encodedWords.push_back(idx);
  }
  encodedWords.push_back(eosIdx);

  GLOW_ASSERT(encodedWords.size() <= maxInputLenOpt &&
              "Sentence length exceeds maxInputLen.");

  // Pad the rest of the input.
  while (encodedWords.size() != maxInputLenOpt) {
    encodedWords.push_back(padIdx);
  }

  // Note: the model expects the input sentence to be in reverse order.
  size_t i = 0;
  for (auto it = encodedWords.rbegin(); it != encodedWords.rend(); it++, i++) {
    // The batch size is 1 for inference models.
    IH.at({i, /* batchSize */ 0}) = *it;
  }
}

/// Load a sentence from std::cin for processing, placing the encoded inputs in
/// \p encoderInputs.
static void loadNextInputTranslationText(Tensor *encoderInputs) {
  llvm::outs() << "Enter a sentence in English to translate to German: ";
  std::string sentence;
  getline(std::cin, sentence);
  encodeString(sentence, encoderInputs);
}

/// Find and return a vector of the best translation given the outputs from the
/// model \p outputTokenBeamList, \p outputScoreBeamList, and \p
/// outputPrevIndexBeamList. A translation is made up of a vector of tokens
/// which must be converted back to words from via the destination dictionary.
static std::vector<size_t>
getBestTranslation(Variable *outputTokenBeamList, Variable *outputScoreBeamList,
                   Variable *outputPrevIndexBeamList) {
  // Get handles to all the outputs from the model run.
  auto tokenBeamListH = outputTokenBeamList->getPayload().getHandle<int64_t>();
  auto scoreBeamListH = outputScoreBeamList->getPayload().getHandle<float>();
  auto prevIndexBeamListH =
      outputPrevIndexBeamList->getPayload().getHandle<int64_t>();

  // This pair represents the ending position of a translation in the beam
  // search grid. The first index corresponds to the length (column index), the
  // second index corresponds to the position in the beam (row index).
  std::pair<size_t, size_t> bestPosition = std::make_pair(0, 0);
  float bestScore = std::numeric_limits<float>::max();

  // Keep track of whether the current hypothesis of best translation has
  // already ended.
  std::vector<bool> prevHypoIsFinished(beamSizeOpt, false);
  std::vector<bool> currentHypoIsFinished(beamSizeOpt, false);
  for (size_t lengthIndex = 0; lengthIndex < maxOutputLenOpt; ++lengthIndex) {
    for (size_t hypoIndex = 0; hypoIndex < beamSizeOpt; ++hypoIndex) {
      // If the current hypothesis was already scored and compared to the best,
      // we can skip it and move onto the next one.
      size_t prevIndex = prevIndexBeamListH.at({lengthIndex, hypoIndex});
      currentHypoIsFinished[hypoIndex] = prevHypoIsFinished[prevIndex];
      if (currentHypoIsFinished[hypoIndex]) {
        continue;
      }

      // If the current token is not the end of sentence, and we haven't reached
      // the max output length, then we cannot yet score/compare it, so keep
      // going until we reach the end.
      if (tokenBeamListH.at({lengthIndex, hypoIndex}) != eosIdx &&
          lengthIndex + 1 != maxOutputLenOpt) {
        continue;
      }

      // At this point we must have reached the end of a hypothesis sentence
      // which has not yet been scored and compared. Set this as finished as we
      // will now score and compare it against the current best score.
      currentHypoIsFinished[hypoIndex] = true;

      // Calculate the current score with length penalty.
      float currScore = scoreBeamListH.at({lengthIndex, hypoIndex}) /
                        pow(lengthIndex + 1, lengthPenaltyOpt);

      // If this translation has a better score, replace the current one.
      if (currScore > -bestScore) {
        bestPosition = std::make_pair(lengthIndex, hypoIndex);
        bestScore = -currScore;
      }
    }

    // Moving onto the next hypothesis, so swap current finished bools into
    // previous, and reset current to all false.
    prevHypoIsFinished.swap(currentHypoIsFinished);
    currentHypoIsFinished.assign(beamSizeOpt, false);
  }

  // Generate the best translation given the end state. Use the previous index
  // beam list to find the next word to add to the translation.
  std::vector<size_t> output;
  size_t lengthIndex = bestPosition.first;
  size_t hypoIndex = bestPosition.second;
  while (lengthIndex > 0) {
    output.emplace_back(tokenBeamListH.at({lengthIndex, hypoIndex}));
    hypoIndex = prevIndexBeamListH.at({lengthIndex, hypoIndex});
    lengthIndex--;
  }

  // Reverse the output order of the translated sentence.
  std::reverse(output.begin(), output.end());

  // Find the EOS token and cut off the rest of the output.
  auto findEos = std::find(output.begin(), output.end(), eosIdx);
  auto findEosIndex = findEos - output.begin();
  output.resize(findEosIndex);

  return output;
}

/// Queries getBestTranslation() for the best translation via the outputs from
/// the model, \p outputTokenBeamList, \p outputScoreBeamList, and \p
/// outputPrevIndexBeamList. Then converts each of the tokens from the returned
/// best translation into words from the dest dictionary, and prints it.
static void
processAndPrintDecodedTranslation(Variable *outputTokenBeamList,
                                  Variable *outputScoreBeamList,
                                  Variable *outputPrevIndexBeamList) {
  std::vector<size_t> translationTokens = getBestTranslation(
      outputTokenBeamList, outputScoreBeamList, outputPrevIndexBeamList);

  // Use the dest dictionary to convert tokens to words, and print it.
  for (size_t i = 0; i < translationTokens.size(); i++) {
    auto wordIdx = translationTokens[i];
    auto word = dstVocab.getWordFromIdx(wordIdx);

    // Check if the word has suffix "@@". This means the current word should be
    // appended to the next word, so remove the "@@" and do not output a space.
    auto wordLength = word.length();
    if (wordLength > 1 && word.substr(wordLength - 2) == "@@") {
      word = word.substr(0, wordLength - 2);
    } else if (i != translationTokens.size() - 1) {
      word = word + " ";
    }
    llvm::outs() << word;
  }
  llvm::outs() << "\n\n";
}

int main(int argc, char **argv) {
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
  std::vector<Tensor *> inputTensors = {
      &encoderInputs, &attnWeights, &prevHyposIndices, &prevScores, &prevToken};

  auto LD = caffe2ModelLoader(loader.getCaffe2NetDescFilename(),
                              loader.getCaffe2NetWeightFilename(), inputNames,
                              inputTensors, *loader.getFunction());

  Variable *encoderInputsVar = LD.getVariableByName("encoder_inputs");

  // Compile the model, and perform quantization/emit a bundle/dump debug info
  // if requested from command line.
  loader.compile();

  assert(!emittingBundle() && "Bundle mode has not been tested.");

  Variable *outputTokenBeamList = LD.getOutputByName("output_token_beam_list");
  Variable *outputScoreBeamList = LD.getOutputByName("output_score_beam_list");
  Variable *outputPrevIndexBeamList =
      LD.getOutputByName("output_prev_index_beam_list");

  while (true) {
    // Load the next string into encoderInputs.
    loadNextInputTranslationText(&encoderInputs);

    // Run actual translation.
    loader.runInference({encoderInputsVar}, {&encoderInputs});

    // Process the outputs to determine the highest likelihood sentence, and
    // print out the decoded translation using the dest dictionary.
    processAndPrintDecodedTranslation(outputTokenBeamList, outputScoreBeamList,
                                      outputPrevIndexBeamList);
  }

  return 0;
}
