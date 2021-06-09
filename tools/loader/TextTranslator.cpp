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

#include "Loader.h"

#include "glow/Base/TensorSerialization.h"
#include "glow/Importer/Caffe2ModelLoader.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <fstream>
#include <iostream>
#include <sstream>

using namespace glow;

namespace {
llvm::cl::OptionCategory textTranslatorCat("Text Translator Options");

llvm::cl::opt<unsigned>
    maxInputLenOpt("max-input-len",
                   llvm::cl::desc("Maximum allowed length of the input "
                                  "sentence. Specified by the input model."),
                   llvm::cl::Optional, llvm::cl::init(10),
                   llvm::cl::cat(textTranslatorCat));

llvm::cl::opt<unsigned>
    maxOutputLenOpt("max-output-len",
                    llvm::cl::desc("Maximum allowed length of the output "
                                   "sentence. Specified by the input model."),
                    llvm::cl::Optional, llvm::cl::init(14),
                    llvm::cl::cat(textTranslatorCat));

llvm::cl::opt<unsigned> beamSizeOpt(
    "beam-size", llvm::cl::desc("Beam size used by the input model."),
    llvm::cl::Optional, llvm::cl::init(6), llvm::cl::cat(textTranslatorCat));

llvm::cl::opt<double>
    lengthPenaltyOpt("length-penalty",
                     llvm::cl::desc("Length penalty to use when determining "
                                    "highest likelihood output sentence."),
                     llvm::cl::Optional, llvm::cl::init(0.0f),
                     llvm::cl::cat(textTranslatorCat));

llvm::cl::opt<std::string> inputSentencesFile(
    "input-text-file",
    llvm::cl::desc(
        "Name of the file containing list of sentences (one per line)"),
    llvm::cl::value_desc("string_name"), llvm::cl::Optional,
    llvm::cl::cat(textTranslatorCat));

llvm::cl::opt<std::string> expectedResultSentencesFile(
    "expected-output-text-file",
    llvm::cl::desc("Name of the file containing list of sentences (one per "
                   "line) corresponding to expected translations provided via "
                   "-input-text-file."),
    llvm::cl::value_desc("string_name"), llvm::cl::Optional,
    llvm::cl::cat(textTranslatorCat));

llvm::cl::opt<bool> dumpBinaryResults(
    "dump-binary-results",
    llvm::cl::desc("Dump raw binary Tensor results after execution."),
    llvm::cl::init(false), llvm::cl::cat(textTranslatorCat));
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
    DCHECK(spaceIdx != llvm::StringRef::npos)
        << "Unexpected format for dict file.";

    auto word = std::string(line.take_front(spaceIdx));
    DCHECK_GT(word.size(), 0) << "Did not find word correctly.";

    word2index_[word] = index2word_.size();
    index2word_.push_back(word);
  }

  Dictionary() = default;

  /// Load a dictionary from text file \p filename, adding each word from each
  /// line of the file.
  void loadDictionaryFromFile(llvm::StringRef filename) {
    std::ifstream file(filename.str());
    std::string word;
    while (getline(file, word)) {
      addWord(word);
    }
  }

  /// Get the index for the input \p word from the dictionary.
  size_t getIdxFromWord(llvm::StringRef word) {
    auto iter = word2index_.find(word.str());
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
  auto IH = encoderInputs->getHandle<int32_t>();

  std::vector<int32_t> encodedWords;
  encodedWords.reserve(maxInputLenOpt);

  // Get each word from the sentence and encode it.
  std::istringstream iss(sentence.str());
  std::string word;
  while (iss >> word) {
    auto idx = srcVocab.getIdxFromWord(word);
    encodedWords.push_back(idx);
  }
  encodedWords.push_back(eosIdx);

  CHECK_LE(encodedWords.size(), maxInputLenOpt)
      << "Sentence length exceeds maxInputLen.";

  // Pad the rest of the input.
  while (encodedWords.size() != maxInputLenOpt) {
    encodedWords.push_back(padIdx);
  }

  // Note: the model expects the input sentence to be in reverse order.
  dim_t i = 0;
  for (auto it = encodedWords.rbegin(); it != encodedWords.rend(); it++, i++) {
    // The batch size is 1 for inference models.
    IH.at({i, /* batchSize */ 0}) = *it;
  }
}

/// Load a sentence from std::cin for processing, or from \p file if it is open,
/// and place the encoded inputs in \p encoderInputs. \returns false if the
/// passed in line was empty.
static bool loadNextInputTranslationText(Tensor *encoderInputs,
                                         std::ifstream &file) {
  std::string sentence;
  if (file.is_open()) {
    getline(file, sentence);
  } else {
    llvm::outs() << "Enter a sentence in English to translate to German: ";
    getline(std::cin, sentence);
  }

  if (sentence.empty()) {
    return false;
  }

  encodeString(sentence, encoderInputs);

  return true;
}

/// Find and return a vector of the best translation given the outputs from the
/// model \p outputTokenBeamList, \p outputScoreBeamList, and \p
/// outputPrevIndexBeamList. A translation is made up of a vector of tokens
/// which must be converted back to words from via the destination dictionary.
static std::vector<dim_t> getBestTranslation(Tensor *outputTokenBeamList,
                                             Tensor *outputScoreBeamList,
                                             Tensor *outputPrevIndexBeamList) {
  // Get handles to all the outputs from the model run.
  auto tokenBeamListH = outputTokenBeamList->getHandle<int32_t>();
  auto scoreBeamListH = outputScoreBeamList->getHandle<float>();
  auto prevIndexBeamListH = outputPrevIndexBeamList->getHandle<int32_t>();

  // This pair represents the ending position of a translation in the beam
  // search grid. The first index corresponds to the length (column index), the
  // second index corresponds to the position in the beam (row index).
  std::pair<dim_t, dim_t> bestPosition = std::make_pair(0, 0);
  float bestScore = std::numeric_limits<float>::max();

  // Keep track of whether the current hypothesis of best translation has
  // already ended.
  std::vector<bool> prevHypoIsFinished(beamSizeOpt, false);
  std::vector<bool> currentHypoIsFinished(beamSizeOpt, false);
  for (dim_t lengthIndex = 0; lengthIndex < maxOutputLenOpt; ++lengthIndex) {
    for (dim_t hypoIndex = 0; hypoIndex < beamSizeOpt; ++hypoIndex) {
      // If the current hypothesis was already scored and compared to the best,
      // we can skip it and move onto the next one.
      dim_t prevIndex = prevIndexBeamListH.at({lengthIndex, hypoIndex});
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
  std::vector<dim_t> output;
  dim_t lengthIndex = bestPosition.first;
  dim_t hypoIndex = bestPosition.second;
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
/// the model, \p outputTokenBeamList, \p outputScoreBeamList, and
/// \p outputPrevIndexBeamList. Then converts each of the tokens from the
/// returned best translation into words from the dest dictionary, and prints
/// it. \p expectedResultFile is a stream of expected results; if open the
/// translation is checked against the expected result, and will \returns
/// whether it does. Otherwise returns true.
static bool processAndPrintDecodedTranslation(
    Tensor *outputTokenBeamList, Tensor *outputScoreBeamList,
    Tensor *outputPrevIndexBeamList, std::ifstream &expectedResultFile) {
  std::vector<dim_t> translationTokens = getBestTranslation(
      outputTokenBeamList, outputScoreBeamList, outputPrevIndexBeamList);

  // Use the dest dictionary to convert tokens to words, and print it.
  std::string result;
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
    result += word;
  }
  llvm::outs() << result;
  bool correctTranslation = true;
  if (expectedResultFile.is_open()) {
    std::string expectedResult;
    CHECK(getline(expectedResultFile, expectedResult))
        << "Did not find expected translation.";
    correctTranslation = expectedResult == result;
    if (!correctTranslation) {
      llvm::outs() << "\n   PREVIOUS TRANSLATION INCORRECT; EXPECTED: "
                   << expectedResult;
    }
  }
  llvm::outs() << "\n\n";
  return correctTranslation;
}

int main(int argc, char **argv) {
  PlaceholderBindings bindings;

  // Verify/initialize command line parameters, and then loader initializes
  // the ExecutionEngine and Function.
  parseCommandLine(argc, argv);
  Loader loader;

  // Load the source and dest dictionaries.
  auto modelDir = loader.getModelOptDir();
  srcVocab.loadDictionaryFromFile(modelDir.str() + "/src_dictionary.txt");
  dstVocab.loadDictionaryFromFile(modelDir.str() + "/dst_dictionary.txt");

  // Encoded input sentence. Note that the batch size is 1 for inference models.
  Tensor encoderInputs(ElemKind::Int32ITy, {maxInputLenOpt, /* batchSize */ 1});

  // Inputs other than tokenized input. These should all be initialized to zero.
  // Note, the init_net already defines these tensors solely as placeholders
  // (with incorrect shapes/elementtypes/data). Glow uses these tensors in their
  // place.
  Tensor attnWeights(ElemKind::FloatTy, {maxInputLenOpt});
  Tensor prevHyposIndices(ElemKind::Int32ITy, {beamSizeOpt});
  Tensor prevScores(ElemKind::FloatTy, {1});
  Tensor prevToken(ElemKind::Int32ITy, {1});

  DCHECK(!loader.getCaffe2NetDescFilename().empty())
      << "Only supporting Caffe2 currently.";

  constexpr char const *inputNames[5] = {"encoder_inputs", "attn_weights",
                                         "prev_hypos_indices", "prev_scores",
                                         "prev_token"};
  std::vector<TypeRef> inputTensors = {
      &encoderInputs.getType(), &attnWeights.getType(),
      &prevHyposIndices.getType(), &prevScores.getType(), &prevToken.getType()};

  Caffe2ModelLoader LD(loader.getCaffe2NetDescFilename().str(),
                       loader.getCaffe2NetWeightFilename().str(), inputNames,
                       inputTensors, *loader.getFunction());

  // Allocate tensors to back all inputs and outputs.
  bindings.allocate(loader.getModule()->getPlaceholders());

  // Get all bound tensors and zero them.
  for (auto &pair : bindings.pairs()) {
    pair.second.zero();
  }

  Placeholder *encoderInputsVar = llvm::cast<Placeholder>(
      EXIT_ON_ERR(LD.getNodeValueByName("encoder_inputs")));

  // Compile the model, and perform quantization/emit a bundle/dump debug info
  // if requested from command line.
  loader.compile(bindings);

  DCHECK(!emittingBundle()) << "Bundle mode has not been tested.";

  Placeholder *outputTokenBeamList =
      EXIT_ON_ERR(LD.getOutputByName("output_token_beam_list"));
  Placeholder *outputScoreBeamList =
      EXIT_ON_ERR(LD.getOutputByName("output_score_beam_list"));
  Placeholder *outputPrevIndexBeamList =
      EXIT_ON_ERR(LD.getOutputByName("output_prev_index_beam_list"));

  std::ifstream inFile, expectedResultFile;
  if (!inputSentencesFile.empty()) {
    inFile.open(inputSentencesFile, std::ifstream::in);
  }
  if (!expectedResultSentencesFile.empty()) {
    expectedResultFile.open(expectedResultSentencesFile, std::ifstream::in);
  }

  // We reuse the same output Tensors for every inference, so it's safe to get
  // them ahead of the inference loop.
  Tensor *outputTokenBeamListT = bindings.get(outputTokenBeamList);
  Tensor *outputScoreBeamListT = bindings.get(outputScoreBeamList);
  Tensor *outputPrevIndexBeamListT = bindings.get(outputPrevIndexBeamList);

  // Store copies of results of each inference for dumping after the inference
  // loop if requested.
  std::vector<Tensor> outputTokenBeamListResults;
  std::vector<Tensor> outputScoreBeamListResults;
  std::vector<Tensor> outputPrevIndexBeamListResults;

  int incorrectTranslationCount = 0;
  while (loadNextInputTranslationText(&encoderInputs, inFile)) {
    // Update the inputs.
    updateInputPlaceholders(bindings, {encoderInputsVar}, {&encoderInputs});

    // Run actual translation.
    loader.runInference(bindings);

    // Keep around copies of all results to dump after inference loop.
    if (dumpBinaryResults) {
      outputTokenBeamListResults.emplace_back(outputTokenBeamListT->clone());
      outputScoreBeamListResults.emplace_back(outputScoreBeamListT->clone());
      outputPrevIndexBeamListResults.emplace_back(
          outputPrevIndexBeamListT->clone());
    }

    // Process the outputs to determine the highest likelihood sentence, and
    // print out the decoded translation using the dest dictionary.
    if (!processAndPrintDecodedTranslation(
            outputTokenBeamListT, outputScoreBeamListT,
            outputPrevIndexBeamListT, expectedResultFile)) {
      incorrectTranslationCount += 1;
    }
  }

  if (dumpBinaryResults) {
    TensorSerializationOptions opts;
    opts.withType = false;
    auto dumpRes = [&](const std::string &name,
                       const std::vector<Tensor> &vecT) {
      std::ofstream fs;
      llvm::SmallString<64> path;
      auto tempFileRes = llvm::sys::fs::createTemporaryFile(name, "bin", path);
      CHECK_EQ(tempFileRes.value(), 0)
          << "Failed to create temp file to write into.";
      fs.open(path.data(), std::ios::binary);
      CHECK(fs.is_open()) << "Error opening file '" << path.data() << "'!";
      std::cout << "Dumping binary results of " << name << " to " << path.data()
                << std::endl;
      for (const Tensor &T : vecT) {
        dumpTensorToBinaryFile(T, fs, opts);
      }
      fs.close();
    };
    dumpRes("outputTokenBeamList", outputTokenBeamListResults);
    dumpRes("outputScoreBeamList", outputScoreBeamListResults);
    dumpRes("outputPrevIndexBeamList", outputPrevIndexBeamListResults);
  }

  // If profiling, generate and serialize the profiling infos now that we
  // have run inference to gather the profile.
  if (profilingGraph()) {
    loader.generateAndSerializeProfilingInfos(bindings);
  }

  return incorrectTranslationCount;
}
