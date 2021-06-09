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
#include "glow/Graph/Graph.h"
#include "glow/Optimizer/GraphOptimizer/GraphOptimizer.h"
#include "glow/Quantization/Quantization.h"
#include "glow/Quantization/Serialization.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Timer.h"

#include <glog/logging.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

using namespace glow;

namespace {
/// Debugging options.
llvm::cl::OptionCategory debugCat("Glow Debugging Options");

llvm::cl::opt<std::string> dumpGraphDAGFileOpt(
    "dump-graph-DAG",
    llvm::cl::desc("Dump the graph to the given file in DOT format."),
    llvm::cl::value_desc("file.dot"), llvm::cl::cat(debugCat));

/// Translator options.
llvm::cl::OptionCategory fr2enCat("French-to-English Translator Options");

llvm::cl::opt<unsigned> batchSizeOpt(
    "batchsize", llvm::cl::desc("Process batches of N sentences at a time."),
    llvm::cl::init(1), llvm::cl::value_desc("N"), llvm::cl::cat(fr2enCat));
llvm::cl::alias batchSizeA("b", llvm::cl::desc("Alias for -batchsize"),
                           llvm::cl::aliasopt(batchSizeOpt),
                           llvm::cl::cat(fr2enCat));

llvm::cl::opt<bool>
    timeOpt("time",
            llvm::cl::desc("Print timer data detailing how long it "
                           "takes for the program to execute translate phase. "
                           "This option will be useful if input is read from "
                           "the file directly."),
            llvm::cl::Optional, llvm::cl::cat(fr2enCat));

llvm::cl::opt<std::string> ExecutionBackend(
    "backend",
    llvm::cl::desc("Backend to use, e.g., Interpreter, CPU, OpenCL:"),
    llvm::cl::Optional, llvm::cl::init("Interpreter"), llvm::cl::cat(fr2enCat));

/// Quantization options.
llvm::cl::OptionCategory quantizationCat("Quantization Options");

llvm::cl::opt<std::string> dumpProfileFileOpt(
    "dump-profile",
    llvm::cl::desc("Perform quantization profiling for a given graph "
                   "and dump result to the file."),
    llvm::cl::value_desc("profile.yaml"), llvm::cl::Optional,
    llvm::cl::cat(quantizationCat));

llvm::cl::opt<std::string> loadProfileFileOpt(
    "load-profile",
    llvm::cl::desc("Load quantization profile file and quantize the graph"),
    llvm::cl::value_desc("profile.yaml"), llvm::cl::Optional,
    llvm::cl::cat(quantizationCat));

} // namespace

const unsigned MAX_LENGTH = 10;
const unsigned EMBEDDING_SIZE = 256;
const unsigned HIDDEN_SIZE = EMBEDDING_SIZE * 3;

/// Stores vocabulary of a language. Contains mapping from word to index and
/// vice versa.
struct Vocabulary {
  std::vector<std::string> index2word_;
  std::unordered_map<std::string, int64_t> word2index_;

  void addWord(llvm::StringRef word) {
    word2index_[word.str()] = index2word_.size();
    index2word_.push_back(word.str());
  }

  Vocabulary() = default;

  void loadVocabularyFromFile(llvm::StringRef filename) {
    std::ifstream file(filename.str());
    std::string word;
    while (getline(file, word))
      addWord(word);
  }
};

/// Loads tensor of floats from binary file.
void loadMatrixFromFile(llvm::StringRef filename, Tensor &result) {
  std::ifstream file(filename.str(), std::ios::binary);
  if (!file.read(result.getUnsafePtr(), result.size() * sizeof(float))) {
    std::cout
        << "Error reading file: " << filename.str() << '\n'
        << "Need to be downloaded by calling:\n"
        << "python ../glow/utils/download_datasets_and_models.py -d fr2en\n";
    exit(1);
  }
}

/// Represents a single RNN model: encoder combined with decoder.
/// Stores vocabulary, compiled Graph (ready to be executed), and
/// few references to input/output Variables.
struct Model {
  unsigned batchSize_;
  ExecutionEngine EE_{ExecutionBackend};
  Function *F_;
  Vocabulary en_, fr_;
  Placeholder *input_;
  Placeholder *seqLength_;
  Placeholder *output_;
  PlaceholderBindings bindings;
  LoweredInfoMap loweredMap_;

  void loadLanguages();
  void loadEncoder();
  void loadDecoder();
  void translate(const std::vector<std::string> &batch);

  Model(unsigned batchSize) : batchSize_(batchSize) {
    F_ = EE_.getModule().createFunction("main");
  }

  void dumpGraphDAG(const char *filename) { F_->dumpDAG(filename); }

  void compile() {
    CompilationContext cctx{&bindings, &loweredMap_};
    PrecisionConfiguration &precConfig = cctx.precisionConfig;

    ::glow::convertPlaceholdersToConstants(F_, bindings,
                                           {input_, seqLength_, output_});

    if (!dumpProfileFileOpt.empty()) {
      precConfig.quantMode = QuantizationMode::Profile;
    }

    // Load the quantization profile and transform the graph.
    if (!loadProfileFileOpt.empty()) {
      precConfig.quantMode = QuantizationMode::Quantize;
      deserializeProfilingInfosFromYaml(
          loadProfileFileOpt, precConfig.quantConfig.graphPreLowerHash,
          precConfig.quantConfig.infos);
      precConfig.quantConfig.assertAllNodesQuantized = true;
    }

    EE_.compile(cctx);

    // After compilation, the original function may be removed/replaced. Need to
    // update F_.
    F_ = EE_.getModule().getFunctions().front();
  }

private:
  Placeholder *embedding_fr_, *embedding_en_;
  Node *encoderHiddenOutput_;

  Placeholder *loadEmbedding(llvm::StringRef langPrefix, dim_t langSize) {
    auto &mod = EE_.getModule();
    auto *result =
        mod.createPlaceholder(ElemKind::FloatTy, {langSize, EMBEDDING_SIZE},
                              "embedding." + langPrefix.str(), false);
    loadMatrixFromFile("fr2en/" + langPrefix.str() + "_embedding.bin",
                       *bindings.allocate(result));

    return result;
  }

  Node *createPyTorchGRUCell(Function *G, Node *input, Node *hidden,
                             Placeholder *wIh, Placeholder *bIh,
                             Placeholder *wHh, Placeholder *bHh) {
    // reference implementation:
    // https://github.com/pytorch/pytorch/blob/dd5c195646b941d3e20a72847ac48c41e272b8b2/torch/nn/_functions/rnn.py#L46
    Node *gi = G->createFullyConnected("pytorch.GRU.gi", input, wIh, bIh);
    Node *gh = G->createFullyConnected("pytorch.GRU.gh", hidden, wHh, bHh);

    Node *i_r = G->createSlice("pytorch.GRU.i_r", gi, {0, 0},
                               {batchSize_, EMBEDDING_SIZE});
    Node *i_i = G->createSlice("pytorch.GRU.i_i", gi, {0, EMBEDDING_SIZE},
                               {batchSize_, 2 * EMBEDDING_SIZE});
    Node *i_n = G->createSlice("pytorch.GRU.i_n", gi, {0, 2 * EMBEDDING_SIZE},
                               {batchSize_, 3 * EMBEDDING_SIZE});

    Node *h_r = G->createSlice("pytorch.GRU.h_r", gh, {0, 0},
                               {batchSize_, EMBEDDING_SIZE});
    Node *h_i = G->createSlice("pytorch.GRU.h_i", gh, {0, EMBEDDING_SIZE},
                               {batchSize_, 2 * EMBEDDING_SIZE});
    Node *h_n = G->createSlice("pytorch.GRU.h_n", gh, {0, 2 * EMBEDDING_SIZE},
                               {batchSize_, 3 * EMBEDDING_SIZE});

    Node *resetgate = G->createSigmoid("pytorch.GRU.resetgate",
                                       G->createAdd("i_r_plus_h_r", i_r, h_r));
    Node *inputgate = G->createSigmoid("pytorch.GRU.inputgate",
                                       G->createAdd("i_i_plus_h_i", i_i, h_i));
    Node *newgate = G->createTanh(
        "pytorch.GRU.newgate",
        G->createAdd("i_n_plus_rg_mult_h_n", i_n,
                     G->createMul("rg_mult_h_n", resetgate, h_n)));
    return G->createAdd(
        "pytorch.GRU.hy", newgate,
        G->createMul("ig_mult_hmng", inputgate,
                     G->createSub("hidden_minus_newgate", hidden, newgate)));
  }
};

void Model::loadLanguages() {
  fr_.loadVocabularyFromFile("fr2en/fr_vocabulary.txt");
  en_.loadVocabularyFromFile("fr2en/en_vocabulary.txt");
  embedding_fr_ = loadEmbedding("fr", fr_.index2word_.size());
  embedding_en_ = loadEmbedding("en", en_.index2word_.size());
}

/// Model part representing Encoder. Remembers input sentence into hidden layer.
/// \p input is Variable representing the sentence.
/// \p seqLength is Variable representing the length of sentence.
/// \p encoderHiddenOutput saves resulting hidden layer.
void Model::loadEncoder() {
  auto &mod = EE_.getModule();
  input_ = mod.createPlaceholder(ElemKind::Int64ITy, {batchSize_, MAX_LENGTH},
                                 "encoder.inputsentence", false);
  bindings.allocate(input_);
  seqLength_ = mod.createPlaceholder(ElemKind::Int64ITy, {batchSize_},
                                     "encoder.seqLength", false);
  bindings.allocate(seqLength_);

  auto *hiddenInit =
      mod.createPlaceholder(ElemKind::FloatTy, {batchSize_, EMBEDDING_SIZE},
                            "encoder.hiddenInit", false);
  auto *hiddenInitTensor = bindings.allocate(hiddenInit);
  hiddenInitTensor->zero();

  Node *hidden = hiddenInit;

  auto *wIh = mod.createPlaceholder(
      ElemKind::FloatTy, {EMBEDDING_SIZE, HIDDEN_SIZE}, "encoder.w_ih", false);
  auto *bIh = mod.createPlaceholder(ElemKind::FloatTy, {HIDDEN_SIZE},
                                    "encoder.b_ih", false);
  auto *wHh = mod.createPlaceholder(
      ElemKind::FloatTy, {EMBEDDING_SIZE, HIDDEN_SIZE}, "encoder.w_hh", false);
  auto *bHh = mod.createPlaceholder(ElemKind::FloatTy, {HIDDEN_SIZE},
                                    "encoder.b_hh", false);

  loadMatrixFromFile("fr2en/encoder_w_ih.bin", *bindings.allocate(wIh));
  loadMatrixFromFile("fr2en/encoder_b_ih.bin", *bindings.allocate(bIh));
  loadMatrixFromFile("fr2en/encoder_w_hh.bin", *bindings.allocate(wHh));
  loadMatrixFromFile("fr2en/encoder_b_hh.bin", *bindings.allocate(bHh));

  Node *inputEmbedded =
      F_->createGather("encoder.embedding", embedding_fr_, input_);

  // TODO: encoder does exactly MAX_LENGTH steps, while input size is smaller.
  // We could use control flow here.
  std::vector<NodeValue> outputs;
  for (unsigned step = 0; step < MAX_LENGTH; step++) {
    Node *inputSlice = F_->createSlice(
        "encoder." + std::to_string(step) + ".inputSlice", inputEmbedded,
        {0, step, 0}, {batchSize_, step + 1, EMBEDDING_SIZE});
    Node *reshape =
        F_->createReshape("encoder." + std::to_string(step) + ".reshape",
                          inputSlice, {batchSize_, EMBEDDING_SIZE}, ANY_LAYOUT);
    hidden = createPyTorchGRUCell(F_, reshape, hidden, wIh, bIh, wHh, bHh);
    outputs.push_back(hidden);
  }

  Node *output = F_->createConcat("encoder.output", outputs, 1);
  Node *r2 =
      F_->createReshape("encoder.output.r2", output,
                        {MAX_LENGTH * batchSize_, EMBEDDING_SIZE}, ANY_LAYOUT);

  encoderHiddenOutput_ = F_->createGather("encoder.outputNth", r2, seqLength_);
}

/// Model part representing Decoder.
/// Uses \p encoderHiddenOutput as final state from Encoder.
/// Resulting translation is put into \p output Variable.
void Model::loadDecoder() {
  auto &mod = EE_.getModule();
  auto *input = mod.createPlaceholder(ElemKind::Int64ITy, {batchSize_},
                                      "decoder.input", false);
  auto *inputTensor = bindings.allocate(input);
  for (dim_t i = 0; i < batchSize_; i++) {
    inputTensor->getHandle<int64_t>().at({i}) = en_.word2index_["SOS"];
  }

  auto *wIh = mod.createPlaceholder(
      ElemKind::FloatTy, {EMBEDDING_SIZE, HIDDEN_SIZE}, "decoder.w_ih", false);
  auto *bIh = mod.createPlaceholder(ElemKind::FloatTy, {HIDDEN_SIZE},
                                    "decoder.b_ih", false);
  auto *wHh = mod.createPlaceholder(
      ElemKind::FloatTy, {EMBEDDING_SIZE, HIDDEN_SIZE}, "decoder.w_hh", false);
  auto *bHh = mod.createPlaceholder(ElemKind::FloatTy, {HIDDEN_SIZE},
                                    "decoder.b_hh", false);
  auto *outW = mod.createPlaceholder(
      ElemKind::FloatTy, {EMBEDDING_SIZE, (dim_t)en_.index2word_.size()},
      "decoder.out_w", false);
  auto *outB =
      mod.createPlaceholder(ElemKind::FloatTy, {(dim_t)en_.index2word_.size()},
                            "decoder.out_b", false);
  loadMatrixFromFile("fr2en/decoder_w_ih.bin", *bindings.allocate(wIh));
  loadMatrixFromFile("fr2en/decoder_b_ih.bin", *bindings.allocate(bIh));
  loadMatrixFromFile("fr2en/decoder_w_hh.bin", *bindings.allocate(wHh));
  loadMatrixFromFile("fr2en/decoder_b_hh.bin", *bindings.allocate(bHh));
  loadMatrixFromFile("fr2en/decoder_out_w.bin", *bindings.allocate(outW));
  loadMatrixFromFile("fr2en/decoder_out_b.bin", *bindings.allocate(outB));

  Node *hidden = encoderHiddenOutput_;
  Node *lastWordIdx = input;

  std::vector<NodeValue> outputs;
  // TODO: decoder does exactly MAX_LENGTH steps, while translation could be
  // smaller. We could use control flow here.
  for (unsigned step = 0; step < MAX_LENGTH; step++) {
    // Use last translated word as an input at the current step.
    Node *embedded =
        F_->createGather("decoder.embedding." + std::to_string(step),
                         embedding_en_, lastWordIdx);

    Node *relu = F_->createRELU("decoder.relu", embedded);
    hidden = createPyTorchGRUCell(F_, relu, hidden, wIh, bIh, wHh, bHh);

    Node *FC = F_->createFullyConnected("decoder.outFC", hidden, outW, outB);
    auto *topK = F_->createTopK("decoder.topK", FC, 1);

    lastWordIdx = F_->createReshape("decoder.reshape", topK->getIndices(),
                                    {batchSize_}, "N");
    outputs.push_back(lastWordIdx);
  }

  Node *concat = F_->createConcat("decoder.output.concat", outputs, 0);
  Node *reshape = F_->createReshape("decoder.output.reshape", concat,
                                    {MAX_LENGTH, batchSize_}, ANY_LAYOUT);
  auto *save = F_->createSave("decoder.output", reshape);
  output_ = save->getPlaceholder();
  bindings.allocate(output_);
}

/// Translation has 2 stages:
/// 1) Input sentence is fed into Encoder word by word.
/// 2) "Memory" of Encoder is written into memory of Decoder.
///    Now Decoder streams resulting translation word by word.
void Model::translate(const std::vector<std::string> &batch) {
  Tensor input(ElemKind::Int64ITy, {batchSize_, MAX_LENGTH});
  Tensor seqLength(ElemKind::Int64ITy, {batchSize_});
  input.zero();

  for (dim_t j = 0; j < batch.size(); j++) {
    std::istringstream iss(batch[j]);
    std::vector<std::string> words;
    std::string word;
    while (iss >> word)
      words.push_back(word);
    words.push_back("EOS");

    CHECK_LE(words.size(), MAX_LENGTH) << "sentence is too long.";

    for (dim_t i = 0; i < words.size(); i++) {
      auto iter = fr_.word2index_.find(words[i]);
      CHECK(iter != fr_.word2index_.end()) << "Unknown word: " << words[i];
      input.getHandle<int64_t>().at({j, i}) = iter->second;
    }
    seqLength.getHandle<int64_t>().at({j}) =
        (words.size() - 1) + j * MAX_LENGTH;
  }

  updateInputPlaceholders(bindings, {input_, seqLength_}, {&input, &seqLength});
  EE_.run(bindings);

  auto OH = bindings.get(output_)->getHandle<int64_t>();
  for (unsigned j = 0; j < batch.size(); j++) {
    for (unsigned i = 0; i < MAX_LENGTH; i++) {
      dim_t wordIdx = OH.at({i, j});
      if (wordIdx == en_.word2index_["EOS"])
        break;

      if (i)
        std::cout << ' ';
      if (en_.index2word_.size() > (wordIdx))
        std::cout << en_.index2word_[wordIdx];
      else
        std::cout << "[" << wordIdx << "]";
    }
    std::cout << "\n\n";
  }

  if (!dumpProfileFileOpt.empty()) {
    std::vector<NodeProfilingInfo> PI =
        quantization::generateNodeProfilingInfos(bindings, F_, loweredMap_);
    serializeProfilingInfosToYaml(dumpProfileFileOpt,
                                  /* graphPreLowerHash */ 0, PI);
  }
}

int main(int argc, char **argv) {
  std::array<const llvm::cl::OptionCategory *, 3> showCategories = {
      {&debugCat, &quantizationCat, &fr2enCat}};
  llvm::cl::HideUnrelatedOptions(showCategories);
  llvm::cl::ParseCommandLineOptions(
      argc, argv, "Translate sentences from French to English");

  Model seq2seq(batchSizeOpt);
  seq2seq.loadLanguages();
  seq2seq.loadEncoder();
  seq2seq.loadDecoder();
  seq2seq.compile();

  if (!dumpGraphDAGFileOpt.empty()) {
    seq2seq.dumpGraphDAG(dumpGraphDAGFileOpt.c_str());
  }

  std::cout << "Please enter a sentence in French, such that its English "
            << "translation starts with one of the following:\n"
            << "\ti am\n"
            << "\the is\n"
            << "\tshe is\n"
            << "\tyou are\n"
            << "\twe are\n"
            << "\tthey are\n"
            << "\n"
            << "Here are some examples:\n"
            << "\tnous sommes desormais en securite .\n"
            << "\tvous etes puissantes .\n"
            << "\til etudie l histoire a l universite .\n"
            << "\tje ne suis pas timide .\n"
            << "\tj y songe encore .\n"
            << "\tje suis maintenant a l aeroport .\n\n";

  llvm::Timer timer("Translate", "Translate");
  if (timeOpt) {
    timer.startTimer();
  }

  std::vector<std::string> batch;
  do {
    batch.clear();
    for (size_t i = 0; i < batchSizeOpt; i++) {
      std::string sentence;
      if (!getline(std::cin, sentence)) {
        break;
      }
      batch.push_back(sentence);
    }
    if (!batch.empty()) {
      seq2seq.translate(batch);
    }
  } while (batch.size() == batchSizeOpt);

  if (timeOpt) {
    timer.stopTimer();
  }

  return 0;
}
