#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Graph/Graph.h"
#include "glow/Support/Support.h"

#include <algorithm>
#include <cassert>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>

using namespace glow;

const unsigned MAX_LENGTH = 10;
const unsigned EMBEDDING_SIZE = 256;
const unsigned HIDDEN_SIZE = EMBEDDING_SIZE * 3;

/// Stores vocabulary of a language. Contains mapping from word to index and
/// vice versa.
struct Vocabulary {
  std::vector<std::string> index2word_;
  std::unordered_map<std::string, size_t> word2index_;

  void addWord(llvm::StringRef word) {
    word2index_[word] = index2word_.size();
    index2word_.push_back(word);
  }

  Vocabulary() = default;

  void loadVocabularyFromFile(llvm::StringRef filename) {
    std::ifstream file(filename);
    std::string word;
    while (getline(file, word))
      addWord(word);
  }
};

/// Loads tensor of floats from binary file.
void loadMatrixFromFile(llvm::StringRef filename, Tensor &result) {
  std::ifstream file(filename.str(), std::ios::binary);
  if (!file.read((char *)result.getRawDataPointer<float>(),
                 result.size() * sizeof(float))) {
    std::cout << "Error reading file: " << filename.str() << '\n';
    exit(1);
  }
}

Node *createPyTorchGRUCell(Graph *G, Node *input, Node *hidden, Variable *w_ih,
                           Variable *b_ih, Variable *w_hh, Variable *b_hh) {
  // reference implementation:
  // https://github.com/pytorch/pytorch/blob/dd5c195646b941d3e20a72847ac48c41e272b8b2/torch/nn/_functions/rnn.py#L46
  Node *gi =
      G->createFullyConnected("pytorch.GRU.gi", input, w_ih, b_ih, HIDDEN_SIZE);
  Node *gh = G->createFullyConnected("pytorch.GRU.gh", hidden, w_hh, b_hh,
                                     HIDDEN_SIZE);

  Node *i_r =
      G->createSlice("pytorch.GRU.i_r", gi, {0, 0}, {1, EMBEDDING_SIZE});
  Node *i_i = G->createSlice("pytorch.GRU.i_i", gi, {0, EMBEDDING_SIZE},
                             {1, 2 * EMBEDDING_SIZE});
  Node *i_n = G->createSlice("pytorch.GRU.i_n", gi, {0, 2 * EMBEDDING_SIZE},
                             {1, 3 * EMBEDDING_SIZE});

  Node *h_r =
      G->createSlice("pytorch.GRU.h_r", gh, {0, 0}, {1, EMBEDDING_SIZE});
  Node *h_i = G->createSlice("pytorch.GRU.h_i", gh, {0, EMBEDDING_SIZE},
                             {1, 2 * EMBEDDING_SIZE});
  Node *h_n = G->createSlice("pytorch.GRU.h_n", gh, {0, 2 * EMBEDDING_SIZE},
                             {1, 3 * EMBEDDING_SIZE});

  Node *resetgate = G->createSigmoid(
      "pytorch.GRU.resetgate",
      G->createArithmetic("i_r_plus_h_r", i_r, h_r, ArithmeticNode::Mode::Add));
  Node *inputgate = G->createSigmoid(
      "pytorch.GRU.inputgate",
      G->createArithmetic("i_i_plus_h_i", i_i, h_i, ArithmeticNode::Mode::Add));
  Node *newgate = G->createTanh(
      "pytorch.GRU.newgate",
      G->createArithmetic("i_n_plus_rg_mult_h_n", i_n,
                          G->createArithmetic("rg_mult_h_n", resetgate, h_n,
                                              ArithmeticNode::Mode::Mul),
                          ArithmeticNode::Mode::Add));
  return G->createArithmetic(
      "pytorch.GRU.hy", newgate,
      G->createArithmetic("ig_mult_hmng", inputgate,
                          G->createArithmetic("hidden_minus_newgate", hidden,
                                              newgate,
                                              ArithmeticNode::Mode::Sub),
                          ArithmeticNode::Mode::Mul),
      ArithmeticNode::Mode::Add);
}

/// Represents a single RNN model: either encoder or decoder.
/// Stores vocabulary, compiled Graph (ready to be executed), embedding and
/// few references to input/output Variables.
struct Model {
  ExecutionEngine EE;
  Vocabulary L;
  Tensor embedding;
  Variable *input;
  Variable *hiddenOutput;

  void loadLanguage(llvm::StringRef lang_prefix) {
    L.loadVocabularyFromFile("fr2en/" + lang_prefix.str() + "_vocabulary.txt");
    embedding.reset(ElemKind::FloatTy, {L.index2word_.size(), EMBEDDING_SIZE});
    loadMatrixFromFile("fr2en/" + lang_prefix.str() + "_embedding.bin",
                       embedding);
  }
};

/// RNN representing Encoder. Remembers input sentense into hidden layer.
/// \p input is Variable representing the sentense. MAX_LENGTH x EMBEDDING_SIZE
/// \p hiddenOutput is Variable representing hidden layer states over time.
/// MAX_LENGTH x EMBEDDING_SIZE
struct Encoder : Model {

  void loadEncoder() {
    Graph &G = EE.getGraph();
    input = G.createVariable(ElemKind::FloatTy, {MAX_LENGTH, EMBEDDING_SIZE},
                             "encoder.inputSentense",
                             Variable::VisibilityKind::Public,
                             Variable::TrainKind::None);
    Variable *hiddenInit = G.createVariable(
        ElemKind::FloatTy, {1, EMBEDDING_SIZE}, "encoder.hiddenInit",
        Variable::VisibilityKind::Public, Variable::TrainKind::None);
    hiddenInit->getPayload().zero();
    hiddenOutput = G.createVariable(
        ElemKind::FloatTy, {MAX_LENGTH, EMBEDDING_SIZE}, "encoder.hiddenOutput",
        Variable::VisibilityKind::Public, Variable::TrainKind::None);

    Node *hidden = hiddenInit;

    Variable *w_ih = G.createVariable(
        ElemKind::FloatTy, {EMBEDDING_SIZE, HIDDEN_SIZE}, "encoder.w_ih",
        Variable::VisibilityKind::Public, Variable::TrainKind::None);
    Variable *b_ih = G.createVariable(
        ElemKind::FloatTy, {HIDDEN_SIZE}, "encoder.b_ih",
        Variable::VisibilityKind::Public, Variable::TrainKind::None);
    Variable *w_hh = G.createVariable(
        ElemKind::FloatTy, {EMBEDDING_SIZE, HIDDEN_SIZE}, "encoder.w_hh",
        Variable::VisibilityKind::Public, Variable::TrainKind::None);
    Variable *b_hh = G.createVariable(
        ElemKind::FloatTy, {HIDDEN_SIZE}, "encoder.b_hh",
        Variable::VisibilityKind::Public, Variable::TrainKind::None);
    loadMatrixFromFile("fr2en/encoder_w_ih.bin", w_ih->getPayload());
    loadMatrixFromFile("fr2en/encoder_b_ih.bin", b_ih->getPayload());
    loadMatrixFromFile("fr2en/encoder_w_hh.bin", w_hh->getPayload());
    loadMatrixFromFile("fr2en/encoder_b_hh.bin", b_hh->getPayload());

    std::vector<Node *> outputs;
    for (unsigned step = 0; step < MAX_LENGTH; step++) {
      Node *inputSlice =
          G.createSlice("encoder." + std::to_string(step) + ".inputSlice",
                        input, {step, 0}, {step + 1, EMBEDDING_SIZE});
      hidden =
          createPyTorchGRUCell(&G, inputSlice, hidden, w_ih, b_ih, w_hh, b_hh);
      outputs.push_back(hidden);
    }

    Node *output = G.createConcat("encoder.output", outputs, 0);
    G.createSave("encoder.saveOutput", output, hiddenOutput);

    EE.compile(CompilationMode::Infer);
  }
};

/// Decoder model, currently not a RNN. Graph contains one step of the net,
/// \p input contains last word of translation. 1 x EMBEDDING_SIZE
/// \p hiddenInput contains hidden layer from previous step. 1 x EMBEDDING_SIZE
/// \p output saves resulting word from current step. 1 x vocabularySize
/// \p hiddenOutput saves hidden layer from current step. 1 x EMBEDDING_SIZE
struct Decoder : Model {
  Variable *hiddenInput;
  Variable *output;

  void loadDecoder() {
    Graph &G = EE.getGraph();
    input = G.createVariable(
        ElemKind::FloatTy, {1, EMBEDDING_SIZE}, "decoder.selfInput",
        Variable::VisibilityKind::Public, Variable::TrainKind::None);
    output = G.createVariable(
        ElemKind::FloatTy, {1, L.index2word_.size()}, "decoder.output",
        Variable::VisibilityKind::Public, Variable::TrainKind::None);
    hiddenInput = G.createVariable(
        ElemKind::FloatTy, {1, EMBEDDING_SIZE}, "decoder.hiddenInput",
        Variable::VisibilityKind::Public, Variable::TrainKind::None);
    hiddenOutput = G.createVariable(
        ElemKind::FloatTy, {1, EMBEDDING_SIZE}, "decoder.hiddenOutput",
        Variable::VisibilityKind::Public, Variable::TrainKind::None);

    Variable *w_ih = G.createVariable(
        ElemKind::FloatTy, {EMBEDDING_SIZE, HIDDEN_SIZE}, "decoder.w_ih",
        Variable::VisibilityKind::Public, Variable::TrainKind::None);
    Variable *b_ih = G.createVariable(
        ElemKind::FloatTy, {HIDDEN_SIZE}, "decoder.b_ih",
        Variable::VisibilityKind::Public, Variable::TrainKind::None);
    Variable *w_hh = G.createVariable(
        ElemKind::FloatTy, {EMBEDDING_SIZE, HIDDEN_SIZE}, "decoder.w_hh",
        Variable::VisibilityKind::Public, Variable::TrainKind::None);
    Variable *b_hh = G.createVariable(
        ElemKind::FloatTy, {HIDDEN_SIZE}, "decoder.b_hh",
        Variable::VisibilityKind::Public, Variable::TrainKind::None);
    Variable *out_w = G.createVariable(
        ElemKind::FloatTy, {EMBEDDING_SIZE, L.index2word_.size()},
        "decoder.out_w", Variable::VisibilityKind::Public,
        Variable::TrainKind::None);
    Variable *out_b = G.createVariable(
        ElemKind::FloatTy, {L.index2word_.size()}, "decoder.out_b",
        Variable::VisibilityKind::Public, Variable::TrainKind::None);
    loadMatrixFromFile("fr2en/decoder_w_ih.bin", w_ih->getPayload());
    loadMatrixFromFile("fr2en/decoder_b_ih.bin", b_ih->getPayload());
    loadMatrixFromFile("fr2en/decoder_w_hh.bin", w_hh->getPayload());
    loadMatrixFromFile("fr2en/decoder_b_hh.bin", b_hh->getPayload());
    loadMatrixFromFile("fr2en/decoder_out_w.bin", out_w->getPayload());
    loadMatrixFromFile("fr2en/decoder_out_b.bin", out_b->getPayload());

    Node *relu = G.createRELU("decoder.relu", input);
    Node *hidden =
        createPyTorchGRUCell(&G, relu, hiddenInput, w_ih, b_ih, w_hh, b_hh);

    Node *FC = G.createFullyConnected("decoder.outFC", hidden, out_w, out_b,
                                      L.index2word_.size());
    G.createSave("decoder.output", FC, output);
    G.createSave("decoder.hiddenOutput", hidden, hiddenOutput);

    EE.compile(CompilationMode::Infer);
  }
};

/// Translation has 2 stages:
/// 1) Input sentense is fed into Encoder model word by word.
/// 2) "Memory" of Encoder is written into memory of Decoder.
///    Now Decoder streams resulting translation word by word.
void translate(Encoder *encoder, Decoder *decoder, llvm::StringRef sentense,
               llvm::StringRef translation) {
  std::istringstream iss(sentense);
  std::vector<std::string> words;
  std::string word;
  while (iss >> word)
    words.push_back(word);
  words.push_back("EOS");

  GLOW_ASSERT(words.size() <= MAX_LENGTH && "Sentense is too long.");

  // TODO: this function must become a single EE.run() call.

  for (size_t i = 0; i < words.size(); i++) {
    auto iter = encoder->L.word2index_.find(words[i]);
    GLOW_ASSERT(iter != encoder->L.word2index_.end() && "Unknown word.");
    size_t wordIndex = iter->second;
    for (unsigned j = 0; j < EMBEDDING_SIZE; j++)
      encoder->input->getPayload().getHandle().at({i, j}) =
          encoder->embedding.getHandle().at({wordIndex, j});
  }

  // TODO: encoder does exactly MAX_LENGTH steps, while input size is smaller.
  // We could use control flow here.
  encoder->EE.run({}, {});

  // TODO: this is not a real RNN. The current Glow IR misses Argmax and
  // Embegging.
  for (unsigned j = 0; j < EMBEDDING_SIZE; j++)
    decoder->hiddenInput->getPayload().getHandle().at({0, j}) =
        encoder->hiddenOutput->getHandle().at({words.size() - 1, j});
  size_t prevWordIdx = decoder->L.word2index_["SOS"];
  for (size_t len = 0; len < MAX_LENGTH * 2; len++) {
    // Use last translated word as an input at the current step.
    for (unsigned j = 0; j < EMBEDDING_SIZE; j++)
      decoder->input->getPayload().getHandle().at({0, j}) =
          decoder->embedding.getHandle().at({prevWordIdx, j});

    decoder->EE.run({}, {});

    decoder->hiddenInput->getPayload().copyFrom(
        &decoder->hiddenOutput->getPayload());

    prevWordIdx = decoder->output->getPayload()
                      .getHandle()
                      .extractSlice(0)
                      .getHandle()
                      .maxArg();

    if (prevWordIdx == decoder->L.word2index_["EOS"])
      break;

    if (len)
      std::cout << ' ';
    std::cout << decoder->L.index2word_[prevWordIdx];
  }
  std::cout << '\n';
}

int main() {
  Encoder encoder;
  encoder.loadLanguage("fr");
  encoder.loadEncoder();

  Decoder decoder;
  decoder.loadLanguage("en");
  decoder.loadDecoder();

  std::cout << "Please enter a sentense in French, such that it's English "
            << "translation starts with one of the following:\n"
            << "\ti am\n"
            << "\the is\n"
            << "\tshe is\n"
            << "\tyou are\n"
            << "\twe are\n"
            << "\tthey are\n"
            << "Here are some examples:\n"
            << "nous sommes desormais en securite .\n"
            << "vous etes puissantes .\n"
            << "il etudie l histoire a l universite .\n"
            << "je ne suis pas timide .\n"
            << "j y songe encore .\n"
            << "je suis maintenant a l aeroport .\n\n";

  std::string sentense, translation;
  while (getline(std::cin, sentense)) {
    translate(&encoder, &decoder, sentense, translation);
    std::cout << translation << '\n';
  }

  return 0;
}
