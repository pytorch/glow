#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Graph/Graph.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

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
    std::cout << "Error reading file: " << filename.str() << '\n'
              << "Need to be downloaded by calling:\n"
              << "python ../Glow/utils/download_test_db.py -d fr2en\n";
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

/// Represents a single RNN model: encoder combined with decoder.
/// Stores vocabulary, compiled Graph (ready to be executed), and
/// few references to input/output Variables.
struct Model {
  ExecutionEngine EE_;
  Vocabulary en_, fr_;
  Variable *input_;
  Variable *seqLength_;
  Variable *output_;

  void loadLanguages();
  void loadEncoder();
  void loadDecoder();

private:
  Variable *embedding_fr_, *embedding_en_;
  Node *encoderHiddenOutput_;

  Variable *loadEmbedding(llvm::StringRef langPrefix, size_t langSize) {
    Variable *result = EE_.getGraph().createVariable(
        ElemKind::FloatTy, {langSize, EMBEDDING_SIZE},
        "embedding." + langPrefix.str(), Variable::VisibilityKind::Private,
        Variable::TrainKind::None);
    loadMatrixFromFile("fr2en/" + langPrefix.str() + "_embedding.bin",
                       result->getPayload());
    return result;
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
  Graph &G = EE_.getGraph();
  input_ = G.createVariable(
      ElemKind::IndexTy, {MAX_LENGTH}, "encoder.inputsentence",
      Variable::VisibilityKind::Public, Variable::TrainKind::None);
  seqLength_ = G.createVariable(ElemKind::IndexTy, {1}, "encoder.seqLength",
                                Variable::VisibilityKind::Public,
                                Variable::TrainKind::None);

  Variable *hiddenInit = G.createVariable(
      ElemKind::FloatTy, {1, EMBEDDING_SIZE}, "encoder.hiddenInit",
      Variable::VisibilityKind::Private, Variable::TrainKind::None);
  hiddenInit->getPayload().zero();

  Node *hidden = hiddenInit;

  Variable *w_ih = G.createVariable(
      ElemKind::FloatTy, {EMBEDDING_SIZE, HIDDEN_SIZE}, "encoder.w_ih",
      Variable::VisibilityKind::Private, Variable::TrainKind::None);
  Variable *b_ih = G.createVariable(
      ElemKind::FloatTy, {HIDDEN_SIZE}, "encoder.b_ih",
      Variable::VisibilityKind::Private, Variable::TrainKind::None);
  Variable *w_hh = G.createVariable(
      ElemKind::FloatTy, {EMBEDDING_SIZE, HIDDEN_SIZE}, "encoder.w_hh",
      Variable::VisibilityKind::Private, Variable::TrainKind::None);
  Variable *b_hh = G.createVariable(
      ElemKind::FloatTy, {HIDDEN_SIZE}, "encoder.b_hh",
      Variable::VisibilityKind::Private, Variable::TrainKind::None);
  loadMatrixFromFile("fr2en/encoder_w_ih.bin", w_ih->getPayload());
  loadMatrixFromFile("fr2en/encoder_b_ih.bin", b_ih->getPayload());
  loadMatrixFromFile("fr2en/encoder_w_hh.bin", w_hh->getPayload());
  loadMatrixFromFile("fr2en/encoder_b_hh.bin", b_hh->getPayload());

  Node *inputEmbedded =
      G.createGather("encoder.embedding", embedding_fr_, input_);

  // TODO: encoder does exactly MAX_LENGTH steps, while input size is smaller.
  // We could use control flow here.
  std::vector<Node *> outputs;
  for (unsigned step = 0; step < MAX_LENGTH; step++) {
    Node *inputSlice =
        G.createSlice("encoder." + std::to_string(step) + ".inputSlice",
                      inputEmbedded, {step, 0}, {step + 1, EMBEDDING_SIZE});
    hidden =
        createPyTorchGRUCell(&G, inputSlice, hidden, w_ih, b_ih, w_hh, b_hh);
    outputs.push_back(hidden);
  }

  Node *output = G.createConcat("encoder.output", outputs, 0);
  encoderHiddenOutput_ =
      G.createGather("encoder.outputNth", output, seqLength_);
}

/// Model part representing Decoder.
/// Uses \p encoderHiddenOutput as final state from Encoder.
/// Resulting translation is put into \p output Variable.
void Model::loadDecoder() {
  Graph &G = EE_.getGraph();
  Variable *input = G.createVariable(ElemKind::IndexTy, {1}, "decoder.input",
                                     Variable::VisibilityKind::Public,
                                     Variable::TrainKind::None);
  input->getPayload().getHandle<size_t>().at({0}) = en_.word2index_["SOS"];

  Variable *w_ih = G.createVariable(
      ElemKind::FloatTy, {EMBEDDING_SIZE, HIDDEN_SIZE}, "decoder.w_ih",
      Variable::VisibilityKind::Private, Variable::TrainKind::None);
  Variable *b_ih = G.createVariable(
      ElemKind::FloatTy, {HIDDEN_SIZE}, "decoder.b_ih",
      Variable::VisibilityKind::Private, Variable::TrainKind::None);
  Variable *w_hh = G.createVariable(
      ElemKind::FloatTy, {EMBEDDING_SIZE, HIDDEN_SIZE}, "decoder.w_hh",
      Variable::VisibilityKind::Private, Variable::TrainKind::None);
  Variable *b_hh = G.createVariable(
      ElemKind::FloatTy, {HIDDEN_SIZE}, "decoder.b_hh",
      Variable::VisibilityKind::Private, Variable::TrainKind::None);
  Variable *out_w = G.createVariable(
      ElemKind::FloatTy, {EMBEDDING_SIZE, en_.index2word_.size()},
      "decoder.out_w", Variable::VisibilityKind::Public,
      Variable::TrainKind::None);
  Variable *out_b = G.createVariable(
      ElemKind::FloatTy, {en_.index2word_.size()}, "decoder.out_b",
      Variable::VisibilityKind::Public, Variable::TrainKind::None);
  loadMatrixFromFile("fr2en/decoder_w_ih.bin", w_ih->getPayload());
  loadMatrixFromFile("fr2en/decoder_b_ih.bin", b_ih->getPayload());
  loadMatrixFromFile("fr2en/decoder_w_hh.bin", w_hh->getPayload());
  loadMatrixFromFile("fr2en/decoder_b_hh.bin", b_hh->getPayload());
  loadMatrixFromFile("fr2en/decoder_out_w.bin", out_w->getPayload());
  loadMatrixFromFile("fr2en/decoder_out_b.bin", out_b->getPayload());

  Node *hidden = encoderHiddenOutput_;
  Node *lastWordIdx = input;

  std::vector<Node *> outputs;
  // TODO: decoder does exactly MAX_LENGTH steps, while translation could be
  // smaller. We could use control flow here.
  for (unsigned step = 0; step < MAX_LENGTH; step++) {
    // Use last translated word as an input at the current step.
    Node *embedded = G.createGather("decoder.embedding." + std::to_string(step),
                                    embedding_en_, lastWordIdx);

    Node *relu = G.createRELU("decoder.relu", embedded);
    hidden = createPyTorchGRUCell(&G, relu, hidden, w_ih, b_ih, w_hh, b_hh);

    Node *FC = G.createFullyConnected("decoder.outFC", hidden, out_w, out_b,
                                      en_.index2word_.size());
    Node *topK = G.createTopK("decoder.topK", FC, 1);

    lastWordIdx = G.createReshape("decoder.reshape", {topK, 1}, {1});
    outputs.push_back(lastWordIdx);
  }

  Node *concat = G.createConcat("decoder.output", outputs, 0);

  output_ = G.createVariable(ElemKind::IndexTy, {MAX_LENGTH}, "decoder.output",
                             Variable::VisibilityKind::Public,
                             Variable::TrainKind::None);
  G.createSave("decoder.output", concat, output_);

  EE_.compile(CompilationMode::Infer);
}

/// Translation has 2 stages:
/// 1) Input sentence is fed into Encoder word by word.
/// 2) "Memory" of Encoder is written into memory of Decoder.
///    Now Decoder streams resulting translation word by word.
void translate(Model *seq2seq, llvm::StringRef sentence) {
  std::istringstream iss(sentence);
  std::vector<std::string> words;
  std::string word;
  while (iss >> word)
    words.push_back(word);
  words.push_back("EOS");

  GLOW_ASSERT(words.size() <= MAX_LENGTH && "sentence is too long.");

  Tensor input(ElemKind::IndexTy, {MAX_LENGTH});
  Tensor seqLength(ElemKind::IndexTy, {1});

  input.zero();
  for (size_t i = 0; i < words.size(); i++) {
    auto iter = seq2seq->fr_.word2index_.find(words[i]);
    GLOW_ASSERT(iter != seq2seq->fr_.word2index_.end() && "Unknown word.");
    input.getHandle<size_t>().at({i}) = iter->second;
  }
  seqLength.getHandle<size_t>().at({0}) = words.size() - 1;

  seq2seq->EE_.run({seq2seq->input_, seq2seq->seqLength_},
                   {&input, &seqLength});

  auto OH = seq2seq->output_->getPayload().getHandle<size_t>();
  for (unsigned i = 0; i < MAX_LENGTH; i++) {
    size_t wordIdx = OH.at({i});
    if (wordIdx == seq2seq->en_.word2index_["EOS"])
      break;

    if (i)
      std::cout << ' ';
    std::cout << seq2seq->en_.index2word_[wordIdx];
  }
  std::cout << "\n\n";
}

int main() {
  Model seq2seq;
  seq2seq.loadLanguages();
  seq2seq.loadEncoder();
  seq2seq.loadDecoder();

  std::cout << "Please enter a sentence in French, such that it's English "
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

  std::string sentence;
  while (getline(std::cin, sentence)) {
    translate(&seq2seq, sentence);
  }

  return 0;
}
