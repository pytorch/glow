#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Graph/Graph.h"
#include "glow/Quantization/Serialization.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Timer.h"

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
    "dumpGraphDAG",
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

llvm::cl::opt<BackendKind> ExecutionBackend(
    llvm::cl::desc("Backend to use:"),
    llvm::cl::values(clEnumValN(BackendKind::Interpreter, "interpreter",
                                "Use interpreter"),
                     clEnumValN(BackendKind::CPU, "cpu", "Use CPU"),
                     clEnumValN(BackendKind::OpenCL, "opencl", "Use OpenCL")),
    llvm::cl::init(BackendKind::Interpreter), llvm::cl::cat(fr2enCat));

/// Quantization options.
llvm::cl::OptionCategory quantizationCat("Quantization Options");

llvm::cl::opt<std::string> dumpProfileFileOpt(
    "dump_profile",
    llvm::cl::desc("Perform quantization profiling for a given graph "
                   "and dump result to the file."),
    llvm::cl::value_desc("profile.yaml"), llvm::cl::Optional,
    llvm::cl::cat(quantizationCat));

llvm::cl::opt<std::string> loadProfileFileOpt(
    "load_profile",
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

/// Represents a single RNN model: encoder combined with decoder.
/// Stores vocabulary, compiled Graph (ready to be executed), and
/// few references to input/output Variables.
struct Model {
  unsigned batchSize_;
  ExecutionEngine EE_{ExecutionBackend};
  Vocabulary en_, fr_;
  Variable *input_;
  Variable *seqLength_;
  Variable *output_;

  void loadLanguages();
  void loadEncoder();
  void loadDecoder();
  void translate(const std::vector<std::string> &batch);

  Model(unsigned batchSize) : batchSize_(batchSize) {
    EE_.getModule().createFunction("main");
  }

  void dumpGraphDAG(const char *filename) {
    EE_.getModule().getFunction("main")->dumpDAG(filename);
  }

  void compile() {
    EE_.compile(CompilationMode::Infer, EE_.getModule().getFunction("main"));
  }

private:
  Variable *embedding_fr_, *embedding_en_;
  Node *encoderHiddenOutput_;

  Variable *loadEmbedding(llvm::StringRef langPrefix, size_t langSize) {
    auto &mod = EE_.getModule();
    Variable *result = mod.createVariable(
        ElemKind::FloatTy, {langSize, EMBEDDING_SIZE},
        "embedding." + langPrefix.str(), Variable::VisibilityKind::Private,
        Variable::TrainKind::None);
    loadMatrixFromFile("fr2en/" + langPrefix.str() + "_embedding.bin",
                       result->getPayload());
    return result;
  }

  Node *createPyTorchGRUCell(Function *G, Node *input, Node *hidden,
                             Variable *w_ih, Variable *b_ih, Variable *w_hh,
                             Variable *b_hh) {
    // reference implementation:
    // https://github.com/pytorch/pytorch/blob/dd5c195646b941d3e20a72847ac48c41e272b8b2/torch/nn/_functions/rnn.py#L46
    Node *gi = G->createFullyConnected("pytorch.GRU.gi", input, w_ih, b_ih);
    Node *gh = G->createFullyConnected("pytorch.GRU.gh", hidden, w_hh, b_hh);

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
  Function *F = mod.getFunction("main");
  input_ = mod.createVariable(
      ElemKind::IndexTy, {batchSize_, MAX_LENGTH}, "encoder.inputsentence",
      Variable::VisibilityKind::Public, Variable::TrainKind::None);
  seqLength_ = mod.createVariable(
      ElemKind::IndexTy, {batchSize_}, "encoder.seqLength",
      Variable::VisibilityKind::Public, Variable::TrainKind::None);

  Variable *hiddenInit = mod.createVariable(
      ElemKind::FloatTy, {batchSize_, EMBEDDING_SIZE}, "encoder.hiddenInit",
      Variable::VisibilityKind::Private, Variable::TrainKind::None);
  hiddenInit->getPayload().zero();

  Node *hidden = hiddenInit;

  Variable *w_ih = mod.createVariable(
      ElemKind::FloatTy, {EMBEDDING_SIZE, HIDDEN_SIZE}, "encoder.w_ih",
      Variable::VisibilityKind::Private, Variable::TrainKind::None);
  Variable *b_ih = mod.createVariable(
      ElemKind::FloatTy, {HIDDEN_SIZE}, "encoder.b_ih",
      Variable::VisibilityKind::Private, Variable::TrainKind::None);
  Variable *w_hh = mod.createVariable(
      ElemKind::FloatTy, {EMBEDDING_SIZE, HIDDEN_SIZE}, "encoder.w_hh",
      Variable::VisibilityKind::Private, Variable::TrainKind::None);
  Variable *b_hh = mod.createVariable(
      ElemKind::FloatTy, {HIDDEN_SIZE}, "encoder.b_hh",
      Variable::VisibilityKind::Private, Variable::TrainKind::None);
  loadMatrixFromFile("fr2en/encoder_w_ih.bin", w_ih->getPayload());
  loadMatrixFromFile("fr2en/encoder_b_ih.bin", b_ih->getPayload());
  loadMatrixFromFile("fr2en/encoder_w_hh.bin", w_hh->getPayload());
  loadMatrixFromFile("fr2en/encoder_b_hh.bin", b_hh->getPayload());

  Node *inputEmbedded =
      F->createGather("encoder.embedding", embedding_fr_, input_);

  // TODO: encoder does exactly MAX_LENGTH steps, while input size is smaller.
  // We could use control flow here.
  std::vector<Node *> outputs;
  for (unsigned step = 0; step < MAX_LENGTH; step++) {
    Node *inputSlice = F->createSlice(
        "encoder." + std::to_string(step) + ".inputSlice", inputEmbedded,
        {0, step, 0}, {batchSize_, step + 1, EMBEDDING_SIZE});
    Node *reshape =
        F->createReshape("encoder." + std::to_string(step) + ".reshape",
                         inputSlice, {batchSize_, EMBEDDING_SIZE});
    hidden = createPyTorchGRUCell(F, reshape, hidden, w_ih, b_ih, w_hh, b_hh);
    outputs.push_back(hidden);
  }

  Node *output = F->createConcat("encoder.output", outputs, 1);
  Node *r2 = F->createReshape("encoder.output.r2", output,
                              {MAX_LENGTH * batchSize_, EMBEDDING_SIZE});

  encoderHiddenOutput_ = F->createGather("encoder.outputNth", r2, seqLength_);
}

/// Model part representing Decoder.
/// Uses \p encoderHiddenOutput as final state from Encoder.
/// Resulting translation is put into \p output Variable.
void Model::loadDecoder() {
  auto &mod = EE_.getModule();
  Function *F = mod.getFunction("main");
  Variable *input = mod.createVariable(
      ElemKind::IndexTy, {batchSize_}, "decoder.input",
      Variable::VisibilityKind::Public, Variable::TrainKind::None);
  for (size_t i = 0; i < batchSize_; i++) {
    input->getPayload().getHandle<size_t>().at({i}) = en_.word2index_["SOS"];
  }

  Variable *w_ih = mod.createVariable(
      ElemKind::FloatTy, {EMBEDDING_SIZE, HIDDEN_SIZE}, "decoder.w_ih",
      Variable::VisibilityKind::Private, Variable::TrainKind::None);
  Variable *b_ih = mod.createVariable(
      ElemKind::FloatTy, {HIDDEN_SIZE}, "decoder.b_ih",
      Variable::VisibilityKind::Private, Variable::TrainKind::None);
  Variable *w_hh = mod.createVariable(
      ElemKind::FloatTy, {EMBEDDING_SIZE, HIDDEN_SIZE}, "decoder.w_hh",
      Variable::VisibilityKind::Private, Variable::TrainKind::None);
  Variable *b_hh = mod.createVariable(
      ElemKind::FloatTy, {HIDDEN_SIZE}, "decoder.b_hh",
      Variable::VisibilityKind::Private, Variable::TrainKind::None);
  Variable *out_w = mod.createVariable(
      ElemKind::FloatTy, {EMBEDDING_SIZE, en_.index2word_.size()},
      "decoder.out_w", Variable::VisibilityKind::Public,
      Variable::TrainKind::None);
  Variable *out_b = mod.createVariable(
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
    Node *embedded =
        F->createGather("decoder.embedding." + std::to_string(step),
                        embedding_en_, lastWordIdx);

    Node *relu = F->createRELU("decoder.relu", embedded);
    hidden = createPyTorchGRUCell(F, relu, hidden, w_ih, b_ih, w_hh, b_hh);

    Node *FC = F->createFullyConnected("decoder.outFC", hidden, out_w, out_b);
    Node *topK = F->createTopK("decoder.topK", FC, 1);

    lastWordIdx = F->createReshape("decoder.reshape", {topK, 1}, {batchSize_});
    outputs.push_back(lastWordIdx);
  }

  Node *concat = F->createConcat("decoder.output.concat", outputs, 0);
  Node *reshape = F->createReshape("decoder.output.reshape", concat,
                                   {MAX_LENGTH, batchSize_});
  output_ = mod.createVariable(
      ElemKind::IndexTy, {MAX_LENGTH, batchSize_}, "decoder.output",
      Variable::VisibilityKind::Public, Variable::TrainKind::None);
  F->createSave("decoder.output", reshape, output_);

  // Handle the request to profile the graph in preperation for quantization.
  if (!dumpProfileFileOpt.empty()) {
    // Perform the high-level optimizations before instrumenting the graph. This
    // optimization phase will remove stuff like repetitive transpose operations
    // perform CSE, etc.
    ::optimize(F, glow::CompilationMode::Infer);

    // Instrument the graph to capture profiles for nodes' outputs.
    glow::profileQuantization(F);
  }

  // Load the quantization profile and transform the graph.
  if (!loadProfileFileOpt.empty()) {
    // The profiled graph was optimized before it was instrumentated. In this
    // part of the code we repeat the same transformation in order to create
    // the same graph structure.
    glow::optimize(F, CompilationMode::Infer);

    auto quantizationInfos = deserializeFromYaml(loadProfileFileOpt);

    // Quantize the graph based on the captured profile.
    glow::quantization::generateQuantizedGraph(EE_, F, quantizationInfos);
  }
}

/// Translation has 2 stages:
/// 1) Input sentence is fed into Encoder word by word.
/// 2) "Memory" of Encoder is written into memory of Decoder.
///    Now Decoder streams resulting translation word by word.
void Model::translate(const std::vector<std::string> &batch) {
  Tensor input(ElemKind::IndexTy, {batchSize_, MAX_LENGTH});
  Tensor seqLength(ElemKind::IndexTy, {batchSize_});
  input.zero();

  for (size_t j = 0; j < batch.size(); j++) {
    std::istringstream iss(batch[j]);
    std::vector<std::string> words;
    std::string word;
    while (iss >> word)
      words.push_back(word);
    words.push_back("EOS");

    GLOW_ASSERT(words.size() <= MAX_LENGTH && "sentence is too long.");

    for (size_t i = 0; i < words.size(); i++) {
      auto iter = fr_.word2index_.find(words[i]);
      GLOW_ASSERT(iter != fr_.word2index_.end() && "Unknown word.");
      input.getHandle<size_t>().at({j, i}) = iter->second;
    }
    seqLength.getHandle<size_t>().at({j}) = (words.size() - 1) + j * MAX_LENGTH;
  }

  EE_.run({input_, seqLength_}, {&input, &seqLength});

  auto OH = output_->getPayload().getHandle<size_t>();
  for (unsigned j = 0; j < batch.size(); j++) {
    for (unsigned i = 0; i < MAX_LENGTH; i++) {
      size_t wordIdx = OH.at({i, j});
      if (wordIdx == en_.word2index_["EOS"])
        break;

      if (i)
        std::cout << ' ';
      std::cout << en_.index2word_[wordIdx];
    }
    std::cout << "\n\n";
  }

  if (!dumpProfileFileOpt.empty()) {
    std::vector<NodeQuantizationInfo> QI =
        quantization::generateNodeQuantizationInfos(
            EE_.getModule().getFunction("main"));
    serializeToYaml(dumpProfileFileOpt, QI);
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
