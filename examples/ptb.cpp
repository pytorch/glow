#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Graph/Graph.h"
#include "glow/Support/Support.h"

#include "llvm/Support/Timer.h"

#include <algorithm>
#include <cassert>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>

using namespace glow;

unsigned loadPTB(Tensor &inputWords, Tensor &targetWords, size_t numSteps,
                 size_t vocabSize, size_t minibatchSize, size_t maxNumWords) {

  std::ifstream ptbInput("ptb/simple-examples/data/ptb.train.txt");
  if (!ptbInput.is_open()) {
    llvm::errs() << "Error loading ptb.train.txt\n";
    std::exit(EXIT_FAILURE);
  }

  std::vector<std::string> words;
  std::string line;

  while (getline(ptbInput, line)) {
    std::istringstream ss(line);
    std::string token;
    while (getline(ss, token, ' ')) {
      if (!token.empty()) {
        words.push_back(token);
      }
    }
    words.push_back("<eos>");
  }
  ptbInput.close();

  // We limit the number of words to 50,000 otherwise things will be slower.
  words = std::vector<std::string>(words.begin(), words.begin() + maxNumWords);
  size_t numWords = words.size();

  GLOW_ASSERT(numWords && "No words were found.");

  std::map<std::string, int> counter;
  // Counter of words occurences in the input text
  for (auto word : words) {
    if (counter.find(word) == counter.end()) {
      counter[word] = 0;
    }
    counter[word] += 1;
  }

  // Sort the counter
  std::vector<std::pair<std::string, int>> counter_v(counter.begin(),
                                                     counter.end());

  sort(counter_v.begin(), counter_v.end(),
       [](const std::pair<std::string, int> &lhs,
          const std::pair<std::string, int> &rhs) {
         if (lhs.second == rhs.second) {
           return rhs.first > lhs.first;
         }
         return lhs.second > rhs.second;
       });

  // Build the word to id map
  std::map<std::string, int> word_to_id;
  for (unsigned i = 0; i < counter_v.size(); i++) {
    auto const &word = counter_v[i].first;
    word_to_id[word] = std::min<size_t>(i, vocabSize - 1);
  }

  // Load the PTB database into two 3d tensors for word inputs and targets.
  size_t batchLength = numWords / minibatchSize;
  size_t numBatches = (batchLength - 1) / numSteps;
  size_t numSequences = minibatchSize * numBatches;

  // While we dont have embedding, we are using one-hot encoding to represent
  // input words. To limit the size of the data we use an upper bound on the
  // vocabulary size.
  inputWords.reset(ElemKind::FloatTy, {numSequences, vocabSize * numSteps});
  targetWords.reset(ElemKind::IndexTy, {numSequences, numSteps});
  auto IIH = inputWords.getHandle<>();
  auto TIH = targetWords.getHandle<size_t>();
  for (unsigned batch = 0; batch < minibatchSize; batch++) {
    for (unsigned iter = 0; iter < numBatches; iter++) {
      size_t sequence = batch + iter * minibatchSize;
      for (unsigned step = 0; step < numSteps; step++) {
        int word_counter_id = step + iter * numSteps + batch * batchLength;
        const std::string word1 = words[word_counter_id];
        const std::string word2 = words[word_counter_id + 1];
        IIH.at({sequence, step * vocabSize + word_to_id[word1]}) = 1;
        TIH.at({sequence, step}) = word_to_id[word2];
      }
    }
  }
  return numWords;
}

void debug(Node *node) {
  std::cout << (std::string)node->getName();
  for (size_t i = 0; i < node->dims().size(); i++) {
    std::cout << " " << node->dims()[i];
  }
  std::cout << std::endl;
}

/// This test builds a RNN language model on the Penn TreeBank dataset.
/// Results for RNN word-level perplexity are reported in
/// https://arxiv.org/pdf/1409.2329.pdf Here we simplify the problem to be able
/// to run it on a single CPU.
/// The results were cross-checked with an equivalent tensorflow implementation
/// as well as a Vanilla implementation inspired from Karpathy's Char-RNN code.
/// Tensorflow https://gist.github.com/mcaounfb/7ba05b0a62383c36e24a33defa3f11aa
/// Vanilla https://gist.github.com/mcaounfb/c4ee98bbddaa6f8505f283ac018f8c34
///
/// The results for the perplexity are expected to look as:
///
/// Iteration 1: 105.041900635
/// Iteration 2: 82.2057647705
/// Iteration 4: 70.7041549683
/// Iteration 6: 63.9953918457
/// Iteration 8: 58.681804657
/// Iteration 10: 54.0125465393
/// Iteration 12: 49.9519844055
/// Iteration 14: 46.4922065735
/// Iteration 16: 43.5101737976
/// Iteration 18: 40.9120101929
/// Iteration 20: 38.6976051331
///
/// For reference, we expect the usage of an LSTM instead of the current
/// simple RNN block will improve the perplexity to ~20.
void testPTB() {
  std::cout << "Loading the ptb database.\n";

  Tensor inputWords;
  Tensor targetWords;

  size_t minibatchSize = 10;
  size_t numSteps = 10;
  size_t numEpochs = 20;

  size_t hiddenSize = 20;
  size_t vocabSize = 500;
  size_t maxNumWords = 10000;

  float learningRate = .1;

  unsigned numWords = loadPTB(inputWords, targetWords, numSteps, vocabSize,
                              minibatchSize, maxNumWords);
  std::cout << "Loaded " << numWords << " words.\n";
  ExecutionEngine EE;

  // Construct the network:
  EE.getConfig().learningRate = learningRate;
  EE.getConfig().momentum = 0;
  EE.getConfig().batchSize = minibatchSize;

  auto &G = EE.getGraph();
  std::cout << "Building" << std::endl;

  Variable *X =
      G.createVariable(ElemKind::FloatTy, {minibatchSize, vocabSize * numSteps},
                       "input", Variable::InitKind::Extern);
  Variable *Y = G.createVariable(ElemKind::IndexTy, {minibatchSize, numSteps},
                                 "selected", Variable::InitKind::Extern);

  // Initialize internal memory variable H:
  Variable *HtInit =
      G.createVariable(ElemKind::FloatTy, {minibatchSize, hiddenSize},
                       "initial_state", Variable::InitKind::Extern);
  HtInit->getPayload().zero();

  std::vector<Node *> outputNodes;
  std::vector<Node *> targetNodes;
  Node *HtLast = HtInit;

  float b = 0.1;
  auto *Whh = G.createVariable(ElemKind::FloatTy, {hiddenSize, hiddenSize},
                               "Whh", Variable::InitKind::Xavier, hiddenSize);
  auto *Bh1 = G.createVariable(ElemKind::FloatTy, {hiddenSize}, "bh1",
                               Variable::InitKind::Broadcast, b);

  auto *Wxh = G.createVariable(ElemKind::FloatTy, {vocabSize, hiddenSize},
                               "Wxh", Variable::InitKind::Xavier, vocabSize);
  auto *Bh2 = G.createVariable(ElemKind::FloatTy, {hiddenSize}, "bh2",
                               Variable::InitKind::Broadcast, b);

  auto *Why = G.createVariable(ElemKind::FloatTy, {hiddenSize, vocabSize},
                               "Why", Variable::InitKind::Xavier, hiddenSize);
  auto *By = G.createVariable(ElemKind::FloatTy, {vocabSize}, "by",
                              Variable::InitKind::Broadcast, b);

  for (unsigned t = 0; t < numSteps; t++) {
    const std::string XtName = "x" + std::to_string(t);
    const std::string YtName = "y" + std::to_string(t);
    const std::string FC1tName = "fc1" + std::to_string(t);
    const std::string FC2tName = "fc2" + std::to_string(t);
    const std::string FCtName = "fc" + std::to_string(t);
    const std::string TanhtName = "tanh" + std::to_string(t);
    const std::string OtName = "o" + std::to_string(t);

    auto *Xt = G.createSlice(XtName, X, {0, t * vocabSize},
                             {minibatchSize, (t + 1) * vocabSize});
    auto *Yt = G.createSlice(YtName, Y, {0, t}, {minibatchSize, t + 1});

    FullyConnectedNode *FC1t =
        G.createFullyConnected(FC1tName, HtLast, Whh, Bh1, hiddenSize);
    auto *FC2t = G.createFullyConnected(FC2tName, Xt, Wxh, Bh2, hiddenSize);
    auto *At =
        G.createArithmetic(FCtName, FC1t, FC2t, ArithmeticNode::Mode::Add);
    auto *Ht = G.createTanh(TanhtName, At);
    HtLast = Ht;
    auto *Ot = G.createFullyConnected(OtName, Ht, Why, By, vocabSize);
    // Ot has shape {minibatchSize, vocabSize}
    outputNodes.push_back(Ot);
    // Yt has shape {minibatchSize, 1}
    targetNodes.push_back(Yt);
  }
  // O has a shape of {numSteps * minibatchSize, vocabSize}
  Node *O = G.createConcat("output", outputNodes, 0);
  // T has shape of {numSteps * minibatchSize, 1}
  Node *T = G.createConcat("target", targetNodes, 0);

  auto *SM = G.createSoftMax("softmax", O, T);
  auto *result = G.createSave("result", SM);

  std::cout << "Dumping graph" << std::endl;

  EE.compile(CompilationMode::Train);

  G.dumpDAG("DAG.dot");

  size_t numBatches = (numWords / minibatchSize - 1) / numSteps;

  std::cout << "Training for " << numBatches << " rounds" << std::endl;

  for (size_t iter = 0; iter < numEpochs; iter++) {
    std::cout << "Training - iteration #" << (iter + 1) << std::endl;

    llvm::Timer timer("Training", "Training");
    timer.startTimer();

    // Compute the perplexity over a few minibatches
    float perplexity = 0;
    size_t perplexityWordsCount = 0;

    for (unsigned batch = 0; batch < numBatches; batch++) {
      Tensor inputWordsBatch(ElemKind::FloatTy,
                             {minibatchSize, vocabSize * numSteps});
      inputWordsBatch.copyConsecutiveSlices(&inputWords, minibatchSize * batch);

      Tensor targetWordsBatch(ElemKind::IndexTy, {minibatchSize, numSteps});
      targetWordsBatch.copyConsecutiveSlices(&targetWords,
                                             minibatchSize * batch);

      EE.runBatch(1, {X, Y}, {&inputWordsBatch, &targetWordsBatch});
      Tensor &res = result->getVariable()->getPayload();
      for (size_t step = 0; step < numSteps; step++) {
        for (unsigned int i = 0; i < minibatchSize; i++) {
          auto T =
              res.getHandle<float>().extractSlice(step * minibatchSize + i);
          size_t correct = targetWords.getHandle<std::size_t>().at(
              {minibatchSize * batch + i, step});
          float soft_guess = -std::log(T.getHandle<float>().at({correct}));
          perplexity += soft_guess;
          perplexityWordsCount += 1;
        }
      }
      if (batch % 10 == 1) {
        std::cout << "perplexity: "
                  << std::exp(perplexity / perplexityWordsCount) << std::endl;
      }
    }
    std::cout << "perplexity: " << std::exp(perplexity / perplexityWordsCount)
              << std::endl;
    timer.stopTimer();
  }
}

int main() {
  std::cout << std::setprecision(12);
  testPTB();

  return 0;
}
