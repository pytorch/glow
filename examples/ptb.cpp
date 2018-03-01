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
  std::vector<std::pair<std::string, int>> counters(counter.begin(),
                                                    counter.end());

  sort(counters.begin(), counters.end(),
       [](const std::pair<std::string, int> &lhs,
          const std::pair<std::string, int> &rhs) {
         if (lhs.second == rhs.second) {
           return rhs.first > lhs.first;
         }
         return lhs.second > rhs.second;
       });

  // Build the word to id map
  std::map<std::string, int> wordToId;
  for (unsigned i = 0; i < counters.size(); i++) {
    auto const &word = counters[i].first;
    wordToId[word] = std::min<size_t>(i, vocabSize - 1);
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
        int wordCounterId = step + iter * numSteps + batch * batchLength;
        const std::string word1 = words[wordCounterId];
        const std::string word2 = words[wordCounterId + 1];
        IIH.at({sequence, step * vocabSize + wordToId[word1]}) = 1;
        TIH.at({sequence, step}) = wordToId[word2];
      }
    }
  }
  return numWords;
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
/// Iteration 1: 105.4579
/// Iteration 2: 82.3274
/// Iteration 4: 70.8094
/// Iteration 6: 63.8546
/// Iteration 8: 58.4330
/// Iteration 10: 53.7943
/// Iteration 12: 49.7214
/// Iteration 14: 46.1715
/// Iteration 16: 43.1474
/// Iteration 18: 40.5605
/// Iteration 20: 38.2837
///
/// For reference, we expect the usage of an LSTM instead of the current
/// simple RNN block will improve the perplexity to ~20.
void testPTB() {
  std::cout << "Loading the ptb database.\n";

  Tensor inputWords;
  Tensor targetWords;

  const size_t minibatchSize = 10;
  const size_t numSteps = 10;
  const size_t numEpochs = 20;

  const size_t hiddenSize = 20;
  const size_t vocabSize = 500;
  const size_t maxNumWords = 10000;

  float learningRate = .1;

  unsigned numWords = loadPTB(inputWords, targetWords, numSteps, vocabSize,
                              minibatchSize, maxNumWords);
  std::cout << "Loaded " << numWords << " words.\n";
  ExecutionEngine EE;

  // Construct the network:
  EE.getConfig().learningRate = learningRate;
  EE.getConfig().momentum = 0;
  EE.getConfig().batchSize = minibatchSize;

  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  std::cout << "Building" << std::endl;

  Variable *X = mod.createVariable(
      ElemKind::FloatTy, {minibatchSize, vocabSize * numSteps}, "input",
      Variable::VisibilityKind::Public, Variable::TrainKind::None);
  Variable *Y = mod.createVariable(ElemKind::IndexTy, {minibatchSize, numSteps},
                                   "selected", Variable::VisibilityKind::Public,
                                   Variable::TrainKind::None);

  std::vector<Node *> slicesX, slicesY;

  for (unsigned t = 0; t < numSteps; t++) {
    auto XtName = "X." + std::to_string(t);
    auto *Xt = F->createSlice(XtName, X, {0, t * vocabSize},
                              {minibatchSize, (t + 1) * vocabSize});
    slicesX.push_back(Xt);
    auto YtName = "Y." + std::to_string(t);
    auto *Yt = F->createSlice(YtName, Y, {0, t}, {minibatchSize, t + 1});
    slicesY.push_back(Yt);
  }

  std::vector<Node *> outputNodes;
  F->createSimpleRNN("rnn", slicesX, minibatchSize, hiddenSize, vocabSize,
                     outputNodes);

  // O has a shape of {numSteps * minibatchSize, vocabSize}
  Node *O = F->createConcat("output", outputNodes, 0);
  // T has shape of {numSteps * minibatchSize, 1}
  Node *T = F->createConcat("target", slicesY, 0);

  auto *SM = F->createSoftMaxWithLoss("softmax", O, T);
  auto *result = F->createSave("result", SM->getResult());
  F->createSave("celoss", SM->getCELoss());

  std::cout << "Dumping graph" << std::endl;

  Function *TF = glow::differentiate(F, EE.getConfig());

  EE.compile(CompilationMode::Train, TF);

  F->dumpDAG("DAG.dot");

  size_t numBatches = (numWords / minibatchSize - 1) / numSteps;

  std::cout << "Training for " << numBatches << " rounds" << std::endl;

  float metricValues[numEpochs];

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
    metricValues[iter] = std::exp(perplexity / perplexityWordsCount);
    std::cout << "perplexity: " << metricValues[iter] << std::endl << std::endl;

    timer.stopTimer();
  }

  std::cout << "Perplexity scores in copy-pastable format:\n";
  for (size_t iter = 0; iter < numEpochs; iter++) {
    if (iter != 0 && iter % 2 == 0)
      continue;
    std::cout << "/// Iteration " << iter + 1 << ": " << metricValues[iter]
              << std::endl;
  }
  std::cout << "Note, that small 1E-4 error is considered acceptable and may "
            << "be coming from fast math optimizations.\n";
}

int main() {
  std::cout.setf(std::ios::fixed);
  std::cout << std::setprecision(4);
  testPTB();

  return 0;
}
