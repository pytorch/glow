#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Graph/Graph.h"
#include "glow/Support/Support.h"

#include "llvm/Support/Timer.h"

#include <cassert>
#include <fstream>
#include <iostream>
#include <map>
#include <algorithm>

using namespace glow;

unsigned loadPTB(Tensor &inputWords, Tensor &targetWords,
		 size_t numSteps, size_t vocabSize) {

  std::ifstream ptbInput("ptb/simple-examples/data/ptb.train.txt");

  std::vector<std::string> words((std::istream_iterator<std::string>(ptbInput)),
				 (std::istream_iterator<std::string>()));

  size_t numWords = words.size();

  GLOW_ASSERT(numWords && "No words were found.");

  std::map<std::string, int> counter;
  // Counter of words occurences in the input text
  for (auto word: words) {
    if (counter.find(word) == counter.end()) {
      counter[word] = 0;
    }
    counter[word] += 1;
  }

  // Sort the counter
  std::vector<std::pair<std::string, int> > counter_v;
  copy(counter.begin(),
       counter.end(),
       std::back_inserter<std::vector<std::pair<std::string, int> > >(counter_v));

  sort(counter_v.begin(), counter_v.end(), [](const std::pair<std::string, int>& lhs,
					      const std::pair<std::string, int>& rhs) {
	 return lhs.second > rhs.second;
       });

  // Build the word to id map
  std::map<std::string, int> word_to_id;
  for (unsigned i = 0; i < counter_v.size(); i++) {
    auto const &word = counter_v[i].first;
    // auto const &word_count = counter_v[i].second;
    // std::cout << "Word: " << word << " : " << word_count << std::endl;
    word_to_id[word] = std::min<size_t>(i, vocabSize - 1);
  }

  // Load the PTB database into two 3d tensors for word inputs and targets.
  size_t numSequences = numWords / numSteps - 1;

  // While we dont have embedding, we are using one-hot encoding to represent
  // input words. To limit the size of the data we use an upper bound on the
  // vocabulary size.
  inputWords.reset(ElemKind::FloatTy, {numSequences, vocabSize * numSteps});
  targetWords.reset(ElemKind::IndexTy, {numSequences, numSteps});
  auto IIH = inputWords.getHandle<>();
  auto TIH = targetWords.getHandle<size_t>();
  int word_counter_id = 0;
  for (unsigned sequence = 0; sequence < numSequences; sequence++) {
    for (unsigned step = 0; step < numSteps; step++) {
      const std::string word1 = words[word_counter_id];
      const std::string word2 = words[word_counter_id + 1];
      IIH.at({sequence, step * vocabSize + word_to_id[word1]}) = 1;
      TIH.at({sequence, step}) = word_to_id[word2];
      word_counter_id += 1;
    }
  }

  return numWords;
}

void debug(Node* node) {
  std::cout << (std::string)node->getName();
  for (int i = 0; i < node->dims().size(); i++) {
    std::cout << " " << node->dims()[i];
  }
  std::cout << std::endl;
}

/// This test builds a RNN language model on the Penn TreeBank dataset.
/// Results for RNN word-level perplexity are reported in https://arxiv.org/pdf/1409.2329.pdf
/// Perplexity reaches 64 on the 20th iteration.
void testPTB() {
  std::cout << "Loading the ptb database.\n";

  Tensor inputWords;
  Tensor targetWords;

  size_t minibatchSize = 32;
  size_t numSteps = 20;
  size_t vocabSize = 10000;
  size_t hiddenSize = 256;

  unsigned numWords = loadPTB(inputWords, targetWords, numSteps, vocabSize);
  std::cout << "Loaded " << numWords << " words.\n";

  ExecutionEngine EE;

  // Construct the network:
  EE.getConfig().learningRate = 0.001;
  EE.getConfig().momentum = 0.9;
  EE.getConfig().L2Decay = 0.001;

  auto &G = EE.getGraph();

  Variable *X = G.createVariable(ElemKind::FloatTy, {minibatchSize, vocabSize * numSteps},
				 "input", Variable::InitKind::Extern);
  Variable *Y = G.createVariable(ElemKind::IndexTy, {minibatchSize, numSteps},
				 "selected", Variable::InitKind::Extern);

  auto *HtInit = G.createVariable(ElemKind::FloatTy, {minibatchSize, numSteps},
				  "initial_state",
				  Variable::InitKind::Broadcast, 0);

  std::vector<SaveNode *> results;
  TanhNode *HtLast;

  for (unsigned t = 0; t < numSteps; t++) {
    auto *Xt = G.createSlice("xt", X, {0, t * vocabSize}, {minibatchSize, vocabSize});
    auto *Yt = G.createSlice("yt", Y, {0, t}, {minibatchSize, 1});
    FullyConnectedNode *FC1t;
    if (t == 0) {
      FC1t = G.createFullyConnected("fc1t", HtInit, hiddenSize);
    } else {
      FC1t = G.createFullyConnected("fc1t", HtLast, hiddenSize);
    }
    auto *FC2t = G.createFullyConnected("fc2t", Xt, hiddenSize);
    auto *At = G.createArithmetic("fct", FC1t, FC2t, ArithmeticNode::Mode::Add);
    auto *Ht = G.createTanh("tanht", At);
    HtLast = Ht;
    auto *Ot = G.createFullyConnected("ot", Ht, vocabSize);
    auto *SMt = G.createSoftMax("smt", Ot, Yt);
    results.push_back(G.createSave("return", SMt));
  }

  EE.compile(CompilationMode::Train);

  // Report progress every this number of training iterations.
  constexpr int reportRate = 10;

  std::cout << "Training" << std::endl;

  for (int iter = 0; iter < 20; iter++) {
    std::cout << "Training - iteration #" << iter << std::endl;

    llvm::Timer timer("Training", "Training");
    timer.startTimer();

    // On each training iteration take an input from imageInputs and update
    // the input variable A, and add take a corresponding label and update the
    // softmax layer.
    EE.train(reportRate, {X, Y}, {&inputWords, &targetWords});

    // Compute the perplexity over a few minibatches
    float perplexity = 0;
    size_t perplexityWordsCount = 0;

    std::cout << "Computing training perplexity" << std::endl;
    size_t numSequences = numWords / numSteps - 1;
    // We should not run infer on training data, these values should be obtained from
    // the training step itself. The code below is very slow.
    size_t perplexitySteps = 20;
    for (unsigned i = 0; i < perplexitySteps; i++) {
      unsigned sequenceIndex = i * ((numSequences - 1) / minibatchSize / perplexitySteps);
      Tensor sample(ElemKind::FloatTy, {minibatchSize, vocabSize * numSteps});
      sample.copyConsecutiveSlices(&inputWords, minibatchSize * sequenceIndex);
      EE.infer({X}, {&sample});
      for (size_t step = 0; step < numSteps; step++) {
	auto result = results[step];
	Tensor &res = result->getOutput()->getPayload();
	for (unsigned int batch = 0; batch < minibatchSize; batch++) {
	  auto T = res.getHandle<>().extractSlice(batch);
	  size_t correct = targetWords.getHandle<>().at({minibatchSize * sequenceIndex + batch, step});
	  double soft_guess = std::log(T.getHandle<>().at({correct}));
	  perplexity += soft_guess;
	  perplexityWordsCount += 1;
	  //size_t guess = T.getHandle<>().maxArg();
	}
      }
    }
    perplexity /= perplexityWordsCount;
    perplexity = std::exp(- perplexity);
    std::cout << perplexity << std::endl;
    timer.stopTimer();
  }
}

int main() {
  testPTB();

  return 0;
}
