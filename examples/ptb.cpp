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

unsigned loadPTB(Tensor &inputWords,
		 Tensor &targetWords,
		 size_t numSteps,
		 size_t vocabSize,
		 size_t minibatchSize,
		 size_t maxNumWords) {

  std::ifstream ptbInput("ptb/simple-examples/data/ptb.train.txt");

  std::vector<std::string> words;
  std::string line;

  if (ptbInput.is_open()) {
    while (getline(ptbInput, line)) {
      std::istringstream ss(line);
      std::string token;
      while (getline(ss, token, ' ')){
	if (!token.empty()) {
	  words.push_back(token);
	}
      }
      words.push_back("<eos>");
    }
    ptbInput.close();
  }

  // We limit the number of words to 50,000 otherwise things will be slower.
  words = std::vector<std::string>(words.begin(), words.begin() + maxNumWords);
  size_t numWords = words.size();
  std::cout << "working on it" << std::endl;

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

void debug(Node* node) {
  std::cout << (std::string)node->getName();
  for (int i = 0; i < node->dims().size(); i++) {
    std::cout << " " << node->dims()[i];
  }
  std::cout << std::endl;
}

/// This test builds a RNN language model on the Penn TreeBank dataset.
/// Results for RNN word-level perplexity are reported in https://arxiv.org/pdf/1409.2329.pdf
/// Here we simplify the problem ti be able to run it on a single CPU.
/// The expected results for the following settings is a perplexity of 42
/// after the 10th iteration.
/// The current code achieves a perplexity of 98.
void testPTB() {
  std::cout << "Loading the ptb database.\n";

  Tensor inputWords;
  Tensor targetWords;

  size_t numSteps = 5;
  size_t hiddenSize = 200;
  size_t vocabSize = 700;
  size_t minibatchSize = 20;
  float learningRate = 0.001;
  size_t maxNumWords = 50000;
  size_t numEpochs = 10;

  unsigned numWords = loadPTB(inputWords,
			      targetWords,
			      numSteps,
			      vocabSize,
			      minibatchSize,
			      maxNumWords);
  std::cout << "Loaded " << numWords << " words.\n";
  ExecutionEngine EE;

  // Construct the network:
  EE.getConfig().learningRate = learningRate;
  EE.getConfig().momentum = 0;

  auto &G = EE.getGraph();

  Variable *X = G.createVariable(ElemKind::FloatTy, {minibatchSize, vocabSize * numSteps},
				 "input", Variable::InitKind::Extern);
  Variable *Y = G.createVariable(ElemKind::IndexTy, {minibatchSize, numSteps},
				 "selected", Variable::InitKind::Extern);

  auto *HtInit = G.createVariable(ElemKind::FloatTy, {minibatchSize, numSteps},
				  "initial_state",
				  Variable::InitKind::Broadcast, 0);

  std::vector<Node *> outputNodes;
  TanhNode *HtLast;

  for (unsigned t = 0; t < numSteps; t++) {
    const std::string XtName = std::to_string("x") + std::to_string(t);
    const std::string FC1tName = std::to_string("fc1") + std::to_string(t);
    const std::string FC2tName = std::to_string("fc2") + std::to_string(t);
    const std::string FCtName = std::to_string("fc") + std::to_string(t);
    const std::string TanhtName = std::to_string("tanh") + std::to_string(t);
    const std::string OtName = std::to_string("o") + std::to_string(t);

    auto *Xt = G.createSlice(XtName, X, {0, t * vocabSize}, {minibatchSize, vocabSize});
    FullyConnectedNode *FC1t;
    if (t == 0) {
      FC1t = G.createFullyConnected(FC1tName, HtInit, hiddenSize);
    } else {
      FC1t = G.createFullyConnected(FC1tName, HtLast, hiddenSize);
    }
    auto *FC2t = G.createFullyConnected(FC2tName, Xt, hiddenSize);
    auto *At = G.createArithmetic(FCtName, FC1t, FC2t, ArithmeticNode::Mode::Add);
    auto *Ht = G.createTanh(TanhtName, At);
    HtLast = Ht;
    auto *Ot = G.createFullyConnected(OtName, Ht, vocabSize);
    // Ot has shape {minibatchSize, vocabSize}
    outputNodes.push_back(Ot);
  }
  Node *O = outputNodes[0];
  for (unsigned t = 1; t < numSteps; t++) {
    const std::string ConcatName = std::to_string("concat") + std::to_string(t);
    O = G.createConcat(ConcatName, O, outputNodes[t], 0);
  }
  // O has final shape of {minibatchSize * numSteps, vocabSize}
  // Y has shape of {minibatchSize, numSteps}
  auto *Yr = G.createReshape("yreshape", Y, {minibatchSize * numSteps});
  auto *SM = G.createSoftMax("softmax", O, Yr);
  auto *result = G.createSave("result", SM);

  std::cout << "Dumping graph" << std::endl;
  G.dumpDAG();

  EE.compile(CompilationMode::Train);

  // Run for those many iterations inside an epoch. Note that this is poorly named
  // as it stands since there is no reporting that happens inside the EE.train(..) call
  // Also note that this value will affect the range of inputs which is actually covered
  // by the algorithm unless it is selected in such a way that it will wrap around. Here
  // we carefully select it in such a way that it will do a single pass on the entire
  // dataset.
  size_t numBatches = (numWords / minibatchSize - 1) / numSteps;

  std::cout << "Training for " << numBatches << " rounds" << std::endl;

  for (int iter = 0; iter < numEpochs; iter++) {
    std::cout << "Training - iteration #" << (iter + 1) << std::endl;

    llvm::Timer timer("Training", "Training");
    timer.startTimer();

    // On each training iteration take an input from imageInputs and update
    // the input variable A, and add take a corresponding label and update the
    // softmax layer.
    EE.train(numBatches, {X, Y}, {&inputWords, &targetWords});

    // Compute the perplexity over a few minibatches
    float perplexity = 0;
    size_t perplexityWordsCount = 0;

    std::cout << "Computing training perplexity" << std::endl;
    // We should not run infer on training data, these values should be obtained from
    // the training step itself.
    for (unsigned batch = 0; batch < numBatches; batch++) {
      Tensor sample(ElemKind::FloatTy, {minibatchSize, vocabSize * numSteps});
      sample.copyConsecutiveSlices(&inputWords, minibatchSize * batch);
      EE.infer({X}, {&sample});
      Tensor &res = result->getOutput()->getPayload();
      for (unsigned int i = 0; i < minibatchSize; i++) {
	for (size_t step = 0; step < numSteps; step++) {
	  auto T = res.getHandle<>().extractSlice(i * numSteps + step);
	  size_t correct = targetWords.getHandle<std::size_t>().at({
	      minibatchSize * batch + i, step});
	  double soft_guess = - std::log(T.getHandle<>().at({correct}));
	  unsigned guess = T.getHandle<>().maxArg();
	  perplexity += soft_guess;
	  perplexityWordsCount += 1;
	}
      }
    }
    perplexity /= perplexityWordsCount;
    perplexity = std::exp(perplexity);
    std::cout << perplexity << std::endl;
    timer.stopTimer();
  }
}

int main() {
  testPTB();

  return 0;
}
