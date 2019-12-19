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
#include "glow/IR/IR.h"
#include "glow/Support/Support.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/Timer.h"

#include <glog/logging.h>

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>

using namespace glow;
using llvm::format;

namespace {
llvm::cl::OptionCategory ptbCat("PTB Options");
llvm::cl::opt<std::string> executionBackend(
    "backend",
    llvm::cl::desc("Backend to use, e.g., Interpreter, CPU, OpenCL:"),
    llvm::cl::Optional, llvm::cl::init("Interpreter"), llvm::cl::cat(ptbCat));

llvm::cl::opt<std::string> dumpInitialGraphDAGFileOpt(
    "dumpInitialGraphDAG",
    llvm::cl::desc(
        "Specify the file to export the initial Graph in DOT format"),
    llvm::cl::value_desc("file.dot"), llvm::cl::cat(ptbCat));

llvm::cl::opt<std::string> dumpTrainingGraphDAGFileOpt(
    "dumpTrainingGraphDAG",
    llvm::cl::desc(
        "Specify the file to export the training Graph in DOT format"),
    llvm::cl::value_desc("file.dot"), llvm::cl::cat(ptbCat));

} // namespace

unsigned loadPTB(Tensor &inputWords, Tensor &targetWords, dim_t numSteps,
                 dim_t vocabSize, dim_t minibatchSize, dim_t maxNumWords) {

  std::ifstream ptbInput("ptb/simple-examples/data/ptb.train.txt");
  CHECK(ptbInput.is_open()) << "Error loading ptb.train.txt";

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

  CHECK_GT(numWords, 0) << "No words were found.";

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
  dim_t batchLength = numWords / minibatchSize;
  dim_t numBatches = (batchLength - 1) / numSteps;
  dim_t numSequences = minibatchSize * numBatches;

  // While we dont have embedding, we are using one-hot encoding to represent
  // input words. To limit the size of the data we use an upper bound on the
  // vocabulary size.
  inputWords.reset(ElemKind::FloatTy, {numSequences, vocabSize * numSteps});
  targetWords.reset(ElemKind::Int64ITy, {numSequences, numSteps});
  auto IIH = inputWords.getHandle<>();
  auto TIH = targetWords.getHandle<int64_t>();
  for (unsigned batch = 0; batch < minibatchSize; batch++) {
    for (unsigned iter = 0; iter < numBatches; iter++) {
      dim_t sequence = batch + iter * minibatchSize;
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
  LOG(INFO) << "Loading the ptb database.";

  Tensor inputWords;
  Tensor targetWords;

  const dim_t minibatchSize = 10;
  const dim_t numSteps = 10;
  const dim_t numEpochs = 20;

  const dim_t hiddenSize = 20;
  const dim_t vocabSize = 500;
  const dim_t maxNumWords = 10000;

  float learningRate = .1;

  unsigned numWords = loadPTB(inputWords, targetWords, numSteps, vocabSize,
                              minibatchSize, maxNumWords);
  LOG(INFO) << "Loaded " << numWords << " words.";
  ExecutionEngine EE(executionBackend);
  PlaceholderBindings bindings;

  // Construct the network:
  TrainingConfig TC;
  TC.learningRate = learningRate;
  TC.momentum = 0;
  TC.batchSize = minibatchSize;

  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  LOG(INFO) << "Building";

  auto *X = mod.createPlaceholder(
      ElemKind::FloatTy, {minibatchSize, vocabSize * numSteps}, "input", false);
  bindings.allocate(X);
  auto *Y = mod.createPlaceholder(ElemKind::Int64ITy, {minibatchSize, numSteps},
                                  "selected", false);
  bindings.allocate(Y);

  std::vector<NodeValue> slicesX;

  for (unsigned t = 0; t < numSteps; t++) {
    auto XtName = "X." + std::to_string(t);
    auto *Xt = F->createSlice(XtName, X, {0, t * vocabSize},
                              {minibatchSize, (t + 1) * vocabSize});
    slicesX.push_back(Xt);
  }

  std::vector<NodeValue> outputNodes;
  F->createSimpleRNN(bindings, "rnn", slicesX, minibatchSize, hiddenSize,
                     vocabSize, outputNodes);

  // O has a shape of {numSteps * minibatchSize, vocabSize}
  Node *O = F->createConcat("output", outputNodes, 0);
  // T has shape of {numSteps * minibatchSize, 1}
  Node *TN = F->createTranspose("Y.transpose", Y, {1, 0});
  Node *T = F->createReshape("Y.reshape", TN, {numSteps * minibatchSize, 1});

  auto *SM = F->createSoftMax("softmax", O, T);
  auto *save = F->createSave("result", SM);
  auto *result = bindings.allocate(save->getPlaceholder());

  if (!dumpInitialGraphDAGFileOpt.empty()) {
    LOG(INFO) << "Dumping initial graph";
    F->dumpDAG(dumpInitialGraphDAGFileOpt.c_str());
  }

  Function *TF = glow::differentiate(F, TC);
  auto tfName = TF->getName();

  EE.compile(CompilationMode::Train);
  bindings.allocate(mod.getPlaceholders());

  if (!dumpTrainingGraphDAGFileOpt.empty()) {
    LOG(INFO) << "Dumping training graph";
    TF->dumpDAG(dumpTrainingGraphDAGFileOpt.c_str());
  }

  size_t numBatches = (numWords / minibatchSize - 1) / numSteps;

  LOG(INFO) << "Training for " << numBatches << " rounds";

  float metricValues[numEpochs];

  for (size_t iter = 0; iter < numEpochs; iter++) {
    llvm::outs() << "Training - iteration #" << (iter + 1) << "\n";

    llvm::Timer timer("Training", "Training");
    timer.startTimer();

    // Compute the perplexity over a few minibatches
    float perplexity = 0;
    size_t perplexityWordsCount = 0;

    // This variable records the number of the next sample to be used for
    // training.
    size_t sampleCounter = 0;

    for (unsigned batch = 0; batch < numBatches; batch++) {
      Tensor inputWordsBatch(ElemKind::FloatTy,
                             {minibatchSize, vocabSize * numSteps});
      inputWordsBatch.copyConsecutiveSlices(&inputWords, minibatchSize * batch);

      Tensor targetWordsBatch(ElemKind::Int64ITy, {minibatchSize, numSteps});
      targetWordsBatch.copyConsecutiveSlices(&targetWords,
                                             minibatchSize * batch);

      runBatch(EE, bindings, 1, sampleCounter, {X, Y},
               {&inputWordsBatch, &targetWordsBatch}, tfName);
      for (dim_t step = 0; step < numSteps; step++) {
        for (unsigned int i = 0; i < minibatchSize; i++) {
          auto T =
              result->getHandle<float>().extractSlice(step * minibatchSize + i);
          dim_t correct = targetWords.getHandle<dim_t>().at(
              {minibatchSize * batch + i, step});
          float soft_guess = -std::log(T.getHandle<float>().at({correct}));
          perplexity += soft_guess;
          perplexityWordsCount += 1;
        }
      }
      if (batch % 10 == 1) {
        llvm::outs() << "perplexity: "
                     << format("%0.4f",
                               std::exp(perplexity / perplexityWordsCount))
                     << "\n";
      }
    }
    metricValues[iter] = std::exp(perplexity / perplexityWordsCount);
    llvm::outs() << "perplexity: " << format("%0.4f", metricValues[iter])
                 << "\n\n";

    timer.stopTimer();
  }

  llvm::outs() << "Perplexity scores in copy-pastable format:\n";
  for (size_t iter = 0; iter < numEpochs; iter++) {
    if (iter != 0 && iter % 2 == 0)
      continue;
    llvm::outs() << "/// Iteration " << iter + 1 << ": "
                 << format("%0.4f", metricValues[iter]) << "\n";
  }
  llvm::outs()
      << "Note, that small 1E-4 error is considered acceptable and may "
      << "be coming from fast math optimizations.\n";
}

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv, " The PTB test\n\n");
  testPTB();

  return 0;
}
