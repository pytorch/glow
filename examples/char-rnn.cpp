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
#include "glow/Optimizer/GraphOptimizer/GraphOptimizer.h"
#include "glow/Support/Support.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Timer.h"

#include <glog/logging.h>

#include <string>

//----------------------------------------------------------------------------//
// This is a small program that's based on Andrej's char-rnn generator. This is
// a small RNN-based neural network that's used to generate random text after
// analyzing some other text. The network is described here:
// http://karpathy.github.io/2015/05/21/rnn-effectiveness/
//----------------------------------------------------------------------------//

using namespace glow;
using llvm::format;

namespace {
llvm::cl::OptionCategory category("char-rnn Options");
static llvm::cl::opt<std::string> inputFilename(llvm::cl::desc("input file"),
                                                llvm::cl::init("-"),
                                                llvm::cl::Positional,
                                                llvm::cl::cat(category));

llvm::cl::opt<std::string> executionBackend(
    "backend",
    llvm::cl::desc("Backend to use, e.g., Interpreter, CPU, OpenCL:"),
    llvm::cl::Optional, llvm::cl::init("Interpreter"), llvm::cl::cat(category));

llvm::cl::opt<unsigned> numEpochs("epochs",
                                  llvm::cl::desc("Process the input N times."),
                                  llvm::cl::init(4), llvm::cl::value_desc("N"),
                                  llvm::cl::cat(category));
llvm::cl::opt<unsigned>
    generateChars("chars", llvm::cl::desc("Generate this number of chars."),
                  llvm::cl::init(10), llvm::cl::value_desc("N"),
                  llvm::cl::cat(category));

} // namespace

/// Clip the value \p c to the range 0..127, which is standard ascii.
static size_t clipASCII(char c) {
  size_t c1 = c;
  if (c1 > 127) {
    c1 = 127;
  }
  return c1;
}

/// Load text into \p inputText that has the format  [B, S, 128], where B is
/// the batch size, S is the length of the sentence, and 128 is the one-hot
/// representation of the text (https://en.wikipedia.org/wiki/One-hot).
/// Load the expected index into \p nextChar that has the format [B, S], where
/// each element is the softmax index of the next char. If \p train is false
/// then only load the first slice of inputText.
static void loadText(Tensor &inputText, Tensor &nextChar, llvm::StringRef text,
                     bool train) {
  DCHECK_GT(text.size(), 2) << "The buffer must contain at least two chars";
  inputText.zero();
  nextChar.zero();

  auto idim = inputText.dims();
  DCHECK_EQ(idim.size(), 3) << "invalid input tensor";
  auto B = idim[0];
  auto S = idim[1];

  auto IH = inputText.getHandle();
  auto NH = nextChar.getHandle<int64_t>();

  // Fill the tensor with slices from the sentence with an offset of 1.
  // Example:
  //  |Hell|o| World
  //  |ello| |World
  //  |llo |W|orld
  //  |lo W|o|rld
  for (dim_t i = 0; i < B; i++) {
    for (dim_t j = 0; j < S; j++) {
      dim_t c = clipASCII(text[i + j]);

      IH.at({i, j, c}) = 1.0;
      if (train) {
        size_t c1 = clipASCII(text[i + j + 1]);
        NH.at({i, j}) = c1;
      }
    }

    // Only load the first slice in the batch when in inference mode.
    if (!train) {
      return;
    }
  }
}

PseudoRNG &getRNG() {
  static PseudoRNG RNG;

  return RNG;
}

/// This method selects a random number based on a softmax distribution. One
/// property of this distribution is that the sum of all probabilities is equal
/// to one. The algorithm that we use here picks a random number between zero
/// and one. Then, we scan the tensor and accumulate the probabilities. We stop
/// and pick the index when sum is greater than the selected random number.
static char getPredictedChar(Tensor &inputText, dim_t slice, dim_t word) {
  auto IH = inputText.getHandle();

  // Pick a random number between zero and one.
  double x = std::abs(getRNG().nextRand());
  double sum = 0;
  // Accumulate the probabilities into 'sum'.
  for (dim_t i = 0; i < 128; i++) {
    sum += IH.at({slice, word, i});
    // As soon as we cross the threshold return the index.
    if (sum > x) {
      return i;
    }
  }
  return 127;
}

/// Loads the content of a file or stdin to a memory buffer.
/// The default filename of "-" reads from stdin.
static std::unique_ptr<llvm::MemoryBuffer> loadFile(llvm::StringRef filename) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileBufOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(filename);
  if (!fileBufOrErr) {
    LOG(ERROR) << "Error! Failed to open file: " << filename.str() << "\n";
    LOG(ERROR) << fileBufOrErr.getError().message() << "\n";
    exit(-1);
  }

  return std::move(fileBufOrErr.get());
}

/// Creates a new RNN network. The network answers the question, given N chars
/// of input, what is the character following each one of these chars.
static Function *createNetwork(Module &mod, PlaceholderBindings &bindings,
                               dim_t minibatchSize, dim_t numSteps,
                               dim_t hiddenSize) {
  Function *F = mod.createFunction("main");

  auto *X = mod.createPlaceholder(
      ElemKind::FloatTy, {minibatchSize, numSteps, 128}, "input", false);
  bindings.allocate(X);

  auto *Y = mod.createPlaceholder(ElemKind::Int64ITy, {minibatchSize, numSteps},
                                  "expected", false);
  bindings.allocate(Y);

  std::vector<NodeValue> slicesX;
  std::vector<Node *> expectedX;

  for (unsigned t = 0; t < numSteps; t++) {
    auto XtName = "X." + std::to_string(t);
    auto *Xt =
        F->createSlice(XtName, X, {0, t, 0}, {minibatchSize, t + 1, 128});
    slicesX.push_back(Xt);

    auto YtName = "Y." + std::to_string(t);
    auto *Yt = F->createSlice(YtName, Y, {0, t}, {minibatchSize, t + 1});
    expectedX.push_back(Yt);
  }

  std::vector<NodeValue> outputNodes;
  F->createLSTM(bindings, "rnn", slicesX, minibatchSize, hiddenSize, 128,
                outputNodes);

  std::vector<NodeValue> resX;
  for (unsigned i = 0; i < numSteps; i++) {
    auto *R =
        F->createReshape("reshapeSelector", expectedX[i], {minibatchSize, 1});
    auto *SM = F->createSoftMax("softmax", outputNodes[i], R);
    auto *K = F->createReshape("reshapeSM", SM, {minibatchSize, 1, 128});
    resX.push_back(K);
  }

  Node *O = F->createConcat("output", resX, 1);
  auto *S = F->createSave("result", O);
  bindings.allocate(S->getPlaceholder());

  return F;
}

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv, " The char-rnn test\n\n");
  auto mb = loadFile(inputFilename);
  auto text = mb.get()->getBuffer();
  LOG(INFO) << "Loaded " << text.size() << " chars.\n";
  PlaceholderBindings inferBindings, trainingBindings;

  const dim_t numSteps = 50;
  const dim_t minibatchSize = 32;
  const dim_t batchSize = text.size() - numSteps;
  const dim_t hiddenSize = 256;

  CHECK_GT(text.size(), numSteps) << "Text is too short";
  TrainingConfig TC;

  ExecutionEngine EET(executionBackend);
  TC.learningRate = 0.001;
  TC.momentum = 0.9;
  TC.batchSize = minibatchSize;

  auto &modT = EET.getModule();

  //// Train the network ////
  Function *F2 = createNetwork(modT, trainingBindings, minibatchSize, numSteps,
                               hiddenSize);
  differentiate(F2, TC);
  EET.compile(CompilationMode::Train);
  trainingBindings.allocate(modT.getPlaceholders());

  auto *XT = modT.getPlaceholderByNameSlow("input");
  auto *YT = modT.getPlaceholderByNameSlow("expected");

  Tensor thisCharTrain(ElemKind::FloatTy, {batchSize, numSteps, 128});
  Tensor nextCharTrain(ElemKind::Int64ITy, {batchSize, numSteps});
  loadText(thisCharTrain, nextCharTrain, text, true);

  // This variable records the number of the next sample to be used for
  // training.
  size_t sampleCounter = 0;

  // Run this number of iterations over the input. On each iteration: train the
  // network on the whole input and then generate some sample text.
  for (unsigned i = 0; i < numEpochs; i++) {

    // Train the network on the whole input.
    LOG(INFO) << "Iteration " << i + 1 << "/" << numEpochs;
    runBatch(EET, trainingBindings, batchSize / minibatchSize, sampleCounter,
             {XT, YT}, {&thisCharTrain, &nextCharTrain});

    ExecutionEngine EEO(executionBackend);
    inferBindings.clear();
    auto &mod = EEO.getModule();
    auto OF =
        createNetwork(mod, inferBindings, minibatchSize, numSteps, hiddenSize);
    auto *X = mod.getPlaceholderByNameSlow("input");
    inferBindings.allocate(mod.getPlaceholders());
    trainingBindings.copyTrainableWeightsTo(inferBindings);

    //// Use the trained network to generate some text ////
    auto *res =
        llvm::cast<SaveNode>(OF->getNodeByName("result"))->getPlaceholder();
    // Promote placeholders to constants.
    ::glow::convertPlaceholdersToConstants(OF, inferBindings, {X, res});
    EEO.compile(CompilationMode::Infer);

    // Load a few characters to start the text that we generate.
    Tensor currCharInfer(ElemKind::FloatTy, {minibatchSize, numSteps, 128});
    Tensor nextCharInfer(ElemKind::Int64ITy, {minibatchSize, numSteps});
    loadText(currCharInfer, nextCharInfer, text.slice(0, 128), false);

    auto *T = inferBindings.get(res);
    std::string result;
    std::string input;
    input.insert(input.begin(), text.begin(), text.begin() + numSteps);
    result = input;

    // Generate a sentence by running inference over and over again.
    for (unsigned i = 0; i < generateChars; i++) {
      // Generate a char:
      updateInputPlaceholders(inferBindings, {X}, {&currCharInfer});
      EEO.run(inferBindings);

      // Pick a char at random from the softmax distribution.
      char c = getPredictedChar(*T, 0, numSteps - 1);

      // Update the inputs for the next iteration:
      result.push_back(c);
      input.push_back(c);
      input.erase(input.begin());
      loadText(currCharInfer, nextCharInfer, input, false);
    }

    llvm::outs() << "Generated output:\n" << result << "\n";
  }

  return 0;
}
