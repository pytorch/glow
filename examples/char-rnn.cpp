/**
 * Copyright (c) 2017-present, Facebook, Inc.
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
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Timer.h"

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

llvm::cl::opt<BackendKind> executionBackend(
    llvm::cl::desc("Backend to use:"),
    llvm::cl::values(clEnumValN(BackendKind::Interpreter, "interpreter",
                                "Use interpreter (default option)"),
                     clEnumValN(BackendKind::CPU, "cpu", "Use CPU"),
                     clEnumValN(BackendKind::OpenCL, "opencl", "Use OpenCL")),
    llvm::cl::init(BackendKind::Interpreter), llvm::cl::cat(category));

llvm::cl::opt<unsigned> numEpochs("epochs",
                                  llvm::cl::desc("Process the input N times."),
                                  llvm::cl::init(4), llvm::cl::value_desc("N"),
                                  llvm::cl::cat(category));
llvm::cl::opt<unsigned>
    generateChars("chars", llvm::cl::desc("Generate this number of chars."),
                  llvm::cl::init(1000), llvm::cl::value_desc("N"),
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
  assert(text.size() > 2 && "The buffer must contain at least two chars");
  inputText.zero();
  nextChar.zero();

  auto idim = inputText.dims();
  assert(idim.size() == 3 && "invalid input tensor");
  auto B = idim[0];
  auto S = idim[1];

  auto IH = inputText.getHandle();
  auto NH = nextChar.getHandle<size_t>();

  // Fill the tensor with slices from the sentence with an offset of 1.
  // Example:
  //  |Hell|o| World
  //  |ello| |World
  //  |llo |W|orld
  //  |lo W|o|rld
  for (size_t i = 0; i < B; i++) {
    for (size_t j = 0; j < S; j++) {
      size_t c = clipASCII(text[i + j]);

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

PseudoRNG RNG;

/// This method selects a random number based on a softmax distribution. One
/// property of this distribution is that the sum of all probabilities is equal
/// to one. The algorithm that we use here picks a random number between zero
/// and one. Then, we scan the tensor and accumulate the probabilities. We stop
/// and pick the index when sum is greater than the selected random number.
static char getPredictedChar(Tensor &inputText, size_t slice, size_t word) {
  auto IH = inputText.getHandle();

  // Pick a random number between zero and one.
  double x = std::abs(RNG.nextRand());
  double sum = 0;
  // Accumulate the probabilities into 'sum'.
  for (size_t i = 0; i < 128; i++) {
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
    llvm::errs() << "Error! Failed to open file: " << filename << "\n";
    llvm::errs() << fileBufOrErr.getError().message() << "\n";
    exit(-1);
  }

  return std::move(fileBufOrErr.get());
}

/// Creates a new RNN network. The network answers the question, given N chars
/// of input, what is the character following each one of these chars.
static Function *createNetwork(Module &mod, size_t minibatchSize,
                               size_t numSteps, size_t hiddenSize) {
  Function *F = mod.createFunction("main");

  Variable *X = mod.createVariable(
      ElemKind::FloatTy, {minibatchSize, numSteps, 128}, "input",
      VisibilityKind::Public, Variable::TrainKind::None);
  Variable *Y = mod.createVariable(ElemKind::IndexTy, {minibatchSize, numSteps},
                                   "expected", VisibilityKind::Public,
                                   Variable::TrainKind::None);
  std::vector<Node *> slicesX;
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
  F->createSimpleRNN("rnn", slicesX, minibatchSize, hiddenSize, 128,
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
  F->createSave("result", O);
  return F;
}

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv, " The char-rnn test\n\n");
  auto mb = loadFile(inputFilename);
  auto text = mb.get()->getBuffer();
  llvm::outs() << "Loaded " << text.size() << " chars.\n";

  const size_t numSteps = 20;
  const size_t minibatchSize = 16;
  const size_t batchSize = text.size() - numSteps;
  const size_t hiddenSize = 128;

  GLOW_ASSERT(text.size() > numSteps && "Text is too short");

  ExecutionEngine EE(executionBackend);
  EE.getConfig().learningRate = 0.01;
  EE.getConfig().momentum = 0;
  EE.getConfig().batchSize = minibatchSize;

  auto &mod = EE.getModule();

  //// Train the network ////
  Function *F = createNetwork(mod, minibatchSize, numSteps, hiddenSize);
  Function *TF = differentiate(F, EE.getConfig());
  EE.compile(CompilationMode::Train, TF);

  auto *X = mod.getVariableByName("input");
  auto *Y = mod.getVariableByName("expected");

  Tensor thisCharTrain(ElemKind::FloatTy, {batchSize, numSteps, 128});
  Tensor nextCharTrain(ElemKind::IndexTy, {batchSize, numSteps});
  loadText(thisCharTrain, nextCharTrain, text, true);

  for (unsigned i = 0; i < numEpochs; i++) {
    llvm::outs() << "Iteration " << i + 1 << "/" << numEpochs;
    EE.runBatch(batchSize / minibatchSize, {X, Y},
                {&thisCharTrain, &nextCharTrain});
    llvm::outs() << ".\n";
  }

  //// Use the trained network to generate some text ////
  EE.compile(CompilationMode::Infer, F);

  // Load a few characters to start the text that we generate.
  Tensor currCharInfer(ElemKind::FloatTy, {minibatchSize, numSteps, 128});
  Tensor nextCharInfer(ElemKind::IndexTy, {minibatchSize, numSteps});
  loadText(currCharInfer, nextCharInfer, text.slice(0, 128), false);

  auto *res = llvm::cast<SaveNode>(F->getNodeByName("result"));
  auto &T = res->getVariable()->getPayload();

  std::string result;
  std::string input;
  input.insert(input.begin(), text.begin(), text.begin() + numSteps);
  result = input;

  for (unsigned i = 0; i < generateChars; i++) {
    EE.run({X}, {&currCharInfer});
    char c = getPredictedChar(T, 0, numSteps - 1);
    result.push_back(c);
    input.push_back(c);
    input.erase(input.begin());
    loadText(currCharInfer, nextCharInfer, input, false);
  }

  llvm::outs() << "Generated output:\n" << result << "\n";
  return 0;
}
