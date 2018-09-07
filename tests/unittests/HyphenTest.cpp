/**
 * Copyright (c) 2018-present, Facebook, Inc.
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

// This file contains a set of tests related to a toy neural network
// that can hyphenate words. This isn't meant to represent great strides
// in machine learning research, but rather to exercise the CPU JIT
// compiler with a small end-to-end example. The toy network is small
// enough that it can be trained as part of the unit test suite.

#include "BackendTestUtils.h"

#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Graph/Graph.h"
#include "glow/IR/IR.h"
#include "glow/IR/IRBuilder.h"
#include "glow/IR/Instrs.h"
#include "glow/Support/Random.h"

#include "gtest/gtest.h"

#include <cctype>
#include <string>

using namespace glow;
using llvm::cast;
using std::string;
using std::vector;

// Network architecture
// ====================
//
// The network is a simple multi-layer perceptron with 27 x 6 input
// nodes, 10 inner nodes, and a 2-way soft-max output node.
//
// The input nodes represent 6 letters of a candidate word, and the
// output node indicates the probability that the word can be hyphenated
// between the 3rd and 4th letters. As a word slides past the 6-letter
// window, the network classifies each possible hyphen position.
//
// Example: "hyphenate"
//
// "..hyph" -> 0, h-yphenate is wrong.
// ".hyphe" -> 1, hy-phenate is right.
// "hyphen" -> 0, hyp-henate is wrong.
// "yphena" -> 0, hyph-enate is wrong.
// "phenat" -> 0, hyphe-nate is wrong.
// "henate" -> 1, hyphen-ate is right.
// "enate." -> 0, hyphena-te is wrong.
// "nate.." -> 0, hyphenat-e is wrong.

/// Parse an already hyphenated word into word windows and hyphen labels.
///
/// Given a word with embedded hyphens, generate a sequence of sliding
/// 6-character windows and associated boolean labels, like the table above.
///
static void dehyphenate(const char *hword, vector<string> &words,
                        vector<bool> &hyphens) {
  EXPECT_EQ(words.size(), hyphens.size());

  // The first character can't be a hyphen, and the word can't be null.
  EXPECT_TRUE(std::islower(*hword));
  string word = "..";
  word.push_back(*hword++);

  // Parse `hword` and add all the letters to `word` and hyphen/no-hyphen
  // entries to `hyphens`.
  for (; *hword; hword++) {
    bool hyph = (*hword == '-');
    hyphens.push_back(hyph);
    if (hyph) {
      hword++;
    }
    // There can't be multiple adjacent hyphens, and the word can't
    // end with a hyphen.
    EXPECT_TRUE(std::islower(*hword));
    word.push_back(*hword);
  }
  word += "..";

  // Now `word` contains the letters of `hword` surrounded by '..' on both
  // sides. Generate all 6-character windows and append them to `words`.
  for (size_t i = 0, e = word.size(); i + 5 < e; i++) {
    words.push_back(word.substr(i, 6));
  }
  EXPECT_EQ(words.size(), hyphens.size());
}

TEST(HyphenTest, dehyphenate) {
  vector<string> words;
  vector<bool> hyphens;

  dehyphenate("x", words, hyphens);
  EXPECT_EQ(words.size(), 0);
  EXPECT_EQ(hyphens.size(), 0);

  dehyphenate("xy", words, hyphens);
  EXPECT_EQ(words, (vector<string>{"..xy.."}));
  EXPECT_EQ(hyphens, (vector<bool>{0}));

  dehyphenate("y-z", words, hyphens);
  EXPECT_EQ(words, (vector<string>{"..xy..", "..yz.."}));
  EXPECT_EQ(hyphens, (vector<bool>{0, 1}));

  words.clear();
  hyphens.clear();
  dehyphenate("hy-phen-ate", words, hyphens);
  EXPECT_EQ(words, (vector<string>{"..hyph", ".hyphe", "hyphen", "yphena",
                                   "phenat", "henate", "enate.", "nate.."}));
  EXPECT_EQ(hyphens, (vector<bool>{0, 1, 0, 0, 0, 1, 0, 0}));
}

/// Map a lower-case letter to an input index in the range 0-26.
/// Use 0 to represent any characters outside the a-z range.
static size_t mapLetter(char l) {
  unsigned d = l - unsigned('a');
  return d < 26 ? d + 1 : 0;
}

TEST(HyphenTest, mapLetter) {
  EXPECT_EQ(mapLetter('a'), 1);
  EXPECT_EQ(mapLetter('d'), 4);
  EXPECT_EQ(mapLetter('z'), 26);
  EXPECT_EQ(mapLetter('.'), 0);
}

/// Map a 6-letter window of a word to an input tensor using a one-hot encoding.
///
/// The tensor must be N x 6 x 27: batch x position x letter.
static void mapLetterWindow(const string &window, size_t idx,
                            Handle<float> tensor) {
  EXPECT_EQ(window.size(), 6);
  for (size_t row = 0; row < 6; row++) {
    size_t col = mapLetter(window[row]);
    tensor.at({idx, row, col}) = 1;
  }
}

// Training data consisting of pre-hyphenated common words.
const vector<const char *> TrainingData{
    "ad-mi-ni-stra-tion",
    "ad-mit",
    "al-low",
    "al-though",
    "an-i-mal",
    "any-one",
    "ar-rive",
    "art",
    "at-tor-ney",
    "be-cause",
    "be-fore",
    "be-ha-vior",
    "can-cer",
    "cer-tain-ly",
    "con-gress",
    "coun-try",
    "cul-tural",
    "cul-ture",
    "de-cide",
    "de-fense",
    "de-gree",
    "de-sign",
    "de-spite",
    "de-velop",
    "di-rec-tion",
    "di-rec-tor",
    "dis-cus-sion",
    "eco-nomy",
    "elec-tion",
    "en-vi-ron-men-tal",
    "es-tab-lish",
    "ev-ery-one",
    "ex-actly",
    "ex-ec-u-tive",
    "ex-ist",
    "ex-pe-ri-ence",
    "ex-plain",
    "fi-nally",
    "for-get",
    "hun-dred",
    "in-crease",
    "in-di-vid-ual",
    "it-self",
    "lan-guage",
    "le-gal",
    "lit-tle",
    "lo-cal",
    "ma-jo-ri-ty",
    "ma-te-rial",
    "may-be",
    "me-di-cal",
    "meet-ing",
    "men-tion",
    "mid-dle",
    "na-tion",
    "na-tional",
    "oc-cur",
    "of-fi-cer",
    "par-tic-u-lar-ly",
    "pat-tern",
    "pe-riod",
    "phy-si-cal",
    "po-si-tion",
    "pol-icy",
    "pos-si-ble",
    "pre-vent",
    "pres-sure",
    "pro-per-ty",
    "pur-pose",
    "re-cog-nize",
    "re-gion",
    "re-la-tion-ship",
    "re-main",
    "re-sponse",
    "re-sult",
    "rea-son",
    "sea-son",
    "sex-ual",
    "si-mi-lar",
    "sig-ni-fi-cant",
    "sim-ple",
    "sud-den-ly",
    "sum-mer",
    "thou-sand",
    "to-day",
    "train-ing",
    "treat-ment",
    "va-ri-ous",
    "value",
    "vi-o-lence",
};

namespace {
struct HyphenNetwork {
  /// The input variable is N x 6 x 27 as encoded by mapLetterWindow().
  Variable *input_;

  /// The expected output index when training: 0 = no hyphen, 1 = hyphen.
  Variable *expected_;

  /// The forward inference function.
  Function *infer_;

  /// The result of the forward inference. N x 1 float with a probability.
  SaveNode *result_;

  /// The corresponding gradient function for training.
  Function *train_;

  HyphenNetwork(Module &mod, TrainingConfig &conf)
      : input_(mod.createVariable(ElemKind::FloatTy, {conf.batchSize, 6, 27},
                                  "input", VisibilityKind::Public, false)),
        expected_(mod.createVariable(ElemKind::Int64ITy, {conf.batchSize, 1},
                                     "expected", VisibilityKind::Public,
                                     false)),
        infer_(mod.createFunction("infer")), result_(nullptr), train_(nullptr) {
    Node *n;

    n = infer_->createFullyConnected("hidden_fc", input_, 10);
    n = infer_->createRELU("hidden", n);
    n = infer_->createFullyConnected("output_fc", n, 2);
    n = infer_->createSoftMax("output", n, expected_);
    result_ = infer_->createSave("result", n);
    train_ = glow::differentiate(infer_, conf);
  }

  // Run `inputs` through the inference function and check the results against
  // `hyphens`. Return the number of errors.
  unsigned inferenceErrors(ExecutionEngine &EE, llvm::StringRef name,
                           Tensor &inputs, const vector<bool> &hyphens,
                           TrainingConfig &TC) {
    // Compilation is destructive because of target-specific lowering.
    // Compile a clone of the inference function.
    EE.compile(CompilationMode::Infer, infer_->clone(name));

    auto batchSize = TC.batchSize;
    auto numSamples = inputs.dims()[0];
    EXPECT_LE(batchSize, numSamples);
    auto resultHandle = result_->getVariable()->getHandle<>();
    unsigned errors = 0;

    for (size_t bi = 0; bi < numSamples; bi += batchSize) {
      // Get a batch-sized slice of inputs and run them through the inference
      // function. Do a bit of overlapping if the batch size doesn't divide the
      // number of samples.
      if (bi + batchSize > numSamples) {
        bi = numSamples - batchSize;
      }
      auto batchInputs = inputs.getUnowned({batchSize, 6, 27}, {bi, 0, 0});
      updateVariables({input_}, {&batchInputs});
      EE.run();

      // Check each output in the batch.
      for (size_t i = 0; i != batchSize; i++) {
        // Note that the two softmax outputs always sum to 1, so we only look at
        // one.
        float value = resultHandle.at({i, 1});
        if ((value > 0.5) != hyphens[bi + i]) {
          errors++;
        }
      }
    }
    return errors;
  }
};
} // namespace

TEST(HyphenTest, network) {
  ExecutionEngine EE(BackendKind::CPU);

  // Convert the training data to word windows and labels.
  vector<string> words;
  vector<bool> hyphens;
  for (auto *hword : TrainingData) {
    dehyphenate(hword, words, hyphens);
  }

  // This depends on the training data, of course.
  const size_t numSamples = 566;
  ASSERT_EQ(hyphens.size(), numSamples);
  ASSERT_EQ(words.size(), numSamples);

  // Randomly shuffle the training data.
  // This is required for stochastic gradient descent training.
  auto &PRNG = EE.getModule().getPRNG();
  for (size_t i = numSamples - 1; i > 0; i--) {
    size_t j = PRNG.nextRandInt(0, i);
    std::swap(words[i], words[j]);
    std::swap(hyphens[i], hyphens[j]);
  }

  // Convert words and hyphens to a tensor representation.
  Tensor inputs(ElemKind::FloatTy, {numSamples, 6, 27});
  Tensor expected(ElemKind::Int64ITy, {numSamples, 1});
  auto inputHandle = inputs.getHandle<float>();
  auto expectedHandle = expected.getHandle<int64_t>();
  for (size_t i = 0; i != numSamples; i++) {
    mapLetterWindow(words[i], i, inputHandle);
    expectedHandle.at({i, 0}) = hyphens[i];
  }

  // Now build the network.
  TrainingConfig TC;
  TC.learningRate = 0.8;
  TC.batchSize = 50;
  HyphenNetwork net(EE.getModule(), TC);

  // This variable records the number of the next sample to be used for
  // training.
  size_t sampleCounter = 0;

  // Train using mini-batch SGD.
  EE.compile(CompilationMode::Train, net.train_);
  runBatch(EE, 1000, sampleCounter, {net.input_, net.expected_},
           {&inputs, &expected});

  // Now test inference on the trained network.
  // Note that we have probably overfitted the data, so we expect 100% accuracy.
  EXPECT_EQ(net.inferenceErrors(EE, "cpu", inputs, hyphens, TC), 0);

  // See of the interpreter gets the same result.
  EE.setBackend(BackendKind::Interpreter);
  EXPECT_EQ(net.inferenceErrors(EE, "interpreter", inputs, hyphens, TC), 0);
}
