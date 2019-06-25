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
#include "glow/Base/Image.h"
#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Graph/Graph.h"
#include "glow/Importer/Caffe2ModelLoader.h"
#include "glow/Optimizer/GraphOptimizer/GraphOptimizer.h"
#include "glow/Optimizer/GraphOptimizer/TrainingPreparation.h"
#include "glow/Support/Support.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/raw_ostream.h"

#include <glog/logging.h>

#include <fstream>

using namespace glow;

const size_t mnistNumImages = 50000;

namespace {
llvm::cl::OptionCategory mnistCat("MNIST Options");
llvm::cl::opt<std::string> executionBackend(
    "backend",
    llvm::cl::desc("Backend to use, e.g., Interpreter, CPU, OpenCL:"),
    llvm::cl::Optional, llvm::cl::init("Interpreter"), llvm::cl::cat(mnistCat));
} // namespace

unsigned loadMNIST(Tensor &imageInputs, Tensor &labelInputs) {
  /// Load the MNIST database into 4D tensor of images and 2D tensor of labels.
  LOG(INFO) << "Loading the mnist database.";

  imageInputs.reset(ElemKind::FloatTy, {50000u, 28, 28, 1});
  labelInputs.reset(ElemKind::Int64ITy, {50000u, 1});

  std::ifstream imgInput("mnist_images.bin", std::ios::binary);
  std::ifstream labInput("mnist_labels.bin", std::ios::binary);

  CHECK(imgInput.is_open()) << "Error loading mnist_images.bin";
  CHECK(labInput.is_open()) << "Error loading mnist_labels.bin";

  std::vector<char> images((std::istreambuf_iterator<char>(imgInput)),
                           (std::istreambuf_iterator<char>()));
  std::vector<char> labels((std::istreambuf_iterator<char>(labInput)),
                           (std::istreambuf_iterator<char>()));
  float *imagesAsFloatPtr = reinterpret_cast<float *>(&images[0]);

  CHECK_EQ(labels.size() * 28 * 28 * sizeof(float), images.size())
      << "The size of the image buffer does not match the labels vector";

  size_t idx = 0;

  auto LIH = labelInputs.getHandle<int64_t>();
  auto IIH = imageInputs.getHandle<>();

  for (unsigned w = 0; w < mnistNumImages; w++) {
    LIH.at({w, 0}) = labels[w];
    for (unsigned x = 0; x < 28; x++) {
      for (unsigned y = 0; y < 28; y++) {
        IIH.at({w, x, y, 0}) = imagesAsFloatPtr[idx++];
      }
    }
  }
  size_t numImages = labels.size();
  CHECK_GT(numImages, 0) << "No images were found.";
  LOG(INFO) << "Loaded " << numImages << " images.";
  return numImages;
}

void createModel(ExecutionEngine &EE, Function *F,
                 PlaceholderBindings &bindings, unsigned minibatchSize,
                 Placeholder *&inputPH, Placeholder *&outputPH,
                 Placeholder *&selectedPH) {
  auto &mod = EE.getModule();

  inputPH = mod.createPlaceholder(ElemKind::FloatTy, {minibatchSize, 28, 28, 1},
                                  "input", false);

  auto *CV0 = F->createConv(bindings, "conv", inputPH, 16, 5, 1, 2, 1);
  auto *RL0 = F->createRELU("relu", CV0);
  auto *MP0 = F->createMaxPool("pool", RL0, 3, 3, 0);

  auto *CV1 = F->createConv(bindings, "conv", MP0, 16, 5, 1, 2, 1);
  auto *RL1 = F->createRELU("relu", CV1);
  auto *MP1 = F->createMaxPool("pool", RL1, 3, 3, 0);

  auto *FCL1 = F->createFullyConnected(bindings, "fc", MP1, 10);
  selectedPH = mod.createPlaceholder(ElemKind::Int64ITy, {minibatchSize, 1},
                                     "selected", false);
  auto *SM = F->createSoftMax("sm", FCL1, selectedPH);
  SaveNode *result = F->createSave("return", SM);
  outputPH = result->getPlaceholder();
}

void trainModel(ExecutionEngine &EE, PlaceholderBindings &bindings, Function *F,
                unsigned minibatchSize, unsigned numIterations,
                Tensor &imageInputs, Tensor &labelInputs, Placeholder *inputPH,
                Placeholder *selectedPH) {
  llvm::Timer timer("Training", "Training");
  /// The training configuration.
  TrainingConfig TC;

  // Construct the network:
  TC.learningRate = 0.001;
  TC.momentum = 0.9;
  TC.L2Decay = 0.001;
  TC.batchSize = minibatchSize;

  Function *TF = glow::differentiate(F, TC);

  EE.compile(CompilationMode::Train, TF);

  LOG(INFO) << "Training.";

  // This variable records the number of the next sample to be used for
  // training.
  size_t sampleCounter = 0;

  for (int epoch = 0; epoch < 60; epoch++) {
    LOG(INFO) << "Training - epoch #" << epoch;

    timer.startTimer();

    // On each training iteration take a slice of imageInputs and labelInputs
    // and put them into variables A and B, then run forward and backward passes
    // and update weights.
    runBatch(EE, bindings, numIterations, sampleCounter, {inputPH, selectedPH},
             {&imageInputs, &labelInputs});

    timer.stopTimer();
  }
}

void validateModel(ExecutionEngine &EE, PlaceholderBindings &bindings,
                   Function *F, unsigned minibatchSize, unsigned numIterations,
                   Tensor &imageInputs, Tensor &labelInputs,
                   Placeholder *inputPH, Placeholder *outputPH,
                   bool transpose) {
  LOG(INFO) << "Validating.";

  ::glow::convertPlaceholdersToConstants(F, bindings, {inputPH, outputPH});
  EE.compile(CompilationMode::Infer, F);

  auto LIH = labelInputs.getHandle<int64_t>();

  // Check how many examples out of eighty previously unseen digits we can
  // classify correctly.
  int rightAnswer = 0;

  for (int iter = numIterations; iter < numIterations + 10; iter++) {
    bindings.get(inputPH)->copyConsecutiveSlices(&imageInputs,
                                                 minibatchSize * iter);
    EE.run(bindings);

    for (unsigned i = 0; i < minibatchSize; i++) {
      auto T = bindings.get(outputPH)->getHandle().extractSlice(i);
      int64_t guess = T.getHandle().minMaxArg().second;

      int64_t correct = LIH.at({minibatchSize * iter + i, 0});
      rightAnswer += (guess == correct);

      if (iter == numIterations) {
        llvm::outs() << "MNIST Input";
        auto I = bindings.get(inputPH)->getHandle().extractSlice(i);
        if (transpose) {
          Tensor IT;
          // Transpose back to the ASCII printable format.
          // CHW -> HWC.
          I.transpose(&IT, {1, 2, 0});
          IT.getHandle().dumpAscii();
        } else {
          I.getHandle().dumpAscii();
        }
        llvm::outs() << "Expected: " << correct << " Guessed: " << guess
                     << "\n";

        T.getHandle<>().dump();
        llvm::outs() << "\n-------------\n";
      }
    }
  }

  llvm::outs() << "Results: guessed/total:" << rightAnswer << "/"
               << minibatchSize * 10 << "\n";
  CHECK_GE(rightAnswer, 74) << "Did not classify as many digits as expected";
}

/// This test classifies digits from the MNIST labeled dataset.
void testMNIST() {
  Tensor imageInputs;
  Tensor labelInputs;
  loadMNIST(imageInputs, labelInputs);

  ExecutionEngine EE(executionBackend);
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  Placeholder *A, *E, *selected;
  PlaceholderBindings bindings;
  unsigned minibatchSize = 8;

  createModel(EE, F, bindings, minibatchSize, A, E, selected);
  bindings.allocate(mod.getPlaceholders());

  const int numIterations = 30;

  trainModel(EE, bindings, F, minibatchSize, numIterations, imageInputs,
             labelInputs, A, selected);

  validateModel(EE, bindings, F, minibatchSize, numIterations, imageInputs,
                labelInputs, A, E, false /*transpose*/);
}

/// This test loads LENET-MNIST model, transferred it into the trainable form,
/// trained, and run prediction.
void testMNISTLoadAndTraining() {
  Tensor imageInputs;
  Tensor labelInputs;
  Tensor imageInputsTransposed;
  loadMNIST(imageInputsTransposed, labelInputs);
  imageInputsTransposed.transpose(&imageInputs, NHWC2NCHW);

  glow::ExecutionEngine EE;
  auto &mod = EE.getModule();
  auto *F = mod.createFunction("lenet_mnist");
  unsigned minibatchSize = 8;

  auto *inputType =
      mod.uniqueType(glow::ElemKind::FloatTy, {minibatchSize, 1, 28, 28});
  const char *inputName = "data";

  llvm::Error errPtr = llvm::Error::success();
  MARK_ERR_CHECKED(errPtr);
  // Load and compile LeNet MNIST model.
  glow::Caffe2ModelLoader loader("lenet_mnist/predict_net.pb",
                                 "lenet_mnist/init_net.pb", {inputName},
                                 {inputType}, *F, &errPtr);

  LOG(INFO) << "Loaded graph topology.";

  if (errPtr) {
    LOG(ERROR) << "Loader failed to load lenet_mnist model.";
    return;
  }

  glow::PlaceholderBindings bindings;
  // Get input and output placeholders.
  auto *A = llvm::cast<glow::Placeholder>(
      EXIT_ON_ERR(loader.getNodeValueByName(inputName)));
  auto *E = EXIT_ON_ERR(loader.getSingleOutput());
  bindings.allocate(mod.getPlaceholders());

  Placeholder *selected{nullptr};
  if ((errPtr = glow::prepareFunctionForTraining(F, bindings, selected))) {
    return;
  }

  const int numIterations = 30;

  trainModel(EE, bindings, F, minibatchSize, numIterations, imageInputs,
             labelInputs, A, selected);

  validateModel(EE, bindings, F, minibatchSize, numIterations, imageInputs,
                labelInputs, A, E, true /*transpose*/);
}

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv, " The MNIST test\n\n");
  testMNIST();
  testMNISTLoadAndTraining();

  return 0;
}
