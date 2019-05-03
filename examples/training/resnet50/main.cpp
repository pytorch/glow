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
#include "glow/Importer/Caffe2ModelLoader.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Timer.h"

#include <fstream>

/// This is an example demonstrating how to load resnet50 model,
/// randomize weights and perform training on a limited data set,
/// and evaluate model.

constexpr size_t IMAGE_COLORS = 3;
constexpr size_t IMAGE_HEIGHT = 224;
constexpr size_t IMAGE_WIDTH = 224;
constexpr size_t LABELS_SIZE = 1000;
constexpr size_t CIFAR_NUM_IMAGES = 20;

using namespace glow;

namespace {
llvm::cl::OptionCategory category("resnet-training Options");
llvm::cl::opt<unsigned> numEpochs("epochs", llvm::cl::desc("Number of epochs."),
                                  llvm::cl::init(32), llvm::cl::value_desc("N"),
                                  llvm::cl::cat(category));
llvm::cl::opt<unsigned> miniBatchSize("mini-batch",
                                      llvm::cl::desc("Mini batch size."),
                                      llvm::cl::init(8),
                                      llvm::cl::value_desc("N"),
                                      llvm::cl::cat(category));
llvm::cl::opt<float> learningRate("learning-rate",
                                  llvm::cl::desc("Learning rate parameter."),
                                  llvm::cl::init(0.001),
                                  llvm::cl::value_desc("N"),
                                  llvm::cl::cat(category));
llvm::cl::opt<float> momentum("momentum", llvm::cl::desc("Momentum parameter."),
                              llvm::cl::init(0.9), llvm::cl::value_desc("N"),
                              llvm::cl::cat(category));
llvm::cl::opt<float> l2Decay("l2-decay", llvm::cl::desc("L2Decay parameter."),
                             llvm::cl::init(0.0001), llvm::cl::value_desc("N"),
                             llvm::cl::cat(category));
llvm::cl::opt<BackendKind> executionBackendKind(
    llvm::cl::desc("Backend to use:"), llvm::cl::Optional,
    llvm::cl::values(clEnumValN(BackendKind::Interpreter, "Interpreter",
                                "Use interpreter (default option)"),
                     clEnumValN(BackendKind::CPU, "cpu", "Use CPU"),
                     clEnumValN(BackendKind::OpenCL, "opencl", "Use OpenCL")),
    llvm::cl::init(BackendKind::Interpreter), llvm::cl::cat(category));

/// Finds the maximum value and its index within \p N range.
size_t findMaxIndex(const Tensor &t, size_t range) {
  auto handle = t.getHandle();
  size_t ret{0};
  float maxVal{handle.raw(0)};
  for (size_t c = 1; c < range; ++c) {
    auto val = handle.raw(c);
    if (maxVal <= val) {
      maxVal = val;
      ret = c;
    }
  }
  return ret;
}

void loadImagesAndLabels(Tensor &images, Tensor &labels) {
  llvm::outs() << "Loading the CIFAR-10 database.\n";
  /// ResNet expects ImageChannelOrder::BGR, ImageLayout::NCHW.
  std::ifstream dbInput("cifar-10-batches-bin/data_batch_1.bin",
                        std::ios::binary);

  if (!dbInput.is_open()) {
    llvm::errs() << "Failed to open cifar10 data file, probably missing.\n";
    llvm::errs() << "Run 'python ../glow/utils/download_test_db.py'\n";
    exit(1);
  }

  /// The CIFAR file format is structured as one byte label in the range 0..9.
  /// The label is followed by an image: 32 x 32 pixels, in RGB format. Each
  /// color is 1 byte. The first 1024 red bytes are followed by 1024 of green
  /// and blue. Each 1024 byte color slice is organized in row-major format.
  /// The database contains 10000 images.
  /// Size: (1 + (32 * 32 * 3)) * 10000 = 30730000.
  llvm::outs() << "Assigning images/labels bytes.\n";
  auto labelsH = labels.getHandle();
  auto imagesH = images.getHandle();
  float scale = 1. / 255.0;
  float bias = 0;

  /// CIFAR image dimensions are 32x32, ResNet50 expects 224x224.
  /// Simple scaling would be one CIFAR pixel copied into 7x7 area
  constexpr unsigned expandImage = 7;
  for (unsigned n = 0; n < CIFAR_NUM_IMAGES; ++n) {
    unsigned long l = dbInput.get();
    labelsH.at({n, l}) = 1;
    // Even ResNet50 model got trained in NCHW format
    for (unsigned c = 0; c < IMAGE_COLORS; ++c) {
      auto bgrc = IMAGE_COLORS - 1 - c;
      for (unsigned h = 0; h < 32; ++h) {
        for (unsigned w = 0; w < 32; ++w) {
          // ResNet BGR color space vs CIFAR RGB.
          auto val = static_cast<float>(static_cast<uint8_t>(dbInput.get()));
          // Normalize and scale image.
          val = (val - imagenetNormMean[c]) * scale / imagenetNormStd[c] + bias;
          for (unsigned sh = 0; sh < expandImage; ++sh) {
            for (unsigned sw = 0; sw < expandImage; ++sw) {
              imagesH.at(
                  {n, bgrc, sh + h * expandImage, sw + w * expandImage}) = val;
            }
          }
        }
      }
    }
  }
}
} // namespace

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    " ResNet50 Training Example\n\n");

  // We expect the input to be NCHW.
  llvm::ArrayRef<size_t> allImagesDims = {CIFAR_NUM_IMAGES, IMAGE_COLORS,
                                          IMAGE_HEIGHT, IMAGE_WIDTH};
  llvm::ArrayRef<size_t> batchImagesDims = {miniBatchSize, IMAGE_COLORS,
                                            IMAGE_HEIGHT, IMAGE_WIDTH};
  llvm::ArrayRef<size_t> initShapeDims = {1, IMAGE_COLORS, IMAGE_HEIGHT,
                                          IMAGE_WIDTH};
  llvm::ArrayRef<size_t> allLabelsDims = {CIFAR_NUM_IMAGES, LABELS_SIZE};

  Module mod;
  PlaceholderBindings bindings;
  const char inputName[] = "gpu_0/data";
  TypeRef inputType = mod.uniqueType(ElemKind::FloatTy, initShapeDims);
  // Load ResNet model.
  Function *F = mod.createFunction("resnet50");
  llvm::outs() << "Loading resnet50 model.\n";
  // Loader has randomly initialized trainable weights.
  Caffe2ModelLoader loader("resnet50/predict_net.pbtxt", "resnet50/init_net.pb",
                           {inputName}, {inputType}, *F, nullptr, &bindings);

  llvm::outs() << "Creating input/output placeholders.\n";
  // Create the input layer.
  auto *A = mod.createPlaceholder(ElemKind::FloatTy, batchImagesDims, "input",
                                  /* isTraining */ false);
  bindings.allocate(A);
  // Create the output layer.
  auto *E =
      mod.createPlaceholder(ElemKind::FloatTy, {miniBatchSize, LABELS_SIZE},
                            "expected", /* isTraining */ false);
  bindings.allocate(E);

  llvm::outs() << "Differentiating graph.\n";
  TrainingConfig TC;
  TC.learningRate = learningRate;
  TC.momentum = momentum;
  TC.L2Decay = l2Decay;
  TC.batchSize = miniBatchSize;
  Function *TF = glow::differentiate(F, TC);

  llvm::outs() << "Compiling backend.\n";
  BackendKind executionBackend{executionBackendKind};
  ExecutionEngine EE(executionBackend);
  EE.compile(CompilationMode::Train, TF);

  // Allocate placeholders without tensors.
  bindings.allocate(mod.getPlaceholders());

  // Get output tensor.
  auto *resultPH = mod.getPlaceholderByName("save_gpu_0_softmax");
  auto *result = bindings.get(resultPH);

  // These tensors allocate memory for all images and labels prepared for
  // training.
  Tensor images(ElemKind::FloatTy, allImagesDims);
  Tensor labels(ElemKind::FloatTy, allLabelsDims);

  loadImagesAndLabels(images, labels);

  // From http://www.cs.toronto.edu/~kriz/cifar.html.
  const char *textualLabels[LABELS_SIZE] = {
      "airplane", "automobile", "bird",  "cat",  "deer",
      "dog",      "frog",       "horse", "ship", "truck"};

  size_t sampleCounter = 0;
  int reportRate = 16;

  llvm::Timer timer("Training", "Training");

  llvm::outs() << "Training.\n";
  auto labelsH = labels.getHandle();
  for (size_t e = 0; e < numEpochs; ++e) {
    llvm::outs() << "Training - epoch #" << e << " from total " << numEpochs
                 << "\n";
    // Train network on all examples.
    timer.startTimer();
    runBatch(EE, bindings, reportRate, sampleCounter, {A, E},
             {&images, &labels});
    timer.stopTimer();

    llvm::outs() << "Evaluating.\n";
    unsigned score = 0;
    unsigned total = 0;

    for (unsigned int i = 0; i < CIFAR_NUM_IMAGES / miniBatchSize; ++i) {
      Tensor sample(ElemKind::FloatTy, batchImagesDims);
      sample.copyConsecutiveSlices(&images, miniBatchSize * i);
      updateInputPlaceholders(bindings, {A}, {&sample});
      EE.run(bindings);

      for (unsigned int example = 0; example < miniBatchSize; ++example) {
        auto correct =
            findMaxIndex(labelsH.extractSlice(miniBatchSize * i + example), 10);
        auto guess = findMaxIndex(*result, 10);

        score += guess == correct;
        ++total;

        if (i == 0) {
          llvm::outs() << example << ") Expected: " << textualLabels[correct]
                       << ", Got: " << textualLabels[guess] << "\n";
        }
      }
    }

    llvm::outs() << "Total accuracy: " << 100. * score / total << "\n";
  }

  return 0;
}
