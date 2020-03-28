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

#include "Loader.h"
#include "LoaderUtils.h"

#include "glow/Base/Image.h"
#include "glow/Graph/Graph.h"
#include "glow/Graph/Nodes.h"
#include "glow/Graph/Utils.h"
#include "glow/Quantization/Serialization.h"
#include "glow/Support/Support.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <chrono>
#include <fstream>
#include <memory>
#include <sstream>

using namespace glow;

namespace {

/// Model Tuner options
llvm::cl::OptionCategory modelTunerCat("Model Tuner Options");

llvm::cl::opt<std::string> datasetFileOpt(
    "dataset-file", llvm::cl::Required,
    llvm::cl::desc("Path to the dataset description file which contains on "
                   "each line a file path and an integer label separated by "
                   "space or comma. The integer labels start with 0 (0,1,..)."
                   "An example might look like this:\n"
                   "  image0.png 0   \n"
                   "  image1.png 13  \n"
                   "  .............  \n"
                   "Another example might look like this:\n"
                   "  image0.png,0,  \n"
                   "  image1.png,13, \n"
                   "  .............  \n"),
    llvm::cl::value_desc("file.txt|file.csv"), llvm::cl::cat(modelTunerCat));

llvm::cl::opt<std::string> datasetPathOpt(
    "dataset-path", llvm::cl::Required,
    llvm::cl::desc("The path of the directory where the dataset entries are "
                   "located."),
    llvm::cl::value_desc("directory path"), llvm::cl::cat(modelTunerCat));

llvm::cl::opt<std::string> dumpTunedProfileFileOpt(
    "dump-tuned-profile",
    llvm::cl::desc("Output quantization profile obtained after tuning."),
    llvm::cl::value_desc("profile_output.yaml"), llvm::cl::Required,
    llvm::cl::cat(modelTunerCat));

llvm::cl::opt<float> targetAccuracyOpt(
    "target-accuracy",
    llvm::cl::desc("Stop the quantization tuning/calibration procedure when \n"
                   "the accuracy has reached or surpassed the given value.  \n"
                   "A float value between 0.0 and 1.0 is expected. If not   \n"
                   "specified, the tuning will run until completion. "),
    llvm::cl::value_desc("float"), llvm::cl::Optional, llvm::cl::init(1.0),
    llvm::cl::cat(modelTunerCat));

llvm::cl::opt<unsigned> maxIterPerNodeOpt(
    "max-iter-per-node",
    llvm::cl::desc("Maximum number of tuning iterations per node (default 3)."),
    llvm::cl::value_desc("int"), llvm::cl::Optional, llvm::cl::init(3),
    llvm::cl::cat(modelTunerCat));

llvm::cl::opt<float> accDropSkipOpt(
    "acc-drop-skip",
    llvm::cl::desc("The accuracy drop for which the tuning of any node is \n"
                   "skipped. The default value is 0.05 (5%)."),
    llvm::cl::value_desc("float"), llvm::cl::Optional, llvm::cl::init(0.05),
    llvm::cl::cat(modelTunerCat));
} // namespace

/// Get maximum confidence class for the Softmax output.
/// Softmax output should have at least two dimensions: [batchSize, numLabels].
/// For this function we expect the batch size to be 1.
static std::pair<unsigned, float> getOutputClass(Tensor *T) {
  assert(T->dims().size() == 2 && "Softmax output should have 2 dimensions");
  assert(T->dims()[0] == 1 && "Softmax batch size must equal to 1");
  auto TH = T->getHandle<float>();
  float maxVal = TH.at({0, 0});
  unsigned maxIdx = 0;
  for (unsigned idx = 1; idx < TH.size(); ++idx) {
    if (TH.at({0, idx}) > maxVal) {
      maxVal = TH.at({0, idx});
      maxIdx = idx;
    }
  }
  return std::make_pair(maxIdx, maxVal);
}

/// Function to quantize the model with the given profiling infos and
/// compute the accuracy on the given data set.
float quantizeAndGetAccuracy(std::vector<NodeProfilingInfo> &pInfos,
                             LabeledDataSet &dataset) {

  // Initialize the loader object.
  Loader loader;

  // Load the model.
  auto protobufLoader = loader.loadModel();

  // Allocate tensors for all placeholders.
  PlaceholderBindings bindings;
  bindings.allocate(loader.getModule()->getPlaceholders());

  // Get input/output placeholders.
  Placeholder *input = EXIT_ON_ERR(protobufLoader->getSingleInput());
  Placeholder *output = EXIT_ON_ERR(protobufLoader->getSingleOutput());

  // Get compilation options for quantization.
  CompilationContext cctx =
      loader.getCompilationContext(QuantizationMode::Quantize);
  cctx.bindings = &bindings;

  // Force the given profiling infos.
  cctx.precisionConfig.quantConfig.infos = pInfos;

  // Compile the function.
  loader.compile(cctx);

  // Run the function for all the dataset.
  size_t correct = 0;
  for (const auto &data : dataset) {
    // Read the image and preprocess.
    Tensor inputImg = readPngImageAndPreprocess(data.first, imageNormMode,
                                                imageChannelOrder, imageLayout);
    auto imgShape = inputImg.getType().dims();
    Tensor inputTensor =
        inputImg.getUnowned({1, imgShape[0], imgShape[1], imgShape[2]});
    updateInputPlaceholders(*cctx.bindings, {input}, {&inputTensor});
    // Run inference.
    loader.runInference(*cctx.bindings, 1);
    // Get output class.
    auto cls = getOutputClass(cctx.bindings->get(output));
    if (cls.first == data.second) {
      ++correct;
    }
  }

  // Compute accuracy.
  return ((float)correct) / dataset.size();
}

/// Function to tune a given node for the given function with the given dataset.
float tuneQuantizationForNode(std::vector<NodeProfilingInfo> &pInfos,
                              LabeledDataSet &dataset, unsigned qIdx,
                              float bestAcc) {

  // Tuning parameters.
  unsigned maxIterPerNode = maxIterPerNodeOpt;
  float accDropSkip = accDropSkipOpt;

  // Backup profiling parameters for this node.
  auto bestTPP = pInfos[qIdx].tensorProfilingParams_;

  // Get quantization configuration.
  auto quantConfig = Loader::getQuantizationConfiguration();

  // Run the tune iterations for this node.
  for (unsigned iterIdx = 0; iterIdx < maxIterPerNode; ++iterIdx) {

    // Only symmetrical schema are supported. For asymmetrical schema
    // we need to shrink the range around the average.
    CHECK(quantConfig.schema == quantization::Symmetric ||
          quantConfig.schema == quantization::SymmetricWithPower2Scale)
        << "Only Symmetric and SymmetricWithPower2Scale schema are supported!";

    // Get current range with symmetrization.
    float rangeAbsMin = std::abs(pInfos[qIdx].tensorProfilingParams_.min);
    float rangeAbsMax = std::abs(pInfos[qIdx].tensorProfilingParams_.max);
    float rangeAbs = rangeAbsMax > rangeAbsMin ? rangeAbsMax : rangeAbsMin;

    // Test the quantization range by repeatedly shrinking with a factor of 2.
    float testMin = -rangeAbs / 2.0f;
    float testMax = +rangeAbs / 2.0f;
    pInfos[qIdx].tensorProfilingParams_.min = testMin;
    pInfos[qIdx].tensorProfilingParams_.max = testMax;
    llvm::outs() << strFormat("  [%d/%d] Testing range = [%.4f, %.4f]\n",
                              iterIdx + 1, maxIterPerNode, testMin, testMax);

    // Quantize model and compute accuracy for current params.
    float currAcc = quantizeAndGetAccuracy(pInfos, dataset);
    llvm::outs() << strFormat("  Accuracy = %.4f %%\n", currAcc * 100);

    // If we obtain EXACTLY the same accuracy then the profiling parameters
    // of this node have no side effects (most probably are not used).
    if (currAcc == bestAcc) {
      llvm::outs() << "  Tunning stopped for this node value (no effect)\n";
      break;
    }

    // If current accuracy is better then save the profiling parameters.
    if (currAcc > bestAcc) {
      bestAcc = currAcc;
      bestTPP = pInfos[qIdx].tensorProfilingParams_;
    }

    // If the current accuracy drops below the best accuracy with a given delta
    // then skip the tuning for the current node.
    bool lastIter = (iterIdx == (maxIterPerNode - 1));
    if (!lastIter && (currAcc < (bestAcc - accDropSkip))) {
      llvm::outs() << "  Tunning stopped for this node\n";
      break;
    }
  }

  // Save best profiling parameters for this node.
  pInfos[qIdx].tensorProfilingParams_ = bestTPP;
  llvm::outs() << strFormat("Best accuracy : %.4f %%\n", bestAcc * 100);
  return bestAcc;
}

int main(int argc, char **argv) {

  // Parse command line parameters. All the options will be available as part of
  // the loader object.
  parseCommandLine(argc, argv);

  // Get the input profile used for tuning.
  auto quantConfig = Loader::getQuantizationConfiguration();
  CHECK(quantConfig.infos.size())
      << "Input profile not found. Use the -load-profile option!";
  auto pInfosTune = quantConfig.infos;
  int nodeQNum = pInfosTune.size();

  // Read tuning dataset.
  LabeledDataSet datasetTune =
      readLabeledDataSet(datasetFileOpt, datasetPathOpt);

  // Set output stream to unbuffered state to flush every time.
  llvm::outs().SetUnbuffered();

  // Compute initial accuracy.
  llvm::outs() << strFormat("\nComputing initial accuracy ... \n");
  float accVal = quantizeAndGetAccuracy(pInfosTune, datasetTune);
  llvm::outs() << strFormat("Initial accuracy: %.4f %%\n", accVal * 100);
  llvm::outs() << strFormat("Number of nodes: %d\n", nodeQNum);
  llvm::outs() << strFormat("Target accuracy: %.4f %%\n\n",
                            targetAccuracyOpt * 100);

  // Perform tuning for all tunable nodes.
  auto startTime = getTimeStamp();
  for (int nodeQIdx = 0; nodeQIdx < nodeQNum; ++nodeQIdx) {

    // Stop tuning if target accuracy is achieved.
    if (accVal > targetAccuracyOpt) {
      llvm::outs() << "Target accuracy achieved! Tuning is stopped ...\n";
      break;
    }

    // Tune the quantization for this node.
    auto nodeName = pInfosTune[nodeQIdx].nodeOutputName_.data();
    llvm::outs() << strFormat("[%d/%d] Tuning node value \"%s\"\n",
                              nodeQIdx + 1, nodeQNum, nodeName);
    accVal = tuneQuantizationForNode(pInfosTune, datasetTune, nodeQIdx, accVal);

    // Display estimated remaining time and stats.
    unsigned iterSec = getDurationSec(startTime) / (nodeQIdx + 1);
    unsigned remSec = iterSec * (nodeQNum - nodeQIdx - 1);
    unsigned remMin = (remSec / 60) % 60;
    unsigned remHrs = (remSec / 60) / 60;
    llvm::outs() << strFormat("Iteration time: %d seconds\n", iterSec);
    llvm::outs() << strFormat("Remaining time: %d hours %d minutes\n\n", remHrs,
                              remMin);
  }

  // Print final accuracy.
  llvm::outs() << strFormat("\nFinal accuracy: %.4f %%\n\n", accVal * 100);

  // Print total time.
  unsigned totSec, totMin, totHrs;
  getDuration(startTime, totSec, totMin, totHrs);
  llvm::outs() << strFormat("Total time: %d hours %d minutes\n\n", totHrs,
                            totMin);

  // Serialize the tuned output profile.
  serializeProfilingInfosToYaml(dumpTunedProfileFileOpt, pInfosTune);

  return 0;
}
