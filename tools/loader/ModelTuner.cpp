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

/// Get maximum confidence class (index and value) for the model output.
static std::pair<unsigned, float> getOutputClass(Tensor *T) {
  CHECK(T->getElementType() == ElemKind::FloatTy)
      << "Model output is expected to be float!";
  auto TH = T->getHandle<float>();
  float maxVal = TH.raw(0);
  unsigned maxIdx = 0;
  for (unsigned idx = 1; idx < TH.size(); ++idx) {
    if (TH.raw(idx) > maxVal) {
      maxVal = TH.raw(idx);
      maxIdx = idx;
    }
  }
  return std::make_pair(maxIdx, maxVal);
}

/// Function to run the model using the given \p dataset and compute the
/// accuracy. If \p quantize flag is given then the model is additionally
/// quantized using the profiling information \p pInfos.
float runModelAndGetAccuracy(LabeledDataSet &dataset, bool quantize,
                             std::vector<NodeProfilingInfo> &pInfos) {

  // Initialize the loader object.
  Loader loader;

  // Load the model.
  loader.loadModel();

  // Allocate tensors for all placeholders.
  PlaceholderBindings bindings;
  bindings.allocate(loader.getModule()->getPlaceholders());

  // Get input/output placeholders.
  auto inpPHMap = loader.getInputPlaceholderMap();
  auto outPHMap = loader.getOutputPlaceholderMap();
  CHECK(inpPHMap.size() == 1) << "Model is expected to have only 1 input!";
  CHECK(outPHMap.size() == 1) << "Model is expected to have only 1 output!";
  Placeholder *input = inpPHMap.begin()->second;
  Placeholder *output = outPHMap.begin()->second;

  // Get compilation options.
  CompilationContext cctx;
  if (quantize) {
    // Get compilation options for quantization.
    cctx = loader.getCompilationContext(QuantizationMode::Quantize);
    // Force the given profiling infos.
    cctx.precisionConfig.quantConfig.infos = pInfos;
  } else {
    // Get compilation options for running the model as-is.
    cctx = loader.getCompilationContext(QuantizationMode::None);
  }
  cctx.bindings = &bindings;

  // Compile the function.
  loader.compile(cctx);

  // Run the function for all the dataset.
  size_t correct = 0;
  for (const auto &data : dataset) {
    // Read the image and preprocess.
    Tensor inputImg =
        readPngImageAndPreprocess(data.first, imageNormMode[0],
                                  imageChannelOrderOpt[0], imageLayoutOpt[0]);
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

/// Function to tune a given tensor for the given function with the given
/// dataset.
float tuneQuantizationForTensor(std::vector<NodeProfilingInfo> &pInfos,
                                LabeledDataSet &dataset, unsigned qIdx,
                                float bestAcc) {

  // Tuning parameters.
  unsigned maxIterPerNode = maxIterPerNodeOpt;
  float accDropSkip = accDropSkipOpt;

  // Backup profiling parameters for this tensor.
  auto bestTPP = pInfos[qIdx].tensorProfilingParams_;

  // Get tensor average value.
  float tensorAvgVal = quantization::getTensorAverageValue(bestTPP);

  // Get quantization configuration.
  auto quantConfig = Loader::getQuantizationConfiguration();

  // Run the tune iterations for this tensor.
  for (unsigned iterIdx = 0; iterIdx < maxIterPerNode; ++iterIdx) {

    // Get current min/max range.
    float rangeMin = pInfos[qIdx].tensorProfilingParams_.min;
    float rangeMax = pInfos[qIdx].tensorProfilingParams_.max;

    // Skip tuning for this tensor if range is empty.
    if (rangeMin == rangeMax) {
      llvm::outs() << "  Tuning skipped for this tensor: not required\n";
      break;
    }

    // Get testing min/max range by repeatedly shrinking with a factor of 2.
    float testMin, testMax;
    if (quantConfig.schema == quantization::Asymmetric) {
      // Shrink tensor min/max range around average value.
      testMin = tensorAvgVal - (tensorAvgVal - rangeMin) / 2.0;
      testMax = tensorAvgVal + (rangeMax - tensorAvgVal) / 2.0;
    } else if (quantConfig.schema == quantization::Symmetric ||
               quantConfig.schema == quantization::SymmetricWithUnsigned ||
               quantConfig.schema == quantization::SymmetricWithPower2Scale) {
      // Shrink tensor min/max range around 0.
      float rangeAbsMin = std::abs(rangeMin);
      float rangeAbsMax = std::abs(rangeMax);
      float rangeAbs = rangeAbsMax > rangeAbsMin ? rangeAbsMax : rangeAbsMin;
      testMin = -rangeAbs / 2.0f;
      testMax = +rangeAbs / 2.0f;
    } else {
      llvm_unreachable("Quantization schema not supported!");
    }

    // Set the testing range.
    pInfos[qIdx].tensorProfilingParams_.min = testMin;
    pInfos[qIdx].tensorProfilingParams_.max = testMax;
    llvm::outs() << strFormat("  [%d/%d] Testing range = [%.4f, %.4f]\n",
                              iterIdx + 1, maxIterPerNode, testMin, testMax);

    // Quantize model and compute accuracy for current params.
    float currAcc = runModelAndGetAccuracy(dataset, true, pInfos);
    llvm::outs() << strFormat("  Accuracy = %.4f %%\n", currAcc * 100);

    // If we obtain EXACTLY the same accuracy then the profiling parameters
    // of this tensor have no side effects (most probably are not used).
    if (currAcc == bestAcc) {
      llvm::outs()
          << "  Tuning stopped for this tensor: accuracy not improved\n";
      break;
    }

    // If current accuracy is better then save the profiling parameters.
    if (currAcc > bestAcc) {
      bestAcc = currAcc;
      bestTPP = pInfos[qIdx].tensorProfilingParams_;
    }

    // If the current accuracy drops below the best accuracy with a given delta
    // then skip the tuning for the current tensor.
    bool lastIter = (iterIdx == (maxIterPerNode - 1));
    if (!lastIter && (currAcc < (bestAcc - accDropSkip))) {
      llvm::outs() << "  Tuning stopped for this tensor: accuracy dropped more "
                      "than \"acc-drop-skip\"\n";
      break;
    }
  }

  // Save best profiling parameters for this tensor.
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
  int tensorQNum = pInfosTune.size();

  // Read tuning dataset.
  LabeledDataSet datasetTune =
      readLabeledDataSet(datasetFileOpt, datasetPathOpt);

  // Set output stream to unbuffered state to flush every time.
  llvm::outs().SetUnbuffered();

  // Compute initial accuracy.
  llvm::outs() << strFormat("\nComputing initial accuracy ... \n");
  float accValF = runModelAndGetAccuracy(datasetTune, false, pInfosTune);
  float accValQ = runModelAndGetAccuracy(datasetTune, true, pInfosTune);
  llvm::outs() << strFormat("Initial accuracy: %.4f %% (FLOAT)\n",
                            accValF * 100);
  llvm::outs() << strFormat("Initial accuracy: %.4f %% (QUANTIZED)\n",
                            accValQ * 100);
  llvm::outs() << strFormat("Target  accuracy: %.4f %% (QUANTIZED)\n",
                            targetAccuracyOpt * 100);
  llvm::outs() << strFormat("Number of tensors: %d\n\n", tensorQNum);

  // Perform tuning for all tunable tensors.
  float accVal = accValQ;
  auto startTime = getTimeStamp();
  for (int tensorQIdx = 0; tensorQIdx < tensorQNum; ++tensorQIdx) {

    // Stop tuning if target accuracy is achieved.
    if (accVal > targetAccuracyOpt) {
      llvm::outs() << "Target accuracy achieved! Tuning is stopped ...\n";
      break;
    }

    // Tune the quantization for this tensor.
    auto tensorName = pInfosTune[tensorQIdx].nodeOutputName_.data();
    llvm::outs() << strFormat("[%d/%d] Tuning quantization for tensor \"%s\"\n",
                              tensorQIdx + 1, tensorQNum, tensorName);
    accVal =
        tuneQuantizationForTensor(pInfosTune, datasetTune, tensorQIdx, accVal);

    // Display estimated remaining time and stats.
    unsigned iterSec = getDurationSec(startTime) / (tensorQIdx + 1);
    unsigned remSec = iterSec * (tensorQNum - tensorQIdx - 1);
    unsigned remMin = (remSec / 60) % 60;
    unsigned remHrs = (remSec / 60) / 60;
    llvm::outs() << strFormat("Iteration time: %d seconds\n", iterSec);
    llvm::outs() << strFormat("Remaining time: %d hours %d minutes\n\n", remHrs,
                              remMin);
  }

  // Print final accuracy.
  llvm::outs() << strFormat("\nFinal accuracy: %.4f %% (QUANTIZED)\n\n",
                            accVal * 100);

  // Print total time.
  unsigned totSec, totMin, totHrs;
  getDuration(startTime, totSec, totMin, totHrs);
  llvm::outs() << strFormat("Total time: %d hours %d minutes\n\n", totHrs,
                            totMin);

  // Serialize the tuned output profile.
  serializeProfilingInfosToYaml(dumpTunedProfileFileOpt,
                                quantConfig.graphPreLowerHash, pInfosTune);

  return 0;
}
