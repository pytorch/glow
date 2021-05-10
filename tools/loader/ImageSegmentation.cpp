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

#include "glow/Base/Image.h"
#include "glow/Converter/TypeAToTypeBFunctionConverter.h"
#include "glow/Graph/Nodes.h"
#include "glow/Importer/Caffe2ModelLoader.h"
#include "glow/Importer/ONNXModelLoader.h"

#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/raw_ostream.h"

#include <atomic>
#include <fstream>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <sstream>
#include <thread>

#include "ExecutorCore.h"
#include "ExecutorCoreHelperFunctions.h"

using namespace glow;

namespace {

/// Segmentation options.
llvm::cl::OptionCategory segmentationCat("Segmentation Options");

llvm::cl::opt<std::string> inputMaskListFile(
    "input-mask-list-file",
    llvm::cl::desc("Name of the file containing list of mask images (one image "
                   "per line). This will also print IoU metric."),
    llvm::cl::value_desc("string_name"), llvm::cl::Optional,
    llvm::cl::cat(segmentationCat));

llvm::cl::opt<bool> saveOutputsOpt(
    "save-tensors",
    llvm::cl::desc(
        "Save model output as numpy array for offline post-processing."),
    llvm::cl::Optional, llvm::cl::init(false), llvm::cl::cat(segmentationCat));

llvm::cl::opt<std::string> pngPaletteFile(
    "png-palette-file",
    llvm::cl::desc(
        "Save output as PNG image by applying specified palette file."),
    llvm::cl::value_desc("string_name"), llvm::cl::Optional,
    llvm::cl::cat(segmentationCat));
} // unnamed namespace

// Vector storing the names of all input images w/o path and extension
// These names are used to name the files containing produced tensors & masks.
std::vector<std::string> outputImageFilenames_;

// Vector storing the names of all masks.
std::vector<std::string> inputMaskFilenames;

// Model produced masks (one for each input image in the batch).
std::vector<Tensor> maskTensors_;

// Model produced tensors, with ARGMAX applied on C dim (NHW dims)
std::vector<Tensor> outTensors_;

static void loadMasksAndPreprocess(const Tensor &inputImageData, size_t startId,
                                   size_t endId, size_t batchSz) {
  if (inputMaskListFile.empty()) {
    return;
  }

  const size_t hCoord = (imageLayoutOpt[0] == ImageLayout::NCHW) ? 2 : 1;
  const size_t wCoord = (imageLayoutOpt[0] == ImageLayout::NCHW) ? 3 : 2;

  maskTensors_.clear();
  std::vector<std::string> inputMaskBatchFilenames;
  VecVec<std::string> batchNames = {inputMaskBatchFilenames};

  getNextMiniBatch(batchNames, {inputMaskFilenames}, startId, batchSz, endId);

  for (const auto file : inputMaskBatchFilenames) {
    Tensor tmp;
    readPngImageIndexed(&tmp, file.c_str());
    maskTensors_.push_back(std::move(tmp));
  }
  DCHECK_EQ(maskTensors_[0].dims()[0], inputImageData.dims()[wCoord])
      << "Mask file dimensions should match image dimensions";
  DCHECK_EQ(maskTensors_[0].dims()[1], inputImageData.dims()[hCoord])
      << "Mask file dimensions should match image dimensions";
}

std::string setOutputFilename(std::string inPath) {
  auto start_pos = inPath.find_last_of("/\\");
  auto end_pos = inPath.find_last_of(".");
  std::string inFileName;
  if (start_pos != std::string::npos && end_pos != std::string::npos) {
    inFileName = inPath.substr(start_pos + 1, (end_pos - start_pos - 1));
  } else {
    llvm::outs() << "Error: Couldn't set filename for image: " << inPath
                 << "\n";
    exit(1);
  }
  return inFileName;
}

/// Apply argmax to input tensor of type ElemTy. Output argmax
/// tensor is always Int64ITy.
template <typename ElemTy>
static Tensor extractArgmax(const Tensor &IT, dim_t batch, dim_t hCoord,
                            dim_t wCoord, dim_t cCoord) {
  auto inH = IT.getHandle<ElemTy>();
  auto inDims = IT.dims();

  Tensor AMT(ElemKind::Int64ITy, {inDims[wCoord], inDims[hCoord]});
  auto outH = AMT.getHandle<int64_t>();

  for (dim_t x = 0; x < inDims[wCoord]; x++) {
    for (dim_t y = 0; y < inDims[hCoord]; y++) {
      ElemTy max = (imageLayoutOpt[0] == ImageLayout::NCHW)
                       ? inH.at({batch, 0, y, x})
                       : inH.at({batch, y, x, 0});
      int64_t idx = 0;
      for (dim_t cls = 0; cls < inDims[cCoord]; cls++) {
        ElemTy elem = (imageLayoutOpt[0] == ImageLayout::NCHW)
                          ? inH.at({batch, cls, y, x})
                          : inH.at({batch, y, x, cls});
        if (elem > max) {
          max = elem;
          idx = cls;
        }
      }
      outH.at({x, y}) = idx;
    }
  }
  return AMT;
}

// Find the best class for each pixel. Write the class to class dim entry zero.
// At this point original tensor is altered and can be consider as class dim
// is squeezed out. Always return int64 tensor.
Tensor getArgmaxTensor(const Tensor &IT, dim_t batch) {

  dim_t hCoord = (imageLayoutOpt[0] == ImageLayout::NCHW) ? 2 : 1;
  dim_t wCoord = (imageLayoutOpt[0] == ImageLayout::NCHW) ? 3 : 2;
  dim_t cCoord = (imageLayoutOpt[0] == ImageLayout::NCHW) ? 1 : 3;
  ShapeVector inShape = {batch, 0, 0, 0};

  auto inDims = IT.dims();
  if (inDims.size() == 3) {
    hCoord = 1;
    wCoord = 2;
    cCoord = 0;
    inShape.pop_back();
  }

  auto elemTy = IT.getElementType();

  // Argmax output should be int64_t but allow int32_t as well.
  if ((elemTy == ElemKind::Int64ITy || elemTy == ElemKind::Int32ITy) &&
      (inDims.size() == 3 || inDims[cCoord] == 1)) {
    llvm::outs() << "Recognizing network w/ argmax. Skip argmax calculation.\n";
    if (elemTy == ElemKind::Int64ITy) {
      return IT.getUnowned({inDims[wCoord], inDims[hCoord]}, inShape);
    }
    if (elemTy == ElemKind::Int32ITy) {
      Tensor tempT(ElemKind::Int64ITy, IT.dims());
      auto inH = IT.getHandle<int32_t>();
      auto outH = tempT.getHandle<int64_t>();
      for (dim_t i = 0; i < inH.size(); i++) {
        outH.raw(i) = inH.raw(i);
      }
      return tempT.getUnowned({inDims[wCoord], inDims[hCoord]}, inShape);
    }
  }

  // Extract argmax from the last layer.
  llvm::outs() << "Recognizing network w/o argmax. Applying argmax.\n";
  Tensor AMT(ElemKind::Int64ITy, {inDims[wCoord], inDims[hCoord]});
  switch (elemTy) {
  case ElemKind::Int32ITy: {
    return extractArgmax<int32_t>(IT, batch, hCoord, wCoord, cCoord);
  }
  case ElemKind::UInt8ITy: {
    return extractArgmax<uint8_t>(IT, batch, hCoord, wCoord, cCoord);
  }
  case ElemKind::Int64ITy: {
    return extractArgmax<int64_t>(IT, batch, hCoord, wCoord, cCoord);
  }
  case ElemKind::FloatTy: {
    return extractArgmax<float>(IT, batch, hCoord, wCoord, cCoord);
  }
  case ElemKind::Float16Ty: {
    return extractArgmax<float16_t>(IT, batch, hCoord, wCoord, cCoord);
  }
  default:
    LOG(FATAL) << "Argmax element type not supported: " << (int)elemTy;
    llvm_unreachable("Type not supported\n");
  }
}

void calcAndPrintIoU(const Tensor &OT, Tensor &MT, uint32_t numClasses,
                     std::string filename) {
  // Interesection, global (entry zero) and per class (entries > 0).
  std::vector<int64_t> I;
  I.resize(numClasses);

  // Union, global (entry zero) and per class (entries > 0).
  std::vector<int64_t> U;
  U.resize(numClasses);

  // Number of pixels belonging to a class in the output tensor.
  // Non-background pixels in entry zero.
  std::vector<int64_t> CSUM;
  CSUM.resize(numClasses);

  // Number of pixels belonging to a class in the mask.
  // All non-background pixels in entry zero.
  std::vector<int64_t> MSUM;
  MSUM.resize(numClasses);

  // Number of non-class mask pixels. Happens as masks can have white edge
  // boundaries around objects (val 255). Possibly compression artif. ?
  // They are cleared before doing any calc.
  uint32_t maskNonClassNum = 0;

  // TODO - change Image.cpp or getArgMaxTensor so tensor val type matches.
  auto OH = OT.getHandle<int64_t>();
  auto MH = MT.getHandle<int32_t>();

  for (dim_t idx = 0, end = OH.size(); idx != end; ++idx) {
    // Clear non-class mask pixels.
    if (MH.raw(idx) < 0 || MH.raw(idx) >= (int32_t)numClasses) {
      maskNonClassNum++;
      MH.raw(idx) = 0;
    }

    uint32_t imgPxl = OH.raw(idx);
    uint32_t mskPxl = MH.raw(idx);
    assert(mskPxl >= 0 && (uint32_t)mskPxl < numClasses &&
           "Non-class pixel found in image!");
    assert(imgPxl >= 0 && (uint32_t)imgPxl < numClasses &&
           "Non-class pixel found in mask!");

    // global intersection (except background)
    I[0] += (imgPxl == mskPxl && imgPxl > 0) ? 1 : 0;

    // global union (except background)
    U[0] += ((imgPxl != 0 || mskPxl != 0) && imgPxl > 0) ? 1 : 0;

    // global sum of all classes
    CSUM[0] += (imgPxl > 0 && (uint32_t)imgPxl <= numClasses) ? 1 : 0;
    MSUM[0] += (mskPxl > 0 && (uint32_t)mskPxl <= numClasses) ? 1 : 0;

    // per class I, U.
    for (uint32_t i = 1; i < numClasses; i++) {
      I[i] += (imgPxl == mskPxl && i == (uint32_t)imgPxl) ? 1 : 0;
      U[i] += ((imgPxl != 0 || mskPxl != 0) && (i == (uint32_t)imgPxl)) ? 1 : 0;
      // per class sum - need to know if class is in the image.
      CSUM[i] += (imgPxl != 0 && i == imgPxl) ? 1 : 0;
      MSUM[i] += (mskPxl != 0 && i == mskPxl) ? 1 : 0;
    }
  }
  // Print global IoU. Also print percentage pixels belonging to classes other
  // than background.
  double imgCov = 100.0 * CSUM[0] / OH.size();
  double mskCov = 100.0 * MSUM[0] / OH.size();
  double iou = 100.0 * I[0] / U[0];
  llvm::outs() << strFormat("Image: %s\n", filename.c_str());
  llvm::outs() << strFormat(
      "    IOU: %.2f (img/mask class coverage: %.2f/%.2f)\n", iou, imgCov,
      mskCov);

  // Print percentage of mask pixels that were cleared.
  llvm::outs() << strFormat("         Mask total non-class pixels: %1.3f\n",
                            100.0 * maskNonClassNum / OH.size());

  // For each class print some useful statics: just IoU for now, if class is
  // seen in image but not in mask, and vice versa. Statictis printed only
  // if a class presence in either mask/image is significant (> 1%).
  for (uint32_t i = 1; i < numClasses; i++) {
    double imgCov = 100.0 * CSUM[i] / OH.size();
    double mskCov = 100.0 * MSUM[i] / OH.size();
    if (imgCov < 1.0 && mskCov < 1.0) {
      continue;
    }
    double iou = 100.0 * I[i] / U[i];
    llvm::outs() << strFormat(
        "  IOU (%2d): %.2f (img/mask class coverage: %.2f/%.2f)\n", i, iou,
        imgCov, mskCov);
  }
}

/// Read all masks from \p inputMaskListFile in to \p inputMaskFilenames.
static void parseInputMaskList(const std::string &inputMaskListFile, int max) {
  std::ifstream inFile;
  inFile.open(inputMaskListFile);
  while (!inFile.eof()) {
    std::string img;
    getline(inFile, img);
    if (!img.empty()) {
      inputMaskFilenames.push_back(img);
    }
    if (--max == 0)
      break;
  }
  inFile.close();
}

/// Given the output Tensor in NCHW write WH tensors to a file, or PNG mask
/// images, one for each image in batch. The function applies argmax over the C
/// dimension to extract the pixel class and then saves tensors to binary file
/// and PNG Indexed image. Finally, some metrics on tensors are produced; IoU,
/// IoU per class (only for classes that have > 1% presence).
static int processAndPrintResults(const llvm::StringMap<Placeholder *> &PHM,
                                  PlaceholderBindings &bindings,
                                  VecVecRef<std::string> imageLists) {
  Placeholder *phOut = getOutputForPostProcessing(PHM);
  if (!phOut) {
    return 0;
  }

  /// No postprocessing if no option is enabled.
  if (!saveOutputsOpt && inputMaskListFile.empty() && pngPaletteFile.empty()) {
    return 0;
  }

  Tensor *SMT = bindings.get(phOut);
  auto dims = SMT->dims();

  // Input to process is always the one that's listed first.
  const std::vector<std::string> &imageList = imageLists[0];

  for (size_t b = 0; b < dims[0]; b++) {
    Tensor AMT = getArgmaxTensor(*SMT, b);

    std::string basename = setOutputFilename(imageList[b]);

    // Save WxH Tensor to a File.
    if (saveOutputsOpt) {
      writeNumpyImage(basename + ".npy", AMT);
    }
    // Write all produced masks as PNG files.
    if (!pngPaletteFile.empty()) {
      std::string name = basename + "_mask.png";
      bool status =
          writePngImageIndexed(&AMT, name.c_str(), pngPaletteFile.c_str());
      CHECK(!status) << "Writing PNG file failed.";
    }
    // calculate and print IOUs (global and per class).
    if (!inputMaskListFile.empty()) {
      calcAndPrintIoU(AMT, maskTensors_[b], dims[1], imageList[b]);
    }
  }
  return 0;
}

class ImageSegmentationProcessResult : public PostProcessOutputDataExtension {
public:
  int processOutputs(const llvm::StringMap<Placeholder *> &PHM,
                     PlaceholderBindings &bindings,
                     VecVecRef<std::string> inputImageBatchFilenames) {
    return processAndPrintResults(PHM, bindings, inputImageBatchFilenames);
  }
};

class SegmentationPreProcessor : public PreProcessInputDataExtension {
public:
  /// Called once per batch after images are loaded in to Tensor.
  void processInputTensor(llvm::ArrayRef<Tensor *> inputImagesData,
                          size_t startId, size_t endId, size_t batchSz) {
    if (!inputMaskListFile.empty()) {
      // Input to process is always the one listed first.
      loadMasksAndPreprocess(*inputImagesData[0], startId, endId, batchSz);
    }
  };
  ~SegmentationPreProcessor(){};
};

int main(int argc, char **argv) {
  glow::Executor core("ImageSegmentation", argc, argv);

  if (!inputMaskListFile.empty()) {
    assert(!inputImageListFileOpt.empty() &&
           "When using -input-mask-list-file images must be specified using "
           "-input-image-list-file option.");
    parseInputMaskList(inputMaskListFile, maxNumImages);
    CHECK_EQ(
        inputImageFilenamesOpt[0].size(),
        inputMaskFilenames.size() &&
            "Image & Mask list files should have the same number of entries.");
  }

  if (!pngPaletteFile.empty()) {
    llvm::outs() << "Using palette file: " << pngPaletteFile << "\n";
  }

  auto printResultCreator =
      []() -> std::unique_ptr<PostProcessOutputDataExtension> {
    return std::make_unique<ImageSegmentationProcessResult>();
  };
  core.registerPostProcessOutputExtension(printResultCreator);

  auto inputProcessing = []() -> std::unique_ptr<PreProcessInputDataExtension> {
    return std::make_unique<SegmentationPreProcessor>();
  };
  core.registerInputDataPreProcessingExtension(inputProcessing);

  int numErrors = core.executeNetwork();
  return numErrors;
}
