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

#include "Loader.h"

#include "glow/Base/Image.h"
#include "glow/IR/IR.h"
#include "glow/Graph/Nodes.h"
#include "glow/Importer/Caffe2.h"
#include "glow/Importer/ONNX.h"

#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>

using namespace glow;

enum class ImageNormalizationMode {
  kneg1to1,     // Values are in the range: -1 and 1.
  k0to1,        // Values are in the range: 0 and 1.
  k0to256,      // Values are in the range: 0 and 256.
  kneg128to127, // Values are in the range: -128 .. 127
};

ImageNormalizationMode strToImageNormalizationMode(const std::string &str) {
  return llvm::StringSwitch<ImageNormalizationMode>(str)
      .Case("neg1to1", ImageNormalizationMode::kneg1to1)
      .Case("0to1", ImageNormalizationMode::k0to1)
      .Case("0to256", ImageNormalizationMode::k0to256)
      .Case("neg128to127", ImageNormalizationMode::kneg128to127);
  GLOW_ASSERT(false && "Unknown image format");
}

/// Convert the normalization to numeric floating poing ranges.
std::pair<float, float> normModeToRange(ImageNormalizationMode mode) {
  switch (mode) {
  case ImageNormalizationMode::kneg1to1:
    return {-1., 1.};
  case ImageNormalizationMode::k0to1:
    return {0., 1.0};
  case ImageNormalizationMode::k0to256:
    return {0., 256.0};
  case ImageNormalizationMode::kneg128to127:
    return {-128., 127.};
  default:
    GLOW_ASSERT(false && "Image format not defined.");
  }
}

namespace {

/// Image loader options.
llvm::cl::OptionCategory imageLoaderCat("Image Loader Options");
llvm::cl::list<std::string> inputImageFilenames(llvm::cl::Positional,
                                                llvm::cl::desc("<input files>"),
                                                llvm::cl::OneOrMore);
llvm::cl::opt<ImageNormalizationMode> imageMode(
    "image_mode", llvm::cl::desc("Specify the image mode:"), llvm::cl::Required,
    llvm::cl::cat(imageLoaderCat),
    llvm::cl::values(clEnumValN(ImageNormalizationMode::kneg1to1, "neg1to1",
                                "Values are in the range: -1 and 1"),
                     clEnumValN(ImageNormalizationMode::k0to1, "0to1",
                                "Values are in the range: 0 and 1"),
                     clEnumValN(ImageNormalizationMode::k0to256, "0to256",
                                "Values are in the range: 0 and 256"),
                     clEnumValN(ImageNormalizationMode::kneg128to127,
                                "neg128to127",
                                "Values are in the range: -128 .. 127")));
llvm::cl::alias imageModeA("i", llvm::cl::desc("Alias for -image_mode"),
                           llvm::cl::aliasopt(imageMode),
                           llvm::cl::cat(imageLoaderCat));
} // namespace

/// Loads and normalizes all PNGs into a tensor in the NCHW 3x224x224 format.
void loadImagesAndPreprocess(const llvm::cl::list<std::string> &filenames,
                             Tensor *result, ImageNormalizationMode normMode) {
  assert(!filenames.empty() &&
         "There must be at least one filename in filenames.");
  auto range = normModeToRange(normMode);
  unsigned numImages = filenames.size();

  // Get first image's dimensions and check if grayscale or color.
  size_t imgHeight, imgWidth;
  bool isGray;
  std::tie(imgHeight, imgWidth, isGray) = getPngInfo(filenames[0].c_str());
  const size_t numChannels = isGray ? 1 : 3;

  // N x C x H x W
  result->reset(ElemKind::FloatTy,
                {numImages, numChannels, imgHeight, imgWidth});
  auto RH = result->getHandle<>();
  // We iterate over all the png files, reading them all into our result tensor
  // for processing
  for (unsigned n = 0; n < filenames.size(); n++) {
    Tensor localCopy;
    bool loadSuccess = !readPngImage(&localCopy, filenames[n].c_str(), range);
    GLOW_ASSERT(loadSuccess && "Error reading input image.");
    auto imageH = localCopy.getHandle<>();

    auto dims = localCopy.dims();
    assert((dims[0] == imgHeight && dims[1] == imgWidth) &&
           "All images must have the same Height and Width");
    assert(dims[2] == numChannels &&
           "All images must have the same number of channels");

    // Convert to BGR, as this is what imagenet models are expecting.
    for (unsigned z = 0; z < numChannels; z++) {
      for (unsigned y = 0; y < dims[1]; y++) {
        for (unsigned x = 0; x < dims[0]; x++) {
          RH.at({n, numChannels - 1 - z, x, y}) = (imageH.at({x, y, z}));
        }
      }
    }
  }
}

int main(int argc, char **argv) {
  // The loader verifies/initializes command line parameters, and initializes
  // the ExecutionEngine and Function.
  Loader loader(argc, argv);

  // Load and process the image data into the data Tensor.
  Tensor data;
  loadImagesAndPreprocess(inputImageFilenames, &data, imageMode);

  // All of our ONNX and Caffe2 models may use one of two names for input.
  constexpr char const *c2Inputs[2] = {"gpu_0/data", "data"};
  constexpr char const *onnxInputs[2] = {"gpu_0/data_0", "data_0"};

  // Create the model based on the input model format.
  std::unique_ptr<ProtobufLoader> LD;
  bool c2Model = !loader.getCaffe2NetDescFilename().empty();
  if (c2Model) {
    LD.reset(new caffe2ModelLoader(
        loader.getCaffe2NetDescFilename(), loader.getCaffe2NetWeightFilename(),
        c2Inputs, {&data, &data}, *loader.getFunction()));
  } else {
    LD.reset(new ONNXModelLoader(loader.getOnnxModelFilename(), onnxInputs,
                                 {&data, &data}, *loader.getFunction()));
  }
  // Get the Variable that the final expected Softmax writes into at the end of
  // image inference.
  Variable *SMVar = LD->getSingleOutput();

  // Create Variables for both possible input names for flexibility for the
  // input model. The input data is mapped to both names. Whichever Variable is
  // unused will be removed in compile().
  Variable *i0 = LD->getVariableByName(c2Model ? c2Inputs[0] : onnxInputs[0]);
  Variable *i1 = LD->getVariableByName(c2Model ? c2Inputs[1] : onnxInputs[1]);

  assert(i0->getVisibilityKind() == VisibilityKind::Public);
  assert(i1->getVisibilityKind() == VisibilityKind::Public);

  // Compile the model, and perform quantization/emit a bundle/dump debug info
  // if requested from command line.
  loader.compile();

  // If in bundle mode, do not run inference.
  if (!emittingBundle()) {
    loader.runInference({i0, i1}, {&data, &data});

    // Print out the inferred image classification.
    Tensor &res = SMVar->getPayload();
    auto H = res.getHandle<>();
    llvm::outs() << "Model: " << loader.getFunction()->getName() << "\n";
    for (unsigned i = 0; i < inputImageFilenames.size(); i++) {
      Tensor slice = H.extractSlice(i);
      auto SH = slice.getHandle<>();
      llvm::outs() << " File: " << inputImageFilenames[i]
                   << " Result:" << SH.minMaxArg().second << "\n";
    }
  }

  return 0;
}
