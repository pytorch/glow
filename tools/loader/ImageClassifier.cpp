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
#include "glow/Graph/Nodes.h"
#include "glow/Importer/Caffe2ModelLoader.h"
#include "glow/Importer/ONNXModelLoader.h"

#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <queue>

using namespace glow;

enum class ImageNormalizationMode {
  kneg1to1,     // Values are in the range: -1 and 1.
  k0to1,        // Values are in the range: 0 and 1.
  k0to256,      // Values are in the range: 0 and 256.
  kneg128to127, // Values are in the range: -128 .. 127
};

enum class ImageLayout {
  NCHW,
  NHWC,
};

enum class ImageChannelOrder {
  BGR,
  RGB,
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
llvm::cl::opt<ImageNormalizationMode> imageNormMode(
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
llvm::cl::alias imageNormModeA("i", llvm::cl::desc("Alias for -image_mode"),
                               llvm::cl::aliasopt(imageNormMode),
                               llvm::cl::cat(imageLoaderCat));

llvm::cl::opt<ImageChannelOrder> imageChannelOrder(
    "image_channel_order", llvm::cl::desc("Specify the image channel order"),
    llvm::cl::Optional, llvm::cl::cat(imageLoaderCat),
    llvm::cl::values(clEnumValN(ImageChannelOrder::BGR, "BGR", "Use BGR"),
                     clEnumValN(ImageChannelOrder::RGB, "RGB", "Use RGB")),
    llvm::cl::init(ImageChannelOrder::BGR));
llvm::cl::opt<ImageLayout>
    imageLayout("image_layout",
                llvm::cl::desc("Specify which image layout to use"),
                llvm::cl::Optional, llvm::cl::cat(imageLoaderCat),
                llvm::cl::values(clEnumValN(ImageLayout::NCHW, "NCHW",
                                            "Use NCHW image layout"),
                                 clEnumValN(ImageLayout::NHWC, "NHWC",
                                            "Use NHWC image layout")),
                llvm::cl::init(ImageLayout::NCHW));
llvm::cl::alias imageLayoutA("l", llvm::cl::desc("Alias for -image_layout"),
                             llvm::cl::aliasopt(imageLayout),
                             llvm::cl::cat(imageLoaderCat));

llvm::cl::opt<unsigned> labelOffset(
    "label_offset",
    llvm::cl::desc("Label offset for TF ONNX models with 1001 classes"),
    llvm::cl::Optional, llvm::cl::init(0), llvm::cl::cat(imageLoaderCat));

llvm::cl::opt<unsigned> topKCount(
    "topk", llvm::cl::desc("Number of highest likelihood labels to print"),
    llvm::cl::Optional, llvm::cl::init(1), llvm::cl::cat(imageLoaderCat));

llvm::cl::opt<std::string> modelInputName(
    "model_input_name",
    llvm::cl::desc("The name of the variable for the model's input image."),
    llvm::cl::value_desc("string_name"), llvm::cl::Required,
    llvm::cl::cat(imageLoaderCat));
} // namespace

/// Loads and normalizes all PNGs into a tensor in the NHWC format with the
/// requested channel ordering.
void loadImagesAndPreprocess(const llvm::cl::list<std::string> &filenames,
                             Tensor *result) {
  assert(!filenames.empty() &&
         "There must be at least one filename in filenames.");
  auto range = normModeToRange(imageNormMode);
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
    // PNG images are loaded as NHWC & RGB
    bool loadSuccess = !readPngImage(&localCopy, filenames[n].c_str(), range);
    GLOW_ASSERT(loadSuccess && "Error reading input image.");
    auto imageH = localCopy.getHandle<>();

    auto dims = localCopy.dims();
    assert((dims[0] == imgHeight && dims[1] == imgWidth) &&
           "All images must have the same Height and Width");
    assert(dims[2] == numChannels &&
           "All images must have the same number of channels");
    assert((imageChannelOrder == ImageChannelOrder::BGR ||
            imageChannelOrder == ImageChannelOrder::RGB) &&
           "Invalid image format");

    // Convert to NCHW with the requested channel ordering.
    for (unsigned z = 0; z < numChannels; z++) {
      for (unsigned y = 0; y < dims[1]; y++) {
        for (unsigned x = 0; x < dims[0]; x++) {
          if (imageChannelOrder == ImageChannelOrder::BGR) {
            RH.at({n, numChannels - 1 - z, x, y}) = (imageH.at({x, y, z}));
          } else { // RGB
            RH.at({n, z, x, y}) = (imageH.at({x, y, z}));
          }
        }
      }
    }
  }
}

/// A pair representing a float and the index where the float was found.
using FloatIndexPair = std::pair<float, size_t>;

/// Given a Handle \p H of a 1D tensor with float elements, \returns the top K
/// (topKCount) [float, index] pairs, i.e. the pairs with the highest floats.
static std::vector<FloatIndexPair> getTopKPairs(Handle<float> H) {
  assert(topKCount <= H.size() && "Function requires k < number of labels.");
  assert(H.dims().size() == 1 && "H must be a Handle of a 1d Tensor.");

  // Use a priority queue of pairs of floats (probabilities) to size_t (indices)
  // to determine the top K pairs, and then return the indices from it.
  std::priority_queue<FloatIndexPair, std::vector<FloatIndexPair>,
                      std::greater<FloatIndexPair>>
      topKQueue;

  // Loop over all the probabilites, finding the highest k probability pairs.
  for (size_t i = 0, e = H.size(); i < e; i++) {
    float currProbability = H.at({i});
    if (topKQueue.size() < topKCount) {
      // Always push the first k elements.
      topKQueue.push(std::make_pair(currProbability, i));
    } else if (topKQueue.top().first < currProbability) {
      // If the lowest element has lower probability than the current, then pop
      // the lowest and insert the current pair.
      topKQueue.pop();
      topKQueue.push(std::make_pair(currProbability, i));
    }
  }

  // We now have the top K pairs in reverse order.
  std::vector<FloatIndexPair> res(topKCount);
  for (size_t i = 0; i < topKCount; i++) {
    res[topKCount - i - 1] = topKQueue.top();
    topKQueue.pop();
  }

  return res;
}

/// Print out the top K pairs to stdout, which were passed in via \p topKPairs.
static void printTopKPairs(const std::vector<FloatIndexPair> &topKPairs) {
  for (size_t i = 0; i < topKPairs.size(); i++) {
    // Some models are trained with more classes. E.g. Some imagenet models
    // exported from TensorFlow have 1 extra "neutral" class.
    const size_t label = topKPairs[i].second - labelOffset;
    // Tab out the label so it aligns nicely with Label-K1.
    if (i != 0) {
      llvm::outs() << "\t\t\t\t\t";
    }
    llvm::outs() << "\tLabel-K" << i + 1 << ": " << label << " (probability: "
                 << llvm::format("%0.4f", topKPairs[i].first) << ")\n";
  }
}

int main(int argc, char **argv) {
  Context ctx;
  // The loader verifies/initializes command line parameters, and initializes
  // the ExecutionEngine and Function.
  Loader loader(argc, argv);

  // Load and process the image data into the data Tensor.
  Tensor data;
  loadImagesAndPreprocess(inputImageFilenames, &data);

  // For ONNX graphs with input in NHWC layout, we transpose the data.
  switch (imageLayout) {
  case ImageLayout::NCHW:
    break;
  case ImageLayout::NHWC:
    Tensor dataNHWC;
    data.transpose(&dataNHWC, NCHW2NHWC);
    data = std::move(dataNHWC);
    break;
  }

  // The image name that the model expects must be passed on the command line.
  const char *inputName = modelInputName.c_str();

  // Create the model based on the input model format.
  std::unique_ptr<ProtobufLoader> LD;
  bool c2Model = !loader.getCaffe2NetDescFilename().empty();
  if (c2Model) {
    LD.reset(new Caffe2ModelLoader(
        loader.getCaffe2NetDescFilename(), loader.getCaffe2NetWeightFilename(),
        {inputName}, {&data.getType()}, *loader.getFunction()));
  } else {
    LD.reset(new ONNXModelLoader(loader.getOnnxModelFilename(), {inputName},
                                 {&data.getType()}, *loader.getFunction()));
  }

  // Allocate tensors to back all inputs and outputs.
  ctx.allocate(loader.getModule()->getPlaceholders());

  // Get the Variable that the final expected Softmax writes into at the end of
  // image inference.
  Placeholder *SMVar = LD->getSingleOutput();
  Tensor *SMVarT = ctx.get(SMVar);

  // Create Variables for both possible input names for flexibility for the
  // input model. The input data is mapped to both names. Whichever Variable is
  // unused will be removed in compile().
  auto *inputImage = llvm::cast<Placeholder>(LD->getNodeValueByName(inputName));

  // Compile the model, and perform quantization/emit a bundle/dump debug info
  // if requested from command line.
  loader.compile(ctx);

  // If in bundle mode, do not run inference.
  if (!emittingBundle()) {

    // Update the inputs.
    updateVariables(ctx, {inputImage}, {&data});

    // Perform the inference execution.
    loader.runInference(ctx);

    // Print out the inferred image classification.
    auto H = SMVarT->getHandle<>();
    llvm::outs() << "Model: " << loader.getFunction()->getName() << "\n";
    for (unsigned i = 0; i < inputImageFilenames.size(); i++) {
      Tensor slice = H.extractSlice(i);
      auto SH = slice.getHandle<>();
      llvm::outs() << " File: " << inputImageFilenames[i];

      auto topKPairs = getTopKPairs(SH);
      printTopKPairs(topKPairs);
    }
  }

  return 0;
}
