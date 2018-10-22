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
#include "glow/Converter/TypeAToTypeBFunctionConverter.h"
#include "glow/Graph/Nodes.h"
#include "glow/Importer/Caffe2ModelLoader.h"
#include "glow/Importer/ONNXModelLoader.h"

#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

#include <iostream>
#include <memory>
#include <queue>
#include <sstream>

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
llvm::cl::list<std::string> inputImageFilenames(
    llvm::cl::Positional,
    llvm::cl::desc("<input files> (note: specifying '-' enables streaming "
                   "mode, where the model is compiled once and then can be run "
                   "many times with new input filenames passed via stdin)"),
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

llvm::cl::opt<bool> convertInAndOutToFp16(
    "convert-inout-to-fp16",
    llvm::cl::desc(
        "Convert the input and output tensors of the network to fp16"),
    llvm::cl::cat(imageLoaderCat));
} // namespace

/// Loads and normalizes all PNGs into a tensor in the NHWC format with the
/// requested channel ordering.
void loadImagesAndPreprocess(const llvm::cl::list<std::string> &filenames,
                             Tensor *inputImageData) {
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
  inputImageData->reset(ElemKind::FloatTy,
                        {numImages, numChannels, imgHeight, imgWidth});
  auto IIDH = inputImageData->getHandle<>();

  // We iterate over all the png files, reading them all into our inputImageData
  // tensor for processing
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
            IIDH.at({n, numChannels - 1 - z, x, y}) = (imageH.at({x, y, z}));
          } else { // RGB
            IIDH.at({n, z, x, y}) = (imageH.at({x, y, z}));
          }
        }
      }
    }
  }

  // For ONNX graphs with input in NHWC layout, we transpose the image data.
  switch (imageLayout) {
  case ImageLayout::NCHW:
    break;
  case ImageLayout::NHWC:
    Tensor dataNHWC;
    inputImageData->transpose(&dataNHWC, NCHW2NHWC);
    *inputImageData = std::move(dataNHWC);
    break;
  }

  // Convert the raw input to fp16.
  if (convertInAndOutToFp16) {
    inputImageData->convertToType(ElemKind::Float16Ty);
  }
}

/// Write a prompt to stdout asking for filenames for classification. Read in
/// those filenames and add them to \p filenames. \p filenames is cleared before
/// adding the new set of filenames from stdin. \returns false if the passed in
/// line was empty.
static bool getNextImageFilenames(std::vector<std::string> *filenames) {
  // Clear out old filenames before adding new ones.
  filenames->clear();

  llvm::outs() << "Enter image filenames to classify: ";

  // Add in each filename to the vector.
  std::string filenamesRaw;
  getline(std::cin, filenamesRaw);
  std::istringstream iss(filenamesRaw);
  std::string filename;
  while (iss >> filename) {
    filenames->push_back(filename);
  }

  return !filenames->empty();
}

/// Creates and \returns the ProtobufLoader given \p loader and the
/// \p inputImageType. Note that this must come after loading images for
/// inference so that \p inputImageType is known.
static std::unique_ptr<ProtobufLoader>
createProtofbufLoader(Loader &loader, TypeRef inputImageType) {
  // The image name that the model expects must be passed on the command line.
  const char *inputName = modelInputName.c_str();

  // Create the model based on the input model format.
  std::unique_ptr<ProtobufLoader> LD;
  bool c2Model = !loader.getCaffe2NetDescFilename().empty();
  if (c2Model) {
    LD.reset(new Caffe2ModelLoader(
        loader.getCaffe2NetDescFilename(), loader.getCaffe2NetWeightFilename(),
        {inputName}, {inputImageType}, *loader.getFunction()));
  } else {
    LD.reset(new ONNXModelLoader(loader.getOnnxModelFilename(), {inputName},
                                 {inputImageType}, *loader.getFunction()));
  }

  return LD;
}

/// Given \p loader, the \p ctx, and \p inputImageType, build the graph from the
/// provided protobuf file found via \p loader. Then compiles and \returns a
/// pair of pointers to the input Placeholder and output Tensor for the Softmax.
static std::pair<Placeholder *, Tensor *>
buildAndCompileAndGetInAndOutPair(Loader &loader, Context &ctx,
                                  TypeRef inputImageType) {
  auto LD = createProtofbufLoader(loader, inputImageType);

  // Allocate tensors to back all inputs and outputs.
  ctx.allocate(loader.getModule()->getPlaceholders());

  // Convert the placeholders for now. The backing Tensor's data will be
  // converted later.
  if (convertInAndOutToFp16) {
    TypeAToTypeBFunctionConverter converter(
        *loader.getFunction(), ElemKind::FloatTy, ElemKind::Float16Ty);
    for (auto *placeholder : loader.getModule()->getPlaceholders()) {
      converter.convertPlaceholder(*placeholder, &ctx);
    }
  }

  // Compile the model, and perform quantization/emit a bundle/dump debug info
  // if requested from command line.
  loader.compile(ctx);

  // The image name that the model expects must be passed on the command line.
  const char *inputName = modelInputName.c_str();
  Placeholder *inputImagePH =
      llvm::cast<Placeholder>(LD->getNodeValueByName(inputName));

  // Get the Tensor from the Placeholder that the final expected Softmax writes
  // into at the end of image inference.
  Placeholder *SMVarPH = LD->getSingleOutput();
  Tensor *SMVarT = ctx.get(SMVarPH);

  return std::make_pair(inputImagePH, SMVarT);
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

/// Given the output Softmax Tensor \p SMTarT and \p functionName, print the
/// results of inference.
static void processAndPrintResults(Tensor *SMVarT,
                                   llvm::StringRef functionName) {
  if (convertInAndOutToFp16) {
    // SMVarT contains the output of the network in FP16. Convert it back to
    // FP32 so that we don't have to special case the printing of the result
    // for FP16.
    SMVarT->convertToType(ElemKind::FloatTy);
  }

  // Print out the inferred image classification.
  auto H = SMVarT->getHandle<>();
  llvm::outs() << "Model: " << functionName << "\n";
  for (unsigned i = 0; i < inputImageFilenames.size(); i++) {
    Tensor slice = H.extractSlice(i);
    auto SH = slice.getHandle<>();
    llvm::outs() << " File: " << inputImageFilenames[i];

    auto topKPairs = getTopKPairs(SH);
    printTopKPairs(topKPairs);
  }
}

int main(int argc, char **argv) {
  Context ctx;
  // The loader verifies/initializes command line parameters, and initializes
  // the ExecutionEngine and Function.
  Loader loader(argc, argv);

  const bool streamInputFilenamesMode =
      inputImageFilenames.size() == 1 && inputImageFilenames.front() == "-";

  GLOW_ASSERT(!(streamInputFilenamesMode && emittingBundle()) &&
              "Cannot emit a bundle and also stream inputs.");

  // Used to make sure we only compile once, and run only once if not streaming.
  bool isFirstRun = true;

  // These will be set during the first run.
  Placeholder *inputImagePH = nullptr;
  Tensor *SMVarT = nullptr;

  Tensor inputImageData;
  while ((streamInputFilenamesMode &&
          getNextImageFilenames(&inputImageFilenames)) ||
         isFirstRun) {
    // Load and process the image data into the inputImageData Tensor.
    loadImagesAndPreprocess(inputImageFilenames, &inputImageData);

    // If this is the first run, then we need to build and compile the model.
    if (isFirstRun) {
      isFirstRun = false;

      // Build and compile the graph, and then get back the input Placeholder
      // and output Softmax Tensor.
      std::pair<Placeholder *, Tensor *> inputOutputPair =
          buildAndCompileAndGetInAndOutPair(loader, ctx,
                                            &inputImageData.getType());

      // If in bundle mode, the bundle has been saved by the above call, so we
      // can safely return.
      if (emittingBundle()) {
        return 0;
      }

      inputImagePH = inputOutputPair.first;
      SMVarT = inputOutputPair.second;
    }
    assert(inputImagePH && SMVarT && "Input and output must be valid.");
    GLOW_ASSERT(inputImagePH->dims() == inputImageData.dims() &&
                "New input shape does not match the compiled function.");

    // About to run inference, so update the input image Placeholder's backing
    // Tensor with inputImageData.
    updateInputPlaceholders(ctx, {inputImagePH}, {&inputImageData});

    // Perform the inference execution, updating SMVarT.
    loader.runInference(ctx);

    // Print the top-k results from the output Softmax tensor.
    processAndPrintResults(SMVarT, loader.getFunction()->getName());
  }

  return 0;
}
