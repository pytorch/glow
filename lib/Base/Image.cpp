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

#include "glow/Base/Image.h"
#include "glow/Base/Tensor.h"
#include "glow/Support/Support.h"

#include "llvm/Support/CommandLine.h"

#ifndef WITH_PNG
#error "Using Glow's PNG library requires installing libpng"
#endif

using namespace glow;

#include <png.h>

namespace glow {

llvm::cl::OptionCategory imageCat("Image Processing Options");

std::vector<ImageNormalizationMode> imageNormMode;
static llvm::cl::list<ImageNormalizationMode,
                      std::vector<ImageNormalizationMode>>
    imageNormModeF(
        "image-mode", llvm::cl::CommaSeparated,
        llvm::cl::desc("Specify the image mode:"), llvm::cl::cat(imageCat),
        llvm::cl::location(imageNormMode),
        llvm::cl::values(clEnumValN(ImageNormalizationMode::kneg1to1, "neg1to1",
                                    "Values are in the range: -1 and 1"),
                         clEnumValN(ImageNormalizationMode::k0to1, "0to1",
                                    "Values are in the range: 0 and 1"),
                         clEnumValN(ImageNormalizationMode::k0to255, "0to255",
                                    "Values are in the range: 0 and 255"),
                         clEnumValN(ImageNormalizationMode::kneg128to127,
                                    "neg128to127",
                                    "Values are in the range: -128 .. 127")));
static llvm::cl::alias imageNormModeA("i",
                                      llvm::cl::desc("Alias for -image-mode"),
                                      llvm::cl::aliasopt(imageNormModeF),
                                      llvm::cl::cat(imageCat));

std::vector<ImageChannelOrder> imageChannelOrderOpt;
static llvm::cl::list<ImageChannelOrder, std::vector<ImageChannelOrder>>
    imageChannelOrderF(
        "image-channel-order", llvm::cl::CommaSeparated,
        llvm::cl::desc("Specify the image channel order"), llvm::cl::ZeroOrMore,
        llvm::cl::cat(imageCat), llvm::cl::location(imageChannelOrderOpt),
        llvm::cl::values(clEnumValN(ImageChannelOrder::BGR, "BGR", "Use BGR"),
                         clEnumValN(ImageChannelOrder::RGB, "RGB", "Use RGB")));

std::vector<ImageLayout> imageLayoutOpt;
static llvm::cl::list<ImageLayout, std::vector<ImageLayout>> imageLayoutOptF(
    "image-layout", llvm::cl::ZeroOrMore, llvm::cl::CommaSeparated,
    llvm::cl::location(imageLayoutOpt), llvm::cl::desc(".\n"),
    llvm::cl::values(
        clEnumValN(ImageLayout::Unspecified, "NonImage",
                   "Use NonImage image layout"),
        clEnumValN(ImageLayout::NCHW, "NCHW", "Use NCHW image layout"),
        clEnumValN(ImageLayout::NHWC, "NHWC", "Use NHWC image layout")),
    llvm::cl::cat(imageCat));
static llvm::cl::alias
    imageLayoutOptA("l", llvm::cl::desc("Alias for -image-layout"),
                    llvm::cl::aliasopt(imageLayoutOptF),
                    llvm::cl::cat(imageCat));

std::vector<ImageLayout> inputLayoutOpt;
static llvm::cl::list<ImageLayout, std::vector<ImageLayout>> inputLayoutF1(
    "input-layout", llvm::cl::ZeroOrMore, llvm::cl::CommaSeparated,
    llvm::cl::location(inputLayoutOpt), llvm::cl::desc(".\n"),
    llvm::cl::values(
        clEnumValN(ImageLayout::Unspecified, "AsIs",
                   "Use -image-layout setting for this input."),
        clEnumValN(ImageLayout::NCHW, "NCHW", "Use NCHW image layout"),
        clEnumValN(ImageLayout::NHWC, "NHWC", "Use NHWC image layout")),
    llvm::cl::cat(imageCat));

bool useImagenetNormalization;
static llvm::cl::opt<bool, true> useImagenetNormalizationF(
    "use-imagenet-normalization", llvm::cl::ZeroOrMore,
    llvm::cl::location(useImagenetNormalization),
    llvm::cl::desc("Use Imagenet Normalization. This works in combination "
                   "with the Image Mode normalization."),
    llvm::cl::cat(imageCat));

// LLVM command line made parser subclasing final in 3.7 yet the only cmd
// line manual still refers to the old data. Also, the change was not clear
// why it's made. Assigning callbacks is not possible, and subclassing
// basic_parser is open to future errors. Thus, relying in LLVM parser is
// minimized - we will just obtain strings and process options.

VecVec<float> meanValuesOpt;
static std::string meanValues_;
static llvm::cl::opt<std::string, true> meanValuesF(
    "mean",
    llvm::cl::desc("Mean values m1,m2,m3..."
                   "Count must be equal to number of input channels."
                   "Order of values must match specified image channel order."),
    llvm::cl::location(meanValues_), llvm::cl::value_desc("string"),
    llvm::cl::cat(imageCat));

VecVec<float> stddevValuesOpt;
static std::string stddevValues_;
static llvm::cl::opt<std::string, true> stddevValuesF(
    "stddev",
    llvm::cl::desc("Standard deviation values s1,s2,s3..."
                   "Count must be equal to number of input channels."
                   "Order of values must match specified image channel order."),
    llvm::cl::location(stddevValues_), llvm::cl::value_desc("string"),
    llvm::cl::cat(imageCat));

} // namespace glow

/// Some global options are set from functions that can be called from
/// multiple threads. Lock the access while setting them.
std::mutex globalOpts;

// Process list of lists command line in the following format:
/// All elements in a list are comma separated. Lists are double-colon
/// separated. For example, "-option=1,2,3:4,5,6" defines two lists each with 3
/// elements. Final destination for the processed command line string \p cmdStr
/// is double vector \p outVec.
template <typename T>
static void processListOfListsCmdOption(size_t numInputs, std::string &cmdStr,
                                        VecVec<T> &outVec) {
  std::vector<std::string> strVec;
  std::vector<T> typeVec;
  if (cmdStr.empty()) {
    outVec.resize(numInputs);
    return;
  }
  outVec.clear();
  std::stringstream ss(cmdStr);
  while (ss) {
    T elem;
    char sep;
    ss >> elem >> sep;
    typeVec.push_back(elem);
    if (sep == ':') {
      outVec.push_back(typeVec);
      typeVec.clear();
    } else {
      CHECK_EQ(sep, ',') << "Expected either ',' or ':' as separator";
    }
  }
  if (!typeVec.empty()) {
    outVec.push_back(typeVec);
  }
}

void glow::initImageCmdArgVars() {
  // clear external storage for all the variables.
  globalOpts.lock();
  imageNormMode.clear();
  imageChannelOrderOpt.clear();
  imageLayoutOpt.clear();
  inputLayoutOpt.clear();
  meanValuesOpt.clear();
  meanValues_.clear();
  stddevValuesOpt.clear();
  stddevValues_.clear();
  globalOpts.unlock();
}

/// Processes special command line options for Image module.
void glow::processImageCmdArgVars(size_t numInputs) {
  globalOpts.lock();
  processListOfListsCmdOption(numInputs, meanValues_, meanValuesOpt);
  processListOfListsCmdOption(numInputs, stddevValues_, stddevValuesOpt);

  // Default for image range is U8.
  if (imageNormMode.empty()) {
    for (size_t i = 0, e = numInputs; i < e; i++) {
      imageNormMode.push_back(ImageNormalizationMode::k0to255);
    }
  }
  // Default for image layout is NCHW.
  if (imageLayoutOpt.empty()) {
    for (size_t i = 0, e = numInputs; i < e; i++) {
      imageLayoutOpt.push_back(ImageLayout::NCHW);
    }
  }
  // If input-layout is empty just copy image-layout to it.
  // If input-layout is not empty, and one of the values is "AsIs", copy
  // the corresponding image-layout value to it.
  if (inputLayoutOpt.empty()) {
    inputLayoutOpt = imageLayoutOpt;
  } else {
    CHECK_EQ(inputLayoutOpt.size(), imageLayoutOpt.size())
        << "Expecting the same number of values in -image-layout and "
           "-input-layout";
    for (size_t i = 0, e = inputLayoutOpt.size(); i < e; i++) {
      if (inputLayoutOpt[i] == ImageLayout::Unspecified) {
        inputLayoutOpt[i] = imageLayoutOpt[i];
      }
    }
  }
  // Default for channel order is BGR.
  if (imageChannelOrderOpt.empty()) {
    for (size_t i = 0, e = numInputs; i < e; i++) {
      imageChannelOrderOpt.push_back(ImageChannelOrder::BGR);
    }
  }

  CHECK_EQ(numInputs, imageNormMode.size())
      << "Number of -image-mode values must match number of inputs";
  CHECK_EQ(numInputs, imageLayoutOpt.size())
      << "Number of -image-layout values must match number of inputs";
  CHECK_EQ(numInputs, imageChannelOrderOpt.size())
      << "Number of -image-channel-order values must match number of inputs";
  CHECK_EQ(numInputs, meanValuesOpt.size())
      << "Number of -mean values must match number of inputs";
  CHECK_EQ(numInputs, stddevValuesOpt.size())
      << "Number of -stddev values must match number of inputs";
  CHECK_EQ(numInputs, inputLayoutOpt.size())
      << "Number of -input-mode values must match number of inputs";
  globalOpts.unlock();
}

/// Convert the normalization to numeric floating poing ranges.
std::pair<float, float> glow::normModeToRange(ImageNormalizationMode mode) {
  switch (mode) {
  case ImageNormalizationMode::kneg1to1:
    return {-1., 1.};
  case ImageNormalizationMode::k0to1:
    return {0., 1.0};
  case ImageNormalizationMode::k0to255:
    return {0., 255.0};
  case ImageNormalizationMode::kneg128to127:
    return {-128., 127.};
  default:
    LOG(FATAL) << "Image format not defined.";
  }
  return {0, 0};
}

/// Returns whether string \p hdr is recognized as PNG.
static bool isPngHdrSignature(uint8_t *header) {
  return png_sig_cmp(header, 0, 8) == 0;
}

/// Returns whether file \p filename is in png format.
bool glow::isPngFormat(const std::string &filename) {
  // open file and test for it being a png.
  FILE *fp = fopen(filename.c_str(), "rb");
  CHECK(fp) << "Can't open image file with name: " << filename;

  unsigned char header[8];
  size_t fread_ret = fread(header, 1, 8, fp);
  fclose(fp);
  CHECK_EQ(fread_ret, 8) << "fread failed for file: " << filename;
  return isPngHdrSignature(header);
}

std::tuple<size_t, size_t, bool> glow::getPngInfo(const char *filename) {
  // open file and test for it being a png.
  FILE *fp = fopen(filename, "rb");
  CHECK(fp) << "Can't open image file with name: " << filename;

  unsigned char header[8];
  size_t fread_ret = fread(header, 1, 8, fp);
  CHECK_EQ(fread_ret, 8) << "fread failed for file: " << filename;
  CHECK(isPngHdrSignature(header)) << "Invalid image file signature.";

  // Initialize stuff.
  png_structp png_ptr =
      png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
  CHECK(png_ptr) << "Image initialization failed.";

  png_infop info_ptr = png_create_info_struct(png_ptr);
  CHECK(info_ptr) << "Could not get png info.";

  int sjmpGetPtr = setjmp(png_jmpbuf(png_ptr));
  CHECK(!sjmpGetPtr) << "Failed getting png_ptr.";

  png_init_io(png_ptr, fp);
  png_set_sig_bytes(png_ptr, 8);
  png_read_info(png_ptr, info_ptr);

  size_t height = png_get_image_height(png_ptr, info_ptr);
  size_t width = png_get_image_width(png_ptr, info_ptr);
  png_byte color_type = png_get_color_type(png_ptr, info_ptr);

  const bool isGray = color_type == PNG_COLOR_TYPE_GRAY;

  png_destroy_read_struct(&png_ptr, &info_ptr, (png_infopp)NULL);
  fclose(fp);

  return std::make_tuple(height, width, isGray);
}

bool glow::readPngImage(Tensor *T, const char *filename,
                        std::pair<float, float> range,
                        llvm::ArrayRef<float> mean,
                        llvm::ArrayRef<float> stddev) {
  unsigned char header[8];
  // open file and test for it being a png.
  FILE *fp = fopen(filename, "rb");
  // Can't open the file.
  if (!fp) {
    return true;
  }

  // Validate signature.
  size_t fread_ret = fread(header, 1, 8, fp);
  if (fread_ret != 8) {
    fclose(fp);
    return true;
  }
  if (png_sig_cmp(header, 0, 8)) {
    fclose(fp);
    return true;
  }

  // Initialize stuff.
  png_structp png_ptr =
      png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
  if (!png_ptr) {
    fclose(fp);
    return true;
  }

  png_infop info_ptr = png_create_info_struct(png_ptr);
  if (!info_ptr) {
    png_destroy_read_struct(&png_ptr, (png_infopp)NULL, (png_infopp)NULL);
    fclose(fp);
    return true;
  }

  if (setjmp(png_jmpbuf(png_ptr))) {
    png_destroy_read_struct(&png_ptr, &info_ptr, (png_infopp)NULL);
    fclose(fp);
    return true;
  }

  png_init_io(png_ptr, fp);
  png_set_sig_bytes(png_ptr, 8);
  png_read_info(png_ptr, info_ptr);

  dim_t width = png_get_image_width(png_ptr, info_ptr);
  dim_t height = png_get_image_height(png_ptr, info_ptr);
  int color_type = png_get_color_type(png_ptr, info_ptr);
  int bit_depth = png_get_bit_depth(png_ptr, info_ptr);

  const bool isGray = color_type == PNG_COLOR_TYPE_GRAY;
  const dim_t numChannels = isGray ? 1 : 3;

  (void)bit_depth;
  DCHECK_EQ(bit_depth, 8) << "Invalid image";
  DCHECK((color_type == PNG_COLOR_TYPE_RGB_ALPHA ||
          color_type == PNG_COLOR_TYPE_RGB || isGray))
      << "Invalid image";
  bool hasAlpha = (color_type == PNG_COLOR_TYPE_RGB_ALPHA);

  int number_of_passes = png_set_interlace_handling(png_ptr);
  (void)number_of_passes;
  DCHECK_EQ(number_of_passes, 1) << "Invalid image";

  png_read_update_info(png_ptr, info_ptr);

  // Error during image read.
  if (setjmp(png_jmpbuf(png_ptr))) {
    png_destroy_read_struct(&png_ptr, &info_ptr, (png_infopp)NULL);
    fclose(fp);
    return true;
  }

  auto *row_pointers = (png_bytep *)malloc(sizeof(png_bytep) * height);
  for (dim_t y = 0; y < height; y++) {
    row_pointers[y] = (png_byte *)malloc(png_get_rowbytes(png_ptr, info_ptr));
  }

  png_read_image(png_ptr, row_pointers);
  png_read_end(png_ptr, info_ptr);

  T->reset(ElemKind::FloatTy, {height, width, numChannels});
  auto H = T->getHandle<>();

  float scale = ((range.second - range.first) / 255.0);
  float bias = range.first;

  for (dim_t row_n = 0; row_n < height; row_n++) {
    png_byte *row = row_pointers[row_n];
    for (dim_t col_n = 0; col_n < width; col_n++) {
      png_byte *ptr =
          &(row[col_n * (hasAlpha ? (numChannels + 1) : numChannels)]);
      for (dim_t i = 0; i < numChannels; i++) {
        float val = float(ptr[i]);
        val = (val - mean[i]) / stddev[i];
        H.at({row_n, col_n, i}) = val * scale + bias;
      }
    }
  }

  for (dim_t y = 0; y < height; y++) {
    free(row_pointers[y]);
  }
  free(row_pointers);
  png_destroy_read_struct(&png_ptr, &info_ptr, (png_infopp)NULL);
  fclose(fp);

  return false;
}

bool glow::writePngImage(Tensor *T, const char *filename,
                         std::pair<float, float> range,
                         llvm::ArrayRef<float> mean,
                         llvm::ArrayRef<float> stddev) {
  /* create file */
  FILE *fp = fopen(filename, "wb");
  if (!fp) {
    return true;
  }

  /* initialize stuff */
  png_structp png_ptr =
      png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);

  if (!png_ptr) {
    return true;
  }

  png_infop info_ptr = png_create_info_struct(png_ptr);
  if (!info_ptr) {
    return true;
  }

  if (setjmp(png_jmpbuf(png_ptr))) {
    return true;
  }

  png_init_io(png_ptr, fp);

  if (setjmp(png_jmpbuf(png_ptr))) {
    return true;
  }

  auto H = T->getHandle<>();

  auto odim = H.dims();
  constexpr size_t numChannels = 3;
  DCHECK_EQ(odim[2], numChannels)
      << "Currently only supports saving RGB images without alpha.";

  size_t width = odim[0];
  size_t height = odim[1];
  int color_type = PNG_COLOR_TYPE_RGB_ALPHA;
  int bit_depth = 8;

  png_set_IHDR(png_ptr, info_ptr, width, height, bit_depth, color_type,
               PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_BASE,
               PNG_FILTER_TYPE_BASE);

  png_write_info(png_ptr, info_ptr);

  if (setjmp(png_jmpbuf(png_ptr))) {
    return true;
  }

  auto *row_pointers = (png_bytep *)malloc(sizeof(png_bytep) * height);
  for (size_t y = 0; y < height; y++) {
    row_pointers[y] = (png_byte *)malloc(png_get_rowbytes(png_ptr, info_ptr));
  }

  float scale = ((range.second - range.first) / 255.0);
  float bias = range.first;

  for (dim_t y = 0; y < height; y++) {
    png_byte *row = row_pointers[y];
    for (dim_t x = 0; x < width; x++) {
      png_byte *ptr = &(row[x * 4]);
      for (dim_t i = 0; i < numChannels; i++) {
        float val = (H.at({y, x, i}) - bias) / scale;
        val = (val * stddev[i]) + mean[i];
        ptr[i] = val;
      }
      ptr[3] = 0xff;
    }
  }

  png_write_image(png_ptr, row_pointers);

  if (setjmp(png_jmpbuf(png_ptr))) {
    return true;
  }

  png_write_end(png_ptr, nullptr);

  /* cleanup heap allocation */
  for (size_t y = 0; y < height; y++) {
    free(row_pointers[y]);
  }
  free(row_pointers);
  png_destroy_write_struct(&png_ptr, &info_ptr);
  fclose(fp);
  return false;
}

Tensor glow::readPngImageAndPreprocess(llvm::StringRef filename,
                                       ImageNormalizationMode imageNormMode,
                                       ImageChannelOrder imageChannelOrder,
                                       ImageLayout imageLayout,
                                       llvm::ArrayRef<float> mean,
                                       llvm::ArrayRef<float> stddev) {
  Tensor imageData;
  readPngImageAndPreprocess(imageData, filename, imageNormMode,
                            imageChannelOrder, imageLayout, mean, stddev);
  return imageData;
}

void glow::readPngImageAndPreprocess(Tensor &imageData,
                                     llvm::StringRef filename,
                                     ImageNormalizationMode imageNormMode,
                                     ImageChannelOrder imageChannelOrder,
                                     ImageLayout imageLayout,
                                     llvm::ArrayRef<float> mean,
                                     llvm::ArrayRef<float> stddev) {

  // PNG images are RGB, so shuffle mean and stddev values to be in RGB order
  // as well, prior applying them to input image.
  std::vector<float> meanRGB(mean);
  std::vector<float> stddevRGB(stddev);
  if (imageChannelOrder == ImageChannelOrder::BGR) {
    std::reverse(meanRGB.begin(), meanRGB.end());
    std::reverse(stddevRGB.begin(), stddevRGB.end());
  }

  auto range = normModeToRange(imageNormMode);
  bool loadSuccess =
      !readPngImage(&imageData, filename.data(), range, meanRGB, stddevRGB);
  CHECK(loadSuccess) "Error reading input image from file: " << filename.str();
  dim_t imgHeight = imageData.dims()[0];
  dim_t imgWidth = imageData.dims()[1];
  dim_t numChannels = imageData.dims()[2];

  // PNG images are NHWC and RGB.  Convert if needed.
  // Convert to requested channel ordering.
  if (imageChannelOrder == ImageChannelOrder::BGR) {
    Tensor swizzled(imageData.getType());
    auto IH = imageData.getHandle();
    auto SH = swizzled.getHandle();
    for (unsigned z = 0; z < numChannels; z++) {
      for (unsigned y = 0; y < imgHeight; y++) {
        for (unsigned x = 0; x < imgWidth; x++) {
          SH.at({y, x, numChannels - 1 - z}) = IH.at({y, x, z});
        }
      }
    }
    imageData = std::move(swizzled);
  }
  // Convert to requested layout.
  if (imageLayout == ImageLayout::NCHW) {
    Tensor transposed;
    imageData.transpose(&transposed, {2u, 0u, 1u});
    imageData = std::move(transposed);
  }
}

/// Entry point for the PNG images loader.
void glow::readPngImagesAndPreprocess(
    Tensor &inputImageData, const llvm::ArrayRef<std::string> &filenames,
    ImageNormalizationMode imageNormMode, ImageChannelOrder imageChannelOrder,
    ImageLayout imageLayout, llvm::ArrayRef<float> meanRef,
    llvm::ArrayRef<float> stddevRef) {
  DCHECK(!filenames.empty())
      << "There must be at least one filename in filenames.";
  DCHECK_EQ((dim_t)filenames.size(), filenames.size());
  dim_t numImages = filenames.size();

  // Get image dimensions and check if grayscale or color.
  dim_t imgHeight;
  dim_t imgWidth;
  bool isGray;
  std::tie(imgHeight, imgWidth, isGray) = getPngInfo(filenames[0].c_str());
  const dim_t numChannels = isGray ? 1 : 3;

  // Assign mean and stddev for input normalization.
  llvm::ArrayRef<float> mean;
  llvm::ArrayRef<float> stddev;
  if (!meanRef.empty()) {
    CHECK_EQ(meanRef.size(), numChannels)
        << "Number of mean values != input channels";
    CHECK(!useImagenetNormalization)
        << "-mean and -use-imagenet-normalization cannot be used together.";
    mean = meanRef;
  } else if (useImagenetNormalization) {
    mean = imagenetNormMean;
  } else {
    mean = zeroMean;
  }

  if (!stddevRef.empty()) {
    CHECK_EQ(stddevRef.size(), numChannels)
        << "Number of stddev values != input channels";
    CHECK(!useImagenetNormalization)
        << "-stddev and -use-imagenet-normalization cannot be used together.";
    stddev = stddevRef;
  } else if (useImagenetNormalization) {
    stddev = imagenetNormStd;
  } else {
    stddev = oneStd;
  }

  // Allocate a tensor for the batch.
  ShapeVector batchDims;
  switch (imageLayout) {
  case ImageLayout::NCHW:
    batchDims = {numImages, numChannels, imgHeight, imgWidth};
    break;
  case ImageLayout::NHWC:
    batchDims = {numImages, imgHeight, imgWidth, numChannels};
    break;
  default:
    LOG(FATAL) << "Unexpected layout\n";
  }
  inputImageData.reset(ElemKind::FloatTy, batchDims);
  auto IIDH = inputImageData.getHandle<>();

  // Read images into local tensors and add to batch.
  for (size_t n = 0; n < filenames.size(); n++) {
    Tensor localCopy;
    readPngImageAndPreprocess(localCopy, filenames[n], imageNormMode,
                              imageChannelOrder, imageLayout, mean, stddev);
    DCHECK(std::equal(localCopy.dims().begin(), localCopy.dims().end(),
                      inputImageData.dims().begin() + 1))
        << "All images must have the same dimensions";
    IIDH.insertSlice(localCopy, n);
  }
}

/// Dispatching loading to the format handlers.
void glow::loadImagesAndPreprocess(
    VecVecRef<std::string> filenamesList,
    llvm::ArrayRef<Tensor *> inputImageDataList,
    llvm::ArrayRef<ImageNormalizationMode> normMode,
    llvm::ArrayRef<ImageChannelOrder> channelOrder,
    llvm::ArrayRef<ImageLayout> imageLayout,
    llvm::ArrayRef<ImageLayout> inputLayout, VecVecRef<float> mean,
    VecVecRef<float> stddev) {

  globalOpts.lock();
  if (normMode.size()) {
    imageNormMode = normMode;
  }
  if (channelOrder.size()) {
    imageChannelOrderOpt = channelOrder;
  }
  if (imageLayout.size()) {
    imageLayoutOpt = imageLayout;
  }
  if (inputLayout.size()) {
    inputLayoutOpt = inputLayout;
  }
  if (stddev.size()) {
    stddevValuesOpt = stddev;
  }
  if (mean.size()) {
    meanValuesOpt = mean;
  }
  globalOpts.unlock();

  CHECK(!filenamesList.empty())
      << "There must be at least one list in filenames.";

  CHECK_EQ(filenamesList.size(), inputImageDataList.size())
      << "Number of image and tensor lists must match.";

  processImageCmdArgVars(inputImageDataList.size());

  for (size_t i = 0; i < filenamesList.size(); i++) {
    // Get list of files for an input.
    auto filenames = filenamesList[i];

    // Get tensor to load for that one selected input.
    auto inputImageData = inputImageDataList[i];
    // All files for an input must be of the same type, thus will just check
    // the first one.
    if (isPngFormat(filenames[0])) {
      readPngImagesAndPreprocess(*inputImageData, filenames, imageNormMode[i],
                                 imageChannelOrderOpt[i], imageLayoutOpt[i],
                                 meanValuesOpt[i], stddevValuesOpt[i]);
    } else if (isNumpyNpyFormat(filenames[0])) {
      loadNumpyImagesAndPreprocess(filenames, *inputImageData, imageNormMode[i],
                                   imageLayoutOpt[i], inputLayoutOpt[i],
                                   meanValuesOpt[i], stddevValuesOpt[i]);
    } else {
      LOG(FATAL) << "Input file format is not recognized: \n" << filenames[0];
    }
  }
}
