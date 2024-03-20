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

#include <fstream>
#include <map>
#include <sstream>
#include <type_traits>

#define NCHW2NHWC                                                              \
  { 0u, 2u, 3u, 1u }
#define NHWC2NCHW                                                              \
  { 0u, 3u, 1u, 2u }

using namespace glow;

// Check if the header starts with numpy magic string.
bool checkNumpyMagicHdr(uint8_t *header) {
  uint8_t MAGIC[] = {0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59};
  return !memcmp((uint8_t *)header, MAGIC, 6);
}

enum class NpyType { I1, U1, I2, U2, I4, U4, I8, U8, F2, F4, F8 };

struct NpyData {
  std::vector<char> data;
  std::vector<dim_t> shape;
  NpyType type;
  size_t nvals;
  template <typename T> T *getData() { return (T *)data.data(); }
  size_t elemSize;
};

template <typename T>
static void convertNumpyToFloatImpl(NpyData &dataNpy,
                                    std::vector<float> &data) {
  T *databuf = dataNpy.getData<T>();
  for (size_t i = 0; i < dataNpy.nvals; i++) {
    data[i] = databuf[i];
  }
}

void convertNumpyToFloat(NpyData &dataNpy, std::vector<float> &data) {
  if (dataNpy.type == NpyType::F4) {
    convertNumpyToFloatImpl<float>(dataNpy, data);
  } else if (dataNpy.type == NpyType::U1) {
    convertNumpyToFloatImpl<uint8_t>(dataNpy, data);
  } else if (dataNpy.type == NpyType::I1) {
    convertNumpyToFloatImpl<int8_t>(dataNpy, data);
  } else if (dataNpy.type == NpyType::I2) {
    convertNumpyToFloatImpl<int16_t>(dataNpy, data);
  } else if (dataNpy.type == NpyType::U2) {
    convertNumpyToFloatImpl<uint16_t>(dataNpy, data);
  } else {
    LOG(FATAL) << " Datatype not supported: " << (int)dataNpy.type;
  }
}

/// Read npy file \p filename and store tensor info into \p npyData.
void numpyReader(const std::string &filename, NpyData &npyData) {

  std::map<std::string, NpyType> npyType = {
      {"i1", NpyType::I1}, {"u1", NpyType::U1}, {"i2", NpyType::I2},
      {"u2", NpyType::U2}, {"i4", NpyType::I4}, {"u4", NpyType::U4},
      {"i8", NpyType::I8}, {"u8", NpyType::U8}, {"f2", NpyType::F2},
      {"f4", NpyType::F4}, {"f8", NpyType::F8}};

  // Wrapper for string find() method, that checks for return value.
  auto findStr = [](std::string &str, const char *name, size_t *loc = nullptr) {
    size_t newloc = loc ? str.find(name, *loc) : str.find(name);
    if (newloc == std::string::npos) {
      LOG(FATAL) << "NPY loader: Couldn't find string " << name;
    }
    return newloc;
  };

  // Wrapper for string findFirstOf method, that checks for return value.
  auto findFirstOf = [](std::string &str, const char *find, size_t loc) {
    loc = str.find_first_of(find, loc);
    if (loc == std::string::npos) {
      LOG(FATAL) << "NPY loader: Couldn't find string " << find;
    }
    return loc;
  };

  // temp vars.
  size_t loc, locend;

  // get file content
  std::ifstream fs(filename.data(), std::ifstream::binary);
  CHECK(fs) << "NPY loader: Couldn't open file: " << filename << "\n";
  fs.seekg(0, fs.end);
  int len = fs.tellg();
  CHECK(len > 0) << "NPY loader: File too short: " << filename;
  fs.seekg(0, fs.beg);
  char *buffer = new char[len];
  fs.read(buffer, len);
  CHECK(fs) << "NPY loader: Reading file failed: " << filename;
  fs.close();

  // verify that the header contains numpy "magic" string.
  if (!checkNumpyMagicHdr((uint8_t *)buffer)) {
    LOG(FATAL) << "NPY loader: Magic number not found";
  }

  // version
  uint8_t version = buffer[6];
  if (version != 1) {
    LOG(FATAL) << "NPY loader: Version is invalid:" << version;
  }

  // get header.
  uint32_t hdrlenLen = version > 1 ? 4 : 2;
  uint32_t hdrlen =
      (hdrlenLen == 2) ? *(uint16_t *)&buffer[8] : *(uint32_t *)&buffer[8];
  std::string hdr(&buffer[8 + hdrlenLen], hdrlen);

  // Find type: Search for little endian identifers, until the end of the value.
  // Type string is 2 chars following endianess identifer.
  loc = findStr(hdr, "descr");
  loc = findStr(hdr, ":", &loc);
  loc = findFirstOf(hdr, "<|,", loc);
  if (hdr[loc] != '<' && hdr[loc] != '|') {
    LOG(FATAL) << "NPY loader: Little-endian supported only got: " << hdr[loc];
  }
  std::string typestr = hdr.substr(loc + 1, 2);
  CHECK(npyType.count(typestr))
      << "NPY loader: Unknown type: " << typestr << "\n";

  npyData.type = npyType[typestr];

  npyData.elemSize = strtol(&typestr[1], nullptr, 10);
  CHECK(npyData.elemSize > 0 && npyData.elemSize <= 8)
      << "NPY loader: Element size wrong: " << npyData.elemSize;

  // find fortran_order : [True | False]
  loc = findStr(hdr, "fortran_order");
  loc = findStr(hdr, ":", &loc);
  loc = hdr.find_first_not_of(" '", loc + 1);
  locend = findFirstOf(hdr, "', ", loc + 1);
  std::string order = hdr.substr(loc, locend - loc);
  CHECK_EQ(order, "False") << "NPY loader: fortran_order must be False";

  // find shape
  loc = findStr(hdr, "shape");
  loc = findStr(hdr, "(", &loc);
  locend = findStr(hdr, ")", &loc);
  std::string shapestr = hdr.substr(loc + 1, locend - loc - 1);
  std::stringstream ss(shapestr);
  std::string token;
  npyData.nvals = 1;
  while (std::getline(ss, token, ',')) {
    size_t val = strtol(&token[0], nullptr, 10);
    CHECK(val > 0) << "NPY loader: Element size wrong: " << val;
    npyData.shape.push_back(val);
    npyData.nvals *= val;
  }

  // move file ptr to data
  int data_pos = 8 + hdrlen + hdrlenLen;
  char *databuf = &buffer[data_pos];

  npyData.data.resize(npyData.nvals * npyData.elemSize);
  memcpy(npyData.data.data(), databuf, npyData.nvals * npyData.elemSize);
  delete[] buffer;
}

static void normalizeData(ImageLayout imageLayout, llvm::ArrayRef<float> mean,
                          llvm::ArrayRef<float> stddev,
                          ImageNormalizationMode imageNormMode,
                          std::vector<float> &data, std::vector<dim_t> &shape,
                          ImgDataRange pixelRange) {
  auto inputRange =
      glow::getPixelValMax(pixelRange) - glow::getPixelValMin(pixelRange);
  auto range = normModeToRange(imageNormMode, pixelRange);

  float scale = (range.second - range.first) / inputRange;
  float bias = range.first;
  float offset = glow::getPixelValMin(pixelRange);

  dim_t numCh = (imageLayout == ImageLayout::Unspecified) ? 1
                : (imageLayout == ImageLayout::NHWC)      ? shape[3]
                                                          : shape[1];
  std::vector<float> zeroMean(numCh, 0.f);
  std::vector<float> oneStd(numCh, 1.f);
  std::vector<float> meanVal(mean.size() ? mean : llvm::makeArrayRef(zeroMean));
  std::vector<float> stddevVal(stddev.size() ? stddev
                                             : llvm::makeArrayRef(oneStd));
  CHECK_EQ(numCh, meanVal.size()) << "NPY loader: mean argument size should "
                                     "match the number of channels: "
                                  << numCh;
  CHECK_EQ(numCh, stddevVal.size()) << "NPY loader: stddev argument size "
                                       "should match the number of channels: "
                                    << numCh;
  size_t chStride = imageLayout == ImageLayout::NCHW ? shape[2] * shape[3] : 1;
  for (size_t i = 0; i < data.size(); i++) {
    size_t chIdx = (i / chStride) % numCh;
    data[i] = (data[i] - meanVal[chIdx]) / stddevVal[chIdx];
    data[i] = (data[i] - offset) * scale + bias;
  }
}

static void setPixelRange(NpyType type, ImgDataRange &pixelRange) {
  switch (type) {
  case (NpyType::I1):
    pixelRange = ImgDataRange::S8;
    break;
  case (NpyType::U1):
    pixelRange = ImgDataRange::U8;
    break;
  case (NpyType::I2):
    pixelRange = ImgDataRange::S16;
    break;
  case (NpyType::U2):
    pixelRange = ImgDataRange::U16;
    break;
  case (NpyType::F4):
    // accept whathever is already set.
    break;
  default:
    LOG(FATAL) << "Wrong image type: " << int(type);
    break;
  }
}

void loadUnspecifiedImageAndPreprocess(
    const llvm::ArrayRef<std::string> &filenames, Tensor &inputData,
    ImageNormalizationMode imageNormMode, llvm::ArrayRef<float> mean,
    llvm::ArrayRef<float> stddev, ImgDataRange &pixelRange) {

  CHECK_EQ(filenames.size(), 1) << "NPY raw image loader: expect single file.";
  CHECK_LE(mean.size(), 1) << "NPY raw image loader: expect single mean value.";
  CHECK_LE(stddev.size(), 1)
      << "NPY raw image loader: expect single stddev value.";

  NpyData dataNpy;
  numpyReader(filenames[0], dataNpy);
  std::vector<float> data(dataNpy.nvals);

  convertNumpyToFloat(dataNpy, data);
  setPixelRange(dataNpy.type, pixelRange);
  normalizeData(ImageLayout::Unspecified, mean, stddev, imageNormMode, data,
                dataNpy.shape, pixelRange);

  inputData.reset(ElemKind::FloatTy, dataNpy.shape);
  inputData.getHandle<>() = data;
}

void glow::loadNumpyImagesAndPreprocess(
    const llvm::ArrayRef<std::string> &filenames, Tensor &inputData,
    ImageNormalizationMode imageNormMode, ImageChannelOrder &imageChannelOrder,
    ImageLayout imageLayout, ImageLayout inputLayout,
    llvm::ArrayRef<float> mean, llvm::ArrayRef<float> stddev,
    ImgDataRange &pixelRange) {

  DCHECK(!filenames.empty())
      << "NPY loader: There must be at least one filename in filenames.";

  imageChannelOrder = ImageChannelOrder::Unspecified;

  if (imageLayout == ImageLayout::Unspecified) {
    return loadUnspecifiedImageAndPreprocess(
        filenames, inputData, imageNormMode, mean, stddev, pixelRange);
  }

  dim_t numImg = filenames.size();

  // Read each tensor file into a vector of tensors.
  std::vector<Tensor> tensors(numImg);
  dim_t batchSize = 0;
  for (dim_t n = 0; n < numImg; n++) {

    NpyData dataNpy;
    numpyReader(filenames[n], dataNpy);
    std::vector<float> data(dataNpy.nvals);
    convertNumpyToFloat(dataNpy, data);
    // Expand 3D to 4D. Supporting 4D only.
    if (dataNpy.shape.size() == 3) {
      dataNpy.shape.insert(dataNpy.shape.begin(), 1);
    }
    CHECK_EQ(dataNpy.shape.size(), 4)
        << "NPY loader: Supporting only 3 or 4 dimensions.";
    normalizeData(imageLayout, mean, stddev, imageNormMode, data, dataNpy.shape,
                  pixelRange);
    // Load tensor from the vector obtained from the npy loader.
    tensors[n].reset(glow::ElemKind::FloatTy, dataNpy.shape);
    tensors[n].getHandle<>() = data;
    auto dims0 = tensors[0].dims();
    auto dims = tensors[n].dims();
    if (n > 0) {
      CHECK_EQ(dims0.size(), dims.size())
          << "NPY loader: Number of dimensions must match.";
      for (dim_t i = 1; i < dims.size(); i++) {
        CHECK_EQ(dims0[i], dims[i])
            << "NPY loader: Non-batch dimensions must match.";
      }
    }
    // Accumulate batch dimension after each dump is loaded.
    batchSize += dims[0];
  }
  // Input tensor dimensions are now fully known.
  inputData.reset(ElemKind::FloatTy,
                  {batchSize, tensors[0].dims()[1], tensors[0].dims()[2],
                   tensors[0].dims()[3]});
  auto IIDH = inputData.getHandle<>();
  // Insert each loaded file (in tensors[] tensors) as the input tensor slices.
  for (dim_t n = 0, batch = 0, e = tensors.size(); n < e; n++) {
    Handle<float> H = tensors[n].getHandle<>();
    IIDH.insertTensors(H, {batch, 0, 0, 0});
    batch += tensors[n].dims()[0];
  }

  // Convert to the requested layout.
  if (inputLayout != imageLayout && inputLayout != ImageLayout::Unspecified) {
    glow::Tensor transposed;
    if (imageLayout == ImageLayout::NCHW) {
      inputData.transpose(&transposed, NHWC2NCHW);
    } else {
      inputData.transpose(&transposed, NCHW2NHWC);
    }
    inputData = std::move(transposed);
  }
}

bool glow::isNumpyNpyFormat(const std::string &filename) {
  std::ifstream fs(filename.data(), std::ifstream::binary);
  CHECK(fs) << "NPY loader: Couldn't open file: " << filename << "\n";
  uint8_t prefix[8];
  fs.read((char *)prefix, 8);
  CHECK(fs) << "NPY loader: Reading file failed: " << filename;
  // check if the header contains numpy magic string.
  return checkNumpyMagicHdr(prefix);
}
