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

using namespace glow;

// Convert data in the raw buffer \p in, of type \p Ti, into type \p To and
// store into the vector \p out.
template <typename Ti, typename To>
static void buffertoTypedVector(size_t elem_size, size_t num_vals, char *in,
                                std::vector<To> &out) {
  // If float requested, allow any integer, or float of equal or less size.
  // If signed int. requested, allow smaller/equal signed or smaller unsigned.
  // If unsigned int. requested, allow smaller or equal unsigned.
  // TODO: check value as well, to allow more conversions.
  bool ok = false;
  if (std::is_floating_point<To>::value) {
    if (std::is_integral<Ti>::value ||
        (std::is_floating_point<Ti>::value && sizeof(Ti) <= sizeof(To))) {
      ok = true;
    }
  }
  if (std::is_integral<To>::value && std::is_signed<To>::value) {
    if ((std::is_integral<Ti>::value && std::is_signed<Ti>::value &&
         sizeof(Ti) <= sizeof(To)) ||
        (std::is_integral<Ti>::value && std::is_unsigned<Ti>::value &&
         sizeof(Ti) < sizeof(To))) {
      ok = true;
    }
  }
  if (std::is_integral<To>::value && std::is_unsigned<To>::value) {
    if (std::is_integral<Ti>::value && std::is_unsigned<Ti>::value &&
        sizeof(Ti) <= sizeof(To)) {
      ok = true;
    }
  }
  CHECK(ok) << "NPY loader: Conversion not allowed, input type doesn't fit "
               "into output type";

  for (size_t i = 0; i < num_vals; i++) {
    out.push_back((To) * (Ti *)&in[i * elem_size]);
  }
}

/// Read npy file \p filename and store data tensor into vector \data. Tensor
/// values are converted into \p type and tensor shape is provided in \p shape.
template <typename T>
static void numpyReader(const std::string &filename, std::vector<dim_t> &shape,
                        std::vector<T> &data) {
  // Map tensor type string to an enum.
  enum NpyType {
    Int8Ty,
    UInt8Ty,
    Int16Ty,
    UInt16Ty,
    Int32Ty,
    UInt32Ty,
    Int64Ty,
    UInt64Ty,
    Float16Ty,
    Float32Ty,
    Float64Ty
  };

  std::map<std::string, NpyType> npyType = {
      {"i1", Int8Ty},    {"u1", UInt8Ty},   {"i2", Int16Ty},  {"u2", UInt16Ty},
      {"i4", Int32Ty},   {"u4", UInt32Ty},  {"i8", Int64Ty},  {"u8", UInt64Ty},
      {"f2", Float16Ty}, {"f4", Float32Ty}, {"f8", Float64Ty}};

  // Wrapper for string find() method, that checks for return value.
  auto find_str = [](std::string &str, const char *name,
                     size_t *loc = nullptr) {
    size_t newloc = loc ? str.find(name, *loc) : str.find(name);
    if (newloc == std::string::npos) {
      LOG(FATAL) << "NPY loader: Couldn't find string " << name;
    }
    return newloc;
  };

  // Wrapper for string find_first_of method, that checks for return value.
  auto find_first_of = [](std::string &str, const char *find, size_t loc) {
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

  // magic
  uint8_t MAGIC[] = {0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59};
  if (memcmp((uint8_t *)buffer, MAGIC, 6)) {
    LOG(FATAL) << "NPY loader: Magic number not found";
  }

  // version
  uint8_t version = buffer[6];
  if (version != 1) {
    LOG(FATAL) << "NPY loader: Version is invalid:" << version;
  }

  // get header.
  uint32_t hdrlen_len = version > 1 ? 4 : 2;
  uint32_t hdrlen =
      (hdrlen_len == 2) ? *(uint16_t *)&buffer[8] : *(uint32_t *)&buffer[8];
  std::string hdr(&buffer[8 + hdrlen_len], hdrlen);

  // Find type: Search for little endian identifers, until the end of the value.
  // Type string is 2 chars following endianess identifer.
  loc = find_str(hdr, "descr");
  loc = find_str(hdr, ":", &loc);
  loc = find_first_of(hdr, "<|,", loc);
  if (hdr[loc] != '<' && hdr[loc] != '|') {
    LOG(FATAL) << "NPY loader: Little-endian supported only got: " << hdr[loc];
  }
  std::string typestr = hdr.substr(loc + 1, 2);
  CHECK(npyType.count(typestr))
      << "NPY loader: Unknown type: " << typestr << "\n";
  auto type = npyType[typestr];

  size_t elem_size = strtol(&typestr[1], NULL, 10);
  CHECK(elem_size > 0 && elem_size <= 8)
      << "NPY loader: Element size wrong: " << elem_size;

  // find fortran_order : [True | False]
  loc = find_str(hdr, "fortran_order");
  loc = find_str(hdr, ":", &loc);
  loc = hdr.find_first_not_of(" '", loc + 1);
  locend = find_first_of(hdr, "', ", loc + 1);
  std::string order = hdr.substr(loc, locend - loc);
  CHECK_EQ(order, "False") << "NPY loader: fortran_order must be False";

  // find shape
  loc = find_str(hdr, "shape");
  loc = find_str(hdr, "(", &loc);
  locend = find_str(hdr, ")", &loc);
  std::string shapestr = hdr.substr(loc + 1, locend - loc - 1);
  std::stringstream ss(shapestr);
  std::string token;
  size_t nvals = 1;
  while (std::getline(ss, token, ',')) {
    size_t val = strtol(&token[0], NULL, 10);
    CHECK(val > 0) << "NPY loader: Element size wrong: " << elem_size;
    shape.push_back(val);
    nvals *= val;
  }

  // move file ptr to data
  int data_pos = 8 + hdrlen + hdrlen_len;
  char *databuf = &buffer[data_pos];

  switch (type) {
  case Int8Ty:
    buffertoTypedVector<int8_t>(elem_size, nvals, databuf, data);
    break;
  case UInt8Ty:
    buffertoTypedVector<uint8_t>(elem_size, nvals, databuf, data);
    break;
  case Int16Ty:
    buffertoTypedVector<int16_t>(elem_size, nvals, databuf, data);
    break;
  case UInt16Ty:
    buffertoTypedVector<uint16_t>(elem_size, nvals, databuf, data);
    break;
  case Int32Ty:
    buffertoTypedVector<int32_t>(elem_size, nvals, databuf, data);
    break;
  case UInt32Ty:
    buffertoTypedVector<uint32_t>(elem_size, nvals, databuf, data);
    break;
  case Int64Ty:
    buffertoTypedVector<int64_t>(elem_size, nvals, databuf, data);
    break;
  case UInt64Ty:
    buffertoTypedVector<uint64_t>(elem_size, nvals, databuf, data);
    break;
  case Float32Ty:
    buffertoTypedVector<float>(elem_size, nvals, databuf, data);
    break;
  case Float64Ty:
    buffertoTypedVector<double>(elem_size, nvals, databuf, data);
    break;
  default:
    LOG(FATAL) << "NPY loader: Not supported type: " << type;
  }
  delete[] buffer;
}

void glow::loadNumpyImagesAndPreprocess(
    const llvm::ArrayRef<std::string> &filenames, Tensor &inputData,
    ImageNormalizationMode imageNormMode, ImageLayout imageLayout,
    ImageLayout inputLayout, llvm::ArrayRef<float> mean,
    llvm::ArrayRef<float> stddev) {
  DCHECK(!filenames.empty())
      << "NPY loader: There must be at least one filename in filenames.";
  dim_t numImg = filenames.size();

  auto normalizeData = [&mean, &stddev, &imageLayout,
                        &imageNormMode](std::vector<float> &data,
                                        std::vector<dim_t> &shape) {
    auto range = normModeToRange(imageNormMode);
    float scale = ((range.second - range.first) / 255.f);
    float bias = range.first;
    dim_t numCh = (imageLayout == ImageLayout::NHWC) ? shape[3] : shape[1];
    std::vector<float> zeroMean(numCh, 0.f);
    std::vector<float> oneStd(numCh, 1.f);
    std::vector<float> meanVal(mean.size() ? mean
                                           : llvm::makeArrayRef(zeroMean));
    std::vector<float> stddevVal(stddev.size() ? stddev
                                               : llvm::makeArrayRef(oneStd));
    CHECK_EQ(numCh, meanVal.size()) << "NPY loader: mean argument size should "
                                       "match the number of channels: "
                                    << numCh;
    CHECK_EQ(numCh, stddevVal.size()) << "NPY loader: stddev argument size "
                                         "should match the number of channels: "
                                      << numCh;
    size_t chStride =
        (imageLayout == ImageLayout::NHWC) ? 1 : shape[2] * shape[3];
    for (size_t i = 0; i < data.size(); i++) {
      size_t chIdx = (i / chStride) % numCh;
      CHECK(data[i] >= 0. && data[i] <= 255.)
          << "NPY loader: U8 data expected, got: " << data[i];
      data[i] = (data[i] - meanVal[chIdx]) / stddevVal[chIdx];
      data[i] = data[i] * scale + bias;
    }
  };

  // Read each tensor file into a vector of tensors.
  std::vector<Tensor> tensors(numImg);
  dim_t batchSize = 0;
  for (dim_t n = 0; n < numImg; n++) {
    std::vector<dim_t> shape;
    std::vector<float> data;
    numpyReader(filenames[n], shape, data);
    // Expand 3D to 4D. Supporting 4D only.
    if (shape.size() == 3) {
      shape.insert(shape.begin(), 1);
    }
    CHECK_EQ(shape.size(), 4)
        << "NPY loader: Supporting only 3 or 4 dimensions.";
    normalizeData(data, shape);
    // Load tensor from the vector obtained from the npy loader.
    tensors[n].reset(glow::ElemKind::FloatTy, shape);
    tensors[n].getHandle<>() = data;
    auto dims0 = tensors[0].dims();
    auto dims = tensors[n].dims();
    if (n > 1) {
      CHECK_EQ(dims0.size(), dims.size())
          << "NPY loader: Number of dimensions must match.";
      for (dim_t i = 1; i < dims.size(); i++) {
        CHECK_EQ(dims0[i], dims[i])
            << "NPY loader: Non-batch dimensions must match.";
      }
    }
    // Accumulate batch dimension after each dump is loaded.
    batchSize += tensors[n].dims()[0];
  }
  // Input tensor dimensions are now fully known.
  inputData.reset(ElemKind::FloatTy,
                  {batchSize, tensors[0].dims()[1], tensors[0].dims()[2],
                   tensors[0].dims()[3]});
  auto IIDH = inputData.getHandle<>();
  // Insert each loaded file (in tensors[] tensors) as the input tensor slices.
  for (dim_t n = 0, e = tensors.size(); n < e; n++) {
    Handle<float> H = tensors[n].getHandle<>();
    IIDH.insertTensors(H, {n, 0, 0, 0});
  }

  // Convert to the requested layout.
  if (inputLayout != imageLayout) {
    glow::Tensor transposed;
    if (imageLayout == ImageLayout::NCHW) {
      inputData.transpose(&transposed, {0u, 3u, 1u, 2u});
    } else {
      inputData.transpose(&transposed, {0u, 2u, 3u, 1u});
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
  // magic
  uint8_t MAGIC[] = {0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59};
  return memcmp((uint8_t *)prefix, MAGIC, 6) ? false : true;
}
