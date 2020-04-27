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

#include "glow/Base/TensorSerialization.h"
#include "glow/Base/Image.h"
#include "glow/Graph/Graph.h"

#include "llvm/Support/CommandLine.h"
#include <fstream>

using namespace glow;

namespace glow {

/// Helper method to dump the tensor content into a raw text file.
template <class ElemTy>
static void dumpToRawTextFileImpl(Handle<ElemTy> handle,
                                  llvm::StringRef filename) {
  std::ofstream fs;
  fs.open(filename.data());
  CHECK(fs.is_open()) << "Error opening file '" << filename.data() << "'!";
  for (dim_t idx = 0, e = handle.actualSize(); idx < e; idx++) {
    fs << handle.raw(idx) << ", ";
  }
  fs.close();
}

/// Helper method to load tensor files into the model input tensor.
static void loadTensorFromFileWithType(Tensor &T, llvm::StringRef filename,
                                       ImageLayout imageLayout) {
  std::ifstream infile(filename.str().c_str());
  CHECK(infile.is_open()) << "Error opening file '" << filename.data() << "'!";
  std::string line;
  ShapeVector dims;

  CHECK(std::getline(infile, line)) << "Failed to read 1st line";
  std::stringstream ss(line);
  for (dim_t i = 0; i < 4; i++) {
    int val;
    CHECK(ss >> val) << "Failed to read dimension " << i;
    dims.push_back(val);
  }
  T.reset(ElemKind::FloatTy, dims);
  // Now read the tensor.
  CHECK(std::getline(infile, line)) << "Failed to read 2nd line";
  auto H = T.getHandle<>();
  std::stringstream ss2(line);
  for (dim_t i = 0, e = H.size(); i < e; i++) {
    float val;
    CHECK(ss2 >> val) << "Error loading file " << filename.data()
                      << " @ element " << i;
    H.raw(i) = val;
  }
  // Convert to requested layout (tensor blob is in NCHW by default).
  if (imageLayout == ImageLayout::NHWC) {
    Tensor transposed;
    T.transpose(&transposed, NCHW2NHWC);
    T = std::move(transposed);
  }
}

/// Set default tensor loader.
static InputTensorFileLoaderFn inputTensorFileLoader_ =
    loadTensorFromFileWithType;

} // namespace glow

void glow::registerInputTensorFileLoader(InputTensorFileLoaderFn loader) {
  inputTensorFileLoader_ = loader;
}

/// Helper method to load the tensor content from a raw text file.
template <class ElemTy>
static void loadFromRawTextFileImpl(Handle<ElemTy> handle,
                                    llvm::StringRef filename) {
  std::ifstream fs;
  fs.open(filename.data());
  CHECK(fs.is_open()) << "Error opening file '" << filename.data() << "'!";
  char ch;
  double val;
  for (dim_t idx = 0, e = handle.actualSize(); idx < e; idx++) {
    CHECK(fs >> val) << "Error loading raw text file '" << filename.data()
                     << "'! Only " << idx
                     << " values were given for loading a tensor "
                     << "with " << e << " elements!";
    handle.raw(idx) = val;
    if (idx < e - 1) {
      CHECK(fs >> ch) << "Error loading raw text file '" << filename.data()
                      << "'! Delimiter character ',' not found!";
      CHECK(ch == ',')
          << "Error loading raw text file '" << filename.data()
          << "'! Delimiter character is expected to be ',' but character '"
          << ch << "' was found!";
    } else {
      fs >> ch;
    }
  }
  CHECK(!(fs >> val)) << "Error loading raw text file '" << filename.data()
                      << "'! Too many values given for loading a tensor with "
                      << handle.actualSize() << " elements!";
  fs.close();
}

void glow::dumpToRawBinaryFile(Tensor &tensor, llvm::StringRef filename) {
  std::ofstream fs;
  fs.open(filename.data(), std::ios::binary);
  CHECK(fs.is_open()) << "Error opening file '" << filename.data() << "'!";
  CHECK(tensor.getUnsafePtr())
      << "Tensor not initialized before dumping to raw binary file!";
  fs.write(tensor.getUnsafePtr(), tensor.getSizeInBytes());
  fs.close();
}

void glow::loadFromRawBinaryFile(Tensor &tensor, llvm::StringRef filename) {
  std::ifstream fs;
  fs.open(filename.data(), std::ios::binary);
  CHECK(fs.is_open()) << "Error opening file '" << filename.data() << "'!";
  CHECK(tensor.getUnsafePtr())
      << "Tensor not initialized before loading from raw binary file!";
  // Verify file size matches tensor size in bytes.
  auto tensorSize = tensor.getSizeInBytes();
  fs.seekg(0, std::ios::end);
  std::streampos fileSize = fs.tellg();
  CHECK(fileSize == tensorSize)
      << "Error loading raw binary file '" << filename.data() << "' with size "
      << fileSize << " bytes into tensor with size " << tensorSize << " bytes!";
  // Read data.
  fs.seekg(0, std::ios::beg);
  fs.read(tensor.getUnsafePtr(), tensorSize);
  fs.close();
}

void glow::dumpToRawTextFile(Tensor &tensor, llvm::StringRef filename) {
  CHECK(tensor.getUnsafePtr())
      << "Tensor not initialized before dumping to raw text file!";
  switch (tensor.getElementType()) {
  case ElemKind::FloatTy:
    return dumpToRawTextFileImpl(tensor.getHandle<float>(), filename);
  case ElemKind::Float16Ty:
    return dumpToRawTextFileImpl(tensor.getHandle<float16_t>(), filename);
  case ElemKind::Int8QTy:
    return dumpToRawTextFileImpl(tensor.getHandle<int8_t>(), filename);
  case ElemKind::UInt8QTy:
    return dumpToRawTextFileImpl(tensor.getHandle<uint8_t>(), filename);
  case ElemKind::Int16QTy:
    return dumpToRawTextFileImpl(tensor.getHandle<int16_t>(), filename);
  case ElemKind::Int32QTy:
    return dumpToRawTextFileImpl(tensor.getHandle<int32_t>(), filename);
  case ElemKind::Int32ITy:
    return dumpToRawTextFileImpl(tensor.getHandle<int32_t>(), filename);
  case ElemKind::Int64ITy:
    return dumpToRawTextFileImpl(tensor.getHandle<int64_t>(), filename);
  case ElemKind::UInt8FusedQTy:
    return dumpToRawTextFileImpl(tensor.getHandle<uint8_t>(), filename);
  case ElemKind::UInt8FusedFP16QTy:
    return dumpToRawTextFileImpl(tensor.getHandle<uint8_t>(), filename);
  case ElemKind::UInt4FusedFP16QTy:
    return dumpToRawTextFileImpl(tensor.getHandle<uint8_t>(), filename);
  case ElemKind::BoolTy:
    return dumpToRawTextFileImpl(tensor.getHandle<bool>(), filename);
  default:
    llvm_unreachable("Tensor type not supported for dumping to raw text file!");
  }
}

void glow::loadFromRawTextFile(Tensor &tensor, llvm::StringRef filename) {
  CHECK(tensor.getUnsafePtr())
      << "Tensor not initialized before loading from raw text file!";
  switch (tensor.getElementType()) {
  case ElemKind::FloatTy:
    return loadFromRawTextFileImpl(tensor.getHandle<float>(), filename);
  case ElemKind::Float16Ty:
    return loadFromRawTextFileImpl(tensor.getHandle<float16_t>(), filename);
  case ElemKind::Int8QTy:
    return loadFromRawTextFileImpl(tensor.getHandle<int8_t>(), filename);
  case ElemKind::UInt8QTy:
    return loadFromRawTextFileImpl(tensor.getHandle<uint8_t>(), filename);
  case ElemKind::Int16QTy:
    return loadFromRawTextFileImpl(tensor.getHandle<int16_t>(), filename);
  case ElemKind::Int32QTy:
    return loadFromRawTextFileImpl(tensor.getHandle<int32_t>(), filename);
  case ElemKind::Int32ITy:
    return loadFromRawTextFileImpl(tensor.getHandle<int32_t>(), filename);
  case ElemKind::Int64ITy:
    return loadFromRawTextFileImpl(tensor.getHandle<int64_t>(), filename);
  case ElemKind::UInt8FusedQTy:
    return loadFromRawTextFileImpl(tensor.getHandle<uint8_t>(), filename);
  case ElemKind::UInt8FusedFP16QTy:
    return loadFromRawTextFileImpl(tensor.getHandle<uint8_t>(), filename);
  case ElemKind::UInt4FusedFP16QTy:
    return loadFromRawTextFileImpl(tensor.getHandle<uint8_t>(), filename);
  case ElemKind::BoolTy:
    return loadFromRawTextFileImpl(tensor.getHandle<bool>(), filename);
  default:
    llvm_unreachable(
        "Tensor type not supported for loading from raw text file!");
  }
}

void glow::loadInputImageFromFileWithType(
    const llvm::ArrayRef<std::string> &filenames, Tensor *inputData,
    ImageLayout imageLayout) {
  DCHECK(!filenames.empty())
      << "There must be at least one filename in filenames.";
  assert((dim_t)filenames.size() == filenames.size());
  dim_t numImages = filenames.size();

  CHECK(inputTensorFileLoader_) << "tensor loader not assigned!";

  // Read each tensor file into a vector of tensors.
  std::vector<Tensor> data(numImages);
  dim_t batchSize = 0;
  for (dim_t n = 0; n < numImages; n++) {
    inputTensorFileLoader_(data[n], filenames[n], imageLayout);
    auto dims0 = data[0].dims();
    auto dims = data[n].dims();
    CHECK_EQ(dims0[1], dims[1]) << "Non batch dimensions must match";
    CHECK_EQ(dims0[2], dims[2]) << "Non batch dimensions must match";
    CHECK_EQ(dims0[3], dims[3]) << "Non batch dimensions must match";
    batchSize += data[n].dims()[0];
  }

  // Input tensor size is known now.
  inputData->reset(ElemKind::FloatTy, {batchSize, data[0].dims()[1],
                                       data[0].dims()[2], data[0].dims()[3]});
  auto IIDH = inputData->getHandle<>();
  // Insert each loaded file (in data[] tensors) as the input tensor slices.
  for (dim_t n = 0, e = data.size(); n < e; n++) {
    Handle<float> H = data[n].getHandle<>();
    IIDH.insertTensors(H, {n, 0, 0, 0});
  }
}

/// Helper function for loadInputTensorFromFileWithType, to produce blob files.
void glow::dumpInputTensorToFileWithType(
    const llvm::ArrayRef<std::string> &filenames, const Tensor &T,
    ImageLayout imageLayout) {
  CHECK_EQ(filenames.size(), 1) << "Dumping support single file only";
  const std::string &filename = filenames[0];
  Tensor localTensor = T.clone();
  // Convert to requested layout (tensor blob is in NCHW by default).
  if (imageLayout == ImageLayout::NHWC) {
    Tensor transposed;
    localTensor.transpose(&transposed, NHWC2NCHW);
    localTensor = std::move(transposed);
  }
  std::ofstream outfile(filename.c_str());
  CHECK(outfile.is_open()) << "Error opening file '" << filename << "'!";
  // write dimensions to 1st line.
  for (dim_t i = 0; i < 4; i++) {
    CHECK(outfile << localTensor.dims()[i] << " ")
        << "Failed to write dimension " << i;
  }
  outfile << "\n";
  // write tensor to 2nd line.
  auto H = localTensor.getHandle<float>();
  for (auto e : H) {
    outfile << e << " ";
  }
}
