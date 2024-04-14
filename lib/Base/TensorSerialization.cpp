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
#ifdef WITH_PNG
#include "glow/Base/Image.h"
#endif
#include "glow/Graph/Graph.h"

#include "llvm/Support/CommandLine.h"
#include <fstream>

using namespace glow;

namespace glow {

/// Helper method to dump the tensor content into a text file.
template <class ElemTy>
static void dumpTensorToTextFileImpl(Tensor &tensor, llvm::StringRef filename,
                                     std::ofstream &fs) {
  Handle<ElemTy> handle = tensor.getHandle<ElemTy>();
  for (dim_t idx = 0, e = handle.actualSize(); idx < e; idx++) {
    fs << (double)handle.raw(idx) << ", ";
  }
}

/// Helper method to load the tensor content from a text file.
template <class ElemTy>
static void loadTensorFromTextFileImpl(Tensor &tensor, llvm::StringRef filename,
                                       std::ifstream &fs) {
  Handle<ElemTy> handle = tensor.getHandle<ElemTy>();
  char ch;
  double val;
  for (dim_t idx = 0, e = handle.actualSize(); idx < e; idx++) {
    // Load tensor value.
    CHECK(fs >> val) << "Error loading text file '" << filename.data()
                     << "'! Only " << idx
                     << " values were given for loading a tensor " << "with "
                     << e << " elements!";
    handle.raw(idx) = val;
    // Check delimiter.
    CHECK(fs >> ch) << "Error loading text file '" << filename.data()
                    << "'! Delimiter character ',' not found!";
    if (idx < e - 1) {
      CHECK(ch == ',')
          << "Error loading text file '" << filename.data()
          << "'! Delimiter character is expected to be ',' but character '"
          << ch << "' was found!";
    }
  }
  CHECK(!(fs >> val)) << "Error loading text file '" << filename.data()
                      << "'! Too many values given for loading a tensor with "
                      << handle.actualSize() << " elements!";
}

#ifdef WITH_PNG

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

#endif // WITH_PNG

} // namespace glow

#ifdef WITH_PNG

void glow::registerInputTensorFileLoader(InputTensorFileLoaderFn loader) {
  inputTensorFileLoader_ = loader;
}

#endif // WITH_PNG

void glow::dumpTensorToBinaryFile(const Tensor &tensor,
                                  llvm::StringRef filename,
                                  const TensorSerializationOptions &opts) {
  std::ofstream fs;
  fs.open(filename.data(), std::ios::binary);
  CHECK(fs.is_open()) << "Error opening file '" << filename.data() << "'!";
  dumpTensorToBinaryFile(tensor, fs, opts);
  fs.close();
}

void glow::dumpTensorToBinaryFile(const Tensor &tensor, std::ofstream &fs,
                                  const TensorSerializationOptions &opts) {
  CHECK(tensor.getUnsafePtr())
      << "Tensor not initialized before dumping to binary file!";
  // Dump tensor type.
  if (opts.withType) {
    std::string typeStr = tensor.getType().toString();
    fs.write(typeStr.c_str(), typeStr.size());
  }
  // Dump tensor data.
  fs.write(tensor.getUnsafePtr(), tensor.getSizeInBytes());
}

void glow::loadTensorFromBinaryFile(Tensor &tensor, llvm::StringRef filename,
                                    const TensorSerializationOptions &opts) {
  std::ifstream fs;
  fs.open(filename.data(), std::ios::binary);
  CHECK(fs.is_open()) << "Error opening file '" << filename.data() << "'!";
  // Load tensor type.
  size_t headerSize = 0;
  if (opts.withType) {
    std::string typeStr;
    char ch;
    do {
      CHECK(fs.read(&ch, 1))
          << "Error loading binary file '" << filename.data()
          << "'! Tensor type delimiter character '>' not found!";
      typeStr += ch;
    } while (ch != '>');
    tensor.reset(Type::fromString(typeStr));
    headerSize = typeStr.size();
  } else {
    CHECK(tensor.getUnsafePtr())
        << "Tensor not initialized before loading from raw binary file!";
  }
  // Verify file data size matches tensor size in bytes.
  size_t tensorSize = tensor.getSizeInBytes();
  fs.seekg(0, std::ios::end);
  size_t fileDataSize = size_t(fs.tellg()) - headerSize;
  CHECK(fileDataSize == tensorSize)
      << "Error loading binary file '" << filename.data()
      << "' with header size " << headerSize << " bytes and data size "
      << fileDataSize << " bytes into " << "tensor with size " << tensorSize
      << " bytes!";

  // Load tensor data.
  fs.seekg(headerSize, std::ios::beg);
  fs.read(tensor.getUnsafePtr(), tensorSize);
  fs.close();
}

void glow::dumpTensorToTextFile(Tensor &tensor, llvm::StringRef filename,
                                const TensorSerializationOptions &opts) {
  std::ofstream fs;
  fs.open(filename.data());
  CHECK(fs.is_open()) << "Error opening file '" << filename.data() << "'!";
  CHECK(tensor.getUnsafePtr())
      << "Tensor not initialized before dumping to text file!";
  // Dump tensor type.
  if (opts.withType) {
    fs << tensor.getType().toString() << "\n";
  }
  // Dump tensor data.
  switch (tensor.getElementType()) {
  case ElemKind::FloatTy:
    return dumpTensorToTextFileImpl<float>(tensor, filename, fs);
  case ElemKind::Float16Ty:
    return dumpTensorToTextFileImpl<float16_t>(tensor, filename, fs);
  case ElemKind::BFloat16Ty:
    return dumpTensorToTextFileImpl<bfloat16_t>(tensor, filename, fs);
  case ElemKind::Int8QTy:
    return dumpTensorToTextFileImpl<int8_t>(tensor, filename, fs);
  case ElemKind::UInt8QTy:
    return dumpTensorToTextFileImpl<uint8_t>(tensor, filename, fs);
  case ElemKind::Int16QTy:
    return dumpTensorToTextFileImpl<int16_t>(tensor, filename, fs);
  case ElemKind::Int32QTy:
    return dumpTensorToTextFileImpl<int32_t>(tensor, filename, fs);
  case ElemKind::Int32ITy:
    return dumpTensorToTextFileImpl<int32_t>(tensor, filename, fs);
  case ElemKind::Int64ITy:
    return dumpTensorToTextFileImpl<int64_t>(tensor, filename, fs);
  case ElemKind::UInt8FusedQTy:
    return dumpTensorToTextFileImpl<uint8_t>(tensor, filename, fs);
  case ElemKind::UInt8FusedFP16QTy:
    return dumpTensorToTextFileImpl<uint8_t>(tensor, filename, fs);
  case ElemKind::UInt4FusedFP16QTy:
    return dumpTensorToTextFileImpl<uint8_t>(tensor, filename, fs);
  case ElemKind::UInt4FusedQTy:
    return dumpTensorToTextFileImpl<uint8_t>(tensor, filename, fs);
  case ElemKind::BoolTy:
    return dumpTensorToTextFileImpl<bool>(tensor, filename, fs);
  default:
    llvm_unreachable("Tensor type not supported for dumping to text file!");
  }
  fs.close();
}

void glow::loadTensorFromTextFile(Tensor &tensor, llvm::StringRef filename,
                                  const TensorSerializationOptions &opts) {
  std::ifstream fs;
  fs.open(filename.data());
  CHECK(fs.is_open()) << "Error opening file '" << filename.data() << "'!";
  // Load tensor type.
  if (opts.withType) {
    std::string typeStr;
    CHECK(std::getline(fs, typeStr))
        << "Error loading text file '" << filename.data()
        << "'! Tensor type not found!";
    tensor.reset(Type::fromString(typeStr));
  } else {
    CHECK(tensor.getUnsafePtr())
        << "Tensor not initialized before loading from raw text file!";
  }
  // Load tensor data.
  switch (tensor.getElementType()) {
  case ElemKind::FloatTy:
    return loadTensorFromTextFileImpl<float>(tensor, filename, fs);
  case ElemKind::Float16Ty:
    return loadTensorFromTextFileImpl<float16_t>(tensor, filename, fs);
  case ElemKind::BFloat16Ty:
    return loadTensorFromTextFileImpl<bfloat16_t>(tensor, filename, fs);
  case ElemKind::Int8QTy:
    return loadTensorFromTextFileImpl<int8_t>(tensor, filename, fs);
  case ElemKind::UInt8QTy:
    return loadTensorFromTextFileImpl<uint8_t>(tensor, filename, fs);
  case ElemKind::Int16QTy:
    return loadTensorFromTextFileImpl<int16_t>(tensor, filename, fs);
  case ElemKind::Int32QTy:
    return loadTensorFromTextFileImpl<int32_t>(tensor, filename, fs);
  case ElemKind::Int32ITy:
    return loadTensorFromTextFileImpl<int32_t>(tensor, filename, fs);
  case ElemKind::Int64ITy:
    return loadTensorFromTextFileImpl<int64_t>(tensor, filename, fs);
  case ElemKind::UInt8FusedQTy:
    return loadTensorFromTextFileImpl<uint8_t>(tensor, filename, fs);
  case ElemKind::UInt8FusedFP16QTy:
    return loadTensorFromTextFileImpl<uint8_t>(tensor, filename, fs);
  case ElemKind::UInt4FusedFP16QTy:
    return loadTensorFromTextFileImpl<uint8_t>(tensor, filename, fs);
  case ElemKind::UInt4FusedQTy:
    return loadTensorFromTextFileImpl<uint8_t>(tensor, filename, fs);
  case ElemKind::BoolTy:
    return loadTensorFromTextFileImpl<bool>(tensor, filename, fs);
  default:
    llvm_unreachable("Tensor type not supported for loading from text file!");
  }
  fs.close();
}

#ifdef WITH_PNG

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

#endif // WITH_PNG
