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

#include <fstream>

using namespace glow;

namespace {

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
    CHECK(fs >> val) << "Error loading text file '" << filename.data()
                     << "'! Only " << idx
                     << " values were given for loading a tensor "
                     << "with " << e << " elements!";
    handle.raw(idx) = val;
    if (idx < e - 1) {
      CHECK(fs >> ch) << "Error loading text file '" << filename.data()
                      << "'! Delimiter character ',' not found!";
      CHECK(ch == ',')
          << "Error loading text file '" << filename.data()
          << "'! Delimiter character is expected to be ',' but character '"
          << ch << "' was found!";
    } else {
      fs >> ch;
    }
  }
  CHECK(!(fs >> val)) << "Error loading text file '" << filename.data()
                      << "'! Too many values given for loading a tensor with "
                      << handle.actualSize() << " elements!";
}
} // namespace

void glow::dumpTensorToBinaryFile(Tensor &tensor, llvm::StringRef filename,
                                  bool dumpType) {
  std::ofstream fs;
  fs.open(filename.data(), std::ios::binary);
  CHECK(fs.is_open()) << "Error opening file '" << filename.data() << "'!";
  CHECK(tensor.getUnsafePtr())
      << "Tensor not initialized before dumping to binary file!";
  // Dump tensor type.
  if (dumpType) {
    std::string typeStr = tensor.getType().toString();
    fs.write(typeStr.c_str(), typeStr.size());
  }
  // Dump tensor data.
  fs.write(tensor.getUnsafePtr(), tensor.getSizeInBytes());
  fs.close();
}

void glow::loadTensorFromBinaryFile(Tensor &tensor, llvm::StringRef filename,
                                    bool loadType) {
  std::ifstream fs;
  fs.open(filename.data(), std::ios::binary);
  CHECK(fs.is_open()) << "Error opening file '" << filename.data() << "'!";
  // Load tensor type.
  size_t headerSize = 0;
  if (loadType) {
    std::string typeStr;
    char ch;
    do {
      CHECK(fs.read(&ch, 1)) << "Error reading file '" << filename.data()
                             << "'! Type delimiter character '>' not found!";
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
      << fileDataSize << " bytes into "
      << "tensor with size " << tensorSize << " bytes!";

  // Load tensor data.
  fs.seekg(headerSize, std::ios::beg);
  fs.read(tensor.getUnsafePtr(), tensorSize);
  fs.close();
}

void glow::dumpTensorToTextFile(Tensor &tensor, llvm::StringRef filename,
                                bool dumpType) {
  std::ofstream fs;
  fs.open(filename.data());
  CHECK(fs.is_open()) << "Error opening file '" << filename.data() << "'!";
  CHECK(tensor.getUnsafePtr())
      << "Tensor not initialized before dumping to text file!";
  // Dump tensor type.
  if (dumpType) {
    fs << tensor.getType().toString() << "\n";
  }
  // Dump tensor data.
  switch (tensor.getElementType()) {
  case ElemKind::FloatTy:
    return dumpTensorToTextFileImpl<float>(tensor, filename, fs);
  case ElemKind::Float16Ty:
    return dumpTensorToTextFileImpl<float16_t>(tensor, filename, fs);
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
  case ElemKind::BoolTy:
    return dumpTensorToTextFileImpl<bool>(tensor, filename, fs);
  default:
    llvm_unreachable("Tensor type not supported for dumping to text file!");
  }
  fs.close();
}

void glow::loadTensorFromTextFile(Tensor &tensor, llvm::StringRef filename,
                                  bool loadType) {
  std::ifstream fs;
  fs.open(filename.data());
  CHECK(fs.is_open()) << "Error opening file '" << filename.data() << "'!";
  // Load tensor type.
  if (loadType) {
    std::string typeStr;
    std::getline(fs, typeStr);
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
  case ElemKind::BoolTy:
    return loadTensorFromTextFileImpl<bool>(tensor, filename, fs);
  default:
    llvm_unreachable("Tensor type not supported for loading from text file!");
  }
  fs.close();
}
