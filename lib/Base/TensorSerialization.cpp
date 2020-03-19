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
} // namespace

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
