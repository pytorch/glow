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

#include "glow/Base/Type.h"
#include "llvm/Support/NativeFormatting.h"
#include "llvm/Support/raw_ostream.h"

namespace glow {
llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const Type &type) {
  os << type.getElementName();

  if (type.isQuantizedType()) {
    os << "[S:";
    llvm::write_double(os, type.getScale(), llvm::FloatStyle::Fixed, 9);
    os << " O:";
    os << type.getOffset();
    os << ']';
    auto valueRange = type.getQuantizedValueRange();
    os << "[";
    llvm::write_double(os, valueRange.first, llvm::FloatStyle::Fixed, 3);
    os << ",";
    llvm::write_double(os, valueRange.second, llvm::FloatStyle::Fixed, 3);
    os << "]";
  }

  os << '<';
  for (unsigned i = 0; i < type.numSizes_; ++i) {
    if (i) {
      os << " x ";
    }
    os << type.sizes_[i];
    if (type.numSizes_ >= 2 && i + 1 < type.numSizes_ &&
        type.strides_[i] != type.strides_[i + 1] * type.sizes_[i + 1]) {
      assert(type.strides_[i] % type.strides_[i + 1] == 0);
      // Print the alignment only if it is not 1.
      os << ":" << (type.strides_[i] / type.strides_[i + 1]);
    }
  }
  os << '>';

  return os;
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const TypeRef &type) {
  if (!type) {
    return os << "<none>";
  }
  return os << *type;
}

void Type::dump(llvm::raw_ostream &out) const { out << this; }

void Type::dump() const { dump(llvm::outs()); }

std::string Type::toString() const {
  std::string storage;
  llvm::raw_string_ostream os(storage);
  os << this;
  return os.str();
}

Type Type::fromString(llvm::StringRef str) {

  // Get element type.
  std::pair<llvm::StringRef, llvm::StringRef> strPair;
  auto strPair1 = str.split('<');
  auto strPair2 = str.split('[');
  if (strPair1.first.size() < strPair2.first.size()) {
    strPair = strPair1;
  } else {
    strPair = strPair2;
  }
  CHECK(strPair.first.size()) << "Type string element type field invalid!";
  ElemKind elemTy = Type::getElementKindFromName(strPair.first);

  // Get scale and offset for quantized type.
  double scale = 0;
  int32_t offset = 0;
  if (isQuantizedElemKind(elemTy)) {
    // Get scale.
    strPair = strPair.second.split(':').second.split(' ');
    CHECK(!strPair.first.getAsDouble(scale))
        << "Type string scale field invalid!";
    // Get offset.
    strPair = strPair.second.split(':').second.split(']');
    CHECK(!strPair.first.getAsInteger(0, offset))
        << "Type string offset field invalid!";
    // Ignore quantized min/max range.
    strPair = strPair.second.split('<');
  }

  // Get shape.
  llvm::StringRef shapeStr = strPair.second;
  CHECK(shapeStr.size()) << "Type string shape field invalid!";
  CHECK_EQ(shapeStr.back(), '>') << "Type string shape field invalid!";
  shapeStr = shapeStr.drop_back();
  CHECK(shapeStr.size()) << "Type string shape field invalid!";

  // Add the delimiter in the end to have the loop self contained.
  // Note: Type alignment field not supported.
  std::string shapeStrExt = shapeStr.str() + " x";
  shapeStr = llvm::StringRef(shapeStrExt);
  ShapeVector dims;
  while (shapeStr.contains('x')) {
    auto splitRes = shapeStr.split('x');
    auto dimStr = splitRes.first.trim();
    CHECK(!dimStr.contains(':')) << "Type with alignment field not supported!";
    dim_t dim;
    CHECK(!dimStr.getAsInteger(0, dim)) << "Type string shape field invalid!";
    dims.push_back(dim);
    shapeStr = splitRes.second;
  }

  // Return type.
  if (isQuantizedElemKind(elemTy)) {
    return Type(elemTy, dims, (float)scale, offset);
  } else {
    return Type(elemTy, dims);
  }
}

std::pair<float, float> getQuantizedValueRange(float scale, int32_t offset,
                                               ElemKind elementType) {
  assert(isQuantizedElemKind(elementType) &&
         "Can't get the quantized value range of a non-quantized type");

  int64_t low = 0, high = 0;
  switch (elementType) {
  case ElemKind::Int32QTy: {
    low = INT32_MIN;
    high = INT32_MAX;
    break;
  }
  case ElemKind::Int16QTy: {
    low = INT16_MIN;
    high = INT16_MAX;
    break;
  }
  case ElemKind::Int8QTy: {
    low = INT8_MIN;
    high = INT8_MAX;
    break;
  }
  case ElemKind::UInt8QTy: {
    low = UINT8_MIN;
    high = UINT8_MAX;
    break;
  }
  default:;
  }

  float lowFloat = (low - offset) * scale;
  float highFloat = (high - offset) * scale;
  return std::make_pair(lowFloat, highFloat);
}

} // namespace glow
