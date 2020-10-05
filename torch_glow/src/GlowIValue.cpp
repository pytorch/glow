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

#include "GlowIValue.h"

#include <ATen/core/ivalue.h>

#include "PyTorchCommon.h"

#include "glow/Support/Error.h"
#include "glow/Support/Support.h"

namespace glow {
// static
const char *GlowIValue::tagToStr(GlowIValue::Tag tag) {
  switch (tag) {
  case GlowIValue::Tag::None:
    return "None";
  case GlowIValue::Tag::Tensor:
    return "Tensor";
  case GlowIValue::Tag::Double:
    return "Double";
  case GlowIValue::Tag::Int:
    return "Int";
  case GlowIValue::Tag::Bool:
    return "Bool";
  case GlowIValue::Tag::IntList:
    return "IntList";
  case GlowIValue::Tag::DoubleList:
    return "DoubleList";
  case GlowIValue::Tag::BoolList:
    return "BoolList";
  case GlowIValue::Tag::NodeValueList:
    return "NodeValueList";
  case GlowIValue::Tag::Tuple:
    return "Tuple";
  case GlowIValue::Tag::PTTensor:
    return "PyTorch Tensor";
  case GlowIValue::Tag::GenericMap:
    return "GenericMap";
  case GlowIValue::Tag::String:
    return "String";
  }
  LOG(DFATAL) << "Cannot reach here.";
}

void GlowIValue::reset() {
  switch (tag_) {
  case Tag::Tensor:
    delete payload_.asTensor;
    break;
  case Tag::IntList:
    delete payload_.asIntList;
    break;
  case Tag::DoubleList:
    delete payload_.asDoubleList;
    break;
  case Tag::BoolList:
    delete payload_.asBoolList;
    break;
  case Tag::NodeValueList:
    delete payload_.asNodeValueList;
    break;
  case Tag::Tuple:
    delete payload_.asTuple;
    break;
  case Tag::PTTensor:
    delete payload_.asPTTensor;
    break;
  case Tag::GenericMap:
    delete payload_.asGenericMap;
    break;
  case Tag::String:
    delete payload_.asString;
    break;
  case Tag::None:
  case Tag::Double:
  case Tag::Int:
  case Tag::Bool:
    // Nothing to free.
    break;
  }
  tag_ = Tag::None;
}

GlowIValue::~GlowIValue() { reset(); }

GlowIValue::GlowIValue(GlowIValue &&other) {
  std::swap(tag_, other.tag_);
  std::swap(payload_, other.payload_);
}

GlowIValue &GlowIValue::operator=(GlowIValue &&other) {
  reset();
  std::swap(tag_, other.tag_);
  std::swap(payload_, other.payload_);
  return *this;
}

GlowIValue::Tag GlowIValue::getTag() const { return tag_; }

const char *GlowIValue::getTagString() const { return tagToStr(tag_); }

Expected<size_t> GlowIValue::hash(const GlowIValue &ival) {
  switch (ival.getTag()) {
  case GlowIValue::Tag::Int:
    return std::hash<int64_t>()(EXIT_ON_ERR(ival.toInt()));
  case GlowIValue::Tag::String:
    return std::hash<std::string>()(*EXIT_ON_ERR(ival.toString()));
  case GlowIValue::Tag::None:
  case GlowIValue::Tag::Double:
  case GlowIValue::Tag::Bool:
  case GlowIValue::Tag::Tensor:
  case GlowIValue::Tag::IntList:
  case GlowIValue::Tag::DoubleList:
  case GlowIValue::Tag::BoolList:
  case GlowIValue::Tag::NodeValueList:
  case GlowIValue::Tag::Tuple:
  case GlowIValue::Tag::PTTensor:
  case GlowIValue::Tag::GenericMap:
    return MAKE_ERR(
        strFormat("No hash function defined for IValues with tag %s",
                  ival.getTagString()));
  }

  LOG(DFATAL) << "Cannot reach here.";
}

bool GlowIValue::equal(const GlowIValue &ivalA, const GlowIValue &ivalB) {
  if (ivalA.getTag() != ivalB.getTag()) {
    return false;
  }

  switch (ivalA.getTag()) {
  case GlowIValue::Tag::Int:
    return EXIT_ON_ERR(ivalA.toInt()) == EXIT_ON_ERR(ivalB.toInt());
  case GlowIValue::Tag::None:
    return true;
  case GlowIValue::Tag::Double:
    return EXIT_ON_ERR(ivalA.toDouble()) == EXIT_ON_ERR(ivalB.toDouble());
  case GlowIValue::Tag::Bool:
    return EXIT_ON_ERR(ivalA.toBool()) == EXIT_ON_ERR(ivalB.toBool());
  case GlowIValue::Tag::Tensor:
    // Equal if they are the same tensor.
    return EXIT_ON_ERR(ivalA.toTensor()) == EXIT_ON_ERR(ivalB.toTensor());
  case GlowIValue::Tag::IntList: {
    const std::vector<int64_t> &vecA = *EXIT_ON_ERR(ivalA.toIntList());
    const std::vector<int64_t> &vecB = *EXIT_ON_ERR(ivalB.toIntList());
    return vecA == vecB;
  }
  case GlowIValue::Tag::DoubleList: {
    const std::vector<double> &vecA = *EXIT_ON_ERR(ivalA.toDoubleList());
    const std::vector<double> &vecB = *EXIT_ON_ERR(ivalB.toDoubleList());
    return vecA == vecB;
  }
  case GlowIValue::Tag::BoolList: {
    const std::vector<bool> &vecA = *EXIT_ON_ERR(ivalA.toBoolList());
    const std::vector<bool> &vecB = *EXIT_ON_ERR(ivalB.toBoolList());
    return vecA == vecB;
  }
  case GlowIValue::Tag::NodeValueList: {
    const std::vector<glow::NodeValue> &vecA =
        *EXIT_ON_ERR(ivalA.toNodeValueList());
    const std::vector<glow::NodeValue> &vecB =
        *EXIT_ON_ERR(ivalB.toNodeValueList());
    return vecA == vecB;
  }
  case GlowIValue::Tag::Tuple: {
    const std::vector<GlowIValue> &vecA = *EXIT_ON_ERR(ivalA.toTuple());
    const std::vector<GlowIValue> &vecB = *EXIT_ON_ERR(ivalB.toTuple());
    return vecA == vecB;
  }
  case GlowIValue::Tag::PTTensor:
    // Equal if they are the same PyTorch tensor.
    return EXIT_ON_ERR(ivalA.toPTTensor()) == EXIT_ON_ERR(ivalB.toPTTensor());
  case GlowIValue::Tag::GenericMap:
    return *EXIT_ON_ERR(ivalA.toGenericMap()) ==
           *EXIT_ON_ERR(ivalB.toGenericMap());
  case GlowIValue::Tag::String:
    return *EXIT_ON_ERR(ivalA.toString()) == *EXIT_ON_ERR(ivalB.toString());
  }

  LOG(DFATAL) << "Cannot reach here.";
}

bool GlowIValue::operator==(const GlowIValue &other) const {
  return GlowIValue::equal(*this, other);
}

bool GlowIValue::isNone() const { return Tag::None == tag_; }
bool GlowIValue::isTensor() const { return Tag::Tensor == tag_; }
bool GlowIValue::isDouble() const { return Tag::Double == tag_; }
bool GlowIValue::isInt() const { return Tag::Int == tag_; }
bool GlowIValue::isBool() const { return Tag::Bool == tag_; }
bool GlowIValue::isIntList() const { return Tag::IntList == tag_; }
bool GlowIValue::isDoubleList() const { return Tag::DoubleList == tag_; }
bool GlowIValue::isBoolList() const { return Tag::BoolList == tag_; }
bool GlowIValue::isNodeValueList() const { return Tag::NodeValueList == tag_; }
bool GlowIValue::isTuple() const { return Tag::Tuple == tag_; }
bool GlowIValue::isPTTensor() const { return Tag::PTTensor == tag_; }
bool GlowIValue::isGenericMap() const { return Tag::GenericMap == tag_; }
bool GlowIValue::isString() const { return Tag::String == tag_; }

#define ExpectTag(EXPECTED_TAG)                                                \
  RETURN_ERR_IF_NOT(tag_ == (EXPECTED_TAG),                                    \
                    strFormat("Expected GlowIValue with tag %s but found %s",  \
                              tagToStr((EXPECTED_TAG)), tagToStr(tag_)))

Expected<Tensor *> GlowIValue::toTensor() {
  ExpectTag(Tag::Tensor);
  return payload_.asTensor;
}

Expected<const Tensor *> GlowIValue::toTensor() const {
  ExpectTag(Tag::Tensor);
  return payload_.asTensor;
}

Expected<double> GlowIValue::toDouble() const {
  ExpectTag(Tag::Double);
  return payload_.asDouble;
}

Expected<int64_t> GlowIValue::toInt() const {
  ExpectTag(Tag::Int);
  return payload_.asInt;
}

Expected<bool> GlowIValue::toBool() const {
  ExpectTag(Tag::Bool);
  return payload_.asBool;
}

Expected<std::vector<int64_t> *> GlowIValue::toIntList() {
  ExpectTag(Tag::IntList);
  return payload_.asIntList;
}

Expected<const std::vector<int64_t> *> GlowIValue::toIntList() const {
  ExpectTag(Tag::IntList);
  return payload_.asIntList;
}

Expected<std::vector<double> *> GlowIValue::toDoubleList() {
  ExpectTag(Tag::DoubleList);
  return payload_.asDoubleList;
}

Expected<const std::vector<double> *> GlowIValue::toDoubleList() const {
  ExpectTag(Tag::DoubleList);
  return payload_.asDoubleList;
}

Expected<std::vector<bool> *> GlowIValue::toBoolList() {
  ExpectTag(Tag::BoolList);
  return payload_.asBoolList;
}

Expected<const std::vector<bool> *> GlowIValue::toBoolList() const {
  ExpectTag(Tag::BoolList);
  return payload_.asBoolList;
}

Expected<std::vector<glow::NodeValue> *> GlowIValue::toNodeValueList() {
  ExpectTag(Tag::NodeValueList);
  return payload_.asNodeValueList;
}

Expected<const std::vector<glow::NodeValue> *>
GlowIValue::toNodeValueList() const {
  ExpectTag(Tag::NodeValueList);
  return payload_.asNodeValueList;
}

Expected<std::vector<GlowIValue> *> GlowIValue::toTuple() {
  ExpectTag(Tag::Tuple);
  return payload_.asTuple;
}

Expected<const std::vector<GlowIValue> *> GlowIValue::toTuple() const {
  ExpectTag(Tag::Tuple);
  return payload_.asTuple;
}

Expected<at::Tensor *> GlowIValue::toPTTensor() {
  ExpectTag(Tag::PTTensor);
  return payload_.asPTTensor;
}

Expected<const at::Tensor *> GlowIValue::toPTTensor() const {
  ExpectTag(Tag::PTTensor);
  return payload_.asPTTensor;
}

Expected<GlowIValueMap *> GlowIValue::toGenericMap() {
  ExpectTag(Tag::GenericMap);
  return payload_.asGenericMap;
}

Expected<const GlowIValueMap *> GlowIValue::toGenericMap() const {
  ExpectTag(Tag::GenericMap);
  return payload_.asGenericMap;
}

Expected<std::string *> GlowIValue::toString() {
  ExpectTag(Tag::String);
  return payload_.asString;
}

Expected<const std::string *> GlowIValue::toString() const {
  ExpectTag(Tag::String);
  return payload_.asString;
}

#undef ExpectTag

void GlowIValue::fromNone() {
  reset();
  tag_ = Tag::None;
}

void GlowIValue::fromTensor(Tensor tensor) {
  reset();
  tag_ = Tag::Tensor;
  payload_.asTensor = new glow::Tensor(std::move(tensor));
}

void GlowIValue::fromDouble(double d) {
  reset();
  tag_ = Tag::Double;
  payload_.asDouble = d;
}

void GlowIValue::fromInt(int64_t i) {
  reset();
  tag_ = Tag::Int;
  payload_.asInt = i;
}

void GlowIValue::fromBool(bool b) {
  reset();
  tag_ = Tag::Bool;
  payload_.asBool = b;
}

void GlowIValue::fromIntList(std::vector<int64_t> intList) {
  reset();
  tag_ = Tag::IntList;
  payload_.asIntList = new std::vector<int64_t>;
  std::swap(intList, *payload_.asIntList);
}

void GlowIValue::fromDoubleList(std::vector<double> doubleList) {
  reset();
  tag_ = Tag::DoubleList;
  payload_.asDoubleList = new std::vector<double>;
  std::swap(doubleList, *payload_.asDoubleList);
}

void GlowIValue::fromBoolList(std::vector<bool> boolList) {
  reset();
  tag_ = Tag::BoolList;
  payload_.asBoolList = new std::vector<bool>;
  std::swap(boolList, *payload_.asBoolList);
}

void GlowIValue::fromNodeValueList(std::vector<glow::NodeValue> nodeValueList) {
  reset();
  tag_ = Tag::NodeValueList;
  payload_.asNodeValueList = new std::vector<glow::NodeValue>;
  std::swap(nodeValueList, *payload_.asNodeValueList);
}

void GlowIValue::fromTuple(std::vector<GlowIValue> glowIValList) {
  reset();
  tag_ = Tag::Tuple;
  payload_.asTuple = new std::vector<GlowIValue>;
  std::swap(glowIValList, *payload_.asTuple);
}

void GlowIValue::fromPTTensor(at::Tensor tensor) {
  CHECK(tensor.is_contiguous());
  reset();
  tag_ = Tag::PTTensor;
  payload_.asPTTensor = new at::Tensor(tensor);
}

void GlowIValue::fromString(std::string str) {
  reset();
  tag_ = Tag::String;
  payload_.asString = new std::string(std::move(str));
}

void GlowIValue::fromGenericMap(GlowIValueMap ivalMap) {
  reset();
  tag_ = Tag::GenericMap;
  payload_.asGenericMap = new GlowIValueMap;
  std::swap(ivalMap, *payload_.asGenericMap);
}

Error GlowIValue::fromIValue(const at::IValue &ival) {
  reset();
  if (ival.isNone()) {
    fromNone();
  } else if (ival.isTensor()) {
    auto at = ival.toTensor();
    glow::Tensor t;
    if (!at.is_contiguous()) {
      at = at.contiguous();
      t = ptTensorToGlowTensor(at).clone();
    } else {
      t = ptTensorToGlowTensor(at);
    }
    fromTensor(std::move(t));
  } else if (ival.isDouble()) {
    fromDouble(ival.toDouble());
  } else if (ival.isInt()) {
    fromInt(ival.toInt());
  } else if (ival.isBool()) {
    fromBool(ival.toBool());
  } else if (ival.isDoubleList()) {
    const auto ivalDoubles = ival.toDoubleList();
    std::vector<double> doubles(ivalDoubles.begin(), ivalDoubles.end());
    fromDoubleList(std::move(doubles));
  } else if (ival.isIntList()) {
    const auto ivalInts = ival.toIntList();
    std::vector<int64_t> ints(ivalInts.begin(), ivalInts.end());
    fromIntList(std::move(ints));
  } else if (ival.isBoolList()) {
    const auto ivalBools = ival.toBoolList();
    std::vector<bool> bools(ivalBools.begin(), ivalBools.end());
    fromBoolList(std::move(bools));
  } else if (ival.isString()) {
    std::string str = ival.toStringRef();
    fromString(std::move(str));
  } else if (ival.isDevice()) {
    fromInt(0); // TODO: Properly handle device iVal
  } else if (ival.isGenericDict()) {
    const auto &genericDict = ival.toGenericDict();
    GlowIValueMap ivalMap;
    for (const auto &kv : genericDict) {
      GlowIValue glowKey;
      GlowIValue glowValue;
      RETURN_IF_ERR(glowKey.fromIValue(kv.key()));
      RETURN_IF_ERR(glowValue.fromIValue(kv.value()));
      ivalMap.emplace(std::move(glowKey), std::move(glowValue));
    }
    fromGenericMap(std::move(ivalMap));
  } else if (ival.isTuple()) {
    const auto ivalTuple = ival.toTuple();
    const auto &elems = ivalTuple->elements();
    std::vector<GlowIValue> tuple;
    for (const auto &elem : elems) {
      GlowIValue glowIVal;
      RETURN_IF_ERR(glowIVal.fromIValue(elem));
      tuple.push_back(std::move(glowIVal));
    }
    fromTuple(std::move(tuple));
  } else {
    RETURN_ERR(strFormat("Encountered unhandled IValue type: %s",
                         ival.tagKind().data()));
  }
  return Error::success();
}

size_t GlowIValueMapHash::operator()(const GlowIValue &ival) const {
  auto hashOrErr = GlowIValue::hash(ival);
  if (hashOrErr) {
    return *hashOrErr;
  } else {
    LOG(DFATAL) << ERR_TO_STRING(hashOrErr.takeError());
  }
  return 0;
}

/// Unwrap a Expected<GlowIValue *> \p expectedIVal and call toDouble,
/// propogate any Errors.
Expected<double> iValToDouble(Expected<GlowIValue *> expectedIVal) {
  if (expectedIVal) {
    return (*expectedIVal)->toDouble();
  } else {
    return expectedIVal.takeError();
  }
}

/// Unwrap a Expected<GlowIValue *> \p expectedIVal and call toInt,
/// propogate any Errors.
Expected<int64_t> iValToInt(Expected<GlowIValue *> expectedIVal) {
  if (expectedIVal) {
    return (*expectedIVal)->toInt();
  } else {
    return expectedIVal.takeError();
  }
}

/// Unwrap a Expected<GlowIValue *> \p expectedIVal and call toBool,
/// propogate any Errors.
Expected<bool> iValToBool(Expected<GlowIValue *> expectedIVal) {
  if (expectedIVal) {
    return (*expectedIVal)->toBool();
  } else {
    return expectedIVal.takeError();
  }
}

/// Unwrap a Expected<GlowIValue *> \p expectedIVal and call toIntList,
/// propogate any Errors.
Expected<std::vector<int64_t> *>
iValToIntList(Expected<GlowIValue *> expectedIVal) {
  if (expectedIVal) {
    return (*expectedIVal)->toIntList();
  } else {
    return expectedIVal.takeError();
  }
}

/// Unwrap a Expected<GlowIValue *> \p expectedIVal and call toDoubleList,
/// propogate any Errors.
Expected<std::vector<double> *>
iValToDoubleList(Expected<GlowIValue *> expectedIVal) {
  if (expectedIVal) {
    return (*expectedIVal)->toDoubleList();
  } else {
    return expectedIVal.takeError();
  }
}

/// Unwrap a Expected<GlowIValue *> \p expectedIVal and call toNodeValueList,
/// propogate any Errors.
Expected<std::vector<glow::NodeValue> *>
iValToNodeValueList(Expected<GlowIValue *> expectedIVal) {
  if (expectedIVal) {
    return (*expectedIVal)->toNodeValueList();
  } else {
    return expectedIVal.takeError();
  }
}

/// Unwrap a Expected<GlowIValue *> \p expectedIVal and call toPTTensor,
/// propogate any Errors.
Expected<at::Tensor *> iValToPTTensor(Expected<GlowIValue *> expectedIVal) {
  if (expectedIVal) {
    return (*expectedIVal)->toPTTensor();
  } else {
    return expectedIVal.takeError();
  }
}

Expected<GlowIValueMap *>
iValToGenericMap(Expected<GlowIValue *> expectedIVal) {
  if (expectedIVal) {
    return (*expectedIVal)->toGenericMap();
  } else {
    return expectedIVal.takeError();
  }
}

Expected<std::string *> iValToString(Expected<GlowIValue *> expectedIVal) {
  if (expectedIVal) {
    return (*expectedIVal)->toString();
  } else {
    return expectedIVal.takeError();
  }
}

} // namespace glow
