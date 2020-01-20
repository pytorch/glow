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
  case GlowIValue::Tag::Tuple:
    return "Tuple";
  case GlowIValue::Tag::PTTensor:
    return "PyTorch Tensor";
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
  case Tag::Tuple:
    delete payload_.asTuple;
    break;
  case Tag::PTTensor:
    delete payload_.asPTTensor;
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

bool GlowIValue::isNone() const { return Tag::None == tag_; }
bool GlowIValue::isTensor() const { return Tag::Tensor == tag_; }
bool GlowIValue::isDouble() const { return Tag::Double == tag_; }
bool GlowIValue::isInt() const { return Tag::Int == tag_; }
bool GlowIValue::isBool() const { return Tag::Bool == tag_; }
bool GlowIValue::isIntList() const { return Tag::IntList == tag_; }
bool GlowIValue::isDoubleList() const { return Tag::DoubleList == tag_; }
bool GlowIValue::isBoolList() const { return Tag::BoolList == tag_; }
bool GlowIValue::isTuple() const { return Tag::Tuple == tag_; }
bool GlowIValue::isPTTensor() const { return Tag::PTTensor == tag_; }

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

} // namespace glow
