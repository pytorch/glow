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

#ifndef GLOW_TORCH_GLOW_SRC_GLOWIVALUE_H
#define GLOW_TORCH_GLOW_SRC_GLOWIVALUE_H

#include <ATen/core/ivalue.h>

#include "glow/Base/Tensor.h"
#include "glow/Graph/NodeValue.h"
#include "glow/Support/Error.h"
#include "glow/Support/Support.h"

namespace glow {
class GlowIValue;

/// Hash functor for GlowIValueMap.
/// NOTE: This functor is only defined for some GlowIValue tags and will assert
/// that the given GlowIValue has one of those tags. See GlowIValue::hash for
/// more.
struct GlowIValueMapHash {
  size_t operator()(const GlowIValue &ival) const;
};

/// Map type for storing mappings from GlowIValue to GlowIValue. Not all IValue
/// kinds can be keys, see GlowIValue::hash for details.
using GlowIValueMap =
    std::unordered_map<GlowIValue, GlowIValue, GlowIValueMapHash>;

/// GlowIValue is a tagged union type analogous to PyTorch's JIT IValue but
/// holds Glow Tensors instead. PyTorch Graph inputs and outputs and the results
/// of aten::Constant can be mapped to Glow as GlowIValues.
class GlowIValue {
public:
  /// All of the possible types of GlowIValues.
  enum class Tag {
    None,
    Tensor,
    Double,
    Int,
    Bool,
    IntList,
    DoubleList,
    BoolList,
    Tuple,
    PTTensor,
    GenericMap,
    String,
    NodeValueList,
  };

private:
  /// Container for whatever is being held by GlowIValue. GlowIValue owns all
  /// of the pointers in Payload and is responsible for freeing them when
  /// destroyed or when the tag changes.
  union Payload {
    Tensor *asTensor;
    double asDouble;
    int64_t asInt;
    bool asBool;
    std::vector<int64_t> *asIntList;
    std::vector<double> *asDoubleList;
    std::vector<bool> *asBoolList;
    std::vector<glow::NodeValue> *asNodeValueList;
    std::vector<GlowIValue> *asTuple;
    at::Tensor *asPTTensor;
    GlowIValueMap *asGenericMap;
    std::string *asString;
  };

  Tag tag_ = Tag::None;
  Payload payload_;

  /// Free any stored payload memory and set the tag to None.
  void reset();

public:
  GlowIValue() = default;

  ~GlowIValue();

  GlowIValue(GlowIValue &&other);

  GlowIValue &operator=(GlowIValue &&other);

  /// \returns the Tag of this GlowIValue.
  Tag getTag() const;

  /// \returns a string representing the Tag of this GlowIValue.
  const char *getTagString() const;

  /// \returns a string representing the Tag \p tag.
  static const char *tagToStr(GlowIValue::Tag tag);

  /// GlowIValue manually manages memory so no copies.
  GlowIValue(const GlowIValue &other) = delete;
  GlowIValue &operator=(const GlowIValue &other) = delete;

  /// Given a GlowIValue \p ival, \returns the hash of that value depending on
  /// the tag or an Error if a hash could not be computed.
  static Expected<size_t> hash(const GlowIValue &ival);

  /// Given GlowIValues \p ivalA and \p ivalB, \returns true iff they have the
  /// same tag and equal values.
  static bool equal(const GlowIValue &ivalA, const GlowIValue &ivalB);

  /// \return true iff this GlowIValue is equal to \p other.
  bool operator==(const GlowIValue &other) const;

  /// Methods to determine the tag of a GlowIValue.
  bool isNone() const;
  bool isTensor() const;
  bool isDouble() const;
  bool isInt() const;
  bool isBool() const;
  bool isIntList() const;
  bool isDoubleList() const;
  bool isBoolList() const;
  bool isNodeValueList() const;
  bool isTuple() const;
  bool isPTTensor() const;
  bool isGenericMap() const;
  bool isString() const;

  /// \returns Payload a glow Tensor or error if the tag is not Tensor.
  Expected<Tensor *> toTensor();

  /// \returns Payload a Tensor* or error if the tag is not Tensor.
  Expected<const Tensor *> toTensor() const;

  /// \returns Payload a double or error if the tag is not Double.
  Expected<double> toDouble() const;

  /// \returns Payload a int or error if the tag is not Int.
  Expected<int64_t> toInt() const;

  /// \returns Payload a bool or error if the tag is not Bool.
  Expected<bool> toBool() const;

  /// \returns Payload a vector of ints or error if the tag is not IntList.
  Expected<std::vector<int64_t> *> toIntList();

  /// \returns Payload a vector of ints or error if the tag is not IntList.
  Expected<const std::vector<int64_t> *> toIntList() const;

  /// \returns Payload a vector of doubles or error if the tag is not
  /// DoubleList.
  Expected<std::vector<double> *> toDoubleList();

  /// \returns Payload a vector of doubles or error if the tag is not
  /// DoubleList.
  Expected<const std::vector<double> *> toDoubleList() const;

  /// \returns Payload a vector of bools or error if the tag is not BoolList.
  Expected<std::vector<bool> *> toBoolList();

  /// \returns Payload a vector of bools or error if the tag is not BoolList.
  Expected<const std::vector<bool> *> toBoolList() const;

  /// \returns Payload a vector of glow::NodeValue or error if the tag is not
  /// NodeValueList.
  Expected<std::vector<glow::NodeValue> *> toNodeValueList();

  /// \returns Payload a vector of glow::NodeValue or error if the tag is not
  /// NodeValueList.
  Expected<const std::vector<glow::NodeValue> *> toNodeValueList() const;

  /// \returns Payload a vector of GlowIValues or error if the tag is not Tuple.
  Expected<std::vector<GlowIValue> *> toTuple();

  /// \returns Payload a vector of GlowIValues or error if the tag is not Tuple.
  Expected<const std::vector<GlowIValue> *> toTuple() const;

  /// \returns Payload a PyTorch Tensor* or error if the tag is not a PyTorch
  /// Tensor.
  Expected<at::Tensor *> toPTTensor();

  /// \returns Payload a const Pytorch Tensor* or error if the tag is not
  /// Tensor.
  Expected<const at::Tensor *> toPTTensor() const;

  /// \returns Payload a GlowIValueMap* or error if the tag is not
  /// GenericMap.
  Expected<GlowIValueMap *> toGenericMap();

  /// \returns Payload a const GlowIValueMap* or error if the tag is not
  /// GenericMap.
  Expected<const GlowIValueMap *> toGenericMap() const;

  /// \returns Payload a std::string* or error if the tag is not
  /// String.
  Expected<std::string *> toString();

  /// \returns Payload a const std::string* or error if the tag is not
  /// String.
  Expected<const std::string *> toString() const;

  /// Set the tag to None.
  void fromNone();

  /// Set the tag to Tensor.
  void fromTensor(Tensor tensor);

  /// Set the tag to PyTorch Tensor.
  void fromPTTensor(at::Tensor tensor);

  /// Set the tag to Double.
  void fromDouble(double d);

  /// Set the tag to Int.
  void fromInt(int64_t i);

  /// Set the tag to Bool.
  void fromBool(bool b);

  /// Set the tag to IntList.
  void fromIntList(std::vector<int64_t> intList);

  /// Set the tag to DoubleList.
  void fromDoubleList(std::vector<double> doubleList);

  /// Set the tag to BoolList.
  void fromBoolList(std::vector<bool> boolList);

  /// Set the tag to NodeValueList.
  void fromNodeValueList(std::vector<glow::NodeValue> nodeValueList);

  /// Set the tag to Tuple.
  void fromTuple(std::vector<GlowIValue> glowIValList);

  /// Set the tag to GenericMap.
  void fromGenericMap(GlowIValueMap glowIValueMap);

  /// Set the tag to String.
  void fromString(std::string str);

  /// Given a PyTorch IValue \p ival, set the tag to the analogous Tag.
  Error fromIValue(const at::IValue &ival);
};

/// Unwrap a Expected<GlowIValue *> \p expectedIVal and call toDouble,
/// propogate any Errors.
Expected<double> iValToDouble(Expected<GlowIValue *> expectedIVal);

/// Unwrap a Expected<GlowIValue *> \p expectedIVal and call toInt,
/// propogate any Errors.
Expected<int64_t> iValToInt(Expected<GlowIValue *> expectedIVal);

/// Unwrap a Expected<GlowIValue *> \p expectedIVal and call toBool,
/// propogate any Errors.
Expected<bool> iValToBool(Expected<GlowIValue *> expectedIVal);

/// Unwrap a Expected<GlowIValue *> \p expectedIVal and call toIntList,
/// propogate any Errors.
Expected<std::vector<int64_t> *>
iValToIntList(Expected<GlowIValue *> expectedIVal);

/// Unwrap a Expected<GlowIValue *> \p expectedIVal and call toDoubleList,
/// propogate any Errors.
Expected<std::vector<double> *>
iValToDoubleList(Expected<GlowIValue *> expectedIVal);

/// Unwrap a Expected<GlowIValue *> \p expectedIVal and call toNodeValueList,
/// propogate any Errors.
Expected<std::vector<glow::NodeValue> *>
iValToNodeValueList(Expected<GlowIValue *> expectedIVal);

/// Unwrap a Expected<GlowIValue *> \p expectedIVal and call toPTTensor,
/// propogate any Errors.
Expected<at::Tensor *> iValToPTTensor(Expected<GlowIValue *> expectedIVal);

/// Unwrap a Expected<GlowIValue *> \p expectedIVal and call toGenericMap,
/// propogate any Errors.
Expected<GlowIValueMap *> iValToGenericMap(Expected<GlowIValue *> expectedIVal);

/// Unwrap a Expected<GlowIValue *> \p expectedIVal and call toString,
/// propogate any Errors.
Expected<std::string *> iValToString(Expected<GlowIValue *> expectedIVal);

} // namespace glow

#endif // GLOW_TORCH_GLOW_SRC_GLOWIVALUE_H
