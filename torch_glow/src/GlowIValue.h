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
#include "glow/Support/Error.h"
#include "glow/Support/Support.h"

namespace glow {

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
    std::vector<GlowIValue> *asTuple;
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

  /// Methods to determine the tag of a GlowIValue.
  bool isNone() const;
  bool isTensor() const;
  bool isDouble() const;
  bool isInt() const;
  bool isBool() const;
  bool isIntList() const;
  bool isDoubleList() const;
  bool isBoolList() const;
  bool isTuple() const;

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

  /// \returns Payload a vector of GlowIValues or error if the tag is not Tuple.
  Expected<std::vector<GlowIValue> *> toTuple();

  /// \returns Payload a vector of GlowIValues or error if the tag is not Tuple.
  Expected<const std::vector<GlowIValue> *> toTuple() const;

  /// Set the tag to None.
  void fromNone();

  /// Set the tag to Tensor.
  void fromTensor(Tensor tensor);

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

  /// Set the tag to Tuple.
  void fromTuple(std::vector<GlowIValue> glowIValList);

  /// Given a PyTorch IValue \p ival, set the tag to the analogous Tag.
  Error fromIValue(const at::IValue &ival);
};

} // namespace glow

#endif // GLOW_TORCH_GLOW_SRC_GLOWIVALUE_H
