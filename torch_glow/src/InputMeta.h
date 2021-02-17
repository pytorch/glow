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

#ifndef GLOW_TORCH_GLOW_SRC_INPUTMETA_H
#define GLOW_TORCH_GLOW_SRC_INPUTMETA_H

#include "GlowCompileSpec.h"

#include "glow/Support/Error.h"

#include <c10/util/ArrayRef.h>

namespace glow {

/// Struct representing the shape and element type of a given input to Glow from
/// PyTorch.
struct InputMeta {
  c10::ScalarType type;
  std::vector<glow::sdim_t> dims;
  double scale;
  int64_t offset;

  InputMeta(c10::ScalarType type_, std::vector<glow::sdim_t> &&dims_,
            double scale_ = 1.0, int64_t offset_ = 0)
      : type(type_), dims(std::move(dims_)), scale(scale_), offset(offset_) {}

  InputMeta(c10::ScalarType type_, const std::vector<glow::sdim_t> &dims_,
            double scale_ = 1.0, int64_t offset_ = 0)
      : type(type_), dims(dims_), scale(scale_), offset(offset_) {}

  InputMeta(c10::ScalarType type_, c10::IntArrayRef dims_, double scale_ = 1.0,
            int64_t offset_ = 0)
      : type(type_),
        dims(std::vector<glow::sdim_t>(dims_.begin(), dims_.end())),
        scale(scale_), offset(offset_) {}

  bool operator==(const InputMeta &other) const {
    return type == other.type && dims == other.dims && scale == other.scale &&
           offset == other.offset;
  }

  /// Produce a printable string for the InputMeta
  std::string print() const;

  /// Hash the InputMeta
  size_t hash() const;
};

/// Struct representing a stack of inputs from PyTorch to Glow
struct InputMetaStack {
  std::vector<InputMeta> inputMetas;

  bool operator==(const InputMetaStack &other) const {
    return inputMetas == other.inputMetas;
  }

  /// Produce a printable string for the InputMetaStack
  std::string print() const;

  /// Hash the InputMetaStack
  size_t hash() const;

  /// This function is to optimize caching map performance.
  /// nominalBatchIdx is index of input to get batchSize.
  /// The return value (batchSize if inputs are valid) will be used as key of
  /// caching graph map
  size_t optimizedHash(int32_t nominalBatchIdx) const;
};

/// \returns a InputMetaStack representing the stack \p inputs or an Error if
/// one occurs. If \p ignoreNonTensors is true then any non-tensor IValues will
/// be ignored, otherwise all IValues on the inputs stack must be PyTorch
/// tensors.
Expected<InputMetaStack>
inputMetaStackFromStack(const c10::ArrayRef<c10::IValue> &inputs,
                        bool ignoreNonTensors = false);

/// Deserialize an InputMeta from string \p raw_data.
InputMetaStack loadInputMeta(const std::string &raw_data);

/// Convert InputSpec vector into InputMetas.
InputMetaStack
getInputMetas(const std::vector<c10::intrusive_ptr<InputSpec>> &inputSet);

} // namespace glow

// custom specialization of std::hash for InputMeta.
namespace std {
template <> struct hash<glow::InputMeta> {
  std::size_t operator()(const glow::InputMeta &meta) const noexcept {
    return meta.hash();
  }
};

// custom specialization of std::hash for InputMetaStack.
template <> struct hash<glow::InputMetaStack> {
  std::size_t operator()(const glow::InputMetaStack &metaStack) const noexcept {
    return metaStack.hash();
  }
};
} // namespace std

#endif // GLOW_TORCH_GLOW_SRC_INPUTMETA_H
