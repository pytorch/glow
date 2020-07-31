// Copyright 2004-present Facebook. All Rights Reserved.

#ifndef GLOW_TORCH_GLOW_SRC_GLOW_COMPILE_SPEC_H
#define GLOW_TORCH_GLOW_SRC_GLOW_COMPILE_SPEC_H

#include "PyTorchCommon.h"
#include <ATen/core/ivalue.h>

namespace glow {

/// Register GlowCompileSpec and related classes as PyTorch custom classes
void registerGlowCompileSpecCustomClass();

/// Specifications for Glow graph input (Tensor meta)
class SpecInputMeta : public torch::jit::CustomClassHolder {
  /// tuple<type, dims>
  using SpecInputMetaSerializationType =
      std::tuple<c10::ScalarType, std::vector<int64_t>>;

public:
  // Constructors
  SpecInputMeta() = default;
  SpecInputMeta(const SpecInputMeta &other);
  SpecInputMeta(std::vector<int64_t> dims, c10::ScalarType type)
      : type_(type), dims_(dims) {}
  SpecInputMeta(const SpecInputMetaSerializationType &state);

  SpecInputMeta &operator=(const SpecInputMeta &other) = default;

  void set(std::vector<int64_t> dims,
           c10::ScalarType type = c10::ScalarType::Float);
  void set_same_as(const at::Tensor &t);

  SpecInputMetaSerializationType serializeToTuple() const;
  // Element type
  c10::ScalarType type() const { return type_; }
  // Tensor dimensions
  const std::vector<int64_t> &dims() const { return dims_; }

private:
  c10::ScalarType type_;
  std::vector<int64_t> dims_;
};

class GlowCompileSpec : public torch::jit::CustomClassHolder {

public:
  GlowCompileSpec() {}
  GlowCompileSpec(std::vector<SpecInputMeta> inputs,
                  PyTorchLoaderSettings settings)
      : inputs_(inputs), settings_(settings) {}
  ~GlowCompileSpec() {}

  void addInputTensor(std::vector<int64_t> dims, c10::ScalarType type);
  void addInputFromTensor(at::Tensor);
  void addInput(c10::intrusive_ptr<SpecInputMeta> input);
  void addInputs(std::vector<c10::intrusive_ptr<SpecInputMeta>> inputs);
  std::vector<SpecInputMeta> inputs() const { return inputs_; }

  torch::Dict<std::string, std::string> serialized_settings() const {
    return settings_.serializeToDict();
  }
  PyTorchLoaderSettings settings() const { return settings_; }
  void set_settings(c10::intrusive_ptr<PyTorchLoaderSettings> settings) {
    settings_ = *settings;
  }

private:
  std::vector<SpecInputMeta> inputs_;
  PyTorchLoaderSettings settings_;
};

} // namespace glow

#endif // GLOW_TORCH_GLOW_SRC_GLOW_COMPILE_SPEC_H
