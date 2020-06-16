// Copyright 2004-present Facebook. All Rights Reserved.

#include "GlowCompileSpec.h"
#include <ATen/core/ivalue.h>
#include <torch/custom_class.h>

namespace glow {

// tuple<tensor_element_type, tensor_dims>
using SpecInputMetaSerializationType =
    std::tuple<std::string, std::vector<int64_t>>;

static c10::ScalarType scalarTypeFromString(const std::string &str) {
  if (str == "float") {
    return c10::ScalarType::Float;
  } else {
    throw std::invalid_argument("Invalid SpecInputMeta type");
  }
}

SpecInputMeta::SpecInputMeta(const SpecInputMeta &other) {
  type_ = other.type_;
  dims_ = other.dims_;
}

SpecInputMeta::SpecInputMeta(const std::string &typeStr,
                             std::vector<int64_t> dims) {
  type_ = scalarTypeFromString(typeStr);
  dims_ = dims;
}

SpecInputMeta::SpecInputMeta(const SpecInputMetaSerializationType &state) {
  std::string typeStr;
  std::tie(typeStr, dims_) = state;
  type_ = scalarTypeFromString(typeStr);
}

void SpecInputMeta::setSpec(const std::string &typeStr,
                            std::vector<int64_t> dims) {
  type_ = scalarTypeFromString(typeStr);
  dims_ = dims;
}

void SpecInputMeta::setSpecFromTensor(const at::Tensor &t) {
  type_ = t.scalar_type();
  std::copy(t.sizes().begin(), t.sizes().end(), std::back_inserter(dims_));
}

SpecInputMetaSerializationType SpecInputMeta::serializeToTuple() const {
  return make_tuple(c10::toString(type_), dims_);
}

void GlowCompileSpec::addInputTensor(const std::string &type,
                                     std::vector<int64_t> dims) {
  inputs_.emplace_back(SpecInputMeta(type, std::move(dims)));
}

void GlowCompileSpec::addInput(c10::intrusive_ptr<SpecInputMeta> input) {
  inputs_.emplace_back(std::move(*input));
}

void GlowCompileSpec::addInputs(
    std::vector<c10::intrusive_ptr<SpecInputMeta>> inputs) {
  for (auto m : inputs) {
    inputs_.emplace_back(std::move(*m));
  }
}

void registerGlowCompileSpecCustomClass() {

  static auto spec_input_meta_registry =
      torch::class_<SpecInputMeta>("glow", "SpecInputMeta")
          .def(torch::init())
          .def("setSpec", &SpecInputMeta::setSpec)
          .def("setSpecFromTensor", &SpecInputMeta::setSpecFromTensor)
          .def_pickle(
              [](const c10::intrusive_ptr<SpecInputMeta> &sim)
                  -> SpecInputMetaSerializationType { // __getstate__
                return sim->serializeToTuple();
              },
              [](SpecInputMetaSerializationType state)
                  -> c10::intrusive_ptr<SpecInputMeta> { // __setstate__
                return c10::make_intrusive<SpecInputMeta>(state);
              });

  using GcsSerializationType =
      std::tuple<std::string, std::vector<SpecInputMetaSerializationType>>;
  static auto glow_compile_spec_registry =
      torch::class_<GlowCompileSpec>("glow", "GlowCompileSpec")
          .def(torch::init())
          .def("setBackend", &GlowCompileSpec::setBackend)
          .def("addInputTensor", &GlowCompileSpec::addInputTensor)
          .def("addInput", &GlowCompileSpec::addInput)
          .def("addInputs", &GlowCompileSpec::addInputs)
          .def_pickle(
              [](const c10::intrusive_ptr<GlowCompileSpec> &gcs)
                  -> GcsSerializationType { // __getstate__
                std::string backend = gcs->getBackend();
                std::vector<SpecInputMetaSerializationType> inputs;
                for (const auto &meta : gcs->inputs()) {
                  inputs.emplace_back(meta.serializeToTuple());
                }
                return std::make_tuple(backend, inputs);
              },
              [](GcsSerializationType state)
                  -> c10::intrusive_ptr<GlowCompileSpec> { // __setstate__
                std::string backend;
                std::vector<SpecInputMetaSerializationType> inputs;
                std::tie(backend, inputs) = state;
                std::vector<SpecInputMeta> glowInputs;
                for (auto inputState : inputs) {
                  glowInputs.emplace_back(SpecInputMeta(inputState));
                }
                return c10::make_intrusive<GlowCompileSpec>(
                    GlowCompileSpec(backend, glowInputs));
              }

          );
}

} // namespace glow
