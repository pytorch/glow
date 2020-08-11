// Copyright 2004-present Facebook. All Rights Reserved.

#include "GlowCompileSpec.h"
#include "PyTorchCommon.h"
#include <ATen/core/ivalue.h>
#include <torch/custom_class.h>

namespace glow {

// tuple<tensor_element_type, tensor_dims>
using SpecInputMetaSerializationType =
    std::tuple<c10::ScalarType, std::vector<int64_t>>;

SpecInputMeta::SpecInputMeta(const SpecInputMeta &other) {
  type_ = other.type_;
  dims_ = other.dims_;
}

SpecInputMeta::SpecInputMeta(const SpecInputMetaSerializationType &state) {
  std::tie(type_, dims_) = state;
}

void SpecInputMeta::set(std::vector<int64_t> dims, c10::ScalarType type) {
  type_ = type;
  dims_ = dims;
}

void SpecInputMeta::set_same_as(const at::Tensor &t) {
  if (t.is_quantized()) {
    throw std::invalid_argument(
        "Quantized PyTorch Tensor is not supported yet in GlowCompileSpec.");
  }
  type_ = t.scalar_type();
  std::copy(t.sizes().begin(), t.sizes().end(), std::back_inserter(dims_));
}

SpecInputMetaSerializationType SpecInputMeta::serializeToTuple() const {
  return make_tuple(type_, dims_);
}

void GlowCompileSpec::addInputTensor(std::vector<int64_t> dims,
                                     c10::ScalarType type) {
  inputs_.emplace_back(SpecInputMeta(std::move(dims), type));
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
  using SettingsSerializationType = torch::Dict<std::string, std::string>;
  static auto glow_options_registry =
      torch::class_<PyTorchLoaderSettings>("glow", "PyTorchLoaderSettings")
          .def(torch::init())
          .def("get_backend_name", &PyTorchLoaderSettings::get_backend_name)
          .def("set_backend_name", &PyTorchLoaderSettings::set_backend_name)
          .def("get_convert_to_fp16",
               &PyTorchLoaderSettings::get_convert_to_fp16)
          .def("set_convert_to_fp16",
               &PyTorchLoaderSettings::set_convert_to_fp16)
          .def("get_convert_fused_to_fp16",
               &PyTorchLoaderSettings::get_convert_fused_to_fp16)
          .def("set_convert_to_fused_fp16",
               &PyTorchLoaderSettings::set_convert_fused_to_fp16)
          .def("get_replication_count",
               &PyTorchLoaderSettings::get_replication_count)
          .def("set_replication_count",
               &PyTorchLoaderSettings::set_replication_count)
          .def("get_saturate_host", &PyTorchLoaderSettings::get_saturate_host)
          .def("set_saturate_host", &PyTorchLoaderSettings::set_saturate_host)
          .def("get_randomize_constants",
               &PyTorchLoaderSettings::get_randomize_constants)
          .def("set_randomize_constants",
               &PyTorchLoaderSettings::set_randomize_constants)
          .def_pickle(
              [](const c10::intrusive_ptr<PyTorchLoaderSettings> &options)
                  -> SettingsSerializationType { // __getstate__
                return options->serializeToDict();
              },
              [](SettingsSerializationType state)
                  -> c10::intrusive_ptr<PyTorchLoaderSettings> { // __setstate__
                return c10::make_intrusive<PyTorchLoaderSettings>(state);
              });

  static auto spec_input_meta_registry =
      torch::class_<SpecInputMeta>("glow", "SpecInputMeta")
          .def(torch::init())
          .def("set", &SpecInputMeta::set)
          .def("set_same_as", &SpecInputMeta::set_same_as)
          .def("type", &SpecInputMeta::type)
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
      std::tuple<std::vector<SpecInputMetaSerializationType>,
                 SettingsSerializationType>;
  static auto glow_compile_spec_registry =
      torch::class_<GlowCompileSpec>("glow", "GlowCompileSpec")
          .def(torch::init())
          .def("addInputTensor", &GlowCompileSpec::addInputTensor)
          .def("addInput", &GlowCompileSpec::addInput)
          .def("addInputs", &GlowCompileSpec::addInputs)
          .def("set_settings", &GlowCompileSpec::set_settings)
          .def_pickle(
              [](const c10::intrusive_ptr<GlowCompileSpec> &gcs)
                  -> GcsSerializationType { // __getstate__
                std::vector<SpecInputMetaSerializationType> inputs;
                for (const auto &meta : gcs->inputs()) {
                  inputs.emplace_back(meta.serializeToTuple());
                }
                SettingsSerializationType settings = gcs->serialized_settings();
                return std::make_tuple(inputs, settings);
              },
              [](GcsSerializationType state)
                  -> c10::intrusive_ptr<GlowCompileSpec> { // __setstate__
                std::vector<SpecInputMetaSerializationType> inputs;
                SettingsSerializationType settings;
                std::tie(inputs, settings) = state;
                std::vector<SpecInputMeta> glowInputs;
                for (auto inputState : inputs) {
                  glowInputs.emplace_back(SpecInputMeta(inputState));
                }
                return c10::make_intrusive<GlowCompileSpec>(GlowCompileSpec(
                    glowInputs, PyTorchLoaderSettings(settings)));
              }

          );
}

} // namespace glow
