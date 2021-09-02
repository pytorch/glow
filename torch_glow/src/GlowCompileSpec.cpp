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

#include "GlowCompileSpec.h"
#include "PyTorchCommon.h"
#include "folly/json.h"
#include <ATen/core/ivalue.h>
#include <torch/custom_class.h>

namespace glow {
namespace {

template <typename ClassType, typename RegistryType>
static void addSerializationDefs(RegistryType &registry) {
  registry.def_pickle(
      [](const c10::intrusive_ptr<ClassType> &value)
          -> std::string { // __getstate__
        auto stringOrErr = value->toJson();
        if (stringOrErr) {
          return *stringOrErr;
        } else {
          const auto errString = ERR_TO_STRING(stringOrErr.takeError());
          throw std::runtime_error(strFormat(
              "Failed to serialize with error: %s", errString.c_str()));
        }
      },
      [](std::string state) -> c10::intrusive_ptr<ClassType> { // __setstate__
        auto value = c10::make_intrusive<ClassType>();
        auto err = value->fromJson(state);
        if (err) {
          const auto errString = ERR_TO_STRING(std::move(err));
          throw std::runtime_error(strFormat(
              "Failed to deserialize with error: %s", errString.c_str()));
        }
        return value;
      });
}
} // namespace

void registerPyTorchGlowCustomClasses() {
#define ADD_BASIC_FIELD_DEFS(registry, ClassType, field_name)                  \
  do {                                                                         \
    (registry).def(strFormat("set_%s", #field_name),                           \
                   &ClassType::set_##field_name);                              \
    (registry).def(strFormat("get_%s", #field_name),                           \
                   &ClassType::get_##field_name);                              \
  } while (0)

#define ADD_MAP_FIELD_DEFS(registry, ClassType, field_name)                    \
  do {                                                                         \
    (registry).def(strFormat("%s_at", #field_name),                            \
                   &ClassType::field_name##_at);                               \
    (registry).def(strFormat("%s_insert", #field_name),                        \
                   &ClassType::field_name##_insert);                           \
  } while (0)

#define ADD_VECTOR_FIELD_DEFS(registry, ClassType, field_name)                 \
  do {                                                                         \
    ADD_BASIC_FIELD_DEFS((registry), ClassType, field_name);                   \
    (registry).def(strFormat("%s_at", #field_name),                            \
                   &ClassType::field_name##_at);                               \
    (registry).def(strFormat("%s_append", #field_name),                        \
                   &ClassType::field_name##_append);                           \
  } while (0)

  // FuserSettings defs
  static auto FuserSettings_registry =
      torch::class_<FuserSettings>("glow", "FuserSettings").def(torch::init());
  ADD_VECTOR_FIELD_DEFS(FuserSettings_registry, FuserSettings, op_blacklist);
  ADD_BASIC_FIELD_DEFS(FuserSettings_registry, FuserSettings,
                       min_fusion_group_size);
  ADD_BASIC_FIELD_DEFS(FuserSettings_registry, FuserSettings,
                       max_fusion_merge_size);
  ADD_BASIC_FIELD_DEFS(FuserSettings_registry, FuserSettings,
                       fusion_start_index);
  ADD_BASIC_FIELD_DEFS(FuserSettings_registry, FuserSettings, fusion_end_index);
  addSerializationDefs<FuserSettings>(FuserSettings_registry);

  // CompilationGroupSettings defs
  static auto CompilationGroupSettings_registry =
      torch::class_<CompilationGroupSettings>("glow",
                                              "CompilationGroupSettings")
          .def(torch::init());
  ADD_BASIC_FIELD_DEFS(CompilationGroupSettings_registry,
                       CompilationGroupSettings, convert_to_fp16);
  ADD_BASIC_FIELD_DEFS(CompilationGroupSettings_registry,
                       CompilationGroupSettings, skip_bias_fp32tofp16_convert);
  ADD_BASIC_FIELD_DEFS(CompilationGroupSettings_registry,
                       CompilationGroupSettings, num_devices_to_use);
  ADD_BASIC_FIELD_DEFS(CompilationGroupSettings_registry,
                       CompilationGroupSettings, replication_count);
  ADD_MAP_FIELD_DEFS(CompilationGroupSettings_registry,
                     CompilationGroupSettings, backend_specific_opts);
  addSerializationDefs<CompilationGroupSettings>(
      CompilationGroupSettings_registry);

  // CompilationSpecSettings defs
  static auto CompilationSpecSettings_registry =
      torch::class_<CompilationSpecSettings>("glow", "CompilationSpecSettings")
          .def(torch::init());
  ADD_BASIC_FIELD_DEFS(CompilationSpecSettings_registry,
                       CompilationSpecSettings, glow_backend);
  ADD_BASIC_FIELD_DEFS(CompilationSpecSettings_registry,
                       CompilationSpecSettings, enable_fuser);
  ADD_BASIC_FIELD_DEFS(CompilationSpecSettings_registry,
                       CompilationSpecSettings, enable_serialize);
  ADD_BASIC_FIELD_DEFS(CompilationSpecSettings_registry,
                       CompilationSpecSettings, enable_deserialize);
  ADD_BASIC_FIELD_DEFS(CompilationSpecSettings_registry,
                       CompilationSpecSettings, use_dag_optimizer);
  ADD_BASIC_FIELD_DEFS(CompilationSpecSettings_registry,
                       CompilationSpecSettings, apl_parallelization_alg);
  ADD_BASIC_FIELD_DEFS(CompilationSpecSettings_registry,
                       CompilationSpecSettings, apl_num_parallel_chunks);
  ADD_BASIC_FIELD_DEFS(CompilationSpecSettings_registry,
                       CompilationSpecSettings, use_max_size_compilation);
  addSerializationDefs<CompilationSpecSettings>(
      CompilationSpecSettings_registry);

  // InputSpec defs
  static auto InputSpec_registry =
      torch::class_<InputSpec>("glow", "InputSpec")
          .def(torch::init())
          .def("get_elem_type", &InputSpec::get_elem_type)
          .def("get_dims", &InputSpec::get_dims)
          .def("set_non_quantized_tensor", &InputSpec::set_non_quantized_tensor)
          .def("set_quantized_tensor", &InputSpec::set_quantized_tensor)
          .def("set_same_as", &InputSpec::set_same_as);
  addSerializationDefs<InputSpec>(InputSpec_registry);

  // CompilationGroup defs
  static auto CompilationGroup_registry =
      torch::class_<CompilationGroup>("glow", "CompilationGroup")
          .def(torch::init())
          .def("get_input_sets", &CompilationGroup::get_input_sets)
          .def("set_input_sets", &CompilationGroup::set_input_sets)
          .def("input_sets_append", &CompilationGroup::input_sets_append);
  ADD_BASIC_FIELD_DEFS(CompilationGroup_registry, CompilationGroup, settings);
  addSerializationDefs<CompilationGroup>(CompilationGroup_registry);

  // CompilationSpec defs
  static auto CompilationSpec_registry =
      torch::class_<CompilationSpec>("glow", "CompilationSpec")
          .def(torch::init());
  ADD_VECTOR_FIELD_DEFS(CompilationSpec_registry, CompilationSpec,
                        compilation_groups);
  ADD_BASIC_FIELD_DEFS(CompilationSpec_registry, CompilationSpec, settings);
  ADD_BASIC_FIELD_DEFS(CompilationSpec_registry, CompilationSpec,
                       fuser_settings);
  ADD_BASIC_FIELD_DEFS(CompilationSpec_registry, CompilationSpec,
                       default_compilation_group_settings);
  addSerializationDefs<CompilationSpec>(CompilationSpec_registry);

#undef ADD_BASIC_FIELD_DEFS
#undef ADD_MAP_FIELD_DEFS
#undef ADD_VECTOR_FIELD_DEFS
}

} // namespace glow
