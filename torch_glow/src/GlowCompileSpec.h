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

#ifndef GLOW_TORCH_GLOW_SRC_GLOW_COMPILE_SPEC_H
#define GLOW_TORCH_GLOW_SRC_GLOW_COMPILE_SPEC_H

#include "PyTorchCommon.h"
#include <ATen/core/ivalue.h>

#include "folly/dynamic.h"
#include "folly/json.h"

namespace glow {

/// Register PyTorch custom classes related to to_glow
void registerPyTorchGlowCustomClasses();

#define CHECK_DYN_IS(dyn, Type)                                                \
  do {                                                                         \
    RETURN_ERR_IF_NOT((dyn).is##Type(),                                        \
                      strFormat("Expected dynamic of type %s but found a %s",  \
                                #Type, (dyn).typeName()));                     \
  } while (0)

#define CHECK_DYN_IS_STRING(dyn) CHECK_DYN_IS(dyn, String)
#define CHECK_DYN_IS_OBJECT(dyn) CHECK_DYN_IS(dyn, Object)
#define CHECK_DYN_IS_BOOL(dyn) CHECK_DYN_IS(dyn, Bool)
#define CHECK_DYN_IS_ARRAY(dyn) CHECK_DYN_IS(dyn, Array)
#define CHECK_DYN_IS_DOUBLE(dyn) CHECK_DYN_IS(dyn, Double)
#define CHECK_DYN_IS_INT(dyn) CHECK_DYN_IS(dyn, Int)

#define CHECK_DYN_CONTAINS_FIELD(dyn, fieldName)                               \
  do {                                                                         \
    RETURN_ERR_IF_NOT(                                                         \
        (dyn).count((fieldName)) > 0,                                          \
        strFormat("Did not find required field %s", (fieldName)));             \
  } while (0)

#define CHECK_DYN_CONTAINS_STRING(dyn, fieldName)                              \
  do {                                                                         \
    CHECK_DYN_CONTAINS_FIELD((dyn), (fieldName));                              \
    RETURN_ERR_IF_NOT(                                                         \
        (dyn).at((fieldName)).isString(),                                      \
        strFormat("Expected field %s to be an string but found a %s",          \
                  (fieldName), (dyn).at((fieldName)).typeName()));             \
  } while (0)

#define CHECK_DYN_CONTAINS_OBJECT(dyn, fieldName)                              \
  do {                                                                         \
    CHECK_DYN_CONTAINS_FIELD((dyn), (fieldName));                              \
    RETURN_ERR_IF_NOT(                                                         \
        (dyn).at((fieldName)).isObject(),                                      \
        strFormat("Expected field %s to be an object but found a %s",          \
                  (fieldName), (dyn).at((fieldName)).typeName()));             \
  } while (0)

#define CHECK_DYN_CONTAINS_BOOL(dyn, fieldName)                                \
  do {                                                                         \
    CHECK_DYN_CONTAINS_FIELD((dyn), (fieldName));                              \
    RETURN_ERR_IF_NOT(                                                         \
        (dyn).at((fieldName)).isBool(),                                        \
        strFormat("Expected field %s to be an bool but found a %s",            \
                  (fieldName), (dyn).at((fieldName)).typeName()));             \
  } while (0)

#define CHECK_DYN_CONTAINS_ARRAY(dyn, fieldName)                               \
  do {                                                                         \
    CHECK_DYN_CONTAINS_FIELD((dyn), (fieldName));                              \
    RETURN_ERR_IF_NOT(                                                         \
        (dyn).at((fieldName)).isArray(),                                       \
        strFormat("Expected field %s to be an array but found a %s",           \
                  (fieldName), (dyn).at((fieldName)).typeName()));             \
  } while (0)

// For checking if contains double, we also consider int as double. This is
// because sometimes when we write a doulbe that doesn't have decimal digit to
// json, the decimal point will be emitted.
#define CHECK_DYN_CONTAINS_DOUBLE(dyn, fieldName)                              \
  do {                                                                         \
    CHECK_DYN_CONTAINS_FIELD((dyn), (fieldName));                              \
    RETURN_ERR_IF_NOT(                                                         \
        (dyn).at((fieldName)).isDouble() || (dyn).at((fieldName)).isInt(),     \
        strFormat("Expected field %s to be an double but found a %s",          \
                  (fieldName), (dyn).at((fieldName)).typeName()));             \
  } while (0)

#define CHECK_DYN_CONTAINS_INT(dyn, fieldName)                                 \
  do {                                                                         \
    CHECK_DYN_CONTAINS_FIELD((dyn), (fieldName));                              \
    RETURN_ERR_IF_NOT(                                                         \
        (dyn).at((fieldName)).isInt(),                                         \
        strFormat("Expected field %s to be an int but found a %s",             \
                  (fieldName), (dyn).at((fieldName)).typeName()));             \
  } while (0)

#define ASSIGN_STRING_FROM_DYN_FIELD_OR_RETURN_ERR(dyn, output, fieldName)     \
  do {                                                                         \
    CHECK_DYN_CONTAINS_STRING((dyn), (fieldName));                             \
    output = (dyn).at(fieldName).getString();                                  \
  } while (0)

#define ASSIGN_BOOL_FROM_DYN_FIELD_OR_RETURN_ERR(dyn, output, fieldName)       \
  do {                                                                         \
    CHECK_DYN_CONTAINS_BOOL((dyn), (fieldName));                               \
    output = (dyn).at(fieldName).getBool();                                    \
  } while (0)

// If the field contains an integer, we'll cast it to double instead of throwing
// error.
#define ASSIGN_DOUBLE_FROM_DYN_FIELD_OR_RETURN_ERR(dyn, output, fieldName)     \
  do {                                                                         \
    CHECK_DYN_CONTAINS_DOUBLE((dyn), (fieldName));                             \
    if ((dyn).at(fieldName).isDouble()) {                                      \
      output = (dyn).at(fieldName).getDouble();                                \
    } else {                                                                   \
      output = static_cast<double>((dyn).at(fieldName).getInt());              \
    }                                                                          \
  } while (0)

#define ASSIGN_INT_FROM_DYN_FIELD_OR_RETURN_ERR(dyn, output, fieldName)        \
  do {                                                                         \
    CHECK_DYN_CONTAINS_INT((dyn), (fieldName));                                \
    output = (dyn).at(fieldName).getInt();                                     \
  } while (0)

template <typename T> struct dynToType;

template <> struct dynToType<std::string> {
  Expected<std::string> operator()(const folly::dynamic &dyn) {
    CHECK_DYN_IS_STRING(dyn);
    return dyn.getString();
  }
};

template <> struct dynToType<bool> {
  Expected<bool> operator()(const folly::dynamic &dyn) {
    CHECK_DYN_IS_BOOL(dyn);
    return dyn.getInt();
  }
};

template <> struct dynToType<double> {
  Expected<bool> operator()(const folly::dynamic &dyn) {
    CHECK_DYN_IS_DOUBLE(dyn);
    return dyn.getDouble();
  }
};

template <> struct dynToType<int64_t> {
  Expected<int64_t> operator()(const folly::dynamic &dyn) {
    CHECK_DYN_IS_INT(dyn);
    return dyn.getInt();
  }
};

template <> struct dynToType<int32_t> {
  Expected<int64_t> operator()(const folly::dynamic &dyn) {
    CHECK_DYN_IS_INT(dyn);
    return int32_t(dyn.getInt());
  }
};

template <typename T>
folly::dynamic dynArrayFromVec(const std::vector<T> &vec) {
  auto dyn = folly::dynamic::array();
  for (const auto &v : vec) {
    dyn.push_back(v);
  }
  return dyn;
}

template <typename T>
Expected<std::vector<T>> dynArrayToVec(const folly::dynamic &dyn) {
  std::vector<T> vec;
  CHECK_DYN_IS_ARRAY(dyn);
  for (const auto &elem : dyn) {
    T typedElem;
    ASSIGN_VALUE_OR_RETURN_ERR(typedElem, dynToType<T>()(elem));
    vec.push_back(typedElem);
  }
  return vec;
}

template <typename K, typename V>
folly::dynamic dynArrayFromMap(const std::map<K, V> &m) {
  auto dyn = folly::dynamic::array();
  for (const auto &kv : m) {
    dyn.push_back(kv.first);
    dyn.push_back(kv.second);
  }
  return dyn;
}

template <typename K, typename V>
Expected<std::map<K, V>> dynArrayToMap(const folly::dynamic &dyn) {
  CHECK_DYN_IS_ARRAY(dyn);
  RETURN_ERR_IF_NOT(dyn.size() % 2 == 0,
                    "Cannot convert a folly::dynamic array into a map if it "
                    "doesn't have an even number of elements");
  std::map<K, V> m;
  for (auto i = 0; i < dyn.size(); i += 2) {
    K typedKey;
    V typedValue;
    ASSIGN_VALUE_OR_RETURN_ERR(typedKey, dynToType<K>()(dyn[i]));
    ASSIGN_VALUE_OR_RETURN_ERR(typedValue, dynToType<V>()(dyn[i + 1]));
    m[typedKey] = typedValue;
  }
  return m;
}

#define ADD_FIELD(field_name, field_type, default_value)                       \
  field_type field_name = default_value;                                       \
  field_type get_##field_name() { return field_name; }                         \
  void set_##field_name(field_type param) { field_name = param; }

#define ADD_STRING_FIELD(field_name, default_value)                            \
  ADD_FIELD(field_name, std::string, default_value)

#define ADD_OBJECT_POINTER_FIELD(field_name, field_type)                       \
  ADD_FIELD(field_name, c10::intrusive_ptr<field_type>,                        \
            c10::make_intrusive<field_type>())

#define ADD_VECTOR_FIELD(field_name, element_type)                             \
  std::vector<element_type> field_name;                                        \
  std::vector<element_type> get_##field_name() { return field_name; }          \
  void set_##field_name(std::vector<element_type> param) {                     \
    field_name = std::move(param);                                             \
  }                                                                            \
  element_type field_name##_at(int64_t i) { return field_name.at(i); }         \
  void field_name##_append(element_type param) {                               \
    field_name.push_back(std::move(param));                                    \
  }

#define ADD_MAP_FIELD(field_name, key_type, value_type)                        \
  std::map<key_type, value_type> field_name;                                   \
  value_type field_name##_at(key_type key) { return field_name.at(key); }      \
  void field_name##_insert(key_type key, value_type value) {                   \
    field_name[key] = std::move(value);                                        \
  }

#define ADD_BOOL_FIELD(field_name, default_value)                              \
  ADD_FIELD(field_name, bool, default_value)

#define ADD_DOUBLE_FIELD(field_name, default_value)                            \
  ADD_FIELD(field_name, double, default_value)

#define ADD_INT_FIELD(field_name, default_value)                               \
  ADD_FIELD(field_name, int64_t, default_value)

struct JsonSerializableCustomClass : public torch::jit::CustomClassHolder {
  virtual int64_t getCurrentBCVersion() const = 0;

  virtual Error validate() const = 0;

  Expected<folly::dynamic> toDynamic() const {
    RETURN_IF_ERR(validate());
    folly::dynamic root = folly::dynamic::object;
    root["_bc_version"] = getCurrentBCVersion();

    folly::dynamic obj = folly::dynamic::object();
    ASSIGN_VALUE_OR_RETURN_ERR(obj, toDynamicImpl());
    root["_object"] = obj;
    return root;
  }

  Error fromDynamic(const folly::dynamic &root) {
    CHECK_DYN_CONTAINS_INT(root, "_bc_version");
    int64_t bc_version_when_serialized = root.at("_bc_version").getInt();

    CHECK_DYN_CONTAINS_OBJECT(root, "_object");
    folly::dynamic obj = root.at("_object");

    RETURN_IF_ERR(fromDynamicImpl(obj, bc_version_when_serialized));
    RETURN_IF_ERR(validate());
    return Error::success();
  }

  Expected<std::string> toJson(bool pretty = false) {
    folly::dynamic root;
    ASSIGN_VALUE_OR_RETURN_ERR(root, toDynamic());
    return pretty ? folly::toPrettyJson(root) : folly::toJson(root);
  }

  Error fromJson(const std::string &json) {
    const auto root = folly::parseJson(json);
    return fromDynamic(root);
  }

private:
  virtual Expected<folly::dynamic> toDynamicImpl() const = 0;

  virtual Error fromDynamicImpl(const folly::dynamic &dyn,
                                int64_t bc_version_when_serialized) = 0;
};

/*
Overview of structures:
for each JIT graph (PyTorch method):
    *CompilationSpec* (1 host manager and 1 CachingGraphRunner)
        [*CompilationGroup*]
            [[*InputSpec*]]
            *CompilationGroupSettings*
                contains compilation specific information like
                fp16 settings, enableRemoveMutation, anything that changes
                the Glow graph compiled for example.
        *CompilationSpecSettings*
            * std::string backendName
            * bool enable_fuser (turn on the fuser for this JIT graph)
        *FuserSettings*
            settings for Glow fuser to use if its enabled
        default_compilation_group_settings: *CompilationGroupSettings*
            Default settings to use when compiling any Glow graphs not specified
            by inputs in a CompilationGroup. Specifically for when the fuser
            runs and needs to compile Glow graphs.

Adding a new field:
* Only add fields for serialization that are needed because they change the
compiled Glow graph and not for debugging.

* Try to be as conservative as possible
about adding new fields since it's much harder to remove than add fields.

* When adding a new field, make sure it is tested in compilation_spec_test

* If a field is added and fromDynamicImpl tries to read from this field by
default then previously serialized models will break so don't do this, this will
require incrementing BC versions and either handling previoulsy serialized
models without the new field separately or deprecating them.
*/

struct FuserSettings : public JsonSerializableCustomClass {
  ADD_VECTOR_FIELD(op_blacklist, std::string)
  ADD_INT_FIELD(min_fusion_group_size, -1)
  ADD_INT_FIELD(max_fusion_merge_size, -1)
  ADD_INT_FIELD(fusion_start_index, -1)
  ADD_INT_FIELD(fusion_end_index, -1)

  Error fromDynamicImpl(const folly::dynamic &dyn,
                        int64_t bc_version_when_serialized) override {
    RETURN_ERR_IF_NOT(
        bc_version_when_serialized == 0,
        strFormat("Only bc_version 0 is supported, got bc_version %d",
                  int(bc_version_when_serialized)));

    if (dyn.count("op_blacklist")) {
      CHECK_DYN_CONTAINS_ARRAY(dyn, "op_blacklist");
      ASSIGN_VALUE_OR_RETURN_ERR(
          op_blacklist, dynArrayToVec<std::string>(dyn.at("op_blacklist")));
    }

    if (dyn.count("min_fusion_group_size")) {
      ASSIGN_INT_FROM_DYN_FIELD_OR_RETURN_ERR(dyn, min_fusion_group_size,
                                              "min_fusion_group_size");
    }

    if (dyn.count("max_fusion_merge_size")) {
      ASSIGN_INT_FROM_DYN_FIELD_OR_RETURN_ERR(dyn, max_fusion_merge_size,
                                              "max_fusion_merge_size");
    }

    if (dyn.count("fusion_start_index")) {
      ASSIGN_INT_FROM_DYN_FIELD_OR_RETURN_ERR(dyn, fusion_start_index,
                                              "fusion_start_index");
    }

    if (dyn.count("fusion_end_index")) {
      ASSIGN_INT_FROM_DYN_FIELD_OR_RETURN_ERR(dyn, fusion_end_index,
                                              "fusion_end_index");
    }

    return Error::success();
  }

  Expected<folly::dynamic> toDynamicImpl() const override {
    folly::dynamic obj = folly::dynamic::object();
    obj["op_blacklist"] = dynArrayFromVec(op_blacklist);
    obj["min_fusion_group_size"] = min_fusion_group_size;
    obj["max_fusion_merge_size"] = max_fusion_merge_size;
    obj["fusion_start_index"] = fusion_start_index;
    obj["fusion_end_index"] = fusion_end_index;
    return obj;
  }

  int64_t getCurrentBCVersion() const override { return 0; }

  Error validate() const override { return Error::success(); }
};

struct CompilationGroupSettings : public JsonSerializableCustomClass {
  ADD_BOOL_FIELD(convert_to_fp16, false)
  ADD_BOOL_FIELD(skip_bias_fp32tofp16_convert, false)
  // -1 indicates use all available devices
  ADD_INT_FIELD(num_devices_to_use, -1)
  ADD_INT_FIELD(replication_count, 1)
  ADD_MAP_FIELD(backend_specific_opts, std::string, std::string)

  Expected<folly::dynamic> toDynamicImpl() const override {
    folly::dynamic obj = folly::dynamic::object();
    obj["convert_to_fp16"] = convert_to_fp16;
    obj["skip_bias_fp32tofp16_convert"] = skip_bias_fp32tofp16_convert;
    obj["num_devices_to_use"] = num_devices_to_use;
    obj["replication_count"] = replication_count;
    obj["backend_specific_opts"] = dynArrayFromMap(backend_specific_opts);
    return obj;
  }

  Error fromDynamicImpl(const folly::dynamic &dyn,
                        int64_t bc_version_when_serialized) override {
    RETURN_ERR_IF_NOT(
        bc_version_when_serialized == 0,
        strFormat("Only bc_version 0 is supported, got bc_version %d",
                  int(bc_version_when_serialized)));

    if (dyn.count("convert_to_fp16")) {
      ASSIGN_BOOL_FROM_DYN_FIELD_OR_RETURN_ERR(dyn, convert_to_fp16,
                                               "convert_to_fp16");
    }

    if (dyn.count("skip_bias_fp32tofp16_convert")) {
      ASSIGN_BOOL_FROM_DYN_FIELD_OR_RETURN_ERR(
          dyn, skip_bias_fp32tofp16_convert, "skip_bias_fp32tofp16_convert");
    }

    if (dyn.count("num_devices_to_use")) {
      ASSIGN_INT_FROM_DYN_FIELD_OR_RETURN_ERR(dyn, num_devices_to_use,
                                              "num_devices_to_use");
    }

    if (dyn.count("replication_count")) {
      ASSIGN_INT_FROM_DYN_FIELD_OR_RETURN_ERR(dyn, replication_count,
                                              "replication_count");
    }

    if (dyn.count("backend_specific_opts")) {
      // Maps are serialized as Arrays
      CHECK_DYN_CONTAINS_ARRAY(dyn, "backend_specific_opts");
      ASSIGN_VALUE_OR_RETURN_ERR(backend_specific_opts,
                                 (dynArrayToMap<std::string, std::string>(
                                     dyn.at("backend_specific_opts"))));
    }

    return Error::success();
  }

  int64_t getCurrentBCVersion() const override { return 0; }

  Error validate() const override {
    RETURN_ERR_IF_NOT(
        replication_count > 0,
        strFormat("replication_count must be greater than 0 but got %d",
                  int(replication_count)));

    return Error::success();
  }
};

struct CompilationSpecSettings : public JsonSerializableCustomClass {
  ADD_STRING_FIELD(glow_backend, "")
  ADD_BOOL_FIELD(enable_fuser, false)
  ADD_BOOL_FIELD(enable_serialize, false)
  ADD_BOOL_FIELD(enable_deserialize, false)
  ADD_BOOL_FIELD(use_dag_optimizer, false)
  ADD_STRING_FIELD(apl_parallelization_alg, "ParallelizeCVHeuristicData")
  ADD_INT_FIELD(apl_num_parallel_chunks, 2)
  ADD_BOOL_FIELD(use_max_size_compilation, false)

  Expected<folly::dynamic> toDynamicImpl() const override {
    folly::dynamic obj = folly::dynamic::object();
    obj["glow_backend"] = glow_backend;
    obj["enable_fuser"] = enable_fuser;
    obj["use_dag_optimizer"] = use_dag_optimizer;
    obj["enable_serialize"] = enable_serialize;
    obj["enable_deserialize"] = enable_deserialize;
    obj["apl_parallelization_alg"] = apl_parallelization_alg;
    obj["apl_num_parallel_chunks"] = apl_num_parallel_chunks;
    obj["use_max_size_compilation"] = use_max_size_compilation;
    return obj;
  }

  Error fromDynamicImpl(const folly::dynamic &dyn,
                        int64_t bc_version_when_serialized) override {
    RETURN_ERR_IF_NOT(
        bc_version_when_serialized == 0,
        strFormat("Only bc_version 0 is supported, got bc_version %d",
                  int(bc_version_when_serialized)));

    ASSIGN_STRING_FROM_DYN_FIELD_OR_RETURN_ERR(dyn, glow_backend,
                                               "glow_backend");

    if (dyn.count("enable_fuser")) {
      ASSIGN_BOOL_FROM_DYN_FIELD_OR_RETURN_ERR(dyn, enable_fuser,
                                               "enable_fuser");
    }

    if (dyn.count("enable_serialize")) {
      ASSIGN_BOOL_FROM_DYN_FIELD_OR_RETURN_ERR(dyn, enable_serialize,
                                               "enable_serialize");
    }

    if (dyn.count("enable_deserialize")) {
      ASSIGN_BOOL_FROM_DYN_FIELD_OR_RETURN_ERR(dyn, enable_deserialize,
                                               "enable_deserialize");
    }

    if (dyn.count("use_dag_optimizer")) {
      ASSIGN_BOOL_FROM_DYN_FIELD_OR_RETURN_ERR(dyn, use_dag_optimizer,
                                               "use_dag_optimizer");
    }

    if (dyn.count("apl_parallelization_alg")) {
      ASSIGN_STRING_FROM_DYN_FIELD_OR_RETURN_ERR(dyn, apl_parallelization_alg,
                                                 "apl_parallelization_alg");
    }

    if (dyn.count("apl_num_parallel_chunks")) {
      ASSIGN_INT_FROM_DYN_FIELD_OR_RETURN_ERR(dyn, apl_num_parallel_chunks,
                                              "apl_num_parallel_chunks");
    }

    if (dyn.count("use_max_size_compilation")) {
      ASSIGN_BOOL_FROM_DYN_FIELD_OR_RETURN_ERR(dyn, use_max_size_compilation,
                                               "use_max_size_compilation");
    }

    return Error::success();
  }

  int64_t getCurrentBCVersion() const override { return 0; }

  Error validate() const override {
    RETURN_ERR_IF_NOT(!glow_backend.empty(), "glow_backend must be specified");
    return Error::success();
  }
};

struct InputSpec : public JsonSerializableCustomClass {
  // initialized field is only used for validation to ensure that the InputSpec
  // was initialized properly before use.
  bool initialized = false;
  c10::ScalarType elem_type;
  std::vector<int64_t> dims;
  double scale;
  int64_t offset;

  c10::ScalarType get_elem_type() { return elem_type; }
  std::vector<int64_t> get_dims() { return dims; }
  double get_scale() { return scale; }
  int64_t get_offset() { return offset; }

  void set_quantized_tensor(std::vector<int64_t> dims_param,
                            c10::ScalarType elem_type_param, double scale_param,
                            int64_t offset_param) {
    CHECK(c10::isQIntType(elem_type_param))
        << "Trying to set quantized tensor but elem_type is not quantized "
           "type.";
    dims = dims_param;
    elem_type = elem_type_param;
    scale = scale_param;
    offset = offset_param;
    initialized = true;
  }

  void set_non_quantized_tensor(std::vector<int64_t> dims_param,
                                c10::ScalarType elem_type_param) {
    CHECK(!c10::isQIntType(elem_type_param))
        << "Trying to set non quantized tensor but elem_type is quantized "
           "type.";
    dims = dims_param;
    elem_type = elem_type_param;
    scale = 1.0;
    offset = 0;
    initialized = true;
  }

  void set_same_as(const at::Tensor &t) {
    std::vector<int64_t> sizesCopy;
    std::copy(t.sizes().begin(), t.sizes().end(),
              std::back_inserter(sizesCopy));

    if (t.is_quantized()) {
      CHECK(t.qscheme() == at::kPerTensorAffine ||
            t.qscheme() == at::kPerTensorSymmetric)
          << "Expect per_tensor quantization scheme";
      set_quantized_tensor(sizesCopy, t.scalar_type(), t.q_scale(),
                           t.q_zero_point());
    } else {
      set_non_quantized_tensor(sizesCopy, t.scalar_type());
    }
  }

  Expected<folly::dynamic> toDynamicImpl() const override {
    folly::dynamic obj = folly::dynamic::object();
    obj["elem_type"] = static_cast<int64_t>(elem_type);
    obj["dims"] = dynArrayFromVec(dims);

    if (c10::isQIntType(elem_type)) {
      obj["scale"] = scale;
      obj["offset"] = offset;
    }

    return obj;
  }

  Error fromDynamicImpl(const folly::dynamic &dyn,
                        int64_t bc_version_when_serialized) override {
    RETURN_ERR_IF_NOT(
        bc_version_when_serialized == 0,
        strFormat("Only bc_version 0 is supported, got bc_version %d",
                  int(bc_version_when_serialized)));

    int64_t elem_type_temp;
    ASSIGN_INT_FROM_DYN_FIELD_OR_RETURN_ERR(dyn, elem_type_temp, "elem_type");

    elem_type = static_cast<c10::ScalarType>(elem_type_temp);

    CHECK_DYN_CONTAINS_ARRAY(dyn, "dims");
    ASSIGN_VALUE_OR_RETURN_ERR(dims, dynArrayToVec<int64_t>(dyn.at("dims")));

    if (dyn.count("scale")) {
      ASSIGN_DOUBLE_FROM_DYN_FIELD_OR_RETURN_ERR(dyn, scale, "scale");
      ASSIGN_INT_FROM_DYN_FIELD_OR_RETURN_ERR(dyn, offset, "offset");
    }

    // initialized should be set at the last so that it doesn't get set
    // prematurely.
    initialized = true;
    return Error::success();
  }

  int64_t getCurrentBCVersion() const override { return 0; }

  Error validate() const override {
    RETURN_ERR_IF_NOT(initialized, "InputSpec was not initialized");
    return Error::success();
  }
};

struct CompilationGroup : public JsonSerializableCustomClass {
  ADD_OBJECT_POINTER_FIELD(settings, CompilationGroupSettings)
  std::vector<std::vector<c10::intrusive_ptr<InputSpec>>> input_sets;

  // input_sets
  std::vector<std::vector<c10::intrusive_ptr<InputSpec>>> get_input_sets() {
    return input_sets;
  }

  void set_input_sets(std::vector<std::vector<c10::intrusive_ptr<InputSpec>>>
                          input_sets_param) {
    input_sets = input_sets_param;
  }

  void input_sets_append(std::vector<c10::intrusive_ptr<InputSpec>> input_set) {
    input_sets.push_back(input_set);
  }

  Expected<folly::dynamic> toDynamicImpl() const override {
    folly::dynamic obj = folly::dynamic::object();

    folly::dynamic settings_dyn;
    ASSIGN_VALUE_OR_RETURN_ERR(settings_dyn, settings->toDynamic());
    obj["settings"] = settings_dyn;

    folly::dynamic input_sets_dyn = folly::dynamic::array();
    for (const auto &input_set : input_sets) {
      folly::dynamic input_set_dyn = folly::dynamic::array();
      for (const auto &input_spec : input_set) {
        folly::dynamic input_spec_dyn;
        ASSIGN_VALUE_OR_RETURN_ERR(input_spec_dyn, input_spec->toDynamic());
        input_set_dyn.push_back(input_spec_dyn);
      }
      input_sets_dyn.push_back(input_set_dyn);
    }

    obj["input_sets"] = std::move(input_sets_dyn);

    return obj;
  }

  Error fromDynamicImpl(const folly::dynamic &dyn,
                        int64_t bc_version_when_serialized) override {
    RETURN_ERR_IF_NOT(
        bc_version_when_serialized == 0,
        strFormat("Only bc_version 0 is supported, got bc_version %d",
                  int(bc_version_when_serialized)));

    CHECK_DYN_CONTAINS_OBJECT(dyn, "settings");
    RETURN_IF_ERR(settings->fromDynamic(dyn.at("settings")));

    CHECK_DYN_CONTAINS_ARRAY(dyn, "input_sets");
    auto &input_sets_dyn = dyn.at("input_sets");
    for (const auto &input_set_dyn : input_sets_dyn) {
      std::vector<c10::intrusive_ptr<InputSpec>> input_set;
      for (const auto &input_spec_dyn : input_set_dyn) {
        auto input_spec = c10::make_intrusive<InputSpec>();
        RETURN_IF_ERR(input_spec->fromDynamic(input_spec_dyn));
        input_set.push_back(std::move(input_spec));
      }
      input_sets.push_back(std::move(input_set));
    }

    return Error::success();
  }

  int64_t getCurrentBCVersion() const override { return 0; }

  Error validate() const override {
    RETURN_ERR_IF_NOT(settings.defined(), "settings is undefined");
    RETURN_IF_ERR(settings->validate());
    for (const auto &input_set : input_sets) {
      for (const auto &input_spec : input_set) {
        RETURN_ERR_IF_NOT(input_spec.defined(), "input_spec is undefined");
        RETURN_IF_ERR(input_spec->validate());
      }
    }
    return Error::success();
  }
};

struct CompilationSpec : public JsonSerializableCustomClass {
  ADD_VECTOR_FIELD(compilation_groups, c10::intrusive_ptr<CompilationGroup>)
  ADD_OBJECT_POINTER_FIELD(settings, CompilationSpecSettings)
  ADD_OBJECT_POINTER_FIELD(fuser_settings, FuserSettings)
  // Settings to be used by default when no explicit compilation group applies
  // like when the fuser is enabled.
  ADD_OBJECT_POINTER_FIELD(default_compilation_group_settings,
                           CompilationGroupSettings)

  Expected<folly::dynamic> toDynamicImpl() const override {
    folly::dynamic obj = folly::dynamic::object();

    folly::dynamic compilation_groups_dyn = folly::dynamic::array();
    for (const auto &compilation_group : compilation_groups) {
      folly::dynamic compilation_group_dyn;
      ASSIGN_VALUE_OR_RETURN_ERR(compilation_group_dyn,
                                 compilation_group->toDynamic());
      compilation_groups_dyn.push_back(compilation_group_dyn);
    }
    obj["compilation_groups"] = std::move(compilation_groups_dyn);

    folly::dynamic settings_dyn;
    ASSIGN_VALUE_OR_RETURN_ERR(settings_dyn, settings->toDynamic());
    obj["settings"] = settings_dyn;

    folly::dynamic fuser_settings_dyn;
    ASSIGN_VALUE_OR_RETURN_ERR(fuser_settings_dyn, fuser_settings->toDynamic());
    obj["fuser_settings"] = fuser_settings_dyn;

    folly::dynamic default_compilation_group_settings_dyn;
    ASSIGN_VALUE_OR_RETURN_ERR(default_compilation_group_settings_dyn,
                               default_compilation_group_settings->toDynamic());
    obj["default_compilation_group_settings"] =
        default_compilation_group_settings_dyn;

    return obj;
  }

  Error fromDynamicImpl(const folly::dynamic &dyn,
                        int64_t bc_version_when_serialized) override {
    RETURN_ERR_IF_NOT(
        bc_version_when_serialized == 0,
        strFormat("Only bc_version 0 is supported, got bc_version %d",
                  int(bc_version_when_serialized)));

    CHECK_DYN_CONTAINS_ARRAY(dyn, "compilation_groups");
    for (const auto &compilation_group_dyn : dyn.at("compilation_groups")) {
      auto compilation_group = c10::make_intrusive<CompilationGroup>();
      RETURN_IF_ERR(compilation_group->fromDynamic(compilation_group_dyn));
      compilation_groups.push_back(compilation_group);
    }

    CHECK_DYN_CONTAINS_OBJECT(dyn, "settings");
    RETURN_IF_ERR(settings->fromDynamic(dyn.at("settings")));

    CHECK_DYN_CONTAINS_OBJECT(dyn, "fuser_settings");
    RETURN_IF_ERR(fuser_settings->fromDynamic(dyn.at("fuser_settings")));

    CHECK_DYN_CONTAINS_OBJECT(dyn, "default_compilation_group_settings");
    RETURN_IF_ERR(default_compilation_group_settings->fromDynamic(
        dyn.at("default_compilation_group_settings")));

    return Error::success();
  }

  int64_t getCurrentBCVersion() const override { return 0; }

  Error validate() const override {
    RETURN_ERR_IF_NOT(settings.defined(), "settings is undefined");
    RETURN_IF_ERR(settings->validate());

    RETURN_ERR_IF_NOT(fuser_settings.defined(), "fuser_settings is undefined");
    RETURN_IF_ERR(fuser_settings->validate());

    RETURN_ERR_IF_NOT(default_compilation_group_settings.defined(),
                      "default_compilation_group_settings is undefined");
    RETURN_IF_ERR(default_compilation_group_settings->validate());

    for (const auto &compilation_group : compilation_groups) {
      RETURN_ERR_IF_NOT(compilation_group.defined(),
                        "compilation_group is undefined");
      RETURN_IF_ERR(compilation_group->validate());
    }
    return Error::success();
  }
};

// This class is a wrapper of PyTorchLoaderSettings:
// https://fburl.com/diffusion/fzhkej4j. We make it serializable for AOT
// compilation: https://fb.quip.com/ABT1A021txMc
struct GlowPyTorchLoaderSettings : public JsonSerializableCustomClass {
  PyTorchLoaderSettings settings_;
  GlowPyTorchLoaderSettings(const PyTorchLoaderSettings &settings =
                                glow::getGlobalPyTorchLoaderSettingsSnapshot())
      : settings_(settings) {}

  PyTorchLoaderSettings getSettings() { return settings_; }
  void overrideSettings(const PyTorchLoaderSettings &settings) {
    settings_ = settings;
  }

  Expected<folly::dynamic> toDynamicImpl() const override {
    folly::dynamic obj = folly::dynamic::object();
    obj["fusionPassEnabled"] = settings_.fusionPassEnabled;
    obj["dumpGlowDag"] = settings_.dumpGlowDag;
    // In Glow serialization, we only use opBlocklistStrVec to save the op block
    // list to avoid duplication.
    std::vector<std::string> opBlocklistStrVec;
    for (const auto &op : settings_.opBlocklist) {
      opBlocklistStrVec.emplace_back(op.toQualString());
    }
    obj["opBlocklistStrVec"] = dynArrayFromVec(opBlocklistStrVec);
    obj["minFusionGroupSize"] = settings_.minFusionGroupSize;
    obj["maxFusionMergeSize"] = settings_.maxFusionMergeSize;
    obj["fusionStartIndex"] = settings_.fusionStartIndex;
    obj["fusionEndIndex"] = settings_.fusionEndIndex;
    obj["convertToFP16"] = settings_.convertToFP16;
    obj["convertFusedToFP16"] = settings_.convertFusedToFP16;
    obj["printJITIndex"] = settings_.printJITIndex;
    obj["ignoreDivRoundingArgs"] = settings_.ignoreDivRoundingArgs;
    obj["clipFP16"] = settings_.clipFP16;
    obj["clipFP16SkipInputs"] = settings_.clipFP16SkipInputs;
    obj["convertPlaceholdersToFP16"] = settings_.convertPlaceholdersToFP16;
    obj["convertConstantsToFP16"] = settings_.convertConstantsToFP16;
    obj["forceFP16AccumSLS"] = settings_.forceFP16AccumSLS;
    obj["dumpFinalGlowGraph"] = settings_.dumpFinalGlowGraph;
    obj["enableGlowTracing"] = settings_.enableGlowTracing;
    obj["enableRemoveMutation"] = settings_.enableRemoveMutation;
    obj["disableLayoutVerifying"] = settings_.disableLayoutVerifying;
    obj["dumpOperatorInventory"] = settings_.dumpOperatorInventory;
    obj["numTracesPerDump"] = settings_.numTracesPerDump;
    obj["replicationCount"] = settings_.replicationCount;
    obj["backendSpecificOpts"] = dynArrayFromMap(settings_.backendSpecificOpts);
    obj["writeToOnnx"] = settings_.writeToOnnx;
    obj["onnxZipMode"] = settings_.onnxZipMode;
    obj["writeOnnxToTmp"] = settings_.writeOnnxToTmp;
    obj["onnxFileNamePrefix"] = settings_.onnxFileNamePrefix;
    obj["jitVsGlowCompare"] = settings_.jitVsGlowCompare;
    obj["backendOptionsFile"] = settings_.backendOptionsFile;
    obj["saturateHost"] = settings_.saturateHost;
    obj["saturateKDevices"] = settings_.saturateKDevices;
    obj["randomizeConstants"] = settings_.randomizeConstants;
    obj["writeWithoutRandomize"] = settings_.writeWithoutRandomize;
    obj["backendName"] = settings_.backendName;
    obj["numDevices"] = settings_.numDevices;
    obj["scanDevices"] = settings_.scanDevices;
    obj["runShapeInference"] = settings_.runShapeInference;
    obj["enableDebugFuser"] = settings_.enableDebugFuser;
    obj["setIncludeLastOffsets"] = settings_.setIncludeLastOffsets;
    obj["debugContinuouslyVerifyDuringModelLoading"] =
        settings_.debugContinuouslyVerifyDuringModelLoading;
    obj["nominalBatchIdx"] = settings_.nominalBatchIdx;
    obj["availableDevices"] = dynArrayFromVec(settings_.availableDevices);
    obj["dumpFailedInputsToOnnxFiles"] = settings_.dumpFailedInputsToOnnxFiles;
    obj["lazyCompile"] = settings_.lazyCompile;
    obj["enableDeviceTracing"] = settings_.enableDeviceTracing;
    obj["use_dag_optimizer"] = settings_.use_dag_optimizer;
    obj["apl_parallelization_alg"] = settings_.apl_parallelization_alg;
    obj["apl_num_parallel_chunks"] = settings_.apl_num_parallel_chunks;
    obj["saveGlowIRIntoONNX"] = settings_.saveGlowIRIntoONNX;
    obj["loadGlowIRFromONNX"] = settings_.loadGlowIRFromONNX;
    obj["skipProvisioning"] = settings_.skipProvisioning;
    obj["debugLayers"] = settings_.debugLayers;
    return obj;
  }

  Error fromDynamicImpl(const folly::dynamic &dyn,
                        int64_t bc_version_when_serialized) override {
    RETURN_ERR_IF_NOT(
        bc_version_when_serialized == 0,
        strFormat("Only bc_version 0 is supported, got bc_version %d",
                  int(bc_version_when_serialized)));

    if (dyn.count("fusionPassEnabled")) {
      ASSIGN_BOOL_FROM_DYN_FIELD_OR_RETURN_ERR(dyn, settings_.fusionPassEnabled,
                                               "fusionPassEnabled");
    }
    if (dyn.count("dumpGlowDag")) {
      ASSIGN_BOOL_FROM_DYN_FIELD_OR_RETURN_ERR(dyn, settings_.dumpGlowDag,
                                               "dumpGlowDag");
    }
    // In Glow deserialization, we first deserialize opBlocklistStrVec to get
    // the op black list, then use the list to initialize opBlockList attribute,
    if (dyn.count("opBlocklistStrVec")) {
      CHECK_DYN_CONTAINS_ARRAY(dyn, "opBlocklistStrVec");
      std::vector<std::string> opBlocklistStrVec;
      ASSIGN_VALUE_OR_RETURN_ERR(
          opBlocklistStrVec,
          dynArrayToVec<std::string>(dyn.at("opBlocklistStrVec")));
      for (const auto &opStr : opBlocklistStrVec) {
        settings_.opBlocklist.insert(torch::jit::Symbol::fromQualString(opStr));
      }
    }
    if (dyn.count("minFusionGroupSize")) {
      ASSIGN_INT_FROM_DYN_FIELD_OR_RETURN_ERR(dyn, settings_.minFusionGroupSize,
                                              "minFusionGroupSize");
    }
    if (dyn.count("maxFusionMergeSize")) {
      ASSIGN_INT_FROM_DYN_FIELD_OR_RETURN_ERR(dyn, settings_.maxFusionMergeSize,
                                              "maxFusionMergeSize");
    }
    if (dyn.count("fusionStartIndex")) {
      ASSIGN_INT_FROM_DYN_FIELD_OR_RETURN_ERR(dyn, settings_.fusionStartIndex,
                                              "fusionStartIndex");
    }
    if (dyn.count("fusionEndIndex")) {
      ASSIGN_INT_FROM_DYN_FIELD_OR_RETURN_ERR(dyn, settings_.fusionEndIndex,
                                              "fusionEndIndex");
    }
    if (dyn.count("convertToFP16")) {
      ASSIGN_BOOL_FROM_DYN_FIELD_OR_RETURN_ERR(dyn, settings_.convertToFP16,
                                               "convertToFP16");
    }
    if (dyn.count("convertFusedToFP16")) {
      ASSIGN_BOOL_FROM_DYN_FIELD_OR_RETURN_ERR(
          dyn, settings_.convertFusedToFP16, "convertFusedToFP16");
    }
    if (dyn.count("printJITIndex")) {
      ASSIGN_BOOL_FROM_DYN_FIELD_OR_RETURN_ERR(dyn, settings_.printJITIndex,
                                               "printJITIndex");
    }
    if (dyn.count("ignoreDivRoundingArgs")) {
      ASSIGN_BOOL_FROM_DYN_FIELD_OR_RETURN_ERR(
          dyn, settings_.ignoreDivRoundingArgs, "ignoreDivRoundingArgs");
    }
    if (dyn.count("clipFP16")) {
      ASSIGN_BOOL_FROM_DYN_FIELD_OR_RETURN_ERR(dyn, settings_.clipFP16,
                                               "clipFP16");
    }
    if (dyn.count("clipFP16SkipInputs")) {
      ASSIGN_BOOL_FROM_DYN_FIELD_OR_RETURN_ERR(
          dyn, settings_.clipFP16SkipInputs, "clipFP16SkipInputs");
    }
    if (dyn.count("convertPlaceholdersToFP16")) {
      ASSIGN_BOOL_FROM_DYN_FIELD_OR_RETURN_ERR(
          dyn, settings_.convertPlaceholdersToFP16,
          "convertPlaceholdersToFP16");
    }
    if (dyn.count("convertConstantsToFP16")) {
      ASSIGN_BOOL_FROM_DYN_FIELD_OR_RETURN_ERR(
          dyn, settings_.convertConstantsToFP16, "convertConstantsToFP16");
    }
    if (dyn.count("forceFP16AccumSLS")) {
      ASSIGN_BOOL_FROM_DYN_FIELD_OR_RETURN_ERR(dyn, settings_.forceFP16AccumSLS,
                                               "forceFP16AccumSLS");
    }
    if (dyn.count("dumpFinalGlowGraph")) {
      ASSIGN_BOOL_FROM_DYN_FIELD_OR_RETURN_ERR(
          dyn, settings_.dumpFinalGlowGraph, "dumpFinalGlowGraph");
    }
    if (dyn.count("enableGlowTracing")) {
      ASSIGN_BOOL_FROM_DYN_FIELD_OR_RETURN_ERR(dyn, settings_.enableGlowTracing,
                                               "enableGlowTracing");
    }
    if (dyn.count("enableRemoveMutation")) {
      ASSIGN_BOOL_FROM_DYN_FIELD_OR_RETURN_ERR(
          dyn, settings_.enableRemoveMutation, "enableRemoveMutation");
    }
    if (dyn.count("disableLayoutVerifying")) {
      ASSIGN_BOOL_FROM_DYN_FIELD_OR_RETURN_ERR(
          dyn, settings_.disableLayoutVerifying, "disableLayoutVerifying");
    }
    if (dyn.count("dumpOperatorInventory")) {
      ASSIGN_BOOL_FROM_DYN_FIELD_OR_RETURN_ERR(
          dyn, settings_.dumpOperatorInventory, "dumpOperatorInventory");
    }
    if (dyn.count("numTracesPerDump")) {
      ASSIGN_INT_FROM_DYN_FIELD_OR_RETURN_ERR(dyn, settings_.numTracesPerDump,
                                              "numTracesPerDump");
    }
    if (dyn.count("replicationCount")) {
      ASSIGN_INT_FROM_DYN_FIELD_OR_RETURN_ERR(dyn, settings_.replicationCount,
                                              "replicationCount");
    }
    if (dyn.count("backendSpecificOpts")) {
      // Maps are serialized as Arrays
      CHECK_DYN_CONTAINS_ARRAY(dyn, "backendSpecificOpts");
      ASSIGN_VALUE_OR_RETURN_ERR(settings_.backendSpecificOpts,
                                 (dynArrayToMap<std::string, std::string>(
                                     dyn.at("backendSpecificOpts"))));
    }
    if (dyn.count("writeToOnnx")) {
      ASSIGN_BOOL_FROM_DYN_FIELD_OR_RETURN_ERR(dyn, settings_.writeToOnnx,
                                               "writeToOnnx");
    }
    if (dyn.count("onnxZipMode")) {
      ASSIGN_BOOL_FROM_DYN_FIELD_OR_RETURN_ERR(dyn, settings_.onnxZipMode,
                                               "onnxZipMode");
    }
    if (dyn.count("writeOnnxToTmp")) {
      ASSIGN_BOOL_FROM_DYN_FIELD_OR_RETURN_ERR(dyn, settings_.writeOnnxToTmp,
                                               "writeOnnxToTmp");
    }
    if (dyn.count("onnxFileNamePrefix")) {
      ASSIGN_STRING_FROM_DYN_FIELD_OR_RETURN_ERR(
          dyn, settings_.onnxFileNamePrefix, "onnxFileNamePrefix");
    }
    if (dyn.count("jitVsGlowCompare")) {
      ASSIGN_BOOL_FROM_DYN_FIELD_OR_RETURN_ERR(dyn, settings_.jitVsGlowCompare,
                                               "jitVsGlowCompare");
    }
    if (dyn.count("backendOptionsFile")) {
      ASSIGN_STRING_FROM_DYN_FIELD_OR_RETURN_ERR(
          dyn, settings_.backendOptionsFile, "backendOptionsFile");
    }
    if (dyn.count("saturateHost")) {
      ASSIGN_BOOL_FROM_DYN_FIELD_OR_RETURN_ERR(dyn, settings_.saturateHost,
                                               "saturateHost");
    }
    if (dyn.count("saturateKDevices")) {
      ASSIGN_INT_FROM_DYN_FIELD_OR_RETURN_ERR(dyn, settings_.saturateKDevices,
                                              "saturateKDevices");
    }
    if (dyn.count("randomizeConstants")) {
      ASSIGN_BOOL_FROM_DYN_FIELD_OR_RETURN_ERR(
          dyn, settings_.randomizeConstants, "randomizeConstants");
    }
    if (dyn.count("writeWithoutRandomize")) {
      ASSIGN_BOOL_FROM_DYN_FIELD_OR_RETURN_ERR(
          dyn, settings_.writeWithoutRandomize, "writeWithoutRandomize");
    }
    if (dyn.count("backendName")) {
      ASSIGN_STRING_FROM_DYN_FIELD_OR_RETURN_ERR(dyn, settings_.backendName,
                                                 "backendName");
    }
    if (dyn.count("numDevices")) {
      ASSIGN_INT_FROM_DYN_FIELD_OR_RETURN_ERR(dyn, settings_.numDevices,
                                              "numDevices");
    }
    if (dyn.count("scanDevices")) {
      ASSIGN_BOOL_FROM_DYN_FIELD_OR_RETURN_ERR(dyn, settings_.scanDevices,
                                               "scanDevices");
    }
    if (dyn.count("runShapeInference")) {
      ASSIGN_BOOL_FROM_DYN_FIELD_OR_RETURN_ERR(dyn, settings_.runShapeInference,
                                               "runShapeInference");
    }
    if (dyn.count("enableDebugFuser")) {
      ASSIGN_BOOL_FROM_DYN_FIELD_OR_RETURN_ERR(dyn, settings_.enableDebugFuser,
                                               "enableDebugFuser");
    }
    if (dyn.count("setIncludeLastOffsets")) {
      ASSIGN_BOOL_FROM_DYN_FIELD_OR_RETURN_ERR(
          dyn, settings_.setIncludeLastOffsets, "setIncludeLastOffsets");
    }
    if (dyn.count("debugContinuouslyVerifyDuringModelLoading")) {
      ASSIGN_BOOL_FROM_DYN_FIELD_OR_RETURN_ERR(
          dyn, settings_.debugContinuouslyVerifyDuringModelLoading,
          "debugContinuouslyVerifyDuringModelLoading");
    }
    if (dyn.count("nominalBatchIdx")) {
      ASSIGN_INT_FROM_DYN_FIELD_OR_RETURN_ERR(dyn, settings_.nominalBatchIdx,
                                              "nominalBatchIdx");
    }
    if (dyn.count("availableDevices")) {
      CHECK_DYN_CONTAINS_ARRAY(dyn, "availableDevices");
      ASSIGN_VALUE_OR_RETURN_ERR(
          settings_.availableDevices,
          dynArrayToVec<int32_t>(dyn.at("availableDevices")));
    }
    if (dyn.count("dumpFailedInputsToOnnxFiles")) {
      ASSIGN_BOOL_FROM_DYN_FIELD_OR_RETURN_ERR(
          dyn, settings_.dumpFailedInputsToOnnxFiles,
          "dumpFailedInputsToOnnxFiles");
    }
    if (dyn.count("lazyCompile")) {
      ASSIGN_BOOL_FROM_DYN_FIELD_OR_RETURN_ERR(dyn, settings_.lazyCompile,
                                               "lazyCompile");
    }
    if (dyn.count("enableDeviceTracing")) {
      ASSIGN_BOOL_FROM_DYN_FIELD_OR_RETURN_ERR(
          dyn, settings_.enableDeviceTracing, "enableDeviceTracing");
    }

    if (dyn.count("use_dag_optimizer")) {
      ASSIGN_BOOL_FROM_DYN_FIELD_OR_RETURN_ERR(dyn, settings_.use_dag_optimizer,
                                               "use_dag_optimizer");
    }

    if (dyn.count("apl_parallelization_alg")) {
      ASSIGN_STRING_FROM_DYN_FIELD_OR_RETURN_ERR(
          dyn, settings_.apl_parallelization_alg, "apl_parallelization_alg");
    }
    if (dyn.count("apl_num_parallel_chunks")) {
      ASSIGN_INT_FROM_DYN_FIELD_OR_RETURN_ERR(
          dyn, settings_.apl_num_parallel_chunks, "apl_num_parallel_chunks");
    }

    if (dyn.count("saveGlowIRIntoONNX")) {
      ASSIGN_BOOL_FROM_DYN_FIELD_OR_RETURN_ERR(
          dyn, settings_.saveGlowIRIntoONNX, "saveGlowIRIntoONNX");
    }

    if (dyn.count("loadGlowIRFromONNX")) {
      ASSIGN_BOOL_FROM_DYN_FIELD_OR_RETURN_ERR(
          dyn, settings_.loadGlowIRFromONNX, "loadGlowIRFromONNX");
    }

    if (dyn.count("skipProvisioning")) {
      ASSIGN_BOOL_FROM_DYN_FIELD_OR_RETURN_ERR(dyn, settings_.skipProvisioning,
                                               "skipProvisioning");
    }

    if (dyn.count("debugLayers")) {
      ASSIGN_INT_FROM_DYN_FIELD_OR_RETURN_ERR(dyn, settings_.debugLayers,
                                              "debugLayers");
    }

    return Error::success();
  }

  int64_t getCurrentBCVersion() const override { return 0; }

  Error validate() const override { return Error::success(); }
};

// This class is used as the spec of Glow deserialiation
struct GlowDeserializationSpec : public JsonSerializableCustomClass {
  ADD_VECTOR_FIELD(inputPHNames, std::string)
  ADD_VECTOR_FIELD(inputPHTypes, std::string)
  ADD_VECTOR_FIELD(staticPHNames, std::string)
  ADD_VECTOR_FIELD(staticPHTypes, std::string)
  ADD_VECTOR_FIELD(outputPHNames, std::string)
  ADD_OBJECT_POINTER_FIELD(pytorchLoaderSettings, GlowPyTorchLoaderSettings)
  ADD_STRING_FIELD(functionName, "")

  Expected<folly::dynamic> toDynamicImpl() const override {
    folly::dynamic obj = folly::dynamic::object();
    obj["inputPHNames"] = dynArrayFromVec(inputPHNames);
    obj["inputPHTypes"] = dynArrayFromVec(inputPHTypes);
    obj["staticPHNames"] = dynArrayFromVec(staticPHNames);
    obj["staticPHTypes"] = dynArrayFromVec(staticPHTypes);
    obj["outputPHNames"] = dynArrayFromVec(outputPHNames);
    folly::dynamic settings_dyn;
    ASSIGN_VALUE_OR_RETURN_ERR(settings_dyn,
                               pytorchLoaderSettings->toDynamic());
    obj["pytorchLoaderSettings"] = settings_dyn;
    obj["functionName"] = functionName;
    return obj;
  }

  Error fromDynamicImpl(const folly::dynamic &dyn,
                        int64_t bc_version_when_serialized) override {
    RETURN_ERR_IF_NOT(
        bc_version_when_serialized == 0,
        strFormat("Only bc_version 0 is supported, got bc_version %d",
                  int(bc_version_when_serialized)));

    if (dyn.count("inputPHNames")) {
      CHECK_DYN_CONTAINS_ARRAY(dyn, "inputPHNames");
      ASSIGN_VALUE_OR_RETURN_ERR(
          inputPHNames, dynArrayToVec<std::string>(dyn.at("inputPHNames")));
    }
    if (dyn.count("inputPHTypes")) {
      CHECK_DYN_CONTAINS_ARRAY(dyn, "inputPHTypes");
      ASSIGN_VALUE_OR_RETURN_ERR(
          inputPHTypes, dynArrayToVec<std::string>(dyn.at("inputPHTypes")));
    }
    if (dyn.count("staticPHNames")) {
      CHECK_DYN_CONTAINS_ARRAY(dyn, "staticPHNames");
      ASSIGN_VALUE_OR_RETURN_ERR(
          staticPHNames, dynArrayToVec<std::string>(dyn.at("staticPHNames")));
    }
    if (dyn.count("staticPHTypes")) {
      CHECK_DYN_CONTAINS_ARRAY(dyn, "staticPHTypes");
      ASSIGN_VALUE_OR_RETURN_ERR(
          staticPHTypes, dynArrayToVec<std::string>(dyn.at("staticPHTypes")));
    }
    if (dyn.count("outputPHNames")) {
      CHECK_DYN_CONTAINS_ARRAY(dyn, "outputPHNames");
      ASSIGN_VALUE_OR_RETURN_ERR(
          outputPHNames, dynArrayToVec<std::string>(dyn.at("outputPHNames")));
    }
    CHECK_DYN_CONTAINS_OBJECT(dyn, "pytorchLoaderSettings");
    RETURN_IF_ERR(
        pytorchLoaderSettings->fromDynamic(dyn.at("pytorchLoaderSettings")));

    if (dyn.count("functionName")) {
      ASSIGN_STRING_FROM_DYN_FIELD_OR_RETURN_ERR(dyn, functionName,
                                                 "functionName");
    }
    return Error::success();
  }

  int64_t getCurrentBCVersion() const override { return 0; }

  Error validate() const override { return Error::success(); }
};

} // namespace glow

#undef CHECK_DYN_IS_STRING
#undef CHECK_DYN_IS_OBJECT
#undef CHECK_DYN_IS_BOOL
#undef CHECK_DYN_IS_ARRAY
#undef CHECK_DYN_IS_DOUBLE
#undef CHECK_DYN_IS_INT
#undef CHECK_DYN_CONTAINS_FIELD
#undef CHECK_DYN_CONTAINS_STRING
#undef CHECK_DYN_CONTAINS_OBJECT
#undef CHECK_DYN_CONTAINS_BOOL
#undef CHECK_DYN_CONTAINS_ARRAY
#undef CHECK_DYN_CONTAINS_DOUBLE
#undef CHECK_DYN_CONTAINS_INT
#undef ASSIGN_STRING_FROM_DYN_FIELD_OR_RETURN_ERR
#undef ASSIGN_BOOL_FROM_DYN_FIELD_OR_RETURN_ERR
#undef ASSIGN_DOUBLE_FROM_DYN_FIELD_OR_RETURN_ERR
#undef ASSIGN_INT_FROM_DYN_FIELD_OR_RETURN_ERR
#undef ADD_FIELD
#undef ADD_STRING_FIELD
#undef ADD_OBJECT_POINTER_FIELD
#undef ADD_VECTOR_FIELD
#undef ADD_MAP_FIELD
#undef ADD_BOOL_FIELD
#undef ADD_DOUBLE_FIELD
#undef ADD_INT_FIELD

#endif // GLOW_TORCH_GLOW_SRC_GLOW_COMPILE_SPEC_H
