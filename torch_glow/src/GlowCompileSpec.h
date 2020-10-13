// Copyright 2004-present Facebook. All Rights Reserved.

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

#define CHECK_DYN_CONTAINS_DOUBLE(dyn, fieldName)                              \
  do {                                                                         \
    CHECK_DYN_CONTAINS_FIELD((dyn), (fieldName));                              \
    RETURN_ERR_IF_NOT(                                                         \
        (dyn).at((fieldName)).isDouble(),                                      \
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

#define ASSIGN_DOUBLE_FROM_DYN_FIELD_OR_RETURN_ERR(dyn, output, fieldName)     \
  do {                                                                         \
    CHECK_DYN_CONTAINS_DOUBLE((dyn), (fieldName));                             \
    output = (dyn).at(fieldName).getDouble();                                  \
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
  // -1 indicates use all available devices
  ADD_INT_FIELD(num_devices_to_use, -1)
  ADD_INT_FIELD(replication_count, 1)
  ADD_MAP_FIELD(backend_specific_opts, std::string, std::string)

  Expected<folly::dynamic> toDynamicImpl() const override {
    folly::dynamic obj = folly::dynamic::object();
    obj["convert_to_fp16"] = convert_to_fp16;
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

  Expected<folly::dynamic> toDynamicImpl() const override {
    folly::dynamic obj = folly::dynamic::object();
    obj["glow_backend"] = glow_backend;
    obj["enable_fuser"] = enable_fuser;
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

  c10::ScalarType get_elem_type() { return elem_type; }
  std::vector<int64_t> get_dims() { return dims; }

  void set(std::vector<int64_t> dims_param, c10::ScalarType elem_type_param) {
    dims = dims_param;
    elem_type = elem_type_param;
    initialized = true;
  }

  void set_same_as(const at::Tensor &t) {
    if (t.is_quantized()) {
      throw std::invalid_argument(
          "Quantized PyTorch Tensor is not supported yet in InputSpec.");
    }
    std::vector<int64_t> sizesCopy;
    std::copy(t.sizes().begin(), t.sizes().end(),
              std::back_inserter(sizesCopy));
    set(sizesCopy, t.scalar_type());
  }

  Expected<folly::dynamic> toDynamicImpl() const override {
    folly::dynamic obj = folly::dynamic::object();
    obj["elem_type"] = static_cast<int64_t>(elem_type);
    obj["dims"] = dynArrayFromVec(dims);
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
