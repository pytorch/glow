/*
 * Copyright (c) Glow Contributors. See CONTRIBUTORS file.
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

#include "NNPIOptions.h"
#include "DebugMacros.h"
#include "nnpi_transformer.h"
#include "nnpi_transformer_types.h"
#include "llvm/Support/CommandLine.h"
#include <cstdio>
#include <glog/logging.h>
#include <mutex>
#include <sstream>

using namespace glow;

/// Defines the types of information of each options (using paramters map,
/// environment variables and llvm command line arguments).
enum NNPIParamOptionDesc {
  NNPIParamOptionName = 0,
  NNPIParamOptionEnvironmentVarName,
  NNPIParamOptionDescription,
  NNPIParamOptionDefault
};

static std::string getEnvVarString(const std::string &varName,
                                   std::string defaultValue = std::string()) {
  std::string var;
  char *pEnvVar = getenv(varName.c_str());
  return (pEnvVar ? std::string(pEnvVar) : defaultValue);
}

static bool isUInt(std::string var, size_t startOffset = 0) {
  if (!var.empty()) {
    bool foundNoDigit =
        (std::find_if(var.begin() + startOffset, var.end(),
                      [](char c) { return !std::isdigit(c); }) != var.end());
    return !foundNoDigit;
  }
  return false;
}

static bool isInt(std::string var) {
  if (!var.empty()) {
    size_t startOffset = 0;
    if (var[0] == '-') {
      startOffset = 1;
    }
    return isUInt(var, startOffset);
  }
  return false;
}

static uint64_t messagesWriteHandler(const void *ptr, uint64_t size,
                                     uint64_t /*count*/, void * /*userData*/) {
  LOG(INFO) << "[NNPI_LOG]" << reinterpret_cast<const char *>(ptr);
  return size;
}

template <typename ParamsMapType>
static std::string getStringFromMap(const ParamsMapType *parameters,
                                    std::string param,
                                    std::string defaultValue) {
  auto it = parameters->find(param);
  if (it != parameters->end()) {
    return it->second;
  }
  return defaultValue;
}

static llvm::StringMap<std::string> buildFilteredSupportedDescription(
    std::vector<NNPIParamOption> filter,
    const std::vector<std::vector<std::string>> &options) {
  llvm::StringMap<std::string> map;
  for (auto option : filter) {
    std::stringstream desc;
    desc << options[option][NNPIParamOptionDescription]
         << " Default=" << options[option][NNPIParamOptionDefault]
         << " Environment Variable:"
         << options[option][NNPIParamOptionEnvironmentVarName];
    map[options[option][NNPIParamOptionName]] = desc.str();
  }
  return map;
}

static std::vector<std::vector<std::string>> buildOptions() {
  std::vector<std::vector<std::string>> options;
  options.resize(NNPIParamOptionsSize);
  options[UseIceTOption] = {"UseIceT", "USE_ICE_T",
                            "Compile to ICE-T if true, ICE-Ref otherwise.",
                            "0"};
  options[InferOnDeviceOption] = {
      "InferOnDevice", "USE_INF_API",
      "Enable execution on device (UseIceT and !InferOnDevice will compile but "
      "not execute inference).",
      "0"};
  options[InternalTestingOption] = {"[InternalTesting", "INVOKE_RUNTIME",
                                    "Enable internal testing", ""};
  options[CompiledFileOption] = {"CompiledFile", "ICE_T_FILE",
                                 "Sets a file name to save the compilation "
                                 "output to the filename specified.",
                                 ""};
  options[UseSymlowpOption] = {
      "UseSymlowp", "SYMLOWP_WA",
      "When this flag is set to true all quantized Int8 tensors are set to "
      "Symlowp when their offset is 0.",
      "0"};
  options[DeviceVersionOption] = {"NNPIDeviceVersion", "NNPI_DEVICE_VERSION",
                                  "Override target device version used for "
                                  "compilation (currently supporting 1-3).",
                                  "-1"};
  options[NumOfWorkersOption] = {"NumOfWorkers", "NNPI_NUM_WORKERS",
                                 "Override the amount of worker threads "
                                 "allocated per network on the device.",
                                 "-1"};
  options[DeviceIDOption] = {
      "DeviceID", "NNPI_DEVICE_ID",
      "Override the target device ID used to run (0,1,...).", "-1"};
  options[IceCoresOption] = {
      "IceCores", "NNPI_ICE_CORES",
      "Force compilation with specified amount of ice cores (1-12).", "-1"};
  options[DeviceTraceOption] = {
      "DeviceTracing", "NNPI_DEVICE_TRACING",
      "Enabled device tracing (host2device, device2host copy infer etc.).",
      "0"};
  options[CompilationLogLevelOption] = {
      "CompilationLogLevel", "NNPI_LOG_LEVEL",
      "Sets the compilation logging level (0-6). 0=Debug, 1=Assert, 2=Info, "
      "3=Warning, 4=Error, 5=Critical, 6=User,",
      "0"};
  options[OverrideNNPIMemoryOption] = {
      "DeviceMemory", "NNPI_DEVICE_MEMORY",
      "Override the amount of DRAM to allocate per NNPI device, in kilobytes.",
      "0"};
  options[CustomDSPLibOption] = {"NNPICustomDSPLib", "NNPI_CUSTOM_DSP_LIB",
                                 "Sets custom DPS kernel file path.", ""};
  options[ShowVarsOption] = {"ShowVars", "NNPI_SHOW_VARS",
                             "Setting this to true will log the status of all "
                             "variables at backend creation.",
                             "1"};
  options[CompilationDebugConfigFileOption] = {
      "NNPICompilationDebugConfigFile", "NNPI_COMPILATION_CONFIG_FILE",
      "JSON file path containing debug options for NNPI compilation.", ""};
  options[CommandListsOption] = {
      "CommandLists", "NNPI_COMMAND_LISTS",
      "Enabled command lists. "
      "\n  0 = disabled. "
      "\n  1+ = enable command list to queue copy/infer. "
      "\n  2+ = enable command list wait instead of locking host resources. "
      "\n  3+ = enable copy command config (partial copies). ",
      "0"};
#ifdef NDEBUG
  // Setting default ERROR log level for release builds.
  options[CompilationLogLevelOption][NNPIParamOptionDefault] = "4";
  // Setting default vars are not dumped to log.
  options[ShowVarsOption][NNPIParamOptionDefault] = "0";
#endif
  return options;
}

std::vector<std::vector<std::string>> &NNPIOptions::getOptions() {
  static std::vector<std::vector<std::string>> options = buildOptions();
  return options;
}

template <typename ParamsMapType>
std::string NNPIOptions::getStringVal(NNPIParamOption option,
                                      const ParamsMapType *params) {
  auto &options = NNPIOptions::getOptions();
  std::string defaultValue = options[option][NNPIParamOptionDefault];
  if (params) {
    defaultValue = getStringFromMap<ParamsMapType>(
        params, options[option][NNPIParamOptionName], defaultValue);
  }
  return getEnvVarString(options[option][NNPIParamOptionEnvironmentVarName],
                         defaultValue);
}

template <typename ParamsMapType>
bool NNPIOptions::getBoolVal(NNPIParamOption option,
                             const ParamsMapType *params) {
  auto val = NNPIOptions::getStringVal<ParamsMapType>(option, params);
  return (val == std::string("1"));
}

template <typename ParamsMapType>
int NNPIOptions::getIntVal(NNPIParamOption option,
                           const ParamsMapType *params) {
  auto val = NNPIOptions::getStringVal(option, params);
  if (isInt(val)) {
    return std::stoi(val);
  }
  return -1;
}

template <typename ParamsMapType>
unsigned NNPIOptions::getUnsignedVal(NNPIParamOption option,
                                     const ParamsMapType *params) {
  auto val = NNPIOptions::getStringVal(option, params);
  unsigned uval = 0;
  if (isUInt(val)) {
    uval = (unsigned)std::stol(val);
  }
  return uval;
}

llvm::StringMap<std::string> NNPIBackendOptions::getSupportedOptions() {
  static llvm::StringMap<std::string> supportedOptions =
      buildFilteredSupportedDescription(
          {UseIceTOption, InferOnDeviceOption, ShowVarsOption},
          NNPIOptions::getOptions());
  return supportedOptions;
}

NNPIBackendOptions::NNPIBackendOptions() {
  useIceT = NNPIOptions::getBoolVal<llvm::StringMap<std::string>>(UseIceTOption,
                                                                  nullptr);
  inferOnDevice = NNPIOptions::getBoolVal<llvm::StringMap<std::string>>(
      InferOnDeviceOption, nullptr);
  showVars = NNPIOptions::getBoolVal<llvm::StringMap<std::string>>(
      ShowVarsOption, nullptr);
}

std::string NNPIBackendOptions::dumpStatus() const {
  auto &options = NNPIOptions::getOptions();
  std::stringstream desc;
  desc << "\nNNPI Backend environment option variables\n";
  desc << "  " << options[UseIceTOption][NNPIParamOptionEnvironmentVarName]
       << "=" << useIceT << "\n";
  desc << "  "
       << options[InferOnDeviceOption][NNPIParamOptionEnvironmentVarName] << "="
       << inferOnDevice << "\n";
  return desc.str();
}

llvm::StringMap<std::string> NNPICompilationOptions::getSupportedOptions() {
  static llvm::StringMap<std::string> supportedOptions =
      buildFilteredSupportedDescription(
          {UseIceTOption, InferOnDeviceOption, CompiledFileOption,
           UseSymlowpOption, DeviceVersionOption, IceCoresOption,
           CompilationLogLevelOption, ShowVarsOption},
          NNPIOptions::getOptions());
  return supportedOptions;
}

NNPICompilationOptions::NNPICompilationOptions(
    const std::map<std::string, std::string> *parameters) {
  useIceT = NNPIOptions::getBoolVal<std::map<std::string, std::string>>(
      UseIceTOption, parameters);
  inferOnDevice = NNPIOptions::getBoolVal<std::map<std::string, std::string>>(
      InferOnDeviceOption, parameters);
  showVars = NNPIOptions::getBoolVal<std::map<std::string, std::string>>(
      ShowVarsOption, parameters);
  compiledFile = NNPIOptions::getStringVal<std::map<std::string, std::string>>(
      CompiledFileOption, parameters);
  useSymlowp = NNPIOptions::getBoolVal<std::map<std::string, std::string>>(
      UseSymlowpOption, parameters);
  deviceVersion = NNPIOptions::getIntVal<std::map<std::string, std::string>>(
      DeviceVersionOption, parameters);
  iceCores = NNPIOptions::getIntVal<std::map<std::string, std::string>>(
      IceCoresOption, parameters);
  compilationLogLevel =
      NNPIOptions::getIntVal<std::map<std::string, std::string>>(
          CompilationLogLevelOption, parameters);
  debugCompileConfigFile =
      NNPIOptions::getStringVal<std::map<std::string, std::string>>(
          CompilationDebugConfigFileOption, parameters);
  customDspKernelsFile =
      NNPIOptions::getStringVal<std::map<std::string, std::string>>(
          CustomDSPLibOption, parameters);

  setLogLevel(compilationLogLevel);
}

void NNPICompilationOptions::setLogLevel(int logLevel) {
  // We have only one log level for NNPI compilation. Setting it will change
  // the level for all compilation instances.
  NNPI_LOG_LEVEL level;
  switch (logLevel) {
  case 0:
    level = NNPI_LOG_LEVEL_DEBUG;
    break;
  case 1:
    level = NNPI_LOG_LEVEL_ASSERT;
    break;
  case 2:
    level = NNPI_LOG_LEVEL_INFO;
    break;
  case 3:
    level = NNPI_LOG_LEVEL_WARNING;
    break;
  case 4:
    level = NNPI_LOG_LEVEL_CRITICAL;
    break;
  case 5:
    level = NNPI_LOG_LEVEL_ERROR;
    break;
  default:
    level = NNPI_LOG_LEVEL_USER;
  }
  nnpiSetLogLevel(level);
  static std::mutex logSyncMutex;
  std::lock_guard<std::mutex> lk(logSyncMutex);
  static bool logStreamSet = false;
  if (logStreamSet) {
    // Set log stream only once.
    return;
  }
  static NNPIStream logStream;
  std::memset(&logStream, 0, sizeof(NNPIStream));
  logStream.writeCallback = messagesWriteHandler;
  nnpiSetLogStream(&logStream);
  logStreamSet = true;
}

std::string NNPICompilationOptions::dumpStatus() const {
  auto &options = NNPIOptions::getOptions();
  std::stringstream desc;
  desc << "\nNNPI Backend compilation option variables\n";
  desc << "  " << options[UseIceTOption][NNPIParamOptionEnvironmentVarName]
       << "=" << useIceT << "\n";
  desc << "  "
       << options[InferOnDeviceOption][NNPIParamOptionEnvironmentVarName] << "="
       << inferOnDevice << "\n";
  desc << "  " << options[CompiledFileOption][NNPIParamOptionEnvironmentVarName]
       << "=" << compiledFile << "\n";
  desc << "  " << options[UseSymlowpOption][NNPIParamOptionEnvironmentVarName]
       << "=" << useSymlowp << "\n";
  desc << "  "
       << options[DeviceVersionOption][NNPIParamOptionEnvironmentVarName] << "="
       << deviceVersion << "\n";
  desc << "  " << options[IceCoresOption][NNPIParamOptionEnvironmentVarName]
       << "=" << iceCores << "\n";
  desc << "  "
       << options[CompilationDebugConfigFileOption]
                 [NNPIParamOptionEnvironmentVarName]
       << "=" << debugCompileConfigFile << "\n";
  desc << "  "
       << options[CompilationLogLevelOption][NNPIParamOptionEnvironmentVarName]
       << "=" << compilationLogLevel << "\n";
  desc << "  " << options[ShowVarsOption][NNPIParamOptionEnvironmentVarName]
       << "=" << showVars << "\n";
  return desc.str();
}

llvm::StringMap<std::string> NNPIDeviceOptions::getSupportedOptions() {
  static llvm::StringMap<std::string> supportedOptions =
      buildFilteredSupportedDescription({UseIceTOption, InferOnDeviceOption,
                                         CompiledFileOption, NumOfWorkersOption,
                                         DeviceIDOption, DeviceTraceOption,
                                         ShowVarsOption},
                                        NNPIOptions::getOptions());
  return supportedOptions;
}

NNPIDeviceOptions::NNPIDeviceOptions(
    const llvm::StringMap<std::string> *parameters) {
  useIceT = NNPIOptions::getBoolVal<llvm::StringMap<std::string>>(UseIceTOption,
                                                                  parameters);
  inferOnDevice = NNPIOptions::getBoolVal<llvm::StringMap<std::string>>(
      InferOnDeviceOption, parameters);
  internalTesting = NNPIOptions::getStringVal<llvm::StringMap<std::string>>(
                        InternalTestingOption, parameters) != "";
  showVars = NNPIOptions::getBoolVal<llvm::StringMap<std::string>>(
      ShowVarsOption, parameters);
  compiledFile = NNPIOptions::getStringVal<llvm::StringMap<std::string>>(
      CompiledFileOption, parameters);
  deviceID = NNPIOptions::getIntVal<llvm::StringMap<std::string>>(
      DeviceIDOption, parameters);
  numWorkers = NNPIOptions::getIntVal<llvm::StringMap<std::string>>(
      NumOfWorkersOption, parameters);
  enabledDeviceTraceing = NNPIOptions::getBoolVal<llvm::StringMap<std::string>>(
      DeviceTraceOption, parameters);
  deviceMemory = NNPIOptions::getUnsignedVal<llvm::StringMap<std::string>>(
      OverrideNNPIMemoryOption, parameters);
#if 1 // Todo: remove default memory size initialization once real device memory
      // is implemented.
  // Set default memory size if option not defined.
  if (deviceMemory <= 0) {
    deviceMemory = 16000000;
  }
#endif

  enabledCommandLists =
      NNPIOptions::getUnsignedVal<llvm::StringMap<std::string>>(
          CommandListsOption, parameters);
}

NNPIDeviceOptions::NNPIDeviceOptions(const NNPIDeviceOptions &other) {
  showVars = other.showVars;
  useIceT = other.useIceT;
  inferOnDevice = other.inferOnDevice;
  compiledFile = other.compiledFile;
  deviceID = other.deviceID;
  numWorkers = other.numWorkers;
  enabledDeviceTraceing = other.enabledDeviceTraceing;
  deviceMemory = other.deviceMemory;
  enabledCommandLists = other.enabledCommandLists;
}

std::string NNPIDeviceOptions::dumpStatus() const {
  auto &options = NNPIOptions::getOptions();
  std::stringstream desc;
  desc << "\nNNPI Backend device variables\n";
  desc << "  " << options[UseIceTOption][NNPIParamOptionEnvironmentVarName]
       << "=" << useIceT << "\n";
  desc << "  "
       << options[InferOnDeviceOption][NNPIParamOptionEnvironmentVarName] << "="
       << inferOnDevice << "\n";
  desc << "  "
       << options[InternalTestingOption][NNPIParamOptionEnvironmentVarName]
       << "=" << internalTesting << "\n";
  desc << "  " << options[CompiledFileOption][NNPIParamOptionEnvironmentVarName]
       << "=" << compiledFile << "\n";
  desc << "  " << options[DeviceIDOption][NNPIParamOptionEnvironmentVarName]
       << "=" << deviceID << "\n";
  desc << "  " << options[NumOfWorkersOption][NNPIParamOptionEnvironmentVarName]
       << "=" << numWorkers << "\n";
  desc << "  " << options[DeviceTraceOption][NNPIParamOptionEnvironmentVarName]
       << "=" << enabledDeviceTraceing << "\n";
  desc << "  "
       << options[OverrideNNPIMemoryOption][NNPIParamOptionEnvironmentVarName]
       << "=" << deviceMemory << "\n";
  desc << "  " << options[CommandListsOption][NNPIParamOptionEnvironmentVarName]
       << "=" << enabledCommandLists << "\n";
  desc << "  " << options[ShowVarsOption][NNPIParamOptionEnvironmentVarName]
       << "=" << showVars << "\n";

  return desc.str();
}
