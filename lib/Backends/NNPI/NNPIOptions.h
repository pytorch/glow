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

#ifndef GLOW_NNPI_ENV_VARIABLES_H
#define GLOW_NNPI_ENV_VARIABLES_H

#include "nnpi_transformer_types.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include <map>
#include <string>
#include <vector>

namespace glow {

/// Parent calls for all NNPI option knobs.
class NNPIOptions {
public:
  static std::string getFromEnv(std::string envName, std::string defVal);

  template <typename T> static T getStringAsType(std::string sVal);

  virtual std::string dumpStatus();
  virtual llvm::StringMap<std::string> getSupportedOptions();

  virtual llvm::StringRef getOptionsName() const = 0;

  virtual ~NNPIOptions(){};

  template <typename MapType>
  std::string getFromMap(const MapType &map, std::string name,
                         std::string defVal) {
    std::string stVal = defVal;
    auto it = map.find(name);
    if (it != map.end()) {
      stVal = it->second;
    }
    return stVal;
  }

protected:
  llvm::StringMap<std::string> loadedOptions_;
  llvm::StringMap<std::string> supportedOptions_;
};

/// Explicit forward decleration of template type.
template <> bool NNPIOptions::getStringAsType<bool>(std::string sVal);
/// Explicit forward decleration of template type.
template <>
std::string NNPIOptions::getStringAsType<std::string>(std::string sVal);
/// Explicit forward decleration of template type.
template <> int NNPIOptions::getStringAsType<int>(std::string sVal);
/// Explicit forward decleration of template type.
template <> unsigned NNPIOptions::getStringAsType<unsigned>(std::string sVal);

#define DECLARE_NNPI_OPTION(VAR_NAME, VAR_TYPE, OPT_NAME, OPT_DESC, OPT_ENV,   \
                            OPT_DEFAULT)                                       \
  class {                                                                      \
  public:                                                                      \
    inline static llvm::StringRef getName() { return OPT_NAME; }               \
    inline static llvm::StringRef getDesc() { return OPT_DESC; }               \
    inline static llvm::StringRef getEnv() { return OPT_ENV; }                 \
    inline static llvm::StringRef getDefault() { return OPT_DEFAULT; }         \
    inline void setValFromString(std::string v) {                              \
      val_ = NNPIOptions::getStringAsType<VAR_TYPE>(v);                        \
    }                                                                          \
    inline void setVal(VAR_TYPE v) { val_ = v; }                               \
    inline const VAR_TYPE &get() const { return (val_); }                      \
                                                                               \
    operator VAR_TYPE() const { return val_; }                                 \
                                                                               \
  private:                                                                     \
    VAR_TYPE val_;                                                             \
  } VAR_NAME;

#define INIT_NNPI_OPTIONS(VAR_NAME, map)                                       \
  {                                                                            \
    supportedOptions_[VAR_NAME.getName()] =                                    \
        llvm::formatv("{0} Default: {1} Environment Variable:",                \
                      VAR_NAME.getDefault(), VAR_NAME.getEnv())                \
            .str();                                                            \
    std::string stVal =                                                        \
        getFromMap(map, VAR_NAME.getName(), VAR_NAME.getDefault());            \
    stVal = NNPIOptions::getFromEnv(VAR_NAME.getEnv(), stVal);                 \
    this->VAR_NAME.setValFromString(stVal);                                    \
    this->loadedOptions_[VAR_NAME.getEnv()] =                                  \
        llvm::formatv("{0}", VAR_NAME).str();                                  \
  }

class NNPIBackendOptions : public NNPIOptions {
public:
  /// Compile for HW (ignored if InferOnDevice is defined).
  DECLARE_NNPI_OPTION(useIceT, bool, "UseIceT",
                      "Compile for HW (ignored if InferOnDevice is defined).",
                      "USE_ICE_T", "0");
  /// Enable execution on device (if true, will also force compilation for HW
  /// and ignore the UseIceT option).
  DECLARE_NNPI_OPTION(inferOnDevice, bool, "InferOnDevice",
                      "Enable execution on device (if true, will also force "
                      "compilation for HW and ignore the UseIceT option).",
                      "USE_INF_API", "0");
  /// Setting this to true will log the status of all variables at backend
  /// creation.
  DECLARE_NNPI_OPTION(showVars, bool, "ShowVars",
                      "Setting this to true will log the status of all "
                      "variables at backend creation.",
                      "NNPI_SHOW_VARS",
#ifdef NDEBUG
                      "0"
#else
                      "1"
#endif
  );
  NNPIBackendOptions() {
    INIT_NNPI_OPTIONS(useIceT, llvm::StringMap<std::string>());
    INIT_NNPI_OPTIONS(inferOnDevice, llvm::StringMap<std::string>());
    INIT_NNPI_OPTIONS(showVars, llvm::StringMap<std::string>());
  }

  virtual llvm::StringRef getOptionsName() const override {
    return "Backend Options";
  };
};

class NNPICompilationOptions : public NNPIOptions {
public:
  /// Compile for HW (ignored if InferOnDevice is defined).
  DECLARE_NNPI_OPTION(useIceT, bool, "UseIceT",
                      "Compile for HW (ignored if InferOnDevice is defined).",
                      "USE_ICE_T", "0");
  /// Enable execution on device (if true, will also force compilation for HW
  /// and ignore the UseIceT option).
  DECLARE_NNPI_OPTION(inferOnDevice, bool, "InferOnDevice",
                      "Enable execution on device (if true, will also force "
                      "compilation for HW and ignore the UseIceT option).",
                      "USE_INF_API", "0");
  /// Setting this to true will log the status of all variables at backend
  /// creation.
  DECLARE_NNPI_OPTION(showVars, bool, "ShowVars",
                      "Setting this to true will log the status of all "
                      "variables at backend creation.",
                      "NNPI_SHOW_VARS",
#ifdef NDEBUG
                      "0"
#else
                      "1"
#endif
  );
  /// Sets the compilation logging level (0-7). 0=Verbose 1=Debug, 2=Assert,
  /// 3=Info, 4=Warning, 5=Error, 6=Critical, 7=User.
  DECLARE_NNPI_OPTION(
      compilationLogLevel, int, "CompilationLogLevel",
      "Sets the compilation logging level (0-7). 0=Verbose 1=Debug, 2=Assert, "
      "3=Info, 4=Warning, 5=Error, 6=Critical, 7=User.",
      "NNPI_LOG_LEVEL",
#ifdef NDEBUG
      "5"
#else
      "1"
#endif
  );
  /// Setting this variable will save the compilation output to the filename
  /// specified.
  DECLARE_NNPI_OPTION(compiledFile, std::string, "CompiledFile",
                      "Sets a file name to save the compilation output to the "
                      "filename specified.",
                      "ICE_T_FILE", "");
  /// Setting this variable will force compilation to use no more than
  /// the set amount of ice cores (1-12), -1 for unlimited.
  DECLARE_NNPI_OPTION(
      iceCores, int, "IceCores",
      "Force compilation with maximum amount of ice cores, -1 for unlimited.",
      "NNPI_ICE_CORES", "-1");
  /// When this flag is set to true all quantized Int8 tensors are set to
  /// Symlowp when their offset is 0.
  DECLARE_NNPI_OPTION(useSymlowp, bool, "UseSymlowp",
                      "When this flag is set to true all quantized Int8 "
                      "tensors are set to Symlowp when their offset is 0.",
                      "SYMLOWP_WA", "0");
  /// Setting this variable will override target device version used for
  /// compilation (currently supporting 1-3).
  DECLARE_NNPI_OPTION(deviceVersion, int, "DeviceVersion",
                      "Override target device version used for compilation "
                      "(currently supporting 1-3).",
                      "NNPI_DEVICE_VERSION", "-1");
  /// A path to a custom DSP kernel library.
  DECLARE_NNPI_OPTION(customDspKernelsFile, std::string, "CustomDSPLib",
                      "Sets custom DPS kernel file path.",
                      "NNPI_CUSTOM_DSP_LIB", "");
  /// Compilation debug configuration file.
  DECLARE_NNPI_OPTION(
      debugCompileConfigFile, std::string, "CompilationDebugConfigFile",
      "JSON file path containing debug options for NNPI compilation.",
      "NNPI_COMPILATION_CONFIG_FILE", "");
  /// Reserve network resources.
  DECLARE_NNPI_OPTION(
      reserveResources, bool, "ResourceReservation",
      "Reserve execution resources for the network on the device.",
      "NNPI_RESOURCE_RESERVATION", "0");

  NNPICompilationOptions(const std::map<std::string, std::string> &parameters) {
    INIT_NNPI_OPTIONS(useIceT, parameters);
    INIT_NNPI_OPTIONS(inferOnDevice, parameters);
    INIT_NNPI_OPTIONS(showVars, parameters);
    INIT_NNPI_OPTIONS(compiledFile, parameters);
    INIT_NNPI_OPTIONS(iceCores, parameters);
    INIT_NNPI_OPTIONS(useSymlowp, parameters);
    INIT_NNPI_OPTIONS(deviceVersion, parameters);
    INIT_NNPI_OPTIONS(customDspKernelsFile, parameters);
    INIT_NNPI_OPTIONS(compilationLogLevel, parameters);
    INIT_NNPI_OPTIONS(debugCompileConfigFile, parameters);
    INIT_NNPI_OPTIONS(reserveResources, parameters);
    setLogLevel(this->compilationLogLevel);
  }

  virtual llvm::StringRef getOptionsName() const override {
    return "Compilation Options";
  };

protected:
  /// There is only on logger for NNPI compilation. Setting it's properties can
  /// be done in a static method.
  static void setLogLevel(int logLevel);
};

class NNPIDeviceOptions : public NNPIOptions {
public:
  /// Compile for HW (ignored if InferOnDevice is defined).
  DECLARE_NNPI_OPTION(useIceT, bool, "UseIceT",
                      "Compile for HW (ignored if InferOnDevice is defined).",
                      "USE_ICE_T", "0");
  /// Enable execution on device (if true, will also force compilation for HW
  /// and ignore the UseIceT option).
  DECLARE_NNPI_OPTION(inferOnDevice, bool, "InferOnDevice",
                      "Enable execution on device (if true, will also force "
                      "compilation for HW and ignore the UseIceT option).",
                      "USE_INF_API", "0");
  /// Setting this to true will log the status of all variables at backend
  /// creation.
  DECLARE_NNPI_OPTION(showVars, bool, "ShowVars",
                      "Setting this to true will log the status of all "
                      "variables at backend creation.",
                      "NNPI_SHOW_VARS",
#ifdef NDEBUG
                      "0"
#else
                      "1"
#endif
  );
  /// Enables internal testing.
  DECLARE_NNPI_OPTION(internalTesting, std::string, "InternalTesting",
                      "Enable internal testing.", "INVOKE_RUNTIME", "");
  /// Setting this variable will override the target device ID used to run
  /// (0,1,...).
  DECLARE_NNPI_OPTION(deviceID, int, "DeviceID",
                      "Override the target device ID used to run (0,1,...).",
                      "NNPI_DEVICE_ID", "-1");
  /// Setting this variable will override the amount of worker threads allocated
  /// per network on the device (default:2).
  DECLARE_NNPI_OPTION(numWorkers, int, "NumOfWorkers",
                      "Override the amount of worker threads allocated per "
                      "network on the device.",
                      "NNPI_NUM_WORKERS", "-1");
  /// Setting this variable will enabled device tracing (host2device,
  /// device2host copy infer etc.).
  DECLARE_NNPI_OPTION(
      enabledDeviceTracing, bool, "DeviceTracing",
      "Enabled device tracing (host2device, device2host copy infer etc.).",
      "NNPI_DEVICE_TRACING", "0");
  /// Overied the max NNPI device memory.
  DECLARE_NNPI_OPTION(
      deviceMemory, unsigned, "DeviceMemory",
      "Override the amount of DRAM to allocate per NNPI device, in kilobytes.",
      "NNPI_DEVICE_MEMORY", "0");
  /// Enable using command list instead of per command queuing.
  DECLARE_NNPI_OPTION(
      enabledCommandLists, int, "CommandLists",
      "Enabled command lists. "
      "\n  0 = disabled. "
      "\n  1+ = enable command list to queue copy/infer. "
      "\n  2+ = enable command list wait instead of locking host resources. "
      "\n  3+ = enable copy command config (partial copies). ",
      "NNPI_COMMAND_LISTS", "3");
  /// Dump IO to files.
  DECLARE_NNPI_OPTION(dumpIOtoFiles, bool, "DumpIOtoFiles",
                      "Dump Inputs/Outputs to files.", "NNPI_DUMP_IO", "0");

  NNPIDeviceOptions(const llvm::StringMap<std::string> &parameters) {
    INIT_NNPI_OPTIONS(useIceT, parameters);
    INIT_NNPI_OPTIONS(inferOnDevice, parameters);
    INIT_NNPI_OPTIONS(showVars, parameters);
    INIT_NNPI_OPTIONS(internalTesting, parameters);
    INIT_NNPI_OPTIONS(deviceID, parameters);
    INIT_NNPI_OPTIONS(numWorkers, parameters);
    INIT_NNPI_OPTIONS(enabledDeviceTracing, parameters);
    INIT_NNPI_OPTIONS(deviceMemory, parameters);
    INIT_NNPI_OPTIONS(enabledCommandLists, parameters);
    INIT_NNPI_OPTIONS(dumpIOtoFiles, parameters);
  }
  virtual llvm::StringRef getOptionsName() const override {
    return "Device Options";
  };
};

#undef DECLARE_NNPI_OPTION
#undef INIT_NNPI_OPTIONS

} // namespace glow
#endif // GLOW_NNPI_ENV_VARIABLES_H
