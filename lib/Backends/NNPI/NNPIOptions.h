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

#include "NNPIUtils.h"

#include "nnpi_transformer_types.h"

#include "glow/Backends/BackendOptions.h"
#include "glow/Support/Error.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

#include <map>
#include <string>
#include <vector>

namespace glow {

// Return true in case cpuinfo contains flag.
static bool isStringFoundInCpuInfo(const char *flag) {
  FILE *cpuinfo = fopen("/proc/cpuinfo", "rb");
  char *arg = nullptr;
  size_t size = 0;
  bool found = false;
  while ((found == false) && (getdelim(&arg, &size, 32, cpuinfo) != -1)) {
    if (strncmp(arg, flag, strlen(flag)) == 0) {
      found = true;
    }
  }
  if (arg) {
    free(arg);
  }
  fclose(cpuinfo);
  return found;
}

/// Parent calls for all NNPI option knobs.
class NNPIOptions {
public:
  static std::string getFromEnv(std::string envName, std::string defVal);

  template <typename T> static T getStringAsType(std::string sVal);

  /// Get the device version stepping for first installed device.
  /// \returns 1 for A0, 2 for B0, 3 for C0 etc. or 0 if not found.
  static unsigned getFirstDeviceSteppingVersion();

  /// \returns an expected wrapping the NNPI_DEVICE_TYPE based on the
  /// \p deviceVersion, \p inferOnDevice, as well as device version found from
  /// \ref getFirstDeviceSteppingVersion().
  static Expected<NNPI_DEVICE_TYPE> getDeviceVersion(bool inferOnDevice,
                                                     int deviceVersion);

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

/// Explicit forward declaration of template type.
template <> bool NNPIOptions::getStringAsType<bool>(std::string sVal);
/// Explicit forward declaration of template type.
template <>
std::string NNPIOptions::getStringAsType<std::string>(std::string sVal);
/// Explicit forward declaration of template type.
template <> int NNPIOptions::getStringAsType<int>(std::string sVal);
/// Explicit forward declaration of template type.
template <> unsigned NNPIOptions::getStringAsType<unsigned>(std::string sVal);
/// Explicit forward declaration of template type.
template <> float NNPIOptions::getStringAsType<float>(std::string sVal);
/// Explicit forward declaration of template type for uint64_t.
template <> uint64_t NNPIOptions::getStringAsType<uint64_t>(std::string sVal);

#define DECLARE_NNPI_OPTION(VAR_NAME, VAR_TYPE, OPT_NAME, OPT_DESC, OPT_ENV,   \
                            OPT_DEFAULT)                                       \
  class {                                                                      \
  public:                                                                      \
    inline static llvm::StringRef getName() { return "NNPI_" OPT_NAME; }       \
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
    std::string stVal = getFromMap(map, VAR_NAME.getName().str(),              \
                                   VAR_NAME.getDefault().str());               \
    stVal = NNPIOptions::getFromEnv(VAR_NAME.getEnv().str(), stVal);           \
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
  /// Dump runtime graph.
  DECLARE_NNPI_OPTION(dumpRuntime, bool, "DumpRuntime",
                      "Dump runtime graph (bindContexts).", "NNPI_DUMP_RUNTIME",
                      "0");

  NNPIBackendOptions() {
    INIT_NNPI_OPTIONS(useIceT, llvm::StringMap<std::string>());
    INIT_NNPI_OPTIONS(inferOnDevice, llvm::StringMap<std::string>());
    INIT_NNPI_OPTIONS(showVars, llvm::StringMap<std::string>());
    INIT_NNPI_OPTIONS(dumpRuntime, llvm::StringMap<std::string>());
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
  /// Normalize layer names.
  DECLARE_NNPI_OPTION(normalizeLayerNames, bool, "NormalizeLayerNames",
                      "Normalize layer names.", "NNPI_NORMALIZE_LAYER_NAMES",
                      "0");
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
  /// Use function name for compilation compilation output filename (works only
  /// when CompiledFile is not empty).
  DECLARE_NNPI_OPTION(
      compileOutputPostfix, bool, "compileOutputPostfix",
      "Use function name as postfix for compilation output filename (or as the "
      "name of the function when CompiledFile option is empty).",
      "ICE_T_FILE_POSTFIX", "0");
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
  /// Disable constant folding during compilation.
  DECLARE_NNPI_OPTION(disableConstFolding, bool, "DisableConstFolding",
                      "Disable constant folding during compilation.",
                      "NNPI_DISABLE_CONSTFOLD", "0");
  /// Setting this variable will override the amount of worker threads allocated
  /// for the network on the device (default:2).
  DECLARE_NNPI_OPTION(numWorkers, int, "NumOfWorkers",
                      "Override the amount of worker threads allocated for the "
                      "network on the device.",
                      "NNPI_NUM_WORKERS", "2");
  /// Power & Performance hints. See more details at:
  /// https://github.com/IntelAI/nnpi-sw/blob/master/include/nnpi_inference_types.h
  DECLARE_NNPI_OPTION(ringPrio, float, "RingPrio",
                      "Set the ring frequency priority.", "NNPI_RING_PRIO",
                      "0.f");
  DECLARE_NNPI_OPTION(iceBOPrio0, float, "IceBOPrio0",
                      "Set ICE-BO 0 frequency priority.", "NNPI_ICEBO_PRIO0",
                      "0.f");
  DECLARE_NNPI_OPTION(iceBOPrio1, float, "IceBOPrio1",
                      "Set ICE-BO 1 frequency priority.", "NNPI_ICEBO_PRIO1",
                      "0.f");
  DECLARE_NNPI_OPTION(iceBOPrio2, float, "IceBOPrio2",
                      "Set ICE-BO 2 frequency priority.", "NNPI_ICEBO_PRIO2",
                      "0.f");
  DECLARE_NNPI_OPTION(iceBOPrio3, float, "IceBOPrio3",
                      "Set ICE-BO 3 frequency priority.", "NNPI_ICEBO_PRIO3",
                      "0.f");
  DECLARE_NNPI_OPTION(iceBOPrio4, float, "IceBOPrio4",
                      "Set ICE-BO 4 frequency priority.", "NNPI_ICEBO_PRIO4",
                      "0.f");
  DECLARE_NNPI_OPTION(iceBOPrio5, float, "IceBOPrio5",
                      "Set ICE-BO 5 frequency priority.", "NNPI_ICEBO_PRIO5",
                      "0.f");
  DECLARE_NNPI_OPTION(ddrBandwidth, float, "DDRBandwidth",
                      "Set an estimated DDR bandwidth in GB/s.", "NNPI_DDR_BW",
                      "0.f");
  /// Disable SLS offload to IA.
  DECLARE_NNPI_OPTION(
      disableSLSOffloadToIA, bool, "DisableSLSOffloadToIA",
      "Disable SLS offloading to IA (SLS will execute on ICE where possible).",
      "NNPI_DISABLE_SLS_OFFLOAD", "1");
  /// Do not allow weights on LLC.
  DECLARE_NNPI_OPTION(forceWeightsOutOfLLC, bool, "forceWeightsOutOfLLC",
                      "Do not allow weights on LLC.", "NNPI_WEIGHTS_OFF_LLC",
                      "0");
  /// Disable Calculation of all one len at runtime.
  DECLARE_NNPI_OPTION(disableSlsAllLenOneCalcAtRunTime, bool,
                      "disableSlsAllLenOneCalcAtRunTime",
                      "Disable Calculation of all one len at runtime.",
                      "NNPI_DISABLE_SLS_ALL_ONE_RUNTIME_CALC", "0");
  /// Enable lightweight compilation.
  DECLARE_NNPI_OPTION(lightCompilation, bool, "LightCompilation",
                      "Enable light compilation (only for gathering metadata).",
                      "NNPI_LIGHT_COMPILATION", "0");
  /// Dump compiler DOT files.
  DECLARE_NNPI_OPTION(dumpDotFiles, bool, "DumpDotFiles",
                      "Dump Dot files of the network during compilation.",
                      "NNPI_DUMP_DOT", "0");
  /// Dump compilation info.
  DECLARE_NNPI_OPTION(dumpCompilationInfo, bool, "dumpCompilationInfo",
                      "Dump the compilation info in text form.",
                      "NNPI_DUMP_COMP_INFO", "0");

#if NNPI_MAJOR_VERSION >= 1 && NNPI_MINOR_VERSION >= 1
  /// Enable Execution Section fusion pass.
  DECLARE_NNPI_OPTION(enableESUnifyAdditionalPass, bool,
                      "enableESUnifyAdditionalPass",
                      "Enable Execution Section fusion pass.",
                      "NNPI_ENABLE_ES_FUSION", "0");

  /// Enable Node Splitter at ICE-T level for all Nodes.
  DECLARE_NNPI_OPTION(enableLayerSplitter, bool, "enableLayerSplitter",
                      "Enable Generic Layer Splitter for all Nodes.",
                      "NNPI_ENABLE_LAYER_SPLITTER", "0");

  /// Enable Spatial splitter for Convolution Nodes
  /// This needs the 'Generic Node splitter' to be enabled.
  DECLARE_NNPI_OPTION(enableConvSpatialSplitter, bool,
                      "enableConvSpatialSplitter",
                      "Enable splits along X-Y Dims of Convolution Nodes.",
                      "NNPI_ENABLE_CONV_SPATIAL_SPLITTER", "0");

  /// Enable Batch splitter for Convolution Nodes
  /// This needs the 'Generic Node splitter' to be enabled.
  DECLARE_NNPI_OPTION(enableConvBatchSplitter, bool, "enableConvBatchSplitter",
                      "Enable splits on Batch Dim of Convolution Nodes.",
                      "NNPI_ENABLE_CONV_BATCH_SPLITTER", "0");

  /// Approximation used in dequantization
  DECLARE_NNPI_OPTION(enableFCDynamicQuantizationAllSA, bool,
                      "enableFCDynamicQuantizationAllSA",
                      "Enable approximation in dequant after dynamic FC.",
                      "NNPI_ENABLE_FC_DQ_ALL_SA", "1");
#endif

  /// Pointer to custom DSP kernels library.
  /// This is only valid if CustomDSPLib is not provided.
  DECLARE_NNPI_OPTION(customDspKernelsLibPtr, uint64_t, "CustomDSPLibPtr",
                      "Pointer to custom DSP kernels lib.",
                      "NNPI_CUSTOM_DSP_LIB_PTR", "0");
  /// Size of custom DSP kernels library.
  DECLARE_NNPI_OPTION(customDspKernelsSize, uint64_t, "CustomDSPLibSize",
                      "Size of custom DSP kernels lib.",
                      "NNPI_CUSTOM_DSP_LIB_SIZE", "0");

  /// Disable weigths from memory pool.
  /// When this flag is true, weights will not be part of memory pool.
  /// This helps runtime to reuse weights based on SHA.
  DECLARE_NNPI_OPTION(disableWeightsInPool, bool, "disableWeightsInPool",
                      "Don't include weights in memory pool.",
                      "NNPI_WEIGHTS_OFF_MEM_POOL", "0");

  /// Dump intermediate buffers.
  DECLARE_NNPI_OPTION(dumpIntermediate, bool, "dumpIntermediate",
                      "Dump Intermediate buffers.", "NNPI_DUMP_INTER", "0");

  /// Number of Parallel Decider Compilations. 0 means all decider in parallel.
  DECLARE_NNPI_OPTION(numDeciderCompilation, uint32_t, "numDeciderCompilation",
                      "Number of Parallel Decider OMP Compilations.",
                      "NNPI_NUM_PARALLEL_COMPILE", "2");

  /// Weights threshold to be in pool, valid if disable weights pool is enabled.
  DECLARE_NNPI_OPTION(thresholdDisableWeightsPool, uint32_t,
                      "thresholdDisableWeightsPool",
                      "Below this threshold, weights to be part of pool.",
                      "NNPI_THRESHOLD_WEIGHTS_OFF_MEM_POOL", "0");

  NNPICompilationOptions(const BackendSpecificOptions &parameters) {
    INIT_NNPI_OPTIONS(useIceT, parameters);
    INIT_NNPI_OPTIONS(inferOnDevice, parameters);
    INIT_NNPI_OPTIONS(normalizeLayerNames, parameters);
    INIT_NNPI_OPTIONS(showVars, parameters);
    INIT_NNPI_OPTIONS(compiledFile, parameters);
    INIT_NNPI_OPTIONS(compileOutputPostfix, parameters);
    INIT_NNPI_OPTIONS(iceCores, parameters);
    INIT_NNPI_OPTIONS(useSymlowp, parameters);
    INIT_NNPI_OPTIONS(deviceVersion, parameters);
    INIT_NNPI_OPTIONS(customDspKernelsFile, parameters);
    INIT_NNPI_OPTIONS(compilationLogLevel, parameters);
    INIT_NNPI_OPTIONS(debugCompileConfigFile, parameters);
    INIT_NNPI_OPTIONS(reserveResources, parameters);
    INIT_NNPI_OPTIONS(disableConstFolding, parameters);
    INIT_NNPI_OPTIONS(numWorkers, parameters);
    setLogLevel(this->compilationLogLevel);
    INIT_NNPI_OPTIONS(ringPrio, parameters);
    INIT_NNPI_OPTIONS(iceBOPrio0, parameters);
    INIT_NNPI_OPTIONS(iceBOPrio1, parameters);
    INIT_NNPI_OPTIONS(iceBOPrio2, parameters);
    INIT_NNPI_OPTIONS(iceBOPrio3, parameters);
    INIT_NNPI_OPTIONS(iceBOPrio4, parameters);
    INIT_NNPI_OPTIONS(iceBOPrio5, parameters);
    INIT_NNPI_OPTIONS(ddrBandwidth, parameters);
    INIT_NNPI_OPTIONS(disableSLSOffloadToIA, parameters);
    INIT_NNPI_OPTIONS(lightCompilation, parameters);
    INIT_NNPI_OPTIONS(forceWeightsOutOfLLC, parameters);
    INIT_NNPI_OPTIONS(disableSlsAllLenOneCalcAtRunTime, parameters);
    INIT_NNPI_OPTIONS(dumpDotFiles, parameters);
    INIT_NNPI_OPTIONS(dumpCompilationInfo, parameters);
#if NNPI_MAJOR_VERSION >= 1 && NNPI_MINOR_VERSION >= 1
    INIT_NNPI_OPTIONS(enableFCDynamicQuantizationAllSA, parameters);
    INIT_NNPI_OPTIONS(enableESUnifyAdditionalPass, parameters);
    INIT_NNPI_OPTIONS(enableLayerSplitter, parameters);
    INIT_NNPI_OPTIONS(enableConvSpatialSplitter, parameters);
    INIT_NNPI_OPTIONS(enableConvBatchSplitter, parameters);
#endif
    INIT_NNPI_OPTIONS(customDspKernelsLibPtr, parameters);
    INIT_NNPI_OPTIONS(customDspKernelsSize, parameters);
    INIT_NNPI_OPTIONS(disableWeightsInPool, parameters);

    INIT_NNPI_OPTIONS(dumpIntermediate, parameters);
    if (inferOnDevice) {
      // dumpIntermediate is not supported for device, works only with ice-ref.
      dumpIntermediate.setVal(false);
    }

    INIT_NNPI_OPTIONS(numDeciderCompilation, parameters);
    INIT_NNPI_OPTIONS(thresholdDisableWeightsPool, parameters);
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
  /// Setting this variable will override the target device ID used to run
  /// (0,1,...).
  DECLARE_NNPI_OPTION(deviceId, int, "DeviceID",
                      "Override the target device ID used to run (0,1,...).",
                      "NNPI_DEVICE_ID", "-1");
  /// Enable Hardware Trace.
  DECLARE_NNPI_OPTION(hardwareTraces, bool, "HardwareTraces",
                      "Enable hardware traces when device traces are started "
                      "(default is disabled).",
                      "NNPI_HW_TRACES", "0");
  /// Max software trace capture size in MB.
  DECLARE_NNPI_OPTION(
      softwareTracesMaxBuffer, uint32_t, "SoftwareTracesMaxBuffer",
      "Set the max internal buffer size for device software traces."
      "(use 0 for hard coded default).",
      "NNPI_SW_TRACES_BUFFER_SIZE", "0");
  /// Max hardware trace capture size in MB.
  DECLARE_NNPI_OPTION(
      hardwareTracesMaxBuffer, uint32_t, "HardwareTracesMaxBuffer",
      "Set the max internal buffer size for device hardware traces."
      "Enabled only when hardwareTraces=1 (use 0 for hard coded default).",
      "NNPI_HW_TRACES_BUFFER_SIZE", "0");
  /// Path to dump raw trace events from NNP-I.
  DECLARE_NNPI_OPTION(
      rawTracesDumpPath, std::string, "RawTracesDumpPath",
      "Set a path (including a file name) to dump raw device events into a "
      "file. If empty, raw events are not dumped.",
      "NNPI_DEVICE_TRACES_DUMP_PATH", "");
  /// Override the max NNPI device memory.
  DECLARE_NNPI_OPTION(
      deviceMemory, unsigned, "DeviceMemory",
      "Override the amount of DRAM to allocate per NNPI device, in kilobytes.",
      "NNPI_DEVICE_MEMORY", "0");
  /// Dump IO to files.
  DECLARE_NNPI_OPTION(dumpIOtoFiles, bool, "DumpIOtoFiles",
                      "Dump Inputs/Outputs to files.", "NNPI_DUMP_IO", "0");
  /// Force using a specific AVX type.
  DECLARE_NNPI_OPTION(avxType, int, "AvxType",
                      "Force using a specific AVX type."
                      "\n  0 = No AVX. "
                      "\n  1 = Use AVX512. ",
                      "NNPI_AVX_TYPE", "-1");
  /// Disable DRT support.
  DECLARE_NNPI_OPTION(disableDRT, bool, "DisableDRT",
                      "Disable DRT support (copy to/from host instead).",
                      "NNPI_DISABLE_DRT", "0");
  /// Disable P2P support.
  DECLARE_NNPI_OPTION(disableP2P, bool, "DisableP2P",
                      "Disable P2P support (copy to/from host instead).",
                      "NNPI_DISABLE_P2P", "0");
  /// Dump runtime graph.
  DECLARE_NNPI_OPTION(dumpRuntime, bool, "DumpRuntime",
                      "Dump runtime graph (bindContexts).", "NNPI_DUMP_RUNTIME",
                      "0");
  /// Disable Device IO Buffers.
  DECLARE_NNPI_OPTION(disableDeviceIOBuffer, bool, "DisableDeviceIOBuffer",
                      "Disable IO buffers allocation by the NNPI stack.",
                      "NNPI_DISABLE_IOBUFFER", "1");
  /// Disable Infer/Copy commands (for overhead measurements).
  DECLARE_NNPI_OPTION(
      disableCommands, int, "DisableCommands",
      "Disable Inference for overhead measurements."
      "\n 0 = Both copy and infer commands work."
      "\n 1 = Copy commands work, infer commands disabled."
      "\n 2 = Both copy and infer commands are disabled."
      "\n 3 = All commands and pre/post processing are disabled.",
      "NNPI_DISABLE_COMMANDS", "0");

  /// Inference timeout threshold in us. Default UINT32_MAX means infinity.
  unsigned inferTimeoutUs{UINT32_MAX};

  NNPIDeviceOptions(const llvm::StringMap<std::string> &parameters) {
    INIT_NNPI_OPTIONS(useIceT, parameters);
    INIT_NNPI_OPTIONS(inferOnDevice, parameters);
    INIT_NNPI_OPTIONS(showVars, parameters);
    INIT_NNPI_OPTIONS(deviceId, parameters);
    INIT_NNPI_OPTIONS(hardwareTraces, parameters);
    INIT_NNPI_OPTIONS(softwareTracesMaxBuffer, parameters);
    INIT_NNPI_OPTIONS(hardwareTracesMaxBuffer, parameters);
    INIT_NNPI_OPTIONS(rawTracesDumpPath, parameters);
    INIT_NNPI_OPTIONS(deviceMemory, parameters);
    INIT_NNPI_OPTIONS(dumpIOtoFiles, parameters);
    INIT_NNPI_OPTIONS(avxType, parameters);
    INIT_NNPI_OPTIONS(disableDRT, parameters);
    INIT_NNPI_OPTIONS(disableP2P, parameters);
    INIT_NNPI_OPTIONS(dumpRuntime, parameters);
    INIT_NNPI_OPTIONS(disableDeviceIOBuffer, parameters);
    INIT_NNPI_OPTIONS(disableCommands, parameters);

    if (avxType == -1) {
      if (isStringFoundInCpuInfo("avx512f")) {
        avxType.setVal(NNPI_AVX_AVX512);
      } else {
        avxType.setVal(NNPI_AVX_NONE);
      }
    }
  }
  virtual llvm::StringRef getOptionsName() const override {
    return "Device Options";
  };
};

#undef DECLARE_NNPI_OPTION
#undef INIT_NNPI_OPTIONS

} // namespace glow
#endif // GLOW_NNPI_ENV_VARIABLES_H
