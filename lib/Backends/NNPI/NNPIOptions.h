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
#include "llvm/Support/raw_ostream.h"
#include <map>
#include <string>
#include <vector>

namespace glow {

/// Defines the supported options and used for storing details of each
/// (including names and defaults).
enum NNPIParamOption {
  UseIceTOption = 0,
  InferOnDeviceOption,
  CompiledFileOption,
  UseSymlowpOption,
  DeviceVersionOption,
  NumOfWorkersOption,
  DeviceIDOption,
  IceCoresOption,
  DeviceTraceOption,
  CompilationLogLevelOption,
  OverrideNNPIMemoryOption,
  ShowVarsOption,
  CustomDSPLibOption,
  CommandListsOption,
  CompilationDebugConfigFileOption,
  InternalTestingOption,
  NNPIParamOptionsSize
};

/// Parent calls for all NNPI option knobs.
class NNPIOptions {
public:
  virtual ~NNPIOptions(){};
  /// Dump the status of all variables.
  virtual std::string dumpStatus() const = 0;

protected:
  static std::vector<std::vector<std::string>> &getOptions();

  template <typename ParamsMapType>
  static bool getBoolVal(NNPIParamOption option, const ParamsMapType *params);

  template <typename ParamsMapType>
  static std::string getStringVal(NNPIParamOption option,
                                  const ParamsMapType *params);

  template <typename ParamsMapType>
  static int getIntVal(NNPIParamOption option, const ParamsMapType *params);

  template <typename ParamsMapType>
  static unsigned getUnsignedVal(NNPIParamOption option,
                                 const ParamsMapType *params);
};

/// This class holds all environment variable knobs for the general NNPI
/// backend.
class NNPIBackendOptions : public NNPIOptions {
public:
  explicit NNPIBackendOptions();
  /// Dump the status of all variables.
  virtual std::string dumpStatus() const override;
  /// Lists the supported options (name=>description).
  static llvm::StringMap<std::string> getSupportedOptions();
  /// Compile to ICE-T if true, ICE-Ref otherwise.
  bool useIceT;
  /// Enable execution on device (useIceT_ and !inferOnDevice_ will compile but
  /// not execute inference).
  bool inferOnDevice;
  /// Setting this to true will log the status of all variables at backend
  /// creation.
  bool showVars;

protected:
  static std::map<std::string, std::pair<std::string, std::string>>
  supportedOptionsMap();
};

/// This class holds all environment variable knobs for the NNPI compilation.
class NNPICompilationOptions : public NNPIOptions {
public:
  /// Compilation paramters used as defaults (if exists) that may be overriden
  /// by environment variables.
  NNPICompilationOptions(const std::map<std::string, std::string> *parameters);
  /// Dump the status of all variables.
  virtual std::string dumpStatus() const override;
  /// Lists the supported options (name=>description).
  static llvm::StringMap<std::string> getSupportedOptions();
  /// Compile to ICE-T if true, ICE-Ref otherwise - this variable cab be
  /// verridden by inferOnDevice_.
  bool useIceT;
  /// Enable execution on device (useIceT_ and !inferOnDevice_ will compile but
  /// not execute inference).
  bool inferOnDevice;
  /// Setting this to true will log the status of all variables at backend
  /// creation.
  bool showVars;
  /// Setting this variable will save the compilation output to the filename
  /// specified.
  std::string compiledFile;
  /// Setting this variable will force compilation to the to use no more than
  /// the amount of ice cores (1-12) - default 0.
  int iceCores;
  /// When this flag is set to true all quantized Int8 tensors are set to
  /// Symlowp when their offset is 0.
  bool useSymlowp;
  /// Setting this variable will override target device version used for
  /// compilation (currently supporting 1-3).
  int deviceVersion;
  /// A path to a custom DSP kernel library.
  std::string customDspKernelsFile;
  /// Setting this variable will control the logging level (0 for debug, 1
  /// assert, 2 info, 3 warning, 4 error, 5 critical ).
  int compilationLogLevel;
  /// Compilation debug configuration file.
  std::string debugCompileConfigFile;

protected:
  /// There is only on logger for NNPI compilation. Setting it's properties can
  /// be done in a static method.
  static void setLogLevel(int logLevel);
};

/// This class holds all environment variable knobs for the NNPI device.
class NNPIDeviceOptions : public NNPIOptions {
public:
  /// Compilation paramters used as defaults (if exists) that may be overriden
  /// by environment variables.
  NNPIDeviceOptions(const llvm::StringMap<std::string> *parameters);
  /// All options are copied as is (environment variable are not re-read).
  NNPIDeviceOptions(const NNPIDeviceOptions &other);

  /// Dump the status of all variables.
  virtual std::string dumpStatus() const override;
  /// Lists the supported options (name=>description).
  static llvm::StringMap<std::string> getSupportedOptions();
  /// Compile to ICE-T if true, ICE-Ref otherwise.
  bool useIceT;
  /// Enable execution on device (useIceT_ and !inferOnDevice_ will compile but
  /// not execute inference).
  bool inferOnDevice;
  /// Enables internal testing
  bool internalTesting;
  /// Setting this to true will log the status of all variables at backend
  /// creation.
  bool showVars;
  /// Setting this variable will override the target device ID used to run
  /// (0,1,...).
  int deviceID;
  /// Setting this variable will load the compilation from filename specified.
  std::string compiledFile;
  /// Setting this variable will override the amount of worker threads allocated
  /// per network on the device (default:2).
  int numWorkers;
  /// Setting this variable will enabled device tracing (host2device,
  /// device2host copy infer etc.).
  bool enabledDeviceTraceing;
  /// Overied the max NNPI device memory.
  unsigned deviceMemory;
  /// Enable using command list instead of per command queuing.
  int enabledCommandLists;
};
} // namespace glow
#endif // GLOW_NNPI_ENV_VARIABLES_H
