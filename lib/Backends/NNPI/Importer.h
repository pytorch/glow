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

#ifndef GLOW_NNPI_IMPORTER_H
#define GLOW_NNPI_IMPORTER_H

#include "glow/Backends/BackendOptions.h"
#include "glow/Support/Compiler.h"
#include "nnpi_network_builder.h"
#include "nnpi_network_builder_EXPERIMENTAL.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"
#include <map>
#include <set>
#include <string>

namespace glow {
class Function;
class Placeholder;
class Value;
class Node;
class Tensor;
class Storage;
struct Type;
class INNPINodeImporter;

/// This class imports Glow IR to the NNPI backend.
class NNPIImporter {
public:
  /// Constructor.
  NNPIImporter();

  /// Destructor.
  ~NNPIImporter();

  /// The main entry point for the importer functionality
  /// Imports a Function \p F using options taken from \p opts
  /// \return true iff the import succeeded.
  NNPINetwork importFunction(Function *F, const BackendOptions &opts);

  /// Get a new name for an internal object.
  std::string getInternalName() {
    return internalName_ + std::to_string(internalNameCounter_++);
  }
  /// Get the network handle.
  NNPINetwork getNetwork() const { return network_; }
  /// Add a value to the network by Glow::Value.
  NNPIErrorCode addValueIfTensor(Value *v);
  /// Add a value to the network by parameters.
  NNPIErrorCode addValue(std::string name, const glow::Type *vType,
                         bool alternativeLayout = false, bool input = false,
                         bool output = false,
                         const std::string &scaleTensor = {},
                         const std::string &offsetTensor = {},
                         bool forceSymlowp = false);
  /// Set given tensor names as inputs/outputs.
  void setUsedTensors(const std::set<std::string> &readTensors = {},
                      const std::set<std::string> &writeTensors = {}) {
    readTensors_.insert(readTensors.begin(), readTensors.end());
    writeTensors_.insert(writeTensors.begin(), writeTensors.end());
  }
  /// Add an external tensor to the network (by name).
  NNPIErrorCode addTensor(std::string name, bool alternativeLayout = false,
                          const std::string &scaleTensor = {},
                          const std::string &offsetTensor = {},
                          bool forceSymlowp = false);
  /// Add an external tensor to the network (by parameters).
  NNPIErrorCode addTensor(std::string name, const NNPITensorDesc &desc,
                          const void *pData);
  /// Update the NNPITensorDesc \p desc by the dimensions array \p glowDims.
  static void updateDescDimsFromGlow(const llvm::ArrayRef<size_t> glowDims,
                                     NNPITensorDesc &desc,
                                     bool alternativeLayout = false);
  /// Update the NNPITensorDesc \p desc quantization params by \p vType.
  void updateDescQuantFromGlow(const glow::Type &t, NNPITensorDesc &desc,
                               const std::string &scaleTensor = {},
                               const std::string &offsetTensor = {},
                               bool forceSymlowp = false);

  static bool isVariableUsingAlternativeLayout(Storage *v);
  bool zeroes(const std::string &name) const;
  /// Internal name header used for variables.
  static const std::string internalName_;

private:
  /// Map of named external tensors (inputs, outputs, weights, etc...).
  std::map<std::string, const Tensor *> constants_;
  /// Set of tensors written to by the function.
  std::set<std::string> writeTensors_;
  /// Set of tensors read from by the function.
  std::set<std::string> readTensors_;
  /// Set of tensors already defined.
  std::set<std::string> definedTensors_;
  /// Number of internal names created for variables.
  size_t internalNameCounter_;

  /// NNPI network handle.
  NNPINetwork network_;

  /// Map of Glow node specific importers.
  static const std::map<std::string, INNPINodeImporter *> nodeImporters_;
};

/// Interface class for all node specific importers.
class INNPINodeImporter {
public:
  /// Import a single node \p n and add it to \p importer
  /// \return the create NNPI layer or nullptr if no layer was created.
  virtual NNPIErrorCode importNode(Node *n, NNPIImporter &importer) = 0;

  /// Destructor.
  virtual ~INNPINodeImporter() = default;
};

class NNPIEnvVariables {
public:
  static std::string getVarString(const std::string &varName);
  static bool getVarBool(const std::string &varName);

private:
  static std::map<std::string, std::string> vars_;
};

inline std::string ICETFilename() {
  auto filename = NNPIEnvVariables::getVarString("ICE_T_FILE");
  return filename.length() ? filename : "";
}

inline bool UseIceT() { return NNPIEnvVariables::getVarBool("USE_ICE_T"); }

inline bool UseInferenceAPI() {
  return NNPIEnvVariables::getVarBool("USE_INF_API");
}

inline std::string EnvDeviceVersion() {
  auto deviceVersion = NNPIEnvVariables::getVarString("NNPI_DEVICE_VERSION");
  return deviceVersion.length() ? deviceVersion : "";
}

inline bool SymlowpWA() { return NNPIEnvVariables::getVarBool("SYMLOWP_WA"); }

} // namespace glow
#endif // GLOW_NNPI_IMPORTER_H
