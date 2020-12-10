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

#ifndef FX_NNPI_IMPORTER_H
#define FX_NNPI_IMPORTER_H

#include "glow/fb/fx_nnpi_importer/Utils.h"
#include "glow/lib/Backends/NNPI/NNPIOptions.h"
#include "nnpi_network_builder.h"
#include "nnpi_network_builder_EXPERIMENTAL.h"
#include "nnpi_transformer.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"
#include <folly/dynamic.h>
#include <unordered_set>

/// This class imports Glow IR to the NNPI backend.
class FXNNPIImporter {
public:
  /// Constructor.
  explicit FXNNPIImporter(const glow::NNPICompilationOptions &compileOptions);

  /// Destructor.
  ~FXNNPIImporter();

  /// The main entry point for the importer functionality
  /// Imports a submodule if \p submodule is specified, otherwise, import the
  /// whole \p FXIR using options taken from \p opts and constant pointers
  /// stored in \p constants \return a nnpi network.
  NNPINetwork importFunction(const folly::dynamic &FXIR,
                             const std::string &submodule,
                             const llvm::StringMap<const void *> &constants);

  /// Get the network handle.
  NNPINetwork getNetwork() const { return network_; }

  /// Add Tensor to the network by parameters.
  NNPIErrorCode addTensor(const std::string &name, const string &dtypeStr,
                          const llvm::ArrayRef<glow::dim_t> dims,
                          bool input = false, bool output = false,
                          const std::string &scaleTensor = {},
                          const std::string &offsetTensor = {},
                          bool forceSymlowp = false);

  /// Set given tensor names as inputs/outputs.
  void
  setUsedTensors(const std::unordered_set<std::string> &readTensors = {},
                 const std::unordered_set<std::string> &writeTensors = {}) {
    readTensors_.insert(readTensors.begin(), readTensors.end());
    writeTensors_.insert(writeTensors.begin(), writeTensors.end());
  }

  /// Update the NNPITensorDesc \p desc by the dimensions array \p dims.
  static void updateDescDimsFromFX(const llvm::ArrayRef<glow::dim_t> &dims,
                                   NNPITensorDesc &desc);

  /// Update the NNPITensorDesc \p desc quantization params by \p dtype.
  void updateDescQuantFromFX(const utils::DTYPE &dtype, NNPITensorDesc &desc,
                             const std::string &scaleTensor = {},
                             const std::string &offsetTensor = {},
                             bool forceSymlowp = false);

  /// \returns the constants
  const void *getConstant(const std::string &name) const;

private:
  /// Mapping from constant name to a void pointer points to where the constant
  /// is actually stored.
  const llvm::StringMap<const void *> *constants_;

  /// Set of tensors written to by the function.
  std::unordered_set<std::string> writeTensors_;

  /// Set of tensors read from by the function.
  std::unordered_set<std::string> readTensors_;

  /// Set of tensors already defined.
  std::unordered_set<std::string> definedTensors_;

  /// NNPI network handle.
  NNPINetwork network_;

  /// NNPI Device configuration.
  const glow::NNPICompilationOptions &compileOptions_;
};

/// Interface class for all node specific importers.
class INNPIFXNodeImporter {
public:
  /// Import a single node \p node and add it to \p importer network.
  /// \p getQualName is used to get the constant name in the constants_
  /// map for a given parameter name like "weight", "bias" and etc.
  /// \returns the create NNPI layer or nullptr if no layer was created.
  virtual NNPIErrorCode
  importNode(const folly::dynamic &node,
             const std::function<string(string)> &getQualName,
             FXNNPIImporter &importer) = 0;

  /// Destructor.
  virtual ~INNPIFXNodeImporter() = default;
};

#endif // FX_NNPI_IMPORTER_H
