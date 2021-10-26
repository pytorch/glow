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

#include "folly/dynamic.h"
#include "glow/fb/fx/nnpi_importer/Utils.h"
#include "glow/lib/Backends/NNPI/NNPIOptions.h"
#include "nnpi_network_builder.h"
#include "nnpi_network_builder_EXPERIMENTAL.h"
#include "nnpi_transformer.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/raw_ostream.h"
#include <folly/dynamic.h>
#include <unordered_set>

/// This class imports Glow IR to the NNPI backend.
class FXNNPIImporter {
public:
  /// Constructor using options taken from \p compileOptions and constant
  /// pointers stored in \p constants.
  explicit FXNNPIImporter(const glow::NNPICompilationOptions &compileOptions,
                          const llvm::StringMap<const void *> &constants);

  /// Destructor.
  ~FXNNPIImporter();

  /// Helper function which iterates through all nodes and logs any unsupported
  /// nodes.
  void logUnsupportedNodes(const folly::dynamic &mod);

  /// The main entry point for the importer functionality. Imports a submodule
  /// if \p submodule is specified, otherwise, import the whole \p FXIR.
  /// \returns an NNPI network.
  NNPINetwork importFunction(const folly::dynamic &FXIR,
                             const std::string &submodule);

  /// Get the network handle.
  NNPINetwork getNetwork() const { return network_; }

  /// Add Tensor to the network by parameters.
  NNPIErrorCode addTensor(const std::string &name, const string &dtypeStr,
                          llvm::ArrayRef<glow::dim_t> dims, bool input = false,
                          bool output = false, const float &scale = 1.f,
                          const int32_t &offset = 0,
                          const std::string &scaleTensor = {},
                          const std::string &offsetTensor = {},
                          bool forceSymlowp = false, bool zeroOffset = false);

  /// Add Tensor to the network by node.
  NNPIErrorCode addTensor(const std::string &name, const folly::dynamic &node,
                          bool input = false, bool output = false);

  /// Set given tensor names as inputs/outputs.
  void
  setUsedTensors(const std::unordered_set<std::string> &readTensors = {},
                 const std::unordered_set<std::string> &writeTensors = {}) {
    for (const std::string &str : readTensors) {
      // Skip empty tensor names. These may be passed in when an optional input
      // was not used.
      if (!str.empty()) {
        readTensors_.insert(str);
      }
    }
    writeTensors_.insert(writeTensors.begin(), writeTensors.end());
  }

  /// Update the NNPITensorDesc \p desc by the dimensions array \p dims.
  static void updateDescDimsFromFX(llvm::ArrayRef<glow::dim_t> dims,
                                   NNPITensorDesc &desc);

  /// Update the NNPITensorDesc \p desc quantization params by \p dtype.
  void updateDescQuantFromFX(const utils::DTYPE &dtype, NNPITensorDesc &desc,
                             const float &scale = 1.f,
                             const int32_t &offset = 0,
                             const std::string &scaleTensor = {},
                             const std::string &offsetTensor = {},
                             bool forceSymlowp = false,
                             bool zeroOffset = false);

  /// \returns whether there is a Constant known by \p name. Does not look
  /// through getattr aliases.
  bool isConstant(llvm::StringRef name) const {
    return constants_.count(name);
  };

  /// \returns the Constant known by \p name. If \p name is from a getattr the
  /// constant will be looked up by the underlying name found by
  /// \ref getConstantName.
  const void *getConstant(llvm::StringRef name) const;

  /// \returns the underlying name of a Constant given provided \p name. If \p
  /// name is already the name of a Constant it is returned, else looks for
  /// getattr aliases to return the name of the actual underlying Constant.
  const char *getConstantName(llvm::StringRef name) const;

  /// \returns the NNPITensorDesc for node by \p name.
  const NNPITensorDesc &getTensorDesc(llvm::StringRef name) const;

  /// \returns the name of some input \p node. Fatals if \p node is not
  /// specified as is_node. If \p optional then \returns an empty string if \p
  /// node is null.
  const std::string &getInputNodeName(const folly::dynamic &node,
                                      bool optional = false) const;

  /// \returns whether the constant with the given \p name contains only zero.
  /// \p dtype is the type of this constant and \p size is the total size of the
  /// constant;
  bool isZeroes(const std::string &name, const utils::DTYPE &dtype,
                const size_t &size) const;

  /// \returns const references to allow partial and requires paddding names.
  const llvm::StringSet<> &getAllowPartialPlaceholderNames() const {
    return allowPartialPlaceholderNames_;
  }
  const llvm::StringSet<> &getRequiresPaddingPlaceholderNames() const {
    return requiresPaddingPlaceholderNames_;
  }

private:
  /// NNPI network handle.
  NNPINetwork network_;

  /// NNPI Device configuration.
  const glow::NNPICompilationOptions &compileOptions_;

  /// Mapping from constant name to a void pointer points to where the constant
  /// is actually stored.
  const llvm::StringMap<const void *> &constants_;

  /// Mapping from getattrs to the underlying name of the constants they alias.
  llvm::StringMap<const std::string> getattrs_;

  /// Mapping from output node names to the tensor descriptor for that node.
  llvm::StringMap<NNPITensorDesc> tensorDescs_;

  /// Set of tensors written to by the function.
  std::unordered_set<std::string> writeTensors_;

  /// Set of tensors read from by the function.
  std::unordered_set<std::string> readTensors_;

  /// Set of tensors already defined.
  std::unordered_set<std::string> definedTensors_;

  /// Sets of placeholder names that allow partial tensors or require padding.
  llvm::StringSet<> allowPartialPlaceholderNames_;
  llvm::StringSet<> requiresPaddingPlaceholderNames_;
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
