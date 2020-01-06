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

#ifndef GLOW_TORCH_GLOW_SRC_CUSTOMPYTORCHOPLOADER_H
#define GLOW_TORCH_GLOW_SRC_CUSTOMPYTORCHOPLOADER_H

#include "glow/Support/Error.h"

#include <torch/csrc/jit/ir.h>

namespace glow {
class PyTorchModelLoader;

/// Base class for implementing loading code for custom PyTorch ops. To load a
/// custom PyTorch op, extend this class and implement the pure virtual
/// functions.
class CustomPyTorchOpLoader {
public:
  /// \returns a list of pytorch qualified strings of the symbols for the kinds
  /// of ops that this loader loads (for example "something::my_awesome_op")
  virtual std::vector<const char *> getSymbols() = 0;

  /// Given a PyTorchModelLoader \p and a specific custom op JIT node \p ptNode,
  /// this implements the logic to load the custom op. \returns an Error on
  /// failure.
  virtual Error loadNode(PyTorchModelLoader &modelLoader,
                         const torch::jit::Node *ptNode) = 0;

  virtual ~CustomPyTorchOpLoader() = default;
};

/// From the list of registered CustomPyTorchOpLoaders, this \returns a map from
/// Symbol to pointer of CustomPyTorchOpLoader that loads nodes with that
/// symbol.
/// NOTE: while other methods are thread safe, this one is not because it is
/// possible for the map to be written to while a reference is held by a reader.
const std::unordered_map<torch::jit::Symbol, CustomPyTorchOpLoader *> &
getCustomPyTorchOpLoaders();

/// \returns a pointer to the CustomPyTorchOpLoader registered for the symbol \p
/// symbol or nullptr if no CustomPyTorchOpLoader was registered for this
/// symbol.
CustomPyTorchOpLoader *
getCustomPyTorchOpLoaderForSymbol(torch::jit::Symbol symbol);

/// Struct for statically registering CustomPyTorchOpLoaders.
struct RegisterCustomPyTorchOpLoader {
  explicit RegisterCustomPyTorchOpLoader(
      std::unique_ptr<CustomPyTorchOpLoader> loader);
};

} // namespace glow

#endif // GLOW_TORCH_GLOW_SRC_CUSTOMPYTORCHOPLOADER_H
