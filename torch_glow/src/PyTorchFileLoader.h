/**
 * Copyright (c) 2017-present, Facebook, Inc.
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

#ifndef GLOW_IMPORTER_PYTORCH_PYTORCHFILELOADER_H
#define GLOW_IMPORTER_PYTORCH_PYTORCHFILELOADER_H

#include "llvm/Support/Error.h"
#include <torch/csrc/jit/import.h>

/// Loads PyTorch model from file to JIT IR subgraphs.
class PyTorchFileLoader {
public:
  /// Takes a model file \p fileName, loads model into torch Module \p module,
  /// reports error assigning \p errPtr.
  PyTorchFileLoader(const std::string &fileName,
                    std::shared_ptr<torch::jit::script::Module> &module,
                    llvm::Error *errPtr);
};

#endif // GLOW_IMPORTER_PYTORCH_PYTORCHFILELOADER_H
