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

#include "PyTorchFileLoader.h"
#include "glow/Support/Error.h"
#include "glow/Support/Support.h"

namespace glow {

PyTorchFileLoader::PyTorchFileLoader(
    const std::string &fileName,
    std::shared_ptr<torch::jit::script::Module> &module, llvm::Error *errPtr) {
  auto setup = [&]() -> llvm::Error {
    try {
      module = std::make_shared<torch::jit::script::Module>(
          torch::jit::load(fileName));
    } catch (const std::exception &x) {
      RETURN_ERR(strFormat("Cannot load model from file: %s, , reason: %s",
                           fileName.c_str(), x.what()));
    }
    return llvm::Error::success();
  };

  if (errPtr) {
    *errPtr = setup();
  } else {
    EXIT_ON_ERR(setup());
  }
}

} // namespace glow
