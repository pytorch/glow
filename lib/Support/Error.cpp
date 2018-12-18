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

#include "glow/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"

namespace glow {
llvm::ExitOnError exitOnErr("Encountered an error, exiting.\n");

std::string addFileAndLineToError(llvm::StringRef str, llvm::StringRef file,
                                  uint32_t line) {
  return llvm::formatv("Error at file {0} line {1} \"{2}\"", file, line, str);
}

char ModelLoadingErr::ID = 0;
char RuntimeErr::ID = 0;

llvm::Error doSomething() {
  RETURN_MODEL_LOADING_ERR("Glow doesn't support 3D convolution",
                           ModelLoadingErr::EC::UNSUPPORTED_SHAPE);
}

/*
// EXAMPLE FROM DEVICE MANAGER
using ReadyCB = std::function<void(llvm::Expected<DeviceNetworkID>)>;
using ResultCB = std::function<void(llvm::Expected<std::unique_ptr<Context>>)>;

addNetwork(networkId, module, [](llvm::Expected<DeviceNetworkID>
deviceNetworIdOrErr) { if (deviceNetworIdOrErr) {
    // do whatever this callback does
    signalResults(llvm::Error::success());
  } else {
    signalResults(std::move(deviceNetworIdOrErr.takeError()));
  }
}

*/
} // namespace glow
