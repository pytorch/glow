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

#include "GetNNPIKernels.h"
#include "glow/Flags/Flags.h"

namespace glow {

std::string GetNNPIKernels::getCompiledIAKernelsFilePath() {
  std::string path = glow::nnpi::flags::InjectedIAOpKernelPath;

  // If the InjectedIAOpKernelPath flag was set then we should use that path.
  if (!path.empty()) {
    return path;
  }

#if FACEBOOK_INTERNAL
  path = GetNNPIKernels::getCompiledIAKernelsFilePathInternal();
#endif

  return path;
}

std::string GetNNPIKernels::getCompiledDSPKernelsFilePath() {
  // We don't need to set a default path value with a gflag because we already
  // have NNPI_CustomDSPLib environment variable for setting the DSP library
  // file manually.
  std::string path;

#if FACEBOOK_INTERNAL
  path = GetNNPIKernels::getCompiledDSPKernelsFilePathInternal();
#endif

  return path;
}

} // namespace glow
