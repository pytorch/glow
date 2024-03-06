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

#include "glow/Support/Debug.h"

#include "llvm/Support/CommandLine.h"

#include <string>

using namespace glow;

static std::string DebugOnlyType;

/// -debug-glow - Command line option to enable DEBUG_GLOW statements.
static llvm::cl::opt<bool, true>
    DebugGlow("debug-glow", llvm::cl::desc("Enable debug output"),
              llvm::cl::Hidden, llvm::cl::location(DebugFlag));

/// -debug-glow-only - Command line option to enable debug output for specific
/// debug types. Multiple comma-separated debug types names can be provided.
static llvm::cl::list<std::string>
    DebugGlowOnly("debug-glow-only",
                  llvm::cl::desc("Enable specific types of debug output"),
                  llvm::cl::CommaSeparated, llvm::cl::Hidden);

namespace glow {

/// Exported boolean set by -debug-glow option.
bool DebugFlag = false;

#if !defined(DISABLE_DEBUG_GLOW)
bool isGlowCurrentDebugType(const char *type) {
  return std::find(DebugGlowOnly.begin(), DebugGlowOnly.end(), type) !=
         DebugGlowOnly.end();
}
#endif

} // namespace glow
