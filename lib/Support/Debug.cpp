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
/// passes.
static llvm::cl::opt<std::string, true>
    DebugGlowOnly("debug-glow-only",
                  llvm::cl::desc("Enable a specific type of debug output"),
                  llvm::cl::Hidden, llvm::cl::location(DebugOnlyType));

namespace glow {

/// Exported boolean set by -debug-glow option.
bool DebugFlag = false;

bool isCurrentDebugType(const char *type) { return DebugOnlyType == type; }

} // namespace glow
