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
#ifndef GLOW_SUPPORT_DEBUG_H
#define GLOW_SUPPORT_DEBUG_H

namespace glow {

#if !defined(DISABLE_DEBUG_GLOW)

/// \returns true if \p type matches the activated debug type.
bool isGlowCurrentDebugType(const char *type);

/// Macro to perform debug actions when TYPE is activated.
#define DEBUG_GLOW_WITH_TYPE(TYPE, X)                                          \
  do {                                                                         \
    if (glow::DebugFlag || glow::isGlowCurrentDebugType(TYPE)) {               \
      X;                                                                       \
    }                                                                          \
  } while (false)

#else

#define DEBUG_GLOW_WITH_TYPE(TYPE, X)                                          \
  do {                                                                         \
  } while (false)

#endif

/// Set to true if '-debug-glow' command line option is specified.
extern bool DebugFlag;

/// DEBUG_GLOW macros.  Used to emit debug information.  Enabled via
/// '-debug-glow' or '-debug-glow-only'.
#define DEBUG_GLOW(X) DEBUG_GLOW_WITH_TYPE(DEBUG_TYPE, X)

} // namespace glow

#endif // GLOW_SUPPORT_DEBUG_H
