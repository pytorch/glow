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

#ifndef GLOW_BACKENDS_HABANA_HABANAUTILS_H
#define GLOW_BACKENDS_HABANA_HABANAUTILS_H

#include "glow/Graph/Nodes.h"

#include <synapse.h>

namespace glow {
/// Given a synStatus \p status, evaluates to Error::success() if status
/// is synSuccess and evaluates to an Error otherwise.
#define chk_make_err(status)                                                   \
  status == synSuccess                                                         \
      ? Error::success()                                                       \
      : MAKE_ERR(                                                              \
            strFormat("Expected synStatus be synSuccess (%d), instead got %d", \
                      synSuccess, status))

/// Given a synStatus \p status, returns an Error::success() if status is
/// synSuccess and returns an Error otherwise.
#define chk(status)                                                            \
  do {                                                                         \
    auto res = (status);                                                       \
    if (res != synSuccess) {                                                   \
      return chk_make_err(res);                                                \
    }                                                                          \
  } while (0)

/// Given a synStatus \p status, checks that status == synSuccess and kills the
/// the program if not.
#define chk_kill(status)                                                       \
  CHECK_EQ((status), synSuccess) << "Expected synStatus be synSuccess"

const char *statusStr(synStatus status);

/// \returns true if \p dst allows \p src to be converted from 64 to 32 bits.
bool allows64To32Downcast(const Node *src, const Node *dst);

/// \returns true if \p V can be converted from a 64 to 32 bit value at runtime.
bool allows64To32Downcast(const Storage *V, const Function *F);

} // namespace glow

#endif // GLOW_BACKENDS_HABANA_HABANAUTILS_H
