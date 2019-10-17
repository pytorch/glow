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
#ifndef GLOW_SUPPORT_COMPILER_H
#define GLOW_SUPPORT_COMPILER_H

#include <stdio.h>

#if !defined(__has_builtin)
#define __has_builtin(builtin) 0
#endif

#ifdef _WIN32
#define glow_aligned_malloc(p, a, s)                                           \
  (((*(p)) = _aligned_malloc((s), (a))), *(p) ? 0 : errno)
#define glow_aligned_free(p) _aligned_free(p)
#else
#define glow_aligned_malloc(p, a, s) posix_memalign(p, a, s)
#define glow_aligned_free(p) free(p)
#endif

#endif // GLOW_SUPPORT_COMPILER_H
