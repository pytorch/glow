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
#ifndef GLOW_SUPPORT_MEMORY_H
#define GLOW_SUPPORT_MEMORY_H

#include "glow/Support/Compiler.h"

#include <glog/logging.h>

#include <cstdlib>

namespace glow {

/// The tensor payload is allocated to be aligned to this value.
constexpr unsigned TensorAlignment = 64;

/// Allocate \p size bytes of memory aligned to \p align bytes.
inline void *alignedAlloc(size_t size, size_t align) {
  DCHECK_GE(align, sizeof(void *)) << "Alignment too small.";
  DCHECK_EQ(align % sizeof(void *), 0)
      << "Alignment is not a multiple of the machine word size.";
  void *ptr;
  int res = glow_aligned_malloc(&ptr, align, size);
  CHECK_EQ(res, 0) << "posix_memalign failed";
  CHECK_EQ((size_t)ptr % align, 0) << "Alignment failed";
  return ptr;
}

/// Free aligned memory.
inline void alignedFree(void *p) { glow_aligned_free(p); }

/// Rounds up \p size to the nearest \p alignment.
inline size_t alignedSize(size_t size, size_t alignment) {
  size_t mod = size % alignment;
  return mod ? size + alignment - mod : size;
}

} // end namespace glow

#endif // GLOW_SUPPORT_MEMORY_H
