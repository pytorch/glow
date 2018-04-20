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
#ifndef GLOW_SUPPORT_MEMORY_H
#define GLOW_SUPPORT_MEMORY_H

#include <cassert>
#include <cstdlib>

namespace glow {

/// The tensor payload is allocated to be aligned to this value.
constexpr unsigned TensorAlignment = 64;

/// Allocate \p size bytes of memory aligned to \p align bytes.
inline void *alignedAlloc(size_t size, size_t align) {
  assert(align >= sizeof(void *) && "Alignment too small.");
  assert(align % sizeof(void *) == 0 &&
         "Alignment is not a multiple of the machine word size.");
  void *ptr;
  int res = posix_memalign(&ptr, align, size);
  assert(res == 0 && "posix_memalign failed");
  (void)res;
  assert((size_t)ptr % align == 0 && "Alignment failed");
  return ptr;
}

/// Free aligned memory.
inline void alignedFree(void *p) { free(p); }

/// Rounds up \p size to the nearest \p alignment.
inline size_t alignedSize(size_t size, size_t alignment) {
  return (size + alignment - 1) & ~(alignment - 1);
}

} // end namespace glow

#endif // GLOW_SUPPORT_MEMORY_H
