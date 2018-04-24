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
