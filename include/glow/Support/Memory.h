#ifndef GLOW_SUPPORT_MEMORY_H
#define GLOW_SUPPORT_MEMORY_H

#include <cassert>
#include <cstdlib>

namespace glow {

inline void *alignedAlloc(size_t size, size_t align) {
  assert(align >= sizeof(void *) && "Invalid alignment");
  void *ptr;
  int res = posix_memalign(&ptr, align, size);
  assert(res == 0 && "posix_memalign failed");
  (void)res;
  assert((size_t)ptr % align == 0 && "Alignment failed");
  return ptr;
}

inline void alignedFree(void *p) { free(p); }

} // end namespace glow

#endif // GLOW_SUPPORT_MEMORY_H
