#ifndef GLOW_BACKENDS_CPU_LIBJIT_DEFS_H
#define GLOW_BACKENDS_CPU_LIBJIT_DEFS_H

#include <string.h>

typedef float float4 __attribute__((ext_vector_type(4)));
typedef float float8 __attribute__((ext_vector_type(8)));

/// Loads a simd float8 value from \p ptr.
#define LoadFloat8(PTR) *((const float8 *)(PTR))

/// Stores the simd float8 value to \p ptr.
#define StoreFloat8(PTR, VAL) *((float8 *)(PTR)) = (VAL);

/// Accumulate (+=) the simd float8 value to \p ptr.
#define AddFloat8(PTR, VAL) *((float8 *)(PTR)) += (VAL);

/// Broadcast the input value to a float8.
#define BroadcastFloat8(VAL) ((float8)(VAL))

#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define AT(tensor, dims, numDims, indices, numIndices)                         \
  tensor[get_element_ptr(tensor, dims, numDims, indices, numIndices)]

/// Perform an unaligned load of a float8 from a float pointer.
inline float8 LoaduFloat8(const float *p) {
  float8 res;
  memcpy(&res, p, sizeof(float8));
  return res;
}

/// Perform an unaligned store to a float pointer.
inline void StoreuFloat8(float *p, float8 v) {
  memcpy(p, &v, sizeof(float8));
}

/// Perform an unaligned addition to a float pointer.
inline void AdduFloat8(float *p, float8 v) {
  StoreuFloat8(p, LoaduFloat8(p) + v);
}

#endif // GLOW_BACKENDS_CPU_LIBJIT_DEFS_H
