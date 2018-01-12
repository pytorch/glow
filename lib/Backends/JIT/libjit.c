#include <stddef.h>
#include <stdint.h>

#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

/// \returns the index of the element at x,y,z,w.
size_t getXYZW(size_t *dims, size_t x, size_t y, size_t z, size_t w) {
  return (x * dims[1] * dims[2] * dims[3]) + (y * dims[2] * dims[3]) +
         (z * dims[3]) + w;
}

/// \returns the index of the element at x,y,z.
size_t getXYZ(size_t *dims, size_t x, size_t y, size_t z) {
  return (x * dims[1] * dims[2]) + (y * dims[2]) + z;
}

/// \returns the index of the element at x,y.
size_t getXY(size_t *dims, size_t x, size_t y) { return (x * dims[1]) + y; }

void splat_f(float *buffer, size_t sz, float val) {
  for (size_t i = 0; i < sz; i++) {
    ((float *)buffer)[i] = val;
  }
}

void elementmax_f(float *dest, float *LHS, float *RHS, size_t sz) {
  for (size_t i = 0; i < sz; i++) {
    dest[i] = MAX(LHS[i], RHS[i]);
  }
}

void batchedmatmul_f(float *dest, float *LHS, float *RHS,
                     size_t *destDims, size_t *lhsDims, size_t *rhsDims) {
  // For each layer in the batch:
  for (size_t n = 0; n < destDims[0]; n++) {
    // Broadcast tensors with a batch size of 1 by selecting the right slice.
    size_t ln = (lhsDims[0] == 1 ? 0 : n);
    size_t rn = (rhsDims[0] == 1 ? 0 : n);

    // For each (x,y) in the destination matrix:
    for (size_t x = 0; x < destDims[1]; x++) {
      for (size_t y = 0; y < destDims[2]; y++) {

        // Perform DOT on the row an column.
        float sum = 0;
        for (size_t i = 0; i < rhsDims[2]; i++) {
          sum +=
              LHS[getXYZ(lhsDims, ln, i, x)] * RHS[getXYZ(rhsDims, rn, y, i)];
        }
        dest[getXYZ(destDims, n, x, y)] = sum;
      }
    }
  } // N
}

void batchedadd_f(float *dest, float *batch, float *slice,
                  size_t numSlice, size_t sliceSize) {
  // For each layer in the batch:
  for (size_t n = 0; n < numSlice; n++) {
    size_t base = n * sliceSize;
    // For each element in the slice.
    for (size_t i = 0; i < sliceSize; i++) {
      dest[base + i] = batch[base + i] + slice[i];
    }
  }
}

void copy_buffer(uint8_t *dest, uint8_t *src, size_t bytes) {
  for (int i = 0; i < bytes; i++) {
    dest[i] = src[i];
  }
}
