#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <sys/types.h>

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

void elementmax0_f(float *dest, float *LHS, size_t sz) {
  for (size_t i = 0; i < sz; i++) {
    dest[i] = MAX(LHS[i], 0);
  }
}

void batchedmatmul_f(float *dest, float *LHS, float *RHS, size_t *destDims,
                     size_t *lhsDims, size_t *rhsDims) {
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
        for (size_t i = 0; i < lhsDims[2]; i++) {
          sum +=
              LHS[getXYZ(lhsDims, ln, x, i)] * RHS[getXYZ(rhsDims, rn, i, y)];
        }
        dest[getXYZ(destDims, n, x, y)] = sum;
      }
    }
  } // N
}

void batchedadd_f(float *dest, float *batch, float *slice, size_t numSlice,
                  size_t sliceSize) {
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

void element_cmp_lte_f(float *dest, float *LHS, float *RHS, size_t numElem) {
  for (size_t i = 0; i < numElem; i++) {
    dest[i] = LHS[i] < RHS[i];
  }
}

void element_sub_f(float *dest, float *LHS, float *RHS, size_t numElem) {
  for (size_t i = 0; i < numElem; i++) {
    dest[i] = LHS[i] - RHS[i];
  }
}

void element_add_f(float *dest, float *LHS, float *RHS, size_t numElem) {
  for (size_t i = 0; i < numElem; i++) {
    dest[i] = LHS[i] + RHS[i];
  }
}

void element_div_f(float *dest, float *LHS, float *RHS, size_t numElem) {
  for (size_t i = 0; i < numElem; i++) {
    dest[i] = LHS[i] / RHS[i];
  }
}

void element_mul_f(float *dest, float *LHS, float *RHS, size_t numElem) {
  for (size_t i = 0; i < numElem; i++) {
    dest[i] = LHS[i] * RHS[i];
  }
}

void convolution_f(float *inW, float *outW, float *filterW, float *biasW,
                   size_t *inWdims, size_t *outWdims, size_t *filterWdims,
                   size_t *biasWdims, size_t filterSize, size_t pad,
                   size_t stride) {

  size_t inChannels = inWdims[3];

  // For each input in the batch:
  for (size_t n = 0; n < inWdims[0]; n++) {

    // For each layer in the output tensor:
    for (size_t d = 0; d < outWdims[3]; d++) {

      // For each convolution 'jump' in the input tensor:
      ssize_t x = -(ssize_t)pad;
      for (size_t ax = 0; ax < outWdims[1]; x += stride, ax++) {
        ssize_t y = -(ssize_t)pad;
        for (size_t ay = 0; ay < outWdims[2]; y += stride, ay++) {

          // For each element in the convolution-filter:
          float sum = 0;
          for (size_t fx = 0; fx < filterSize; fx++) {
            for (size_t fy = 0; fy < filterSize; fy++) {
              ssize_t ox = x + fx;
              ssize_t oy = y + fy;

              // Ignore index access below zero (this is due to padding).
              if (ox < 0 || oy < 0 || ox >= (ssize_t)inWdims[1] ||
                  oy >= (ssize_t)inWdims[2]) {
                continue;
              }
              for (size_t fd = 0; fd < inChannels; fd++) {
                sum += filterW[getXYZW(filterWdims, d, fx, fy, fd)] *
                       inW[getXYZW(inWdims, n, (size_t)ox, (size_t)oy, fd)];
              }
            }
          }

          sum += biasW[d];
          outW[getXYZW(outWdims, n, ax, ay, d)] = sum;
        } // W
      }   // H
    }     // C
  }       // N
}

void pool_max_f(float *inW, float *outW, size_t *inWdims, size_t *outWdims,
                size_t pad, size_t filterSize, size_t stride) {
  // For each input in the batch:
  for (size_t n = 0; n < outWdims[0]; n++) {

    // For each layer in the output tensor:
    for (size_t z = 0; z < inWdims[3]; z++) {
      // For each convolution 'jump' in the input tensor:
      ssize_t x = -(ssize_t)pad;
      for (size_t ax = 0; ax < outWdims[1]; x += stride, ax++) {
        ssize_t y = -(ssize_t)pad;
        for (size_t ay = 0; ay < outWdims[2]; y += stride, ay++) {
          int first = 1;
          float max = 0;

          for (size_t fx = 0; fx < filterSize; fx++) {
            for (size_t fy = 0; fy < filterSize; fy++) {
              ssize_t ox = x + fx;
              ssize_t oy = y + fy;

              // Ignore index access below zero (this is due to padding).
              if (ox < 0 || oy < 0 || ox >= (ssize_t)inWdims[1] ||
                  oy >= (ssize_t)inWdims[2]) {
                continue;
              }

              float val = inW[getXYZW(inWdims, n, (size_t)ox, (size_t)oy, z)];

              if (first || (val >= max)) {
                first = 0;
                max = val;
              }
            }
          }

          outW[getXYZW(outWdims, n, ax, ay, z)] = max;
        } // H
      }   // W
    }     // C
  }       // N
}

void pool_avg_f(float *inW, float *outW, size_t *inWdims, size_t *outWdims,
                size_t pad, size_t filterSize, size_t stride) {
  float filterArea = filterSize * filterSize;
  // For each input in the batch:
  for (size_t n = 0; n < outWdims[0]; n++) {
    // For each layer in the output tensor:
    for (size_t z = 0; z < inWdims[3]; z++) {
      // For each convolution 'jump' in the input tensor:
      ssize_t x = -(ssize_t)pad;
      for (size_t ax = 0; ax < outWdims[1]; x += stride, ax++) {
        ssize_t y = -(ssize_t)pad;
        for (size_t ay = 0; ay < outWdims[2]; y += stride, ay++) {
          float sum = 0;

          for (size_t fx = 0; fx < filterSize; fx++) {
            for (size_t fy = 0; fy < filterSize; fy++) {
              ssize_t ox = x + fx;
              ssize_t oy = y + fy;

              // Ignore index access below zero (this is due to padding).
              if (ox < 0 || oy < 0 || ox >= (ssize_t)inWdims[1] ||
                  oy >= (ssize_t)inWdims[2]) {
                continue;
              }

              sum += inW[getXYZW(inWdims, n, (size_t)ox, (size_t)oy, z)];
            }
          }

          outW[getXYZW(outWdims, n, ax, ay, z)] = sum / filterArea;
        } // H
      }   // W
    }     // C
  }       // N
}

void softmax_f(float *inW, float *outW, size_t *idim, size_t *odim) {
  for (size_t n = 0; n < idim[0]; n++) {
    float max = inW[getXY(idim, n, 0)];

    // Find Max.
    for (size_t i = 1; i < idim[1]; i++) {
      max = MAX(max, inW[getXY(idim, n, i)]);
    }

    float sum = 0;

    // Compute exp.
    for (size_t i = 0; i < idim[1]; i++) {
      float e = expf(inW[getXY(idim, n, i)] - max);
      sum += e;
      outW[getXY(odim, n, i)] = e;
    }

    // Normalize the output.
    for (size_t i = 0; i < idim[1]; i++) {
      outW[getXY(odim, n, i)] = outW[getXY(odim, n, i)] / sum;
    }
  } // N
}

void sigmoid_f(float *inW, float *outW, size_t *idim, size_t *odim) {
  for (size_t n = 0; n < idim[0]; ++n) {
    for (size_t i = 0; i < idim[1]; ++i) {
      float e = expf(inW[getXY(idim, n, i)]);
      outW[getXY(odim, n, i)] = e / (e + 1);
    }
  } // N
}

void tanh_f(float *inW, float *outW, size_t *idim, size_t *odim) {
  for (size_t n = 0; n < idim[0]; ++n) {
    for (size_t i = 0; i < idim[1]; ++i) {
      outW[getXY(odim, n, i)] = tanhf(inW[getXY(idim, n, i)]);
    }
  } // N
}

void transpose_f(float *inW, float *outW, size_t *idim, size_t *odim,
                 size_t *shuffle, size_t numDims) {
  // Source coordinate.
  size_t SC[4];

  if (numDims == 4) {
    for (size_t x = 0; x < odim[0]; x++)
      for (size_t y = 0; y < odim[1]; y++)
        for (size_t z = 0; z < odim[2]; z++)
          for (size_t w = 0; w < odim[3]; w++) {
            SC[shuffle[0]] = x;
            SC[shuffle[1]] = y;
            SC[shuffle[2]] = z;
            SC[shuffle[3]] = w;
            outW[getXYZW(odim, x, y, z, w)] =
                inW[getXYZW(idim, SC[0], SC[1], SC[2], SC[3])];
          }
    return;
  }
  if (numDims == 3) {
    for (size_t x = 0; x < odim[0]; x++)
      for (size_t y = 0; y < odim[1]; y++)
        for (size_t z = 0; z < odim[2]; z++) {
          SC[shuffle[0]] = x;
          SC[shuffle[1]] = y;
          SC[shuffle[2]] = z;
          outW[getXYZ(odim, x, y, z)] = inW[getXYZ(idim, SC[0], SC[1], SC[2])];
        }
    return;
  }
  if (numDims == 2) {
    for (size_t x = 0; x < odim[0]; x++)
      for (size_t y = 0; y < odim[1]; y++) {
        SC[shuffle[0]] = x;
        SC[shuffle[1]] = y;
        outW[getXY(odim, x, y)] = inW[getXY(idim, SC[0], SC[1])];
      }
    return;
  }
}
