#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/types.h>

#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define AT(tensor, dims, numDims, indices, numIndices)                         \
  tensor[get_element_ptr(tensor, dims, numDims, indices, numIndices)]

namespace {

/// \returns the index of the element at x,y,z,w.
size_t libjit_getXYZW(const size_t *dims, size_t x, size_t y, size_t z,
                      size_t w) {
  return (x * dims[1] * dims[2] * dims[3]) + (y * dims[2] * dims[3]) +
         (z * dims[3]) + w;
}

/// \returns the index of the element at x,y,z.
size_t libjit_getXYZ(const size_t *dims, size_t x, size_t y, size_t z) {
  return (x * dims[1] * dims[2]) + (y * dims[2]) + z;
}

/// \returns the index of the element at x,y.
size_t libjit_getXY(const size_t *dims, size_t x, size_t y) {
  return (x * dims[1]) + y;
}

template <class ElemTy>
static void libjit_dump_tensor_impl(ElemTy *tensor, size_t *dims,
                                    size_t numDims) {
  // Check for empty tensor.
  if (!numDims) {
    printf("[ Empty tensor ]\n");
    return;
  }

  // Output shape.
  printf("shape: ( ");
  for (int i = 0; i < numDims; ++i) {
    printf("%zu ", dims[i]);
  }
  printf(")\n");

  ElemTy mx = tensor[0];
  ElemTy mn = tensor[0];

  size_t size = 1;
  size_t sliceSize[numDims];
  for (int i = 0; i < numDims; ++i) {
    size *= dims[i];
  }

  for (int i = numDims - 1, curSliceSize = 1; i >= 0; --i) {
    sliceSize[i] = curSliceSize;
    curSliceSize *= dims[i];
  }

  for (size_t i = 0, e = size; i < e; i++) {
    mx = MAX(mx, tensor[i]);
    mn = MIN(mn, tensor[i]);
  }

  // Check for zero tensor.
  if (mn == .0 && mx == .0) {
    printf("[ Zero tensor ]\n");
    return;
  }

  // Output max and min.
  printf("max: %.3f  min: %.3f\n", (float)mx, (float)mn);

  const unsigned maxNumElem = 100;

  printf("[");

  for (size_t i = 0, e = MIN(maxNumElem, size); i < e; i++) {

    // Print one open brace at the beginning of every row, slice, and tensor.
    for (size_t j = 0, e = numDims - 1; numDims > 1 && j < e; j++) {
      if (i % sliceSize[j] == 0) {
        // This iteration of outer loop is a new row, slice or tensor.
        printf("[");
      }
    }

    // Print the value at the current index.
    printf("%.3f", (float)tensor[i]);

    // Print one closed brace at the end of every row, slice, or tensor.
    for (size_t j = 0, e = numDims - 1; numDims > 1 && j < e; j++) {
      size_t next_index = i + 1;
      if (next_index % sliceSize[j] == 0u) {
        printf("]");
      }
    }

    printf(", ");

    // Print one newline at the end of every row, slice, or tensor.
    for (size_t j = 0, e = numDims - 1; numDims > 1 && j < e; j++) {
      size_t next_index = i + 1;
      if (next_index % sliceSize[j] == 0u) {
        // Next iteration of outer loop will be a new row, slice or tensor.
        printf("\n");
      }
    }
  }

  if (size > maxNumElem) {
    printf("...");
  }

  printf("]\n");
}

template <typename ElemTy>
static size_t get_element_ptr(ElemTy *tensor, size_t *dims, size_t numDims,
                              size_t *indices, size_t numIndices) {
  size_t index = 0;
  size_t subdimensionSize = 1;
  for (size_t i = numDims; i > 0; i--) {
    size_t curIndicesValue = (i <= numIndices) ? indices[i - 1] : 0;
    index += subdimensionSize * curIndicesValue;
    subdimensionSize *= dims[i - 1];
  }
  return index;
}

template <typename ElemTy>
static void libjit_insert_tensor_impl(ElemTy *tensor, ElemTy *slice,
                                      size_t *offset, size_t *sliceCoor,
                                      size_t *fusedCoor, size_t *tensorDim,
                                      size_t *sliceDim, size_t numDimsTensor,
                                      size_t numDimsSliceCoor,
                                      size_t numDimsFusedCoor,
                                      unsigned isInsert, unsigned d) {
  unsigned isDone = (d == numDimsSliceCoor);

  if (isDone) {
    if (isInsert) {
      AT(tensor, tensorDim, numDimsTensor, fusedCoor, numDimsFusedCoor) =
          AT(slice, sliceDim, numDimsSliceCoor, sliceCoor, numDimsSliceCoor);
    } else {
      AT(slice, sliceDim, numDimsSliceCoor, sliceCoor, numDimsSliceCoor) =
          AT(tensor, tensorDim, numDimsTensor, fusedCoor, numDimsFusedCoor);
    }
    return;
  }

  for (size_t i = 0, e = sliceDim[d]; i < e; i++) {
    // Construct the coordinates for the slice and for the joint shape.
    // Add the 'offset' to the dimension that we concat the shapes on.
    sliceCoor[d] = i;
    fusedCoor[d] = i + offset[d];
    libjit_insert_tensor_impl(
        tensor, slice, offset, sliceCoor, fusedCoor, tensorDim, sliceDim,
        numDimsTensor, numDimsSliceCoor, numDimsFusedCoor, isInsert, d + 1);
  }
}

template <typename ElemTy>
void libjit_insert_tensor(ElemTy *tensor, ElemTy *slice, size_t *offset,
                          size_t *tensorDim, size_t *sliceDim,
                          size_t numDimsTensor, size_t numDimsSlice,
                          size_t offsetDim) {
  // Reserve statically enough memory to avoid dynamic memory allocation.
  size_t sliceCoor[10];
  size_t fusedCoor[10];
  memcpy(sliceCoor, sliceDim, sizeof(*sliceDim) * numDimsSlice);
  memcpy(fusedCoor, tensorDim, sizeof(*tensorDim) * numDimsTensor);
  libjit_insert_tensor_impl(tensor, slice, offset, sliceCoor, fusedCoor,
                            tensorDim, sliceDim, numDimsTensor, numDimsSlice,
                            offsetDim, 1, 0);
}

template <typename ElemTy>
void libjit_extract_tensor(ElemTy *tensor, ElemTy *slice, size_t *offset,
                           size_t *tensorDim, size_t *sliceDim,
                           size_t numDimsTensor, size_t numDimsSlice,
                           size_t offsetDim) {
  // Reserve statically enough memory to avoid dynamic memory allocation.
  size_t sliceCoor[10];
  size_t fusedCoor[10];
  memcpy(sliceCoor, sliceDim, sizeof(*sliceDim) * numDimsSlice);
  memcpy(fusedCoor, tensorDim, sizeof(*tensorDim) * numDimsTensor);
  libjit_insert_tensor_impl(tensor, slice, offset, sliceCoor, fusedCoor,
                            tensorDim, sliceDim, numDimsTensor, numDimsSlice,
                            offsetDim, 0, 0);
}

} // namespace

extern "C" {

void libjit_splat_f(float *buffer, size_t sz, float val) {
  for (size_t i = 0; i < sz; i++) {
    ((float *)buffer)[i] = val;
  }
}

void libjit_splat_u(size_t *buffer, size_t sz, float val) {
  for (size_t i = 0; i < sz; i++) {
    ((size_t *)buffer)[i] = val;
  }
}

void libjit_elementmax_f(float *dest, const float *LHS, const float *RHS,
                         size_t sz) {
  for (size_t i = 0; i < sz; i++) {
    dest[i] = MAX(LHS[i], RHS[i]);
  }
}

void libjit_elementmax0_f(float *dest, const float *LHS, size_t sz) {
  for (size_t i = 0; i < sz; i++) {
    dest[i] = MAX(LHS[i], 0);
  }
}

void libjit_elementmin_f(float *dest, const float *LHS, const float *RHS,
                         size_t sz) {
  for (size_t i = 0; i < sz; i++) {
    dest[i] = MIN(LHS[i], RHS[i]);
  }
}

void libjit_elementselect_f(float *dest, const float *cond, const float *LHS,
                            const float *RHS, size_t sz) {
  for (size_t i = 0; i < sz; i++) {
    dest[i] = (cond[i] != 0.0) ? LHS[i] : RHS[i];
  }
}

void libjit_batchedmatmul_f(float *dest, const float *LHS, const float *RHS,
                            const size_t *destDims, const size_t *lhsDims,
                            const size_t *rhsDims) {
  size_t destSize = destDims[0] * destDims[1] * destDims[2];
  for (size_t i = 0; i < destSize; ++i)
    dest[i] = 0;

  // For each layer in the batch:
  for (size_t n = 0; n < destDims[0]; n++) {
    // Broadcast tensors with a batch size of 1 by selecting the right slice.
    size_t ln = (lhsDims[0] == 1 ? 0 : n);
    size_t rn = (rhsDims[0] == 1 ? 0 : n);

    for (size_t i = 0; i < lhsDims[2]; i++) {
      // For each (x,y) in the destination matrix:
      for (size_t x = 0; x < destDims[1]; x++) {
        for (size_t y = 0; y < destDims[2]; y++) {
          // This loop order is very cache friendly.
          // dest and rhs are accessed sequentially.
          // lhs access is invariant inside the inner-most loop and can be
          // hoisted.
          dest[libjit_getXYZ(destDims, n, x, y)] +=
              LHS[libjit_getXYZ(lhsDims, ln, x, i)] *
              RHS[libjit_getXYZ(rhsDims, rn, i, y)];
        }
      }
    }
  } // N
}

void libjit_batchedadd_f(float *dest, const float *batch, const float *slice,
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

void libjit_batchedreduceadd_f(float *dest, const float *batch, size_t destSize,
                               size_t numSlice, size_t sliceSize) {
  for (size_t i = 0; i < destSize; i++) {
    dest[i] = 0.0;
  }
  for (size_t n = 0; n < numSlice; n++) {
    size_t base = n * sliceSize;
    for (size_t i = 0; i < sliceSize; i++) {
      dest[i] += batch[base + i];
    }
  }
}

void libjit_copy(void *dest, const void *src, size_t bytes) {
  memcpy(dest, src, bytes);
}

void libjit_element_cmp_lte_f(float *dest, const float *LHS, const float *RHS,
                              size_t numElem) {
  for (size_t i = 0; i < numElem; i++) {
    dest[i] = LHS[i] < RHS[i];
  }
}

void libjit_element_sub_f(float *dest, const float *LHS, const float *RHS,
                          size_t numElem) {
  for (size_t i = 0; i < numElem; i++) {
    dest[i] = LHS[i] - RHS[i];
  }
}

void libjit_element_add_f(float *dest, const float *LHS, const float *RHS,
                          size_t numElem) {
  for (size_t i = 0; i < numElem; i++) {
    dest[i] = LHS[i] + RHS[i];
  }
}

void libjit_element_div_f(float *dest, const float *LHS, const float *RHS,
                          size_t numElem) {
  for (size_t i = 0; i < numElem; i++) {
    dest[i] = LHS[i] / RHS[i];
  }
}

void libjit_element_mul_f(float *dest, const float *LHS, const float *RHS,
                          size_t numElem) {
  for (size_t i = 0; i < numElem; i++) {
    dest[i] = LHS[i] * RHS[i];
  }
}

void libjit_convolution_unroll_k4_f(
    const float *inW, float *outW, const float *filterW, const float *biasW,
    const size_t *inWdims, const size_t *outWdims, const size_t *filterWdims,
    const size_t *biasWdims, size_t filterSize, size_t stride, size_t pad) {
  size_t inChannels = inWdims[3];

  // For each input in the batch:
  for (size_t n = 0; n < inWdims[0]; n++) {

    // For each layer in the output tensor:
    for (size_t d = 0; d < outWdims[3]; d += 4) {

      // For each convolution 'jump' in the input tensor:
      ssize_t x = -(ssize_t)pad;
      for (size_t ax = 0; ax < outWdims[1]; x += stride, ax++) {
        ssize_t y = -(ssize_t)pad;
        for (size_t ay = 0; ay < outWdims[2]; y += stride, ay++) {

          // For each element in the convolution-filter:
          float sum0 = 0;
          float sum1 = 0;
          float sum2 = 0;
          float sum3 = 0;
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
                float in =
                    inW[libjit_getXYZW(inWdims, n, (size_t)ox, (size_t)oy, fd)];
                sum0 +=
                    filterW[libjit_getXYZW(filterWdims, d + 0, fx, fy, fd)] *
                    in;
              }
              for (size_t fd = 0; fd < inChannels; fd++) {
                float in =
                    inW[libjit_getXYZW(inWdims, n, (size_t)ox, (size_t)oy, fd)];
                sum1 +=
                    filterW[libjit_getXYZW(filterWdims, d + 1, fx, fy, fd)] *
                    in;
              }
              for (size_t fd = 0; fd < inChannels; fd++) {
                float in =
                    inW[libjit_getXYZW(inWdims, n, (size_t)ox, (size_t)oy, fd)];
                sum2 +=
                    filterW[libjit_getXYZW(filterWdims, d + 2, fx, fy, fd)] *
                    in;
              }
              for (size_t fd = 0; fd < inChannels; fd++) {
                float in =
                    inW[libjit_getXYZW(inWdims, n, (size_t)ox, (size_t)oy, fd)];
                sum3 +=
                    filterW[libjit_getXYZW(filterWdims, d + 3, fx, fy, fd)] *
                    in;
              }
            }
          }

          sum0 += biasW[d + 0];
          sum1 += biasW[d + 1];
          sum2 += biasW[d + 2];
          sum3 += biasW[d + 3];
          outW[libjit_getXYZW(outWdims, n, ax, ay, d + 0)] = sum0;
          outW[libjit_getXYZW(outWdims, n, ax, ay, d + 1)] = sum1;
          outW[libjit_getXYZW(outWdims, n, ax, ay, d + 2)] = sum2;
          outW[libjit_getXYZW(outWdims, n, ax, ay, d + 3)] = sum3;
        } // W
      }   // H
    }     // C
  }       // N
}

void libjit_convolution_f(const float *inW, float *outW, const float *filterW,
                          const float *biasW, const size_t *inWdims,
                          const size_t *outWdims, const size_t *filterWdims,
                          const size_t *biasWdims, size_t filterSize,
                          size_t stride, size_t pad) {
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
                sum +=
                    filterW[libjit_getXYZW(filterWdims, d, fx, fy, fd)] *
                    inW[libjit_getXYZW(inWdims, n, (size_t)ox, (size_t)oy, fd)];
              }
            }
          }

          sum += biasW[d];
          outW[libjit_getXYZW(outWdims, n, ax, ay, d)] = sum;
        } // W
      }   // H
    }     // C
  }       // N
}

void libjit_convolution_grad_f(float *inG, const float *outG, const float *inW,
                               float *filterG, float *biasG,
                               const float *filterW, const size_t *outGdims,
                               const size_t *inWdims, const size_t *filterGdims,
                               const size_t kernel, const size_t stride,
                               const size_t pad) {
  // NHWC format is assumed
  // Clear inG, filterG, and biasG
  size_t p = sizeof(float) * inWdims[3];
  memset(inG, 0, inWdims[0] * inWdims[1] * inWdims[2] * p);
  memset(filterG, 0, outGdims[3] * kernel * kernel * p);
  memset(biasG, 0, sizeof(float) * outGdims[3]);

  // For each input in the batch:
  for (size_t n = 0; n < outGdims[0]; n++) {
    for (size_t d = 0; d < outGdims[3]; d++) {
      ssize_t x = -(ssize_t)pad;
      for (size_t bx = 0; bx < outGdims[1]; bx++, x += stride) {
        ssize_t y = -(ssize_t)pad;
        for (size_t by = 0; by < outGdims[2]; by++, y += stride) {
          float grad = outG[libjit_getXYZW(outGdims, n, bx, by, d)];

          for (size_t kx = 0; kx < kernel; kx++) {
            for (size_t ky = 0; ky < kernel; ky++) {
              ssize_t ax = x + kx;
              ssize_t ay = y + ky;

              if (ax < 0 || ay < 0 || ax >= (ssize_t)inWdims[1] ||
                  ay >= (ssize_t)inWdims[2]) {
                continue;
              }

              for (size_t c = 0; c < inWdims[3]; c++) {
                inG[libjit_getXYZW(inWdims, n, (size_t)ax, (size_t)ay, c)] +=
                    filterW[libjit_getXYZW(filterGdims, d, kx, ky, c)] * grad;
                filterG[libjit_getXYZW(filterGdims, d, kx, ky, c)] +=
                    inW[libjit_getXYZW(inWdims, n, (size_t)ax, (size_t)ay, c)] *
                    grad;
              }
            }
          }

          biasG[d] += grad;
        } // W
      }   // H
    }     // C
  }       // N
}

void libjit_gather_f(float *dest, const float *data, const size_t *indices,
                     size_t numIndices, size_t sliceSize) {
  for (size_t i = 0; i < numIndices; i++) {
    size_t slice = indices[i];
    memcpy(dest + i * sliceSize, data + slice * sliceSize,
           sliceSize * sizeof(float));
  }
}

void libjit_local_response_normalization_f(float *outW, const float *inW,
                                           float *scaleCache,
                                           const size_t *outWdims,
                                           const size_t *inWdims,
                                           size_t halfWindow, float alpha,
                                           float beta, float k) {
  size_t window = 2 * halfWindow + 1;
  float normedAlpha = alpha / window;

  for (size_t n = 0; n < inWdims[0]; n++) {
    for (size_t h = 0; h < inWdims[1]; h++) {
      for (size_t w = 0; w < inWdims[2]; w++) {
        for (size_t c = 0; c < inWdims[3]; c++) {
          float m2 = 0.0;
          for (size_t i = (c >= halfWindow ? c - halfWindow : 0);
               i <= MIN(c + halfWindow, inWdims[3] - 1); i++) {
            float val = inW[libjit_getXYZW(inWdims, n, h, w, i)];
            m2 += val * val;
          }

          float scale = k + normedAlpha * m2;
          scaleCache[libjit_getXYZW(inWdims, n, h, w, c)] = scale;
          float normFactor = pow(scale, -beta);
          outW[libjit_getXYZW(outWdims, n, h, w, c)] =
              inW[libjit_getXYZW(inWdims, n, h, w, c)] * normFactor;
        } // C
      }   // W
    }     // H
  }       // N
}

void libjit_local_response_normalization_grad_f(
    float *inG, const float *outG, const float *inW, const float *outW,
    const float *scaleCache, const size_t *outWdims, size_t halfWindow,
    float alpha, float beta) {
  size_t window = 2 * halfWindow + 1;
  float normedAlpha = alpha / window;
  float coeff = 2 * normedAlpha * beta;

  for (size_t n = 0; n < outWdims[0]; n++) {
    for (size_t h = 0; h < outWdims[1]; h++) {
      for (size_t w = 0; w < outWdims[2]; w++) {
        // Prepare right half of sliding window based at c = 0
        float sum = 0.0;
        for (size_t i = 0; i < MIN(halfWindow, outWdims[3]); i++) {
          float outg = outG[libjit_getXYZW(outWdims, n, h, w, i)];
          float outw = outW[libjit_getXYZW(outWdims, n, h, w, i)];
          float scale = scaleCache[libjit_getXYZW(outWdims, n, h, w, i)];
          sum += outg * (outw / scale);
        }

        for (size_t c = 0; c < outWdims[3]; c++) {
          if (c > halfWindow) {
            size_t j = c - halfWindow - 1;
            float outg = outG[libjit_getXYZW(outWdims, n, h, w, j)];
            float outw = outW[libjit_getXYZW(outWdims, n, h, w, j)];
            float scale = scaleCache[libjit_getXYZW(outWdims, n, h, w, j)];
            sum -= outg * (outw / scale);
          }

          size_t j = c + halfWindow;
          if (j < outWdims[3]) {
            float outg = outG[libjit_getXYZW(outWdims, n, h, w, j)];
            float outw = outW[libjit_getXYZW(outWdims, n, h, w, j)];
            float scale = scaleCache[libjit_getXYZW(outWdims, n, h, w, j)];
            sum += outg * (outw / scale);
          }

          float outg = outG[libjit_getXYZW(outWdims, n, h, w, c)];
          float inw = inW[libjit_getXYZW(outWdims, n, h, w, c)];
          float scale = scaleCache[libjit_getXYZW(outWdims, n, h, w, c)];
          inG[libjit_getXYZW(outWdims, n, h, w, c)] =
              outg * pow(scale, -beta) - coeff * inw * sum;
        }
      } // W
    }   // H
  }     // N
}

void libjit_pool_max_f(const float *inW, float *outW, const size_t *inWdims,
                       const size_t *outWdims, size_t filterSize, size_t stride,
                       size_t pad) {
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

              float val =
                  inW[libjit_getXYZW(inWdims, n, (size_t)ox, (size_t)oy, z)];

              if (first || (val >= max)) {
                first = 0;
                max = val;
              }
            }
          }

          outW[libjit_getXYZW(outWdims, n, ax, ay, z)] = max;
        } // W
      }   // H
    }     // C
  }       // N
}

void libjit_pool_max_xy_f(const float *inW, float *outW, size_t *inXY,
                          const size_t *inWdims, const size_t *outWdims,
                          size_t kernel, size_t stride, size_t pad) {
  // For each input in the batch:
  for (size_t n = 0; n < outWdims[0]; n++) {
    // For each channel in the input:
    for (size_t z = 0; z < outWdims[3]; z++) {
      ssize_t x = -(ssize_t)pad;
      for (size_t ax = 0; ax < outWdims[1]; x += stride, ax++) {
        ssize_t y = -(ssize_t)pad;
        for (size_t ay = 0; ay < outWdims[2]; y += stride, ay++) {
          size_t maxX = x;
          size_t maxY = y;
          int first = 1;
          float max = 0;

          for (size_t kx = 0; kx < kernel; kx++) {
            for (size_t ky = 0; ky < kernel; ky++) {
              ssize_t ox = x + kx;
              ssize_t oy = y + ky;

              if (ox < 0 || oy < 0 || ox >= (ssize_t)inWdims[1] ||
                  oy >= (ssize_t)inWdims[2]) {
                continue;
              }

              float val =
                  inW[libjit_getXYZW(inWdims, n, (size_t)ox, (size_t)oy, z)];
              if (first || (val >= max)) {
                first = 0;
                max = val;
                maxX = ox;
                maxY = oy;
              }
            }
          }

          outW[libjit_getXYZW(outWdims, n, ax, ay, z)] = max;
          // For the x and y argmax's, we use a 5-dimensional
          // tensor whose fifth dimension has size 2:
          size_t ix = 2 * libjit_getXYZW(outWdims, n, ax, ay, z);
          inXY[ix] = maxX;
          inXY[ix + 1] = maxY;
        } // W
      }   // H
    }     // C
  }       // N
}

void libjit_pool_max_xy_grad_f(float *inG, const float *outG,
                               const size_t *inXY, const size_t *inGdims,
                               const size_t *outWdims) {
  // NHWC format is assumed
  for (size_t n = 0; n < outWdims[0]; n++) {
    for (size_t z = 0; z < outWdims[3]; z++) {
      // Clear inG
      for (size_t x = 0; x < inGdims[1]; x++) {
        for (size_t y = 0; y < inGdims[2]; y++) {
          inG[libjit_getXYZW(inGdims, n, x, y, z)] = 0.0;
        }
      }

      for (size_t ax = 0; ax < outWdims[1]; ax++) {
        for (size_t ay = 0; ay < outWdims[2]; ay++) {
          // For the x and y argmax's, we use a 5-dimensional
          // tensor whose fifth dimension has size 2:
          size_t ix = 2 * libjit_getXYZW(outWdims, n, ax, ay, z);
          size_t maxX = inXY[ix];
          size_t maxY = inXY[ix + 1];

          float df = outG[libjit_getXYZW(outWdims, n, ax, ay, z)];
          inG[libjit_getXYZW(inGdims, n, maxX, maxY, z)] += df;
        } // W
      }   // H
    }     // C
  }       // N
}

void libjit_pool_avg_f(const float *inW, float *outW, const size_t *inWdims,
                       const size_t *outWdims, size_t filterSize, size_t stride,
                       size_t pad) {
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

              sum += inW[libjit_getXYZW(inWdims, n, (size_t)ox, (size_t)oy, z)];
            }
          }

          outW[libjit_getXYZW(outWdims, n, ax, ay, z)] = sum / filterArea;
        } // W
      }   // H
    }     // C
  }       // N
}

void libjit_pool_avg_grad_f(float *inG, const float *outG,
                            const size_t *inGdims, const size_t *outWdims,
                            size_t kernel, size_t stride, size_t pad) {
  float kernelArea = kernel * kernel;

  // NHWC format is assumed
  for (size_t n = 0; n < outWdims[0]; n++) {
    for (size_t z = 0; z < outWdims[3]; z++) {
      // Clear inG
      for (size_t x = 0; x < inGdims[1]; x++) {
        for (size_t y = 0; y < inGdims[2]; y++) {
          inG[libjit_getXYZW(inGdims, n, x, y, z)] = 0.0;
        }
      }

      ssize_t x = -(ssize_t)pad;
      for (size_t ax = 0; ax < outWdims[1]; x += stride, ax++) {
        ssize_t y = -(ssize_t)pad;
        for (size_t ay = 0; ay < outWdims[2]; y += stride, ay++) {
          float df = outG[libjit_getXYZW(outWdims, n, ax, ay, z)] / kernelArea;
          for (size_t kx = 0; kx < kernel; kx++) {
            for (size_t ky = 0; ky < kernel; ky++) {
              ssize_t ox = x + kx;
              ssize_t oy = y + ky;
              if (ox < 0 || oy < 0 || ox >= (ssize_t)inGdims[1] ||
                  oy >= (ssize_t)inGdims[2]) {
                continue;
              }
              inG[libjit_getXYZW(inGdims, n, (size_t)ox, (size_t)oy, z)] += df;
            }
          }
        } // W
      }   // H
    }     // C
  }       // N
}

void libjit_quantize_i8(int8_t *outW, const float *inW, size_t numElem,
                        float scale, int32_t offset) {
  for (size_t i = 0; i < numElem; i++) {
    int32_t result = (int32_t)roundf(inW[i] / scale + offset);
    outW[i] = MAX(INT8_MIN, MIN(INT8_MAX, result));
  }
}

void libjit_dequantize_f(float *outW, const int8_t *inW, size_t numElem,
                         float scale, int32_t offset) {
  for (size_t i = 0; i < numElem; i++) {
    outW[i] = scale * (inW[i] - offset);
  }
}

void libjit_softmax_f(const float *inW, float *outW, const size_t *idim,
                      const size_t *odim) {
  for (size_t n = 0; n < idim[0]; n++) {
    float max = inW[libjit_getXY(idim, n, 0)];

    // Find Max.
    for (size_t i = 1; i < idim[1]; i++) {
      max = MAX(max, inW[libjit_getXY(idim, n, i)]);
    }

    float sum = 0;

    // Compute exp.
    for (size_t i = 0; i < idim[1]; i++) {
      float e = expf(inW[libjit_getXY(idim, n, i)] - max);
      sum += e;
      outW[libjit_getXY(odim, n, i)] = e;
    }

    // Normalize the output.
    for (size_t i = 0; i < idim[1]; i++) {
      outW[libjit_getXY(odim, n, i)] = outW[libjit_getXY(odim, n, i)] / sum;
    }
  } // N
}

void libjit_softmax_grad_f(float *inG, float *outW, const size_t *selectedW,
                           const size_t *idim, const size_t *selectdim) {
  for (size_t n = 0; n < idim[0]; n++) {
    for (size_t i = 0; i < idim[1]; i++) {
      float delta = (selectedW[libjit_getXY(selectdim, n, 0)] == i);
      inG[libjit_getXY(idim, n, i)] = outW[libjit_getXY(idim, n, i)] - delta;
    }
  }
}

void libjit_sigmoid_f(const float *inW, float *outW, size_t numElem) {
  for (size_t i = 0; i < numElem; i++) {
    float e = expf(inW[i]);
    outW[i] = e / (e + 1);
  }
}

void libjit_tanh_f(const float *inW, float *outW, size_t numElem) {
  for (size_t i = 0; i < numElem; i++) {
    outW[i] = tanhf(inW[i]);
  }
}

namespace {
struct value_index {
  float value;
  size_t index;
};

int value_index_sort(const void *va, const void *vb) {
  value_index *a = (value_index *)va;
  value_index *b = (value_index *)vb;
  if (a->value != b->value)
    return a->value > b->value ? -1 : 1;
  return a->index < b->index ? -1 : 1;
}
} // namespace

void libjit_topk_f(const float *input, float *values, size_t *indices,
                   size_t *scratch, size_t k, size_t n, size_t size) {
  size_t in = 0;
  size_t out = 0;

  value_index *buf = (value_index *)scratch;
  while (in < size) {
    for (size_t i = 0; i < n; i++) {
      buf[i].value = input[in++];
      buf[i].index = i;
    }
    qsort(buf, n, sizeof(value_index), value_index_sort);
    for (size_t i = 0; i < k; i++) {
      values[out] = buf[i].value;
      indices[out] = buf[i].index;
      out++;
    }
  }
}

void libjit_transpose_f(const float *inW, float *outW, const size_t *idim,
                        const size_t *odim, const size_t *shuffle,
                        size_t numDims) {
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
            outW[libjit_getXYZW(odim, x, y, z, w)] =
                inW[libjit_getXYZW(idim, SC[0], SC[1], SC[2], SC[3])];
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
          outW[libjit_getXYZ(odim, x, y, z)] =
              inW[libjit_getXYZ(idim, SC[0], SC[1], SC[2])];
        }
    return;
  }
  if (numDims == 2) {
    for (size_t x = 0; x < odim[0]; x++)
      for (size_t y = 0; y < odim[1]; y++) {
        SC[shuffle[0]] = x;
        SC[shuffle[1]] = y;
        outW[libjit_getXY(odim, x, y)] = inW[libjit_getXY(idim, SC[0], SC[1])];
      }
    return;
  }
}

void libjit_insert_tensor_f(float *tensor, float *slice, size_t *offset,
                            size_t *tensorDim, size_t *sliceDim,
                            size_t numDimsTensor, size_t numDimsSlice,
                            size_t offsetDim) {
  libjit_insert_tensor(tensor, slice, offset, tensorDim, sliceDim,
                       numDimsTensor, numDimsSlice, offsetDim);
}

void libjit_extract_tensor_f(float *tensor, float *slice, size_t *offset,
                             size_t *tensorDim, size_t *sliceDim,
                             size_t numDimsTensor, size_t numDimsSlice,
                             size_t offsetDim) {
  libjit_extract_tensor(tensor, slice, offset, tensorDim, sliceDim,
                        numDimsTensor, numDimsSlice, offsetDim);
}

void libjit_insert_tensor_u(size_t *tensor, size_t *slice, size_t *offset,
                            size_t *tensorDim, size_t *sliceDim,
                            size_t numDimsTensor, size_t numDimsSlice,
                            size_t offsetDim) {
  libjit_insert_tensor(tensor, slice, offset, tensorDim, sliceDim,
                       numDimsTensor, numDimsSlice, offsetDim);
}

void libjit_extract_tensor_u(size_t *tensor, size_t *slice, size_t *offset,
                             size_t *tensorDim, size_t *sliceDim,
                             size_t numDimsTensor, size_t numDimsSlice,
                             size_t offsetDim) {
  libjit_extract_tensor(tensor, slice, offset, tensorDim, sliceDim,
                        numDimsTensor, numDimsSlice, offsetDim);
}

__attribute__((noinline)) void
libjit_dump_tensor(uint8_t *tensor, size_t *tensorDim, size_t numDimsTensor,
                   size_t elemKind, const char *name) {
  printf("%s\n", name);
  /// This definition should match the defintion in Glow.
  enum ElemKind {
    FloatTy,
    Int8QTy,
    Int32QTy,
    IndexTy,
  };
  // Dump the content of a tensor.
  switch (elemKind) {
  case FloatTy:
    libjit_dump_tensor_impl((float *)tensor, tensorDim, numDimsTensor);
    break;
  case IndexTy:
    libjit_dump_tensor_impl((size_t *)tensor, tensorDim, numDimsTensor);
    break;
  default:
    printf("Dumping this type of payload is not supported: %zu\n", elemKind);
    break;
  }
}
} // extern "C"
