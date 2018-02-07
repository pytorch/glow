#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <sys/types.h>

#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

/// \returns the index of the element at x,y,z,w.
size_t getXYZW(const size_t *dims, size_t x, size_t y, size_t z, size_t w) {
  return (x * dims[1] * dims[2] * dims[3]) + (y * dims[2] * dims[3]) +
         (z * dims[3]) + w;
}

/// \returns the index of the element at x,y,z.
size_t getXYZ(const size_t *dims, size_t x, size_t y, size_t z) {
  return (x * dims[1] * dims[2]) + (y * dims[2]) + z;
}

/// \returns the index of the element at x,y.
size_t getXY(const size_t *dims, size_t x, size_t y) {
  return (x * dims[1]) + y;
}

void splat_f(float *buffer, size_t sz, float val) {
  for (size_t i = 0; i < sz; i++) {
    ((float *)buffer)[i] = val;
  }
}

void elementmax_f(float *dest, const float *LHS, const float *RHS, size_t sz) {
  for (size_t i = 0; i < sz; i++) {
    dest[i] = MAX(LHS[i], RHS[i]);
  }
}

void elementmax0_f(float *dest, const float *LHS, size_t sz) {
  for (size_t i = 0; i < sz; i++) {
    dest[i] = MAX(LHS[i], 0);
  }
}

void elementmin_f(float *dest, const float *LHS, const float *RHS, size_t sz) {
  for (size_t i = 0; i < sz; i++) {
    dest[i] = MIN(LHS[i], RHS[i]);
  }
}

void elementselect_f(float *dest, const float *cond, const float *LHS,
                     const float *RHS, size_t sz) {
  for (size_t i = 0; i < sz; i++) {
    dest[i] = (cond[i] != 0.0) ? LHS[i] : RHS[i];
  }
}

void batchedmatmul_f(float *dest, const float *LHS, const float *RHS,
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
          dest[getXYZ(destDims, n, x, y)] +=
              LHS[getXYZ(lhsDims, ln, x, i)] * RHS[getXYZ(rhsDims, rn, i, y)];
        }
      }
    }
  } // N
}

void batchedadd_f(float *dest, const float *batch, const float *slice,
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

void batchedreduceadd_f(float *dest, const float *batch, size_t destSize,
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

void copy_buffer(uint8_t *dest, uint8_t *src, size_t bytes) {
  for (int i = 0; i < bytes; i++) {
    dest[i] = src[i];
  }
}

void element_cmp_lte_f(float *dest, const float *LHS, const float *RHS,
                       size_t numElem) {
  for (size_t i = 0; i < numElem; i++) {
    dest[i] = LHS[i] < RHS[i];
  }
}

void element_sub_f(float *dest, const float *LHS, const float *RHS,
                   size_t numElem) {
  for (size_t i = 0; i < numElem; i++) {
    dest[i] = LHS[i] - RHS[i];
  }
}

void element_add_f(float *dest, const float *LHS, const float *RHS,
                   size_t numElem) {
  for (size_t i = 0; i < numElem; i++) {
    dest[i] = LHS[i] + RHS[i];
  }
}

void element_div_f(float *dest, const float *LHS, const float *RHS,
                   size_t numElem) {
  for (size_t i = 0; i < numElem; i++) {
    dest[i] = LHS[i] / RHS[i];
  }
}

void element_mul_f(float *dest, const float *LHS, const float *RHS,
                   size_t numElem) {
  for (size_t i = 0; i < numElem; i++) {
    dest[i] = LHS[i] * RHS[i];
  }
}

void convolution_f_unroll_k4(const float *inW, float *outW,
                             const float *filterW, const float *biasW,
                             const size_t *inWdims, const size_t *outWdims,
                             const size_t *filterWdims, const size_t *biasWdims,
                             size_t filterSize, size_t stride, size_t pad) {
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
                float in = inW[getXYZW(inWdims, n, (size_t)ox, (size_t)oy, fd)];
                sum0 += filterW[getXYZW(filterWdims, d + 0, fx, fy, fd)] * in;
              }
              for (size_t fd = 0; fd < inChannels; fd++) {
                float in = inW[getXYZW(inWdims, n, (size_t)ox, (size_t)oy, fd)];
                sum1 += filterW[getXYZW(filterWdims, d + 1, fx, fy, fd)] * in;
              }
              for (size_t fd = 0; fd < inChannels; fd++) {
                float in = inW[getXYZW(inWdims, n, (size_t)ox, (size_t)oy, fd)];
                sum2 += filterW[getXYZW(filterWdims, d + 2, fx, fy, fd)] * in;
              }
              for (size_t fd = 0; fd < inChannels; fd++) {
                float in = inW[getXYZW(inWdims, n, (size_t)ox, (size_t)oy, fd)];
                sum3 += filterW[getXYZW(filterWdims, d + 3, fx, fy, fd)] * in;
              }
            }
          }

          sum0 += biasW[d + 0];
          sum1 += biasW[d + 1];
          sum2 += biasW[d + 2];
          sum3 += biasW[d + 3];
          outW[getXYZW(outWdims, n, ax, ay, d + 0)] = sum0;
          outW[getXYZW(outWdims, n, ax, ay, d + 1)] = sum1;
          outW[getXYZW(outWdims, n, ax, ay, d + 2)] = sum2;
          outW[getXYZW(outWdims, n, ax, ay, d + 3)] = sum3;
        } // W
      }   // H
    }     // C
  }       // N
}

void convolution_f(const float *inW, float *outW, const float *filterW,
                   const float *biasW, const size_t *inWdims,
                   const size_t *outWdims, const size_t *filterWdims,
                   const size_t *biasWdims, size_t filterSize, size_t stride,
                   size_t pad) {

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

void pool_max_f(const float *inW, float *outW, const size_t *inWdims,
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

              float val = inW[getXYZW(inWdims, n, (size_t)ox, (size_t)oy, z)];

              if (first || (val >= max)) {
                first = 0;
                max = val;
              }
            }
          }

          outW[getXYZW(outWdims, n, ax, ay, z)] = max;
        } // W
      }   // H
    }     // C
  }       // N
}

void pool_max_xy_f(const float *inW, float *outW, size_t *inXY,
                   const size_t *inWdims, const size_t *outWdims, size_t kernel,
                   size_t stride, size_t pad) {
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

              float val = inW[getXYZW(inWdims, n, (size_t)ox, (size_t)oy, z)];
              if (first || (val >= max)) {
                first = 0;
                max = val;
                maxX = ox;
                maxY = oy;
              }
            }
          }

          outW[getXYZW(outWdims, n, ax, ay, z)] = max;
          // For the x and y argmax's, we use a 5-dimensional
          // tensor whose fifth dimension has size 2:
          size_t ix = 2 * getXYZW(outWdims, n, ax, ay, z);
          inXY[ix] = maxX;
          inXY[ix + 1] = maxY;
        } // W
      }   // H
    }     // C
  }       // N
}

void pool_max_xy_grad_f(float *inG, const float *outG, const size_t *inXY,
                        const size_t *inGdims, const size_t *outWdims) {
  // NHWC format is assumed
  for (size_t n = 0; n < outWdims[0]; n++) {
    for (size_t z = 0; z < outWdims[3]; z++) {
      // Clear inG
      for (size_t x = 0; x < inGdims[1]; x++) {
        for (size_t y = 0; y < inGdims[2]; y++) {
          inG[getXYZW(inGdims, n, x, y, z)] = 0.0;
        }
      }

      for (size_t ax = 0; ax < outWdims[1]; ax++) {
        for (size_t ay = 0; ay < outWdims[2]; ay++) {
          // For the x and y argmax's, we use a 5-dimensional
          // tensor whose fifth dimension has size 2:
          size_t ix = 2 * getXYZW(outWdims, n, ax, ay, z);
          size_t maxX = inXY[ix];
          size_t maxY = inXY[ix + 1];

          float df = outG[getXYZW(outWdims, n, ax, ay, z)];
          inG[getXYZW(inGdims, n, maxX, maxY, z)] += df;
        } // W
      }   // H
    }     // C
  }       // N
}

void pool_avg_f(const float *inW, float *outW, const size_t *inWdims,
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

              sum += inW[getXYZW(inWdims, n, (size_t)ox, (size_t)oy, z)];
            }
          }

          outW[getXYZW(outWdims, n, ax, ay, z)] = sum / filterArea;
        } // W
      }   // H
    }     // C
  }       // N
}

void pool_avg_grad_f(float *inG, const float *outG, const size_t *inGdims,
                     const size_t *outWdims, size_t kernel, size_t stride,
                     size_t pad) {
  float kernelArea = kernel * kernel;

  // NHWC format is assumed
  for (size_t n = 0; n < outWdims[0]; n++) {
    for (size_t z = 0; z < outWdims[3]; z++) {
      // Clear inG
      for (size_t x = 0; x < inGdims[1]; x++) {
        for (size_t y = 0; y < inGdims[2]; y++) {
          inG[getXYZW(inGdims, n, x, y, z)] = 0.0;
        }
      }

      ssize_t x = -(ssize_t)pad;
      for (size_t ax = 0; ax < outWdims[1]; x += stride, ax++) {
        ssize_t y = -(ssize_t)pad;
        for (size_t ay = 0; ay < outWdims[2]; y += stride, ay++) {
          float df = outG[getXYZW(outWdims, n, ax, ay, z)] / kernelArea;
          for (size_t kx = 0; kx < kernel; kx++) {
            for (size_t ky = 0; ky < kernel; ky++) {
              ssize_t ox = x + kx;
              ssize_t oy = y + ky;
              if (ox < 0 || oy < 0 || ox >= (ssize_t)inGdims[1] ||
                  oy >= (ssize_t)inGdims[2]) {
                continue;
              }
              inG[getXYZW(inGdims, n, (size_t)ox, (size_t)oy, z)] += df;
            }
          }
        } // W
      }   // H
    }     // C
  }       // N
}

void sgd_f(float *W, const float *G, float *Gsum, float L1Decay, float L2Decay,
           float learningRate, float momentum, size_t batchSize, size_t Wsize) {
  for (size_t i = 0; i < Wsize; i++) {
    float L1Grad = L1Decay * (W[i] > 0 ? 1 : -1);
    float L2Grad = L2Decay * W[i];
    float Gij = (L2Grad + L1Grad + G[i]) / batchSize;

    if (momentum > 0.0) {
      float dx = momentum * Gsum[i] - learningRate * Gij;
      Gsum[i] = dx;
      W[i] += dx;
    } else {
      W[i] -= learningRate * Gij;
    }
  }
}

void softmax_f(const float *inW, float *outW, const size_t *idim,
               const size_t *odim) {
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

void softmaxgrad_f(float *inG, float *outW, const size_t *selectedW,
                   const size_t *idim, const size_t *selectdim) {
  for (size_t n = 0; n < idim[0]; n++) {
    for (size_t i = 0; i < idim[1]; i++) {
      float delta = (selectedW[getXY(selectdim, n, 0)] == i);
      inG[getXY(idim, n, i)] = outW[getXY(idim, n, i)] - delta;
    }
  }
}

void sigmoid_f(const float *inW, float *outW, size_t numElem) {
  for (size_t i = 0; i < numElem; i++) {
    float e = expf(inW[i]);
    outW[i] = e / (e + 1);
  }
}

void tanh_f(const float *inW, float *outW, size_t numElem) {
  for (size_t i = 0; i < numElem; i++) {
    outW[i] = tanhf(inW[i]);
  }
}

void transpose_f(const float *inW, float *outW, const size_t *idim,
                 const size_t *odim, const size_t *shuffle, size_t numDims) {
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
