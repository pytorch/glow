
static const char* SHADER_CODE = R"(

typedef struct {
  unsigned long n; // Number of samples
  unsigned long h; // Height
  unsigned long w; // Width
  unsigned long c; // Number of channels
} ShapeNHWC;

/// \returns the index of the element at x,y,z,w.
unsigned long getNHWC(ShapeNHWC s, unsigned long x, unsigned long y, unsigned long z, unsigned long w) {
  return (x * s.c * s.w * s.h) + (y * s.c * s.w) + (z * s.c) + w;
}

__kernel void batchedreduceaddK(__global float *dest, __global float *batch,
                                unsigned long numSlice, unsigned long sliceSize) {
  unsigned long s = get_global_id(0);
  dest[s] = 0;
  for (unsigned long n = 0; n < numSlice; n++) {
    dest[s] += batch[n * sliceSize + s];
  }
}

__kernel void batchedreduceaddW(__global void *mem, unsigned long dest, unsigned long batch,
                                unsigned long numSlice, unsigned long sliceSize) {
  batchedreduceaddK(&mem[dest], &mem[batch], numSlice, sliceSize);
}

__kernel void batchedaddK(__global float *dest, __global float *batch,
                          __global float *slice, unsigned long numSlice,
                          unsigned long sliceSize) {
  unsigned long s = get_global_id(0);
  for (unsigned long n = 0; n < numSlice; n++) {
    dest[n * sliceSize + s] = batch[n * sliceSize + s] + slice[s];
  }
}

__kernel void batchedaddW(__global void *mem, unsigned long dest, unsigned long batch,
                          unsigned long slice, unsigned long numSlice, unsigned long sliceSize) {
  batchedaddK(&mem[dest], &mem[batch], &mem[slice], numSlice, sliceSize);
}

__kernel void batchedmatmulK(__global float *dest, __global float *lhs,
                             __global float *rhs, ShapeNHWC ddim,
                             ShapeNHWC ldim, ShapeNHWC rdim) {
  // For each layer in the batch.
  unsigned long n = get_global_id(0);
  // For each X in the destination matrix.
  unsigned long x = get_global_id(1);
  // For each Y in the destination matrix.
  unsigned long y = get_global_id(2);

  // Broadcast tensors with a batch size of 1 by selecting the right slice.
  unsigned long ln = (ldim.n == 1 ? 0 : n);
  unsigned long rn = (rdim.n == 1 ? 0 : n);

  // Perform DOT on the row an column.
  float sum = 0;
  for (unsigned long i = 0; i < ldim.w; i++) {
    sum += lhs[getNHWC(ldim, ln, x, i, 0)] * rhs[getNHWC(rdim, rn, i, y, 0)];
  }

  dest[getNHWC(ddim, n, x, y, 0)] = sum;
}

__kernel void batchedmatmulW(__global void *mem, unsigned long dest, unsigned long lhs,
                             unsigned long rhs, ShapeNHWC ddim, ShapeNHWC ldim,
                             ShapeNHWC rdim) {
  batchedmatmulK(&mem[dest], &mem[lhs], &mem[rhs], ddim, ldim, rdim);
}

__kernel void splatK(__global float *dest, float val) {
  unsigned long i = get_global_id(0);
  dest[i] = val;
}

__kernel void splatW(__global void *mem, unsigned long dest, float val) {
  splatK(&mem[dest], val);
}

__kernel void sigmoidK(__global float *dest, __global float *src) {
  unsigned long i = get_global_id(0);
  dest[i] = 1 / (1 + exp(-src[i]));
}

__kernel void sigmoidW(__global void *mem, unsigned long dest, unsigned long src) {
  sigmoidK(&mem[dest], &mem[src]);
}

__kernel void tanhK(__global float *dest, __global float *src) {
  unsigned long i = get_global_id(0);
  float val = src[i];
  float exp_val = exp(val);
  float exp_neg_val = exp(-val);
  dest[i] = (exp_val - exp_neg_val) / (exp_val + exp_neg_val);
}

__kernel void tanhW(__global void *mem, unsigned long dest, unsigned long src) {
  tanhK(&mem[dest], &mem[src]);
}

__kernel void elementaddK(__global float *dest, __global float *LHS,
                          __global float *RHS) {
  unsigned long i = get_global_id(0);
  dest[i] = LHS[i] + RHS[i];
}

__kernel void elementaddW(__global void *mem, unsigned long dest, unsigned long LHS,
                          unsigned long RHS) {
  elementaddK(&mem[dest], &mem[LHS], &mem[RHS]);
}

__kernel void elementmaxK(__global float *dest, __global float *LHS,
                          __global float *RHS) {
  unsigned long i = get_global_id(0);
  dest[i] = max(LHS[i], RHS[i]);
}

__kernel void elementmaxW(__global void *mem, unsigned long dest, unsigned long LHS,
                          unsigned long RHS) {
  elementmaxK(&mem[dest], &mem[LHS], &mem[RHS]);
}

__kernel void elementminK(__global float *dest, __global float *LHS,
                          __global float *RHS) {
  unsigned long i = get_global_id(0);
  dest[i] = min(LHS[i], RHS[i]);
}

__kernel void elementminW(__global void *mem, unsigned long dest, unsigned long LHS,
                          unsigned long RHS) {
  elementminK(&mem[dest], &mem[LHS], &mem[RHS]);
}

__kernel void elementcmplteK(__global float *dest, __global float *LHS,
                             __global float *RHS) {
  unsigned long i = get_global_id(0);
  dest[i] = LHS[i] <= RHS[i];
}

__kernel void elementcmplteW(__global void *mem, unsigned long dest, unsigned long LHS,
                             unsigned long RHS) {
  elementcmplteK(&mem[dest], &mem[LHS], &mem[RHS]);
}

__kernel void elementsubK(__global float *dest, __global float *LHS,
                          __global float *RHS) {
  unsigned long i = get_global_id(0);
  dest[i] = LHS[i] - RHS[i];
}

__kernel void elementsubW(__global void *mem, unsigned long dest, unsigned long LHS,
                          unsigned long RHS) {
  elementsubK(&mem[dest], &mem[LHS], &mem[RHS]);
}

__kernel void elementmulK(__global float *dest, __global float *LHS,
                          __global float *RHS) {
  unsigned long i = get_global_id(0);
  dest[i] = LHS[i] * RHS[i];
}

__kernel void elementmulW(__global void *mem, unsigned long dest, unsigned long LHS,
                          unsigned long RHS) {
  elementmulK(&mem[dest], &mem[LHS], &mem[RHS]);
}

__kernel void elementdivK(__global float *dest, __global float *LHS,
                          __global float *RHS) {
  unsigned long i = get_global_id(0);
  dest[i] = LHS[i] / RHS[i];
}

__kernel void elementdivW(__global void *mem, unsigned long dest, unsigned long LHS,
                          unsigned long RHS) {
  elementdivK(&mem[dest], &mem[LHS], &mem[RHS]);
}

__kernel void softmaxK(__global float *dest, __global float *src,
                       __global float *e_cache, unsigned long sliceSize) {
  unsigned long i = get_global_id(0);
  float max_ = src[i * sliceSize];
  for (unsigned long j = 0; j < sliceSize; j++) {
    max_ = max(max_, src[i * sliceSize + j]);
  }
  float sum = 0;
  for (unsigned long j = 0; j < sliceSize; j++) {
    float e = exp(src[i * sliceSize + j] - max_);
    sum += e;
    dest[i * sliceSize + j] = e;
  }
  for (unsigned long j = 0; j < sliceSize; j++) {
    dest[i * sliceSize + j] /= sum;
    if (e_cache)
      e_cache[i * sliceSize + j] = dest[i * sliceSize + j];
  }
}

__kernel void softmaxW(__global void *mem, unsigned long dest, unsigned long src,
                       unsigned long sliceSize) {
  softmaxK(&mem[dest], &mem[src], (__global float *)0, sliceSize);
}

__kernel void convolutionK(__global float *dest, __global float *src,
                           __global float *filter, __global float *bias,
                           unsigned long filterSize, unsigned long pad, unsigned long stride,
                           ShapeNHWC odim, ShapeNHWC idim,
                           ShapeNHWC filterDim) {
  unsigned long ax = get_global_id(0);
  unsigned long ay = get_global_id(1);
  unsigned long d = get_global_id(2);

  typedef int long;
  // For each convolution 'jump' in the input tensor:
  long x = -(long)pad + ax * stride;
  long y = -(long)pad + ay * stride;

  // For each input in the batch:
  for (unsigned long n = 0; n < idim.n; n++) {

    // For each element in the convolution-filter:
    float sum = 0;
    for (unsigned long fx = 0; fx < filterSize; fx++) {
      for (unsigned long fy = 0; fy < filterSize; fy++) {
        long ox = x + fx;
        long oy = y + fy;

        // Ignore index access below zero (this is due to padding).
        if (ox < 0 || oy < 0 || ox >= (long)idim.h ||
            oy >= (long)idim.w) {
          continue;
        }

        for (unsigned long fd = 0; fd < idim.c; fd++) {
          sum += filter[getNHWC(filterDim, d, fx, fy, fd)] *
                 src[getNHWC(idim, n, (unsigned long)ox, (unsigned long)oy, fd)];
        }
      }
    }

    sum += bias[d];
    dest[getNHWC(odim, n, ax, ay, d)] = sum;
  } // N
}

__kernel void convolutionW(__global void *mem, unsigned long dest, unsigned long src,
                           unsigned long filter, unsigned long bias, unsigned long filterSize,
                           unsigned long pad, unsigned long stride, ShapeNHWC odim,
                           ShapeNHWC idim, ShapeNHWC filterDim) {
  convolutionK(&mem[dest], &mem[src], &mem[filter], &mem[bias], filterSize, pad,
               stride, odim, idim, filterDim);
}

__kernel void poolmaxK(__global float *dest, __global float *src,
                       unsigned long filterSize, unsigned long pad,
                       unsigned long stride, ShapeNHWC odim, ShapeNHWC idim) {
  unsigned long ax = get_global_id(0);
  unsigned long ay = get_global_id(1);
  unsigned long d = get_global_id(2);

  typedef int long;
  // For each convolution 'jump' in the input tensor:
  long x = -(long)pad + ax * stride;
  long y = -(long)pad + ay * stride;

  // For each input in the batch:
  for (unsigned long n = 0; n < idim.n; n++) {
    float maxVal = 0;
    bool first = true;

    // For each element in the convolution-filter:
    for (unsigned long fx = 0; fx < filterSize; fx++) {
      for (unsigned long fy = 0; fy < filterSize; fy++) {
        long ox = x + fx;
        long oy = y + fy;

        // Ignore index access below zero (this is due to padding).
        if (ox < 0 || oy < 0 || ox >= (long)idim.h ||
            oy >= (long)idim.w) {
          continue;
        }

        float val = src[getNHWC(idim, n, (unsigned long)ox, (unsigned long)oy, d)];

        if (first || (val >= maxVal)) {
          first = false;
          maxVal = val;
        }
      }
    }
    dest[getNHWC(odim, n, ax, ay, d)] = maxVal;
  } // N
}

__kernel void poolmaxW(__global void *mem, unsigned long dest, unsigned long src,
                       unsigned long filterSize, unsigned long pad,
                       unsigned long stride, ShapeNHWC odim, ShapeNHWC idim) {
  poolmaxK(&mem[dest], &mem[src], filterSize, pad, stride, odim,
           idim);
}

__kernel void poolmaxwithxyK(__global float *dest, __global float *src,
                             __global float *srcXY, unsigned long filterSize,
                             unsigned long pad, unsigned long stride, ShapeNHWC odim,
                             ShapeNHWC idim) {
  unsigned long ax = get_global_id(0);
  unsigned long ay = get_global_id(1);
  unsigned long d = get_global_id(2);

  typedef int long;
  // For each convolution 'jump' in the input tensor:
  long x = -(long)pad + ax * stride;
  long y = -(long)pad + ay * stride;

  // For each input in the batch:
  for (unsigned long n = 0; n < idim.n; n++) {
    float maxVal = 0;
    bool first = true;

    // For each element in the convolution-filter:
    for (unsigned long fx = 0; fx < filterSize; fx++) {
      for (unsigned long fy = 0; fy < filterSize; fy++) {
        long ox = x + fx;
        long oy = y + fy;

        // Ignore index access below zero (this is due to padding).
        if (ox < 0 || oy < 0 || ox >= (long)idim.h ||
            oy >= (long)idim.w) {
          continue;
        }

        float val = src[getNHWC(idim, n, (unsigned long)ox, (unsigned long)oy, d)];

        if (first || (val >= maxVal)) {
          first = false;
          maxVal = val;
        }
      }
    }
    dest[getNHWC(odim, n, ax, ay, d)] = maxVal;
  } // N
}

__kernel void poolmaxwithxyW(__global void *mem, unsigned long dest, unsigned long src,
                             unsigned long srcXY, unsigned long filterSize, unsigned long pad,
                             unsigned long stride, ShapeNHWC odim, ShapeNHWC idim) {
  poolmaxwithxyK(&mem[dest], &mem[src], &mem[srcXY], filterSize, pad, stride,
                 odim, idim);
}

__kernel void poolavgK(__global float *dest, __global float *src,
                       unsigned long filterSize, unsigned long pad, unsigned long stride,
                       ShapeNHWC odim, ShapeNHWC idim) {
  unsigned long ax = get_global_id(0);
  unsigned long ay = get_global_id(1);
  unsigned long d = get_global_id(2);

  typedef int long;
  // For each convolution 'jump' in the input tensor:
  long x = -(long)pad + ax * stride;
  long y = -(long)pad + ay * stride;

  float filterArea = filterSize * filterSize;

  // For each input in the batch:
  for (unsigned long n = 0; n < idim.n; n++) {
    float sumVal = 0;
    // For each element in the convolution-filter:
    for (unsigned long fx = 0; fx < filterSize; fx++) {
      for (unsigned long fy = 0; fy < filterSize; fy++) {
        long ox = x + fx;
        long oy = y + fy;

        // Ignore index access below zero (this is due to padding).
        if (ox < 0 || oy < 0 || ox >= (long)idim.h ||
            oy >= (long)idim.w) {
          continue;
        }

        sumVal += src[getNHWC(idim, n, (unsigned long)ox, (unsigned long)oy, d)];
      }
    }
    dest[getNHWC(odim, n, ax, ay, d)] = sumVal / filterArea;
  } // N
}

__kernel void poolavgW(__global void *mem, unsigned long dest, unsigned long src,
                       unsigned long filterSize, unsigned long pad, unsigned long stride,
                       ShapeNHWC odim, ShapeNHWC idim) {
  poolavgK(&mem[dest], &mem[src], filterSize, pad, stride, odim, idim);
}

__kernel void transposeK(__global float *dest, __global float *src,
                         ShapeNHWC odim, ShapeNHWC idim, ShapeNHWC shuffle) {
  unsigned long d0 = get_global_id(0);
  unsigned long res[4];
  res[0] = d0;

  for (unsigned long d1 = 0; d1 < idim.h; d1++) {
    res[1] = d1;
    for (unsigned long d2 = 0; d2 < idim.w; d2++) {
      res[2] = d2;
      for (unsigned long d3 = 0; d3 < idim.c; d3++) {
        res[3] = d3;
        unsigned long dstIdx = getNHWC(odim, res[shuffle.n], res[shuffle.h],
                                res[shuffle.w], res[shuffle.c]);
        unsigned long srcIdx = getNHWC(idim, d0, d1, d2, d3);
        dest[dstIdx] = src[srcIdx];
      }
    }
  }
}

__kernel void transposeW(__global void *mem, unsigned long dest, unsigned long src,
                         ShapeNHWC odim, ShapeNHWC idim, ShapeNHWC shuffle) {
  transposeK(&mem[dest], &mem[src], odim, idim, shuffle);
}

__kernel void inserttensorK(__global float *dest, __global float *src,
                            ShapeNHWC odim, ShapeNHWC idim, ShapeNHWC offset) {
  unsigned long d0 = get_global_id(0);
  for (unsigned long d1 = 0; d1 < idim.h; d1++) {
    for (unsigned long d2 = 0; d2 < idim.w; d2++) {
      for (unsigned long d3 = 0; d3 < idim.c; d3++) {
        unsigned long r0 = d0 + offset.n;
        unsigned long r1 = d1 + offset.h;
        unsigned long r2 = d2 + offset.w;
        unsigned long r3 = d3 + offset.c;
        unsigned long srcIdx = getNHWC(idim, d0, d1, d2, d3);
        unsigned long destIdx = getNHWC(odim, r0, r1, r2, r3);
        dest[destIdx] = src[srcIdx];
      }
    }
  }
}

__kernel void inserttensorW(__global void *mem, unsigned long dest, unsigned long src,
                            ShapeNHWC odim, ShapeNHWC idim, ShapeNHWC offset) {
  inserttensorK(&mem[dest], &mem[src], odim, idim, offset);
}
)";
