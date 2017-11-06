

static const char* SHADER_CODE = R"(

typedef struct {
  size_t n; // Number of samples
  size_t h; // Height
  size_t w; // Width
  size_t c; // Number of channels
} ShapeNHWC;

/// \returns the index of the element at x,y,z,w.

size_t getNHWC(ShapeNHWC s, size_t x, size_t y, size_t z, size_t w) {
  return (x * s.c * s.w * s.h) + (y * s.c * s.w) + (z * s.c) + w;
}

__kernel void reluK(__global float *dest, __global float *src) {
  size_t i = get_global_id(0);
  dest[i] = fmax(src[i], 0);
}

__kernel void sigmoidK(__global float *dest, __global float *src) {
  size_t i = get_global_id(0);
  dest[i] = 1 / (1 + exp(-src[i]));
}

__kernel void tanhK(__global float *dest, __global float *src) {
  size_t i = get_global_id(0);
  float val = src[i];
  float exp_val = exp(val);
  float exp_neg_val = exp(-val);
  dest[i] = (exp_val - exp_neg_val) / (exp_val + exp_neg_val);
};

__kernel void elementaddK(__global float *dest, __global float *LHS,
                          __global float *RHS) {
  size_t i = get_global_id(0);
  dest[i] = LHS[i] + RHS[i];
}

__kernel void elementmulK(__global float *dest, __global float *LHS,
                          __global float *RHS) {
  size_t i = get_global_id(0);
  dest[i] = LHS[i] * RHS[i];
}

__kernel void fullyconnectedK(__global float *dest, __global float *src,
                              __global float *filter, __global float *bias,
                              unsigned sliceSize) {
  size_t depth = get_global_id(0);
  size_t N = get_global_id(1);
  size_t inBase = N * sliceSize;
  float sum = 0;
  for (size_t j = 0; j < sliceSize; j++) {
    sum += src[inBase + j] * filter[depth * sliceSize + j];
  }
  sum += bias[depth];
  dest[N * sliceSize + depth] = sum;
}

__kernel void regressionK(__global float *dest, __global float *src,
                          __global float *exp) {
  size_t i = get_global_id(0);
  dest[i] = src[i];
}

__kernel void softmaxK(__global float *dest, __global float *src,
                       __global float *e_cache, __global unsigned *selected,
                       unsigned sliceSize) {
  size_t i = get_global_id(0);
  float max_ = src[i * sliceSize];
  for (size_t j = 0; j < sliceSize; j++) {
    max_ = max(max_, src[i * sliceSize + j]);
  }
  float sum = 0;
  for (size_t j = 0; j < sliceSize; j++) {
    float e = exp(src[i * sliceSize + j] - max_);
    sum += e;
    e_cache[i * sliceSize + j] = e;
  }
  for (size_t j = 0; j < sliceSize; j++) {
    e_cache[i * sliceSize + j] /= sum;
    dest[i * sliceSize + j] = e_cache[i * sliceSize + j];
  }
}

__kernel void convolutionK(__global float *dest, __global float *src,
                           __global float *filter, __global float *bias,
                           size_t filterSize, size_t pad, size_t stride,
                           ShapeNHWC odim, ShapeNHWC idim,
                           ShapeNHWC filterDim) {
  size_t ax = get_global_id(0);
  size_t ay = get_global_id(1);

  // For each input in the batch:
  for (size_t n = 0; n < idim.n; n++) {

    // For each layer in the output tensor:
    for (size_t d = 0; d < odim.c; d++) {

      typedef int ssize_t;
      // For each convolution 'jump' in the input tensor:
      ssize_t x = -ssize_t(pad) + ax * stride;
      ssize_t y = -ssize_t(pad) + ay * stride;

      // For each element in the convolution-filter:
      float sum = 0;
      for (size_t fx = 0; fx < filterSize; fx++) {
        for (size_t fy = 0; fy < filterSize; fy++) {
          ssize_t ox = x + fx;
          ssize_t oy = y + fy;

          // Ignore index access below zero (this is due to padding).
          if (ox < 0 || oy < 0 || ox >= ssize_t(odim.h) ||
              oy >= ssize_t(odim.w)) {
            continue;
          }

          for (size_t fd = 0; fd < idim.c; fd++) {
            sum += filter[getNHWC(filterDim, d, fx, fy, fd)] *
                   src[getNHWC(idim, n, (size_t)ox, (size_t)oy, fd)];
          }
        }
      }

      sum += bias[d];
      dest[getNHWC(odim, n, ax, ay, d)] = sum;
    } // D
  }   // N
}

)";
