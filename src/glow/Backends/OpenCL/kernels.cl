
static const char* SHADER_CODE = R"(
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
)";

