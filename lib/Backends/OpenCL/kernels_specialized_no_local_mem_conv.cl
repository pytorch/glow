/**
 * Copyright (c) Glow Contributors. See CONTRIBUTORS file.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#define X_MIN -(sdim_t)CONVK_PADS_TOP
#define Y_MIN -(sdim_t)CONVK_PADS_LEFT

#define X_MAX (X_MIN + (CONVK_ODIM_H - 1) * CONVK_STRIDES_H)
#define Y_MAX (Y_MIN + (CONVK_ODIM_W - 1) * CONVK_STRIDES_W)

#define OX_MAX (X_MAX + (CONVK_KERNEL_H - 1) * CONVK_DILATION)
#define OY_MAX (Y_MAX + (CONVK_KERNEL_W - 1) * CONVK_DILATION)

/**
 * Convolution which does not use local memory and that is aggressively
 * per-node specialized using compile time constants passed as macros.
 */
/// \returns the index of the element at w, h, c, n
inline dim_t getNHWC(dim_t sw, dim_t sh, dim_t sc, unsigned sn, unsigned h,
                     unsigned w, unsigned c) {
  return (sn * sc * sw * sh) + (h * sc * sw) + (w * sc) + c;
}

kernel void convolutionK(global float *restrict dest,
                         const global float *restrict src,
                         const global float *restrict filter,
                         const global float *restrict bias,
                         const int fuseReLU) {
  dim_t ax = get_global_id(0);
  dim_t ay = get_global_id(1);
  dim_t d = get_global_id(2);
  const dim_t inCperG = CONVK_IDIM_C / CONVK_GROUP;
  const dim_t outCperG = CONVK_ODIM_C / CONVK_GROUP;
  const dim_t inChannelOffset = d / outCperG * inCperG;

  // For each convolution 'jump' in the input tensor:
  sdim_t x = -(sdim_t)CONVK_PADS_TOP + ax * CONVK_STRIDES_H;
  sdim_t y = -(sdim_t)CONVK_PADS_LEFT + ay * CONVK_STRIDES_W;

  // For each input in the batch:
  for (dim_t n = 0; n < CONVK_BATCHES; n++) {

    // For each element in the convolution-filter:
    float sum = 0;
    for (dim_t fx = 0; fx < CONVK_KERNEL_H; fx++) {
      for (dim_t fy = 0; fy < CONVK_KERNEL_W; fy++) {
        sdim_t ox = x + fx * CONVK_DILATION;
        sdim_t oy = y + fy * CONVK_DILATION;

        // Ignore index access below zero (this is due to padding).
        // Use compile time evaluated predicates to expose branches
        // that will never be taken and thus can be removed at compile
        // time.
        if ((X_MIN < 0 && ox < 0) || (Y_MIN < 0 && oy < 0) ||
            (OX_MAX >= (sdim_t)CONVK_IDIM_H && ox >= (sdim_t)CONVK_IDIM_H) ||
            (OY_MAX >= (sdim_t)CONVK_IDIM_W && oy >= (sdim_t)CONVK_IDIM_W)) {
          continue;
        }
        for (dim_t fd = 0; fd < inCperG; fd++) {
          sum += filter[getNHWC(CONVK_FILTER_W, CONVK_FILTER_H, CONVK_FILTER_C,
                                d, fx, fy, fd)] *
                 src[getNHWC(CONVK_IDIM_W, CONVK_IDIM_H, CONVK_IDIM_C, n,
                             (unsigned)ox, (unsigned)oy, fd + inChannelOffset)];
        }
      }
    }

    sum += bias[d];
    if (fuseReLU) {
      sum = max(sum, 0.0f);
    }
    dest[getNHWC(CONVK_ODIM_W, CONVK_ODIM_H, CONVK_ODIM_C, n, ax, ay, d)] = sum;
  } // N
}

kernel void convolutionW(global void *mem, unsigned dest, unsigned src,
                         unsigned filter, unsigned bias) {
  convolutionK(&mem[dest], &mem[src], &mem[filter], &mem[bias], 0);
}

kernel void convolutionW_ReLU(global void *mem, unsigned dest, unsigned src,
                              unsigned filter, unsigned bias) {
  convolutionK(&mem[dest], &mem[src], &mem[filter], &mem[bias], 1);
}
