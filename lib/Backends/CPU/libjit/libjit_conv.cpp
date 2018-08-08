/**
 * Copyright (c) 2017-present, Facebook, Inc.
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
#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

#include "libjit_defs.h"

namespace {
// Initialize the convolution output frame for slice \p N with the bias \p
// biasW.
void libjit_conv_init_output_with_bias(uint64_t N, float *outW,
                                       const float *biasW,
                                       const uint64_t *outWdims,
                                       const uint64_t *biasWdims) {
  // For each (x,y) step in the output tensor:
  for (uint64_t ax = 0; ax < outWdims[1]; ax++) {
    for (uint64_t ay = 0; ay < outWdims[2]; ay++) {
      // For each output channel:
      for (uint64_t d = 0; d < outWdims[3]; d++) {
        // Store the results to the output buffer.
        float bias = biasW[d];
        auto outIdx = libjit_getXYZW(outWdims, N, ax, ay, d);
        outW[outIdx] = bias;
      } // For each depth in the output.
    }   // For each Y in the output.
  }     // For each X in the output.
}

/// Perform the heart of the convolution. Load \p ywidth scalars in a specific
/// channel, broadcast them, and multiply them with
/// [ywidth * float8 * numDepthRegs] depth values and accumulate them to create
/// [ywidth * float8 * numDepthRegs] depth result values.
void libjit_convDKKC8_convolve_channel(
    float *outW, const float *inW, const float *filterW,
    const uint64_t *outWdims, const uint64_t *inWdims,
    const uint64_t *filterWdims, uint64_t sampleN, uint64_t outChannel,
    unsigned numDepthRegs, unsigned ywidth, uint64_t numChannels, int64_t inX,
    int64_t inY, uint64_t outX, uint64_t outY, uint64_t filterX,
    uint64_t filterY, uint64_t stride, uint64_t group) {

  // Process N * YWidth * 8 output pixels at once. Each value here is a
  // scalar that represents the sum for (x,y..y+ywidth) and the filter. The
  // SIMD dimension represents multiple layers of the depth
  // (output channel).
  float8 sum[numDepthRegs][ywidth];
  for (unsigned wu = 0; wu < ywidth; wu++) {
    for (unsigned du = 0; du < numDepthRegs; du++) {
      sum[du][wu] = 0;
    }
  }

  // Perform the heart of the convolution.

  // For each input channel:
  for (uint64_t fd = 0; fd < numChannels; fd++) {
    // First, load and broadcast the scalar data from the input buffer.
    float8 in8[ywidth];
    for (unsigned wu = 0; wu < ywidth; wu++) {
      // Load a single pixel from the input image and broadcast it.
      auto inIdx = libjit_getXYZW(inWdims, sampleN, inX, inY + wu * stride,
                                  fd + group * numChannels);
      in8[wu] = inW[inIdx];
    }

    // For each y pixel:
    for (unsigned wu = 0; wu < ywidth; wu++) {
      // Load N x 8 elements from the filter layer. The filter is
      // pre-swizzled to ensure efficient access.
      for (unsigned du = 0; du < numDepthRegs; du++) {
        auto filterIdx = libjit_getXYZWQ(filterWdims, outChannel / 8 + du,
                                         filterX, filterY, fd, 0);
        float8 ff0 = LoadFloat8(&filterW[filterIdx]);
        sum[du][wu] += ff0 * in8[wu];
      }
    }
  }

  // Store the results to the output buffer.
  for (unsigned wu = 0; wu < ywidth; wu++) {
    for (unsigned du = 0; du < numDepthRegs; du++) {
      // Add the partial sum to the tile.
      auto outIdx = libjit_getXYZW(outWdims, sampleN, outX, outY + wu,
                                   outChannel + du * 8);
      AddFloat8(&outW[outIdx], sum[du][wu]);
    }
  }
}

/// Process the input buffer in the convolution by iterating on the filter and
/// then on the pixels. This means that we process the whole input image for
/// each pixel in the filter. We try to unroll and process multiple inputs on
/// the Y row together.
void libjit_convDKKC8_foreach_xy_filter_pixels(
    uint64_t sampleN, uint64_t outChannel, unsigned numDepthRegs,
    unsigned depthStrips, unsigned sizeGroupY, uint64_t numChannels,
    float *outW, const float *inW, const float *filterW, const float *biasW,
    const uint64_t *outWdims, const uint64_t *inWdims,
    const uint64_t *filterWdims, const uint64_t *biasWdims,
    uint64_t *filterSizes, uint64_t *strides, uint64_t *pads, uint64_t group,
    uint64_t endChannelIndex) {
  // The loops below look scary but the the idea is simple. We iterate over
  // the pixels in the output tensor and calculate the coordinate of the source
  // tensor. When we process the Y row we try to process [sizeGroupY] elements
  // at once. After we finish the row we handle the odd cases by handling one y
  // value at a time.

  uint64_t pad_t = pads[0];
  uint64_t pad_l = pads[1];
  uint64_t stride_h = strides[0];
  uint64_t stride_w = strides[1];
  uint64_t kernel_h = filterSizes[0];
  uint64_t kernel_w = filterSizes[1];
  // For each element in the convolution-filter:
  for (uint64_t fx = 0; fx < kernel_h; fx++) {
    for (uint64_t fy = 0; fy < kernel_w; fy++) {

      // For each x step in the input/output tensor:
      for (uint64_t outx = 0; outx < outWdims[1]; outx++) {
        int64_t inx = (int64_t)outx * stride_h - pad_t + fx;

        // Ignore out-of-bounds X values.
        if (inx < 0 || inx >= (int64_t)inWdims[1]) {
          continue;
        }

        // For each y step in the input/output tensor, in steps of \p
        // sizeGroupY. We process \p sizeGroupY pixels of Y in one iteration.
        uint64_t outy = 0;
        while (outy < outWdims[2]) {
          int64_t iny = (int64_t)outy * stride_w - pad_l + fy;

          if ((iny + (int64_t)stride_w * sizeGroupY) >= (int64_t)inWdims[2]) {
            // If we've passed the upper bound, we don't want to increment
            // `outy` again, since we're going to handle the remaining y steps
            // in the following loop.
            break;
          }
          // Ignore out of bound indices.
          if (iny < 0) {
            /// We know iny is out of bounds, so we have nothing to do for outy.
            /// But we can't skip ahead by sizeGroupY, because we haven't
            /// checked outy + 1.
            outy += 1;
            continue;
          }

          // Convolve the (x,y .. y + ywidth) values.
          uint64_t outC = outChannel;
          for (unsigned strip = 0;
               strip < depthStrips && outC < endChannelIndex; strip++) {
            libjit_convDKKC8_convolve_channel(
                outW, inW, filterW, outWdims, inWdims, filterWdims, sampleN,
                outC, numDepthRegs, sizeGroupY, numChannels, inx, iny, outx,
                outy, fx, fy, stride_w, group);
            outC += numDepthRegs * 8;
          }

          outy += sizeGroupY;
        } // For each Y group in the output.

        // Handle the remaining Y in the row in groups of size 1.
        for (; outy < outWdims[2]; outy++) {
          int64_t iny = (int64_t)outy * stride_w - pad_l + fy;
          // Ignore out of bound indices.
          if (iny < 0 || iny >= (int64_t)inWdims[2]) {
            continue;
          }

          // Convolve a single (x,y) value.
          uint64_t outC = outChannel;
          for (unsigned strip = 0;
               strip < depthStrips && outC < endChannelIndex; strip++) {
            libjit_convDKKC8_convolve_channel(
                outW, inW, filterW, outWdims, inWdims, filterWdims, sampleN,
                outC, numDepthRegs, 1, numChannels, inx, iny, outx, outy, fx,
                fy, stride_w, group);
            outC += numDepthRegs * 8;
          }
        } // For each Y, in step of 1, in the output.

      } // For each X in the output.
    }   // For each Y in the filter.
  }     // For each X in the filter.
}

// Process the input buffer in the convolution by iterating on the input buffer
// and then on the filter. This means that we process the whole input filter for
// each pixel in the input buffer.
void libjit_convDKKC8_foreach_xy_pixels_filter(
    uint64_t sampleN, uint64_t outChannel, unsigned numDepthRegs,
    unsigned depthStrips, unsigned sizeGroupY, uint64_t numChannels,
    float *outW, const float *inW, const float *filterW, const float *biasW,
    const uint64_t *outWdims, const uint64_t *inWdims,
    const uint64_t *filterWdims, const uint64_t *biasWdims,
    uint64_t *filterSizes, uint64_t *strides, uint64_t *pads, uint64_t group,
    uint64_t endChannelIndex) {

  uint64_t pad_t = pads[0];
  uint64_t pad_l = pads[1];
  uint64_t stride_h = strides[0];
  uint64_t stride_w = strides[1];
  uint64_t kernel_h = filterSizes[0];
  uint64_t kernel_w = filterSizes[1];
  // For each (x,y) step in the input/output tensor:
  for (uint64_t outx = 0; outx < outWdims[1]; outx++) {
    for (uint64_t outy = 0; outy < outWdims[2]; outy++) {

      // For each element in the convolution-filter:
      for (uint64_t fx = 0; fx < kernel_h; fx++) {
        for (uint64_t fy = 0; fy < kernel_w; fy++) {

          // Calculate the specific input x,y that we process in this
          // iteration.
          int64_t inx = (int64_t)outx * stride_h - pad_t + fx;
          int64_t iny = (int64_t)outy * stride_w - pad_l + fy;

          // Ignore index access below zero (this is due to padding).
          if (inx < 0 || iny < 0 || inx >= (int64_t)inWdims[1] ||
              iny >= (int64_t)inWdims[2]) {
            continue;
          }

          uint64_t outC = outChannel;
          for (unsigned strip = 0;
               strip < depthStrips && outC < endChannelIndex; strip++) {
            libjit_convDKKC8_convolve_channel(
                outW, inW, filterW, outWdims, inWdims, filterWdims, sampleN,
                outC, numDepthRegs, 1, numChannels, inx, iny, outx, outy, fx,
                fy, stride_w, group);
            outC += numDepthRegs * 8;
          }
        } // For each Y in the filter.
      }   // For each X in the filter.
    }     // For each Y in the output.
  }       // For each X in the output.
}

} // namespace

extern "C" {
void libjit_convDKKC8_f(float *outW, const float *inW, const float *filterW,
                        const float *biasW, const uint64_t *outWdims,
                        const uint64_t *inWdims, const uint64_t *filterWdims,
                        const uint64_t *biasWdims, uint64_t *filterSizes,
                        uint64_t *strides, uint64_t *pads, uint64_t group,
                        unsigned pixelScanFirst, unsigned numDepthRegs,
                        unsigned sizeGroupY, unsigned depthStrips) {
  uint64_t inChannels = inWdims[3];
  uint64_t outChannels = outWdims[3];
  uint64_t inCperG = inChannels / group;
  uint64_t outCperG = outChannels / group;

  // Select the order in which we iterate over the pixels in the picture.
  auto eachPixelConv =
      (pixelScanFirst ? &libjit_convDKKC8_foreach_xy_pixels_filter
                      : &libjit_convDKKC8_foreach_xy_filter_pixels);

  // For each input in the batch:
  for (uint64_t n = 0; n < inWdims[0]; n++) {

    // Initialize the output frame for the N'th slice with the bias.
    // Later we will accumulate values into this slice.
    libjit_conv_init_output_with_bias(n, outW, biasW, outWdims, biasWdims);

    // For each group of input channels:
    for (uint64_t g = 0; g < group; g++) {

      // For each output channel, process [numDepthRegs x float8] elements.
      uint64_t startChannelIndex = g * outCperG;
      uint64_t endChannelIndex = (g + 1) * outCperG;
      for (uint64_t d = startChannelIndex; d < endChannelIndex;
           d += 8 * numDepthRegs * depthStrips) {

        // Perform the convolution for each pixel.
        eachPixelConv(n, d, numDepthRegs, depthStrips, sizeGroupY, inCperG,
                      outW, inW, filterW, biasW, outWdims, inWdims, filterWdims,
                      biasWdims, filterSizes, strides, pads, g,
                      endChannelIndex);

      } // For each D (the depth, or the output channel).
    }   // for each G, the group
  }     // For each N, the sample in the batch.
}

void libjit_convolution_f(float *outW, const float *inW, const float *filterW,
                          const float *biasW, const uint64_t *outWdims,
                          const uint64_t *inWdims, const uint64_t *filterWdims,
                          const uint64_t *biasWdims, uint64_t *filterSizes,
                          uint64_t *strides, uint64_t *pads, uint64_t group,
                          unsigned depthUnroll) {
  uint64_t inChannels = inWdims[3];
  uint64_t outChannels = outWdims[3];
  uint64_t inCperG = inChannels / group;
  uint64_t outCperG = outChannels / group;

  // The output dims are calculated already from all of the pads,
  // therefore we only need the top and left pads here to control the starting
  // position.
  uint64_t pad_t = pads[0];
  uint64_t pad_l = pads[1];
  uint64_t stride_h = strides[0];
  uint64_t stride_w = strides[1];
  uint64_t kernel_h = filterSizes[0];
  uint64_t kernel_w = filterSizes[1];
  // The size of the input-channel tile. High channel count allow for SIMD
  // parallelism but create register pressure. Low channel count reduces the
  // memory pressure and allows things to fit in cache, but require additional
  // compute (horizontal add) to sum the values in the block. This value is a
  // compromise between the two.
  constexpr unsigned cbSize = 512;

  // For each input in the batch:
  for (uint64_t n = 0; n < inWdims[0]; n++) {

    // Initialize the output frame for the N'th slice with the bias.
    // Later we will accumulate values into this slice.
    libjit_conv_init_output_with_bias(n, outW, biasW, outWdims, biasWdims);

    // For each group of input channels:
    for (uint64_t g = 0; g < group; g++) {
      // Process the body of the loop in tiles of "channel-block".
      for (uint64_t cb = 0; cb < inCperG; cb += cbSize) {

        // For each output channel in the group. Process 'depthUnroll' output
        // layers together.
        for (uint64_t d = g * outCperG; d < (g + 1) * outCperG;
             d += depthUnroll) {

          // For each element in the convolution-filter:
          for (uint64_t fx = 0; fx < kernel_h; fx++) {
            for (uint64_t fy = 0; fy < kernel_w; fy++) {

              // For each convolution 'jump' in the input tensor:
              for (uint64_t outx = 0; outx < outWdims[1]; outx++) {
                for (uint64_t outy = 0; outy < outWdims[2]; outy++) {

                  // Process 'depthUnroll' output pixels at once. Each scalar
                  // here represents the convolution sum for one (x,y) point in
                  // the output. We process the same pixel for different output
                  // channel (D) values. The compiler should perform scalar
                  // replacement of aggregates and split this tiny array to
                  // registers.
                  float sum[depthUnroll];
                  for (unsigned i = 0; i < depthUnroll; i++) {
                    sum[i] = 0;
                  }

                  // Calculate the specific input x,y that we process in this
                  // iteration.
                  int64_t inx = (int64_t)outx * stride_h - pad_t + fx;
                  int64_t iny = (int64_t)outy * stride_w - pad_l + fy;

                  // Ignore index access below zero (this is due to padding).
                  if (inx < 0 || iny < 0 || inx >= (int64_t)inWdims[1] ||
                      iny >= (int64_t)inWdims[2]) {
                    continue;
                  }

                  // Calculate the indices into the Filter and Input buffers.
                  uint64_t inIdx = libjit_getXYZW(inWdims, n, (uint64_t)inx,
                                                  (uint64_t)iny, g * inCperG);
                  uint64_t filterIdx =
                      libjit_getXYZW(filterWdims, d, fx, fy, 0);
                  uint64_t sliceSize =
                      filterWdims[1] * filterWdims[2] * filterWdims[3];

                  // Perform the heart of the convolution, 4 elements at a time
                  // to reduce register pressure.
                  for (uint64_t fd = cb, e = MIN(cb + cbSize, inCperG); fd < e;
                       fd++) {
                    float in = inW[inIdx + fd];
                    for (unsigned i = 0; i < MIN(4, depthUnroll); i++) {
                      sum[i] += filterW[filterIdx + (sliceSize * i) + fd] * in;
                    }
                  }

                  // And run the innermost loop again for the second group of
                  // depth slices:
                  if (depthUnroll > 4) {
                    for (uint64_t fd = cb, e = MIN(cb + cbSize, inCperG);
                         fd < e; fd++) {
                      float in = inW[inIdx + fd];
                      for (unsigned i = 4; i < MIN(8, depthUnroll); i++) {
                        sum[i] +=
                            filterW[filterIdx + (sliceSize * i) + fd] * in;
                      }
                    }
                  }

                  // Store the results to the output buffer.
                  for (unsigned i = 0; i < depthUnroll; i++) {
                    outW[libjit_getXYZW(outWdims, n, outx, outy, d + i)] +=
                        sum[i];
                  }
                }
              }
            } // For each Y in the filter.
          }   // For each X in the filter.
        }     // For each D (the depth, or the output channel).
      }       // For each block in the input channel.
    }         // For each group in the input channel.
  }           // For each N, the sample in the batch.
}

void libjit_convolution_i8(
    int8_t *outW, const int8_t *inW, const int8_t *filterW, const int8_t *biasW,
    const uint64_t *outWdims, const uint64_t *inWdims,
    const uint64_t *filterWdims, const uint64_t *biasWdims,
    uint64_t *filterSizes, uint64_t *strides, uint64_t *pads, uint64_t group,
    int32_t outOffset, int32_t inOffset, int32_t filterOffset,
    int32_t biasOffset, int32_t biasPre, int32_t biasPost, int32_t biasScale,
    int32_t outPre, int32_t outPost, int32_t outScale, unsigned depthUnroll) {
  uint64_t inChannels = inWdims[3];
  uint64_t outChannels = outWdims[3];
  uint64_t inCperG = inChannels / group;
  uint64_t outCperG = outChannels / group;
  uint64_t pad_t = pads[0];
  uint64_t pad_l = pads[1];
  uint64_t stride_h = strides[0];
  uint64_t stride_w = strides[1];
  uint64_t kernel_h = filterSizes[0];
  uint64_t kernel_w = filterSizes[1];
  // For each input in the batch:
  for (uint64_t n = 0; n < inWdims[0]; n++) {
    // For each group of input channels:
    for (uint64_t g = 0; g < group; g++) {

      // For each output channel in the group. Process 'depthUnroll' output
      // layers together.
      for (uint64_t d = g * outCperG; d < (g + 1) * outCperG;
           d += depthUnroll) {
        // For each convolution 'jump' in the input tensor:
        int64_t x = -(int64_t)pad_t;
        for (uint64_t ax = 0; ax < outWdims[1]; x += stride_h, ax++) {
          int64_t y = -(int64_t)pad_l;
          for (uint64_t ay = 0; ay < outWdims[2]; y += stride_w, ay++) {
            int32_t sum[depthUnroll];

            for (unsigned i = 0; i < depthUnroll; i++) {
              // Scale the bias to match the scale of the matrix multiplication.
              sum[i] = libjit_scale_i32i8((int32_t)biasW[d + i] - biasOffset,
                                          biasPre, biasPost, biasScale, 0);
            }

            // For each element in the convolution-filter:
            for (uint64_t fx = 0; fx < kernel_h; fx++) {
              for (uint64_t fy = 0; fy < kernel_w; fy++) {
                int64_t ox = x + fx;
                int64_t oy = y + fy;

                // Ignore index access below zero (this is due to padding).
                if (ox < 0 || oy < 0 || ox >= (int64_t)inWdims[1] ||
                    oy >= (int64_t)inWdims[2]) {
                  continue;
                }

                // Calculate the indices into the Filter and Input buffers.
                uint64_t inIdx = libjit_getXYZW(inWdims, n, (uint64_t)ox,
                                                (uint64_t)oy, g * inCperG);
                uint64_t filterIdx = libjit_getXYZW(filterWdims, d, fx, fy, 0);
                uint64_t sliceSize =
                    filterWdims[1] * filterWdims[2] * filterWdims[3];

                // Perform the innermost loop of the convolution using 4 vector
                // registers.
                for (uint64_t fd = 0; fd < inCperG; fd++) {
                  int32_t in = inW[inIdx + fd] - inOffset;
                  for (unsigned i = 0; i < MIN(4, depthUnroll); i++) {
                    sum[i] += (filterW[filterIdx + (sliceSize * i) + fd] -
                               filterOffset) *
                              in;
                  }
                }

                // And perform the innermost loop again with 4 more registers.
                if (depthUnroll > 4)
                  for (uint64_t fd = 0; fd < inCperG; fd++) {
                    int32_t in = inW[inIdx + fd] - inOffset;
                    for (unsigned i = 4; i < MIN(8, depthUnroll); i++) {
                      sum[i] += (filterW[filterIdx + (sliceSize * i) + fd] -
                                 filterOffset) *
                                in;
                    }
                  }
              }
            }

            for (unsigned i = 0; i < depthUnroll; i++) {
              // Scale the result back to the expected destination scale.
              int32_t scaledSum = libjit_scale_i32i8(sum[i], outPre, outPost,
                                                     outScale, outOffset);
              outW[libjit_getXYZW(outWdims, n, ax, ay, d + i)] =
                  libjit_clip(scaledSum);
            }
          } // W
        }   // H
      }     // C
    }       // G
  }         // N
}

void libjit_convolution_grad_f(float *inG, const float *outG, const float *inW,
                               float *filterG, float *biasG,
                               const float *filterW, const uint64_t *outGdims,
                               const uint64_t *inWdims,
                               const uint64_t *filterGdims,
                               const uint64_t *kernels, const uint64_t *strides,
                               const uint64_t *pads, const uint64_t group) {
  // NHWC format is assumed
  // Clear inG, filterG, and biasG
  uint64_t p = sizeof(float);
  memset(inG, 0, inWdims[0] * inWdims[1] * inWdims[2] * inWdims[3] * p);
  memset(filterG, 0,
         filterGdims[0] * filterGdims[1] * filterGdims[2] * filterGdims[3] * p);
  memset(biasG, 0, outGdims[3] * p);

  uint64_t pad_t = pads[0];
  uint64_t pad_l = pads[1];
  uint64_t stride_h = strides[0];
  uint64_t stride_w = strides[1];
  uint64_t kernel_h = kernels[0];
  uint64_t kernel_w = kernels[1];
  uint64_t inCperG = inWdims[3] / group;
  uint64_t outCperG = outGdims[3] / group;

  // For each input in the batch:
  for (uint64_t n = 0; n < outGdims[0]; n++) {
    // For each group of input channels:
    for (uint64_t g = 0; g < group; g++) {
      for (uint64_t d = g * outCperG; d < (g + 1) * outCperG; d++) {
        int64_t x = -(int64_t)pad_t;
        for (uint64_t bx = 0; bx < outGdims[1]; bx++, x += stride_h) {
          int64_t y = -(int64_t)pad_l;
          for (uint64_t by = 0; by < outGdims[2]; by++, y += stride_w) {
            float grad = outG[libjit_getXYZW(outGdims, n, bx, by, d)];

            for (uint64_t kx = 0; kx < kernel_h; kx++) {
              for (uint64_t ky = 0; ky < kernel_w; ky++) {
                int64_t ax = x + kx;
                int64_t ay = y + ky;

                if (ax < 0 || ay < 0 || ax >= (int64_t)inWdims[1] ||
                    ay >= (int64_t)inWdims[2]) {
                  continue;
                }

                for (uint64_t c = 0; c < inCperG; c++) {
                  inG[libjit_getXYZW(inWdims, n, (uint64_t)ax, (uint64_t)ay,
                                     g * inCperG + c)] +=
                      filterW[libjit_getXYZW(filterGdims, d, kx, ky, c)] * grad;
                  filterG[libjit_getXYZW(filterGdims, d, kx, ky, c)] +=
                      inW[libjit_getXYZW(inWdims, n, (uint64_t)ax, (uint64_t)ay,
                                         g * inCperG + c)] *
                      grad;
                }
              }
            }

            biasG[d] += grad;
          } // W
        }   // H
      }     // C
    }       // G
  }         // N
}
}
