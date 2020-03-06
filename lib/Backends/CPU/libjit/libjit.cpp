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
#include <algorithm>
#include <assert.h>
#include <chrono>
#include <cmath>
#include <math.h>
#include <numeric>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

#include "libjit_defs.h"

namespace {

template <class ElemTy>
static void libjit_dump_tensor_impl(ElemTy *tensor, dim_t *dims,
                                    dim_t numDims) {
  // Check for 0-dimensional tensor.
  if (!numDims) {
    printf("[ Scalar containing: %.3f ]\n", (float)tensor[0]);
    return;
  }

  // Output shape.
  printf("shape: ( ");
  for (size_t i = 0; i < numDims; ++i) {
    printf("%zu ", (size_t)dims[i]);
  }
  printf(")\n");

  ElemTy mx = tensor[0];
  ElemTy mn = tensor[0];

  size_t size = 1;
  size_t sliceSize[numDims];
  for (size_t i = 0; i < numDims; ++i) {
    size *= dims[i];
  }

  for (ssize_t i = numDims - 1, curSliceSize = 1; i >= 0; --i) {
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
static dim_t get_element_ptr(const ElemTy *tensor, const dim_t *dims,
                             dim_t numDims, const dim_t *indices,
                             dim_t numIndices) {
  dim_t index = 0;
  dim_t subdimensionSize = 1;
  for (dim_t i = numDims; i > 0; i--) {
    dim_t curIndicesValue = (i <= numIndices) ? indices[i - 1] : 0;
    index += subdimensionSize * curIndicesValue;
    subdimensionSize *= dims[i - 1];
  }
  return index;
}

template <typename ElemTy>
static void libjit_insert_tensor(ElemTy *tensor, ElemTy *slice, dim_t *offset,
                                 dim_t *tensorDim, dim_t *sliceDim,
                                 dim_t numDimsTensor, dim_t numDimsSlice,
                                 dim_t offsetDim, dim_t count, dim_t axis) {
  // Destination coordinates.
  dim_t C[5];

  // A local copy of the offsets buffer. We copy the buffer to make it clear
  // to the optimizer that the inputs don't alias. This loop is optimized away.
  dim_t offsets_cpy[5];
  for (dim_t i = 0; i < numDimsSlice; i++) {
    offsets_cpy[i] = offset[i];
  }

  if (numDimsSlice == 5) {
    for (dim_t c = 0; c < count; c++)
      for (dim_t x = 0; x < sliceDim[0]; x++)
        for (dim_t y = 0; y < sliceDim[1]; y++)
          for (dim_t z = 0; z < sliceDim[2]; z++)
            for (dim_t w = 0; w < sliceDim[3]; w++)
              for (dim_t q = 0; q < sliceDim[4]; q++) {
                const dim_t countAxisOffset = c * sliceDim[axis];
                C[0] = x + offsets_cpy[0] + ((axis == 0) ? countAxisOffset : 0);
                C[1] = y + offsets_cpy[1] + ((axis == 1) ? countAxisOffset : 0);
                C[2] = z + offsets_cpy[2] + ((axis == 2) ? countAxisOffset : 0);
                C[3] = w + offsets_cpy[3] + ((axis == 3) ? countAxisOffset : 0);
                C[4] = q + offsets_cpy[4] + ((axis == 4) ? countAxisOffset : 0);
                tensor[libjit_getXYZWQ(tensorDim, C[0], C[1], C[2], C[3],
                                       C[4])] =
                    slice[libjit_getXYZWQ(sliceDim, x, y, z, w, q)];
              }
    return;
  }

  if (numDimsSlice == 4) {
    for (dim_t c = 0; c < count; c++)
      for (dim_t x = 0; x < sliceDim[0]; x++)
        for (dim_t y = 0; y < sliceDim[1]; y++)
          for (dim_t z = 0; z < sliceDim[2]; z++)
            for (dim_t w = 0; w < sliceDim[3]; w++) {
              const dim_t countAxisOffset = c * sliceDim[axis];
              C[0] = x + offsets_cpy[0] + ((axis == 0) ? countAxisOffset : 0);
              C[1] = y + offsets_cpy[1] + ((axis == 1) ? countAxisOffset : 0);
              C[2] = z + offsets_cpy[2] + ((axis == 2) ? countAxisOffset : 0);
              C[3] = w + offsets_cpy[3] + ((axis == 3) ? countAxisOffset : 0);
              tensor[libjit_getXYZW(tensorDim, C[0], C[1], C[2], C[3])] =
                  slice[libjit_getXYZW(sliceDim, x, y, z, w)];
            }
    return;
  }

  if (numDimsSlice == 3) {
    for (dim_t c = 0; c < count; c++)
      for (dim_t x = 0; x < sliceDim[0]; x++)
        for (dim_t y = 0; y < sliceDim[1]; y++)
          for (dim_t z = 0; z < sliceDim[2]; z++) {
            const dim_t countAxisOffset = c * sliceDim[axis];
            C[0] = x + offsets_cpy[0] + ((axis == 0) ? countAxisOffset : 0);
            C[1] = y + offsets_cpy[1] + ((axis == 1) ? countAxisOffset : 0);
            C[2] = z + offsets_cpy[2] + ((axis == 2) ? countAxisOffset : 0);
            tensor[libjit_getXYZ(tensorDim, C[0], C[1], C[2])] =
                slice[libjit_getXYZ(sliceDim, x, y, z)];
          }
    return;
  }

  if (numDimsSlice == 2) {
    for (dim_t c = 0; c < count; c++)
      for (dim_t x = 0; x < sliceDim[0]; x++)
        for (dim_t y = 0; y < sliceDim[1]; y++) {
          const dim_t countAxisOffset = c * sliceDim[axis];
          C[0] = x + offsets_cpy[0] + ((axis == 0) ? countAxisOffset : 0);
          C[1] = y + offsets_cpy[1] + ((axis == 1) ? countAxisOffset : 0);
          tensor[libjit_getXY(tensorDim, C[0], C[1])] =
              slice[libjit_getXY(sliceDim, x, y)];
        }
    return;
  }

  if (numDimsSlice == 1) {
    for (dim_t c = 0; c < count; c++)
      for (dim_t x = 0; x < sliceDim[0]; x++) {
        const dim_t countAxisOffset = c * sliceDim[axis];
        tensor[x + offsets_cpy[0] + ((axis == 0) ? countAxisOffset : 0)] =
            slice[x];
      }
    return;
  }
}

template <typename ElemTy>
static void libjit_extract_tensor(ElemTy *tensor, ElemTy *slice, dim_t *offset,
                                  dim_t *tensorDim, dim_t *sliceDim,
                                  dim_t numDimsTensor, dim_t numDimsSlice,
                                  dim_t offsetDim) {
  // Source coordinates.
  dim_t C[5];

  // A local copy of the offsets buffer. We copy the buffer to make it clear
  // to the optimizer that the inputs don't alias. This loop is optimized away.
  dim_t offsets_cpy[5];
  for (dim_t i = 0; i < numDimsSlice; i++) {
    offsets_cpy[i] = offset[i];
  }

  if (numDimsSlice == 5) {
    for (dim_t x = 0; x < sliceDim[0]; x++)
      for (dim_t y = 0; y < sliceDim[1]; y++)
        for (dim_t z = 0; z < sliceDim[2]; z++)
          for (dim_t w = 0; w < sliceDim[3]; w++)
            for (dim_t q = 0; q < sliceDim[4]; q++) {
              C[0] = x + offsets_cpy[0];
              C[1] = y + offsets_cpy[1];
              C[2] = z + offsets_cpy[2];
              C[3] = w + offsets_cpy[3];
              C[4] = q + offsets_cpy[4];
              slice[libjit_getXYZWQ(sliceDim, x, y, z, w, q)] =
                  tensor[libjit_getXYZWQ(tensorDim, C[0], C[1], C[2], C[3],
                                         C[4])];
            }
    return;
  }

  if (numDimsSlice == 4) {
    for (dim_t x = 0; x < sliceDim[0]; x++)
      for (dim_t y = 0; y < sliceDim[1]; y++)
        for (dim_t z = 0; z < sliceDim[2]; z++)
          for (dim_t w = 0; w < sliceDim[3]; w++) {
            C[0] = x + offsets_cpy[0];
            C[1] = y + offsets_cpy[1];
            C[2] = z + offsets_cpy[2];
            C[3] = w + offsets_cpy[3];
            slice[libjit_getXYZW(sliceDim, x, y, z, w)] =
                tensor[libjit_getXYZW(tensorDim, C[0], C[1], C[2], C[3])];
          }
    return;
  }

  if (numDimsSlice == 3) {
    for (dim_t x = 0; x < sliceDim[0]; x++)
      for (dim_t y = 0; y < sliceDim[1]; y++)
        for (dim_t z = 0; z < sliceDim[2]; z++) {
          C[0] = x + offsets_cpy[0];
          C[1] = y + offsets_cpy[1];
          C[2] = z + offsets_cpy[2];
          slice[libjit_getXYZ(sliceDim, x, y, z)] =
              tensor[libjit_getXYZ(tensorDim, C[0], C[1], C[2])];
        }
    return;
  }

  if (numDimsSlice == 2) {
    for (dim_t x = 0; x < sliceDim[0]; x++)
      for (dim_t y = 0; y < sliceDim[1]; y++) {
        C[0] = x + offsets_cpy[0];
        C[1] = y + offsets_cpy[1];
        slice[libjit_getXY(sliceDim, x, y)] =
            tensor[libjit_getXY(tensorDim, C[0], C[1])];
      }
    return;
  }

  if (numDimsSlice == 1) {
    for (dim_t x = 0; x < sliceDim[0]; x++) {
      slice[x] = tensor[x + offsets_cpy[0]];
    }
    return;
  }
}

/// Helper struct for TopK
template <typename T, typename TI> struct value_index {
  TI index;
  T value;
};

/// Helper function for TopK
template <typename T, typename TI>
static int value_index_sort(const void *va, const void *vb) {
  value_index<T, TI> *a = (value_index<T, TI> *)va;
  value_index<T, TI> *b = (value_index<T, TI> *)vb;
  if (a->value != b->value)
    return a->value > b->value ? -1 : 1;
  return a->index < b->index ? -1 : 1;
}

/// Generic Top-K function. Here, \p scratch is some allocated buffer space, \p
/// size is the size of the input, and \p n is the size of the last dimension of
/// the input.
template <typename T, typename TI>
static void libjit_topk(T *values, TI *indices, const T *input, TI *scratch,
                        dim_t k, dim_t n, dim_t size) {
  dim_t in = 0;
  dim_t out = 0;

  value_index<T, TI> *buffer = (value_index<T, TI> *)scratch;

  // Specialize TopK for the case where K is 1.
  if (k == 1) {
    while (in < size) {
      // Find the largest value by iterating over the array instead of calling
      // 'sort'.
      value_index<T, TI> mx = {0, input[in]};
      for (TI i = 1; i < n; i++) {
        if (input[i + in] > mx.value) {
          mx = {i, input[i + in]};
        }
      }
      indices[out] = mx.index;
      values[out] = mx.value;
      out++;
      in += n;
    }
    return;
  }

  while (in < size) {
    for (dim_t i = 0; i < n; i++) {
      buffer[i].index = i;
      buffer[i].value = input[in++];
    }
    qsort(buffer, n, sizeof(value_index<T, TI>), value_index_sort<T, TI>);
    for (dim_t i = 0; i < k; i++) {
      indices[out] = buffer[i].index;
      values[out] = buffer[i].value;
      out++;
    }
  }
}

template <typename T, typename IDX>
static void libjit_gather(T *dest, const T *data, const IDX *indices,
                          dim_t numIndices, dim_t sliceSize, dim_t numSamples,
                          dim_t sampleSize) {
  // The index of the slice that is being written.
  dim_t outIdx = 0;

  // For each sample in our batch:
  for (dim_t sample = 0; sample < numSamples; sample++) {
    dim_t sampleStart = sample * sampleSize;

    // For each slice that we fetch:
    for (dim_t i = 0; i < numIndices; i++) {
      dim_t slice = indices[i];

      // Copy the slice.
      memcpy(dest + outIdx * sliceSize, data + sampleStart + slice * sliceSize,
             sliceSize * sizeof(T));

      // Point to the next location in the destination tensor.
      outIdx++;
    }
  }
}

template <typename T, typename U>
static void libjit_gatherranges(T *output, U *lengths, const T *data,
                                const U *ranges, dim_t numExamples,
                                dim_t exampleSize) {
  // Indices into the output and range buffers.
  dim_t outputIdx = 0;
  dim_t rangesIdx = 0;

  // For each example:
  for (dim_t example = 0; example < numExamples; ++example) {
    // Keep track of the total length of the gathered ranges for the example.
    U totalLen = 0;

    // For each range:
    for (dim_t range = 0; range < exampleSize; ++range) {
      // Get the start and length of the range.
      const U start = ranges[rangesIdx];
      const U len = ranges[rangesIdx + 1];

      // Copy the specified elements.
      memcpy(output + outputIdx, data + start, len * sizeof(T));

      // len elements were copied, so increment the output index by len.
      outputIdx += len;

      // Each range is of the form (start, len), so increment the ranges
      // index by 2 to get to the next range.
      rangesIdx += 2;

      // Increment the total length for the example by len.
      totalLen += len;
    }

    // Record the total length of gathered ranges for the current example in
    // the lengths buffer.
    lengths[example] = totalLen;
  }
}

template <typename T, typename T2>
static void libjit_scatterdatacopy(T *data, const dim_t *dataDims,
                                   const T2 *indices, const T *slices,
                                   dim_t numIndices, dim_t indexSize,
                                   dim_t sliceSize) {
  for (dim_t i = 0; i < numIndices; i++) {
    dim_t destDataIdx = indices[i * indexSize];
    for (dim_t j = 1; j < indexSize; j++) {
      destDataIdx *= dataDims[j];
      destDataIdx += indices[i * indexSize + j];
    }
    memcpy(data + destDataIdx * sliceSize, slices + i * sliceSize,
           sliceSize * sizeof(T));
  }
}

template <typename T, typename T2>
static void libjit_scatterdataaddfloat(T *data, const dim_t *dataDims,
                                       const T2 *indices, const T *slices,
                                       dim_t numIndices, dim_t indexSize,
                                       dim_t sliceSize) {
  for (dim_t i = 0; i < numIndices; i++) {
    dim_t destDataIdx = indices[i * indexSize];
    for (dim_t j = 1; j < indexSize; j++) {
      destDataIdx *= dataDims[j];
      destDataIdx += indices[i * indexSize + j];
    }
    for (dim_t j = 0; j < sliceSize; j++) {
      data[destDataIdx * sliceSize + j] += slices[i * sliceSize + j];
    }
  }
}

template <typename T, typename T2>
static void libjit_scatterdataaddquantized(T *data, const dim_t *dataDims,
                                           const T2 *indices, const T *slices,
                                           dim_t numIndices, dim_t indexSize,
                                           dim_t sliceSize, float dataScale,
                                           int32_t dataOffset, float sliceScale,
                                           int32_t sliceOffset) {

  for (size_t i = 0; i < numIndices; i++) {
    size_t destDataIdx = indices[i * indexSize];
    for (size_t j = 1; j < indexSize; j++) {
      destDataIdx *= dataDims[j];
      destDataIdx += indices[i * indexSize + j];
    }
    for (size_t j = 0; j < sliceSize; j++) {
      float lhs = (data[destDataIdx * sliceSize + j] - dataOffset) * dataScale;
      float rhs = (slices[i * sliceSize + j] - sliceOffset) * sliceScale;
      T result = libjit_clip((lhs + rhs) / dataScale + dataOffset);
      data[destDataIdx * sliceSize + j] = result;
    }
  }
}

template <typename T>
static void libjit_transpose_generic(const T *inW, T *outW, const dim_t *idim,
                                     const dim_t *odim, const dim_t *shuffle,
                                     dim_t numDims) {
  // Transpose 2d matrices one tile at a time. This access pattern ensures
  // that the whole tile is kept in L1 cache. When scanning the whole row at
  // once we invalidate many cache lines when we touch a single column.
  const unsigned tileSize = 64;

  // Source coordinate.
  dim_t SC[5];

  if (numDims == 5) {
    for (dim_t x = 0; x < odim[0]; x++)
      for (dim_t y = 0; y < odim[1]; y++)
        for (dim_t z = 0; z < odim[2]; z++)
          for (dim_t w = 0; w < odim[3]; w++)
            for (dim_t q = 0; q < odim[4]; q++) {
              SC[shuffle[0]] = x;
              SC[shuffle[1]] = y;
              SC[shuffle[2]] = z;
              SC[shuffle[3]] = w;
              SC[shuffle[4]] = q;
              outW[libjit_getXYZWQ(odim, x, y, z, w, q)] =
                  inW[libjit_getXYZWQ(idim, SC[0], SC[1], SC[2], SC[3], SC[4])];
            }
    return;
  }
  if (numDims == 4) {
    for (dim_t x = 0; x < odim[0]; x++)
      for (dim_t y = 0; y < odim[1]; y++)
        for (dim_t z = 0; z < odim[2]; z++)
          for (dim_t w = 0; w < odim[3]; w++) {
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
    for (dim_t x = 0; x < odim[0]; x++) {
      // Process the tiles in the innermost two dimensions:
      for (dim_t sy = 0; sy < odim[1]; sy += tileSize) {
        for (dim_t sz = 0; sz < odim[2]; sz += tileSize) {
          // Process the inner tile:
          for (dim_t y = sy; y < MIN(sy + tileSize, odim[1]); y++) {
            for (dim_t z = sz; z < MIN(sz + tileSize, odim[2]); z++) {
              SC[shuffle[0]] = x;
              SC[shuffle[1]] = y;
              SC[shuffle[2]] = z;
              outW[libjit_getXYZ(odim, x, y, z)] =
                  inW[libjit_getXYZ(idim, SC[0], SC[1], SC[2])];
            }
          }
        }
      }
    }
    return;
  }

  if (numDims == 2) {
    // Process the tiles in the matrix:
    for (dim_t sx = 0; sx < odim[0]; sx += tileSize) {
      for (dim_t sy = 0; sy < odim[1]; sy += tileSize) {
        // Process the inner tile:
        for (dim_t x = sx; x < MIN(sx + tileSize, odim[0]); x++) {
          for (dim_t y = sy; y < MIN(sy + tileSize, odim[1]); y++) {
            SC[shuffle[0]] = x;
            SC[shuffle[1]] = y;
            outW[libjit_getXY(odim, x, y)] =
                inW[libjit_getXY(idim, SC[0], SC[1])];
          }
        }
      }
    }
    return;
  }
}

template <typename T>
static void libjit_flip_generic(const T *inW, T *outW, const dim_t *dims,
                                dim_t axis, dim_t numDims) {

  // Product of outer dimensions excluding the flip dimension.
  dim_t outerLen = 1;
  for (dim_t idx = 0; idx < axis; idx++) {
    outerLen *= dims[idx];
  }

  // Flip dimension.
  dim_t len = dims[axis];

  // Product of inner dimensions excluding the flip dimension.
  dim_t innerLen = 1;
  for (dim_t idx = axis + 1; idx < numDims; idx++) {
    innerLen *= dims[idx];
  }

  // Flip axis such that input data is read linearly.
  const T *inpPtr = inW;
  T *outPtr = outW + (len - 1) * innerLen;
  for (dim_t outerIdx = 0; outerIdx < outerLen; outerIdx++) {
    for (dim_t idx = 0; idx < len; idx++) {
      for (dim_t innerIdx = 0; innerIdx < innerLen; innerIdx++) {
        *outPtr++ = *inpPtr++;
      }
      outPtr -= 2 * innerLen;
    }
    outPtr += 2 * len * innerLen;
  }
}

template <typename T>
static void libjit_max_pool_generic(const T *inW, T *outW, const dim_t *inWdims,
                                    const dim_t *outWdims, dim_t *kernelSizes,
                                    dim_t *strides, dim_t *pads) {
  dim_t pad_t = pads[0];
  dim_t pad_l = pads[1];
  dim_t stride_h = strides[0];
  dim_t stride_w = strides[1];
  dim_t kernel_h = kernelSizes[0];
  dim_t kernel_w = kernelSizes[1];
  // For each sample in the batch:
  for (dim_t n = 0; n < outWdims[0]; n++) {
    // For each (x,y) step in the input/output tensor:
    sdim_t x = -(sdim_t)pad_t;
    for (dim_t ax = 0; ax < outWdims[1]; x += stride_h, ax++) {
      sdim_t y = -(sdim_t)pad_l;
      for (dim_t ay = 0; ay < outWdims[2]; y += stride_w, ay++) {

        // For each layer in the output tensor:
        for (dim_t z = 0; z < inWdims[3]; z++) {
          int first = 1;
          T max = 0;

          // For each element in the pool filter:
          for (dim_t fx = 0; fx < kernel_h; fx++) {
            for (dim_t fy = 0; fy < kernel_w; fy++) {
              sdim_t ox = x + fx;
              sdim_t oy = y + fy;

              // Ignore index access below zero (this is due to padding).
              if (ox < 0 || oy < 0 || ox >= (sdim_t)inWdims[1] ||
                  oy >= (sdim_t)inWdims[2]) {
                continue;
              }

              float val =
                  inW[libjit_getXYZW(inWdims, n, (dim_t)ox, (dim_t)oy, z)];

              if (first || (val >= max)) {
                first = 0;
                max = val;
              }
            }
          }

          outW[libjit_getXYZW(outWdims, n, ax, ay, z)] = max;
        } // C
      }   // W
    }     // H
  }       // N
}

template <typename T, typename T2>
static void
libjit_max_pool_argmax_generic(const T *inW, T *outW, T2 *argmax,
                               const dim_t *inWdims, const dim_t *outWdims,
                               dim_t *kernels, dim_t *strides, dim_t *pads) {
  dim_t pad_t = pads[0];
  dim_t pad_l = pads[1];
  dim_t stride_h = strides[0];
  dim_t stride_w = strides[1];
  dim_t kernel_h = kernels[0];
  dim_t kernel_w = kernels[1];
  // For each input in the batch:
  for (dim_t n = 0; n < outWdims[0]; n++) {

    // For each (x,y) step in the input/output tensor:
    sdim_t x = -(sdim_t)pad_t;
    for (dim_t ax = 0; ax < outWdims[1]; x += stride_h, ax++) {
      sdim_t y = -(sdim_t)pad_l;
      for (dim_t ay = 0; ay < outWdims[2]; y += stride_w, ay++) {

        // For each channel in the output tensor:
        for (dim_t z = 0; z < outWdims[3]; z++) {
          int64_t argmaxNHWC = 0;
          int first = 1;
          T max = 0;

          for (dim_t kx = 0; kx < kernel_h; kx++) {
            for (dim_t ky = 0; ky < kernel_w; ky++) {
              sdim_t ox = x + kx;
              sdim_t oy = y + ky;

              if (ox < 0 || oy < 0 || ox >= (sdim_t)inWdims[1] ||
                  oy >= (sdim_t)inWdims[2]) {
                continue;
              }
              const dim_t flatIndex =
                  libjit_getXYZW(inWdims, n, (dim_t)ox, (dim_t)oy, z);
              T val = inW[flatIndex];
              if (first || (val >= max)) {
                first = 0;
                max = val;
                argmaxNHWC = flatIndex;
              }
            }
          }

          const dim_t flatIndex = libjit_getXYZW(outWdims, n, ax, ay, z);
          outW[flatIndex] = max;
          argmax[flatIndex] = argmaxNHWC;
        } // C
      }   // W
    }     // H
  }       // N
}

template <typename T, typename T2>
static void libjit_arg_max_generic(const T *inW, T2 *outW, const dim_t *inWdims,
                                   size_t axis) {

  dim_t a, b, c, d = 0;

  dim_t *dim[4];
  dim[(axis + 1) % 4] = &a;
  dim[(axis + 2) % 4] = &b;
  dim[(axis + 3) % 4] = &c;
  dim[axis] = &d;

  dim_t odim[4] = {inWdims[0], inWdims[1], inWdims[2], inWdims[3]};
  odim[axis] = 1;

  // Iterate over axes != argmax axis.
  for (a = 0; a < inWdims[(axis + 1) % 4]; a++) {
    for (b = 0; b < inWdims[(axis + 2) % 4]; b++) {
      for (c = 0; c < inWdims[(axis + 3) % 4]; c++) {

        T max = inW[libjit_getXYZW(inWdims, *dim[0], *dim[1], *dim[2], 0)];
        dim_t maxi = 0;

        // Iterate over argmax axis.
        for (d = 0; d < inWdims[axis]; d++) {
          T elem =
              inW[libjit_getXYZW(inWdims, *dim[0], *dim[1], *dim[2], *dim[3])];
          if (elem > max) {
            max = elem;
            maxi = d;
          }
        }
        *dim[axis] = 0;
        outW[libjit_getXYZW(odim, *dim[0], *dim[1], *dim[2], *dim[3])] = maxi;
      }
    }
  }
}

template <typename T>
static void
libjit_batchedadd_quantized(int8_t *dest, const int8_t *batch, const T *slice,
                            dim_t numSlice, dim_t sliceSize, int32_t destOffset,
                            int32_t batchOffset, int32_t sliceOffset,
                            int32_t batchPre, int32_t batchPost,
                            int32_t batchScale, int32_t slicePre,
                            int32_t slicePost, int32_t sliceScale) {
  for (dim_t n = 0; n < numSlice; n++) {
    dim_t base = n * sliceSize;
    for (dim_t i = 0; i < sliceSize; i++) {
      int32_t b = batch[base + i] - batchOffset;
      int32_t s = slice[i] - sliceOffset;
      int32_t x = libjit_scale_i32i8(b, batchPre, batchPost, batchScale, 0);
      int32_t y = libjit_scale_i32i8(s, slicePre, slicePost, sliceScale, 0);
      dest[base + i] = libjit_clip(x + y + destOffset);
    }
  }
}

static void find_min_max_f(float *tensor, dim_t size, float &min, float &max) {
  min = tensor[0];
  max = tensor[0];

  for (dim_t i = 1; i < size; ++i) {
    float tensorVal = tensor[i];
    if (tensorVal < min)
      min = tensorVal;

    if (tensorVal > max)
      max = tensorVal;
  }
}

static int check_all_zeros(float *arrayToCheck, dim_t size) {
  for (dim_t i = 0; i < size; ++i) {
    if (arrayToCheck[i] != 0) {
      return 0;
    }
  }
  return 1;
}

/// Gen a bin number to insert \p value into the histogram which has \p nBins
/// with \p minValue and binWidth in histogram.
static dim_t get_bin(dim_t nBins, float binWidth, float minValue, float value) {
  dim_t result =
      binWidth == 0
          ? 0
          : MIN(static_cast<dim_t>((value - minValue) / binWidth), nBins - 1);
  return result;
}

template <typename T>
static void libjit_space_to_depth_generic(const T *inPtr, T *outPtr,
                                          dim_t blockSize, const dim_t *inDims,
                                          const dim_t *outDims) {
  dim_t inHeight = inDims[1];
  dim_t inWidth = inDims[2];
  dim_t inDepth = inDims[3];

  dim_t outBatch = outDims[0];
  dim_t outHeight = outDims[1];
  dim_t outWidth = outDims[2];
  dim_t outDepth = outDims[3];

  for (dim_t b = 0; b < outBatch; ++b) {
    for (dim_t h = 0; h < outHeight; ++h) {
      for (dim_t w = 0; w < outWidth; ++w) {
        for (dim_t c = 0; c < outDepth; ++c) {
          // NHWC
          // c +
          // w * outDepth +
          // h * outDepth * outWidth +
          // b * outDepth * outWidth * outHeight
          dim_t outIndex = c + outDepth * (w + outWidth * (h + b * outHeight));

          // Gets the block layer we are on
          dim_t blockDepthLayer = c / inDepth;
          // every multiple of block size we reset to 0 offset
          dim_t iw = w * blockSize + blockDepthLayer % blockSize;
          // every multiple of blockSize we start height traversal + 1
          dim_t ih = h * blockSize + blockDepthLayer / blockSize;
          // at every multiple of inDepth index in to input depths resets to 0
          dim_t id = c % inDepth;

          dim_t inIndex = id + inDepth * (iw + inWidth * (ih + b * inHeight));
          outPtr[outIndex] = inPtr[inIndex];
        }
      }
    }
  }
}

template <typename DstType, typename SrcType>
static void
libjit_copy_kernel_with_conversion(DstType *dstPtr, const SrcType *srcPtr,
                                   const dim_t *dims, dim_t numDims) {
  dim_t dimSize = 1;
  for (dim_t i = 0; i < numDims; ++i) {
    dimSize *= dims[i];
  }

  for (dim_t i = 0; i < dimSize; ++i) {
    dstPtr[i] = DstType(srcPtr[i]);
  }
}

/// The dimensions passed in here are pre-expanded in LLVMIRGen with 1s so that
/// we can iterate over the shape here, regardless of the shape of the tensor.
template <typename T>
static void libjit_reducemin(T *dest, const T *batch, size_t destSize,
                             const dim_t *destDims, const dim_t *batchDims,
                             T init) {
  for (dim_t i = 0; i < destSize; i++) {
    dest[i] = init;
  }

  unsigned int axis[6];
  for (dim_t i = 0; i < 6; i++) {
    axis[i] = (destDims[i] > 1);
  }

  for (dim_t x = 0, dx = 0; x < batchDims[0]; x++, dx += axis[0]) {
    for (dim_t y = 0, dy = 0; y < batchDims[1]; y++, dy += axis[1]) {
      for (dim_t z = 0, dz = 0; z < batchDims[2]; z++, dz += axis[2]) {
        for (dim_t w = 0, dw = 0; w < batchDims[3]; w++, dw += axis[3]) {
          for (dim_t q = 0, dq = 0; q < batchDims[4]; q++, dq += axis[4]) {
            for (dim_t r = 0, dr = 0; r < batchDims[5]; r++, dr += axis[5]) {
              T fdest =
                  dest[libjit_getXYZWQR(destDims, dx, dy, dz, dw, dq, dr)];
              T fnew = batch[libjit_getXYZWQR(batchDims, x, y, z, w, q, r)];
              dest[libjit_getXYZWQR(destDims, dx, dy, dz, dw, dq, dr)] =
                  std::min(fdest, fnew);
            }
          }
        }
      }
    }
  }
}

template <typename T, typename T2>
static void libjit_cross_entropy_loss_generic(T *CE, T *P, T2 *labels,
                                              dim_t *dims) {
  CE[0] = 0.0;
  for (dim_t n = 0; n < dims[0]; ++n) {
    auto y = labels[n];
    auto p_n = P[libjit_getXY(dims, n, y)];
    CE[0] -= log(p_n);
  }
}

template <typename T, typename T2>
static void libjit_sparse_lengths_sum_generic(T *dest, T *data, T2 *indices,
                                              int32_t *lengths, dim_t segments,
                                              dim_t lineSize) {
  memset(dest, 0, segments * lineSize * sizeof(float));
  dim_t curIndex = 0;
  for (dim_t i = 0; i < segments; i++) {
    for (int32_t j = 0; j < lengths[i]; j++) {
      dim_t line = indices[curIndex];
      for (dim_t k = 0; k < lineSize; k++) {
        dest[i * lineSize + k] += data[line * lineSize + k];
      }
      curIndex++;
    }
  }
}

template <typename T, typename T2>
static void
libjit_sparse_lengths_weighted_sum_generic(T *dest, T *data, float *weights,
                                           T2 *indices, int32_t *lengths,
                                           dim_t segments, dim_t lineSize) {
  memset(dest, 0, segments * lineSize * sizeof(float));
  dim_t curIndex = 0;
  for (dim_t i = 0; i < segments; i++) {
    for (int32_t j = 0; j < lengths[i]; j++) {
      float weight = weights[curIndex];
      dim_t line = indices[curIndex];
      for (dim_t k = 0; k < lineSize; k++) {
        dest[i * lineSize + k] += weight * data[line * lineSize + k];
      }
      curIndex++;
    }
  }
}

template <typename T, typename T2>
static void libjit_sparse_lengths_weighted_sum_grad_generic(
    const T *destGrad, T *dataGrad, T *weightsGrad, const T *data,
    const T *weights, const T2 *indices, const int32_t *lengths, dim_t segments,
    dim_t lineSize, dim_t dataGradRawSize) {
  // The data gradients not touched by this operation should
  // be 0, so set the entire buffer to 0 to start with.
  memset(dataGrad, 0, dataGradRawSize);

  for (dim_t i = 0, curIndex = 0; i < segments; ++i) {
    for (int32_t j = 0; j < lengths[i]; ++j, ++curIndex) {
      // For each index in each segment:
      //    1) accumulate into the corresponding data gradient the product of
      //    the gradient of the result it was added to and the weight that it
      //    was multiplied by during the SparseLengthsWeightedSum operation.
      //
      //    2) accumulate into each weight gradient the reduced sum of the
      //    elementwise product of the result slice that the corresponding
      //    weight produced and the input slice that the weight was multiplied
      //    with.
      float weightGrad = 0.0f;
      float weight = weights[curIndex];
      dim_t line = indices[curIndex];
      for (dim_t k = 0; k < lineSize; ++k) {
        dataGrad[line * lineSize + k] += weight * destGrad[i * lineSize + k];
        weightGrad += destGrad[i * lineSize + k] * data[line * lineSize + k];
      }
      weightsGrad[curIndex] = weightGrad;
    }
  }
}

template <typename T, typename T2>
static void libjit_rowwise_quantized_sparse_lengths_weighted_sum_generic(
    T *dest, uint8_t *data, T *scales, T *offsets, T *weights, T2 *indices,
    int32_t *lengths, dim_t segments, dim_t lineSize) {
  memset(dest, 0, segments * lineSize * sizeof(float));
  dim_t curIndex = 0;
  for (dim_t i = 0; i < segments; i++) {
    for (int32_t j = 0; j < lengths[i]; j++) {
      const float weight = weights[curIndex];
      const dim_t line = indices[curIndex];
      const float scale = scales[line];
      const float offset = offsets[line];
      for (dim_t k = 0; k < lineSize; k++) {
        const float fData = scale * data[line * lineSize + k] + offset;
        dest[i * lineSize + k] += weight * fData;
      }
      curIndex++;
    }
  }
}

template <typename T, typename T2>
static void libjit_fused_rowwise_quantized_sparse_lengths_weighted_sum_generic(
    T *dest, int8_t *data, T *weights, T2 *indices, int32_t *lengths,
    dim_t segments, dim_t inLineSize, dim_t outLineSize) {
  memset(dest, 0, segments * outLineSize * sizeof(float));
  dim_t curIndex = 0;
  for (dim_t i = 0; i < segments; i++) {
    for (int32_t j = 0, e = lengths[i]; j < e; j++) {
      const float weight = weights[curIndex];
      const dim_t line = indices[curIndex];
      const int8_t *currRowScaleOffsetPtr =
          data + ((line + 1) * inLineSize) - 2 * sizeof(float);
      float scale, offset;
      memcpy(&scale, currRowScaleOffsetPtr, sizeof(float));
      memcpy(&offset, currRowScaleOffsetPtr + sizeof(float), sizeof(float));
      for (dim_t k = 0; k < outLineSize; k++) {
        const float fData =
            (scale * (uint8_t)(data[line * inLineSize + k])) + offset;
        dest[i * outLineSize + k] += weight * fData;
      }
      curIndex++;
    }
  }
}

template <typename T, typename T2>
static void libjit_sparse_to_dense_generic(T *dest, const T2 *indices,
                                           const T *values, dim_t numIndices,
                                           dim_t destSize, dim_t valueSize) {
  memset(dest, 0, destSize * sizeof(float));

  for (dim_t i = 0, valuesOffset = 0; i < numIndices;
       ++i, valuesOffset += valueSize) {
    dim_t idx = indices[i];
    dim_t destOffset = idx * valueSize;

    for (size_t j = 0; j < valueSize; ++j) {
      dest[destOffset + j] += values[valuesOffset + j];
    }
  }
}

struct ClassBox {
  float score{0.0f};
  size_t index{0};
};

struct Box {
  float v0{0.0f};
  float v1{0.0f};
  float v2{0.0f};
  float v3{0.0f};
};

struct OutBox {
  float classValue{0.0f};
  size_t batchIndex{0};
  size_t classIndex{0};
  size_t boxIndex{0};
};

static void maxMin(float lhs, float rhs, float &min, float &max) {
  if (lhs >= rhs) {
    min = rhs;
    max = lhs;
  } else {
    min = lhs;
    max = rhs;
  }
}

static bool checkIOU(const Box &sb, const Box &cb, float iouThreshold,
                     size_t centerPointBox) {
  float xSMin = 0.0f;
  float ySMin = 0.0f;
  float xSMax = 0.0f;
  float ySMax = 0.0f;

  float xCMin = 0.0f;
  float yCMin = 0.0f;
  float xCMax = 0.0f;
  float yCMax = 0.0f;

  // Standardizing coordinates so that (xmin, ymin) is upper left corner of a
  // box and (xmax, ymax) is lower right corner of the box.
  if (!centerPointBox) {
    // 0 means coordinates for diagonal ends of a box.
    // Coordinates can either be absolute or normalized.
    maxMin(sb.v0, sb.v2, xSMin, xSMax);
    maxMin(sb.v1, sb.v3, ySMin, ySMax);

    maxMin(cb.v0, cb.v2, xCMin, xCMax);
    maxMin(cb.v1, cb.v3, yCMin, yCMax);
  } else {
    float halfWidthS = sb.v2 / 2.0f;
    float halfHeightS = sb.v3 / 2.0f;
    float halfWidthC = cb.v2 / 2.0f;
    float halfHeightC = cb.v3 / 2.0f;

    xSMin = sb.v0 - halfWidthS;
    ySMin = sb.v1 - halfHeightS;
    xSMax = sb.v0 + halfWidthS;
    ySMax = sb.v1 + halfHeightS;

    xCMin = cb.v0 - halfWidthC;
    yCMin = cb.v1 - halfHeightC;
    xCMax = cb.v0 + halfWidthC;
    yCMax = cb.v1 + halfHeightC;
  }

  // finding upper left and lower right corner of a box formed by intersection.
  float xMin = MAX(xSMin, xCMin);
  float yMin = MAX(ySMin, yCMin);
  float xMax = MIN(xSMax, xCMax);
  float yMax = MIN(ySMax, yCMax);

  float intersectionArea = MAX((0.0f), xMax - xMin) * MAX((0.0f), yMax - yMin);

  if (intersectionArea == 0.0f) {
    return false;
  }

  float sArea = (xSMax - xSMin) * (ySMax - ySMin);
  float cArea = (xCMax - xCMin) * (yCMax - yCMin);
  float unionArea = sArea + cArea - intersectionArea;

  return intersectionArea > iouThreshold * unionArea;
}

// ONNX
// Class/Score [BatchNum][ClassNum][BoxNum]
// Box [BatchNum][BoxNum][4]
// Result [BatchNum*MaxOutputPerBatch][3]
// V4
// Class/Score [BatchNum][BoxNum]
// Boxes [BatdhNum][BoxNum][4]
// Result [BatchNum*MaxOutputPerBatch]
// NumberOfIndicesDetected [BatchNum*MaxOutputPerBatch]
template <typename T>
static void
libjit_nms_generic(T *indices, T *numDetected, const float *boxTensor,
                   const dim_t *boxTensorDims, size_t boxTensorDimSize,
                   const float *scoresTensor, const dim_t *scoresTensorDims,
                   size_t scoresTensorDimSize, const dim_t *resultTensorDims,
                   size_t resultTensorDimSize, unsigned centerPointBox,
                   unsigned maxOutputBoxesPerClass, float iouThreshold,
                   float scoreThreshold, bool isV4) {
  int boxesBoxDim = boxTensorDimSize - 2;

  size_t numBatches = 1;
  size_t numClasses = 1;
  size_t numBoxes = boxTensorDims[boxesBoxDim];

  size_t maxOutputPerBatch = 0;
  if (!isV4) {
    int boxesBatchDim = boxTensorDimSize - 3;
    int scoresBatchDim = scoresTensorDimSize - 3;

    int scoresBoxDim = scoresTensorDimSize - 1;
    int scoresClassDim = scoresTensorDimSize - 2;

    assert(scoresTensorDims[scoresBoxDim] == boxTensorDims[boxesBoxDim] &&
           "Mismatch between number of scores and number of boxes.");
    assert(scoresTensorDims[scoresBatchDim] == boxTensorDims[boxesBatchDim] &&
           "Scores and Box Batch Dimensions don't match.");
    (void)boxesBatchDim;
    (void)scoresBoxDim;
    numBatches = scoresTensorDims[scoresBatchDim];
    numClasses = scoresTensorDims[scoresClassDim];
    numBoxes = boxTensorDims[boxesBoxDim];
    maxOutputPerBatch = resultTensorDims[resultTensorDimSize - 2] / numBatches;
  } else {
    maxOutputPerBatch = resultTensorDims[resultTensorDimSize - 1] / numBatches;
  }

  static_assert(sizeof(Box) == 4 * sizeof(float),
                "Can't reinterpret raw float data as a Box.");
  const Box *boxes = reinterpret_cast<const Box *>(boxTensor);

  auto cmpFunc = [](const ClassBox &cb1, const ClassBox &cb2) -> bool {
    return cb1.score > cb2.score;
  };

  size_t outPutBoxIndex = 0;
  for (size_t batchIndex = 0; batchIndex < numBatches; ++batchIndex) {
    int32_t detectedPerBatch = 0;
    OutBox minBox{scoresTensor[batchIndex * numClasses], batchIndex, 0, 0};
    for (size_t classIndex = 0; classIndex < numClasses; ++classIndex) {
      ClassBox selectedIndices[numBoxes];
      ClassBox potentialBoxes[numBoxes];
      size_t indexPBoxes = 0;
      const float *currClass =
          &scoresTensor[(batchIndex * numClasses + classIndex) * numBoxes];
      for (size_t boxIndex = 0; boxIndex < numBoxes; ++boxIndex) {
        float classScore = currClass[boxIndex];
        if (classScore > scoreThreshold) {
          ClassBox &b = potentialBoxes[indexPBoxes++];
          b.score = classScore;
          b.index = boxIndex;
        }
      }

      std::sort(potentialBoxes, potentialBoxes + indexPBoxes, cmpFunc);

      size_t indexSBoxes = 0;
      size_t detectedPerClass = 0;
      float tScore = minBox.classValue;
      for (unsigned int i = 0; i < indexPBoxes; ++i) {
        ClassBox &pbI = potentialBoxes[i];
        const Box &potentialBox = boxes[batchIndex * numBoxes + pbI.index];
        bool selected = true;
        for (unsigned int j = 0; j < indexSBoxes && selected; ++j) {
          ClassBox &sbI = selectedIndices[j];
          const Box &selectedBox = boxes[batchIndex * numBoxes + sbI.index];
          selected = !checkIOU(selectedBox, potentialBox, iouThreshold,
                               centerPointBox);
        }

        if (selected) {
          selectedIndices[indexSBoxes++] = pbI;
          if (isV4) {
            indices[outPutBoxIndex] = pbI.index;
          } else {
            indices[outPutBoxIndex * 3 + 0] = batchIndex;
            indices[outPutBoxIndex * 3 + 1] = classIndex;
            indices[outPutBoxIndex * 3 + 2] = pbI.index;
          }

          tScore = pbI.score;
          ++outPutBoxIndex;
          ++detectedPerClass;
          ++detectedPerBatch;
        }

        if (detectedPerClass == maxOutputBoxesPerClass) {
          break;
        }
      }

      if (tScore < minBox.classValue) {
        minBox.classValue = tScore;
        if (isV4) {
          minBox.boxIndex = indices[outPutBoxIndex - 1];
        } else {
          minBox.boxIndex = indices[(outPutBoxIndex - 1) * 3 + 2];
        }
        minBox.classIndex = classIndex;
      }
    }

    // Filling the rest of the class with minimum value.
    for (size_t i = detectedPerBatch; i < maxOutputPerBatch; ++i) {
      if (isV4) {
        indices[outPutBoxIndex] = minBox.boxIndex;
      } else {
        indices[outPutBoxIndex * 3 + 0] = minBox.batchIndex;
        indices[outPutBoxIndex * 3 + 1] = minBox.classIndex;
        indices[outPutBoxIndex * 3 + 2] = minBox.boxIndex;
      }

      ++outPutBoxIndex;
    }
    // For ONNX NMS it's not used, for TF Batch Dimension is 1.
    for (size_t i = 0; i < maxOutputBoxesPerClass; ++i) {
      numDetected[batchIndex * maxOutputBoxesPerClass + i] = detectedPerBatch;
    }
  }
}

template <typename T, typename T2>
void libjit_softmax_grad_generic(T *inG, T *outW, const T2 *selectedW,
                                 const dim_t *idim, const dim_t *selectdim) {
  for (dim_t n = 0; n < idim[0]; n++) {
    for (dim_t i = 0; i < idim[1]; i++) {
      float delta = (selectedW[libjit_getXY(selectdim, n, 0)] == i);
      inG[libjit_getXY(idim, n, i)] = outW[libjit_getXY(idim, n, i)] - delta;
    }
  }
}

template <typename T, typename T2>
void libjit_max_pool_argmax_grad_generic(T *inG, const T *outG,
                                         const T2 *argmax, const dim_t *inGdims,
                                         const dim_t *outWdims) {
  // NHWC format is assumed
  for (dim_t n = 0; n < outWdims[0]; n++) {
    for (dim_t z = 0; z < outWdims[3]; z++) {
      // Clear inG
      for (dim_t x = 0; x < inGdims[1]; x++) {
        for (dim_t y = 0; y < inGdims[2]; y++) {
          inG[libjit_getXYZW(inGdims, n, x, y, z)] = 0.0;
        }
      }

      for (dim_t ax = 0; ax < outWdims[1]; ax++) {
        for (dim_t ay = 0; ay < outWdims[2]; ay++) {
          // Reuse precomputed linear index of max element from argmax.
          const dim_t flatIndex = libjit_getXYZW(outWdims, n, ax, ay, z);
          float df = outG[flatIndex];
          inG[argmax[flatIndex]] += df;
        } // W
      }   // H
    }     // C
  }       // N
}
} // namespace

extern "C" {

/// Macro to define a mini-kernel for data-parallel operations. The body of the
/// kernel is auto-generated by the macro.
/// \p name the name of the kernel
/// \p type the type of the tensor elements and of the return value
/// \p body the operation to be performed
#define DEFINE_DATA_PARALLEL_KERNEL(name, type, body)                          \
  type name(dim_t idx, const type *LHS, const type *RHS, const type *op3) {    \
    return body;                                                               \
  }

/// Macro to define a mini-kernel for data-parallel operations. The body of the
/// kernel is not auto-generated by the macro.
/// \p name the name of the kernel
#define DEFINE_DATA_PARALLEL_KERNEL_FUNC(name)                                 \
  float name(dim_t idx, const float *LHS, const float *RHS, const float *op3)

/// Macro to define a mini-kernel for data-parallel operations with immediate
/// operands.
/// \p name the name of the kernel
/// \p type the type of the tensor elements and of the return value
/// \p body the operation to be performed
#define DEFINE_DATA_PARALLEL_KERNEL_WITH_IMM_OPERAND(name, type, body)         \
  type name(dim_t idx, type val, const type *LHS, const type *RHS) {           \
    return body;                                                               \
  }

/// Macro to define a mini-kernel for data-parallel arithmetic quantized
/// operations. The body of the kernel is auto-generated by the macro.
/// \p name the name of the kernel
/// \p type the type of the tensor elements
/// \p body the operation to be performed
#define DEFINE_DATA_PARALLEL_KERNEL_QUANTIZED(name, type, body)                \
  type name(dim_t idx, const type *LHS, const type *RHS, int32_t destOffset,   \
            int32_t lhsOffset, int32_t rhsOffset, int32_t lhsPre,              \
            int32_t lhsPost, int32_t lhsScale, int32_t rhsPre,                 \
            int32_t rhsPost, int32_t rhsScale) {                               \
    int32_t lhs = libjit_scale_i32i8(LHS[idx] - lhsOffset, lhsPre, lhsPost,    \
                                     lhsScale, 0);                             \
    int32_t rhs = libjit_scale_i32i8(RHS[idx] - rhsOffset, rhsPre, rhsPost,    \
                                     rhsScale, 0);                             \
    return libjit_clip((body) + destOffset);                                   \
  }

/// Macro to define a mini-kernel for data-parallel multiplicative quantized
/// operations. The body of the kernel is auto-generated by the macro.
/// \p name the name of the kernel
/// \p type the type of the tensor elements
/// \p body the operation to be performed
#define DEFINE_DATA_PARALLEL_KERNEL_QUANTIZED_M(name, body)                    \
  int8_t name(dim_t idx, const int8_t *LHS, const int8_t *RHS,                 \
              int32_t destOffset, int32_t lhsOffset, int32_t rhsOffset,        \
              int32_t pre, int32_t post, int32_t scale) {                      \
    int32_t lhs = LHS[idx] - lhsOffset;                                        \
    int32_t rhs = RHS[idx] - rhsOffset;                                        \
    return libjit_clip(                                                        \
        libjit_scale_i32i8((body), pre, post, scale, destOffset));             \
  }

/// Define mini-kernels for all data parallel operations. They are invoked from
/// the generated kernels for sequences of data parallel operations.
DEFINE_DATA_PARALLEL_KERNEL(libjit_elementmax_kernel_f, float,
                            MAX(LHS[idx], RHS[idx]))
DEFINE_DATA_PARALLEL_KERNEL(libjit_elementmin_kernel_f, float,
                            MIN(LHS[idx], RHS[idx]))
DEFINE_DATA_PARALLEL_KERNEL(libjit_copy_kernel_f, float, LHS[idx])
DEFINE_DATA_PARALLEL_KERNEL(libjit_copy_kernel_u, int64_t, LHS[idx])
DEFINE_DATA_PARALLEL_KERNEL(libjit_copy_kernel_i8, int8_t, LHS[idx])
DEFINE_DATA_PARALLEL_KERNEL(libjit_copy_kernel_i32, int32_t, LHS[idx])
DEFINE_DATA_PARALLEL_KERNEL(libjit_copy_kernel_b, int8_t, LHS[idx])
DEFINE_DATA_PARALLEL_KERNEL(libjit_element_add_kernel_f, float,
                            LHS[idx] + RHS[idx])
DEFINE_DATA_PARALLEL_KERNEL(libjit_element_add_kernel_i32, int32_t,
                            LHS[idx] + RHS[idx])
DEFINE_DATA_PARALLEL_KERNEL(libjit_element_sub_kernel_f, float,
                            LHS[idx] - RHS[idx])
DEFINE_DATA_PARALLEL_KERNEL(libjit_element_div_kernel_f, float,
                            LHS[idx] / RHS[idx])
DEFINE_DATA_PARALLEL_KERNEL(libjit_element_div_kernel_u, int64_t,
                            LHS[idx] / RHS[idx])
DEFINE_DATA_PARALLEL_KERNEL(libjit_element_div_kernel_i32, int32_t,
                            LHS[idx] / RHS[idx])
DEFINE_DATA_PARALLEL_KERNEL(libjit_element_mul_kernel_f, float,
                            LHS[idx] * RHS[idx])
DEFINE_DATA_PARALLEL_KERNEL(libjit_element_mul_kernel_i32, int32_t,
                            LHS[idx] * RHS[idx])
DEFINE_DATA_PARALLEL_KERNEL(libjit_element_pow_kernel_f, float,
                            pow(LHS[idx], RHS[idx]))
DEFINE_DATA_PARALLEL_KERNEL(libjit_element_log_kernel_f, float, log(LHS[idx]))
DEFINE_DATA_PARALLEL_KERNEL(libjit_element_exp_kernel_f, float, exp(LHS[idx]))
DEFINE_DATA_PARALLEL_KERNEL_QUANTIZED(libjit_element_add_kernel_i8, int8_t,
                                      lhs + rhs)
DEFINE_DATA_PARALLEL_KERNEL_QUANTIZED(libjit_element_sub_kernel_i8, int8_t,
                                      lhs - rhs)
DEFINE_DATA_PARALLEL_KERNEL_QUANTIZED(libjit_elementmax_kernel_i8, int8_t,
                                      MAX(lhs, rhs))
DEFINE_DATA_PARALLEL_KERNEL_QUANTIZED(libjit_elementmin_kernel_i8, int8_t,
                                      MIN(lhs, rhs))
DEFINE_DATA_PARALLEL_KERNEL_QUANTIZED_M(libjit_element_mul_kernel_i8, lhs *rhs)
DEFINE_DATA_PARALLEL_KERNEL_QUANTIZED_M(libjit_element_div_kernel_i8, lhs / rhs)

/// This is a variable used by Glow backends to determine the actual type used
/// for size_t and dim_t variables when libjit was compiled.
size_t libjit_sizeTVar;
dim_t libjit_dimTVar;

/// Specialize the Modulo kernel into two functions based on the
/// value of SignFollowDivisor.
int64_t libjit_element_modulo_kernel_sign_follow_u(dim_t idx,
                                                   const int64_t divisor,
                                                   const int64_t *input) {
  int64_t res = input[idx] % divisor;
  if (res && ((res > 0) != (divisor > 0))) {
    res += divisor;
  }
  return res;
}

int64_t libjit_element_modulo_kernel_no_sign_follow_u(dim_t idx,
                                                      const int64_t divisor,
                                                      const int64_t *input) {
  return input[idx] % divisor;
}

int32_t libjit_element_modulo_kernel_sign_follow_i32(dim_t idx,
                                                     const int64_t divisor,
                                                     const int32_t *input) {
  int32_t res = input[idx] % divisor;
  if (res && ((res > 0) != (divisor > 0))) {
    res += divisor;
  }
  return res;
}

int32_t libjit_element_modulo_kernel_no_sign_follow_i32(dim_t idx,
                                                        const int64_t divisor,
                                                        const int32_t *input) {
  return input[idx] % divisor;
}

int8_t libjit_element_cmp_eq_kernel_u(dim_t idx, const size_t *LHS,
                                      const size_t *RHS) {
  return LHS[idx] == RHS[idx] ? 1 : 0;
}

int8_t libjit_element_cmp_eq_kernel_i32(size_t idx, const int32_t *LHS,
                                        const int32_t *RHS) {
  return LHS[idx] == RHS[idx] ? 1 : 0;
}

int8_t libjit_element_is_nan_kernel_f(dim_t idx, const float *input) {
  return std::isnan(input[idx]) ? 1 : 0;
}

int8_t libjit_element_cmp_lte_kernel_f(dim_t idx, const float *LHS,
                                       const float *RHS) {
  return LHS[idx] <= RHS[idx] ? 1 : 0;
}

int8_t libjit_element_cmp_lte_kernel_i8(dim_t idx, const int8_t *LHS,
                                        const int8_t *RHS, int32_t lhsOffset,
                                        int32_t rhsOffset, int32_t pre,
                                        int32_t post, int32_t scale) {
  int32_t lhs = LHS[idx] - lhsOffset;
  int32_t rhs = RHS[idx] - rhsOffset;
  return libjit_scale_i32i8(lhs, pre, post, scale, 0) <= rhs ? 1 : 0;
}

int8_t libjit_element_cmp_lt_kernel_f(dim_t idx, const float *LHS,
                                      const float *RHS) {
  return LHS[idx] < RHS[idx] ? 1 : 0;
}

int8_t libjit_element_cmp_lt_kernel_i32(dim_t idx, const int32_t *LHS,
                                        const int32_t *RHS) {
  return LHS[idx] < RHS[idx] ? 1 : 0;
}

int8_t libjit_element_cmp_lt_kernel_i8(dim_t idx, const int8_t *LHS,
                                       const int8_t *RHS, int32_t lhsOffset,
                                       int32_t rhsOffset, int32_t pre,
                                       int32_t post, int32_t scale) {
  int32_t lhs = LHS[idx] - lhsOffset;
  int32_t rhs = RHS[idx] - rhsOffset;
  return libjit_scale_i32i8(lhs, pre, post, scale, 0) < rhs ? 1 : 0;
}

// tanh cannot be vectorized by LLVM yet. Therefore we use the following
// formula instead: 1 - 2 / (exp(x * 2) + 1), which is also used by Caffe2 and
// provides a good accuracy.
// Once LLVM supports the vectorization of tanh, we can replace this
// approximation by a direct tanh call.
DEFINE_DATA_PARALLEL_KERNEL(libjit_tanh_kernel_f, float,
                            1 - 2 / (expf(LHS[idx] * 2) + 1))

int8_t libjit_intlookuptable_kernel_i8(dim_t idx, const int8_t *src,
                                       const int8_t *mapping) {
  return mapping[src[idx] + 128];
}

float libjit_elementselect_kernel_f(dim_t idx, const int8_t *cond,
                                    const float *LHS, const float *RHS) {
  return (cond[idx] != 0) ? LHS[idx] : RHS[idx];
}

int8_t libjit_elementselect_kernel_i8(dim_t idx, const int8_t *cond,
                                      const int8_t *LHS, const int8_t *RHS,
                                      int32_t destOffset, int32_t lhsOffset,
                                      int32_t rhsOffset, int32_t lhsPre,
                                      int32_t lhsPost, int32_t lhsScale,
                                      int32_t rhsPre, int32_t rhsPost,
                                      int32_t rhsScale) {
  return (cond[idx] != 0)
             ? libjit_clip(libjit_scale_i32i8(LHS[idx] - lhsOffset, lhsPre,
                                              lhsPost, lhsScale, destOffset))
             : libjit_clip(libjit_scale_i32i8(RHS[idx] - rhsOffset, rhsPre,
                                              rhsPost, rhsScale, destOffset));
}

DEFINE_DATA_PARALLEL_KERNEL_FUNC(libjit_sigmoid_kernel_f) {
  float e = expf(-LHS[idx]);
  return 1 / (e + 1);
}
DEFINE_DATA_PARALLEL_KERNEL_WITH_IMM_OPERAND(libjit_element_maxsplat_kernel_f,
                                             float, MAX(LHS[idx], val))
DEFINE_DATA_PARALLEL_KERNEL_WITH_IMM_OPERAND(libjit_element_maxsplat_kernel_i8,
                                             int8_t, MAX(LHS[idx], val))
DEFINE_DATA_PARALLEL_KERNEL_WITH_IMM_OPERAND(libjit_splat_kernel_f, float, val)
DEFINE_DATA_PARALLEL_KERNEL_WITH_IMM_OPERAND(libjit_splat_kernel_u, int64_t,
                                             val)
DEFINE_DATA_PARALLEL_KERNEL_WITH_IMM_OPERAND(libjit_splat_kernel_i8, int8_t,
                                             val)
DEFINE_DATA_PARALLEL_KERNEL_WITH_IMM_OPERAND(libjit_splat_kernel_i32, int32_t,
                                             val)
DEFINE_DATA_PARALLEL_KERNEL_WITH_IMM_OPERAND(libjit_splat_kernel_b, int8_t, val)

#undef DEFINE_DATA_PARALLEL_KERNEL
#undef DEFINE_DATA_PARALLEL_KERNEL_FUNC
#undef DEFINE_DATA_PARALLEL_KERNEL_FUNC
#undef DEFINE_DATA_PARALLEL_KERNEL_WITH_IMM_OPERAND

void libjit_batchedadd_f(float *dest, const float *batch, const float *slice,
                         dim_t numSlice, dim_t sliceSize) {
  // For each layer in the batch:
  for (dim_t n = 0; n < numSlice; n++) {
    dim_t base = n * sliceSize;
    // For each element in the slice.
    for (dim_t i = 0; i < sliceSize; i++) {
      dest[base + i] = batch[base + i] + slice[i];
    }
  }
}

void libjit_batchedadd_i8(int8_t *dest, const int8_t *batch,
                          const int8_t *slice, dim_t numSlice, dim_t sliceSize,
                          int32_t destOffset, int32_t batchOffset,
                          int32_t sliceOffset, int32_t batchPre,
                          int32_t batchPost, int32_t batchScale,
                          int32_t slicePre, int32_t slicePost,
                          int32_t sliceScale) {
  libjit_batchedadd_quantized(dest, batch, slice, numSlice, sliceSize,
                              destOffset, batchOffset, sliceOffset, batchPre,
                              batchPost, batchScale, slicePre, slicePost,
                              sliceScale);
}

void libjit_batchedadd_i32_i8(int8_t *dest, const int8_t *batch,
                              const int32_t *slice, dim_t numSlice,
                              dim_t sliceSize, int32_t destOffset,
                              int32_t batchOffset, int32_t sliceOffset,
                              int32_t batchPre, int32_t batchPost,
                              int32_t batchScale, int32_t slicePre,
                              int32_t slicePost, int32_t sliceScale) {
  libjit_batchedadd_quantized(dest, batch, slice, numSlice, sliceSize,
                              destOffset, batchOffset, sliceOffset, batchPre,
                              batchPost, batchScale, slicePre, slicePost,
                              sliceScale);
}

/// The dimensions passed in here are pre-expanded in LLVMIRGen with 1s so that
/// we can iterate over the shape here, regardless of the shape of the tensor.
void libjit_batchedreduceadd_f(float *dest, const float *batch, dim_t destSize,
                               const dim_t *destDims, const dim_t *batchDims,
                               dim_t axis) {
  for (dim_t i = 0; i < destSize; i++)
    dest[i] = 0.0;

  for (dim_t x = 0; x < batchDims[0]; x++)
    for (dim_t y = 0; y < batchDims[1]; y++)
      for (dim_t z = 0; z < batchDims[2]; z++)
        for (dim_t w = 0; w < batchDims[3]; w++)
          for (dim_t q = 0; q < batchDims[4]; q++)
            for (dim_t r = 0; r < batchDims[5]; r++) {
              dim_t I[] = {x, y, z, w, q, r};
              I[axis] = 0;
              dest[libjit_getXYZWQR(destDims, I[0], I[1], I[2], I[3], I[4],
                                    I[5])] +=
                  batch[libjit_getXYZWQR(batchDims, x, y, z, w, q, r)];
            }
}

void libjit_reducemin_f(float *dest, const float *batch, size_t destSize,
                        const dim_t *destDims, const dim_t *batchDims) {
  libjit_reducemin(dest, batch, destSize, destDims, batchDims,
                   std::numeric_limits<float>::max());
}

void libjit_reducemin_i32(int32_t *dest, const int32_t *batch, size_t destSize,
                          const dim_t *destDims, const dim_t *batchDims) {
  libjit_reducemin(dest, batch, destSize, destDims, batchDims,
                   std::numeric_limits<int32_t>::max());
}

void libjit_reducemin_u(int64_t *dest, const int64_t *batch, size_t destSize,
                        const dim_t *destDims, const dim_t *batchDims) {
  libjit_reducemin(dest, batch, destSize, destDims, batchDims,
                   std::numeric_limits<int64_t>::max());
}

/// Same as the non-quantized version, the dimensions here are pre-expanded in
/// LLVMIRGen. However, for quantization, we must accumulate in the inner-most
/// loop with higher precision (int32_t) and then clip the result back into the
/// dest tensor. Thus we add max_tensor_dimensions different cases for this to
/// ensure the axis is used as the inner-most loop.
void libjit_batchedreduceadd_i8(int8_t *dest, const int8_t *batch,
                                const dim_t *destDims, const dim_t *batchDims,
                                int32_t destOffset, int32_t batchOffset,
                                int32_t batchPre, int32_t batchPost,
                                int32_t batchScale, dim_t axis) {
  switch (axis) {
#define LOOP_AXIS_CASE(_D0, _D1, _D2, _D3, _D4, _D5_AXIS)                      \
  case _D5_AXIS:                                                               \
    for (dim_t i##_D0 = 0; i##_D0 < batchDims[_D0]; i##_D0++)                  \
      for (dim_t i##_D1 = 0; i##_D1 < batchDims[_D1]; i##_D1++)                \
        for (dim_t i##_D2 = 0; i##_D2 < batchDims[_D2]; i##_D2++)              \
          for (dim_t i##_D3 = 0; i##_D3 < batchDims[_D3]; i##_D3++)            \
            for (dim_t i##_D4 = 0; i##_D4 < batchDims[_D4]; i##_D4++) {        \
              int32_t sum = 0.0;                                               \
              for (dim_t i##_D5_AXIS = 0; i##_D5_AXIS < batchDims[_D5_AXIS];   \
                   i##_D5_AXIS++) {                                            \
                sum += batch[libjit_getXYZWQR(batchDims, i0, i1, i2, i3, i4,   \
                                              i5)] -                           \
                       batchOffset;                                            \
              }                                                                \
              dim_t i##_D5_AXIS = 0;                                           \
              int32_t res = libjit_scale_i32i8(sum, batchPre, batchPost,       \
                                               batchScale, destOffset);        \
              dest[libjit_getXYZWQR(destDims, i0, i1, i2, i3, i4, i5)] =       \
                  libjit_clip(res);                                            \
            }                                                                  \
    return;

    // Each loop order, with the inner-most dimension/index equal to the axis.
    LOOP_AXIS_CASE(1, 2, 3, 4, 5, 0);
    LOOP_AXIS_CASE(0, 2, 3, 4, 5, 1);
    LOOP_AXIS_CASE(0, 1, 3, 4, 5, 2);
    LOOP_AXIS_CASE(0, 1, 2, 4, 5, 3);
    LOOP_AXIS_CASE(0, 1, 2, 3, 5, 4);
    LOOP_AXIS_CASE(0, 1, 2, 3, 4, 5);
#undef LOOP_AXIS_CASE
  }
}

void libjit_cross_entropy_loss_f_u(float *CE, float *P, size_t *labels,
                                   dim_t *dims) {
  libjit_cross_entropy_loss_generic(CE, P, labels, dims);
}

void libjit_cross_entropy_loss_f_i32(float *CE, float *P, int32_t *labels,
                                     dim_t *dims) {
  libjit_cross_entropy_loss_generic(CE, P, labels, dims);
}

void libjit_gather64_f(float *dest, const float *data, const int64_t *indices,
                       dim_t numIndices, dim_t sliceSize, dim_t numSamples,
                       dim_t sampleSize) {
  libjit_gather(dest, data, indices, numIndices, sliceSize, numSamples,
                sampleSize);
}

void libjit_gather64_i8(int8_t *dest, const int8_t *data,
                        const int64_t *indices, dim_t numIndices,
                        dim_t sliceSize, dim_t numSamples, dim_t sampleSize) {
  libjit_gather(dest, data, indices, numIndices, sliceSize, numSamples,
                sampleSize);
}

void libjit_gather64_u(int64_t *dest, const int64_t *data,
                       const int64_t *indices, dim_t numIndices,
                       dim_t sliceSize, dim_t numSamples, dim_t sampleSize) {
  libjit_gather(dest, data, indices, numIndices, sliceSize, numSamples,
                sampleSize);
}

void libjit_gather32_f(float *dest, const float *data, const int32_t *indices,
                       dim_t numIndices, dim_t sliceSize, dim_t numSamples,
                       dim_t sampleSize) {
  libjit_gather(dest, data, indices, numIndices, sliceSize, numSamples,
                sampleSize);
}

void libjit_gather32_i8(int8_t *dest, const int8_t *data,
                        const int32_t *indices, dim_t numIndices,
                        dim_t sliceSize, dim_t numSamples, dim_t sampleSize) {
  libjit_gather(dest, data, indices, numIndices, sliceSize, numSamples,
                sampleSize);
}

void libjit_gather32_u(int64_t *dest, const int64_t *data,
                       const int32_t *indices, dim_t numIndices,
                       dim_t sliceSize, dim_t numSamples, dim_t sampleSize) {
  libjit_gather(dest, data, indices, numIndices, sliceSize, numSamples,
                sampleSize);
}

void libjit_gather32_i32(int32_t *dest, const int32_t *data,
                         const int32_t *indices, dim_t numIndices,
                         dim_t sliceSize, dim_t numSamples, dim_t sampleSize) {
  libjit_gather(dest, data, indices, numIndices, sliceSize, numSamples,
                sampleSize);
}

void libjit_gatherranges64_f(float *output, int64_t *lengths, const float *data,
                             const int64_t *ranges, dim_t numExamples,
                             dim_t exampleSize) {
  libjit_gatherranges(output, lengths, data, ranges, numExamples, exampleSize);
}

void libjit_gatherranges64_i8(int8_t *output, int64_t *lengths,
                              const int8_t *data, const int64_t *ranges,
                              dim_t numExamples, dim_t exampleSize) {
  libjit_gatherranges(output, lengths, data, ranges, numExamples, exampleSize);
}

void libjit_gatherranges64_u(int64_t *output, int64_t *lengths,
                             const int64_t *data, const int64_t *ranges,
                             dim_t numExamples, dim_t exampleSize) {
  libjit_gatherranges(output, lengths, data, ranges, numExamples, exampleSize);
}

void libjit_gatherranges32_f(float *output, int32_t *lengths, const float *data,
                             const int32_t *ranges, dim_t numExamples,
                             dim_t exampleSize) {
  libjit_gatherranges(output, lengths, data, ranges, numExamples, exampleSize);
}

void libjit_gatherranges32_i8(int8_t *output, int32_t *lengths,
                              const int8_t *data, const int32_t *ranges,
                              dim_t numExamples, dim_t exampleSize) {
  libjit_gatherranges(output, lengths, data, ranges, numExamples, exampleSize);
}

void libjit_gatherranges32_u(uint64_t *output, int32_t *lengths,
                             const uint64_t *data, const int32_t *ranges,
                             dim_t numExamples, dim_t exampleSize) {
  libjit_gatherranges(output, lengths, data, ranges, numExamples, exampleSize);
}

void libjit_gatherranges32_i32(int32_t *output, int32_t *lengths,
                               const int32_t *data, const int32_t *ranges,
                               size_t numExamples, size_t exampleSize) {
  libjit_gatherranges(output, lengths, data, ranges, numExamples, exampleSize);
}

void libjit_lengths_range_fill_i32(const int32_t *lengths, int32_t *output,
                                   const dim_t lengthsSize) {
  dim_t curIdx = 0;
  for (dim_t i = 0, e = lengthsSize; i < e; i++) {
    for (int32_t j = 0, f = lengths[i]; j < f; j++) {
      output[curIdx++] = j;
    }
  }
}

void libjit_scatterdata_f_i32(float *data, const dim_t *dataDims,
                              const int32_t *indices, const float *slices,
                              dim_t numIndices, dim_t indexSize,
                              dim_t sliceSize, bool isCumulative) {
  if (isCumulative) {
    libjit_scatterdataaddfloat(data, dataDims, indices, slices, numIndices,
                               indexSize, sliceSize);
  } else {
    libjit_scatterdatacopy(data, dataDims, indices, slices, numIndices,
                           indexSize, sliceSize);
  }
}

void libjit_scatterdata_i8_u(int8_t *data, const dim_t *dataDims,
                             const size_t *indices, const int8_t *slices,
                             size_t numIndices, size_t indexSize,
                             size_t sliceSize, bool isCumulative,
                             float dataScale, int32_t dataOffset,
                             float sliceScale, int32_t sliceOffset) {
  if (isCumulative) {
    libjit_scatterdataaddquantized(data, dataDims, indices, slices, numIndices,
                                   indexSize, sliceSize, dataScale, dataOffset,
                                   sliceScale, sliceOffset);
  } else {
    libjit_scatterdatacopy(data, dataDims, indices, slices, numIndices,
                           indexSize, sliceSize);
  }
}

void libjit_scatterdata_i8_i32(int8_t *data, const dim_t *dataDims,
                               const int32_t *indices, const int8_t *slices,
                               size_t numIndices, size_t indexSize,
                               size_t sliceSize, bool isCumulative,
                               float dataScale, int32_t dataOffset,
                               float sliceScale, int32_t sliceOffset) {
  if (isCumulative) {
    libjit_scatterdataaddquantized(data, dataDims, indices, slices, numIndices,
                                   indexSize, sliceSize, dataScale, dataOffset,
                                   sliceScale, sliceOffset);
  } else {
    libjit_scatterdatacopy(data, dataDims, indices, slices, numIndices,
                           indexSize, sliceSize);
  }
}

void libjit_lengths_to_ranges_i32(int32_t *ranges, const int32_t *lengths,
                                  dim_t size) {
  int32_t offset = 0;
  for (dim_t i = 0; i < size; i++) {
    auto length = lengths[i];
    ranges[i * 2] = offset;
    ranges[i * 2 + 1] = length;
    offset += length;
  }
}

void libjit_sparse_lengths_sum_f_u(float *dest, float *data, size_t *indices,
                                   int32_t *lengths, dim_t segments,
                                   dim_t lineSize) {
  libjit_sparse_lengths_sum_generic(dest, data, indices, lengths, segments,
                                    lineSize);
}

void libjit_sparse_lengths_sum_f_i32(float *dest, float *data, int32_t *indices,
                                     int32_t *lengths, dim_t segments,
                                     dim_t lineSize) {
  libjit_sparse_lengths_sum_generic(dest, data, indices, lengths, segments,
                                    lineSize);
}

void libjit_sparse_lengths_weighted_sum_f_u(float *dest, float *data,
                                            float *weights, size_t *indices,
                                            int32_t *lengths, dim_t segments,
                                            dim_t lineSize) {
  libjit_sparse_lengths_weighted_sum_generic(dest, data, weights, indices,
                                             lengths, segments, lineSize);
}

void libjit_sparse_lengths_weighted_sum_f_i32(float *dest, float *data,
                                              float *weights, int32_t *indices,
                                              int32_t *lengths, dim_t segments,
                                              dim_t lineSize) {
  libjit_sparse_lengths_weighted_sum_generic(dest, data, weights, indices,
                                             lengths, segments, lineSize);
}

void libjit_embedding_bag_f(float *dest, float *data, float *weights,
                            size_t *indices, size_t *offsets, dim_t segments,
                            dim_t lineSize, dim_t totalLength,
                            bool hasEndOffset) {
  if (hasEndOffset) {
    --segments;
  }
  memset(dest, 0, segments * lineSize * sizeof(float));
  dim_t curIndex = 0;
  for (dim_t i = 0; i < segments; i++) {
    int64_t start = offsets[i];
    int64_t end =
        !hasEndOffset && i == segments - 1 ? totalLength : offsets[i + 1];
    for (int64_t j = start; j < end; j++) {
      float weight = weights[curIndex];
      dim_t line = indices[curIndex];
      for (dim_t k = 0; k < lineSize; k++) {
        dest[i * lineSize + k] += weight * data[line * lineSize + k];
      }
      curIndex++;
    }
  }
}

void libjit_sparse_lengths_weighted_sum_grad_f_u(
    const float *destGrad, float *dataGrad, float *weightsGrad,
    const float *data, const float *weights, const size_t *indices,
    const int32_t *lengths, dim_t segments, dim_t lineSize,
    dim_t dataGradRawSize) {
  libjit_sparse_lengths_weighted_sum_grad_generic(
      destGrad, dataGrad, weightsGrad, data, weights, indices, lengths,
      segments, lineSize, dataGradRawSize);
}

void libjit_sparse_lengths_weighted_sum_grad_f_i32(
    const float *destGrad, float *dataGrad, float *weightsGrad,
    const float *data, const float *weights, const int32_t *indices,
    const int32_t *lengths, dim_t segments, dim_t lineSize,
    dim_t dataGradRawSize) {
  libjit_sparse_lengths_weighted_sum_grad_generic(
      destGrad, dataGrad, weightsGrad, data, weights, indices, lengths,
      segments, lineSize, dataGradRawSize);
}

void libjit_rowwise_quantized_sparse_lengths_weighted_sum_f_u(
    float *dest, uint8_t *data, float *scales, float *offsets, float *weights,
    size_t *indices, int32_t *lengths, dim_t segments, dim_t lineSize) {
  libjit_rowwise_quantized_sparse_lengths_weighted_sum_generic(
      dest, data, scales, offsets, weights, indices, lengths, segments,
      lineSize);
}

void libjit_rowwise_quantized_sparse_lengths_weighted_sum_f_i32(
    float *dest, uint8_t *data, float *scales, float *offsets, float *weights,
    int32_t *indices, int32_t *lengths, dim_t segments, dim_t lineSize) {
  libjit_rowwise_quantized_sparse_lengths_weighted_sum_generic(
      dest, data, scales, offsets, weights, indices, lengths, segments,
      lineSize);
}

void libjit_fused_rowwise_quantized_sparse_lengths_weighted_sum_f_u(
    float *dest, int8_t *data, float *weights, size_t *indices,
    int32_t *lengths, dim_t segments, dim_t inLineSize, dim_t outLineSize) {
  libjit_fused_rowwise_quantized_sparse_lengths_weighted_sum_generic(
      dest, data, weights, indices, lengths, segments, inLineSize, outLineSize);
}

void libjit_fused_rowwise_quantized_sparse_lengths_weighted_sum_f_i32(
    float *dest, int8_t *data, float *weights, int32_t *indices,
    int32_t *lengths, dim_t segments, dim_t inLineSize, dim_t outLineSize) {
  libjit_fused_rowwise_quantized_sparse_lengths_weighted_sum_generic(
      dest, data, weights, indices, lengths, segments, inLineSize, outLineSize);
}

void libjit_fused_rowwise_quantized_sparse_lengths_weighted_sum_f(
    float *dest, int8_t *data, float *weights, dim_t *indices, int32_t *lengths,
    dim_t segments, dim_t inLineSize, dim_t outLineSize) {
  memset(dest, 0, segments * outLineSize * sizeof(float));
  dim_t curIndex = 0;
  for (dim_t i = 0; i < segments; i++) {
    for (int32_t j = 0, e = lengths[i]; j < e; j++) {
      const float weight = weights[curIndex];
      const dim_t line = indices[curIndex];
      const int8_t *currRowScaleOffsetPtr =
          data + ((line + 1) * inLineSize) - 2 * sizeof(float);
      float scale, offset;
      memcpy(&scale, currRowScaleOffsetPtr, sizeof(float));
      memcpy(&offset, currRowScaleOffsetPtr + sizeof(float), sizeof(float));
      for (dim_t k = 0; k < outLineSize; k++) {
        const float fData =
            (scale * (uint8_t)(data[line * inLineSize + k])) + offset;
        dest[i * outLineSize + k] += weight * fData;
      }
      curIndex++;
    }
  }
}

void libjit_embedding_bag_byte_rowwise_offsets_f(
    float *dest, int8_t *data, float *weights, size_t *indices, size_t *offsets,
    dim_t segments, dim_t numIndices, dim_t inLineSize, dim_t outLineSize,
    bool hasEndOffset) {
  if (hasEndOffset) {
    --segments;
  }
  memset(dest, 0, segments * outLineSize * sizeof(float));
  for (dim_t i = 0; i < segments; i++) {
    dim_t start = offsets[i];
    dim_t end =
        !hasEndOffset && i == segments - 1 ? numIndices : offsets[i + 1];
    for (dim_t j = start; j < end; j++) {
      const float weight = weights[j];
      const dim_t line = indices[j];
      const int8_t *currRowScaleOffsetPtr =
          data + ((line + 1) * inLineSize) - 2 * sizeof(float);
      float scale, offset;
      memcpy(&scale, currRowScaleOffsetPtr, sizeof(float));
      memcpy(&offset, currRowScaleOffsetPtr + sizeof(float), sizeof(float));
      for (dim_t k = 0; k < outLineSize; k++) {
        const float fData =
            (scale * (uint8_t)(data[line * inLineSize + k])) + offset;
        dest[i * outLineSize + k] += weight * fData;
      }
    }
  }
}

void libjit_sparse_to_dense_f_u(float *dest, const size_t *indices,
                                const float *values, dim_t numIndices,
                                dim_t destSize, dim_t valueSize) {
  libjit_sparse_to_dense_generic(dest, indices, values, numIndices, destSize,
                                 valueSize);
}

void libjit_sparse_to_dense_f_i32(float *dest, const int32_t *indices,
                                  const float *values, dim_t numIndices,
                                  dim_t destSize, dim_t valueSize) {
  libjit_sparse_to_dense_generic(dest, indices, values, numIndices, destSize,
                                 valueSize);
}

void libjit_lengths_sum_f(float *dest, const float *data,
                          const int32_t *lengths, dim_t destSize,
                          dim_t lengthsSize, dim_t sliceSize) {
  memset(dest, 0, destSize * sizeof(float));

  dim_t offsetOut = 0;
  dim_t offsetIn = 0;

  for (dim_t i = 0; i < lengthsSize; ++i) {
    for (int32_t j = 0; j < lengths[i]; ++j) {
      for (dim_t k = 0; k < sliceSize; ++k) {
        dest[offsetOut + k] += data[offsetIn + k];
      }
      offsetIn += sliceSize;
    }
    offsetOut += sliceSize;
  }
}

void libjit_local_response_normalization_f(
    float *outW, const float *inW, float *scaleCache, const dim_t *outWdims,
    const dim_t *inWdims, dim_t halfWindow, float alpha, float beta, float k) {
  dim_t window = 2 * halfWindow + 1;
  float normedAlpha = alpha / window;

  for (dim_t n = 0; n < inWdims[0]; n++) {
    for (dim_t h = 0; h < inWdims[1]; h++) {
      for (dim_t w = 0; w < inWdims[2]; w++) {
        for (dim_t c = 0; c < inWdims[3]; c++) {
          float m2 = 0.0;
          for (dim_t i = (c >= halfWindow ? c - halfWindow : 0);
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
    const float *scaleCache, const dim_t *outWdims, dim_t halfWindow,
    float alpha, float beta) {
  dim_t window = 2 * halfWindow + 1;
  float normedAlpha = alpha / window;
  float coeff = 2 * normedAlpha * beta;

  for (dim_t n = 0; n < outWdims[0]; n++) {
    for (dim_t h = 0; h < outWdims[1]; h++) {
      for (dim_t w = 0; w < outWdims[2]; w++) {
        // Prepare right half of sliding window based at c = 0
        float sum = 0.0;
        for (dim_t i = 0; i < MIN(halfWindow, outWdims[3]); i++) {
          float outg = outG[libjit_getXYZW(outWdims, n, h, w, i)];
          float outw = outW[libjit_getXYZW(outWdims, n, h, w, i)];
          float scale = scaleCache[libjit_getXYZW(outWdims, n, h, w, i)];
          sum += outg * (outw / scale);
        }

        for (dim_t c = 0; c < outWdims[3]; c++) {
          if (c > halfWindow) {
            dim_t j = c - halfWindow - 1;
            float outg = outG[libjit_getXYZW(outWdims, n, h, w, j)];
            float outw = outW[libjit_getXYZW(outWdims, n, h, w, j)];
            float scale = scaleCache[libjit_getXYZW(outWdims, n, h, w, j)];
            sum -= outg * (outw / scale);
          }

          dim_t j = c + halfWindow;
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

void libjit_max_pool_i8(const int8_t *inW, int8_t *outW, const dim_t *inWdims,
                        const dim_t *outWdims, dim_t *kernelSizes,
                        dim_t *strides, dim_t *pads) {
  libjit_max_pool_generic(inW, outW, inWdims, outWdims, kernelSizes, strides,
                          pads);
}

void libjit_max_pool_f(const float *inW, float *outW, const dim_t *inWdims,
                       const dim_t *outWdims, dim_t *kernelSizes,
                       dim_t *strides, dim_t *pads) {
  libjit_max_pool_generic(inW, outW, inWdims, outWdims, kernelSizes, strides,
                          pads);
}

void libjit_max_pool_argmax_i8_u(const int8_t *inW, int8_t *outW,
                                 int64_t *argmax, const dim_t *inWdims,
                                 const dim_t *outWdims, dim_t *kernels,
                                 dim_t *strides, dim_t *pads) {
  libjit_max_pool_argmax_generic(inW, outW, argmax, inWdims, outWdims, kernels,
                                 strides, pads);
}

void libjit_max_pool_argmax_f_u(const float *inW, float *outW, int64_t *argmax,
                                const dim_t *inWdims, const dim_t *outWdims,
                                dim_t *kernels, dim_t *strides, dim_t *pads) {
  libjit_max_pool_argmax_generic(inW, outW, argmax, inWdims, outWdims, kernels,
                                 strides, pads);
}

void libjit_max_pool_argmax_i8_i32(const int8_t *inW, int8_t *outW,
                                   int32_t *argmax, const dim_t *inWdims,
                                   const dim_t *outWdims, dim_t *kernels,
                                   dim_t *strides, dim_t *pads) {
  libjit_max_pool_argmax_generic(inW, outW, argmax, inWdims, outWdims, kernels,
                                 strides, pads);
}

void libjit_max_pool_argmax_f_i32(const float *inW, float *outW,
                                  int32_t *argmax, const dim_t *inWdims,
                                  const dim_t *outWdims, dim_t *kernels,
                                  dim_t *strides, dim_t *pads) {
  libjit_max_pool_argmax_generic(inW, outW, argmax, inWdims, outWdims, kernels,
                                 strides, pads);
}

void libjit_arg_max_i8_u(const int8_t *inW, int64_t *outW, const dim_t *inWdims,
                         size_t axis) {
  libjit_arg_max_generic(inW, outW, inWdims, axis);
}

void libjit_arg_max_i8_i32(const int8_t *inW, int32_t *outW,
                           const dim_t *inWdims, size_t axis) {
  libjit_arg_max_generic(inW, outW, inWdims, axis);
}

void libjit_arg_max_f_u(const float *inW, int64_t *outW, const dim_t *inWdims,
                        size_t axis) {
  libjit_arg_max_generic(inW, outW, inWdims, axis);
}

void libjit_arg_max_f_i32(const float *inW, int32_t *outW, const dim_t *inWdims,
                          size_t axis) {
  libjit_arg_max_generic(inW, outW, inWdims, axis);
}

void libjit_max_pool_argmax_grad_f_u(float *inG, const float *outG,
                                     const int64_t *argmax,
                                     const dim_t *inGdims,
                                     const dim_t *outWdims) {
  libjit_max_pool_argmax_grad_generic(inG, outG, argmax, inGdims, outWdims);
}

void libjit_max_pool_argmax_grad_f_i32(float *inG, const float *outG,
                                       const int32_t *argmax,
                                       const dim_t *inGdims,
                                       const dim_t *outWdims) {
  libjit_max_pool_argmax_grad_generic(inG, outG, argmax, inGdims, outWdims);
}

void libjit_avg_pool_i8(const int8_t *inW, int8_t *outW, const dim_t *inWdims,
                        const dim_t *outWdims, dim_t *kernelSizes,
                        dim_t *strides, dim_t *pads, int32_t outOffset,
                        int32_t inOffset, int32_t outPre, int32_t outPost,
                        int32_t outScale) {
  dim_t pad_t = pads[0];
  dim_t pad_l = pads[1];
  dim_t stride_h = strides[0];
  dim_t stride_w = strides[1];
  dim_t kernel_h = kernelSizes[0];
  dim_t kernel_w = kernelSizes[1];
  // For each input in the batch:
  for (dim_t n = 0; n < outWdims[0]; n++) {
    // For each (x,y) step in the input/output tensor:
    sdim_t x = -sdim_t(pad_t);
    for (dim_t ax = 0; ax < outWdims[1]; x += stride_h, ax++) {
      sdim_t y = -sdim_t(pad_l);
      for (dim_t ay = 0; ay < outWdims[2]; y += stride_w, ay++) {
        // For each layer in the output tensor:
        for (dim_t z = 0; z < inWdims[3]; z++) {
          int32_t sum = 0;

          for (dim_t fx = 0; fx < kernel_h; fx++) {
            for (dim_t fy = 0; fy < kernel_w; fy++) {
              sdim_t ox = x + fx;
              sdim_t oy = y + fy;

              // Ignore index access below zero (this is due to padding).
              if (ox < 0 || oy < 0 || ox >= (sdim_t)inWdims[1] ||
                  oy >= (sdim_t)inWdims[2]) {
                continue;
              }
              sum += inW[libjit_getXYZW(inWdims, n, (dim_t)ox, (dim_t)oy, z)] -
                     inOffset;
            }
          }

          outW[libjit_getXYZW(outWdims, n, ax, ay, z)] = libjit_clip(
              libjit_scale_i32i8(sum, outPre, outPost, outScale, outOffset));
        } // C
      }   // W
    }     // H
  }       // N
}

void libjit_avg_pool_f(const float *inW, float *outW, const dim_t *inWdims,
                       const dim_t *outWdims, dim_t *kernelSizes,
                       dim_t *strides, dim_t *pads) {
  dim_t pad_t = pads[0];
  dim_t pad_l = pads[1];
  dim_t stride_h = strides[0];
  dim_t stride_w = strides[1];
  dim_t kernel_h = kernelSizes[0];
  dim_t kernel_w = kernelSizes[1];
  float filterArea = kernel_h * kernel_w;
  // For each input in the batch:
  for (dim_t n = 0; n < outWdims[0]; n++) {
    // For each (x,y) step in the input/output tensor:
    sdim_t x = -(sdim_t)pad_t;
    for (dim_t ax = 0; ax < outWdims[1]; x += stride_h, ax++) {
      sdim_t y = -(sdim_t)pad_l;
      for (dim_t ay = 0; ay < outWdims[2]; y += stride_w, ay++) {
        // For each layer in the output tensor:
        for (dim_t z = 0; z < inWdims[3]; z++) {

          float sum = 0;

          for (dim_t fx = 0; fx < kernel_h; fx++) {
            for (dim_t fy = 0; fy < kernel_w; fy++) {
              sdim_t ox = x + fx;
              sdim_t oy = y + fy;

              // Ignore index access below zero (this is due to padding).
              if (ox < 0 || oy < 0 || ox >= (sdim_t)inWdims[1] ||
                  oy >= (sdim_t)inWdims[2]) {
                continue;
              }

              sum += inW[libjit_getXYZW(inWdims, n, (dim_t)ox, (dim_t)oy, z)];
            }
          }

          outW[libjit_getXYZW(outWdims, n, ax, ay, z)] = sum / filterArea;
        } // C
      }   // W
    }     // H
  }       // N
}

void libjit_adaptive_avg_pool_f(const float *inW, float *outW,
                                const dim_t *inWdims, const dim_t *outWdims) {
// https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/AdaptiveAveragePooling.cpp
#define START_IND(a, b, c) (size_t) std::floor((float)((a) * (c)) / (b))
#define END_IND(a, b, c) (size_t) std::ceil((float)(((a) + 1) * (c)) / (b))

  // For each input in the batch:
  for (dim_t n = 0; n < outWdims[0]; n++) {
    // For each layer in the output tensor:
    for (dim_t z = 0; z < inWdims[3]; z++) {
      // For each value in the output tensor:
      for (dim_t ax = 0; ax < outWdims[1]; ax++) {

        dim_t x = START_IND(ax, outWdims[1], inWdims[1]);
        dim_t kH = END_IND(ax, outWdims[1], inWdims[1]) - x;

        for (dim_t ay = 0; ay < outWdims[2]; ay++) {

          dim_t y = START_IND(ay, outWdims[2], inWdims[2]);
          dim_t kW = END_IND(ay, outWdims[2], inWdims[2]) - y;

          float sum = 0;
          for (dim_t fx = 0; fx < kH; fx++) {
            for (dim_t fy = 0; fy < kW; fy++) {
              dim_t ox = x + fx;
              dim_t oy = y + fy;

              sum += inW[libjit_getXYZW(inWdims, n, ox, oy, z)];
            }
          }
          outW[libjit_getXYZW(outWdims, n, ax, ay, z)] = (sum / kW / kH);
        } // W
      }   // H
    }     // C
  }       // N
#undef START_IND
#undef END_IND
}

void libjit_avg_pool_grad_f(float *inG, const float *outG, const dim_t *inGdims,
                            const dim_t *outWdims, dim_t *kernels,
                            dim_t *strides, dim_t *pads) {
  dim_t pad_t = pads[0];
  dim_t pad_l = pads[1];
  dim_t stride_h = strides[0];
  dim_t stride_w = strides[1];
  dim_t kernel_h = kernels[0];
  dim_t kernel_w = kernels[1];
  float kernelArea = kernel_h * kernel_w;

  // NHWC format is assumed
  for (dim_t n = 0; n < outWdims[0]; n++) {
    for (dim_t z = 0; z < outWdims[3]; z++) {
      // Clear inG
      for (dim_t x = 0; x < inGdims[1]; x++) {
        for (dim_t y = 0; y < inGdims[2]; y++) {
          inG[libjit_getXYZW(inGdims, n, x, y, z)] = 0.0;
        }
      }

      sdim_t x = -(sdim_t)pad_t;
      for (dim_t ax = 0; ax < outWdims[1]; x += stride_h, ax++) {
        sdim_t y = -(sdim_t)pad_l;
        for (dim_t ay = 0; ay < outWdims[2]; y += stride_w, ay++) {
          float df = outG[libjit_getXYZW(outWdims, n, ax, ay, z)] / kernelArea;
          for (dim_t kx = 0; kx < kernel_h; kx++) {
            for (dim_t ky = 0; ky < kernel_w; ky++) {
              sdim_t ox = x + kx;
              sdim_t oy = y + ky;
              if (ox < 0 || oy < 0 || ox >= (sdim_t)inGdims[1] ||
                  oy >= (sdim_t)inGdims[2]) {
                continue;
              }
              inG[libjit_getXYZW(inGdims, n, (dim_t)ox, (dim_t)oy, z)] += df;
            }
          }
        } // W
      }   // H
    }     // C
  }       // N
}

int8_t libjit_element_quantize_kernel_i8(dim_t idx, const float *inW,
                                         float scale, int32_t offset) {
  int32_t result = (int32_t)nearbyintf(inW[idx] / scale + offset);
  return (int8_t)MAX(INT8_MIN, MIN(INT8_MAX, result));
}

int32_t libjit_element_quantize_kernel_i32(dim_t idx, const float *inW,
                                           float scale, int32_t offset) {
  int32_t result = (int32_t)nearbyintf(inW[idx] / scale + offset);
  return result;
}

float libjit_element_dequantize_kernel_f(dim_t idx, const int8_t *inW,
                                         float scale, int32_t offset) {
  return scale * (inW[idx] - offset);
}

int8_t libjit_element_rescale_kernel_i8(dim_t idx, const int8_t *inW,
                                        int32_t outOffset, int32_t inOffset,
                                        int32_t pre, int32_t post,
                                        int32_t scale) {
  int32_t s =
      libjit_scale_i32i8(inW[idx] - inOffset, pre, post, scale, outOffset);
  return libjit_clip(s);
}

void libjit_softmax_f(const float *inW, float *outW, const dim_t *idim,
                      const dim_t *odim) {
  for (dim_t n = 0; n < idim[0]; n++) {
    float max = inW[libjit_getXY(idim, n, 0)];

    // Find Max.
    for (dim_t i = 1; i < idim[1]; i++) {
      max = MAX(max, inW[libjit_getXY(idim, n, i)]);
    }

    float sum = 0;

    // Compute exp.
    for (dim_t i = 0; i < idim[1]; i++) {
      float e = expf(inW[libjit_getXY(idim, n, i)] - max);
      sum += e;
      outW[libjit_getXY(odim, n, i)] = e;
    }

    // Normalize the output.
    for (dim_t i = 0; i < idim[1]; i++) {
      outW[libjit_getXY(odim, n, i)] = outW[libjit_getXY(odim, n, i)] / sum;
    }
  } // N
}

void libjit_softmax_grad_f_u(float *inG, float *outW, const size_t *selectedW,
                             const dim_t *idim, const dim_t *selectdim) {
  libjit_softmax_grad_generic(inG, outW, selectedW, idim, selectdim);
}

void libjit_softmax_grad_f_i32(float *inG, float *outW,
                               const int32_t *selectedW, const dim_t *idim,
                               const dim_t *selectdim) {
  libjit_softmax_grad_generic(inG, outW, selectedW, idim, selectdim);
}

void libjit_sigmoid_f(const float *inW, float *outW, dim_t numElem) {
  for (dim_t i = 0; i < numElem; i++) {
    float e = expf(-inW[i]);
    outW[i] = 1 / (e + 1);
  }
}

void libjit_topk_f_u(float *values, size_t *indices, const float *input,
                     size_t *scratch, dim_t k, dim_t n, dim_t size) {
  libjit_topk(values, indices, input, scratch, k, n, size);
}

void libjit_topk_f_i32(float *values, int32_t *indices, const float *input,
                       int32_t *scratch, dim_t k, dim_t n, dim_t size) {
  libjit_topk(values, indices, input, scratch, k, n, size);
}

void libjit_topk_i8_u(int8_t *values, size_t *indices, const int8_t *input,
                      size_t *scratch, dim_t k, dim_t n, dim_t size) {
  libjit_topk(values, indices, input, scratch, k, n, size);
}

void libjit_topk_i8_i32(int8_t *values, int32_t *indices, const int8_t *input,
                        int32_t *scratch, dim_t k, dim_t n, dim_t size) {
  libjit_topk(values, indices, input, scratch, k, n, size);
}

void libjit_transpose_i8(const int8_t *inW, int8_t *outW, const dim_t *idim,
                         const dim_t *odim, const dim_t *shuffle,
                         dim_t numDims) {
  libjit_transpose_generic(inW, outW, idim, odim, shuffle, numDims);
}

void libjit_transpose_f(const float *inW, float *outW, const dim_t *idim,
                        const dim_t *odim, const dim_t *shuffle,
                        dim_t numDims) {
  libjit_transpose_generic(inW, outW, idim, odim, shuffle, numDims);
}

void libjit_transpose_u(const int64_t *inW, int64_t *outW, const dim_t *idim,
                        const dim_t *odim, const dim_t *shuffle,
                        dim_t numDims) {
  libjit_transpose_generic(inW, outW, idim, odim, shuffle, numDims);
}

void libjit_transpose_b(const bool *inW, bool *outW, const dim_t *idim,
                        const dim_t *odim, const dim_t *shuffle,
                        dim_t numDims) {
  libjit_transpose_generic(inW, outW, idim, odim, shuffle, numDims);
}

void libjit_flip_i8(const int8_t *inW, int8_t *outW, const dim_t *dims,
                    dim_t axis, dim_t numDims) {
  libjit_flip_generic(inW, outW, dims, axis, numDims);
}

void libjit_flip_i16(const int16_t *inW, int16_t *outW, const dim_t *dims,
                     dim_t axis, dim_t numDims) {
  libjit_flip_generic(inW, outW, dims, axis, numDims);
}

void libjit_flip_i32(const int32_t *inW, int32_t *outW, const dim_t *dims,
                     dim_t axis, dim_t numDims) {
  libjit_flip_generic(inW, outW, dims, axis, numDims);
}

void libjit_flip_u(const int64_t *inW, int64_t *outW, const dim_t *dims,
                   dim_t axis, dim_t numDims) {
  libjit_flip_generic(inW, outW, dims, axis, numDims);
}

void libjit_flip_f(const float *inW, float *outW, const dim_t *dims, dim_t axis,
                   dim_t numDims) {
  libjit_flip_generic(inW, outW, dims, axis, numDims);
}

void libjit_flip_b(const bool *inW, bool *outW, const dim_t *dims, dim_t axis,
                   dim_t numDims) {
  libjit_flip_generic(inW, outW, dims, axis, numDims);
}

void libjit_insert_tensor_f(float *tensor, float *slice, dim_t *offset,
                            dim_t *tensorDim, dim_t *sliceDim,
                            dim_t numDimsTensor, dim_t numDimsSlice,
                            dim_t offsetDim, dim_t count, dim_t axis) {
  libjit_insert_tensor(tensor, slice, offset, tensorDim, sliceDim,
                       numDimsTensor, numDimsSlice, offsetDim, count, axis);
}

void libjit_insert_tensor_i32(int32_t *tensor, int32_t *slice, dim_t *offset,
                              dim_t *tensorDim, dim_t *sliceDim,
                              dim_t numDimsTensor, dim_t numDimsSlice,
                              dim_t offsetDim, dim_t count, dim_t axis) {
  libjit_insert_tensor(tensor, slice, offset, tensorDim, sliceDim,
                       numDimsTensor, numDimsSlice, offsetDim, count, axis);
}

void libjit_extract_tensor_f(float *tensor, float *slice, dim_t *offset,
                             dim_t *tensorDim, dim_t *sliceDim,
                             dim_t numDimsTensor, dim_t numDimsSlice,
                             dim_t offsetDim) {
  libjit_extract_tensor(tensor, slice, offset, tensorDim, sliceDim,
                        numDimsTensor, numDimsSlice, offsetDim);
}

void libjit_extract_tensor_i8(int8_t *tensor, int8_t *slice, dim_t *offset,
                              dim_t *tensorDim, dim_t *sliceDim,
                              dim_t numDimsTensor, dim_t numDimsSlice,
                              dim_t offsetDim) {
  libjit_extract_tensor(tensor, slice, offset, tensorDim, sliceDim,
                        numDimsTensor, numDimsSlice, offsetDim);
}

void libjit_extract_tensor_i32(int32_t *tensor, int32_t *slice, dim_t *offset,
                               dim_t *tensorDim, dim_t *sliceDim,
                               dim_t numDimsTensor, dim_t numDimsSlice,
                               dim_t offsetDim) {
  libjit_extract_tensor(tensor, slice, offset, tensorDim, sliceDim,
                        numDimsTensor, numDimsSlice, offsetDim);
}

void libjit_insert_tensor_u(int64_t *tensor, int64_t *slice, dim_t *offset,
                            dim_t *tensorDim, dim_t *sliceDim,
                            dim_t numDimsTensor, dim_t numDimsSlice,
                            dim_t offsetDim, dim_t count, dim_t axis) {
  libjit_insert_tensor(tensor, slice, offset, tensorDim, sliceDim,
                       numDimsTensor, numDimsSlice, offsetDim, count, axis);
}

void libjit_extract_tensor_u(int64_t *tensor, int64_t *slice, dim_t *offset,
                             dim_t *tensorDim, dim_t *sliceDim,
                             dim_t numDimsTensor, dim_t numDimsSlice,
                             dim_t offsetDim) {
  libjit_extract_tensor(tensor, slice, offset, tensorDim, sliceDim,
                        numDimsTensor, numDimsSlice, offsetDim);
}

void libjit_insert_tensor_i8(int8_t *tensor, int8_t *slice, dim_t *offset,
                             dim_t *tensorDim, dim_t *sliceDim,
                             dim_t numDimsTensor, dim_t numDimsSlice,
                             dim_t offsetDim, dim_t count, dim_t axis) {
  libjit_insert_tensor(tensor, slice, offset, tensorDim, sliceDim,
                       numDimsTensor, numDimsSlice, offsetDim, count, axis);
}

void libjit_insert_tensor_b(int8_t *tensor, int8_t *slice, dim_t *offset,
                            dim_t *tensorDim, dim_t *sliceDim,
                            dim_t numDimsTensor, dim_t numDimsSlice,
                            dim_t offsetDim, dim_t count, dim_t axis) {
  libjit_insert_tensor(tensor, slice, offset, tensorDim, sliceDim,
                       numDimsTensor, numDimsSlice, offsetDim, count, axis);
}

void libjit_space_to_depth_f(const float *inTensor, float *outTensor,
                             dim_t blockSize, const dim_t *inDims,
                             const dim_t *outDims) {
  libjit_space_to_depth_generic(inTensor, outTensor, blockSize, inDims,
                                outDims);
}

void libjit_space_to_depth_i8(const int8_t *inTensor, int8_t *outTensor,
                              dim_t blockSize, const dim_t *inDims,
                              const dim_t *outDims) {
  libjit_space_to_depth_generic(inTensor, outTensor, blockSize, inDims,
                                outDims);
}
__attribute__((noinline)) void
libjit_dump_tensor(uint8_t *tensor, dim_t *tensorDim, dim_t numDimsTensor,
                   dim_t elemKind, const char *name) {
  printf("%s\n", name);
  /// This definition should match the defintion in Glow.
  enum class ElemKind : unsigned char {
    FloatTy,       // 32-bit float type (float)
    Float16Ty,     // 16-bit float type (half, fp16)
    Int8QTy,       // 8-bit quantized type (int8_t)
    UInt8QTy,      // unsigned 8-bit quantized type (uint8_t)
    Int16QTy,      // 16-bit quantized type (int16_t)
    Int32QTy,      // 32-bit quantized type (int32_t)
    Int32ITy,      // 32-bit index type (int32_t)
    Int64ITy,      // 64-bit index type (int64_t)
    UInt8FusedQTy, // 8-bit quantized type with fused scale/offset (uint8_t)
    BoolTy,        // Bool type (bool)
  };
  // Dump the content of a tensor.
  switch ((ElemKind)elemKind) {
  case ElemKind::FloatTy:
    libjit_dump_tensor_impl((float *)tensor, tensorDim, numDimsTensor);
    break;
  case ElemKind::Int64ITy:
    libjit_dump_tensor_impl((dim_t *)tensor, tensorDim, numDimsTensor);
    break;
  case ElemKind::Int8QTy:
    libjit_dump_tensor_impl((int8_t *)tensor, tensorDim, numDimsTensor);
    break;
  case ElemKind::Int32QTy:
    libjit_dump_tensor_impl((int32_t *)tensor, tensorDim, numDimsTensor);
    break;
  default:
    printf("Dumping this type of payload is not supported: %zu\n",
           (size_t)elemKind);
    break;
  }
  puts("");
}

void libjit_write_timestamp(uint64_t *tensor, dim_t offset) {
  // We are using C++ timer here to a avoid issues with gettimeofday
  // Issue #2397 covers migrating this to a libc approach but if you have issues
  // with a lack of C++ symbols at runtime check there first.
  uint64_t ts = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now().time_since_epoch())
                    .count();
  memcpy(tensor + offset, &ts, sizeof(uint64_t));
}

/// Copies a kernel with type conversion
void libjit_convertTo_f_i32(float *dstPtr, const int32_t *srcPtr,
                            const dim_t *dims, dim_t numDims) {
  libjit_copy_kernel_with_conversion<float, int32_t>(dstPtr, srcPtr, dims,
                                                     numDims);
}

void libjit_convertTo_i32_u(int32_t *dstPtr, const int64_t *srcPtr,
                            const dim_t *dims, dim_t numDims) {
  libjit_copy_kernel_with_conversion<int32_t, int64_t>(dstPtr, srcPtr, dims,
                                                       numDims);
}

void libjit_convertTo_u_i32(int64_t *dstPtr, const int32_t *srcPtr,
                            const dim_t *dims, dim_t numDims) {
  libjit_copy_kernel_with_conversion<int64_t, int32_t>(dstPtr, srcPtr, dims,
                                                       numDims);
}

/// Update min/max values \p compInfo and histogram \p existingHistogram with
/// data collected from tensor \p inputTensor.
/// Note: code ported from Profile.cpp: generateTensorHistogram
__attribute__((noinline)) void
libjit_quantization_profile(float *inputTensor, dim_t tensorSize,
                            float *compInfo, float *existingHistogram,
                            dim_t *histDim) {
  dim_t nBins = histDim[0];

  // Min/max computed from previous runs. If this is the first run, compInfo is
  // expected to be initialized as following:
  // compInfo[0]: std::numeric_limits<float>::max()
  // compInfo[1]: std::numeric_limits<float>::lowest()
  float min = compInfo[0];
  float max = compInfo[1];

  // Min/max value for entire current input tensor.
  float minInput;
  float maxInput;
  find_min_max_f(inputTensor, tensorSize, minInput, maxInput);

  // Update the global min/max.
  float newMin = MIN(minInput, min);
  float newMax = MAX(maxInput, max);
  compInfo[0] = newMin;
  compInfo[1] = newMax;

  // Initial profile.
  if (check_all_zeros(existingHistogram, nBins) == 1) {
    min = minInput;
    max = maxInput;
  }

  // If the min/max range changes, there is the need to rescale the histogram.
  if (newMin < min || newMax > max) {
    float destBinWidth = (newMax - newMin) / nBins;
    float srcBinWidth = (max - min) / nBins;
    float scaledHistogram[nBins];
    for (dim_t i = 0; i < nBins; ++i) {
      scaledHistogram[i] = 0.0f;
    }

    for (dim_t i = 0; i < nBins; ++i) {
      if (existingHistogram[i] == 0)
        continue;

      float srcBinBegin = min + srcBinWidth * i;
      dim_t destBin = (srcBinBegin - newMin) / destBinWidth;
      float destBinEnd = newMin + destBinWidth * (destBin + 1);

      float srcBinEnd = srcBinBegin + srcBinWidth;
      dim_t destBinToVerify = (srcBinEnd - newMin) / destBinWidth;
      // Make sure that destination bin is mapped at most to 2 final bins, based
      // on that redistribute percentage is calculated.
      assert(destBinToVerify <= destBin + 2);
      (void)destBinToVerify;

      // Calculate how much we need to redistribute.
      uint64_t dstBinCnt = static_cast<uint64_t>(
          MIN(static_cast<float>(round((destBinEnd - srcBinBegin) /
                                       srcBinWidth * existingHistogram[i])),
              existingHistogram[i]));

      dim_t newBin = get_bin(nBins, destBinWidth, newMin, srcBinBegin);
      scaledHistogram[newBin] += dstBinCnt;

      if (dstBinCnt < existingHistogram[i]) {
        dim_t newBin =
            get_bin(nBins, destBinWidth, newMin, srcBinBegin + destBinWidth);
        scaledHistogram[newBin] += existingHistogram[i] - dstBinCnt;
      }
    }

    // Copy scaled histogram back to the existing histogram.
    for (dim_t i = 0, e = nBins; i < e; ++i) {
      existingHistogram[i] = scaledHistogram[i];
    }

    // Update global min and max.
    min = newMin;
    max = newMax;
  }

  // Update the histogram with the values of the current input tensor.
  float binWidth = (max - min) / nBins;
  for (dim_t i = 0, e = tensorSize; i < e; ++i) {
    dim_t newBin = get_bin(nBins, binWidth, min, inputTensor[i]);
    existingHistogram[newBin]++;
  }
}

__attribute__((noinline)) void
libjit_nms_u(size_t *indices, size_t *numDetected, const float *boxTensor,
             const dim_t *boxTensorDims, size_t boxTensorDimSize,
             const float *scoresTensor, const dim_t *scoresTensorDims,
             size_t scoresTensorDimSize, const dim_t *resultTensorDims,
             size_t resultTensorDimSize, unsigned centerPointBox,
             unsigned maxOutputBoxesPerClass, float iouThreshold,
             float scoreThreshold, bool isV4) {
  libjit_nms_generic(indices, numDetected, boxTensor, boxTensorDims,
                     boxTensorDimSize, scoresTensor, scoresTensorDims,
                     scoresTensorDimSize, resultTensorDims, resultTensorDimSize,
                     centerPointBox, maxOutputBoxesPerClass, iouThreshold,
                     scoreThreshold, isV4);
}

__attribute__((noinline)) void
libjit_nms_i32(int32_t *indices, int32_t *numDetected, const float *boxTensor,
               const dim_t *boxTensorDims, size_t boxTensorDimSize,
               const float *scoresTensor, const dim_t *scoresTensorDims,
               size_t scoresTensorDimSize, const dim_t *resultTensorDims,
               size_t resultTensorDimSize, unsigned centerPointBox,
               unsigned maxOutputBoxesPerClass, float iouThreshold,
               float scoreThreshold, bool isV4) {
  libjit_nms_generic(indices, numDetected, boxTensor, boxTensorDims,
                     boxTensorDimSize, scoresTensor, scoresTensorDims,
                     scoresTensorDimSize, resultTensorDims, resultTensorDimSize,
                     centerPointBox, maxOutputBoxesPerClass, iouThreshold,
                     scoreThreshold, isV4);
}
} // extern "C"
