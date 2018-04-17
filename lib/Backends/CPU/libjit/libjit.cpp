#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/types.h>

#include "libjit_defs.h"

namespace {

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
  for (size_t i = 0; i < numDims; ++i) {
    printf("%zu ", dims[i]);
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
static size_t get_element_ptr(const ElemTy *tensor, const size_t *dims,
                              size_t numDims, const size_t *indices,
                              size_t numIndices) {
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

/// Helper struct for TopK
template <typename T> struct value_index {
  size_t index;
  T value;
};

/// Helper function for TopK
template <typename T> int value_index_sort(const void *va, const void *vb) {
  value_index<T> *a = (value_index<T> *)va;
  value_index<T> *b = (value_index<T> *)vb;
  if (a->value != b->value)
    return a->value > b->value ? -1 : 1;
  return a->index < b->index ? -1 : 1;
}

/// Generic Top-K function. Here, \p scratch is some allocated buffer space, \p
/// size is the size of the input, and \p n is the size of the last dimension of
/// the input.
template <typename T>
void libjit_topk(T *values, size_t *indices, const T *input, size_t *scratch,
                 size_t k, size_t n, size_t size) {
  size_t in = 0;
  size_t out = 0;

  value_index<T> *buffer = (value_index<T> *)scratch;
  while (in < size) {
    for (size_t i = 0; i < n; i++) {
      buffer[i].index = i;
      buffer[i].value = input[in++];
    }
    qsort(buffer, n, sizeof(value_index<T>), value_index_sort<T>);
    for (size_t i = 0; i < k; i++) {
      indices[out] = buffer[i].index;
      values[out] = buffer[i].value;
      out++;
    }
  }
}

/// Helper function for Broadcast. Increments an "index" dimension vector, \p
/// dest_i, with respect to \p dest_dims.  This corresponds to striding through
/// all \p n dimensions using \p n nested for-loops.
///
/// \returns false when the index equals the dimension.
bool increment_and_check_dims(size_t *dest_i, const size_t *dest_dims,
                              size_t n) {
  for (size_t i = 0; i < n; i++) {
    dest_i[i] += 1;
    if (dest_i[i] == dest_dims[i]) {
      dest_i[i] = 0;
    } else {
      return true;
    }
  }
  return false;
}

/// Helper function for Broadcast. Given a destination index \p dest_i of a
/// broadcast operation, compute the source index \p src_i given a source tensor
/// with dimensions \p src_dims.
///
/// Any source dimension containing a 1 is broadcast to all other dimensions by
/// selecting index 0 in that dimension.
void get_src_dim(size_t *src_i, const size_t *dest_i, const size_t *src_dims,
                 size_t n) {
  for (size_t i = 0; i < n; i++) {
    src_i[i] = (src_dims[i] == 1) ? 0 : dest_i[i];
  }
}

template <typename T>
void libjit_gather(T *dest, const T *data, const size_t *indices,
                   size_t numIndices, size_t sliceSize) {
  for (size_t i = 0; i < numIndices; i++) {
    size_t slice = indices[i];
    memcpy(dest + i * sliceSize, data + slice * sliceSize,
           sliceSize * sizeof(T));
  }
}

template <typename T>
void libjit_transpose_generic(const T *inW, T *outW, const size_t *idim,
                              const size_t *odim, const size_t *shuffle,
                              size_t numDims) {
  // Source coordinate.
  size_t SC[5];

  if (numDims == 5) {
    for (size_t x = 0; x < odim[0]; x++)
      for (size_t y = 0; y < odim[1]; y++)
        for (size_t z = 0; z < odim[2]; z++)
          for (size_t w = 0; w < odim[3]; w++)
            for (size_t q = 0; q < odim[4]; q++) {
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

template <typename T>
void libjit_pool_max_generic(const T *inW, T *outW, const size_t *inWdims,
                             const size_t *outWdims, size_t filterSize,
                             size_t stride, size_t pad) {
  // For each sample in the batch:
  for (size_t n = 0; n < outWdims[0]; n++) {
    // For each (x,y) step in the input/output tensor:
    ssize_t x = -(ssize_t)pad;
    for (size_t ax = 0; ax < outWdims[1]; x += stride, ax++) {
      ssize_t y = -(ssize_t)pad;
      for (size_t ay = 0; ay < outWdims[2]; y += stride, ay++) {

        // For each layer in the output tensor:
        for (size_t z = 0; z < inWdims[3]; z++) {
          int first = 1;
          T max = 0;

          // For each element in the convolution-filter:
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
        } // C
      }   // W
    }     // H
  }       // N
}

template <typename T>
void libjit_pool_max_xy_generic(const T *inW, T *outW, size_t *inXY,
                                const size_t *inWdims, const size_t *outWdims,
                                size_t kernel, size_t stride, size_t pad) {
  // For each input in the batch:
  for (size_t n = 0; n < outWdims[0]; n++) {

    // For each (x,y) step in the input/output tensor:
    ssize_t x = -(ssize_t)pad;
    for (size_t ax = 0; ax < outWdims[1]; x += stride, ax++) {
      ssize_t y = -(ssize_t)pad;
      for (size_t ay = 0; ay < outWdims[2]; y += stride, ay++) {

        // For each channel in the output tensor:
        for (size_t z = 0; z < outWdims[3]; z++) {
          size_t maxX = x;
          size_t maxY = y;
          int first = 1;
          T max = 0;

          for (size_t kx = 0; kx < kernel; kx++) {
            for (size_t ky = 0; ky < kernel; ky++) {
              ssize_t ox = x + kx;
              ssize_t oy = y + ky;

              if (ox < 0 || oy < 0 || ox >= (ssize_t)inWdims[1] ||
                  oy >= (ssize_t)inWdims[2]) {
                continue;
              }

              T val =
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
        } // C
      }   // W
    }     // H
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
  type name(size_t idx, const type *LHS, const type *RHS, const type *op3) {   \
    return body;                                                               \
  }

/// Macro to define a mini-kernel for data-parallel operations. The body of the
/// kernel is not auto-generated by the macro.
/// \p name the name of the kernel
#define DEFINE_DATA_PARALLEL_KERNEL_FUNC(name)                                 \
  float name(size_t idx, const float *LHS, const float *RHS, const float *op3)

/// Macro to define a mini-kernel for data-parallel operations with immediate
/// operands.
/// \p name the name of the kernel
/// \p type the type of the tensor elements and of the return value
/// \p body the operation to be performed
#define DEFINE_DATA_PARALLEL_KERNEL_WITH_IMM_OPERAND(name, type, body)         \
  type name(size_t idx, type val, const type *LHS, const type *RHS) {          \
    return body;                                                               \
  }

/// Macro to define a mini-kernel for data-parallel arithmetic quantized
/// operations. The body of the kernel is auto-generated by the macro.
/// \p name the name of the kernel
/// \p type the type of the tensor elements
/// \p body the operation to be performed
#define DEFINE_DATA_PARALLEL_KERNEL_QUANTIZED(name, type, body)                \
  type name(size_t idx, const type *LHS, const type *RHS, int32_t destOffset,  \
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
  int8_t name(size_t idx, const int8_t *LHS, const int8_t *RHS,                \
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
DEFINE_DATA_PARALLEL_KERNEL(libjit_copy_kernel_u, size_t, LHS[idx])
DEFINE_DATA_PARALLEL_KERNEL(libjit_copy_kernel_i8, int8_t, LHS[idx])
DEFINE_DATA_PARALLEL_KERNEL(libjit_element_cmp_lte_kernel_f, float,
                            LHS[idx] <= RHS[idx] ? 1.0 : 0.0)
DEFINE_DATA_PARALLEL_KERNEL(libjit_element_add_kernel_f, float,
                            LHS[idx] + RHS[idx])
DEFINE_DATA_PARALLEL_KERNEL(libjit_element_sub_kernel_f, float,
                            LHS[idx] - RHS[idx])
DEFINE_DATA_PARALLEL_KERNEL(libjit_element_div_kernel_f, float,
                            LHS[idx] / RHS[idx])
DEFINE_DATA_PARALLEL_KERNEL(libjit_element_mul_kernel_f, float,
                            LHS[idx] * RHS[idx])
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

int8_t libjit_element_cmp_lte_kernel_i8(size_t idx, const int8_t *LHS,
                                        const int8_t *RHS, int32_t lhsOffset,
                                        int32_t rhsOffset, int32_t pre,
                                        int32_t post, int32_t scale) {
  int32_t lhs = LHS[idx] - lhsOffset;
  int32_t rhs = RHS[idx] - rhsOffset;
  return libjit_scale_i32i8(lhs, pre, post, scale, 0) <= rhs ? 1 : 0;
}

// tanh cannot be vectorized by LLVM yet. Therefore we use the following
// formula instead: 1 - 2 / (exp(x * 2) + 1), which is also used by Caffe2 and
// provides a good accuracy.
// Once LLVM supports the vectorization of tanh, we can replace this
// approximation by a direct tanh call.
DEFINE_DATA_PARALLEL_KERNEL(libjit_tanh_kernel_f, float,
                            1 - 2 / (expf(LHS[idx] * 2) + 1))
DEFINE_DATA_PARALLEL_KERNEL(libjit_elementselect_kernel_f, float,
                            (LHS[idx] != 0.0) ? RHS[idx] : op3[idx])

int8_t libjit_elementselect_kernel_i8(size_t idx, const int8_t *cond,
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
  float e = expf(LHS[idx]);
  return e / (e + 1);
}
DEFINE_DATA_PARALLEL_KERNEL_WITH_IMM_OPERAND(libjit_element_maxsplat_kernel_f,
                                             float, MAX(LHS[idx], val))
DEFINE_DATA_PARALLEL_KERNEL_WITH_IMM_OPERAND(libjit_element_maxsplat_kernel_i8,
                                             int8_t, MAX(LHS[idx], val))
DEFINE_DATA_PARALLEL_KERNEL_WITH_IMM_OPERAND(libjit_element_pow_kernel_f, float,
                                             pow(LHS[idx], val))
DEFINE_DATA_PARALLEL_KERNEL_WITH_IMM_OPERAND(libjit_splat_kernel_f, float, val)
DEFINE_DATA_PARALLEL_KERNEL_WITH_IMM_OPERAND(libjit_splat_kernel_u, size_t, val)
DEFINE_DATA_PARALLEL_KERNEL_WITH_IMM_OPERAND(libjit_splat_kernel_i8, int8_t,
                                             val)

#undef DEFINE_DATA_PARALLEL_KERNEL
#undef DEFINE_DATA_PARALLEL_KERNEL_FUNC
#undef DEFINE_DATA_PARALLEL_KERNEL_FUNC
#undef DEFINE_DATA_PARALLEL_KERNEL_WITH_IMM_OPERAND

void libjit_broadcast_f(float *dest, const float *src, const size_t *dest_dims,
                        const size_t *src_dims, size_t n_dims) {
  size_t dest_i[6] = {0};
  size_t src_i[6] = {0};
  do {
    get_src_dim(src_i, dest_i, src_dims, n_dims);
    size_t sptr = get_element_ptr(src, src_dims, n_dims, src_i, n_dims);
    size_t dptr = get_element_ptr(dest, dest_dims, n_dims, dest_i, n_dims);
    dest[dptr] = src[sptr];
  } while (increment_and_check_dims(dest_i, dest_dims, n_dims));
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

void libjit_batchedadd_i8(int8_t *dest, const int8_t *batch,
                          const int8_t *slice, size_t numSlice,
                          size_t sliceSize, int32_t destOffset,
                          int32_t batchOffset, int32_t sliceOffset,
                          int32_t batchPre, int32_t batchPost,
                          int32_t batchScale, int32_t slicePre,
                          int32_t slicePost, int32_t sliceScale) {
  for (size_t n = 0; n < numSlice; n++) {
    size_t base = n * sliceSize;
    for (size_t i = 0; i < sliceSize; i++) {
      int32_t b = batch[base + i] - batchOffset;
      int32_t s = slice[i] - sliceOffset;
      int32_t x = libjit_scale_i32i8(b, batchPre, batchPost, batchScale, 0);
      int32_t y = libjit_scale_i32i8(s, slicePre, slicePost, sliceScale, 0);
      dest[base + i] = libjit_clip(x + y + destOffset);
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

void libjit_batchedreduceadd_i8(int8_t *dest, const int8_t *batch,
                                size_t destSize, size_t numSlice,
                                size_t sliceSize, int32_t destOffset,
                                int32_t batchOffset, int32_t batchPre,
                                int32_t batchPost, int32_t batchScale) {
  for (size_t i = 0; i < sliceSize; i++) {
    int32_t sum = 0;
    for (size_t n = 0; n < numSlice; n++) {
      sum += batch[n * sliceSize + i] - batchOffset;
    }
    int32_t q =
        libjit_scale_i32i8(sum, batchPre, batchPost, batchScale, destOffset);
    dest[i] = libjit_clip(q);
  }
}

void libjit_gather_f(float *dest, const float *data, const size_t *indices,
                     size_t numIndices, size_t sliceSize) {
  libjit_gather(dest, data, indices, numIndices, sliceSize);
}

void libjit_gather_i8(int8_t *dest, const int8_t *data, const size_t *indices,
                      size_t numIndices, size_t sliceSize) {
  libjit_gather(dest, data, indices, numIndices, sliceSize);
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

void libjit_pool_max_i8(const int8_t *inW, int8_t *outW, const size_t *inWdims,
                        const size_t *outWdims, size_t filterSize,
                        size_t stride, size_t pad) {
  libjit_pool_max_generic(inW, outW, inWdims, outWdims, filterSize, stride,
                          pad);
}
void libjit_pool_max_f(const float *inW, float *outW, const size_t *inWdims,
                       const size_t *outWdims, size_t filterSize, size_t stride,
                       size_t pad) {
  libjit_pool_max_generic(inW, outW, inWdims, outWdims, filterSize, stride,
                          pad);
}

void libjit_pool_max_xy_i8(const int8_t *inW, int8_t *outW, size_t *inXY,
                           const size_t *inWdims, const size_t *outWdims,
                           size_t kernel, size_t stride, size_t pad) {
  libjit_pool_max_xy_generic(inW, outW, inXY, inWdims, outWdims, kernel, stride,
                             pad);
}

void libjit_pool_max_xy_f(const float *inW, float *outW, size_t *inXY,
                          const size_t *inWdims, const size_t *outWdims,
                          size_t kernel, size_t stride, size_t pad) {
  libjit_pool_max_xy_generic(inW, outW, inXY, inWdims, outWdims, kernel, stride,
                             pad);
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

void libjit_pool_avg_i8(const int8_t *inW, int8_t *outW, const size_t *inWdims,
                        const size_t *outWdims, size_t filterSize,
                        size_t stride, size_t pad, int32_t outOffset,
                        int32_t inOffset, int32_t outPre, int32_t outPost,
                        int32_t outScale) {
  // For each input in the batch:
  for (size_t n = 0; n < outWdims[0]; n++) {
    // For each (x,y) step in the input/output tensor:
    ssize_t x = -ssize_t(pad);
    for (size_t ax = 0; ax < outWdims[1]; x += stride, ax++) {
      ssize_t y = -ssize_t(pad);
      for (size_t ay = 0; ay < outWdims[2]; y += stride, ay++) {
        // For each layer in the output tensor:
        for (size_t z = 0; z < inWdims[3]; z++) {
          int32_t sum = 0;

          for (size_t fx = 0; fx < filterSize; fx++) {
            for (size_t fy = 0; fy < filterSize; fy++) {
              ssize_t ox = x + fx;
              ssize_t oy = y + fy;

              // Ignore index access below zero (this is due to padding).
              if (ox < 0 || oy < 0 || ox >= (ssize_t)inWdims[1] ||
                  oy >= (ssize_t)inWdims[2]) {
                continue;
              }
              sum +=
                  inW[libjit_getXYZW(inWdims, n, (size_t)ox, (size_t)oy, z)] -
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

void libjit_pool_avg_f(const float *inW, float *outW, const size_t *inWdims,
                       const size_t *outWdims, size_t filterSize, size_t stride,
                       size_t pad) {
  float filterArea = filterSize * filterSize;
  // For each input in the batch:
  for (size_t n = 0; n < outWdims[0]; n++) {
    // For each (x,y) step in the input/output tensor:
    ssize_t x = -(ssize_t)pad;
    for (size_t ax = 0; ax < outWdims[1]; x += stride, ax++) {
      ssize_t y = -(ssize_t)pad;
      for (size_t ay = 0; ay < outWdims[2]; y += stride, ay++) {
        // For each layer in the output tensor:
        for (size_t z = 0; z < inWdims[3]; z++) {

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
        } // C
      }   // W
    }     // H
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

void libjit_rescale_i8(int8_t *outW, const int8_t *inW, size_t numElem,
                       int32_t outOffset, int32_t inOffset, int32_t pre,
                       int32_t post, int32_t scale) {
  for (size_t i = 0; i < numElem; i++) {
    int32_t s =
        libjit_scale_i32i8(inW[i] - inOffset, pre, post, scale, outOffset);
    outW[i] = libjit_clip(s);
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

void libjit_topk_f(float *values, size_t *indices, const float *input,
                   size_t *scratch, size_t k, size_t n, size_t size) {
  libjit_topk(values, indices, input, scratch, k, n, size);
}

void libjit_topk_i8(int8_t *values, size_t *indices, const int8_t *input,
                    size_t *scratch, size_t k, size_t n, size_t size) {
  libjit_topk(values, indices, input, scratch, k, n, size);
}

void libjit_transpose_i8(const int8_t *inW, int8_t *outW, const size_t *idim,
                         const size_t *odim, const size_t *shuffle,
                         size_t numDims) {
  libjit_transpose_generic(inW, outW, idim, odim, shuffle, numDims);
}

void libjit_transpose_f(const float *inW, float *outW, const size_t *idim,
                        const size_t *odim, const size_t *shuffle,
                        size_t numDims) {
  libjit_transpose_generic(inW, outW, idim, odim, shuffle, numDims);
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

void libjit_insert_tensor_i8(int8_t *tensor, int8_t *slice, size_t *offset,
                             size_t *tensorDim, size_t *sliceDim,
                             size_t numDimsTensor, size_t numDimsSlice,
                             size_t offsetDim) {
  libjit_insert_tensor(tensor, slice, offset, tensorDim, sliceDim,
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
