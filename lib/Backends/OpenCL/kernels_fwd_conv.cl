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

// Kernels for the forward convolution.

// This is a parameterized convolution kernel heavily based on the kernels
// produced by libDNN (https://github.com/naibaf7/libdnn). The libDNN library
// in turn is inspired by ideas and approaches presented
// in https://cnugteren.github.io/tutorial/pages/page1.html.
//
// The kernel is parameterized by means of macro-definitions, which
// define such propeties like size of the filter, padding, stride,
// sizes of workgroups, etc.
//
// The parameters of the kernel are:
//
// v_nax - Number of spacial axes.
// v_g - Number of groups.
// v_k_0, v_k_1 - dimensions of the kernel
// v_p_0, v_p_1 - padding
// v_s_0, v_s_1 - stride
// v_d_0, v_d_1 - dilation
// v_fin - Number of kernel input channels.
// v_fout - Number of kernel output channels.
// v_bmul - Bias multiplier.
// v_imsi_0, v_imsi_1 - Spacial dimensions of input.
// v_imso_0, v_imso_1 - Spacial dimensions of output.
// v_pad_A - Padding required for tiles of A.
// v_pad_B - Padding required for tiles of A.
// workgroup_size_0, workgroup_size_1 - workgroup sizes along each dimension.
// VWN - vector width in dimension N.
// VWM - vector width in dimension M.

#define Dtype float
#define Dtype1 float
#define Dtype2 float2
#define Dtype4 float4
#define Dtype8 float8
#define Dtype16 float16

#define VEC_1_0(X) X
#define VEC_2_0(X) X.x
#define VEC_2_1(X) X.y
#define VEC_4_0(X) X.x
#define VEC_4_1(X) X.y
#define VEC_4_2(X) X.z
#define VEC_4_3(X) X.w
#define VEC_8_0(X) X.s0
#define VEC_8_1(X) X.s1
#define VEC_8_2(X) X.s2
#define VEC_8_3(X) X.s3
#define VEC_8_4(X) X.s4
#define VEC_8_5(X) X.s5
#define VEC_8_6(X) X.s6
#define VEC_8_7(X) X.s7
#define VEC_16_0(X) X.s0
#define VEC_16_1(X) X.s1
#define VEC_16_2(X) X.s2
#define VEC_16_3(X) X.s3
#define VEC_16_4(X) X.s4
#define VEC_16_5(X) X.s5
#define VEC_16_6(X) X.s6
#define VEC_16_7(X) X.s7
#define VEC_16_8(X) X.s8
#define VEC_16_9(X) X.s9
#define VEC_16_10(X) X.sA
#define VEC_16_11(X) X.sB
#define VEC_16_12(X) X.sC
#define VEC_16_13(X) X.sD
#define VEC_16_14(X) X.sE
#define VEC_16_15(X) X.sF

#define int_tp int
#define uint_tp unsigned int
#define int_tpc int
#define uint_tpc unsigned int

// Input image size in pixels.
#define v_imsi (v_imsi_0 * v_imsi_1)
// Output image size in pixels.
#define v_imso (v_imso_0 * v_imso_1)
// Input image batch offset.
#define v_B_off (v_fin * v_imsi)
// Output image batch offset.
#define v_C_off (v_fout * v_imso)
// Definitions used by the GEMM kernel.
#define MG v_fout
#define MM (v_fout / v_g)
#define NN v_imso
#define KG (v_fin * v_k_0 * v_k_1)
#define KK ((v_fin / v_g) * v_k_0 * v_k_1)
// The tile-size in dimension M.
#define TSM (WPTM * workgroup_size_1)
// The tile-size in dimension N.
#define TSN (WPTN * workgroup_size_0)
// The reduced tile-size in dimension M.
#define RTSM workgroup_size_1
// The reduced tile-size in dimension N.
#define RTSN workgroup_size_0
// Loads per thread for A.
#ifdef LPTA
#undef LPTA
#endif
#define LPTA ((TSK * TSM) / (RTSM * RTSN))
// Loads per thread for B.
#ifdef LPTB
#undef LPTB
#endif
#define LPTB ((TSK * TSN) / (RTSM * RTSN))
#ifdef v_num_tiles
#undef v_num_tiles
#endif
#define v_num_tiles (((KK - 1) / (TSK * 2) + 1) * 2)

__kernel
    __attribute__((reqd_work_group_size(workgroup_size_0, workgroup_size_1, 1)))
    __attribute__((vec_type_hint(Dtype4))) void
    conv_forward_mem(__global void *mem, unsigned im_in_offset,
                     unsigned wg_offset, unsigned bias_offset,
                     unsigned im_out_offset) {
  __global const Dtype *im_in = &mem[im_in_offset];
  __global const Dtype *wg = &mem[wg_offset];
  __global const Dtype *bias = &mem[bias_offset];
  __global Dtype *im_out = &mem[im_out_offset];
  // Thread identifiers.
  // Local row ID (max: RTSM=TSM/WPTM).
  const int_tp tidn = get_local_id(0);
  // Local col ID (max: RTSN=TSN/WPTN).
  const int_tp tidm = get_local_id(1);
  // Work-group offset.
  const int_tp offN = TSN * get_group_id(0);
  // Work-group offset.
  const int_tp offM = TSM * get_group_id(1);
  // Local tile memory.
  // Asub for loading weights & shuffling the output.
  volatile __local Dtype Asub[TSM][TSK + v_pad_A];
  // Bsub for loading the input image and shuffling the output image.
  volatile __local Dtype Bsub[TSK][TSN + v_pad_B];
  int_tp batch = get_global_id(2);
  __global const Dtype *Aptr = wg;
  __global const Dtype *Bptr = im_in + v_B_off * batch;
  __global Dtype *Cptr = im_out + v_C_off * batch;
  __global const Dtype *Dptr = bias;
  // Initialize the accumulation registers.
  {
    Dtype4 Creg[WPTM][WPTN / VWN];
// Initialize the accumulation registers.
#pragma unroll
    for (int_tp wm = 0; wm < WPTM; ++wm) {
#pragma unroll
      for (int_tp wn = 0; wn < WPTN / VWN; ++wn) {
        VEC_4_0(Creg[wm][wn]) = 0.0;
        VEC_4_1(Creg[wm][wn]) = 0.0;
        VEC_4_2(Creg[wm][wn]) = 0.0;
        VEC_4_3(Creg[wm][wn]) = 0.0;
      }
    }
    {
// Loop over all tiles.
#pragma unroll 1
      for (int_tp t = 0; t < v_num_tiles; ++t) {
        // Load one tile of A into local memory.
        {
#pragma unroll 4
          for (int_tp la = 0; la < LPTA; ++la) {
            int_tp tid = tidm * RTSN + tidn;
            int_tp id = la * RTSN * RTSM + tid;
            int_tp row = id / TSK;
            int_tp col = id % TSK;
            int_tp tiledIndex = TSK * t + col;
            if ((offM + row) < MM && tiledIndex < KK) {
              Asub[row][col] = Aptr[(offM + row) * KK + tiledIndex];
            } else {
              Asub[row][col] = 0.0;
            }
          }
        }
        // Load one tile of B into local memory.
        {
#pragma unroll 4
          for (int_tp lb = 0; lb < LPTB; ++lb) {
            int_tp tid = tidm * RTSN + tidn;
            int_tp id = lb * RTSN * RTSM + tid;
            int_tp col = id % TSN;
            int_tp row = id / TSN;
            int_tp tiledIndex = TSK * t + row;
            if ((offN + col) < NN && tiledIndex < KK) {
              int_tp d_iter_0;
              int_tp d_temp_0;
              int_tp d_iter_1;
              int_tp d_temp_1;
              int_tp imageIndex = offN + col;
              // Compute d_iter, final tiledIndex becomes input feature map ID.
              // Scale d_iter by the dilation factor.
              d_iter_1 = (tiledIndex % v_k_1) * v_d_1;
              tiledIndex = tiledIndex / v_k_1;
              // Compute d_temp.
              // Scale d_temp by the stride and subtract the padding.
              d_temp_1 = (imageIndex % v_imso_1) * v_s_1 - v_p_1;
              imageIndex = imageIndex / v_imso_1;
              // Compute d_iter, final tiledIndex becomes input feature map ID.
              // Scale d_iter by the dilation factor.
              d_iter_0 = (tiledIndex % v_k_0) * v_d_0;
              tiledIndex = tiledIndex / v_k_0;
              // Compute d_temp.
              // Scale d_temp by the stride and subtract the padding.
              d_temp_0 = (imageIndex % v_imso_0) * v_s_0 - v_p_0;
              imageIndex = imageIndex / v_imso_0;
              // Recombine final index, compute in-range.
              bool skip_range_check = false;
              // Used only if padding is not 0.
              bool in_range = !skip_range_check;
              int_tp d_iter_im;
              // Here, d_temp_ represents the column shift,
              // while d_iter_ is the kernel shift.
              d_iter_im = d_temp_0 + d_iter_0;
              tiledIndex = tiledIndex * v_imsi_0 + d_iter_im;
              if (!skip_range_check) {
                in_range &= d_iter_im >= 0 && d_iter_im < v_imsi_0;
              }
              // Here, d_temp_ represents the column shift,
              // while d_iter_ is the kernel shift.
              d_iter_im = d_temp_1 + d_iter_1;
              tiledIndex = tiledIndex * v_imsi_1 + d_iter_im;
              if (!skip_range_check) {
                in_range &= d_iter_im >= 0 && d_iter_im < v_imsi_1;
              }
              if (skip_range_check || in_range) {
                // tiledIndex now holds the memory offset for the input image.
                Bsub[row][col] = Bptr[tiledIndex];
              } else {
                Bsub[row][col] = 0.0;
              }
            }
          }
        }
        // Synchronize to make sure the tile is loaded.
        barrier(CLK_LOCAL_MEM_FENCE);
        // Temporary registers for A and B.
        Dtype4 Areg;
        Dtype4 Breg[WPTN / VWN];
// Loop over the values of a single tile.
#pragma unroll 1
        for (int_tp kt = 0; kt < TSK; kt += TSK_UNROLL) {
#pragma unroll 1
          for (int_tp ku = 0; ku < TSK_UNROLL; ++ku) {
            int_tp k = kt + ku;
// Cache the values of Bsub in registers.
#pragma unroll
            for (int_tp wn = 0; wn < WPTN / VWN; ++wn) {
              int_tp col = tidn + wn * VWN * RTSN;
              VEC_4_0(Breg[wn]) = Bsub[k][col + 0 * RTSN];
              VEC_4_1(Breg[wn]) = Bsub[k][col + 1 * RTSN];
              VEC_4_2(Breg[wn]) = Bsub[k][col + 2 * RTSN];
              VEC_4_3(Breg[wn]) = Bsub[k][col + 3 * RTSN];
            }
// Perform the computation.
#pragma unroll
            for (int_tp wm = 0; wm < WPTM / VWM; ++wm) {
              int_tp row = tidm + wm * VWM * RTSM;
              VEC_4_0(Areg) = Asub[row + 0 * RTSM][k];
              VEC_4_1(Areg) = Asub[row + 1 * RTSM][k];
              VEC_4_2(Areg) = Asub[row + 2 * RTSM][k];
              VEC_4_3(Areg) = Asub[row + 3 * RTSM][k];
#pragma unroll
              for (int_tp wn = 0; wn < WPTN / VWN; ++wn) {
                VEC_4_0(Creg[wm * VWM + 0][wn]) +=
                    VEC_4_0(Areg) * VEC_4_0(Breg[wn]);
                VEC_4_0(Creg[wm * VWM + 1][wn]) +=
                    VEC_4_1(Areg) * VEC_4_0(Breg[wn]);
                VEC_4_0(Creg[wm * VWM + 2][wn]) +=
                    VEC_4_2(Areg) * VEC_4_0(Breg[wn]);
                VEC_4_0(Creg[wm * VWM + 3][wn]) +=
                    VEC_4_3(Areg) * VEC_4_0(Breg[wn]);
                VEC_4_1(Creg[wm * VWM + 0][wn]) +=
                    VEC_4_0(Areg) * VEC_4_1(Breg[wn]);
                VEC_4_1(Creg[wm * VWM + 1][wn]) +=
                    VEC_4_1(Areg) * VEC_4_1(Breg[wn]);
                VEC_4_1(Creg[wm * VWM + 2][wn]) +=
                    VEC_4_2(Areg) * VEC_4_1(Breg[wn]);
                VEC_4_1(Creg[wm * VWM + 3][wn]) +=
                    VEC_4_3(Areg) * VEC_4_1(Breg[wn]);
                VEC_4_2(Creg[wm * VWM + 0][wn]) +=
                    VEC_4_0(Areg) * VEC_4_2(Breg[wn]);
                VEC_4_2(Creg[wm * VWM + 1][wn]) +=
                    VEC_4_1(Areg) * VEC_4_2(Breg[wn]);
                VEC_4_2(Creg[wm * VWM + 2][wn]) +=
                    VEC_4_2(Areg) * VEC_4_2(Breg[wn]);
                VEC_4_2(Creg[wm * VWM + 3][wn]) +=
                    VEC_4_3(Areg) * VEC_4_2(Breg[wn]);
                VEC_4_3(Creg[wm * VWM + 0][wn]) +=
                    VEC_4_0(Areg) * VEC_4_3(Breg[wn]);
                VEC_4_3(Creg[wm * VWM + 1][wn]) +=
                    VEC_4_1(Areg) * VEC_4_3(Breg[wn]);
                VEC_4_3(Creg[wm * VWM + 2][wn]) +=
                    VEC_4_2(Areg) * VEC_4_3(Breg[wn]);
                VEC_4_3(Creg[wm * VWM + 3][wn]) +=
                    VEC_4_3(Areg) * VEC_4_3(Breg[wn]);
              }
            }
          }
        }

        // Synchronize before loading the next tile.
        barrier(CLK_LOCAL_MEM_FENCE);
      }
    }
// Store the final results in C.
#pragma unroll
    for (int_tp wm = 0; wm < WPTM; ++wm) {
      int_tp globalRow = offM + tidm + wm * RTSM;
      Dtype biasval = Dptr[globalRow];
#pragma unroll
      for (int_tp wn = 0; wn < WPTN; ++wn) {
        int_tp globalCol = offN + tidn + wn * RTSN;
        if (globalRow < MM && globalCol < NN) {
          Cptr[globalRow * NN + globalCol] =
              ((Dtype *)(&(Creg[wm][wn / VWN])))[wn % VWN] + v_bmul * biasval;
        }
      }
    }
  }
}
