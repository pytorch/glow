/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <array>
#include <string>
#include <type_traits>

/*
 * Copied from fbgemm/ConvUtils.h to avoid the dependency on fbgemm
 */
template <int N, int... Vals>
constexpr
    typename std::enable_if<N == sizeof...(Vals), std::array<int, N>>::type
    array_of_ones() {
  return std::array<int, N>{{Vals...}};
}

template <int N, int... Vals>
constexpr
    typename std::enable_if<N != sizeof...(Vals), std::array<int, N>>::type
    array_of_ones() {
  return array_of_ones<N, Vals..., 1>();
}

/**
 * @brief A struct to conveniently store all convolution parameters.
 */
template <int SPATIAL_DIM = 2> struct conv_param_t {
  int MB;                              ///< Mini Batch size
  int IC;                              ///< Number of Input Channels
  int OC;                              ///< Number of Output Channels
  std::array<int, SPATIAL_DIM> IN_DIM; ///< Input Image Dimension
  int G;                               ///< Number of Groups
  std::array<int, SPATIAL_DIM> K;      ///< Filter (Kernel) dimensions
  std::array<int, SPATIAL_DIM> stride; //< Strides
  std::array<int, SPATIAL_DIM * 2>
      pad; //< Padding (first SPATIAL_DIM is for prev/top/left padding, second
           // SPATIAL_DIM is for next/bottom/right padding)
  std::array<int, SPATIAL_DIM> dilation; //< Kernel dilation

  // The following are derived parameters
  std::array<int, SPATIAL_DIM> OUT_DIM; //< Output Image Dimension
  std::array<int, SPATIAL_DIM> IN_DIMP; //< Input Image Dimension Padded

  /**
   * @brief Constructor for initializing the convolution parameters.
   */
  conv_param_t(
      int mb, int ic, int oc, std::array<int, SPATIAL_DIM> in_dim, int g,
      std::array<int, SPATIAL_DIM> k, std::array<int, SPATIAL_DIM> strd,
      std::array<int, SPATIAL_DIM * 2> pd,
      std::array<int, SPATIAL_DIM> dilations = array_of_ones<SPATIAL_DIM>())
      : MB(mb), IC(ic), OC(oc), IN_DIM(in_dim), G(g), K(k), stride(strd),
        pad(pd), dilation(dilations) {
    if (ic % g != 0) {
      throw std::runtime_error(
          "groups = " + std::to_string(g) +
          " does not divide number of input channels = " + std::to_string(ic));
    }
    if (oc % g != 0) {
      throw std::runtime_error(
          "groups = " + std::to_string(g) +
          " does not divide number of output channels = " + std::to_string(oc));
    }

    for (int d = 0; d < SPATIAL_DIM; ++d) {
      IN_DIMP[d] = IN_DIM[d] + pad[d] + pad[SPATIAL_DIM + d];
      OUT_DIM[d] = (IN_DIMP[d] - dilation[d] * (K[d] - 1) - 1) / stride[d] + 1;
    }
  }

  /**
   * @brief Helper function to get convolution parameters as string.
   */
  std::string toString() const {
    std::string dim_string[3] = {"T", "H", "W"};

    std::string out = "";
    out += "MB:" + std::to_string(MB) + ", ";
    out += "IC:" + std::to_string(IC) + ", ";
    out += "OC:" + std::to_string(OC) + ", ";
    if (SPATIAL_DIM <= 3) {
      for (int d = 0; d < SPATIAL_DIM; ++d) {
        out += "I" + dim_string[3 - SPATIAL_DIM + d] + ":" +
               std::to_string(IN_DIM[d]) + ", ";
      }
    } else {
      for (int d = 0; d < SPATIAL_DIM; ++d) {
        out += "I" + std::to_string(d) + ":" + std::to_string(IN_DIM[d]) + ", ";
      }
    }
    out += "G:" + std::to_string(G) + ", ";
    if (SPATIAL_DIM <= 3) {
      for (int d = 0; d < SPATIAL_DIM; ++d) {
        out += "K" + dim_string[3 - SPATIAL_DIM + d] + ":" +
               std::to_string(K[d]) + ", ";
      }
      for (int d = 0; d < SPATIAL_DIM; ++d) {
        out += "stride_" + dim_string[3 - SPATIAL_DIM + d] + ":" +
               std::to_string(stride[d]) + ", ";
      }
      for (int d = 0; d < SPATIAL_DIM * 2; ++d) {
        out += "pad_" + dim_string[3 - SPATIAL_DIM + (d % SPATIAL_DIM)] + ":" +
               std::to_string(pad[d]) + ", ";
      }
      for (int d = 0; d < SPATIAL_DIM; ++d) {
        out += "dilation_" + dim_string[3 - SPATIAL_DIM + d] + ":" +
               std::to_string(dilation[d]);
        if (d < SPATIAL_DIM - 1) {
          out += ", ";
        }
      }
    } else {
      for (int d = 0; d < SPATIAL_DIM; ++d) {
        out += "K" + std::to_string(d) + ":" + std::to_string(K[d]) + ", ";
      }
      for (int d = 0; d < SPATIAL_DIM; ++d) {
        out += "stride_" + std::to_string(d) + ":" + std::to_string(stride[d]) +
               ", ";
      }
      for (int d = 0; d < SPATIAL_DIM; ++d) {
        out += "pad_" + std::to_string(d) + ":" + std::to_string(pad[d]);
        if (d < SPATIAL_DIM * 2 - 1) {
          out += ", ";
        }
      }
      for (int d = 0; d < SPATIAL_DIM; ++d) {
        out += "dilation_" + std::to_string(d) + ":" +
               std::to_string(dilation[d]) + ", ";
      }
    }
    return out;
  }
};

/**
 * @brief A struct to conveniently store all convolution parameters.
 */
template <int SPATIAL_DIM = 2> struct avg_pool_param_t {
  int MB;                              ///< Mini Batch size
  int IC;                              ///< Number of Input Channels
  std::array<int, SPATIAL_DIM> IN_DIM; ///< Input Image Dimension
  std::array<int, SPATIAL_DIM> K;      ///< Kernel dimensions
  std::array<int, SPATIAL_DIM> stride; //< Strides
  std::array<int, SPATIAL_DIM * 2>
      pad; //< Padding (first SPATIAL_DIM is for prev/top/left padding, second
           // SPATIAL_DIM is for next/bottom/right padding)

  // The following are derived parameters
  int OC;                               ///< Number of Ouptut Channels
  std::array<int, SPATIAL_DIM> OUT_DIM; //< Output Image Dimension
  std::array<int, SPATIAL_DIM> IN_DIMP; //< Input Image Dimension Padded

  /**
   * @brief Constructor for initializing the average pool parameters.
   */
  avg_pool_param_t(int mb, int ic, std::array<int, SPATIAL_DIM> in_dim,
                   std::array<int, SPATIAL_DIM> k,
                   std::array<int, SPATIAL_DIM> strd,
                   std::array<int, SPATIAL_DIM * 2> pd)
      : MB(mb), IC(ic), IN_DIM(in_dim), K(k), stride(strd), pad(pd) {
    for (int d = 0; d < SPATIAL_DIM; ++d) {
      OC = IC;
      IN_DIMP[d] = IN_DIM[d] + pad[d] + pad[SPATIAL_DIM + d];
      OUT_DIM[d] = (IN_DIMP[d] - (K[d] - 1) - 1) / stride[d] + 1;
    }
  }

  /**
   * @brief Helper function to get convolution parameters as string.
   */
  std::string toString() const {
    std::string dim_string[3] = {"T", "H", "W"};

    std::string out = "";
    out += "MB:" + std::to_string(MB) + ", ";
    out += "IC:" + std::to_string(IC) + ", ";
    out += "OC:" + std::to_string(OC) + ", ";
    if (SPATIAL_DIM <= 3) {
      for (int d = 0; d < SPATIAL_DIM; ++d) {
        out += "I" + dim_string[3 - SPATIAL_DIM + d] + ":" +
               std::to_string(IN_DIM[d]) + ", ";
      }
    } else {
      for (int d = 0; d < SPATIAL_DIM; ++d) {
        out += "I" + std::to_string(d) + ":" + std::to_string(IN_DIM[d]) + ", ";
      }
    }
    if (SPATIAL_DIM <= 3) {
      for (int d = 0; d < SPATIAL_DIM; ++d) {
        out += "K" + dim_string[3 - SPATIAL_DIM + d] + ":" +
               std::to_string(K[d]) + ", ";
      }
      for (int d = 0; d < SPATIAL_DIM; ++d) {
        out += "stride_" + dim_string[3 - SPATIAL_DIM + d] + ":" +
               std::to_string(stride[d]) + ", ";
      }
      for (int d = 0; d < SPATIAL_DIM * 2; ++d) {
        out += "pad_" + dim_string[3 - SPATIAL_DIM + (d % SPATIAL_DIM)] + ":" +
               std::to_string(pad[d]) + ", ";
      }
    } else {
      for (int d = 0; d < SPATIAL_DIM; ++d) {
        out += "K" + std::to_string(d) + ":" + std::to_string(K[d]) + ", ";
      }
      for (int d = 0; d < SPATIAL_DIM; ++d) {
        out += "stride_" + std::to_string(d) + ":" + std::to_string(stride[d]) +
               ", ";
      }
      for (int d = 0; d < SPATIAL_DIM; ++d) {
        out += "pad_" + std::to_string(d) + ":" + std::to_string(pad[d]);
        if (d < SPATIAL_DIM * 2 - 1) {
          out += ", ";
        }
      }
    }
    return out;
  }
};
