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
#include <cstdlib>
#include <random>

#include "Bench.h"

using namespace glow;

extern "C" {
// Forward declare functions from libjit.
extern void libjit_conv2d_f(float *outW, const float *inW, const float *filterW,
                            const float *biasW, const size_t *outWdims,
                            const size_t *inWdims, const size_t *filterWdims,
                            const size_t *biasWdims, const size_t *kernelSizes,
                            const size_t *strides, const size_t *pads,
                            size_t group, unsigned depthUnroll);
}

/// Benchmark a convolution with specified parameters on square inputs.
class ConvBench : public Benchmark {
  /// Matrices
  std::vector<float> outW;
  std::vector<float> inW;
  std::vector<float> filterW;
  std::vector<float> biasW;

  /// Dimensions
  // [batch, h, w, channels]
  size_t outWdims[4];
  size_t inWdims[4];
  // [outputChannels, h, w, inputChannels]
  size_t filterWdims[4];

  /// Parameters
  size_t kernelSizes[2];
  size_t strides[2];
  size_t pads[2];
  size_t group;
  unsigned depthUnroll;

public:
  ConvBench(size_t inputBatch, size_t inputEdgeSize, size_t inputChannels,
            size_t filterMultiplier, size_t kernelSize, size_t stride,
            size_t pad, size_t group)
      : kernelSizes{kernelSize, kernelSize}, strides{stride, stride},
        pads{pad, pad}, group(group) {

    inWdims[0] = inputBatch;
    inWdims[1] = inputEdgeSize;
    inWdims[2] = inputEdgeSize;
    inWdims[3] = inputChannels;

    filterWdims[0] = filterMultiplier * group;
    filterWdims[1] = kernelSize;
    filterWdims[2] = kernelSize;
    filterWdims[3] = inWdims[3] / group;

    size_t outEdgeSize =
        ((inputEdgeSize + (2 * pad) - kernelSize) / stride) + 1;
    outWdims[0] = inWdims[0];
    outWdims[1] = outEdgeSize;
    outWdims[2] = outEdgeSize;
    outWdims[3] = filterWdims[0];

    depthUnroll = (((outWdims[3] / group) % 8) == 0) ? 8 : 1;
  }

  virtual void setup() override {
    size_t outSize = mapMult(outWdims, 4);
    size_t inSize = mapMult(inWdims, 4);
    size_t filterSize = mapMult(filterWdims, 4);
    size_t biasSize = filterWdims[0];

    outW.resize(outSize);
    inW.resize(inSize);
    filterW.resize(filterSize);
    biasW.resize(biasSize);

    randomize(inSize, inW.data());
    randomize(filterSize, filterW.data());
    randomize(biasSize, biasW.data());
  }

  virtual void run() override {
    // biasWDims isn't used in libjit_conv2d_f, so we're passing NULL.
    libjit_conv2d_f(outW.data(), inW.data(), filterW.data(), biasW.data(),
                    outWdims, inWdims, filterWdims, NULL, kernelSizes, strides,
                    pads, group, depthUnroll);
  }

  virtual void teardown() override {}

private:
  size_t mapMult(size_t *vec, int size) {
    size_t result = 1;
    for (int i = 0; i < size; i++) {
      result *= vec[i];
    }
    return result;
  }

  void randomize(size_t size, float *a) {
    std::mt19937 gen;
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    for (size_t i = 0; i < size; i++) {
      a[i] = dis(gen);
    }
  }
};

int main() {
  constexpr int reps = 10;
  printf("inputBatch, inputEdgeSize, inputChannels, filterMultiplier, "
         "kernelSize, stride, pad, group, bestInSeconds\n");

  for (size_t inputBatch : {1, 3}) {
    for (size_t inputEdgeSize : {7, 56, 224}) {
      for (size_t inputChannels : {64, 128, 1024}) {
        for (size_t filterMultiplier : {1, 8}) {
          for (size_t kernelSize : {1, 3}) {
            for (size_t stride : {1, 4}) {
              size_t pad = kernelSize / 2;
              if ((inputEdgeSize + (pad * 2)) <= kernelSize)
                continue;
              for (size_t group : {1, 112}) {
                if (inputChannels % group != 0)
                  continue;
                ConvBench b(inputBatch, inputEdgeSize, inputChannels,
                            filterMultiplier, kernelSize, stride, pad, group);
                auto times = bench(&b, reps);
                double time = *(std::min_element(times.begin(), times.end()));
                printf("%zu, %zu, %zu, %zu, %zu, %zu, %zu, %zu, %f\n",
                       inputBatch, inputEdgeSize, inputChannels,
                       filterMultiplier, kernelSize, stride, pad, group, time);
              } // group
            } // stride
          } // kernelSize
        } // filterMultiplier
      } // inputChannels
    } // inputEdgeSize
  } // inputBatch
}
