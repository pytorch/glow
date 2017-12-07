#ifndef GLOW_BASE_TRAIN_H
#define GLOW_BASE_TRAIN_H

#include "glow/Base/Tensor.h"

#include <cstddef>
#include <cstdint>
#include <unordered_map>

namespace glow {

class Tensor;

/// This is a list of parameters that the network trainers (such as sgd and
/// adam) use for training the network.
struct TrainingConfig {
  float L1Decay{0};
  float L2Decay{0};
  float learningRate{0.01};
  float momentum{0.0};
  unsigned maxNumThreads{256};
  unsigned batchSize{1};
};

} // namespace glow

#endif // GLOW_BASE_TRAIN_H
