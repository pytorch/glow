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
  float learningRate{0.01f};
  float momentum{0.0};
  unsigned batchSize{1};
};

} // namespace glow

#endif // GLOW_BASE_TRAIN_H
