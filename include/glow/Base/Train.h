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

/// These are all the supported training algorithms.
enum class TrainingAlgorithm { None, StochasticGradientDescent, Adagrad };

/// This is a list of common parameters that all of the training algorithms use.
struct TrainingParameters {
  float learningRate{0.01f};
  unsigned batchSize{1};
};

/// Additional training parameters used by stochastic gradient descent.
struct SGDParameters : public TrainingParameters {
  float L1Decay{0};
  float L2Decay{0};
  float momentum{0.0};
};

/// Additional training parameters used by Adagrad.
struct AdagradParameters : public TrainingParameters {
  float epsilon{1e-5};
};

/// Combination of training algorithm type and training parameters. Parameters
/// should be casted to the appropriate type based on the algorithm type.
struct TrainingConfig {
  TrainingConfig(
      TrainingAlgorithm algo = TrainingAlgorithm::StochasticGradientDescent)
      : algorithm(algo) {
    if (algorithm == TrainingAlgorithm::StochasticGradientDescent) {
      parameters = llvm::make_unique<SGDParameters>();
    } else if (algorithm == TrainingAlgorithm::Adagrad) {
      parameters = llvm::make_unique<AdagradParameters>();
    } else {
      llvm_unreachable("Invalid training algorithm.");
    }
  }

  template <class ParametersTy = SGDParameters>
  ParametersTy *getParams() const {
    return static_cast<ParametersTy *>(parameters.get());
  }

  TrainingAlgorithm algorithm{TrainingAlgorithm::None};
  std::unique_ptr<TrainingParameters> parameters{nullptr};
};

} // namespace glow

#endif // GLOW_BASE_TRAIN_H
