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

#ifndef GLOW_TORCH_GLOW_SRC_TRAINING_TORCHGLOWTRAINING_H
#define GLOW_TORCH_GLOW_SRC_TRAINING_TORCHGLOWTRAINING_H

#include "PyTorchCommon.h"
#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Graph/Graph.h"
#include <torch/csrc/jit/ir/ir.h>

namespace glow {

/// Loads and trains Glow models from PyTorch/ONNX.
class TorchGlowTraining {
public:
  /// Exporter parameters.
  struct ONNXWriterParameters {
    size_t irVersion{3};
    size_t opsetVersion{10};
  };

  /// Explains how to prepare the input model for training.
  enum class RandomizeWeights {
    // Detects mode automatically depending on file extension.
    // PyTorch models trigger the weights randomization YES,
    // ONNX models don't -> NO.
    AUTO = 0,
    YES = 1,
    NO = 2,
  };

private:
  ExecutionEngine engine_;
  PlaceholderBindings bindings_;
  Function *F_{nullptr};
  Function *TF_{nullptr};
  std::vector<glow::Placeholder *> inputPHs_;
  std::vector<glow::Placeholder *> outputPHs_;
  Placeholder *selectedPH_{nullptr};
  ONNXWriterParameters parameters_;

  /// Releases internal resources.
  void clear();

public:
  /// Construct TorchGlowTraining object.
  TorchGlowTraining() = default;

  /// Cleans up the internals.
  ~TorchGlowTraining();

  /// Public interface, methods must be called in the strict order, i.e.
  /// once init() -> repeatedly train() -> repeatedly save().

  /// Initializes internal Glow objects from \p modelFile file, uses provided
  /// \p backend name, ONNX exporter \p parameters, \p inputs, \p config,
  /// randomizes weights according to the provided \p mode.
  /// \returns error on failure.
  Error init(llvm::StringRef modelFile, std::vector<torch::jit::IValue> &inputs,
             llvm::StringRef backend, const ONNXWriterParameters &parameters,
             const TrainingConfig &config,
             RandomizeWeights mode = RandomizeWeights::AUTO);

  /// Trains the loaded model from the provided \p samples and \p labels.
  /// Samples and labels must have the compatible dimensions and types.
  /// Caller can provide one or more samples and correspondently labels.
  /// Method can be invoked as many times as required.
  /// \returns error in case of uninitiated model or invalid input parameters.
  Error train(const Tensor &samples, const Tensor &labels);

  /// Saves the trained model in ONNX (extended) format to the provided
  /// \p snapshotFile. It's safe to call this method any time after train()
  /// calls. Method leaves the internal trained weights unaffected, and caller
  /// can continue to call train() method again.
  /// \returns error on failure.
  Error save(llvm::StringRef snapshotFile);
};

/// Wrapper class helps to integrate TorchGlowTraining class functionality into
/// Python environment.
class TorchGlowTrainingWrapper {
  // Trainer itself.
  TorchGlowTraining trainer_;
  // Required settings/parameters/configs.
  TorchGlowTraining::ONNXWriterParameters parameters_;
  TrainingConfig config_;

public:
  /// Initializes internal trainer by provided model \p modelPathPyTorch, inputs
  /// \p ptTensors, \p backend type, and flag if weights should be randomized,
  /// \returns false on a failure.
  bool init(const std::string &modelPath,
            const std::vector<at::Tensor> &ptTensors,
            const std::string &backend, bool randomizeWeights);

  /// Takes PyTorch \p ptSamples and \p ptLabels tensors and performs training,
  /// \returns false on a failure.
  bool train(const at::Tensor &ptSamples, const at::Tensor &ptLabels);

  /// Saves Glow model into \p snapshotFile using ONNX format
  /// \returns false on a failure.
  bool save(const std::string &snapshotFile);

  /// Sets ONNXWriterParameters.
  void setONNXWriterParameters(size_t irVersion, size_t opsetVersion);

  /// Sets TrainingConfig.
  void setTrainingConfig(float L1Decay, float L2Decay, float learningRate,
                         float momentum, unsigned batchSize);
};

} // namespace glow

#endif // GLOW_TORCH_GLOW_SRC_TRAINING_TORCHGLOWTRAINING_H
