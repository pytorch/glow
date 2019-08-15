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

#include "TorchGlowTraining.h"
#include "PyTorchFileLoader.h"
#include "glow/Exporter/ONNXModelWriter.h"
#include "glow/Importer/ONNXModelLoader.h"
#include "glow/Optimizer/GraphOptimizer/TrainingPreparation.h"
#include "glow/Support/Error.h"

namespace glow {

namespace {

/// Checks if \p samples Tensor contains a single training example with the
/// correct shape, matching \p input.
bool isValidSamplesForSingleInput(const Tensor &input, const Tensor &samples) {
  return input.dims() == samples.dims().slice(1) &&
         input.getElementType() == samples.getElementType();
}

/// Checks if \p samples Tensor contains multiple training examples with the
/// correct shape, matching \p input.
bool isValidSamplesForSliceInput(const Tensor &input, const Tensor &samples) {
  return input.dims().size() > 1 &&
         input.dims().slice(1) == samples.dims().slice(1) &&
         input.getElementType() == samples.getElementType();
}

} // namespace

void TorchGlowTraining::clear() {
  F_ = nullptr;
  TF_ = nullptr;
  inputPHs_.clear();
  outputPHs_.clear();
  selectedPH_ = nullptr;
  bindings_.clear();
  engine_.clear();
}

TorchGlowTraining::~TorchGlowTraining() { clear(); }

llvm::Error TorchGlowTraining::init(llvm::StringRef modelFile,
                                    std::vector<torch::jit::IValue> &inputs,
                                    llvm::StringRef backend,
                                    const ONNXWriterParameters &parameters,
                                    const PyTorchLoaderSettings &settings,
                                    const TrainingConfig &config,
                                    RandomizeWeights mode) {
  // Clean up all previous allocations, if any.
  clear();
  // Initialize execution engine.
  engine_.setBackendName(backend);
  // Create glow Function, only one object will be in module.
  F_ = engine_.getModule().createFunction("torch_glow_model");
  // Execution lambda helps to use compact Glow RETURN_* macros inside, and
  // have possibility to clean-up resources before leaving the function scope.
  auto setup = [&]() -> llvm::Error {
    // Detect the proper loader.
    if (modelFile.endswith_lower(".pt")) {
      RETURN_IF_ERR(PyTorchFileLoader::loadPyTorchGraph(
          modelFile.str(), inputs, *F_, inputPHs_, outputPHs_, settings, true));
      if (mode == RandomizeWeights::AUTO) {
        mode = RandomizeWeights::YES;
      }
    } else if (modelFile.endswith_lower(".onnx")) {
      llvm::Error err = llvm::Error::success();
      MARK_ERR_CHECKED(err);
      ONNXModelLoader loader(modelFile.str(), {}, {}, *F_, &err);
      RETURN_IF_ERR(err);
      if (mode == RandomizeWeights::AUTO) {
        mode = RandomizeWeights::NO;
      }
    } else {
      RETURN_ERR("Unrecognized file extension, expect *.pt or *.onnx.");
    }

    RETURN_ERR_IF_NOT(inputPHs_.size() == 1,
                      glow::strFormat("Only 1 input is supported, got %lu",
                                      inputPHs_.size()));
    RETURN_ERR_IF_NOT(outputPHs_.size() == 1,
                      glow::strFormat("Only 1 output is supported, got %lu",
                                      outputPHs_.size()));

    // Initialization succeeded, prepare for training.
    bindings_.allocate(engine_.getModule().getPlaceholders());
    TensorInitializer initializer;
    switch (mode) {
    case RandomizeWeights::AUTO:
      assert(false);
      break;
    case RandomizeWeights::YES:
      initializer = getDefaultTensorInitializer();
      break;
    case RandomizeWeights::NO:
      initializer = [](Function *, Node *, unsigned, Tensor *) {};
      break;
    }

    RETURN_IF_ERR(glow::prepareFunctionForTraining(F_, bindings_, selectedPH_,
                                                   std::move(initializer)));

    TF_ = glow::differentiate(F_, config);
    engine_.compile(CompilationMode::Train);

    return llvm::Error::success();
  };

  llvm::Error err = setup();
  if (err) {
    // On failure cleanup resources.
    clear();
    return err;
  }

  parameters_ = parameters;
  return llvm::Error::success();
}

llvm::Error TorchGlowTraining::train(const Tensor &samples,
                                     const Tensor &labels) {
  RETURN_ERR_IF_NOT(TF_, "Class instance, wasn't properly initialized.");
  auto *input = bindings_.get(inputPHs_[0]);
  auto *label = bindings_.get(selectedPH_);
  auto TFName = TF_->getName();

  // Sanity check.
  const bool isSlice = isValidSamplesForSliceInput(*input, samples) &&
                       isValidSamplesForSliceInput(*label, labels);

  if (!isSlice && (!isValidSamplesForSingleInput(*input, samples) ||
                   !isValidSamplesForSingleInput(*label, labels))) {
    RETURN_ERR("Invalid samples/labels dimensions.");
  }

  size_t numEpochs = samples.dims()[0] / input->dims()[0];
  for (size_t e = 0; e < numEpochs; ++e) {
    isSlice ? input->copyConsecutiveSlices(&samples, e)
            : input->copySlice(&samples, e);
    isSlice ? label->copyConsecutiveSlices(&labels, e)
            : label->copySlice(&labels, e);
    engine_.run(bindings_, TFName);
  }

  return llvm::Error::success();
}

llvm::Error TorchGlowTraining::save(llvm::StringRef snapshotFile) {
  llvm::Error err = llvm::Error::success();
  MARK_ERR_CHECKED(err);
  // Detects output ONNX file format, text or binary.
  const bool textMode = snapshotFile.endswith_lower(".onnxtxt");

  // Move placeholder tensors into constants.
  llvm::ArrayRef<Placeholder *> phs = {inputPHs_[0], outputPHs_[0]};
  std::list<std::pair<Placeholder *, Constant *>> cache;
  for (auto &PH : F_->findPlaceholders()) {
    if (std::find(phs.begin(), phs.end(), PH) != phs.end()) {
      continue;
    }
    auto *tensor = bindings_.get(PH);
    if (!tensor) {
      continue;
    }
    // Don't make a tensor copy, just move it.
    auto *constant =
        engine_.getModule().createConstant(PH->getName(), std::move(*tensor));
    PH->getOutput().replaceAllUsesOfWith(constant, F_);
    cache.push_back({PH, constant});
  }

  ONNXModelWriter writer(snapshotFile, *F_, parameters_.irVersion,
                         parameters_.opsetVersion, &err, textMode);

  // Move tensors back to bindings.
  for (auto &entry : cache) {
    auto *constant = entry.second;
    auto *PH = entry.first;

    constant->getOutput().replaceAllUsesOfWith(PH, F_);
    // Don't make a tensor copy, just move it.
    bindings_.insert(PH, std::move(constant->getPayloadMutable()));
    engine_.getModule().eraseConstant(constant);
  }

  return err;
}

} // namespace glow
