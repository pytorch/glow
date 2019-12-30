/*
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

#include "InlineOnnxifi.h"

#include "glow/Optimizer/GraphOptimizer/GraphOptimizer.h"
#include "glow/Quantization/Quantization.h"
#include "glow/Quantization/Serialization.h"
#include "glow/Support/Support.h"

#include "llvm/Support/MD5.h"

namespace glow {
namespace onnxifi {

extern bool GlowSaveOnnxifiModel;

namespace {
std::string getProfileFile(llvm::StringRef hash) {
  return strFormat("/tmp/glow-profile-%s.yaml", hash.str().c_str());
}

void computeModelHash(const void *onnxModel, size_t onnxModelSize,
                      llvm::SmallString<32> &str) {
  llvm::MD5::MD5Result res;
  llvm::MD5 MD5;
  MD5.update(llvm::makeArrayRef((uint8_t *)onnxModel, onnxModelSize));
  MD5.final(res);
  llvm::MD5::stringifyResult(res, str);
}
} // namespace

onnxStatus
InlineGraph::initGraph(const void *onnxModel, size_t onnxModelSize,
                       uint32_t weightCount,
                       const onnxTensorDescriptorV1 *weightDescriptors,
                       uint32_t maxSeqLength, void * /*unused */) {
  function_ = executionEngine_.getModule().createFunction("function");

  std::unique_ptr<ONNXIFIModelLoader> loader =
      EXIT_ON_ERR(ONNXIFIModelLoader::parse(
          onnxModel, onnxModelSize, weightCount, weightDescriptors, *function_,
          true /*loadInputsAsPlaceholders*/, backendPtr_->getUseOnnx()));

  onnxInputToPlaceholder_ = loader->getInputVarsMapping();
  onnxOutputToPlaceholder_ = loader->getOutputVarsMapping();
  if (GlowSaveOnnxifiModel) {
    saveOnnxifiModel(function_);
  }

  setZeroLengthSequence(maxSeqLength);
  computeModelHash(onnxModel, onnxModelSize, modelHash_);
  optimize(function_, CompilationMode::Infer);

  PlaceholderBindings dummyBindings;
  CompilationContext cctx{&dummyBindings, &loweredMap_};
  PrecisionConfiguration &precConfig = cctx.precisionConfig;
  precConfig.quantMode = quantizationMode_;

  // If quantizing, load quantization infos and setup the schema.
  if (quantizationMode_ == QuantizationMode::Quantize) {
    precConfig.quantConfig.infos =
        deserializeFromYaml(getProfileFile(modelHash_));
    precConfig.quantConfig.schema = quantization::Schema::Symmetric;
  }

  executionEngine_.compile(CompilationMode::Infer);

  return ONNXIFI_STATUS_SUCCESS;
}

onnxStatus InlineGraph::run(std::unique_ptr<ExecutionContext> ctx,
                            EventPtr outputEvent,
                            onnxTraceEventList *traceEvents) {
  executionEngine_.run(*ctx);

  // Dump profile if requested.
  // TODO: enable configuration of quantization schema
  if (quantizationMode_ == QuantizationMode::Profile) {
    auto QI = quantization::generateNodeQuantizationInfos(
        *(ctx->getPlaceholderBindings()), function_, loweredMap_,
        quantization::Schema::Symmetric, ElemKind::Int8QTy);
    serializeToYaml(getProfileFile(modelHash_), QI);
  }

  if (auto *traceContext = ctx->getTraceContext()) {
    setTraceEvents(traceEvents, traceContext);
  }

  outputEvent->signal(ONNXIFI_STATUS_SUCCESS);
  return ONNXIFI_STATUS_SUCCESS;
}

} // namespace onnxifi
} // namespace glow
