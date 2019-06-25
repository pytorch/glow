/*
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

#include "InlineOnnxifi.h"

#include "glow/Optimizer/GraphOptimizer/GraphOptimizer.h"
#include "glow/Quantization/Quantization.h"
#include "glow/Quantization/Serialization.h"
#include "glow/Support/Support.h"

#include "llvm/Support/MD5.h"

namespace glow {
namespace onnxifi {

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
                       const onnxTensorDescriptorV1 *weightDescriptors) {
  function_ = executionEngine_.getModule().createFunction("function");

  std::unique_ptr<ONNXIFIModelLoader> loader =
      TEMP_EXIT_ON_ERR(ONNXIFIModelLoader::parse(
          onnxModel, onnxModelSize, weightCount, weightDescriptors, *function_,
          true /*loadInputsAsPlaceholders*/, backendPtr_->getUseOnnx()));

  onnxInputToPlaceholder_ = loader->getInputVarsMapping();
  onnxOutputToPlaceholder_ = loader->getOutputVarsMapping();

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

  executionEngine_.compile(CompilationMode::Infer, function_);

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

  outputEvent->signal();
  return ONNXIFI_STATUS_SUCCESS;
}

} // namespace onnxifi
} // namespace glow
