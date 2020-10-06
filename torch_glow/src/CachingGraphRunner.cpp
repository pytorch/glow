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

#include "CachingGraphRunner.h"

#include <mutex>

#include "glow/Exporter/ONNXModelWriter.h"
#include "glow/Support/Support.h"

#include <c10/util/hash.h>
#include <torch/csrc/jit/runtime/argument_spec.h>

#include "ShapeInferenceEngine.h"

namespace glow {

namespace {

// Hashing a stack of tensors using their types.
size_t hashTensorStack(const torch::jit::Stack &stack, size_t numInputs,
                       const CachingGraphRunner *const runnerPtr) {
  const auto inputs = torch::jit::last(stack, numInputs);
  // Start off hash with pointer to this CachingGraphRunner to avoid collisions
  // with Glow functions created by other CachingGraphRunners.
  size_t hash = reinterpret_cast<size_t>(runnerPtr);
  for (auto &input : inputs) {
    CHECK(input.isTensor()) << "Found non-tensor input. Glow AOT compiled "
                               "graph accepts tensor inputs only.";
    // hash on input Tensor type
    const auto ptTensorType = c10::TensorType::create(input.toTensor());
    size_t tensorHash = std::hash<c10::TensorType>()(*ptTensorType);
    hash = torch::hash_combine(hash, tensorHash);
  }
  return hash;
}

// Use inputMeta to create a fake stack of empty tensors. Helper function to
// enable calling hashTensorStack from warmCache.
torch::jit::Stack
createFakeStackFromInputMeta(const std::vector<InputMeta> &inputMeta) {
  torch::jit::Stack stack;
  for (auto &meta : inputMeta) {
    std::vector<int64_t> dims;
    dims.reserve(meta.dims.size());
    for (auto d : meta.dims) {
      dims.push_back(d);
    }
    stack.push_back(
        c10::IValue(at::empty(dims, at::TensorOptions().dtype(meta.type))));
  }
  return stack;
}
} // namespace

// TODO: this should also return the list of TensorTypes used to compute the
// hash to check for equality. Will make a nicer wrapper for this in the future.
size_t CachingGraphRunner::computeGraphHash(
    const c10::ArrayRef<c10::IValue> inputs) const {
  // Start off hash with pointer to this CachingGraphRunner to avoid collisions
  // with Glow functions created by other CachingGraphRunners.
  size_t hash = reinterpret_cast<size_t>(this);

  for (auto &input : inputs) {
    if (input.isTensor()) {
      // hash on input Tensor type
      const auto ptTensorType = c10::TensorType::create(input.toTensor());
      size_t tensorHash = std::hash<c10::TensorType>()(*ptTensorType);
      hash = torch::hash_combine(hash, tensorHash);
    } else if (input.isBool()) {
      size_t inputHash = std::hash<bool>()(input.toBool());
      hash = torch::hash_combine(hash, inputHash);
    } else if (input.isInt()) {
      // just doing Int and IntList for now.
      size_t inputHash = std::hash<int64_t>()(input.toInt());
      hash = torch::hash_combine(hash, inputHash);
    } else if (input.isIntList()) {
      std::vector<int64_t> inputList = input.toIntVector();
      size_t inputHash = 0; // std::hash<std::vector<int64_t>>()(inputList);
      for (auto el : inputList) {
        size_t elHash = std::hash<int64_t>()(el);
        inputHash = torch::hash_combine(inputHash, elHash);
      }
      hash = torch::hash_combine(hash, inputHash);
    } // else continue;;
  }
  return hash;
}

// Hashing input tensors using their shapes.
size_t CachingGraphRunner::hashTensorShape(
    const c10::ArrayRef<c10::IValue> &inputs) const {

  assert(perGlowGraphInfoMap_.size() == 1);
  // Start off hash with pointer to this CachingGraphRunner to avoid collisions
  // with Glow functions created by other CachingGraphRunners.
  size_t hash = reinterpret_cast<size_t>(this);
  for (auto &input : inputs) {
    CHECK(input.isTensor()) << "Found non-tensor input. Glow AOT compiled "
                               "graph accepts tensor inputs only.";
    // hash on input tensor shape
    for (auto dimSize : input.toTensor().sizes()) {
      size_t tensorHash = std::hash<int64_t>()(dimSize);
      hash = torch::hash_combine(hash, tensorHash);
    }
  }
  return hash;
}

void CachingGraphRunner::aggregateAndDumpTraces(TraceContext *traceContext,
                                                bool flush) {
  size_t numTracesPerDump = settings_.numTracesPerDump;
  bool doDump = false;
  std::string filename;
  {
    std::unique_lock<std::mutex> lock(tracesMutex_);

    if (traceContext) {
      mergedTraceContext_->merge(traceContext);
      numTraces_++;
    } else if (mergedTraceContext_->getTraceEvents().empty()) {
      return;
    }
    size_t numTraces = numTraces_;

    // If numTracesPerDump <= 0, it means we don't merge unless there is a flush
    if (flush || (numTracesPerDump > 0 && numTraces % numTracesPerDump == 0)) {
      // Initial way of differentiating the dump files when there are multiple
      // graph runners
      // TODO(allwu): find a better way to generate trace file names
      size_t hash = reinterpret_cast<size_t>(this);
      size_t dumpNum = numTraceDumps_++;
      filename =
          strFormat("glow-trace-%04lx-%zu.json", hash % (1 << 16), dumpNum);
      doDump = true;
    }
  }
  if (doDump) {
    mergedTraceContext_->dump(filename);
    mergedTraceContext_ = glow::make_unique<TraceContext>(TraceLevel::STANDARD);
  }
}

Expected<std::shared_ptr<CachingGraphRunner::PerGlowGraphInfo>>
CachingGraphRunner::loadImpl(torch::jit::Stack &stack,
                             const PyTorchLoaderSettings &settings,
                             TraceContext *traceContext) {
  if (settings.preCompilePyTorchModule) {
    return MAKE_ERR(
        "Calling JIT compilation when preCompilePyTorchModule is set");
  }

  TRACE_EVENT_SCOPE(traceContext, TraceLevel::RUNTIME, "torch_glow::loadImpl");
  const auto inputs = torch::jit::last(stack, graph_->inputs().size());

  TRACE_EVENT_BEGIN(traceContext, TraceLevel::RUNTIME, "computeGraphHash");
  size_t hash = computeGraphHash(stack);
  TRACE_EVENT_END(traceContext, TraceLevel::RUNTIME, "computeGraphHash");

  // If we already have a Glow function compiled for this graph with and the
  // given inputs then use that.
  std::unique_lock<std::shared_timed_mutex> wlock(graphInfoMapMutex);
  auto it = perGlowGraphInfoMap_.find(hash);
  if (it != perGlowGraphInfoMap_.end()) {
    return it->second;
  }
  auto info = std::make_shared<PerGlowGraphInfo>(
      strFormat("pt_function_%lu", hash), settings);

  std::unique_ptr<Module> module = glow::make_unique<Module>();
  Function *f = module->createFunction(info->functionName);

  TRACE_EVENT_BEGIN(traceContext, TraceLevel::RUNTIME, "loadJITGraph");
  RETURN_IF_ERR(PyTorchModelLoader::loadJITGraph(
      *f, *graph_, info->inputPlaceholders, info->outputPlaceholders,
      outputCorrectType_, settings, inputs, {}));
  TRACE_EVENT_END(traceContext, TraceLevel::RUNTIME, "loadJITGraph");

  glow::CompilationContext cctx;
  initializeCompiliationContextFromSettings(cctx, settings);

  if (settings.convertToFP16) {
    cctx.precisionConfig.precisionModeKindSet.insert(
        Kinded::Kind::ChannelwiseQuantizedConvolutionNodeKind);
    cctx.precisionConfig.precisionModeKindSet.insert(
        Kinded::Kind::RowwiseQuantizedFullyConnectedNodeKind);
  }
  cctx.replicationCount = settings.replicationCount;
  cctx.backendOpts.backendSpecificOpts = settings.backendSpecificOpts;

  TRACE_EVENT_BEGIN(traceContext, TraceLevel::RUNTIME, "addNetwork");
  // If --load-backend-specific-opts was passed from python, add it to the
  // compile context so the host manager knows to load backend options from
  // yaml.

  if (!settings.backendOptionsFile.empty()) {
    std::pair<std::string, std::string> loadBackendSpecificOpts(
        "loadBackendSpecificOptions", settings.backendOptionsFile);
    cctx.backendOpts.backendSpecificOpts.insert(loadBackendSpecificOpts);
  }

  RETURN_IF_ERR(hostManager_->addNetwork(std::move(module), cctx));

  TRACE_EVENT_END(traceContext, TraceLevel::RUNTIME, "addNetwork");

  auto ret = perGlowGraphInfoMap_.emplace(hash, info);
  CHECK(ret.second);

  return ret.first->second;
}

Expected<MetaStack *>
CachingGraphRunner::loadShape(const c10::ArrayRef<c10::IValue> &inputs,
                              TraceContext *traceContext) {

  TRACE_EVENT_SCOPE(traceContext, TraceLevel::RUNTIME, "torch_glow::loadShape");
  TRACE_EVENT_BEGIN(traceContext, TraceLevel::RUNTIME, "computeShapeHash");
  size_t hash = hashTensorShape(inputs);
  TRACE_EVENT_END(traceContext, TraceLevel::RUNTIME, "computeShapeHash");

  // If we already have a shape info for this graph output with and the
  // given inputs then use that.
  auto it = perGlowGraphShapeMap_.find(hash);
  if (it != perGlowGraphShapeMap_.end()) {
    return &(it->second);
  }

  // If we don't have a shape info for this graph output with and the
  // given inputs then run shape inference, then push into the map.
  TRACE_EVENT_BEGIN(traceContext, TraceLevel::RUNTIME, "runShapeInference");

  ShapeInferenceEngine shapeG(*graph_, inputs);
  RETURN_IF_ERR(shapeG.run());
  auto outputShape = shapeG.getGraphOutputShape();

  TRACE_EVENT_END(traceContext, TraceLevel::RUNTIME, "runShapeInference");

  auto ret = perGlowGraphShapeMap_.emplace(hash, outputShape);
  CHECK(ret.second);
  return &(ret.first->second);
}

int64_t CachingGraphRunner::runOnJit(torch::jit::Stack &stack) {
  static std::mutex runJitLock;
  std::lock_guard<std::mutex> guard(runJitLock);
  bool temp = getPyTorchLoaderSettings().fusionPassEnabled;
  getPyTorchLoaderSettings().fusionPassEnabled = false;
  int64_t startTime = TraceEvent::now();
  ptGraphExecutor_.run(stack);
  int64_t runTime = TraceEvent::now() - startTime;
  getPyTorchLoaderSettings().fusionPassEnabled = temp;
  return runTime;
}

struct TensorCompareResult {
  double relErr;
  double maxErr;
  double maxRelErr;
};

template <typename Ty>
TensorCompareResult compareTensors(glow::Tensor &RefT, glow::Tensor &CmpT) {
  TensorCompareResult result = {INFINITY, INFINITY, INFINITY};
  if (CmpT.getHandle<Ty>().size() != RefT.getHandle<Ty>().size()) {
    LOG(ERROR) << "Dimension mismatch: "
               << "\tReference dims: " << RefT.getHandle().getType().dims()
               << "\tGlow dims: " << CmpT.getHandle().getType().dims()
               << std::endl;
    return result;
  }
  double totalErrSq = 0.0;
  double totalMagSq = 0.0;
  double maxErr = 0.0;
  double maxRelErr = 0.0;
  for (dim_t idx = 0; idx < RefT.getHandle().size(); idx++) {
    double refVal = (double)RefT.getHandle<Ty>().raw(idx);
    double cmpVal = (double)CmpT.getHandle<Ty>().raw(idx);
    double diff = refVal - cmpVal;
    double mag = refVal * refVal;
    double eltRelErr = (fabs(refVal)) > 0.0 ? fabs(diff) / fabs(refVal) : 0.0;
    totalErrSq += diff * diff;
    totalMagSq += mag;
    maxErr = (fabs(diff) > maxErr) ? fabs(diff) : maxErr;
    maxRelErr = (eltRelErr > maxRelErr) ? eltRelErr : maxRelErr;
  }
  result.relErr = (totalMagSq > 0.0) ? std::sqrt(totalErrSq / totalMagSq) : 0.0;
  result.maxErr = maxErr;
  result.maxRelErr = maxRelErr;
  return result;
}

/// This function slice the input Tensor according to the expected shape in the
/// zero dimension.
/// TODO: Multi-dimension slicing will be supported later.
at::Tensor sliceTensor(at::Tensor &t, const TensorShape &shape) {
  CHECK_GT(shape.size(), 0);
  return at::native::slice(t, 0, 0, shape[0]);
}

Error CachingGraphRunner::runImpl(const PerGlowGraphInfo &info,
                                  torch::jit::Stack &stack,
                                  std::unique_ptr<ExecutionContext> &ctx) {
  size_t runId = numRuns_++;

  int64_t jitRunningTime = 0;
  const PyTorchLoaderSettings &settings = info.settings;

  // Run the subgraph using JIT for comparison with Glow.
  torch::jit::Stack copyStack;
  if (settings.writeToOnnx || settings.jitVsGlowCompare) {
    for (auto &ival : stack) {
      if (ival.isTensor()) {
        copyStack.push_back(ival.deepcopy());
      } else {
        copyStack.push_back(ival);
      }
    }
    jitRunningTime = runOnJit(copyStack);
  }

  TraceContext *traceContext = ctx->getTraceContext();
  TRACE_EVENT_BEGIN(traceContext, TraceLevel::RUNTIME, "torch_glow::runImpl");

  auto *bindings = ctx->getPlaceholderBindings();

  TRACE_EVENT_BEGIN(traceContext, TraceLevel::RUNTIME, "adjustInputs");

  size_t numInputs = graph_->inputs().size();
  const auto inputs = torch::jit::last(stack, numInputs);

  // We only hold placeholders for tensor inputs so indexing them is different
  // than indexing all inputs.
  size_t placeholderI = 0;

  for (const auto &input : inputs) {
    if (input.isTensor()) {

      glow::Placeholder *ph = info.inputPlaceholders[placeholderI++];
      glow::TypeRef ty = ph->getType();

      auto ptTensor = input.toTensor();
      bool needClone = false;

      if (ptTensor.is_quantized()) {
        ptTensor = convertQuantizedToDtype(ptTensor, at::kQInt8);
        // We need to clone a new tensor here since
        // convertQuantizedToDtype might create a temporary tensor
        needClone = true;
      }

      if (!ptTensor.is_contiguous()) {
        ptTensor = ptTensor.contiguous();
        needClone = true;
      }

      // Check Tensor size, making sure enough memory is allocated
      if (ptTensor.numel() > ty->size()) {
        std::stringstream ss;
        ss << "Input tensor is too large: " << ptTensor.numel() << " vs "
           << ty->size() << ": " << ph->getName().str();
        return MAKE_ERR(ss.str());
      }

      if (ty->dims().size() == ptTensor.ndimension() &&
          std::equal(ty->dims().begin(), ty->dims().end(),
                     ptTensor.sizes().begin())) {
        glow::Tensor t;
        if (needClone) {
          t = glow::Tensor(ptTensor.data_ptr(), ty).clone();
        } else {
          t = glow::Tensor(ptTensor.data_ptr(), ty);
        }
        bindings->insert(ph, std::move(t));
      } else if (ptTensor.data_ptr() && ptTensor.numel() > 0 &&
                 backend_.supportsPartialTensors()) {
        // This is a partial tensor, to create padded unown tensor
        glow::Tensor t;
        if (needClone) {
          t = glow::Tensor(ptTensor.data_ptr(), ty, ptTensor.nbytes()).clone();
        } else {
          t = glow::Tensor(ptTensor.data_ptr(), ty, ptTensor.nbytes());
        }
        bindings->insert(ph, std::move(t));
      } else if (ptTensor.numel() == 0) {
        // Handles zero-size input tensor
        // Here zeroLengthSequence_ is pre-allocated if warmCache is called
        assert(zeroLengthSequence_.getUnsafePtr());
        bindings->insert(
            ph, glow::Tensor((void *)zeroLengthSequence_.getUnsafePtr(), ty));
      } else {
        // For backends that does not support partial tensor, manually pad zeros
        auto inputTensorOpt = tensorPool_.get(ty);
        if (!inputTensorOpt.hasValue()) {
          std::stringstream ss;
          ss << "Tensorpool tensor not found for input " << ptTensor.name();
          return MAKE_ERR(ss.str());
        }
        // We want fresh DeviceResidencyInfo for this fresh Tensor.
        Tensor inputTensor(std::move(inputTensorOpt.getValue()));
        inputTensor.resetDeviceInfo();
        if (ptTensor.data_ptr()) {
          memcpy(inputTensor.getUnsafePtr(), ptTensor.data_ptr(),
                 ptTensor.nbytes());
          // Pad remaining space with zeroes.
          memset(inputTensor.getUnsafePtr() + ptTensor.nbytes(), 0,
                 inputTensor.getSizeInBytes() - ptTensor.nbytes());
        } else {
          inputTensor.zero();
        }
        bindings->insert(ph, std::move(inputTensor));
      }
    } else if (input.isObject()) {
      // Objects are only used for loading attributes at compile time.
      continue;
    } else if (!(input.isBool() || input.isInt() || input.isIntList())) {
      return MAKE_ERR(
          "Only Int/IntList, Tensor and Object IValue inputs are accepted");
    }
  }

  if (settings.writeToOnnx) {
    std::string filename =
        strFormat("%s_input_%zu.onnx", info.functionName.c_str(), runId);
    std::ofstream of(filename, std::ios::binary);
    if (!of) {
      LOG(ERROR) << "Cannot create input file " << filename;
    } else {
      ONNX_NAMESPACE::GraphProto inputG;
      for (const auto &p : bindings->pairs()) {
        auto *onnxT = inputG.add_initializer();
        const auto ph = p.first;
        const auto &t = p.second;
        onnxT->set_name(ph->getName());
        size_t unpaddedSize = t.getUnpaddedSizeInBytes();
        size_t tensorSize = t.getSizeInBytes();
        if (unpaddedSize == tensorSize) {
          ONNXModelWriter::writeTensor(t, onnxT, /*useGlowCustomOps*/ true);
        } else {
          // If the input is a partial tensor, then save only the part that has
          // data.
          auto ty = t.getType();
          auto dims = ty.dims().vec();
          assert(dims.size() > 0);
          dims[0] = dims[0] * unpaddedSize / tensorSize;
          const auto &resized = t.getUnowned(dims);
          ONNXModelWriter::writeTensor(resized, onnxT,
                                       /*useGlowCustomOps*/ true);
        }
      }
      std::string buffer;
      inputG.SerializeToString(&buffer);
      of << buffer;
    }
  }

  TRACE_EVENT_END(traceContext, TraceLevel::RUNTIME, "adjustInputs");
  TRACE_EVENT_BEGIN(traceContext, TraceLevel::RUNTIME, "setupOutput");

  std::vector<at::IValue> outputs;
  for (auto *ph : info.outputPlaceholders) {
    std::vector<int64_t> sizes;
    for (auto size : ph->dims()) {
      sizes.push_back(static_cast<int64_t>(size));
    }

    auto ptT = glowTypeToEmptyPTTensor(*ph->getType());

    glow::Tensor t(ptT.data_ptr(), ph->getType());

    outputs.push_back(std::move(ptT));
    bindings->insert(ph, std::move(t));
  }

  TRACE_EVENT_END(traceContext, TraceLevel::RUNTIME, "setupOutput");
  TRACE_EVENT_BEGIN(traceContext, TraceLevel::RUNTIME, "runNetwork");

  int64_t glowRunStartTime = TraceEvent::now();
  auto err = hostManager_->runNetworkBlocking(info.functionName, ctx);
  int64_t glowRuningnTime = TraceEvent::now() - glowRunStartTime;

  // Reset the traceContext again in case it was changed during run.
  traceContext = ctx->getTraceContext();

  TRACE_EVENT_END(traceContext, TraceLevel::RUNTIME, "runNetwork");
  TRACE_EVENT_BEGIN(traceContext, TraceLevel::RUNTIME, "setOutputs");

  MetaStack *ptrOutputShape;
  if (settings_.runShapeInference) {
    /// load shape. If already existed, extracted directly, if not, run shape
    /// inference
    ASSIGN_VALUE_OR_RETURN_ERR(ptrOutputShape, loadShape(inputs, traceContext));
    if (outputs.size() != (*ptrOutputShape).size()) {
      return MAKE_ERR("Fail to infer shape for outputs");
    }
  }

  torch::jit::drop(stack, numInputs);

  ONNX_NAMESPACE::GraphProto outputG;
  ONNX_NAMESPACE::GraphProto jitOutputG;
  for (int i = 0; i < outputs.size(); i++) {
    auto &output = outputs[i];
    auto ptTensor = output.toTensor();

    if (ptTensor.is_quantized()) {
      c10::ScalarType dtype = outputCorrectType_[i];
      if (dtype == c10::ScalarType::QUInt8 || dtype == c10::ScalarType::QInt8) {
        ptTensor = convertQuantizedToDtype(ptTensor, dtype);
      } else {
        return MAKE_ERR(
            strFormat("Fail to propagate quantized dtype to output"));
      }
    }

    if (settings.writeToOnnx) {
      glow::Tensor glowT = ptTensorToGlowTensor(ptTensor);
      auto *onnxT = outputG.add_initializer();
      onnxT->set_name(info.outputPlaceholders[i]->getName());
      ONNXModelWriter::writeTensor(glowT, onnxT, /*useGlowCustomOps*/ true);
    }

    if (settings.writeToOnnx) {
      auto &jitOutput = torch::jit::peek(copyStack, i, outputs.size());
      auto jitPtTensor = jitOutput.toTensor().contiguous();
      glow::Tensor jitGlowT = ptTensorToGlowTensor(jitPtTensor);
      auto *jitOnnxT = jitOutputG.add_initializer();
      jitOnnxT->set_name(info.outputPlaceholders[i]->getName());
      ONNXModelWriter::writeTensor(jitGlowT, jitOnnxT,
                                   /*useGlowCustomOps*/ true);
    }

    if (settings.runShapeInference) {
      if (i < (*ptrOutputShape).size()) {
        // Assuming all the outputs are tensors for now
        // TODO Add support for other output types, e.g., tensor[]
        const TensorShape &expectedShape =
            (*ptrOutputShape)[i].shape<TensorShape>();
        ptTensor = sliceTensor(ptTensor, expectedShape);
      }
    }

    if (settings.jitVsGlowCompare) {
      glow::Tensor glowT = ptTensorToGlowTensor(ptTensor);
      auto &jitOutput = torch::jit::peek(copyStack, i, outputs.size());
      auto jitPtTensor = jitOutput.toTensor().contiguous();
      glow::Tensor jitGlowT = ptTensorToGlowTensor(jitPtTensor);
      auto tensorElemType = jitGlowT.getType().getElementType();
      if (tensorElemType == glow::ElemKind::FloatTy) {
        TensorCompareResult res = compareTensors<float_t>(jitGlowT, glowT);
        LOG(INFO) << "Correctness check | Function: " << info.functionName
                  << "\tTensor: "
                  << std::string(info.outputPlaceholders[i]->getName())
                  << "\tRelError: " << res.relErr << "\tMaxErr: " << res.maxErr
                  << "\tMaxRelErr: " << res.maxRelErr << std::endl;
      } else {
        LOG(INFO) << "Correctness Check | Function: " << info.functionName
                  << "\tTensor: "
                  << std::string(info.outputPlaceholders[i]->getName())
                  << "\tUnsupported type: "
                  << std::string(glow::Type::getElementName(tensorElemType))
                  << std::endl;
      }
    }
    stack.push_back(at::IValue(std::move(ptTensor)));
  }

  if (settings.jitVsGlowCompare) {
    LOG(INFO) << "Perf comparison | Function: " << info.functionName
              << "\tGlow run time: " << glowRuningnTime
              << " us\tCPU run time: " << jitRunningTime << " us" << std::endl;
  }

  if (settings.writeToOnnx) {
    std::string filename =
        strFormat("%s_glow_output_%zu.onnx", info.functionName.c_str(), runId);
    std::ofstream of(filename, std::ios::binary);
    if (!of) {
      LOG(ERROR) << "Cannot create output file " << filename;
    } else {
      std::string buffer;
      outputG.SerializeToString(&buffer);
      of << buffer;
    }
  }

  if (settings.writeToOnnx) {
    std::string filename = strFormat("%s_pytorch_output_%zu.onnx",
                                     info.functionName.c_str(), runId);
    std::ofstream of(filename, std::ios::binary);
    if (!of) {
      LOG(ERROR) << "Cannot create output file " << filename;
    } else {
      std::string buffer;
      jitOutputG.SerializeToString(&buffer);
      of << buffer;
    }
  }

  TRACE_EVENT_END(traceContext, TraceLevel::RUNTIME, "setOutputs");

  TRACE_EVENT_END(traceContext, TraceLevel::RUNTIME, "torch_glow::runImpl");
  return err;
}

Error CachingGraphRunner::run(torch::jit::Stack &stack) {
  std::unique_ptr<ExecutionContext> ctx = glow::make_unique<ExecutionContext>();

  TraceContext *traceContext = nullptr;
  if (getSettings().enableGlowTracing) {
    ctx->setTraceContext(glow::make_unique<TraceContext>(TraceLevel::STANDARD));
    traceContext = ctx->getTraceContext();
    traceContext->setThreadName("torch_glow");
  }

  TRACE_EVENT_BEGIN(traceContext, TraceLevel::RUNTIME, "torch_glow::run");

  std::shared_ptr<PerGlowGraphInfo> info;
  ASSIGN_VALUE_OR_RETURN_ERR(info,
                             loadImpl(stack, getSettings(), traceContext));
  auto err = runImpl(*DCHECK_NOTNULL(info.get()), stack, ctx);

  // Reset the traceContext again in case it was changed during run.
  traceContext = ctx->getTraceContext();

  TRACE_EVENT_END(traceContext, TraceLevel::RUNTIME, "torch_glow::run");

  aggregateAndDumpTraces(traceContext);

  return err;
}

Error CachingGraphRunner::runOnly(torch::jit::Stack &stack) {
  std::shared_ptr<PerGlowGraphInfo> info;
  if (useMaxSizeCompilation_) {
    std::unique_lock<std::shared_timed_mutex> wlock(graphInfoMapMutex);
    if (perGlowGraphInfoMap_.size() != 1) {
      return MAKE_ERR(strFormat(
          "There should be one and only one compiled graph, but got %lu",
          perGlowGraphInfoMap_.size()));
    }
    info = perGlowGraphInfoMap_.begin()->second;
  } else {
    size_t hash = hashTensorStack(stack, graph_->inputs().size(), this);
    std::unique_lock<std::shared_timed_mutex> wlock(graphInfoMapMutex);
    auto it = perGlowGraphInfoMap_.find(hash);
    if (it == perGlowGraphInfoMap_.end()) {
      return MAKE_ERR(strFormat("No compiled graph found for hash: %lu", hash));
    }
    info = it->second;
  }

  std::unique_ptr<ExecutionContext> ctx = glow::make_unique<ExecutionContext>();
  TraceContext *traceContext = nullptr;
  if (getSettings().enableGlowTracing) {
    ctx->setTraceContext(glow::make_unique<TraceContext>(TraceLevel::STANDARD));
    traceContext = ctx->getTraceContext();
    traceContext->setThreadName("torch_glow");
  }
  TRACE_EVENT_BEGIN(traceContext, TraceLevel::RUNTIME, "torch_glow::runOnly");
  auto err = runImpl(*DCHECK_NOTNULL(info.get()), stack, ctx);

  // Reset the traceContext again in case it was changed during run.
  traceContext = ctx->getTraceContext();

  TRACE_EVENT_END(traceContext, TraceLevel::RUNTIME, "torch_glow::runOnly");

  aggregateAndDumpTraces(traceContext);
  return err;
}

void CachingGraphRunner::initializeCompiliationContextFromSettings(
    glow::CompilationContext &cctx, const PyTorchLoaderSettings &settings) {
  if (settings.convertToFP16) {
    cctx.precisionConfig.convertToFP16 = settings.convertToFP16;
    LOG(INFO) << "Conversion to fp16 enabled";
  }
  if (settings.convertPlaceholdersToFP16) {
    cctx.precisionConfig.convertPlaceholdersToFP16 =
        settings.convertFusedToFP16;
    LOG(INFO) << "Conversion of Placeholders to fp16 enabled";
  }
  if (settings.convertConstantsToFP16) {
    cctx.precisionConfig.convertConstantsToFP16 =
        settings.convertConstantsToFP16;
    LOG(INFO) << "Conversion of Constants to fp16 enabled";
  }
  if (settings.convertFusedToFP16) {
    cctx.precisionConfig.convertFusedToFP16 = settings.convertFusedToFP16;
    LOG(INFO) << "Conversion of fused scales/offsets to fp16 enabled";
  }
  if (settings.clipFP16) {
    cctx.precisionConfig.clipFP16 = settings.clipFP16;
    LOG(INFO) << "Clipping to fp16 enabled";
  }
  if (settings.clipFP16SkipInputs) {
    cctx.precisionConfig.clipFP16SkipInputs = settings.clipFP16SkipInputs;
    LOG(INFO) << "Skipping clipping for fp16 Node inputs fp16";
  }
  if (settings.forceFP16AccumSLS) {
    cctx.precisionConfig.forceFP16AccumSLS = settings.forceFP16AccumSLS;
    LOG(INFO) << "Forcing all SLS/SLWS ops to use FP16 accumulation enabled";
  }

  if (settings.dumpFinalGlowGraph) {
    cctx.dumpFinalGraph = settings.dumpFinalGlowGraph;
  }
  if (settings.saturateHost) {
    cctx.saturateHost = settings.saturateHost;
  }
}

Error CachingGraphRunner::warmCache(const std::vector<InputMeta> &inputMeta,
                                    const PyTorchLoaderSettings &settings,
                                    bool useMaxSizeCompilation) {
  if (!hostManager_) {
    return MAKE_ERR("Host manager is null!");
  }

  if (!graph_) {
    return MAKE_ERR("No graph found!");
  }

  std::unique_ptr<TraceContext> traceContext;
  if (settings.enableGlowTracing) {
    traceContext = std::make_unique<TraceContext>(TraceLevel::STANDARD);
    traceContext->setThreadName("torch_glow");
  }
  TRACE_EVENT_BEGIN(traceContext.get(), TraceLevel::RUNTIME,
                    "torch_glow::warmCache");

  // If this setting is missing we will not use pre compiled model at runtime,
  // which will cause unexpected behaviors.
  if (!settings.preCompilePyTorchModule) {
    return MAKE_ERR(
        "Calling AOT compilation when preCompilePyTorchModule is not set");
  }

  // hash should be unique in the following mappings:
  // 1) perGlowGraphInfoMap_ - a specifc instance of a runner corresponds to
  //    a single graph (i.e. Glow fusion group) that may have multiple
  //    Glow functions to serve different input shapes.
  // 2) HostManager mapping a functionName to a Glow function.
  // The input-based hash is combined with the pointer to this runner to
  // produce a unique mapping for HostManager.
  size_t hash;
  if (useMaxSizeCompilation) {
    useMaxSizeCompilation_ = true;
    hash = reinterpret_cast<size_t>(this);
  } else {
    useMaxSizeCompilation_ = false;
    torch::jit::Stack fakeStack = createFakeStackFromInputMeta(inputMeta);
    hash = hashTensorStack(fakeStack, inputMeta.size(), this);
  }
  {
    std::unique_lock<std::shared_timed_mutex> wlock(graphInfoMapMutex);
    if (perGlowGraphInfoMap_.find(hash) != perGlowGraphInfoMap_.end()) {
      return MAKE_ERR(
          strFormat("There is already a compiled graph for hash: %lu", hash));
    }
  }
  auto info = std::make_shared<PerGlowGraphInfo>(
      strFormat("pt_function_%lu", hash), settings);

  std::unique_ptr<Module> glowModule = std::make_unique<Module>();
  Function *f = glowModule->createFunction(info->functionName);

  TRACE_EVENT_BEGIN(traceContext.get(), TraceLevel::RUNTIME, "loadJITGraph");
  RETURN_IF_ERR(PyTorchModelLoader::loadJITGraph(
      *f, *graph_, info->inputPlaceholders, info->outputPlaceholders,
      outputCorrectType_, info->settings, {}, inputMeta));
  TRACE_EVENT_END(traceContext.get(), TraceLevel::RUNTIME, "loadJITGraph");

  // Obtain maxSeqLength from inputMeta
  // This step can also be done with a user input, but the overhead of the
  // following code should not be significant
  for (auto meta : inputMeta) {
    maxSeqLength_ =
        std::max(maxSeqLength_,
                 (size_t)std::accumulate(meta.dims.begin(), meta.dims.end(), 1,
                                         std::multiplies<glow::dim_t>()));
  }
  // Allocate zeroLengthSequence with maximum size
  // Similar to the impl in Onnxifi/Base.cpp
  glow::Type zt(ElemKind::Int64ITy, {maxSeqLength_});
  zeroLengthSequence_.reset(zt);
  zeroLengthSequence_.zero();

  glow::CompilationContext cctx;
  initializeCompiliationContextFromSettings(cctx, settings);

  TRACE_EVENT_BEGIN(traceContext.get(), TraceLevel::RUNTIME, "addNetwork");
  RETURN_IF_ERR(hostManager_->addNetwork(std::move(glowModule), cctx));
  TRACE_EVENT_END(traceContext.get(), TraceLevel::RUNTIME, "addNetwork");

  // There should be only one element in the map when model is precompiled.
  perGlowGraphInfoMap_.emplace(hash, info);

  TRACE_EVENT_END(traceContext.get(), TraceLevel::RUNTIME,
                  "torch_glow::warmCache");

  aggregateAndDumpTraces(traceContext.get());
  return Error::success();
}

const PyTorchLoaderSettings &CachingGraphRunner::getSettings() const {
  return settings_;
}

CachingGraphRunner::CachingGraphRunner(
    std::shared_ptr<torch::jit::Graph> graph,
    std::shared_ptr<runtime::HostManager> hostManager,
    PyTorchLoaderSettings settings)
    : graph_(graph), ptGraphExecutor_(graph, "forward"),
      hostManager_(hostManager),
      backend_(*EXIT_ON_ERR(hostManager->getBackend())), settings_(settings) {
  mergedTraceContext_ = glow::make_unique<TraceContext>(TraceLevel::STANDARD);
}

CachingGraphRunner::~CachingGraphRunner() {
  // Dump trace for the last time if there are remaining
  aggregateAndDumpTraces(nullptr, true);

  // Remove Glow functions saved in HostManager when being destroyed.
  std::unique_lock<std::shared_timed_mutex> wlock(graphInfoMapMutex);
  for (auto &kv : perGlowGraphInfoMap_) {
    ERR_TO_BOOL(hostManager_->removeNetwork(kv.second->functionName));
  }
}

} // namespace glow
