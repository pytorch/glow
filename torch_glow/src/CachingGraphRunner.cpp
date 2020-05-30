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

#include "glow/Exporter/ONNXModelWriter.h"
#include "glow/Support/Support.h"

#include <mutex>
#include <torch/csrc/jit/runtime/argument_spec.h>
#include <torch/csrc/utils/hash.h>

namespace glow {
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

void CachingGraphRunner::aggregateAndDumpTraces(TraceContext *traceContext,
                                                bool flush) {
  if (!traceContext && !flush) {
    return;
  }

  size_t numTracesPerDump = settings_.numTracesPerDump;

  std::unique_lock<std::mutex> lock(tracesMutex_);

  auto numTraces = ++numTraces_;

  if (flush) {
    if (mergedTraceContext_) {
      size_t dumpNum = numTraces / numTracesPerDump;
      std::string filename = strFormat("glow-trace-%zu.json", dumpNum);
      if (traceContext) {
        mergedTraceContext_->merge(traceContext);
      }
      mergedTraceContext_->dump(filename);
      mergedTraceContext_ = nullptr;
    }
    return;
  }

  if (!mergedTraceContext_) {
    mergedTraceContext_ = glow::make_unique<TraceContext>(TraceLevel::STANDARD);
  }

  mergedTraceContext_->merge(traceContext);

  if (numTraces % numTracesPerDump == 0) {
    size_t dumpNum = (numTraces / numTracesPerDump) - 1;
    std::string filename = strFormat("glow-trace-%zu.json", dumpNum);
    mergedTraceContext_->dump(filename);
    mergedTraceContext_ = nullptr;
  }
}

Expected<std::shared_ptr<CachingGraphRunner::PerGlowGraphInfo>>
CachingGraphRunner::loadImpl(torch::jit::Stack &stack,
                             TraceContext *traceContext) {
  if (settings_.preCompilePyTorchModule) {
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
  {
    std::shared_lock<std::shared_timed_mutex> rlock(graphInfoMapMutex);
    auto it = perGlowGraphInfoMap_.find(hash);
    if (it != perGlowGraphInfoMap_.end()) {
      return it->second;
    }
  }

  std::unique_lock<std::shared_timed_mutex> wlock(graphInfoMapMutex);
  auto it = perGlowGraphInfoMap_.find(hash);
  if (it != perGlowGraphInfoMap_.end()) {
    return it->second;
  }
  auto info = std::make_shared<PerGlowGraphInfo>();
  info->functionName = strFormat("pt_function_%lu", hash);

  std::unique_ptr<Module> module = glow::make_unique<Module>();
  Function *f = module->createFunction(info->functionName);

  TRACE_EVENT_BEGIN(traceContext, TraceLevel::RUNTIME, "loadJITGraph");
  RETURN_IF_ERR(PyTorchModelLoader::loadJITGraph(
      *f, *graph_, info->inputPlaceholders, info->outputPlaceholders,
      outputCorrectType_, getPyTorchLoaderSettings(), inputs, {}));
  TRACE_EVENT_END(traceContext, TraceLevel::RUNTIME, "loadJITGraph");

  glow::CompilationContext cctx;

  if (settings_.convertToFP16) {
    cctx.precisionConfig.convertToFP16 = true;
    cctx.precisionConfig.precisionModeKindSet.insert(
        Kinded::Kind::ChannelwiseQuantizedConvolutionNodeKind);
    cctx.precisionConfig.precisionModeKindSet.insert(
        Kinded::Kind::RowwiseQuantizedFullyConnectedNodeKind);
  }
  cctx.replicationCount = settings_.replicationCount;

  if (settings_.dumpFinalGlowGraph) {
    cctx.dumpFinalGraph = true;
  }

  cctx.backendOpts.backendSpecificOpts = settings_.backendSpecificOpts;

  TRACE_EVENT_BEGIN(traceContext, TraceLevel::RUNTIME, "addNetwork");
  // If --load-backend-specific-opts was passed from python, add it to the
  // compile context so the host manager knows to load backend options from
  // yaml.

  if (!settings_.backendOptionsFile.empty()) {
    std::pair<std::string, std::string> loadBackendSpecificOpts(
        "loadBackendSpecificOptions", settings_.backendOptionsFile);
    cctx.backendOpts.backendSpecificOpts.insert(loadBackendSpecificOpts);
  }

  cctx.replicationCount = settings_.replicationCount;
  cctx.saturateHost = settings_.saturateHost;
  RETURN_IF_ERR(hostManager_->addNetwork(std::move(module), cctx));

  TRACE_EVENT_END(traceContext, TraceLevel::RUNTIME, "addNetwork");

  auto ret = perGlowGraphInfoMap_.emplace(hash, info);
  CHECK(ret.second);

  return ret.first->second;
}

void CachingGraphRunner::runOnJit(torch::jit::Stack &stack) {
  static std::mutex runJitLock;
  std::lock_guard<std::mutex> guard(runJitLock);
  bool temp = getPyTorchLoaderSettings().fusionPassEnabled;
  getPyTorchLoaderSettings().fusionPassEnabled = false;
  ptGraphExecutor_.run(stack);
  getPyTorchLoaderSettings().fusionPassEnabled = temp;
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

Error CachingGraphRunner::runImpl(const PerGlowGraphInfo &info,
                                  torch::jit::Stack &stack,
                                  std::unique_ptr<ExecutionContext> &ctx) {
  size_t runId = numRuns_++;

  // Run the subgraph using JIT for comparison with Glow.
  torch::jit::Stack copyStack;
  if (settings_.writeToOnnx || settings_.jitVsGlowCompare) {
    for (auto &ival : stack) {
      if (ival.isTensor()) {
        copyStack.push_back(ival.deepcopy());
      } else {
        copyStack.push_back(ival);
      }
    }
    runOnJit(copyStack);
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
    } else if (!(input.isInt() || input.isIntList())) {
      return MAKE_ERR(
          "Only Int/IntList, Tensor and Object IValue inputs are accepted");
    }
  }

  if (settings_.writeToOnnx) {
    std::string filename = strFormat("input_%zu.onnx", runId);
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

  auto err = hostManager_->runNetworkBlocking(info.functionName, ctx);

  // Reset the traceContext again in case it was changed during run.
  traceContext = ctx->getTraceContext();

  TRACE_EVENT_END(traceContext, TraceLevel::RUNTIME, "runNetwork");
  TRACE_EVENT_BEGIN(traceContext, TraceLevel::RUNTIME, "setOutputs");

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

    if (settings_.writeToOnnx) {
      glow::Tensor glowT = ptTensorToGlowTensor(ptTensor);
      auto *onnxT = outputG.add_initializer();
      onnxT->set_name(info.outputPlaceholders[i]->getName());
      ONNXModelWriter::writeTensor(glowT, onnxT, /*useGlowCustomOps*/ true);
    }

    if (settings_.writeToOnnx) {
      auto &jitOutput = torch::jit::peek(copyStack, i, outputs.size());
      auto jitPtTensor = jitOutput.toTensor().contiguous();
      glow::Tensor jitGlowT = ptTensorToGlowTensor(jitPtTensor);
      auto *jitOnnxT = jitOutputG.add_initializer();
      jitOnnxT->set_name(info.outputPlaceholders[i]->getName());
      ONNXModelWriter::writeTensor(jitGlowT, jitOnnxT,
                                   /*useGlowCustomOps*/ true);
    }

    if (settings_.jitVsGlowCompare) {
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

    auto var = torch::autograd::make_variable(ptTensor);
    stack.push_back(at::IValue(var));
  }

  if (settings_.writeToOnnx) {
    std::string filename = strFormat("glow_output_%zu.onnx", runId);
    std::ofstream of(filename, std::ios::binary);
    if (!of) {
      LOG(ERROR) << "Cannot create output file " << filename;
    } else {
      std::string buffer;
      outputG.SerializeToString(&buffer);
      of << buffer;
    }
  }

  if (settings_.writeToOnnx) {
    std::string filename = strFormat("pytorch_output_%zu.onnx", runId);
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
  ASSIGN_VALUE_OR_RETURN_ERR(info, loadImpl(stack, traceContext));
  auto err = runImpl(*DCHECK_NOTNULL(info.get()), stack, ctx);

  // Reset the traceContext again in case it was changed during run.
  traceContext = ctx->getTraceContext();

  TRACE_EVENT_END(traceContext, TraceLevel::RUNTIME, "torch_glow::run");

  aggregateAndDumpTraces(traceContext);

  return err;
}

Error CachingGraphRunner::runOnly(torch::jit::Stack &stack) {
  std::shared_ptr<PerGlowGraphInfo> info;
  {
    std::shared_lock<std::shared_timed_mutex> rlock(graphInfoMapMutex);
    if (perGlowGraphInfoMap_.size() != 1) {
      return MAKE_ERR(strFormat(
          "There should be one and only one compiled graph, but got %lu",
          perGlowGraphInfoMap_.size()));
    }
    info = perGlowGraphInfoMap_.at(0);
  }

  std::unique_ptr<ExecutionContext> ctx = glow::make_unique<ExecutionContext>();
  return runImpl(*DCHECK_NOTNULL(info.get()), stack, ctx);
}

Error CachingGraphRunner::warmCache(const std::vector<InputMeta> &inputMeta) {
  if (!hostManager_) {
    return MAKE_ERR("Host manager is null!");
  }

  if (!graph_) {
    return MAKE_ERR("No graph found!");
  }

  // If this setting is missing we will not use pre compiled model at runtime,
  // which will cause unexpected behaviors.
  if (!settings_.preCompilePyTorchModule) {
    return MAKE_ERR(
        "Calling AOT compilation when preCompilePyTorchModule is not set");
  }

  std::unique_lock<std::shared_timed_mutex> wlock(graphInfoMapMutex);
  if (perGlowGraphInfoMap_.size() != 0) {
    return MAKE_ERR(strFormat("There is already a compiled graph!"));
  }

  auto info = std::make_shared<PerGlowGraphInfo>();
  info->functionName = strFormat("PTFunction_precompiled");

  std::unique_ptr<Module> glowModule = llvm::make_unique<Module>();
  Function *f = glowModule->createFunction(info->functionName);

  RETURN_IF_ERR(PyTorchModelLoader::loadJITGraph(
      *f, *graph_, info->inputPlaceholders, info->outputPlaceholders,
      outputCorrectType_, getPyTorchLoaderSettings(), {}, inputMeta));

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

  cctx.precisionConfig.convertToFP16 = settings_.convertToFP16;
  cctx.precisionConfig.convertFusedToFP16 = settings_.convertFusedToFP16;
  cctx.dumpFinalGraph = settings_.dumpFinalGlowGraph;

  RETURN_IF_ERR(hostManager_->addNetwork(std::move(glowModule), cctx));
  // Randomly picked one key. There should be only one element in the map
  // when model is precompiled.
  perGlowGraphInfoMap_[0] = info;
  return Error::success();
}

const PyTorchLoaderSettings &CachingGraphRunner::getSettings() const {
  return settings_;
}

CachingGraphRunner::CachingGraphRunner(
    std::shared_ptr<torch::jit::Graph> graph,
    std::shared_ptr<runtime::HostManager> hostManager, const char *backendName,
    PyTorchLoaderSettings settings)
    : graph_(graph), ptGraphExecutor_(graph, "forward"),
      hostManager_(hostManager), backend_(hostManager->getBackend(backendName)),
      settings_(settings) {}

CachingGraphRunner::~CachingGraphRunner() {
  // Remove Glow functions saved in HostManager when being destroyed.
  std::unique_lock<std::shared_timed_mutex> wlock(graphInfoMapMutex);
  for (auto &kv : perGlowGraphInfoMap_) {
    ERR_TO_BOOL(hostManager_->removeNetwork(kv.second->functionName));
  }
}

} // namespace glow
