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

#include "HabanaFunction.h"
#include "HabanaUtils.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

#include "synapse.h"

namespace glow {

HabanaIOBuffer::HabanaIOBuffer(
    uint32_t deviceId, uint8_t *buffer,
    const std::unordered_map<const Placeholder *, off_t> &offsets)
    : deviceId_(deviceId), buffer_(buffer), offsets_(offsets) {}

Expected<uint8_t *> HabanaIOBuffer::get(const Placeholder *p) const {
  RETURN_ERR_IF_NOT(offsets_.count(p) > 0, "Placeholder not in IO buffer!");
  return buffer_ + offsets_.find(p)->second;
}

HabanaIOBufferPool::HabanaIOBufferPool(uint32_t deviceId,
                                       const PlaceholderList &inputs,
                                       const PlaceholderList &outputs,
                                       unsigned numBuffers)
    : deviceId_(deviceId), numBuffers_(numBuffers) {
  size_t currentOffset = 0;

  // Iterate through input Placeholders and assign offsets to each one.
  for (const auto &ph : inputs) {
    offsets_.insert(std::make_pair(ph, currentOffset));
    currentOffset += ph->getType()->getSizeInBytes();
  }

  // Iterate through output Placeholders and assign offsets to each one.
  for (const auto &ph : outputs) {
    offsets_.insert(std::make_pair(ph, currentOffset));
    currentOffset += ph->getType()->getSizeInBytes();
  }

  // Now that the total size of one buffer has been determined, allocate storage
  // for the whole pool.
  perBufferSize_ = currentOffset;
  allBuffersSize_ = perBufferSize_ * numBuffers_;
  chk_kill(synMalloc(deviceId_, allBuffersSize_, synMemFlags::synMemHost,
                     (void **)&buffer_));

  // Create HabanaIOBuffer instances for the pool with offsets into buffer_ that
  // are perBufferSize_ apart.
  uint8_t *copyOffset = buffer_;
  for (unsigned i = 0; i < numBuffers_; ++i) {
    ioBuffers_.push(
        glow::make_unique<HabanaIOBuffer>(deviceId_, copyOffset, offsets_));
    copyOffset += perBufferSize_;
  }
}

HabanaIOBufferPool::~HabanaIOBufferPool() {
  CHECK_EQ(ioBuffers_.size(), numBuffers_)
      << "IO buffer pool destroyed while some buffers still in use!";
  chk_kill(synFree(deviceId_, buffer_, synMemFlags::synMemHost));
}

std::unique_ptr<HabanaIOBuffer> HabanaIOBufferPool::get() {
  std::unique_lock<std::mutex> lk(mtx_);
  // If the queue of buffers is empty, wait until one is returned.
  cv_.wait(lk, [this]() { return !ioBuffers_.empty(); });
  std::unique_ptr<HabanaIOBuffer> buf(std::move(ioBuffers_.front()));
  ioBuffers_.pop();
  return buf;
}

void HabanaIOBufferPool::put(std::unique_ptr<HabanaIOBuffer> buffer) {
  std::lock_guard<std::mutex> lk(mtx_);
  bool wasEmpty = ioBuffers_.empty();
  ioBuffers_.push(std::move(buffer));
  // If the queue was empty before the push, threads might be waiting for a
  // buffer. Signal them.
  if (wasEmpty) {
    cv_.notify_all();
  }
}

HabanaWaitHandle::HabanaWaitHandle()
    : valid_(false), deviceId_(0), handle_(nullptr) {}

HabanaWaitHandle::HabanaWaitHandle(uint32_t deviceId, synWaitHandle handle,
                                   std::vector<EnqueueTensorInfo> &&inputInfo,
                                   std::vector<EnqueueTensorInfo> &&outputInfo)
    : valid_(true), deviceId_(deviceId), handle_(handle),
      inputInfo_(std::move(inputInfo)), outputInfo_(std::move(outputInfo)) {}

HabanaWaitHandle::~HabanaWaitHandle() {
  if (valid_) {
    synDestroyHandle(handle_);
    valid_ = false;
  }
}

HabanaWaitHandle::HabanaWaitHandle(HabanaWaitHandle &&o) {
  *this = std::move(o);
}

HabanaWaitHandle &HabanaWaitHandle::operator=(HabanaWaitHandle &&o) {
  std::swap(deviceId_, o.deviceId_);
  std::swap(handle_, o.handle_);
  inputInfo_ = std::move(o.inputInfo_);
  outputInfo_ = std::move(o.outputInfo_);
  std::swap(valid_, o.valid_);
  o.valid_ = false;
  return *this;
}

bool HabanaWaitHandle::wait() {
  if (!valid_) {
    return false;
  }

  synStatus status = synWaitForEvent(deviceId_, handle_);
  return status == synSuccess;
}

HabanaFunction::HabanaFunction(runtime::RuntimeBundle &&bundle,
                               const std::string &recipeName, Function *F)
    : CompiledFunction(std::move(bundle)), recipeName_(recipeName) {
  findIOPlaceholders(F);
}

void HabanaFunction::findIOPlaceholders(Function *F) {
  for (auto const &V : F->getParent()->getPlaceholders()) {
    if (!usedInFunction(V, F)) {
      continue;
    }
    if (getOutputSave(F, V)) {
      outputs_.push_back(V);
    } else {
      inputs_.push_back(V);
      if (allowsPartialInput(V, F)) {
        partialInputs_.insert(V);
      }
      if (allows64To32Downcast(V, F)) {
        downcastInt64Inputs_.insert(V);
      }
    }
  }
}

/// Retrieve and dump debug info about a topology.
static Error dumpTopologyInfo(uint32_t deviceId, uint64_t topologyId) {
  uint32_t numOfInputs;
  uint32_t numOfOutputs;
  uint32_t numOfIntermediates;
  chk(synGetIOTensorsAmount(deviceId, topologyId, numOfInputs, numOfOutputs,
                            numOfIntermediates));

  using TensorNames = char[ENQUEUE_TENSOR_NAME_MAX_SIZE];
  auto inputTensorNames = glow::make_unique<TensorNames[]>(numOfInputs);
  auto outputTensorNames = glow::make_unique<TensorNames[]>(numOfOutputs);
  auto intermediateTensorNames =
      glow::make_unique<TensorNames[]>(numOfIntermediates);

  chk(synGetTensorsName(deviceId, topologyId, inputTensorNames.get(),
                        numOfInputs, outputTensorNames.get(), numOfOutputs,
                        intermediateTensorNames.get(), numOfIntermediates));

  for (uint32_t i = 0; i < numOfInputs; i++) {
    VLOG(1) << "Topology input: " << inputTensorNames[i];
  }
  for (uint32_t i = 0; i < numOfOutputs; i++) {
    VLOG(1) << "Topology output: " << outputTensorNames[i];
  }
  for (uint32_t i = 0; i < numOfIntermediates; i++) {
    VLOG(1) << "Topology intermediates: " << intermediateTensorNames[i];
  }

  return Error::success();
}

HabanaFunction::~HabanaFunction() {
  CHECK(!llvm::sys::fs::remove(recipeName_))
      << "Failed to remove file at " << recipeName_;
  CHECK(!llvm::sys::fs::remove(recipeName_ + ".bin"))
      << "Failed to remove file at " << recipeName_ << ".bin";
}

Error HabanaFunction::execute(ExecutionContext *context) {
  auto *tc = context->getTraceContext();
  TRACE_EVENT_SCOPE_NAMED(tc, TraceLevel::RUNTIME, "execute", exEvent);
  exEvent.addArg("recipe", recipeName_);

  uint32_t deviceId =
      static_cast<HabanaBindings *>(context->getDeviceBindings())
          ->getDeviceId();
  uint64_t topologyId =
      static_cast<HabanaBindings *>(context->getDeviceBindings())
          ->getTopologyId();
  HabanaIOBuffer *ioBuffer =
      static_cast<HabanaBindings *>(context->getDeviceBindings())
          ->getIOBufferUnsafePtr();

  std::vector<EnqueueTensorInfo> inputInfo;
  std::vector<EnqueueTensorInfo> outputInfo;

  // Set up input buffers and record bindings for enqueuing.
  TRACE_EVENT_SCOPE_NAMED(tc, TraceLevel::RUNTIME, "copyInputs", ciEvent);
  size_t tensors{0}, bytes{0};
  auto *bindings = context->getPlaceholderBindings();
  for (auto *P : getInputs()) {
    Tensor *T = bindings->get(P);
    if (!T) {
      T = bindings->get(bindings->getPlaceholderByName(P->getName()));
    }
    tensors++;
    RETURN_ERR_IF_NOT(T, "Failed to get input tensor.");

    bool isPartial = partialInputs_.count(P);
    bool downcastInt64 = downcastInt64Inputs_.count(P);

    size_t elemSize =
        downcastInt64 ? sizeof(int32_t) : T->getType().getElementSize();
    size_t paddedSize = T->size() * elemSize;
    size_t unpaddedSize = T->getUnpaddedSizeInBytes();
    if (downcastInt64) {
      unpaddedSize /= 2;
    }

    EnqueueTensorInfo eti;
    llvm::StringRef name = P->getName();
    eti.tensorName = name.data();
    eti.tensorSize = paddedSize;
    uint8_t *ioBufferData;
    ASSIGN_VALUE_OR_RETURN_ERR(ioBufferData, ioBuffer->get(P));
    eti.pTensorData = (char *)ioBufferData;

    bytes += eti.tensorSize;

    inputInfo.push_back(eti);

    // Copy from the tensor into the designated IO buffer.
    if (downcastInt64) {
      // Copy int64 elements to int32.
      auto *device = reinterpret_cast<int32_t *>(eti.pTensorData);
      auto *host = reinterpret_cast<int64_t *>(T->getUnsafePtr());
      auto hostElems = unpaddedSize / sizeof(int32_t);
      for (size_t i = 0; i < hostElems; i++) {
        device[i] = host[i];
      }
    } else {
      memcpy(eti.pTensorData, T->getUnsafePtr(), unpaddedSize);
    }
    if (!isPartial) {
      memset(eti.pTensorData + unpaddedSize, 0, paddedSize - unpaddedSize);
    }
  }
  ciEvent.addArg("tensors", std::to_string(tensors));
  ciEvent.addArg("bytes", std::to_string(bytes));
  TRACE_EVENT_SCOPE_END_NAMED(ciEvent);

  TRACE_EVENT_SCOPE_NAMED(tc, TraceLevel::RUNTIME, "registerOutputs", roEvent);
  // Set up output buffers and record bindings for enqueuing.
  for (auto *P : getOutputs()) {
    Tensor *T = bindings->get(P);
    if (!T) {
      T = bindings->get(bindings->getPlaceholderByName(P->getName()));
    }
    RETURN_ERR_IF_NOT(T, "Failed to get output tensor.");

    EnqueueTensorInfo eti;
    llvm::StringRef name = P->getName();
    eti.tensorName = name.data();
    eti.tensorSize = T->getUnpaddedSizeInBytes();
    uint8_t *ioBufferData;
    ASSIGN_VALUE_OR_RETURN_ERR(ioBufferData, ioBuffer->get(P));
    eti.pTensorData = (char *)ioBufferData;

    outputInfo.push_back(eti);
  }

  EnqueueTensorInfo noInputEti = {"unused", (char *)nullptr, 0};
  TRACE_EVENT_SCOPE_END_NAMED(roEvent);

  // Enqueue the run and wait for it to come back.
  synWaitHandle handle;
  if (VLOG_IS_ON(1)) {
    RETURN_IF_ERR(dumpTopologyInfo(deviceId, topologyId));
  }
  TRACE_EVENT_SCOPE_NAMED(tc, TraceLevel::RUNTIME, "synEnqueue", seEvent);
  auto res = synEnqueueByName(
      deviceId, inputInfo.empty() ? &noInputEti : inputInfo.data(),
      inputInfo.size(), outputInfo.data(), outputInfo.size(), &handle);
  seEvent.addArg("result", statusStr(res));
  if (res != synSuccess) {
    return MAKE_ERR(strFormat("synEnqueueByName failed: %s", statusStr(res)));
  }
  TRACE_EVENT_SCOPE_END_NAMED(seEvent);

  static_cast<HabanaBindings *>(context->getDeviceBindings())
      ->setHandle(HabanaWaitHandle(deviceId, handle, std::move(inputInfo),
                                   std::move(outputInfo)));
  return Error::success();
}

} // namespace glow
