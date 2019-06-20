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
#ifndef GLOW_BACKENDS_HABANA_HABANA_H
#define GLOW_BACKENDS_HABANA_HABANA_H

#include "glow/Backend/Backend.h"
#include "glow/Backend/CompiledFunction.h"
#include "glow/IR/Instrs.h"
#include "glow/Quantization/Base/Base.h"

#include "synapse.h"

#include <condition_variable>
#include <mutex>
#include <queue>
#include <string>

/// Given a synStatus \p status, evaluates to llvm::Error::success() if status
/// is synSuccess and evaluates to an llvm::Error otherwise.
#define chk_make_err(status)                                                   \
  status == synSuccess                                                         \
      ? llvm::Error::success()                                                 \
      : MAKE_ERR(                                                              \
            strFormat("Expected synStatus be synSuccess (%d), instead got %d", \
                      synSuccess, status))

/// Given a synStatus \p status, returns an llvm::Error::success() if status is
/// synSuccess and returns an llvm::Error otherwise.
#define chk(status)                                                            \
  do {                                                                         \
    auto res = (status);                                                       \
    if (res != synSuccess) {                                                   \
      return chk_make_err(res);                                                \
    }                                                                          \
  } while (0)

/// Given a synStatus \p status, checks that status == synSuccess and kills the
/// the program if not.
#define chk_kill(status)                                                       \
  CHECK_EQ((status), synSuccess) << "Expected synStatus be synSuccess"

namespace glow {

/// Buffer used for storing input/output tensors for a
/// HabanaFunction.
class HabanaIOBuffer {
public:
  /// Constructor.
  HabanaIOBuffer(uint32_t deviceId, uint8_t *buffer,
                 const std::unordered_map<const Placeholder *, off_t> &offsets);
  /// Destructor.
  ~HabanaIOBuffer() = default;

  /// Prohibit copy, assignment, and moves.
  HabanaIOBuffer(const HabanaIOBuffer &src) = delete;
  HabanaIOBuffer &operator=(const HabanaIOBuffer &src) = delete;
  HabanaIOBuffer(HabanaIOBuffer &&src) = delete;
  HabanaIOBuffer &operator=(HabanaIOBuffer &&src) = delete;

  /// Get a pointer to the buffer at which to read/store Placeholder \p p.
  /// \returns a GlowErr if an error occurred.
  llvm::Expected<uint8_t *> get(const Placeholder *p) const;

private:
  /// The device that this buffer is located on.
  uint32_t deviceId_;
  /// Pointer to the start of the buffer. All placeholders in offsets_ are
  /// allocated contiguously starting at this address.
  uint8_t *buffer_;
  /// Offsets into buffer_ for different input/output Placeholders.
  const std::unordered_map<const Placeholder *, off_t> &offsets_;
};

/// A pool of HabanaIOBuffer objects that threads must "check
/// out" before they execute a HabanaFunction and "check in" after execution is
/// complete. To enable concurrency, this class can allocate several times the
/// required memory for all input/output Placeholders to allow multiple threads
/// to execute on the device at once.
class HabanaIOBufferPool {
public:
  /// Constructor.
  HabanaIOBufferPool(uint32_t deviceId, const PlaceholderList &inputs,
                     const PlaceholderList &outputs,
                     unsigned numBuffers = kDefaultNumBuffers);

  /// Destructor.
  ~HabanaIOBufferPool();

  /// Prohibit copy, assignment, and moves.
  HabanaIOBufferPool(const HabanaIOBufferPool &src) = delete;
  HabanaIOBufferPool &operator=(const HabanaIOBufferPool &src) = delete;
  HabanaIOBufferPool(HabanaIOBufferPool &&src) = delete;
  HabanaIOBufferPool &operator=(HabanaIOBufferPool &&src) = delete;

  /// Get a HabanaIOBuffer instance from the pool. This confers exclusive
  /// ownership to the calling thread.
  std::unique_ptr<HabanaIOBuffer> get();

  /// Return a HabanaIOBuffer instance to the pool. This returns exclusive
  /// ownership back to the pool.
  void put(std::unique_ptr<HabanaIOBuffer> buffer);

private:
  /// The device that the underlying buffer resides on.
  uint32_t deviceId_;
  /// Offsets for different input/output Placeholders. This is the same for all
  /// HabanaIOBuffer instances, so the real copy is kept here and all
  /// HabanaIOBuffer instances in the pool receive const references to this one.
  std::unordered_map<const Placeholder *, off_t> offsets_;
  /// The size of all input and output Placeholders provided during
  /// construction. This is the effective size of one HabanaIOBuffer in this
  /// pool.
  size_t perBufferSize_;
  /// The combined size of all HabanaIOBuffers in this pool (i.e. perBufferSize_
  /// * numBuffers_).
  size_t allBuffersSize_;
  /// Buffer that backs all of the HOmaIOBuffers in this pool. The first buffer
  /// starts at buffer_, the second at buffer_ + perBufferSize_, etc. The last
  /// *ends* at buffer_ + allBuffersSize_.
  uint8_t *buffer_;
  /// The number of buffers in the pool.
  unsigned numBuffers_{kDefaultNumBuffers};
  static constexpr unsigned kDefaultNumBuffers{10};

  /// Queue of HabanaIOBuffers and a mutex and condition variable for
  /// synchronized access.
  std::mutex mtx_;
  std::condition_variable cv_;
  std::queue<std::unique_ptr<HabanaIOBuffer>> ioBuffers_;
};

/// Wrapper class for synWaitHandle and all related state that must persist
/// until the handle is waited on.
class HabanaWaitHandle {
public:
  /// Constructors.
  HabanaWaitHandle();
  HabanaWaitHandle(uint32_t deviceId, synWaitHandle handle,
                   std::vector<EnqueueTensorInfo> &&inputInfo,
                   std::vector<EnqueueTensorInfo> &&outputInfo);
  /// Destructor.
  ~HabanaWaitHandle();

  /// Allow moves.
  HabanaWaitHandle(HabanaWaitHandle &&);
  HabanaWaitHandle &operator=(HabanaWaitHandle &&);

  /// Prohibit copy and assignment.
  HabanaWaitHandle(const HabanaWaitHandle &) = delete;
  HabanaWaitHandle &operator=(const HabanaWaitHandle &) = delete;

  /// Wait on the underlying handle. \returns true if wait succeeded, false
  /// otherwise.
  bool wait();

private:
  /// If true, the instance points to a valid handle. Used to ensure proper
  /// destruction in the event of moves.
  bool valid_;
  /// The device that the enqueue corresponding to the handle was performed on.
  uint32_t deviceId_;
  /// The underlying synWaitHandle.
  synWaitHandle handle_;
  /// Inputs passed to the enqueue call that generated handle_.
  std::vector<EnqueueTensorInfo> inputInfo_;
  /// Outputs passed to the enqueue call that generated handle_.
  std::vector<EnqueueTensorInfo> outputInfo_;
};

class HabanaBackend final : public Backend {
public:
  /// Constructor.
  HabanaBackend() = default;

  /// @name Backend methods.
  /// This is the implementation of the Backend interface.
  ///@{
  ~HabanaBackend() override = default;

  std::string getBackendName() const override { return getName(); }
  static std::string getName() { return "Habana"; }

  llvm::Expected<std::unique_ptr<CompiledFunction>>
  compile(Function *F, const BackendOptions &opts) const override;

  bool isOpSupported(const NodeInfo &NI) const override;

  bool shouldLower(const Node *N) const override;

  bool transformPostLowering(Function *F,
                             CompilationContext &cctx) const override;

  bool shouldShareBuffers() const override { return false; }

  bool supportsPartialTensors() const override { return true; }
  /// @}

  static bool isVersionBiggerEqualTo(std::string versionToCompare);
};

class HabanaBindings : public DeviceBindings {
public:
  HabanaBindings(uint32_t deviceId, uint64_t topologyId)
      : DeviceBindings(HabanaBackend::getName()), deviceId_(deviceId),
        topologyId_(topologyId) {}

  virtual ~HabanaBindings() {}

  std::unique_ptr<DeviceBindings> clone() override {
    return llvm::make_unique<HabanaBindings>(deviceId_, topologyId_);
  }

  uint32_t getDeviceId() const { return deviceId_; }

  uint64_t getTopologyId() const { return topologyId_; }

  HabanaWaitHandle &getHandle() { return handle_; }

  void setHandle(HabanaWaitHandle &&handle) { handle_ = std::move(handle); }

  HabanaIOBuffer *getIOBufferUnsafePtr() { return ioBuffer_.get(); }
  std::unique_ptr<HabanaIOBuffer> getIOBuffer() { return std::move(ioBuffer_); }

  void setIOBuffer(std::unique_ptr<HabanaIOBuffer> ioBuffer) {
    ioBuffer_ = std::move(ioBuffer);
  }

private:
  uint32_t deviceId_;
  uint64_t topologyId_;
  std::unique_ptr<HabanaIOBuffer> ioBuffer_;
  HabanaWaitHandle handle_;
};

class HabanaFunction final : public CompiledFunction {
public:
  /// Constructor.
  HabanaFunction(const runtime::RuntimeBundle &bundle,
                 const std::string &recipeName, Function *F);

  /// @name CompiledFunction interface
  ///@{
  ~HabanaFunction() override;

  const std::string &getRecipeName() const { return recipeName_; }

  llvm::Error execute(ExecutionContext *context) override;
  ///@}

  /// \returns the backend used to compile this function.
  std::string getCompileBackendName() const override { return "Habana"; }

  const PlaceholderList &getInputs() const { return inputs_; }

  const PlaceholderList &getOutputs() const { return outputs_; }

private:
  /// Build the list of input and output placeholders.
  void findIOPlaceholders(Function *F);

  std::string recipeName_;
  PlaceholderList inputs_;
  PlaceholderList outputs_;
  std::unordered_set<const Placeholder *> partialInputs_;
};

} // namespace glow

#endif // GLOW_BACKENDS_HABANA_HABANA_H
