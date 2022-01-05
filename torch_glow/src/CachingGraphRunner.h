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

#ifndef GLOW_TORCH_GLOW_SRC_CACHINGGRAPHRUNNER_H
#define GLOW_TORCH_GLOW_SRC_CACHINGGRAPHRUNNER_H

#include "PyTorchModelLoader.h"
#include "glow/Backend/BlockStreamBase.h"
#include "glow/Runtime/HostManager/HostManager.h"
#include "glow/Runtime/InputSanitizer.h"
#include "glow/Support/TensorPool.h"

#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/serialization/import.h>

#include "ShapeInferenceEngine.h"
#include <shared_mutex>

namespace glow {
/// For a given PyTorch JIT graph, this class is responsible for maintaining a
/// mapping from PyTorch input information to Glow Function used to run that
/// graph in Glow.
class CachingGraphRunner {
public:
  /// Information that is stored per-Glow graph for running it using
  /// HostManager.
  struct PerGlowGraphInfo {

    PerGlowGraphInfo() = delete;
    PerGlowGraphInfo(const std::string &func,
                     PyTorchLoaderSettings settingsParam)
        : functionName(func), settings(std::move(settingsParam)) {}

    PerGlowGraphInfo(const PerGlowGraphInfo &) = delete;
    PerGlowGraphInfo &operator=(const PerGlowGraphInfo &) = delete;

    /// Input and output placeholders to the Glow function.
    std::vector<glow::Placeholder *> inputPlaceholders;
    std::vector<glow::Placeholder *> outputPlaceholders;

    /// Input sanitizers that should help prevent passing invalid
    /// inputs to the backend.
    std::vector<runtime::InputSanitizerPtr> inputSanitizers;

    /// Name of the Glow function maintained by HostManager for this subgraph.
    std::string functionName;

    /// PyTorchLoaderSettings used to compile this function
    PyTorchLoaderSettings settings;
  };

private:
  /// The PyTorch JIT Graph that this CachingGraphRunner caches Glow functions
  /// for.
  std::shared_ptr<torch::jit::Graph> graph_;

  /// The PyTorch JIT Graph that this CachingGraphRunner caches for before
  /// any preprocessing is done. Used for running on JIT later.
  std::shared_ptr<torch::jit::Graph> origGraph_;

  /// GraphExecutor used to execute graph_ on PyTorch for debugging purposes.
  torch::jit::GraphExecutor ptGraphExecutor_;

  /// The PyTorch module of the graph.
  /// It is used as first input when running origGraph_ on JIT.
  c10::IValue module_;

  /// The HostManager used to store and run Glow graphs.
  std::shared_ptr<runtime::HostManager> hostManager_;

  /// The Backend associated with the HostManager
  const Backend &backend_;

  /// Tensorpool to host padded tensors
  TensorPool tensorPool_;

  /// Mapping from hash of PyTorch inputs to PerGlowGraphInfo for the Glow
  /// function that will run inputs matching that hash.
  std::unordered_map<size_t, std::shared_ptr<PerGlowGraphInfo>>
      perGlowGraphInfoMap_;

  /// Mapping from hash of PyTorch inputs to PyTorchLoaderSettings for the Glow
  /// function that will run inputs matching that hash.
  std::unordered_map<size_t, PyTorchLoaderSettings> pyTorchLoaderSettingsMap_;

  /// Here we assume this is only one corresponding Glow function.
  /// Mapping from hash of PyTorch inputs to PerGlowGraphShape for the Glow
  /// function that will run inputs matching that hash.
  std::unordered_map<size_t, MetaStack> perGlowGraphShapeMap_;

  /// Mutex that protects perGlowGraphShapeMap_.
  std::mutex glowGraphShapeMapMutex_;

  /// In AOT flow, compile a single Glow function and use it for all input
  /// sizes. The PyTorch tensor inputs in this case should be smaller that the
  /// compiled inputs, and they'll be padded with zeros by Glow.
  bool useMaxSizeCompilation_ = true;

  /// Indicate which type will propagate to output.
  /// It is supposely to be the correct PyTorch ScalarType
  /// in the corresponding JIT node for each output
  /// placeholder from Glow graph.
  /// Use for quantization int8/uint8 rescale.
  std::vector<at::ScalarType> outputCorrectTypes_;

  /// Mutex that protects numTraces_ and mergedTraceContext_.
  std::mutex tracesMutex_;

  /// The number of runs traced
  size_t numTraces_{0};

  /// The number of trace dumps already generated
  size_t numTraceDumps_{0};

  /// TraceContext used to aggregate traces from runs before dumping them
  /// in groups to file.
  std::unique_ptr<TraceContext> mergedTraceContext_;

  /// Lock for concurrent accessing to perGlowGraphInfoMap_.
  std::shared_timed_mutex graphInfoMapMutex;

  /// The number of times any Glow graph managed by this CachingGraphRunner has
  /// been run.
  std::atomic<size_t> numRuns_{0};

  /// The maximum size of input tensors, to allocate the zerolength tensor
  size_t maxSeqLength_ = 1;

  /// Pre-allocated tensor for zero-length tensors, to avoid repeated zero
  /// paddings during ExecutionContext building
  glow::Tensor zeroLengthSequence_;

  /// Settings used when compiling and running Glow graphs in cases where a
  /// PyTorchLoaderSettings object isn't provided directly like when compiling
  /// on the fly for new input shapes.
  PyTorchLoaderSettings defaultSettings_;

  /// If true will call runOnly which skips input hashing and other costs.
  bool useRunOnly_ = false;

  int nominalInputIndex_ = -1;

  /// Given a PyTorch input stack \p stack, this generates a hash from the
  /// values on the stack and checks to see if a matching function was loaded
  /// previously. If a matching function was loaded previously then its cached
  /// info is returned immediately. Otherwise this loads the
  /// subgraph into the owned HostManager, creates a PerGlowGraphInfo which is
  /// cached for the given inputs, and then \returns this PerGlowGraphInfo.
  Expected<std::shared_ptr<PerGlowGraphInfo>>
  loadImpl(torch::jit::Stack &stack, const PyTorchLoaderSettings &settings,
           TraceContext *traceContext);

  /// Given a PyTorch inputs \p inputs, this generates a hash from the input
  /// shape and checks to see if the graph output shape with the given input
  /// was loaded previously. If was loaded previously then its output shape
  /// info is returned immediately. Otherwise this will run the shape inference
  /// engine, push the generated the output shape into \p perGlowGraphShapeMap_,
  /// and then \returns the output shape pointer.
  Expected<MetaStack *> loadShape(const c10::ArrayRef<c10::IValue> &inputs,
                                  TraceContext *traceContext);

  /// Given a PerGlowGraphInfo \p info for a subgraph that was previously
  /// loaded, this runs the Glow function that corresponds to that
  /// PerGlowGraphInfo in the shape of the inputs with the given \p stack with
  /// the given ExecutionContext \p ctx.
  Error runImpl(const PerGlowGraphInfo &info, torch::jit::Stack &stack,
                std::unique_ptr<ExecutionContext> &ctx);

  /// Run the graph_ on \p stack on using ptGraphExecutor_. This is for
  /// debugging purposes only. \returns how long running took in usecs.
  int64_t runOnJit(torch::jit::Stack &stack);

  /// Given a TraceContext \p traceContext, aggregate it with previous
  /// TraceContexts and if enough have been aggregated according to settings
  /// then dump them to file. If flush is true then dump aggregated traces to
  /// file no matter what.
  void aggregateAndDumpTraces(TraceContext *traceContext, bool flush = false);

  /// Converts PyTorch input tensor \p ptTensor to a Glow input tensor for the
  /// Placeholder \p ph. \returns the pair of the created Glow tensor and
  /// PyTorch tensor which may have been created in order to make it the
  /// original PyTorch tensor more suitable for Glow for example by making it
  /// contiguous. The new PyTorch tensor owns the memory used by the Glow tensor
  /// so much live at least as long as it.
  Expected<std::pair<glow::Tensor, torch::Tensor>>
  convertPyTorchInputToGlowInput(torch::Tensor &&ptTensor,
                                 const glow::Placeholder *ph);

  /// Calls convertPyTorchInputToGlowInput for several \p inputs and \p
  /// inputPlaceholders.
  Expected<std::pair<std::vector<glow::Tensor>, std::vector<torch::Tensor>>>
  processPyTorchInputs(at::ArrayRef<at::IValue> inputs,
                       const std::vector<Placeholder *> &inputPlaceholders);

  /// The Glow Function should've already been created. Returns an error if not.
  Error runOnly(torch::jit::Stack &stack);

  /// Get key of caching graph map from inputMetaStack.
  size_t getGraphMapKeyFromInputStack(const InputMetaStack &metaStack);

  /// Find the PerGlowGraphInfo corresponding to a given \p stack.
  Expected<std::shared_ptr<PerGlowGraphInfo>>
  findGraphInfoForStack(const torch::jit::Stack &stack);

public:
  CachingGraphRunner(std::shared_ptr<torch::jit::Graph> graph,
                     std::shared_ptr<runtime::HostManager> hostManager,
                     PyTorchLoaderSettings settings, bool useRunOnly = false,
                     std::shared_ptr<torch::jit::Graph> origGraph = nullptr,
                     c10::IValue module = c10::IValue());

  ~CachingGraphRunner();

  /// Given a PyTorch Stack \p stack of inputs, run he stored PyTorch graph on
  /// those inputs. If this is the first time this PyTorch graph has been run
  /// with inputs matching the hash of those on the stack then this first loads
  /// it as a Glow Function and compiles. \returns error of failure.
  Error run(torch::jit::Stack &stack);

  /// Warm up the cache by compiling one Glow function per metaStack and storing
  /// its info in perGlowGraphInfoMap_ with the hash computed using metaStack in
  /// \p metaStacks. Each metaStack in \p metaStacks is used to pass Glow shapes
  /// and types (Only tensors are valid inputs) for one Glow function. \p
  /// settings enable different settings for each compilation. If \p
  /// useMaxSizeCompilation , compile only a single Glow graph with an
  /// upper-bound on the input sizes (smaller inputs will be padded by Glow.)
  /// \p glowAOTSerializationSpecStrPtr and \p glowAOTSerializationModelStrPtr
  /// are used in offline Glow AOT compilation (i.e., Glow serialization), while
  /// \p serializationSpec and \p onnxModelFile are used for online serving
  /// (i.e., Glow deserialization)
  Error warmCache(
      const std::vector<InputMetaStack> &metaStacks,
      const PyTorchLoaderSettings &settings,
      runtime::DeferredWeightLoader *loader, bool useMaxSizeCompilation = true,
      bool useDeserialize = false,
      std::shared_ptr<std::unordered_map<std::string, std::vector<char>>>
          nameToFunctions = nullptr,
      std::shared_ptr<std::string> glowAOTSerializationSpecStrPtr = nullptr,
      std::shared_ptr<std::string> glowAOTSerializationModelStrPtr = nullptr,
      const std::string &serializationSpec = "",
      const std::string &onnxModelFile = "",
      const c10::optional<ModelCompilationConfigOverride>
          &modelCompilationConfigOverride = c10::nullopt);

  /// Warmup Graphoutput shape Map by getting output value shapes for each
  /// batch size.
  Error warmupGraphOutputShapeMap(
      const c10::ArrayRef<torch::jit::Value *> &graphOutputValues,
      const BatchShapesMapType &graphShapeMetaMap);

  /// Set nominalInputIndex from graphInputs.
  Error setNominalInputIndex(
      const c10::ArrayRef<torch::jit::Value *> &graphInputValues,
      const BatchShapesMapType &graphShapeMetaMap);

  /// Get nominalInputIndex.
  int getNominalInputIndex();

  /// Writes PyTorch tensor inputs on the \p stack to file \p inputFilePrefix,
  /// then runs the JIT GraphExecutor to get the outputs and writes those to \p
  /// outputFilePrefix.
  Error writeJitIOToOnnxFile(const std::string &inputFilePrefix,
                             const std::string &outputFilePrefix,
                             const torch::jit::Stack &stack);

  /// Get all serialized function maps run in backend.
  /// Each time it is called, it will refresh the map.
  std::unique_ptr<
      std::unordered_map<std::string, std::unique_ptr<BlockStreamBase>>>
  getAllSerializedFunctionsMap();
};

} // namespace glow

#endif // GLOW_TORCH_GLOW_SRC_CACHINGGRAPHRUNNER_H
