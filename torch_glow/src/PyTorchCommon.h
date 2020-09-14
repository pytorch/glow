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

#ifndef GLOW_TORCH_GLOW_SRC_COMMON_H
#define GLOW_TORCH_GLOW_SRC_COMMON_H

#include "glow/Base/Tensor.h"
#include "glow/Base/Type.h"
#include "glow/Importer/CommonOperatorLoader.h"
#include "glow/Runtime/HostManager/HostManager.h"

#include <torch/csrc/jit/ir/ir.h>

DECLARE_bool(dumpFinalGlowGraph);

namespace glow {

struct InputMeta;

/// Various settings to be used by code that loads PyTorch models. There should
/// only be one of these and it should be obtained by calling
/// getPyTorchLoaderSettings().
struct PyTorchLoaderSettings : public torch::jit::CustomClassHolder {
public:
  PyTorchLoaderSettings();
  PyTorchLoaderSettings(torch::Dict<std::string, std::string> dict);
  ~PyTorchLoaderSettings() {}

  void initSettings();
  std::string toString() const;

  torch::Dict<std::string, std::string> serializeToDict() const;

  /// This should be used with CachingGraphRunner::warmCache. When this flag is
  /// enabled, it assumes the glow graph is compiled ahead of time instead of
  /// at PyTorch JIT runtime. And the registered glow operator will run
  /// the precompiled results directly.
  bool preCompilePyTorchModule = false;

  /// Whether or not run the custom pass that fuses jit nodes into a glow node.
  bool fusionPassEnabled = false;

  /// Dump Glow dot graph to file after model loading is finished.
  bool dumpGlowDag = false;

  /// A list of symbols for nodes that will be ignored by the Glow fuser and
  /// thus will not be fused to Glow.
  std::unordered_set<torch::jit::Symbol> opBlacklist;

  /// The minimum size of a glow fusion groups in terms of number of PyTorch
  /// nodes. 0 indicates no minimum size.
  size_t minFusionGroupSize = 0;

  /// The maximum total number of nodes which are allowed to merge when
  /// fusing groups. The resulting group may be larger than this limit
  /// however as additional nodes may be inserted during the merge.
  size_t maxFusionMergeSize = 0;

  /// Index (inclusive) of the first node in the JIT graph to fuse. Ignored if
  /// negative.
  /// NOTE: this should only be used for debugging.
  int64_t fusionStartIndex = -1;

  /// Index (exclusive) of the last node in the JIT graph to fuse. Ignored if
  /// negative.
  /// NOTE: this should only be used for debugging.
  int64_t fusionEndIndex = -1;

  /// Convert fp32 opts to fp16 ops during Glow compilation.
  bool convertToFP16 = false;
  bool get_convert_to_fp16() { return convertToFP16; }
  void set_convert_to_fp16(bool val) { convertToFP16 = val; }

  /// Convert fp32 fused opts to fp16 ops during Glow compilation.
  bool convertFusedToFP16 = false;
  bool get_convert_fused_to_fp16() { return convertFusedToFP16; }
  void set_convert_fused_to_fp16(bool val) { convertFusedToFP16 = val; }

  /// Add clip operators after each fp16 ops during Glow compilation.
  bool clipFP16 = false;

  /// Force glow to skip clipping fp16 Node inputs to min/max
  bool clipFP16SkipInputs = true;

  /// Enable fp16 conversion for Placeholders
  bool convertPlaceholdersToFP16 = true;

  /// Enable fp16 conversion for Constants
  bool convertConstantsToFP16 = true;

  /// Force all SLS/SLWS ops to use FP16 accumulation.
  bool forceFP16AccumSLS = true;

  /// Dump Glow dot graph to file after Glow compilation is finished.
  bool dumpFinalGlowGraph = false;

  /// Enable tracing inside of Glow.
  bool enableGlowTracing = false;

  /// Enable the auto removal of muation in JIT graph, i.e, inline ops.
  bool enableRemoveMutation = true;

  /// Number of traces per json trace file dump.
  size_t numTracesPerDump = 1;

  /// Replication count of a graph on a device.
  size_t replicationCount = 1;
  int64_t get_replication_count() {
    return static_cast<int64_t>(replicationCount);
  }
  void set_replication_count(int64_t val) { replicationCount = val; }

  /// Backend-specific options to be put into the CompilationContext and passed
  /// to the Glow backend.
  std::map<std::string, std::string> backendSpecificOpts;

  /// Whether or not to write the loaded Glow function and inputs and outputs to
  /// and from the function to file as ONNX graphs.
  bool writeToOnnx = false;

  /// Whether or not to use zip mode when writing graphs to ONNX files
  bool onnxZipMode = false;

  /// Whether or not to do a numerical comparions of Glow and jit outputs
  bool jitVsGlowCompare = false;

  /// Name of a YAML file containing backend specific options.
  std::string backendOptionsFile;

  /// Whether not to set the saturateHost flag (use all available device) when
  /// adding networks to HostManager.
  bool saturateHost = false;
  bool get_saturate_host() { return saturateHost; }
  void set_saturate_host(bool val) { saturateHost = val; }

  /// If true then randomize the Constants in the Function loaded by
  /// PyTorchModelLoader.
  bool randomizeConstants = false;
  bool get_randomize_constants() { return randomizeConstants; }
  void set_randomize_constants(bool val) { randomizeConstants = val; }

  /// Name of the Glow backend to use.
  std::string backendName = "Interpreter";
  std::string get_backend_name() { return backendName; }
  void set_backend_name(std::string name) { backendName = name; }

  /// Number of Glow devices to use.
  int32_t numDevices = -1;

  // Whether to run shape inference of meta input
  bool runShapeInference = false;

  /// Run Fusion flow within to_glow compile function
  /// TODO: move to GlowCompileSpec
  bool enableDebugFuser = false;

  /// Whether to enforce module conversion to set include_last_offset for all
  /// embedding-bag-like operators. This is default to true since it is
  /// currently a requirement if we want to support partial inputs
  bool setIncludeLastOffsets = true;

  /// infer shape for entire model and run AOT compilation
  bool inferShapeForCompilation = false;
};

/// Given a PyTorch ScalarType \p ty, \returns a matching Glow ElemKind.
ElemKind scalarTypeToElemKind(c10::ScalarType ty);

// Given a Glow ElemKind \p ty, \returns a matching PyTorch ScalarType.
c10::ScalarType elemKindToScalarType(glow::ElemKind ty);

/// Given a c10 typekind \p ty, \returns a matching Glow ElemKind.
ElemKind typeKindToElemKind(c10::TypeKind ty);

/// \returns the PyTorchLoaderSettings singleton to be used throughout Glow's
/// PyTorch model loading code.
PyTorchLoaderSettings &getPyTorchLoaderSettings();

/// \returns the HostManager singleton used to run all PyTorch graphs in Glow.
std::shared_ptr<runtime::HostManager> getHostManager();

/// \returns the HostManager singleton used to run all PyTorch graphs with for
/// the Glow backend \p backendName. The HostManager will have \p numDevices
/// devices. If a previous HostManager is actively being used with the same
/// backend but a different number of devices then this is an error. If
/// numDevices is -1 then the active HostManager for the given backend will be
/// returned, if no active HostManager is found then a HostManager with 1 device
/// will be returned.
std::shared_ptr<runtime::HostManager>
getHostManager(const std::string &backendName, int32_t numDevices = -1);

/// \returns the PyTorch symbol to be used for the PyTorch node which represents
/// the subgraph that Glow will compile and run. \p g is the PyTorch graph to
/// lower, and if specified, will be used to generate unique symbol
c10::Symbol getGlowSymbol(std::shared_ptr<torch::jit::Graph> g = nullptr);

/// Given a PyTorch TensorType \p ptType, \returns a matching Glow Type.
glow::Type ptTypeToGlowType(const c10::TensorType &ptType);

/// Given a PyTorch Tensor \p ptTensor and a PyTorch scalar type \p dtype,
/// returns a new tensor which is \p ptTensor converted to \p dtype.
at::Tensor convertQuantizedToDtype(const at::Tensor ptTensor,
                                   c10::ScalarType dtype);

/// Given a PyTorch Tensor \p ptTensor, \returns an unowned Glow Tensor with a
/// matching type backed by the same memory as ptTensor.
glow::Tensor ptTensorToGlowTensor(const at::Tensor &ptTensor);

/// Given a Glow Type \p glowType, \returns an empty PyTorch Tensor with a
/// matching type.
at::Tensor glowTypeToEmptyPTTensor(const glow::Type &glowType);

/// Load the \p InputMeta data contains Glow fusion node's input size and type
/// info from \p raw_data stored in string format.
std::vector<glow::InputMeta> loadInputMeta(const std::string &raw_data);

/// Lower a pytorch \p module to glow before execution. \p inputMetaStr is the
/// raw string containing the meta data of the glow fuser node input.
void glowAOTFusion(torch::jit::Module &module, const std::string &inputMetaStr);

/// Lower a pytorch \p module to glow before execution. \p inputMeta is a
/// vector containing the meta data of the model inputs.
void glowAOTFusionWithShapeInference(torch::jit::Module &module,
                                     const std::vector<glow::InputMeta> &);

/// Enable overriding signal handlers while exeucting torch_glow code. This
/// should only be used in Python to enable easier debugging and not in
/// production C++ multithreaded environments. \p enable is used to enable or
/// disable overriding if set to false.
void enableSignalHandlerOverrides(bool enable = true);

/// \returns whether or not signal handler overriding is enabled.
bool signalHandlerOverridesEnabled();

} // namespace glow

#endif // GLOW_TORCH_GLOW_SRC_COMMON_H
