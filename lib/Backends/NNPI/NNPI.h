/*
 * Copyright (c) Glow Contributors. See CONTRIBUTORS file.
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

#ifndef GLOW_NNPI_BACKEND_H
#define GLOW_NNPI_BACKEND_H

#include "NNPIAdapterContainer.h"
#include "NNPIOptions.h"
#include "glow/Backend/Backend.h"
#include <folly/dynamic.h>
#include <vector>

namespace glow {

/// This is the Intel Neural-Network Processor for Inference (NNPI) backend.
class NNPIBackend final : public Backend {
public:
  /// Ctor.
  explicit NNPIBackend() = default;

  /// @name Backend methods.
  /// This is the implementation of the Backend interface.
  ///@{
  ~NNPIBackend() override = default;

  std::string getBackendName() const override { return getName(); }
  static std::string getName() { return "NNPI"; }
  static unsigned numDevices();
  static std::vector<unsigned> scanDeviceIDs();

  Expected<std::unique_ptr<CompiledFunction>>
  compile(Function *F, const BackendOptions &opts) const override;

#if FACEBOOK_INTERNAL
  Expected<std::unique_ptr<CompiledFunction>>
  compileFX(const folly::dynamic &FXIR, const std::string &submod,
            const llvm::StringMap<const void *> &constants,
            const BackendOptions &opts, Module *glowModule) const override;
#endif

  bool acceptForExecution(const NodeInfo &NI) const override;
  bool isOpSupported(const NodeInfo &NI) const override;
  bool shouldLower(const Node *N) const override;
  bool shouldShareBuffers() const override { return false; }
  bool supportsPartialTensors() const override { return true; }
  bool supportsStaticPlaceholders() const override { return true; }
  std::unique_ptr<FunctionPassPipeline>
  getOptimizationPipeline() const override;
  Expected<bool>
  transformPostOptPipeline(Function *F,
                           CompilationContext &cctx) const override;

  /// Helper to lower nodes which need further lowering. This is useful for when
  /// we need to lower Nodes based on precision, as we do lowering pre-precision
  /// transformation. \returns whether \p F was modified.
  bool lowerRequiredNodes(Function *F, CompilationContext &cctx) const;

  runtime::DeviceManager *
  createDeviceManager(const runtime::DeviceConfig &deviceConfig) override;

  Expected<bool> transformPostLowering(
      Function *F, CompilationContext &cctx,
      const glow::runtime::DeviceInfo *devInfo = nullptr) const override;

  virtual llvm::StringMap<std::string>
  getSupportedCompiledFunctionOptions() const override {
    NNPICompilationOptions options({});
    return options.getSupportedOptions();
  };

  virtual llvm::StringMap<std::string>
  getSupportedDeviceManagerOptions() const override {
    NNPIDeviceOptions options({});
    return options.getSupportedOptions();
  };

  virtual Error bindContexts(llvm::ArrayRef<runtime::ContextBinding> bindings,
                             const runtime::DAGNode *root, bool enableP2P,
                             bool enableDRT) override;

  /// Get the number of copies of inputs/outputs that will be reserved on a
  /// device per network.
  unsigned getContextCount(CompilationContext &cctx) const override;

  /// Swaps in specialized lookup tables for all supported nodes in \p F.
  /// \returns whether the function was modified.
  Expected<bool> swapInSpecializedLUT(Function *F,
                                      CompilationContext &cctx) const;

  /// Estimate performance cost for a given Node \p N.
  /// \returns a unitless value to be used when comparing to other estimates.
  /// or -1 if no estimate could be generated.
  /// SparseLength and EmbeddingBag type nodes are supported.
  double
  estimateEmbeddingNode(const glow::NodeInfo &NI, bool fp32Accumulation = false,
                        glow::LengthsMode lengthsMode = LengthsMode::Variable,
                        float averageLength = NAN) const;

#if NNPI_MAJOR_VERSION >= 1 && NNPI_MINOR_VERSION >= 7
  /// Estimate performance cost for a given BatchNorm Node \p BN.
  /// \returns a unitless value to be used when comparing to other estimates.
  /// or -1 if no estimate could be generated.
  double estimateBatchNormalizationNode(const BatchNormalizationNode *BN) const;

  /// Estimate performance cost for a given AvgPool Node \p avgPoolNode.
  /// \returns a unitless value to be used when comparing to other estimates.
  /// or -1 if no estimate could be generated.
  double estimateAvgPoolNode(const AvgPoolNode *avgPoolNode) const;

#endif // NNPI >= 1.7

#if NNPI_MAJOR_VERSION >= 1 && NNPI_MINOR_VERSION >= 8
  /// Estimate performance cost for a given LayerNorm Node \p LN.
  /// \returns a unitless value to be used when comparing to other estimates.
  /// or -1 if no estimate could be generated.
  double estimateLayerNormalizationNode(const LayerNormalizationNode *LN) const;

  /// Estimate performance cost for a given Binary Eltwise Node \p glowEltwise.
  /// \returns a unitless value to be used when comparing to other estimates.
  /// or -1 if no estimate could be generated.
  template <class EltwiseNodeType, NNPI_ELTWISE_TYPE eltwiseType>
  double estimateBinaryEltwiseNode(const glow::Node *glowEltwise) const;

  /// Estimate performance cost for a given Unary Eltwise Node \p glowEltwise.
  /// \returns a unitless value to be used when comparing to other estimates.
  /// or -1 if no estimate could be generated.
  template <class EltwiseNodeType, NNPI_ELTWISE_TYPE eltwiseType>
  double estimateUnaryEltwiseNode(const glow::Node *glowEltwise) const;

#endif // NNPI >= 1.8

  /// \returns a unitless value to be used when comparing Nodes or
  /// error if no estimate can be generated.
  Expected<double> estimateNodeCost(const glow::Node *node) const override;
  /// @}

private:
#if FACEBOOK_INTERNAL
  /// Performs FB-private transformations on \p F given \p cctx.
  /// \returns whether \p F is modified.
  bool transformPrivate(Function *F, CompilationContext &cctx) const;
#endif /* FACEBOOK_INTERNAL */

  static NNPIBackendOptions backendOptions_;
  static NNPIAdapterContainer adapter_;
};

/// These are used for parsing backend-specific node options.
constexpr char numParallelChunksKey[] = "NNPI_numParallelChunks";
constexpr char parallelTransformKindKey[] = "NNPI_parallelTransformKind";
constexpr char extraEdgesTargetNameKey[] = "NNPI_extraEdgesTargetName";
constexpr char extraEdgesTargetSuffixKey[] = "NNPI_extraEdgesTargetSuffix";
constexpr char extraEdgesSourceSuffixKey[] = "NNPI_extraEdgesSourceSuffix";
constexpr char coreAssignmentsKey[] = "NNPI_coreAssignments";
constexpr char coreAssignmentsSuffixKey[] = "NNPI_coreAssignmentsSuffix";
constexpr char tensorAssignmentNamesKey[] = "NNPI_tensorAssignmentNames";
constexpr char tensorAssignmentValuesKey[] = "NNPI_tensorAssignmentValues";

} // namespace glow
#endif // GLOW_NNPI_BACKEND_H
