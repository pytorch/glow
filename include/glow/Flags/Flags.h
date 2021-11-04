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
#ifndef GLOW_FLAGS_FLAGS_H
#define GLOW_FLAGS_FLAGS_H

#include "llvm/ADT/StringRef.h"
#include <gflags/gflags.h>
#include <map>

namespace glow {
namespace flags {

// Generic Constants
extern int32_t NumDevices;
extern bool ScanDevices;
extern bool SaturateHost;
extern bool EnableQuantParamChanges;
extern size_t MaxActiveRequests;
extern size_t MaxActiveRequestsPerInstance;
extern size_t MaxQueueSize;
extern size_t ExecutorThreads;
extern bool DelayAndRecordConstantModification;
extern bool UseTrackedDummyQuantParams;
extern bool EnablePartialTensors;
extern bool UseCustomOpsForExport;
extern std::string BackendSpecificOpts;
extern bool EnableLoadBalancedPartitioning;
extern bool SkipProvisioning;
extern bool DisableLayoutVerifying;
extern bool DisableFreeCompilationResource;
extern bool SinkTanhBelowConcat;

// FP16 Constants
extern bool ConvertToFP16;
extern bool SkipBiasFp32tofp16Convert;
extern bool ConvertPlaceholdersToFP16;
extern bool ConvertConstantsToFP16;
extern bool ConvertFusedScaleOffsetToFP16;
extern bool ClipToFP16;
extern bool SkipInputsOnClipToFP16;
extern bool ForceSLSToFP16Accum;
extern bool ClipZeroScaleFP16;
extern bool ClipQuantRangeToFP16;

// FP32 constants
extern bool ConvertFusedScaleOffsetToFP32;

// Debug Constants
extern int32_t NumDebugTracesPerDump;
extern bool DumpDebugTraces;
extern bool LogPartition;
extern bool DumpPartition;
extern bool DumpCompilationLog;
extern bool DumpBackendSpecificIRJSON;
extern bool DumpGraph;
extern std::string DumpGraphPath;
extern bool DumpInitialLoadedGraph;

// Sparse NN Partitioning Scheme Constants
extern int32_t SparseNNPartitioningSchemeNumCards;
extern int64_t SparseNNPartitioningSchemeSLSTableKBytesPerCard;
extern int32_t SparseNNPartitioningSchemeNumCoresSLS;
extern int32_t SparseNNPartitioningSchemeNumCoresOther;
extern bool UseSparseNNPartitioningScheme;
extern bool SparseNNPartitioningAddSLSConcats;
extern bool SparseNNPartitioningBalancePerfModel;
extern bool SparseNNPartitioningPairLNWithSLS;
extern bool SparseNNPartitioningPairTileWithSLS;
extern std::string SparseNNPartitioningPairSLSWith;
extern int32_t SparseNNPartitioningConcatSplitSize;
extern bool SparseNNParallelizeReshapeOnBatchDim;

// Dag Optimizer Constants
extern bool UseDAGOptimizer;
extern int32_t DAGOptimizerNumParallelChunks;
extern std::string DAGOptimizerPlacementTaggingAlgorithm;
extern std::string DAGOptimizerParallelizationTaggingAlgorithm;

/// Helper for processing opts in \p optsStr into \p opts. \returns if there is
/// any error encountered when processing \p optsStr.
bool processBackendSpecificOpts(std::map<std::string, std::string> &optsMap,
                                llvm::StringRef optsStr);
} // namespace flags
} // namespace glow

namespace glow {
namespace nnpi {
namespace flags {
extern int32_t ModelParallelSplitAlignment;
extern int32_t NumParallelChunks;
extern bool LowerAllBatchMatMul;
extern bool AcceptUnarySLS;
extern bool SpecializeAllOneSLS;
extern bool DisableTransforms;
extern bool EnableCustomIAKernels;
extern bool EnableCustomDSPKernels;
extern bool DumpCompilerData;
extern bool UsePerPartitionIcetConfig;
extern std::string InjectedIAOpKernelPath;
extern bool DumpCustomKernelFiles;
} // namespace flags
} // namespace nnpi
} // namespace glow

namespace glow {
namespace interpreter {
namespace flags {
extern bool LowerBatchMatMul;
extern bool LowerLayerNormalization;
} // namespace flags
} // namespace interpreter
} // namespace glow

namespace glow {
namespace torch_glow {
namespace flags {
extern bool ImaginaryFlag; // Placeholder Flag
}
} // namespace torch_glow
} // namespace glow

namespace glow {
namespace onnxifi {
namespace flags {
extern std::string BackendName;
extern bool SaveModel;
extern bool SaveIO;
extern bool SaveDAG;
extern bool SaveDAGWithConstants;
extern bool SaveDAGInZipMode;
} // namespace flags
} // namespace onnxifi
} // namespace glow

namespace glow {
namespace runtime {
namespace flags {
extern unsigned CPUMemory;

extern unsigned HabanaMemory;

extern unsigned NNPIMemory;
extern unsigned NNPITimeoutMs;

extern std::string AvailableDevices;
extern unsigned InterpreterMemory;
extern bool EnableP2P;
extern bool EnableDRT;
extern unsigned DeviceInitTimeoutMs;
extern uint64_t BigTableThresholdBytes;
extern unsigned SanitizeInputsPercent;
extern unsigned NumCompilationThreads;
} // namespace flags
} // namespace runtime
} // namespace glow

/// Flags which may have their default values overridden:
DECLARE_bool(glow_global_fp16);
DECLARE_bool(glow_skip_bias_fp32tofp16_convert);
DECLARE_bool(glow_clip_fp16);
DECLARE_bool(glow_global_fused_scale_offset_fp16);
DECLARE_bool(glow_global_fused_scale_offset_fp32);
DECLARE_int32(glow_snn_partitioning_kbytes_per_card);
DECLARE_int32(glow_snn_partitioning_num_cores_sls);
DECLARE_int32(glow_snn_partitioning_num_cores_other);

/// Signifiers for flags which we may load from e.g. a proto.
constexpr char clipQuantRangeToFP16Key[] = "GlowLoader_clipQuantRangeToFP16";

#endif /* GLOW_FLAGS_FLAGS_H */
