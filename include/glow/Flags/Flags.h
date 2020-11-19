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

#include <gflags/gflags.h>

namespace glow {
namespace flags {

// Generic Constants
extern int32_t NumDevices;
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

// FP16 Constants
extern bool ConvertToFP16;
extern bool ConvertPlaceholdersToFP16;
extern bool ConvertConstantsToFP16;
extern bool ConvertFusedScaleOffsetToFP16;
extern bool ClipToFP16;
extern bool SkipInputsOnClipToFP16;
extern bool ForceSLSToFP16Accum;
extern bool ClipZeroScaleFP16;
extern bool ClipQuantRangeToFP16;

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

// Dag Optimizer Constants
extern bool UseDAGOptimizer;
extern bool UseDAGOptimizerAOT;
extern int32_t DAGOptimizerNumParallelChunks;
extern std::string DAGOptimizerPlacementTaggingAlgorithm;
extern std::string DAGOptimizerParallelizationTaggingAlgorithm;

} // namespace flags
} // namespace glow

#ifdef GLOW_WITH_NNPI

namespace glow {
namespace nnpi {
namespace flags {
extern int32_t ModelParallelSplitAlignment;
extern int32_t NumParallelChunks;
extern bool LowerAllBatchMatMul;
extern bool AcceptUnarySLS;
extern bool SpecializeAllOneSLS;
extern bool DisableTransforms;
extern bool DisablePrivateTransforms;
extern bool DumpCompilerData;
extern bool UsePerPartitionIcetConfig;
} // namespace flags
} // namespace nnpi
} // namespace glow

#endif

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
} // namespace flags
} // namespace onnxifi
} // namespace glow

namespace glow {
namespace runtime {
namespace flags {
#ifdef GLOW_WITH_CPU
extern unsigned CPUMemory;
#endif

#ifdef GLOW_WITH_HABANA
extern unsigned HabanaMemory;
#endif

#ifdef GLOW_WITH_NNPI
extern unsigned NNPIMemory;
extern unsigned NNPITimeoutMs;
#endif

extern std::string AvailableDevices;
extern unsigned InterpreterMemory;
extern bool EnableP2P;
extern bool EnableDRT;
extern unsigned DeviceInitTimeoutMs;
} // namespace flags
} // namespace runtime
} // namespace glow

/// Flags which may have their default values overridden:
DECLARE_bool(glow_global_fp16);
DECLARE_bool(glow_clip_fp16);
DECLARE_bool(glow_global_fused_scale_offset_fp16);
DECLARE_int32(glow_snn_partitioning_kbytes_per_card);
DECLARE_int32(glow_snn_partitioning_num_cores_sls);
DECLARE_int32(glow_snn_partitioning_num_cores_other);

/// Signifiers for flags which we may load from e.g. a proto.
constexpr char clipQuantRangeToFP16Key[] = "GlowLoader_clipQuantRangeToFP16";

#endif /* GLOW_FLAGS_FLAGS_H */
