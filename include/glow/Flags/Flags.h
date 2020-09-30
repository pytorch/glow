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
namespace onnxifi {
extern int32_t GlowNumDevices;
extern int32_t GlowSparseNNPartitioningSchemeNumCards;
extern int64_t GlowSparseNNPartitioningSchemeSLSTableKBytesPerCard;
extern int32_t GlowSparseNNPartitioningSchemeNumCoresSLS;
extern int32_t GlowSparseNNPartitioningSchemeNumCoresOther;
extern bool GlowDumpDebugTraces;
extern int32_t GlowNumDebugTracesPerDump;
extern std::string GlowOnnxifiBackend;
extern bool GlowFP16;
extern bool GlowFP16Placeholders;
extern bool GlowFP16Constants;
extern bool GlowFusedScaleOffsetFP16;
extern bool GlowForceSLSAccumFP16;
extern bool GlowClipFP16;
extern bool GlowClipFP16SkipInputs;
extern bool GlowEnableQuantParamChanges;
extern bool GlowSaturateHost;
extern bool GlowSaveOnnxifiModel;
extern bool GlowSaveOnnxifiDAG;
extern bool GlowSaveOnnxifiIO;
extern bool GlowDelayAndRecordConstantModification;
extern bool GlowUseTrackedDummyQuantParams;
extern bool GlowEnablePartialTensors;
extern bool GlowUseCustomOpsForExport;
extern bool GlowUseSparseNNPartitioningScheme;
extern bool GlowSparseNNPartitioningAddSLSConcats;
extern bool GlowSparseNNPartitioningBalancePerfModel;
extern bool GlowSparseNNPartitioningPairLNWithSLS;
extern bool GlowDumpGraph;
extern std::string GlowDumpGraphPath;
extern bool GlowDumpInitialLoadedGraph;
extern bool GlowUseDAGOptimizer;
extern bool GlowUseDAGOptimizerAOT;
extern std::string GlowDAGOptimizerPlacementTaggingAlgorithm;
extern std::string GlowDAGOptimizerParallelizationTaggingAlgorithm;
extern int32_t GlowDAGOptimizerNumParallelChunks;
extern size_t GlowMaxActiveRequests;
extern size_t GlowMaxActiveRequestsPerInstance;
extern size_t GlowMaxQueueSize;
extern size_t GlowExecutorThreads;

extern bool GlowDumpNNPICompilerData;
extern bool GlowUsePerPartitionIcetConfig;
extern bool GlowDisableNNPITransforms;
extern bool GlowDisableNNPIPrivateTransforms;
extern int32_t GlowNNPINumParallelChunks;
extern int32_t GlowNNPIModelParallelSplitAlignment;

} // namespace onnxifi

extern bool GlowEnableLoadBalancedPartitioning;
extern bool GlowNNPILowerAllBatchMatMul;
extern bool GlowNNPIAcceptUnarySLS;
extern bool GlowNNPISpecializeAllOneSLS;

namespace runtime {
extern unsigned GlowInterpreterMemory;
extern unsigned GlowCPUMemory;
extern unsigned GlowHabanaMemory;
extern unsigned GlowNNPIMemory;
extern unsigned GlowNNPITimeout;
extern bool GlowEnableDRT;
extern bool GlowEnableP2P;
extern unsigned GlowDeviceInitTimeoutMs;
extern std::string GlowAvailableDevices;
} // namespace runtime

extern bool GlowDumpCompilationLog;
extern bool GlowDumpBackendSpecificIRJSON;
extern bool GlowLogPartition;
extern bool GlowDumpPartition;
} // namespace glow

/// Flags which may have their default values overridden:
DECLARE_bool(glow_global_fp16);
DECLARE_bool(glow_clip_fp16);
DECLARE_bool(glow_global_fused_scale_offset_fp16);
DECLARE_int32(glow_snn_partitioning_kbytes_per_card);
DECLARE_int32(glow_snn_partitioning_num_cores_sls);
DECLARE_int32(glow_snn_partitioning_num_cores_other);

#endif /* GLOW_FLAGS_FLAGS_H */
