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

#include "glow/Flags/Flags.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <map>

/* Flags should generally go in as specific of namespace as makes sense.
 *  That is, if a flag is specific to torch_glow, it should go in the
 * flags::torch_glow namespace. Flags that have a generic nature, but are not
 * supported in specific contexts, can go in a specific domain. An example is
 * AcceptUnarySLS living in the glow::nnpi::flags namespace, as that's the only
 * domain for which it is supported. In the same vein, it is encouraged to make
 * flags as generic as is possible.
 */
namespace glow {
namespace flags {

// Generic Constants
int32_t NumDevices = 1;
bool ScanDevices = false;
bool SaturateHost = false;
bool EnableQuantParamChanges = true;
size_t MaxActiveRequests = 48;
size_t MaxActiveRequestsPerInstance = 48;
size_t MaxQueueSize = 200;
size_t ExecutorThreads = 10;
bool DelayAndRecordConstantModification = false;
bool UseTrackedDummyQuantParams = false;
bool EnablePartialTensors = true;
bool UseCustomOpsForExport = true;
std::string BackendSpecificOpts = "";
bool EnableLoadBalancedPartitioning = true;
bool SkipProvisioning = false;
bool DisableLayoutVerifying = false;
bool DisableFreeCompilationResource = false;
bool SinkTanhBelowConcat = false;

// FP16 Constants
bool ConvertToFP16 = false;
bool SkipBiasFp32tofp16Convert = false;
bool ConvertPlaceholdersToFP16 = false;
bool ConvertConstantsToFP16 = true;
bool ConvertFusedScaleOffsetToFP16 = false;
bool ClipToFP16 = false;
bool SkipInputsOnClipToFP16 = true;
bool ForceSLSToFP16Accum = true;
bool ClipQuantRangeToFP16 = false;
bool ClipZeroScaleFP16 = false;

// Fp32 constants
bool ConvertFusedScaleOffsetToFP32 = false;

// Debug Constants
int32_t NumDebugTracesPerDump = 100;
bool DumpDebugTraces = false;
bool LogPartition = true;
bool DumpPartition = false;
bool DumpCompilationLog = false;
bool DumpBackendSpecificIRJSON = false;
bool DumpGraph = false;
std::string DumpGraphPath = "./";
bool DumpInitialLoadedGraph = false;

// Sparse NN Partitioning Scheme Constants
int32_t SparseNNPartitioningSchemeNumCards = 1;
int64_t SparseNNPartitioningSchemeSLSTableKBytesPerCard = 1;
int32_t SparseNNPartitioningSchemeNumCoresSLS = 1;
int32_t SparseNNPartitioningSchemeNumCoresOther = 1;
bool UseSparseNNPartitioningScheme = false;
bool SparseNNPartitioningAddSLSConcats = false;
bool SparseNNPartitioningBalancePerfModel = false;
bool SparseNNPartitioningPairLNWithSLS = false;
bool SparseNNPartitioningPairTileWithSLS = false;
std::string SparseNNPartitioningPairSLSWith = "";
int32_t SparseNNPartitioningConcatSplitSize = 1;
bool SparseNNParallelizeReshapeOnBatchDim = true;

// Dag Optimizer Constants
bool UseDAGOptimizer = false;
int32_t DAGOptimizerNumParallelChunks = 1;
std::string DAGOptimizerPlacementTaggingAlgorithm = "None";
std::string DAGOptimizerParallelizationTaggingAlgorithm = "None";

} // namespace flags
} // namespace glow

namespace glow {
namespace nnpi {
namespace flags {
int32_t ModelParallelSplitAlignment = 1;
int32_t NumParallelChunks = 0; // Zero val for an ugly hack in NNPI.cpp
bool LowerAllBatchMatMul = false;
bool AcceptUnarySLS = false;
bool SpecializeAllOneSLS = false;
bool DisableTransforms = false;
bool EnableCustomIAKernels = false;
bool EnableCustomDSPKernels = false;
bool DumpCompilerData = false;
bool UsePerPartitionIcetConfig = false;
std::string InjectedIAOpKernelPath = "";
bool DumpCustomKernelFiles = false;

} // namespace flags
} // namespace nnpi
} // namespace glow

namespace glow {
namespace interpreter {
namespace flags {
bool LowerBatchMatMul = true;
bool LowerLayerNormalization = true;
} // namespace flags
} // namespace interpreter
} // namespace glow

namespace glow {
namespace torch_glow {
namespace flags {
bool ImaginaryFlag = false; // Placeholder flag
}
} // namespace torch_glow
} // namespace glow

namespace glow {
namespace onnxifi {
namespace flags {
std::string BackendName = "";
bool SaveModel = false;
bool SaveIO = false;
bool SaveDAG = false;
bool SaveDAGWithConstants = false;
bool SaveDAGInZipMode = false;
} // namespace flags
} // namespace onnxifi
} // namespace glow

namespace glow {
namespace runtime {
namespace flags {

unsigned CPUMemory = 0;
unsigned HabanaMemory = 7 << 20;
unsigned NNPIMemory = 16 << 20;
unsigned NNPITimeoutMs = 0;

std::string AvailableDevices = "";
unsigned InterpreterMemory = 0;
bool EnableP2P = false;
bool EnableDRT = false;
unsigned DeviceInitTimeoutMs = 5000;
unsigned SanitizeInputsPercent = 0;
uint64_t BigTableThresholdBytes = 104857600; // 100MB
unsigned NumCompilationThreads = 1;
} // namespace flags
} // namespace runtime
} // namespace glow

/*
 * Note: Validators are used to assign instead of direct assignment because
 * direct assignment seems to result in a static order initialization fiasco.
 */
DEFINE_int32(glow_num_devices, glow::flags::NumDevices,
             "Number of devices for Glow backend");
DEFINE_validator(glow_num_devices, [](const char *, int32_t val) {
  glow::flags::NumDevices = val;
  return true;
});
DEFINE_bool(glow_scan_devices, glow::flags::ScanDevices,
            "Scan available devices for Glow backend");
DEFINE_validator(glow_scan_devices, [](const char *, bool val) {
  glow::flags::ScanDevices = val;
  return true;
});
DEFINE_int32(glow_snn_partitioning_num_cards,
             glow::flags::SparseNNPartitioningSchemeNumCards,
             "Number of devices to distribute tables across in SparseNN "
             "partitioning");
DEFINE_validator(glow_snn_partitioning_num_cards,
                 [](const char *, int32_t val) {
                   glow::flags::SparseNNPartitioningSchemeNumCards = val;
                   return true;
                 });
DEFINE_int32(glow_snn_partitioning_kbytes_per_card,
             glow::flags::SparseNNPartitioningSchemeSLSTableKBytesPerCard,
             "Bytes per card used for SLS tables in SparseNN partitioning");
DEFINE_validator(glow_snn_partitioning_kbytes_per_card, [](const char *,
                                                           int32_t val) {
  glow::flags::SparseNNPartitioningSchemeSLSTableKBytesPerCard = val;
  return true;
});
DEFINE_int32(
    glow_snn_partitioning_num_cores_sls,
    glow::flags::SparseNNPartitioningSchemeNumCoresSLS,
    "Number of cores to assign to SLS partition in SparseNN partitioning");
DEFINE_validator(glow_snn_partitioning_num_cores_sls,
                 [](const char *, int32_t val) {
                   glow::flags::SparseNNPartitioningSchemeNumCoresSLS = val;
                   return true;
                 });
DEFINE_int32(
    glow_snn_partitioning_num_cores_other,
    glow::flags::SparseNNPartitioningSchemeNumCoresOther,
    "Number of cores to assign to non-SLS partition in SparseNN partitioning");
DEFINE_validator(glow_snn_partitioning_num_cores_other,
                 [](const char *, int32_t val) {
                   glow::flags::SparseNNPartitioningSchemeNumCoresOther = val;
                   return true;
                 });
DEFINE_bool(glow_dump_debug_traces, glow::flags::DumpDebugTraces,
            "Dump traces to /tmp");
DEFINE_validator(glow_dump_debug_traces, [](const char *, bool val) {
  glow::flags::DumpDebugTraces = val;
  return true;
});
DEFINE_int32(glow_num_debug_traces_per_dump, glow::flags::NumDebugTracesPerDump,
             "Maximum number of traces in each debug dump.");
DEFINE_validator(glow_num_debug_traces_per_dump, [](const char *, int32_t val) {
  glow::flags::NumDebugTracesPerDump = val;
  return true;
});
DEFINE_string(glow_onnxifi_backend, glow::onnxifi::flags::BackendName,
              "Glow backend used for ONNXIFI");
DEFINE_validator(glow_onnxifi_backend,
                 [](const char *, const std::string &val) {
                   glow::onnxifi::flags::BackendName = val;
                   return true;
                 });
DEFINE_string(
    glow_available_devices, glow::runtime::flags::AvailableDevices,
    "Comma separated list of devices which should be used, example 2,3,4");
DEFINE_validator(glow_available_devices,
                 [](const char *, const std::string &val) {
                   glow::runtime::flags::AvailableDevices = val;
                   return true;
                 });
DEFINE_bool(glow_global_fp16, glow::flags::ConvertToFP16,
            "Enable fp16 lowering for all ops on the net");
DEFINE_validator(glow_global_fp16, [](const char *, bool val) {
  glow::flags::ConvertToFP16 = val;
  return true;
});
DEFINE_bool(glow_skip_bias_fp32tofp16_convert,
            glow::flags::SkipBiasFp32tofp16Convert,
            "Skip fp32 -> fp16 convertion for Bias in FC");
DEFINE_validator(glow_skip_bias_fp32tofp16_convert, [](const char *, bool val) {
  glow::flags::SkipBiasFp32tofp16Convert = val;
  return true;
});
DEFINE_bool(torch_glow_imaginary_flag, glow::torch_glow::flags::ImaginaryFlag,
            "Enable fp16 lowering for all ops on the net");
DEFINE_validator(torch_glow_imaginary_flag, [](const char *, bool val) {
  glow::torch_glow::flags::ImaginaryFlag = val;
  return true;
});
DEFINE_bool(glow_global_fp16_placeholders,
            glow::flags::ConvertPlaceholdersToFP16,
            "Enable fp16 conversion for Placeholders");
DEFINE_validator(glow_global_fp16_placeholders, [](const char *, bool val) {
  glow::flags::ConvertPlaceholdersToFP16 = val;
  return true;
});
DEFINE_bool(glow_global_fp16_constants, glow::flags::ConvertConstantsToFP16,
            "Enable fp16 conversion for Constants");
DEFINE_validator(glow_global_fp16_constants, [](const char *, bool val) {
  glow::flags::ConvertConstantsToFP16 = val;
  return true;
});
DEFINE_bool(glow_global_fused_scale_offset_fp16,
            glow::flags::ConvertFusedScaleOffsetToFP16,
            "Enable fp16 lowering for all op inputs using fused scale/offset");
DEFINE_validator(glow_global_fused_scale_offset_fp16,
                 [](const char *, bool val) {
                   glow::flags::ConvertFusedScaleOffsetToFP16 = val;
                   return true;
                 });
DEFINE_bool(
    glow_global_fused_scale_offset_fp32,
    glow::flags::ConvertFusedScaleOffsetToFP32,
    "Enable converting scale/offset in sls's input data from fp16 to fp32");
DEFINE_validator(glow_global_fused_scale_offset_fp32,
                 [](const char *, bool val) {
                   glow::flags::ConvertFusedScaleOffsetToFP32 = val;
                   return true;
                 });
DEFINE_bool(
    glow_global_force_sls_fp16_accum, glow::flags::ForceSLSToFP16Accum,
    "Force all SLS/SLWS ops to use FP16 accumulation. True by default.");
DEFINE_validator(glow_global_force_sls_fp16_accum, [](const char *, bool val) {
  glow::flags::ForceSLSToFP16Accum = val;
  return true;
});
DEFINE_bool(glow_enable_quant_param_changes,
            glow::flags::EnableQuantParamChanges,
            "Enable quantization param changes during optimizations");
DEFINE_validator(glow_enable_quant_param_changes, [](const char *, bool val) {
  glow::flags::EnableQuantParamChanges = val;
  return true;
});
DEFINE_bool(glow_use_sparsenn_partitioning_scheme,
            glow::flags::UseSparseNNPartitioningScheme,
            "Force glow to use SparseNN partitioning scheme");
DEFINE_validator(glow_use_sparsenn_partitioning_scheme,
                 [](const char *, bool val) {
                   glow::flags::UseSparseNNPartitioningScheme = val;
                   return true;
                 });
DEFINE_bool(glow_sparsenn_partitioning_add_sls_concats,
            glow::flags::SparseNNPartitioningAddSLSConcats,
            "Add extra concats inside of SLS partitions for more efficient "
            "inter-partitition transfers");
DEFINE_validator(glow_sparsenn_partitioning_add_sls_concats,
                 [](const char *, bool val) {
                   glow::flags::SparseNNPartitioningAddSLSConcats = val;
                   return true;
                 });
DEFINE_bool(glow_sparsenn_partitioning_balance_perf_model,
            glow::flags::SparseNNPartitioningBalancePerfModel,
            "Balance SLS tables across cards using a perf model");
DEFINE_validator(glow_sparsenn_partitioning_balance_perf_model,
                 [](const char *, bool val) {
                   glow::flags::SparseNNPartitioningBalancePerfModel = val;
                   return true;
                 });
DEFINE_bool(glow_sparsenn_partitioning_pair_ln_with_sls,
            glow::flags::SparseNNPartitioningPairLNWithSLS,
            "Put layer normalization nodes immediately following SLS into SLS "
            "Partitions");
DEFINE_validator(glow_sparsenn_partitioning_pair_ln_with_sls,
                 [](const char *, bool val) {
                   glow::flags::SparseNNPartitioningPairLNWithSLS = val;
                   return true;
                 });
DEFINE_bool(
    glow_sparsenn_partitioning_pair_tile_with_sls,
    glow::flags::SparseNNPartitioningPairTileWithSLS,
    "Put tile nodes immediately following SLS for user embeddings into SLS "
    "Partitions");
DEFINE_validator(glow_sparsenn_partitioning_pair_tile_with_sls,
                 [](const char *, bool val) {
                   glow::flags::SparseNNPartitioningPairTileWithSLS = val;
                   return true;
                 });
DEFINE_string(
    glow_sparsenn_partitioning_pair_sls_with,
    glow::flags::SparseNNPartitioningPairSLSWith,
    "Put nodes specified immediately following SLS into SLS partitions."
    "Supported for LayerNorm, Tile, Concat, and Tanh nodes"
    "Comma separated list of node names, e.g. LayerNorm,Tile.");
DEFINE_validator(glow_sparsenn_partitioning_pair_sls_with,
                 [](const char *, const std::string &val) {
                   glow::flags::SparseNNPartitioningPairSLSWith = val;
                   return true;
                 });
DEFINE_int32(glow_sparsenn_partitioning_concat_split_size,
             glow::flags::SparseNNPartitioningConcatSplitSize,
             "The number of inputs to split each concat to be moved into SLS "
             "partitions to");
DEFINE_validator(glow_sparsenn_partitioning_concat_split_size,
                 [](const char *, const int32_t val) {
                   glow::flags::SparseNNPartitioningConcatSplitSize = val;
                   return true;
                 });
DEFINE_bool(glow_sparsenn_parallelize_reshape_on_batch_dim,
            glow::flags::SparseNNParallelizeReshapeOnBatchDim,
            "Force parallelizing the reshape operators on the batch dimension");
DEFINE_validator(glow_sparsenn_parallelize_reshape_on_batch_dim,
                 [](const char *, bool val) {
                   glow::flags::SparseNNParallelizeReshapeOnBatchDim = val;
                   return true;
                 });
DEFINE_bool(glow_clip_fp16, glow::flags::ClipToFP16,
            "Force glow to clip fp16 values to min/max");
DEFINE_validator(glow_clip_fp16, [](const char *, bool val) {
  glow::flags::ClipToFP16 = val;
  return true;
});
DEFINE_bool(glow_clip_fp16_skip_inputs, glow::flags::SkipInputsOnClipToFP16,
            "Force glow to skip clipping fp16 Node inputs to min/max");
DEFINE_validator(glow_clip_fp16_skip_inputs, [](const char *, bool val) {
  glow::flags::SkipInputsOnClipToFP16 = val;
  return true;
});
DEFINE_bool(glow_saturate_host, glow::flags::SaturateHost,
            "Try to use all available devices on the host");
DEFINE_validator(glow_saturate_host, [](const char *, bool val) {
  glow::flags::SaturateHost = val;
  return true;
});
DEFINE_bool(
    glow_save_onnxifi_dag, glow::onnxifi::flags::SaveDAG,
    "Whether to serialize the DAG that has been optimized and partitioned.");
DEFINE_validator(glow_save_onnxifi_dag, [](const char *, bool val) {
  glow::onnxifi::flags::SaveDAG = val;
  return true;
});
DEFINE_bool(glow_save_onnxifi_dag_with_constants,
            glow::onnxifi::flags::SaveDAGWithConstants,
            "Whether to serialize constants in the DAG that has been optimized "
            "and partitioned.");
DEFINE_validator(glow_save_onnxifi_dag_with_constants,
                 [](const char *, bool val) {
                   glow::onnxifi::flags::SaveDAGWithConstants = val;
                   return true;
                 });
DEFINE_bool(glow_save_onnxifi_dag_in_zip_mode,
            glow::onnxifi::flags::SaveDAGWithConstants,
            "Whether to serialize the DAG that has been optimized and "
            "partitioned in ZIP mode.");
DEFINE_validator(glow_save_onnxifi_dag_in_zip_mode, [](const char *, bool val) {
  glow::onnxifi::flags::SaveDAGInZipMode = val;
  return true;
});
DEFINE_bool(
    glow_delay_and_record_constant_modification,
    glow::flags::DelayAndRecordConstantModification,
    "Whether to delay and record constant modification for serialization.");
DEFINE_validator(glow_delay_and_record_constant_modification,
                 [](const char *, bool val) {
                   glow::flags::DelayAndRecordConstantModification = val;
                   return true;
                 });
DEFINE_bool(glow_use_tracked_dummy_quant_params,
            glow::flags::UseTrackedDummyQuantParams,
            "Whether to use uniqued dummy quant params when loading the model, "
            "which are then mapped to loaded names for serialization.");
DEFINE_validator(glow_use_tracked_dummy_quant_params,
                 [](const char *, bool val) {
                   glow::flags::UseTrackedDummyQuantParams = val;
                   return true;
                 });
DEFINE_bool(glow_clip_zero_scale_fp16, glow::flags::ClipZeroScaleFP16,
            "Whether to clip qparam scales below 1/65504 to that val.");
DEFINE_validator(glow_clip_zero_scale_fp16, [](const char *, bool val) {
  glow::flags::ClipZeroScaleFP16 = val;
  return true;
});
DEFINE_bool(glow_clip_quant_range_to_fp16, glow::flags::ClipQuantRangeToFP16,
            "Whether to clip quantization parameters inside the fp16 range.");
DEFINE_validator(glow_clip_quant_range_to_fp16, [](const char *, bool val) {
  glow::flags::ClipQuantRangeToFP16 = val;
  return true;
});
DEFINE_int32(glow_max_active_requests, glow::flags::MaxActiveRequests,
             "Number of max active requests before host manager start queuing");
DEFINE_validator(glow_max_active_requests, [](const char *, int32_t val) {
  glow::flags::MaxActiveRequests = val;
  return true;
});
DEFINE_int32(glow_max_active_requests_per_instance,
             glow::flags::MaxActiveRequestsPerInstance,
             "Number of max active requests per instance of a network.");
DEFINE_validator(glow_max_active_requests_per_instance,
                 [](const char *, int32_t val) {
                   glow::flags::MaxActiveRequestsPerInstance = val;
                   return true;
                 });
DEFINE_int32(
    glow_max_queue_size, glow::flags::MaxQueueSize,
    "Max number of pending requeusts in glow's host manager queue before "
    "rejecting new request");
DEFINE_validator(glow_max_queue_size, [](const char *, int32_t val) {
  glow::flags::MaxQueueSize = val;
  return true;
});
DEFINE_int32(glow_executor_threads, glow::flags::ExecutorThreads,
             "Number of executor threads for host manager");
DEFINE_validator(glow_executor_threads, [](const char *, int32_t val) {
  glow::flags::ExecutorThreads = val;
  return true;
});
DEFINE_bool(glow_partitioner_enable_load_balance,
            glow::flags::EnableLoadBalancedPartitioning,
            "Enable a partitioner pass to optimize for load balance in "
            "addition to memory capacity constraints");
DEFINE_validator(glow_partitioner_enable_load_balance,
                 [](const char *, bool val) {
                   glow::flags::EnableLoadBalancedPartitioning = val;
                   return true;
                 });
DEFINE_bool(glow_skip_provisioning, glow::flags::SkipProvisioning,
            "Skip provisioning. Used for AOT opts or debugging.");
DEFINE_validator(glow_skip_provisioning, [](const char *, bool val) {
  glow::flags::SkipProvisioning = val;
  return true;
});
DEFINE_bool(glow_sink_tanh_below_concat, glow::flags::SinkTanhBelowConcat,
            "Sink tanh ops below concat.");
DEFINE_validator(glow_sink_tanh_below_concat, [](const char *, bool val) {
  glow::flags::SinkTanhBelowConcat = val;
  return true;
});
DEFINE_bool(glow_save_onnxifi_model, glow::onnxifi::flags::SaveModel,
            "Package the glow function and weights right before lowering");
DEFINE_validator(glow_save_onnxifi_model, [](const char *, bool val) {
  glow::onnxifi::flags::SaveModel = val;
  return true;
});
DEFINE_bool(glow_save_onnxifi_io, glow::onnxifi::flags::SaveIO,
            "Save the input and output result around ONNXIFI boundary");
DEFINE_validator(glow_save_onnxifi_io, [](const char *, bool val) {
  glow::onnxifi::flags::SaveIO = val;
  return true;
});
DEFINE_bool(glow_enable_partial_tensors, glow::flags::EnablePartialTensors,
            "Save the input and output result around ONNXIFI boundary");
DEFINE_validator(glow_enable_partial_tensors, [](const char *, bool val) {
  glow::flags::EnablePartialTensors = val;
  return true;
});
DEFINE_bool(glow_use_custom_ops_for_export, glow::flags::UseCustomOpsForExport,
            "Use custom ONNX ops when exporting Glow protos.");
DEFINE_validator(glow_use_custom_ops_for_export, [](const char *, bool val) {
  glow::flags::UseCustomOpsForExport = val;
  return true;
});
DEFINE_bool(glow_dump_graph, glow::flags::DumpGraph,
            "Dump the glow Graph into files before compilation");
DEFINE_validator(glow_dump_graph, [](const char *, bool val) {
  glow::flags::DumpGraph = val;
  return true;
});
DEFINE_string(glow_dump_graph_path, glow::flags::DumpGraphPath,
              "Directory path for the dumped graphs.");
DEFINE_validator(glow_dump_graph_path,
                 [](const char *, const std::string &val) {
                   glow::flags::DumpGraphPath = val;
                   return true;
                 });
DEFINE_bool(glow_dump_initial_loaded_graph, glow::flags::DumpInitialLoadedGraph,
            "Dump the glow Graph right after onnxification");
DEFINE_validator(glow_dump_initial_loaded_graph, [](const char *, bool val) {
  glow::flags::DumpInitialLoadedGraph = val;
  return true;
});
DEFINE_bool(glow_use_dag_optimizer, glow::flags::UseDAGOptimizer,
            "Whether to call the DAG optimizer");
DEFINE_validator(glow_use_dag_optimizer, [](const char *, bool val) {
  glow::flags::UseDAGOptimizer = val;
  return true;
});
DEFINE_int32(glow_dag_optimizer_num_parallel_chunks,
             glow::flags::DAGOptimizerNumParallelChunks,
             "Number of parallel chunks for DAGOptimizer parallelization");
DEFINE_validator(glow_dag_optimizer_num_parallel_chunks,
                 [](const char *, int32_t val) {
                   glow::flags::DAGOptimizerNumParallelChunks = val;
                   return true;
                 });
DEFINE_string(glow_dag_optimizer_placement_tagging_algorithm,
              glow::flags::DAGOptimizerPlacementTaggingAlgorithm,
              "Name of placement tagging algorithm to run in DAGOptimizer");
DEFINE_validator(glow_dag_optimizer_placement_tagging_algorithm,
                 [](const char *, const std::string &val) {
                   glow::flags::DAGOptimizerPlacementTaggingAlgorithm = val;
                   return true;
                 });

DEFINE_string(
    glow_dag_optimizer_parallelization_tagging_algorithm,
    glow::flags::DAGOptimizerParallelizationTaggingAlgorithm,
    "Name of parallelization tagging algorithm to run in DAGOptimizer");
DEFINE_validator(glow_dag_optimizer_parallelization_tagging_algorithm,
                 [](const char *, const std::string &val) {
                   glow::flags::DAGOptimizerParallelizationTaggingAlgorithm =
                       val;
                   return true;
                 });
// Defined in glow/lib/Backends/NNPI/NNPI.cpp
DEFINE_bool(glow_use_per_partition_icet_config,
            glow::nnpi::flags::UsePerPartitionIcetConfig,
            "Read an icet_config.json file for each partition");
DEFINE_validator(glow_use_per_partition_icet_config,
                 [](const char *, bool val) {
                   glow::nnpi::flags::UsePerPartitionIcetConfig = val;
                   return true;
                 });
DEFINE_bool(glow_dump_nnpi_compiler_data, glow::nnpi::flags::DumpCompilerData,
            "Dump the NNPI compiler data into files before NNPI compilation");
DEFINE_validator(glow_dump_nnpi_compiler_data, [](const char *, bool val) {
  glow::nnpi::flags::DumpCompilerData = val;
  return true;
});
DEFINE_bool(glow_nnpi_specialize_all_one_sls,
            glow::nnpi::flags::SpecializeAllOneSLS,
            "Whether to import SLS ops with AllOne attribute to NNPI.");
DEFINE_validator(glow_nnpi_specialize_all_one_sls, [](const char *, bool val) {
  glow::nnpi::flags::SpecializeAllOneSLS = val;
  return true;
});
DEFINE_bool(glow_disable_nnpi_transforms, glow::nnpi::flags::DisableTransforms,
            "Disable running NNPIBackend::transformPostLowering().");
DEFINE_validator(glow_disable_nnpi_transforms, [](const char *, bool val) {
  glow::nnpi::flags::DisableTransforms = val;
  return true;
});
DEFINE_bool(glow_enable_nnpi_custom_ia_kernels,
            glow::nnpi::flags::EnableCustomIAKernels,
            "Enable running NNPIBackend::transformPrivate().");
DEFINE_validator(glow_enable_nnpi_custom_ia_kernels,
                 [](const char *, bool val) {
                   glow::nnpi::flags::EnableCustomIAKernels = val;
                   return true;
                 });
DEFINE_bool(glow_enable_nnpi_custom_dsp_kernels,
            glow::nnpi::flags::EnableCustomDSPKernels,
            "Enable running NNPIBackend::transformPrivate().");
DEFINE_validator(glow_enable_nnpi_custom_dsp_kernels,
                 [](const char *, bool val) {
                   glow::nnpi::flags::EnableCustomDSPKernels = val;
                   return true;
                 });

DEFINE_string(glow_injected_ia_op_kernel_path,
              glow::nnpi::flags::InjectedIAOpKernelPath,
              "Path to IA kernels library to use");
DEFINE_validator(glow_injected_ia_op_kernel_path,
                 [](const char *, const std::string &val) {
                   glow::nnpi::flags::InjectedIAOpKernelPath = val;
                   return true;
                 });

DEFINE_bool(glow_dump_custom_kernel_files,
            glow::nnpi::flags::DumpCustomKernelFiles,
            "Enable dumping the compiled custom IA and DSP kernels to file.");
DEFINE_validator(glow_dump_custom_kernel_files, [](const char *, bool val) {
  glow::nnpi::flags::DumpCustomKernelFiles = val;
  return true;
});

DEFINE_bool(glow_nnpi_lower_all_batch_matmul,
            glow::nnpi::flags::LowerAllBatchMatMul,
            "Whether to override default lowering for NNPI and always lower "
            "BatchMatMul to a series of MatMuls.");
DEFINE_validator(glow_nnpi_lower_all_batch_matmul, [](const char *, bool val) {
  glow::nnpi::flags::LowerAllBatchMatMul = val;
  return true;
});
DEFINE_bool(glow_nnpi_accept_unary_sls, glow::nnpi::flags::AcceptUnarySLS,
            "Whether to accept unary SLS ops during ONNXIFI loading.");
DEFINE_validator(glow_nnpi_accept_unary_sls, [](const char *, bool val) {
  glow::nnpi::flags::AcceptUnarySLS = val;
  return true;
});
DEFINE_int32(glow_nnpi_num_parallel_chunks,
             glow::nnpi::flags::NumParallelChunks,
             "Number of parallel chunks for NNPI");
DEFINE_validator(glow_nnpi_num_parallel_chunks, [](const char *, int32_t val) {
  glow::nnpi::flags::NumParallelChunks = val;
  return true;
});
DEFINE_int32(glow_nnpi_model_parallel_split_alignment,
             glow::nnpi::flags::ModelParallelSplitAlignment,
             "Alignment value for model parallel splits");
DEFINE_validator(glow_nnpi_model_parallel_split_alignment,
                 [](const char *, int32_t val) {
                   glow::nnpi::flags::ModelParallelSplitAlignment = val;
                   return true;
                 });
DEFINE_int32(glow_nnpi_memory, glow::runtime::flags::NNPIMemory,
             "Amount of DRAM to allocate per NNPI device in KiB");
DEFINE_validator(glow_nnpi_memory, [](const char *, int32_t val) {
  glow::runtime::flags::NNPIMemory = val;
  return true;
});
DEFINE_int32(glow_nnpi_timeout_ms, glow::runtime::flags::NNPITimeoutMs,
             "Timeout threshold for inferecnce in milliseconds. Default 0 "
             "means infinity");
DEFINE_validator(glow_nnpi_timeout_ms, [](const char *, int32_t val) {
  glow::runtime::flags::NNPITimeoutMs = val;
  return true;
});

DEFINE_bool(glow_interpreter_lower_batch_matmul,
            glow::interpreter::flags::LowerBatchMatMul,
            "Lower batch matmul node.");
DEFINE_validator(glow_interpreter_lower_batch_matmul,
                 [](const char *, bool val) {
                   glow::interpreter::flags::LowerBatchMatMul = val;
                   return true;
                 });
DEFINE_bool(glow_interpreter_lower_layer_normalization,
            glow::interpreter::flags::LowerLayerNormalization,
            "Lower layer normalization node.");
DEFINE_validator(glow_interpreter_lower_layer_normalization,
                 [](const char *, bool val) {
                   glow::interpreter::flags::LowerLayerNormalization = val;
                   return true;
                 });

DEFINE_int32(glow_interpreter_memory, glow::runtime::flags::InterpreterMemory,
             "Amount of DRAM to allocate per Interpreter in KiB");
DEFINE_validator(glow_interpreter_memory, [](const char *, int32_t val) {
  glow::runtime::flags::InterpreterMemory = val;
  return true;
});
DEFINE_int32(glow_cpu_memory, glow::runtime::flags::CPUMemory,
             "Amount of DRAM to allocate per CPU in KiB");
DEFINE_validator(glow_cpu_memory, [](const char *, int32_t val) {
  glow::runtime::flags::CPUMemory = val;
  return true;
});

DEFINE_int32(glow_habana_memory, glow::runtime::flags::HabanaMemory,
             "Amount of DRAM to allocate per Habana device in KiB");
DEFINE_validator(glow_habana_memory, [](const char *, int32_t val) {
  glow::runtime::flags::HabanaMemory = val;
  return true;
});

DEFINE_int32(
    glow_num_compilation_threads, glow::runtime::flags::NumCompilationThreads,
    "Maximum number of threads to spawn per call to Backend::compileFunctions");
DEFINE_validator(glow_num_compilation_threads, [](const char *, int32_t val) {
  glow::runtime::flags::NumCompilationThreads = val;
  return true;
});

DEFINE_bool(glow_log_partition, glow::flags::LogPartition,
            "Enable logging partition info");
DEFINE_validator(glow_log_partition, [](const char *, bool val) {
  glow::flags::LogPartition = val;
  return true;
});
DEFINE_bool(glow_enable_p2p, glow::runtime::flags::EnableP2P,
            "Enable peer to peer support");
DEFINE_validator(glow_enable_p2p, [](const char *, bool val) {
  glow::runtime::flags::EnableP2P = val;
  return true;
});
DEFINE_bool(glow_enable_drt, glow::runtime::flags::EnableDRT,
            "Enable device resident tensor support");
DEFINE_validator(glow_enable_drt, [](const char *, bool val) {
  glow::runtime::flags::EnableDRT = val;
  return true;
});
DEFINE_int32(glow_device_init_timeout_ms,
             glow::runtime::flags::DeviceInitTimeoutMs,
             "Timeout threshold for device initialization in milliseconds. "
             "Default 5000");
DEFINE_validator(glow_device_init_timeout_ms, [](const char *, int32_t val) {
  glow::runtime::flags::DeviceInitTimeoutMs = val;
  return true;
});
DEFINE_uint64(
    glow_partition_big_table_threshold_bytes,
    glow::runtime::flags::BigTableThresholdBytes,
    "Threshold to determin big tables, and used in partitioning algorithm. "
    "Default 104857600(100MB)");
DEFINE_validator(glow_partition_big_table_threshold_bytes,
                 [](const char *, uint64_t val) {
                   glow::runtime::flags::BigTableThresholdBytes = val;
                   return true;
                 });
DEFINE_int32(glow_enable_sanitize_inputs,
             glow::runtime::flags::SanitizeInputsPercent,
             "Sanitize a percentage of inferences");
DEFINE_validator(glow_enable_sanitize_inputs, [](const char *, int32_t val) {
  if (val < 0 || val > 100) {
    return false;
  }

  glow::runtime::flags::SanitizeInputsPercent = val;
  return true;
});

DEFINE_bool(glow_dump_partition, glow::flags::DumpPartition,
            "Enable dumping the graph of each partition");
DEFINE_validator(glow_dump_partition, [](const char *, bool val) {
  glow::flags::DumpPartition = val;
  return true;
});
DEFINE_bool(glow_dump_compilation_log, glow::flags::DumpCompilationLog,
            "Dump the glow compilation log into /tmp during compilation");
DEFINE_validator(glow_dump_compilation_log, [](const char *, bool val) {
  glow::flags::DumpCompilationLog = val;
  return true;
});
DEFINE_bool(glow_dump_backend_specific_ir_json,
            glow::flags::DumpBackendSpecificIRJSON,
            "Dump the backend-specific IR JSON file");
DEFINE_validator(glow_dump_backend_specific_ir_json,
                 [](const char *, bool val) {
                   glow::flags::DumpBackendSpecificIRJSON = val;
                   return true;
                 });
DEFINE_string(glow_backend_specific_opts, glow::flags::BackendSpecificOpts,
              "Glow backend specific options. Comma separated list of "
              "key=value pairs, e.g. key1=val1,key2=val2.");
DEFINE_validator(glow_backend_specific_opts,
                 [](const char *, const std::string &val) {
                   glow::flags::BackendSpecificOpts = val;
                   return true;
                 });

bool glow::flags::processBackendSpecificOpts(
    std::map<std::string, std::string> &optsMap, llvm::StringRef optsStr) {
  if (optsStr.empty()) {
    return true;
  }
  llvm::SmallVector<llvm::StringRef, 4> splitOpts;
  optsStr.split(splitOpts, ',');

  for (const llvm::StringRef &opt : splitOpts) {
    LOG(INFO) << "Adding backend specific option: " << opt.str();
    auto keyValPair = opt.split('=');
    if (keyValPair.second.empty()) {
      LOG(ERROR) << "No '=' found in backend-specific opt " << opt.str();
      return false;
    }
    optsMap.emplace(keyValPair.first, keyValPair.second);
  }
  return true;
}

namespace {
llvm::cl::OptionCategory flagsLibCat("Glow Flags Lib CmdLine Options");
/// Allows enabling DRT support.
llvm::cl::opt<bool, /* ExternalStorage */ true>
    enableDRT("enable-DRT",
              llvm::cl::desc(
                  "Deprecated. Enabled DRT support. Alias to glow_enable_drt."),
              llvm::cl::Optional,
              llvm::cl::location(glow::runtime::flags::EnableDRT),
              llvm::cl::cat(flagsLibCat));

/// Allows enabling P2P support.
llvm::cl::opt<bool, /* ExternalStorage */ true>
    enableP2P("enable-P2P",
              llvm::cl::desc(
                  "Deprecated. Enabled P2P support. Alias to glow_enable_drt."),
              llvm::cl::Optional,
              llvm::cl::location(glow::runtime::flags::EnableP2P),
              llvm::cl::cat(flagsLibCat));
} // namespace
