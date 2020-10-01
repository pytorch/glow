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

#include <gflags/gflags.h>

namespace glow {
bool GlowEnableLoadBalancedPartitioning = false;
bool GlowLogPartition = false;
bool GlowDumpPartition = false;
bool GlowDumpCompilationLog = false;
bool GlowDumpBackendSpecificIRJSON = false;
bool GlowNNPILowerAllBatchMatMul = false;
bool GlowNNPIAcceptUnarySLS = false;
bool GlowNNPISpecializeAllOneSLS = false;

namespace onnxifi {
int32_t GlowNumDevices = 0;
int32_t GlowSparseNNPartitioningSchemeNumCards = 1;
int64_t GlowSparseNNPartitioningSchemeSLSTableKBytesPerCard = 0;
int32_t GlowSparseNNPartitioningSchemeNumCoresSLS = 1;
int32_t GlowSparseNNPartitioningSchemeNumCoresOther = 1;
bool GlowDumpDebugTraces = false;
int32_t GlowNumDebugTracesPerDump = 100;
bool GlowSaturateHost = false;
bool GlowFP16 = false;
bool GlowFP16Placeholders = true;
bool GlowFP16Constants = true;
bool GlowEnableQuantParamChanges = true;
bool GlowDumpGraph = false;
std::string GlowDumpGraphPath = "./";
bool GlowDumpInitialLoadedGraph = false;
bool GlowUseDAGOptimizer = false;
bool GlowUseDAGOptimizerAOT = false;
std::string GlowDAGOptimizerPlacementTaggingAlgorithm = "None";
std::string GlowDAGOptimizerParallelizationTaggingAlgorithm = "None";
int32_t GlowDAGOptimizerNumParallelChunks = 1;
bool GlowFusedScaleOffsetFP16 = false;
bool GlowForceSLSAccumFP16 = false;
bool GlowClipFP16 = false;
bool GlowClipFP16SkipInputs = true;
bool GlowUseSparseNNPartitioningScheme = false;
bool GlowSparseNNPartitioningAddSLSConcats = false;
bool GlowSparseNNPartitioningBalancePerfModel = false;
bool GlowSparseNNPartitioningPairLNWithSLS = false;
size_t GlowMaxActiveRequests = 48;
size_t GlowMaxActiveRequestsPerInstance = 48;
size_t GlowMaxQueueSize = 100;
size_t GlowExecutorThreads = 10;
bool GlowDelayAndRecordConstantModification = false;
bool GlowUseTrackedDummyQuantParams = false;
std::string GlowOnnxifiBackend = "";
bool GlowSaveOnnxifiModel = false;
bool GlowSaveOnnxifiIO = false;
bool GlowSaveOnnxifiDAG = false;
bool GlowEnablePartialTensors = true;
bool GlowUseCustomOpsForExport = true;
bool GlowDumpNNPICompilerData = false;
bool GlowUsePerPartitionIcetConfig = false;
bool GlowDisableNNPITransforms = false;
bool GlowDisableNNPIPrivateTransforms = false;
int32_t GlowNNPINumParallelChunks = 0;
int32_t GlowNNPIModelParallelSplitAlignment = 1;
std::string GlowBackendSpecificOpts = "";
} // namespace onnxifi

namespace runtime {
unsigned GlowInterpreterMemory = 0;
unsigned GlowCPUMemory = 0;
unsigned GlowHabanaMemory = 7 << 20; // 7 GB.
unsigned GlowNNPIMemory = 0;
unsigned GlowNNPITimeout = 0;
bool GlowEnableP2P = false;
bool GlowEnableDRT = false;
unsigned GlowDeviceInitTimeoutMs = 5000;
std::string GlowAvailableDevices = "";
} // namespace runtime

} // namespace glow

//===--------------------------------------------------------------------===//
//                    gflags config for all above flags
//===--------------------------------------------------------------------===//

DEFINE_int32(glow_num_devices, 1, "Number of devices for Glow backend");
DEFINE_validator(glow_num_devices, [](const char *flagname, int32_t value) {
  glow::onnxifi::GlowNumDevices = value;
  return true;
});

DEFINE_int32(
    glow_snn_partitioning_num_cards, 1,
    "Number of devices to distribute tables across in SparseNN partitioning");
DEFINE_validator(glow_snn_partitioning_num_cards,
                 [](const char * /* flagname */, int32_t value) {
                   glow::onnxifi::GlowSparseNNPartitioningSchemeNumCards =
                       value;
                   return true;
                 });

DEFINE_int32(glow_snn_partitioning_kbytes_per_card, 1,
             "Bytes per card used for SLS tables in SparseNN partitioning");
DEFINE_validator(
    glow_snn_partitioning_kbytes_per_card,
    [](const char * /* flagname */, int32_t value) {
      glow::onnxifi::GlowSparseNNPartitioningSchemeSLSTableKBytesPerCard =
          value;
      return true;
    });

DEFINE_int32(
    glow_snn_partitioning_num_cores_sls, 1,
    "Number of cores to assign to SLS partition in SparseNN partitioning");
DEFINE_validator(glow_snn_partitioning_num_cores_sls,
                 [](const char * /* flagname */, int32_t value) {
                   glow::onnxifi::GlowSparseNNPartitioningSchemeNumCoresSLS =
                       value;
                   return true;
                 });

DEFINE_int32(
    glow_snn_partitioning_num_cores_other, 1,
    "Number of cores to assign to non-SLS partition in SparseNN partitioning");
DEFINE_validator(glow_snn_partitioning_num_cores_other,
                 [](const char * /* flagname */, int32_t value) {
                   glow::onnxifi::GlowSparseNNPartitioningSchemeNumCoresOther =
                       value;
                   return true;
                 });

DEFINE_bool(glow_dump_debug_traces, false, "Dump traces to /tmp");
DEFINE_validator(glow_dump_debug_traces, [](const char *flagname, bool value) {
  glow::onnxifi::GlowDumpDebugTraces = value;
  return true;
});
DEFINE_int32(glow_num_debug_traces_per_dump, 100,
             "Maximum number of traces in each debug dump.");
DEFINE_validator(glow_num_debug_traces_per_dump,
                 [](const char * /* flagname */, int32_t value) {
                   glow::onnxifi::GlowNumDebugTracesPerDump = value;
                   return true;
                 });

DEFINE_string(glow_onnxifi_backend, "", "Glow backend used for ONNXIFI");
DEFINE_validator(glow_onnxifi_backend,
                 [](const char *flagname, const std::string &value) {
                   glow::onnxifi::GlowOnnxifiBackend = value;
                   return true;
                 });

DEFINE_string(
    glow_available_devices, "",
    "Comma separated list of devices which should be used, example 2,3,4");
DEFINE_validator(glow_available_devices,
                 [](const char * /* unused */, const std::string &value) {
                   glow::runtime::GlowAvailableDevices = value;
                   return true;
                 });

DEFINE_bool(glow_global_fp16, false,
            "Enable fp16 lowering for all ops on the net");
DEFINE_validator(glow_global_fp16, [](const char * /* unused */, bool value) {
  glow::onnxifi::GlowFP16 = value;
  return true;
});

DEFINE_bool(glow_global_fp16_placeholders, true,
            "Enable fp16 conversion for Placeholders");
DEFINE_validator(glow_global_fp16_placeholders,
                 [](const char * /* unused */, bool value) {
                   glow::onnxifi::GlowFP16Placeholders = value;
                   return true;
                 });

DEFINE_bool(glow_global_fp16_constants, true,
            "Enable fp16 conversion for Constants");
DEFINE_validator(glow_global_fp16_constants,
                 [](const char * /* unused */, bool value) {
                   glow::onnxifi::GlowFP16Constants = value;
                   return true;
                 });

DEFINE_bool(glow_global_fused_scale_offset_fp16, false,
            "Enable fp16 lowering for all op inputs using fused scale/offset");
DEFINE_validator(glow_global_fused_scale_offset_fp16,
                 [](const char * /* unused */, bool value) {
                   glow::onnxifi::GlowFusedScaleOffsetFP16 = value;
                   return true;
                 });

DEFINE_bool(
    glow_global_force_sls_fp16_accum, true,
    "Force all SLS/SLWS ops to use FP16 accumulation. True by default.");
DEFINE_validator(glow_global_force_sls_fp16_accum,
                 [](const char * /* unused */, bool value) {
                   glow::onnxifi::GlowForceSLSAccumFP16 = value;
                   return true;
                 });

DEFINE_bool(glow_enable_quant_param_changes, true,
            "Enable quantization param changes during optimizations");
DEFINE_validator(glow_enable_quant_param_changes,
                 [](const char * /* unused */, bool value) {
                   glow::onnxifi::GlowEnableQuantParamChanges = value;
                   return true;
                 });

DEFINE_bool(glow_use_sparsenn_partitioning_scheme, false,
            "Force glow to use SparseNN partitioning scheme");
DEFINE_validator(glow_use_sparsenn_partitioning_scheme,
                 [](const char * /* flagname */, bool value) {
                   glow::onnxifi::GlowUseSparseNNPartitioningScheme = value;
                   return true;
                 });

DEFINE_bool(glow_sparsenn_partitioning_add_sls_concats, false,
            "Add extra concats inside of SLS partitions for more efficient "
            "inter-partitition transfers");
DEFINE_validator(glow_sparsenn_partitioning_add_sls_concats,
                 [](const char * /* flagname */, bool value) {
                   glow::onnxifi::GlowSparseNNPartitioningAddSLSConcats = value;
                   return true;
                 });

DEFINE_bool(glow_sparsenn_partitioning_balance_perf_model, false,
            "Balance SLS tables across cards using a perf model");
DEFINE_validator(glow_sparsenn_partitioning_balance_perf_model,
                 [](const char * /* flagname */, bool value) {
                   glow::onnxifi::GlowSparseNNPartitioningBalancePerfModel =
                       value;
                   return true;
                 });

DEFINE_bool(glow_sparsenn_partitioning_pair_ln_with_sls, false,
            "Put layer normalization nodes immediately following SLS into SLS "
            "Partitions");
DEFINE_validator(glow_sparsenn_partitioning_pair_ln_with_sls,
                 [](const char * /* flagname */, bool value) {
                   glow::onnxifi::GlowSparseNNPartitioningPairLNWithSLS = value;
                   return true;
                 });

DEFINE_bool(glow_clip_fp16, false, "Force glow to clip fp16 values to min/max");
DEFINE_validator(glow_clip_fp16, [](const char *flagname, bool value) {
  glow::onnxifi::GlowClipFP16 = value;
  return true;
});

DEFINE_bool(glow_clip_fp16_skip_inputs, true,
            "Force glow to skip clipping fp16 Node inputs to min/max");
DEFINE_validator(glow_clip_fp16_skip_inputs,
                 [](const char *flagname, bool value) {
                   glow::onnxifi::GlowClipFP16SkipInputs = value;
                   return true;
                 });

DEFINE_bool(glow_saturate_host, false,
            "Try to use all available devices on the host");
DEFINE_validator(glow_saturate_host, [](const char *flagname, bool value) {
  glow::onnxifi::GlowSaturateHost = value;
  return true;
});

DEFINE_bool(
    glow_save_onnxifi_dag, false,
    "Whether to serialize the DAG that has been optimized and partitioned.");
DEFINE_validator(glow_save_onnxifi_dag, [](const char *flagname, bool value) {
  glow::onnxifi::GlowSaveOnnxifiDAG = value;
  return true;
});

DEFINE_bool(
    glow_delay_and_record_constant_modification, false,
    "Whether to delay and record constant modification for serialization.");
DEFINE_validator(glow_delay_and_record_constant_modification,
                 [](const char *flagname, bool value) {
                   glow::onnxifi::GlowDelayAndRecordConstantModification =
                       value;
                   return true;
                 });

DEFINE_bool(glow_use_tracked_dummy_quant_params, false,
            "Whether to use uniqued dummy quant params when loading the model, "
            "which are then mapped to loaded names for serialization.");
DEFINE_validator(glow_use_tracked_dummy_quant_params,
                 [](const char *flagname, bool value) {
                   glow::onnxifi::GlowUseTrackedDummyQuantParams = value;
                   return true;
                 });

DEFINE_int32(glow_max_active_requests, 48,
             "Number of max active requests before host manager start queuing");
DEFINE_validator(glow_max_active_requests,
                 [](const char *flagname, int32_t value) {
                   glow::onnxifi::GlowMaxActiveRequests = value;
                   return true;
                 });

DEFINE_int32(glow_max_active_requests_per_instance, 48,
             "Number of max active requests per instance of a network.");
DEFINE_validator(glow_max_active_requests_per_instance,
                 [](const char * /* unused */, int32_t value) {
                   glow::onnxifi::GlowMaxActiveRequestsPerInstance = value;
                   return true;
                 });

DEFINE_int32(
    glow_max_queue_size, 100,
    "Max number of pending requeusts in glow's host manager queue before "
    "rejecting new request");
DEFINE_validator(glow_max_queue_size, [](const char *flagname, int32_t value) {
  glow::onnxifi::GlowMaxQueueSize = value;
  return true;
});

DEFINE_int32(glow_executor_threads, 10,
             "Number of executor threads for host manager");
DEFINE_validator(glow_executor_threads,
                 [](const char *flagname, int32_t value) {
                   glow::onnxifi::GlowExecutorThreads = value;
                   return true;
                 });

DEFINE_bool(glow_partitioner_enable_load_balance, true,
            "Enable a partitioner pass to optimize for load balance in "
            "addition to memory capacity constraints");
DEFINE_validator(glow_partitioner_enable_load_balance,
                 [](const char *flagname, bool value) {
                   glow::GlowEnableLoadBalancedPartitioning = value;
                   return true;
                 });

DEFINE_bool(glow_save_onnxifi_model, false,
            "Package the glow function and weights right before lowering");
DEFINE_validator(glow_save_onnxifi_model,
                 [](const char * /* unused */, bool value) {
                   glow::onnxifi::GlowSaveOnnxifiModel = value;
                   return true;
                 });

DEFINE_bool(glow_save_onnxifi_io, false,
            "Save the input and output result around ONNXIFI boundary");
DEFINE_validator(glow_save_onnxifi_io,
                 [](const char * /* unused */, bool value) {
                   glow::onnxifi::GlowSaveOnnxifiIO = value;
                   return true;
                 });

DEFINE_bool(glow_enable_partial_tensors, true,
            "Save the input and output result around ONNXIFI boundary");
DEFINE_validator(glow_enable_partial_tensors,
                 [](const char * /* unused */, bool value) {
                   glow::onnxifi::GlowEnablePartialTensors = value;
                   return true;
                 });

DEFINE_bool(glow_use_custom_ops_for_export, true,
            "Use custom ONNX ops when exporting Glow protos.");
DEFINE_validator(glow_use_custom_ops_for_export,
                 [](const char * /* unused */, bool value) {
                   glow::onnxifi::GlowUseCustomOpsForExport = value;
                   return true;
                 });

DEFINE_bool(glow_dump_graph, false,
            "Dump the glow Graph into files before compilation");
DEFINE_validator(glow_dump_graph, [](const char * /* unused */, bool value) {
  glow::onnxifi::GlowDumpGraph = value;
  return true;
});

DEFINE_string(glow_dump_graph_path, "./",
              "Directory path for the dumped graphs.");
DEFINE_validator(glow_dump_graph_path,
                 [](const char * /* unused */, const std::string &value) {
                   glow::onnxifi::GlowDumpGraphPath = value;
                   return true;
                 });

DEFINE_bool(glow_dump_initial_loaded_graph, false,
            "Dump the glow Graph right after onnxification");
DEFINE_validator(glow_dump_initial_loaded_graph,
                 [](const char * /* unused */, bool value) {
                   glow::onnxifi::GlowDumpInitialLoadedGraph = value;
                   return true;
                 });

DEFINE_bool(glow_use_dag_optimizer, false, "Whether to call the DAG optimizer");
DEFINE_validator(glow_use_dag_optimizer,
                 [](const char * /* unused */, bool value) {
                   glow::onnxifi::GlowUseDAGOptimizer = value;
                   return true;
                 });

DEFINE_int32(glow_dag_optimizer_num_parallel_chunks, 1,
             "Number of parallel chunks for DAGOptimizer parallelization");
DEFINE_validator(glow_dag_optimizer_num_parallel_chunks,
                 [](const char * /* flagname */, int32_t value) {
                   glow::onnxifi::GlowDAGOptimizerNumParallelChunks = value;
                   return true;
                 });

DEFINE_string(glow_dag_optimizer_placement_tagging_algorithm, "None",
              "Name of placement tagging algorithm to run in DAGOptimizer");
DEFINE_validator(glow_dag_optimizer_placement_tagging_algorithm,
                 [](const char * /* flagname */, const std::string &value) {
                   glow::onnxifi::GlowDAGOptimizerPlacementTaggingAlgorithm =
                       value;
                   return true;
                 });

DEFINE_string(
    glow_dag_optimizer_parallelization_tagging_algorithm, "None",
    "Name of parallelization tagging algorithm to run in DAGOptimizer");
DEFINE_validator(
    glow_dag_optimizer_parallelization_tagging_algorithm,
    [](const char * /* flagname */, const std::string &value) {
      glow::onnxifi::GlowDAGOptimizerParallelizationTaggingAlgorithm = value;
      return true;
    });

#ifdef GLOW_WITH_NNPI
// Defined in glow/lib/Backends/NNPI/NNPI.cpp
DEFINE_bool(glow_use_per_partition_icet_config, false,
            "Read an icet_config.json file for each partition");
DEFINE_validator(glow_use_per_partition_icet_config,
                 [](const char * /* unused */, bool value) {
                   glow::onnxifi::GlowUsePerPartitionIcetConfig = value;
                   return true;
                 });
DEFINE_bool(glow_dump_nnpi_compiler_data, false,
            "Dump the NNPI compiler data into files before NNPI compilation");
DEFINE_validator(glow_dump_nnpi_compiler_data,
                 [](const char * /* unused */, bool value) {
                   glow::onnxifi::GlowDumpNNPICompilerData = value;
                   return true;
                 });

DEFINE_bool(glow_nnpi_specialize_all_one_sls, false,
            "Whether to import SLS ops with AllOne attribute to NNPI.");
DEFINE_validator(glow_nnpi_specialize_all_one_sls,
                 [](const char * /*unused*/, bool value) {
                   glow::GlowNNPISpecializeAllOneSLS = value;
                   return true;
                 });

DEFINE_bool(glow_disable_nnpi_transforms, false,
            "Disable running NNPIBackend::transformPostLowering().");
DEFINE_validator(glow_disable_nnpi_transforms,
                 [](const char * /* unused */, bool value) {
                   glow::onnxifi::GlowDisableNNPITransforms = value;
                   return true;
                 });
DEFINE_bool(glow_disable_nnpi_private_transforms, false,
            "Disable running NNPIBackend::transformPrivate().");
DEFINE_validator(glow_disable_nnpi_private_transforms,
                 [](const char * /* unused */, bool value) {
                   glow::onnxifi::GlowDisableNNPIPrivateTransforms = value;
                   return true;
                 });

DEFINE_bool(glow_nnpi_lower_all_batch_matmul, false,
            "Whether to override default lowering for NNPI and always lower "
            "BatchMatMul to a series of MatMuls.");
DEFINE_validator(glow_nnpi_lower_all_batch_matmul,
                 [](const char * /* unused */, bool value) {
                   glow::GlowNNPILowerAllBatchMatMul = value;
                   return true;
                 });
DEFINE_bool(glow_nnpi_accept_unary_sls, false,
            "Whether to accept unary SLS ops during ONNXIFI loading.");
DEFINE_validator(glow_nnpi_accept_unary_sls,
                 [](const char * /* unused */, bool value) {
                   glow::GlowNNPIAcceptUnarySLS = value;
                   return true;
                 });
DEFINE_int32(glow_nnpi_num_parallel_chunks, 0,
             "Number of parallel chunks for NNPI");
DEFINE_validator(glow_nnpi_num_parallel_chunks,
                 [](const char * /* flagname */, int32_t value) {
                   glow::onnxifi::GlowNNPINumParallelChunks = value;
                   return true;
                 });
DEFINE_int32(glow_nnpi_model_parallel_split_alignment, 1,
             "Alignment value for model parallel splits");
DEFINE_validator(glow_nnpi_model_parallel_split_alignment,
                 [](const char * /* flagname */, int32_t value) {
                   glow::onnxifi::GlowNNPIModelParallelSplitAlignment = value;
                   return true;
                 });
#endif /* GLOW_WITH_NNPI */

DEFINE_int32(glow_interpreter_memory, 0,
             "Amount of DRAM to allocate per Interpreter in KiB");
DEFINE_validator(glow_interpreter_memory, [](const char *, int32_t value) {
  glow::runtime::GlowInterpreterMemory = value;
  return true;
});

#ifdef GLOW_WITH_CPU
DEFINE_int32(glow_cpu_memory, 0, "Amount of DRAM to allocate per CPU in KiB");
DEFINE_validator(glow_cpu_memory, [](const char *, int32_t value) {
  glow::runtime::GlowCPUMemory = value;
  return true;
});
#endif

#ifdef GLOW_WITH_HABANA
DEFINE_int32(glow_habana_memory, 7 << 20,
             "Amount of DRAM to allocate per Habana device in KiB");
DEFINE_validator(glow_habana_memory, [](const char *flagname, int32_t value) {
  glow::runtime::GlowHabanaMemory = value;
  return true;
});
#endif

#ifdef GLOW_WITH_NNPI
DEFINE_int32(glow_nnpi_memory, 16 << 20,
             "Amount of DRAM to allocate per NNPI device in KiB");
DEFINE_validator(glow_nnpi_memory, [](const char *flagname, int32_t value) {
  glow::runtime::GlowNNPIMemory = value;
  return true;
});

DEFINE_int32(glow_nnpi_timeout_ms, 0,
             "Timeout threshold for inferecnce in milliseconds. Default 0 "
             "means infinity");
DEFINE_validator(glow_nnpi_timeout_ms,
                 [](const char * /*unused*/, int32_t value) {
                   glow::runtime::GlowNNPITimeout = value * 1000;
                   return true;
                 });

#endif

DEFINE_bool(glow_log_partition, true, "Enable logging partition info");
DEFINE_validator(glow_log_partition, [](const char * /*unused*/, bool value) {
  glow::GlowLogPartition = value;
  return true;
});

DEFINE_bool(glow_enable_p2p, false, "Enable peer to peer support");
DEFINE_validator(glow_enable_p2p, [](const char * /*unused*/, bool value) {
  glow::runtime::GlowEnableP2P = value;
  return true;
});

DEFINE_bool(glow_enable_drt, false, "Enable device resident tensor support");
DEFINE_validator(glow_enable_drt, [](const char * /*unused*/, bool value) {
  glow::runtime::GlowEnableDRT = value;
  return true;
});

DEFINE_int32(glow_device_init_timeout_ms, 5000,
             "Timeout threshold for device initialization in milliseconds. "
             "Default 5000");
DEFINE_validator(glow_device_init_timeout_ms,
                 [](const char * /*unused*/, int32_t value) {
                   glow::runtime::GlowDeviceInitTimeoutMs = value;
                   return true;
                 });

DEFINE_bool(glow_dump_partition, false,
            "Enable dumping the graph of each partition");
DEFINE_validator(glow_dump_partition, [](const char * /*unused*/, bool value) {
  glow::GlowDumpPartition = value;
  return true;
});

DEFINE_bool(glow_dump_compilation_log, false,
            "Dump the glow compilation log into /tmp during compilation");
DEFINE_validator(glow_dump_compilation_log,
                 [](const char * /*unused*/, bool value) {
                   glow::GlowDumpCompilationLog = value;
                   return true;
                 });

DEFINE_bool(glow_dump_backend_specific_ir_json, false,
            "Dump the backend-specific IR JSON file");
DEFINE_validator(glow_dump_backend_specific_ir_json,
                 [](const char * /*unused*/, bool value) {
                   glow::GlowDumpBackendSpecificIRJSON = value;
                   return true;
                 });

DEFINE_string(glow_backend_specific_opts, "",
              "Glow backend specific options. Comma separated list of "
              "key=value pairs, e.g. key1=val1,key2=val2.");
DEFINE_validator(glow_backend_specific_opts,
                 [](const char *flagname, const std::string &value) {
                   glow::onnxifi::GlowBackendSpecificOpts = value;
                   return true;
                 });
