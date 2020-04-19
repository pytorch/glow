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
extern bool GlowSaturateHost;
extern bool GlowSaveOnnxifiModel;
extern bool GlowSaveOnnxifiIO;
extern bool GlowEnablePartialTensors;
extern bool GlowUseCustomOpsForExport;
extern bool GlowUseSparseNNPartitioningScheme;
extern bool GlowSparseNNPartitioningAddSLSConcats;
extern bool GlowDumpGraph;
extern size_t GlowMaxActiveRequests;
extern size_t GlowMaxQueueSize;
extern size_t GlowExecutorThreads;

#ifdef GLOW_WITH_NNPI
// Defined in glow/lib/Backends/NNPI/NNPI.cpp
extern bool GlowDumpNNPICompilerData;
extern bool GlowUsePerPartitionIcetConfig;
extern bool GlowDisableNNPITransforms;
extern bool GlowDisableNNPIPrivateTransforms;
extern int32_t GlowNNPINumParallelChunks;
#endif

} // namespace onnxifi

extern bool GlowEnableLoadBalancedPartitioning;
extern bool GlowNNPILowerAllBatchMatMul;
extern bool GlowNNPIAcceptUnarySLS;

namespace runtime {
extern unsigned GlowInterpreterMemory;
extern unsigned GlowCPUMemory;
extern unsigned GlowHabanaMemory;
extern unsigned GlowNNPIMemory;
} // namespace runtime

extern bool GlowDumpCompilationLog;
extern bool GlowLogPartition;
} // namespace glow

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

DEFINE_int32(glow_max_active_requests, 48,
             "Number of max active requests before host manager start queuing");
DEFINE_validator(glow_max_active_requests,
                 [](const char *flagname, int32_t value) {
                   glow::onnxifi::GlowMaxActiveRequests = value;
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

DEFINE_bool(glow_partitioner_enable_load_balance, false,
            "Enable a partitioner "
            "pass to optimize for load balance in addition to memory capacity "
            "constraints");
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
DEFINE_int32(glow_nnpi_num_parallel_chunks, 1,
             "Number of parallel chunks for NNPI");
DEFINE_validator(glow_nnpi_num_parallel_chunks,
                 [](const char * /* flagname */, int32_t value) {
                   glow::onnxifi::GlowNNPINumParallelChunks = value;
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
#endif

DEFINE_bool(glow_log_partition, false, "Enable logging partition info");
DEFINE_validator(glow_log_partition, [](const char * /*unused*/, bool value) {
  glow::GlowLogPartition = value;
  return true;
});

DEFINE_bool(glow_dump_compilation_log, false,
            "Dump the glow compilation log into /tmp during compilation");
DEFINE_validator(glow_dump_compilation_log,
                 [](const char * /*unused*/, bool value) {
                   glow::GlowDumpCompilationLog = value;
                   return true;
                 });
