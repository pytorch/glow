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

#include <pybind11/pybind11.h>

#include "FuseKnownPatterns.h"
#include "GlowCompileSpec.h"
#include "GlowFuser.h"
#include "PyTorchCommon.h"
#include "Registration.h"
#include "TorchGlowBackend.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "glow/Graph/Graph.h"

namespace py = pybind11;

using namespace glow;

/// The torch_glow pybind11 module.
#ifdef TORCH_GLOW_MODULE_NAME
PYBIND11_MODULE(TORCH_GLOW_MODULE_NAME, m) {
#else
PYBIND11_MODULE(_torch_glow, m) {
#endif
  /// Register Glow op and FusionPass, enable the fusion pass if
  /// fusionPassEnabled is set in PyTorchLoaderSettings.
  registerGlowFusionOpAndPass(
      []() { return getPyTorchLoaderSettings().fusionPassEnabled; });

  /// 1) Registers TorchGlowBackend as a PyTorch backend.
  /// 2) Registers custom classes used by TorchGlowBackend.
  /// 3) Registers JIT IR ops that are used in preprocessing.
  registerTorchGlowBackendAndDeps();

  /// Enable overriding signal handlers for torch_glow to make interruping long
  /// running processes possible. This should only be used when running
  /// torch_glow with Python.
  enableSignalHandlerOverrides();

  /// Enable compiling PyTorch subgraphs to Glow Functions.
  m.def("enableFusionPass",
        []() { getPyTorchLoaderSettings().fusionPassEnabled = true; });

  /// Disable compiling PyTorch subgraphs to Glow Functions.
  m.def("disableFusionPass",
        []() { getPyTorchLoaderSettings().fusionPassEnabled = false; });

  /// Enable dumping Glow DAG to file after model loading finishes.
  m.def("enableDumpGlowDag",
        []() { getPyTorchLoaderSettings().dumpGlowDag = true; });

  /// Disable dumping Glow DAG to file after model loading finishes.
  m.def("disableDumpGlowDag",
        []() { getPyTorchLoaderSettings().dumpGlowDag = false; });

  /// Enable printing index of all jit node for debugging
  m.def("enable_printing_jit_node_indices",
        []() { getPyTorchLoaderSettings().printJITIndex = true; });

  /// Disable printing index of all jit node for debugging
  m.def("disable_printing_jit_node_indices",
        []() { getPyTorchLoaderSettings().printJITIndex = false; });

  /// Enable converting fp32 ops to fp16.
  m.def("enable_convert_to_fp16",
        []() { getPyTorchLoaderSettings().convertToFP16 = true; });

  /// Disable converting fp32 ops to fp16.
  m.def("disable_convert_to_fp16",
        []() { getPyTorchLoaderSettings().convertToFP16 = false; });

  /// Enable converting fp32 ops to fp16.
  m.def("enable_clip_fp16",
        []() { getPyTorchLoaderSettings().clipFP16 = true; });

  /// Disable converting fp32 ops to fp16.
  m.def("disable_clip_fp16",
        []() { getPyTorchLoaderSettings().clipFP16 = false; });

  /// Enable converting fp32 fused ops to fp16.
  m.def("enable_convert_fused_to_fp16",
        []() { getPyTorchLoaderSettings().convertFusedToFP16 = true; });

  /// Disable converting fp32 fused ops to fp16.
  m.def("disable_convert_fused_to_fp16",
        []() { getPyTorchLoaderSettings().convertFusedToFP16 = false; });

  /// Enable dumping the final Glow dag after compilation.
  m.def("enable_dump_final_glow_graph",
        []() { getPyTorchLoaderSettings().dumpFinalGlowGraph = true; });

  /// Disable dumping the final Glow dag after compilation.
  m.def("disable_dump_final_glow_graph",
        []() { getPyTorchLoaderSettings().dumpFinalGlowGraph = false; });

  /// Enable tracing in Glow runtime.
  m.def("enable_glow_tracing",
        []() { getPyTorchLoaderSettings().enableGlowTracing = true; });

  // Enable the auto removal of mutation in JIT graph, i.e, inline ops.
  m.def("enable_remove_mutation",
        []() { getPyTorchLoaderSettings().enableRemoveMutation = true; });

  // Disable the auto removal of mutation in JIT graph
  m.def("disable_remove_mutation",
        []() { getPyTorchLoaderSettings().enableRemoveMutation = false; });

  /// Set the number of traces to dump per trace file.
  m.def("set_num_traces_per_dump", [](size_t numTracesPerDump) {
    getPyTorchLoaderSettings().numTracesPerDump = numTracesPerDump;
  });

  /// Set the number of replications on each device.
  m.def("set_replication_count", [](size_t replicationCount) {
    getPyTorchLoaderSettings().replicationCount = replicationCount;
  });

  /// Disable tracing in Glow runtime.
  m.def("disable_glow_tracing",
        []() { getPyTorchLoaderSettings().enableGlowTracing = false; });

  /// Enable write Glow graph to onnx after model loading finishes.
  m.def("enable_write_to_onnx",
        []() { getPyTorchLoaderSettings().writeToOnnx = true; });

  /// Disable write Glow graph to onnx after model loading finishes.
  m.def("disable_write_to_onnx",
        []() { getPyTorchLoaderSettings().writeToOnnx = false; });

  /// Enable zip mode when writing ONNX model to file
  m.def("enable_onnx_zip_mode",
        []() { getPyTorchLoaderSettings().onnxZipMode = true; });

  /// Disable zip mode when writing ONNX model to file
  m.def("disable_onnx_zip_mode",
        []() { getPyTorchLoaderSettings().onnxZipMode = false; });

  /// Enable randomizing Constants in loaded Functions.
  m.def("enable_randomize_constants",
        []() { getPyTorchLoaderSettings().randomizeConstants = true; });

  /// Disable randomizing Constants in loaded Functions.
  m.def("disable_randomize_constants",
        []() { getPyTorchLoaderSettings().randomizeConstants = false; });

  /// Enable check Glow vs jit correctness.
  m.def("enable_jit_vs_glow_compare",
        []() { getPyTorchLoaderSettings().jitVsGlowCompare = true; });

  /// Disable check Glow vs jit correctness.
  m.def("disable_jit_vs_glow_compare",
        []() { getPyTorchLoaderSettings().jitVsGlowCompare = false; });

  /// Enable saturateHost mode in Glow runtime.
  m.def("enable_saturate_host",
        []() { getPyTorchLoaderSettings().saturateHost = true; });

  /// Disable saturateHost mode in Glow runtime.
  m.def("disable_saturate_host",
        []() { getPyTorchLoaderSettings().saturateHost = false; });

  /// Enable shape inference engine.
  m.def("enable_shape_inference_engine",
        []() { getPyTorchLoaderSettings().runShapeInference = true; });

  /// Disable shape inference engine.
  m.def("disable_shape_inference_engine",
        []() { getPyTorchLoaderSettings().runShapeInference = false; });

  /// Add all of the symbols in \p blacklist to the fusion blacklist so that
  /// nodes with these symbols will not be fused to Glow.
  m.def("setFusionBlacklist", [](const std::vector<std::string> &blacklist) {
    auto &bl = getPyTorchLoaderSettings().opBlacklist;
    bl.clear();
    for (const auto &kind : blacklist) {
      bl.insert(torch::jit::Symbol::fromQualString(kind));
    }
  });

  /// Clear the fusion blacklist.
  m.def("clearFusionBlacklist",
        []() { getPyTorchLoaderSettings().opBlacklist.clear(); });

  /// Set the index (inclusive) of the first node in the graph to fuse.
  m.def("setFusionStartIndex", [](int64_t startIndex) {
    getPyTorchLoaderSettings().fusionStartIndex = startIndex;
  });

  /// Set the index (exclusive) of the last node in the graph to fuse.
  m.def("setFusionEndIndex", [](int64_t endIndex) {
    getPyTorchLoaderSettings().fusionEndIndex = endIndex;
  });

  /// Clear the start and end fusion indices.
  m.def("clearFusionIndices", []() {
    getPyTorchLoaderSettings().fusionStartIndex = -1;
    getPyTorchLoaderSettings().fusionEndIndex = -1;
  });

  /// Set the active HostManager to one that owns 1 of type \p backendName.
  m.def("setGlowBackend", [](const std::string &backendName) {
    getPyTorchLoaderSettings().backendName = backendName;
  });

  /// \returns the name of the device backend used by the active HostManager.
  m.def("getGlowBackendName",
        []() { return getPyTorchLoaderSettings().backendName; });

  /// Set the quantity of the device backends used by the active
  /// HostManager.
  m.def("setGlowBackendNumDevices", [](int32_t numDevices) {
    return getPyTorchLoaderSettings().numDevices = numDevices;
  });

  /// \returns the quantity of the device backends used by the active
  /// HostManager.
  m.def("getGlowBackendNumDevices",
        []() { return getPyTorchLoaderSettings().numDevices; });

  /// Inform host manager to load backend specific options from YAML file.
  m.def("loadBackendSpecificOptions", [](const std::string &yamlFile) {
    getPyTorchLoaderSettings().backendOptionsFile = yamlFile;
  });

  /// Calls all of the fusion passes that get run before the PyTorchModelLoader
  /// run.
  /// NOTE: This is only exposed for testing.
  m.def("fuseKnownPatterns_", fuseKnownPatterns);

  /// Calls the removeException pass.
  /// NOTE: This is only exposed for testing.
  m.def("removeExceptions_", glow::detail::removeExceptions);

  /// Calls the fuseBranchedLinearPattern pass.
  /// NOTE: This is only exposed for testing.
  m.def("fuseBranchedLinearPattern_", glow::detail::fuseBranchedLinearPattern);

  /// Set the minimum fusion group size.
  m.def("setMinFusionGroupSize",
        [](size_t k) { getPyTorchLoaderSettings().minFusionGroupSize = k; });

  /// Set the maximum fusion merge size
  m.def("setMaxFusionMergeSize",
        [](size_t k) { getPyTorchLoaderSettings().maxFusionMergeSize = k; });

  /// Call the Glow fuser and accept all node kinds but don't actually run the
  /// PyTorchModelLoader.
  /// NOTE: This is only exposed for testing.
  m.def("glowCustomFuseDebug_", [](std::shared_ptr<torch::jit::Graph> graph) {
    return glowCustomFuse(graph);
  });

  /// NOTE: This is only exposed for testing.
  m.def("glowCustomFuseDebug_", [](std::shared_ptr<torch::jit::Graph> graph,
                                   std::vector<std::string> &acceptableKinds) {
    return glowCustomFuseDebug(graph, acceptableKinds);
  });

  /// Enable running fusion pass in to_glow as a debug flow
  m.def("enable_debug_fuser",
        []() { getPyTorchLoaderSettings().enableDebugFuser = true; });

  /// Disable running fusion pass in to_glow as a debug flow
  m.def("disable_debug_fuser",
        []() { getPyTorchLoaderSettings().enableDebugFuser = false; });
}
