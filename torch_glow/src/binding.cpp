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
#include "GlowFuser.h"
#include "PyTorchCommon.h"
#include "Registration.h"
#include "TorchGlowTraining.h"
#include <pybind11/pybind11.h>
/// Required include files for a proper binding TorchGlowTrainingWrapper class.
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

  /// Enable compiling PyTorch subgraphs to Glow Functions.
  m.def("enableFusionPass",
        []() { getPyTorchLoaderSettings().fusionPassEnabled = true; });

  /// Disable compiling PyTorch subgraphs to Glow Functions.
  m.def("disableFusionPass",
        []() { getPyTorchLoaderSettings().fusionPassEnabled = false; });

  /// Enable freezing weights as Constants in PyTorch subgraphs loaded in Glow.
  m.def("enableWeightFreezing",
        []() { getPyTorchLoaderSettings().weightFreezingEnabled = true; });

  /// Disable freezing weights as Constants in PyTorch subgraphs loaded in Glow.
  m.def("disableWeightFreezing",
        []() { getPyTorchLoaderSettings().weightFreezingEnabled = false; });

  /// Enable dumping Glow DAG to file after model loading finishes.
  m.def("enableDumpGlowDag",
        []() { getPyTorchLoaderSettings().dumpGlowDag = true; });

  /// Disable dumping Glow DAG to file after model loading finishes.
  m.def("disableDumpGlowDag",
        []() { getPyTorchLoaderSettings().dumpGlowDag = false; });

  /// Enable converting fp32 ops to fp16.
  m.def("enable_convert_to_fp16",
        []() { getPyTorchLoaderSettings().convertToFP16 = true; });

  /// Disable converting fp32 ops to fp16.
  m.def("disable_convert_to_fp16",
        []() { getPyTorchLoaderSettings().convertToFP16 = false; });

  /// Enable dumping the final Glow dag after compilation.
  m.def("enable_dump_final_glow_graph",
        []() { getPyTorchLoaderSettings().dumpFinalGlowGraph = true; });

  /// Disable dumping the final Glow dag after compilation.
  m.def("disable_dump_final_glow_graph",
        []() { getPyTorchLoaderSettings().dumpFinalGlowGraph = false; });

  /// Enable tracing in Glow runtime.
  m.def("enable_glow_tracing",
        []() { getPyTorchLoaderSettings().enableGlowTracing = true; });

  /// Set the number of traces to dump per trace file.
  m.def("set_num_traces_per_dump", [](size_t numTracesPerDump) {
    getPyTorchLoaderSettings().numTracesPerDump = numTracesPerDump;
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

  /// Enable saturateHost mode in Glow runtime.
  m.def("enable_saturate_host",
        []() { getPyTorchLoaderSettings().saturateHost = true; });

  /// Disable saturateHost mode in Glow runtime.
  m.def("disable_saturate_host",
        []() { getPyTorchLoaderSettings().saturateHost = false; });

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
  m.def("setGlowBackend", [](const std::string &glowBackendName) {
    setHostManager(glowBackendName);
  });

  /// Set the active HostManager to one that owns \p numDevices of type
  /// \p backendName.
  m.def("setGlowBackend",
        [](const std::string &glowBackendName, size_t numDevices) {
          setHostManager(glowBackendName, numDevices);
        });

  /// \returns the name of the device backend used by the active HostManager.
  m.def("getGlowBackendName", []() { return getBackendName(); });

  /// \returns the quantity of the device backends used by the active
  /// HostManager.
  m.def("getGlowBackendNumDevices", []() { return getBackendNumDevices(); });

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

  /// Binding wrapper class for TorchGlowTraining and its settings.
  py::class_<TorchGlowTrainingWrapper>(m, "TorchGlowTrainingWrapper")
      .def(py::init())
      .def("init", &TorchGlowTrainingWrapper::init)
      .def("train", &TorchGlowTrainingWrapper::train)
      .def("save", &TorchGlowTrainingWrapper::save)
      .def("parameters", &TorchGlowTrainingWrapper::setONNXWriterParameters)
      .def("config", &TorchGlowTrainingWrapper::setTrainingConfig);
}
