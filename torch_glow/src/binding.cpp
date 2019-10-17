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

#include "PyTorchCommon.h"
#include "PyTorchModelLoader.h"
#include "TorchGlowTraining.h"
#include <pybind11/pybind11.h>
/// Required include files for a proper binding TorchGlowTrainingWrapper class.
#include <pybind11/stl.h>
#include <torch/csrc/utils/pybind.h>

#include "glow/Graph/Graph.h"

namespace py = pybind11;

using namespace glow;

/// The torch_glow pybind11 module.
PYBIND11_MODULE(_torch_glow, m) {
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

  /// Binding wrapper class for TorchGlowTraining and its settings.
  py::class_<TorchGlowTrainingWrapper>(m, "TorchGlowTrainingWrapper")
      .def(py::init())
      .def("init", &TorchGlowTrainingWrapper::init)
      .def("train", &TorchGlowTrainingWrapper::train)
      .def("save", &TorchGlowTrainingWrapper::save)
      .def("parameters", &TorchGlowTrainingWrapper::setONNXWriterParameters)
      .def("config", &TorchGlowTrainingWrapper::setTrainingConfig);
}
