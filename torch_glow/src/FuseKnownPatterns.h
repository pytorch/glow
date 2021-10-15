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

#ifndef GLOW_TORCH_GLOW_SRC_FUSE_KNOWN_PATERNS_H
#define GLOW_TORCH_GLOW_SRC_FUSE_KNOWN_PATERNS_H

#include <torch/csrc/jit/ir/ir.h>

namespace glow {
/// Fuse known node patterns in \p graph to assist the PyTorchModelLoader.
/// Ignore any patterns that contain ops in \p opBlockList.
void fuseKnownPatterns(
    std::shared_ptr<torch::jit::Graph> &graph,
    const std::unordered_set<torch::jit::Symbol> &opBlocklist);

void unfuseDummyOperators(std::shared_ptr<torch::jit::Graph> &graph);

/// Passes in detail namespace should not be used directly except for by
/// unittests.
namespace detail {
/// Pass that removes all prim::RaiseException nodes from the \p graph.
void removeExceptions(std::shared_ptr<torch::jit::Graph> &graph);

/// Pass that fuses the output pattern of Linear module (which contains a branch
/// based on the dims of the input) in \p graph to a glow::fused_linear op so
/// that it can be loaded by Glow with the control flow happening at graph
/// compile time.
void fuseBranchedLinearPattern(std::shared_ptr<torch::jit::Graph> &graph);

/// Pass that fuses prim::ListConstruct -> aten::cat patterns in \p graph into
/// prim::FusedConcat node so that the number of tensors being concatenated is
/// known at graph compile time.
void fuseConcat(std::shared_ptr<torch::jit::Graph> &graph);

/// Pass that fuses quantized::conv2d_prepack -> quantized::conv2d patterns in
/// \p graph into glow::unpacked_quantized_conv2d.
void fuseConvPrepack(std::shared_ptr<torch::jit::Graph> &graph);

/// Pass that fuses quantized::linear_prepack -> quantized::linear patterns in
/// \p graph into glow::unpacked_quantized_linear.
void fuseLinearPrepack(std::shared_ptr<torch::jit::Graph> &graph);

/// Pass that replaces quantized::linear with quantized::linear_unpack +
/// glow::unpacked_quantized_linear in \p graph.
void rewriteQuantizedLinear(std::shared_ptr<torch::jit::Graph> &graph);

/// Pass that eliminates prim::NumToTensor -> aten::Int patterns in
/// \p graph.
void fuseNumToTensorToNum(std::shared_ptr<torch::jit::Graph> &graph);
} // namespace detail

} // namespace glow

#endif // GLOW_TORCH_GLOW_SRC_FUSE_KNOWN_PATERNS_H
