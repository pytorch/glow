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

#ifndef GLOW_TORCH_GLOW_SRC_FUSE_PREPACK_H
#define GLOW_TORCH_GLOW_SRC_FUSE_PREPACK_H

#include <torch/csrc/jit/ir.h>

namespace glow {
/// Fuse weight packing operation into quantized convolution op thus skipping
/// weight packing.
void fuseConvPrepack(std::shared_ptr<torch::jit::Graph> &graph);

/// Fuse weight packing operation into quantized linear op thus skipping
/// weight packing.
void fuseLinearPrepack(std::shared_ptr<torch::jit::Graph> &graph);
} // namespace glow

#endif // GLOW_TORCH_GLOW_SRC_FUSE_PREPACK_H
