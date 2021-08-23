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

#include "glow/fb/fx/fx_glow/fx_glow.h"
#include "glow/glow/tests/unittests/ReproFXLib.h"
#include "glow/glow/torch_glow/src/GlowCompileSpec.h"
#include <folly/dynamic.h>
#include <folly/json.h>
#include <fstream>
#include <istream>
#include <sstream>
#include <string>
#include <torch/script.h>
#include <torch/torch.h>
#include <vector>

int main(int argc, char **argv) {
  ReproFXLib repro;
  repro.parseCommandLine(argc, argv);
  std::vector<torch::Tensor> output = repro.run();
}
