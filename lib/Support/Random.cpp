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

#include "glow/Support/Random.h"
#include "llvm/Support/CommandLine.h"

#include <cassert>

static llvm::cl::opt<glow::PseudoRNG::result_type>
    pseudoRandomSeed("pseudo-random-seed",
                     llvm::cl::desc("Seed for pseudo-random numbers"),
                     llvm::cl::init(glow::PseudoRNG::Engine::default_seed));

namespace glow {

PseudoRNG::PseudoRNG() : engine_(pseudoRandomSeed.getValue()) {}

// Constant uniform distribution used as a template in nextRand.
// This computes the parameters once instead of on each call.
const static std::uniform_real_distribution<> uniformReal(-1, 1);

double PseudoRNG::nextRand() {
  return std::uniform_real_distribution<double>(uniformReal.param())(engine_);
}

double PseudoRNG::nextRandReal(double a, double b) {
  return std::uniform_real_distribution<double>(a, b)(engine_);
}

} // namespace glow
