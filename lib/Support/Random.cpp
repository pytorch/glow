/**
 * Copyright (c) 2017-present, Facebook, Inc.
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

static llvm::cl::opt<bool> useRandomDevice(
    "use-random-device",
    llvm::cl::desc(
        "Use a non-deterministic random number generator to seed the PRNG"),
    llvm::cl::Optional, llvm::cl::init(false));

static glow::PseudoRNG::result_type getRandomSeed() {
  if (useRandomDevice) {
    return std::random_device()();
  }
  return pseudoRandomSeed.getValue();
}

namespace glow {

PseudoRNG::PseudoRNG() : engine_(getRandomSeed()) {}

// Constant uniform distribution used as a template in nextRand.
// This computes the parameters once instead of on each call.
const static std::uniform_real_distribution<> uniformReal(-1, 1);

double PseudoRNG::nextRand() {
  return std::uniform_real_distribution<double>(uniformReal.param())(engine_);
}

} // namespace glow
