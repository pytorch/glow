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

#include "Int8GemmBench.h"

#include <array>
#include <cstdlib>
#include <fstream>
#include <future>
#include <random>
#include <string>

#include "Bench.h"

#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Optimizer/GraphOptimizer/GraphOptimizer.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Signals.h"

#include "tests/unittests/BackendTestUtils.h"

using namespace glow;

#define DEVICE_ID 9

/*
 * This class implements a Int8 Quantized GEMM/FC microbenchmark. There are a
 * set of (m x k) * (k x n) = (m x n) matrix multiplications, chained together
 * in multiple layers.
 *
 * Microbenchmarks are generally useful for understanding performance
 * through targeted experimentation and are not representative of
 * end-to-end workloads.
 */
// TODO: Move all the args passed by command line to LLVM options.
llvm::cl::OptionCategory int8GemmBenchCat("Int8GemmBench Category");
llvm::cl::opt<bool> checkCorrectness(
    "check-results",
    llvm::cl::desc("Check the correctness of the results against the reference "
                   "backend (Interpreter)"),
    llvm::cl::Optional, llvm::cl::init(false), llvm::cl::cat(int8GemmBenchCat));

Int8GemmParam parseArgs(int argc, char *argv[]) {
  Int8GemmParam param;

  param.m_ = atoi(argv[1]);
  param.n_ = atoi(argv[2]);
  param.k_ = atoi(argv[3]);
  param.numLayers_ = atoi(argv[4]);
  param.numReps_ = atoi(argv[5]);
  param.numAsyncLaunches_ = atoi(argv[6]);
  param.numSplits_ = atoi(argv[7]);
  param.backendStr_ = std::string(argv[8]);

  printf("m %zu\n", (size_t)param.m_);
  printf("n %zu\n", (size_t)param.n_);
  printf("k %zu\n", (size_t)param.k_);
  printf("numLayers %zu\n", (size_t)param.numLayers_);
  printf("numReps %zu\n", (size_t)param.numReps_);
  printf("numAsyncLaunches %zu\n", (size_t)param.numAsyncLaunches_);
  printf("numSplits %zu\n", (size_t)param.numSplits_);
  printf("backendStr %s\n", param.backendStr_.c_str());

  if (argc > DEVICE_ID) {
    printf("devId %s\n", argv[DEVICE_ID]);
    param.devId_ = std::string(argv[DEVICE_ID]);
  } else {
    param.devId_ = std::string("");
  }
  printf("\n\n");
  return param;
}

int main(int argc, char *argv[]) {
  printf("GEMM Microbenchmark\n");
  printf("Usage: GemmBench m(Int) n(Int) k(Int) numLayers(Int) numReps(Int) "
         "numAsyncLaunches(Int) numSplits(Int) backendStr(String) "
         "dev_id(Int)\n");
  printf("Standard Glow command-line options may be passed via the GLOW_OPTS "
         "environment variable\n");
  benchParseGlowOpts(argc, argv);

  std::vector<Int8GemmParam> params;
  std::string runHeader;
  std::string runPrefix;

  // Using a config file
  if (argc == 2) {
    auto fname = std::string(argv[1]);
    std::ifstream fin(fname.c_str());
    if (!fin) {
      std::cout << "Could not open file: " << fname << std::endl;
      exit(0);
    }
    std::string line;
    while (getline(fin, line)) {
      std::array<char, 1024> buf;
      char *saveptr = nullptr;
      std::vector<char *> argVec;
      strcpy(buf.data(), line.c_str());
      char *ptr = strtok_r(buf.data(), " ", &saveptr);
      while (ptr != nullptr) {
        argVec.push_back(ptr);
        ptr = strtok_r(nullptr, " ", &saveptr);
      }
      Int8GemmParam param = parseArgs(argVec.size(), argVec.data());
      params.push_back(param);
      runHeader = std::string("_,benchName,_,filename");
      runPrefix = std::string(strFormat("GemmBench,SW,%s", fname.c_str()));
    }
  } else if (argc == 9 || argc == 10) {
    Int8GemmParam param = parseArgs(argc, argv);
    params.push_back(param);
    runHeader = std::string(
        "_,benchName,_,m,n,k,numLayers,numReps,numAsyncLaunches,numSplits,"
        "backendStr\n");
    runPrefix = std::string(strFormat(
        "GemmBench,SW,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%s", (size_t)param.m_,
        (size_t)param.n_, (size_t)param.k_, (size_t)param.numLayers_,
        (size_t)param.numReps_, (size_t)param.numAsyncLaunches_,
        (size_t)param.numSplits_, argv[8]));
  } else {
    llvm_unreachable("Invalid command line");
  }

  for (auto param : params) {
    Int8GemmBench b(param);
    auto times = bench(&b, param.numReps_);

    printf("%s,runtime,gflopPerSec\n", runHeader.c_str());
    for (auto t : times) {
      printf("BenchResult,%s,%f,%f\n", runPrefix.c_str(),
             t / param.numAsyncLaunches_,
             b.gops() * param.numAsyncLaunches_ / t);
    }
    double min = *(std::min_element(times.begin(), times.end()));
    dim_t midElt = times.size() / 2;
    std::nth_element(times.begin(), times.begin() + midElt, times.end());
    double median = times[midElt];
    double medianRuntime = median / ((double)param.numAsyncLaunches_);
    double minRuntime = min / ((double)param.numAsyncLaunches_);
    printf("%s,medianRuntime,minRuntime,medianGflopPerSec,maxGflopPerSec\n",
           runHeader.c_str());
    printf("BenchSummary,%s,%f,%f,%f,%f\n", runPrefix.c_str(), medianRuntime,
           minRuntime, b.gops() / medianRuntime, b.gops() / minRuntime);
  }
}
